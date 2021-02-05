#include "Contact.h"

using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

Contact::
Contact(const dart::simulation::WorldPtr& wPtr)
{
    mWorld = wPtr;
}

Contact::
~Contact()
{

}

void
Contact::
Initialize(const dart::dynamics::SkeletonPtr& skel, double m, int steps)
{
    mSkeleton = skel;
    mWindowSize = 1200;
    mMass = m;
    mNumSteps = steps;
}

void
Contact::
Clear()
{
    for(auto iter = mContactObjects.begin(); iter != mContactObjects.end(); iter++ )
        (iter->second).clear();
}

void
Contact::
Set()
{
    this->Clear();

    const dart::collision::CollisionResult& result = mWorld->getLastCollisionResult();

    for(auto iter = mContactObjects.begin(); iter != mContactObjects.end(); iter++)
    {
        std::string name = iter->first;
        const dart::dynamics::BodyNode* body = mSkeleton->getBodyNode(name);
        for(const auto& shapeNode : body->getShapeNodesWith<dart::dynamics::CollisionAspect>())
        {
            Eigen::Vector3d force_vector = Eigen::Vector3d::Zero();
            for(const auto& contact : result.getContacts())
            {
                if(shapeNode == contact.collisionObject1->getShapeFrame() ||
                    shapeNode == contact.collisionObject2->getShapeFrame())
                {
                    (iter->second).push_back(std::make_pair(contact.force, contact.point));
                    Eigen::Vector3d cur_force = contact.force;
                    force_vector += cur_force;
                }
            }
            this->SetContactForce(name, force_vector.norm());
            mContactForceDequeMap[name].pop_back();
            mContactForceDequeMap[name].push_front(this->GetContactForce(name));
        }
    }
}

void
Contact::
SetContactObject(std::string name)
{
    if(mContactObjects.count(name) > 0)
    {
        std::cout << "Already exist contact!!" << std::endl;
        return ;
    }

    mContactObjects[name] = std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>();
    mContactForceDequeMap[name] = std::deque<double>(mWindowSize);
}

void
Contact::
SetContactForces(std::string name, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> forces)
{
    mContactObjects[name] = forces;
}

void
Contact::
AddContactForce(std::string name, Eigen::Vector3d f, Eigen::Vector3d p)
{
    mContactObjects[name].push_back(std::make_pair(f,p));
}

const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>&
Contact::
GetContactForces(std::string name)
{
    return mContactObjects[name];
}

void
Contact::
SetContactForce(std::string name, double f)
{
    mContactForces[name] = f;
}

double
Contact::
GetContactForce(std::string name)
{
    return mContactForces[name];
}

double
Contact::
GetReward()
{
    double err_scale = 1.0;
    double contact_scale = 1.0;
    double contact_err = 0;

    std::vector<double> contact_errs;
    for(auto iter = mContactForceDequeMap.begin(); iter != mContactForceDequeMap.end(); iter++)
    {
        contact_err = 0;
        for(int i=0; i<mNumSteps; i++)
            contact_err += 0.001*(iter->second).at(i);
        contact_err /= (mNumSteps * mMass);
        contact_errs.push_back(contact_err);
    }

    for(auto err : contact_errs)
        contact_err += err;

    // if(contact_err < 5.0)
    //  contact_err = 0.0;
    // else
    //  contact_err -= 5.0;
    double reward = exp(-err_scale * contact_scale * contact_err);
    return reward;

}

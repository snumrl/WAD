#include "Contact.h"

namespace WAD
{

Contact::
Contact(const WorldPtr& wPtr)
{
	mWorld = wPtr;
}

Contact::
~Contact()
{

}

void
Contact::
Initialize(std::string name, BodyNode* bn)
{
	mName = name;
	mBodyNode = bn;
}

void
Contact::
Reset()
{

}

void
Contact::
AddGround(BodyNode* grnd)
{
	mGround = grnd;
}

void
Contact::
Set()
{
	mContact = false;
	const dart::collision::CollisionResult& result = mWorld->getLastCollisionResult();
	for(const auto& sf: mBodyNode->getShapeNodesWith<CollisionAspect>())
	{
		for(const auto& sfGround : mGround->getShapeNodesWith<CollisionAspect>())
		{
			for(const auto& contact : result.getContacts())
			{
				const dart::dynamics::ShapeFrame* sf1 = contact.collisionObject1->getShapeFrame();
				const dart::dynamics::ShapeFrame* sf2 = contact.collisionObject2->getShapeFrame();	
				if((sf == sf1 && sfGround == sf2) || (sfGround == sf1 && sf == sf2))
				{
					mContact = true;
							
					// (iter->second).push_back(std::make_pair(contact.force, contact.point));
						// std::cout << name << " : " << body->getCOM()[0] << " " << body->getCOM()[1] << " " << body->getCOM()[2] << std::endl;
					// Eigen::Vector3d cur_force = contact.force;
					// force_vector += cur_force;
				}
			}
		}		
		// this->SetContactForce(name, force_vector.norm());
		// mContactForceDequeMap[name].pop_back();
		// mContactForceDequeMap[name].push_front(this->GetContactForce(name));
	}
}

bool
Contact::
isContact()
{
	return mContact;
}

// void
// Contact::
// Initialize(const SkeletonPtr& skel, double m, int steps)
// {
// 	mSkeleton = skel;
// 	mWindowSize = 1200;
// 	mMass = m;
// 	mNumSteps = steps;
// }

// void
// Contact::
// SetContactObject(std::string name)
// {
// 	if(mObjects.count(name) > 0)
// 	{
// 		std::cout << "Already Added Contact Object!" << std::endl;
// 		return ;
// 	}

// 	mObjects[name] = 

// 	mContactObjects[name] = std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>();
// 	mContactForceDequeMap[name] = std::deque<double>(mWindowSize);
// }


// // void
// // Contact::
// // Clear()
// // {
// // 	for(auto iter = mContactObjects.begin(); iter != mContactObjects.end(); iter++ )
// // 		(iter->second).clear();
// // }

// void
// Contact::
// Set()
// {
// 	this->Clear();
// 	const dart::collision::CollisionResult& result = mWorld->getLastCollisionResult();

// 	for(auto iter = mContactObjects.begin(); iter != mContactObjects.end(); iter++)
// 	{
// 		std::string name = iter->first;
// 		const BodyNode* body = mSkeleton->getBodyNode(name);
// 		for(const auto& shapeNode : body->getShapeNodesWith<CollisionAspect>())
// 		{
// 			Eigen::Vector3d force_vector = Eigen::Vector3d::Zero();
// 			for(const auto& contact : result.getContacts())
// 			{
// 				if(shapeNode == contact.collisionObject1->getShapeFrame() ||
// 					shapeNode == contact.collisionObject2->getShapeFrame())
// 				{
// 					(iter->second).push_back(std::make_pair(contact.force, contact.point));
// 					// std::cout << name << " : " << body->getCOM()[0] << " " << body->getCOM()[1] << " " << body->getCOM()[2] << std::endl;
// 					Eigen::Vector3d cur_force = contact.force;
// 					force_vector += cur_force;
// 				}
// 			}
// 			this->SetContactForce(name, force_vector.norm());
// 			mContactForceDequeMap[name].pop_back();
// 			mContactForceDequeMap[name].push_front(this->GetContactForce(name));
// 		}
// 	}
// }

// void
// Contact::
// SetContactObject(std::string name)
// {
// 	if(mContactObjects.count(name) > 0)
// 	{
// 		std::cout << "Already exist contact!!" << std::endl;
// 		return ;
// 	}

// 	mContactObjects[name] = std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>();
// 	mContactForceDequeMap[name] = std::deque<double>(mWindowSize);
// }

// void
// Contact::
// SetContactForces(std::string name, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> forces)
// {
// 	mContactObjects[name] = forces;
// }

// void
// Contact::
// AddContactForce(std::string name, Eigen::Vector3d f, Eigen::Vector3d p)
// {
// 	mContactObjects[name].push_back(std::make_pair(f,p));
// }

// const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>&
// Contact::
// GetContactForces(std::string name)
// {
// 	return mContactObjects[name];
// }

// void
// Contact::
// SetContactForce(std::string name, double f)
// {
// 	mContactForces[name] = f;
// }

// double
// Contact::
// GetContactForce(std::string name)
// {
// 	return mContactForces[name];
// }

// double
// Contact::
// GetReward()
// {
// 	double err_scale = 1.0;
// 	double contact_scale = 1.0;
// 	double contact_err = 0;

// 	std::vector<double> contact_errs;
// 	for(auto iter = mContactForceDequeMap.begin(); iter != mContactForceDequeMap.end(); iter++)
// 	{
// 		contact_err = 0;
// 		for(int i=0; i<mNumSteps; i++)
// 			contact_err += 0.001*(iter->second).at(i);
// 		contact_err /= (mNumSteps * mMass);
// 		contact_errs.push_back(contact_err);
// 	}

// 	for(auto err : contact_errs)
// 		contact_err += err;

// 	// if(contact_err < 5.0)
// 	//  contact_err = 0.0;
// 	// else
// 	//  contact_err -= 5.0;
// 	double reward = exp(-err_scale * contact_scale * contact_err);
// 	return reward;

// }

}

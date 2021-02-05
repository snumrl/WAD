#include "JointTorque.h"

using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

JointTorque::
JointTorque()
{
}

JointTorque::
~JointTorque()
{
}

void
JointTorque::
Initialize(const dart::dynamics::SkeletonPtr& skel)
{
    windowSize = 200;
    int jointNum = skel->getNumJoints();
    for(int i=0; i<jointNum; i++)\
    {
        const auto& joint = skel->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();

        if(joint->getType() == "FreeJoint")
        {
            mJointTorques[name+"_x"] = std::deque<double>(windowSize);
            mJointTorques[name+"_y"] = std::deque<double>(windowSize);
            mJointTorques[name+"_z"] = std::deque<double>(windowSize);
            mJointTorques[name+"_a"] = std::deque<double>(windowSize);
            mJointTorques[name+"_b"] = std::deque<double>(windowSize);
            mJointTorques[name+"_c"] = std::deque<double>(windowSize);
        }
        else if(joint->getType() == "BallJoint")
        {
            mJointTorques[name+"_x"] = std::deque<double>(windowSize);
            mJointTorques[name+"_y"] = std::deque<double>(windowSize);
            mJointTorques[name+"_z"] = std::deque<double>(windowSize);
        }
        else if(joint->getType() == "RevoluteJoint")
        {
            mJointTorques[name]  = std::deque<double>(windowSize);
        }
        else
        {
        }
    }
}

void
JointTorque::
Reset()
{
    for(auto iter = mJointTorques.begin(); iter != mJointTorques.begin(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);
}

void
JointTorque::
Set(const dart::dynamics::SkeletonPtr& skel, const Eigen::VectorXd& torques)
{
    int jointNum = skel->getNumJoints();
    for(int i=0; i<jointNum; i++)
    {
        auto joint = skel->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();

        if(joint->getType() == "FreeJoint")
        {
            this->Set(name+"_x", torques[idx+0]);
            this->Set(name+"_y", torques[idx+1]);
            this->Set(name+"_z", torques[idx+2]);
            this->Set(name+"_a", torques[idx+3]);
            this->Set(name+"_b", torques[idx+4]);
            this->Set(name+"_c", torques[idx+5]);
        }
        else if(joint->getType() == "BallJoint")
        {
            this->Set(name+"_x", torques[idx+0]);
            this->Set(name+"_y", torques[idx+1]);
            this->Set(name+"_z", torques[idx+2]);
        }
        else if(joint->getType() == "RevoluteJoint")
        {
            this->Set(name, torques[idx+0]);
        }
        else
        {
        }
    }
}

void
JointTorque::
Set(std::string name, double torque)
{
    mJointTorques[name].pop_back();
    mJointTorques[name].push_front(torque);
}

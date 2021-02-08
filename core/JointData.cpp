#include "JointData.h"

using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

JointData::
JointData()
{
}

JointData::
~JointData()
{
}

void
JointData::
Initialize(const dart::dynamics::SkeletonPtr& skel)
{
    windowSize = 200;
    mSkeleton = skel;
    int jointNum = mSkeleton->getNumJoints();
    for(int i=0; i<jointNum; i++)\
    {
        const auto& joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();

        if(joint->getType() == "FreeJoint")
        {
            mTorques[name+"_x"] = std::deque<double>(windowSize);
            mTorques[name+"_y"] = std::deque<double>(windowSize);
            mTorques[name+"_z"] = std::deque<double>(windowSize);
            mTorques[name+"_a"] = std::deque<double>(windowSize);
            mTorques[name+"_b"] = std::deque<double>(windowSize);
            mTorques[name+"_c"] = std::deque<double>(windowSize);
        }
        else if(joint->getType() == "BallJoint")
        {
            mTorques[name+"_x"] = std::deque<double>(windowSize);
            mTorques[name+"_y"] = std::deque<double>(windowSize);
            mTorques[name+"_z"] = std::deque<double>(windowSize);
        }
        else if(joint->getType() == "RevoluteJoint")
        {
            mTorques[name]  = std::deque<double>(windowSize);
        }
        else
        {
        }
    }
}

void
JointData::
Reset()
{
    for(auto iter = mTorques.begin(); iter != mTorques.begin(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);
}

void
JointData::
SetTorques(const Eigen::VectorXd& torques)
{
    int jointNum = mSkeleton->getNumJoints();
    for(int i=0; i<jointNum; i++)
    {
        auto joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();

        if(joint->getType() == "FreeJoint")
        {
            this->SetTorques(name+"_x", torques[idx+0]);
            this->SetTorques(name+"_y", torques[idx+1]);
            this->SetTorques(name+"_z", torques[idx+2]);
            this->SetTorques(name+"_a", torques[idx+3]);
            this->SetTorques(name+"_b", torques[idx+4]);
            this->SetTorques(name+"_c", torques[idx+5]);
        }
        else if(joint->getType() == "BallJoint")
        {
            this->SetTorques(name+"_x", torques[idx+0]);
            this->SetTorques(name+"_y", torques[idx+1]);
            this->SetTorques(name+"_z", torques[idx+2]);
        }
        else if(joint->getType() == "RevoluteJoint")
        {
            this->SetTorques(name, torques[idx+0]);
        }
        else
        {
        }
    }
}

void
JointData::
SetTorques(std::string name, double torque)
{
    mTorques[name].pop_back();
    mTorques[name].push_front(torque);
}

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
    int frameNum = 34;
    int stepNum = 16;
    int dataSize = frameNum*stepNum;

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

        mAngles[name+"_sagittal"] = std::deque<double>(dataSize);
        mAngles[name+"_frontal"] = std::deque<double>(dataSize);
        mAngles[name+"_transverse"] = std::deque<double>(dataSize);

        mAnglesByFrame[name+"_sagittal"] = std::vector<std::vector<double>>(frameNum);
        mAnglesByFrame[name+"_frontal"] = std::vector<std::vector<double>>(frameNum);
        mAnglesByFrame[name+"_transverse"] = std::vector<std::vector<double>>(frameNum);
    }
}

void
JointData::
Reset()
{
    for(auto iter = mTorques.begin(); iter != mTorques.end(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);

    for(auto iter = mAngles.begin(); iter != mAngles.end(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);

    for(auto iter = mAnglesByFrame.begin(); iter != mAnglesByFrame.end(); iter++){
        for(int i = 0; i != (iter->second).size(); i++)
            (iter->second).at(i) = std::vector<double>();
    }
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

void
JointData::
SetAngles(int frame)
{
    int jointNum = mSkeleton->getNumJoints();
    Eigen::VectorXd pos = mSkeleton->getPositions();
    for(int i=0; i<jointNum; i++)
    {
        auto joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();

        if(joint->getType() == "FreeJoint")
        {
            this->SetAngles(name+"_sagittal", pos[idx]);
            this->SetAngles(name+"_frontal", pos[idx+1]);
            this->SetAngles(name+"_transverse", pos[idx+2]);

            this->SetAngles(name+"_sagittal", pos[idx], frame);
            this->SetAngles(name+"_frontal", pos[idx+1], frame);
            this->SetAngles(name+"_transverse", pos[idx+2], frame);
        }
        else if(joint->getType() == "BallJoint")
        {
            this->SetAngles(name+"_sagittal", pos[idx]);
            this->SetAngles(name+"_frontal", pos[idx+1]);
            this->SetAngles(name+"_transverse", pos[idx+2]);

            this->SetAngles(name+"_sagittal", pos[idx], frame);
            this->SetAngles(name+"_frontal", pos[idx+1], frame);
            this->SetAngles(name+"_transverse", pos[idx+2], frame);
        }
        else if(joint->getType() == "RevoluteJoint")
        {
            this->SetAngles(name+"_sagittal", pos[idx]);
            this->SetAngles(name+"_frontal", 0.0);
            this->SetAngles(name+"_transverse", 0.0);

            this->SetAngles(name+"_sagittal", pos[idx], frame);
            this->SetAngles(name+"_frontal", 0.0, frame);
            this->SetAngles(name+"_transverse", 0.0, frame);

        }
        else
        {
        }
    }
}

void
JointData::
SetAngles(std::string name, double angle, int frame)
{
    mAnglesByFrame[name].at(frame).push_back(angle);
}

void
JointData::
SetAngles(std::string name, double angle)
{
    mAngles[name].pop_back();
    mAngles[name].push_front(angle);
}

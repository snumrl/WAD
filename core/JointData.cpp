#include "JointData.h"

namespace WAD
{

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
Initialize(const SkeletonPtr& skel)
{
    mSkeleton = skel;
    mOnlyLowerBody = true;
    if(mOnlyLowerBody)
        mJointNum = 11;
    else 
        mJointNum = mSkeleton->getNumJoints();

    mDof = mSkeleton->getNumDofs();

    mOnCycle = false;
    mPhasePrev = -1;
    mCycleStep = 0;
    mCycleTorqueErr = 0.0;
    mCycleTorqueSum = 0.0;

    mWindowSize = 540;
   
    int jointNum = mSkeleton->getNumJoints();
    for(int i=0; i<jointNum; i++)
    {
        const auto& joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();
        // std::cout << i << " :" << name << std::endl;

        if(joint->getType() == "FreeJoint")
        {
            mTorques[name+"_x"] = std::deque<double>(mWindowSize);
            mTorques[name+"_y"] = std::deque<double>(mWindowSize);
            mTorques[name+"_z"] = std::deque<double>(mWindowSize);
            mTorques[name+"_a"] = std::deque<double>(mWindowSize);
            mTorques[name+"_b"] = std::deque<double>(mWindowSize);
            mTorques[name+"_c"] = std::deque<double>(mWindowSize);

            // mTorquesNorm[name] = std::deque<double>(mWindowSize);

            mTorquesPhase[name+"_x"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_y"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_z"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_a"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_b"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_c"] = std::deque<std::pair<double, double>>();

            mTorquesNormPhase[name] = std::deque<std::pair<double, double>>(mWindowSize);
        }
        else if(joint->getType() == "BallJoint")
        {
            mTorques[name+"_x"] = std::deque<double>(mWindowSize);
            mTorques[name+"_y"] = std::deque<double>(mWindowSize);
            mTorques[name+"_z"] = std::deque<double>(mWindowSize);
            // mTorquesNorm[name] = std::deque<double>(mWindowSize);

            mTorquesPhase[name+"_x"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_y"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_z"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesNormPhase[name] = std::deque<std::pair<double, double>>(mWindowSize);
        }
        else if(joint->getType() == "RevoluteJoint")
        {
           mTorques[name+"_x"] = std::deque<double>(mWindowSize);
            mTorques[name+"_y"] = std::deque<double>(mWindowSize);
            mTorques[name+"_z"] = std::deque<double>(mWindowSize);

            mTorquesPhase[name+"_x"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_y"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesPhase[name+"_z"] = std::deque<std::pair<double, double>>(mWindowSize);
            mTorquesNormPhase[name] = std::deque<std::pair<double, double>>(mWindowSize);
        }

        mAngles[name+"_sagittal"] = std::deque<double>(mWindowSize);
        mAngles[name+"_frontal"] = std::deque<double>(mWindowSize);
        mAngles[name+"_transverse"] = std::deque<double>(mWindowSize);

        mAnglesPhase[name+"_sagittal"] = std::deque<std::pair<double, double>>(mWindowSize);
        mAnglesPhase[name+"_frontal"] = std::deque<std::pair<double, double>>(mWindowSize);
        mAnglesPhase[name+"_transverse"] = std::deque<std::pair<double, double>>(mWindowSize);
    }
}

void
JointData::
Reset()
{
    for(auto iter = mTorques.begin(); iter != mTorques.end(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);

    for(auto iter = mTorquesNorm.begin(); iter != mTorquesNorm.end(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);

    for(auto iter = mAngles.begin(); iter != mAngles.end(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);

    for(auto iter = mTorquesPhase.begin(); iter != mTorquesPhase.end(); iter++)
        (iter->second) = std::deque<std::pair<double, double>>(mWindowSize);

    for(auto iter = mTorquesNormPhase.begin(); iter != mTorquesNormPhase.end(); iter++)
        (iter->second) = std::deque<std::pair<double, double>>(mWindowSize);

    for(auto iter = mAnglesPhase.begin(); iter != mAnglesPhase.end(); iter++)
        (iter->second) = std::deque<std::pair<double, double>>(mWindowSize);

    mOnCycle = false;
    mPhasePrev = -1;
    mCycleStep = 0;
    mCycleTorqueErr = 0.0;
    mCycleTorqueSum = 0.0;
}

void
JointData::
SetTorques(std::string name, double torque, double phase)
{
    mTorques[name].pop_back();
    mTorques[name].push_front(torque);

    mTorquesPhase[name].pop_back();
    mTorquesPhase[name].push_front(std::make_pair(phase, torque));
}

void
JointData::
SetTorquesNorm(std::string name, double norm, double phase)
{
    mTorquesNorm[name].pop_back();
    mTorquesNorm[name].push_front(norm);

    mTorquesNormPhase[name].pop_back();
    mTorquesNormPhase[name].push_front(std::make_pair(phase, norm));
}

void
JointData::
SetTorques(const Eigen::VectorXd& torques, double phase, double frame)
{
    if(mPhasePrev > phase)
    {
        if(mOnCycle){
            mCycleTorqueErr = mCycleTorqueSum;
            mCycleTorqueErr /= (double)(mDof-6);
            mCycleTorqueErr /= (double)(mCycleStep);            
        }
        else{
            mOnCycle = true;
        }
        mCycleStep = 0;
        mCycleTorqueSum = 0;
    }

    if(mOnCycle)
    {
        int jointNum = mSkeleton->getNumJoints();
        for(int i=0; i<jointNum; i++)
        {
            auto joint = mSkeleton->getJoint(i);
            int idx = joint->getIndexInSkeleton(0);
            std::string name = joint->getName();
            std::string type = joint->getType();

            double norm;
            if(type == "FreeJoint")
            {
                // norm = (torques.segment(idx, 6)).norm();
                // if(i < mJointNum)
                //     mCycleTorqueSum += norm;
                // this->SetTorquesNorm(name, norm, phase);

                this->SetTorques(name+"_x", torques[idx+0], phase);
                this->SetTorques(name+"_y", torques[idx+1], phase);
                this->SetTorques(name+"_z", torques[idx+2], phase);
                this->SetTorques(name+"_a", torques[idx+3], phase);
                this->SetTorques(name+"_b", torques[idx+4], phase);
                this->SetTorques(name+"_c", torques[idx+5], phase);
            }
            else if(type == "BallJoint")
            {
                // norm = (torques.segment(idx, 3)).norm();
                // if(i < mJointNum)
                //     mCycleTorqueSum += norm;
                // this->SetTorquesNorm(name, norm, phase);

                mCycleTorqueSum += fabs(torques[idx])/(double)mMaxForces[idx];
                mCycleTorqueSum += fabs(torques[idx+1])/(double)mMaxForces[idx+1];
                mCycleTorqueSum += fabs(torques[idx+2])/(double)mMaxForces[idx+2];
                
                this->SetTorques(name+"_x", torques[idx+0], phase);
                this->SetTorques(name+"_y", torques[idx+1], phase);
                this->SetTorques(name+"_z", torques[idx+2], phase);
            }
            else if(type == "RevoluteJoint")
            {
                // norm = fabs(torques[idx]);
                // if(i < mJointNum)
                //     mCycleTorqueSum += norm;
                // this->SetTorquesNorm(name, norm, phase);

                mCycleTorqueSum += fabs(torques[idx])/(double)mMaxForces[idx];

                this->SetTorques(name+"_x", torques[idx+0], phase);
                this->SetTorques(name+"_y", 0.0, phase);
                this->SetTorques(name+"_z", 0.0, phase);
            }
            else
            {
            }
        }
        mCycleStep++;
    }

    mPhasePrev = phase;
}

void
JointData::
SetAngles(std::string name, double angle, double phase)
{
    mAngles[name].pop_back();
    mAngles[name].push_front(angle);

    mAnglesPhase[name].pop_back();
    mAnglesPhase[name].push_front(std::make_pair(phase, angle));
}

void
JointData::
SetAngles(double phase)
{
    if(mOnCycle)
    {
        int jointNum = mSkeleton->getNumJoints();
        Eigen::VectorXd pos = mSkeleton->getPositions();
        for(int i=0; i<jointNum; i++)
        {
            auto joint = mSkeleton->getJoint(i);
            int idx = joint->getIndexInSkeleton(0);
            std::string name = joint->getName();
            std::string type = joint->getType();

            if(type == "FreeJoint")
            {
                this->SetAngles(name+"_sagittal", pos[idx], phase);
                this->SetAngles(name+"_frontal", pos[idx+1], phase);
                this->SetAngles(name+"_transverse", pos[idx+2], phase);
            }
            else if(type == "BallJoint")
            {
                this->SetAngles(name+"_sagittal", pos[idx], phase);
                this->SetAngles(name+"_frontal", pos[idx+1], phase);
                this->SetAngles(name+"_transverse", pos[idx+2], phase);
            }
            else if(type == "RevoluteJoint")
            {
                this->SetAngles(name+"_sagittal", pos[idx], phase);
                this->SetAngles(name+"_frontal", 0.0, phase);
                this->SetAngles(name+"_transverse", 0.0, phase);
            }
            else
            {
            }
        }
    }
}

double
JointData::
GetReward()
{
    double err_scale = 1.0;
    double torque_scale = 10.0;
    double torque_err = 0.0;

    double reward = 0.0;
    if(mCycleTorqueErr != 0.0){
        torque_err = mCycleTorqueErr;
        reward = exp(-err_scale * torque_scale * torque_err);
        // std::cout << "err : " << torque_err << std::endl;
        // std::cout << "rew : " << reward << std::endl;
        mCycleTorqueErr = 0.0;
    }

    return reward;
}

}

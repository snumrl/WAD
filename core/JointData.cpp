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
    mCycleStep = 0;
    mCycleTorqueErr = 0.0;
    mCycleTorqueSum = 0.0;
    
    mPhaseStateRight = 1;
    mPhaseStateRightPrev = 1;

    mWindowSize = 540;
    
    int jointNum = mSkeleton->getNumJoints();
    for(int i=0; i<jointNum; i++)
    {
        const auto& joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();
        
        if(joint->getType() == "FreeJoint")
        {
            mTorques[name+"_a"] = std::deque<double>(mWindowSize);
            mTorques[name+"_b"] = std::deque<double>(mWindowSize);
            mTorques[name+"_c"] = std::deque<double>(mWindowSize);

            mTorquesGaitPhase[name+"_a"] = std::deque<double>();
            mTorquesGaitPhase[name+"_b"] = std::deque<double>();
            mTorquesGaitPhase[name+"_c"] = std::deque<double>();

            mTorquesGaitPhasePrev[name+"_a"] = std::deque<double>();
            mTorquesGaitPhasePrev[name+"_b"] = std::deque<double>();
            mTorquesGaitPhasePrev[name+"_c"] = std::deque<double>();
        }
        
        mTorques[name+"_x"] = std::deque<double>(mWindowSize);
        mTorques[name+"_y"] = std::deque<double>(mWindowSize);
        mTorques[name+"_z"] = std::deque<double>(mWindowSize);

        mTorquesGaitPhase[name+"_x"] = std::deque<double>();
        mTorquesGaitPhase[name+"_y"] = std::deque<double>();
        mTorquesGaitPhase[name+"_z"] = std::deque<double>();

        mTorquesGaitPhasePrev[name+"_x"] = std::deque<double>();
        mTorquesGaitPhasePrev[name+"_y"] = std::deque<double>();
        mTorquesGaitPhasePrev[name+"_z"] = std::deque<double>();
     
        mAngles[name+"_sagittal"] = std::deque<double>(mWindowSize);
        mAngles[name+"_frontal"] = std::deque<double>(mWindowSize);
        mAngles[name+"_transverse"] = std::deque<double>(mWindowSize);

        mAnglesGaitPhase[name+"_sagittal"] = std::deque<double>();
        mAnglesGaitPhase[name+"_frontal"] = std::deque<double>();
        mAnglesGaitPhase[name+"_transverse"] = std::deque<double>();

        mAnglesGaitPhasePrev[name+"_sagittal"] = std::deque<double>();
        mAnglesGaitPhasePrev[name+"_frontal"] = std::deque<double>();
        mAnglesGaitPhasePrev[name+"_transverse"] = std::deque<double>();

        mAnglesGaitPhaseRef[name+"_sagittal"] = std::deque<double>();
        mAnglesGaitPhaseRef[name+"_frontal"] = std::deque<double>();
        mAnglesGaitPhaseRef[name+"_transverse"] = std::deque<double>();

        mAnglesGaitPhaseRefPrev[name+"_sagittal"] = std::deque<double>();
        mAnglesGaitPhaseRefPrev[name+"_frontal"] = std::deque<double>();
        mAnglesGaitPhaseRefPrev[name+"_transverse"] = std::deque<double>();
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

    for(auto iter = mTorquesGaitPhase.begin(); iter != mTorquesGaitPhase.end(); iter++)
        (iter->second).clear();

    for(auto iter = mTorquesGaitPhasePrev.begin(); iter != mTorquesGaitPhasePrev.end(); iter++)
        (iter->second).clear();

    for(auto iter = mAnglesGaitPhase.begin(); iter != mAnglesGaitPhase.end(); iter++)
        (iter->second).clear();

    for(auto iter = mAnglesGaitPhasePrev.begin(); iter != mAnglesGaitPhasePrev.end(); iter++)
        (iter->second).clear();

    for(auto iter = mAnglesGaitPhaseRef.begin(); iter != mAnglesGaitPhaseRef.end(); iter++)
        (iter->second).clear();

    for(auto iter = mAnglesGaitPhaseRefPrev.begin(); iter != mAnglesGaitPhaseRefPrev.end(); iter++)
        (iter->second).clear();
    
    mOnCycle = false;
    
    mCycleStep = 0;
    mCycleTorqueErr = 0.0;
    mCycleTorqueSum = 0.0;

    mPhaseStateRight = 1;
    mPhaseStateRightPrev = 1;
}

void
JointData::
SetPhaseState(int phaseState, double time)
{
    mPhaseStateRight = phaseState;
    if(mPhaseStateRight != mPhaseStateRightPrev)
    {
        if(mPhaseStateRight == 1)
        {
            if(mOnCycle)
            {
                mCycleTorqueErr = mCycleTorqueSum;
                mCycleTorqueErr /= (double)(mDof-6);
                mCycleTorqueErr /= (double)(mCycleStep);            
            }
            else
            {
                mOnCycle = true;
            }
            mCycleStep = 0;
            mCycleTorqueSum = 0;    

            for(auto t : mTorquesGaitPhase)
            {
                mTorquesGaitPhasePrev[t.first].clear();
                mTorquesGaitPhasePrev[t.first] = t.second;
                mTorquesGaitPhase[t.first].clear();                
            }

            for(auto a : mAnglesGaitPhase)
            {
                mAnglesGaitPhasePrev[a.first].clear();
                mAnglesGaitPhasePrev[a.first] = a.second;
                mAnglesGaitPhase[a.first].clear();                
            }

            for(auto a : mAnglesGaitPhaseRef)
            {
                mAnglesGaitPhaseRefPrev[a.first].clear();
                mAnglesGaitPhaseRefPrev[a.first] = a.second;
                mAnglesGaitPhaseRef[a.first].clear();                
            }
        }
    }    

    mPhaseStateRightPrev = mPhaseStateRight;
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
SetTorquesGaitPhase(std::string name, double torque)
{
    mTorquesGaitPhase[name].push_front(torque);
}

void
JointData::
SetTorques(const Eigen::VectorXd& torques)
{
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
                this->SetTorques(name+"_x", torques[idx+0]);
                this->SetTorques(name+"_y", torques[idx+1]);
                this->SetTorques(name+"_z", torques[idx+2]);
                this->SetTorques(name+"_a", torques[idx+3]);
                this->SetTorques(name+"_b", torques[idx+4]);
                this->SetTorques(name+"_c", torques[idx+5]);

                this->SetTorquesGaitPhase(name+"_x", torques[idx+0]);
                this->SetTorquesGaitPhase(name+"_y", torques[idx+1]);
                this->SetTorquesGaitPhase(name+"_z", torques[idx+2]);
                this->SetTorquesGaitPhase(name+"_a", torques[idx+3]);
                this->SetTorquesGaitPhase(name+"_b", torques[idx+4]);
                this->SetTorquesGaitPhase(name+"_c", torques[idx+5]);
            }
            else if(type == "BallJoint")
            {
                mCycleTorqueSum += fabs(torques[idx])/(double)mMaxForces[idx];
                mCycleTorqueSum += fabs(torques[idx+1])/(double)mMaxForces[idx+1];
                mCycleTorqueSum += fabs(torques[idx+2])/(double)mMaxForces[idx+2];
                
                this->SetTorques(name+"_x", torques[idx+0]);
                this->SetTorques(name+"_y", torques[idx+1]);
                this->SetTorques(name+"_z", torques[idx+2]);

                this->SetTorquesGaitPhase(name+"_x", torques[idx+0]);
                this->SetTorquesGaitPhase(name+"_y", torques[idx+1]);
                this->SetTorquesGaitPhase(name+"_z", torques[idx+2]);
            }
            else if(type == "RevoluteJoint")
            {
                mCycleTorqueSum += fabs(torques[idx])/(double)mMaxForces[idx];

                this->SetTorques(name+"_x", torques[idx+0]);
                this->SetTorques(name+"_y", 0.0);
                this->SetTorques(name+"_z", 0.0);

                this->SetTorquesGaitPhase(name+"_x", torques[idx+0]);
                this->SetTorquesGaitPhase(name+"_y", 0.0);
                this->SetTorquesGaitPhase(name+"_z", 0.0);
            }
            else
            {
            }
        }
        mCycleStep++;
    }
}

void
JointData::
SetAngles(std::string name, double angle)
{
    mAngles[name].pop_back();
    mAngles[name].push_back(angle);    
}

void
JointData::
SetAnglesGaitPhase(std::string name, double angle)
{
    mAnglesGaitPhase[name].push_back(angle);    
}

void
JointData::
SetAngles()
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
                this->SetAngles(name+"_sagittal", pos[idx]);
                this->SetAngles(name+"_frontal", pos[idx+1]);
                this->SetAngles(name+"_transverse", pos[idx+2]);

                this->SetAnglesGaitPhase(name+"_sagittal", pos[idx]);
                this->SetAnglesGaitPhase(name+"_frontal", pos[idx+1]);
                this->SetAnglesGaitPhase(name+"_transverse", pos[idx+2]);
            }
            else if(type == "BallJoint")
            {
                this->SetAngles(name+"_sagittal", pos[idx]);
                this->SetAngles(name+"_frontal", pos[idx+1]);
                this->SetAngles(name+"_transverse", pos[idx+2]);

                this->SetAnglesGaitPhase(name+"_sagittal", pos[idx]);
                this->SetAnglesGaitPhase(name+"_frontal", pos[idx+1]);
                this->SetAnglesGaitPhase(name+"_transverse", pos[idx+2]);
            }
            else if(type == "RevoluteJoint")
            {
                this->SetAngles(name+"_sagittal", pos[idx]);
                this->SetAngles(name+"_frontal", 0.0);
                this->SetAngles(name+"_transverse", 0.0);

                this->SetAnglesGaitPhase(name+"_sagittal", pos[idx]);
                this->SetAnglesGaitPhase(name+"_frontal", 0.0);
                this->SetAnglesGaitPhase(name+"_transverse", 0.0);            
            }
            else
            {
            }
        }
    }
}


void
JointData::
SetAnglesGaitPhaseRef(std::string name, double angle)
{
    mAnglesGaitPhaseRef[name].push_back(angle);        
}

void
JointData::
SetAnglesRef()
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
                this->SetAnglesGaitPhaseRef(name+"_sagittal", pos[idx]);
                this->SetAnglesGaitPhaseRef(name+"_frontal", pos[idx+1]);
                this->SetAnglesGaitPhaseRef(name+"_transverse", pos[idx+2]);
            }
            else if(type == "BallJoint")
            {
                this->SetAnglesGaitPhaseRef(name+"_sagittal", pos[idx]);
                this->SetAnglesGaitPhaseRef(name+"_frontal", pos[idx+1]);
                this->SetAnglesGaitPhaseRef(name+"_transverse", pos[idx+2]);
            }
            else if(type == "RevoluteJoint")
            {
                this->SetAnglesGaitPhaseRef(name+"_sagittal", pos[idx]);
                this->SetAnglesGaitPhaseRef(name+"_frontal", 0.0);
                this->SetAnglesGaitPhaseRef(name+"_transverse", 0.0);            
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

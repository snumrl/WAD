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
Initialize(const SkeletonPtr& skel, int simHz, int conHz)
{
    mSkeleton = skel;
    mDof = mSkeleton->getNumDofs();
    mJointNum = mSkeleton->getNumJoints();
    
    mSimulationHz = simHz;
    mControlHz = conHz;

    mOnCycle = false;
    mCycleStep = 0;
    mCycleTorqueErr = 0.0;
    mCycleTorqueSum = 0.0;
    
    mPhaseStateRight = 1;
    mPhaseStateRightPrev = 1;

    mStrideLeft = 0.0;
    mStrideRight = 0.0;
    mCadenceLeft = 0.0;
    mCadenceRight = 0.0;

    mTimeLeft = 0.0;
    mTimeLeftPrev = 0.0;
    mTimeRight = 0.0;
    mTimeRightPrev = 0.0;

    mComLeft  = Eigen::Vector3d::Zero();
    mComRight = Eigen::Vector3d::Zero();
    mComLeftPrev  = Eigen::Vector3d::Zero();
    mComRightPrev = Eigen::Vector3d::Zero();

    mWindowSize = 540;
    this->Initialize_Torques();
    this->Initialize_Moments();
    this->Initialize_Angles();    
}

void
JointData::
Initialize_Torques()
{
    for(int i=0; i<mJointNum; i++)
    {
        const auto& joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();
        
        if(joint->getType() == "FreeJoint")
        {
            for(int i=0; i<3; i++)
            {
                std::string namePost;
                if(i==0)
                    namePost = "_a";
                else if(i==1)
                    namePost = "_b";
                else if(i==2)
                    namePost = "_c";
            
                mTorques[name+namePost] = std::deque<double>(mWindowSize);    
                mTorquesGaitPhase[name+namePost] = std::deque<double>();
                mTorquesGaitPhasePrev[name+namePost] = std::deque<double>();
            }
        }

        for(int i=0; i<3; i++)
        {
            std::string namePost;
            if(i==0)
                namePost = "_x";
            else if(i==1)
                namePost = "_y";
            else if(i==2)
                namePost = "_z";
            
            mTorques[name+namePost] = std::deque<double>(mWindowSize);    
            mTorquesGaitPhase[name+namePost] = std::deque<double>();
            mTorquesGaitPhasePrev[name+namePost] = std::deque<double>();            
        }
    }

    mDeviceTorques["FemurL_x"] = std::deque<double>(mWindowSize);
    mDeviceTorquesGaitPhase["FemurL_x"] = std::deque<double>();
    mDeviceTorquesGaitPhasePrev["FemurL_x"] = std::deque<double>();

    mDeviceTorques["FemurR_x"] = std::deque<double>(mWindowSize);
    mDeviceTorquesGaitPhase["FemurR_x"] = std::deque<double>();
    mDeviceTorquesGaitPhasePrev["FemurR_x"] = std::deque<double>();
}

void
JointData::
Initialize_Moments()
{
    for(int i=0; i<mJointNum; i++)
    {
        const auto& joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();
        
        if(joint->getType() == "FreeJoint")
        {
            for(int i=0; i<3; i++)
            {
                std::string namePost;
                if(i==0)
                    namePost = "_a";
                else if(i==1)
                    namePost = "_b";
                else if(i==2)
                    namePost = "_c";
            
                mMoments[name+namePost] = std::deque<double>(mWindowSize);    
                mMomentsGaitPhase[name+namePost] = std::deque<double>();
                mMomentsGaitPhasePrev[name+namePost] = std::deque<double>();
            }
        }

        for(int i=0; i<3; i++)
        {
            std::string namePost;
            if(i==0)
                namePost = "_x";
            else if(i==1)
                namePost = "_y";
            else if(i==2)
                namePost = "_z";
            
            mMoments[name+namePost] = std::deque<double>(mWindowSize);    
            mMomentsGaitPhase[name+namePost] = std::deque<double>();
            mMomentsGaitPhasePrev[name+namePost] = std::deque<double>();            
        }
    }
}

void
JointData::
Initialize_Angles()
{
    for(int i=0; i<mJointNum; i++)
    {
        const auto& joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();
        
        for(int i=0; i<3; i++)
        {
            std::string namePost;
            if(i==0)
                namePost = "_sagittal";
            else if(i==1)
                namePost = "_frontal";
            else if(i==2)
                namePost = "_transverse";
                       
            mAngles[name+namePost] = std::deque<double>(mWindowSize);    
            mAnglesGaitPhaseLeft[name+namePost] = std::deque<double>();
            mAnglesGaitPhaseLeftPrev[name+namePost] = std::deque<double>();
            mAnglesGaitPhaseRight[name+namePost] = std::deque<double>();
            mAnglesGaitPhaseRightPrev[name+namePost] = std::deque<double>();
        }
    }
}

void
JointData::
Reset()
{
    for(auto iter = mTorques.begin(); iter != mTorques.end(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);

    for(auto iter = mTorquesGaitPhase.begin(); iter != mTorquesGaitPhase.end(); iter++)
        (iter->second).clear();

    for(auto iter = mTorquesGaitPhasePrev.begin(); iter != mTorquesGaitPhasePrev.end(); iter++)
        (iter->second).clear();

    for(auto iter = mDeviceTorques.begin(); iter != mDeviceTorques.end(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);

    for(auto iter = mDeviceTorquesGaitPhase.begin(); iter != mDeviceTorquesGaitPhase.end(); iter++)
        (iter->second).clear();

    for(auto iter = mDeviceTorquesGaitPhasePrev.begin(); iter != mDeviceTorquesGaitPhasePrev.end(); iter++)
        (iter->second).clear();

    for(auto iter = mMomentsGaitPhase.begin(); iter != mMomentsGaitPhase.end(); iter++)
        (iter->second).clear();

    for(auto iter = mMomentsGaitPhasePrev.begin(); iter != mMomentsGaitPhasePrev.end(); iter++)
        (iter->second).clear();

    for(auto iter = mAngles.begin(); iter != mAngles.end(); iter++)
        std::fill(iter->second.begin(), iter->second.end(), 0.0);
    
    for(auto iter = mAnglesGaitPhaseLeft.begin(); iter != mAnglesGaitPhaseLeft.end(); iter++)
        (iter->second).clear();

    for(auto iter = mAnglesGaitPhaseLeftPrev.begin(); iter != mAnglesGaitPhaseLeftPrev.end(); iter++)
        (iter->second).clear();
    
    for(auto iter = mAnglesGaitPhaseRight.begin(); iter != mAnglesGaitPhaseRight.end(); iter++)
        (iter->second).clear();

    for(auto iter = mAnglesGaitPhaseRightPrev.begin(); iter != mAnglesGaitPhaseRightPrev.end(); iter++)
        (iter->second).clear();

    mTimeLeft = 0.0;
    mTimeLeftPrev = 0.0;
    mTimeRight = 0.0;
    mTimeRightPrev = 0.0;

    mComLeft.setZero();
    mComRight.setZero();
    mComLeftPrev.setZero();
    mComRightPrev.setZero();

    mOnCycle = false;
    mCycleStep = 0;
    mCycleTorqueErr = 0.0;
    mCycleTorqueSum = 0.0;

    mPhaseStateRight = 1;
    mPhaseStateRightPrev = 1;
}

void
JointData::
SetPhaseState(int stateLeft, Eigen::Vector3d comLeft, int stateRight, Eigen::Vector3d comRight, double time)
{
    this->SetPhaseStateLeft(stateLeft, comLeft, time);
    this->SetPhaseStateRight(stateRight, comRight, time);    
    mCycleStep++;
}

void
JointData::
SetPhaseStateLeft(int phaseState, Eigen::Vector3d com, double time)
{
    mPhaseStateLeft = phaseState;
    if(mPhaseStateLeft != mPhaseStateLeftPrev)
    {
        if(mPhaseStateLeft == 1)
        {
            for(auto a : mAnglesGaitPhaseLeft)
            {
                mAnglesGaitPhaseLeftPrev[a.first].clear();
                mAnglesGaitPhaseLeftPrev[a.first] = a.second;
                mAnglesGaitPhaseLeft[a.first].clear();                
            }            
        }        
    }    

    mPhaseStateLeftPrev = mPhaseStateLeft;
}

void
JointData::
SetPhaseStateRight(int phaseState, Eigen::Vector3d com, double time)
{
    mPhaseStateRight = phaseState;
    if(mPhaseStateRight != mPhaseStateRightPrev)
    {
        if(mPhaseStateRight == 1)
        {
            mTimeRight = time;
            mComRight = com;

            this->ChangePhaseTorques();
            this->ChangePhaseMoments();

            for(auto a : mAnglesGaitPhaseRight)
            {
                mAnglesGaitPhaseRightPrev[a.first].clear();
                mAnglesGaitPhaseRightPrev[a.first] = a.second;
                mAnglesGaitPhaseRight[a.first].clear();                
            }

            mTimeRightPrev = mTimeRight;
            mComRightPrev = mComRight;  
        }
    }    

    mPhaseStateRightPrev = mPhaseStateRight;
}

void
JointData::
ChangePhaseTorques()
{
    if(mComRightPrev.norm() == 0){
        mStrideRight = 1.34;
        mCadenceRight = 0;
    }
    else{
        mStrideRight = (mComRight - mComRightPrev).norm();
        mCadenceRight = 60.0/(mTimeRight - mTimeRightPrev);
    }

    if(mOnCycle)
    {
        mCycleTorqueErr = mCycleTorqueSum * (1.0/(double)mSimulationHz)*mCycleStep;
        // mCycleTorqueErr /= (double)(mDof-6);
        mCycleTorqueErr /= (double)(14);
        mCycleTorqueErr /= (double)(mStrideRight);            
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

    for(auto t : mDeviceTorquesGaitPhase)
    {
        mDeviceTorquesGaitPhasePrev[t.first].clear();
        mDeviceTorquesGaitPhasePrev[t.first] = t.second;
        mDeviceTorquesGaitPhase[t.first].clear();                
    }
}

void
JointData::
ChangePhaseMoments()
{
    for(auto t : mMomentsGaitPhase)
    {
        mMomentsGaitPhasePrev[t.first].clear();
        mMomentsGaitPhasePrev[t.first] = t.second;
        mMomentsGaitPhase[t.first].clear();                
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
                if(name == "FemurL" || name == "FemurR" || name == "TalusL" || name == "TalusR")
                {
                    mCycleTorqueSum += fabs(torques[idx])/(double)mMaxForces[idx];
                    mCycleTorqueSum += fabs(torques[idx+1])/(double)mMaxForces[idx+1];
                    mCycleTorqueSum += fabs(torques[idx+2])/(double)mMaxForces[idx+2];
                }
                                
                this->SetTorques(name+"_x", torques[idx+0]);
                this->SetTorques(name+"_y", torques[idx+1]);
                this->SetTorques(name+"_z", torques[idx+2]);

                this->SetTorquesGaitPhase(name+"_x", torques[idx+0]);
                this->SetTorquesGaitPhase(name+"_y", torques[idx+1]);
                this->SetTorquesGaitPhase(name+"_z", torques[idx+2]);
            }
            else if(type == "RevoluteJoint")
            {
                if(name == "TibiaL" || name == "TibiaR")
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
    }
}

void
JointData::
SetDeviceTorques(std::string name, double torque)
{
    mDeviceTorques[name].pop_back();
    mDeviceTorques[name].push_front(torque);
}

void
JointData::
SetDeviceTorquesGaitPhase(std::string name, double torque)
{
    mDeviceTorquesGaitPhase[name].push_front(torque);
}

void
JointData::
SetDeviceTorques(const Eigen::VectorXd& torques)
{
    if(mOnCycle)
    {        
        this->SetDeviceTorques("FemurL_x", 10*torques[6]);
        this->SetDeviceTorques("FemurR_x", 10*torques[9]);

        this->SetDeviceTorquesGaitPhase("FemurL_x", 10*torques[6]);
        this->SetDeviceTorquesGaitPhase("FemurR_x", 10*torques[9]);
    }
}

void
JointData::
SetMoments(std::string name, double moment)
{
    mMoments[name].pop_back();
    mMoments[name].push_front(moment);
}

void
JointData::
SetMomentsGaitPhase(std::string name, double moment)
{
    mMomentsGaitPhase[name].push_front(moment);    
}

void
JointData::
SetMoments(const Eigen::VectorXd& moments)
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
                this->SetMoments(name+"_x", moments[idx+0]);
                this->SetMoments(name+"_y", moments[idx+1]);
                this->SetMoments(name+"_z", moments[idx+2]);
                this->SetMoments(name+"_a", moments[idx+3]);
                this->SetMoments(name+"_b", moments[idx+4]);
                this->SetMoments(name+"_c", moments[idx+5]);

                this->SetMomentsGaitPhase(name+"_x", moments[idx+0]);
                this->SetMomentsGaitPhase(name+"_y", moments[idx+1]);
                this->SetMomentsGaitPhase(name+"_z", moments[idx+2]);
                this->SetMomentsGaitPhase(name+"_a", moments[idx+3]);
                this->SetMomentsGaitPhase(name+"_b", moments[idx+4]);
                this->SetMomentsGaitPhase(name+"_c", moments[idx+5]);
            }
            else if(type == "BallJoint")
            {
                this->SetMoments(name+"_x", moments[idx+0]);
                this->SetMoments(name+"_y", moments[idx+1]);
                this->SetMoments(name+"_z", moments[idx+2]);

                this->SetMomentsGaitPhase(name+"_x", moments[idx+0]);
                this->SetMomentsGaitPhase(name+"_y", moments[idx+1]);
                this->SetMomentsGaitPhase(name+"_z", moments[idx+2]);
            }
            else if(type == "RevoluteJoint")
            {
                this->SetMoments(name+"_x", moments[idx+0]);
                this->SetMoments(name+"_y", 0.0);
                this->SetMoments(name+"_z", 0.0);

                this->SetMomentsGaitPhase(name+"_x", moments[idx+0]);
                this->SetMomentsGaitPhase(name+"_y", 0.0);
                this->SetMomentsGaitPhase(name+"_z", 0.0);
            }
            else
            {
            }
        }        
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
    mAnglesGaitPhaseLeft[name].push_back(angle);    
    mAnglesGaitPhaseRight[name].push_back(angle);    
}

void
JointData::
SetAngles()
{
    if(mOnCycle)
    {
        Eigen::VectorXd pos = mSkeleton->getPositions();
        for(int i=0; i<mJointNum; i++)
        {
            auto joint = mSkeleton->getJoint(i);
            int idx = joint->getIndexInSkeleton(0);
            std::string name = joint->getName();
            std::string type = joint->getType();
            double rad2deg = 180.0/M_PI;

            double p0 = pos[idx+0]*rad2deg;
            double p1 = (type == "RevoluteJoint") ? 0.0 : pos[idx+1]*rad2deg;
            double p2 = (type == "RevoluteJoint") ? 0.0 : pos[idx+2]*rad2deg;

            this->SetAngles(name+"_sagittal",  p0);
            this->SetAngles(name+"_frontal",   p1);
            this->SetAngles(name+"_transverse",p2);

            this->SetAnglesGaitPhase(name+"_sagittal",  p0);
            this->SetAnglesGaitPhase(name+"_frontal",   p1);
            this->SetAnglesGaitPhase(name+"_transverse",p2);
        }
    }
}

double
JointData::
GetReward()
{
    double err_scale = 1.0;
    double torque_scale = 0.01;
    double torque_err = 0.0;

    double reward = 0.0;
    if(mCycleTorqueErr != 0.0){
        torque_err = mCycleTorqueErr;
        reward = exp(-err_scale * torque_scale * torque_err);        
        mCycleTorqueErrPrev = mCycleTorqueErr;
        mCycleTorqueErr = 0.0;        
    }

    return reward;
}

}

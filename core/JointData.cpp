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
    mWindowSize = 200;
    mNumFrames = 34;
    mNumSteps = 16;
    mDataSize = mNumFrames*mNumSteps;

    mTorquesNormCum = 0.0;

    mSkeleton = skel;
    int jointNum = mSkeleton->getNumJoints();
    for(int i=0; i<jointNum; i++)\
    {
        const auto& joint = mSkeleton->getJoint(i);
        int idx = joint->getIndexInSkeleton(0);
        std::string name = joint->getName();

        if(joint->getType() == "FreeJoint")
        {
            mTorques[name+"_x"] = std::deque<double>(mWindowSize);
            mTorques[name+"_y"] = std::deque<double>(mWindowSize);
            mTorques[name+"_z"] = std::deque<double>(mWindowSize);
            mTorques[name+"_a"] = std::deque<double>(mWindowSize);
            mTorques[name+"_b"] = std::deque<double>(mWindowSize);
            mTorques[name+"_c"] = std::deque<double>(mWindowSize);

            // mTorquesNorm[name] = std::deque<double>(mNumFrames);
            mTorquesNorm[name] = std::deque<double>(mDataSize);
        }
        else if(joint->getType() == "BallJoint")
        {
            mTorques[name+"_x"] = std::deque<double>(mWindowSize);
            mTorques[name+"_y"] = std::deque<double>(mWindowSize);
            mTorques[name+"_z"] = std::deque<double>(mWindowSize);

            // mTorquesNorm[name] = std::deque<double>(mNumFrames);
            mTorquesNorm[name] = std::deque<double>(mDataSize);
        }
        else if(joint->getType() == "RevoluteJoint")
        {
            mTorques[name]  = std::deque<double>(mWindowSize);

            // mTorquesNorm[name] = std::deque<double>(mNumFrames);
            mTorquesNorm[name] = std::deque<double>(mDataSize);
        }

        mAngles[name+"_sagittal"] = std::deque<double>(mDataSize);
        mAngles[name+"_frontal"] = std::deque<double>(mDataSize);
        mAngles[name+"_transverse"] = std::deque<double>(mDataSize);

        mAnglesByFrame[name+"_sagittal"] = std::vector<std::vector<double>>(mNumFrames);
        mAnglesByFrame[name+"_frontal"] = std::vector<std::vector<double>>(mNumFrames);
        mAnglesByFrame[name+"_transverse"] = std::vector<std::vector<double>>(mNumFrames);
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

    for(auto iter = mAnglesByFrame.begin(); iter != mAnglesByFrame.end(); iter++){
        for(int i = 0; i != (iter->second).size(); i++)
            (iter->second).at(i) = std::vector<double>();
    }
}

void
JointData::
SetTorques(const Eigen::VectorXd& torques)
{
    // if(isFirst == true){
    //     curStep = 0;
    //     isFirst = false;
    // }
    // curStep++;

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

            this->SetTorquesNorm(name, (torques.segment(idx, 6)).norm());
        }
        else if(joint->getType() == "BallJoint")
        {
            this->SetTorques(name+"_x", torques[idx+0]);
            this->SetTorques(name+"_y", torques[idx+1]);
            this->SetTorques(name+"_z", torques[idx+2]);

            this->SetTorquesNorm(name, (torques.segment(idx, 3)).norm());
        }
        else if(joint->getType() == "RevoluteJoint")
        {
            this->SetTorques(name, torques[idx+0]);

            this->SetTorquesNorm(name, fabs(torques[idx]));
        }
        else
        {
        }
    }

    // if(phase*mCycleFrames >= mCycleFrames-1)
    // {
    //     if(isTotalFirst){
    //         isTotalFirst = false;
    //     }
    //     else{
    //               for(int i=0; i<HOUD06_deque.size(); i++)
    //                 HOUD06_cum += HOUD06_deque[i];
    //         }
    //     }

    // for(int i=0; i<mTorquesNorm.size(); i++)
    //     mTorquesNormCum += mTorquesNorm[i];

    // if(curStep == mNumSteps-1)
    // {
    //     double torqueNorm = 0.0;
    //     for(int i=0; i<jointNum; i++)
    //     {
    //         auto joint = mSkeleton->getJoint(i);
    //         int idx = joint->getIndexInSkeleton(0);
    //         std::string name = joint->getName();

    //         double jointTorqueNorm = 0.0;

    //         if(joint->getType() == "FreeJoint")
    //         {
    //         }
    //         else if(joint->getType() == "BallJoint")
    //         {
    //             for(int j=0; j<mNumSteps; j++ ){
    //                 double x = mTorques[name+"_x"].at(j);
    //                 double y = mTorques[name+"_y"].at(j);
    //                 double z = mTorques[name+"_z"].at(j);
    //                 jointTorqueNorm += std::sqrt(x*x + y*y +z*z);
    //             }
    //             torqueNorm += (jointTorqueNorm/(double)mNumSteps);
    //         }
    //         else if(joint->getType() == "RevoluteJoint")
    //         {
    //             this->SetTorques(name, torques[idx+0]);
    //             for(int j=0; j<mNumSteps; j++){
    //                 double t = mTorques[name].at(j);
    //                 jointTorqueNorm += std::sqrt(t*t);
    //             }
    //             torqueNorm += (jointTorqueNorm/(double)mNumSteps);
    //         }
    //         else
    //         {
    //         }
    //     }
    //     mTorqueNorm_deque.pop_back();
    //     mTorqueNorm_deque.push_front(torqueNorm);

    //     if(phase*mCycleFrames >= mCycleFrames-1)
    //     {
    //         if(isTotalFirst){
    //             isTotalFirst = false;
    //         }
    //         else{
    //             for(int i=0; i<mTorqueNorm_deque.size(); i++){
    //                 mTorqueNorm_cum += mTorqueNorm_deque[i];
    //             }
    //         }
    //     }

    //     isFirst = true;
    // }
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
SetTorquesNorm(std::string name, double norm)
{
    mTorquesNorm[name].pop_back();
    mTorquesNorm[name].push_front(norm);
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


double
JointData::
GetReward()
{
    double err_scale = 1.0;
    double torque_scale = 0.01;
    double torque_err = 0.0;

    double reward = 0.0;
    // if(HOUD06_cum != 0.0){
    //     metabolic_err = HOUD06_cum/(double)mCycleFrames;
    //     // reward = 1.0 - exp(-err_scale * metabolic_scale * metabolic_err);
    //     reward = exp(-err_scale * metabolic_scale * metabolic_err);
    //     HOUD06_cum = 0.0;
    //     BHAR04_cum = 0.0;
    // }

    // if(isTotalFirst)
    //     reward = 0;
    return reward;
}

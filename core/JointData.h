#ifndef __JOINT_DATA_H__
#define __JOINT_DATA_H__
#include "dart/dart.hpp"
#include <deque>
#include <map>

using namespace dart::dynamics;
namespace WAD
{

class JointData
{
public:
    JointData();
    ~JointData();

    void Initialize(const SkeletonPtr& skel, int simHz, int conHz);
    void Initialize_Torques();
    void Initialize_Moments();
    void Initialize_Angles();
    void Reset();

    void SetMaxForces(const Eigen::VectorXd& forces){mMaxForces = forces;}

    void SetPhaseState(int stateLeft, Eigen::Vector3d comLeft, int stateRight, Eigen::Vector3d comRight, double time);
    void SetPhaseStateLeft(int phaseState, Eigen::Vector3d com, double time);
    void SetPhaseStateRight(int phaseState, Eigen::Vector3d com, double time);  
    
    void SetTorques(const Eigen::VectorXd& torques);
    void SetTorques(std::string name, double torque);
    void SetTorquesGaitPhase(std::string name, double torque);

    void SetDeviceTorques(const Eigen::VectorXd& torques);
    void SetDeviceTorques(std::string name, double torque);
    void SetDeviceTorquesGaitPhase(std::string name, double torque);

    void SetMoments(const Eigen::VectorXd& moments);
    void SetMoments(std::string name, double moment);
    void SetMomentsGaitPhase(std::string name, double moment);

    void SetAngles();
    void SetAngles(std::string name, double angle);
    void SetAnglesGaitPhase(std::string name, double angle);

    void ChangePhaseTorques();
    void ChangePhaseMoments();

    double GetStrideRight(){ return mStrideRight; }
    double GetStrideLeft(){ return mStrideLeft; }
    double GetCadenceRight(){ return mCadenceRight; }
    double GetCadenceLeft(){ return mCadenceLeft; }

    double GetTorqueEnergyPrev(){ return mCycleTorqueErrPrev;}

    const std::map<std::string, std::deque<double>>& GetTorques(){return mTorques;}
    const std::map<std::string, std::deque<double>>& GetTorquesGaitPhase(){return mTorquesGaitPhase;}
    const std::map<std::string, std::deque<double>>& GetTorquesGaitPhasePrev(){return mTorquesGaitPhasePrev;}

    const std::map<std::string, std::deque<double>>& GetDeviceTorques(){return mDeviceTorques;}
    const std::map<std::string, std::deque<double>>& GetDeviceTorquesGaitPhase(){return mDeviceTorquesGaitPhase;}
    const std::map<std::string, std::deque<double>>& GetDeviceTorquesGaitPhasePrev(){return mDeviceTorquesGaitPhasePrev;}

    const std::map<std::string, std::deque<double>>& GetMoments(){return mMoments;}
    const std::map<std::string, std::deque<double>>& GetMomentsGaitPhase(){return mMomentsGaitPhase;}
    const std::map<std::string, std::deque<double>>& GetMomentsGaitPhasePrev(){return mMomentsGaitPhasePrev;}

    const std::map<std::string, std::deque<double>>& GetAngles(){return mAngles;}
    const std::map<std::string, std::deque<double>>& GetAnglesGaitPhaseLeft(){return mAnglesGaitPhaseLeft;}        
    const std::map<std::string, std::deque<double>>& GetAnglesGaitPhaseLeftPrev(){return mAnglesGaitPhaseLeftPrev;}        
    const std::map<std::string, std::deque<double>>& GetAnglesGaitPhaseRight(){return mAnglesGaitPhaseRight;}        
    const std::map<std::string, std::deque<double>>& GetAnglesGaitPhaseRightPrev(){return mAnglesGaitPhaseRightPrev;}        
    
    double GetReward();

private:
    SkeletonPtr mSkeleton;
    int mDof;
    int mJointNum;
    int mWindowSize;
    int mCycleStep;
   
    bool mOnCycle;
    double mCycleTorqueSum;
    double mCycleTorqueErr;
    double mCycleTorqueErrPrev;
   
    int mSimulationHz;
    int mControlHz;

    int mPhaseStateLeft;
    int mPhaseStateLeftPrev;

    int mPhaseStateRight;
    int mPhaseStateRightPrev;

    double mStrideLeft, mStrideRight;
    double mCadenceLeft, mCadenceRight;

    double mTimeLeft, mTimeLeftPrev;
    double mTimeRight, mTimeRightPrev;
    
    Eigen::Vector3d mComLeft, mComLeftPrev;
    Eigen::Vector3d mComRight, mComRightPrev;

    Eigen::VectorXd mMaxForces;
    std::map<std::string, std::deque<double>> mTorques;
    std::map<std::string, std::deque<double>> mTorquesGaitPhase;
    std::map<std::string, std::deque<double>> mTorquesGaitPhasePrev;

    std::map<std::string, std::deque<double>> mDeviceTorques;
    std::map<std::string, std::deque<double>> mDeviceTorquesGaitPhase;
    std::map<std::string, std::deque<double>> mDeviceTorquesGaitPhasePrev;

    std::map<std::string, std::deque<double>> mMoments;
    std::map<std::string, std::deque<double>> mMomentsGaitPhase;
    std::map<std::string, std::deque<double>> mMomentsGaitPhasePrev;

    std::map<std::string, std::deque<double>> mAngles;
    std::map<std::string, std::deque<double>> mAnglesGaitPhaseLeft;
    std::map<std::string, std::deque<double>> mAnglesGaitPhaseLeftPrev;
    std::map<std::string, std::deque<double>> mAnglesStancePhaseLeftPrev;    
    std::map<std::string, std::deque<double>> mAnglesSwingPhaseLeftPrev;    
    std::map<std::string, std::deque<double>> mAnglesGaitPhaseRight;
    std::map<std::string, std::deque<double>> mAnglesGaitPhaseRightPrev;
    // std::map<std::string, std::deque<double>> mAnglesGaitPhaseTmpRight;
    std::map<std::string, std::deque<double>> mAnglesStancePhaseRightPrev;    
    std::map<std::string, std::deque<double>> mAnglesSwingPhaseRightPrev;    
};

}
#endif

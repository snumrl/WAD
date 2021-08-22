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

    void Initialize(const SkeletonPtr& skel);
    void Initialize_Torques();
    void Initialize_Angles();
    void Reset();

    void SetMaxForces(const Eigen::VectorXd& forces){mMaxForces = forces;}

    void SetPhaseState(int stateLeft, int stateRight, double time);
    void SetPhaseStateLeft(int phaseState);
    void SetPhaseStateRight(int phaseState);  
    
    void SetTorques(const Eigen::VectorXd& torques);
    void SetTorques(std::string name, double torque);
    void SetTorquesGaitPhase(std::string name, double torque);

    void SetAngles();
    void SetAngles(std::string name, double angle);
    void SetAnglesGaitPhase(std::string name, double angle);

    void ChangePhaseTorques();

    const std::map<std::string, std::deque<double>>& GetTorques(){return mTorques;}
    const std::map<std::string, std::deque<double>>& GetTorquesGaitPhase(){return mTorquesGaitPhase;}
    const std::map<std::string, std::deque<double>>& GetTorquesGaitPhasePrev(){return mTorquesGaitPhasePrev;}

    const std::map<std::string, std::deque<double>>& GetAngles(){return mAngles;}
    const std::map<std::string, std::deque<double>>& GetAnglesGaitPhaseLeft(){return mAnglesGaitPhaseLeft;}        
    const std::map<std::string, std::deque<double>>& GetAnglesStancePhaseLeftPrev(){return mAnglesStancePhaseLeftPrev;}        
    const std::map<std::string, std::deque<double>>& GetAnglesSwingPhaseLeftPrev(){return mAnglesSwingPhaseLeftPrev;}        
    const std::map<std::string, std::deque<double>>& GetAnglesGaitPhaseRight(){return mAnglesGaitPhaseRight;}        
    const std::map<std::string, std::deque<double>>& GetAnglesStancePhaseRightPrev(){return mAnglesStancePhaseRightPrev;}        
    const std::map<std::string, std::deque<double>>& GetAnglesSwingPhaseRightPrev(){return mAnglesSwingPhaseRightPrev;}        

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
   
    int mPhaseStateLeft;
    int mPhaseStateLeftPrev;

    int mPhaseStateRight;
    int mPhaseStateRightPrev;
    
    Eigen::VectorXd mMaxForces;
    std::map<std::string, std::deque<double>> mTorques;
    std::map<std::string, std::deque<double>> mTorquesGaitPhase;
    std::map<std::string, std::deque<double>> mTorquesGaitPhasePrev;

    std::map<std::string, std::deque<double>> mAngles;
    std::map<std::string, std::deque<double>> mAnglesGaitPhaseLeft;
    std::map<std::string, std::deque<double>> mAnglesStancePhaseLeftPrev;    
    std::map<std::string, std::deque<double>> mAnglesSwingPhaseLeftPrev;    
    std::map<std::string, std::deque<double>> mAnglesGaitPhaseRight;
    std::map<std::string, std::deque<double>> mAnglesStancePhaseRightPrev;    
    std::map<std::string, std::deque<double>> mAnglesSwingPhaseRightPrev;    
};

}
#endif

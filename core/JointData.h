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
    void Reset();

    void SetMaxForces(const Eigen::VectorXd& forces){mMaxForces = forces;}

    void SetPhaseState(int phaseState, double time);

    void SetTorques(const Eigen::VectorXd& torques);
    void SetTorques(std::string name, double torque);
    void SetTorquesGaitPhase(std::string name, double torque);

    void SetAngles();
    void SetAngles(std::string name, double angle);
    void SetAnglesGaitPhase(std::string name, double angle);

    void SetAnglesRef();
    void SetAnglesRef(std::string name, double angle);
    void SetAnglesGaitPhaseRef(std::string name, double angle);

    const std::map<std::string, std::deque<double>>& GetTorques(){return mTorques;}
    const std::map<std::string, std::deque<double>>& GetTorquesGaitPhasePrev(){return mTorquesGaitPhasePrev;}
    const std::map<std::string, std::deque<double>>& GetAngles(){return mAngles;}
    const std::map<std::string, std::deque<double>>& GetAnglesGaitPhasePrev(){return mAnglesGaitPhasePrev;}    
    const std::map<std::string, std::deque<double>>& GetAnglesGaitPhaseRefPrev(){return mAnglesGaitPhaseRefPrev;}        

    double GetReward();
private:
    SkeletonPtr mSkeleton;
    int mDof;
    int mJointNum;
    int mWindowSize;
    int mCycleStep;
   
    bool mOnlyLowerBody;
    bool mOnCycle;
    double mCycleTorqueSum;
    double mCycleTorqueErr;
   
    int mPhaseStateRight;
    int mPhaseStateRightPrev;

    Eigen::VectorXd mMaxForces;

    std::map<std::string, std::deque<double>> mTorques;
    std::map<std::string, std::deque<double>> mTorquesGaitPhase;
    std::map<std::string, std::deque<double>> mTorquesGaitPhasePrev;

    std::map<std::string, std::deque<double>> mAngles;
    std::map<std::string, std::deque<double>> mAnglesGaitPhase;
    std::map<std::string, std::deque<double>> mAnglesGaitPhasePrev;
    std::map<std::string, std::deque<double>> mAnglesGaitPhaseRef;
    std::map<std::string, std::deque<double>> mAnglesGaitPhaseRefPrev;
};

}
#endif

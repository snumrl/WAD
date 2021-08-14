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

    void SetTorques(const Eigen::VectorXd& torques);
    void SetTorques(const Eigen::VectorXd& torques, double phase, double frame);

    void SetTorques(std::string name, double torque, double phase);
    void SetTorquesNorm(std::string name, double norm, double phase);

    void SetMaxForces(const Eigen::VectorXd& forces){mMaxForces = forces;}

    void SetAngles(double phase);
    void SetAngles(std::string name, double angle, double phase);

    const std::map<std::string, std::deque<double>>& GetTorques(){return mTorques;}
    const std::map<std::string, std::deque<double>>& GetTorquesNorm(){return mTorquesNorm;}
    const std::map<std::string, std::deque<double>>& GetAngles(){return mAngles;}

    const std::map<std::string, std::deque<std::pair<double, double>>>& GetTorquesPhase(){ return mTorquesPhase;}
    const std::map<std::string, std::deque<std::pair<double, double>>>& GetTorquesNormPhase(){ return mTorquesNormPhase;}
    const std::map<std::string, std::deque<std::pair<double, double>>>& GetAnglesPhase(){ return mAnglesPhase;}

    double GetReward();
private:
    SkeletonPtr mSkeleton;

    int mWindowSize;
    int mCycleStep;
    int mJointNum;
    int mDof;
    double mPhasePrev;
    double mCycleTorqueSum;
    double mCycleTorqueErr;
    bool mOnCycle;
    bool mOnlyLowerBody;

    Eigen::VectorXd mMaxForces;

    std::map<std::string, std::deque<double>> mTorques;
    std::map<std::string, std::deque<double>> mTorquesNorm;
    std::map<std::string, std::deque<double>> mAngles;

    std::map<std::string, std::deque<std::pair<double, double>>> mTorquesPhase;
    std::map<std::string, std::deque<std::pair<double, double>>> mTorquesNormPhase;
    std::map<std::string, std::deque<std::pair<double, double>>> mAnglesPhase;

};

}
#endif

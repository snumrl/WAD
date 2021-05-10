#ifndef __MASS_JOINT_DATA_H__
#define __MASS_JOINT_DATA_H__
#include "dart/dart.hpp"
#include <deque>
#include <map>

namespace MASS
{

class JointData
{
public:
    JointData();
    ~JointData();

    void Initialize(const dart::dynamics::SkeletonPtr& skel);
    void Reset();
    void SetTorques(const Eigen::VectorXd& torques);
    void SetTorques(const Eigen::VectorXd& torques, double phase, int frame);
    void SetTorques(std::string name, double torque);
    void SetTorquesNorm(std::string name, double norm);
    const std::map<std::string, std::deque<double>>& GetTorques(){return mTorques;}

    void SetAngles(int frame);
    // void SetAngles(std::string name, double angle);
    void SetAngles(std::string name, double angle, int frame);
    const std::map<std::string, std::deque<double>>& GetAngles(){return mAngles;}
    const std::map<std::string, std::vector<std::vector<double>>>& GetAnglesByFrame(){return mAnglesByFrame;}

    std::deque<double> GetTorquesNormCycle(){return mTorquesNormCycle;}
    std::deque<int> GetTorquesFrame(){return mTorquesFrame;}

    double GetReward();
private:
    dart::dynamics::SkeletonPtr mSkeleton;

    int mWindowSize;
    int mDataSize;
    int mNumFrames;
    int mNumSteps;
    int mCycleFrames;
    int curStep;

    bool isFirst;
    bool isTotalFirst;

    double mTorquesNormCum;

    std::map<std::string, std::deque<double>> mTorques;
    std::map<std::string, std::deque<double>> mAngles;
    std::map<std::string, std::deque<double>> mTorquesNorm;
    std::deque<double> mTorquesNormCycle;
    std::deque<int> mTorquesFrame;

    std::map<std::string, std::vector<std::vector<double>>> mAnglesByFrame;
};

};
#endif

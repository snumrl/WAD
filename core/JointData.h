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
    void SetTorques(std::string name, double torque);
    const std::map<std::string, std::deque<double>>& GetTorques(){return mTorques;}

    void SetAngles(const Eigen::VectorXd& angles);
private:
    dart::dynamics::SkeletonPtr mSkeleton;

    int windowSize;
    std::map<std::string, std::deque<double>> mTorques;
    std::map<std::string, std::deque<double>> mAngles;
};

};
#endif

#ifndef __MASS_JOINT_TORQUE_H__
#define __MASS_JOINT_TORQUE_H__
#include "dart/dart.hpp"
#include <deque>
#include <map>

namespace MASS
{

class JointTorque
{
public:
    JointTorque();
    ~JointTorque();

    void Initialize(const dart::dynamics::SkeletonPtr& skel);
    void Reset();
    void Set(const dart::dynamics::SkeletonPtr& skel, const Eigen::VectorXd& torques);
    void Set(std::string name, double torque);
    const std::map<std::string, std::deque<double>>& Get(){return mJointTorques;}
private:
    int windowSize;
    std::map<std::string, std::deque<double>> mJointTorques;
};

};
#endif

#ifndef __UTILS_H__
#define __UTILS_H__
#include "dart/dart.hpp"

using namespace dart::dynamics;
namespace WAD
{

namespace Utils
{
    double exp_of_squared(const Eigen::VectorXd& vec,double w);
    double exp_of_squared(const Eigen::Vector3d& vec,double w);
    double exp_of_squared(double val,double w);
    double exp_of_sum(const Eigen::VectorXd& vec,double w);
    double exp_of_sum(const Eigen::Vector3d& vec,double w);
    double exp_of_sum(double val,double w);

    double Pulse_Constant(double t);
    double Pulse_Linear(double t);
    double Pulse_Period(double t);
    double Pulse_Period(double t, double offset);

    double QuatDotQuat(const Eigen::Quaterniond& _a, const Eigen::Quaterniond& _b);

    Eigen::Quaterniond Slerp(const Eigen::Quaterniond& _a, const Eigen::Quaterniond& _b, double interp);
    Eigen::Quaterniond GetQuaternionSlerp(const Eigen::Quaterniond& _a, const Eigen::Quaterniond& _b, double interp);
    Eigen::Vector3d GetQuaternionSlerp(const Eigen::Vector3d& _a, const Eigen::Vector3d& _b, double interp);
    Eigen::VectorXd GetPoseSlerp(const SkeletonPtr& skeleton, double ratio, const Eigen::VectorXd& p1, const Eigen::VectorXd& p2);

    Eigen::Vector4d GetPoint4d(Eigen::Vector3d v);
    Eigen::Vector4d GetVector4d(Eigen::Vector3d v);
    Eigen::Vector3d GetVector3d(Eigen::Vector4d v);
    Eigen::Vector3d AffineTransPoint(Eigen::Isometry3d t,Eigen::Vector3d p);
    Eigen::Vector3d AffineTransVector(Eigen::Isometry3d t,Eigen::Vector3d v);

    Eigen::Isometry3d GetOriginTrans(const SkeletonPtr& skeleton);
    Eigen::Isometry3d GetJointTransform(const BodyNode* body);
    Eigen::Isometry3d GetBodyTransform(const BodyNode* body);

    int Clamp(int val, int min, int max);
    void Clamp(Eigen::VectorXd min, Eigen::VectorXd max, Eigen::VectorXd& out_vec);
    double Clamp(double val, double min, double max);
    double NormalizeAngle(double theta);
    Eigen::Matrix4d TranslateMat(const Eigen::Vector3d& trans);
    Eigen::Matrix3d ScaleMat(double scale);
    Eigen::Matrix3d ScaleMat(const Eigen::Vector3d& scale);
    Eigen::Matrix3d RotateMat(const Eigen::Vector3d& euler); // euler angles order rot(Z) * rot(Y) * rot(X)
    Eigen::Matrix3d RotateMat(const Eigen::Vector3d& axis, double theta);
    Eigen::Matrix3d RotateMat(const Eigen::Quaterniond& q);

    Eigen::Matrix3d InvRigidMat(const Eigen::Matrix3d& mat);
    Eigen::Vector3d GetRigidTrans(const Eigen::Matrix3d& mat);
    Eigen::Vector3d InvEuler(const Eigen::Vector3d& euler);
    void RotMatToAxisAngle(const Eigen::Matrix3d& mat, Eigen::Vector3d& out_axis, double& out_theta);
    Eigen::Vector3d RotMatToEuler(const Eigen::Matrix3d& mat);
    Eigen::Quaterniond RotMatToQuaternion(const Eigen::Matrix3d& mat);
    void EulerToAxisAngle(const Eigen::Vector3d& euler, Eigen::Vector3d& out_axis, double& out_theta);
    Eigen::Vector3d AxisAngleToEuler(const Eigen::Vector3d& axis, double theta);
    Eigen::Quaterniond EulerToQuaternion(const Eigen::Vector3d& euler);
    Eigen::Vector3d QuaternionToEuler(const Eigen::Quaterniond& q);
    Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::Vector3d& in);
    Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::Vector3d& axis, double theta);
    void QuaternionToAxisAngle(const Eigen::Quaterniond& q, Eigen::Vector3d& out_axis, double& out_theta);
    Eigen::Vector3d QuaternionToAxisAngle(const Eigen::Quaterniond& q);
    Eigen::Matrix3d BuildQuaternionDiffMat(const Eigen::Quaterniond& q);
    Eigen::Vector3d CalcQuaternionVel(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1, double dt);
    Eigen::Vector3d CalcQuaternionVel(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, double dt);
    Eigen::Vector3d CalcQuaternionVelRel(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1, double dt);
    Eigen::Vector3d CalcQuaternionVelRel(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, double dt);
    Eigen::Quaterniond VecToQuat(const Eigen::Vector4d& v);
    Eigen::Vector4d QuatToVec(const Eigen::Quaterniond& q);
    Eigen::Quaterniond QuatDiff(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1);
    double QuatDiffTheta(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1);
    double QuatTheta(const Eigen::Quaterniond& dq);
    void QuatNormalize(Eigen::Quaterniond& in);
    void ButterworthFilter(double dt, double cutoff, Eigen::VectorXd& out_x);

    Eigen::Vector3d projectToXZ(const Eigen::Vector3d& v);
    Eigen::Vector3d NearestOnGeodesicCurve3d(const Eigen::Vector3d& targetAxis, const Eigen::Vector3d& targetPosition, const Eigen::Vector3d& position);
    Eigen::VectorXd BlendPosition(const Eigen::VectorXd& target_a, const Eigen::VectorXd& target_b, double weight, bool blend_rootpos);
    Eigen::VectorXd solveMCIKRoot(const SkeletonPtr& skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints);
}

}
#endif

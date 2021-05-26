#ifndef __DART_HELPER_H__
#define __DART_HELPER_H__
#include "dart/dart.hpp"

namespace Eigen {
	using Vector1d = Matrix<double, 1, 1>;
	using Matrix1d = Matrix<double, 1, 1>;
}

std::vector<double> split_to_double(const std::string& input, int num);
Eigen::Vector1d string_to_vector1d(const std::string& input);
Eigen::Vector3d string_to_vector3d(const std::string& input);
Eigen::Vector4d string_to_vector4d(const std::string& input);
Eigen::VectorXd string_to_vectorXd(const std::string& input, int n);
Eigen::Matrix3d string_to_matrix3d(const std::string& input);

using namespace dart::dynamics;
namespace MASS
{
	ShapePtr MakeSphereShape(double radius);
	ShapePtr MakeCapsuleShape(double radius, double height);
	ShapePtr MakeBoxShape(const Eigen::Vector3d& size);

	Inertia MakeInertia(const ShapePtr& shape,double mass);

	FreeJoint::Properties* MakeFreeJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());

	PlanarJoint::Properties* MakePlanarJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());

	BallJoint::Properties* MakeBallJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Vector3d& lower = Eigen::Vector3d::Constant(-2.0),const Eigen::Vector3d& upper = Eigen::Vector3d::Constant(2.0),const Eigen::Vector3d& force_lower = Eigen::Vector3d::Constant(-1000.0),const Eigen::Vector3d& force_upper = Eigen::Vector3d::Constant(1000.0),const std::string& actuator_type="FORCE");

	RevoluteJoint::Properties* MakeRevoluteJointProperties(const std::string& name,const Eigen::Vector3d& axis,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Vector1d& lower = Eigen::Vector1d::Constant(-2.0),const Eigen::Vector1d& upper = Eigen::Vector1d::Constant(2.0),const Eigen::Vector1d& force_lower = Eigen::Vector1d::Constant(-1000.0),const Eigen::Vector1d& force_upper = Eigen::Vector1d::Constant(1000.0),const std::string& actuator_type="FORCE");

	WeldJoint::Properties* MakeWeldJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());

	BodyNode* MakeBodyNode(const SkeletonPtr& skeleton,BodyNode* parent,Joint::Properties* joint_properties,const std::string& joint_type,Inertia inertia);

	SkeletonPtr BuildFromFile(const std::string& path, bool load_obj = false, double mass_ratio=1.0);

}
#endif

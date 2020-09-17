#include "Utils.h"

namespace MASS
{
namespace Utils
{

double exp_of_squared(const Eigen::VectorXd& vec,double w)
{
    return exp(-w*vec.squaredNorm());
}

double exp_of_squared(const Eigen::Vector3d& vec,double w)
{
    return exp(-w*vec.squaredNorm());
}

double exp_of_squared(double val,double w)
{
    return exp(-w*val*val);
}

double exp_of_sum(const Eigen::VectorXd& vec,double w)
{
    double sum = 0;
    for(int i=0; i<vec.size(); i++)
        sum += vec[i];
    return exp(-w*sum);
}

double exp_of_sum(const Eigen::Vector3d& vec,double w)
{
    double sum = 0;
    for(int i=0; i<vec.size(); i++)
        sum += vec[i];
    return exp(-w*sum);
}

double exp_of_sum(double val,double w)
{
    return exp(-w*val);
}

double Pulse_Constant(double t)
{
    double ratio = 1.0;
    return ratio;
}

double Pulse_Linear(double t)
{
    double ratio = 1.0;
    if(t <= 0.5)
        ratio = t/0.5;
    else
        ratio = (1.0-t)/0.5;
    return ratio;
}

double Pulse_Period(double t)
{
    double ratio = 1.0;
    if(t <= 0.5)
        ratio = 1.0;
    else
        ratio = 0.0;
    return ratio;
}

double Pulse_Period(double t, double offset)
{
    double ratio = 1.0;
    if(t <= offset)
        ratio = 1.0;
    else
        ratio = 0.0;
    return ratio;
}

double QuatDotQuat(const Eigen::Quaterniond& _a, const Eigen::Quaterniond& _b)
{
    return _a.w()*_b.w() + _a.x()*_b.x() + _a.y()*_b.y() + _a.z()*_b.z();
}

Eigen::Quaterniond Slerp(const Eigen::Quaterniond& _a, const Eigen::Quaterniond& _b, double interp)
{
    Eigen::Quaterniond a = _a;
    Eigen::Quaterniond b = _b;

    double d = QuatDotQuat(a, b);
    if (d < 0.0)
        QuatNormalize(a);
    if (d >= 1.0)
        return a;

    float theta = acosf(d);
    if (theta == 0.0f)
        return a;

    float coeff_a = (sinf(theta - interp * theta) / sinf(theta));
    float coeff_b = (sinf(interp * theta) / sinf(theta));

    return VecToQuat(QuatToVec(a)*coeff_a + QuatToVec(b)*coeff_b);
}

Eigen::Quaterniond GetQuaternionSlerp(const Eigen::Quaterniond& _a, const Eigen::Quaterniond& _b, double interp)
{
    return Slerp(_a, _b, interp);
}

Eigen::Vector3d GetQuaternionSlerp(const Eigen::Vector3d& _a, const Eigen::Vector3d& _b, double interp)
{
    Eigen::Quaterniond q_a = Utils::AxisAngleToQuaternion(_a);
    Eigen::Quaterniond q_b = Utils::AxisAngleToQuaternion(_b);
    return QuaternionToAxisAngle(Slerp(q_a, q_b, interp));
}

Eigen::Vector4d GetPoint4d(Eigen::Vector3d v)
{
    Eigen::Vector4d p;
    p.segment(0,3) = v;
    p[3] = 1.0;

    return p;
}

Eigen::Vector4d GetVector4d(Eigen::Vector3d v)
{
    Eigen::Vector4d p;
    p.segment(0,3) = v;
    p[3] = 0.0;

    return p;
}

Eigen::Vector3d GetVector3d(Eigen::Vector4d v)
{
    Eigen::Vector3d p;
    p = v.segment(0,3);
    return p;
}

Eigen::Vector3d AffineTransPoint(Eigen::Isometry3d t, Eigen::Vector3d p)
{
    Eigen::Vector4d p4 = GetPoint4d(p);
    p4 = t * p4;
    p = p4.segment(0,3);
    return p;
}

Eigen::Vector3d AffineTransVector(Eigen::Isometry3d t, Eigen::Vector3d v)
{
    Eigen::Vector4d v4 = GetVector4d(v);
    v4 = t * v4;
    v = v4.segment(0,3);
    return v;
}

Eigen::Isometry3d GetOriginTrans(const dart::dynamics::SkeletonPtr& skeleton)
{
    dart::dynamics::BodyNode* root = skeleton->getBodyNode(0);
    Eigen::Vector3d origin = root->getTransform().translation();
    origin[1] = 0;

    Eigen::Vector3d ref_dir(1.0, 0.0, 0.0);
    Eigen::Quaterniond root_rot(root->getTransform().rotation());
    Eigen::Vector3d rot_dir = root_rot * ref_dir;
    double heading = std::atan2(-rot_dir[2], rot_dir[0]);

    Eigen::Vector3d axis(0.0, 1.0, 0.0);

    double c = std::cos(-heading);
    double s = std::sin(-heading);
    double x = axis[0];
    double y = axis[1];
    double z = axis[2];

    Eigen::Isometry3d mat;

    Eigen::Matrix3d mat_rotation;
    mat_rotation
        << c+x*x*(1 - c), x*y*(1 - c) - z*s, x*z*(1 - c) + y*s,
            y*x*(1 - c) + z*s, c + y*y*(1 - c), y*z*(1 - c) - x*s,
            z*x*(1 - c) - y*s,  z*y*(1 - c) + x*s,  c + z*z*(1 - c);

    Eigen::Isometry3d mat1;
    mat1.linear() = mat_rotation;
    mat1.translation() = Eigen::Vector3d::Zero();

    Eigen::Isometry3d mat2;
    mat2.linear() = Eigen::Matrix3d::Identity();
    mat2.translation() = -origin;

    mat = mat1 * mat2;

    return mat;
}

Eigen::Isometry3d GetJointTransform(dart::dynamics::BodyNode* body)
{
    Eigen::Isometry3d transform = body->getTransform()*body->getParentJoint()->getJointProperties().mT_ChildBodyToJoint;

    return transform;
}

Eigen::Isometry3d GetBodyTransform(dart::dynamics::BodyNode* body)
{
    Eigen::Isometry3d transform = body->getTransform();

    return transform;
}

int Clamp(int val, int min, int max)
{
    return std::max(min, std::min(val, max));
}

void Clamp(Eigen::VectorXd min, Eigen::VectorXd max, Eigen::VectorXd& out_vec)
{
    out_vec = out_vec.cwiseMin(max).cwiseMax(min);
}

double Clamp(double val, double min, double max)
{
    return std::max(min, std::min(val, max));
}

double NormalizeAngle(double theta)
{
    // normalizes theta to be between [-pi, pi]
    double norm_theta = fmod(theta, 2 * M_PI);
    if (norm_theta > M_PI)
    {
        norm_theta = -2 * M_PI + norm_theta;
    }
    else if (norm_theta < -M_PI)
    {
        norm_theta = 2 * M_PI + norm_theta;
    }
    return norm_theta;
}


Eigen::Matrix4d TranslateMat(const Eigen::Vector3d& trans)
{
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    mat(0, 3) = trans[0];
    mat(1, 3) = trans[1];
    mat(2, 3) = trans[2];
    return mat;
}

Eigen::Matrix3d RotateMat(const Eigen::Vector3d& euler)
{
    double x = euler[0];
    double y = euler[1];
    double z = euler[2];

    double x_s = std::sin(x);
    double x_c = std::cos(x);
    double y_s = std::sin(y);
    double y_c = std::cos(y);
    double z_s = std::sin(z);
    double z_c = std::cos(z);

    Eigen::Matrix3d mat = Eigen::Matrix3d::Identity();
    mat(0, 0) = y_c * z_c;
    mat(1, 0) = y_c * z_s;
    mat(2, 0) = -y_s;

    mat(0, 1) = x_s * y_s * z_c - x_c * z_s;
    mat(1, 1) = x_s * y_s * z_s + x_c * z_c;
    mat(2, 1) = x_s * y_c;

    mat(0, 2) = x_c * y_s * z_c + x_s * z_s;
    mat(1, 2) = x_c * y_s * z_s - x_s * z_c;
    mat(2, 2) = x_c * y_c;

    return mat;
}

Eigen::Matrix3d RotateMat(const Eigen::Vector3d& axis, double theta)
{
    assert(std::abs(axis.squaredNorm() - 1) < 0.0001);

    double c = std::cos(theta);
    double s = std::sin(theta);
    double x = axis[0];
    double y = axis[1];
    double z = axis[2];

    Eigen::Matrix3d mat;
    mat <<  c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s,
            y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s,
            z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c);

    return mat;
}

Eigen::Matrix3d RotateMat(const Eigen::Quaterniond& q)
{
    Eigen::Matrix3d mat = Eigen::Matrix3d::Identity();

    double sqw = q.w() * q.w();
    double sqx = q.x()*  q.x();
    double sqy = q.y() * q.y();
    double sqz = q.z() * q.z();
    double invs = 1 / (sqx + sqy + sqz + sqw);

    mat(0, 0) = (sqx - sqy - sqz + sqw) * invs;
    mat(1, 1) = (-sqx + sqy - sqz + sqw) * invs;
    mat(2, 2) = (-sqx - sqy + sqz + sqw) * invs;

    double tmp1 = q.x()*q.y();
    double tmp2 = q.z()*q.w();
    mat(1, 0) = 2.0 * (tmp1 + tmp2) * invs;
    mat(0, 1) = 2.0 * (tmp1 - tmp2) * invs;

    tmp1 = q.x()*q.z();
    tmp2 = q.y()*q.w();
    mat(2, 0) = 2.0 * (tmp1 - tmp2) * invs;
    mat(0, 2) = 2.0 * (tmp1 + tmp2) * invs;

    tmp1 = q.y()*q.z();
    tmp2 = q.x()*q.w();
    mat(2, 1) = 2.0 * (tmp1 + tmp2) * invs;
    mat(1, 2) = 2.0 * (tmp1 - tmp2) * invs;

    return mat;
}

Eigen::Matrix4d InvRigidMat(const Eigen::Matrix4d& mat)
{
    Eigen::Matrix4d inv_mat = Eigen::Matrix4d::Zero();
    inv_mat.block(0, 0, 3, 3) = mat.block(0, 0, 3, 3).transpose();
    inv_mat.col(3) = -inv_mat * mat.col(3);
    inv_mat(3, 3) = 1;
    return inv_mat;
}

Eigen::Vector3d GetRigidTrans(const Eigen::Matrix4d& mat)
{
    return Eigen::Vector3d(mat(0, 3), mat(1, 3), mat(2, 3));
}

Eigen::Vector3d InvEuler(const Eigen::Vector3d& euler)
{
    Eigen::Matrix3d inv_mat = RotateMat(Eigen::Vector3d(1, 0, 0), -euler[0])
                    * RotateMat(Eigen::Vector3d(0, 1, 0), -euler[1])
                    * RotateMat(Eigen::Vector3d(0, 0, 1), -euler[2]);
    Eigen::Vector3d inv_euler = RotMatToEuler(inv_mat);
    return inv_euler;
}

void RotMatToAxisAngle(const Eigen::Matrix3d& mat, Eigen::Vector3d& out_axis, double& out_theta)
{
    double c = (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1) * 0.5;
    c = Clamp(c, -1.0, 1.0);

    out_theta = std::acos(c);
    if (std::abs(out_theta) < 0.00001)
    {
        out_axis = Eigen::Vector3d(0, 0, 1);
    }
    else
    {
        double m21 = mat(2, 1) - mat(1, 2);
        double m02 = mat(0, 2) - mat(2, 0);
        double m10 = mat(1, 0) - mat(0, 1);
        double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
        out_axis[0] = m21 / denom;
        out_axis[1] = m02 / denom;
        out_axis[2] = m10 / denom;
    }
}

Eigen::Vector3d RotMatToEuler(const Eigen::Matrix3d& mat)
{
    Eigen::Vector3d euler;
    euler[0] = std::atan2(mat(2, 1), mat(2, 2));
    euler[1] = std::atan2(-mat(2, 0), std::sqrt(mat(2, 1) * mat(2, 1) + mat(2, 2) * mat(2, 2)));
    euler[2] = std::atan2(mat(1, 0), mat(0, 0));
    euler[3] = 0;
    return euler;
}

Eigen::Quaterniond RotMatToQuaternion(const Eigen::Matrix3d& mat)
{
    double tr = mat(0, 0) + mat(1, 1) + mat(2, 2);
    Eigen::Quaterniond q;

    if (tr > 0) {
        double S = sqrt(tr + 1.0) * 2; // S=4*qw
        q.w() = 0.25 * S;
        q.x() = (mat(2, 1) - mat(1, 2)) / S;
        q.y() = (mat(0, 2) - mat(2, 0)) / S;
        q.z() = (mat(1, 0) - mat(0, 1)) / S;
    }
    else if ((mat(0, 0) > mat(1, 1) && (mat(0, 0) > mat(2, 2)))) {
        double S = sqrt(1.0 + mat(0, 0) - mat(1, 1) - mat(2, 2)) * 2; // S=4*qx
        q.w() = (mat(2, 1) - mat(1, 2)) / S;
        q.x() = 0.25 * S;
        q.y() = (mat(0, 1) + mat(1, 0)) / S;
        q.z() = (mat(0, 2) + mat(2, 0)) / S;
    }
    else if (mat(1, 1) > mat(2, 2)) {
        double S = sqrt(1.0 + mat(1, 1) - mat(0, 0) - mat(2, 2)) * 2; // S=4*qy
        q.w() = (mat(0, 2) - mat(2, 0)) / S;
        q.x() = (mat(0, 1) + mat(1, 0)) / S;
        q.y() = 0.25 * S;
        q.z() = (mat(1, 2) + mat(2, 1)) / S;
    }
    else {
        double S = sqrt(1.0 + mat(2, 2) - mat(0, 0) - mat(1, 1)) * 2; // S=4*qz
        q.w() = (mat(1, 0) - mat(0, 1)) / S;
        q.x() = (mat(0, 2) + mat(2, 0)) / S;
        q.y() = (mat(1, 2) + mat(2, 1)) / S;
        q.z() = 0.25 * S;
    }

    return q;
}

void EulerToAxisAngle(const Eigen::Vector3d& euler, Eigen::Vector3d& out_axis, double& out_theta)
{
    double x = euler[0];
    double y = euler[1];
    double z = euler[2];

    double x_s = std::sin(x);
    double x_c = std::cos(x);
    double y_s = std::sin(y);
    double y_c = std::cos(y);
    double z_s = std::sin(z);
    double z_c = std::cos(z);

    double c = (y_c * z_c + x_s * y_s * z_s + x_c * z_c + x_c * y_c - 1) * 0.5;
    c = Clamp(c, -1.0, 1.0);

    out_theta = std::acos(c);
    if (std::abs(out_theta) < 0.00001)
    {
        out_axis = Eigen::Vector3d(0, 0, 1);
    }
    else
    {
        double m21 = x_s * y_c - x_c * y_s * z_s + x_s * z_c;
        double m02 = x_c * y_s * z_c + x_s * z_s + y_s;
        double m10 = y_c * z_s - x_s * y_s * z_c + x_c * z_s;
        double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
        out_axis[0] = m21 / denom;
        out_axis[1] = m02 / denom;
        out_axis[2] = m10 / denom;
    }
}

Eigen::Vector3d AxisAngleToEuler(const Eigen::Vector3d& axis, double theta)
{
    Eigen::Quaterniond q = AxisAngleToQuaternion(axis, theta);
    return QuaternionToEuler(q);
}

Eigen::Quaterniond EulerToQuaternion(const Eigen::Vector3d& euler)
{
    Eigen::Vector3d axis;
    double theta;
    EulerToAxisAngle(euler, axis, theta);
    return AxisAngleToQuaternion(axis, theta);
}

Eigen::Vector3d QuaternionToEuler(const Eigen::Quaterniond& q)
{
    double sinr = 2.0 * (q.w() * q.x() + q.y() * q.z());
    double cosr = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
    double x = std::atan2(sinr, cosr);

    double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
    double y = 0;
    if (fabs(sinp) >= 1)
    {
        y = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    }
    else
    {
        y = asin(sinp);
    }

    double siny = 2.0 * (q.w() * q.z() + q.x() * q.y());
    double cosy = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    double z = std::atan2(siny, cosy);

    return Eigen::Vector3d(x, y, z);
}

Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::Vector3d& in)
{
    if( in.norm() < 1e-8 ){
        return Eigen::Quaterniond::Identity();
    }
    Eigen::AngleAxisd aa(in.norm(), in.normalized());
    Eigen::Quaterniond q(aa);
    QuatNormalize(q);
    return q;
}

Eigen::Quaterniond AxisAngleToQuaternion(const Eigen::Vector3d& axis, double theta)
{
    // axis must be normalized
    double c = std::cos(theta / 2);
    double s = std::sin(theta / 2);
    Eigen::Quaterniond q;
    q.w() = c;
    q.x() = s * axis[0];
    q.y() = s * axis[1];
    q.z() = s * axis[2];
    return q;
}

void QuaternionToAxisAngle(const Eigen::Quaterniond& q, Eigen::Vector3d& out_axis, double& out_theta)
{
    out_theta = 0;
    out_axis = Eigen::Vector3d(0, 0, 1);

    Eigen::Quaterniond q1 = q;
    if (q1.w() > 1)
    {
        q1.normalize();
    }

    double sin_theta = std::sqrt(1 - q1.w() * q1.w());
    if (sin_theta > 0.000001)
    {
        out_theta = 2 * std::acos(q1.w());
        out_theta = NormalizeAngle(out_theta);
        out_axis = Eigen::Vector3d(q1.x(), q1.y(), q1.z()) / sin_theta;
    }
}

Eigen::Vector3d QuaternionToAxisAngle(const Eigen::Quaterniond& q)
{
    double out_theta = 0;
    Eigen::Vector3d out_axis = Eigen::Vector3d(0, 0, 1);

    Eigen::Quaterniond q1 = q;
    QuatNormalize(q1);
    if (q1.w() > 1)
    {
        q1.normalize();
    }

    double sin_theta = std::sqrt(1 - q1.w() * q1.w());
    if (sin_theta > 0.000001)
    {
        out_theta = 2 * std::acos(q1.w());
        out_theta = NormalizeAngle(out_theta);
        out_axis = Eigen::Vector3d(q1.x(), q1.y(), q1.z()) / sin_theta;
    }

    return out_theta*out_axis;
}

Eigen::Vector3d CalcQuaternionVel(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1, double dt)
{
    Eigen::Quaterniond q_diff = QuatDiff(q0, q1);
    Eigen::Vector3d axis;
    double theta;
    QuaternionToAxisAngle(q_diff, axis, theta);
    return (theta / dt) * axis;
}

Eigen::Vector3d CalcQuaternionVel(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, double dt)
{
    Eigen::AngleAxisd aa0(v0.norm(), v0.normalized());
    Eigen::AngleAxisd aa1(v1.norm(), v1.normalized());

    Eigen::Quaterniond q0(aa0);
    Eigen::Quaterniond q1(aa1);

    Eigen::Quaterniond q_diff = QuatDiff(q0, q1);
    Eigen::Vector3d axis;
    double theta;
    QuaternionToAxisAngle(q_diff, axis, theta);
    return (theta / dt) * axis;
}

Eigen::Vector3d CalcQuaternionVelRel(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1, double dt)
{
    // calculate relative rotational velocity in the coordinate frame of q0
    Eigen::Quaterniond q_diff = q0.conjugate() * q1;
    Eigen::Vector3d axis;
    double theta;
    QuaternionToAxisAngle(q_diff, axis, theta);
    return (theta / dt) * axis;
}

Eigen::Vector3d CalcQuaternionVelRel(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, double dt)
{
    // calculate relative rotational velocity in the coordinate frame of q0
    Eigen::AngleAxisd aa0(v0.norm(), v0.normalized());
    Eigen::AngleAxisd aa1(v1.norm(), v1.normalized());

    Eigen::Quaterniond q0(aa0);
    Eigen::Quaterniond q1(aa1);

    Eigen::Quaterniond q_diff = q0.conjugate() * q1;
    Eigen::Vector3d axis;
    double theta;
    QuaternionToAxisAngle(q_diff, axis, theta);
    return (theta / dt) * axis;
}

Eigen::Quaterniond VecToQuat(const Eigen::Vector4d& v)
{
    return Eigen::Quaterniond(v[0], v[1], v[2], v[3]);
}

Eigen::Vector4d QuatToVec(const Eigen::Quaterniond& q)
{
    return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
}

Eigen::Quaterniond QuatDiff(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1)
{
    return q1 * q0.conjugate();
}

double QuatDiffTheta(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1)
{
    Eigen::Quaterniond dq = QuatDiff(q0, q1);
    return QuatTheta(dq);
}

double QuatTheta(const Eigen::Quaterniond& dq)
{
    double theta = 0;
    Eigen::Quaterniond q1 = dq;
    if (q1.w() > 1)
    {
        q1.normalize();
    }

    double sin_theta = std::sqrt(1 - q1.w() * q1.w());
    if (sin_theta > 0.0001)
    {
        theta = 2 * std::acos(q1.w());
        theta = NormalizeAngle(theta);
    }
    return theta;
}

void QuatNormalize(Eigen::Quaterniond& in)
{
    if(in.w() < 0.0)
        in.coeffs() *= -1;
}

Eigen::Quaterniond VecDiffQuat(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1)
{
    return Eigen::Quaterniond::FromTwoVectors(v0, v1);
}

Eigen::Vector3d QuatRotVec(const Eigen::Quaterniond& q, const Eigen::Vector3d& dir)
{
    Eigen::Vector3d rot_dir = Eigen::Vector3d::Zero();
    rot_dir  = q * dir;
    return rot_dir;
}

void ButterworthFilter(double dt, double cutoff, Eigen::VectorXd& out_x)
{
    double sampling_rate = 1 / dt;
    int n = static_cast<int>(out_x.size());

    double wc = std::tan(cutoff * M_PI / sampling_rate);
    double k1 = std::sqrt(2) * wc;
    double k2 = wc * wc;
    double a = k2 / (1 + k1 + k2);
    double b = 2 * a;
    double c = a;
    double k3 = b / k2;
    double d = -2 * a + k3;
    double e = 1 - (2 * a) - k3;

    double xm2 = out_x[0];
    double xm1 = out_x[0];
    double ym2 = out_x[0];
    double ym1 = out_x[0];

    for (int s = 0; s < n; ++s)
    {
        double x = out_x[s];
        double y = a * x + b * xm1 + c * xm2 + d * ym1 + e * ym2;

        out_x[s] = y;
        xm2 = xm1;
        xm1 = x;
        ym2 = ym1;
        ym1 = y;
    }

    double yp2 = out_x[n - 1];
    double yp1 = out_x[n - 1];
    double zp2 = out_x[n - 1];
    double zp1 = out_x[n - 1];

    for (int t = n - 1; t >= 0; --t)
    {
        double y = out_x[t];
        double z = a * y + b * yp1 + c * yp2 + d * zp1 + e * zp2;

        out_x[t] = z;
        yp2 = yp1;
        yp1 = y;
        zp2 = zp1;
        zp1 = z;
    }
}

}
}

#include "Device.h"
#include "DARTHelper.h"
#include <iostream>

using namespace MASS;
using namespace dart::dynamics;

Device::
Device()
{

}

Device::
Device(dart::dynamics::SkeletonPtr dPtr)
:mNumState(0),mNumAction(0),mNumDof(0),mNumActiveDof(0),mRootJointDof(0),mUseNN(false),mPhase(0),mTorqueMax(0.0),qr(0.0),ql(0.0),qr_prev(0.0),ql_prev(0.0)
{
    mSkeleton = dPtr;
}

Device::
~Device()
{
    mCharacter = nullptr;
}

void
Device::
LoadSkeleton(const std::string& path)
{
    mSkeleton = BuildFromFile(path);
}

void
Device::
SetCharacter(Character* character)
{
    mCharacter = character;
}

void
Device::
Initialize(dart::simulation::WorldPtr& wPtr, bool nn)
{
    if(mSkeleton == nullptr)
    {
        std::cout<<"Initialize Device First"<<std::endl;
        exit(0);
    }

    this->SetWorld(wPtr);
    mWorld->addSkeleton(mSkeleton);

    mUseNN = nn;

    const std::string& type =
        mSkeleton->getRootBodyNode()->getParentJoint()->getType();

    if(type == "FreeJoint")
        mRootJointDof = 6;
    else if(type == "PlanarJoint")
        mRootJointDof = 3;
    else
        mRootJointDof = 0;

    mDeviceSignals_y = std::deque<double>(1800,0);
    mDeviceSignals_L = std::deque<double>(1800,0);
    mDeviceSignals_R = std::deque<double>(1800,0);

    mNumDof = mSkeleton->getNumDofs();
    mNumActiveDof = mNumDof-mRootJointDof;
    mNumState = this->GetState().rows();
    mNumAction = mNumActiveDof;
    mAction.resize(mNumAction);

    mTorqueMax = 15.0;

    mDesiredTorque = Eigen::VectorXd::Zero(mNumDof);
    mDesiredTorque_Buffer.resize(1800);
    for(auto& t : mDesiredTorque_Buffer)
        t = Eigen::VectorXd::Zero(mNumDof);
}

void
Device::
Reset()
{
    mSkeleton->clearConstraintImpulses();
    mSkeleton->clearInternalForces();
    mSkeleton->clearExternalForces();

    dart::dynamics::SkeletonPtr skel_char = mCharacter->GetSkeleton();

    Eigen::VectorXd p(mNumDof);
    Eigen::VectorXd v(mNumDof);
    p.setZero();
    v.setZero();
    p.head(6) = skel_char->getPositions().head(6);
    v.head(6) = skel_char->getVelocities().head(6);
    p.segment<3>(6) = skel_char->getJoint("FemurL")->getPositions();
    p.segment<3>(9) = skel_char->getJoint("FemurR")->getPositions();
    v.segment<3>(6) = skel_char->getJoint("FemurL")->getVelocities();
    v.segment<3>(9) = skel_char->getJoint("FemurR")->getVelocities();

    mSkeleton->setPositions(p);
    mSkeleton->setVelocities(v);
    mSkeleton->computeForwardKinematics(true, false, false);

    mDesiredTorque_Buffer.clear();
    mDesiredTorque_Buffer.resize(1800);
    for(auto& t : mDesiredTorque_Buffer)
        t = Eigen::VectorXd::Zero(12);

    mDeviceSignals_y.clear();
    mDeviceSignals_y.resize(1800);
    mDeviceSignals_L.clear();
    mDeviceSignals_L.resize(1800);
    mDeviceSignals_R.clear();
    mDeviceSignals_R.resize(1800);

    qr = 0.0;
    ql = 0.0;
    qr_prev = 0.0;
    ql_prev = 0.0;
}

void
Device::
Step(const Eigen::VectorXd& a_)
{
    mSkeleton->setForces(a_);
}

void
Device::
Step(double t)
{
    if(mUseNN)
    {
        SetDesiredTorques(t);
        // SetSignals();
        mSkeleton->setForces(mDesiredTorque);
    }
    else
    {
        SetDesiredTorques2();

        Eigen::VectorXd tmp = Eigen::VectorXd::Zero(mDesiredTorque.size());
        tmp[6] = mDesiredTorque[6];
        tmp[9] = mDesiredTorque[7];
        mSkeleton->setForces(tmp);
    }
}

Eigen::VectorXd
Device::
GetState()
{
    // root Pos & Vel
    // Eigen::VectorXd positions = mSkeleton->getPositions();
    // Eigen::VectorXd velocities = mSkeleton->getVelocities();

    // dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
    // Eigen::Quaterniond rotation(root->getWorldTransform().rotation());
    // Eigen::Vector3d root_linvel = root->getCOMLinearVelocity();
    // Eigen::Vector3d root_angvel = root->getAngularVelocity();
    // Eigen::VectorXd state(23);

    // state << mPhase, rotation.w(), rotation.x(), rotation.y(), rotation.z(),
    //             root_linvel / 10., root_angvel/10., positions.tail<6>(), velocities.tail<6>()/10.;

    Eigen::VectorXd state(8);
    int delta_t = 180;
    for(int i=0; i<4; i++)
    {
        double torque = mDeviceSignals_y.at(delta_t-i*60);
        double des_torque_l =  1*torque;
        double des_torque_r = -1*torque;
        state[i*2] = des_torque_l/15.0;
        state[i*2+1] = des_torque_r/15.0 ;
    }

    return state;
}

void
Device::
SetAction(const Eigen::VectorXd& a)
{
    double action_scale = 1.0;
    mAction = a*action_scale;
    for(int i=0; i<mAction.size()-2; i++)
        mAction *= mTorqueMax;
}

void
Device::
SetDesiredTorques(double t)
{
}

double
lp_filter(double cur, double prev, double alpha)
{
    return (1-alpha)*prev + (alpha)*cur;
}

void
Device::
SetDesiredTorques2()
{
    if(qr==0.0 && ql==0.0 && qr_prev==0.0 && ql_prev==0.0)
    {
        ql = GetAngleQ("FemurL");
        qr = GetAngleQ("FemurR");
        ql_prev = ql;
        qr_prev = qr;
    }
    else{
        ql = GetAngleQ("FemurL");
        qr = GetAngleQ("FemurR");
    }

    double alpha = 0.05;
    ql = lp_filter(ql, ql_prev, alpha);
    qr = lp_filter(qr, qr_prev, alpha);
    ql_prev = ql;
    qr_prev = qr;

    double y = sin(qr) - sin(ql);

    int delta_t = 180;
    double k_ = 30.0;
    double beta_L = 1.0;
    double beta_Lhip = 1.0;
    double beta_R = 1.0;
    double beta_Rhip = 1.0;

    mDeviceSignals_y.pop_back();
    mDeviceSignals_y.push_front(k_*y);

    // double torque = k_ * y_delta_t;
    double torque = mDeviceSignals_y.at(delta_t);
    double des_torque_l =  1*torque*beta_L*beta_Lhip;
    double des_torque_r = -1*torque*beta_R*beta_Rhip;

    mDeviceSignals_L.pop_back();
    mDeviceSignals_L.push_front(des_torque_l);

    mDeviceSignals_R.pop_back();
    mDeviceSignals_R.push_front(des_torque_r);

    mDesiredTorque[6] = des_torque_l;
    mDesiredTorque[7] = des_torque_r;
}

double
Device::
GetAngleQ(const std::string& name)
{
    dart::dynamics::SkeletonPtr skel_char = mCharacter->GetSkeleton();
    Eigen::Vector3d dir = skel_char->getBodyNode(0)->getCOMLinearVelocity();
    dir /= dir.norm();

    Eigen::Vector3d p12 = skel_char->getBodyNode(name)->getCOM()-skel_char->getBodyNode(0)->getCOM();
    double p12_len = p12.norm();

    double l2 = dir[0]*p12[0] + dir[2]*p12[2];
    double l1 = sqrt(p12[0]*p12[0]+p12[2]*p12[2] - l2*l2);
    double x = sqrt(p12_len*p12_len - l1*l1);

    double sin = l2 / x;

    return asin(sin);
}

Eigen::VectorXd
Device::
GetDesiredTorques()
{
    return mDesiredTorque;
}

Eigen::VectorXd
Device::
GetDesiredTorques2()
{
    return mDesiredTorque;
}

std::deque<double>
Device::
GetSignals(int idx)
{
    if(idx==0)
        return mDeviceSignals_y;
    else if(idx==1)
        return mDeviceSignals_L;
    else if(idx==2)
        return mDeviceSignals_R;
}

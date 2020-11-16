#include "Device.h"
#include "DARTHelper.h"
#include <iostream>

using namespace MASS;
using namespace dart::dynamics;

Device::
Device()
:mNumState(0),mNumAction(0),mNumDof(0),mNumActiveDof(0),mRootJointDof(0),mUseNN(false),mTorqueMax(0.0),qr(0.0),ql(0.0),qr_prev(0.0),ql_prev(0.0)
{
    mDelta_t = 180;
    mK_ = 15.0;
}

Device::
Device(dart::dynamics::SkeletonPtr dPtr)
:Device()
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
    mSimulationHz = character->GetSimHz();
    mControlHz = character->GetConHz();
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

    signal_size = 1200 + mDelta_t;
    mDeviceSignals_y = std::deque<double>(signal_size,0);
    mDeviceSignals_L = std::deque<double>(signal_size,0);
    mDeviceSignals_R = std::deque<double>(signal_size,0);

    mNumDof = mSkeleton->getNumDofs();
    mNumActiveDof = mNumDof-mRootJointDof;
    mNumState = this->GetState().rows();
    mNumAction = mNumActiveDof;
    mAction.resize(mNumAction);

    mTorqueMax = 15.0;

    mDesiredTorque = Eigen::VectorXd::Zero(mNumDof);

    this->Reset();
}

void
Device::
Reset()
{
    dart::dynamics::SkeletonPtr skel_char = mCharacter->GetSkeleton();

    Eigen::VectorXd p(mNumDof);
    Eigen::VectorXd v(mNumDof);
    p.head(6) = skel_char->getPositions().head(6);
    v.head(6) = skel_char->getVelocities().head(6);
    p.segment<3>(6) = skel_char->getJoint("FemurL")->getPositions();
    p.segment<3>(9) = skel_char->getJoint("FemurR")->getPositions();
    v.segment<3>(6) = skel_char->getJoint("FemurL")->getVelocities();
    v.segment<3>(9) = skel_char->getJoint("FemurR")->getVelocities();

    mSkeleton->clearConstraintImpulses();
    mSkeleton->clearInternalForces();
    mSkeleton->clearExternalForces();

    mSkeleton->setPositions(p);
    mSkeleton->setVelocities(v);
    mSkeleton->computeForwardKinematics(true, false, false);

    mDeviceSignals_y.clear();
    mDeviceSignals_y.resize(signal_size);
    mDeviceSignals_L.clear();
    mDeviceSignals_L.resize(signal_size);
    mDeviceSignals_R.clear();
    mDeviceSignals_R.resize(signal_size);

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

        Eigen::VectorXd f = Eigen::VectorXd::Zero(mDesiredTorque.size());
        f[6] = mDesiredTorque[6];
        f[9] = mDesiredTorque[9];
        mSkeleton->setForces(f);
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
    // Eigen::VectorXd state(22);

    // state << rotation.w(), rotation.x(), rotation.y(), rotation.z(),
    //             root_linvel / 10., root_angvel/10., positions.tail<6>(), velocities.tail<6>()/10.;

    double history_window = 0.3;
    double history_interval = 0.05;
    int offset = (history_interval * mSimulationHz);
    int history_num = (history_window+0.001)/(history_interval)+1;

    Eigen::VectorXd state(history_num*2);

    double scaler = mK_/2.0;
    for(int i=0; i<history_num; i++)
    {
        double torque = mDeviceSignals_y.at(mDelta_t + i*offset);
        double des_torque_l =  1*torque;
        double des_torque_r = -1*torque;
        state[i*2] = des_torque_l/scaler;
        state[i*2+1] = des_torque_r/scaler;
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

    double beta_L = 1.0;
    double beta_Lhip = 1.0;
    double beta_R = 1.0;
    double beta_Rhip = 1.0;

    mDeviceSignals_y.pop_back();
    mDeviceSignals_y.push_front(mK_*y);

    // double torque = k_ * y_delta_t;
    double torque = mDeviceSignals_y.at(mDelta_t);
    double des_torque_l =  1*torque*beta_L*beta_Lhip;
    double des_torque_r = -1*torque*beta_R*beta_Rhip;

    mDeviceSignals_L.pop_back();
    mDeviceSignals_L.push_front(des_torque_l);

    mDeviceSignals_R.pop_back();
    mDeviceSignals_R.push_front(des_torque_r);

    mDesiredTorque[6] = des_torque_l;
    mDesiredTorque[9] = des_torque_r;
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
        return mDeviceSignals_L;
    else if(idx==1)
        return mDeviceSignals_R;
    else if(idx==2)
        return mDeviceSignals_y;
}

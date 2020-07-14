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

    mNumDof = mSkeleton->getNumDofs();
    mNumActiveDof = mNumDof-mRootJointDof;
    mNumState = this->GetState().rows();
    mNumAction = mNumActiveDof;
    mAction.resize(mNumAction);

    mTorqueMax = 15.0;

    mDesiredTorque = Eigen::VectorXd::Zero(12);
    mDesiredTorque_Buffer.resize(600);
    for(auto& t : mDesiredTorque_Buffer)
        t = Eigen::VectorXd::Zero(12);
    mDeviceSignals_y.resize(600);
    mDeviceSignals_L.resize(600);
    mDeviceSignals_R.resize(600);
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
    mDesiredTorque_Buffer.resize(600);
    for(auto& t : mDesiredTorque_Buffer)
        t = Eigen::VectorXd::Zero(12);

    mDeviceSignals_y.clear();
    mDeviceSignals_y.resize(600);
    mDeviceSignals_L.clear();
    mDeviceSignals_L.resize(600);
    mDeviceSignals_R.clear();
    mDeviceSignals_R.resize(600);

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

        int dof = mCharacter->GetNumDof();
        int root_dof = mCharacter->GetRootJointDof();
        Eigen::VectorXd char_torques = Eigen::VectorXd::Zero(dof);
        char_torques.tail(dof-root_dof) = mCharacter->GetDesiredTorques();
        char_torques[6] += mDesiredTorque[6];
        char_torques[15] += mDesiredTorque[7];
        mCharacter->GetSkeleton()->setForces(char_torques);

        Eigen::VectorXd tmp = Eigen::VectorXd::Zero(mDesiredTorque.size());
        mSkeleton->setForces(tmp);
    }
}

Eigen::VectorXd
Device::
GetState()
{
    // root Pos & Vel
    Eigen::VectorXd positions = mSkeleton->getPositions();
    Eigen::VectorXd velocities = mSkeleton->getVelocities();

    dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
    Eigen::Quaterniond rotation(root->getWorldTransform().rotation());
    Eigen::Vector3d root_linvel = root->getCOMLinearVelocity();
    Eigen::Vector3d root_angvel = root->getAngularVelocity();
    Eigen::VectorXd state(23);

    state << mPhase, rotation.w(), rotation.x(), rotation.y(), rotation.z(),
                root_linvel / 10., root_angvel/10., positions.tail<6>(), velocities.tail<6>()/10.;
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
    double offset = 60.0;

    for(int i=0; i<mAction.size(); i++)
    {
        if(mAction[i] > offset)
            mAction[i] = offset;
        if(mAction[i] < -offset)
            mAction[i] = -offset;
    }

    // double ratio = Pulse_Constant(t);
    double ratio = 1.0;

    Eigen::VectorXd cur_action;
    cur_action.resize(12);
    cur_action.head<6>().setZero();
    cur_action.segment<3>(6) = ratio * mAction.head<3>();
    cur_action.segment<3>(9) = ratio * mAction.segment<3>(3);
    mDesiredTorque_Buffer.pop_back();
    mDesiredTorque_Buffer.push_front(cur_action);

    mDesiredTorque = mDesiredTorque_Buffer.at(180);

    // double offset_L = mAction_Device[6];
    // if(offset_L<-2.0)
    //  offset_L = -2.0;
    // if(offset_L>2.0)
    //  offset_L = 2.0;

    // offset_L = 0.5 + offset_L/4.0;

    // double offset_R = mAction_Device[7];
    // if(offset_R<-2.0)
    //  offset_R = -2.0;
    // if(offset_R>2.0)
    //  offset_R = 2.0;

    // offset_R = 0.5 + offset_R/4.0;

    // double ratio = Pulse_Linear(t);
    // double ratio_L = Pulse_Period(t, offset_L);
    // double ratio_R = Pulse_Period(t, offset_R);

    // mDesiredTorque_Device.head<6>().setZero();
    // mDesiredTorque_Device.segment<3>(6) = ratio * mAction_Device.head<3>();
    // mDesiredTorque_Device.segment<3>(9) = ratio * mAction_Device.segment<3>(3);
    // mDesiredTorque_Device[6] = ratio * mAction_Device[0];
    // mDesiredTorque_Device[7] = ratio * mAction_Device[1];
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
        qr = GetAngleQ(1);
        ql = GetAngleQ(6);
        qr_prev = qr;
        ql_prev = ql;
    }
    else{
        qr = GetAngleQ(1);
        ql = GetAngleQ(6);
    }

    double alpha = 0.05;
    qr = lp_filter(qr, qr_prev, alpha);
    ql = lp_filter(ql, ql_prev, alpha);

    qr_prev = qr;
    ql_prev = ql;

    double y = sin(qr) - sin(ql);

    // std::cout << "y : " << y << std::endl;
    // std::cout << "qr : " << sin(qr) << std::endl;
    // std::cout << "ql : " << sin(ql) << std::endl;

    int delta_t = 180;
    double k_ = 10.0;
    double beta_R = 1.0;
    double beta_Rhip = 1.0;
    double beta_L = 1.0;
    double beta_Lhip = 1.0;

    mDeviceSignals_y.pop_back();
    mDeviceSignals_y.push_front(k_*y);

    // double torque = k_ * y_delta_t;
    double torque = mDeviceSignals_y.at(delta_t);
    double des_torque_r = -1*torque*beta_R*beta_Rhip;
    double des_torque_l =  1*torque*beta_L*beta_Lhip;

    mDeviceSignals_L.pop_back();
    mDeviceSignals_L.push_front(des_torque_l);

    mDeviceSignals_R.pop_back();
    mDeviceSignals_R.push_front(des_torque_r);

    mDesiredTorque[6] = des_torque_r;
    mDesiredTorque[7] = des_torque_l;
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

double
Device::
GetAngleQ(int idx)
{
    dart::dynamics::SkeletonPtr skel_char = mCharacter->GetSkeleton();
    Eigen::Vector3d dir = skel_char->getBodyNode(0)->getCOMLinearVelocity();
    dir /= dir.norm();

    Eigen::Vector3d p12 = skel_char->getBodyNode(idx)->getCOM()-skel_char->getBodyNode(0)->getCOM();
    double p12_len = p12.norm();

    double l2 = dir[0]*p12[0] + dir[2]*p12[2];
    double l1 = sqrt(p12[0]*p12[0]+p12[2]*p12[2] - l2*l2);
    double x = sqrt(p12_len*p12_len - l1*l1);

    double sin = l2 / x;

    return asin(sin);
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

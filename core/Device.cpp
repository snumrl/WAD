#include "Device.h"
#include <iostream>

using namespace MASS;
using namespace dart::dynamics;

Device::
Device()
{

}

Device::
Device(dart::dynamics::SkeletonPtr device)
{
    mSkeleton = device;
}

void
Device::
Initialize()
{
    if(this->GetSkeleton() == nullptr)
    {
        std::cout<<"Initialize Device First"<<std::endl;
        exit(0);
    }

    if(this->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="FreeJoint")
        mRootJointDof = 6;
    else if(this->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="PlanarJoint")
        mRootJointDof = 3;
    else
        mRootJointDof = 0;

    mNumActiveDof = this->GetSkeleton()->getNumDofs()-mRootJointDof;
    mNumState = this->GetState(0.0, 1.0).rows();

    mTorqueMax_Device = 100.;

    mNumAction = mNumActiveDof;
    mNumState = this->GetState(0.0, 1.0).rows();
}

void
Device::
Reset()
{
    mSkeleton->clearConstraintImpulses();
    mSkeleton->clearInternalForces();
    mSkeleton->clearExternalForces();
}

Eigen::VectorXd
Device::
GetState(double worldTime, double maxTime)
{
    // root Pos & Vel
    Eigen::VectorXd positions = mSkeleton->getPositions();
    Eigen::VectorXd velocities = mSkeleton->getVelocities();

    dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
    Eigen::Quaterniond rotation(root->getWorldTransform().rotation());
    Eigen::Vector3d root_linvel = root->getCOMLinearVelocity();
    Eigen::Vector3d root_angvel = root->getAngularVelocity();

    double t_phase = maxTime;
    double phi = std::fmod(worldTime,t_phase)/t_phase;

    // int n = mSkeleton->getNumJoints();
    // int dofs = mSkeleton->getNumDofs();
    // for(int i=0; i<n; i++)
    // {
    //     std::cout << "device joint num : " << n << std::endl;
    //     std::cout << "device dofs  : " << dofs << std::endl;
    //     std::cout << i << " : " << mSkeleton->getJoint(i)->getName() << " dof : " << mSkeleton->getJoint(i)->getNumDofs() << std::endl;
    //     std::cout << i << " : " << mSkeleton->getJoint(i)->getName() << " force : " << mSkeleton->getJoint(i)->getForces() << std::endl;
    //     std::cout << "position size : " << mSkeleton->getPositions().size() << std::endl;
    // }

    Eigen::VectorXd state(23);
    state << phi, rotation.w(), rotation.x(), rotation.y(), rotation.z(),
                root_linvel / 10., root_angvel/10., positions.tail<6>(), velocities.tail<6>()/10.;

    return state;
}

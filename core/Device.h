#ifndef __MASS_DEVICE_H__
#define __MASS_DEVICE_H__
#include "dart/dart.hpp"

namespace MASS
{

class Device
{
public:
    Device();
    Device(dart::dynamics::SkeletonPtr device);

    void Initialize();
    void Reset();
    Eigen::VectorXd GetState(double worldTime, double maxTime);

    const dart::dynamics::SkeletonPtr& GetSkeleton(){ return mSkeleton; }

    int GetNumState(){ return mNumState;}
    int GetNumActiveDof(){ return mNumActiveDof;}
    int GetNumAction(){ return mNumAction;}

public:
    dart::dynamics::SkeletonPtr mSkeleton;

    double mAssistiveTorqueMax;
    int mNumState;
    int mNumActiveDof;
    int mRootJointDof;
    int mNumAction;

public:
    //Dynamics
};

}
#endif

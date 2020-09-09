#ifndef __MASS_DEVICE_H__
#define __MASS_DEVICE_H__
#include "dart/dart.hpp"
#include "Character.h"

namespace MASS
{
class Device
{
public:
    Device();
    Device(dart::dynamics::SkeletonPtr dPtr);
    ~Device();

    void LoadSkeleton(const std::string& path);
    void Initialize(dart::simulation::WorldPtr& wPtr, bool nn);

    void SetWorld(dart::simulation::WorldPtr& wPtr){ mWorld = wPtr; }
    void SetCharacter(Character* character);
    const dart::dynamics::SkeletonPtr& GetSkeleton(){ return mSkeleton; }

    void Reset();
    void Step(const Eigen::VectorXd& a_);
    void Step(double t);

    Eigen::VectorXd GetState();

    void SetAction(const Eigen::VectorXd& a);
    Eigen::VectorXd GetAction(){ return mAction; }

    void SetDesiredTorques(double t);
    Eigen::VectorXd GetDesiredTorques();

    void SetDesiredTorques2();
    Eigen::VectorXd GetDesiredTorques2();

    // Learning
    int GetNumState(){ return mNumState;}
    int GetNumAction(){ return mNumAction;}
    int GetNumDofs(){ return mNumDof;}
    int GetNumActiveDof(){ return mNumActiveDof;}
    int GetRootJointDof(){ return mRootJointDof;}

    // Common
    void SetTorqueMax(double m){ mTorqueMax = m;}
    double GetTorqueMax(){ return mTorqueMax;}

    double GetAngleQ(const std::string& name);
    // void SetSignals();
    std::deque<double> GetSignals(int idx);

    void SetPhase(double p){mPhase = p;}
    double GetPhase(){ return mPhase; }

    void setDelta_t(int t){mDelta_t = t;}
    void setK_(double k){mK_ = k;}

    int getDelta_t(){return mDelta_t;}
    double getK_(){return mK_;}

private:
    dart::dynamics::SkeletonPtr mSkeleton;
    dart::simulation::WorldPtr mWorld;
    Character* mCharacter;

    int mNumState;
    int mNumAction;
    int mNumDof;
    int mNumActiveDof;
    int mRootJointDof;

    bool mUseNN;

    double mPhase;
    double mTorqueMax;
    Eigen::VectorXd mAction;
    Eigen::VectorXd mDesiredTorque;
    std::deque<Eigen::VectorXd> mDesiredTorque_Buffer;

    std::deque<double> mDeviceSignals_y;
    std::deque<double> mDeviceSignals_L;
    std::deque<double> mDeviceSignals_R;

    double qr;
    double ql;
    double qr_prev;
    double ql_prev;

    int signal_size;
    int mDelta_t;
    double mK_;
};

}
#endif

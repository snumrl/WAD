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

    void SetDelta_t(double t);
    double GetDelta_t(){return mDelta_t;}

    void SetK_(double k);
    double GetK_(){return mK_;}

    void SetNumParamState(int n);
    int GetNumParamState(){return mNumParamState;}
    void SetParamState(Eigen::VectorXd paramState);
    Eigen::VectorXd GetParamState(){return mParamState;}
    Eigen::VectorXd GetMinV(){return mMin_v;}
    Eigen::VectorXd GetMaxV(){return mMax_v;}
    void SetMinMaxV(int idx, double lower, double upper);
    void SetAdaptiveParams(std::string name, double lower, double upper);

private:
    dart::dynamics::SkeletonPtr mSkeleton;
    dart::simulation::WorldPtr mWorld;
    Character* mCharacter;

    int mNumState;
    int mNumAction;
    int mNumDof;
    int mNumActiveDof;
    int mRootJointDof;

    int mSimulationHz;
    int mControlHz;

    bool mUseNN;

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
    double mDelta_t;
    double mDelta_t_scaler;
    double mK_;
    double mK_scaler;

    int mNumParamState;
    Eigen::VectorXd mParamState;
    Eigen::VectorXd mMin_v;
    Eigen::VectorXd mMax_v;
};

}
#endif

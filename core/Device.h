#ifndef __MASS_DEVICE_H__
#define __MASS_DEVICE_H__
#include "dart/dart.hpp"
#include "Character.h"

namespace MASS
{
class Device
{
public:
    Device(dart::simulation::WorldPtr& wPtr);
    ~Device();

    void LoadSkeleton(const std::string& path, bool load_obj);
    const dart::dynamics::SkeletonPtr& GetSkeleton(){ return mSkeleton; }

    void Initialize();

    void SetWorld(dart::simulation::WorldPtr& wPtr){ mWorld = wPtr; }
    void SetCharacter(Character* character){mCharacter = character;}

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
    void SetUseDeviceNN(bool b){ mUseDeviceNN = b; }

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

    void SetHz(int cHz, int sHz);
    void SetControlHz(int hz){mControlHz = hz;}
    int GetControlHz(){return mControlHz;}
    void SetSimulationHz(int hz){mSimulationHz = hz;}
    int GetSimulationHz(){return mSimulationHz;}
    int GetNumSteps(){ return mNumSteps; }
    void SetNumSteps(int step){ mNumSteps=step; }

    void SetNumParamState(int n);
    int GetNumParamState(){return mNumParamState;}

    void SetParamState(const Eigen::VectorXd& paramState);
    Eigen::VectorXd GetParamState(){return mParamState;}

    Eigen::VectorXd GetMinV(){return mMin_v;}
    Eigen::VectorXd GetMaxV(){return mMax_v;}
    void SetMinMaxV(int idx, double lower, double upper);
    void SetAdaptiveParams(std::map<std::string, std::pair<double, double>>& p);
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
    int mNumSteps;

    bool mUseDeviceNN;

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

    double mDelta_t;
    double mDelta_t_scaler;
    int mDelta_t_idx;
    double mK_;
    double mK_scaler;

    int mNumParamState;
    Eigen::VectorXd mParamState;
    Eigen::VectorXd mMin_v;
    Eigen::VectorXd mMax_v;
};

}
#endif

#ifndef __DEVICE_H__
#define __DEVICE_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "DARTHelper.h"

using namespace dart::dynamics;
using namespace dart::simulation;
namespace WAD
{

class Device
{
public:
    Device(WorldPtr& wPtr);
    ~Device();

    void LoadSkeleton(const std::string& path, bool load_obj);
    const SkeletonPtr& GetSkeleton(){ return mSkeleton; }

    void Initialize();

    void SetWorld(WorldPtr& wPtr){ mWorld = wPtr; }
    void SetCharacter(Character* character){mCharacter = character;}

    void Reset();
    void Step(const Eigen::VectorXd& a_);
    void Step(double t);

    Eigen::VectorXd GetState() const;
    void SetAction(const Eigen::VectorXd& a);
    const Eigen::VectorXd& GetAction(){ return mAction; }

    void SetDesiredTorques(double t);
    void SetDesiredTorques2();
    const Eigen::VectorXd& GetDesiredTorques();

    // Learning
    int GetNumState() const { return mNumState;}
    int GetNumAction() const { return mNumAction;}
    int GetNumDofs() const { return mNumDof;}
    int GetNumActiveDof() const { return mNumActiveDof;}
    int GetRootJointDof() const { return mRootJointDof;}
    void SetUseDeviceNN(bool b) { mUseDeviceNN = b; }

    // Common
    double GetAngleQ(const std::string& name);
    const std::deque<double>& GetSignals(int idx);

    void SetTorqueMax(double m){ mTorqueMax = m;}
    double GetTorqueMax() const { return mTorqueMax;}

    void SetDelta_t(double t);
    double GetDelta_t(){return mDelta_t;}

    void SetK_(double k);
    double GetK_(){return mK_;}

    double GetSignalY(){return mDeviceSignals_y[0];}

    void SetHz(int cHz, int sHz);
    void SetControlHz(int hz){mControlHz = hz;}
    void SetSimulationHz(int hz){mSimulationHz = hz;}
    void SetNumSteps(int step){ mNumSteps=step; }
    int GetControlHz() const {return mControlHz;}
    int GetSimulationHz() const {return mSimulationHz;}
    int GetNumSteps() const { return mNumSteps; }

    void SetNumParamState(int n);
    void SetParamState(const Eigen::VectorXd& paramState);
    void SetMinMaxV(int idx, double lower, double upper);
    void SetAdaptiveParams(std::map<std::string, std::pair<double,double>>& p);
    void SetAdaptiveParams(std::string name, double lower, double upper);
    const std::map<std::string, std::pair<double,double>>& GetAdaptiveParams(){return mAdaptiveParams;}

    int GetNumParamState() const {return mNumParamState;}
    const Eigen::VectorXd& GetParamState(){return mParamState;}
    const Eigen::VectorXd& GetMinV(){return mMin_v;}
    const Eigen::VectorXd& GetMaxV(){return mMax_v;}

private:
    SkeletonPtr mSkeleton;
    WorldPtr mWorld;
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

    std::map<std::string, std::pair<double, double>> mAdaptiveParams;
    int mNumParamState;
    Eigen::VectorXd mParamState;
    Eigen::VectorXd mMin_v;
    Eigen::VectorXd mMax_v;
};

}
#endif

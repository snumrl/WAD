#ifndef __MASS_CHARACTER_H__
#define __MASS_CHARACTER_H__
#include "dart/dart.hpp"
#include "JointData.h"
#include "MetabolicEnergy.h"
#include "Contact.h"
#include <deque>
#include <map>

namespace MASS
{

class BVH;
class Muscle;
class Device;
class JointData;
class MetabolicEnergy;
class Contact;
// class Torques;

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::MatrixXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};

class Character
{
public:
	Character(dart::simulation::WorldPtr& wPtr);
	~Character();

	void LoadSkeleton(const std::string& path, bool load_obj);
	void LoadMuscles(const std::string& path);
	void LoadBVH(const std::string& path,bool cyclic=true);
	void LoadBVHset(double lower, double upper);

	const dart::dynamics::SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<dart::dynamics::BodyNode*>& GetEndEffectors(){return mEndEffectors;}
	const Device* GetDevice(){return mDevice;}
	BVH* GetBVH(){return mBVH;}
	const std::map<std::string, std::vector<Muscle*>> GetMusclesMap(){return mMuscles_Map;}

	void Initialize();
	void Initialize_Muscles();
	void Initialize_Rewards();
	void Initialize_Forces();
	void Initialize_Mass();
	void Initialize_Speed();
	void Initialize_JointWeights();

	void SetWorld(const dart::simulation::WorldPtr& wPtr){ mWorld = wPtr; }
	void SetDevice(Device* device);

	void SetHz(int sHz, int cHz);
	void SetSimulationHz(int hz){ mSimulationHz=hz; }
	void SetControlHz(int hz){ mControlHz=hz; }
	void SetNumSteps(int step){ mNumSteps=step; }
	int GetSimulationHz(){ return mSimulationHz; }
	int GetControlHz(){ return mControlHz; }
	int GetNumSteps(){ return mNumSteps; }
	void SetPhase();
	void SetPhases();
	double GetPhase(){ return mPhase; }
	std::pair<double,double> GetPhases(){ return mPhases; }

	void SetConstraints();
	void RemoveConstraints();

	void SetPDParameters();
	const Eigen::VectorXd& GetPDParameters_Kp(){ return mKp; }
	const Eigen::VectorXd& GetPDParameters_Kv(){ return mKv; }

	Eigen::VectorXd GetState();
	Eigen::VectorXd GetState_Character();

	void Reset();
	void Reset_Muscles();

	void Step(bool isRender);
	void Step_Muscles(int simCount, int randomSampleIndex, bool isRender);

	double GetReward();
	double GetReward_Character();
	double GetReward_Character_Imitation();
	double GetReward_Character_Efficiency();
	double GetReward_ActionReg();
	double GetReward_TorqueMin();
	double GetReward_Vel();
	double GetCurReward(){return mCurReward;}

	void SetAction(const Eigen::VectorXd& a);
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	void SetDesiredTorques();
	void SetTargetPosAndVel(double t);
	void SetTargetPositions(double t,double dt,int frame, int frameNext, double frameFraction);
	void SetTargetVelocities(double t,double dt,int frame, int frameNext, double frameFraction);

	const Eigen::VectorXd& GetAction(){ return mAction; }
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetTargetPositions(){ return mTargetPositions; }
	const Eigen::VectorXd& GetTargetVelocities(){ return mTargetVelocities; }
	const Eigen::VectorXd& GetAdaptiveTargetPositions(){return mAdaptiveTargetPositions;}
	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired);

	Eigen::VectorXd GetMuscleTorques();
	MuscleTuple& GetCurrentMuscleTuple(){ return mCurrentMuscleTuple; }
	std::vector<MuscleTuple>& GetMuscleTuples(){ return mMuscleTuples; }

	void SetDevice_OnOff(bool on);
	bool GetDevice_OnOff(){ return mOnDevice; }

	void SetDevice_On();
	void SetDevice_Off();

	int GetNumMuscles(){return mNumMuscle;}
	int GetNumMusclesMap(){return mNumMuscleMap;}
	int GetNumState(){return mNumState;}
	int GetNumState_Char(){return mNumState_Char;}
	int GetNumDof(){return mDof;}
	int GetNumActiveDof(){return mNumActiveDof;}
	int GetNumAction(){return mNumAction;}
	int GetRootJointDof(){return mRootJointDof;}
	int GetNumTotalRelatedDofs(){return mNumTotalRelatedDof;}

	void SetUseMuscle(bool b){mUseMuscle = b;}
	bool GetUseMuscle(){return mUseMuscle;}

	void SetUseDevice(bool b){mUseDevice = b;}
	bool GetUseDevice(){return mUseDevice;}

	void SetRewards();
	std::map<std::string, std::deque<double>> GetRewards(){return mRewards;}

	double GetCurFrame(){ return mCurFrame; }
	std::deque<double> GetSignals(int idx);
	JointData* GetJointDatas(){return mJointDatas;}
	MetabolicEnergy* GetMetabolicEnergy(){return mMetabolicEnergy;}
	Contact* GetContacts(){return mContacts;}

	void SetMass();
	void SetMassRatio(double r);
	double GetMass(){return mMass;}
	double GetMassRatio(){return mMassRatio;}

	double SetSpeedIdx(double s);
	void SetBVHidx(double r);
	void SetSpeedRatio(double r);
	void SetForceRatio(double r);
	void SetMaxForces();

	double GetSpeedRatio(){return mSpeedRatio;}
	double GetForceRatio(){return mForceRatio;}
	Eigen::VectorXd GetMaxForces(){return mMaxForces;}

	void SetMeasure();
	void SetCoT();
	void SetCurVelocity();
	void SetTrajectory();

	double GetCoT(){return mCurCoT;}
	double GetCurVelocity(){return mCurVel;}

	void SetMinMaxV(int idx, double lower, double upper);
	const Eigen::VectorXd& GetMinV(){return mParamMin;}
	const Eigen::VectorXd& GetMaxV(){return mParamMax;}

	void SetNumParamState(int n);
	int GetNumParamState(){return mNumParamState;}
	void SetAdaptiveParams(std::map<std::string, std::pair<double, double>>& p);
	void SetParamState(Eigen::VectorXd paramState);
	const Eigen::VectorXd& GetParamState(){return mParamState;}

private:
	dart::simulation::WorldPtr mWorld;
	dart::dynamics::SkeletonPtr mSkeleton;
	std::vector<dart::dynamics::BodyNode*> mEndEffectors;
	std::vector<Muscle*> mMuscles;
	std::vector<Muscle*> mMuscles_Femur;
	std::map<std::string, std::vector<Muscle*>> mMuscles_Map;

	BVH* mBVH;
	BVH* mBVH_;
	std::string mBVHpath;
	std::vector<BVH*> mBVHset;
	std::map<std::string,std::string> mBVHmap;
	bool mBVHcyclic;

	Device* mDevice;
	JointData* mJointDatas;
	MetabolicEnergy* mMetabolicEnergy;
	Contact* mContacts;

	int mDof;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumState;
	int mNumState_Char;
	int mNumAction;
	int mNumAdaptiveDof;
	int mNumTotalRelatedDof;
	int mNumMuscle;
	int mNumMuscleMap;
	int mNumBodyNodes;
	int mNumJoints;

	int mControlHz;
	int mSimulationHz;
	int mNumSteps;
	double mCurFrame;
	double mPhase;
	std::pair<double,double> mPhases;

	bool mUseDevice;
	bool mUseMuscle;
	bool mOnDevice;
	bool mLowerBody;

	double mMass;
	double mMassRatio;
	double mForceRatio;
	double mSpeedRatio;

	int mStepCnt;
	int mStepCnt_total;

	double mCurCoT;
	double mCurVel;
	Eigen::Vector3d mCurVel3d;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mAdaptiveTargetPositions;
	Eigen::VectorXd mAdaptiveTargetVelocities;
	Eigen::VectorXd mDesiredTorque;

	Eigen::VectorXd mAction;
	Eigen::VectorXd mAction_prev;
	Eigen::VectorXd mActivationLevels;

	Eigen::VectorXd mAngVel, mAngVel_prev;
	Eigen::VectorXd mPos, mPos_prev;
	Eigen::Vector3d mRootPos, mRootPos_prev;

	Eigen::VectorXd mJointWeights;
	Eigen::VectorXd mDefaultMass;
	Eigen::VectorXd mMaxForces;
	Eigen::VectorXd mDefaultForces;

	std::vector<std::deque<double>> mFemurSignals;
	std::deque<Eigen::Vector3d> mRootTrajectory;

	std::vector<std::string> mRewardTags;
	std::map<std::string, double> mReward;
	std::map<std::string, std::deque<double>> mRewards;
	double mCurReward = 0.0;

	MuscleTuple mCurrentMuscleTuple;
	std::vector<MuscleTuple> mMuscleTuples;

	dart::constraint::WeldJointConstraintPtr mWeldJoint_Hip;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_LeftLeg;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_RightLeg;

    int mNumParamState;
    Eigen::VectorXd mParamState;
    Eigen::VectorXd mParamMin;
    Eigen::VectorXd mParamMax;
};



};

#endif

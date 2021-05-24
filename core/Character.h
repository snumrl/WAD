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

	bool isLowerBody(std::string& body);

	const dart::dynamics::SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<dart::dynamics::BodyNode*>& GetEndEffectors(){return mEndEffectors;}
	const Device* GetDevice(){return mDevice;}
	BVH* GetBVH(){return mBVH;}
	const std::map<std::string, std::vector<Muscle*>> GetMusclesMap(){return mMusclesMap;}

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
	void SetFrame();
	void SetPhase();
	void SetPhases();
	double GetPhase(){ return mPhase; }
	double GetPhasePrev(){ return mPhasePrev; }
	double GetAdaptivePhase(){ return mAdaptivePhase; }
	double GetAdaptivePhasePrev(){ return mAdaptivePhasePrev; }
	double GetAdaptiveTime(){ return mAdaptiveTime; }
	std::pair<double,double> GetPhases(){ return mPhases; }
	std::pair<double,double> GetAdaptivePhases(){ return mAdaptivePhases; }

	double GetCurTime();
	double GetControlTimeStep();

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
	void SetMuscleTuple();

	std::pair<double,double> GetReward();
	std::pair<double,double> GetReward_Character();
	std::pair<double,double> GetReward_Character_Imitation();
	std::pair<double,double> GetReward_Character_Efficiency();
	// std::vector<double> GetReward();
	// std::vector<double> GetReward_Character();
	// double GetReward_Character_Imitation();
	// std::vector<double> GetReward_Character_Efficiency();
	double GetReward_ActionReg();
	double GetReward_Vel();
	double GetReward_Width();
	double GetReward_Height();
	double GetReward_Pose();
	double GetCurReward(){return mCurReward;}

	void SetAction(const Eigen::VectorXd& a);
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	void SetDesiredTorques();

	void GetFrameNum(double t, double dt, int& frame, int& frameNext, double& frameFraction);
	void GetPosAndVel(double t, Eigen::VectorXd& pos, Eigen::VectorXd& vel);
	void GetPos(double t, double dt, int frame, int frameNext, double frameFraction, Eigen::VectorXd& pos);
	void GetVel(double t, double dt, int frame, int frameNext, double frameFraction, Eigen::VectorXd& vel);

	const Eigen::VectorXd& GetAction(){ return mAction; }
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetTargetPositions(){ return mTargetPositions; }
	const Eigen::VectorXd& GetTargetVelocities(){ return mTargetVelocities; }
	const Eigen::VectorXd& GetReferencePositions(){return mReferencePositions;}
	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired);

	void GetNextPosition(Eigen::VectorXd cur, Eigen::VectorXd delta, Eigen::VectorXd& next);
	Eigen::VectorXd GetEndEffectorStatePosAndVel(const Eigen::VectorXd pos, const Eigen::VectorXd vel);

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
	int GetNumState_Char(){return mNumStateChar;}
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

	double GetFrame(){ return mFrame;}
	double GetFramePrev(){ return mFramePrev;}
	double GetAdaptiveFrame(){ return mAdaptiveFrame;}
	double GetAdaptiveFramePrev(){ return mAdaptiveFramePrev;}
	std::deque<double> GetSignals(int idx);
	JointData* GetJointDatas(){return mJointDatas;}
	MetabolicEnergy* GetMetabolicEnergy(){return mMetabolicEnergy;}
	Contact* GetContacts(){return mContacts;}

	void SetAdaptiveMotion(bool b){ mAdaptiveMotion = b;}

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

	void SetMeasure(bool isRender);
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
	std::vector<Muscle*> mMusclesFemur;
	std::map<std::string, std::vector<Muscle*>> mMusclesMap;

	BVH* mBVH;
	BVH* mBVH_;
	std::string mBVHpath;
	std::vector<BVH*> mBVHset;
	std::map<std::string,std::string> mBVHmap;
	bool mBVHcyclic;

	Device* mDevice;
	Contact* mContacts;
	JointData* mJointDatas;
	MetabolicEnergy* mMetabolicEnergy;

	int mDof;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumAdaptiveDof;
	int mNumAdaptiveSpatialDof;
	int mNumAdaptiveTemporalDof;
	int mNumTotalRelatedDof;
	int mLowerMuscleRelatedDof;
	int mUpperMuscleRelatedDof;

	int mNumJoints;
	int mNumBodyNodes;
	int mNumState;
	int mNumStateChar;
	int mNumAction;
	int mNumMuscle;
	int mNumMuscleMap;
	int mNumSteps;
	int mControlHz;
	int mSimulationHz;
	int mStepCnt;
	int mStepCntTotal;

	double mAdaptiveTime;
	double mTemporalDisplacement;
	double mFrame;
	double mFramePrev;
	double mAdaptiveFrame;
	double mAdaptiveFramePrev;
	double mPhase;
	double mPhasePrev;
	double mAdaptivePhase;
	double mAdaptivePhasePrev;
	std::pair<double,double> mPhases;
	std::pair<double,double> mAdaptivePhases;

	double mTimeOffset;

	bool mUseDevice;
	bool mUseMuscle;
	bool mOnDevice;
	bool mLowerBody;
	bool mAdaptiveMotion;
	bool mAdaptiveLowerBody;

	double mMass;
	double mMassRatio;
	double mForceRatio;
	double mSpeedRatio;

	double mCurCoT;
	double mCurVel;
	double mCurHeadVel;
	Eigen::Vector3d mCurVel3d;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mReferencePositions;
	Eigen::VectorXd mReferenceVelocities;

	Eigen::VectorXd mDesiredTorque;

	Eigen::VectorXd mAction;
	Eigen::VectorXd mAction_prev;
	Eigen::VectorXd mActivationLevels;

	Eigen::VectorXd mJointWeights;
	Eigen::VectorXd mDefaultMass;
	Eigen::VectorXd mMaxForces;
	Eigen::VectorXd mDefaultForces;

	std::vector<std::deque<double>> mFemurSignals;
	std::deque<Eigen::Vector3d> mRootTrajectory;
	std::deque<Eigen::Vector3d> mHeadTrajectory;

	std::vector<std::string> mRewardTags;
	std::map<std::string, double> mReward;
	std::map<std::string, std::deque<double>> mRewards;
	double mCurReward = 0.0;

	MuscleTuple mCurrentMuscleTuple;
	std::vector<MuscleTuple> mMuscleTuples;

	dart::constraint::WeldJointConstraintPtr mWeldJointHip;
    dart::constraint::WeldJointConstraintPtr mWeldJointLeftLeg;
    dart::constraint::WeldJointConstraintPtr mWeldJointRightLeg;

    int mNumParamState;
    Eigen::VectorXd mParamState;
    Eigen::VectorXd mParamMin;
    Eigen::VectorXd mParamMax;
};



};

#endif

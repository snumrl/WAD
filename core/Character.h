#ifndef __CHARACTER_H__
#define __CHARACTER_H__

#include "dart/dart.hpp"
#include "BVH.h"
#include "Muscle.h"
#include "Utils.h"
#include "DARTHelper.h"
#include "Contact.h"
#include "JointData.h"
#include "MetabolicEnergy.h"

#include <deque>
#include <map>

using namespace dart::dynamics;
using namespace dart::simulation;
namespace WAD
{

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::MatrixXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};

class Device;
class Character
{
public:
	Character(WorldPtr& wPtr);
	~Character();

	void LoadSkeleton(const std::string& path, bool load_obj);
	void LoadMuscles(const std::string& path);
	void LoadBVH(const std::string& path,bool cyclic=true);
	
	bool isLowerBody(std::string& body);

	const SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<BodyNode*>& GetEndEffectors(){return mEndEffectors;}
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
	void Initialize_Contacts();
	void SetJointPositionLimits();

	void SetWorld(const WorldPtr& wPtr){ mWorld = wPtr; }
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

	double GetReward_Energy();
	double GetReward_Pose();
	double GetReward_Vel();
	double GetReward_Stride();
	double GetReward_Time();
	double GetReward_Width();
	double GetCurReward(){return mCurReward;}

	void SetAction(const Eigen::VectorXd& a);
	void SetActionAdaptiveMotion(const Eigen::VectorXd& a);
	void SetActionImitationLearning(const Eigen::VectorXd& a);
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

	void SetUseAdaptiveSampling(bool b){mAdaptiveSampling = b;}
	bool GetUseAdaptiveSampling(){return mAdaptiveSampling;}

	void SetUseAdaptiveMotion(bool b){mAdaptiveMotion = b;}
	bool GetUseAdaptiveMotion(){return mAdaptiveMotion;}

	void SetRewards();
	std::map<std::string, std::deque<double>> GetRewards(){return mRewards;}

	double GetFrame(){ return mFrame;}
	double GetFramePrev(){ return mFramePrev;}
	double GetAdaptiveFrame(){ return mAdaptiveFrame;}
	double GetAdaptiveFramePrev(){ return mAdaptiveFramePrev;}
	std::deque<double> GetSignals(int idx);
	JointData* GetJointDatas(){return mJointDatas;}
	MetabolicEnergy* GetMetabolicEnergy(){return mMetabolicEnergy;}
	// Contact* GetContacts(){return mContacts;}

	void SetMass();
	void SetMassRatio(double r);
	double GetMass(){return mMass;}
	double GetMassRatio(){return mMassRatio;}

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
	void SetComHistory();
	void SetFoot();
	void SetContact();

	double GetCoT(){return mCurCoT;}
	double GetCurVelocity(){return mCurVel;}
	double GetStride(){return mStride;}
	double GetStrideL(){return mStrideL;}
	double GetStrideR(){return mStrideR;}

	void SetMinMaxV(int idx, double lower, double upper);
	const Eigen::VectorXd& GetMinV(){return mParamMin;}
	const Eigen::VectorXd& GetMaxV(){return mParamMax;}

	void SetNumParamState(int n);
	int GetNumParamState(){return mNumParamState;}
	void SetAdaptiveParams(std::map<std::string, std::pair<double, double>>& p);
	void SetParamState(Eigen::VectorXd paramState);
	const Eigen::VectorXd& GetParamState(){return mParamState;}
	const std::map<std::string, std::pair<double,double>>& GetAdaptiveParams(){return mAdaptiveParams;}
private:
	WorldPtr mWorld;
	SkeletonPtr mSkeleton;
	std::vector<BodyNode*> mEndEffectors;
	std::vector<Muscle*> mMuscles;
	std::vector<Muscle*> mMusclesFemur;
	std::map<std::string, std::vector<Muscle*>> mMusclesMap;

	BVH* mBVH;
	std::map<std::string,std::string> mBVHmap;
	
	Device* mDevice;
	// Contact* mContacts;
	std::map<std::string, Contact*> mContacts;
	JointData* mJointDatas;
	MetabolicEnergy* mMetabolicEnergy;

	int mDof;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumAdaptiveDof;
	int mNumAdaptiveSpatialDof;
	int mNumAdaptiveTemporalDof;
	int mAdaptiveLowerDof;
	int mAdaptiveUpperDof;
	int mNumTotalRelatedDof;
	
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
	int mPhaseStateRight;
	int mPhaseStateLeft;
	double mPhaseChangeTimeRight;
	double mPhaseChangeTimeLeft;
	std::deque<std::pair<double, int>> mGaitPhaseRight;
	std::deque<std::pair<double, int>> mGaitPhaseLeft;

	double mTimeOffset;
	double mStride;
	double mStrideL,mStrideR;
	double mStrideCurL;
	double mStrideCurR;
	
	bool mUseDevice;
	bool mUseMuscle;
	bool mOnDevice;
	bool mLowerBody;
	bool mAdaptiveSampling;
	bool mAdaptiveMotion;
	bool mAdaptiveMotionSP;	
	bool mAdaptiveLowerBody;
	bool mIsFirstFoot;

	double mMass;
	double mMassRatio;
	double mMassLower;
	double mForceRatio;
	double mSpeedRatio;

	double mCurCoT;
	double mCurVel;
	double mCurHeadVel;
	Eigen::Vector3d mCurVel3d;
	std::deque<Eigen::Vector4d> mComHistory;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mReferencePositions;
	Eigen::VectorXd mReferenceVelocities;

	Eigen::VectorXd mDesiredTorque;
	Eigen::VectorXd mDesiredTorquePrev;

	Eigen::VectorXd mAction;	
	Eigen::VectorXd mActionPrev;	
	Eigen::VectorXd mActivationLevels;

	Eigen::VectorXd mJointPositionLowerLimits;
	Eigen::VectorXd mJointPositionUpperLimits;
	Eigen::VectorXd mJointWeights;
	Eigen::VectorXd mDefaultMass;
	Eigen::VectorXd mMaxForces;
	Eigen::VectorXd mDefaultForces;

	Eigen::Vector3d mFootL;
	Eigen::Vector3d mFootR;

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
	std::map<std::string, std::pair<double, double>> mAdaptiveParams;
};

}

#endif

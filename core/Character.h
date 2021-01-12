#ifndef __MASS_CHARACTER_H__
#define __MASS_CHARACTER_H__
#include "dart/dart.hpp"
#include <deque>
#include <map>

namespace MASS
{

class BVH;
class Muscle;
class Device;
class Torques;

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
	Device* GetDevice(){return mDevice;}
	BVH* GetBVH(){return mBVH;}

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
	double GetPhase();

	void SetConstraints();
	void RemoveConstraints();

	void SetPDParameters();
	const Eigen::VectorXd& GetPDParameters_Kp(){ return mKp; }
	const Eigen::VectorXd& GetPDParameters_Kv(){ return mKv; }

	Eigen::VectorXd GetState();
	Eigen::VectorXd GetState_Character();
	Eigen::VectorXd GetState_Device();

	void Reset();
	void Reset_Muscles();

	void Step();
	void Step_Muscles(int simCount, int randomSampleIndex);

	double GetReward();
	double GetReward_Character();
	double GetReward_Character_Imitation();
	double GetReward_Character_Efficiency();
	double GetReward_TorqueMin();
	double GetReward_ContactForce();
	double GetReward_Device();

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
	int GetNumState(){return mNumState;}
	int GetNumState_Char(){return mNumState_Char;}
	int GetNumDof(){return mDof;}
	int GetNumActiveDof(){return mNumActiveDof;}
	int GetNumAction(){return mNumActiveDof;}
	int GetRootJointDof(){return mRootJointDof;}
	int GetNumTotalRelatedDofs(){return mNumTotalRelatedDof;}

	void SetUseMuscle(bool b){mUseMuscle = b;}
	bool GetUseMuscle(){return mUseMuscle;}

	void SetUseDevice(bool b){mUseDevice = b;}
	bool GetUseDevice(){return mUseDevice;}

	void SetTorques();
	Torques* GetTorques(){return mTorques;}
	std::deque<double> GetSignals(int idx);

	void SetRewards();
	std::map<std::string, std::deque<double>> GetRewards(){return mRewards;}

	void SetMaxForces();
	Eigen::VectorXd GetMaxForces(){return mMaxForces;}

	double GetForceRatio(){return mForceRatio;}
	void SetForceRatio(double r);

	double GetMassRatio(){return mMassRatio;}
	void SetMassRatio(double r);

	double GetSpeedRatio(){return mSpeedRatio;}
	void SetSpeedRatio(double r);
	void SetBVHidx(double r);

	bool isEdgeTime();

	void SetMeasure();
	void SetCoT();
	void SetCurVelocity();
	void SetContactForce();

	void SetMetabolicEnergyRate();
	double GetMetabolicEnergyRate(){return mMetabolicEnergyRate;}

	double GetCoT(){return mCurCoT;}
	double GetCurVelocity(){return mCurVel;}
	const std::vector<Eigen::Vector3d>& GetContactForces(){return mContactForces;}
	const std::vector<double>& GetContactForcesNormAvg(){return mContactForces_norm;}
	const std::vector<double>& GetContactForcesNorm(){return mContactForces_cur_norm;}

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

	BVH* mBVH;
	BVH* mBVH_;
	std::string mBVHpath;
	std::vector<BVH*> mBVHset;
	std::map<std::string,std::string> mBVHmap;
	bool mBVHcyclic;

	Device* mDevice;
	Torques* mTorques;

	int mDof;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumState;
	int mNumState_Char;
	int mNumTotalRelatedDof;
	int mNumMuscle;
	int mNumBodyNodes;
	int mNumJoints;

	int mControlHz;
	int mSimulationHz;
	int mNumSteps;

	bool mUseDevice;
	bool mUseMuscle;
	bool mOnDevice;

	double mMass;
	double mMassRatio;
	double mForceRatio;
	double mSpeedRatio;

	int mStepCnt;
	int mStepCnt_total;

	double mCurCoT;
	double mCurVel;
	double mMetabolicEnergyRate;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mDesiredTorque;

	Eigen::VectorXd mAction;
	Eigen::VectorXd mActivationLevels;

	Eigen::VectorXd mAngVel, mAngVel_prev;
	Eigen::VectorXd mPos, mPos_prev;
	Eigen::Vector3d mRootPos, mRootPos_prev;

	Eigen::VectorXd mJointWeights;
	Eigen::VectorXd mDefaultMass;
	Eigen::VectorXd mMaxForces;
	Eigen::VectorXd mDefaultForces;

	std::vector<Eigen::Vector3d> mContactForces;
	std::vector<double> mContactForces_cur_norm;
	std::vector<double> mContactForces_norm;

	std::vector<std::deque<double>> mFemurSignals;

	std::vector<std::string> mRewardTags;
	std::map<std::string, double> mReward;
	std::map<std::string, std::deque<double>> mRewards;

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

class Torques
{
public:
	Torques();

	void Initialize(dart::dynamics::SkeletonPtr skel);
	void Reset();
	void SetTorques(const Eigen::VectorXd& desTorques);
	std::vector<std::deque<double>>& GetTorques(){return mTorquesDofs;}

private:
	int mDof;
	std::vector<std::deque<double>> mTorquesDofs;
};

};

#endif

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
	void LoadBVH(const std::string& path,bool cyclic=true);
	void LoadBVHset(double lower, double upper);
	void LoadMuscles(const std::string& path);

	void Initialize();
	void Initialize_Muscles();
	void Initialize_Rewards();
	void Initialize_Forces();
	void Initialize_Mass();
	void Initialize_Speed();
	void Initialize_JointWeights();

	void SetWorld(dart::simulation::WorldPtr& wPtr){ mWorld = wPtr; }
	void SetDevice(Device* device);

	double GetPhase();

	void SetHz(int sHz, int cHz);
	int GetSimulationHz(){ return mSimulationHz; }
	void SetSimulationHz(int hz){ mSimulationHz=hz; }
	int GetControlHz(){ return mControlHz; }
	void SetControlHz(int hz){ mControlHz=hz; }
	int GetNumSteps(){ return mNumSteps; }
	void SetNumSteps(int step){ mNumSteps=step; }

	Eigen::VectorXd GetPDParameters_Kp(){ return mKp; }
	Eigen::VectorXd GetPDParameters_Kv(){ return mKv; }
	void SetPDParameters();

	void Reset();
	void Reset_Muscles();

	Eigen::VectorXd GetState();
	Eigen::VectorXd GetState_Character();
	Eigen::VectorXd GetState_Device();

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
	Eigen::VectorXd GetAction(){ return mAction; }

	void SetDesiredTorques();
	Eigen::VectorXd GetDesiredTorques();

	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired);

	void SetTargetPosAndVel(double t, int controlHz);
	std::pair<Eigen::VectorXd,Eigen::VectorXd> GetTargetPosAndVel(double t,double dt);

	Eigen::VectorXd GetTargetPositions(double t,double dt,int frame,int frameNext, double frameFraction);
	Eigen::VectorXd GetTargetVelocities(double t,double dt,int frame,int frameNext, double frameFraction);

	Eigen::VectorXd GetTargetPositions(){ return mTargetPositions; }
	Eigen::VectorXd GetTargetVelocities(){ return mTargetVelocities; }

	Eigen::VectorXd GetMuscleTorques();
	MuscleTuple& GetCurrentMuscleTuple(){ return mCurrentMuscleTuple; }
	std::vector<MuscleTuple>& GetMuscleTuples(){ return mMuscleTuples; }

	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}

	void SetConstraints();
	void RemoveConstraints();

	void SetDevice_OnOff(bool on);
	bool GetDevice_OnOff(){ return mDevice_On; }
	void SetDevice_On();
	void SetDevice_Off();

	const dart::dynamics::SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<dart::dynamics::BodyNode*>& GetEndEffectors(){return mEndEffectors;}
	Device* GetDevice(){return mDevice;}
	BVH* GetBVH(){return mBVH;}

	Eigen::VectorXd GetMaxForces(){return mMaxForces;}
	void SetMaxForces();

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

	void SetRewards();
	std::map<std::string, std::deque<double>> GetRewards(){return mRewards;}

	double GetForceRatio(){return mForceRatio;}
	void SetForceRatio(double r);

	double GetMassRatio(){return mMassRatio;}
	void SetMassRatio(double r);

	double GetSpeedRatio(){return mSpeedRatio;}
	void SetSpeedRatio(double r);
	void SetBVHidx(double r);

	bool isEdgeTime();
	std::deque<double> GetSignals(int idx);

	std::vector<Eigen::Vector3d> GetContactForces(){return mContactForces; }
	std::vector<double> GetContactForcesNormAvg(){return mContactForces_norm; }
	std::vector<double> GetContactForcesNorm(){return mContactForces_cur_norm; }

	void SetMeasure();

	void SetCoT();
	double GetCoT(){return mCurCoT;}

	void SetContactForce();

	void SetCurVelocity();
	double GetCurVelocity(){return mCurVel;}

	void SetNumParamState(int n);
	void SetMinMaxV(int idx, double lower, double upper);
	void SetAdaptiveParams(std::string name, double lower, double upper);
	void SetAdaptiveParams(std::map<std::string, std::pair<double, double>>& p);
	void SetParamState(Eigen::VectorXd paramState);
	int GetNumParamState(){return mNumParamState;}
	Eigen::VectorXd GetParamState(){return mParamState;}
	Eigen::VectorXd GetMinV(){return mMin_v;}
	Eigen::VectorXd GetMaxV(){return mMax_v;}

private:
	dart::simulation::WorldPtr mWorld;
	dart::dynamics::SkeletonPtr mSkeleton;
	std::vector<dart::dynamics::BodyNode*> mEndEffectors;
	std::vector<Muscle*> mMuscles;
	std::vector<Muscle*> mMuscles_Femur;

	BVH* mBVH;
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
	bool mDevice_On;

	double mMassRatio;
	double mForceRatio;
	double mSpeedRatio;

	int mStepCnt;
	int mStepCnt_total;

	double mCurCoT;
	double mCurVel;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mDesiredTorque;

	Eigen::VectorXd mAction;
	Eigen::VectorXd mActivationLevels;

	Eigen::VectorXd mAngVel;
	Eigen::VectorXd mAngVel_prev;
	Eigen::VectorXd mPos;
	Eigen::VectorXd mPos_prev;
	Eigen::VectorXd mPosTmp;
	Eigen::VectorXd mPosTmp_prev;
	Eigen::Vector3d mRootPos;
	Eigen::Vector3d mRootPos_prev;

	Eigen::VectorXd mJointWeights;
	Eigen::VectorXd mDefaultMass;
	Eigen::VectorXd mMaxForces;
	Eigen::VectorXd mDefaultForces;

	std::vector<Eigen::Vector3d> mContactForces;
	std::vector<double> mContactForces_cur_norm;
	std::vector<double> mContactForces_norm;

	double mContactForceL_norm;
	double mContactForceR_norm;
	double mContactForceL_cur_norm;
	double mContactForceR_cur_norm;

   	std::deque<double> mFemurSignals_L;
    std::deque<double> mFemurSignals_R;

	std::map<std::string, std::deque<double>> mRewards;
    std::deque<double> reward_;
    std::deque<double> pose_;
    std::deque<double> vel_;
    std::deque<double> root_;
    std::deque<double> com_;
    std::deque<double> ee_;
    std::deque<double> smooth_;
    std::deque<double> imit_;
    std::deque<double> min_;
	std::deque<double> contact_;
	std::deque<double> effi_;

    double mReward;
    double com_reward;
    double vel_reward;
    double pose_reward;
    double root_reward;
    double end_eff_reward;
    double smooth_reward;
    double imit_reward;
    double min_reward;
	double contact_reward;
	double effi_reward;

	MuscleTuple mCurrentMuscleTuple;
	std::vector<MuscleTuple> mMuscleTuples;

	dart::constraint::WeldJointConstraintPtr mWeldJoint_Hip;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_LeftLeg;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_RightLeg;

    int mNumParamState;
    Eigen::VectorXd mParamState;
    Eigen::VectorXd mMin_v;
    Eigen::VectorXd mMax_v;
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

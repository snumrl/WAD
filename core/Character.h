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
	Character();
	~Character();

	void LoadSkeleton(const std::string& path,bool create_obj = false);
	void LoadBVH(const std::string& path,bool cyclic=true);
	void LoadBVHset(double lower, double upper);
	void LoadMuscles(const std::string& path);

	void Initialize(dart::simulation::WorldPtr& wPtr, int conHz, int simHz);
	void Initialize_Muscles();
	void Initialize_Rewards();
	void Initialize_MaxForces();
	void Initialize_JointWeights();

	void SetWorld(dart::simulation::WorldPtr& wPtr){ mWorld = wPtr; }
	void SetDevice(Device* device);

	double GetPhase();

	int GetSimulationHz(){ return mSimulationHz; }
	void SetSimulationHz(int hz){ mSimulationHz = hz; }
	int GetControlHz(){ return mControlHz; }
	void SetControlHz(int hz){ mControlHz = hz; }

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
	int GetNumDof(){return mNumDof;}
	int GetNumActiveDof(){return mNumActiveDof;}
	int GetNumAction(){return mNumActiveDof;}
	int GetRootJointDof(){return mRootJointDof;}
	int GetNumTotalRelatedDofs(){return mNumTotalRelatedDof;}

	void SetUseMuscle(bool b);
	bool GetUseMuscle(){return mUseMuscle;}

	void SetTorques();
	Torques* GetTorques(){return mTorques;}

	void SetRewards();
	std::map<std::string, std::deque<double>> GetRewards(){return mRewards;}

	double GetForceRatio(){return force_ratio;}
	void SetForceRatio(double r);

	double GetMassRatio(){return mass_ratio;}
	void SetMassRatio(double r);

	double GetSpeedRatio(){return speed_ratio;}
	void SetSpeedRatio(double r);
	void SetBVHidx(double r);
	void SetBVHset(double lower, double upper);

	bool isEdgeTime();
	std::deque<double> GetSignals(int idx);

	Eigen::Vector3d GetContactForceL(){return mContactForceL;}
	Eigen::Vector3d GetContactForceR(){return mContactForceR;}
	double GetContactForceL_norm(){return mContactForceL_norm;}
	double GetContactForceR_norm(){return mContactForceR_norm;}

	void SetCoT();
	double GetCoT(){return mCurCoT;}

	void SetCollisionForce();

	void SetCurVelocity();
	double GetCurVelocity(){return mCurVel;}

	void SetNumParamState(int n);
	void SetMinMaxV(int idx, double lower, double upper);
	void SetAdaptiveParams(std::string name, double lower, double upper);
	void SetParamState(Eigen::VectorXd paramState);
	int GetNumParamState(){return mNumParamState;}
	Eigen::VectorXd GetParamState(){return mParamState;}
	Eigen::VectorXd GetMinV(){return mMin_v;}
	Eigen::VectorXd GetMaxV(){return mMax_v;}

private:
	dart::dynamics::SkeletonPtr mSkeleton;
	dart::simulation::WorldPtr mWorld;
	std::vector<dart::dynamics::BodyNode*> mEndEffectors;
	std::vector<Muscle*> mMuscles;
	std::vector<Muscle*> mMuscles_Femur;
	Device* mDevice;
	BVH* mBVH;
	std::vector<BVH*> mBVHset;
	std::map<std::string,std::string> bvh_map;
	std::string bvh_path;
	bool bvh_cyclic;

	int mNumDof;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumState;
	int mNumState_Char;
	int mNumTotalRelatedDof;
	int mNumMuscle;
	int mNumBodyNodes;

	int mControlHz;
	int mSimulationHz;

	bool mUseDevice;
	bool mUseMuscle;
	bool mDevice_On;

	Eigen::VectorXd mAction;
	Eigen::VectorXd mActivationLevels;

	double mCurCoT;
	double mCurVel;
	Eigen::VectorXd mAngVel;
	Eigen::VectorXd mAngVel_prev;
	Eigen::VectorXd mPos;
	Eigen::VectorXd mPos_prev;
	Eigen::VectorXd mPosTmp;
	Eigen::VectorXd mPosTmp_prev;
	Eigen::Vector3d mRootPos;
	Eigen::Vector3d mRootPos_prev;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mJointWeights;
	Eigen::VectorXd mDefaultMass;
	Eigen::VectorXd mMaxForces;
	Eigen::VectorXd mDefaultForces;

	Eigen::Vector3d mContactForceL;
	Eigen::Vector3d mContactForceR;
	double mContactForceL_norm;
	double mContactForceR_norm;
	double mContactForceL_cur_norm;
	double mContactForceR_cur_norm;

	double mass_ratio;
	double force_ratio;
	double speed_ratio;

	int mStepCnt;
	int mStepCnt_total;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mDesiredTorque;

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

	Torques* mTorques;

   	std::deque<double> mFemurSignals_L;
    std::deque<double> mFemurSignals_R;

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
	std::vector<std::deque<double>>& GetTorques(){return mTorques_dofs;}

private:
	int num_dofs;
	std::vector<std::deque<double>> mTorques_dofs;
};

};

#endif

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

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::MatrixXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
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

class Character
{
public:
	Character();
	~Character();

	void LoadSkeleton(const std::string& path,bool create_obj = false);
	void LoadBVH(const std::string& path,bool cyclic=true);
	void LoadMuscles(const std::string& path);

	void SetWorld(dart::simulation::WorldPtr& wPtr){ mWorld = wPtr; }
	void Initialize(dart::simulation::WorldPtr& wPtr, int conHz, int simHz);
	void Initialize_Muscles();
	void Initialize_Rewards();
	void Initialize_JointWeights();
	void Initialize_MaxForces();

	Eigen::VectorXd GetKp(){return mKp;}
	Eigen::VectorXd GetKv(){return mKv;}

	int GetSimHz(){return mSimulationHz;}
	int GetConHz(){return mControlHz;}

	void Reset();
	void Reset_Muscles();

	void Step();
	void Step_Muscles(int simCount, int randomSampleIndex);

	Eigen::VectorXd GetState();
	Eigen::VectorXd GetState_Character();
	double GetPhase();
	double GetReward();
	double GetReward_Character();
	double GetTorqueReward();

	void SetAction(const Eigen::VectorXd& a);
	void SetDesiredTorques();
	Eigen::VectorXd GetDesiredTorques();

	void SetPDParameters();
	void SetTargetPosAndVel(double t, int controlHz);

	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired);
	std::pair<Eigen::VectorXd,Eigen::VectorXd> GetTargetPosAndVel(double t,double dt);
	Eigen::VectorXd GetTargetPositions(double t,double dt,int frame,int frameNext, double frameFraction);
	Eigen::VectorXd GetTargetVelocities(double t,double dt,int frame,int frameNext, double frameFraction);
	Eigen::VectorXd GetTargetPositions(){return mTargetPositions;}
	Eigen::VectorXd GetTargetVelocities(){return mTargetVelocities;}

	Eigen::VectorXd GetMuscleTorques();
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	MuscleTuple& GetCurrentMuscleTuple(){return mCurrentMuscleTuple;}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;}

	void SetConstraints();
	void RemoveConstraints();
	void SetDevice(Device* device);
	void SetOnDevice(bool onDevice);
	bool GetOnDevice(){ return mOnDevice; }
	void On_Device();
	void Off_Device();

	const dart::dynamics::SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<dart::dynamics::BodyNode*>& GetEndEffectors(){return mEndEffectors;}
	Device* GetDevice(){return mDevice;}
	BVH* GetBVH(){return mBVH;}
	Eigen::VectorXd GetMaxForces(){return mMaxForces;}

	int GetNumMuscles(){return mNumMuscle;}
	int GetNumState(){return mNumState;}
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

	std::deque<double> GetSignals(int idx);

	void SetCurVelocity();
	double GetCurVelocity(){return mCurVel;}
	double GetCurVelocityH(){return mCurVelH;}

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
	BVH* mBVH;
	Device* mDevice;
	std::vector<Muscle*> mMuscles;

	int mNumDof;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumState;
	int mNumTotalRelatedDof;
	int mNumMuscle;

	int mControlHz;
	int mSimulationHz;

	bool mUseDevice;
	bool mUseMuscle;
	bool mOnDevice;

	Eigen::VectorXd mAction;
	Eigen::VectorXd mAction_prev;
	Eigen::VectorXd mActivationLevels;

	Eigen::VectorXd mAngVel;
	Eigen::VectorXd mAngVel_prev;
	Eigen::Vector3d mCurPos;
	Eigen::Vector3d mPrevPos;
	double mCurVel;
	double mCurVelH;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mJointWeights;
	Eigen::VectorXd mDefaultMass;
	Eigen::VectorXd mMaxForces;
	Eigen::VectorXd mDefaultForces;

	double mass_ratio;
	double force_ratio;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mTargetPositionsNoise;
	Eigen::VectorXd mTargetVelocitiesNoise;
	Eigen::VectorXd mDesiredTorque;
	Eigen::VectorXd mDesiredTorque_prev;

	double w_q,w_v,w_ee,w_com,w_character;
	double r_q,r_v,r_ee,r_com,r_character;

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

    double mReward;
    double com_reward;
    double vel_reward;
    double pose_reward;
    double root_reward;
    double end_eff_reward;
    double smooth_reward;
    double imit_reward;
    double min_reward;

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

};

#endif

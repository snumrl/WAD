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
	Eigen::VectorXd GetMaxForces(){return maxForces;}

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

	std::deque<double> GetSignals(int idx);

private:
	dart::dynamics::SkeletonPtr mSkeleton;
	dart::simulation::WorldPtr mWorld;
	BVH* mBVH;
	Device* mDevice;
	std::vector<Muscle*> mMuscles;
	std::vector<dart::dynamics::BodyNode*> mEndEffectors;

	Torques* mTorques;
	double mTotalTorques;

	int mNumDof;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumState;
	int mNumTotalRelatedDof;
	int mNumMuscle;

	bool mUseDevice;
	bool mUseMuscle;
	bool mOnDevice;

	int mControlHz;
	int mSimulationHz;

	Eigen::VectorXd mAction;
	Eigen::VectorXd mAction_prev;
	Eigen::VectorXd mActivationLevels;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;
	Eigen::VectorXd maxForces;
	Eigen::VectorXd mJointWeights;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;

	Eigen::VectorXd mTargetPositionsNoise;
	Eigen::VectorXd mTargetVelocitiesNoise;

	Eigen::VectorXd mDesiredTorque;
	Eigen::VectorXd mDesiredTorque_prev;

	std::deque<double> mFemurSignals_L;
    std::deque<double> mFemurSignals_R;

	double w_q,w_v,w_ee,w_com,w_character;
	double r_q,r_v,r_ee,r_com,r_character;

	std::map<std::string, std::deque<double>> mRewards;
    std::deque<double> reward_;
    std::deque<double> pose_;
    std::deque<double> vel_;
    std::deque<double> root_;
    std::deque<double> com_;
    std::deque<double> ee_;
    std::deque<double> imit_;
    std::deque<double> min_;

    double mReward;
    double pose_reward;
    double vel_reward;
    double end_eff_reward;
    double root_reward;
    double com_reward;
    double imit_reward;
    double min_reward;

    double force_ratio;
    double mass_ratio;

	MuscleTuple mCurrentMuscleTuple;
	std::vector<MuscleTuple> mMuscleTuples;

	dart::constraint::WeldJointConstraintPtr mWeldJoint_Hip;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_LeftLeg;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_RightLeg;
};

};

#endif

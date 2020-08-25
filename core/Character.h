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

	void Init(dart::dynamics::SkeletonPtr skel);
	void Reset();
	void Set();
	void SetTorque(int dof, int phase, double val);
	double GetTorque(int dof, int phase);
	std::vector<double>& GetTorquesCur(){return mTorques_cur;}
	std::vector<double>& GetTorquesAvg(){return mTorques_avg;}
	std::vector<std::vector<double>>& GetTorquesDofsCur(){return mTorques_dofs_cur;}
	std::vector<std::vector<double>>& GetTorquesDofsAvg(){return mTorques_dofs_cur;}

private:
	int num_dofs;
	int num_phase;
	std::vector<double> mTorques_cur;
	std::vector<double> mTorques_avg;
	std::vector<std::vector<double>> mTorques_dofs_cur;
	std::vector<std::vector<double>> mTorques_dofs_avg;
	std::vector<std::vector<int>> mTorques_dofs_num;
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
	void Initialize_Analysis();

	Eigen::VectorXd GetKp(){return mKp;}
	Eigen::VectorXd GetKv(){return mKv;}

	void SetKp(double kp);
	void SetKv(double kv);

	void Reset();
	void Reset_Muscles();

	void Step();
	void Step_Muscles(int simCount, int randomSampleIndex);

	Eigen::VectorXd GetState();
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
	// Eigen::VectorXd GetTargetPositions(double t,double dt);
	Eigen::VectorXd GetTargetPositions(double t,double dt,int frame,int frameNext, double frameFraction);
	Eigen::VectorXd GetTargetVelocities(double t,double dt,int frame,int frameNext, double frameFraction);

	Eigen::VectorXd GetTargetPositions(){return mTargetPositions;}
	Eigen::VectorXd GetTargetVelocities(){return mTargetVelocities;}

	Eigen::VectorXd GetMuscleTorques();
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	MuscleTuple& GetCurrentMuscleTuple(){return mCurrentMuscleTuple;}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;}

	void SetDevice(Device* device);
	void SetConstraints();
	void SetOnDevice(bool onDevice);
	bool GetOnDevice(){ return mOnDevice; }
	void On_Device();
	void Off_Device();

	const dart::dynamics::SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<dart::dynamics::BodyNode*>& GetEndEffectors(){return mEndEffectors;}
	Device* GetDevice(){return mDevice;}
	BVH* GetBVH(){return mBVH;}
	Torques* GetTorques(){return mTorques;}

	int GetNumMuscles(){return mNumMuscle;}
	int GetNumState(){return mNumState;}
	int GetNumDof(){return mNumDof;}
	int GetNumActiveDof(){return mNumActiveDof;}
	int GetNumAction(){return mNumActiveDof;}
	int GetRootJointDof(){return mRootJointDof;}
	int GetNumTotalRelatedDofs(){return mNumTotalRelatedDof;}

	void SetUseMuscle(bool b);
	void SetPhase();
	void SetTorques();
	// void SetEnergy();
	void SetReward_Graph();

	bool GetUseMuscle(){return mUseMuscle;}
	double GetPhase(){return mPhase;}
	// std::map<std::string, std::vector<double>> GetEnergy(int idx);
	std::vector<double> GetReward_Graph(int idx);
	std::deque<double> GetSignals(int idx);
	std::map<std::string, std::deque<double>> GetRewardMap(){return mReward_map;}

	void get_record();
	std::vector<double> getFemurLavg(){return mFemur_L_Avg;}
	std::vector<double> getFemurRavg(){return mFemur_R_Avg;}

	Eigen::VectorXd GetPoseSlerp(double timeStep, double frameFraction, const Eigen::VectorXd& frameData, const Eigen::VectorXd& frameDataNext);

private:
	dart::dynamics::SkeletonPtr mSkeleton;
	dart::simulation::WorldPtr mWorld;
	BVH* mBVH;
	Device* mDevice;
	std::vector<Muscle*> mMuscles;
	std::vector<dart::dynamics::BodyNode*> mEndEffectors;

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

	double w_q,w_v,w_ee,w_com,w_character;
	double r_q,r_v,r_ee,r_com,r_character;
	double mReward;
	double mPhase;

	Eigen::VectorXd mAction;
	Eigen::VectorXd mAction_prev;
	Eigen::VectorXd mActivationLevels;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;
	Eigen::VectorXd maxForces;
	Eigen::VectorXd mJointWeights;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mDesiredTorque;

	double r_torque_min;
	std::vector<double> mRewards;
	std::vector<int> mRewards_num;
	std::vector<double> mRewards_Device;
	std::vector<int> mRewards_Device_num;

	std::deque<double> mFemurSignals_L;
    std::deque<double> mFemurSignals_R;

    std::vector<double> mFemur_L_Avg;
    std::vector<double> mFemur_R_Avg;

    std::map<std::string, std::deque<double>> mReward_map;

    std::deque<double> pose_;
    std::deque<double> vel_;
    std::deque<double> root_;
    std::deque<double> com_;
    std::deque<double> ee_;

    double pose_reward = 0;
    double vel_reward = 0;
    double end_eff_reward = 0;
    double root_reward = 0;
    double com_reward = 0;

	Torques* mTorques;

	MuscleTuple mCurrentMuscleTuple;
	std::vector<MuscleTuple> mMuscleTuples;

	dart::constraint::WeldJointConstraintPtr mWeldJoint_Hip;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_LeftLeg;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_RightLeg;
};

};

#endif

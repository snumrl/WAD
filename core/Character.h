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

class Energy
{
public:
	Energy();

	void Init(dart::dynamics::SkeletonPtr skel);
	void Reset();
	void SetEnergy(std::string name, int t, double val);
	double GetEnergy(std::string name, int t);
	std::map<std::string, std::vector<double>>& Get(){return mE;}
	std::map<std::string, std::vector<int>>& GetNum(){return mE_num;}

private:
	std::map<std::string, std::vector<double>> mE;
	std::map<std::string, std::vector<int>> mE_num;	
};

class Character
{
public:
	Character();

	void LoadSkeleton(const std::string& path,bool create_obj = false);
	void LoadBVH(const std::string& path,bool cyclic=true);
	void LoadMuscles(const std::string& path);
	void LoadDevice(const std::string& path);

	void Initialize();
	void Initialize_Debug();
	void Initialize_Muscles();
	void Initialize_Device(dart::simulation::WorldPtr& wPtr);

	void On_Device(dart::simulation::WorldPtr& wPtr);
	void Off_Device(dart::simulation::WorldPtr& wPtr);

	void Reset(double worldTime, int controlHz);
	void Reset_Muscles();
	void Reset_Device();

	Eigen::VectorXd GetState(double worldTime);
	Eigen::VectorXd GetState_Device(double worldTime);

	double GetReward();
	double GetReward_Character();
	double GetReward_Device();
	std::map<std::string,double> GetRewardSep();

	void Step();
	void Step_Muscles(int simCount, int randomSampleIndex);
	void Step_Device(double t);
	void Step_Device(const Eigen::VectorXd& a_);
	
	void SetOnDevice(bool onDevice){mOnDevice = onDevice;}

	void SetAction(const Eigen::VectorXd& a);
	void SetAction_Device(const Eigen::VectorXd& a);

	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com);
	void SetRewardParameters_Device();
	void SetPDParameters(double kp, double kv);

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetDesiredTorques_Device(double t);
	Eigen::VectorXd GetMuscleTorques();

	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired);

	void SetTargetPosAndVel(double t, int controlHz);
	std::pair<Eigen::VectorXd,Eigen::VectorXd> GetTargetPosAndVel(double t,double dt);
	Eigen::VectorXd GetTargetPositions(double t,double dt);
	Eigen::VectorXd GetTargetPositions(){return mTargetPositions;}

	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	MuscleTuple& GetCurrentMuscleTuple(){return mCurrentMuscleTuple;}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;}

	const dart::dynamics::SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<dart::dynamics::BodyNode*>& GetEndEffectors(){return mEndEffectors;}
	Device* GetDevice(){return mDevice;}
	BVH* GetBVH(){return mBVH;}

	int GetNumState(){return mNumState;}
	int GetNumActiveDof(){return mNumActiveDof;}
	int GetRootJointDof(){return mRootJointDof;}
	int GetNumTotalRelatedDof(){return mNumTotalRelatedDof;}

	std::deque<double> GetDeviceSignals(int idx);
	Eigen::VectorXd GetDeviceForce(){return mDeviceForce;}

	double GetPhase(){return mPhase;}
	void SetEnergy();
	void SetRewards();
	std::map<std::string, std::vector<double>> GetEnergy(int idx);
	std::vector<double> GetReward_Graph(int idx);
	
public:
	dart::dynamics::SkeletonPtr mSkeleton;
	BVH* mBVH;
	Device* mDevice;
	std::vector<Muscle*> mMuscles;
	std::vector<dart::dynamics::BodyNode*> mEndEffectors;
	
	int mNumState;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumTotalRelatedDof;

	bool mUseDevice;
	bool mUseMuscle;
	bool mOnDevice;

	double mTorqueMax_Device;

	double w_q,w_v,w_ee,w_com,w_character,w_device;
	double r_q,r_v,r_ee,r_com,r_character,r_device;
	double mReward;
	double mPhase = 0.0;

	Eigen::VectorXd mAction_;
	Eigen::VectorXd mAction_Device;
	Eigen::VectorXd mActivationLevels;

	Eigen::Isometry3d mTc;
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;
	Eigen::VectorXd mDesiredTorque;
	Eigen::VectorXd mDesiredTorque_Device;
	
	Eigen::VectorXd mDeviceForce;
	std::deque<double> mDeviceSignals_L;
	std::deque<double> mDeviceSignals_R;
	std::deque<double> mFemurForce_R;
	
	std::vector<double> mRewards;
	std::vector<int> mRewards_num;
	std::vector<double> mRewards_Device;
	std::vector<int> mRewards_Device_num;
	
	Energy* mEnergy;
	Energy* mEnergy_Device;

	MuscleTuple mCurrentMuscleTuple;
	std::vector<MuscleTuple> mMuscleTuples;
	
	dart::constraint::WeldJointConstraintPtr mWeldJoint_Hip;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_LeftLeg;
    dart::constraint::WeldJointConstraintPtr mWeldJoint_RightLeg;
};

};

#endif

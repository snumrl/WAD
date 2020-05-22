#ifndef __MASS_CHARACTER_H__
#define __MASS_CHARACTER_H__
#include "dart/dart.hpp"

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

class Character
{
public:
	Character();

	void LoadSkeleton(const std::string& path,bool create_obj = false);
	void LoadBVH(const std::string& path,bool cyclic=true);
	void LoadMuscles(const std::string& path);
	void LoadDevice(const std::string& path);

	void Initialize();
	void Initialize_Muscles();
	void Initialize_Device();

	void Reset(double worldTime, int controlHz);
	void Reset_Muscles();
	void Reset_Device();

	void Clone();
	void Clone_Back();
	void Print();

	Eigen::VectorXd GetState(double worldTime);
	Eigen::VectorXd GetState_Device(double worldTime);

	double GetReward();
	double GetReward_Device();
	std::map<std::string,double> GetRewardSep();

	void Step();
	void Step_Muscles(int simCount, int randomSampleIndex);
	void Step_Device();
	void Step_Device(const Eigen::VectorXd& a);

	void SetAction(const Eigen::VectorXd& a);
	void SetAction_Device(const Eigen::VectorXd& a);

	void SetRewardCharacterOnly(double r){r_character_only = r;}

	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com);
	void SetRewardParameters_Device();
	void SetPDParameters(double kp, double kv);

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetDesiredTorques_Device();
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

	// void AddEndEffector(const std::string& body_name){mEndEffectors.push_back(mSkeleton->getBodyNode(body_name));}

public:
	dart::dynamics::SkeletonPtr mSkeleton; // clone
	BVH* mBVH;
	Device* mDevice; // clone
	std::vector<Muscle*> mMuscles; //haveto clone
	std::vector<dart::dynamics::BodyNode*> mEndEffectors; // clone

	std::string skel_path;

	int mNumState;
	int mNumActiveDof;
	int mRootJointDof;
	int mNumTotalRelatedDof;

	bool mUseDevice;
	bool mUseMuscle;

	double mTorqueMax_Device;

	double w_q,w_v,w_ee,w_com,w_character,w_device;
	double r_q,r_v,r_ee,r_com,r_character,r_device;
	double r_character_only = 0.0;	

	Eigen::VectorXd mAction_;
	Eigen::VectorXd mAction_Device;
	Eigen::VectorXd mActivationLevels;

	Eigen::Isometry3d mTc; //haveto clone
	Eigen::VectorXd mKp, mKv;

	Eigen::VectorXd mTargetPositions; // check
	Eigen::VectorXd mTargetVelocities; // check
	Eigen::VectorXd mDesiredTorque;  // check
	Eigen::VectorXd mDesiredTorque_Device; // check

	MuscleTuple mCurrentMuscleTuple; // have to clone
	std::vector<MuscleTuple> mMuscleTuples; // have to clone

	dart::constraint::WeldJointConstraintPtr mWeldJoint_Hip; // clone
    dart::constraint::WeldJointConstraintPtr mWeldJoint_LeftLeg; // clone
    dart::constraint::WeldJointConstraintPtr mWeldJoint_RightLeg; // clone

    // clone
    std::vector<Muscle> mMuscles_clone;
    Eigen::Isometry3d mTc_clone;
	MuscleTuple mCurrentMuscleTuple_clone; // have to clone
	std::vector<MuscleTuple> mMuscleTuples_clone; // have to clone

	std::map<std::string,std::string> bvh_map;
};

};

#endif

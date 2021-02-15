#include "Character.h"
#include "BVH.h"
#include "DARTHelper.h"
#include "Muscle.h"
#include "Device.h"
#include "Utils.h"
#include "dart/gui/gui.hpp"
#include <tinyxml.h>
#include <ctime>
using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

Character::
Character(dart::simulation::WorldPtr& wPtr)
	:mSkeleton(nullptr),mBVH(nullptr),mDevice(nullptr),mTc(Eigen::Isometry3d::Identity()),mUseMuscle(false),mUseDevice(false),mOnDevice(false),mNumParamState(0),mMass(0)
{
	this->SetWorld(wPtr);

	mMassRatio = 1.0;
	mForceRatio = 1.0;
	mSpeedRatio = 1.0;

	mLowerBody = true;
}

Character::
~Character()
{
	for(int i=0; i<mEndEffectors.size(); i++)
		delete(mEndEffectors[i]);

	for(int i=0; i<mMuscles.size(); i++)
		delete(mMuscles[i]);

	delete mBVH;
	delete mDevice;
}

void
Character::
LoadSkeleton(const std::string& path, bool load_obj)
{
	mSkeleton = BuildFromFile(path, load_obj, mMassRatio);

	TiXmlDocument doc;
	doc.LoadFile(path);
	TiXmlElement* skel_elem = doc.FirstChildElement("Skeleton");
	for(TiXmlElement* node = skel_elem->FirstChildElement("Node"); node != nullptr; node = node->NextSiblingElement("Node"))
	{
		if(node->Attribute("endeffector") != nullptr)
		{
			std::string ee = node->Attribute("endeffector");
			if(ee == "True"){
				mEndEffectors.push_back(mSkeleton->getBodyNode(std::string(node->Attribute("name"))));
			}
		}

		TiXmlElement* joint_elem = node->FirstChildElement("Joint");
		if(joint_elem->Attribute("bvh") != nullptr)
			mBVHmap.insert(std::make_pair(node->Attribute("name"),joint_elem->Attribute("bvh")));
	}
}

void
Character::
LoadBVH(const std::string& path, bool cyclic)
{
	if(path == ""){
		std::cout<<"BVH path is NULL"<<std::endl;
		return;
	}

	mBVHpath = path;
	mBVHcyclic = cyclic;
	mBVH = new BVH(mSkeleton, mBVHmap);
	mBVH->SetSpeedRatio(mSpeedRatio);
	mBVH->Parse(mBVHpath, mBVHcyclic);
	mBVH_ = mBVH;
}

void
Character::
LoadMuscles(const std::string& path)
{
	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open Muscle file : " << path << std::endl;
		return;
	}

	std::string mFileName = "Femur_related_muscle";
	std::ofstream mFile;
	mFile.open(mFileName);

	bool isExist = false;
	double time_step = 1.0/mSimulationHz;
	TiXmlElement *muscledoc = doc.FirstChildElement("Muscle");
	for(TiXmlElement* unit = muscledoc->FirstChildElement("Unit");unit!=nullptr;unit = unit->NextSiblingElement("Unit"))
	{
		std::string name = unit->Attribute("name");
		double f0 = std::stod(unit->Attribute("f0"));
		double lm = std::stod(unit->Attribute("lm"));
		double lt = std::stod(unit->Attribute("lt"));
		double pa = std::stod(unit->Attribute("pen_angle"));
		double lmax = std::stod(unit->Attribute("lmax"));

		isExist = true;
		if(mLowerBody)
		{
			for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))
			{
			std::string body = waypoint->Attribute("body");
			if(body == "ShoulderL" || body == "ArmL" || body == "ForeArmL" || body == "HandL")
				isExist = false;
			if(body == "ShoulderR" || body == "ArmR" || body == "ForeArmR" || body == "HandR")
				isExist = false;
			if(body == "Head" || body == "Neck")
				isExist = false;
			}
		}

		if(isExist)
		{
			Muscle* muscle_elem = new Muscle(name,f0,lm,lt,pa,lmax);

			int num_waypoints = 0;
			for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))
				num_waypoints++;

			int i = 0;
			for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))
			{
				std::string body = waypoint->Attribute("body");
				Eigen::Vector3d glob_pos = string_to_vector3d(waypoint->Attribute("p"));
				if(mSkeleton->getBodyNode(body) == NULL)
				{
					std::cout << body << " node is NULL" << std::endl;
					return;
				}

				if(body == "FemurL" || body == "FemurR"){
					muscle_elem->SetFemur(true);
					mFile << name + "\n";
				}

				if(i == 0 || i == num_waypoints-1)
					muscle_elem->AddAnchor(mSkeleton->getBodyNode(body),glob_pos);
				else
					muscle_elem->AddAnchor(mSkeleton,mSkeleton->getBodyNode(body),glob_pos,2);

				i++;
			}
			muscle_elem->SetMt0Default();
			muscle_elem->SetTimeStep(time_step);
			muscle_elem->SetMass();

			std::string muscle_name = muscle_elem->GetName();
			int name_size = muscle_name.size();
			std::string sub_name;
			if(muscle_name[name_size-1] >= 48 && muscle_name[name_size-1] <= 57){
				std::string sub_name = muscle_name.substr(2,name_size-3);
				mMuscles_Map[sub_name].push_back(muscle_elem);
			}
			else{
				std::string sub_name = muscle_name.substr(2,name_size-2);
				if(mMuscles_Map.count(sub_name) == 0){
					mMuscles_Map.insert(std::make_pair(sub_name, std::vector<Muscle*>()));
					mMuscles_Map[sub_name].push_back(muscle_elem);
				}
				else{
					mMuscles_Map[sub_name].push_back(muscle_elem);
				}
			}

			if(muscle_elem->GetFemur())
			{
				muscle_elem->SetMt0Ratio(1.0);
				muscle_elem->SetF0Ratio(1.0);
				mMuscles_Femur.push_back(muscle_elem);
			}

			mMuscles.push_back(muscle_elem);
		}
	}
	mNumMuscle = mMuscles.size();
	mNumMuscleMap = mMuscles_Map.size();
	mFile.close();
}

void
Character::
Initialize()
{
	if(mSkeleton == nullptr) {
		std::cout<<"Initialize Character First"<<std::endl;
		exit(0);
	}

	mWorld->addSkeleton(mSkeleton);

	mDof = mSkeleton->getNumDofs();
	mNumBodyNodes = mSkeleton->getNumBodyNodes();
	mNumJoints = mSkeleton->getNumJoints();

	mRootJointDof = 6;
	mNumActiveDof = mDof - mRootJointDof;
	mNumState_Char = this->GetState().rows();
	mNumState = mNumState_Char;

	mTc = mBVH->GetT0();
	mTc.translation()[1] = 0.0;

	mStepCnt = 0;
	mStepCnt_total = 0;

	mNumAdaptiveDof = 18;
	mNumAction = mNumActiveDof + mNumAdaptiveDof;
	mAction = Eigen::VectorXd::Zero(mNumAction);
	mDesiredTorque = Eigen::VectorXd::Zero(mDof);

	this->SetPDParameters();
	this->SetTargetPosAndVel(mWorld->getTime());

	mAngVel = Eigen::VectorXd::Zero(mNumBodyNodes*3);
	mAngVel_prev = Eigen::VectorXd::Zero(mNumBodyNodes*3);

	mPos = Eigen::VectorXd::Zero(mNumBodyNodes*3);
	mPos_prev = Eigen::VectorXd::Zero(mNumBodyNodes*3);

	this->Initialize_JointWeights();
	this->Initialize_Rewards();
	this->Initialize_Forces();
	this->Initialize_Mass();
	if(mUseMuscle)
		this->Initialize_Muscles();

	mCurVel = 0.0;
	mCurCoT = 0.0;
	mStepCnt = 0;

	mFemurSignals.push_back(std::deque<double>(1200));
	mFemurSignals.push_back(std::deque<double>(1200));

	int frames = mBVH->GetNumTotalFrames();
	double ratio = 1.0;
	if(mLowerBody)
		ratio = 52.8/74.2;
	mMetabolicEnergy = new MetabolicEnergy(mWorld);
	mMetabolicEnergy->Initialize(this->GetMuscles(), mMass, mNumSteps, frames, ratio);

	mJointDatas = new JointData();
	mJointDatas->Initialize(mSkeleton);

	mContacts = new Contact(mWorld);
	mContacts->Initialize(mSkeleton, mMass, mNumSteps);
	mContacts->SetContactObject("TalusL");
	mContacts->SetContactObject("TalusR");

	this->Reset();
}

void
Character::
Initialize_JointWeights()
{
	mJointWeights.resize(mNumJoints);
	mJointWeights <<
		0.5,                    //Pelvis
		0.5, 0.3, 0.2, 0.2, 0.2,//Left Leg
		0.5, 0.3, 0.2, 0.2, 0.2,//Right Leg
		0.5, 0.3, 0.2, 0.2,     //Torso & Neck
		0.5, 0.3, 0.2, 0.1,     //Left Arm
		0.5, 0.3, 0.2, 0.1;     //Right Arm

	mJointWeights /= mJointWeights.sum();
}

void
Character::
Initialize_Mass()
{
	mDefaultMass = Eigen::VectorXd::Zero(mNumBodyNodes);
	for(int i=0; i<mNumBodyNodes; i++){
		mDefaultMass[i] = mSkeleton->getBodyNode(i)->getMass();
		mMass += mDefaultMass[i];
	}
}

void
Character::
Initialize_Forces()
{
	mMaxForces.resize(mDof);
	mDefaultForces.resize(mDof);
	mDefaultForces <<
		 0, 0, 0, 0, 0, 0,   //pelvis
		 300, 300, 300,      //Femur L
		 300,                //Tibia L
		 300, 300, 300,      //Talus L
		 300, 300,           //Thumb, Pinky L
		 300, 300, 300,      //Femur R
		 300,                //Tibia R
		 300, 300, 300,      //Talus R
		 300, 300,           //Thumb, Pinky R
		 300, 300, 300,      //Spine
		 300, 300, 300,      //Torso
		 300, 300, 300,      //Neck
		 300, 300, 300,      //Head
		 300, 300, 300,      //Shoulder L
		 300, 300, 300,      //Arm L
		 300,                //ForeArm L
		 300, 300, 300,      //Hand L
		 300, 300, 300,      //Shoulder R
		 300, 300, 300,      //Arm R
		 300,                //ForeArm R
		 300, 300, 300;      //Hand R

	// mDefaultForces <<
	//      0, 0, 0, 0, 0, 0,   //pelvis
	//      200, 100, 150,      //Femur L
	//      100,                //Tibia L
	//      150, 50, 50,        //Talus L
	//      30, 30,             //Thumb, Pinky L
	//      200, 100, 150,      //Femur R
	//      100,                //Tibia R
	//      150, 50, 50,        //Talus R
	//      30, 30,             //Thumb, Pinky R
	//      80, 80, 80,         //Spine
	//      80, 80, 80,         //Torso
	//      30, 30, 30,         //Neck
	//      30, 30, 30,         //Head
	//      50, 50, 50,         //Shoulder L
	//      50, 50, 50,         //Arm L
	//      30,                 //ForeArm L
	//      30, 30, 30,         //Hand L
	//      50, 50, 50,         //Shoulder R
	//      50, 50, 50,         //Arm R
	//      30,                 //ForeArm R
	//      30, 30, 30;         //Hand R

	mMaxForces = mForceRatio * mDefaultForces;
}

void
Character::
Initialize_Muscles()
{
	mNumTotalRelatedDof = 0;
	for(auto m : this->GetMuscles()){
		m->Update();
		mNumTotalRelatedDof += m->GetNumRelatedDofs();
	}

	mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(mNumTotalRelatedDof);
	mCurrentMuscleTuple.L = Eigen::MatrixXd::Zero(mNumActiveDof, mNumMuscle);
	mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof);
	mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof);
	mActivationLevels = Eigen::VectorXd::Zero(mNumMuscle);
}

void
Character::
Initialize_Rewards()
{
	mRewardTags.push_back("reward");
	mRewardTags.push_back("pose");
	mRewardTags.push_back("vel");
	mRewardTags.push_back("root");
	mRewardTags.push_back("ee");
	mRewardTags.push_back("com");
	mRewardTags.push_back("smooth");
	mRewardTags.push_back("min");
	mRewardTags.push_back("contact");
	mRewardTags.push_back("reg");
	mRewardTags.push_back("effi");

	int reward_window = 70;
	for(auto tag : mRewardTags){
		mReward.insert(std::make_pair(tag, 0.0));
		mRewards.insert(std::make_pair(tag, std::deque<double>(reward_window)));
	}
}

void
Character::
SetRewards()
{
	for(auto tag : mRewardTags){
		mRewards[tag].pop_back();
		mRewards[tag].push_front(mReward[tag]);
	}
}

void
Character::
SetHz(int sHz, int cHz)
{
	mSimulationHz = sHz;
	mControlHz = cHz;
	this->SetNumSteps(mSimulationHz/mControlHz);
}

void
Character::
SetPDParameters()
{
	mKp.resize(mDof);
	mKv.resize(mDof);

	mKp << 0, 0, 0, 0, 0, 0,
		500, 500, 500,
		500,
		400, 400, 400,
		100, 100,
		500, 500, 500,
		500,
		400, 400, 400,
		100, 100,
		1000, 1000, 1000,
		500, 500, 500,
		100, 100, 100,
		100, 100, 100,
		400, 400, 400,
		300, 300, 300,
		300,
		100, 100, 100,
		400, 400, 400,
		300, 300, 300,
		300,
		100, 100, 100;

	mKv << 0, 0, 0, 0, 0, 0,
		50, 50, 50,
		50,
		40, 40, 40,
		10, 10,
		50, 50, 50,
		50,
		40, 40, 40,
		10, 10,
		100, 100, 100,
		50, 50, 50,
		10, 10, 10,
		10, 10, 10,
		40, 40, 40,
		30, 30, 30,
		30,
		10, 10, 10,
		40, 40, 40,
		30, 30, 30,
		30,
		10, 10, 10;
}

void
Character::
Reset()
{
	mSkeleton->clearConstraintImpulses();
	mSkeleton->clearInternalForces();
	mSkeleton->clearExternalForces();

	double worldTime = mWorld->getTime();
	this->SetPhase();
	this->SetTargetPosAndVel(worldTime);

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true,false,false);

	mAction.setZero();
	mDesiredTorque.setZero();

	mRootPos = mSkeleton->getCOM();
	mRootPos_prev = mSkeleton->getCOM();

	mAngVel.setZero();
	mPos.setZero();
	for(int i=0; i<mNumBodyNodes; i++){
		mAngVel_prev.segment<3>(i*3) = mSkeleton->getBodyNode(i)->getAngularVelocity();
		mPos_prev.segment<3>(i*3) = mSkeleton->getBodyNode(i)->getCOM();
	}

	for(int i=0; i<mFemurSignals.at(0).size(); i++){
		mFemurSignals.at(0).at(i) = 0.0;
		mFemurSignals.at(1).at(i) = 0.0;
	}//Initialze L,R at the same time

	mStepCnt = 0;
	mCurCoT = 0.0;
	mCurVel = 0.0;
	mCurVel3d.setZero();

	mMetabolicEnergy->Reset();
	mJointDatas->Reset();
	if(mUseMuscle)
		Reset_Muscles();

	this->GetReward();
}

void
Character::
Reset_Muscles()
{
	for(auto m : mMuscles)
		m->Reset();

	(mCurrentMuscleTuple.JtA).setZero();
	(mCurrentMuscleTuple.L).setZero();
	(mCurrentMuscleTuple.b).setZero();
	(mCurrentMuscleTuple.tau_des).setZero();
	mActivationLevels.setZero();
}

void
Character::
Step(bool isRender)
{
	SetDesiredTorques();
	mSkeleton->setForces(mDesiredTorque);

	if(mStepCnt == mNumSteps)
		mStepCnt = 0;
	mStepCnt++;
	mStepCnt_total++;

	if(isRender)
		this->SetMeasure();
}

void
Character::
Step_Muscles(int simCount, int randomSampleIndex, bool isRender)
{
	int count = 0;
	for(auto muscle : mMuscles)
	{
		muscle->SetActivation(mActivationLevels[count++]);
		muscle->Update();
		muscle->ApplyForceToBody();
	}

	if(simCount == randomSampleIndex)
	{
		int m = mMuscles.size();
		Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(mDof,m);
		Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(mDof);

		for(int i=0; i<mMuscles.size(); i++)
		{
			auto muscle = mMuscles[i];
			// muscle->Update();
			Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
			auto Ap = muscle->GetForceJacobianAndPassive();

			JtA.block(0,i,mDof,1) = Jt*Ap.first;
			Jtp += Jt*Ap.second;
		}

		mCurrentMuscleTuple.JtA = GetMuscleTorques();
		mCurrentMuscleTuple.L =JtA.block(mRootJointDof,0,mDof-mRootJointDof,m);
		mCurrentMuscleTuple.b =Jtp.segment(mRootJointDof,mDof-mRootJointDof);
		if(mLowerBody){
			Eigen::VectorXd tmp = mDesiredTorque;
			tmp.tail<26>().setZero();
			mCurrentMuscleTuple.tau_des = tmp.tail(tmp.rows()-mRootJointDof);
		}
		else
			mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
		mMuscleTuples.push_back(mCurrentMuscleTuple);
	}

	if(mLowerBody){
		mDesiredTorque.head<30>().setZero();
		mSkeleton->setForces(mDesiredTorque);
	}

	if(mStepCnt == mNumSteps)
		mStepCnt = 0;
	mStepCnt++;
	mStepCnt_total++;

	this->SetCurVelocity();
	mMetabolicEnergy->Set(this->GetMuscles(), mCurVel3d, this->GetPhase(), (int)mCurFrame);
	if(isRender)
		this->SetMeasure();
}

void
Character::
SetMeasure()
{
	mJointDatas->SetTorques(mDesiredTorque);
	mJointDatas->SetAngles((int)mCurFrame);
	mContacts->Set();
	// this->SetCoT();
}

void
Character::
SetCoT()
{
	Eigen::VectorXd vel = mSkeleton->getVelocities();
	Eigen::VectorXd tor = mDesiredTorque;
	// double vel_tor = vel.dot(tor);
	double vel_tor = 0.0;

	int idx = 6;
	for(int i=1; i<mNumJoints; i++)
	{
		auto* joint = mSkeleton->getJoint(i);
		if(joint->getType()=="RevoluteJoint"){
			double cur = vel[idx] * tor[idx];
			vel_tor += fabs(cur);
			idx += 1;
		}
		else if(joint->getType()=="BallJoint"){
			double cur = vel.segment(idx,3).dot(tor.segment(idx,3));
			vel_tor += fabs(cur);
			idx += 3;
		}
	}

	// vel_tor = vel.dot(tor);

	double g = 9.8;
	double v = mCurVel;
	double m = 0.0;
	for(int i=0; i<mNumBodyNodes; i++)
	{
		m += mSkeleton->getBodyNode(i)->getMass();
	}

	if(v == 0){
		mCurCoT = 0;
		return;
	}

	mCurCoT *= (mStepCnt_total-1);
	mCurCoT += vel_tor / (m * g * v);
	mCurCoT /= mStepCnt_total;
}

void
Character::
SetCurVelocity()
{
	mRootPos = mSkeleton->getCOM();
	double time_step = mSkeleton->getTimeStep();
	double x_diff = (mRootPos[0]-mRootPos_prev[0])/time_step;
	double z_diff = (mRootPos[2]-mRootPos_prev[2])/time_step;
	mRootPos_prev = mRootPos;
	mCurVel = std::sqrt(x_diff*x_diff + z_diff*z_diff);

	mCurVel3d = mSkeleton->getCOMLinearVelocity();
	// mCurVel = mCurVel3d.norm();

}

Eigen::VectorXd
Character::
GetState()
{
	int state_dim = 0;

	Eigen::VectorXd state_character = this->GetState_Character();
	state_dim += state_character.rows();

	Eigen::VectorXd state(state_dim);
	state << state_character;
	if(mUseDevice)
	{
		Eigen::VectorXd state_device;
		state_device = mDevice->GetState();
		state_dim += state_device.rows();
		state.resize(state_dim);
		state << state_character, state_device;
	}

	return state;
}

Eigen::VectorXd
Character::
GetState_Character()
{
	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	double h = root->getCOM()[1];
	double w = root->getCOM()[0];

	Eigen::VectorXd p,v;
	p.resize((mNumBodyNodes-1)*3);
	v.resize((mNumBodyNodes)*3);

	for(int i=1; i<mNumBodyNodes; i++)
	{
		p.segment<3>(3*(i-1)) = mSkeleton->getBodyNode(i)->getCOM() - root->getCOM();
		v.segment<3>(3*(i-1)) = mSkeleton->getBodyNode(i)->getCOMLinearVelocity() - root->getCOMLinearVelocity();
	}
	v.tail<3>() = root->getCOMLinearVelocity();

	p *= 0.8;
	v *= 0.2;

	std::pair<double, double> phase = this->GetPhases();

	Eigen::VectorXd state;
	state.resize(2 + p.rows() + v.rows() + 2);
	state << h, w, p, v, phase.first, phase.second;

	return state;
}

double
Character::
GetReward()
{
	double reward_character = this->GetReward_Character();
	double reward = reward_character + 0;
	mReward["reward"] = reward;
	mCurReward =reward;

	this->SetRewards();

	return reward;
}

double
Character::
GetReward_Character()
{
	double reward_imit = GetReward_Character_Imitation();
	double reward_effi = GetReward_Character_Efficiency();

 	mReward["imit"] = reward_imit;
 	mReward["effi"] = reward_effi;

	double r = 1.0 * reward_imit + 0.1 * reward_effi;

	return r;
}

double
Character::
GetReward_Character_Imitation()
{
	std::vector<Eigen::Matrix3d> cur_ee_r, ref_ee_r;
	std::vector<Eigen::Vector3d> cur_ee_p, ref_ee_p;

	Eigen::VectorXd cur_q = mSkeleton->getPositions();
	Eigen::VectorXd cur_dq = mSkeleton->getVelocities();
	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);

	// Current Position and Velocity
	Eigen::VectorXd p_cur, v_cur;
	p_cur.resize((mNumBodyNodes-1)*3);
	v_cur.resize((mNumBodyNodes-1)*3);
	for (int i=1; i<mNumBodyNodes; i++)
	{
		dart::dynamics::BodyNode* cur_body = mSkeleton->getBodyNode(i);
		p_cur.segment<3>(3*(i-1)) = cur_body->getCOM(root);
		v_cur.segment<3>(3*(i-1)) = cur_body->getCOMLinearVelocity(root);
	}

	auto ees = this->GetEndEffectors();
	for(auto ee : ees){
		cur_ee_r.push_back(ee->getTransform().linear());
		cur_ee_p.push_back(ee->getTransform().translation());
	}

	// Target Position and Velocity
	// mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setPositions(mAdaptiveTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true, true, false);

	Eigen::VectorXd p_ref, v_ref;
	p_ref.resize((mNumBodyNodes-1)*3);
	v_ref.resize((mNumBodyNodes-1)*3);
	for (int i=1; i<mNumBodyNodes; i++)
	{
		dart::dynamics::BodyNode* cur_body = mSkeleton->getBodyNode(i);
		p_ref.segment<3>(3*(i-1)) = cur_body->getCOM(root);
		v_ref.segment<3>(3*(i-1)) = cur_body->getCOMLinearVelocity(root);
	}

	ees = this->GetEndEffectors();
	for(auto ee : ees){
		ref_ee_r.push_back(ee->getTransform().linear());
		ref_ee_p.push_back(ee->getTransform().translation());
	}

	mSkeleton->setPositions(cur_q);
	mSkeleton->setVelocities(cur_dq);
	mSkeleton->computeForwardKinematics(true, true, false);

	// Position and Velocity Difference
	Eigen::VectorXd p_diff, v_diff;
	p_diff.resize((mNumBodyNodes-1)*3);
	v_diff.resize((mNumBodyNodes-1)*3);

	p_diff = (p_cur - p_ref);
	v_diff = (v_cur - v_ref);

	// p_diff[0] = root->getCOM()[0];
	// v_diff = (v_cur - v_ref);

	// Endeffector
	Eigen::VectorXd ee_rot_diff(ees.size());
	Eigen::VectorXd ee_pos_diff(ees.size());
	for(int i = 0; i < ees.size(); i++)
	{
		double w = 0.0;
		if(ees[i]->getName() == "Head")
			w = 0.3;
		if(ees[i]->getName() == "Pelvis")
			w = 0.3;
		if(ees[i]->getName() == "TalusR" || ees[i]->getName() == "TalusL")
			w = 0.3;
		if(ees[i]->getName() == "HandR" || ees[i]->getName() == "HandL")
			w = 0.1;

		ee_rot_diff[i] = w * Eigen::AngleAxisd(ref_ee_r[i].inverse() * cur_ee_r[i]).angle();
		ee_pos_diff[i] = w * (ref_ee_p[i] - cur_ee_p[i]).norm();
	}

	//Angle Difference
	// Eigen::VectorXd q_diff = mSkeleton->getPositionDifferences(cur_q, mTargetPositions);
	Eigen::VectorXd q_diff = mSkeleton->getPositionDifferences(cur_q, mAdaptiveTargetPositions);

	//=====================================

	double sig_p = 4.0;
	double sig_q = 0.4;
	double sig_ee_rot = 20.0;
	double sig_ee_pos = 40.0;

	double r_p = Utils::exp_of_squared(p_diff, sig_p);
	double r_q = Utils::exp_of_squared(q_diff, sig_q);
	double r_ee_rot = Utils::exp_of_squared(ee_rot_diff, sig_ee_rot);
	double r_ee_pos = Utils::exp_of_squared(ee_pos_diff, sig_ee_pos);

	double r_total = r_p * r_q * r_ee_rot * r_ee_pos;

	// std::cout << "p : " << r_p << std::endl;
	// std::cout << "q : " << r_q << std::endl;
	// std::cout << "rot : " << r_ee_rot << std::endl;
	// std::cout << "pos : " << r_ee_pos << std::endl;

	mReward["pose"] = r_p;
	mReward["ee"] = r_ee_rot;
	mReward["smooth"] = r_ee_pos;
	mReward["vel"] = r_q;

	return r_total;
}


double
Character::
GetReward_Character_Efficiency()
{
	double r_EnergyMin = 1.0;
	if(mUseMuscle)
		r_EnergyMin = mMetabolicEnergy->GetReward();
	// else
	// 	r_EnergyMin = this->GetReward_TorqueMin();

	double r_ContactForce = 1.0;
	// double r_ContactForce = mContacts->GetReward();

	double r_ActionReg = 1.0;
	r_ActionReg = this->GetReward_ActionReg();

	mReward["min"] = r_EnergyMin;
	mReward["contact"] = r_ContactForce;
	mReward["reg"] = r_ActionReg;

	double r = r_EnergyMin + r_ActionReg;
	return r;
}

double
Character::
GetReward_ActionReg()
{
	double actionNorm = mAction.segment(50, mNumAdaptiveDof).norm();

	double err_scale = 1.0;
	double actionReg_scale = 2.0;
	double actionReg_err = 0.0;

	actionReg_err = actionNorm;

	double reward = 0.0;
	reward = exp(-err_scale * actionReg_scale * actionReg_err);

	return reward;
}

double
Character::
GetReward_TorqueMin()
{

	// std::vector<std::deque<double>> ts = mTorques->GetTorques();
	// int idx = 0;
	// double sum = 0.0;
	// for(int i=6; i<mMaxForces.size(); i++)
	// {
	// 	double ratio = fabs(ts[i].at(0))/mMaxForces[i];
	// 	if(ratio > 0.4)
	// 		sum += ratio;
	// 	idx++;
	// }
	// // for(int i=6; i<mMaxForces.size(); i++)
	// // {
	// //  if(fabs(ts[i].at(0)) > 0.4*mMaxForces[i])
	// //      sum += 1.0;
	// //  idx++;
	// // }
	// sum /= (double)(idx);

	// return -10.0 * sum;
	return 0;
}

void
Character::
SetPhases()
{
	double rad = 2*M_PI*mPhase;
	double cos = std::cos(rad);
	double sin = std::sin(rad);

	mPhases = std::pair<double, double>(cos, sin);
}

void
Character::
SetPhase()
{
	double t = mWorld->getTime();
	double cycleTime = mBVH->GetMaxTime();
	int cycleCount = (int)(t/cycleTime);
	double phase = t;
	if(mBVH->IsCyclic())
		phase = (t - cycleCount*cycleTime)/cycleTime;
	if(phase < 0)
		phase += (cycleTime)/cycleTime;

	mPhase = phase;

	this->SetPhases();
}

void
Character::
SetAction(const Eigen::VectorXd& a)
{
	double action_scale = 0.1;
	mAction = a*action_scale;
	mAction.segment(50,mNumAdaptiveDof) *= 0.2;

	double t = mWorld->getTime();
	this->SetTargetPosAndVel(t);
}

void
Character::
SetDesiredTorques()
{
	// Eigen::VectorXd p_des = mTargetPositions;
	// p_des.tail(mTargetPositions.rows() - mRootJointDof) += mAction.segment(0,50);
	Eigen::VectorXd p_des = mAdaptiveTargetPositions;
	p_des.tail(mAdaptiveTargetPositions.rows() - mRootJointDof) += mAction.segment(0,50);
	mDesiredTorque = this->GetSPDForces(p_des);

	for(int i=0; i<mDesiredTorque.size(); i++){
		mDesiredTorque[i] = Utils::Clamp(mDesiredTorque[i], -mMaxForces[i], mMaxForces[i]);
	}

	mFemurSignals.at(0).pop_back();
	mFemurSignals.at(0).push_front(mDesiredTorque[6]);

	mFemurSignals.at(1).pop_back();
	mFemurSignals.at(1).push_front(mDesiredTorque[15]);
}

Eigen::VectorXd
Character::
GetDesiredTorques()
{
	if(mLowerBody){
		Eigen::VectorXd tmp = mDesiredTorque;
		tmp.tail<26>().setZero();
		return tmp.tail(tmp.rows()-mRootJointDof);
	}
	else
		return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}

Eigen::VectorXd
Character::
GetSPDForces(const Eigen::VectorXd& p_desired)
{
	Eigen::VectorXd q = mSkeleton->getPositions();
	Eigen::VectorXd dq = mSkeleton->getVelocities();
	double dt = mSkeleton->getTimeStep();

	Eigen::MatrixXf M = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt * mKv.asDiagonal())).cast<float>();
	Eigen::MatrixXd M_inv = M.inverse().cast<double>();

	Eigen::VectorXd qdqdt = q + dq*dt;
	Eigen::VectorXd p_diff = -mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt,p_desired));
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq);
	Eigen::VectorXd ddq = M_inv*(-mSkeleton->getCoriolisAndGravityForces() + p_diff + v_diff + mSkeleton->getConstraintForces());
	Eigen::VectorXd tau = p_diff + v_diff - dt*mKv.cwiseProduct(ddq);

	tau.head<6>().setZero();
	return tau;
}

void
Character::
SetTargetPosAndVel(double t)
{
	double frameTime = t;
	double dt = 1.0/(double)mControlHz;

	if(mBVH->IsCyclic()){
		double cycleTime = mBVH->GetMaxTime();
		int cycleCount = (int)(t/cycleTime);

		frameTime = t - cycleCount*cycleTime;
		if(frameTime < 0)
			frameTime += cycleTime;
	}

	int frame = (int)(frameTime/dt);
	int frameNext = frame + 1;

	if(mBVH->IsCyclic()){
		if(frameNext >= mBVH->GetNumTotalFrames())
			frameNext = frame;
	}
	else{
		if(frameNext > 941){
			frameNext = 941;
			frame = 941;
		}
	}

	double frameFraction = (frameTime - frame*dt)/dt;
	mCurFrame = frame + frameFraction;

	this->SetTargetPositions(t,dt,frame,frameNext,frameFraction);
	this->SetTargetVelocities(t,dt,frame,frameNext,frameFraction);
}

void
Character::
SetTargetPositions(double t,double dt,int frame,int frameNext, double frameFraction)
{
	Eigen::VectorXd frameData, frameDataNext;
	if(mBVH->IsCyclic()){
		frameData = mBVH->GetMotion(frame);
		frameDataNext = mBVH->GetMotion(frameNext);
	}
	else{
		frameData = mBVH->GetMotionNonCyclic(frame);
		frameDataNext = mBVH->GetMotionNonCyclic(frameNext);
	}

	// Eigen::VectorXd p = Utils::GetPoseSlerp(mSkeleton, frameFraction, frameData, frameDataNext);
	mTargetPositions = (1-frameFraction)*frameData + (frameFraction)* frameDataNext;

	if(mBVH->IsCyclic()) {
		double cycleTime = mBVH->GetMaxTime();
		int cycleCount = (int)(t/cycleTime);

		Eigen::Vector3d cycleOffset = mBVH->GetCycleOffset();
		cycleOffset[1] = 0.0;
		mTargetPositions.segment(3,3) += cycleCount*cycleOffset;
	}

	mAdaptiveTargetPositions = mTargetPositions;
	// mTargetPositions.segment(6,mNumAdaptiveDof) += mAction.segment(50,mNumAdaptiveDof);
	mAdaptiveTargetPositions.segment(6,mNumAdaptiveDof) += mAction.segment(50,mNumAdaptiveDof);

	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(mTargetPositions.head<6>());
	T_current = mBVH->GetT0().inverse()*T_current;
	Eigen::Isometry3d T_head = mTc*T_current;
	Eigen::Vector6d p_head = dart::dynamics::FreeJoint::convertToPositions(T_head);
	mTargetPositions.head<6>() = p_head;
	mAdaptiveTargetPositions.head<6>() = p_head;
}

void
Character::
SetTargetVelocities(double t,double dt,int frame,int frameNext, double frameFraction)
{
	Eigen::VectorXd frameVel, frameNextVel;
	if(mBVH->IsCyclic()){
		frameVel = mBVH->GetMotionVel(frame);
		frameNextVel = mBVH->GetMotionVel(frameNext);
	}
	else{
		frameVel = mBVH->GetMotionVelNonCyclic(frame);
		frameNextVel = mBVH->GetMotionVelNonCyclic(frameNext);
	}

	mTargetVelocities = (1-frameFraction)*frameVel + (frameFraction)*frameNextVel;
}

Eigen::VectorXd
Character::
GetMuscleTorques()
{
	int index = 0;
	mCurrentMuscleTuple.JtA.setZero();
	for(auto muscle : mMuscles)
	{
		muscle->Update();
		Eigen::VectorXd JtA_i = muscle->GetRelatedJtA();
		mCurrentMuscleTuple.JtA.segment(index, JtA_i.rows()) = JtA_i;
		index += JtA_i.rows();
	}

	return mCurrentMuscleTuple.JtA;
}

std::deque<double>
Character::
GetSignals(int idx)
{
	return mFemurSignals.at(idx);
}

void
Character::
SetDevice(Device* device)
{
	mDevice = device;
	mOnDevice = true;

	mNumState += mDevice->GetState().rows();
}

void
Character::
SetConstraints()
{
	mWeldJoint_Hip = std::make_shared<dart::constraint::WeldJointConstraint>(
		mSkeleton->getBodyNode(0), mDevice->GetSkeleton()->getBodyNode(0)
		);

	mWeldJoint_LeftLeg = std::make_shared<dart::constraint::WeldJointConstraint>(
		mSkeleton->getBodyNode("FemurL"), mDevice->GetSkeleton()->getBodyNode("RodLeft")
		);

	mWeldJoint_RightLeg = std::make_shared<dart::constraint::WeldJointConstraint>(
		mSkeleton->getBodyNode("FemurR"), mDevice->GetSkeleton()->getBodyNode("RodRight")
		);

	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_Hip);
	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_LeftLeg);
	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_RightLeg);
}

void
Character::
RemoveConstraints()
{
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_Hip);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_LeftLeg);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_RightLeg);
}

void
Character::
SetDevice_OnOff(bool on)
{
	if(on^mOnDevice)
	{
		if(on)
			this->SetDevice_On();
		else
			this->SetDevice_Off();
	}

	mOnDevice = on;
}

void
Character::
SetDevice_On()
{
	mDevice->Reset();

	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_Hip);
	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_LeftLeg);
	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_RightLeg);
	mWorld->addSkeleton(mDevice->GetSkeleton());
}

void
Character::
SetDevice_Off()
{
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_Hip);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_LeftLeg);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_RightLeg);
	mWorld->removeSkeleton(mDevice->GetSkeleton());
}

void
Character::
SetMassRatio(double r)
{
	mMassRatio = r;
	this->SetMass();

	double param = 0.0;
	if(mParamMax[0] == mParamMin[0])
	{
		mParamState[0] = mParamMin[0];
	}
	else
	{
		double ratio = (mMassRatio-mParamMin[0])/(mParamMax[0]-mParamMin[0]);
		param = ratio*2.0 - 1.0;
		mParamState[0] = param;
	}
}

void
Character::
SetMass()
{
	mMass = 0;
	for(int i=0; i<mNumBodyNodes; i++)
	{
		dart::dynamics::BodyNode* body = mSkeleton->getBodyNode(i);
		dart::dynamics::Inertia inertia;
		auto shape = body->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get();
		double mass = mMassRatio * mDefaultMass[i];
		inertia.setMass(mass);
		inertia.setMoment(shape->computeInertia(mass));
		body->setInertia(inertia);
		mMass += mass;
	}

	mMetabolicEnergy->SetMass(mMass);
	mContacts->SetMass(mMass);
}

void
Character::
SetForceRatio(double r)
{
	mForceRatio = r;

	if(mUseMuscle)
	{
		for(int i=0; i<mMuscles.size(); i++)
			mMuscles.at(i)->SetF0Ratio(mForceRatio);
		// for(int i=0; i<mMuscles_Femur.size(); i++)
		// 	mMuscles_Femur.at(i)->SetF0Ratio(mForceRatio);
	}
	else
	{
		mMaxForces = mForceRatio * mDefaultForces;
	}

	double param = 0.0;
	if(mParamMax[1] == mParamMin[1])
	{
		mParamState[1] = mParamMin[1];
	}
	else
	{
		double ratio = (mForceRatio-mParamMin[1])/(mParamMax[1]-mParamMin[1]);
		param = ratio*2.0 - 1.0;
		mParamState[1] = param;
	}
}

void
Character::
SetSpeedRatio(double r)
{
	mSpeedRatio = r;

	double param = 0.0;
	if(mParamMax[2] == mParamMin[2])
	{
		mParamState[2] = mParamMin[2];
	}
	else
	{
		double ratio = (r-mParamMin[2])/(mParamMax[2]-mParamMin[2]);
		param = ratio*2.0 - 1.0;
		mParamState[2] = param;
	}

	// std::cout << "speedratio : " << mSpeedRatio << std::endl;
	// std::cout << "param : " << mParamState[2] << std::endl;
}

void
Character::
LoadBVHset(double lower, double upper)
{
	if(lower == upper)
	{
		if(!mBVH->IsParsed()){
			mBVH->SetSpeedRatio(lower);
			mBVH->Parse(mBVHpath, mBVHcyclic);
		}
		mBVHset.push_back(mBVH);
		return;
	}

	double sr = lower;
	while(sr <= upper)
	{
		if(sr == mSpeedRatio){
			mBVHset.push_back(mBVH_);
		}
		else{
			BVH* newBVH = new BVH(mSkeleton, mBVHmap);
			newBVH->SetSpeedRatio(sr);
			newBVH->Parse(mBVHpath, mBVHcyclic);
			mBVHset.push_back(newBVH);
		}
		sr += 0.1;
	}

	mBVH = mBVHset.at(mBVHset.size()-1);
}

void
Character::
SetBVHidx(double r)
{
	double speed_min = mParamMin[2];
	double tmp = r-speed_min;
	int idx = (int)(tmp/0.1);
	mBVH = mBVHset.at(idx);

	this->Reset();
	if(mUseDevice)
		mDevice->Reset();
}

void
Character::
SetNumParamState(int n)
{
	mNumParamState = n;
	mParamState = Eigen::VectorXd::Zero(mNumParamState);
	mParamMin = Eigen::VectorXd::Zero(mNumParamState);
	mParamMax = Eigen::VectorXd::Zero(mNumParamState);
}

void
Character::
SetParamState(Eigen::VectorXd paramState)
{
	mParamState = paramState;
	double param = 0.0;
	for(int i=0; i<paramState.size(); i++)
	{
		param = paramState[i];
		param = mParamMin[i]+(mParamMax[i]-mParamMin[i])*(param+1.0)/2.0;
		if(i==0) // Mass
			this->SetMassRatio(param);
		else if(i==1) // Force
			this->SetForceRatio(param);
		else if(i==2){ // Speed
			this->SetSpeedRatio(this->SetSpeedIdx(param) + mParamMin[2]);
			this->SetBVHidx(this->SetSpeedIdx(param) + mParamMin[2]);
		}
	}
}

double
Character::
SetSpeedIdx(double s)
{
	double speed_min = mParamMin[2];
	double diff = s-speed_min;
	int idx = (int)(diff/0.1);

	return idx * 0.1;
}

void
Character::
SetMinMaxV(int idx, double lower, double upper)
{
	// 0 : mass // 1 : force // 2 : speed
	mParamMin[idx] = lower;
	mParamMax[idx] = upper;
}

void
Character::
SetAdaptiveParams(std::map<std::string, std::pair<double, double>>& p)
{
	for(auto p_ : p){
		std::string name = p_.first;
		double lower = (p_.second).first;
		double upper = (p_.second).second;

		if(name == "mass"){
			this->SetMinMaxV(0, lower, upper);
			this->SetMassRatio(lower);
		}
		else if(name == "force"){
			this->SetMinMaxV(1, lower, upper);
			this->SetForceRatio(upper);
		}
		else if(name == "speed"){
			if(lower!=upper)
				this->SetMinMaxV(2, lower, upper+0.0999);
			else
				this->SetMinMaxV(2, lower, upper);
			this->LoadBVHset(lower, upper);
			this->SetSpeedRatio(upper);
		}
	}
}

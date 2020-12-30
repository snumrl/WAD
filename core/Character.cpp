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
	:mSkeleton(nullptr),mBVH(nullptr),mDevice(nullptr),mTc(Eigen::Isometry3d::Identity()),mUseMuscle(false),mUseDevice(false),mDevice_On(false),mNumParamState(0),mMass(0)
{
	this->SetWorld(wPtr);

	mMassRatio = 1.0;
	mForceRatio = 1.0;
	mSpeedRatio = 1.0;
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
	delete mTorques;
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
			if(ee == "True")
				mEndEffectors.push_back(mSkeleton->getBodyNode(std::string(node->Attribute("name"))));
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
	mBVH->Parse(mBVHpath, mBVHcyclic);
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

	TiXmlElement *muscledoc = doc.FirstChildElement("Muscle");
	for(TiXmlElement* unit = muscledoc->FirstChildElement("Unit");unit!=nullptr;unit = unit->NextSiblingElement("Unit"))
	{
		std::string name = unit->Attribute("name");
		double f0 = std::stod(unit->Attribute("f0"));
		double lm = std::stod(unit->Attribute("lm"));
		double lt = std::stod(unit->Attribute("lt"));
		double pa = std::stod(unit->Attribute("pen_angle"));
		double lmax = std::stod(unit->Attribute("lmax"));
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

			if(body == "FemurL" || body == "FemurR")
				muscle_elem->SetFemur(true);
			// if(body == "FemurL")
			//  muscle_elem->SetFemur(true);

			if(i == 0 || i == num_waypoints-1)
				muscle_elem->AddAnchor(mSkeleton->getBodyNode(body),glob_pos);
			else
				muscle_elem->AddAnchor(mSkeleton,mSkeleton->getBodyNode(body),glob_pos,2);

			muscle_elem->SetMt0Default();
			i++;
		}

		std::string muscle_name = muscle_elem->GetName();

		if(muscle_elem->GetFemur())
		{
			muscle_elem->SetMt0Ratio(1.0);
			muscle_elem->SetF0Ratio(1.0);
			mMuscles_Femur.push_back(muscle_elem);
		}

		mMuscles.push_back(muscle_elem);
	}
	mNumMuscle = mMuscles.size();
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

	this->SetPDParameters();
	this->SetTargetPosAndVel(0, mControlHz);

	mRootJointDof = 6;
	mNumActiveDof = mDof - mRootJointDof;
	mNumState_Char = this->GetState().rows();
	mNumState = mNumState_Char;

	mStepCnt = 0;
	mStepCnt_total = 0;

	mTc = mBVH->GetT0();
	mTc.translation()[1] = 0.0;

	mAction = Eigen::VectorXd::Zero(mNumActiveDof);
	mDesiredTorque = Eigen::VectorXd::Zero(mDof);

	mAngVel = Eigen::VectorXd::Zero(mNumBodyNodes*3);
	mAngVel_prev = Eigen::VectorXd::Zero(mNumBodyNodes*3);

	mPos = Eigen::VectorXd::Zero(mNumBodyNodes*3);
	mPos_prev = Eigen::VectorXd::Zero(mNumBodyNodes*3);

	mFemurSignals.push_back(std::deque<double>(1200));
	mFemurSignals.push_back(std::deque<double>(1200));
	mContactForces.push_back(Eigen::Vector3d::Zero());
	mContactForces.push_back(Eigen::Vector3d::Zero());
	mContactForces_norm.push_back(0.0);
	mContactForces_norm.push_back(0.0);
	mContactForces_cur_norm.push_back(0.0);
	mContactForces_cur_norm.push_back(0.0);

	mTorques = new Torques();
	mTorques->Initialize(mSkeleton);

	this->Initialize_JointWeights();
	this->Initialize_Rewards();
	this->Initialize_Forces();
	this->Initialize_Mass();
	if(mUseMuscle)
		this->Initialize_Muscles();

	this->Reset();
}

void
Character::
Initialize_JointWeights()
{
	mJointWeights.resize(mNumJoints);

	mJointWeights <<
		1.0,                    //Pelvis
		0.5, 0.3, 0.2, 0.1, 0.1,//Left Leg
		0.5, 0.3, 0.2, 0.1, 0.1,//Right Leg
		0.5, 0.5, 0.2, 0.2,     //Torso & Neck
		0.3, 0.2, 0.2, 0.1,     //Left Arm
		0.3, 0.2, 0.2, 0.1;     //Right Arm

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
	mReward = 0;
	pose_reward = 0;
	vel_reward = 0;
	end_eff_reward = 0;
	root_reward = 0;
	com_reward = 0;
	smooth_reward = 0;
	contact_reward = 0;
	imit_reward = 0;
	effi_reward = 0;
	min_reward = 0;

	int reward_window = 70;

	reward_ = std::deque<double>(reward_window);
	pose_ = std::deque<double>(reward_window);
	vel_ = std::deque<double>(reward_window);
	root_ = std::deque<double>(reward_window);
	ee_ = std::deque<double>(reward_window);
	com_ = std::deque<double>(reward_window);
	smooth_ = std::deque<double>(reward_window);
	imit_ = std::deque<double>(reward_window);
	min_ = std::deque<double>(reward_window);
	contact_ = std::deque<double>(reward_window);
	effi_ = std::deque<double>(reward_window);

	mRewards.insert(std::make_pair("reward", reward_));
	mRewards.insert(std::make_pair("pose", pose_));
	mRewards.insert(std::make_pair("vel", vel_));
	mRewards.insert(std::make_pair("root", root_));
	mRewards.insert(std::make_pair("ee", ee_));
	mRewards.insert(std::make_pair("com", com_));
	mRewards.insert(std::make_pair("smooth", smooth_));
	mRewards.insert(std::make_pair("imit", imit_));
	mRewards.insert(std::make_pair("min", min_));
	mRewards.insert(std::make_pair("contact", contact_));
	mRewards.insert(std::make_pair("effi", effi_));
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
	this->SetTargetPosAndVel(worldTime, mControlHz);

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true,false,false);

	mAction.setZero();
	mDesiredTorque.setZero();

	mAngVel.setZero();
	mPos.setZero();

	for(int i=0; i<mNumBodyNodes; i++){
		mAngVel_prev.segment<3>(i*3) = mSkeleton->getBodyNode(i)->getAngularVelocity();
		mPos_prev.segment<3>(i*3) = mSkeleton->getBodyNode(i)->getCOM();
	}

	mRootPos = mSkeleton->getCOM();
	mRootPos_prev = mSkeleton->getCOM();

	for(int i=0; i<mFemurSignals.at(0).size(); i++){
		mFemurSignals.at(0).at(i) = 0.0;
		mFemurSignals.at(1).at(i) = 0.0;
	}//Initialze L,R at the same time

	mContactForces.at(0).setZero();
	mContactForces.at(1).setZero();

	mCurVel = 0.0;
	mCurCoT = 0.0;
	mStepCnt = 0;

	mTorques->Reset();
	if(mUseMuscle)
		Reset_Muscles();
}

void
Character::
Reset_Muscles()
{
	(mCurrentMuscleTuple.JtA).setZero();
	(mCurrentMuscleTuple.L).setZero();
	(mCurrentMuscleTuple.b).setZero();
	(mCurrentMuscleTuple.tau_des).setZero();
	mActivationLevels.setZero();
}

void
Character::
Step()
{
	SetDesiredTorques();
	mSkeleton->setForces(mDesiredTorque);

	if(mStepCnt == 20)
		mStepCnt = 0;
	mStepCnt++;
	mStepCnt_total++;

	this->SetMeasure();
}

void
Character::
Step_Muscles(int simCount, int randomSampleIndex)
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
		mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
		mMuscleTuples.push_back(mCurrentMuscleTuple);
	}

	this->SetMeasure();
}

void
Character::
SetMeasure()
{
	this->SetContactForce();
	this->SetCoT();
	this->SetCurVelocity();
	this->SetTorques();
}

void
Character::
SetContactForce()
{
	if(mStepCnt == 1){
		for(int i=0; i<mContactForces.size();i++)
			mContactForces.at(i).setZero();
		for(int i=0; i<mContactForces_cur_norm.size();i++)
			mContactForces_cur_norm.at(i) = 0.0;
	}

	double forceL = 0;
	double forceR = 0;
	dart::dynamics::BodyNode* bn_TalusL = mSkeleton->getBodyNode("TalusL");
	dart::dynamics::BodyNode* bn_TalusR = mSkeleton->getBodyNode("TalusR");
	const dart::collision::CollisionResult& result = mWorld->getLastCollisionResult();
	for(const auto& contact : result.getContacts())
	{
		for(const auto& shapeNode : bn_TalusL->getShapeNodesWith<dart::dynamics::CollisionAspect>())
		{
			if(shapeNode == contact.collisionObject1->getShapeFrame() ||
				shapeNode == contact.collisionObject2->getShapeFrame())
			{
				mContactForces.at(0) += contact.force;
				forceL += contact.force.norm();
			}
		}

		for(const auto& shapeNode : bn_TalusR->getShapeNodesWith<dart::dynamics::CollisionAspect>())
		{
			if(shapeNode == contact.collisionObject1->getShapeFrame() ||
				shapeNode == contact.collisionObject2->getShapeFrame())
			{
				mContactForces.at(1) += contact.force;
				forceR += contact.force.norm();
			}
		}
	}

	mContactForces_cur_norm.at(0) += forceL;
	mContactForces_cur_norm.at(1) += forceR;

	mContactForces_norm.at(0) *= (mStepCnt_total-1);
	mContactForces_norm.at(0) += forceL;
	mContactForces_norm.at(0) /= mStepCnt_total;

	mContactForces_norm.at(1) *= (mStepCnt_total-1);
	mContactForces_norm.at(1) += forceR;
	mContactForces_norm.at(1) /= mStepCnt_total;
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

	// mCurVel = std::sqrt(x_diff*x_diff + z_diff*z_diff);
	mCurVel *= (mStepCnt_total-1);
	mCurVel += std::sqrt(x_diff*x_diff + z_diff*z_diff);
	mCurVel /= mStepCnt_total;

	mRootPos_prev = mRootPos;
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
		state_device = this->GetState_Device();
		state_dim += state_device.rows();
		state.resize(state_dim);
		state << state_character, state_device;
	}

	return state;
}

Eigen::VectorXd
Character::
GetState_Device()
{
	return mDevice->GetState();
}

Eigen::VectorXd
Character::
GetState_Character()
{
	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

	Eigen::Isometry3d origin_trans = Utils::GetOriginTrans(mSkeleton);
	Eigen::Quaterniond origin_quat(origin_trans.rotation());
	Utils::QuatNormalize(origin_quat);

	Eigen::VectorXd pos,ori,lin_v,ang_v;
	Eigen::Vector4d talus_l, talus_r;

	Eigen::Vector4d talus_1 = {-0.0425, -0.03, -0.075, 1.0};
	Eigen::Vector4d talus_2 = {-0.0425, -0.03,  0.075, 1.0};
	Eigen::Vector4d talus_3 = { 0.0425, -0.03,  0.075, 1.0};
	Eigen::Vector4d talus_4 = { 0.0425, -0.03, -0.075, 1.0};

	Eigen::Isometry3d tr_l = mSkeleton->getBodyNode("TalusL")->getTransform();
	Eigen::Isometry3d tr_r = mSkeleton->getBodyNode("TalusR")->getTransform();
	talus_l << (tr_l*talus_1)[1],(tr_l*talus_2)[1],(tr_l*talus_3)[1],(tr_l*talus_4)[1];
	talus_r << (tr_r*talus_1)[1],(tr_r*talus_2)[1],(tr_r*talus_3)[1],(tr_r*talus_4)[1];

	pos.resize(mNumBodyNodes*3+1); //3dof + root world y
	ori.resize(mNumBodyNodes*4);   //4dof (quaternion)
	lin_v.resize(mNumBodyNodes*3);
	ang_v.resize(mNumBodyNodes*3); //dof - root_dof

	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	Eigen::Isometry3d trans = Utils::GetBodyTransform(root);
	Eigen::Vector3d root_pos = trans.translation();
	Eigen::Vector3d root_pos_rel = root_pos;

	root_pos_rel = Utils::AffineTransPoint(origin_trans, root_pos_rel);
	pos(0) = root_pos_rel[1];
	int idx_pos = 1;
	int idx_ori = 0;
	int idx_linv = 0;
	int idx_angv = 0;
	for(int i=0; i<mNumBodyNodes; i++)
	{
		dart::dynamics::BodyNode* body = mSkeleton->getBodyNode(i);
		trans = Utils::GetBodyTransform(body);

		Eigen::Vector3d body_pos = trans.translation();
		body_pos = Utils::AffineTransPoint(origin_trans, body_pos);
		body_pos -= root_pos_rel;
		pos.segment(idx_pos, 3) = body_pos.segment(0, 3);
		idx_pos += 3;

		Eigen::Quaterniond body_ori(trans.rotation());
		body_ori = origin_quat * body_ori;
		Utils::QuatNormalize(body_ori);
		ori.segment(idx_ori, 4) = Utils::QuatToVec(body_ori).segment(0, 4);
		idx_ori += 4;

		Eigen::Vector3d lin_vel = body->getLinearVelocity();
		lin_vel = Utils::AffineTransVector(origin_trans, lin_vel);
		lin_v.segment(idx_linv, 3) = lin_vel;
		idx_linv += 3;

		Eigen::Vector3d ang_vel = body->getAngularVelocity();
		ang_vel = Utils::AffineTransVector(origin_trans, ang_vel);
		ang_v.segment(idx_angv, 3) = ang_vel;
		idx_angv += 3;
	}

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true, false, false);

	Eigen::VectorXd pos_diff,ori_diff,lin_v_diff,ang_v_diff;

	pos_diff.resize(mNumBodyNodes*3+1); //3dof + root world y
	ori_diff.resize(mNumBodyNodes*4); //4dof (quaternion)
	lin_v_diff.resize(mNumBodyNodes*3);
	ang_v_diff.resize(mNumBodyNodes*3); //dof - root_dof

	dart::dynamics::BodyNode* root_kin = mSkeleton->getBodyNode(0);
	Eigen::Isometry3d trans_kin = Utils::GetBodyTransform(root_kin);
	Eigen::Vector3d root_pos_kin = trans_kin.translation();
	Eigen::Vector3d root_pos_rel_kin = root_pos_kin;

	root_pos_rel_kin = Utils::AffineTransPoint(origin_trans, root_pos_rel_kin);

	pos_diff(0) = root_pos_rel_kin[1] - pos(0);
	int idx_pos_diff = 1;
	int idx_ori_diff = 0;
	int idx_linv_diff = 0;
	int idx_angv_diff = 0;
	for(int i=0; i<mNumBodyNodes; i++)
	{
		dart::dynamics::BodyNode* body_kin = mSkeleton->getBodyNode(i);
		trans_kin = Utils::GetBodyTransform(body_kin);

		Eigen::Vector3d body_pos_kin = trans_kin.translation();
		body_pos_kin = Utils::AffineTransPoint(origin_trans, body_pos_kin);
		body_pos_kin -= root_pos_rel_kin;
		pos_diff.segment(idx_pos_diff,3) = body_pos_kin.segment(0,3) - pos.segment(idx_pos_diff,3);
		idx_pos_diff += 3;

		Eigen::Quaterniond body_ori_kin(trans_kin.rotation());
		body_ori_kin = origin_quat * body_ori_kin;
		Utils::QuatNormalize(body_ori_kin);

		Eigen::Quaterniond qDiff = Utils::QuatDiff(body_ori_kin, Utils::VecToQuat(ori.segment(idx_ori_diff, 4)));
		Utils::QuatNormalize(qDiff);

		ori_diff.segment(idx_ori_diff, 4) = Utils::QuatToVec(qDiff).segment(0, 4);
		idx_ori_diff += 4;

		Eigen::Vector3d lin_vel_kin = body_kin->getLinearVelocity();
		lin_vel_kin = Utils::AffineTransVector(origin_trans, lin_vel_kin);
		lin_v_diff.segment(idx_linv_diff, 3) = lin_vel_kin - lin_v.segment(idx_linv_diff, 3);
		idx_linv_diff += 3;

		Eigen::Vector3d ang_vel_kin = body_kin->getAngularVelocity();
		ang_vel_kin = Utils::AffineTransVector(origin_trans, ang_vel_kin);
		ang_v_diff.segment(idx_angv_diff, 3) = ang_vel_kin - ang_v.segment(idx_angv_diff, 3);
		idx_angv_diff += 3;
	}

	Eigen::VectorXd state(pos.rows()+ori.rows()+lin_v.rows()+ang_v.rows()+pos_diff.rows()+ori_diff.rows()+lin_v_diff.rows()+ang_v_diff.rows()+talus_l.rows()+talus_r.rows()+mNumParamState);

	mSkeleton->setPositions(cur_pos);
	mSkeleton->setVelocities(cur_vel);
	mSkeleton->computeForwardKinematics(true, false, false);

	if(mNumParamState > 0)
		state << pos,ori,lin_v,ang_v,pos_diff,ori_diff,lin_v_diff,ang_v_diff,talus_l, talus_r,mParamState;
	else
		state << pos,ori,lin_v,ang_v,pos_diff,ori_diff,lin_v_diff,ang_v_diff,talus_l,talus_r;

	return state;
}

double
Character::
GetPhase()
{
	double t = mWorld->getTime();
	double cycleTime = mBVH->GetMaxTime();
	int cycleCount = (int)(t/cycleTime);
	double phase = t;
	if(mBVH->IsCyclic())
		phase = t - cycleCount*cycleTime;
	if(phase < 0)
		phase += cycleTime;

	return phase;
}

double
Character::
GetReward()
{
	double reward_character = this->GetReward_Character();
	mReward = reward_character;

	this->SetRewards();

	return mReward;
}

double
Character::
GetReward_Character()
{
	imit_reward = GetReward_Character_Imitation();
	effi_reward = GetReward_Character_Efficiency();

	double r = imit_reward * effi_reward;

	return r;
}

double::
Character::
GetReward_Character_Imitation()
{
	double err_scale = 2.0;  // error scale

	double pose_scale = 5.0;
	double vel_scale = 0.1;
	double end_eff_scale = 20.0;
	double root_scale = 2.0;
	double com_scale = 10.0;
	double smooth_vel_scale = 0.0;
	double smooth_pos_scale = 1.0;

	double pose_err = 0;
	double vel_err = 0;
	double end_eff_err = 0;
	double root_err = 0;
	double com_err = 0;
	double smooth_vel_err = 0;
	double smooth_pos_err = 0;

	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

	Eigen::Vector3d comSim, comSimVel;
	comSim = mSkeleton->getCOM();
	comSimVel = mSkeleton->getCOMLinearVelocity();

	double root_rot_w = mJointWeights[0];

	dart::dynamics::BodyNode* rootSim = mSkeleton->getBodyNode(0);
	Eigen::Isometry3d rootTransSim = Utils::GetJointTransform(rootSim);
	Eigen::Vector3d rootPosSim = rootTransSim.translation();
	Eigen::Quaterniond rootOrnSim(rootTransSim.rotation());
	Utils::QuatNormalize(rootOrnSim);

	Eigen::Vector3d linVelSim = mSkeleton->getRootBodyNode()->getLinearVelocity();
	Eigen::Vector3d angVelSim = mSkeleton->getRootBodyNode()->getAngularVelocity();

	Eigen::Isometry3d origin_trans_sim = Utils::GetOriginTrans(mSkeleton);

	auto ees = this->GetEndEffectors();
	Eigen::VectorXd ee_diff(ees.size()*3);
	std::vector<Eigen::Quaterniond> ee_ori_diff(ees.size());
	for(int i=0; i<ees.size(); i++){
		Eigen::Vector4d cur_ee = Utils::GetPoint4d(ees[i]->getCOM());
		ee_diff.segment<3>(i*3) = (origin_trans_sim * cur_ee).segment(0,3);
		Eigen::Isometry3d cur_ee_trans = Utils::GetBodyTransform(ees[i]);
		cur_ee_trans = origin_trans_sim * cur_ee_trans;
		Eigen::Quaterniond cur_ee_ori(cur_ee_trans.rotation());
		ee_ori_diff[i] = cur_ee_ori;
	}

	for(int i=1; i<mNumJoints; i++)
	{
		double curr_pose_err = 0;
		double curr_vel_err = 0;
		double w = mJointWeights[i]; // mJointWeights

		auto joint = mSkeleton->getJoint(i);
		int idx = joint->getIndexInSkeleton(0);
		double angle = 0;
		if(joint->getType()=="RevoluteJoint"){
			double angle = cur_pos[idx] - mTargetPositions[idx];
			double velDiff = cur_vel[idx] - mTargetVelocities[idx];
			curr_pose_err = angle * angle;
			curr_vel_err = velDiff * velDiff;
		}
		else if(joint->getType()=="BallJoint"){
			Eigen::Vector3d cur = cur_pos.segment<3>(idx);
			Eigen::Vector3d tar = mTargetPositions.segment<3>(idx);

			Eigen::Quaterniond cur_q = Utils::AxisAngleToQuaternion(cur);
			Eigen::Quaterniond tar_q = Utils::AxisAngleToQuaternion(tar);

			double angle = Utils::QuatDiffTheta(cur_q, tar_q);
			curr_pose_err = angle * angle;

			Eigen::Vector3d cur_v = cur_vel.segment<3>(idx);
			Eigen::Vector3d tar_v = mTargetVelocities.segment<3>(idx);

			curr_vel_err = (cur_v-tar_v).squaredNorm();
		}
		else if(joint->getType()=="WeldJoint"){
		}

		pose_err += w * curr_pose_err;
		vel_err += w * curr_vel_err;
	}

	for(int i=0; i<mNumBodyNodes; i++)
	{
		mAngVel.segment<3>(i*3) = mSkeleton->getBodyNode(i)->getAngularVelocity();
		mPos.segment<3>(i*3) = mSkeleton->getBodyNode(i)->getCOM();

		smooth_vel_err += (mAngVel.segment<3>(i*3) - mAngVel_prev.segment<3>(i*3)).squaredNorm()*M_PI/180.0;
		smooth_pos_err += (mPos.segment<3>(i*3) - mPos_prev.segment<3>(i*3)).squaredNorm();
	}
	mAngVel_prev = mAngVel;
	mPos_prev = mPos;

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true, false, false);

	Eigen::Vector3d comKin, comKinVel;
	comKin = mSkeleton->getCOM();
	comKinVel = mSkeleton->getCOMLinearVelocity();

	dart::dynamics::BodyNode* rootKin = mSkeleton->getBodyNode(0);
	Eigen::Isometry3d rootTransKin = Utils::GetJointTransform(rootKin);
	Eigen::Vector3d rootPosKin = rootTransKin.translation();
	Eigen::Quaterniond rootOrnKin(rootTransKin.rotation());
	Utils::QuatNormalize(rootOrnKin);

	Eigen::Vector3d linVelKin = mSkeleton->getRootBodyNode()->getLinearVelocity();
	Eigen::Vector3d angVelKin = mSkeleton->getRootBodyNode()->getAngularVelocity();

	double root_pos_err = (rootPosSim - rootPosKin).squaredNorm();

	double root_rot_diff = Utils::QuatDiffTheta(rootOrnSim, rootOrnKin);
	double root_rot_err = root_rot_diff * root_rot_diff;
	pose_err += root_rot_w * root_rot_err;

	double root_vel_err = (linVelSim - linVelKin).squaredNorm();
	double root_ang_vel_err = (angVelSim - angVelKin).squaredNorm();
	vel_err += root_rot_w * root_ang_vel_err;

	root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err;
	//root_err = root_rot_err;

	Eigen::Isometry3d origin_trans_kin = Utils::GetOriginTrans(mSkeleton);

	double end_ori_err = 0;
	ees = this->GetEndEffectors();
	for(int i=0; i<ees.size(); i++)
	{
		Eigen::Vector4d cur_ee = Utils::GetPoint4d(ees[i]->getCOM());
		ee_diff.segment<3>(i*3) -= (origin_trans_kin*cur_ee).segment(0,3);

		Eigen::Isometry3d cur_ee_trans = Utils::GetBodyTransform(ees[i]);
		cur_ee_trans = origin_trans_kin * cur_ee_trans;
		Eigen::Quaterniond cur_ee_ori(cur_ee_trans.rotation());

		double theta_ = Utils::QuatDiffTheta(ee_ori_diff[i], cur_ee_ori);
		end_ori_err += theta_ * theta_;
	}

	end_eff_err = ee_diff.squaredNorm();
	end_eff_err += end_ori_err;
	end_eff_err /= ees.size();

	com_err = 0.1 * (comKinVel - comSimVel).squaredNorm();

	pose_reward = exp(-err_scale * pose_scale * pose_err);
	vel_reward = exp(-err_scale * vel_scale * vel_err);
	end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);
	root_reward = exp(-err_scale * root_scale * root_err);
	com_reward = exp(-err_scale * com_scale * com_err);
	smooth_reward  = exp(-err_scale * smooth_pos_scale * smooth_pos_err);
	smooth_reward *= exp(-err_scale * smooth_vel_scale * smooth_vel_err);

	// imit_reward = pose_reward * vel_reward * end_eff_reward * root_reward * com_reward * smooth_reward;
	imit_reward = pose_reward * end_eff_reward * root_reward * com_reward;

	mSkeleton->setPositions(cur_pos);
	mSkeleton->setVelocities(cur_vel);
	mSkeleton->computeForwardKinematics(true, false, false);

	return imit_reward;
}

double
Character::
GetReward_Character_Efficiency()
{
	// double r_TorqueMin = this->GetReward_TorqueMin();
	double r_TorqueMin = 1.0;
	double r_ContactForce = this->GetReward_ContactForce();
	// double r_ContactForce = 1.0;

	min_reward = r_TorqueMin;
	contact_reward = r_ContactForce;

	double r = r_TorqueMin * r_ContactForce;

	return r;
}

double
Character::
GetReward_ContactForce()
{
	double err_scale = 2.0;
	double contact_scale = 0.02;
	double contact_err = 0;

	contact_err = (mContactForces_cur_norm.at(0) + mContactForces_cur_norm.at(1))/20.0;
	contact_err /= mMass;

	if(contact_err < 5.0)
		contact_err = 0.0;
	else
		contact_err -= 5.0;

	contact_reward = exp(-err_scale * contact_scale * contact_err);
	return contact_reward;
}

double
Character::
GetReward_TorqueMin()
{
	std::vector<std::deque<double>> ts = mTorques->GetTorques();
	int idx = 0;
	double sum = 0.0;
	for(int i=6; i<mMaxForces.size(); i++)
	{
		double ratio = fabs(ts[i].at(0))/mMaxForces[i];
		if(ratio > 0.4)
			sum += ratio;
		idx++;
	}
	// for(int i=6; i<mMaxForces.size(); i++)
	// {
	//  if(fabs(ts[i].at(0)) > 0.4*mMaxForces[i])
	//      sum += 1.0;
	//  idx++;
	// }
	sum /= (double)(idx);

	return -10.0 * sum;
}

bool
Character::
isEdgeTime()
{
	double t = mWorld->getTime();
	double cycleTime = mBVH->GetMaxTime();
	int cycleCount = (int)(t/cycleTime);
	double frameTime = t;
	if(mBVH->IsCyclic())
		frameTime = t - cycleCount*cycleTime;
	if(frameTime < 0)
		frameTime += cycleTime;

	if(frameTime < 0.0333*2)
		return true;
	if(frameTime > (cycleTime - 0.0333*2))
		return true;

	return false;
}

void
Character::
SetAction(const Eigen::VectorXd& a)
{
	double action_scale = 0.1;
	mAction = a*action_scale;

	double t = mWorld->getTime();
	this->SetTargetPosAndVel(t, mControlHz);
}

void
Character::
SetDesiredTorques()
{
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.tail(mTargetPositions.rows() - mRootJointDof) += mAction;
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
SetTargetPosAndVel(double t, int controlHz)
{
	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = this->GetTargetPosAndVel(t, 1.0/controlHz);
	mTargetPositions = pv.first;
	mTargetVelocities = pv.second;
}

std::pair<Eigen::VectorXd,Eigen::VectorXd>
Character::
GetTargetPosAndVel(double t,double dt)
{
	double frameTime = t;
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

	Eigen::VectorXd p = this->GetTargetPositions(t,dt,frame,frameNext,frameFraction);
	Eigen::VectorXd v = this->GetTargetVelocities(t,dt,frame,frameNext,frameFraction);

	return std::make_pair(p,v);
}

Eigen::VectorXd
Character::
GetTargetPositions(double t,double dt,int frame,int frameNext, double frameFraction)
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
	Eigen::VectorXd p = frameFraction * frameData + (1-frameFraction)* frameDataNext;

	if(mBVH->IsCyclic())	{
		double cycleTime = mBVH->GetMaxTime();
		int cycleCount = (int)(t/cycleTime);

		Eigen::Vector3d cycleOffset = mBVH->GetCycleOffset();
		cycleOffset[1] = 0.0;
		p.segment(3,3) += cycleCount*cycleOffset;
	}

	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(p.head<6>());
	T_current = mBVH->GetT0().inverse()*T_current;
	Eigen::Isometry3d T_head = mTc*T_current;
	Eigen::Vector6d p_head = dart::dynamics::FreeJoint::convertToPositions(T_head);
	p.head<6>() = p_head;

	return p;
}

Eigen::VectorXd
Character::
GetTargetVelocities(double t,double dt,int frame,int frameNext, double frameFraction)
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

	Eigen::VectorXd v = frameVel + frameFraction*(frameNextVel - frameVel);

	return v;
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
	mDevice_On = true;

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
	if(on^mDevice_On)
	{
		if(on)
			this->SetDevice_On();
		else
			this->SetDevice_Off();
	}

	mDevice_On = on;
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

	for(int i=0; i<mNumBodyNodes; i++)
	{
		dart::dynamics::BodyNode* body = mSkeleton->getBodyNode(i);
		dart::dynamics::Inertia inertia;
		auto shape = body->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get();
		double mass = mMassRatio * mDefaultMass[i];
		inertia.setMass(mass);
		inertia.setMoment(shape->computeInertia(mass));
		body->setInertia(inertia);
	}

	double param = 0.0;
	if(mMax_v[0] == mMin_v[0])
	{
		mParamState[0] = mMin_v[0];
	}
	else
	{
		double ratio = (mMassRatio-mMin_v[0])/(mMax_v[0]-mMin_v[0]);
		param = ratio*2.0 - 1.0;
		mParamState[0] = param;
	}
}

void
Character::
SetForceRatio(double r)
{
	mForceRatio = r;

	if(mUseMuscle)
	{
		for(int i=0; i<mMuscles_Femur.size(); i++)
			mMuscles_Femur.at(i)->SetF0Ratio(mForceRatio);
	}
	else
	{
		mMaxForces = mForceRatio * mDefaultForces;
	}

	double param = 0.0;
	if(mMax_v[1] == mMin_v[1])
	{
		mParamState[1] = mMin_v[1];
	}
	else
	{
		double ratio = (mForceRatio-mMin_v[1])/(mMax_v[1]-mMin_v[1]);
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
	if(mMax_v[2] == mMin_v[2])
	{
		mParamState[2] = mMin_v[2];
	}
	else
	{
		double ratio = (r-mMin_v[2])/(mMax_v[2]-mMin_v[2]);
		param = ratio*2.0 - 1.0;
		mParamState[2] = param;
	}
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

	for(double d=lower; d<=upper; )
	{
		if(d == mSpeedRatio){
			mBVHset.push_back(mBVH);
			continue;
		}

		BVH* newBVH = new BVH(mSkeleton, mBVHmap);
		newBVH->SetSpeedRatio(d);
		newBVH->Parse(mBVHpath, mBVHcyclic);
		mBVHset.push_back(newBVH);

		d += 0.1;
	}

	mBVH = mBVHset.at(0);
}

void
Character::
SetBVHidx(double r)
{
	double speed_max = mMax_v[2];
	double speed_min = mMin_v[2];

	double idx_max = speed_max * 10.0;
	double idx_min = speed_min * 10.0;
	double range = r*(idx_max - idx_min + 0.99);
	int idx = (int)(range/1.0);
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
	mMin_v = Eigen::VectorXd::Zero(mNumParamState);
	mMax_v = Eigen::VectorXd::Zero(mNumParamState);
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
		param = mMin_v[i]+(mMax_v[i]-mMin_v[i])*(param+1.0)/2.0;
		if(i==0) // Mass
			this->SetMassRatio(param);
		else if(i==1) // Force
			this->SetForceRatio(param);
		else if(i==2){ // Speed
			this->SetSpeedRatio(param);
			this->SetBVHidx(param);
		}
	}
}

void
Character::
SetMinMaxV(int idx, double lower, double upper)
{
	// 0 : mass // 1 : force // 2 : speed
	mMin_v[idx] = lower;
	mMax_v[idx] = upper;
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
			this->SetForceRatio(lower);
		}
		else if(name == "speed"){
			this->SetMinMaxV(2, lower, upper);
			this->LoadBVHset(lower, upper);
			this->SetSpeedRatio(lower);
		}
	}
}

void
Character::
SetAdaptiveParams(std::string name, double lower, double upper)
{
	if(name == "mass"){
		this->SetMinMaxV(0, lower, upper);
		this->SetMassRatio(lower);
	}
	else if(name == "force"){
		this->SetMinMaxV(1, lower, upper);
		this->SetForceRatio(lower);
	}
	else if(name == "speed"){
		this->SetMinMaxV(2, lower, upper);
		this->LoadBVHset(lower, upper);
		this->SetSpeedRatio(lower);
	}
}

void
Character::
SetRewards()
{
	(mRewards.find("reward")->second).pop_back();
	(mRewards.find("reward")->second).push_front(mReward);

	(mRewards.find("pose")->second).pop_back();
	(mRewards.find("pose")->second).push_front(pose_reward);
	(mRewards.find("vel")->second).pop_back();
	(mRewards.find("vel")->second).push_front(vel_reward);
	(mRewards.find("root")->second).pop_back();
	(mRewards.find("root")->second).push_front(root_reward);
	(mRewards.find("ee")->second).pop_back();
	(mRewards.find("ee")->second).push_front(end_eff_reward);
	(mRewards.find("com")->second).pop_back();
	(mRewards.find("com")->second).push_front(com_reward);
	(mRewards.find("smooth")->second).pop_back();
	(mRewards.find("smooth")->second).push_front(smooth_reward);
	(mRewards.find("imit")->second).pop_back();
	(mRewards.find("imit")->second).push_front(imit_reward);

	(mRewards.find("min")->second).pop_back();
	(mRewards.find("min")->second).push_front(min_reward);
	(mRewards.find("contact")->second).pop_back();
	(mRewards.find("contact")->second).push_front(contact_reward);
	(mRewards.find("effi")->second).pop_back();
	(mRewards.find("effi")->second).push_front(effi_reward);
}

void
Character::
SetTorques()
{
	mTorques->SetTorques(mDesiredTorque);
}

Torques::Torques()
{
}

void
Torques::
Initialize(dart::dynamics::SkeletonPtr skel)
{
	mDof = skel->getNumDofs();
	for(int i=0; i<mDof; i++)
		mTorquesDofs.push_back(std::deque<double>(1200));
}

void
Torques::
Reset()
{
	for(int i=0; i<mDof; i++)
		std::fill(mTorquesDofs[i].begin(), mTorquesDofs[i].end(), 0) ;
}

void
Torques::
SetTorques(const Eigen::VectorXd& desTorques)
{
	for(int i=6; i<desTorques.size(); i++)
	{
		mTorquesDofs[i].pop_back();
		mTorquesDofs[i].push_front(desTorques[i]);
	}

	double sum = 0;
	sum += desTorques.segment(6,3).norm();
	sum += fabs(desTorques[9]);
	sum += desTorques.segment(10,3).norm();
	sum += desTorques.segment(15,3).norm();
	sum += fabs(desTorques[18]);
	sum += desTorques.segment(19,3).norm();

	mTorquesDofs[0].pop_back();
	mTorquesDofs[0].push_front(sum);
}


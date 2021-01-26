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
	// delete mTorques;
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

			i++;
		}
		muscle_elem->SetMt0Default();
		muscle_elem->SetTimeStep(time_step);
		muscle_elem->SetMass();

		std::string muscle_name = muscle_elem->GetName();
		if(f0 == 700.0){
			int name_size = muscle_name.size();
			// std::string sub_name;
			// if(muscle_name[name_size-1] >= 48 && muscle_name[name_size-1] <= 57){
			// 	std::string sub_name = muscle_name.substr(2,name_size-3);
			// 	mMuscles_Map[sub_name].push_back(muscle_elem);
			// }
			// else{
				std::string sub_name = muscle_name.substr(2,name_size-2);
				if(mMuscles_Map.count(sub_name) == 0){
					mMuscles_Map.insert(std::make_pair(sub_name, std::vector<Muscle*>()));
					mMuscles_Map[sub_name].push_back(muscle_elem);
				}
				else{
					mMuscles_Map[sub_name].push_back(muscle_elem);
				}
			// }
		}

		if(muscle_elem->GetFemur())
		{
			muscle_elem->SetMt0Ratio(1.0);
			muscle_elem->SetF0Ratio(1.0);
			mMuscles_Femur.push_back(muscle_elem);
		}

		mMuscles.push_back(muscle_elem);
	}
	mNumMuscle = mMuscles.size();
	mNumMuscleMap = mMuscles_Map.size();
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
	this->SetTargetPosAndVel(mWorld->getTime());

	mRootJointDof = 6;
	mNumActiveDof = mDof - mRootJointDof;
	mNumState_Char = this->GetState().rows();
	mNumState = mNumState_Char;

	mTc = mBVH->GetT0();
	mTc.translation()[1] = 0.0;

	mStepCnt = 0;
	mStepCnt_total = 0;

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

	// mMetabolicEnergy = new MetabolicEnergy(mWorld, this->GetMuscles());
	mMetabolicEnergy = new MetabolicEnergy(mWorld);
	mMetabolicEnergy->Initialize(this->GetMuscles());

	mJointTorques = new JointTorque();
	mJointTorques->Initialize(mSkeleton);

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
	// mJointWeights <<
	// 	1.0,                    //Pelvis
	// 	1.0, 1.0, 1.0, 1.0, 1.0,//Left Leg
	// 	1.0, 1.0, 1.0, 1.0, 1.0,//Right Leg
	// 	1.0, 1.0, 1.0, 1.0,     //Torso & Neck
	// 	1.0, 1.0, 1.0, 1.0,     //Left Arm
	// 	1.0, 1.0, 1.0, 1.0;     //Right Arm

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
	this->SetTargetPosAndVel(worldTime);

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

	mMetabolicEnergy->Reset();
	mJointTorques->Reset();
	if(mUseMuscle)
		Reset_Muscles();
}

void
Character::
Reset_Muscles()
{
	for(auto m : mMuscles)
	{
		m->Reset();
	}

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

	this->SetCurVelocity();
	if(mStepCnt%5 == 0){
		mMetabolicEnergy->Set(this->GetMuscles(), mMass, mCurVel);
		mJointTorques->Set(mSkeleton, mDesiredTorque);
	}

	if(mStepCnt == 20)
		mStepCnt = 0;
	mStepCnt++;
	mStepCnt_total++;
}

void
Character::
SetMeasure()
{
	// this->SetContactForce();
	// this->SetCoT();
	// this->SetCurVelocity();
	// this->SetJointTorques();
	// mMetabolicEnergy->Set(this->GetMuscles(), mMass, mCurVel);
	// this->SetMetabolicEnergyRate();
}

void
Character::
SetMetabolicEnergyRate()
{
	double dE_BHAR04 = 0.0;
	double dE_HOUD06 = 0.0;
	double dE = 0.0;
	double h_A = 0.0;
	double h_M = 0.0;
	double h_SL = 0.0;
	double W = 0.0;
	for(auto m : mMuscles){
		dE_BHAR04 += m->GetMetabolicEnergyRate_BHAR04();
		dE_HOUD06 += m->GetMetabolicEnergyRate_HOUD06();
		// dE += m->GetMetabolicEnergyRate();
		// h_A += m->Geth_A();
		// h_M += m->Geth_M();
		// h_SL += m->Geth_SL();
		// W += m->GetW();
	}

	// std::cout << "h_A : " << h_A << std::endl;
	// std::cout << "h_M : " << h_M << std::endl;
	// std::cout << "h_SL : " << h_SL << std::endl;
	// std::cout << "W : " << W << std::endl;
	// std::cout << "dE : " << dE << std::endl;

	double dB = 1.51 * mMass;
	// std::cout << "dB : " << dB << std::endl;

	mMetabolicEnergyRate_BHAR04 = dE_BHAR04 + dB;
	mMetabolicEnergyRate_BHAR04 = mMetabolicEnergyRate_BHAR04 / (mMass*mCurVel);

	mMetabolicEnergyRate_HOUD06 = dE_HOUD06 + dB;
	mMetabolicEnergyRate_HOUD06 = mMetabolicEnergyRate_HOUD06 / (mMass*mCurVel);

	// mMetabolicEnergyRate = dE + dB;
	// mMetabolicEnergyRate = mMetabolicEnergyRate / (mMass*mCurVel);
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

	double forceL=0; double forceR=0;
	const dart::dynamics::BodyNode* bn_TalusL = mSkeleton->getBodyNode("TalusL");
	const dart::dynamics::BodyNode* bn_TalusR = mSkeleton->getBodyNode("TalusR");
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
GetState_Character()
{
	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	Eigen::VectorXd p,v;

	p.resize((mNumBodyNodes-1)*3);
	v.resize((mNumBodyNodes)*3);

	for(int i=1; i<mNumBodyNodes; i++)
	{
		p.segment<3>(3*(i-1)) = mSkeleton->getBodyNode(i)->getCOM(root);
		v.segment<3>(3*(i-1)) = mSkeleton->getBodyNode(i)->getCOMLinearVelocity();
	}

	v.tail<3>() = root->getCOMLinearVelocity();

	std::pair<double, double> phase = this->GetPhases();

	p *= 0.8;
	v *= 0.2;

    Eigen::VectorXd state(p.rows() + v.rows() + 2);
    state << p, v, phase.first, phase.second;

    return state;
}

// Eigen::VectorXd
// Character::
// GetState_Character()
// {
// 	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
// 	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

// 	Eigen::Isometry3d origin_trans = Utils::GetOriginTrans(mSkeleton);
// 	Eigen::Quaterniond origin_quat(origin_trans.rotation());
// 	Utils::QuatNormalize(origin_quat);

// 	Eigen::VectorXd pos,ori,lin_v,ang_v;
// 	// Eigen::Vector4d talus_l, talus_r;

// 	// Eigen::Vector4d talus_1 = {-0.0425, -0.03, -0.075, 1.0};
// 	// Eigen::Vector4d talus_2 = {-0.0425, -0.03,  0.075, 1.0};
// 	// Eigen::Vector4d talus_3 = { 0.0425, -0.03,  0.075, 1.0};
// 	// Eigen::Vector4d talus_4 = { 0.0425, -0.03, -0.075, 1.0};

// 	// Eigen::Isometry3d tr_l = mSkeleton->getBodyNode("TalusL")->getTransform();
// 	// Eigen::Isometry3d tr_r = mSkeleton->getBodyNode("TalusR")->getTransform();
// 	// talus_l << (tr_l*talus_1)[1],(tr_l*talus_2)[1],(tr_l*talus_3)[1],(tr_l*talus_4)[1];
// 	// talus_r << (tr_r*talus_1)[1],(tr_r*talus_2)[1],(tr_r*talus_3)[1],(tr_r*talus_4)[1];
// 	// talus_l *= 10.0; talus_r *= 10.0;

// 	pos.resize(mNumBodyNodes*3+1); //3dof + root world y
// 	ori.resize(mNumBodyNodes*4);   //4dof (quaternion)
// 	lin_v.resize(mNumBodyNodes*3);
// 	ang_v.resize(mNumBodyNodes*3); //dof - root_dof

// 	const dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
// 	Eigen::Isometry3d trans = Utils::GetBodyTransform(root);
// 	Eigen::Vector3d root_pos = trans.translation();
// 	Eigen::Vector3d root_pos_rel = root_pos;

// 	root_pos_rel = Utils::AffineTransPoint(origin_trans, root_pos_rel);
// 	pos(0) = root_pos_rel[1];

// 	int idx_pos=1; int idx_ori=0; int idx_linv=0; int idx_angv=0;
// 	for(int i=0; i<mNumBodyNodes; i++)
// 	{
// 		const dart::dynamics::BodyNode* body = mSkeleton->getBodyNode(i);
// 		trans = Utils::GetBodyTransform(body);

// 		Eigen::Vector3d body_pos = trans.translation();
// 		body_pos = Utils::AffineTransPoint(origin_trans, body_pos);
// 		body_pos -= root_pos_rel;
// 		pos.segment(idx_pos, 3) = body_pos.segment(0, 3);
// 		idx_pos += 3;

// 		Eigen::Quaterniond body_ori(trans.rotation());
// 		body_ori = origin_quat * body_ori;
// 		Utils::QuatNormalize(body_ori);
// 		ori.segment(idx_ori, 4) = Utils::QuatToVec(body_ori).segment(0, 4);
// 		idx_ori += 4;

// 		Eigen::Vector3d lin_vel = body->getLinearVelocity();
// 		lin_vel = Utils::AffineTransVector(origin_trans, lin_vel);
// 		lin_v.segment(idx_linv, 3) = lin_vel;
// 		idx_linv += 3;

// 		Eigen::Vector3d ang_vel = body->getAngularVelocity();
// 		ang_vel = Utils::AffineTransVector(origin_trans, ang_vel);
// 		ang_v.segment(idx_angv, 3) = ang_vel;
// 		idx_angv += 3;
// 	}

// 	mSkeleton->setPositions(mTargetPositions);
// 	mSkeleton->setVelocities(mTargetVelocities);
// 	mSkeleton->computeForwardKinematics(true, false, false);

// 	Eigen::VectorXd pos_diff,ori_diff,lin_v_diff,ang_v_diff;

// 	pos_diff.resize(mNumBodyNodes*3+1); //3dof + root world y
// 	ori_diff.resize(mNumBodyNodes*4); //4dof (quaternion)
// 	lin_v_diff.resize(mNumBodyNodes*3);
// 	ang_v_diff.resize(mNumBodyNodes*3); //dof - root_dof

// 	const dart::dynamics::BodyNode* root_kin = mSkeleton->getBodyNode(0);
// 	Eigen::Isometry3d trans_kin = Utils::GetBodyTransform(root_kin);
// 	Eigen::Vector3d root_pos_kin = trans_kin.translation();
// 	Eigen::Vector3d root_pos_rel_kin = root_pos_kin;

// 	root_pos_rel_kin = Utils::AffineTransPoint(origin_trans, root_pos_rel_kin);
// 	pos_diff(0) = root_pos_rel_kin[1] - pos(0);

// 	int idx_pos_diff=1;	int idx_ori_diff=0; int idx_linv_diff=0; int idx_angv_diff=0;
// 	for(int i=0; i<mNumBodyNodes; i++)
// 	{
// 		const dart::dynamics::BodyNode* body_kin = mSkeleton->getBodyNode(i);
// 		trans_kin = Utils::GetBodyTransform(body_kin);

// 		Eigen::Vector3d body_pos_kin = trans_kin.translation();
// 		body_pos_kin = Utils::AffineTransPoint(origin_trans, body_pos_kin);
// 		body_pos_kin -= root_pos_rel_kin;
// 		pos_diff.segment(idx_pos_diff,3) = body_pos_kin.segment(0,3) - pos.segment(idx_pos_diff,3);
// 		idx_pos_diff += 3;

// 		Eigen::Quaterniond body_ori_kin(trans_kin.rotation());
// 		body_ori_kin = origin_quat * body_ori_kin;
// 		Utils::QuatNormalize(body_ori_kin);

// 		Eigen::Quaterniond qDiff = Utils::QuatDiff(body_ori_kin, Utils::VecToQuat(ori.segment(idx_ori_diff, 4)));
// 		Utils::QuatNormalize(qDiff);

// 		ori_diff.segment(idx_ori_diff, 4) = Utils::QuatToVec(qDiff).segment(0, 4);
// 		idx_ori_diff += 4;

// 		Eigen::Vector3d lin_vel_kin = body_kin->getLinearVelocity();
// 		lin_vel_kin = Utils::AffineTransVector(origin_trans, lin_vel_kin);
// 		lin_v_diff.segment(idx_linv_diff, 3) = lin_vel_kin - lin_v.segment(idx_linv_diff, 3);
// 		idx_linv_diff += 3;

// 		Eigen::Vector3d ang_vel_kin = body_kin->getAngularVelocity();
// 		ang_vel_kin = Utils::AffineTransVector(origin_trans, ang_vel_kin);
// 		ang_v_diff.segment(idx_angv_diff, 3) = ang_vel_kin - ang_v.segment(idx_angv_diff, 3);
// 		idx_angv_diff += 3;
// 	}

// 	Eigen::VectorXd state(pos.rows()+ori.rows()+lin_v.rows()+ang_v.rows()+pos_diff.rows()+ori_diff.rows()+lin_v_diff.rows()+ang_v_diff.rows()+2+mNumParamState);
// 	// Eigen::VectorXd state(pos.rows()+ori.rows()+lin_v.rows()+pos_diff.rows()+ori_diff.rows()+lin_v_diff.rows()+mNumParamState);
// 	// Eigen::VectorXd state(pos.rows()+ori.rows()+lin_v.rows()+ang_v.rows()+1);

// 	// double cur_t = mWorld->getTime();
// 	// double phase = this->GetPhase();
// 	std::pair<double, double> phase = this->GetPhases();
// 	if(mNumParamState > 0)
// 		state << pos,ori,lin_v,ang_v,pos_diff,ori_diff,lin_v_diff,ang_v_diff,mParamState;
// 	else
// 		state << pos,ori,lin_v,ang_v,pos_diff,ori_diff,lin_v_diff,ang_v_diff,phase.first, phase.second;

// 	// state << pos,ori,lin_v,ang_v,phase;

// 	mSkeleton->setPositions(cur_pos);
// 	mSkeleton->setVelocities(cur_vel);
// 	mSkeleton->computeForwardKinematics(true, false, false);

// 	return state;
// }

Eigen::VectorXd
Character::
GetState_Device()
{
	return mDevice->GetState();
}

std::pair<double, double>
Character::
GetPhases()
{
	double phase = this->GetPhase();
	double rad = 2*M_PI*phase;
	double cos = std::cos(rad);
	double sin = std::sin(rad);

	return std::pair<double, double>(cos, sin);
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

	double r = reward_imit * reward_effi;

	return r;
}

double::
Character::
GetReward_Character_Imitation()
{
	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();
	Eigen::VectorXd p_diff_all = mSkeleton->getPositionDifferences(mTargetPositions,cur_pos);
	Eigen::VectorXd v_diff_all = mSkeleton->getPositionDifferences(mTargetVelocities,cur_vel);

	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());

	Eigen::Isometry3d origin_trans_sim = Utils::GetOriginTrans(mSkeleton);

	const auto& bvh_map = mBVH->GetBVHMap();
	for(auto ss : bvh_map)
	{
		auto joint = mSkeleton->getBodyNode(ss.first)->getParentJoint();
		int idx = joint->getIndexInSkeleton(0);
		if(joint->getType()=="FreeJoint")
			continue;
		else if(joint->getType()=="RevoluteJoint")
			p_diff[idx] = p_diff_all[idx];
		else if(joint->getType()=="BallJoint")
			p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
	}

	auto ees = this->GetEndEffectors();
	Eigen::VectorXd ee_diff(ees.size()*3);
	std::vector<Eigen::Quaterniond> ee_ori_diff(ees.size());
	Eigen::VectorXd com_diff;

	for(int i =0;i<ees.size();i++){
		ee_diff.segment<3>(i*3) = ees[i]->getCOM();

		Eigen::Isometry3d cur_ee_trans = Utils::GetBodyTransform(ees[i]);
		cur_ee_trans = origin_trans_sim * cur_ee_trans;
		Eigen::Quaterniond cur_ee_ori(cur_ee_trans.rotation());
		ee_ori_diff[i] = cur_ee_ori;
	}
	com_diff = mSkeleton->getCOM();

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->computeForwardKinematics(true,false,false);

	Eigen::Isometry3d origin_trans_kin = Utils::GetOriginTrans(mSkeleton);

	com_diff -= mSkeleton->getCOM();
	double end_ori_err = 0.0;
	for(int i=0;i<ees.size();i++){
		ee_diff.segment<3>(i*3) -= ees[i]->getCOM()+com_diff;

		Eigen::Isometry3d cur_ee_trans = Utils::GetBodyTransform(ees[i]);
		cur_ee_trans = origin_trans_kin * cur_ee_trans;
		Eigen::Quaterniond cur_ee_ori(cur_ee_trans.rotation());

		double theta_ = Utils::QuatDiffTheta(ee_ori_diff[i], cur_ee_ori);
		end_ori_err += theta_ * theta_;
	}

	double end_eff_err = ee_diff.squaredNorm();
	end_eff_err += end_ori_err;
	end_eff_err /= ees.size();

	mSkeleton->setPositions(cur_pos);
	mSkeleton->computeForwardKinematics(true,false,false);

	double r_q = Utils::exp_of_squared(p_diff,1.0);
	double r_v = Utils::exp_of_squared(v_diff,0.1);
	// double r_ee = Utils::exp_of_squared(ee_diff,80.0);
	double r_ee = exp(-60.0 * end_eff_err);
	double r_com = Utils::exp_of_squared(com_diff,20.0);

	double w_v = 0.1;
	double w_q = 0.6;
	double w_ee = 0.2;
	double w_com = 0.2;

	// double r_imit = r_ee*(w_q*r_q + w_v*r_v);
	// double r_imit = r_q*r_ee*r_com;
	double r_imit = w_q*r_q + w_ee*r_ee + w_com*r_com;

	mReward["pose"] = r_q;
	mReward["vel"] = r_v;
	mReward["ee"] = r_ee;
	mReward["com"] = r_com;
	// mReward["root"] = r_root;
	// mReward["smooth"] = r_smooth;
	// mReward["imit"] = r_imit;

	return r_imit;
}

// double::
// Character::
// GetReward_Character_Imitation()
// {
// 	double err_scale = 1.0;  // error scale

// 	double pose_scale = 10.0;
// 	double end_eff_scale = 10.0;
// 	double root_scale = 2.0;
// 	double com_scale = 5.0;

// 	double vel_scale = 0.1;
// 	double smooth_vel_scale = 0.02;
// 	double smooth_pos_scale = 1.0;

// 	double pose_err = 0;
// 	double vel_err = 0;
// 	double end_eff_err = 0;
// 	double root_err = 0;
// 	double com_err = 0;
// 	double smooth_vel_err = 0;
// 	double smooth_pos_err = 0;

// 	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
// 	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

// 	Eigen::Vector3d comSim = mSkeleton->getCOM();
// 	Eigen::Vector3d comSimVel = mSkeleton->getCOMLinearVelocity();

// 	double root_rot_w = mJointWeights[0];

// 	const dart::dynamics::BodyNode* rootSim = mSkeleton->getBodyNode(0);
// 	Eigen::Isometry3d rootTransSim = Utils::GetJointTransform(rootSim);
// 	Eigen::Vector3d rootPosSim = rootTransSim.translation();
// 	Eigen::Quaterniond rootOrnSim(rootTransSim.rotation());
// 	Utils::QuatNormalize(rootOrnSim);

// 	Eigen::Vector3d linVelSim = rootSim->getLinearVelocity();
// 	Eigen::Vector3d angVelSim = rootSim->getAngularVelocity();

// 	Eigen::Isometry3d origin_trans_sim = Utils::GetOriginTrans(mSkeleton);

// 	auto ees = this->GetEndEffectors();
// 	Eigen::VectorXd ee_diff(ees.size()*3);
// 	std::vector<Eigen::Quaterniond> ee_ori_diff(ees.size());
// 	for(int i=0; i<ees.size(); i++){
// 		Eigen::Vector4d cur_ee = Utils::GetPoint4d(ees[i]->getCOM());
// 		ee_diff.segment<3>(i*3) = (origin_trans_sim * cur_ee).segment(0,3);

// 		Eigen::Isometry3d cur_ee_trans = Utils::GetBodyTransform(ees[i]);
// 		cur_ee_trans = origin_trans_sim * cur_ee_trans;
// 		Eigen::Quaterniond cur_ee_ori(cur_ee_trans.rotation());
// 		ee_ori_diff[i] = cur_ee_ori;
// 	}

// 	for(int i=1; i<mNumJoints; i++)
// 	{
// 		double curr_pose_err = 0 , curr_vel_err = 0;
// 		double w = mJointWeights[i]; // mJointWeights

// 		auto joint = mSkeleton->getJoint(i);
// 		int idx = joint->getIndexInSkeleton(0);
// 		double angle = 0;
// 		if(joint->getType()=="RevoluteJoint"){
// 			double angle = cur_pos[idx] - mTargetPositions[idx];
// 			double velDiff = cur_vel[idx] - mTargetVelocities[idx];
// 			curr_pose_err = angle * angle;
// 			curr_vel_err = velDiff * velDiff;
// 		}
// 		else if(joint->getType()=="BallJoint"){
// 			Eigen::Vector3d cur = cur_pos.segment<3>(idx);
// 			Eigen::Vector3d tar = mTargetPositions.segment<3>(idx);

// 			Eigen::Quaterniond cur_q = Utils::AxisAngleToQuaternion(cur);
// 			Eigen::Quaterniond tar_q = Utils::AxisAngleToQuaternion(tar);

// 			double angle = Utils::QuatDiffTheta(cur_q, tar_q);
// 			curr_pose_err = angle * angle;

// 			Eigen::Vector3d cur_v = cur_vel.segment<3>(idx);
// 			Eigen::Vector3d tar_v = mTargetVelocities.segment<3>(idx);

// 			curr_vel_err = (cur_v-tar_v).squaredNorm();
// 		}
// 		else if(joint->getType()=="WeldJoint"){
// 		}

// 		pose_err += w * curr_pose_err;
// 		vel_err += w * curr_vel_err;
// 	}

// 	mSkeleton->setPositions(mTargetPositions);
// 	mSkeleton->setVelocities(mTargetVelocities);
// 	mSkeleton->computeForwardKinematics(true, false, false);

// 	Eigen::Vector3d comKin = mSkeleton->getCOM();
// 	Eigen::Vector3d comKinVel = mSkeleton->getCOMLinearVelocity();

// 	const dart::dynamics::BodyNode* rootKin = mSkeleton->getBodyNode(0);
// 	Eigen::Isometry3d rootTransKin = Utils::GetJointTransform(rootKin);
// 	Eigen::Vector3d rootPosKin = rootTransKin.translation();
// 	Eigen::Quaterniond rootOrnKin(rootTransKin.rotation());
// 	Utils::QuatNormalize(rootOrnKin);

// 	Eigen::Vector3d linVelKin = rootKin->getLinearVelocity();
// 	Eigen::Vector3d angVelKin = rootKin->getAngularVelocity();

// 	double root_pos_err = (rootPosSim - rootPosKin).squaredNorm();
// 	double root_rot_diff = Utils::QuatDiffTheta(rootOrnSim, rootOrnKin);
// 	double root_rot_err = root_rot_diff * root_rot_diff;
// 	pose_err += root_rot_w * root_rot_err;

// 	// root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err;
// 	root_err = root_pos_err + root_rot_err;
// 	// root_err = root_rot_err;

// 	Eigen::Isometry3d origin_trans_kin = Utils::GetOriginTrans(mSkeleton);

// 	double end_ori_err = 0;
// 	ees = this->GetEndEffectors();
// 	for(int i=0; i<ees.size(); i++)
// 	{
// 		Eigen::Vector4d cur_ee = Utils::GetPoint4d(ees[i]->getCOM());
// 		ee_diff.segment<3>(i*3) -= (origin_trans_kin*cur_ee).segment(0,3);

// 		Eigen::Isometry3d cur_ee_trans = Utils::GetBodyTransform(ees[i]);
// 		cur_ee_trans = origin_trans_kin * cur_ee_trans;
// 		Eigen::Quaterniond cur_ee_ori(cur_ee_trans.rotation());
// 		double theta_ = Utils::QuatDiffTheta(ee_ori_diff[i], cur_ee_ori);
// 		end_ori_err += theta_ * theta_;
// 	}

// 	end_eff_err = ee_diff.squaredNorm();
// 	end_eff_err += end_ori_err;
// 	end_eff_err /= ees.size();

// 	com_err = 0.1 * (comKinVel - comSimVel).squaredNorm();

// 	double r_pose = exp(-err_scale * pose_scale * pose_err);
// 	double r_vel = exp(-err_scale * vel_scale * vel_err);
// 	double r_ee = exp(-err_scale * end_eff_scale * end_eff_err);
// 	double r_root = exp(-err_scale * root_scale * root_err);
// 	double r_com = exp(-err_scale * com_scale * com_err);
// 	double r_smooth_pos = exp(-err_scale * smooth_pos_scale * smooth_pos_err);
// 	double r_smooth_vel = exp(-err_scale * smooth_vel_scale * smooth_vel_err);
// 	double r_smooth = r_smooth_pos * r_smooth_vel;

// 	// double r_imit = r_pose * r_vel * r_ee * r_root * r_com * r_smooth;
// 	// double r_imit = r_pose * r_ee * r_root * r_com * r_smooth;
// 	double r_imit = r_pose * r_ee * r_root * r_com;

// 	mReward["pose"] = r_pose;
// 	mReward["vel"] = r_vel;
// 	mReward["ee"] = r_ee;
// 	mReward["root"] = r_root;
// 	mReward["com"] = r_com;
// 	mReward["smooth"] = r_smooth;
// 	mReward["imit"] = r_imit;

// 	mSkeleton->setPositions(cur_pos);
// 	mSkeleton->setVelocities(cur_vel);
// 	mSkeleton->computeForwardKinematics(true, false, false);

// 	return r_imit;
// }

// double::
// Character::
// GetReward_Character_Imitation()
// {
// 	double err_scale = 2.0;  // error scale

// 	double pose_scale = 10.0;
// 	double vel_scale = 0.1;
// 	double end_eff_scale = 10.0;
// 	double root_scale = 2.0;
// 	double com_scale = 5.0;
// 	double smooth_vel_scale = 0.02;
// 	double smooth_pos_scale = 1.0;

// 	double pose_err = 0;
// 	double vel_err = 0;
// 	double end_eff_err = 0;
// 	double root_err = 0;
// 	double com_err = 0;
// 	double smooth_vel_err = 0;
// 	double smooth_pos_err = 0;

// 	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
// 	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

// 	Eigen::Vector3d comSim, comSimVel;
// 	comSim = mSkeleton->getCOM();
// 	comSimVel = mSkeleton->getCOMLinearVelocity();

// 	double root_rot_w = mJointWeights[0];

// 	const dart::dynamics::BodyNode* rootSim = mSkeleton->getBodyNode(0);
// 	Eigen::Isometry3d rootTransSim = Utils::GetJointTransform(rootSim);
// 	Eigen::Vector3d rootPosSim = rootTransSim.translation();
// 	Eigen::Quaterniond rootOrnSim(rootTransSim.rotation());
// 	Utils::QuatNormalize(rootOrnSim);

// 	Eigen::Vector3d linVelSim = mSkeleton->getRootBodyNode()->getLinearVelocity();
// 	Eigen::Vector3d angVelSim = mSkeleton->getRootBodyNode()->getAngularVelocity();

// 	Eigen::Isometry3d origin_trans_sim = Utils::GetOriginTrans(mSkeleton);

// 	auto ees = this->GetEndEffectors();
// 	Eigen::VectorXd ee_diff(ees.size()*3);
// 	std::vector<Eigen::Quaterniond> ee_ori_diff(ees.size());
// 	for(int i=0; i<ees.size(); i++){
// 		Eigen::Vector4d cur_ee = Utils::GetPoint4d(ees[i]->getCOM());
// 		ee_diff.segment<3>(i*3) = (origin_trans_sim * cur_ee).segment(0,3);
// 		Eigen::Isometry3d cur_ee_trans = Utils::GetBodyTransform(ees[i]);
// 		cur_ee_trans = origin_trans_sim * cur_ee_trans;
// 		Eigen::Quaterniond cur_ee_ori(cur_ee_trans.rotation());
// 		ee_ori_diff[i] = cur_ee_ori;
// 	}

// 	Eigen::VectorXd body_diff((mNumBodyNodes-1)*3);
// 	Eigen::VectorXd body_vel_diff((mNumBodyNodes-1)*3);
// 	for(int i=1; i<mNumBodyNodes; i++)
// 	{
// 		auto body = mSkeleton->getBodyNode(i);
// 		Eigen::Vector4d cur_body = 	Utils::GetPoint4d(body->getCOM());
// 		Eigen::Vector4d cur_body_vel = 	Utils::GetPoint4d(body->getLinearVelocity());
// 		body_diff.segment<3>((i-1)*3) = (origin_trans_sim * cur_body).segment(0,3);
// 		body_vel_diff.segment<3>((i-1)*3) = (origin_trans_sim * cur_body_vel).segment(0,3);
// 	}

// 	for(int i=1; i<mNumJoints; i++)
// 	{
// 		double curr_pose_err = 0;
// 		double curr_vel_err = 0;
// 		double w = mJointWeights[i]; // mJointWeights

// 		auto joint = mSkeleton->getJoint(i);
// 		int idx = joint->getIndexInSkeleton(0);
// 		double angle = 0;
// 		if(joint->getType()=="RevoluteJoint"){
// 			double angle = cur_pos[idx] - mTargetPositions[idx];
// 			double velDiff = cur_vel[idx] - mTargetVelocities[idx];
// 			curr_pose_err = angle * angle;
// 			curr_vel_err = velDiff * velDiff;
// 		}
// 		else if(joint->getType()=="BallJoint"){
// 			Eigen::Vector3d cur = cur_pos.segment<3>(idx);
// 			Eigen::Vector3d tar = mTargetPositions.segment<3>(idx);

// 			Eigen::Quaterniond cur_q = Utils::AxisAngleToQuaternion(cur);
// 			Eigen::Quaterniond tar_q = Utils::AxisAngleToQuaternion(tar);

// 			double angle = Utils::QuatDiffTheta(cur_q, tar_q);
// 			curr_pose_err = angle * angle;

// 			Eigen::Vector3d cur_v = cur_vel.segment<3>(idx);
// 			Eigen::Vector3d tar_v = mTargetVelocities.segment<3>(idx);

// 			curr_vel_err = (cur_v-tar_v).squaredNorm();
// 		}
// 		else if(joint->getType()=="WeldJoint"){
// 		}

// 		pose_err += w * curr_pose_err;
// 		vel_err += w * curr_vel_err;
// 	}

// 	for(int i=0; i<mNumBodyNodes; i++)
// 	{
// 		mAngVel.segment<3>(i*3) = mSkeleton->getBodyNode(i)->getAngularVelocity();
// 		mPos.segment<3>(i*3) = mSkeleton->getBodyNode(i)->getCOM();

// 		smooth_vel_err += (mAngVel.segment<3>(i*3) - mAngVel_prev.segment<3>(i*3)).squaredNorm()*M_PI/180.0;
// 		smooth_pos_err += (mPos.segment<3>(i*3) - mPos_prev.segment<3>(i*3)).squaredNorm();
// 	}
// 	mAngVel_prev = mAngVel;
// 	mPos_prev = mPos;

// 	mSkeleton->setPositions(mTargetPositions);
// 	mSkeleton->setVelocities(mTargetVelocities);
// 	mSkeleton->computeForwardKinematics(true, false, false);

// 	Eigen::Vector3d comKin, comKinVel;
// 	comKin = mSkeleton->getCOM();
// 	comKinVel = mSkeleton->getCOMLinearVelocity();

// 	const dart::dynamics::BodyNode* rootKin = mSkeleton->getBodyNode(0);
// 	Eigen::Isometry3d rootTransKin = Utils::GetJointTransform(rootKin);
// 	Eigen::Vector3d rootPosKin = rootTransKin.translation();
// 	Eigen::Quaterniond rootOrnKin(rootTransKin.rotation());
// 	Utils::QuatNormalize(rootOrnKin);

// 	Eigen::Vector3d linVelKin = mSkeleton->getRootBodyNode()->getLinearVelocity();
// 	Eigen::Vector3d angVelKin = mSkeleton->getRootBodyNode()->getAngularVelocity();

// 	double root_pos_err = (rootPosSim - rootPosKin).squaredNorm();

// 	double root_rot_diff = Utils::QuatDiffTheta(rootOrnSim, rootOrnKin);
// 	double root_rot_err = root_rot_diff * root_rot_diff;
// 	pose_err += root_rot_w * root_rot_err;

// 	double root_vel_err = (linVelSim - linVelKin).squaredNorm();
// 	double root_ang_vel_err = (angVelSim - angVelKin).squaredNorm();
// 	vel_err += root_rot_w * root_ang_vel_err;

// 	root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err;
// 	// root_err = root_rot_err;

// 	Eigen::Isometry3d origin_trans_kin = Utils::GetOriginTrans(mSkeleton);

// 	double end_ori_err = 0;
// 	ees = this->GetEndEffectors();
// 	for(int i=0; i<ees.size(); i++)
// 	{
// 		Eigen::Vector4d cur_ee = Utils::GetPoint4d(ees[i]->getCOM());
// 		ee_diff.segment<3>(i*3) -= (origin_trans_kin*cur_ee).segment(0,3);

// 		Eigen::Isometry3d cur_ee_trans = Utils::GetBodyTransform(ees[i]);
// 		cur_ee_trans = origin_trans_kin * cur_ee_trans;
// 		Eigen::Quaterniond cur_ee_ori(cur_ee_trans.rotation());

// 		double theta_ = Utils::QuatDiffTheta(ee_ori_diff[i], cur_ee_ori);
// 		end_ori_err += theta_ * theta_;
// 	}

// 	for(int i=1; i<mNumBodyNodes; i++)
// 	{
// 		auto body = mSkeleton->getBodyNode(i);
// 		Eigen::Vector4d cur_body = 	Utils::GetPoint4d(body->getCOM());
// 		Eigen::Vector4d cur_body_vel = 	Utils::GetPoint4d(body->getLinearVelocity());
// 		body_diff.segment<3>((i-1)*3) -= (origin_trans_kin * cur_body).segment(0,3);
// 		body_vel_diff.segment<3>((i-1)*3) -= (origin_trans_kin * cur_body_vel).segment(0,3);
// 	}

// 	double body_err = 0.0;
// 	body_err = body_diff.squaredNorm();
// 	body_err /= mNumBodyNodes-1;

// 	double body_vel_err = 0.0;
// 	body_vel_err = body_vel_diff.squaredNorm();
// 	body_vel_err /= mNumBodyNodes-1;

// 	end_eff_err = ee_diff.squaredNorm();
// 	end_eff_err += end_ori_err;
// 	end_eff_err /= ees.size();

// 	com_err = 0.1 * (comKinVel - comSimVel).squaredNorm();

// 	double r_pose = exp(-err_scale * pose_scale * pose_err);
// 	double r_vel = exp(-err_scale * vel_scale * vel_err);
// 	double r_ee = exp(-err_scale * end_eff_scale * end_eff_err);
// 	double r_root = exp(-err_scale * root_scale * root_err);
// 	double r_com = exp(-err_scale * com_scale * com_err);
// 	double r_smooth_pos = exp(-err_scale * smooth_pos_scale * smooth_pos_err);
// 	double r_smooth_vel = exp(-err_scale * smooth_vel_scale * smooth_vel_err);
// 	double r_smooth = r_smooth_pos * r_smooth_vel;

// 	double r_body = exp(-err_scale * 90.0 * body_err);
// 	double r_body_vel = exp(-err_scale * 0.3 * body_vel_err);
// 	// std::cout << "body err : " << body_err << std::endl;
// 	// std::cout << "body vel err : " << body_vel_err << std::endl;
// 	// double r_imit = r_body * r_com * r_ee * r_root;

// 	// double r_imit = r_pose * r_vel * r_ee * r_root * r_com * r_smooth;
// 	double r_imit = r_pose * r_ee * r_root * r_com * r_smooth;

// 	mReward["pose"] = r_pose;
// 	// mReward["pose"] = r_body;
// 	mReward["vel"] = r_vel;
// 	// mReward["vel"] = r_body_vel;
// 	mReward["ee"] = r_ee;
// 	mReward["root"] = r_root;
// 	mReward["com"] = r_com;
// 	mReward["smooth"] = r_smooth;
// 	mReward["imit"] = r_imit;

// 	mSkeleton->setPositions(cur_pos);
// 	mSkeleton->setVelocities(cur_vel);
// 	mSkeleton->computeForwardKinematics(true, false, false);

// 	return r_imit;
// }

double
Character::
GetReward_Character_Efficiency()
{
	// double r_TorqueMin = this->GetReward_TorqueMin();
	double r_TorqueMin = 1.0;
	// double r_ContactForce = this->GetReward_ContactForce();
	double r_ContactForce = 1.0;

	mReward["min"] = r_TorqueMin;
	mReward["contact"] = r_ContactForce;

	double r = r_TorqueMin * r_ContactForce;
	return r;
}

double
Character::
GetReward_MetabolicEnergy()
{
	double err_scale = 2.0;
	double metabolic_scale = 0.1;
	double metabolic_err = 0.0;

	metabolic_err = this->GetMetabolicEnergyRate_HOUD06();
	metabolic_err /= mMass;

	double r_metabolic = exp(-err_scale * metabolic_scale * metabolic_err);
	return r_metabolic;
}

double
Character::
GetReward_ContactForce()
{
	double err_scale = 2.0;
	double contact_scale = 0.01;
	double contact_err = 0;

	contact_err = (mContactForces_cur_norm.at(0) + mContactForces_cur_norm.at(1))/20.0;
	contact_err /= mMass;
	// if(contact_err < 5.0)
	// 	contact_err = 0.0;
	// else
	// 	contact_err -= 5.0;

	double r_contact = exp(-err_scale * contact_scale * contact_err);
	return r_contact;
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
	this->SetTargetPosAndVel(t);
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
SetTargetPosAndVel(double t)
{
	double frameTime = t;
	double dt = 1.0/mControlHz;

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
	mTargetPositions = frameFraction * frameData + (1-frameFraction)* frameDataNext;

	if(mBVH->IsCyclic()) {
		double cycleTime = mBVH->GetMaxTime();
		int cycleCount = (int)(t/cycleTime);

		Eigen::Vector3d cycleOffset = mBVH->GetCycleOffset();
		cycleOffset[1] = 0.0;
		mTargetPositions.segment(3,3) += cycleCount*cycleOffset;
	}

	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(mTargetPositions.head<6>());
	T_current = mBVH->GetT0().inverse()*T_current;
	Eigen::Isometry3d T_head = mTc*T_current;
	Eigen::Vector6d p_head = dart::dynamics::FreeJoint::convertToPositions(T_head);
	mTargetPositions.head<6>() = p_head;
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

	mTargetVelocities = frameVel + frameFraction*(frameNextVel - frameVel);
}


// void
// Character::
// SetTargetPosAndVel(double t, int controlHz)
// {
// 	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = this->GetTargetPosAndVel(t, 1.0/controlHz);
// 	mTargetPositions = pv.first;
// 	mTargetVelocities = pv.second;
// }

// std::pair<Eigen::VectorXd,Eigen::VectorXd>
// Character::
// GetTargetPosAndVel(double t,double dt)
// {
// 	double frameTime = t;
// 	if(mBVH->IsCyclic()){
// 		double cycleTime = mBVH->GetMaxTime();
// 		int cycleCount = (int)(t/cycleTime);

// 		frameTime = t - cycleCount*cycleTime;
// 		if(frameTime < 0)
// 			frameTime += cycleTime;
// 	}

// 	int frame = (int)(frameTime/dt);
// 	int frameNext = frame + 1;

// 	if(mBVH->IsCyclic()){
// 		if(frameNext >= mBVH->GetNumTotalFrames())
// 			frameNext = frame;
// 	}
// 	else{
// 		if(frameNext > 941){
// 			frameNext = 941;
// 			frame = 941;
// 		}
// 	}

// 	double frameFraction = (frameTime - frame*dt)/dt;

// 	Eigen::VectorXd p = this->GetTargetPositions(t,dt,frame,frameNext,frameFraction);
// 	Eigen::VectorXd v = this->GetTargetVelocities(t,dt,frame,frameNext,frameFraction);

// 	return std::make_pair(p,v);
// }

// Eigen::VectorXd
// Character::
// GetTargetPositions(double t,double dt,int frame,int frameNext, double frameFraction)
// {
// 	Eigen::VectorXd frameData, frameDataNext;
// 	if(mBVH->IsCyclic()){
// 		frameData = mBVH->GetMotion(frame);
// 		frameDataNext = mBVH->GetMotion(frameNext);
// 	}
// 	else{
// 		frameData = mBVH->GetMotionNonCyclic(frame);
// 		frameDataNext = mBVH->GetMotionNonCyclic(frameNext);
// 	}

// 	// Eigen::VectorXd p = Utils::GetPoseSlerp(mSkeleton, frameFraction, frameData, frameDataNext);
// 	Eigen::VectorXd p = frameFraction * frameData + (1-frameFraction)* frameDataNext;

// 	if(mBVH->IsCyclic())	{
// 		double cycleTime = mBVH->GetMaxTime();
// 		int cycleCount = (int)(t/cycleTime);

// 		Eigen::Vector3d cycleOffset = mBVH->GetCycleOffset();
// 		cycleOffset[1] = 0.0;
// 		p.segment(3,3) += cycleCount*cycleOffset;
// 	}

// 	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(p.head<6>());
// 	T_current = mBVH->GetT0().inverse()*T_current;
// 	Eigen::Isometry3d T_head = mTc*T_current;
// 	Eigen::Vector6d p_head = dart::dynamics::FreeJoint::convertToPositions(T_head);
// 	p.head<6>() = p_head;

// 	return p;
// }

// Eigen::VectorXd
// Character::
// GetTargetVelocities(double t,double dt,int frame,int frameNext, double frameFraction)
// {
// 	Eigen::VectorXd frameVel, frameNextVel;
// 	if(mBVH->IsCyclic()){
// 		frameVel = mBVH->GetMotionVel(frame);
// 		frameNextVel = mBVH->GetMotionVel(frameNext);
// 	}
// 	else{
// 		frameVel = mBVH->GetMotionVelNonCyclic(frame);
// 		frameNextVel = mBVH->GetMotionVelNonCyclic(frameNext);
// 	}

// 	Eigen::VectorXd v = frameVel + frameFraction*(frameNextVel - frameVel);

// 	return v;
// }

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

	mBVH = mBVHset.at(0);
}

void
Character::
SetBVHidx(double r)
{
	double speed_max = mParamMax[2];
	double speed_min = mParamMin[2];

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
			this->SetForceRatio(lower);
		}
		else if(name == "speed"){
			this->SetMinMaxV(2, lower, upper);
			this->LoadBVHset(lower, upper);
			this->SetSpeedRatio(lower);
		}
	}
}

// void
// Character::
// SetJointTorques()
// {
// 	for(int i=0; i<mNumJoints; i++)
// 	{
// 		auto joint = mSkeleton->getJoint(i);
// 		int idx = joint->getIndexInSkeleton(0);
// 		std::string name = joint->getName();

// 		if(joint->getType() == "FreeJoint")
// 		{
// 			mJointTorques->Set(name+"_x", mDesiredTorque[idx+0]);
// 			mJointTorques->Set(name+"_y", mDesiredTorque[idx+1]);
// 			mJointTorques->Set(name+"_z", mDesiredTorque[idx+2]);
// 			mJointTorques->Set(name+"_a", mDesiredTorque[idx+3]);
// 			mJointTorques->Set(name+"_b", mDesiredTorque[idx+4]);
// 			mJointTorques->Set(name+"_c", mDesiredTorque[idx+5]);
// 		}
// 		else if(joint->getType() == "BallJoint")
// 		{
// 			mJointTorques->Set(name+"_x", mDesiredTorque[idx+0]);
// 			mJointTorques->Set(name+"_y", mDesiredTorque[idx+1]);
// 			mJointTorques->Set(name+"_z", mDesiredTorque[idx+2]);
// 		}
// 		else if(joint->getType() == "RevoluteJoint")
// 		{
// 			mJointTorques->Set(name, mDesiredTorque[idx+0]);
// 		}
// 		else
// 		{
// 		}
// 	}
// }

// Torques::Torques()
// {
// }

// void
// Torques::
// Initialize(dart::dynamics::SkeletonPtr skel)
// {
// 	mDof = skel->getNumDofs();
// 	for(int i=0; i<mDof; i++)
// 		mTorquesDofs.push_back(std::deque<double>(1200));
// }

// void
// Torques::
// Reset()
// {
// 	for(int i=0; i<mDof; i++)
// 		std::fill(mTorquesDofs[i].begin(), mTorquesDofs[i].end(), 0) ;
// }

// void
// Torques::
// SetTorques(const Eigen::VectorXd& desTorques)
// {
// 	for(int i=6; i<desTorques.size(); i++)
// 	{
// 		mTorquesDofs[i].pop_back();
// 		mTorquesDofs[i].push_front(desTorques[i]);
// 	}

// 	double sum = 0;
// 	sum += desTorques.segment(6,3).norm();
// 	sum += fabs(desTorques[9]);
// 	sum += desTorques.segment(10,3).norm();
// 	sum += desTorques.segment(15,3).norm();
// 	sum += fabs(desTorques[18]);
// 	sum += desTorques.segment(19,3).norm();

// 	mTorquesDofs[0].pop_back();
// 	mTorquesDofs[0].push_front(sum);
// }


#include "Character.h"
#include "BVH.h"
#include "DARTHelper.h"
#include "Muscle.h"
#include "Device.h"
#include "Utils.h"
#include "dart/gui/gui.hpp"
#include <tinyxml.h>
using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

Character::
Character()
	:mSkeleton(nullptr),mBVH(nullptr),mDevice(nullptr),mTc(Eigen::Isometry3d::Identity()),mUseMuscle(false),mUseDevice(false),mOnDevice(false),mPhase(0.0)
{
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
LoadSkeleton(const std::string& path,bool create_obj)
{
	mSkeleton = BuildFromFile(path,create_obj);
	std::map<std::string,std::string> bvh_map;
	TiXmlDocument doc;
	doc.LoadFile(path);
	TiXmlElement* skel_elem = doc.FirstChildElement("Skeleton");
	for(TiXmlElement* node = skel_elem->FirstChildElement("Node");node != nullptr;node = node->NextSiblingElement("Node"))
	{
		if(node->Attribute("endeffector")!=nullptr)
		{
			std::string ee =node->Attribute("endeffector");
			if(ee == "True")
			{
				mEndEffectors.push_back(mSkeleton->getBodyNode(std::string(node->Attribute("name"))));
			}
		}

		TiXmlElement* joint_elem = node->FirstChildElement("Joint");
		if(joint_elem->Attribute("bvh")!=nullptr)
		{
			bvh_map.insert(std::make_pair(node->Attribute("name"),joint_elem->Attribute("bvh")));
		}
	}

	mBVH = new BVH(mSkeleton, bvh_map);
}

void
Character::
LoadBVH(const std::string& path,bool cyclic)
{
	if(mBVH == nullptr){
		std::cout<<"Initialize BVH class first"<<std::endl;
		return;
	}

	mBVH->Parse(path, cyclic);
}

void
Character::
LoadMuscles(const std::string& path)
{
	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open file : " << path << std::endl;
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
		bool isValid = true;
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
				isValid = false;
				break;
			}
			if(i==0||i==num_waypoints-1)
				muscle_elem->AddAnchor(mSkeleton->getBodyNode(body),glob_pos);
			else
				muscle_elem->AddAnchor(mSkeleton,mSkeleton->getBodyNode(body),glob_pos,2);

			i++;
		}
		if(isValid)
			mMuscles.push_back(muscle_elem);
	}

	mNumMuscle = mMuscles.size();
}

void
Character::
Initialize(dart::simulation::WorldPtr& wPtr, int conHz, int simHz)
{
	if(mSkeleton == nullptr)
	{
		std::cout<<"Initialize Character First"<<std::endl;
		exit(0);
	}

	this->SetWorld(wPtr);
	mWorld->addSkeleton(mSkeleton);

	mControlHz = conHz;
	mSimulationHz = simHz;
	this->SetPDParameters();

	const std::string& type =
		mSkeleton->getRootBodyNode()->getParentJoint()->getType();
	if(type == "FreeJoint")
		mRootJointDof = 6;
	else if(type == "PlanarJoint")
		mRootJointDof = 3;
	else
		mRootJointDof = 0;

	mTorques = new Torques();
	mTorques->Init(mSkeleton);

	this->Reset();

	mNumDof = mSkeleton->getNumDofs();
	mNumActiveDof = mNumDof - mRootJointDof;
	// mNumState = this->GetState().rows();
	mNumState = 400;

	mAction.resize(mNumActiveDof);
	mAction_prev.resize(mNumActiveDof);

	mDesiredTorque.resize(mNumDof);
	mDesiredTorque_prev.resize(mNumDof);

	mFemurSignals_L.resize(1200);
	mFemurSignals_R.resize(1200);

	std::deque<double> pose_(70, 0);
	std::deque<double> vel_(70, 0);
	std::deque<double> root_(70, 0);
	std::deque<double> ee_(70, 0);
	std::deque<double> com_(70, 0);
	std::deque<double> min_(70, 0);

	mReward_map;
	mReward_map.insert(std::make_pair("pose", pose_));
	mReward_map.insert(std::make_pair("vel", vel_));
	mReward_map.insert(std::make_pair("root", root_));
	mReward_map.insert(std::make_pair("ee", ee_));
	mReward_map.insert(std::make_pair("com", com_));
	mReward_map.insert(std::make_pair("min", min_));

	int num_joint = mSkeleton->getNumJoints();
	mJointWeights.resize(num_joint);
	// for(int i=0; i<mJointWeights.size(); i++)
	// 	mJointWeights[i] = 0.1;
	mJointWeights <<
			1.0,			//Pelvis
			0.5, 0.3, 0.2,	//Left Leg
			0.5, 0.3, 0.2,	//Right Leg
			0.5, 0.2,		//Torso & Neck
			0.3, 0.2, 0.1,	//Left Arm
			0.3, 0.2, 0.1;	//Right Arm
	mJointWeights /= mJointWeights.sum();

	int dof = mSkeleton->getNumDofs();
	maxForces.resize(dof);
	// for(int i=0; i<maxForces.size(); i++)
	// 	maxForces[i] = 200.0;
	maxForces <<
			0, 0, 0, 0, 0, 0,	//pelvis
			200, 200, 200,		//Femur L
			150,				//Tibia L
			90, 90, 90,			//Talus L
			200, 200, 200,		//Femur R
			150,				//Tibia R
			90, 90, 90,			//Talus R
			150, 150, 150,		//Torso
			30, 30, 30,			//Neck
			90, 90, 90,		//Shoulder L
			60,					//Arm L
			30, 30, 30,			//Hand L
			90, 90, 90,		//Shoulder R
			60,					//Arm R
			30, 30, 30;			//Hand R
}

void
Character::
SetPDParameters()
{
	int dof = mSkeleton->getNumDofs();
	// mKp.resize(dof, 500);
	// mKv.resize(dof, 50);
	mKp.resize(dof);
	mKv.resize(dof);

	mKp << 0, 0, 0, 0, 0, 0,
		500, 500, 500,
		500,
		400, 400, 400,
		500, 500, 500,
		500,
		400, 400, 400,
		1000, 1000, 1000,
		100, 100, 100,
		400, 400, 400,
		300,
		100, 100, 100,
		400, 400, 400,
		300,
		100, 100, 100;

	mKv << 0, 0, 0, 0, 0, 0,
		50, 50, 50,
		50,
		40, 40, 40,
		50, 50, 50,
		50,
		40, 40, 40,
		100, 100, 100,
		10, 10, 10,
		40, 40, 40,
		30,
		10, 10, 10,
		40, 40, 40,
		30,
		10, 10, 10;
}

void
Character::
SetKp(double kp)
{
	int dof = mSkeleton->getNumDofs();
	mKp = Eigen::VectorXd::Constant(dof,kp);
}

void
Character::
SetKv(double kv)
{
	int dof = mSkeleton->getNumDofs();
	mKv = Eigen::VectorXd::Constant(dof,kv);
}

void
Character::
Initialize_Muscles()
{
	mUseMuscle = true;
	mNumTotalRelatedDof = 0;
	for(auto m : this->GetMuscles()){
		m->Update();
		mNumTotalRelatedDof += m->GetNumRelatedDofs();
	}

	Reset_Muscles();
}

void
Character::
Initialize_Analysis()
{
	mTorques = new Torques();
	mTorques->Init(mSkeleton);

	for(int i=0; i<70; i++)
		mRewards.push_back(0.0);
}

void
Character::
Reset()
{
	mTc = mBVH->GetT0();
	mTc.translation()[1] = 0.0;

	mSkeleton->clearConstraintImpulses();
	mSkeleton->clearInternalForces();
	mSkeleton->clearExternalForces();

	double worldTime = mWorld->getTime();
	this->SetTargetPosAndVel(worldTime, mControlHz);

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true,false,false);

	mDesiredTorque.setZero();
	mDesiredTorque_prev.setZero();

	mAction.setZero();
	mAction_prev.setZero();
	mTorques->Reset();
	mFemurSignals_L.clear();
	mFemurSignals_R.clear();
	mFemurSignals_L.resize(1200);
	mFemurSignals_R.resize(1200);

	if(mUseMuscle)
		Reset_Muscles();
}

void
Character::
Reset_Muscles()
{
	mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(mNumTotalRelatedDof);
	mCurrentMuscleTuple.L = Eigen::MatrixXd::Zero(mNumActiveDof, mNumMuscle);
	mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof);
	mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof);
	mActivationLevels = Eigen::VectorXd::Zero(mNumMuscle);
}

void
Character::
Step()
{
	SetDesiredTorques();
	this->SetTorques();
	mSkeleton->setForces(mDesiredTorque);
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
		int n = mSkeleton->getNumDofs();
		int m = mMuscles.size();
		Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n,m);
		Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);

		for(int i=0;i<mMuscles.size();i++)
		{
			auto muscle = mMuscles[i];
			// muscle->Update();
			Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
			auto Ap = muscle->GetForceJacobianAndPassive();

			JtA.block(0,i,n,1) = Jt*Ap.first;
			Jtp += Jt*Ap.second;
		}

		mCurrentMuscleTuple.JtA = GetMuscleTorques();
		mCurrentMuscleTuple.L = JtA.block(mRootJointDof,0,n-mRootJointDof,m);
		mCurrentMuscleTuple.b = Jtp.segment(mRootJointDof,n-mRootJointDof);
		mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
		mMuscleTuples.push_back(mCurrentMuscleTuple);
	}
}

Eigen::VectorXd
Character::
GetState()
{
	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

	Eigen::Isometry3d origin_trans = Utils::GetOriginTrans(mSkeleton);
	Eigen::Quaterniond origin_quat(origin_trans.rotation());
	Utils::QuatNormalize(origin_quat);

	Eigen::VectorXd pos,ori,lin_v,ang_v;

	int body_num = mSkeleton->getNumBodyNodes();

	pos.resize(body_num*3+1); //3dof + root world y
	ori.resize(body_num*4); //4dof (quaternion)
	lin_v.resize(body_num*3);
	ang_v.resize(body_num*3); //dof - root_dof

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
	for(int i=0; i<body_num; i++)
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

	pos_diff.resize(body_num*3+1); //3dof + root world y
	ori_diff.resize(body_num*4); //4dof (quaternion)
	lin_v_diff.resize(body_num*3);
	ang_v_diff.resize(body_num*3); //dof - root_dof

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
	for(int i=0; i<body_num; i++)
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

	Eigen::VectorXd device_state = mDevice->GetState();
	Eigen::VectorXd state(pos.rows()+ori.rows()+lin_v.rows()+ang_v.rows()+pos_diff.rows()+ori_diff.rows()+lin_v_diff.rows()+ang_v_diff.rows()+device_state.rows());

	// Eigen::VectorXd state(pos.rows()+ori.rows()+lin_v.rows()+ang_v.rows()+pos_diff.rows()+ori_diff.rows()+lin_v_diff.rows()+ang_v_diff.rows());

	this->SetPhase();

	mSkeleton->setPositions(cur_pos);
	mSkeleton->setVelocities(cur_vel);
	mSkeleton->computeForwardKinematics(true, false, false);

	state<<pos,ori,lin_v,ang_v,pos_diff,ori_diff,lin_v_diff,ang_v_diff,device_state;
	// state<<pos,ori,lin_v,ang_v,pos_diff,ori_diff,lin_v_diff,ang_v_diff;

	return state;
}

double
Character::
GetReward()
{
	r_character = this->GetReward_Character();

	mReward = r_character;

	return mReward;
}

double
Character::
GetReward_Character()
{
	double pose_scale = 2.0;
	double vel_scale = 0.1;
	double end_eff_scale = 40.0;
	double root_scale = 5.0;
	double com_scale = 10.0;
	double err_scale = 2.0;  // error scale

	double pose_err = 0;
	double vel_err = 0;
	double end_eff_err = 0;
	double root_err = 0;
	double com_err = 0;

	double reward = 0;

	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

	Eigen::Vector3d comSim, comSimVel;
	comSim = mSkeleton->getCOM();
	comSimVel = mSkeleton->getCOMLinearVelocity();

	int num_joints = mSkeleton->getNumJoints();

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
	for(int i=0; i<ees.size(); i++){
		Eigen::Vector4d cur_ee = Utils::GetPoint4d(ees[i]->getCOM());
		ee_diff.segment<3>(i*3) = (origin_trans_sim * cur_ee).segment(0,3);
	}

	for(int i=1; i<num_joints; i++)
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

	Eigen::Isometry3d origin_trans_kin = Utils::GetOriginTrans(mSkeleton);

	ees = this->GetEndEffectors();
	for(int i=0; i<ees.size(); i++)
	{
		Eigen::Vector4d cur_ee = Utils::GetPoint4d(ees[i]->getCOM());
		ee_diff.segment<3>(i*3) -= (origin_trans_kin*cur_ee).segment(0,3);
	}

	end_eff_err = ee_diff.squaredNorm();
	end_eff_err /= ees.size();

	com_err = 0.1 * (comKinVel - comSimVel).squaredNorm();

	pose_reward = exp(-err_scale * pose_scale * pose_err);
	vel_reward = exp(-err_scale * vel_scale * vel_err);
	end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);
	root_reward = exp(-err_scale * root_scale * root_err);
	com_reward = exp(-err_scale * com_scale * com_err);

	double r_imitation = pose_reward * vel_reward * end_eff_reward * root_reward * com_reward;

	// min_reward = this->GetTorqueReward();
	// double r_torque_min = min_reward;

	// double r_ = 0.8*r_imitation + 0.2*r_torque_min;
	double r_ = r_imitation;

	mSkeleton->setPositions(cur_pos);
	mSkeleton->setVelocities(cur_vel);
	mSkeleton->computeForwardKinematics(true, false, false);

	this->SetReward_Graph();

	return r_;
}

double
Character::
GetTorqueReward()
{
	mTorques->Set();
	std::vector<double> torques = mTorques->GetTorquesCur();

	double sum = 0;
	// int idx = 0;
	// for(int i=0; i<torques.size(); i++)
	// {
	// 	if(maxForces[i] != 0){
	// 		sum += torques[i]/maxForces[i];
	// 		idx++;
	// 	}
	// }
	// sum /= (double)(idx);

	int idx = 0;
	for(int i=6; i<20; i++)
	{
		sum += fabs(mDesiredTorque[i])/maxForces[i];
		idx++;
	}

	// sum += fabs(mDesiredTorque[6])/maxForces[6];
	// sum += fabs(mDesiredTorque[7])/maxForces[7];
	// sum += fabs(mDesiredTorque[8])/maxForces[8];
	// sum += fabs(mDesiredTorque[13])/maxForces[13];
	// sum += fabs(mDesiredTorque[14])/maxForces[14];
	// sum += fabs(mDesiredTorque[15])/maxForces[15];

	// sum += torques[0]/maxForces[6];
	// sum += torques[1]/maxForces[7];
	// sum += torques[2]/maxForces[8];
	// sum += torques[7]/maxForces[13];
	// sum += torques[8]/maxForces[14];
	// sum += torques[9]/maxForces[15];

	sum /= (double)(idx);
	return exp(-1.0 * 10.0 * sum);
}

void
Character::
SetAction(const Eigen::VectorXd& a)
{
	double action_scale = 0.1;
	mAction = a*action_scale;
	// mAction = mAction*0.5 + mAction_prev*0.5;
	// mAction_prev = mAction;

	double t = mWorld->getTime();
	this->SetTargetPosAndVel(t, mControlHz);
}

void
Character::
SetDesiredTorques()
{
	mDesiredTorque_prev = mDesiredTorque;
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.tail(mTargetPositions.rows() - mRootJointDof) += mAction;
	mDesiredTorque = this->GetSPDForces(p_des);

	mDesiredTorque = 0.5*mDesiredTorque  + 0.5*mDesiredTorque_prev;

	for(int i=0; i<mDesiredTorque.size(); i++){
		mDesiredTorque[i] = Utils::Clamp(mDesiredTorque[i], -maxForces[i], maxForces[i]);
	}

	mFemurSignals_L.pop_back();
	// mFemurSignals_L.push_front(mDesiredTorque[6]);
	mFemurSignals_L.push_front(mDesiredTorque.segment(9, 3).norm());

	mFemurSignals_R.pop_back();
	// mFemurSignals_R.push_front(mDesiredTorque[13]);
	mFemurSignals_R.push_front(mDesiredTorque.segment(13,3).norm());
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
	Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt*mKv.asDiagonal())).inverse();
	Eigen::VectorXd qdqdt = q + dq*dt;

	Eigen::VectorXd p_diff = -mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt,p_desired));
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq);
	Eigen::VectorXd ddq = M_inv*(-mSkeleton->getCoriolisAndGravityForces() + p_diff + v_diff+mSkeleton->getConstraintForces());
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
	double cycleTime = mBVH->GetMaxTime();
	int cycleCount = (int)(t/cycleTime);
	double frameTime = t;
	if(mBVH->IsCyclic())
		frameTime = t - cycleCount*cycleTime;
	if(frameTime < 0)
		frameTime += cycleTime;

	int frame = (int)(frameTime/dt);
	int frameNext = frame + 1;
	if(frameNext >= mBVH->GetNumTotalFrames())
		frameNext = frame;

	double frameFraction = (frameTime - frame*dt)/dt;

	Eigen::VectorXd p = this->GetTargetPositions(t,dt,frame,frameNext,frameFraction);
	Eigen::VectorXd v = this->GetTargetVelocities(t,dt,frame,frameNext,frameFraction);

	return std::make_pair(p,v);
}

Eigen::VectorXd
Character::
GetTargetPositions(double t,double dt,int frame,int frameNext, double frameFraction)
{
	double cycleTime = mBVH->GetMaxTime();
	int cycleCount = (int)(t/cycleTime);

	Eigen::VectorXd frameData = mBVH->GetMotion(frame);
	Eigen::VectorXd frameDataNext = mBVH->GetMotion(frameNext);

	Eigen::VectorXd p = this->GetPoseSlerp(dt, frameFraction, frameData, frameDataNext);

	Eigen::Vector3d cycleOffset = mBVH->GetCycleOffset();
	cycleOffset[1] = 0.0;
	p.segment(3,3) += cycleCount*cycleOffset;

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
	Eigen::VectorXd frameVel = mBVH->GetMotionVel(frame);
	Eigen::VectorXd frameNextVel = mBVH->GetMotionVel(frameNext);
	Eigen::VectorXd v = frameVel + frameFraction*(frameNextVel - frameVel);

	return v;
}

Eigen::VectorXd
Character::
GetPoseSlerp(double timeStep, double frameFraction, const Eigen::VectorXd& frameData, const Eigen::VectorXd& frameDataNext)
{
	double ketFrameDuration = timeStep;
	Eigen::VectorXd pose(frameData.size());

	int num_joints = mSkeleton->getNumJoints();
	for(int i=0; i<num_joints; i++)
	{
		auto joint = mSkeleton->getJoint(i);
		int idx = joint->getIndexInSkeleton(0);
		if(joint->getType()=="FreeJoint"){
			Eigen::Vector3d basePos1Start = frameData.segment(idx,3);
			Eigen::Vector3d basePos1End = frameDataNext.segment(idx,3);
			Eigen::Vector3d basePos = basePos1Start + frameFraction*(basePos1End - basePos1Start);

			pose.segment(idx,3) = basePos;
			pose.segment(idx+3,3) = Utils::GetQuaternionSlerp(frameData.segment(idx+3,3),frameDataNext.segment(idx+3,3),frameFraction);
		}
		else if(joint->getType()=="RevoluteJoint"){
			pose[idx] = frameData[idx] + frameFraction * (frameDataNext[idx] - frameData[idx]);
		}
		else if(joint->getType()=="BallJoint"){
			pose.segment(idx,3) = Utils::GetQuaternionSlerp(frameData.segment(idx,3),frameDataNext.segment(idx,3),frameFraction);
		}
	}
	return pose;
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
	if(idx==0)
		return mFemurSignals_L;
	else if(idx==1)
		return mFemurSignals_R;
}

void
Character::
SetDevice(Device* device)
{
	mDevice = device;
	mOnDevice = true;
	mUseDevice = true;
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
SetOnDevice(bool onDevice)
{
	if(onDevice ^ mOnDevice)
	{
		if(onDevice)
			this->On_Device();
		else
			this->Off_Device();
	}

	mOnDevice = onDevice;
}

void
Character::
On_Device()
{
	mDevice->Reset();

	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_Hip);
	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_LeftLeg);
	mWorld->getConstraintSolver()->addConstraint(mWeldJoint_RightLeg);
	mWorld->addSkeleton(mDevice->GetSkeleton());
}

void
Character::
Off_Device()
{
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_Hip);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_LeftLeg);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJoint_RightLeg);
	mWorld->removeSkeleton(mDevice->GetSkeleton());
}

void
Character::
SetReward_Graph()
{
	mRewards.pop_back();
	mRewards.push_front(mReward);

	mReward_map;
	(mReward_map.find("pose")->second).pop_back();
	(mReward_map.find("pose")->second).push_front(pose_reward);
	(mReward_map.find("vel")->second).pop_back();
	(mReward_map.find("vel")->second).push_front(vel_reward);
	(mReward_map.find("root")->second).pop_back();
	(mReward_map.find("root")->second).push_front(root_reward);
	(mReward_map.find("ee")->second).pop_back();
	(mReward_map.find("ee")->second).push_front(end_eff_reward);
	(mReward_map.find("com")->second).pop_back();
	(mReward_map.find("com")->second).push_front(com_reward);
	(mReward_map.find("min")->second).pop_back();
	(mReward_map.find("min")->second).push_front(min_reward);
}

void
Character::
SetPhase()
{
	double worldTime = mWorld->getTime();
	double t_phase = mBVH->GetMaxTime();
	double phi = std::fmod(worldTime, t_phase)/t_phase;
	mPhase = phi;

	if(mUseDevice)
		mDevice->SetPhase(mPhase);
}

std::deque<double>
Character::
GetRewards()
{
	return mRewards;
}

Torques::Torques()
{
}

void
Torques::
Init(dart::dynamics::SkeletonPtr skel)
{
	num_dofs = skel->getNumDofs();
	num_phase = 34;

	// mTorques_cur.resize(num_dofs);
	mTorques_cur.resize(14);
	mTorques_avg.resize(14);
	for(int i=0; i<14; i++)
	{
		mTorques_dofs_cur.push_back(std::vector<double>(num_phase));
		mTorques_dofs_avg.push_back(std::vector<double>(num_phase));
		mTorques_dofs_num.push_back(std::vector<int>(num_phase));
	}

	for(int i=0; i<num_dofs; i++)
	{
		mTorques_dofs.push_back(std::deque<double>(1200));
	}
}

void
Torques::
Reset()
{
	std::fill(mTorques_cur.begin(), mTorques_cur.end(), 0);
	std::fill(mTorques_avg.begin(), mTorques_avg.end(), 0);
	for(int i=0; i<14; i++)
	{
		for(int j=0; j<num_phase; j++)
		{
			mTorques_dofs_cur[i][j] = mTorques_dofs_avg[i][j];
		}
	}

	for(int i=0; i<num_dofs; i++)
	{
		std::fill(mTorques_dofs[i].begin(), mTorques_dofs[i].end(), 0) ;
	}
}

void
Character::
SetTorques()
{
	int num_dofs = mSkeleton->getNumDofs();
	for(int i=0; i<14; i++)
	{
		mTorques->SetTorque(i,(int)(mPhase/0.0303),mDesiredTorque[i+6]);
	}

	mTorques->SetTorqueDofs(mDesiredTorque);
}

void
Torques::
SetTorqueDofs(const Eigen::VectorXd& desTorques)
{
	for(int i=0; i<desTorques.size(); i++)
	{
		mTorques_dofs[i].pop_back();
		mTorques_dofs[i].push_front(desTorques[i]);
	}
}

void
Torques::
Set()
{
	for(int i=0; i<14; i++)
	{
		double sum = 0;
		for(int j=0; j<num_phase; j++){
			if(mTorques_dofs_cur[i][j]<0)
				sum += -1*(mTorques_dofs_cur[i][j]);
			else
				sum +=  1*(mTorques_dofs_cur[i][j]);
		}
		mTorques_cur[i] = sum;
	}
}

void
Torques::
SetTorque(int dof, int phase, double val)
{
	if(phase >= 34)
		return;

	// if(val<0)
	// 	val *= -1;

	mTorques_dofs_cur[dof][phase] = val;

	int n = mTorques_dofs_num[dof][phase];
	mTorques_dofs_avg[dof][phase] = (mTorques_dofs_avg[dof][phase]*n + val)/(double)(n+1);

	mTorques_dofs_num[dof][phase] += 1;
}

double
Torques::
GetTorque(int dof, int phase)
{
	return mTorques_dofs_cur[dof][phase];
}


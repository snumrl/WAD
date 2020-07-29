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
	:mSkeleton(nullptr),mBVH(nullptr),mDevice(nullptr),mTc(Eigen::Isometry3d::Identity()),mUseMuscle(false),mUseDevice(false),mOnDevice(false),mPhase(0.0),prev_p(Eigen::Vector3d::Zero())
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
		mMuscles.push_back(new Muscle(name,f0,lm,lt,pa,lmax));

		int num_waypoints = 0;
		for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint"); waypoint!=nullptr; waypoint = waypoint->NextSiblingElement("Waypoint"))
			num_waypoints++;

		int i = 0;
		for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))
		{
			std::string body = waypoint->Attribute("body");
			Eigen::Vector3d glob_pos = string_to_vector3d(waypoint->Attribute("p"));
			if(i==0||i==num_waypoints-1)
				mMuscles.back()->AddAnchor(mSkeleton->getBodyNode(body),glob_pos);
			else
				mMuscles.back()->AddAnchor(mSkeleton,mSkeleton->getBodyNode(body),glob_pos,2);
			i++;
		}
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
	// double kp = 300.0;
	// this->SetPDParameters(kp, sqrt(2*kp));
	this->SetPDParameters(500, 50);

	const std::string& type =
		mSkeleton->getRootBodyNode()->getParentJoint()->getType();
	if(type == "FreeJoint")
		mRootJointDof = 6;
	else if(type == "PlanarJoint")
		mRootJointDof = 3;
	else
		mRootJointDof = 0;

	this->Reset();

	mNumDof = mSkeleton->getNumDofs();
	mNumActiveDof = mNumDof - mRootJointDof;
	mNumState = this->GetState().rows();
	mAction.resize(mNumActiveDof);
	mFemurSignals_L.resize(600);
	mFemurSignals_R.resize(600);

	mJointWeights.resize(15);
    mJointWeights[0] = 0.10;
    mJointWeights[1] = 0.10;
    mJointWeights[2] = 0.06;
    mJointWeights[3] = 0.04;
    mJointWeights[4] = 0.10;
    mJointWeights[5] = 0.06;
    mJointWeights[6] = 0.04;
    mJointWeights[7] = 0.10;
    mJointWeights[8] = 0.04;
    mJointWeights[9] = 0.10;
    mJointWeights[10] = 0.06;
    mJointWeights[11] = 0.04;
    mJointWeights[12] = 0.10;
    mJointWeights[13] = 0.06;
    mJointWeights[14] = 0.04;

    mJointWeights /= 1.14;

    maxForces.resize(40);
    maxForces <<
          0, 0, 0, 0, 0, 0,
          200, 200, 200,
          150,
          90, 90, 90,
          200, 200, 200,
          150,
          90, 90, 90,
          200, 200, 200,
          150, 150, 150,
          100, 100, 100,
          60,
          0, 0, 0,
          100, 100, 100,
          60,
          0, 0, 0;
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
	mEnergy = new Energy();
	mEnergy->Init(mSkeleton);

	for(int i=0; i<33; i++)
	{
		mRewards.push_back(0.0);
		mRewards_num.push_back(0);
	}

	mEnergy_Device = new Energy();
	mEnergy_Device->Init(mSkeleton);
	for(int i=0; i<33; i++)
	{
		mRewards_Device.push_back(0.0);
		mRewards_Device_num.push_back(0);
	}
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

	mAction.setZero();
	mEnergy->Reset();

	mFemurSignals_L.clear();
	mFemurSignals_R.clear();
	mFemurSignals_L.resize(600);
	mFemurSignals_R.resize(600);

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

	// double offset = 30.0;

	// for(int i=6; i<10; i++)
	// {
	// 	if(mDesiredTorque[i] > offset)
	// 		mDesiredTorque[i] = offset;
	// 	if(mDesiredTorque[i] < -offset)
	// 		mDesiredTorque[i] = -offset;
	// }

	SetEnergy();
	SetReward_Graph();

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

	Eigen::VectorXd pos,ori,lin_v,ang_v;

	int body_num = mSkeleton->getNumBodyNodes();

	pos.resize(body_num*3+1); //3dof + root world y
	ori.resize(body_num*4); //4dof (quaternion)
	lin_v.resize(body_num*3);
	ang_v.resize(body_num*3); //dof - root_dof

	Eigen::Isometry3d origin_trans = Utils::GetOriginTrans(mSkeleton);
	Eigen::Quaterniond origin_quat(origin_trans.rotation());
	Utils::QuatNormalize(origin_quat);

	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	Eigen::Isometry3d trans = Utils::GetBodyTransform(root);
	Eigen::Vector3d root_pos = trans.translation();
	Eigen::Vector3d root_pos_rel = root_pos;

	Eigen::Vector4d root_pos_rel4 = Utils::GetPoint4d(root_pos_rel);
	root_pos_rel4 = origin_trans * root_pos_rel4;
	root_pos_rel = root_pos_rel4.segment(0,3);

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
		Eigen::Quaterniond body_ori(trans.rotation());
		Utils::QuatNormalize(body_ori);

		Eigen::Vector4d pos4 = Utils::GetPoint4d(body_pos);
		pos4 = origin_trans * pos4;
		body_pos = pos4.segment(0,3);
		body_pos -= root_pos_rel;

		pos.segment(idx_pos, 3) = body_pos.segment(0, 3);
		idx_pos += 3;

		body_ori = origin_quat * body_ori;
		Utils::QuatNormalize(body_ori);

		ori.segment(idx_ori, 4) = Utils::QuatToVec(body_ori).segment(0, 4);
		idx_ori += 4;

		Eigen::Vector3d lin_vel = body->getLinearVelocity();
		Eigen::Vector4d lin_vel4 = Utils::GetVector4d(lin_vel);;
		lin_vel4 = origin_trans * lin_vel4;
		lin_vel = lin_vel4.segment(0,3);

		lin_v.segment(idx_linv, 3) = lin_vel;
		idx_linv += 3;

		Eigen::Vector3d ang_vel = body->getAngularVelocity();
		Eigen::Vector4d ang_vel4 = Utils::GetVector4d(ang_vel);;
		ang_vel4 = origin_trans * ang_vel4;
		ang_vel = ang_vel4.segment(0,3);

		ang_v.segment(idx_angv, 3) = ang_vel;
		idx_angv += 3;
	}

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true, false, false);

	Eigen::VectorXd pos_kin,ori_kin,lin_v_kin,ang_v_kin;

	pos_kin.resize(body_num*3+1); //3dof + root world y
	ori_kin.resize(body_num*4); //4dof (quaternion)
	lin_v_kin.resize(body_num*3);
	ang_v_kin.resize(body_num*3); //dof - root_dof

	// Eigen::Isometry3d origin_trans_kin = Utils::GetOriginTrans(mSkeleton);
	// Eigen::Quaterniond origin_quat_kin(origin_trans_kin.rotation());
	// Utils::QuatNormalize(origin_quat_kin);

	dart::dynamics::BodyNode* root_kin = mSkeleton->getBodyNode(0);
	Eigen::Isometry3d trans_kin = Utils::GetBodyTransform(root_kin);
	Eigen::Vector3d root_pos_kin = trans_kin.translation();
	Eigen::Vector3d root_pos_rel_kin = root_pos_kin;

	Eigen::Vector4d root_pos_rel4_kin = Utils::GetPoint4d(root_pos_rel_kin);
	root_pos_rel4_kin = origin_trans * root_pos_rel4_kin;
	root_pos_rel_kin = root_pos_rel4_kin.segment(0,3);

	pos_kin(0) = root_pos_rel_kin[1] - pos(0);
	int idx_pos_kin = 1;
	int idx_ori_kin = 0;
	int idx_linv_kin = 0;
	int idx_angv_kin = 0;
	for(int i=0; i<body_num; i++)
	{
		dart::dynamics::BodyNode* body_kin = mSkeleton->getBodyNode(i);

		trans_kin = Utils::GetBodyTransform(body_kin);
		Eigen::Vector3d body_pos_kin = trans_kin.translation();
		Eigen::Quaterniond body_ori_kin(trans_kin.rotation());
		Utils::QuatNormalize(body_ori_kin);

		Eigen::Vector4d pos4_kin = Utils::GetPoint4d(body_pos_kin);
		pos4_kin = origin_trans * pos4_kin;
		body_pos_kin = pos4_kin.segment(0,3);
		body_pos_kin -= root_pos_rel_kin;

		pos_kin.segment(idx_pos_kin, 3) = body_pos_kin.segment(0, 3) - pos.segment(idx_pos_kin, 3);
		idx_pos_kin += 3;

		body_ori_kin = origin_quat * body_ori_kin;
		Utils::QuatNormalize(body_ori_kin);

		Eigen::Quaterniond qDiff = Utils::QuatDiff( body_ori_kin, Utils::VecToQuat(ori.segment(idx_ori_kin, 4)));
		Utils::QuatNormalize(qDiff);

		ori_kin.segment(idx_ori_kin, 4) = Utils::QuatToVec(qDiff).segment(0, 4);
		idx_ori_kin += 4;

		Eigen::Vector3d lin_vel_kin = body_kin->getLinearVelocity();
		Eigen::Vector4d lin_vel4_kin = Utils::GetVector4d(lin_vel_kin);
		lin_vel4_kin = origin_trans * lin_vel4_kin;
		lin_vel_kin = lin_vel4_kin.segment(0,3);

		lin_v_kin.segment(idx_linv_kin, 3) = lin_vel_kin - lin_v.segment(idx_linv_kin, 3);
		idx_linv_kin += 3;

		Eigen::Vector3d ang_vel_kin = body_kin->getAngularVelocity();
		Eigen::Vector4d ang_vel4_kin = Utils::GetVector4d(ang_vel_kin);;
		ang_vel4_kin = origin_trans * ang_vel4_kin;
		ang_vel_kin = ang_vel4_kin.segment(0,3);

		ang_v_kin.segment(idx_angv_kin, 3) = ang_vel_kin - ang_v.segment(idx_angv_kin, 3);
		idx_angv_kin += 3;
	}

	// Eigen::Vector3d root_diff = (origin_trans * Utils::GetVector4d(root_pos_kin - root_pos)).segment(0,3);

	// Eigen::VectorXd state(pos.rows()+ori.rows()+lin_v.rows()+ang_v.rows()+pos_kin.rows()+ori_kin.rows()+lin_v_kin.rows()+ang_v_kin.rows()+ root_diff.size());
	Eigen::VectorXd state(pos.rows()+ori.rows()+lin_v.rows()+ang_v.rows()+pos_kin.rows()+ori_kin.rows()+lin_v_kin.rows()+ang_v_kin.rows());
	this->SetPhase();

	mSkeleton->setPositions(cur_pos);
	mSkeleton->setVelocities(cur_vel);
	mSkeleton->computeForwardKinematics(true, false, false);

	state<<pos,ori,lin_v,ang_v,pos_kin,ori_kin,lin_v_kin,ang_v_kin;
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
	double pose_w = 0.50;
	double vel_w = 0.05;
	double end_eff_w = 0.15;
	double root_w = 0.20;
	double com_w = 0.10;

	double total_w = pose_w + vel_w + end_eff_w + root_w + com_w;
	pose_w /= total_w;
	vel_w /= total_w;
	end_eff_w /= total_w;
	root_w /= total_w;
	com_w /= total_w;

	double pose_scale = 2;
    double vel_scale = 0.1;
    double end_eff_scale = 40;
    double root_scale = 5;
    double com_scale = 10;
    double err_scale = 2;  // error scale

    double reward = 0;

    double pose_err = 0;
    double vel_err = 0;
    double end_eff_err = 0;
    double root_err = 0;
    double com_err = 0;

	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

    Eigen::Vector3d comSim, comSimVel;
    comSim = mSkeleton->getCOM();
    comSimVel = mSkeleton->getCOMLinearVelocity();

    int num_joints = mSkeleton->getNumJoints();

    double root_rot_w = mJointWeights[0]; // mJointWeights

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

	double pose_reward = exp(-err_scale * pose_scale * pose_err);
    double vel_reward = exp(-err_scale * vel_scale * vel_err);
    double end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);
    double root_reward = exp(-err_scale * root_scale * root_err);
    double com_reward = exp(-err_scale * com_scale * com_err);

    // double r_ = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + root_w * root_reward + com_w * com_reward;
    double r_ = pose_reward * vel_reward * end_eff_reward * root_reward * com_reward;

    // std::cout << pose_w * pose_reward << " / " << pose_w << "  " << pose_w * (pose_reward - 1.0) << " pose" << std::endl;
    // std::cout << vel_w * vel_reward << " / " << vel_w << "  " << vel_w * (vel_reward - 1.0) << " vel" << std::endl;
    // std::cout << end_eff_w * end_eff_reward << " / " << end_eff_w << "  " << end_eff_w * (end_eff_reward - 1.0) << " ee" << std::endl;
    // std::cout << root_w * root_reward << " / " << root_w << "  " << root_w * (root_reward - 1.0) << " root"<< std::endl;
    // std::cout << com_w * com_reward << " / " << com_w << "  " << com_w * (com_reward - 1.0) << " com" << std::endl;
    // std::cout << std::endl;

    mSkeleton->setPositions(cur_pos);
	mSkeleton->setVelocities(cur_vel);
	mSkeleton->computeForwardKinematics(true, false, false);

    return r_;
}

std::map<std::string,double>
Character::
GetRewardSep()
{
	std::map<std::string, double> r_sep;
	if(mUseDevice)
	{
		// r_sep["r_character"] = w_character*r_character;
		// r_sep["r_device"] = w_device*r_device;
	}
	else{
		r_sep["r_q"] = w_q*r_q;
		r_sep["r_v"] = w_v*r_v;
		r_sep["r_ee"] = w_ee*r_ee;
		r_sep["r_com"] = w_com*r_com;
	}

	return r_sep;
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

	// for(int i=0; i<46; i++)
	for(int i=0; i<40; i++)
	{
		if(mDesiredTorque[i] > maxForces[i])
			mDesiredTorque[i] = maxForces[i];
		if(mDesiredTorque[i] < -maxForces[i])
			mDesiredTorque[i] = -maxForces[i];
	}

	mFemurSignals_R.pop_back();
	mFemurSignals_R.push_front(0.1*mDesiredTorque[6]);

	mFemurSignals_L.pop_back();
	mFemurSignals_L.push_front(0.1*mDesiredTorque[13]);

	// if(mUseDeviceNN){
	// 	mDevice->SetDesiredTorques2();
	// 	Eigen::VectorXd des = mDevice->GetDesiredTorques2();
	// 	mDesiredTorque[6] += des[0];
	// 	mDesiredTorque[15] += des[1];
	// }
}

Eigen::VectorXd
Character::
GetDesiredTorques()
{
	return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}

void
Character::
SetPDParameters(double kp, double kv)
{
	int dof = mSkeleton->getNumDofs();
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
	Eigen::VectorXd p = this->GetTargetPositions(t,dt);
	Eigen::Isometry3d Tc = mTc;
	Eigen::VectorXd p1 = this->GetTargetPositions(t+dt,dt);
	mTc = Tc;

	Eigen::VectorXd v = Eigen::VectorXd::Zero(p.rows());
	int offset = 0;
	for(int i=0;i<mSkeleton->getNumJoints();i++)
	{
		Joint* jn = mSkeleton->getJoint(i);
		if(jn->getType() == "FreeJoint"){
			v.segment<3>(offset) = Utils::CalcQuaternionVel(p,p1,dt);
			v.segment<3>(offset+3) = (p1.segment<3>(offset+3)-p.segment<3>(offset+3))/dt;
			offset += 6;
		}
		else if(jn->getType() == "BallJoint")
		{
			v.segment<3>(offset) = Utils::CalcQuaternionVelRel(p,p1,dt);
			offset += 3;
		}
		else if(jn->getType() == "RevoluteJoint")
		{
			v[offset] = (p1[offset]-p[offset])/dt;
			offset += 1;
		}
		else if(jn->getType() == "WeldJoint")
		{

		}
	}

	return std::make_pair(p,v);
}

Eigen::VectorXd
Character::
GetTargetPositions(double t,double dt)
{
	Eigen::VectorXd p = mBVH->GetMotion(t);
	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(p.head<6>());
	T_current = mBVH->GetT0().inverse()*T_current;
	Eigen::Isometry3d T_head = mTc*T_current;
	Eigen::Vector6d p_head = dart::dynamics::FreeJoint::convertToPositions(T_head);
	p.head<6>() = p_head;

	if(mBVH->IsCyclic())
	{
		double t_mod = std::fmod(t, mBVH->GetMaxTime());
		t_mod = t_mod/mBVH->GetMaxTime();

		double r = 0.95;
		if(t_mod>r)
		{
			double ratio = 1.0/(r-1.0)*t_mod - 1.0/(r-1.0);
			Eigen::Isometry3d T01 = mBVH->GetT1()*(mBVH->GetT0().inverse());
			double delta = T01.translation()[1];
			delta *= ratio;
			p[5] += delta;
		}

		double tdt_mod = std::fmod(t+dt, mBVH->GetMaxTime());
		if(tdt_mod-dt<0.0){
			Eigen::Isometry3d T01 = mBVH->GetT1()*(mBVH->GetT0().inverse());
			Eigen::Vector3d p01 = dart::math::logMap(T01.linear());
			p01[0] =0.0;
			p01[2] =0.0;
			T01.linear() = dart::math::expMapRot(p01);

			mTc = T01*mTc;
			mTc.translation()[1] = 0.0;
		}
	}

	return p;
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
        return mFemurSignals_R;
    else if(idx==1)
        return mFemurSignals_L;
}


void
Character::
SetDevice(Device* device)
{
	mDevice = device;
	mUseDevice = true;
	mOnDevice = true;
	this->SetConstraints();
}

void
Character::
SetConstraints()
{
	mWeldJoint_Hip = std::make_shared<dart::constraint::WeldJointConstraint>(
        mSkeleton->getBodyNode(0), mDevice->GetSkeleton()->getBodyNode(0)
        );

	mWeldJoint_LeftLeg = std::make_shared<dart::constraint::WeldJointConstraint>(
       mSkeleton->getBodyNode("FemurL"), mDevice->GetSkeleton()->getBodyNode("FastenerLeftOut")
        );

	mWeldJoint_RightLeg = std::make_shared<dart::constraint::WeldJointConstraint>(
        mSkeleton->getBodyNode("FemurR"), mDevice->GetSkeleton()->getBodyNode("FastenerRightOut")
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
SetEnergy()
{
	int offset = 6;
	int n = mSkeleton->getNumJoints();
	for(int i=1; i<n; i++)
	{
		std::string name = mSkeleton->getJoint(i)->getName();
		int dof = mSkeleton->getJoint(i)->getNumDofs();
		double torque = 0.0;
		if(dof == 1)
		{
			torque = mDesiredTorque[offset];
			if(torque<0)
				torque *= -1;
			offset += 1;
		}
		else if(dof == 3)
		{
			Eigen::Vector3d t_ = mDesiredTorque.segment(offset,3);
			torque = t_.norm();
			offset += 3;
		}

		if(mOnDevice)
			mEnergy_Device->SetEnergy(name, (int)(mPhase/0.0303), torque);
		else
			mEnergy->SetEnergy(name, (int)(mPhase/0.0303), torque);
	}
}

void
Character::
SetReward_Graph()
{
	this->GetReward();
	int phase_idx = (int)(mPhase/0.0303);
	if(mOnDevice){
		int n = mRewards_Device_num[phase_idx];
		if(n == 0)
			mRewards_Device[phase_idx] = mReward;
		else
			mRewards_Device[phase_idx] = (mRewards_Device[phase_idx]*n+mReward)/(double)(n+1);

		mRewards_Device_num[phase_idx] += 1;
	}
	else
	{
		int n = mRewards_num[phase_idx];
		if(n == 0)
			mRewards[phase_idx] = mReward;
		else
			mRewards[phase_idx] = (mRewards[phase_idx]*n+mReward)/(double)(n+1);

		mRewards_num[phase_idx] += 1;
	}
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

std::vector<double>
Character::
GetReward_Graph(int idx)
{
	if(idx==0) // OFF Device
		return mRewards;
	else // ON Device
		return mRewards_Device;
}

std::map<std::string, std::vector<double>>
Character::
GetEnergy(int idx)
{
	if(idx==0)
		return mEnergy->Get();
	else
		return mEnergy_Device->Get();
}

Energy::Energy()
{
}

void
Energy::
Init(dart::dynamics::SkeletonPtr skel)
{
	int n = skel->getNumJoints();
	for(int i=1; i<n; i++)
	{
		std::string name = skel->getJoint(i)->getName();
		std::vector<double> e(34, 0.0);
		std::vector<int> e_num(34, 0);
		mE.insert(std::make_pair(name, e));
		mE_num.insert(std::make_pair(name, e_num));
	}
}

void
Energy::
Reset()
{
}

void
Energy::
SetEnergy(std::string name, int t, double val)
{
	int n = (mE_num.find(name)->second).at(t);
	if(n==0)
		(mE.find(name)->second).at(t) = val;
	else
		(mE.find(name)->second).at(t) = ((mE.find(name)->second).at(t)*n + val)/(double)(n+1);

	(mE_num.find(name)->second).at(t) += 1;
}

double
Energy::
GetEnergy(std::string name, int t)
{
	return (mE.find(name)->second).at(t);
}


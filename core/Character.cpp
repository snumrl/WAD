#include "Character.h"
#include "BVH.h"
#include "DARTHelper.h"
#include "Muscle.h"
#include "Device.h"
#include <tinyxml.h>
using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

double exp_of_squared(const Eigen::VectorXd& vec,double w);
double exp_of_squared(const Eigen::Vector3d& vec,double w);
double exp_of_squared(double val,double w);
double Pulse_Constant(double t);
double Pulse_Linear(double t);
double Pulse_Period(double t);
double Pulse_Period(double t, double offset);

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

	double kp = 300.0;
	this->SetPDParameters(kp, sqrt(2*kp));

	const std::string& type =
		mSkeleton->getRootBodyNode()->getParentJoint()->getType();
	if(type == "FreeJoint")
		mRootJointDof = 6;
	else if(type == "PlanarJoint")
		mRootJointDof = 3;
	else
		mRootJointDof = 0;

	mNumDof = mSkeleton->getNumDofs();
	mNumActiveDof = mNumDof - mRootJointDof;
	mNumState = this->GetState().rows();
	mAction.resize(mNumActiveDof);
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
	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	int num_body_nodes = mSkeleton->getNumBodyNodes();
	Eigen::VectorXd p,v;

	p.resize((num_body_nodes-1)*3);
	v.resize((num_body_nodes)*3);

	for(int i = 1;i<num_body_nodes;i++)
	{
		p.segment<3>(3*(i-1)) = mSkeleton->getBodyNode(i)->getCOM(root);
		v.segment<3>(3*(i-1)) = mSkeleton->getBodyNode(i)->getCOMLinearVelocity();
	}
	v.tail<3>() = root->getCOMLinearVelocity();

	p *= 0.8;
	v *= 0.2;

	Eigen::VectorXd state(p.rows()+v.rows()+1);
	this->SetPhase();

	state<<p,v,mPhase;
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
	Eigen::VectorXd cur_pos = mSkeleton->getPositions();
	Eigen::VectorXd cur_vel = mSkeleton->getVelocities();

	Eigen::VectorXd p_diff_all = mSkeleton->getPositionDifferences(mTargetPositions, cur_pos);
	Eigen::VectorXd v_diff_all = mSkeleton->getPositionDifferences(mTargetVelocities, cur_vel);

	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());

	const auto& bvh_map = mBVH->GetBVHMap();

	for(auto ss : bvh_map)
	{
		auto joint = mSkeleton->getBodyNode(ss.first)->getParentJoint();
		int idx = joint->getIndexInSkeleton(0);
		if(joint->getType()=="FreeJoint"){
			continue;
		}
		else if(joint->getType()=="RevoluteJoint"){
			p_diff[idx] = p_diff_all[idx];
			v_diff[idx] = v_diff_all[idx];
		}
		else if(joint->getType()=="BallJoint"){
			p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
			v_diff.segment<3>(idx) = v_diff_all.segment<3>(idx);
		}
	}

	auto ees = this->GetEndEffectors();
	Eigen::VectorXd ee_diff(ees.size()*3);
	Eigen::VectorXd com_diff;
	for(int i =0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) = ees[i]->getCOM();

	com_diff = mSkeleton->getCOM();
	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->computeForwardKinematics(true, false, false);
	com_diff -= mSkeleton->getCOM();

	for(int i=0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) -= ees[i]->getCOM()+com_diff;

	mSkeleton->setPositions(cur_pos);
	mSkeleton->computeForwardKinematics(true, false, false);

	r_q = exp_of_squared(p_diff, 2.0);
	r_v = exp_of_squared(v_diff, 0.1);
	r_ee = exp_of_squared(ee_diff, 40.0);
	r_com = exp_of_squared(com_diff, 10.0);

	// double r_ = r_ee*(w_q*r_q + w_v*r_v);
	w_q = 0.6;
	w_v = 0.2;
	w_com = 0.3;
	double r_ = r_ee*(w_q*r_q + w_v*r_v + w_com*r_com);
	// std::cout << "w_q : " << w_q << std::endl;
	// std::cout << "w_v : " << w_v << std::endl;
	// std::cout << "w_com : " << w_com << std::endl;

	// std::cout << w_q*r_q << " " << (w_v*r_v)/(w_q*r_q) << " " << (w_com*r_com)/(w_q*r_q) << std::endl;
	// std::cout << w_q*r_q << " " << (w_v*r_v)/(w_q*r_q) << std::endl;
	// std::cout << std::endl;
	// std::cout << "r : " << r_ << std::endl;
	// std::cout << "r_ee : " << r_ee << std::endl;
	// std::cout << "r_q : " << w_q*r_q << std::endl;
	// std::cout << "r_v : " << w_v*r_v << std::endl;
	// std::cout << "r_com : " << w_com*r_com << std::endl;
	// std::cout << "v_diff : " << v_diff.squaredNorm() << std::endl;

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
	mKp = Eigen::VectorXd::Constant(dof,kp);
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

	return std::make_pair(p,(p1-p)/dt);
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

double exp_of_squared(const Eigen::VectorXd& vec,double w)
{
	return exp(-w*vec.squaredNorm());
}

double exp_of_squared(const Eigen::Vector3d& vec,double w)
{
	return exp(-w*vec.squaredNorm());
}

double exp_of_squared(double val,double w)
{
	return exp(-w*val*val);
}

double Pulse_Constant(double t)
{
	double ratio = 1.0;
	return ratio;
}

double Pulse_Linear(double t)
{
	double ratio = 1.0;
	if(t <= 0.5)
		ratio = t/0.5;
	else
		ratio = (1.0-t)/0.5;
	return ratio;
}

double Pulse_Period(double t)
{
	double ratio = 1.0;
	if(t <= 0.5)
		ratio = 1.0;
	else
		ratio = 0.0;
	return ratio;
}

double Pulse_Period(double t, double offset)
{
	double ratio = 1.0;
	if(t <= offset)
		ratio = 1.0;
	else
		ratio = 0.0;
	return ratio;
}

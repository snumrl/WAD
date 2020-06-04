#include "Character.h"
#include "BVH.h"
#include "DARTHelper.h"
#include "Muscle.h"
#include "Device.h"
#include <tinyxml.h>
using namespace dart;
using namespace dart::dynamics;
using namespace MASS;
Character::
Character()
	:mSkeleton(nullptr),mBVH(nullptr),mDevice(nullptr),mTc(Eigen::Isometry3d::Identity()),w_q(0.75),w_v(0.1),w_ee(0.0),w_com(0.15),w_character(1.0),w_device(0.0),r_q(0.0),r_v(0.0),r_ee(0.0),r_com(0.0),r_character(0.0),r_device(0.0),mUseMuscle(false),mUseDevice(false),mOnDevice(false)
{

}

void
Character::
LoadSkeleton(const std::string& path,bool create_obj)
{
	mSkeleton = BuildFromFile(path,create_obj);
	std::map<std::string,std::string> bvh_map;
	TiXmlDocument doc;
	doc.LoadFile(path);
	TiXmlElement *skel_elem = doc.FirstChildElement("Skeleton");

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

	mBVH->Parse(path,cyclic);
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
}

void
Character::
LoadDevice(const std::string& path)
{
	mDevice = new Device(BuildFromFile(path));
	mOnDevice = true;
}

void
Character::
Initialize()
{
	if(this->GetSkeleton() == nullptr)
	{
		std::cout<<"Initialize Character First"<<std::endl;
		exit(0);
	}

	if(this->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="FreeJoint")
		mRootJointDof = 6;
	else if(this->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="PlanarJoint")
		mRootJointDof = 3;
	else
		mRootJointDof = 0;

	mNumActiveDof = this->GetSkeleton()->getNumDofs()-mRootJointDof;
	mNumState = this->GetState(0.0).rows();
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
Initialize_Device(dart::simulation::WorldPtr& wPtr)
{
	mUseDevice = true;
	mDevice->Initialize();

	mDesiredTorque_Device = Eigen::VectorXd::Zero(12);

	mWeldJoint_Hip = std::make_shared<dart::constraint::WeldJointConstraint>(
        mSkeleton->getBodyNode(0), mDevice->GetSkeleton()->getBodyNode(0)
        );

    mWeldJoint_LeftLeg = std::make_shared<dart::constraint::WeldJointConstraint>(
        mSkeleton->getBodyNode("FemurL"), mDevice->GetSkeleton()->getBodyNode("FastenerLeftOut")
        );

    mWeldJoint_RightLeg = std::make_shared<dart::constraint::WeldJointConstraint>(
        mSkeleton->getBodyNode("FemurR"), mDevice->GetSkeleton()->getBodyNode("FastenerRightOut")
        );

    wPtr->getConstraintSolver()->addConstraint(mWeldJoint_Hip);
	wPtr->getConstraintSolver()->addConstraint(mWeldJoint_LeftLeg);
	wPtr->getConstraintSolver()->addConstraint(mWeldJoint_RightLeg);
	wPtr->addSkeleton(mDevice->GetSkeleton());

	mDesiredTorque_Device = Eigen::VectorXd::Zero(12);
	mTorqueMax_Device = 15.0;
}

void
Character::
On_Device(dart::simulation::WorldPtr& wPtr)
{
	Reset_Device();
	wPtr->getConstraintSolver()->addConstraint(mWeldJoint_Hip);
	wPtr->getConstraintSolver()->addConstraint(mWeldJoint_LeftLeg);
	wPtr->getConstraintSolver()->addConstraint(mWeldJoint_RightLeg);
	wPtr->addSkeleton(mDevice->GetSkeleton());
}

void
Character::
Off_Device(dart::simulation::WorldPtr& wPtr)
{
	wPtr->getConstraintSolver()->removeConstraint(mWeldJoint_Hip);
	wPtr->getConstraintSolver()->removeConstraint(mWeldJoint_LeftLeg);
	wPtr->getConstraintSolver()->removeConstraint(mWeldJoint_RightLeg);
	wPtr->removeSkeleton(mDevice->GetSkeleton());
}

void
Character::
Reset(double worldTime, int controlHz)
{
	mTc = mBVH->GetT0();
	mTc.translation()[1] = 0.0;

	mSkeleton->clearConstraintImpulses();
	mSkeleton->clearInternalForces();
	mSkeleton->clearExternalForces();

	this->SetTargetPosAndVel(worldTime, controlHz);

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true,false,false);

	if(mUseMuscle)
		Reset_Muscles();

	if(mUseDevice)
		Reset_Device();
}

void
Character::
Reset_Muscles()
{
	int muscleSize = this->GetMuscles().size();
	mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(mNumTotalRelatedDof);
	mCurrentMuscleTuple.L = Eigen::MatrixXd::Zero(mNumActiveDof, muscleSize);
	mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof);
	mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof);
	mActivationLevels = Eigen::VectorXd::Zero(muscleSize);
}

void
Character::
Reset_Device()
{
	mDevice->Reset();

	Eigen::VectorXd p(mDevice->GetSkeleton()->getNumDofs());
    Eigen::VectorXd v(mDevice->GetSkeleton()->getNumDofs());

    p.setZero();
    v.setZero();
    p.head(6) = mSkeleton->getPositions().head(6);
    v.head(6) = mSkeleton->getVelocities().head(6);
    p.segment<3>(6) = mSkeleton->getJoint("FemurL")->getPositions();
    p.segment<3>(9) = mSkeleton->getJoint("FemurR")->getPositions();
    v.segment<3>(6) = mSkeleton->getJoint("FemurL")->getVelocities();
    v.segment<3>(9) = mSkeleton->getJoint("FemurR")->getVelocities();

    mDevice->GetSkeleton()->setPositions(p);
    mDevice->GetSkeleton()->setVelocities(v);
    mDevice->GetSkeleton()->computeForwardKinematics(true, false, false);
}

void
Character::
StepBack()
{
	// mSkeleton->setPositions(mStoredPositions);
	// mSkeleton->setVelocities(mStoredVelocities);
	// mSkeleton->computeForwardKinematics(true, false, false);

	// mDevice->GetSkeleton()->setPositions(mStoredPositions_Device);
	// mDevice->GetSkeleton()->setVelocities(mStoredVelocities_Device);
	// mDevice->GetSkeleton()->computeForwardKinematics(true, false, false);
}

void
Character::
Step()
{
	GetDesiredTorques();

	double offset = 30.0;

	for(int i=6; i<10; i++)
	{
		if(mDesiredTorque[i] > offset)
			mDesiredTorque[i] = offset;
		if(mDesiredTorque[i] < -offset)
			mDesiredTorque[i] = -offset;
	}

	mSkeleton->setForces(mDesiredTorque);

	// int dofs = 0;
	// int n = mSkeleton->getNumJoints();
	// for(int i=0; i<n; i++)
	// {
	// 	std::cout << i << " : " << mSkeleton->getJoint(i)->getName() << " " << mSkeleton->getJoint(i)->getNumDofs()  << std::endl;
	// 	std::cout << i << " : " << mSkeleton->getJoint(i)->getName() << " " << mSkeleton->getJoint(i)->getForceLowerLimit(0)  << std::endl;
	// 	dofs += mSkeleton->getJoint(i)->getNumDofs();
	// }

	// std::cout << "dofs : " << dofs << std::endl;
}

void
Character::
Step_Muscles(int simCount, int randomSampleIndex)
{
	int count = 0;
	for(auto muscle : mMuscles)
	{
		muscle->activation = mActivationLevels[count++];
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

void
Character::
Step_Device(const Eigen::VectorXd& a_)
{
	mDevice->GetSkeleton()->setForces(a_);
}

void
Character::
Step_Device()
{
	GetDesiredTorques_Device();

	double offset = 30.0;

	for(int i=6; i<12; i++)
	{
		if(mDesiredTorque_Device[i] > offset)
			mDesiredTorque_Device[i] = offset;
		if(mDesiredTorque_Device[i] < -offset)
			mDesiredTorque_Device[i] = -offset;
	}

	mDevice->GetSkeleton()->setForces(mDesiredTorque_Device);

	// std::cout << "device L : " << mDesiredTorque_Device[6] << " " << mDesiredTorque_Device[7] << " " << mDesiredTorque_Device[8] << std::endl;
	// std::cout << "device R : " << mDesiredTorque_Device[9] << " " << mDesiredTorque_Device[10] << " " << mDesiredTorque_Device[11] << std::endl;

	// int n = mDevice->GetSkeleton()->getNumJoints();
	// int dofs = mDevice->GetSkeleton()->getNumDofs();
	// for(int i=0; i<n; i++)
	// {
	// 	Eigen::VectorXd f_ = mDevice->GetSkeleton()->getForces();
	// 	for(int j=0; j<f_.size(); j++)
	// 	{
	// 		std::cout << "device forces " << j << " : " << f_[j] << std::endl;
	// 	}

	// 	std::cout << "device joint num : " << n << std::endl;
	// 	std::cout << "device dofs  : " << dofs << std::endl;
	// 	std::cout << i << " : " << mDevice->GetSkeleton()->getJoint(i)->getName() << " " << mDevice->GetSkeleton()->getJoint(i)->getNumDofs() << std::endl;
	// 	std::cout << i << " : " << mDevice->GetSkeleton()->getJoint(i)->getName() << " " << mDevice->GetSkeleton()->getJoint(i)->getForces() << std::endl;
	// }
}

void
Character::
SetAction(const Eigen::VectorXd& a)
{
	mAction_ = a;
}

void
Character::
SetAction_Device(const Eigen::VectorXd& a)
{
	mAction_Device = a;
}

Eigen::VectorXd
Character::
GetState(double worldTime)
{
	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	int num_body_nodes = mSkeleton->getNumBodyNodes()-1;
	Eigen::VectorXd p,v;

	p.resize((num_body_nodes-1)*3);
	v.resize((num_body_nodes)*3);

	for(int i = 1;i<num_body_nodes;i++)
	{
		p.segment<3>(3*(i-1)) = mSkeleton->getBodyNode(i)->getCOM(root);
		v.segment<3>(3*(i-1)) = mSkeleton->getBodyNode(i)->getCOMLinearVelocity();
	}

	v.tail<3>() = root->getCOMLinearVelocity();

	double t_phase = mBVH->GetMaxTime();
	double phi = std::fmod(worldTime, t_phase)/t_phase;

	p *= 0.8;
	v *= 0.2;

	Eigen::VectorXd state(p.rows()+v.rows()+1);

	state<<p,v,phi;
	return state;
}

Eigen::VectorXd
Character::
GetState_Device(double worldTime)
{
	double maxTime = mBVH->GetMaxTime();
	return mDevice->GetState(worldTime, maxTime);
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

void
Character::
SetRewardParameters(double w_q,double w_v,double w_ee,double w_com)
{
	this->w_q = w_q;
	this->w_v = w_v;
	this->w_ee = w_ee;
	this->w_com = w_com;
}

void
Character::
SetRewardParameters_Device()
{

}

double
Character::
GetReward()
{
	if(mUseDevice){
		// r_character = this->GetReward_Character();
		// return (w_character*r_character - r_cur)*10000.0;
		// std::cout << "reward : " << 0.1*exp((r_device - r_cur)*100) << std::endl;
		// return 0.1*exp((r_device - r_cur)*10);
		r_character = this->GetReward_Character();
		return r_character;
	}
	else
		r_character = this->GetReward_Character();
		return r_character;
}

std::map<std::string,double>
Character::
GetRewardSep()
{
	std::map<std::string, double> r_sep;
	if(mUseDevice)
	{
		r_sep["r_character"] = w_character*r_character;
		r_sep["r_device"] = w_device*r_device;
	}
	else{
		r_sep["r_q"] = w_q*r_q;
		r_sep["r_v"] = w_v*r_v;
		r_sep["r_ee"] = w_ee*r_ee;
		r_sep["r_com"] = w_com*r_com;
	}

	return r_sep;
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
		if(joint->getType()=="FreeJoint")
			continue;
		else if(joint->getType()=="RevoluteJoint")
			p_diff[idx] = p_diff_all[idx];
		else if(joint->getType()=="BallJoint")
			p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
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

	double r_ = r_ee*(w_q*r_q + w_v*r_v);

	return r_;
}

double
Character::
GetReward_Device()
{
	return exp(-1.*(mDesiredTorque_Device/mTorqueMax_Device).squaredNorm());;
}

Eigen::VectorXd
Character::
GetDesiredTorques()
{
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.tail(mTargetPositions.rows() - mRootJointDof) += mAction_;
	mDesiredTorque = this->GetSPDForces(p_des);

	return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}

Eigen::VectorXd
Character::
GetDesiredTorques_Device()
{
	mDesiredTorque_Device.head<6>().setZero();
	mDesiredTorque_Device.segment<6>(6) = mTorqueMax_Device * mAction_Device;

	return mDesiredTorque_Device;
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
SetPDParameters(double kp, double kv)
{
	int dof = mSkeleton->getNumDofs();
	mKp = Eigen::VectorXd::Constant(dof,kp);
	mKv = Eigen::VectorXd::Constant(dof,kv);
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

void
Character::
SetTargetPosAndVel(double t, int controlHz)
{
	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = this->GetTargetPosAndVel(t, 1.0/controlHz);
	mTargetPositions = pv.first;
	mTargetVelocities = pv.second;
}

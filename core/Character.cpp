#include "Character.h"
#include "Device.h"
#include <tinyxml.h>
#include <ctime>

namespace MASS
{

Character::
Character(WorldPtr& wPtr)
	:mSkeleton(nullptr),mBVH(nullptr),mDevice(nullptr),mTc(Eigen::Isometry3d::Identity()),mUseMuscle(false),mUseDevice(false),mOnDevice(false),mNumParamState(0),mMass(0)
{
	this->SetWorld(wPtr);

	mMassRatio = 1.0;
	mForceRatio = 1.0;
	mSpeedRatio = 1.0;

	mLowerBody = false;
	mLowerMuscleRelatedDof = 30;
	mUpperMuscleRelatedDof = 26;

	mAdaptiveMotion = true;
	mAdaptiveLowerBody = false;
	mAdaptiveLowerDof = 24;
	mAdaptiveUpperDof = 32;

	mTimeOffset = 2.0;
}

Character::
~Character()
{
	for(int i=0; i<mEndEffectors.size(); i++)
		delete(mEndEffectors[i]);

	for(int i=0; i<mMuscles.size(); i++)
		delete(mMuscles[i]);

	for(int i=0; i<mMusclesFemur.size(); i++)
		delete(mMusclesFemur[i]);

	for(int i=0; i<mBVHset.size(); i++)
		delete(mBVHset[i]);

	delete mBVH;
	delete mDevice;
	delete mContacts;
	delete mJointDatas;
	delete mMetabolicEnergy;
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

bool
Character::
isLowerBody(std::string& body)
{
	if(body == "ShoulderL" || body == "ArmL" || body == "ForeArmL" || body == "HandL" || body == "ShoulderR" || body == "ArmR" || body == "ForeArmR" || body == "HandR" || body == "Head" || body == "Neck")
		return false;
	else
		return true;
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
				isExist = this->isLowerBody(body);
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
				mMusclesMap[sub_name].push_back(muscle_elem);
			}
			else{
				std::string sub_name = muscle_name.substr(2,name_size-2);
				if(mMusclesMap.count(sub_name) == 0){
					mMusclesMap.insert(std::make_pair(sub_name, std::vector<Muscle*>()));
					mMusclesMap[sub_name].push_back(muscle_elem);
				}
				else{
					mMusclesMap[sub_name].push_back(muscle_elem);
				}
			}

			if(muscle_elem->GetFemur())
			{
				muscle_elem->SetMt0Ratio(1.0);
				muscle_elem->SetF0Ratio(1.0);
				mMusclesFemur.push_back(muscle_elem);
			}

			mMuscles.push_back(muscle_elem);
		}
	}
	mNumMuscle = mMuscles.size();
	mNumMuscleMap = mMusclesMap.size();
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
	mNumJoints = mSkeleton->getNumJoints();
	mNumBodyNodes = mSkeleton->getNumBodyNodes();

	mRootJointDof = 6; // free joint
	mNumActiveDof = mDof - mRootJointDof;
	mNumAction = mNumActiveDof;
	if(mAdaptiveMotion)
	{
		if(mAdaptiveLowerBody)
			mNumAdaptiveSpatialDof = mAdaptiveLowerDof;
		else
			mNumAdaptiveSpatialDof = mAdaptiveLowerDof+mAdaptiveUpperDof;
		mNumAdaptiveTemporalDof = 1;
		mNumAdaptiveDof = mNumAdaptiveSpatialDof + mNumAdaptiveTemporalDof;
		mNumAction += mNumAdaptiveDof;
	}

	mAction = Eigen::VectorXd::Zero(mNumAction);
	mAction_prev = Eigen::VectorXd::Zero(mNumAction);
	mDesiredTorque = Eigen::VectorXd::Zero(mDof);

	mAdaptiveTime = 0.0;
	mTemporalDisplacement = 0.0;

	this->SetPDParameters();
	this->GetPosAndVel(mWorld->getTime(), mTargetPositions, mTargetVelocities);
	this->GetPosAndVel(mWorld->getTime(), mReferencePositions, mReferenceVelocities);

	mNumStateChar = this->GetState().rows();
	mNumState = mNumStateChar;

	mTc = mBVH->GetT0();
	mTc.translation()[1] = 0.0;

	mStepCnt = 0;
	mStepCntTotal = 0;

	mCurVel = 0.0;
	mCurCoT = 0.0;

	this->Initialize_JointWeights();
	this->Initialize_Rewards();
	this->Initialize_Forces();
	this->Initialize_Mass();
	if(mUseMuscle)
		this->Initialize_Muscles();

	mFemurSignals.push_back(std::deque<double>(1200));
	mFemurSignals.push_back(std::deque<double>(1200));

	for(int i=0; i<32; i++)
		mRootTrajectory.push_back(Eigen::Vector3d::Zero());
	for(int i=0; i<32; i++)
		mHeadTrajectory.push_back(Eigen::Vector3d::Zero());
	for(int i=0; i<200; i++)
		mComHistory.push_back(Eigen::Vector4d::Zero());

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
	mRewardTags.push_back("imit");
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
	mAdaptiveTime = worldTime;
	mTemporalDisplacement = 0.0;
	this->SetPhase();
	this->SetFrame();
	this->GetPosAndVel(worldTime, mTargetPositions, mTargetVelocities);
	this->GetPosAndVel(worldTime, mReferencePositions, mReferenceVelocities);

	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true,false,false);

	mAction.setZero();
	mAction_prev.setZero();
	mDesiredTorque.setZero();

	for(int i=0; i<mFemurSignals.at(0).size(); i++){
		mFemurSignals.at(0).at(i) = 0.0;
		mFemurSignals.at(1).at(i) = 0.0;
	}//Initialze L,R at the same time

	mRootTrajectory.clear();
	for(int i=0; i<32; i++)
		mRootTrajectory.push_back(Eigen::Vector3d::Zero());

	mHeadTrajectory.clear();
	for(int i=0; i<32; i++)
		mHeadTrajectory.push_back(Eigen::Vector3d::Zero());

	mComHistory.clear();
	for(int i=0; i<200; i++)
		mComHistory.push_back(Eigen::Vector4d::Zero());

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

	this->SetMeasure(isRender);

	if(mStepCnt == mNumSteps-1)
		mAdaptiveTime += mTemporalDisplacement;

	mStepCnt++;
	mStepCntTotal++;

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
		this->SetMuscleTuple();

	if(mLowerBody){
		mDesiredTorque.head<30>().setZero();
		mSkeleton->setForces(mDesiredTorque); // upper body torque control
	}

	this->SetMeasure(isRender);

	if(mStepCnt == mNumSteps-1)
		mAdaptiveTime += mTemporalDisplacement;

	mStepCnt++;
	mStepCntTotal++;
}

void
Character::
SetMuscleTuple()
{
	int m = mMuscles.size();
	Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(mDof,m);
	Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(mDof);

	for(int i=0; i<m; i++)
	{
		auto muscle = mMuscles[i];
		Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
		auto Ap = muscle->GetForceJacobianAndPassive();

		JtA.block(0,i,mDof,1) = Jt*Ap.first;
		Jtp += Jt*Ap.second;
	}

	mCurrentMuscleTuple.JtA = GetMuscleTorques();
	mCurrentMuscleTuple.L =JtA.block(mRootJointDof,0,mDof-mRootJointDof,m);
	mCurrentMuscleTuple.b =Jtp.segment(mRootJointDof,mDof-mRootJointDof);

	if(mLowerBody)
	{
		Eigen::VectorXd lbTorque = mDesiredTorque; // lower body related torque
		lbTorque.tail<26>().setZero();
		mCurrentMuscleTuple.tau_des = lbTorque.tail(mNumActiveDof);
	}
	else
	{
		mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mNumActiveDof);
	}

	mMuscleTuples.push_back(mCurrentMuscleTuple);
}

void
Character::
SetComHistory()
{
	Eigen::Vector4d curCom;
	curCom.segment(0,3) = mSkeleton->getCOM();
	curCom(3) = mAdaptiveTime;
	mComHistory.pop_back();
	mComHistory.push_front(curCom);
}

void
Character::
SetMeasure(bool isRender)
{
	if(mStepCnt == mNumSteps-1){
		this->SetTrajectory();
		this->SetCurVelocity();
	}
	this->SetComHistory();

	double phase = mPhase;
	double frame = mFrame;
	if(mAdaptiveMotion)
	{
		phase = mAdaptivePhase;
		frame = mAdaptiveFrame;
	}

	if(mUseMuscle)
		mMetabolicEnergy->Set(this->GetMuscles(), mCurVel3d, phase, frame);
	else
		mJointDatas->SetTorques(mDesiredTorque, phase, frame);

	if(isRender)
	{
		mJointDatas->SetAngles(phase);
		mContacts->Set();
		// this->SetCoT();
	}
}

void
Character::
SetTrajectory()
{
	mRootTrajectory.pop_back();
	mRootTrajectory.push_front(mSkeleton->getCOM());

	mHeadTrajectory.pop_back();
	mHeadTrajectory.push_front(mEndEffectors[2]->getCOM());
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

	mCurCoT *= (mStepCntTotal-1);
	mCurCoT += vel_tor / (m * g * v);
	mCurCoT /= mStepCntTotal;
}

void
Character::
SetCurVelocity()
{
	double time_step = 1.0/(double)(mControlHz);

	double x_diff = (mRootTrajectory[0][0] - mRootTrajectory[mRootTrajectory.size()-1][0]);
	double z_diff = (mRootTrajectory[0][2]-mRootTrajectory[mRootTrajectory.size()-1][2]);

	x_diff /= time_step*mRootTrajectory.size();
	z_diff /= time_step*mRootTrajectory.size();

	mCurVel = std::sqrt(x_diff*x_diff + z_diff*z_diff);

	x_diff = (mHeadTrajectory[0][0] - mHeadTrajectory[mHeadTrajectory.size()-1][0]);
	z_diff = (mHeadTrajectory[0][2]-mHeadTrajectory[mHeadTrajectory.size()-1][2]);

	x_diff /= time_step*mHeadTrajectory.size();
	z_diff /= time_step*mHeadTrajectory.size();

	mCurHeadVel = std::sqrt(x_diff*x_diff + z_diff*z_diff);

		// mCurVel = (mRootTrajectory[0] - mRootTrajectory[mRootTrajectory.size()-1]).norm();

	mCurVel3d = mSkeleton->getCOMLinearVelocity();
	// mCurVel = mCurVel3d.norm();
}

double
Character::
GetCurTime()
{
	double time = 0.0;
	if(mAdaptiveMotion)
		time = mAdaptiveTime;
	else
		time = mWorld->getTime();

	return time;
}

double
Character::
GetControlTimeStep()
{
	return (double)mNumSteps*mWorld->getTimeStep();
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
		Eigen::VectorXd state_device = mDevice->GetState();
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
	Eigen::VectorXd cur_p = mSkeleton->getPositions();
	Eigen::VectorXd cur_v = mSkeleton->getVelocities();

	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	Eigen::Vector3d root_com = root->getCOM();
	Eigen::Vector3d root_com_vel = root->getCOMLinearVelocity();
	double w = root_com[0];
	double h = root_com[1];

	Eigen::VectorXd p,v;
	p.resize((mNumBodyNodes-1)*3);
	v.resize((mNumBodyNodes)*3);

	for(int i=1; i<mNumBodyNodes; i++)
	{
		dart::dynamics::BodyNode* cur_body = mSkeleton->getBodyNode(i);
		p.segment<3>(3*(i-1)) = cur_body->getCOM() - root->getCOM();
		v.segment<3>(3*(i-1)) = cur_body->getCOMLinearVelocity() - root->getCOMLinearVelocity();
	}
	v.tail<3>() = root->getCOMLinearVelocity();

	p *= 0.8;
	v *= 0.2;

	double curTime = this->GetCurTime();
	double nextTime = curTime + this->GetControlTimeStep();

	Eigen::VectorXd cur_ref_pos, cur_ref_vel;
	Eigen::VectorXd next_ref_pos, next_ref_vel;
	this->GetPosAndVel(curTime, cur_ref_pos, cur_ref_vel);
	this->GetPosAndVel(nextTime, next_ref_pos, next_ref_vel);

	Eigen::VectorXd delta_pos = next_ref_pos - cur_ref_pos;

	Eigen::VectorXd p_next, p_cur;
	p_cur = mTargetPositions - cur_p;
	p_next = delta_pos;

	p_cur *= 0.8;
	p_next *= 0.8;

	std::pair<double, double> phase;
	if(mAdaptiveMotion)
		phase = this->GetAdaptivePhases();
	else
		phase = this->GetPhases();

	double cur_time = mWorld->getTime() * 0.1;
	if(cur_time > 0.20)
		cur_time = 0.21;

	Eigen::VectorXd action;
	if(mAdaptiveLowerBody)
		action = mAction.segment(mNumActiveDof,24);
	else
		action = mAction.segment(mNumActiveDof,56);

	Eigen::VectorXd state;
	state.resize(2+p.rows()+v.rows()+2+p_cur.rows()+p_next.rows()+1+action.size());
	state << h,w,p,v,phase.first,phase.second,p_cur,p_next,cur_time,action;

	return state;
}

void
Character::
GetNextPosition(Eigen::VectorXd cur, Eigen::VectorXd delta, Eigen::VectorXd& next)
{
	Eigen::AngleAxisd cur_root_ori= Eigen::AngleAxisd(cur.segment<3>(0).norm(), cur.segment<3>(0).normalized());
	delta.segment<3>(3) = cur_root_ori * delta.segment<3>(3);
	next.segment<3>(3) = cur.segment<3>(3) + delta.segment<3>(3);

	for(int i=1; i<mNumBodyNodes; i++)
	{
		int idx = mSkeleton->getBodyNode(i)->getParentJoint()->getIndexInSkeleton(0);
		Eigen::AngleAxisd target_diff_aa = Eigen::AngleAxisd(delta.segment<3>(idx).norm(), delta.segment<3>(idx).normalized());
		Eigen::AngleAxisd cur_aa = Eigen::AngleAxisd(cur.segment<3>(idx).norm(), cur.segment<3>(idx).normalized());
		target_diff_aa = cur_aa * target_diff_aa;
		next.segment<3>(idx) = target_diff_aa.angle() * target_diff_aa.axis();
	}
}

Eigen::VectorXd
Character::
GetEndEffectorStatePosAndVel(const Eigen::VectorXd pos, const Eigen::VectorXd vel) {
	Eigen::VectorXd ret;
	dart::dynamics::BodyNode* root = mSkeleton->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	int num_ee = mEndEffectors.size();
	Eigen::VectorXd p_save = mSkeleton->getPositions();
	Eigen::VectorXd v_save = mSkeleton->getVelocities();

	mSkeleton->setPositions(pos);
	mSkeleton->setVelocities(vel);
	mSkeleton->computeForwardKinematics(true, true, false);

	ret.resize((num_ee)*9+12);
	for(int i=0;i<num_ee;i++)
	{
		Eigen::Isometry3d transform = cur_root_inv * mEndEffectors[i]->getWorldTransform();
		Eigen::Vector3d rot = Utils::QuaternionToAxisAngle(Eigen::Quaterniond(transform.linear()));
		ret.segment<6>(6*i) << rot, transform.translation();
	}

	for(int i=0;i<num_ee;i++)
	{
	    int idx = mEndEffectors[i]->getParentJoint()->getIndexInSkeleton(0);
	    ret.segment<3>(6*num_ee + 3*i) << vel.segment<3>(idx);
	}

	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * root->getWorldTransform();
	Eigen::Vector3d rot = Utils::QuaternionToAxisAngle(Eigen::Quaterniond(transform.linear()));
	Eigen::Vector3d root_angular_vel_relative = cur_root_inv.linear() * root->getAngularVelocity();
	Eigen::Vector3d root_linear_vel_relative = cur_root_inv.linear() * root->getCOMLinearVelocity();

//	ret.tail<9>() << rot, root_angular_vel_relative, root_linear_vel_relative;
	ret.tail<12>() << rot, transform.translation(), root_angular_vel_relative, root_linear_vel_relative;

	// restore
	mSkeleton->setPositions(p_save);
	mSkeleton->setVelocities(v_save);
	mSkeleton->computeForwardKinematics(true, true, false);

	return ret;
}

std::pair<double,double>
Character::
GetReward()
{
	std::pair<double,double> rewards_character = this->GetReward_Character();

	double r_continuous = 0.0;
	double r_spike = 0.0;

	r_continuous += rewards_character.first;
	r_spike += rewards_character.second;

	double r_total = r_continuous + r_spike;
	mReward["reward"] = r_total;
	mCurReward = r_total;

	this->SetRewards();
	return std::make_pair(r_continuous, r_spike);
}

std::pair<double,double>
Character::
GetReward_Character()
{
	std::pair<double,double> r_imit = GetReward_Character_Imitation();
	std::pair<double,double> r_effi =
		GetReward_Character_Efficiency();

	mReward["imit"] = r_imit.first + r_imit.second;
 	mReward["effi"] = r_effi.first + r_effi.second;

	double ratio_imit = 1.0;
	double ratio_effi = 1.0;

	r_imit.first *= ratio_imit;
	r_imit.second *= ratio_imit;
	r_effi.first *= ratio_effi;
	r_effi.second *= ratio_effi;

	double r_continuous = r_imit.first + r_effi.first;
	double r_spike = r_imit.second + r_effi.second;

	return std::make_pair(r_continuous,r_spike);
}

std::pair<double,double>
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
	mSkeleton->setPositions(mTargetPositions);
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
	Eigen::VectorXd q_diff = mSkeleton->getPositionDifferences(cur_q, mTargetPositions);

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

	return std::make_pair(r_total,0.0);
}

std::pair<double,double>
Character::
GetReward_Character_Efficiency()
{
	if(mWorld->getTime() < mTimeOffset)
		return std::make_pair(0.0,0.0);

	double r_EnergyMin = 1.0;
	r_EnergyMin = this->GetReward_Energy();

	double r_Pose = 1.0;
	r_Pose = this->GetReward_Pose();

	double r_ActionReg = 1.0;
	r_ActionReg = this->GetReward_ActionReg();

	double r_Vel = 1.0;
	r_Vel = this->GetReward_Vel();

	double r_Width = 1.0;
	r_Width = this->GetReward_Width();

	// double r_ContactForce = 1.0;
	// double r_ContactForce = mContacts->GetReward();

	// double r_Height = 1.0;
	// r_Height = this->GetReward_Height();

	mReward["min"] = r_Vel;
	mReward["reg"] = r_Pose;
	mReward["com"] = r_ActionReg;

	double r_continuous = 0.10 * r_Width + 0.30 * r_Pose + 0.30 * r_Vel + 0.30 * r_ActionReg;
	double r_spike = 0.0 * r_EnergyMin;

	return std::make_pair(r_continuous,r_spike);
}

double
Character::
GetReward_Energy()
{
	double reward = 0;
	if(mUseMuscle)
		reward = mMetabolicEnergy->GetReward();
	else
		reward = mJointDatas->GetReward();

	return reward;
}

double
Character::
GetReward_Pose()
{
	Eigen::VectorXd pose_cur, pose_ref;

	int num_ee = mEndEffectors.size();
	Eigen::VectorXd p_save = mSkeleton->getPositions();
	Eigen::VectorXd v_save = mSkeleton->getVelocities();

	Eigen::VectorXd p_reward;
	Eigen::VectorXd v_reward;
	this->GetPosAndVel(mAdaptiveTime, p_reward, v_reward);

	mSkeleton->setPositions(p_reward);
	mSkeleton->computeForwardKinematics(true, true, false);

	dart::dynamics::BodyNode* root_ref = mSkeleton->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root_ref->getWorldTransform().inverse();

	Eigen::VectorXd root_rot_ref = Utils::QuaternionToAxisAngle(Eigen::Quaterniond((root_ref->getWorldTransform()).linear()));
	Eigen::VectorXd head_ref = Eigen::VectorXd::Zero(4);

	// pose_ref.resize((num_ee)*3);
	pose_ref.resize((num_ee)*6);
	for(int i=0;i<num_ee;i++)
	{
		Eigen::Isometry3d transform = cur_root_inv * mEndEffectors[i]->getWorldTransform();
		Eigen::Vector3d rot = Utils::QuaternionToAxisAngle(Eigen::Quaterniond(transform.linear()));

		// pose_ref.segment<3>(3*i) << rot;
		// pose_ref.segment<3>(3*i) << transform.translation();
		pose_ref.segment<6>(6*i) << rot, transform.translation();
		if(i==2)
			head_ref.segment<3>(0) << rot;
	}

	head_ref[3] = mSkeleton->getBodyNode("Head")->getCOM()[1];

	// restore
	mSkeleton->setPositions(mTargetPositions);
	mSkeleton->setVelocities(mTargetVelocities);
	mSkeleton->computeForwardKinematics(true, true, false);

	dart::dynamics::BodyNode* root_cur = mSkeleton->getRootBodyNode();
	cur_root_inv = root_cur->getWorldTransform().inverse();

	Eigen::VectorXd root_rot_cur = Utils::QuaternionToAxisAngle(Eigen::Quaterniond((root_cur->getWorldTransform()).linear()));
	Eigen::VectorXd head_cur = Eigen::VectorXd::Zero(4);

	// pose_cur.resize((num_ee)*3);
	pose_cur.resize((num_ee)*6);
	for(int i=0;i<num_ee;i++)
	{
		Eigen::Isometry3d transform = cur_root_inv * mEndEffectors[i]->getWorldTransform();
		Eigen::Vector3d rot = Utils::QuaternionToAxisAngle(Eigen::Quaterniond(transform.linear()));
		// pose_cur.segment<3>(3*i) << rot;
		// pose_cur.segment<3>(3*i) << transform.translation();
		pose_cur.segment<6>(6*i) << rot, transform.translation();
		if(i==2)
			head_cur.segment<3>(0) << rot;
	}

	head_cur[3] = mSkeleton->getBodyNode("Head")->getCOM()[1];

	double err_scale = 1.0;

	double pose_scale = 8.0;
	double head_scale = 4.0;
	double root_scale = 2.0;

	double pose_err = (pose_cur-pose_ref).norm();
	double head_err = (head_cur-head_ref).norm();
	double root_err = (root_rot_cur-root_rot_ref).norm();

	double pose_reward = exp(-1.0 * pose_scale * pose_err);
	double head_reward = exp(-1.0 * head_scale * head_err);
	double root_reward = exp(-1.0 * root_scale * root_err);

	// std::cout << "pose : " << pose_err*pose_scale << std::endl;
	// std::cout << "head : " << head_err*head_scale << std::endl;
	// std::cout << "r pose : " << pose_reward << std::endl;
	// std::cout << "r head : " << head_reward << std::endl;

	// double reward = 0.4 * pose_reward + 0.4 * head_reward + 0.2 * root_reward;
	double reward = 0.5 * pose_reward + 0.5 * head_reward;

	mSkeleton->setPositions(p_save);
	mSkeleton->setVelocities(v_save);
	mSkeleton->computeForwardKinematics(true, true, false);

	return reward;
}

double
Character::
GetReward_Height()
{
	double err_scale = 1.0;
	double height_scale = 1.0;
	double height_err = 0.0;

	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	Eigen::Vector3d root_com = root->getCOM();
	double h = root_com[1];
	double h_tar = mTargetPositions[5];
	double h_ref = mReferencePositions[5];

	// if(h > 0.98)
	// 	height_err = fabs(h-0.98);
	// else if(h < 0.94)
	// 	height_err = fabs(h-0.94);

	height_err = fabs(h_tar-h_ref);

	double reward = 0.0;
	reward = exp(-1.0 * height_scale * height_err);

	return reward;
}

double
Character::
GetReward_Width()
{
	double err_scale = 1.0;
	double width_scale = 1.0;
	double width_err = 0.0;

	Eigen::VectorXd reference_pos;
	Eigen::VectorXd reference_vel;
	this->GetPosAndVel(mAdaptiveTime, reference_pos, reference_vel);

	dart::dynamics::BodyNode* root = mSkeleton->getBodyNode(0);
	Eigen::Vector3d root_com = root->getCOM();
	double w = root_com[0];
	double w_tar = mTargetPositions[3];
	double w_ref = reference_pos[3];

	// width_err = fabs(w-w_ref);
	width_err = fabs(w_tar-w_ref);

	double reward = 0.0;
	reward = exp(-1.0 * width_scale * width_err);

	// std::cout << "cur vel : " << mCurVel << std::endl;
	// std::cout << "vel err : " << vel_err << std::endl;
	// std::cout << "reward : " << reward << std::endl;
	// std::cout << std::endl;

	return reward;
}

double
Character::
GetReward_Vel()
{
	double err_scale = 1.0;
	double vel_scale = 2.0;
	double vel_err = 0.0;

	int last_idx = mComHistory.size()-1;
	Eigen::Vector4d past = mComHistory[last_idx];
	Eigen::Vector4d cur = mComHistory[0];

	double diff_x = (cur[0]-past[0]);
	double diff_z = (cur[2]-past[2]);
	double diff = std::sqrt(diff_x*diff_x + diff_z*diff_z);
	double vel = diff/(cur[3]-past[3]);

	vel_err = fabs(vel-1.5);

	// vel_err = fabs(mCurVel - 1.5);
	// vel_err += fabs(mCurHeadVel - 1.5);

	double reward = 0.0;
	reward = exp(-1.0 * vel_scale * vel_err);

	// std::cout << "cur vel : " << mCurVel << std::endl;
	// std::cout << "vel err : " << vel_err << std::endl;
	// std::cout << "reward : " << reward << std::endl;
	// std::cout << std::endl;

	return reward;
}

double
Character::
GetReward_ActionReg()
{
	Eigen::VectorXd prev = mAction_prev.segment(mNumActiveDof,56);
	Eigen::VectorXd cur = mAction.segment(mNumActiveDof,56);

	double actionDiff = (prev-cur).norm();

	double err_scale = 1.0;
	double actionReg_scale = 10.0;
	double actionReg_err = 0.0;

	actionReg_err = actionDiff;

	double reward = 0.0;
	reward = exp(-err_scale * actionReg_scale * actionReg_err);

	return reward;
}

void
Character::
SetPhases()
{
	double rad = 2*M_PI*mPhase;
	double cos = std::cos(rad);
	double sin = std::sin(rad);

	mPhases = std::pair<double, double>(cos, sin);

	double rad2 = 2*M_PI*mAdaptivePhase;
	double cos2 = std::cos(rad2);
	double sin2 = std::sin(rad2);

	mAdaptivePhases = std::pair<double, double>(cos2, sin2);
}

void
Character::
SetPhase()
{
	double worldTime = mWorld->getTime();
	double cycleTime = mBVH->GetMaxTime();

	mPhasePrev = mPhase;
	double phase = worldTime;
	int cycleCount = (int)(worldTime/cycleTime);
	if(mBVH->IsCyclic())
		phase = (worldTime - cycleCount*cycleTime)/cycleTime;
	if(phase < 0)
		phase += (cycleTime)/cycleTime;

	mPhase = phase;

	mAdaptivePhasePrev = mAdaptivePhase;
	double adaptivePhase = mAdaptiveTime;
	int adaptiveCycleCount = (int)(mAdaptiveTime/cycleTime);
	if(mBVH->IsCyclic())
		adaptivePhase = (mAdaptiveTime - adaptiveCycleCount*cycleTime)/cycleTime;
	if(adaptivePhase < 0)
		adaptivePhase += (cycleTime)/cycleTime;

	mAdaptivePhase = adaptivePhase;

	this->SetPhases();
}

void
Character::
SetFrame()
{
	double t = mWorld->getTime();
	int frame, frameNext;
	double fraction;
	double dt = 1.0/(double)mControlHz;
	this->GetFrameNum(t,dt,frame,frameNext,fraction);

	mFramePrev = mFrame;
	mFrame = frame + fraction;

	if(mAdaptiveMotion)
	{
		t = mAdaptiveTime;
		this->GetFrameNum(t,dt,frame,frameNext,fraction);
		mAdaptiveFramePrev = mAdaptiveFrame;
		mAdaptiveFrame = frame + fraction;
	}
}

void
Character::
SetActionAdaptiveMotion(const Eigen::VectorXd& a)
{
	int pd_dof = mNumActiveDof;
	int l_dof = mAdaptiveLowerDof;
	int u_dof = mAdaptiveUpperDof;
	int sp_dof = mNumAdaptiveSpatialDof;
	int ad_dof = mNumAdaptiveDof;

	double pd_scale = 0.1;
	double root_ori_scale = 0.002;
	double root_pos_scale = 0.004;
	double lower_scale = 0.004; // adaptive lower body
	double upper_scale = 0.004; // adaptive upper body
	double temporal_scale = 0.2; // adaptive temporal displacement

	mAction.segment(0,pd_dof) = a.segment(0,pd_dof) * pd_scale;
	mAction.segment(pd_dof,ad_dof) = Eigen::VectorXd::Zero(ad_dof);

	double t = mWorld->getTime();
	if(t >= mTimeOffset){
		mAction.segment(pd_dof,3) = a.segment(pd_dof,3) * root_ori_scale;
		mAction.segment(pd_dof+3,3) = a.segment(pd_dof+3,3) * root_pos_scale;
		mAction.segment(pd_dof+6,l_dof-6) = a.segment(pd_dof+6,l_dof-6) * lower_scale;
		mAction.segment(pd_dof+l_dof,(sp_dof-l_dof)) = a.segment(pd_dof+l_dof,(sp_dof-l_dof)) * upper_scale;
		mAction[pd_dof+sp_dof] = a[pd_dof+sp_dof] * temporal_scale;

		for(int i=pd_dof; i<mAction.size(); i++){
			if(i < pd_dof+3)
				mAction[i] = Utils::Clamp(mAction[i], -0.01, 0.01);
			else if(i >= pd_dof+3 && i < pd_dof+6)
				mAction[i] = Utils::Clamp(mAction[i], -0.02, 0.02);
			else if(i >= pd_dof+6 && i < pd_dof+l_dof)
				mAction[i] = Utils::Clamp(mAction[i], -0.02, 0.02);
			else if(i >= pd_dof+l_dof && i < pd_dof+sp_dof)
				mAction[i] = Utils::Clamp(mAction[i], -0.02, 0.02);
			else
				mAction[i] = Utils::Clamp(mAction[i], -1.0, 1.0);
		}
	}

	double timeStep = (double)mNumSteps*mWorld->getTimeStep();
	mTemporalDisplacement = timeStep * exp(mAction[pd_dof+sp_dof]);

	double cur_time = mAdaptiveTime;
	double next_time = mAdaptiveTime + mTemporalDisplacement;
	Eigen::VectorXd cur_pos, cur_vel;
	Eigen::VectorXd next_pos, next_vel;
	this->GetPosAndVel(cur_time, cur_pos, cur_vel);
	this->GetPosAndVel(next_time, next_pos, next_vel);

	Eigen::VectorXd delta_pos = next_pos - cur_pos;
	delta_pos.segment(0, sp_dof) += mAction.segment(pd_dof, sp_dof);

	mTargetPositions += delta_pos;
	mTargetVelocities = next_vel;

	this->GetPosAndVel(mWorld->getTime()+timeStep, mReferencePositions, mReferenceVelocities);
}

void
Character::
SetActionImitationLearning(const Eigen::VectorXd& a)
{
	int pd_dof = mNumActiveDof;
	double pd_scale = 0.1;
	mAction.segment(0,pd_dof) = a.segment(0,pd_dof) * pd_scale;

	double t = mWorld->getTime();
	double timeStep = (double)mNumSteps*mWorld->getTimeStep();
	this->GetPosAndVel(t+timeStep, mTargetPositions, mTargetVelocities);
}

void
Character::
SetAction(const Eigen::VectorXd& a)
{
	mAction_prev = mAction;

	if(mAdaptiveMotion)
		this->SetActionAdaptiveMotion(a);
	else
		this->SetActionImitationLearning(a);

	mStepCnt = 0;
}

void
Character::
SetDesiredTorques()
{
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.tail(mTargetPositions.rows() - mRootJointDof) += mAction.segment(0,mNumActiveDof);
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
		return tmp.tail(mNumActiveDof);
	}
	else
		return mDesiredTorque.tail(mNumActiveDof);
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
GetFrameNum(double t, double dt, int& frame, int& frameNext, double& frameFraction)
{
	double frameTime = t;

	if(mBVH->IsCyclic()){
		double cycleTime = mBVH->GetMaxTime();
		int cycleCount = (int)(t/cycleTime);

		frameTime = t - cycleCount*cycleTime;
		if(frameTime < 0)
			frameTime += cycleTime;
	}

	frame = (int)(frameTime/dt);
	frameNext = frame + 1;

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

	frameFraction = (frameTime - frame*dt)/dt;
}

void
Character::
GetPosAndVel(double t, Eigen::VectorXd& pos, Eigen::VectorXd& vel)
{
	int frame, frameNext;
	double frameFraction;
	double dt = 1.0/(double)mControlHz;

	this->GetFrameNum(t, dt, frame, frameNext, frameFraction);
	this->GetPos(t,dt,frame,frameNext,frameFraction,pos);
	this->GetVel(t,dt,frame,frameNext,frameFraction,vel);
}

void
Character::
GetPos(double t, double dt, int frame, int frameNext, double frameFraction, Eigen::VectorXd& pos)
{
	Eigen::VectorXd frameData, frameDataNext;
	frameData = mBVH->GetMotion(frame);
	frameDataNext = mBVH->GetMotion(frameNext);

	// Eigen::VectorXd p = Utils::GetPoseSlerp(mSkeleton, frameFraction, frameData, frameDataNext);
	pos = (1-frameFraction)*frameData + (frameFraction)* frameDataNext;

	if(mBVH->IsCyclic()) {
		double cycleTime = mBVH->GetMaxTime();
		int cycleCount = (int)(t/cycleTime);

		Eigen::Vector3d cycleOffset = mBVH->GetCycleOffset();
		cycleOffset[1] = 0.0;
		pos.segment(3,3) += cycleCount*cycleOffset;
	}

	Eigen::Isometry3d T_current_reference = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
	T_current_reference = mBVH->GetT0().inverse()*T_current_reference;
	Eigen::Isometry3d T_head_reference = mTc*T_current_reference;
	Eigen::Vector6d p_head_reference = dart::dynamics::FreeJoint::convertToPositions(T_head_reference);
	pos.head<6>() = p_head_reference;
}

void
Character::
GetVel(double t, double dt, int frame, int frameNext, double frameFraction, Eigen::VectorXd& vel)
{
	Eigen::VectorXd frameVel, frameNextVel;
	frameVel = mBVH->GetMotionVel(frame);
	frameNextVel = mBVH->GetMotionVel(frameNext);

	vel = (1-frameFraction)*frameVel + (frameFraction)*frameNextVel;
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
	mWeldJointHip = std::make_shared<dart::constraint::WeldJointConstraint>(
		mSkeleton->getBodyNode(0), mDevice->GetSkeleton()->getBodyNode(0)
		);

	mWeldJointLeftLeg = std::make_shared<dart::constraint::WeldJointConstraint>(
		mSkeleton->getBodyNode("FemurL"), mDevice->GetSkeleton()->getBodyNode("RodLeft")
		);

	mWeldJointRightLeg = std::make_shared<dart::constraint::WeldJointConstraint>(
		mSkeleton->getBodyNode("FemurR"), mDevice->GetSkeleton()->getBodyNode("RodRight")
		);

	mWorld->getConstraintSolver()->addConstraint(mWeldJointHip);
	mWorld->getConstraintSolver()->addConstraint(mWeldJointLeftLeg);
	mWorld->getConstraintSolver()->addConstraint(mWeldJointRightLeg);
}

void
Character::
RemoveConstraints()
{
	mWorld->getConstraintSolver()->removeConstraint(mWeldJointHip);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJointLeftLeg);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJointRightLeg);
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

	mWorld->getConstraintSolver()->addConstraint(mWeldJointHip);
	mWorld->getConstraintSolver()->addConstraint(mWeldJointLeftLeg);
	mWorld->getConstraintSolver()->addConstraint(mWeldJointRightLeg);
	mWorld->addSkeleton(mDevice->GetSkeleton());
}

void
Character::
SetDevice_Off()
{
	mWorld->getConstraintSolver()->removeConstraint(mWeldJointHip);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJointLeftLeg);
	mWorld->getConstraintSolver()->removeConstraint(mWeldJointRightLeg);
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
		// for(int i=0; i<mMusclesFemur.size(); i++)
		// 	mMusclesFemur.at(i)->SetF0Ratio(mForceRatio);
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

}

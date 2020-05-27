#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "Device.h"
#include "dart/collision/bullet/bullet.hpp"
using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;

Environment::
Environment()
	:mControlHz(30),mSimulationHz(900),mWorld(std::make_shared<World>()),mUseMuscle(true)
{

}

void
Environment::
Initialize(const std::string& meta_file, bool load_obj)
{

	std::ifstream ifs(meta_file);
	if(!(ifs.is_open()))
	{
		std::cout<<"Can't read file "<<meta_file<<std::endl;
		return;
	}

	std::string str, index;
	std::stringstream ss;
	MASS::Character* character = new MASS::Character();
	while(!ifs.eof())
	{
		ss.clear();
		str.clear();
		index.clear();

		std::getline(ifs, str);
		ss.str(str);
		ss>>index;
		if(!index.compare("use_muscle"))
		{
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseMuscle(true);
			else
				this->SetUseMuscle(false);
		}
		else if(!index.compare("use_device"))
		{
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseDevice(true);
			else
				this->SetUseDevice(false);
		}
		else if(!index.compare("con_hz")){
			int hz;
			ss>>hz;
			this->SetControlHz(hz);
		}
		else if(!index.compare("sim_hz")){
			int hz;
			ss>>hz;
			this->SetSimulationHz(hz);
		}
		else if(!index.compare("skel_file")){
			std::string str2;
			ss>>str2;
			character->LoadSkeleton(std::string(MASS_ROOT_DIR)+str2,load_obj);
		}
		else if(!index.compare("muscle_file")){
			std::string str2;
			ss>>str2;
			if(this->GetUseMuscle())
				character->LoadMuscles(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("device_file")){
			std::string str2;
			ss>>str2;
			if(this->GetUseDevice())
				character->LoadDevice(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("bvh_file")){
			std::string str2,str3;

			ss>>str2>>str3;
			bool cyclic = false;
			if(!str3.compare("true"))
				cyclic = true;
			character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
		}
		else if(!index.compare("reward_param")){
			double a,b,c,d;
			ss>>a>>b>>c>>d;
			character->SetRewardParameters(a,b,c,d);
		}
	}
	ifs.close();

	double kp = 300.0;
	character->SetPDParameters(kp,sqrt(2*kp));
	this->SetCharacter(character);
	this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

	this->Initialize();
}

void
Environment::
Initialize()
{
	mCharacter->Initialize();
	mWorld->addSkeleton(mCharacter->GetSkeleton());

	if(mUseMuscle)
		mCharacter->Initialize_Muscles();

  	if(mUseDevice)
  		mCharacter->Initialize_Device(mWorld);

	mWorld->addSkeleton(mGround);
	mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));
	mWorld->setTimeStep(1.0/mSimulationHz);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());

	Reset(false);
}

void
Environment::
Reset(bool RSI)
{
	double t = 0.0;
	if(RSI)
		t = dart::math::random(0.0, mCharacter->GetBVH()->GetMaxTime()*0.9);

	mWorld->reset();
	mWorld->setTime(t);

	mCharacter->Reset(mWorld->getTime(), mControlHz);

	mAction.setZero();
}

void
Environment::
Step()
{
	if(mUseMuscle)
		mCharacter->Step_Muscles(mSimCount, mRandomSampleIndex);
	else
		mCharacter->Step();

	if(mUseDevice)
		mCharacter->Step_Device();

	mWorld->step();

	mSimCount++;
}

void
Environment::
StepDeviceOnly()
{
	Eigen::VectorXd pos_ = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd vel_ = mCharacter->GetSkeleton()->getVelocities();

	Eigen::VectorXd pos_d = mCharacter->mDevice->GetSkeleton()->getPositions();
	Eigen::VectorXd vel_d = mCharacter->mDevice->GetSkeleton()->getVelocities();

	if(mUseMuscle)
		mCharacter->Step_Muscles(mSimCount, mRandomSampleIndex);
	else
		mCharacter->Step();

	if(mUseDevice)
		mCharacter->Step_Device(Eigen::VectorXd::Zero(12));

	mWorld->step();

	if(mUseDevice){
		double r = mCharacter->GetReward_Character();
		if(mSimCount < mSimulationHz/mControlHz)
		{
			r_only += r;
			mCharacter->r_cur = r_only;
		}

		mCharacter->GetSkeleton()->setPositions(pos_);
		mCharacter->GetSkeleton()->setVelocities(vel_);
		mCharacter->GetSkeleton()->computeForwardKinematics(true, false, false);

		mCharacter->mDevice->GetSkeleton()->setPositions(pos_d);
		mCharacter->mDevice->GetSkeleton()->setVelocities(vel_d);
		mCharacter->mDevice->GetSkeleton()->computeForwardKinematics(true, false, false);

		if(!mUseMuscle){
			this->StepBack();
			// mCharacter->StepBack();

			mCharacter->Step();
		}
		else
		{
			// muscle step back not implemented
		}


		mCharacter->Step_Device();

		mWorld->step();

		double r_ = mCharacter->GetReward_Character();
		if(mSimCount < mSimulationHz/mControlHz)
		{
			r_d += r_;
			mCharacter->r_device = r_d;
		}
	}

	mSimCount++;
}

void
Environment::
StepBack()
{
	mWorld->setTime(mWorld->getTime()-mWorld->getTimeStep());
}

bool
Environment::
IsEndOfEpisode()
{
	bool isTerminal = false;

	Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();

	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];

	if(root_y<1.3)
		isTerminal =true;
	else if (dart::math::isNan(p) || dart::math::isNan(v))
		isTerminal =true;
	else if(mWorld->getTime()>10.0)
		isTerminal =true;

	// if(mUseDevice)
	// {

	// }

	return isTerminal;
}

void
Environment::
SetAction(const Eigen::VectorXd& a)
{
	double action_scale = 0.1;
	mAction = a*action_scale;
	mCharacter->SetAction(mAction);

	double t = mWorld->getTime();
	mCharacter->SetTargetPosAndVel(t, mControlHz);

	mSimCount = 0;
	r_only = 0.0;
	mCharacter->r_cur = 0.0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
}

void
Environment::
SetAction_Device(const Eigen::VectorXd& a)
{
	double action_scale = 1.0;
	mAction_Device = a*action_scale;
	mCharacter->SetAction_Device(mAction_Device);

	double t = mWorld->getTime();
	mCharacter->SetTargetPosAndVel(t, mControlHz);

	mSimCount = 0;
	r_d = 0.0;
	mCharacter->r_device = 0.0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
}

void
Environment::
SetActivationLevels(const Eigen::VectorXd& a)
{
	mCharacter->SetActivationLevels(a);
}

Eigen::VectorXd
Environment::
GetState()
{
	return mCharacter->GetState(mWorld->getTime());
}

Eigen::VectorXd
Environment::
GetState_Device()
{
	return mCharacter->GetState_Device(mWorld->getTime());
}

int
Environment::
GetNumState()
{
	return mCharacter->GetNumState();
}

int
Environment::
GetNumState_Device()
{
	return mCharacter->GetDevice()->GetNumState();
}

int
Environment::
GetNumAction()
{
	return mCharacter->GetNumActiveDof();
}

int
Environment::
GetNumAction_Device()
{
	return mCharacter->GetDevice()->GetNumAction();
}

double
Environment::
GetReward()
{
	return mCharacter->GetReward();
}

std::map<std::string,double>
Environment::
GetRewardSep()
{
	return mCharacter->GetRewardSep();
}

int
Environment::
GetNumTotalRelatedDofs()
{
	return (mCharacter->GetCurrentMuscleTuple()).JtA.rows();
}

std::vector<MuscleTuple>&
Environment::
GetMuscleTuples()
{
	return mCharacter->GetMuscleTuples();
}

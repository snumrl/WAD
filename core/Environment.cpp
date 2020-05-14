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
  	{
  		mCharacter->Initialize_Device();
		mWorld->addSkeleton(mCharacter->GetDevice()->GetSkeleton());
  	}

	mWorld->addSkeleton(mGround);
	mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));
	mWorld->setTimeStep(1.0/mSimulationHz);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());

	// device constraint
	if(mUseDevice)
	{
		mWorld->getConstraintSolver()->addConstraint(mCharacter->mWeldJoint_Hip);
		mWorld->getConstraintSolver()->addConstraint(mCharacter->mWeldJoint_LeftLeg);
		mWorld->getConstraintSolver()->addConstraint(mCharacter->mWeldJoint_RightLeg);
	}

	mAction = Eigen::VectorXd::Zero(GetNumAction());

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

	// Eigen::VectorXd getF = mCharacter->GetSkeleton()->getForces();
	// std::cout << "===================================" << std::endl;
	// std::cout << std::endl;
	// for(int i=0; i<getF.size(); i++)
	// {
	// 	std::cout << "idx " << i << " : " << getF[i] << std::endl;
	// }

	mWorld->step();

	mSimCount++;
}

void
Environment::
SetActivationLevels(const Eigen::VectorXd& a)
{
	mCharacter->SetActivationLevels(a);
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

void
Environment::
SetAction(const Eigen::VectorXd& a)
{
	mAction = a*0.1;
	// for(int i=0; i<a.size(); i++)
	// {
	// 	if(a[i] > 10.0)
	// 	{
	// 		std::cout << "action over 10 : " << a[i] << std::endl;
	// 	}
	// }
	mCharacter->SetAction(mAction);

	double t = mWorld->getTime();
	mCharacter->SetTargetPosAndVel(t, mControlHz);

	mSimCount = 0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
}

void
Environment::
SetAction_Device(const Eigen::VectorXd& a)
{
	mAction = a*0.1;
	for(int i=0; i<a.size(); i++)
	{
		if(a[i] > 10.0)
		{
			std::cout << "device action over 10 : " << a[i] << std::endl;
		}
	}
	mCharacter->SetAction_Device(mAction);

	double t = mWorld->getTime();
	mCharacter->SetTargetPosAndVel(t, mControlHz);

	mSimCount = 0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
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

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
	:mWorld(std::make_shared<World>())
{
}

Environment::
~Environment()
{
	delete mCharacter;
	delete mDevice;
}

void
Environment::
Initialize(const std::string& meta_file, bool load_obj)
{
	std::ifstream ifs(meta_file);
	if(!(ifs.is_open()))
	{
		std::cout << "Can't read file " << meta_file << std::endl;
		return;
	}

	std::string str, index;
	std::stringstream ss;
	MASS::Character* character = new MASS::Character();
	MASS::Device* device;
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
			if(!str2.compare("true")){
				this->SetUseDevice(true);
				device = new MASS::Device();
			}
			else{
				this->SetUseDevice(false);
			}
		}
		else if(!index.compare("use_device_nn"))
		{
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseDeviceNN(true);
			else
				this->SetUseDeviceNN(false);
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
				device->LoadSkeleton(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("bvh_file")){
			std::string str2,str3;

			ss>>str2>>str3;
			bool cyclic = false;
			if(!str3.compare("true"))
				cyclic = true;
			character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
		}
	}
	ifs.close();

	this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));
	this->SetCharacter(character);
	if(mUseDevice)
		this->SetDevice(device);

	this->Initialize();

	// auto weld_pelvis = std::make_shared<dart::constraint::WeldJointConstraint>(mCharacter->GetSkeleton()->getBodyNode("Pelvis"));
	// auto weld_spine = std::make_shared<dart::constraint::WeldJointConstraint>(mCharacter->GetSkeleton()->getBodyNode("Spine"));
	// auto weld_handr = std::make_shared<dart::constraint::WeldJointConstraint>(mCharacter->GetSkeleton()->getBodyNode("HandR"));
	// auto weld_handl = std::make_shared<dart::constraint::WeldJointConstraint>(mCharacter->GetSkeleton()->getBodyNode("HandL"));
	// auto weld_talusr = std::make_shared<dart::constraint::WeldJointConstraint>(mCharacter->GetSkeleton()->getBodyNode("TalusR"));

	// mWorld->getConstraintSolver()->addConstraint(weld_pelvis);
	// mWorld->getConstraintSolver()->addConstraint(weld_spine);
	// mWorld->getConstraintSolver()->addConstraint(weld_handr);
	// mWorld->getConstraintSolver()->addConstraint(weld_handl);
	// mWorld->getConstraintSolver()->addConstraint(weld_talusr);
}

void
Environment::
Initialize()
{
	mCharacter->Initialize(mWorld, mControlHz, mSimulationHz);
	if(mUseMuscle)
		mCharacter->Initialize_Muscles();

	if(mUseDevice)
	{
		mDevice->Initialize(mWorld, mUseDeviceNN);

		mDevice->SetCharacter(mCharacter);
		mCharacter->SetDevice(mDevice);
	}

	mCharacter->Initialize_Analysis();

	mNumSteps = mSimulationHz / mControlHz;

	mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));
	mWorld->setTimeStep(1.0/mSimulationHz);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());

	mWorld->addSkeleton(mGround);

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
	mCharacter->Reset();
	if(mUseDevice)
		mDevice->Reset();
}

void
Environment::
Step(bool onDevice)
{
	if(mUseMuscle)
		mCharacter->Step_Muscles(mSimCount, mRandomSampleIndex);
	else
		mCharacter->Step();

	if(mUseDevice)
	{
		mCharacter->SetOnDevice(onDevice);
		if(onDevice)
			mDevice->Step((double)mSimCount/(double)mNumSteps);
	}

	mWorld->step();

	mSimCount++;
}

bool
Environment::
IsEndOfEpisode()
{
	bool isTerminal = false;

	Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();

	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];

	if(root_y < 1.3)
		isTerminal =true;
	else if (dart::math::isNan(p) || dart::math::isNan(v))
		isTerminal =true;
	else if(mWorld->getTime() > 5.0)
		isTerminal =true;

	return isTerminal;
}

void
Environment::
SetAction(const Eigen::VectorXd& a)
{
	mCharacter->SetAction(a);

	mSimCount = 0;
	mRandomSampleIndex = rand()%(mNumSteps);
}

void
Environment::
SetAction_Device(const Eigen::VectorXd& a)
{
	mDevice->SetAction(a);
}

Eigen::VectorXd
Environment::
GetState()
{
	return mCharacter->GetState();
}

Eigen::VectorXd
Environment::
GetState_Device()
{
	return mDevice->GetState();
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
GetNumState()
{
	return mCharacter->GetNumState();
}

int
Environment::
GetNumState_Device()
{
	return mDevice->GetNumState();
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
	return mDevice->GetNumAction();
}

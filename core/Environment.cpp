#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "Device.h"
#include "dart/collision/bullet/bullet.hpp"
#include <ctime>
#include <regex>
using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;
using namespace std;

Environment::
Environment()
	:mWorld(std::make_shared<World>())
{
	std::srand(std::time(nullptr));
}

Environment::
~Environment()
{
	delete mCharacter;
	delete mDevice;
}

void
Environment::
ParseMetaFile(const std::string& meta_file)
{
	std::ifstream ifs(meta_file);
	if(!(ifs.is_open()))
	{
		std::cout << "Can't read file " << meta_file << std::endl;
		return;
	}

	std::string str, index;
	std::stringstream ss;
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
		else if(!index.compare("use_device_nn"))
		{
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseDeviceNN(true);
			else
				this->SetUseDeviceNN(false);
		}
		else if(!index.compare("use_adaptive_sampling"))
		{
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseAdaptiveSampling(true);
			else
				this->SetUseAdaptiveSampling(false);
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
			mSkelFile = std::string(MASS_ROOT_DIR)+str2;
		}
		else if(!index.compare("muscle_file")){
			std::string str2;
			ss>>str2;
			if(this->GetUseMuscle())
				mMuscleFile = std::string(MASS_ROOT_DIR)+str2;
		}
		else if(!index.compare("device_file")){
			std::string str2;
			ss>>str2;
			if(this->GetUseDevice())
				mDeviceFile = std::string(MASS_ROOT_DIR)+str2;
		}
		else if(!index.compare("bvh_file")){
			std::string str2,str3;
			ss>>str2>>str3;
			mCyclic = false;
			if(!str3.compare("true"))
				mCyclic = true;
			mBVHFile = std::string(MASS_ROOT_DIR)+str2;
		}
	}
	ifs.close();
}

void
Environment::
ParseAdaptiveFile(const std::string& adaptive_file)
{
	std::ifstream ifs(adaptive_file);
	if(!(ifs.is_open()))
	{
		std::cout << "Can't read file " << adaptive_file << std::endl;
		return;
	}

	std::string str, index;
	std::stringstream ss;
	while(!ifs.eof())
	{
		ss.clear();
		str.clear();
		index.clear();

		std::getline(ifs, str);
		ss.str(str);
		ss>>index;
		if(!index.compare("character_param")){
			int numParamState;
			ss>>numParamState;
			mNumParamState_Character = numParamState;
		}
		else if(!index.compare("device_param")){
			int numParamState;
			ss>>numParamState;
			mNumParamState_Device = numParamState;
		}
		else if(!index.compare("mass")){
			double lower, upper;
			ss>>lower>>upper;
			mParam_Character.insert(std::make_pair("mass", std::make_pair(lower,upper)));
		}
		else if(!index.compare("force")){
			double lower, upper;
			ss>>lower>>upper;
			mParam_Character.insert(std::make_pair("force", std::make_pair(lower,upper)));
		}
		else if(!index.compare("speed")){
			double lower, upper;
			ss>>lower>>upper;
			mParam_Character.insert(std::make_pair("speed", std::make_pair(lower,upper)));
		}
		else if(!index.compare("device_k")){
			double lower, upper;
			ss>>lower>>upper;
			if(mUseDevice)
				mParam_Device.insert(std::make_pair("k", std::make_pair(lower,upper)));
		}
		else if(!index.compare("delta_t")){
			double lower, upper;
			ss>>lower>>upper;
			if(mUseDevice)
				mParam_Device.insert(std::make_pair("delta_t", std::make_pair(lower,upper)));
		}
	}
	ifs.close();
}

void
Environment::
SetWorld()
{
	mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));
	mWorld->setTimeStep(1.0/mSimulationHz);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());

	mGround = MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml"));
	mWorld->addSkeleton(mGround);
}

void
Environment::
Initialize(const std::string& meta_file, bool load_obj)
{
	this->ParseMetaFile(meta_file);
	this->SetWorld();
	this->SetNumSteps(mSimulationHz/mControlHz);

	MASS::Character* character = new MASS::Character(mWorld);
	character->LoadSkeleton(mSkelFile, load_obj);
	character->LoadBVH(mBVHFile, mCyclic);
	character->SetHz(mSimulationHz, mControlHz);
	if(mUseMuscle){
		character->SetUseMuscle(true);
		character->LoadMuscles(mMuscleFile);
	}
	this->SetCharacter(character);

	MASS::Device* device;
	if(mUseDevice){
		device = new MASS::Device(mWorld);
		device->LoadSkeleton(mDeviceFile,load_obj);
		device->SetHz(mSimulationHz, mControlHz);
		device->SetCharacter(character);
		if(mUseDeviceNN)
			device->SetUseDeviceNN(true);
		this->SetDevice(device);
	}

	if(mUseAdaptiveSampling)
		this->SetAdaptiveParamNums();

	mCharacter->Initialize();

	if(mUseDevice){
		mDevice->Initialize();
		mCharacter->SetUseDevice(true);
		mCharacter->SetDevice(mDevice);
		mCharacter->SetConstraints();
	}

	if(mUseAdaptiveSampling)
		this->SetAdaptiveParams();

	this->Reset();
	// auto weld_pelvis = std::make_shared<dart::constraint::WeldJointConstraint>(mCharacter->GetSkeleton()->getBodyNode("Pelvis"));
	// mWorld->getConstraintSolver()->addConstraint(weld_pelvis);
}

void
Environment::
SetAdaptiveParamNums()
{
	this->ParseAdaptiveFile("../data/adaptive.txt");

	int numParam = mNumParamState_Character;
	mCharacter->SetNumParamState(mNumParamState_Character);

	if(mUseDevice){
		mDevice->SetNumParamState(mNumParamState_Device);
		numParam += mNumParamState_Device;
	}

	this->SetNumParamState(numParam);
}

void
Environment::
SetAdaptiveParams()
{
	mCharacter->SetAdaptiveParams(mParam_Character);
	if(mUseDevice)
		mDevice->SetAdaptiveParams(mParam_Device);
}

void
Environment::
Reset(bool RSI)
{
	double t = 0.04;
	if(RSI)
		t = dart::math::random(0.0,mCharacter->GetBVH()->GetMaxTime()*0.9);

	mWorld->reset();
	mWorld->setTime(t);
	mCharacter->Reset();
	if(mUseDevice)
		mDevice->Reset();
}

void
Environment::
Step(bool device_onoff)
{
	if(mUseMuscle)
		mCharacter->Step_Muscles(mSimCount, mRandomSampleIndex);
	else
		mCharacter->Step();

	if(mUseDevice)
	{
		mCharacter->SetDevice_OnOff(device_onoff);
		if(device_onoff)
			mDevice->Step((double)mSimCount/(double)mNumSteps);
	}

	// if(mStepCnt == 20)
	// 	mStepCnt = 0;
	// mStepCnt++;
	// mStepCnt_total++;

	mWorld->step();
	mSimCount++;
}

bool
Environment::
IsEndOfEpisode()
{
	bool isTerminal = false;

	auto char_skel = mCharacter->GetSkeleton();

	Eigen::VectorXd p = char_skel->getPositions();
	Eigen::VectorXd v = char_skel->getVelocities();

	double root_y = char_skel->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];

	if(root_y < 1.4)
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

std::map<std::string, std::deque<double>>
Environment::
GetRewards()
{
	return mCharacter->GetRewards();
}

int
Environment::
GetNumState()
{
	return mCharacter->GetNumState();
}

int
Environment::
GetNumState_Char()
{
	return mCharacter->GetNumState_Char();
}

int
Environment::
GetNumParamState_Char()
{
	return mCharacter->GetNumParamState();
}

int
Environment::
GetNumState_Device()
{
	return mDevice->GetNumState();
}

int
Environment::
GetNumParamState_Device()
{
	return mDevice->GetNumParamState();
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

void
Environment::
SetParamState(Eigen::VectorXd paramState)
{
	int paramState_Char = mCharacter->GetNumParamState();
	mCharacter->SetParamState(paramState.segment(0,paramState_Char));

	if(mUseDevice)
	{
		int paramState_Device = mDevice->GetNumParamState();
		mDevice->SetParamState(paramState.segment(paramState_Char,paramState_Device));
	}
}

Eigen::VectorXd
Environment::
GetParamState()
{
	int param_dim = 0;
	int char_dim = mCharacter->GetNumParamState();
	int device_dim = mDevice->GetNumParamState();

	param_dim += char_dim;
	if(mUseDevice)
		param_dim += device_dim;

	Eigen::VectorXd paramState(param_dim);
	paramState << mCharacter->GetParamState();
	if(mUseDevice)
		paramState << mCharacter->GetParamState(),mDevice->GetParamState();

	return paramState;
}

Eigen::VectorXd
Environment::
GetMinV()
{
	int param_dim = 0;
	param_dim += mCharacter->GetNumParamState();
	if(mUseDevice)
		param_dim += mDevice->GetNumParamState();

	Eigen::VectorXd min_v(param_dim);
	min_v << mCharacter->GetMinV();
	if(mUseDevice)
		min_v << mCharacter->GetMinV(), mDevice->GetMinV();

	return min_v;
}

Eigen::VectorXd
Environment::
GetMaxV()
{
	int param_dim = 0;
	param_dim += mCharacter->GetNumParamState();
	if(mUseDevice)
		param_dim += mDevice->GetNumParamState();

	Eigen::VectorXd max_v(param_dim);

	max_v << mCharacter->GetMaxV();
	if(mUseDevice)
		max_v << mCharacter->GetMaxV(), mDevice->GetMaxV();

	return max_v;
}

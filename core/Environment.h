#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
#include <map>
#include <deque>

namespace MASS
{

class Environment
{
public:
	Environment();
	~Environment();

	void Initialize(const std::string& meta_file, bool load_obj);
	void ParseMetaFile(const std::string& meta_file);
	void ParseAdaptiveFile(const std::string& adaptive_file);

	void Reset(bool RSI = true);
	void Step(bool device_onoff, bool isRender);
	bool IsEndOfEpisode();

	void SetAction(const Eigen::VectorXd& a);
	void SetAction_Device(const Eigen::VectorXd& a);

	Eigen::VectorXd GetState();
	Eigen::VectorXd GetState_Device();
	double GetReward();
	std::map<std::string, std::deque<double>> GetRewards();

	int GetNumState();
	int GetNumState_Char();
	int GetNumState_Device();
	int GetNumParamState_Char();
	int GetNumParamState_Device();
	int GetNumAction();
	int GetNumAction_Device();

public:
	void SetWorld();
	void SetId(int i){mId = i;}
	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetUseDevice(bool use_device){mUseDevice = use_device;}
	void SetUseDeviceNN(bool use_device_nn){mUseDeviceNN = use_device_nn;}
	void SetUseAdaptiveSampling(bool b){mUseAdaptiveSampling = b;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}
	void SetNumSteps(int n) {mNumSteps = n;}
	void SetCharacter(Character* character) {mCharacter = character;}
	void SetDevice(Device* device) {mDevice = device;}

	int GetId(){return mId;}
	bool GetUseMuscle(){return mUseMuscle;}
	bool GetUseDevice(){return mUseDevice;}
	bool GetUseDeviceNN(){return mUseDeviceNN;}
	bool GetUseAdaptiveSampling(){return mUseAdaptiveSampling;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumSteps(){return mNumSteps;}
	Character* GetCharacter(){return mCharacter;}
	Device* GetDevice(){return mDevice;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}

	void SetAdaptiveParamNums();
	void SetAdaptiveParams();
	void SetParamState(Eigen::VectorXd paramState);
	void SetNumParamState(int n){mNumParamState=n;}
	int GetNumParamState(){return mNumParamState;}
	Eigen::VectorXd GetParamState();
	Eigen::VectorXd GetMinV();
	Eigen::VectorXd GetMaxV();

private:
	dart::simulation::WorldPtr mWorld;
	dart::dynamics::SkeletonPtr mGround;
	Character* mCharacter;
	Device* mDevice;

	std::string mSkelFile;
	std::string mMuscleFile;
	std::string mDeviceFile;
	std::string mBVHFile;

	bool mCyclic;

	int mId;
	int mControlHz;
	int mSimulationHz;
	int mNumSteps;
	int mSimCount;
	int mRandomSampleIndex;

	bool mUseMuscle;
	bool mUseDevice;
	bool mUseDeviceNN;
	bool mUseAdaptiveSampling;

	int mNumParamState;
	int mNumParamState_Character;
	int mNumParamState_Device;
	std::map<std::string, std::pair<double, double>> mParam_Character;
	std::map<std::string, std::pair<double, double>> mParam_Device;
};
};

#endif

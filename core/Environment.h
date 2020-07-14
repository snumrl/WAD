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

	void Initialize(const std::string& meta_file, bool load_obj = false);
	void Initialize();

	void Reset(bool RSI = true);
	void Step(bool onDevice);
	bool IsEndOfEpisode();
	void SetAction(const Eigen::VectorXd& a);
	void SetAction_Device(const Eigen::VectorXd& a);

	Eigen::VectorXd GetState();
	Eigen::VectorXd GetState_Device();

	double GetReward();
	std::map<std::string,double> GetRewardSep();

	int GetNumState();
	int GetNumState_Device();
	int GetNumAction();
	int GetNumAction_Device();

public:
	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetUseDevice(bool use_device){mUseDevice = use_device;}
	void SetUseDeviceNN(bool use_device_nn){mUseDeviceNN = use_device_nn;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}
	void SetCharacter(Character* character) {mCharacter = character;}
	void SetDevice(Device* device) {mDevice = device;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

	bool GetUseMuscle(){return mUseMuscle;}
	bool GetUseDevice(){return mUseDevice;}
	bool GetUseDeviceNN(){return mUseDeviceNN;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumSteps(){return mNumSteps;}
	Character* GetCharacter(){return mCharacter;}
	Device* GetDevice(){return mDevice;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}

	// double GetPhase(){return mCharacter->GetPhase();}
	// std::map<std::string, std::vector<double>> GetEnergy(int idx){return mCharacter->GetEnergy(idx);}
	// std::vector<double> GetReward_Graph(int idx){return mCharacter->GetReward_Graph(idx);}
	// std::deque<double> GetDeviceSignals(int idx);
	// std::deque<double> GetSignals(){return mCharacter->GetSignals();}

private:
	dart::simulation::WorldPtr mWorld;
	dart::dynamics::SkeletonPtr mGround;
	Character* mCharacter;
	Device* mDevice;

	int mControlHz;
	int mSimulationHz;
	int mNumSteps;

	bool mUseMuscle;
	bool mUseDevice;
	bool mUseDeviceNN;

	int mSimCount;
	int mRandomSampleIndex;
};
};

#endif

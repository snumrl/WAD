#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
#include <map>

namespace MASS
{

class Environment
{
public:
	Environment();

	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetUseDevice(bool use_device){mUseDevice = use_device;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}

	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

	void Initialize();
	void Initialize(const std::string& meta_file, bool load_obj = false);

public:
	void Step(bool onDevice);
	void StepDeviceOnly();
	void StepBack();
	void Reset(bool RSI = true);
	bool IsEndOfEpisode();
	void SetAction(const Eigen::VectorXd& a);
	void SetAction_Device(const Eigen::VectorXd& a);
	void SetActivationLevels(const Eigen::VectorXd& a);
	double GetReward();

	std::map<std::string,double> GetRewardSep();

	Eigen::VectorXd GetState();
	Eigen::VectorXd GetState_Device();

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetMuscleTorques();
	std::vector<MuscleTuple>& GetMuscleTuples();

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	Character* GetCharacter(){return mCharacter;}

	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs();

	int GetNumState();
	int GetNumState_Device();
	int GetNumAction();
	int GetNumAction_Device();
	int GetNumSteps(){return mSimulationHz/mControlHz;}

	bool GetUseMuscle(){return mUseMuscle;}
	bool GetUseDevice(){return mUseDevice;}

private:
	dart::simulation::WorldPtr mWorld;
	dart::dynamics::SkeletonPtr mGround;
	Character* mCharacter;
	Eigen::VectorXd mAction;
	Eigen::VectorXd mAction_Device;

	int mControlHz;
	int mSimulationHz;

	bool mUseMuscle;
	bool mUseDevice;

	int mSimCount;
	int mRandomSampleIndex;

	double r_only = 0.0;
	double r_d = 0.0;
};
};

#endif

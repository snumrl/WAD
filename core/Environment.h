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
	void Reset(bool RSI = true);
	bool IsEndOfEpisode();
	void SetAction(const Eigen::VectorXd& a);
	void SetAction_Device(const Eigen::VectorXd& a);
	void SetActivationLevels(const Eigen::VectorXd& a);
	void SetNumSteps();
	std::map<std::string,double> GetRewardSep();

	Eigen::VectorXd GetState();
	Eigen::VectorXd GetState_Device();

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetMuscleTorques();
	std::vector<MuscleTuple>& GetMuscleTuples();

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	Character* GetCharacter(){return mCharacter;}

	double GetReward();

	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumSteps(){return mNumSteps;}

	int GetNumState();
	int GetNumState_Device();
	int GetNumAction();
	int GetNumAction_Device();
	int GetNumTotalRelatedDofs();

	bool GetUseMuscle(){return mUseMuscle;}
	bool GetUseDevice(){return mUseDevice;}
	std::map<std::string, std::vector<double>> GetEnergy(int idx){return mCharacter->GetEnergy(idx);}
	std::vector<double> GetReward_Graph(int idx){return mCharacter->GetReward_Graph(idx);}
	std::deque<double> GetDeviceSignals(int idx);
	double GetPhase(){return mCharacter->GetPhase();}

private:
	dart::simulation::WorldPtr mWorld;
	dart::dynamics::SkeletonPtr mGround;
	Character* mCharacter;
	Eigen::VectorXd mAction;
	Eigen::VectorXd mAction_Device;

	int mControlHz;
	int mSimulationHz;
	int mNumSteps;

	bool mUseMuscle;
	bool mUseDevice;

	int mSimCount;
	int mRandomSampleIndex;

};
};

#endif

#ifndef __ENV_MANAGER_H__
#define __ENV_MANAGER_H__
#include "Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "NumPyHelper.h"
class EnvManager
{
public:
	EnvManager(std::string meta_file,int num_envs);

	void Step(int id);
	void Steps(int num, bool onDevice);
	void StepsAtOnce(bool onDevice);

	void Reset(bool RSI, int id);
	void Resets(bool RSI);

	np::ndarray GetState(int id);
	np::ndarray GetStates();
	np::ndarray GetState_Device(int id);
	np::ndarray GetStates_Device();

	double GetReward(int id);
	np::ndarray GetRewards();

	void SetAction(np::ndarray np_array, int id);
	void SetActions(np::ndarray np_array);
	void SetActions_Device(np::ndarray np_array);
	void SetActivationLevels(np::ndarray np_array);

	bool IsEndOfEpisode(int id);
	np::ndarray IsEndOfEpisodes();

	int GetNumState();
	int GetNumState_Device();
	int GetNumAction();
	int GetNumAction_Device();
	int GetNumSteps();
	int GetControlHz();
	int GetSimulationHz();

	bool UseMuscle();
	bool UseDevice();
	bool UseDeviceNN();

	void SetDesiredTorques();
	np::ndarray GetDesiredTorques();
	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs();
	int GetNumMuscles();
	np::ndarray GetMuscleTorques();
	p::list GetMuscleTuples();

	// adaptive sampling
	bool UseAdaptiveSampling();
	void SetParamState(int id, np::ndarray np_array);
	int GetNumParamState();
	np::ndarray GetMinV();
	np::ndarray GetMaxV();

private:
	std::vector<MASS::Environment*> mEnvs;

	int mNumEnvs;
};

#endif

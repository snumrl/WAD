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
	void Steps(int num);
	void StepsAtOnce();
	void StepsTrain(int num);
	void StepsAtOnceTrain();

	void Reset(bool RSI,int id);
	void Resets(bool RSI);

	np::ndarray GetState(int id);
	np::ndarray GetStates();
	np::ndarray GetStates_Device();

	double GetReward(int id);
	p::dict GetRewardSep(int id);
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

	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs(){return mEnvs[0]->GetNumTotalRelatedDofs();};
	int GetNumMuscles(){return mEnvs[0]->GetCharacter()->GetMuscles().size();}
	np::ndarray GetMuscleTorques();
	np::ndarray GetDesiredTorques();
	p::list GetMuscleTuples();

private:
	std::vector<MASS::Environment*> mEnvs;

	int mNumEnvs;
};

#endif
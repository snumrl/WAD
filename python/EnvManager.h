#ifndef __ENV_MANAGER_H__
#define __ENV_MANAGER_H__
#include "Environment.h"
#include "NumPyHelper.h"
#include "DARTHelper.h"

namespace py = pybind11;
class EnvManager
{
public:
	EnvManager(std::string meta_file, int num_envs);

	void Step(int id);
	void Steps(int num, bool onDevice);
	void StepsAtOnce(bool onDevice);

	void Reset(bool RSI, int id);
	void Resets(bool RSI);

	py::array_t<float> GetState(int id);
	py::array_t<float> GetStates();
	py::array_t<float> GetState_Device(int id);
	py::array_t<float> GetStates_Device();

	py::array_t<float> GetReward(int id);

	double GetAdaptiveTime(int id);
	py::array_t<float> GetAdaptiveTimes();

	void SetAction(py::array_t<float> np_array, int id);
	void SetActions(py::array_t<float> np_array);
	void SetActions_Device(py::array_t<float> np_array);
	void SetActivationLevels(py::array_t<float> np_array);

	bool IsEndOfEpisode(int id);
	py::array_t<float> IsEndOfEpisodes();

	int GetNumState();
	int GetNumState_Char();
	int GetNumState_Device();
	int GetNumAction();
	int GetNumAction_Device();
	int GetNumActiveDof();
	int GetNumSteps();
	int GetControlHz();
	int GetSimulationHz();

	bool UseMuscle();
	bool UseDevice();
	bool UseDeviceNN();
	bool UseAdaptiveMotion();

	void SetDesiredTorques();
	py::array_t<float> GetDesiredTorques();

	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs();
	int GetNumMuscles();
	py::list GetMuscleTuples();
	py::array_t<float> GetMuscleTorques();

	// adaptive sampling
	bool UseAdaptiveSampling();
	void SetParamState(int id, py::array_t<float> np_array);
	int GetNumParamState();
	int GetNumParamState_Char();
	int GetNumParamState_Device();
	py::array_t<float> GetMinV();
	py::array_t<float> GetMaxV();

private:
	std::vector<MASS::Environment*> mEnvs;
	int mNumEnvs;
};

#endif

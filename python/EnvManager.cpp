#include "EnvManager.h"
#include <omp.h>

EnvManager::
EnvManager(std::string meta_file,int num_envs)
	:mNumEnvs(num_envs)
{
	dart::math::seedRand();
	omp_set_num_threads(mNumEnvs);
	for(int i = 0;i<mNumEnvs;i++){
		mEnvs.push_back(new WAD::Environment());
		WAD::Environment* env = mEnvs.back();

		env->SetId(i);
		env->Initialize(meta_file, false);
	}
}

void
EnvManager::
Step(int num, bool onDevice, int id)
{
	for(int i=0; i<num; i++)
		mEnvs[id]->Step(onDevice, false);
}

void
EnvManager::
Steps(int num, bool onDevice)
{
#pragma omp parallel for
	for (int id=0; id<mNumEnvs; ++id)
	{
		for(int j=0; j<num; j++)
			mEnvs[id]->Step(onDevice, false);
	}
}

void
EnvManager::
StepsAtOnce(bool onDevice)
{
	int num = this->GetNumSteps();
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++)
			mEnvs[id]->Step(onDevice, false);
	}
}

void
EnvManager::
Reset(bool RSI, int id)
{
	mEnvs[id]->Reset(RSI);
}

void
EnvManager::
Resets(bool RSI)
{
#pragma omp parallel for
	for (int id=0; id<mNumEnvs; ++id)
	{
		mEnvs[id]->Reset(RSI);
	}
}

py::array_t<float>
EnvManager::
GetState(int id)
{
	return toNumPyArray(mEnvs[id]->GetState());
}

py::array_t<float>
EnvManager::
GetStates()
{
	Eigen::MatrixXd states(mNumEnvs,this->GetNumState());
#pragma omp parallel for
	for (int id=0; id<mNumEnvs; ++id)
	{
		states.row(id) = mEnvs[id]->GetState().transpose();
	}

	return toNumPyArray(states);
}

py::array_t<float>
EnvManager::
GetState_Device(int id)
{
	return toNumPyArray(mEnvs[id]->GetState_Device());
}

py::array_t<float>
EnvManager::
GetStates_Device()
{
	Eigen::MatrixXd states(mNumEnvs,this->GetNumState_Device());
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		states.row(id) = mEnvs[id]->GetState_Device().transpose();
	}

	return toNumPyArray(states);
}

py::array_t<float>
EnvManager::
GetReward(int id)
{
	return toNumPyArray(mEnvs[id]->GetReward());
}

double
EnvManager::
GetAdaptiveTime(int id)
{
	return mEnvs[id]->GetAdaptiveTime();
}

py::array_t<float>
EnvManager::
GetAdaptiveTimes()
{
	std::vector<float> adaptiveTimes(mNumEnvs);
	for (int id = 0;id<mNumEnvs;++id)
	{
		adaptiveTimes[id] = mEnvs[id]->GetAdaptiveTime();
	}
	return toNumPyArray(adaptiveTimes);
}

void
EnvManager::
SetAction(py::array_t<float> np_array, int id)
{
	mEnvs[id]->SetAction(toEigenVector(np_array));
}

void
EnvManager::
SetActions(py::array_t<float> np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetAction(action.row(id).transpose());
	}
}

void
EnvManager::
SetActions_Device(py::array_t<float> np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetAction_Device(action.row(id).transpose());
	}
}

void
EnvManager::
SetActivationLevel(py::array_t<float> np_array, int id)
{	
	mEnvs[id]->GetCharacter()->SetActivationLevels(toEigenVector(np_array));
}

void
EnvManager::
SetActivationLevels(py::array_t<float> np_array)
{
	std::vector<Eigen::VectorXd> activations = toEigenVectorVector(np_array);
#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mEnvs[id]->GetCharacter()->SetActivationLevels(activations[id]);
	}
}

bool
EnvManager::
IsEndOfEpisode(int id)
{
	return mEnvs[id]->IsEndOfEpisode();
}

py::array_t<float>
EnvManager::
IsEndOfEpisodes()
{
	std::vector<bool> is_end_vector(mNumEnvs);
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		is_end_vector[id] = mEnvs[id]->IsEndOfEpisode();
	}

	return toNumPyArray(is_end_vector);
}

int
EnvManager::
GetNumState()
{
	return mEnvs[0]->GetNumState();
}

int
EnvManager::
GetNumState_Char()
{
	return mEnvs[0]->GetNumState_Char();
}

int
EnvManager::
GetNumState_Device()
{
	return mEnvs[0]->GetNumState_Device();
}

int
EnvManager::
GetNumActiveDof()
{
	return mEnvs[0]->GetNumActiveDof();
}

int
EnvManager::
GetNumAction()
{
	return mEnvs[0]->GetNumAction();
}

int
EnvManager::
GetNumAction_Device()
{
	return mEnvs[0]->GetNumAction_Device();
}

int
EnvManager::
GetNumSteps()
{
	return mEnvs[0]->GetNumSteps();
}

int
EnvManager::
GetControlHz()
{
	return mEnvs[0]->GetControlHz();
}

int
EnvManager::
GetSimulationHz()
{
	return mEnvs[0]->GetSimulationHz();
}

bool
EnvManager::
UseMuscle()
{
	return mEnvs[0]->GetUseMuscle();
}

bool
EnvManager::
UseDevice()
{
	return mEnvs[0]->GetUseDevice();
}

bool
EnvManager::
UseDeviceNN()
{
	return mEnvs[0]->GetUseDeviceNN();
}

bool
EnvManager::
UseAdaptiveMotion()
{
	return mEnvs[0]->GetUseAdaptiveMotion();
}

void
EnvManager::
SetDesiredTorques()
{
#pragma omp parallel for
	for (int id=0; id < mNumEnvs; ++id)
	{
		mEnvs[id]->GetCharacter()->SetDesiredTorques();
	}
}

py::array_t<float>
EnvManager::
GetDesiredTorques()
{
	std::vector<Eigen::VectorXd> tau_des(mNumEnvs);

#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		tau_des[id] = mEnvs[id]->GetCharacter()->GetDesiredTorques();
	}
	return toNumPyArray(tau_des);
}

int
EnvManager::
GetNumTotalMuscleRelatedDofs()
{
	std::cout << "related dofs : " << mEnvs[0]->GetCharacter()->GetNumTotalRelatedDofs() << std::endl;
	return mEnvs[0]->GetCharacter()->GetNumTotalRelatedDofs();
}

int
EnvManager::
GetNumMuscles()
{
	return mEnvs[0]->GetCharacter()->GetNumMuscles();
}

py::array_t<float>
EnvManager::
GetMuscleTorques()
{
	std::vector<Eigen::VectorXd> mt(mNumEnvs);

#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mt[id] = mEnvs[id]->GetCharacter()->GetMuscleTorques();
	}
	return toNumPyArray(mt);
}

py::list
EnvManager::
GetMuscleTuples()
{
	py::list all;
	for (int id = 0; id < mNumEnvs; ++id)
	{
		auto& tps = mEnvs[id]->GetCharacter()->GetMuscleTuples();
		for(int j=0;j<tps.size();j++)
		{
			py::list t;
			t.append(toNumPyArray(tps[j].JtA));
			t.append(toNumPyArray(tps[j].tau_des));
			t.append(toNumPyArray(tps[j].L));
			t.append(toNumPyArray(tps[j].b));
			all.append(t);
		}
		tps.clear();
	}

	return all;
}

bool
EnvManager::UseAdaptiveSampling()
{
	return mEnvs[0]->GetUseAdaptiveSampling();
}

void
EnvManager::
SetParamState(int id, py::array_t<float> np_array)
{
	mEnvs[id]->SetParamState(toEigenVector(np_array));
}

int
EnvManager::
GetNumParamState()
{
	return mEnvs[0]->GetNumParamState();
}

int
EnvManager::
GetNumParamState_Char()
{
	return mEnvs[0]->GetNumParamState_Char();
}

int
EnvManager::
GetNumParamState_Device()
{
	return mEnvs[0]->GetNumParamState_Device();
}

py::array_t<float>
EnvManager::
GetMinV()
{
	return toNumPyArray(mEnvs[0]->GetMinV());
}

py::array_t<float>
EnvManager::
GetMaxV()
{
	return toNumPyArray(mEnvs[0]->GetMaxV());
}

PYBIND11_MODULE(pywad, m){
	py::class_<EnvManager>(m, "EnvManager")
		.def(py::init<std::string, int>())
		.def("Step",&EnvManager::Step)
		.def("Steps",&EnvManager::Steps)
		.def("StepsAtOnce",&EnvManager::StepsAtOnce)
		.def("Reset",&EnvManager::Reset)
		.def("Resets",&EnvManager::Resets)
		.def("GetState",&EnvManager::GetState)
		.def("GetStates",&EnvManager::GetStates)
		.def("GetState_Device",&EnvManager::GetState_Device)
		.def("GetStates_Device",&EnvManager::GetStates_Device)
		.def("GetReward",&EnvManager::GetReward)
		.def("GetAdaptiveTime",&EnvManager::GetAdaptiveTime)
		.def("GetAdaptiveTimes",&EnvManager::GetAdaptiveTimes)
		.def("SetAction",&EnvManager::SetAction)
		.def("SetActions",&EnvManager::SetActions)
		.def("SetActions_Device",&EnvManager::SetActions_Device)
		.def("SetActivationLevel",&EnvManager::SetActivationLevel)
		.def("SetActivationLevels",&EnvManager::SetActivationLevels)
		.def("IsEndOfEpisode",&EnvManager::IsEndOfEpisode)
		.def("IsEndOfEpisodes",&EnvManager::IsEndOfEpisodes)
		.def("GetNumState",&EnvManager::GetNumState)
		.def("GetNumState_Char",&EnvManager::GetNumState_Char)
		.def("GetNumState_Device",&EnvManager::GetNumState_Device)
		.def("GetNumActiveDof",&EnvManager::GetNumActiveDof)
		.def("GetNumAction",&EnvManager::GetNumAction)
		.def("GetNumAction_Device",&EnvManager::GetNumAction_Device)
		.def("GetNumSteps",&EnvManager::GetNumSteps)
		.def("GetControlHz",&EnvManager::GetControlHz)
		.def("GetSimulationHz",&EnvManager::GetSimulationHz)
		.def("UseMuscle",&EnvManager::UseMuscle)
		.def("UseDevice",&EnvManager::UseDevice)
		.def("UseDeviceNN",&EnvManager::UseDeviceNN)
		.def("UseAdaptiveMotion",&EnvManager::UseAdaptiveMotion)
		.def("SetDesiredTorques",&EnvManager::SetDesiredTorques)
		.def("GetDesiredTorques",&EnvManager::GetDesiredTorques)
		.def("GetNumTotalMuscleRelatedDofs",&EnvManager::GetNumTotalMuscleRelatedDofs)
		.def("GetNumMuscles",&EnvManager::GetNumMuscles)
		.def("GetMuscleTorques",&EnvManager::GetMuscleTorques)
		.def("GetMuscleTuples",&EnvManager::GetMuscleTuples)
		.def("UseAdaptiveSampling", &EnvManager::UseAdaptiveSampling)
		.def("SetParamState", &EnvManager::SetParamState)
		.def("GetNumParamState", &EnvManager::GetNumParamState)
		.def("GetNumParamState_Char", &EnvManager::GetNumParamState_Char)
		.def("GetNumParamState_Device", &EnvManager::GetNumParamState_Device)
		.def("GetMinV", &EnvManager::GetMinV)
		.def("GetMaxV", &EnvManager::GetMaxV);
}

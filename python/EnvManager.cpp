#include "EnvManager.h"
#include "DARTHelper.h"
#include "Device.h"
#include <omp.h>

EnvManager::
EnvManager(std::string meta_file,int num_envs)
	:mNumEnvs(num_envs)
{
	dart::math::seedRand();
	omp_set_num_threads(mNumEnvs);
	for(int i = 0;i<mNumEnvs;i++){
		mEnvs.push_back(new MASS::Environment());
		MASS::Environment* env = mEnvs.back();

		env->Initialize(meta_file,false);
	}
}

void
EnvManager::
Step(int id)
{
	mEnvs[id]->Step(false);
}

void
EnvManager::
Steps(int num, bool onDevice)
{
#pragma omp parallel for
	for (int id=0; id<mNumEnvs; ++id)
	{
		for(int j=0; j<num; j++)
			mEnvs[id]->Step(onDevice);
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
			mEnvs[id]->Step(onDevice);
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
	for (int id=0; id<mNumEnvs; ++id)
	{
		mEnvs[id]->Reset(RSI);
	}
}

np::ndarray
EnvManager::
GetState(int id)
{
	return toNumPyArray(mEnvs[id]->GetState());
}

np::ndarray
EnvManager::
GetStates()
{
	Eigen::MatrixXd states(mNumEnvs,this->GetNumState());
	for (int id=0; id<mNumEnvs; ++id)
	{
		states.row(id) = mEnvs[id]->GetState().transpose();
	}

	return toNumPyArray(states);
}

np::ndarray
EnvManager::
GetState_Device(int id)
{
	return toNumPyArray(mEnvs[id]->GetState_Device());
}

np::ndarray
EnvManager::
GetStates_Device()
{
	Eigen::MatrixXd states(mNumEnvs,this->GetNumState_Device());
	for (int id = 0;id<mNumEnvs;++id)
	{
		states.row(id) = mEnvs[id]->GetState_Device().transpose();
	}

	return toNumPyArray(states);
}

double
EnvManager::
GetReward(int id)
{
	return mEnvs[id]->GetReward();
}

np::ndarray
EnvManager::
GetRewards()
{
	std::vector<float> rewards(mNumEnvs);
	for (int id = 0;id<mNumEnvs;++id)
	{
		rewards[id] = mEnvs[id]->GetReward();
	}
	return toNumPyArray(rewards);
}

void
EnvManager::
SetAction(np::ndarray np_array, int id)
{
	mEnvs[id]->SetAction(toEigenVector(np_array));
}

void
EnvManager::
SetActions(np::ndarray np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetAction(action.row(id).transpose());
	}
}

void
EnvManager::
SetActions_Device(np::ndarray np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetAction_Device(action.row(id).transpose());
	}
}

void
EnvManager::
SetActivationLevels(np::ndarray np_array)
{
	std::vector<Eigen::VectorXd> activations = toEigenVectorVector(np_array);
	for (int id = 0; id < mNumEnvs; ++id)
		mEnvs[id]->GetCharacter()->SetActivationLevels(activations[id]);
}

bool
EnvManager::
IsEndOfEpisode(int id)
{
	return mEnvs[id]->IsEndOfEpisode();
}

np::ndarray
EnvManager::
IsEndOfEpisodes()
{
	std::vector<bool> is_end_vector(mNumEnvs);
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
GetNumState_Device()
{
	return mEnvs[0]->GetNumState_Device();
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

np::ndarray
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

np::ndarray
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

p::list
EnvManager::
GetMuscleTuples()
{
	p::list all;
	for (int id = 0; id < mNumEnvs; ++id)
	{
		auto& tps = mEnvs[id]->GetCharacter()->GetMuscleTuples();
		for(int j=0;j<tps.size();j++)
		{
			p::list t;
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
SetParamState(int id, np::ndarray np_array)
{
	mEnvs[id]->SetParamState(toEigenVector(np_array));
}

int
EnvManager::
GetNumParamState()
{
	return mEnvs[0]->GetNumParamState();
}

np::ndarray
EnvManager::
GetMinV()
{
	return toNumPyArray(mEnvs[0]->GetMinV());
}

np::ndarray
EnvManager::
GetMaxV()
{
	return toNumPyArray(mEnvs[0]->GetMaxV());
}

using namespace boost::python;

BOOST_PYTHON_MODULE(pymss)
{
	Py_Initialize();
	np::initialize();

	class_<EnvManager>("EnvManager",init<std::string,int>())
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
		.def("GetRewards",&EnvManager::GetRewards)
		.def("SetAction",&EnvManager::SetAction)
		.def("SetActions",&EnvManager::SetActions)
		.def("SetActions_Device",&EnvManager::SetActions_Device)
		.def("SetActivationLevels",&EnvManager::SetActivationLevels)
		.def("IsEndOfEpisode",&EnvManager::IsEndOfEpisode)
		.def("IsEndOfEpisodes",&EnvManager::IsEndOfEpisodes)
		.def("GetNumState",&EnvManager::GetNumState)
		.def("GetNumState_Device",&EnvManager::GetNumState_Device)
		.def("GetNumAction",&EnvManager::GetNumAction)
		.def("GetNumAction_Device",&EnvManager::GetNumAction_Device)
		.def("GetNumSteps",&EnvManager::GetNumSteps)
		.def("GetControlHz",&EnvManager::GetControlHz)
		.def("GetSimulationHz",&EnvManager::GetSimulationHz)
		.def("UseMuscle",&EnvManager::UseMuscle)
		.def("UseDevice",&EnvManager::UseDevice)
		.def("UseDeviceNN",&EnvManager::UseDeviceNN)
		.def("SetDesiredTorques",&EnvManager::SetDesiredTorques)
		.def("GetDesiredTorques",&EnvManager::GetDesiredTorques)
		.def("GetNumTotalMuscleRelatedDofs",&EnvManager::GetNumTotalMuscleRelatedDofs)
		.def("GetNumMuscles",&EnvManager::GetNumMuscles)
		.def("GetMuscleTorques",&EnvManager::GetMuscleTorques)
		.def("GetMuscleTuples",&EnvManager::GetMuscleTuples)
		.def("UseAdaptiveSampling", &EnvManager::UseAdaptiveSampling)
		.def("SetParamState", &EnvManager::SetParamState)
		.def("GetNumParamState", &EnvManager::GetNumParamState)
		.def("GetMinV", &EnvManager::GetMinV)
		.def("GetMaxV", &EnvManager::GetMaxV);
}

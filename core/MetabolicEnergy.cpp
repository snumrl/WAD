#include "MetabolicEnergy.h"
#include "Muscle.h"

using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

MetabolicEnergy::
MetabolicEnergy(const dart::simulation::WorldPtr& wPtr)
{
	mWorld = wPtr;
	mLowerBody = false;
}

MetabolicEnergy::
~MetabolicEnergy()
{

}

void
MetabolicEnergy::
Initialize(const std::vector<Muscle*>& muscles, double m, int steps, int frames, double ratio)
{
	mNumSteps = steps;
	mCycleFrames = frames;
	mMassRatio = ratio;
	if(mMassRatio != 1.0)
		mLowerBody = true;
	isFirst = true;

	mMass = m * mMassRatio;

	BHAR04 = 0.0;
	HOUD06 = 0.0;

	BHAR04_deque = std::deque<double>(mCycleFrames);
	HOUD06_deque = std::deque<double>(mCycleFrames);

	for(const auto& m : muscles){
		std::string name = m->GetName();
		BHAR04_deque_map[name] = std::deque<double>(mCycleFrames);
		HOUD06_deque_map[name] = std::deque<double>(mCycleFrames);
		BHAR04_cur_map[name] = 0.0;
		HOUD06_cur_map[name] = 0.0;
		BHAR04_tmp_map[name] = 0.0;
		HOUD06_tmp_map[name] = 0.0;
	}
}

void
MetabolicEnergy::
Reset()
{
	BHAR04 = 0.0;
	HOUD06 = 0.0;
	avgVel.setZero();

	curStep = 0;

	std::fill(BHAR04_deque.begin(), BHAR04_deque.end(), 0.0);
	std::fill(HOUD06_deque.begin(), HOUD06_deque.end(), 0.0);

	for(auto iter = BHAR04_deque_map.begin(); iter != BHAR04_deque_map.end(); iter++)
		std::fill(iter->second.begin(), iter->second.end(), 0.0);
	for(auto iter = HOUD06_deque_map.begin(); iter != HOUD06_deque_map.end(); iter++)
		std::fill(iter->second.begin(), iter->second.end(), 0.0);
	for(auto iter = BHAR04_cur_map.begin(); iter != BHAR04_cur_map.end(); iter++)
		iter->second = 0.0;
	for(auto iter = HOUD06_cur_map.begin(); iter != HOUD06_cur_map.end(); iter++)
		iter->second = 0.0;
}

void
MetabolicEnergy::
Set(const std::vector<Muscle*>& muscles, Eigen::Vector3d vel, double phase)
{
	if(isFirst == true){
		this->Reset();
		isFirst = false;
	}
	curStep++;

	for(const auto& m : muscles){
		std::string name = m->GetName();
		double curBHAR04 = m->GetMetabolicEnergyRate_BHAR04();
		double curHOUD06 = m->GetMetabolicEnergyRate_HOUD06();

		BHAR04_tmp_map[name] += curBHAR04;
		HOUD06_tmp_map[name] += curHOUD06;

		// dE += m->GetMetabolicEnergyRate();
		// h_A += m->Geth_A();
		// h_M += m->Geth_M();
		// h_SL += m->Geth_SL();
		// W += m->GetW();
	}

	avgVel += vel;

	if(curStep == mNumSteps-1)
	{
		BHAR04 = 0.0;
		HOUD06 = 0.0;
		for(const auto& m : muscles){
			std::string name = m->GetName();
			double cBHAR04 = BHAR04_tmp_map[name] / mNumSteps;
			BHAR04 += cBHAR04;
			BHAR04_deque_map[name].pop_back();
			BHAR04_deque_map[name].push_front(cBHAR04);

			double cHOUD06 = HOUD06_tmp_map[name] / mNumSteps;
			HOUD06 += cHOUD06;
			HOUD06_deque_map[name].pop_back();
			HOUD06_deque_map[name].push_front(cHOUD06);

			BHAR04_tmp_map[name] = 0.0;
			HOUD06_tmp_map[name] = 0.0;
		}

		double vel_norm = (avgVel.norm())/(double)mNumSteps;

		double dB = 1.51 * mMass;
		// double dB = 0.0;

		BHAR04 += dB;
		BHAR04 = BHAR04 / (mMass*vel_norm);

		BHAR04_deque.pop_back();
		BHAR04_deque.push_front(BHAR04);

		HOUD06 += dB;
		HOUD06 = HOUD06 / (mMass*vel_norm);

		HOUD06_deque.pop_back();
		HOUD06_deque.push_front(HOUD06);

		avgVel.setZero();
		BHAR04 = 0;
		HOUD06 = 0;

		curStep = 0;

		if(phase*mCycleFrames >= mCycleFrames-1){
			cumBHAR04 = 0, cumHOUD06 = 0;
			for(int i=0; i<BHAR04_deque.size(); i++)
				cumBHAR04 += BHAR04_deque[i];
			for(int i=0; i<HOUD06_deque.size(); i++){
				std::cout << i << " : " << HOUD06_deque[i] << std::endl;
				cumHOUD06 += HOUD06_deque[i];
			}

			isFirst = true;
		}
	}


}

double
MetabolicEnergy::
GetReward()
{
	double err_scale = 1.0;
	double metabolic_scale = 0.05;
	double metabolic_err = 0.0;

	metabolic_err = cumHOUD06/(double)mCycleFrames;
	double reward = exp(-err_scale * metabolic_scale * metabolic_err);
	return reward;
}

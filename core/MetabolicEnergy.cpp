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

	mMass = m * mMassRatio;

	isFirst = true;
	isTotalFirst = true;
	curStep = 0;

	BHAR04 = 0.0;
	HOUD06 = 0.0;

	BHAR04_deque = std::deque<double>(mCycleFrames);
	HOUD06_deque = std::deque<double>(mCycleFrames);

	HOUD06_ByFrame = std::vector<std::vector<double>>(mCycleFrames);

	for(const auto& m : muscles){
		std::string name = this->GetCoreName(m->GetName());
		mNameSet.insert(name);

		BHAR04_map_deque[name] = std::deque<double>(mCycleFrames);
		HOUD06_map_deque[name] = std::deque<double>(mCycleFrames);

		BHAR04_map_tmp[name] = 0.0;
		HOUD06_map_tmp[name] = 0.0;

		HOUD06_mapByFrame[name] = std::vector<std::vector<double>>(mCycleFrames);
	}
}

void
MetabolicEnergy::
Reset()
{
	isTotalFirst = true;

	for(auto iter = HOUD06_mapByFrame.begin(); iter != HOUD06_mapByFrame.end(); iter++){
        for(int i = 0; i != (iter->second).size(); i++)
            (iter->second).at(i) = std::vector<double>();
    }

    for(int i=0; i<HOUD06_ByFrame.size(); i++)
    	HOUD06_ByFrame.at(i) = std::vector<double>();

	this->ResetCycle();
}

void
MetabolicEnergy::
ResetCycle()
{
	vel_tmp.setZero();
	curStep = 0;
}

void
MetabolicEnergy::
Set(const std::vector<Muscle*>& muscles, Eigen::Vector3d vel, double phase, int frame)
{
	if(isFirst == true){
		this->ResetCycle();
		isFirst = false;
	}
	curStep++;

	for(const auto& m : muscles){
		std::string name = this->GetCoreName(m->GetName());

		BHAR04_map_tmp[name] += m->GetMetabolicEnergyRate_BHAR04();
		HOUD06_map_tmp[name] += m->GetMetabolicEnergyRate_HOUD06();
	}

	vel_tmp += vel;

	if(curStep == mNumSteps-1)
	{
		BHAR04 = 0.0;
		HOUD06 = 0.0;
		for(const auto& name : mNameSet){
			double BHAR04_avg = BHAR04_map_tmp[name]/(double)mNumSteps;
			BHAR04 += BHAR04_avg;
			BHAR04_map_deque[name].pop_back();
			BHAR04_map_deque[name].push_front(BHAR04_avg);

			double HOUD06_avg = HOUD06_map_tmp[name]/(double)mNumSteps;
			HOUD06 += HOUD06_avg;
			HOUD06_map_deque[name].pop_back();
			HOUD06_map_deque[name].push_front(HOUD06_avg);

			HOUD06_mapByFrame[name].at(frame).push_back(HOUD06_avg);

			BHAR04_map_tmp[name] = 0.0;
			HOUD06_map_tmp[name] = 0.0;
		}

		double vel_avg = (vel_tmp.norm())/(double)mNumSteps;
		double dB = 0.0;
		// double dB = 1.51 * mMass;

		BHAR04 += dB;
		BHAR04 = BHAR04/(mMass*vel_avg);
		BHAR04_deque.pop_back();
		BHAR04_deque.push_front(BHAR04);

		HOUD06 += dB;
		HOUD06 = HOUD06/(mMass*vel_avg);
		HOUD06_deque.pop_back();
		HOUD06_deque.push_front(HOUD06);

		HOUD06_ByFrame.at(frame).push_back(HOUD06);

		if(phase*mCycleFrames >= mCycleFrames-1)
		{
			BHAR04_cum = 0;
			HOUD06_cum = 0;

			if(isTotalFirst){
				isTotalFirst = false;
			}
			else{
				for(int i=0; i<BHAR04_deque.size(); i++)
					BHAR04_cum += BHAR04_deque[i];

				for(int i=0; i<HOUD06_deque.size(); i++)
					HOUD06_cum += HOUD06_deque[i];
			}
		}

		isFirst = true;
	}
}

double
MetabolicEnergy::
GetReward()
{
	double err_scale = 1.0;
	double metabolic_scale = 0.05;
	double metabolic_err = 0.0;

	metabolic_err = HOUD06_cum/(double)mCycleFrames;
	double reward = exp(-err_scale * metabolic_scale * metabolic_err);

	if(isTotalFirst)
		reward = 0;
	return reward;
}

std::string
MetabolicEnergy::
GetCoreName(std::string name)
{
	std::string coreName;

	int nSize = name.size();
	if(name[nSize-1]>=48 && name[nSize-1]<=57)
		coreName = name.substr(0,nSize-1);
	else{
		coreName = name.substr(0,nSize);
	}

	return coreName;
}

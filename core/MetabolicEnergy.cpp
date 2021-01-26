#include "MetabolicEnergy.h"
#include "Muscle.h"

using namespace dart;
using namespace dart::dynamics;
using namespace MASS;

MetabolicEnergy::
MetabolicEnergy(const dart::simulation::WorldPtr& wPtr)
{
	mWorld = wPtr;
}

MetabolicEnergy::
~MetabolicEnergy()
{

}

void
MetabolicEnergy::
Initialize(const std::vector<Muscle*>& muscles)
{
	windowSize = 200;

	BHAR04 = 0.0;
	HOUD06 = 0.0;

	BHAR04_deque = std::deque<double>(windowSize);
	HOUD06_deque = std::deque<double>(windowSize);

	for(const auto& m : muscles){
		std::string name = m->GetName();
		BHAR04_deque_map[name] = std::deque<double>(windowSize);
		HOUD06_deque_map[name] = std::deque<double>(windowSize);
		BHAR04_cur_map[name] = 0.0;
		HOUD06_cur_map[name] = 0.0;
	}
}

void
MetabolicEnergy::
Reset()
{
	BHAR04 = 0.0;
	HOUD06 = 0.0;

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
Set(const std::vector<Muscle*>& muscles, double mass, double vel)
{
	BHAR04 = 0.0;
	HOUD06 = 0.0;

	for(const auto& m : muscles){
		std::string name = m->GetName();
		double curBHAR04 = m->GetMetabolicEnergyRate_BHAR04();
		double curHOUD06 = m->GetMetabolicEnergyRate_HOUD06();

		BHAR04 += curBHAR04;
		HOUD06 += curHOUD06;

		BHAR04_cur_map[name] = curBHAR04;
		BHAR04_deque_map[name].pop_back();
		BHAR04_deque_map[name].push_front(curBHAR04);

		HOUD06_cur_map[name] = curHOUD06;
		HOUD06_deque_map[name].pop_back();
		HOUD06_deque_map[name].push_front(curHOUD06);
	}

	double dB = 1.51 * mass;

	BHAR04 += dB;
	BHAR04 = BHAR04 / (mass*vel);

	BHAR04_deque.pop_back();
	BHAR04_deque.push_front(BHAR04);

	HOUD06 += dB;
	HOUD06 = HOUD06 / (mass*vel);

	HOUD06_deque.pop_back();
	HOUD06_deque.push_front(HOUD06);
}

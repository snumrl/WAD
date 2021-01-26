#ifndef __MASS_METABOLIC_ENERGY_H__
#define __MASS_METABOLIC_ENERGY_H__
#include "dart/dart.hpp"
#include <deque>
#include <map>

namespace MASS
{

class BVH;
class Muscle;
class Character;
class MetabolicEnergy
{
public:
    MetabolicEnergy(const dart::simulation::WorldPtr& wPtr);
    // MetabolicEnergy(const dart::simulation::WorldPtr& wPtr);
    ~MetabolicEnergy();

    void Initialize(const std::vector<Muscle*>& muscles);
    void Reset();
    void Set(const std::vector<Muscle*>& muscles, double mass, double vel);

    double GetWindowSize(){return windowSize;}

    const double& GetBHAR04(){return BHAR04;}
    const double& GetHOUD06(){return HOUD06;}

    const std::deque<double>& GetBHAR04_deque(){return BHAR04_deque;}
    const std::deque<double>& GetHOUD06_deque(){return HOUD06_deque;}

    const std::map<std::string, std::deque<double>>& GetBHAR04_deque_map(){return BHAR04_deque_map;}
    const std::map<std::string, std::deque<double>>& GetHOUD06_deque_map(){return HOUD06_deque_map;}

    const std::map<std::string, double>& GetBHAR04_cur_map(){return BHAR04_cur_map;}
    const std::map<std::string, double>& GetHOUD06_cur_map(){return HOUD06_cur_map;}

private:
    dart::simulation::WorldPtr mWorld;
    // Character* mCharacter;

    double windowSize;
    double BHAR04, HOUD06;
    std::deque<double> BHAR04_deque, HOUD06_deque;
    std::map<std::string, std::deque<double>> BHAR04_deque_map, HOUD06_deque_map;
    std::map<std::string, double> BHAR04_cur_map, HOUD06_cur_map;
};

};

#endif

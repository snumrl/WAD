#ifndef __MASS_METABOLIC_ENERGY_H__
#define __MASS_METABOLIC_ENERGY_H__
#include "dart/dart.hpp"
#include <deque>
#include <map>
#include <set>

namespace MASS
{

class BVH;
class Muscle;
class Character;
class MetabolicEnergy
{
public:
    MetabolicEnergy(const dart::simulation::WorldPtr& wPtr);
    ~MetabolicEnergy();

    void Initialize(const std::vector<Muscle*>& muscles, double m, int steps, int frames, double ratio);
    void Reset();
    void ResetCycle();
    void Set(const std::vector<Muscle*>& muscles, double vel, double phase);
    void Set(const std::vector<Muscle*>& muscles, Eigen::Vector3d vel, double phase, int frame);

    double GetCycleFramse(){return mCycleFrames;}
    double GetReward();
    void SetMass(double m){mMass = m;}

    std::string GetCoreName(std::string name);

    const double& GetBHAR04(){return BHAR04;}
    const double& GetHOUD06(){return HOUD06;}

    const std::deque<double>& GetBHAR04_deque(){return BHAR04_deque;}
    const std::deque<double>& GetHOUD06_deque(){return HOUD06_deque;}

    const std::map<std::string, std::deque<double>>& GetBHAR04_map_deque(){return BHAR04_map_deque;}
    const std::map<std::string, std::deque<double>>& GetHOUD06_map_deque(){return HOUD06_map_deque;}

    const std::vector<std::vector<double>>& GetHOUD06_ByFrame(){return HOUD06_ByFrame;}
    const std::map<std::string, std::vector<std::vector<double>>>& GetHOUD06_mapByFrame(){return HOUD06_mapByFrame;}

private:
    dart::simulation::WorldPtr mWorld;

    bool isFirst;
    bool isTotalFirst;
    bool mLowerBody;

    int curStep;
    int mNumSteps;
    int mNumFrames;
    int mCycleFrames;

    double mMass;
    double mReward;
    double mMassRatio;

    Eigen::Vector3d vel_tmp;
    std::set<std::string> mNameSet;

    double BHAR04;
    double BHAR04_cum;
    std::deque<double> BHAR04_deque;
    std::map<std::string, double> BHAR04_map_tmp;
    std::map<std::string, std::deque<double>> BHAR04_map_deque;

    double HOUD06;
    double HOUD06_cum;
    std::deque<double> HOUD06_deque;
    std::map<std::string, double> HOUD06_map_tmp;
    std::map<std::string, std::deque<double>> HOUD06_map_deque;

    std::vector<std::vector<double>> HOUD06_ByFrame;
    std::map<std::string, std::vector<std::vector<double>>> HOUD06_mapByFrame;
};

};

#endif

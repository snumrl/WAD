#ifndef __ANALYSIS_DATA_H__
#define __ANALYSIS_DATA_H__
#include "dart/dart.hpp"
#include <deque>
#include <map>

using namespace dart::dynamics;
namespace WAD
{

class AnalysisData
{
public:
    AnalysisData();
    ~AnalysisData();

    void Initialize();
    const std::map<std::string, std::deque<double>>& GetRealJointData(){return mRealJointData;}
    
private:
    std::map<std::string, std::deque<double>> mRealJointData;    
};

}
#endif

#ifndef __CONTACT_H__
#define __CONTACT_H__

#include "dart/dart.hpp"
#include <deque>
#include <map>

using namespace dart::dynamics;
using namespace dart::simulation;
namespace WAD
{

class Contact
{
public:
    Contact(const WorldPtr& wPtr);
    ~Contact();

    void Initialize(std::string name, BodyNode* bn);
    void Set();
    void Reset();

    bool isContact();

    // void Initialize(const SkeletonPtr& skel, double m, int steps);
    // void Clear();
    // double GetReward();

    // void SetMass(double m){mMass = m;}
    // void Set();
    // void SetContactObject(std::string name);
    // void SetContactForce(std::string name, double f);
    // void SetContactForces(std::string name, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> forces);
    // void AddContactForce(std::string name, Eigen::Vector3d f, Eigen::Vector3d p);

    // double GetContactForce(std::string name);
    // const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& GetContactForces(std::string name);
    // std::map<std::string, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>>& GetContactObjects(){return mContactObjects;}
    // const std::deque<double>& GetContactForceDeque(std::string name){return mContactForceDequeMap[name];}

private:
    WorldPtr mWorld;
    BodyNode* mBodyNode;
    std::string mName;

    bool mContact;
    // SkeletonPtr mSkeleton;
    // int mWindowSize;
    // int mNumSteps
    ;
    // double mMass;
    // std::map<std::string, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>> mContactObjects;
    // std::map<std::string, double> mContactForces;
    // std::map<std::string, std::deque<double>> mContactForceDequeMap;
};

}

#endif

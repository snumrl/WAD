#ifndef __MASS_BVH_H__
#define __MASS_BVH_H__

#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "Utils.h"
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <initializer_list>

using namespace dart::dynamics;
namespace MASS
{

class BVHNode
{
public:
	enum CHANNEL{
		Xpos=0, Ypos=1, Zpos=2, Xrot=3, Yrot=4, Zrot=5
	};

public:
	std::map<std::string, CHANNEL> CHANNEL_NAME;

	BVHNode(const std::string& name,BVHNode* parent);
	void SetChannel(int c_offset,std::vector<std::string>& c_name);
	void Set(const Eigen::VectorXd& m_t);
	void Set(const Eigen::Matrix3d& R_t);
	void SetOffset(double x, double y, double z);
	const Eigen::Matrix3d& Get();

	void AddChild(BVHNode* child);
	std::vector<BVHNode*>& GetChildren(){return mChildren;}
	BVHNode* GetNode(const std::string& name);
	int GetChannelOffset(){return mChannelOffset;}
	int GetNumChannels(){return mNumChannels;}
	Eigen::Vector3d GetOffset(){return mOffset;}

private:
	BVHNode* mParent;
	std::vector<BVHNode*> mChildren;

	Eigen::Matrix3d mR;
	Eigen::Vector3d mOffset;
	std::string mName;

	int mChannelOffset;
	int mNumChannels;
	std::vector<BVHNode::CHANNEL> mChannel;
};

class BVH
{
public:
	BVH(const SkeletonPtr& skel,const std::map<std::string,std::string>& bvh_map);

	BVHNode* ReadHierarchy(BVHNode* parent,const std::string& name,int& channel_offset,std::ifstream& is);
	void Parse(const std::string& file, bool cyclic=true);

	Eigen::Matrix3d Get(const std::string& bvh_node);
	const Eigen::VectorXd& GetMotion(int k);
	Eigen::VectorXd GetMotionVel(int k);
	
	void SetSkeleton(const SkeletonPtr& skel){mSkeleton = skel;}
	void SetBVHMap(const std::map<std::string,std::string>& bvh_map){mBVHMap = bvh_map;}
	
	void SetMotionTransform();
	void SetMotionFrames();
	void SetMotionVelFrames();

	void SetMotionFramesNonCyclic(int frames, bool blend);
	void SetMotionVelFramesNonCyclic(int frames, bool blend);

	void Draw();
	void DrawRecursive(BVHNode* node);

	int GetNumTotalFrames(){return mNumTotalFrames;}
	double GetMaxTime() {return (mNumTotalFrames)*mTimeStep;}
	double GetTimeStep(){return mTimeStep;}

	const SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::map<std::string,std::string>& GetBVHMap(){return mBVHMap;}
	const Eigen::Isometry3d& GetT0(){return T0;}
	const Eigen::Isometry3d& GetT1(){return T1;}
	bool IsCyclic(){return mCyclic;}
	Eigen::VectorXd GetCycleOffset(){return mCycleOffset;}
	bool IsParsed(){return mParse;}
	void SetParsed(bool p){mParse = p;}

private:
	bool mParse;
	bool mCyclic;
	int mNumTotalChannels;
	int mNumTotalFrames;
	double mTimeStep;
	
	BVHNode* mRoot;
	SkeletonPtr mSkeleton;

	std::map<std::string, BVHNode*> mMap;
	std::map<std::string,std::string> mBVHMap;

	Eigen::Isometry3d T0,T1;
	std::vector<Eigen::VectorXd> mMotions;
	std::vector<Eigen::VectorXd> mMotionFrames;
	std::vector<Eigen::VectorXd> mMotionFramesNonCyclic;
	std::vector<Eigen::VectorXd> mMotionFramesNonCyclicTmp;
	Eigen::MatrixXd mMotionVelFrames;
	Eigen::MatrixXd mMotionVelFramesNonCyclic;
	Eigen::Vector3d mCycleOffset;
};
}
#endif

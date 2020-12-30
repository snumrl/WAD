#include "BVH.h"
#include "Utils.h"
#include <iostream>
#include <Eigen/Geometry>
#include "dart/dart.hpp"

using namespace dart::dynamics;
namespace MASS
{
Eigen::Matrix3d
R_x(double x)
{
	double cosa = cos(x*3.141592/180.0);
	double sina = sin(x*3.141592/180.0);
	Eigen::Matrix3d R;
	R<<	1,0		,0	  ,
		0,cosa	,-sina,
		0,sina	,cosa ;
	return R;
}

Eigen::Matrix3d R_y(double y)
{
	double cosa = cos(y*3.141592/180.0);
	double sina = sin(y*3.141592/180.0);
	Eigen::Matrix3d R;
	R <<cosa ,0,sina,
		0    ,1,   0,
		-sina,0,cosa;
	return R;
}

Eigen::Matrix3d R_z(double z)
{
	double cosa = cos(z*3.141592/180.0);
	double sina = sin(z*3.141592/180.0);
	Eigen::Matrix3d R;
	R<<	cosa,-sina,0,
		sina,cosa ,0,
		0   ,0    ,1;
	return R;
}

BVHNode::
BVHNode(const std::string& name,BVHNode* parent)
	:mParent(parent),mName(name),mChannelOffset(0),mNumChannels(0)
{
}

void
BVHNode::
SetChannel(int c_offset,std::vector<std::string>& c_name)
{
	mChannelOffset = c_offset;
	mNumChannels = c_name.size();
	for(const auto& cn : c_name)
		mChannel.push_back(CHANNEL_NAME[cn]);
}

void
BVHNode::
Set(const Eigen::VectorXd& m_t)
{
	mR.setIdentity();

	for(int i=0;i<mNumChannels;i++)
	{
		switch(mChannel[i])
		{
		case Xpos:break;
		case Ypos:break;
		case Zpos:break;
		case Xrot:mR = mR*R_x(m_t[mChannelOffset+i]);break;
		case Yrot:mR = mR*R_y(m_t[mChannelOffset+i]);break;
		case Zrot:mR = mR*R_z(m_t[mChannelOffset+i]);break;
		default:break;
		}
	}

}

void
BVHNode::
Set(const Eigen::Matrix3d& R_t)
{
	mR = R_t;
}

void
BVHNode::
SetOffset(double x, double y, double z)
{
	mOffset[0] = x;
	mOffset[1] = y;
	mOffset[2] = z;
}

Eigen::Matrix3d
BVHNode::
Get()
{
	return mR;
}

void
BVHNode::
AddChild(BVHNode* child)
{
	mChildren.push_back(child);
}

BVHNode*
BVHNode::
GetNode(const std::string& name)
{
	if(!mName.compare(name))
		return this;

	for(auto& c : mChildren)
	{
		BVHNode* bn = c->GetNode(name);
		if(bn!=nullptr)
			return bn;
	}

	return nullptr;
}

BVH::
BVH(const dart::dynamics::SkeletonPtr& skel,const std::map<std::string,std::string>& bvh_map)
	:mSkeleton(skel),mBVHMap(bvh_map),mCyclic(true),mParse(false)
{
	mSpeedRatio = 1.0;
}

BVHNode*
BVH::
ReadHierarchy(BVHNode* parent,const std::string& name,int& channel_offset,std::ifstream& is)
{
	char buffer[256];
	double offset[3];
	std::vector<std::string> c_name;

	BVHNode* new_node = new BVHNode(name,parent);
	mMap.insert(std::make_pair(name,new_node));

	is>>buffer; //{

	while(is>>buffer)
	{
		if(!strcmp(buffer,"}"))
			break;
		if(!strcmp(buffer,"OFFSET"))
		{
			//Ignore
			double x,y,z;

			is>>x;
			is>>y;
			is>>z;
			new_node->SetOffset(6.0*x,6.0*y,6.0*z);
			// new_node->SetOffset(100*x,100*y,100*z);
		}
		else if(!strcmp(buffer,"CHANNELS"))
		{

			is>>buffer;
			int n;
			n= atoi(buffer);

			for(int i=0;i<n;i++)
			{
				is>>buffer;
				c_name.push_back(std::string(buffer));
			}

			new_node->SetChannel(channel_offset,c_name);
			channel_offset += n;
		}
		else if(!strcmp(buffer,"JOINT"))
		{
			is>>buffer;
			BVHNode* child = ReadHierarchy(new_node,std::string(buffer),channel_offset,is);
			new_node->AddChild(child);
		}
		else if(!strcmp(buffer,"End"))
		{
			is>>buffer;
			BVHNode* child = ReadHierarchy(new_node,std::string("EndEffector"),channel_offset,is);
			new_node->AddChild(child);
		}
	}

	return new_node;
}

void
BVH::
Parse(const std::string& file, bool cyclic)
{
	mParse = true;
	mCyclic = cyclic;
	std::ifstream is(file);

	char buffer[256];

	if(!is)
	{
		std::cout<<"Can't Open BVH File"<<std::endl;
		return;
	}

	while(is>>buffer)
	{
		if(!strcmp(buffer,"HIERARCHY"))
		{
			is>>buffer;//Root
			is>>buffer;//Name
			int c_offset = 0;
			mRoot = ReadHierarchy(nullptr,buffer,c_offset,is);
			mNumTotalChannels = c_offset;
		}
		else if(!strcmp(buffer,"MOTION"))
		{
			is>>buffer; //Frames:
			is>>buffer; //num_frames
			mNumTotalFrames = atoi(buffer);
			is>>buffer; //Frame
			is>>buffer; //Time:
			is>>buffer; //time step
			mTimeStep = atof(buffer);
			mTimeStep = 0.0333;
			mMotions.resize(mNumTotalFrames/4+1);
			for(auto& m_t : mMotions)
				m_t = Eigen::VectorXd::Zero(mNumTotalChannels);

			double val;
			for(int i=0; i<mNumTotalFrames; i++)
			{
				for(int j=0; j<mNumTotalChannels; j++)
				{
					is>>val;
					if(i%4==0)
						mMotions[i/4][j] = val;

					if(i==mNumTotalFrames-1)
						mMotions[i/4+1][j] = val;
				}
			}
			mNumTotalFrames = mNumTotalFrames/4+1;
		}
	}
	is.close();

	if(mCyclic)
	{
		this->SetMotionTransform();
		this->SetMotionFrames();
		this->SetMotionVelFrames();
	}
	else{
		this->SetMotionTransform();
		this->SetMotionFrames();
		this->SetMotionFramesNonCyclic(1000, true);
		this->SetMotionVelFramesNonCyclic(1000, true);
	}
}

Eigen::Matrix3d
BVH::
Get(const std::string& bvh_node)
{
	return mMap[bvh_node]->Get();
}

void
BVH::
SetMotionTransform()
{
	BodyNode* root = mSkeleton->getRootBodyNode();
	std::string root_bvh_name = mBVHMap[root->getName()];
	Eigen::VectorXd mData = mMotions[0];

	mData.segment<3>(3) *= mSpeedRatio;
	mMap[root_bvh_name]->Set(mData);
	T0.linear() = this->Get(root_bvh_name);
	T0.translation() = 0.01*mData.segment<3>(0);

	Eigen::VectorXd mDataLast = mMotions[mNumTotalFrames-1];

	mDataLast.segment<3>(3) *= mSpeedRatio;
	mMap[root_bvh_name]->Set(mDataLast);
	T1.linear() = this->Get(root_bvh_name);
	T1.translation() = 0.01*(mSpeedRatio*mDataLast.segment<3>(0) + (1.0-mSpeedRatio)*mData.segment<3>(0));

	mCycleOffset = T1.translation() + 0.01*(mSpeedRatio)*(mMotions[1]-mMotions[0]).segment<3>(0) - T0.translation();
}

void
BVH::
SetMotionFrames()
{
	int dof = mSkeleton->getNumDofs();
	mMotionFrames.resize(mNumTotalFrames);
	mMotionFramesNonCyclic.resize(1000);
	for(int i=0; i<mNumTotalFrames; i++)
	{
		Eigen::VectorXd m_t = mMotions[i];
		for(auto& bn: mMap)
			bn.second->Set(m_t);

		Eigen::VectorXd p = Eigen::VectorXd::Zero(dof);
		for(auto ss : mBVHMap)
		{
			BodyNode* bn = mSkeleton->getBodyNode(ss.first);
			Eigen::Matrix3d R = this->Get(ss.second);
			Joint* jn = bn->getParentJoint();

			int idx = jn->getIndexInSkeleton(0);
			std::string jointName = jn->getName();
			if(jn->getType()=="FreeJoint")
			{
				Eigen::Isometry3d T;
				T.translation() = 0.01*m_t.segment<3>(0);
				T.linear() = R;
				p.segment<6>(idx) = FreeJoint::convertToPositions(T);

				if(jointName == "Pelvis"){
					p[idx]   *= mSpeedRatio;
					p[idx+1] *= mSpeedRatio;
					p[idx+2] *= mSpeedRatio;
					p[idx+3] *= mSpeedRatio;
					p[idx+4] *= 1.0;
					p[idx+5] *= mSpeedRatio;
					if(p[idx+2] > 0)
						p[idx+2]   *= 0.8;
					else
						p[idx+2]   *= 0.4;
				}
			}
			else if(jn->getType()=="BallJoint"){
				p.segment<3>(idx) = BallJoint::convertToPositions(R);

				if(jointName == "Spine"){
					p[idx] -= 0.06;
					// p[idx] += 0.25;
				}

				if(jointName == "Torso"){
					// p[idx] += 0.3;
					if(p[idx+2] > 0)
						p[idx+2] *= 0.3;
				}

				// if(jointName == "ShoulderL" || jointName == "ShoulderR")
				// 	p[idx] -= 0.3;

				if(jointName == "ArmL")
					p[idx+2] -= 0.1;

				if(jointName == "ArmR")
					p[idx+2] += 0.1;

				if(jointName == "FemurL" || jointName == "FemurR"){
					p[idx]   *= mSpeedRatio;
					p[idx+1] *= mSpeedRatio;
					p[idx+2] *= mSpeedRatio;
					if(p[idx+2] > 0)
						p[idx+2] *= 0.5;
				}

				if(jointName == "TalusL" || jointName == "TalusR"){
					p[idx]   *= mSpeedRatio;
					p[idx+1] *= mSpeedRatio;
					p[idx+2] *= mSpeedRatio;
				}
			}
			else if(jn->getType()=="RevoluteJoint")
			{
				Eigen::Vector3d u = dynamic_cast<RevoluteJoint*>(jn)->getAxis();
				Eigen::Vector3d aa = BallJoint::convertToPositions(R);
				double val;
				if((u-Eigen::Vector3d::UnitX()).norm()<1E-4)
					val = aa[0];
				else if((u-Eigen::Vector3d::UnitY()).norm()<1E-4)
					val = aa[1];
				else
					val = aa[2];

				if(val>M_PI)
					val -= 2*M_PI;
				else if(val<-M_PI)
					val += 2*M_PI;

				p[idx] = val;

				if(jointName == "ForeArmL" || jointName == "ForeArmR")
					p[idx] *= mSpeedRatio;

				if(jointName == "TibiaL" || jointName == "TibiaR")
					p[idx] *= mSpeedRatio;

				if(jointName == "FootThumbL" || jointName == "FootThumbR")
					p[idx] *= mSpeedRatio;

				if(jointName == "FootPinkyL" || jointName == "FootPinkyR")
					p[idx] *= mSpeedRatio;
			}
		}
		mMotionFrames[i] = p;
	}
}

void
BVH::
SetMotionFramesNonCyclic(int frames, bool blend)
{
	mSkeleton->setPositions(mMotionFrames[0]);
	mSkeleton->computeForwardKinematics(true,false,false);

	Eigen::Vector3d p0_footl = mSkeleton->getBodyNode("TalusL")->getWorldTransform().translation();
	Eigen::Vector3d p0_footr = mSkeleton->getBodyNode("TalusR")->getWorldTransform().translation();

	Eigen::Isometry3d T0_phase = dart::dynamics::FreeJoint::convertToTransform(mMotionFrames[0].head<6>());
	Eigen::Isometry3d T1_phase = dart::dynamics::FreeJoint::convertToTransform(mMotionFrames.back().head<6>());

	Eigen::Isometry3d T0_nc = T0_phase;
	Eigen::Isometry3d T01 = T1_phase*T0_phase.inverse();
	Eigen::Vector3d p01 = dart::math::logMap(T01.linear());
	T01.linear() = dart::math::expMapRot(Utils::projectToXZ(p01));
	T01.translation()[1] = 0;

	int totalLength = mMotionFrames.size();
	int smooth_time = 15;
	for(int i = 0; i < frames; i++) {
		int phase = i % totalLength;
		Eigen::VectorXd newMotion;
		if(i < totalLength)
		{
			newMotion = mMotionFrames[i];
			mMotionFramesNonCyclic.at(i) = newMotion;
		}
		else
		{
			Eigen::VectorXd pos;
			if(phase == 0)
			{
				std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>> constraints;

				mSkeleton->setPositions(mMotionFramesNonCyclic.at(i-1));
				mSkeleton->computeForwardKinematics(true,false,false);

				Eigen::Vector3d p_footl = mSkeleton->getBodyNode("TalusL")->getWorldTransform().translation();
				Eigen::Vector3d p_footr = mSkeleton->getBodyNode("TalusR")->getWorldTransform().translation();

				constraints.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>("TalusL", p_footl, Eigen::Vector3d(0, 0, 0)));
				constraints.push_back(std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>("TalusR", p_footr, Eigen::Vector3d(0, 0, 0)));

				Eigen::VectorXd p = mMotionFrames[phase];
				p.segment<3>(3) = mMotionFramesNonCyclic.at(i-3).segment<3>(3);
				mSkeleton->setPositions(p);
				mSkeleton->computeForwardKinematics(true,false,false);

				//// rotate "root" to seamlessly stitch foot
				pos = Utils::solveMCIKRoot(mSkeleton, constraints);
				pos[4] -= 0.006;
				T0_nc = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
			}
			else
			{
				pos = mMotionFrames[phase];
				Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
				Eigen::Isometry3d T0_phase_nc = T0_nc * T0_phase.inverse();

				if(phase < smooth_time)
				{
					Eigen::Quaterniond Q0_phase_nc(T0_phase_nc.linear());
					double slerp_t = (double)phase/smooth_time;
					slerp_t = 0.5*(1-cos(M_PI*slerp_t)); //smooth slerp t [0,1]

					Eigen::Quaterniond Q_blend = Q0_phase_nc.slerp(slerp_t, Eigen::Quaterniond::Identity());
					T0_phase_nc.linear() = Eigen::Matrix3d(Q_blend);
					T_current = T0_phase_nc* T_current;
				}
				else
				{
					T0_phase_nc.linear() = Eigen::Matrix3d::Identity();
					T_current = T0_phase_nc* T_current;
				}

				pos.head<6>() = dart::dynamics::FreeJoint::convertToPositions(T_current);
				pos[28] -= phase * 0.002;
				pos[4] += 0.002;
				if(phase == 31 || phase == 1)
					pos[4] += 0.005;
				if(phase == 30 || phase == 2)
					pos[4] += 0.004;
				if(phase == 29 || phase == 3)
					pos[4] += 0.003;
				if(phase == 28 || phase == 4)
					pos[4] += 0.002;

				if(phase == 16 || phase == 20)
					pos[4] += 0.002;
				if(phase == 17 || phase == 19)
					pos[4] += 0.004;
				if(phase == 18)
					pos[4] += 0.005;
			}

			mMotionFramesNonCyclic.at(i) = pos;

			int mBlendingInterval = 3;
			if(blend && phase == 0)
			{
				for(int j = mBlendingInterval; j > 0; j--)
				{
					double weight = 1.0 - j/(double)(mBlendingInterval+1);
					Eigen::VectorXd prevPos = mMotionFramesNonCyclic[i-j];
					mMotionFramesNonCyclic[i-j] = Utils::BlendPosition(prevPos, pos, weight, true);
				}
			}
		}
	}

	int idx = 0;
	for(int i=0; i<1000; i++)
	{
		double rem = i%34;
		if(rem == 32 || rem == 33){
			continue;
		}
		else{
			if(rem == 12)
				mMotionFramesNonCyclic[i][5] += 0.007;
			if(rem == 13)
				mMotionFramesNonCyclic[i][5] += 0.009;
			if(rem == 14)
				mMotionFramesNonCyclic[i][5] += 0.010;
			if(rem == 15)
				mMotionFramesNonCyclic[i][5] += 0.011;
			if(rem == 16)
				mMotionFramesNonCyclic[i][5] += 0.012;
			if(rem == 17)
				mMotionFramesNonCyclic[i][5] += 0.013;
			if(rem == 18)
				mMotionFramesNonCyclic[i][5] += 0.013;
			if(rem == 19)
				mMotionFramesNonCyclic[i][5] += 0.014;
			if(rem == 20)
				mMotionFramesNonCyclic[i][5] += 0.015;
			if(rem == 21)
				mMotionFramesNonCyclic[i][5] += 0.016;
			if(rem == 22)
				mMotionFramesNonCyclic[i][5] += 0.017;
			if(rem == 23)
				mMotionFramesNonCyclic[i][5] += 0.020;
			if(rem == 24)
				mMotionFramesNonCyclic[i][5] += 0.018;
			if(rem == 25)
				mMotionFramesNonCyclic[i][5] += 0.015;
			if(rem == 26)
				mMotionFramesNonCyclic[i][5] += 0.011;
			if(rem == 27)
				mMotionFramesNonCyclic[i][5] += 0.010;
			if(rem == 28)
				mMotionFramesNonCyclic[i][5] += 0.009;
			if(rem == 29)
				mMotionFramesNonCyclic[i][5] += 0.006;
			if(rem == 30)
				mMotionFramesNonCyclic[i][5] += 0.005;
			if(rem == 31)
				mMotionFramesNonCyclic[i][5] += 0.003;

			mMotionFramesNonCyclic[i][4] += idx*0.0002;
			mMotionFramesNonCyclicTmp.push_back(mMotionFramesNonCyclic[i]);
			idx++;
		}
	}
}

Eigen::VectorXd
BVH::
GetMotion(int k)
{
	return mMotionFrames[k];
}

Eigen::VectorXd
BVH::
GetMotionNonCyclic(int k)
{
	return mMotionFramesNonCyclicTmp[k];
}

void
BVH::
SetMotionVelFrames()
{
	int dof = mSkeleton->getNumDofs();
	mMotionVelFrames = Eigen::MatrixXd::Zero(mNumTotalFrames, dof);

	int num_joint = mSkeleton->getNumJoints();
	for(int i=0; i<mNumTotalFrames-1; i++)
	{
		Eigen::VectorXd p0 = mMotionFrames[i];
		Eigen::VectorXd p1 = mMotionFrames[i+1];
		Eigen::VectorXd v = Eigen::VectorXd::Zero(dof);
		int offset = 0;
		for(int j=0; j<num_joint; j++)
		{
			Joint* jn = mSkeleton->getJoint(j);
			if(jn->getType() == "FreeJoint"){
				v.segment<3>(offset) = Utils::CalcQuaternionVel(p0,p1,mTimeStep);
				v.segment<3>(offset+3) = (p1.segment<3>(offset+3)-p0.segment<3>(offset+3))/mTimeStep;
				offset += 6;
			}
			else if(jn->getType() == "BallJoint")
			{
				v.segment<3>(offset) = Utils::CalcQuaternionVelRel(p0,p1,mTimeStep);
				offset += 3;
			}
			else if(jn->getType() == "RevoluteJoint")
			{
				v[offset] = (p1[offset]-p0[offset])/mTimeStep;
				offset += 1;
			}
		}
		mMotionVelFrames.row(i) = v;
	}

	mMotionVelFrames.row(mNumTotalFrames-1) = mMotionVelFrames.row(mNumTotalFrames-2);

	for (int i=0; i<dof; ++i)
	{
		Eigen::VectorXd x = mMotionVelFrames.col(i);
		Utils::ButterworthFilter(mTimeStep, 6, x);
		mMotionVelFrames.col(i) = x;
	}
}

Eigen::VectorXd
BVH::
GetMotionVel(int k)
{
	return mMotionVelFrames.row(k);
}

void
BVH::
SetMotionVelFramesNonCyclic(int frames, bool blend)
{
	int dof = mSkeleton->getNumDofs();
	mMotionVelFramesNonCyclic = Eigen::MatrixXd::Zero(frames, dof);

	frames = mMotionFramesNonCyclicTmp.size();
	int num_joint = mSkeleton->getNumJoints();
	for(int i=0; i<frames-1; i++)
	{
		Eigen::VectorXd p0 = mMotionFramesNonCyclicTmp[i];
		Eigen::VectorXd p1 = mMotionFramesNonCyclicTmp[i+1];
		Eigen::VectorXd vel = Eigen::VectorXd::Zero(dof);
		int offset = 0;
		for(int j=0; j<num_joint; j++)
		{
			Joint* jn = mSkeleton->getJoint(j);
			if(jn->getType() == "FreeJoint"){
				vel.segment<3>(offset) = Utils::CalcQuaternionVel(p0,p1,mTimeStep);
				vel.segment<3>(offset+3) = (p1.segment<3>(offset+3)-p0.segment<3>(offset+3))/mTimeStep;
				offset += 6;
			}
			else if(jn->getType() == "BallJoint")
			{
				vel.segment<3>(offset) = Utils::CalcQuaternionVelRel(p0,p1,mTimeStep);
				offset += 3;
			}
			else if(jn->getType() == "RevoluteJoint")
			{
				vel[offset] = (p1[offset]-p0[offset])/mTimeStep;
				offset += 1;
			}
		}
		mMotionVelFramesNonCyclic.row(i) = vel;
	}

	mMotionVelFramesNonCyclic.row(frames-1) = mMotionVelFramesNonCyclic.row(frames-2);

	for (int i=0; i<dof; ++i)
	{
		Eigen::VectorXd x = mMotionVelFramesNonCyclic.col(i);
		Utils::ButterworthFilter(mTimeStep, 6, x);
		mMotionVelFramesNonCyclic.col(i) = x;
	}
}

Eigen::VectorXd
BVH::
GetMotionVelNonCyclic(int k)
{
	return mMotionVelFramesNonCyclic.row(k);
}

std::map<std::string,MASS::BVHNode::CHANNEL> BVHNode::CHANNEL_NAME =
{
	{"Xposition",Xpos},
	{"XPOSITION",Xpos},
	{"Yposition",Ypos},
	{"YPOSITION",Ypos},
	{"Zposition",Zpos},
	{"ZPOSITION",Zpos},
	{"Xrotation",Xrot},
	{"XROTATION",Xrot},
	{"Yrotation",Yrot},
	{"YROTATION",Yrot},
	{"Zrotation",Zrot},
	{"ZROTATION",Zrot}
};

};


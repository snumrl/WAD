#include "Device.h"
#include <iostream>

namespace MASS
{

Device::
Device(WorldPtr& wPtr)
:mUseDeviceNN(false),mNumParamState(0)
{
	mWorld = wPtr;
	mDelta_t = 0.3;
	mK_ = 15.0;
}

Device::
~Device()
{
	mCharacter = nullptr;
}

void
Device::
LoadSkeleton(const std::string& path, bool load_obj)
{
	mSkeleton = BuildFromFile(path, load_obj);
}

void
Device::
Initialize()
{
	if(mSkeleton == nullptr)
	{
		std::cout<<"Load Device First"<<std::endl;
		exit(0);
	}

	mWorld->addSkeleton(mSkeleton);

	mRootJointDof = 6;
	mNumDof = mSkeleton->getNumDofs();
	mNumActiveDof = mNumDof-mRootJointDof;
	mNumAction = mNumActiveDof;
	mAction = Eigen::VectorXd::Zero(mNumAction);
	mDesiredTorque = Eigen::VectorXd::Zero(mNumDof);

	mDelta_t_scaler = mSimulationHz;
	mDelta_t_idx = (int)(mDelta_t*mDelta_t_scaler);
	mK_scaler = 30.0;

	mDeviceSignals_y = std::deque<double>(1200+180, 0);
	mDeviceSignals_L = std::deque<double>(1200+180, 0);
	mDeviceSignals_R = std::deque<double>(1200+180, 0);

	mTorqueMax = 15.0;

	mNumState = this->GetState().rows();
	this->Reset();
}

void
Device::
SetHz(int sHz, int cHz)
{
	mSimulationHz = sHz;
	mControlHz = cHz;
	this->SetNumSteps(mSimulationHz/mControlHz);
}

void
Device::
Reset()
{
	SkeletonPtr skel_char = mCharacter->GetSkeleton();

	Eigen::VectorXd p(mNumDof);
	Eigen::VectorXd v(mNumDof);
	p.head(6) = skel_char->getPositions().head(6);
	v.head(6) = skel_char->getVelocities().head(6);
	p.segment<3>(6) = skel_char->getJoint("FemurL")->getPositions();
	p.segment<3>(9) = skel_char->getJoint("FemurR")->getPositions();
	v.segment<3>(6) = skel_char->getJoint("FemurL")->getVelocities();
	v.segment<3>(9) = skel_char->getJoint("FemurR")->getVelocities();

	mSkeleton->clearConstraintImpulses();
	mSkeleton->clearInternalForces();
	mSkeleton->clearExternalForces();

	mSkeleton->setPositions(p);
	mSkeleton->setVelocities(v);
	mSkeleton->computeForwardKinematics(true, false, false);

	mAction.setZero();
	mDesiredTorque.setZero();

	for(int i=0; i<mDeviceSignals_y.size(); i++){
		mDeviceSignals_y.at(i) = 0.0;
		mDeviceSignals_L.at(i) = 0.0;
		mDeviceSignals_R.at(i) = 0.0;
	}

	qr = 0.0;
	ql = 0.0;
	qr_prev = 0.0;
	ql_prev = 0.0;
}

void
Device::
Step(const Eigen::VectorXd& a_)
{
	mSkeleton->setForces(a_);
}

void
Device::
Step(double t)
{
	if(mUseDeviceNN)
	{
		SetDesiredTorques(t);
		// SetSignals();
		mSkeleton->setForces(mDesiredTorque);
	}
	else
	{
		SetDesiredTorques2();
		mSkeleton->setForces(mDesiredTorque);
	}
}

Eigen::VectorXd
Device::
GetState() const
{
	// root Pos & Vel
	// Eigen::VectorXd positions = mSkeleton->getPositions();
	// Eigen::VectorXd velocities = mSkeleton->getVelocities();

	// BodyNode* root = mSkeleton->getBodyNode(0);
	// Eigen::Quaterniond rotation(root->getWorldTransform().rotation());
	// Eigen::Vector3d root_linvel = root->getCOMLinearVelocity();
	// Eigen::Vector3d root_angvel = root->getAngularVelocity();
	// Eigen::VectorXd state(22);

	// state << rotation.w(), rotation.x(), rotation.y(), rotation.z(),
	//             root_linvel / 10., root_angvel/10., positions.tail<6>(), velocities.tail<6>()/10.;

	// double history_window = 0.20;
	// double history_interval = 0.05;
	// int offset = (history_interval * mSimulationHz);
	// int history_num = (history_window+0.001)/(history_interval)+1;

	double history_interval = 0.15;
	int offset = (history_interval * mSimulationHz);
	int history_num = 5;

	int parameter_num = mNumParamState;
	Eigen::VectorXd state(history_num*2+parameter_num);
	double scaler = 2.0;
	for(int i=0; i<history_num; i++)
	{
		double signal_y = mDeviceSignals_y.at(mDelta_t_idx - (i-2)*offset);
		double torque_l = mK_ * signal_y;
		double torque_r = mK_ * signal_y;
		double des_torque_l =  1*torque_l;
		double des_torque_r = -1*torque_r;

		state[i*2] = des_torque_l/mK_ * scaler;
		state[i*2+1] = des_torque_r/mK_ * scaler;
	}

	for(int i=0; i<parameter_num; i++)
		state[history_num*2 + i] = mParamState[i];

	return state;
}

void
Device::
SetAction(const Eigen::VectorXd& a)
{
	double action_scale = 1.0;
	mAction = a*action_scale;
	for(int i=0; i<mAction.size()-2; i++)
		mAction *= mTorqueMax;
}

void
Device::
SetDesiredTorques(double t)
{
}

double
lp_filter(double cur, double prev, double alpha)
{
	return (1-alpha)*prev + (alpha)*cur;
}

void
Device::
SetDesiredTorques2()
{
	mDesiredTorque.setZero();

	if(qr==0.0 && ql==0.0 && qr_prev==0.0 && ql_prev==0.0)
	{
		ql = GetAngleQ("FemurL");
		qr = GetAngleQ("FemurR");
		ql_prev = ql;
		qr_prev = qr;
	}
	else{
		ql = GetAngleQ("FemurL");
		qr = GetAngleQ("FemurR");
	}

	double alpha = 0.05;
	ql = lp_filter(ql, ql_prev, alpha);
	qr = lp_filter(qr, qr_prev, alpha);
	ql_prev = ql;
	qr_prev = qr;

	double y = sin(qr) - sin(ql);

	double beta_L = 1.0;
	double beta_Lhip = 1.0;
	double beta_R = 1.0;
	double beta_Rhip = 1.0;

	mDeviceSignals_y.pop_back();
	mDeviceSignals_y.push_front(y);

	// double torque = k_ * y_delta_t;
	double torque_l = mK_ * mDeviceSignals_y.at(mDelta_t_idx);
	double torque_r = mK_ * mDeviceSignals_y.at(mDelta_t_idx);
	double des_torque_l =  1*torque_l*beta_L*beta_Lhip;
	double des_torque_r = -1*torque_r*beta_R*beta_Rhip;

	mDeviceSignals_L.pop_back();
	mDeviceSignals_L.push_front(des_torque_l);

	mDeviceSignals_R.pop_back();
	mDeviceSignals_R.push_front(des_torque_r);

	mDesiredTorque[6] = des_torque_l;
	mDesiredTorque[9] = des_torque_r;
}

double
Device::
GetAngleQ(const std::string& name)
{
	SkeletonPtr skel_char = mCharacter->GetSkeleton();
	Eigen::Vector3d dir = skel_char->getBodyNode(0)->getCOMLinearVelocity();
	dir /= dir.norm();

	Eigen::Vector3d p12 = skel_char->getBodyNode(name)->getCOM()-skel_char->getBodyNode(0)->getCOM();
	double p12_len = p12.norm();

	double l2 = dir[0]*p12[0] + dir[2]*p12[2];
	double l1 = sqrt(p12[0]*p12[0]+p12[2]*p12[2] - l2*l2);
	double x = sqrt(p12_len*p12_len - l1*l1);

	double sin = l2 / x;

	return asin(sin);
}

const Eigen::VectorXd&
Device::
GetDesiredTorques()
{
	return mDesiredTorque;
}

const std::deque<double>&
Device::
GetSignals(int idx)
{
	if(idx==0)
		return mDeviceSignals_L;

	if(idx==1)
		return mDeviceSignals_R;

	if(idx==2)
		return mDeviceSignals_y;
}

void
Device::
SetK_(double k)
{
	mK_ = k;

	double param = 0.0;
	if(mMax_v[0] == mMin_v[0])
	{
		mParamState[0] = mMin_v[0];
	}
	else
	{
		double ratio = (mK_/mK_scaler-mMin_v[0])/(mMax_v[0]-mMin_v[0]);
		param = ratio*2.0 - 1.0;
		mParamState[0] = param;
	}
}

void
Device::
SetDelta_t(double t)
{
	mDelta_t = t;
	mDelta_t_idx = (int)(mDelta_t*mDelta_t_scaler);

	double param = 0.0;
	if(mMax_v[1] == mMin_v[1])
	{
		mParamState[1] = mMin_v[1];
	}
	else
	{
		double ratio = (mDelta_t-mMin_v[1])/(mMax_v[1]-mMin_v[1]);
		param = ratio*2.0 - 1.0;
		mParamState[1] = param;
	}
}

void
Device::
SetNumParamState(int n)
{
	mNumParamState = n;
	mParamState = Eigen::VectorXd::Zero(mNumParamState);
	mMin_v = Eigen::VectorXd::Zero(mNumParamState);
	mMax_v = Eigen::VectorXd::Zero(mNumParamState);
}

void
Device::
SetParamState(const Eigen::VectorXd& paramState)
{
	mParamState = paramState;
	double param = 0.0;
	for(int i=0; i<paramState.size(); i++)
	{
		param = paramState[i];
		param = mMin_v[i]+(mMax_v[i]-mMin_v[i])*(param+1.0)/2.0;
		if(i==0)
			this->SetK_(param*mK_scaler);
		else if(i==1)
			this->SetDelta_t(param);
	}
}

void
Device::
SetMinMaxV(int idx, double lower, double upper)
{
	// 0 : k
	// 1 : delta t
	mMin_v[idx] = lower;
	mMax_v[idx] = upper;
}

void
Device::
SetAdaptiveParams(std::map<std::string, std::pair<double, double>>& p)
{
	for(auto p_ : p){
		std::string name = p_.first;
		double lower = (p_.second).first;
		double upper = (p_.second).second;

		if(name == "k"){
			this->SetMinMaxV(0, lower, upper);
			this->SetK_(lower*mK_scaler);
		}
		else if(name == "delta_t"){
			this->SetMinMaxV(1, lower, upper);
			this->SetDelta_t(lower);
		}
	}
}

void
Device::
SetAdaptiveParams(std::string name, double lower, double upper)
{
	if(name == "k"){
		this->SetMinMaxV(0, lower, upper);
		this->SetK_(lower*mK_scaler);
	}
	else if(name == "delta_t"){
		this->SetMinMaxV(1, lower, upper);
		this->SetDelta_t(lower);
	}
}

}

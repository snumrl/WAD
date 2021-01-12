#include "Muscle.h"

using namespace MASS;
using namespace dart::dynamics;

std::vector<int> sort_indices(const std::vector<double>& val)
{
	std::vector<int> idx(val.size());
	std::iota(idx.begin(),idx.end(),0);

	std::sort(idx.begin(),idx.end(),[&val](int i1,int i2){return val[i1]<val[i2];});

	return idx;
}

Anchor::
Anchor(std::vector<BodyNode*> bns,std::vector<Eigen::Vector3d> lps,std::vector<double> ws)
	:bodynodes(bns),local_positions(lps),weights(ws),num_related_bodies(bns.size())
{
}

Anchor::
~Anchor()
{
	for(int i=0; i<bodynodes.size(); i++)
		delete bodynodes[i];
}

Eigen::Vector3d
Anchor::
GetPoint()
{
	Eigen::Vector3d p = Eigen::Vector3d::Zero();
	for(int i=0; i<num_related_bodies; i++)
		p += weights[i]*(bodynodes[i]->getTransform()*local_positions[i]);
	return p;
}

Muscle::
Muscle(std::string name,double _f0,double _lm0,double _lt0,double _pen_angle,double lmax)
	:mName(name),f0(_f0),l_m0(_lm0),l_mt(1.0),l_t0(_lt0),l_mt0(0.0),activation(0.0),activation_prev(0.0),f_toe(0.33),k_toe(3.0),k_lin(51.878788),e_toe(0.02),e_t0(0.033),k_pe(4.0),e_mo(0.6),gamma(0.5),l_mt_max(lmax),dl_m(0.0),mMass(0.0),mass_scaler(0.1)
{
	l_m = l_mt - l_t0;
	l_m_prev = l_m;
	f0_default = f0;
}

Muscle::
~Muscle()
{
	for(int i=0; i<mAnchors.size(); i++)
		delete mAnchors[i];
}

void
Muscle::
Reset()
{
	activation = 0;
	activation_prev = 0;

	l_mt = Getl_mt();
	l_m = l_mt - l_t0;
	l_m_prev = l_m;

	dl_m = 0.0;
}

void
Muscle::
AddAnchor(const dart::dynamics::SkeletonPtr& skel,dart::dynamics::BodyNode* bn,const Eigen::Vector3d& glob_pos,int num_related_bodies)
{
	std::vector<double> distance;
	distance.resize(skel->getNumBodyNodes(), 0.0);

	std::vector<Eigen::Vector3d> local_positions;
	local_positions.resize(skel->getNumBodyNodes());

	for(int i=0; i<skel->getNumBodyNodes(); i++)
	{
		Eigen::Isometry3d T, T_i;
		T_i = skel->getBodyNode(i)->getTransform();
		T = T_i * skel->getBodyNode(i)->getParentJoint()->getTransformFromChildBodyNode();

		local_positions[i] = T_i.inverse() * glob_pos;
		distance[i] = (glob_pos - T.translation()).norm();
	}
	std::vector<int> index_sort_by_distance = sort_indices(distance);

	std::vector<dart::dynamics::BodyNode*> lbs_body_nodes;
	std::vector<Eigen::Vector3d> lbs_local_positions;
	std::vector<double> lbs_weights;

	double total_weight = 0.0;
	int idx_zero = index_sort_by_distance[0];
	double distance_idx_zero = distance[idx_zero];
	if(distance_idx_zero < 0.08)
	{
		lbs_weights.push_back(1.0/sqrt(distance_idx_zero));
		total_weight += lbs_weights[0];
		lbs_body_nodes.push_back(skel->getBodyNode(idx_zero));
		lbs_local_positions.push_back(local_positions[idx_zero]);

		if(lbs_body_nodes[0]->getParentBodyNode() != nullptr)
		{
			auto bn_parent = lbs_body_nodes[0]->getParentBodyNode();
			lbs_weights.push_back(1.0/sqrt(distance[bn_parent->getIndexInSkeleton()]));
			total_weight += lbs_weights[1];
			lbs_body_nodes.push_back(bn_parent);
			lbs_local_positions.push_back(local_positions[bn_parent->getIndexInSkeleton()]);
		}
	}
	else
	{
		total_weight = 1.0;
		lbs_weights.push_back(1.0);
		lbs_body_nodes.push_back(bn);
		lbs_local_positions.push_back(bn->getTransform().inverse()*glob_pos);
	}

	for(int i=0; i<lbs_body_nodes.size(); i++)
		lbs_weights[i] /= total_weight;

	mAnchors.push_back(new Anchor(lbs_body_nodes,lbs_local_positions,lbs_weights));

	int n = mAnchors.size();
	if(n>1){
		l_mt0 += (mAnchors[n-1]->GetPoint()-mAnchors[n-2]->GetPoint()).norm();
		mMass += l_mt0 * mass_scaler;
	}
	mCachedAnchorPositions.resize(n);

	this->Update();
	Eigen::MatrixXd Jt = GetJacobianTranspose();
	auto Ap = GetForceJacobianAndPassive();
	Eigen::VectorXd JtA = Jt*Ap.first;
	mNumRelatedDofs = 0;
	mRelatedDofIndices.clear();
	for(int i=0; i<JtA.rows(); i++)
	{
		if(std::abs(JtA[i]) > 1E-3){
			mNumRelatedDofs++;
			mRelatedDofIndices.push_back(i);
		}
	}
}

void
Muscle::
AddAnchor(dart::dynamics::BodyNode* bn,const Eigen::Vector3d& glob_pos)
{
	std::vector<dart::dynamics::BodyNode*> lbs_body_nodes;
	std::vector<Eigen::Vector3d> lbs_local_positions;
	std::vector<double> lbs_weights;

	lbs_body_nodes.push_back(bn);
	lbs_local_positions.push_back(bn->getTransform().inverse()*glob_pos);
	lbs_weights.push_back(1.0);

	mAnchors.push_back(new Anchor(lbs_body_nodes,lbs_local_positions,lbs_weights));

	int n = mAnchors.size();
	if(n>1){
		l_mt0 += (mAnchors[n-1]->GetPoint()-mAnchors[n-2]->GetPoint()).norm();
		mMass += l_mt0 * mass_scaler;
	}

	mCachedAnchorPositions.resize(n);
	this->Update();
	Eigen::MatrixXd Jt = GetJacobianTranspose();
	auto Ap = GetForceJacobianAndPassive();
	Eigen::VectorXd JtA = Jt*Ap.first;
	mNumRelatedDofs = 0;
	mRelatedDofIndices.clear();
	for(int i=0; i<JtA.rows(); i++)
	{
		if(std::abs(JtA[i])>1E-3)
		{
			mNumRelatedDofs++;
			mRelatedDofIndices.push_back(i);
		}
	}
}

void
Muscle::
ApplyForceToBody()
{
	double f = GetForce();

	for(int i=0; i<mAnchors.size()-1; i++)
	{
		Eigen::Vector3d dir = (mCachedAnchorPositions[i+1]-mCachedAnchorPositions[i]);
		dir.normalize();
		dir = f*dir;
		mAnchors[i]->bodynodes[0]->addExtForce(dir,mCachedAnchorPositions[i],false,false);
	}

	for(int i=1; i<mAnchors.size(); i++)
	{
		Eigen::Vector3d dir = (mCachedAnchorPositions[i-1]-mCachedAnchorPositions[i]);
		dir.normalize();
		dir = f*dir;
		mAnchors[i]->bodynodes[0]->addExtForce(dir,mCachedAnchorPositions[i],false,false);
	}
}

void
Muscle::
SetActivation(double a)
{
	activation_prev = activation;
	activation = a;
}

double
Muscle::
GetStimulation()
{
	double stim = (activation - activation_prev) / mTimeStep;
	if(stim > 0)
		return stim;
	else
		return 0.0;
}

void
Muscle::
Update()
{
	for(int i=0; i<mAnchors.size(); i++)
		mCachedAnchorPositions[i] = mAnchors[i]->GetPoint();

	l_m_prev = l_m;

	l_mt = Getl_mt();
	l_m = l_mt - l_t0;

	dl_m = (l_m - l_m_prev) / mTimeStep;
}

double
Muscle::
GetForce()
{
	for(int i=0; i<20; i++)
	{
		double force = 0.0;
		force = f0* (g_al(0.1*i) + g_pl(0.1*i));
	}

	return Getf_A()*activation + Getf_p();
}

double
Muscle::
Getf_A()
{
	return f0*g_al(l_m/l_m0);
}

double
Muscle::
Getf_p()
{
	return f0*g_pl(l_m/l_m0);
}

double
Muscle::
Getl_mt()
{
	l_mt = 0.0;
	for(int i=1; i<mAnchors.size(); i++)
		l_mt += (mCachedAnchorPositions[i]-mCachedAnchorPositions[i-1]).norm();

	return l_mt*(l_m0 + l_t0) / (l_mt0 * l_m0) - (l_t0 / l_m0) + l_t0;
}


double
Muscle::
g(double _l_m)
{
	double e_t = (l_mt-_l_m-l_t0)/l_t0;
	_l_m = _l_m/l_m0;
	double f = g_t(e_t) - (g_pl(_l_m) + activation*g_al(_l_m));
	return f;
}

double
Muscle::
g_t(double e_t)
{
	double f_t;
	if(e_t <= e_t0)
		f_t = f_toe/(exp(k_toe)-1)*(exp(k_toe*e_t/e_toe)-1);
	else
		f_t = k_lin*(e_t-e_toe)+f_toe;

	return f_t;
}

double
Muscle::
g_pl(double _l_m)
{
	double f_pl = (exp(k_pe*(_l_m-1.0)/e_mo)-1.0)/(exp(k_pe)-1.0);
	if(_l_m < 1.0)
		return 0.0;
	else
		return f_pl;
}

double
Muscle::
g_al(double _l_m)
{
	return exp(-(_l_m-1.0)*(_l_m-1.0)/gamma);
}

void
Muscle::
SetMt0Ratio(double ratio)
{
	l_mt0_ratio = ratio;
	l_mt0 = l_mt0_default * l_mt0_ratio;
}

void
Muscle::
SetF0Ratio(double ratio)
{
	f0_ratio = ratio;
	f0 = f0_default * f0_ratio;
}


Eigen::VectorXd
Muscle::
GetRelatedJtA()
{
	Eigen::MatrixXd Jt_reduced = GetReducedJacobianTranspose();
	Eigen::VectorXd A = GetForceJacobianAndPassive().first;
	Eigen::VectorXd JtA_reduced = Jt_reduced*A;
	return JtA_reduced;
}

Eigen::MatrixXd
Muscle::
GetReducedJacobianTranspose()
{
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	Eigen::MatrixXd Jt(mNumRelatedDofs, 3*mAnchors.size());

	Jt.setZero();
	for(int i=0; i<mAnchors.size(); i++){
		auto bn = mAnchors[i]->bodynodes[0];
		dart::math::Jacobian J = dart::math::Jacobian::Zero(6, mNumRelatedDofs);
		for(int j=0; j<mNumRelatedDofs; j++){
			auto& indices = bn->getDependentGenCoordIndices();
			int idx = std::find(indices.begin(), indices.end(), mRelatedDofIndices[j]) - indices.begin();
			if(idx != indices.size())
				J.col(j) = bn->getJacobian().col(idx);
		}
		// from https://github.com/dartsim/dart/blob/master/dart/dynamics/detail/TemplatedJacobianNode.hpp#L121
		Eigen::Vector3d offset = mAnchors[i]->bodynodes[0]->getTransform().inverse()*mCachedAnchorPositions[i];
		dart::math::LinearJacobian JLinear = J.bottomRows<3>() + J.topRows<3>().colwise().cross(offset);
		Jt.block(0,i*3,mNumRelatedDofs,3) = (bn->getTransform().linear() * JLinear).transpose();
	}
	return Jt;
}

Eigen::MatrixXd
Muscle::
GetJacobianTranspose()
{
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	int dof = skel->getNumDofs();
	Eigen::MatrixXd Jt(dof,3*mAnchors.size());

	Jt.setZero();
	for(int i=0; i<mAnchors.size(); i++){
		Jt.block(0,i*3,dof,3) = skel->getLinearJacobian(mAnchors[i]->bodynodes[0],mAnchors[i]->bodynodes[0]->getTransform().inverse()*mCachedAnchorPositions[i]).transpose();
	}

	return Jt;
}

std::pair<Eigen::VectorXd,Eigen::VectorXd>
Muscle::
GetForceJacobianAndPassive()
{
	double f_a = Getf_A();
	double f_p = Getf_p();

	std::vector<Eigen::Vector3d> force_dir;
	for(int i=0; i<mAnchors.size(); i++)
		force_dir.push_back(Eigen::Vector3d::Zero());

	for(int i=0; i<mAnchors.size()-1; i++)
	{
		Eigen::Vector3d dir = mCachedAnchorPositions[i+1]-mCachedAnchorPositions[i];
		dir.normalize();
		force_dir[i] += dir;
	}

	for(int i=1; i<mAnchors.size(); i++)
	{
		Eigen::Vector3d dir = mCachedAnchorPositions[i-1]-mCachedAnchorPositions[i];
		dir.normalize();
		force_dir[i] += dir;
	}

	Eigen::VectorXd A(3*mAnchors.size());
	Eigen::VectorXd p(3*mAnchors.size());
	A.setZero();
	p.setZero();

	for(int i=0; i<mAnchors.size(); i++)
	{
		A.segment<3>(i*3) = force_dir[i]*f_a;
		p.segment<3>(i*3) = force_dir[i]*f_p;
	}

	return std::make_pair(A,p);
}

void
Muscle::
ComputeJacobians()
{
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	int dof = skel->getNumDofs();
	mCachedJs.resize(mAnchors.size());
	for(int i=0; i<mAnchors.size(); i++)
	{
		mCachedJs[i].resize(3, skel->getNumDofs());
		mCachedJs[i].setZero();
		for(int j=0; j<mAnchors[i]->num_related_bodies; j++){
			mCachedJs[i] += mAnchors[i]->weights[j]*skel->getLinearJacobian(mAnchors[i]->bodynodes[j],mAnchors[i]->local_positions[j]);
		}
	}
}

double
Muscle::
GetMetabolicEnergyRate()
{
	double h_A = 0.0; // activation_heat_rate

	double u = 0.0; // excitation == stimulation
	u = this->GetStimulation();
	double A_f = 133; // activation heat rate. fast twitch muscle
	double A_s = 40; // activation heat rate. slow twitch muscle
	double u_f = 1-cos(M_PI/2.0 * u); // 1 - cos(PI/2 * u_t)
	double u_s = sin(M_PI/2.0 * u); // sin(PI/2 * u_t)
	double f_FT = 0.5; // percentages of fast twitch muscles
	double f_ST = 0.5; // percentages of fast twitch muscles

	h_A = mMass * (A_s*u_s*f_ST + A_f*u_f*f_FT);

	double h_M = 0.0; // maintenance_heat_rate

	double l_M = 0.0;
	double l_m = this->Getl_m();
	double l_m_opt = this->Getl_m0();
	double l_ce = l_m/l_m_opt;
	if(l_ce <= 0.5)
		l_M = 0.5;
	else if(l_ce <= 1.0)
		l_M = l_ce;
	else if(l_ce <= 1.5)
		l_M = -2.0 * l_ce + 3.0;
	else
		l_M = 0.0;

	double M_f = 111;
	double M_s = 74;

	h_M = mMass * l_M * (M_s*u_s*f_ST + M_f*u_f*f_FT);

	double h_SL = 0.0; // shortening,lengthening heat rate
	double alpha = 0.0;
	double v_CE = this->Getdl_m();
	double force = this->GetForce();
	if(v_CE <= 0)
		alpha = 0.16*activation*f0*g_al(l_m) + 0.18*force;
	else
		alpha = 0.157*force;

	h_SL = -1*alpha*v_CE;

	double w = 0.0; // work
	w = -1*v_CE*force;

	mMetabolicEnergyRate = h_A + h_M + h_SL + w;
	return mMetabolicEnergyRate;
}

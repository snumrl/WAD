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
Muscle(std::string _name,double _f0,double _lm0,double _lt0,double _pen_angle,double lmax)
	:name(_name),f0(_f0),l_m0(_lm0),l_mt(1.0),l_t0(_lt0),l_mt0(0.0),activation(0.0),f_toe(0.33),k_toe(3.0),k_lin(51.878788),e_toe(0.02),e_t0(0.033),k_pe(4.0),e_mo(0.6),gamma(0.5),l_mt_max(lmax)
{
	l_m = l_mt - l_t0;
}

Muscle::
~Muscle()
{
	for(int i=0; i<mAnchors.size(); i++)
		delete mAnchors[i];
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
	}

	mCachedAnchorPositions.resize(n);

	this->Update();
	Eigen::MatrixXd Jt = GetJacobianTranspose();
	auto Ap = GetForceJacobianAndPassive();
	Eigen::VectorXd JtA = Jt*Ap.first;
	num_related_dofs = 0;
	related_dof_indices.clear();
	for(int i=0; i<JtA.rows(); i++)
	{
		if(std::abs(JtA[i]) > 1E-3){
			num_related_dofs++;
			related_dof_indices.push_back(i);
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
	}

	mCachedAnchorPositions.resize(n);
	this->Update();
	Eigen::MatrixXd Jt = GetJacobianTranspose();
	auto Ap = GetForceJacobianAndPassive();
	Eigen::VectorXd JtA = Jt*Ap.first;
	num_related_dofs = 0;
	related_dof_indices.clear();
	for(int i=0; i<JtA.rows(); i++)
	{
		if(std::abs(JtA[i])>1E-3)
		{
			num_related_dofs++;
			related_dof_indices.push_back(i);
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
Update()
{
	for(int i=0; i<mAnchors.size(); i++)
		mCachedAnchorPositions[i] = mAnchors[i]->GetPoint();

	l_mt = Getl_mt();
	l_m = l_mt - l_t0;
}

double
Muscle::
GetForce()
{
	// if(!name.compare("L_Rectus_Abdominis1")||
	// 	!name.compare("R_Rectus_Abdominis1")||
	// 	!name.compare("L_Transversus_Abdominis4")||
	// 	!name.compare("R_Transversus_Abdominis4"))
	// {
	// 	double f_A = Getf_A();
	// 	double f_p = Getf_p();
	// 	if(activation > 0)
	// 	{
	// 		std::cout << "name : " << name << std::endl;
	// 		std::cout << "f_A : " << f_A << std::endl;
	// 		std::cout << "activation : " << activation << std::endl;
	// 		std::cout << "f_p : " << f_p << std::endl;
	// 		std::cout << std::endl;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "name : " << name << std::endl;
	// 		std::cout << "f_p : " << f_p << std::endl;
	// 		std::cout << std::endl;
	// 	}
	// }

	// if(!name.compare("L_Multifidus")||
	// 	!name.compare("R_Multifidus")||
	// 	!name.compare("L_Quadratus_Lumborum1")||
	// 	!name.compare("R_Quadratus_Lumborum1")||
	// 	!name.compare("L_Transversus_Abdominis")||
	// 	!name.compare("R_Transversus_Abdominis"))
	// {
	// 	double f_A = Getf_A();
	// 	double f_p = Getf_p();
	// 	if(activation > 0)
	// 	{
	// 		std::cout << "name : " << name << std::endl;
	// 		std::cout << "f_A : " << f_A << std::endl;
	// 		std::cout << "activation : " << activation << std::endl;
	// 		std::cout << "f_p : " << f_p << std::endl;
	// 		std::cout << std::endl;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "name : " << name << std::endl;
	// 		std::cout << "f_p : " << f_p << std::endl;
	// 		std::cout << std::endl;
	// 	}
	// }

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

	return l_mt/l_mt0;
}

Eigen::VectorXd
Muscle::
GetRelatedJtA()
{
	Eigen::MatrixXd Jt_reduced = GetReducedJacobianTranspose();
	Eigen::VectorXd A = GetForceJacobianAndPassive().first;
	Eigen::VectorXd JtA_reduced = Jt_reduced*A;
	return JtA_reduced;

	// Eigen::MatrixXd Jt = GetJacobianTranspose();
	// Eigen::VectorXd A = GetForceJacobianAndPassive().first;
	// Eigen::VectorXd JtA = Jt*A;
	// Eigen::VectorXd JtA_reduced = Eigen::VectorXd::Zero(num_related_dofs);

	// for(int i =0;i<num_related_dofs;i++)
	// 	JtA_reduced[i] = JtA[related_dof_indices[i]];

	// return JtA_reduced;
}

Eigen::MatrixXd
Muscle::
GetReducedJacobianTranspose()
{
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	Eigen::MatrixXd Jt(num_related_dofs, 3*mAnchors.size());

	Jt.setZero();
	for(int i=0; i<mAnchors.size(); i++){
		auto bn = mAnchors[i]->bodynodes[0];
		dart::math::Jacobian J = dart::math::Jacobian::Zero(6, num_related_dofs);
		for(int j=0; j<num_related_dofs; j++){
			auto& indices = bn->getDependentGenCoordIndices();
			int idx = std::find(indices.begin(), indices.end(), related_dof_indices[j]) - indices.begin();
			if(idx != indices.size())
				J.col(j) = bn->getJacobian().col(idx);
		}
		// from https://github.com/dartsim/dart/blob/master/dart/dynamics/detail/TemplatedJacobianNode.hpp#L121
		Eigen::Vector3d offset = mAnchors[i]->bodynodes[0]->getTransform().inverse()*mCachedAnchorPositions[i];
		dart::math::LinearJacobian JLinear = J.bottomRows<3>() + J.topRows<3>().colwise().cross(offset);
		Jt.block(0,i*3,num_related_dofs,3) = (bn->getTransform().linear() * JLinear).transpose();
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

std::vector<dart::dynamics::Joint*>
Muscle::
GetRelatedJoints()
{
	auto skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	std::map<dart::dynamics::Joint*,int> jns;
	std::vector<dart::dynamics::Joint*> jns_related;
	for(int i=0; i<skel->getNumJoints(); i++)
		jns.insert(std::make_pair(skel->getJoint(i),0));

	Eigen::VectorXd dl_dtheta = Getdl_dtheta();
	for(int i=0; i<dl_dtheta.rows(); i++)
	{
		if(std::abs(dl_dtheta[i])>1E-6)
			jns[skel->getDof(i)->getJoint()]+=1;
	}

	for(auto jn : jns)
	{
		if(jn.second>0)
			jns_related.push_back(jn.first);
	}

	return jns_related;
}

std::vector<dart::dynamics::BodyNode*>
Muscle::
GetRelatedBodyNodes()
{
	std::vector<dart::dynamics::BodyNode*> bns_related;
	auto rjs = GetRelatedJoints();
	for(auto joint : rjs)
		bns_related.push_back(joint->getChildBodyNode());

	return bns_related;
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

Eigen::VectorXd
Muscle::
Getdl_dtheta()
{
	ComputeJacobians();
	const auto& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	Eigen::VectorXd dl_dtheta(skel->getNumDofs());
	dl_dtheta.setZero();
	for(int i=0; i<mAnchors.size()-1; i++)
	{
		Eigen::Vector3d pi = mCachedAnchorPositions[i+1] - mCachedAnchorPositions[i];
		Eigen::MatrixXd dpi_dtheta = mCachedJs[i+1] - mCachedJs[i];
		Eigen::VectorXd dli_d_theta = (dpi_dtheta.transpose()*pi)/(l_mt0*pi.norm());
		dl_dtheta += dli_d_theta;
	}

	for(int i=0; i<dl_dtheta.rows(); i++)
	{
		if(std::abs(dl_dtheta[i])<1E-6)
			dl_dtheta[i] = 0.0;
	}

	return dl_dtheta;
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

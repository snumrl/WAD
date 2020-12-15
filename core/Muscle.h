#ifndef __MASS_MUSCLE_H__
#define __MASS_MUSCLE_H__
#include "dart/dart.hpp"

namespace MASS
{
struct Anchor
{
	Anchor(std::vector<dart::dynamics::BodyNode*> bns,std::vector<Eigen::Vector3d> lps,std::vector<double> ws);
	~Anchor();

	Eigen::Vector3d GetPoint();

	int num_related_bodies;
	std::vector<dart::dynamics::BodyNode*> bodynodes;
	std::vector<Eigen::Vector3d> local_positions;
	std::vector<double> weights;
};

class Muscle
{
public:
	Muscle(std::string _name,double f0,double lm0,double lt0,double pen_angle,double lmax);
	~Muscle();

	void AddAnchor(const dart::dynamics::SkeletonPtr& skel,dart::dynamics::BodyNode* bn,const Eigen::Vector3d& glob_pos,int num_related_bodies);
	void AddAnchor(dart::dynamics::BodyNode* bn,const Eigen::Vector3d& glob_pos);
	const std::vector<Anchor*>& GetAnchors(){return mAnchors;}
	void Update();
	void ApplyForceToBody();
	double GetForce();
	double Getf_A();
	double Getf_p();
	double Getl_mt();
	std::string GetName(){return name;}
	void SetFemur(bool b){isFemur = b;}
	bool GetFemur(){return isFemur;}

	Eigen::MatrixXd GetJacobianTranspose();
	Eigen::MatrixXd GetReducedJacobianTranspose();
	std::pair<Eigen::VectorXd,Eigen::VectorXd> GetForceJacobianAndPassive();

	int GetNumRelatedDofs(){return num_related_dofs;};
	Eigen::VectorXd GetRelatedJtA();

	std::vector<dart::dynamics::Joint*> GetRelatedJoints();
	std::vector<dart::dynamics::BodyNode*> GetRelatedBodyNodes();
	void ComputeJacobians();
	Eigen::VectorXd Getdl_dtheta();


private:
	std::string name;
	std::vector<Anchor*> mAnchors;
	int num_related_dofs;
	std::vector<int> related_dof_indices;

	std::vector<Eigen::Vector3d> mCachedAnchorPositions;
	std::vector<Eigen::MatrixXd> mCachedJs;

	bool isFemur = false;

public:
	//Dynamics
	double g(double _l_m);
	double g_t(double e_t);
	double g_pl(double _l_m);
	double g_al(double _l_m);

	void SetActivation(double a){ activation = a;}
	double GetActivation(){ return activation;}

	void SetF0(double f){ f0 = f;}
	double GetF0(){ return f0;}

	void SetMt0Ratio(double ratio){l_mt0 *= ratio;}
	void SetF0Ratio(double ratio){f0 *= ratio;}

private:

	double l_mt,l_mt_max;
	double l_m;
	double activation;

	double f0;
	double l_mt0,l_m0,l_t0;

	double f_toe,e_toe,k_toe,k_lin,e_t0; //For g_t
	double k_pe,e_mo; //For g_pl
	double gamma; //For g_al
};

}
#endif

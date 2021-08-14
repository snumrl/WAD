#ifndef __MUSCLE_H__
#define __MUSCLE_H__
#include "dart/dart.hpp"

using namespace dart::dynamics;
namespace WAD
{

struct Anchor
{
	Anchor(std::vector<BodyNode*> bns,std::vector<Eigen::Vector3d> lps,std::vector<double> ws);
	~Anchor();

	Eigen::Vector3d GetPoint();

	int num_related_bodies;
	std::vector<BodyNode*> bodynodes;
	std::vector<Eigen::Vector3d> local_positions;
	std::vector<double> weights;
};

class Muscle
{
public:
	Muscle(std::string _name,double f0,double lm0,double lt0,double pen_angle,double lmax);
	~Muscle();

	void Reset();

	void AddAnchor(const SkeletonPtr& skel,BodyNode* bn,const Eigen::Vector3d& glob_pos,int num_related_bodies);
	void AddAnchor(BodyNode* bn,const Eigen::Vector3d& glob_pos);
	const std::vector<Anchor*>& GetAnchors(){return mAnchors;}
	void Update();
	void ApplyForceToBody();
	double GetForce();
	double Getf_A();
	double Getf_p();
	double Getl_mt();
	std::string GetName(){return mName;}
	void SetFemur(bool b){isFemur = b;}
	bool GetFemur(){return isFemur;}

	Eigen::MatrixXd GetJacobianTranspose();
	Eigen::MatrixXd GetReducedJacobianTranspose();
	std::pair<Eigen::VectorXd,Eigen::VectorXd> GetForceJacobianAndPassive();

	int GetNumRelatedDofs(){return mNumRelatedDofs;};
	Eigen::VectorXd GetRelatedJtA();

	void ComputeJacobians();

private:
	std::string mName;
	std::vector<Anchor*> mAnchors;
	int mNumRelatedDofs;
	std::vector<int> mRelatedDofIndices;

	std::vector<Eigen::Vector3d> mCachedAnchorPositions;
	std::vector<Eigen::MatrixXd> mCachedJs;

	bool isFemur = false;

public:
	//Dynamics
	// double g(double _l_m);
	double g_t(double e_t);
	double g_pl(double _l_m);
	double g_al(double _l_m);

	void SetActivation(double a);
	double GetActivation(){ return activation;}
	double GetActivationPrev(){ return activation_prev;}

	void SetConTimeStep(double t){mConTimeStep = t;}
	void SetSimTimeStep(double t){mSimTimeStep = t;}
	
	double GetExcitation();

	void SetF0(double f){ f0 = f;}
	double GetF0(){ return f0;}

	void SetMt0(double mt){ l_mt0 = mt;}
	double GetMt0(){ return l_mt0;}

	void SetMt0Ratio(double ratio);
	void SetF0Ratio(double ratio);
	void SetMt0Default(){l_mt0_default = l_mt0;}

	double GetF0Ratio(){return f0_ratio;}
	double GetF0Default(){return f0_default;}
	double GetMt0Ratio(){return l_mt0_ratio;}
	double GetMt0Default(){return l_mt0_default;}

	double Getl_m(){return l_m;}
	double Getl_m0(){return l_m0;}
	double Getdl_m(){return dl_m;}

	double GetMetabolicEnergyRate();
	double GetMetabolicEnergyRate_BHAR04();
	double GetMetabolicEnergyRate_HOUD06();
	void SetMass();

	double Geth_A(){return h_A;}
	double Geth_M(){return h_M;}
	double Geth_SL(){return h_SL;}
	double GetW(){return w;}

private:
	double mSimTimeStep;
	double mConTimeStep;
	double h_A, h_M, h_SL, w;

	double l_mt,l_mt_max;
	double l_m;
	double l_m_prev;
	double activation;
	double activation_prev;
	double dl_m;
	double mMass;
	double mass_scaler;

	double f0;
	double l_mt0,l_m0,l_t0;
	double l_mt0_default, f0_default;
	double l_mt0_ratio, f0_ratio;

	double f_toe,e_toe,k_toe,k_lin,e_t0; //For g_t
	double k_pe,e_mo; //For g_pl
	double gamma; //For g_al

	double mMetabolicEnergyRate;
	double mMetabolicEnergyRate_BHAR04;
	double mMetabolicEnergyRate_HOUD06;

	double vcemax;
	double vcemin;
};

}
#endif

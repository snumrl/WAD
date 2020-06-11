#ifndef __MASS_WINDOW_H__
#define __MASS_WINDOW_H__
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace MASS
{

class Environment;
class Muscle;
class Window : public dart::gui::Win3D
{
public:
	Window(Environment* env);
	Window(Environment* env,const std::string& nn_path);
	Window(Environment* env,const std::string& nn_path,const std::string& muscle_nn_path);
	Window(Environment* env,const std::string& nn_path,const std::string& muscle_nn_path,const std::string& device_nn_path);

	void LoadMuscleNN(const std::string& muscle_nn_path);
	void LoadDeviceNN(const std::string& device_nn_path);

	void draw() override;
	void keyboard(unsigned char _key, int _x, int _y) override;
	void displayTimer(int _val) override;
	
	void Step();
	void Reset();

	void SetFocusing();
	void SetViewMatrix();
	void SetTrajectory();

	void DrawGround();
	void DrawCharacter();
	void DrawDevice();
	void DrawDeviceForce();
	void DrawDeviceSignals();
	void DrawTrajectory();
	void DrawProgressBar();
	
	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);

	void DrawMuscles(const std::vector<Muscle*>& muscles);
	void DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y);
	void DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y);
	
	void DrawQuads(double x, double y, double w, double h, Eigen::Vector4d color);
	void DrawString(double x, double y, std::string str);
	void DrawLine(double p1_x, double p1_y, double p2_x, double p2_y, Eigen::Vector4d color, double line_width);
	void DrawLineStrip(double x, double y, double offset_x, double offset_y, Eigen::Vector4d color, double line_width, std::deque<double>& data);
	
	Eigen::VectorXd GetActionFromNN();
	Eigen::VectorXd GetActionFromNN_Device();
	Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt);

private:

	p::object mm,mns,sys_module,nn_module,muscle_nn_module,device_nn_module;

	Environment* mEnv;
	bool mFocus;
	bool mSimulating;
	bool mDrawBVH;
	bool mDrawOBJ;
	bool mDrawShadow;
	bool mDrawDeviceForce;
	bool mDrawTrajectory;
	bool mDrawProgressBar;
	bool mNNLoaded;
	bool mMuscleNNLoaded;
	bool mDeviceNNLoaded;
	bool mOnDevice;

	bool mTalusL = false;
	bool mTalusR = false;
	
	Eigen::Affine3d mViewMatrix;
	std::vector<Eigen::Vector3d> mTrajectory;
	std::vector<Eigen::Vector3d> mFootprint;
};
};


#endif

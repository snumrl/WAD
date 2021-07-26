#ifndef MY_WORLD_NODE_H_
#define MY_WORLD_NODE_H_

#include <dart/dart.hpp>
#include <dart/gui/osg/osg.hpp>
#include <dart/utils/utils.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include "Environment.h"

using namespace dart::dynamics;
namespace py = pybind11;

class MyWorldNode : public dart::gui::osg::RealTimeWorldNode
{
public:
	/// Constructor
	MyWorldNode(MASS::Environment* env, const dart::simulation::WorldPtr& world);
	MyWorldNode(MASS::Environment* env, const dart::simulation::WorldPtr& world, const std::string& nn_path);
	MyWorldNode(MASS::Environment* env, const dart::simulation::WorldPtr& world, const std::string& nn_path, const std::string& nn_path2);
	MyWorldNode(MASS::Environment* env, const dart::simulation::WorldPtr& world, const std::string& nn_path, const std::string& muscle_nn_path, const std::string& device_nn_path);
	
	void LoadMuscleNN(const std::string& muscle_nn_path);
	void LoadDeviceNN(const std::string& device_nn_path);
  
	void refresh() override;
	void displayTimer(int _val);
	void Reset();
	void Step();

	Eigen::VectorXd GetActionFromNN();
	Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt);

  // Documentation inherited
//   void customPreStep() override;

//   void reset();

//   void pushForwardAtlas(double force = 500, int frames = 100);
//   void pushBackwardAtlas(double force = 500, int frames = 100);
//   void pushLeftAtlas(double force = 500, int frames = 100);
//   void pushRightAtlas(double force = 500, int frames = 100);

//   void switchToNormalStrideWalking();
//   void switchToShortStrideWalking();
//   void switchToNoControl();

//   void showShadow();
//   void hideShadow();

protected:
	MASS::Environment* mEnv;
//   std::unique_ptr<Controller> mController;
//   Eigen::Vector3d mExternalForce;
//   int mForceDuration;

private:
	py::object mm,mns,sys_module,nn_module,muscle_nn_module,device_nn_module,rms_module;

	int mDisplayIter = 0;

	bool mDevice_On;
	bool mMuscleNNLoaded;
};

#endif // DART_EXAMPLE_OSG_OSGATLASSIMBICON_ATLASSIMBICONWORLDNODE_HPP_
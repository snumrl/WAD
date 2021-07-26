#include "MyWorldNode.h"
#include <iostream>

// #include <osgShadow/ShadowMap>

//==============================================================================
MyWorldNode::
MyWorldNode(MASS::Environment* env, const dart::simulation::WorldPtr& world)
  : dart::gui::osg::RealTimeWorldNode(world), mEnv(env)    
{
    // mBackground[0] = 0.96;
	// mBackground[1] = 0.96;
	// mBackground[2] = 0.97;
	// mBackground[3] = 0.7;
	//     SetFocus();
	// mZoom = 0.30;
	// mNNLoaded = false;
	mDevice_On = env->GetCharacter()->GetDevice_OnOff();
	// mFootinterval.resize(20);
	// mDisplayTimeout = 1000 / 60.0;
    
    mm = py::module::import("__main__");
	mns = mm.attr("__dict__");
	
	py::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();
	sys_module = py::module::import("sys");
	sys_module.attr("path").attr("insert")(1, module_dir);
	
	py::exec("import torch",mns);
	py::exec("import torch.nn as nn",mns);
	py::exec("import torch.optim as optim",mns);
	py::exec("import torch.nn.functional as F",mns);
	py::exec("import numpy as np",mns);
	py::exec("from Model import *",mns);
	py::exec("from RunningMeanStd import *",mns);
}

MyWorldNode::
MyWorldNode(MASS::Environment* env, const dart::simulation::WorldPtr& world, const std::string& nn_path)
:MyWorldNode(env, world)
{
    // mNNLoaded = true;
   
   py::str str;
	str = ("num_state = "+std::to_string(mEnv->GetCharacter()->GetNumState())).c_str();
	py::exec(str, mns);
	str = ("num_action = "+std::to_string(mEnv->GetCharacter()->GetNumAction())).c_str();
	py::exec(str, mns);

	nn_module = py::eval("SimulationNN(num_state,num_action)", mns);
	py::object load = nn_module.attr("load");
	load(nn_path);

	rms_module = py::eval("RunningMeanStd()", mns);
	py::object load_rms = rms_module.attr("load2");
	load_rms(nn_path);
}

MyWorldNode::
MyWorldNode(MASS::Environment* env, const dart::simulation::WorldPtr& world, const std::string& nn_path, const std::string& nn_path2)
{
    if(env->GetUseMuscle())
		LoadMuscleNN(nn_path2);

	if(env->GetUseDeviceNN())
		LoadDeviceNN(nn_path2);
}

MyWorldNode::
MyWorldNode(MASS::Environment* env, const dart::simulation::WorldPtr& world, const std::string& nn_path, const std::string& muscle_nn_path, const std::string& device_nn_path)
{
    if(env->GetUseMuscle())
		LoadMuscleNN(muscle_nn_path);

	if(env->GetUseDevice())
		LoadDeviceNN(device_nn_path);
}

void
MyWorldNode::
LoadMuscleNN(const std::string& muscle_nn_path)
{
	mMuscleNNLoaded = true;

	py::str str;
	str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetCharacter()->GetNumTotalRelatedDofs())).c_str();
	py::exec(str,mns);
	str = ("num_actions = "+std::to_string(mEnv->GetCharacter()->GetNumActiveDof())).c_str();
	py::exec(str,mns);
	str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetNumMuscles())).c_str();
	py::exec(str,mns);

	// mMuscleNum = mEnv->GetCharacter()->GetNumMuscles();
	// mMuscleMapNum = mEnv->GetCharacter()->GetNumMusclesMap();

	muscle_nn_module = py::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);

	py::object load = muscle_nn_module.attr("load");
	load(muscle_nn_path);
}

void
MyWorldNode::
LoadDeviceNN(const std::string& device_nn_path)
{
	// mDeviceNNLoaded = true;
	// mDevice_On = true;

	py::str str;
	str = ("num_state_device = "+std::to_string(mEnv->GetCharacter()->GetDevice()->GetNumState())).c_str();
	py::exec(str,mns);
	str = ("num_action_device = "+std::to_string(mEnv->GetCharacter()->GetDevice()->GetNumAction())).c_str();
	py::exec(str,mns);

	device_nn_module = py::eval("SimulationNN(num_state_device,num_action_device)",mns);

	py::object load = device_nn_module.attr("load");
	load(device_nn_path);
}

void 
MyWorldNode::
refresh()
{
//   customPreRefresh();

    // clearChildUtilizationFlags();

    if (mSimulating)
    {
        this->Step();
    }

    refreshSkeletons();
    refreshSimpleFrames();

    // clearUnusedNodes();
}

// void
// MyWorldNode::
// displayTimer(int _val)
// {
// 	if(mSimulating){
// 		this->Step();
// 	}

// 	glutPostRedisplay();
// 	glutTimerFunc(mDisplayTimeout, refreshTimer, _val);
// }

void
MyWorldNode::
Reset()
{
	// mFootprint.clear();
	// mFootinterval.clear();
	// mFootinterval.resize(20);
	mEnv->Reset();
	// mDisplayIter = 0;
}

void
MyWorldNode::
Step()
{
	int num = mEnv->GetNumSteps()/2.0;
	Eigen::VectorXd action;
	Eigen::VectorXd action_device;

	if(mDisplayIter % 2 == 0)
	{
		// if(mNNLoaded)
		// 	action = GetActionFromNN();
		// else
		// 	action = Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetNumAction());

		// if(mDeviceNNLoaded)
		// 	action_device = GetActionFromNN_Device();
        action = GetActionFromNN();

		mEnv->SetAction(action);
		// if(mDeviceNNLoaded)
		// 	mEnv->GetDevice()->SetAction(action_device);

		// this->Write();
		mDisplayIter = 0;
	}

	if(mEnv->GetUseMuscle())
	{
		int inference_per_sim = 2;
		for(int i=0; i<num; i+=inference_per_sim){
			Eigen::VectorXd mt = mEnv->GetCharacter()->GetMuscleTorques();
			mEnv->GetCharacter()->SetActivationLevels(GetActivationFromNN(mt));
			for(int j=0; j<inference_per_sim; j++)
				mEnv->Step(mDevice_On, true);
		}
	}
	else
	{
		for(int i=0; i<num; i++){
			// this->Write();
			mEnv->Step(mDevice_On, true);
		}
	}

	mEnv->GetReward();
	// this->SetTrajectory();
	mDisplayIter++;
	// glutPostRedisplay();
}

py::array_t<float> toNumPyArray(const Eigen::VectorXd& vec)
{
	int n = vec.rows();
	py::array_t<float> array = py::array_t<float>(n);

	auto array_buf = array.request(true);
	float* dest = reinterpret_cast<float*>(array_buf.ptr);
	for(int i =0;i<n;i++)
	{
		dest[i] = vec[i];
	}

	return array;
}

Eigen::VectorXd
MyWorldNode::
GetActionFromNN()
{
	Eigen::VectorXd state = mEnv->GetCharacter()->GetState();
	py::array_t<float> state_np = py::array_t<float>(state.rows());
	py::buffer_info state_buf = state_np.request(true);
	float* dest = reinterpret_cast<float*>(state_buf.ptr);

	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];

	py::object apply;
	apply = rms_module.attr("apply_no_update");
	py::object state_np_tmp = apply(state_np);
	py::array_t<float> state_np_ = py::array_t<float>(state_np_tmp);

	py::object get_action;
	get_action = nn_module.attr("get_action");
	py::object temp = get_action(state_np_);
	py::array_t<float> action_np = py::array_t<float>(temp);

	py::buffer_info action_buf = action_np.request(true);
	float* srcs = reinterpret_cast<float*>(action_buf.ptr);

	Eigen::VectorXd action(mEnv->GetCharacter()->GetNumAction());
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];

	return action;
}

Eigen::VectorXd
MyWorldNode::
GetActivationFromNN(const Eigen::VectorXd& mt)
{
	if(!mMuscleNNLoaded)
	{
		mEnv->GetCharacter()->GetDesiredTorques();
		return Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
	}

	py::object get_activation = muscle_nn_module.attr("get_activation");
	mEnv->GetCharacter()->SetDesiredTorques();
	Eigen::VectorXd dt = mEnv->GetCharacter()->GetDesiredTorques();
	py::array_t<float> mt_np = toNumPyArray(mt);
	py::array_t<float> dt_np = toNumPyArray(dt);
	py::array_t<float> activation_np = get_activation(mt_np,dt_np);
	py::buffer_info activation_np_buf = activation_np.request(false);
	float* srcs = reinterpret_cast<float*>(activation_np_buf.ptr);

	Eigen::VectorXd activation(mEnv->GetCharacter()->GetMuscles().size());
	for(int i=0; i<activation.rows(); i++){
		activation[i] = srcs[i];
	}

	return activation;
}


// //==============================================================================
// void MyWorldNode::customPreStep()
// {
//   auto pelvis = mController->getAtlasRobot()->getBodyNode("pelvis");
//   pelvis->addExtForce(mExternalForce);
//   mController->update();

//   if (mForceDuration > 0)
//     mForceDuration--;
//   else
//     mExternalForce.setZero();
// }

// //==============================================================================
// void MyWorldNode::reset()
// {
//   mExternalForce.setZero();
//   mController->resetRobot();
// }

// //==============================================================================
// void MyWorldNode::pushForwardAtlas(double force, int frames)
// {
//   mExternalForce.x() = force;
//   mForceDuration = frames;
// }

// //==============================================================================
// void MyWorldNode::pushBackwardAtlas(double force, int frames)
// {
//   mExternalForce.x() = -force;
//   mForceDuration = frames;
// }

// //==============================================================================
// void MyWorldNode::pushLeftAtlas(double force, int frames)
// {
//   mExternalForce.z() = force;
//   mForceDuration = frames;
// }

// //==============================================================================
// void MyWorldNode::pushRightAtlas(double force, int frames)
// {
//   mExternalForce.z() = -force;
//   mForceDuration = frames;
// }

// //==============================================================================
// void MyWorldNode::switchToNormalStrideWalking()
// {
//   mController->changeStateMachine("walking", mWorld->getTime());
// }

// //==============================================================================
// void MyWorldNode::switchToShortStrideWalking()
// {
//   mController->changeStateMachine("running", mWorld->getTime());
// }

// //==============================================================================
// void MyWorldNode::switchToNoControl()
// {
//   mController->changeStateMachine("standing", mWorld->getTime());
// }

// //==============================================================================
// void MyWorldNode::showShadow()
// {
//   auto shadow
//       = dart::gui::osg::WorldNode::createDefaultShadowTechnique(mViewer);
//   if (auto sm = dynamic_cast<::osgShadow::ShadowMap*>(shadow.get()))
//   {
//     auto mapResolution = static_cast<short>(std::pow(2, 12));
//     sm->setTextureSize(::osg::Vec2s(mapResolution, mapResolution));
//   }

//   setShadowTechnique(shadow);
// }

// //==============================================================================
// void MyWorldNode::hideShadow()
// {
//   setShadowTechnique(nullptr);
// }

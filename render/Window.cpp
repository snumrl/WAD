#include "Window.h"
#include "Environment.h"
#include "Character.h"
#include "Device.h"
#include "BVH.h"
#include "Utils.h"
#include "Muscle.h"
#include <iostream>
#include <deque>
#include <ctime>
using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;
using namespace std;
Eigen::Matrix3d
R_x(double x)
{
	double cosa = cos(x*3.141592/180.0);
	double sina = sin(x*3.141592/180.0);
	Eigen::Matrix3d R;
	R<< 1,0     ,0    ,
		0,cosa  ,-sina,
		0,sina  ,cosa ;
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
	R<< cosa,-sina,0,
		sina,cosa ,0,
		0   ,0    ,1;
	return R;
}

Eigen::Vector4d white(0.9, 0.9, 0.9, 1.0);
Eigen::Vector4d black(0.1, 0.1, 0.1, 1.0);
Eigen::Vector4d grey(0.6, 0.6, 0.6, 1.0);
Eigen::Vector4d red(0.8, 0.2, 0.2, 1.0);
Eigen::Vector4d green(0.2, 0.8, 0.2, 1.0);
Eigen::Vector4d blue(0.2, 0.2, 0.8, 1.0);
Eigen::Vector4d yellow(0.8, 0.8, 0.2, 1.0);
Eigen::Vector4d purple(0.8, 0.2, 0.8, 1.0);

Eigen::Vector4d white_trans(0.9, 0.9, 0.9, 0.2);
Eigen::Vector4d black_trans(0.1, 0.1, 0.1, 0.2);
Eigen::Vector4d grey_trans(0.6, 0.6, 0.6, 0.2);
Eigen::Vector4d red_trans(0.8, 0.2, 0.2, 0.2);
Eigen::Vector4d green_trans(0.2, 0.8, 0.2, 0.2);
Eigen::Vector4d blue_trans(0.2, 0.2, 0.8, 0.2);
Eigen::Vector4d yellow_trans(0.8, 0.8, 0.2, 0.2);
Eigen::Vector4d purple_trans(0.8, 0.2, 0.8, 0.2);

Window::
Window(Environment* env)
	:mEnv(env),mFocus(true),mSimulating(false),mDrawCharacter(true),mDrawTarget(false),mDrawOBJ(false),mDrawShadow(true),mMuscleNNLoaded(false),mDeviceNNLoaded(false),mDevice_On(false),isDrawCharacter(false),isDrawTarget(false),isDrawDevice(false),mDrawArrow(false),mDrawGraph(false),mGraphMode(0),mCharacterMode(0),mParamMode(0),mViewMode(0),mDrawParameter(true),mTalusL(false),mTalusR(false)
{
	mBackground[0] = 0.96;
	mBackground[1] = 0.96;
	mBackground[2] = 0.97;
	mBackground[3] = 0.7;
	SetFocus();
	mZoom = 0.30;

	mNNLoaded = false;
	mDevice_On = env->GetCharacter()->GetDevice_OnOff();
	mFootinterval.resize(20);

	mm = p::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = p::import("sys");
	p::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();
	sys_module.attr("path").attr("insert")(1, module_dir);
	p::exec("import torch",mns);
	p::exec("import torch.nn as nn",mns);
	p::exec("import torch.optim as optim",mns);
	p::exec("import torch.nn.functional as F",mns);
	p::exec("import torchvision.transforms as T",mns);
	p::exec("import numpy as np",mns);
	p::exec("from Model import *",mns);
	p::exec("from RunningMeanStd import *",mns);
}

Window::
Window(Environment* env, const std::string& nn_path)
	:Window(env)
{
	mNNLoaded = true;

	boost::python::str str;
	str = ("num_state = "+std::to_string(mEnv->GetCharacter()->GetNumState())).c_str();
	p::exec(str, mns);
	str = ("num_action = "+std::to_string(mEnv->GetCharacter()->GetNumAction())).c_str();
	p::exec(str, mns);

	nn_module = p::eval("SimulationNN(num_state,num_action)", mns);
	p::object load = nn_module.attr("load");
	load(nn_path);

	rms_module = p::eval("RunningMeanStd()", mns);
	p::object load_rms = rms_module.attr("load2");
	load_rms(nn_path);

}

Window::
Window(Environment* env,const std::string& nn_path, const std::string& nn_path2)
	:Window(env, nn_path)
{
	if(env->GetUseMuscle())
		LoadMuscleNN(nn_path2);

	if(env->GetUseDeviceNN())
		LoadDeviceNN(nn_path2);
}

Window::
Window(Environment* env,const std::string& nn_path,const std::string& muscle_nn_path, const std::string& device_nn_path)
	:Window(env, nn_path)
{
	if(env->GetUseMuscle())
		LoadMuscleNN(muscle_nn_path);

	if(env->GetUseDevice())
		LoadDeviceNN(device_nn_path);
}

void
Window::
LoadMuscleNN(const std::string& muscle_nn_path)
{
	mMuscleNNLoaded = true;

	boost::python::str str;
	str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetCharacter()->GetNumTotalRelatedDofs())).c_str();
	p::exec(str,mns);
	str = ("num_actions = "+std::to_string(mEnv->GetCharacter()->GetNumAction())).c_str();
	p::exec(str,mns);
	str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetNumMuscles())).c_str();
	p::exec(str,mns);

	muscle_nn_module = p::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);

	p::object load = muscle_nn_module.attr("load");
	load(muscle_nn_path);
	std::cout << "load muscle" << std::endl;
}

void
Window::
LoadDeviceNN(const std::string& device_nn_path)
{
	mDeviceNNLoaded = true;
	mDevice_On = true;

	boost::python::str str;
	str = ("num_state_device = "+std::to_string(mEnv->GetCharacter()->GetDevice()->GetNumState())).c_str();
	p::exec(str,mns);
	str = ("num_action_device = "+std::to_string(mEnv->GetCharacter()->GetDevice()->GetNumAction())).c_str();
	p::exec(str,mns);

	device_nn_module = p::eval("SimulationNN(num_state_device,num_action_device)",mns);

	p::object load = device_nn_module.attr("load");
	load(device_nn_path);
}

void
Window::
ParamChange(bool b)
{
	if(!mEnv->GetUseAdaptiveSampling())
		return;
	Eigen::VectorXd min_v = mEnv->GetMinV();
	Eigen::VectorXd max_v = mEnv->GetMaxV();
	if(b)
	{
		if(mParamMode == 1)
		{
			double m_ratio = mEnv->GetCharacter()->GetMassRatio();
			m_ratio += 0.10;
			if(m_ratio > max_v[0])
				m_ratio = max_v[0];
			mEnv->GetCharacter()->SetMassRatio(m_ratio);
		}
		else if(mParamMode == 2)
		{
			double f_ratio = mEnv->GetCharacter()->GetForceRatio();
			f_ratio += 0.05;
			if(f_ratio > max_v[1])
				f_ratio = max_v[1];
			mEnv->GetCharacter()->SetForceRatio(f_ratio);
		}
		else if(mParamMode ==3)
		{
			double s_ratio = mEnv->GetCharacter()->GetSpeedRatio();
			s_ratio += 0.1;
			if(s_ratio > max_v[2])
				s_ratio = max_v[2];
			mEnv->GetCharacter()->SetSpeedRatio(s_ratio);
			mEnv->GetCharacter()->SetBVHidx(s_ratio);
		}
		else if(mParamMode == 4)
		{
			double k_ = mEnv->GetDevice()->GetK_();
			k_ += 2.0;
			if(k_ > max_v[3]*30.0)
				k_ = max_v[3]*30.0;
			mEnv->GetDevice()->SetK_(k_);
		}
		else if(mParamMode == 5)
		{
			double t_ = mEnv->GetDevice()->GetDelta_t();
			t_ += 0.01;
			if(t_ > max_v[4])
				t_ = max_v[4];
			mEnv->GetDevice()->SetDelta_t(t_);
		}
	}
	else
	{
		if(mParamMode == 1)
		{
			double m_ratio = mEnv->GetCharacter()->GetMassRatio();
			m_ratio -= 0.10;
			if(m_ratio < min_v[0])
				m_ratio = min_v[0];
			mEnv->GetCharacter()->SetMassRatio(m_ratio);
		}
		else if(mParamMode == 2)
		{
			double f_ratio = mEnv->GetCharacter()->GetForceRatio();
			f_ratio -= 0.05;
			if(f_ratio < min_v[1])
				f_ratio = min_v[1];
			mEnv->GetCharacter()->SetForceRatio(f_ratio);
		}
		else if(mParamMode ==3)
		{
			double s_ratio = mEnv->GetCharacter()->GetSpeedRatio();
			s_ratio -= 0.1;
			if(s_ratio < min_v[2])
				s_ratio = min_v[2];
			mEnv->GetCharacter()->SetSpeedRatio(s_ratio);
			mEnv->GetCharacter()->SetBVHidx(s_ratio);
		}
		else if(mParamMode == 4)
		{
			double k_ = mEnv->GetDevice()->GetK_();
			k_ -= 2.0;
			if(k_ < min_v[3]*30.0)
				k_ = min_v[3]*30.0;
			mEnv->GetDevice()->SetK_(k_);
		}
		else if(mParamMode == 5)
		{
			double t_ = mEnv->GetDevice()->GetDelta_t();
			t_ -= 0.01;
			if(t_ < min_v[4])
				t_ = min_v[4];
			mEnv->GetDevice()->SetDelta_t(t_);
		}
	}
}

void
Window::
keyboard(unsigned char _key, int _x, int _y)
{
	double f_ = 0.0;
	switch (_key)
	{
	case 'r': this->Reset();break;
	case 's': this->Step();break;
	case 'f': mFocus = !mFocus;break;
	case 'o': mDrawOBJ = !mDrawOBJ;break;
	case 'g': mDrawGraph = !mDrawGraph;break;
	case 'a': mDrawArrow = !mDrawArrow;break;
	case 't': mDrawTarget = !mDrawTarget;break;
	case ' ': mSimulating = !mSimulating;break;
	case 'c': mDrawCharacter = !mDrawCharacter;break;
	case 'p': mDrawParameter = !mDrawParameter;break;
	case 'd':
		if(mEnv->GetUseDevice())
			mDevice_On = !mDevice_On;
		break;
	case 'v' : mViewMode = (mViewMode+1)%4;break;
	case '\t': mGraphMode = (mGraphMode+1)%6;break;
	case '`' : mCharacterMode = (mCharacterMode+1)%2; break;
	case '1' : mParamMode = 1; break;
	case '2' : mParamMode = 2; break;
	case '3' : mParamMode = 3; break;
	case '4' : mParamMode = 4; break;
	case '5' : mParamMode = 5; break;
	case '+' : ParamChange(true); break;
	case '-' : ParamChange(false); break;
	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}
}

void
Window::
displayTimer(int _val)
{
	if(mSimulating)
		Step();
	glutPostRedisplay();
	glutTimerFunc(mDisplayTimeout, refreshTimer, _val);
}

void
Window::
Reset()
{
	mFootprint.clear();
	mFootinterval.clear();
	mFootinterval.resize(20);
	mEnv->Reset();
}

void
Window::
Step()
{
	int num = mEnv->GetNumSteps();
	Eigen::VectorXd action;
	Eigen::VectorXd action_device;

	if(mNNLoaded)
		action = GetActionFromNN();
	else
		action = Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetNumAction());

	if(mDeviceNNLoaded)
		action_device = GetActionFromNN_Device();

	mEnv->SetAction(action);
	if(mDeviceNNLoaded)
		mEnv->GetDevice()->SetAction(action_device);

	if(mEnv->GetUseMuscle())
	{
		int inference_per_sim = 1;
		for(int i=0; i<num; i+=inference_per_sim){
			Eigen::VectorXd mt = mEnv->GetCharacter()->GetMuscleTorques();
			mEnv->GetCharacter()->SetActivationLevels(GetActivationFromNN(mt));
			for(int j=0; j<inference_per_sim; j++)
				mEnv->Step(mDevice_On);
		}
	}
	else
	{
		for(int i=0; i<num; i++)
			mEnv->Step(mDevice_On);
	}

	mEnv->GetReward();
	// this->SetTrajectory();

	glutPostRedisplay();
}

void
Window::
SetTrajectory()
{
	const SkeletonPtr& skel = mEnv->GetCharacter()->GetSkeleton();
	const BodyNode* talusR = skel->getBodyNode("TalusR");
	const BodyNode* talusL = skel->getBodyNode("TalusL");

	if(talusR->getCOM()[1] > 0.035)
		mTalusR = false;
	if(talusR->getCOM()[1] < 0.035 && !mTalusR)
	{
		mFootprint.push_back(talusR->getCOM());
		int idx = mFootprint.size()-1;
		if(idx > 0)
		{
			Eigen::Vector3d prev = mFootprint[idx-1];
			Eigen::Vector3d cur = mFootprint[idx];
			double len = (cur-prev).norm();
			mFootinterval.pop_back();
			mFootinterval.push_front(len);
		}
		mTalusR = true;
	}

	if(talusL->getCOM()[1] > 0.035)
		mTalusL = false;
	if(talusL->getCOM()[1] < 0.035 && !mTalusL)
	{
		mFootprint.push_back(talusL->getCOM());
		int idx = mFootprint.size()-1;
		if(idx > 0)
		{
			Eigen::Vector3d prev = mFootprint[idx-1];
			Eigen::Vector3d cur = mFootprint[idx];
			double len = (cur-prev).norm();
			mFootinterval.pop_back();
			mFootinterval.push_front(len);
		}
		mTalusL = true;
	}
}

void
Window::
SetFocus()
{
	if(mFocus)
	{
		mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
		mTrans *= 1000.0;
		Eigen::Quaterniond origin_r = mTrackBall.getCurrQuat();
		if (mViewMode == 0)
		{
			mTrackBall.setQuaternion(Eigen::Quaterniond::Identity());
			Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(1 * 0.05 * M_PI, Eigen::Vector3d::UnitX()));
			mTrackBall.setQuaternion(r);
		}
		else if (mViewMode == 1 && Eigen::AngleAxisd(origin_r).angle() < 0.5 * M_PI)
		{
			Eigen::Vector3d axis(0.0, cos(0.05*M_PI), sin(0.05*M_PI));
			Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(1 * 0.01 * M_PI, axis)) * origin_r;
			mTrackBall.setQuaternion(r);
			}
		else if (mViewMode == 2 && Eigen::AngleAxisd(origin_r).axis()[1] > 0)
		{
			Eigen::Vector3d axis(0.0, cos(0.05*M_PI), sin(0.05*M_PI));
			Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(-1 * 0.01 * M_PI, axis)) * origin_r;
			mTrackBall.setQuaternion(r);
		}
	}
}

void
Window::
SetViewMatrix()
{
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);

	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<
	matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<
	matrix[12],matrix[13],matrix[14];

	mViewMatrix.linear() = A;
	mViewMatrix.translation() = b;
}

void
Window::
draw()
{
	SetViewMatrix();

	DrawGround();
	DrawCharacter();
	if(mEnv->GetUseDevice())
		DrawDevice();

	if(mEnv->GetNumParamState() > 0 && mDrawParameter)
		DrawParameter();

	// DrawTrajectory();
	// DrawStride();
	SetFocus();
}

void
Window::
DrawContactForce()
{
	DrawGLBegin();

	double fL = (mEnv->GetCharacter()->GetContactForcesNormAvg()).at(0);
	double fR = (mEnv->GetCharacter()->GetContactForcesNormAvg()).at(1);
	bool big = true;

	DrawString(0.70, 0.44, big, "Contact L : " + std::to_string(fL));
	DrawString(0.70, 0.41, big, "Contact R : " + std::to_string(fR));

	DrawGLEnd();
}

void
Window::
DrawCoT()
{
	DrawGLBegin();

	double cot = mEnv->GetCharacter()->GetCoT();
	bool big = true;

	DrawString(0.70, 0.47, big, "CoT : " + std::to_string(cot));

	DrawGLEnd();
}

void
Window::
DrawVelocity()
{
	DrawGLBegin();

	double vel = mEnv->GetCharacter()->GetCurVelocity();
	double vel_h = vel;
	bool big = true;

	DrawString(0.70, 0.50, big, "Velocity : " + std::to_string(vel_h) + " m/s");

	DrawGLEnd();
}

void
Window::
DrawParameter()
{
	DrawGLBegin();

	double w = 0.20;
	double h = 0.34;
	double h_offset = 0.20;
	double x = 0.69;
	double y = 0.62;

	DrawQuads(x, y, w, h, white);

	if(mEnv->GetCharacter()->GetNumParamState() > 0)
	{
		Eigen::VectorXd min_v = mEnv->GetCharacter()->GetMinV();
		Eigen::VectorXd max_v = mEnv->GetCharacter()->GetMaxV();

		double m_ratio = mEnv->GetCharacter()->GetMassRatio();
		DrawQuads(x+0.01, y+0.01, 0.02, (m_ratio)*h_offset, red);
		DrawQuads(x+0.01, y+0.01+(m_ratio)*h_offset, 0.02, (max_v[0]-m_ratio)*h_offset, red_trans);

		double f_ratio = mEnv->GetCharacter()->GetForceRatio();
		DrawQuads(x+0.05, y+0.01, 0.02, (f_ratio)*h_offset, yellow);
		DrawQuads(x+0.05, y+0.01+(f_ratio)*h_offset, 0.02, (max_v[1]-f_ratio)*h_offset, yellow_trans);

		double s_ratio = mEnv->GetCharacter()->GetSpeedRatio();
		DrawQuads(x+0.09, y+0.01, 0.02, (s_ratio)*h_offset, green);
		DrawQuads(x+0.09, y+0.01+(s_ratio)*h_offset, 0.02, (max_v[2]-s_ratio)*h_offset, green_trans);

		DrawString(x+0.00, y+(m_ratio)*h_offset+0.02, std::to_string(m_ratio));
		DrawString(x+0.00, y-0.02, "Mass");

		DrawString(x+0.05, y+(f_ratio)*h_offset+0.02, std::to_string(f_ratio));
		DrawString(x+0.04, y-0.02, "Force");

		DrawString(x+0.09, y+(s_ratio)*h_offset+0.02, std::to_string(s_ratio));
		DrawString(x+0.075, y-0.02, "Speed");
	}

	if(mEnv->GetUseDevice() && mEnv->GetDevice()->GetNumParamState() > 0)
	{
		Eigen::VectorXd min_v_dev = mEnv->GetDevice()->GetMinV();
		Eigen::VectorXd max_v_dev = mEnv->GetDevice()->GetMaxV();

		double k_ = mEnv->GetDevice()->GetK_();
		DrawQuads(x+0.13, y+0.01, 0.02, (k_/30.0)*h_offset, blue);
		DrawQuads(x+0.13, y+0.01+(k_/30.0)*h_offset, 0.02, (max_v_dev[0]-k_/30.0)*h_offset, blue_trans);

		double t_ = mEnv->GetDevice()->GetDelta_t();
		DrawQuads(x+0.17, y+0.01, 0.02, 3.333*t_*h_offset, purple);
		DrawQuads(x+0.17, y+0.01+3.333*t_*h_offset, 0.02, 3.333*(max_v_dev[1]-t_)*h_offset, purple_trans);

		DrawString(x+0.13, y+(k_/30.0)*h_offset+0.02, std::to_string(k_));
		DrawString(x+0.12, y-0.02, "Device");

		DrawString(x+0.17, y+3.0*t_*h_offset+0.02, std::to_string(t_));
		DrawString(x+0.165, y-0.02, "Delta t");
	}

	DrawGLEnd();
}

void
Window::
DrawGround()
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glDisable(GL_LIGHTING);

	auto ground = mEnv->GetGround();
	float y = ground->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(ground->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;

	int count = 0;
	double w = 2.0;
	double h = 2.0;
	for(double x=-100.0; x<=100.0; x+=2.0)
	{
		for(double z=-100.0; z<=100.0; z+=2.0)
		{
			if(count%2==0)
				glColor3f(0.82, 0.80, 0.78);
			else
				glColor3f(0.72, 0.70, 0.68);

			glBegin(GL_QUADS);
			glVertex3f(x  , y, z  );
			glVertex3f(x+w, y, z  );
			glVertex3f(x+w, y, z+h);
			glVertex3f(x  , y, z+h);
			glEnd();

			count++;
		}
	}
	glEnable(GL_LIGHTING);
}

void
Window::
DrawCharacter()
{
	SkeletonPtr skeleton = mEnv->GetCharacter()->GetSkeleton();

	if(mDrawCharacter)
	{
		isDrawCharacter = true;
		DrawSkeleton(skeleton);
		if(mEnv->GetUseMuscle())
			DrawMuscles(mEnv->GetCharacter()->GetMuscles());
		isDrawCharacter = false;
	}

	if(mDrawTarget)
		DrawTarget();

	if(mDrawGraph){
		if(!mEnv->GetUseDevice())
			DrawFemurSignals();
		DrawTorques();
		DrawReward();
	}
	else{
		this->DrawVelocity();
		this->DrawCoT();
		this->DrawContactForce();
	}
}

void
Window::
DrawSkeleton(const SkeletonPtr& skel)
{
	DrawBodyNode(skel->getRootBodyNode());
}

void
Window::
DrawBodyNode(const BodyNode* bn)
{
	if(!bn)
		return;
	if(!mRI)
		return;

	mRI->pushMatrix();
	mRI->transform(bn->getRelativeTransform());

	auto sns = bn->getShapeNodesWith<VisualAspect>();
	for(const auto& sn : sns)
		DrawShapeFrame(sn);

	for(const auto& et : bn->getChildEntities())
		DrawEntity(et);

	mRI->popMatrix();
}

void
Window::
DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	if(!mRI)
		return;

	const auto& va = sf->getVisualAspect();

	if(!va || va->isHidden())
		return;

	mRI->pushMatrix();
	mRI->transform(sf->getRelativeTransform());

	mColor = va->getRGBA();
	if(isDrawCharacter && mDrawOBJ)
		mColor << 0.75, 0.75, 0.75, 1.0;
	if(isDrawTarget)
		mColor << 1.0, 0.6, 0.6, 0.3;
	if(isDrawDevice)
		mColor << 0.3, 0.3, 0.3, 1.0;
	DrawShape(sf->getShape().get(), mColor);
	mRI->popMatrix();
}

void
Window::
DrawShape(const Shape* shape, const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	if(!mRI)
		return;

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	mRI->setPenColor(color);
	if(mDrawOBJ == false)
	{
		if (shape->is<SphereShape>())
		{
			const auto* sphere = static_cast<const SphereShape*>(shape);
			mRI->drawSphere(sphere->getRadius());
		}
		else if (shape->is<BoxShape>())
		{
			const auto* box = static_cast<const BoxShape*>(shape);
			mRI->drawCube(box->getSize());
		}
		else if (shape->is<CapsuleShape>())
		{
			const auto* capsule = static_cast<const CapsuleShape*>(shape);
			mRI->drawCapsule(capsule->getRadius(), capsule->getHeight());
		}
	}
	else
	{
		if (shape->is<MeshShape>())
		{
			const auto& mesh = static_cast<const MeshShape*>(shape);
			float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;

			mShapeRenderer.renderMesh(mesh, false, y, color);
		}
	}
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_DEPTH_TEST);
}

void
Window::
DrawEntity(const Entity* entity)
{
	if (!entity)
		return;
	const auto& bn = dynamic_cast<const BodyNode*>(entity);
	if(bn)
	{
		DrawBodyNode(bn);
		return;
	}

	const auto& sf = dynamic_cast<const ShapeFrame*>(entity);
	if(sf)
	{
		DrawShapeFrame(sf);
		return;
	}
}

void
Window::
DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y)
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glScalef(scale[0],scale[1],scale[2]);
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];

	Eigen::Affine3d M;
	M.linear() = A;
	M.translation() = b;
	M = (mViewMatrix.inverse()) * M;

	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(mViewMatrix.data());
	DrawAiMesh(mesh,mesh->mRootNode,M,y);
	glPopMatrix();
	glPopMatrix();
	glEnable(GL_LIGHTING);
}

void
Window::
DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y)
{
	unsigned int i;
	unsigned int n = 0, t;
	Eigen::Vector3d v;
	Eigen::Vector3d dir(0.4,0,-0.4);
	glColor3f(0.3,0.3,0.3);

	// update transform
	// draw all meshes assigned to this node
	for (; n < nd->mNumMeshes; ++n) {
		const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

		for (t = 0; t < mesh->mNumFaces; ++t) {
			const struct aiFace* face = &mesh->mFaces[t];
			GLenum face_mode;

			switch(face->mNumIndices) {
				case 1: face_mode = GL_POINTS; break;
				case 2: face_mode = GL_LINES; break;
				case 3: face_mode = GL_TRIANGLES; break;
				default: face_mode = GL_POLYGON; break;
			}
			glBegin(face_mode);
			for (i = 0; i < face->mNumIndices; i++)
			{
				int index = face->mIndices[i];

				v[0] = (&mesh->mVertices[index].x)[0];
				v[1] = (&mesh->mVertices[index].x)[1];
				v[2] = (&mesh->mVertices[index].x)[2];
				v = M*v;
				double h = v[1]-y;

				v += h*dir;

				v[1] = y+0.001;
				glVertex3f(v[0],v[1],v[2]);
			}
			glEnd();
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n) {
		DrawAiMesh(sc, nd->mChildren[n],M,y);
	}
}

void
Window::
DrawMuscles(const std::vector<Muscle*>& muscles)
{
	int count = 0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	for (auto muscle : muscles)
	{
        double a = muscle->GetActivation();
        Eigen::Vector4d color(1.0+(3.0*a), 1.0, 1.0, 1.0);
        glColor4dv(color.data());
	    mShapeRenderer.renderMuscle(muscle);
	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}

void
Window::
DrawTarget()
{
	isDrawTarget = true;

	Character* character = mEnv->GetCharacter();
	SkeletonPtr skeleton = character->GetSkeleton();

	Eigen::VectorXd cur_pos = skeleton->getPositions();

	skeleton->setPositions(character->GetTargetPositions());
	DrawBodyNode(skeleton->getRootBodyNode());

	skeleton->setPositions(cur_pos);

	isDrawTarget = false;
}

void
Window::
DrawTrajectory()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// Eigen::Vector4d color(0.5, 0.5, 0.5, 0.7);
	// mRI->setPenColor(color);
	// mRI->setLineWidth(8.0);

	// glBegin(GL_LINE_STRIP);
	// for(int i=0; i<mTrajectory.size(); i++)
	// {
	// 	glVertex3f(mTrajectory[i][0], 0.0, mTrajectory[i][2]);
	// }
	// glEnd();

	Eigen::Vector4d color1(0.5, 0.8, 0.5, 0.7);
	mRI->setPenColor(color1);
	for(int i=0; i<mFootprint.size(); i++)
	{
		glPushMatrix();
		glTranslatef(mFootprint[i][0], 0.0, mFootprint[i][2]);
		glutSolidCube(0.05);
		glPopMatrix();
	}

	glDisable(GL_COLOR_MATERIAL);
}

void
Window::
DrawFemurSignals()
{
	DrawGLBegin();

	double p_w = 0.30;
	double p_h = 0.14;
	double pl_x = 0.69;
	double pl_y = 0.84;
	double pr_x = 0.69;
	double pr_y = 0.68;

	double offset_x = 0.00024;
	double offset_y = 0.0006;
	double offset = 0.005;
	double ratio_y = 0.3;

	Character* character = mEnv->GetCharacter();
	std::deque<double> data_L_femur = character->GetSignals(0);
	std::deque<double> data_R_femur = character->GetSignals(1);

	DrawBaseGraph(pl_x, pl_y, p_w, p_h, ratio_y, offset, "Femur L");
	DrawLineStrip(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, red, 2.0, data_L_femur);
	DrawStringMax(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, data_L_femur, red);

	DrawBaseGraph(pr_x, pr_y, p_w, p_h, ratio_y, offset, "Femur R");
	DrawLineStrip(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset, red, 2.0, data_R_femur);
	DrawStringMax(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset, data_R_femur, red);

	DrawGLEnd();
}

void
Window::
DrawTorques()
{
	DrawGLBegin();

	double p_w = 0.30;
	double p_h = 0.14;
	double p_x = 0.01;
	double p_y = 0.84;
	double offset_y = 0.16;

	mTorques = mEnv->GetCharacter()->GetTorques();
	if(mGraphMode == 0){
		DrawTorqueGraph("Lower", 0, p_w, p_h, p_x, p_y-0*offset_y);
		DrawTorqueGraph("FemurL_x", 6, p_w, p_h, p_x, p_y-1*offset_y);
		DrawTorqueGraph("FemurL_y", 7, p_w, p_h, p_x, p_y-2*offset_y);
		DrawTorqueGraph("FemurL_z", 8, p_w, p_h, p_x, p_y-3*offset_y);
		DrawTorqueGraph("FemurR_x", 15, p_w, p_h, p_x, p_y-4*offset_y);
		DrawTorqueGraph("FemurR_y", 16, p_w, p_h, p_x, p_y-5*offset_y);
		// DrawTorqueGraph("FemurL_x", 6, p_w, p_h, p_x, p_y-0*offset_y);
		// DrawTorqueGraph("FemurL_y", 7, p_w, p_h, p_x, p_y-1*offset_y);
		// DrawTorqueGraph("FemurL_z", 8, p_w, p_h, p_x, p_y-2*offset_y);
		// DrawTorqueGraph("FemurR_x", 15, p_w, p_h, p_x, p_y-3*offset_y);
		// DrawTorqueGraph("FemurR_y", 16, p_w, p_h, p_x, p_y-4*offset_y);
		// DrawTorqueGraph("FemurR_z", 17, p_w, p_h, p_x, p_y-5*offset_y);
	}
	else if(mGraphMode == 1){
		DrawTorqueGraph("Tibia_L", 9, p_w, p_h, p_x, p_y-0*offset_y);
		DrawTorqueGraph("Tibia_R", 18, p_w, p_h, p_x, p_y-1*offset_y);
		DrawTorqueGraph("Elbow_L", 42, p_w, p_h, p_x, p_y-2*offset_y);
		DrawTorqueGraph("Elbow_R", 52, p_w, p_h, p_x, p_y-3*offset_y);
		DrawTorqueGraph("Thumb_L", 13, p_w, p_h, p_x, p_y-4*offset_y);
		DrawTorqueGraph("Thumb_R", 22, p_w, p_h, p_x, p_y-5*offset_y);
	}
	else if(mGraphMode == 2){
		DrawTorqueGraph("TalusL_x", 10, p_w, p_h, p_x, p_y-0*offset_y);
		DrawTorqueGraph("TalusL_y", 11, p_w, p_h, p_x, p_y-1*offset_y);
		DrawTorqueGraph("TalusL_z", 12, p_w, p_h, p_x, p_y-2*offset_y);
		DrawTorqueGraph("TalusR_x", 19, p_w, p_h, p_x, p_y-3*offset_y);
		DrawTorqueGraph("TalusR_y", 20, p_w, p_h, p_x, p_y-4*offset_y);
		DrawTorqueGraph("TalusR_z", 21, p_w, p_h, p_x, p_y-5*offset_y);
	}
	else if(mGraphMode == 3){
		DrawTorqueGraph("Spine_x", 24, p_w, p_h, p_x, p_y-0*offset_y);
		DrawTorqueGraph("Spine_y", 25, p_w, p_h, p_x, p_y-1*offset_y);
		DrawTorqueGraph("Spine_z", 26, p_w, p_h, p_x, p_y-2*offset_y);
		DrawTorqueGraph("Torso_x", 27, p_w, p_h, p_x, p_y-3*offset_y);
		DrawTorqueGraph("Torso_y", 28, p_w, p_h, p_x, p_y-4*offset_y);
		DrawTorqueGraph("Torso_z", 29, p_w, p_h, p_x, p_y-5*offset_y);
	}
	else if(mGraphMode == 4){
		DrawTorqueGraph("ShoulderL_x", 36, p_w, p_h, p_x, p_y-0*offset_y);
		DrawTorqueGraph("ShoulderL_y", 37, p_w, p_h, p_x, p_y-1*offset_y);
		DrawTorqueGraph("ShoulderL_z", 38, p_w, p_h, p_x, p_y-2*offset_y);
		DrawTorqueGraph("ShoulderR_x", 46, p_w, p_h, p_x, p_y-3*offset_y);
		DrawTorqueGraph("ShoulderR_Y", 47, p_w, p_h, p_x, p_y-4*offset_y);
		DrawTorqueGraph("ShoulderR_z", 48, p_w, p_h, p_x, p_y-5*offset_y);
	}
	else if(mGraphMode == 5){
		DrawTorqueGraph("ArmL_x", 39, p_w, p_h, p_x, p_y-0*offset_y);
		DrawTorqueGraph("ArmL_y", 40, p_w, p_h, p_x, p_y-1*offset_y);
		DrawTorqueGraph("ArmL_z", 41, p_w, p_h, p_x, p_y-2*offset_y);
		DrawTorqueGraph("ArmR_x", 49, p_w, p_h, p_x, p_y-3*offset_y);
		DrawTorqueGraph("ArmR_y", 50, p_w, p_h, p_x, p_y-4*offset_y);
		DrawTorqueGraph("ArmR_z", 51, p_w, p_h, p_x, p_y-5*offset_y);
	}
	// if(mGraphMode == 0){
	// 	DrawTorqueGraph("FemurL_x", 6, p_w, p_h, p_x, p_y-0*offset_y);
	// 	DrawTorqueGraph("FemurL_y", 7, p_w, p_h, p_x, p_y-1*offset_y);
	// 	DrawTorqueGraph("FemurL_z", 8, p_w, p_h, p_x, p_y-2*offset_y);
	// 	DrawTorqueGraph("FemurR_x", 13, p_w, p_h, p_x, p_y-3*offset_y);
	// 	DrawTorqueGraph("FemurR_y", 14, p_w, p_h, p_x, p_y-4*offset_y);
	// 	DrawTorqueGraph("FemurR_z", 15, p_w, p_h, p_x, p_y-5*offset_y);
	// }
	// else if(mGraphMode == 1){
	// 	DrawTorqueGraph("Tibia_L", 9, p_w, p_h, p_x, p_y-0*offset_y);
	// 	DrawTorqueGraph("Tibia_R", 16, p_w, p_h, p_x, p_y-1*offset_y);
	// 	DrawTorqueGraph("Elbow_L", 29, p_w, p_h, p_x, p_y-2*offset_y);
	// 	DrawTorqueGraph("Elbow_R", 36, p_w, p_h, p_x, p_y-3*offset_y);
	// }
	// else if(mGraphMode == 2){
	// 	DrawTorqueGraph("TalusL_x", 10, p_w, p_h, p_x, p_y-0*offset_y);
	// 	DrawTorqueGraph("TalusL_y", 11, p_w, p_h, p_x, p_y-1*offset_y);
	// 	DrawTorqueGraph("TalusL_z", 12, p_w, p_h, p_x, p_y-2*offset_y);
	// 	DrawTorqueGraph("TalusR_x", 17, p_w, p_h, p_x, p_y-3*offset_y);
	// 	DrawTorqueGraph("TalusR_y", 18, p_w, p_h, p_x, p_y-4*offset_y);
	// 	DrawTorqueGraph("TalusR_z", 19, p_w, p_h, p_x, p_y-5*offset_y);
	// }
	// else if(mGraphMode == 3){
	// 	DrawTorqueGraph("Torso_x", 20, p_w, p_h, p_x, p_y-0*offset_y);
	// 	DrawTorqueGraph("Torso_y", 21, p_w, p_h, p_x, p_y-1*offset_y);
	// 	DrawTorqueGraph("Torso_z", 22, p_w, p_h, p_x, p_y-2*offset_y);
	// 	DrawTorqueGraph("Neck_x", 23, p_w, p_h, p_x, p_y-3*offset_y);
	// 	DrawTorqueGraph("Neck_y", 24, p_w, p_h, p_x, p_y-4*offset_y);
	// 	DrawTorqueGraph("Neck_z", 25, p_w, p_h, p_x, p_y-5*offset_y);
	// }
	// else if(mGraphMode == 4){
	// 	DrawTorqueGraph("ShoulderL_x", 26, p_w, p_h, p_x, p_y-0*offset_y);
	// 	DrawTorqueGraph("ShoulderL_y", 27, p_w, p_h, p_x, p_y-1*offset_y);
	// 	DrawTorqueGraph("ShoulderL_z", 28, p_w, p_h, p_x, p_y-2*offset_y);
	// 	DrawTorqueGraph("ShoulderR_x", 33, p_w, p_h, p_x, p_y-3*offset_y);
	// 	DrawTorqueGraph("ShoulderR_Y", 34, p_w, p_h, p_x, p_y-4*offset_y);
	// 	DrawTorqueGraph("ShoulderR_z", 35, p_w, p_h, p_x, p_y-5*offset_y);
	// }
	// else if(mGraphMode == 5){
	// 	DrawTorqueGraph("HandL_x", 30, p_w, p_h, p_x, p_y-0*offset_y);
	// 	DrawTorqueGraph("HandL_y", 31, p_w, p_h, p_x, p_y-1*offset_y);
	// 	DrawTorqueGraph("HandL_z", 32, p_w, p_h, p_x, p_y-2*offset_y);
	// 	DrawTorqueGraph("HandR_x", 37, p_w, p_h, p_x, p_y-3*offset_y);
	// 	DrawTorqueGraph("HandR_y", 38, p_w, p_h, p_x, p_y-4*offset_y);
	// 	DrawTorqueGraph("HandR_z", 39, p_w, p_h, p_x, p_y-5*offset_y);
	// }

	DrawGLEnd();
}

void
Window::
DrawTorqueGraph(std::string name, int idx, double w, double h, double x, double y)
{
	double offset_x = 0.00024;
	double offset_y = 0.0005;
	double offset = 0.005;
	double ratio_y = 0.3;

	std::deque<double> data_ = (mTorques->GetTorques())[idx];

	if(idx == 0)
		offset_y *= 0.25;

	DrawBaseGraph(x, y, w, h, ratio_y, offset, name);
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, data_);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data_, red);
	if(idx == 0){
		DrawStringMean(x, y, w, h, ratio_y, offset_x, offset_y, offset, data_, green);
	}
}

void
Window::
DrawStride()
{
	DrawGLBegin();

	double w = 0.15;
	double h = 0.11;
	double x = 0.69;
	double y = 0.47;

	double offset_x = 0.003;
	double offset_y = 1.0;
	double offset = 0.005;
	double ratio_y = 0.0;

	y = 0.49;
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "stride");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, mFootinterval, 0.7, 0.8);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, mFootinterval, green, 0.7, 0.8);
	// DrawStringMean(x, y, w, h, ratio_y, offset_x, offset_y, offset, mFootinterval, green);

	DrawGLEnd();
}

void
Window::
DrawReward()
{
	DrawGLBegin();

	double w = 0.15;
	double h = 0.11;
	double x = 0.69;
	double y = 0.49;

	double offset_x = 0.002;
	double offset_y = 0.1;
	double offset = 0.005;
	double ratio_y = 0.0;

	std::map<std::string, std::deque<double>> map = mEnv->GetRewards();

	y = 0.49;
	std::deque<double> reward = map["reward"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "reward");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, reward);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, reward, green);

	y = 0.37;
	std::deque<double> imit = map["imit"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "imit");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, imit);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, imit, green);

	y = 0.25;
	std::deque<double> min = map["min"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "min");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, min);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, min, green);

	y = 0.13;
	std::deque<double> contact = map["contact"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "contact");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, contact);

	y = 0.01;
	std::deque<double> smooth = map["smooth"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "smooth");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, smooth);

	x = 0.85;
	y = 0.49;
	std::deque<double> pose = map["pose"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "pose");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, pose);

	y = 0.37;
	std::deque<double> vel = map["vel"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "vel");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, vel);

	y = 0.25;
	std::deque<double> ee = map["ee"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "ee");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, ee);

	y = 0.13;
	std::deque<double> root = map["root"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "root");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, root);

	y = 0.01;
	std::deque<double> com = map["com"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "com");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, com);

	DrawGLEnd();
}

void
Window::
DrawDevice()
{
	Character* character = mEnv->GetCharacter();
	if(character->GetDevice_OnOff())
	{
		isDrawDevice = true;
		DrawSkeleton(mEnv->GetDevice()->GetSkeleton());
		if(mDrawGraph)
			DrawDeviceSignals();
		if(mDrawArrow)
			DrawArrow();
		isDrawDevice = false;
	}
}

void
Window::
DrawDeviceSignals()
{
	DrawGLBegin();

	double p_w = 0.30;
	double p_h = 0.14;
	double p_x = 0.01;
	double p_y = 0.84;

	double offset_x = 0.00024;
	double offset_y = 0.0006;
	double offset = 0.005;
	double ratio_y = 0.3;
	Device* device = mEnv->GetDevice();
	std::deque<double> data_L = device->GetSignals(0);
	std::deque<double> data_R = device->GetSignals(1);
	if(mGraphMode == 0){
		DrawLineStrip(p_x, p_y-1*0.16, p_h, ratio_y, offset_x, offset_y, offset, blue, 1.5, data_L, 180);
		DrawStringMax(p_x, p_y-1*0.16, p_h, ratio_y, offset_x, offset_y, offset, data_L, blue);
		DrawLineStrip(p_x, p_y-4*0.16, p_h, ratio_y, offset_x, offset_y, offset, blue, 1.5, data_R, 180);
		DrawStringMax(p_x, p_y-4*0.16, p_h, ratio_y, offset_x, offset_y, offset, data_R, blue);
	}

	DrawGLEnd();
}

void
Window::
DrawArrow()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);

	Eigen::Vector4d color(0.6, 1.2, 0.6, 0.8);
	mRI->setPenColor(color);

	Eigen::VectorXd f = mEnv->GetDevice()->GetDesiredTorques2();

	Eigen::Isometry3d trans_L = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getTransform();
	Eigen::Vector3d p_L = trans_L.translation();
	Eigen::Matrix3d rot_L = trans_L.rotation();
	Eigen::Vector3d dir_L1 = rot_L.col(2);
	Eigen::Vector3d dir_L2 = rot_L.col(2);
	dir_L2[2] *= -1;

	if(f[6] < 0)
		drawArrow3D(p_L, dir_L2,-0.04*f[6], 0.01, 0.03);
	else
		drawArrow3D(p_L, dir_L1, 0.04*f[6], 0.01, 0.03);

	Eigen::Isometry3d trans_R = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurR")->getTransform();
	Eigen::Vector3d p_R = trans_R.translation();
	Eigen::Matrix3d rot_R = trans_R.rotation();
	Eigen::Vector3d dir_R1 = rot_R.col(2);
	Eigen::Vector3d dir_R2 = rot_R.col(2);
	dir_R2[2] *= -1;

	if(f[7] < 0)
		drawArrow3D(p_R, dir_R2,-0.04*f[7], 0.015, 0.03);
	else
		drawArrow3D(p_R, dir_R1, 0.04*f[7], 0.015, 0.03);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_COLOR_MATERIAL);
}

void
Window::
DrawGLBegin()
{
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_LIGHTING);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
}

void
Window::
DrawGLEnd()
{
	glDisable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(oldMode);
}

void
Window::
DrawBaseGraph(double x, double y, double w, double h, double ratio, double offset, std::string name)
{
	h -= 2*offset;
	w -= 2*offset;
	x += offset;
	y += offset;
	DrawQuads(x-offset, y-offset, w+2*offset, h+2*offset, white);
	DrawLine(x, y+ratio*h, x+w, y+ratio*h, black, 1.5);
	DrawLine(x, y, x, y+h, black, 1.5);
	DrawString(x+0.45*w, y+offset, name);
}

void
Window::
DrawQuads(double x, double y, double w, double h, Eigen::Vector4d color)
{
	mRI->setPenColor(color);

	glBegin(GL_QUADS);
	glVertex2f(x    , y);
	glVertex2f(x    , y + h);
	glVertex2f(x + w, y + h);
	glVertex2f(x + w, y);
	glEnd();
}

void
Window::
DrawLine(double p1_x, double p1_y, double p2_x, double p2_y, Eigen::Vector4d color, double line_width)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	glBegin(GL_LINES);
	glVertex2f(p1_x, p1_y);
	glVertex2f(p2_x, p2_y);
	glEnd();
}

void
Window::
DrawLineStipple(double p1_x, double p1_y, double p2_x, double p2_y, Eigen::Vector4d color, double line_width)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	glLineStipple(3, 0xAAAA);
	glEnable(GL_LINE_STIPPLE);
	glBegin(GL_LINES);
	glVertex2f(p1_x, p1_y);
	glVertex2f(p2_x, p2_y);
	glEnd();
}

void
Window::
DrawString(double x, double y, std::string str)
{
	Eigen::Vector4d black(0.0, 0.0, 0.0, 1.0);
	mRI->setPenColor(black);

	glRasterPos2f(x, y);
	unsigned int length = str.length();
	for (unsigned int c = 0; c < length; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str.at(c));
}

void
Window::
DrawString(double x, double y, bool big, std::string str)
{
	Eigen::Vector4d black(0.0, 0.0, 0.0, 1.0);
	mRI->setPenColor(black);

	glRasterPos2f(x, y);
	unsigned int length = str.length();
	for (unsigned int c = 0; c < length; c++)
		if(big)
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, str.at(c));
		else
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str.at(c));
}

void
Window::
DrawString(double x, double y, std::string str, Eigen::Vector4d color)
{
	mRI->setPenColor(color);

	glRasterPos2f(x, y);
	unsigned int length = str.length();
	for (unsigned int c = 0; c < length; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str.at(c));
}

void
Window::
DrawStringMean(double x, double y, double w, double h, double ratio, double offset_x, double offset_y, double offset, std::vector<double> data, Eigen::Vector4d color)
{
	double sum = 0;
	for(int i=0; i<data.size(); i++)
		sum += data[i];

	double mean = sum/(double)(data.size());

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+w, y+mean*offset_y, std::to_string(mean), color);

	mRI->setPenColor(color);
	mRI->setLineWidth(1.5);

	glBegin(GL_LINE_STRIP);
	glVertex2f(x,         y+mean*offset_y);
	glVertex2f(x+w-2*offset,y+mean*offset_y);
	glEnd();
}

void
Window::
DrawStringMean(double x, double y, double w, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color)
{
	double sum = 0;
	for(int i=0; i<data.size(); i++)
		sum += data.at(i);

	double mean = sum/(double)(data.size());

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+w, y+mean*offset_y, std::to_string(mean), color);

	mRI->setPenColor(color);
	mRI->setLineWidth(1.5);

	glBegin(GL_LINE_STRIP);
	glVertex2f(x,         y+mean*offset_y);
	glVertex2f(x+w-2*offset,y+mean*offset_y);
	glEnd();
}

void
Window::
DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::vector<double> data, Eigen::Vector4d color)
{
	double max = 0;
	int idx = 0;
	for(int i=0; i<data.size(); i++)
	{
		if(data[i] > max)
		{
			max = data[i];
			idx = i;
		}
	}

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+idx*offset_x, y+max*offset_y, std::to_string(max), color);
}

void
Window::
DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color)
{
	double max = 0;
	int idx = 0;
	for(int i=0; i<data.size(); i++)
	{
		if(data[i] > max)
		{
			max = data.at(i);
			idx = i;
		}
	}

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+idx*offset_x, y+max*offset_y, std::to_string(max), color);
}

void
Window::
DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color, double lower, double upper)
{
	double max = 0;
	int idx = 0;
	for(int i=0; i<data.size(); i++)
	{
		if(data[i] > upper)
			data[i] = upper;

		if(data[i] < lower)
			data[i] = lower;

		if(data[i] > max)
		{
			max = data.at(i);
			idx = i;
		}
	}

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+idx*offset_x, y+(max-lower)*offset_y, std::to_string(max), color);
}

void
Window::
DrawStringMinMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color)
{
	double max = 0;
	double min = 0;
	int idx_max = 0;
	int idx_min = 0;
	for(int i=0; i<data.size(); i++)
	{
		if(data[i] > max)
		{
			max = data.at(i);
			idx_max = i;
		}

		if(data[i] < min)
		{
			min = data.at(i);
			idx_min = i;
		}
	}

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+idx_max*offset_x, y+max*offset_y, std::to_string(max), color);
	DrawString(x+idx_min*offset_x, y+min*offset_y, std::to_string(min), color);
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, int idx)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=idx; i<data.size(); i++)
		glVertex2f(x + offset_x*(i-idx), y + offset_y*data.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::vector<double>& data, Eigen::Vector4d color1, double line_width1, std::vector<double>& data1)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data.at(i));
	glEnd();

	mRI->setPenColor(color1);
	mRI->setLineWidth(line_width1);

	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data1.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data1.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::vector<double>& data)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, int idx, Eigen::Vector4d color1, double line_width1, std::deque<double>& data1, int idx1)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=idx; i<data.size(); i++)
		glVertex2f(x + offset_x*(i-idx), y + offset_y*data.at(i));
	glEnd();

	mRI->setPenColor(color1);
	mRI->setLineWidth(line_width1);

	glBegin(GL_LINE_STRIP);
	for(int i=idx1; i<data1.size(); i++)
		glVertex2f(x + offset_x*(i-idx1), y + offset_y*data1.at(i));
	glEnd();

	Eigen::Vector4d blend((color[0]+color1[0])/2.0, (color[1]+color1[1])/2.0, (color[2]+color1[2])/2.0, (color[3]+color1[3])/2.0);

	// mRI->setPenColor(blend);

	// glBegin(GL_LINE_STRIP);
	// for(int i=0; i<data1.size(); i++)
	//  glVertex2f(x + offset_x*i, y + offset_y*(data.at(i)+data1.at(i)));
	// for(int i=180; i<data.size(); i++)
	//  glVertex2f(x + offset_x*(i-180), y + offset_y*(data.at(i)+0.5*data1.at(i-180)));
	// glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, Eigen::Vector4d color1, double line_width1, std::deque<double>& data1)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=180; i<data.size(); i++)
		glVertex2f(x + offset_x*(i-180), y + offset_y*data.at(i));
	glEnd();

	mRI->setPenColor(color1);
	mRI->setLineWidth(line_width1);

	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data1.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data1.at(i));
	glEnd();

	Eigen::Vector4d blend((color[0]+color1[0])/2.0, (color[1]+color1[1])/2.0, (color[2]+color1[2])/2.0, (color[3]+color1[3])/2.0);

	// mRI->setPenColor(blend);

	// glBegin(GL_LINE_STRIP);
	// for(int i=0; i<data1.size(); i++)
	//  glVertex2f(x + offset_x*i, y + offset_y*(data.at(i)+data1.at(i)));
	// for(int i=180; i<data.size(); i++)
	//  glVertex2f(x + offset_x*(i-180), y + offset_y*(data.at(i)+0.5*data1.at(i-180)));
	// glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, double lower, double upper)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
	{
		if(data.at(i) < lower)
			glVertex2f(x + offset_x*i, y);
		else if(data.at(i) > upper)
			glVertex2f(x + offset_x*i, y + offset_y*(upper));
		else
			glVertex2f(x + offset_x*i, y + offset_y*(data.at(i)-lower));
	}
	glEnd();
}


Eigen::VectorXd
Window::
GetActionFromNN_Device()
{
	p::object get_action_device;
	get_action_device = device_nn_module.attr("get_action");
	Eigen::VectorXd state = mEnv->GetCharacter()->GetDevice()->GetState();
	p::tuple shape = p::make_tuple(state.rows());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];

	p::object temp = get_action_device(state_np);
	np::ndarray action_np = np::from_object(temp);

	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	Eigen::VectorXd action(mEnv->GetCharacter()->GetDevice()->GetNumAction());
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];

	return action;
}

np::ndarray toNumPyArray(const Eigen::VectorXd& vec)
{
	int n = vec.rows();
	p::tuple shape = p::make_tuple(n);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	for(int i =0;i<n;i++)
		dest[i] = vec[i];

	return array;
}

Eigen::VectorXd
Window::
GetActionFromNN()
{
	Eigen::VectorXd state = mEnv->GetCharacter()->GetState();
	p::tuple shape = p::make_tuple(state.rows());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];

	p::object apply;
	apply = rms_module.attr("apply_no_update");
	p::object state_np_tmp = apply(state_np);
	np::ndarray state_np_ = np::from_object(state_np_tmp);

	p::object get_action;
	get_action = nn_module.attr("get_action");
	p::object temp = get_action(state_np_);
	np::ndarray action_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	Eigen::VectorXd action(mEnv->GetCharacter()->GetNumAction());
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];

	return action;
}

Eigen::VectorXd
Window::
GetActivationFromNN(const Eigen::VectorXd& mt)
{
	if(!mMuscleNNLoaded)
	{
		mEnv->GetCharacter()->GetDesiredTorques();
		return Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
	}

	p::object get_activation = muscle_nn_module.attr("get_activation");
	mEnv->GetCharacter()->SetDesiredTorques();
	Eigen::VectorXd dt = mEnv->GetCharacter()->GetDesiredTorques();
	np::ndarray mt_np = toNumPyArray(mt);
	np::ndarray dt_np = toNumPyArray(dt);

	p::object temp = get_activation(mt_np,dt_np);
	np::ndarray activation_np = np::from_object(temp);

	Eigen::VectorXd activation(mEnv->GetCharacter()->GetMuscles().size());
	float* srcs = reinterpret_cast<float*>(activation_np.get_data());
	for(int i=0; i<activation.rows(); i++)
		activation[i] = srcs[i];

	return activation;
}

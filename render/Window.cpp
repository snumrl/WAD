#include "Window.h"
#include "Environment.h"
#include "Character.h"
#include "Device.h"
#include "BVH.h"
#include "Utils.h"
#include "Muscle.h"
#include <iostream>
#include <deque>
using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;

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

Eigen::Vector4d white(1.0, 1.0, 1.0, 1.0);
Eigen::Vector4d black(0.0, 0.0, 0.0, 1.0);
Eigen::Vector4d grey(0.6, 0.6, 0.6, 1.0);

Eigen::Vector4d red(1.0, 0.0, 0.0, 1.0);
Eigen::Vector4d green(0.2, 0.8, 0.2, 1.0);
Eigen::Vector4d blue(0.0, 0.0, 1.0, 1.0);

Window::
Window(Environment* env)
	:mEnv(env),mFocus(true),mSimulating(false),mDrawCharacter(true),mDrawTarget(false),mDrawOBJ(false),mDrawShadow(true),mMuscleNNLoaded(false),mDeviceNNLoaded(false),mOnDevice(false),mDrawTrajectory(false),mDrawProgressBar(false),mTalusL(false),mTalusR(false),isDrawTarget(false)
{
	mBackground[0] = 1.0;
	mBackground[1] = 1.0;
	mBackground[2] = 1.0;
	mBackground[3] = 1.0;
	SetFocus();
	mZoom = 0.25;
	mFocus = false;
	mNNLoaded = false;
	mGain = 300;
	mOnDevice = env->GetCharacter()->GetOnDevice();

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

	mOffset.resize(6);
	mOffset.setZero();
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

	if(env->GetUseDevice())
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
}

void
Window::
LoadDeviceNN(const std::string& device_nn_path)
{
	mDeviceNNLoaded = true;
	mOnDevice = true;

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
keyboard(unsigned char _key, int _x, int _y)
{
	Eigen::Vector3d force = Eigen::Vector3d::Zero();
	switch (_key)
	{
	// case '+': force[0] += 500.0;break;
	// case '-': force[0] -= 500.0;break;
	// case '+':
	// 	mGain += 50.0;
	// 	mEnv->GetCharacter()->SetKp(mGain);
	// 	mEnv->GetCharacter()->SetKv(mGain*0.1);
	// 	std::cout << "kp : " << mGain << std::endl;
	// 	std::cout << "kv : " << mGain*0.1 << std::endl;
	// 	break;
	// case '-':
	// 	mGain -= 50.0;
	// 	mEnv->GetCharacter()->SetKp(mGain);
	// 	mEnv->GetCharacter()->SetKv(mGain*0.1);
	// 	std::cout << "kp : " << mGain << std::endl;
	// 	std::cout << "kv : " << mGain*0.1 << std::endl;
	// 	break;
	case '+': mOffset[offsetIdx]+=0.2;break;
	case '-': mOffset[offsetIdx]-=0.2;break;
	case '1': offsetIdx = 0;break;
	case '2': offsetIdx = 1;break;
	case '3': offsetIdx = 2;break;
	case '4': offsetIdx = 3;break;
	case '5': offsetIdx = 4;break;
	case '6': offsetIdx = 5;break;
	case 's': this->Step();break;
	case 'r': this->Reset();break;
	case ' ': mSimulating = !mSimulating;break;
	case 'f': mFocus = !mFocus;break;
	case 'o': mDrawOBJ = !mDrawOBJ;break;
	case 't': mDrawTarget = !mDrawTarget;break;
	case 'c': mDrawCharacter = !mDrawCharacter;break;
	case 'd':
		if(mEnv->GetUseDevice())
			mOnDevice = !mOnDevice;
		break;
	// case 't': mDrawTrajectory = !mDrawTrajectory;break;
	case 'p': mDrawProgressBar = !mDrawProgressBar;break;
	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}

	// std::cout << mOffset[0] << " " << mOffset[1] << " " << mOffset[2] << " " << mOffset[3] << " " << mOffset[4] << " " << mOffset[5] << std::endl;
	// mEnv->GetCharacter()->GetBVH()->SetMotionOffset(mOffset);
	// Eigen::VectorXd f = Eigen::VectorXd::Zero(12);
	// f.segment<3>(6) = force;
	// f.segment<3>(9) = force;

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
	mTrajectory.clear();
	mFootprint.clear();
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
		int inference_per_sim = 2;
		for(int i=0; i<num/2; i+=inference_per_sim){
			Eigen::VectorXd mt = mEnv->GetCharacter()->GetMuscleTorques();
			mEnv->GetCharacter()->SetActivationLevels(GetActivationFromNN(mt));
			for(int j=0; j<inference_per_sim; j++)
				mEnv->Step(mOnDevice);
		}
	}
	else
	{
		for(int i=0; i<num; i++)
			mEnv->Step(mOnDevice);
	}

	SetTrajectory();
	glutPostRedisplay();
}

void
Window::
SetTrajectory()
{
	const SkeletonPtr& skel = mEnv->GetCharacter()->GetSkeleton();
	const BodyNode* pelvis = skel->getBodyNode("Pelvis");
	const BodyNode* talusR = skel->getBodyNode("TalusR");
	const BodyNode* talusL = skel->getBodyNode("TalusL");

	mTrajectory.push_back(pelvis->getCOM());

	if(talusR->getCOM()[1] > 0.03)
		mTalusR = false;
	if(talusR->getCOM()[1] < 0.0255 && !mTalusR)
	{
		mFootprint.push_back(talusR->getCOM());
		mTalusR = true;
	}

	if(talusL->getCOM()[1] > 0.03)
		mTalusL = false;
	if(talusL->getCOM()[1] < 0.0255 && !mTalusL)
	{
		mFootprint.push_back(talusL->getCOM());
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
		mTrans[1] -= 0.3;

		mTrans *=1000.0;
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
	if(mDrawProgressBar)
		DrawProgressBar();


	SetFocus();
}

void
Window::
DrawGround()
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glDisable(GL_LIGHTING);
	glBegin(GL_QUADS);

	auto ground = mEnv->GetGround();
	float y = ground->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(ground->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;

	int count = 0;
	double w = 1.0;
	double h = 1.0;
	for(double x=-100.0; x<100.01; x+=1.0)
	{
		for(double z=-100.0; z<100.01; z+=1.0)
		{
			if(count%2==0)
				glColor3f(216.0/255.0,211.0/255.0,204.0/255.0);
			else
				glColor3f(216.0/255.0-0.1,211.0/255.0-0.1,204.0/255.0-0.1);

			glVertex3f(x  , y, z  );
			glVertex3f(x+w, y, z  );
			glVertex3f(x+w, y, z+h);
			glVertex3f(x  , y, z+h);
			count++;
		}
	}
	glEnd();
	glEnable(GL_LIGHTING);
}

Eigen::Matrix3d
QuatToMat(Eigen::Quaterniond q)
{
	Eigen::Matrix3d m;
	m << 1-2*q.y()*q.y()-2*q.z()*q.z(), 2*(q.x()*q.y() + q.w()*q.z()), 2*(q.x()*q.z() - q.w()*q.y()),
		2*(q.x()*q.y() - q.w()*q.z()), 1-2*q.x()*q.x()-2*q.z()*q.z(), 2*(q.y()*q.z() + q.w()*q.x()),
		2*(q.x()*q.z() + q.w()*q.y()), 2*(q.y()*q.z() + q.w()*q.x()), 1-2*q.x()*q.x()-2*q.y()*q.y();

	return m;
}

void
Window::
DrawCharacter()
{
	SkeletonPtr skeleton = mEnv->GetCharacter()->GetSkeleton();
	if(mDrawCharacter)
		DrawSkeleton(skeleton);

	if(mDrawTarget)
		DrawTarget();

	// if(mDrawTrajectory)
	// 	DrawTrajectory();

	DrawEnergy();
	DrawReward();
	DrawRewardMap();

	if(mEnv->GetUseMuscle())
		DrawMuscles(mEnv->GetCharacter()->GetMuscles());

 // 	dart::dynamics::SkeletonPtr skel = mEnv->GetCharacter()->GetSkeleton();
	// dart::dynamics::BodyNode* root = skel->getBodyNode(0);
	// Eigen::Isometry3d trans = Utils::GetBodyTransform(root);

	// Eigen::Isometry3d origin_trans = Utils::GetOriginTrans(skel);
	// Eigen::Quaterniond origin_quat(origin_trans.rotation());

	// int body_num = skel->getNumBodyNodes();
	// for(int i=0; i<body_num; i++)
	// {
	// 	dart::dynamics::BodyNode* body = skel->getBodyNode(i);
	// 	Eigen::Vector3d lin_vel = body->getCOMLinearVelocity();
	// 	Eigen::Vector4d lin_vel4;
	// 	lin_vel4.segment(0,3) = lin_vel;
	// 	lin_vel4 = trans * lin_vel4;
	// 	lin_vel = lin_vel4.segment(0,3);

	// 	Eigen::Vector3d o = Utils::GetBodyTransform(body).translation();
	// 	// Eigen::Vector4d o4;
	// 	// o4.segment(0,3) = o;
	// 	// o4[3] = 1.0;
	// 	// o4 = origin_trans * o4;
	// 	// o = o4.segment(0,3);

	// 	glPushMatrix();

	// 	glBegin(GL_LINES);
	// 	glColor3f(1.0, 0.0, 0.0);
	// 	glVertex3f(o[0], o[1], o[2]);
	// 	glVertex3f(o[0]+0.1*lin_vel[0], o[1]+0.1*lin_vel[1], o[2]+0.1*lin_vel[2]);
	// 	glEnd();

	// 	glPopMatrix();
	// }
}
	// dart::dynamics::BodyNode* root = skeleton->getBodyNode(0);

	// glLineWidth(3.0);

	// glPushMatrix();

	// glBegin(GL_LINES);
	// glColor3f(1.0, 0.0, 0.0);
	// glVertex3f(0.0, 0.0, 0.0);
	// glVertex3f(1.0, 0.0, 0.0);
	// glEnd();

	// glBegin(GL_LINES);
	// glColor3f(0.0, 1.0, 0.0);
	// glVertex3f(0.0, 0.0, 0.0);
	// glVertex3f(0.0, 1.0, 0.0);
	// glEnd();

	// glBegin(GL_LINES);
	// glColor3f(0.0, 0.0, 1.0);
	// glVertex3f(0.0, 0.0, 0.0);
	// glVertex3f(0.0, 0.0, 1.0);
	// glEnd();

	// glPopMatrix();

	// Eigen::Isometry3d origin_trans = mEnv->GetCharacter()->GetOriginTrans();
	// Eigen::Quaterniond origin_quat(origin_trans.rotation());
	// int body_num = skeleton->getNumBodyNodes();
	// for(int i=0; i<body_num; i++)
	// {
	// 	dart::dynamics::BodyNode* body = skeleton->getBodyNode(i);
	// 	Eigen::Isometry3d trans = body->getTransform() * body->getParentJoint()->getJointProperties().mT_ChildBodyToJoint;
	// 	Eigen::Vector3d pos = trans.translation();
	// 	Eigen::Matrix3d ori = trans.rotation();

	// 	Eigen::Vector3d o_ = pos;
	// 	Eigen::Vector3d x_ = pos + ori.col(0)*0.1;
	// 	Eigen::Vector3d y_ = pos + ori.col(1)*0.1;
	// 	Eigen::Vector3d z_ = pos + ori.col(2)*0.1;

	// 	glBegin(GL_LINES);
	// 	glColor3f(1.0, 0.0, 0.0);
	// 	glVertex3f(o_[0], o_[1], o_[2]);
	// 	glVertex3f(x_[0], x_[1], x_[2]);
	// 	glEnd();

	// 	glBegin(GL_LINES);
	// 	glColor3f(0.0, 1.0, 0.0);
	// 	glVertex3f(o_[0], o_[1], o_[2]);
	// 	glVertex3f(y_[0], y_[1], y_[2]);
	// 	glEnd();

	// 	glBegin(GL_LINES);
	// 	glColor3f(0.0, 0.0, 1.0);
	// 	glVertex3f(o_[0], o_[1], o_[2]);
	// 	glVertex3f(z_[0], z_[1], z_[2]);
	// 	glEnd();
	// }

	// int joint_num = skeleton->getNumJoints();
	// int body_num = skeleton->getNumBodyNodes();
	// for(int i=0; i<joint_num; i++)
	// {
	// 	dart::dynamics::BodyNode* body = skeleton->getBodyNode(i);
	// 	Eigen::Isometry3d transform = body->getTransform()*body->getParentJoint()->getJointProperties().mT_ChildBodyToJoint;

	// 	Eigen::Vector4d o(0,0,0,1);
	// 	Eigen::Vector4d x(0.1,0,0,1);
	// 	Eigen::Vector4d y(0,0.1,0,1);
	// 	Eigen::Vector4d z(0,0,0.1,1);

	// 	Eigen::Vector4d o1 = transform * o;
	// 	// Eigen::Vector4d x1 = transform * x;
	// 	// Eigen::Vector4d y1 = transform * y;
	// 	// Eigen::Vector4d z1 = transform * z;

	// 	// glBegin(GL_LINES);
	// 	// glColor3f(1.0, 0.0, 0.0);
	// 	// glVertex3f(o1[0], o1[1], o1[2]);
	// 	// glVertex3f(x1[0], x1[1], x1[2]);
	// 	// glEnd();

	// 	// glBegin(GL_LINES);
	// 	// glColor3f(0.0, 1.0, 0.0);
	// 	// glVertex3f(o1[0], o1[1], o1[2]);
	// 	// glVertex3f(y1[0], y1[1], y1[2]);
	// 	// glEnd();

	// 	// glBegin(GL_LINES);
	// 	// glColor3f(0.0, 0.0, 1.0);
	// 	// glVertex3f(o1[0], o1[1], o1[2]);
	// 	// glVertex3f(z1[0], z1[1], z1[2]);
	// 	// glEnd();


	// 	// Eigen::Vector3d axis = skeleton->getJoint(i)->getAxis();

	// 	// glBegin(GL_LINES);
	// 	// glColor3f(1.0, 0.0, 0.0);
	// 	// glVertex3f(o1[0], o1[1], o1[2]);
	// 	// glVertex3f(o1[0] + 0.1* axis[0], o1[1] + 0.1* axis[1], o1[2] + 0.1* axis[2]);
	// 	// glEnd();


	// 	// std::cout << "i : " << i << std::endl;
	// 	// std::cout << "v size " << v.size() << std::endl;

	// 	// Eigen::Vector4d o2 = R * o;
	// 	// Eigen::Vector4d x2 = R * x;
	// 	// Eigen::Vector4d y2 = R * y;
	// 	// Eigen::Vector4d z2 = R * z;

	// 	// glBegin(GL_LINES);
	// 	// glColor3f(1.0, 0.0, 0.0);
	// 	// glVertex3f(o2[0], o2[1], o2[2]);
	// 	// glVertex3f(x2[0], x2[1], x2[2]);
	// 	// glEnd();

	// 	// glBegin(GL_LINES);
	// 	// glColor3f(0.0, 1.0, 0.0);
	// 	// glVertex3f(o2[0], o2[1], o2[2]);
	// 	// glVertex3f(y2[0], y2[1], y2[2]);
	// 	// glEnd();

	// 	// glBegin(GL_LINES);
	// 	// glColor3f(0.0, 0.0, 1.0);
	// 	// glVertex3f(o2[0], o2[1], o2[2]);
	// 	// glVertex3f(z2[0], z2[1], z2[2]);
	// 	// glEnd();

	// }


	// Eigen::Isometry3d origin_trans = mEnv->GetCharacter()->GetOriginTrans();
	// Eigen::Quaterniond origin_quat(origin_trans.rotation());
	// int body_num = skeleton->getNumBodyNodes();
	// for(int i=0; i<body_num; i++)
	// {
	// 	dart::dynamics::BodyNode* body = skeleton->getBodyNode(i);
	// 	Eigen::Isometry3d trans = body->getTransform() * body->getParentJoint()->getJointProperties().mT_ChildBodyToJoint;
	// 	Eigen::Vector3d pos = trans.translation();
	// 	Eigen::Matrix3d ori = trans.rotation();

	// 	// Eigen::Vector3d o_cur;
	// 	// o_cur[0] = o_point[0];
	// 	// o_cur[1] = o_point[1];
	// 	// o_cur[2] = o_point[2];

	// 	Eigen::Vector3d o_ = pos;
	// 	Eigen::Vector3d x_ = pos + ori_.col(0)*0.1;
	// 	Eigen::Vector3d y_ = pos + ori_.col(1)*0.1;
	// 	Eigen::Vector3d z_ = pos + ori_.col(2)*0.1;

	// 	glBegin(GL_LINES);
	// 	glColor3f(0.0, 1.0, 0.0);
	// 	glVertex3f(o_[0], o_[1], o_[2]);
	// 	glVertex3f(x_[0], x_[1], x_[2]);
	// 	glEnd();

	// 	glBegin(GL_LINES);
	// 	glColor3f(0.0, 1.0, 0.0);
	// 	glVertex3f(o_[0], o_[1], o_[2]);
	// 	glVertex3f(y_[0], y_[1], y_[2]);
	// 	glEnd();

	// 	glBegin(GL_LINES);
	// 	glColor3f(0.0, 0.0, 1.0);
	// 	glVertex3f(o_[0], o_[1], o_[2]);
	// 	glVertex3f(z_[0], z_[1], z_[2]);
	// 	glEnd();


		// dart::dynamics::BodyNode* body = skeleton->getBodyNode(i);
		// Eigen::Isometry3d trans = body->getTransform()*body->getParentJoint()->getJointProperties().mT_ChildBodyToJoint;
		// Eigen::Vector3d pos = trans.translation();
		// Eigen::Matrix3d ori = trans.rotation();
		// Eigen::Quaterniond ori_quat(ori);



	// 	Eigen::Vector3d lin_vel = body->getCOMLinearVelocity();
	// 	Eigen::Vector4d lin_vel_vector;
	// 	lin_vel_vector[0] = lin_vel[0];
	// 	lin_vel_vector[1] = lin_vel[1];
	// 	lin_vel_vector[2] = lin_vel[2];
	// 	lin_vel_vector[3] = 0.0;
	// 	lin_vel_vector = origin_trans * lin_vel_vector;
	// 	lin_vel[0] = lin_vel_vector[0];
	// 	lin_vel[1] = lin_vel_vector[1];
	// 	lin_vel[2] = lin_vel_vector[2];

	// 	Eigen::Vector3d o = pos;
	// 	Eigen::Vector4d o_point;
	// 	o_point[0] = o[0];
	// 	o_point[1] = o[1];
	// 	o_point[2] = o[2];
	// 	o_point[3] = 1.0;
	// 	o_point = origin_trans * o_point;
	// 	pos[0] = o_point[0];
	// 	pos[1] = o_point[1];
	// 	pos[2] = o_point[2];

	// 	glPushMatrix();

	// 	glBegin(GL_LINES);
	// 	glColor3f(1.0, 0.0, 0.0);
	// 	glVertex3f(pos[0], pos[1], pos[2]);
	// 	glVertex3f(pos[0] + lin_vel[0], pos[1] + lin_vel[1], pos[2] + lin_vel[2]);
	// 	glEnd();

	// 	glPopMatrix();

	// }

		// ori_quat = origin_quat * ori_quat;
		// if(ori_quat.w() < 0)
		// {
		// 	ori_quat.w() *= -1;
		// 	ori_quat.x() *= -1;
		// 	ori_quat.y() *= -1;
		// 	ori_quat.z() *= -1;
		// }

		// Eigen::Matrix3d ori_ = QuatToMat(ori_quat);

		// dart::dynamics::BodyNode* body = skeleton->getBodyNode(i);
		// Eigen::Isometry3d trans = body->getTransform() * body->getParentJoint()->getJointProperties().mT_ChildBodyToJoint;
		// Eigen::Vector3d pos = trans.translation();
		// Eigen::Matrix3d ori = trans.rotation();

		// Eigen::Vector3d o_cur;
		// o_cur[0] = o_point[0];
		// o_cur[1] = o_point[1];
		// o_cur[2] = o_point[2];

		// Eigen::Vector3d o_ = o_cur;
		// Eigen::Vector3d x_ = o_cur + ori_.col(0)*0.1;
		// Eigen::Vector3d y_ = o_cur + ori_.col(1)*0.1;
		// Eigen::Vector3d z_ = o_cur + ori_.col(2)*0.1;


		// glBegin(GL_LINES);
		// glColor3f(0.0, 1.0, 0.0);
		// glVertex3f(o_[0], o_[1], o_[2]);
		// glVertex3f(y_[0], y_[1], y_[2]);
		// glEnd();

		// glBegin(GL_LINES);
		// glColor3f(0.0, 0.0, 1.0);
		// glVertex3f(o_[0], o_[1], o_[2]);
		// glVertex3f(z_[0], z_[1], z_[2]);
		// glEnd();

		// Eigen::Vector3d o_ = pos;
		// Eigen::Vector3d x_ = pos + ori.col(0)*0.1;
		// Eigen::Vector3d y_ = pos + ori.col(1)*0.1;
		// Eigen::Vector3d z_ = pos + ori.col(2)*0.1;

		// Eigen::Vector4d o_prj;
		// o_prj[0] = o_[0];
		// o_prj[1] = o_[1];
		// o_prj[2] = o_[2];
		// o_prj[3] = 1.0;
		// o_prj = origin_trans * o_prj;

		// Eigen::Vector4d x_prj;
		// x_prj[0] = x_[0];
		// x_prj[1] = x_[1];
		// x_prj[2] = x_[2];
		// x_prj[3] = 1.0;
		// x_prj = origin_trans * x_prj;

		// Eigen::Vector4d y_prj;
		// y_prj[0] = y_[0];
		// y_prj[1] = y_[1];
		// y_prj[2] = y_[2];
		// y_prj[3] = 1.0;
		// y_prj = origin_trans * y_prj;

		// Eigen::Vector4d z_prj;
		// z_prj[0] = z_[0];
		// z_prj[1] = z_[1];
		// z_prj[2] = z_[2];
		// z_prj[3] = 1.0;
		// z_prj = origin_trans * z_prj;

		// glPushMatrix();

		// glBegin(GL_LINES);
		// glColor3f(1.0, 0.0, 0.0);
		// glVertex3f(o_[0], o_[1], o_[2]);
		// glVertex3f(x_[0], x_[1], x_[2]);
		// glEnd();

		// glBegin(GL_LINES);
		// glColor3f(0.0, 1.0, 0.0);
		// glVertex3f(o_[0], o_[1], o_[2]);
		// glVertex3f(y_[0], y_[1], y_[2]);
		// glEnd();

		// glBegin(GL_LINES);
		// glColor3f(0.0, 0.0, 1.0);
		// glVertex3f(o_[0], o_[1], o_[2]);
		// glVertex3f(z_[0], z_[1], z_[2]);
		// glEnd();

		// glBegin(GL_LINES);
		// glColor3f(1.0, 0.0, 0.0);
		// glVertex3f(o_prj[0], o_prj[1], o_prj[2]);
		// glVertex3f(x_prj[0], x_prj[1], x_prj[2]);
		// glEnd();

		// glBegin(GL_LINES);
		// glColor3f(0.0, 1.0, 0.0);
		// glVertex3f(o_prj[0], o_prj[1], o_prj[2]);
		// glVertex3f(y_prj[0], y_prj[1], y_prj[2]);
		// glEnd();

		// glBegin(GL_LINES);
		// glColor3f(0.0, 0.0, 1.0);
		// glVertex3f(o_prj[0], o_prj[1], o_prj[2]);
		// glVertex3f(z_prj[0], z_prj[1], z_prj[2]);
		// glEnd();

		// glPopMatrix();
	// }




	// SkeletonPtr skel = mEnv->GetCharacter()->GetSkeleton();

	// glPushMatrix();

	// Eigen::Vector3d o(0.0, 0.0, 0.0);
	// Eigen::Vector3d x(0.2, 0.0, 0.0);
	// Eigen::Vector3d y(0.0, 0.2, 0.0);
	// Eigen::Vector3d z(0.0, 0.0, 0.2);

	// o = skel->getBodyNode(i)->getWorldTransform()* skel->getBodyNode(i)->getParentJoint()->getJointProperties().mT_ChildBodyToJoint  * o;
	// x = skel->getBodyNode(i)->getWorldTransform()* skel->getBodyNode(i)->getParentJoint()->getJointProperties().mT_ChildBodyToJoint  * x;
	// y = skel->getBodyNode(i)->getWorldTransform()* skel->getBodyNode(i)->getParentJoint()->getJointProperties().mT_ChildBodyToJoint  * y;
	// z = skel->getBodyNode(i)->getWorldTransform()* skel->getBodyNode(i)->getParentJoint()->getJointProperties().mT_ChildBodyToJoint  * z;

	// glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	// glEnable(GL_COLOR_MATERIAL);

	// Eigen::Vector4d red(1.0, 0.0, 0.0, 1.0);
	// mRI->setPenColor(red);
	// glBegin(GL_LINES);
	// glVertex3f(o[0], o[1], o[2]);
	// glVertex3f(x[0], x[1], x[2]);
	// glEnd();

	// Eigen::Vector4d green(0.0, 1.0, 0.0, 1.0);
	// mRI->setPenColor(green);
	// glBegin(GL_LINES);
	// glVertex3f(o[0], o[1], o[2]);
	// glVertex3f(y[0], y[1], y[2]);
	// glEnd();

	// Eigen::Vector4d blue(0.0, 0.0, 1.0, 1.0);
	// mRI->setPenColor(blue);
	// glBegin(GL_LINES);
	// glVertex3f(o[0], o[1], o[2]);
	// glVertex3f(z[0], z[1], z[2]);
	// glEnd();

	// glDisable(GL_COLOR_MATERIAL);
	// glPopMatrix();
// }

// void
// Window::
// DrawRelativeCoor()
// {
// 	dart::dynamics::BodyNode* root = skeleton->getBodyNode(0);

// 	glLineWidth(3.0);

// 	glPushMatrix();

// 	glBegin(GL_LINES);
// 	glColor3f(1.0, 0.0, 0.0);
// 	glVertex3f(0.0, 0.0, 0.0);
// 	glVertex3f(1.0, 0.0, 0.0);
// 	glEnd();

// 	glBegin(GL_LINES);
// 	glColor3f(0.0, 1.0, 0.0);
// 	glVertex3f(0.0, 0.0, 0.0);
// 	glVertex3f(0.0, 1.0, 0.0);
// 	glEnd();

// 	glBegin(GL_LINES);
// 	glColor3f(0.0, 0.0, 1.0);
// 	glVertex3f(0.0, 0.0, 0.0);
// 	glVertex3f(0.0, 0.0, 1.0);
// 	glEnd();

// 	glPopMatrix();

// 	Eigen::Isometry3d origin_trans = mEnv->GetCharacter()->GetOriginTrans();
// 	Eigen::Quaterniond origin_quat(origin_trans.rotation());
// 	int body_num = skeleton->getNumBodyNodes();
// 	for(int i=0; i<body_num; i++)
// 	{
// 		dart::dynamics::BodyNode* body = skeleton->getBodyNode(i);
// 		Eigen::Isometry3d trans = body->getTransform() * body->getParentJoint()->getJointProperties().mT_ChildBodyToJoint;
// 		Eigen::Vector3d pos = trans.translation();
// 		Eigen::Matrix3d ori = trans.rotation();
// 		Eigen::Quaterniond ori_quat(ori);

// 		Eigen::Vector3d o = pos;
// 		Eigen::Vector4d o_point;
// 		o_point[0] = o[0];
// 		o_point[1] = o[1];
// 		o_point[2] = o[2];
// 		o_point[3] = 1.0;
// 		o_point = origin_trans * o_point;

// 		ori_quat = origin_quat * ori_quat;
// 		if(ori_quat.w() < 0)
// 		{
// 			ori_quat.w() *= -1;
// 			ori_quat.x() *= -1;
// 			ori_quat.y() *= -1;
// 			ori_quat.z() *= -1;
// 		}

// 		Eigen::Matrix3d ori_ = QuatToMat(ori_quat);

// 		// dart::dynamics::BodyNode* body = skeleton->getBodyNode(i);
// 		// Eigen::Isometry3d trans = body->getTransform() * body->getParentJoint()->getJointProperties().mT_ChildBodyToJoint;
// 		// Eigen::Vector3d pos = trans.translation();
// 		// Eigen::Matrix3d ori = trans.rotation();

// 		Eigen::Vector3d o_cur;
// 		o_cur[0] = o_point[0];
// 		o_cur[1] = o_point[1];
// 		o_cur[2] = o_point[2];

// 		Eigen::Vector3d o_ = o_cur;
// 		Eigen::Vector3d x_ = o_cur + ori_.col(0)*0.1;
// 		Eigen::Vector3d y_ = o_cur + ori_.col(1)*0.1;
// 		Eigen::Vector3d z_ = o_cur + ori_.col(2)*0.1;

// 		// Eigen::Vector3d o_ = pos;
// 		// Eigen::Vector3d x_ = pos + ori.col(0)*0.1;
// 		// Eigen::Vector3d y_ = pos + ori.col(1)*0.1;
// 		// Eigen::Vector3d z_ = pos + ori.col(2)*0.1;

// 		// Eigen::Vector4d o_prj;
// 		// o_prj[0] = o_[0];
// 		// o_prj[1] = o_[1];
// 		// o_prj[2] = o_[2];
// 		// o_prj[3] = 1.0;
// 		// o_prj = origin_trans * o_prj;

// 		// Eigen::Vector4d x_prj;
// 		// x_prj[0] = x_[0];
// 		// x_prj[1] = x_[1];
// 		// x_prj[2] = x_[2];
// 		// x_prj[3] = 1.0;
// 		// x_prj = origin_trans * x_prj;

// 		// Eigen::Vector4d y_prj;
// 		// y_prj[0] = y_[0];
// 		// y_prj[1] = y_[1];
// 		// y_prj[2] = y_[2];
// 		// y_prj[3] = 1.0;
// 		// y_prj = origin_trans * y_prj;

// 		// Eigen::Vector4d z_prj;
// 		// z_prj[0] = z_[0];
// 		// z_prj[1] = z_[1];
// 		// z_prj[2] = z_[2];
// 		// z_prj[3] = 1.0;
// 		// z_prj = origin_trans * z_prj;

// 		glPushMatrix();

// 		glBegin(GL_LINES);
// 		glColor3f(1.0, 0.0, 0.0);
// 		glVertex3f(o_[0], o_[1], o_[2]);
// 		glVertex3f(x_[0], x_[1], x_[2]);
// 		glEnd();

// 		glBegin(GL_LINES);
// 		glColor3f(0.0, 1.0, 0.0);
// 		glVertex3f(o_[0], o_[1], o_[2]);
// 		glVertex3f(y_[0], y_[1], y_[2]);
// 		glEnd();

// 		glBegin(GL_LINES);
// 		glColor3f(0.0, 0.0, 1.0);
// 		glVertex3f(o_[0], o_[1], o_[2]);
// 		glVertex3f(z_[0], z_[1], z_[2]);
// 		glEnd();

// 		// glBegin(GL_LINES);
// 		// glColor3f(1.0, 0.0, 0.0);
// 		// glVertex3f(o_prj[0], o_prj[1], o_prj[2]);
// 		// glVertex3f(x_prj[0], x_prj[1], x_prj[2]);
// 		// glEnd();

// 		// glBegin(GL_LINES);
// 		// glColor3f(0.0, 1.0, 0.0);
// 		// glVertex3f(o_prj[0], o_prj[1], o_prj[2]);
// 		// glVertex3f(y_prj[0], y_prj[1], y_prj[2]);
// 		// glEnd();

// 		// glBegin(GL_LINES);
// 		// glColor3f(0.0, 0.0, 1.0);
// 		// glVertex3f(o_prj[0], o_prj[1], o_prj[2]);
// 		// glVertex3f(z_prj[0], z_prj[1], z_prj[2]);
// 		// glEnd();

// 		glPopMatrix();
// 	}

// }

void
Window::
DrawTarget()
{
	isDrawTarget = true;
	SkeletonPtr skeleton = mEnv->GetCharacter()->GetSkeleton();

	Eigen::VectorXd cur_pos = skeleton->getPositions();
	skeleton->setPositions(mEnv->GetCharacter()->GetTargetPositions());

	glPushMatrix();
	DrawBodyNode(skeleton->getRootBodyNode());
	glPopMatrix();

	skeleton->setPositions(cur_pos);
	isDrawTarget = false;
}

void
Window::
DrawRewardGraph(std::string name, double w, double h, double x, double y)
{
	// graph
	double offset_x = 0.004;
	double offset_y = 0.1;

	DrawQuads(x, y, w, h, white);
	DrawLine(x+0.005, y+0.01, x+w-0.005, y+0.01, black, 2.0);
	DrawLine(x+0.005, y+0.01, x+0.005, y+h-0.01, black, 2.0);
	DrawString(x+0.4*w, y-0.015, name);

	std::vector<double> data_ = mEnv->GetCharacter()->GetReward_Graph(0);
	std::vector<double> data_device_ = mEnv->GetCharacter()->GetReward_Graph(1);

	DrawLineStrip(x+0.005, y+0.01, offset_x, offset_y, red, 1.5, data_, blue, 2.0, data_device_);

	DrawStringMax(x+0.005, y+0.01, offset_x, offset_y, data_, red);
	DrawStringMax(x+0.005, y+0.01, offset_x, offset_y, data_device_, blue);
}

void
Window::
DrawReward()
{
	GLint oldMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// graph coord & size
	double p_w = 0.15;
	double p_h = 0.13;
	double p_x = 0.7;

	DrawRewardGraph("Reward", p_w, p_h, p_x, 0.40);

	glDisable(GL_COLOR_MATERIAL);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(oldMode);
}

void
Window::
DrawRewardMap()
{
	GLint oldMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// graph coord & size
	double w = 0.125;
	double h = 0.101;
	double x = 0.86;
	double y = 0.405;

	double offset_x = 0.0004;
	double offset_y = 0.1;

	std::map<std::string, std::deque<double>> map = mEnv->GetRewardMap();

	y = 0.404;
	DrawQuads(x, y, w, h, white);
	DrawLine(x+0.005, y+0.01, x+w-0.005, y+0.01, black, 1.5);
	DrawLine(x+0.005, y+0.01, x+0.005, y+h+0.01, black, 1.5);
	DrawString(x+0.4*w, y+0.015, "pose");

	std::deque<double> pose = map.at("pose");
	DrawLineStrip(x+0.005, y+0.01, offset_x, offset_y, green, 1.0, pose);

	y = 0.303;
	DrawQuads(x, y, w, h, white);
	DrawLine(x+0.005, y+0.01, x+w-0.005, y+0.01, black, 1.5);
	DrawLine(x+0.005, y+0.01, x+0.005, y+h+0.01, black, 1.5);
	DrawString(x+0.4*w, y+0.015, "vel");

	std::deque<double> vel = map.at("vel");
	DrawLineStrip(x+0.005, y+0.01, offset_x, offset_y, green, 1.0, vel);

	y = 0.202;
	DrawQuads(x, y, w, h, white);
	DrawLine(x+0.005, y+0.01, x+w-0.005, y+0.01, black, 1.5);
	DrawLine(x+0.005, y+0.01, x+0.005, y+h+0.01, black, 1.5);
	DrawString(x+0.4*w, y+0.015, "ee");

	std::deque<double> ee = map.at("ee");
	DrawLineStrip(x+0.005, y+0.01, offset_x, offset_y, green, 1.0, ee);

	y = 0.101;
	DrawQuads(x, y, w, h, white);
	DrawLine(x+0.005, y+0.01, x+w-0.005, y+0.01, black, 1.5);
	DrawLine(x+0.005, y+0.01, x+0.005, y+h+0.01, black, 1.5);
	DrawString(x+0.4*w, y+0.015, "root");

	std::deque<double> root = map.at("root");
	DrawLineStrip(x+0.005, y+0.01, offset_x, offset_y, green, 1.0, root);

	y = 0.0;
	DrawQuads(x, y, w, h, white);
	DrawLine(x+0.005, y+0.01, x+w-0.005, y+0.01, black, 1.5);
	DrawLine(x+0.005, y+0.01, x+0.005, y+h+0.01, black, 1.5);
	DrawString(x+0.4*w, y+0.015, "com");

	std::deque<double> com = map.at("com");
	DrawLineStrip(x+0.005, y+0.01, offset_x, offset_y, green, 1.0, com);

	glDisable(GL_COLOR_MATERIAL);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(oldMode);
}


void
Window::
DrawEnergyGraph(std::string name, double w, double h, double x, double y)
{
	// graph
	double offset_x = 0.004;
	double offset_y = 0.001;

	DrawQuads(x, y, w, h, white);
	DrawLine(x+0.005, y+0.01, x+w-0.005, y+0.01, black, 2.0);
	DrawLine(x+0.005, y+0.01, x+0.005, y+h-0.01, black, 2.0);
	DrawString(x+0.4*w, y-0.015, name);

	std::vector<double> data_ = mEnv->GetCharacter()->GetEnergy(0).at(name);
	std::vector<double> data_device_ = mEnv->GetCharacter()->GetEnergy(1).at(name);

	DrawLineStrip(x+0.005, y+0.01, offset_x, offset_y, red, 1.5, data_, blue, 2.0, data_device_);

	DrawStringMax(x+0.005, y+0.01, offset_x, offset_y, data_, red);
	DrawStringMax(x+0.005, y+0.01, offset_x, offset_y, data_device_, blue);
}

void
Window::
DrawEnergy()
{
	GLint oldMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// graph coord & size
	double p_w = 0.15;
	double p_h = 0.13;
	double p_x = 0.01;

	DrawEnergyGraph("FemurR", p_w, p_h, p_x, 0.85);
	DrawEnergyGraph("FemurL", p_w, p_h, p_x, 0.70);
	DrawEnergyGraph("TibiaR", p_w, p_h, p_x, 0.55);
	DrawEnergyGraph("TibiaL", p_w, p_h, p_x, 0.40);
	DrawEnergyGraph("TalusR", p_w, p_h, p_x, 0.25);
	DrawEnergyGraph("TalusL", p_w, p_h, p_x, 0.10);

	glDisable(GL_COLOR_MATERIAL);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(oldMode);
}

void
Window::
DrawTrajectory()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	Eigen::Vector4d color(0.5, 0.5, 0.5, 0.7);
	mRI->setPenColor(color);
	mRI->setLineWidth(8.0);

	glBegin(GL_LINE_STRIP);
	for(int i=0; i<mTrajectory.size(); i++)
	{
		glVertex3f(mTrajectory[i][0], 0.0, mTrajectory[i][2]);
	}
	glEnd();

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
DrawDevice()
{
	if(mEnv->GetCharacter()->GetOnDevice())
	{
		DrawSkeleton(mEnv->GetDevice()->GetSkeleton());
		DrawDeviceSignals();
	}

	// SkeletonPtr skel = mEnv->GetCharacter()->GetDevice()->GetSkeleton();

	// glPushMatrix();

	// Eigen::Vector3d o(0.0, 0.0, 0.0);
	// Eigen::Vector3d x(0.2, 0.0, 0.0);
	// Eigen::Vector3d y(0.0, 0.2, 0.0);
	// Eigen::Vector3d z(0.0, 0.0, 0.2);

	// o = skel->getBodyNode(i)->getParentJoint()->getJointProperties().mT_ChildBodyToJoint * skel->getBodyNode(i)->getWorldTransform() * o;

	// x = skel->getBodyNode(i)->getParentJoint()->getJointProperties().mT_ChildBodyToJoint * skel->getBodyNode(i)->getWorldTransform() * x;

	// y = skel->getBodyNode(i)->getParentJoint()->getJointProperties().mT_ChildBodyToJoint * skel->getBodyNode(i)->getWorldTransform() * y;

	// z = skel->getBodyNode(i)->getParentJoint()->getJointProperties().mT_ChildBodyToJoint * skel->getBodyNode(i)->getWorldTransform() * z;

	// glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	// glEnable(GL_COLOR_MATERIAL);

	// Eigen::Vector4d red(1.0, 0.0, 0.0, 1.0);
	// mRI->setPenColor(red);
	// glBegin(GL_LINES);
	// glVertex3f(o[0], o[1], o[2]);
	// glVertex3f(x[0], x[1], x[2]);
	// glEnd();

	// Eigen::Vector4d green(0.0, 1.0, 0.0, 1.0);
	// mRI->setPenColor(green);
	// glBegin(GL_LINES);
	// glVertex3f(o[0], o[1], o[2]);
	// glVertex3f(y[0], y[1], y[2]);
	// glEnd();

	// Eigen::Vector4d blue(0.0, 0.0, 1.0, 1.0);
	// mRI->setPenColor(blue);
	// glBegin(GL_LINES);
	// glVertex3f(o[0], o[1], o[2]);
	// glVertex3f(z[0], z[1], z[2]);
	// glEnd();
	// glDisable(GL_COLOR_MATERIAL);
	// glPopMatrix();

	// std::cout << "name : " << sf->getName() << std::endl;
	// if( sf->getName()=="Controller_ShapeNode_0" ||
	// 	sf->getName()=="ControllerFront_ShapeNode_0" ||
	// 	sf->getName()=="ControllerRight_ShapeNode_0" ||
	// 	sf->getName()=="ControllerLeft_ShapeNode_0")
	// {
		// glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		// glEnable(GL_COLOR_MATERIAL);

		// Eigen::Vector4d red(1.0, 0.0, 0.0, 1.0);
		// mRI->setPenColor(red);
		// glBegin(GL_LINES);
		// glVertex3f(0.0, 0.0, 0.0);
		// glVertex3f(0.5, 0.0, 0.0);
		// glEnd();

		// Eigen::Vector4d green(0.0, 1.0, 0.0, 1.0);
		// mRI->setPenColor(green);
		// glBegin(GL_LINES);
		// glVertex3f(0.0, 0.0, 0.0);
		// glVertex3f(0.0, 0.5, 0.0);
		// glEnd();

		// Eigen::Vector4d blue(0.0, 0.0, 1.0, 1.0);
		// mRI->setPenColor(blue);
		// glBegin(GL_LINES);
		// glVertex3f(0.0, 0.0, 0.0);
		// glVertex3f(0.0, 0.0, 0.5);
		// glEnd();
		// glDisable(GL_COLOR_MATERIAL);
	// }
}

void
Window::
DrawDeviceSignals()
{
	GLint oldMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// graph coord & size
	double p_w = 0.24;
	double p_h = 0.15;

	double pl_x = 0.70;
	double pl_y = 0.83;

	double pr_x = 0.70;
	double pr_y = 0.65;

	// graph
	double offset_x = 0.0003;
	double offset_y = 0.003;

	std::deque<double> data_y = mEnv->GetDevice()->GetSignals(0);

	// device L
	DrawQuads(pl_x, pl_y, p_w, p_h, white);
	DrawLine(pl_x+0.01, pl_y+0.01, pl_x+p_w-0.01, pl_y+0.01, black, 1.0);
	DrawLine(pl_x+0.01, pl_y+0.01, pl_x+0.01, pl_y+p_h-0.01, black, 1.0);
	DrawLine(pl_x+0.01+offset_x*340, pl_y+0.01, pl_x+0.01+offset_x*340, pl_y+p_h-0.01, grey, 1.0);
	DrawString(pl_x+0.5*p_w, pl_y-0.01, "Device L");

	std::deque<double> data_L = mEnv->GetDevice()->GetSignals(1);
	// std::deque<double> data_L_femur = mEnv->GetCharacter()->GetSignals(1);
	// DrawLineStrip(pl_x+0.01, pl_y+0.01, offset_x, offset_y, red, 2.0, data_L, blue, 2.0, data_L_femur);
	DrawLineStrip(pl_x+0.01, pl_y+0.01, offset_x, offset_y, red, 2.0, data_L, blue, 2.0, data_y);

 	// device R
	DrawQuads(pr_x, pr_y, p_w, p_h, white);
	DrawLine(pr_x+0.01, pr_y+0.01, pr_x+p_w-0.01, pr_y+0.01, black, 1.0);
	DrawLine(pr_x+0.01, pr_y+0.01, pr_x+0.01, pr_y+p_h-0.01, black, 1.0);
	DrawLine(pr_x+0.01+offset_x*340, pr_y+0.01, pr_x+0.01+offset_x*340, pr_y+p_h-0.01, grey, 1.0);
	DrawString(pr_x+0.5*p_w, pr_y-0.01, "Device R");

	std::deque<double> data_R = mEnv->GetDevice()->GetSignals(2);
	// std::deque<double> data_R_femur = mEnv->GetCharacter()->GetSignals(0);
	// DrawLineStrip(pr_x+0.01, pr_y+0.01, offset_x, offset_y, red, 2.0, data_R, blue, 2.0, data_R_femur);
	DrawLineStrip(pr_x+0.01, pr_y+0.01, offset_x, offset_y, red, 2.0, data_R, blue, 2.0, data_y);

	glDisable(GL_COLOR_MATERIAL);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(oldMode);
}

void
Window::
DrawProgressBar()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	Eigen::Vector4d color(0.5, 0.5, 0.5, 1.0);
	mRI->setPenColor(color);

	double phase = mEnv->GetCharacter()->GetPhase();
	dart::gui::drawProgressBar(phase*100, 100);

	glDisable(GL_COLOR_MATERIAL);
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
DrawSkeleton(const SkeletonPtr& skel)
{
	glPushMatrix();
	DrawBodyNode(skel->getRootBodyNode());
	glPopMatrix();
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

	Eigen::Vector4d color = va->getRGBA();
	if(isDrawTarget)
	{
		color[0] = 1.0;
		color[1] = 0.6;
		color[2] = 0.6;
		color[3] = 0.3;
	}

	DrawShape(sf->getShape().get(), color);
	mRI->popMatrix();
}

void
Window::
DrawShape(const Shape* shape,const Eigen::Vector4d& color)
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
			glDisable(GL_COLOR_MATERIAL);
			mRI->drawMesh(mesh->getScale(), mesh->getMesh());
			float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
		}

	}

	glDisable(GL_COLOR_MATERIAL);
}

void
Window::
DrawMuscles(const std::vector<Muscle*>& muscles)
{
	int count =0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	for(auto muscle : muscles)
	{
		auto aps = muscle->GetAnchors();
		bool lower_body = true;
		double a = muscle->GetActivation();
		Eigen::Vector4d color(0.4+(2.0*a),0.4,0.4,1.0);//0.7*(1.0-3.0*a));
		mRI->setPenColor(color);
		for(int i=0;i<aps.size();i++)
		{
			Eigen::Vector3d p = aps[i]->GetPoint();
			mRI->pushMatrix();
			mRI->translate(p);
			mRI->drawSphere(0.005*sqrt(muscle->GetF0()/1000.0));
			mRI->popMatrix();
		}

		for(int i=0;i<aps.size()-1;i++)
		{
			Eigen::Vector3d p = aps[i]->GetPoint();
			Eigen::Vector3d p1 = aps[i+1]->GetPoint();

			Eigen::Vector3d u(0,0,1);
			Eigen::Vector3d v = p-p1;
			Eigen::Vector3d mid = 0.5*(p+p1);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) =  axis(2);
			w_bracket(0, 2) =  axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) =  axis(0);


			Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(0.005*sqrt(muscle->GetF0()/1000.0),len);
			mRI->popMatrix();
		}

	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
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

    // draw all children
    for (n = 0; n < nd->mNumChildren; ++n) {
        DrawAiMesh(sc, nd->mChildren[n],M,y);
    }
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
	Eigen::VectorXd dt = mEnv->GetCharacter()->GetDesiredTorques();
	np::ndarray mt_np = toNumPyArray(mt);
	np::ndarray dt_np = toNumPyArray(dt);

	p::object temp = get_activation(mt_np,dt_np);
	np::ndarray activation_np = np::from_object(temp);

	Eigen::VectorXd activation(mEnv->GetCharacter()->GetMuscles().size());
	float* srcs = reinterpret_cast<float*>(activation_np.get_data());
	for(int i=0;i<activation.rows();i++)
		activation[i] = srcs[i];

	return activation;
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
DrawStringMax(double x, double y, double offset_x, double offset_y, std::vector<double> data, Eigen::Vector4d color)
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

	DrawString(x+idx*offset_x, y+max*offset_y, std::to_string(max), color);
}

void
Window::
DrawLineStrip(double x, double y, double offset_x, double offset_y, Eigen::Vector4d color, double line_width, std::deque<double>& data)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double offset_x, double offset_y, Eigen::Vector4d color, double line_width, std::vector<double>& data, Eigen::Vector4d color1, double line_width1, std::vector<double>& data1)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

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
DrawLineStrip(double x, double y, double offset_x, double offset_y, Eigen::Vector4d color, double line_width, std::vector<double>& data)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double offset_x, double offset_y, Eigen::Vector4d color, double line_width, std::deque<double>& data, Eigen::Vector4d color1, double line_width1, std::deque<double>& data1)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

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

	// Eigen::Vector4d blend((color[0]+color1[0])/2.0, (color[1]+color1[1])/2.0, (color[2]+color1[2])/2.0, (color[3]+color1[3])/2.0);

	// mRI->setPenColor(blend);

	// glBegin(GL_LINE_STRIP);
	// for(int i=0; i<data1.size(); i++)
	// 	glVertex2f(x + offset_x*i, y + offset_y*(data.at(i)+data1.at(i)));
	// glEnd();

}

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
Eigen::Vector4d green(0.2, 0.6, 0.2, 1.0);
Eigen::Vector4d blue(0.2, 0.2, 0.8, 1.0);

Window::
Window(Environment* env)
	:mEnv(env),mFocus(true),mSimulating(false),mDrawCharacter(true),mDrawTarget(false),mDrawOBJ(false),mDrawShadow(true),mMuscleNNLoaded(false),mDeviceNNLoaded(false),mOnDevice(false),isDrawTarget(false),mDrawArrow(false)
{
	mBackground[0] = 1.0;
	mBackground[1] = 1.0;
	mBackground[2] = 0.9;
	mBackground[3] = 0.5;
	SetFocus();
	mZoom = 0.25;
	mFocus = false;
	mNNLoaded = false;
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
	double f_ = 0.0;
	switch (_key)
	{
	case 'r': this->Reset();break;
	case 's': this->Step();break;
	case 'f': mFocus = !mFocus;break;
	case 'o': mDrawOBJ = !mDrawOBJ;break;
	case 'a': mDrawArrow = !mDrawArrow;break;
	case 't': mDrawTarget = !mDrawTarget;break;
	case ' ': mSimulating = !mSimulating;break;
	case 'c': mDrawCharacter = !mDrawCharacter;break;
	case 'd':
		if(mEnv->GetUseDevice())
			mOnDevice = !mOnDevice;
		break;
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

	mEnv->GetReward();

	glutPostRedisplay();
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

	SetFocus();
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
	double w = 1.0;
	double h = 1.0;
	for(double x=-100.0; x<=100.0; x+=1.0)
	{
		for(double z=-100.0; z<=100.0; z+=1.0)
		{
			if(count%2==0)
				glColor3f(0.85, 0.83, 0.81);
			else
				glColor3f(0.75, 0.73, 0.71);

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
		DrawSkeleton(skeleton);

	if(mEnv->GetUseMuscle())
		DrawMuscles(mEnv->GetCharacter()->GetMuscles());

	if(mDrawTarget)
		DrawTarget();

	if(!mEnv->GetUseDevice())
		DrawFemurSignals();

	DrawTorques();
	DrawReward();
	DrawRewardMap();
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
	int count =0;
	glDisable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	for(auto muscle : muscles)
	{
		auto aps = muscle->GetAnchors();
		bool lower_body = true;
		double a = muscle->GetActivation();
		Eigen::Vector4d color(0.4+(2.0*a),0.4,0.4,1.0);
		//0.7*(1.0-3.0*a));
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
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
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
DrawFemurSignals()
{
	DrawGLBegin();

	double p_w = 0.30;
	double p_h = 0.15;
	double pl_x = 0.70;
	double pl_y = 0.83;
	double pr_x = 0.70;
	double pr_y = 0.60;

	double offset_x = 0.00024;
	double offset_y = 0.0006;
	double offset = 0.005;
	double ratio_y = 0.3;

	Character* character = mEnv->GetCharacter();
	std::deque<double> data_L_femur = character->GetSignals(0);
	std::deque<double> data_R_femur = character->GetSignals(1);

	DrawBaseGraph(pl_x, pl_y, p_w, p_h, ratio_y, offset, "Device L");
	DrawLineStrip(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, red, 2.0, data_L_femur);
	DrawStringMax(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, data_L_femur, red);

	DrawBaseGraph(pr_x, pr_y, p_w, p_h, ratio_y, offset, "Device R");
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

	mTorques = mEnv->GetCharacter()->GetTorques();
	DrawTorqueGraph("FemurL_x", 6, p_w, p_h, p_x, p_y);
	DrawTorqueGraph("FemurL_y", 7, p_w, p_h, p_x, p_y-0.16);
	DrawTorqueGraph("FemurL_z", 8, p_w, p_h, p_x, p_y-0.32);
	DrawTorqueGraph("FemurR_x", 13, p_w, p_h, p_x, p_y-0.48);
	DrawTorqueGraph("FemurR_y", 14, p_w, p_h, p_x, p_y-0.64);
	DrawTorqueGraph("FemurR_z", 15, p_w, p_h, p_x, p_y-0.80);

	DrawGLEnd();
}

void
Window::
DrawTorqueGraph(std::string name, int idx, double w, double h, double x, double y)
{
	double offset_x = 0.00024;
	double offset_y = 0.0006;
	double offset = 0.005;
	double ratio_y = 0.3;

	std::deque<double> data_ = (mTorques->GetTorquesDofs())[idx];

	DrawBaseGraph(x, y, w, h, ratio_y, offset, name);
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, data_);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data_, red);
}

void
Window::
DrawReward()
{
	DrawGLBegin();

	double p_w = 0.15;
	double p_h = 0.11;
	double p_x = 0.69;
	double p_y = 0.49;

	DrawRewardGraph("Reward", p_x, p_y, p_w, p_h);

	DrawGLEnd();
}

void
Window::
DrawRewardGraph(std::string name, double x, double y, double w, double h)
{
	double offset_x = 0.002;
	double offset_y = 0.1;
	double offset = 0.005;
	double ratio_y = 0.0;

	Character* character = mEnv->GetCharacter();
	std::deque<double> data_ = character->GetReward_Graph(0);
	std::deque<double> data_device_ = character->GetReward_Graph(1);

	DrawBaseGraph(x, y, w, h, ratio_y, offset, name);
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, data_, green, 1.5, data_device_);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data_, red);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data_device_, green);
}

void
Window::
DrawRewardMap()
{
	DrawGLBegin();

	double w = 0.15;
	double h = 0.11;
	double x = 0.85;
	double y = 0.49;

	double offset_x = 0.002;
	double offset_y = 0.1;
	double offset = 0.005;
	double ratio_y = 0.0;

	std::map<std::string, std::deque<double>> map = mEnv->GetRewardMap();
	std::deque<double> pose = map.at("pose");
	std::deque<double> vel = map.at("vel");
	std::deque<double> ee = map.at("ee");
	std::deque<double> root = map.at("root");
	std::deque<double> com = map.at("com");

	y = 0.49;
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "pose");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, pose);

	y = 0.37;
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "vel");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, vel);

	y = 0.25;
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "ee");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, ee);

	y = 0.13;
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "root");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, root);

	y = 0.01;
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "com");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, com);

	DrawGLEnd();
}

void
Window::
DrawDevice()
{
	Character* character = mEnv->GetCharacter();
	if(character->GetOnDevice())
	{
		DrawSkeleton(mEnv->GetDevice()->GetSkeleton());
		DrawDeviceSignals();
		if(mDrawArrow)
			DrawArrow();
	}
}

void
Window::
DrawDeviceSignals()
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

	Device* device = mEnv->GetDevice();
	Character* character = mEnv->GetCharacter();
	std::deque<double> data_L = device->GetSignals(0);
	std::deque<double> data_L_femur = character->GetSignals(0);
	std::deque<double> data_R = device->GetSignals(1);
	std::deque<double> data_R_femur = character->GetSignals(1);
	// std::deque<double> data_y = device->GetSignals(2);

	DrawBaseGraph(pl_x, pl_y, p_w, p_h, ratio_y, offset, "Device L");
	DrawLineStrip(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, blue, 1.5, data_L, red, 1.5, data_L_femur);
	DrawStringMax(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, data_L, blue);
	DrawStringMax(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset,  data_L_femur, red);

	DrawBaseGraph(pr_x, pr_y, p_w, p_h, ratio_y, offset, "Device R");
	DrawLineStrip(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset, blue, 1.5, data_R, red, 1.5, data_R_femur);
	DrawStringMax(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset, data_R, blue);
	DrawStringMax(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset,  data_R_femur, red);

	DrawGLEnd();
}

void
Window::
DrawArrow()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

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

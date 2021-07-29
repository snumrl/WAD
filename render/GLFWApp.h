#ifndef GLFWAPP_H
#define GLFWAPP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
// #include <pybind11/eigen.h>

#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
#include <imgui.h>
#include <implot.h>

#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include "dart/gui/Trackball.hpp"

#include "Environment.h"
#include "Character.h"
#include "Muscle.h"
#include "BVH.h"
#include "ShapeRenderer.h"
#include "GLFunctions.h"

struct GLFWwindow;

using namespace dart::gui;
using namespace dart::dynamics;
using namespace dart::simulation;
namespace py = pybind11;
namespace MASS 
{

class GLFWApp
{
public:
	GLFWApp(Environment* env);
	GLFWApp(Environment* env, const std::string& nn_path);
	GLFWApp(Environment* env, const std::string& nn_path, const std::string& muscle_nn_path );
	~GLFWApp();

	void Initialize();
	void InitViewer();
	void InitCamera(int idx);
	void InitLights();
	void InitGLFW();
	void InitGL();

	void StartLoop();
	void Reset();
	void Update();
	void SetFocus();	

	void Draw();
	void DrawSimFrame();
    void DrawUiFrame();
	void DrawUiFrame_SimState(double x, double y, double w, double h);
    void DrawUiFrame_Learning(double x, double y, double w, double h);
    void DrawUiFrame_Analysis(double x, double y, double w, double h);

	void DrawGround();
	void DrawCharacter();
	void DrawCharacter_();
	void DrawTarget();
	void DrawReference();
	void DrawDevice();

	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawMuscles(const std::vector<Muscle*>& muscles);

	void DrawOriginCoord();

	void SetWindowWidth(double w){mWindowWidth=w;}
	void SetWindowHeight(double h){mWindowHeight=h;}

	void keyboardPress(int key, int scancode, int action, int mods);
    void mouseMove(double xpos, double ypos);
    void mousePress(int button, int action, int mods);
    void mouseScroll(double xoffset, double yoffset);

	Eigen::VectorXd GetActionFromNN();
	Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt);


private:
	py::object mm, mns, sys_module, nn_module, muscle_nn_module, rms_module;

	Environment* mEnv;
	GLFWwindow* mWindow;
	ShapeRenderer mShapeRenderer;
	
	dart::gui::Trackball mTrackball;
	std::vector<dart::gui::Trackball> mSplitTrackballs;

	Eigen::Vector3d mTrans;
	float mZoom, mPersp;
	float mMouseX, mMouseY;
	
	bool mMouseDown, mMouseDrag;
	bool mCapture, mRotate, mTranslate;

	bool mFocus;
	bool mSimulating;

	double mWindowWidth, mWindowHeight;
	double mViewerWidth, mViewerHeight;
	double mImguiWidth, mImguiHeight; 

	int mMuscleNum;
	int mMuscleMapNum;
	int mDisplayIter;

	int mViewMode;
	int mSplitIdx;
	int mSplitViewNum;
	bool mNNLoaded, mMuscleNNLoaded;
	bool mDevice_On;
	bool mDrawOBJ, mDrawCharacter, mDrawDevice, mDrawTarget, mDrawReference;
	bool isDrawCharacter, isDrawDevice, isDrawTarget, isDrawReference;

	std::map<std::string, double> perfStats;

	
};
}

#endif
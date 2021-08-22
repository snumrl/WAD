#ifndef GLFWAPP_H
#define GLFWAPP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
// #include <pybind11/eigen.h>

#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>

#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include "dart/gui/Trackball.hpp"

#include "Environment.h"
#include "Character.h"
#include "JointData.h"
#include "Muscle.h"
#include "BVH.h"
#include "ShapeRenderer.h"
#include "GLFunctions.h"
#include "AnalysisData.h"

struct GLFWwindow;

using namespace dart::gui;
using namespace dart::dynamics;
using namespace dart::simulation;
namespace py = pybind11;
namespace WAD 
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
	void InitUI();
	void InitAnalysis();

	void StartLoop();
	void Reset();
	void Update();
	void SetFocus();	

	void Draw();
	void DrawSimFrame();
    void DrawUiFrame();
	void DrawUiFrame_Manager();
	void DrawUiFrame_SimState(double x, double y, double w, double h);
    void DrawUiFrame_Learning(double x, double y, double w, double h);
    void DrawUiFrame_Analysis(double x, double y, double w, double h);

	void DrawDequeGraph(std::string name, std::string xAxis, std::string yAxis, std::deque<double> data, double w, double h);
	void DrawDequeGraph(std::string name, std::string xAxis, std::string yAxis, std::deque<double> data, double yMin, double yMax, double w, double h);
	void DrawJointAngle(std::string name, std::deque<double> stance, std::deque<double> swing, double yMin, double yMax, double w, double h);
	void DrawJointAngle(std::string name, std::deque<double> stanceL, std::deque<double> swingL, std::deque<double> stanceR, std::deque<double> swingR, double yMin, double yMax, double w, double h);
	void DrawJointAngle(std::string name, std::deque<double> stanceL, std::deque<double> swingL, std::deque<double> stanceR, std::deque<double> swingR, std::deque<double> data3, double yMin, double yMax, double w, double h);
	void DrawJointAngle(std::string name, std::deque<double> stanceL, std::deque<double> swingL, std::deque<double> stanceR, std::deque<double> swingR, std::deque<double> cmpData, std::deque<double> cmpStd1, std::deque<double> cmpStd2, double yMin, double yMax, double w, double h);
	void DrawJointTorque(std::string name, std::deque<double> data, double yMin, double yMax, double w, double h);
	void DrawJointTorque(std::string name, std::deque<double> data1, std::deque<double> data2, double yMin, double yMax, double w, double h);
	
	void DrawLine(Eigen::Vector3d v0, Eigen::Vector3d v1, Eigen::Vector4d color, double lineWidth);
	
	bool ShowCompareDataSelecter();

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

	ImGuiContext* context1;
	std::map<std::string, ImPlotContext*> mPlotContexts;

	Environment* mEnv;
	GLFWwindow* mWindow;
	ShapeRenderer mShapeRenderer;
	AnalysisData* mAnalysisData;
	std::map<std::string, std::deque<double>> mRealJointData;

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
	float mUiWidthRatio, mUiHeightRatio, mUiViewerRatio;

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
	bool isFirstUImanager;
	bool mDrawCoordinate;

	bool mLegend;
	int mComparePersonIdx;

	std::map<std::string, double> perfStats;
	std::map<std::string, std::pair<double,double>> mJointAngleMinMax;
	std::map<std::string, std::pair<double,double>> mJointTorqueMinMax;
	std::map<std::string, std::pair<double,double>> mAdaptiveParams;

	ImVec4 mUiBackground;	 	

};
}

#endif
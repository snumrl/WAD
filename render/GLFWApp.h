#ifndef GLFWAPP_H
#define GLFWAPP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ShapeRenderer.h"

namespace py = pybind11;

struct GLFWwindow;

namespace dart {
    namespace gui {
        class Trackball;
    }
}

namespace MASS {
class Environment;
class Muscle;
class GLFWApp
{
public:

	GLFWApp(int argc, char** argv);
	~GLFWApp();

    void startLoop();

private:
    void drawSimFrame();
    void drawUiFrame();
    void update();

    void keyboardPress(int key, int scancode, int action, int mods);
    void mouseMove(double xpos, double ypos);
    void mousePress(int button, int action, int mods);
    void mouseScroll(double xoffset, double yoffset);

    void initGL();
	void initLights();

	void draw(int cameraIdx = -1);

	void loadCheckpoint(const std::string& checkpoint_path);
	void calculateMarginalValues();

    void SetFocusing();

	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawMuscles(const std::vector<Muscle*>& muscles);
	void DrawGround(double y);

	void ClearSkeleton(const dart::dynamics::SkeletonPtr& skel);

	void Reset();

	Eigen::VectorXd GetActionFromNN();
//	Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt, const Eigen::VectorXd& pmt);
	Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt);

    GLFWwindow* window;

    py::object scope;
    py::object policy_nn, muscle_nn, marginal_nn;
    py::object mm,mns,sys_module,nn_module,muscle_nn_module,device_nn_module,rms_module;

    std::string checkpoint_path;
    py::object config;
    std::set<int> checkpoints;
    bool load_checkpoint_directory;
    int selected_checkpoint;

    float marginalValueAvg;
    Eigen::MatrixXf valueFunction;
    Eigen::MatrixXf probDistFunction;

	Environment* mEnv;
	bool mFocus;
	bool mSimulating;
	bool mDrawOBJ;
	bool mDrawShadow;
	bool mNNLoaded, mMuscleNNLoaded, mMarginalNNLoaded;
	bool mNoise, mKinematic, mPhysics;
	Eigen::Affine3d mViewMatrix;

	std::unique_ptr<dart::gui::Trackball> defaultTrackball;
	std::vector<dart::gui::Trackball> splitTrackballs;
	int selectedTrackballIdx = 0;

	Eigen::Vector3d mTrans;
	Eigen::Vector3d mEye;
	Eigen::Vector3d mUp;
	float mZoom;
	float mPersp;
	float mMouseX, mMouseY;
	bool mMouseDown, mMouseDrag, mCapture = false;

	bool mRotate = false, mTranslate = false, mZooming = false;

	double width, height;
	double viewportWidth, imguiWidth;
	bool mSplitViewport = false;

	std::map<std::string, double> perfStats;

	ShapeRenderer mShapeRenderer;
};
};

#endif
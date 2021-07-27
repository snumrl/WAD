#include "GLFWApp.h"

#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <implot.h>
#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
// #include <../imgui/imgui.h>
// #include <../implot/implot.h>
// #include <../imgui/backends/imgui_impl_glfw.h>
// #include <../imgui/backends/imgui_impl_opengl3.h>

#include <iostream>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "dart/gui/Trackball.hpp"

#include "Environment.h"
#include "BVH.h"
#include "Muscle.h"
#include "GLFunctions.h"
// #include "GLfunctions.h"

// #include "Init.h"

using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;

static const double PI = acos(-1);

GLFWApp::GLFWApp(int argc, char** argv)
        : mFocus(true), mSimulating(false), mDrawOBJ(true), mDrawShadow(false), mMuscleNNLoaded(false),
          mNoise(false), mKinematic(false), mPhysics(true),
          mTrans(0.0, 0.0, 0.0),
          mEye(0.0, 0.0, 1.0),
          mUp(0.0, 1.0, 0.0),
          mZoom(1.0),
          mPersp(45.0),
          mRotate(false),
          mTranslate(false),
          mZooming(false) {


///////////////////////////////////////////////////////////////////
    MASS::Environment* env = new MASS::Environment();

	if(argc==1)
	{
		std::cout<<"Provide Metadata.txt"<<std::endl;		
	}
	env->Initialize(std::string(argv[1]), true);

	// mm = py::module::import("__main__");
	// mns = mm.attr("__dict__");
	
	// py::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();
	// sys_module = py::module::import("sys");
	// sys_module.attr("path").attr("insert")(1, module_dir);
	
	// py::exec("import torch",mns);
	// py::exec("import torch.nn as nn",mns);
	// py::exec("import torch.optim as optim",mns);
	// py::exec("import torch.nn.functional as F",mns);
	// py::exec("import numpy as np",mns);
	// py::exec("from Model import *",mns);
	// py::exec("from RunningMeanStd import *",mns);

  
 	width = 1920; height = 1080;

	viewportWidth = 1500;
	imguiWidth = width - viewportWidth;

	defaultTrackball = std::make_unique<Trackball>();
    defaultTrackball->setTrackball(Eigen::Vector2d(viewportWidth * 0.5, height * 0.5), viewportWidth * 0.5);
    defaultTrackball->setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())));

    splitTrackballs.resize(3);
    for(int i = 0; i < 3; i++){
        splitTrackballs[i].setTrackball(Eigen::Vector2d(viewportWidth / 3 * (i + 0.5), height * 0.5),
                                        viewportWidth / 3 * 0.5);
        splitTrackballs[i].setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(i * PI / 2, Eigen::Vector3d::UnitY())));
    }

	mZoom = 0.25;
	mFocus = false;
	mNNLoaded = false;

	// scope = py::module::import("__main__").attr("__dict__");

	// py::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();
	// py::module::import("sys").attr("path").attr("insert")(1, module_dir);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(width, height, "render", nullptr, nullptr);
	if (window == NULL) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
	    glfwTerminate();
	    exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);

   	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
	    std::cerr << "Failed to initialize GLAD" << std::endl;
	    exit(EXIT_FAILURE);
	}

	glViewport(0, 0, width, height);

	glfwSetWindowUserPointer(window, this);

	auto framebufferSizeCallback = [](GLFWwindow* window, int width, int height) {
	    GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
	    app->width = width;
	    app->height = height;
	    glViewport(0, 0, width, height);
	};
	glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

	auto keyCallback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
	    auto& io = ImGui::GetIO();
	    if (!io.WantCaptureKeyboard) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->keyboardPress(key, scancode, action, mods);
        }
	};
	glfwSetKeyCallback(window, keyCallback);

	auto cursorPosCallback = [](GLFWwindow* window, double xpos, double ypos) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mouseMove(xpos, ypos);
        }
	};
	glfwSetCursorPosCallback(window, cursorPosCallback);

	auto mouseButtonCallback = [](GLFWwindow* window, int button, int action, int mods) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mousePress(button, action, mods);
        }
	};
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	auto scrollCallback = [](GLFWwindow* window, double xoffset, double yoffset) {
	    auto& io = ImGui::GetIO();
	    if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mouseScroll(xoffset, yoffset);
	    }
	};
	glfwSetScrollCallback(window, scrollCallback);

	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 150");

	ImPlot::CreateContext();

//////////////////////////////////////////////////////////////////



    /*
    auto res = InitEnv(argc, argv);
    mEnv = res.env;
    checkpoint_path = res.checkpoint_path;
    config = res.config;
    load_checkpoint_directory = res.load_checkpoint_directory;

    std::string checkpoint_file;
    if (load_checkpoint_directory) {
        auto cpath = fs::path(checkpoint_path);
        for (auto& p : fs::directory_iterator(cpath)) {
            if (fs::is_directory(p.path()) && p.path().stem().string().rfind("checkpoint", 0) == 0) {
                std::string cstr = p.path().stem().string();
                int checkpoint_num = std::stoi(cstr.substr(cstr.find('_') + 1));
                checkpoints.insert(checkpoint_num);
            }
        }
        int idx = *checkpoints.rbegin();
        checkpoint_file = cpath /
                (std::string("checkpoint_") + std::to_string(idx)) /
                (std::string("checkpoint-") + std::to_string(idx));
    }
    else {
        checkpoint_file = checkpoint_path;
    }

    if (!checkpoint_file.empty()) {
        loadCheckpoint(checkpoint_file);
	}
*/

    SetFocusing();
}

GLFWApp::~GLFWApp() {
    ImPlot::DestroyContext();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void GLFWApp::startLoop() {
#if 0
    const double frameTime = 1.0 / 30.0;
    double previous = glfwGetTime();
    double lag = 0;

    while (!glfwWindowShouldClose(window)) {
        double current = glfwGetTime();
        double elapsed = current - previous;
        previous = current;
        lag += elapsed;

        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        while (lag >= frameTime) {
            update();
            lag -= frameTime;
        }

        drawSimFrame();
        drawUiFrame();
        glfwSwapBuffers(window);
    }
#else
    // bool performUpdate = false;
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        // performUpdate = !performUpdate;
        // if (performUpdate) {
        //     update();
        // }
        double lastTime;
        lastTime = glfwGetTime();
        update();
        perfStats["update"] = std::ceil((glfwGetTime() - lastTime) * 100000) / 100;

        lastTime = glfwGetTime();
        drawSimFrame();
        perfStats["render_sim"] = std::ceil((glfwGetTime() - lastTime) * 100000) / 100;

        lastTime = glfwGetTime();
        drawUiFrame();
        perfStats["render_ui"] = std::ceil((glfwGetTime() - lastTime) * 100000) / 100;

        glfwSwapBuffers(window);
    }
#endif
}

void GLFWApp::drawSimFrame() {
    glViewport(0, 0, width, height);
    initGL();
    if (mSplitViewport) {
        for (int i = 0; i < 3; i++) {
            glViewport(viewportWidth / 3 * i, 0, viewportWidth / 3, height);
            draw(i);
        }
    }
    else {
        glViewport(0, 0, viewportWidth, height);
        draw();
    }
    glViewport(0, 0, width, height);
}

void GLFWApp::drawUiFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(viewportWidth, 0));
    ImGui::SetNextWindowSize(ImVec2(imguiWidth, height));

    ImGui::Begin("Inspector");
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File"))
        {

        }
        ImGui::EndMenuBar();
    }

    if (ImGui::CollapsingHeader("Performance")) {
        for (auto& [name, value] : perfStats) {
            ImGui::Text("%15s: %3f ms", name.c_str(), value);
        }
    }

    // if (ImGui::CollapsingHeader("Checkpoints")) {
    //     if (load_checkpoint_directory) {
    //         if (ImGui::ListBoxHeader("##Select checkpoint")) {
    //             for (int checkpoint_idx : checkpoints) {
    //                 std::string label = std::string("checkpoint-") + std::to_string(checkpoint_idx);
    //                 if (ImGui::Selectable(label.c_str(), selected_checkpoint == checkpoint_idx)) {
    //                     selected_checkpoint = checkpoint_idx;
    //                     std::string checkpoint_file = fs::path(checkpoint_path) /
    //                                                   (std::string("checkpoint_") + std::to_string(checkpoint_idx)) /
    //                                                   (std::string("checkpoint-") + std::to_string(checkpoint_idx));
    //                     loadCheckpoint(checkpoint_file);
    //                 }
    //             }
    //             ImGui::ListBoxFooter();
    //         }
    //     }
    //     else {
    //         ImGui::Text("Checkpoint: %s", checkpoint_path.c_str());
    //     }
    // }

    // static bool showValueGraph = false;
    // static bool showProbGraph = false;
    // if (ImGui::CollapsingHeader("Parameters")) {
    //     bool edited = false;
    //     auto& config = mEnv->GetConfig();
    //     auto& params = config.params;
    //     auto categoryNames = params.getCategoryNames();
    //     if (config.use_marginal_value_learning) {
    //         ImGui::Checkbox("Show value graph", &showValueGraph);
    //         if (config.use_adaptive_sampling) {
    //             ImGui::Checkbox("Show prob graph", &showProbGraph);
    //         }
    //     }
    //     for (auto& categoryName : categoryNames) {
    //         if (ImGui::TreeNode(categoryName.c_str())) {
    //             auto& indexMap = params.paramNameMap.at(categoryName).indexMap;
    //             auto paramMap = params.getValuesInCategoryAsMap(categoryName);
    //             for (auto& [name, value] : paramMap) {
    //                 std::string label = name;
    //                 if (categoryName == "length_ratio" && config.symmetry.count(name)) {
    //                     label += " / ";
    //                     label += config.symmetry.at(name);
    //                 }
    //                 float imguiValue = (float)value;
    //                 auto bounds = params.getBounds(categoryName, name);
    //                 ImGui::Text("%s", name.c_str());
    //                 std::string sliderLabel = "##";
    //                 sliderLabel += categoryName; sliderLabel += "_"; sliderLabel += name;
    //                 ImGui::SetNextItemWidth(300);
    //                 if (ImGui::SliderFloat(sliderLabel.c_str(), &imguiValue, bounds.first, bounds.second)) {
    //                     params.getValue(categoryName, name) = value = (double)imguiValue;
    //                     edited = true;
    //                     if (config.use_marginal_value_learning) calculateMarginalValues();
    //                 }
    //                 if (config.use_marginal_value_learning) {
    //                     int N = valueFunction.cols();
    //                     std::vector<float> pValues(N);
    //                     for (int k = 0; k < N; k++) {
    //                         pValues[k] = bounds.first + (bounds.second - bounds.first) * float(k) / float(N-1);
    //                     }
    //                     uint32_t idx = indexMap.at(name);
    //                     if (showValueGraph) {
    //                         Eigen::VectorXf values = valueFunction.row(idx);
    //                         ImPlot::SetNextPlotLimits(bounds.first, bounds.second,
    //                                                   -0.01, values.maxCoeff()+0.01);
    //                         auto lineLabel = categoryName + "_" + name + "_values";
    //                         if (ImPlot::BeginPlot(lineLabel.c_str(), nullptr, nullptr, ImVec2(300, 200))) {
    //                             ImPlot::PlotLine<float>(lineLabel.c_str(), pValues.data(), values.data(), pValues.size());
    //                             ImPlot::EndPlot();
    //                         }
    //                     }
    //                     if (showProbGraph) {
    //                         Eigen::VectorXf probs = probDistFunction.row(idx);
    //                         ImPlot::SetNextPlotLimits(bounds.first, bounds.second,
    //                                                   -0.01, probs.maxCoeff()+0.01);
    //                         auto lineLabel = categoryName + "_" + name + "_probs";
    //                         if (ImPlot::BeginPlot(lineLabel.c_str(), nullptr, nullptr, ImVec2(300, 200))) {
    //                             ImPlot::PlotLine<float>(lineLabel.c_str(), pValues.data(), probs.data(), pValues.size());
    //                             ImPlot::EndPlot();
    //                         }
    //                     }
    //                 }
    //             }
    //             ImGui::TreePop();
    //         }
    //     }
    //     if (edited) {
    //         mEnv->ResetWithCurrentParams();
    //     }
    // }

    // if (ImGui::CollapsingHeader("Marginal Values")) {

    // }
    // ImGui::End();

    // ImPlot::ShowDemoWindow();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GLFWApp::keyboardPress(int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_SPACE: mSimulating = !mSimulating; break;
            case GLFW_KEY_R: Reset(); break;
            case GLFW_KEY_O: mDrawOBJ = !mDrawOBJ; break;
            case GLFW_KEY_F: mFocus = !mFocus; break;
            case GLFW_KEY_V: mSplitViewport = !mSplitViewport; break;
            case GLFW_KEY_K: mKinematic = !mKinematic; break;
            case GLFW_KEY_P: mPhysics = !mPhysics; break;
            default: break;
        }
    }
}

void GLFWApp::mouseMove(double xpos, double ypos) {
    double deltaX = xpos - mMouseX;
    double deltaY = ypos - mMouseY;

    mMouseX = xpos;
    mMouseY = ypos;

    if (mRotate)
    {
        if (deltaX != 0 || deltaY != 0) {
            if (mSplitViewport) {
                splitTrackballs[selectedTrackballIdx].updateBall(xpos, height - ypos);
            }
            else {
                defaultTrackball->updateBall(xpos, height - ypos);
            }
        }
    }
    if (mTranslate)
    {
        Eigen::Matrix3d rot;
        if (mSplitViewport) {
            rot = splitTrackballs[selectedTrackballIdx].getRotationMatrix();
        }
        else {
            rot = defaultTrackball->getRotationMatrix();
        }
        mTrans += (1 / mZoom) * rot.transpose()
                  * Eigen::Vector3d(deltaX, -deltaY, 0.0);
    }
    if (mZooming)
    {
        mZoom = std::max(0.01, mZoom + deltaY * 0.01);
    }
}

void GLFWApp::mousePress(int button, int action, int mods) {
    mMouseDown = true;
    if (action == GLFW_PRESS) {
        if (mSplitViewport) {
            for (int i = 0; i < 3; i++) {
                if (width / 3 * i <= mMouseX && mMouseX < width / 3 * (i+1)) {
                    selectedTrackballIdx = i;
                }
            }
        }
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = true;
            if (mSplitViewport) {
                splitTrackballs[selectedTrackballIdx].startBall(mMouseX, height - mMouseY);
            }
            else {
                defaultTrackball->startBall(mMouseX, height - mMouseY);
            }
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = true;
        }
    }
    else if (action == GLFW_RELEASE) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = false;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = false;
        }
    }

}

void GLFWApp::mouseScroll(double xoffset, double yoffset) {
    mZoom += yoffset * 0.01;
}

void GLFWApp::draw(int cameraIdx)
{
    /* Preprocessing */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (cameraIdx == -1) {
        gluPerspective(mPersp, viewportWidth / height, 0.1, 10.0);
    }
    else {
        gluPerspective(mPersp, viewportWidth / 3 / height, 0.1, 10.0);
    }
    gluLookAt(mEye[0], mEye[1], mEye[2], 0.0, 0.0, -1.0, mUp[0], mUp[1], mUp[2]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (cameraIdx == -1) {
        defaultTrackball->applyGLRotation();
    }
    else {
        splitTrackballs[cameraIdx].applyGLRotation();
    }

    // Draw world origin indicator
    if (!mCapture)
    {
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_LIGHTING);
        glLineWidth(2.0);
        if (mRotate || mTranslate || mZooming)
        {
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(-0.1f, 0.0f, -0.0f);
            glVertex3f(0.15f, 0.0f, -0.0f);
            glEnd();

            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(0.0f, -0.1f, 0.0f);
            glVertex3f(0.0f, 0.15f, 0.0f);
            glEnd();

            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            glVertex3f(0.0f, 0.0f, -0.1f);
            glVertex3f(0.0f, 0.0f, 0.15f);
            glEnd();
        }
    }

    // TODO: Apply camera transform based on idx

    glScalef(mZoom, mZoom, mZoom);
    glTranslatef(mTrans[0] * 0.001, mTrans[1] * 0.001, mTrans[2] * 0.001);

    initLights();

    GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A << matrix[0], matrix[4], matrix[8],
            matrix[1], matrix[5], matrix[9],
            matrix[2], matrix[6], matrix[10];
    b << matrix[12], matrix[13], matrix[14];
    mViewMatrix.linear() = A;
    mViewMatrix.translation() = b;

    auto ground = mEnv->GetGround();
    float y = ground->getBodyNode(0)->getTransform().translation()[1] +
              dynamic_cast<const BoxShape *>(ground->getBodyNode(
                      0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1] * 0.5;

    DrawGround(y);
    if (mPhysics) {
        DrawMuscles(mEnv->GetCharacter()->GetMuscles());
        DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());
    }

    // if (mKinematic) {
    //     auto skel = mEnv->GetCharacter()->GetSkeleton()->cloneSkeleton();
    //     skel->setPosition(mEnv->GetCharacter()->GetTargetPositions())
    //     DrawSkeleton(skel);
    // }
}

void GLFWApp::initGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT, GL_FILL);
}

void GLFWApp::initLights() {
    static float ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float diffuse[] = {0.6, 0.6, 0.6, 1.0};
    static float front_mat_shininess[] = {60.0};
    static float front_mat_specular[] = {0.2, 0.2, 0.2, 1.0};
    static float front_mat_diffuse[] = {0.5, 0.28, 0.38, 1.0};
    static float lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float lmodel_twoside[] = {GL_FALSE};

    GLfloat position[] = {1.0, 0.0, 0.0, 0.0};
    GLfloat position1[] = {-1.0, 0.0, 0.0, 0.0};

    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, position);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);

    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, position1);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);

    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glEnable(GL_NORMALIZE);
}

void GLFWApp::update()
{
    SetFocusing();

    if (!mSimulating) return;

    // This app updates in Hz
    int num = mEnv->GetSimulationHz() / 60;

    static bool shouldUpdateAction = false;

    shouldUpdateAction = !shouldUpdateAction;
    if (shouldUpdateAction) {
        Eigen::VectorXd action;
        if (mNNLoaded)
            action = GetActionFromNN();
        else
            action = Eigen::VectorXd::Zero(mEnv->GetNumAction());
        mEnv->SetAction(action);
    }

    if (mEnv->GetUseMuscle()) {
        int inference_per_sim = 2;
        for (int i = 0; i < num; i += inference_per_sim) {
            Eigen::VectorXd mt = mEnv->GetCharacter()->GetMuscleTorques();
//			Eigen::VectorXd pmt = mEnv->GetPassiveMuscleTorques();
//			mEnv->SetActivationLevels(GetActivationFromNN(mt, pmt));
            mEnv->GetCharacter()->SetActivationLevels(GetActivationFromNN(mt));
            for (int j = 0; j < inference_per_sim; j++)
                mEnv->Step(false, true);
        }
    } else {
        for (int i = 0; i < num; i++)
            mEnv->Step(false, true);
    }

    // if (mEnv->getIsRender()) {

    // }
    // TODO
}

// void GLFWApp::loadCheckpoint(const std::string& checkpoint_path) {
//     auto model_res = InitModel(checkpoint_path, config, "cpu");
//     policy_nn = model_res.policy_nn;
//     muscle_nn = model_res.muscle_nn;
//     marginal_nn = model_res.marginal_nn;

//     mNNLoaded = true;
//     auto& config = mEnv->GetConfig();
//     if (config.use_muscle) {
//         mMuscleNNLoaded = true;
//     }
//     if (config.use_marginal_value_learning) {
//         mMarginalNNLoaded = true;
//         marginalValueAvg = model_res.marginalValueAvg;
//     }

//     std::cout << "Loaded " << checkpoint_path << "!" << std::endl;

//     if (config.use_marginal_value_learning) {
//         calculateMarginalValues();
//     }
// }

// void GLFWApp::calculateMarginalValues() {
//     assert(mEnv->GetUseAdaptiveSampling());
//     auto get_value = marginal_nn.attr("get_value");

//     const auto& config = mEnv->GetConfig();
//     const auto& params = config.params;

//     auto numParams = params.values.size();
//     int N = 101;
//     valueFunction.resize(numParams, N);
//     probDistFunction.resize(numParams, N);

//     for (const auto& category : params.getCategoryNames()) {
//         auto& indexMap = params.paramNameMap.at(category).indexMap;
//         for (const auto& [name, value] : params.getValuesInCategoryAsMap(category)) {
//             int i = indexMap.at(name);
//             auto [pmin, pmax] = params.bounds[i];
//             Eigen::MatrixXf paramCandidates(N, numParams);
//             for (int j = 0; j < N; j++) {
//                 float p = pmin + (pmax - pmin)/float(N-1) * j;
//                 for (int k = 0; k < numParams; k++) {
//                     paramCandidates(j, k) = k == i? p : params.values[k];
//                 }
//             }
//             Eigen::VectorXf values = get_value(paramCandidates).cast<Eigen::VectorXf>();
//             for (int j = 0; j < N; j++) {
//                 valueFunction(i, j) = values(j);
//                 probDistFunction(i, j) = exp(config.marginal_k * (1.0 - values(j) / marginalValueAvg));
//             }
//             float meanProbValue = probDistFunction.row(i).mean();
//             for (int j = 0; j < N; j++) {
//                 probDistFunction(i, j) /= (meanProbValue * (pmax - pmin));
//             }
//         }
//     }
// }

void GLFWApp::Reset()
{
	mEnv->Reset();

    // if (mEnv->getIsRender()) {
    //     // TODO
    // }
}

void GLFWApp::SetFocusing()
{
    if (mFocus) {
        if (mPhysics) {
            mTrans = -mEnv->GetWorld()->getSkeleton("Skeleton")->getRootBodyNode()->getCOM();
            mTrans[1] -= 0.3;
        } else {
            mTrans = -mEnv->GetCharacter()->GetTargetPositions().segment<3>(3);
            mTrans[0] -= 1.0;
            mTrans[1] -= 1.0;
        }

        mTrans *= 1000.0;
    }
}

Eigen::VectorXd
GLFWApp::
GetActionFromNN()
{
    using namespace pybind11::literals;

	Eigen::VectorXf state = mEnv->GetState().cast<float>();
    py::array_t<float> py_action = policy_nn.attr("get_action")(state);
    Eigen::VectorXf action = py_action.cast<Eigen::VectorXf>();
    return action.cast<double>();
}

Eigen::VectorXd
GLFWApp::
GetActivationFromNN(const Eigen::VectorXd& mt) //, const Eigen::VectorXd& pmt)
{
	if(!mMuscleNNLoaded)
	{
		mEnv->GetCharacter()->GetDesiredTorques();
		return Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
	}
	Eigen::VectorXd dt = mEnv->GetCharacter()->GetDesiredTorques();

//	py::object temp = get_activation(mt_np,pmt_np,dt_np);
	py::array_t<float> activation_np = muscle_nn.attr("get_activation")(mt.cast<float>(), dt.cast<float>());
	Eigen::VectorXf activation = activation_np.cast<Eigen::VectorXf>();

	return activation.cast<double>();
}

void GLFWApp::DrawEntity(const Entity* entity)
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

void GLFWApp::DrawBodyNode(const BodyNode* bn)
{	
	if(!bn)
		return;

	glPushMatrix();
	Eigen::Affine3d tmp = bn->getRelativeTransform();
	glMultMatrixd(tmp.data());

	auto sns = bn->getShapeNodesWith<VisualAspect>();
	for(const auto& sn : sns)
		DrawShapeFrame(sn);

	for(const auto& et : bn->getChildEntities())
		DrawEntity(et);

	glPopMatrix();
}

void GLFWApp::DrawSkeleton(const SkeletonPtr& skel)
{
	DrawBodyNode(skel->getRootBodyNode());
}

void GLFWApp::DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	const auto& va = sf->getVisualAspect();

	if(!va || va->isHidden())
		return;

	glPushMatrix();
	Eigen::Affine3d tmp = sf->getRelativeTransform();
	glMultMatrixd(tmp.data());

	DrawShape(sf->getShape().get(),va->getRGBA());

	glPopMatrix();
}

void GLFWApp::DrawShape(const Shape* shape, const Eigen::Vector4d& color)
{
	if(!shape)
		return;

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	if(mDrawOBJ == false)
	{
        glColor4dv(color.data());
		if (shape->is<SphereShape>())
		{
			const auto* sphere = dynamic_cast<const SphereShape*>(shape);
            GUI::DrawSphere(sphere->getRadius());
		}
		else if (shape->is<BoxShape>())
		{
			const auto* box = dynamic_cast<const BoxShape*>(shape);
			GUI::DrawCube(box->getSize());
		}
		else if (shape->is<CapsuleShape>())
		{
			const auto* capsule = dynamic_cast<const CapsuleShape*>(shape);
			GUI::DrawCapsule(capsule->getRadius(), capsule->getHeight());
		}	
	}
	else
	{
		if (shape->is<MeshShape>())
		{
			const auto& mesh = dynamic_cast<const MeshShape*>(shape);
			glDisable(GL_COLOR_MATERIAL);
            float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			// GUI::DrawMesh(mesh->getScale(), mesh->getMesh());
			// this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
            mShapeRenderer.renderMesh(mesh, false, y, color);
		}

	}
	
	glDisable(GL_COLOR_MATERIAL);
}
void GLFWApp::DrawMuscles(const std::vector<Muscle*>& muscles)
{
	int count = 0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	
	for (auto muscle : muscles)
	{
        double a = muscle->GetActivation();
        Eigen::Vector4d color(0.4+(2.0*a),0.4,0.4,1.0);//0.7*(1.0-3.0*a));
        glColor4dv(color.data());

	    mShapeRenderer.renderMuscle(muscle);
	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}

void GLFWApp::DrawGround(double y)
{
    constexpr int N = 64;
    GLubyte imageData[N][N];
    for (int i = 0; i < N; i++)  {
        for (int j = 0; j < N; j++) {
            if ((i/(N/2) + j/(N/2)) % 2 == 0) {
                imageData[i][j] = (GLubyte)216;
            }
            else {
                imageData[i][j] = (GLubyte)196;
            }
        }
    }

    glTexImage2D(GL_TEXTURE_2D, 0, 1, N, N, 0, GL_RED, GL_UNSIGNED_BYTE, imageData);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1, 1, 1, 1);
    // glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glBegin(GL_QUADS);
    glTexCoord2f(-50, -50); glVertex3f(-100, y, -100);
    glTexCoord2f(50, -50); glVertex3f(100, y, -100);
    glTexCoord2f(50, 50); glVertex3f(100, y, 100);
    glTexCoord2f(-50, 50); glVertex3f(-100, y, 100);
    glEnd();

    glEnable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
}

void GLFWApp::ClearSkeleton(const SkeletonPtr& skel)
{
    auto sns = skel->getRootBodyNode()->getShapeNodesWith<VisualAspect>();
    for (const auto& sn : sns) {
        auto shape = sn->getShape().get();
        if (shape->is<MeshShape>()) {
            mShapeRenderer.clearMesh(dynamic_cast<const MeshShape*>(shape));
        }
    }
}
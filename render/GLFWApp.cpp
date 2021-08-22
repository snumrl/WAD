#include "GLFWApp.h"
#include <iostream>

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

namespace WAD
{

#define IM_MIN(A, B)            (((A) < (B)) ? (A) : (B))
#define IM_MAX(A, B)            (((A) >= (B)) ? (A) : (B))
#define IM_CLAMP(V, MN, MX)     ((V) < (MN) ? (MN) : (V) > (MX) ? (MX) : (V))

#define WindowWidth 2160;
#define WinodwHeight 1080;

template <typename T>
inline T RandomRange(T min, T max) {
    T scale = rand() / (T) RAND_MAX;
    return min + scale * ( max - min );
}

GLFWApp::
GLFWApp(Environment* env)
    : mEnv(env),mFocus(true),mSimulating(false),mNNLoaded(false),mMuscleNNLoaded(false),
      mMouseDown(false), mMouseDrag(false),mCapture(false),mRotate(false),mTranslate(false),mDisplayIter(0),
      isDrawCharacter(false),isDrawDevice(false),isDrawTarget(false),isDrawReference(false),
      mDrawOBJ(false),mDrawCharacter(true),mDrawDevice(true),mDrawTarget(false),mDrawReference(false),
      mSplitViewNum(2),mSplitIdx(0),mViewMode(0),isFirstUImanager(true)
{
    mm = py::module::import("__main__");
	mns = mm.attr("__dict__");
    py::str module_dir = (std::string(WAD_ROOT_DIR)+"/python").c_str();
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

GLFWApp::
GLFWApp(Environment* env, const std::string& nn_path)
    :GLFWApp(env)
{
	mNNLoaded = true;

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

GLFWApp::
GLFWApp(Environment* env, const std::string& nn_path, const std::string& muscle_nn_path )
    :GLFWApp(env, nn_path)/*  */
{
    mMuscleNNLoaded = true;

	py::str str;
	str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetCharacter()->GetNumTotalRelatedDofs())).c_str();
	py::exec(str,mns);
	str = ("num_actions = "+std::to_string(mEnv->GetCharacter()->GetNumActiveDof())).c_str();
	py::exec(str,mns);
	str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetNumMuscles())).c_str();
	py::exec(str,mns);

	mMuscleNum = mEnv->GetCharacter()->GetNumMuscles();
	mMuscleMapNum = mEnv->GetCharacter()->GetNumMusclesMap();

	muscle_nn_module = py::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);

	py::object load = muscle_nn_module.attr("load");
	load(muscle_nn_path);
}

GLFWApp::
~GLFWApp() 
{
    for(auto p : mPlotContexts)
    {
        ImPlot::SetCurrentContext(p.second);
        ImPlot::DestroyContext();
    }    

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(mWindow);
    glfwTerminate();

    delete mAnalysisData;
}

void
GLFWApp::
Initialize()
{
    mDevice_On = mEnv->GetCharacter()->GetDevice_OnOff();
    this->InitGLFW();    
    this->InitAnalysis();
}

void
GLFWApp::
InitViewer()
{
    mZoom = 0.25;
    mPersp = 45.0;
	
    mUiWidthRatio = 0.1;
    mUiHeightRatio = 0.1;
    mUiViewerRatio = 0.4;

    //window size
    mWindowWidth = WindowWidth; 
    mWindowHeight = WinodwHeight;
    
    mViewerWidth = mWindowWidth*(mUiViewerRatio);
    mViewerHeight = mWindowHeight;

	mImguiWidth = mWindowWidth - mViewerWidth;
    mImguiHeight = mWindowHeight;
    
    double smaller = mViewerWidth < mViewerHeight ? mViewerWidth : mViewerHeight;
    mTrackball.setTrackball(Eigen::Vector2d(mViewerWidth*0.5, mViewerHeight*0.5), smaller*0.5);
    mTrackball.setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())));
    
    mSplitTrackballs.resize(mSplitViewNum);
    for(int i = 0; i < mSplitViewNum; i++){
        double w = mViewerWidth/mSplitViewNum;
        double h = mViewerHeight;
        double x = w*(i+0.5);
        double y = h*0.5;
        double radius = w < h ? w*0.5 : h*0.5;
        mSplitTrackballs[i].setTrackball(Eigen::Vector2d(x,y), radius);
        mSplitTrackballs[i].setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(i*M_PI/2, Eigen::Vector3d::UnitY())));
    }    
}

void
GLFWApp::
InitGLFW()
{
    this->InitViewer();

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, "render", nullptr, nullptr);
	if (mWindow == NULL) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
	    glfwTerminate();
	    exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(mWindow);

   	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
	    exit(EXIT_FAILURE);
	}

	glViewport(0, 0, mWindowWidth, mWindowHeight);
	glfwSetWindowUserPointer(mWindow, this);

    //CallBack
	auto framebufferSizeCallback = [](GLFWwindow* window, int width, int height) {
	    GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
	    app->SetWindowWidth(width);
	    app->SetWindowHeight(height);
	    glViewport(0, 0, width, height);
	};
	glfwSetFramebufferSizeCallback(mWindow, framebufferSizeCallback);

	auto keyCallback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
	    auto& io = ImGui::GetIO();
	    if (!io.WantCaptureKeyboard) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->keyboardPress(key, scancode, action, mods);
        }
	};
	glfwSetKeyCallback(mWindow, keyCallback);

	auto cursorPosCallback = [](GLFWwindow* window, double xpos, double ypos) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mouseMove(xpos, ypos);
        }
	};
	glfwSetCursorPosCallback(mWindow, cursorPosCallback);

	auto mouseButtonCallback = [](GLFWwindow* window, int button, int action, int mods) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mousePress(button, action, mods);
        }
	};
	glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);

	auto scrollCallback = [](GLFWwindow* window, double xoffset, double yoffset) {
	    auto& io = ImGui::GetIO();
	    if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mouseScroll(xoffset, yoffset);
	    }
	};
	glfwSetScrollCallback(mWindow, scrollCallback);

	ImGui::CreateContext();
    ImGui::StyleColorsClassic();
    
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
	ImGui_ImplOpenGL3_Init("#version 150");
    
    mPlotContexts["Main"] = ImPlot::CreateContext();
    ImPlot::StyleColorsDark();
    ImPlot::GetStyle().AntiAliasedLines = true;
    
    mPlotContexts["Analysis"] = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(mPlotContexts["Analysis"]);
    ImPlot::StyleColorsDark();
    ImPlot::GetStyle().AntiAliasedLines = true;

    mPlotContexts["Learning"] = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(mPlotContexts["Learning"]);
    ImPlot::StyleColorsDark();
    ImPlot::GetStyle().AntiAliasedLines = true;

    mPlotContexts["Simulation"] = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(mPlotContexts["Simulation"]);
    ImPlot::StyleColorsDark();
    ImPlot::GetStyle().AntiAliasedLines = true;
    
    ImPlot::SetCurrentContext(mPlotContexts["Main"]);

    this->InitUI();
}

void
GLFWApp::
InitUI()
{
    mUiBackground = ImVec4(0.0f, 0.0f, 0.0f, 0.8f);        
}

void 
GLFWApp::
InitGL() 
{
    glClearColor(0.96, 0.96, 0.97, 0.7);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT, GL_FILL);    
}

void
GLFWApp::
InitCamera(int idx)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (idx == -1)
        gluPerspective(mPersp, mViewerWidth/mViewerHeight, 0.1, 10.0);
    else 
        gluPerspective(mPersp, (mViewerWidth/mSplitViewNum)/mViewerHeight, 0.1, 10.0);

    gluLookAt(0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0); // eye, at, up
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    if (idx == -1)
        mTrackball.applyGLRotation();
    else 
        mSplitTrackballs[idx].applyGLRotation();

    if (!mCapture)
    {
        if (mRotate || mTranslate)
            this->DrawOriginCoord();       
    }
    
    // TODO: Apply camera transform based on idx
    glScalef(mZoom, mZoom, mZoom);
    glTranslatef(mTrans[0] * 0.001, mTrans[1] * 0.001, mTrans[2] * 0.001);    
}

void 
GLFWApp::
InitLights() 
{
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

void
GLFWApp::
InitAnalysis()
{
    mAnalysisData = new AnalysisData();
    mRealJointData = mAnalysisData->GetRealJointData(); 

    mJointAngleMinMax["Hip_sagittal"] = std::pair(-40.0, 40.0);
    mJointAngleMinMax["Hip_frontal"] = std::pair(-20.0, 40.0);
    mJointAngleMinMax["Hip_transverse"] = std::pair(-20.0, 20.0);

    mJointAngleMinMax["Knee_sagittal"] = std::pair(-15.0, 75.0);
    mJointAngleMinMax["Knee_frontal"] = std::pair(-20.0, 20.0);
    mJointAngleMinMax["Knee_transverse"] = std::pair(-20.0, 20.0);

    mJointAngleMinMax["Ankle_sagittal"] = std::pair(-40.0, 40.0);
    mJointAngleMinMax["Ankle_frontal"] = std::pair(-20.0, 20.0);
    mJointAngleMinMax["Ankle_transverse"] = std::pair(-40.0, 40.0);

    mJointTorqueMinMax["Hip_x"] = std::pair(-200.0,200.0);
    mJointTorqueMinMax["Hip_y"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Hip_z"] = std::pair(-200.0, 200.0);

    mJointTorqueMinMax["Knee_x"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Knee_y"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Knee_z"] = std::pair(-200.0, 200.0);

    mJointTorqueMinMax["Ankle_x"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Ankle_y"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Ankle_z"] = std::pair(-200.0, 200.0);

    mJointTorqueMinMax["Spine_x"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Spine_y"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Spine_z"] = std::pair(-200.0, 200.0);

    mJointTorqueMinMax["Torso_x"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Torso_y"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Torso_z"] = std::pair(-200.0, 200.0);

    mJointTorqueMinMax["Neck_x"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Neck_y"] = std::pair(-200.0, 200.0);
    mJointTorqueMinMax["Neck_z"] = std::pair(-200.0, 200.0);

    mAdaptiveParams = mEnv->GetCharacter()->GetAdaptiveParams();       
}

void
GLFWApp::
Reset()
{
	mEnv->Reset();
	mDisplayIter = 0;
}

void 
GLFWApp::
StartLoop() 
{
    const double frameTime = 1.0 / 60.0;
    double previous = glfwGetTime();
    double lag = 0;
    while (!glfwWindowShouldClose(mWindow)) {
        double current = glfwGetTime();
        double elapsed = current - previous;
        previous = current;
        lag += elapsed; 
    
        glfwPollEvents();
        
        while (lag >= frameTime) {
            if(mSimulating)
                this->Update();
            lag -= frameTime;
        }
        this->Draw();
    
        glfwSwapBuffers(mWindow);        
    }    
}

void GLFWApp::Update()
{
    if(mDisplayIter%2 == 0)
    {
        Eigen::VectorXd action;
        if (mNNLoaded)
            action = GetActionFromNN();
        else
            action = Eigen::VectorXd::Zero(mEnv->GetNumAction());

        mEnv->SetAction(action);
    }
 
    int num = mEnv->GetNumSteps()/2.0;
    if (mEnv->GetUseMuscle()) {
        int inference_per_sim = 2;
        for (int i=0; i<num; i+=inference_per_sim) {
            Eigen::VectorXd mt = mEnv->GetCharacter()->GetMuscleTorques();
            mEnv->GetCharacter()->SetActivationLevels(GetActivationFromNN(mt));
            for (int j = 0; j < inference_per_sim; j++)
                mEnv->Step(mDevice_On, true);
        }
    } 
    else 
    {
        for (int i=0; i<num; i++)
            mEnv->Step(mDevice_On, true);
    }
  
    mEnv->GetReward();
    mDisplayIter++;    
}

void 
GLFWApp::
SetFocus()
{
    if(mFocus) 
    {       
        mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
		mTrans[1] -= 0.3;
		mTrans *= 1000.0;

        double focusAngle = 0.05*M_PI;
        Eigen::Quaterniond focusQuat = Eigen::Quaterniond(Eigen::AngleAxisd(focusAngle, Eigen::Vector3d::UnitX()));
        Eigen::Quaterniond curQuat = mTrackball.getCurrQuat();
        if(mViewMode == 1)
        {
            mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
		    Eigen::Quaterniond r = focusQuat;
		    mTrackball.setQuaternion(r);            
        }
        else if(mViewMode == 2 && Eigen::AngleAxisd(curQuat).angle() < 0.5 * M_PI)
        {
            mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
		    Eigen::Vector3d axis(0.0, cos(focusAngle), sin(focusAngle));               
            Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(0.01*M_PI, axis)) * curQuat;
            mTrackball.setQuaternion(r);
        }
        else if(mViewMode == 3)
        {
            for(int i=0; i<mSplitViewNum; i++)
            {      
                mSplitTrackballs[i].setQuaternion(Eigen::Quaterniond::Identity());
		        Eigen::Quaterniond r = focusQuat;
                if(i==1){
                    Eigen::Vector3d axis(0.0, cos(focusAngle), sin(focusAngle));               
                    r = Eigen::Quaterniond(Eigen::AngleAxisd(0.5*M_PI, axis)) * focusQuat;
                }                    
		        mSplitTrackballs[i].setQuaternion(r);
            }
        }
    }
}

void 
GLFWApp::
Draw() 
{   
    this->InitGL();
    this->SetFocus();    
    
    this->DrawSimFrame();
    this->DrawUiFrame();    
}

void 
GLFWApp::
DrawSimFrame()
{
    int st = -1, ed = 0;
    int x = 0, y = 0;
    int w = mViewerWidth, h = mViewerHeight;
    
    if(mViewMode == 3)
    {
        st = 0, ed = mSplitViewNum;
        w /= mSplitViewNum, x = w;
    }

    for(int i=st; i<ed; i++)
    {
        glViewport(x*i, y, w, h);
        this->InitCamera(i);   
        this->InitLights();
        
        this->DrawGround();
        this->DrawCharacter();    
        if(mEnv->GetUseDevice())
            this->DrawDevice();            
    }       
}

void
GLFWApp::
DrawUiFrame() 
{
    double x1 = mViewerWidth;
    double x2 = mViewerWidth + mUiWidthRatio * mImguiWidth;
    double y1 = 0;
    double y2 = mUiHeightRatio * mImguiHeight;

    double w1 = mUiWidthRatio * mImguiWidth;
    double w2 = (1-mUiWidthRatio) * mImguiWidth;
    double h1 = mUiHeightRatio * mImguiHeight;
    double h2 = (1-mUiHeightRatio) * mImguiHeight;
    
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();    
    
    this->DrawUiFrame_Manager();
    this->DrawUiFrame_SimState(x1, y1, w1, h1+h2);
    this->DrawUiFrame_Learning(x2, y1, w2, h1);
    this->DrawUiFrame_Analysis(x2, y2, w2, h2);
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool ShowStyleSelector(const char* label)
{
    static int style_idx = 0;
    if (ImGui::Combo(label, &style_idx, "Classic\0Dark\0Light\0"))
    {
        switch (style_idx)
        {
        case 0: ImGui::StyleColorsClassic(); break;
        case 1: ImGui::StyleColorsDark(); break;
        case 2: ImGui::StyleColorsLight(); break;
        }
        return true;
    }
    return false;
}

void
GLFWApp::
DrawUiFrame_Manager()
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImGuiStyle ref_saved_style;
    ref_saved_style = style;
    ImGuiStyle* ref = &ref_saved_style;
    
    ImGui::Begin("UI Manager");   
    if(isFirstUImanager){
        ImGui::SetWindowPos("UI Manager", ImVec2(0, 0));
        isFirstUImanager = false;
    }
    ImGui::SetWindowSize("UI Manager", ImVec2(400, 240));

    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_None))
    {
        if (ImGui::BeginTabItem("Basic"))
        {
            style.FrameBorderSize  = 1.0f;
            style.WindowPadding = ImVec2(12,12);
            style.FramePadding = ImVec2(10,4);
            style.ItemSpacing = ImVec2(5,5);
         
            ImGui::Text("Style");
            if (ShowStyleSelector("Styles##Selector"))
                ref_saved_style = style;

            ImGui::Text("Frame");
            ImGui::SliderFloat("Width Ratio", &mUiWidthRatio, 0.1f, 1.0f);
            ImGui::SliderFloat("Height Ratio", &mUiHeightRatio, 0.1f, 1.0f);
            if(ImGui::SliderFloat("Viewer Ratio", &mUiViewerRatio, 0.1f, 1.0f))
            {
                mViewerWidth = mWindowWidth*(mUiViewerRatio);
                mViewerHeight = mWindowHeight;
                
                mImguiWidth = mWindowWidth - mViewerWidth;
                mImguiHeight = mWindowHeight;

                double smaller = mViewerWidth < mViewerHeight ? mViewerWidth : mViewerHeight;
                mTrackball.setTrackball(Eigen::Vector2d(mViewerWidth*0.5, mViewerHeight*0.5), smaller*0.5);
                mTrackball.setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())));
                
                for(int i = 0; i < mSplitViewNum; i++){
                    double w = mViewerWidth/mSplitViewNum;
                    double h = mViewerHeight;
                    double x = w*(i+0.5);
                    double y = h*0.5;
                    double radius = w < h ? w*0.5 : h*0.5;
                    mSplitTrackballs[i].setTrackball(Eigen::Vector2d(x,y), radius);
                    mSplitTrackballs[i].setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(i*M_PI/2, Eigen::Vector3d::UnitY())));
                }    
            }

            // if(ImGui::ColorEdit4("Background", (float*)&mUiBackground))
            //     ImGui::PushStyleColor(ImGuiCol_WindowBg, mUiBackground);        

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Colors"))
        {
            static ImGuiColorEditFlags alpha_flags = 0;
            if (ImGui::RadioButton("Opaque", alpha_flags == ImGuiColorEditFlags_None))             { alpha_flags = ImGuiColorEditFlags_None; } ImGui::SameLine();
            if (ImGui::RadioButton("Alpha",  alpha_flags == ImGuiColorEditFlags_AlphaPreview))     { alpha_flags = ImGuiColorEditFlags_AlphaPreview; } ImGui::SameLine();
            if (ImGui::RadioButton("Both",   alpha_flags == ImGuiColorEditFlags_AlphaPreviewHalf)) { alpha_flags = ImGuiColorEditFlags_AlphaPreviewHalf; } ImGui::SameLine();
            ImGui::Spacing();

            ImGui::BeginChild("##colors", ImVec2(0, 0), true, ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_AlwaysHorizontalScrollbar | ImGuiWindowFlags_NavFlattened);
            ImGui::PushItemWidth(-160);
            for (int i = 0; i < ImGuiCol_COUNT; i++)
            {
                const char* name = ImGui::GetStyleColorName(i);
                ImGui::PushID(i);
                ImGui::ColorEdit4("##color", (float*)&style.Colors[i], ImGuiColorEditFlags_AlphaBar | alpha_flags);
                if (memcmp(&style.Colors[i], &ref->Colors[i], sizeof(ImVec4)) != 0)
                {
                    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x); if (ImGui::Button("Save")) { ref->Colors[i] = style.Colors[i]; }
                    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x); if (ImGui::Button("Revert")) { style.Colors[i] = ref->Colors[i]; }
                }
                ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
                ImGui::TextUnformatted(name);
                ImGui::PopID();
            }
            ImGui::PopItemWidth();
            ImGui::EndChild();

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Fonts"))
        {
            ImGuiIO& io = ImGui::GetIO();
            // ImFontAtlas* atlas = io.Fonts;
            // ShowFontAtlas(atlas);

            // Post-baking font scaling. Note that this is NOT the nice way of scaling fonts, read below.
            // (we enforce hard clamping manually as by default DragFloat/SliderFloat allows CTRL+Click text to get out of bounds).
            const float MIN_SCALE = 0.3f;
            const float MAX_SCALE = 2.0f;
            static float window_scale = 1.0f;
            ImGui::PushItemWidth(ImGui::GetFontSize() * 8);
            if (ImGui::DragFloat("window scale", &window_scale, 0.005f, MIN_SCALE, MAX_SCALE, "%.2f", ImGuiSliderFlags_AlwaysClamp)) // Scale only this window
                ImGui::SetWindowFontScale(window_scale);
            ImGui::DragFloat("global scale", &io.FontGlobalScale, 0.005f, MIN_SCALE, MAX_SCALE, "%.2f", ImGuiSliderFlags_AlwaysClamp); // Scale everything
            ImGui::PopItemWidth();

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Plot"))
        {
            ImPlot::SetCurrentContext(mPlotContexts["Analysis"]);
            ImPlot::ShowColormapSelector("Analysis Color");            

            ImPlot::SetCurrentContext(mPlotContexts["Learning"]);
            ImPlot::ShowColormapSelector("Learning Color");      

            ImPlot::SetCurrentContext(mPlotContexts["Simulation"]);
            ImPlot::ShowColormapSelector("Simulation Color");      

            ImPlot::SetCurrentContext(mPlotContexts["Main"]);

            ImGui::EndTabItem();
        }   

        // if (ImGui::BeginTabItem("Rendering"))
        // {
        //     ImGui::Checkbox("Anti-aliased lines", &style.AntiAliasedLines);
        //     ImGui::SameLine();
            
        //     ImGui::Checkbox("Anti-aliased lines use texture", &style.AntiAliasedLinesUseTex);
        //     ImGui::SameLine();
            
        //     ImGui::Checkbox("Anti-aliased fill", &style.AntiAliasedFill);
        //     ImGui::PushItemWidth(ImGui::GetFontSize() * 8);
        //     ImGui::DragFloat("Curve Tessellation Tolerance", &style.CurveTessellationTol, 0.02f, 0.10f, 10.0f, "%.2f");
        //     if (style.CurveTessellationTol < 0.10f) style.CurveTessellationTol = 0.10f;

        //     // When editing the "Circle Segment Max Error" value, draw a preview of its effect on auto-tessellated circles.
        //     // ImGui::DragFloat("Circle Tessellation Max Error", &style.CircleTessellationMaxError , 0.005f, 0.10f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        //     if (ImGui::IsItemActive())
        //     {
        //         ImGui::SetNextWindowPos(ImGui::GetCursorScreenPos());
        //         ImGui::BeginTooltip();
        //         ImGui::TextUnformatted("(R = radius, N = number of segments)");
        //         ImGui::Spacing();
        //         ImDrawList* draw_list = ImGui::GetWindowDrawList();
        //         const float min_widget_width = ImGui::CalcTextSize("N: MMM\nR: MMM").x;
        //         for (int n = 0; n < 8; n++)
        //         {
        //             const float RAD_MIN = 5.0f;
        //             const float RAD_MAX = 70.0f;
        //             const float rad = RAD_MIN + (RAD_MAX - RAD_MIN) * (float)n / (8.0f - 1.0f);

        //             ImGui::BeginGroup();

        //             // ImGui::Text("R: %.f\nN: %d", rad, draw_list->_CalcCircleAutoSegmentCount(rad));

        //             const float canvas_width = IM_MAX(min_widget_width, rad * 2.0f);
        //             const float offset_x     = floorf(canvas_width * 0.5f);
        //             const float offset_y     = floorf(RAD_MAX);

        //             const ImVec2 p1 = ImGui::GetCursorScreenPos();
        //             draw_list->AddCircle(ImVec2(p1.x + offset_x, p1.y + offset_y), rad, ImGui::GetColorU32(ImGuiCol_Text));
        //             ImGui::Dummy(ImVec2(canvas_width, RAD_MAX * 2));

        //             /*
        //             const ImVec2 p2 = ImGui::GetCursorScreenPos();
        //             draw_list->AddCircleFilled(ImVec2(p2.x + offset_x, p2.y + offset_y), rad, ImGui::GetColorU32(ImGuiCol_Text));
        //             ImGui::Dummy(ImVec2(canvas_width, RAD_MAX * 2));
        //             */

        //             ImGui::EndGroup();
        //             ImGui::SameLine();
        //         }
        //         ImGui::EndTooltip();
        //     }
        //     ImGui::SameLine();
            
        //     ImGui::DragFloat("Global Alpha", &style.Alpha, 0.005f, 0.20f, 1.0f, "%.2f"); // Not exposing zero here so user doesn't "lose" the UI (zero alpha clips all widgets). But application code could have a toggle to switch between zero and non-zero.
        //     ImGui::PopItemWidth();

        //     ImGui::EndTabItem();
        // }

        ImGui::EndTabBar();
    }
   
    ImGui::End();
}

void
GLFWApp::
DrawUiFrame_SimState(double x, double y, double w, double h)
{
    ImGui::Begin("Simulation");                          // Create a window called "Hello, world!" and append into it.
    ImGui::SetWindowPos("Simulation", ImVec2(x, y));    
    ImGui::SetWindowSize("Simulation", ImVec2(w, h));
        
    ImPlot::SetCurrentContext(mPlotContexts["Simulation"]);        
    
    static bool coordinate = false;
    ImGui::Checkbox("Coordinate", &coordinate);
    mDrawCoordinate = coordinate;

    if(mEnv->GetUseAdaptiveSampling())
    {   
        int idx = 0;
        Eigen::VectorXd params = mEnv->GetParamState();
        for(auto p : mAdaptiveParams)
        {
            std::string name = p.first;
            double lower = p.second.first;
            double upper = p.second.second;
            float value = params[idx];
            ImGui::SliderFloat(name.c_str(), &value, lower, upper, "%.2f x");    
            params[idx] = (double)value;
            idx++;
        }        
        mEnv->SetParamState(params);        
    }

    ImPlot::SetCurrentContext(mPlotContexts["Main"]);
    ImGui::End();
}

void
GLFWApp::
DrawUiFrame_Learning(double x, double y, double w, double h)
{
    ImGui::Begin("Learning");                          // Create a window called "Hello, world!" and append into it.
    ImGui::SetWindowPos("Learning", ImVec2(x, y));    
    ImGui::SetWindowSize("Learning", ImVec2(w, h));
    
    ImPlot::SetCurrentContext(mPlotContexts["Learning"]);


    ImPlot::SetCurrentContext(mPlotContexts["Main"]);
    ImGui::End();
}

void
GLFWApp::
DrawDequeGraph(std::string name, std::string xAxis, std::string yAxis, std::deque<double> data, double w, double h)
{
    int size = data.size();
    float x[size]; 
    float data_[size];
    double min = 10000;
    double max = -10000;
    for(int i=0; i<size; i++)
    {
        x[i] = i*(1.0/(size-1.0));
        data_[i] = data[i];
        if(data_[i] > max)
            max = data_[i];
        if(data_[i] < min)
            min = data_[i];
    }

    ImPlot::SetNextPlotLimits(0.0, 1.0, min, max, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), xAxis.c_str(), yAxis.c_str(), ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
        ImPlot::PlotLine(yAxis.c_str(),x,data_,size);                        
        ImPlot::EndPlot();
    }
}

void
GLFWApp::
DrawDequeGraph(std::string name, std::string xAxis, std::string yAxis, std::deque<double> data, double yMin, double yMax, double w, double h)
{
    int size = data.size();
    float x[size]; 
    float data_[size];
    double min = yMin;
    double max = yMax;
    for(int i=0; i<size; i++)
    {
        x[i] = i*(1.0/(size-1.0));
        data_[i] = data[i];
        if(data_[i] > max)
            max = data_[i];
        if(data_[i] < min)
            min = data_[i];
    }

    if(max > yMax)
        yMax = max;
    if(min < yMin)
        yMin = min;

    ImPlot::SetNextPlotLimits(0.0, 1.0, yMin, yMax, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), xAxis.c_str(), yAxis.c_str(), ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
        ImPlot::PlotLine(yAxis.c_str(),x,data_,size);                        
        ImPlot::EndPlot();
    }    
}

void
GLFWApp::
DrawJointAngle(std::string name, std::deque<double> stance, std::deque<double> swing, double yMin, double yMax, double w, double h)
{
    int size1 = stance.size();
    int size2 = swing.size();
    int size = size1 + size2;
    float x[size];
    float data[size];
    for(int i=0; i<size1; i++)
    {
        x[i] = i*(0.6/(size1));
        data[i] = stance[i];        
    }
    for(int i=0; i<size2; i++)
    {
        x[i+size1] = i*(0.4/(size2-1.0))+0.6;
        data[i+size1] = swing[i];        
    }

    ImPlot::SetNextPlotLimits(0.0, 1.0, yMin, yMax, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), "%", "deg", ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
        ImPlot::PlotLine("deg",x,data,size);                        
        ImPlot::EndPlot();
    }
}

void
GLFWApp::
DrawJointAngle(std::string name, std::deque<double> stanceL, std::deque<double> swingL,  std::deque<double> stanceR, std::deque<double> swingR, double yMin, double yMax, double w, double h)
{
    int sizeL1 = stanceL.size();
    int sizeL2 = swingL.size();
    int sizeL = sizeL1 + sizeL2;
    float xL[sizeL];
    float dataL[sizeL];
    for(int i=0; i<sizeL1; i++)
    {
        xL[i] = i*(0.6/(sizeL1));
        dataL[i] = stanceL[i];        
    }
    for(int i=0; i<sizeL2; i++)
    {
        xL[i+sizeL1] = i*(0.4/(sizeL2-1.0))+0.6;
        dataL[i+sizeL1] = swingL[i];        
    }

    int sizeR1 = stanceR.size();
    int sizeR2 = swingR.size();
    int sizeR = sizeR1 + sizeR2;
    float xR[sizeR];
    float dataR[sizeR];
    for(int i=0; i<sizeR1; i++)
    {
        xR[i] = i*(0.6/(sizeR1));
        dataR[i] = stanceR[i];        
    }
    for(int i=0; i<sizeR2; i++)
    {
        xR[i+sizeR1] = i*(0.4/(sizeR2-1.0))+0.6;
        dataR[i+sizeR1] = swingR[i];        
    }   
    
    int plotStyle = ImPlotFlags_CanvasOnly;
    if(mLegend)
        plotStyle = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMousePos;
    ImPlot::SetNextPlotLimits(0.0, 1.0, yMin, yMax, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), "%", "deg", ImVec2(w,h), plotStyle, ImPlotAxisFlags_Lock)) {
        ImPlot::PlotLine("Left",xL,dataL,sizeL);                
        ImPlot::PlotLine("Right",xR,dataR,sizeR);                
        ImPlot::EndPlot();
    }
}

void
GLFWApp::
DrawJointAngle(std::string name, std::deque<double> stanceL, std::deque<double> swingL, std::deque<double> stanceR, std::deque<double> swingR, std::deque<double> data3, double yMin, double yMax, double w, double h)
{
    int sizeL1 = stanceL.size();
    int sizeL2 = swingL.size();
    int sizeL = sizeL1 + sizeL2;
    float xL[sizeL];
    float dataL[sizeL];
    for(int i=0; i<sizeL1; i++)
    {
        xL[i] = i*(0.6/(sizeL1));
        dataL[i] = stanceL[i];        
    }
    for(int i=0; i<sizeL2; i++)
    {
        xL[i+sizeL1] = i*(0.4/(sizeL2-1.0))+0.6;
        dataL[i+sizeL1] = swingL[i];        
    }

    int sizeR1 = stanceR.size();
    int sizeR2 = swingR.size();
    int sizeR = sizeR1 + sizeR2;
    float xR[sizeR];
    float dataR[sizeR];
    for(int i=0; i<sizeR1; i++)
    {
        xR[i] = i*(0.6/(sizeR1));
        dataR[i] = stanceR[i];        
    }
    for(int i=0; i<sizeR2; i++)
    {
        xR[i+sizeR1] = i*(0.4/(sizeR2-1.0))+0.6;
        dataR[i+sizeR1] = swingR[i];        
    }   

    int size3 = data3.size();
    float x3[size3];
    float data3_[size3];
    for(int i=0; i<size3; i++)
    {
        x3[i] = i*(1.0/(size3-1.0));
        data3_[i] = data3[i];        
    }
    
    int plotStyle = ImPlotFlags_CanvasOnly;
    if(mLegend)
        plotStyle = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMousePos;
    ImPlot::SetNextPlotLimits(0.0, 1.0, yMin, yMax, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), "%", "deg", ImVec2(w,h), plotStyle, ImPlotAxisFlags_Lock)) {
        ImPlot::PlotLine("left",xL,dataL,sizeL);                
        ImPlot::PlotLine("right",xR,dataR,sizeR);                
        ImPlot::PlotLine("real",x3,data3_,size3);                
        ImPlot::EndPlot();
    }
}

void
GLFWApp::
DrawJointAngle(std::string name, std::deque<double> stanceL, std::deque<double> swingL, std::deque<double> stanceR, std::deque<double> swingR, std::deque<double> cmpData, std::deque<double> cmpStd1, std::deque<double> cmpStd2, double yMin, double yMax, double w, double h)
{
    int sizeL1 = stanceL.size();
    int sizeL2 = swingL.size();
    int sizeL = sizeL1 + sizeL2;
    float xL[sizeL];
    float dataL[sizeL];
    for(int i=0; i<sizeL1; i++)
    {
        xL[i] = i*(0.6/(sizeL1));
        dataL[i] = stanceL[i];        
    }
    for(int i=0; i<sizeL2; i++)
    {
        xL[i+sizeL1] = i*(0.4/(sizeL2-1.0))+0.6;
        dataL[i+sizeL1] = swingL[i];        
    }

    int sizeR1 = stanceR.size();
    int sizeR2 = swingR.size();
    int sizeR = sizeR1 + sizeR2;
    float xR[sizeR];
    float dataR[sizeR];
    for(int i=0; i<sizeR1; i++)
    {
        xR[i] = i*(0.6/(sizeR1));
        dataR[i] = stanceR[i];        
    }
    for(int i=0; i<sizeR2; i++)
    {
        xR[i+sizeR1] = i*(0.4/(sizeR2-1.0))+0.6;
        dataR[i+sizeR1] = swingR[i];        
    }   

    int size3 = cmpData.size();
    float x3[size3];
    float data[size3];
    float std1[size3];
    float std2[size3];
    for(int i=0; i<size3; i++)
    {
        x3[i] = i*(1.0/(size3-1.0));
        data[i] = cmpData[i];        
        std1[i] = cmpStd1[i];
        std2[i] = cmpStd2[i];        
    }
    
    int plotStyle = ImPlotFlags_CanvasOnly;
    if(mLegend)
        plotStyle = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMousePos;
    ImPlot::SetNextPlotLimits(0.0, 1.0, yMin, yMax, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), "%", "deg", ImVec2(w,h), plotStyle, ImPlotAxisFlags_Lock)) {
        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.2f);
        ImPlot::PlotLine("left",xL,dataL,sizeL);                
        ImPlot::PlotLine("right",xR,dataR,sizeR);                        
        ImPlot::PlotShaded("real",x3,std1,std2,size3);                
        ImPlot::PlotLine("real",x3,data,size3);                        
        ImPlot::EndPlot();
    }
}

void
GLFWApp::
DrawJointTorque(std::string name, std::deque<double> data, double yMin, double yMax, double w, double h)
{
    int size = data.size();
    float x[size]; 
    float data_[size];
    for(int i=0; i<size; i++)
    {
        x[i] = i*(1.0/(size-1.0));
        data_[i] = data[i];
    }

    ImPlot::SetNextPlotLimits(0.0, 1.0, yMin, yMax, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), "%", "Nm", ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
        ImPlot::PlotLine("deg",x,data_,size);                        
        ImPlot::EndPlot();
    }
}

void
GLFWApp::
DrawJointTorque(std::string name, std::deque<double> data1, std::deque<double> data2, double yMin, double yMax, double w, double h)
{
    int size1 = data1.size();
    float x1[size1];    
    float data1_[size1];
    for(int i=0; i<size1; i++)
    {
        x1[i] = i*(1.0/(size1-1.0));
        data1_[i] = data1[i];        
    }

    int size2 = data2.size();
    float x2[size2];
    float data2_[size2];
    for(int i=0; i<size2; i++)
    {
        x2[i] = i*(1.0/(size2-1.0));
        data2_[i] = data2[i];        
    }
    
    ImPlot::SetNextPlotLimits(0.0, 1.0, yMin, yMax, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), "%", "Nm", ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
        ImPlot::PlotLine("char1",x1,data1_,size1);                
        ImPlot::PlotLine("char2",x2,data2_,size2);                
        ImPlot::EndPlot();
    }
}

bool
GLFWApp::
ShowCompareDataSelecter()
{   
    static int person_idx = -1;
    if(ImGui::Combo("person", &person_idx, "Person 0\0Person 1\0Person 2\0"))
    {
        switch(person_idx)
        {
            case 0: mComparePersonIdx = 0; break; 
            case 1: mComparePersonIdx = 1; break; 
            case 2: mComparePersonIdx = 2; break; 
        }
        return true;
    }
    return false;
}

void
GLFWApp::
DrawUiFrame_Analysis(double x, double y, double w, double h)
{
    ImGui::Begin("Analysis");                         
    ImGui::SetWindowPos("Analysis", ImVec2(x, y));    
    ImGui::SetWindowSize("Analysis", ImVec2(w, h));
    
    ImPlot::SetCurrentContext(mPlotContexts["Analysis"]);

    static bool compare = false;
    static bool legend = false;
    ImGui::Checkbox("Legend", &legend);
    mLegend = legend;
    ImGui::SameLine();
    ImGui::Checkbox("Compare", &compare);
    if(compare)
    { 
        ImGui::SameLine();
        this->ShowCompareDataSelecter();
    }
    ImGui::Separator();
    
    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_None))
    {
        JointData* jointData = mEnv->GetCharacter()->GetJointDatas(); 
        auto angles = jointData->GetAngles();
        auto anglesStanceLeft = jointData->GetAnglesStancePhaseLeftPrev();
        auto anglesSwingLeft = jointData->GetAnglesSwingPhaseLeftPrev();
        auto anglesStanceRight = jointData->GetAnglesStancePhaseRightPrev();
        auto anglesSwingRight = jointData->GetAnglesSwingPhaseRightPrev();
        
        auto torques = jointData->GetTorques();
        auto torquesPhase = jointData->GetTorquesGaitPhasePrev();

        if (ImGui::BeginTabItem("Joint Angle"))
        {            
            double w_ = w/3.0 - 10.0;
            double h_ = h/3.0 - 30.0;
           
            double yMin, yMax;
            std::string plotName, dataName;
            std::string plotNamePre, dataNamePre, namePost;
            std::deque<double> angleStanceL;
            std::deque<double> angleStanceR;
            std::deque<double> angleSwingL;
            std::deque<double> angleSwingR;
           
            for(int i=0; i<3; i++)
            {
                if(i==0){
                    plotNamePre = "Hip"; dataNamePre = "Femur";
                }
                else if(i==1){
                    plotNamePre = "Knee"; dataNamePre = "Tibia";
                }
                else if(i==2){
                    plotNamePre = "Ankle"; dataNamePre = "Talus";
                }
    
                for(int j=0; j<3; j++)
                {
                    if(j==0)
                        namePost = "_sagittal";
                    else if(j==1)
                        namePost = "_frontal";
                    else if(j==2)
                        namePost = "_transverse";
                    
                    for(int k=0; k<2; k++)
                    {
                        std::string cur;
                        if(k==0)
                            cur = "L";
                        else if(k==1)
                            cur = "R";

                        plotName = plotNamePre + namePost;
                        dataName = dataNamePre + cur + namePost;

                        if(k==0){
                            angleStanceL = anglesStanceLeft[dataName];
                            angleSwingL = anglesSwingLeft[dataName];
                        }                            
                        else if(k==1){
                            angleStanceR = anglesStanceRight[dataName];
                            angleSwingR = anglesSwingRight[dataName];                           
                        }                       
                       
                        if(dataName == "FemurL_sagittal" || dataName == "FemurL_transverse" || dataName == "TalusL_sagittal")
                        {
                            for(int i=0; i<angleStanceL.size(); i++)
                                angleStanceL[i] *= -1;  
                            for(int i=0; i<angleSwingL.size(); i++)
                                angleSwingL[i] *= -1;                        
                        }

                        if(dataName == "FemurR_sagittal" || dataName == "FemurR_frontal" || dataName == "TalusR_sagittal" || dataName == "TalusR_transverse")
                        {
                            for(int i=0; i<angleStanceR.size(); i++)
                                angleStanceR[i] *= -1;   
                            for(int i=0; i<angleSwingR.size(); i++)
                                angleSwingR[i] *= -1;                        
                        }
                    }

                    yMin = mJointAngleMinMax[plotName].first;
                    yMax = mJointAngleMinMax[plotName].second;
                    if(compare)
                    {
                        if(plotName == "Hip_sagittal" || plotName == "Hip_frontal" || plotName == "Hip_transverse"|| plotName == "Knee_sagittal" || plotName == "Ankle_sagittal" || plotName == "Ankle_transverse")
                            this->DrawJointAngle(plotName, angleStanceL, angleSwingL, angleStanceR, angleSwingR, mRealJointData[plotName], mRealJointData[plotName+"_std1"], mRealJointData[plotName +"_std2"], yMin, yMax, w_, h_);
                        else
                            this->DrawJointAngle(plotName, angleStanceL, angleSwingL, angleStanceR, angleSwingR, mRealJointData[plotName], yMin, yMax, w_, h_);                        
                    }
                    else{
                        this->DrawJointAngle(plotName, angleStanceL, angleSwingL, angleStanceR, angleSwingR, yMin, yMax, w_, h_);
                    }
                    // if(compare)
                    // {
                    //     if(plotName == "Hip_sagittal" || plotName == "Hip_frontal" || plotName == "Hip_transverse"|| plotName == "Knee_sagittal" || plotName == "Ankle_sagittal" || plotName == "Ankle_transverse")
                    //         this->DrawJointAngle(plotName, angleData1, angleData2, mRealJointData[plotName], mRealJointData[plotName+"_std1"], mRealJointData[plotName +"_std2"], yMin, yMax, w_, h_);
                    //     else
                    //         this->DrawJointAngle(plotName, angleData1, angleData2, mRealJointData[plotName], yMin, yMax, w_, h_);                        
                    // }
                    // else{
                    //     this->DrawJointAngle(plotName, angleData1, angleData2, yMin, yMax, w_, h_);
                    // }
                        

                    if(j<2)
                        ImGui::SameLine();
                }
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Joint Torque"))
        {   
            double w_ = w/3.0 - 10.0;
            double h_ = h/3.0 - 30.0;

            std::deque<double> torqueData;
            double yMin, yMax;
            std::string plotName, dataName;
            std::string plotNamePre, dataNamePre;
            std::string namePost;

            for(int i=0; i<6; i++)
            {
                if(i==0){
                    plotNamePre = "Hip"; dataNamePre = "FemurL";
                }
                else if(i==1){
                    plotNamePre = "Knee"; dataNamePre = "TibiaL";
                }
                else if(i==2){
                    plotNamePre = "Ankle"; dataNamePre = "TalusL";
                }
                else if(i==3){
                    plotNamePre = "Spine"; dataNamePre = "Spine";
                }
                else if(i==4){
                    plotNamePre = "Torso"; dataNamePre = "Torso";
                }
                else if(i==5){
                    plotNamePre = "Neck"; dataNamePre = "Neck";
                }
    
                for(int j=0; j<3; j++)
                {
                    if(j==0)
                        namePost = "_x";
                    else if(j==1)
                        namePost = "_y";
                    else if(j==2)
                        namePost = "_z";
                    
                    plotName = plotNamePre + namePost;
                    dataName = dataNamePre + namePost;
                    // torqueData = torques[dataName];
                    torqueData = torquesPhase[dataName];                    
                    yMin = mJointTorqueMinMax[plotName].first;
                    yMax = mJointTorqueMinMax[plotName].second;
                
                    if(compare)
                    {
                        //get data from mComparePersonIdx
                        // int size = torqueData.size()/2;
                        // std::deque<double> compareTorqueData(size);
                        // for(int i=0; i<compareTorqueData.size(); i++)
                        //     compareTorqueData[i] = torqueData[i] - 5.0;
                        this->DrawJointTorque(plotName, torqueData, yMin, yMax, w_, h_);
                        // this->DrawJointTorque(plotName, torqueData, compareTorqueData, yMin, yMax, w_, h_);
                    }
                    else
                        this->DrawJointTorque(plotName, torqueData, yMin, yMax, w_, h_);

                    if(j<2)
                        ImGui::SameLine();
                }
            }

            ImGui::EndTabItem();
        }

        MetabolicEnergy* metaEnergy = mEnv->GetCharacter()->GetMetabolicEnergy(); 
        const auto& energy = metaEnergy->GetHOUD06_map_deque();

        if (ImGui::BeginTabItem("Metabolic Energy"))
        {
            double w_ = w/3.0 - 10.0;
            double h_ = h/3.0 - 30.0;

            std::deque<double> energyData;
            std::string xAxisName = "%";
            std::string yAxisName = "Metabolic Energy";
            int idx = 0;
            for(auto e : energy)
            {   
                this->DrawDequeGraph(e.first, xAxisName, yAxisName, e.second, 0.0, 1.0, w_, h_);
                if(idx%3 < 2)
                    ImGui::SameLine();
                idx++;
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Activation Levels"))
        {
            // Eigen::VectorXd al = mEnv->GetCharacter()->GetActivationLevels();
            
            // int size = 20;
            // double xs[size];
            // float ys[size];
            // const char* labels[size];
            // for(int i=0; i<size; i++)
            // {
            //     labels[i] = std::to_string(i).c_str();
            //     xs[i] = (double)i;
            //     ys[i] = al[i];
            // }

            // ImPlot::SetNextPlotLimits(-0.5, 19.5, 0, 1, ImGuiCond_Always);
            // ImPlot::SetNextPlotTicksX((const double*)xs, 3, labels);
            
            // double w = mImguiWidth*(1.0-mUiWidthRatio) - 20;
            // double h = mImguiHeight*(1.0-mUiHeightRatio)/4.0 - 20;
            // if (ImPlot::BeginPlot("Bar Plot", "Muscle", "level",
            //                 ImVec2(w,h), 0, 0, 0))
            // {
            //     double level = 1.0+(3.0*a), 1.0, 1.0, 1.0
            //     ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1,1.0,1.0,1.0));
            //     ImPlot::SetLegendLocation(ImPlotLocation_South, ImPlotOrientation_Horizontal);
            //     ImPlot::PlotBars("Activation", ys, size, 0.1, 0);
            //     ImPlot::PopStyleColor();
            //     ImPlot::EndPlot();
            // }
                                   
            ImGui::EndTabItem();
        }
    }

    ImPlot::SetCurrentContext(mPlotContexts["Main"]);
    ImGui::End();
}

void
GLFWApp::
DrawDevice()
{
    if(mEnv->GetCharacter()->GetDevice_OnOff() && mDrawDevice)
	{
		isDrawDevice = true;
		DrawSkeleton(mEnv->GetDevice()->GetSkeleton());
		isDrawDevice = false;
        // if(mDrawGraph)
		// 	DrawDeviceSignals();
		// if(mDrawArrow)
		// 	DrawArrow();
	}
}

void 
GLFWApp::
DrawCharacter_()
{
    isDrawCharacter = true;
    this->DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());
    if(mEnv->GetUseMuscle())
        DrawMuscles(mEnv->GetCharacter()->GetMuscles());
    isDrawCharacter = false;
}

void
GLFWApp::
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
GLFWApp::
DrawReference()
{
    isDrawReference = true;

	Character* character = mEnv->GetCharacter();
	SkeletonPtr skeleton = character->GetSkeleton();

	Eigen::VectorXd cur_pos = skeleton->getPositions();

	skeleton->setPositions(character->GetReferencePositions());
	DrawBodyNode(skeleton->getRootBodyNode());

	skeleton->setPositions(cur_pos);

	isDrawReference = false;
}

void
GLFWApp::
DrawCharacter()
{
    if(mDrawCharacter)
        this->DrawCharacter_();

    if(mDrawTarget)
		this->DrawTarget();

	if(mDrawReference)
		this->DrawReference();
}


void 
GLFWApp::
DrawGround()
{
    auto ground = mEnv->GetGround();
    float y = ground->getBodyNode(0)->getTransform().translation()[1] +
              dynamic_cast<const BoxShape *>(ground->getBodyNode(
                      0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1] * 0.5;

    constexpr int N = 64;
    GLubyte imageData[N][N];
    for (int i = 0; i < N; i++)  {
        for (int j = 0; j < N; j++) {
            if ((i/(N/2) + j/(N/2)) % 2 == 0) {
                imageData[i][j] = (GLubyte)(256*0.8);
            }
            else {
                imageData[i][j] = (GLubyte)(256*0.7);
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

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);    
}


void 
GLFWApp::
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
GLFWApp::
DrawBodyNode(const BodyNode* bn)
{	
	if(!bn)
		return;

	glPushMatrix();
	Eigen::Affine3d tmp = bn->getRelativeTransform();
	glMultMatrixd(tmp.data());

    // if(bn->getName() == "FemurL" || bn->getName() == "FemurR" || bn->getName() == "TibiaL" || bn->getName() == "TibiaR" || bn->getName() == "TalusL" || bn->getName() == "TalusR")
	// 	mDrawCoordinate = true;
	// else
	// 	mDrawCoordinate = false;
    // mDrawCoordinate = true;

	auto sns = bn->getShapeNodesWith<VisualAspect>();
	for(const auto& sn : sns)
		DrawShapeFrame(sn);
    
	for(const auto& et : bn->getChildEntities())
		DrawEntity(et);

	glPopMatrix();
}

void 
GLFWApp::
DrawSkeleton(const SkeletonPtr& skel)
{
	DrawBodyNode(skel->getRootBodyNode());
}

void 
GLFWApp::
DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	const auto& va = sf->getVisualAspect();

	if(!va || va->isHidden())
		return;

	glPushMatrix();
	Eigen::Affine3d tmp = sf->getRelativeTransform();
	glMultMatrixd(tmp.data());

    Eigen::Vector4d color = va->getRGBA();
    if(isDrawCharacter)
    {
        if(mDrawOBJ)
            color << 0.75, 0.75, 0.75, 0.3;
        else
            color[3] = 0.8;
    }
    if(isDrawTarget)
		color << 0.6, 1.0, 0.6, 0.3;
	if(isDrawReference)
		color << 1.0, 0.6, 0.6, 0.3;
	if(isDrawDevice)
		color << 0.3, 0.3, 0.3, 1.0;

	DrawShape(sf->getShape().get(), color);
	glPopMatrix();
}

void 
GLFWApp::
DrawShape(const Shape* shape, const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	
    glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);   
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor4dv(color.data());
    if(mDrawOBJ == false)
	{
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
			float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			mShapeRenderer.renderMesh(mesh, false, y, color);
		}
	}   

    if(mDrawCoordinate && shape->is<SphereShape>()){
        Eigen::Vector3d o(0.0, 0.0, 0.0);
        Eigen::Vector3d x(0.1, 0.0, 0.0);
        Eigen::Vector3d y(0.0, 0.1, 0.0);
        Eigen::Vector3d z(0.0, 0.0, 0.1);
        double w = 3.0;
		DrawLine(o, x, red, w);
		DrawLine(o, y, green, w);
		DrawLine(o, z, blue, w);
	}

	glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DEPTH_TEST);    
}

void 
GLFWApp::
DrawMuscles(const std::vector<Muscle*>& muscles)
{
    glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	for (auto muscle : muscles)
	{
        double a = muscle->GetActivation();
        Eigen::Vector4d color(1.0+(3.0*a), 1.0, 1.0, 1.0);
        glColor4dv(color.data());

	    mShapeRenderer.renderMuscle(muscle);
	}
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_DEPTH_TEST);
}

void
GLFWApp::
DrawOriginCoord()
{
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glLineWidth(2.0);

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
       
    glDisable(GL_DEPTH_TEST);    
    glEnable(GL_LIGHTING);    
}

void
GLFWApp::
DrawLine(Eigen::Vector3d v0, Eigen::Vector3d v1, Eigen::Vector4d color, double lineWidth)
{
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glLineWidth(lineWidth);

    glColor3f(color[0], color[1], color[2]);
    
    glBegin(GL_LINES);
    glVertex3f(v0[0], v0[1], v0[2]);
    glVertex3f(v1[0], v1[1], v1[2]);
    glEnd();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
}

void 
GLFWApp::
keyboardPress(int key, int scancode, int action, int mods) 
{
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE: exit(0); break;
            case GLFW_KEY_SPACE: mSimulating = !mSimulating; break;
            case GLFW_KEY_R: this->Reset(); break;
            case GLFW_KEY_S: this->Update(); break;
            case GLFW_KEY_O: mDrawOBJ = !mDrawOBJ; break;
            case GLFW_KEY_F: mFocus = !mFocus; break;
            case GLFW_KEY_V: mViewMode = (mViewMode+1)%4; break;
            case GLFW_KEY_T: mDrawTarget = !mDrawTarget; break;
            case GLFW_KEY_C: mDrawCharacter = !mDrawCharacter; break;
            case GLFW_KEY_D: mDrawDevice = !mDrawDevice; break;
            default: break;
        }
    }
}

void 
GLFWApp::
mouseMove(double xpos, double ypos) 
{
    double deltaX = xpos - mMouseX;
    double deltaY = ypos - mMouseY;

    mMouseX = xpos;
    mMouseY = ypos;

    // if(mMouseX < mViewerWidth && mMouseY < mViewerHeight)
    if(true)
    {
        if(mRotate)
        {
            if (deltaX != 0 || deltaY != 0) 
            {
                if (mViewMode == 3)
                    mSplitTrackballs[mSplitIdx].updateBall(xpos, mViewerHeight - ypos);
                else 
                    mTrackball.updateBall(xpos, mViewerHeight - ypos);
            }
        }

        if(mTranslate)
        {
            Eigen::Matrix3d rot;
            if (mViewMode == 3) 
                rot = mSplitTrackballs[mSplitIdx].getRotationMatrix();
            else 
                rot = mTrackball.getRotationMatrix();

            mTrans += (1/mZoom) * rot.transpose() * Eigen::Vector3d(deltaX, -deltaY, 0.0);
        }    
    }
}

void 
GLFWApp::
mousePress(int button, int action, int mods) 
{
    if(action == GLFW_PRESS) 
    {
        mMouseDown = true;

        if(mViewMode == 3) 
        {
            for (int i=0; i<mSplitViewNum; i++) {
                double splitViewWidth = mViewerWidth/mSplitViewNum;
                if (mMouseX >= splitViewWidth*i && mMouseX < splitViewWidth*(i+1)) 
                    mSplitIdx = i;
            }
        }

        // if(mMouseX < mViewerWidth && mMouseY < mViewerHeight)
        if(true)
        {
            if(button == GLFW_MOUSE_BUTTON_LEFT) 
            {
                mRotate = true;
                if (mViewMode == 3) 
                    mSplitTrackballs[mSplitIdx].startBall(mMouseX, mViewerHeight - mMouseY);
                else 
                    mTrackball.startBall(mMouseX, mViewerHeight - mMouseY);                        
            }
            else if(button == GLFW_MOUSE_BUTTON_RIGHT) 
            {
                mTranslate = true;
            }
        }
    }
    else if(action == GLFW_RELEASE) 
    {
        mMouseDown = false;
        // if(mMouseX < mViewerWidth && mMouseY < mViewerHeight)
        if(true)
        {
            if (button == GLFW_MOUSE_BUTTON_LEFT)
                mRotate = false;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT)
                mTranslate = false;
        }
        
    }
}

void 
GLFWApp::
mouseScroll(double xoffset, double yoffset) 
{
    mZoom += yoffset * 0.01;
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
GLFWApp::
GetActionFromNN()
{
	Eigen::VectorXd state = mEnv->GetCharacter()->GetState();
	py::array_t<float> state_np = py::array_t<float>(state.rows());
	py::buffer_info state_buf = state_np.request(true);
	float* dest = reinterpret_cast<float*>(state_buf.ptr);

	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];

	py::object apply = rms_module.attr("apply_no_update");
	py::object state_np_tmp = apply(state_np);
	py::array_t<float> state_np_ = py::array_t<float>(state_np_tmp);

	py::object get_action = nn_module.attr("get_action");
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
GLFWApp::
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

}
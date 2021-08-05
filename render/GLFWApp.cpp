#include "GLFWApp.h"
#include <iostream>

namespace MASS
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
    ImPlot::DestroyContext();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(mWindow);
    glfwTerminate();
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
	
    mUiWidthRatio = 0.5;
    mUiHeightRatio = 0.2;
    mUiViewerRatio = 0.4;

    //window size
    mWindowWidth = WindowWidth; 
    mWindowHeight = WinodwHeight;
    
    mViewerWidth = mWindowWidth*(mUiViewerRatio);
    mViewerHeight = mWindowHeight;

	mImguiWidth = mWindowWidth - mViewerWidth;
    mImguiHeight = mWindowHeight;
    
    // mTrackball = std::make_unique<Trackball>();
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

	ImPlot::CreateContext();
    ImPlot::StyleColorsDark();
    ImPlot::SetColormap(ImPlotColormap_Default);
    ImPlot::GetStyle().AntiAliasedLines = true;
    
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
    mJointAngleMinMax["Hip_sagittal"] = std::pair(-40.0, 40.0);
    mJointAngleMinMax["Hip_frontal"] = std::pair(-20.0, 20.0);
    mJointAngleMinMax["Hip_transverse"] = std::pair(-20.0, 20.0);

    mJointAngleMinMax["Knee_sagittal"] = std::pair(-15.0, 75.0);
    mJointAngleMinMax["Knee_frontal"] = std::pair(-20.0, 20.0);
    mJointAngleMinMax["Knee_transverse"] = std::pair(-20.0, 20.0);

    mJointAngleMinMax["Ankle_sagittal"] = std::pair(-40.0, 40.0);
    mJointAngleMinMax["Ankle_frontal"] = std::pair(-20.0, 20.0);
    mJointAngleMinMax["Ankle_transverse"] = std::pair(-40.0, 0.0);
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
#if 1
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
#else
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

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
    
    // mEnv->GetReward();
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

    // bool show_demo_window = true;
    // if (show_demo_window)
    //     ImGui::ShowDemoWindow(&show_demo_window);

    this->DrawUiFrame_Manager();
    this->DrawUiFrame_SimState(x1, y1, w1, h1);
    this->DrawUiFrame_Learning(x2, y1, w2, h1);
    this->DrawUiFrame_Analysis(x1, y2, w1+w2, h2);

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
    // You can pass in a reference ImGuiStyle structure to compare to, revert to and save to
    // (without a reference style pointer, we will use one compared locally as a reference)
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
            ImPlot::ShowStyleSelector("ImPlot Style");
            ImPlot::ShowColormapSelector("ImPlot Colormap");
            float indent = ImGui::CalcItemWidth() - ImGui::GetFrameHeight();
            ImGui::Indent(ImGui::CalcItemWidth() - ImGui::GetFrameHeight());
            ImGui::Checkbox("Anti-Aliased Lines", &ImPlot::GetStyle().AntiAliasedLines);
            ImGui::Unindent(indent);

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

    ImGui::Text("Simulation FPS %.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
    ImGui::End();
}

void
GLFWApp::
DrawUiFrame_Learning(double x, double y, double w, double h)
{
    ImGui::Begin("Learning");                          // Create a window called "Hello, world!" and append into it.
    ImGui::SetWindowPos("Learning", ImVec2(x, y));    
    ImGui::SetWindowSize("Learning", ImVec2(w, h));

    ImGui::End();
}

void
GLFWApp::
DrawJointAngle(std::string name, std::deque<double> data, double yMin, double yMax, double w, double h)
{
    int size = data.size();
    float x[size]; 
    float data_[size];
    for(int i=0; i<size; i++)
    {
        x[i] = i*(1.0/(size-1.0));
        data_[i] = data[i]*180.0/M_PI;
    }

    ImPlot::SetNextPlotLimits(0.0, 1.0, yMin, yMax, ImGuiCond_Always);                
    if (ImPlot::BeginPlot(name.c_str(), "%", "deg", ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
        ImPlot::PlotLine("deg",x,data_,size);                
        ImPlot::EndPlot();
    }
}

void
GLFWApp::
DrawUiFrame_Analysis(double x, double y, double w, double h)
{
    ImGui::Begin("Analysis");                         
    ImGui::SetWindowPos("Analysis", ImVec2(x, y));    
    ImGui::SetWindowSize("Analysis", ImVec2(w, h));
    
    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_None))
    {
        JointData* data = mEnv->GetCharacter()->GetJointDatas(); 
        auto angles = data->GetAngles();

        if (ImGui::BeginTabItem("Joint Angle"))
        {
            double w = mImguiWidth/3.0 - 10.0;
            double h = (mImguiHeight*(1.0 - mUiHeightRatio))/3.0 - 30.0;

            std::deque<double> angleData;
            double yMin, yMax;
            std::string plotName, dataName;
            std::string plotNamePre, dataNamePre;
            
            for(int i=0; i<3; i++)
            {
                if(i==0)
                {
                    plotNamePre = "Hip";
                    dataNamePre = "FemurL";
                }
                else if(i==1)
                {
                    plotNamePre = "Knee";
                    dataNamePre = "TibiaL";
                }
                else if(i==2)
                {
                    plotNamePre = "Ankle";
                    dataNamePre = "TalusL";
                }
             
                plotName = plotNamePre + "_sagittal";
                dataName = dataNamePre + "_sagittal";
                angleData = angles[dataName];
                yMin = mJointAngleMinMax[plotName].first;
                yMax = mJointAngleMinMax[plotName].second;
                this->DrawJointAngle(plotName, angleData, yMin, yMax, w, h);

                ImGui::SameLine();
                plotName = plotNamePre + "_frontal";
                dataName = dataNamePre + "_frontal";
                angleData = angles[dataName];
                yMin = mJointAngleMinMax[plotName].first;
                yMax = mJointAngleMinMax[plotName].second;
                this->DrawJointAngle(plotName, angleData, yMin, yMax, w, h);


                ImGui::SameLine();
                plotName = plotNamePre + "_transverse";
                dataName = dataNamePre + "_transverse";
                angleData = angles[dataName];
                yMin = mJointAngleMinMax[plotName].first;
                yMax = mJointAngleMinMax[plotName].second;
                this->DrawJointAngle(plotName, angleData, yMin, yMax, w, h);
            }
            
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Metabolic Energy"))
        {

            ImGui::EndTabItem();
        }

        // if (ImGui::BeginTabItem("Configuration"))
        // {
        //     ImPlot::ShowStyleSelector("ImPlot Style");
        //     ImPlot::ShowColormapSelector("ImPlot Colormap");
        //     float indent = ImGui::CalcItemWidth() - ImGui::GetFrameHeight();
        //     ImGui::Indent(ImGui::CalcItemWidth() - ImGui::GetFrameHeight());
        //     ImGui::Checkbox("Anti-Aliased Lines", &ImPlot::GetStyle().AntiAliasedLines);
        //     ImGui::Unindent(indent);

        //     ImGui::EndTabItem();
        // }
    }


    // if (ImGui::CollapsingHeader("Configuration")) {
        // ImPlot::ShowStyleSelector("ImPlot Style");
        // ImPlot::ShowColormapSelector("ImPlot Colormap");
        // float indent = ImGui::CalcItemWidth() - ImGui::GetFrameHeight();
        // ImGui::Indent(ImGui::CalcItemWidth() - ImGui::GetFrameHeight());
        // ImGui::Checkbox("Anti-Aliased Lines", &ImPlot::GetStyle().AntiAliasedLines);
        // ImGui::Unindent(indent);
    // }

    // if (ImGui::CollapsingHeader("Joint Angle")) {
    //     int dNumL = 101; 
    //     int dNumR = 101;
    //     float xL[dNumL], xR[dNumR];
    //     float yL[dNumL], yR[dNumR]; 
    //     for (int i = 0; i < dNumL; ++i) {
    //         xL[i] = i * 1.0f;
    //         yL[i] = 10.0f + 10.0f * sinf(1.0f* xL[i]);             
    //     }
    //     for (int i = 0; i < dNumR; ++i) {
    //         xR[i] = i * 1.0f;
    //         yR[i] = 11.0f + 14.0f * sinf(1.0f* xR[i]);             
    //     }
      
    //     double w = mImguiWidth / 3.0 - 10.0;
    //     double h = (mImguiHeight * (1.0 - mUiHeightRatio)) / 3.0 - 30.0;

    //     for(int i=0; i<3; i++)
    //     {
    //         std::string plotNamePre, plotName;
    //         float x_min, x_max, y_min, y_max;
    //         x_min = 0.0; x_max = 100.0;
    //         if(i==0){
    //             plotNamePre = "Hip";
    //             y_min = -20.0; y_max = 40.0; 
    //         }
    //         else if(i==1){
    //             plotNamePre = "Knee";
    //             y_min = 0.0; y_max = 80.0;                 
    //         }
    //         else if(i==2){
    //             plotNamePre = "Ankle";
    //             y_min = -40.0; y_max = 20.0;                 
    //         }
            
    //         plotName = plotNamePre + "-Flexion-Extension";            
    //         ImPlot::SetNextPlotLimits(x_min, x_max, y_min, y_max, ImGuiCond_Always);                
    //         if (ImPlot::BeginPlot(plotName.c_str(), "%", "deg", ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
    //             ImPlot::PlotLine("left",xL,yL,dNumL);
    //             ImPlot::PlotLine("right",xR,yR,dNumR);                
    //             ImPlot::EndPlot();
    //         }

    //         ImGui::SameLine();
    //         plotName = plotNamePre + "-Aduction-Abduction";            
    //         ImPlot::SetNextPlotLimits(x_min, x_max, y_min, y_max, ImGuiCond_Always);                
    //         if (ImPlot::BeginPlot(plotName.c_str(), "%", "deg", ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
    //             // ImPlot::PushColormap(ImPlotColormap_Jet);
    //             ImPlot::PlotLine("left",xL,yL,dNumL);
    //             ImPlot::PlotLine("right",xR,yR,dNumR);
    //             ImPlot::EndPlot();
    //             // ImPlot::PopColormap(ImPlotColormap_Jet);
    //         }
            
    //         ImGui::SameLine();
    //         plotName = plotNamePre + "-Internal-External";            
    //         ImPlot::SetNextPlotLimits(x_min, x_max, y_min, y_max, ImGuiCond_Always);                
    //         if (ImPlot::BeginPlot(plotName.c_str(), "%", "deg", ImVec2(w,h), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_Lock)) {
    //             ImPlot::PlotLine("left",xL,yL,dNumL);
    //             ImPlot::PlotLine("right",xR,yR,dNumR);
    //             ImPlot::EndPlot();
    //         }
    //     }        
    // }

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
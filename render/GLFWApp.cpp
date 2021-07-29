#include "GLFWApp.h"
#include <iostream>

namespace MASS
{

#define WindowWidth 2160;
#define WinodwHeight 1080;

GLFWApp::
GLFWApp(Environment* env)
    : mEnv(env),mFocus(true),mSimulating(false),mNNLoaded(false),mMuscleNNLoaded(false),
      mMouseDown(false), mMouseDrag(false),mCapture(false),mRotate(false),mTranslate(false),mDisplayIter(0),
      isDrawCharacter(false),isDrawDevice(false),isDrawTarget(false),isDrawReference(false),
      mDrawOBJ(false),mDrawCharacter(true),mDrawDevice(true),mDrawTarget(false),mDrawReference(false),
      mSplitViewNum(2),mSplitIdx(0),mViewMode(0)
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
    this->InitGLFW();

    mDevice_On = mEnv->GetCharacter()->GetDevice_OnOff();
}

void
GLFWApp::
InitViewer()
{
    mZoom = 0.25;
    mPersp = 45.0;
	
    //window size
    mWindowWidth = WindowWidth; 
    mWindowHeight = WinodwHeight;
    
    mViewerWidth = mWindowWidth*3.0/5.0;
    mViewerHeight = mWindowHeight;

	mImguiWidth = mWindowWidth - mViewerWidth;
    mImguiHeight = mWindowHeight;
    
	// mTrackball = std::make_unique<Trackball>();
    double smaller = mWindowWidth < mWindowHeight ? mWindowWidth : mWindowHeight;
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
    ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
	// ImGui_ImplOpenGL3_Init("#version 150");

	ImPlot::CreateContext();
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
    double widthRatio = 0.4;
    double heightRatio = 0.3;

    double x1 = mViewerWidth;
    double x2 = mViewerWidth + widthRatio * mImguiWidth;
    double y1 = 0;
    double y2 = heightRatio * mImguiHeight;

    double w1 = widthRatio * mImguiWidth;
    double w2 = (1-widthRatio) * mImguiWidth;
    double h1 = heightRatio * mImguiHeight;
    double h2 = (1-heightRatio) * mImguiHeight;

    this->DrawUiFrame_SimState(x1, y1, w1, h1);
    // this->DrawUiFrame_Learning(x2, y1, w2, h1);
    // this->DrawUiFrame_Analysis(x1, y2, w1+w2, h2);
}

void
GLFWApp::
DrawUiFrame_SimState(double x, double y, double w, double h)
{
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    

     if (show_demo_window)
        ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
        static float f = 0.0f;
        static int counter = 0;


        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
        // ImGui::SetNextWindowPos(ImVec2(x, y));
        // ImGui::SetNextWindowSize(ImVec2(w, h));        

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &show_another_window);

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }
   
    // 3. Show another simple window.
    if (show_another_window)
    {
        ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
        ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
            show_another_window = false;
        ImGui::End();
    }

    // ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

   
    //     static float f = 0.0f;
    //     static int counter = 0;

    //     ImGui::Begin("Simulation");                   

    //     ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        
    //     ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //     ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //     if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //         counter++;
    //     ImGui::SameLine();
    //     ImGui::Text("counter = %d", counter);


    //     ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    //     ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void
GLFWApp::
DrawUiFrame_Learning(double x, double y, double w, double h)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(x, y));
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Learning");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        
        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);


        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
  
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void
GLFWApp::
DrawUiFrame_Analysis(double x, double y, double w, double h)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(x, y));
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImVec4 clear_color = ImVec4(0.65f, 0.75f, 0.80f, 1.00f);

    static float f = 0.0f;
    static int counter = 0;

        ImGui::Begin("Analysis");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        
        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);


        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
   
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// void 
// GLFWApp::
// DrawUiFrame() 
// {
//     ImGui_ImplOpenGL3_NewFrame();
//     ImGui_ImplGlfw_NewFrame();
//     ImGui::NewFrame();

//     ImGui::SetNextWindowPos(ImVec2(mViewerWidth, 0));
//     ImGui::SetNextWindowSize(ImVec2(mImguiWidth, mImguiHeight*0.5));

//     bool show_demo_window = false;
//     bool show_another_window = false;
//     ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

//     // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
//     if (show_demo_window)
//         ImGui::ShowDemoWindow(&show_demo_window);

//     // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
//     {
//         static float f = 0.0f;
//         static int counter = 0;

//         ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

//         ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
//         ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
//         ImGui::Checkbox("Another Window", &show_another_window);

//         ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
//         ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

//         if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
//             counter++;
//         ImGui::SameLine();
//         ImGui::Text("counter = %d", counter);

//         ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
//         ImGui::End();
//     }

//     // 3. Show another simple window.
//     if (show_another_window)
//     {
//         ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
//         ImGui::Text("Hello from another window!");
//         if (ImGui::Button("Close Me"))
//             show_another_window = false;
//         ImGui::End();
//     }

//     // Rendering
//     ImGui::Render();
//     // int display_w, display_h;
//     // glfwGetFramebufferSize(window, &display_w, &display_h);
//     // glViewport(mViewerWidth, 0, imguiWidth, height);
//     // glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
//     // glClear(GL_COLOR_BUFFER_BIT);
//     ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

//     // ImGui::Begin("Inspector");
//     // if (ImGui::BeginMenuBar()) {
//     //     if (ImGui::BeginMenu("File"))
//     //     {
//     //         ImGui::EndMenu();

//     //     }
//     //     ImGui::EndMenuBar();
//     // }

//     // if (ImGui::CollapsingHeader("Performance")) {
//     //     for (auto& [name, value] : perfStats) {
//     //         ImGui::Text("%15s: %3f ms", name.c_str(), value);
//     //     }
//     // }

//     // if (ImGui::CollapsingHeader("Checkpoints")) {
//     //     if (load_checkpoint_directory) {
//     //         if (ImGui::ListBoxHeader("##Select checkpoint")) {
//     //             for (int checkpoint_idx : checkpoints) {
//     //                 std::string label = std::string("checkpoint-") + std::to_string(checkpoint_idx);
//     //                 if (ImGui::Selectable(label.c_str(), selected_checkpoint == checkpoint_idx)) {
//     //                     selected_checkpoint = checkpoint_idx;
//     //                     std::string checkpoint_file = fs::path(checkpoint_path) /
//     //                                                   (std::string("checkpoint_") + std::to_string(checkpoint_idx)) /
//     //                                                   (std::string("checkpoint-") + std::to_string(checkpoint_idx));
//     //                     loadCheckpoint(checkpoint_file);
//     //                 }
//     //             }
//     //             ImGui::ListBoxFooter();
//     //         }
//     //     }
//     //     else {
//     //         ImGui::Text("Checkpoint: %s", checkpoint_path.c_str());
//     //     }
//     // }

//     // static bool showValueGraph = false;
//     // static bool showProbGraph = false;
//     // if (ImGui::CollapsingHeader("Parameters")) {
//     //     bool edited = false;
//     //     auto& config = mEnv->GetConfig();
//     //     auto& params = config.params;
//     //     auto categoryNames = params.getCategoryNames();
//     //     if (config.use_marginal_value_learning) {
//     //         ImGui::Checkbox("Show value graph", &showValueGraph);
//     //         if (config.use_adaptive_sampling) {
//     //             ImGui::Checkbox("Show prob graph", &showProbGraph);
//     //         }
//     //     }
//     //     for (auto& categoryName : categoryNames) {
//     //         if (ImGui::TreeNode(categoryName.c_str())) {
//     //             auto& indexMap = params.paramNameMap.at(categoryName).indexMap;
//     //             auto paramMap = params.getValuesInCategoryAsMap(categoryName);
//     //             for (auto& [name, value] : paramMap) {
//     //                 std::string label = name;
//     //                 if (categoryName == "length_ratio" && config.symmetry.count(name)) {
//     //                     label += " / ";
//     //                     label += config.symmetry.at(name);
//     //                 }
//     //                 float imguiValue = (float)value;
//     //                 auto bounds = params.getBounds(categoryName, name);
//     //                 ImGui::Text("%s", name.c_str());
//     //                 std::string sliderLabel = "##";
//     //                 sliderLabel += categoryName; sliderLabel += "_"; sliderLabel += name;
//     //                 ImGui::SetNextItemWidth(300);
//     //                 if (ImGui::SliderFloat(sliderLabel.c_str(), &imguiValue, bounds.first, bounds.second)) {
//     //                     params.getValue(categoryName, name) = value = (double)imguiValue;
//     //                     edited = true;
//     //                     if (config.use_marginal_value_learning) calculateMarginalValues();
//     //                 }
//     //                 if (config.use_marginal_value_learning) {
//     //                     int N = valueFunction.cols();
//     //                     std::vector<float> pValues(N);
//     //                     for (int k = 0; k < N; k++) {
//     //                         pValues[k] = bounds.first + (bounds.second - bounds.first) * float(k) / float(N-1);
//     //                     }
//     //                     uint32_t idx = indexMap.at(name);
//     //                     if (showValueGraph) {
//     //                         Eigen::VectorXf values = valueFunction.row(idx);
//     //                         ImPlot::SetNextPlotLimits(bounds.first, bounds.second,
//     //                                                   -0.01, values.maxCoeff()+0.01);
//     //                         auto lineLabel = categoryName + "_" + name + "_values";
//     //                         if (ImPlot::BeginPlot(lineLabel.c_str(), nullptr, nullptr, ImVec2(300, 200))) {
//     //                             ImPlot::PlotLine<float>(lineLabel.c_str(), pValues.data(), values.data(), pValues.size());
//     //                             ImPlot::EndPlot();
//     //                         }
//     //                     }
//     //                     if (showProbGraph) {
//     //                         Eigen::VectorXf probs = probDistFunction.row(idx);
//     //                         ImPlot::SetNextPlotLimits(bounds.first, bounds.second,
//     //                                                   -0.01, probs.maxCoeff()+0.01);
//     //                         auto lineLabel = categoryName + "_" + name + "_probs";
//     //                         if (ImPlot::BeginPlot(lineLabel.c_str(), nullptr, nullptr, ImVec2(300, 200))) {
//     //                             ImPlot::PlotLine<float>(lineLabel.c_str(), pValues.data(), probs.data(), pValues.size());
//     //                             ImPlot::EndPlot();
//     //                         }
//     //                     }
//     //                 }
//     //             }
//     //             ImGui::TreePop();
//     //         }
//     //     }
//     //     if (edited) {
//     //         mEnv->ResetWithCurrentParams();
//     //     }
//     // }

//     // if (ImGui::CollapsingHeader("Marginal Values")) {

//     // }
//     // ImGui::End();

//     // ImPlot::ShowDemoWindow();

//     // ImGui::End();
//     // ImGui::Render();
//     // ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
// }

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
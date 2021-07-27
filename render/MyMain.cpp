// #include <GL/glew.h>
// #include <GL/glut.h>
// #include "Environment.h"

// #include <dart/dart.hpp>
// #include <dart/gui/osg/osg.hpp>
// #include <dart/utils/urdf/urdf.hpp>
// #include <dart/utils/utils.hpp>

// #include "MyEventHandler.h"
// #include "MyWidget.h"
// #include "MyWorldNode.h"
// // #include "Window.h"

// int main(int argc,char** argv)
// {
// 	MASS::Environment* env = new MASS::Environment();

// 	if(argc==1)
// 	{
// 		std::cout<<"Provide Metadata.txt"<<std::endl;
// 		return 0;
// 	}
// 	env->Initialize(std::string(argv[1]), true);

// 	pybind11::scoped_interpreter guard{};
// 	glutInit(&argc, argv);

//     osg::ref_ptr<dart::gui::osg::ImGuiViewer> viewer = new dart::gui::osg::ImGuiViewer();
//     osg::ref_ptr<MyWorldNode> node;
    
//     if(argc == 2)
// 	{
//         node = new MyWorldNode(env, env->GetWorld());
//     }
// 	else if(argc == 3)
// 	{
//         node = new MyWorldNode(env, env->GetWorld(), argv[2]);
// 	}
// 	else if(argc == 4)
// 	{
//         node = new MyWorldNode(env, env->GetWorld(), argv[2], argv[3]);
// 	}
// 	else if(argc == 5)
// 	{
//         node = new MyWorldNode(env, env->GetWorld(), argv[2], argv[3], argv[4]);
// 	}
// 	else{
// 		std::cout<<"Please check your input"<<std::endl;
// 		std::cout<<"Input format is"<<std::endl;
// 		std::cout<<"./render/render metadata (model_network) (muscle_network) (device_network)"<<std::endl;

// 		return 0;
// 	}

//     std::cout << "before viewer setting" << std::endl;
//     viewer->addWorldNode(node);
//     viewer->getImGuiHandler()->addWidget(std::make_shared<MyWidget>(viewer, node));
//     viewer->setUpViewInWindow(0, 0, 1280, 960);
//     viewer->getCameraManipulator()->setHomePosition(
//       ::osg::Vec3d(5.14, 3.28, 6.28),
//       ::osg::Vec3d(1.00, 0.00, 0.00),
//       ::osg::Vec3d(0.00, 0.1, 0.00));  
//     viewer->getCamera()->setClearColor(
//         ::osg::Vec4d(1.0,1.0,1.0,1.0));      
//     viewer->setCameraManipulator(viewer->getCameraManipulator());
//     viewer->run();


// 	// MASS::Window* window;
// 	// if(argc == 2)
// 	// {
// 	// 	window = new MASS::Window(env);
// 	// }
// 	// else if(argc == 3)
// 	// {
// 	// 	window = new MASS::Window(env, argv[2]);
// 	// }
// 	// else if(argc == 4)
// 	// {
// 	// 	window = new MASS::Window(env, argv[2], argv[3]);
// 	// }
// 	// else if(argc == 5)
// 	// {
// 	// 	window = new MASS::Window(env, argv[2], argv[3], argv[4]);
// 	// }
// 	// else{
// 	// 	std::cout<<"Please check your input"<<std::endl;
// 	// 	std::cout<<"Input format is"<<std::endl;
// 	// 	std::cout<<"./render/render metadata (model_network) (muscle_network) (device_network)"<<std::endl;

// 	// 	return 0;
// 	// }

// 	// window->initWindow(1440,1080,"gui");

// 	// GLenum err = glewInit();
//     // if (err != GLEW_OK){
//     //     /* Problem: glewInit failed, something is seriously wrong. */
//     //     fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
//     // }

// 	// glutMainLoop();
// }

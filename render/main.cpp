#include <GL/glew.h>
#include <GL/glut.h>
#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "Window.h"
#include "dart/gui/gui.hpp"
namespace p = boost::python;
namespace np = boost::python::numpy;
int main(int argc,char** argv)
{
	MASS::Environment* env = new MASS::Environment();

	if(argc==1)
	{
		std::cout<<"Provide Metadata.txt"<<std::endl;
		return 0;
	}
	env->Initialize(std::string(argv[1]),true);

	Py_Initialize();
	np::initialize();
	glutInit(&argc, argv);

	MASS::Window* window;
	if(argc == 2)
	{
		window = new MASS::Window(env);
	}
	else if(argc == 3)
	{
		window = new MASS::Window(env, argv[2]);
	}
	else if(argc == 4)
	{
		window = new MASS::Window(env, argv[2], argv[3]);
	}
	else if(argc == 5)
	{
		window = new MASS::Window(env, argv[2], argv[3], argv[4]);
	}
	else{
		std::cout<<"Please check your input"<<std::endl;
		std::cout<<"Input format is"<<std::endl;
		std::cout<<"./render/render metadata (model_network) (muscle_network) (device_network)"<<std::endl;

		return 0;
	}

	window->initWindow(1440,1080,"gui");

	GLenum err = glewInit();
    if (err != GLEW_OK){
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }

	glutMainLoop();
}

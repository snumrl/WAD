#include "GLFWApp.h"
#include "Environment.h"
#include <pybind11/embed.h>

int main(int argc, char** argv) {
    MASS::Environment* env = new MASS::Environment();
    if(argc == 1)
	{
		std::cout<<"Provide Metadata.txt"<<std::endl;
		return 0;
	}
	env->Initialize(std::string(argv[1]), true);

    pybind11::scoped_interpreter guard{};

    MASS::GLFWApp* app;
    if(argc == 2){
        app = new MASS::GLFWApp(env);
    }
    else if(argc == 3){
        app = new MASS::GLFWApp(env, argv[2]);
    }
    else if(argc == 4){
        app = new MASS::GLFWApp(env, argv[2], argv[3]);
    }
    else{
        std::cout<<"Please check your input"<<std::endl;
		std::cout<<"Input format is"<<std::endl;
		std::cout<<"./render/render metadata (model_network) (muscle_network)"<<std::endl;

		return 0;
    }

    app->Initialize();
    app->StartLoop();

    return 0;
}


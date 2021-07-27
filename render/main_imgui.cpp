#include "GLFWApp.h"
#include <pybind11/embed.h>

int main(int argc, char** argv) {
    pybind11::scoped_interpreter guard{};

    MASS::GLFWApp app(argc, argv);
    app.startLoop();

    return 0;
}
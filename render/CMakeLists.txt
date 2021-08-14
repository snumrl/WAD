find_package(DART REQUIRED COMPONENTS gui gui-osg utils-urdf collision-bullet CONFIG)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${DART_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})

add_compile_options(-fvisibility=hidden)

set(srcs_render ShapeRenderer.cpp Window.cpp main.cpp)
add_executable(render ${srcs_render})
target_include_directories(render PRIVATE ../core/)
target_link_libraries(render 
            wad dart dart-gui imgui GL GLEW pybind11::embed) 

set(srcs_viewer ShapeRenderer_imgui.cpp GLFWApp.cpp GLFunctions.cpp main_imgui.cpp)
add_executable(viewer ${srcs_viewer})
target_include_directories(viewer PRIVATE ../core/)
target_link_libraries(viewer
            wad dart dart-gui imgui GL glad glfw pybind11::embed)


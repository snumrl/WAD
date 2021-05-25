cmake_minimum_required(VERSION 3.5.1)
project(render)

link_directories(../core/)
include_directories(../core/)

if (POLICY CMP0072)
    cmake_policy (SET CMP0072 OLD)
endif(POLICY CMP0072)

find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)
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

file(GLOB srcs "*.h" "*.cpp")
add_executable(render ${srcs})
target_link_libraries(render ${DART_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut GLEW::GLEW mss pybind11::embed)

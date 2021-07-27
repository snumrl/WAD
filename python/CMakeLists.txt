cmake_minimum_required(VERSION 3.5.1)
project(pymss)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

file(GLOB srcs "*.h" "*.cpp")
add_library(pymss SHARED ${srcs})
target_include_directories(pymss PRIVATE ../core/)
target_link_libraries(pymss ${DART_LIBRARIES} ${PYTHON_LIBRARIES} mss pybind11::embed)
set_target_properties(pymss PROPERTIES PREFIX "" )

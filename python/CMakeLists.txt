cmake_minimum_required(VERSION 3.5.1)
project(pymss)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

file(GLOB srcs "*.h" "*.cpp")
add_library(pymss SHARED ${srcs})
target_include_directories(pymss PRIVATE ../core/)
target_link_libraries(pymss ${DART_LIBRARIES} ${PYTHON_LIBRARIES} mss pybind11::embed)
set_target_properties(pymss PROPERTIES PREFIX "" )

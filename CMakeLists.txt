cmake_minimum_required(VERSION 3.5.1)
project(wad)

set(OpenGL_GL_PREFERENCE LEGACY)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-fPIC)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)

add_definitions(-DWAD_ROOT_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")

add_subdirectory( core )
add_subdirectory( render )
add_subdirectory( python )
add_subdirectory( libs )

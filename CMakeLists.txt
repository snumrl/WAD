cmake_minimum_required(VERSION 3.5.1)
project(mss)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

add_compile_options(-DHAVE_CSTDDEF)
add_definitions(-DMASS_ROOT_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")

find_package(Boost 1.66.0 EXACT COMPONENTS regex filesystem python3 numpy3)

add_subdirectory( core )
add_subdirectory( render )
add_subdirectory( python )

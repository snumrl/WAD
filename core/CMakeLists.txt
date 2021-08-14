cmake_minimum_required(VERSION 3.5.1)
project(wad)

SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")

find_package(DART REQUIRED COMPONENTS collision-bullet CONFIG)
find_package(TinyXML REQUIRED)

file(GLOB srcs "*.h" "*.cpp")
add_library(wad ${srcs})
target_link_libraries(wad ${DART_LIBRARIES} ${TinyXML_LIBRARIES})

# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jaedong/Lab/WAD

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jaedong/Lab/WAD/python

# Include any dependencies generated for this target.
include core/CMakeFiles/mss.dir/depend.make

# Include the progress variables for this target.
include core/CMakeFiles/mss.dir/progress.make

# Include the compile flags for this target's objects.
include core/CMakeFiles/mss.dir/flags.make

core/CMakeFiles/mss.dir/BVH.cpp.o: core/CMakeFiles/mss.dir/flags.make
core/CMakeFiles/mss.dir/BVH.cpp.o: ../core/BVH.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaedong/Lab/WAD/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object core/CMakeFiles/mss.dir/BVH.cpp.o"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mss.dir/BVH.cpp.o -c /home/jaedong/Lab/WAD/core/BVH.cpp

core/CMakeFiles/mss.dir/BVH.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mss.dir/BVH.cpp.i"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaedong/Lab/WAD/core/BVH.cpp > CMakeFiles/mss.dir/BVH.cpp.i

core/CMakeFiles/mss.dir/BVH.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mss.dir/BVH.cpp.s"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaedong/Lab/WAD/core/BVH.cpp -o CMakeFiles/mss.dir/BVH.cpp.s

core/CMakeFiles/mss.dir/BVH.cpp.o.requires:

.PHONY : core/CMakeFiles/mss.dir/BVH.cpp.o.requires

core/CMakeFiles/mss.dir/BVH.cpp.o.provides: core/CMakeFiles/mss.dir/BVH.cpp.o.requires
	$(MAKE) -f core/CMakeFiles/mss.dir/build.make core/CMakeFiles/mss.dir/BVH.cpp.o.provides.build
.PHONY : core/CMakeFiles/mss.dir/BVH.cpp.o.provides

core/CMakeFiles/mss.dir/BVH.cpp.o.provides.build: core/CMakeFiles/mss.dir/BVH.cpp.o


core/CMakeFiles/mss.dir/Character.cpp.o: core/CMakeFiles/mss.dir/flags.make
core/CMakeFiles/mss.dir/Character.cpp.o: ../core/Character.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaedong/Lab/WAD/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object core/CMakeFiles/mss.dir/Character.cpp.o"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mss.dir/Character.cpp.o -c /home/jaedong/Lab/WAD/core/Character.cpp

core/CMakeFiles/mss.dir/Character.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mss.dir/Character.cpp.i"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaedong/Lab/WAD/core/Character.cpp > CMakeFiles/mss.dir/Character.cpp.i

core/CMakeFiles/mss.dir/Character.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mss.dir/Character.cpp.s"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaedong/Lab/WAD/core/Character.cpp -o CMakeFiles/mss.dir/Character.cpp.s

core/CMakeFiles/mss.dir/Character.cpp.o.requires:

.PHONY : core/CMakeFiles/mss.dir/Character.cpp.o.requires

core/CMakeFiles/mss.dir/Character.cpp.o.provides: core/CMakeFiles/mss.dir/Character.cpp.o.requires
	$(MAKE) -f core/CMakeFiles/mss.dir/build.make core/CMakeFiles/mss.dir/Character.cpp.o.provides.build
.PHONY : core/CMakeFiles/mss.dir/Character.cpp.o.provides

core/CMakeFiles/mss.dir/Character.cpp.o.provides.build: core/CMakeFiles/mss.dir/Character.cpp.o


core/CMakeFiles/mss.dir/DARTHelper.cpp.o: core/CMakeFiles/mss.dir/flags.make
core/CMakeFiles/mss.dir/DARTHelper.cpp.o: ../core/DARTHelper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaedong/Lab/WAD/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object core/CMakeFiles/mss.dir/DARTHelper.cpp.o"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mss.dir/DARTHelper.cpp.o -c /home/jaedong/Lab/WAD/core/DARTHelper.cpp

core/CMakeFiles/mss.dir/DARTHelper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mss.dir/DARTHelper.cpp.i"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaedong/Lab/WAD/core/DARTHelper.cpp > CMakeFiles/mss.dir/DARTHelper.cpp.i

core/CMakeFiles/mss.dir/DARTHelper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mss.dir/DARTHelper.cpp.s"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaedong/Lab/WAD/core/DARTHelper.cpp -o CMakeFiles/mss.dir/DARTHelper.cpp.s

core/CMakeFiles/mss.dir/DARTHelper.cpp.o.requires:

.PHONY : core/CMakeFiles/mss.dir/DARTHelper.cpp.o.requires

core/CMakeFiles/mss.dir/DARTHelper.cpp.o.provides: core/CMakeFiles/mss.dir/DARTHelper.cpp.o.requires
	$(MAKE) -f core/CMakeFiles/mss.dir/build.make core/CMakeFiles/mss.dir/DARTHelper.cpp.o.provides.build
.PHONY : core/CMakeFiles/mss.dir/DARTHelper.cpp.o.provides

core/CMakeFiles/mss.dir/DARTHelper.cpp.o.provides.build: core/CMakeFiles/mss.dir/DARTHelper.cpp.o


core/CMakeFiles/mss.dir/Device.cpp.o: core/CMakeFiles/mss.dir/flags.make
core/CMakeFiles/mss.dir/Device.cpp.o: ../core/Device.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaedong/Lab/WAD/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object core/CMakeFiles/mss.dir/Device.cpp.o"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mss.dir/Device.cpp.o -c /home/jaedong/Lab/WAD/core/Device.cpp

core/CMakeFiles/mss.dir/Device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mss.dir/Device.cpp.i"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaedong/Lab/WAD/core/Device.cpp > CMakeFiles/mss.dir/Device.cpp.i

core/CMakeFiles/mss.dir/Device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mss.dir/Device.cpp.s"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaedong/Lab/WAD/core/Device.cpp -o CMakeFiles/mss.dir/Device.cpp.s

core/CMakeFiles/mss.dir/Device.cpp.o.requires:

.PHONY : core/CMakeFiles/mss.dir/Device.cpp.o.requires

core/CMakeFiles/mss.dir/Device.cpp.o.provides: core/CMakeFiles/mss.dir/Device.cpp.o.requires
	$(MAKE) -f core/CMakeFiles/mss.dir/build.make core/CMakeFiles/mss.dir/Device.cpp.o.provides.build
.PHONY : core/CMakeFiles/mss.dir/Device.cpp.o.provides

core/CMakeFiles/mss.dir/Device.cpp.o.provides.build: core/CMakeFiles/mss.dir/Device.cpp.o


core/CMakeFiles/mss.dir/Environment.cpp.o: core/CMakeFiles/mss.dir/flags.make
core/CMakeFiles/mss.dir/Environment.cpp.o: ../core/Environment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaedong/Lab/WAD/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object core/CMakeFiles/mss.dir/Environment.cpp.o"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mss.dir/Environment.cpp.o -c /home/jaedong/Lab/WAD/core/Environment.cpp

core/CMakeFiles/mss.dir/Environment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mss.dir/Environment.cpp.i"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaedong/Lab/WAD/core/Environment.cpp > CMakeFiles/mss.dir/Environment.cpp.i

core/CMakeFiles/mss.dir/Environment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mss.dir/Environment.cpp.s"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaedong/Lab/WAD/core/Environment.cpp -o CMakeFiles/mss.dir/Environment.cpp.s

core/CMakeFiles/mss.dir/Environment.cpp.o.requires:

.PHONY : core/CMakeFiles/mss.dir/Environment.cpp.o.requires

core/CMakeFiles/mss.dir/Environment.cpp.o.provides: core/CMakeFiles/mss.dir/Environment.cpp.o.requires
	$(MAKE) -f core/CMakeFiles/mss.dir/build.make core/CMakeFiles/mss.dir/Environment.cpp.o.provides.build
.PHONY : core/CMakeFiles/mss.dir/Environment.cpp.o.provides

core/CMakeFiles/mss.dir/Environment.cpp.o.provides.build: core/CMakeFiles/mss.dir/Environment.cpp.o


core/CMakeFiles/mss.dir/Muscle.cpp.o: core/CMakeFiles/mss.dir/flags.make
core/CMakeFiles/mss.dir/Muscle.cpp.o: ../core/Muscle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaedong/Lab/WAD/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object core/CMakeFiles/mss.dir/Muscle.cpp.o"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mss.dir/Muscle.cpp.o -c /home/jaedong/Lab/WAD/core/Muscle.cpp

core/CMakeFiles/mss.dir/Muscle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mss.dir/Muscle.cpp.i"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaedong/Lab/WAD/core/Muscle.cpp > CMakeFiles/mss.dir/Muscle.cpp.i

core/CMakeFiles/mss.dir/Muscle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mss.dir/Muscle.cpp.s"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaedong/Lab/WAD/core/Muscle.cpp -o CMakeFiles/mss.dir/Muscle.cpp.s

core/CMakeFiles/mss.dir/Muscle.cpp.o.requires:

.PHONY : core/CMakeFiles/mss.dir/Muscle.cpp.o.requires

core/CMakeFiles/mss.dir/Muscle.cpp.o.provides: core/CMakeFiles/mss.dir/Muscle.cpp.o.requires
	$(MAKE) -f core/CMakeFiles/mss.dir/build.make core/CMakeFiles/mss.dir/Muscle.cpp.o.provides.build
.PHONY : core/CMakeFiles/mss.dir/Muscle.cpp.o.provides

core/CMakeFiles/mss.dir/Muscle.cpp.o.provides.build: core/CMakeFiles/mss.dir/Muscle.cpp.o


core/CMakeFiles/mss.dir/Utils.cpp.o: core/CMakeFiles/mss.dir/flags.make
core/CMakeFiles/mss.dir/Utils.cpp.o: ../core/Utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jaedong/Lab/WAD/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object core/CMakeFiles/mss.dir/Utils.cpp.o"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mss.dir/Utils.cpp.o -c /home/jaedong/Lab/WAD/core/Utils.cpp

core/CMakeFiles/mss.dir/Utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mss.dir/Utils.cpp.i"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jaedong/Lab/WAD/core/Utils.cpp > CMakeFiles/mss.dir/Utils.cpp.i

core/CMakeFiles/mss.dir/Utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mss.dir/Utils.cpp.s"
	cd /home/jaedong/Lab/WAD/python/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jaedong/Lab/WAD/core/Utils.cpp -o CMakeFiles/mss.dir/Utils.cpp.s

core/CMakeFiles/mss.dir/Utils.cpp.o.requires:

.PHONY : core/CMakeFiles/mss.dir/Utils.cpp.o.requires

core/CMakeFiles/mss.dir/Utils.cpp.o.provides: core/CMakeFiles/mss.dir/Utils.cpp.o.requires
	$(MAKE) -f core/CMakeFiles/mss.dir/build.make core/CMakeFiles/mss.dir/Utils.cpp.o.provides.build
.PHONY : core/CMakeFiles/mss.dir/Utils.cpp.o.provides

core/CMakeFiles/mss.dir/Utils.cpp.o.provides.build: core/CMakeFiles/mss.dir/Utils.cpp.o


# Object files for target mss
mss_OBJECTS = \
"CMakeFiles/mss.dir/BVH.cpp.o" \
"CMakeFiles/mss.dir/Character.cpp.o" \
"CMakeFiles/mss.dir/DARTHelper.cpp.o" \
"CMakeFiles/mss.dir/Device.cpp.o" \
"CMakeFiles/mss.dir/Environment.cpp.o" \
"CMakeFiles/mss.dir/Muscle.cpp.o" \
"CMakeFiles/mss.dir/Utils.cpp.o"

# External object files for target mss
mss_EXTERNAL_OBJECTS =

../core/libmss.a: core/CMakeFiles/mss.dir/BVH.cpp.o
../core/libmss.a: core/CMakeFiles/mss.dir/Character.cpp.o
../core/libmss.a: core/CMakeFiles/mss.dir/DARTHelper.cpp.o
../core/libmss.a: core/CMakeFiles/mss.dir/Device.cpp.o
../core/libmss.a: core/CMakeFiles/mss.dir/Environment.cpp.o
../core/libmss.a: core/CMakeFiles/mss.dir/Muscle.cpp.o
../core/libmss.a: core/CMakeFiles/mss.dir/Utils.cpp.o
../core/libmss.a: core/CMakeFiles/mss.dir/build.make
../core/libmss.a: core/CMakeFiles/mss.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jaedong/Lab/WAD/python/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX static library ../../core/libmss.a"
	cd /home/jaedong/Lab/WAD/python/core && $(CMAKE_COMMAND) -P CMakeFiles/mss.dir/cmake_clean_target.cmake
	cd /home/jaedong/Lab/WAD/python/core && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mss.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
core/CMakeFiles/mss.dir/build: ../core/libmss.a

.PHONY : core/CMakeFiles/mss.dir/build

core/CMakeFiles/mss.dir/requires: core/CMakeFiles/mss.dir/BVH.cpp.o.requires
core/CMakeFiles/mss.dir/requires: core/CMakeFiles/mss.dir/Character.cpp.o.requires
core/CMakeFiles/mss.dir/requires: core/CMakeFiles/mss.dir/DARTHelper.cpp.o.requires
core/CMakeFiles/mss.dir/requires: core/CMakeFiles/mss.dir/Device.cpp.o.requires
core/CMakeFiles/mss.dir/requires: core/CMakeFiles/mss.dir/Environment.cpp.o.requires
core/CMakeFiles/mss.dir/requires: core/CMakeFiles/mss.dir/Muscle.cpp.o.requires
core/CMakeFiles/mss.dir/requires: core/CMakeFiles/mss.dir/Utils.cpp.o.requires

.PHONY : core/CMakeFiles/mss.dir/requires

core/CMakeFiles/mss.dir/clean:
	cd /home/jaedong/Lab/WAD/python/core && $(CMAKE_COMMAND) -P CMakeFiles/mss.dir/cmake_clean.cmake
.PHONY : core/CMakeFiles/mss.dir/clean

core/CMakeFiles/mss.dir/depend:
	cd /home/jaedong/Lab/WAD/python && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jaedong/Lab/WAD /home/jaedong/Lab/WAD/core /home/jaedong/Lab/WAD/python /home/jaedong/Lab/WAD/python/core /home/jaedong/Lab/WAD/python/core/CMakeFiles/mss.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : core/CMakeFiles/mss.dir/depend


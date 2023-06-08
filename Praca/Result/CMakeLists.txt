cmake_minimum_required(VERSION 3.5)

project(MyProject)

# Set the path to the Shark library
#set(SHARK_ROOT_DIR "/usr/local/lib/libshark.a")
#set(SHARK_DIR "~/lib/Shark/build")
set(Boost_USE_MULTITHREADED ON)

# Find the Shark package
#find_package(Boost 1.74 REQUIRED serialization)
#find_package(Shark REQUIRED)
#find_package(OpenMP)

#set(USE_OPENMP 1)

#add_library(openblas STATIC IMPORTED)

#if (OPENMP_FOUND)
#	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

find_package(Shogun REQUIRED)
find_package(dlib REQUIRED)

# Add the main.cpp source file to the project
add_executable(MyApp test.cpp)

target_include_directories(MyApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
					 ${Boost_INCLUDE_DIRS})
					# ${Shark_INCLUDE_DIRS})
#set_property(TARGET openblas PROPERTY IMPORTED_LOCATION "/usr/local/lib/libopenblas.a")

# Link against the libraries
target_link_libraries(MyApp dlib shogun)# Shark)

# Set the include directories
target_include_directories(MyApp PRIVATE ${DLIB_INCLUDE_DIRS} ${SHOGUN_INCLUDE_DIRS})
set_property(TARGET MyApp PROPERTY CXX_STANDARD 14)

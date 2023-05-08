cmake_minimum_required(VERSION 3.5)

project(MyProject)

# Set the path to the Shark library
set(SHARK_ROOT_DIR "/usr/local/lib/libshark.a")
set(SHARK_DIR "~/lib/Shark/build")
set(Boost_USE_MULTITHREADED ON)

# Find the Shark package
find_package(Boost 1.74 REQUIRED serialization)
find_package(Shark REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#find_package(Shogun REQUIRED)

# Add the main.cpp source file to the project
add_executable(MyApp test.cpp)

target_include_directories(MyApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
					 ${Boost_INCLUDE_DIRS}
					 ${Shark_INCLUDE_DIRS})
# Link against the Shark library
target_link_libraries(MyApp /usr/local/lib/libshark.a ${Boost_LIBRARIES})
#target_link_libraries(MyApp Shogun)

# Set the include directorie
#set_property(TARGET MyApp PROPERTY CXX_STANDARD 14)
set_property(TARGET MyApp PROPERTY CXX_STANDARD 20)
target_compile_options(MyApp PRIVATE "-fopenmp")

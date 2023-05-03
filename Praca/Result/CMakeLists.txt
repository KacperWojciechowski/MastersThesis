cmake_minimum_required(VERSION 3.5)

project(MyProject)

# Set the path to the Shark library
set(SHARK_ROOT_DIR "/usr/local/lib/libshark.a")

# Find the Shark package
find_package(Shark REQUIRED)
#find_package(Shogun REQUIRED)

# Add the main.cpp source file to the project
add_executable(MyApp test.cpp)

target_include_directories(MyApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against the Shark library
target_link_libraries(MyApp Shark)

# Set the include directories
target_include_directories(MyApp PRIVATE ${SHARK_INCLUDE_DIRS})
set_property(TARGET MyApp PROPERTY CXX_STANDARD 14)
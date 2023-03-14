#include <Dlib/matrix.h>
#include <fstream>
#include <iostream>

using namespace Dlib;

// [...]


matrix<double> data;
std::ifstream file("data_file.csv");
file >> data;
std::cout << data << std::endl;
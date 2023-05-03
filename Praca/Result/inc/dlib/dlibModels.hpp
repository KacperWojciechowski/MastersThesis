#pragma once

#include <inc/dlib/linear.hpp>
#include <inc/dlib/linear.hpp>
#include <inc/dlib/svm.hpp>
#include <inc/dlib/neural.hpp>

#include <fstream>

inline void dlibModels()
{
    using namespace dlib;
    
    matrix<double> classData;
    matrix<double> regData;
    std::ifstream classFile("wdbc_data_with_labels.csv");
    std::ifstream regFile("IronGlutathione.csv");
    classFile >> classData;
    regFile >> regData;

    std::cout << classData << std::endl << std::endl;
    std::cout << regData << std::endl << std::endl;
}
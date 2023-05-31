#pragma once

#include <inc/dlib/linear.hpp>
#include <inc/dlib/svm.hpp>

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

    auto classInputs = dlib::subm(classData, 0, 0, classData.nr(), classData.nc()-1);
    auto classOutputs = dlib::subm(classData, 0, classData.nc()-1, classData.nr(), 1);

    auto regInputs = dlib::subm(regData, 0, 0, regData.nr(), regData.nc()-1);
    auto regOutputs = dlib::subm(regData, 0, regData.nc()-1, regData.nr(), 1);

    using ClassSampleType = matrix<double, 11, 1>;
    using RegSampleType = matrix<double, 5, 1>;

    auto classTrainSamplesCount = static_cast<long>(classData.nr() * 0.8);
    auto regTrainSamplesCount = static_cast<long>(regData.nr() * 0.8);

    std::vector<classSampleType> classTestSamples;
    std::vector<double> classTestLabels;
    for (int row = 0; row < classTrainSamplesCount; row++)
    {
        classTestSamples.emplace_back(dlib::reshape_to_column_vector(
	    dlib::subm_clipped(inputs, row, 0, 1, classData.nc())));
	classTestLabels.emplace_back(classOutputs(row, 0));
    }

    std::vector<classSampleType> classTrainSamples;
    std::vector<double> classTrainLabels;
    for (int row = classTrainSamplesCount; row < classInputs.nr(); ++row)
    {
        classTrainSamples.emplace_back(dlib::reshape_to_column_vector(
	    dlib::subm_clipped(classInputs, row, 0, 1, classData.nc())));
	classTrainLabels.emplace_back(classOutputs(row, 0));
    }


}

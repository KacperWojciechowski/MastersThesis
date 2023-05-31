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
    // rozpakowanie danych klasyfikacyjnych
    auto classInputs = dlib::subm(classData, 0, 0, classData.nr(), 
                                  classData.nc()-1);
    auto classOutputs = dlib::subm(classData, 0, classData.nc()-1, 
                                   classData.nr(), 1);
    // rozpakowanie danych regresyjnych
    auto regInputs = dlib::subm(regData, 0, 0, regData.nr(), regData.nc()-1);
    auto regOutputs = dlib::subm(regData, 0, regData.nc()-1, regData.nr(), 1);

    using ClassSampleType = matrix<double, 11, 1>;
    using RegSampleType = matrix<double, 5, 1>;
    // obliczenie granicy podziału danych
    auto classTrainSamplesCount = static_cast<long>(classData.nr() * 0.8);
    auto regTrainSamplesCount = static_cast<long>(regData.nr() * 0.8);
    // przepakowanie testowych danych klasyfikacyjnych
    std::vector<classSampleType> classTestSamples;
    std::vector<double> classTestLabels;
    for (int row = 0; row < classTrainSamplesCount; row++)
    {
        classTestSamples.emplace_back(dlib::reshape_to_column_vector(
	    dlib::subm_clipped(classInputs, row, 0, 1, classData.nc())));
	classTestLabels.emplace_back(classOutputs(row, 0));
    }
    // przepakowanie treningowych danych klasyfikacyjnych
    std::vector<classSampleType> classTrainSamples;
    std::vector<double> classTrainLabels;
    for (int row = classTrainSamplesCount; row < classInputs.nr(); ++row)
    {
        classTrainSamples.emplace_back(dlib::reshape_to_column_vector(
	        dlib::subm_clipped(classInputs, row, 0, 1, classData.nc())));
	    classTrainLabels.emplace_back(classOutputs(row, 0));
    }
    // przepakowanie testowych danych regresyjnych
    std::vector<classSampleType> regTestSamples;
    std::vector<double> regTestLabels;
    for (int row = 0; row < regTrainSamplesCount; row++)
    {
        regTestSamples.emplace_back(dlib::reshape_to_column_vector(
	        dlib::subm_clipped(regInputs, row, 0, 1, regData.nc())));
	    regTestLabels.emplace_back(regOutputs(row, 0));
    }
    // przepakowanie treningowych danych regresyjnych
    std::vector<classSampleType> regTrainSamples;
    std::vector<double> regTrainLabels;
    for (int row = regTrainSamplesCount; row < regInputs.nr(); ++row)
    {
        regTrainSamples.emplace_back(dlib::reshape_to_column_vector(
	        dlib::subm_clipped(regInputs, row, 0, 1, regData.nc())));
	    regTrainLabels.emplace_back(regOutputs(row, 0));
    }
}

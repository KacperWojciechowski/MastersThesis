#pragma once

#include <dlib/matrix.h>
#include <dlib/svm_threaded.h>
#include <inc/dlib/eval.hpp>
#include <vector>

inline void dlibSVM(std::vector<dlib::matrix<double>> data,
                    std::vector<double> labels)
{
    using namespace dlib;
    // podział danych
    auto dataSplit = data.begin() + data.size() * 0.8;
    auto trainData = std::vector<matrix<double>>(
        data.begin(), dataSplit);
    auto testData = std::vector<matrix<double>>(
        dataSplit, data.end());
    auto labelSplit = labels.begin() + labels.size() * 0.8;
    auto trainLabels = std::vector<matrix<double>>(
        labels.begin(), labelSplit);
    auto testLabels = std::vector<matrix<double>>(
        labelSplit, labels.end());   
    
    using OVOTrainer = one_vs_one_trainer<any_trainer<double>>;
    using Kernel = radial_basis_kernel<double>;
    // utworzenie trenera maszyny wektorów nośnych
    svm_nu_trainer<Kernel> svmTrainer;
    svmTrainer.set_kernel(Kernel(0.1));
    // utworzenie trenera klasyfikatora wieloklasowego
    OVOTrainer trainer;
    trainer.set_trainer(svmTrainer);
    // utworzenie modelu
    one_vs_one_decision_function<OVOTrainer> model = 
        trainer.train(trainData, trainLabels);
    // ewaluacja
    std::cout << "----- Dlib SVM -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = std::vector<double>(trainData.size());
    for (auto& sample : trainData)
    {
        predictions.emplace_back(model(sample));
    }
    dlibEval(predictions, trainLabels, Task::CLASSIFICATION);
    predictions.clear();
    std::cout << "Test data:" << std::endl;
    for (auto& sample : testData)
    {
        predictions.emplace_back(model(sample));
    }
    dlibEval(predictions, testLabels, Task::CLASSIFICATION);
}
#pragma once

#include <dlib/matrix.h>
#include <dlib/svm_threaded.h>

#include <inc/dlib/eval.hpp>

#include <vector>

inline void dlibKRR(
    std::vector<dlib::matrix<double>> trainData,
    std::vector<dlib::matrix<double>> testData,
    std::vector<double> trainLabels,
    std::vector<double> testLabels)
{
    using namespace dlib;
    
    using OVOTrainer = one_vs_one_trainer<any_trainer<double>>;
    using Kernel = radial_basis_kernel<double>;
    // utworzenie trenera maszyny wektorów nośnych
    krr_trainer<Kernel> krrTrainer;
    krrTrainer.set_kernel(Kernel(0.1));
    // utworzenie trenera klasyfikatora wieloklasowego
    OVOTrainer trainer;
    trainer.set_trainer(krrTrainer);
    // utworzenie modelu
    one_vs_one_decision_function<OVOTrainer> model = 
        trainer.train(trainData, trainLabels);
    // ewaluacja
    std::cout << "----- Dlib KRR -----" << std::endl;
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
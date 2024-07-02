#pragma once

#include <dlib/matrix.h>
#include <dlib/svm.h>

#include <inc/dlib/eval.hpp>

inline void dlibLinear(
    std::vector<dlib::matrix<double, 5, 1>> trainData,
    std::vector<dlib::matrix<double, 5, 1>> testData,
    std::vector<double> trainLabels,
    std::vector<double> testLabels)
{
    using namespace dlib;
    // creating and configuring the trainer and kernel
    using linearKernel = linear_kernel<matrix<double, 5, 1>>;
    krr_trainer<linearKernel> trainer;
    trainer.set_kernel(linearKernel());
    // training
    decision_function<linearKernel> model = trainer.train(
        trainData, trainLabels);
    // evaluation
    std::cout << "----- Dlib Linear -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = std::vector<double>(trainData.size());
    for (auto& sample : trainData)
    {
        predictions.emplace_back(model(sample));
    }
    dlibEval(predictions, trainLabels, Task::REGRESSION);
    predictions.clear();
    std::cout << "Test data:" << std::endl;
    for (auto& sample : testData)
    {
        predictions.emplace_back(model(sample));
    }
    dlibEval(predictions, testLabels, Task::REGRESSION);
}

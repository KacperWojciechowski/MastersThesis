#pragma once

#include <dlib/matrix.h>
#include <dlib/svm.h>

#include <inc/dlib/eval.hpp>

inline void dlibLinear(std::vector<dlib::matrix<double>> data,
                       std::vector<double> labels)
{
    using namespace dlib;
    // utworzenie oraz konfiguracja trenera i jądra
    using linearKernel = linear_kernel<matrix<double>>;
    krr_trainer<matrix<double>> trainer;
    trainer.set_kernel(linearKernel());
    // podział danych
    auto dataSplit = data.begin() + data.size() * 0.8;
    auto trainData = std::vector<matrix<double>>(data.begin(), dataSplit);
    auto testData = std::vector<matrix<double>>(dataSplit, data.end());
    auto labelSplit = labels.begin() + labels.size() * 0.8;
    auto trainLabels = std::vector<matrix<double>>(labels.begin(), labelSplit);
    auto testLabels = std::vector<matrix<double>>(labelSplit, labels.end());
    // trening
    decision_function<linearKernel> model = trainer.train(trainData, trainLabels);
    // ewaluacja
    std::cout << "----- Dlib Linear -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = model(trainData);
    dlibEval(predictions, trainLabels, Task::REGRESSION);
    
    std::cout << "Test data:" << std::endl;
    predictions = model(testData);
    dlibEval(predictions, testLabels, Task::REGRESSION); 
}
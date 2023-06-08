#pragma once

#define SHARK_CV_VERBOSE 1
#include <inc/shark/printEvaluation.hpp>
#include <shark/Algorithms/KMeans.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Classifier.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Regularizer.h>

inline void sharkSVM(const shark::ClassificationDataset& trainData,
                     const shark::ClassificationDataset& testData)
{
    using namespace shark;

    // utworzenie jądra
    // auto tmp = binarySubProblem(trainData, 0, 1);
    double gamma = 0.1111;
    GaussianRbfKernel<> kernel(gamma);
    KernelClassifier<RealVector> svm;
    double regularization = 1.0;
    bool bias = true;
    // utworzenie i konfiguracja modelu
    CSvmTrainer<RealVector> trainer(
        &kernel, regularization, bias);
    trainer.sparsify() = false;
    //trainer.stoppingCondition().minAccuracy=1e-6;
    //trainer.setCacheSize(0x1000000);
    // trening
    trainer.train(svm, trainData);
    // ewaluacja
    std::cout << "-----Shark SVM-----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = svm(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions);

    std::cout << "Test data:" << std::endl;
    predictions = svm(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions);
}

#pragma once

#include <iostream>

inline void sharkLogistic(const shark::ClassificationDataset& trainData,
                          const shark::ClassificationDataset& testData)
{
    using namespace shark;

    // utworzenie modelu
    LinearClassifier<RealVector> logisticModel;
    LogisticRegression<RealVector> trainer;
    // trening
    trainer.train(logisticModel, trainData);
    // ewaluacja
    std::cout << "-----Shark Logistic Regression-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    auto predictions = logisticModel(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions, Task::CLASSIFICATION);

    std::cout << "Test data model evaluation:" << std::endl;
    predictions = logisticModel(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions, Task::CLASSIFICATION); 
}
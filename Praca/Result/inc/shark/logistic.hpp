#pragma once

#include <iostream>

#define SHARK_CV_VERBOSE 1
#include <inc/shark/printEvaluation.hpp>
#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Classifier.h>
#include <iostream>

inline void sharkLogistic(const shark::ClassificationDataset& trainData,
                          const shark::ClassificationDataset& testData)
{
    using namespace shark;

    // creating model
    std::cout << "Start logistic\n";
    LinearClassifier<RealVector> logisticModel;
    LogisticRegression<RealVector> trainer;
    // trainig
    std::cout << "Training\n";
    trainer.train(logisticModel, trainData);
    // evaluation
    std::cout << "-----Shark Logistic Regression-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    auto predictions = logisticModel(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions);

    std::cout << "Test data model evaluation:" << std::endl;
    predictions = logisticModel(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions);
}

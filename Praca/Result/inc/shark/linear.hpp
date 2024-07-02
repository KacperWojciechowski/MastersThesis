#pragma once

#include <iostream>
#include <inc/shark/printEvaluation.hpp>
#include <shark/Algorithms/Trainers/LinearRegression.h>
#include <shark/Models/LinearModel.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>

inline void sharkLinear(const shark::RegressionDataset& trainData,
                        const shark::RegressionDataset& testData)
{
    using namespace shark;

    // preparing the model and the trainer
    LinearModel<> model;
    LinearRegression trainer;
    // training
    trainer.train(model, trainData);
    // validation
    std::cout << "----- Shark Linear -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = model(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions);

    std::cout << "Test data:" << std::endl;
    predictions = model(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions);
}

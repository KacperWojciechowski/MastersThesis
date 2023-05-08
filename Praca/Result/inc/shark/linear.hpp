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

    // przygotowanie modelu i trenera
    LinearModel<> model;
    LinearRegression trainer;
    //trening
    trainer.train(model, trainData);
    // ewaluacja
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

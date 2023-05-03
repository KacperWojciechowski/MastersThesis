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

    // przygotowanie modelu
    LinearModel<> model(inputDimension(trainData), labelDimension(trainData));
    SquaredLoss<> loss;
    ErrorFunction errorFunction(trainData, &model, &loss);
    // przygotowanie i wyszkolenie optymalizatora
    CG optimizer;
    errorFunction.init();
    optimizer.init(errorFunction);
    for (int i = 0; i < 100; ++i)
    {
        optimizer.step(errorFunction);
    }
    // zastosowanie wytrenowanych parametrów modelu
    model.setParameterVector(optimizer.solution().point);
    // ewaluacja
    std::cout << "----- Shark Linear -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = model(trainData.inputs());
    printSharkModelEvaluation(
        trainData.outputs(), predictions, Task::REGRESSION);

    std::cout << "Test data:" << std::endl;
    predictions = model(testData.inputs());
    printSharkModelEvaluation(
        trainData.outputs(), predictions, Task::REGRESSION);
}
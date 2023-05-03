#pragma once

#include <iostream>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/NegativeAUC.h>
#include <shark/Statistics/Statistics.h>


enum class Task
{
    CLASSIFICATION,
    REGRESSION
};

inline void printSharkModelEvaluation(const auto& labels, 
                                      const auto& predictions,
                                      Task task)
{
    using namespace shark;

    // błąd średniokwadratowy
    SquaredLoss<> loss;
    auto mse = loss(labels, predictions);
    std::cout << "MSE: " << mse << std::endl;
    
    // metryka R^2
    auto var = variance(labels);
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;

    // wartość krzywej ROC
    if (task == Task::CLASSIFICATION)
    {
        constexpr bool invertToPositiveROC = true;
        NegativeAUC<> roc(invertToPositiveROC);
        auto auc_roc = roc(labels, predictions);
        std::cout << "AUC ROC: " << auc_roc << std::endl << std::endl;
    }
}
#pragma once

#include <iostream>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/NegativeAUC.h>
#include <shark/Statistics/Statistics.h>
#include <type_traits>
#include <cmath>

enum class Task
{
    CLASSIFICATION,
    REGRESSION
};

inline void printSharkModelEvaluation(
    const shark::Data<shark::RealVector>& labels, 
    const auto& predictions)
{
    using namespace shark;
    using namespace shark::statistics;

    // błąd średniokwadratowy
    SquaredLoss<> loss;
    auto mse = loss(labels, predictions);
    std::cout << "MSE: " << mse << std::endl;
    
    // metryka R^2
    auto var = Variance().statistics({labels.elements().begin(), labels.elements().end()});
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;
}

inline void printSharkModelEvaluation(
    const auto& labels, 
    const auto& predictions)
{
    using namespace shark;
    using namespace shark::statistics;

    // błąd średniokwadratowy
    auto squaredSum = 0.0;
    for (int i = 0; i < labels.numberOfElements(); i++)
    {
        squaredSum += std::pow(static_cast<unsigned int>(predictions.element(i).element(0)) - static_cast<unsigned int>(labels.element(i).element(0)), 2);
    }
    auto mse = std::sqrt(squaredSum / labels.numberOfElements());

    // metryka R^2
    auto var = Variance().statistics({labels.elements().begin(), labels.elements().end()});
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;
}
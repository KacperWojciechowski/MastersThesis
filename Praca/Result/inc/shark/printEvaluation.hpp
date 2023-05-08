#pragma once

#include <iostream>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/NegativeAUC.h>
#include <shark/Statistics/Statistics.h>
#include <type_traits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <shark/LinAlg/Base.h>

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
    std::vector<RealVector> tmp;
    std::for_each(labels.elements().begin(), labels.elements().end(), [&](const auto e){ tmp.emplace_back(e); });
    auto var = Variance().statistics(tmp);
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;
}

inline void printSharkModelEvaluation(
    const shark::Data<unsigned int>& labels, 
    const shark::Data<shark::RealVector>& predictions)
{
    using namespace shark;
    using namespace shark::statistics;

    // błąd średniokwadratowy
    auto squaredSum = 0.0;
    for (int i = 0; i < labels.numberOfElements(); i++)
    {
        squaredSum += std::pow(static_cast<const int>(predictions.element(i)[0]) - static_cast<const double>(labels.element(i)), 2);
    }
    auto mse = std::sqrt(squaredSum / labels.numberOfElements());
    std::cout << "MSE: " << mse << std::endl;

    // metryka R^2
    std::vector<RealVector> tmp;
    std::for_each(labels.elements().begin(), labels.elements().end(), [&](const auto e){ RealVector rv(1); rv(0) = e; tmp.emplace_back(rv);});
    auto var = Variance().statistics(tmp);
    std::cout << var(0) << std::endl;
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
        squaredSum += std::pow(static_cast<const int>(predictions.element(i)) - static_cast<const int>(labels.element(i)), 2);
    }
    auto mse = std::sqrt(squaredSum / labels.numberOfElements());
    std::cout << "MSE: " << mse << std::endl;

    // metryka R^2
    std::vector<RealVector> tmp;
    std::for_each(labels.elements().begin(), labels.elements().end(), [&](const auto e){RealVector rv(1); rv(0) = e; tmp.emplace_back(rv);});
    auto var = Variance().statistics(tmp);
    std::cout << var(0) << std::endl;
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;
}

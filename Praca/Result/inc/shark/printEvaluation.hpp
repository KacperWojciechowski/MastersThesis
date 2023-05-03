#pragma once

#include <iostream>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/NegativeAUC.h>
#include <shark/Statistics/Statistics.h>
#include <type_traits>

enum class Task
{
    CLASSIFICATION,
    REGRESSION
};

template <typename LabelType, typename PredType,
std::enable_if_t<std::is_same<shark::RealVector, typename LabelType::LabelT>::value,
    bool> = true> 
inline void printSharkModelEvaluation(
    const LabelType& labels, 
    const PredType& predictions)
{
    using namespace shark;
    using namespace shark::statistics;

    // błąd średniokwadratowy
    SquaredLoss<> loss;
    auto mse = loss(labels, predictions);
    std::cout << "MSE: " << mse << std::endl;
    
    // metryka R^2
    auto var = Variance().statistics(labels);
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;
}

template <typename LabelType, typename PredType,
std::enable_if_t<std::is_same<shark::RealVector, typename LabelType::LabelT>::value,
    bool> = false> 
inline void printSharkModelEvaluation(
    const LabelType& labels, 
    const PredType& predictions)
{
    using namespace shark;
    using namespace shark::statistics;

    // wartość krzywej ROC
    constexpr bool invertToPositiveROC = true;
    NegativeAUC<> roc(invertToPositiveROC);
    auto auc_roc = roc(labels, predictions);
    std::cout << "AUC ROC: " << auc_roc << std::endl << std::endl;
}    
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

std::vector<shark::RealVector> repackToRealVectorRange(const auto& dataContainer)
{
    using namespace shark;

    // przepakowanie danych z typu unsigned int na typ RealVector
    std::vector<RealVector> v;
    std::for_each(dataContainer.elements().begin(), dataContainer.elements().end(), [&](const auto e){RealVector rv(1); rv(0) = static_cast<double>(e); v.emplace_back(rv); });
    return v;
}

inline void printSharkModelEvaluation(
    const shark::Data<unsigned int>& labels,
    const auto& predictions)
{
    using namespace shark;

    // przygotowanie solvera pola pod wykresem ROC
    constexpr bool invert = false;
    NegativeAUC<unsigned int, RealVector> auc(invert);
 
    // przepakowanie danych
    auto predVec = repackToRealVectorRange(predictions);
    auto predData = createDataFromRange(predVec);
    // obliczenie AUC ROC
    auto roc = auc(labels, predData);
    std::cout << "ROC: " << (-1*roc) << std::endl << std::endl;
}

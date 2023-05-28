#pragma once

#include <iostream>
#include <cmath>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/evaluation/MeanAbsoluteError.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/MulticlassLabels.h>

inline void shogunVerifyModel(
    const shogun::Some<shogun::CRegressionLabels>& predictions, 
    const shogun::Some<shogun::CRegressionLabels>& targets)
{
    using namespace shogun;

    // błąd średniokwadratowy
    auto mseError = some<CMeanSquaredError>();
    auto mse = mseError->evaluate(predictions, targets);
    std::cout << "MSE = " << mse << std::endl;
    // metryka R^2
    float64_t avg = 0.0;
    float64_t sumFromErr = 0.0;
    float64_t sumFromAvg = 0.0;
    // obliczenie średniej i wariancji
    for (index_t i = 0; i < targets->get_num_labels(); i++)
    {
        avg += targets->get_label(i);
    }
    avg /= targets->get_num_labels();
    for (index_t i = 0; i < targets->get_num_labels(); i++)
    {
        sumFromAvg += std::pow(targets->get_label(i) - avg, 2);
	sumFromErr += 
		std::pow(targets->get_label(i) - predictions->get_label(i), 2);
    }
    // obliczenie metryki R^2
    auto r_square = 1 - (sumFromErr / sumFromAvg);
    std::cout << "R^2 = " << r_square << std::endl << std::endl;
}

inline void shogunVerifyModel(
    const shogun::Some<shogun::CMulticlassLabels>& predictions, 
    const shogun::Some<shogun::CMulticlassLabels>& targets)
{
    using namespace shogun;

    // obliczenie pola pod wykresem ROC
    auto roc = some<CROCEvaluation>();
    roc->evaluate(predictions->get_binary_for_class(1), 
		  targets->get_binary_for_class(1));
    std::cout << "AUC ROC = " << roc->get_auROC() << std::endl;
}

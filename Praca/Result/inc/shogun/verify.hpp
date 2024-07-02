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

    // mean squared error
    auto mseError = some<CMeanSquaredError>();
    auto mse = mseError->evaluate(predictions, targets);
    std::cout << "MSE = " << mse << std::endl;
    // R^2 metric
    float64_t avg = 0.0;
    float64_t sum = 0.0;
    // mean calculation
    for (index_t i = 0; i < targets->get_num_labels(); i++)
    {
        avg += targets->get_label(i);
    }
    avg /= targets->get_num_labels();
    // variance calculation
    for (index_t i = 0; i < targets->get_num_labels(); i++)
    {
        sum += std::pow(targets->get_label(i), 2);
    }
    float64_t variance = (sum / targets->get_num_labels()) - std::pow(avg, 2);
    // R^2 metric calculation
    auto r_square = 1 - (mse / variance);
    std::cout << "R^2 = " << r_square << std::endl << std::endl;
}

inline void shogunVerifyModel(
    const shogun::Some<shogun::CMulticlassLabels>& predictions,
    const shogun::Some<shogun::CMulticlassLabels>& targets)
{
    using namespace shogun;

    // auc roc calculation
    auto roc = some<CROCEvaluation>();
    roc->evaluate(predictions->get_binary_for_class(1),
		  targets->get_binary_for_class(1));
    std::cout << "AUC ROC = " << roc->get_auROC() << std::endl;
}

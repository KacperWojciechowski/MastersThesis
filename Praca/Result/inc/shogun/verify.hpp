#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/evaluation/MeanAbsoluteError.h>
#include <shogun/evaluation/ROCEvaluation.h>

inline void shogunVerifyModel(
    const shogun::Some<shogun::CRegressionLabels>& predictions, 
    const shogun::Some<shogun::CregressionLabels>& targets)
{
    using namespace shogun;

    // błąd średniokwadratowy
    auto mseError = some<CMeanSquaredError>();
    auto mse = mseError->evaluate(predictions, targets);
    std::cout << "MSE = " << mse << std::endl;
    // metryka R^2
    float64_t avg = 0.0;
    float64_t sum = 0.0;
    // obliczenie średniej i wariancji
    for (index_t i = 0; i < targets->get_num_labels(); i++)
    {
        avg += (targets->get_label(i));
    }
    avg /= targets->get_num_labels();
    for (index_t i = 0; i < targets->get_num_labels(); i++)
    {
        sum += std::pow(targets->get_label(i) - avg, 2);
    }
    auto variance = sum/targets->get_num_labels();
    // obliczenie metryki R^2
    auto r_square = 1.0 - mse / variance;
    std::cout << "R^2 = " << r_square << std::endl << std::endl;
}

inline void shogunVerifyModel(
    const shogun::Some<shogun::CMulticlassLabels>& predictions, 
    const shogun::Some<shogun::CmulticlassLabels>& targets)
{
    using namespace shogun;

    std::vector<float64_t> predVec;
    predVec.reserve(predictions->get_num_labels());
    for (index_t i = 0; i < predictions->get_num_labels(); ++i)
    {
        if (predictions->get_label(i).get_multiclass_confidences()[1] >= 0.7) 
	    predVec.emplace_back(1.0);
        else predVec.emplace_back(0.0);
    }
    std::vector<float64_t> targetVec;
    targetVec.reserve(targets->get_num_labels());
    for (index_t i = 0; i < targets->get_num_labels(); ++i)
    {
        if (targets->get_label(i).get_multiclass_confidences()[1] >= 0.7) 
	    targetVec.emplace_back(1.0);
        else targetVec.emplace_back(0.0);
    }
    auto predReg = some<CRegressionLabels>(predVec.begin(), predVec.end());
    auto targetReg = some<CRegressionLabels>(targetVec.begin(), targetVec.end());
    shogunVerifyModel(predReg, targetReg);

    auto roc = some<CROCEvaluation>();
    roc->evaluate(predictions->get_binary_for_class(1), 
		  targets->get_binary_for_class(1));
    std::cout << "AUC ROC = " << roc->get_auROC() << std::endl;
}

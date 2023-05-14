#pragma once

#include <iostream>
#include <cmath>

#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/evaluation/MeanAbsoluteError.h>
#include <shogun/evaluation/ROCEvaluation.h>

enum class Task
{
    CLASSIFICATION,
    REGRESSION
};

inline auto shogunVerifyModel(const auto& predictions, const auto& targets, Task task)
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
    if (task == Task::CLASSIFICATION)
    {
        auto roc = some<CROCEvaluation>();
        roc->evaluate(predictions, targets);
        std::cout << "AUC ROC = " << roc->get_auROC() << std::endl;
    }
}

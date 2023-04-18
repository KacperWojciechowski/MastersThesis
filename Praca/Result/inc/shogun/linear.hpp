#pragma once

#include <iostream>

#include <inc/shogun/verify.hpp>

#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/regression/LinearRidgeRegression.h>

inline void shogunLinear(shogun::Some<CDenseFeatures<float64_t>>& inputs,
                         shogun::Some<CRegressionLabels>& outputs)
{
    using namespace shogun;

    // utworzenie modelu
    float64_t tauRegularization = 0.0001;
    auto linear =
        some<CLinearRidgeRegression>(tauRegularization, nullptr, nullptr);
    // podział danych na testowe i uczące
    auto testSamples = static_cast<int>(0.8*inputs.num_cols());
    auto trainInputs = inputs.submatrix(0, testSamples).clone();
    auto trainOutputs = outputs.submatrix(0, testSamples).clone();
    auto testInputs =
        inputs.submatrix(testSamples, inputs.num_cols()).clone();
    auto testOutputs =
        outputs.submatrix(testSamples, outputs.num_cols()).clone();
    // nauczanie
    linear->set_labels(trainOutputs);
    linear->train(trainInputs);
    // weryfikacja modelu
    std::cout << "----- Shogun Linear -----" << std::endl;
    std::cout << "Train data: " << std::endl;
    auto predictions = wrap(linear->apply_regression(trainInputs));
    shogunVerifyModel(predictions, trainOutputs, Task::REGRESSION);

    std::cout << "Test data: " << std::endl;
    predictions = wrap(linear->apply_regression(testInputs));
    shogunVerifyModel(predictions, testOutputs, Task::REGRESSION);
}
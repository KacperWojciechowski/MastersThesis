#pragma once

#include <iostream>
#include <numeric>
#include <inc/shogun/verify.hpp>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <vector>

inline void shogunLinear(
    shogun::Some<shogun::CDenseFeatures<float64_t>>& trainInputs,
    shogun::Some<shogun::CDenseFeatures<float64_t>>& testInputs,
    shogun::Some<shogun::CRegressionLabels>& trainOutputs,
    shogun::Some<shogun::CRegressionLabels>& testOutputs)
{
    using namespace shogun;

    // creating the model
    float64_t tauRegularization = 0.0001;
    auto linear =
        some<CLinearRidgeRegression>(tauRegularization, nullptr, nullptr);
    // training
    linear->set_labels(trainOutputs);
    linear->train(trainInputs);
    // validation
    std::cout << "----- Shogun Linear -----" << std::endl;
    std::cout << "Train data: " << std::endl;
    auto predictions = wrap(linear->apply_regression(trainInputs));
    shogunVerifyModel(predictions, trainOutputs);

    std::cout << "Test data: " << std::endl;
    auto predictions2 = wrap(linear->apply_regression(testInputs));
    shogunVerifyModel(predictions2, testOutputs);
}

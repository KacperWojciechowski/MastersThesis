#pragma once

#include <iostream>

#include <inc/shogun/verify.hpp>

#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/regression/LinearRidgeRegression.h>

inline void shogunLinear(
    shogun::Some<shogun::CDenseFeatures<float64_t>>& inputs,
    shogun::Some<shogun::CRegressionLabels>& outputs)
{
    using namespace shogun;

    // utworzenie modelu
    float64_t tauRegularization = 0.0001;
    auto linear =
        some<CLinearRidgeRegression>(tauRegularization, nullptr, nullptr);
    // podział danych na testowe i uczące
    auto testSamples = static_cast<int>(0.8*inputs.get_num_vectors());
    std::vector<index_t> testIndeces(testSamples);
    std::iota(testIndeces.begin(), testIndeces.end(), 1);
    std::for_each(testIndeces.begin(), testIndeces.end(), [](auto& i) {return --i;});
    auto trainInputs = inputs.copy_subset(testIndeces);
    auto trainOutputs = outputs.copy_subset(testIndeces);
    
    std::vector<index_t> trainIndeces(inputs.get_num_vectors() - testSamples);
    std::for_each(trainIndeces.begin(), trainIndeces.end(), [testSamples](auto& i) 
        {
            static int idx = testSamples;
            return idx++;
        });
    auto testInputs =
        inputs.copy_subset(trainIndeces);
    auto testOutputs =
        outputs.copy_subset(trainIndeces);
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
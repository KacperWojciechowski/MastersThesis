#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/MulticlassLogisticRegression.h>


inline void shogunLogistic(
    shogun::Some<shogun::CDenseFeatures<float64_t>>& inputs,
    shogun::Some<shogun::CBinaryLabels>& outputs)
{
    using namespace shogun;

    // podział danych na testowe i uczące
    auto testSamples = static_cast<int>(0.8*inputs->num_cols());
    auto trainInputs = inputs->submatrix(0, testSamples).clone();
    auto trainOutputs = outputs->submatrix(0, testSamples).clone();
    auto testInputs =
        inputs.submatrix(testSamples, inputs->num_cols()).clone();
    auto testOutputs =
        outputs.submatrix(testSamples, outputs->num_cols()).clone();

    // utworzenie modelu
    auto logReg = some<CMulticlassLogisticRegression>();

    // nauka modelu
    logReg->set_labels(trainOutputs);
    logReg->train(trainInputs);

    // ewaluacja modelu
    std::cout << "----- Shogun Logistic -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(logReg->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs, Task::CLASSIFICATION);

    std::cout << "Test:" << std::endl;
    prediction = wrap(logReg->apply_multiclass(testInputs));
    shogunVerifyModel(prediction, testOutputs, Task::CLASSIFICATION);
}
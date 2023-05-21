#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/MulticlassLogisticRegression.h>


inline void shogunLogistic(
    shogun::Some<shogun::CDenseFeatures<float64_t>>& trainInputs,
    shogun::Some<shogun::CDenseFeatures<float64_t>>& testInputs,
    shogun::Some<shogun::CMulticlassLabels>& trainOutputs,
    shogun::Some<shogun::CMulticlassLabels>& testOutputs)
{
    using namespace shogun;

    // utworzenie modelu
    auto logReg = some<CMulticlassLogisticRegression>();

    // nauka modelu
    logReg->set_labels(trainOutputs);
    logReg->train(trainInputs);

    // ewaluacja modelu
    std::cout << "----- Shogun Logistic -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(logReg->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs);

    std::cout << "Test:" << std::endl;
    auto prediction2 = wrap(logReg->apply_multiclass(testInputs));
    shogunVerifyModel(prediction2, testOutputs);
}

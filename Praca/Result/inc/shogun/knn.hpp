#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>

inline void shogunKNN(shogun::Some<CDenseFeatures>& inputs,
                      shogun::Some<CMulticlassLabels>& outputs)
{
    using namespace shogun;

    // podział danych na testowe i uczące
    auto testSamples = static_cast<int>(0.8*inputs.num_cols());
    auto trainInputs = inputs.submatrix(0, testSamples).clone();
    auto trainOutputs = outputs.submatrix(0, testSamples).clone();
    auto testInputs =
        inputs.submatrix(testSamples, inputs.num_cols()).clone();
    auto testOutputs =
        outputs.submatrix(testSamples, outputs.num_cols()).clone();
    // przygotowanie dystansu
    auto distance = some<CEuclideanDistance>(trainInputs, trainInputs);
    // przygotowanie modelu
    std::int32_t k = 3;
    auto knn = some<CKNN>(k, distance, trainOutputs);
    // ewaluacja modelu
    std::cout << "----- Shogun KNN -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(knn->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs, Task::CLASSIFICATION);

    std::cout << "Test:" << std::endl;
    prediction = wrap(knn->apply_multiclass(testInputs));
    shogunVerifyModel(prediction, testOutputs, Task::CLASSIFICATION);
}
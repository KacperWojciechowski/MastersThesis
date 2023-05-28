#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>

inline void shogunKNN(
		shogun::Some<shogun::CDenseFeatures>& trainInputs,
                shogun::Some<shogun::CDenseFeatures>& testInputs,
                shogun::Some<shogun::CMulticlassLabels>& trainOutputs,
                shogun::Some<shogun::CMulticlassLabels>& testOutputs)
{
    using namespace shogun;

    // przygotowanie dystansu
    auto distance = some<CEuclideanDistance>(trainInputs, trainInputs);
    // przygotowanie modelu
    std::int32_t k = 3;
    auto knn = some<CKNN>(k, distance, trainOutputs);
    // ewaluacja modelu
    std::cout << "----- Shogun KNN -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(knn->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs);

    std::cout << "Test:" << std::endl;
    auto prediction2 = wrap(knn->apply_multiclass(testInputs));
    shogunVerifyModel(prediction2, testOutputs);
}

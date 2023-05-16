#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>

inline void shogunSVM(shogun::Some<shogun::CDenseFeatures<float64_t>>& trainInputs,
                      shogun::Some<shogun::CDenseFeatures<float64_t>>& testInputs,
                      shogun::Some<shogun::CMulticlassLabels>& trainOutputs,
                      shogun::Some<shogun::CMulticlassLabels>& testOutputs)
{
    using namespace shogun;

    // utworzenie jądra
    auto kernel = some<CGaussianKernel>(trainInputs, trainInputs, 5);
    // utworzenie i konfiguracja modelu
    auto svm = some<CMulticlassLibSVM>();
    svm->set_kernel(kernel);

    // trening
    svm->set_labels(trainOutputs);
    svm->train(trainInputs);

    // ewaluacja modelu
    std::cout << "----- Shogun SVM -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(svm->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs);

    std::cout << "Test:" << std::endl;
    auto prediction2 = wrap(svm->apply_multiclass(testInputs));
    shogunVerifyModel(prediction2, testOutputs);
}

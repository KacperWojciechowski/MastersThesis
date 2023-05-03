#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>

inline void shogunSVM(shogun::Some<shogun::CDenseFeatures>& trainInputs,
                      shogun::Some<shogun::CDenseFeatures>& testInputs,
                      shogun::Some<shogun::CBinaryLabels>& trainOutputs,
                      shogun::Some<shogun::CBinaryLabels>& testOutputs)
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
    auto prediction = wrap(svm->apply_binary(trainInputs));
    shogunVerifyModel(prediction, trainOutputs, Task::CLASSIFICATION);

    std::cout << "Test:" << std::endl;
    prediction = wrap(svm->apply_binary(testInputs));
    shogunVerifyModel(prediction, testOutputs, Task::CLASSIFICATION);
}
#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>

inline void shogunSVM(shogun::Some<shogun::CDenseFeatures>& inputs,
                      shogun::Some<shogun::CBinaryLabels>& outputs)
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
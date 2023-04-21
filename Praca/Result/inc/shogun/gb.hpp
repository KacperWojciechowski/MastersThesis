#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/multiclass/tree/CARTree.h>

inline void shogunGradientBoost(
    shogun::Some<shogun::CDenseFeatures>& inputs,
    shogun::Some<shogun::CMulticlassLabels>& outputs)
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
    // oznaczenie regresorów jako ciągłe
    SGVector<bool> featureType(1);
    featureType.set_const(false);
    // utworzenie binarnego drzewa decyzyjnego
    auto tree = some<CCARTree>(featureType, PT_REGRESSION);
    tree->set_max_depth(3);
    // utworzenie funkcji straty
    auto loss = some<CSquaredLoss>();
    // utworzenie i konfiguracja modelu
    constexpr int iterations = 100;
    constexpr int learningRate = 0.1;
    constexpr int subsetFraction = 1.0;
    auto model = some<CStochasticGBMachine>(tree,
                                            loss,
                                            iterations,
                                            learningRate,
                                            subsetFraction);
    // trening
    model->set_labels(trainOutputs);
    model->train(trainInputs);
    // ewaluacja modelu
    std::cout << "----- Shogun Gradient Boost -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(model->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs, Task::CLASSIFICATION);

    std::cout << "Test:" << std::endl;
    prediction = wrap(model->apply_multiclass(testInputs));
    shogunVerifyModel(prediction, testOutputs, Task::CLASSIFICATION);
}
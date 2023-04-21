#pragma once

#include "inc/shogun/verify.hpp"

#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/machine/RandomForest.h>

inline void shogunRandomForest(
    shogun::Some<shogun::CDenseFeatures<DataType>> inputs,
    shogun::Some<shogun::CRegressionLabels> outputs)
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
    // utworzenie i konfiguracja modelu
    constexpr std::int32_t numRandFeats = 1;
    constexpr std::int32_t numBags = 10;
    auto randForest = 
        some<CRandomForest>(numRandFeats, numBags);
    auto vote = some<CMajorityVote>();
    randForest->set_combination_rule(vote);
    // oznaczenie danych jako ciągłe
    SGVector<bool> featureType(1);
    featureType.set_const(false);
    randForest->set_feature_type(featureType);
    // trening
    randForest->set_labels(trainOutputs);
    randForest->set_machine_problem_type(PT_REGRESSION);
    randForest->train(trainInputs);
    // ewaluacja modelu
    std::cout << "----- Shogun Random Forest -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = wrap(randForest->apply_regression(trainInputs));
    shogunVerifyModel(predictions, trainOutputs, Task::REGRESSION);

    std::cout << "Test data:" << std::endl;
    predictions = wrap(randForest->apply_regression(testInputs));
    shogunVerifyModel(predictions, testOutputs, Task::REGRESSION);
}
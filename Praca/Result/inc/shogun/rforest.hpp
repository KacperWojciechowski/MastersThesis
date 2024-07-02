#pragma once

#include "inc/shogun/verify.hpp"

#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/machine/RandomForest.h>

inline void shogunRandomForest(
    shogun::Some<shogun::CDenseFeatures<DataType>> trainInputs,
    shogun::Some<shogun::CDenseFeatures<DataType>> testInputs,
    shogun::Some<shogun::CRegressionLabels> trainOutputs,
    shogun::Some<shogun::CRegressionLabels> testOutputs)
{
    using namespace shogun;

    // creating and configuring the model
    constexpr std::int32_t numRandFeats = 1;
    constexpr std::int32_t numBags = 10;
    auto randForest = 
        some<CRandomForest>(numRandFeats, numBags);
    auto vote = some<CMajorityVote>();
    randForest->set_combination_rule(vote);
    // marking data as continuous
    SGVector<bool> featureType(1);
    featureType.set_const(false);
    randForest->set_feature_type(featureType);
    // training
    randForest->set_labels(trainOutputs);
    randForest->set_machine_problem_type(PT_REGRESSION);
    randForest->train(trainInputs);
    // evaluation
    std::cout << "----- Shogun Random Forest -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = wrap(randForest->apply_regression(trainInputs));
    shogunVerifyModel(predictions, trainOutputs);

    std::cout << "Test data:" << std::endl;
    auto predictions2 = wrap(randForest->apply_regression(testInputs));
    shogunVerifyModel(predictions2, testOutputs);
}

#pragma once

#include "inc/shogun/verify.hpp"

#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/neuralnets/NeuralLayers.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/base/some.h>

inline void sharkNeural(
    shogun::Some<shogun::CDenseFeatures<float64_t>> inputs,
    shogun::Some<shogun::CBinaryLabels> outputs)
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
    // konstrukcja architektury sieci
    auto dimensions = trainInputs.get_num_features();
    auto layers = some<CNeuralLayers>();
    layers = wrap(layers->input(dimensions));
    layers = wrap(layers->rectified_linear(5));
    layers = wrap(layers->rectified_linear(5));
    layers = wrap(layers->logistic(1));
    auto allLayers = layers->done();
    // utworzenie sieci
    auto network = some<CNeuralNetwork>(allLayers);
    network->quick_connect();
    network->initialize_neural_network();
    // konfiguracja sieci
    network->set_pptimization_method(NNOM_GRADIENT_DESCENT);
    network->set_gd_mini_batch_size(64);
    network->set_l2_coefficient(0.0001);
    network->set_max_num_epochs(500);
    network->set_epsilon(0.0);
    network->set_gd_learning_rate(0.01);
    network->set_gd_momentum(0.5);
    // trening
    network->set_labels(trainOutputs);
    network->train(trainInputs);
    // walidacja
    std::cout << "----- Shogun Neural Network -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = network->apply_binary(trainInputs);
    shogunVerifyModel(predictions, trainOutputs, Task::CLASSIFICATION);

    std::cout << "Test data:" << std::endl;
    predictions = network->apply_binary(testInputs);
    shogunVerifyModel(predictions, testOutputs, Task::CLASSIFICATION);
}
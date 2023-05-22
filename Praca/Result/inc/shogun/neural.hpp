#pragma once

#include "inc/shogun/verify.hpp"

#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/neuralnets/NeuralLayers.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/base/some.h>

inline void sharkNeural(
    shogun::Some<shogun::CDenseFeatures<float64_t>> trainInputs,
    shogun::Some<shogun::CDenseFeatures<float64_t>> testInputs,
    shogun::Some<shogun::CMulticlassLabels> trainOutputs,
    shogun::Some<shogun::CMulticlassLabels> testOutputs)
{
    using namespace shogun;   

    // konstrukcja architektury sieci
    auto dimensions = trainInputs->get_num_features();
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
    network->set_optimization_method(NNOM_GRADIENT_DESCENT);
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
    auto predictions = network->apply_multiclass(trainInputs);
    shogunVerifyModel(predictions, trainOutputs);

    std::cout << "Test data:" << std::endl;
    auto predictions2 = network->apply_multiclass(testInputs);
    shogunVerifyModel(predictions2, testOutputs);
}

#pragma once

#include <dlib/dnn.h>
#include <dlib/matrix.h>

#include <inc/dlib/eval.hpp>

#include <vector>

inline void dlibNeural(
    std::vector<dlib::matrix<double>> trainData,
    std::vector<dlib::matrix<double>> testData,
    std::vector<double> trainLabels,
    std::vector<double> testLabels)
{
    using namespace dlib;
    // defining the network architecture
    using Architecture = loss_mean_squared<fc <1, 
                            htan<fc<5, 
                            htan<fc<5, 
                            input<matrix<double>>>>>>>>;
    // creating the network
    Architecture model;
    // creagting and configuring the optimizing algorithm
    float weightDecay = 0.0001f;
    float momentum = 0.5f;
    sgd solver(weightDecay, momentum);
    // creating and configuring the trainer
    dnn_trainer<Architecture> trainer(model, solver);
    trainer.set_learning_rate(0.1);
    trainer.set_learning_rate_shrink_factor(1);
    trainer.set_mini_batch_size(64);
    trainer.set_max_num_epochs(100);
    trainer.be_verbose();
    // training
    trainer.train(trainData, trainLabels);
    model.clean();
    // evaluation
    std::cout << "----- Dlib Neural -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = model(trainData);
    dlibEval(predictions, trainLabels, Task::CLASSIFICATION);
    
    std::cout << "Test data:" << std::endl;
    predictions = model(testData);
    dlibEval(predictions, testLabels, Task::CLASSIFICATION);
}
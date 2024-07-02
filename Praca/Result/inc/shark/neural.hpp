#pragma once

#include <iostream>
#include <inc/shark/printEvaluation.hpp>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/LinearModel.h>
#include <shark/Data/Dataset.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Regularizer.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>

inline void sharkNN(const shark::ClassificationDataset& trainData,
                    const shark::ClassificationDataset& testData)
{
    using namespace shark;
    
    // defining network layers
    using DenseTanhLayer = LinearModel<RealVector, TanhNeuron>;
    using DenseLogisticLayer = LinearModel<RealVector, LogisticNeuron>;
    DenseTanhLayer layer1(inputDimension(trainData), 5, true);
    DenseTanhLayer layer2(5, 5, true);
    DenseLogisticLayer output(5, numberOfClasses(trainData), true);
    // connecting the layers
    auto network = layer1 >> layer2 >> output;
    // creating and configuring the loss function
    CrossEntropy<unsigned int, RealVector> loss;
    ErrorFunction<> error(trainData, &network, &loss, true);
    
    // regularization
    TwoNormRegularizer<> regularizer(error.numberOfVariables());
    double weightDecay = 0.01;
    error.setRegularizer(weightDecay, &regularizer);
    error.init();
    
    // weights initialization
    initRandomNormal(network, 0.001);
    // creating and configuring the optimizer
    SteepestDescent<> optimizer;
    optimizer.setMomentum(0.5);
    optimizer.setLearningRate(0.1);
    optimizer.init(error);
    // training
    std::size_t epochs = 1000;
    std::size_t iterations = trainData.numberOfBatches();
    // loop that goes through epochs
    for (std::size_t epoch = 0; epoch != epochs; ++epoch)
    {
        double avgLoss = 0.0;
        // loop that goes through batches
        for (std::size_t i = 0; i != iterations; ++i)
        {
            // performing optimizer step
            optimizer.step(error);
            // saving partial loss
            if (i % 100 == 0)
            {
                avgLoss += optimizer.solution().value;
            }
        }
        // calculating average loss value
        avgLoss /= iterations;
        std::cout << "Epoch " << epoch << " | Avg. loss " << avgLoss 
                  << std::endl;
    }
    // configuring the final model
    network.setParameterVector(optimizer.solution().point);
    std::cout << "In Shape=" << network.inputShape() << std::endl;
    std::cout << "Out Shape=" << network.outputShape() << std::endl;
    Classifier<ConcatenatedModel<RealVector>> model(network);

    // validation
    std::cout << "-----Shark Neural -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    std::cout << "trainData.numberOfBatches() = " << 
	    trainData.numberOfBatches() << std::endl;
    auto predictions = network(trainData.inputs());
    std::cout << "predictions.numberOfBatches() = " << 
	    predictions.numberOfBatches();
    printSharkModelEvaluation(
        trainData.labels(), predictions);

    std::cout << "Test data:" << std::endl;
    predictions = network(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions); 
}

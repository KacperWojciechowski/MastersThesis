#pragma once

#include <iostream>
#include <inc/shark/printEvaluation.hpp>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/Models/LinearModel.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/Regularizer.h>
#include <shark/ObjectiveFunctions/Loss/DiscreteLoss.h>

inline void sharkNN(const shark::ClassificationDataset& trainData,
                    const shark::ClassificationDataset& testData)
{
    using namespace shark;
    
    // zdefiniowanie warstw sieci
    using DenseTanhLayer = LinearModel<RealVector, TanhNeuron>;
    using DenseLogisticLayer = LinearModel<RealVector, LogisticNeuron>;
    DenseTanhLayer layer1(inputDimension(trainData), 5, true);
    DenseTanhLayer layer2(5, 5, true);
    DenseLogisticLayer output(5, 1, true);
    // połączenie warstw
    auto network = layer1 >> layer2 >> output;
    // utworzenie i konfiguracja funkcji straty
    SquaredLoss<> loss;
    ErrorFunction<> error(trainData, &network, &loss, true);
    TwoNormRegularizer<> regularizer(error.numberOfVariables());
    double weightDecay = 0.0001;
    error.setRegularizer(weightDecay, &regularizer);
    error.init();
    // inicjalizacja wag sieci
    initRandomNormal(network, 0.001);
    // utworzenie i konfiguracja optymalizatora
    SteepestDescent<> optimizer;
    optimizer.setMomentum(0.5);
    optimizer.setLearningRate(0.1);
    optimizer.init(error);
    // przeprowadzenie procesu uczenia
    std::size_t epochs = 1000;
    std::size_t iterations = trainData.numberOfBatches();
    // pętla przechodząca poszczególne epoki
    for (std::size_t epoch = 0; epoch != epochs; ++epoch)
    {
        double avgLoss = 0.0;
        // pętla operująca na pojedynczych batch'ach
        for (std::size_t i = 0; i != iterations; ++i)
        {
            // wykonanie kroku optymalizatora
            optimizer.step(error);
            // zapisanie częściowej wartości funkcji straty
            if (i % 100 == 0)
            {
                avgLoss += optimizer.solution().value;
            }
        }
        // wyliczenie średniej wartości funkcji straty
        avgLoss /= iterations;
        std::cout << "Epoch " << epoch << " | Avg. loss " << avgLoss 
                  << std::endl;
    }
    // konfiguracja modelu docelowego
    network.setParameterVector(optimizer.solution().point);
    // walidacja
    std::cout << "-----Shark Neural -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = network(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions);

    std::cout << "Test data:" << std::endl;
    predictions = network(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions); 
}
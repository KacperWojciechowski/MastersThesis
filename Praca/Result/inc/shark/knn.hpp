#pragma once

#include <iostream>

inline void sharkKNN(const shark::ClassificationDataset& trainData,
                     const shark::ClassificationDataset& testData)
{
    using namespace shark;

    // utworzenie i konfiguracja drzewa oraz algorytmu
    KDTree<RealVector> tree(trainData.inputs());
    TreeNearestNeighbors<RealVector,unsigned int> algorithm(
        trainData, &tree);

    // konfiguracja modelu
    const unsigned int K = 2; // ilość sąsiadów dla algorytmu kNN
    NearestNeighborModel<RealVector, unsigned int> KNN(&algorithm, K);

    // ewaluacja modelu
    std::cout << "-----Shark KNN-----" << std::endl;
    std::cout << "Train data :" << std::endl;
    auto predictions = KNN(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions, Task::CLASSIFICATION);

    std::cout << "Test data:" << std::endl;
    predictions = KNN(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions, Task::CLASSIFICATION);
}
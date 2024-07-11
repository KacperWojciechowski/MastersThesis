#pragma once

#define SHARK_CV_VERBOSE 1
#include <inc/shark/printEvaluation.hpp>
#include <shark/Algorithms/KMeans.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Classifier.h>
#include <shark/Models/NearestNeighborModel.h>
#include <shark/Models/Trees/KDTree.h>
#include <iostream>

inline void sharkKNN(const shark::ClassificationDataset& trainData,
                     const shark::ClassificationDataset& testData)
{
    using namespace shark;

    // creating and configuring the tree and algorithm
    KDTree<RealVector> tree(trainData.inputs());
    TreeNearestNeighbors<RealVector, unsigned int> algorithm(
        trainData, &tree);

    // model configuration
    const unsigned int K = 2; // neighbor count for kNN algorithm
    NearestNeighborModel<RealVector, unsigned int> KNN(&algorithm, K);

    // evaluation
    std::cout << "-----Shark KNN-----" << std::endl;
    std::cout << "Train data :" << std::endl;
    auto predictions = KNN(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions);

    std::cout << "Test data:" << std::endl;
    predictions = KNN(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions);
}
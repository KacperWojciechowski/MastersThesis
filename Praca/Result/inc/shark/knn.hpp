#pragma once

inline void sharkKNN(const shark::ClassificationDataset& trainData,
              const shark::ClassificationDataset& testData)
{
    // utworzenie i konfiguracja drzewa oraz algorytmu
    KDTree<RealVector> tree(trainData.inputs());
    TreeNearestNeighbors<RealVector,unsigned int> algorithm(trainData, &tree);

    // konfiguracja modelu
    const unsigned int K = 2; // ilość sąsiadów dla algorytmu kNN
    NearestNeighborModel<RealVector, unsigned int> KNN(&algorithm, K);

    // ewaluacja modelu
    auto trainPredictions = KNN(trainData.inputs());
    auto testPredictions = KNN(testData.inputs());

    std::cout << "-----Shark KNN-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    printSharkModelEvaluation(trainData.labels(). trainPredictions);

    std::cout << "Test data model evaluation:" << std::endl;
    printSharkModelEvaluation(testData.labels(), testPredictions);
}
#pragma once

inline void sharkLogisticRegression(const shark::ClassificationDataset& trainData,
                             const shark::ClassificationDataset& testData)
{
    // utworzenie modelu
    shark::LinearClassifier<RealVector> logisticModel;
    shark::LogisticRegression<RealVector> trainer;
    
    // wytrenowanie
    trainer.train(logisticModel, trainData);

    // ewaluacja
    auto trainPredictions = logisticModel(trainData.inputs());
    auto testPredictions = logisticModel(testData.inputs());

    std::cout << "-----Shark Logistic Regression-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    printSharkModelEvaluation(trainData.labels(), trainPredictions);

    std::cout << "Test data model evaluation:" << std::endl;
    printSharkModelEvaluation(testData.labels(), testPredictions); 
}
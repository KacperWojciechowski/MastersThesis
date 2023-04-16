#pragma once

inline void sharkSVM(const shark::ClassificationDataset& trainData,
              const shark::ClassificationDataset& testData)
{
    // utworzenie jądra
    double gamma = 0.5;
    shark::GaussianRbfKernel<> kernel(gamma);
    KernelClassifier<RealVector> svm;
    double regularization = 1000.0;
    bool bias = true;
    
    // utworzenie i konfiguracja modelu
    CSvmTrainer<RealVector, double> trainer(&kernel, regularization, bias);
    trainer.sparsify() = false;
    trainer.stopCondition().minAccuracy=1e-6;
    trainer.setCacheSize(0x1000000);

    // trening
    trainer.train(svm, trainData);

    // ewaluacja
    auto trainPredictions = svm(trainData.inputs());
    auto testPredictions = svm(testData.inputs());

    std::cout << "-----Shark SVM-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    printSharkModelEvaluation(trainData.labels(). trainPredictions);

    std::cout << "Test data model evaluation:" << std::endl;
    printSharkModelEvaluation(testData.labels(), testPredictions);
}
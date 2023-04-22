#pragma once

inline void sharkSVM(const shark::ClassificationDataset& trainData,
                     const shark::ClassificationDataset& testData)
{
    using namespace shark;

    // utworzenie jądra
    double gamma = 0.5;
    GaussianRbfKernel<> kernel(gamma);
    KernelClassifier<RealVector> svm;
    double regularization = 1000.0;
    bool bias = true;
    // utworzenie i konfiguracja modelu
    CSvmTrainer<RealVector, double> trainer(
        &kernel, regularization, bias);
    trainer.sparsify() = false;
    trainer.stopCondition().minAccuracy=1e-6;
    trainer.setCacheSize(0x1000000);
    // trening
    trainer.train(svm, trainData);
    // ewaluacja
    std::cout << "-----Shark SVM-----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = svm(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions, Task::CLASSIFICATION);

    std::cout << "Test data:" << std::endl;
    predictions = svm(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions, Task::CLASSIFICATION);
}
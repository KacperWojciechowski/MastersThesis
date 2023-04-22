#pragma once

inline void sharkLinear(const shark::RegressionDataset& trainData,
                        const shark::RegressionDataset& testData)
{
    using namespace shark;

    // utworzenie modelu
    LinearRegression trainer;
    LinearModel<> model;
    // trening modelu
    trainer.train(model, trainData);
    // odczytanie parametrów modelu
    std::cout << "intercept: " << model.offset() << std::endl;
    std::cout << "matrix: " << model.matrix() << std::endl;
    // ewaluacja
    std::cout << "----- Shark Linear -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = model(trainData.inputs());
    printSharkModelEvaluation(
        trainData.outputs(), predictions, Task::REGRESSION);

    std::cout << "Test data:" << std::endl;
    predictions = model(testData.inputs());
    printSharkModelEvaluation(
        trainData.outputs(), predictions, Task::REGRESSION);
}

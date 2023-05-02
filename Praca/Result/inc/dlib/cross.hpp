#pragma once
#include <dlib/global_optimization.h>
#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <cmath>

inline void dlibCrossValidate(std::vector<dlib::matrix<double>> data,
                              std::vector<double> labels)
{
    using namespace dlib;
    // podział danych
    auto dataSplit = data.begin() + data.size() * 0.8;
    auto trainData = std::vector<matrix<double>>(
        data.begin(), dataSplit);
    auto testData = std::vector<matrix<double>>(
        dataSplit, data.end());
    auto labelSplit = labels.begin() + labels.size() * 0.8;
    auto trainLabels = std::vector<matrix<double>>(
        labels.begin(), labelSplit);
    auto testLabels = std::vector<matrix<double>>(
        labelSplit, labels.end());   
    // utworzenie funkcji sprawdzianu krzyżowego
    auto crossValidationScore = [&](const double gamma, const double c,
                                    const double degreeIn)
    {
        using namespace dlib;

        auto degree = std::floor(degreeIn);
        // zdefiniowanie jądra
        using Kernel = polynomial_kernel<double>;
        // przygotowanie i konfiguracja trenera
        svr_trainer<Kernel> trainer;
        trainer.set_kernel(Kernel(gamma, c, degree));
        // obliczenie metryk sprawdzianu krzyżowego
        matrix<double> result = cross_validate_regression_trainer(
            trainer, trainData, trainLabels, 10);
        // zwrócenie metryki błędu średniokwadratowego
        return result(0, 0);
    }
    // przeprowadzenie sprawdzianu
    auto result = find_min_global(
        crossValidationScore,
        {0.01, 1e-8, 5}, // wartości minimalne
        {0.1, 1, 15},    // wartości maksymalne
        max_function_calls(50));
    // odczytanie ustalonych wartości hiperparametrów
    auto gamma = result.x(0);
    auto c = result.x(1);
    auto degree = result.x(2);
    // utworzenie modelu w oparciu o ustalone hiperparametry
    using Kernel = polynomial_kernel<double>;
    svr_trainer<Kernel> trainer;
    trainer.set_kernel(Kernel(gamma, c, degree));
    auto model = trainer.train(trainData, trainLabels);
    // ewaluacja
    std::cout << "----- Dlib CrossValidated SVM -----" << std::endl;
    std::cout << "Train data:" << std::endl;
    auto predictions = std::vector<double>(trainData.size());
    for (auto& sample : trainData)
    {
        predictions.emplace_back(model(sample));
    }
    dlibEval(predictions, trainLabels, Task::REGRESSION);
    predictions.clear();
    std::cout << "Test data:" << std::endl;
    for (auto& sample : testData)
    {
        predictions.emplace_back(model(sample));
    }
    dlibEval(predictions, testLabels, Task::REGRESSION);
}
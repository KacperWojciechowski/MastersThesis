#pragma once

#include <inc/shark/csv.hpp>
#include <inc/shark/knn.hpp>
#include <inc/shark/logistic.hpp>
#include <inc/shark/neural.hpp>
#include <inc/shark/printEvaluation.hpp>
#include <inc/shark/svm.hpp>
#include <inc/shark/linear.hpp>

inline void sharkModels()
{
    using namespace shark;

    // odczytanie i podział danych
    auto classificationTrainData = sharkReadCsvData<ClassificationDataset>(
        "wdbc_data_with_labels.csv");
    auto classificationTestData = splitAtElement(
        classificationTrainData,
        static_cast<std::size_t>(0.8*classificationTrainData.numberOfElements()));
    auto regressionTrainData = sharkReadCsvData<RegressionDataset>(
        "IronGlutathione.csv");
    auto regressionTestData = splitAtElement(
        regressionTrainData,
        static_cast<std::size_t>(0.8*regressionTrainData.numberOfElements()));

    // wywołanie modeli
    sharkLinear(regressionTrainData, regressionTestData);
    sharkLogistic(classificationTrainData, classificationTestData);
    sharkSVM(classificationTrainData, classificationTestData);
    sharkKNN(classificationTrainData, classificationTestData);
}
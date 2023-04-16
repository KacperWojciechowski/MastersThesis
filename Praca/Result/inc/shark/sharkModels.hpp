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
    auto classificationTrainData = sharkReadCsvData("wdbc_data_with_labels.csv")
    auto classificationTestData = shark::splitAtElement(
        classificationTrainData,
        static_cast<std::size_t>(0.8*classificationTrainData.numberOfElements()));

    // wywołanie modeli
    sharkLogisticRegression(classificationTrainData, classificationTestData);
    sharkSVM(classificationTrainData, classificationTestData);
    sharkKNN(classificationTrainData, classificationTestData);
}
#pragma once

#include <iostream>
#include <inc/shark/csv.hpp>
#include <inc/shark/printEvaluation.hpp>
#include <inc/shark/svm.hpp>
#include <inc/shark/linear.hpp>

inline void sharkModels()
{
    using namespace shark;

    // readingg and splitting data
    auto classificationTrainData = sharkReadCsvData<ClassificationDataset>(
        "wdbc_data_with_labels_tn.csv");
    auto classificationTestData = splitAtElement(
        classificationTrainData,
        static_cast<std::size_t>(0.8*classificationTrainData.numberOfElements()));
    auto regressionTrainData = sharkReadCsvData<RegressionDataset>(
        "IronGlutathione.csv");
    auto regressionTestData = splitAtElement(
        regressionTrainData,
        static_cast<std::size_t>(0.8*regressionTrainData.numberOfElements()));

    // calling the models
    sharkLinear(regressionTrainData, regressionTestData);
    sharkSVM(classificationTrainData, classificationTestData);
}

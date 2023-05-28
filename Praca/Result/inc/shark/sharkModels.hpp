#pragma once

#include <iostream>
#include <inc/shark/csv.hpp>
//#include <inc/shark/logistic.hpp>
//#include <inc/shark/neural.hpp>
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
    //sharkLogistic(classificationTrainData, classificationTestData);
    std::cout << "Test1" << std::endl;
    sharkSVM(classificationTrainData, classificationTestData);
    std::cout << "Test2" << std::endl;
    //sharkNN(classificationTrainData, classificationTestData);
}

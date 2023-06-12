#pragma once

#include <inc/shogun/csv.hpp>
#include <inc/shogun/linear.hpp>
#include <inc/shogun/logistic.hpp>
#include <inc/shogun/svm.hpp>
#include <inc/shogun/neural.hpp>

#include <shogun/base/init.h>
#include <iostream>

inline void shogunModels()
{
    using namespace shogun;

    init_shogun_with_defaults();
    // odczytanie danych we własnym pośrednim typie danych
    auto classificationDatasetTemp =
        readShogunCsvData("wdbc_data_with_labels_tn.csv", LabelPos::LAST);
    auto regressionDatasetTemp =
        readShogunCsvData("IronGlutathione.csv", LabelPos::LAST);
    // rozdzielenie danych na regresory i zmienne odpowiedzi
    auto classificationTrainFeatures =
        some<CDenseFeatures<float64_t>>(
            classificationDatasetTemp.trainInputs);
    auto classificationTestFeatures =
        some<CDenseFeatures<float64_t>>(
            classificationDatasetTemp.testInputs);
    auto classificationTrainLabels =
        some<CMulticlassLabels>(
            classificationDatasetTemp.trainOutputs);
    auto classificationTestLabels =
        some<CMulticlassLabels>(
            classificationDatasetTemp.testOutputs);
    auto regressionTrainFeatures =
        some<CDenseFeatures<float64_t>>(
            regressionDatasetTemp.trainInputs);
    auto regressionTestFeatures =
        some<CDenseFeatures<float64_t>>(
            regressionDatasetTemp.testInputs);
    auto regressionTrainLabels =
        some<CRegressionLabels>(
            regressionDatasetTemp.trainOutputs);
    auto regressionTestLabels =
        some<CRegressionLabels>(
            regressionDatasetTemp.testOutputs);
    
    // wywołanie modeli
    shogunLinear(
        regressionTrainFeatures, 
        regressionTestFeatures, 
        regressionTrainLabels, 
        regressionTestLabels);
    shogunSVM(
        classificationTrainFeatures, 
        classificationTestFeatures, 
        classificationTrainLabels, 
        classificationTestLabels);

    exit_shogun();
}

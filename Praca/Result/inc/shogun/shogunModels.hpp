#pragma once

#include <inc/shogun/csv.hpp>
#include <inc/shogun/linear.hpp>
#include <inc/shogun/logistic.hpp>
#include <inc/shogun/svm.hpp>
#include <inc/shogun/neural.hpp>

#include <shogun/base/init.h>

inline void shogunModels()
{
    using namespace shogun;

    init_shogun_with_defaults();

    // odczytanie danych we własnym pośrednim typie danych
    auto classificationDatasetTemp =
        readShogunCsvData("wdbc_data_with_labels.csv", LabelPos::FIRST);
    auto regressionDatasetTemp =
        readShogunCsvData("IronGlutathione.csv", LabelPos::LAST);
    // rozdzielenie danych na regresory i zmienne odpowiedzi
    auto classificationFeatures =
        some<CDenseFeatures<float64_t>>(
            classificationDatasetTemp.inputs);
    auto classificationLabels =
        some<CBinaryLabels>(
            classificationDatasetTemp.outputs);
    auto regressionFeatures =
        some<CDesneFeatures<float64_t>>(
            regressionDatasetTemp.inputs);
    auto regressionLabels =
        some<CRegressionLabels>(
            regressionDatasetTemp.outputs);

    // wywołanie modeli
    shogunLinear(regressionFeatures, regressionLabels);
    shogunLogistic(classificationFeatures, classificationLabels);
    shogunSVM(classificationFeatures, classificationLabels);
    sharkNeural(classificationFeatures, classificationLabels);

    exit_shogun();
}
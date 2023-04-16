#pragma once

#include <inc/shogun/csv.hpp>

inline void shogunModels()
{
    using namespace shogun;

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
        wrap(new CMulticlassLabels(
            classificationDatasetTemp.outputs.get_column(0)));
    auto regressionFeatures =
        some<CDesneFeatures<float64_t>>(
            regressionDatasetTemp.inputs);
    auto regressionLabels =
        some<CRegressionLabels>(
            regressionDatasetTemp.outputs);

}
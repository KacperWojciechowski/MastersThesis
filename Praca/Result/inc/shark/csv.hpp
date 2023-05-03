#pragma once

#include <inc/shark/printEvaluation.hpp>

#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Data/Csv.h>
#include <shark/Data/SparseData.h>

inline auto sharkReadCsvData(std::string filePath, Task task)
{
    using namespace shark;

    // odczytaj zawartość pliku
    std::ifstream file(filePath);

    if (task == Task::CLASSIFICATION)
    {
        ClassificationDataset trainData;
        importCSV(trainData, filePath.c_str(), FIRST_COLUMN, ',');
        return trainData;
    }
    else
    {
        RegressionDataset trainData;
        importCSV(trainData, filePath.c_str(), LAST_COLUMN, ',');
        return trainData;
    }
}
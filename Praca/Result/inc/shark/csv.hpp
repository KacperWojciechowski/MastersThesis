#pragma once

#include <inc/shark/printEvaluation.hpp>

#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Data/Csv.h>
#include <shark/Data/SparseData.h>

template<typename DatasetType>
inline DatasetType sharkReadCsvData(std::string filePath, Task task)
{
    using namespace shark;

    // odczytaj zawartość pliku
    std::ifstream file(filePath);

    if (task == Task::CLASSIFICATION)
    {
        DatasetType trainData;
        importCSV(trainData, filePath.c_str(), FIRST_COLUMN, ',');
        return trainData;
    }
    else
    {
        DatasetType trainData;
        importCSV(trainData, filePath.c_str(), LAST_COLUMN, ',');
        return trainData;
    }
}
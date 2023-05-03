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
    std::string trainDataString(std::istreambuf_iterator<char>(file), 
                                std::istreambuf_iterator<char>());

    if (task == Task::CLASSIFICATION)
    {
        ClassificationDataset trainData;
        csvStringToData(trainData, trainDataString, FIRST_COLUMN, ',');
        return trainData;
    }
    else
    {
        RegressionDataset trainData;
        csvStringToData(trainData, trainDataString, LAST_COLUMN, ',');
        return trainData;
    }
}
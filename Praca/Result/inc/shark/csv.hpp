#pragma once

#include <inc/shark/printEvaluation.hpp>

inline auto sharkReadCsvData(std::string filePath, constexpr Task task)
{
    using namespace shark;

    // odczytaj zawartość pliku
    std::ifstream file(filePath);
    std::string trainDataString(std::istreambuf_iterator<char>(file), 
                                std::istreambuf_iterator<char>());

    if constexpr (task == Task::CLASSIFICATION)
    {
        ClassificationDataset trainData;
        csvStringToData(trainData, trainDataString, FIRST_COLUMN);
        return trainData;
    }
    else
    {
        RegressionDataset trainData;
        csvStringToData(trainData, trainDataString, LAST_COLUMN);
        return trainData;
    }
}
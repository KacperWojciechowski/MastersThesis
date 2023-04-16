#pragma once

inline auto sharkReadCsvData(std::string filePath)
{
    // odczytaj zawartość pliku
    std::ifstream file(filePath);
    std::string trainDataString(std::istreambuf_iterator<char>(file), 
                                std::istreambuf_iterator<char>());

    shark::ClassificationDataset trainData;
    shark::csvStringToData(trainData, trainDataString, shark::FIRST_COLUMN);

    return trainData;
}
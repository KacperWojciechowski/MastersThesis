#pragma once

#include <vector>
#include <dlib/matrix.h>

inline std::vector<dlib::matrix<double>> dlibNormalize(
    const std::vector<dlib::matrix<double>& data)
{
    using namespace dlib;
    // creating and training the normalizer
    vector_normalizer<matrix<double>> normalizer;
    normalizer.train(data);
    // processing data
    std::vector<matrix<double>> processedData(data.size());
    for (auto dataMatrix : data)
        processedData.emplace_back(normalizer(data));
    return processedData;
}
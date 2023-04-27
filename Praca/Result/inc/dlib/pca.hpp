#pragma once

#include <dlib/matrix.h>
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/statistics.h>
#include <vector>

inline std::vector<dlib::matrix<double>> dlibPCA(
    const std::vector<dlib::matrix<double>>& data, 
    std::size_t desiredDimensions)
{
    using namespace dlib;
    // utworzenie i trening reduktora
    vector_normalizer_pca<matrix<double>> pca;
    pca.train(data, desiredDimensions/data[0].nr());
    // przygotowanie kontenera na przetworzone dane
    std::vector<matrix<double>> processedData;
    processedData.reserve(data.size());
    // przetwarzanie danych
    for(auto& sample : data)
    {
        processedData.emplace_back(pca(sample));
    }
    return processedData;
}
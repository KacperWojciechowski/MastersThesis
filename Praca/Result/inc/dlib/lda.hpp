#pragma once

#include <dlib/matrix.h>
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/statistics.h>
#include <vector>

inline std::vector<dlib::matrix<double>> dlibLDA(
    const dlib::matrix<double>& data,
    const std::vector<unsigned long>& labels 
    std::size_t desiredDimensions)
{
    using namespace dlib;
    // creating objects needed for reduction
    matrix<double, 0, 1> mean;
    auto transform = data;
    // converting data into reduction matrix
    compute_lda_transform(transform, mean, labels, desiredDimensions);
    // preparing container for transformed data
    std::vector<matrix<double>> transformedData;
    transformedData.reserve(data.nr());
    // dimensionality reduction
    for (long i = 0; i < data.nr(); ++i)
    {
        transformedData.emplace_back(transform * trans(rowm(data, i)) - mean);
    }
    // returning the transformed rows vector
    return transformedData;
}
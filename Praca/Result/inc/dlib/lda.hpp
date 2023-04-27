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
    // utworzenie obiektow potrzebnych do redukcji
    matrix<double, 0, 1> mean;
    auto transform = data;
    // przeksztalcenie danych w macierz redukcyjną
    compute_lda_transform(transform, mean, labels, desiredDimensions);
    // przygotowanie kontenera na przetworzone dane
    std::vector<matrix<double>> transformedData;
    transformedData.reserve(data.nr());
    // redukcja wymiarowości
    for (long i = 0; i < data.nr(); ++i)
    {
        transformedData.emplace_back(transform * trans(rowm(data, i)) - mean);
    }
    // zwrócenie wektora przetworzonych wierszy
    return transformedData;
}
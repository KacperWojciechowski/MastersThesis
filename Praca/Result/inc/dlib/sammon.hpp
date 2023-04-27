#pragma once

#include <dlib/matrix.h>
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/statistics.h>
#include <vector>

inline std::vector<matrix<double>> dlibSammon(
    const std::vector<matrix<double>>& data,
    long desiredDimensions)
{
    using namespace dlib;
    // utworzenie obiektu mapowania
    dlib::sammon_projekction sammon;
    // redukcja wymiarowości
    return sammon(data, desiredDimensions);
}
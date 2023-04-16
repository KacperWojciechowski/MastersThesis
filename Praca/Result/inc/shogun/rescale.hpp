#pragma once

#include <shogun/preprocessor/RescaleFeatures.h>

inline void normalize(auto& inputs)
{
    using namespace shogun;

    // utworzenie normalizera
    auto scaler = wrap(new CRescaleFeatures);
    // nauka normalizera oraz przeprowadzenie normalizacji
    scaler->fit(inputs);
    scaler->transform(inputs);
}
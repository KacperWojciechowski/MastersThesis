#pragma once

#include <shogun/preprocessor/RescaleFeatures.h>

inline void normalize(auto& inputs)
{
    using namespace shogun;

    // creating normalizer
    auto scaler = wrap(new CRescaleFeatures);
    // training and transforming
    scaler->fit(inputs);
    scaler->transform(inputs);
}

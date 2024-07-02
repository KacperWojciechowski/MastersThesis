#include <shogun/preprocessor/RescaleFeatures.h>

// [...]
// training data creation
auto features = shogun::some<shogun::CDenseFeatures<DataType>>(inputs);
// [...]
// normalizer object creation
auto scaler = shogun::wrap(new shogun::CRescaleFeatures());
// teaching the normalizer the minimal and maximal data values
scaler->fit(features);
// processing the training data with the min-max function
scaler->transform(features);

// print normalizing result
auto features_matrix = features->get_feature_matrix();
for (int i = 0; i < n; ++i)
{
    std::cout << "Sample idx " << i << " ";
    features_matrix.get_column(i).display_vector();
}
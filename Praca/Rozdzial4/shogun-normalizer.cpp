#include <shogun/preprocessor/RescaleFeatures.h>

// [...]
// utwórz obiekt danych uczących
auto features = shogun::some<shogun::CDenseFeatures<DataType>>(inputs);
// [...]
// utwórz obiekt normalizera
auto scaler = shogun::wrap(new shogun::CRescaleFeatures());
// naucz normalizer minimalnych i maksymalnych wartości danych
scaler->fit(features);
// przetwórz dane uczące według funkcji min-max
scaler->transform(features);

// wyświetl wynik normalizacji
auto features_matrix = features->get_feature_matrix();
for (int i = 0; i < n; ++i)
{
    std::cout << "Sample idx " << i << " ";
    features_matrix.get_column(i).display_vector();
}
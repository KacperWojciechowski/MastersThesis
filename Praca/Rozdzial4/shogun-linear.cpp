using namespace shogun;
// [...]
// utworzenie zestawu danych
auto x = some<CDenseFeatures<float64_t>>(x_values);
auto y = some<CRegressionLabels>(y_values);
// utworzenie modelu
float64_t tau_regularization = 0.0001;
auto lr = some<ClinearRidgeRegression>(tau_regularization, nullptr, nullptr);
// konfiguracja i trening modelu
lr->set_labels(y);
lr->train(x);
// wykonanie predykcji dla nowych danych
auto new_x = some<CDenseFeatures<float64_t>>(new_x_values);
auto y_predict = lr->apply_regression(new_x);
// odczytanie wag
auto weights = lr->get_w();
// wyliczenie wartości funkcji straty
y_predict = lr->apply_regression(x);
auto eval = some<CMeanSquaredError>();
auto mse = eval->evaluate(y_predict, y);
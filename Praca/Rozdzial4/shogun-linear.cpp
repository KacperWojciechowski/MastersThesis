using namespace shogun;
// [...]
// creating the datasets
auto x = some<CDenseFeatures<float64_t>>(x_values);
auto y = some<CRegressionLabels>(y_values);
// model creation
float64_t tau_regularization = 0.0001;
auto lr = some<ClinearRidgeRegression>(tau_regularization, nullptr, nullptr);
// model configuration and training
lr->set_labels(y);
lr->train(x);
// calculating predictions
auto new_x = some<CDenseFeatures<float64_t>>(new_x_values);
auto y_predict = lr->apply_regression(new_x);
// reading the weights
auto weights = lr->get_w();
// calculating the loss function value
y_predict = lr->apply_regression(x);
auto eval = some<CMeanSquaredError>();
auto mse = eval->evaluate(y_predict, y);
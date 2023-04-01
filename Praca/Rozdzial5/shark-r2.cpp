using namespace shark;

// [...]

// błąd średniokwadratowy
SquaredLoss<> mse_loss;
auto mse = mse_loss(train_data.labels(), predictions);

// metryka R^2
auto var = variance(train_data.labels());
auto r_squared = 1 - mse / var(0);

// metryka adjusted R^2
auto adj_r_squared = 1 - (1 - r_squared)((num_regressors - 1)/
                     (num_regressors - data_size - 1));

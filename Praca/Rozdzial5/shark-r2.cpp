using namespace shark;

// [...]

// mean squared error
SquaredLoss<> mse_loss;
auto mse = mse_loss(train_data.labels(), predictions);

// R^2 metric
auto var = variance(train_data.labels());
auto r_squared = 1 - mse / var(0);

// adjusted R^2 metric
auto adj_r_squared = 1 - (1 - r_squared)((num_regressors - 1)/
                     (num_regressors - data_size - 1));

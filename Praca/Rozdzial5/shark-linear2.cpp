// [...]

using namespace shark;

// [...]

LinearRegression trainer;
LinearModel<> model;
trainer.train(model, data);

std::cout << "intercept: " << model.offset() << std::endl;
std::cout << "matrix: " << model.matrix() << std::endl;

auto prediction = model(test);
SquaredLoss<> loss;
auto se = loss(test.labels(), prediction);
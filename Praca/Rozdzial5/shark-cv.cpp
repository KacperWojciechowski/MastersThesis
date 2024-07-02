using namespace shark;

// [...]

// processing the training data
const unsigned int num_folds = 5;
CVFolds<RegressionDataset> folds =
    createCVSameSize<RealVector, RealVector>(train_data, num_folds);

// preparing the target parameters for the model
double regularization_factor = 0.0;
double polynomial_degree = 8;
int num_epochs = 300;

// configuring the target model
PolynomialModel<> model;
PolynomialRegression trainer(regularization_factor, polynomial_degree,
                             num_epochs);

// creating the loss object and cross validation
AbsoluteLoss<> loss;
CrossValidationError<PolynomialModel<>, RealVector> cv_error(
    folds, &trainer, &model, &trainer, &loss);

// generating a grid
GridSearch grid;
std::vector<double> min(2);
std::vector<double> max(2);
std::vector<std::size_t> sections(2);

// regularization
min[0] = 0.0;
max[0] = 0.00001;
sections[0] = 6;

// polynomial degree
min[1] = 4;
max[1] = 10.0;
sections[1] = 6;
grid.configure(min, max, sections);

// training and configuration process
grid.step(cv_error);
trainer.setParameterVector(grid.solution().point);
trainer.train(model, train_data);
using namespace shark;

// [...]

// przetworzenie danych uczących
const unsigned int num_folds = 5;
CVFolds<RegressionDataset> folds =
    createCVSameSize<RealVector, RealVector>(train_data, num_folds);

// przygotowanie parametrów dla docelowego modelu
double regularization_factor = 0.0;
double polynomial_degree = 8;
int num_epochs = 300;

// konfiguracja docelowego modelu
PolynomialModel<> model;
PolynomialRegression trainer(regularization_factor, polynomial_degree,
                             num_epochs);

// utworzenie obiektu błędu oraz sprawdzianu krzyżowego
AbsoluteLoss<> loss;
CrossValidationError<PolynomialModel<>, RealVector> cv_error(
    folds, &trainer, &model, &trainer, &loss);

// utworzenie siatki
GridSearch grid;
std::vector<double> min(2);
std::vector<double> max(2);
std::vector<std::size_t> sections(2);

// regularyzacja
min[0] = 0.0;
max[0] = 0.00001;
sections[0] = 6;

// stopień wielomianu
min[1] = 4;
max[1] = 10.0;
sections[1] = 6;
grid.configure(min, max, sections);

// proces uczenia i konfiguracja modelu
grid.step(cv_error);

trainer.setParameterVector(grid.solution().point);
trainer.train(model, train_data);
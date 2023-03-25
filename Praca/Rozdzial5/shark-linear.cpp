#include <shark/Data/Csv.h>
#include <shark/Algorithms/GradientDescent/CG.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Models/LinearModel.h>

using namespace shark;

RegressionDataset loadData(const std::string& dataFile, 
                           const std::string& labelsFile)
{
    Data<RealVector> inputs;
    Data<RealVector> labels;

    try
    {
        importCSV(inputs, regressorsFile, ' ');
        importCSV(labels, targetFile, ' ');
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }
    RegressionDataset data(inputs, labels);
    return data;
}
// [...]

// odczytanie danych z plików oraz podział na dane uczące i testowe
RegressionDataset data = loadData("data/regressionInputs.csv", 
                                  "data/regressionLabels.csv");
RegressionDataset test = splitAtElement(data, static_cast<std::size_t>(
                                                0.8*data.numberOfElements()));
// przygotowanie modelu
LinearModel<> model(inputDimension(data), labelDimension(data));
SquaredLoss<> loss;
ErrorFunction errorFunction(data, &model, &loss);

// przygotowanie i wyszkolenie optymizatora
CG optimizer;
errorFunction.init();
optimizer.init(errorFunction);
for (int i = 0; i < 100; ++i)
{
    optimizer.step(errorFunction);
}

// przeliczenie predykcji modelu dla danych testowych
model.setParameterVector(optimizer.solution().point);
Data<RealVector> predictions = model(test.inputs());
double testError = loss.eval(test.labels(), predictions);
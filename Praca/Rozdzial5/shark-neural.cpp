using namespace shark;

// [...]

// utworzenie zestawu danych
size_t n = 10000;
std::vector<RealVector> x_data[n];
std::vector<RealVector> y_data[n];
Data<RealVector> x = createDataFromRange(x_data);
Data<RealVector> y = createDataFromRange(y_data);
RegressionDataset train_data(x, y);

// zdefiniowanie warstw sieci
using DenseLayer = LinearModel<RealVector, TanhNeuron>;
DenseLayer layer1(1, 32, true);
DenseLayer layer2(32, 16, true);
DenseLayer layer3(16, 8, true);
LinearModel<RealVector> output(8, 1, true);
// połączenie warstw
auto network = layer1 >> layer2 >> layer3 >> output;

// utworzenie i konfiguracja funkcji straty
SquaredLoss<> loss;
ErrorFunction<> error(train_data, &network, &loss, true);
TwoNormRegularizer<> regularizer(error.numberOfVariables());
double weight_decay = 0.0001;
error.setRegularizer(weight_decay, &regularizer);
error.init();

// inicjalizacja wag sieci
initRandomNormal(network, 0.001);

// utworzenie i konfiguracja optymalizatora
SteepestDescent<> optimizer;
optimizer.setMomentum(0.5);
optimizer.setLearningRate(0.01);
optimizer.init(error);

// przeprowadzenie procesu uczenia
std::size_t epochs = 1000;
std::size_t iterations = train_data.numberOfBatches();
// pętla przechodząca przez kolejne epoki
for (std::size_t epoch = 0; epoch != epochs; ++epoch)
{
    double avg_loss = 0.0;
    // pętla operująca na pojedynczych batch'ach
    for (std::size_t i = 0; i != iterations; ++i)
    {
        // wykonanie kroku optymalizatora
        optimizer.step(error);
        // zapisanie częściowej wartości średniej funkcji straty
        if (i % 100 == 0)
        {
            avg_loss += optimizer.solution().value;
        }
    }
    // obliczenie średniej wartości funkcji straty
    avg_loss /= iterations;
    std::cout << "Epoch " << epoch << " | Avg. Loss " << avg_loss << std::endl;
}
// konfiguracja modelu do docelowego użycia
network.setParameterVector(optimizer.solution().point);
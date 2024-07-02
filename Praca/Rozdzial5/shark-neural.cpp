using namespace shark;

// [...]

// creating a dataset
size_t n = 10000;
std::vector<RealVector> x_data[n];
std::vector<RealVector> y_data[n];
Data<RealVector> x = createDataFromRange(x_data);
Data<RealVector> y = createDataFromRange(y_data);
RegressionDataset train_data(x, y);

// defining neural layers
using DenseLayer = LinearModel<RealVector, TanhNeuron>;
DenseLayer layer1(1, 32, true);
DenseLayer layer2(32, 16, true);
DenseLayer layer3(16, 8, true);
LinearModel<RealVector> output(8, 1, true);
// connecting the layers
auto network = layer1 >> layer2 >> layer3 >> output;

// creating and configuring a loss function
SquaredLoss<> loss;
ErrorFunction<> error(train_data, &network, &loss, true);
TwoNormRegularizer<> regularizer(error.numberOfVariables());
double weight_decay = 0.0001;
error.setRegularizer(weight_decay, &regularizer);
error.init();

// initializing the weights
initRandomNormal(network, 0.001);

// creating and configuring the optimizer
SteepestDescent<> optimizer;
optimizer.setMomentum(0.5);
optimizer.setLearningRate(0.01);
optimizer.init(error);

// training
std::size_t epochs = 1000;
std::size_t iterations = train_data.numberOfBatches();
// loop that goes through epochs
for (std::size_t epoch = 0; epoch != epochs; ++epoch)
{
    double avg_loss = 0.0;
    // loop that goes through batches
    for (std::size_t i = 0; i != iterations; ++i)
    {
        // performing an optimizer step
        optimizer.step(error);
        // saving partial loss function value
        if (i % 100 == 0)
        {
            avg_loss += optimizer.solution().value;
        }
    }
    // calculating the average loss value
    avg_loss /= iterations;
    std::cout << "Epoch " << epoch << " | Avg. Loss " << avg_loss << std::endl;
}
// configuring the model for the final use
network.setParameterVector(optimizer.solution().point);
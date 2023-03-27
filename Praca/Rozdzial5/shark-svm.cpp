#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

// umożliwia wygenerowanie przykładowego zestawu danych
#include <shark/Data/DataDistribution.h>

using namespace shark;

// ...

// wygenerowanie przykładowego zestawu danych
unsigned int trainingDataPoints = 500;
unsigned int testDataPoints = 10000; 
Chessboard problem;
ClassificationDataset training = problem.generateDataset(trainingDataPoints);
ClassificationDataset test = problem.generateDataset(testDataPoints);

// przygotowanie kernelu
double gamma = 0.5;     // przepustowość kernelu
GaussianRbfKernel<> kernel(gamma);
KernelClassifier<RealVector> kc; // liniowa funkcja dla przestrzeni kernelu

// przygotowanie klasy trenera
double regularization = 1000.0;
bool bias = true;
// drugi parametr szablonu określa wykorzystanie typu double dla pamięci 
// cache modelu zamiast float
CSvmTrainer<RealVector, double> trainer(&kernel, regularization, bias);

// konfiguracja modelu
trainer.sparsify() = false; // zachowanie wektorów nie-nośnych
trainer.stopCondition().minAccuracy = 1e-6;
trainer.setCacheSize(0x1000000);


// trenowanie modelu
trainer.train(kc, training);

// wyświetlenie informacji diagnostycznych o uczeniu
std::cout << "Needed " << trainer.solutionProperties().seconds 
          << " seconds to reach a dual of "
          << trainer.solutionProperties().value << std::endl;

// użycie modelu
auto predictions = kc(test.inputs());
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

// allows for generating a sample dataset
#include <shark/Data/DataDistribution.h>

using namespace shark;

// ...

// generating of sample data
unsigned int trainingDataPoints = 500;
unsigned int testDataPoints = 10000; 
Chessboard problem;
ClassificationDataset training = problem.generateDataset(trainingDataPoints);
ClassificationDataset test = problem.generateDataset(testDataPoints);

// preparing the kernel
double gamma = 0.5;     // throughput
GaussianRbfKernel<> kernel(gamma);
KernelClassifier<RealVector> kc; // linear function for the kernel space

// preparing the trainer
double regularization = 1000.0;
bool bias = true;
// the second parameter of the template specifies the use of double type 
// for the cache memory of the model instead of float
CSvmTrainer<RealVector, double> trainer(&kernel, regularization, bias);

// model configuration
trainer.sparsify() = false; // keeping the non-support vector
trainer.stopCondition().minAccuracy = 1e-6;
trainer.setCacheSize(0x1000000);

// training
trainer.train(kc, training);

// printing diagnostic data
std::cout << "Needed " << trainer.solutionProperties().seconds 
          << " seconds to reach a dual of "
          << trainer.solutionProperties().value << std::endl;

// model use
auto predictions = kc(test.inputs());
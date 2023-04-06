#include <fstream>
#include <filesystem>
#include <regex>
#include <string>
#include <string_view>

#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/LinearModel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

void printSharkModelEvaluation(const auto& labels, const auto& predictions)
{
    SquaredLoss<> loss;
    auto mse = loss(labels, predictions);
    std::cout << "MSE: " << mse << std::endl;
    
    auto var = shark::variance(labels);
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;

    constexpr bool invertToPositiveROC = true;
    NegativeAUC roc(invertToPositiveROC);
    auto auc_roc = roc(labels, predictions);
    std::cout << "AUC ROC: " << auc_roc << std::endl << std::endl;
}

void sharkLogisticRegression(const shark::ClassificationDataset& trainData,
                             const shark::ClassificationDataset& testData)
{
    shark::LinearClassifier<RealVector> logisticModel;
    shark::LogisticRegression<RealVector> trainer;
    
    trainer.train(logisticModel, trainData);

    auto trainPredictions = logisticModel(trainData.inputs());
    auto testPredictions = logisticModel(testData.inputs());

    std::cout << "-----SharkLogisticRegression-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    printSharkModelEvaluation(trainData.labels(), trainPredictions);

    std::cout << "Test data model evaluation:" << std::endl;
    printSharkModelEvaluation(testData.labels(), testPredictions); 
}

void sharkSVM(const shark::ClassificationDataset& trainData,
              const shark::ClassificationDataset& testData)
{
    double gamma = 0.5;
    shark::GaussianRbfKernel<> kernel(gamma);
    KernelClassifier<RealVector> svm;
    double regularization = 1000.0;
    bool bias = true;
    
    CSvmTrainer<RealVector, double> trainer(&kernel, regularization, bias);
    trainer.sparsify() = false;
    trainer.stopCondition().minAccuracy=1e-6;
    trainer.setCacheSize(0x1000000);

    trainer.train(svm, trainData);

    auto trainPredictions = svm(trainData.inputs());
    auto testPredictions = svm(testData.inputs());

    std::cout << "-----SharkSVM-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    printSharkModelEvaluation(trainData.labels(). trainPredictions);

    std::cout << "Test data model evaluation:" << std::endl;
    printSharkModelEvaluation(testData.labels(), testPredictions);
}

auto preprocessAndReadCsvData(std::string filePath)
{
    // odczytaj zawartość pliku
    std::ifstream file(filePath);
    std::string trainDataString(std::istreambuf_iterator<char>(file), 
                                std::istreambuf_iterator<char>());

    trainDataString.erase(0, trainDataString.find_first_of("\n") + 1);

    trainDataString = 
        std::regex_replace(trainDataString, std::regex("B"), "0");
    trainDataString = 
        std::regex_replace(trainDataString, std::regex("M"), "1");

    shark::ClassificationDataset trainData;
    shark::csvStringToData(trainData, trainDataString, shark::FIRST_COLUMN);

    return trainData;
}

int main()
{
    constexpr std::string_view dataFilePath = "wdbc_data_with_labels.csv"; 
    auto trainData = preprocessAndReadCsvData(dataFilePath.data());

    auto testData = shark::splitAtElement(
    trainData, static_cast<std::size_t>(
        0.8 * trainData.numberOfElements()));

    sharkLogisticRegression(trainData, testData);
    sharkSVM(trainData, testData);

}
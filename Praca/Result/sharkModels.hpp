#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/LinearModel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

inline void printSharkModelEvaluation(const auto& labels, const auto& predictions)
{
    // błąd średniokwadratowy
    shark::SquaredLoss<> loss;
    auto mse = loss(labels, predictions);
    std::cout << "MSE: " << mse << std::endl;
    
    // metryka R^2
    auto var = shark::variance(labels);
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;

    // wartość krzywej ROC
    constexpr bool invertToPositiveROC = true;
    shark::NegativeAUC roc(invertToPositiveROC);
    auto auc_roc = roc(labels, predictions);
    std::cout << "AUC ROC: " << auc_roc << std::endl << std::endl;
}

inline void sharkLogisticRegression(const shark::ClassificationDataset& trainData,
                             const shark::ClassificationDataset& testData)
{
    // utworzenie modelu
    shark::LinearClassifier<RealVector> logisticModel;
    shark::LogisticRegression<RealVector> trainer;
    
    // wytrenowanie
    trainer.train(logisticModel, trainData);

    // ewaluacja
    auto trainPredictions = logisticModel(trainData.inputs());
    auto testPredictions = logisticModel(testData.inputs());

    std::cout << "-----Shark Logistic Regression-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    printSharkModelEvaluation(trainData.labels(), trainPredictions);

    std::cout << "Test data model evaluation:" << std::endl;
    printSharkModelEvaluation(testData.labels(), testPredictions); 
}

inline void sharkSVM(const shark::ClassificationDataset& trainData,
              const shark::ClassificationDataset& testData)
{
    // utworzenie jądra
    double gamma = 0.5;
    shark::GaussianRbfKernel<> kernel(gamma);
    KernelClassifier<RealVector> svm;
    double regularization = 1000.0;
    bool bias = true;
    
    // utworzenie i konfiguracja modelu
    CSvmTrainer<RealVector, double> trainer(&kernel, regularization, bias);
    trainer.sparsify() = false;
    trainer.stopCondition().minAccuracy=1e-6;
    trainer.setCacheSize(0x1000000);

    // trening
    trainer.train(svm, trainData);

    // ewaluacja
    auto trainPredictions = svm(trainData.inputs());
    auto testPredictions = svm(testData.inputs());

    std::cout << "-----Shark SVM-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    printSharkModelEvaluation(trainData.labels(). trainPredictions);

    std::cout << "Test data model evaluation:" << std::endl;
    printSharkModelEvaluation(testData.labels(), testPredictions);
}

inline auto sharkReadCsvData(std::string filePath)
{
    // odczytaj zawartość pliku
    std::ifstream file(filePath);
    std::string trainDataString(std::istreambuf_iterator<char>(file), 
                                std::istreambuf_iterator<char>());

    shark::ClassificationDataset trainData;
    shark::csvStringToData(trainData, trainDataString, shark::FIRST_COLUMN);

    return trainData;
}

inline void sharkNN(const shark::ClassificationDataset& trainData,
             const shark::ClassificationDataset& testData)
{

}

inline void sharkKNN(const shark::ClassificationDataset& trainData,
              const shark::ClassificationDataset& testData)
{
    // utworzenie i konfiguracja drzewa oraz algorytmu
    KDTree<RealVector> tree(trainData.inputs());
    TreeNearestNeighbors<RealVector,unsigned int> algorithm(trainData, &tree);

    // konfiguracja modelu
    const unsigned int K = 2; // ilość sąsiadów dla algorytmu kNN
    NearestNeighborModel<RealVector, unsigned int> KNN(&algorithm, K);

    // ewaluacja modelu
    auto trainPredictions = KNN(trainData.inputs());
    auto testPredictions = KNN(testData.inputs());

    std::cout << "-----Shark KNN-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    printSharkModelEvaluation(trainData.labels(). trainPredictions);

    std::cout << "Test data model evaluation:" << std::endl;
    printSharkModelEvaluation(testData.labels(), testPredictions);
}

inline void sharkModels()
{
    auto classificationTrainData = sharkReadCsvData("wdbc_data_with_labels.csv")
    auto classificationTestData = shark::splitAtElement(
        classificationTrainData,
        static_cast<std::size_t>(0.8*classificationTrainData.numberOfElements()));

    // wywołanie modeli
    sharkLogisticRegression(classificationTrainData, classificationTestData);
    sharkSVM(classificationTrainData, classificationTestData);
    sharkKNN(classificationTrainData, classificationTestData);
}
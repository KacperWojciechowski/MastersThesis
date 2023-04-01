#include <shark/Algorithms/Trainers/LDA.h>
// [...]

using namespace shark;

int main(int argc, char **argv)
{
    // import danych
    ClassificationDataset data;
    try
    {
        importCSV(data, argv[1], LAST_COLUMN, ' ');
    }
    catch(...)
    {
        std::cerr << "Unable to read data from file " << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }

    // wyświetlenie informacji o danych
    std::cout << "overall number of data points: " << data.numberOfElements()
              << " number of classes: " << numberOfClasses(data)
              << " input dimension: " << inputDimenstion(Data) << std::endl;

    // wyodrębnienie danych testowych
    auto test = splitAtElement(data, .5 * data.numberOfElements());
    // utworzenie i wytrenowanie modelu
    LDA ldaTrainer;
    LinearClassifier<> lda;
    ldaTrainer.train(lda, data);

    // analiza predykcji i dokładności modelu
    Data<unsigned int> prediction;
    ZeroOneLoss<unsigned int> loss;
    prediction = lda(data.inputs());
    std::cout << "LDA on training set accuracy: "
        << 1. - loss(data.labels(), prediction) << std::endl;
    prediction = lda(test.inputs());
    std::cout << "LDA on test set accuracy: " 
        << 1. - loss(test.labels(), prediction) << std::endl;

    return 0;
}
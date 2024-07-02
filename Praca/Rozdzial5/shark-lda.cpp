#include <shark/Algorithms/Trainers/LDA.h>
// [...]

using namespace shark;

int main(int argc, char **argv)
{
    // importing data
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

    // presenting data information
    std::cout << "overall number of data points: " << data.numberOfElements()
              << " number of classes: " << numberOfClasses(data)
              << " input dimension: " << inputDimenstion(Data) << std::endl;

    // extracting validation data
    auto test = splitAtElement(data, .5 * data.numberOfElements());
    // creating and training the model
    LDA ldaTrainer;
    LinearClassifier<> lda;
    ldaTrainer.train(lda, data);

    // prediction and model accuracy analysis
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
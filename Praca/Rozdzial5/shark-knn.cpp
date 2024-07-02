#include <shark/Data/Csv.h>
#include <shark/Models/NearestNeighborModel.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Models/Trees/KDTree.h>

using namespace shark;

int main() 
{
    std::string filename = "sample_data_file.csv"
    
    // reading data from file
    ClassificationDataset data;
    try 
    {
        importCSV(data, filename, LAST_COLUMN, ' ');
    }
    catch (...) 
    {
        std::cerr << "unable to read data from file " <<  filename 
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // presenting data information
    std::cout << "number of data points: " << data.numberOfElements()
              << " number of classes: " << numberOfClasses(data)
              << " input dimension: " << inputDimension(data) 
              << std::endl;

    // selecting the  validation data
    ClassificationDataset dataTest = splitAtElement(
        data, 
        static_cast<std::size_t>(
            .5 * data.numberOfElements())
        );
    // printing data information
    std::cout << "training data points: " << data.numberOfElements() 
              << std::endl;
    std::cout << "test data points: " << dataTest.numberOfElements() 
              << std::endl;

    // creation and configuration of the tree and algorithm
    KDTree<RealVector> tree(data.inputs());
    TreeNearestNeighbors<RealVector,unsigned int> algorithm(data, &tree);

    // model configuration
    const unsigned int K = 1; // neighbor count for kNN algorithm
    NearestNeighborModel<RealVector, unsigned int> KNN(&algorithm, K);

    // example use of the model
    ZeroOneLoss<unsigned int> loss;
    auto prediction = KNN(data.inputs());
    std::cout << K << "-KNN on training set accuracy: " 
              << 1. - loss.eval(data.labels(), prediction) << std::endl;
    prediction = KNN(dataTest.inputs());
    std::cout << K << "-KNN on test set accuracy: " 
              << 1. - loss.eval(dataTest.labels(), prediction) 
              << std::endl;

    return 0;
}
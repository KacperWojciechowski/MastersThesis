using namespace std;

void RFClassification(const shark::ClassificationDataset& train,
                      const shark::ClassificationDataset& test)
{
    using namespace shark;

    // creating and configuring the trainer
    RFTrainer<unsigned int> trainer;
    trainer.setNTrees(100);
    trainer.setMinSplit(10);
    trainer.setMaxDepth(10);
    trainer.setNodeSize(5);
    trainer.minImpurity(1.e-10);
    // creating the classifier
    RFClassifier<unsigned int> rf;
    trainer.train(rf, train);
    // evaluation
    ZeroOneLoss<unsigned int> loss;
    auto predictions = rf(test.inputs());
    double accuracy = 1. - loss.eval(test.labels(), predictions);
    std::cout << "Random Forest accuracy = " << accuracy << std::endl;
}
using namespace std;

void RFClassification(const shark::ClassificationDataset& train,
                      const shark::ClassificationDataset& test)
{
    using namespace shark;

    // utworzenie i konfiguracja trenera
    RFTrainer<unsigned int> trainer;
    trainer.setNTrees(100);
    trainer.setMinSplit(10);
    trainer.setMaxDepth(10);
    trainer.setNodeSize(5);
    trainer.minImpurity(1.e-10);
    // utworzenie klasyfikatora
    RFClassifier<unsigned int> rf;
    trainer.train(rf, train);
    // ewaluacja
    ZeroOneLoss<unsigned int> loss;
    auto predictions = rf(test.inputs());
    double accuracy = 1. - loss.eval(test.labels(), predictions);
    std::cout << "Random Forest accuracy = " << accuracy << std::endl;
}
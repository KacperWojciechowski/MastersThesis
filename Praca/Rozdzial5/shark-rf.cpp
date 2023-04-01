using namespace std;

// [...]

void RFClassification(const ClassificationDataset& train,
                      const ClassificationDataset& test)
{
    RFTrainer<unsigned int> trainer;
    trainer.setNTrees(100);
    trainer.setMinSplit(10);
    trainer.setMaxDepth(10);


    // DOKOŃCZYĆ !!!!
}
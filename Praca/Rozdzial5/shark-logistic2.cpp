using namespace shark;

// [...]
void SimpleLR(const ClassificationDataset& train,
              const ClassificationDataset& test)
{
    // creating the model and the trainer
    LinearClassifier<RealVector> model;
    LogisticRegression<RealVector> trainer;
    
    // training
    trainer.train(model, train);

    // use example
    auto predictions = model(test.inputs());
    // [...]
}

using namespace shark;

// [...]
void SimpleLR(const ClassificationDataset& train,
              const ClassificationDataset& test)
{
    // utworzenie modelu oraz trenera
    LinearClassifier<RealVector> model;
    LogisticRegression<RealVector> trainer;
    
    // trenowanie modelu
    trainer.train(model, train);

    // wykorzystanie modelu
    auto predictions = model(test.inputs());
    // [...]
}

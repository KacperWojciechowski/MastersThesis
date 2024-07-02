using namespace shark;

// [...]

void LRClassification(const ClassificationDataset& train,
                      const ClassificationDataset& test,
                      unsigned int num_classes)
{
    // creating the classifier object and an array of sub-classifiers
    OneVersusOneClassifier<RealVector> ovo;
    auto pairs = num_classes * (num_classes - 1) / 2;
    std::vector<LinearClassifier<RealVector>> lr(pairs);

    // iterative configuration of sub-classifiers
    for (std::size_t n = 0, cls1=1; cls1 < num_classes; ++cls1)
    {
        using BinaryClassifierType = 
            OneVersusOneClassifier<RealVector>::binary_classifier_type;
        std::vector<BinaryClassifierType*> ovo_classifiers;
        for (std::size_t cls2 = 0; cls2 < cls1; ++cls2, ++n)
        {
            // getting a binary subproblem
            ClassificationDataset binary_cls_data =
                binarySubProblem(train, cls2, cls1);

            // training the sub-model
            LogisticRegression<RealVector> trainer;
            trainer.train(lr[n], binary_cls_data);
            
            // loading sub-model into the series
            ovo_classifiers.push_back(&lr[n]);
        }
        // loading the series into the main classifier
        ovo.addClass(ovo_classifiers);
    }
    // using the model
    auto predictions = ovo(test.inputs());
    // [...]
}
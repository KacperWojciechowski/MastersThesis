using namespace shark;

void SVMClassification(const ClassificationDataset& train,
                       const ClassificationDataset& test,
                       unsigned int num_classes)
{
    double gamma = 0.5;
    GaussianRbfKernel<> kernel(gamma);
    // creating the composite model
    OneVersusOneClassifier<RealVector> ovo;
    // creating container for sub-problems
    unsigned int pairs = num_classes * (num_classes - 1) / 2;
    std::vector<KernelClassifier<RealVector>> svm(pairs);
    for (std::size_t n = 0, cls1 = 1; cls1 < num_classes; cls1++)
    {
        // creating a set of classifiers for a given class
        using BinaryClassifierType = 
            OneVersusOneClassifier<RealVector>::binary_classifier_type;
        std::vector<BinaryClassifierType*> ovo_classifiers;
        for (std::size_t cls2 = 0; cls2 < cls1; cls2++, n++)
        {
            // creating a binary sub-problem
            ClassificationDataset binary_cls_data = 
                binarySubProblem(train, cls2, cls1);
            // training the sub-model
            double c = 10.0;
            CSvmTrainer<RealVector> trainer(&kernel, c, false);
            trainer.train(svm[n], binary_cls_data);
            ovo_classifiers.push_back(&svm[n]);
        }
        // adding the set of classifiers to the main model
        ovo.addClass(ovo_classifiers);
    }
    // using the model
    auto predictions = ovo(test.inputs());
}
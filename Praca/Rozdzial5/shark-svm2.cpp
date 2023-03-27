using namespace shark;

// ...

void SVMClassification(const ClassificationDataset& train,
                       const ClassificationDataset& test,
                       unsigned int num_classes)
{
    double gamma = 0.5;
    GaussianRbfKernel<> kernel(gamma);

    // utworzenie obiektu modelu docelowego
    OneVersusOneClassifier<RealVector> ovo;

    // utworzenie kontenera na poszczególne podproblemy
    unsigned int pairs = num_classes * (num_classes - 1) / 2;
    std::vector<KernelClassifier<RealVector>> svm(pairs);

    for (std::size_t n = 0, cls1 = 1; cls1 < num_classes; cls1++)
    {
        // utworzenie zestawu klasyfikatorów podproblemów dla danej klasy
        using BinaryClassifierType = 
            OneVersusOneClassifier<RealVector>::binary_classifier_type;
        std::vector<BinaryClassifierType*> ovo_classifiers;

        for (std::size_t cls2 = 0; cls2 < cls1; cls2++, n++)
        {
            // utworzenie podproblemu binarnego
            ClassificationDataset binary_cls_data = 
                binarySubProblem(train, cls2, cls1);

            // trenowanie modelu składowego
            double c = 10.0;
            CSvmTrainer<RealVector> trainer(&kernel, c, false);
            trainer.train(svm[n], binary_cls_data);
            ovo_classifiers.push_back(&svm[n]);
        }
        // dołożenie zestawu klasyfikatorów do głównego modelu
        ovo.addClass(ovo_classifiers);
    }
    
    // użycie modelu
    auto predictions = ovo(test.inputs());
}
using namespace shark;

// [...]

void LRClassification(const ClassificationDataset& train,
                      const ClassificationDataset& test,
                      unsigned int num_classes)
{
    // utworzenie obiektu docelowego klasyfikatora oraz tablicy
    // klasyfikatorów składowych
    OneVersusOneClassifier<RealVector> ovo;
    auto pairs = num_classes * (num_classes - 1) / 2;
    std::vector<LinearClassifier<RealVector>> lr(pairs);

    // iteracyjne konfigurowanie klasyfikatorów składowych
    for (std::size_t n = 0, cls1=1; cls1 < num_classes; ++cls1)
    {
        using BinaryClassifierType = 
            OneVersusOneClassifier<RealVector>::binary_classifier_type;
        std::vector<BinaryClassifierType*> ovo_classifiers;
        for (std::size_t cls2 = 0; cls2 < cls1; ++cls2, ++n)
        {
            // pobranie binarnego podproblemu
            ClassificationDataset binary_cls_data =
                binarySubProblem(train, cls2, cls1);

            // trening modelu składowego
            LogisticRegression<RealVector> trainer;
            trainer.train(lr[n], binary_cls_data);
            
            // załadowanie modelu składowego do serii
            ovo_classifiers.push_back(&lr[n]);
        }
        // podłączenie serii do głównego klasyfikatora
        ovo.addClass(ovo_classifiers);
    }
    
    // użycie modelu
    auto predictions = ovo(test.inputs());
    // [...]
}
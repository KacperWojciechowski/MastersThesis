using namespace shogun;
// [...]
void KNNClassification(Some<CDenseFeatures<DataType>> features,
                       Some<CmulticlassLabels> labels,
                       Some<CDenseFeatures<DataType>> test_features,
                       Some<CmulticlassLabels> test_labels)
{
    // przygotowanie modelu
    std::int32_t k = 3;
    auto distance = some<CEuclideanDistance>(features, features);
    auto knn = some<CKNN>(k, distance, labels);

    // wygenerowanie predykcji
    auto new_labels = wrap(knn->apply_multiclass(test_features));

    // obliczenie dokładności
    auto eval_criterium = some<CMulticlassAccuracy>();
    auto accuracy = eval_criterium->evaluate(new_labels, test_labels);

    // przetwarzanie wyników
    auto feature_matrix = test_features->get_feature_matrix();
    for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
    {
        auto label_idx_pred = new_labels->get_label(i);
        // [...]
    }
}
using namespace shogun;
// [...]
void KNNClassification(Some<CDenseFeatures<DataType>> features,
                       Some<CmulticlassLabels> labels,
                       Some<CDenseFeatures<DataType>> test_features,
                       Some<CmulticlassLabels> test_labels)
{
    // preparing the model
    std::int32_t k = 3;
    auto distance = some<CEuclideanDistance>(features, features);
    auto knn = some<CKNN>(k, distance, labels);

    // generating predictions
    auto new_labels = wrap(knn->apply_multiclass(test_features));

    // calculating the accuracy
    auto eval_criterium = some<CMulticlassAccuracy>();
    auto accuracy = eval_criterium->evaluate(new_labels, test_labels);

    // processing the results
    auto feature_matrix = test_features->get_feature_matrix();
    for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
    {
        auto label_idx_pred = new_labels->get_label(i);
        // [...]
    }
}
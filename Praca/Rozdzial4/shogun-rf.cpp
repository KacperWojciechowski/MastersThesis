using namespace shogun;
// [...]
void RFClassification(Some<CDenseFeatures<DataType>> features,
                      Some<CRegressionLabels> labels,
                      Some<CDenseFeatures<DataType>> test_features,
                      Some<CRegressionLabels> test_labels)
{
    std::int32_t num_rand_feats = 1;
    std::int32_t num_bags = 10;

    // creating and configuring the model
    auto rand_forest = 
        some<CRandomForest>(num_rand_feats, num_bags);
    auto vote =some<CMajorityVote>();
    rand_forest->set_combination_rule(vote);
    // marking data as continuous
    SGVector<bool> feature_type(1);
    feature_type.set_const(false);
    rand_forest->set_feature_types(feature_type);

    // training
    rand_forest->set_labels(labels);
    rand_forest->set_machine_problem_type(PT_REGRESSION);
    rand_forest->train(features);

    // model evalation
    auto new_labels = wrap(rand_forest->apply_regression(test_features));
    auto eval_criterium = some<CMeanSquaredError>();
    auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
}
using namespace shogun;
// [...]
void GBMClassification(Some<CDenseFeatures<DataType>> features,
                       Some<CRegressionLabels> labels,
                       Some<CDesneFeatures<DataType>> test_features,
                       Some<CRegressionLabels> test_labels)
{
    // marking regressors as continuous
    SGVector<bool> feature_type(1);
    feature_type.set_const(false);

    // creating a base decision tree
    auto tree = some<CCARTree>(feature_type, PT_REGRESSION);
    tree->set_max_depth(3);
    // creating a loss function
    auto loss = some<CSquaredLoss>();
    constexpr int iterations = 100;
    constexpr int learning_rate = 0.1;
    constexpr int sub_set_fraction = 1.0;
    auto model = some<CStochasticGBMachine>(tree,
                                            loss, 
                                            iterations, 
                                            learning_rate, 
                                            sub_set_fraction);
    // model configuration
    model->set_labels(labels);
    model->train(features);

    // model evaluation
    auto new_labels = wrap(model->apply_regression(test_features));
    auto eval_criterium = some<CMeanSquaredError>();
    auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
}
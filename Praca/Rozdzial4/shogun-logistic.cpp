using namespace shogun;
// [...]
// training and validation data
Some<CDenseFeatures<DataType>> features;
Some<CMulticlassLabels> labels;
Some<CDenseFeatures<DataType>> test_features;
Some<CMulticlassLabels> test_labels;

// creating parameters tree
auto root = some<CModelSelectionParameters>();
// regularization coefficient
CModelSelectionParameters* z = new CModelSelectionParameters("m_z");
root->append_child(z);
z->build_values(0.2, 1.0, R_LINEAR, 0.1);

// creating the tree partition strategy
index_t k = 3;
CStatifiedCrossValidationSplitting* splitting = 
    new CStatifiedCrossValidationSplitting(labels, k);

// creating the evaluation criterion for the decision tree
auto eval_criterium = some<CMulticlassAccuracy>();
// creating the logistic regression model
auto log_reg = some<CMulticlassLogisticRegression>();
// creating a cross validation object
auto cross = some<CCrossValidation>(log_reg, features, labels, 
                                    splitting, eval_criterium);
cross->set_num_runs(1);

auto model_selection = some<CGridSearchModelSelection>(cross, root);
// selecting model parameters
CParameterCombination* best_params = 
    wrap(model_selection->select_model(false));
// applying parameters to the model
best_params->apply_to_machine(log_reg);
// printing the decision tree
best_params->print_tree();

// training
log_reg->set_labels(labels);
log_reg->train(features);

// model evaluation
auto new_labels = wrap(log_reg->apply_multiclass(test_features));
auto accuracy = eval_criterium->evaluate(new_labels, test_labels);

// results processing
auto feature_matrix = test_features->get_feature_matrix();
for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
{
    auto label_idx_pred = new_labels->get_label(i);
    auto vector = feature_matrix.get_column(i);
    // [...]
}
using namespace shogun;
// [...]
// training and validation data
Some<CDenseFeatures<DataType>> features;
Some<CMulticlassLabels> labels;
Some<CDenseFeatures<DataType>> test_features;
Some<CMulticlassLabels> test_labels;

// creating the kernel
auto kernel = some<CGaussianKernel>(features, features, 5);

// creating multiclass support vector machine based on
// one-versus-one classification
auto svm = some<CMulticlassLibSVM>();
svm->set_kernel(kernel);

// creating the parameters decision tree
auto root = some<CModelSelectionParameters>();
// missclassification avoidance parameter
CModelSelectionParameters* c = new CModelSelectionParameters("C");
root->append_child(c);
c->build_values(1.0, 1000.0, R_LINEAR, 100.);
// adding parameter selection kernel
auto params_kernel = some<CModelSelectionParameters>("kernel", kernel);
root->append_child(params_kernel);
// kernel configuration
auto params_kernel_width = 
    some<CmodelSelectionParameters>("combined_kernel_weight");
params_kernel_width->build_values(0.1, 10.0, R_LINEAR, 0.5);
params_kernel->append_child(params_kernel_width);

// preparing the tree partition strategy
index_t k = 3;
CStatifiedCrossValidationSplitting* splitting =
    new CStatifiedCrossValidationSplitting(labels, k);

// preparing the evaluation criterion
auto eval_criterium = some<CMulticlassAccuracy>();

// preparing the cross validation object
auto cross = 
    some<CCrossValidation>(svm, features, labels, splitting, eval_criterium);
cross->set_num_runs(1);

// selecting and applying parameters for the model
auto model_selection = some<CGridSearchModelSelection>(cross, root);
CParameterCombination* best_params =
    wrap(model_selection->select_model(false));
best_params->apply_to_machine(svm);
best_params->print_tree();

// training
svm->set_labels(labels);
svm->train(features);

// evaluation
auto new_labels = wrap(svm->apply_multiclass(test_features));
auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
std::cout << "svm " << name << " accuracy = " << accuracy << std::endl;

// processing results
auto feature_matrix = test_features->get_feature_matrix();
for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
{
    auto label_idx_pred = new_labels->get_label(i);
    auto vector = feature_matrix.get_column(i);
    // [...]
}
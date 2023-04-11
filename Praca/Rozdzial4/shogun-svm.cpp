using namespace shogun;
// [...]
// dane uczące i walidacyjne
Some<CDenseFeatures<DataType>> features;
Some<CMulticlassLabels> labels;
Some<CDenseFeatures<DataType>> test_features;
Some<CMulticlassLabels> test_labels;

// utworzenie jądra
auto kernel = some<CGaussianKernel>(features, features, 5);
// utworzenie wieloklasowej maszyny wektorów nośnych opartej
// o klasyfikację one-versus-one
auto svm = some<CMulticlassLibSVM>();
svm->set_kernel(kernel);

// utworzenie drzewa decyzyjnego dla parametrów
auto root = some<CModelSelectionParameters>();
// parametr określający stopień unikania missklasyfikacji
CModelSelectionParameters* c = new CModelSelectionParameters("C");
root->append_child(c);
c->build_values(1.0, 1000.0, R_LINEAR, 100.);
// dodanie jądra wyboru parametrów
auto params_kernel = some<CModelSelectionParameters>("kernel", kernel);
root->append_child(params_kernel);
// konfiuracja jądra
auto params_kernel_width = 
    some<CmodelSelectionParameters>("combined_kernel_weight");
params_kernel_width->build_values(0.1, 10.0, R_LINEAR, 0.5);
params_kernel->append_child(params_kernel_width);

// przygotowanie strategii podziału dla drzewa decyzyjnego
index_t k = 3;
CStatifiedCrossValidationSplitting* splitting =
    new CStatifiedCrossValidationSplitting(labels, k);

// przygotowanie kryterium ewaluacji modelu
auto eval_criterium = some<CMulticlassAccuracy>();

// przygotowanie obiektu sprawdzianu krzyżowego
auto cross = 
    some<CCrossValidation>(svm, features, labels, splitting, eval_criterium);
cross->set_num_runs(1);

// wybór i zaaplikowanie parametrów dla modelu
auto model_selection = some<CGridSearchModelSelection>(cross, root);
CParameterCombination* best_params =
    wrap(model_selection->select_model(false));
best_params->apply_to_machine(svm);
best_params->print_tree();

// trening
svm->set_labels(labels);
svm->train(features);

// ewaluacja modelu
auto new_labels = wrap(svm->apply_multiclass(test_features));
// obliczenie dokładności
auto accuracy = eval_criterium->evaluate(new_labels, test_labels);
std::cout << "svm " << name << " accuracy = " << accuracy << std::endl;

// przetworzenie wyników
auto feature_matrix = test_features->get_feature_matrix();
for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
{
    auto label_idx_pred = new_labels->get_label(i);
    auto vector = feature_matrix.get_column(i);
    // [...]
}
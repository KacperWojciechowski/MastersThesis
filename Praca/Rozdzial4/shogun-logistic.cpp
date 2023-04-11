using namespace shogun;
// [...]
// dane uczące i walidacyjne
Some<CDenseFeatures<DataType>> features;
Some<CMulticlassLabels> labels;
Some<CDenseFeatures<DataType>> test_features;
Some<CMulticlassLabels> test_labels;

// utworzenie drzewa parametrów
auto root = some<CModelSelectionParameters>();
// współczynnik regularyzacji
CModelSelectionParameters* z = new CModelSelectionParameters("m_z");
root->append_child(z);
z->build_values(0.2, 1.0, R_LINEAR, 0.1);

// utworzenie strategii podziału dla drzewa decyzyjnego
index_t k = 3;
CStatifiedCrossValidationSplitting* splitting = 
    new CStatifiedCrossValidationSplitting(labels, k);

// utworzenie kryterium ewaluacji dla drzewa decyzyjnego parametrów
auto eval_criterium = some<CMulticlassAccuracy>();
// utworzenie modelu regresji logistycznej
auto log_reg = some<CMulticlassLogisticRegression>();
// utworzenie obiektu sprawdzianu krzyżowego
auto cross = some<CCrossValidation>(log_reg, features, labels, 
                                    splitting, eval_criterium);
cross->set_num_runs(1);

auto model_selection = some<CGridSearchModelSelection>(cross, root);
// wybranie parametrów dla modelu
CParameterCombination* best_params = 
    wrap(model_selection->select_model(false));
// zaaplikowanie parametrów dla modelu
best_params->apply_to_machine(log_reg);
// wyświetlenie drzewa decyzyjnego
best_params->print_tree();

// trenowanie
log_reg->set_labels(labels);
log_reg->train(features);

// ewaluacja modelu dla danych testowych
auto new_labels = wrap(log_reg->apply_multiclass(test_features));

// ocena dokładności
auto accuracy = eval_criterium->evaluate(new_labels, test_labels);

// przetworzenie wyników
auto feature_matrix = test_features->get_feature_matrix();
for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
{
    auto label_idx_pred = new_labels->get_label(i);
    auto vector = feature_matrix.get_column(i);
    // [...]
}
#pragma once

inline void shogunCrossValidLogistic(
    shogun::Some<shogun::CDenseFeatures<float64_t>> inputs,
    shogun::Some<shogun::CMulticlassLabels> outputs)
{
    using namespace shogun;

    // podział danych na testowe i uczące
    auto testSamples = static_cast<int>(0.8*inputs.num_cols());
    auto trainInputs = inputs.submatrix(0, testSamples).clone();
    auto trainOutputs = outputs.submatrix(0, testSamples).clone();
    auto testInputs =
        inputs.submatrix(testSamples, inputs.num_cols()).clone();
    auto testOutputs =
        outputs.submatrix(testSamples, outputs.num_cols()).clone();
    // utworzenie drzewa parametrów
    auto root = some<CModelSelectionParameters>();
    // współczynnik regularyzacji
    CModelSelectionParameters* z = new CModelSelectionParameters("m_z");
    root->append_child(z);
    z->build_values(0.2, 1.0, R_LINEAR, 0.1);
    // utworzenie strategii podziału drzewa decyzyjnego
    index_t k = 3;
    CStatifiedCrossValidationSplitting* splitting =
        new CStatifiedCrossValidationSplitting(labels, k);
    // utworzenie kryterium ewaluacji dla drzewa decyzyjnego
    auto evalCriterium = some<CMulticlassAccuracy>();
    // utworzenie modelu regresji logistycznej
    auto logReg = some<CMulticlassLogisticRegression>();
    // utworzenie obiektu sprawdzianu krzyżowego
    auto cross = some<CCrossValidation>(logReg, trainInputs, trainOutputs,
                                        splitting, evalCriterium);
    cross->set_num_runs(1);
    auto modelSelection = some<CGridSearchModelSelection>(cross, root);
    // wybranie parametrów dla modelu
    CParameterCombination* bestParams = 
        wrap(modelSelection->select_model(false));
    // zaaplikowanie parametrów dla modelu
    bestParams->apply_to_machine(logReg);
    // wyświetlenie drzewa decyzyjnego
    bestParams->print_tree();
    // trenowanie docelowego modelu
    logReg->set_labels(trainOutputs);
    logReg->train(trainInputs);

    // ewaluacja modelu
    std::cout << "----- Shogun CV Logistic -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(logReg->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs, Task::CLASSIFICATION);

    std::cout << "Test:" << std::endl;
    prediction = wrap(logReg->apply_multiclass(testInputs));
    shogunVerifyModel(prediction, testOutputs, Task::CLASSIFICATION);

    delete splitting;    
}
#pragma once

inline void shogunCrossValidLogistic(
    shogun::Some<shogun::CDenseFeatures<float64_t>>& trainInputs,
    shogun::Some<shogun::CDenseFeatures<float64_t>>& testInputs
    shogun::Some<shogun::CMulticlassLabels> trainOutputs,
    shogun::Some<shogun::CMulticlassLabels> testOutputs)
{
    using namespace shogun;

    // creating the parameter tree
    auto root = some<CModelSelectionParameters>();
    // regularization coefficient
    CModelSelectionParameters* z = new CModelSelectionParameters("m_z");
    root->append_child(z);
    z->build_values(0.2, 1.0, R_LINEAR, 0.1);
    // creating the decision tree partition strategy
    index_t k = 3;
    CStatifiedCrossValidationSplitting* splitting =
        new CStatifiedCrossValidationSplitting(labels, k);
    // creating the evaluation criterion
    auto evalCriterium = some<CMulticlassAccuracy>();
    // creating the logistic regression model
    auto logReg = some<CMulticlassLogisticRegression>();
    // creating the cross-validation object
    auto cross = some<CCrossValidation>(logReg, trainInputs, trainOutputs,
                                        splitting, evalCriterium);
    cross->set_num_runs(1);
    auto modelSelection = some<CGridSearchModelSelection>(cross, root);
    // selecting model parameters
    CParameterCombination* bestParams = 
        wrap(modelSelection->select_model(false));
    // applying parameters to the model
    bestParams->apply_to_machine(logReg);
    // printing the decision tree
    bestParams->print_tree();

    // training
    logReg->set_labels(trainOutputs);
    logReg->train(trainInputs);

    // evaluation
    std::cout << "----- Shogun CV Logistic -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(logReg->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs);

    std::cout << "Test:" << std::endl;
    auto prediction2 = wrap(logReg->apply_multiclass(testInputs));
    shogunVerifyModel(prediction2, testOutputs);

    delete splitting;
}

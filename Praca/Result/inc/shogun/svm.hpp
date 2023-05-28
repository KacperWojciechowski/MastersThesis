#pragma once

#include <inc/shogun/verify.hpp>

#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/ModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ParameterCombination.h>

inline void shogunSVM(
		shogun::Some<shogun::CDenseFeatures<float64_t>>& trainInputs,
                shogun::Some<shogun::CDenseFeatures<float64_t>>& testInputs,
                shogun::Some<shogun::CMulticlassLabels>& trainOutputs,
                shogun::Some<shogun::CMulticlassLabels>& testOutputs)
{
    using namespace shogun;

    // utworzenie jądra
    auto kernel = some<CGaussianKernel>();
    kernel->init(trainInputs, trainInputs);
    // utworzenie i konfiguracja modelu
    auto svm = some<CMulticlassLibSVM>(LIBSVM_C_SVC);
    svm->set_kernel(kernel);

    // poszukiwanie hiperparametrów
    auto root = some<CModelSelectionParameters>();
    // stopień unikania missklasyfikacji
    CModelSelectionParameters* c = new CModelSelectionParameters("C");
    root->append_child(c);
    c->build_values(1.0, 1000.0, R_LINEAR, 100.);
    // utworzenie wskaźnika na jądro
    auto paramsKernel = some<CModelSelectionParameters>("kernel", kernel);
    root->append_child(paramsKernel);
    // utworzenie wskaźnika na wagi
    auto paramsKernelWidth = 
	    some<CModelSelectionParameters>("combined_kernel_weight");
    paramsKernelWidth->build_values(0.1, 10.0, R_LINEAR, 0.5);
    paramsKernel->append_child(paramsKernelWidth);
    // utworzenie podziału do sprawdzianu krzyżowego
    index_t k = 3;
    CStratifiedCrossValidationSplitting* splitting =
	    new CStratifiedCrossValidationSplitting(trainOutputs, k);
    // utworzenie kryterium trafności
    auto evalCriterium = some<CMulticlassAccuracy>();
    // utworzenie obiektu sprawdzianu krzyżowego
    auto cross =
	    some<CCrossValidation>(
		svm, trainInputs, trainOutputs, splitting, evalCriterium);
    cross->set_num_runs(1);
    // utworzenie obiektu selekcji modelu
    auto modelSelection = some<CGridSearchModelSelection>(cross, root);
    // wybór i zaaplikowanie parametrów
    CParameterCombination* bestParams = 
	    wrap(modelSelection->select_model(false));
    bestParams->apply_to_machine(svm);
    bestParams->print_tree();
 

    // trening
    svm->set_labels(trainOutputs);
    std::cout << "t" << std::endl;
    svm->train(trainInputs);
    std::cout << "u" << std::endl;

    // ewaluacja modelu
    std::cout << "----- Shogun SVM -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(svm->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs);

    std::cout << "Test:" << std::endl;
    auto prediction2 = wrap(svm->apply_multiclass(testInputs));
    shogunVerifyModel(prediction2, testOutputs);
}

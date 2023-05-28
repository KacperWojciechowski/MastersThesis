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

inline void shogunSVM(shogun::Some<shogun::CDenseFeatures<float64_t>>& trainInputs,
                      shogun::Some<shogun::CDenseFeatures<float64_t>>& testInputs,
                      shogun::Some<shogun::CMulticlassLabels>& trainOutputs,
                      shogun::Some<shogun::CMulticlassLabels>& testOutputs)
{
    using namespace shogun;

    std::cout << "a" << std::endl;
    // utworzenie jądra
    auto kernel = some<CGaussianKernel>();
    std::cout << "b" << std::endl;
    kernel->init(trainInputs, trainInputs);
    std::cout << "c" << std::endl;
    // utworzenie i konfiguracja modelu
    auto svm = some<CMulticlassLibSVM>(LIBSVM_C_SVC);
    svm->set_kernel(kernel);
    std::cout << "d" << std::endl;

    // poszukiwanie hiperparametrów
    auto root = some<CModelSelectionParameters>();
    // stopień unikania missklasyfikacji
    CModelSelectionParameters* c = new CModelSelectionParameters("C");
    std::cout << "e" << std::endl;
    root->append_child(c);
    std::cout << "f" << std::endl;
    c->build_values(1.0, 1000.0, R_LINEAR, 100.);
    std::cout << "g" << std::endl;
    auto paramsKernel = some<CModelSelectionParameters>("kernel", kernel);
    std::cout << "h" << std::endl;
    root->append_child(paramsKernel);
    std::cout << "i" << std::endl;
    auto paramsKernelWidth = some<CModelSelectionParameters>("combined_kernel_weight");
    std::cout << "j" << std::endl;
    paramsKernelWidth->build_values(0.1, 10.0, R_LINEAR, 0.5);
    std::cout << "k" << std::endl;
    paramsKernel->append_child(paramsKernelWidth);
    index_t k = 3;
    std::cout << "l" << std::endl;
    CStratifiedCrossValidationSplitting* splitting =
	    new CStratifiedCrossValidationSplitting(trainOutputs, k);
    std::cout << "m" << std::endl;
    auto evalCriterium = some<CMulticlassAccuracy>();
    auto cross =
	    some<CCrossValidation>(svm, trainInputs, trainOutputs, splitting, evalCriterium);
    std::cout << "n" << std::endl;
    cross->set_num_runs(1);
    std::cout << "o" << std::endl;
    auto modelSelection = some<CGridSearchModelSelection>(cross, root);
    std::cout << "p" << std::endl;
    CParameterCombination* bestParams = wrap(modelSelection->select_model(false));
    std::cout << "q" << std::endl;
    bestParams->apply_to_machine(svm);
    std::cout << "r" << std::endl;
    bestParams->print_tree();
    std::cout << "s" << std::endl;

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

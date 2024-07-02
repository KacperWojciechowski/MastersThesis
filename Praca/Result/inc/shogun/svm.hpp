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

    // creating kernel
    auto kernel = some<CGaussianKernel>();
    kernel->init(trainInputs, trainInputs);
    // creating and configuring the model
    auto svm = some<CMulticlassLibSVM>(LIBSVM_C_SVC);
    svm->set_kernel(kernel);

    // searching for hyperparameters
    auto root = some<CModelSelectionParameters>();
    // missclasification avoidance degree
    CModelSelectionParameters* c = new CModelSelectionParameters("C");
    root->append_child(c);
    c->build_values(1.0, 1000.0, R_LINEAR, 100.);
    // creating the kernel pointer
    auto paramsKernel = some<CModelSelectionParameters>("kernel", kernel);
    root->append_child(paramsKernel);
    // creating the weights pointer
    auto paramsKernelWidth = 
	    some<CModelSelectionParameters>("combined_kernel_weight");
    paramsKernelWidth->build_values(0.1, 10.0, R_LINEAR, 0.5);
    paramsKernel->append_child(paramsKernelWidth);
    // creating the partition for cross validation
    index_t k = 3;
    CStratifiedCrossValidationSplitting* splitting =
	    new CStratifiedCrossValidationSplitting(trainOutputs, k);
    // creating the accuracy criterion
    auto evalCriterium = some<CMulticlassAccuracy>();
    // creating the cross-validation object
    auto cross =
	    some<CCrossValidation>(
		svm, trainInputs, trainOutputs, splitting, evalCriterium);
    cross->set_num_runs(1);
    // creating model selection object
    auto modelSelection = some<CGridSearchModelSelection>(cross, root);
    // selecting and applying the parameters
    CParameterCombination* bestParams = 
	    wrap(modelSelection->select_model(false));
    bestParams->apply_to_machine(svm);
    bestParams->print_tree();
 
    // training
    svm->set_labels(trainOutputs);
    svm->train(trainInputs);

    // model evaluation
    std::cout << "----- Shogun SVM -----" << std::endl;
    std::cout << "Train:" << std::endl;
    auto prediction = wrap(svm->apply_multiclass(trainInputs));
    shogunVerifyModel(prediction, trainOutputs);

    std::cout << "Test:" << std::endl;
    auto prediction2 = wrap(svm->apply_multiclass(testInputs));
    shogunVerifyModel(prediction2, testOutputs);
}

#pragma once

#include <iostream>

#define SHARK_CV_VERBOSE 1
#include <inc/shark/printEvaluation.hpp>
#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Classifier.h>
#include <iostream>
#include <shark/Models/OneVersusOneClassifier.h>

inline void sharkLogistic(const shark::ClassificationDataset& trainData,
                          const shark::ClassificationDataset& testData)
{
    using namespace shark;

    auto num_classes = 2;
    OneVersusOneClassifier<RealVector> ovo;
    unsigned int pairs = num_classes * (num_classes - 1) / 2;
    std::vector<LinearClassifier<RealVector> > lr(pairs);
    //for (std::size_t n = 0, cls1 = 1; cls1 < num_classes; cls1++) {
    std::vector<OneVersusOneClassifier<RealVector>::binary_classifier_type*>
        ovo_classifiers;
    //for (std::size_t cls2 = 0; cls2 < cls1; cls2++, n++) {
      // get the binary subproblem
      //ClassificationDataset binary_cls_data =
      //    binarySubProblem(trainData, cls2, cls1);

      // train the binary machine
      LogisticRegression<RealVector> trainer;
      trainer.train(lr[0], trainData);
      ovo_classifiers.push_back(&lr[0]);
    //}
    ovo.addClass(ovo_classifiers);
  //}

    // ewaluacja
    std::cout << "-----Shark Logistic Regression-----" << std::endl;
    std::cout << "Train data model evaluation:" << std::endl;
    auto predictions = ovo(trainData.inputs());
    printSharkModelEvaluation(
        trainData.labels(), predictions);

    std::cout << "Test data model evaluation:" << std::endl;
    predictions = ovo(testData.inputs());
    printSharkModelEvaluation(
        testData.labels(), predictions);
}

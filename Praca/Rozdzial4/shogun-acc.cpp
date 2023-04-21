using namespace shogun;
// [...]
auto acc_measure = some<CMulticlassAccuracy>();
auto acc = acc_measure->evaluate(predictions, train_labels);
auto confusionMatrix = acc_measure->get_confusion_matrix(
    predictions, train_labels);
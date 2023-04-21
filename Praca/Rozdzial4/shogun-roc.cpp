using namespace shogun;
// [...]
auto roc = some<CROCEvaluation>();
roc->evaluate(predictions, targets);
std::cout << "AUC ROC = " << roc->get_auROC() << std::endl;
    
using namespace std;

// [...]

constexpr bool invertToPositiveROC = true;
NegativeAUC<> roc(invertToPositiveROC);
auto auc_roc = roc(train_data.labels(), predictions);
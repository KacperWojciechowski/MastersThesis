using namespace shogun;
// [...]
auto logLoss = some<CLogLoss>();
auto sqareGradient = logLoss->get_square_grad(prediction, label);
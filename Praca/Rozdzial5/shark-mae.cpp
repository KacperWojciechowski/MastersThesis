using namespace shark;

// [...]

AbsoluteLoss<> abs_loss;
auto mae = abs_loss(train_data.labels(), prediction);
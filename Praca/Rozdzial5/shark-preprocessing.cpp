// [...]

// shuffling data and extracting validation dataset
train_data.shuffle();
auto test_data = shark::splitAtElement(train_data, 120);

// creating normalizer
using Trainer = shark::NormalizeComponentsUnitVariance<shark::RealVector>;
bool remove_mean = true;
shark::Normalizer<shark::RealVector> normalizer;
Trainer normalizing_trainer(remove_mean);

// giving the average and variance of the training data to the normalizer
normalizing_trainer.train(normalizer, train_data.inputs());

// training data transformation
train_data = shark::transformInputs(train_data, normalizer);
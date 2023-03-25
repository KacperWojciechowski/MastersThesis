// [...]

// przemieszanie danych i wyznaczenie danych testowych
train_data.shuffle();
auto test_data = shark::splitAtElement(train_data, 120);

// utworzenie normalizera
using Trainer = shark::NormalizeComponentsUnitVariance<shark::RealVector>;
bool remove_mean = true;
shark::Normalizer<shark::RealVector> normalizer;
Trainer normalizing_trainer(remove_mean);

// nauczenie normalizera średniej i wariancji danych treningowych
normalizing_trainer.train(normalizer, train_data.inputs());

// transformacja danych uczących
train_data = shark::transformInputs(train_data, normalizer);
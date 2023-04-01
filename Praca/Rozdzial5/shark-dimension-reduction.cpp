// [...]

// utworzenie trenera PCA
shark::PCA pca(data);
shark::LinearModel<> enc;

// konfiguracja enkodera do redukcji wymiarów
constexpr int nbOfDim = 2;
pca.encoder(enc, nbOfDim);
auto encoded_data = enc(data);
// [...]

// creating PCA kernel
shark::PCA pca(data);
shark::LinearModel<> enc;

// configuring the encoder for dimensionality reduction
constexpr int nbOfDim = 2;
pca.encoder(enc, nbOfDim);
auto encoded_data = enc(data);
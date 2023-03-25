// [...]

// utworzenie trenera PCA
shark::PCA pca(data.inputs());
shark::LinearModel<> enc;

// konfiguracja enkodera do redukcji wymiarów
pca.encoder(enc, 2);
shark::Data<shark::RealVector> encoded_data = enc(data.inputs());
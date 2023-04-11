using namespace shogun;
// [...]
void KernelPCAReduction(Some<CDenseFeatures<DataType>> features,
                        const int target_dim)
{
    // utworzenie jądra dla metody kernel PCA
    auto gauss_kernel = some<CGaussianKernel>(features, features, 0.5);
    // utworzenie reduktora
    auto pca = some<CKernelPCA>();
    // konfiguracja reduktora
    pca->set_kernel(gauss_kernel.get());
    pca->set_target_dim(target_dim);
    // nauczenie reduktora
    pca->fit(features);

    // przetworzenie danych
    auto feature_matrix = features->get_feature_matrix();
    for (index_t i = 0; i < features->get_num_vectors(); ++i)
    {
        // utworzenie przetworzonego wektora
        auto vector = feature_matrix.get_colimn(i);
        auto new_vector = pca->apply_to_feature_vector(vector);
    }
}
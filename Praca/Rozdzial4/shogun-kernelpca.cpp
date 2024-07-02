using namespace shogun;
// [...]
void KernelPCAReduction(Some<CDenseFeatures<DataType>> features,
                        const int target_dim)
{
    // creation of kernel for the kernel PCA method
    auto gauss_kernel = some<CGaussianKernel>(features, features, 0.5);
    // creating reductor
    auto pca = some<CKernelPCA>();
    // configuring the reductor
    pca->set_kernel(gauss_kernel.get());
    pca->set_target_dim(target_dim);
    // training the reductor
    pca->fit(features);

    // data processing
    auto feature_matrix = features->get_feature_matrix();
    for (index_t i = 0; i < features->get_num_vectors(); ++i)
    {
        // creating processed data vectors
        auto vector = feature_matrix.get_colimn(i);
        auto new_vector = pca->apply_to_feature_vector(vector);
    }
}
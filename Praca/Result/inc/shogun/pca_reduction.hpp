#pragma once

inline void KernelPCA(
    shogun::Some<shogun::CDenseFeatures<float64_t>> inputs,
    const int target_dim)
{
    using namespace shogun;

    // creating the kernel
    auto gaussKernel = some<CGaussianKernel>(inputs, inputs, 0.5);
    // creating the reductor
    auto pca = some<CKernelPCA>();
    // configuring the reductor
    pca->set_kernel(gaussKernel.get());
    pca->set_target_dim(target_dim);
    // training
    pca->fit(inputs);
    // dimensionality reduction
    auto featureMatrix = inputs->get_feature_matrix();
    for (index_t i = 0; i < inputs->get_num_vectors(); ++i)
    {
        auto vector = featureMatrix.get_column(i);
        auto newVector = pca->apply_to_feature_vector(vector);
    }
}

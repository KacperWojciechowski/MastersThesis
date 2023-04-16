#pragma once

inline void KernelPCA(shogun::Some<CDenseFeatures<float64_t>> inputs,
                      const int target_dim)
{
    using namespace shogun;

    // utworzenie jądra
    auto gaussKernel = some<CGaussianKernel>(inputs, inputs, 0.5);
    // utworzenie obiektu reduktora wymiarowości
    auto pca = some<CKernelPCA>();
    // konfiguracja reduktora
    pca->set_kernel(gaussKernel.get());
    pca->set_target_dim(target_dim);
    // nauczenie reduktora
    pca->fit(inputs);
    // zastosowanie redukcji
    auto featureMatrix = inputs->get_feature_matrix();
    for (index_t i = 0; i < inputs->get_num_vectors(); ++i)
    {
        auto vector = featureMatrix.get_column(i);
        auto newVector = pca->apply_to_feature_vector(vector);
    }
}
#pragma once

inline auto sharkPreprocess(auto& trainData)
{
    using namespace shark;

    // shuffling the data
    trainData.shuffle();
    // creating the normalizer
    using Trainer = NormalizeComponentsUnitVariance<RealVector>;
    bool removeMean = true;
    Normalizer<RealVector> normalizer;
    Trainer normalizingTrainer(removeMean);
    // teaching the normalizer the mean and variance of the training data
    normalizingTrainer.train(normalizer, trainData.inputs());
    // transforming the training data
    return transformInputs(trainData, normalizer);
}
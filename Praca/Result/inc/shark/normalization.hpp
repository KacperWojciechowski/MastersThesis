#pragma once

inline auto sharkPreprocess(auto& trainData)
{
    using namespace shark;

    // przemieszanie danych
    trainData.shuffle();
    // utworzenie normalizera
    using Trainer = NormalizeComponentsUnitVariance<RealVector>;
    bool removeMean = true;
    Normalizer<RealVector> normalizer;
    Trainer normalizingTrainer(removeMean);
    // nauczenie normalizera średniej i wariancji danych treningowych
    normalizingTrainer.train(normalizer, trainData.inputs());
    // transformacja danych uczących
    return transformInputs(trainData, normalizer);
}
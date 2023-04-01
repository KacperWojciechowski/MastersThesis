using namespace shark;

// [...]

void LDAReduction(const UnlabeledData<RealVector>& data,
                  const UnlabeledData<RealVector>& labels,
                  std::size_t target_dim)
{
    // utworzenie obiektów LDA
    LinearClassifier<> encoder;
    LDA lda;

    // utworzenie zestawu danych
    LabeledData<RealVector, unsigned int> dataset(
        labels.numberOfElements(), InputLabelPair<RealVector, unsigned int>(
            RealVector(data.element(0).size()), 0));
    
    // wypełnienie zbioru danymi
    for (std::size_t i = 0; i < labels.numberOfElements(); ++i)
    {
        // zmiana indeksów klas aby zaczynały się od 0
        dataset.element(i).label =
            static_cast<unsigned int>(labels.element(i)[0]) - 1;
        dataset.element[i].input = data.element(i)
    }

    // trening enkodera
    lda.train(encoder, dataset);

    // utworzenie zredukowanego zestawu danych
    auto new_labels = encoder(data);
    auto new_data = encoder.decisionFunction()(data);
}
using namespace shark;

void LDAReduction(const UnlabeledData<RealVector>& data,
                  const UnlabeledData<RealVector>& labels,
                  std::size_t target_dim)
{
    // creating LDA object and encoder
    LinearClassifier<> encoder;
    LDA lda;
    // creating dataset
    LabeledData<RealVector, unsigned int> dataset(
        labels.numberOfElements(), InputLabelPair<RealVector, unsigned int>(
            RealVector(data.element(0).size()), 0));
    // populating the dataset
    for (std::size_t i = 0; i < labels.numberOfElements(); ++i)
    {
        // changing the classes indexes to start from 0
        dataset.element(i).label =
            static_cast<unsigned int>(labels.element(i)[0]) - 1;
        dataset.element[i].input = data.element(i)
    }
    // encoder training
    lda.train(encoder, dataset);
    // creating reduced dataset
    auto new_labels = encoder(data);
    auto new_data = encoder.decisionFunction()(data);
}
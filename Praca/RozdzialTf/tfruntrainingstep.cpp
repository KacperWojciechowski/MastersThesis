void RunTrainStep(const std::vector<float>& input_batch,
                    const std::vector<float>& target_batch) {
    TF_CHECK_OK(session_->Run({{"input", MakeTensor(input_batch)},
                               {"target", MakeTensor(target_batch)}},
                              {}, {"train"}, nullptr));
}
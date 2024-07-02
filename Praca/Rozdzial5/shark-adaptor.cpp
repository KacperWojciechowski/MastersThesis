// sample data contained within the C++ STL vector class
std::vector<float> data{1, 2, 3, 4};

// wrapping data into a 2x2 matrix
auto m = remora::dense_matrix_adaptor<float>(data.data(), 2, 2);

// wrapping data into a 1x4 vector
auto v = remora::dense_vector_adaptor<float>(data.data(), 4);
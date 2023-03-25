// przykładowe dane zawarte w kontenerze std::vector biblioteki 
// standardowej C++
std::vector<float> data{1, 2, 3, 4};

// opakowanie danych do postaci macierzy 2 x 2
auto m = remora::dense_matrix_adaptor<float>(data.data(), 2, 2);

// opakowanie danych do postaci wektora 1 x 4
auto v = remora::dense_vector_adaptor<float>(data.data(), 4);
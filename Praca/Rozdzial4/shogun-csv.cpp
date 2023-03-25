#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/io/File.h>

using namespace shogun;
using Matrix = shogun::SGMatrix<float64_t>

// [...]
// odczyt danych z pliku .csv
auto csv_file = shogun::some<shogun::CCSVFile>("sample_file.csv");
Matrix data;
data.load(vsv_file);

// transpozycja i rozdzielenie danych
Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);
Matrix inputs = data.submatrix(0, data.num_cols - 1); // utworzenie widoku
inputs = inputs.clone(); // przekopiowanie danych

// utworzenie widoku
Matrix outputs = data.submatrix(data.num_cols - 1, data.num_cols); 
outputs = outputs.clone(); // przekopiowanie danych

// powrotna transpozycja macierzy danych treningowych dla algorytmów
// biblioteki
Matrix::transpose_matrix(inputs.matrix, inputs.num_rows, inputs.num_cols);

// przygotowanie klas wrapujących do wykorzystania przy uczeniu
auto features = shogun::some<shogun::CDenseFeatures<float64_t>>(inputs);
auto labels = 
    shogun::wrap(new shogun::CMulticlassLabels(outputs.get_column(0)));
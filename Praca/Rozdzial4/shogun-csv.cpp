#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/io/File.h>

using namespace shogun;
using Matrix = shogun::SGMatrix<float64_t>

// [...]
// reading data from .csv file
auto csv_file = shogun::some<shogun::CCSVFile>("sample_file.csv");
Matrix data;
data.load(vsv_file);

// transposing and splitting the data
Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);
// creating a view of the predictions
Matrix inputs = data.submatrix(0, data.num_cols - 1);
inputs = inputs.clone(); // copying data

// creating a view of the labels
Matrix outputs = data.submatrix(data.num_cols - 1, data.num_cols); 
outputs = outputs.clone(); // copying data

// transposing the matrix back for the library algorithms
Matrix::transpose_matrix(inputs.matrix, inputs.num_rows, inputs.num_cols);

// creating wrappers for the models
auto features = shogun::some<shogun::CDenseFeatures<float64_t>>(inputs);
auto labels = 
    shogun::wrap(new shogun::CMulticlassLabels(outputs.get_column(0)));
#pragma once

#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/io/File.h>
#include <shogun/io/CSVFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <iostream>

// helper intermediate packaging for data set
struct Dataset
{
    shogun::SGMatrix<float64_t> trainInputs;
    shogun::SGMatrix<float64_t> testInputs;
    shogun::SGMatrix<float64_t> trainOutputs;
    shogun::SGMatrix<float64_t> testOutputs;
};

// helper struct to distinguish label position
enum class LabelPos
{
    FIRST,
    LAST
};

inline Dataset readShogunCsvData(
		std::string filename, LabelPos labelPos)
{
    using namespace shogun;
    using Matrix = SGMatrix<float64_t>;

    Dataset ret;

    // reading raw csv file content and parsing it into matrix
    auto csvFile = some<CCSVFile>(filename.c_str());
    Matrix data;
    data.load(csvFile);
    // transposing to human-friendly form (column-wise)
    Matrix::transpose_matrix(data.matrix, data.num_rows, 
		             data.num_cols);
    // partitioning the matrix into regressors and response variable
    switch(labelPos)
    {
        case LabelPos::FIRST:
            ret.trainInputs = data.submatrix(1, data.num_cols)
		    .clone();
            ret.trainOutputs = data.submatrix(0, 1).clone();
            break;
        case LabelPos::LAST:
            ret.trainInputs = data.submatrix(0, data.num_cols - 1)
		    .clone();
            ret.trainOutputs =
                data.submatrix(data.num_cols - 1, data.num_cols)
		.clone();
            break;
    };

    // transposing back to library-friendly form (row-wise)
    Matrix::transpose_matrix(ret.trainInputs.matrix, 
		             ret.trainInputs.num_rows,
                             ret.trainInputs.num_cols);
    Matrix::transpose_matrix(ret.trainOutputs.matrix, 
		             ret.trainOutputs.num_rows,
                             ret.trainOutputs.num_cols);
    // splitting data into training and validation
    auto temp = ret.testInputs = ret.trainInputs.submatrix(
        static_cast<long>(0.8 * ret.trainInputs.num_cols), 
	ret.trainInputs.num_cols).clone();
    ret.testInputs = std::move(temp);
    auto temp2 = ret.trainInputs.submatrix(
        0, static_cast<long>(0.8 * ret.trainInputs.num_cols))
	    .clone();
    ret.trainInputs = std::move(temp2);
    auto temp3 = ret.trainOutputs.submatrix(
        static_cast<long>(0.8 * ret.trainOutputs.num_cols), 
	ret.trainOutputs.num_cols).clone();
    ret.testOutputs = std::move(temp3);
    auto temp4 = ret.trainOutputs.submatrix(
        0, static_cast<long>(0.8 * ret.trainOutputs.num_cols))
	    .clone();
    ret.trainOutputs = std::move(temp4);
    return ret;
}

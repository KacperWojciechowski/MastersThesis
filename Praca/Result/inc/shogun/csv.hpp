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

// pomocnicze pośrednie opakowanie na zestaw danych
struct Dataset
{
    shogun::SGMatrix<float64_t> trainInputs;
    shogun::SGMatrix<float64_t> testInputs;
    shogun::SGMatrix<float64_t> trainOutputs;
    shogun::SGMatrix<float64_t> testOutputs;
};

// pomocnicza struktura określająca pozycję zmiennej odpowiedzi
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

    // odczytanie surowej zawartości pliku csv i sparsowanie 
    // jej do macierzy
    auto csvFile = some<CCSVFile>(filename.c_str());
    Matrix data;
    data.load(csvFile);
    // transpozycja do postaci docelowej dla człowieka
    // (działanie na kolumnach)
    Matrix::transpose_matrix(data.matrix, data.num_rows, 
		             data.num_cols);
    // podział macierzy na część regresorów i zmiennej odpowiedzi
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
    // ponowna transpozycja do positaci docelowej dla algorytmów
    // uczących (operowanie na wierszach)
    Matrix::transpose_matrix(ret.trainInputs.matrix, 
		             ret.trainInputs.num_rows,
                             ret.trainInputs.num_cols);
    Matrix::transpose_matrix(ret.trainOutputs.matrix, 
		             ret.trainOutputs.num_rows,
                             ret.trainOutputs.num_cols);
    // podział danych na część treningową i testową
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

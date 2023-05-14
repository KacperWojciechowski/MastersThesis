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

inline Dataset readShogunCsvData(std::string filename, LabelPos labelPos)
{
    using namespace shogun;
    using Matrix = SGMatrix<float64_t>;

    Dataset ret;

    // odczytanie surowej zawartości pliku csv i sparsowanie jej do
    // macierzy
    auto csvFile = some<CCSVFile>(filename.c_str());
    Matrix data;
    data.load(csvFile);
    // transpozycja do postaci docelowej dla człowieka
    // (działanie na kolumnach)
    Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);
    // podział macierzy na część regresorów i zmiennej odpowiedzi
    switch(labelPos)
    {
        case LabelPos::FIRST:
            ret.trainInputs = data.submatrix(1, data.num_cols).clone();
            ret.trainOutputs = data.submatrix(0, 1).clone();
            break;
        case LabelPos::LAST:
            ret.trainInputs = data.submatrix(0, data.num_cols - 1).clone();
            ret.trainOutputs =
                data.submatrix(data.num_cols - 1, data.num_cols).clone();
            break;
    };
    // ponowna transpozycja do positaci docelowej dla algorytmów uczących
    // (operowanie na wierszach)
    Matrix::transpose_matrix(ret.trainInputs.matrix, ret.trainInputs.num_rows,
                             ret.trainInputs.num_cols);
    // podział danych na część treningową i testową
    std::cout << "Test 1\n";
    auto temp = ret.testInputs = ret.trainInputs.submatrix(
        static_cast<long>(0.8 * ret.trainInputs.num_cols), ret.trainInputs.num_cols).clone();
    std::cout << "Test 2\n";
    ret.testInputs = std::move(temp);
    std::cout << "Test 3\n";
    ret.trainInputs = ret.trainInputs.submatrix(
        0, static_cast<long>(0.8 * ret.trainInputs.num_cols)).clone();
    ret.testOutputs = ret.trainOutputs.submatrix(
        static_cast<long>(0.8 * ret.trainOutputs.num_cols), ret.trainInputs.num_cols).clone();
    ret.trainOutputs = ret.trainOutputs.submatrix(
        0, static_cast<long>(0.8 * ret.trainOutputs.num_cols)).clone();
    return ret;
}

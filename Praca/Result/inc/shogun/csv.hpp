#pragma once

#include <shogun/src/shogun/base/init.h>
#include <shogun/src/shogun/base/some.h>
#include <shogun/src/shogun/io/File.h>

// pomocnicze pośrednie opakowanie na zestaw danych
struct Dataset
{
    shogun::SGMatrix<float64_t> inputs;
    shogun::SGMatrix<float64_t> outputs;
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
    auto csvFile = some<CCSVFile>(filename);
    Matrix data;
    data.load(csvFile);
    // transpozycja do postaci docelowej dla człowieka 
    // (działanie na kolumnach)
    Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols);
    // podział macierzy na część regresorów i zmiennej odpowiedzi
    switch(labelPos)
    {
        case FIRST:
            ret.inputs = data.submatrix(1, data.num_cols).clone();
            ret.outputs = data.submatrix(0, 1).clone();
            break;
        case LAST:
            ret.inputs = data.submatrix(0, data.num_cols - 1).clone();
            ret.outputs = 
                data.submatrix(data.num_cols - 1, data.num_cols).clone();
            break;
    };
    // ponowna transpozycja do positaci docelowej dla algorytmów uczących
    // (operowanie na wierszach)
    Matrix::transpose_matrix(ret.inputs.matrix, ret.inputs.num_rows,
                             ret.inputs.num_cols);
    return ret;
}
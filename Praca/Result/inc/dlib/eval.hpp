#pragma once

#include <cmath>
#include <dlib/statistics.h>

enum class Task
{
    CLASSIFICATION,
    REGRESSION
};

inline void dlibEval(std::vector<double> predictions,
                     std::vector<double> labels,
                     Task task)
{
    using namespace dlib;

    if (task == Task::CLASSIFICATION)
    {
        // przygotowanie kontenerów na podzielone dane
        std::vector<double> correct;
        std::vector<double> incorrect;
        // przygotowanie wartości detektora
        constexpr double positiveDetectionScore = 0.75;
        constexpr double negativeDetectionScore = 0.25;
        // podział danych
        for (int i = 0; i < predictions.size() && i < labels.size(); ++i)
        {
            if (predictions[i] == labels[i])
                correct.emplace_back(positiveDetectionScore);
            else
                incorrect.emplace_back(negativeDetectionScore);
        }
        // obliczenie krzywej roc
        auto roc = compute_roc_curve(correct, incorrect);
        // obliczenie pola pod wykresem roc
        double aucRoc = 0.0;
        for (int i = 0; i < roc.size() - 1; i++)
        {
	    if (roc[i+1].false_positive_rate != 0.0)
	    {
		aucRoc += (roc[i].true_positive_rate + roc[i+1].true_positive_rate) 
			* (roc[i+1].false_positive_rate - roc[i].false_positive_rate) / 2;
	    }
        }
        std::cout << "AUC ROC: " << aucRoc << std::endl; 
    }
    else
    {
        // obliczenie sumy kwadratów różnic
        auto sum = 0.0;
        for (int i = 0; i < predictions.size() && i < labels.size(); ++i)
        {
            sum += pow(labels[i] - predictions[i], 2);
        }
        // obliczenie błędu średniokwadratowego
        auto mse = sqrt(sum / predictions.size());
        std::cout << "MSE: " << mse << std::endl;
    }   
}

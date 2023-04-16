#pragma once
inline void printSharkModelEvaluation(const auto& labels, const auto& predictions)
{
    // błąd średniokwadratowy
    shark::SquaredLoss<> loss;
    auto mse = loss(labels, predictions);
    std::cout << "MSE: " << mse << std::endl;
    
    // metryka R^2
    auto var = shark::variance(labels);
    auto r_squared = 1 - mse / var(0);
    std::cout << "R^2: " << r_squared << std::endl;

    // wartość krzywej ROC
    constexpr bool invertToPositiveROC = true;
    shark::NegativeAUC roc(invertToPositiveROC);
    auto auc_roc = roc(labels, predictions);
    std::cout << "AUC ROC: " << auc_roc << std::endl << std::endl;
}
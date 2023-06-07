#include <vector>
#include <algorithm>
#include <iterator>
#include <shark/LinAlg/Base.h>

std::vector<shark::RealVector> repackToRealVectorRange(
		const auto& dataContainer)
{
    using namespace shark;

    // przepakowanie danych z typu unsigned int na typ RealVector
    std::vector<RealVector> v;
    std::for_each(dataContainer.elements().begin(), 
		    dataContainer.elements().end(), 
		    [&](const auto e){
		    	RealVector rv(1); 
		    	rv(0) = static_cast<double>(e); 
		    	v.emplace_back(rv); });
    return v;
}

inline void printSharkModelEvaluation(
    const shark::Data<unsigned int>& labels,
    const auto& predictions)
{
    using namespace shark;

    // przygotowanie solvera pola pod wykresem ROC
    constexpr bool invert = false;
    NegativeAUC<unsigned int, RealVector> auc(invert);
 
    // przepakowanie danych
    auto predVec = repackToRealVectorRange(predictions);
    auto predData = createDataFromRange(predVec);
    // obliczenie AUC ROC
    auto roc = auc(labels, predData);
    std::cout << "ROC: " << (-1 * roc) << std::endl << std::endl;
}

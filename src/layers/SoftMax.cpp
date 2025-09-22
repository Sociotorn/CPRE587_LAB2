#include "SoftMax.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the  soft max layer for the layer data
void SoftMaxLayer::computeNaive(const LayerData& dataIn) const {

    const LayerParams& in_params = getInputParams();

    size_t batch_size = 1;

    // number of inputs should equal number of outputs
    size_t num_inputs = in_params.dims[0];

    // std::cout << dataIn.get<fp32>(0) << std::endl;
    // std::cout << dataIn.get<fp32>(1) << std::endl;
    // std::cout << dataIn.get<fp32>(2) << std::endl;
    // std::cout << dataIn.get<fp32>(3) << std::endl;
    // std::cout << dataIn.get<fp32>(4) << std::endl;

    for(size_t n = 0; n < batch_size; n++)
    {
        fp32 sum_e = 0;

        // Calculate sum of exponents
        size_t i;
        for(i = 0; i < num_inputs; i++)
        {
            sum_e += exp(dataIn.get<fp32>(n * num_inputs + i));
        }
        for(i = 0; i < num_inputs; i++)
        {
            getOutputData().get<fp32>(n * num_inputs + i) = exp(dataIn.get<fp32>(n* num_inputs + i)) / sum_e;
        }
    }

    // std::cout << getOutputData().get<fp32>(0) << std::endl;
    // std::cout << getOutputData().get<fp32>(1) << std::endl;
    // std::cout << getOutputData().get<fp32>(2) << std::endl;
    // std::cout << getOutputData().get<fp32>(3) << std::endl;
    // std::cout << getOutputData().get<fp32>(4) << std::endl;
}

// Compute the soft max layer using threads
void SoftMaxLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the soft max layer using a tiled approach
void SoftMaxLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the soft max layer using SIMD
void SoftMaxLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML

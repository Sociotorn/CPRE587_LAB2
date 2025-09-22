#include "Dense.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the Fully connected layer for the layer data
void DenseLayer::computeNaive(const LayerData& dataIn) const {

    const LayerParams& in_params = getInputParams();
    const LayerParams& out_params = getOutputParams();

    size_t batch_size = 1;

    size_t in_chan    = in_params.dims[0];

    size_t out_chan   = out_params.dims[0];

    // for every batch Basically due a matrix multiply by our vector input including bias and activation function
    for(size_t n = 0; n < batch_size; n++)
    {
        // Every output channel
        for(size_t m = 0; m < out_chan; m++)
        {
            size_t out_ind = n * out_chan + m;

            fp32 sum = 0;
            // Every Input channel
            for(size_t c = 0; c < in_chan; c++)
            {
                size_t in_ind = n * in_chan + c;
                size_t filt_ind = c * out_chan + m;

                sum += dataIn.get<fp32>(in_ind) * getWeightData().get<fp32>(filt_ind);
            }

            getOutputData().get<fp32>(out_ind) = sum + getBiasData().get<fp32>(m);

            if(use_relu)
            {
                // Perform ReLU
                if(getOutputData().get<fp32>(out_ind) < 0)
                {
                    getOutputData().get<fp32>(out_ind) = 0;
                }
            }

        }
    }
}

// Compute the filly connected layer using threads
void DenseLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the fully connected layer using a tiled approach
void DenseLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the fully connected layer using SIMD
void DenseLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML

#include "Convolutional.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the convultion for the layer data
void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {

    const LayerParams& in_params = getInputParams();
    const LayerParams& weight_params = getWeightParams();
    // const LayerParams& bias_params = getBiasParams();
    const LayerParams& out_params = getOutputParams();

    size_t batch_size = 1;
    size_t stride = 1;

    size_t in_height  = in_params.dims[ParamIndex::HEIGHT];
    size_t in_width   = in_params.dims[ParamIndex::WIDTH];
    size_t in_chan    = in_params.dims[ParamIndex::CHANNELS];

    size_t out_height = out_params.dims[ParamIndex::HEIGHT];
    size_t out_width  = out_params.dims[ParamIndex::WIDTH];
    size_t out_chan   = out_params.dims[ParamIndex::CHANNELS];

    size_t filt_height = weight_params.dims[ParamIndex::HEIGHT];
    size_t filt_width = weight_params.dims[ParamIndex::WIDTH];
    // Input channels is equal to filter channels
    // Output channels is equal to the number of filters
    // ie. 5 x 5 x in_dim[2] x out_dim[2]

    // Number of biases is also equal to number of output channels

    // For every batch - probably not necessary but include anyways
    for(size_t n = 0; n < batch_size; n++)
    {
        // For every output channel
        for(size_t m = 0; m < out_chan; m++)
        {
            // For every out height pixel
            for(size_t p = 0; p < out_height; p++)
            {
                // For every out width pixel
                for(size_t q = 0; q < out_width; q++)
                {
                    // Convert indices to flattened array. Following example in ML, out_data should be in form out_data[batch][height][width][chan]
                    // Gotta be a way to access this like an array right?
                    size_t out_ind = n * out_height * out_width * out_chan + p * out_width * out_chan + q * out_chan + m;

                    fp32 weight_sum = 0;

                    // For every input channel
                    for(size_t c = 0; c < in_chan; c++)
                    {
                        // For every filter height pixel
                        for(size_t r = 0; r < filt_height; r++)
                        {
                            // For every filter width pixel
                            for(size_t s = 0; s < filt_width; s++)
                            {

                                // Convert indices to flattened array.
                                // Following example in ML, in_data should be in form in_data[batch][height][width][chan]
                                size_t in_ind = n * (in_height * in_width * in_chan) +
                                                (stride * p + r) * (in_width * in_chan) +
                                                (stride * q + s) * (in_chan) +
                                                c;

                                size_t f_ind = r * (filt_width * in_chan * out_chan) +
                                               s * (in_chan * out_chan) +
                                               c * (out_chan) +
                                               m;

                                weight_sum += dataIn.get<fp32>(in_ind) * getWeightData().get<fp32>(f_ind);
                            }
                        }
                    }
                    getOutputData().get<fp32>(out_ind) = weight_sum + getBiasData().get<fp32>(m);

                    // Perform ReLU
                    if(getOutputData().get<fp32>(out_ind) < 0)
                    {
                        getOutputData().get<fp32>(out_ind) = 0;
                    }
                }
            }
        }
    }
}

// Compute the convolution using threads
void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using a tiled approach
void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML

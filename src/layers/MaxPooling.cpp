#include "MaxPooling.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the max pooling layer for the layer data
void MaxPoolingLayer::computeNaive(const LayerData& dataIn) const {

    const LayerParams& in_params = getInputParams();
    const LayerParams& out_params = getOutputParams();

    size_t batch_size = 1;

    size_t in_height = in_params.dims[ParamIndex::HEIGHT];
    size_t in_width = in_params.dims[ParamIndex::WIDTH];

    size_t out_height = out_params.dims[ParamIndex::HEIGHT];
    size_t out_width = out_params.dims[ParamIndex::WIDTH];

    // in channels equals out channels
    size_t num_channels = in_params.dims[ParamIndex::CHANNELS];

    // These should be integers and will be 2 in this scenario
    size_t pool_vert_stride = in_height / out_height;
    size_t pool_horz_stride = in_width / out_width;

    // For every batch, channel
    for(size_t n = 0; n < batch_size; n++)
    {
        for(size_t c = 0; c < num_channels; c++)
        {

            // For every pixel on the output map
            for(size_t h = 0; h < out_height; h++)
            {
                for(size_t w = 0; w < out_width; w++)
                {
                    size_t out_ind =  n * (out_height * out_width * num_channels)
                                    + h * (out_width * num_channels)
                                    + w * (num_channels )
                                    + c;

                    // Find associated input index
                    size_t in_ind =   n * (in_height * in_width * num_channels)
                                    + (h * pool_vert_stride) * (in_width * num_channels)
                                    + (w * pool_horz_stride) * (num_channels)
                                    + c;

                    fp32 max = dataIn.get<fp32>(in_ind);

                    // Find the max in vert_stride x horz_stride areas
                    for(size_t i=0; i < pool_vert_stride; i++)
                    {
                        for(size_t j=0; j < pool_horz_stride; j++)
                        {
                            size_t in_ind =   n * (in_height * in_width * num_channels)
                                            + (h * pool_vert_stride + i) * (in_width * num_channels)
                                            + (w * pool_horz_stride + j) * (num_channels)
                                            + c;

                            if(dataIn.get<fp32>(in_ind) > max)
                            {
                                max = dataIn.get<fp32>(in_ind);
                            }
                        }
                    }

                    getOutputData().get<fp32>(out_ind) = max;
                }
            }
        }
    }
    // std::cout << dataIn.get<fp32>(0) << dataIn.get<fp32>(num_channels) << std::endl;
}

// Compute the max pooling layer using threads
void MaxPoolingLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the max pooling layer using a tiled approach
void MaxPoolingLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the max pooling layer using SIMD
void MaxPoolingLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML

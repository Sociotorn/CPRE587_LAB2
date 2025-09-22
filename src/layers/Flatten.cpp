#include "Flatten.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the  soft max layer for the layer data
void Flatten::computeNaive(const LayerData& dataIn) const {

    const LayerParams& out_params = getOutputParams();

    // number of inputs should equal number of outputs
    size_t out_channels = out_params.dims[0];

    memcpy(getOutputData().raw(), dataIn.raw(), out_channels * sizeof(fp32));
}

// Compute the soft max layer using threads
void Flatten::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the soft max layer using a tiled approach
void Flatten::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the soft max layer using SIMD
void Flatten::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML
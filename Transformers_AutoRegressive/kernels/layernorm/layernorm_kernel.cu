#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layernorm_forward_kernel(const float* __restrict__ input, float* __restrict__ output, float eps, int outer_size, int norm_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer_size) return;

    const float* row_input = input + row * norm_size;
    float* row_output = output + row * norm_size;

    // Step 1: mean
    float mean = 0.0f;
    for (int i = 0; i < norm_size; i++) {
        mean += row_input[i];
    }
    mean /= norm_size;

    // Step 2: variance
    float var = 0.0f;
    for (int i = 0; i < norm_size; i++) {
        float diff = row_input[i] - mean;
        var += diff * diff;
    }
    var /= norm_size;
    float inv_std = rsqrtf(var + eps);

    // Step 3: normalize and scale
    for (int i = 0; i < norm_size; i++) {
        float norm = (row_input[i] - mean) * inv_std;
        row_output[i] = norm;
    }

}

void layernorm_forward_cuda(torch::Tensor input, torch::Tensor output, float eps, int dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA Tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA Tensor");
    int threads = 256;
    int ndim = input.dim();
    if (dim < 0) dim += ndim;

    
    int outer_size = 1;
    int norm_size = 1;
    for (int i = 0; i < ndim; ++i) {
        if (i < ndim) outer_size *= input.size(i);
        else norm_size *= input.size(i);
    }
    
    int blocks = (outer_size + threads - 1)/threads;


    layernorm_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), eps, outer_size, norm_size);
}
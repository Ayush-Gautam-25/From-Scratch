#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float inline  gelu_forward(float x) {
    // General Formula of GELU
    return 0.5f * x * (1.f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = gelu_forward(input[i]);
    }
}

void gelu_forward_cuda(torch::Tensor input, torch::Tensor output) {
    int N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    gelu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
}

__device__ float gelu_backward(float x, float grad_output) {
    // Using Chain Rule, calculate the dx
    float tanh_out = tanhf(0.79788456f * (x + 0.044715f * x * x * x));
    float sech_sq = 1.0f - tanh_out * tanh_out;
    float dx = 0.5f * tanh_out + 
               (0.5f * x * sech_sq * 0.79788456f * (1 + 3 * 0.044715f * x * x)) + 
               0.5f;
    return grad_output * dx;
}

__global__ void gelu_backward_kernel(const float* grad_output, const float* input, float* grad_input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad_input[i] = gelu_backward(input[i], grad_output[i]);
    }
}

void gelu_backward_cuda(torch::Tensor grad_output, torch::Tensor input, torch::Tensor grad_input) {
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    gelu_backward_kernel<<<blocks, threads>>>(grad_output.data_ptr<float>(), input.data_ptr<float>(), grad_input.data_ptr<float>(), size);
}

#include <torch/extension.h>

void layernorm_forward_cuda(torch::Tensor input, torch::Tensor output, float eps, int dim);

torch::Tensor layernorm_forward(torch::Tensor input, float eps, int dim) {
    auto output = torch::empty_like(input);
    layernorm_forward_cuda(input, output, eps, dim);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)");
}

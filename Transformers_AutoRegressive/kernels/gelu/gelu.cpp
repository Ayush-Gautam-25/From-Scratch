#include <torch/extension.h>

// Declare CUDA launcher
void gelu_forward_cuda(torch::Tensor input, torch::Tensor output);

void gelu_backward_cuda(torch::Tensor grad_output, torch::Tensor input, torch::Tensor grad_input);


// C++ entry point exposed to Python
torch::Tensor gelu_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    gelu_forward_cuda(input, output);
    return output;
}

torch::Tensor gelu_backward(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::empty_like(input);
    gelu_backward_cuda(grad_output, input, grad_input);
    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Forward function of GELU");
    m.def("backward", &gelu_backward, "Backward function of GELU");
}


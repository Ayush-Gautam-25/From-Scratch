import torch.nn as nn
import os
from torch.autograd import Function
from torch.utils.cpp_extension import load
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

this_dir = os.path.dirname(__file__)

gelu_ext = load(
    name="gelu_cuda", 
    sources=[    
        os.path.join(this_dir, "gelu.cpp"),
        os.path.join(this_dir, "gelu_kernel.cu"),
], 
    extra_cuda_cflags=["-ccbin", "/usr/bin/g++-10"],
    verbose=True
)


class FusedGELU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_ext.forward(input) 

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return gelu_ext.backward(grad_output, input)

def gelu_cuda(x):
    return FusedGELU.apply(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu_cuda(x)  # Calls your FusedGELU autograd function

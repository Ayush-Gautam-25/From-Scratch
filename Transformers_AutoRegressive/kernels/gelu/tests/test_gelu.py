import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from gelu import gelu_cuda
x = torch.randn(4, 4, device="cuda", requires_grad=True)

# Forward using custom GELU
y = gelu_cuda(x)

print("Input:")
print(x)
print("GELU output:")
print(y)


# # Backward: compute gradients
loss = y.sum()
loss.backward()

# Print gradients
print("Input gradients:")
print(x.grad)

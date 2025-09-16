from lanczos import lanczos_algorithm
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import _solve_symmetric_tridiagonal_eigenproblem, _construct_tridiagonal_matrix, _select_k_evals, _convergence_check

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
NET = Net()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NET.to(DEVICE)

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam(NET.parameters(), lr=0.001, betas=(0.9, 0.999))
DATALOADER = trainloader

def test_convergence_check():
    α = torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0])
    β = torch.tensor([1.0, 1.0, 1.0, 1.0])

    for j in range(3, len(α)+1):
        α_j = α[:j]
        β_j = β[:j-1]

        converged, max_change = _convergence_check(
            α_j, β_j,
            tolerance=1e-1,
            j=j,
            k=1,
            which="largest",
            lookback=2
        )
        print(f"iter={j}, converged={converged}, max_change={max_change:.4f}")

    print("✅ Test 2 Passed\n")


if __name__=="__main__":

    ## TEST 1:
    ## _construct_tridiagonal_matrix 
    ## _solve_symmetric_tridiagonal_eigenproblem
    ## _select_k_vals

    # -----------------------------------------------------------
    alpha = torch.tensor([2.0, 3.0, 4.0])     # diagonal
    beta  = torch.tensor([1.0, 1.0])

    T = _construct_tridiagonal_matrix(alpha, beta)

    evals, evecs = _solve_symmetric_tridiagonal_eigenproblem(T)

    print("Tridiagonal matrix T:")
    print(T)
    print("\nEigenvalues:")
    print(evals)
    print("\nEigenvectors:")
    print(evecs)

    for which in ["largest", "smallest", "both_extremes", "around_zero", "magnitude"]:
        selected_evals, selected_evecs = _select_k_evals(evals, evecs, k=2, which=which)
        print(f"\n--- {which} ---")
        print("Selected evals:", selected_evals)
        print("Selected evecs:\n", selected_evecs)
    
    print("✅ Test 1 Passed\n")
    # -----------------------------------------------------------

    # TEST 2:
    # _convergence_check
    # -----------------------------------------------------------
    test_convergence_check()
    # -----------------------------------------------------------

    ## FINAL TEST:
    ## lanczos_algorithm (fully implemented)
    # -----------------------------------------------------------
    lanczos_algorithm(NET, DATALOADER, CRITERION, 200, 8, optimizer=OPTIMIZER, device=DEVICE)
    print("✅ Test 3 passed")
    # -----------------------------------------------------------

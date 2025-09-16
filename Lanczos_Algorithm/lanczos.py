# For more insight to the algorithm, refer 
# Algorithm - https://en.wikipedia.org/wiki/Lanczos_algorithm
# Algorithm 1 of "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density" - https://arxiv.org/pdf/1901.10159

from torch.optim import optimizer, Optimizer
import torch
from torch import nn
from utils import _hessian_vector_product, _convergence_check, _construct_tridiagonal_matrix, _solve_symmetric_tridiagonal_eigenproblem, _select_k_evals, _reshape_to_model_params, _compute_residual_norms
    
def lanczos_algorithm(model: torch.nn.Module, 
                      dataloader: torch.utils.data.DataLoader , 
                      criterion: torch.nn.Module, 
                      m_max: int,
                      k: int = None,
                      which: str = "largest",
                      tolerance: float = 1e-2,
                      device: str = "cuda",
                      hvp_method: str = 'autodiff', # Hessian-Vector Product Method,
                      verbose: bool = True
                      ):
    """
    Input:
        - model: PyTorch neural network model
        - dataloader: DataLoader with training/validation data
        - criterion: loss function (e.g., CrossEntropyLoss, MSELoss)
        - m_max: maximum Lanczos iterations
        - k: number of eigenvalues/eigenvectors to extract (None for all)
        - which: 'largest', 'smallest', 'both_extremes', or 'around_zero'
        - optimizer: optimizer (for parameter grouping)
        - device: computation device ('cuda' or 'cpu')
        - hvp_method: method to calculate hessian-vector product, 'autodiff' or 'finite_diff'

    Output:
        - eigenvalues: approximate eigenvalues of Hessian
        - eigenvectors: corresponding eigenvectors (optional)
        - convergence_info: iteration count, residual norms
    """

    ## INITIALIZATION
    # Extract and flatten all trainable parameters
    param_shape = []
    n_total = 0

    for param in model.parameters():
        if param.requires_grad:
            param_shape.append(param.shape)
            # print(param.flatten().shape)

            n_total += param.numel()

    n = n_total

    v_prev = torch.randn(n, device=device) / (n**0.5)
    v_prev = torch.div(v_prev, torch.norm(v_prev))

    α = torch.zeros(m_max+1, device=device)
    β = torch.zeros(m_max+1, device=device)
    V = torch.zeros(n, m_max+1, device=device) # Lanczos vectors

    data_batch = next(iter(dataloader))
    print(f"[LOG] Shape of Data Batch: {len(data_batch)}")
    X, y = data_batch[0].to(device), data_batch[1].to(device)
    print(f"[LOG] Shape of First X, y in data_batch, X: {X.shape}, y: {y.shape}")

    β_curr = 0
    v_curr = None
    m = None
    
    w_prevd = _hessian_vector_product(model, criterion, X, y, v_prev, hvp_method, device) 
    α_curr = torch.dot(v_prev, w_prevd)
    w_prev = w_prevd - α_curr * v_prev

    V[:, 1] = v_prev

    α[1] = α_curr
    β[1] = β_curr

    for j in range(2, m_max+1):
        β_curr = torch.norm(w_prev)
        if (β_curr != 0):
            v_curr = w_prev / β_curr
        else:
            v_curr = torch.randn(n, device=device)
            v_curr /= torch.norm(v_curr)
 
        for _ in range(2):
            for k_ in range(1, j):
                proj = torch.dot(v_curr, V[:, k_])
                v_curr = v_curr - proj * V[:, k_]

        v_norm = torch.norm(v_curr)
        if v_norm < 1e-12:
            if verbose:
                print(f"[WARNING] Lanczos breakdown at iteration {j}")
            m = j - 1
            break
        v_curr /= v_norm  

        w_prevd = _hessian_vector_product(model, criterion, X, y, v_curr, hvp_method, device) - β_curr * v_prev
        α_curr = torch.dot(v_curr, w_prevd)
        w_prev = w_prevd - α_curr * v_curr

        v_prev = v_curr
        V[:, j] = v_prev
        α[j] = α_curr
        β[j] = β_curr

        if j % 5 == 0 and j >= 5:
            converged, residual_norm = _convergence_check(α[1:j+1], β[2:j+1], tolerance, j, k, which)
            if verbose:
                print(f"[LOG] Iteration {j}: max residual = {residual_norm:.2e} converged: {converged}")
            if converged:
                if verbose:
                    print(f"[LOG] Converged at iteration {j}")
                m = j
                break

    if m is None:
        m = m_max
    
    T = _construct_tridiagonal_matrix(alpha=α[1:m+1], beta=β[2:m+1])
    evals, evecs = _solve_symmetric_tridiagonal_eigenproblem(T)

    evals_k, evecs_k = _select_k_evals(evals, evecs, k, which)

    # compute Ritz vectors (approximate eigenvectors of H)
    eigenvectors = V[:, 1:m+1] @ evecs_k

    reshaped_eigenvectors = []
    for i in range(eigenvectors.shape[1]):
        flat_vector = eigenvectors[:, i]
        reshaped = _reshape_to_model_params(flat_vector, param_shape)
        reshaped_eigenvectors.append(reshaped)

    ## RESIDUAL CHECK:  r_i = ||Hv - λv||
    residuals = []
    for i in range(eigenvectors.shape[1]):
        v = eigenvectors[:, i]
        λ = evals_k[i]
        Hv = _hessian_vector_product(model, criterion, X, y, v, hvp_method, device)
        r_norm = torch.norm(Hv - λ*v).item()
        residuals.append(r_norm)
        if verbose:
            print(f"[CHECK] Eigenvalue {i}: λ={λ:.4f}, residual={r_norm:.2e}")

    return {
        "eigenvalues": evals_k,
        "eigenvectors": reshaped_eigenvectors,
        "residuals": residuals,
        "iterations": m,
    }

    
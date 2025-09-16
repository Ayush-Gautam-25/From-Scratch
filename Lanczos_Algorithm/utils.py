import torch
import math
from functools import reduce
from operator import mul

def _hessian_vector_product(model: torch.nn.Module, 
                            criterion: torch.nn.Module,
                            X: torch.Tensor,
                            y: torch.Tensor,
                            v: torch.Tensor,
                            hvp_method: str = "autodiff",
                            device: str = "cuda",
                            retain_graph: bool = False,
                            ):
    """
    Compute Hessian-vector product (HVP) for a given model, loss, and vector v.

    Args:
        model: torch.nn.Module
        criterion: torch.nn loss function
        X: input tensor
        y: target tensor
        v: vector to multiply Hessian with (must match total #params)
        retain_graph: if True, allows multiple HVP calls w/o recomputing forward
        device: device string ("cpu" or "cuda")

    Returns:
        hvp: torch.Tensor (flattened Hessian-vector product)
    """
    if hvp_method=="autodiff":
        # Method 1: Exact computation via automatic differentiation
        model.zero_grad(set_to_none=True)

        outputs = model(X)
        loss = criterion(outputs, y)

        # Compute Gradient-Vector Product
        grad_tuple = torch.autograd.grad(
            loss, 
            model.parameters(), 
            create_graph=True, 
            retain_graph=True, 
            allow_unused=True
        )

        # Flatten gradients
        grad_flat = torch.cat([g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
                           for g, p in zip(grad_tuple, model.parameters())])
        num_params = grad_flat.numel()
        if v.numel() != num_params:
            raise ValueError(f"Shape mismatch: v has {v.numel()} elements, "
                            f"but model has {num_params} parameters.")

        v = v.to(device)
        gvp = torch.dot(grad_flat, v)

        # Compute Hessian-Vector Product
        hvp_tuple = torch.autograd.grad(
            gvp,
            model.parameters(),
            retain_graph=retain_graph,
            allow_unused=True
        )

        # Flatten gradients
        hvp_flat = torch.cat([h.reshape(-1) if h is not None else torch.zeros(p.numel(), device=device) for h, p in zip(hvp_tuple, model.parameters())])
        return hvp_flat.to(device)
    else:
        raise ValueError(f"[ERROR] Unknown hvp_method: {hvp_method}")
    
def _construct_tridiagonal_matrix(
        alpha: torch.Tensor,
        beta: torch.Tensor
        ) -> torch.Tensor:
    """
    Construct the symmetric tridiagonal matrix T from alpha and beta coefficients.

    Args:
        alpha: tensor of shape (m,) containing diagonal elements α₁...α_m
        beta:  tensor of shape (m-1,) containing off-diagonal elements β₂...β_m

    Returns:
        T: (m x m) symmetric tridiagonal matrix
    """
    m = alpha.shape[0]
    T = torch.diag(alpha)  # main diagonal

    if m > 1:
        T += torch.diag(beta, diagonal=-1) # sub-diagonal
        T += torch.diag(beta, diagonal=1)  # super-diagonal

    return T

def _solve_symmetric_tridiagonal_eigenproblem(T: torch.Tensor):
    """
    Solves the eigenproblem for a real symmetric tridiagonal matrix T.
    Input:
        T: (m x m) dense symmetric tridiagonal matrix (torch.Tensor)
    Output:
        eigenvalues: (m,)
        eigenvectors: (m x m) matrix with columns as eigenvectors
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    return eigenvalues, eigenvectors

def _select_k_evals(evals: torch.Tensor, evecs: torch.Tensor, k: int, which: str):
    n = evals.shape[0]
    if k is None:
        k = n

    k = min(k, n)

    sorted_indices = torch.argsort(evals, descending=True)
    sorted_evals = evals[sorted_indices]
    sorted_evecs = evecs[:, sorted_indices]

    if which == "largest":
        selected_indices = torch.arange(0, k)
    
    elif which == "smallest":
        selected_indices = torch.arange(n - k, n)
    
    elif which == "both_extremes":
        k_half = k // 2
        k_rem = k % 2

        largest_indices = torch.arange(0, k_half + k_rem)
        smallest_indices = torch.arange(n - k_half, n)
        selected_indices = torch.cat([largest_indices, smallest_indices])

    elif which == "around_zero":
        abs_evals = torch.abs(sorted_evals)
        zero_sorted_indices = torch.argsort(abs_evals)
        selected_indices = zero_sorted_indices[:k]

    elif which == "magnitude":
        abs_evals = torch.abs(sorted_evals)
        magnitude_indices = torch.argsort(abs_evals, descending=True)
        selected_indices = magnitude_indices[:k]

    else:
        raise ValueError(f"Unknown selection criterion: {which}")
    
    selected_evals = sorted_evals[selected_indices]
    selected_evecs = sorted_evecs[:, selected_indices]

    return selected_evals, selected_evecs

def _reshape_to_model_params(
        flat_vector: torch.Tensor,
        param_shape
        ):
    params = []
    start = 0
    for shape in param_shape:
        size = reduce(mul, shape, 1)
        param = flat_vector[start:start + size].reshape(shape)
        params.append(param)
        start += size
    return params

def _compute_residual_norms(A_fn, eigenvalues, eigenvectors):
    """
    Compute residual norms ||A v - λ v|| for each Ritz pair.

    Args:
        A_fn (callable): function that applies A to a vector (torch.Tensor).
        eigenvalues (torch.Tensor): shape (k,), approximate eigenvalues.
        eigenvectors (torch.Tensor): shape (n, k), approximate eigenvectors.

    Returns:
        torch.Tensor: residual norms, shape (k,)
    """
    residuals = []
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        Av = A_fn(v)
        res = torch.norm(Av - eigenvalues[i] * v)
        residuals.append(res)
    return torch.tensor(residuals, device=eigenvalues.device)    

def _convergence_check(
        α: torch.Tensor,
        β: torch.Tensor,
        tolerance: float,
        j: int,
        k: int, 
        which: str,
        lookback: int = 10,
        # verbose: bool = True        
       ):
    # Sanity checks
    assert α.ndim == 1 and β.ndim == 1, "[ERROR] α and β must be 1D tensors"
    assert len(α) == j, f"[ERROR] Expected α of length {j}, got {len(α)}"
    assert len(β) == j-1, f"[ERROR] Expected β of length {j-1}, got {len(β)}"

    if len(α) < 2 or len(β) < 1:
        print(f"[WARNING] alpha < 2 or beta < 1, alpha: {α}, beta: {β}")
        return False, float("inf")

    if j < lookback: 
        print(f"[WARNING] iterate is less than lookback, iterate: {j}, lookback: {lookback}")
        return False, float("inf")

    # Current α and β
    T_curr = _construct_tridiagonal_matrix(α, β)
    evals_curr, evecs_curr = _solve_symmetric_tridiagonal_eigenproblem(T_curr)
    evals_curr, _ = _select_k_evals(evals_curr, evecs_curr, k, which)

    # α's and β's but excluding last lookbacks
    T_prev = _construct_tridiagonal_matrix(α[:-lookback], β[:-lookback])
    evals_prev, evecs_prev = _solve_symmetric_tridiagonal_eigenproblem(T_prev)
    evals_prev, _ = _select_k_evals(evals_prev, evecs_prev, k, which)

    if evals_prev is None or len(evals_prev) == 0:
        print(f"[WARNING] evals_prev is None or empty, evals_prev: {evals_prev}")
        return False, float("inf")
    
    evals_curr, _ = torch.sort(evals_curr)
    evals_prev, _ = torch.sort(evals_prev)

    # if verbose:
    #     print(f"[VERBOSE] iter={j}")
    #     print("[VERBOSE]  evals_curr:", evals_curr.tolist())
    #     print("[VERBOSE]  evals_prev:", evals_prev.tolist())

    if len(evals_prev) < k or len(evals_curr) < k:
        print(f"[WARNING] got number of eigenvalues less than asked, got {min(len(evals_curr), len(evals_prev))}, asked {k}")
        return False, float("inf")

    compare_k = min(len(evals_curr), len(evals_prev))
    if compare_k == 0:
        print(f"[WARNING] got zero diffierence in current and previous eigenvalues: {compare_k}")
        return False, float("inf")

    changes = torch.abs(evals_curr[:compare_k] - evals_prev[:compare_k])
    rel_changes = changes / (torch.abs(evals_prev[:compare_k]) + 1e-12)
    max_change = torch.max(rel_changes).item()
    return max_change < tolerance, max_change
"""
S4: Structured State Spaces for Sequence Modeling
Paper: "Efficiently Modeling Long Sequences with Structured State Spaces"
       Albert Gu, Karan Goel, Christopher Re (2021)
       arXiv: 2111.00396

This module provides numpy-based demonstrations of S4 computations.
"""

import numpy as np
import json


def build_hippo_legs_matrix(N):
    """
    Build the HiPPO-LegS (Legendre State) matrix A and vector B.

    From the HiPPO paper (Gu et al., 2020):
        A_nk = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
        A_nk = -(n+1)                         if n = k
        A_nk = 0                               if n < k

        B_n = (2n+1)^{1/2}

    Returns: A (N, N), B (N, 1)
    """
    A = np.zeros((N, N))
    B = np.zeros((N, 1))

    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
            elif n == k:
                A[n, k] = -(n + 1)
            # else: A[n,k] = 0 (already initialized)
        B[n, 0] = np.sqrt(2 * n + 1)

    return A, B


def discretize_bilinear(A, B, C, delta):
    """
    Bilinear (Tustin) discretization of continuous SSM.

    From S4 paper:
        A_bar = (I - delta/2 * A)^{-1} (I + delta/2 * A)
        B_bar = (I - delta/2 * A)^{-1} * delta * B
        C_bar = C

    Args:
        A: (N, N) state matrix
        B: (N, 1) input matrix
        C: (1, N) output matrix
        delta: scalar step size

    Returns: A_bar, B_bar, C_bar
    """
    N = A.shape[0]
    I = np.eye(N)

    inv_term = np.linalg.inv(I - (delta / 2) * A)
    A_bar = inv_term @ (I + (delta / 2) * A)
    B_bar = inv_term @ (delta * B)
    C_bar = C.copy()

    return A_bar, B_bar, C_bar


def discretize_zoh(A, B, C, delta):
    """
    Zero-Order Hold (ZOH) discretization.

    Used by Mamba and later SSMs:
        A_bar = exp(delta * A)
        B_bar = A^{-1} (A_bar - I) * B   (or approximated)
        C_bar = C

    For simplicity we use first-order approximation when A is singular.
    """
    N = A.shape[0]
    I = np.eye(N)

    dA = delta * A
    # Matrix exponential via eigendecomposition for small N
    A_bar = np.real(expm_simple(dA))

    # B_bar = A^{-1}(A_bar - I) B
    try:
        A_inv = np.linalg.inv(A)
        B_bar = A_inv @ (A_bar - I) @ B
    except np.linalg.LinAlgError:
        # Fallback: first-order approximation
        B_bar = delta * B

    C_bar = C.copy()
    return A_bar, B_bar, C_bar


def expm_simple(M):
    """Simple matrix exponential using eigendecomposition."""
    eigenvalues, V = np.linalg.eig(M)
    exp_eigenvalues = np.exp(eigenvalues)
    return V @ np.diag(exp_eigenvalues) @ np.linalg.inv(V)


def ssm_recurrent(A_bar, B_bar, C_bar, u):
    """
    Recurrent (RNN) mode of SSM.

    x_k = A_bar * x_{k-1} + B_bar * u_k
    y_k = C_bar * x_k

    Args:
        A_bar: (N, N) discretized state matrix
        B_bar: (N, 1) discretized input matrix
        C_bar: (1, N) output matrix
        u: (L,) input sequence

    Returns:
        x_history: (L, N) state at each step
        y: (L,) output sequence
    """
    N = A_bar.shape[0]
    L = len(u)

    x = np.zeros(N)
    x_history = np.zeros((L, N))
    y = np.zeros(L)

    for k in range(L):
        x = A_bar @ x + B_bar.flatten() * u[k]
        x_history[k] = x
        y[k] = C_bar.flatten() @ x

    return x_history, y


def ssm_convolution(A_bar, B_bar, C_bar, u):
    """
    Convolution mode of SSM.

    K = (C_bar B_bar, C_bar A_bar B_bar, ..., C_bar A_bar^{L-1} B_bar)
    y = K * u   (convolution, not multiplication)

    Args:
        A_bar: (N, N) discretized state matrix
        B_bar: (N, 1) discretized input matrix
        C_bar: (1, N) output matrix
        u: (L,) input sequence

    Returns:
        K: (L,) convolution kernel
        y: (L,) output sequence
    """
    L = len(u)
    C_flat = C_bar.flatten()
    B_flat = B_bar.flatten()

    # Build kernel: K_i = C * A^i * B
    K = np.zeros(L)
    A_power = np.eye(A_bar.shape[0])  # A^0 = I
    for i in range(L):
        K[i] = C_flat @ A_power @ B_flat
        A_power = A_power @ A_bar

    # Convolution via FFT
    # Pad to avoid circular convolution
    K_padded = np.zeros(2 * L)
    K_padded[:L] = K
    u_padded = np.zeros(2 * L)
    u_padded[:L] = u

    y_full = np.real(np.fft.ifft(np.fft.fft(K_padded) * np.fft.fft(u_padded)))
    y = y_full[:L]

    return K, y


def generate_s4_demo_data(N=4, L=16, seed=42):
    """
    Generate a complete S4 demo with all intermediate tensors.
    Returns a dict of JSON-serializable data for visualization.
    """
    np.random.seed(seed)

    # Build HiPPO matrix
    A, B = build_hippo_legs_matrix(N)
    C = np.random.randn(1, N) * 0.1
    delta = 0.1

    # Input signal: simple sine wave
    t = np.linspace(0, 2 * np.pi, L)
    u = np.sin(t)

    # Discretize
    A_bar, B_bar, C_bar = discretize_bilinear(A, B, C, delta)

    # Recurrent computation (with step-by-step tracking)
    x_history, y_rec = ssm_recurrent(A_bar, B_bar, C_bar, u)

    # Convolution computation
    K, y_conv = ssm_convolution(A_bar, B_bar, C_bar, u)

    return {
        "N": N,
        "L": L,
        "delta": delta,
        # Matrices (as nested lists for JSON)
        "A_continuous": A.tolist(),
        "B_continuous": B.tolist(),
        "C": C.tolist(),
        "A_bar": A_bar.tolist(),
        "B_bar": B_bar.tolist(),
        "C_bar": C_bar.tolist(),
        # Sequences
        "input_u": u.tolist(),
        "time_t": t.tolist(),
        "state_history": x_history.tolist(),
        "output_recurrent": y_rec.tolist(),
        "kernel_K": K.tolist(),
        "output_convolution": y_conv.tolist(),
        # Shapes for display
        "shapes": {
            "A": f"({N}, {N})",
            "B": f"({N}, 1)",
            "C": f"(1, {N})",
            "u": f"({L},)",
            "x": f"({N},)",
            "y": f"({L},)",
            "K": f"({L},)",
            "A_bar": f"({N}, {N})",
            "B_bar": f"({N}, 1)",
        }
    }


def generate_s4_execution_steps(N=4, L=8, seed=42):
    """
    Generate step-by-step execution data for the S4 recurrence.
    Each step corresponds to one line of code being executed,
    with all intermediate tensor values.
    """
    np.random.seed(seed)
    A, B = build_hippo_legs_matrix(N)
    C = np.random.randn(1, N) * 0.1
    delta = 0.1
    t = np.linspace(0, 2 * np.pi, L)
    u = np.sin(t)
    A_bar, B_bar, C_bar = discretize_bilinear(A, B, C, delta)

    code = [
        {"line": "import torch", "indent": 0},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "def ssm_recurrent(", "indent": 0},
        {"line": f"    A_bar: torch.Tensor,  # ({N}, {N})", "indent": 0},
        {"line": f"    B_bar: torch.Tensor,  # ({N}, 1)", "indent": 0},
        {"line": f"    C_bar: torch.Tensor,  # (1, {N})", "indent": 0},
        {"line": f"    u: torch.Tensor,      # ({L},) input sequence", "indent": 0},
        {"line": f") -> torch.Tensor:", "indent": 0},
        {"line": f"L = u.shape[0]", "indent": 1, "comment": f"= {L}"},
        {"line": f"N = A_bar.shape[0]", "indent": 1, "comment": f"= {N}"},
        {"line": f"x = torch.zeros(N)", "indent": 1, "comment": f"hidden state ({N},)"},
        {"line": f"y = torch.zeros(L)", "indent": 1, "comment": f"output ({L},)"},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "for k in range(L):", "indent": 1},
        {"line": "# State transition: matrix-vector multiply", "indent": 2},
        {"line": "Ax = torch.mv(A_bar, x)", "indent": 2, "comment": f"({N},{N}) @ ({N},) -> ({N},)"},
        {"line": "# Input injection: scale B by input scalar", "indent": 2},
        {"line": "Bu = B_bar.squeeze(-1) * u[k]", "indent": 2, "comment": f"({N},) * scalar -> ({N},)"},
        {"line": "# New state = retained + new input", "indent": 2},
        {"line": "x = Ax + Bu", "indent": 2, "comment": f"({N},) + ({N},) -> ({N},)"},
        {"line": "# Read output from state", "indent": 2},
        {"line": "y[k] = C_bar @ x", "indent": 2, "comment": f"(1,{N}) @ ({N},) -> scalar"},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "return y", "indent": 1, "comment": f"torch.Tensor ({L},)"},
    ]

    steps = []
    x = np.zeros(N)
    y = np.zeros(L)

    # Initial state
    steps.append({
        "lineIdx": 10,
        "description": f"Initialize hidden state x as a zero vector of dimension N={N}. This is our 'memory' - it compresses the entire input history into {N} numbers.",
        "tensors": {
            "x": {"shape": f"({N},)", "value": x.tolist(), "justComputed": True,
                   "annotation": "Hidden state initialized to zeros"},
            "A_bar": {"shape": f"({N},{N})", "value": A_bar.tolist(),
                      "annotation": "Discretized state matrix (from HiPPO + bilinear)"},
            "B_bar": {"shape": f"({N},1)", "value": B_bar.tolist(),
                      "annotation": "Discretized input projection"},
            "C_bar": {"shape": f"(1,{N})", "value": C_bar.tolist(),
                      "annotation": "Output projection (reads from state)"},
        }
    })

    steps.append({
        "lineIdx": 11,
        "description": f"Initialize output array y of length L={L}. Each y[k] will be a scalar output at timestep k.",
        "tensors": {
            "y": {"shape": f"({L},)", "value": y.tolist(), "justComputed": True,
                   "annotation": "Output sequence (to be filled)"},
            "x": {"shape": f"({N},)", "value": x.tolist()},
            "u": {"shape": f"({L},)", "value": u.tolist(),
                   "annotation": "Input signal (sine wave)"},
        }
    })

    for k in range(min(L, 5)):  # Show first 5 timesteps
        u_k = u[k]

        # Step 1: Ax = A_bar @ x
        Ax = A_bar @ x
        steps.append({
            "lineIdx": 15,
            "description": f"k={k}: Multiply state matrix A_bar ({N}x{N}) by current state x ({N},). This applies the state transition - A_bar controls how quickly the state 'forgets' old information.",
            "tensors": {
                "k": {"shape": "scalar", "value": k, "justComputed": True, "color": "#facc15"},
                "x": {"shape": f"({N},)", "value": x.tolist(),
                       "annotation": "Current hidden state (from previous step)"},
                "A_bar @ x": {"shape": f"({N},)", "value": Ax.tolist(), "justComputed": True,
                              "annotation": f"State after transition: how much of x is retained", "color": "#a78bfa"},
            }
        })

        # Step 2: Bu = B_bar * u[k]
        Bu = B_bar.flatten() * u_k
        steps.append({
            "lineIdx": 17,
            "description": f"k={k}: Scale input projection B_bar by input u[{k}]={u_k:.4f}. This determines how the current input token enters the hidden state.",
            "tensors": {
                "u[k]": {"shape": "scalar", "value": round(u_k, 4), "justComputed": True, "color": "#facc15"},
                "B_bar": {"shape": f"({N},1)", "value": B_bar.tolist()},
                "B_bar * u[k]": {"shape": f"({N},)", "value": Bu.tolist(), "justComputed": True,
                                  "annotation": f"Input contribution: B_bar scaled by u[{k}]", "color": "#4ade80"},
            }
        })

        # Step 3: x = Ax + Bu
        x_new = Ax + Bu
        steps.append({
            "lineIdx": 19,
            "description": f"k={k}: Add state transition and input contribution to get new state. x = (retained old state) + (new input). This is the core SSM recurrence.",
            "tensors": {
                "A_bar @ x": {"shape": f"({N},)", "value": Ax.tolist(), "color": "#a78bfa",
                              "annotation": "Retained from previous state"},
                "B_bar * u[k]": {"shape": f"({N},)", "value": Bu.tolist(), "color": "#4ade80",
                                  "annotation": "New input contribution"},
                "x_new": {"shape": f"({N},)", "value": x_new.tolist(), "justComputed": True,
                          "annotation": "New state = old retained + new input", "color": "#60a5fa"},
            }
        })
        x = x_new

        # Step 4: y[k] = C_bar @ x
        y_k = float(C_bar.flatten() @ x)
        y[k] = y_k
        steps.append({
            "lineIdx": 21,
            "description": f"k={k}: Read output from state by multiplying C_bar (1x{N}) with state x ({N},). This 'reads' a scalar from the {N}-dimensional state. y[{k}] = {y_k:.6f}",
            "tensors": {
                "C_bar": {"shape": f"(1,{N})", "value": C_bar.tolist()},
                "x": {"shape": f"({N},)", "value": x.tolist()},
                "y[k]": {"shape": "scalar", "value": round(y_k, 6), "justComputed": True,
                          "annotation": f"Output at step {k}: dot product of C and state", "color": "#fb923c"},
                "y (so far)": {"shape": f"({L},)", "value": [round(v, 4) for v in y.tolist()],
                               "annotation": f"Output sequence (filled up to k={k})", "highlightIdx": k},
            }
        })

    # Final return
    _, y_full = ssm_recurrent(A_bar, B_bar, C_bar, u)
    steps.append({
        "lineIdx": 23,
        "description": f"After processing all {L} timesteps, return the complete output sequence y. Each output value was computed by reading from the evolving {N}-dimensional hidden state.",
        "tensors": {
            "y (complete)": {"shape": f"({L},)", "value": [round(v, 4) for v in y_full.tolist()],
                             "justComputed": True, "annotation": "Complete output sequence",
                             "color": "#fb923c"},
            "u (input)": {"shape": f"({L},)", "value": [round(v, 4) for v in u.tolist()],
                          "annotation": "Original input for comparison"},
        }
    })

    return {"code": code, "steps": steps}

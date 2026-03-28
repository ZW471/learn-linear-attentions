"""
Mamba-1: Linear-Time Sequence Modeling with Selective State Spaces
Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
       Albert Gu, Tri Dao (2023)
       arXiv: 2312.00752

Key innovation: Input-dependent (selective) B, C, and delta parameters.
"""

import numpy as np


def selective_scan_naive(u, delta, A, B, C, D=None):
    """
    Naive (non-hardware-aware) selective scan.

    The selective SSM uses input-dependent parameters:
        x_t = A_bar_t * x_{t-1} + B_bar_t * u_t
        y_t = C_t @ x_t + D * u_t

    Where discretization is done per-timestep with ZOH:
        A_bar_t = exp(delta_t * A)       # (N,) diagonal
        B_bar_t = (exp(delta_t * A) - I) * A^{-1} * delta_t * B_t
                ≈ delta_t * B_t          # first-order approx

    Args:
        u: (B, L, D) input tensor
        delta: (B, L, D) step sizes (input-dependent)
        A: (D, N) state transition (diagonal, shared)
        B: (B, L, N) input-dependent input matrix
        C: (B, L, N) input-dependent output matrix
        D: (D,) skip connection (optional)

    Returns:
        y: (B, L, D) output
        states: list of (B, D, N) hidden states for visualization
    """
    batch, L, D_dim = u.shape
    N = A.shape[1]

    # Initialize
    x = np.zeros((batch, D_dim, N))  # (B, D, N) hidden state
    y = np.zeros((batch, L, D_dim))
    states = []

    for t in range(L):
        # Per-timestep discretization (ZOH approximation)
        # delta: (B, L, D) -> delta_t: (B, D)
        delta_t = delta[:, t, :]  # (B, D)

        # A_bar_t = exp(delta_t * A)  where A is (D, N), delta_t is (B, D)
        # Broadcast: (B, D, 1) * (D, N) -> (B, D, N)
        dA = delta_t[:, :, None] * A[None, :, :]  # (B, D, N)
        A_bar = np.exp(dA)  # (B, D, N)

        # B_bar_t ≈ delta_t * B_t (first-order approx)
        # delta_t: (B, D), B_t: (B, N)
        # -> (B, D, 1) * (B, 1, N) = (B, D, N)
        B_t = B[:, t, :]  # (B, N)
        dB = delta_t[:, :, None] * B_t[:, None, :]  # (B, D, N)

        # State update: x_t = A_bar_t * x_{t-1} + dB * u_t
        # u_t: (B, D) -> (B, D, 1) for broadcasting
        u_t = u[:, t, :]  # (B, D)
        x = A_bar * x + dB * u_t[:, :, None]  # (B, D, N)

        # Output: y_t = C_t @ x_t
        # C_t: (B, N), x: (B, D, N)
        # -> sum over N: (B, D, N) * (B, 1, N) -> sum -> (B, D)
        C_t = C[:, t, :]  # (B, N)
        y_t = np.sum(x * C_t[:, None, :], axis=-1)  # (B, D)

        if D is not None:
            y_t = y_t + D[None, :] * u_t

        y[:, t, :] = y_t
        states.append(x.copy())

    return y, states


def generate_selectivity_demo(B_size=1, L=12, D=4, N=3, seed=42):
    """
    Generate demo showing how selectivity works in Mamba.
    Shows how different inputs produce different A_bar, B_bar.
    """
    np.random.seed(seed)

    # Simulate a simple input
    u = np.random.randn(B_size, L, D) * 0.5

    # A is learned but fixed (diagonal, typically negative for stability)
    A = -np.ones((D, N)) * np.arange(1, N + 1)[None, :]  # (D, N)

    # Input-dependent projections (simplified)
    # In real Mamba: delta, B, C = linear_projections(u)
    delta = np.abs(np.random.randn(B_size, L, D)) * 0.1 + 0.05  # positive step sizes
    B_input = np.random.randn(B_size, L, N) * 0.3
    C_input = np.random.randn(B_size, L, N) * 0.3
    D_skip = np.ones(D) * 0.1

    # Run selective scan
    y, states = selective_scan_naive(u, delta, A, B_input, C_input, D_skip)

    # Compute A_bar for each timestep to show selectivity
    A_bars = []
    for t in range(L):
        delta_t = delta[0, t, :]  # (D,)
        dA = delta_t[:, None] * A  # (D, N)
        A_bars.append(np.exp(dA).tolist())

    return {
        "B": B_size, "L": L, "D": D, "N": N,
        "input_u": u[0].tolist(),
        "A_diagonal": A.tolist(),
        "delta": delta[0].tolist(),
        "B_input": B_input[0].tolist(),
        "C_input": C_input[0].tolist(),
        "output_y": y[0].tolist(),
        "states": [s[0].tolist() for s in states],
        "A_bars": A_bars,
        "shapes": {
            "u": f"(B={B_size}, L={L}, D={D})",
            "A": f"(D={D}, N={N})",
            "delta": f"(B={B_size}, L={L}, D={D})",
            "B": f"(B={B_size}, L={L}, N={N})",
            "C": f"(B={B_size}, L={L}, N={N})",
            "x": f"(B={B_size}, D={D}, N={N})",
            "y": f"(B={B_size}, L={L}, D={D})",
        }
    }


def generate_mamba_block_info():
    """Return info about the Mamba block architecture."""
    return {
        "layers": [
            {"name": "Input Projection", "in_shape": "(B, L, D)", "out_shape": "(B, L, E)",
             "desc": "Linear projection expanding D to E=2D"},
            {"name": "Conv1D", "in_shape": "(B, L, E)", "out_shape": "(B, L, E)",
             "desc": "1D convolution with kernel_size=4, groups=E (depthwise)"},
            {"name": "SiLU Activation", "in_shape": "(B, L, E)", "out_shape": "(B, L, E)",
             "desc": "SiLU(x) = x * sigmoid(x)"},
            {"name": "SSM Projections", "in_shape": "(B, L, E)", "out_shape": "delta(B,L,E), B(B,L,N), C(B,L,N)",
             "desc": "Input-dependent: s_B(x), s_C(x), s_delta(x)"},
            {"name": "Selective Scan", "in_shape": "(B,L,E) + params", "out_shape": "(B, L, E)",
             "desc": "Core SSM with input-dependent discretization"},
            {"name": "Output Gate", "in_shape": "(B, L, E)", "out_shape": "(B, L, E)",
             "desc": "Element-wise multiply with gating branch: y * SiLU(z)"},
            {"name": "Output Projection", "in_shape": "(B, L, E)", "out_shape": "(B, L, D)",
             "desc": "Linear projection back to model dimension D"},
        ]
    }


def generate_mamba1_execution_steps(L=6, D=3, N=2, seed=42):
    """Generate step-by-step execution for Mamba selective scan."""
    np.random.seed(seed)
    u = np.random.randn(1, L, D) * 0.5
    A = -np.ones((D, N)) * np.arange(1, N + 1)[None, :]
    delta = np.abs(np.random.randn(1, L, D)) * 0.1 + 0.05
    B_in = np.random.randn(1, L, N) * 0.3
    C_in = np.random.randn(1, L, N) * 0.3

    code = [
        {"line": "import torch", "indent": 0},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "def selective_scan(", "indent": 0},
        {"line": f"    u: torch.Tensor,      # ({L}, {D}) input", "indent": 0},
        {"line": f"    A: torch.Tensor,      # ({D}, {N}) state matrix (diagonal, learned)", "indent": 0},
        {"line": f"    delta: torch.Tensor,  # ({L}, {D}) step sizes (input-dependent!)", "indent": 0},
        {"line": f"    B: torch.Tensor,      # ({L}, {N}) input projection (input-dependent!)", "indent": 0},
        {"line": f"    C: torch.Tensor,      # ({L}, {N}) output projection (input-dependent!)", "indent": 0},
        {"line": ") -> torch.Tensor:", "indent": 0},
        {"line": f"x = torch.zeros({D}, {N})", "indent": 1, "comment": "hidden state"},
        {"line": "outputs: list[torch.Tensor] = []", "indent": 1},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": f"for t in range({L}):", "indent": 1},
        {"line": "# Per-timestep ZOH discretization", "indent": 2},
        {"line": "A_bar = torch.exp(delta[t, :, None] * A)", "indent": 2, "comment": f"({D},{N})"},
        {"line": "B_bar = delta[t, :, None] * B[t, None, :]", "indent": 2, "comment": f"({D},{N})"},
        {"line": "# State update (element-wise, selective!)", "indent": 2},
        {"line": "x = A_bar * x + B_bar * u[t, :, None]", "indent": 2, "comment": f"({D},{N})"},
        {"line": "# Output: contract over state dim N", "indent": 2},
        {"line": "y_t = (x * C[t, None, :]).sum(dim=-1)", "indent": 2, "comment": f"({D},)"},
        {"line": "outputs.append(y_t)", "indent": 2},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "return torch.stack(outputs)", "indent": 1, "comment": f"({L}, {D})"},
    ]

    steps = []
    x = np.zeros((D, N))

    steps.append({
        "lineIdx": 9,
        "description": f"Initialize hidden state x as zeros ({D}x{N}). Unlike S4, this state will be updated with INPUT-DEPENDENT parameters at each step.",
        "tensors": {
            "x (state)": {"shape": f"({D},{N})", "value": x.tolist(), "justComputed": True,
                          "annotation": "Hidden state: D feature channels, each with N state dims"},
            "A (fixed)": {"shape": f"({D},{N})", "value": A.tolist(),
                          "annotation": "Diagonal state matrix (learned, shared across time)"},
        }
    })

    for t in range(min(L, 4)):
        dt = delta[0, t, :]
        B_t = B_in[0, t, :]
        C_t = C_in[0, t, :]
        u_t = u[0, t, :]

        # Discretize
        A_bar = np.exp(dt[:, None] * A)
        B_bar = dt[:, None] * B_t[None, :]

        steps.append({
            "lineIdx": 14,
            "description": f"t={t}: SELECTIVE discretization. delta[{t}] = [{', '.join(f'{v:.3f}' for v in dt)}] controls how much to attend. Large delta = 'remember this token', small delta = 'skip it'. A_bar = exp(delta * A).",
            "tensors": {
                "delta[t]": {"shape": f"({D},)", "value": [round(v,4) for v in dt.tolist()],
                             "justComputed": True, "color": "#facc15",
                             "annotation": "Input-dependent step sizes (THE selectivity mechanism)"},
                "A_bar": {"shape": f"({D},{N})", "value": [[round(v,4) for v in row] for row in A_bar.tolist()],
                          "justComputed": True, "color": "#a78bfa",
                          "annotation": "Discretized decay: closer to 1 = more memory retention"},
                "B_bar": {"shape": f"({D},{N})", "value": [[round(v,4) for v in row] for row in B_bar.tolist()],
                          "justComputed": True, "color": "#4ade80",
                          "annotation": "Discretized input matrix (input-dependent!)"},
            }
        })

        # State update
        x_new = A_bar * x + B_bar * u_t[:, None]
        steps.append({
            "lineIdx": 17,
            "description": f"t={t}: State update. x = A_bar * x_prev + B_bar * u[{t}]. The state transition is DIFFERENT at every timestep because A_bar and B_bar depend on the input.",
            "tensors": {
                "A_bar * x_prev": {"shape": f"({D},{N})", "value": [[round(v,4) for v in row] for row in (A_bar * x).tolist()],
                                    "annotation": "Retained state (element-wise product)"},
                "u[t]": {"shape": f"({D},)", "value": [round(v,4) for v in u_t.tolist()], "color": "#22d3ee"},
                "x (updated)": {"shape": f"({D},{N})", "value": [[round(v,4) for v in row] for row in x_new.tolist()],
                                "justComputed": True, "color": "#60a5fa",
                                "annotation": "New state: retained + input contribution"},
            }
        })
        x = x_new

        # Output
        y_t = np.sum(x * C_t[None, :], axis=-1)
        steps.append({
            "lineIdx": 19,
            "description": f"t={t}: Output. y_t = sum(x * C[{t}]) across N={N} state dims. C is also input-dependent, so the 'read' operation adapts to the input.",
            "tensors": {
                "C[t]": {"shape": f"({N},)", "value": [round(v,4) for v in C_t.tolist()],
                         "color": "#fb923c", "annotation": "Input-dependent output projection"},
                "y_t": {"shape": f"({D},)", "value": [round(v,4) for v in y_t.tolist()],
                        "justComputed": True, "color": "#4ade80",
                        "annotation": f"Output at step {t}: weighted read from state"},
            }
        })

    return {"code": code, "steps": steps}

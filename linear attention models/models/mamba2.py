"""
Mamba-2: State Space Duality (SSD)
Paper: "Transformers are SSMs: Generalized Models and Efficient Algorithms
        Through Structured State Space Duality"
       Tri Dao, Albert Gu (2024)
       arXiv: 2405.21060

Key insight: SSMs and structured masked attention are dual views of the same
computation. Restricting A to scalar enables factoring M = L * (C B^T).
"""

import numpy as np


def build_1ss_mask(a_scalars):
    """
    Build the 1-semiseparable (1-SS) mask matrix L.

    L_{ij} = prod_{k=j+1}^{i} a_k   for i >= j
    L_{ij} = 0                         for i < j

    This is the "discount factor" matrix:
        L = [[1,      0,      0,    ...],
             [a_1,    1,      0,    ...],
             [a_2*a_1, a_2,   1,    ...],
             [...                       ]]

    Args:
        a_scalars: (T,) scalar decay factors per timestep

    Returns:
        L: (T, T) lower-triangular 1-SS mask
    """
    T = len(a_scalars)
    L = np.zeros((T, T))

    for i in range(T):
        for j in range(i + 1):
            if i == j:
                L[i, j] = 1.0
            else:
                # Product of a_{j+1} * a_{j+2} * ... * a_i
                L[i, j] = np.prod(a_scalars[j + 1:i + 1])

    return L


def build_semiseparable_matrix(A_scalars, B, C):
    """
    Build the full N-semiseparable matrix M.

    M_{ij} = C_i^T * (A_{i} * A_{i-1} * ... * A_{j+1}) * B_j   for i >= j
    M_{ij} = 0                                                     for i < j

    When A is scalar, this factors as: M = L * (C @ B^T)
    where * is element-wise (Hadamard) product.

    Args:
        A_scalars: (T,) scalar state transitions
        B: (T, N) input projections (= "keys")
        C: (T, N) output projections (= "queries")

    Returns:
        M: (T, T) semiseparable matrix
    """
    T = len(A_scalars)
    L = build_1ss_mask(A_scalars)

    # C @ B^T -> (T, T)
    CB = C @ B.T  # (T, N) @ (N, T) = (T, T)

    # Element-wise product
    M = L * CB

    return M, L, CB


def ssd_quadratic_form(A_scalars, B, C, X):
    """
    Quadratic (attention-like) form of SSD.

    Y = M @ X
    where M = L * (C @ B^T)

    This is O(T^2) in sequence length - good for short sequences.

    Args:
        A_scalars: (T,) scalar state transitions
        B: (T, N) keys
        C: (T, N) queries
        X: (T, P) values

    Returns:
        Y: (T, P) output
        intermediates: dict of intermediate tensors
    """
    T, P = X.shape
    N = B.shape[1]

    L = build_1ss_mask(A_scalars)
    CB = C @ B.T  # (T, T)
    M = L * CB    # (T, T)
    Y = M @ X     # (T, P)

    return Y, {
        "L": L.tolist(),
        "CB": CB.tolist(),
        "M": M.tolist(),
        "Y": Y.tolist(),
    }


def ssd_recurrent_form(A_scalars, B, C, X):
    """
    Linear (recurrent) form of SSD.

    h_t = a_t * h_{t-1} + B_t * X_t^T    # h_t is (N, P)
    y_t = C_t^T @ h_t                      # y_t is (P,)

    This is O(T * N * P) - good for long sequences.

    Args:
        A_scalars: (T,) scalar state transitions
        B: (T, N) keys
        C: (T, N) queries
        X: (T, P) values

    Returns:
        Y: (T, P) output
        states: list of (N, P) hidden states
    """
    T, P = X.shape
    N = B.shape[1]

    h = np.zeros((N, P))
    Y = np.zeros((T, P))
    states = []

    for t in range(T):
        # State update: h_t = a_t * h_{t-1} + B_t outer X_t
        h = A_scalars[t] * h + np.outer(B[t], X[t])  # (N, P)
        states.append(h.copy())

        # Output: y_t = C_t^T @ h_t
        Y[t] = C[t] @ h  # (N,) @ (N, P) = (P,)

    return Y, states


def ssd_chunked_form(A_scalars, B, C, X, chunk_size=4):
    """
    Chunked SSD algorithm (the actual Mamba-2 algorithm).

    Splits sequence into chunks of size Q:
    1. Intra-chunk: quadratic form (matmul, O(Q^2))
    2. Inter-chunk: recurrent state passing

    Args:
        A_scalars: (T,) scalar state transitions
        B: (T, N) keys
        C: (T, N) queries
        X: (T, P) values
        chunk_size: Q, the chunk size

    Returns:
        Y: (T, P) output
        chunk_info: dict with per-chunk details
    """
    T, P = X.shape
    N = B.shape[1]
    Q = chunk_size
    num_chunks = (T + Q - 1) // Q

    # Pad if needed
    pad = num_chunks * Q - T
    if pad > 0:
        A_scalars = np.concatenate([A_scalars, np.ones(pad)])
        B = np.vstack([B, np.zeros((pad, N))])
        C = np.vstack([C, np.zeros((pad, N))])
        X = np.vstack([X, np.zeros((pad, P))])

    Y = np.zeros((num_chunks * Q, P))
    chunk_details = []
    running_state = np.zeros((N, P))  # inter-chunk state

    for c in range(num_chunks):
        start = c * Q
        end = start + Q

        A_chunk = A_scalars[start:end]
        B_chunk = B[start:end]
        C_chunk = C[start:end]
        X_chunk = X[start:end]

        # Step 1: Intra-chunk (quadratic form, ignoring initial state)
        L_chunk = build_1ss_mask(A_chunk)
        CB_chunk = C_chunk @ B_chunk.T
        M_chunk = L_chunk * CB_chunk
        Y_intra = M_chunk @ X_chunk

        # Step 2: Compute chunk's contribution to state
        # Final state of this chunk (for passing to next)
        chunk_state = np.zeros((N, P))
        h = np.zeros((N, P))
        for t in range(Q):
            h = A_chunk[t] * h + np.outer(B_chunk[t], X_chunk[t])
        chunk_final_state = h

        # Step 3: Apply initial state contribution
        # Y_off[t] = C_t^T @ (a_{t} * a_{t-1} * ... * a_{0}) @ running_state
        decay_cumulative = np.ones(Q)
        for t in range(Q):
            decay_cumulative[t] = np.prod(A_chunk[:t + 1])

        Y_off = np.zeros((Q, P))
        for t in range(Q):
            Y_off[t] = decay_cumulative[t] * (C_chunk[t] @ running_state)

        # Combine
        Y_chunk = Y_intra + Y_off
        Y[start:end] = Y_chunk

        # Update running state for next chunk
        decay_full = np.prod(A_chunk)
        running_state = decay_full * running_state + chunk_final_state

        chunk_details.append({
            "chunk_idx": c,
            "L_chunk": L_chunk.tolist(),
            "M_chunk": M_chunk.tolist(),
            "Y_intra": Y_intra.tolist(),
            "Y_off": Y_off.tolist(),
            "chunk_state_norm": float(np.linalg.norm(chunk_final_state)),
        })

    return Y[:T], chunk_details


def generate_ssd_demo(T=8, N=3, P=2, seed=42):
    """Generate demo data for SSD visualization."""
    np.random.seed(seed)

    # Parameters
    A_scalars = np.random.uniform(0.8, 0.99, T)  # decay close to 1
    B = np.random.randn(T, N) * 0.3
    C = np.random.randn(T, N) * 0.3
    X = np.random.randn(T, P) * 0.5

    # Three computation modes
    Y_quad, quad_inter = ssd_quadratic_form(A_scalars, B, C, X)
    Y_rec, rec_states = ssd_recurrent_form(A_scalars, B, C, X)
    Y_chunk, chunk_info = ssd_chunked_form(A_scalars, B, C, X, chunk_size=4)

    return {
        "T": T, "N": N, "P": P,
        "A_scalars": A_scalars.tolist(),
        "B": B.tolist(),
        "C": C.tolist(),
        "X": X.tolist(),
        "Y_quadratic": Y_quad.tolist(),
        "Y_recurrent": Y_rec.tolist(),
        "Y_chunked": Y_chunk.tolist(),
        "quadratic_intermediates": quad_inter,
        "recurrent_states": [s.tolist() for s in rec_states],
        "chunk_info": chunk_info,
        "max_diff_quad_rec": float(np.max(np.abs(Y_quad - Y_rec))),
        "max_diff_quad_chunk": float(np.max(np.abs(Y_quad - Y_chunk))),
        "shapes": {
            "A": f"(T={T},)  [scalar per timestep]",
            "B": f"(T={T}, N={N})  [keys]",
            "C": f"(T={T}, N={N})  [queries]",
            "X": f"(T={T}, P={P})  [values]",
            "L": f"(T={T}, T={T})  [1-SS mask]",
            "M": f"(T={T}, T={T})  [semiseparable matrix]",
            "h": f"(N={N}, P={P})  [hidden state]",
            "Y": f"(T={T}, P={P})  [output]",
        }
    }


def generate_mamba2_execution_steps(T=6, N=2, P=2, seed=42):
    """Generate step-by-step execution for SSD quadratic form."""
    np.random.seed(seed)
    A_scalars = np.random.uniform(0.85, 0.95, T)
    B = np.random.randn(T, N) * 0.3
    C = np.random.randn(T, N) * 0.3
    X = np.random.randn(T, P) * 0.5

    code = [
        {"line": "import torch", "indent": 0},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "def ssd_quadratic(", "indent": 0},
        {"line": f"    A: torch.Tensor,  # ({T},) scalar decay per timestep", "indent": 0},
        {"line": f"    B: torch.Tensor,  # ({T}, {N}) keys", "indent": 0},
        {"line": f"    C: torch.Tensor,  # ({T}, {N}) queries", "indent": 0},
        {"line": f"    X: torch.Tensor,  # ({T}, {P}) values", "indent": 0},
        {"line": ") -> torch.Tensor:", "indent": 0},
        {"line": "# Build 1-semiseparable mask (cumulative decay)", "indent": 1},
        {"line": "L = build_1ss_mask(A)", "indent": 1, "comment": f"({T}, {T}) lower-tri decay"},
        {"line": "# Attention scores: like Q @ K.T in transformers", "indent": 1},
        {"line": f"G = C @ B.T", "indent": 1, "comment": f"({T},{N}) @ ({N},{T}) -> ({T},{T})"},
        {"line": "# Apply structured decay mask (replaces softmax!)", "indent": 1},
        {"line": "M = L * G", "indent": 1, "comment": f"({T},{T}) Hadamard product"},
        {"line": "# Output: like attention_weights @ V", "indent": 1},
        {"line": "Y = M @ X", "indent": 1, "comment": f"({T},{T}) @ ({T},{P}) -> ({T},{P})"},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "return Y", "indent": 1, "comment": f"torch.Tensor ({T}, {P})"},
    ]

    steps = []
    L_mask = build_1ss_mask(A_scalars)
    G = C @ B.T
    M = L_mask * G
    Y = M @ X

    r = lambda arr: [[round(v,3) for v in row] for row in arr] if hasattr(arr[0], '__len__') else [round(v,3) for v in arr]

    steps.append({
        "lineIdx": 9,
        "description": f"Build the 1-semiseparable mask L ({T}x{T}). L[i,j] = product of decay scalars a_{{j+1}} * ... * a_i. This is the 'discount factor' - how much past tokens are attenuated. Diagonal = 1 (no decay for self), below diagonal = exponential decay.",
        "tensors": {
            "A (scalars)": {"shape": f"({T},)", "value": r(A_scalars), "color": "#facc15",
                            "annotation": "Scalar decay per timestep (close to 1 = slow decay)"},
            "L (1-SS mask)": {"shape": f"({T},{T})", "value": r(L_mask), "justComputed": True,
                              "annotation": "Lower-triangular mask: L[i,j] = cumulative decay from j to i"},
        }
    })

    steps.append({
        "lineIdx": 11,
        "description": f"Compute G = C @ B^T ({T}x{T}). This is like QK^T in standard attention! C plays the role of queries, B plays the role of keys. Each G[i,j] = dot(C[i], B[j]) measures similarity.",
        "tensors": {
            "C (queries)": {"shape": f"({T},{N})", "value": r(C), "color": "#a78bfa"},
            "B (keys)": {"shape": f"({T},{N})", "value": r(B), "color": "#22d3ee"},
            "G = CB^T": {"shape": f"({T},{T})", "value": r(G), "justComputed": True,
                         "annotation": "Attention scores (like QK^T in Transformers!)"},
        }
    })

    steps.append({
        "lineIdx": 13,
        "description": f"Apply structured mask: M = L * G (element-wise). This is the KEY difference from standard attention: instead of softmax, we multiply by the decay mask L. Tokens far apart get attenuated by cumulative decay.",
        "tensors": {
            "L (mask)": {"shape": f"({T},{T})", "value": r(L_mask),
                         "annotation": "Decay structure (replaces softmax!)"},
            "G (scores)": {"shape": f"({T},{T})", "value": r(G)},
            "M = L * G": {"shape": f"({T},{T})", "value": r(M), "justComputed": True, "color": "#60a5fa",
                          "annotation": "Masked attention matrix: M = L ⊙ (CB^T)"},
        }
    })

    steps.append({
        "lineIdx": 15,
        "description": f"Compute output Y = M @ X ({T}x{T}) @ ({T}x{P}) -> ({T}x{P}). Just like attention output = attention_weights @ values. Each output row is a weighted combination of value vectors.",
        "tensors": {
            "M (attention)": {"shape": f"({T},{T})", "value": r(M)},
            "X (values)": {"shape": f"({T},{P})", "value": r(X), "color": "#4ade80"},
            "Y = M @ X": {"shape": f"({T},{P})", "value": r(Y), "justComputed": True, "color": "#fb923c",
                          "annotation": "Final output: weighted sum of values using masked attention"},
        }
    })

    return {"code": code, "steps": steps}

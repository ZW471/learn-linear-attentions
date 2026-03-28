"""
DeltaNet: Linear Attention with the Delta Update Rule
Papers:
  - "Linear Transformers Are Secretly Fast Weight Programmers"
    Schlag, Irie, Schmidhuber (2021), arXiv: 2102.11174
  - "Gated Delta Networks: Improving Mamba2 with Delta Rule"
    Yang, Kautz, Hatamizadeh (2024)

Key insight: Standard linear attention accumulates associations but never
removes old ones. The delta rule subtracts the old value before writing
the new one, enabling proper memory management.
"""

import numpy as np


def standard_attention(Q, K, V):
    """
    Standard softmax attention for comparison.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q: (T, d) queries
        K: (T, d) keys
        V: (T, d_v) values

    Returns:
        Y: (T, d_v) output
        attn_weights: (T, T) attention matrix
    """
    T, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)  # (T, T)

    # Causal mask
    mask = np.triu(np.ones((T, T)) * (-1e9), k=1)
    scores = scores + mask

    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    Y = attn_weights @ V
    return Y, attn_weights


def linear_attention_quadratic(Q, K, V):
    """
    Linear attention (quadratic form, for comparison).

    y_t = (sum_{s<=t} phi(Q_t) phi(K_s)^T V_s) / (sum_{s<=t} phi(Q_t) phi(K_s)^T)

    Simplified (without normalization, using identity feature map):
    Y = tril(Q @ K^T) @ V

    Args:
        Q: (T, d) queries
        K: (T, d) keys
        V: (T, d_v) values

    Returns:
        Y: (T, d_v) output
        M: (T, T) attention-like matrix
    """
    T = Q.shape[0]
    QK = Q @ K.T  # (T, T)
    causal_mask = np.tril(np.ones((T, T)))
    M = QK * causal_mask
    Y = M @ V
    return Y, M


def linear_attention_recurrent(Q, K, V):
    """
    Linear attention (recurrent form).

    S_t = S_{t-1} + V_t @ K_t^T     # (d_v, d) state accumulation
    y_t = S_t @ Q_t                   # (d_v,) output

    Note: S is a (d_v, d) matrix that accumulates outer products.
    This NEVER forgets -- it only adds new associations.

    Args:
        Q: (T, d) queries
        K: (T, d) keys
        V: (T, d_v) values

    Returns:
        Y: (T, d_v) output
        states: list of (d_v, d) memory matrices
    """
    T, d = Q.shape
    d_v = V.shape[1]

    S = np.zeros((d_v, d))
    Y = np.zeros((T, d_v))
    states = []

    for t in range(T):
        # Accumulate: write new association
        S = S + np.outer(V[t], K[t])  # (d_v, d)
        states.append(S.copy())

        # Read
        Y[t] = S @ Q[t]  # (d_v,)

    return Y, states


def delta_rule_recurrent(Q, K, V, beta=None):
    """
    Delta rule linear attention.

    The key innovation: before writing a new association, subtract the old one.

    S_t = S_{t-1} + beta_t * (V_t - S_{t-1} @ K_t) @ K_t^T

    Equivalently:
    S_t = (I - beta_t * K_t @ K_t^T) @ S_{t-1} + beta_t * V_t @ K_t^T
    S_t = S_{t-1} - beta_t * (S_{t-1} @ K_t) @ K_t^T + beta_t * V_t @ K_t^T

    Interpretation:
    - (S_{t-1} @ K_t): retrieve what's currently stored for key K_t
    - V_t - (S_{t-1} @ K_t): the "delta" = new value minus old value
    - Write the delta, not the raw value

    Args:
        Q: (T, d) queries (normalized)
        K: (T, d) keys (normalized)
        V: (T, d_v) values
        beta: (T,) learning rates, or None for all 1.0

    Returns:
        Y: (T, d_v) output
        states: list of (d_v, d) memory matrices
        deltas: list of delta vectors for visualization
    """
    T, d = Q.shape
    d_v = V.shape[1]

    if beta is None:
        beta = np.ones(T)

    S = np.zeros((d_v, d))
    Y = np.zeros((T, d_v))
    states = []
    deltas = []

    for t in range(T):
        # Retrieve current value for this key
        retrieved = S @ K[t]  # (d_v,) - what's currently stored

        # Compute delta: difference between new and old
        delta = V[t] - retrieved  # (d_v,)
        deltas.append({
            "retrieved": retrieved.tolist(),
            "new_value": V[t].tolist(),
            "delta": delta.tolist(),
            "beta": float(beta[t]),
        })

        # Update: write the delta (not the raw value)
        S = S + beta[t] * np.outer(delta, K[t])  # (d_v, d)
        states.append(S.copy())

        # Read
        Y[t] = S @ Q[t]  # (d_v,)

    return Y, states, deltas


def generate_delta_net_demo(T=8, d=4, d_v=3, seed=42):
    """
    Generate comparison demo: standard linear attention vs delta rule.
    Uses a scenario where delta rule clearly helps.
    """
    np.random.seed(seed)

    # Create keys that repeat (to show delta rule's advantage)
    # Keys 0 and 4 are similar, keys 1 and 5 are similar, etc.
    K_base = np.random.randn(T // 2, d)
    K = np.vstack([K_base, K_base + np.random.randn(T // 2, d) * 0.1])

    # Normalize keys
    K = K / np.linalg.norm(K, axis=1, keepdims=True)

    # Values change over time (same key, different value = need to update)
    V = np.random.randn(T, d_v) * 0.5

    # Queries
    Q = np.random.randn(T, d)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

    # Learning rates
    beta = np.ones(T) * 0.8

    # Standard linear attention
    Y_linear, states_linear = linear_attention_recurrent(Q, K, V)

    # Delta rule
    Y_delta, states_delta, deltas = delta_rule_recurrent(Q, K, V, beta)

    # Standard softmax attention (for reference)
    Y_softmax, attn_weights = standard_attention(Q, K, V)

    return {
        "T": T, "d": d, "d_v": d_v,
        "Q": Q.tolist(),
        "K": K.tolist(),
        "V": V.tolist(),
        "beta": beta.tolist(),
        # Outputs
        "Y_linear": Y_linear.tolist(),
        "Y_delta": Y_delta.tolist(),
        "Y_softmax": Y_softmax.tolist(),
        # States
        "states_linear": [s.tolist() for s in states_linear],
        "states_delta": [s.tolist() for s in states_delta],
        # Delta details
        "deltas": deltas,
        # Attention weights (softmax only)
        "attn_weights": attn_weights.tolist(),
        # Shapes
        "shapes": {
            "Q": f"(T={T}, d={d})",
            "K": f"(T={T}, d={d})",
            "V": f"(T={T}, d_v={d_v})",
            "S": f"(d_v={d_v}, d={d})  [memory matrix]",
            "Y": f"(T={T}, d_v={d_v})",
            "beta": f"(T={T},)  [learning rates]",
        }
    }


def generate_delta_execution_steps(T=6, d=3, d_v=2, seed=42):
    """Generate step-by-step execution for delta rule attention."""
    np.random.seed(seed)

    K = np.random.randn(T, d) * 0.5
    K = K / np.linalg.norm(K, axis=1, keepdims=True)
    V = np.random.randn(T, d_v) * 0.5
    Q = np.random.randn(T, d) * 0.5
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    beta = np.ones(T) * 0.8

    code = [
        {"line": "import torch", "indent": 0},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "def delta_rule_attention(", "indent": 0},
        {"line": f"    Q: torch.Tensor,     # ({T}, {d}) queries (normalized)", "indent": 0},
        {"line": f"    K: torch.Tensor,     # ({T}, {d}) keys (L2-normalized)", "indent": 0},
        {"line": f"    V: torch.Tensor,     # ({T}, {d_v}) values", "indent": 0},
        {"line": f"    beta: torch.Tensor,  # ({T},) learning rates in (0,1)", "indent": 0},
        {"line": ") -> torch.Tensor:", "indent": 0},
        {"line": f"S = torch.zeros({d_v}, {d})", "indent": 1, "comment": f"memory matrix ({d_v}x{d})"},
        {"line": "outputs: list[torch.Tensor] = []", "indent": 1},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": f"for t in range({T}):", "indent": 1},
        {"line": "# Retrieve what's currently stored for this key", "indent": 2},
        {"line": "retrieved = torch.mv(S, K[t])", "indent": 2, "comment": f"({d_v},{d}) @ ({d},) -> ({d_v},)"},
        {"line": "# Compute error: new value minus old value", "indent": 2},
        {"line": "delta = V[t] - retrieved", "indent": 2, "comment": f"({d_v},) - ({d_v},) -> ({d_v},)"},
        {"line": "# Update memory: erase old, write new", "indent": 2},
        {"line": "S = S + beta[t] * torch.outer(delta, K[t])", "indent": 2, "comment": f"({d_v},{d})"},
        {"line": "# Read from updated memory with query", "indent": 2},
        {"line": "y_t = torch.mv(S, Q[t])", "indent": 2, "comment": f"({d_v},{d}) @ ({d},) -> ({d_v},)"},
        {"line": "outputs.append(y_t)", "indent": 2},
        {"line": "", "indent": 0, "isBlank": True},
        {"line": "return torch.stack(outputs)", "indent": 1, "comment": f"({T}, {d_v})"},
    ]

    steps = []
    S = np.zeros((d_v, d))
    outputs = []

    # Init
    steps.append({
        "lineIdx": 8,
        "description": f"Initialize memory matrix S as zeros ({d_v}x{d}). This matrix stores key-value associations. Think of it as a lookup table that maps d-dimensional keys to d_v-dimensional values.",
        "tensors": {
            "S (memory)": {"shape": f"({d_v},{d})", "value": S.tolist(), "justComputed": True,
                           "annotation": "Empty memory - no associations stored yet"},
            "K (keys)": {"shape": f"({T},{d})", "value": [[round(v,3) for v in row] for row in K.tolist()],
                         "annotation": "All keys (L2-normalized)"},
            "V (values)": {"shape": f"({T},{d_v})", "value": [[round(v,3) for v in row] for row in V.tolist()],
                           "annotation": "All values to store"},
        }
    })

    for t in range(min(T, 4)):  # Show first 4 steps
        k_t = K[t]
        v_t = V[t]
        q_t = Q[t]
        b_t = beta[t]

        # Step 1: Retrieve
        retrieved = S @ k_t
        steps.append({
            "lineIdx": 13,
            "description": f"t={t}: Retrieve what's currently stored for key k_{t}. Multiply memory S ({d_v}x{d}) by key k_{t} ({d},). If this key was stored before, we get back the old value. If memory is empty/orthogonal, we get ~zeros.",
            "tensors": {
                "k_t": {"shape": f"({d},)", "value": [round(v,4) for v in k_t.tolist()],
                        "justComputed": True, "color": "#22d3ee",
                        "annotation": f"Key at step {t} (normalized)"},
                "S (memory)": {"shape": f"({d_v},{d})", "value": [[round(v,3) for v in row] for row in S.tolist()]},
                "retrieved": {"shape": f"({d_v},)", "value": [round(v,4) for v in retrieved.tolist()],
                              "justComputed": True, "color": "#fb923c",
                              "annotation": "What the memory currently returns for this key"},
            }
        })

        # Step 2: Compute delta
        delta = v_t - retrieved
        steps.append({
            "lineIdx": 15,
            "description": f"t={t}: Compute the delta = (new value) - (old retrieved value). This is the ERROR between what we want to store and what's currently stored. Large delta = big correction needed.",
            "tensors": {
                "V[t] (new)": {"shape": f"({d_v},)", "value": [round(v,4) for v in v_t.tolist()],
                               "color": "#4ade80", "annotation": "What we WANT to store"},
                "retrieved (old)": {"shape": f"({d_v},)", "value": [round(v,4) for v in retrieved.tolist()],
                                    "color": "#fb923c", "annotation": "What's CURRENTLY stored"},
                "delta": {"shape": f"({d_v},)", "value": [round(v,4) for v in delta.tolist()],
                          "justComputed": True, "color": "#f87171",
                          "annotation": "Correction needed: new - old"},
            }
        })

        # Step 3: Update
        update = b_t * np.outer(delta, k_t)
        S_new = S + update
        steps.append({
            "lineIdx": 17,
            "description": f"t={t}: Update memory: S = S + beta * outer(delta, k). beta={b_t} controls update strength. The outer product creates a ({d_v}x{d}) update matrix that corrects the association for this key.",
            "tensors": {
                "beta": {"shape": "scalar", "value": round(b_t, 2), "color": "#facc15"},
                "outer(delta, k)": {"shape": f"({d_v},{d})", "value": [[round(v,4) for v in row] for row in np.outer(delta, k_t).tolist()],
                                     "annotation": "Rank-1 update to memory"},
                "S (updated)": {"shape": f"({d_v},{d})", "value": [[round(v,3) for v in row] for row in S_new.tolist()],
                                "justComputed": True, "color": "#60a5fa",
                                "annotation": "Memory after delta correction"},
            }
        })
        S = S_new

        # Step 4: Read
        y_t = S @ q_t
        outputs.append(y_t)
        steps.append({
            "lineIdx": 19,
            "description": f"t={t}: Read from updated memory using query q_{t}. Output y_{t} = S @ q_{t}. The quality of this output depends on how well S has learned the key-value associations so far.",
            "tensors": {
                "q_t": {"shape": f"({d},)", "value": [round(v,4) for v in q_t.tolist()],
                        "color": "#a78bfa", "annotation": f"Query at step {t}"},
                "y_t": {"shape": f"({d_v},)", "value": [round(v,4) for v in y_t.tolist()],
                        "justComputed": True, "color": "#4ade80",
                        "annotation": f"Output: memory read with query"},
                "S (memory)": {"shape": f"({d_v},{d})", "value": [[round(v,3) for v in row] for row in S.tolist()],
                               "annotation": f"Memory state after {t+1} updates"},
            }
        })

    return {"code": code, "steps": steps}

# QUBO modularity formulations from Wang et al. 2024
# (Quantum Computing in Community Detection for Anti-Fraud Applications, Entropy).
#
# Two formulations:
#   - gen_q_dict_modularity_2community(G):    eq. (6), single-bit-per-node, k=2
#   - gen_q_dict_modularity_multi(G, k, P):  eq. (5), one-hot-per-node, k>=2
#
# The multi formulation linearises the (vertex, community) double index into a
# single global index i = v*k + c (zero-based). Total problem size N = n*k.

from collections import defaultdict
import numpy as np
import networkx as nx


def linearize_index(v, c, k):
    """Global index for the (vertex v, community c) variable, given k communities."""
    return v * k + c


def delinearize_index(i, k):
    """Inverse of linearize_index: returns (v, c) from global index i."""
    return i // k, i % k


def _modularity_matrix(nx_G):
    """Return (n, B, m) where B is the n*n Newman modularity matrix and m = |E|."""
    nodes = list(nx_G.nodes())
    n = len(nodes)
    A = nx.to_numpy_array(nx_G, nodelist=nodes, weight=None)
    deg = A.sum(axis=1)
    m = nx_G.number_of_edges()
    two_m = 2.0 * m
    B = A - np.outer(deg, deg) / two_m
    return n, B, m


def gen_q_dict_modularity_2community(nx_G):
    """QUBO for the simplified 2-community modularity (Wang et al., eq. (6) collapsed).

    H(x) = -sum_{v,w} B_{vw} x_v x_w  ==>  Q = -B (full symmetric, both halves stored).
    """
    n, B, _ = _modularity_matrix(nx_G)
    Q = -B
    Q_dic = defaultdict(int)
    for i in range(n):
        for j in range(n):
            val = float(Q[i, j])
            if val != 0.0:
                Q_dic[(i, j)] = val
    return Q_dic


def gen_q_dict_modularity_multi(nx_G, k, P=None):
    """QUBO for k-community modularity with one-hot constraint (Wang et al., eq. (5)).

        min_X  -1/(2m) sum_c sum_{v,w} B_{vw} x_{v,c} x_{w,c}
               + P sum_v (sum_c x_{v,c} - 1)^2

    Linearisation: i = v*k + c, total size N = n*k. Returned dict has upper-triangular
    storage (i <= j) so x.T Q x with a dense Q_mat reproduces the cost exactly.

    Parameters
    ----------
    nx_G : networkx graph
    k    : number of communities (>= 2)
    P    : penalty weight; if None, set to 10 * max(|B|).

    Returns
    -------
    Q_dic : defaultdict(int) mapping (i, j) -> float, with i <= j.
    """
    if k < 2:
        raise ValueError(f'k must be >= 2, got {k}')

    n, B, m = _modularity_matrix(nx_G)
    two_m = 2.0 * m

    if P is None:
        P = 10.0 * float(np.max(np.abs(B)))

    Q_dic = defaultdict(int)

    # Linear (diagonal) terms: modularity self-term + constraint linear part.
    # (sum_c x_{v,c} - 1)^2 = sum_c x_{v,c} (since x in {0,1} -> x^2=x)
    #                        + 2 sum_{c1<c2} x_{v,c1} x_{v,c2}
    #                        - 2 sum_c x_{v,c} + 1
    # Linear coefficient on x_{v,c}: P * (1 - 2) = -P.
    for v in range(n):
        Bvv = float(B[v, v])
        for c in range(k):
            i = linearize_index(v, c, k)
            Q_dic[(i, i)] += -Bvv / two_m       # modularity self-loop
            Q_dic[(i, i)] += -P                  # constraint linear

    # Quadratic modularity: same community c, distinct vertices v < w.
    # Coefficient on x_{v,c} x_{w,c} from -1/(2m) * 2 * B_{vw} = -B_{vw}/m.
    for v in range(n):
        for w in range(v + 1, n):
            Bvw = float(B[v, w])
            if Bvw == 0.0:
                continue
            coef = -Bvw / m
            for c in range(k):
                i = linearize_index(v, c, k)
                j = linearize_index(w, c, k)
                a, b = (i, j) if i <= j else (j, i)
                Q_dic[(a, b)] += coef

    # Quadratic constraint: same vertex v, distinct communities c1 < c2.
    # Coefficient 2P on x_{v,c1} x_{v,c2}.
    for v in range(n):
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                i = linearize_index(v, c1, k)
                j = linearize_index(v, c2, k)
                Q_dic[(i, j)] += 2.0 * P

    # Drop zeros that may have cancelled (e.g. modular self-term = 0 on isolated nodes)
    for key in list(Q_dic.keys()):
        if Q_dic[key] == 0:
            del Q_dic[key]

    return Q_dic


# ============================================================================
# Unit test (run as `python qubo_modularity.py`)
# ============================================================================
if __name__ == '__main__':
    import itertools

    print('=' * 70)
    print('Unit test 1: Karate, k=2, structural checks')
    print('=' * 70)
    G = nx.karate_club_graph()
    n, B, m = _modularity_matrix(G)
    P_default = 10.0 * float(np.max(np.abs(B)))
    print(f'n={n}, m={m}, max|B|={np.max(np.abs(B)):.4f}, default P={P_default:.4f}')

    Q = gen_q_dict_modularity_multi(G, k=2)
    N_expected = n * 2
    keys_max = max(max(i, j) for (i, j) in Q.keys())
    print(f'Expected size N = n*k = {N_expected}')
    print(f'Max linearised index seen in Q: {keys_max} (must be < {N_expected})')
    assert keys_max < N_expected, 'index out of bounds'
    print(f'#nonzero entries in Q: {len(Q)}')

    # Every (i, j) in Q must have i <= j (upper-triangular storage).
    bad = [(i, j) for (i, j) in Q.keys() if i > j]
    assert not bad, f'lower-triangular leak: {bad[:3]}'
    print('Upper-triangular storage: OK')

    # Coefficient sanity check.
    # For v=0, c1=0, c2=1: i=0, j=1. Contribution comes from constraint only
    # (different communities, same vertex) -> 2P.
    print(f'Q[(0, 1)] (constraint pair, expect ~2P={2*P_default:.4f}) = {Q[(0, 1)]:.4f}')
    # For v=0, w=1, c=0: i=0, j=2. Modularity coef = -B[0,1]/m.
    # Plus possible constraint contribution? No: different vertex, same community -> only modularity.
    expected_mod = -B[0, 1] / m
    print(f'Q[(0, 2)] (mod pair v=0,w=1,c=0; expect -B[0,1]/m={expected_mod:.4f}) = {Q[(0, 2)]:.4f}')
    # Diagonal at i=0 (v=0, c=0): -B[0,0]/(2m) + (-P)
    expected_diag0 = -B[0, 0] / (2 * m) - P_default
    print(f'Q[(0, 0)] (diag; expect {expected_diag0:.4f}) = {Q[(0, 0)]:.4f}')
    assert abs(Q[(0, 0)] - expected_diag0) < 1e-10
    assert abs(Q[(0, 2)] - expected_mod) < 1e-10
    assert abs(Q[(0, 1)] - 2 * P_default) < 1e-10
    print('Coefficient values: OK')

    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('Unit test 2: Karate, k=2, equivalence with gen_q_dict_modularity_2community')
    print('=' * 70)
    Q2 = gen_q_dict_modularity_2community(G)

    # Take ground-truth labels: Mr. Hi -> 0, Officer -> 1.
    nodes = list(G.nodes())
    club_to_label = {'Mr. Hi': 0, 'Officer': 1}
    labels = np.array([club_to_label[G.nodes[v]['club']] for v in nodes])

    # Multi cost at ground truth.
    X = np.zeros((n, 2))
    for v_idx, lbl in enumerate(labels):
        X[v_idx, lbl] = 1
    x_flat = X.reshape(-1)

    def eval_qubo(Qd, x):
        s = 0.0
        for (i, j), val in Qd.items():
            s += val * x[i] * x[j]
        return s

    H_multi = eval_qubo(Q, x_flat)

    # 2community cost at ground-truth y (= column-0 of X).
    y = X[:, 0]
    H_2c = eval_qubo(Q2, y)

    # Why these don't match by a simple ratio: the QUBO-encoded constraint
    # contributes -n*P when the one-hot constraint is satisfied (we dropped the
    # +1-per-vertex constant from the (sum_c x - 1)^2 expansion, since constants
    # don't affect argmin). The true relation, after adding back that constant:
    #     H_multi + n*P  ==  (1/m) * H_2community
    # Both formulations therefore have aligned argmins.
    P_used = 10.0 * float(np.max(np.abs(B)))
    H_multi_shifted = H_multi + n * P_used
    print(f'H_multi(ground truth)            = {H_multi:.6f}')
    print(f'H_multi + n*P (constant-shifted) = {H_multi_shifted:.6f}')
    print(f'H_2community(ground truth)       = {H_2c:.6f}')
    print(f'(H_multi + n*P) / H_2c           = {H_multi_shifted / H_2c:.6f}  (expected 1/m = {1/m:.6f})')
    assert abs(H_multi_shifted / H_2c - 1.0 / m) < 1e-9, (
        'after constant shift, cost must be (1/m) * H_2community'
    )
    print('Cost equivalence (up to additive n*P, multiplicative 1/m): OK')

    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('Unit test 3: small graph 5 nodes, exhaustive argmin agreement')
    print('=' * 70)
    Gs = nx.Graph()
    Gs.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])  # triangle + tail
    n_s, B_s, m_s = _modularity_matrix(Gs)
    Q_multi_s = gen_q_dict_modularity_multi(Gs, k=2)
    Q_2c_s = gen_q_dict_modularity_2community(Gs)

    best_2c, best_2c_y = (1e18, None)
    for bits in itertools.product([0, 1], repeat=n_s):
        v = np.array(bits, dtype=float)
        c = eval_qubo(Q_2c_s, v)
        if c < best_2c:
            best_2c = c
            best_2c_y = bits

    best_multi, best_multi_assn = (1e18, None)
    # Only enumerate one-hot assignments (constraint satisfied) for k=2.
    for bits in itertools.product([0, 1], repeat=n_s):
        Xs = np.zeros((n_s, 2))
        for v_idx, lbl in enumerate(bits):
            Xs[v_idx, lbl] = 1
        c = eval_qubo(Q_multi_s, Xs.reshape(-1))
        if c < best_multi:
            best_multi = c
            best_multi_assn = bits

    print(f'2community optimum: y={best_2c_y}, H={best_2c:.4f}')
    print(f'multi(k=2) optimum: a={best_multi_assn}, H={best_multi:.4f}')
    # Optimum bitstrings should match (or be the bit-flipped pair, since the
    # 2community formulation has 0<->1 symmetry and multi has c=0<->c=1 symmetry).
    flipped = tuple(1 - b for b in best_2c_y)
    assert best_multi_assn == best_2c_y or best_multi_assn == flipped, (
        f'argmin mismatch: 2c={best_2c_y}, multi={best_multi_assn}'
    )
    print('argmin bitstrings agree (up to label-flip symmetry): OK')

    print()
    print('All tests passed.')

'''
Statistical utilities for the scalar MMSuSiE workflows: single-effect Bayes
factors, PIP, credible-set construction/purity, and sparse block assembly.

Author: Chao Ning
'''

import numpy as np
from scipy import sparse


def neg_logbf(sigma0, betahats, shat2s, prior_weights):
    """
    Calculate the negative log Bayes factor for a given set of parameters.

    Args:
        sigma0 (np.ndarray): Array of prior variances.
        betahats (np.ndarray): Array of estimated regression coefficients from simple regressions.
        shat2s (np.ndarray): Array of estimated variances from simple regressions.
        prior_weights (np.ndarray): Array of prior weights for variables.

    Returns:
        float: Negative log Bayes factor.
    """
    zscore2 = betahats * betahats / shat2s
    lbf_arr = 0.5 * np.log(shat2s / (shat2s + sigma0[0])) + \
               0.5 * zscore2 * sigma0[0] / (shat2s + sigma0[0])
    maxlbf = np.max(lbf_arr)
    bf_arr = np.exp(lbf_arr - maxlbf)
    bf_weighted_arr = bf_arr * prior_weights
    fx = -(np.log(np.sum(bf_weighted_arr)) + maxlbf)
    return fx


def calAlpha(sigma0, betahats, shat2s, prior_weights):
    zscore2 = betahats * betahats / shat2s
    lbf_arr = 0.5 * np.log(shat2s / (shat2s + sigma0[0])) + \
               0.5 * zscore2 * sigma0[0] / (shat2s + sigma0[0])
    maxlbf = np.max(lbf_arr)
    bf_arr = np.exp(lbf_arr - maxlbf)
    bf_weighted_arr = bf_arr * prior_weights
    lbf_model = np.log(np.sum(bf_weighted_arr)) + maxlbf
    bf_weighted_arr = bf_arr * prior_weights
    alpha_arr = bf_weighted_arr / np.sum(bf_weighted_arr)
    return alpha_arr, lbf_model


def filter_prior_components_mmsusie(alpha_arr2, mu_arr2, sigma0_arr, prior_tol):
    """
    Filter prior components based on a tolerance threshold.

    Args:
        alpha_LJ (ndarray): Prior weights matrix of shape (L, J).
        mu_LJQ (list of ndarray): List of posterior means, length L.
        sigma0_L (ndarray): Array of prior variances, length L.
        prior_tol (float): Threshold for filtering priors.

    Returns:
        alpha_LJ_new (ndarray): Filtered prior weights matrix.
        mu_LJQ_new (list of ndarray): Filtered list of posterior means.
    """
    # Identify components with prior variance greater than the threshold
    valid_components = sigma0_arr > prior_tol

    # Filter alpha_LJ rows
    alpha_arr2_new = alpha_arr2[valid_components, :]

    # Filter mu_LJQ entries
    mu_arr2_new = mu_arr2[valid_components, :]

    return alpha_arr2_new, mu_arr2_new


def getPIP(alpha_arr2):
    """
    Posterior inclusion probability (PIP) per variant, aggregated over the
    L single effects: ``PIP_j = 1 − Π_l (1 − α_{l,j})``.

    Args:
        alpha_arr2 (np.ndarray): Assignment probabilities α (L, p).

    Returns:
        np.ndarray: PIP per variant (p,).
    """
    L, p = alpha_arr2.shape
    pip_arr = np.ones(p)
    for l in range(L):
        pip_arr = pip_arr * (1 - alpha_arr2[l, :])
    return 1 - pip_arr


def in_CS_x(alpha_row: np.ndarray, coverage: float):
    """
    Build the ``coverage``-level credible set for one single effect: the smallest
    set of variants whose assignment probabilities sum to ≥ ``coverage``.

    Args:
        alpha_row (np.ndarray): Assignment probabilities α_l for one effect (p,).
        coverage (float): Target cumulative probability (e.g. 0.95).

    Returns:
        np.ndarray: 0/1 membership indicator (p,).
    """
    # Add variants in descending probability until the cumulative mass reaches coverage.
    sorted_indices = np.argsort(-alpha_row)
    sorted_alpha = alpha_row[sorted_indices]

    cumulative_sum = 0.0
    count = 0
    for i in range(len(sorted_alpha)):
        cumulative_sum += sorted_alpha[i]
        count += 1
        if cumulative_sum >= coverage:
            break

    result = np.zeros_like(alpha_row, dtype=int)
    result[sorted_indices[:count]] = 1
    return result


def in_CS(alpha_arr2: np.ndarray, coverage: float):
    """
    Apply :func:`in_CS_x` to every single effect, returning an (L, p) 0/1 matrix
    of credible-set membership (one row per effect).
    """
    L, p = alpha_arr2.shape
    status = np.zeros((L, p), dtype=int)
    for l in range(L):
        status[l, :] = in_CS_x(alpha_arr2[l, :], coverage)
    return status


def get_CS(status: np.ndarray) -> list[list[int]]:
    """
    Convert the (L, p) 0/1 membership matrix from :func:`in_CS` into a list of
    credible sets, each a list of the included variant indices.
    """
    cs = []
    for l in range(status.shape[0]):
        cs.append([j for j in range(status.shape[1]) if status[l, j] != 0])
    return cs


def compute_claimed_coverage(cs: list[list[int]], alpha: np.ndarray) -> np.ndarray:
    """
    Actual probability mass captured by each credible set — the sum of its
    variants' assignment probabilities (``Σ_{j∈cs_l} α_{l,j}``).
    """
    claimed_coverage = np.zeros(len(cs))
    for l, current_set in enumerate(cs):
        claimed_coverage[l] = sum(alpha[l, index] for index in current_set)
    return claimed_coverage


def get_cs_purity(cs: list[list[int]],
                  claimed_coverage: np.ndarray,
                  X: np.ndarray,
                  min_abs_corr: float) -> tuple[list[list[int]], np.ndarray]:
    """
    Filter credible sets based on their purity using minimum absolute correlation.

    Parameters:
    - cs: List of credible sets (each a list of variable indices)
    - claimed_coverage: 1D NumPy array of coverage values
    - X: 2D NumPy array (samples x features) - full design matrix
    - min_abs_corr: Minimum absolute correlation threshold

    Returns:
    - A tuple of (filtered credible sets, filtered coverage values)
    """
    is_purity = []

    for i, csi in enumerate(cs):
        if len(csi) == 1:
            is_purity.append(i)
        else:
            X_sub = X[:, csi]
            corr_matrix = np.corrcoef(X_sub, rowvar=False)
            tril_indices = np.tril_indices_from(corr_matrix, k=-1)
            corr_tril = np.abs(corr_matrix[tril_indices])
            min_corr = np.min(corr_tril)
            if min_corr > min_abs_corr:
                is_purity.append(i)

    # Filter the CS and claimed coverage
    cs_purity = [cs[i] for i in is_purity]
    claimed_coverage_purity = claimed_coverage[is_purity]

    return cs_purity, claimed_coverage_purity


def make_sparse_block(block_lst):
    """
    Make the sparse GRM from a list of blocks

    Args:
        block_lst (List[np.ndarray]): A list of blocks

    Returns:
        sparse: sparse block diag matrix
    """
    if not block_lst:
        return sparse.csr_matrix((0, 0), format='csr')

    if len(block_lst[0]) != 0:
        part1 = sparse.diags(block_lst[0], format='csr')
        if len(block_lst) > 1:
            part2 = sparse.block_diag(block_lst[1:], format='csr')
            return sparse.block_diag((part1, part2), format='csr')
        else:
            return part1
    else:
        return sparse.block_diag(block_lst[1:], format='csr') if len(block_lst) > 1 else sparse.csr_matrix((0, 0))


def block_grm_covariances(nvc, grm_block, is_singleton, env_block=None, num_env=None):
    """
    Build the per-block variance-component matrices ``A_k`` for a single-GRM model
    with 1-4 components — the single source of truth for the V structure shared by
    ``cal_spVi`` and the sparse REML routines:

        nvc==1: [I]
        nvc==2: [GRM, I]
        nvc==3: [GRM, GRM∘EE'/K, I]
        nvc==4: [GRM, GRM∘EE'/K, diag(‖e‖²/K), I]

    (I / residual is always last, so ``varcom[-1]`` is the residual variance.)

    Args:
        nvc (int): Number of variance components (1-4).
        grm_block: For a singleton block, the 1-D GRM diagonals; for a related
            block, the dense GRM sub-matrix.
        is_singleton (bool): True for the singleton diagonal block (returns 1-D
            vectors), False for a dense related block (returns matrices).
        env_block (np.ndarray or None): Environment rows for this block (nb, K);
            required when ``nvc >= 3``.
        num_env (int or None): Number of environments K; required when ``nvc >= 3``.

    Returns:
        list: ``A_k`` for k = 0 .. nvc-1 (vectors if singleton, else matrices).
    """
    if is_singleton:
        nb = len(grm_block)
        residual = np.ones(nb)
        if nvc == 1:
            return [residual]
        A = [grm_block]
        if nvc >= 3:
            nxe = np.sum(env_block * env_block, axis=1) / num_env   # ‖e‖²/K
            A.append(nxe * grm_block)                               # diag(GRM∘EE'/K)
            if nvc == 4:
                A.append(nxe)                                      # diag(‖e‖²/K)
        A.append(residual)
        return A

    nb = grm_block.shape[0]
    residual = np.eye(nb)
    if nvc == 1:
        return [residual]
    A = [grm_block]
    if nvc >= 3:
        gxe = (env_block @ env_block.T) / num_env * grm_block       # GRM∘EE'/K
        A.append(gxe)
        if nvc == 4:
            nxe = np.sum(env_block * env_block, axis=1) / num_env
            A.append(np.diag(nxe))
    A.append(residual)
    return A


def gls_residualize(mat, fixed, Vi):
    """
    Project the fixed effects out of ``mat`` in the V^{-1} metric:
    ``(I − X(X'V⁻¹X)⁻¹X'V⁻¹) mat`` (full Frisch–Waugh–Lovell in the GLS metric).

    Works for a vector or a 2-D matrix; ``Vi`` may be dense or ``scipy.sparse``.
    Falls back to a pseudo-inverse if ``X'V⁻¹X`` is rank-deficient.

    Args:
        mat (np.ndarray): Target to residualize — (n,) or (n, m).
        fixed (np.ndarray): Fixed-effect design X — (n, k).
        Vi: Inverse phenotypic covariance V^{-1} — dense or sparse (n, n).

    Returns:
        np.ndarray: Residualized target, same shape as ``mat``.
    """
    Vi_fixed = Vi @ fixed
    if sparse.issparse(Vi_fixed):
        Vi_fixed = Vi_fixed.toarray()
    gram = fixed.T @ Vi_fixed                      # X'V⁻¹X (k×k)
    arr = np.asarray(mat, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[:, None]
    rhs = Vi_fixed.T @ arr                          # X'V⁻¹ mat
    if np.linalg.matrix_rank(gram) < gram.shape[0]:
        coef = np.linalg.pinv(gram) @ rhs
    else:
        coef = np.linalg.solve(gram, rhs)
    resid = arr - fixed @ coef
    return resid.ravel() if was_1d else resid

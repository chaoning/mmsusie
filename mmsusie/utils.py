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
        p = alpha_arr2.shape[1]
        L = alpha_arr2.shape[0]
        alpha_arr2_tmp = 1 - alpha_arr2
        pip_arr = np.ones(p)
        for l in range(L):
            pip_arr = pip_arr * alpha_arr2_tmp[l, :]
        pip_arr = 1 - pip_arr
        return pip_arr


def in_CS_x(x: np.ndarray, coverage: float):

    # Get the indices that would sort x in descending order
    sorted_indices = np.argsort(-x)
    sorted_x = x[sorted_indices]

    # Compute the cumulative sum of sorted values and Find the minimum number of elements needed to reach the coverage threshold
    cumulative_sum = 0.0
    count = 0
    for i in range(len(sorted_x)):
        cumulative_sum += sorted_x[i]
        count += 1
        if cumulative_sum >= coverage:
            break

    # Create a binary result vector indicating selected elements
    result = np.zeros_like(x, dtype=int)
    result[sorted_indices[:count]] = 1
    return result


def in_CS(alpha_arr2: np.ndarray, coverage: float):
    p = alpha_arr2.shape[1]
    L = alpha_arr2.shape[0]
    status = np.zeros((L, p), dtype=int)
    for i in range(L):
        x = alpha_arr2[i, :]
        status[i, :] = in_CS_x(x, coverage)
    return status


def get_CS(status: np.ndarray) -> list[list[int]]:
    cs = []

    for i in range(status.shape[0]):
        current_row_indices = []
        for j in range(status.shape[1]):
            if status[i, j] != 0:
                current_row_indices.append(j)
        cs.append(current_row_indices)

    return cs

def compute_claimed_coverage(cs: list[list[int]], alpha: np.ndarray) -> np.ndarray:
    claimed_coverage = np.zeros(len(cs))

    for i, current_set in enumerate(cs):
        total = sum(alpha[i, index] for index in current_set)
        claimed_coverage[i] = total

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

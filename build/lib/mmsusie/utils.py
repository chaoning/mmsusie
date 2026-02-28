'''
Description: 
Author: Chao Ning
Date: 2025-04-01 09:47:34
LastEditTime: 2025-05-19 13:53:21
LastEditors: Chao Ning
'''

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import norm


def compute_XtViX_and_inv(Gj, E, Vi_sp, regularize=0.001):
    """
    Compute XtViX matrix, its inverse, and the log-determinant (negated) for SNP j.

    Parameters:
    - Gj (ndarray): Genotype vector (N x 1) for the jth SNP
    - E (ndarray): Environment matrix (N x K)
    - Vi_sp (ndarray): Inverse of phenotypic variance matrix (N x N), sparse
    - regularize (float): Diagonal regularization coefficient

    Returns:
    - XtViX_j (ndarray): K x K matrix (GE_j^T @ Vi_sp @ GE_j)
    - XtViX_inv_j (ndarray): Inverse of XtViX_j
    - logdet_XtViX_inv_j (float): Log-determinant of inverse matrix (i.e., -logdet(XtViX_j))
    """
    GEj = Gj * E                             # Element-wise interaction (N x K)
    XtViX_j = GEj.T @ Vi_sp @ GEj                   # Cross-product matrix (K x K)
    XtViX_j += np.diag(np.diag(XtViX_j) * regularize)  # Add regularization to diagonal
    XtViX_inv_j = np.linalg.inv(XtViX_j)            # Invert the matrix
    _, logdet_XtViX_j = np.linalg.slogdet(XtViX_j)  # Compute log-determinant of XtViX_j
    return XtViX_j, XtViX_inv_j, -logdet_XtViX_j    # Return -logdet to get logdet of inverse


def compute_all_XtViX(G, E, Vi_sp, n_jobs=32, regularize=0.001):
    """
    Compute XtViX matrices, their inverses, and log-determinants for all SNPs in parallel.

    Parameters:
    - G (ndarray): Genotype matrix (N x J)
    - E (ndarray): Environment matrix (N x K)
    - Vi_sp (ndarray): Inverse of residual variance matrix (N x N)
    - n_jobs (int): Number of parallel jobs
    - regularize (float): Diagonal regularization coefficient

    Returns:
    - Shat_lst (list of ndarray): List of XtViX matrices (J items, each K x K)
    - Shat_inv_lst (list of ndarray): List of XtViX inverses (J items, each K x K)
    - logdet_S_hat_lst (ndarray): Array of log-determinants of inverses (length J)
    """
    J = G.shape[1]  # Total number of SNPs
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_XtViX_and_inv)(G[:, [j]], E, Vi_sp, regularize) for j in tqdm(range(J))
    )
    Shat_inv_lst, Shat_lst, logdet_S_hat_lst = zip(*results)
    return list(Shat_inv_lst), list(Shat_lst), np.array(logdet_S_hat_lst)


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

def compute_lbf_j(sigma0, U_lst, wt_mix_arr, betahat, inv_Shat, Shat, logdet_Shat):
    Q = U_lst[0].shape[0]
    identity_Q = np.eye(Q)
    inv_Shat_b = inv_Shat @ betahat

    K = len(U_lst)
    lbf_mix_arr = np.empty(K)
    for k in range(K):
        S0 = sigma0 * U_lst[k]
        S1 = S0 @ np.linalg.inv(identity_Q + inv_Shat @ S0)
        _, logdet_S_sum = np.linalg.slogdet(S0 + Shat)
        quad_term = inv_Shat_b.T @ (S1 @ inv_Shat_b)
        lbf_mix_arr[k] = 0.5 * (logdet_Shat - logdet_S_sum + quad_term)

    maxlbf_mix = np.max(lbf_mix_arr)
    bf_mix_arr = np.exp(lbf_mix_arr - maxlbf_mix)
    weighted_bf_mix = bf_mix_arr * wt_mix_arr
    return np.log(np.sum(weighted_bf_mix)) + maxlbf_mix


def neg_logbf_mix(lsigma0, prior_cov, betahats_lst, Shat_inv_lst, Shat_lst, logdet_S_hat_arr, prior_weights, n_jobs=10):
    sigma0 = np.exp(lsigma0[0])
    U_lst = prior_cov["U"]
    wt_mix_arr = prior_cov["mixture_weights"]
    J = len(betahats_lst)

    # Parallel over all J SNPs
    lbf_arr = Parallel(n_jobs=n_jobs)(
        delayed(compute_lbf_j)(sigma0, U_lst, wt_mix_arr,
                               betahats_lst[j], Shat_inv_lst[j], Shat_lst[j], logdet_S_hat_arr[j])
        for j in range(J)
    )

    # Combine log Bayes factors with prior weights
    lbf_arr = np.array(lbf_arr)
    maxlbf = np.max(lbf_arr)
    bf_arr = np.exp(lbf_arr - maxlbf)
    weighted_bf_arr = bf_arr * prior_weights
    fx = -(np.log(np.sum(weighted_bf_arr)) + maxlbf)
    return fx


def optim_sigma0_em(sigma0, prior_cov, betahats_lst, Shat_inv_lst, Shat_lst, logdet_S_hat_arr, 
                    prior_weights, maxiter=100, tol=1e-9):
    """
    Estimate the optimal sigma0 using the Expectation-Maximization (EM) algorithm.
    """
    K = len(prior_cov["mixture_weights"])          # Number of prior components
    J = len(betahats_lst)           # Number of SNPs

    # Extract and cache prior covariance matrices and weights
    wt_mix_arr = prior_cov["mixture_weights"]
    U_lst = prior_cov["U"]
    inv_U_lst = prior_cov["U_pinv"]
    U_rank_arr = prior_cov["U_rank"]

    Q = U_lst[0].shape[0]
    identity_Q = np.eye(Q)

    posterior_mixture_weights = np.zeros((J, K))  # Posterior weights of mixture components

    for iter in range(maxiter):
        lbf_arr = np.zeros(J)                      # Log Bayes factors for each SNP
        b1_JK_lst = [[None] * K for _ in range(J)] # Posterior means for each SNP and mixture component
        S1_JK_lst = [[None] * K for _ in range(J)] # Posterior covariances for each SNP and mixture component
        b1_mix_lst, S1_mix_lst = [], [] # for each SNPs

        # E-step: compute posterior for each SNP
        for j in range(J):

            betahat = betahats_lst[j]
            Shat = Shat_lst[j]
            inv_Shat = Shat_inv_lst[j]
            logdet_S_hat = logdet_S_hat_arr[j]
            inv_Shat_b = inv_Shat @ betahat

            lbf_mix_arr = np.zeros(K)
            for k in range(K):
                S0 = sigma0 * U_lst[k]
                S1 = S0 @ np.linalg.inv(identity_Q + inv_Shat @ S0)
                _, logdet_S_sum = np.linalg.slogdet(S0 + Shat)
                quad_term = inv_Shat_b.T @ (S1 @ inv_Shat_b)
                lbf_mix_arr[k] = 0.5 * (logdet_S_hat - logdet_S_sum + quad_term)
                
                b1 = S1 @ (inv_Shat @ betahat)
                b1_JK_lst[j][k] = b1
                S1_JK_lst[j][k] = S1


            # Normalize to avoid overflow in exp
            maxlbf_mix = np.max(lbf_mix_arr)
            bf_mix_arr = np.exp(lbf_mix_arr - maxlbf_mix)
            weighted_bf_mix = bf_mix_arr * wt_mix_arr

            lbf_arr[j] = np.log(np.sum(weighted_bf_mix)) + maxlbf_mix
            posterior_mixture_weights[j, :] = weighted_bf_mix / np.sum(weighted_bf_mix)

            # Compute posterior mean and covariance of mixture
            b1_mix = sum(posterior_mixture_weights[j, k] * b1_JK_lst[j][k] for k in range(K))
            b1_mix_lst.append(b1_mix)

            S1_mix = -np.outer(b1_mix, b1_mix)
            for k in range(K):
                b1k = b1_JK_lst[j][k]
                S1_mix += posterior_mixture_weights[j, k] * (np.outer(b1k, b1k) + S1_JK_lst[j][k])
            S1_mix_lst.append(S1_mix)

        # M-step: update sigma0
        maxlbf = np.max(lbf_arr)
        bf_arr = np.exp(lbf_arr - maxlbf)
        weighted_bf_arr = bf_arr * prior_weights
        alpha_arr = weighted_bf_arr / np.sum(weighted_bf_arr)

        sigma0_new = 0.0
        for j in range(J):
            for k in range(K):
                b1 = b1_JK_lst[j][k]
                S1 = S1_JK_lst[j][k]
                expected_bbt = np.outer(b1, b1) + S1
                sigma0_new += alpha_arr[j] * posterior_mixture_weights[j, k] * np.sum(expected_bbt * inv_U_lst[k]) / U_rank_arr[k]

        # Convergence check
        if np.abs(sigma0_new - sigma0) < tol:
            break
        sigma0 = sigma0_new
        logging.info(f"sigma0: {sigma0}")

    return sigma0

def optim_sigma0(sigma0, prior_cov, betahats_lst, Shat_inv_lst, Shat_lst, logdet_S_hat_arr,
                  prior_weights, method="EM", maxiter=100, tol=1e-9):
    if method == "EM":
        sigma0 = optim_sigma0_em(sigma0, prior_cov, betahats_lst, Shat_inv_lst, Shat_lst, logdet_S_hat_arr, 
                                 prior_weights, maxiter, tol)
        return sigma0, True
    elif method == "optim":
        lsigma0 = np.log(sigma0)
        res = minimize(neg_logbf_mix, x0=[lsigma0], args=(prior_cov, betahats_lst, Shat_inv_lst, Shat_lst, logdet_S_hat_arr, prior_weights),
                               method="L-BFGS-B", bounds=[(-30, 10)])
        sigma0 = np.exp(res.x[0])
        return sigma0, res.success
    else:
        raise ValueError("Method must be 'EM' or 'optim'")


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

def compute_clfsr(NegativeProb, ZeroProb):
    condition = NegativeProb > 0.5 * (1 - ZeroProb)
    clfsr = np.where(condition, 1 - NegativeProb, NegativeProb + ZeroProb)
    return np.maximum(clfsr, 0.0)

def process_single_j(sigma0, wt_mix_arr, U_lst, betahats, Shat, inv_Shat, logdet_S_hat):
    K = len(U_lst)
    Q = betahats.shape[0]
    identity_envs = np.eye(Q)

    lbf_mix_K = np.zeros(K)
    b1_QK = np.zeros((Q, K))
    S1_lst = []
    zero_QK = np.zeros((Q, K))
    neg_QK = np.zeros((Q, K))

    for k in range(K):
        # Prior and posterior covariance
        S0 = U_lst[k] * sigma0
        S1 = S0 @ np.linalg.inv(identity_envs + inv_Shat @ S0)
        S1_lst.append(S1)

        # Posterior mean
        b1 = S1 @ (inv_Shat @ betahats)
        b1_QK[:, k] = b1

        # Log Bayes Factor
        inv_Shat_b = inv_Shat @ betahats
        _, logdet_S_sum = np.linalg.slogdet(S0 + Shat)
        log_BF = 0.5 * (logdet_S_hat - logdet_S_sum + inv_Shat_b @ (S1 @ inv_Shat_b))
        lbf_mix_K[k] = log_BF

        # Compute zero and negative probabilities
        S1_diag = np.diag(S1)
        sqrt_S1_diag = np.sqrt(S1_diag, where=S1_diag >= 1e-9, out=np.zeros_like(S1_diag))
        is_zero_var = S1_diag < 1e-9
        zero_QK[is_zero_var, k] = 1.0
        neg_QK[is_zero_var, k] = 0.0
        valid_idx = ~is_zero_var
        neg_QK[valid_idx, k] = norm.cdf(0, loc=b1[valid_idx], scale=sqrt_S1_diag[valid_idx])

    # Mixture log BF and posterior weights
    maxlbf_mix = np.max(lbf_mix_K)
    bf_mix_arr = np.exp(lbf_mix_K - maxlbf_mix)
    bf_mix_weighted_arr = bf_mix_arr * wt_mix_arr
    lbf_mix = np.log(np.sum(bf_mix_weighted_arr)) + maxlbf_mix

    # posterior mixture assignment probabilities
    numerator = wt_mix_arr * bf_mix_arr
    post_mix_wt_arr = numerator / np.sum(numerator)

    # Combine mixture components
    zero_prob_j = zero_QK @ post_mix_wt_arr
    neg_prob_j = neg_QK @ post_mix_wt_arr

    b1_mix = np.sum(b1_QK * post_mix_wt_arr.reshape(1, -1), axis=1)
    S1_mix = -np.outer(b1_mix, b1_mix)
    for k in range(K):
        S1_mix += post_mix_wt_arr[k] * (np.outer(b1_QK[:, k], b1_QK[:, k]) + S1_lst[k])

    neg_prob2_j = norm.cdf(0, loc=b1_mix, scale=np.sqrt(np.diag(S1_mix)))

    return lbf_mix, b1_mix, S1_mix, zero_prob_j, neg_prob_j, neg_prob2_j

def calAlphaMix(sigma0, prior_cov, betahats_lst, Shat_inv_lst, Shat_lst, logdet_S_hat_arr, prior_weights, n_jobs=32):
    K = len(prior_cov["mixture_weights"])
    J = len(betahats_lst)
    Q = len(betahats_lst[0])

    # Extract and cache prior covariance matrices and weights
    wt_mix_arr = prior_cov["mixture_weights"]
    U_lst = prior_cov["U"]
    
    # Parallel computation across J signals
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_j)(
            j, sigma0, wt_mix_arr, U_lst, betahats_lst[j],
            Shat_lst[j], Shat_inv_lst[j], logdet_S_hat_arr[j]
        ) for j in range(J)
    )

    # Aggregate results
    lbf_J, b1_mix_JQ, S1_mix_JQQ, zero_prob_JQ, neg_prob_JQ, neg_prob2_JQ = map(np.array, zip(*results))

    maxlbf = np.max(lbf_J)
    bf_arr = np.exp(lbf_J - maxlbf)
    bf_weighted_arr = bf_arr * prior_weights
    lbf_model = np.log(np.sum(bf_weighted_arr)) + maxlbf
    alpha_J = bf_weighted_arr / np.sum(bf_weighted_arr)

    # Compute clfsr and lfsr
    clfsr = compute_clfsr(neg_prob_JQ, zero_prob_JQ)
    clfsr2 = compute_clfsr(neg_prob2_JQ, zero_prob_JQ)
    lfsr = 1 - alpha_J.reshape(-1, 1) * (1 - clfsr)
    lfsr2 = 1 - alpha_J.reshape(-1, 1) * (1 - clfsr2)
    lfdr = 1 - alpha_J.reshape(-1, 1) * (1 - zero_prob_JQ)

    return alpha_J, lbf_model, b1_mix_JQ, S1_mix_JQQ, zero_prob_JQ, neg_prob_JQ, clfsr, np.maximum(lfsr, 0), lfdr


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


def filter_prior_components(alpha_LJ, mu_LJQ, zero_prob_lst, neg_prob_lst, clfsr_lst, lfsr_lst, sigma0_L, prior_tol):
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
    valid_components = sigma0_L > prior_tol

    # Filter alpha_LJ rows
    alpha_LJ_new = alpha_LJ[valid_components, :]

    # Filter mu_LJQ entries
    mu_LJQ_new = [mu for mu, keep in zip(mu_LJQ, valid_components) if keep]
    zero_prob_new = [zero_prob for zero_prob, keep in zip(zero_prob_lst, valid_components) if keep]
    neg_prob_new = [neg_prob for neg_prob, keep in zip(neg_prob_lst, valid_components) if keep]
    clfsr_new = [clfsr for clfsr, keep in zip(clfsr_lst, valid_components) if keep]
    lfsr_new = [lfsr for lfsr, keep in zip(lfsr_lst, valid_components) if keep]

    return alpha_LJ_new, mu_LJQ_new, zero_prob_new, neg_prob_new, clfsr_new, lfsr_new


def envi_lfsr(alpha_LJ, clfsr_lst, cs_lst):
    """
    Compute the environment-level local false sign rate (lfsr) for each credible set.

    Args:
        alpha_LJ (ndarray): Prior weights matrix of shape (L, J).
        clfsr_lst (list of ndarray): List of local false sign rates for each component.
        cs_lst (list of list): List of credible sets.

    Returns:
        envi_lfsr_arr (ndarray): Environment-level lfsr for each credible set.
    """
    L = alpha_LJ.shape[0]
    Q = clfsr_lst[0].shape[1]

    envi_lfsr_arr = np.zeros((len(cs_lst), Q))
    for ind, cs in enumerate(cs_lst):
        envi_lfsr = np.zeros((L, Q))
        for l in range(L):
            for q in range(Q):
                envi_lfsr[l, q] = np.sum(alpha_LJ[l, cs] * clfsr_lst[l][cs, q])
        envi_lfsr_arr[ind, :] = np.min(envi_lfsr, axis=0)
    return envi_lfsr_arr
    


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


def compute_min_correlation(matrix: np.ndarray) -> float:
    """
    Compute the minimum absolute correlation between variables (columns) in the matrix.

    Parameters:
    - matrix: 2D numpy array (n_samples, n_features)

    Returns:
    - Minimum absolute value of pairwise correlation coefficients (excluding diagonal)
    """
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    # Get the lower triangle (excluding diagonal) and flatten
    tril_indices = np.tril_indices_from(corr_matrix, k=-1)
    corr_tril = np.abs(corr_matrix[tril_indices])
    return np.min(corr_tril)


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


def compute_GEj_mu(Gj, E, alpha_LJ_lj, mu_LJQ_lj):
    """
    Compute the contribution of SNP j to the residual update.

    Parameters:
    - j: SNP index
    - G: Genotype matrix of shape (N, J)
    - E: Environment matrix of shape (N, K)
    - alpha_LJ_l: PIP vector for component l (length J)
    - mu_LJQ_l: List of effect mean vectors for component l (each K x 1)

    Returns:
    - contribution: N x 1 vector, the predicted value from SNP j
    """
    GEj = Gj * E  # Element-wise multiplication, shape: (N, K)
    return GEj @ (alpha_LJ_lj * mu_LJQ_lj)  # Shape: (N, 1)

def update_Xresi_parallel(G, E, alpha_LJ_l, mu_LJQ_l, n_jobs=32):
    """
    Compute the sum of contributions from all SNPs in parallel.

    Parameters:
    - G: Genotype matrix (N x J)
    - E: Environment matrix (N x K)
    - alpha_LJ_l: Posterior inclusion probabilities for component l (length J)
    - mu_LJQ_l: List of effect mean vectors for component l (length J, each K x 1)
    - n_jobs: Number of parallel jobs (CPU cores) to use

    Returns:
    - delta: Residual update vector (N x 1)
    """
    J = G.shape[1]

    # Parallel computation of each SNP's contribution
    contributions = Parallel(n_jobs=n_jobs)(
        delayed(compute_GEj_mu)(G[:, [j]], E, alpha_LJ_l[j], mu_LJQ_l[j, :]) for j in range(J)
    )

    # Sum all contributions from SNPs
    delta = np.sum(contributions, axis=0)  # Shape: (N, 1)
    return delta


def compute_betahat_j(Gj, E, Viy, Shat_lst_j):
    """
    Compute the beta estimate for SNP j.

    Parameters:
    - j: SNP index
    - G: Genotype matrix (N x J)
    - E: Environment matrix (N x K)
    - Viy: Pre-multiplied phenotype residual vector (N x 1)
    - Shat_lst: List of (K x K) inverse matrices for each SNP

    Returns:
    - betahat_j: Estimated beta vector (K x 1) for SNP j
    """
    GEj = Gj * E              # Shape: (N x K)
    score = GEj.T @ Viy              # Shape: (K x 1)
    return Shat_lst_j @ score       # Shape: (K x 1)

def compute_betahats_parallel(G, E, Viy, Shat_lst, n_jobs=32):
    """
    Compute beta estimates for all SNPs in parallel.

    Returns:
    - betahats_lst: List of (K x 1) beta vectors, one for each SNP
    """
    J = G.shape[1]

    betahats_lst =  Parallel(n_jobs=n_jobs)(
        delayed(compute_betahat_j)(G[:, [j]], E, Viy, Shat_lst[j]) for j in range(J)
    )

    return list(betahats_lst)

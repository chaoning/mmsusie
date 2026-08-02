'''
MMSuSiEDense: mixed-model SuSiE fine-mapping with a dense GRM covariance.

Author: Chao Ning
'''


import logging
import pandas as pd
import numpy as np
from mmsusie.utils import (
    neg_logbf,
    calAlpha,
    getPIP,
    in_CS,
    get_CS,
    compute_claimed_coverage,
    get_cs_purity,
    filter_prior_components_mmsusie,
    gls_residualize,
)
from mmsusie.io import read_genotype_matrix, ld_prune_assoc
import scipy
from scipy.optimize import minimize


def _sigma_neg_loglik_and_grad(varcom, gmat, y, X, Xresi, alpha_arr2, mu_arr2, mu2_arr2,
                               fixed=None):
    """
    Negative log-likelihood and gradient w.r.t. [sigma_g2, sigma_e2].
    V = sigma_g2 * gmat + sigma_e2 * I

    When ``fixed`` (the fixed-effect design F projected out of y/X) is given, the
    restricted-likelihood term ``0.5*log|F'V^{-1}F|`` and its gradient are added, so
    the update is REML rather than ML — this removes the downward variance bias that
    plain ML incurs from the k projected fixed effects.

    Returned gradient matches what scipy L-BFGS-B expects (grad of neg-loglik).
    """
    n = len(y)
    sigma_g2, sigma_e2 = varcom
    V = sigma_g2 * gmat + sigma_e2 * np.eye(n)
    sign, logdet = np.linalg.slogdet(V)
    if sign <= 0:
        return np.inf, np.full(2, np.inf)
    Vi = np.linalg.inv(V)

    r = y - Xresi
    Vir = Vi @ r
    ViX = Vi @ X
    xtVix = X.T @ ViX

    # SuSiE posterior correction:
    # sum_l {diag(alpha_l * mu2_l) - (alpha_l * mu_l)(alpha_l * mu_l)'}.
    mean_arr2 = alpha_arr2 * mu_arr2
    correction_mat = np.diag(np.sum(alpha_arr2 * mu2_arr2, axis=0)) - mean_arr2.T @ mean_arr2

    neg_ll = 0.5 * logdet + 0.5 * (r @ Vir + np.sum(xtVix * correction_mat))

    grad = np.zeros(2)

    # dV/d(sigma_g2) = gmat
    Vi_G_ViX = Vi @ (gmat @ ViX)
    grad[0] = (0.5 * np.sum(Vi * gmat)
               - 0.5 * (Vir @ (gmat @ Vir))
               - 0.5 * np.sum((X.T @ Vi_G_ViX) * correction_mat))

    # dV/d(sigma_e2) = I  →  Vi dV Vi = Vi²
    Vi2X = Vi @ ViX
    grad[1] = (0.5 * np.trace(Vi)
               - 0.5 * (Vir @ Vir)
               - 0.5 * np.sum((X.T @ Vi2X) * correction_mat))

    # REML restricted-likelihood correction for the k projected fixed effects F:
    #   +0.5*log|F'V^{-1}F|,   d/dσ²_j = -0.5*tr(M^{-1} U'A_j U),  U=V^{-1}F, M=F'V^{-1}F.
    if fixed is not None:
        U = Vi @ fixed
        M = fixed.T @ U
        sign_m, logdet_m = np.linalg.slogdet(M)
        if sign_m <= 0:
            return np.inf, np.full(2, np.inf)
        Minv = np.linalg.inv(M)
        neg_ll += 0.5 * logdet_m
        grad[0] += -0.5 * np.sum(Minv * (U.T @ (gmat @ U)))   # A_0 = gmat
        grad[1] += -0.5 * np.sum(Minv * (U.T @ U))            # A_1 = I

    return neg_ll, grad


class MMSuSiEDense:
    def __init__(self):
        self.iid_used = None
        self.iid_in_data = None
        self.df = None
        self.trait = None
        self.env_int = []
        self.env_int_arr2 = None
        self.Vi = None  # Inverse of V
        self.V_logdet = 0  # log|V|
        self.last_snp_ids = None  # SNP ids used in the latest get_genotype call
        self.gmat = None   # GRM stored by cal_Vi for estimate_sigma
        self.varcom = None # [sigma_g2, sigma_e2] stored by cal_Vi

    def read_data(self, data_file, trait, env_int=[], iid_col=0):
        """
        Read and preprocess the data file.

        Args:
            data_file (str): Path to the input data file. Space/Tab separated.
            trait (str): Column name of the target trait.
            env_int (list): List of column names for interacting environmental covariates.
            iid_col (int): Index of the column containing individual IDs. Defaults to 0.
        """
        self.trait = trait
        self.env_int = env_int

        with open(data_file, 'r') as f:
            head_line = f.readline().strip().split()
            iid_column_name = head_line[iid_col]

        usedcols_lst = [iid_column_name, trait] + list(env_int)
        if len(usedcols_lst) != len(set(usedcols_lst)):
            duplicated = [col for col in usedcols_lst if usedcols_lst.count(col) > 1]
            raise ValueError(f"Duplicate column names detected: {set(duplicated)}")

        dtype_map = {iid_column_name: str}
        dtype_map.update({col: float for col in list(env_int) + [trait]})

        df = pd.read_csv(data_file, sep=r"\s+", usecols=usedcols_lst, dtype=dtype_map)

        initial_rows = df.shape[0]
        df = df.dropna()
        dropped_rows = initial_rows - df.shape[0]
        if dropped_rows > 0:
            logging.warning(f"Dropped {dropped_rows} rows due to missing values.")

        # Index the IID column by NAME, not position: pandas `usecols` returns the
        # kept columns in FILE order, so `iloc[:, iid_col]` would point at the wrong
        # column whenever an unselected column precedes the IID.
        self.iid_column_name = iid_column_name
        self.iid_in_data = df[iid_column_name].tolist()
        if len(set(self.iid_in_data)) != len(self.iid_in_data):
            raise ValueError("Duplicated IIDs in data file!")

        self.iid_used = self.iid_in_data[:]
        logging.info(f"The number of used IIDs in data file: {len(self.iid_in_data)}")
        self.df = df

    def get_env_int(self, scale=True):
        """
        Get the interacting environmental covariates matrix.

        Args:
            scale (bool): Whether to standardize. Defaults to True.

        Returns:
            np.ndarray: Environmental covariate matrix (n, K).
        """
        self.env_int_arr2 = self.df.loc[:, self.env_int].values.astype(float)
        if scale:
            mean_arr = np.mean(self.env_int_arr2, axis=0).reshape(1, -1)
            std_arr = np.std(self.env_int_arr2, axis=0).reshape(1, -1)
            std_arr[std_arr == 0] = 1.0   # constant column -> leave it centred, don't divide by 0
            self.env_int_arr2 = (self.env_int_arr2 - mean_arr) / std_arr
        return self.env_int_arr2

    def get_y(self, adjust=True, scale=True):
        """
        Get the target trait values, optionally adjusting for environmental covariates.

        Args:
            adjust (bool): Whether to project out env covariates. Defaults to True.
            scale (bool): Whether to standardize. Defaults to True.

        Returns:
            np.ndarray: Trait values (n,).
        """
        y = self.df.loc[:, self.trait].values.astype(float)
        if adjust and self.env_int:
            # Lazily build the env matrix if get_env_int() was not called first; use
            # pinv so a rank-deficient / constant env design does not blow up.
            if self.env_int_arr2 is None:
                self.get_env_int()
            E = self.env_int_arr2
            y = y - E @ np.linalg.pinv(E.T @ E) @ (E.T @ y)
        if scale:
            sd = np.std(y)
            y = (y - np.mean(y)) / (sd if sd > 0 else 1.0)   # constant phenotype -> don't divide by 0
        return y

    def ld_pure(self, assoc_file, bed_file, ld_r2=0.1, snp="SNP", p="p_gxe", p_cutoff=5e-8):
        return ld_prune_assoc(assoc_file, bed_file, ld_r2=ld_r2, snp=snp, p=p, p_cutoff=p_cutoff)


    def process_y(self, y, X, adjust=True):
        """
        GLS-residualize the phenotype against the fixed effects ``X``:
        ``y − X(X'V⁻¹X)⁻¹X'V⁻¹y`` (projection in the V^{-1} metric). Requires
        :meth:`cal_Vi` to have set ``self.Vi``.

        Args:
            y (np.ndarray): Phenotype (n,) or (n, 1).
            X (np.ndarray): Fixed-effect design (n, k) — intercept + covariates.
            adjust (bool): If False, just flatten and return ``y`` unchanged.

        Returns:
            np.ndarray: Covariate-adjusted phenotype (n,).
        """
        if adjust:
            y = y - X @ (np.linalg.pinv(X.T @ self.Vi @ X) @ (X.T @ (self.Vi @ y)))
        return y.flatten()
    
    def get_genotype(self, bedfile, iid_lst, sid_lst=None, scale=True, *, start=None, end=None):
        """
        Get genotype matrix for selected individuals and SNPs from PLINK binary files.

        Exactly one SNP selection mode must be used:
        1) `sid_lst`: explicit SNP ids
        2) `start` + `end`: SNP id range in `.bim` order (inclusive)
        """
        genotype_matrix, self.last_snp_ids = read_genotype_matrix(
            bedfile, iid_lst, sid_lst=sid_lst, scale=scale, start=start, end=end
        )
        return genotype_matrix


    def cal_Vi(self, gmat, varcom):
        """
        Construct phenotypic covariance matrix V, compute its inverse and log-determinant.

        Args:
            varcom (list or np.ndarray): Variance components [sigma_g^2, sigma_e^2]

        Raises:
            ValueError: If variance components are invalid or V is not positive-definite
        """
        if len(varcom) != 2:
            raise ValueError("varcom must be a list or array with two elements: [sigma_g^2, sigma_e^2]")

        sigma_g2, sigma_e2 = varcom
        # sigma_g^2 == 0 is a valid boundary (V = sigma_e^2 I, i.e. standard SuSiE);
        # only the residual variance must be strictly positive. Positive-definiteness
        # of the assembled V is enforced by the slogdet sign check below.
        if sigma_g2 < 0 or sigma_e2 <= 0:
            raise ValueError("sigma_g^2 must be >= 0 and sigma_e^2 must be > 0.")

        # Construct V = sigma_g^2 * GRM + sigma_e^2 * I
        n = gmat.shape[0]
        V = sigma_g2 * gmat + sigma_e2 * np.identity(n)

        sign, logdet = np.linalg.slogdet(V)
        if sign <= 0:
            raise ValueError("Covariance matrix V is not positive definite; log-determinant undefined.")

        self.V_logdet = logdet

        # Compute V inverse (can be replaced with cho_solve for better numerical stability if needed)
        self.Vi = np.linalg.inv(V)
        self.gmat = gmat
        self.varcom = np.array([sigma_g2, sigma_e2], dtype=float)

    def _gls_residualize(self, mat, fixed):
        """Project ``fixed`` out of ``mat`` in the current V^{-1} metric."""
        return gls_residualize(mat, fixed, self.Vi)

    def fit(self, X, y, L=10, maxiter=100, tol=1e-3, coverage=0.95,
                min_abs_corr=0.5, prior_tol=1e-09, pip_index=None, estimate_sigma=True,
                fixed=None):
        """
        Run MMSuSiE fine-mapping on genotype ``X`` and phenotype ``y``.

        ``fixed`` (optional): the fixed-effect design (the ``xmat`` used for
        variance-component estimation, e.g. ``prepare_varcom_inputs(...)["xmat"]``).
        When provided, both ``y`` and the genotype are projected onto the covariate
        orthogonal complement in the V^{-1} metric (full Frisch–Waugh–Lovell),
        refreshed whenever ``estimate_sigma`` updates V. When None (default), only
        ``y`` is treated as pre-adjusted (backward-compatible; use ``process_y``).

        ``estimate_sigma=True`` re-estimates the variance components each sweep via a
        profile-ML / mixed-model (generalized-EM) update — NOT REML (the standalone
        :class:`WeightEMAI` is REML).
        """
        # Normalize/validate shapes up front so a (n, 1) phenotype or a mismatched
        # design fails with a clear message instead of a cryptic broadcast error.
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, p); got shape {X.shape}.")
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"y length ({y.shape[0]}) must match X rows ({X.shape[0]}).")
        if fixed is not None and np.asarray(fixed).shape[0] != X.shape[0]:
            raise ValueError(f"fixed rows ({np.asarray(fixed).shape[0]}) must match X rows ({X.shape[0]}).")

        p = X.shape[1]
        n = X.shape[0]
        if p < L:
            L = p
        yVar = np.var(y)

        if estimate_sigma and self.gmat is None:
            raise ValueError("estimate_sigma=True requires gmat to be available; call cal_Vi() first.")

        # Local copies updated each time estimate_sigma re-estimates V.
        Vi = self.Vi
        V_logdet = self.V_logdet

        # Raw standardized genotype, kept for credible-set purity: purity measures
        # biological LD between the CS variants, so it is computed on the genotype
        # itself, not on the FWL-projected / GLS design used for fitting.
        X_geno = X

        # Full Frisch–Waugh–Lovell: project fixed effects out of BOTH y and the
        # genotype in the V^{-1} metric (keep raw copies to refresh on V updates).
        fixed_arr = None
        if fixed is not None:
            fixed_arr = np.asarray(fixed, dtype=float)
            X_raw, y_raw = X, y
            X = self._gls_residualize(X, fixed_arr)
            y = self._gls_residualize(y, fixed_arr)

        logging.info("Starting mmsusie...")
        logging.info("Calculating shat2s...")
        vX = Vi @ X
        if scipy.sparse.issparse(vX):
            vX = vX.toarray()

        xtVix_mat = X.T @ vX
        xtVix = np.diag(xtVix_mat)
        if np.any(~np.isfinite(xtVix)) or np.any(xtVix <= 0):
            bad = np.where((~np.isfinite(xtVix)) | (xtVix <= 0))[0]
            raise ValueError(f"Non-positive or non-finite X'V^-1X diagonal entries at columns: {bad.tolist()}")
        shat2s = 1 / xtVix
        logging.info(f"shat2s: {shat2s[:5].T}")

        # Initialize susie fit
        prior_weights = np.full(p, 1.0 / p)  # uniform prior weights for each variable having the non-zero effect
        alpha_arr2 = np.full((L, p), 1.0 / p) # PIPs
        mu_arr2 = np.zeros((L, p))  # Posterior means
        mu2_arr2 = np.zeros((L, p)) # Posterior second moments
        Xresi = np.zeros(n)  # fitted values
        KL_arr = np.full(L, np.nan)
        lbf_arr = np.full(L, np.nan) # log Bayes factors
        sigma0_arr = np.full(L, yVar * 0.2) # Prior variance for each effect
        elbo_arr = np.full(maxiter + 1, np.nan) # ELBO values
        elbo_arr[0] = -np.inf
        res_dct = {}
        for iter in range(maxiter):
            logging.info(f"Iteration: {iter + 1}")
            # Variance-component (M-step) update from the PREVIOUS sweep's posterior,
            # applied at the TOP of the sweep so the V used below is exactly the V left
            # in self.Vi / self.varcom on return: posterior, ELBO and V stay consistent
            # even when the loop exits at maxiter (no dangling post-sweep update).
            if estimate_sigma and iter > 0:
                res_sigma = minimize(
                    _sigma_neg_loglik_and_grad,
                    x0=self.varcom.copy(),
                    args=(self.gmat, y, X, Xresi, alpha_arr2, mu_arr2, mu2_arr2,
                          fixed_arr),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(1e-10, None), (1e-10, None)],
                )
                if res_sigma.success:
                    self.varcom = res_sigma.x
                else:
                    logging.warning("Sigma optimization failed; keeping previous variances.")
                sigma_g2, sigma_e2 = self.varcom
                V = sigma_g2 * self.gmat + sigma_e2 * np.eye(n)
                _, V_logdet = np.linalg.slogdet(V)
                Vi = np.linalg.inv(V)
                self.Vi = Vi
                self.V_logdet = V_logdet
                if fixed_arr is not None:
                    # Re-project y and genotype with the updated V and rebuild the
                    # residual fit so it stays consistent with the re-projected X.
                    X = self._gls_residualize(X_raw, fixed_arr)
                    y = self._gls_residualize(y_raw, fixed_arr)
                    Xresi = X @ np.sum(alpha_arr2 * mu_arr2, axis=0)
                vX = Vi @ X
                xtVix_mat = X.T @ vX
                xtVix = np.diag(xtVix_mat)
                if np.any(~np.isfinite(xtVix)) or np.any(xtVix <= 0):
                    bad = np.where((~np.isfinite(xtVix)) | (xtVix <= 0))[0]
                    raise ValueError(f"Non-positive or non-finite X'V^-1X diagonal entries at columns: {bad.tolist()}")
                shat2s = 1 / xtVix
                logging.info(f"Updated variances: sigma_g2={self.varcom[0]:.6g}, sigma_e2={self.varcom[1]:.6g}")
            # update each effect once
            for l in range(L):
                # Remove lth effect from fitted values
                Xresi = Xresi - X @ (alpha_arr2[l, :] * mu_arr2[l, :])
                
                # Compute residuals
                resi = y - Xresi

                # Bayesian single-effect linear regression using residuals as outcomes
                XtViy = X.T @ (Vi @ resi)
                betahats = shat2s * XtViy # betas for p least-squares

                # optimize the prior variance
                sigma0 = sigma0_arr[l]
                res = minimize(neg_logbf, x0=[sigma0], args=(betahats, shat2s, prior_weights),
                               method="L-BFGS-B", bounds=[(1e-10, 1e10)])
                if res.success:
                    sigma0 = res.x[0]
                    sigma0_arr[l] = sigma0
                else:
                    logging.warning(
                        "Optimization of priors failed (%s); using priors from the previous iteration.",
                        res.message,
                    )
                
                alpha_arr, lbf_model = calAlpha([sigma0], betahats, shat2s, prior_weights)
                loglik = lbf_model - 0.5 * n * np.log(2 * np.pi) - 0.5 * V_logdet - \
                            0.5 * (resi @ (Vi @ resi))
                
                post_var_arr = 1 / (1 / sigma0 + 1 / shat2s) # Posterior variance.
                post_mean_arr = betahats / shat2s * post_var_arr
                post_mean2_arr = post_var_arr + post_mean_arr * post_mean_arr  # Second moment.

                # update
                mu_arr2[l, :] = post_mean_arr
                alpha_arr2[l, :] = alpha_arr
                mu2_arr2[l, :] = post_mean2_arr
                lbf_arr[l] = lbf_model

                SER_posterior_e_loglik = - 0.5 * n * np.log(2 * np.pi) - 0.5 * V_logdet \
                            - 0.5 * ( resi @ (Vi @ resi) -
                                      2 * np.sum(resi @ (Vi @ (X @ (alpha_arr * post_mean_arr)))) +
                                      np.sum(xtVix * (alpha_arr * post_mean2_arr)) )
                KL_arr[l] = -loglik + SER_posterior_e_loglik
                Xresi = Xresi + X @ (alpha_arr * post_mean_arr)
            
            logging.info(f"Estimated prior variances: {sigma0_arr.T}")
            mean_arr2 = alpha_arr2 * mu_arr2
            posterior_correction = (
                np.sum(np.sum(alpha_arr2 * mu2_arr2, axis=0) * xtVix)
                - np.sum((mean_arr2 @ xtVix_mat) * mean_arr2)
            )
            elbo_arr[iter + 1] = - 0.5 * n * np.log(2 * np.pi) - 0.5 * V_logdet \
                    - 0.5 * ((y - Xresi) @ (Vi @ (y - Xresi)) +
                    posterior_correction) - np.sum(KL_arr)
            logging.info(f"ELBO: {elbo_arr[iter + 1]}")
            if np.absolute(elbo_arr[iter + 1] - elbo_arr[iter]) < tol:
                break

        # Prune near-null effects and slice EVERY per-effect array by the same mask so
        # the returned arrays stay row-aligned (alpha/mu/sigma0/lbf/KL share one length).
        alpha_arr2, mu_arr2, kept = filter_prior_components_mmsusie(alpha_arr2, mu_arr2, sigma0_arr, prior_tol)
        sigma0_arr, lbf_arr, KL_arr = sigma0_arr[kept], lbf_arr[kept], KL_arr[kept]
        res_dct["alpha"] = alpha_arr2
        res_dct["mu"] = mu_arr2
        res_dct["kept_effects"] = np.where(kept)[0]
        if pip_index is None:
            if self.last_snp_ids is not None and len(self.last_snp_ids) == p:
                pip_index = self.last_snp_ids
            else:
                pip_index = list(range(p))
        else:
            pip_index = list(pip_index)
            if len(pip_index) != p:
                raise ValueError(f"`pip_index` length ({len(pip_index)}) does not match number of SNPs ({p}).")
        res_dct["pip"] = pd.DataFrame({"pip": getPIP(alpha_arr2)}, index=pip_index)
        res_dct["snp_ids"] = [str(sid) for sid in pip_index]
        status = in_CS(alpha_arr2, coverage)
        cs_lst = get_CS(status)
        claimed_coverage_arr = compute_claimed_coverage(cs_lst, alpha_arr2)
        cs_lst, claimed_coverage_arr = get_cs_purity(cs_lst, claimed_coverage_arr, X_geno, min_abs_corr)
        res_dct["cs"] = cs_lst
        res_dct["claimed_coverage"] = claimed_coverage_arr
        res_dct["lbf"] = lbf_arr
        res_dct["sigma0"] = sigma0_arr
        res_dct["elbo"] = elbo_arr
        res_dct["KL"] = KL_arr
        return res_dct
    

    
    def out(self, res_dct, out_file):
        """
        Write the fine-mapping result tables to ``<out_file>.{pip,alpha,mu,cs}.txt``.
        Uses SNP ids from ``res_dct["snp_ids"]`` as column/index labels when their
        count matches the number of variants, otherwise falls back to numeric ids.
        """
        alpha = res_dct["alpha"]
        mu = res_dct["mu"]
        p = alpha.shape[1]

        snp_ids = res_dct.get("snp_ids")
        has_snp_ids = snp_ids is not None and len(snp_ids) == p

        if has_snp_ids:
            pip_df = pd.DataFrame({"pip": res_dct["pip"]["pip"].to_numpy()}, index=snp_ids)
            pip_df.index.name = "SNP"
            pip_df.to_csv(out_file + ".pip.txt", sep="\t")

            alpha_df = pd.DataFrame(alpha, columns=snp_ids)
            alpha_df.to_csv(out_file + ".alpha.txt", sep="\t", index=False)

            mu_df = pd.DataFrame(mu, columns=snp_ids)
            mu_df.to_csv(out_file + ".mu.txt", sep="\t", index=False)

            with open(out_file + ".cs.txt", "w") as fin:
                for vec in res_dct["cs"]:
                    fin.write(" ".join([snp_ids[int(i)] for i in vec]) + "\n")
        else:
            np.savetxt(out_file + ".pip.txt", res_dct["pip"])
            np.savetxt(out_file + ".alpha.txt", alpha)
            np.savetxt(out_file + ".mu.txt", mu)
            with open(out_file + ".cs.txt", "w") as fin:
                for vec in res_dct["cs"]:
                    fin.write(" ".join([str(int(i)) for i in vec]) + "\n")

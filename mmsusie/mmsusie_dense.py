'''
Description: 
Author: Chao Ning
Date: 2025-03-17 20:33:53
LastEditTime: 2025-05-19 14:02:43
LastEditors: Chao Ning
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
)
import scipy
from scipy.optimize import minimize
from pysnptools.snpreader import Bed


def _sigma_neg_loglik_and_grad(varcom, gmat, y, X, Xresi, alpha_arr2, mu_arr2, mu2_arr2):
    """
    Negative log-likelihood and gradient w.r.t. [sigma_g2, sigma_e2].
    V = sigma_g2 * gmat + sigma_e2 * I

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
    xtVix = np.einsum('ij,ij->j', X, ViX)

    # SuSiE posterior correction: Σₗ αₗⱼ μ²ₗⱼ - Σₗ (αₗⱼ μₗⱼ)²
    delta = np.sum(alpha_arr2 * mu2_arr2, axis=0) - np.sum(np.square(alpha_arr2 * mu_arr2), axis=0)

    neg_ll = 0.5 * logdet + 0.5 * (r @ Vir + np.dot(delta, xtVix))

    grad = np.zeros(2)

    # dV/d(sigma_g2) = gmat
    Vi_G_ViX = Vi @ (gmat @ ViX)
    grad[0] = (0.5 * np.sum(Vi * gmat)
               - 0.5 * (Vir @ (gmat @ Vir))
               - 0.5 * np.dot(delta, np.einsum('ij,ij->j', X, Vi_G_ViX)))

    # dV/d(sigma_e2) = I  →  Vi dV Vi = Vi²
    Vi2X = Vi @ ViX
    grad[1] = (0.5 * np.trace(Vi)
               - 0.5 * (Vir @ Vir)
               - 0.5 * np.dot(delta, np.einsum('ij,ij->j', X, Vi2X)))

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

        self.iid_in_data = df.iloc[:, iid_col].tolist()
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
        self.env_int_arr2 = self.df.loc[:, self.env_int].values
        if scale:
            mean_arr = np.mean(self.env_int_arr2, axis=0).reshape(1, -1)
            std_arr = np.std(self.env_int_arr2, axis=0).reshape(1, -1)
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
        y = self.df.loc[:, self.trait].values
        if adjust:
            y = y - self.env_int_arr2 @ np.linalg.inv(self.env_int_arr2.T @ self.env_int_arr2) @ (self.env_int_arr2.T @ y)
        if scale:
            y = (y - np.mean(y)) / np.std(y)
        return y

    def ld_pure(self, assoc_file, bed_file, ld_r2=0.1, snp="SNP", p="p_gxe", p_cutoff=5e-8):
        df = pd.read_csv(assoc_file, sep=r"\s+")
        df = df[df[p] < p_cutoff].copy()
        df = df.sort_values(by=p)
        sig_snp_lst = df[snp].tolist()
        if not sig_snp_lst:
            raise ValueError("No significant SNPs found with the given p-value cutoff.")
        logging.info(f"The number of significant SNPs: {len(sig_snp_lst)}")

        # Read the bim file and get the index of the used SNPs
        bim_file = bed_file + ".bim"
        df_bim = pd.read_csv(bim_file, sep=r"\s+", header=None, dtype={0: str, 1: str})
        missing_snps = set(sig_snp_lst) - set(df_bim[1].tolist())
        if missing_snps:
            raise ValueError(f"Missing SNPs in bim file: {missing_snps}")
        dct = {df_bim.iloc[i, 1]: i for i in range(df_bim.shape[0])}
        sig_snp_index = [dct[sid] for sid in sig_snp_lst]

        # Read the genotype matrix from the bed file
        snp_on_disk = Bed(bed_file, count_A1=True)
        genotype_matrix = snp_on_disk[:, sig_snp_index].read().val
        genotype_matrix = pd.DataFrame(genotype_matrix, columns=sig_snp_lst)
        ld_r2_mat = genotype_matrix.corr() ** 2
        
        leading_snps = []
        while not ld_r2_mat.empty:
            leading_snps.append(ld_r2_mat.columns[0])
            corr_arr = ld_r2_mat.iloc[0, 1:].to_numpy()
            ld_r2_mat = ld_r2_mat.iloc[1:, 1:]
            ld_r2_mat = ld_r2_mat.loc[corr_arr < ld_r2, corr_arr < ld_r2]
        
        df_leading = df[df[snp].isin(leading_snps)].copy()
        return df_leading

    
    def process_y(self, y, X, adjust=True):
        if adjust:
            # Adjust y by X
            y = y - X @ (np.linalg.pinv(X.T @ self.Vi @ X) @ (X.T @ (self.Vi @ y)))
        return y.flatten()
    
    def get_genotype(self, bedfile, iid_lst, sid_lst=None, scale=True, *, start=None, end=None):
        """
        Get genotype matrix for selected individuals and SNPs from PLINK binary files.

        Exactly one SNP selection mode must be used:
        1) `sid_lst`: explicit SNP ids
        2) `start` + `end`: SNP id range in `.bim` order (inclusive)
        """
        use_sid_lst = sid_lst is not None
        use_range = start is not None or end is not None
        if use_sid_lst == use_range:
            raise ValueError("Use exactly one SNP selector: either `sid_lst` or `start`+`end`.")
        if use_range and (start is None or end is None):
            raise ValueError("Both `start` and `end` are required when using range selection.")
        if iid_lst is None or len(iid_lst) == 0:
            raise ValueError("`iid_lst` cannot be empty.")

        # Get the index of used individuals in the fam file.
        fam_file = bedfile + ".fam"
        df_fam = pd.read_csv(fam_file, sep=r"\s+", header=None, usecols=[1], dtype={1: str})
        fam_iids = pd.Index(df_fam[1])
        iid_used_index = fam_iids.get_indexer(iid_lst)
        if np.any(iid_used_index < 0):
            missing_iids = [iid_lst[i] for i in np.where(iid_used_index < 0)[0]]
            raise ValueError(f"Missing iids in fam file: {missing_iids}")

        # Read the bim file and build SNP indexes from either explicit IDs or ID range.
        bim_file = bedfile + ".bim"
        df_bim = pd.read_csv(bim_file, sep=r"\s+", header=None, usecols=[1], dtype={1: str})
        bim_sids = pd.Index(df_bim[1])
        if use_sid_lst:
            if len(sid_lst) == 0:
                raise ValueError("`sid_lst` cannot be empty.")
            snp_used_index = bim_sids.get_indexer(sid_lst)
            if np.any(snp_used_index < 0):
                missing_snps = [sid_lst[i] for i in np.where(snp_used_index < 0)[0]]
                raise ValueError(f"Missing SNPs in bim file: {missing_snps}")
        else:
            start_id = str(start)
            end_id = str(end)
            range_index = bim_sids.get_indexer([start_id, end_id])
            start_idx, end_idx = range_index[0], range_index[1]
            if start_idx < 0 or end_idx < 0:
                missing_ids = []
                if start_idx < 0:
                    missing_ids.append(start_id)
                if end_idx < 0:
                    missing_ids.append(end_id)
                raise ValueError(f"Missing range SNP IDs in bim file: {missing_ids}")
            if start_idx > end_idx:
                raise ValueError(
                    f"`start` SNP ({start_id}) appears after `end` SNP ({end_id}) in bim order."
                )
            snp_used_index = np.arange(start_idx, end_idx + 1, dtype=int)
        snp_used_ids = bim_sids[snp_used_index].astype(str).tolist()

        # Read the genotype matrix from the bed file.
        snp_on_disk = Bed(bedfile, count_A1=True)
        genotype_matrix = snp_on_disk[iid_used_index, snp_used_index].read().val
        genotype_matrix = pd.DataFrame(genotype_matrix)
        mean_genotype = genotype_matrix.mean()
        genotype_matrix.fillna(mean_genotype, inplace=True)
        genotype_matrix = genotype_matrix.values

        if scale:
            # Scale the genotype matrix; keep monomorphic SNPs as 0 after centering.
            mean_genotype = np.mean(genotype_matrix, axis=0).reshape(1, -1)
            std_genotype = np.std(genotype_matrix, axis=0).reshape(1, -1)
            std_genotype[std_genotype == 0] = 1.0
            genotype_matrix = (genotype_matrix - mean_genotype) / std_genotype
        self.last_snp_ids = snp_used_ids

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
        if sigma_g2 <= 0 or sigma_e2 <= 0:
            raise ValueError("Both variance components must be non-negative, and sigma_e^2 must be > 0.")

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

    def fit(self, X, y, L=10, maxiter=100, tol=1e-3, coverage=0.95,
                min_abs_corr=0.5, prior_tol=1e-09, pip_index=None, estimate_sigma=False):
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

        logging.info("Starting mmsusie...")
        logging.info("Calculating shat2s...")
        vX = Vi @ X
        if scipy.sparse.issparse(vX):
            vX = vX.toarray()

        xtVix = np.einsum('ij,ij->j', X, vX)
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
                    logging.warning("Optimization of priors failed; using priors from the previous iteration.")
                
                alpha_arr, lbf_model = calAlpha([sigma0], betahats, shat2s, prior_weights)
                loglik = lbf_model - 0.5 * n * np.log(2 * np.pi) - 0.5 * V_logdet - \
                            0.5 * (resi @ (Vi @ resi))
                
                post_var_arr = 1 / (1 / sigma0 + 1 / shat2s) # Posterior variance.
                post_mean_arr = betahats / shat2s * post_var_arr
                post_mean2_arr = post_var_arr + post_mean_arr * post_mean_arr; # Second moment.

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
            elbo_arr[iter + 1] = - 0.5 * n * np.log(2 * np.pi) - 0.5 * V_logdet \
                    - 0.5 * ( (y - Xresi) @ (Vi @ (y - Xresi)) +
                    np.sum(np.sum(alpha_arr2 * mu2_arr2, axis=0) * xtVix) -
                    np.sum(np.sum(np.square(alpha_arr2 * mu_arr2), axis=0) * xtVix)) - np.sum(KL_arr)
            logging.info(f"ELBO: {elbo_arr[iter + 1]}")
            if np.absolute(elbo_arr[iter + 1] - elbo_arr[iter]) < tol:
                break

            if estimate_sigma:
                res_sigma = minimize(
                    _sigma_neg_loglik_and_grad,
                    x0=self.varcom.copy(),
                    args=(self.gmat, y, X, Xresi, alpha_arr2, mu_arr2, mu2_arr2),
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
                vX = Vi @ X
                xtVix = np.einsum('ij,ij->j', X, vX)
                shat2s = 1 / xtVix
                logging.info(f"Updated variances: sigma_g2={self.varcom[0]:.6g}, sigma_e2={self.varcom[1]:.6g}")
        
        alpha_arr2, mu_arr2 = filter_prior_components_mmsusie(alpha_arr2, mu_arr2, sigma0_arr, prior_tol)
        res_dct["alpha"] = alpha_arr2
        res_dct["mu"] = mu_arr2
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
        cs_lst, claimed_coverage_arr = get_cs_purity(cs_lst, claimed_coverage_arr, X, min_abs_corr)
        res_dct["cs"] = cs_lst
        res_dct["claimed_coverage"] = claimed_coverage_arr
        res_dct["lbf"] = lbf_arr
        res_dct["sigma0"] = sigma0_arr
        res_dct["elbo"] = elbo_arr
        res_dct["KL"] = KL_arr
        return res_dct
    

    
    def out(self, res_dct, out_file):
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

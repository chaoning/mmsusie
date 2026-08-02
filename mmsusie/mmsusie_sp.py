'''
MMSuSiESp: mixed-model SuSiE fine-mapping with a sparse block-diagonal GRM.

Author: Chao Ning
'''


import logging
import pandas as pd
import numpy as np
from mmsusie.utils import neg_logbf, calAlpha, getPIP, in_CS, get_CS, compute_claimed_coverage, get_cs_purity, make_sparse_block
from mmsusie.utils import filter_prior_components_mmsusie, gls_residualize, block_grm_covariances
from mmsusie.io import read_genotype_matrix, ld_prune_assoc
import scipy
from scipy.optimize import minimize
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm


def _sigma_neg_loglik_and_grad_sp(varcom, grm_blocks, y, X, Xresi, alpha_arr2, mu_arr2, mu2_arr2,
                                   env_int_arr2=None):
    """
    Negative log-likelihood and gradient w.r.t. variance components for a
    sparse block-diagonal GRM.  Mirrors the four cases of cal_spVi:

      len==1: V = σ_e² I
      len==2: V = σ_g² GRM + σ_e² I
      len==3: V = σ_g² GRM + σ_gxe² (GRM ⊙ EE'/K) + σ_e² I
      len==4: V = σ_g² GRM + σ_gxe² (GRM ⊙ EE'/K) + σ_gxe2_E diag(‖e‖²/K) + σ_e² I

    grm_blocks[0]  — 1-D diagonal GRM values for singleton individuals.
    grm_blocks[1:] — 2-D dense GRM sub-matrices for related groups.
    env_int_arr2   — (n, K) environment matrix; required when len(varcom) >= 3.
    """
    nvc = len(varcom)
    r = y - Xresi
    # SuSiE posterior correction:
    # sum_l {diag(alpha_l * mu2_l) - (alpha_l * mu_l)(alpha_l * mu_l)'}.
    mean_arr2 = alpha_arr2 * mu_arr2
    correction_mat = np.diag(np.sum(alpha_arr2 * mu2_arr2, axis=0)) - mean_arr2.T @ mean_arr2

    num_envi_int = env_int_arr2.shape[1] if nvc >= 3 else None

    V_logdet = 0.0
    rVir = 0.0
    xtVix = np.zeros((X.shape[1], X.shape[1]))
    grad = np.zeros(nvc)

    start = 0
    for i, G_b in enumerate(grm_blocks):
        if i == 0:                          # diagonal block (singletons)
            nb = len(G_b)
            if nb == 0:
                continue

            env_b = env_int_arr2[start:start + nb, :] if nvc >= 3 else None
            A_diags = block_grm_covariances(nvc, G_b, True, env_b, num_envi_int)
            v_diag = sum(varcom[j] * A_diags[j] for j in range(nvc))

            if np.any(v_diag <= 0):
                return np.inf, np.full(nvc, np.inf)
            vi_diag = 1.0 / v_diag
            V_logdet += np.sum(np.log(v_diag))

            r_b   = r[start:start + nb]
            X_b   = X[start:start + nb, :]
            vir_b = vi_diag * r_b
            viX_b = vi_diag[:, None] * X_b

            rVir  += np.dot(vir_b, r_b)
            xtVix += X_b.T @ viX_b

            for k, a_d in enumerate(A_diags):
                grad[k] += 0.5 * np.dot(vi_diag, a_d)
                grad[k] -= 0.5 * np.dot(vir_b ** 2, a_d)
                ViAViX_b = (vi_diag ** 2 * a_d)[:, None] * X_b
                grad[k] -= 0.5 * np.sum((X_b.T @ ViAViX_b) * correction_mat)

        else:                               # dense block
            nb = G_b.shape[0]

            env_b = env_int_arr2[start:start + nb, :] if nvc >= 3 else None
            A_mats = block_grm_covariances(nvc, G_b, False, env_b, num_envi_int)
            V_b = sum(varcom[j] * A_mats[j] for j in range(nvc))

            sign, logdet = np.linalg.slogdet(V_b)
            if sign <= 0:
                return np.inf, np.full(nvc, np.inf)
            V_logdet += logdet
            Vi_b = np.linalg.inv(V_b)

            r_b   = r[start:start + nb]
            X_b   = X[start:start + nb, :]
            vir_b = Vi_b @ r_b
            viX_b = Vi_b @ X_b

            rVir  += np.dot(vir_b, r_b)
            xtVix += X_b.T @ viX_b

            for k, A_k in enumerate(A_mats):
                grad[k] += 0.5 * np.sum(Vi_b * A_k)
                grad[k] -= 0.5 * (vir_b @ (A_k @ vir_b))
                ViAViX_b = A_k @ viX_b
                grad[k] -= 0.5 * np.sum((viX_b.T @ ViAViX_b) * correction_mat)

        start += nb

    neg_ll = 0.5 * V_logdet + 0.5 * (rVir + np.sum(xtVix * correction_mat))
    return neg_ll, grad


class MMSuSiESp:
    def __init__(self):
        self.iid_used = None
        self.iid_in_data = None
        self.iid_in_grm = None
        self.df = None # data frame

        self.trait = None  # Column name of the target trait.
        self.env_int = [] # List of column names for interacting environmental covariates
        self.covariate_cols = []  # Numeric fixed-effect covariate columns
        self.categorical_cols = []  # Categorical fixed-effect columns (one-hot encoded)

        self.grm_blocks = [] # List of blocks for GRMs which are clustered and sorted by group size
        self.env_int_arr2 = None # numpy array for interacting environmental covariates
        
        self.Vi = None # Inverse of V
        self.V_logdet = 0 # log|V|
        self.varcom = None  # [sigma_g2, sigma_e2] stored by cal_spVi for estimate_sigma
        self.last_snp_ids = None  # SNP ids from the latest get_genotype call
    
    def mmsusie_lead_gxe(self, pheno_file, trait, env_int, grm_file, bedfile, snp_id, varcom_file, out_file,
               L=10, maxiter=100, tol=1e-3, coverage=0.95, min_abs_corr=0.5, prior_tol=1e-09,
               estimate_sigma=False, covariate_cols=None, categorical_cols=None, iid_col=0):
        """
        End-to-end GxE fine-mapping workflow.

        Runs read_data -> read_sp_grm -> get_env_int -> cal_spVi -> get_genotype,
        then GLS-residualizes y and GxE against [1, covariates, categoricals, E, G],
        runs mmsusie on residualized GxE, and writes out_mmsusie outputs.

        Args:
            pheno_file (str): Path to phenotype data file (space/tab separated).
            trait (str): Column name of the target trait.
            env_int (list): Column names for GxE interacting environmental covariates.
            grm_file (str): Prefix for the sparse GRM files.
            bedfile (str): Prefix for PLINK binary files (.bed/.bim/.fam).
            snp_id (str): Single SNP ID for GxE analysis.
            varcom_file (str): Path to file containing 1-4 variance components.
            out_file (str): Output file prefix for result tables.
            L (int): Maximum number of non-zero effects. Defaults to 10.
            maxiter (int): Maximum IBSS iterations. Defaults to 100.
            tol (float): ELBO convergence tolerance. Defaults to 1e-3.
            coverage (float): Credible set coverage. Defaults to 0.95.
            min_abs_corr (float): Minimum absolute correlation for credible set purity. Defaults to 0.5.
            prior_tol (float): Tolerance for filtering prior components. Defaults to 1e-09.
            estimate_sigma (bool): If True, jointly re-estimate variance components during IBSS. Defaults to False.
            covariate_cols (list or None): Numeric fixed-effect covariates. Defaults to None.
            categorical_cols (list or None): Categorical fixed-effect covariates. Defaults to None.
            iid_col (int): Index of the IID column in pheno_file. Defaults to 0.

        Returns:
            dict: Results from mmsusie().
        """
        if len(env_int) == 0:
            raise ValueError("`env_int` must contain at least one environmental covariate.")

        self.read_data(
            pheno_file,
            trait,
            env_int,
            covariate_cols=[] if covariate_cols is None else covariate_cols,
            categorical_cols=[] if categorical_cols is None else categorical_cols,
            iid_col=iid_col,
        )
        self.read_sp_grm(grm_file)

        E = self.get_env_int(scale=True)

        _vc = np.loadtxt(varcom_file)
        if np.ndim(_vc) == 0:
            varcom = np.array([float(_vc)])
        elif _vc.ndim == 1:
            varcom = np.asarray(_vc, dtype=float)
        elif 1 in _vc.shape:
            varcom = np.asarray(_vc, dtype=float).reshape(-1)
        else:
            varcom = np.asarray(_vc[:, 0], dtype=float)
        if len(varcom) not in (1, 2, 3, 4):
            raise ValueError(f"`varcom_file` must contain 1-4 variance components; got {len(varcom)}.")
        self.cal_spVi(varcom)
        G = self.get_genotype(bedfile, sid_lst=[snp_id], scale=True)
        lead_snp_ids = self.last_snp_ids

        GE = G * E
        y_raw = self.df.loc[:, self.trait].values

        nuisance_blocks = [np.ones((len(y_raw), 1))]
        if self.covariate_cols:
            nuisance_blocks.append(self.df.loc[:, self.covariate_cols].values)
        if self.categorical_cols:
            cat_df = pd.get_dummies(
                self.df[self.categorical_cols].astype("category"),
                drop_first=True, dtype=float,
            )
            if cat_df.shape[1] > 0:
                nuisance_blocks.append(cat_df.values)
        nuisance_blocks.extend([E, G])
        nuisance = np.hstack(nuisance_blocks)

        self.last_snp_ids = None
        try:
            # Pass the RAW interaction/phenotype with fixed=nuisance and let fit() do the
            # FWL projection, so it is re-projected whenever estimate_sigma updates V
            # (rather than frozen at the initial V by a manual pre-projection here).
            res = self.fit(GE, y_raw, L=L, maxiter=maxiter, tol=tol, coverage=coverage,
                           min_abs_corr=min_abs_corr, prior_tol=prior_tol,
                           estimate_sigma=estimate_sigma, fixed=nuisance)
            res["lead_snp"] = lead_snp_ids[0] if lead_snp_ids else str(snp_id)
            self.out(res, out_file)
        finally:
            self.last_snp_ids = lead_snp_ids
        return res

    def ld_pure(self, assoc_file, bed_file, ld_r2=0.1, snp="SNP", p="p_gxe", p_cutoff=5e-8):
        return ld_prune_assoc(assoc_file, bed_file, ld_r2=ld_r2, snp=snp, p=p, p_cutoff=p_cutoff)

    def read_data(self, data_file, trait, env_int=[], covariate_cols=[], categorical_cols=[], iid_col=0):
        """
        Read and preprocess the data file.

        Args:
            data_file (str): Path to the input data file. Space/Tab separated.
            trait (str): Column name of the target trait.
            env_int (list): Column names for GxE interacting environmental covariates.
            covariate_cols (list): Numeric column names for fixed-effect adjustment.
            categorical_cols (list): Categorical column names for fixed-effect adjustment
                (one-hot encoded with drop_first=True).
            iid_col (int): Index of the column containing individual IDs. Defaults to 0.
        """
        self.trait = trait
        self.env_int = list(env_int)
        self.covariate_cols = list(covariate_cols)
        self.categorical_cols = list(categorical_cols)

        with open(data_file, 'r') as f:
            head_line = f.readline().strip().split()
            iid_column_name = head_line[iid_col]

        usedcols_lst = [iid_column_name, trait] + list(env_int) + list(covariate_cols) + list(categorical_cols)
        if len(usedcols_lst) != len(set(usedcols_lst)):
            duplicated = [col for col in usedcols_lst if usedcols_lst.count(col) > 1]
            raise ValueError(f"Duplicate column names detected: {set(duplicated)}")

        dtype_map = {iid_column_name: str}
        dtype_map.update({col: float for col in list(env_int) + list(covariate_cols) + [trait]})
        dtype_map.update({col: str for col in list(categorical_cols)})

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


    def read_sp_grm(self, grm_file):
        """
        Read the sparse genetic relationship matrix and update the data frame

        Args:
            grm_file (str): Prefix for the genetic relationship matrix

        Raises:
            ValueError: None
        """
        
        # Read the GRM group file
        df_group = pd.read_csv(grm_file + ".grm.group", sep=r"\s+", header=None,
                               dtype={0: "Int64", 1: str, 2: "Int64", 3: "Int64"})
        self.iid_in_grm = df_group.iloc[:, 1].tolist()
        logging.info(f"The number of IIDs in the GRM file: {len(self.iid_in_grm)}")

        # By default, use all individuals from the GRM
        self.iid_used = self.iid_in_grm[:]

        # If `iid_in_data` is provided, filter IIDs accordingly
        if self.iid_in_data:
            self.iid_used = list(set(self.iid_in_data) & set(self.iid_in_grm))

            if not self.iid_used:
                raise ValueError("No overlapping IIDs found between the GRM and data file.")
            else:
                logging.info(f"The number of IIDs used after matching with the data file: {len(self.iid_used)}")

        # Filter `df_group` to keep only rows with IIDs in `iid_used`
        iid_used_set = set(self.iid_used)
        df_group = df_group[df_group.iloc[:, 1].isin(iid_used_set)].copy()

        # Compute `group_size` using `value_counts()` instead of a manual loop
        df_group[3] = df_group[2].map(df_group[2].value_counts())

        # Sort by `group_size` and then by `group ID`
        df_group = df_group.sort_values(by=[3, 2])

        # Update `iid_used` based on the sorted order
        self.iid_used = df_group[1].tolist()

        # Filter `self.df` to rows whose IID is in `iid_used` (index by IID column
        # NAME, since it is not necessarily the first column of the loaded frame).
        iid_col_name = self.iid_column_name
        self.df = self.df.loc[self.df[iid_col_name].isin(iid_used_set)]

        # Sort `self.df` based on the order of `iid_used`
        self.df = self.df.assign(sort_order=pd.Categorical(self.df[iid_col_name], categories=self.iid_used, ordered=True)).sort_values(by="sort_order").drop(columns=["sort_order"])

        # Load GRM file
        df_GRM = pd.read_csv(
            f"{grm_file}.grm.index_triplet",
            sep=r"\s+",
            header=None,
            dtype={0: "Int64", 1: "Int64", 2: "float64"},
            names=["i", "j", "val"]
        )

        GRM_blocks = []
        # Pre-build a nested dictionary for fast lookup of GRM values
        # grm_dict[i][j] = value of genetic relationship between i and j
        grm_dict = defaultdict(dict)
        for i, j, val in zip(df_GRM["i"], df_GRM["j"], df_GRM["val"]):
            grm_dict[i][j] = val
            grm_dict[j][i] = val  # Ensure symmetry in the matrix
        
        # Process group with size == 1
        df_sub_group = df_group[df_group[3] == 1].copy()
        GRM_groupSizeOnes = np.array([])

        if df_sub_group.shape[0] > 0:
            sub_group_index = df_sub_group[0].tolist()
            GRM_groupSizeOnes = np.array([grm_dict[i][i] for i in sub_group_index])
        
        GRM_blocks.append(GRM_groupSizeOnes)

        # Process groups with size > 1
        df_sub_group = df_group[df_group[3] != 1].copy()

        for (_, _), sub_df in tqdm(df_sub_group.groupby([3, 2])):
            # Extract individual index in this subgroup
            sub_group_index = sub_df[0].tolist()
            # Create a mapping from individual index to matrix index
            index_map = {idx: i for i, idx in enumerate(sub_group_index)}
            n = len(sub_group_index)
            # Initialize identity matrix (diagonal = 1, off-diagonal = 0)
            GRM_sub = np.eye(n)
            # Fill in the GRM values for the subgroup
            for i in sub_group_index:
                for j in sub_group_index:
                    val = grm_dict[i].get(j, 0.0)  # Default to 0.0 if not found
                    GRM_sub[index_map[i], index_map[j]] = val
                    GRM_sub[index_map[j], index_map[i]] = val
            GRM_blocks.append(GRM_sub)
        self.grm_blocks = GRM_blocks
    
    def get_env_int(self, scale=True):
        """
        Get the interacting environmental covariates matrix.

        Args:
            scale (bool): Whether to standardize (mean 0, std 1). Defaults to True.

        Returns:
            np.ndarray: Environmental covariate matrix (n, K).
        """
        self.env_int_arr2 = self.df.loc[:, self.env_int].values
        if scale:
            mean_arr = np.mean(self.env_int_arr2, axis=0).reshape(1, -1)
            std_arr = np.std(self.env_int_arr2, axis=0).reshape(1, -1)
            std_arr[std_arr == 0] = 1.0
            self.env_int_arr2 = (self.env_int_arr2 - mean_arr) / std_arr
        return self.env_int_arr2
    
    def get_y(self, adjust=True):
        """
        Get the target trait values, optionally adjusting for fixed effects.

        Uses GLS (β̂ = (X'V⁻¹X)⁻¹X'V⁻¹y) when ``cal_spVi()`` has been called,
        otherwise falls back to OLS.

        When ``adjust=True``, projects out intercept plus all of:
        ``covariate_cols``, ``categorical_cols`` (one-hot encoded), and
        ``env_int_arr2`` (if ``get_env_int()`` has been called first).

        Args:
            adjust (bool): Whether to project out fixed effects. Defaults to True.

        Returns:
            np.ndarray: Trait values (n,).
        """
        y = self.df.loc[:, self.trait].values
        if adjust:
            xmat = self.get_fixed()
            if self.Vi is not None:
                Vi_xmat = self.Vi @ xmat
                if scipy.sparse.issparse(Vi_xmat):
                    Vi_xmat = Vi_xmat.toarray()
                beta = np.linalg.solve(xmat.T @ Vi_xmat, Vi_xmat.T @ y)
            else:
                beta = np.linalg.lstsq(xmat, y, rcond=None)[0]
            y = y - xmat @ beta
        return y

    def get_fixed(self):
        """
        Build the fixed-effect design matrix used for covariate adjustment:
        intercept + ``covariate_cols`` + ``categorical_cols`` (one-hot,
        drop_first) + ``env_int_arr2`` (if ``get_env_int()`` was called).

        Pass the returned matrix as ``fixed=`` to :meth:`mmsusie` to enable full
        Frisch–Waugh–Lovell adjustment (project covariates out of the genotype as
        well as the phenotype).

        Returns:
            np.ndarray: Fixed-effect design (n, k).
        """
        n = self.df.shape[0]
        x_blocks = [np.ones((n, 1))]
        if self.covariate_cols:
            x_blocks.append(self.df.loc[:, self.covariate_cols].values)
        if self.categorical_cols:
            cat_df = pd.get_dummies(
                self.df[self.categorical_cols].astype("category"),
                drop_first=True, dtype=float,
            )
            if cat_df.shape[1] > 0:
                x_blocks.append(cat_df.values)
        if self.env_int_arr2 is not None and len(self.env_int) > 0:
            x_blocks.append(self.env_int_arr2)
        return np.hstack(x_blocks)

    def _gls_residualize(self, mat, fixed):
        """Project ``fixed`` out of ``mat`` in the current V^{-1} metric."""
        return gls_residualize(mat, fixed, self.Vi)

    def get_genotype(self, bedfile, sid_lst=None, scale=True, *, start=None, end=None):
        """
        Get genotype matrix for self.iid_used ordered individuals.

        IID order is fixed by read_sp_grm() to match the sparse block structure.

        Exactly one SNP selection mode must be used:
        1) `sid_lst`: explicit SNP ids
        2) `start` + `end`: SNP id range in `.bim` order (inclusive)
        """
        genotype_matrix, self.last_snp_ids = read_genotype_matrix(
            bedfile, self.iid_used, sid_lst=sid_lst, scale=scale, start=start, end=end
        )
        return genotype_matrix


    def cal_spVi(self, varcom):
        """
        Build the sparse block-diagonal V^{-1} and log|V| for 1-4 variance
        components (single GRM); the V structure comes from
        ``utils.block_grm_covariances`` (shared with the sparse REML routines).

        Args:
            varcom (np.ndarray): Variance components (length 1-4).
        """
        varcom = np.asarray(varcom, dtype=float)
        nvc = len(varcom)
        # Validate up front so an invalid spec fails loudly instead of silently
        # producing a NaN log-determinant or a non-PD "inverse".
        if nvc not in (1, 2, 3, 4):
            raise ValueError(f"varcom must have 1-4 components, got {nvc}.")
        if np.any(varcom < 0) or varcom[-1] <= 0:
            raise ValueError(
                f"variance components must be >= 0 with a positive residual (last); got {varcom.tolist()}."
            )
        if nvc >= 3 and getattr(self, "env_int_arr2", None) is None:
            raise ValueError("nvc >= 3 requires the environment matrix; call get_env_int() first.")
        self.varcom = varcom

        if nvc == 1:                                    # V = σ_e² I
            num_iid_used = len(self.iid_used)
            self.Vi = sparse.identity(num_iid_used) / varcom[0]
            self.V_logdet = num_iid_used * np.log(varcom[0])
            return

        num_env = self.env_int_arr2.shape[1] if nvc >= 3 else None
        V_logdet = 0.0
        Vi_lst = []
        start = 0
        for i, grm_block in enumerate(self.grm_blocks):
            num_element = grm_block.shape[0]
            if i == 0 and num_element == 0:             # no singleton individuals
                Vi_lst.append(np.array([]))             # keep the empty singleton slot so
                continue                                # make_sparse_block treats block 0 as
                                                        # the (empty) diagonal, not a dense block
            env_block = (self.env_int_arr2[start:start + num_element, :]
                         if nvc >= 3 else None)
            A = block_grm_covariances(nvc, grm_block, i == 0, env_block, num_env)
            V_block = sum(varcom[j] * A[j] for j in range(nvc))
            if i == 0:                                  # singleton diagonal block
                if np.any(V_block <= 0):
                    raise ValueError("Non-positive singleton variance; check the variance components.")
                V_logdet += np.sum(np.log(V_block))
                Vi_lst.append(1.0 / V_block)
            else:                                       # dense related block
                sign, logdet = np.linalg.slogdet(V_block)
                if sign <= 0:
                    raise ValueError("A related-block covariance is not positive definite; check the variance components.")
                V_logdet += logdet
                Vi_lst.append(np.linalg.inv(V_block))
            start += num_element
        self.Vi = make_sparse_block(Vi_lst)
        self.V_logdet = V_logdet

    def fit(self, X, y, L=10, maxiter=100, tol=1e-3, coverage=0.95,
                min_abs_corr=0.5, prior_tol=1e-09, estimate_sigma=True, fixed=None):
        """
        Mixed-model SuSiE fine-mapping via the IBSS coordinate-ascent algorithm.
        Named ``fit`` to mirror :class:`MMSuSiEDense`; ``mmsusie`` is a
        backward-compatible alias.

        Fits ``y ~ N(Σ_l X (α_l ∘ μ_l), V)`` — a sum of ``L`` single-effect
        regressions on the genotype ``X`` with residual covariance ``V`` encoding
        the (sparse) GRM. Each IBSS sweep updates one single-effect regression
        (SER) at a time in the ``V^{-1}`` (GLS) metric until the ELBO converges.

        Args:
            X (np.ndarray): Genotype matrix (n, p), standardized — the fine-mapping
                design; each column is a candidate causal variant.
            y (np.ndarray): Phenotype (n,), already covariate-adjusted (see
                :meth:`get_y`) unless ``fixed`` is supplied.
            L (int): Maximum number of causal effects (single-effect components).
            maxiter (int): Maximum IBSS sweeps. tol (float): ELBO convergence tol.
            coverage (float): Target credible-set coverage (e.g. 0.95).
            min_abs_corr (float): Minimum absolute correlation for CS purity filtering.
            prior_tol (float): Prune effects whose prior variance falls below this.
            estimate_sigma (bool): Jointly re-estimate the variance components each
                sweep via a profile-ML / mixed-model (generalized-EM) update — this is
                NOT REML (the standalone :class:`WeightEMAISp` is REML). Requires
                :meth:`cal_spVi` to have been called.
            fixed (np.ndarray or None): Fixed-effect design (e.g. :meth:`get_fixed`).
                When given, both ``y`` and the genotype are projected onto the
                covariate orthogonal complement in the ``V^{-1}`` metric — full
                Frisch–Waugh–Lovell — and re-projected whenever ``estimate_sigma``
                updates ``V``. When None (default) only ``y`` is pre-adjusted
                (backward-compatible).

        Returns:
            dict: ``pip`` (per-variant PIP), ``cs`` (credible sets), ``alpha``
            (L×p assignment probabilities), ``mu`` (posterior mean effects),
            ``sigma0`` (per-effect prior variances), ``lbf``/``KL``/``elbo``.

        Notation (standard SuSiE / mixed-model symbols used below):
            Vi = V^{-1};  xtVix_mat = X'V^{-1}X,  xtVix = its diagonal;
            shat2s = 1/xtVix = per-variant GLS effect variance (ŝ²);
            betahats = single-variant GLS effect estimates;
            alpha_arr2 (L×p) = α, mu_arr2 = μ (posterior mean), mu2_arr2 = 2nd moment;
            Xresi = current fitted genotype signal Σ_l X(α_l∘μ_l);
            sigma0 = SER prior variance;  lbf = log Bayes factor;  KL = KL divergence.
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

        if estimate_sigma:
            if self.varcom is None or len(self.varcom) not in (1, 2, 3, 4):
                raise ValueError(
                    "estimate_sigma=True requires 1–4 variance components; "
                    "call cal_spVi() first."
                )

        # Local copies updated when estimate_sigma re-estimates V.
        Vi = self.Vi
        V_logdet = self.V_logdet

        # Raw standardized genotype, kept for credible-set purity: purity measures
        # biological LD between the CS variants, so it is computed on the genotype
        # itself, not on the FWL-projected / GLS design used for fitting.
        X_geno = X

        # Full Frisch–Waugh–Lovell: project the fixed effects out of BOTH y and
        # the genotype in the V^{-1} metric. Keep the raw copies so the projection
        # can be refreshed whenever estimate_sigma updates V.
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
                nvc = len(self.varcom)
                res_sigma = minimize(
                    _sigma_neg_loglik_and_grad_sp,
                    x0=self.varcom.copy(),
                    args=(self.grm_blocks, y, X, Xresi, alpha_arr2, mu_arr2, mu2_arr2,
                          self.env_int_arr2),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(1e-10, None)] * nvc,
                )
                if res_sigma.success:
                    self.varcom = res_sigma.x
                else:
                    logging.warning("Sigma optimization failed; keeping previous variances.")
                self.cal_spVi(self.varcom)
                Vi = self.Vi
                V_logdet = self.V_logdet
                if fixed_arr is not None:
                    # Re-project y and genotype with the updated V, and rebuild the
                    # residual fit so it stays consistent with the re-projected X.
                    X = self._gls_residualize(X_raw, fixed_arr)
                    y = self._gls_residualize(y_raw, fixed_arr)
                    Xresi = X @ np.sum(alpha_arr2 * mu_arr2, axis=0)
                vX = Vi @ X
                if scipy.sparse.issparse(vX):
                    vX = vX.toarray()
                xtVix_mat = X.T @ vX
                xtVix = np.diag(xtVix_mat)
                if np.any(~np.isfinite(xtVix)) or np.any(xtVix <= 0):
                    bad = np.where((~np.isfinite(xtVix)) | (xtVix <= 0))[0]
                    raise ValueError(f"Non-positive or non-finite X'V^-1X diagonal entries at columns: {bad.tolist()}")
                shat2s = 1 / xtVix
                logging.info(f"Updated varcom: {self.varcom}")
            # update each effect once
            for l in range(L):
                # Remove lth effect from fitted values
                Xresi = Xresi - X @ (alpha_arr2[l, :] * mu_arr2[l, :])

                # Compute residuals
                resi = y - Xresi

                # V^{-1}resi is reused below (XtViy, and the loglik/KL quadratic
                # forms), so compute it once. Vi is symmetric, hence
                #   X'V^{-1}resi = (Vi·resi)'X   and   resi'V^{-1}resi = resi·(Vi·resi).
                Vir = np.asarray(Vi @ resi).ravel()
                resiVir = resi @ Vir

                # Bayesian single-effect linear regression using residuals as outcomes
                XtViy = X.T @ Vir
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
                            0.5 * resiVir

                post_var_arr = 1 / (1 / sigma0 + 1 / shat2s) # Posterior variance.
                post_mean_arr = betahats / shat2s * post_var_arr
                post_mean2_arr = post_var_arr + post_mean_arr * post_mean_arr # Second moment.

                # update
                mu_arr2[l, :] = post_mean_arr
                alpha_arr2[l, :] = alpha_arr
                mu2_arr2[l, :] = post_mean2_arr
                lbf_arr[l] = lbf_model

                # KL(posterior || prior) for this effect = (expected log-lik under
                # the SER posterior) − (marginal log-lik); the E[·] over the effect
                # expands resi'Vi resi into mean and second-moment terms.
                # resi'Vi(X·(α∘μ)) = (X'Vi·resi)·(α∘μ) = XtViy·(α∘μ) by symmetry of Vi,
                # so no extra Vi mat-vec is needed here.
                SER_posterior_e_loglik = - 0.5 * n * np.log(2 * np.pi) - 0.5 * V_logdet \
                            - 0.5 * ( resiVir -
                                      2 * (XtViy @ (alpha_arr * post_mean_arr)) +
                                      np.sum(xtVix * (alpha_arr * post_mean2_arr)) )
                KL_arr[l] = -loglik + SER_posterior_e_loglik

                # Add this effect back into the fitted genotype signal.
                Xresi = Xresi + X @ (alpha_arr * post_mean_arr)

            logging.info(f"Estimated prior variances: {sigma0_arr.T}")
            # Posterior correction for the fitted signal's variance:
            #   Σ_l E[(Xβ_l)'Vi(Xβ_l)] − Σ_l (E[Xβ_l])'Vi(E[Xβ_l]),
            # i.e. the extra variance from posterior uncertainty across effects.
            mean_arr2 = alpha_arr2 * mu_arr2                       # posterior mean effect α∘μ
            posterior_correction = (
                np.sum(np.sum(alpha_arr2 * mu2_arr2, axis=0) * xtVix)
                - np.sum((mean_arr2 @ xtVix_mat) * mean_arr2)
            )
            # ELBO = Gaussian data term (residual + posterior correction) − Σ KL.
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
        if self.last_snp_ids is not None and len(self.last_snp_ids) == p:
            feat_names = self.last_snp_ids
        elif self.env_int and len(self.env_int) == p:
            feat_names = list(self.env_int)
        else:
            feat_names = list(range(p))
        res_dct["alpha"] = alpha_arr2
        res_dct["mu"] = mu_arr2
        res_dct["kept_effects"] = np.where(kept)[0]
        # Feature (SNP/ENV) names, matching MMSuSiEDense so callers can map credible
        # sets to names uniformly, e.g. [[res["snp_ids"][i] for i in cs] for cs in res["cs"]].
        res_dct["snp_ids"] = [str(f) for f in feat_names]
        res_dct["pip"] = pd.DataFrame({"pip": getPIP(alpha_arr2)}, index=feat_names)
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

    # Backward-compatible alias for the pre-rename method name.
    mmsusie = fit

    def out(self, res_dct, out_file):
        """
        Write the fine-mapping result tables to ``<out_file>.{pip,alpha,mu,cs}.txt``.
        Named ``out`` to mirror :class:`MMSuSiEDense`; ``out_mmsusie`` is a
        backward-compatible alias.

        The PIP index is labelled ``SNP`` when the columns are genotype variants
        (``last_snp_ids`` set by :meth:`get_genotype`) and ``ENV`` otherwise.
        """
        pip_df = res_dct["pip"]
        env_names = pip_df.index.tolist()
        if self.last_snp_ids is not None and len(self.last_snp_ids) == len(env_names):
            pip_df.index.name = "SNP"
        else:
            pip_df.index.name = "ENV"
        pip_df.to_csv(out_file + ".pip.txt", sep="\t")
        pd.DataFrame(res_dct["alpha"], columns=env_names).to_csv(
            out_file + ".alpha.txt", sep="\t", index=False
        )
        pd.DataFrame(res_dct["mu"], columns=env_names).to_csv(
            out_file + ".mu.txt", sep="\t", index=False
        )
        with open(out_file + ".cs.txt", "w") as f:
            for vec in res_dct["cs"]:
                # str() so integer feature names (plain-matrix use, no SNP/ENV labels)
                # don't crash the join.
                f.write(" ".join([str(env_names[int(i)]) for i in vec]) + "\n")

    # Backward-compatible alias for the pre-rename method name.
    out_mmsusie = out

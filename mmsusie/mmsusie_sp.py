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
from mmsusie.utils import neg_logbf, calAlpha, getPIP, in_CS, get_CS, compute_claimed_coverage, get_cs_purity, make_sparse_block
from mmsusie.utils import neg_logbf_mix, calAlphaMix, optim_sigma0, compute_all_XtViX
from mmsusie.utils import update_Xresi_parallel, compute_betahats_parallel
from mmsusie.utils import filter_prior_components, filter_prior_components_mmsusie
from mmsusie.utils import envi_lfsr
import scipy
from scipy.optimize import minimize
from scipy import sparse
from pysnptools.snpreader import Bed
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
    delta = (np.sum(alpha_arr2 * mu2_arr2, axis=0)
             - np.sum(np.square(alpha_arr2 * mu_arr2), axis=0))

    if nvc >= 3:
        num_envi_int = env_int_arr2.shape[1]
        nxe_arr = np.sum(env_int_arr2 ** 2, axis=1) / num_envi_int  # (n,)

    V_logdet = 0.0
    rVir = 0.0
    xtVix = np.zeros(X.shape[1])
    grad = np.zeros(nvc)

    start = 0
    for i, G_b in enumerate(grm_blocks):
        if i == 0:                          # diagonal block (singletons)
            nb = len(G_b)
            if nb == 0:
                continue

            if nvc >= 3:
                nxe_b  = nxe_arr[start:start + nb]
                gxe_b  = nxe_b * G_b        # diag of GRM ⊙ EE'/K

            if nvc == 1:
                v_diag = np.full(nb, varcom[0])
                A_diags = [np.ones(nb)]
            elif nvc == 2:
                v_diag  = G_b * varcom[0] + varcom[1]
                A_diags = [G_b, np.ones(nb)]
            elif nvc == 3:
                v_diag  = G_b * varcom[0] + gxe_b * varcom[1] + varcom[2]
                A_diags = [G_b, gxe_b, np.ones(nb)]
            else:
                v_diag  = G_b * varcom[0] + gxe_b * varcom[1] + nxe_b * varcom[2] + varcom[3]
                A_diags = [G_b, gxe_b, nxe_b, np.ones(nb)]

            if np.any(v_diag <= 0):
                return np.inf, np.full(nvc, np.inf)
            vi_diag = 1.0 / v_diag
            V_logdet += np.sum(np.log(v_diag))

            r_b   = r[start:start + nb]
            X_b   = X[start:start + nb, :]
            vir_b = vi_diag * r_b
            viX_b = vi_diag[:, None] * X_b

            rVir  += np.dot(vir_b, r_b)
            xtVix += np.einsum('ij,ij->j', X_b, viX_b)

            for k, a_d in enumerate(A_diags):
                grad[k] += 0.5 * np.dot(vi_diag, a_d)
                grad[k] -= 0.5 * np.dot(vir_b ** 2, a_d)
                ViAViX_b = (vi_diag ** 2 * a_d)[:, None] * X_b
                grad[k] -= 0.5 * np.dot(delta, np.einsum('ij,ij->j', X_b, ViAViX_b))

        else:                               # dense block
            nb = G_b.shape[0]

            if nvc >= 3:
                env_b = env_int_arr2[start:start + nb, :]
                GxE_b = (env_b @ env_b.T) / num_envi_int * G_b
                nxe_b = nxe_arr[start:start + nb]

            if nvc == 1:
                V_b    = np.eye(nb) * varcom[0]
                A_mats = [np.eye(nb)]
            elif nvc == 2:
                V_b    = G_b * varcom[0] + np.eye(nb) * varcom[1]
                A_mats = [G_b, np.eye(nb)]
            elif nvc == 3:
                V_b    = G_b * varcom[0] + GxE_b * varcom[1] + np.eye(nb) * varcom[2]
                A_mats = [G_b, GxE_b, np.eye(nb)]
            else:
                V_b    = (G_b * varcom[0] + GxE_b * varcom[1]
                          + np.diag(nxe_b) * varcom[2] + np.eye(nb) * varcom[3])
                A_mats = [G_b, GxE_b, np.diag(nxe_b), np.eye(nb)]

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
            xtVix += np.einsum('ij,ij->j', X_b, viX_b)

            for k, A_k in enumerate(A_mats):
                grad[k] += 0.5 * np.sum(Vi_b * A_k)
                grad[k] -= 0.5 * (vir_b @ (A_k @ vir_b))
                ViAViX_b = A_k @ viX_b
                grad[k] -= 0.5 * np.dot(delta, np.einsum('ij,ij->j', viX_b, ViAViX_b))

        start += nb

    neg_ll = 0.5 * V_logdet + 0.5 * (rVir + np.dot(delta, xtVix))
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
               estimate_sigma=True):
        """
        End-to-end GxE fine-mapping workflow.

        Runs read_data → read_sp_grm → cal_spVi → get_env_int → get_genotype →
        get_y (GLS for covariates + E) → GLS residualize G → mmsusie on GxE → out_mmsusie.

        Args:
            pheno_file (str): Path to phenotype data file (space/tab separated).
            trait (str): Column name of the target trait.
            env_int (list): Column names for GxE interacting environmental covariates.
            grm_file (str): Prefix for the sparse GRM files.
            bedfile (str): Prefix for PLINK binary files (.bed/.bim/.fam).
            snp_id (str): Single SNP ID for GxE analysis.
            varcom_file (str): Path to file containing variance components [sigma_g2, sigma_e2].
            out_file (str): Output file prefix for result tables.
            L (int): Maximum number of non-zero effects. Defaults to 10.
            maxiter (int): Maximum IBSS iterations. Defaults to 100.
            tol (float): ELBO convergence tolerance. Defaults to 1e-3.
            coverage (float): Credible set coverage. Defaults to 0.95.
            min_abs_corr (float): Minimum absolute correlation for credible set purity. Defaults to 0.5.
            prior_tol (float): Tolerance for filtering prior components. Defaults to 1e-09.
            estimate_sigma (bool): If True, jointly re-estimate variance components during IBSS. Defaults to True.

        Returns:
            dict: Results from mmsusie().
        """
        self.read_data(pheno_file, trait, env_int)
        self.read_sp_grm(grm_file)

        E = self.get_env_int(scale=True)

        _vc = np.loadtxt(varcom_file)
        varcom = _vc[:, 0] if _vc.ndim == 2 else _vc
        self.cal_spVi(varcom)
        G = self.get_genotype(bedfile, sid_lst=[snp_id], scale=True)

        # get_y adjusts for [1, covariates, categoricals, E] via GLS
        y = self.get_y(adjust=True)

        # Additionally project out G main effect via GLS (FWL: sequential == joint)
        Vi_G = self.Vi @ G
        if scipy.sparse.issparse(Vi_G):
            Vi_G = Vi_G.toarray()
        beta_G = np.linalg.solve(G.T @ Vi_G, Vi_G.T @ y)
        y = y - G @ beta_G

        GE = G * E
        res = self.mmsusie(GE, y, L=L, maxiter=maxiter, tol=tol, coverage=coverage,
                           min_abs_corr=min_abs_corr, prior_tol=prior_tol,
                           estimate_sigma=estimate_sigma)
        self.out_mmsusie(res, out_file)
        return res

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

        self.iid_in_data = df.iloc[:, iid_col].tolist()
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

        # Filter `self.df` to retain only rows where the first column is in `iid_used`
        self.df = self.df.loc[self.df.iloc[:, 0].isin(iid_used_set)]

        # Sort `self.df` based on the order of `iid_used`
        self.df = self.df.assign(sort_order=pd.Categorical(self.df.iloc[:, 0], categories=self.iid_used, ordered=True)).sort_values(by="sort_order").drop(columns=["sort_order"])

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
            x_blocks = [np.ones((len(y), 1))]
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
            xmat = np.hstack(x_blocks)
            if self.Vi is not None:
                Vi_xmat = self.Vi @ xmat
                if scipy.sparse.issparse(Vi_xmat):
                    Vi_xmat = Vi_xmat.toarray()
                beta = np.linalg.solve(xmat.T @ Vi_xmat, Vi_xmat.T @ y)
            else:
                beta = np.linalg.lstsq(xmat, y, rcond=None)[0]
            y = y - xmat @ beta
        return y
    
    def get_genotype(self, bedfile, sid_lst=None, scale=True, *, start=None, end=None):
        """
        Get genotype matrix for self.iid_used ordered individuals.

        IID order is fixed by read_sp_grm() to match the sparse block structure.

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

        fam_file = bedfile + ".fam"
        df_fam = pd.read_csv(fam_file, sep=r"\s+", header=None, usecols=[1], dtype={1: str})
        fam_iids = pd.Index(df_fam[1])
        iid_used_index = fam_iids.get_indexer(self.iid_used)
        if np.any(iid_used_index < 0):
            missing = [self.iid_used[i] for i in np.where(iid_used_index < 0)[0]]
            raise ValueError(f"Missing iids in fam file: {missing}")

        bim_file = bedfile + ".bim"
        df_bim = pd.read_csv(bim_file, sep=r"\s+", header=None, usecols=[1], dtype={1: str})
        bim_sids = pd.Index(df_bim[1])
        if use_sid_lst:
            if len(sid_lst) == 0:
                raise ValueError("`sid_lst` cannot be empty.")
            snp_used_index = bim_sids.get_indexer(sid_lst)
            if np.any(snp_used_index < 0):
                missing = [sid_lst[i] for i in np.where(snp_used_index < 0)[0]]
                raise ValueError(f"Missing SNPs in bim file: {missing}")
        else:
            start_id, end_id = str(start), str(end)
            range_index = bim_sids.get_indexer([start_id, end_id])
            start_idx, end_idx = range_index[0], range_index[1]
            missing_ids = ([start_id] if start_idx < 0 else []) + ([end_id] if end_idx < 0 else [])
            if missing_ids:
                raise ValueError(f"Missing range SNP IDs in bim file: {missing_ids}")
            if start_idx > end_idx:
                raise ValueError(
                    f"`start` SNP ({start_id}) appears after `end` SNP ({end_id}) in bim order."
                )
            snp_used_index = np.arange(start_idx, end_idx + 1, dtype=int)

        self.last_snp_ids = bim_sids[snp_used_index].astype(str).tolist()

        snp_on_disk = Bed(bedfile, count_A1=True)
        genotype_matrix = snp_on_disk[iid_used_index, snp_used_index].read().val
        genotype_matrix = pd.DataFrame(genotype_matrix)
        genotype_matrix.fillna(genotype_matrix.mean(), inplace=True)
        genotype_matrix = genotype_matrix.values

        if scale:
            mean_genotype = np.mean(genotype_matrix, axis=0).reshape(1, -1)
            std_genotype = np.std(genotype_matrix, axis=0).reshape(1, -1)
            std_genotype[std_genotype == 0] = 1.0
            genotype_matrix = (genotype_matrix - mean_genotype) / std_genotype

        return genotype_matrix


    def cal_spVi(self, varcom):
        """
        Calculate the Vi and log|V|

        Args:
            varcom (np.ndarray): Variance components
        """
        self.varcom = np.array(varcom, dtype=float)
        Vi_sp = np.array([])
        V_logdet = 0
        if len(varcom) == 1:
            num_iid_used = len(self.iid_used)
            Vi_sp = sparse.identity(num_iid_used) / varcom[0]
            V_logdet = num_iid_used * np.log(varcom[0])
        elif len(varcom) == 2:
            V_logdet = 0
            Vi_lst = []
            for i in range(len(self.grm_blocks)):
                tmp_grm_arr2 = self.grm_blocks[i]
                num_element = tmp_grm_arr2.shape[0]
                if i == 0:
                    if num_element != 0:
                        tmp_arr2 = tmp_grm_arr2 * varcom[0] + varcom[1]
                        V_logdet += np.sum(np.log(tmp_arr2))
                        tmp_arr2 = 1 / tmp_arr2
                        Vi_lst.append(tmp_arr2)
                else:
                    tmp_arr2 = tmp_grm_arr2 * varcom[0] + np.eye(num_element) * varcom[1]
                    _, logdet = np.linalg.slogdet(tmp_arr2)
                    V_logdet += logdet
                    tmp_arr2 = np.linalg.inv(tmp_arr2)
                    Vi_lst.append(tmp_arr2)
            Vi_sp = make_sparse_block(Vi_lst)
        elif len(varcom) == 3:
            num_envi_int = self.env_int_arr2.shape[1]
            V_logdet = 0
            start_index = 0
            Vi_lst = []
            for i in range(len(self.grm_blocks)):
                tmp_grm_arr2 = self.grm_blocks[i]
                num_element = tmp_grm_arr2.shape[0]
                env_int_arr2_part = self.env_int_arr2[start_index:(start_index + num_element), :]
                if i == 0:
                    if num_element != 0:
                        tmp_gxe_arr2 = np.sum(env_int_arr2_part * env_int_arr2_part, axis=1) / num_envi_int
                        tmp_gxe_arr2 = tmp_gxe_arr2 * tmp_grm_arr2
                        tmp_arr2 = tmp_grm_arr2 * varcom[0] + tmp_gxe_arr2 * varcom[1] + varcom[2]
                        V_logdet += np.sum(np.log(tmp_arr2))
                        tmp_arr2 = 1 / tmp_arr2
                        Vi_lst.append(tmp_arr2)
                else:
                    tmp_gxe_arr2 = env_int_arr2_part @ env_int_arr2_part.T / num_envi_int
                    tmp_gxe_arr2 = tmp_gxe_arr2 * tmp_grm_arr2
                    tmp_arr2 = tmp_grm_arr2 * varcom[0] + tmp_gxe_arr2 * varcom[1] + np.eye(num_element) * varcom[2]
                    _, logdet = np.linalg.slogdet(tmp_arr2)
                    V_logdet += logdet
                    tmp_arr2 = np.linalg.inv(tmp_arr2)
                    Vi_lst.append(tmp_arr2)
                start_index += num_element
            Vi_sp = make_sparse_block(Vi_lst)
        elif len(varcom) == 4:
            num_envi_int = self.env_int_arr2.shape[1]
            nxe_arr = np.sum(self.env_int_arr2 * self.env_int_arr2, axis=1) / num_envi_int
            V_logdet = 0
            start_index = 0
            Vi_lst = []
            for i in range(len(self.grm_blocks)):
                tmp_grm_arr2 = self.grm_blocks[i]
                num_element = tmp_grm_arr2.shape[0]
                env_int_arr2_part = self.env_int_arr2[start_index:(start_index + num_element), :]
                nxe_arr_part = nxe_arr[start_index:(start_index + num_element)]
                if i == 0:
                    if num_element != 0:
                        tmp_gxe_arr2 = np.sum(env_int_arr2_part * env_int_arr2_part, axis=1) / num_envi_int
                        tmp_gxe_arr2 = tmp_gxe_arr2 * tmp_grm_arr2
                        tmp_arr2 = tmp_grm_arr2 * varcom[0] + tmp_gxe_arr2 * varcom[1] + nxe_arr_part * varcom[2] + varcom[-1]
                        V_logdet += np.sum(np.log(tmp_arr2))
                        tmp_arr2 = 1 / tmp_arr2
                        Vi_lst.append(tmp_arr2)
                else:
                    tmp_gxe_arr2 = env_int_arr2_part @ env_int_arr2_part.T / num_envi_int
                    tmp_gxe_arr2 = tmp_gxe_arr2 * tmp_grm_arr2
                    tmp_arr2 = tmp_grm_arr2 * varcom[0] + tmp_gxe_arr2 * varcom[1] + np.diag(nxe_arr_part) * varcom[2] + np.eye(num_element) * varcom[-1]
                    _, logdet = np.linalg.slogdet(tmp_arr2)
                    V_logdet += logdet
                    tmp_arr2 = np.linalg.inv(tmp_arr2)
                    Vi_lst.append(tmp_arr2)
                start_index += num_element
            Vi_sp = make_sparse_block(Vi_lst)
        self.Vi = Vi_sp
        self.V_logdet = V_logdet
    
    def mmsusie(self, X, y, L=10, maxiter=100, tol=1e-3, coverage=0.95,
                min_abs_corr=0.5, prior_tol=1e-09, estimate_sigma=True):
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
                post_mean2_arr = post_var_arr + post_mean_arr * post_mean_arr # Second moment.

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
                vX = Vi @ X
                if scipy.sparse.issparse(vX):
                    vX = vX.toarray()
                xtVix = np.einsum('ij,ij->j', X, vX)
                shat2s = 1 / xtVix
                logging.info(f"Updated varcom: {self.varcom}")
        
        alpha_arr2, mu_arr2 = filter_prior_components_mmsusie(alpha_arr2, mu_arr2, sigma0_arr, prior_tol)
        if self.last_snp_ids is not None and len(self.last_snp_ids) == p:
            feat_names = self.last_snp_ids
        elif self.env_int and len(self.env_int) == p:
            feat_names = list(self.env_int)
        else:
            feat_names = list(range(p))
        res_dct["alpha"] = alpha_arr2
        res_dct["mu"] = mu_arr2
        res_dct["pip"] = pd.DataFrame({"pip": getPIP(alpha_arr2)}, index=feat_names)
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
    
    def create_mixture_prior(self):
        nE = self.env_int_arr2.shape[1]
        U_lst = [np.identity(nE)]
        mixture_weights_lst = [0.5]
        U_rank_lst = [np.linalg.matrix_rank(U_lst[0])]
        U_pinv_lst = [np.linalg.pinv(U_lst[0])]
        for i in range(nE):
            cov = np.zeros((nE, nE))
            cov[i, i] = 1
            U_lst.append(cov)
            mixture_weights_lst.append(0.5 / nE)
            U_rank_lst.append(np.linalg.matrix_rank(cov))
            U_pinv_lst.append(np.linalg.pinv(cov))
        prior_cov = {
            "U": U_lst,
            "mixture_weights": np.array(mixture_weights_lst),
            "U_rank": np.array(U_rank_lst),
            "U_pinv": U_pinv_lst,
        }
        return prior_cov
    

    def mrsusie(self, G, E, y, prior_cov, L=10, prior_weights=None, maxiter=100, tol=1e-3, coverage=0.95,
                min_abs_corr=0.5, n_jobs=8, estimate_prior_method="optim", prior_tol=1e-09):
        """
        Run MR-SuSiE for GxE fine-mapping with a sparse block-diagonal GRM.

        Each effect is modelled as a multivariate (Q-dimensional) GxE vector
        for a single SNP, using a mixture-of-normals prior on the effect size.

        Args:
            G (np.ndarray): Genotype matrix (n, J), standardized.
            E (np.ndarray): Environmental covariate matrix (n, Q), standardized.
            y (np.ndarray): Phenotype vector (n,), GRM-adjusted and standardized.
            prior_cov (np.ndarray): Prior covariance matrices (K, Q, Q) for the
                mixture components.
            L (int): Maximum number of non-zero effects. Defaults to 10.
            prior_weights (np.ndarray or None): Prior inclusion probability per SNP
                (length J). Defaults to uniform 1/J.
            maxiter (int): Maximum IBSS iterations. Defaults to 100.
            tol (float): ELBO convergence tolerance. Defaults to 1e-3.
            coverage (float): Credible set coverage. Defaults to 0.95.
            min_abs_corr (float): Minimum purity for credible sets. Defaults to 0.5.
            n_jobs (int): Parallel workers for per-SNP computations. Defaults to 8.
            estimate_prior_method (str): Method for prior variance optimisation
                (``"optim"``). Defaults to ``"optim"``.
            prior_tol (float): Threshold for pruning negligible effects. Defaults to 1e-9.

        Returns:
            dict: Keys include ``alpha`` (L, J), ``mu`` (L, J, Q), ``pip`` (J,),
                ``lfsr`` (J, Q), ``lfdr`` (J, Q), ``cs`` (list of index arrays),
                ``lfsr_cs`` (per-CS lfsr), ``claimed_coverage``.
        """
        logging.info("Starting mrsusie...")
        
        J = G.shape[1] # number of SNPs
        # X_lst = [G[:, [j]] * E for j in range(J)] # SNPs * environmental covariates
        Q = E.shape[1] # number of environmental covariates
        n = G.shape[0] # number of individuals
        if J < L:
            L = J
        yVar = np.var(y)

        logging.info("Calculating shat2s...")
        Shat_inv_lst, Shat_lst, logdet_S_hat_arr = compute_all_XtViX(G, E, self.Vi, n_jobs=n_jobs)

        # Initialize susie fit
        if prior_weights is None:
            prior_weights = np.full(J, 1.0 / J)  # uniform prior weights for each variable having the non-zero effect
        
        # Initialize prior inclusion probabilities (PIPs)
        alpha_LJ = np.full((L, J), 1.0 / J)

        # Initialize posterior means and second moments as 3D numpy arrays for better performance
        mu_LJQ = np.zeros((L, J, Q)) # Posterior means
        S1_LJQQ = np.zeros((L, J, Q, Q)) # Posterior second moments
        
        # Initialize list of posterior probability metrics
        zero_prob_LJQ = np.zeros((L, J, Q))
        neg_prob_LJQ = np.zeros((L, J, Q))
        clfsr_LJQ = np.zeros((L, J, Q))
        lfsr_LJQ = np.zeros((L, J, Q))
        lfdr_LJQ = np.zeros((L, J, Q))

        # Initialize residual fit, Bayes factors, prior variances, and ELBO
        Xresi = np.zeros(n)  # fitted values
        KL_L = np.full(L, np.nan)
        lbf_L = np.full(L, np.nan) # log Bayes factors
        sigma0_L = np.full(L, yVar * 0.2) # Prior variance for each effect
        elbo_arr = np.full(maxiter + 1, np.nan) # ELBO values
        elbo_arr[0] = -np.inf

        # Empty result dictionary
        res_dct = {}

        for iter in range(maxiter):
            logging.info(f"Iteration: {iter + 1}")

            # update each effect once
            for l in range(L):
                logging.info(f"    {l + 1}th effect")

                # Remove lth effect from fitted values
                delta = update_Xresi_parallel(G, E, alpha_LJ[l, :], mu_LJQ[l, :, :], n_jobs=n_jobs)
                Xresi = Xresi - delta

                # Compute residuals
                resi = y - Xresi

                # Bayesian single-effect linear regression using residuals as outcomes
                Viy = self.Vi @ resi
                betahats_lst = compute_betahats_parallel(G, E, Viy, Shat_lst, n_jobs=n_jobs)
                
                # optimize the prior variance
                sigma0 = sigma0_L[l]
                res = optim_sigma0(sigma0, prior_cov, betahats_lst, Shat_inv_lst, Shat_lst, logdet_S_hat_arr,
                                    prior_weights, method=estimate_prior_method, maxiter=1, tol=1e-9)
                
                if res[1]:
                    sigma0 = res[0]
                    sigma0_L[l] = sigma0
                else:
                    logging.warning("Optimization of priors failed; using priors from the previous iteration.")
                
                alpha_J, lbf_model, b1_mix_JQ, S1_mix_JQQ, zero_prob_JQ, neg_prob_JQ, clfsr, lfsr, lfdr = \
                        calAlphaMix(sigma0, prior_cov, betahats_lst, Shat_inv_lst, Shat_lst, logdet_S_hat_arr, prior_weights)
                loglik = lbf_model - 0.5 * n * np.log(2 * np.pi) - 0.5 * self.V_logdet - \
                            0.5 * (resi @ (self.Vi @ resi))
                
                # update
                mu_LJQ[l] = b1_mix_JQ
                S1_LJQQ[l] = S1_mix_JQQ
                zero_prob_LJQ[l] = zero_prob_JQ
                neg_prob_LJQ[l] = neg_prob_JQ
                clfsr_LJQ[l] = clfsr
                lfsr_LJQ[l] = lfsr
                lfdr_LJQ[l] = lfdr
                alpha_LJ[l, :] = alpha_J
                lbf_L[l] = lbf_model
                delta = update_Xresi_parallel(G, E, alpha_LJ[l, :], mu_LJQ[l], n_jobs=n_jobs)
                resi_Xb = resi - delta
                SER_posterior_e_loglik = - 0.5 * n * np.log(2 * np.pi) - 0.5 * self.V_logdet \
                            - 0.5 * ( resi_Xb @ (self.Vi @ resi_Xb) + 
                                      np.sum([np.sum(Shat_inv_lst[j] * (alpha_J[j] * S1_mix_JQQ[j])) for j in range(J)]) )
                KL_L[l] = -loglik + SER_posterior_e_loglik
                Xresi = Xresi + delta
            
            logging.info(f"Estimated prior variances: {sigma0_L.T}")
            elbo_arr[iter + 1] = - 0.5 * n * np.log(2 * np.pi) - 0.5 * self.V_logdet \
                    - 0.5 * ( (y - Xresi) @ (self.Vi @ (y - Xresi)) + 
                    np.sum([np.sum([np.sum(Shat_inv_lst[j] * (alpha_LJ[l, j] * S1_LJQQ[l, j, :, :])) for j in range(J)]) for l in range(L)]) ) - np.sum(KL_L)
            logging.info(f"ELBO: {elbo_arr[iter + 1]}")
            if np.absolute(elbo_arr[iter + 1] - elbo_arr[iter]) < tol: 
                break
        alpha_LJ, mu_LJQ, zero_prob_LJQ, neg_prob_LJQ, clfsr_LJQ, lfsr_LJQ = \
            filter_prior_components(alpha_LJ, mu_LJQ, zero_prob_LJQ, neg_prob_LJQ, clfsr_LJQ, lfsr_LJQ, sigma0_L, prior_tol)
        res_dct["alpha"] = alpha_LJ
        res_dct["mu"] = mu_LJQ
        res_dct["pip"] = getPIP(alpha_LJ)
        res_dct["lfdr"] = np.min(np.stack(lfdr_LJQ, axis=0), axis=0)
        res_dct["lfsr"] = np.min(np.stack(lfsr_LJQ, axis=0), axis=0)
        status = in_CS(alpha_LJ, coverage)
        cs_lst = get_CS(status)
        claimed_coverage_arr = compute_claimed_coverage(cs_lst, alpha_LJ)
        cs_lst, claimed_coverage_arr = get_cs_purity(cs_lst, claimed_coverage_arr, G, min_abs_corr)
        res_dct["cs"] = cs_lst
        res_dct["lfsr_cs"] = envi_lfsr(alpha_LJ, clfsr_LJQ, cs_lst)
        res_dct["claimed_coverage"] = claimed_coverage_arr
        res_dct["lbf"] = lbf_L
        res_dct["sigma0"] = sigma0_L
        res_dct["elbo"] = elbo_arr
        res_dct["KL"] = KL_L
        return res_dct
    
    def out_mmsusie(self, res_dct, out_file):
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
                f.write(" ".join([env_names[int(i)] for i in vec]) + "\n")
    
    def out_mrsusie(self, res_dct, out_file):
        snp_ids = self.last_snp_ids
        has_ids = snp_ids is not None and len(snp_ids) == res_dct["alpha"].shape[1]
        if has_ids:
            pd.DataFrame({"pip": res_dct["pip"]}, index=snp_ids).rename_axis("SNP").to_csv(
                out_file + ".pip.txt", sep="\t"
            )
            pd.DataFrame(res_dct["alpha"], columns=snp_ids).to_csv(
                out_file + ".alpha.txt", sep="\t", index=False
            )
            pd.DataFrame(res_dct["lfsr"], index=snp_ids).to_csv(
                out_file + ".lfsr.txt", sep="\t"
            )
            pd.DataFrame(res_dct["lfdr"], index=snp_ids).to_csv(
                out_file + ".lfdr.txt", sep="\t"
            )
            with open(out_file + ".cs.txt", "w") as f:
                for vec in res_dct["cs"]:
                    f.write(" ".join([snp_ids[int(i)] for i in vec]) + "\n")
        else:
            np.savetxt(out_file + ".pip.txt", res_dct["pip"])
            np.savetxt(out_file + ".alpha.txt", res_dct["alpha"])
            np.savetxt(out_file + ".lfsr.txt", res_dct["lfsr"])
            np.savetxt(out_file + ".lfdr.txt", res_dct["lfdr"])
            with open(out_file + ".cs.txt", "w") as f:
                for vec in res_dct["cs"]:
                    f.write(" ".join([str(int(i)) for i in vec]) + "\n")
        np.save(out_file + ".mu.npy", res_dct["mu"])
        np.savetxt(out_file + ".lfsr_cs.txt", res_dct["lfsr_cs"])

    
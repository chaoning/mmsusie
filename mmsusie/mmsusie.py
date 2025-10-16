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
import logging
from mmsusie.utils import neg_logbf, calAlpha, getPIP, in_CS, get_CS, compute_claimed_coverage, get_cs_purity, make_sparse_block
from mmsusie.utils import neg_logbf_mix, calAlphaMix, optim_sigma0, compute_all_XtViX
from mmsusie.utils import update_Xresi_parallel, compute_betahats_parallel
from mmsusie.utils import filter_prior_components, filter_prior_components_mmsusie
from mmsusie.utils import envi_lfsr
import scipy
from scipy.optimize import minimize
from scipy import sparse
from numpy.linalg import slogdet
from pysnptools.snpreader import Bed
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from scipy.stats import chi2


class MMSuSiE:
    def __init__(self):
        self.iid_used = None
        self.iid_in_data = None
        self.iid_in_grm = None
        self.df = None # data frame

        self.trait = None  # Column name of the target trait.
        self.env_int = [] # List of column names for interacting environmental covariates
        
        self.grm_mat_dense = None # Genetic relationship matrix
        self.grm_blocks = [] # List of blocks for GRMs which are clustered and sorted by group size
        self.env_int_arr2 = None # numpy array for interacting environmental covariates
        
        self.Vi = None # Inverse of V

        self.V_logdet = 0 # log|V|
    
    def run(self, pheno_file, trait, env_int, grm_file, bedfile, snp_id, varcom_file, out_file,
            L=10, maxiter=100, tol=1e-3, coverage=0.95, min_abs_corr=0.5, prior_tol = 1e-09):
        """
        Main function to run the MMSuSiE analysis.

        Args:
            pheno_file (str): Path to the phenotype data file.
            trait (str): Column name of the target trait.
            env_int (list): List of column names for interacting environmental covariates.
            grm_file (str): Prefix path to the GRM files.
            bedfile (str): Path to the plink binary file.
            snp_id (str): SNP ID to include in the genotype matrix.
            varcom_file (str): Path to the file containing variance components.
            out_file (str): Path to the output file.
            L (int, optional): Maximum number of non-zero effects. Defaults to 10.
            maxiter (int, optional): Maximum number of iterations. Defaults to 100.
            tol (float, optional): Convergence tolerance for ELBO. Defaults to 1e-3.
            coverage (float, optional): Credible set coverage. Defaults to 0.95.
            min_abs_corr (float, optional): Minimum absolute correlation for credible set purity. Defaults to 0.5.
            prior_tol (float, optional): Tolerance for filtering prior components. Defaults to 1e-09.

        Returns:
            dict: Results from the MMSuSiE analysis.
        """
        # Step 1: Read and preprocess data
        self.read_data(pheno_file, trait, env_int)

        # Step 2: Read GRM and align with phenotype data
        self.read_sp_grm(grm_file)

        # Step 3: Extract environmental covariates and phenotype
        E = self.get_env_int(scale=True)            # (n, K)

        # Step 4: Get genotype matrix (force (n,1) for single SNP)
        G = self.get_genotype(bedfile, [snp_id], scale=True)

        # Build main-effects design with intercept
        n = G.shape[0]
        X = np.hstack((np.ones((n, 1)), G, E))       # (n, 1+1+K)

        # Phenotype, then project out main effects (OLS-safe; see GLS note below)
        y = self.get_y(adjust=False, scale=False)
        XtX = X.T @ X
        Xty = X.T @ y
        # Solve (X'X) beta = X'y
        beta = np.linalg.solve(XtX, Xty)
        y = y - X @ beta

        # Standardize y
        y = (y - np.mean(y)) / np.std(y)

        # Step 5: Load variance components and calculate Vi and log|V|
        varcom = np.loadtxt(varcom_file)[:, 0]
        self.cal_spVi(varcom)

        # Step 6: Build GE interactions
        GE = G * E                                   # (n, K), broadcasts over K envs

        # Step 7: Run MMSuSiE on GE interactions
        res = self.mmsusie(GE, y, L=L, maxiter=maxiter, tol=tol, coverage=coverage,
                        min_abs_corr=min_abs_corr, prior_tol=prior_tol)
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

    def read_data(self, data_file, trait, env_int=[], iid_col=0):
        """
        Read and preprocess the data file.

        Args:
            data_file (str): Path to the input data file. Space/Tab seperated.
            trait (str): Column name of the target trait.
            env_int (list): List of column names for interacting environmental covariates
            iid_col (int, optional): Index of the column containing individual IDs. Defaults to 0.

        Return:
            None
        """
        self.trait = trait
        self.env_int = env_int
        
        # Read head line and get the iid column name
        iid_column_name = []
        with open(data_file, 'r') as f:
            head_line = f.readline()
            head_line = head_line.strip()
            lst = head_line.split()
            iid_column_name = lst[iid_col]
        
        # Combine used column names
        usedcols_lst = [iid_column_name, trait] + list(env_int)

        # Check for duplicates and raise an error if found
        if len(usedcols_lst) != len(set(usedcols_lst)):
            duplicated = [col for col in usedcols_lst if usedcols_lst.count(col) > 1]
            raise ValueError(f"Duplicate column names detected: {set(duplicated)}")
        
        # Define column types:
        # - Categorical columns (strings): ID
        # - Numeric columns (floats): interactions, and trait
        str_cols = [iid_column_name]
        float_cols = list(env_int) + [trait]

        # Build dtype mapping for pandas
        dtype_map = {col: str for col in str_cols if col in usedcols_lst}
        dtype_map.update({col: float for col in float_cols if col in usedcols_lst})

        # Read the file with specific column selection and type enforcement
        df = pd.read_csv(data_file, sep=r"\s+", usecols=usedcols_lst, dtype=dtype_map)

        # Handle missing values before conversion
        initial_rows = df.shape[0]
        df = df.dropna()
        dropped_rows = initial_rows - df.shape[0]
        if dropped_rows > 0:
            logging.warning(f"Dropped {dropped_rows} rows due to missing values.")
        
        # Store first column values and check for duplicates
        self.iid_in_data = df.iloc[:, iid_col].tolist()
        if len(set(self.iid_in_data)) != len(self.iid_in_data):
            raise ValueError("Duplicated IIDs in data file!")
        
        self.iid_used = self.iid_in_data[:]
        logging.info(f"The number of used IIDs in data file: {len(self.iid_in_data)}")
        self.df = df


    def read_grm(self, grm_prefix):
        """
        Load the genetic relationship matrix (GRM) and align it with phenotype data.

        Args:
            grm_prefix (str): Prefix path to the GRM files (expects .agrm.id and .agrm.mat_fmt)
        """
        # Step 1: files
        grm_id_file = f"{grm_prefix}.agrm.id"
        grm_mat_file = f"{grm_prefix}.agrm.mat_fmt"

        # Step 2: Read GRM individual IDs
        self.iid_in_grm = pd.read_csv(grm_id_file, sep=r"\s+", header=None, dtype={0: str}).iloc[:, 0].tolist()
        logging.info(f"Total individuals in GRM: {len(self.iid_in_grm)}")

        # Step 3: Determine overlapping individuals (GRM ∩ phenotype)
        if self.iid_in_data:
            iid_overlap_set = set(self.iid_in_data) & set(self.iid_in_grm)
            if not iid_overlap_set:
                raise ValueError("No overlapping individuals between GRM and phenotype data.")
            logging.info(f"Number of overlapping individuals: {len(iid_overlap_set)}")
        else:
            iid_overlap_set = set(self.iid_in_grm)

        # Step 4: Filter phenotype data to retain only overlapping individuals
        self.df = self.df[self.df.iloc[:, 0].isin(iid_overlap_set)]

        # Step 5: Extract ordered list of used IIDs from phenotype data
        self.iid_used = self.df.iloc[:, 0].tolist()

        # Step 6: Map used IIDs to GRM matrix row/column indices
        grm_iid_to_idx = {iid: idx for idx, iid in enumerate(self.iid_in_grm)}
        try:
            grm_idx = [grm_iid_to_idx[iid] for iid in self.iid_used]
        except KeyError as e:
            raise ValueError(f"Missing IID in GRM that exists in phenotype: {e}")
        
        # Step 7: Load and subset GRM matrix
        grm_full = np.loadtxt(grm_mat_file, dtype=np.float64)
        self.grm_mat_dense = grm_full[np.ix_(grm_idx, grm_idx)]
    
        
    def read_sp_grm(self, grm_file):
        """
        Read the sparse genetic relationship matrix and update the data frame

        Args:
            grm_file (str): Prefix for the genetic relationship matrix

        Raises:
            ValueError: None
        """
        
        # Read the GRM group file
        df_group = pd.read_csv(grm_file + ".agrm.group", sep=r"\s+", header=None,
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
            f"{grm_file}.agrm.ind_fmt",
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

        if df_sub_group.shape[0] > 1:
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
        Get the interacting environmental covariates matrix
        """
        self.env_int_arr2 = self.df.loc[:, self.env_int].values
        if scale:
            mean_arr = np.mean(self.env_int_arr2, axis=0).reshape(1, -1)
            std_arr = np.std(self.env_int_arr2, axis=0).reshape(1, -1)
            self.env_int_arr2 = (self.env_int_arr2 - mean_arr) / std_arr
        return self.env_int_arr2
    
    def get_y(self, adjust=True, scale=True):
        """
        Get the target trait values after adjusting for environmental covariates
        and scaling.

        Args:
            adjust (bool, optional): whether to adjust the trait values by interacting environmental covariates. Defaults to True.
            scale (bool, optional): whether to scale the trait values. Defaults to True.

        Returns:
            np.ndarray: Target trait values. 1D numpy array.
        """
        y = self.df.loc[:, self.trait].values
        if adjust:
            # Adjust y by environmental covariates
            y = y - self.env_int_arr2 @ np.linalg.inv(self.env_int_arr2.T @ self.env_int_arr2) @ (self.env_int_arr2.T @ y)
        if scale:
            # Scale y
            mean_y = np.mean(y)
            std_y = np.std(y)
            y = (y - mean_y) / std_y
        return y
    
    def get_genotype(self, bedfile, sid_lst, scale=True):
        """
        Get the genotype matrix from plink binary file.

        Args:
            bedfile (str): Path to the plink binary file.
            sid_lst (list): List of SNP IDs to include in the genotype matrix.
            scale (bool, optional): whether to scale the genotype matrix. Defaults to True.

        Returns:
            np.ndarray: Genotype matrix. 2D numpy array.
        """
        # get the index of the used individuals in the fam file
        fam_file = bedfile + ".fam"
        df_fam = pd.read_csv(fam_file, sep=r"\s+", header=None, dtype={0: str, 1: str})
        missing_iids = set(self.iid_used) - set(df_fam[1].tolist())
        if missing_iids:
            raise ValueError(f"Missing iids in fam file: {missing_iids}")
        dct = {df_fam.iloc[i, 1]: i for i in range(df_fam.shape[0])}
        iid_used_index = [dct[iid] for iid in self.iid_used]
        
        # Read the bim file and get the index of the used SNPs
        bim_file = bedfile + ".bim"
        df_bim = pd.read_csv(bim_file, sep=r"\s+", header=None, dtype={0: str, 1: str})
        missing_snps = set(sid_lst) - set(df_bim[1].tolist())
        if missing_snps:
            raise ValueError(f"Missing SNPs in bim file: {missing_snps}")
        dct = {df_bim.iloc[i, 1]: i for i in range(df_bim.shape[0])}
        snp_used_index = [dct[sid] for sid in sid_lst]
        
        # Read the genotype matrix from the bed file
        snp_on_disk = Bed(bedfile, count_A1=True)
        genotype_matrix = snp_on_disk[iid_used_index, snp_used_index].read().val
        genotype_matrix = pd.DataFrame(genotype_matrix)
        mean_genotype = genotype_matrix.mean()
        genotype_matrix.fillna(mean_genotype, inplace=True)
        genotype_matrix = genotype_matrix.values
        
        if scale:
            # Scale the genotype matrix
            mean_genotype = np.mean(genotype_matrix, axis=0).reshape(1, -1)
            std_genotype = np.std(genotype_matrix, axis=0).reshape(1, -1)
            genotype_matrix = (genotype_matrix - mean_genotype) / std_genotype
        
        return genotype_matrix


    def cal_spVi(self, varcom):
        """
        Calculate the Vi and log|V|

        Args:
            varcom (np.ndarray): Variance components
        """
        Vi_sp = np.array([])
        V_logdet = 0
        if len(varcom) == 1:
            num_iid_used = len(self.iid_used)
            Vi_sp = sparse.identity(num_iid_used) / varcom[0]
            V_logdet = num_iid_used * np.log(1/varcom[0])
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
            Vi_sp = make_sparse_block(Vi_lst)
        self.Vi = Vi_sp
        self.V_logdet = V_logdet
    
    def cal_Vi(self, varcom):
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
        n = self.grm_mat_dense.shape[0]
        V = sigma_g2 * self.grm_mat_dense + sigma_e2 * np.identity(n)

        # Compute log-determinant using slogdet
        sign, logdet = slogdet(V)
        if sign <= 0:
            raise ValueError("Covariance matrix V is not positive definite; log-determinant undefined.")

        self.V_logdet = logdet

        # Compute V inverse (can be replaced with cho_solve for better numerical stability if needed)
        self.Vi = np.linalg.inv(V)
        
    
    def mmsusie(self, X, y, L=10, maxiter=100, tol=1e-3, coverage=0.95, 
                min_abs_corr=0.5, prior_tol = 1e-09):
        p = X.shape[1]
        n = X.shape[0]
        if p < L:
            L = p
        yVar = np.var(y)
        
        logging.info("Starting mmsusie...")
        logging.info("Calculating shat2s...")
        vX = self.Vi @ X
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
                XtViy = X.T @ (self.Vi @ resi)
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
                loglik = lbf_model - 0.5 * n * np.log(2 * np.pi) - 0.5 * self.V_logdet - \
                            0.5 * (resi @ (self.Vi @ resi))
                
                post_var_arr = 1 / (1 / sigma0 + 1 / shat2s) # Posterior variance.
                post_mean_arr = betahats / shat2s * post_var_arr
                post_mean2_arr = post_var_arr + post_mean_arr * post_mean_arr; # Second moment.

                # update
                mu_arr2[l, :] = post_mean_arr
                alpha_arr2[l, :] = alpha_arr
                mu2_arr2[l, :] = post_mean2_arr
                lbf_arr[l] = lbf_model

                SER_posterior_e_loglik = - 0.5 * n * np.log(2 * np.pi) - 0.5 * self.V_logdet \
                            - 0.5 * ( resi @ (self.Vi @ resi) - 
                                      2 * np.sum(resi @ (self.Vi @ (X @ (alpha_arr * post_mean_arr)))) + 
                                      np.sum(xtVix * (alpha_arr * post_mean2_arr)) )
                KL_arr[l] = -loglik + SER_posterior_e_loglik
                Xresi = Xresi + X @ (alpha_arr * post_mean_arr)
            
            logging.info(f"Estimated prior variances: {sigma0_arr.T}")
            elbo_arr[iter + 1] = - 0.5 * n * np.log(2 * np.pi) - 0.5 * self.V_logdet \
                    - 0.5 * ( (y - Xresi) @ (self.Vi @ (y - Xresi)) + 
                    np.sum(np.sum(alpha_arr2 * mu2_arr2, axis=0) * xtVix) -
                    np.sum(np.sum(np.square(alpha_arr2 * mu_arr2), axis=0) * xtVix)) - np.sum(KL_arr)
            logging.info(f"ELBO: {elbo_arr[iter + 1]}")
            if np.absolute(elbo_arr[iter + 1] - elbo_arr[iter]) < tol: 
                break
        
        alpha_arr2, mu_arr2 = filter_prior_components_mmsusie(alpha_arr2, mu_arr2, sigma0_arr, prior_tol)
        res_dct["alpha"] = alpha_arr2
        res_dct["mu"] = mu_arr2
        res_dct["pip"] = getPIP(alpha_arr2)
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
    

    def mrsusie(self, G, E, y, prior_cov, L=10, prior_weights = None, maxiter=100, tol=1e-3, coverage=0.95, 
                min_abs_corr=0.5, n_jobs=8, estimate_prior_method="optim", prior_tol = 1e-09):
        """_summary_

        Args:
            G (_type_): _description_
            E (_type_): _description_
            y (_type_): _description_
            prior_cov (_type_): _description_
            L (int, optional): _description_. Defaults to 10.
            maxiter (int, optional): _description_. Defaults to 100.
            tol (_type_, optional): _description_. Defaults to 1e-3.
            coverage (float, optional): _description_. Defaults to 0.95.
            min_abs_corr (float, optional): _description_. Defaults to 0.5.
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
        np.savetxt(out_file + ".pip.txt", res_dct["pip"])
        np.savetxt(out_file + ".alpha.txt", res_dct["alpha"])
        np.savetxt(out_file + ".mu.txt", res_dct["mu"])
        with open(out_file + ".cs.txt", "w") as fin:
            for vec in res_dct["cs"]:
                fin.write(" ".join([str(int(i)) for i in vec]) + "\n")
    
    def out_mrsusie(self, res_dct, out_file):
        np.savetxt(out_file + ".pip.txt", res_dct["pip"])
        np.savetxt(out_file + ".alpha.txt", res_dct["alpha"])
        np.savetxt(out_file + ".lfsr.txt", res_dct["lfsr"])
        np.savetxt(out_file + ".lfdr.txt", res_dct["lfdr"])
        with open(out_file + ".cs.txt", "w") as fin:
            for vec in res_dct["cs"]:
                fin.write(" ".join([str(int(i)) for i in vec]) + "\n")
        np.save(out_file + ".mu.npy", res_dct["mu"])
        np.savetxt(out_file + ".lfsr_cs.txt", res_dct["lfsr_cs"])
        
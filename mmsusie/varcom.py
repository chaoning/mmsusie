import logging
import os

import numpy as np
import pandas as pd
from scipy import linalg


class WeightEMAI:
    """
    Weighted EM-AI estimator for variance components in mixed models.

    The estimator supports the V-matrix formulation via `fit`.
    """

    def __init__(self, maxiter=100, cc_par=1.0e-8, step=0.01):
        self.maxiter = int(maxiter)
        self.cc_par = float(cc_par)
        self.step = float(step)

        if self.maxiter < 1:
            raise ValueError("maxiter must be >= 1.")
        if self.cc_par <= 0:
            raise ValueError("cc_par must be > 0.")
        if self.step <= 0:
            raise ValueError("step must be > 0.")

        self.weight_vec = self._build_weight_vec(self.step)
        self.last_iterations = 0
        self.last_converged = False
        self.last_update_norm = np.inf

    @staticmethod
    def _build_weight_vec(step):
        weight_vec = list(np.arange(0.0, 1.0, step))
        if not weight_vec or not np.isclose(weight_vec[-1], 1.0):
            weight_vec.append(1.0)
        return weight_vec

    def _validate_common_inputs(self, y, xmat, gmat_lst, init):
        if len(gmat_lst) < 1:
            raise ValueError("gmat_lst must contain at least one matrix.")
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n = y.shape[0]
        xmat = np.asarray(xmat, dtype=float).reshape(n, -1)

        gmat_checked = []
        for idx, gmat in enumerate(gmat_lst):
            gmat_arr = np.asarray(gmat, dtype=float)
            if gmat_arr.ndim != 2 or gmat_arr.shape[0] != gmat_arr.shape[1]:
                raise ValueError(f"gmat_lst[{idx}] must be a square 2D matrix.")
            if gmat_arr.shape[0] != n:
                raise ValueError(
                    f"gmat_lst[{idx}] size {gmat_arr.shape} does not match phenotype size ({n})."
                )
            gmat_checked.append(gmat_arr)

        num_var = len(gmat_checked) + 1
        var_com = np.ones(num_var, dtype=float)
        if init is not None:
            init = np.asarray(init, dtype=float).reshape(-1)
            if init.size != num_var:
                raise ValueError(f"init must contain {num_var} values.")
            if np.any(init <= 0):
                raise ValueError("All initial variance components must be positive.")
            var_com = init.copy()

        return y, xmat, gmat_checked, var_com, n, num_var

    @staticmethod
    def _relative_update_norm(delta, var_com_new):
        numerator = float(np.sum(delta * delta))
        denominator = float(np.sum(var_com_new * var_com_new))
        if denominator == 0.0:
            return np.inf
        return np.sqrt(numerator / denominator)

    @staticmethod
    def _build_em_matrix(var_com, random_sizes, n):
        diag = [size / (2.0 * var_com[idx] * var_com[idx]) for idx, size in enumerate(random_sizes)]
        diag.append(n / (2.0 * var_com[-1] * var_com[-1]))
        return np.diag(diag)

    def _find_positive_update(self, fd_mat, ai_mat, em_mat, var_com):
        for weight in self.weight_vec:
            wemai_mat = weight * em_mat + (1.0 - weight) * ai_mat
            try:
                delta = linalg.solve(wemai_mat, fd_mat)
            except linalg.LinAlgError:
                continue

            var_com_new = var_com + delta
            if np.min(var_com_new) > 0:
                return weight, wemai_mat, delta, var_com_new

        raise RuntimeError(
            "Failed to find a positive variance update. "
            "Try a smaller step, different initialization, or check model conditioning."
        )

    def fit(self, y, xmat, gmat_lst, init=None):
        """
        Estimate variance components with the V-matrix formulation.

        Parameters
        ----------
        y : array-like
            Phenotype vector.
        xmat : array-like
            Design matrix for fixed effects.
        gmat_lst : list
            List of relationship matrices for random effects. Each matrix must be n x n.
        init : array-like, optional
            Initial variance components, length = len(gmat_lst) + 1.

        Returns
        -------
        numpy.ndarray
            Estimated variance components.
        """
        logging.info("##### Prepare (vmat) #####")
        y, xmat, gmat_lst, var_com, n, num_var = self._validate_common_inputs(y, xmat, gmat_lst, init)

        random_sizes = [gmat.shape[0] for gmat in gmat_lst]
        logging.info("Initial variances: %s", " ".join(np.asarray(var_com, dtype=str)))
        logging.info("##### Start iteration #####")

        cc_par_val = np.inf
        converged = False
        for iter_idx in range(1, self.maxiter + 1):
            logging.info("Round: %d", iter_idx)

            # Build V = sum_k sigma_k * G_k + sigma_e * I.
            vmat = np.eye(n, dtype=float) * var_com[-1]
            for k, gmat in enumerate(gmat_lst):
                vmat += var_com[k] * gmat

            # P = V^-1 - V^-1 X (X'V^-1X)^-1 X'V^-1
            vmat_inv = linalg.inv(vmat)
            vxmat = vmat_inv @ xmat
            xvxmat_inv = linalg.inv(xmat.T @ vxmat)
            pmat = vmat_inv - vxmat @ xvxmat_inv @ vxmat.T
            pymat = pmat @ y

            # First derivatives and AI matrix.
            fd_mat = np.zeros(num_var, dtype=float)
            fd_mat[-1] = -0.5 * (np.trace(pmat) - float((pymat.T @ pymat)[0, 0]))

            wv_list = []
            for k, gmat in enumerate(gmat_lst):
                fd_part = np.sum(pmat * gmat) - float((pymat.T @ gmat @ pymat)[0, 0])
                fd_mat[k] = -0.5 * fd_part
                wv_list.append(gmat @ pymat)

            wv_list.append(pymat)
            wv_mat = np.concatenate(wv_list, axis=1)
            ai_mat = 0.5 * (wv_mat.T @ pmat @ wv_mat)

            em_mat = self._build_em_matrix(var_com, random_sizes, n)
            weight, wemai_mat, delta, var_com_new = self._find_positive_update(fd_mat, ai_mat, em_mat, var_com)

            cc_par_val = self._relative_update_norm(delta, var_com_new)
            var_com = var_com_new

            logging.info("EM weight value: %.6f", weight)
            logging.debug("fd matrix:\n%s", fd_mat)
            logging.debug("AI matrix:\n%s", ai_mat)
            logging.debug("weighted matrix:\n%s", wemai_mat)
            logging.info("Norm of update vector: %.6e", cc_par_val)
            logging.info("Updated variances: %s", " ".join(np.asarray(var_com, dtype=str)))

            if cc_par_val < self.cc_par:
                converged = True
                break

        self.last_iterations = iter_idx
        self.last_update_norm = cc_par_val
        self.last_converged = converged

        if converged:
            logging.info("Variances converged.")
        else:
            logging.info("Variances not converged.")
        return var_com

def _normalize_grm_prefix(grm_prefix):
    """
    Normalize GRM prefix by stripping a trailing `.grm` suffix if present.

    Examples:
    - input: /path/to/out        -> /path/to/out
    - input: /path/to/out.grm    -> /path/to/out
    """
    base = str(grm_prefix).strip()
    if base.endswith(".grm"):
        return base[:-4]
    return base


def _find_first_existing_file(candidates, description):
    for file_path in candidates:
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError(
        f"Unable to find {description}. Tried: {candidates}"
    )


def _read_grm_ids(grm_prefix):
    """
    Read GRM id file: <prefix>.grm.id
    """
    base = _normalize_grm_prefix(grm_prefix)
    id_file = _find_first_existing_file(
        [f"{base}.grm.id"],
        "GRM ID file",
    )

    df_id = pd.read_csv(id_file, sep=r"\s+", header=None, dtype=str)
    if df_id.shape[1] >= 2:
        # FID is ignored; only IID (2nd column) is used.
        iid_series = df_id.iloc[:, 1].astype(str)
    elif df_id.shape[1] == 1:
        iid_series = df_id.iloc[:, 0].astype(str)
    else:
        raise ValueError(f"Invalid GRM ID file format: {id_file}")

    if iid_series.duplicated().any():
        raise ValueError(f"Duplicate IID found in GRM ID file: {id_file}")

    out = pd.DataFrame({"IID": iid_series})
    return out, id_file


def _load_grm_matrix(grm_prefix, n_id):
    """
    Load dense GRM matrix generated by gmatrix.py.

    Supported formats:
    - *.grm.matrix       (dense text matrix)
    - *.grm.index_triplet (lower-triangle row/col/value, 1-based)
    """
    base = _normalize_grm_prefix(grm_prefix)
    mat_file_candidates = [
        f"{base}.grm.matrix",
    ]
    ind_file_candidates = [
        f"{base}.grm.index_triplet",
    ]

    existing_mat = [p for p in mat_file_candidates if os.path.exists(p)]
    if existing_mat:
        mat_file = existing_mat[0]
        grm = np.loadtxt(mat_file, dtype=float)
        if grm.ndim == 0:
            grm = np.array([[float(grm)]], dtype=float)
        elif grm.ndim == 1:
            if n_id == 1:
                grm = grm.reshape(1, 1)
            elif grm.size == n_id * n_id:
                grm = grm.reshape(n_id, n_id)
            else:
                raise ValueError(f"Unexpected GRM matrix shape in file: {mat_file}")
        if grm.shape != (n_id, n_id):
            raise ValueError(
                f"GRM matrix shape {grm.shape} does not match ID count ({n_id}) in file: {mat_file}"
            )
        return grm, mat_file

    existing_ind = [p for p in ind_file_candidates if os.path.exists(p)]
    if existing_ind:
        ind_file = existing_ind[0]
        df_ind = pd.read_csv(
            ind_file,
            sep=r"\s+",
            header=None,
            names=["row", "col", "val"],
            dtype={"row": int, "col": int, "val": float},
        )
        grm = np.zeros((n_id, n_id), dtype=float)
        row = df_ind["row"].to_numpy(dtype=int) - 1
        col = df_ind["col"].to_numpy(dtype=int) - 1
        val = df_ind["val"].to_numpy(dtype=float)
        if row.min(initial=0) < 0 or col.min(initial=0) < 0 or row.max(initial=0) >= n_id or col.max(initial=0) >= n_id:
            raise ValueError(f"GRM index out of range in file: {ind_file}")
        grm[row, col] = val
        grm[col, row] = val
        return grm, ind_file

    raise FileNotFoundError(
        f"Unable to find GRM matrix file. "
        f"Tried: {mat_file_candidates + ind_file_candidates}"
    )


def prepare_varcom_inputs(
    data_file,
    trait_col,
    grm_prefix,
    categorical_cols=None,
    covariate_cols=None,
    iid_col="IID",
    sep=r"\s+",
    add_intercept=True,
    drop_first=True,
):
    """
    Read phenotype/covariate file and align it directly to GRM individuals.

    Assumption:
    - One phenotype row corresponds to one individual.
    - Therefore Z is identity and is not explicitly constructed.

    Parameters
    ----------
    data_file : str
        Phenotype/covariate file path. "IID" column is used by default.
    trait_col : str
        Column name of phenotype.
    categorical_cols : list[str], optional
        Categorical columns used in fixed effects (one-hot encoded).
    covariate_cols : list[str], optional
        Numeric covariate columns used in fixed effects.
    grm_prefix : str
        Prefix of GRM files generated by gmatrix.py.
        Example: if files are out.grm.id and out.grm.matrix, grm_prefix should be "out".
    iid_col : int or str, default="IID"
        IID column. If int, uses column index from file header.
        If str, uses the provided column name (case-insensitive).
        If the resolved column is FID and an IID column exists, IID will be used.
    sep : str, default=r"\\s+"
        Separator for data file.
    add_intercept : bool, default=True
        If True, prepend intercept column to xmat.
    drop_first : bool, default=True
        If True, drop first level in one-hot encoding for categorical variables.

    Returns
    -------
    dict
        {
            "data": aligned phenotype dataframe,
            "y": phenotype array (n, 1),
            "xmat": fixed-effect matrix (n, p),
            "gmat": aligned GRM matrix (n, n),
            "used_iids": IID list corresponding to gmat rows/cols,
            "x_columns": fixed-effect column names,
            "grm_id_file": resolved GRM ID file path,
            "grm_matrix_file": resolved GRM matrix file path,
        }
    """
    if grm_prefix is None or str(grm_prefix).strip() == "":
        raise ValueError("grm_prefix cannot be None or empty.")

    categorical_cols = list(categorical_cols or [])
    covariate_cols = list(covariate_cols or [])

    header_cols = pd.read_csv(data_file, sep=sep, nrows=0).columns.tolist()
    header_cols_lower_map = {str(col).strip().lower(): col for col in header_cols}
    if isinstance(iid_col, int):
        if iid_col < 0 or iid_col >= len(header_cols):
            raise ValueError(f"iid_col index {iid_col} is out of range for data file.")
        iid_col_name = header_cols[iid_col]
    else:
        iid_col_requested = str(iid_col).strip()
        iid_col_name = header_cols_lower_map.get(iid_col_requested.lower())
        if iid_col_name is None:
            raise ValueError(f'iid_col "{iid_col_requested}" not found in data file columns.')
    if str(iid_col_name).strip().lower() == "fid":
        iid_candidate = header_cols_lower_map.get("iid")
        if iid_candidate is None:
            raise ValueError("Selected iid_col resolves to FID, but no IID column was found in data file.")
        logging.info(
            "Ignoring FID column '%s'; using IID column '%s' for alignment.",
            iid_col_name,
            iid_candidate,
        )
        iid_col_name = iid_candidate

    required_cols = [iid_col_name, trait_col] + categorical_cols + covariate_cols
    if len(required_cols) != len(set(required_cols)):
        raise ValueError("Duplicate column names detected in required inputs.")

    dtype_map = {iid_col_name: str}
    df = pd.read_csv(data_file, sep=sep, usecols=required_cols, dtype=dtype_map)

    for col in [trait_col] + covariate_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_drop = df.shape[0]
    df = df.dropna(subset=required_cols).copy()
    dropped_missing = before_drop - df.shape[0]
    if dropped_missing > 0:
        logging.info("Dropped %d rows with missing values in required columns.", dropped_missing)
    if df.empty:
        raise ValueError("No valid rows remain after dropping missing values.")

    # One individual should have one row in phenotype table.
    if df[iid_col_name].astype(str).duplicated().any():
        raise ValueError("Duplicate IID found in phenotype file. Expected one row per individual.")

    grm_ids, grm_id_file = _read_grm_ids(grm_prefix)
    grm_full, grm_matrix_file = _load_grm_matrix(grm_prefix, n_id=grm_ids.shape[0])

    # Match by IID only.
    grm_keys = grm_ids["IID"].astype(str).tolist()
    data_keys = df[iid_col_name].astype(str).tolist()

    data_key_set = set(data_keys)
    grm_key_set = set(grm_keys)
    if not data_key_set:
        raise ValueError("No valid phenotype IDs available after filtering.")

    missing_in_grm = data_key_set - grm_key_set
    if missing_in_grm:
        preview = list(missing_in_grm)[:5]
        raise ValueError(
            f"{len(missing_in_grm)} phenotype IDs are not found in GRM. Examples: {preview}"
        )
    missing_in_data = grm_key_set - data_key_set
    if missing_in_data:
        logging.info(
            "%d GRM IDs are not found in phenotype file and will be excluded.",
            len(missing_in_data),
        )

    # Align phenotype rows to GRM order, then subset GRM accordingly.
    data_row_map = {key: idx for idx, key in enumerate(data_keys)}
    used_grm_idx = [idx for idx, key in enumerate(grm_keys) if key in data_key_set]
    if len(used_grm_idx) == 0:
        raise ValueError("No overlapping individuals between phenotype data and GRM.")
    ordered_keys = [grm_keys[idx] for idx in used_grm_idx]
    ordered_data_idx = [data_row_map[key] for key in ordered_keys]
    df = df.iloc[ordered_data_idx].reset_index(drop=True)

    used_grm_idx = np.asarray(used_grm_idx, dtype=int)
    gmat = grm_full[np.ix_(used_grm_idx, used_grm_idx)]
    grm_ids_used = grm_ids.iloc[used_grm_idx].reset_index(drop=True)
    if gmat.shape[0] != df.shape[0]:
        raise ValueError("Aligned phenotype rows and GRM size are inconsistent.")

    # Build fixed effect matrix.
    x_blocks = []
    x_columns = []
    if add_intercept:
        x_blocks.append(np.ones((df.shape[0], 1), dtype=float))
        x_columns.append("Intercept")

    if covariate_cols:
        x_blocks.append(df[covariate_cols].to_numpy(dtype=float))
        x_columns.extend(covariate_cols)

    if categorical_cols:
        cat_df = pd.get_dummies(
            df[categorical_cols].astype("category"),
            drop_first=drop_first,
            dtype=float,
        )
        if cat_df.shape[1] > 0:
            x_blocks.append(cat_df.to_numpy(dtype=float))
            x_columns.extend(cat_df.columns.tolist())

    if x_blocks:
        xmat = np.hstack(x_blocks)
    else:
        xmat = np.empty((df.shape[0], 0), dtype=float)

    y = df[trait_col].to_numpy(dtype=float).reshape(-1, 1)

    logging.info(
        "Prepared varcom inputs: n_obs=%d, n_grm=%d, x_cols=%d",
        df.shape[0],
        gmat.shape[0],
        xmat.shape[1],
    )

    return {
        "data": df,
        "y": y,
        "xmat": xmat,
        "gmat": gmat,
        "used_iids": grm_ids_used["IID"].astype(str).tolist(),
        "iid_col": iid_col_name,
        "x_columns": x_columns,
        "grm_id_file": grm_id_file,
        "grm_matrix_file": grm_matrix_file,
    }


if __name__ == "__main__":
    # get current script directory and set it as working directory
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.chdir("../example")
    data_file = "data.txt"
    grm_prefix = "output/test_grm"
    trait_col = "pheno"
    covariate_cols = ["cov1", "cov2", "cov3"]
    inputs = prepare_varcom_inputs(
        data_file=data_file,
        trait_col=trait_col,
        grm_prefix=grm_prefix,
        iid_col="IID",
        covariate_cols=covariate_cols,
    )
    varcom_estimator = WeightEMAI(maxiter=100, cc_par=1e-6, step=0.1)
    var_com = varcom_estimator.fit(
        y=inputs["y"],
        xmat=inputs["xmat"],
        gmat_lst=[inputs["gmat"]],
    )
    print("Estimated variance components:", var_com)

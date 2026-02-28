import argparse
import logging
import time

import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed
from tqdm import tqdm


VALID_METHODS = {"vanraden", "gcta"}
VALID_OUTPUT_FORMATS = {"mat", "row_col_val", "id_id_val"}


def output_mat(mat: np.ndarray, id_df: pd.Series, out_file: str, out_fmt: str) -> None:
    """
    Write matrix to disk in one of the supported formats.

    Supported formats:
    - mat: dense matrix text
    - row_col_val: lower triangle in row/col/value
    - id_id_val: lower triangle in id/id/value
    """
    if out_fmt not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f'Unknown output format "{out_fmt}". '
            f"Use one of {sorted(VALID_OUTPUT_FORMATS)}."
        )

    if out_fmt == "mat":
        np.savetxt(f"{out_file}.mat_fmt", mat)
        return

    indices = np.tril_indices_from(mat)
    if out_fmt == "row_col_val":
        df = pd.DataFrame(
            {
                "row": indices[0] + 1,
                "col": indices[1] + 1,
                "val": mat[indices],
            }
        )
        df.to_csv(
            f"{out_file}.ind_fmt",
            sep=" ",
            index=False,
            header=False,
            columns=["row", "col", "val"],
        )
        return

    ids = np.asarray(id_df)
    df = pd.DataFrame(
        {
            "id0": ids[indices[0]],
            "id1": ids[indices[1]],
            "val": mat[indices],
        }
    )
    df.to_csv(
        f"{out_file}.id_fmt",
        sep=" ",
        index=False,
        header=False,
        columns=["id0", "id1", "val"],
    )


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in VALID_METHODS:
        raise ValueError(f'Unknown method "{method}". Use one of {sorted(VALID_METHODS)}.')
    return normalized


def _build_block_starts(num_snp: int, npart: int) -> np.ndarray:
    if num_snp < 1:
        raise ValueError("No SNPs found in input bed file.")
    if npart < 1:
        raise ValueError("npart must be >= 1.")
    return np.linspace(0, num_snp, npart + 1, dtype=int)


def _impute_missing_by_column_mean(snp_mat: np.ndarray) -> np.ndarray:
    """
    Mean-impute missing genotypes column-wise.

    For each SNP column, NaN is replaced with the observed column mean.
    If an entire SNP column is missing, it is filled with 0.
    """
    missing_mask = np.isnan(snp_mat)
    if not missing_mask.any():
        return snp_mat

    valid_counts = np.sum(~missing_mask, axis=0)
    col_sums = np.nansum(snp_mat, axis=0)
    col_means = np.divide(
        col_sums,
        valid_counts,
        out=np.zeros_like(col_sums, dtype=np.float64),
        where=valid_counts > 0,
    )
    snp_mat[missing_mask] = col_means[np.where(missing_mask)[1]]
    return snp_mat


def agmat(
    bed_file: str,
    out_file: str,
    out_fmt: str = "mat",
    npart: int = 10,
    small_val: float = 0.001,
    method: str = "vanraden",
) -> np.ndarray:
    """
    Compute additive genomic relationship matrix (GRM).

    This function computes GRM using either VanRaden or GCTA normalization.

    Method:
    - vanraden: G = W W' / sum_j(2 p_j (1 - p_j))
    - gcta:     G = Z Z' / M, where each SNP is standardized by sqrt(2 p_j (1 - p_j))
    """
    if small_val < 0:
        raise ValueError("small_val must be >= 0.")
    method = _normalize_method(method)

    logging.info("{:#^80}".format(f"Read SNP and compute GRM ({method.upper()})"))
    snp_on_disk = Bed(bed_file, count_A1=True)
    num_snp = snp_on_disk.sid_count
    num_id = snp_on_disk.iid_count
    logging.info("Detected %s individuals and %s SNPs.", num_id, num_snp)

    block_starts = _build_block_starts(num_snp, npart)
    block_count = len(block_starts) - 1

    t0 = time.perf_counter()
    scale = 0.0
    n_snp_used = 0
    kin = np.zeros((num_id, num_id), dtype=np.float64)

    for idx in tqdm(range(block_count), desc="GRM blocks", unit="block"):
        start = block_starts[idx]
        end = block_starts[idx + 1]
        if start == end:
            continue

        snp_mat = snp_on_disk[:, start:end].read().val.astype(np.float64, copy=False)
        snp_mat = _impute_missing_by_column_mean(snp_mat)

        freq = np.mean(snp_mat, axis=0) / 2.0
        denom = 2.0 * freq * (1.0 - freq)
        valid = np.isfinite(denom) & (denom > 0.0)
        if not np.any(valid):
            continue

        centered = snp_mat[:, valid] - 2.0 * freq[valid]
        if method == "vanraden":
            kin += centered @ centered.T
            scale += float(np.sum(denom[valid]))
        else:
            standardized = centered / np.sqrt(denom[valid])
            kin += standardized @ standardized.T
            n_snp_used += int(np.sum(valid))

    if method == "vanraden":
        if scale <= 0.0:
            raise ValueError("No valid polymorphic SNPs were found for VanRaden GRM.")
        kin /= scale
    else:
        if n_snp_used == 0:
            raise ValueError("No valid polymorphic SNPs were found for GCTA GRM.")
        kin /= n_snp_used

    kin = 0.5 * (kin + kin.T)
    if small_val > 0.0:
        diag_idx = np.diag_indices_from(kin)
        kin[diag_idx] *= 1.0 + small_val

    elapsed = time.perf_counter() - t0
    logging.info("Finished GRM computation in %.3f seconds.", elapsed)

    logging.info("{:#^80}".format("Write outputs"))
    fam_info = pd.read_csv(f"{bed_file}.fam", sep=r"\s+", header=None)
    logging.info("Output prefix: %s", out_file)
    fam_info.iloc[:, 1].to_csv(f"{out_file}.id", index=False, header=False, sep=" ")
    output_mat(kin, fam_info.iloc[:, 1], f"{out_file}.grm", out_fmt)
    return kin



if __name__ == "__main__":
    # get current directory and change to it, so that relative paths work as expected
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("../example")
    os.makedirs("output", exist_ok=True)  
    print("Current working directory:", os.getcwd())
    bed_file = "test"
    out_file = "output/test_grm"
    kin = agmat(bed_file, out_file, out_fmt="mat", npart=5, small_val=0.001, method="vanraden")
    print("mean kinship diagonal:", np.mean(np.diag(kin)))
    print("mean kinship off-diagonal:", np.mean(kin[np.tril_indices_from(kin, k=-1)]))


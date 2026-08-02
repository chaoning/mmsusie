"""
Genotype / association-file IO helpers shared by the MMSuSiE workflows.

Kept separate from ``utils.py`` (pure numpy/scipy statistics) so that the
file-reading concerns (PLINK bed/bim/fam, association tables) live in one place.

Author: Chao Ning
"""

import logging

import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed


def read_genotype_matrix(bedfile, iid_lst, sid_lst=None, scale=True, *, start=None, end=None):
    """
    Read a genotype matrix for the given individuals and SNPs from PLINK
    binary files (``.bed``/``.bim``/``.fam``).

    Individuals are returned in the order of ``iid_lst``. Exactly one SNP
    selector must be provided:

    1) ``sid_lst`` — explicit SNP ids
    2) ``start`` + ``end`` — inclusive SNP id range in ``.bim`` order

    Missing genotypes are mean-imputed per SNP; when ``scale=True`` each SNP is
    standardized (monomorphic SNPs stay 0 after centering).

    Args:
        bedfile (str): PLINK binary prefix (without ``.bed``/``.bim``/``.fam``).
        iid_lst (list): Individual IDs; also fixes the row order of the output.
        sid_lst (list or None): Explicit SNP ids (mutually exclusive with range).
        scale (bool): Standardize each SNP column. Defaults to True.
        start, end (str or None): Inclusive SNP id range in ``.bim`` order.

    Returns:
        tuple[np.ndarray, list[str]]: (genotype matrix ``(n, m)``, SNP ids used).
    """
    use_sid_lst = sid_lst is not None
    use_range = start is not None or end is not None
    if use_sid_lst == use_range:
        raise ValueError("Use exactly one SNP selector: either `sid_lst` or `start`+`end`.")
    if use_range and (start is None or end is None):
        raise ValueError("Both `start` and `end` are required when using range selection.")
    if iid_lst is None or len(iid_lst) == 0:
        raise ValueError("`iid_lst` cannot be empty.")

    # Map requested individuals to fam-file rows.
    fam_file = bedfile + ".fam"
    df_fam = pd.read_csv(fam_file, sep=r"\s+", header=None, usecols=[1], dtype={1: str})
    fam_iids = pd.Index(df_fam[1])
    iid_used_index = fam_iids.get_indexer(iid_lst)
    if np.any(iid_used_index < 0):
        missing_iids = [iid_lst[i] for i in np.where(iid_used_index < 0)[0]]
        raise ValueError(f"Missing iids in fam file: {missing_iids}")

    # Build SNP indexes from either explicit ids or an id range.
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

    snp_used_ids = bim_sids[snp_used_index].astype(str).tolist()

    # Read, mean-impute, and optionally standardize.
    snp_on_disk = Bed(bedfile, count_A1=True)
    genotype_matrix = snp_on_disk[iid_used_index, snp_used_index].read().val
    genotype_matrix = pd.DataFrame(genotype_matrix)
    genotype_matrix.fillna(genotype_matrix.mean(), inplace=True)
    # A column that is entirely missing has a NaN column-mean, so the fill above
    # leaves it NaN (which would then propagate through standardization). Fill any
    # such fully-missing SNP with 0 and warn instead of silently emitting NaNs.
    all_missing = [c for c in genotype_matrix.columns if genotype_matrix[c].isna().all()]
    if all_missing:
        miss_ids = [snp_used_ids[c] for c in all_missing]
        logging.warning(
            f"{len(all_missing)} SNP(s) entirely missing; filled with 0: "
            f"{miss_ids[:10]}{' ...' if len(miss_ids) > 10 else ''}"
        )
        genotype_matrix.fillna(0.0, inplace=True)
    genotype_matrix = genotype_matrix.values

    if scale:
        mean_genotype = np.mean(genotype_matrix, axis=0).reshape(1, -1)
        std_genotype = np.std(genotype_matrix, axis=0).reshape(1, -1)
        std_genotype[std_genotype == 0] = 1.0
        genotype_matrix = (genotype_matrix - mean_genotype) / std_genotype

    return genotype_matrix, snp_used_ids


def ld_prune_assoc(assoc_file, bed_file, ld_r2=0.1, snp="SNP", p="p_gxe", p_cutoff=5e-8):
    """
    Greedy LD pruning of significant association hits.

    Keeps SNPs with ``p < p_cutoff``, then in ascending-``p`` order greedily
    selects lead SNPs, dropping any remaining SNP whose ``r^2`` to a chosen lead
    exceeds ``ld_r2``. Returns the association rows for the retained lead SNPs.

    Args:
        assoc_file (str): Whitespace-separated association table.
        bed_file (str): PLINK binary prefix used to compute LD.
        ld_r2 (float): r^2 threshold for pruning. Defaults to 0.1.
        snp (str): SNP-id column name in the association table. Defaults to "SNP".
        p (str): p-value column name. Defaults to "p_gxe".
        p_cutoff (float): Significance threshold. Defaults to 5e-8.

    Returns:
        pandas.DataFrame: Association rows for the retained lead SNPs.
    """
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

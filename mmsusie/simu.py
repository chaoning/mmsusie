import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed


def _impute_by_column_mean(x):
    x = np.asarray(x, dtype=float)
    miss = np.isnan(x)
    if not miss.any():
        return x
    counts = np.sum(~miss, axis=0)
    means = np.divide(
        np.nansum(x, axis=0),
        counts,
        out=np.zeros(x.shape[1], dtype=float),
        where=counts > 0,
    )
    x[miss] = means[np.where(miss)[1]]
    return x


def _standardize_columns(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (x - mean) / std


def _scale_to_variance(x, target_var):
    if target_var <= 0:
        return np.zeros_like(x, dtype=float)
    var_x = float(np.var(x))
    if var_x <= 0:
        return np.zeros_like(x, dtype=float)
    return x / np.sqrt(var_x) * np.sqrt(target_var)


def simulate_pheno_from_bed(
    bed_file,
    out_file,
    n_cov=3,
    n_cat=0,
    n_level=3,
    n_causal=200,
    h2=0.5,
    cov_h2=0.01,
    cat_h2=0.005,
    trait_name="pheno",
    seed=42,
):
    """
    Simulate phenotype using genotype from a PLINK bed prefix.
    Output columns: FID IID cov* cat* <trait_name>
    Causal SNP file shares the same prefix as out_file.
    """
    if h2 < 0 or cov_h2 < 0 or cat_h2 < 0:
        raise ValueError("h2/cov_h2/cat_h2 must be >= 0.")
    if h2 + cov_h2 + cat_h2 >= 1:
        raise ValueError("h2 + cov_h2 + cat_h2 must be < 1 to leave variance for noise.")
    if n_cov < 0 or n_cat < 0:
        raise ValueError("n_cov and n_cat must be >= 0.")
    if n_level < 2 and n_cat > 0:
        raise ValueError("n_level must be >= 2 when n_cat > 0.")

    rng = np.random.default_rng(seed)

    bed = Bed(bed_file, count_A1=True)
    n = bed.iid_count
    m = bed.sid_count
    if n == 0 or m == 0:
        raise ValueError("Empty genotype matrix from bed file.")

    fam_file = f"{bed_file}.fam"
    fam = pd.read_csv(fam_file, sep=r"\s+", header=None, dtype=str)
    if fam.shape[0] != n:
        raise ValueError("Row count mismatch between .bed and .fam.")

    n_causal = int(max(1, min(n_causal, m)))
    causal_idx = rng.choice(m, size=n_causal, replace=False)
    causal_sid = np.asarray(bed.sid)[causal_idx]
    # Read only causal SNPs to reduce memory usage.
    geno_causal = bed[:, causal_idx].read().val
    geno_causal = _impute_by_column_mean(geno_causal)
    g_causal = _standardize_columns(geno_causal)
    beta_g = rng.normal(0.0, 1.0, size=n_causal)
    g_raw = g_causal @ beta_g
    g_eff = _scale_to_variance(g_raw, h2)

    cov_names = [f"cov{i + 1}" for i in range(n_cov)]
    if n_cov > 0:
        cov = rng.normal(size=(n, n_cov))
        beta_cov = rng.normal(size=n_cov)
        cov_raw = cov @ beta_cov
        cov_eff = _scale_to_variance(cov_raw, cov_h2)
    else:
        cov = np.empty((n, 0), dtype=float)
        cov_eff = np.zeros(n, dtype=float)

    cat_names = [f"cat{i + 1}" for i in range(n_cat)]
    if n_cat > 0:
        cat = rng.integers(0, n_level, size=(n, n_cat))
        cat_df = pd.DataFrame(cat, columns=cat_names)
        cat_dummies = pd.get_dummies(cat_df.astype("category"), drop_first=True, dtype=float)
        if cat_dummies.shape[1] > 0:
            beta_cat = rng.normal(size=cat_dummies.shape[1])
            cat_raw = cat_dummies.to_numpy() @ beta_cat
        else:
            cat_raw = np.zeros(n, dtype=float)
        cat_eff = _scale_to_variance(cat_raw, cat_h2)
    else:
        cat = np.empty((n, 0), dtype=int)
        cat_eff = np.zeros(n, dtype=float)

    noise_var = 1.0 - (h2 + cov_h2 + cat_h2)
    noise = rng.normal(0.0, np.sqrt(noise_var), size=n)
    y = g_eff + cov_eff + cat_eff + noise

    out_df = pd.DataFrame(
        {
            "IID": fam.iloc[:, 1].astype(str).to_numpy(),
        }
    )
    for i, col in enumerate(cov_names):
        out_df[col] = cov[:, i]
    for i, col in enumerate(cat_names):
        out_df[col] = cat[:, i].astype(int)
    out_df[trait_name] = y

    out_path = Path(out_file)
    out_prefix = out_path.with_suffix("") if out_path.suffix else out_path
    out_path = out_prefix.with_suffix(".txt")
    out_df.to_csv(out_path, sep="\t", index=False)

    causal_path = out_path.with_name(f"{out_prefix.name}.causal_snps.txt")

    causal_df = pd.DataFrame(
        {
            "causal_rank": np.arange(1, n_causal + 1, dtype=int),
            "snp_index0": causal_idx.astype(int),
            "snp_index1": (causal_idx + 1).astype(int),
            "snp_id": causal_sid.astype(str),
            "beta_g": beta_g,
            "abs_beta_g": np.abs(beta_g),
        }
    )
    causal_df = causal_df.sort_values("abs_beta_g", ascending=False).reset_index(drop=True)
    causal_df.to_csv(causal_path, sep="\t", index=False)

    logging.info("Saved simulated phenotype file: %s", out_path)
    logging.info("Saved causal SNP file: %s", causal_path)
    logging.info("n=%d, m=%d, n_causal=%d", n, m, n_causal)
    return out_df


def _build_parser():
    parser = argparse.ArgumentParser(description="Simulate phenotype with random covariates/categorical variables from bed.")
    parser.add_argument("--bed-file", required=True, help="PLINK bed prefix (without .bed/.bim/.fam).")
    parser.add_argument("--out-file", required=True, help="Output file path (.txt).")
    parser.add_argument("--n-cov", type=int, default=3, help="Number of continuous covariates.")
    parser.add_argument("--n-cat", type=int, default=0, help="Number of categorical variables.")
    parser.add_argument("--n-level", type=int, default=3, help="Number of levels for each categorical variable.")
    parser.add_argument("--n-causal", type=int, default=200, help="Number of causal SNPs.")
    parser.add_argument("--h2", type=float, default=0.5, help="Variance proportion of genetic effect.")
    parser.add_argument("--cov-h2", type=float, default=0.01, help="Variance proportion of covariate effects.")
    parser.add_argument("--cat-h2", type=float, default=0.005, help="Variance proportion of categorical effects.")
    parser.add_argument("--trait-name", default="pheno", help="Output phenotype column name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    simulate_pheno_from_bed(
        bed_file=args.bed_file,
        out_file=args.out_file,
        n_cov=args.n_cov,
        n_cat=args.n_cat,
        n_level=args.n_level,
        n_causal=args.n_causal,
        h2=args.h2,
        cov_h2=args.cov_h2,
        cat_h2=args.cat_h2,
        trait_name=args.trait_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

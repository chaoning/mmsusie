# MMSuSiE

MMSuSiE is a Python package for mixed-model SuSiE fine-mapping.
It provides an end-to-end workflow for:

- building additive genetic relationship matrices (GRM) from PLINK files,
- estimating variance components by weighted EM-AI REML — dense (`WeightEMAI`) or
  sparse block-diagonal with 1-4 components for GxE (`WeightEMAISp`),
- running MMSuSiE fine-mapping with GRM-adjusted (GLS) covariance and full
  Frisch–Waugh–Lovell covariate adjustment, returning per-SNP posterior inclusion
  probabilities (PIP) and purity-filtered credible sets.

## Project Layout

```
mmsusie/
├── mmsusie_dense.py   # Dense-GRM workflow (MMSuSiEDense)
├── mmsusie_sp.py      # Sparse block-diagonal GRM workflow (MMSuSiESp)
├── utils.py           # Statistical utility functions
├── io.py              # Genotype / association-file readers (shared)
├── varcom.py          # Variance component estimation (WeightEMAI dense / WeightEMAISp sparse)
├── gmatrix.py         # GRM construction (agmat)
└── simu.py            # Phenotype simulation
```

Two classes are exported at the package level:

| Class | File | GRM type | Key methods |
|---|---|---|---|
| `MMSuSiEDense` | `mmsusie_dense.py` | Dense (`cal_Vi`) | `fit`, `process_y`, `out` |
| `MMSuSiESp` | `mmsusie_sp.py` | Sparse block-diagonal (`cal_spVi`) | `fit`, `get_y`, `get_fixed`, `out` |

Also exported: `agmat` (GRM), `WeightEMAI` / `WeightEMAISp` (variance components),
`prepare_varcom_inputs`.

## Requirements

- Python >= 3.8
- numpy >= 1.22
- pandas >= 1.3
- scipy >= 1.7
- pysnptools >= 0.5
- tqdm >= 4.60

## Installation

```bash
git clone https://github.com/chaoning/mmsusie.git
cd mmsusie
pip install .
```

## End-to-End API Example

### Dense GRM (`MMSuSiEDense`)

```python
import os, logging
logging.basicConfig(level=logging.INFO)

from mmsusie import agmat, MMSuSiEDense, WeightEMAI, prepare_varcom_inputs

os.chdir("example")
os.makedirs("output", exist_ok=True)

# 1) Build GRM from test.bed/.bim/.fam
agmat("test", "output/test_grm")

# 2) Align phenotype/covariates with GRM
inputs = prepare_varcom_inputs(
    data_file="data.txt",
    trait_col="pheno",
    grm_prefix="output/test_grm",
    covariate_cols=["cov1", "cov2", "cov3"],
)

# 3) Estimate variance components
var_com = WeightEMAI().fit(
    y=inputs["y"],
    xmat=inputs["xmat"],
    gmat_lst=[inputs["gmat"]],
)

# 4) Run MMSuSiEDense with full FWL covariate adjustment
ms = MMSuSiEDense()
ms.cal_Vi(inputs["gmat"], var_com)
G = ms.get_genotype(
    "test",
    iid_lst=inputs["used_iids"],
    start="rs2165666",
    end="rs4863332",
)
# fixed=inputs["xmat"] gives full Frisch–Waugh–Lovell: the covariates are projected
# out of BOTH y and the genotype in the V^{-1} metric (no separate process_y needed).
# estimate_sigma=True jointly re-estimates the variance components and refreshes the
# projection each IBSS sweep.
result = ms.fit(G, inputs["y"].flatten(), L=10, estimate_sigma=True, fixed=inputs["xmat"])

# 5) Export the FWL-projected y and G (for an apples-to-apples susieR comparison)
import pandas as pd
y_adj = ms._gls_residualize(inputs["y"].flatten(), inputs["xmat"])
G_adj = ms._gls_residualize(G, inputs["xmat"])
df_out = pd.DataFrame({"IID": inputs["used_iids"], "y_adj": y_adj})
df_G = pd.DataFrame(G_adj, columns=result["snp_ids"])
pd.concat([df_out, df_G], axis=1).to_csv("output/test_mmsusie_data.txt", sep="\t", index=False)

# 6) Export result tables
ms.out(result, out_file="output/test_mmsusie")
```

The example region (`rs2165666 … rs4863332`) contains **two independent causal
SNPs** — `rs1487590` and `rs1462069` (in low mutual LD) — so fine-mapping returns
**two credible sets**, one per signal, each pinpointing its causal SNP (PIP ≈ 1).
Both `MMSuSiEDense` / `MMSuSiESp` and `susieR` recover the same two sets.

### Comparison with susieR

Read `output/test_mmsusie_data.txt` in R and run `susie()` on the same `y_adj` and `G`:

```r
library(susieR)

dat <- read.table("output/test_mmsusie_data.txt", header = TRUE, sep = "\t")
y   <- dat$y_adj
G   <- as.matrix(dat[, -(1:2)])   # drop IID and y_adj columns

fit <- susie(G, y, L = 10)

# PIP
pip_df <- data.frame(SNP = colnames(G), pip = susie_get_pip(fit))
write.table(pip_df, "output/susieR_pip.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# Credible sets (SNP names)
cs  <- susie_get_cs(fit)
cs_named <- lapply(cs$cs, function(idx) colnames(G)[idx])
print(cs_named)
```

### Sparse Block-diagonal GRM (`MMSuSiESp`)

Before running Python, use `fastgxe` ([download](https://github.com/chaoning/fastGxE)) to build the sparse GRM
(run from the `example/` directory):

```bash
# 1) Build GRM from PLINK files
fastgxe --make-grm --bfile test --out ./output/test

# 2) Compute relatedness groups (threshold 0.05)
fastgxe --process-grm --group --grm ./output/test --cut-value 0.05

# 3) Reformat GRM to sparse index-triplet format
#    --out must share the same prefix as --grm so that read_sp_grm()
#    can find both .grm.group and .grm.index_triplet under one prefix.
fastgxe --process-grm --reformat --sparse --grm ./output/test --out-fmt 1 --out ./output/test
```

> On some clusters `fastgxe` (Intel OpenMP) aborts with a `KMP_AFFINITY` assertion.
> If so, prefix the commands with `KMP_AFFINITY=disabled`.

These commands produce the sparse GRM files `MMSuSiESp` reads:

| File | Used by |
|---|---|
| `output/test.grm.group` | `read_sp_grm` (sample IDs + relatedness groups) |
| `output/test.grm.index_triplet` | `read_sp_grm` (GRM values) |

`fastgxe` is only needed to build the sparse GRM (steps 1-3). **Variance components
are estimated natively with `WeightEMAISp`** (weighted EM-AI REML on the sparse
block-diagonal GRM, 1-4 components incl. GxE) — no `fastgxe --test-main` needed. On
the example data it reproduces `fastgxe`'s REML to ~4 decimals; because it inverts the
GRM block by block, its cost tracks the relatedness structure rather than `n` itself.

```python
import os, logging
logging.basicConfig(level=logging.INFO)

from mmsusie import MMSuSiESp, WeightEMAISp

os.chdir("example")
os.makedirs("output", exist_ok=True)

# 1) Load phenotype and sparse GRM
ms = MMSuSiESp()
ms.read_data("data.txt", trait="pheno", covariate_cols=["cov1", "cov2", "cov3"])
ms.read_sp_grm("output/test")

# 2) Estimate variance components on the sparse GRM (native REML) -> build sparse V^{-1}
#    Pass the raw phenotype + fixed-effect design (intercept + covariates). For GxE
#    variance components use n_varcom=3 or 4 and pass env_int_arr2=ms.get_env_int().
varcom = WeightEMAISp().fit(ms.get_y(adjust=False), ms.get_fixed(), ms.grm_blocks, n_varcom=2)
ms.cal_spVi(varcom)  # varcom = [sigma_g2, sigma_e2]

# 3) Raw phenotype (fixed= below projects it; no separate GLS pre-adjustment needed)
y = ms.get_y(adjust=False)

# 4) Load genotype for region of interest
G = ms.get_genotype("test", start="rs2165666", end="rs4863332")

# 5) Run MMSuSiE with full FWL covariate adjustment
# fixed=ms.get_fixed() projects the covariates out of BOTH y and the genotype in the
# V^{-1} metric (no separate get_y adjustment needed). estimate_sigma=True jointly
# re-estimates the variance components and refreshes the projection each IBSS sweep.
# Omit fixed= (default) to adjust only y (backward-compatible).
result = ms.fit(G, y, L=10, estimate_sigma=True, fixed=ms.get_fixed())

# 6) Credible sets by SNP name (result["snp_ids"] mirrors MMSuSiEDense)
cs_named = [[result["snp_ids"][i] for i in cs] for cs in result["cs"]]
print("credible sets:", cs_named)   # [['rs1487590'], ['rs1462069']]

# 7) Export result tables
ms.out(result, out_file="output/test_mmsusie_sp")
```

`MMSuSiESp.fit` / `out` mirror `MMSuSiEDense` so both workflows share the same
fine-mapping and export calls (`mmsusie` / `out_mmsusie` remain as aliases).

## Covariate (fixed-effect) handling

Fixed effects — intercept, numeric covariates, one-hot categoricals, and (for GxE)
environment main effects — are removed by projection in the V^{-1} (GLS) metric,
not carried as columns in the SuSiE model:

- **Variance components** are estimated by REML, which projects the fixed effects
  out via `P = V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹` (the `log|X'V⁻¹X|` term).
- **Fine-mapping** removes them by passing `fixed=` to `fit` (as in both examples):
  the covariates are projected out of **both** `y` and the genotype in the V⁻¹ metric
  — full Frisch–Waugh–Lovell — and re-projected whenever `estimate_sigma` updates V.
  Build the matrix with `MMSuSiESp.get_fixed()`, or (dense) pass the same `xmat` used
  for variance-component estimation. Omit `fixed=` to adjust only `y`
  (`get_y(adjust=True)` / `process_y`), a lighter approximation.

Full FWL matters when covariates correlate with the region's genotypes (ancestry
PCs, a PRS, other SNPs) or in small samples; the correction on each variant scales
with its R² to the covariates. Otherwise the default (adjust `y` only) is a close
approximation.

**Two variance-component modes (REML vs profile-ML).** The standalone `WeightEMAI` /
`WeightEMAISp` estimators are **REML** — they carry the restricted-likelihood
`log|X'V⁻¹X|` term. The in-loop `fit(estimate_sigma=True)` refit is instead a
**profile-ML / mixed-model (generalized-EM) update**: each IBSS sweep maximizes the
profile likelihood at the current SuSiE fit under the current projection, then
re-projects once V changes at the next sweep. The two agree closely but are not
identical — use the REML estimator up front to obtain the variance components, and
`estimate_sigma=True` to let them drift jointly with the fine-mapping.

## Input Data Notes

- PLINK genotype files must share one prefix: `<prefix>.bed/.bim/.fam`.
- Phenotype table must include one row per individual and an IID column (default: first column).
- **Dense GRM** (`MMSuSiEDense`): built with `agmat()`, produces `.grm.id` and `.grm.matrix`.
- **Sparse GRM** (`MMSuSiESp`): built with `fastgxe`; `read_sp_grm` reads two files under the same prefix:
  - `.grm.group` — sample IDs + relatedness groups (`fastgxe --process-grm --group`)
  - `.grm.index_triplet` — lower-triangle GRM triplets (`fastgxe --process-grm --reformat --sparse`)

For SNP selection in `get_genotype`, use exactly one mode:

- explicit list: `sid_lst=["rs1", "rs2", ...]`
- inclusive BIM-order range: `start="rsA", end="rsB"`

## Main Output Files

- `.pip.txt`: posterior inclusion probabilities by SNP
- `.alpha.txt`: posterior assignment probabilities for each effect component
- `.mu.txt`: posterior mean effects
- `.cs.txt`: credible sets

## License

GPL-3.0. See `LICENSE.md`.

---
layout: page
title: Tutorial
description:
order: 3
---

This tutorial runs both mmsusie workflows on the bundled `example/` data — real
PLINK genotypes (427 individuals × 256,868 SNPs) with a simulated phenotype in which
**two independent causal SNPs** (`rs1487590`, `rs1462069`) were planted in the region
`rs2165666 … rs4863332`. A correct fine-mapping run therefore returns **two credible
sets**, one per signal.

Pick a workflow by the GRM you have:

| Class | GRM | Best for | Variance components |
| --- | --- | --- | --- |
| `MMSuSiEDense` | dense `n × n` | small, densely-related samples | `WeightEMAI` |
| `MMSuSiESp` | sparse block-diagonal | large samples, sparse relatedness, GxE | `WeightEMAISp` |

Both end in the same **IBSS** fit and write `PIP`, credible sets, `alpha` and `mu`.

## 1. Dense GRM (`MMSuSiEDense`)

The dense workflow builds the GRM internally with `agmat()`, so it needs no external
tool. Run from the `example/` directory.

```python
import os, logging
logging.basicConfig(level=logging.INFO)

from mmsusie import agmat, MMSuSiEDense, WeightEMAI, prepare_varcom_inputs

os.chdir("example")
os.makedirs("output", exist_ok=True)

# 1) Build the GRM from test.bed/.bim/.fam
agmat("test", "output/test_grm")

# 2) Align phenotype / covariates with the GRM
inputs = prepare_varcom_inputs(
    data_file="data.txt",
    trait_col="pheno",
    grm_prefix="output/test_grm",
    covariate_cols=["cov1", "cov2", "cov3"],
)

# 3) Estimate variance components (REML)
var_com = WeightEMAI().fit(
    y=inputs["y"], xmat=inputs["xmat"], gmat_lst=[inputs["gmat"]]
)

# 4) Fine-map the region with full FWL covariate adjustment
ms = MMSuSiEDense()
ms.cal_Vi(inputs["gmat"], var_com)
G = ms.get_genotype("test", iid_lst=inputs["used_iids"],
                    start="rs2165666", end="rs4863332")
result = ms.fit(G, inputs["y"].flatten(), L=10,
                estimate_sigma=True, fixed=inputs["xmat"])

# 5) Export result tables
ms.out(result, out_file="output/test_mmsusie")
```

`fixed=inputs["xmat"]` turns on **full Frisch–Waugh–Lovell** adjustment: the
covariates are projected out of *both* `y` and `G` in the `V^{-1}` metric, so no
separate `process_y` step is needed. `estimate_sigma=True` re-estimates the variance
components each IBSS sweep and refreshes the projection.

The result has two credible sets, one for each planted causal:

```
credible sets: [['rs1487590'], ['rs1462069']]   # both PIP ≈ 1
```

### Cross-check with susieR

With `V = I`, `MMSuSiEDense` reduces to standard SuSiE and reproduces
[`susieR`](https://github.com/stephenslab/susieR) to machine precision. Export the
covariate-projected `y` and `G` and run `susie()` on the same data:

```python
import pandas as pd
y_adj = ms._gls_residualize(inputs["y"].flatten(), inputs["xmat"])
G_adj = ms._gls_residualize(G, inputs["xmat"])
pd.concat([pd.DataFrame({"IID": inputs["used_iids"], "y_adj": y_adj}),
           pd.DataFrame(G_adj, columns=result["snp_ids"])], axis=1
          ).to_csv("output/test_mmsusie_data.txt", sep="\t", index=False)
```

```r
library(susieR)
dat <- read.table("output/test_mmsusie_data.txt", header = TRUE, sep = "\t")
fit <- susie(as.matrix(dat[, -(1:2)]), dat$y_adj, L = 10)
lapply(susie_get_cs(fit)$cs, function(i) colnames(dat[, -(1:2)])[i])
#> two credible sets — rs1487590 and rs1462069, matching mmsusie
```

## 2. Sparse Block-diagonal GRM (`MMSuSiESp`)

For a large sample with sparse relatedness, first build the sparse GRM with
[`fastgxe`](https://github.com/chaoning/fastGxE) (run from `example/`):

```bash
fastgxe --make-grm --bfile test --out ./output/test
fastgxe --process-grm --group --grm ./output/test --cut-value 0.05
fastgxe --process-grm --reformat --sparse --grm ./output/test --out-fmt 1 --out ./output/test
```

These write `output/test.grm.group` and `output/test.grm.index_triplet`, the two
files `read_sp_grm` reads. **Variance components are estimated natively** by
`WeightEMAISp` — `fastgxe --test-main` is not needed.

```python
import os, logging
logging.basicConfig(level=logging.INFO)

from mmsusie import MMSuSiESp, WeightEMAISp

os.chdir("example")
os.makedirs("output", exist_ok=True)

# 1) Phenotype + sparse GRM
ms = MMSuSiESp()
ms.read_data("data.txt", trait="pheno", covariate_cols=["cov1", "cov2", "cov3"])
ms.read_sp_grm("output/test")

# 2) Variance components (native sparse REML) -> sparse V^{-1}
varcom = WeightEMAISp().fit(ms.get_y(adjust=False), ms.get_fixed(),
                           ms.grm_blocks, n_varcom=2)
ms.cal_spVi(varcom)   # [sigma_g2, sigma_e2]

# 3) Fine-map the region with full FWL
y = ms.get_y(adjust=False)
G = ms.get_genotype("test", start="rs2165666", end="rs4863332")
result = ms.fit(G, y, L=10, estimate_sigma=True, fixed=ms.get_fixed())

# 4) Credible sets by SNP name — result["snp_ids"] mirrors MMSuSiEDense
cs_named = [[result["snp_ids"][i] for i in cs] for cs in result["cs"]]
print("credible sets:", cs_named)   # [['rs1487590'], ['rs1462069']]

# 5) Export result tables
ms.out(result, out_file="output/test_mmsusie_sp")
```

`MMSuSiESp.fit` / `out` mirror `MMSuSiEDense`, so both workflows share the same
fine-mapping and export calls (`mmsusie` / `out_mmsusie` stay as aliases).

Like the dense run, this returns `[['rs1487590'], ['rs1462069']]`. On the example
data the native `WeightEMAISp` estimate matches `fastgxe`'s REML to ~4 decimals.

### GxE variance components

For gene-by-environment fine-mapping, load the interacting environment(s) with
`get_env_int()` and estimate **3 or 4** variance components
(`[σ²_g, σ²_gxe, σ²_e]` or `[σ²_g, σ²_gxe, σ²_gxe2_E, σ²_e]`):

```python
E = ms.get_env_int(scale=True)                 # (n, K) environment matrix
varcom = WeightEMAISp().fit(ms.get_y(adjust=False), ms.get_fixed(),
                           ms.grm_blocks, env_int_arr2=E, n_varcom=3)
ms.cal_spVi(varcom)
```

## 3. Covariate (fixed-effect) handling

Fixed effects — intercept, numeric covariates, one-hot categoricals, and (for GxE)
environment main effects — are removed by **projection in the `V^{-1}` (GLS)
metric**, not carried as columns in the SuSiE model:

- **Variance components** are estimated by REML, which projects the fixed effects out
  via `P = V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹`.
- **Fine-mapping** with `fixed=` (as above) projects the covariates out of **both** the
  phenotype and the genotype (full FWL), refreshed whenever `estimate_sigma` updates
  `V`. Build the matrix with `MMSuSiESp.get_fixed()`, or (dense) pass the same `xmat`
  used for variance-component estimation. Omit `fixed=` to adjust only `y`.

Full FWL matters when covariates correlate with the region's genotypes (ancestry PCs,
a PRS, other SNPs) or in small samples; the correction on each variant scales with its
R² to the covariates. Otherwise adjusting `y` only is a close approximation.

## 4. Output files

`out` (both `MMSuSiEDense` and `MMSuSiESp`) writes:

| File | Contents |
| --- | --- |
| `<prefix>.pip.txt` | posterior inclusion probability per SNP |
| `<prefix>.alpha.txt` | posterior assignment probabilities per effect component |
| `<prefix>.mu.txt` | posterior mean effects |
| `<prefix>.cs.txt` | credible sets (one per line, SNP names) |

## 5. Input data notes

- PLINK genotype files must share one prefix: `<prefix>.bed/.bim/.fam`.
- The phenotype table has one row per individual and an IID column (default: first).
- **Dense GRM** (`agmat`) produces `.grm.id` and `.grm.matrix`.
- **Sparse GRM** (`fastgxe`) — `read_sp_grm` reads `.grm.group` (IDs + relatedness
  groups) and `.grm.index_triplet` (GRM values).
- Select the fine-mapping region in `get_genotype` with exactly one mode: an explicit
  `sid_lst=["rs1", ...]` or an inclusive BIM-order range `start="rsA", end="rsB"`.

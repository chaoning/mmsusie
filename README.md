# MMSuSiE

MMSuSiE is a Python package for mixed-model SuSiE fine-mapping.
It provides an end-to-end workflow for:

- building additive genetic relationship matrices (GRM) from PLINK files,
- estimating variance components with weighted EM-AI,
- running MMSuSiE fine-mapping with GRM-adjusted covariance.

## Project Layout

```
mmsusie/
├── mmsusie_dense.py   # Dense-GRM workflow (MMSuSiEDense)
├── mmsusie_sp.py      # Sparse block-diagonal GRM + MR-SuSiE workflow (MMSuSiESp)
├── utils.py           # Statistical utility functions
├── varcom.py          # Variance component estimation (WeightEMAI)
├── gmatrix.py         # GRM construction (agmat)
└── simu.py            # Phenotype simulation
```

Two classes are exported at the package level:

| Class | File | GRM type | Key methods |
|---|---|---|---|
| `MMSuSiEDense` | `mmsusie_dense.py` | Dense (`cal_Vi`) | `fit`, `out` |
| `MMSuSiESp` | `mmsusie_sp.py` | Sparse block-diagonal (`cal_spVi`) | `mmsusie`, `mrsusie`, `out_mmsusie` |

## Requirements

- Python >= 3.8
- numpy >= 1.22
- pandas >= 1.3
- scipy >= 1.7
- pysnptools >= 0.5
- tqdm >= 4.60
- joblib >= 1.2

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

# 4) Run MMSuSiEDense
ms = MMSuSiEDense()
ms.cal_Vi(inputs["gmat"], var_com)
y_adj = ms.process_y(inputs["y"], inputs["xmat"], adjust=True)
G = ms.get_genotype(
    "test",
    iid_lst=inputs["used_iids"],
    start="rs11132426",
    end="rs7694910",
)
# Pass estimate_sigma=True to jointly re-estimate variance components
# during IBSS iterations (requires cal_Vi() to have been called first).
result = ms.fit(G, y_adj, L=10, estimate_sigma=True)

# 5) Export y_adj and G to text (for comparison with susieR)
import pandas as pd
df_out = pd.DataFrame({"IID": inputs["used_iids"], "y_adj": y_adj.flatten()})
df_G = pd.DataFrame(G, columns=result["snp_ids"])
pd.concat([df_out, df_G], axis=1).to_csv("output/test_mmsusie_data.txt", sep="\t", index=False)

# 6) Export result tables
ms.out(result, out_file="output/test_mmsusie")
```

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

Before running Python, use `fastgxe` ([download](https://github.com/chaoning/fastGxE)) to build the sparse GRM and estimate
variance components (run from the `example/` directory):

```bash
# 1) Build GRM from PLINK files
fastgxe --make-grm --bfile test --out ./output/test

# 2) Compute relatedness groups (threshold 0.05)
fastgxe --process-grm --group --grm ./output/test --cut-value 0.05

# 3) Reformat GRM to sparse index-triplet format
#    --out must share the same prefix as --grm so that read_sp_grm()
#    can find both .grm.group and .grm.index_triplet under one prefix.
fastgxe --process-grm --reformat --sparse --grm ./output/test --out-fmt 1 --out ./output/test

# 4) Estimate variance components (sigma_g2, sigma_e2)
#    Adding --bfile also runs a full GWAS; omit it if only variance
#    components are needed.
fastgxe --test-main --grm ./output/test --data data.txt \
        --trait pheno --covar cov1 cov2 cov3 --out ./output/test_main
```

These commands produce the files `MMSuSiESp` reads:

| File | Used by |
|---|---|
| `output/test.grm.id` | `read_sp_grm` |
| `output/test.grm.group` | `read_sp_grm` |
| `output/test.grm.index_triplet` | `read_sp_grm` |
| `output/test_main.var` | `cal_spVi` (variance components) |

```python
import os, logging
import numpy as np
logging.basicConfig(level=logging.INFO)

from mmsusie import MMSuSiESp

os.chdir("example")
os.makedirs("output", exist_ok=True)

# 1) Load phenotype and sparse GRM
ms = MMSuSiESp()
ms.read_data("data.txt", trait="pheno", covariate_cols=["cov1", "cov2", "cov3"])
ms.read_sp_grm("output/test")

# 2) Load variance components and build sparse V^{-1}
varcom = np.loadtxt("output/test_main.var")  # [sigma_g2, sigma_e2]
ms.cal_spVi(varcom)

# 3) Prepare y using GLS (regress out covariates with V^{-1} weighting)
y = ms.get_y(adjust=True)

# 4) Load genotype for region of interest
G = ms.get_genotype("test", start="rs11132426", end="rs7694910")

# 5) Run MMSuSiE
# Pass estimate_sigma=True to jointly re-estimate variance components
# during IBSS iterations (recommended when varcom may be imprecise).
result = ms.mmsusie(G, y, L=10, estimate_sigma=True)

# 6) Export result tables
ms.out_mmsusie(result, out_file="output/test_mmsusie_sp")
```

## Input Data Notes

- PLINK genotype files must share one prefix: `<prefix>.bed/.bim/.fam`.
- Phenotype table must include one row per individual and an IID column (default: first column).
- **Dense GRM** (`MMSuSiEDense`): built with `agmat()`, produces `.grm.id` and `.grm.matrix`.
- **Sparse GRM** (`MMSuSiESp`): built with `fastgxe`, requires three files under the same prefix:
  - `.grm.id` — sample IDs (`fastgxe --make-grm`)
  - `.grm.group` — relatedness groups (`fastgxe --process-grm --group`)
  - `.grm.index_triplet` — lower-triangle triplets (`fastgxe --process-grm --reformat --sparse`)

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

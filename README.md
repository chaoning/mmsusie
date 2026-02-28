## Installation

```bash
git clone https://github.com/chaoning/mmsusie.git
cd mmsusie
pip install .


"""
MMSuSiE main workflow usage example.

Run:
    python test/test_mmsusie_main.py

This script demonstrates an end-to-end pipeline:
1) Build GRM from PLINK bed/bim/fam.
2) Prepare y/X/GRM-aligned IDs for variance component estimation.
3) Estimate variance components by EM-AI.
4) Run MMSuSiE fine-mapping on a selected SNP region.
5) Export results with SNP IDs.

Required files (under example/):
- test.bed, test.bim, test.fam
- data.txt (contains IID + covariates + phenotype)
"""

import os

from mmsusie.gmatrix import agmat
from mmsusie.mmsusie_main import MMSuSiE
from mmsusie.varcom import WeightEMAI, prepare_varcom_inputs

# Resolve directories from script location, so execution does not depend on current cwd.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
example_dir = os.path.join(repo_root, "example")
output_dir = os.path.join(example_dir, "output")
os.makedirs(output_dir, exist_ok=True)
os.chdir(example_dir)

# ----------------------------
# Step 1: Build GRM
# ----------------------------
# bed_file is PLINK prefix -> reads "test.bed/.bim/.fam".
bed_file = "test"
grm_out_prefix = "./output/test_grm"
agmat(bed_file, grm_out_prefix)

# ----------------------------
# Step 2: Prepare varcom inputs
# ----------------------------
# data.txt should include:
# - IID column (used for alignment with GRM IDs)
# - covariates: cov1, cov2, cov3
# - phenotype: pheno
pheno_file = "data.txt"
trait_col = "pheno"
covariate_cols = ["cov1", "cov2", "cov3"]
categorical_cols = None
inputs = prepare_varcom_inputs(
    data_file=pheno_file,
    trait_col=trait_col,
    grm_prefix=grm_out_prefix,
    covariate_cols=covariate_cols,
    categorical_cols=categorical_cols,
)
print("Prepared inputs:", inputs.keys())

# ----------------------------
# Step 3: Estimate variance components
# ----------------------------
# gmat_lst can contain multiple random-effect matrices.
# Here we only use one GRM + one residual component.
varcom_estimator = WeightEMAI()
var_com = varcom_estimator.fit_vmat(
    y=inputs["y"],
    xmat=inputs["xmat"],
    gmat_lst=[inputs["gmat"]],
)
print("Estimated variance components:", var_com)

# ----------------------------
# Step 4: Run MMSuSiE
# ----------------------------
ms = MMSuSiE()
y = inputs["y"]
xmat = inputs["xmat"]
gmat = inputs["gmat"]

# Build V^{-1} and log|V| from estimated variance components.
ms.cal_Vi(gmat, var_com)

# Regress out fixed effects from phenotype.
y_adj = ms.process_y(y, xmat, adjust=True)

# Select genotype matrix for aligned individuals.
# Two supported selector modes:
# - sid_lst=[...]
# - start="rsA", end="rsB" (inclusive by BIM order)
used_iids = inputs["used_iids"]
start = "rs11132426"
end = "rs7694910"
g = ms.get_genotype(bed_file, iid_lst=used_iids, start=start, end=end)

# Fit model; result includes PIP/CS/ELBO and SNP IDs.
result = ms.fit(g, y_adj)

# ----------------------------
# Step 5: Export outputs
# ----------------------------
# Output files:
# - test_mmsusie.pip.txt   (SNP, pip)
# - test_mmsusie.alpha.txt (alpha matrix with SNP-ID columns)
# - test_mmsusie.mu.txt    (mu matrix with SNP-ID columns)
# - test_mmsusie.cs.txt    (credible sets in SNP IDs)
ms.out(result, out_file="./output/test_mmsusie")



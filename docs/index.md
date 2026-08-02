---
layout: full
homepage: true
disable_anchors: true
description: Mixed-model SuSiE fine-mapping with GRM-adjusted covariance
---
## mmsusie Overview

**mmsusie** brings **SuSiE** (Sum of Single Effects) fine-mapping into the
**mixed-model** setting: it fits

```
y ~ N( Σ_l G (α_l ∘ μ_l) , V ),   V = σ²_g K + σ²_e I,
```

a sum of `L` single-effect regressions on the genotype `G`, with the residual
covariance `V` encoding a **genetic relationship matrix (GRM)** `K`. Working in the
`V^{-1}` (GLS) metric lets the fine-mapping account for relatedness and population
structure that would otherwise inflate signals. From the fitted model mmsusie
returns per-variant **posterior inclusion probabilities (PIP)** and purity-filtered
**credible sets** — the same interpretable output as standard SuSiE, but corrected
for the sample's genetic covariance.

Two workflows share one **IBSS** (iterative Bayesian stepwise selection) engine:

- **`MMSuSiEDense`** — a **dense** GRM. Best for small, densely-related samples
  (families, livestock, mouse HS panels); the `n × n` covariance is formed directly.
  With `V = I` it reproduces [`susieR`](https://github.com/stephenslab/susieR) to
  machine precision.
- **`MMSuSiESp`** — a **sparse block-diagonal** GRM. Built for large biobank-scale
  samples where most individuals are unrelated with a few small related clusters;
  the covariance is inverted block by block, and it additionally supports
  **gene-by-environment (GxE)** variance components.

mmsusie also provides the pieces around the core:

- **GRM construction** (`agmat`, VanRaden / GCTA) from PLINK files.
- **Variance-component estimation by weighted EM-AI REML** — `WeightEMAI` (dense) and
  `WeightEMAISp` (sparse block-diagonal, 1–4 components incl. GxE), so the whole
  pipeline is self-contained.
- **Full Frisch–Waugh–Lovell covariate adjustment** — covariates are projected out
  of **both** the phenotype and the genotype in the `V^{-1}` metric, refreshed as the
  variance components are re-estimated.

mmsusie is pure Python (numpy / pandas / scipy / pysnptools).

## User's Guide: [Installation](./documentation/02_installation.html) · [Tutorial](./documentation/03_Tutorial.html)

## Citation
Chao Ning. *mmsusie: mixed-model SuSiE fine-mapping* (in preparation). The
single-effect regression follows SuSiE (Wang, Sarkar, Carbonetto & Stephens,
*JRSS-B* 2020, [susieR](https://github.com/stephenslab/susieR)); sparse block-diagonal
GRM handling and variance-component estimation follow
[fastGxE](https://github.com/chaoning/fastGxE).

## Contact
For questions, open an issue on [GitHub](https://github.com/chaoning/mmsusie/issues)
or email me at ningchao91@gmail.com

For other tools, see [chaoning.github.io/software.html](https://chaoning.github.io/software.html).

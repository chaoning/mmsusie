# MMSuSiE Example Data

This directory holds a small worked example used by the code snippets in the
top-level [`README.md`](../README.md).

## Files

| File | Description |
|---|---|
| `test.bed` / `test.bim` / `test.fam` | PLINK genotypes — **427 individuals × 256,868 SNPs** (real genotype panel). |
| `data.txt` | Simulated phenotype table: `IID  cov1  cov2  cov3  pheno`. |
| `data.causal_snps.txt` | The **2 planted causal SNPs**, their effect sizes, and the fine-mapping region. |
| `output/` | Results written by the example scripts (git-ignored). |

## How the phenotype was simulated

`data.txt` was generated from `test.bed` with `mmsusie/simu.py`
(`simulate_finemap_example`), which **plants two causal SNPs in a chosen region**
so a fine-mapping run recovers exactly them:

```python
from mmsusie.simu import simulate_finemap_example
simulate_finemap_example(
    "test", "data",
    causal_snps=["rs1487590", "rs1462069"],   # two independent (low-LD) causals
    causal_effects=[0.5, 0.45],               # effect sizes, in phenotype SD units
    region=("rs2165666", "rs4863332"),        # 150-SNP fine-mapping window
    seed=42,
)
```

The phenotype is `y = focal signals + polygenic background + covariates + noise`:

| Component | How it is generated |
|---|---|
| Focal signals | `0.5·Z(rs1487590) + 0.45·Z(rs1462069)` — the two planted causal SNPs (`Z` = standardized genotype) |
| Polygenic background | 200 random causal SNPs **outside** the region, scaled to SD `0.35` |
| Covariates | `cov1, cov2, cov3 ~ N(0, 1)` with small random effects (SD `0.15`) |
| Noise | `ε ~ N(0, 1)` |

Outputs:
- `data.txt` — `IID`, the three covariates, and the phenotype `pheno`.
- `data.causal_snps.txt` — `snp_id`, `beta`, and the `region_start` / `region_end`.

The covariates are independent of the genotypes; the region is excluded from the
polygenic background, so it contains **only the two planted signals**.

## Why the fine-mapping region works (two signals)

The two causal SNPs `rs1487590` and `rs1462069` sit in the region
`rs2165666 … rs4863332` (150 SNPs) and are in **low mutual LD** (r² ≈ 0), so they
are two *independent* signals. Fine-mapping therefore returns **two credible
sets** — one for each causal — with each true causal SNP correctly identified
(PIP ≈ 1). This showcases SuSiE's ability to resolve multiple signals in a locus,
which single-SNP tests cannot.

## Reproducing

Running the snippet above regenerates `data.txt` and `data.causal_snps.txt` (the
simulation is fully seeded). To run the fine-mapping workflows on this data, see
the **Dense** and **Sparse** examples in the top-level [`README.md`](../README.md).

---
layout: page
title: Installation
description: ~
order: 2
---

`mmsusie` is a pure-Python package. Install it from source with `pip`; the only
requirements are the standard scientific-Python stack plus
[`pysnptools`](https://github.com/fastlmm/PySnpTools) for reading PLINK files.

## 1. Requirements

- Python ≥ 3.8
- numpy ≥ 1.22
- pandas ≥ 1.3
- scipy ≥ 1.7
- pysnptools ≥ 0.5
- tqdm ≥ 4.60

All of these are declared in `setup.py` / `requirements.txt` and are pulled in
automatically by `pip`.

## 2. Install from source

```bash
git clone https://github.com/chaoning/mmsusie.git
cd mmsusie
pip install .
```

For an editable (development) install, so that source edits take effect without
reinstalling:

```bash
pip install -e .
```

## 3. Quick validation

Confirm the package imports and the public API is available:

```python
python -c "from mmsusie import MMSuSiEDense, MMSuSiESp, WeightEMAI, WeightEMAISp, agmat, prepare_varcom_inputs; print('mmsusie OK')"
```

You should see `mmsusie OK`.

## 4. Optional external tools

- **[fastGxE](https://github.com/chaoning/fastGxE)** — used only to *build* the
  sparse block-diagonal GRM files (`.grm.group`, `.grm.index_triplet`) that
  `MMSuSiESp.read_sp_grm` reads. Variance components are then estimated **natively**
  by `WeightEMAISp`, so `fastgxe --test-main` is not required. The dense workflow
  (`MMSuSiEDense`) builds its GRM internally with `agmat()` and needs no external tool.
- **[susieR](https://github.com/stephenslab/susieR)** (R) — only if you want to
  cross-check results; with `V = I`, `MMSuSiEDense` reproduces `susieR` to machine
  precision (see the [Tutorial](./03_Tutorial.html)).

## 5. Troubleshooting

- **`pysnptools` fails to read the `.bed` file**: make sure the three PLINK files
  share one prefix (`<prefix>.bed/.bim/.fam`) and that the `.fam` IIDs match the
  phenotype table.
- **A stale installed copy shadows your edits**: if you edited the source but see old
  behaviour, you likely have a non-editable install in `site-packages`. Reinstall with
  `pip install -e .`, or run from the repository root so the local `mmsusie/` package
  is imported first.

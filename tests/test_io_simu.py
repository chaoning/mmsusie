"""Genotype I/O and the simulation helper (example-data integration tests)."""
import os

import numpy as np
import pandas as pd
import pytest

from mmsusie.io import read_genotype_matrix
from mmsusie.simu import simulate_finemap_example
from conftest import REGION, EXAMPLE_BED


def _example_iids():
    fam = pd.read_csv(EXAMPLE_BED + ".fam", sep=r"\s+", header=None, dtype={1: str})
    return fam.iloc[:, 1].astype(str).tolist()


@pytest.mark.slow
def test_get_genotype_range_matches_list(example_dir):
    """Selecting a region by (start, end) matches selecting the same SNPs by id list."""
    iids = _example_iids()
    by_range, ids = read_genotype_matrix(EXAMPLE_BED, iids, start=REGION[0], end=REGION[1])
    by_list, ids2 = read_genotype_matrix(EXAMPLE_BED, iids, sid_lst=ids)
    assert ids == ids2
    np.testing.assert_allclose(by_range, by_list)


@pytest.mark.slow
def test_get_genotype_is_finite_and_standardized(example_dir):
    """No NaN survives mean-imputation, and scaled columns are ~mean 0 / unit sd."""
    iids = _example_iids()
    mat, _ = read_genotype_matrix(EXAMPLE_BED, iids, start=REGION[0], end=REGION[1])
    assert np.isfinite(mat).all()
    assert np.abs(mat.mean(axis=0)).max() < 1e-8
    sd = mat.std(axis=0)
    # non-monomorphic columns are unit-sd; monomorphic ones stay 0 (std set to 1)
    assert np.all((np.abs(sd - 1) < 1e-8) | (sd == 0))


@pytest.mark.slow
def test_simu_rejects_causal_outside_region(example_dir, tmp_path):
    """#8: a causal SNP outside the region is rejected."""
    bim = pd.read_csv(EXAMPLE_BED + ".bim", sep=r"\s+", header=None, dtype={1: str})
    outside = bim.iloc[0, 1]  # first SNP, well before the region
    with pytest.raises(ValueError, match="outside the region"):
        simulate_finemap_example(EXAMPLE_BED, str(tmp_path / "bad"),
                                 causal_snps=[outside, "rs1487590"],
                                 causal_effects=[0.5, 0.4], region=REGION,
                                 n_bg_causal=20, seed=1)


@pytest.mark.slow
def test_simu_is_reproducible(example_dir, tmp_path):
    """The same seed reproduces the same phenotype."""
    kw = dict(causal_snps=["rs1487590", "rs1462069"], causal_effects=[0.5, 0.45],
              region=REGION, n_bg_causal=20, seed=1)
    a = simulate_finemap_example(EXAMPLE_BED, str(tmp_path / "a"), **kw)
    b = simulate_finemap_example(EXAMPLE_BED, str(tmp_path / "b"), **kw)
    np.testing.assert_allclose(a["pheno"].values, b["pheno"].values)

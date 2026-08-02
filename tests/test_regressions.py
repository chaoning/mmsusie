"""One regression test per bug fixed during the code review.

Each test fails on the pre-fix code and passes on the current code.
"""
import numpy as np
import pandas as pd
import pytest

from mmsusie import MMSuSiEDense, MMSuSiESp
from mmsusie.utils import get_cs_purity
from conftest import make_dense, make_sparse


def test_cal_spvi_no_singleton(synthetic):
    """#3: a sparse GRM with no singleton individuals must not crash cal_spVi."""
    ms = MMSuSiESp()
    ms.iid_used = [f"i{k}" for k in range(synthetic["n"])]
    ms.grm_blocks = [np.array([]), synthetic["K"]]   # empty singleton slot
    ms.env_int_arr2 = None
    ms.cal_spVi(np.array([0.4, 0.6]))
    assert ms.Vi.shape == (synthetic["n"], synthetic["n"])


@pytest.mark.parametrize("cls", [MMSuSiEDense, MMSuSiESp])
def test_iid_col_by_name(cls, tmp_path):
    """#6: iid_col must select the IID column by name even with a preceding column."""
    path = tmp_path / "d.txt"
    # File column order: junk, IID, pheno  -> IID is at index 1.
    pd.DataFrame({"junk": [9, 9, 9], "IID": ["a", "b", "c"], "pheno": [1.0, 2.0, 3.0]}
                 ).to_csv(path, sep=" ", index=False)
    ms = cls()
    ms.read_data(str(path), trait="pheno", iid_col=1)
    assert ms.iid_in_data == ["a", "b", "c"]


def test_result_arrays_row_aligned(synthetic):
    """#10: every per-effect array in the result shares one length, plus kept_effects."""
    res = make_dense(synthetic).fit(synthetic["G"], synthetic["y"], L=10,
                                    estimate_sigma=False)
    lengths = {np.asarray(res[k]).shape[0] for k in ("alpha", "mu", "sigma0", "lbf", "KL")}
    assert len(lengths) == 1
    assert "kept_effects" in res and len(res["kept_effects"]) == np.asarray(res["alpha"]).shape[0]


def test_cal_vi_allows_zero_sigma_g(synthetic):
    """#8: sigma_g2 = 0 is a valid boundary; a negative component is rejected."""
    ms = MMSuSiEDense()
    ms.cal_Vi(synthetic["K"], np.array([0.0, 1.0]))   # must not raise
    with pytest.raises(ValueError):
        MMSuSiEDense().cal_Vi(synthetic["K"], np.array([-1.0, 1.0]))


def test_out_handles_integer_feature_names(synthetic, tmp_path):
    """#4: MMSuSiESp.out must not crash when features are unnamed (integer indices)."""
    ms = make_sparse(synthetic)
    ms.last_snp_ids = None                            # plain-matrix use, no SNP labels
    res = ms.fit(synthetic["G"], synthetic["y"], L=8, estimate_sigma=False)
    out_prefix = str(tmp_path / "plain")
    ms.out(res, out_file=out_prefix)                  # would raise TypeError pre-fix
    assert (tmp_path / "plain.cs.txt").exists()


def test_fit_accepts_column_phenotype_and_validates(synthetic):
    """#7: an (n, 1) phenotype is flattened; a mismatched length is rejected."""
    ms = make_sparse(synthetic)
    res = ms.fit(synthetic["G"], synthetic["y"].reshape(-1, 1), L=5, estimate_sigma=False)
    assert res["pip"].shape[0] == synthetic["p"]
    with pytest.raises(ValueError):
        make_sparse(synthetic).fit(synthetic["G"], synthetic["y"][:10], L=5,
                                   estimate_sigma=False)


def test_joint_update_consistency(synthetic):
    """#1: the trailing variance update is gone.

    With maxiter=1 the M-step (guarded by iter > 0) never runs, so the returned
    posterior is computed under the initial V and self.varcom is left untouched,
    matching estimate_sigma=False.
    """
    init = np.array([0.5, 0.5])
    off = make_dense(synthetic, varcom=init.copy()).fit(
        synthetic["G"], synthetic["y"], L=10, maxiter=1, estimate_sigma=False)
    ms = make_dense(synthetic, varcom=init.copy())
    on = ms.fit(synthetic["G"], synthetic["y"], L=10, maxiter=1, estimate_sigma=True)
    np.testing.assert_allclose(off["alpha"], on["alpha"], atol=1e-12)
    np.testing.assert_allclose(ms.varcom, init, atol=1e-12)


def test_purity_uses_correlation(synthetic):
    """#2 (purity semantics): get_cs_purity keeps a highly-correlated CS and drops an
    uncorrelated one, using ordinary genotype correlation."""
    rng = np.random.default_rng(7)
    n = 200
    a = rng.standard_normal(n)
    X = np.column_stack([a, a + 1e-3 * rng.standard_normal(n),   # cols 0,1 ~ perfectly correlated
                         rng.standard_normal(n)])                 # col 2 independent
    kept, _ = get_cs_purity([[0, 1]], np.array([0.95]), X, min_abs_corr=0.5)
    dropped, _ = get_cs_purity([[0, 2]], np.array([0.95]), X, min_abs_corr=0.5)
    assert kept == [[0, 1]]
    assert dropped == []

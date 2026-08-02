"""Core fine-mapping behaviour: dense/sparse equivalence, signal recovery, and an
end-to-end run on the bundled example data."""
import os

import numpy as np
import pytest

from mmsusie import MMSuSiEDense, WeightEMAI, prepare_varcom_inputs
from conftest import make_dense, make_sparse, REGION, PLANTED_CAUSALS


def _pip(res):
    return np.asarray(res["pip"]["pip"].values, dtype=float)


def test_dense_sparse_equivalence(synthetic):
    """With the same V, MMSuSiEDense and MMSuSiESp must give the same fit."""
    dense = make_dense(synthetic).fit(
        synthetic["G"], synthetic["y"], L=10, estimate_sigma=False
    )
    sparse = make_sparse(synthetic).fit(
        synthetic["G"], synthetic["y"], L=10, estimate_sigma=False
    )
    # PIP is length p regardless of effect pruning, so it is the robust comparison.
    np.testing.assert_allclose(_pip(dense), _pip(sparse), atol=1e-7)
    assert [sorted(c) for c in dense["cs"]] == [sorted(c) for c in sparse["cs"]]


def test_signal_recovery(synthetic):
    """The two planted causals should be recovered (high PIP, one per credible set)."""
    res = make_dense(synthetic).fit(
        synthetic["G"], synthetic["y"], L=10, estimate_sigma=False
    )
    pip = _pip(res)
    for j in synthetic["causal"]:
        assert pip[j] > 0.5, f"causal column {j} has low PIP {pip[j]:.3f}"
    recovered = {i for cs in res["cs"] for i in cs}
    assert set(synthetic["causal"]).issubset(recovered)


def test_vi_identity_runs_and_recovers(synthetic):
    """sigma_g2 = 0 gives V = I (standard SuSiE); the fit must still recover signals."""
    res = make_dense(synthetic, varcom=np.array([0.0, 1.0])).fit(
        synthetic["G"], synthetic["y"], L=10, estimate_sigma=False
    )
    pip = _pip(res)
    assert all(pip[j] > 0.5 for j in synthetic["causal"])


@pytest.mark.slow
def test_example_two_causals(example_dir, tmp_path):
    """End-to-end dense workflow on the example data recovers the two planted causals."""
    grm_prefix = str(tmp_path / "grm")
    from mmsusie import agmat

    cwd = os.getcwd()
    os.chdir(example_dir)
    try:
        agmat("test", grm_prefix)
        inputs = prepare_varcom_inputs(
            data_file="data.txt", trait_col="pheno", grm_prefix=grm_prefix,
            covariate_cols=["cov1", "cov2", "cov3"],
        )
        var_com = WeightEMAI().fit(
            y=inputs["y"], xmat=inputs["xmat"], gmat_lst=[inputs["gmat"]]
        )
        ms = MMSuSiEDense()
        ms.cal_Vi(inputs["gmat"], var_com)
        G = ms.get_genotype("test", iid_lst=inputs["used_iids"],
                            start=REGION[0], end=REGION[1])
        res = ms.fit(G, inputs["y"].flatten(), L=10,
                    estimate_sigma=True, fixed=inputs["xmat"])
    finally:
        os.chdir(cwd)

    cs_names = {res["snp_ids"][i] for cs in res["cs"] for i in cs}
    assert cs_names == PLANTED_CAUSALS
    assert len(res["cs"]) == 2

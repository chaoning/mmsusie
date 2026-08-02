"""Variance-component estimation: sparse/dense agreement, analytic-gradient checks
against finite differences, and cal_spVi input validation."""
import numpy as np
import pytest

from mmsusie import WeightEMAI, WeightEMAISp
from mmsusie.mmsusie_dense import _sigma_neg_loglik_and_grad
from mmsusie.mmsusie_sp import _sigma_neg_loglik_and_grad_sp
from conftest import make_sparse


def _valid_ser_state(n, p, L, rng):
    """A valid (alpha, mu, mu2, Xresi) posterior state for the sigma objective."""
    alpha = rng.random((L, p))
    alpha /= alpha.sum(axis=1, keepdims=True)
    mu = rng.standard_normal((L, p)) * 0.1
    mu2 = mu ** 2 + rng.random((L, p)) * 0.05      # 2nd moment >= mean^2
    Xresi = rng.standard_normal(n) * 0.1
    return alpha, mu, mu2, Xresi


def test_sparse_dense_varcom_agree(synthetic):
    """WeightEMAI (dense GRM) and WeightEMAISp (one-block GRM) give the same REML fit."""
    n = synthetic["n"]
    xmat = np.ones((n, 1))
    dense = WeightEMAI().fit(y=synthetic["y"], xmat=xmat, gmat_lst=[synthetic["K"]])
    sparse = WeightEMAISp().fit(synthetic["y"], xmat,
                                [np.array([]), synthetic["K"]], n_varcom=2)
    np.testing.assert_allclose(np.asarray(dense).ravel(), np.asarray(sparse).ravel(),
                               rtol=1e-3, atol=1e-4)


def test_sigma_gradient_fd_dense(synthetic):
    """Analytic gradient of the dense sigma objective matches finite differences."""
    rng = np.random.default_rng(1)
    n, p, L = synthetic["n"], synthetic["p"], 3
    alpha, mu, mu2, Xresi = _valid_ser_state(n, p, L, rng)
    args = (synthetic["K"], synthetic["y"], synthetic["G"], Xresi, alpha, mu, mu2)
    v = np.array([0.5, 0.7])
    f0, g = _sigma_neg_loglik_and_grad(v, *args)
    eps = 1e-6
    for i in range(len(v)):
        vp, vm = v.copy(), v.copy()
        vp[i] += eps; vm[i] -= eps
        fd = (_sigma_neg_loglik_and_grad(vp, *args)[0]
              - _sigma_neg_loglik_and_grad(vm, *args)[0]) / (2 * eps)
        assert abs(fd - g[i]) < 1e-4, f"dense grad[{i}]: analytic {g[i]}, fd {fd}"


def test_sigma_gradient_fd_reml(synthetic):
    """The REML term (fixed= given) keeps analytic gradients matching finite
    differences, for both the dense and sparse sigma objectives."""
    rng = np.random.default_rng(4)
    n, p, L = synthetic["n"], synthetic["p"], 3
    alpha, mu, mu2, Xresi = _valid_ser_state(n, p, L, rng)
    F = np.column_stack([np.ones(n), rng.standard_normal((n, 4))])   # fixed design
    K, y, G = synthetic["K"], synthetic["y"], synthetic["G"]

    def fd_ok(func, args, v, eps=1e-6):
        f0, g = func(v, *args)
        for i in range(len(v)):
            vp, vm = v.copy(), v.copy()
            vp[i] += eps; vm[i] -= eps
            fd = (func(vp, *args)[0] - func(vm, *args)[0]) / (2 * eps)
            assert abs(fd - g[i]) < 1e-4, f"REML grad[{i}]: analytic {g[i]}, fd {fd}"

    fd_ok(_sigma_neg_loglik_and_grad, (K, y, G, Xresi, alpha, mu, mu2, F), np.array([0.5, 0.7]))
    E = rng.standard_normal((n, 2))
    fd_ok(_sigma_neg_loglik_and_grad_sp,
          ([np.array([]), K], y, G, Xresi, alpha, mu, mu2, E, F), np.array([0.4, 0.3, 0.5]))


@pytest.mark.parametrize("nvc", [1, 2])
def test_sigma_gradient_fd_sparse(synthetic, nvc):
    """Analytic gradient of the sparse sigma objective matches finite differences."""
    rng = np.random.default_rng(2)
    n, p, L = synthetic["n"], synthetic["p"], 3
    alpha, mu, mu2, Xresi = _valid_ser_state(n, p, L, rng)
    grm_blocks = [np.array([]), synthetic["K"]]
    args = (grm_blocks, synthetic["y"], synthetic["G"], Xresi, alpha, mu, mu2, None)
    v = np.full(nvc, 0.6)
    f0, g = _sigma_neg_loglik_and_grad_sp(v, *args)
    eps = 1e-6
    for i in range(nvc):
        vp, vm = v.copy(), v.copy()
        vp[i] += eps; vm[i] -= eps
        fd = (_sigma_neg_loglik_and_grad_sp(vp, *args)[0]
              - _sigma_neg_loglik_and_grad_sp(vm, *args)[0]) / (2 * eps)
        assert abs(fd - g[i]) < 1e-4, f"sparse grad[{i}]: analytic {g[i]}, fd {fd}"


@pytest.mark.parametrize("nvc", [1, 2, 3, 4])
def test_weightemaisp_all_component_counts(synthetic, nvc):
    """1-4 variance components all fit and return the right length."""
    n = synthetic["n"]
    xmat = np.ones((n, 1))
    env = np.random.default_rng(3).standard_normal((n, 2)) if nvc >= 3 else None
    out = WeightEMAISp().fit(synthetic["y"], xmat, [np.array([]), synthetic["K"]],
                             env_int_arr2=env, n_varcom=nvc)
    assert len(out) == nvc
    assert np.all(np.asarray(out) > 0)


def test_cal_spvi_rejects_bad_varcom(synthetic):
    """cal_spVi validates its input instead of silently producing NaN (regression #5)."""
    ms = make_sparse(synthetic)  # already built with a valid varcom
    with pytest.raises(ValueError):
        ms.cal_spVi(np.array([-1.0, 0.5]))        # negative component
    with pytest.raises(ValueError):
        ms.cal_spVi(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))  # too many components

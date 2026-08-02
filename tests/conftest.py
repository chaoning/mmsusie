"""Shared fixtures and helpers for the mmsusie test suite.

Most tests run on small *synthetic* data (a random positive-definite GRM plus a
standardized genotype matrix) so they need neither PLINK files nor the external
``fastgxe`` tool. A handful of integration tests use the bundled ``example/`` data
(genotypes are committed; the dense GRM is built in-process with ``agmat``); they
skip automatically if the files are missing.
"""
import os

import numpy as np
import pytest

from mmsusie import MMSuSiEDense, MMSuSiESp

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "example")
EXAMPLE_BED = os.path.join(EXAMPLE_DIR, "test")
EXAMPLE_DATA = os.path.join(EXAMPLE_DIR, "data.txt")
REGION = ("rs2165666", "rs4863332")
PLANTED_CAUSALS = {"rs1487590", "rs1462069"}


def _has_example():
    return all(os.path.exists(EXAMPLE_BED + ext) for ext in (".bed", ".bim", ".fam"))


@pytest.fixture(scope="session")
def example_dir():
    """Path to the bundled example data, or skip if it is not present."""
    if not _has_example():
        pytest.skip("example PLINK data not available")
    return EXAMPLE_DIR


def make_pd_grm(n, rng, m=200):
    """A random positive-definite GRM with unit diagonal (VanRaden-like)."""
    z = rng.standard_normal((n, m))
    k = z @ z.T / m
    d = np.sqrt(np.diag(k))
    return k / np.outer(d, d)


@pytest.fixture
def synthetic():
    """A small, fully in-memory fine-mapping problem.

    Returns a dict with: ``n``, ``p``, ``K`` (n×n PD GRM), ``G`` (n×p standardized
    genotype), ``y`` (n, standardized phenotype with two planted signals),
    ``causal`` (the planted column indices) and ``varcom`` = [sigma_g2, sigma_e2].
    """
    rng = np.random.default_rng(0)
    n, p = 80, 12
    K = make_pd_grm(n, rng)
    G = rng.standard_normal((n, p))
    G = (G - G.mean(0)) / G.std(0)

    causal = [2, 8]
    beta = np.zeros(p)
    for j in causal:
        beta[j] = rng.choice([-1.0, 1.0]) * 1.2

    chol = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    g = chol @ rng.standard_normal(n) * np.sqrt(0.3)
    y = G @ beta + g + rng.standard_normal(n) * np.sqrt(0.4)
    y = (y - y.mean()) / y.std()

    return {
        "n": n, "p": p, "K": K, "G": G, "y": y,
        "causal": causal, "varcom": np.array([0.4, 0.6]),
    }


def make_dense(synthetic, varcom=None):
    """MMSuSiEDense with V^{-1} built from the synthetic GRM."""
    ms = MMSuSiEDense()
    ms.cal_Vi(synthetic["K"], synthetic["varcom"] if varcom is None else varcom)
    return ms


def make_sparse(synthetic, varcom=None):
    """MMSuSiESp holding the synthetic GRM as a single related block (no singletons)."""
    ms = MMSuSiESp()
    ms.iid_used = [f"i{k}" for k in range(synthetic["n"])]
    ms.grm_blocks = [np.array([]), synthetic["K"]]  # [empty singleton slot, one dense block]
    ms.env_int_arr2 = None
    ms.cal_spVi(synthetic["varcom"] if varcom is None else varcom)
    return ms

"""Validate from-scratch LDA/QDA against sklearn on synthetic Gaussian blobs."""

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as SKLDA,
    QuadraticDiscriminantAnalysis as SKQDA,
)
import pytest

from discriminant import (
    DegenerateClassError, accuracy_ci, fit_lda, fit_qda, pooled_eigen, predict,
)

SEED = 0


def make_data():
    rng = np.random.default_rng(SEED)
    mus = np.array([[1.0, 0.5], [1.4, 0.9], [0.3, 0.1]])
    covs = [np.array([[0.05, 0.03], [0.03, 0.04]]),
            np.array([[0.09, 0.05], [0.05, 0.06]]),
            np.array([[0.02, 0.01], [0.01, 0.02]])]
    ns = [200, 180, 160]
    X, y = [], []
    for i, (m, c, nk) in enumerate(zip(mus, covs, ns)):
        X.append(rng.multivariate_normal(m, c, nk))
        y += [i] * nk
    return np.vstack(X), np.array(y)


def test_shapes():
    X, y = make_data()
    p, K = X.shape[1], len(np.unique(y))
    lda, qda = fit_lda(X, y), fit_qda(X, y)
    assert lda["cov"].shape == (p, p)
    assert qda["covs"].shape == (K, p, p)
    assert lda["means"].shape == (K, p)
    assert np.isclose(lda["priors"].sum(), 1.0)


def test_matches_sklearn():
    X, y = make_data()
    Xte = X[::3]
    assert np.array_equal(predict(fit_lda(X, y), Xte), SKLDA().fit(X, y).predict(Xte))
    assert np.array_equal(predict(fit_qda(X, y), Xte), SKQDA().fit(X, y).predict(Xte))


def test_recovers_means():
    X, y = make_data()
    m = fit_qda(X, y)
    truth = np.array([[1.0, 0.5], [1.4, 0.9], [0.3, 0.1]])
    assert np.allclose(m["means"], truth, atol=0.05)


def test_ci_arithmetic():
    yt = np.r_[np.ones(248), np.zeros(22)]
    acc, lo, hi, se = accuracy_ci(yt, np.ones(270))
    assert np.isclose(acc, 0.9185, atol=1e-3)
    assert np.isclose(se, 0.01665, atol=1e-4)


def test_ci_is_wilson():
    """The two Wald failures: zero width at acc = 1, bounds outside [0, 1]."""
    acc, lo, hi, se = accuracy_ci(np.ones(270), np.ones(270))
    assert acc == 1.0 and se == 0.0
    assert hi == 1.0 and 0.98 < lo < 1.0, (lo, hi)          # Wald gave [1, 1]
    acc, lo, hi, se = accuracy_ci(np.ones(10), np.zeros(10))
    assert acc == 0.0 and lo == 0.0 and 0.2 < hi < 0.3, (lo, hi)
    # Closed form at 248/270, z = 1.96: centre (p + z^2/2n)/(1 + z^2/n).
    yt = np.r_[np.ones(248), np.zeros(22)]
    acc, lo, hi, se = accuracy_ci(yt, np.ones(270))
    n, z = 270, 1.96
    centre = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))
    assert np.isclose(lo, centre - half) and np.isclose(hi, centre + half)
    assert lo < acc - z * se and hi < acc + z * se          # shifted toward 0.5 vs Wald


def test_pooled_eigen_weighted():
    """QDA covariances pooled with n_k - 1 equal fit_lda's pooled covariance."""
    X, y = make_data()                       # 200 / 180 / 160 per class
    qda, lda = fit_qda(X, y), fit_lda(X, y)
    w = qda["counts"] - 1
    pooled = np.tensordot(w.astype(float), qda["covs"], axes=1) / w.sum()
    assert np.allclose(pooled, lda["cov"])
    vals_q, _ = pooled_eigen(qda)
    vals_l, _ = pooled_eigen(lda)
    assert np.allclose(vals_q, vals_l)
    # And it is NOT the unweighted mean, which is what used to be here.
    unweighted = np.linalg.eigvalsh(qda["covs"].mean(axis=0))[::-1]
    assert not np.allclose(unweighted, vals_q)


def test_degenerate_guard_is_le():
    """n_k == p must raise: p centred points span only p - 1 dimensions."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(2, 2)), rng.normal(size=(30, 2))])
    y = np.array([0, 0] + [1] * 30)
    with pytest.raises(DegenerateClassError):
        fit_qda(X, y)
    X3 = np.vstack([rng.normal(size=(3, 2)), rng.normal(size=(30, 2))])
    y3 = np.array([0, 0, 0] + [1] * 30)
    fit_qda(X3, y3)                          # n_k = p + 1 is the minimum


if __name__ == "__main__":
    for fn in [test_shapes, test_matches_sklearn, test_recovers_means,
               test_ci_arithmetic, test_ci_is_wilson, test_pooled_eigen_weighted,
               test_degenerate_guard_is_le]:
        fn()
        print("PASS", fn.__name__)
    X, y = make_data()
    vals, _ = pooled_eigen(fit_lda(X, y))
    print("eigenvalues:", vals, "  cond:", vals[0] / vals[-1])
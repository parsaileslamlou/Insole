"""Validate from-scratch LDA/QDA against sklearn on synthetic Gaussian blobs."""

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as SKLDA,
    QuadraticDiscriminantAnalysis as SKQDA,
)
from discriminant import fit_lda, fit_qda, predict, accuracy_ci, pooled_eigen

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


if __name__ == "__main__":
    for fn in [test_shapes, test_matches_sklearn, test_recovers_means, test_ci_arithmetic]:
        fn()
        print("PASS", fn.__name__)
    X, y = make_data()
    vals, _ = pooled_eigen(fit_lda(X, y))
    print("eigenvalues:", vals, "  cond:", vals[0] / vals[-1])
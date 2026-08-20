"""LDA and QDA from scratch (numpy only) + CLT confidence interval on accuracy."""

import warnings

import numpy as np

# RETUNE: eigenvalue floor for the rank check. Mirrors the default tol of
# sklearn's QuadraticDiscriminantAnalysis, which counts rank as the number of
# covariance eigenvalues strictly above tol and REFUSES to fit when that is
# below the feature count. This threshold is absolute, not relative, so it is
# scale-sensitive: the same data z-scored can pass where the raw data fails.
RANK_TOL = 1e-4


class SingularCovarianceError(np.linalg.LinAlgError):
    """A covariance is not positive definite, so its log-determinant is undefined.

    Subclasses LinAlgError so callers already catching that keep working.
    """


class DegenerateClassError(ValueError):
    """A class has too few samples to estimate a covariance from."""


class IllConditionedCovarianceWarning(UserWarning):
    """A covariance has eigenvalues below RANK_TOL. Not fatal here."""


def _rank_deficient_eigenvalues(cov, tol):
    """Eigenvalues at or below tol — the ones sklearn would not count toward rank."""
    vals = np.linalg.eigvalsh(cov)
    return vals[vals <= tol]


def _class_stats(X, y):
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])
    counts = np.array([np.sum(y == c) for c in classes])
    priors = counts / len(y)
    return classes, means, counts, priors


def fit_lda(X, y, reg=0.0):
    classes, means, counts, priors = _class_stats(X, y)
    n, p = X.shape
    S = np.zeros((p, p))
    for i, c in enumerate(classes):
        Xc = X[y == c] - means[i]
        S += Xc.T @ Xc
    S /= (n - len(classes))
    S += reg * np.eye(p)
    return {"kind": "lda", "classes": classes, "means": means,
            "priors": priors, "cov": S}


def fit_qda(X, y, reg=0.0, tol=RANK_TOL):
    classes, means, counts, priors = _class_stats(X, y)
    p = X.shape[1]

    # Fail at fit time, naming the class. Without this, n_k == 1 divides by
    # counts[i] - 1 == 0 and yields a silently non-finite covariance, and
    # n_k < p surfaces much later as a bare "Singular matrix" LinAlgError
    # thrown from inside _log_discriminants at predict time.
    for i, c in enumerate(classes):
        if counts[i] <= 1:
            raise DegenerateClassError(
                f"class {str(c)!r} has {counts[i]} sample(s); QDA needs at least 2 "
                f"to estimate a covariance (divides by n_k - 1)")
        if counts[i] <= p:
            raise DegenerateClassError(
                f"class {str(c)!r} has {counts[i]} samples for {p} features; its "
                f"covariance cannot have full rank. n_k centered points span at "
                f"most n_k - 1 dimensions, so n_k must exceed p, not merely "
                f"reach it. Reduce the feature count, pool with LDA, or pass "
                f"reg > 0")

    covs = []
    for i, c in enumerate(classes):
        Xc = X[y == c] - means[i]
        covs.append((Xc.T @ Xc) / (counts[i] - 1) + reg * np.eye(p))

    # Advisory only: this is exactly the condition on which sklearn's QDA
    # raises LinAlgError at its default tol. We still fit.
    for i, c in enumerate(classes):
        small = _rank_deficient_eigenvalues(covs[i], tol)
        if small.size:
            warnings.warn(
                f"class {str(c)!r}: {small.size} of {p} covariance eigenvalues "
                f"<= tol={tol:g} (smallest {small.min():.3g}); effective rank "
                f"{p - small.size} < {p}. sklearn's QuadraticDiscriminantAnalysis "
                f"would refuse to fit this class at the same tol. Fitting anyway.",
                IllConditionedCovarianceWarning, stacklevel=2)

    return {"kind": "qda", "classes": classes, "means": means,
            "priors": priors, "covs": np.stack(covs)}


def _log_discriminants(model, X):
    K = len(model["classes"])
    out = np.zeros((X.shape[0], K))
    for k in range(K):
        S = model["cov"] if model["kind"] == "lda" else model["covs"][k]
        d = X - model["means"][k]
        # slogdet before solve: on a singular S, solve raises a bare
        # "Singular matrix" LinAlgError that names nothing. Checking the sign
        # first means the caller gets an error that says which class.
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            where = ("pooled covariance" if model["kind"] == "lda"
                     else f"covariance of class {str(model['classes'][k])!r}")
            raise SingularCovarianceError(
                f"{where} is not positive definite (slogdet sign={sign:g}); "
                f"its log-determinant is undefined. Pass reg > 0 or drop "
                f"collinear features.")
        sol = np.linalg.solve(S, d.T).T
        maha = np.sum(d * sol, axis=1)
        out[:, k] = -0.5 * maha - 0.5 * logdet + np.log(model["priors"][k])
    return out


def predict(model, X):
    return model["classes"][np.argmax(_log_discriminants(model, X), axis=1)]


def accuracy_ci(y_true, y_pred, z=1.96):
    correct = np.asarray(y_true) == np.asarray(y_pred)
    n = len(correct)
    acc = correct.mean()
    se = np.sqrt(acc * (1 - acc) / n)
    return acc, acc - z * se, acc + z * se, se


def pooled_eigen(model):
    S = model["cov"] if model["kind"] == "lda" else model["covs"].mean(axis=0)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]
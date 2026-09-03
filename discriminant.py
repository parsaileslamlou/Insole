"""LDA and QDA from scratch (numpy only) + Wilson confidence interval on accuracy.

Persistence: save_model / load_model round-trip a fitted model through JSON so
a deployment script (infer_live.py) can predict without refitting at startup.
The JSON carries whatever metadata dict the caller attaches under "meta";
this module does not interpret it.
"""

import json
import time
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
            "priors": priors, "counts": counts, "cov": S}


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
            "priors": priors, "counts": counts, "covs": np.stack(covs)}


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
    """Accuracy with a Wilson score interval. Returns (acc, lo, hi, se).

    Wilson replaces the Wald interval acc +/- z*sqrt(acc(1-acc)/n) that used
    to be here. Wald has two failures that both showed up in bakeoff.py:
    zero width at acc = 1.0 (se = 0, so the interval claims certainty from
    finite n) and bounds outside [0, 1] near-perfect. Wilson's bounds are
    the roots of |acc - p| = z*sqrt(p(1-p)/n) solved for p, so they stay in
    [0, 1] and have positive width at acc = 0 or 1.

    `se` is still the Wald standard error sqrt(acc(1-acc)/n). It is returned
    because bakeoff.py compares it to a session-level standard error, and
    that comparison is about the independence assumption, not the interval.
    It is NOT half the interval width any more.
    """
    correct = np.asarray(y_true) == np.asarray(y_pred)
    n = len(correct)
    acc = correct.mean()
    se = np.sqrt(acc * (1 - acc) / n)
    z2 = z * z
    centre = (acc + z2 / (2 * n)) / (1 + z2 / n)
    half = (z / (1 + z2 / n)) * np.sqrt(acc * (1 - acc) / n + z2 / (4 * n * n))
    return acc, max(0.0, centre - half), min(1.0, centre + half), se


def pooled_eigen(model):
    """Eigen-decomposition of the pooled within-class covariance.

    For LDA that is model["cov"] as fitted. For QDA the per-class covariances
    are pooled with weights n_k - 1, the same weighting fit_lda uses, so the
    QDA-pooled matrix equals what fit_lda would have produced on the same
    data (up to reg). The unweighted covs.mean(axis=0) that used to be here
    is only equal to that when every class has the same size; at the
    bake-off's 305/365/183 split it was off by a few percent.
    """
    if model["kind"] == "lda":
        S = model["cov"]
    else:
        w = np.asarray(model["counts"], dtype=float) - 1.0
        S = np.tensordot(w, model["covs"], axes=1) / w.sum()
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
MODEL_SCHEMA = 1
_ARRAY_KEYS = ("means", "priors", "counts", "cov", "covs")


def save_model(model, path, meta=None):
    """Write a fitted LDA/QDA model to JSON. `meta` is stored verbatim.

    Class labels are stored as strings. Every array is stored as a nested
    list; load_model turns them back into numpy arrays. Nothing else in the
    model dict is persisted, so a model with extra keys loses them here.
    """
    doc = {
        "schema": MODEL_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": model["kind"],
        "classes": [str(c) for c in model["classes"]],
    }
    for k in _ARRAY_KEYS:
        if k in model:
            doc[k] = np.asarray(model[k]).tolist()
    if meta is not None:
        doc["meta"] = meta
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    return doc


def load_model(path):
    """Read a model written by save_model. Returns the model dict plus "meta"."""
    with open(path, "r") as f:
        doc = json.load(f)
    if doc.get("kind") not in ("lda", "qda"):
        raise ValueError(f"{path}: kind={doc.get('kind')!r} is not lda/qda")
    model = {"kind": doc["kind"], "classes": np.array(doc["classes"])}
    for k in _ARRAY_KEYS:
        if k in doc:
            model[k] = np.array(doc[k], dtype=float)
    if "counts" in model:
        model["counts"] = model["counts"].astype(int)
    if model["kind"] == "lda" and "cov" not in model:
        raise ValueError(f"{path}: lda model has no 'cov'")
    if model["kind"] == "qda" and "covs" not in model:
        raise ValueError(f"{path}: qda model has no 'covs'")
    model["meta"] = doc.get("meta", {})
    return model

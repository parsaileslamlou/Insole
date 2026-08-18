"""LDA and QDA from scratch (numpy only) + CLT confidence interval on accuracy."""

import numpy as np


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


def fit_qda(X, y, reg=0.0):
    classes, means, counts, priors = _class_stats(X, y)
    p = X.shape[1]
    covs = []
    for i, c in enumerate(classes):
        Xc = X[y == c] - means[i]
        covs.append((Xc.T @ Xc) / (counts[i] - 1) + reg * np.eye(p))
    return {"kind": "qda", "classes": classes, "means": means,
            "priors": priors, "covs": np.stack(covs)}


def _log_discriminants(model, X):
    K = len(model["classes"])
    out = np.zeros((X.shape[0], K))
    for k in range(K):
        S = model["cov"] if model["kind"] == "lda" else model["covs"][k]
        d = X - model["means"][k]
        sol = np.linalg.solve(S, d.T).T
        maha = np.sum(d * sol, axis=1)
        sign, logdet = np.linalg.slogdet(S)
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
"""Exact AUROC estimands and a fast matched unit bootstrap.

All four estimands are functions of a unit-by-unit concordance matrix K, where K[u, v] is the number of
(positive frame of unit u, negative frame of unit v) pairs that are correctly ordered (ties count 1/2),
together with the positive and negative counts of each unit. For a bootstrap draw with unit multiplicities c,
  E1    = c'Kc / ((c'P)(c'N))                       pooled AUROC of the concatenated resample
  W     = sum_u c_u K_uu / sum_u c_u P_u N_u         pair-weighted within-unit AUROC
  E2    = mean of K_uu / (P_u N_u) over drawn mixed-label units (with multiplicity)
  alpha = sum_u c_u P_u N_u / ((c'P)(c'N))
  X     = (E1 - alpha W) / (1 - alpha)               cross-unit AUROC (Proposition 1)
so a draw costs one matrix-vector product. This reproduces the estimands of the published audit
(oea_audit.estimands) exactly, including the pairs between repeated copies of a unit.
"""
import numpy as np


def prepare(units_all, unit, y, s):
    """units_all fixes the order of units (shared by every model so that draws are matched)."""
    units_all = np.asarray(units_all)
    unit, y, s = np.asarray(unit), np.asarray(y), np.asarray(s, dtype=float)
    if any(a.ndim != 1 for a in (units_all, unit, y, s)) or not (len(unit) == len(y) == len(s)):
        raise ValueError("unit, y and score must be one-dimensional arrays of equal length")
    if not len(unit) or len(np.unique(units_all)) != len(units_all):
        raise ValueError("Predictions must be nonempty and units_all must contain unique units")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Labels must be binary and scores must be finite")
    if not np.isin(unit, units_all).all():
        raise ValueError("Prediction unit is missing from units_all")
    pos_idx = {u: k for k, u in enumerate(units_all)}
    U = len(units_all)
    ui = np.array([pos_idx[u] for u in unit])
    y = np.asarray(y).astype(bool); s = np.asarray(s, dtype=float)
    P = np.bincount(ui[y], minlength=U).astype(float)
    N = np.bincount(ui[~y], minlength=U).astype(float)
    sp, up = s[y], ui[y]
    K = np.zeros((U, U))
    for v in range(U):
        neg = np.sort(s[(~y) & (ui == v)])
        if neg.size == 0 or sp.size == 0:
            continue
        lt = np.searchsorted(neg, sp, "left"); le = np.searchsorted(neg, sp, "right")
        K[:, v] = np.bincount(up, lt + 0.5 * (le - lt), minlength=U)
    return dict(K=K, P=P, N=N, d=np.diag(K).copy(), w=P * N, U=U)


def from_counts(R, c):
    """Estimands for multiplicity matrix c (B x U) or vector (U,)."""
    c = np.atleast_2d(np.asarray(c, dtype=float))
    K, P, N, d, w = R["K"], R["P"], R["N"], R["d"], R["w"]
    tot = (c @ P) * (c @ N)
    with np.errstate(invalid="ignore", divide="ignore"):
        E1 = np.einsum("bu,bu->b", c @ K, c) / tot
        W = (c @ d) / (c @ w)
        mixed = (w > 0).astype(float)
        a = np.where(w > 0, d / np.where(w > 0, w, 1), 0.0)
        E2 = (c @ (mixed * a)) / (c @ mixed)
        alpha = (c @ w) / tot
        # Compute cross-unit pairs directly: W is undefined when no unit has both classes.
        X = (np.einsum("bu,bu->b", c @ K, c) - c @ d) / (tot - c @ w)
    return dict(E1=E1, X=X, W=W, E2=E2, alpha=alpha)


def point(R):
    r = {k: float(v[0]) for k, v in from_counts(R, np.ones(R["U"])).items()}
    w = R["w"]
    r["eligible"] = int((w > 0).sum())
    r["neff"] = float(w.sum() ** 2 / (w ** 2).sum()) if (w > 0).any() else np.nan
    return r


def estimands_point(unit, y, s):
    units_all = np.unique(unit)
    return point(prepare(units_all, unit, y, s))


def draws(U, B, seed):
    """Multiplicity matrix of B unit-bootstrap resamples of U units."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, U, size=(B, U))
    c = np.zeros((B, U), dtype=np.float32)
    np.add.at(c, (np.repeat(np.arange(B), U), idx.ravel()), 1)
    return c


def bh(p):
    p = np.asarray(p, dtype=float); out = np.full(p.size, np.nan)
    ok = ~np.isnan(p); q = p[ok]; n = q.size
    if n == 0:
        return out
    o = np.argsort(q); r = np.empty(n); prev = 1.0
    for k in range(n - 1, -1, -1):
        prev = min(prev, q[o[k]] * n / (k + 1)); r[o[k]] = prev
    out[ok] = r
    return out


def kendall_tau(a, b):
    """Kendall tau-b between two score vectors (models)."""
    a = np.asarray(a); b = np.asarray(b)
    i, j = np.triu_indices(a.size, 1)
    da = np.sign(a[i] - a[j]); db = np.sign(b[i] - b[j])
    num = np.sum(da * db)
    den = np.sqrt(np.sum(da != 0) * np.sum(db != 0))
    return num / den if den > 0 else np.nan

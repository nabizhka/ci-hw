"""VaR of $1M of AAPL over 10/31/18 -> 11/1/18 with confidence intervals (MF731 HW 13, 13_CI_pub)."""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V, ALPHA, BETA, M, SEED = 1_000_000.0, 0.97, 0.02, 125_000, 731
z = stats.norm.ppf(1 - ALPHA)            # N^{-1}(1 - alpha) < 0


def load(col="Close"):
    d = pd.read_csv(os.path.join(ROOT, "data", "aapl_close.csv"), index_col=0, parse_dates=True)
    p = d.loc["2016-11-01":"2018-10-31", col]
    nxt = d.loc["2018-11-01", col]
    return p, nxt


def var_normal(mu, sig, v=V):
    """Formula (1): V (1 - exp(mu + sigma N^{-1}(1 - alpha)))."""
    return v * (1 - np.exp(mu + sig * z))


def var_empirical(x, v=V, a=ALPHA):
    """Course definition: L_(ceil(n alpha)) of the losses -V(e^X - 1)."""
    L = np.sort(-v * np.expm1(x))
    return L[int(np.ceil(len(L) * a)) - 1]


def part_a(x):
    n, xb, s = len(x), x.mean(), x.std(ddof=1)
    q_lo, q_hi = stats.chi2(n - 1).ppf(BETA / 2), stats.chi2(n - 1).ppf(1 - BETA / 2)
    s_lo, s_hi = np.sqrt((n - 1) / q_hi) * s, np.sqrt((n - 1) / q_lo) * s
    return dict(n=n, xbar=xb, s=s, q_lo=q_lo, q_hi=q_hi, s_lo=s_lo, s_hi=s_hi,
                emp=var_empirical(x), point=var_normal(xb, s),
                lo=var_normal(xb, s_lo), hi=var_normal(xb, s_hi))


def part_b(x, m=M, seed=SEED):
    """Slide 30: sigma^m from chi2_{n-1}, then mu^m ~ N(xbar, (sigma^m)^2 / n)."""
    rng = np.random.default_rng(seed)
    n, xb, s = len(x), x.mean(), x.std(ddof=1)
    sig = s * np.sqrt((n - 1) / rng.chisquare(n - 1, size=m))
    mu = rng.normal(xb, sig / np.sqrt(n))
    Y = var_normal(mu, sig)
    return dict(Y=Y, sig=sig, mu=mu, mean=Y.mean(), se=Y.std(ddof=1) / np.sqrt(m),
                A=np.quantile(Y, BETA / 2), B=np.quantile(Y, 1 - BETA / 2), median=np.median(Y))


def run():
    p, nxt = load()
    x = np.log(p).diff().dropna().values
    a, b = part_a(x), part_b(x)
    # checks
    ck = {}
    # (i) the sigma-only simulation reproduces the chi2 interval exactly in distribution
    rng = np.random.default_rng(SEED + 1)
    sig_only = a["s"] * np.sqrt((a["n"] - 1) / rng.chisquare(a["n"] - 1, size=M))
    Ya = var_normal(a["xbar"], sig_only)
    ck["a_sim"] = (np.quantile(Ya, BETA / 2), np.quantile(Ya, 1 - BETA / 2))
    # (ii) coverage of the part (a) interval when data really are iid normal with mu known
    rng = np.random.default_rng(SEED + 2)
    true_mu, true_s, n = a["xbar"], a["s"], a["n"]
    true_var = var_normal(true_mu, true_s)
    R = 20_000
    sims = rng.normal(true_mu, true_s, size=(R, n))
    ss = sims.std(axis=1, ddof=1)
    lo = var_normal(true_mu, np.sqrt((n - 1) / a["q_hi"]) * ss)
    hi = var_normal(true_mu, np.sqrt((n - 1) / a["q_lo"]) * ss)
    ck["cover_a"] = np.mean((lo <= true_var) & (true_var <= hi))
    # (iii) coverage of the part (b) interval (mu also re-estimated), smaller M per replication
    R2, m2 = 2_000, 20_000
    hit = 0
    for r in range(R2):
        xs = sims[r]
        bb = part_b(xs, m=m2, seed=SEED + 10 + r)
        hit += bb["A"] <= true_var <= bb["B"]
    ck["cover_b"] = hit / R2
    # (iv) fat tails: normality diagnostics on the actual data
    ck["skew"], ck["exkurt"] = stats.skew(x), stats.kurtosis(x)
    ck["jb_p"] = stats.jarque_bera(x).pvalue
    ck["n_exceed"] = int(np.sum(-V * np.expm1(x) > a["point"]))
    # (v) seed stability of part (b)
    ck["seeds"] = [part_b(x, seed=s) for s in (1, 2, 3)]
    # adjusted closes as a robustness line
    pa, _ = load("AdjClose")
    xa = np.log(pa).diff().dropna().values
    adj = dict(a=part_a(xa), b=part_b(xa))
    realized = -V * (nxt / p.iloc[-1] - 1)
    return dict(p=p, x=x, a=a, b=b, ck=ck, adj=adj, realized=realized, nxt=nxt)


if __name__ == "__main__":
    r = run()
    a, b, ck = r["a"], r["b"], r["ck"]
    print(f"n={a['n']} first={r['p'].index[0].date()} last={r['p'].index[-1].date()}")
    print(f"xbar={a['xbar']:.6e} S={a['s']:.6f} q=({a['q_lo']:.3f},{a['q_hi']:.3f}) sig CI=({a['s_lo']:.6f},{a['s_hi']:.6f})")
    print(f"(a) empirical {a['emp']:,.2f}  formula {a['point']:,.2f}  CI ({a['lo']:,.2f}, {a['hi']:,.2f})")
    print(f"(b) mean {b['mean']:,.2f} (se {b['se']:.2f}) median {b['median']:,.2f} CI ({b['A']:,.2f}, {b['B']:,.2f})")
    print("check a_sim", ck["a_sim"], "cover_a", ck["cover_a"], "cover_b", ck["cover_b"])
    print("skew", ck["skew"], "exkurt", ck["exkurt"], "jb_p", ck["jb_p"], "exceed", ck["n_exceed"])
    print("seeds", [(round(s['mean'], 2), round(s['A'], 2), round(s['B'], 2)) for s in ck["seeds"]])
    aa = r["adj"]
    print("adj", aa["a"]["emp"], aa["a"]["point"], aa["a"]["lo"], aa["a"]["hi"], aa["b"]["mean"], aa["b"]["A"], aa["b"]["B"])
    print("realized loss 11/1", r["realized"])

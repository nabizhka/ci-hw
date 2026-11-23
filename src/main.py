"""Run ci.py, assert the checks, write generated/values.tex and the figure."""
import os
import numpy as np
from scipy import stats, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ci

ROOT = ci.ROOT
GEN, FIG = os.path.join(ROOT, "generated"), os.path.join(ROOT, "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
r = ci.run()
a, b, ck = r["a"], r["b"], r["ck"]
m = {}


def put(k, v, fmt="{:,.0f}"):
    m[k] = v if isinstance(v, str) else fmt.format(v)


# ---- checks
n = a["n"]
assert n == 503 and r["p"].index[0].strftime("%Y-%m-%d") == "2016-11-01" and r["p"].index[-1].strftime("%Y-%m-%d") == "2018-10-31"
assert a["lo"] < a["point"] < a["hi"] and b["A"] < b["mean"] < b["B"]
assert abs(ck["a_sim"][0] / a["lo"] - 1) < 2e-3 and abs(ck["a_sim"][1] / a["hi"] - 1) < 2e-3   # sim of sigma only = chi2 CI
assert abs(ck["cover_a"] - 0.98) < 0.005                    # 20,000 reps: SE 0.1%
assert abs(ck["cover_b"] - 0.98) < 0.01                     # 2,000 reps: SE 0.3%
assert b["B"] - b["A"] > a["hi"] - a["lo"]                  # mu uncertainty widens the interval
assert max(abs(s["mean"] - b["mean"]) for s in ck["seeds"]) < 15
emp_k = int(np.ceil(n * ci.ALPHA))
# E[sigma^m]/S_n = sqrt((n-1)/2) Gamma((n-2)/2)/Gamma((n-1)/2)
bias = np.sqrt((n - 1) / 2) * np.exp(special.gammaln((n - 2) / 2) - special.gammaln((n - 1) / 2))
assert abs(b["sig"].mean() / a["s"] - bias) < 2e-4

# ---- macros
put("N", n, "{:d}"); put("Nprices", n + 1, "{:d}"); put("EmpK", emp_k, "{:d}"); put("EmpRank", n - emp_k + 1, "{:d}")
put("Xbar", 100 * a["xbar"], "{:.4f}"); put("Sn", 100 * a["s"], "{:.4f}")
put("Zq", ci.z, "{:.4f}"); put("Zalpha", -ci.z, "{:.4f}")
put("QLo", a["q_lo"], "{:.2f}"); put("QHi", a["q_hi"], "{:.2f}")
put("SigLo", 100 * a["s_lo"], "{:.4f}"); put("SigHi", 100 * a["s_hi"], "{:.4f}")
put("VarEmp", a["emp"], "{:,.2f}"); put("VarPt", a["point"], "{:,.2f}")
put("VarLo", a["lo"], "{:,.2f}"); put("VarHi", a["hi"], "{:,.2f}")
put("WidthA", a["hi"] - a["lo"], "{:,.0f}")
put("YBar", b["mean"], "{:,.2f}"); put("YSE", b["se"], "{:.2f}"); put("YMed", b["median"], "{:,.2f}")
put("BA", b["A"], "{:,.2f}"); put("BB", b["B"], "{:,.2f}"); put("WidthB", b["B"] - b["A"], "{:,.0f}")
put("WidthRatio", (b["B"] - b["A"]) / (a["hi"] - a["lo"]), "{:.2f}")
put("EmpOver", a["emp"] - a["hi"], "{:,.0f}")
put("CoverA", 100 * ck["cover_a"], "{:.2f}"); put("CoverB", 100 * ck["cover_b"], "{:.2f}")
put("SimALo", ck["a_sim"][0], "{:,.0f}"); put("SimAHi", ck["a_sim"][1], "{:,.0f}")
put("Skew", ck["skew"], "{:.2f}"); put("ExKurt", ck["exkurt"], "{:.2f}"); put("JBp", f"{ck['jb_p']:.1e}".replace("e-", r"\times10^{-") + "}")
put("NExceed", ck["n_exceed"], "{:d}"); put("NExpect", n * (1 - ci.ALPHA), "{:.1f}")
put("SigBias", 100 * (bias - 1), "{:.3f}"); put("YMinusPt", b["mean"] - a["point"], "{:,.0f}")
put("SeedMeans", ", ".join(f"{s['mean']:,.0f}" for s in ck["seeds"]))
aa = r["adj"]
put("AdjEmp", aa["a"]["emp"]); put("AdjPt", aa["a"]["point"]); put("AdjLo", aa["a"]["lo"]); put("AdjHi", aa["a"]["hi"])
put("AdjY", aa["b"]["mean"]); put("AdjA", aa["b"]["A"]); put("AdjB", aa["b"]["B"])
put("Realized", -r["realized"], "{:,.0f}"); put("NextRet", 100 * (r["nxt"] / r["p"].iloc[-1] - 1), "{:.2f}")
put("M", ci.M, "{:,d}"); put("Seed", ci.SEED, "{:d}")

with open(os.path.join(GEN, "values.tex"), "w") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- figure
INK, MUTED, BLUE, ORANGE, GREEN = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.hist(b["Y"] / 1e3, bins=120, color="#c9d7ea", edgecolor="none", density=True)
ymax = ax.get_ylim()[1]
ax.hlines(ymax * 0.96, a["lo"] / 1e3, a["hi"] / 1e3, color=BLUE, lw=4, label="(a) $\\chi^2$ interval, $\\mu=\\bar X_n$")
ax.hlines(ymax * 0.88, b["A"] / 1e3, b["B"] / 1e3, color=ORANGE, lw=4, label="(b) simulation interval $(A,B)$")
ax.axvline(a["point"] / 1e3, color=BLUE, lw=1.2, ls="--", label="formula (1) with $\\bar X_n, S_n$")
ax.axvline(a["emp"] / 1e3, color=GREEN, lw=1.6, label="empirical")
ax.set_xlabel(r"VaR$_{.97}$ (\$ thousands)", color=INK); ax.set_yticks([])
ax.set_title("Simulated $Y_m$, $M$ = 125,000: one-day VaR on \\$1M of AAPL", loc="left", color=INK)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
ax.set_ylim(0, ymax * 1.0)
ax.text(a["lo"] / 1e3, ymax * 0.96, "(a)", color=BLUE, ha="right", va="center", fontsize=9)
ax.text(b["A"] / 1e3, ymax * 0.88, "(b)", color=ORANGE, ha="right", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "var_ci.pdf")); fig.savefig(os.path.join(FIG, "var_ci.png"), dpi=170)
for k, v in m.items():
    print(k, v)

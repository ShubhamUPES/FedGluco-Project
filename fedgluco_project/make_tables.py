"""Builds the paper tables from whatever runs exist in results/ (works after any phase).

  python make_tables.py
Output: results/paper_tables.xlsx (one sheet per table) and results/tables_latex.tex
Values: mean ± std over the seeds found. Round tables show the mean over seeds.
"""
import glob
import numpy as np, pandas as pd
import config as K

R = K.RESULTS
CL = list(K.CLIENTS.values()); HZ = list(K.HORIZONS); MET = ["r2", "rmse", "mae"]
LBL = {"r2": "R²", "rmse": "RMSE", "mae": "MAE"}
ORDER = {"Local": 0, "FedAvg": 1, "FedProx": 2}


def load(pattern):
    files = sorted(glob.glob(str(R / pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


final, rounds = load("final_*.csv"), load("rounds_*.csv")
if final.empty:
    raise SystemExit("No results yet: run run_all.py first.")


def ms(x, d=3):
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def fmt(metric):
    return 3 if metric == "r2" else 2


sheets, latex = {}, []
seeds = final.groupby(["model", "setup"]).seed.nunique()

# ---- Table III: Local ----
loc = final[final.setup == "Local"]
if not loc.empty:
    rows = []
    for c in CL:
        for mdl in sorted(loc.model.unique()):
            g = loc[(loc.client == c) & (loc.model == mdl)]
            r = {"Client": c, "Model": mdl}
            for hz in HZ:
                for m in MET:
                    r[f"{LBL[m]} {hz} min"] = ms(g[g.horizon_min == hz][m], fmt(m))
            rows.append(r)
    sheets["Table III Local"] = pd.DataFrame(rows)

# ---- Tables IV-VI: per round ----
if not rounds.empty:
    for num, m in (("IV", "r2"), ("V", "rmse"), ("VI", "mae")):
        for hz in HZ:
            g = rounds[rounds.horizon_min == hz].groupby(["model", "setup", "client", "round"])[m].mean().unstack("round")
            g.columns = [f"R{c}" for c in g.columns]
            g = g.reset_index().rename(columns={"model": "Model", "setup": "Algorithm", "client": "Client"})
            g["c"] = g.Client.map({c: i for i, c in enumerate(CL)})
            g = g.sort_values(["Model", "Algorithm", "c"]).drop(columns="c").round(fmt(m))
            sheets[f"Table {num} {LBL[m]} {hz}min"] = g

# ---- Tables VII-VIII: final comparison ----
for num, hz in (("VII", 30), ("VIII", 60)):
    rows = []
    for (mdl, st), g in sorted(final[final.horizon_min == hz].groupby(["model", "setup"]), key=lambda x: (x[0][0], ORDER[x[0][1]])):
        r = {"Model": mdl, "Setup": st}
        for c in CL:
            for m in MET:
                r[f"{c} {LBL[m]}"] = ms(g[g.client == c][m], fmt(m))
        rows.append(r)
    sheets[f"Table {num} final {hz}min"] = pd.DataFrame(rows)

# ---- Table IX: % change vs Local ----
means = final.groupby(["model", "setup", "client", "horizon_min"])[MET].mean()
rows = []
for (mdl, st) in sorted({(a, b) for a, b, *_ in means.index if b != "Local"}, key=lambda x: (x[0], ORDER[x[1]])):
    if (mdl, "Local") not in {(a, b) for a, b, *_ in means.index}:
        continue
    for c in CL:
        r = {"Model": mdl, "Algorithm": st, "Client": c}
        for hz in HZ:
            for m in MET:
                base, val = means.loc[(mdl, "Local", c, hz), m], means.loc[(mdl, st, c, hz), m]
                pct = 100 * (val - base) / abs(base)
                better = (pct > 0) if m == "r2" else (pct < 0)
                r[f"{LBL[m]} {hz} min"] = f"{'↑' if pct > 0 else '↓'}{abs(pct):.1f}% ({'better' if better else 'worse'})"
        rows.append(r)
if rows:
    sheets["Table IX change vs Local"] = pd.DataFrame(rows)

with pd.ExcelWriter(R / "paper_tables.xlsx") as xw:
    pd.DataFrame({"Info": [f"Seeds per model/setup: {seeds.to_dict()}",
                           "Values: mean ± std over seeds (round tables: mean).",
                           "RMSE and MAE in mg/dL. R² computed per client.",
                           "Final models = best validation checkpoint (Local: over epochs; FL: over rounds)."]}).to_excel(xw, sheet_name="Info", index=False)
    for name, t in sheets.items():
        t.to_excel(xw, sheet_name=name[:31], index=False)
        latex.append(f"% {name}\n" + t.to_latex(index=False).replace("±", r"$\pm$").replace("²", r"$^2$")
                     .replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$").replace("%", r"\%"))
(R / "tables_latex.tex").write_text("\n\n".join(latex), encoding="utf-8")

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
for name, t in sheets.items():
    print(f"\n=== {name} ===\n{t.to_string(index=False)}")
print(f"\nSaved {R / 'paper_tables.xlsx'} and tables_latex.tex")

"""Line charts for the paper (run after phase 3; partial runs also work).

  python make_figures.py --model lstm     # or tst
Fig. 2 / 3 : actual vs predicted glucose, 30 / 60 min horizon, one test patient-day per client.
             Patient selection is fixed in advance: the test series with the most windows; its first 96 windows (24 h).
             Seed 0 models.
Fig. 4     : RMSE of the global model per round (mean ± std over seeds) for FedAvg and FedProx,
             with Local RMSE as a dashed reference line.
Output: results/figures/*.png (300 dpi) and *.pdf
"""
import argparse, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as K

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lstm", choices=K.MODELS)
a = ap.parse_args()
M = a.model.upper(); R = K.RESULTS; OUT = R / "figures"; OUT.mkdir(parents=True, exist_ok=True)
CL = list(K.CLIENTS.values())
COL = {"Actual": "#1f1f1d", "Local": "#2a78d6", "FedAvg": "#eb6834", "FedProx": "#1baf7a"}
STY = {"Actual": dict(lw=2.0), "Local": dict(lw=1.4, ls="-"), "FedAvg": dict(lw=1.4, ls="--"), "FedProx": dict(lw=1.4, ls=":")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6})


def load(pattern):
    f = sorted(glob.glob(str(R / pattern)))
    return pd.concat([pd.read_csv(x) for x in f], ignore_index=True) if f else pd.DataFrame()


# ---------- Fig 2 / 3 ----------
pred = load(f"pred_{a.model}_*_s0.csv")
if not pred.empty:
    loc = pred[(pred.setup == "Local") & (pred.horizon_min == 30)]
    pick = {}
    for c in CL:                                   # fixed rule, decided before looking at errors
        g = loc[loc.client == c]
        sid = g.series_id.value_counts().sort_index().idxmax() if not g.empty else None
        pick[c] = sid
    for fig_no, hz in ((2, 30), (3, 60)):
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        for j, c in enumerate(CL):
            x = ax[j]; sid = pick[c]
            if sid is None: continue
            first = True
            for st in ["Local", "FedAvg", "FedProx"]:
                g = pred[(pred.client == c) & (pred.horizon_min == hz) & (pred.setup == st) & (pred.series_id == sid)]
                if g.empty: continue
                g = g.sort_values("last_input_time").head(96)
                t = (pd.to_datetime(g.last_input_time) - pd.to_datetime(g.last_input_time).iloc[0]).dt.total_seconds() / 3600 + hz / 60
                if first:
                    x.plot(t, g.actual, color=COL["Actual"], label="Actual", **STY["Actual"]); first = False
                x.plot(t, g.pred, color=COL[st], label=st, **STY[st])
            x.set_title(c, fontweight="bold"); x.set_xlabel("Time (h)")
            if j == 0: x.set_ylabel("Glucose (mg/dL)")
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, ncol=4, loc="upper center", frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        for ext in ("png", "pdf"): fig.savefig(OUT / f"fig{fig_no}_{a.model}_actual_vs_pred_{hz}min.{ext}", dpi=300)
        plt.close(fig)
    print("Fig 2/3 series used:", pick)

# ---------- Fig 4 ----------
rounds, final = load(f"rounds_{a.model}_*.csv"), load(f"final_{a.model}_local_*.csv")
if not rounds.empty:
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
    for i, hz in enumerate(K.HORIZONS):
        for j, c in enumerate(CL):
            x = ax[i, j]
            for st in ["FedAvg", "FedProx"]:
                g = rounds[(rounds.setup == st) & (rounds.client == c) & (rounds.horizon_min == hz)].groupby("round").rmse
                if g.ngroups == 0: continue
                mu, sd = g.mean(), g.std().fillna(0)
                x.plot(mu.index, mu.values, color=COL[st], marker="o", ms=4, lw=1.6, label=st)
                x.fill_between(mu.index, mu - sd, mu + sd, color=COL[st], alpha=0.15, lw=0)
            if not final.empty:
                lv = final[(final.client == c) & (final.horizon_min == hz)].rmse.mean()
                x.axhline(lv, color=COL["Local"], ls="--", lw=1.4, label="Local (best checkpoint)")
            if i == 0: x.set_title(c, fontweight="bold")
            if j == 0: x.set_ylabel(f"RMSE (mg/dL), {hz} min")
            if i == 1: x.set_xlabel("Communication round")
            x.set_xticks(range(1, K.FL_ROUNDS + 1))
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"): fig.savefig(OUT / f"fig4_{a.model}_rmse_vs_round.{ext}", dpi=300)
    plt.close(fig)
print(f"Figures saved in {OUT}")

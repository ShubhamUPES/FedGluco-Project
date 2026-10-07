"""Runs one phase (or all) of the plan.

  python run_all.py --phase 1     # Local LSTM, 3 seeds           (3 runs; each trains 3 client models)
  python run_all.py --phase 2     # Local TST, 3 seeds            (3 runs)
  python run_all.py --phase 3     # FedAvg + FedProx, LSTM + TST  (12 runs)
  python run_all.py --phase all
Already-finished runs are skipped, so you can stop and restart safely.
"""
import argparse, subprocess, sys
import config as K

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=["1", "2", "3", "all"])
a = ap.parse_args()

jobs = []
if a.phase in ("1", "all"):
    jobs += [(["train_local.py", "--model", "lstm", "--seed", str(s)], f"final_lstm_local_s{s}.csv") for s in K.SEEDS]
if a.phase in ("2", "all"):
    jobs += [(["train_local.py", "--model", "tst", "--seed", str(s)], f"final_tst_local_s{s}.csv") for s in K.SEEDS]
if a.phase in ("3", "all"):
    jobs += [(["train_federated.py", "--model", m, "--algo", al, "--seed", str(s)], f"final_{m}_{al}_s{s}.csv")
             for m in K.MODELS for al in K.FL_ALGOS for s in K.SEEDS]

for i, (cmd, out) in enumerate(jobs, 1):
    if (K.RESULTS / out).exists():
        print(f"[{i}/{len(jobs)}] skip (exists): {out}"); continue
    print(f"[{i}/{len(jobs)}] python {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable] + cmd, check=True, cwd=K.ROOT)
print("phase finished")

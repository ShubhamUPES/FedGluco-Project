"""Local training: each client trains its own model on its own data only.

Example:  python train_local.py --model lstm --seed 0
Output :  results/final_lstm_local_s0.csv, pred_..., log_...
"""
import argparse, copy, time
import torch
import config as K
from common import DEVICE, seed_all, load_clients, make_model, n_params, train_one_epoch, val_mse, save_outputs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=K.MODELS)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=K.LOCAL_EPOCHS_TOTAL)
a = ap.parse_args()

t0 = time.time()
data = load_clients()
final, log = {}, dict(model=a.model, setup="local", seed=a.seed, device=str(DEVICE), epochs=a.epochs, clients={})
for c in data:
    seed_all(a.seed)
    m = make_model(a.model); opt = torch.optim.Adam(m.parameters(), lr=K.LR)
    Xtr, Ytr, _ = data[c]["train"]; Xv, Yv, _ = data[c]["val"]
    best, best_state, best_ep, curve = float("inf"), None, 0, []
    for ep in range(1, a.epochs + 1):
        train_one_epoch(m, Xtr, Ytr, opt)
        v = val_mse(m, Xv, Yv); curve.append(round(v, 6))
        if v < best:                                 # model selection on validation only
            best, best_state, best_ep = v, copy.deepcopy(m.state_dict()), ep
    m.load_state_dict(best_state); final[c] = m
    log["clients"][c] = dict(best_epoch=best_ep, val_curve=curve, train_windows=len(Xtr))
    print(f"  {c}: best epoch {best_ep}")
log.update(params=n_params(m), seconds=round(time.time() - t0, 1))

rows = save_outputs(f"{a.model}_local_s{a.seed}", a.model, "Local", a.seed, final, data, log)
for r in rows:
    print(f"{a.model.upper()} Local seed{a.seed} {r['client']:8s} {r['horizon_min']}min  "
          f"R2 {r['r2']:.3f}  RMSE {r['rmse']:.2f}  MAE {r['mae']:.2f}")
print(f"done in {time.time() - t0:.0f}s on {DEVICE}")

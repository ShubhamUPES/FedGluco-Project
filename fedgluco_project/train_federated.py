"""Federated training (FedAvg / FedProx), simulated: each client trains on its own data only;
the server sees only model weights.

Per round: every client starts from the global model, trains FL_LOCAL_EPOCHS on its own data,
sends weights; the server averages them weighted by each client's number of training windows.
After every round the global model is scored on each client's test set (for the round tables /
convergence plot only). The final model is the round with the lowest validation loss.

Example:  python train_federated.py --model lstm --algo fedavg --seed 0
Output :  results/final_lstm_fedavg_s0.csv, rounds_..., pred_..., log_...
"""
import argparse, copy, time
import pandas as pd, torch
import config as K
from common import DEVICE, seed_all, load_clients, make_model, n_params, train_one_epoch, val_mse, evaluate, save_outputs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=K.MODELS)
ap.add_argument("--algo", required=True, choices=K.FL_ALGOS)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=K.FL_ROUNDS)
ap.add_argument("--local_epochs", type=int, default=K.FL_LOCAL_EPOCHS)
ap.add_argument("--mu", type=float, default=K.FEDPROX_MU)
a = ap.parse_args()
SETUP = {"fedavg": "FedAvg", "fedprox": "FedProx"}[a.algo]

t0 = time.time()
data = load_clients(); clients = list(data)
seed_all(a.seed)
glob = make_model(a.model)
n = {c: len(data[c]["train"][0]) for c in clients}; N = sum(n.values())
best, best_state, best_round, per_round, val_curve = float("inf"), None, 0, [], []

for r in range(1, a.rounds + 1):
    states = {}
    for c in clients:                                           # client side
        local = copy.deepcopy(glob)
        opt = torch.optim.Adam(local.parameters(), lr=K.LR)
        ref = {k: v.detach().clone() for k, v in glob.named_parameters()} if a.algo == "fedprox" else None
        Xtr, Ytr, _ = data[c]["train"]
        for _ in range(a.local_epochs):
            train_one_epoch(local, Xtr, Ytr, opt, prox_ref=ref, mu=a.mu)
        states[c] = local.state_dict()                          # only weights leave the client
    glob.load_state_dict({k: sum(n[c] / N * states[c][k].float() for c in clients).to(states[clients[0]][k].dtype)
                          for k in states[clients[0]]})         # server: weighted average

    v = sum(n[c] / N * val_mse(glob, *data[c]["val"][:2]) for c in clients)
    val_curve.append(round(v, 6))
    for c in clients:                                           # reporting only, not used for selection
        sc, _ = evaluate(glob, data[c])
        for hz, d in sc.items():
            per_round.append(dict(model=a.model.upper(), setup=SETUP, seed=a.seed, round=r, client=c, horizon_min=hz, **d))
    if v < best:
        best, best_state, best_round = v, copy.deepcopy(glob.state_dict()), r
    msg = " | ".join(f"{x['client']} {x['horizon_min']}m RMSE {x['rmse']:.2f}" for x in per_round[-6:])
    print(f"round {r:2d}  val {v:.4f}  {msg}")

K.RESULTS.mkdir(exist_ok=True)
pd.DataFrame(per_round).to_csv(K.RESULTS / f"rounds_{a.model}_{a.algo}_s{a.seed}.csv", index=False)
glob.load_state_dict(best_state)
log = dict(model=a.model, setup=a.algo, seed=a.seed, device=str(DEVICE), rounds=a.rounds, local_epochs=a.local_epochs,
           mu=a.mu if a.algo == "fedprox" else None, best_round=best_round, val_curve=val_curve,
           client_weights={c: round(n[c] / N, 4) for c in clients}, params=n_params(glob), seconds=round(time.time() - t0, 1))
rows = save_outputs(f"{a.model}_{a.algo}_s{a.seed}", a.model, SETUP, a.seed, {c: glob for c in clients}, data, log)
print(f"best round {best_round}")
for r in rows:
    print(f"{a.model.upper()} {SETUP} seed{a.seed} {r['client']:8s} {r['horizon_min']}min  "
          f"R2 {r['r2']:.3f}  RMSE {r['rmse']:.2f}  MAE {r['mae']:.2f}")
print(f"done in {time.time() - t0:.0f}s on {DEVICE}")

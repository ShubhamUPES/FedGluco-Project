"""Shared code: data windows, models, training step, metrics, saving."""
import json, random
import numpy as np, pandas as pd, torch, torch.nn as nn
import config as K

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


scale = lambda x: (x - K.SCALE_CENTER) / K.SCALE_SPREAD
unscale = lambda z: z * K.SCALE_SPREAD + K.SCALE_CENTER


# ---------------- data ----------------
def _windows(g):
    """Sliding windows inside each continuous series. A window is dropped if any target point
    was interpolated or sensor-censored (its true value is unknown)."""
    X, Y, meta = [], [], []
    for sid, s in g.groupby("series_id", sort=True):
        v = s.glucose_mg_dl.to_numpy(); ts = s.timestamp.to_numpy()
        bad = ((s.interpolated == 1) | (s.censored == 1)).to_numpy(); pid = s.patient_id.iloc[0]
        for i in range(len(v) - K.INPUT_STEPS - K.OUTPUT_STEPS + 1):
            if bad[i + K.INPUT_STEPS:i + K.INPUT_STEPS + K.OUTPUT_STEPS].any():
                continue
            X.append(v[i:i + K.INPUT_STEPS]); Y.append(v[i + K.INPUT_STEPS:i + K.INPUT_STEPS + K.OUTPUT_STEPS])
            meta.append((pid, sid, ts[i + K.INPUT_STEPS - 1]))
    X = torch.tensor(scale(np.array(X)), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(scale(np.array(Y)), dtype=torch.float32)
    return X, Y, pd.DataFrame(meta, columns=["patient_id", "series_id", "last_input_time"])


def load_clients():
    """Returns {client: {'train'|'val'|'test': (X, Y, meta)}}. Each client's data stays in its own entry."""
    df = pd.read_csv(K.DATA_FILE)
    df["client"] = df.hospital.map(K.CLIENTS)
    return {c: {sp: _windows(g[g[K.SPLIT_COLUMN] == sp]) for sp in ("train", "val", "test")}
            for c, g in df.groupby("client")}


# ---------------- models (~18k parameters each) ----------------
class LSTMNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, K.OUTPUT_STEPS)

    def forward(self, x):
        return self.head(self.body(x)[0][:, -1])


class TSTNet(nn.Module):
    """Time-series Transformer encoder: linear embedding + learned positions + 2 encoder layers + mean pooling."""
    def __init__(self, d=32, heads=4, ff=64, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(1, K.INPUT_STEPS, d)); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, K.OUTPUT_STEPS)

    def forward(self, x):
        return self.head(self.enc(self.inp(x) + self.pos).mean(1))


def make_model(name):
    return {"lstm": LSTMNet, "tst": TSTNet}[name]().to(DEVICE)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------- training ----------------
def train_one_epoch(model, X, Y, opt, prox_ref=None, mu=0.0):
    model.train(); idx = torch.randperm(len(X))
    for i in range(0, len(X), K.BATCH):
        b = idx[i:i + K.BATCH]
        xb, yb = X[b].to(DEVICE), Y[b].to(DEVICE)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xb), yb)
        if prox_ref is not None:      # FedProx: (mu/2) * ||w - w_global||^2
            loss = loss + mu / 2 * sum(((p - prox_ref[n]) ** 2).sum() for n, p in model.named_parameters())
        loss.backward(); opt.step()


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.cat([model(X[i:i + 2048].to(DEVICE)).cpu() for i in range(0, len(X), 2048)])


@torch.no_grad()
def val_mse(model, X, Y):
    return nn.functional.mse_loss(predict(model, X), Y).item()


# ---------------- metrics ----------------
def evaluate(model, data):
    """R2, RMSE, MAE (mg/dL) on the client's test set at each horizon, plus the predictions."""
    X, Y, meta = data["test"]
    P, T = unscale(predict(model, X)).numpy(), unscale(Y).numpy()
    scores, preds = {}, []
    for hz, k in K.HORIZONS.items():
        y, p = T[:, k - 1], P[:, k - 1]; e = p - y
        scores[hz] = dict(r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                          rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), n=int(len(y)))
        pr = meta.copy(); pr["horizon_min"] = hz; pr["actual"] = y.round(2); pr["pred"] = p.round(2)
        preds.append(pr)
    return scores, pd.concat(preds)


def save_outputs(tag, model_name, setup, seed, final_models, data, log):
    K.RESULTS.mkdir(exist_ok=True)
    rows, preds = [], []
    for c, m in final_models.items():
        sc, pr = evaluate(m, data[c])
        for hz, d in sc.items():
            rows.append(dict(model=model_name.upper(), setup=setup, seed=seed, client=c, horizon_min=hz, **d))
        pr.insert(0, "client", c); preds.append(pr)
    pd.DataFrame(rows).to_csv(K.RESULTS / f"final_{tag}.csv", index=False)
    pd.concat(preds).assign(model=model_name.upper(), setup=setup, seed=seed).to_csv(K.RESULTS / f"pred_{tag}.csv", index=False)
    json.dump(log, open(K.RESULTS / f"log_{tag}.json", "w"), indent=1, default=str)
    return rows

# FedGluco: Local vs Federated glucose forecasting (3 clinical sites)

Real CGM data from three institutions, each a federated client:

| Client | Institution | Country | Population | Patients (train/val/test) |
|---|---|---|---|---|
| Madrid | Hospital Universitario de Móstoles | Spain | at risk of / with type 2 diabetes | 40 / 8 / 8 |
| Stanford | Stanford University School of Medicine | USA | normoglycaemic, prediabetic, type 2 | 37 / 8 / 8 |
| T1DX | T1D Exchange Clinic Network (Jaeb Center) | USA | type 1 diabetes | 37 / 8 / 8 |

Task: last 16 readings (4 h, 15-min grid) → glucose 30 and 60 min ahead. Metrics: R², RMSE, MAE (mg/dL), per client.

## Setup (VS Code)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
For an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org first; the scripts use the GPU automatically.

## Run (one phase at a time, then share the printed results)

```bash
python run_all.py --phase 1     # Local LSTM: 3 clients x 3 seeds, 50 epochs
python run_all.py --phase 2     # Local TST
python run_all.py --phase 3     # FedAvg + FedProx, LSTM + TST, 10 rounds x 5 local epochs, 3 seeds
python make_tables.py           # -> results/paper_tables.xlsx, results/tables_latex.tex
python make_figures.py --model lstm
python make_figures.py --model tst
```
Single runs: `python train_local.py --model lstm --seed 0`, `python train_federated.py --model tst --algo fedprox --seed 1`.
Finished runs are skipped by `run_all.py`, so it is safe to stop and restart.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every setting (epochs, rounds, μ, seeds, horizons). Change values only here. |
| `common.py` | Data windows, LSTM and TST models, training step, metrics, saving |
| `train_local.py` | Each client trains alone (50 epochs, best validation checkpoint) |
| `train_federated.py` | FedAvg / FedProx simulation; logs every round |
| `run_all.py` | Runs a phase |
| `make_tables.py` | Paper Tables III–IX from `results/` |
| `make_figures.py` | Fig. 2–3 actual vs predicted (30 / 60 min), Fig. 4 RMSE vs round |
| `data/glucose_3hospitals_30k_strat.csv` | Dataset with the stratified patient split (`strat_split`) |

## Fairness rules built into the code
- Same test patients for every run; a patient is never in two splits.
- Same model size (LSTM 17.5k, TST 17.8k parameters), optimizer (Adam 1e-3), batch 64, MSE loss.
- Same training budget: Local 50 epochs = FL 10 rounds × 5 local epochs.
- Model selection uses validation data only. Per-round test scores are logged for the round tables and plots, never for choosing a model.
- Targets that were interpolated or sensor-censored are never scored.
- Fixed scaling (x − 100)/50 at every client, so no statistics are shared.

## Outputs in `results/`
`final_*.csv` final metrics · `rounds_*.csv` per-round metrics (FL) · `pred_*.csv` test predictions · `log_*.json` config, best epoch/round, validation curves, runtime.

## Data and citation
Source data: GlucoBench (Sergazinov et al., ICLR 2024) raw data — Colas et al. 2019 (CC BY), Hall et al. 2018 (CC BY), Weinstock et al. 2016 (CC BY-SA). Dates in Madrid and T1DX are de-identified placeholders; time intervals are real.

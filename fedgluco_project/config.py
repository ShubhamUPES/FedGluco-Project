"""All fixed settings in one place. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "glucose_3hospitals_30k_strat.csv"
RESULTS = ROOT / "results"

# clients (column `hospital` in the CSV -> short name used in the paper)
CLIENTS = {"H1_Spain_Colas": "Madrid", "H2_USA_Hall": "Stanford", "H3_USA_Weinstock": "T1DX"}
SPLIT_COLUMN = "strat_split"          # stratified patient split: train / val / test

# task
INPUT_STEPS = 16                      # 16 x 15 min = 4 h of history
OUTPUT_STEPS = 4                      # next 60 min
HORIZONS = {30: 2, 60: 4}             # minutes -> step index (1-based)
SCALE_CENTER, SCALE_SPREAD = 100.0, 50.0   # x' = (x - 100) / 50, identical for every client

# training (shared by all setups)
LR = 1e-3
BATCH = 64
LOCAL_EPOCHS_TOTAL = 50               # Local training
FL_ROUNDS = 10                        # Federated
FL_LOCAL_EPOCHS = 5                   # per round -> 10 x 5 = 50 epochs of local work (same budget)
FEDPROX_MU = 0.01
SEEDS = [0, 1, 2]

MODELS = ["lstm", "tst"]
FL_ALGOS = ["fedavg", "fedprox"]

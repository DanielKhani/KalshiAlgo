"""
Central configuration for the NCAAB Kalshi prediction model.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "stored"
MODEL_DIR = PROJECT_ROOT / "model" / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── Seasons ────────────────────────────────────────────────────────────────
# College basketball season runs Nov -> April
# Format: use the year the season ends (e.g., 2025 = 2024-25 season)
CURRENT_SEASON = 2026
TRAINING_SEASONS = list(range(2003, 2026))  # 2003-15 through 2024-25

# ── Feature Engineering ────────────────────────────────────────────────────
ROLLING_WINDOWS = [3, 5, 10]  # games to look back for rolling stats
MIN_GAMES_PLAYED = 5  # minimum games before we trust team stats

# ── Model ──────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "seed": 42,
}

# Walk-forward validation: train on N seasons, validate on next 1
WALK_FORWARD_TRAIN_SEASONS = 3
WALK_FORWARD_VAL_SEASONS = 1


# ── Betting / Kalshi ──────────────────────────────────────────────────────
STARTING_BANKROLL = float(os.getenv("STARTING_BANKROLL", 200))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", 0.25))  # quarter Kelly
MIN_EDGE_THRESHOLD = float(os.getenv("MIN_EDGE_THRESHOLD", 0.20))  # x% min edge to bet
MAX_BET_FRACTION = 0.03  # never risk more than x% of bankroll on one game

# Kalshi specific
KALSHI_FEE_RATE = 0.0  # Kalshi charges no fees on contracts currently
# but they take a cut on withdrawals — factor that in separately

# ── Odds API ───────────────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT_KEY = "basketball_ncaab"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads,totals"

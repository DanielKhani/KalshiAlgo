# NCAAB Kalshi Prediction Model

A machine learning pipeline for predicting NCAA Men's Basketball game outcomes,
designed for trading on Kalshi prediction markets.

## Overview

This project uses LightGBM to predict win probabilities for college basketball games,
then compares those probabilities to Kalshi market prices to identify value trades.

### Pipeline

1. **Data Collection** — Fetch team stats and historical game data
2. **Feature Engineering** — Build predictive features (efficiency, tempo, SOS, etc.)
3. **Model Training** — Train LightGBM with walk-forward validation, optimized for log loss
4. **Calibration** — Apply isotonic regression for well-calibrated probabilities
5. **Backtesting** — Evaluate edge vs historical market prices
6. **Prediction** — Generate daily predictions and Kelly-sized trade recommendations

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ncaab-kalshi-model.git
cd ncaab-kalshi-model

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your API keys
```

## API Keys Needed

- **The Odds API** (free tier): https://the-odds-api.com/ — for current/historical odds
- **Optional: KenPom** ($25/yr): https://kenpom.com/ — for advanced efficiency metrics

## Usage

```bash
# Step 1: Fetch historical data and build dataset
python -m data.fetch_team_stats

# Step 2: Train the model
python -m model.train

# Step 3: Run backtest
python -m backtest.evaluate

# Step 4: Get today's predictions
python -m model.predict
```

## Project Structure

```
ncaab-kalshi-model/
├── README.md
├── requirements.txt
├── config.py              # Central configuration
├── .env.example           # Template for API keys
├── data/
│   ├── __init__.py
│   ├── fetch_team_stats.py   # Scrape/API team stats + game results
│   └── fetch_odds.py         # Pull odds from The Odds API
├── features/
│   ├── __init__.py
│   └── engineer.py           # Feature engineering pipeline
├── model/
│   ├── __init__.py
│   ├── train.py              # LightGBM training with walk-forward CV
│   ├── calibrate.py          # Probability calibration
│   └── predict.py            # Generate predictions + trade recs
├── backtest/
│   ├── __init__.py
│   └── evaluate.py           # Backtest against historical lines
└── notebooks/
    └── exploration.ipynb     # EDA and experimentation
```

## Bankroll Management

The model uses a fractional Kelly Criterion for bet sizing:
- Full Kelly is mathematically optimal but volatile
- We default to **quarter Kelly** for safety
- With a $200 bankroll, expect $2-8 per trade
- Track everything — the model logs all predictions and outcomes

## Important Notes

- **Paper trade first.** Run for at least 2-3 weeks before using real money.
- **This is not financial advice.** Sports betting involves real risk of loss.
- **Start small.** Even with a validated edge, variance is significant.
- **The model will be wrong sometimes.** That's expected. Trust the process over a large sample.

## License

MIT

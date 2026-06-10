"""
Backtesting framework for evaluating model profitability — honest edition.

What changed vs v1 (IMPORTANT):
  v1 generated synthetic market prices FROM THE GAME OUTCOME
  (y_true * 0.3 + 0.35 + noise) — the simulated market knew who won,
  so every ROI number it produced was meaningless.

  v2 supports two market sources:
    1. REAL closing lines via --odds-file (CSV keyed by game_id with a
       `market_home_prob` column). This is the only mode whose ROI means
       anything about live trading.
    2. DIAGNOSTIC mode (default): the "market" is a walk-forward logistic
       regression on two public features (season margin diff, win pct diff).
       It never sees outcomes. Beating it shows your model extracts more
       signal than a naive public model — it does NOT prove an edge over
       real markets, which are far sharper. Output is labeled accordingly.

  Also new: Kalshi taker fees modeled on every entry, and predictions are
  the walk-forward CALIBRATED out-of-fold probabilities saved by train.py.

Usage:
    python -m backtest.evaluate                      # diagnostic market
    python -m backtest.evaluate --odds-file lines.csv  # real closing lines
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from pathlib import Path
import argparse
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from model.predict import kelly_criterion


def load_oof_predictions() -> pd.DataFrame:
    path = config.MODEL_DIR / "oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No OOF predictions at {path}. Run `python -m model.train` first."
        )
    oof = pd.read_csv(path, parse_dates=["date"])
    prob_col = "cal_prob" if "cal_prob" in oof.columns else "raw_prob"
    oof["model_prob"] = oof[prob_col]
    return oof.sort_values("date").reset_index(drop=True)


def build_diagnostic_market(oof: pd.DataFrame) -> np.ndarray:
    """
    Walk-forward logistic 'market': for each season, fit on PRIOR seasons'
    simple features only. Never sees the current season or any outcome it
    is pricing. Seasons without enough history get NaN (excluded).
    """
    feats = [c for c in ["diff_season_margin", "diff_season_win_pct"] if c in oof.columns]
    if not feats:
        raise ValueError("OOF file lacks simple feature columns — retrain with updated train.py")

    prices = np.full(len(oof), np.nan)
    seasons = sorted(oof["season"].unique())

    for s in seasons:
        prior = oof[oof["season"] < s].dropna(subset=feats)
        cur_mask = (oof["season"] == s) & oof[feats].notna().all(axis=1)
        if len(prior) < 2000 or cur_mask.sum() == 0:
            continue
        lr = LogisticRegression(max_iter=1000)
        lr.fit(prior[feats].values, prior["y_true"].values)
        prices[cur_mask.values] = lr.predict_proba(oof.loc[cur_mask, feats].values)[:, 1]

    return np.clip(prices, 0.03, 0.97)


def simulate_kalshi_trading(
    y_true: np.ndarray,
    model_probs: np.ndarray,
    market_prices: np.ndarray,
    bankroll: float = None,
    kelly_fraction: float = None,
    min_edge: float = None,
    flat_staking: bool = True,
    seasons: np.ndarray = None,
) -> dict:
    """
    Simulate Kalshi trading. market_prices are REQUIRED — this function no
    longer invents a market. Fees are charged on entry per Kalshi's formula.
    """
    bankroll = bankroll or config.STARTING_BANKROLL
    kelly_fraction = kelly_fraction or config.KELLY_FRACTION
    min_edge = min_edge if min_edge is not None else config.MIN_EDGE_THRESHOLD
    # Flat staking: size every bet off the starting bankroll. Honest for
    # long backtests; compounding across decades produces absurd numbers.

    current_bankroll = bankroll
    peak_bankroll = bankroll
    trades = []
    bankroll_history = [bankroll]

    for i in range(len(y_true)):
        p, price, outcome = model_probs[i], market_prices[i], y_true[i]
        if np.isnan(price):
            continue

        # Evaluate both sides; fee makes the effective entry price worse
        candidates = []
        for side, side_p, side_price, wins in [
            ("HOME", p, price, outcome == 1),
            ("AWAY", 1 - p, 1 - price, outcome == 0),
        ]:
            fee_per_contract = config.kalshi_fee(1, side_price)
            eff_price = side_price + fee_per_contract
            edge = side_p - eff_price
            if edge > min_edge:
                candidates.append((edge, side, side_p, side_price, eff_price, wins))

        trade = {"game_idx": i, "model_prob": p, "market_price": price,
                 "outcome": outcome, "side": "PASS", "bet_amount": 0.0,
                 "fee": 0.0, "profit": 0.0, "won": None,
                 "season": seasons[i] if seasons is not None else None}

        if candidates:
            edge, side, side_p, side_price, eff_price, wins = max(candidates)
            bet_frac = kelly_criterion(side_p, eff_price, kelly_fraction)
            sizing_base = bankroll if flat_staking else current_bankroll
            bet_amount = min(sizing_base * bet_frac,
                             sizing_base * config.MAX_BET_FRACTION)

            if bet_amount > 0.50:
                contracts = bet_amount / side_price
                fee = config.kalshi_fee(contracts, side_price)
                profit = contracts * (1 - side_price) - fee if wins else -(bet_amount + fee)
                current_bankroll += profit
                peak_bankroll = max(peak_bankroll, current_bankroll)
                trade.update({"side": side, "edge": edge, "bet_amount": bet_amount,
                              "fee": fee, "profit": profit, "won": bool(wins)})

        trade["bankroll_after"] = current_bankroll
        trades.append(trade)
        bankroll_history.append(current_bankroll)

        if current_bankroll < 1.0:
            print(f"💀 Bankrupt at game {i}!")
            break

    trades_df = pd.DataFrame(trades)
    active = trades_df[trades_df["side"] != "PASS"]

    if len(active) > 0:
        total_wagered = active["bet_amount"].sum()
        total_profit = active["profit"].sum()
        total_fees = active["fee"].sum()
        win_rate = active["won"].astype(bool).mean()
        roi = total_profit / total_wagered
        bh = np.array(bankroll_history)
        running_max = np.maximum.accumulate(bh)
        max_drawdown = ((bh - running_max) / running_max).min()
    else:
        total_wagered = total_profit = total_fees = win_rate = roi = max_drawdown = 0

    return {
        "starting_bankroll": bankroll, "ending_bankroll": current_bankroll,
        "total_profit": total_profit, "total_wagered": total_wagered,
        "total_fees": total_fees, "roi": roi,
        "n_games": int(len(y_true)), "n_bets": int(len(active)),
        "bet_rate": len(active) / max(len(y_true), 1),
        "win_rate": win_rate, "max_drawdown": max_drawdown,
        "peak_bankroll": peak_bankroll,
        "bankroll_history": bankroll_history, "trades": trades_df,
    }


def run_backtest(odds_file: str = None) -> dict:
    oof = load_oof_predictions()

    if odds_file:
        market_label = f"REAL closing lines ({odds_file})"
        lines = pd.read_csv(odds_file)
        oof = oof.merge(lines[["game_id", "market_home_prob"]], on="game_id", how="left")
        market = oof["market_home_prob"].values
        matched = np.isfinite(market).sum()
        print(f"📊 Matched real lines for {matched}/{len(oof)} games")
    else:
        market_label = "DIAGNOSTIC (walk-forward logistic on public features)"
        market = build_diagnostic_market(oof)

    valid = np.isfinite(market)
    y_true = oof.loc[valid, "y_true"].values
    model_probs = oof.loc[valid, "model_prob"].values
    market = market[valid]

    # Model quality vs the market, independent of betting strategy
    model_ll = log_loss(y_true, model_probs)
    market_ll = log_loss(y_true, market)

    print(f"\n{'='*64}")
    print(f"BACKTEST — market source: {market_label}")
    print(f"{'='*64}")
    print(f"Games: {len(y_true)} | Bankroll: ${config.STARTING_BANKROLL:.0f} | "
          f"Kelly: {config.KELLY_FRACTION} | Min edge: {config.MIN_EDGE_THRESHOLD:.0%} (after fees)")
    print(f"\nLog loss — model: {model_ll:.4f} | market: {market_ll:.4f} "
          f"({'model BEATS market' if model_ll < market_ll else 'market beats model'})")

    seasons_arr = oof.loc[valid, "season"].values
    results = simulate_kalshi_trading(y_true, model_probs, market, seasons=seasons_arr)

    print(f"\n{'─'*64}\nRESULTS\n{'─'*64}")
    print(f"  Ending bankroll:  ${results['ending_bankroll']:.2f}")
    print(f"  Total profit:     ${results['total_profit']:.2f}")
    print(f"  Total fees paid:  ${results['total_fees']:.2f}")
    print(f"  ROI:              {results['roi']:.1%}")
    print(f"  Bets placed:      {results['n_bets']} / {results['n_games']} "
          f"({results['bet_rate']:.0%} of games)")
    print(f"  Win rate:         {results['win_rate']:.1%}")
    print(f"  Max drawdown:     {results['max_drawdown']:.1%}")
    print(f"{'─'*64}")

    # Per-season breakdown (flat staking, so seasons are comparable)
    tdf = results["trades"]
    act = tdf[tdf["side"] != "PASS"]
    if len(act) > 0 and act["season"].notna().any():
        print(f"\n  {'Season':>6} {'Bets':>6} {'WinRate':>8} {'Wagered':>10} {'Profit':>10} {'ROI':>7}")
        for s, grp in act.groupby("season"):
            roi_s = grp["profit"].sum() / grp["bet_amount"].sum()
            print(f"  {int(s):>6} {len(grp):>6} {grp['won'].astype(bool).mean():>8.1%} "
                  f"${grp['bet_amount'].sum():>9.0f} ${grp['profit'].sum():>9.2f} {roi_s:>7.1%}")
        n_pos = sum(grp['profit'].sum() > 0 for _, grp in act.groupby('season'))
        print(f"\n  Profitable seasons: {n_pos}/{act['season'].nunique()}")

    if odds_file is None:
        print("\n⚠️  DIAGNOSTIC MODE: the market here is a naive logistic model.")
        print("   Beating it means your model adds signal beyond simple public")
        print("   stats. It does NOT demonstrate an edge over real markets —")
        print("   sportsbooks and Kalshi are far sharper than this proxy.")
        print("   Get real closing lines and rerun with --odds-file before")
        print("   drawing ANY conclusion about profitability.")

    summary = {k: (float(v) if isinstance(v, (np.floating, np.integer, int, float)) else v)
               for k, v in results.items() if k not in ["bankroll_history", "trades"]}
    summary["market_source"] = market_label
    summary["model_log_loss"] = float(model_ll)
    summary["market_log_loss"] = float(market_ll)

    out = config.LOGS_DIR / "backtest_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📝 Results saved to {out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--odds-file", type=str, default=None,
                        help="CSV with game_id + market_home_prob columns (real closing lines)")
    args = parser.parse_args()
    print("🏀 NCAAB Kalshi Backtest")
    print("=" * 64)
    run_backtest(odds_file=args.odds_file)

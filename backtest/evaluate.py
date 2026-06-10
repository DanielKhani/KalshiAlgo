"""
Backtesting framework for evaluating model profitability.

Simulates Kalshi trading over historical data using walk-forward
predictions to estimate real-world performance.

Key metrics:
  - ROI (return on investment per bet)
  - Total P&L
  - Sharpe ratio of daily returns
  - Max drawdown
  - Win rate on placed bets
  - Calibration quality

Usage:
    python -m backtest.evaluate
"""
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from model.predict import kelly_criterion


def simulate_kalshi_trading(
    y_true: np.ndarray,
    model_probs: np.ndarray,
    market_prices: np.ndarray = None,
    bankroll: float = None,
    kelly_fraction: float = None,
    min_edge: float = None,
) -> dict:
    """
    Simulate Kalshi trading over a sequence of games.
    
    Args:
        y_true: Actual outcomes (1 = home win, 0 = away win)
        model_probs: Model's predicted P(home win)
        market_prices: Kalshi market prices for home team
                      (if None, generates synthetic market prices)
        bankroll: Starting bankroll
        kelly_fraction: Kelly fraction to use
        min_edge: Minimum edge threshold to place a bet
    
    Returns:
        Dict with full trading simulation results
    """
    if bankroll is None:
        bankroll = config.STARTING_BANKROLL
    if kelly_fraction is None:
        kelly_fraction = config.KELLY_FRACTION
    if min_edge is None:
        min_edge = config.MIN_EDGE_THRESHOLD
    
    if market_prices is None:
        # Simulate market prices as noisy version of true probabilities
        # In reality, you'd use historical Kalshi/odds data
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, len(y_true))
        market_prices = np.clip(
            y_true.astype(float) * 0.3 + 0.35 + noise,  # centered around true prob
            0.1, 0.9
        )
    
    current_bankroll = bankroll
    peak_bankroll = bankroll
    
    trades = []
    bankroll_history = [bankroll]
    
    for i in range(len(y_true)):
        model_prob = model_probs[i]
        market_price = market_prices[i]
        outcome = y_true[i]
        
        # Check both sides for value
        home_edge = model_prob - market_price
        away_edge = (1 - model_prob) - (1 - market_price)
        
        trade = {
            "game_idx": i,
            "model_prob": model_prob,
            "market_price": market_price,
            "outcome": outcome,
        }
        
        if home_edge > min_edge:
            # Bet on home team
            bet_frac = kelly_criterion(model_prob, market_price, kelly_fraction)
            bet_amount = min(current_bankroll * bet_frac, current_bankroll * config.MAX_BET_FRACTION)
            
            if bet_amount > 0.50:  # minimum bet size
                # Buy home YES contract at market_price
                contracts = bet_amount / market_price
                
                if outcome == 1:  # home wins
                    profit = contracts * (1 - market_price)  # payout - cost
                else:
                    profit = -bet_amount  # lose entire bet
                
                current_bankroll += profit
                peak_bankroll = max(peak_bankroll, current_bankroll)
                
                trade.update({
                    "side": "HOME",
                    "edge": home_edge,
                    "bet_amount": bet_amount,
                    "profit": profit,
                    "won": outcome == 1,
                    "bankroll_after": current_bankroll,
                })
            else:
                trade.update({"side": "PASS", "edge": home_edge, "bet_amount": 0, "profit": 0, "won": None, "bankroll_after": current_bankroll})
        
        elif away_edge > min_edge:
            # Bet on away team
            away_market = 1 - market_price
            bet_frac = kelly_criterion(1 - model_prob, away_market, kelly_fraction)
            bet_amount = min(current_bankroll * bet_frac, current_bankroll * config.MAX_BET_FRACTION)
            
            if bet_amount > 0.50:
                contracts = bet_amount / away_market
                
                if outcome == 0:  # away wins
                    profit = contracts * (1 - away_market)
                else:
                    profit = -bet_amount
                
                current_bankroll += profit
                peak_bankroll = max(peak_bankroll, current_bankroll)
                
                trade.update({
                    "side": "AWAY",
                    "edge": away_edge,
                    "bet_amount": bet_amount,
                    "profit": profit,
                    "won": outcome == 0,
                    "bankroll_after": current_bankroll,
                })
            else:
                trade.update({"side": "PASS", "edge": away_edge, "bet_amount": 0, "profit": 0, "won": None, "bankroll_after": current_bankroll})
        else:
            trade.update({"side": "PASS", "edge": max(home_edge, away_edge), "bet_amount": 0, "profit": 0, "won": None, "bankroll_after": current_bankroll})
        
        trades.append(trade)
        bankroll_history.append(current_bankroll)
        
        # Stop if bankrupt
        if current_bankroll < 1.0:
            print(f"💀 Bankrupt at game {i}!")
            break
    
    trades_df = pd.DataFrame(trades)
    active_trades = trades_df[trades_df["side"] != "PASS"]
    
    # Compute summary stats
    if len(active_trades) > 0:
        total_wagered = active_trades["bet_amount"].sum()
        total_profit = active_trades["profit"].sum()
        win_rate = active_trades["won"].mean()
        roi = total_profit / total_wagered if total_wagered > 0 else 0
        avg_edge = active_trades["edge"].mean()
        
        # Max drawdown
        bh = np.array(bankroll_history)
        running_max = np.maximum.accumulate(bh)
        drawdowns = (bh - running_max) / running_max
        max_drawdown = drawdowns.min()
    else:
        total_wagered = total_profit = win_rate = roi = avg_edge = 0
        max_drawdown = 0
    
    results = {
        "starting_bankroll": bankroll,
        "ending_bankroll": current_bankroll,
        "total_profit": total_profit,
        "total_wagered": total_wagered,
        "roi": roi,
        "n_games": len(y_true),
        "n_bets": len(active_trades),
        "bet_rate": len(active_trades) / len(y_true) if len(y_true) > 0 else 0,
        "win_rate": win_rate,
        "avg_edge": avg_edge,
        "max_drawdown": max_drawdown,
        "peak_bankroll": peak_bankroll,
        "bankroll_history": bankroll_history,
        "trades": trades_df,
    }
    
    return results


def run_backtest(cv_results: dict = None) -> dict:
    """
    Run a full backtest using walk-forward CV predictions.
    
    If cv_results not provided, runs the full pipeline.
    """
    if cv_results is None:
        # Run full pipeline
        from data.fetch_team_stats import load_historical_dataset
        from features.engineer import build_features
        from model.train import walk_forward_cv
        
        print("📊 Loading data...")
        df = load_historical_dataset()
        
        print("🔧 Building features...")
        X, y = build_features(df)
        
        print("🏋️ Running walk-forward CV...")
        cv_results = walk_forward_cv(X, y)
    
    y_true = cv_results["all_val_true"]
    model_probs = cv_results["all_val_preds"]
    
    print(f"\n{'='*60}")
    print(f"BACKTEST SIMULATION")
    print(f"{'='*60}")
    print(f"Games: {len(y_true)}")
    print(f"Starting bankroll: ${config.STARTING_BANKROLL:.2f}")
    print(f"Kelly fraction: {config.KELLY_FRACTION}")
    print(f"Min edge: {config.MIN_EDGE_THRESHOLD:.0%}")
    
    results = simulate_kalshi_trading(y_true, model_probs)
    
    print(f"\n{'─'*60}")
    print(f"RESULTS")
    print(f"{'─'*60}")
    print(f"  Ending bankroll:  ${results['ending_bankroll']:.2f}")
    print(f"  Total profit:     ${results['total_profit']:.2f}")
    print(f"  ROI:              {results['roi']:.1%}")
    print(f"  Bets placed:      {results['n_bets']} / {results['n_games']} ({results['bet_rate']:.0%} of games)")
    print(f"  Win rate:         {results['win_rate']:.1%}")
    print(f"  Avg edge:         {results['avg_edge']:.1%}")
    print(f"  Max drawdown:     {results['max_drawdown']:.1%}")
    print(f"  Peak bankroll:    ${results['peak_bankroll']:.2f}")
    print(f"{'─'*60}")
    
    # Profitability assessment
    if results["roi"] > 0.03:
        print(f"\n✅ Model shows positive ROI ({results['roi']:.1%})")
        print(f"   This is promising but backtest ≠ live performance.")
        print(f"   Paper trade for 2-3 weeks before using real money.")
    elif results["roi"] > 0:
        print(f"\n⚠️  Model shows marginal ROI ({results['roi']:.1%})")
        print(f"   Edge is small — fees and variance could eat this up.")
        print(f"   Consider improving features before going live.")
    else:
        print(f"\n❌ Model shows negative ROI ({results['roi']:.1%})")
        print(f"   Don't trade with real money. Improve the model first.")
    
    # Save backtest results
    results_summary = {k: v for k, v in results.items() 
                       if k not in ["bankroll_history", "trades"]}
    results_summary = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in results_summary.items()}
    
    output_path = config.LOGS_DIR / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n📝 Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    print("🏀 NCAAB Kalshi Backtest")
    print("=" * 60)
    results = run_backtest()

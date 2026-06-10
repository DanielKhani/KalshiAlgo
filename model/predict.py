"""
Generate predictions and Kalshi trade recommendations using the TRAINED model.

Pipeline:
  1. Fetch today's odds from The Odds API
  2. For each game, look up both teams' recent stats from historical data
  3. Engineer features using the same pipeline as training
  4. Run through the calibrated LightGBM model
  5. Compare model probability to Kalshi market price
  6. Output Kelly-sized trade recommendations

Usage:
    python -m model.predict
    python -m model.predict --bankroll 500
    python -m model.predict --dry-run    # no API call, uses synthetic odds
"""
import pandas as pd
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.fetch_odds import fetch_current_odds, generate_synthetic_odds
from data.fetch_team_stats import load_historical_dataset
from features.engineer import compute_team_season_stats, compute_rest_days, compute_strength_of_schedule, get_feature_columns


def load_model():
    """Load the trained calibrated model and artifacts."""
    model_path = config.MODEL_DIR / "lgbm_model.pkl"
    artifacts_path = config.MODEL_DIR / "artifacts.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run `python -m model.train` first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(artifacts_path, "r") as f:
        artifacts = json.load(f)

    return model, artifacts


def get_team_features(
    team_name: str,
    team_stats: dict,
    current_season_df: pd.DataFrame,
    label: str,
) -> dict:
    """
    Look up the most recent rolling features for a team.

    Args:
        team_name: Full team name (e.g., "Duke Blue Devils")
        team_stats: Pre-computed team stats dict from compute_team_season_stats
        current_season_df: DataFrame of current season games
        label: "home" or "away" — used as feature prefix

    Returns:
        Dict of features with proper prefixed column names
    """
    features = {}

    if team_name not in team_stats:
        return features

    team_data = team_stats[team_name]
    if len(team_data) == 0:
        return features

    latest = team_data.iloc[-1]

    for window in config.ROLLING_WINDOWS:
        wp = f"roll_{window}"
        features[f"{label}_{wp}_pts_for"] = latest.get(f"{wp}_pts_for", np.nan)
        features[f"{label}_{wp}_pts_against"] = latest.get(f"{wp}_pts_against", np.nan)
        features[f"{label}_{wp}_margin"] = latest.get(f"{wp}_margin", np.nan)
        features[f"{label}_{wp}_win_pct"] = latest.get(f"{wp}_win_pct", np.nan)
        features[f"{label}_{wp}_pace"] = latest.get(f"{wp}_pace", np.nan)

    features[f"{label}_season_margin"] = latest.get("season_margin", np.nan)
    features[f"{label}_season_win_pct"] = latest.get("season_win_pct", np.nan)
    features[f"{label}_off_efficiency"] = latest.get("off_efficiency", np.nan)
    features[f"{label}_def_efficiency"] = latest.get("def_efficiency", np.nan)
    features[f"{label}_net_efficiency"] = latest.get("net_efficiency", np.nan)
    features[f"{label}_games_played"] = latest.get("games_played", np.nan)

    # v2 features
    features[f"{label}_win_streak"] = latest.get("win_streak", 0)
    features[f"{label}_margin_volatility"] = latest.get("margin_volatility", np.nan)
    features[f"{label}_scoring_consistency"] = latest.get("scoring_consistency", np.nan)
    features[f"{label}_blowout_rate"] = latest.get("blowout_rate", np.nan)
    features[f"{label}_close_game_rate"] = latest.get("close_game_rate", np.nan)

    return features


def find_team_name(search: str, known_teams: list[str]) -> str | None:
    """
    Fuzzy match a team name from odds data to our historical team names.

    The Odds API uses names like "Duke Blue Devils" which should match ESPN's
    naming. But sometimes there are slight differences.
    """
    # Exact match first
    if search in known_teams:
        return search

    # Case-insensitive match
    search_lower = search.lower()
    for t in known_teams:
        if t.lower() == search_lower:
            return t

    # Substring match — check if the search term contains the team name or vice versa
    for t in known_teams:
        if search_lower in t.lower() or t.lower() in search_lower:
            return t

    # Match by significant words (skip common words)
    skip_words = {"the", "university", "of", "state", "college"}
    search_words = {w.lower() for w in search.split() if w.lower() not in skip_words}
    best_match = None
    best_overlap = 0

    for t in known_teams:
        t_words = {w.lower() for w in t.split() if w.lower() not in skip_words}
        overlap = len(search_words & t_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = t

    if best_overlap >= 2:
        return best_match

    return None


def kelly_criterion(model_prob: float, market_price: float, fraction: float = None) -> float:
    """
    Calculate Kelly Criterion bet size for Kalshi contracts.

    You pay `market_price` for a contract, receive $1.00 if correct.
    Kelly fraction = (p - market_price) / (1 - market_price)
    """
    if fraction is None:
        fraction = config.KELLY_FRACTION

    if market_price <= 0 or market_price >= 1:
        return 0.0

    edge = model_prob - market_price
    if edge <= 0:
        return 0.0

    kelly = edge / (1 - market_price)
    fractional_kelly = kelly * fraction

    return min(fractional_kelly, config.MAX_BET_FRACTION)


def aggregate_odds_by_game(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw odds data into one row per game with consensus market prices.

    The Odds API returns multiple bookmaker lines per game.
    We average across bookmakers to get a consensus implied probability.
    """
    # If already aggregated (synthetic data), return as-is
    if "home_moneyline_prob" in odds_df.columns:
        return odds_df

    if "market" not in odds_df.columns or len(odds_df) == 0:
        return odds_df

    # Filter to moneyline (h2h) market
    h2h = odds_df[odds_df["market"] == "h2h"].copy()
    if len(h2h) == 0:
        return odds_df

    # Convert American odds to implied probability
    def american_to_prob(odds):
        try:
            odds = float(odds)
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        except (ValueError, TypeError):
            return np.nan

    h2h["implied_prob"] = h2h["price"].apply(american_to_prob)

    # Group by game and average across bookmakers
    games = []
    for game_id in h2h["odds_game_id"].unique():
        game_rows = h2h[h2h["odds_game_id"] == game_id]

        home_team = game_rows["home_team"].iloc[0]
        away_team = game_rows["away_team"].iloc[0]
        commence = game_rows["commence_time"].iloc[0]

        home_probs = game_rows[game_rows["outcome_name"] == home_team]["implied_prob"]
        away_probs = game_rows[game_rows["outcome_name"] == away_team]["implied_prob"]

        if len(home_probs) == 0 or len(away_probs) == 0:
            continue

        home_prob = home_probs.mean()
        away_prob = away_probs.mean()

        # Normalize to remove vig (probabilities should sum to 1)
        total = home_prob + away_prob
        if total > 0:
            home_prob /= total
            away_prob /= total

        games.append({
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence,
            "home_moneyline_prob": round(home_prob, 4),
            "away_moneyline_prob": round(away_prob, 4),
            "kalshi_home_price": round(home_prob, 3),
            "kalshi_away_price": round(away_prob, 3),
            "n_bookmakers": len(home_probs),
        })

    return pd.DataFrame(games)


def generate_predictions(
    odds_df: pd.DataFrame = None,
    bankroll: float = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Generate real predictions using the trained model.

    1. Load trained model
    2. Load historical data and compute current team stats
    3. Fetch today's odds
    4. For each game: engineer features → predict → compare to market → size bet
    """
    if bankroll is None:
        bankroll = config.STARTING_BANKROLL

    # Load model
    print("🤖 Loading trained model...")
    model, artifacts = load_model()
    feature_cols = artifacts["feature_cols"]
    train_medians = np.array(artifacts["train_medians"])

    # Load historical data and compute team stats
    print("📊 Loading team stats...")
    hist_df = load_historical_dataset()

    # Use only current season for team stats lookup
    current_season = config.CURRENT_SEASON
    current_df = hist_df[hist_df["season"] == current_season].copy()

    if len(current_df) == 0:
        print(f"❌ No games found for current season {current_season}")
        return pd.DataFrame()

    print(f"   {len(current_df)} games in current season")

    # Compute team rolling stats for current season
    print("🔧 Computing current team stats...")
    team_stats = compute_team_season_stats(current_df)
    known_teams = list(team_stats.keys())
    print(f"   Stats computed for {len(known_teams)} teams")

    # Compute strength of schedule
    print("📏 Computing strength of schedule...")
    sos = compute_strength_of_schedule(current_df, team_stats)
    print(f"   SOS computed for {len(sos)} teams")

    # Compute rest days
    current_df = compute_rest_days(current_df)
    last_game_dates = {}
    for _, row in current_df.iterrows():
        last_game_dates[row["home_team"]] = pd.to_datetime(row["date"])
        last_game_dates[row["away_team"]] = pd.to_datetime(row["date"])

    # Fetch odds
    if odds_df is None:
        if dry_run:
            print("🧪 Dry run — using synthetic odds")
            odds_df = generate_synthetic_odds()
        else:
            odds_df = fetch_current_odds()

    # Aggregate odds to one row per game
    odds_df = aggregate_odds_by_game(odds_df)

    if len(odds_df) == 0:
        print("No games found in odds data.")
        return pd.DataFrame()

    print(f"\n📊 Analyzing {len(odds_df)} games...")
    print(f"💰 Current bankroll: ${bankroll:.2f}")
    print(f"📏 Kelly fraction: {config.KELLY_FRACTION}")
    print(f"🎯 Min edge threshold: {config.MIN_EDGE_THRESHOLD:.0%}")
    print()

    recommendations = []
    matched = 0
    unmatched = 0

    for _, game in odds_df.iterrows():
        home_odds = game["home_team"]
        away_odds = game["away_team"]
        kalshi_home = game.get("kalshi_home_price", 0.5)
        kalshi_away = game.get("kalshi_away_price", 0.5)

        # Match team names from odds to our historical data
        home_matched = find_team_name(home_odds, known_teams)
        away_matched = find_team_name(away_odds, known_teams)

        if not home_matched or not away_matched:
            unmatched += 1
            continue

        matched += 1

        # Get team features
        home_features = get_team_features(home_matched, team_stats, current_df, "home")
        away_features = get_team_features(away_matched, team_stats, current_df, "away")

        if not home_features or not away_features:
            continue

        # Build full feature dict
        features = {**home_features, **away_features}

        # SOS lookup
        for team, label in [(home_matched, "home"), (away_matched, "away")]:
            if team in sos:
                team_sos = sos[team]
                if team_sos:
                    latest_date = max(team_sos.keys())
                    features[f"{label}_sos"] = team_sos[latest_date]

        # Matchup differentials
        for window in config.ROLLING_WINDOWS:
            wp = f"roll_{window}"
            h_m = features.get(f"home_{wp}_margin")
            a_m = features.get(f"away_{wp}_margin")
            if h_m is not None and a_m is not None:
                features[f"diff_{wp}_margin"] = h_m - a_m

            h_s = features.get(f"home_{wp}_pts_for")
            a_s = features.get(f"away_{wp}_pts_for")
            if h_s is not None and a_s is not None:
                features[f"diff_{wp}_scoring"] = h_s - a_s

        h_net = features.get("home_net_efficiency")
        a_net = features.get("away_net_efficiency")
        if h_net is not None and a_net is not None:
            features["diff_net_efficiency"] = h_net - a_net

        h_wp = features.get("home_season_win_pct")
        a_wp = features.get("away_season_win_pct")
        if h_wp is not None and a_wp is not None:
            features["diff_season_win_pct"] = h_wp - a_wp

        # NEW v2 differentials
        h_sm = features.get("home_season_margin")
        a_sm = features.get("away_season_margin")
        if h_sm is not None and a_sm is not None:
            features["diff_season_margin"] = h_sm - a_sm

        h_sos = features.get("home_sos")
        a_sos = features.get("away_sos")
        if h_sos is not None and a_sos is not None:
            features["diff_sos"] = h_sos - a_sos

        h_streak = features.get("home_win_streak", 0)
        a_streak = features.get("away_win_streak", 0)
        features["diff_win_streak"] = (h_streak or 0) - (a_streak or 0)

        h_blow = features.get("home_blowout_rate")
        a_blow = features.get("away_blowout_rate")
        if h_blow is not None and a_blow is not None:
            features["diff_blowout_rate"] = h_blow - a_blow

        # SOS-adjusted margin
        if h_sm is not None and h_sos is not None:
            features["home_adj_margin"] = h_sm - (h_sos or 0)
        if a_sm is not None and a_sos is not None:
            features["away_adj_margin"] = a_sm - (a_sos or 0)
        h_adj = features.get("home_adj_margin")
        a_adj = features.get("away_adj_margin")
        if h_adj is not None and a_adj is not None:
            features["diff_adj_margin"] = h_adj - a_adj

        # Rest days
        today = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        home_rest = (today - last_game_dates.get(home_matched, today - pd.Timedelta(days=3))).days
        away_rest = (today - last_game_dates.get(away_matched, today - pd.Timedelta(days=3))).days
        features["home_rest_days"] = min(home_rest, 14)
        features["away_rest_days"] = min(away_rest, 14)
        features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]

        # Context features
        # Detect conference game by checking if these teams played a conference game before
        is_conf = 0
        if "conference_competition" in current_df.columns:
            matchups = current_df[
                ((current_df["home_team"] == home_matched) & (current_df["away_team"] == away_matched)) |
                ((current_df["home_team"] == away_matched) & (current_df["away_team"] == home_matched))
            ]
            if len(matchups) > 0 and "conference_competition" in matchups.columns:
                is_conf = int(matchups["conference_competition"].max())
        features["is_conference_game"] = is_conf
        features["neutral_site"] = 0

        # Build feature vector in correct column order
        feature_vector = []
        for col in feature_cols:
            val = features.get(col, np.nan)
            feature_vector.append(val if val is not None else np.nan)

        feature_vector = np.array(feature_vector, dtype=float).reshape(1, -1)

        # Fill NaN with training medians
        for col_idx in range(feature_vector.shape[1]):
            if np.isnan(feature_vector[0, col_idx]):
                if col_idx < len(train_medians):
                    feature_vector[0, col_idx] = train_medians[col_idx]

        # Predict
        model_home_prob = float(model.predict(feature_vector)[0])
        model_away_prob = 1 - model_home_prob

        # Evaluate edge on both sides
        home_edge = model_home_prob - kalshi_home
        away_edge = model_away_prob - kalshi_away

        # Kelly sizing
        home_kelly = kelly_criterion(model_home_prob, kalshi_home)
        away_kelly = kelly_criterion(model_away_prob, kalshi_away)

        # Determine best trade
        if home_edge > config.MIN_EDGE_THRESHOLD and home_kelly > 0:
            bet_side = "HOME"
            edge = home_edge
            kelly_frac = home_kelly
            model_prob = model_home_prob
            market_price = kalshi_home
            team = home_odds
        elif away_edge > config.MIN_EDGE_THRESHOLD and away_kelly > 0:
            bet_side = "AWAY"
            edge = away_edge
            kelly_frac = away_kelly
            model_prob = model_away_prob
            market_price = kalshi_away
            team = away_odds
        else:
            bet_side = "PASS"
            edge = max(home_edge, away_edge)
            kelly_frac = 0
            model_prob = model_home_prob if home_edge > away_edge else model_away_prob
            market_price = kalshi_home if home_edge > away_edge else kalshi_away
            team = home_odds if home_edge > away_edge else away_odds

        bet_amount = round(bankroll * kelly_frac, 2) if kelly_frac > 0 else 0
        potential_profit = round(
            bet_amount * (1 - market_price) / market_price, 2
        ) if bet_amount > 0 and market_price > 0 else 0

        recommendations.append({
            "home_team": home_odds,
            "away_team": away_odds,
            "model_home_prob": round(model_home_prob, 3),
            "kalshi_home_price": round(kalshi_home, 3),
            "model_away_prob": round(model_away_prob, 3),
            "kalshi_away_price": round(kalshi_away, 3),
            "best_edge": round(edge, 3),
            "recommendation": bet_side,
            "bet_team": team if bet_side != "PASS" else "",
            "kelly_fraction": round(kelly_frac, 4),
            "bet_amount": bet_amount,
            "potential_profit": potential_profit,
        })

    recs_df = pd.DataFrame(recommendations)

    if len(recs_df) == 0:
        print("No games could be analyzed (team matching failed).")
        print(f"   Matched: {matched}, Unmatched: {unmatched}")
        return pd.DataFrame()

    # Display results
    print(f"   Teams matched: {matched}, unmatched: {unmatched}")
    print()
    print(f"{'─'*100}")
    print(f"{'Game':<45} {'Model':>6} {'Mkt':>5} {'Rec':>5} {'Edge':>6} {'Bet':>8} {'Profit':>8}")
    print(f"{'─'*100}")

    for _, r in recs_df.iterrows():
        matchup = f"{r['away_team'][:20]} @ {r['home_team'][:20]}"
        model_p = f"{r['model_home_prob']:.0%}"
        mkt_p = f"{r['kalshi_home_price']:.0%}"
        rec = r["recommendation"]
        edge = f"{r['best_edge']:+.1%}"
        bet = f"${r['bet_amount']:.2f}" if r["bet_amount"] > 0 else "—"
        profit = f"${r['potential_profit']:.2f}" if r["potential_profit"] > 0 else "—"

        indicator = "🟢" if rec != "PASS" else "⚪"
        print(f"{indicator} {matchup:<43} {model_p:>6} {mkt_p:>5} {rec:>5} {edge:>6} {bet:>8} {profit:>8}")

    # Summary
    active = recs_df[recs_df["recommendation"] != "PASS"]
    total_wagered = active["bet_amount"].sum()
    total_potential = active["potential_profit"].sum()

    print(f"{'─'*100}")
    print(f"\n📋 Summary:")
    print(f"   Games analyzed: {len(recs_df)}")
    print(f"   Bets recommended: {len(active)}")
    print(f"   Total wagered: ${total_wagered:.2f}")
    print(f"   Total potential profit: ${total_potential:.2f}")
    if bankroll > 0:
        print(f"   Bankroll at risk: {total_wagered / bankroll:.1%}")

    # Save predictions log
    log_path = config.LOGS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    recs_df.to_csv(log_path, index=False)
    print(f"\n📝 Predictions logged to {log_path}")

    return recs_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NCAAB predictions")
    parser.add_argument("--bankroll", type=float, default=None, help="Current bankroll")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic odds")
    args = parser.parse_args()

    print("🏀 NCAAB Kalshi Predictions")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    recs = generate_predictions(bankroll=args.bankroll, dry_run=args.dry_run)
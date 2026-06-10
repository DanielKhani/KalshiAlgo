"""
Feature engineering pipeline for NCAAB prediction model (v2).

Improvements over v1:
  - Strength of schedule (SOS) — avg opponent margin
  - Adjusted margin (margin × opponent quality)
  - Win streak / momentum features
  - Scoring volatility (consistency indicator)
  - Season stage (early vs late season)
  - Absolute team quality features (not just differentials)
  - Better efficiency calculation

Usage:
    from features.engineer import build_features
    X, y = build_features(games_df)
"""
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def compute_team_season_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute rolling team-level stats up to each game date.
    Returns a dict mapping team -> DataFrame of rolling stats per date.
    """
    df = df.sort_values("date").copy()

    home_records = df[["date", "season", "home_team", "home_score", "away_score", "pace"]].copy()
    home_records.columns = ["date", "season", "team", "pts_for", "pts_against", "pace"]

    away_records = df[["date", "season", "away_team", "away_score", "home_score", "pace"]].copy()
    away_records.columns = ["date", "season", "team", "pts_for", "pts_against", "pace"]

    all_records = pd.concat([home_records, away_records]).sort_values("date")

    team_stats = {}

    for team in all_records["team"].unique():
        team_games = all_records[all_records["team"] == team].copy()

        # Margin per game
        team_games["margin"] = team_games["pts_for"] - team_games["pts_against"]
        team_games["win"] = (team_games["margin"] > 0).astype(float)

        for window in config.ROLLING_WINDOWS:
            prefix = f"roll_{window}"
            team_games[f"{prefix}_pts_for"] = (
                team_games["pts_for"].rolling(window, min_periods=1).mean()
            )
            team_games[f"{prefix}_pts_against"] = (
                team_games["pts_against"].rolling(window, min_periods=1).mean()
            )
            team_games[f"{prefix}_pace"] = (
                team_games["pace"].rolling(window, min_periods=1).mean()
            )
            team_games[f"{prefix}_margin"] = (
                team_games["margin"].rolling(window, min_periods=1).mean()
            )
            team_games[f"{prefix}_win_pct"] = (
                team_games["win"].rolling(window, min_periods=1).mean()
            )

        # Season-level cumulative stats
        team_games["season_pts_for"] = (
            team_games.groupby("season")["pts_for"]
            .expanding().mean().reset_index(level=0, drop=True)
        )
        team_games["season_pts_against"] = (
            team_games.groupby("season")["pts_against"]
            .expanding().mean().reset_index(level=0, drop=True)
        )
        team_games["season_margin"] = team_games["season_pts_for"] - team_games["season_pts_against"]
        team_games["season_win_pct"] = (
            team_games["win"]
            .groupby(team_games["season"]).expanding().mean()
            .reset_index(level=0, drop=True)
        )
        team_games["games_played"] = (
            team_games.groupby("season").cumcount() + 1
        )

        # Offensive and defensive efficiency (pts per ~70 possessions)
        pace_clip = team_games["pace"].clip(lower=50)
        team_games["off_efficiency"] = team_games["season_pts_for"] / pace_clip * 70
        team_games["def_efficiency"] = team_games["season_pts_against"] / pace_clip * 70
        team_games["net_efficiency"] = team_games["off_efficiency"] - team_games["def_efficiency"]

        # ── NEW: Win streak (consecutive wins, negative = consecutive losses) ──
        streaks = []
        current_streak = 0
        for w in team_games["win"]:
            if w == 1:
                current_streak = max(1, current_streak + 1)
            else:
                current_streak = min(-1, current_streak - 1)
            streaks.append(current_streak)
        team_games["win_streak"] = streaks

        # ── NEW: Scoring volatility (std dev of margin over last 10 games) ──
        team_games["margin_volatility"] = (
            team_games["margin"].rolling(10, min_periods=3).std()
        )

        # ── NEW: Scoring consistency (coefficient of variation) ──
        roll_mean = team_games["pts_for"].rolling(10, min_periods=3).mean()
        roll_std = team_games["pts_for"].rolling(10, min_periods=3).std()
        team_games["scoring_consistency"] = roll_std / roll_mean.clip(lower=1)

        # ── NEW: Dominance ratio (blowout wins vs close games) ──
        team_games["is_blowout_win"] = (team_games["margin"] >= 15).astype(float)
        team_games["is_close_game"] = (team_games["margin"].abs() <= 5).astype(float)
        team_games["blowout_rate"] = (
            team_games["is_blowout_win"].rolling(15, min_periods=3).mean()
        )
        team_games["close_game_rate"] = (
            team_games["is_close_game"].rolling(15, min_periods=3).mean()
        )

        team_stats[team] = team_games

    return team_stats


def compute_strength_of_schedule(df: pd.DataFrame, team_stats: dict) -> dict[str, list]:
    """
    Compute running strength of schedule for each team.
    SOS = average season_margin of opponents faced so far.

    Optimized: builds lookup table instead of iterating per-opponent.
    """
    df = df.sort_values("date").copy()

    # Build a quick lookup: team -> latest season_margin at each date
    team_latest_margin = {}
    for team, stats in team_stats.items():
        margins = stats[["date", "season_margin"]].copy()
        team_latest_margin[team] = margins

    # For each team, compute rolling average of opponent margins
    sos_results = {}

    for team in team_stats:
        home_games = df[df["home_team"] == team][["date", "away_team"]].copy()
        home_games.columns = ["date", "opponent"]
        away_games = df[df["away_team"] == team][["date", "home_team"]].copy()
        away_games.columns = ["date", "opponent"]

        all_opps = pd.concat([home_games, away_games]).sort_values("date")

        opp_margins = []
        for _, row in all_opps.iterrows():
            opp = row["opponent"]
            if opp in team_latest_margin:
                opp_data = team_latest_margin[opp]
                prior = opp_data[opp_data["date"] < row["date"]]
                if len(prior) > 0:
                    opp_margins.append(prior.iloc[-1]["season_margin"])
                else:
                    opp_margins.append(0.0)
            else:
                opp_margins.append(0.0)

        # Running average SOS
        sos_values = pd.Series(opp_margins).expanding().mean().tolist()
        sos_results[team] = dict(zip(all_opps["date"].values, sos_values))

    return sos_results


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days of rest for each team entering each game."""
    df = df.sort_values("date").copy()

    last_game = {}
    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        date = pd.to_datetime(row["date"])

        if row["home_team"] in last_game:
            rest = (date - last_game[row["home_team"]]).days
            home_rest.append(min(rest, 14))
        else:
            home_rest.append(7)

        if row["away_team"] in last_game:
            rest = (date - last_game[row["away_team"]]).days
            away_rest.append(min(rest, 14))
        else:
            away_rest.append(7)

        last_game[row["home_team"]] = date
        last_game[row["away_team"]] = date

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]

    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Main feature engineering function (v2).

    New features compared to v1:
      - Strength of schedule (home_sos, away_sos, diff_sos)
      - Win streak / momentum
      - Margin volatility
      - Scoring consistency
      - Blowout rate and close game rate
      - Season stage (games_played as proxy)
      - Absolute season margin (not just differential)
    """
    print("🔧 Engineering features (v2)...")
    df = df.sort_values("date").copy()

    # Step 1: Compute team rolling stats
    print("   Computing team rolling stats...")
    team_stats = compute_team_season_stats(df)

    # Step 2: Compute strength of schedule
    print("   Computing strength of schedule...")
    sos = compute_strength_of_schedule(df, team_stats)

    # Step 3: Compute rest days
    print("   Computing rest days...")
    df = compute_rest_days(df)

    # Step 4: Build feature matrix
    print("   Building feature matrix...")
    feature_rows = []

    for idx, game in df.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        date = game["date"]

        features = {
            "game_id": game["game_id"],
            "date": date,
            "season": game["season"],
            "home_team": home,
            "away_team": away,
        }

        # Get latest team stats BEFORE this game
        for team, label in [(home, "home"), (away, "away")]:
            if team in team_stats:
                prior = team_stats[team][team_stats[team]["date"] < date]
                if len(prior) >= config.MIN_GAMES_PLAYED:
                    latest = prior.iloc[-1]

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

                    # NEW v2 features
                    features[f"{label}_win_streak"] = latest.get("win_streak", 0)
                    features[f"{label}_margin_volatility"] = latest.get("margin_volatility", np.nan)
                    features[f"{label}_scoring_consistency"] = latest.get("scoring_consistency", np.nan)
                    features[f"{label}_blowout_rate"] = latest.get("blowout_rate", np.nan)
                    features[f"{label}_close_game_rate"] = latest.get("close_game_rate", np.nan)

            # SOS lookup
            if team in sos:
                team_sos = sos[team]
                # Find most recent SOS before this date
                prior_dates = [d for d in team_sos if d < date]
                if prior_dates:
                    latest_date = max(prior_dates)
                    features[f"{label}_sos"] = team_sos[latest_date]

        # ── Matchup differentials ──
        for window in config.ROLLING_WINDOWS:
            wp = f"roll_{window}"
            h_margin = features.get(f"home_{wp}_margin")
            a_margin = features.get(f"away_{wp}_margin")
            if h_margin is not None and a_margin is not None:
                features[f"diff_{wp}_margin"] = h_margin - a_margin

            h_eff = features.get(f"home_{wp}_pts_for")
            a_eff = features.get(f"away_{wp}_pts_for")
            if h_eff is not None and a_eff is not None:
                features[f"diff_{wp}_scoring"] = h_eff - a_eff

        # Efficiency differential
        h_net = features.get("home_net_efficiency")
        a_net = features.get("away_net_efficiency")
        if h_net is not None and a_net is not None:
            features["diff_net_efficiency"] = h_net - a_net

        # Win pct differential
        h_wp = features.get("home_season_win_pct")
        a_wp = features.get("away_season_win_pct")
        if h_wp is not None and a_wp is not None:
            features["diff_season_win_pct"] = h_wp - a_wp

        # Season margin differential (key for distinguishing blowouts)
        h_sm = features.get("home_season_margin")
        a_sm = features.get("away_season_margin")
        if h_sm is not None and a_sm is not None:
            features["diff_season_margin"] = h_sm - a_sm

        # NEW: SOS differential
        h_sos = features.get("home_sos")
        a_sos = features.get("away_sos")
        if h_sos is not None and a_sos is not None:
            features["diff_sos"] = h_sos - a_sos

        # NEW: Win streak differential
        h_streak = features.get("home_win_streak", 0)
        a_streak = features.get("away_win_streak", 0)
        features["diff_win_streak"] = (h_streak or 0) - (a_streak or 0)

        # NEW: Blowout rate differential
        h_blow = features.get("home_blowout_rate")
        a_blow = features.get("away_blowout_rate")
        if h_blow is not None and a_blow is not None:
            features["diff_blowout_rate"] = h_blow - a_blow

        # NEW: SOS-adjusted margin (team margin minus SOS)
        # A team with +10 margin against a +5 SOS schedule is less impressive
        # than +10 margin against a -5 SOS schedule
        if h_sm is not None and h_sos is not None:
            features["home_adj_margin"] = h_sm - (h_sos or 0)
        if a_sm is not None and a_sos is not None:
            features["away_adj_margin"] = a_sm - (a_sos or 0)
        h_adj = features.get("home_adj_margin")
        a_adj = features.get("away_adj_margin")
        if h_adj is not None and a_adj is not None:
            features["diff_adj_margin"] = h_adj - a_adj

        # Rest and context features
        features["home_rest_days"] = game["home_rest_days"]
        features["away_rest_days"] = game["away_rest_days"]
        features["rest_advantage"] = game["rest_advantage"]
        features["is_conference_game"] = game.get("is_conference_game", 0)
        features["neutral_site"] = game.get("neutral_site", 0)

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    # Drop rows with too many missing features (early season games)
    feature_cols = get_feature_columns(features_df)
    features_df = features_df.dropna(subset=feature_cols, thresh=len(feature_cols) * 0.5)

    # Align target
    y = df.loc[features_df.index, "home_win"]

    print(f"   ✅ Built {len(features_df)} samples with {len(feature_cols)} features")

    return features_df, y


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to use as model features."""
    exclude = {"game_id", "date", "season", "home_team", "away_team"}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]
    ]


if __name__ == "__main__":
    from data.fetch_team_stats import load_historical_dataset

    print("🔧 Feature Engineering Pipeline (v2)")
    print("=" * 50)

    df = load_historical_dataset()
    X, y = build_features(df)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"\nFeature columns ({len(get_feature_columns(X))}):")
    for c in get_feature_columns(X):
        print(f"   {c}")
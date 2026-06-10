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
        # Use *season-average* pace, not the single most recent game's pace —
        # otherwise one weird game distorts the season-long efficiency numbers.
        season_pace = (
            team_games.groupby("season")["pace"]
            .expanding().mean().reset_index(level=0, drop=True)
        ).clip(lower=50)
        team_games["off_efficiency"] = team_games["season_pts_for"] / season_pace * 70
        team_games["def_efficiency"] = team_games["season_pts_against"] / season_pace * 70
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



STAT_COLS_BASE = ["season_margin", "season_win_pct", "off_efficiency", "def_efficiency",
                  "net_efficiency", "games_played", "win_streak", "margin_volatility",
                  "scoring_consistency", "blowout_rate", "close_game_rate"]

def _team_records(df):
    h = df[["date","season","home_team","home_score","away_score","pace"]].copy()
    h.columns = ["date","season","team","pts_for","pts_against","pace"]
    a = df[["date","season","away_team","away_score","home_score","pace"]].copy()
    a.columns = ["date","season","team","pts_for","pts_against","pace"]
    return pd.concat([h, a]).sort_values("date", kind="stable").reset_index(drop=True)

def _compute_stats(r):
    r = r.copy()
    r["margin"] = r["pts_for"] - r["pts_against"]
    r["win"] = (r["margin"] > 0).astype(float)
    g = r.groupby("team", sort=False)
    for w in config.ROLLING_WINDOWS:
        p = f"roll_{w}"
        for col in ["pts_for","pts_against","pace","margin","win"]:
            name = "win_pct" if col == "win" else col
            r[f"{p}_{name}"] = g[col].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    gs = r.groupby(["team","season"], sort=False)
    r["season_pts_for"] = gs["pts_for"].transform(lambda s: s.expanding().mean())
    r["season_pts_against"] = gs["pts_against"].transform(lambda s: s.expanding().mean())
    r["season_margin"] = r["season_pts_for"] - r["season_pts_against"]
    r["season_win_pct"] = gs["win"].transform(lambda s: s.expanding().mean())
    r["games_played"] = gs.cumcount() + 1
    season_pace = gs["pace"].transform(lambda s: s.expanding().mean()).clip(lower=50)
    r["off_efficiency"] = r["season_pts_for"] / season_pace * 70
    r["def_efficiency"] = r["season_pts_against"] / season_pace * 70
    r["net_efficiency"] = r["off_efficiency"] - r["def_efficiency"]
    def streaks(wins):
        out, cur = [], 0
        for w in wins:
            cur = max(1, cur + 1) if w == 1 else min(-1, cur - 1)
            out.append(cur)
        return pd.Series(out, index=wins.index)
    r["win_streak"] = g["win"].transform(streaks)
    r["margin_volatility"] = g["margin"].transform(lambda s: s.rolling(10, min_periods=3).std())
    rm = g["pts_for"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    rs = g["pts_for"].transform(lambda s: s.rolling(10, min_periods=3).std())
    r["scoring_consistency"] = rs / rm.clip(lower=1)
    r["blowout_rate"] = g["margin"].transform(lambda s: (s >= 15).astype(float).rolling(15, min_periods=3).mean())
    r["close_game_rate"] = g["margin"].transform(lambda s: (s.abs() <= 5).astype(float).rolling(15, min_periods=3).mean())
    return r

def _prev_distinct_date(rec, cols):
    """Per (team,date): value of `cols` at the team's last row of the PREVIOUS
    distinct date. Returns a frame with unique (team,date) keys."""
    last = rec.groupby(["team","date"], sort=False)[cols].last().reset_index()
    shifted = last.groupby("team", sort=False)[cols].shift(1)
    return pd.concat([last[["team","date"]], shifted], axis=1)

def build_features(df):
    df = df.sort_values("date", kind="stable").copy()
    rec = _compute_stats(_team_records(df))

    roll_cols = [f"roll_{w}_{n}" for w in config.ROLLING_WINDOWS
                 for n in ["pts_for","pts_against","margin","win_pct","pace"]]
    stat_cols = roll_cols + STAT_COLS_BASE

    g = rec.groupby("team", sort=False)
    rec["gnum"] = g.cumcount()

    prior = _prev_distinct_date(rec, stat_cols)
    nprior = rec.groupby(["team","date"], sort=False)["gnum"].min().reset_index()
    prior = prior.merge(nprior, on=["team","date"])
    below = prior["gnum"] < config.MIN_GAMES_PLAYED
    prior.loc[below, stat_cols] = np.nan
    prior = prior.set_index(["team","date"])

    # ── SOS ──
    opp_prior = _prev_distinct_date(rec, ["season_margin"]).rename(
        columns={"season_margin": "opp_prior_sm"}).set_index(["team","date"])["opp_prior_sm"]
    hh = df[["date","home_team","away_team"]].copy(); hh.columns=["date","team","opponent"]
    aa = df[["date","away_team","home_team"]].copy(); aa.columns=["date","team","opponent"]
    longg = pd.concat([hh, aa]).sort_values("date", kind="stable").reset_index(drop=True)
    longg["opp_m"] = opp_prior.reindex(
        pd.MultiIndex.from_arrays([longg["opponent"], longg["date"]])).values
    longg["opp_m"] = longg["opp_m"].fillna(0.0)
    longg["sos_incl"] = longg.groupby("team", sort=False)["opp_m"].transform(
        lambda s: s.expanding().mean())
    sos_prior = _prev_distinct_date(longg, ["sos_incl"]).rename(
        columns={"sos_incl": "sos"}).set_index(["team","date"])["sos"]

    # ── Rest days: per-GAME, replicating the original sequential loop exactly
    # (a doubleheader's second game gets rest=0, keyed by game not by date) ──
    n = len(df)
    teams_iv = np.empty(2 * n, dtype=object)
    teams_iv[0::2] = df["home_team"].values
    teams_iv[1::2] = df["away_team"].values
    dates_iv = np.repeat(pd.to_datetime(df["date"]).values, 2)
    longr = pd.DataFrame({"team": teams_iv, "date": dates_iv})
    prevd = longr.groupby("team", sort=False)["date"].shift(1)
    rest_iv = (longr["date"] - prevd).dt.days.clip(upper=14).fillna(7).values

    # ── Assemble per-game ──
    out = df[["game_id","date","season","home_team","away_team"]].copy()
    out["_target_home_win"] = df["home_win"].values
    for label, tcol in [("home","home_team"), ("away","away_team")]:
        pidx = pd.MultiIndex.from_arrays([df[tcol], df["date"]])
        sub = prior.reindex(pidx)
        for c in stat_cols:
            out[f"{label}_{c}"] = sub[c].values
        out[f"{label}_sos"] = sos_prior.reindex(pidx).values
    out["home_rest_days"] = rest_iv[0::2]
    out["away_rest_days"] = rest_iv[1::2]
    out["rest_advantage"] = out["home_rest_days"] - out["away_rest_days"]

    for w in config.ROLLING_WINDOWS:
        p = f"roll_{w}"
        out[f"diff_{p}_margin"] = out[f"home_{p}_margin"] - out[f"away_{p}_margin"]
        out[f"diff_{p}_scoring"] = out[f"home_{p}_pts_for"] - out[f"away_{p}_pts_for"]
    out["diff_net_efficiency"] = out["home_net_efficiency"] - out["away_net_efficiency"]
    out["diff_season_win_pct"] = out["home_season_win_pct"] - out["away_season_win_pct"]
    out["diff_season_margin"] = out["home_season_margin"] - out["away_season_margin"]
    out["diff_sos"] = out["home_sos"] - out["away_sos"]
    out["diff_win_streak"] = out["home_win_streak"].fillna(0) - out["away_win_streak"].fillna(0)
    out["diff_blowout_rate"] = out["home_blowout_rate"] - out["away_blowout_rate"]
    out["home_adj_margin"] = out["home_season_margin"] - out["home_sos"]
    out["away_adj_margin"] = out["away_season_margin"] - out["away_sos"]
    out["diff_adj_margin"] = out["home_adj_margin"] - out["away_adj_margin"]
    out["is_conference_game"] = df.get("is_conference_game", pd.Series(0, index=df.index)).values
    out["neutral_site"] = df.get("neutral_site", pd.Series(0, index=df.index)).values

    fcols = get_feature_columns(out)
    out = out.dropna(subset=fcols, thresh=int(len(fcols) * 0.5))
    y = out.pop("_target_home_win").astype(int)
    return out.reset_index(drop=True), y.reset_index(drop=True)




def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to use as model features."""
    exclude = {"game_id", "date", "season", "home_team", "away_team", "_target_home_win"}
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
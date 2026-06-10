"""
Fetch NCAA Men's Basketball team stats and game results from REAL data sources.

Data sources (in priority order):
  1. ESPN API (free, no key needed) — game-by-game scores, teams, conferences
  2. sportsdataverse-py — ESPN wrapper with cleaner interface
  3. Kaggle datasets — historical team-level season stats
  4. NCAA_Hoops GitHub — pre-cleaned historical CSV results

Usage:
    python -m data.fetch_team_stats
    python -m data.fetch_team_stats --seasons 2024 2025 2026
    python -m data.fetch_team_stats --quick          # last 2 seasons only
    python -m data.fetch_team_stats --current-only   # current season only
    python -m data.fetch_team_stats --clear-cache    # force re-fetch
"""
import pandas as pd
import numpy as np
import requests
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── ESPN API (Source 1 — Primary) ──────────────────────────────────────────
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)
ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams"
)


def fetch_espn_teams() -> pd.DataFrame:
    """Fetch all D1 men's basketball teams from ESPN."""
    print("📡 Fetching team list from ESPN...")
    try:
        resp = requests.get(
            ESPN_TEAMS_URL, params={"limit": 500}, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        teams = []
        for t in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            team = t.get("team", {})
            teams.append({
                "espn_id": team.get("id"),
                "team_name": team.get("displayName", ""),
                "abbreviation": team.get("abbreviation", ""),
                "location": team.get("location", ""),
            })
        df = pd.DataFrame(teams)
        print(f"   Found {len(df)} teams")
        return df
    except Exception as e:
        print(f"   ❌ Error fetching teams: {e}")
        return pd.DataFrame()


def fetch_espn_scoreboard(date_str: str) -> list[dict]:
    """
    Fetch all games for a single date from ESPN scoreboard API.
    date_str format: 'YYYYMMDD'

    Returns list of parsed game dicts.
    """
    params = {
        "dates": date_str,
        "groups": 50,       # All D1 conferences
        "limit": 400,       # Get all games for the day
    }

    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    games = []
    for event in data.get("events", []):
        try:
            competition = event["competitions"][0]

            # Skip games that aren't final
            status = competition.get("status", {}).get("type", {}).get("name", "")
            if status != "STATUS_FINAL":
                continue

            competitors = competition.get("competitors", [])
            if len(competitors) != 2:
                continue

            home = away = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home = c
                else:
                    away = c

            if not home or not away:
                continue

            home_team_data = home.get("team", {})
            away_team_data = away.get("team", {})

            game = {
                "game_id": event.get("id", ""),
                "date": event.get("date", "")[:10],
                "home_team": home_team_data.get("displayName", ""),
                "away_team": away_team_data.get("displayName", ""),
                "home_id": home_team_data.get("id", ""),
                "away_id": away_team_data.get("id", ""),
                "home_abbreviation": home_team_data.get("abbreviation", ""),
                "away_abbreviation": away_team_data.get("abbreviation", ""),
                "home_score": int(home.get("score", 0)),
                "away_score": int(away.get("score", 0)),
                "home_win": int(int(home.get("score", 0)) > int(away.get("score", 0))),
                "neutral_site": int(competition.get("neutralSite", False)),
                "conference_competition": int(
                    competition.get("conferenceCompetition", False)
                ),
            }
            games.append(game)
        except (KeyError, IndexError, ValueError):
            continue

    return games


def fetch_espn_season(season: int, progress: bool = True) -> pd.DataFrame:
    """
    Fetch all games for a full college basketball season from ESPN.

    Season convention: season=2026 means the 2025-26 season.
    Regular season runs ~Nov 4 through mid-March.
    Tournament runs through early April.

    ESPN API is free with no key, but we add small delays to be polite.
    """
    # Season date range
    start_date = datetime(season - 1, 11, 1)
    end_date = min(datetime(season, 4, 15), datetime.now())

    if start_date > datetime.now():
        print(f"   ⚠️  Season {season} hasn't started yet.")
        return pd.DataFrame()

    print(f"📊 Fetching {season - 1}-{str(season)[2:]} season from ESPN "
          f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")

    all_games = []
    total_days = (end_date - start_date).days

    iterator = range(total_days + 1)
    if progress:
        iterator = tqdm(iterator, desc=f"   Season {season}", unit="day")

    for i in iterator:
        day = start_date + timedelta(days=i)
        date_str = day.strftime("%Y%m%d")

        games = fetch_espn_scoreboard(date_str)
        for g in games:
            g["season"] = season
        all_games.extend(games)

        # Be polite to ESPN — small delay between requests
        time.sleep(0.15)

    df = pd.DataFrame(all_games)
    if len(df) > 0:
        print(f"   ✅ {len(df)} games fetched for {season - 1}-{str(season)[2:]}")
    else:
        print(f"   ⚠️  No games found for {season - 1}-{str(season)[2:]}")
    return df


# ── Kaggle Dataset (Source 3 — Supplementary) ─────────────────────────────

def load_kaggle_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Load the Andrew Sundberg College Basketball Dataset from Kaggle.

    Download from:
      https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset

    Place the CSV(s) in data/stored/kaggle/

    This dataset contains team-level season aggregates including:
    - ADJOE (adjusted offensive efficiency)
    - ADJDE (adjusted defensive efficiency)
    - BARTHAG (power rating)
    - EFG_O / EFG_D (effective FG%)
    - TOR / TORD (turnover rate)
    - ORB / DRB (rebound rates)
    - FTR / FTRD (free throw rate)
    - ADJ_T (adjusted tempo)
    """
    kaggle_dir = config.DATA_DIR / "kaggle"

    if filepath:
        path = Path(filepath)
    else:
        if kaggle_dir.exists():
            csvs = list(kaggle_dir.glob("*.csv"))
            if csvs:
                path = csvs[0]
            else:
                print("   ⚠️  No Kaggle CSV found in data/stored/kaggle/. Skipping.")
                return pd.DataFrame()
        else:
            print("   ⚠️  data/stored/kaggle/ directory not found. Skipping Kaggle data.")
            return pd.DataFrame()

    print(f"📦 Loading Kaggle dataset from {path.name}...")
    try:
        df = pd.read_csv(path)
        print(f"   ✅ Loaded {len(df)} team-seasons from Kaggle")
        return df
    except Exception as e:
        print(f"   ❌ Error loading Kaggle data: {e}")
        return pd.DataFrame()


# ── Combine All Sources ───────────────────────────────────────────────────

def build_historical_dataset(
    seasons: list[int] = None,
    use_espn: bool = True,
    use_kaggle: bool = True,
) -> pd.DataFrame:
    """
    Build the full historical dataset by combining all available sources.

    ESPN API is free but takes ~3 min per season to scrape
    (one request per day x ~165 days). Results are cached to disk
    so you only need to scrape each season once.
    """
    if seasons is None:
        seasons = config.TRAINING_SEASONS + [config.CURRENT_SEASON]

    all_games = []

    for season in seasons:
        # Check cache first
        cache_path = config.DATA_DIR / f"espn_games_{season}.csv"

        if cache_path.exists():
            print(f"💾 Loading cached season {season - 1}-{str(season)[2:]}...")
            df = pd.read_csv(cache_path)
            all_games.append(df)
            continue

        if use_espn:
            df = fetch_espn_season(season)
            if len(df) > 0:
                df.to_csv(cache_path, index=False)
                all_games.append(df)
                print(f"   💾 Cached to {cache_path}")
            else:
                print(f"   ⚠️  No data for season {season}.")

    if not all_games:
        print("\n❌ No game data collected. Check your internet connection.")
        return pd.DataFrame()

    full_df = pd.concat(all_games, ignore_index=True)

    # Clean up
    full_df = full_df.drop_duplicates(subset=["game_id"], keep="first")
    full_df = full_df.sort_values("date").reset_index(drop=True)

    # Add derived columns
    full_df["home_margin"] = full_df["home_score"] - full_df["away_score"]
    full_df["total_points"] = full_df["home_score"] + full_df["away_score"]

    # Estimate pace (rough proxy from total points)
    # Real pace = possessions per 40 min. Total points correlates ~0.85 with pace.
    full_df["pace"] = full_df["total_points"] / 2 * (40 / 32)

    # Map conference_competition to is_conference_game for compatibility
    if "conference_competition" in full_df.columns:
        full_df["is_conference_game"] = full_df["conference_competition"]
    else:
        full_df["is_conference_game"] = 0

    # Save combined dataset
    output_path = config.DATA_DIR / "historical_games.csv"
    full_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅ Dataset built: {len(full_df)} games")
    print(f"   Seasons: {full_df['season'].min()} - {full_df['season'].max()}")
    print(f"   Date range: {full_df['date'].min()} to {full_df['date'].max()}")
    print(f"   Unique teams: {full_df['home_team'].nunique()}")
    print(f"   Home win rate: {full_df['home_win'].mean():.1%}")
    print(f"   Avg total points: {full_df['total_points'].mean():.1f}")
    print(f"   Saved to: {output_path}")
    print(f"{'='*60}")

    # Load Kaggle data if available (for feature engineering later)
    if use_kaggle:
        kaggle_df = load_kaggle_dataset()
        if len(kaggle_df) > 0:
            kaggle_path = config.DATA_DIR / "kaggle_stats.csv"
            kaggle_df.to_csv(kaggle_path, index=False)
            print(f"   Kaggle stats saved to {kaggle_path}")

    return full_df


def load_historical_dataset() -> pd.DataFrame:
    """Load the cached historical dataset."""
    path = config.DATA_DIR / "historical_games.csv"
    if not path.exists():
        print("No cached data found. Building dataset...")
        return build_historical_dataset()
    return pd.read_csv(path)


def clear_cache(seasons: list[int] = None):
    """Clear cached season data to force re-fetch."""
    if seasons is None:
        for f in config.DATA_DIR.glob("espn_games_*.csv"):
            f.unlink()
            print(f"🗑️  Deleted {f.name}")
        combined = config.DATA_DIR / "historical_games.csv"
        if combined.exists():
            combined.unlink()
            print(f"🗑️  Deleted historical_games.csv")
    else:
        for season in seasons:
            f = config.DATA_DIR / f"espn_games_{season}.csv"
            if f.exists():
                f.unlink()
                print(f"🗑️  Deleted {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NCAAB game data")
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=None,
        help="Seasons to fetch (e.g., 2024 2025 2026)",
    )
    parser.add_argument(
        "--current-only", action="store_true",
        help="Only fetch the current season",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear cached data before fetching",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: only fetch last 2 seasons",
    )
    args = parser.parse_args()

    print("🏀 NCAAB Data Pipeline (Real Data)")
    print("=" * 60)

    if args.clear_cache:
        clear_cache(args.seasons)

    if args.current_only:
        seasons = [config.CURRENT_SEASON]
    elif args.quick:
        seasons = [config.CURRENT_SEASON - 1, config.CURRENT_SEASON]
    elif args.seasons:
        seasons = args.seasons
    else:
        seasons = None  # Uses config defaults

    df = build_historical_dataset(seasons=seasons)

    if len(df) > 0:
        print(f"\nSample games (most recent):")
        cols = ["date", "home_team", "away_team", "home_score", "away_score", "home_win"]
        print(df[cols].tail(10).to_string(index=False))
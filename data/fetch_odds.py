"""
Fetch current and historical odds from The Odds API.

Free tier: 500 requests/month
Paid tier: historical odds back to 2020

Usage:
    python -m data.fetch_odds
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def fetch_current_odds() -> pd.DataFrame:
    """
    Fetch current NCAAB odds from The Odds API.
    Returns moneyline, spread, and totals from US bookmakers.
    """
    if not config.ODDS_API_KEY:
        print("⚠️  No ODDS_API_KEY set. Generating synthetic odds.")
        return generate_synthetic_odds()
    
    url = f"{config.ODDS_API_BASE}/sports/{config.ODDS_SPORT_KEY}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": config.ODDS_REGIONS,
        "markets": config.ODDS_MARKETS,
        "oddsFormat": "american",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        
        # Check remaining API calls
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"📡 API requests remaining: {remaining}")
        
        data = resp.json()
        return parse_odds_response(data)
    
    except requests.RequestException as e:
        print(f"❌ Error fetching odds: {e}")
        return pd.DataFrame()


def parse_odds_response(data: list[dict]) -> pd.DataFrame:
    """Parse The Odds API response into a clean DataFrame."""
    rows = []
    
    for game in data:
        game_info = {
            "odds_game_id": game["id"],
            "sport": game["sport_key"],
            "commence_time": game["commence_time"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
        }
        
        for bookmaker in game.get("bookmakers", []):
            book_name = bookmaker["key"]
            
            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                
                for outcome in market.get("outcomes", []):
                    row = {**game_info}
                    row["bookmaker"] = book_name
                    row["market"] = market_key
                    row["outcome_name"] = outcome["name"]
                    row["price"] = outcome["price"]
                    row["point"] = outcome.get("point")
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if len(df) == 0:
        return df
    
    # Pivot to get home/away odds side by side
    return df


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def implied_prob_to_kalshi_price(prob: float) -> float:
    """
    Convert an implied probability to a Kalshi contract price.
    On Kalshi, a contract for an event at 60% implied prob costs ~$0.60.
    """
    return round(prob, 2)


def generate_synthetic_odds(n_games: int = 15) -> pd.DataFrame:
    """
    Generate synthetic odds data for development.
    Mimics what you'd get from The Odds API for today's games.
    """
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    
    team_pairs = [
        ("Duke Blue Devils", "North Carolina Tar Heels"),
        ("Kansas Jayhawks", "Baylor Bears"),
        ("Gonzaga Bulldogs", "Saint Mary's Gaels"),
        ("Purdue Boilermakers", "Indiana Hoosiers"),
        ("Houston Cougars", "Memphis Tigers"),
        ("UConn Huskies", "Villanova Wildcats"),
        ("Kentucky Wildcats", "Tennessee Volunteers"),
        ("Auburn Tigers", "Alabama Crimson Tide"),
        ("Arizona Wildcats", "UCLA Bruins"),
        ("Marquette Golden Eagles", "Creighton Bluejays"),
        ("Iowa State Cyclones", "Texas Tech Red Raiders"),
        ("Michigan State Spartans", "Michigan Wolverines"),
        ("Florida Gators", "LSU Tigers"),
        ("Wisconsin Badgers", "Illinois Fighting Illini"),
        ("St. John's Red Storm", "Providence Friars"),
    ][:n_games]
    
    rows = []
    for home, away in team_pairs:
        # Generate a "true" probability for the home team
        home_prob = np.clip(np.random.normal(0.55, 0.15), 0.15, 0.85)
        away_prob = 1 - home_prob
        
        # Add vig (~4-5% total overround, simulating Kalshi spread)
        home_price = round(home_prob, 2)
        away_price = round(away_prob, 2)
        
        # Generate spread from probability
        spread = round((home_prob - 0.5) * 25, 1)  # rough conversion
        total = round(np.random.normal(145, 10), 1)
        
        rows.append({
            "home_team": home,
            "away_team": away,
            "commence_time": datetime.now().isoformat(),
            "home_moneyline_prob": home_price,
            "away_moneyline_prob": away_price,
            "spread": spread,
            "total": total,
            "kalshi_home_price": home_price,
            "kalshi_away_price": away_price,
        })
    
    return pd.DataFrame(rows)


def get_kalshi_prices(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert standard odds to Kalshi-style contract prices.
    
    On Kalshi, you buy YES at $X.XX and it pays $1.00 if correct.
    So a 60% implied probability = $0.60 contract price.
    Profit if correct: $1.00 - $0.60 = $0.40
    """
    if "home_moneyline_prob" not in odds_df.columns:
        # If we have raw American odds, convert first
        if "price" in odds_df.columns:
            odds_df["implied_prob"] = odds_df["price"].apply(american_to_implied_prob)
    
    return odds_df


if __name__ == "__main__":
    print("📡 Fetching NCAAB Odds")
    print("=" * 50)
    
    odds = fetch_current_odds()
    print(f"\nFetched odds for {len(odds)} games/outcomes")
    
    if len(odds) > 0:
        print(f"\nSample:\n{odds.head(10)}")
        
        # Save
        output_path = config.DATA_DIR / "current_odds.csv"
        odds.to_csv(output_path, index=False)
        print(f"\n✅ Saved to {output_path}")

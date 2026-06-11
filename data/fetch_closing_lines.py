"""
Download and parse historical NCAAB closing moneylines from
sportsbookreviewsonline.com (seasons 2007-08 through 2020-21 as .xlsx),
match them to ESPN game_ids, and emit a CSV the backtest can consume:

    python -m data.fetch_closing_lines
    python -m backtest.evaluate --odds-file data/stored/closing_lines.csv

RUN THIS ON YOUR OWN MACHINE — it downloads from sportsbookreviewsonline.com.

SBR format: one row per TEAM (two per game), paired V(isitor)/H(ome) or N/N
for neutral sites. Columns: Date (e.g. 1112 = Nov 12), VH, Team (squished,
e.g. "MichiganSt"), Final, Open, Close, ML (American odds), 2H.

Requires: pip install openpyxl requests
"""
import pandas as pd
import numpy as np
import re
import requests
from io import BytesIO
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

BASE = ("https://sportsbookreviewsonline.com/wp-content/uploads/"
        "sportsbookreviewsonline_com_737/ncaa-basketball-{s}.xlsx")
SEASONS = {  # label -> season (year it ends, matching historical_games.csv)
    "2007-08": 2008, "2008-09": 2009, "2009-10": 2010, "2010-11": 2011,
    "2011-12": 2012, "2012-13": 2013, "2013-14": 2014, "2014-15": 2015,
    "2015-16": 2016, "2016-17": 2017, "2017-18": 2018, "2018-19": 2019,
    "2019-20": 2020, "2020-21": 2021,
}

# SBR squishes names and abbreviates; expand common tokens before matching.
ABBREV = {
    "st": "state", "u": "", "univ": "", "intl": "international",
    "tx": "texas", "fl": "florida", "ga": "georgia", "nc": "northcarolina",
    "sc": "southcarolina", "tn": "tennessee", "va": "virginia",
    "miss": "mississippi", "la": "louisiana", "ill": "illinois",
    "ark": "arkansas", "ala": "alabama", "wash": "washington",
    "mich": "michigan", "wisc": "wisconsin", "minn": "minnesota",
    "okla": "oklahoma", "colo": "colorado", "conn": "connecticut",
}

# Explicit overrides for SBR names the fuzzy matcher can't resolve
# (verified against ESPN displayNames in historical_games.csv).
EXPLICIT_MAP = {
    "CalPolySLO": "Cal Poly Mustangs",
    "CharlestonSou": "Charleston Southern Buccaneers",
    "CollCharleston": "Charleston Cougars",
    "CollOfCharleston": "Charleston Cougars",
    "DixieState": "Utah Tech Trailblazers",     # renamed in 2022
    "EastTennSt.": "East Tennessee State Buccaneers",
    "EastTennState": "East Tennessee State Buccaneers",
    "FairDickinson": "Fairleigh Dickinson Knights",
    "FlaAtlantic": "Florida Atlantic Owls",
    "FlaGulfCoast": "Florida Gulf Coast Eagles",
    "FullertonSt.": "Cal State Fullerton Titans",
    "GeoWashington": "George Washington Revolutionaries",
    "IPFW": "Purdue Fort Wayne Mastodons",       # renamed in 2018
    "MDBaltimoreCo": "UMBC Retrievers",
    "MDEasternShore": "Maryland Eastern Shore Hawks",
    "MoKansasCity": "Kansas City Roos",
    "Mt.St.Mary's": "Mount St. Mary's Mountaineers",
    "NDakotaSt": "North Dakota State Bison",
    "No.Colorado": "Northern Colorado Bears",
    "NoIllinois": "Northern Illinois Huskies",
    "NorthernArz": "Northern Arizona Lumberjacks",
    # Second batch (the first unmatched printout was truncated at 25 names):
    "SCUpstate": "South Carolina Upstate Spartans",
    "SCarUpstate": "South Carolina Upstate Spartans",
    "USC-Upstate": "South Carolina Upstate Spartans",
    "USCUpstate": "South Carolina Upstate Spartans",
    "SDakotaSt": "South Dakota State Jackrabbits",
    "SaintMarysCA": "Saint Mary's Gaels",
    "SanJoseState": "San José State Spartans",     # accented é broke the fuzzy match
    "SoCarolinaSt": "South Carolina State Bulldogs",
    "SoIllinois": "Southern Illinois Salukis",
    "SoMississippi": "Southern Miss Golden Eagles",
    "StephenAustin": "Stephen F. Austin Lumberjacks",
    "TXPanAmerican": "UT Rio Grande Valley Vaqueros",  # renamed 2015
    "TennMartin": "UT Martin Skyhawks",
    "TexSanAntonio": "UTSA Roadrunners",
    "UL-Lafayette": "Louisiana Ragin' Cajuns",
    "ULLafayette": "Louisiana Ragin' Cajuns",
    "UMKC": "Kansas City Roos",
    "UTRioGrandValley": "UT Rio Grande Valley Vaqueros",
    "UtahValleySt": "Utah Valley Wolverines",
    "WesternKy": "Western Kentucky Hilltoppers",
    "IUPUI": "IU Indianapolis Jaguars",             # renamed 2024
    # Non-D1 exhibition opponents — no ESPN equivalent, intentionally absent:
    # CentralMissouriSt, MetroState, PRMayaguez
}


def normalize(name: str) -> str:
    """Lowercase, strip punctuation/spaces: 'Ohio State' -> 'ohiostate'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def sbr_normalize(name: str) -> str:
    """SBR names are CamelCase-squished; split tokens, expand abbreviations."""
    tokens = re.findall(r"[A-Z][a-z]*|\d+", str(name))
    out = []
    for t in tokens:
        tl = t.lower()
        out.append(ABBREV.get(tl, tl))
    return "".join(out)


def american_to_prob(ml) -> float:
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    if ml == 0 or abs(ml) < 100:
        return np.nan
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def build_team_mapping(espn_teams: list[str]) -> dict[str, str]:
    """
    Map normalized ESPN names to display names. Index by both the full
    displayName and the location prefix (everything except the mascot).
    'Michigan State Spartans' indexes as michiganstatespartans,
    michiganstate, michigan-state variants, etc.
    """
    mapping = {}
    for t in espn_teams:
        full = normalize(t)
        mapping[full] = t
        words = t.split()
        for cut in range(len(words) - 1, 0, -1):  # drop mascot words
            loc = normalize("".join(words[:cut]))
            mapping.setdefault(loc, t)
    return mapping


def match_team(sbr_name: str, mapping: dict[str, str]) -> str | None:
    if sbr_name in EXPLICIT_MAP:
        return EXPLICIT_MAP[sbr_name]
    key = sbr_normalize(sbr_name)
    if key in mapping:
        return mapping[key]
    raw = normalize(sbr_name)
    if raw in mapping:
        return mapping[raw]
    # Last resort: unique prefix match
    hits = {v for k, v in mapping.items() if k.startswith(key) or key.startswith(k)}
    return hits.pop() if len(hits) == 1 else None


def parse_season(xls_bytes: bytes, season: int) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(xls_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    need = {"Date", "VH", "Team", "Final", "ML"}
    if not need.issubset(df.columns):
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    games = []
    rows = df.to_dict("records")
    for i in range(0, len(rows) - 1, 2):
        a, b = rows[i], rows[i + 1]
        # Pair must be (V,H) or (N,N); SBR lists visitor first
        vh = (str(a.get("VH")).strip(), str(b.get("VH")).strip())
        if vh not in [("V", "H"), ("N", "N")]:
            continue
        try:
            mmdd = int(a["Date"])
        except (TypeError, ValueError):
            continue
        month, day = mmdd // 100, mmdd % 100
        year = season - 1 if month >= 8 else season
        date = f"{year:04d}-{month:02d}-{day:02d}"

        p_away = american_to_prob(a.get("ML"))
        p_home = american_to_prob(b.get("ML"))
        if np.isnan(p_away) or np.isnan(p_home):
            continue
        total = p_home + p_away  # remove vig
        games.append({
            "date": date, "season": season,
            "sbr_away": a["Team"], "sbr_home": b["Team"],
            "market_home_prob": round(p_home / total, 4),
            "neutral": int(vh == ("N", "N")),
        })
    return pd.DataFrame(games)


def main():
    hist = pd.read_csv(config.DATA_DIR / "historical_games.csv")
    espn_teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
    mapping = build_team_mapping(espn_teams)

    all_lines = []
    for label, season in SEASONS.items():
        url = BASE.format(s=label)
        print(f"📡 {label} ... ", end="", flush=True)
        try:
            r = requests.get(url, timeout=60,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            lines = parse_season(r.content, season)
            print(f"{len(lines)} games")
            all_lines.append(lines)
        except Exception as e:
            print(f"❌ {e}")

    lines = pd.concat(all_lines, ignore_index=True)

    # Match team names
    unique_sbr = pd.unique(pd.concat([lines["sbr_home"], lines["sbr_away"]]))
    name_map = {n: match_team(n, mapping) for n in unique_sbr}
    unmatched = sorted(n for n, v in name_map.items() if v is None)
    print(f"\nTeam matching: {len(unique_sbr) - len(unmatched)}/{len(unique_sbr)} matched")
    if unmatched:
        print(f"   Unmatched (add to ABBREV or ignore): {unmatched}")

    lines["home_team"] = lines["sbr_home"].map(name_map)
    lines["away_team"] = lines["sbr_away"].map(name_map)
    lines = lines.dropna(subset=["home_team", "away_team"])

    # Merge to ESPN game_ids. TWO quirks to handle:
    #   1. ESPN dates are UTC — evening ET games are stamped with the NEXT
    #      day, so each SBR date must also try sbr_date + 1.
    #   2. Neutral-site games may have home/away flipped between sources.
    # Candidates are tried in priority order: same-date straight, next-date
    # straight, same-date flipped, next-date flipped.
    key_cols = ["date", "home_team", "away_team"]
    hist_keyed = hist[["game_id"] + key_cols]

    lines = lines.reset_index(drop=True)
    lines["sbr_row"] = lines.index

    candidates = []
    for prio, (shift, flip) in enumerate(
            [(0, False), (1, False), (0, True), (1, True)]):
        c = lines.copy()
        if shift:
            c["date"] = (pd.to_datetime(c["date"])
                         + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        if flip:
            c[["home_team", "away_team"]] = c[["away_team", "home_team"]].values
            c["market_home_prob"] = 1 - c["market_home_prob"]
        c["prio"] = prio
        candidates.append(c)

    merged = (pd.concat(candidates, ignore_index=True)
              .merge(hist_keyed, on=key_cols, how="inner")
              .sort_values("prio"))
    # One ESPN game per SBR row and vice versa, best priority wins
    merged = merged.drop_duplicates(subset="sbr_row")
    merged = merged.drop_duplicates(subset="game_id")
    out = merged[["game_id", "date", "season", "market_home_prob", "neutral"]]
    out_path = config.DATA_DIR / "closing_lines.csv"
    out.to_csv(out_path, index=False)

    n_hist = len(hist[hist["season"].isin(SEASONS.values())])
    print(f"\n✅ {len(out)} games with closing lines matched to ESPN game_ids")
    print(f"   ({len(out)/max(n_hist,1):.0%} of {n_hist} ESPN games in those seasons)")
    print(f"   Saved to {out_path}")
    print(f"\nNext: python -m backtest.evaluate --odds-file {out_path}")


if __name__ == "__main__":
    main()

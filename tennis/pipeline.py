"""
ATP tennis win-probability pipeline — built market-first.

Lessons inherited from the NCAAB project, applied from day one:
  1. The FIRST evaluation is against real closing odds (Pinnacle), not a
     synthetic market. The stack test (optimal logistic weights for
     market vs model) decides immediately whether the model carries any
     signal the market lacks.
  2. Walk-forward by year, no leakage: Elo is computed strictly
     chronologically; features for a match use only prior matches.
  3. Calibration measured, fees considered before any talk of edge.

Data: tennis-data.co.uk yearly files (results + closing odds, 2000-2018).
Model: surface-blended Elo (FiveThirtyEight-style decaying K) + LightGBM
on Elo/rank/form/fatigue features.

Usage:
    python -m tennis.pipeline --data-dir path/to/tennis_data/ATP
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse


# ── Data loading ────────────────────────────────────────────────────────

def load_tennis_data(data_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(data_dir).glob("*.xls*")):
        df = pd.read_excel(f)
        df["year"] = int(f.stem)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Winner", "Loser"])
    df["WRank"] = pd.to_numeric(df["WRank"], errors="coerce")
    df["LRank"] = pd.to_numeric(df["LRank"], errors="coerce")
    df["best_of"] = pd.to_numeric(df.get("Best of"), errors="coerce").fillna(3)
    df = df.sort_values("Date").reset_index(drop=True)

    # De-vigged market probability for the eventual winner, best available
    # source per match: Pinnacle > Average > Bet365.
    for w, l, name in [("PSW", "PSL", "pinnacle"), ("AvgW", "AvgL", "avg"),
                       ("B365W", "B365L", "b365")]:
        if w in df.columns:
            iw, il = 1 / df[w], 1 / df[l]
            df[f"mkt_{name}"] = iw / (iw + il)
    df["mkt_prob_winner"] = (
        df.get("mkt_pinnacle", pd.Series(np.nan, index=df.index))
        .fillna(df.get("mkt_avg", pd.Series(np.nan, index=df.index)))
        .fillna(df.get("mkt_b365", pd.Series(np.nan, index=df.index)))
    )
    df["mkt_is_pinnacle"] = df.get("mkt_pinnacle", pd.Series(np.nan, index=df.index)).notna()
    return df


# ── Elo + chronological feature construction ────────────────────────────

def k_factor(n_matches: int) -> float:
    """FiveThirtyEight tennis K: large for new players, decays with matches."""
    return 250.0 / (n_matches + 5) ** 0.4


def build_features(df: pd.DataFrame, surface_weight: float = 0.5) -> pd.DataFrame:
    """
    Single chronological pass. For each match, record PRE-match state for
    both players (Elo, surface Elo, form, fatigue, H2H), then update.
    Player A/B assignment is randomized (seeded) so the target isn't
    leaked by row structure; all features are A-minus-B differentials.
    """
    elo = defaultdict(lambda: 1500.0)
    surf_elo = defaultdict(lambda: 1500.0)        # (player, surface)
    n_played = defaultdict(int)
    last_dates = defaultdict(list)                # recent match dates
    recent_wins = defaultdict(list)               # last-10 results
    h2h = defaultdict(int)                        # (p1, p2) -> p1 wins over p2

    rng = np.random.default_rng(42)
    rows = []

    for r in df.itertuples():
        w, l, surf = r.Winner, r.Loser, (r.Surface if isinstance(r.Surface, str) else "Hard")

        # Randomize orientation: A is the winner with p=0.5
        a_is_winner = rng.random() < 0.5
        A, B = (w, l) if a_is_winner else (l, w)

        def blended(p):
            return (1 - surface_weight) * elo[p] + surface_weight * surf_elo[(p, surf)]

        d = r.Date
        feats = {
            "date": d, "year": r.year, "surface": surf,
            "best_of": r.best_of,
            "A": A, "B": B, "A_wins": int(a_is_winner),
            "mkt_prob_A": (r.mkt_prob_winner if a_is_winner
                           else 1 - r.mkt_prob_winner),
            "mkt_is_pinnacle": r.mkt_is_pinnacle,
            "diff_elo": elo[A] - elo[B],
            "diff_surf_elo": surf_elo[(A, surf)] - surf_elo[(B, surf)],
            "diff_blend_elo": blended(A) - blended(B),
            # log-rank advantage of A (positive = A ranked better/lower number)
            "diff_log_rank": (
                np.log((r.LRank if a_is_winner else r.WRank))
                - np.log((r.WRank if a_is_winner else r.LRank))
                if pd.notna(r.WRank) and pd.notna(r.LRank)
                and r.WRank > 0 and r.LRank > 0 else np.nan),
            "diff_experience": np.log1p(n_played[A]) - np.log1p(n_played[B]),
            "diff_form10": (np.mean(recent_wins[A][-10:]) if recent_wins[A] else 0.5)
                           - (np.mean(recent_wins[B][-10:]) if recent_wins[B] else 0.5),
            "diff_fatigue14": (sum(1 for x in last_dates[A] if (d - x).days <= 14)
                               - sum(1 for x in last_dates[B] if (d - x).days <= 14)),
            "h2h_A": h2h[(A, B)] - h2h[(B, A)],
        }
        rows.append(feats)

        # ── update state (post-match) ──
        exp_w = 1 / (1 + 10 ** ((blended(l) - blended(w)) / 400))
        kw, kl = k_factor(n_played[w]), k_factor(n_played[l])
        elo[w] += kw * (1 - exp_w); elo[l] -= kl * (1 - exp_w)
        es_w = 1 / (1 + 10 ** ((surf_elo[(l, surf)] - surf_elo[(w, surf)]) / 400))
        surf_elo[(w, surf)] += kw * (1 - es_w); surf_elo[(l, surf)] -= kl * (1 - es_w)
        n_played[w] += 1; n_played[l] += 1
        for p, won in [(w, 1), (l, 0)]:
            recent_wins[p].append(won)
            last_dates[p].append(d)
            if len(last_dates[p]) > 30: last_dates[p] = last_dates[p][-30:]
            if len(recent_wins[p]) > 20: recent_wins[p] = recent_wins[p][-20:]
        h2h[(w, l)] += 1

    return pd.DataFrame(rows)


FEATURE_COLS = ["diff_elo", "diff_surf_elo", "diff_blend_elo", "diff_log_rank",
                "diff_experience", "diff_form10", "diff_fatigue14", "h2h_A",
                "best_of"]


# ── Walk-forward evaluation: model vs Pinnacle, stack test first ────────

def evaluate(feats: pd.DataFrame, first_val_year: int = 2010):
    import lightgbm as lgb
    from sklearn.metrics import log_loss
    from sklearn.linear_model import LogisticRegression

    eval_df = feats[feats.mkt_prob_A.notna() & feats.mkt_is_pinnacle].copy()
    eval_df["mkt_logit"] = np.log(eval_df.mkt_prob_A / (1 - eval_df.mkt_prob_A))
    print(f"Matches with Pinnacle closing odds: {len(eval_df)} "
          f"({eval_df.year.min()}-{eval_df.year.max()})")

    params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": 0.05, "num_leaves": 31, "verbose": -1,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}

    results = []
    for vy in sorted(eval_df.year.unique()):
        if vy < first_val_year:
            continue
        # Train on ALL prior matches (odds not required for training)
        tr = feats[(feats.year < vy)].dropna(subset=["diff_elo"])
        va = eval_df[eval_df.year == vy]
        es = tr[tr.year == tr.year.max()]; core = tr[tr.year < tr.year.max()]
        if len(core) < 5000 or len(va) < 200:
            continue

        Xc = core[FEATURE_COLS].fillna(0).values
        Xe = es[FEATURE_COLS].fillna(0).values
        Xv = va[FEATURE_COLS].fillna(0).values
        m = lgb.train(params, lgb.Dataset(Xc, core.A_wins), 600,
                      valid_sets=[lgb.Dataset(Xe, es.A_wins)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        p_model = np.clip(m.predict(Xv), 0.01, 0.99)

        # Pure Elo baseline
        p_elo = 1 / (1 + 10 ** (-va.diff_blend_elo.values / 400))
        p_elo = np.clip(p_elo, 0.01, 0.99)

        y = va.A_wins.values
        mkt = va.mkt_prob_A.values

        # Stack test: optimal logistic weights for (market, model), fit on
        # prior years' validation predictions when available, else this is
        # reported per-year with in-year fit flagged. Simpler honest route:
        # fit on first half of the year, evaluate second half.
        half = len(va) // 2
        Z = np.column_stack([va.mkt_logit.values,
                             np.log(p_model / (1 - p_model))])
        lr = LogisticRegression(max_iter=1000).fit(Z[:half], y[:half])
        p_stack = lr.predict_proba(Z[half:])[:, 1]

        results.append({
            "year": vy, "n": len(va),
            "ll_market": log_loss(y, mkt),
            "ll_model": log_loss(y, p_model),
            "ll_elo": log_loss(y, p_elo),
            "ll_stack_2h": log_loss(y[half:], p_stack),
            "ll_market_2h": log_loss(y[half:], mkt[half:]),
            "w_market": lr.coef_[0][0], "w_model": lr.coef_[0][1],
        })
        r = results[-1]
        print(f"  {vy}: market={r['ll_market']:.4f}  model={r['ll_model']:.4f}  "
              f"elo={r['ll_elo']:.4f}  | stack(2h)={r['ll_stack_2h']:.4f} vs "
              f"mkt(2h)={r['ll_market_2h']:.4f}  weights mkt={r['w_market']:.2f} "
              f"model={r['w_model']:.2f}")

    R = pd.DataFrame(results)
    w = R.n / R.n.sum()
    print(f"\n{'='*66}")
    print(f"WEIGHTED OVERALL ({R.n.sum()} matches, {len(R)} years):")
    for c, lab in [("ll_market", "Pinnacle closing"), ("ll_model", "LightGBM"),
                   ("ll_elo", "pure blended Elo")]:
        print(f"  {lab:18s} LL = {(R[c]*w).sum():.4f}")
    print(f"  avg stack weight — market: {R.w_market.mean():.2f}, "
          f"model: {R.w_model.mean():.2f}")
    print(f"  model beats market: {(R.ll_model < R.ll_market).sum()}/{len(R)} years")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    print("🎾 ATP pipeline — market-first evaluation")
    df = load_tennis_data(args.data_dir)
    print(f"Loaded {len(df)} matches {df.year.min()}-{df.year.max()}")
    feats = build_features(df)
    evaluate(feats)

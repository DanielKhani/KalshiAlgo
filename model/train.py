"""
LightGBM model training with walk-forward cross-validation.

Walk-forward validation mimics real-world conditions:
  - Train on seasons 1..N
  - Validate on season N+1
  - Slide forward and repeat

Fixes vs v1:
  - Early stopping now uses an INNER split (most recent training season),
    never the validation season. Previously the fold's own validation data
    chose the number of boosting rounds, which leaked information and made
    CV metrics optimistic.
  - The isotonic calibrator is now fit on pooled OUT-OF-FOLD predictions
    instead of a single held-out season, so it (a) sees ~10x more data and
    (b) reflects the same train/predict gap the live model will face.
  - OOF predictions (with season/date metadata) are saved to disk so the
    backtest can consume honest, chronological, calibrated predictions.

Usage:
    python -m model.train
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from pathlib import Path
import json
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.fetch_team_stats import build_historical_dataset, load_historical_dataset
from features.engineer import build_features, get_feature_columns
from model.calibrate import calibrate_probabilities, CalibratedModel


def _fill_nan(X_train, *others):
    """Fill NaN with training-set medians; apply same medians to other sets."""
    medians = np.nanmedian(X_train, axis=0)
    out = []
    for arr in (X_train, *others):
        arr = arr.copy()
        for c in range(arr.shape[1]):
            arr[np.isnan(arr[:, c]), c] = medians[c]
        out.append(arr)
    return (*out, medians)


def _train_one_fold(X_tr, y_tr, X_es, y_es, params_full):
    """Train with early stopping on a held-out EARLY-STOP set (inner split)."""
    params = {k: v for k, v in params_full.items()
              if k not in ["n_estimators", "early_stopping_rounds"]}
    train_data = lgb.Dataset(X_tr, label=y_tr)
    es_data = lgb.Dataset(X_es, label=y_es, reference=train_data)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=params_full["n_estimators"],
        valid_sets=[es_data],
        callbacks=[
            lgb.early_stopping(params_full["early_stopping_rounds"]),
            lgb.log_evaluation(0),
        ],
    )
    return model


def walk_forward_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Walk-forward cross-validation with leak-free early stopping.

    For each validation season:
      1. Take up to WALK_FORWARD_TRAIN_SEASONS prior seasons
      2. Inner split: most recent training season = early-stop set
      3. Train on the rest, early-stop on the inner set
      4. Predict the (untouched) validation season

    Returns per-fold metrics plus chronological OOF predictions.
    """
    feature_cols = get_feature_columns(X)
    seasons = sorted(X["season"].unique())

    results = []
    oof_records = []

    print(f"\n{'='*60}")
    print(f"Walk-Forward Cross-Validation (leak-free early stopping)")
    print(f"{'='*60}")
    print(f"Seasons: {seasons[0]}-{seasons[-1]} | "
          f"Train window: {config.WALK_FORWARD_TRAIN_SEASONS} | "
          f"Features: {len(feature_cols)}")
    print(f"{'='*60}\n")

    for i, val_season in enumerate(seasons):
        if i < config.WALK_FORWARD_TRAIN_SEASONS:
            continue

        window = seasons[max(0, i - config.WALK_FORWARD_TRAIN_SEASONS):i]
        inner_es_season = window[-1]          # early stopping set
        core_train_seasons = window[:-1]      # actual training seasons

        tr_mask = X["season"].isin(core_train_seasons)
        es_mask = X["season"] == inner_es_season
        val_mask = X["season"] == val_season

        X_tr = X.loc[tr_mask, feature_cols].values.astype(float)
        y_tr = y.loc[tr_mask].values
        X_es = X.loc[es_mask, feature_cols].values.astype(float)
        y_es = y.loc[es_mask].values
        X_val = X.loc[val_mask, feature_cols].values.astype(float)
        y_val = y.loc[val_mask].values

        if min(len(X_tr), len(X_es), len(X_val)) == 0:
            continue

        X_tr, X_es, X_val, medians = _fill_nan(X_tr, X_es, X_val)

        model = _train_one_fold(X_tr, y_tr, X_es, y_es, config.LGBM_PARAMS)

        val_probs = model.predict(X_val)
        ll = log_loss(y_val, val_probs)
        acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
        brier = brier_score_loss(y_val, val_probs)

        results.append({
            "val_season": int(val_season),
            "train_seasons": [int(s) for s in core_train_seasons],
            "early_stop_season": int(inner_es_season),
            "n_train": len(X_tr), "n_val": len(X_val),
            "log_loss": ll, "accuracy": acc, "brier_score": brier,
            "best_iteration": model.best_iteration,
        })

        fold_meta = X.loc[val_mask, ["game_id", "date", "season"]].copy()
        # Carry simple public-info features so the backtest can construct a
        # naive "market" proxy without re-running feature engineering
        for simple_col in ["diff_season_margin", "diff_season_win_pct", "diff_net_efficiency"]:
            if simple_col in X.columns:
                fold_meta[simple_col] = X.loc[val_mask, simple_col].values
        fold_meta["y_true"] = y_val
        fold_meta["raw_prob"] = val_probs
        oof_records.append(fold_meta)

        print(f"  Season {val_season}: LogLoss={ll:.4f} | Acc={acc:.3f} | "
              f"Brier={brier:.4f} | rounds={model.best_iteration} | n_val={len(X_val)}")

    oof = pd.concat(oof_records, ignore_index=True).sort_values("date").reset_index(drop=True)

    aggregate = {
        "mean_log_loss": float(np.mean([r["log_loss"] for r in results])),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in results])),
        "mean_brier": float(np.mean([r["brier_score"] for r in results])),
        "overall_log_loss": float(log_loss(oof["y_true"], oof["raw_prob"])),
        "overall_accuracy": float(accuracy_score(oof["y_true"], (oof["raw_prob"] >= 0.5).astype(int))),
        "overall_brier": float(brier_score_loss(oof["y_true"], oof["raw_prob"])),
        "median_best_iteration": int(np.median([r["best_iteration"] for r in results])),
    }

    print(f"\n{'='*60}")
    print(f"Aggregate: LogLoss={aggregate['overall_log_loss']:.4f} | "
          f"Acc={aggregate['overall_accuracy']:.3f} | "
          f"Brier={aggregate['overall_brier']:.4f}")
    print(f"{'='*60}")

    return {"folds": results, "aggregate": aggregate, "oof": oof}


def add_walk_forward_calibration(oof: pd.DataFrame, min_history: int = 3000) -> pd.DataFrame:
    """
    Add a `cal_prob` column to OOF predictions, calibrated WALK-FORWARD:
    each season's predictions are calibrated using an isotonic fit only on
    OOF predictions from PRIOR seasons. The first seasons (insufficient
    history) keep raw probabilities. This is exactly what you could have
    done live, so the backtest stays honest.
    """
    from sklearn.isotonic import IsotonicRegression

    oof = oof.sort_values("date").reset_index(drop=True)
    oof["cal_prob"] = oof["raw_prob"]

    seasons = sorted(oof["season"].unique())
    for s in seasons:
        prior = oof[oof["season"] < s]
        if len(prior) < min_history:
            continue
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        iso.fit(prior["raw_prob"].values, prior["y_true"].values)
        mask = oof["season"] == s
        oof.loc[mask, "cal_prob"] = iso.predict(oof.loc[mask, "raw_prob"].values)

    return oof


def train_final_model(X: pd.DataFrame, y: pd.Series, cv_results: dict) -> CalibratedModel:
    """
    Train the production model on ALL data and calibrate on pooled
    out-of-fold predictions from the walk-forward CV.

    Round count = median best_iteration from CV (no early stopping possible
    when training on everything).
    """
    feature_cols = get_feature_columns(X)
    oof = cv_results["oof"]
    n_rounds = cv_results["aggregate"]["median_best_iteration"]

    X_all = X[feature_cols].values.astype(float)
    y_all = y.values
    X_all, medians = _fill_nan(X_all)[0], np.nanmedian(X[feature_cols].values.astype(float), axis=0)
    # (_fill_nan returns (filled, medians); recomputed for clarity)

    print(f"\n🏋️ Training final model on all {len(X_all)} games ({n_rounds} rounds)...")
    params = {k: v for k, v in config.LGBM_PARAMS.items()
              if k not in ["n_estimators", "early_stopping_rounds"]}
    model = lgb.train(params, lgb.Dataset(X_all, label=y_all), num_boost_round=n_rounds)

    print("📐 Calibrating on pooled out-of-fold predictions "
          f"({len(oof)} games across {oof['season'].nunique()} seasons)...")
    calibrated_model = calibrate_probabilities(
        model, oof["raw_prob"].values, oof["y_true"].values, probs_precomputed=True
    )

    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    print("\n📊 Top 10 Features (by gain):")
    for feat, imp in list(importance.items())[:10]:
        print(f"   {feat}: {imp:.0f}")

    with open(config.MODEL_DIR / "lgbm_model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)

    artifacts = {
        "feature_cols": feature_cols,
        "train_medians": medians.tolist(),
        "n_rounds": n_rounds,
        "cv_aggregate": cv_results["aggregate"],
        "feature_importance": {k: float(v) for k, v in importance.items()},
    }
    with open(config.MODEL_DIR / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    print(f"\n✅ Model + artifacts saved to {config.MODEL_DIR}")
    return calibrated_model


if __name__ == "__main__":
    print("🏀 NCAAB Model Training Pipeline")
    print("=" * 60)

    df = load_historical_dataset()
    X, y = build_features(df)

    cv_results = walk_forward_cv(X, y)
    cv_results["oof"] = add_walk_forward_calibration(cv_results["oof"])

    # Persist OOF predictions for the backtest
    oof_path = config.MODEL_DIR / "oof_predictions.csv"
    cv_results["oof"].to_csv(oof_path, index=False)
    print(f"📝 OOF predictions saved to {oof_path}")

    model = train_final_model(X, y, cv_results)
    print("\n🎉 Training complete!")

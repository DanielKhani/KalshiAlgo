"""
LightGBM model training with walk-forward cross-validation.

Walk-forward validation mimics real-world conditions:
  - Train on seasons 1..N
  - Validate on season N+1
  - Slide forward and repeat

This prevents lookahead bias and gives realistic performance estimates.

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


def walk_forward_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Walk-forward cross-validation.
    
    For each validation season:
      1. Train on all prior seasons (up to WALK_FORWARD_TRAIN_SEASONS)
      2. Validate on the current season
      3. Record metrics
    
    Returns dict with per-fold and aggregate metrics.
    """
    feature_cols = get_feature_columns(X)
    seasons = sorted(X["season"].unique())
    
    results = []
    all_val_preds = []
    all_val_true = []
    
    print(f"\n{'='*60}")
    print(f"Walk-Forward Cross-Validation")
    print(f"{'='*60}")
    print(f"Seasons available: {seasons[0]} - {seasons[-1]}")
    print(f"Train window: {config.WALK_FORWARD_TRAIN_SEASONS} seasons")
    print(f"Features: {len(feature_cols)}")
    print(f"{'='*60}\n")
    
    for i, val_season in enumerate(seasons):
        # Need at least WALK_FORWARD_TRAIN_SEASONS before we can validate
        if i < config.WALK_FORWARD_TRAIN_SEASONS:
            continue
        
        # Training seasons: up to WALK_FORWARD_TRAIN_SEASONS prior seasons
        train_seasons = seasons[max(0, i - config.WALK_FORWARD_TRAIN_SEASONS):i]
        
        # Split
        train_mask = X["season"].isin(train_seasons)
        val_mask = X["season"] == val_season
        
        X_train = X.loc[train_mask, feature_cols].values
        y_train = y.loc[train_mask].values
        X_val = X.loc[val_mask, feature_cols].values
        y_val = y.loc[val_mask].values
        
        if len(X_val) == 0 or len(X_train) == 0:
            continue
        
        # Fill NaN with column medians from training set
        train_medians = np.nanmedian(X_train, axis=0)
        for col_idx in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, col_idx])
            X_train[mask, col_idx] = train_medians[col_idx]
            mask = np.isnan(X_val[:, col_idx])
            X_val[mask, col_idx] = train_medians[col_idx]
        
        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {k: v for k, v in config.LGBM_PARAMS.items() 
                  if k not in ["n_estimators", "early_stopping_rounds"]}
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=config.LGBM_PARAMS["n_estimators"],
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(config.LGBM_PARAMS["early_stopping_rounds"]),
                lgb.log_evaluation(0),  # suppress output
            ],
        )
        
        # Predict probabilities
        val_probs = model.predict(X_val)
        val_preds = (val_probs >= 0.5).astype(int)
        
        # Metrics
        ll = log_loss(y_val, val_probs)
        acc = accuracy_score(y_val, val_preds)
        brier = brier_score_loss(y_val, val_probs)
        
        fold_result = {
            "val_season": val_season,
            "train_seasons": train_seasons,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "log_loss": ll,
            "accuracy": acc,
            "brier_score": brier,
            "best_iteration": model.best_iteration,
        }
        results.append(fold_result)
        
        all_val_preds.extend(val_probs.tolist())
        all_val_true.extend(y_val.tolist())
        
        print(f"  Season {val_season}: "
              f"LogLoss={ll:.4f} | Acc={acc:.3f} | Brier={brier:.4f} | "
              f"n_train={len(X_train)}, n_val={len(X_val)}")
    
    # Aggregate metrics
    all_val_preds = np.array(all_val_preds)
    all_val_true = np.array(all_val_true)
    
    aggregate = {
        "mean_log_loss": np.mean([r["log_loss"] for r in results]),
        "mean_accuracy": np.mean([r["accuracy"] for r in results]),
        "mean_brier": np.mean([r["brier_score"] for r in results]),
        "overall_log_loss": log_loss(all_val_true, all_val_preds),
        "overall_accuracy": accuracy_score(all_val_true, (all_val_preds >= 0.5).astype(int)),
        "overall_brier": brier_score_loss(all_val_true, all_val_preds),
    }
    
    print(f"\n{'='*60}")
    print(f"Aggregate Results:")
    print(f"  Mean LogLoss:  {aggregate['mean_log_loss']:.4f}")
    print(f"  Mean Accuracy: {aggregate['mean_accuracy']:.3f}")
    print(f"  Mean Brier:    {aggregate['mean_brier']:.4f}")
    print(f"  Overall Acc:   {aggregate['overall_accuracy']:.3f}")
    print(f"{'='*60}")
    
    return {
        "folds": results,
        "aggregate": aggregate,
        "all_val_preds": all_val_preds,
        "all_val_true": all_val_true,
    }


def train_final_model(X: pd.DataFrame, y: pd.Series) -> CalibratedModel:
    """
    Train the final production model on all available data,
    with calibration applied.
    
    Uses the most recent season as a calibration holdout.
    """
    feature_cols = get_feature_columns(X)
    seasons = sorted(X["season"].unique())
    
    # Hold out last season for calibration
    cal_season = seasons[-1]
    train_seasons = seasons[:-1]
    
    train_mask = X["season"].isin(train_seasons)
    cal_mask = X["season"] == cal_season
    
    X_train = X.loc[train_mask, feature_cols].values
    y_train = y.loc[train_mask].values
    X_cal = X.loc[cal_mask, feature_cols].values
    y_cal = y.loc[cal_mask].values
    
    # Fill NaN
    train_medians = np.nanmedian(X_train, axis=0)
    for col_idx in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, col_idx])
        X_train[mask, col_idx] = train_medians[col_idx]
        mask = np.isnan(X_cal[:, col_idx])
        X_cal[mask, col_idx] = train_medians[col_idx]
    
    # Train
    print("\n🏋️ Training final model...")
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {k: v for k, v in config.LGBM_PARAMS.items() 
              if k not in ["n_estimators", "early_stopping_rounds"]}
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.LGBM_PARAMS["n_estimators"],
    )
    
    # Calibrate
    print("📐 Calibrating probabilities...")
    cal_probs = model.predict(X_cal)
    calibrated_model = calibrate_probabilities(model, X_cal, y_cal)
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    print("\n📊 Top 10 Features (by gain):")
    for feat, imp in list(importance.items())[:10]:
        print(f"   {feat}: {imp:.0f}")
    
    # Save model and artifacts
    model_path = config.MODEL_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    
    # Save feature columns and medians for inference
    artifacts = {
        "feature_cols": feature_cols,
        "train_medians": train_medians.tolist(),
        "feature_importance": {k: float(v) for k, v in importance.items()},
    }
    artifacts_path = config.MODEL_DIR / "artifacts.json"
    with open(artifacts_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Artifacts saved to {artifacts_path}")
    
    return calibrated_model


if __name__ == "__main__":
    print("🏀 NCAAB Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_historical_dataset()
    
    # Build features
    X, y = build_features(df)
    
    # Walk-forward CV
    cv_results = walk_forward_cv(X, y)
    
    # Train final model
    model = train_final_model(X, y)
    
    print("\n🎉 Training complete!")

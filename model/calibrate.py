"""
Probability calibration for the NCAAB prediction model.

Well-calibrated probabilities are critical for Kalshi trading:
- If model says 65%, the event should happen ~65% of the time
- LightGBM raw probabilities are often overconfident
- Isotonic regression corrects this non-parametrically

Usage:
    from model.calibrate import calibrate_probabilities
"""
import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss
import pickle


class CalibratedModel:
    """
    Wraps a LightGBM model with isotonic calibration.
    
    Predict flow:
    1. Raw LightGBM probability
    2. Isotonic regression mapping
    3. Calibrated probability output
    """
    
    def __init__(self, base_model, calibrator: IsotonicRegression):
        self.base_model = base_model
        self.calibrator = calibrator
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        raw_probs = self.base_model.predict(X)
        calibrated = self.calibrator.predict(raw_probs)
        # Clip to valid probability range
        return np.clip(calibrated, 0.01, 0.99)
    
    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw (uncalibrated) probabilities."""
        return self.base_model.predict(X)


def calibrate_probabilities(
    model, 
    X_cal: np.ndarray, 
    y_cal: np.ndarray,
    probs_precomputed: bool = False,
) -> CalibratedModel:
    """
    Fit isotonic regression calibrator.

    Args:
        model: Trained LightGBM model
        X_cal: Calibration features, OR precomputed raw probabilities
               (e.g. out-of-fold CV predictions) if probs_precomputed=True
        y_cal: Calibration labels
        probs_precomputed: If True, X_cal is already raw probabilities.
                           Preferred: pass pooled out-of-fold predictions,
                           which match the distribution the model produces
                           on genuinely unseen games.

    Returns:
        CalibratedModel wrapping the original model + calibrator
    """
    # Get raw probabilities on calibration set
    raw_probs = X_cal if probs_precomputed else model.predict(X_cal)
    
    # Fit isotonic regression
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(raw_probs, y_cal)
    
    # Evaluate improvement
    raw_brier = brier_score_loss(y_cal, raw_probs)
    cal_probs = calibrator.predict(raw_probs)
    cal_brier = brier_score_loss(y_cal, cal_probs)
    
    print(f"   Raw Brier Score:        {raw_brier:.4f}")
    print(f"   Calibrated Brier Score: {cal_brier:.4f}")
    print(f"   Improvement:            {(raw_brier - cal_brier) / raw_brier * 100:.1f}%")
    
    # Print calibration table
    print("\n   Calibration Check (predicted vs actual):")
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (cal_probs >= bins[i]) & (cal_probs < bins[i + 1])
        if mask.sum() > 0:
            pred_avg = cal_probs[mask].mean()
            actual_avg = y_cal[mask].mean()
            n = mask.sum()
            print(f"   [{bins[i]:.1f}-{bins[i+1]:.1f}): pred={pred_avg:.3f}, actual={actual_avg:.3f}, n={n}")
    
    return CalibratedModel(model, calibrator)


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """
    Compute calibration metrics.
    
    Returns:
        dict with ECE (Expected Calibration Error), MCE (Max Calibration Error),
        and per-bin calibration data.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            pred_avg = y_prob[mask].mean()
            actual_avg = y_true[mask].mean()
            count = mask.sum()
            bin_data.append({
                "bin_lower": bins[i],
                "bin_upper": bins[i + 1],
                "predicted": pred_avg,
                "actual": actual_avg,
                "count": count,
                "abs_error": abs(pred_avg - actual_avg),
            })
    
    total = sum(b["count"] for b in bin_data)
    ece = sum(b["abs_error"] * b["count"] / total for b in bin_data)
    mce = max(b["abs_error"] for b in bin_data) if bin_data else 0.0
    
    return {
        "ece": ece,
        "mce": mce,
        "bins": bin_data,
    }

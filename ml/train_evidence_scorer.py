"""Train an evidence quality scorer to replace the hardcoded 70% word-overlap heuristic.

Generates training data locally from sample patient visit notes:
  - Positive pairs: real sentences extracted from visit notes (genuine evidence)
  - Negative pairs: sentences from OTHER patients, shuffled tokens, partial fabrications

Trains a LogisticRegression classifier on hand-crafted features (TF-IDF cosine,
Jaccard overlap, weighted word overlap, longest common substring ratio, etc.).

Usage:
    python -m ml.train_evidence_scorer
    python -m ml.train_evidence_scorer --C 1.0 --folds 5 --seed 42
"""

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from ml.evidence_scorer import FEATURE_NAMES, extract_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
SAMPLE_DATA = PROJECT_ROOT / "sample_data" / "patient_data.json"


# --------------------------------------------------------------------------
# Training data generation
# --------------------------------------------------------------------------


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out very short ones."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if len(s.strip().split()) >= 5]


def generate_training_data(
    patients: list[dict], seed: int = 42
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generate labeled feature vectors from patient visit notes.

    For each patient:
      - Positive: real sentences from their own notes (evidence that IS grounded)
      - Negative: sentences from other patients' notes, shuffled versions, truncated

    Returns (X, y, metadata) where metadata tracks the raw text for error analysis.
    """
    rng = random.Random(seed)
    features_list = []
    labels = []
    metadata = []

    # Pre-extract all sentences per patient
    patient_sentences = {}
    for i, patient in enumerate(patients):
        all_sents = []
        for note in patient["visit_notes"]:
            all_sents.extend(extract_sentences(note))
        patient_sentences[i] = all_sents

    for i, patient in enumerate(patients):
        notes = patient["visit_notes"]
        own_sentences = patient_sentences[i]
        if not own_sentences:
            continue

        # --- Positive samples: real sentences from this patient's notes ---
        for sent in own_sentences:
            feat = extract_features(sent, notes)
            features_list.append(feat)
            labels.append(1)
            metadata.append({"evidence": sent, "patient_idx": i, "label": 1, "type": "real"})

        # --- Negative samples ---
        # Type 1: Sentences from other patients (wrong patient context)
        other_indices = [j for j in range(len(patients)) if j != i]
        for _ in range(min(len(own_sentences), 8)):
            other_idx = rng.choice(other_indices)
            other_sents = patient_sentences[other_idx]
            if other_sents:
                wrong_evidence = rng.choice(other_sents)
                feat = extract_features(wrong_evidence, notes)
                features_list.append(feat)
                labels.append(0)
                metadata.append({
                    "evidence": wrong_evidence, "patient_idx": i,
                    "label": 0, "type": "wrong_patient",
                })

        # Type 2: Shuffled tokens from own sentences (destroys substring match)
        for sent in rng.sample(own_sentences, min(4, len(own_sentences))):
            tokens = sent.split()
            rng.shuffle(tokens)
            shuffled = " ".join(tokens)
            feat = extract_features(shuffled, notes)
            features_list.append(feat)
            labels.append(0)
            metadata.append({
                "evidence": shuffled, "patient_idx": i,
                "label": 0, "type": "shuffled",
            })

        # Type 3: Fabricated medical text (plausible but not in notes)
        fabricated = [
            "Patient reported severe allergic reaction to penicillin last month.",
            "Lab results show hemoglobin A1c of 9.2%, indicating poor glycemic control.",
            "MRI of lumbar spine reveals herniation at L4-L5 with nerve compression.",
            "Patient has been on metformin 1000mg twice daily for three years.",
            "Echocardiogram shows ejection fraction of 35%, consistent with heart failure.",
            "Colonoscopy performed last year showed three benign polyps.",
            "Patient denies any history of substance abuse or psychiatric illness.",
            "Thyroid function tests within normal limits, TSH 2.1 mIU/L.",
        ]
        for _ in range(min(3, len(own_sentences))):
            fake = rng.choice(fabricated)
            feat = extract_features(fake, notes)
            features_list.append(feat)
            labels.append(0)
            metadata.append({
                "evidence": fake, "patient_idx": i,
                "label": 0, "type": "fabricated",
            })

    X = np.array(features_list)
    y = np.array(labels)
    logger.info(f"Generated {len(y)} samples: {sum(y)} positive, {len(y) - sum(y)} negative")
    return X, y, metadata


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    metadata: list[dict],
    C_values: list[float],
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler, dict]:
    """Train LogisticRegression with grid search over C, evaluate with stratified k-fold.

    Returns (best_model, scaler, results_dict).
    """
    np.random.seed(seed)

    # Scale features for stable logistic regression convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Grid search with stratified CV
    logger.info(f"Running {n_folds}-fold stratified CV, C search: {C_values}")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        LogisticRegression(random_state=seed, max_iter=1000, solver="lbfgs"),
        param_grid={"C": C_values},
        cv=cv,
        scoring="roc_auc",
        return_train_score=True,
        refit=True,
    )
    grid.fit(X_scaled, y)

    best_model = grid.best_estimator_
    best_C = grid.best_params_["C"]
    logger.info(f"Best C: {best_C}, CV AUROC: {grid.best_score_:.4f}")

    # Log all C values tried
    for i, C in enumerate(C_values):
        mean_train = grid.cv_results_["mean_train_score"][i]
        mean_test = grid.cv_results_["mean_test_score"][i]
        logger.info(f"  C={C:<8} train_auroc={mean_train:.4f}  val_auroc={mean_test:.4f}")

    # Final evaluation on full dataset (model was refit on all data)
    y_pred = best_model.predict(X_scaled)
    y_prob = best_model.predict_proba(X_scaled)[:, 1]

    auroc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, y_pred)
    logloss = log_loss(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    results = {
        "best_C": best_C,
        "cv_auroc": float(grid.best_score_),
        "final_auroc": float(auroc),
        "final_f1": float(f1),
        "final_log_loss": float(logloss),
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y),
        "n_positive": int(sum(y)),
        "n_negative": int(len(y) - sum(y)),
        "n_folds": n_folds,
        "seed": seed,
        "feature_names": FEATURE_NAMES,
        "coefficients": dict(zip(FEATURE_NAMES, best_model.coef_[0].tolist())),
    }

    # --- Error analysis ---
    logger.info("\n=== Error Analysis ===")
    logger.info(f"Confusion matrix:\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}")

    # Find false negatives (model says bad evidence, but it's actually real)
    fn_indices = [i for i in range(len(y)) if y[i] == 1 and y_pred[i] == 0]
    if fn_indices:
        logger.info(f"\nFalse negatives ({len(fn_indices)} total) — model missed real evidence:")
        for idx in fn_indices[:3]:
            m = metadata[idx]
            logger.info(f"  [{m['type']}] \"{m['evidence'][:80]}...\"")
            logger.info(f"    features: {dict(zip(FEATURE_NAMES, X[idx].round(3)))}")

    # Find false positives (model says good evidence, but it's fabricated/wrong)
    fp_indices = [i for i in range(len(y)) if y[i] == 0 and y_pred[i] == 1]
    if fp_indices:
        logger.info(f"\nFalse positives ({len(fp_indices)} total) — model accepted bad evidence:")
        for idx in fp_indices[:3]:
            m = metadata[idx]
            logger.info(f"  [{m['type']}] \"{m['evidence'][:80]}...\"")
            logger.info(f"    features: {dict(zip(FEATURE_NAMES, X[idx].round(3)))}")

    # Classification report
    logger.info(f"\n{classification_report(y, y_pred, target_names=['bad_evidence', 'real_evidence'])}")

    # Feature importance (logistic regression coefficients)
    logger.info("Feature coefficients (positive = evidence is real):")
    for name, coef in sorted(zip(FEATURE_NAMES, best_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        logger.info(f"  {name:<20} {coef:+.4f}")

    return best_model, scaler, results


# --------------------------------------------------------------------------
# Save checkpoint
# --------------------------------------------------------------------------


def save_checkpoint(
    model: LogisticRegression,
    scaler: StandardScaler,
    results: dict,
    checkpoint_dir: Path,
) -> str:
    """Save model, scaler, and training results to checkpoint directory."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = checkpoint_dir / f"evidence_scorer_{timestamp}.joblib"
    scaler_path = checkpoint_dir / f"scaler_{timestamp}.joblib"
    results_path = checkpoint_dir / f"results_{timestamp}.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save as "latest" for easy loading
    joblib.dump(model, checkpoint_dir / "evidence_scorer_latest.joblib")
    joblib.dump(scaler, checkpoint_dir / "scaler_latest.joblib")

    logger.info(f"Checkpoint saved: {model_path.name}")
    logger.info(f"Scaler saved: {scaler_path.name}")
    logger.info(f"Results saved: {results_path.name}")

    return model_path.name


def main():
    parser = argparse.ArgumentParser(description="Train evidence quality scorer")
    parser.add_argument("--C", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                        help="Regularization values to search (default: 0.001 0.01 0.1 1 10 100)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Evidence Quality Scorer — Training")
    logger.info("=" * 60)

    # Load patient data
    with open(SAMPLE_DATA) as f:
        patients = json.load(f)
    logger.info(f"Loaded {len(patients)} patients from {SAMPLE_DATA.name}")

    # Generate training data
    X, y, metadata = generate_training_data(patients, seed=args.seed)

    # Train and evaluate
    model, scaler, results = train_and_evaluate(
        X, y, metadata,
        C_values=args.C,
        n_folds=args.folds,
        seed=args.seed,
    )

    # Save checkpoint
    checkpoint_name = save_checkpoint(model, scaler, results, CHECKPOINT_DIR)

    # --- Final validation log line (for the form) ---
    logger.info("=" * 60)
    logger.info(
        f"FINAL | checkpoint={checkpoint_name} | "
        f"log_loss={results['final_log_loss']:.4f} | "
        f"auroc={results['final_auroc']:.4f} | "
        f"f1={results['final_f1']:.4f} | "
        f"cv_auroc={results['cv_auroc']:.4f} | "
        f"C={results['best_C']} | "
        f"samples={results['n_samples']} (pos={results['n_positive']}, neg={results['n_negative']})"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

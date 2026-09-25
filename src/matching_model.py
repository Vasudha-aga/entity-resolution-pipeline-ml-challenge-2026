"""
Pairwise entity matching model and evaluation module for Amazon ML Challenge 2026.

Implements EntityMatchingModel using StandardScaler and LogisticRegression,
RuleBasedBaseline benchmark matcher, along with pairwise precision/recall/F0.5/F1 metrics,
S1 exact match evaluation, and threshold tuning.
"""

import os
import pickle
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, confusion_matrix


class RuleBasedBaseline:
    """
    Simple interpretable rule-based benchmark matcher for comparison against ML models.
    Rule: Predicts match (1.0 probability) if normalized business names match exactly
    and country matches, or high token similarity (>= 0.80) with matching country.
    """

    def __init__(self, name_exact_idx: int = 0, country_match_idx: int = 22, token_jaccard_idx: int = 1):
        self.name_exact_idx = name_exact_idx
        self.country_match_idx = country_match_idx
        self.token_jaccard_idx = token_jaccard_idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return deterministic rule confidence scores between 0.0 and 1.0."""
        if len(X) == 0:
            return np.array([], dtype=np.float32)

        name_exact = X[:, self.name_exact_idx]
        country_match = X[:, self.country_match_idx]
        token_jaccard = X[:, self.token_jaccard_idx]

        probs = np.zeros(len(X), dtype=np.float32)
        probs[(name_exact == 1.0) & (country_match == 1.0)] = 1.0
        probs[(name_exact == 0.0) & (token_jaccard >= 0.80) & (country_match == 1.0)] = 0.85
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


class EntityMatchingModel:
    """
    Supervised pairwise matching classifier using StandardScaler + LogisticRegression.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 1000,
        random_state: int = 42
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs"
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EntityMatchingModel":
        """Fit scaler and logistic regression classifier on training feature matrix X and labels y."""
        if len(X) == 0:
            raise ValueError("Training matrix X is empty.")

        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of candidate pair being a true match (class 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        if len(X) == 0:
            return np.array([], dtype=np.float32)

        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)[:, 1]
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary match labels (0 or 1) based on decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate_pairwise(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute pairwise classification metrics (Precision, Recall, F0.5, F1, CM)."""
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return {
            "precision": float(prec),
            "recall": float(rec),
            "f05": float(f05),
            "f1": float(f1),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "total_pairs": len(y_true),
            "positive_pairs": int(np.sum(y_true)),
            "predicted_positive_pairs": int(np.sum(y_pred))
        }

    def evaluate_s1_exact_match(
        self,
        pairs: List[Tuple[str, str]],
        probs: np.ndarray,
        threshold: float,
        ground_truth: Dict[str, List[str]],
        target_s1_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate S1-level exact match rate.
        For each S1 entity, compares the set of predicted matched candidate IDs
        against the complete ground-truth match set for that S1 entity.
        """
        predicted_matches: Dict[str, List[str]] = {}
        for (s1_id, cand_id), p in zip(pairs, probs):
            if p >= threshold:
                if s1_id not in predicted_matches:
                    predicted_matches[s1_id] = []
                predicted_matches[s1_id].append(cand_id)

        eval_s1_ids = target_s1_ids if target_s1_ids is not None else set(ground_truth.keys())

        exact_matches = 0
        total_eval_entities = len(eval_s1_ids)

        for s1_id in eval_s1_ids:
            true_set = set(ground_truth.get(s1_id, []))
            pred_set = set(predicted_matches.get(s1_id, []))
            if true_set == pred_set:
                exact_matches += 1

        exact_match_rate = (exact_matches / total_eval_entities) * 100 if total_eval_entities > 0 else 0.0

        return {
            "total_s1_entities": total_eval_entities,
            "exact_matched_s1_entities": exact_matches,
            "s1_exact_match_rate_pct": float(exact_match_rate),
            "total_predicted_links": sum(len(v) for v in predicted_matches.values())
        }

    def run_threshold_experiment(
        self,
        val_pairs: List[Tuple[str, str]],
        val_probs: np.ndarray,
        val_y_true: np.ndarray,
        val_gt: Dict[str, List[str]],
        val_s1_ids: Set[str],
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate decision thresholds: [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90].
        Selects best threshold based ONLY on validation pairwise F0.5 score.
        """
        if thresholds is None:
            thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        results = []
        best_f05 = -1.0
        best_threshold = 0.5

        for t in thresholds:
            t = float(np.round(t, 2))
            y_pred = (val_probs >= t).astype(int)
            pairwise_metrics = self.evaluate_pairwise(val_y_true, y_pred)
            s1_metrics = self.evaluate_s1_exact_match(val_pairs, val_probs, t, val_gt, val_s1_ids)

            entry = {
                "threshold": t,
                "precision": pairwise_metrics["precision"],
                "recall": pairwise_metrics["recall"],
                "f05": pairwise_metrics["f05"],
                "f1": pairwise_metrics["f1"],
                "tp": pairwise_metrics["tp"],
                "fp": pairwise_metrics["fp"],
                "fn": pairwise_metrics["fn"],
                "s1_exact_match_rate_pct": s1_metrics["s1_exact_match_rate_pct"],
                "predicted_matches": pairwise_metrics["predicted_positive_pairs"]
            }
            results.append(entry)

            if pairwise_metrics["f05"] > best_f05:
                best_f05 = pairwise_metrics["f05"]
                best_threshold = t

        return {
            "threshold_sweep": results,
            "best_threshold": best_threshold,
            "best_validation_f05": best_f05
        }

    def save(self, filepath: str) -> None:
        """Save model and scaler state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"scaler": self.scaler, "classifier": self.classifier, "is_fitted": self.is_fitted}, f)

    def load(self, filepath: str) -> "EntityMatchingModel":
        """Load model and scaler state from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.classifier = data["classifier"]
            self.is_fitted = data["is_fitted"]
        return self

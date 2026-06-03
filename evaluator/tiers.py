"""Progressive evaluation tiers for CertFE."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from .adapter import Evaluator


@dataclass
class EvalResult:
    passed: bool
    delta_metric: float
    delta_metric_std: float = 0.0
    tier_reached: str = ""
    cost_seconds: float = 0.0
    rejection_reason: str = ""


class Tier1Proxy:
    """Sanity check: non-NaN, non-constant."""

    @staticmethod
    def check(feature_col: pd.Series) -> tuple[bool, str]:
        if feature_col is None or len(feature_col) == 0:
            return False, "TIER1_NAN"
        nan_rate = feature_col.isna().mean()
        if nan_rate > 0.5:
            return False, "TIER1_NAN"
        valid = feature_col.dropna()
        if len(valid) == 0:
            return False, "TIER1_NAN"
        if pd.api.types.is_numeric_dtype(valid):
            if valid.std() < 1e-9:
                return False, "TIER1_CONSTANT"
        else:
            if valid.nunique() <= 1:
                return False, "TIER1_CONSTANT"
        return True, ""


class Tier2SmallSample:
    """Small-sample model fit to detect 'no signal' early."""

    def __init__(self, evaluator: Evaluator, epsilon_rel: float = 0.01, n1: int = 2000):
        self.evaluator = evaluator
        self.epsilon_rel = epsilon_rel
        self.n1 = n1

    def evaluate(self, feature_train: pd.Series, existing_train: list[pd.Series]) -> tuple[bool, float]:
        bundle = self.evaluator.bundle
        n_sample = min(len(bundle.x_train), self.n1)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(bundle.x_train), size=n_sample, replace=False)

        x_sample = bundle.x_train.iloc[idx].copy()
        y_sample = bundle.y_train.iloc[idx]

        # Split sample into mini-train / mini-val (80/20)
        n_tr = int(n_sample * 0.8)
        x_tr = x_sample.iloc[:n_tr].copy()
        y_tr = y_sample.iloc[:n_tr]
        x_ev = x_sample.iloc[n_tr:].copy()
        y_ev = y_sample.iloc[n_tr:]

        x_tr_base = x_tr.copy()
        x_ev_base = x_ev.copy()
        for i, col in enumerate(existing_train):
            x_tr_base[f"certfe_{i}"] = col.reindex(x_tr_base.index)
            x_ev_base[f"certfe_{i}"] = col.reindex(x_ev_base.index)

        x_tr_new = x_tr_base.copy()
        x_ev_new = x_ev_base.copy()
        x_tr_new[f"certfe_{len(existing_train)}"] = feature_train.reindex(x_tr_new.index)
        x_ev_new[f"certfe_{len(existing_train)}"] = feature_train.reindex(x_ev_new.index)

        score_base = self.evaluator._score(x_tr_base, x_ev_base, y_train=y_tr, y_eval=y_ev)
        score_new = self.evaluator._score(x_tr_new, x_ev_new, y_train=y_tr, y_eval=y_ev)
        delta = (score_new - score_base) / max(abs(score_base), 1e-9)

        threshold = 0.3 * self.epsilon_rel
        return delta >= threshold, delta


class Tier3SingleSplit:
    """Full train/val single-split evaluation."""

    def __init__(self, evaluator: Evaluator, epsilon_rel: float = 0.01):
        self.evaluator = evaluator
        self.epsilon_rel = epsilon_rel

    def evaluate(
        self,
        new_feature_train: pd.Series,
        new_feature_val: pd.Series,
        existing_columns_train: list[pd.Series],
        existing_columns_val: list[pd.Series],
    ) -> tuple[bool, float]:
        # Marginal delta: score(raw+S+new_f) - score(raw+S), normalized by score(raw+S)
        x_tr_base = self.evaluator.bundle.x_train.copy()
        x_ev_base = self.evaluator.bundle.x_val.copy()
        for i, (ct, cv) in enumerate(zip(existing_columns_train, existing_columns_val)):
            x_tr_base[f"certfe_{i}"] = ct.reindex(x_tr_base.index)
            x_ev_base[f"certfe_{i}"] = cv.reindex(x_ev_base.index)

        x_tr_new = x_tr_base.copy()
        x_ev_new = x_ev_base.copy()
        idx = len(existing_columns_train)
        x_tr_new[f"certfe_{idx}"] = new_feature_train.reindex(x_tr_new.index)
        x_ev_new[f"certfe_{idx}"] = new_feature_val.reindex(x_ev_new.index)

        score_base = self.evaluator._score(x_tr_base, x_ev_base)
        score_new = self.evaluator._score(x_tr_new, x_ev_new)
        delta = (score_new - score_base) / max(abs(score_base), 1e-9)

        threshold = 0.7 * self.epsilon_rel
        passed = delta >= threshold
        return passed, delta


class Tier4MultiSplit:
    """5-fold CV for stability estimation (produces delta_metric_mean + std)."""

    def __init__(self, evaluator: Evaluator, epsilon_rel: float = 0.01, n_folds: int = 5, seed: int = 42):
        self.evaluator = evaluator
        self.epsilon_rel = epsilon_rel
        self.n_folds = n_folds
        self.seed = seed

    def evaluate(
        self,
        feature_col: pd.Series,
        existing_cols: list[pd.Series],
    ) -> tuple[bool, float, float]:
        """Returns (passed, mean_delta, std_delta)."""
        bundle = self.evaluator.bundle
        x_full = pd.concat([bundle.x_train, bundle.x_val])
        y_full = pd.concat([bundle.y_train, bundle.y_val])

        for i, col in enumerate(existing_cols):
            x_full[f"certfe_{i}"] = col.reindex(x_full.index)
        x_base = x_full.copy()
        x_full[f"certfe_{len(existing_cols)}"] = feature_col.reindex(x_full.index)

        if bundle.task == "classification":
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            splits = list(kf.split(x_full, y_full))
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            splits = list(kf.split(x_full))

        deltas = []
        for train_idx, val_idx in splits:
            x_tr_base = x_base.iloc[train_idx]
            x_val_base = x_base.iloc[val_idx]
            x_tr_new = x_full.iloc[train_idx]
            x_val_new = x_full.iloc[val_idx]
            y_tr = y_full.iloc[train_idx]
            y_val = y_full.iloc[val_idx]

            score_base = self.evaluator._score(x_tr_base, x_val_base, y_train=y_tr, y_eval=y_val)
            score_new = self.evaluator._score(x_tr_new, x_val_new, y_train=y_tr, y_eval=y_val)
            delta = (score_new - score_base) / max(abs(score_base), 1e-9)
            deltas.append(delta)

        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas))
        passed = mean_d >= self.epsilon_rel
        return passed, mean_d, std_d


class ProgressiveEvaluator:
    """Runs Tier 1 → Tier 2 → Tier 3 → Tier 4 sequentially."""

    def __init__(self, evaluator: Evaluator, epsilon_rel: float = 0.01, use_tier2: bool = False, use_tier4: bool = False):
        self.evaluator = evaluator
        self.tier2 = Tier2SmallSample(evaluator, epsilon_rel) if use_tier2 else None
        self.tier3 = Tier3SingleSplit(evaluator, epsilon_rel)
        self.tier4 = Tier4MultiSplit(evaluator, epsilon_rel) if use_tier4 else None
        self.epsilon_rel = epsilon_rel

    def evaluate(
        self,
        feature_train: pd.Series,
        feature_val: pd.Series,
        existing_columns_train: list[pd.Series],
        existing_columns_val: list[pd.Series],
    ) -> EvalResult:
        t0 = time.time()

        # Tier 1
        ok, reason = Tier1Proxy.check(feature_train)
        if not ok:
            return EvalResult(passed=False, delta_metric=0.0, tier_reached="tier1",
                              cost_seconds=time.time() - t0, rejection_reason=reason)

        # Tier 2 (optional — disabled by default for small datasets)
        if self.tier2:
            passed2, delta2 = self.tier2.evaluate(feature_train, existing_columns_train)
            if not passed2:
                return EvalResult(passed=False, delta_metric=delta2, tier_reached="tier2",
                                  cost_seconds=time.time() - t0, rejection_reason="TIER2_NO_SIGNAL")

        # Tier 3
        passed3, delta3 = self.tier3.evaluate(
            feature_train, feature_val, existing_columns_train, existing_columns_val)
        if not passed3:
            return EvalResult(passed=False, delta_metric=delta3, tier_reached="tier3",
                              cost_seconds=time.time() - t0, rejection_reason="TIER3_SINGLE_FAIL")

        if delta3 < self.epsilon_rel:
            return EvalResult(passed=False, delta_metric=delta3, tier_reached="tier3",
                              cost_seconds=time.time() - t0, rejection_reason="TIER3_SINGLE_FAIL")

        # Tier 4 (optional)
        if self.tier4:
            feature_full = pd.concat([feature_train, feature_val])
            existing_full = [pd.concat([t, v]) for t, v in zip(existing_columns_train, existing_columns_val)]
            passed4, mean4, std4 = self.tier4.evaluate(feature_full, existing_full)
            cost = time.time() - t0
            if not passed4:
                return EvalResult(passed=False, delta_metric=mean4, delta_metric_std=std4,
                                  tier_reached="tier4", cost_seconds=cost, rejection_reason="TIER4_LOW_GAIN")
            return EvalResult(passed=True, delta_metric=mean4, delta_metric_std=std4,
                              tier_reached="tier4", cost_seconds=cost)

        # No Tier 4 — accept based on Tier 3
        return EvalResult(passed=True, delta_metric=delta3, tier_reached="tier3",
                          cost_seconds=time.time() - t0)

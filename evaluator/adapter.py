"""Lightweight evaluation adapter for CertFE.

Loads data using the same split logic as PromptFE (same seed, same test_size/val_size)
but evaluates CertFE FeatureProgram objects directly via DSL execution.
Supports RF (default), LR, LGB downstream models with one_hot=False (C5-clean).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PromptFE"))
from metrics import r2_score as promptfe_r2_score


@dataclass
class DatasetBundle:
    """Holds train/val/test splits and metadata for one dataset."""
    name: str
    task: str
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    col_types: dict[int, str]
    col_descriptions: list[str] | None
    n_original_cols: int


def load_dataset(name: str, seed: int = 0) -> DatasetBundle:
    """Load dataset with PromptFE-compatible splits (one_hot=False)."""
    data_dir = Path(__file__).resolve().parents[2] / "PromptFE" / "data"
    data = pd.read_csv(data_dir / f"{name}.csv", header=None)
    with open(data_dir / f"{name}.json") as f:
        meta = json.load(f)

    desc = None
    desc_path = data_dir / f"{name}.txt"
    if desc_path.exists():
        desc = [line.strip() for line in desc_path.read_text().splitlines()]

    task = "classification" if meta["task"] == "C" else "regression"
    col_types = {int(k): v for k, v in meta.items() if k != "task"}

    train_val, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.2, shuffle=True, random_state=seed)

    def split_xy(df):
        return df.iloc[:, :-1].copy(), df.iloc[:, -1].copy()

    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(val)
    x_test, y_test = split_xy(test)

    if task == "classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(data.iloc[:, -1])
        y_train = pd.Series(le.transform(y_train), index=y_train.index)
        y_val = pd.Series(le.transform(y_val), index=y_val.index)
        y_test = pd.Series(le.transform(y_test), index=y_test.index)

    return DatasetBundle(
        name=name, task=task,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        col_types=col_types,
        col_descriptions=desc,
        n_original_cols=x_train.shape[1],
    )


class Evaluator:
    """Evaluates CertFE feature programs on PromptFE-compatible splits.

    Supports model_name in {"RF", "LR", "LGB"}.
    """

    def __init__(self, bundle: DatasetBundle, model_name: str = "RF", n_estimators: int = 100,
                 use_cv_baseline: bool = False):
        self.bundle = bundle
        self.model_name = model_name
        self.n_estimators = n_estimators
        self.use_cv_baseline = use_cv_baseline
        self._baseline_score: float | None = None

    @property
    def baseline_score(self) -> float:
        if self._baseline_score is None:
            if self.use_cv_baseline:
                self._baseline_score = self._cv_score()
            else:
                self._baseline_score = self._score(self.bundle.x_train, self.bundle.x_val)
        return self._baseline_score

    def _cv_score(self) -> float:
        """5-fold CV baseline on train data only — aligned with PromptFE mode='train'."""
        x = self.bundle.x_train.copy()
        y = self.bundle.y_train.copy()

        # Same preprocessing as _score
        x = x.replace([np.inf, -np.inf], np.nan).fillna(-1)
        x.columns = x.columns.astype(str)
        for col in x.columns:
            if not pd.api.types.is_numeric_dtype(x[col]):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col].astype(str))

        if self.model_name == "LR":
            scaler = MinMaxScaler()
            x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)

        if self.bundle.task == "classification":
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            if self.model_name == "LR":
                model = LogisticRegression(max_iter=2000, random_state=0, n_jobs=-1)
            elif self.model_name == "LGB":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(n_estimators=self.n_estimators, random_state=0, verbose=-1, n_jobs=1)
            else:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0, n_jobs=-1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            scores = cross_val_score(model, x, y, cv=cv, scoring="f1_micro", n_jobs=-1)
            return float(scores.mean())
        else:
            from sklearn.model_selection import KFold, cross_val_score
            if self.model_name == "LR":
                model = Lasso(max_iter=2000, random_state=0)
            elif self.model_name == "LGB":
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(n_estimators=self.n_estimators, random_state=0, verbose=-1, n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0, n_jobs=-1)
            cv = KFold(n_splits=5, shuffle=True, random_state=0)
            scores = cross_val_score(model, x, y, cv=cv, scoring=make_scorer(promptfe_r2_score), n_jobs=-1)
            return float(scores.mean())

    def evaluate_programs(
        self,
        programs_columns: list[pd.Series],
        mode: str = "val",
    ) -> float:
        """Evaluate a set of feature columns added to raw features."""
        if mode == "val":
            x_tr = self.bundle.x_train.copy()
            x_ev = self.bundle.x_val.copy()
            y_ev = self.bundle.y_val
        else:
            x_tr = pd.concat([self.bundle.x_train, self.bundle.x_val])
            x_ev = self.bundle.x_test.copy()
            y_ev = self.bundle.y_test

        for i, col in enumerate(programs_columns):
            col_train = col.reindex(x_tr.index)
            col_eval = col.reindex(x_ev.index)
            x_tr[f"certfe_{i}"] = col_train
            x_ev[f"certfe_{i}"] = col_eval

        return self._score(x_tr, x_ev, y_eval=y_ev)

    def _score(self, x_train: pd.DataFrame, x_eval: pd.DataFrame, y_train=None, y_eval=None) -> float:
        x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(-1)
        x_eval = x_eval.replace([np.inf, -np.inf], np.nan).fillna(-1)
        x_train.columns = x_train.columns.astype(str)
        x_eval.columns = x_eval.columns.astype(str)

        for col in x_train.columns:
            if not pd.api.types.is_numeric_dtype(x_train[col]):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                combined = pd.concat([x_train[col].astype(str), x_eval[col].astype(str)])
                le.fit(combined)
                x_train[col] = le.transform(x_train[col].astype(str))
                x_eval[col] = le.transform(x_eval[col].astype(str))

        if y_train is None:
            y_train = self.bundle.y_train if len(x_train) == len(self.bundle.y_train) else pd.concat(
                [self.bundle.y_train, self.bundle.y_val]
            )
        if y_eval is None:
            y_eval = self.bundle.y_val

        # LR needs scaling
        if self.model_name == "LR":
            scaler = MinMaxScaler()
            x_train_s = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
            x_eval_s = pd.DataFrame(scaler.transform(x_eval), columns=x_eval.columns, index=x_eval.index)
            x_train, x_eval = x_train_s, x_eval_s

        if self.bundle.task == "classification":
            if self.model_name == "LR":
                model = LogisticRegression(max_iter=2000, random_state=0, n_jobs=-1)
            elif self.model_name == "LGB":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(n_estimators=self.n_estimators, random_state=0, verbose=-1, n_jobs=1)
            else:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0, n_jobs=-1)
            model.fit(x_train, y_train)
            preds = model.predict(x_eval)
            return f1_score(y_eval, preds, average="micro")
        else:
            if self.model_name == "LR":
                model = Lasso(max_iter=2000, random_state=0)
            elif self.model_name == "LGB":
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(n_estimators=self.n_estimators, random_state=0, verbose=-1, n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0, n_jobs=-1)
            model.fit(x_train, y_train)
            preds = model.predict(x_eval)
            return promptfe_r2_score(y_eval, preds)

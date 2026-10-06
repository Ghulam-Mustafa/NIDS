"""Model factory and fitting helpers."""
import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              IsolationForest, RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "decision_tree": DecisionTreeClassifier,
    "isolation_forest": IsolationForest,
}


def build_model(kind: str, params: dict):
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown model type '{kind}'. Choose from {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**(params or {}))


def fit_model(model, X, y):
    """HistGradientBoosting has no class_weight: use sqrt-balanced sample weights instead
    (helps rare classes without letting tiny classes dominate)."""
    if isinstance(model, HistGradientBoostingClassifier):
        w = np.sqrt(compute_sample_weight("balanced", y))
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)
    return model

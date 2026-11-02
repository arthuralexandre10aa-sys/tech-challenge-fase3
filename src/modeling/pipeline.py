"""Pipeline completo: engenharia de atributos + pré-processamento + modelo."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.preprocessing.features import DerivedFeatures, UFRelativeStats


def preprocessador() -> ColumnTransformer:
    numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numerico, make_column_selector(dtype_include=np.number)),
         ("cat", categorico, make_column_selector(dtype_exclude=np.number))],
        verbose_feature_names_out=False,
    )


def monta_pipeline(modelo) -> Pipeline:
    return Pipeline([
        ("derivadas", DerivedFeatures()),
        ("uf_stats", UFRelativeStats()),
        ("prep", preprocessador()),
        ("modelo", modelo),
    ])


def candidatos() -> dict:
    return {
        "dummy": (DummyClassifier(strategy="prior"), {}),
        "logreg": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                   {"modelo__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
        "random_forest": (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
                          {"modelo__max_depth": [4, 6, 8, 12, None],
                           "modelo__min_samples_leaf": [5, 10, 20, 40],
                           "modelo__max_features": ["sqrt", 0.3, 0.5]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                              {"modelo__n_estimators": [100, 200, 300],
                               "modelo__learning_rate": [0.02, 0.05, 0.1],
                               "modelo__max_depth": [2, 3, 4],
                               "modelo__min_samples_leaf": [10, 20, 40],
                               "modelo__subsample": [0.7, 0.85, 1.0]}),
    }


def nomes_features(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())

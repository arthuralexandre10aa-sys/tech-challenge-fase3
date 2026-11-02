"""Pipeline completo: engenharia de atributos + pré-processamento + filtro + modelo.

Medidas contra overfitting (todas DENTRO do Pipeline, logo refeitas a cada fold):
  * ``CorrelationFilter`` remove colunas quase duplicadas;
  * hiperparâmetros restritivos (árvores rasas, folhas grandes, subamostragem, taxa de aprendizado baixa);
  * regularização L2 forte na regressão logística.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, USAR_UF_NO_MODELO
from src.preprocessing.features import CorrelationFilter, DerivedFeatures, UFRelativeStats

LIMITE_CORRELACAO = 0.90


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


def monta_pipeline(modelo, limite_corr: float = LIMITE_CORRELACAO) -> Pipeline:
    etapas = [("derivadas", DerivedFeatures())]
    if USAR_UF_NO_MODELO:  # variáveis relativas à UF codificam o estado; ficam fora por padrão
        etapas.append(("uf_stats", UFRelativeStats()))
    etapas += [("prep", preprocessador()), ("filtro", CorrelationFilter(limite_corr)), ("modelo", modelo)]
    return Pipeline(etapas)


def candidatos() -> dict:
    """Espaços de busca deliberadamente restritivos (menos capacidade de memorizar)."""
    return {
        "dummy": (DummyClassifier(strategy="prior"), {}),
        "logreg": (LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
                   {"modelo__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1]}),
        "random_forest": (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
                          {"modelo__max_depth": [3, 4, 6],
                           "modelo__min_samples_leaf": [40, 80, 120],
                           "modelo__max_features": ["sqrt", 0.2, 0.4],
                           "modelo__max_samples": [0.3, 0.5, 0.7]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                              {"modelo__n_estimators": [50, 100, 150],
                               "modelo__learning_rate": [0.03, 0.05, 0.1],
                               "modelo__max_depth": [1, 2],
                               "modelo__min_samples_leaf": [40, 80, 120],
                               "modelo__subsample": [0.5, 0.7],
                               "modelo__max_features": [0.3, 0.6]}),
    }


def nomes_features(pipe: Pipeline) -> list[str]:
    """Nomes das colunas que chegam ao modelo (depois do pré-processamento e do filtro)."""
    nomes = pipe.named_steps["prep"].get_feature_names_out()
    if "filtro" in pipe.named_steps:
        nomes = pipe.named_steps["filtro"].get_feature_names_out(nomes)
    return list(nomes)

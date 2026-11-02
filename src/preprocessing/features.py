"""Engenharia de atributos como transformadores do scikit-learn.

Ficam DENTRO do Pipeline: qualquer estatística aprendida (ex.: médias por UF) é
calculada só com o treino de cada fold, o que evita data leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_LOG_HINTS = ("populacao_residente", "pib", "qtd_", "area_da_unidade", "densidade_demografica",
              "matricula", "repasse", "familias")


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Atributos derivados sem estado: gap_ant, evolucao_req, sem_historico, pib_per_capita e log1p."""

    def fit(self, X, y=None):
        self.log_cols_ = [c for c in X.columns
                          if any(h in c for h in _LOG_HINTS) and pd.api.types.is_numeric_dtype(X[c])]
        return self

    def transform(self, X):
        X = X.copy()
        if {"pct_ant", "meta_alvo"} <= set(X.columns):
            X["gap_ant"] = X["pct_ant"] - X["meta_alvo"]
            X["evolucao_req"] = (X["meta_alvo"] - X["pct_ant"]).clip(lower=0)
            X["sem_historico"] = X["pct_ant"].isna().astype(int)
        pib = next((c for c in X.columns if c.endswith("pib_mil_reais")), None)
        pop = next((c for c in X.columns if "populacao_residente" in c), None)
        extra = []
        if pib and pop:
            X["pib_per_capita"] = (X[pib] * 1000) / X[pop].replace(0, np.nan)
            extra = ["pib_per_capita"]
        for c in [*self.log_cols_, *extra]:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X


class UFRelativeStats(BaseEstimator, TransformerMixin):
    """Posição do município relativa à média da sua UF (aprendida só no treino)."""

    def fit(self, X, y=None):
        self.cols_ = [c for c in ("pct_ant", "meta_alvo") if c in X.columns]
        self.medias_ = {c: X.groupby("sigla_uf")[c].mean() for c in self.cols_}
        self.global_ = {c: X[c].mean() for c in self.cols_}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            media_uf = X["sigla_uf"].map(self.medias_[c]).fillna(self.global_[c])
            X[f"{c}_rel_uf"] = X[c] - media_uf
        return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove colunas quase duplicadas (|correlação| > limite). Decidido só com o treino de cada fold.

    Reduz overfitting: o Atlas, por exemplo, traz dezenas de variáveis quase idênticas (IDHM e seus
    componentes, taxas de pobreza em várias faixas) e os indicadores de ausência repetem o mesmo padrão.
    """

    def __init__(self, limite: float = 0.90):
        self.limite = limite

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        manter = np.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            if manter[i]:
                manter[i + 1:][c[i, i + 1:] > self.limite] = False
        self.mask_ = manter
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mask_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)[self.mask_]

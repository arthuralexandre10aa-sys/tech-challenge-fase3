"""Interpretabilidade: Feature Importance, Permutation Importance e SHAP."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.config import IMAGES, RANDOM_STATE
from src.modeling.pipeline import nomes_features


def importancia_nativa(pipe) -> pd.DataFrame:
    modelo = pipe.named_steps["modelo"]
    nomes = nomes_features(pipe)
    if hasattr(modelo, "feature_importances_"):
        v = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        v = np.abs(modelo.coef_.ravel())
    else:
        return pd.DataFrame(columns=["feature", "importancia"])
    return pd.DataFrame({"feature": nomes, "importancia": v}).sort_values("importancia", ascending=False)


def importancia_permutacao(pipe, X_te, y_te, n_repeats: int = 20) -> pd.DataFrame:
    """Importância por permutação nas colunas ORIGINAIS (antes do pré-processamento)."""
    r = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": X_te.columns, "queda_roc_auc": r.importances_mean,
                          "desvio": r.importances_std})
            .sort_values("queda_roc_auc", ascending=False))


def shap_valores(pipe, X_te: pd.DataFrame, amostra: int = 800):
    X_s = X_te.sample(min(amostra, len(X_te)), random_state=RANDOM_STATE)
    pre = pipe[:-1]
    Xt = pre.transform(X_s)
    nomes = nomes_features(pipe)
    modelo = pipe.named_steps["modelo"]
    if hasattr(modelo, "feature_importances_"):
        sv = shap.TreeExplainer(modelo).shap_values(Xt)
    else:
        sv = shap.LinearExplainer(modelo, pre.transform(X_te)).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv, np.asarray(Xt), nomes


def salva_shap(sv, Xt, nomes, arquivo: str = "shap_summary.png"):
    IMAGES.mkdir(exist_ok=True)
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=nomes, show=False, max_display=15)
    plt.title("SHAP: impacto de cada variável no risco de não atingir a meta")
    plt.savefig(IMAGES / arquivo, dpi=140, bbox_inches="tight")
    plt.close()
    resumo = (pd.DataFrame({"feature": nomes, "shap_medio_abs": np.abs(sv).mean(axis=0)})
              .sort_values("shap_medio_abs", ascending=False))
    return str(IMAGES / arquivo), resumo

"""Funções de visualização (salvam PNG em ``images/``)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.config import IMAGES, TARGET

sns.set_theme(style="whitegrid", context="notebook")


def _salva(fig, nome: str) -> str:
    IMAGES.mkdir(exist_ok=True)
    caminho = IMAGES / nome
    fig.savefig(caminho, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(caminho)


def distribuicao_alfabetizacao(ind: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ano, g in ind.groupby("ano"):
        sns.kdeplot(g["percentual_alfabetizado"].dropna(), label=str(ano), fill=True, alpha=.3, ax=ax)
    ax.set(title="Distribuição do % de alfabetizados por município", xlabel="% alfabetizados", ylabel="densidade")
    ax.legend(title="ano")
    return _salva(fig, "eda_distribuicao_alfabetizacao.png")


def alfabetizacao_por_regiao(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordem = df.groupby("regiao")["pct_alvo"].median().sort_values().index
    sns.boxplot(data=df, x="regiao", y="pct_alvo", order=ordem, ax=ax, color="#7fb3d5")
    ax.set(title="% alfabetizados em 2024 por região", xlabel="", ylabel="% alfabetizados")
    return _salva(fig, "eda_alfabetizacao_por_regiao.png")


def risco_por_uf(df: pd.DataFrame) -> str:
    t = df.groupby("sigla_uf")[TARGET].agg(["mean", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(t.index, t["mean"] * 100, color=sns.color_palette("RdYlGn_r", len(t)))
    ax.axvline(df[TARGET].mean() * 100, ls="--", c="k", lw=1, label="média Brasil")
    ax.set(title="% de municípios que NÃO atingiram a meta (2024)", xlabel="% de municípios", ylabel="UF")
    ax.legend()
    return _salva(fig, "eda_risco_por_uf.png")


def correlacoes(df: pd.DataFrame, cols: list[str]) -> str:
    c = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(6, .6 * len(cols)), max(5, .55 * len(cols))))
    sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlação entre variáveis numéricas")
    return _salva(fig, "eda_correlacoes.png")


def relacao_ant_alvo(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(data=df, x="pct_ant", y="pct_alvo", hue=df[TARGET].map({0: "atingiu", 1: "não atingiu"}),
                    alpha=.5, s=14, ax=ax)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set(title="% alfabetizados: 2023 x 2024", xlabel="2023", ylabel="2024")
    return _salva(fig, "eda_2023_vs_2024.png")


def curvas_roc_pr(y, proba) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax[0])
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax[1])
    ax[0].set_title("Curva ROC (teste)")
    ax[1].set_title("Curva Precisão x Recall (teste)")
    return _salva(fig, "modelo_roc_pr.png")


def matriz_confusao(y, proba, limiar: float = .5) -> str:
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ConfusionMatrixDisplay.from_predictions(y, (np.asarray(proba) >= limiar).astype(int), ax=ax,
                                            display_labels=["atingiu", "em risco"], colorbar=False)
    ax.set_title(f"Matriz de confusão (limiar {limiar:.2f})")
    return _salva(fig, "modelo_matriz_confusao.png")


def importancias(df_imp: pd.DataFrame, coluna: str, titulo: str, nome: str, topo: int = 15) -> str:
    d = df_imp.sort_values(coluna, ascending=True).tail(topo)
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, .38 * len(d))))
    ax.barh(d["feature"], d[coluna], color="#5499c7")
    ax.set(title=titulo, xlabel=coluna)
    return _salva(fig, nome)


def clusters_pca(coords: np.ndarray, rotulos: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=rotulos, palette="tab10", s=14, alpha=.7, ax=ax)
    ax.set(title="Grupos de municípios (K-Means, projeção PCA)", xlabel="PC1", ylabel="PC2")
    ax.legend(title="grupo")
    return _salva(fig, "cluster_pca.png")


def composicao_clusters(tab: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (tab.div(tab.sum(axis=1), axis=0) * 100).plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set(title="Composição dos grupos por região", ylabel="% dos municípios do grupo", xlabel="grupo")
    ax.legend(title="região", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _salva(fig, "cluster_composicao_regiao.png")

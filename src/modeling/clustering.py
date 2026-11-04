"""Agrupamento de municípios com padrões semelhantes (K-Means) para a análise regional."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TARGET


def colunas_cluster(df: pd.DataFrame) -> list[str]:
    base = ["pct_ant", "pct_alvo", "prof_ant", "prof_alvo"]
    ext = [c for c in df.columns if c.startswith("ext_") and pd.api.types.is_numeric_dtype(df[c])
           and df[c].notna().mean() > .6]
    return [c for c in base if c in df.columns] + ext


def agrupa(df: pd.DataFrame, k_min: int = 2, k_max: int = 8):
    cols = colunas_cluster(df)
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = prep.fit_transform(df[cols])
    sil = {}
    for k in range(k_min, k_max + 1):
        rot = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)
        sil[k] = silhouette_score(Z, rot, sample_size=min(3000, len(Z)), random_state=RANDOM_STATE)
    k_bom = max(sil, key=sil.get)
    km = KMeans(k_bom, n_init=10, random_state=RANDOM_STATE).fit(Z)
    out = df.copy()
    out["grupo"] = km.labels_
    coords = PCA(2, random_state=RANDOM_STATE).fit_transform(Z)
    return out, coords, pd.Series(sil, name="silhouette"), cols


def perfil(df_g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    p = df_g.groupby("grupo")[cols].median()
    p.insert(0, "n_municipios", df_g.groupby("grupo").size())
    p.insert(1, "taxa_risco", df_g.groupby("grupo")[TARGET].mean())
    return p.round(3)

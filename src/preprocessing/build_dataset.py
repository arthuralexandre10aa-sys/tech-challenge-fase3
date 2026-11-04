"""Monta o dataset de modelagem a partir da Gold (Fase 2) + fontes externas.

Desenho temporal (evita leakage):
    features  = desempenho em ANO_ANT (2023) + contexto territorial/socioeconômico/educacional
                (a meta de ANO_ALVO define o alvo e fica FORA das features; ver config.USAR_META_NO_MODELO)
    alvo      = município NÃO atingiu a meta em ANO_ALVO (2024)

Também corrige um bug herdado da Fase 2: onde não existe meta (todo 2023),
``atingiu_meta`` aparecia como False. Aqui o alvo fica nulo e a linha é descartada.

Uso:  python -m src.preprocessing.build_dataset
"""
from __future__ import annotations

import pandas as pd

from src.config import (ANO_ALVO, ANO_ANT, DATA_EXT, DATA_PROC, DATA_RAW, FORBIDDEN_FEATURES,
                        ID_COLS, META_COLS, PREFIXOS_UF_PROXY, TARGET, UF_COLS, UF_REGIAO, USAR_META_NO_MODELO,
                        USAR_UF_NO_MODELO)

NAO_FEATURES = set(ID_COLS) | FORBIDDEN_FEATURES | {"nome_uf", "ano"}


def carrega_gold() -> tuple[pd.DataFrame, pd.DataFrame]:
    ind = pd.read_parquet(DATA_RAW / "gold_indicador_por_municipio.parquet")
    mv = pd.read_parquet(DATA_RAW / "gold_meta_vs_resultado.parquet")
    for d in (ind, mv):
        d["id_municipio"] = d["id_municipio"].astype(str).str.zfill(7)
        d["ano"] = d["ano"].astype(int)
    return ind, mv


def monta_transicao(ind: pd.DataFrame, mv: pd.DataFrame, ano_ant: int, ano_alvo: int) -> pd.DataFrame:
    """Uma linha por município: desempenho em ``ano_ant`` -> resultado em ``ano_alvo``."""
    a = ind[ind["ano"] == ano_ant][["id_municipio", "proficiencia_media", "percentual_alfabetizado"]]
    a = a.rename(columns={"proficiencia_media": "prof_ant", "percentual_alfabetizado": "pct_ant"})
    b = ind[ind["ano"] == ano_alvo][["id_municipio", "nome_municipio", "sigla_uf", "nome_uf",
                                     "proficiencia_media", "percentual_alfabetizado"]]
    b = b.rename(columns={"proficiencia_media": "prof_alvo", "percentual_alfabetizado": "pct_alvo"})
    df = b.merge(a, on="id_municipio", how="left")
    m = mv[mv["ano"] == ano_alvo][["id_municipio", "meta_percentual"]].rename(
        columns={"meta_percentual": "meta_alvo"})
    df = df.merge(m, on="id_municipio", how="left")
    df["regiao"] = df["sigla_uf"].map(UF_REGIAO)
    return df


def define_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """Alvo = 1 se NÃO atingiu a meta. Sem meta ou sem resultado => descartado."""
    df = df.dropna(subset=["meta_alvo", "pct_alvo"]).copy()
    df[TARGET] = (df["pct_alvo"] < df["meta_alvo"]).astype(int)
    return df


def anexa_externas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Junta as fontes externas disponíveis em cache. Retorna (df, lista de fontes usadas)."""
    from src.preprocessing.external_sources import FONTES, caminho_cache

    usadas = []
    for nome, (_, nivel, _, _, eh_feature) in FONTES.items():
        p = caminho_cache(nome)
        if not eh_feature or not p.exists():
            continue
        ext = pd.read_parquet(p)
        chave = "sigla_uf" if nivel == "uf" else "id_municipio"
        ext = ext.rename(columns={c: f"ext_{nome}__{c}" for c in ext.columns if c != chave})
        df = df.merge(ext, on=chave, how="left")
        usadas.append(nome)
    return df, usadas


def colunas_features(df: pd.DataFrame) -> list[str]:
    """Colunas usadas pelo modelo. Com ``USAR_UF_NO_MODELO=False`` removem-se UF, região e
    fontes estaduais (que identificariam a UF indiretamente); com ``USAR_META_NO_MODELO=False``,
    a meta do ano-alvo (limiar do alvo e política de metas do estado)."""
    excluir = set(NAO_FEATURES)
    if not USAR_UF_NO_MODELO:
        excluir |= set(UF_COLS)
    if not USAR_META_NO_MODELO:
        excluir |= set(META_COLS)
    cols = [c for c in df.columns if c not in excluir]
    if not USAR_UF_NO_MODELO:
        cols = [c for c in cols if not c.startswith(PREFIXOS_UF_PROXY)]
    return cols


def verifica_leakage(cols: list[str]) -> None:
    """Falha se qualquer coluna proibida (alvo ou posterior ao alvo) estiver nas features."""
    ruins = [c for c in cols if c in FORBIDDEN_FEATURES or (c.endswith("_alvo") and c != "meta_alvo")]
    if ruins:
        raise ValueError(f"Data leakage: colunas proibidas nas features: {ruins}")


def constroi() -> tuple[pd.DataFrame, list[str]]:
    ind, mv = carrega_gold()
    df = define_alvo(monta_transicao(ind, mv, ANO_ANT, ANO_ALVO))
    df, usadas = anexa_externas(df)
    verifica_leakage(colunas_features(df))
    return df.reset_index(drop=True), usadas


def monta_projecao(ind: pd.DataFrame, metas_largas: pd.DataFrame, ano_ant: int, ano_alvo: int) -> pd.DataFrame:
    """Projeta ``ano_alvo`` (ainda sem resultado) com o desempenho de ``ano_ant`` e a meta oficial.

    ``metas_largas``: metas municipais em formato largo (colunas ``meta_alfabetizacao_<ano>``).
    """
    col = f"meta_alfabetizacao_{ano_alvo}"
    if col not in metas_largas.columns:
        raise KeyError(f"coluna {col} não existe nas metas municipais")
    metas = metas_largas[["id_municipio", col]].copy()
    metas["id_municipio"] = metas["id_municipio"].astype(str).str.zfill(7)
    metas = metas.groupby("id_municipio", as_index=False)[col].mean().rename(columns={col: "meta_alvo"})
    a = ind[ind["ano"] == ano_ant][["id_municipio", "nome_municipio", "sigla_uf", "nome_uf",
                                    "proficiencia_media", "percentual_alfabetizado"]]
    a = a.rename(columns={"proficiencia_media": "prof_ant", "percentual_alfabetizado": "pct_ant"})
    df = a.merge(metas, on="id_municipio", how="left").dropna(subset=["meta_alvo"])
    df["regiao"] = df["sigla_uf"].map(UF_REGIAO)
    return df


def main() -> None:
    df, usadas = constroi()
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROC / "modelagem.parquet", index=False)
    feats = colunas_features(df)
    print(f"Dataset: {df.shape[0]} municípios | {len(feats)} colunas candidatas a feature")
    print(f"Fontes externas usadas: {usadas or 'nenhuma (somente Gold)'}")
    print(f"Taxa de risco (não atingiu a meta): {df[TARGET].mean():.3f}")
    print(f"Nulos em pct_ant (sem histórico {ANO_ANT}): {df['pct_ant'].isna().sum()}")


if __name__ == "__main__":
    main()

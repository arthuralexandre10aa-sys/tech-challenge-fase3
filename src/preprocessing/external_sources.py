"""Coleta e padronização das fontes externas de enriquecimento.

Cada fonte vira um Parquet em ``data/external/<nome>.parquet`` com chave
``id_municipio`` (string, 7 dígitos) ou ``sigla_uf`` (fontes estaduais, ex.: PNAD).

Três tipos de coleta:
  * ``sidra``    : API pública do IBGE (sem credencial).
  * ``bigquery`` : Base dos Dados (exige projeto GCP em modo sandbox, ver .env).
  * ``manual``   : arquivo baixado pelo grupo em ``data/external/manual/``
                   (Atlas do Desenvolvimento Humano, Cadastro Único, FUNDEB, PNAD).

Uma fonte que falha NÃO derruba o pipeline: o status vai para
``reports/enriquecimento_status.csv`` e o treino usa o que estiver disponível.

Uso:
    python -m src.preprocessing.external_sources            # coleta tudo
    python -m src.preprocessing.external_sources --only ibge_territorio
    python -m src.preprocessing.external_sources --status   # só mostra o status
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from src.config import ANO_ALVO, DATA_EXT, DATA_MANUAL, REPORTS

SIDRA = "https://apisidra.ibge.gov.br/values"
TIMEOUT = 180


# ----------------------------------------------------------------- utilidades
def slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def ibge6_to_7(cod6) -> str:
    """Acrescenta o dígito verificador IBGE a um código de município de 6 dígitos."""
    s = str(int(cod6)).zfill(6)
    soma = 0
    for i, ch in enumerate(s):
        p = int(ch) * (1 if i % 2 == 0 else 2)
        soma += p // 10 + p % 10
    return s + str((10 - soma % 10) % 10)


def normaliza_id_municipio(serie: pd.Series) -> pd.Series:
    def _f(v):
        if pd.isna(v):
            return None
        s = re.sub(r"\D", "", str(v).split(".")[0])
        if len(s) == 6:
            return ibge6_to_7(s)
        return s.zfill(7) if len(s) == 7 else None
    return serie.map(_f)


def _bq(sql: str) -> pd.DataFrame:
    import basedosdados as bd
    from dotenv import load_dotenv

    load_dotenv()
    projeto = os.getenv("BD_BILLING_PROJECT_ID")
    if not projeto or projeto.startswith("SEU-"):
        raise RuntimeError("Defina BD_BILLING_PROJECT_ID no arquivo .env (ver .env.example).")
    return bd.read_sql(sql, billing_project_id=projeto)


def _colunas_existentes(dataset: str, tabela: str) -> set[str]:
    df = _bq(
        f"SELECT column_name FROM `basedosdados.{dataset}.INFORMATION_SCHEMA.COLUMNS` "
        f"WHERE table_name = '{tabela}'"
    )
    return set(df["column_name"])


def _sidra(caminho: str) -> pd.DataFrame:
    """Chama a API SIDRA e devolve DataFrame com os rótulos da 1ª linha como colunas."""
    r = requests.get(f"{SIDRA}/{caminho}", timeout=TIMEOUT)
    r.raise_for_status()
    dados = r.json()
    cab, linhas = dados[0], dados[1:]
    return pd.DataFrame(linhas).rename(columns=cab)


def _col_por_prefixo(df: pd.DataFrame, prefixo: str) -> str:
    for c in df.columns:
        if str(c).lower().startswith(prefixo.lower()):
            return c
    raise KeyError(f"coluna iniciada por '{prefixo}' não encontrada em {list(df.columns)}")


def _sidra_para_municipio(df: pd.DataFrame, prefixo_var: str = "Variável") -> pd.DataFrame:
    """Pivota a resposta SIDRA (uma linha por município x variável) para colunas."""
    c_id = _col_por_prefixo(df, "Município (Código)")
    c_var = _col_por_prefixo(df, prefixo_var)
    c_val = _col_por_prefixo(df, "Valor")
    c_ano = "Ano" if "Ano" in df.columns else _col_por_prefixo(df, "Ano")
    out = df[[c_id, c_var, c_val, c_ano]].copy()
    out.columns = ["id_municipio", "variavel", "valor", "ano"]
    out["valor"] = pd.to_numeric(out["valor"], errors="coerce")
    out["ano"] = pd.to_numeric(out["ano"], errors="coerce")
    out["variavel"] = out["variavel"].map(slug)
    out = out.sort_values("ano").groupby(["id_municipio", "variavel"], as_index=False).last()
    wide = out.pivot(index="id_municipio", columns="variavel", values="valor").reset_index()
    wide.columns.name = None
    return wide


# ------------------------------------------------------------------- coletores
def coleta_ibge_territorio() -> pd.DataFrame:
    """População residente, área e densidade demográfica (Censo 2022, SIDRA 4714)."""
    return _sidra_para_municipio(_sidra("t/4714/n6/all/v/all/p/all"))


def coleta_ibge_pib() -> pd.DataFrame:
    """PIB municipal a preços correntes (SIDRA 5938, período mais recente)."""
    df = _sidra_para_municipio(_sidra("t/5938/n6/all/v/37/p/last"))
    return df.rename(columns={c: "pib_mil_reais" for c in df.columns if c != "id_municipio"})


def coleta_inep_censo_escolar() -> pd.DataFrame:
    """Infraestrutura escolar agregada por município (Censo Escolar, ano anterior ao alvo)."""
    ano = ANO_ALVO - 1
    existentes = _colunas_existentes("br_inep_censo_escolar", "escola")
    candidatas = ["banda_larga", "equipamento_computador", "equipamento_tv",
                  "equipamento_copiadora", "equipamento_impressora", "internet",
                  "biblioteca", "laboratorio_informatica", "quadra_esportes"]
    usar = [c for c in candidatas if c in existentes]
    medias = ",\n           ".join(f"AVG(SAFE_CAST({c} AS FLOAT64)) AS pct_{c}" for c in usar)
    medias = (",\n           " + medias) if medias else ""
    sql = f"""
        SELECT id_municipio, COUNT(*) AS qtd_escolas{medias}
        FROM `basedosdados.br_inep_censo_escolar.escola`
        WHERE ano = {ano} AND tipo_situacao_funcionamento = '1'
        GROUP BY id_municipio
    """
    return _bq(sql)


def coleta_inep_alunos_avaliados() -> pd.DataFrame:
    """Número de alunos avaliados por município no ano anterior ao alvo (confiabilidade)."""
    ano = ANO_ALVO - 1
    sql = f"""
        SELECT id_municipio, COUNT(*) AS qtd_alunos_avaliados
        FROM `basedosdados.br_inep_avaliacao_alfabetizacao.alunos`
        WHERE ano = {ano}
        GROUP BY id_municipio
    """
    return _bq(sql)


def coleta_inep_ideb() -> pd.DataFrame:
    """IDEB municipal (anos iniciais, rede pública), edição mais recente <= ano anterior."""
    df = _bq(f"SELECT * FROM `basedosdados.br_inep_ideb.municipio` WHERE ano <= {ANO_ALVO - 1}")
    if "ensino" in df:
        df = df[df["ensino"].astype(str).str.contains("fundamental", case=False)]
    if "anos_escolares" in df:
        df = df[df["anos_escolares"].astype(str).str.contains("inic", case=False)]
    if "rede" in df:
        alvo = "publica" if (df["rede"].astype(str).str.lower() == "publica").any() else "total"
        df = df[df["rede"].astype(str).str.lower() == alvo]
    df = df.sort_values("ano").groupby("id_municipio", as_index=False).last()
    nums = [c for c in df.select_dtypes("number").columns if c != "ano"]
    return df[["id_municipio", *nums]]


def coleta_inep_metas_municipio() -> pd.DataFrame:
    """Metas municipais de alfabetização (formato largo), usadas para prever metas futuras."""
    return _bq("SELECT * FROM `basedosdados.br_inep_avaliacao_alfabetizacao.meta_alfabetizacao_municipio`")


def aliases_ids() -> list[str]:
    return ["codmun6", "codmun7", "cod_ibge", "codigo_municipio", "co_municipio",
            "cod_municipio", "municipio_codigo", "codigo_ibge", "id_municipio", "ano"]


def _coleta_manual(nome: str, nivel: str):
    def _f() -> pd.DataFrame:
        arqs = [p for ext in ("csv", "xlsx", "xls") for p in DATA_MANUAL.glob(f"{nome}.{ext}")]
        if not arqs:
            raise FileNotFoundError(
                f"coloque o arquivo em data/external/manual/{nome}.csv (ver README da pasta)")
        p = arqs[0]
        df = pd.read_csv(p, sep=None, engine="python") if p.suffix == ".csv" else pd.read_excel(p)
        df.columns = [slug(c) for c in df.columns]
        if nivel == "uf":
            col = next((c for c in ("sigla_uf", "uf") if c in df.columns), None)
            if col is None:
                raise KeyError("arquivo estadual precisa da coluna 'sigla_uf'")
            df = df.rename(columns={col: "sigla_uf"})
            df["sigla_uf"] = df["sigla_uf"].astype(str).str.upper().str.strip()
            chave = "sigla_uf"
        else:
            col = next((c for c in aliases_ids() if c in df.columns and c != "ano"), None)
            if col is None:
                raise KeyError(f"nenhuma coluna de código de município entre {aliases_ids()}")
            df["id_municipio"] = normaliza_id_municipio(df[col])
            df = df.dropna(subset=["id_municipio"])
            chave = "id_municipio"
        nums = [c for c in df.columns if c != chave and pd.api.types.is_numeric_dtype(df[c])
                and c not in aliases_ids()]
        return df[[chave, *nums]].drop_duplicates(chave)
    return _f


# Catálogo: nome -> (tipo, nível, função, descrição, é feature?)
FONTES = {
    "ibge_territorio":       ("sidra",    "municipio", coleta_ibge_territorio,
                              "IBGE: população, área e densidade (Censo 2022)", True),
    "ibge_pib":              ("sidra",    "municipio", coleta_ibge_pib,
                              "IBGE: PIB municipal", True),
    "inep_censo_escolar":    ("bigquery", "municipio", coleta_inep_censo_escolar,
                              "Censo Escolar: nº de escolas e infraestrutura", True),
    "inep_alunos_avaliados": ("bigquery", "municipio", coleta_inep_alunos_avaliados,
                              "INEP: nº de alunos avaliados (confiabilidade do indicador)", True),
    "inep_ideb":             ("bigquery", "municipio", coleta_inep_ideb,
                              "INEP: IDEB (educacional complementar)", True),
    "inep_metas_municipio":  ("bigquery", "municipio", coleta_inep_metas_municipio,
                              "Metas municipais futuras (para projeções; não é feature)", False),
    "atlas_idhm":            ("manual",   "municipio", _coleta_manual("atlas_idhm", "municipio"),
                              "Atlas do Desenvolvimento Humano (IDHM e componentes)", True),
    "cadunico":              ("manual",   "municipio", _coleta_manual("cadunico", "municipio"),
                              "Cadastro Único: famílias e pobreza por município", True),
    "fundeb":                ("manual",   "municipio", _coleta_manual("fundeb", "municipio"),
                              "FUNDEB: matrículas/repasses por município", True),
    "pnad_uf":               ("manual",   "uf",        _coleta_manual("pnad_uf", "uf"),
                              "PNAD Contínua: indicadores socioeconômicos por UF", True),
}


def caminho_cache(nome: str) -> Path:
    return DATA_EXT / f"{nome}.parquet"


def coleta(nome: str) -> dict:
    tipo, nivel, fn, desc, _ = FONTES[nome]
    reg = {"fonte": nome, "tipo": tipo, "nivel": nivel, "descricao": desc,
           "status": "ok", "linhas": 0, "colunas": 0, "erro": ""}
    try:
        df = fn()
        if nivel == "municipio":
            df["id_municipio"] = normaliza_id_municipio(df["id_municipio"])
            df = df.dropna(subset=["id_municipio"]).drop_duplicates("id_municipio")
        df.to_parquet(caminho_cache(nome), index=False)
        reg.update(linhas=len(df), colunas=df.shape[1] - 1)
    except Exception as e:  # fonte opcional: registra e segue
        reg.update(status="falhou", erro=f"{type(e).__name__}: {str(e)[:200]}")
    return reg


def status_atual() -> pd.DataFrame:
    linhas = []
    for nome, (tipo, nivel, _, desc, _) in FONTES.items():
        p = caminho_cache(nome)
        if p.exists():
            d = pd.read_parquet(p)
            linhas.append({"fonte": nome, "tipo": tipo, "nivel": nivel, "descricao": desc,
                           "status": "ok", "linhas": len(d), "colunas": d.shape[1] - 1, "erro": ""})
        else:
            linhas.append({"fonte": nome, "tipo": tipo, "nivel": nivel, "descricao": desc,
                           "status": "ausente", "linhas": 0, "colunas": 0, "erro": ""})
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="coletar só estas fontes")
    ap.add_argument("--status", action="store_true", help="apenas mostrar o status do cache")
    a = ap.parse_args()
    DATA_EXT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    if a.status:
        res = status_atual()
    else:
        nomes = a.only or list(FONTES)
        res = pd.DataFrame([coleta(n) for n in nomes])
        resto = status_atual()
        resto = resto[~resto["fonte"].isin(nomes)]
        res = pd.concat([res, resto], ignore_index=True)
    res.to_csv(REPORTS / "enriquecimento_status.csv", index=False)
    print(res[["fonte", "tipo", "status", "linhas", "colunas", "erro"]].to_string(index=False))


if __name__ == "__main__":
    main()

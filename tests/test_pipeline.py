import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.config import TARGET
from src.modeling.pipeline import candidatos, monta_pipeline
from src.preprocessing.build_dataset import define_alvo, verifica_leakage
from src.preprocessing.external_sources import ibge6_to_7, normaliza_id_municipio
from src.preprocessing.features import UFRelativeStats


def _df(n=300, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"sigla_uf": r.choice(["CE", "SP", "BA"], n), "regiao": r.choice(["N", "S"], n),
                      "pct_ant": r.uniform(0, 100, n), "prof_ant": r.normal(750, 20, n),
                      "meta_alvo": r.uniform(40, 80, n)})
    d.loc[r.choice(n, 30, replace=False), "pct_ant"] = np.nan
    d[TARGET] = ((d["pct_ant"].fillna(50) - d["meta_alvo"] + r.normal(0, 10, n)) < 0).astype(int)
    return d


def test_alvo_descarta_sem_meta():
    d = pd.DataFrame({"pct_alvo": [50., 60., 70.], "meta_alvo": [40., np.nan, 80.]})
    out = define_alvo(d)
    assert len(out) == 2 and out[TARGET].tolist() == [0, 1]


def test_leakage_barrado():
    with pytest.raises(ValueError):
        verifica_leakage(["pct_ant", "pct_alvo"])
    verifica_leakage(["pct_ant", "meta_alvo", "sigla_uf"])


def test_pipeline_lida_com_nulos_e_categoria_nova():
    d = _df()
    feats = [c for c in d.columns if c != TARGET]
    Xtr, Xte, ytr, yte = train_test_split(d[feats], d[TARGET], random_state=0)
    Xte = Xte.copy(); Xte.iloc[0, Xte.columns.get_loc("sigla_uf")] = "XX"  # UF nunca vista
    for nome, (m, _) in candidatos().items():
        p = monta_pipeline(m).fit(Xtr, ytr)
        proba = p.predict_proba(Xte)[:, 1]
        assert np.isfinite(proba).all(), nome


def test_estatisticas_por_uf_vem_so_do_treino():
    d = _df()
    tr, te = d.iloc[:200], d.iloc[200:]
    t = UFRelativeStats().fit(tr)
    assert t.medias_["pct_ant"].equals(tr.groupby("sigla_uf")["pct_ant"].mean())
    assert "pct_ant_rel_uf" in t.transform(te).columns


def test_codigo_ibge():
    assert ibge6_to_7("230440") == "2304400"  # Fortaleza
    out = normaliza_id_municipio(pd.Series(["2304400", 230440, "x"]))
    assert out.iloc[:2].tolist() == ["2304400", "2304400"] and pd.isna(out.iloc[2])


def test_sidra_usa_nome_da_variavel_e_nao_o_codigo():
    from src.preprocessing.external_sources import _sidra_para_municipio
    raw = pd.DataFrame({
        "Município (Código)": ["1100015", "1100015", "1100015"],
        "Município": ["A - RO"] * 3,
        "Variável (Código)": ["93", "614", "6318"],
        "Variável": ["População residente", "Densidade demográfica", "Área da unidade territorial"],
        "Valor": ["1000", "5.5", "180"],
        "Ano": ["2022"] * 3,
    })
    out = _sidra_para_municipio(raw)
    assert {"populacao_residente", "densidade_demografica", "area_da_unidade_territorial"} <= set(out.columns)
    assert "93" not in out.columns


def test_uf_regiao_e_proxies_fora_das_features():
    from src.preprocessing.build_dataset import colunas_features
    d = _df()
    d["ext_pnad_uf__renda"] = 1.0
    d["ext_atlas_idhm__idhm"] = 0.7
    d["id_municipio"] = "x"; d["nome_municipio"] = "x"
    cols = colunas_features(d)
    assert "sigla_uf" not in cols and "regiao" not in cols
    assert "ext_pnad_uf__renda" not in cols          # constante por UF = "código da UF"
    assert "ext_atlas_idhm__idhm" in cols and "pct_ant" in cols
    assert "meta_alvo" not in cols                   # limiar do alvo e política de metas do estado


def test_pipeline_funciona_sem_coluna_de_uf():
    d = _df().drop(columns=["sigla_uf", "regiao"])
    feats = [c for c in d.columns if c != TARGET]
    for nome, (m, _) in candidatos().items():
        p = monta_pipeline(m).fit(d[feats], d[TARGET])
        assert np.isfinite(p.predict_proba(d[feats])[:, 1]).all(), nome
    nomes = list(monta_pipeline(candidatos()["logreg"][0]).fit(d[feats], d[TARGET])
                 .named_steps["prep"].get_feature_names_out())
    assert not any("uf" in n.lower() and "rel" in n.lower() for n in nomes)


def test_carregador_manual_usa_ano_mais_recente_ate_2023(tmp_path, monkeypatch):
    from src.preprocessing import external_sources as ex
    pd.DataFrame({"Codmun6": [230440] * 4, "ANO": [1991, 2010, 2023, 2025],
                  "IDHM": [0.3, 0.5, 0.7, 0.9]}).to_csv(tmp_path / "atlas_idhm.csv", index=False)
    monkeypatch.setattr(ex, "DATA_MANUAL", tmp_path)
    out = ex._coleta_manual("atlas_idhm", "municipio")()
    assert out["id_municipio"].tolist() == ["2304400"]   # código de 6 dígitos ganhou o dígito verificador
    assert out["idhm"].iloc[0] == 0.7                    # 2023: o mais recente <= ANO_ANT (2025 é futuro)


def test_carregador_manual_escolhe_aba_com_codigo(tmp_path, monkeypatch):
    from src.preprocessing import external_sources as ex
    with pd.ExcelWriter(tmp_path / "cadunico.xlsx") as w:
        pd.DataFrame({"nota": ["texto"]}).to_excel(w, sheet_name="Leia-me", index=False)
        pd.DataFrame({"cod_ibge": [2304400, 3550308], "familias": [10, 20]}).to_excel(
            w, sheet_name="dados", index=False)
    monkeypatch.setattr(ex, "DATA_MANUAL", tmp_path)
    out = ex._coleta_manual("cadunico", "municipio")()
    assert sorted(out["id_municipio"]) == ["2304400", "3550308"] and "familias" in out.columns


def test_filtro_de_correlacao_remove_duplicadas_e_so_usa_o_treino():
    from src.preprocessing.features import CorrelationFilter
    r = np.random.default_rng(1)
    a = r.normal(size=500); c = r.normal(size=500)
    X = np.c_[a, a * 3 + r.normal(scale=0.01, size=500), c]
    f = CorrelationFilter(0.9).fit(X)
    assert f.mask_.tolist() == [True, False, True]
    assert f.transform(X).shape == (500, 2)
    assert f.get_feature_names_out(["a", "a2", "c"]).tolist() == ["a", "c"]


def test_regra_do_1_erro_padrao_escolhe_o_que_menos_memoriza():
    from src.modeling.train import escolhe_modelo
    cmp = pd.DataFrame({"modelo": ["gradient_boosting", "random_forest", "logreg", "dummy"],
                        "cv_roc_auc": [0.730, 0.726, 0.700, 0.5], "cv_roc_auc_std": [0.010, 0.010, 0.010, 0.0],
                        "gap_treino_cv": [0.10, 0.05, 0.01, np.nan]})
    assert escolhe_modelo(cmp)[0] == "random_forest"      # dentro de 1 EP e com menor gap; logreg ficou fora
    cmp.loc[cmp.modelo == "random_forest", "cv_roc_auc"] = 0.720
    assert escolhe_modelo(cmp)[0] == "gradient_boosting"  # rf saiu da margem


def test_modelo_sem_meta_funciona_e_derivadas_dependentes_somem():
    d = _df()
    feats = [c for c in d.columns if c not in (TARGET, "meta_alvo", "sigla_uf", "regiao")]
    p = monta_pipeline(candidatos()["logreg"][0]).fit(d[feats], d[TARGET])
    assert np.isfinite(p.predict_proba(d[feats])[:, 1]).all()
    from src.modeling.pipeline import nomes_features
    assert not any("meta" in n or "gap_ant" in n for n in nomes_features(p))

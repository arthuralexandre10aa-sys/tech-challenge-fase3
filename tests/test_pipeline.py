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

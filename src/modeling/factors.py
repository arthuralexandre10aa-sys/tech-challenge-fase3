"""Fatores associados ao NÍVEL de alfabetização (regressão) — complementa o modelo de risco.

Por quê: o modelo de risco prevê "atingir a meta", e as metas são calibradas pelo desempenho-base de cada
município (e do estado), então o contexto socioeconômico quase não aparece nele. Para responder
"quais fatores mais impactam a alfabetização?" modelamos o PERCENTUAL de alfabetizados em 2024.

Três variantes:
  * ``contexto``                : todas as variáveis externas (socioeconômicas, territoriais, educacionais);
  * ``contexto_sem_desempenho`` : o mesmo SEM o IDEB/SAEB. As notas do SAEB medem aprendizagem, ou seja, são outro
                                  indicador de desempenho educacional; ficam fora para isolar os fatores de contexto;
  * ``contexto_mais_historico`` : contexto + desempenho de 2023 (pct_ant, prof_ant).
UF e região ficam de fora (mesma decisão do modelo de risco). Associação não é causalidade.

Uso:  python -m src.modeling.factors
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline

from src.config import DATA_PROC, IMAGES, PREFIXOS_UF_PROXY, RANDOM_STATE, REPORTS, TEST_SIZE
from src.evaluation import interpretability as interp
from src.modeling.pipeline import LIMITE_CORRELACAO, preprocessador
from src.preprocessing.features import CorrelationFilter, DerivedFeatures
from src.visualization import plots

warnings.filterwarnings("ignore")
ALVO = "pct_alvo"  # % de alfabetizados em 2024


def pipeline_reg(modelo) -> Pipeline:
    return Pipeline([("derivadas", DerivedFeatures()), ("prep", preprocessador()),
                     ("filtro", CorrelationFilter(LIMITE_CORRELACAO)), ("modelo", modelo)])


def variantes(df: pd.DataFrame) -> dict[str, list[str]]:
    ext = [c for c in df.columns if c.startswith("ext_") and not c.startswith(PREFIXOS_UF_PROXY)]
    if not ext:
        raise SystemExit("Nenhuma fonte externa em modelagem.parquet: rode external_sources e build_dataset.")
    hist = [c for c in ("pct_ant", "prof_ant") if c in df.columns]
    sem_desempenho = [c for c in ext if not c.startswith("ext_inep_ideb__")]
    return {"contexto": ext, "contexto_sem_desempenho": sem_desempenho, "contexto_mais_historico": ext + hist}


def main() -> None:
    REPORTS.mkdir(exist_ok=True); IMAGES.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA_PROC / "modelagem.parquet")
    y = df[ALVO]
    tr, te = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    linhas, melhor_por_variante = [], {}

    for var, cols in variantes(df).items():
        X = df[cols]
        modelos = {
            "ridge": GridSearchCV(pipeline_reg(Ridge()), {"modelo__alpha": [1, 10, 100, 300, 1000]},
                                  cv=cv, scoring="r2", n_jobs=-1),
            "gradient_boosting": RandomizedSearchCV(
                pipeline_reg(GradientBoostingRegressor(random_state=RANDOM_STATE)),
                {"modelo__n_estimators": [100, 200, 300], "modelo__learning_rate": [0.02, 0.05, 0.1],
                 "modelo__max_depth": [2, 3], "modelo__min_samples_leaf": [20, 40, 80],
                 "modelo__subsample": [0.5, 0.7, 0.85], "modelo__max_features": [0.3, 0.6, 1.0]},
                n_iter=15, cv=cv, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1),
        }
        for nome, busca in modelos.items():
            busca.fit(X.loc[tr], y.loc[tr])
            est = busca.best_estimator_
            r2_cv = cross_val_score(clone(est), X.loc[tr], y.loc[tr], cv=cv, scoring="r2", n_jobs=-1)
            r2_uf = cross_val_score(clone(est), X.loc[tr], y.loc[tr], cv=GroupKFold(5),
                                    groups=df.loc[tr, "sigla_uf"], scoring="r2", n_jobs=-1)
            p_te = est.predict(X.loc[te])
            linhas.append({"variante": var, "modelo": nome, "n_variaveis": len(cols),
                           "r2_cv": r2_cv.mean(), "r2_cv_std": r2_cv.std(), "r2_por_uf": r2_uf.mean(),
                           "r2_treino": r2_score(y.loc[tr], est.predict(X.loc[tr])),
                           "r2_teste": r2_score(y.loc[te], p_te),
                           "mae_teste_pp": mean_absolute_error(y.loc[te], p_te),
                           "melhores_params": json.dumps(busca.best_params_, default=str)})
            melhor_por_variante.setdefault(var, {})[nome] = est
            print(f"{var:24s} {nome:18s} R² CV={r2_cv.mean():.3f}  por UF={r2_uf.mean():.3f}  teste={linhas[-1]['r2_teste']:.3f}")

    res = pd.DataFrame(linhas)
    res["gap_r2_treino_cv"] = res["r2_treino"] - res["r2_cv"]
    res.to_csv(REPORTS / "fatores_comparacao.csv", index=False)

    # importância nas variantes de contexto (as que respondem "quais fatores?"), com o melhor modelo por R² de CV
    resumo = {}
    for var in ("contexto", "contexto_sem_desempenho"):
        sub = res[res["variante"] == var].sort_values("r2_cv", ascending=False)
        nome_ok = sub.iloc[0]["modelo"]
        est = melhor_por_variante[var][nome_ok]
        cols = variantes(df)[var]
        perm = interp.importancia_permutacao(est, df.loc[te, cols], y.loc[te], scoring="r2")
        perm.to_csv(REPORTS / f"fatores_importancia_permutacao_{var}.csv", index=False)
        plots.importancias(perm, "queda_r2", f"Fatores do nível de alfabetização ({var}): queda de R² ({nome_ok})",
                           f"fatores_importancia_permutacao_{var}.png")
        if nome_ok == "gradient_boosting":
            sv, Xt, nomes = interp.shap_valores(est, df.loc[te, cols])
            _, shap_resumo = interp.salva_shap(sv, Xt, nomes, f"fatores_shap_summary_{var}.png",
                                               f"SHAP ({var}): impacto no % de alfabetizados")
            shap_resumo.to_csv(REPORTS / f"fatores_importancia_shap_{var}.csv", index=False)
        a_ = sub.iloc[0]
        resumo[var] = {"modelo": nome_ok, "r2_cv": a_["r2_cv"], "r2_teste": a_["r2_teste"],
                       "r2_por_uf": a_["r2_por_uf"], "mae_teste_pp": a_["mae_teste_pp"],
                       "top5_fatores": perm.head(5)[["feature", "queda_r2"]].to_dict("records")}
    b_ = res[res["variante"] == "contexto_mais_historico"].sort_values("r2_cv", ascending=False).iloc[0]
    resumo["contexto_mais_historico"] = {"modelo": b_["modelo"], "r2_cv": b_["r2_cv"], "r2_teste": b_["r2_teste"],
                                         "r2_por_uf": b_["r2_por_uf"], "mae_teste_pp": b_["mae_teste_pp"]}
    (REPORTS / "fatores_resumo.json").write_text(json.dumps(resumo, indent=2, default=float, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, default=float, ensure_ascii=False))

if __name__ == "__main__":
    main()

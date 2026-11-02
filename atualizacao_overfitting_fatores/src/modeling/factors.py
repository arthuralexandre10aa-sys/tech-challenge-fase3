"""Fatores associados ao NÍVEL de alfabetização (regressão) — complementa o modelo de risco.

Por quê: o modelo de risco prevê "atingir a meta", e as metas são calibradas pelo desempenho-base de cada
município (e do estado), então o contexto socioeconômico quase não aparece nele. Para responder
"quais fatores mais impactam a alfabetização?" modelamos o PERCENTUAL de alfabetizados em 2024.

Duas variantes (mede-se o quanto o contexto agrega além do histórico):
  * ``contexto``              : só variáveis externas (socioeconômicas, territoriais, educacionais);
  * ``contexto_mais_historico``: contexto + desempenho de 2023 (pct_ant, prof_ant).
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
    return {"contexto": ext, "contexto_mais_historico": ext + hist}


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

    # importância na variante de contexto (a que responde "quais fatores?"), com o melhor modelo por R² de CV
    sub = res[res["variante"] == "contexto"].sort_values("r2_cv", ascending=False)
    nome_ok = sub.iloc[0]["modelo"]
    est = melhor_por_variante["contexto"][nome_ok]
    cols = variantes(df)["contexto"]
    perm = interp.importancia_permutacao(est, df.loc[te, cols], y.loc[te], scoring="r2")
    perm.to_csv(REPORTS / "fatores_importancia_permutacao.csv", index=False)
    plots.importancias(perm, "queda_r2", f"Fatores do nível de alfabetização — queda de R² por permutação ({nome_ok})",
                       "fatores_importancia_permutacao.png")
    if nome_ok == "gradient_boosting":
        sv, Xt, nomes = interp.shap_valores(est, df.loc[te, cols])
        _, shap_resumo = interp.salva_shap(sv, Xt, nomes, "fatores_shap_summary.png",
                                           "SHAP: impacto de cada variável no % de alfabetizados")
        shap_resumo.to_csv(REPORTS / "fatores_importancia_shap.csv", index=False)

    # o contexto agrega informação além do histórico?
    a = res[(res.variante == "contexto") & (res.modelo == nome_ok)].iloc[0]
    b = res[(res.variante == "contexto_mais_historico")].sort_values("r2_cv", ascending=False).iloc[0]
    resumo = {"modelo_contexto": nome_ok, "r2_cv_contexto": a["r2_cv"], "r2_teste_contexto": a["r2_teste"],
              "r2_por_uf_contexto": a["r2_por_uf"], "r2_cv_com_historico": b["r2_cv"],
              "r2_teste_com_historico": b["r2_teste"], "top5_fatores": perm.head(5)[["feature", "queda_r2"]].to_dict("records")}
    (REPORTS / "fatores_resumo.json").write_text(json.dumps(resumo, indent=2, default=float, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()

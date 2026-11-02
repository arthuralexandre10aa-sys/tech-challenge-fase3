"""Gera as análises pós-treino: curvas, interpretabilidade, risco, clusters e projeção.

Uso:  python -m src.evaluation.run_evaluation
"""
from __future__ import annotations

import json
import warnings

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import ANO_ALVO, DATA_EXT, DATA_PROC, MODELS, RANDOM_STATE, REPORTS, TARGET
from src.evaluation import interpretability as interp
from src.modeling.clustering import agrupa, perfil
from src.preprocessing.build_dataset import anexa_externas, carrega_gold, monta_projecao
from src.visualization import plots

warnings.filterwarnings("ignore")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA_PROC / "modelagem.parquet")
    art = joblib.load(MODELS / "modelo.joblib")
    pipe, feats = art["pipeline"], art["features"]
    idx_te = pd.read_csv(DATA_PROC / "idx_teste.csv")["idx"].values
    X_te, y_te = df.loc[idx_te, feats], df.loc[idx_te, TARGET]

    proba_te = pipe.predict_proba(X_te)[:, 1]
    plots.curvas_roc_pr(y_te, proba_te)
    plots.matriz_confusao(y_te, proba_te, 0.5)

    nat = interp.importancia_nativa(pipe)
    nat.to_csv(REPORTS / "importancia_nativa.csv", index=False)
    plots.importancias(nat, "importancia", "Feature Importance do modelo (pós-transformação)",
                       "modelo_importancia_nativa.png")
    perm = interp.importancia_permutacao(pipe, X_te, y_te)
    perm.to_csv(REPORTS / "importancia_permutacao.csv", index=False)
    plots.importancias(perm, "queda_roc_auc", "Permutation importance (queda de ROC-AUC no teste)",
                       "modelo_importancia_permutacao.png")
    sv, Xt, nomes = interp.shap_valores(pipe, X_te)
    _, shap_resumo = interp.salva_shap(sv, Xt, nomes)
    shap_resumo.to_csv(REPORTS / "importancia_shap.csv", index=False)

    # risco por município: predição out-of-fold para TODOS (nenhum município é previsto por modelo que o viu)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    proba_oof = cross_val_predict(clone(pipe), df[feats], df[TARGET], cv=cv, method="predict_proba")[:, 1]
    risco = df[["id_municipio", "nome_municipio", "sigla_uf", "regiao", "pct_ant", "meta_alvo",
                "pct_alvo", TARGET]].copy()
    risco["prob_risco"] = proba_oof
    risco = risco.sort_values("prob_risco", ascending=False)
    risco.to_csv(REPORTS / "risco_municipios_2024.csv", index=False)
    uf = (risco.groupby("sigla_uf")
          .agg(municipios=("id_municipio", "count"), risco_medio=("prob_risco", "mean"),
               pct_em_risco_real=(TARGET, "mean"),
               pct_prob_acima_70=("prob_risco", lambda s: (s >= .7).mean()))
          .sort_values("risco_medio", ascending=False).round(3))
    uf.to_csv(REPORTS / "risco_por_uf.csv")

    df_g, coords, sil, cols = agrupa(df)
    perfil(df_g, cols).to_csv(REPORTS / "perfil_grupos.csv")
    sil.to_csv(REPORTS / "silhouette_por_k.csv")
    plots.clusters_pca(coords, df_g["grupo"].values)
    plots.composicao_clusters(pd.crosstab(df_g["grupo"], df_g["regiao"]))
    df_g[["id_municipio", "nome_municipio", "sigla_uf", "regiao", "grupo"]].to_csv(
        REPORTS / "grupos_municipios.csv", index=False)

    p_metas = DATA_EXT / "inep_metas_municipio.parquet"
    status = "metas futuras não coletadas (rode external_sources)"
    if p_metas.exists():
        ind, _ = carrega_gold()
        try:
            prox = monta_projecao(ind, pd.read_parquet(p_metas), ANO_ALVO, ANO_ALVO + 1)
            prox, _ = anexa_externas(prox)
            prox["prob_risco_proxima_meta"] = pipe.predict_proba(prox[feats])[:, 1]
            prox.sort_values("prob_risco_proxima_meta", ascending=False)[
                ["id_municipio", "nome_municipio", "sigla_uf", "pct_ant", "meta_alvo", "prob_risco_proxima_meta"]
            ].to_csv(REPORTS / f"projecao_risco_meta_{ANO_ALVO + 1}.csv", index=False)
            status = f"projeção {ANO_ALVO + 1} gerada para {len(prox)} municípios"
        except KeyError as e:
            status = f"projeção não gerada: {e}"
    print("Projeção:", status)

    resumo = {"top5_shap": shap_resumo.head(5).to_dict("records"),
              "top5_permutacao": perm.head(5).to_dict("records"),
              "k_clusters": int(df_g["grupo"].nunique()), "silhouette": sil.round(3).to_dict(),
              "ufs_maior_risco": uf.head(5).index.tolist(), "projecao": status}
    (REPORTS / "resumo_avaliacao.json").write_text(json.dumps(resumo, indent=2, default=float, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()

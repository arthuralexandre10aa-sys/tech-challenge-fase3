"""Treino, otimização e validação.

Protocolo (anti-leakage e anti-overfitting):
  1. Hold-out de teste (20%, estratificado) separado ANTES de qualquer ajuste, usado só no fim.
  2. Dentro do treino: CV estratificada (5 folds) compara candidatos e RandomizedSearchCV
     otimiza hiperparâmetros (pré-processamento no Pipeline, reajustado a cada fold).
  3. Limiar de decisão escolhido em predições out-of-fold do treino (não no teste).
  4. Robustez: validação por UF (GroupKFold) mede generalização para estados "novos".
  5. Diagnóstico de overfitting: score de treino x CV x teste.

Uso:  python -m src.modeling.train
"""
from __future__ import annotations

import json
import warnings

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (GroupKFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)

from src.config import DATA_PROC, MODELS, RANDOM_STATE, REPORTS, TARGET, TEST_SIZE
from src.evaluation.metrics import calcula_metricas, melhor_limiar_f1
from src.modeling.pipeline import candidatos, monta_pipeline
from src.preprocessing.build_dataset import colunas_features, verifica_leakage

warnings.filterwarnings("ignore")
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "accuracy": "accuracy"}
N_ITER = 12


def _tam(espaco: dict) -> int:
    n = 1
    for v in espaco.values():
        n *= len(v)
    return n


def carrega():
    df = pd.read_parquet(DATA_PROC / "modelagem.parquet")
    feats = colunas_features(df)
    verifica_leakage(feats)
    return df, feats


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df, feats = carrega()
    X, y = df[feats], df[TARGET]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"treino={len(X_tr)} teste={len(X_te)} | features={len(feats)} | risco={y.mean():.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    linhas, melhores = [], {}
    for nome, (modelo, espaco) in candidatos().items():
        pipe = monta_pipeline(modelo)
        if espaco:
            busca = RandomizedSearchCV(pipe, espaco, n_iter=min(N_ITER, _tam(espaco)), cv=cv,
                                       scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1,
                                       refit=True, return_train_score=True)
            busca.fit(X_tr, y_tr)
            est, params = busca.best_estimator_, busca.best_params_
            tr_score = busca.cv_results_["mean_train_score"][busca.best_index_]
        else:
            est, params = pipe.fit(X_tr, y_tr), {}
            tr_score = float("nan")
        res = cross_validate(clone(est), X_tr, y_tr, cv=cv, scoring=SCORING, n_jobs=-1)
        linha = {"modelo": nome, **{f"cv_{k}": res[f"test_{k}"].mean() for k in SCORING},
                 **{f"cv_{k}_std": res[f"test_{k}"].std() for k in ("roc_auc", "f1")},
                 "train_roc_auc": tr_score, "melhores_params": json.dumps(params, default=str)}
        linhas.append(linha)
        melhores[nome] = est
        print(f"{nome:18s} CV roc_auc={linha['cv_roc_auc']:.4f} f1={linha['cv_f1']:.4f}")

    cmp = pd.DataFrame(linhas).sort_values("cv_roc_auc", ascending=False)
    cmp.to_csv(REPORTS / "comparacao_modelos_cv.csv", index=False)
    escolhido = cmp.iloc[0]["modelo"]
    modelo_final = melhores[escolhido]
    print("Modelo escolhido (maior ROC-AUC em CV no treino):", escolhido)

    oof = cross_val_predict(clone(modelo_final), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    limiar = melhor_limiar_f1(y_tr, oof)

    proba_te = modelo_final.predict_proba(X_te)[:, 1]
    proba_tr = modelo_final.predict_proba(X_tr)[:, 1]
    m_te = calcula_metricas(y_te, proba_te, 0.5)
    m_te_lim = calcula_metricas(y_te, proba_te, limiar)
    m_tr = calcula_metricas(y_tr, proba_tr, 0.5)
    base = {n: calcula_metricas(y_te, e.predict_proba(X_te)[:, 1], 0.5)
            for n, e in melhores.items() if n in ("dummy", "logreg")}

    grupos = df.loc[idx_tr, "sigla_uf"]
    gk = cross_validate(clone(modelo_final), X_tr, y_tr, groups=grupos, cv=GroupKFold(5),
                        scoring=SCORING, n_jobs=-1)
    grupo_cv = {k: float(gk[f"test_{k}"].mean()) for k in SCORING}

    sat = {"modelo": escolhido, "n_treino": int(len(X_tr)), "n_teste": int(len(X_te)),
           "n_features_brutas": len(feats), "taxa_risco": float(y.mean()),
           "limiar_otimo_oof": limiar, "teste_limiar_0.5": m_te, "teste_limiar_otimo": m_te_lim,
           "treino_limiar_0.5": m_tr, "baselines_teste": base, "cv_por_uf_treino": grupo_cv,
           "gap_roc_auc_treino_vs_teste": m_tr["roc_auc"] - m_te["roc_auc"]}
    (REPORTS / "metricas_finais.json").write_text(json.dumps(sat, indent=2, default=float))
    pd.DataFrame({"idx": idx_te, "y_true": y_te.values, "proba_risco": proba_te}).to_csv(
        REPORTS / "predicoes_teste.csv", index=False)
    joblib.dump({"pipeline": modelo_final, "features": feats, "limiar": limiar}, MODELS / "modelo.joblib")
    pd.Series(idx_te).to_csv(DATA_PROC / "idx_teste.csv", index=False, header=["idx"])
    print(json.dumps({k: sat[k] for k in ("modelo", "gap_roc_auc_treino_vs_teste", "cv_por_uf_treino")}, indent=2))
    print("TESTE @0.5:", {k: round(v, 4) for k, v in m_te.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()

"""Ablação do modelo de risco: quanto cada bloco de variáveis contribui?

A meta (``meta_alvo``) foi retirada do modelo final (config.USAR_META_NO_MODELO=False); a ablação mede o que
essa decisão custou, comparando com um modelo de referência que ainda a inclui.

Cenários (mesma divisão treino/teste, mesmo pipeline e mesma busca do ``train.py``):
  * com_meta_referencia : todas as variáveis, incluindo a meta (não é o modelo final);
  * modelo_final        : sem ``meta_alvo`` (o limiar do alvo e a política de metas do estado);
  * so_contexto         : sem meta e sem histórico, apenas contexto territorial/educacional/socioeconômico;
  * contexto_sem_ideb   : o anterior sem IDEB/SAEB (que medem aprendizagem): só fatores socioeconômicos e estruturais.

Uso:  python -m src.modeling.ablation
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupKFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)

from src.config import RANDOM_STATE, REPORTS, TARGET, TEST_SIZE
from src.modeling.pipeline import candidatos, monta_pipeline
from src.modeling.train import carrega

warnings.filterwarnings("ignore")
CENARIOS = {
    "com_meta_referencia": [],
    "modelo_final": ["meta_alvo"],
    "so_contexto": ["meta_alvo", "prof_ant", "pct_ant"],
    "contexto_sem_ideb": ["meta_alvo", "prof_ant", "pct_ant", "PREFIXO:ext_inep_ideb__"],
}


def main() -> None:
    df, feats = carrega()
    feats = feats + [c for c in ("meta_alvo",) if c in df.columns and c not in feats]  # só para a referência
    y = df[TARGET]
    idx_tr, idx_te = train_test_split(df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cands = candidatos()
    linhas = []
    for cen, drop in CENARIOS.items():
        prefixos = tuple(d.split(":", 1)[1] for d in drop if d.startswith("PREFIXO:"))
        cols = [c for c in feats if c not in drop and not (prefixos and c.startswith(prefixos))]
        for nome in ("logreg", "gradient_boosting"):
            modelo, espaco = cands[nome]
            busca = RandomizedSearchCV(monta_pipeline(modelo), espaco, n_iter=20, cv=cv, scoring="roc_auc",
                                       random_state=RANDOM_STATE, n_jobs=-1)
            busca.fit(df.loc[idx_tr, cols], y.loc[idx_tr])
            est = busca.best_estimator_
            uf = cross_val_score(clone(est), df.loc[idx_tr, cols], y.loc[idx_tr], cv=GroupKFold(5),
                                 groups=df.loc[idx_tr, "sigla_uf"], scoring="roc_auc", n_jobs=-1)
            linhas.append({"cenario": cen, "modelo": nome, "n_variaveis": len(cols),
                           "cv_roc_auc": busca.best_score_,
                           "teste_roc_auc": roc_auc_score(y.loc[idx_te], est.predict_proba(df.loc[idx_te, cols])[:, 1]),
                           "cv_por_uf_roc_auc": uf.mean()})
            print(f"{cen:14s} {nome:18s} CV={linhas[-1]['cv_roc_auc']:.3f} teste={linhas[-1]['teste_roc_auc']:.3f} "
                  f"por UF={linhas[-1]['cv_por_uf_roc_auc']:.3f}", flush=True)
    pd.DataFrame(linhas).to_csv(REPORTS / "ablacao_risco.csv", index=False)


if __name__ == "__main__":
    main()

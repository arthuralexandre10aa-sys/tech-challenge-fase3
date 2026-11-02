"""Métricas de avaliação (classe positiva = município em RISCO, i.e. não atinge a meta)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def calcula_metricas(y_true, proba, limiar: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "limiar": float(limiar),
        "accuracy": accuracy_score(y_true, pred),
        "precision_risco": precision_score(y_true, pred, zero_division=0),
        "recall_risco": recall_score(y_true, pred, zero_division=0),
        "f1_risco": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def melhor_limiar_f1(y_true, proba) -> float:
    """Limiar que maximiza F1 (escolhido em OOF do treino, nunca no teste)."""
    grade = np.linspace(0.2, 0.8, 61)
    f1s = [f1_score(y_true, (np.asarray(proba) >= t).astype(int), zero_division=0) for t in grade]
    return float(grade[int(np.argmax(f1s))])

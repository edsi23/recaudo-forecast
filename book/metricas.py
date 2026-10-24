# Métricas de las dos tareas. La pérdida que se optimiza:
#   - regresión: RMSE en log1p (la misma de la entrega 1)
#   - clasificación: 1 - F1 de la clase "caída" (ver cap. 9)
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

from comun import a_log


def perdida(tarea, y, pred):
    """Pérdida a minimizar en la búsqueda (menor es mejor)."""
    if tarea == "reg":
        return float(np.sqrt(np.mean((a_log(y) - a_log(pred)) ** 2)))
    return 1.0 - f1_score(y, pred, zero_division=0)


def metricas_reg(y, pred):
    """RMSE y MAE en COP y en log1p, WAPE y R².

    Parameters
    ----------
    y, pred : array-like
        Venta real y pronosticada en COP.

    Returns
    -------
    dict
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    e, el = y - pred, a_log(y) - a_log(pred)
    return {"rmse": np.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e)),
            "rmse_log": np.sqrt(np.mean(el ** 2)), "mae_log": np.mean(np.abs(el)),
            "wape": np.abs(e).sum() / np.abs(y).sum(), "r2": r2_score(y, pred),
            "r2_log": r2_score(a_log(y), a_log(pred))}


def metricas_clf(y, pred, score=None, proba=True):
    """Accuracy, precision, recall, F1 y, si hay puntaje, AUC, AP y Brier.

    Parameters
    ----------
    y, pred : array-like
        Clase real y predicha (0/1).
    score : array-like, optional
        Puntaje continuo de la clase positiva.
    proba : bool
        Si ``score`` es una probabilidad (para el Brier). La SVM da distancias.

    Returns
    -------
    dict
    """
    m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}
    if score is not None:
        m["auc"] = roc_auc_score(y, score)
        m["ap"] = average_precision_score(y, score)
        m["brier"] = brier_score_loss(y, score) if proba else np.nan
    return m


def ece(y, p, bins=10):
    """Error de calibración esperado: promedio, ponderado por la cantidad de casos,
    de |frecuencia observada - probabilidad media| en bins de igual ancho."""
    y, p = np.asarray(y), np.asarray(p)
    cortes = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, cortes[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return total

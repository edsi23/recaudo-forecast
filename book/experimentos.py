# Corre las 140 combinaciones (28 de regresión + 112 de clasificación) con validación anidada.
# Uso:  python experimentos.py            (todas, en paralelo)
#       python experimentos.py reg xgb    (filtra por tarea y modelo, para probar)
# Cada combinación se guarda en su propio archivo, así que si se corta se puede retomar.
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # un hilo por proceso: el paralelismo va entre combinaciones

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from comun import DATA, SEED
from datos import X_COLS, folds_externos, folds_internos, partir, preparar
from metricas import metricas_clf, metricas_reg
from modelos import (BALANCEOS, MODELOS_CLF, MODELOS_REG, OPTIMIZADORES, ajustar,
                     crear_modelo, puntaje)
from optimizadores import PRESUPUESTO, buscar

# las predicciones tienen ventas por negocio: van a la carpeta privada de datos
SALIDA = DATA / "processed" / "experimentos"
MAESTRA = Path(__file__).parent / "resultados" / "maestra.csv"


def combinaciones():
    """Lista de las 140 combinaciones ``(tarea, modelo, balanceo, optimizador)``."""
    reg = [("reg", m, "ninguno", o) for m, o in itertools.product(MODELOS_REG, OPTIMIZADORES)]
    clf = [("clf", m, b, o) for m, b, o in itertools.product(MODELOS_CLF, BALANCEOS, OPTIMIZADORES)]
    return reg + clf


def _evaluar(tarea, nombre, balanceo, m, X, y):
    pred = m.predict(X)
    if tarea == "reg":
        return pred, None, metricas_reg(y, pred)
    s = puntaje(m, X)
    return pred, s, metricas_clf(y, pred, s, proba=hasattr(m, "predict_proba"))


def correr(tarea, nombre, balanceo, metodo, train, test, semilla=SEED):
    """Validación anidada de una combinación + ajuste final y predicción en test.

    Bucle externo: 4 folds de 7 días (``folds_externos``). En cada uno se corre la
    búsqueda completa con 3 folds internos sobre el train de ese fold, se reentrena
    con los mejores hiperparámetros y se mide en la validación externa (esta es la
    métrica que se reporta). Al final se repite la búsqueda con todo train y se
    predice el test (3-23 sep), que solo se usa para los análisis de los cap. 12-14.

    Returns
    -------
    dict
        Resultados completos (historiales, predicciones, métricas y tiempos).
    """
    objetivo = "venta_total" if tarea == "reg" else "caida"
    X, y = train[X_COLS], train[objetivo].to_numpy()
    t0 = time.perf_counter()
    res = {"tarea": tarea, "modelo": nombre, "balanceo": balanceo, "optimizador": metodo,
           "semilla": semilla, "folds": [], "pred_val": []}

    for k, (tr, va) in enumerate(folds_externos(train["fecha"])):
        internos = folds_internos(train["fecha"].iloc[tr])
        Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
        params, valor, hist = buscar(metodo, tarea, nombre, balanceo, Xtr, ytr, internos, semilla=semilla)
        m = crear_modelo(tarea, nombre, params, balanceo, semilla)
        ajustar(m, Xtr, ytr, tarea, nombre, balanceo)
        pred, s, met = _evaluar(tarea, nombre, balanceo, m, X.iloc[va], y[va])
        res["folds"].append({"fold": k, "params": params, "perdida_interna": valor,
                             "historial": hist, "metricas": met})
        res["pred_val"].append(pd.DataFrame({"fila": va, "fold": k, "pred": pred,
                                             "score": s if s is not None else np.nan}))

    # modelo final con todo train
    internos = folds_internos(train["fecha"], dias_val=7)
    params, valor, hist = buscar(metodo, tarea, nombre, balanceo, X, y, internos, semilla=semilla)
    t_fit = time.perf_counter()
    m = crear_modelo(tarea, nombre, params, balanceo, semilla)
    ajustar(m, X, y, tarea, nombre, balanceo)
    t_fit = time.perf_counter() - t_fit
    t_pred = time.perf_counter()
    pred, s, met = _evaluar(tarea, nombre, balanceo, m, test[X_COLS], test[objetivo].to_numpy())
    t_pred = time.perf_counter() - t_pred
    res.update({"params_final": params, "historial_final": hist, "metricas_test": met,
                "pred_test": pd.DataFrame({"pred": pred, "score": s if s is not None else np.nan}),
                "pred_val": pd.concat(res["pred_val"], ignore_index=True),
                "seg_total": time.perf_counter() - t0, "seg_fit_final": t_fit, "seg_pred_test": t_pred,
                "n_evals": sum(len(f["historial"]) for f in res["folds"]) + len(hist)})
    return res


def _archivo(c):
    return SALIDA / ("__".join(c) + ".pkl")


def _tarea(c, train, test):
    if _archivo(c).exists():
        return
    res = correr(*c, train, test)
    with open(_archivo(c), "wb") as fh:
        pickle.dump(res, fh)


def fila_maestra(res):
    """Resume un resultado en una fila de la tabla maestra (media ± desv. del bucle externo)."""
    fila = {k: res[k] for k in ["tarea", "modelo", "balanceo", "optimizador", "semilla", "n_evals"]}
    met = pd.DataFrame([f["metricas"] for f in res["folds"]])
    for c in met:
        fila[f"{c}_media"] = met[c].mean()
        fila[f"{c}_desv"] = met[c].std()
    for c, v in res["metricas_test"].items():
        fila[f"{c}_test"] = v
    fila["seg_total"] = res["seg_total"]
    fila["seg_por_eval"] = sum(h["seg"] for f in res["folds"] for h in f["historial"]) / max(
        sum(len(f["historial"]) for f in res["folds"]), 1)
    fila["seg_fit_final"], fila["seg_pred_test"] = res["seg_fit_final"], res["seg_pred_test"]
    fila["params_final"] = json.dumps(res["params_final"], default=float)
    return fila


def armar_maestra():
    """Junta todos los resultados guardados en ``resultados/maestra.csv``."""
    filas = []
    for c in combinaciones():
        if _archivo(c).exists():
            with open(_archivo(c), "rb") as fh:
                filas.append(fila_maestra(pickle.load(fh)))
    tabla = pd.DataFrame(filas)
    MAESTRA.parent.mkdir(exist_ok=True)
    tabla.to_csv(MAESTRA, index=False)
    return tabla


def cargar(tarea, modelo, balanceo, optimizador):
    """Lee el resultado completo de una combinación."""
    with open(_archivo((tarea, modelo, balanceo, optimizador)), "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    train, test = partir(preparar())
    combos = combinaciones()
    if len(sys.argv) > 1:
        combos = [c for c in combos if c[0] == sys.argv[1] and (len(sys.argv) < 3 or c[1] == sys.argv[2])]
    # los más lentos primero, para repartir mejor la carga entre procesos
    orden = {"rf": 0, "xgb": 1, "knn": 2, "logistica": 3, "svm": 4, "svr": 4, "arbol": 5}
    combos.sort(key=lambda c: orden.get(c[1], 6))
    print(f"{len(combos)} combinaciones, presupuesto {PRESUPUESTO} evaluaciones por búsqueda", flush=True)
    t = time.perf_counter()
    Parallel(n_jobs=int(os.environ.get("NJOBS", 10)), backend="loky", verbose=10)(
        delayed(_tarea)(c, train, test) for c in combos)
    print(f"listo en {(time.perf_counter() - t) / 60:.1f} min", flush=True)
    print(armar_maestra().shape)

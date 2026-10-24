# Estudio aparte de los optimizadores (cap. 10): con un presupuesto más grande (60 evaluaciones)
# y varias semillas, para ver la convergencia, la diversidad del genético y la estabilidad.
# Se hace sobre XGBoost (el modelo con más hiperparámetros) en las dos tareas, con el fold
# externo 4: se busca con 3 folds internos sobre los 43 días anteriores y el mejor de cada
# búsqueda se mide en la semana 27 ago - 2 sep.
# Uso:  python estudio_optimizadores.py
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform

from comun import DATA, RMSE_LOG, SEED
from datos import X_COLS, folds_externos, folds_internos, partir, preparar
from metricas import perdida
from modelos import REJILLAS, ajustar, crear_modelo
import optimizadores as opt

SEMILLAS = [42, 7, 2026]
PRESUPUESTO_GRANDE = 60
SALIDA = DATA / "processed" / "estudio_optimizadores.pkl"
# rejilla de 60 puntos para que el grid tenga el mismo presupuesto
REJILLA_60 = {"learning_rate": [0.01, 0.03, 0.1, 0.3], "max_depth": [3, 6, 9], "n_estimators": [100, 200, 300, 400, 500]}
CONFIG = {"reg": ("xgb", "ninguno", "venta_total"), "clf": ("xgb", "balanced", "caida")}


def una_corrida(tarea, metodo, semilla, train):
    """Una búsqueda de 60 evaluaciones en el fold externo 4 y la pérdida del mejor en su validación."""
    nombre, balanceo, objetivo = CONFIG[tarea]
    tr, va = folds_externos(train["fecha"])[-1]
    X, y = train[X_COLS], train[objetivo].to_numpy()
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
    internos = folds_internos(train["fecha"].iloc[tr], dias_val=5)
    diversidad = []
    rejilla_original = REJILLAS["xgb"]
    opt.REJILLAS["xgb"] = REJILLA_60
    t = time.perf_counter()
    params, valor, hist = opt.buscar(metodo, tarea, nombre, balanceo, Xtr, ytr, internos,
                                     presupuesto=PRESUPUESTO_GRANDE, semilla=semilla, diversidad=diversidad)
    seg = time.perf_counter() - t
    opt.REJILLAS["xgb"] = rejilla_original
    m = crear_modelo(tarea, nombre, params, balanceo, semilla)
    ajustar(m, Xtr, ytr, tarea, nombre, balanceo)
    externa = perdida(tarea, y[va], m.predict(X.iloc[va]))
    return {"tarea": tarea, "metodo": metodo, "semilla": semilla, "params": params, "perdida_interna": valor,
            "perdida_externa": externa, "historial": hist, "diversidad": diversidad, "seg": seg}


def halving(tarea, train):
    """Successive Halving contra Random Search con las mismas 60 configuraciones candidatas.

    El recurso es el número de árboles de XGBoost: empieza con 20 y se multiplica por 3
    (factor de reducción) en cada ronda, quedándose con el mejor tercio.
    """
    nombre, balanceo, objetivo = CONFIG[tarea]
    tr, va = folds_externos(train["fecha"])[-1]
    X, y = train[X_COLS], train[objetivo].to_numpy()
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
    internos = folds_internos(train["fecha"].iloc[tr], dias_val=5)
    pref = "regressor__modelo__" if tarea == "reg" else "modelo__"
    dist = {pref + "learning_rate": loguniform(0.01, 0.3), pref + "max_depth": randint(3, 11),
            pref + "subsample": uniform(0.6, 0.4), pref + "colsample_bytree": uniform(0.6, 0.4),
            pref + "min_child_weight": loguniform(1, 50)}
    base = crear_modelo(tarea, nombre, {"n_estimators": 540}, balanceo, SEED)
    # las mismas métricas de la búsqueda principal: RMSE en log (regresión) y F1 (clasificación)
    puntaje = RMSE_LOG if tarea == "reg" else "f1"
    peso = {}
    if tarea == "clf":
        from sklearn.utils.class_weight import compute_sample_weight
        peso = {"modelo__sample_weight": compute_sample_weight("balanced", ytr)}
    res = {}
    for etiqueta, busq in [
        ("halving", HalvingRandomSearchCV(base, dist, n_candidates=60, resource=pref + "n_estimators",
                                          min_resources=20, max_resources=540, factor=3, cv=internos,
                                          scoring=puntaje, random_state=SEED, n_jobs=1, refit=True)),
        ("random completo", RandomizedSearchCV(base, dist, n_iter=60, cv=internos, scoring=puntaje,
                                               random_state=SEED, n_jobs=1, refit=True)),
    ]:
        t = time.perf_counter()
        busq.fit(Xtr, ytr, **peso)
        seg = time.perf_counter() - t
        externa = perdida(tarea, y[va], busq.predict(X.iloc[va]))
        cv = pd.DataFrame(busq.cv_results_)
        res[etiqueta] = {"seg": seg, "mejor_interno": busq.best_score_, "perdida_externa": externa,
                         "params": busq.best_params_, "n_ajustes": len(cv) * len(internos),
                         # árboles entrenados en total: la medida de presupuesto de esta comparación
                         "arboles_totales": int(cv["n_resources"].sum() * len(internos))
                         if etiqueta == "halving" else 60 * 540 * len(internos),
                         "rondas": cv["iter"].value_counts().sort_index().to_dict() if "iter" in cv else None}
    return tarea, res


if __name__ == "__main__":
    train, _ = partir(preparar())
    tareas = [(t, m, s) for t in CONFIG for m in ["grid", "random", "bayes", "genetico"]
              for s in (SEMILLAS if m != "grid" else [SEED])]
    t0 = time.perf_counter()
    corridas = Parallel(n_jobs=10, backend="loky", verbose=5)(delayed(una_corrida)(*c, train) for c in tareas)
    halv = Parallel(n_jobs=2, backend="loky")(delayed(halving)(t, train) for t in CONFIG)
    with open(SALIDA, "wb") as fh:
        pickle.dump({"corridas": corridas, "halving": dict(halv)}, fh)
    print(f"listo en {(time.perf_counter() - t0) / 60:.1f} min")

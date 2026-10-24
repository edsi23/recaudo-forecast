# Validación anidada del modelo original (ACS) con los mismos folds externos y la misma semilla que los
# 28 modelos de regresión de la entrega 2, más la ablación y las reglas estándar de combinación.
# Uso:  python modelo_original.py
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from original_model import ArbolCombinacionSimplex, EnsambleRegimen, proyectar_simplex, resolver_pesos  # noqa: E402

from comun import DATA, SEED, a_log
from datos import X_COLS, folds_externos, partir, preparar
from experimentos import cargar
from modelos import crear_modelo

optuna.logging.set_verbosity(optuna.logging.WARNING)
SALIDA = DATA / "processed" / "modelo_original.pkl"
# los expertos son la mejor combinación de la entrega 2 de cada uno de estos modelos (con sus
# hiperparámetros elegidos en el bucle interno de cada fold)
EXPERTOS = {"rf": "grid", "xgb": "random", "ridge": "genetico", "knn": "genetico"}
REGIMEN = ["std7", "dias_sin_venta7", "media7", "lag1", "nivel", "festivo", "dia_semana", "part_efectivo"]
PRESUPUESTO = 16  # el mismo de los benchmarks


def ensamble(params_expertos, arbol=None):
    """Arma el ensamble con los expertos y sus hiperparámetros."""
    expertos = [(m, crear_modelo("reg", m, params_expertos[m])) for m in EXPERTOS]
    return EnsambleRegimen(expertos=expertos, columnas_x=X_COLS, columnas_fijas=("lag7",), columnas_regimen=REGIMEN,
                           n_bloques=4, dias_bloque=5, desacuerdo=True,
                           arbol=arbol or ArbolCombinacionSimplex(random_state=SEED))


def buscar_arbol(e, semilla=SEED):
    """Optimización bayesiana (TPE, 16 evaluaciones) de los hiperparámetros del árbol.

    Bucle interno: los pronósticos fuera de muestra se parten en el tiempo; el árbol se ajusta con los
    primeros 3/4 de los días y se mide (RMSE en log) en el último 1/4.
    """
    Z, y, G = e.oof_
    n = len(y)
    corte = int(n * 0.75)  # las filas de oof_ están en orden temporal (bloques consecutivos)
    XZ = np.column_stack([Z, G])
    historia = []

    def objetivo(trial):
        p = {"max_depth": trial.suggest_int("max_depth", 0, 4),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 100, 2000, log=True),
             "lam": trial.suggest_float("lam", 1e-2, 10.0, log=True)}
        a = ArbolCombinacionSimplex(n_expertos=Z.shape[1], random_state=semilla, **p).fit(XZ[:corte], y[:corte])
        v = float(np.sqrt(np.mean((a.predict(XZ[corte:]) - y[corte:]) ** 2)))
        historia.append({"eval": len(historia) + 1, "perdida": v, **p})
        return v

    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=semilla, n_startup_trials=5))
    estudio.optimize(objetivo, n_trials=PRESUPUESTO)
    return estudio.best_params, historia


def reglas_estandar(Zoof, yoof, Zval):
    """Reglas de combinación de referencia, ajustadas con los mismos pronósticos fuera de muestra."""
    K = Zoof.shape[1]
    ecm = ((Zoof - yoof[:, None]) ** 2).mean(0)
    w_inv = (1 / ecm) / (1 / ecm).sum()
    A, b = Zoof.T @ Zoof / len(yoof), Zoof.T @ yoof / len(yoof)
    w_simplex, _ = resolver_pesos(A, b, np.full(K, 1 / K), 1e-6)
    # stacking lineal sin restricciones (con intercepto, ridge muy leve)
    Zc = np.column_stack([np.ones(len(yoof)), Zoof])
    beta = np.linalg.solve(Zc.T @ Zc + 1e-6 * np.eye(K + 1), Zc.T @ yoof)
    return {"promedio simple": Zval.mean(1), "pesos por error global (1/ECM)": Zval @ w_inv,
            "pesos globales en el símplex": Zval @ w_simplex,
            "stacking lineal": np.column_stack([np.ones(len(Zval)), Zval]) @ beta}


def params_entrega2(k):
    """Hiperparámetros de cada experto elegidos en el bucle interno del fold k (entrega 2)."""
    return {m: cargar("reg", m, "ninguno", o)["folds"][k]["params"] for m, o in EXPERTOS.items()}


def un_fold(k, train, params=None, semilla=SEED):
    """Un fold externo: expertos con los hiperparámetros del fold, búsqueda del árbol y predicción de la
    validación externa del ACS, de la ablación y de las reglas estándar."""
    tr, va = folds_externos(train["fecha"])[k]
    Xtr, Xva = train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)
    y = train["venta_total"].to_numpy()
    params = params if params is not None else params_entrega2(k)
    t0 = time.perf_counter()
    e = ensamble(params).fit(Xtr, y[tr])
    t_expertos = time.perf_counter() - t0
    mejores, historia = buscar_arbol(e, semilla)
    e.reajustar_arbol(**mejores)
    t_fit = time.perf_counter() - t0
    t = time.perf_counter()
    M = e.matriz(Xva)
    pred_acs = e.arbol_.predict(M)
    t_pred = time.perf_counter() - t
    K = e.n_expertos_
    Zval = M[:, :K]
    Zoof, yoof, Goof = e.oof_
    variantes = {"ACS": pred_acs}
    variantes.update(reglas_estandar(Zoof, yoof, Zval))
    # ablación: se retira un componente a la vez y se reajusta con los mismos hiperparámetros
    base = dict(mejores)
    abl = {"sin árbol (profundidad 0)": {**base, "max_depth": 0}, "sin encogimiento (lam = 0)": {**base, "lam": 0.0}}
    XZ = np.column_stack([Zoof, Goof])
    for nombre, p in abl.items():
        a = ArbolCombinacionSimplex(n_expertos=K, random_state=semilla, **p).fit(XZ, yoof)
        variantes[nombre] = a.predict(M)
    # sin la variable de desacuerdo
    a = ArbolCombinacionSimplex(n_expertos=K, random_state=semilla, **base).fit(XZ[:, :-1], yoof)
    variantes["sin desacuerdo"] = a.predict(M[:, :-1])
    # sin cada experto
    for j, nombre in enumerate(e.nombres_):
        keep = [i for i in range(K) if i != j]
        a = ArbolCombinacionSimplex(n_expertos=K - 1, random_state=semilla, **base).fit(XZ[:, keep + list(range(K, XZ.shape[1]))], yoof)
        variantes[f"sin {nombre}"] = a.predict(M[:, keep + list(range(K, M.shape[1]))])
    for i, nombre in enumerate(e.nombres_):
        variantes[f"experto {nombre}"] = Zval[:, i]
    filas = pd.DataFrame({"fila": va, "fold": k, **{n: v for n, v in variantes.items()}})
    return {"fold": k, "params_arbol": mejores, "historia": historia, "pred": filas, "n_hojas": e.arbol_.n_hojas(),
            "pesos_hojas": [n["pesos"].tolist() for n in e.arbol_.nodos_ if "var" not in n],
            "arbol": e.arbol_, "nombres": e.nombres_, "seg_fit": t_fit, "seg_expertos": t_expertos,
            "seg_pred": t_pred, "n_val": len(va),
            "iteraciones": e.arbol_.iteraciones_}


if __name__ == "__main__":
    train, test = partir(preparar())
    from joblib import Parallel, delayed
    t = time.perf_counter()
    folds = Parallel(n_jobs=4, backend="loky")(delayed(un_fold)(k, train) for k in range(4))
    print(f"validación anidada: {(time.perf_counter() - t) / 60:.1f} min")
    y_log = a_log(train["venta_total"].to_numpy())
    pred = pd.concat([f["pred"] for f in folds], ignore_index=True)
    for f in folds:
        p = f["pred"]
        print(f"fold {f['fold'] + 1}: árbol {f['params_arbol']}, {f['n_hojas']} hojas | "
              + " ".join(f"{c} {np.sqrt(np.mean((y_log[p['fila']] - p[c]) ** 2)):.4f}"
                         for c in ["ACS", "experto rf", "experto xgb", "pesos globales en el símplex", "stacking lineal"]))

    # modelo final: expertos con los hiperparámetros finales de la entrega 2 y búsqueda del árbol con todo train
    params = {m: json.loads(json.dumps(cargar("reg", m, "ninguno", o)["params_final"])) for m, o in EXPERTOS.items()}
    t = time.perf_counter()
    final = ensamble(params).fit(train, train["venta_total"].to_numpy())
    mejores, historia_final = buscar_arbol(final)
    final.reajustar_arbol(**mejores)
    seg_fit_final = time.perf_counter() - t
    t = time.perf_counter()
    pred_test = final.predict(test)
    seg_pred_test = time.perf_counter() - t
    print(f"test RMSE log ACS = {np.sqrt(np.mean((a_log(test['venta_total']) - a_log(pred_test)) ** 2)):.4f}")
    with open(SALIDA, "wb") as fh:
        pickle.dump({"folds": [{k: v for k, v in f.items() if k != "pred"} for f in folds], "pred_val": pred,
                     "final": final, "params_final": mejores, "historia_final": historia_final,
                     "pred_test": pred_test, "seg_fit_final": seg_fit_final, "seg_pred_test": seg_pred_test}, fh)
    print("guardado en", SALIDA)

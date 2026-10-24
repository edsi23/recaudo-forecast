# Bloque B (entrega 3): reentrena la mejor combinación de cada modelo de regresión con todo train,
# la guarda en artifacts/ con su metadata.json y verifica la prueba de ida y vuelta.
# Uso:  python artefactos.py
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from comun import SEED
from datos import X_COLS, partir, preparar
from experimentos import armar_maestra
from metricas import metricas_reg
from modelos import ajustar, crear_modelo

ARTEFACTOS = Path(__file__).resolve().parents[1] / "artifacts"


def esquema(X):
    """Nombre, tipo y rango válido (mínimo y máximo de train) de cada columna de entrada."""
    cols = []
    for c in X.columns:
        s = X[c]
        cols.append({"nombre": c, "tipo": "entero" if c in ("dia_semana", "festivo") else "real",
                     "min": float(np.nanmin(s)), "max": float(np.nanmax(s)), "admite_nulos": bool(s.isna().any())})
    return cols


def mejores_combinaciones(maestra):
    """Para cada modelo, la combinación con menor RMSE log medio en el bucle externo."""
    reg = maestra[maestra["tarea"] == "reg"]
    return reg.loc[reg.groupby("modelo")["rmse_log_media"].idxmin()].sort_values("rmse_log_media")


if __name__ == "__main__":
    ARTEFACTOS.mkdir(exist_ok=True)
    train, test = partir(preparar())
    X, y = train[X_COLS], train["venta_total"].to_numpy()
    mejores = mejores_combinaciones(armar_maestra())
    versiones = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__}
    filas = []
    for _, f in mejores.iterrows():
        params = json.loads(f["params_final"])
        modelo = crear_modelo("reg", f["modelo"], params)
        t = time.perf_counter()
        ajustar(modelo, X, y, "reg", f["modelo"], "ninguno")
        seg_fit = time.perf_counter() - t
        ruta = ARTEFACTOS / f"{f['modelo']}.joblib"
        joblib.dump(modelo, ruta, compress=3)

        # prueba de ida y vuelta: el modelo cargado debe predecir exactamente lo mismo
        t = time.perf_counter()
        pred_mem = modelo.predict(test[X_COLS])
        seg_pred = time.perf_counter() - t
        pred_disco = joblib.load(ruta).predict(test[X_COLS])
        identico = bool(np.array_equal(pred_mem, pred_disco))
        assert identico, f"{f['modelo']}: el modelo serializado no reproduce las predicciones"

        meta = {"modelo": f["modelo"], "optimizador": f["optimizador"], "semilla": SEED, "versiones": versiones,
                "hiperparametros": params, "objetivo": "venta_total (COP), entrenado en log1p",
                "entrenado_con": f"{train['fecha'].min():%Y-%m-%d} a {train['fecha'].max():%Y-%m-%d} ({len(train)} filas)",
                "esquema_entrada": esquema(X),
                "metricas_bucle_externo": {m: {"media": float(f[f"{m}_media"]), "desv": float(f[f"{m}_desv"])}
                                           for m in ["rmse_log", "mae_log", "rmse", "mae", "wape", "r2"]},
                "metricas_test": {k: float(v) for k, v in metricas_reg(test["venta_total"], pred_mem).items()},
                "seg_entrenamiento": seg_fit, "seg_inferencia_test": seg_pred,
                "prueba_ida_y_vuelta": identico, "mb_en_disco": ruta.stat().st_size / 1e6}
        with open(ARTEFACTOS / f"{f['modelo']}.metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        filas.append({"modelo": f["modelo"], "optimizador": f["optimizador"],
                      "rmse_log": f"{f['rmse_log_media']:.3f} ± {f['rmse_log_desv']:.3f}",
                      "MB": round(meta["mb_en_disco"], 1), "ida y vuelta": identico, "seg fit": round(seg_fit, 1)})
    print(pd.DataFrame(filas).to_string(index=False))

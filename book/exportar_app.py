# Entrega 3: exporta lo que necesita el dashboard (data/ y artifacts/ en la raíz del repo).
# El dashboard nunca entrena ni optimiza: solo carga estos archivos.
# Uso:  python exportar_app.py   (después de artefactos.py, modelo_original.py y validacion_original.py)
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from artefactos import esquema  # noqa: E402
from comun import DATA, SEED, a_log  # noqa: E402
from datos import X_COLS, partir, preparar  # noqa: E402
from experimentos import armar_maestra, cargar  # noqa: E402
from metricas import metricas_reg  # noqa: E402

APP_DATA, ART, RES = RAIZ / "data", RAIZ / "artifacts", Path(__file__).parent / "resultados"
NOMBRE = {"knn": "KNN", "ridge": "Ridge", "lasso": "Lasso", "arbol": "Árbol", "rf": "Random Forest", "xgb": "XGBoost",
          "svr": "SVR lineal", "hgb": "HistGradientBoosting", "acs": "ACS (original)"}
OPT = {"grid": "Grid", "random": "Random", "bayes": "Bayes", "genetico": "Genético"}


def guardar_acs(train, test):
    """Serializa el ACS final (expertos + árbol) con su metadata y prueba de ida y vuelta."""
    r = pickle.load(open(DATA / "processed" / "modelo_original.pkl", "rb"))
    final = r["final"]
    joblib.dump(final, ART / "acs.joblib", compress=3)
    p_mem = final.predict(test)
    p_disco = joblib.load(ART / "acs.joblib").predict(test)
    assert np.array_equal(p_mem, p_disco), "el ACS serializado no reproduce las predicciones"
    res = pd.read_csv(RES / "original_resumen.csv", index_col=0).loc["ACS (original)"]
    meta = {"modelo": "acs", "nombre": "Árbol de Combinación en el Símplex (ACS)", "optimizador": "bayes (TPE, 16 evaluaciones)",
            "semilla": SEED, "versiones": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                          "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
            "hiperparametros": {"arbol": r["params_final"], "expertos": final.nombres_},
            "esquema_entrada": esquema(train[X_COLS]),
            "metricas_bucle_externo": {"rmse_log": {"media": res["RMSE log media"], "desv": res["RMSE log desv"]},
                                       "mae_log": {"media": res["MAE log media"], "desv": res["MAE log desv"]}},
            "metricas_test": {k: float(v) for k, v in metricas_reg(test["venta_total"], p_mem).items()},
            "seg_entrenamiento": r["seg_fit_final"], "seg_inferencia_test": r["seg_pred_test"],
            "prueba_ida_y_vuelta": True, "mb_en_disco": (ART / "acs.joblib").stat().st_size / 1e6}
    json.dump(meta, open(ART / "acs.metadata.json", "w"), indent=2, ensure_ascii=False, default=float)
    return final


def tabla_lideres():
    """Mejor combinación de cada modelo (bucle externo, RMSE log), el mejor HGB y el ACS."""
    res = pd.read_csv(RES / "original_resumen.csv", index_col=0)
    maestra = armar_maestra()
    reg = maestra[maestra["tarea"] == "reg"]
    mejores = reg.loc[reg.groupby("modelo")["rmse_log_media"].idxmin()]
    hgb = min(OPT, key=lambda o: np.mean([f["metricas"]["rmse_log"] for f in cargar("reg", "hgb", "ninguno", o)["folds"]]))
    filas = [(m, o) for m, o in zip(mejores["modelo"], mejores["optimizador"])] + [("hgb", hgb)]
    out = []
    for m, o in filas:
        n = f"{NOMBRE[m]} · {OPT[o]}"
        meta = json.load(open(ART / f"{m}.metadata.json")) if (ART / f"{m}.metadata.json").exists() else {}
        out.append({"clave": m, "modelo": NOMBRE[m], "optimizador": OPT[o], **res.loc[n].to_dict(),
                    "MB": meta.get("mb_en_disco", np.nan)})
    a = res.loc["ACS (original)"].to_dict()
    out.append({"clave": "acs", "modelo": NOMBRE["acs"], "optimizador": "Bayes", **a,
                "MB": (ART / "acs.joblib").stat().st_size / 1e6})
    t = pd.DataFrame(out).sort_values("RMSE log media")
    return t, hgb


def series_temporales(panel):
    """STL, ACF/PACF y pruebas de estacionariedad de la venta total diaria (en log), por grupo."""
    filas, comp = [], []
    for g, d in [("todos", panel)] + list(panel.groupby("grupo")):
        s = np.log(d.groupby("fecha")["venta_total"].sum().clip(lower=1))
        stl = STL(s, period=7, robust=True).fit()
        comp.append(pd.DataFrame({"grupo": g, "fecha": s.index, "serie": s.values, "tendencia": stl.trend.values,
                                  "estacional": stl.seasonal.values, "residuo": stl.resid.values}))
        a, pa = acf(s - stl.trend, nlags=21), pacf(s - stl.trend, nlags=21)
        adf_p = adfuller(s, autolag="AIC")[1]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
        fuerza = max(0.0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))
        filas += [{"grupo": g, "rezago": k, "acf": a[k], "pacf": pa[k], "adf_p": adf_p, "kpss_p": kpss_p,
                   "fuerza_semanal": fuerza} for k in range(22)]
    return pd.concat(comp), pd.DataFrame(filas)


if __name__ == "__main__":
    APP_DATA.mkdir(exist_ok=True)
    panel_full = pd.read_csv(DATA / "processed" / "panel_negocio_dia.csv", parse_dates=["fecha"])
    train, test = partir(preparar(panel_full))
    acs = guardar_acs(train, test)
    lideres, hgb_opt = tabla_lideres()
    # el HGB no estaba entre los 7 artefactos de la entrega 2: se reentrena igual que los demás
    from artefactos import ARTEFACTOS
    from modelos import ajustar, crear_modelo
    r = cargar("reg", "hgb", "ninguno", hgb_opt)
    mod = crear_modelo("reg", "hgb", r["params_final"]); ajustar(mod, train[X_COLS], train["venta_total"].to_numpy(), "reg", "hgb", "ninguno")
    joblib.dump(mod, ARTEFACTOS / "hgb.joblib", compress=3)
    assert np.array_equal(mod.predict(test[X_COLS]), joblib.load(ARTEFACTOS / "hgb.joblib").predict(test[X_COLS]))
    json.dump({"modelo": "hgb", "optimizador": hgb_opt, "semilla": SEED, "hiperparametros": r["params_final"],
               "esquema_entrada": esquema(train[X_COLS]), "prueba_ida_y_vuelta": True,
               "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
               "metricas_bucle_externo": {"rmse_log": {"media": float(lideres.set_index("clave").loc["hgb", "RMSE log media"]),
                                                       "desv": float(lideres.set_index("clave").loc["hgb", "RMSE log desv"])}},
               "mb_en_disco": (ARTEFACTOS / "hgb.joblib").stat().st_size / 1e6},
              open(ARTEFACTOS / "hgb.metadata.json", "w"), indent=2, ensure_ascii=False, default=float)
    lideres.loc[lideres["clave"] == "hgb", "MB"] = (ARTEFACTOS / "hgb.joblib").stat().st_size / 1e6

    # predicciones de test de cada modelo del tablero (diagnósticos) + tiempo de inferencia por 1 000 filas
    modelos = {k: joblib.load(ART / f"{k}.joblib") for k in lideres["clave"]}
    pred, ms = {}, {}
    for k, m in modelos.items():
        t = time.perf_counter(); pred[k] = m.predict(test); ms[k] = (time.perf_counter() - t) / len(test) * 1000 * 1000
    lideres["ms por 1 000 filas"] = lideres["clave"].map(ms)
    lideres.to_csv(APP_DATA / "lideres.csv", index=False)

    # intervalo de predicción conforme (split): cuantil 90 % del |error en log| de la validación externa
    y_val = None
    q = {}
    for k, row in lideres.set_index("clave").iterrows():
        if k == "acs":
            pv = pickle.load(open(DATA / "processed" / "modelo_original.pkl", "rb"))["pred_val"]
            sub = train.iloc[pv["fila"]]
            e = a_log(sub["venta_total"].to_numpy()) - pv["ACS"].to_numpy()
        else:
            rr = cargar("reg", k, "ninguno", {v: kk for kk, v in OPT.items()}[row["optimizador"]])
            sub = train.iloc[rr["pred_val"]["fila"]]
            e = a_log(sub["venta_total"].to_numpy()) - a_log(rr["pred_val"]["pred"].to_numpy())
        q[k] = {"q80": float(np.quantile(np.abs(e), 0.8)), "q90": float(np.quantile(np.abs(e), 0.9))}
    json.dump(q, open(APP_DATA / "intervalos.json", "w"), indent=2)

    # tabla de modelado para el EDA y la predicción en vivo
    grupos = pd.read_csv(RAIZ.parent / "recaudo-dash" / "src" / "datos" / "perfil.csv", index_col=0)["grupo"]
    cols = ["negocio", "fecha"] + X_COLS + ["venta_total", "caida", "med7"]
    mod_df = pd.concat([train[cols].assign(tramo="train"), test[cols].assign(tramo="test")])
    mod_df["grupo"] = mod_df["negocio"].map(grupos).fillna("sin perfil")
    mod_df.to_csv(APP_DATA / "modelado.csv.gz", index=False, float_format="%.17g")  # precisión completa: los árboles son sensibles al redondeo
    tp = test[["negocio", "fecha", "venta_total"]].copy()
    for k in pred:
        tp[k] = pred[k]
    tp.to_csv(APP_DATA / "pred_test.csv.gz", index=False, float_format="%.17g")

    # panel diario completo (EDA temporal) y series del EDA
    panel = panel_full.copy()
    panel["grupo"] = panel["negocio"].map(grupos).fillna("sin perfil")
    panel.to_csv(APP_DATA / "panel.csv.gz", index=False, float_format="%.6g")
    comp, acfs = series_temporales(panel)
    comp.to_csv(APP_DATA / "stl.csv", index=False); acfs.to_csv(APP_DATA / "acf.csv", index=False)

    # SHAP global de cada modelo del tablero (300 filas de test, mismas para todos)
    muestra = test.sample(300, random_state=SEED)
    fondo = shap.kmeans(train[X_COLS].fillna(train[X_COLS].median()), 20)
    imp = []
    for k, m in modelos.items():
        f = (lambda Z, m=m: a_log(m.predict(pd.DataFrame(Z, columns=X_COLS))))
        ex = shap.KernelExplainer(f, fondo)
        sv = ex.shap_values(muestra[X_COLS].fillna(train[X_COLS].median()).to_numpy()[:60], nsamples=200, silent=True)
        imp += [{"clave": k, "variable": c, "shap_abs": float(np.abs(sv[:, j]).mean())} for j, c in enumerate(X_COLS)]
        print("shap", k, flush=True)
    pd.DataFrame(imp).to_csv(APP_DATA / "shap_global.csv", index=False)

    # resultados del modelo original y de la comparación estadística
    for f in ["original_pruebas_rmse.csv", "original_pruebas_mae.csv", "original_mcs_rmse.csv", "original_mcs_mae.csv",
              "original_ablacion.csv", "original_resumen.csv"]:
        pd.read_csv(RES / f).to_csv(APP_DATA / f, index=False)
    r = pickle.load(open(DATA / "processed" / "modelo_original.pkl", "rb"))
    pd.DataFrame(r["historia_final"]).to_csv(APP_DATA / "original_optuna.csv", index=False)
    it = pd.Series(acs.arbol_.iteraciones_)
    it.to_frame("iteraciones").to_csv(APP_DATA / "original_iteraciones.csv", index=False)
    json.dump({"nodos": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in n.items()} for n in acs.arbol_.nodos_],
               "expertos": acs.nombres_, "regimen": list(acs.columnas_regimen) + ["desacuerdo"]},
              open(APP_DATA / "original_arbol.json", "w"), indent=2, default=float)
    print(lideres.round(4).to_string(index=False))

# Entrega 3, bloque D: comparación estadística del modelo original (ACS) contra los benchmarks, con las
# mismas filas de validación externa (mismos folds y semilla). Guarda las tablas en resultados/.
# Uso:  python validacion_original.py
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import SPA
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comun import DATA, SEED, a_log
from datos import partir, preparar
from estadistica import (cliff_delta, cohen_d, diebold_mariano, giacomini_white, ic_estacionario,
                         magnitud_cliff, model_confidence_set)
from experimentos import cargar
from modelos import MODELOS_REG, OPTIMIZADORES

RES = Path(__file__).parent / "resultados"
NOMBRE = {"knn": "KNN", "ridge": "Ridge", "lasso": "Lasso", "arbol": "Árbol", "rf": "Random Forest", "xgb": "XGBoost",
          "svr": "SVR lineal", "hgb": "HistGradientBoosting"}
OPT = {"grid": "Grid", "random": "Random", "bayes": "Bayes", "genetico": "Genético"}
PERDIDAS = {"rmse": lambda e: e ** 2, "mae": lambda e: np.abs(e)}


def cargar_todo():
    """Predicciones de la validación externa (log) de los 32 benchmarks, el ACS y sus variantes."""
    train, test = partir(preparar())
    acs = pickle.load(open(DATA / "processed" / "modelo_original.pkl", "rb"))
    pv = acs["pred_val"]
    filas = pv["fila"].to_numpy()
    sub = train.iloc[filas].reset_index(drop=True)
    preds = {"ACS (original)": pv["ACS"].to_numpy()}
    for m, o in product(MODELOS_REG + ["hgb"], OPTIMIZADORES):
        r = cargar("reg", m, "ninguno", o)
        p = r["pred_val"].set_index("fila").loc[filas, "pred"].to_numpy()
        preds[f"{NOMBRE[m]} · {OPT[o]}"] = a_log(p)
    preds["Ingenuo estacional"] = sub["lag7"].to_numpy()
    preds["Persistencia"] = sub["lag1"].to_numpy()
    variantes = {c: pv[c].to_numpy() for c in pv.columns if c not in ("fila", "fold", "ACS")}
    return train, test, sub, a_log(sub["venta_total"].to_numpy()), pv["fold"].to_numpy(), preds, variantes, acs


def resumen(y, fold, preds, tiempos):
    """Métrica por fold (media ± desv.) de cada modelo, en RMSE log y MAE log."""
    filas = []
    for n, p in preds.items():
        e = y - p
        rm = [np.sqrt(np.mean(e[fold == k] ** 2)) for k in range(4)]
        ma = [np.mean(np.abs(e[fold == k])) for k in range(4)]
        filas.append({"modelo": n, "RMSE log media": np.mean(rm), "RMSE log desv": np.std(rm, ddof=1),
                      "MAE log media": np.mean(ma), "MAE log desv": np.std(ma, ddof=1), **tiempos.get(n, {})})
    return pd.DataFrame(filas).set_index("modelo")


def pruebas(y, sub, preds, rival, metrica):
    """Pruebas de hipótesis del ACS contra el benchmark ``rival`` para una pérdida (rmse o mae)."""
    L = PERDIDAS[metrica]
    perd = {n: L(y - p) for n, p in preds.items()}
    dia = pd.DataFrame({n: pd.Series(v).groupby(sub["fecha"].to_numpy()).mean() for n, v in perd.items()})
    neg = pd.DataFrame({n: pd.Series(v).groupby(sub["negocio"].to_numpy()).mean() for n, v in perd.items()})
    if metrica == "rmse":
        neg = np.sqrt(neg)
    filas = []
    # 1) MCS sobre todos los modelos (control de data snooping)
    mcs = model_confidence_set(dia, alfa=0.10, bloque=3, reps=2000)
    p_acs = float(mcs.loc["ACS (original)", "p MCS"])
    filas.append({"prueba": "Model Confidence Set (35 modelos)",
                  "H0": "el ACS pertenece al conjunto de los mejores (igual pérdida esperada)",
                  "supuestos": "pérdida diaria estacionaria; bootstrap por bloques de 3 días",
                  "estadístico": np.nan, "p crudo": p_acs,
                  "conclusión": ("ACS en el conjunto de confianza al 90 %" if p_acs > 0.10 else "ACS eliminado")
                                + f"; único modelo en el conjunto: {mcs.index[mcs['en el conjunto']].tolist() == ['ACS (original)']}"})
    d = (dia["ACS (original)"] - dia[rival]).to_numpy()
    # 2) Diebold-Mariano con HLN
    dm, p = diebold_mariano(d)
    lo, hi = ic_estacionario(d)
    filas.append({"prueba": f"Diebold-Mariano (HLN) vs {rival}", "H0": "igual pérdida esperada (bilateral)",
                  "supuestos": "diferencia de pérdida estacionaria (ADF), h = 1", "estadístico": dm, "p crudo": p,
                  "IC 95 % (bootstrap estacionario)": f"[{lo:.4f}, {hi:.4f}]", "d de Cohen": cohen_d(d)})
    # 3) Giacomini-White
    gw, p = giacomini_white(d)
    filas.append({"prueba": f"Giacomini-White vs {rival}", "H0": "igual habilidad predictiva condicional",
                  "supuestos": "instrumentos: constante y diferencia del día anterior", "estadístico": gw, "p crudo": p,
                  "d de Cohen": cohen_d(d)})
    # 4) SPA de Hansen: ¿el ACS supera al benchmark? (benchmark = rival, alternativa = ACS)
    spa = SPA(dia[rival].to_numpy(), dia[["ACS (original)"]].to_numpy(), reps=2000, block_size=3, seed=SEED)
    spa.compute()
    filas.append({"prueba": f"SPA de Hansen (benchmark {rival})", "H0": "el ACS no es mejor que el benchmark",
                  "supuestos": "bootstrap estacionario, bloques de 3", "estadístico": np.nan,
                  "p crudo": float(spa.pvalues["consistent"])})
    # 5) Wilcoxon por negocio (los negocios son unidades independientes)
    dn = neg["ACS (original)"] - neg[rival]
    w = stats.wilcoxon(dn)
    dc = cliff_delta(neg["ACS (original)"], neg[rival])
    filas.append({"prueba": f"Wilcoxon por negocio vs {rival}", "H0": "mediana de la diferencia por negocio = 0",
                  "supuestos": "negocios independientes; diferencias simétricas", "estadístico": w.statistic,
                  "p crudo": w.pvalue, "delta de Cliff": dc, "magnitud": magnitud_cliff(dc),
                  "ACS mejor en (% negocios)": float((dn < 0).mean() * 100)})
    # 6) permutación pareada por negocio (cambio de signo)
    rng = np.random.default_rng(SEED)
    obs = dn.mean()
    sig = rng.choice([-1, 1], size=(10000, len(dn)))
    p_perm = float((np.abs((sig * dn.to_numpy()).mean(1)) >= abs(obs)).mean())
    filas.append({"prueba": f"Permutación pareada por negocio vs {rival}", "H0": "intercambiabilidad de las pérdidas",
                  "supuestos": "negocios independientes", "estadístico": obs, "p crudo": p_perm})
    t = pd.DataFrame(filas)
    t["p Holm"] = multipletests(t["p crudo"].fillna(1), method="holm")[1]
    t["p BH"] = multipletests(t["p crudo"].fillna(1), method="fdr_bh")[1]
    t["métrica"] = metrica.upper() + " log"
    return t, mcs, dia


if __name__ == "__main__":
    train, test, sub, y, fold, preds, variantes, acs = cargar_todo()
    tiempos = {"ACS (original)": {"seg entrenamiento": acs["seg_fit_final"], "seg inferencia (test)": acs["seg_pred_test"]}}
    for m, o in product(MODELOS_REG + ["hgb"], OPTIMIZADORES):
        r = cargar("reg", m, "ninguno", o)
        tiempos[f"{NOMBRE[m]} · {OPT[o]}"] = {"seg entrenamiento": r["seg_fit_final"], "seg inferencia (test)": r["seg_pred_test"]}
    tab = resumen(y, fold, preds, tiempos)
    tab.to_csv(RES / "original_resumen.csv")
    # el mejor benchmark de cada métrica según la media del bucle externo
    bench = tab.drop(index=["ACS (original)", "Ingenuo estacional", "Persistencia"])
    salida = {}
    for metrica, col in [("rmse", "RMSE log media"), ("mae", "MAE log media")]:
        rival = bench[col].idxmin()
        t, mcs, dia = pruebas(y, sub, preds, rival, metrica)
        t.to_csv(RES / f"original_pruebas_{metrica}.csv", index=False)
        mcs.to_csv(RES / f"original_mcs_{metrica}.csv")
        salida[metrica] = rival
        print(f"\n=== {metrica.upper()} log: mejor benchmark = {rival}")
        print(tab.loc[["ACS (original)", rival], [col]].round(4).to_string())
        print(t[["prueba", "estadístico", "p crudo", "p Holm"]].round(4).to_string(index=False))
    # ablación y reglas estándar
    abl = resumen(y, fold, variantes, {})
    abl.to_csv(RES / "original_ablacion.csv")
    print(abl[["RMSE log media", "MAE log media"]].round(4).to_string())

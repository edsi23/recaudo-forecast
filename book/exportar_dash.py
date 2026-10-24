# Exporta los datos que usa el dashboard (repo recaudo-dash): panel con pronósticos fuera de muestra,
# probabilidad de caída, grupo de cada negocio, tabla maestra y curvas de los optimizadores.
# Uso:  python exportar_dash.py ../../recaudo-dash/src/datos
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from comun import DATA, SEED, cargar_panel
from datos import partir, preparar
from experimentos import armar_maestra, cargar

destino = Path(sys.argv[1])
destino.mkdir(parents=True, exist_ok=True)
panel = cargar_panel()
train, test = partir(preparar(panel))

# pronósticos fuera de muestra: validación externa (6 ago - 2 sep) y test (3 - 23 sep)
reg = cargar("reg", "rf", "ninguno", "grid")
clf = cargar("clf", "xgb", "balanced", "bayes")
val = train.iloc[reg["pred_val"]["fila"]].assign(pred=reg["pred_val"]["pred"].to_numpy(), tramo="validación")
val["p_caida"] = train.iloc[clf["pred_val"]["fila"]].assign(s=clf["pred_val"]["score"].to_numpy())["s"].to_numpy()
te = test.assign(pred=reg["pred_test"]["pred"].to_numpy(), p_caida=clf["pred_test"]["score"].to_numpy(), tramo="test")
pron = pd.concat([val, te])[["negocio", "fecha", "pred", "p_caida", "caida", "med7", "tramo"]]
pron["ingenuo"] = np.expm1(pd.concat([val, te])["lag7"].to_numpy())

# grupos de negocio del cap. 4b (mismo perfil y k-means)
jul_ago = panel[panel["fecha"] < "2026-09-01"].copy()
jul_ago["y_log"] = np.log1p(jul_ago["venta_total"].clip(lower=0))
cv = jul_ago[jul_ago["venta_total"] > 0]
g = cv.groupby("negocio")
perfil = pd.DataFrame({"nivel": g["y_log"].mean(), "volatilidad": g["y_log"].std(),
                       "pct_sin_venta": jul_ago.groupby("negocio")["sin_registro"].mean() * 100,
                       "n_fuentes": jul_ago.groupby("negocio")["n_fuentes"].mean()})
f = cv.assign(fin=cv["fecha"].dt.dayofweek >= 5).groupby(["negocio", "fin"])["venta_total"].mean().unstack()
perfil["finde"] = np.log(f[True] / f[False])
tot = jul_ago.groupby("negocio")[[c for c in panel if c.startswith("pm_")] + ["venta_total"]].sum()
for c in [c for c in panel if c.startswith("pm_")]:
    perfil["part_" + c[3:]] = tot[c] / tot["venta_total"]
perfil = perfil.dropna()
km = KMeans(n_clusters=5, n_init=20, random_state=SEED).fit(StandardScaler().fit_transform(perfil))
orden = perfil.groupby(km.labels_)["nivel"].mean().sort_values().index
nombres = ["G1 casi inactivos", "G2 KCO", "G3 dos fuentes", "G4 tiendas", "G5 EDS"]
perfil["grupo"] = pd.Series(km.labels_, index=perfil.index).map({k: nombres[i] for i, k in enumerate(orden)})

hist = panel[["negocio", "fecha", "venta_total", "sin_registro"] + [c for c in panel if c.startswith("pm_")]]
hist = hist.merge(pron, on=["negocio", "fecha"], how="left")
hist = hist.merge(perfil[["grupo"]], left_on="negocio", right_index=True, how="left")
hist["grupo"] = hist["grupo"].fillna("sin perfil")
hist.to_csv(destino / "panel.csv.gz", index=False, float_format="%.4f")
perfil.round(4).to_csv(destino / "perfil.csv")
armar_maestra().to_csv(destino / "maestra.csv", index=False)

# curvas anytime del estudio de optimizadores (cap. 10)
with open(DATA / "processed" / "estudio_optimizadores.pkl", "rb") as fh:
    est = pickle.load(fh)
filas = []
for c in est["corridas"]:
    h = pd.DataFrame(c["historial"])
    filas.append(pd.DataFrame({"tarea": c["tarea"], "metodo": c["metodo"], "semilla": c["semilla"],
                               "eval": h["eval"], "perdida": h["perdida"], "mejor": h["perdida"].cummin()}))
pd.concat(filas).to_csv(destino / "anytime.csv", index=False)
print("listo:", sorted(p.name for p in destino.iterdir()))

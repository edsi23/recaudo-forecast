# Cosas que se usan en varios capítulos: estilo de gráficos, fechas clave,
# variables del modelo y folds temporales. Así no se copian a mano en cada notebook.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

SEED = 42
DATA = Path("../../Data")

# fechas clave (cap. 2 y 3)
INICIO = pd.Timestamp("2026-07-01")
CORTE = pd.Timestamp("2026-09-02")  # último día de train
FIN = pd.Timestamp("2026-09-23")    # último día de test
FESTIVOS = pd.to_datetime(["2026-07-20", "2026-08-07", "2026-08-17"])

# paleta del libro
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
AZUL, NARANJA, VERDE, AMARILLO = PALETA[:4]
GRIS = "#8a8983"
MESES = {7: "jul", 8: "ago", 9: "sep"}
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=PALETA),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.edgecolor": GRIS, "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.axisbelow": True,
    })
    pd.set_option("display.float_format", "{:,.2f}".format)
    pd.set_option("display.max_columns", 30)
    np.random.seed(SEED)


def fechas_eje(ax, fechas):
    # pone ticks el 1 y el 15 de cada mes, en español
    ticks = [f for f in pd.to_datetime(fechas) if f.day in (1, 15)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{f.day} {MESES[f.month]}" for f in ticks])


def cargar_panel():
    return pd.read_csv(DATA / "processed" / "panel_negocio_dia.csv", parse_dates=["fecha"])


# ---- variables del modelo (cap. 6 y 7) ----
MEDIOS = ["efectivo", "datafono", "rumbo", "mi_empresa", "credito_clientes"]
REZAGOS = [1, 7, 14]
VARIABLES_NUM = (["lag1", "lag7", "lag14", "media7", "std7", "nivel"]
                 + [f"part_{m}" for m in MEDIOS] + ["dias_sin_venta7", "festivo"])
VARIABLES_CAT = ["dia_semana"]


def crear_variables(panel):
    # ojo: todo usa datos hasta el día anterior (shift), nunca el mismo día
    d = panel.sort_values(["negocio", "fecha"]).copy()

    # el panel no tiene huecos dentro de cada negocio, así que correr 1 fila = correr 1 día
    salto = d.groupby("negocio")["fecha"].diff().dt.days.dropna()
    assert (salto == 1).all(), "hay días faltantes dentro de un negocio"

    d["y_log"] = np.log1p(d["venta_total"].clip(lower=0))
    g = d.groupby("negocio")
    for k in REZAGOS:
        d[f"lag{k}"] = g["y_log"].shift(k)

    ayer = g["y_log"].shift(1)
    d["media7"] = ayer.groupby(d["negocio"]).transform(lambda s: s.rolling(7).mean())
    d["std7"] = ayer.groupby(d["negocio"]).transform(lambda s: s.rolling(7).std())
    d["nivel"] = ayer.groupby(d["negocio"]).transform(lambda s: s.expanding().mean())

    # participación de cada medio en la venta de los 7 días anteriores
    venta7 = g["venta_total"].transform(lambda s: s.shift(1).rolling(7).sum())
    for m in MEDIOS:
        suma7 = g[f"pm_{m}"].transform(lambda s: s.shift(1).rolling(7).sum())
        d[f"part_{m}"] = (suma7 / venta7).replace([np.inf, -np.inf], np.nan)
    d["dias_sin_venta7"] = g["sin_registro"].transform(lambda s: s.shift(1).rolling(7).sum())

    # calendario del día a pronosticar: se conoce de antemano
    d["dia_semana"] = d["fecha"].dt.dayofweek
    d["festivo"] = d["fecha"].isin(FESTIVOS).astype(int)
    return d


def folds_temporales(fechas, n_splits=4, dias_val=7, gap=7):
    # TimeSeriesSplit sobre los días (no sobre las filas) y luego lo paso a filas
    fechas = pd.Series(pd.to_datetime(fechas)).reset_index(drop=True)
    dias = np.sort(fechas.unique())
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=dias_val, gap=gap)
    folds = []
    for i_tr, i_val in tss.split(dias):
        folds.append((np.where(fechas.isin(dias[i_tr]))[0], np.where(fechas.isin(dias[i_val]))[0]))
    return folds


def formato_p(df, cols=("p", "p Holm")):
    # p-valores en notación científica para que no salgan como 0.00
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c].map(lambda v: f"{v:.1e}")
    return df


# ---- pipeline del modelo (cap. 7) ----
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR


def a_log(y):
    # los reversos (negativos) se recortan a 0 solo para poder sacar el log
    return np.log1p(np.clip(y, 0, None))


def crear_pipeline(C=1.0, epsilon=0.0, num=VARIABLES_NUM, cat=VARIABLES_CAT):
    numericas = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())])
    pre = ColumnTransformer([("num", numericas, list(num)), ("dia", OneHotEncoder(drop="first"), list(cat))])
    modelo = Pipeline([("pre", pre),
                       ("svr", LinearSVR(C=C, epsilon=epsilon, dual=True, max_iter=50000, random_state=SEED))])
    # el objetivo pasa a log dentro del pipeline y la predicción vuelve a COP
    return TransformedTargetRegressor(regressor=modelo, func=a_log, inverse_func=np.expm1, check_inverse=False)


def rmse_log(y, y_pred):
    return np.sqrt(np.mean((a_log(y) - a_log(y_pred)) ** 2))


RMSE_LOG = make_scorer(rmse_log, greater_is_better=False)


def datos_modelo(panel):
    # variables + solo filas con historia suficiente (el rezago más largo es 14 días)
    d = crear_variables(panel)
    return d[d[f"lag{max(REZAGOS)}"].notna()].reset_index(drop=True)

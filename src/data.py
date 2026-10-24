# Carga de datos y modelos del dashboard. Todo se lee una sola vez (lru_cache) y los modelos se cargan
# bajo demanda, para no pasar el límite de memoria del plan gratuito de Render.
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA, ART = RAIZ / "data", RAIZ / "artifacts"
# los modelos serializados referencian funciones de book/comun.py y la clase de src/original_model.py
for p in (RAIZ / "book", RAIZ / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

X_COLS = ["lag1", "lag7", "lag14", "media7", "std7", "nivel", "part_efectivo", "part_datafono", "part_rumbo",
          "part_mi_empresa", "part_credito_clientes", "dias_sin_venta7", "festivo", "dia_semana"]
DESCRIPCION = {
    "lag1": "log(1 + venta) de ayer", "lag7": "log(1 + venta) hace 7 días", "lag14": "log(1 + venta) hace 14 días",
    "media7": "media de log(1 + venta) de los 7 días anteriores", "std7": "desviación de log(1 + venta) en los 7 días anteriores",
    "nivel": "media de log(1 + venta) desde el inicio", "part_efectivo": "participación del efectivo en la venta de 7 días",
    "part_datafono": "participación del datáfono", "part_rumbo": "participación de Rumbo (crédito de flotas)",
    "part_mi_empresa": "participación de Mi Empresa", "part_credito_clientes": "participación del crédito propio",
    "dias_sin_venta7": "días sin venta en la última semana", "festivo": "1 si el día a pronosticar es festivo",
    "dia_semana": "día de la semana a pronosticar (0 = lunes)",
}
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def a_log(v):
    return np.log1p(np.clip(v, 0, None))


@lru_cache(maxsize=None)
def tabla(nombre):
    """Lee un archivo de data/ (csv, csv.gz o json) una sola vez."""
    ruta = DATA / nombre
    if nombre.endswith(".json"):
        return json.loads(ruta.read_text())
    fechas = ["fecha"] if nombre.split(".")[0] in ("panel", "modelado", "pred_test", "stl") else None
    return pd.read_csv(ruta, parse_dates=fechas)


@lru_cache(maxsize=None)
def metadata(clave):
    return json.loads((ART / f"{clave}.metadata.json").read_text())


@lru_cache(maxsize=2)
def modelo(clave):
    """Carga un artefacto entrenado (nunca se entrena nada en el servidor). Guarda a lo sumo 2 en memoria."""
    import joblib
    return joblib.load(ART / f"{clave}.joblib")


def lideres():
    return tabla("lideres.csv")


def esquema():
    """Esquema de entrada (nombre, tipo, rango válido) tomado de la metadata del mejor modelo."""
    return metadata("rf")["esquema_entrada"]


@lru_cache(maxsize=1)
def fondo_shap(n=20):
    """Filas reales de train (no centroides: el one-hot del día de la semana solo acepta enteros) como referencia."""
    m = tabla("modelado.csv.gz")
    tr = m[m["tramo"] == "train"][X_COLS]
    return tr.fillna(tr.median()).sample(n, random_state=42).reset_index(drop=True)


def explicar(clave, fila, n_perm=40, semilla=42):
    """Valores de Shapley de una predicción (en log) por muestreo de permutaciones.

    Estimador de Štrumbelj y Kononenko (2014): para cada permutación de las variables se agregan una a una
    los valores de la fila sobre una fila de referencia y se mide el cambio de la predicción. El promedio
    sobre permutaciones y referencias converge al valor de Shapley con fondo ``fondo_shap``; cumple la
    eficiencia exactamente (las contribuciones suman predicción - valor base) en cada permutación.

    Se usa en lugar de la librería ``shap`` para no cargar numba en el servidor (límite de memoria de Render).

    Returns
    -------
    tuple
        ``(valor_base, Series de contribuciones por variable)``.
    """
    m = modelo(clave)
    fondo = fondo_shap()
    x = fila[X_COLS].to_numpy(dtype=float)
    rng = np.random.default_rng(semilla)
    p = len(X_COLS)
    # todas las filas intermedias de todas las permutaciones se predicen en un solo llamado
    perms = [rng.permutation(p) for _ in range(n_perm)]
    refs = rng.integers(0, len(fondo), n_perm)
    filas = []
    for perm, r in zip(perms, refs):
        z = fondo.iloc[r].to_numpy(dtype=float).copy()
        filas.append(z.copy())
        for j in perm:
            z[j] = x[j]
            filas.append(z.copy())
    pred = a_log(m.predict(pd.DataFrame(filas, columns=X_COLS)))
    phi = np.zeros(p)
    bases = []
    for i, perm in enumerate(perms):
        bloque = pred[i * (p + 1):(i + 1) * (p + 1)]
        bases.append(bloque[0])
        phi[perm] += np.diff(bloque)
    return float(np.mean(bases)), pd.Series(phi / n_perm, index=X_COLS)

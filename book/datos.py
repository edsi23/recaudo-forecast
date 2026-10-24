# Carga del panel y armado de las dos tareas de la entrega 2:
#   - regresión: venta total del día siguiente (igual que en la entrega 1)
#   - clasificación: ¿el negocio tendrá una caída fuerte mañana?
# Las variables son las mismas de comun.py (todo con datos hasta el día anterior).
import numpy as np
import pandas as pd

from comun import (CORTE, FIN, SEED, VARIABLES_CAT, VARIABLES_NUM,
                   cargar_panel, datos_modelo, folds_temporales)

UMBRAL_CAIDA = 0.5  # caída = venta < 50 % de la mediana de los 7 días anteriores
X_COLS = VARIABLES_NUM + VARIABLES_CAT


def preparar(panel=None):
    """Arma la tabla de modelado con el objetivo de regresión y el de clasificación.

    Parameters
    ----------
    panel : pandas.DataFrame, optional
        Panel negocio x día del cap. 2. Si no se pasa, se lee de disco.

    Returns
    -------
    pandas.DataFrame
        Una fila por negocio-día con historia suficiente (14 días), las
        variables del modelo, ``venta_total`` y ``caida`` (0/1).

    Notes
    -----
    La referencia de la caída (``med7``) es la mediana de la venta de los
    7 días anteriores, así que solo usa el pasado. La mediana y no la media
    para que un día anómalo de la semana pasada no mueva el umbral.
    """
    if panel is None:
        panel = cargar_panel()
    # la mediana se saca sobre el panel completo, antes de quitar los primeros 14 días
    panel = panel.sort_values(["negocio", "fecha"]).copy()
    panel["med7"] = panel.groupby("negocio")["venta_total"].transform(lambda s: s.shift(1).rolling(7).median())
    d = datos_modelo(panel)
    d["caida"] = (d["venta_total"] < UMBRAL_CAIDA * d["med7"]).astype(int)
    return d.sort_values(["fecha", "negocio"]).reset_index(drop=True)


def partir(d):
    """Separa train (1 jul - 2 sep) y test (3 - 23 sep) con la regla del cap. 3.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train, test)`` con índices reiniciados.
    """
    train = d[d["fecha"] <= CORTE].reset_index(drop=True)
    test = d[(d["fecha"] > CORTE) & (d["fecha"] <= FIN)].reset_index(drop=True)
    return train, test


def folds_externos(fechas):
    """Bucle externo de la validación anidada: 4 folds de 7 días, ventana creciente.

    Parameters
    ----------
    fechas : array-like
        Fecha de cada fila de train.

    Returns
    -------
    list of tuple
        ``(idx_train, idx_val)`` en posiciones de fila.

    Notes
    -----
    Sin gap (en la entrega 1 era de 7 días): con 50 días útiles un gap de 7
    deja el primer fold interno con muy pocos días. Las variables ya solo usan
    el pasado (cap. 5), así que el gap no evita fuga; solo hacía la validación
    más pesimista que el uso real (h = 1).
    """
    return folds_temporales(fechas, n_splits=4, dias_val=7, gap=0)


def folds_internos(fechas, dias_val=4):
    """Bucle interno: 3 folds temporales dentro del train de un fold externo.

    Parameters
    ----------
    fechas : array-like
        Fechas de las filas del train externo.
    dias_val : int
        Días de validación por fold interno.

    Returns
    -------
    list of tuple
        ``(idx_train, idx_val)`` relativos a ``fechas``.
    """
    return folds_temporales(fechas, n_splits=3, dias_val=dias_val, gap=0)

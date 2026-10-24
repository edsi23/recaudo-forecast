# Pruebas para comparar modelos (cap. 12 y 13).
# Clasificación: Friedman -> Nemenyi / diagrama CD -> DeLong (+ Holm) -> delta de Cliff.
# Regresión: MCS -> Giacomini-White / Clark-West -> Diebold-Mariano (HLN) + bootstrap estacionario -> d de Cohen.
import numpy as np
import pandas as pd
from arch.bootstrap import MCS, StationaryBootstrap
from scipy import stats
from statsmodels.stats.multitest import multipletests

from comun import SEED


def holm(p):
    """p-valores ajustados por Holm-Bonferroni."""
    return multipletests(np.asarray(p), method="holm")[1]


def friedman_nemenyi(perdidas, alfa=0.05):
    """Prueba de Friedman y diferencia crítica de Nemenyi.

    Parameters
    ----------
    perdidas : pandas.DataFrame
        Una fila por bloque (día o negocio) y una columna por modelo; menor es mejor.
    alfa : float

    Returns
    -------
    dict
        ``chi2``, ``p``, ``rangos`` (rango medio por modelo, 1 = mejor), ``cd`` y ``n``.

    Notes
    -----
    CD = q_alfa * sqrt(k (k + 1) / (6 N)), con q_alfa el cuantil del rango
    studentizado para k modelos dividido por sqrt(2) (Demšar, 2006).
    """
    p = perdidas.dropna()
    chi2, pval = stats.friedmanchisquare(*[p[c] for c in p])
    rangos = p.rank(axis=1).mean().sort_values()
    k, n = p.shape[1], p.shape[0]
    q = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    return {"chi2": chi2, "p": pval, "rangos": rangos, "cd": q * np.sqrt(k * (k + 1) / (6 * n)), "n": n}


def diagrama_cd(rangos, cd, ax, titulo=""):
    """Diagrama de diferencia crítica: modelos en su rango medio y barras que unen
    a los que no se distinguen según Nemenyi."""
    r = rangos.sort_values()
    k = len(r)
    ax.set_xlim(0.5, k + 0.5 if k < 12 else r.max() + 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.85, r.min() - 0.3, r.max() + 0.3, color="#52514e", lw=1)
    mitad = (k + 1) // 2
    for i, (nombre, v) in enumerate(r.items()):
        izq = i < mitad
        y = 0.75 - 0.6 * ((i if izq else k - 1 - i) + 1) / (mitad + 1)
        ax.vlines(v, y, 0.85, color="#52514e", lw=0.8)
        x_txt = r.min() - 0.4 if izq else r.max() + 0.4
        ax.hlines(y, min(v, x_txt), max(v, x_txt), color="#52514e", lw=0.8)
        ax.text(x_txt, y, f"{nombre} ({v:.2f})", ha="right" if izq else "left", va="center", fontsize=8)
    # grupos sin diferencia: desde cada modelo hasta el último que está a menos de CD
    grupos, y_barra = [], 0.80
    for i in range(k):
        j = max(jj for jj in range(k) if r.iloc[jj] - r.iloc[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for a, b in grupos:
        ax.hlines(y_barra, r.iloc[a] - 0.05, r.iloc[b] + 0.05, color="#e34948", lw=3)
        y_barra -= 0.03
    ax.hlines(0.95, r.min(), r.min() + cd, color="#2a78d6", lw=2)
    ax.text(r.min() + cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", fontsize=8, color="#2a78d6")
    ax.set_title(titulo)


def cliff_delta(a, b):
    """Delta de Cliff: P(a > b) - P(a < b). |d| < 0.147 despreciable, < 0.33 pequeño,
    < 0.474 mediano, y grande por encima (Romano et al., 2006)."""
    a, b = np.asarray(a), np.asarray(b)
    return float((np.sign(a[:, None] - b[None, :])).mean())


def magnitud_cliff(d):
    d = abs(d)
    return "despreciable" if d < 0.147 else "pequeño" if d < 0.33 else "mediano" if d < 0.474 else "grande"


def cohen_d(dif):
    """d de Cohen de una diferencia de pérdidas pareada: media / desviación."""
    dif = np.asarray(dif)
    return float(dif.mean() / dif.std(ddof=1))


def delong(y, s1, s2):
    """Prueba de DeLong para dos AUC correlacionadas (mismas observaciones).

    Returns
    -------
    tuple
        ``(auc1, auc2, z, p)`` con p bilateral.
    """
    y = np.asarray(y).astype(bool)
    m, n = y.sum(), (~y).sum()
    v10, v01, aucs = [], [], []
    for s in (np.asarray(s1), np.asarray(s2)):
        pos, neg = s[y], s[~y]
        rz = stats.rankdata(np.concatenate([pos, neg]))
        rx, ry = stats.rankdata(pos), stats.rankdata(neg)
        aucs.append((rz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((rz[:m] - rx) / n)        # componente de cada positivo
        v01.append(1 - (rz[m:] - ry) / m)    # componente de cada negativo
    s = np.cov(np.array(v10)) / m + np.cov(np.array(v01)) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def diebold_mariano(d, h=1):
    """Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (1997).

    Parameters
    ----------
    d : array-like
        Serie de diferencias de pérdida (modelo A - modelo B). d > 0: B es mejor.
    h : int
        Horizonte del pronóstico.

    Returns
    -------
    tuple
        ``(estadístico, p)`` bilateral con t de Student con n - 1 g.l.
    """
    d = np.asarray(d, float)
    n = len(d)
    dc = d - d.mean()
    gamma = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = d.mean() / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * stats.t.sf(abs(dm), n - 1)


def clark_west(y, f_restr, f_amplio):
    """Clark-West (2007) para modelos anidados (en la escala que se pase, aquí log).

    Ajusta la diferencia de errores cuadráticos sumando (f_restr - f_amplio)^2, que
    es el ruido que agrega estimar los parámetros de más del modelo amplio.

    Returns
    -------
    tuple
        ``(t, p)`` unilateral: H1 = el modelo amplio pronostica mejor.
    """
    y, a, b = (np.asarray(v, float) for v in (y, f_restr, f_amplio))
    f = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))
    return t, stats.norm.sf(t)


def giacomini_white(d):
    """Prueba de habilidad predictiva condicional de Giacomini y White (2006), h = 1.

    Instrumentos: constante y la diferencia de pérdida del periodo anterior. Si se
    rechaza, uno de los modelos es mejor en promedio o en ciertos "estados" (por
    ejemplo, después de un día en el que le fue mejor).

    Returns
    -------
    tuple
        ``(estadístico, p)`` con chi2 de 2 g.l.
    """
    d = np.asarray(d, float)
    z = np.column_stack([np.ones(len(d) - 1), d[:-1]]) * d[1:, None]
    zbar = z.mean(0)
    omega = (z.T @ z) / len(z)
    est = len(z) * zbar @ np.linalg.solve(omega, zbar)
    return est, stats.chi2.sf(est, z.shape[1])


def model_confidence_set(perdidas, alfa=0.10, bloque=3, reps=2000):
    """Model Confidence Set (Hansen, Lunde y Nason, 2011) con bootstrap por bloques.

    Parameters
    ----------
    perdidas : pandas.DataFrame
        Serie de pérdidas (filas = periodos, columnas = modelos).

    Returns
    -------
    pandas.DataFrame
        p-valor MCS por modelo y si queda en el conjunto de confianza.
    """
    mcs = MCS(perdidas, size=alfa, reps=reps, block_size=bloque, method="R", seed=SEED)
    mcs.compute()
    out = mcs.pvalues.rename(columns={"Pvalue": "p MCS"})
    out["en el conjunto"] = out.index.isin(mcs.included)
    out["pérdida media"] = perdidas.mean()[out.index]
    return out.sort_values("pérdida media")


def ic_estacionario(d, bloque=3, reps=2000):
    """IC 95 % de la media de una serie dependiente con bootstrap estacionario
    (Politis y Romano, 1994): bloques de largo aleatorio con media ``bloque``."""
    bs = StationaryBootstrap(bloque, np.asarray(d, float), seed=SEED)
    return tuple(bs.conf_int(np.mean, reps=reps, method="percentile").ravel())


def ic_bca_negocios(negocio, func, reps=1000):
    """IC 95 % BCa de una métrica remuestreando negocios (bootstrap por conglomerados).

    Parameters
    ----------
    negocio : array-like
        Negocio de cada fila.
    func : callable
        Recibe un vector booleano/índices de filas y devuelve la métrica.

    Returns
    -------
    tuple
        ``(inferior, superior)``.
    """
    negocio = np.asarray(negocio)
    grupos = np.unique(negocio)
    filas = {g: np.where(negocio == g)[0] for g in grupos}

    def estad(idx):
        return func(np.concatenate([filas[grupos[i]] for i in idx]))

    res = stats.bootstrap((np.arange(len(grupos)),), estad, vectorized=False, n_resamples=reps,
                          method="BCa", random_state=SEED)
    return res.confidence_interval.low, res.confidence_interval.high

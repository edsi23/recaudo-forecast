# Los 4 métodos de búsqueda de hiperparámetros con la MISMA función de evaluación:
# en cada evaluación se arma un pipeline nuevo y se ajusta en cada fold interno,
# así que el preprocesamiento nunca ve la validación (también en Optuna y DEAP,
# que no traen esto incorporado).
import copy
import random
import time

import numpy as np
import optuna
from deap import base, creator, tools
from sklearn.model_selection import ParameterGrid

from comun import SEED
from metricas import perdida
from modelos import ESPACIOS, REJILLAS, ajustar, crear_modelo

optuna.logging.set_verbosity(optuna.logging.WARNING)
PRESUPUESTO = 16  # evaluaciones por búsqueda (cada una = 3 ajustes, uno por fold interno)


def evaluar_config(params, tarea, nombre, balanceo, X, y, folds, semilla=SEED):
    """Pérdida media de una configuración en los folds internos.

    Parameters
    ----------
    params : dict
    tarea, nombre, balanceo : str
        Ver ``modelos.crear_modelo``.
    X : pandas.DataFrame
    y : numpy.ndarray
    folds : list of tuple
        Folds internos ``(idx_train, idx_val)``.

    Returns
    -------
    float
        Pérdida media (menor es mejor): RMSE log en regresión, 1 - F1 en clasificación.
        Si el ajuste falla (por ejemplo ADASYN sin vecinos), devuelve ``inf``.
    """
    valores = []
    for tr, va in folds:
        try:
            m = crear_modelo(tarea, nombre, params, balanceo, semilla)
            ajustar(m, X.iloc[tr], y[tr], tarea, nombre, balanceo)
            valores.append(perdida(tarea, y[va], m.predict(X.iloc[va])))
        except (ValueError, RuntimeError):
            return np.inf
    return float(np.mean(valores))


def _decodificar(espacio, genes):
    # genes en [0, 1] -> hiperparámetros reales (lo usan random y el genético)
    p = {}
    for dim, g in zip(espacio, genes):
        if dim[1] == "cat":
            ops = dim[2]
            p[dim[0]] = ops[min(int(g * len(ops)), len(ops) - 1)]
            continue
        nombre, tipo, lo, hi, log = dim
        v = np.exp(np.log(lo) + g * (np.log(hi) - np.log(lo))) if log else lo + g * (hi - lo)
        p[nombre] = int(round(v)) if tipo == "int" else float(v)
    return p


class Historial:
    """Guarda cada evaluación: pérdida, tiempo y mejor valor hasta ese momento
    (la curva 'anytime performance')."""

    def __init__(self, objetivo):
        self.objetivo, self.filas, self.t0 = objetivo, [], time.perf_counter()

    def __call__(self, params):
        t = time.perf_counter()
        v = self.objetivo(params)
        mejor = min([f["perdida"] for f in self.filas] + [v])
        self.filas.append({"eval": len(self.filas) + 1, "perdida": v, "mejor": mejor,
                           "seg": time.perf_counter() - t, "params": params})
        return v

    def mejor(self):
        f = min(self.filas, key=lambda f: f["perdida"])
        return f["params"], f["perdida"]


def grid(espacio, rejilla, f, presupuesto, semilla):
    for p in list(ParameterGrid(rejilla))[:presupuesto]:
        f(p)


def aleatorio(espacio, rejilla, f, presupuesto, semilla):
    rng = np.random.default_rng(semilla)
    for _ in range(presupuesto):
        f(_decodificar(espacio, rng.random(len(espacio))))


def bayes(espacio, rejilla, f, presupuesto, semilla):
    # TPE: modela p(x | bueno) y p(x | malo) y propone el x que maximiza su cociente,
    # que es proporcional a la mejora esperada (EI). Los primeros 5 puntos son al azar.
    def objetivo(trial):
        p = {}
        for dim in espacio:
            if dim[1] == "cat":
                p[dim[0]] = trial.suggest_categorical(dim[0], dim[2])
            elif dim[1] == "int":
                p[dim[0]] = trial.suggest_int(dim[0], dim[2], dim[3], log=dim[4])
            else:
                p[dim[0]] = trial.suggest_float(dim[0], dim[2], dim[3], log=dim[4])
        return f(p)

    muestreador = optuna.samplers.TPESampler(seed=semilla, n_startup_trials=5)
    estudio = optuna.create_study(direction="minimize", sampler=muestreador)
    estudio.optimize(objetivo, n_trials=presupuesto)


# DEAP necesita las clases creadas una sola vez
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individuo", list, fitness=creator.FitnessMin)

GENETICO = {"poblacion": 6, "elite": 1, "torneo": 3, "p_cruce": 0.7, "alfa_cruce": 0.5,
            "p_mutacion": 0.3, "sigma": 0.15, "indpb": 0.5}


def genetico(espacio, rejilla, f, presupuesto, semilla, cfg=GENETICO, diversidad=None):
    """Algoritmo genético con DEAP sobre genes en [0, 1].

    Selección por torneo (k = 3), cruce blend (BLX-0.5), mutación gaussiana
    (sigma = 0.15) y elitismo (el mejor pasa intacto). Se corre hasta gastar el
    presupuesto. Si se pasa la lista ``diversidad``, se le agrega la distancia
    media entre individuos de cada generación.
    """
    rnd = random.Random(semilla)
    tb = base.Toolbox()
    tb.register("seleccionar", tools.selTournament, tournsize=cfg["torneo"])
    tb.register("cruzar", tools.cxBlend, alpha=cfg["alfa_cruce"])
    tb.register("mutar", tools.mutGaussian, mu=0, sigma=cfg["sigma"], indpb=cfg["indpb"])

    def evaluar(ind):
        ind.fitness.values = (f(_decodificar(espacio, ind)),)

    # DEAP usa el módulo random global: le fijo la semilla
    random.seed(semilla)
    pob = [creator.Individuo([rnd.random() for _ in espacio]) for _ in range(cfg["poblacion"])]
    gastado = 0
    for ind in pob:
        if gastado < presupuesto:
            evaluar(ind)
            gastado += 1
    pob = [i for i in pob if i.fitness.valid]
    while gastado < presupuesto:
        if diversidad is not None:
            diversidad.append(_diversidad(pob))
        elite = tools.selBest(pob, cfg["elite"])
        hijos = [copy.deepcopy(i) for i in tb.seleccionar(pob, len(pob) - cfg["elite"])]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if rnd.random() < cfg["p_cruce"]:
                tb.cruzar(a, b)
                del a.fitness.values, b.fitness.values
        for h in hijos:
            if rnd.random() < cfg["p_mutacion"]:
                tb.mutar(h)
                del h.fitness.values
            h[:] = [min(max(g, 0.0), 1.0) for g in h]
        # solo se evalúan los hijos que cambiaron (los demás ya tienen su pérdida)
        for h in hijos:
            if not h.fitness.valid and gastado < presupuesto:
                evaluar(h)
                gastado += 1
        pob = [copy.deepcopy(e) for e in elite] + [h for h in hijos if h.fitness.valid]
    if diversidad is not None:
        diversidad.append(_diversidad(pob))


def _diversidad(pob):
    g = np.array(pob)
    if len(g) < 2:
        return 0.0
    dif = g[:, None, :] - g[None, :, :]
    d = np.sqrt((dif ** 2).sum(-1))
    return float(d[np.triu_indices(len(g), 1)].mean())


METODOS = {"grid": grid, "random": aleatorio, "bayes": bayes, "genetico": genetico}


def buscar(metodo, tarea, nombre, balanceo, X, y, folds, presupuesto=PRESUPUESTO, semilla=SEED,
           diversidad=None):
    """Corre una búsqueda completa y devuelve la mejor configuración y el historial.

    Returns
    -------
    tuple
        ``(mejores_params, mejor_perdida, historial)`` donde ``historial`` es una
        lista de dicts con ``eval``, ``perdida``, ``mejor``, ``seg`` y ``params``.
    """
    f = Historial(lambda p: evaluar_config(p, tarea, nombre, balanceo, X, y, folds, semilla))
    extra = {"diversidad": diversidad} if metodo == "genetico" else {}
    METODOS[metodo](ESPACIOS[nombre], REJILLAS[nombre], f, presupuesto, semilla, **extra)
    params, valor = f.mejor()
    return params, valor, f.filas

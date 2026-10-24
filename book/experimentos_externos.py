# Entrega 3: evaluación en los 3 conjuntos externos y benchmark del estado del arte (HistGradientBoosting).
# Mismo protocolo que el panel de Masser: validación anidada 4 x 3, misma semilla, presupuesto de 16
# evaluaciones con optimización bayesiana para todos los modelos, y luego el ACS con los expertos
# elegidos en cada fold.
# Uso:  python experimentos_externos.py
import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from joblib import Parallel, delayed

from comun import DATA, SEED
from datos import preparar
from experimentos import correr
from modelo_original import EXPERTOS, un_fold

EXT = DATA / "externos"
MODELOS = ["knn", "ridge", "lasso", "arbol", "rf", "xgb", "svr", "hgb"]
CONJUNTOS = ["m5", "retail", "electricidad"]


def cargar_conjunto(nombre):
    """Panel con las variables del modelo; test = últimos 21 días, train = el resto."""
    panel = pd.read_csv(EXT / f"panel_{nombre}.csv.gz", parse_dates=["fecha"])
    d = preparar(panel)
    corte = d["fecha"].max() - pd.Timedelta(days=21)
    return d[d["fecha"] <= corte].reset_index(drop=True), d[d["fecha"] > corte].reset_index(drop=True)


def benchmark(conjunto, modelo, metodo="bayes"):
    archivo = EXT / "resultados" / f"{conjunto}__{modelo}__{metodo}.pkl"
    if archivo.exists():
        return
    train, test = cargar_conjunto(conjunto)
    res = correr("reg", modelo, "ninguno", metodo, train, test)
    with open(archivo, "wb") as fh:
        pickle.dump(res, fh)


def acs(conjunto):
    archivo = EXT / "resultados" / f"{conjunto}__acs.pkl"
    if archivo.exists():
        return
    train, _ = cargar_conjunto(conjunto)
    res = {m: pickle.load(open(EXT / "resultados" / f"{conjunto}__{m}__bayes.pkl", "rb")) for m in EXPERTOS}
    folds = [un_fold(k, train, {m: res[m]["folds"][k]["params"] for m in EXPERTOS}) for k in range(4)]
    with open(archivo, "wb") as fh:
        pickle.dump([{k: v for k, v in f.items() if k != "arbol"} for f in folds], fh)


def hgb_masser(metodo):
    """HistGradientBoosting en el panel de Masser con el mismo runner de la entrega 2."""
    from datos import partir
    from experimentos import _tarea
    train, test = partir(preparar())
    _tarea(("reg", "hgb", "ninguno", metodo), train, test)


if __name__ == "__main__":
    (EXT / "resultados").mkdir(exist_ok=True)
    t = time.perf_counter()
    trabajos = [delayed(benchmark)(c, m) for c in CONJUNTOS for m in MODELOS]
    trabajos += [delayed(hgb_masser)(o) for o in ["grid", "random", "bayes", "genetico"]]
    Parallel(n_jobs=10, backend="loky", verbose=5)(trabajos)
    Parallel(n_jobs=3, backend="loky")(delayed(acs)(c) for c in CONJUNTOS)
    print(f"listo en {(time.perf_counter() - t) / 60:.1f} min")

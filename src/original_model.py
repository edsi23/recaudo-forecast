# Modelo original (entrega 3): Árbol de Combinación en el Símplex (ACS).
#
# Idea: varios modelos ("expertos") pronostican lo mismo, y cuál conviene depende del régimen del
# negocio ese día (estable, volátil, con días sin venta, ...). El ACS parte el espacio de régimen con un
# árbol y en cada hoja aprende pesos de combinación w >= 0, sum(w) = 1 resolviendo un problema
# cuadrático convexo con gradiente proyectado. Los pesos de cada hoja se encogen hacia los del padre.
#
# Solo usa numpy (y scikit-learn para la interfaz BaseEstimator / validaciones).
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def proyectar_simplex(v):
    """Proyección euclidiana de v sobre el símplex {w : w >= 0, sum(w) = 1}.

    Algoritmo exacto por ordenamiento (Held et al., 1974; Duchi et al., 2008), O(K log K).

    Parameters
    ----------
    v : numpy.ndarray, forma (K,)

    Returns
    -------
    numpy.ndarray
        El punto del símplex más cercano a ``v``.
    """
    u = np.sort(v)[::-1]
    acum = np.cumsum(u) - 1.0
    k = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - acum / k > 0)[0][-1]
    theta = acum[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def resolver_pesos(A, b, previo, lam, max_iter=5000, tol=1e-9):
    """Pesos óptimos en el símplex para una hoja, por gradiente proyectado.

    Minimiza  f(w) = w'Aw - 2b'w + lam * ||w - previo||^2   sujeto a  w >= 0, sum(w) = 1.
    En el ACS, A = E'E/n es la matriz de segundos momentos de los errores de los expertos y b = 0
    (ver ``_momentos``), así que f es el error cuadrático medio de la combinación más la penalización.

    Parameters
    ----------
    A : numpy.ndarray, forma (K, K)
        Matriz de segundos momentos de los expertos (semidefinida positiva).
    b : numpy.ndarray, forma (K,)
    previo : numpy.ndarray, forma (K,)
        Pesos hacia los que se encoge la solución (los del nodo padre).
    lam : float
        Fuerza del encogimiento (>= 0). Con lam > 0 el problema es fuertemente convexo.
    max_iter : int
    tol : float
        Parada cuando ||w_{t+1} - w_t||_inf < tol.

    Returns
    -------
    tuple
        ``(w, n_iter)``.

    Notes
    -----
    El gradiente 2(Aw - b) + 2 lam (w - previo) es Lipschitz con constante L = 2(lambda_max(A) + lam).
    Con paso 1/L el gradiente proyectado converge a la solución global (el símplex es convexo y
    cerrado), con f(w_t) - f* <= L ||w_0 - w*||^2 / (2t); si lam > 0 la convergencia es lineal
    (ver la demostración en el capítulo del modelo original del libro).
    """
    L = 2.0 * (np.linalg.eigvalsh(A)[-1] + lam)
    paso = 1.0 / max(L, 1e-12)
    w = previo.copy()
    for t in range(1, max_iter + 1):
        grad = 2.0 * (A @ w - b) + 2.0 * lam * (w - previo)
        nuevo = proyectar_simplex(w - paso * grad)
        if np.max(np.abs(nuevo - w)) < tol:
            return nuevo, t
        w = nuevo
    return w, max_iter


def _momentos(Z, y):
    """Matriz de segundos momentos de los errores de los expertos, D = E'E/n con E = y·1' - Z.

    En el símplex (sum(w) = 1) el error de la combinación es y - w'z = w'(y·1 - z), así que el ECM es
    exactamente w'Dw. Usar D en lugar de los momentos de Z da el mismo problema pero mucho mejor
    condicionado: con pronósticos en log (todos cerca de 15), Z'Z/n tiene un autovalor enorme en la
    dirección del vector de unos, que es irrelevante dentro del símplex.
    """
    E = y[:, None] - Z
    K = Z.shape[1]
    return E.T @ E / len(y), np.zeros(K), 0.0


def _ecm(A, b, c, w):
    # error cuadrático medio de la combinación Zw usando solo los momentos
    return float(w @ A @ w - 2.0 * b @ w + c)


class ArbolCombinacionSimplex(BaseEstimator, RegressorMixin):
    """Árbol de Combinación en el Símplex (ACS).

    Las primeras ``n_expertos`` columnas de ``X`` son los pronósticos de K expertos (en la escala del
    objetivo) y el resto son variables de régimen. El árbol parte el espacio de régimen y en cada hoja
    combina a los expertos con pesos en el símplex.

    Parameters
    ----------
    n_expertos : int
        Número K de columnas de pronósticos al inicio de X.
    max_depth : int
        Profundidad máxima del árbol (0 = un solo juego de pesos global).
    min_samples_leaf : int
        Mínimo de observaciones por hoja.
    lam : float
        Encogimiento de los pesos de cada hijo hacia los del padre. Se escala por n_padre / n_hijo, así
        que las hojas pequeñas se encogen más.
    n_umbrales : int
        Umbrales candidatos por variable (cuantiles), para que la búsqueda no dependa de n.
    ganancia_min : float
        Mejora relativa mínima del ECM para aceptar una partición.
    max_iter, tol : int, float
        Parada del gradiente proyectado.
    random_state : int
        No se usa (el algoritmo es determinista); se guarda por compatibilidad.

    Attributes
    ----------
    nodos_ : list of dict
        El árbol: cada nodo tiene ``pesos``, ``n``, ``ecm`` y, si no es hoja, ``var``, ``umbral``,
        ``izq`` y ``der`` (índices en la lista).
    iteraciones_ : list of int
        Iteraciones del gradiente proyectado en cada problema de hoja resuelto (para la convergencia).
    """

    def __init__(self, n_expertos=2, max_depth=3, min_samples_leaf=200, lam=0.1, n_umbrales=16,
                 ganancia_min=1e-3, max_iter=5000, tol=1e-9, random_state=42):
        self.n_expertos = n_expertos
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lam = lam
        self.n_umbrales = n_umbrales
        self.ganancia_min = ganancia_min
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _validar_parametros(self):
        if not (isinstance(self.n_expertos, (int, np.integer)) and self.n_expertos >= 1):
            raise ValueError("n_expertos debe ser un entero >= 1")
        if self.max_depth < 0 or self.min_samples_leaf < 1 or self.lam < 0 or self.n_umbrales < 1:
            raise ValueError("max_depth >= 0, min_samples_leaf >= 1, lam >= 0 y n_umbrales >= 1")

    def fit(self, X, y):
        """Construye el árbol.

        Parameters
        ----------
        X : array-like, forma (n, K + q)
            K pronósticos seguidos de q variables de régimen (sin nulos).
        y : array-like, forma (n,)

        Returns
        -------
        self
        """
        self._validar_parametros()
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        K = self.n_expertos
        if X.shape[1] < K:
            raise ValueError(f"X tiene {X.shape[1]} columnas y n_expertos = {K}")
        self.n_features_in_ = X.shape[1]
        Z, G = X[:, :K], X[:, K:]
        self.nodos_, self.iteraciones_ = [], []
        uniforme = np.full(K, 1.0 / K)
        A, b, c = _momentos(Z, y)
        # la raíz se encoge hacia el promedio simple con la misma fuerza lam: estimar K pesos tiene un error de
        # estimación que suele superar la ganancia de combinar ("puzzle de la combinación", Smith y Wallis, 2009)
        w, it = resolver_pesos(A, b, uniforme, max(self.lam, 1e-6), self.max_iter, self.tol)
        self.iteraciones_.append(it)
        self._crecer(np.arange(len(y)), Z, G, y, w, _ecm(A, b, c, w), 0)
        self.pesos_globales_ = self.nodos_[0]["pesos"]
        return self

    def _crecer(self, idx, Z, G, y, w, ecm, prof):
        nodo = {"pesos": w, "n": len(idx), "ecm": ecm}
        self.nodos_.append(nodo)
        yo = len(self.nodos_) - 1
        if prof >= self.max_depth or len(idx) < 2 * self.min_samples_leaf or G.shape[1] == 0:
            return yo
        mejor = self._mejor_particion(idx, Z, G, y, w)
        if mejor is None or ecm - mejor["ecm"] <= self.ganancia_min * ecm:
            return yo
        nodo.update(var=mejor["var"], umbral=mejor["umbral"])
        izq = idx[G[idx, mejor["var"]] <= mejor["umbral"]]
        der = idx[G[idx, mejor["var"]] > mejor["umbral"]]
        nodo["izq"] = self._crecer(izq, Z, G, y, mejor["w_izq"], mejor["ecm_izq"], prof + 1)
        nodo["der"] = self._crecer(der, Z, G, y, mejor["w_der"], mejor["ecm_der"], prof + 1)
        return yo

    def _mejor_particion(self, idx, Z, G, y, w_padre):
        """Busca (variable, umbral) que minimiza el ECM ponderado de los dos hijos.

        Usa sumas acumuladas de los productos de errores e e' ordenadas por la variable: evaluar un
        umbral cuesta O(K^3 + I K^2) y no depende de n.
        """
        n, m = len(idx), self.min_samples_leaf
        # errores de los expertos en el nodo y sus productos por fila: e e' (ver _momentos)
        En = y[idx][:, None] - Z[idx]
        ee = np.einsum("ij,ik->ijk", En, En)
        mejor = None
        for j in range(G.shape[1]):
            g = G[idx, j]
            orden = np.argsort(g, kind="mergesort")
            gs = g[orden]
            c_ee = np.cumsum(ee[orden], 0)
            tot_ee = c_ee[-1]
            K = Z.shape[1]
            umbrales = np.unique(np.quantile(gs, np.linspace(0, 1, self.n_umbrales + 2)[1:-1]))
            for u in umbrales:
                k = np.searchsorted(gs, u, side="right")  # filas con g <= u
                if k < m or n - k < m:
                    continue
                res = []
                for nh, s_ee in [(k, c_ee[k - 1]), (n - k, tot_ee - c_ee[k - 1])]:
                    A, b, c = s_ee / nh, np.zeros(K), 0.0
                    lam_h = self.lam * n / nh
                    wh, it = resolver_pesos(A, b, w_padre, lam_h, self.max_iter, self.tol)
                    self.iteraciones_.append(it)
                    res.append((wh, _ecm(A, b, c, wh), nh))
                ecm = (res[0][1] * res[0][2] + res[1][1] * res[1][2]) / n
                if mejor is None or ecm < mejor["ecm"]:
                    mejor = {"var": j, "umbral": float(u), "ecm": ecm, "w_izq": res[0][0], "ecm_izq": res[0][1],
                             "w_der": res[1][0], "ecm_der": res[1][1]}
        return mejor

    def hoja(self, X):
        """Índice del nodo hoja al que llega cada fila."""
        check_is_fitted(self, "nodos_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X debe tener {self.n_features_in_} columnas")
        G = X[:, self.n_expertos:]
        hojas = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            nodo = 0
            while "var" in self.nodos_[nodo]:
                n = self.nodos_[nodo]
                nodo = n["izq"] if G[i, n["var"]] <= n["umbral"] else n["der"]
            hojas[i] = nodo
        return hojas

    def pesos(self, X):
        """Pesos de combinación de cada fila, forma (n, K)."""
        return np.array([self.nodos_[h]["pesos"] for h in self.hoja(X)])

    def predict(self, X):
        """Pronóstico combinado: suma de los expertos ponderados por los pesos de su hoja."""
        X = check_array(X, dtype=float)
        return np.sum(self.pesos(X) * X[:, :self.n_expertos], axis=1)

    def n_hojas(self):
        check_is_fitted(self, "nodos_")
        return sum("var" not in n for n in self.nodos_)


class EnsambleRegimen(BaseEstimator, RegressorMixin):
    """Ensamble por régimen: entrena los expertos, genera sus pronósticos fuera de muestra en orden
    temporal y ajusta un :class:`ArbolCombinacionSimplex` sobre ellos.

    Parameters
    ----------
    expertos : list of (str, estimator)
        Modelos de scikit-learn que reciben las columnas ``columnas_x`` y predicen el objetivo.
    columnas_x : list of str
        Variables de entrada de los expertos.
    columnas_fijas : list of str
        Columnas que ya son un pronóstico en escala log (por ejemplo ``lag7``, el ingenuo estacional).
    columnas_regimen : list of str
        Variables de régimen para el árbol.
    col_tiempo : str
        Columna con la fecha, para generar los pronósticos fuera de muestra respetando el tiempo.
    n_bloques, dias_bloque : int
        Bloques para generar los pronósticos fuera de muestra (ver ``modo_oof``).
    modo_oof : {"cruzado", "temporal"}
        ``"cruzado"`` (por defecto): todos los días de train se parten en ``n_bloques`` bloques contiguos y
        cada bloque se pronostica con expertos entrenados con los demás bloques (ajuste cruzado del
        stacking; los expertos tienen casi la misma cantidad de datos que los finales).
        ``"temporal"``: solo los últimos ``n_bloques * dias_bloque`` días, cada bloque con expertos
        entrenados con los días anteriores (ventana creciente; los expertos ven menos datos).
    desacuerdo : bool
        Si se agrega como variable de régimen la desviación estándar entre los pronósticos de los
        expertos (mide la incertidumbre del ensamble en ese punto).
    arbol : ArbolCombinacionSimplex
        Combinador (sus ``n_expertos`` se fijan solos).
    """

    def __init__(self, expertos=None, columnas_x=None, columnas_fijas=("lag7",), columnas_regimen=None,
                 col_tiempo="fecha", n_bloques=4, dias_bloque=5, desacuerdo=True, arbol=None, modo_oof="cruzado"):
        self.expertos = expertos
        self.columnas_x = columnas_x
        self.columnas_fijas = columnas_fijas
        self.columnas_regimen = columnas_regimen
        self.col_tiempo = col_tiempo
        self.n_bloques = n_bloques
        self.dias_bloque = dias_bloque
        self.desacuerdo = desacuerdo
        self.arbol = arbol
        self.modo_oof = modo_oof

    def _pronosticos(self, modelos, X):
        z = [np.log1p(np.clip(m.predict(X[self.columnas_x]), 0, None)) for m in modelos]
        z += [X[c].to_numpy(dtype=float) for c in self.columnas_fijas]
        return np.column_stack(z)

    def _regimen(self, X, Z):
        G = X[list(self.columnas_regimen)].to_numpy(dtype=float)
        G = np.where(np.isnan(G), self.medianas_, G)
        if self.desacuerdo:
            G = np.column_stack([G, Z.std(axis=1)])
        return G

    def fit(self, X, y):
        """Ajusta expertos y combinador. ``X`` es un DataFrame que incluye ``col_tiempo``."""
        y = np.asarray(y, dtype=float)
        fechas = np.asarray(X[self.col_tiempo])
        dias = np.sort(np.unique(fechas))
        # como mucho la mitad de los días se pronostica fuera de muestra: la otra mitad entrena el primer bloque
        bloque = max(1, min(self.dias_bloque, (len(dias) // 2) // self.n_bloques))
        n_oof = self.n_bloques * bloque
        if len(dias) < 2 * self.n_bloques:
            raise ValueError("no hay suficientes días para generar pronósticos fuera de muestra")
        self.medianas_ = np.nanmedian(X[list(self.columnas_regimen)].to_numpy(dtype=float), axis=0)
        # 1) pronósticos fuera de muestra de cada experto, bloque por bloque
        Zs, ys, Gs = [], [], []
        if self.modo_oof == "cruzado":
            cortes = np.array_split(dias, self.n_bloques)
            limites = [(c[0], c[-1]) for c in cortes]
        else:
            limites = [(dias[len(dias) - n_oof + b * bloque], dias[min(len(dias) - n_oof + (b + 1) * bloque, len(dias)) - 1])
                       for b in range(self.n_bloques)]
        for ini, fin in limites:
            va = (fechas >= ini) & (fechas <= fin)
            tr = ~va if self.modo_oof == "cruzado" else fechas < ini
            modelos = [clone(e).fit(X.loc[tr, self.columnas_x], y[tr]) for _, e in self.expertos]
            Zb = self._pronosticos(modelos, X.loc[va])
            Zs.append(Zb); ys.append(y[va]); Gs.append(self._regimen(X.loc[va], Zb))
        Z, y_oof, G = np.vstack(Zs), np.concatenate(ys), np.vstack(Gs)
        Z = np.where(np.isnan(Z), np.nanmedian(Z, axis=1, keepdims=True), Z)
        self.oof_ = (Z, np.log1p(np.clip(y_oof, 0, None)), G)
        # 2) combinador sobre los pronósticos fuera de muestra (en log)
        self.n_expertos_ = Z.shape[1]
        self.arbol_ = clone(self.arbol if self.arbol is not None else ArbolCombinacionSimplex())
        self.arbol_.set_params(n_expertos=self.n_expertos_)
        self.arbol_.fit(np.column_stack([Z, G]), self.oof_[1])
        # 3) expertos finales con todo train
        self.modelos_ = [clone(e).fit(X[self.columnas_x], y) for _, e in self.expertos]
        self.nombres_ = [n for n, _ in self.expertos] + list(self.columnas_fijas)
        return self

    def reajustar_arbol(self, **params):
        """Vuelve a ajustar solo el combinador sobre los pronósticos fuera de muestra ya guardados
        (sirve para buscar sus hiperparámetros sin reentrenar los expertos)."""
        check_is_fitted(self, "oof_")
        Z, y_log, G = self.oof_
        self.arbol_ = clone(self.arbol_).set_params(**params)
        self.arbol_.fit(np.column_stack([Z, G]), y_log)
        return self

    def matriz(self, X):
        """[pronósticos de los expertos en log | variables de régimen] para el árbol."""
        check_is_fitted(self, "arbol_")
        Z = self._pronosticos(self.modelos_, X)
        Z = np.where(np.isnan(Z), np.nanmedian(Z, axis=1, keepdims=True), Z)
        return np.column_stack([Z, self._regimen(X, Z)])

    def predict(self, X):
        """Pronóstico en la escala original (COP)."""
        return np.expm1(self.arbol_.predict(self.matriz(X)))

    def pesos(self, X):
        """Peso de cada experto en cada fila (DataFrame)."""
        import pandas as pd
        return pd.DataFrame(self.arbol_.pesos(self.matriz(X)), columns=self.nombres_, index=X.index)

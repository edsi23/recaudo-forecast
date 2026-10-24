# Pruebas unitarias del modelo original (Árbol de Combinación en el Símplex).
# Correr desde la raíz del repo:  python -m pytest tests -q
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from original_model import ArbolCombinacionSimplex, proyectar_simplex, resolver_pesos  # noqa: E402


def datos_dos_regimenes(n=4000, semilla=0):
    """Dos expertos y un régimen g: si g <= 0 el bueno es el experto 1, si g > 0 el experto 2.
    La solución conocida son pesos (1, 0) a la izquierda y (0, 1) a la derecha, con corte en 0."""
    rng = np.random.default_rng(semilla)
    g = rng.uniform(-1, 1, n)
    y = rng.normal(0, 1, n)
    ruido_malo, ruido_bueno = rng.normal(0, 1.0, n), rng.normal(0, 0.05, n)
    e1 = np.where(g <= 0, y + ruido_bueno, y + ruido_malo)
    e2 = np.where(g > 0, y + ruido_bueno, y + ruido_malo)
    return np.column_stack([e1, e2, g]), y


# ---- proyección y solver ----
def test_proyeccion_queda_en_el_simplex():
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = proyectar_simplex(rng.normal(0, 3, 6))
        assert np.all(w >= 0) and np.isclose(w.sum(), 1.0)


def test_proyeccion_de_un_punto_del_simplex_es_el_mismo_punto():
    w = np.array([0.2, 0.5, 0.3])
    assert np.allclose(proyectar_simplex(w), w)


def test_solver_coincide_con_busqueda_exhaustiva():
    # con K = 2 el símplex es un segmento: comparo con una rejilla muy fina
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(500, 2)); y = 0.3 * Z[:, 0] + 0.7 * Z[:, 1] + rng.normal(0, 0.1, 500)
    A, b = Z.T @ Z / 500, Z.T @ y / 500
    w, _ = resolver_pesos(A, b, np.array([0.5, 0.5]), 0.0, max_iter=5000, tol=1e-12)
    rejilla = np.linspace(0, 1, 100001)
    f = [np.array([t, 1 - t]) @ A @ np.array([t, 1 - t]) - 2 * b @ np.array([t, 1 - t]) for t in rejilla]
    assert abs(w[0] - rejilla[int(np.argmin(f))]) < 1e-3


def test_solver_cumple_kkt_con_expertos_correlacionados():
    # caso que motivó la reformulación con la matriz de errores: 5 expertos en escala log (~15) muy
    # correlacionados. En el óptimo del símplex, el gradiente es igual (= nu) en las coordenadas
    # positivas y >= nu en las que valen 0 (condiciones de KKT).
    from original_model import _momentos
    rng = np.random.default_rng(4)
    y = 15 + rng.normal(0, 1, 3000)
    Z = np.column_stack([y + rng.normal(0, s, 3000) for s in (0.3, 0.35, 0.5, 0.6, 1.2)])
    A, b, _ = _momentos(Z, y)
    w, it = resolver_pesos(A, b, np.full(5, 0.2), 1e-6)
    g = 2 * (A @ w - b)
    activos = w > 1e-8
    assert it < 5000
    assert np.ptp(g[activos]) < 1e-4 and np.all(g[~activos] >= g[activos].min() - 1e-4)


# ---- (i) aprende en un problema sintético con solución conocida ----
def test_aprende_los_dos_regimenes():
    X, y = datos_dos_regimenes()
    m = ArbolCombinacionSimplex(n_expertos=2, max_depth=1, min_samples_leaf=100, lam=0.0, n_umbrales=99).fit(X, y)
    raiz = m.nodos_[0]
    assert raiz["var"] == 0 and abs(raiz["umbral"]) < 0.15
    izq, der = m.nodos_[raiz["izq"]]["pesos"], m.nodos_[raiz["der"]]["pesos"]
    assert izq[0] > 0.95 and der[1] > 0.95


def test_no_es_peor_que_el_mejor_experto_en_entrenamiento():
    # Proposición 4: ECM(árbol) <= min_k ECM(experto k) + lam (1 - 1/K)
    X, y = datos_dos_regimenes()
    lam = 0.5
    m = ArbolCombinacionSimplex(n_expertos=2, max_depth=2, min_samples_leaf=100, lam=lam).fit(X, y)
    ecm_arbol = np.mean((m.predict(X) - y) ** 2)
    ecm_expertos = [np.mean((X[:, k] - y) ** 2) for k in range(2)]
    assert ecm_arbol <= min(ecm_expertos) + lam * (1 - 1 / 2) + 1e-9


def test_profundidad_cero_da_pesos_globales():
    X, y = datos_dos_regimenes()
    m = ArbolCombinacionSimplex(n_expertos=2, max_depth=0).fit(X, y)
    assert m.n_hojas() == 1 and np.allclose(m.pesos(X), m.pesos_globales_)


# ---- (ii) reproducibilidad ----
def test_misma_semilla_mismo_resultado():
    X, y = datos_dos_regimenes(semilla=3)
    p1 = ArbolCombinacionSimplex(n_expertos=2, random_state=7).fit(X, y).predict(X)
    p2 = ArbolCombinacionSimplex(n_expertos=2, random_state=7).fit(X, y).predict(X)
    assert np.array_equal(p1, p2)


# ---- (iii) entradas inválidas ----
@pytest.mark.parametrize("params", [{"n_expertos": 0}, {"max_depth": -1}, {"min_samples_leaf": 0}, {"lam": -1.0}])
def test_parametros_invalidos(params):
    X, y = datos_dos_regimenes(n=200)
    with pytest.raises(ValueError):
        ArbolCombinacionSimplex(**{"n_expertos": 2, **params}).fit(X, y)


def test_nulos_y_dimensiones_incorrectas():
    X, y = datos_dos_regimenes(n=300)
    Xn = X.copy(); Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        ArbolCombinacionSimplex(n_expertos=2).fit(Xn, y)
    m = ArbolCombinacionSimplex(n_expertos=2).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :2])
    with pytest.raises(ValueError):
        ArbolCombinacionSimplex(n_expertos=5).fit(X, y)


def test_predecir_sin_entrenar():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        ArbolCombinacionSimplex(n_expertos=2).predict(np.zeros((3, 3)))


# ---- (iv) compatibilidad con scikit-learn ----
def test_clone_y_get_params():
    m = ArbolCombinacionSimplex(n_expertos=2, max_depth=4, lam=0.3)
    c = clone(m)
    assert c.get_params() == m.get_params() and c is not m


def test_funciona_en_cross_val_score():
    from sklearn.model_selection import cross_val_score
    X, y = datos_dos_regimenes(n=1500)
    s = cross_val_score(ArbolCombinacionSimplex(n_expertos=2, max_depth=1, min_samples_leaf=50), X, y, cv=3,
                        scoring="neg_mean_squared_error")
    assert np.all(np.isfinite(s))

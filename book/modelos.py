# Definición de los 7 modelos de cada tarea, sus espacios de búsqueda y los pipelines.
# Todo el preprocesamiento (imputar, escalar, one-hot, SMOTE/ADASYN, log del objetivo)
# va dentro del pipeline, así que en cada fold se ajusta solo con el train de ese fold.
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from comun import SEED, VARIABLES_CAT, VARIABLES_NUM, a_log

MODELOS_REG = ["knn", "ridge", "lasso", "arbol", "rf", "xgb", "svr"]
MODELOS_CLF = ["knn", "nb", "logistica", "arbol", "rf", "xgb", "svm"]
BALANCEOS = ["ninguno", "smote", "adasyn", "balanced"]
OPTIMIZADORES = ["grid", "random", "bayes", "genetico"]

# ---- espacios de búsqueda ----
# (nombre, tipo, bajo, alto, log)  o  (nombre, "cat", [opciones])
# Los rangos salen de lo que se sabe del problema (ver cap. 9): ~20 mil filas, 19 columnas.
ESPACIOS = {
    "knn": [("n_neighbors", "int", 5, 200, True), ("weights", "cat", ["uniform", "distance"])],
    "ridge": [("alpha", "float", 1e-3, 1e3, True)],
    "lasso": [("alpha", "float", 1e-5, 1e-1, True)],
    "nb": [("var_smoothing", "float", 1e-12, 1e-2, True)],
    "logistica": [("C", "float", 1e-3, 1e2, True), ("penalty", "cat", ["l1", "l2"])],
    "arbol": [("max_depth", "int", 3, 20, False), ("min_samples_leaf", "int", 1, 300, True)],
    "rf": [("n_estimators", "int", 50, 150, False), ("max_depth", "int", 5, 20, False),
           ("min_samples_leaf", "int", 1, 100, True), ("max_features", "float", 0.2, 0.7, False)],
    "xgb": [("n_estimators", "int", 100, 500, False), ("learning_rate", "float", 0.01, 0.3, True),
            ("max_depth", "int", 3, 10, False), ("subsample", "float", 0.6, 1.0, False),
            ("colsample_bytree", "float", 0.6, 1.0, False), ("min_child_weight", "float", 1, 50, True)],
    "svr": [("C", "float", 1e-3, 1, True), ("epsilon", "float", 0.0, 0.5, False)],
    "svm": [("C", "float", 1e-4, 10, True)],
}

# rejillas del grid search: a lo sumo 16 puntos, el mismo presupuesto de los demás métodos
REJILLAS = {
    "knn": {"n_neighbors": [5, 10, 20, 40, 80, 160], "weights": ["uniform", "distance"]},
    "ridge": {"alpha": list(np.logspace(-3, 3, 16))},
    "lasso": {"alpha": list(np.logspace(-5, -1, 16))},
    "nb": {"var_smoothing": list(np.logspace(-12, -2, 16))},
    "logistica": {"C": list(np.logspace(-3, 2, 8)), "penalty": ["l1", "l2"]},
    "arbol": {"max_depth": [4, 8, 12, 16], "min_samples_leaf": [1, 10, 50, 200]},
    "rf": {"n_estimators": [50, 150], "max_depth": [8, 16], "min_samples_leaf": [2, 20],
           "max_features": [0.3, 0.6]},
    "xgb": {"n_estimators": [150, 400], "learning_rate": [0.03, 0.1], "max_depth": [4, 8],
            "subsample": [0.7, 1.0]},
    "svr": {"C": list(np.logspace(-3, 0, 8)), "epsilon": [0.0, 0.1]},
    "svm": {"C": list(np.logspace(-4, 1, 16))},
}


class KNNBalanceado(KNeighborsClassifier):
    """KNN con pesos de clase: divide la proporción de vecinos de cada clase por
    la frecuencia de esa clase en train. Es el equivalente de ``class_weight='balanced'``
    para un modelo que no lo trae."""

    def fit(self, X, y):
        super().fit(X, y)
        self.prior_ = np.bincount(np.asarray(y)) / len(y)
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X) / self.prior_
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _preprocesador():
    numericas = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())])
    return ColumnTransformer([("num", numericas, VARIABLES_NUM),
                              ("dia", OneHotEncoder(drop="first"), VARIABLES_CAT)])


def _regresor(nombre, p, semilla):
    if nombre == "knn":
        return KNeighborsRegressor(**p)
    if nombre == "ridge":
        return Ridge(**p)
    if nombre == "lasso":
        return Lasso(max_iter=5000, **p)
    if nombre == "arbol":
        return DecisionTreeRegressor(random_state=semilla, **p)
    if nombre == "rf":
        return RandomForestRegressor(max_samples=0.5, random_state=semilla, n_jobs=1, **p)
    if nombre == "xgb":
        return XGBRegressor(tree_method="hist", random_state=semilla, n_jobs=1, **p)
    if nombre == "svr":
        return LinearSVR(dual=True, max_iter=20000, random_state=semilla, **p)
    raise ValueError(nombre)


def _clasificador(nombre, p, semilla, balanced):
    cw = "balanced" if balanced else None
    if nombre == "knn":
        return KNNBalanceado(**p) if balanced else KNeighborsClassifier(**p)
    if nombre == "nb":
        # NB no tiene class_weight: con priors iguales cada clase pesa lo mismo
        return GaussianNB(priors=[0.5, 0.5] if balanced else None, **p)
    if nombre == "logistica":
        return LogisticRegression(solver="saga", max_iter=3000, tol=1e-3, class_weight=cw,
                                  random_state=semilla, **p)
    if nombre == "arbol":
        return DecisionTreeClassifier(class_weight=cw, random_state=semilla, **p)
    if nombre == "rf":
        return RandomForestClassifier(max_samples=0.5, class_weight=cw, random_state=semilla, n_jobs=1, **p)
    if nombre == "xgb":
        # XGBoost no tiene class_weight: se pasa con sample_weight en el fit (ver ajustar)
        return XGBClassifier(tree_method="hist", random_state=semilla, n_jobs=1, eval_metric="logloss", **p)
    if nombre == "svm":
        return LinearSVC(dual=False, class_weight=cw, random_state=semilla, max_iter=5000, **p)
    raise ValueError(nombre)


def crear_modelo(tarea, nombre, params, balanceo="ninguno", semilla=SEED):
    """Arma el pipeline completo (preprocesamiento + balanceo + modelo).

    Parameters
    ----------
    tarea : {"reg", "clf"}
    nombre : str
        Uno de ``MODELOS_REG`` o ``MODELOS_CLF``.
    params : dict
        Hiperparámetros del modelo.
    balanceo : {"ninguno", "smote", "adasyn", "balanced"}
        Solo aplica a clasificación.
    semilla : int

    Returns
    -------
    estimator
        En regresión, un ``TransformedTargetRegressor`` que entrena en log1p y
        predice en COP. En clasificación, un ``imblearn.Pipeline``.

    Notes
    -----
    SMOTE y ADASYN van después del escalado (trabajan con distancias) y dentro del
    pipeline de imblearn, que solo los aplica en ``fit``: en validación y test los
    datos no se tocan.
    """
    params = dict(params)
    if tarea == "reg":
        modelo = Pipeline([("pre", _preprocesador()), ("modelo", _regresor(nombre, params, semilla))])
        return TransformedTargetRegressor(regressor=modelo, func=a_log, inverse_func=np.expm1,
                                          check_inverse=False)
    pasos = [("pre", _preprocesador())]
    if balanceo == "smote":
        pasos.append(("balanceo", SMOTE(k_neighbors=5, random_state=semilla)))
    elif balanceo == "adasyn":
        pasos.append(("balanceo", ADASYN(n_neighbors=5, random_state=semilla)))
    pasos.append(("modelo", _clasificador(nombre, params, semilla, balanceo == "balanced")))
    return ImbPipeline(pasos)


def ajustar(modelo, X, y, tarea, nombre, balanceo):
    """Ajusta el pipeline. Para XGBoost con ``balanced`` pasa los pesos de clase
    como ``sample_weight`` (es lo mismo que ``class_weight='balanced'``)."""
    if tarea == "clf" and nombre == "xgb" and balanceo == "balanced":
        return modelo.fit(X, y, modelo__sample_weight=compute_sample_weight("balanced", y))
    return modelo.fit(X, y)


def puntaje(modelo, X):
    """Puntaje continuo de la clase positiva: probabilidad si el modelo la da,
    si no la distancia al hiperplano (SVM). Sirve para AUC y curvas ROC."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)

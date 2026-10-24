# Sección 3: Modelos. Tabla de líderes, predicción en vivo con explicación local, diagnósticos,
# comparación estadística y el modelo original (ACS).
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dash_table, dcc, html, no_update

import plots as P
from data import DESCRIPCION, DIAS, X_COLS, a_log, esquema, explicar, lideres, metadata, modelo, tabla

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Pronóstico de ventas")

LID = lideres()
NOMBRES = dict(zip(LID["clave"], LID["modelo"] + " · " + LID["optimizador"]))
ESQ = {c["nombre"]: c for c in esquema()}
INTERVALOS = tabla("intervalos.json")
pred_test = tabla("pred_test.csv.gz")
mod = tabla("modelado.csv.gz")
test_rows = mod[mod["tramo"] == "test"].reset_index(drop=True)
EJEMPLOS = test_rows.sample(60, random_state=7).sort_values(["negocio", "fecha"])
ENTEROS = {"festivo", "dia_semana", "dias_sin_venta7"}


def pm(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


tabla_lideres = pd.DataFrame({
    "modelo": LID["modelo"], "optimizador": LID["optimizador"],
    "RMSE log (media ± desv.)": [pm(a, b) for a, b in zip(LID["RMSE log media"], LID["RMSE log desv"])],
    "MAE log (media ± desv.)": [pm(a, b, 4) for a, b in zip(LID["MAE log media"], LID["MAE log desv"])],
    "RMSE log media": LID["RMSE log media"].round(4), "MAE log media": LID["MAE log media"].round(4),
    "inferencia (ms / 1 000 filas)": LID["ms por 1 000 filas"].round(1), "tamaño (MB)": LID["MB"].round(1)})


def campo(c):
    e = ESQ[c]
    paso = 1 if c in ENTEROS else "any"
    return dbc.Col([html.Label(c, htmlFor={"tipo": "entrada", "col": c}, className="small fw-bold mb-0"),
                    dbc.Input(id={"tipo": "entrada", "col": c}, type="number", step=paso, min=e["min"], max=e["max"], size="sm"),
                    html.Small(f"{DESCRIPCION[c]} · rango [{e['min']:.3g}, {e['max']:.3g}]", className="text-muted")],
                   xs=12, sm=6, lg=4, className="mb-2")


def seccion(titulo, *hijos, sub=None):
    return dbc.Card(dbc.CardBody([html.H4(titulo), html.P(sub, className="text-muted") if sub else None, *hijos]), className="mb-3")


PSEUDO = """Algoritmo ACS (Árbol de Combinación en el Símplex), versión final
Entrada: train (X, y, fecha), K expertos, variables de régimen g, hiperparámetros
         (max_depth D, min_samples_leaf m, λ, B umbrales)
Salida : árbol T cuyas hojas tienen pesos w_hoja en el símplex Δ_K = {w ≥ 0, Σ w = 1}

1. Ajuste cruzado: partir los días de train en 4 bloques contiguos; cada bloque se pronostica
   con los expertos entrenados en los otros 3   →  Z (n × K, en log);  E = y·1ᵀ − Z (errores)
2. Raíz: w₀ = argmin_{w ∈ Δ_K}  wᵀ D w + λ ‖w − 1/K‖²,  D = EᵀE / n      (encoge al promedio)
3. CRECER(nodo, w_padre, profundidad):
     si profundidad = D o n_nodo < 2m: es hoja
     para cada variable de régimen j y cada umbral u entre B cuantiles:
         D_h = Σ_{i ∈ h} e_i e_iᵀ / n_h       (sumas acumuladas: no depende de n)
         w_h = argmin_{w ∈ Δ_K} wᵀ D_h w + λ (n / n_h) ‖w − w_padre‖²
     elegir (j, u) que minimiza el ECM ponderado de los hijos
     si la mejora relativa ≤ ganancia_min: es hoja; si no, CRECER cada hijo con su w_h
4. Gradiente proyectado (cada hoja): w ← Π_Δ( w − ∇f(w) / L ),  L = 2(λ_max(D_h) + λ'),
   parar cuando ‖w_{t+1} − w_t‖_∞ < 10⁻⁹ (tope 5 000 iteraciones)
5. Predicción: reentrenar los K expertos con todo train; para una fila nueva, bajar por el árbol
   según g y devolver  ŷ = expm1( Σ_k w_hoja,k · z_k )"""

layout = html.Div([
    html.H2("Modelos"),
    seccion("Tabla de líderes: la mejor combinación de cada modelo",
            dash_table.DataTable(id="t-lideres", data=tabla_lideres.to_dict("records"),
                                 columns=[{"name": c, "id": c} for c in tabla_lideres.columns if c not in ("RMSE log media", "MAE log media")]
                                 + [{"name": "RMSE log media", "id": "RMSE log media", "type": "numeric"}],
                                 sort_action="native", filter_action="native", style_table={"overflowX": "auto"},
                                 style_cell={"fontSize": 13, "textAlign": "left"},
                                 style_data_conditional=[{"if": {"filter_query": '{modelo} contains "ACS"'}, "backgroundColor": "#fdf1eb"},
                                                         {"if": {"row_index": 0}, "fontWeight": "bold"}]),
            html.Div("Cada fila es el optimizador con menor RMSE log medio en el bucle externo de la validación anidada "
                     "(4 folds temporales). El mejor global, Random Forest · Grid, está en el conjunto de confianza del MCS "
                     "de la entrega 2. Se puede ordenar y filtrar cada columna.", className="insight"),
            sub="Criterio estadístico: menor RMSE en log medio del bucle externo (media ± desviación de los 4 folds)."),
    dbc.Row([dbc.Col([html.Label("Modelo para predecir y diagnosticar", htmlFor="sel-modelo", className="fw-bold"),
                      dcc.Dropdown(id="sel-modelo", options=[{"label": v, "value": k} for k, v in NOMBRES.items()],
                                   value="rf", clearable=False)], lg=6)], className="mb-3"),
    seccion("Predicción en vivo",
            dbc.Row([dbc.Col([html.Label("Cargar un negocio-día real de test (opcional)", htmlFor="ejemplo"),
                              dcc.Dropdown(id="ejemplo", options=[{"label": f"{r.negocio} · {r.fecha:%d %b} ({DIAS[int(r.dia_semana)]})",
                                                                   "value": i} for i, r in EJEMPLOS.iterrows()],
                                           placeholder="elija un ejemplo para llenar el formulario")], lg=6)], className="mb-2"),
            dbc.Row([campo(c) for c in X_COLS]),
            dbc.Button("Predecir", id="btn-predecir", color="primary", className="mt-2"),
            html.Div(id="alerta-pred", className="mt-2"),
            dcc.Loading(dbc.Row([dbc.Col(html.Div(id="salida-pred"), lg=4),
                                 dbc.Col(dcc.Graph(id="g-shap-local", figure=P.vacia("Elija un ejemplo o llene el formulario y pulse «Predecir»")), lg=8)],
                                className="mt-2")),
            sub="El formulario sale del esquema de entrada guardado con el modelo (nombre, tipo y rango válido de train). "
                "La salida es el pronóstico, un intervalo de predicción del 80 % y 90 % y la explicación SHAP de esa predicción (valores de Shapley estimados por muestreo de 40 permutaciones, Štrumbelj y Kononenko, 2014)."),
    seccion("Diagnóstico del modelo seleccionado (test: 3 – 23 sep)",
            dbc.Row([dbc.Col(dcc.Graph(id="g-real-pred"), lg=6), dbc.Col(dcc.Graph(id="g-resid"), lg=6)]),
            dbc.Row([dbc.Col(dcc.Graph(id="g-resid-acf"), lg=6), dbc.Col(dcc.Graph(id="g-resid-hist"), lg=6)]),
            html.Div(id="i-diag")),
    seccion("Interpretabilidad global (SHAP)", dcc.Graph(id="g-shap-global"), html.Div(id="i-shap-global"),
            sub="Importancia media |SHAP| (KernelSHAP en log sobre 60 filas de test, misma referencia para todos los modelos; calculada en el libro)."),
    seccion("Comparación estadística: Model Confidence Set (MCS) con el modelo original incluido",
            dbc.RadioItems(id="mcs-metrica", options=[{"label": " RMSE log (métrica principal)", "value": "rmse"},
                                                      {"label": " MAE log (secundaria)", "value": "mae"}], value="rmse", inline=True),
            html.Div(id="t-mcs"), html.Div(id="i-mcs"),
            sub="35 modelos (28 de la entrega 2, 4 HistGradientBoosting, el ACS y 2 líneas base) sobre la pérdida diaria de "
                "los 28 días del bucle externo; α = 0,10, bootstrap por bloques de 3 días."),
    html.H3("Modelo original: Árbol de Combinación en el Símplex (ACS)", className="mt-4"),
    dbc.Row([
        dbc.Col(seccion("Idea", html.P(
            "Varios modelos pronostican lo mismo, pero cuál conviene depende del régimen del negocio ese día (estable, volátil, "
            "con días sin venta, con desacuerdo entre los modelos). El ACS parte ese espacio de régimen con un árbol y en cada "
            "hoja aprende pesos de combinación no negativos que suman 1, resolviendo un problema cuadrático convexo (sobre la "
            "matriz de errores de los expertos) con un gradiente proyectado programado desde cero. La raíz se encoge hacia el "
            "promedio simple y cada hoja hacia los pesos de su padre. El árbol se entrena con pronósticos fuera de muestra "
            "obtenidos por ajuste cruzado."),
            html.Ul([html.Li("Expertos: Random Forest, XGBoost, Ridge, KNN (mejor combinación de la entrega 2) e ingenuo estacional."),
                     html.Li("Propiedades demostradas en el libro: convergencia del solver, terminación del árbol, que en entrenamiento "
                             "no es peor que el mejor experto más λ(1 − 1/K), y complejidad lineal en n.")]),
            html.H6("Diferencias con el trabajo más cercano"),
            dbc.Table.from_dataframe(pd.DataFrame([
                ["Stacking / Super Learner", "pesos globales", "pesos distintos por región del espacio de régimen"],
                ["FFORMA (meta-aprendizaje)", "pesos por serie con XGBoost y softmax", "pesos por negocio-día, convexos y exactos por hoja"],
                ["Árboles de modelos (M5, MOB)", "hojas con regresiones lineales", "hojas con combinaciones en el símplex + encogimiento al padre"],
                ["Selección dinámica (DES)", "elige un experto por vecindario", "combina todos con pesos óptimos por hoja"]],
                columns=["método", "qué hace", "qué cambia el ACS"]), size="sm", bordered=True, striped=True)), lg=6),
        dbc.Col(seccion("Pseudocódigo", html.Pre(PSEUDO, className="pseudo")), lg=6),
    ]),
    dbc.Row([dbc.Col(seccion("Convergencia y búsqueda de hiperparámetros", dcc.Graph(id="g-conv"), html.Div(id="i-conv")), lg=6),
             dbc.Col(seccion("Ablación y reglas estándar de combinación", dcc.Graph(id="g-abl"), html.Div(id="i-abl")), lg=6)]),
    seccion("Pruebas de hipótesis: ACS contra el mejor benchmark (Random Forest · Grid)", html.Div(id="t-pruebas"),
            sub="p-valores ajustados por Holm y Benjamini-Hochberg dentro de cada métrica (6 pruebas por métrica)."),
    html.Div(id="veredicto"),
    dcc.Store(id="dummy"),
])


@callback(Output({"tipo": "entrada", "col": ALL}, "value"), Input("ejemplo", "value"), prevent_initial_call=True)
def llenar(i):
    if i is None:
        return [no_update] * len(X_COLS)
    r = EJEMPLOS.loc[i]
    return [None if pd.isna(r[c]) else (int(r[c]) if c in ENTEROS else round(float(r[c]), 5)) for c in X_COLS]


@callback(Output("alerta-pred", "children"), Output("salida-pred", "children"), Output("g-shap-local", "figure"),
          Input("btn-predecir", "n_clicks"), State({"tipo": "entrada", "col": ALL}, "value"), State("sel-modelo", "value"),
          State("ejemplo", "value"), prevent_initial_call=True)
def predecir(n, valores, clave, ej):
    # validación de tipos y rangos antes de llamar al modelo
    errores, fila = [], {}
    for c, v in zip(X_COLS, valores):
        e = ESQ[c]
        if v is None:
            if e["admite_nulos"]:
                fila[c] = np.nan
                continue
            errores.append(f"{c}: falta el valor")
            continue
        if c in ENTEROS and float(v) != int(v):
            errores.append(f"{c}: debe ser entero")
        if not (e["min"] - 1e-9 <= v <= e["max"] + 1e-9):
            errores.append(f"{c}: {v} está fuera del rango válido [{e['min']:.3g}, {e['max']:.3g}]")
        fila[c] = float(v)
    if errores:
        return dbc.Alert([html.B("Revise las entradas:"), html.Ul([html.Li(x) for x in errores])], color="danger"), no_update, no_update
    X = pd.DataFrame([fila])[X_COLS]
    try:
        p = float(modelo(clave).predict(X)[0])
        Xs = X.fillna(mod[mod["tramo"] == "train"][X_COLS].median())
        base, sv = explicar(clave, Xs.iloc[0])
    except Exception as ex:  # cualquier error se muestra sin tumbar la app
        return dbc.Alert(f"No se pudo predecir: {ex}", color="danger"), no_update, no_update
    z = a_log(p)
    q = INTERVALOS[clave]
    lo80, hi80 = np.expm1(max(z - q["q80"], 0)), np.expm1(z + q["q80"])
    lo90, hi90 = np.expm1(max(z - q["q90"], 0)), np.expm1(z + q["q90"])
    real = None
    if ej is not None:
        real = EJEMPLOS.loc[ej, "venta_total"]
    tarjeta = dbc.Card(dbc.CardBody([
        html.Div("Venta pronosticada para el día siguiente", className="kpi-nota"),
        html.Div(P.cop(p), className="kpi-valor"),
        html.Div(f"Intervalo 80 %: {P.cop(lo80)} – {P.cop(hi80)}", className="kpi-nota"),
        html.Div(f"Intervalo 90 %: {P.cop(lo90)} – {P.cop(hi90)}", className="kpi-nota"),
        html.Div(f"Venta real de ese día: {P.cop(real)}" if real is not None else "", className="kpi-nota mt-2 fw-bold"),
        html.Small("Intervalo conforme: cuantiles del error absoluto en log del bucle externo de este modelo.", className="text-muted"),
    ]))
    extra = []
    if clave == "acs":
        w = modelo("acs").pesos(X).iloc[0]
        extra = [html.Div(["Pesos del ACS para esta fila: ", ", ".join(f"{k} {v:.2f}" for k, v in w.items())],
                          className="insight mt-2")]
    return None, [tarjeta, *extra], P.shap_local(base, sv, f"Por qué {NOMBRES[clave]} predice este valor (SHAP local)")


@callback(Output("g-real-pred", "figure"), Output("g-resid", "figure"), Output("g-resid-acf", "figure"),
          Output("g-resid-hist", "figure"), Output("i-diag", "children"), Output("g-shap-global", "figure"),
          Output("i-shap-global", "children"), Input("sel-modelo", "value"))
def diagnostico(clave):
    y, p = pred_test["venta_total"].to_numpy(), pred_test[clave].to_numpy()
    e = a_log(y) - a_log(p)
    f1 = P.real_vs_pred(y, p, f"Real contra predicho · {NOMBRES[clave]}")
    f2 = P.residuos_vs_ajuste(p, e, "Residuos contra ajustados")
    diario = pd.Series(e).groupby(pred_test["fecha"].to_numpy()).mean()
    from statsmodels.tsa.stattools import acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    a = acf(diario, nlags=7, fft=False)[1:]
    lb = acorr_ljungbox(diario, lags=[7])["lb_pvalue"].iloc[0]
    f3 = P.barras_acf(a, "ACF del residuo medio diario", 1.96 / np.sqrt(len(diario)))
    f4 = P.histograma(pd.Series(e).clip(-3, 3), "Histograma de residuos (recortado a ±3)", "residuo en log", color=P.GRIS)
    cero = y <= 0
    ins = html.Div(f"RMSE log en test {np.sqrt(np.mean(e ** 2)):.3f}; WAPE {np.abs(y - p).sum() / np.abs(y).sum():.1%}. "
                   f"Los {cero.sum()} días sin venta ({cero.mean():.1%}) explican el {(e[cero] ** 2).sum() / (e ** 2).sum():.0%} "
                   f"del error cuadrático: son los puntos aislados abajo a la derecha. Ljung-Box (rezago 7) del residuo diario: "
                   f"p = {lb:.2f} ({'sin' if lb > 0.05 else 'con'} autocorrelación remanente).", className="insight")
    sh = tabla("shap_global.csv")
    s = sh[sh["clave"] == clave].set_index("variable")["shap_abs"]
    f5 = P.barras_h(s, f"Importancia global · {NOMBRES[clave]}", "media |SHAP| (log)")
    ins2 = html.Div(f"Las variables que más mueven la predicción son {', '.join(s.sort_values(ascending=False).index[:3])}: el "
                    "nivel reciente del negocio domina. El calendario y el mix de medios pesan poco.", className="insight")
    return f1, f2, f3, f4, ins, f5, ins2


@callback(Output("t-mcs", "children"), Output("i-mcs", "children"), Input("mcs-metrica", "value"))
def mcs(metrica):
    t = tabla(f"original_mcs_{metrica}.csv").rename(columns={"Model name": "modelo"})
    t = t.rename(columns={t.columns[0]: "modelo"}) if "modelo" not in t else t
    t["p MCS"] = t["p MCS"].round(3)
    t["pérdida media"] = t["pérdida media"].round(4)
    t["en el conjunto"] = np.where(t["en el conjunto"], "sí", "no")
    tab = dash_table.DataTable(data=t.to_dict("records"), columns=[{"name": c, "id": c} for c in t.columns], page_size=12,
                               sort_action="native", style_cell={"fontSize": 12, "textAlign": "left"},
                               style_data_conditional=[{"if": {"filter_query": '{modelo} contains "ACS"'}, "backgroundColor": "#fdf1eb", "fontWeight": "bold"},
                                                       {"if": {"filter_query": '{en el conjunto} = "sí"'}, "color": "#1baf7a"}])
    dentro = t.loc[t["en el conjunto"] == "sí", "modelo"].tolist()
    texto = (f"Quedan {len(dentro)} modelos en el conjunto de confianza al 90 %. "
             + ("El ACS es el único (es el mejor de forma significativa)." if dentro == ["ACS (original)"] else
                "El ACS está en el conjunto, pero no solo: no se distingue de los mejores." if "ACS (original)" in dentro else
                "El ACS queda fuera del conjunto."))
    return tab, html.Div(texto, className="insight")


def veredicto(res):
    """Veredicto de superioridad calculado con los criterios de la guía (sección 6.1), sin números escritos a mano."""
    bench = res.drop(index=["ACS (original)", "Ingenuo estacional", "Persistencia"])
    partes, superior = [], {}
    for m, col, nombre in [("rmse", "RMSE log media", "RMSE log (principal)"), ("mae", "MAE log media", "MAE log (secundaria)")]:
        t = tabla(f"original_pruebas_{m}.csv")
        par = t[~t["prueba"].str.startswith("Model Confidence")]
        rech = int((par["p Holm"] < 0.05).sum())
        d = par["d de Cohen"].dropna().iloc[0]
        dc = par["delta de Cliff"].dropna().iloc[0]
        mcs_t = tabla(f"original_mcs_{m}.csv")
        dentro = mcs_t[mcs_t["en el conjunto"]].iloc[:, 0].tolist()
        mejor_media = res.loc["ACS (original)", col] < bench[col].min()
        efecto = abs(d) >= 0.2 or abs(dc) >= 0.147
        superior[m] = mejor_media and rech >= 3 and efecto
        partes.append(html.Li(f"{nombre}: ACS {res.loc['ACS (original)', col]:.4f} contra {bench[col].min():.4f} del mejor benchmark "
                              f"({bench[col].idxmin()}); {rech} de {len(par)} pruebas pareadas rechazan con Holm; "
                              f"d de Cohen {d:.2f}, δ de Cliff {dc:.3f}; MCS con {len(dentro)} modelo(s)"
                              + (" (el ACS es el único)" if dentro == ["ACS (original)"] else "") + "."))
    if superior["rmse"]:
        titulo, color = "Veredicto: el ACS es superior a los benchmarks en la métrica principal (R3 se cumple)", "success"
    else:
        titulo, color = "Veredicto: el ACS no es superior a los benchmarks en la métrica principal (R3 no se cumple)", "warning"
    nota = ("La métrica principal (RMSE log) se fijó en la entrega 2, antes de diseñar el modelo; la MAE log se reporta como "
            "secundaria y no se usa para declarar superioridad. En los tres conjuntos de datos externos el ACS no es el mejor "
            "modelo (capítulo 20 del libro).")
    return dbc.Alert([html.H4(titulo, className="alert-heading"), html.Ul(partes), html.P(nota, className="mb-0")], color=color)


@callback(Output("g-conv", "figure"), Output("i-conv", "children"), Output("g-abl", "figure"), Output("i-abl", "children"),
          Output("t-pruebas", "children"), Output("veredicto", "children"), Input("dummy", "data"))
def original(_):
    from plotly.subplots import make_subplots
    h = tabla("original_optuna.csv")
    it = tabla("original_iteraciones.csv")["iteraciones"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Optuna (TPE): mejor RMSE log interno", "Iteraciones del gradiente proyectado"])
    fig.add_trace(go.Scatter(x=h["eval"], y=h["perdida"], mode="markers", marker=dict(color=P.GRIS), name="evaluación"), 1, 1)
    fig.add_trace(go.Scatter(x=h["eval"], y=h["perdida"].cummin(), line=dict(color=P.NARANJA, width=3), name="mejor hasta t"), 1, 1)
    fig.add_trace(go.Histogram(x=it, marker_color=P.AZUL, name="problemas de hoja", nbinsx=40), 1, 2)
    fig.update_layout(template=P.PLANTILLA, height=360)
    fig.update_xaxes(title_text="evaluación t", row=1, col=1); fig.update_xaxes(title_text="iteraciones hasta tol", row=1, col=2)
    i1 = html.Div(f"El TPE encuentra su mejor configuración en la evaluación {int(h['perdida'].idxmin()) + 1} de {len(h)}. El "
                  f"gradiente proyectado converge en una mediana de {int(it.median())} iteraciones (máximo {int(it.max())}, tope "
                  f"5 000) en los {len(it):,} problemas de hoja resueltos al construir el árbol final.", className="insight")
    a = tabla("original_ablacion.csv").set_index("modelo") if "modelo" in tabla("original_ablacion.csv") else \
        tabla("original_ablacion.csv").set_index(tabla("original_ablacion.csv").columns[0])
    res = tabla("original_resumen.csv").set_index(tabla("original_resumen.csv").columns[0])
    a.loc["ACS (completo)"] = res.loc["ACS (original)"]
    a = a.drop(index=[i for i in a.index if i.startswith("experto")])
    s = a["RMSE log media"].sort_values()
    f2 = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                          marker_color=[P.NARANJA if i == "ACS (completo)" else P.GRIS if i.startswith("sin") else P.AZUL for i in s.index],
                          hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    f2.add_vline(x=res.loc["Random Forest · Grid", "RMSE log media"], line_dash="dash", line_color=P.OSCURO,
                 annotation_text="Random Forest", annotation_position="top")
    f2.update_layout(template=P.PLANTILLA, height=420, title="RMSE log medio del bucle externo (menor es mejor)")
    f2.update_xaxes(range=[s.min() - 0.01, s.max() + 0.01])
    acs_v, rf_v = res.loc["ACS (original)", "RMSE log media"], res.loc["Random Forest · Grid", "RMSE log media"]
    peores = [i for i in s.index if i.startswith("sin") and s[i] > acs_v]
    mejores_que_acs = [i for i in s.index if i.startswith("sin") and s[i] < acs_v]
    reglas = [i for i in s.index if not i.startswith("sin") and i != "ACS (completo)"]
    gana_reglas = sum(s[i] > acs_v for i in reglas)
    i2 = html.Div(f"Gris: el ACS sin uno de sus componentes; azul: reglas estándar. Retirar {len(peores)} de los "
                  f"{len(peores) + len(mejores_que_acs)} componentes empeora el error"
                  + (f" ({', '.join(mejores_que_acs)} lo mejoran levemente)" if mejores_que_acs else "")
                  + f". El ACS supera a {gana_reglas} de las {len(reglas)} reglas estándar. La línea punteada es el mejor "
                  f"benchmark: el ACS queda {acs_v - rf_v:+.4f} en RMSE log frente a él.", className="insight")
    filas = []
    for m in ["rmse", "mae"]:
        t = tabla(f"original_pruebas_{m}.csv")
        for _, r in t.iterrows():
            ef = r.get("d de Cohen")
            ef = f"d = {ef:.2f}" if pd.notna(ef) else (f"δ = {r['delta de Cliff']:.3f} ({r['magnitud']})" if pd.notna(r.get("delta de Cliff")) else "—")
            filas.append({"métrica": r["métrica"], "prueba": r["prueba"].replace(" vs Random Forest · Grid", ""), "H0": r["H0"],
                          "supuestos": r["supuestos"], "estadístico": "—" if pd.isna(r["estadístico"]) else f"{r['estadístico']:.3f}",
                          "p crudo": f"{r['p crudo']:.4f}", "p Holm": f"{r['p Holm']:.4f}", "p BH": f"{r['p BH']:.4f}",
                          "tamaño del efecto": ef,
                          "conclusión": ("rechaza H0" if r["p Holm"] < 0.05 else "no rechaza") if "Model Confidence" not in r["prueba"]
                          else r["conclusión"]})
    pr = pd.DataFrame(filas)
    tab = dash_table.DataTable(data=pr.to_dict("records"), columns=[{"name": c, "id": c} for c in pr.columns],
                               style_cell={"fontSize": 12, "textAlign": "left", "whiteSpace": "normal", "height": "auto", "maxWidth": 260},
                               style_table={"overflowX": "auto"},
                               style_data_conditional=[{"if": {"filter_query": '{conclusión} = "rechaza H0"'}, "color": "#1baf7a", "fontWeight": "bold"}])
    ver = veredicto(res)
    return fig, i1, f2, i2, tab, ver

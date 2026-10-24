# Sección 2: EDA. Filtros globales (grupo, fechas, tamaño del negocio) que actualizan todas las figuras.
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from scipy import stats

import plots as P
from data import DESCRIPCION, DIAS, X_COLS, tabla

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Pronóstico de ventas")

mod = tabla("modelado.csv.gz")
panel = tabla("panel.csv.gz")
stl, acfs = tabla("stl.csv"), tabla("acf.csv")
GRUPOS = sorted(mod["grupo"].unique())
nivel_neg = np.log10(panel.groupby("negocio")["venta_total"].mean().clip(lower=1))
VARS = X_COLS + ["venta_total"]
FMIN, FMAX = panel["fecha"].min().date(), panel["fecha"].max().date()


def insight(texto, alerta=False):
    return html.Div(texto, className="insight" + (" insight-alerta" if alerta else ""))


def tarjeta(titulo, grafico_id, insight_id, extra=None):
    return dbc.Card(dbc.CardBody([html.H5(titulo), *(extra or []), dcc.Loading(dcc.Graph(id=grafico_id, config={"displaylogo": False})),
                                  html.Div(id=insight_id)]), className="mb-3")


filtros = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([html.Label("Grupo de negocio", htmlFor="f-grupo"),
             dcc.Dropdown(id="f-grupo", options=[{"label": g, "value": g} for g in GRUPOS], value=GRUPOS, multi=True)], lg=4),
    dbc.Col([html.Label("Rango de fechas", htmlFor="f-fechas"),
             dcc.DatePickerRange(id="f-fechas", min_date_allowed=FMIN, max_date_allowed=FMAX, start_date=FMIN, end_date=FMAX,
                                 display_format="DD MMM YYYY")], lg=4),
    dbc.Col([html.Label("Venta media diaria del negocio (COP, escala log₁₀)", htmlFor="f-nivel"),
             dcc.RangeSlider(id="f-nivel", min=3, max=9, step=0.5, value=[3, 9],
                             marks={i: f"10^{i}" for i in range(3, 10)})], lg=4),
])), className="mb-3 sticky-filtros")

layout = html.Div([
    html.H2("Análisis exploratorio"),
    html.P("Los filtros de arriba actualizan todas las figuras. Las unidades son negocio-días (una fila = un negocio en un día).",
           className="text-muted"),
    filtros,
    html.Div(id="eda-resumen", className="mb-2"),
    dbc.Row([
        dbc.Col(tarjeta("Distribución de la variable objetivo", "g-objetivo", "i-objetivo",
                        [dbc.RadioItems(id="escala", options=[{"label": " escala log", "value": "log"}, {"label": " escala COP", "value": "cop"}],
                                        value="log", inline=True)]), lg=6),
        dbc.Col(tarjeta("Histograma de una variable", "g-hist", "i-hist",
                        [dcc.Dropdown(id="var-hist", options=[{"label": f"{c} — {DESCRIPCION[c]}", "value": c} for c in X_COLS],
                                      value="std7", clearable=False)]), lg=6),
    ]),
    dbc.Row([
        dbc.Col(tarjeta("Correlación de Spearman", "g-corr", "i-corr"), lg=7),
        dbc.Col([tarjeta("Valores faltantes", "g-faltantes", "i-faltantes"),
                 tarjeta("Atípicos por grupo (z robusta por negocio, |z| > 3.5)", "g-atipicos", "i-atipicos")], lg=5),
    ]),
    tarjeta("Explorador bivariado", "g-bivar", "i-bivar", [dbc.Row([
        dbc.Col([html.Label("Eje x"), dcc.Dropdown(id="bx", options=VARS, value="lag1", clearable=False)], md=4),
        dbc.Col([html.Label("Eje y"), dcc.Dropdown(id="by", options=VARS, value="venta_total", clearable=False)], md=4),
        dbc.Col([html.Label("Color"), dcc.Dropdown(id="bc", options=["grupo", "dia_semana", "festivo", "dias_sin_venta7"],
                                                   value="grupo", clearable=False)], md=4)])]),
    html.H3("Serie de tiempo", className="mt-4"),
    tarjeta("Venta total diaria y descomposición STL (periodo semanal)", "g-stl", "i-stl"),
    dbc.Row([dbc.Col(tarjeta("ACF y PACF de la serie sin tendencia", "g-acf", "i-acf"), lg=7),
             dbc.Col(dbc.Card(dbc.CardBody([html.H5("Pruebas de estacionariedad"), html.Div(id="t-estac")])), lg=5)]),
])


def filtrar(grupos, ini, fin, nivel):
    negs = nivel_neg[(nivel_neg >= nivel[0]) & (nivel_neg <= nivel[1])].index
    m = mod[mod["grupo"].isin(grupos or []) & mod["negocio"].isin(negs) & mod["fecha"].between(ini, fin)]
    p = panel[panel["grupo"].isin(grupos or []) & panel["negocio"].isin(negs) & panel["fecha"].between(ini, fin)]
    return m, p


@callback(Output("eda-resumen", "children"), Output("g-objetivo", "figure"), Output("i-objetivo", "children"),
          Output("g-hist", "figure"), Output("i-hist", "children"), Output("g-corr", "figure"), Output("i-corr", "children"),
          Output("g-faltantes", "figure"), Output("i-faltantes", "children"), Output("g-atipicos", "figure"),
          Output("i-atipicos", "children"),
          Input("f-grupo", "value"), Input("f-fechas", "start_date"), Input("f-fechas", "end_date"), Input("f-nivel", "value"),
          Input("escala", "value"), Input("var-hist", "value"))
def actualizar(grupos, ini, fin, nivel, escala, var):
    m, p = filtrar(grupos, ini, fin, nivel)
    if len(m) < 50:
        msg = dbc.Alert("Con estos filtros quedan menos de 50 negocio-días: amplíe el rango.", color="warning")
        v = P.vacia()
        return msg, v, "", v, "", v, "", v, "", v, ""
    resumen = dbc.Alert(f"{m['negocio'].nunique()} negocios · {m['fecha'].nunique()} días · {len(m):,} negocio-días seleccionados",
                        color="light", className="py-2")
    y = m["venta_total"]
    # objetivo
    g1 = P.histograma(y, "Venta total del día", "log(1 + venta)" if escala == "log" else "venta (COP)", log=escala == "log")
    ceros = (y <= 0).mean()
    i1 = insight(f"En COP la venta es muy asimétrica (asimetría {stats.skew(y):.1f}): unos pocos negocios grandes "
                             f"dominan. En log se ven los grupos de tamaño. El {ceros:.1%} de los días no hubo venta: forman un pico "
                             f"aislado en 0 que concentra gran parte del error de los modelos.")
    # histograma de una variable
    s = m[var].dropna()
    g2 = P.histograma(s, f"{var}: {DESCRIPCION[var]}", var, color=P.VERDE)
    i2 = insight(f"Mediana {s.median():.3g}, rango intercuartílico [{s.quantile(.25):.3g}, {s.quantile(.75):.3g}]. "
                 + ("La mayoría de las semanas no tienen días sin venta; los pocos negocios con días sin venta son los más volátiles."
                    if var == "dias_sin_venta7" else "Las variables de nivel (lags, medias) heredan la forma multimodal del tamaño de los negocios."
                    if var in ("lag1", "lag7", "lag14", "media7", "nivel") else
                    "Distribución concentrada con cola larga: los negocios irregulares son pocos pero los más difíciles de pronosticar."
                    if var == "std7" else "El mix de medios distingue tipos de negocio (KCO todo en efectivo; EDS con crédito de flotas)."))
    # correlación
    c = m[X_COLS + ["venta_total"]].assign(venta_total=np.log1p(m["venta_total"].clip(lower=0))).corr(method="spearman")
    g3 = P.heatmap_corr(c.round(2), "Spearman entre variables (venta en log)")
    top = c["venta_total"].drop("venta_total").abs().sort_values(ascending=False)
    i3 = insight(f"Las más asociadas a la venta son {top.index[0]} (ρ = {c.loc[top.index[0], 'venta_total']:.2f}) y "
                 f"{top.index[1]} ({c.loc[top.index[1], 'venta_total']:.2f}). Las variables de nivel están muy correlacionadas "
                 "entre sí (colinealidad): por eso los modelos lineales reparten el peso de forma inestable.")
    # faltantes
    falt = m[X_COLS].isna().mean().mul(100)
    falt = falt[falt > 0] if (falt > 0).any() else falt.head(3)
    g4 = P.barras_h(falt, "Porcentaje de valores faltantes por variable", "% de filas", color=P.NARANJA, alto=260)
    i4 = insight("Solo las participaciones de medios tienen faltantes (semanas con venta total 0, donde la participación no "
                 "está definida). Se imputan con la mediana de train dentro del pipeline, sin fuga.")
    # atípicos: z robusta del log de la venta dentro de cada negocio
    lv = np.log1p(m["venta_total"].clip(lower=0))
    med = lv.groupby(m["negocio"]).transform("median")
    mad = (lv - med).abs().groupby(m["negocio"]).transform("median") * 1.4826
    z = (lv - med) / mad.replace(0, np.nan)
    at = (z.abs() > 3.5).groupby(m["grupo"]).mean().mul(100)
    g5 = P.barras_h(at, "% de negocio-días atípicos", "% de negocio-días", color=P.GRIS, alto=260)
    i5 = insight(f"El {(z.abs() > 3.5).mean():.1%} de los negocio-días es atípico frente a su propio negocio, casi siempre por "
                 "debajo de lo normal (cierres, días sin venta). No se eliminan: son reales y el modelo debe verlos.", alerta=True)
    return resumen, g1, i1, g2, i2, g3, i3, g4, i4, g5, i5


@callback(Output("g-bivar", "figure"), Output("i-bivar", "children"),
          Input("f-grupo", "value"), Input("f-fechas", "start_date"), Input("f-fechas", "end_date"), Input("f-nivel", "value"),
          Input("bx", "value"), Input("by", "value"), Input("bc", "value"))
def bivariado(grupos, ini, fin, nivel, x, y, color):
    m, _ = filtrar(grupos, ini, fin, nivel)
    if len(m) < 50:
        return P.vacia(), ""
    d = m.sample(min(4000, len(m)), random_state=42).copy()
    for col in (x, y):
        if col == "venta_total":
            d[col] = np.log1p(d[col].clip(lower=0))
    d[color] = d[color].map(lambda v: DIAS[int(v)]) if color == "dia_semana" else d[color].astype(str)
    import plotly.express as px
    fig = px.scatter(d, x=x, y=y, color=color, opacity=0.55, template=P.PLANTILLA, color_discrete_sequence=P.PALETA + px.colors.qualitative.Safe,
                     labels={"venta_total": "log(1 + venta)"}, hover_data=["negocio", "fecha"])
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=430, title=f"{y} contra {x} (muestra de {len(d):,} negocio-días)")
    rho = stats.spearmanr(d[x], d[y], nan_policy="omit")[0]
    return fig, insight(f"Spearman ρ = {rho:.2f}. Ojo: con todos los negocios juntos casi cualquier par de variables de nivel "
                        "sale muy correlacionado porque refleja el tamaño del negocio; dentro de un mismo negocio la relación es más débil.")


@callback(Output("g-stl", "figure"), Output("i-stl", "children"), Output("g-acf", "figure"), Output("i-acf", "children"),
          Output("t-estac", "children"), Input("f-grupo", "value"))
def temporal(grupos):
    g = grupos[0] if grupos and len(grupos) == 1 else "todos"
    s = stl[stl["grupo"] == g]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=["log de la venta total diaria", "tendencia", "componente semanal", "residuo"])
    for i, (c, col) in enumerate([("serie", P.OSCURO), ("tendencia", P.AZUL), ("estacional", P.VERDE), ("residuo", P.GRIS)]):
        fig.add_trace(go.Scatter(x=s["fecha"], y=s[c], line=dict(color=col), name=c, showlegend=False,
                                 hovertemplate="%{x|%d %b}: %{y:.3f}<extra></extra>"), row=i + 1, col=1)
    fig.update_layout(template=P.PLANTILLA, height=620, title=f"Descomposición STL · {('grupo ' + g) if g != 'todos' else 'todos los negocios'}")
    a = acfs[acfs["grupo"] == g]
    fuerza = a["fuerza_semanal"].iloc[0]
    i_stl = insight(f"Fuerza de la estacionalidad semanal: {fuerza:.2f} (0 = nada, 1 = todo). El ciclo semanal es la estructura "
                    "dominante; la tendencia casi no cambia en el trimestre. Las caídas puntuales en el residuo son los festivos."
                    + (" Seleccione un solo grupo para ver su descomposición." if g == "todos" else ""))
    fa = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
    k = a["rezago"].to_numpy()[1:]
    banda = 1.96 / np.sqrt(85)
    for j, c in enumerate(["acf", "pacf"]):
        fa.add_trace(go.Bar(x=k, y=a[c].to_numpy()[1:], marker_color=P.AZUL, showlegend=False,
                            hovertemplate="rezago %{x}: %{y:.2f}<extra></extra>"), row=1, col=j + 1)
        fa.add_hrect(y0=-banda, y1=banda, fillcolor=P.GRIS, opacity=0.18, line_width=0, row=1, col=j + 1)
    fa.update_layout(template=P.PLANTILLA, height=330, title="Autocorrelación de la serie sin tendencia (banda = ±1,96/√n)")
    fa.update_xaxes(title="rezago (días)")
    i_acf = insight("Picos en los rezagos 7, 14 y 21: el patrón semanal se repite. Por eso los modelos usan lag7 y lag14, y la "
                    "validación cruzada respeta el orden temporal.")
    adf, kp = a["adf_p"].iloc[0], a["kpss_p"].iloc[0]
    t = dbc.Table([html.Thead(html.Tr([html.Th("prueba"), html.Th("H0"), html.Th("p-valor"), html.Th("lectura")])),
                   html.Tbody([html.Tr([html.Td("ADF"), html.Td("hay raíz unitaria (no estacionaria)"), html.Td(f"{adf:.3f}"),
                                        html.Td("rechaza: estacionaria" if adf < 0.05 else "no rechaza")]),
                               html.Tr([html.Td("KPSS"), html.Td("es estacionaria"), html.Td(f"{kp:.3f}"),
                                        html.Td("rechaza: no estacionaria" if kp < 0.05 else "no rechaza: estacionaria")])])],
                  bordered=True, size="sm")
    return fig, i_stl, fa, i_acf, [t, insight("Las dos pruebas tienen hipótesis nulas opuestas; con 85 días tienen poca potencia. "
                                              "El libro (cap. 6) repite las pruebas negocio por negocio.")]

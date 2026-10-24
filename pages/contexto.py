# Sección 1: Contexto. Narrativa estática: problema, pregunta, datos, metodología y limitaciones.
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, html

from data import DESCRIPCION, lideres, tabla

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Pronóstico de ventas")

AUTOR = "edsi23"  # autor del proyecto
VERSION, FECHA = "3.0 (tercera entrega)", "25 de octubre de 2026"

panel = tabla("panel.csv.gz")
lid = lideres().set_index("clave")


def kpi(titulo, valor, nota):
    return dbc.Col(dbc.Card(dbc.CardBody([html.Div(titulo, className="kpi-nota"), html.Div(valor, className="kpi-valor"),
                                          html.Div(nota, className="kpi-nota")])), xs=12, sm=6, lg=3, className="mb-3")


def paso(texto, color):
    return dbc.Col(html.Div(texto, className="paso", style={"background": color}), xs=6, md=True, className="mb-2")


diccionario = pd.DataFrame([{"variable": "venta_total (objetivo)", "tipo": "real (COP)",
                             "descripción": "venta total del negocio en el día a pronosticar, por todos los medios de pago"}]
                           + [{"variable": k, "tipo": "entero" if k in ("festivo", "dia_semana", "dias_sin_venta7") else "real",
                               "descripción": v} for k, v in DESCRIPCION.items()])

layout = html.Div([
    html.Div(className="hero mb-4", children=[
        html.H1("¿Cuánto venderá mañana cada estación?"),
        html.P("Pronóstico de la venta total del día siguiente para 452 negocios (estaciones de servicio, tiendas de "
               "conveniencia y kioscos) operados por Masser, a partir de su historia reciente por medio de pago.",
               className="lead mb-0"),
    ]),
    dbc.Row([
        kpi("Estaciones de servicio en Colombia", "≈ 6 518", "Comce, citado por La República (4 mar 2024) [1]"),
        kpi("Negocios analizados", f"{panel['negocio'].nunique()}", "1 jul – 23 sep 2026 · 85 días"),
        kpi("Error del mejor modelo", f"{lid.loc['rf', 'RMSE log media']:.2f}".replace(".", ","),
            "RMSE log medio del bucle externo (Random Forest)"),
        kpi("Modelos comparados", "32 + 1 propio", "28 de la entrega 2, 4 de estado del arte (HGB) y el ACS"),
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Por qué importa"),
            html.P("La distribución minorista de combustibles en Colombia está concentrada en pocas marcas: con unas 6 518 "
                   "estaciones de servicio en el país, Terpel opera 2 365 (36,2 %) [1]. Un operador de estaciones, tiendas y "
                   "kioscos necesita saber cuánto venderá cada punto al día siguiente para planear el efectivo que debe "
                   "recaudar, el inventario y el personal, y para detectar a tiempo un negocio que vende mucho menos de lo normal."),
            html.P("El problema es difícil: los negocios difieren hasta 9 000 veces en tamaño, la venta cambia con el día de la "
                   "semana y los festivos, y algunos días un negocio no vende nada sin aviso previo."),
        ])), lg=6, className="mb-3"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Pregunta de investigación y objetivos"),
            html.P(html.B("¿Con qué precisión se puede pronosticar la venta total del día siguiente de cada negocio a partir de "
                          "su historia reciente, y supera un ensamble que adapta los pesos al régimen del negocio a los mejores "
                          "modelos individuales?")),
            html.Ul([html.Li("Construir y comparar 28 modelos de regresión (7 algoritmos × 4 optimizadores) con validación anidada."),
                     html.Li("Diseñar e implementar un modelo original (Árbol de Combinación en el Símplex, ACS) y validarlo "
                             "estadísticamente contra los benchmarks."),
                     html.Li("Poner los modelos a disposición en este tablero, con predicción en vivo y su explicación.")]),
        ])), lg=6, className="mb-3"),
    ]),
    dbc.Card(dbc.CardBody([
        html.H4("Datos"),
        dbc.Row([
            dbc.Col([
                html.P(["Fuente: recibos de los puntos de venta (POS) de Masser, ", html.B("uso autorizado por la empresa"),
                        ". Periodo: 1 de julio al 23 de septiembre de 2026. Unidad: negocio × día (37 828 filas, 452 negocios, "
                        "anonimizados N001…N453). Tarea: ", html.B("regresión"), " (venta total del día siguiente, h = 1)."]),
                html.P("Partición temporal: train 1 jul – 2 sep (los primeros 14 días solo dan historia a los rezagos) y test "
                       "3 – 23 sep. Las variables solo usan información hasta el día anterior (verificado con una prueba de "
                       "perturbación en el capítulo 5 del libro)."),
            ], lg=5),
            dbc.Col(dash_table.DataTable(data=diccionario.to_dict("records"),
                                         columns=[{"name": c, "id": c} for c in diccionario.columns],
                                         style_cell={"textAlign": "left", "fontSize": 12, "whiteSpace": "normal", "height": "auto"},
                                         style_header={"fontWeight": "bold"}, page_size=15), lg=7),
        ]),
    ]), className="mb-3"),
    dbc.Card(dbc.CardBody([
        html.H4("Metodología"),
        dbc.Row([paso("ETL: recibos → panel negocio × día", "#1f2a37"), paso("EDA: univariado, temporal, multivariado", "#2a78d6"),
                 paso("28 modelos × 4 optimizadores", "#2a78d6"), paso("Validación anidada temporal 4 × 3", "#1baf7a"),
                 paso("Comparación: MCS, DM, GW, Wilcoxon", "#1baf7a"), paso("Modelo original: ACS", "#eb6834"),
                 paso("Artefactos + este tablero", "#8a8983")], className="g-2"),
        html.P("Cada etapa está documentada en el libro del proyecto, con el código que reproduce todas las tablas y figuras.",
               className="text-muted small mt-2"),
    ]), className="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Limitaciones, ética y sesgos"),
            html.Ul([
                html.Li("Un solo trimestre y 3 festivos (todos en train): no hay información sobre temporadas altas ni el efecto festivo en test."),
                html.Li("Una sola operadora: los resultados no se extrapolan a otras redes ni a negocios nuevos sin historia."),
                html.Li("Los negocios están anonimizados; los datos se publican con autorización de la empresa y no contienen "
                        "información de clientes ni de empleados."),
                html.Li("Riesgo de uso: un pronóstico bajo no debe usarse para sancionar a un punto de venta; es un apoyo para planear."),
                html.Li("Sesgo de supervivencia: se excluyó un negocio (N093) por una migración de POS inconsistente."),
            ]),
        ])), lg=8, className="mb-3"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Autoría y versión"),
            html.P([html.B("Autor: "), AUTOR]), html.P([html.B("Versión: "), VERSION]), html.P([html.B("Actualizado: "), FECHA]),
            html.P(["Código y libro: ", html.A("github.com/edsi23/recaudo-forecast", href="https://github.com/edsi23/recaudo-forecast", target="_blank")]),
            html.Hr(),
            html.Small(["[1] La República (4 de marzo de 2024). Terpel, Biomax y Primax lideran el mercado de las estaciones de "
                        "gasolina en Colombia. Datos de Comce. ",
                        html.A("larepublica.co", href="https://www.larepublica.co/empresas/terpel-biomax-y-primax-lideran-el-mercado-de-las-estaciones-de-gasolina-en-colombia-3813893",
                               target="_blank")]),
        ])), lg=4, className="mb-3"),
    ]),
])

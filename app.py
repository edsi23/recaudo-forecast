# Punto de entrada del dashboard: crea la app multipágina y expone "server" para gunicorn (Render).
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
for p in (RAIZ / "src", RAIZ / "book"):
    sys.path.insert(0, str(p))

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import html  # noqa: E402

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY],
                title="Pronóstico de ventas · Masser", suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server  # para Render: gunicorn app:server

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand([html.Img(src="/assets/logo.svg", height="28px", className="me-2", alt="logo del proyecto"),
                         "Pronóstico de ventas diarias"], href="/"),
        dbc.Nav([dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
                 dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
                 dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact")),
                 dbc.NavItem(dbc.NavLink("Libro ↗", href="https://edsi23.github.io/recaudo-forecast/", target="_blank"))],
                className="ms-auto", navbar=True),
    ], fluid=True),
    color="primary", dark=True, sticky="top", className="mb-3 shadow-sm",
)

app.layout = html.Div([navbar, dbc.Container(dash.page_container, fluid=True, className="px-4 pb-5")])

if __name__ == "__main__":
    app.run(debug=True)

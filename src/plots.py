# Funciones que construyen las figuras del dashboard. Una sola plantilla y una paleta de 5 colores.
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

AZUL, NARANJA, VERDE, GRIS, OSCURO = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983", "#1f2a37"
PALETA = [AZUL, NARANJA, VERDE, GRIS, OSCURO]

pio.templates["recaudo"] = go.layout.Template(
    layout=dict(font=dict(family="Inter, system-ui, sans-serif", size=13, color=OSCURO),
                colorway=PALETA, paper_bgcolor="white", plot_bgcolor="white",
                title=dict(font=dict(size=15), x=0.01, xanchor="left"),
                xaxis=dict(gridcolor="#ebeae6", zeroline=False, linecolor="#c9c8c3", automargin=True),
                yaxis=dict(gridcolor="#ebeae6", zeroline=False, linecolor="#c9c8c3", automargin=True),
                legend=dict(orientation="h", y=-0.22, x=0), margin=dict(l=55, r=20, t=55, b=45),
                hoverlabel=dict(font_size=12)))
PLANTILLA = "recaudo"


def _base(fig, titulo, x=None, y=None, alto=360):
    fig.update_layout(template=PLANTILLA, title=titulo, height=alto)
    if x is not None:
        fig.update_xaxes(title=x)
    if y is not None:
        fig.update_yaxes(title=y)
    return fig


def cop(v):
    """Monto en pesos con separador de miles en español (espacio fino)."""
    return f"{v:,.0f}".replace(",", "\u202f") + " COP"


def vacia(texto="Sin datos para estos filtros"):
    fig = go.Figure()
    fig.add_annotation(text=texto, showarrow=False, font=dict(size=14, color=GRIS))
    fig.update_layout(template=PLANTILLA, height=300, xaxis_visible=False, yaxis_visible=False)
    return fig


def histograma(serie, titulo, eje, log=False, color=AZUL, nbins=50):
    v = np.log1p(serie.clip(lower=0)) if log else serie
    fig = go.Figure(go.Histogram(x=v, nbinsx=nbins, marker_color=color, hovertemplate="%{x}: %{y} filas<extra></extra>"))
    return _base(fig, titulo, eje, "número de negocio-días")


def linea_total(df, titulo, columnas, nombres, colores, eje_y="miles de M COP", escala=1e9):
    fig = go.Figure()
    for c, n, col in zip(columnas, nombres, colores):
        fig.add_trace(go.Scatter(x=df["fecha"], y=df[c] / escala, name=n, line=dict(color=col, width=2),
                                 hovertemplate="%{x|%d %b}: %{y:,.2f}<extra>" + n + "</extra>"))
    return _base(fig, titulo, "fecha", eje_y)


def heatmap_corr(corr, titulo):
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=[[0, NARANJA], [0.5, "white"], [1, AZUL]],
                               text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=9),
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    return _base(fig, titulo, alto=520)


def barras_h(serie, titulo, eje_x, color=AZUL, alto=380):
    s = serie.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=color,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    return _base(fig, titulo, eje_x, None, alto)


def shap_local(base, sv, titulo):
    """Cascada SHAP: de la predicción media (base) a la predicción de la fila, en log."""
    s = sv.reindex(sv.abs().sort_values(ascending=False).index)
    top, resto = s.iloc[:8], s.iloc[8:].sum()
    etiquetas = list(top.index) + (["otras variables"] if abs(resto) > 0 else [])
    valores = list(top.values) + ([resto] if abs(resto) > 0 else [])
    fig = go.Figure(go.Waterfall(orientation="h", y=["valor base"] + etiquetas + ["predicción"],
                                 x=[base] + valores + [0], measure=["absolute"] + ["relative"] * len(valores) + ["total"],
                                 increasing=dict(marker_color=AZUL), decreasing=dict(marker_color=NARANJA),
                                 totals=dict(marker_color=OSCURO), hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    # el eje empieza cerca del valor base: si arranca en 0, la base (~15) aplasta las contribuciones
    acum = base + np.cumsum([0] + valores)
    lo, hi = min(acum.min(), base + sum(valores)), max(acum.max(), base + sum(valores))
    pad = max(0.15, 0.15 * (hi - lo))
    fig.update_xaxes(range=[lo - pad, hi + pad])
    fig.update_traces(text=[f"{v:.2f}" for v in [base] + valores + [base + sum(valores)]], textposition="outside")
    return _base(fig, titulo, "log(1 + venta): valor base → predicción", None, 420)


def real_vs_pred(y, p, titulo):
    ly, lp = np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(p, 0, None))
    fig = go.Figure(go.Scattergl(x=lp, y=ly, mode="markers", marker=dict(size=4, color=AZUL, opacity=0.35),
                                 hovertemplate="predicho %{x:.2f}<br>real %{y:.2f}<extra></extra>"))
    lim = [min(lp.min(), ly.min()), max(lp.max(), ly.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GRIS, dash="dash"), name="y = x"))
    fig.update_layout(showlegend=False)
    return _base(fig, titulo, "predicho: log(1 + venta)", "real: log(1 + venta)")


def residuos_vs_ajuste(p, e, titulo):
    fig = go.Figure(go.Scattergl(x=np.log1p(np.clip(p, 0, None)), y=e, mode="markers",
                                 marker=dict(size=4, color=VERDE, opacity=0.35),
                                 hovertemplate="ajustado %{x:.2f}<br>residuo %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color=GRIS)
    return _base(fig, titulo, "ajustado: log(1 + venta)", "residuo (log)")


def barras_acf(valores, titulo, banda):
    k = np.arange(1, len(valores) + 1)
    fig = go.Figure(go.Bar(x=k, y=valores, marker_color=AZUL, hovertemplate="rezago %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hrect(y0=-banda, y1=banda, fillcolor=GRIS, opacity=0.18, line_width=0)
    return _base(fig, titulo, "rezago (días)", "autocorrelación")

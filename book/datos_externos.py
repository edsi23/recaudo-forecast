# Entrega 3: tres conjuntos de datos públicos con la misma tarea (pronóstico del valor diario del día
# siguiente para un panel de entidades), llevados al mismo formato y diseño que el panel de Masser:
# 85 días por entidad = 14 de historia + 50 de train + 21 de test.
#
# Fuentes (descarga pública, sin registro):
#   M5 (Walmart):     https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip  (Makridakis et al., 2022)
#   Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii          (Chen, 2019)
#   Electricity:      https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 (Trindade, 2015)
# Uso:  python datos_externos.py
import numpy as np
import pandas as pd

from comun import DATA, MEDIOS

EXT = DATA / "externos"
DIAS = 85


def a_panel(largo, nombre):
    """Convierte (entidad, fecha, valor) al formato del panel de la entrega 1.

    Las columnas de medios de pago no existen en estos datos: se fijan en 0, así que las variables
    ``part_*`` quedan constantes y no aportan a ningún modelo (todos reciben las mismas columnas).
    """
    p = largo.rename(columns={"entidad": "negocio", "valor": "venta_total"})
    fechas = np.sort(p["fecha"].unique())[-DIAS:]
    p = p[p["fecha"].isin(fechas)]
    # rejilla completa entidad x día: lo que falta es 0 (no hubo ventas / consumo)
    idx = pd.MultiIndex.from_product([p["negocio"].unique(), fechas], names=["negocio", "fecha"])
    p = p.set_index(["negocio", "fecha"])["venta_total"].reindex(idx, fill_value=0.0).reset_index()
    p["sin_registro"] = (p["venta_total"] == 0).astype(int)
    p["n_fuentes"] = 1
    for m in MEDIOS + ["otros"]:
        p[f"pm_{m}"] = 0.0
    p.to_csv(EXT / f"panel_{nombre}.csv.gz", index=False)
    print(f"{nombre}: {p['negocio'].nunique()} entidades, {p['fecha'].nunique()} días "
          f"({p['fecha'].min():%Y-%m-%d} a {p['fecha'].max():%Y-%m-%d}), {len(p):,} filas, "
          f"ceros {p['sin_registro'].mean():.1%}")
    return p


def m5():
    # unidades vendidas por tienda x departamento (10 tiendas x 7 departamentos = 70 series)
    ventas = pd.read_csv(EXT / "m5" / "sales_train_evaluation.csv")
    # el calendario de esta copia no trae la columna d: la fila i corresponde a d_{i+1} (d_1 = 2011-01-29)
    cal = pd.read_csv(EXT / "m5" / "calendar.csv", usecols=["date"])
    cal["d"] = [f"d_{i + 1}" for i in range(len(cal))]
    dias = [c for c in ventas.columns if c.startswith("d_")][-DIAS:]
    agg = ventas.groupby(["store_id", "dept_id"])[dias].sum()
    agg.index = agg.index.map(lambda t: f"{t[0]}_{t[1]}")
    largo = agg.reset_index().melt(id_vars="index", var_name="d", value_name="valor").rename(columns={"index": "entidad"})
    largo["fecha"] = pd.to_datetime(largo["d"].map(cal.set_index("d")["date"]))
    return a_panel(largo[["entidad", "fecha", "valor"]], "m5")


def retail():
    # ingreso diario (cantidad x precio, incluidas devoluciones) de los 60 productos con más ingreso
    df = pd.read_excel(EXT / "online_retail_ii" / "online_retail_II.xlsx", sheet_name="Year 2010-2011")
    df["fecha"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df["valor"] = df["Quantity"] * df["Price"]
    df = df[df["fecha"] >= df["fecha"].max() - pd.Timedelta(days=DIAS + 30)]
    top = df.groupby("StockCode")["valor"].sum().nlargest(60).index
    largo = df[df["StockCode"].isin(top)].groupby(["StockCode", "fecha"])["valor"].sum().reset_index()
    largo = largo.rename(columns={"StockCode": "entidad"})
    largo["entidad"] = largo["entidad"].astype(str)
    # el comercio no abre algunos días (sábados): se completan con 0 como cualquier día sin venta
    todas = pd.date_range(largo["fecha"].min(), largo["fecha"].max(), freq="D")
    largo = (largo.set_index(["entidad", "fecha"])["valor"]
             .reindex(pd.MultiIndex.from_product([largo["entidad"].unique(), todas], names=["entidad", "fecha"]), fill_value=0)
             .reset_index())
    return a_panel(largo, "retail")


def electricidad():
    # consumo diario (kWh) de los clientes, a partir de lecturas de potencia cada 15 minutos (kW)
    df = pd.read_csv(EXT / "electricity" / "LD2011_2014.txt", sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df[df.index >= df.index.max() - pd.Timedelta(days=DIAS + 2)]
    diario = (df.resample("D").sum() / 4.0).iloc[:-1]  # kW cada 15 min -> kWh; el último día está incompleto
    diario = diario.loc[:, (diario > 0).mean() > 0.5]   # clientes activos en el periodo
    largo = diario.reset_index().melt(id_vars=diario.index.name or "index", var_name="entidad", value_name="valor")
    largo = largo.rename(columns={diario.index.name or "index": "fecha"})
    return a_panel(largo, "electricidad")


if __name__ == "__main__":
    m5()
    retail()
    electricidad()

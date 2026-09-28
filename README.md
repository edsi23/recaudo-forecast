# Pronóstico de ventas diarias por negocio

Primer entregable del proyecto de Machine Learning: pronóstico de la venta total del día siguiente para 452 negocios (EDS, tiendas de conveniencia y kioscos) a partir de su historia por medio de pago. Regresión sobre panel temporal con línea base ingenua y SVR lineal.

**Libro publicado:** https://edsi23.github.io/recaudo-forecast/

## Estructura

| Ruta | Contenido |
|---|---|
| `book/01…08_*.ipynb` | Capítulos del libro (datos, limpieza, partición, EDA, fuga, EDA temporal, preprocesamiento, modelo base) |
| `book/comun.py` | Código compartido: variables del modelo, folds temporales, Pipeline y estilo de gráficos |
| `requirements.txt` | Versiones exactas de las librerías |

## Datos

Los datos son de uso privado y **no están en el repositorio**. El acceso se da bajo solicitud, con autorización de la empresa. Los notebooks se ejecutaron localmente y se publican con sus salidas.

Para reproducir con acceso a los datos: colocar la carpeta `Data/` al lado del repositorio (`../Data` respecto a la raíz), ejecutar los notebooks en orden empezando por el 02 (que genera `Data/processed/panel_negocio_dia.csv`) y construir con `jupyter-book build book`. Semilla fija: `SEED = 42`.

# Pronóstico de ventas diarias por negocio

Proyecto de Machine Learning sobre la venta diaria de 452 negocios (EDS, tiendas de conveniencia y kioscos) a partir de su historia por medio de pago:

- **Regresión**: venta total del día siguiente (7 modelos × 4 optimizadores = 28 combinaciones).
- **Clasificación**: alerta de caída fuerte del día siguiente (7 modelos × 4 balanceos × 4 optimizadores = 112 combinaciones).

Todo con validación cruzada anidada y temporal.

**Libro publicado:** https://edsi23.github.io/recaudo-forecast/
**Primera entrega (versión calificada):** tag [`entrega-1`](https://github.com/edsi23/recaudo-forecast/tree/entrega-1)

## Estructura

| Ruta | Contenido |
|---|---|
| `book/01…08_*.ipynb` | Primera entrega: datos, limpieza, partición, EDA, fuga, EDA temporal, preprocesamiento, modelo base |
| `book/04b_*.ipynb` | EDA multivariado y días sin venta (corrección de la primera entrega) |
| `book/09…15_*.ipynb` | Segunda entrega: diseño experimental, optimizadores, optimización computacional, regresión, clasificación, interpretabilidad, conclusiones |
| `book/comun.py` | Fechas, variables del modelo, estilo de gráficos |
| `book/datos.py` | Objetivos, partición y folds de la validación anidada |
| `book/modelos.py` | Los 14 modelos, espacios de búsqueda y pipelines con balanceo |
| `book/optimizadores.py` | Grid, Random, Bayesiana (Optuna/TPE) y Genética (DEAP) |
| `book/metricas.py`, `book/estadistica.py` | Métricas y pruebas estadísticas |
| `book/experimentos.py` | Corre las 140 combinaciones y arma la tabla maestra |
| `book/estudio_optimizadores.py` | Estudio de convergencia de los optimizadores (60 evaluaciones, 3 semillas) |
| `book/resultados/maestra.csv` | Tabla maestra: hiperparámetros, métricas del bucle externo y de test, tiempos y semilla de cada combinación |
| `requirements.txt` | Versiones exactas de las librerías |

## Datos

Los datos son de uso privado y **no están en el repositorio**. El acceso se da bajo solicitud, con autorización de la empresa. Los notebooks se ejecutaron localmente y se publican con sus salidas.

## Reproducir

Con acceso a los datos, colocar la carpeta `Data/` al lado del repositorio (`../Data` respecto a la raíz) y, desde `book/`:

1. Ejecutar `02_carga_y_limpieza.ipynb` (genera `Data/processed/panel_negocio_dia.csv`).
2. `python experimentos.py` (las 140 combinaciones; ~1 h con 10 núcleos) y `python estudio_optimizadores.py`.
3. Ejecutar los notebooks en orden y construir con `jupyter-book build book` (desde la raíz).

Semilla global fija: `SEED = 42` (`comun.py`).

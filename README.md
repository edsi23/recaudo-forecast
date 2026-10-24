# Pronóstico de ventas diarias por negocio

Proyecto de Machine Learning (tres entregas) sobre la venta diaria de 452 negocios (estaciones de servicio, tiendas de conveniencia y kioscos) operados por Masser.

- **Libro (Jupyter Book):** https://edsi23.github.io/recaudo-forecast/
- **Tablero (Dash, Render):** _URL al desplegar_
- **Versiones entregadas:** tags [`entrega-1`](https://github.com/edsi23/recaudo-forecast/tree/entrega-1), [`entrega-2`](https://github.com/edsi23/recaudo-forecast/tree/entrega-2)

| Entrega | Contenido |
|---|---|
| 1 | ETL, EDA, auditoría de fuga, modelo base (cap. 1–8) |
| 2 | EDA multivariado, 140 modelos (regresión y clasificación) con validación anidada, optimización, comparación estadística, interpretabilidad (cap. 4b, 9–15) |
| 3 | Mejores combinaciones y artefactos, modelo original (**Árbol de Combinación en el Símplex**, ACS), revisión de literatura, validación estadística, tablero multipágina (cap. 16–23) |

**Resultado principal.** Random Forest es el mejor modelo (RMSE log 0,887 en la validación anidada). El ACS, implementado desde cero, queda en el grupo de los mejores pero **no lo supera de forma significativa** (condición R3 no cumplida; ver cap. 20 y 22).

## Estructura

```
app.py              # tablero multipágina (expone server para gunicorn)
pages/              # contexto.py, eda.py, modelos.py
src/                # original_model.py (ACS), data.py, plots.py
artifacts/          # 9 modelos .joblib + metadata.json
data/               # datos procesados del tablero (anonimizados, con autorización)
assets/             # CSS y logo
tests/              # pruebas unitarias del ACS (pytest)
book/               # Jupyter Book y scripts de los experimentos
requirements.txt    # dependencias del tablero (Render)
requirements-libro.txt  # entorno completo del libro
render.yaml         # despliegue en Render
```

## Uso

```bash
# tablero local
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:8050
gunicorn app:server                # como en producción

# pruebas del modelo original
pip install pytest && python -m pytest tests

# libro (entorno completo)
pip install -r requirements-libro.txt
jupyter-book build book
```

El orden completo para reproducir los experimentos está en el capítulo 23 del libro. Semilla global: `SEED = 42` (`book/comun.py`).

## Despliegue

Render → **New → Blueprint** → este repositorio. El `render.yaml` instala `requirements.txt` (Python 3.11.9) e inicia `gunicorn app:server`.

## Datos

Los recibos de origen no están en el repositorio. Los datos procesados del tablero (`data/`) y los modelos (`artifacts/`) se publican con autorización de la empresa y con los negocios anonimizados. Los conjuntos externos (M5, UCI Online Retail II, UCI ElectricityLoadDiagrams) se descargan con `book/datos_externos.py`.

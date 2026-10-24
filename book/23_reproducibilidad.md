# 🔁 23. Reproducibilidad y bibliografía

## 23.1 Entorno

| Uso | Archivo | Python |
|---|---|---|
| Libro y experimentos | `requirements-libro.txt` (entorno conda `ml_venv`) | 3.9.25 |
| Tablero (Render) | `requirements.txt` | 3.11.9 |

Las dos listas fijan las **mismas versiones** de numpy (1.26.4), pandas (2.1.0), scipy (1.11.2), scikit-learn (1.3.0), xgboost (2.1.4) y joblib (1.3.2), así que los artefactos entrenados con el libro se cargan igual en el tablero (prueba de ida y vuelta en los dos entornos, cap. {doc}`17_mejores_combinaciones`). En macOS, XGBoost requiere OpenMP: `conda install -c conda-forge llvm-openmp`.

**Semilla global:** `SEED = 42` (en `book/comun.py`), propagada a numpy, scikit-learn, XGBoost, SMOTE/ADASYN, Optuna (TPE), el algoritmo genético (DEAP), el muestreo de las explicaciones de Shapley y las particiones del bootstrap.

## 23.2 Orden de ejecución

Con la carpeta `Data/` al lado del repositorio (acceso con autorización de la empresa), desde `book/`:

| Paso | Comando | Produce |
|---|---|---|
| 1 | ejecutar `02_carga_y_limpieza.ipynb` | `Data/processed/panel_negocio_dia.csv` |
| 2 | `python experimentos.py` | las 140 combinaciones (≈ 50 min con 10 núcleos) |
| 3 | `python estudio_optimizadores.py` | estudio de convergencia de los optimizadores |
| 4 | `python artefactos.py` | 7 artefactos de la entrega 2 con su metadata |
| 5 | `python datos_externos.py` | paneles M5, Online Retail II y Electricity (descarga pública) |
| 6 | `python modelo_original.py` | validación anidada del ACS (≈ 3 min) |
| 7 | `python experimentos_externos.py` | 8 benchmarks + ACS en los 3 conjuntos externos y HistGradientBoosting en Masser |
| 8 | `python validacion_original.py` | pruebas de hipótesis del ACS |
| 9 | `python exportar_app.py` | artefacto del ACS y del HGB, datos del tablero (`data/`) |
| 10 | `python revision_literatura.py` | búsqueda de literatura (`resultados/literatura_*.csv`) |
| 11 | ejecutar los notebooks y `jupyter-book build book` (desde la raíz) | el libro |
| 12 | `python -m pytest tests` (desde la raíz) | 16 pruebas del modelo original |

Los cómputos costosos (pasos 2, 3, 6 y 7) guardan sus resultados en archivos; los notebooks los leen y reproducen las tablas y figuras.

## 23.3 Tabla maestra de experimentos

`book/resultados/maestra_final.csv` (145 filas): las 112 combinaciones de clasificación y las 28 de regresión de la entrega 2, las 4 de HistGradientBoosting y el modelo original. Cada fila tiene la tarea, el modelo, el balanceo, el optimizador, la semilla, el número de evaluaciones, la media y la desviación de cada métrica en el bucle externo, las métricas de test, los tiempos de entrenamiento e inferencia y los hiperparámetros finales.

## 23.4 Bibliografía

Las referencias de la tercera entrega están en `book/references.bib` (formato BibTeX). Las más usadas: combinación de pronósticos {cite:p}`bates1969,smith2009,wang2023`, pesos por régimen {cite:p}`elliott2005,tetereva2026`, stacking {cite:p}`wolpert1992,breiman1996,vanderlaan2007`, árboles de modelos {cite:p}`quinlan1992,zeileis2008,godahewa2023`, meta-aprendizaje {cite:p}`monteromanso2020,li2022`, optimización {cite:p}`beck2017,held1974,duchi2008`, comparación estadística {cite:p}`demsar2006,hansen2011`, interpretabilidad {cite:p}`strumbelj2014`, estado del arte {cite:p}`ke2017`, aplicaciones cercanas {cite:p}`pan2022,kubek2025,e2022,seyedan2022`, datos externos {cite:p}`makridakis2022,chen2019retail,trindade2015` y contexto {cite:p}`larepublica2024`.

```{bibliography}
```

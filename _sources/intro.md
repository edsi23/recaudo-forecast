# Pronóstico de ventas diarias por negocio

Proyecto de Machine Learning sobre la **venta diaria** de 452 negocios (estaciones de servicio, tiendas de conveniencia y kioscos de autoservicio o KCO) operados por Masser, a partir de su historia reciente por medio de pago.

| | |
|---|---|
| **Periodo** | 1 jul – 23 sep 2026 (85 días); train 1 jul – 2 sep, test 3 – 23 sep |
| **Regresión** | Venta total del día siguiente por negocio (7 modelos × 4 optimizadores = 28) |
| **Clasificación** | ¿Tendrá el negocio mañana una caída fuerte (venta < 50 % de la mediana de su semana anterior)? (7 modelos × 4 balanceos × 4 optimizadores = 112) |
| **Validación** | Cruzada anidada y temporal (4 folds externos × 3 internos) |

**Primera entrega (cap. 1–8):** datos, limpieza, partición, análisis exploratorio, auditoría de fuga, preprocesamiento y un modelo base (SVR lineal contra líneas base ingenuas).

**Tercera entrega (cap. 16–23):** corrección de la entrega 2, selección de las mejores combinaciones y artefactos, un **modelo original** (Árbol de Combinación en el Símplex) con revisión de literatura, demostraciones, pruebas unitarias y validación estadística, y un **tablero en Dash** desplegado en Render.

**Segunda entrega (cap. 4b y 9–15):** EDA multivariado (PCA, agrupación de negocios, atípicos multivariados y días sin venta), el pipeline combinatorio de 140 modelos con Grid, Random, Bayesiana (Optuna) y Genética (DEAP), optimización computacional, evaluación de regresión y clasificación con pruebas estadísticas jerárquicas, calibración e interpretabilidad con SHAP y LIME.

```{admonition} Tablero en línea
:class: tip
**URL pública (Render):** [https://recaudo-forecast.onrender.com](https://recaudo-forecast.onrender.com) · El primer acceso puede tardar 30–60 s (plan gratuito).
```

**Resumen ejecutivo de la tercera entrega.** El mejor modelo para pronosticar la venta del día siguiente es **Random Forest** (RMSE en log 0,887 en la validación anidada; 0,622 en test). Se propuso un modelo original, el **ACS**, que combina varios modelos con pesos que dependen del régimen del negocio. Es novedoso para este problema, se implementó desde cero y queda en el grupo de los mejores, pero **no supera a Random Forest de forma estadísticamente significativa** (condición R3 no cumplida, cap. 20).

| Capítulo de la guía | Capítulo de este libro |
|---|---|
| 0. Introducción | Esta página |
| 1. Trazabilidad de correcciones | {doc}`16_trazabilidad` |
| 2. Datos: ETL y EDA | Caps. 1–7 y {doc}`04b_eda_multivariado` |
| 3. Modelos benchmark | {doc}`17_mejores_combinaciones` (y caps. 9–14) |
| 4. Revisión de literatura y novedad | {doc}`18_literatura` |
| 5. Modelo original | {doc}`19_modelo_original` |
| 6. Validación estadística | {doc}`20_validacion_estadistica` |
| 7. Dashboard y despliegue | {doc}`21_dashboard` |
| 8. Conclusiones y limitaciones | {doc}`22_conclusiones_e3` |
| 9. Reproducibilidad y bibliografía | {doc}`23_reproducibilidad` |

```{note}
Los recibos de origen (datos crudos) no están en el repositorio; el acceso es bajo solicitud y con autorización de la empresa. Los **datos procesados del tablero** (`data/`, negocio × día con negocios anonimizados N001…N453) y los modelos entrenados (`artifacts/`) **sí se publican, con autorización de la empresa**. Los notebooks se ejecutaron localmente y se publican con sus salidas guardadas; la tabla maestra de los experimentos está en `book/resultados/maestra_final.csv`.
```

```{tableofcontents}
```

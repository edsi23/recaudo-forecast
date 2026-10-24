# Pronóstico de ventas diarias por negocio

Proyecto de Machine Learning sobre la **venta diaria** de 452 negocios (estaciones de servicio, tiendas de conveniencia y kioscos de autoservicio o KCO) operados por Masser, a partir de su historia reciente por medio de pago.

| | |
|---|---|
| **Periodo** | 1 jul – 23 sep 2026 (85 días); train 1 jul – 2 sep, test 3 – 23 sep |
| **Regresión** | Venta total del día siguiente por negocio (7 modelos × 4 optimizadores = 28) |
| **Clasificación** | ¿Tendrá el negocio mañana una caída fuerte (venta < 50 % de la mediana de su semana anterior)? (7 modelos × 4 balanceos × 4 optimizadores = 112) |
| **Validación** | Cruzada anidada y temporal (4 folds externos × 3 internos) |

**Primera entrega (cap. 1–8):** datos, limpieza, partición, análisis exploratorio, auditoría de fuga, preprocesamiento y un modelo base (SVR lineal contra líneas base ingenuas).

**Segunda entrega (cap. 4b y 9–15):** EDA multivariado (PCA, agrupación de negocios, atípicos multivariados y días sin venta), el pipeline combinatorio de 140 modelos con Grid, Random, Bayesiana (Optuna) y Genética (DEAP), optimización computacional, evaluación de regresión y clasificación con pruebas estadísticas jerárquicas, calibración e interpretabilidad con SHAP y LIME.

```{note}
Los datos son confidenciales y no están en el repositorio (acceso bajo solicitud y con autorización de la empresa). Los notebooks se ejecutaron localmente y se publican con sus salidas guardadas. La tabla maestra de los experimentos (`book/resultados/maestra.csv`) sí está publicada: solo tiene métricas, tiempos e hiperparámetros.
```

```{tableofcontents}
```

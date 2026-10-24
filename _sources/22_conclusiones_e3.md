# ✅ 22. Conclusiones de la tercera entrega

## 22.1 Hallazgos

| Pregunta | Respuesta | Dónde |
|---|---|---|
| ¿Cuál es el mejor modelo de regresión? | **Random Forest · Grid** (RMSE log 0,887 ± 0,127 en el bucle externo; 0,622 en test). Está en el Model Confidence Set junto con XGBoost y HistGradientBoosting | Caps. {doc}`17_mejores_combinaciones`, {doc}`20_validacion_estadistica` |
| ¿Aporta un método reciente del estado del arte? | **No.** HistGradientBoosting (estilo LightGBM) queda 3.º (0,905), detrás de Random Forest y XGBoost | Cap. {doc}`17_mejores_combinaciones` |
| ¿El modelo original (ACS) supera a los benchmarks? | **No.** RMSE log 0,898 ± 0,127 (3.º) y MAE log 0,288 ± 0,051 (1.º, pero sin significancia). Ninguna de las 11 pruebas pareadas rechaza después de Holm; el MCS lo mantiene junto con otros 10 modelos (RMSE) y 18 (MAE); es mejor que Random Forest en el 45 % de los negocios | Cap. {doc}`20_validacion_estadistica` |
| ¿Es útil de todos modos? | Es **competitivo y robusto**: mejor que las cuatro reglas estándar de combinación, deja de usar a un experto degradado (19.9), es interpretable por construcción y en test queda levemente mejor que Random Forest en las tres semillas | Cap. {doc}`19_modelo_original` |
| ¿Por qué no gana? | El error dominante (días sin venta súbitos) no tiene señal el día anterior; los expertos están muy correlacionados; con 28 días de validación una diferencia de 1 % no es detectable | Caps. {doc}`19_modelo_original`, {doc}`20_validacion_estadistica` |

## 22.2 Veredicto sobre R1, R2 y R3

| Condición | Veredicto | Evidencia |
|---|---|---|
| **R1. Novedad en la aplicación** | **Sí**, al mejor de nuestro conocimiento | Búsqueda en 4 bases con 10 cadenas: no se encontró un árbol de combinación de pronósticos aplicado a la venta diaria de estaciones de servicio y tiendas. La idea general de pesos por régimen definidos con árboles sí existe en otro problema (Tetereva y Kazak, 2026, riesgo de cola financiero) y se declara (cap. {doc}`18_literatura`) |
| **R2. Implementación algorítmica propia** | **Sí** | Clase `ArbolCombinacionSimplex` compatible con scikit-learn, programada con numpy: proyección exacta al símplex, gradiente proyectado sobre la matriz de errores, búsqueda de cortes con sumas acumuladas, encogimiento jerárquico; 16 pruebas unitarias (cap. {doc}`19_modelo_original`) |
| **R3. Superioridad estadísticamente verificada** | **No** | No mejora la media del bucle externo del mejor benchmark y ninguna prueba rechaza (cap. {doc}`20_validacion_estadistica`) |

```{important}
**Tope de la nota según la guía: 4,0** (R1 sí, R2 sí, R3 no). El resultado se reporta tal como salió: la métrica principal se fijó antes de diseñar el modelo, y en ningún momento se cambiaron particiones, semillas ni métricas para buscar significancia.
```

## 22.3 Lecciones del desarrollo del modelo original

1. **Un error numérico puede parecer un buen resultado.** La primera versión del ACS (momentos sin centrar) no convergía, y sus pesos se quedaban cerca del promedio simple; eso le daba una ventaja aparente y significativa en MAE. Al corregir el solver la ventaja desapareció. La prueba unitaria de las condiciones de optimalidad (KKT), agregada después, habría detectado el problema desde el principio (cap. 19.11).
2. **El puzzle de la combinación también aparece aquí.** Con el óptimo exacto, el árbol tiende a sobreajustar los pronósticos fuera de muestra; los pesos globales en el símplex y el encogimiento hacia el promedio simple son difíciles de mejorar.
3. **Iterar el diseño viendo la validación externa contamina la estimación.** Por eso se reporta la historia de las cuatro versiones y se usan el test y los conjuntos externos como comprobación (cap. 19.11).

## 22.4 Limitaciones

- Un trimestre de datos y 28 días de validación externa: poca potencia para diferencias pequeñas.
- La revisión de literatura no incluye Scopus ni Web of Science (sin acceso) y Semantic Scholar respondió de forma parcial.
- Los conjuntos externos son 3 (más Masser): Friedman con 4 conjuntos no puede separar modelos (cap. 20.5).
- El ACS depende de la calidad y diversidad de sus expertos; con expertos muy correlacionados la ganancia posible es pequeña.

## 22.5 Trabajo futuro

1. **Expertos más diversos** (por ejemplo, el SVR lineal, que ganó en dos conjuntos externos), decididos antes de evaluar.
2. **Información del mismo día** (ventas por hora, alarmas del POS) para anticipar los días sin venta, que dominan el error.
3. **Pérdida robusta en las hojas** (Huber) para que los días anómalos no dominen los pesos.
4. **Más historia** (un año) y más conjuntos de datos para pruebas entre conjuntos con potencia.
5. **Intervalos de predicción por régimen**: usar las hojas del ACS para dar intervalos más anchos en los regímenes inciertos.

## 22.6 Declaración de autoría y de uso de herramientas de inteligencia artificial

```{admonition} Declaración
:class: important

**Autor:** Edgar De Jesús Siado Ferrer — usuario de GitHub `edsi23`.

**Uso de herramientas de IA.** Durante las tres entregas usé un asistente de IA generativa (Claude, de Anthropic, a través de Claude Code) de dos formas:

- **Como guía:** para discutir el diseño del proyecto y del modelo original, explicarme conceptos, el código y las pruebas estadísticas, y revisar la redacción del libro.
- **Como apoyo principal en la programación:** el asistente escribió gran parte del código (módulos de los experimentos, el modelo original y sus pruebas unitarias, y el tablero), que yo revisé, ejecuté y validé. También ejecutó por API la búsqueda de literatura y propuso borradores de las interpretaciones de tablas y figuras.

**Responsabilidad del autor.** Definí el problema, aporté y gestioné la autorización de los datos, tomé las decisiones del proyecto (tarea, métricas, criterio de veredicto y reportar la condición R3 como no cumplida), revisé los resultados y soy responsable de comprender y poder defender cada línea entregada, como exige la guía. Todas las cifras del libro se generan con el código incluido.

Firma: ______________________   Fecha: 25 de octubre de 2026
```

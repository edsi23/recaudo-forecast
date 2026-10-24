# ✅ 15. Conclusiones

## Respuestas a las preguntas del proyecto

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿Qué modelo pronostica mejor la venta del día siguiente? | **Random Forest**, con XGBoost estadísticamente indistinguible | Menor RMSE log en el bucle externo (0,887 ± 0,127) y en test (0,62). El MCS al 90 % deja solo Random Forest y XGBoost. Giacomini-White y Diebold-Mariano no los separan (cap. 12) |
| ¿Cuánto mejora frente a la primera entrega? | **10 % en RMSE log y 30 % en MAE en COP** frente al SVR lineal | Test: RMSE log 0,62 contra 0,69; MAE 1,85 M contra 2,64 M COP. Contra el ingenuo estacional, ahora también gana en pesos (WAPE 7,8 % contra 8,9 %), que era donde la entrega 1 empataba |
| ¿Es una mejora para todos los negocios? | **No: se concentra en los negocios difíciles** | Por rangos (Friedman–Nemenyi por negocio), Random Forest, Lasso y Ridge empatan. Delta de Cliff = 0 contra los modelos lineales. La ventaja media viene de los negocios irregulares |
| ¿Se pueden anticipar las caídas fuertes? | **En parte.** AUC ≈ 0,88 y F1 ≈ 0,30 (XGBoost o Random Forest balanceados) | Detecta las caídas "anunciadas" (rachas sin venta, semanas volátiles, festivos) y no las súbitas en negocios estables (cap. 13–14) |
| ¿Sirve el balanceo de clases? | **Sí para el F1, no para la discriminación** | Las 3 técnicas suben el F1 en 24 de 28 casos (+0,05), subiendo el recall (+0,45) a costa de la precision (−0,31). El AUC casi no cambia. `balanced`, SMOTE y ADASYN no se distinguen entre sí y las tres superan a no balancear (Friedman p = 10⁻⁶, Nemenyi) |
| ¿Están bien calibradas las probabilidades? | **No, por el balanceo**; Platt o isotónica lo corrigen | ECE de 0,04–0,26 a ~0,01. Después de calibrar hay que elegir el umbral según el costo de las alertas |
| ¿Qué optimizador conviene? | **Bayesiano (TPE)** | Con presupuesto igual, todos llegan a una calidad final parecida (Friedman n.s.), pero el TPE converge antes (menor área bajo el *regret*; llega al mejor en el 61–74 % de los problemas contra 29–57 %) y es el más estable entre semillas. El Grid es el peor en espacios de 4 o más dimensiones; el genético converge prematuramente con poblaciones chicas |
| ¿Funcionó el modelo en dos partes? | **No mejora** (RMSE log 0,62 en los dos; DM p = 0,84) | Los días sin venta que se pueden anticipar ya los predice bajos el bosque; los que no se anticipan no se arreglan cambiando la forma del modelo. Es un límite de **información** |

## Respuesta a la retroalimentación de la primera entrega

| Comentario | Qué se hizo |
|---|---|
| Falta un análisis multivariado más completo (PCA, atípicos multivariados, clustering) | Capítulo 4b: PCA de perfiles de negocio (5 componentes para el 80 %), k-means con 5 grupos interpretables (KCO, tiendas, EDS, negocios con dos fuentes, casi inactivos), atípicos con Isolation Forest y Mahalanobis robusta (y por qué el corte χ² no sirve con estos datos) |
| Tratar los días sin venta, que concentran casi todo el error | Capítulo 4b (quién, cuándo, rachas, festivos), la tarea de clasificación de caídas (cap. 13) y el modelo en dos partes (cap. 12.7) |
| Probar un modelo en dos partes o *gradient boosting* | Los dos. XGBoost y Random Forest son los mejores en ambas tareas; el modelo en dos partes no mejora, y se explica por qué |

## Lecciones metodológicas

1. **El modelo importa más que el optimizador.** Entre familias de modelos las diferencias son claras y significativas; entre optimizadores del mismo modelo, no. Con una superficie de pérdida plana, conviene más invertir el cómputo en mejores modelos y variables que en búsquedas más largas.
2. **La validación anidada estimó bien.** El orden de los modelos en el bucle externo coincide con el de test en las dos tareas. Las métricas del bucle externo son algo más pesimistas porque sus folds tienen festivos y menos días de entrenamiento.
3. **La pregunta estadística correcta cambia la respuesta.** Por media de error, Random Forest le gana a Ridge; por rangos por negocio, empatan. Las dos cosas son ciertas: el bosque gana donde es difícil. Reportar solo una de las dos sería engañoso.
4. **El error que queda es de información, no de modelo.** Los días sin venta súbitos y las caídas en negocios estables no tienen señal en el día anterior. Mejorar requeriría datos nuevos: ventas por hora del mismo día, alarmas del POS, inventario, calendario de mantenimientos.

## Límites

- **Un solo trimestre** (jul – sep 2026) y **3 festivos**, todos en train. No se puede decir nada de temporadas altas (diciembre), estacionalidad mensual ni del efecto festivo en test.
- La validación externa tiene **28 días** y el test **21**: las pruebas sobre series diarias (DM, GW, MCS) tienen poca potencia, y por eso varias diferencias quedan sin resolver.
- **Una sola operadora** (Masser) y negocios con al menos 14 días de historia: no se puede extrapolar a negocios nuevos (arranque en frío).
- El F1 de la alerta (~0,3) es útil como priorización (AUC 0,88), pero no como alarma automática sin revisión.

## Próximos pasos

1. **Umbral por costo** para la alerta, con probabilidades recalibradas (Platt) y un análisis de costo de falsas alarmas contra caídas no detectadas.
2. **Intervalos de predicción** (regresión cuantílica o *conformal prediction*), más anchos para los negocios irregulares que identificó la prueba de White.
3. **Más historia** para modelar festivos y estacionalidad mensual; variables de calendario comercial (quincenas, primas).
4. **Variables por tipo de negocio** (los grupos del cap. 4b) y un tratamiento de arranque en frío.
5. Para el genético, poblaciones más grandes y menos presión de selección; para la multi-fidelidad, usar una métrica continua en las rondas tempranas.

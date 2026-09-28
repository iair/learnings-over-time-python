# E5 · Scorecard clásico vs machine learning (y el feature engineering automático)

**Serie: Modelador de Riesgo en Profundidad** · Fase 2, extensión 5 de 8
Origen: reservas C1-29 ("¿por qué no XGBoost desde el día uno?") y C2-37 (featuretools/deep feature synthesis)

---

## 1. El marco honesto: es un trade-off, no una guerra

La pregunta bien planteada no es "¿qué modelo es mejor?" sino "¿cuánta discriminación adicional compra el ML, y cuánto cuesta en las dimensiones que este dominio valora?". Las dos posiciones caricaturescas — "la logística es del siglo pasado" y "el ML es una caja negra prohibida" — son ambas falsas. Los hechos razonablemente establecidos:

1. **La brecha de discriminación existe pero es modesta en admisión tabular:** con las mismas variables bien construidas, un gradient boosting (XGBoost/LightGBM) suele ganar 2-5 puntos de Gini sobre el scorecard WoE+logística en datos de admisión. La brecha crece con interacciones fuertes y datos ricos (comportamiento, transaccional fino) y se achica con pocas variables y pocos malos (donde el ML sobreajusta más fácil).
2. **La mayor parte de la brecha viene de interacciones y formas no monótonas** que el scorecard aplana a propósito (M7 §1). El costo de la explicabilidad estructural es esa señal residual.
3. **El pipeline de diseño es idéntico:** target, ventanas, anclas, muestras, fugas, calidad (M1-M6) no cambian ni un milímetro con el algoritmo. Un XGBoost con fuga de ventana es exactamente tan inútil como una logística con fuga de ventana — y más peligroso, porque exprime mejor la fuga. El 80% del valor del curso es algoritmo-agnóstico.

## 2. Qué pierde el ML en este dominio (la lista completa)

- **Reason codes:** la normativa y la práctica exigen razones de rechazo específicas y accionables. Del scorecard salen gratis (la tabla de puntos: "utilización sobre 56%: −18 puntos"). De un boosting salen vía SHAP — técnicamente sólido, pero (a) las atribuciones locales pueden ser inestables entre versiones del modelo, (b) explican el score, no la decisión con sus interacciones, y (c) su aceptación regulatoria varía por jurisdicción y todavía se litiga caso a caso.
- **Monotonicidad garantizada:** el comité puede exigir "más deuda nunca mejora el score". En el scorecard es una propiedad estructural del binning; en boosting se logra con restricciones de monotonía (soportadas por XGBoost/LightGBM y de uso obligado en crédito), al costo de parte de la ventaja de discriminación.
- **Estabilidad y mantenimiento:** el scorecard degrada suave y se recalibra por intercepto (E2); un boosting con cientos de árboles interactuando puede degradar de formas menos diagnosticables. El monitoreo de un scorecard es por variable (PSI por bin); el de un ML necesita herramientas extra (drift multivariado, monitoreo de SHAP).
- **Gobernanza:** validar internamente un scorecard es un procedimiento maduro con décadas de práctica; validar un ML exige capacidades que la segunda línea de muchos bancos aún está construyendo. El costo de gobernanza es real y se paga en tiempo-a-producción.

## 3. Qué pierde el scorecard (para ser simétricos)

- La señal de interacciones (el joven con deuda alta Y consultas recientes es peor que la suma de sus partes) salvo que se construyan a mano.
- Formas no monótonas legítimas que la disciplina del coarse classing simplifica.
- Velocidad de iteración cuando hay MUCHOS datos y variables: el flujo fine/coarse manual no escala a 5.000 features.

## 4. Las arquitecturas híbridas (la práctica madura 2020s)

1. **ML como challenger:** el scorecard es el campeón productivo; un boosting corre en sombra sobre las mismas matrices. Sirve de cota superior de discriminación ("¿cuánta señal estamos dejando?"), detector de interacciones (las SHAP interactions sugieren variables cruzadas para la próxima versión del scorecard) y alerta temprana de deterioro.
2. **ML para segmentar, scorecard para decidir:** árboles someros descubren la segmentación (¿thin file vs establecidos?) y cada segmento recibe su scorecard clásico.
3. **Boosting restringido como modelo productivo:** monotonía forzada + pocas variables auditadas + SHAP para reason codes + gobernanza reforzada. Es la ruta de fintechs y de bancos con validación madura; el Gini extra paga el costo de gobernanza cuando el volumen es grande.
4. **WoE + ML:** entrenar el boosting sobre las variables ya binneadas/WoE. Pierde parte de la gracia del ML (las formas finas) pero hereda robustez a outliers y missing, y acota el espacio de sobreajuste. Punto intermedio razonable y subvalorado.

**La respuesta del curso, contextualizada:** para un equipo formándose, un primer modelo, y un comité que debe confiar — scorecard primero. No es tecnofobia: es secuenciamiento. El ML llega mejor cuando el pipeline de diseño ya es sólido y la organización ya sabe validar.

## 5. Feature engineering automático (la reserva C2-37)

Herramientas tipo featuretools (deep feature synthesis) generan miles de variables cruzando tablas con agregadores automáticos. El diagnóstico del curso es exacto y conviene entender los tres mecanismos:

1. **Auditoría temporal imposible a escala:** cada feature generada necesita verificación de ancla (M2). Con 5.000 features, nadie la hace; el "cutoff time" de las herramientas ayuda pero no cubre lags por fuente ni semánticas ex-post (una columna `estado` cruzada automáticamente genera fugas industrializadas).
2. **Multiplicidad estadística:** 5.000 candidatas con 165 malos = la Trampa 1 y 2 de M7 a escala industrial: cientos de IVs fantasma "fuertes" por puro azar. El screening univariado deja de proteger (el mínimo de 5.000 ruidos es muy "significativo").
3. **Explicabilidad:** `MAX(MEAN(transactions.amount WHERE ...))` de tercera generación no es narrable a comité ni replicable con confianza en el motor de decisión.

**Dónde SÍ sirve:** como generador de **hipótesis** en la fase exploratoria (correr DFS, mirar qué familias de features sugiere, y reconstruir a mano las 5 prometedoras dentro de la fábrica declarativa de M5, con ancla verificada y nombre limpio). Herramienta de research, no de producción — la misma frontera que MICE en M6.

## 6. Para profundizar

- Lessmann et al. (2015), *Benchmarking state-of-the-art classification algorithms for credit scoring* — el benchmark académico de referencia sobre la brecha de discriminación.
- Lundberg & Lee (2017) para SHAP; y la literatura de *counterfactual explanations* para reason codes accionables.
- FICO y los papers de "interpretable ML in credit" (Rudin, *Stop explaining black box models...*, 2019) para la posición pro-modelos-inherentemente-interpretables.
- Documentación de restricciones de monotonía de XGBoost/LightGBM (práctica obligada si vas por la ruta 3).

## 7. Preguntas de autoevaluación

1. ¿Por qué un XGBoost "exprime mejor" una fuga que una logística? ¿Qué implica para el orden en que un equipo debería adoptar ML?
2. Defiende ante un CTO entusiasta la ruta challenger (arquitectura 1) contra el reemplazo directo del scorecard. ¿Qué evidencia comprometes a producir en 6 meses?
3. Diseña los reason codes de un boosting restringido: ¿SHAP crudo, SHAP agrupado por familia de variables, o contrafactuales? Ventajas y riesgos de cada uno ante un regulador.
4. ¿En qué caso concreto del pipeline del curso una interacción a mano (variable cruzada) capturaría la mayor parte de la ventaja del ML? Propón dos candidatas con su justificación de negocio.
5. Te pasan 4.800 features de featuretools con sus IVs. Describe el procedimiento completo para extraer valor sin tragarte las trampas (pista: HO intocado, corrección por multiplicidad, reconstrucción en la fábrica).

# E2 · Calibración: de score a PD, y la tendencia central

**Serie: Modelador de Riesgo en Profundidad** · Fase 2, extensión 2 de 8
Origen: teaser de C4; la respuesta del curso al "¿y el deterioro de 2025 no invalida entrenar con 2024?"

---

## 1. Discriminar no es lo mismo que acertar el nivel

Un modelo puede **ordenar** perfectamente (Gini alto: los malos concentrados en los scores bajos) y aun así **mentir en el nivel**: decir "PD 4%" donde la realidad de largo plazo es 7%. Son dos propiedades independientes:

- **Discriminación:** ¿separa buenos de malos? (AUC/Gini/KS — invariantes a transformaciones monótonas del score).
- **Calibración:** ¿la PD predicha coincide con la frecuencia observada? (curvas de calibración, test de Hosmer-Lemeshow y sucesores, binomial por tramo).

La decisión de negocio necesita **ambas**: el cutoff de M1 se define en términos de PD (PD* = Margen/(Margen+LGD×EAD)), así que una PD descalibrada corre el cutoff efectivo sin que nadie lo decida. Y provisiones/pricing consumen la PD directamente: ahí la calibración ES el producto.

## 2. Por qué el scorecard nace descalibrado

Tres fuentes, todas presentes en el proyecto del curso:

1. **La época de entrenamiento no es la época de uso.** El modelo aprende la tasa de malos de las cosechas 2024-07→2025-02; decidirá sobre 2026-2027. Si el ciclo se movió (el deterioro 2025 del caso), el intercepto del modelo lleva incrustada una tasa que ya no es. **Punto clave: el ordenamiento suele sobrevivir al ciclo mucho mejor que el nivel** — los mismos perfiles siguen siendo relativamente peores; lo que cambia es cuánto cae todo el mundo. Por eso la respuesta del curso ("el ordenamiento se mantiene; el nivel se corrige en calibración") es la doctrina estándar y no un parche.
2. **Manipulaciones de la muestra.** Excluir indeterminados cambia la tasa de la muestra respecto de la población; sobremuestrear malos (si se hiciera) también. El modelo aprende la tasa de SU muestra.
3. **El score no es probabilidad.** Tras la transformación a puntos (C3: PDO, offset), el output ni siquiera está en [0,1]: la relación puntos→PD hay que construirla.

## 3. La maquinaria de calibración

### 3.1 De score a PD observada

Se agrupa el score en tramos (típicamente deciles o bandas de puntos), y por tramo se mide la tasa de malos observada en una muestra honesta (HO/OOT). Esa tabla tramo→tasa es la calibración empírica cruda. Se suaviza ajustando una curva — el estándar es una **regresión logística del target sobre el score** (recalibración de intercepto y pendiente, "Platt scaling" en jerga ML): PD(s) = 1/(1+e^{−(a+b·s)}). Si b≈1 y solo se ajusta a, el ordenamiento no se toca: solo se corrige el nivel — la cirugía mínima.

### 3.2 La tendencia central (central tendency)

La pregunta: ¿a qué **nivel promedio** anclo la PD? Opciones:

- **Tasa del período de desarrollo:** incorrecta si el ciclo se movió (es justo lo que queremos corregir).
- **Tasa reciente (OOT/últimas cosechas):** mejor, pero es una foto de UN punto del ciclo — calibrar al peor momento sobreestima en la recuperación, y viceversa.
- **Tendencia central de largo plazo (long-run average):** el promedio de la tasa de malos a través de un ciclo completo (idealmente 5+ años, capturando al menos una recesión). Es el ancla estándar para modelos regulatorios (las PD "through-the-cycle" de Basilea, E6) y la práctica sana incluso en modelos gerenciales: la PD calibrada a tendencia central es estable y prudente.
- **PIT vs TTC:** la distinción de fondo. Una PD *point-in-time* refleja el momento del ciclo (la exige IFRS 9, con overlay de escenarios macro); una *through-the-cycle* refleja el promedio del ciclo (la prefiere la gestión de capital). Un mismo scorecard puede alimentar ambas con calibraciones distintas — y el modelador debe saber cuál está entregando.

**Mecánica del ajuste a tendencia central:** con la PD del modelo p̂ y la relación entre la tasa muestral π_muestra y la tendencia central π_LR, el ajuste estándar opera en log-odds: se desplaza el intercepto en ln[π_LR/(1−π_LR)] − ln[π_muestra/(1−π_muestra)]. Todos los scores se mueven en paralelo en log-odds: el ordenamiento queda intacto, el nivel queda anclado. (Es la misma álgebra del ajuste por sobremuestreo/prior correction.)

### 3.3 Cómo se verifica

- **Curva de calibración** (PD predicha vs observada por tramo) en OOT: los puntos sobre la diagonal.
- **Test binomial por tramo:** ¿la tasa observada es compatible con la PD predicha dado el n? (con pocos malos, los intervalos son anchos: E3).
- **Brier score** y su descomposición (calibración + resolución): la métrica escalar que castiga ambas cosas.
- **Monitoreo continuo:** la calibración es lo primero que se degrada con el ciclo (antes que el Gini). El semáforo de C5 debería tener una luz específica para "PD predicha vs observada por cosecha".

## 4. Errores clásicos

1. **Recalibrar tocando pendiente y bins sin gobernanza:** una recalibración de intercepto es mantenimiento menor; recalibrar pendiente o re-binnear es un rediseño con revalidación. Confundirlos es un hallazgo de auditoría.
2. **Calibrar en DEV:** la curva queda perfecta por construcción y no dice nada. Calibración se estima/verifica en muestras honestas.
3. **Ignorar la incertidumbre del ancla:** π_LR estimada con 3 años de historia sin recesión no es "largo plazo": es la parte buena del ciclo. Documentar el período usado y su representatividad.
4. **Un solo número para todo:** productos o segmentos con dinámicas distintas pueden necesitar tendencias centrales separadas.

## 5. Conexión con el resto de la serie

La calculadora de M1 asume que la PD que entra es la verdadera: esta extensión es lo que hace verdadera esa entrada. El deterioro 2025 de C1 se maneja con: OOT que lo contiene (diseño, M4) + calibración a tendencia central que lo pondera (aquí) + monitoreo que lo sigue (C5) + stress que lo exagera (E7).

## 6. Para profundizar

- Siddiqi, caps. de scaling y calibración (la mecánica PDO/offset y el ajuste de intercepto).
- Van der Burgt (2008) y Tasche, *The art of probability-of-default curve calibration* (2013) — el tratamiento técnico de referencia para calibrar curvas score→PD con pocas observaciones.
- Basel Committee, *Studies on the Validation of Internal Rating Systems* (WP14) — PIT vs TTC y validación de calibración en el marco regulatorio.

## 7. Preguntas de autoevaluación

1. Gini 62 con calibración pésima vs Gini 55 bien calibrado: ¿cuál prefieres para (a) fijar cutoff, (b) provisionar, (c) priorizar cobranza? ¿Cambia la respuesta?
2. Tu muestra de desarrollo tiene tasa 4,2% (indeterminados excluidos); la población con indeterminados prorrateados da 5,1%; la tendencia central a 7 años es 6,3%. Escribe el ajuste de intercepto en log-odds y qué le pasa al cutoff nominal.
3. ¿Por qué el ordenamiento sobrevive al ciclo mejor que el nivel? ¿Qué tendría que pasar en la economía para que TAMBIÉN se rompa el ordenamiento (pista: heterogeneidad de shocks por segmento)?
4. IFRS 9 te pide PD PIT con forward-looking y gestión de capital te pide TTC. ¿Puede el mismo scorecard alimentar ambas? ¿Qué cambia exactamente?

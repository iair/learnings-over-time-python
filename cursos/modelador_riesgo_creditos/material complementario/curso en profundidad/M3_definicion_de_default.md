# M3 · La definición de default: roll rates, cura, maduración e indeterminados

**Serie: Modelador de Riesgo en Profundidad** · Fase 1, módulo 3 de 8
Profundiza: C1 láminas 12-15, figuras de cura/maduración/cosechas, y tus apuntes de C2 (zona gris, eventos raros)
Material asociado: `M3_simulador_default.py` (notebook Marimo: cadena de Markov de mora + sensibilidad del umbral)

---

## 1. "Malo" es una definición operativa, no una opinión

La frase de apertura de la Parte 2 de C1 merece desempaque: en la mayoría de los problemas de clasificación el target viene dado (el cliente compró o no, el correo era spam o no). En riesgo de crédito **el target se diseña**, y ese diseño tiene cuatro perillas:

1. **El evento:** ¿mora 30, 60, 90? ¿castigo contable? ¿renegociación? ¿quiebra?
2. **La ventana:** ¿en cuántos meses desde t₀ debe ocurrir?
3. **La zona gris:** ¿qué hago con los que no son claramente buenos ni claramente malos?
4. **Las exclusiones:** ¿qué población queda fuera de la definición y por qué?

Cada combinación de perillas define un target distinto, con distinta tasa de malos, distinta señal disponible y distinto significado de negocio. El estándar de industria (90+ DPD en 12 meses, indeterminados 30-89 fuera) no es arbitrario, pero **tampoco es ley**: es una convención que hay que sustentar con la evidencia de la cartera propia. Las dos herramientas para sustentarla son los roll rates y las curvas de maduración. Este módulo desarrolla ambas más allá del uso del curso, incluyendo su formalización como cadenas de Markov y el análisis de sensibilidad del umbral.

---

## 2. Roll rates: la matriz de transición de la mora

### 2.1 Construcción

Se discretizan los DPD en bandas (al día, 1-29, 30-59, 60-89, 90+) y se cuentan las transiciones mes a mes de cada cliente: de cada estado, ¿a qué estado llegó el mes siguiente? Normalizando por fila se obtiene la **matriz de roll rates**, una matriz de transición empírica. La construcción del curso (`shift(-1)` por cliente sobre el panel ordenado + `crosstab` normalizada) es la implementación canónica.

Con los números de Banco Austral, la columna que importa es la de **cura** (volver a "al día"):

| Desde | Cura al mes siguiente | Lectura |
|---|---|---|
| 1-29 | ~34% | Mora blanda: un tercio se arregla solo. Etiquetarlos como malos sería castigar ruido. |
| 30-59 | ~11% | La cura ya se desplomó a un tercio de la anterior. |
| 60-89 | ~4% | Casi nadie vuelve; ~60% rueda directo a 90+. |
| 90+ | ~2% | Cura residual: **punto de no retorno**. |

El argumento del umbral 90+ es ese quiebre: si desde 90+ prácticamente nadie se recupera, entonces 90+ es un estado **absorbente en la práctica** y funciona como definición estable de "malo". Un umbral más blando (30+) etiquetaría como malos a muchos que se curan solos (34%): el modelo aprendería a castigar un comportamiento que se autocorrige, y la tasa de malos artificialmente alta distorsionaría la calibración. Un umbral más duro (castigo contable, típicamente 180+) haría el evento tan tardío y escaso que la señal disponible se encoge y el modelo "llega tarde": para cuando el castigo ocurre, el deterioro llevaba un año siendo visible.

### 2.2 Formalización: cadena de Markov

Si llamamos P a la matriz de transición entre bandas, el modelo implícito es una cadena de Markov de primer orden: el estado del mes t+1 depende solo del estado en t. Es una simplificación (la mora real tiene memoria más larga: quien ya estuvo dos veces en 30-59 transita distinto que quien llega por primera vez), pero permite tres análisis potentes que el notebook implementa:

1. **Proyección:** el vector de distribución de la cartera evoluciona como πₜ₊₁ = πₜ·P. Con la P empírica puedes proyectar cuántos clientes estarán en cada banda en 6 meses — la base de los modelos de provisión por matrices de transición.
2. **Absorción:** tratando 90+ como estado absorbente, la teoría de cadenas absorbentes entrega la probabilidad de que un cliente que hoy está en la banda b termine absorbido en 90+ dentro de h meses: exactamente la "PD por banda" que sustenta cuantitativamente el umbral. La fórmula: con Q la submatriz de transiciones entre estados no absorbentes y R la columna hacia 90+, la probabilidad de absorción en h pasos es la entrada correspondiente de (I + Q + Q² + … + Qʰ⁻¹)·R.
3. **Sensibilidad del umbral:** repetir el análisis moviendo el umbral (¿y si "malo" fuera 60+?) muestra cómo cambian tasa de malos, estabilidad y adelanto de la señal. La pregunta del gerente comercial (*"bajemos el umbral a 60 para vender más"*) se responde con esta tabla — y con la aclaración conceptual de M1: cambiar la **definición** del target no aprueba más gente; eso lo hace el **cutoff**.

### 2.3 Limitaciones y variantes que conviene conocer

- **No-markovianidad:** la práctica avanzada agrega memoria (estado = banda actual × peor banda histórica) o usa "ever-delinquent" flags. Para sustentar un umbral, la versión de primer orden basta.
- **Roll rates por cosecha o por segmento:** una matriz global puede esconder heterogeneidad (los clientes nuevos ruedan distinto que los antiguos). Si el punto de no retorno cambia por segmento, es un argumento para segmentar el scorecard.
- **Unidad cliente vs crédito:** el curso mide mora a nivel cliente (la peor de sus obligaciones). La alternativa a nivel crédito cambia los números; la definición debe declarar la unidad.
- **Renegociaciones:** un cliente que renegocia y "queda al día" resetea su mora administrativamente sin resolver su riesgo. Tratarlo como bueno crea el incentivo perverso de renegociar para maquillar cartera; el estándar conservador cuenta la renegociación forzada como evento de default (así lo hace Basilea vía *distressed restructuring*). Es la pregunta de reserva de C1 y una de las decisiones que más discusión genera en comités reales.

---

## 3. Curvas de maduración: la evidencia de la ventana

### 3.1 Qué muestran

Para cada cosecha, el % acumulado de créditos que ha tocado 90+ a los k meses del cursado. La forma típica es una S suave: pocos defaults los primeros meses (el crédito nuevo casi siempre paga las primeras cuotas), aceleración entre los meses 4 y 12, y desaceleración posterior.

### 3.2 El trade-off de la ventana, con números

La ventana de desempeño ideal capturaría el 100% del default de la vida del crédito, pero cada mes adicional de ventana **envejece un mes la última cosecha utilizable**. Con datos hasta 2026-06:

| Ventana | Última cosecha con target | Default capturado (vs visible a 18m, en Austral) |
|---|---|---|
| 12 meses | 2025-06 | ~60% |
| 18 meses | 2024-12 | 100% (por definición del benchmark) |
| 24 meses | 2024-06 | >100% del visible a 18 |

Usar 18 meses captura más evento pero deja el desarrollo anclado en cosechas del "mundo pre-deterioro" (2024), justo cuando la cartera 2025 cambió. La ventana de 12 es el compromiso: suficiente evento (~60% del visible), cosechas suficientemente recientes. **La honestidad del curso importa:** con el ciclo moviéndose, la curva NO se aplana del todo a los 12 meses — la ventana es un trade-off documentado, no un dogma. Un validador va a preguntar exactamente esto, y la respuesta correcta es la curva de maduración de la propia cartera más el argumento de recencia.

### 3.3 Maduración y comparabilidad

Corolario práctico que responde otra pregunta de reserva de C1: la tasa de malos de una cosecha **a medio madurar** (2025-05 con 8 meses observados) no es comparable con una madura (2024-08 con 12+). Las comparaciones legítimas son (a) a igual mes de maduración (tasa a 8 meses vs tasa a 8 meses — leer las curvas verticalmente), o (b) proyectando la cola faltante con la forma de las curvas maduras (los "loss forecasting triangles" de la práctica de provisiones). Comparar tasas crudas de cosechas con distinta maduración es el error de lectura más común en reportes de gestión.

---

## 4. La zona gris: indeterminados

### 4.1 Por qué existen y por qué se excluyen

Un caso cuya peor mora en la ventana quedó entre 30 y 89 días no cumple la definición de bueno (tuvo mora relevante) ni la de malo (no llegó a 90+). Tus apuntes capturan el punto fino que mucha gente confunde: **la exclusión no es para balancear la muestra** — es para no contaminar las clases:

- Etiquetarlos como buenos ensucia la clase buena con gente de riesgo intermedio-alto → el contraste bueno/malo se atenúa → menos señal.
- Etiquetarlos como malos cambia la definición de default de facto a 30+, con todas las consecuencias del umbral blando (§2.1).

La evidencia de que son una población genuinamente intermedia son los mismos roll rates: desde 30-59 cura el 11% (ni el 34% de la mora blanda ni el 2% del punto de no retorno).

### 4.2 Las reglas de manejo

1. **Se excluyen solo del entrenamiento.** En producción igual llegan solicitudes de este perfil y el modelo las scorea — el score de un indeterminado es válido; lo que no hay es etiqueta limpia para aprender de él.
2. **Se documenta cuántos son.** Un % de indeterminados alto (>10-15% es una señal de alerta habitual) indica que la definición parte la población en un lugar incómodo y amerita revisar umbral o ventana.
3. **Se reintroducen en el análisis de impacto.** Al evaluar la estrategia (C4), la tasa de malos "de verdad" de la cartera aprobada incluye a los indeterminados con algún tratamiento (peor caso, prorrateo, o seguimiento a ventana extendida).
4. **Si se quisieran incluir** habría que redefinir el target ex-ante y sustentarlo con evidencia y costos de negocio (tus apuntes citan la respuesta del profesor casi textual). Lo inadmisible es la decisión silenciosa.

---

## 5. Exclusiones y eventos extraordinarios

### 5.1 Exclusiones estándar

| Exclusión | Razón | Riesgo si no se hace |
|---|---|---|
| Fraude confirmado | La marca es ex-post (solo se sabe después); el fraude es otro fenómeno, con otros modelos | Mezcla dos procesos generadores; la marca como predictor sería fuga tipo 1 |
| Sin ventana completa | No hay target medible | Tratarlos como buenos sesga la tasa hacia abajo |
| Sin historia mínima (< 6 meses) | Las variables de ventana no existen o son ruido | Missing masivo no informativo; en la práctica estos clientes van a un scorecard/política aparte |
| Empleados, cuentas internas, montos de prueba | No son la población objetivo | Ruido y riesgo reputacional |

Regla transversal: **toda exclusión se cuantifica y documenta** (cuántos casos, % de la población, tasa de malos del grupo excluido). Las exclusiones son donde viven los sesgos del modelo — segunda pregunta de comité de C1.

### 5.2 Eventos extraordinarios (de tus apuntes)

Los períodos anómalos (retiros de AFP, pandemia, crisis) se analizan por separado para medir cuánto alteraron el comportamiento normal. Opciones, todas documentables:

- **Excluir el período del entrenamiento** si distorsiona la relación variable→default que regirá en condiciones normales (análogo a excluir crisis al modelar precios, como notaste — con la misma advertencia: la decisión depende del objetivo del modelo).
- **Mantenerlo con una marca de régimen** si se espera que el fenómeno se repita.
- **Conservarlo siempre para stress testing** (E7): el período de crisis excluido del desarrollo es exactamente el escenario adverso con el que después se estresa el modelo.

Lo que nunca: borrarlo en silencio. La diferencia entre limpieza y manipulación es la documentación y la medición del impacto de la decisión.

---

## 6. El contrato completo del target (plantilla)

Todo proyecto serio deja escrito su "contrato del target". El del curso, como plantilla reutilizable:

```
Unidad de análisis : solicitud cursada de crédito de consumo
Regla de identidad : cada cliente entra UNA vez (primera solicitud del período)
Evento (malo=1)    : peor mora ≥ 90 DPD dentro de (t₀, t₀+12]
Bueno (malo=0)     : peor mora ≤ 29 DPD en la misma ventana
Indeterminado      : peor mora en [30, 89] → fuera del entrenamiento, cuantificado
Sin target         : ventana incompleta (cosechas recientes) → muestra TTD
Exclusiones        : fraude confirmado, antigüedad < 6 meses, [otras documentadas]
Evidencia          : matriz de roll rates (cura 34/11/4/2%) + curvas de maduración
Limitaciones       : sesgo de selección por rechazados (reject inference no aplicada, E1);
                     deterioro 2025 → OOT + calibración a tendencia central (E2)
```

Poder escribir y defender cada línea de ese bloque es, en buena medida, la diferencia entre un data scientist que hace modelos de crédito y un modelador de riesgo.

---

## 7. El notebook: simulador de default

`M3_simulador_default.py` (Marimo) trae dos piezas interactivas sobre una cartera sintética:

1. **Cadena de Markov de mora:** matriz de transición empírica con la banda de cura destacada, proyección de la distribución de cartera a h meses, y probabilidad de absorción en 90+ por banda inicial (la "PD por banda" que sustenta el punto de no retorno).
2. **Sensibilidad del umbral:** un slider mueve el umbral de default (30/60/90/120) y otro la ventana (6-18 meses); la tabla muestra tasa de malos, % de indeterminados resultante y estabilidad entre cosechas para cada combinación. El gráfico de curvas de maduración marca la ventana elegida.

---

## 8. Preguntas de autoevaluación

1. Reconstruye el argumento del umbral 90+ usando solo la columna de cura de los roll rates. ¿En qué banda pondrías el umbral si la cura fuera 34/25/18/12%?
2. ¿Por qué excluir indeterminados NO es una técnica de balanceo de clases? ¿Qué pasaría con la calibración si se etiquetaran como malos?
3. Tu cartera muestra que la cosecha 2025-05 tiene tasa de malos 3.1% y la 2024-08 tiene 5.2%. ¿Puedes concluir que 2025 viene mejor? ¿Qué chequearías primero?
4. Un cliente renegocia en el mes 7 de su ventana y queda al día. Argumenta las dos posiciones (bueno vs malo) y los incentivos que crea cada una. ¿Cuál adopta Basilea y por qué?
5. Con la matriz Q y el vector R de tu cartera, ¿cómo calcularías la probabilidad de que un cliente hoy en 30-59 esté en 90+ dentro de 12 meses? ¿Para qué sirve ese número más allá de sustentar el umbral?

**Siguiente módulo:** M4 · Esquema muestral y las 4 fugas — el laboratorio donde cada fuga se fabrica a propósito y se mide su precio.

# M6 · Calidad de datos: el missing que informa, los outliers que mienten

**Serie: Modelador de Riesgo en Profundidad** · Fase 1, módulo 6 de 8
Profundiza: C2 láminas 15-18 (missing informativo, tres tipos de missing, outliers y sanity checks)
Material asociado: `M6_experimento_missing.py` (notebook Marimo: imputar vs bin propio, con el daño medido)

---

## 1. La pregunta previa a todo poder predictivo

El subtítulo de la Parte 3 de C2 es la pregunta correcta: *"¿estos números significan lo que creo que significan?"*. La calidad de datos en riesgo no es una etapa de limpieza previa al modelamiento: es una etapa de **interpretación**. La diferencia importa porque los dos errores simétricos son caros:

- Tratar como ruido lo que es señal (imputar un missing informativo, winsorizar un patrón real de clientes extremos).
- Tratar como señal lo que es ruido o error (dejar centinelas 999999 arrastrando un coeficiente, creer un IV inflado por un error de sistema).

El marco de este módulo: cada anomalía (missing, outlier, valor imposible) tiene un **mecanismo generador**, y el tratamiento correcto depende del mecanismo, no de la anomalía. Diagnosticar el mecanismo antes de tratar es lo que separa la limpieza profesional de la cosmética.

---

## 2. La taxonomía del missing (Rubin, aterrizada a crédito)

La clasificación clásica de Rubin (1976) distingue por la relación entre la probabilidad de faltar y los datos. La versión aterrizada del curso agrega una categoría que Rubin no necesitaba pero crédito sí:

### 2.1 Missing estructural: el dato no existe

**Mecanismo:** la definición misma de la variable hace imposible el dato para cierta subpoblación. Δ12m para clientes con 6-11 meses de historia (5,8% en Banco Austral: 903 de 15.665, exactamente los de historia corta); saldo de tarjeta para quien no tiene tarjeta; recencia de mora en quien no tiene historia observable.

**Diagnóstico:** el missing es 100% predecible desde otra columna (antigüedad, tenencia de producto). Si puedes escribir la regla que lo genera, es estructural.

**Tratamiento:** bin propio ("sin historia suficiente", "sin producto"). **Nunca imputar**: imputar un Δ12m para un cliente de 8 meses inventa una tendencia para alguien cuya característica real —ser nuevo, no tener el producto— es información en sí misma, con su propio perfil de riesgo. Además el bin estructural suele ser grande y estable, ideal para el binning.

### 2.2 MNAR (Missing Not At Random): la ausencia ES la señal

**Mecanismo:** la probabilidad de faltar depende del valor no observado o de características del cliente correlacionadas con el riesgo. El caso del curso: 36% de los independientes no declara renta contra 7% de los dependientes — no es azar, es informalidad, y la informalidad correlaciona con riesgo. El bin MISSING tiene WoE propio (−0,24: peores que el promedio).

**Diagnóstico:** cruzar el indicador de missing contra (a) el target — ¿la tasa de malos de los missing difiere de la de los no-missing? — y (b) otras características — ¿el missing se concentra en un segmento? Si cualquiera da sí, tratar como MNAR.

**Tratamiento:** bin propio con su WoE. La ausencia entra al modelo como categoría legítima con su propio peso. Imputar por la media hace doble daño: **borra la señal** (los missing-informativos quedan camuflados en el centro de la distribución) **y contamina la distribución** (un 13% de masa artificial en la media deforma los cuantiles y por tanto los bins de los no-missing).

### 2.3 MCAR (Missing Completely At Random): el único imputable

**Mecanismo:** la ausencia no depende de nada — fallo puntual de carga de un mes, un batch que no llegó.

**Diagnóstico:** el missing no correlaciona ni con el target ni con segmentos ni con el tiempo... salvo con el evento operacional que lo causó (todos los missing son de marzo 2025 = el batch caído). La trazabilidad operacional es el mejor diagnóstico.

**Tratamiento:** imputable **si es marginal**, documentando el criterio y midiendo el impacto (IV con y sin imputación). Para el flujo scorecard incluso el MCAR suele ir a bin propio o al bin vecino por simplicidad: el binning hace que la imputación fina rara vez pague.

### 2.4 La frase que ordena todo

*"El error caro no es imputar: es imputar sin preguntarse a qué tipo pertenece. La imputación por la media es la respuesta correcta a la pregunta menos frecuente."* En crédito de consumo, la mayoría del missing relevante es estructural o MNAR; el MCAR —el único caso donde la media es defendible— es la minoría. El reflejo `fillna(mean)` heredado de ML general es exactamente el hábito a desaprender.

### 2.5 Nota para tu caja de herramientas de DS

Las técnicas sofisticadas de imputación (MICE, KNN, imputación por modelo) están diseñadas para MAR (missing at random condicional a lo observado) y para **estimación de parámetros poblacionales**, no para scoring individual en producción. En un scorecard además chocan con dos restricciones: reproducibilidad exacta en el motor de decisión (¿vas a correr MICE en línea?) y explicabilidad ante comité. El flujo binning + bin MISSING resuelve el 95% de los casos con costo cero de gobernanza. Guarda MICE para los informes de research, no para el modelo productivo.

---

## 3. Outliers: error, cliente real extremo, o centinela disfrazado

El triaje del curso es la taxonomía correcta; lo que agrega este módulo es el procedimiento de diagnóstico para cada rama:

### 3.1 Centinelas

Valores especiales usados como "sin dato" por algún sistema upstream: 999999, −1, 0 (el más traicionero porque 0 suele ser también un valor legítimo).

**Detección:** mirar la **masa puntual**, no el promedio: el % de observaciones en el valor exacto más frecuente. Un 8% de la cartera con deuda exactamente 999999 no es una coincidencia; un histograma con un pico aislado en −1 tampoco. Complemento: los centinelas suelen documentarse en los diccionarios de los sistemas fuente — preguntar antes de deducir.

**Tratamiento:** convertir a missing **y luego** diagnosticar el missing resultante (usualmente es MNAR: el sistema que escribe 999999 lo hace por una razón correlacionada con el cliente).

### 3.2 Imposibles de negocio

Valores que violan la lógica del dominio. La distinción fina del curso: utilización 1,2 puede ser **sobregiro pactado, legítimo**; utilización 47 es un error (probablemente un saldo en pesos dividido por un cupo en UF, o un cupo desactualizado). La frontera entre extremo-legítimo y error no es estadística, es de negocio: hay que saber qué contratos permiten qué.

**Tratamiento:** los errores confirmados se corrigen en la fuente o se anulan a missing (documentado); los extremos legítimos **se quedan** — son exactamente los clientes que más informan.

### 3.3 Clientes reales extremos

La cola pesada genuina: el cliente con 40 consultas al bureau, la deuda 20× la mediana. Son señal, no ruido — y suelen ser señal de riesgo alto.

**Tratamiento en el flujo scorecard:** la buena noticia de la lámina 18 — **el binning acota el daño solo**: el extremo cae en el bin de los extremos y no arrastra ningún coeficiente (a diferencia de una regresión sobre el valor crudo, donde un punto de palanca puede torcer la pendiente). Por eso en el flujo WoE la winsorización es menos necesaria de lo que el hábito sugiere.

**Winsorizar** (acotar a p1/p99) sigue siendo defendible cuando: se usan valores crudos en algún análisis, se quiere estabilizar promedios de reportes, o los extremos son errores no confirmables. **Borrar filas casi nunca lo es**: elimina clientes reales de la población, sesga la tasa de malos y no es replicable en producción (¿vas a rechazar-por-outlier en línea?).

### 3.4 El chequeo mínimo por variable (el estándar del curso, sistematizado)

Para cada variable de la matriz, siempre: `n, % missing, mínimo, p1, mediana, p99, máximo, % en el valor más frecuente`. Lecturas rápidas: mínimo/máximo fuera del rango de negocio → imposibles; p99 << máximo → cola o error puntual; masa puntual alta en un valor raro → centinela; % missing que coincide con un segmento → estructural o MNAR. Este reporte se **auto-genera en la fábrica** (M5) para las 108 candidatas en cada corrida: la calidad de datos es un test de pipeline, no un ritual manual de una vez.

---

## 4. Decisiones documentadas: la plantilla

Cada decisión de calidad queda escrita con este esqueleto (es lo que la pauta del Lab pide como "reporte de calidad con decisiones tomadas"):

```
Variable        : renta_declarada
Anomalía        : 13% missing global; 36% en independientes vs 7% dependientes
Diagnóstico     : MNAR (informalidad); evidencia: tasa de malos de missing 1.4× la global
Decisión        : bin MISSING propio; NO imputar; ratios usan abonos observados (M5 §5.1)
Impacto medido  : IV con bin propio 0.18 vs 0.11 imputando por la media
Dueño / fecha   : [modelador], discutido con área comercial, [fecha]
```

La última línea no es burocracia: es la diferencia entre una decisión y una opinión. El validador va a pedir exactamente este registro.

---

## 5. El notebook: el experimento del daño medido

`M6_experimento_missing.py` (Marimo) genera una variable de renta con missing MNAR realista (concentrado en un segmento de mayor riesgo) y compara cuatro tratamientos midiendo IV y forma de la distribución:

| Tratamiento | Qué le pasa a la señal |
|---|---|
| Bin MISSING propio | IV completo: la ausencia aporta su propio WoE |
| Imputación por la media | La señal del missing se borra; la distribución se deforma (pico artificial) |
| Imputación por la mediana | Ídem, con pico en otro lado |
| Imputación "inteligente" por segmento | Recupera algo, pero sigue perdiendo el WoE propio y agrega complejidad de producción |

Un slider controla la intensidad del MNAR (cuánto más riesgosos son los que no declaran) para ver cómo crece el costo de imputar a medida que el missing se vuelve más informativo. Segunda pieza: un detector de centinelas por masa puntual corriendo sobre variables con 999999 y −1 plantados.

---

## 6. Preguntas de autoevaluación

1. Clasifica: (a) fecha de último uso de tarjeta en clientes sin tarjeta; (b) renta no declarada; (c) el panel de febrero que no cargó para el 2% de los clientes; (d) deuda = 999999. Mecanismo, diagnóstico y tratamiento de cada uno.
2. ¿Por qué imputar por la media hace *doble* daño en MNAR? Explica los dos mecanismos por separado.
3. ¿Por qué "el binning acota el daño de los outliers solo"? ¿Qué pasa en cambio en una regresión sobre valores crudos con un punto de palanca?
4. Utilización 1,15 y utilización 47: mismo tratamiento sí o no, y por qué. ¿Qué información de negocio necesitas para decidir?
5. Tu detector de masa puntual marca 6% de las deudas en exactamente 0. ¿Centinela o valor legítimo? Diseña el diagnóstico.
6. ¿Por qué MICE es mala idea en un scorecard productivo aunque sea buena estadística? Da las dos razones (gobernanza y producción).

**Siguiente módulo:** M7 · Binning, WoE e IV — la derivación de las fórmulas, la mecánica fine→coarse y las tres trampas reproducidas desde cero.

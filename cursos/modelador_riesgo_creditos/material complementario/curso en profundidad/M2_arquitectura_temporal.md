# M2 · Arquitectura temporal: t₀, ventanas, anclas y sus variaciones

**Serie: Modelador de Riesgo en Profundidad** · Fase 1, módulo 2 de 8
Profundiza: C1 láminas 8-10, C2 láminas 5-6 y Trampa 3 (C2-27)
Material asociado: `M2_simulador_anclas.py` (notebook Marimo: mueve el ancla y mira el IV inflarse)

---

## 1. La única regla que no admite excepción

Todo el edificio del scorecard descansa en una frontera:

> **Toda variable se calcula SOLO con información ≤ t₀. Todo target se mide SOLO con información > t₀.**

Cruzarla en cualquier dirección es una fuga de información (M4 cataloga los cuatro tipos; este módulo se concentra en la mecánica temporal que los previene). La regla parece trivial hasta que uno intenta implementarla sobre datos reales, donde tres complicaciones la vuelven un problema de ingeniería: los datos tienen **lag de consolidación**, cada solicitud tiene **su propia línea de tiempo**, y las fuentes tienen **rezagos distintos entre sí**. Este módulo desarma las tres.

---

## 2. La anatomía: tres piezas y una frontera

```
        VENTANA DE OBSERVACIÓN          t₀           VENTANA DE DESEMPEÑO
  [ t₀−12  ...................  t₀−1 ]  ●  [ t₀+1  ......................  t₀+12 ]
        aquí viven las VARIABLES     solicitud        aquí vive el TARGET
        (pasado ya consolidado)     (mes en curso)    (prohibido para variables)
```

- **t₀, punto de observación:** el mes de la solicitud. Es el "presente" simulado del modelo.
- **Ventana de observación:** hacia atrás. De ella salen todas las variables (M5). Su largo (3/6/12m) es un eje de la fábrica, no una decisión única.
- **Ventana de desempeño:** hacia adelante. De ella sale el target (¿tocó 90+ en 12 meses?). Su largo se sustenta con curvas de maduración (M3).

### 2.1 Por qué la ventana de observación termina en t₀−1 y no en t₀

Porque **el mes de la solicitud está en curso**: su cierre (facturación, pagos, mora al último día) se consolida días o semanas después. Si una variable usa el cierre de t₀:

- En **desarrollo** funciona perfecto: los datos históricos ya tienen ese cierre.
- En **producción** el motor de decisión tendría que esperar semanas para calcular la variable — y el crédito se decide hoy. La variable simplemente no existe cuando se necesita.

El resultado es el peor tipo de bug: uno que no lanza ninguna excepción. El pipeline de producción o (a) usa un valor viejo silenciosamente, (b) imputa el missing, o (c) recibe un dato parcial del mes en curso. En los tres casos la variable de producción **no es la variable con la que se entrenó**, y el desempeño real cae respecto al validado sin que nada "falle".

**Regla operativa:** la ventana es [t₀−k, t₀−1]. Ni un mes más. La pregunta de auditoría del curso lo condensa: *"¿esta variable estaría disponible, con estos valores, el día de la decisión?"* Si duda, es no.

### 2.2 Anclas corridas: cuando t₀−1 tampoco existe

t₀−1 supone que el cierre del mes anterior está disponible al decidir. Es cierto para datos internos (core bancario cierra en días), pero no para todas las fuentes:

| Fuente | Lag típico | Ancla real |
|---|---|---|
| Core interno (saldos, mora, pagos) | días | t₀−1 |
| Bureau externo (archivo mensual) | 1-2 meses | t₀−2 o t₀−3 |
| Información tributaria / previsional | 1-12 meses | según fuente |
| Variables macro (desempleo, IMACEC) | 1-2 meses | t₀−2 |

Dos consecuencias de diseño:

1. **El ancla es por-fuente, no global.** Una variable de bureau con ventana "6 meses" es [t₀−7, t₀−2] si el bureau llega con un mes extra de rezago. El diccionario de variables (M5) debe registrar el ancla de cada una.
2. **El ancla de desarrollo debe replicar el lag de producción, no el de la base histórica.** Este es el error sutil: en la base histórica el archivo de bureau de enero "está" en enero (fue backfilled), pero en producción llegó en marzo. Si desarrollas con ancla t₀−1 sobre bureau, entrenas con información que producción no tendrá. Es una fuga de ventana (tipo 2) disfrazada de convención razonable.

### 2.3 Variaciones de industria en las ventanas

- **Ventana de observación:** 12m es el estándar de admisión de consumo. Productos de ciclo corto (microcrédito) usan 6m; hipotecario puede usar 24m. La restricción práctica: exigir k meses de historia excluye a los clientes con menos de k meses (missing estructural, M6) y reduce la población — el trade-off cobertura/riqueza se decide con datos.
- **Ventana de desempeño:** 12m en consumo, 18-24m en productos de maduración lenta (hipotecario, comercial). Más larga = más defaults capturados pero cosechas más viejas disponibles (M3 §5 desarrolla el trade-off con las curvas de maduración).
- **Ventana móvil vs fija:** el curso usa ventana fija (12m para todos = vara común). La alternativa "hasta donde haya datos" (ventana variable) contamina la comparación entre cosechas: una solicitud con 18 meses observados tiene más oportunidades de caer que una con 12. Si ves tasas de malos que crecen con la antigüedad de la cosecha, sospecha ventana variable.

---

## 3. La máquina del tiempo, formalizada

La confusión que todo el mundo trae (*"¿cómo voy a saber si paga a 12 meses si no tengo esa información?"*) se disuelve con un cambio de sistema de referencia:

- **Para entrenar** no nos paramos en el presente: nos paramos en un punto del **pasado** (digamos 2024-10). Desde ahí, "el futuro" (2024-11 → 2025-10) ya ocurrió y está escrito en el panel de comportamiento. Vemos el futuro entre comillas: todo es histórico.
- **En producción** el modelo scorea una solicitud de HOY: ahí el futuro sí es desconocido, el modelo lo predice, y el monitoreo (C5) verificará 12 meses después si tenía razón.

La formalización útil para alguien con tu formación: el dataset de entrenamiento es una colección de pares **(X(t₀), Y(t₀, t₀+12])** donde X es una función solo del pasado hasta t₀−1 e Y solo del futuro estricto. La condición de legalidad de una variable es medibilidad respecto a la información disponible en t₀ — la σ-álgebra de lo ya ocurrido y consolidado. Toda fuga es una violación de medibilidad: la variable "sabe" algo que en t₀ no era conocible.

Esta formalización no es pedantería: da un test mecánico. Para cada columna candidata, pregunta *¿de qué fechas depende su cálculo?* Si alguna fecha > t₀−lag(fuente), la variable es ilegal. El validador automático de la fábrica de M5 implementa exactamente este test.

---

## 4. Ventanas escalonadas: cada crédito con SU reloj

La segunda confusión clásica: la ventana NO es un año calendario. Cada solicitud arranca su propia ventana de desempeño en su propia fecha: al crédito de enero lo observo hasta el enero siguiente; al de junio, hasta el junio siguiente. Consecuencias:

1. **Vara común:** todas las cosechas quedan medidas con exactamente 12 meses de oportunidad de caer. La tasa de malos de 2024-07 es comparable con la de 2025-02.
2. **Frontera de datos:** con datos hasta 2026-06, la última cosecha con ventana completa es 2025-06. Todo lo posterior queda **sin target** — no "bueno por defecto", sin target. Tratarlas como buenas es uno de los errores que la pauta del curso caza: sesga la tasa de malos hacia abajo en las cosechas recientes exactamente donde el deterioro sería más informativo.
3. **La muestra TTD:** esas solicitudes recientes sin target sirven para otra cosa: son la foto de a quién se scorea HOY. Comparar su distribución de variables contra DEV (PSI) responde si la población cambió — sin necesitar target.
4. **Implementación:** el patrón del curso (pivotear el panel a matriz cliente × mes y indexar con posiciones enteras) es el correcto: `peor_dpd_12m = M[i, t0+1 : t0+13].max()`, con el guard `if t0+12 <= último_mes else NaN`. Los dos errores de índice clásicos: incluir `t0` en el slice (target contaminado con la mora del mes de la solicitud, que puede reflejar la deuda que motiva la solicitud) y olvidar el guard (ventanas incompletas tratadas como completas).

---

## 5. La Trampa 3, diseccionada: cuánto "paga" cruzar la línea

La lámina C2-27 muestra la autopsia con datos de Banco Austral. Vale la pena entender el **mecanismo** de cada barra, porque es el patrón que reconocerás en auditorías reales:

| Variable | IV | Qué está pasando |
|---|---|---|
| `dias_mora_max_12m` bien anclada [t₀−12, t₀−1] | 0.39 | Señal honesta: la mora pasada predice la futura, moderadamente. |
| Misma variable con ventana corrida a t₀+1 | 0.90 | Un error de índice de UN mes. La variable ahora incluye la primera cuota del crédito nuevo: quien parte atrasándose es casi seguro malo. El IV se duplica con información que en t₀ no existe. |
| Δ cupo t₀+3 vs t₀−1 | 3.17 | La variable mide la **reacción del banco** al deterioro temprano (recorte de cupo). Es un termómetro del desenlace, no un predictor. |
| Δ cupo t₀+12 vs t₀−1 | 11.13 | El desenlace casi puro: el banco recortó el cupo PORQUE el cliente cayó. En producción esa variable no existe: es el futuro. |

Tres lecciones generalizables:

1. **La relación fuga→IV no es lineal, es explosiva.** Un mes de contaminación duplica; doce meses multiplican por 28. Por eso el umbral "IV > 0.5 = auditar" es tan efectivo: las fugas graves se delatan solas.
2. **Las fugas más peligrosas son las de acción institucional.** Columnas que registran decisiones del banco (recortes de cupo, bloqueos, gestiones de cobranza, castigos) son desenlaces disfrazados: el banco actuó porque el cliente se deterioró. Cualquier variable derivada de acciones posteriores a t₀ es fuga aunque "parezca" comportamiento del cliente.
3. **El IV alto es el síntoma, la ventana es el diagnóstico.** Frente a un IV sospechoso, la primera auditoría es siempre: ¿cuál es la ventana exacta de cálculo de esta variable, mes por mes? El notebook de este módulo te deja reproducir la autopsia con datos sintéticos y verificar que entiendes el mecanismo, no solo el resultado.

---

## 6. El notebook: simulador de anclas

`M2_simulador_anclas.py` (Marimo) genera una cartera sintética con dinámica realista (mora persistente, deterioro previo al default, y recorte de cupo del banco como reacción) y expone dos sliders:

- **Desplazamiento del ancla** (−3 a +12 meses respecto de t₀−1): mueve el fin de la ventana de una variable de mora y recalcula su IV. Verás la meseta honesta en anclas ≤ t₀−1 y la explosión al cruzar t₀.
- **Variable de reacción del banco** (Δ cupo a k meses vista): reproduce las barras rojas de la Trampa 3.

El gráfico final —IV vs posición del ancla, con la zona legal sombreada— es la figura que deberías poder dibujar de memoria en una entrevista.

---

## 7. Checklist de auditoría temporal (para llevar)

Antes de dar por buena cualquier matriz de modelación:

1. ¿Cada variable declara su ventana exacta [inicio, fin] y su fuente? (El nombre debería decirlo: M5.)
2. ¿Algún fin de ventana > t₀−lag(fuente)? → ilegal.
3. ¿El slice del target es [t₀+1, t₀+12] con guard de ventana completa?
4. ¿Las solicitudes sin ventana completa quedaron sin target (NaN), no en 0?
5. ¿Hay columnas de estado "actual" o de acciones del banco en la matriz? → fuera (o solo para exclusiones).
6. ¿El ancla de desarrollo replica el lag de producción para cada fuente externa?
7. ¿Existe un test automático que verifique 1-6 en cada corrida del pipeline? (Si la respuesta es no, tu experiencia en ingeniería sabe exactamente qué hacer: M4 y M5 traen la test-suite.)

---

## 8. Preguntas de autoevaluación

1. ¿Por qué "la ventana termina en t₀−1" es una regla de producción y no de estadística? ¿Qué falla exactamente si se usa t₀?
2. Tu bureau llega con dos meses de rezago pero la base histórica lo tiene backfilled al mes de referencia. ¿Con qué ancla desarrollas y por qué? ¿Qué pasa si no lo haces?
3. Explica el mecanismo por el cual Δcupo t₀+12 alcanza IV 11: ¿qué está midiendo realmente esa variable?
4. Una solicitud de 2025-10 con datos hasta 2026-06: ¿qué target tiene y a qué muestra va?
5. Diseña el assert de pipeline que detectaría automáticamente la ventana corrida de la Trampa 3.

**Siguiente módulo:** M3 · La definición de default — roll rates como cadenas de Markov, curas, maduración y qué pasa al mover el umbral.

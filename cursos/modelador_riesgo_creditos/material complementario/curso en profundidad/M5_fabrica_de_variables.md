# M5 · La fábrica de variables como pipeline de ingeniería

**Serie: Modelador de Riesgo en Profundidad** · Fase 1, módulo 5 de 8
Profundiza: C2 láminas 7-13 (fábrica, convención de nombres, agregadores, detalles finos, ratios, denominador)
Material asociado: `M5_fabrica_declarativa.py` (notebook Marimo: la fábrica completa con validador de ancla y diccionario auto-generado) · `M5_catalogo_variables.csv` (catálogo de las 108 candidatas, importable a Google Sheets)

---

## 1. El error del principiante y la idea profesional

El principiante inventa variables de a una: "se me ocurrió que el uso de la línea podría predecir". El profesional define una **convención de tres ejes** y la deja producir:

```
FAMILIAS (qué mido)  ×  VENTANAS (desde cuándo)  ×  AGREGADORES (cómo resumo)
```

7 familias × 3 ventanas × ~5 agregadores aplicables ≈ 75 variables de ventana; más fotos al último cierre, ratios de negocio y demografía: **~108 candidatas**. Las tres ventajas del curso merecen una relectura de ingeniero:

1. **Cobertura:** el producto cartesiano no olvida combinaciones. Lo que el cerebro humano hace mal (enumerar exhaustivamente) lo hace la convención.
2. **Auditoría:** el revisor entiende 108 variables leyendo UNA convención. El costo de revisión escala con las reglas, no con las columnas.
3. **Reutilización:** el próximo modelo (otra cartera, otro producto) reusa la fábrica cambiando la lista de familias. La fábrica es un activo; las variables son su output.

Para alguien con tu perfil la traducción es directa: **la fábrica es un pipeline declarativo** — una especificación (datos) separada de un motor (código), con contratos verificables. Todo lo que sabes de pipelines aplica: la especificación se versiona, el motor se testea, los contratos se validan en cada corrida. Este módulo diseña esa arquitectura pieza por pieza.

---

## 2. La arquitectura: especificación + motor + contratos

### 2.1 La estructura de datos correcta: matrices cliente × mes

El paso 2 de la demo del curso ("pivotear el panel a matrices cliente × mes: la estructura que hace todo lo demás fácil") es una decisión de diseño con nombre: pasar de formato largo (una fila por cliente-mes) a **matrices densas indexadas por posición** (una matriz por métrica: mora, saldo, cupo, pagos...). Ventajas:

- Toda ventana se vuelve un slice: `M[i, t0-6 : t0]` — vectorizable, legible, y con el ancla **visible en el código**.
- El validador de anclas se reduce a inspeccionar los índices del slice.
- El costo (memoria O(clientes × meses) por métrica) es trivial a escala de cartera de consumo; a escala de decenas de millones de filas se reemplaza por agregaciones window de SQL/Polars con la misma semántica.

### 2.2 La especificación declarativa

La fábrica del curso es "una lista de tuplas → 75 variables". La versión de producción enriquece cada tupla hasta volverla un contrato completo:

```python
SPEC = [
    # (familia,       metrica_fuente, agregador,  ventana, lag_fuente)
    ("mora_propia",   "dias_mora",    "max",      12,      1),
    ("mora_propia",   "dias_mora",    "recencia", 12,      1),
    ("utilizacion",   "uso_linea",    "prom",     6,       1),
    ("utilizacion",   "uso_linea",    "max",      6,       1),
    ("deuda",         "deuda_total",  "delta",    6,       1),
    ("consultas",     "n_consultas",  "suma",     3,       2),   # bureau: lag 2
    ...
]
```

Cada entrada produce una variable con nombre `<concepto>_<agregador>_<ventana>m`, ventana [t₀−ventana−lag+1, t₀−lag], y metadatos que alimentan el diccionario. El motor es UNA función (`agregar_ventana()` en el curso) que recibe la matriz, el slice y el agregador. Especificación y motor evolucionan por separado: agregar una familia es una línea de datos; agregar un agregador es una función pura testeada.

### 2.3 Los contratos (lo que hace a la fábrica auditable)

1. **Contrato de ancla:** cada variable declara su ventana y lag; el validador (test T1 de M4) verifica `fin ≤ t₀ − lag` mecánicamente. Ninguna variable entra a la matriz sin pasar por él.
2. **Contrato de nombre:** el nombre se **genera** desde la especificación, nunca se escribe a mano. Así la lámina "el nombre ES la documentación" deja de ser una aspiración y pasa a ser un invariante: es imposible que el nombre mienta sobre la ventana porque ambos salen de la misma tupla.
3. **Contrato de diccionario:** el diccionario de variables (nombre, familia, fórmula, ventana exacta, fuente, lag, tratamiento de missing) se auto-genera en la misma corrida. Es el documento que ingeniería usará para replicar el cálculo en el motor de decisión — y donde mueren la mayoría de las discrepancias desarrollo/producción.

---

## 3. Los agregadores, uno por uno (y sus trampas)

Cada agregador responde una pregunta de negocio distinta; promedio y máximo sobre la misma serie **no** son redundantes (uso promedio 40% con máximo 95% cuenta otra historia que 40% con 45%). Detalles finos por agregador:

| Agregador | Pregunta | Trampas |
|---|---|---|
| **Promedio** | ¿Cómo se comporta habitualmente? | Ignora meses faltantes silenciosamente (¿promedio de 4 meses cuando pediste 12?). Decisión: exigir n mínimo de meses o reportar `n_meses_disponibles` como variable acompañante. |
| **Máximo / mínimo** | ¿Cuál fue su peor/mejor momento? | Sensible a errores puntuales de datos (un mes con dato basura fija el máximo). El reporte de calidad (M6) protege. |
| **Suma / conteo** | ¿Cuánta actividad acumuló? | No comparable entre clientes con distinta historia disponible: 3 consultas en 3 meses ≠ 3 consultas en 12. Normalizar o exigir ventana completa. |
| **Tendencia (Δ)** | ¿Mejora o empeora? | Necesita AMBOS extremos → missing estructural cuando falta t₀−k (el 5,8% de Banco Austral). Variante robusta: pendiente de regresión sobre la ventana (usa todos los meses) en vez de delta entre extremos. Δ absoluto vs relativo (%): el relativo explota con denominadores chicos — acotar. |
| **Recencia** | ¿Hace cuánto que no pasa? | La censura (§4). Además define "el evento" con cuidado: ¿mora > 0 o mora ≥ 30? Cada definición es otra variable. |
| **Volatilidad (extra de industria)** | ¿Qué tan errático es? | Desviación estándar del uso o de los pagos en la ventana. Potente en comportamiento; exige ≥ 4-6 meses. |
| **Racha (extra)** | ¿Cuántos meses seguidos...? | Meses consecutivos al día / en mora. Captura persistencia que el promedio diluye. |

### 3.1 Ventanas: cortas vs largas

- **3m:** captura el estado reciente; reactiva pero ruidosa; disponible para casi todos los clientes.
- **12m:** captura el patrón estructural; estable pero lenta para reflejar cambios; excluye o llena de missing a los clientes cortos.
- La **combinación** corta/larga es donde vive la señal de tendencia implícita: `uso_prom_3m` muy sobre `uso_prom_12m` = aceleración del uso, un predictor clásico de estrés. Muchos modeladores construyen explícitamente el ratio corto/largo como variable.

---

## 4. Los dos detalles finos, en profundidad

### 4.1 Recencia con censura: el valor 13

El problema: si "tuvo mora hace 12 meses" y "nunca tuvo mora en la ventana" comparten el valor 12, el modelo mezcla dos poblaciones opuestas — el que apenas sale del rango con el impecable. La convención del curso: **sin evento en la ventana → 13** (un valor fuera del rango 0-12 que el binning aislará en su propio bin).

Por qué funciona y cuándo no:

- Funciona porque el binning (M7) trata la variable como ordinal por tramos: el bin {13} queda separado con su propio WoE, que empíricamente será el más "bueno". La numeración es solo un truco de implementación para no crear una columna categórica aparte.
- El equivalente estadístico formal es el de datos censurados por la derecha (análisis de supervivencia): "recencia ≥ 12" es lo único que sabemos. La convención 13 es la versión pragmática; la versión sofisticada (raramente necesaria en scorecards) modelaría el tiempo-al-evento.
- **Trampa:** si alguien trata la variable como numérica continua en un modelo sin binning (un gradient boosting directo), el 13 introduce una discontinuidad arbitraria. La convención es segura DENTRO del flujo binning→WoE; el diccionario debe advertirlo.

### 4.2 Missing estructural del Δ12m

El delta de 12 meses necesita el extremo t₀−12. Un cliente con 8 meses de historia no lo tiene: el dato **no existe, no falta**. Resultado real en Banco Austral: 5,8% de missing en las Δ12m, exactamente los clientes con 6-11 meses de historia — no es un error de datos, es la consecuencia aritmética de exigir 6 meses de antigüedad y calcular ventanas de 12.

Tratamiento (M6 lo generaliza): bin propio "sin historia suficiente", nunca imputación — imputar inventaría una tendencia para alguien cuya característica real es *ser nuevo*, que es información en sí misma. Nota de diseño: este missing es **predecible desde la especificación** (toda variable con ventana > antigüedad mínima lo tendrá), así que la fábrica puede pre-declararlo en el diccionario en vez de "descubrirlo" en el reporte de calidad.

---

## 5. Ratios de negocio: donde entra el criterio

*"Un ratio es una hipótesis de negocio escrita en fórmula. Ningún algoritmo los inventa por usted."* Los seis del curso, con la hipótesis explícita y las decisiones de implementación que cada uno esconde:

| Ratio | Hipótesis | Decisiones escondidas |
|---|---|---|
| Utilización = saldo/cupo | La presión sobre el crédito disponible precede al default | ¿Saldo y cupo del mismo mes? ¿Qué pasa con cupo 0 (línea cerrada)? >1 es sobregiro pactado legítimo; 47 es error |
| Carga financiera = deuda total/ingreso | La capacidad de pago comprometida limita la resiliencia | ¿Qué ingreso? (§5.1). ¿Deuda solo propia o del sistema (bureau)? |
| Pago sobre facturación = pagos 6m/facturado 6m | Quien paga el mínimo revuelve; quien paga el total, no | Ventanas alineadas de numerador y denominador; facturación 0 |
| Deuda interna/sistema | Cuánto del riesgo del cliente es nuestro | Requiere bureau: hereda su lag (ancla t₀−2) |
| Cuota estimada/ingreso | El esfuerzo del crédito que pide AHORA | Única variable de la solicitud misma (monto/plazo): legal porque es información de t₀ aportada por el cliente al decidir |
| Ahorro/ingreso | El colchón ante imprevistos amortigua shocks | Saldo de ahorro es volátil: ¿foto t₀−1 o promedio 3m? |

**Regla universal de denominadores:** siempre acotar (`max(denominador, 1)` o el mínimo con sentido de negocio) y dejarlo explícito en el código. Un ratio con denominador cero no es un infinito: es una división que alguien olvidó pensar.

### 5.1 El debate del denominador: ¿qué ingreso?

La decisión del curso es un caso de estudio de criterio profesional. Opciones:

- **Renta declarada** (del formulario): foto del pasado, sin verificar, con 13% de missing — y missing MNAR (los independientes no declaran: M6).
- **Abonos observados** (promedio de abonos a cuenta en 6 meses): lo que el banco VE entrar mes a mes. Verificable, mensual, sin auto-reporte... pero sesgado para quien reparte sus abonos entre bancos, y solo existe para clientes con cuenta.

Decisión del curso: **los ratios usan abonos observados; la renta declarada entra como variable aparte.** La elegancia está en lo que logra simultáneamente: (a) si la renta falta, el ratio igual existe; (b) la ausencia de renta declarada se modela como información propia (bin MISSING con su WoE); (c) la discrepancia renta declarada vs abonos observados queda disponible como señal implícita (quien declara mucho y abona poco...). Y la meta-lección: *esta decisión se discute con el área comercial y queda escrita en el informe — es exactamente el tipo de supuesto que un validador va a buscar.* Las decisiones de denominador nunca son técnicas: son supuestos de negocio con dueño.

---

## 6. Fotos y demografía: lo que completa las 108

- **Fotos al último cierre (t₀−1):** saldo de ahorro, deuda total, n° de productos. Sin agregación; máxima recencia, mínima estabilidad. Complementan (no sustituyen) a los promedios.
- **Demografía y solicitud:** edad, antigüedad como cliente, región, tipo de empleo, monto/plazo solicitado. Estables y disponibles para todos (incluso thin-file). Advertencias: (a) variables protegidas o proxies (sexo, y según jurisdicción edad/región) exigen análisis de equidad antes de usarse — en el screening de Banco Austral, sexo y canal marcan IV 0.00 y salen solas, pero eso es suerte del dataset, no política; (b) el "ruido plantado" del curso (n_dependientes con IV 0.01) muestra el screening funcionando: es tan importante que el ruido salga como que la señal entre.

---

## 7. El notebook y el catálogo

`M5_fabrica_declarativa.py` (Marimo) implementa la arquitectura completa sobre datos sintéticos:

1. **Especificación** como lista de tuplas enriquecidas (familia, métrica, agregador, ventana, lag).
2. **Motor:** `agregar_ventana()` genérico + agregadores como funciones puras (incluye recencia con censura y delta con missing estructural).
3. **Validador de ancla:** rechaza en construcción una variable-trampa incluida a propósito (ventana que cruza t₀).
4. **Diccionario auto-generado** con la ventana exacta y el missing estructural pre-declarado.
5. **Slider de antigüedad mínima:** mueve el requisito (3-12 meses) y observa el trade-off población incluida vs % de missing estructural en las Δ12m — la decisión de diseño de C1 hecha tangible.

`M5_catalogo_variables.csv` es un catálogo de ~100 candidatas estilo Banco Austral (nombre, familia, agregador, ventana, ancla, fuente, lag, missing esperado), listo para importar a Google Sheets como base del diccionario de tu propio proyecto.

---

## 8. Preguntas de autoevaluación

1. ¿Por qué generar el nombre desde la especificación (y no escribirlo a mano) convierte "el nombre es la documentación" en un invariante verificable?
2. `uso_prom_3m / uso_prom_12m` > 1.4: ¿qué historia cuenta ese cliente? ¿Por qué esa señal no está en ninguno de los dos promedios por separado?
3. Diseña el tratamiento completo de `meses_desde_mora_12m` para un cliente (a) con mora hace 3 meses, (b) sin mora en 12 meses, (c) con 7 meses de historia. ¿Qué valor recibe cada uno y a qué bin irá?
4. Defiende la decisión "abonos observados en el denominador" ante un gerente comercial que insiste en la renta declarada "porque es la oficial". ¿Qué evidencia llevarías?
5. Tu fábrica corre en desarrollo con matrices numpy y en producción con SQL. ¿Qué contratos garantizan que ambas implementaciones calculan LA MISMA variable? ¿Cómo lo testearías?

**Siguiente módulo:** M6 · Calidad de datos — el missing que informa, los outliers que mienten, y el experimento que muestra cuánta señal destruye una imputación bienintencionada.

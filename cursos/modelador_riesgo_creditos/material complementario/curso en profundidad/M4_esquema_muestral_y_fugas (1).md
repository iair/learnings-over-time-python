# M4 · Esquema muestral y las cuatro fugas de información

**Serie: Modelador de Riesgo en Profundidad** · Fase 1, módulo 4 de 8
Profundiza: C1 láminas 17-19 y la autopsia de la demo (AUC 0.991 de `estado_actual`)
Material asociado: `M4_laboratorio_fugas.py` (notebook Marimo: cada fuga fabricada a propósito, con su precio medido, más una test-suite anti-fugas reutilizable)

---

## 1. Cuatro muestras, cuatro preguntas

El subtítulo de la Parte 3 de C1 es la mejor definición del tema: *"confundir las muestras es la forma más elegante de mentirse"*. Cada muestra existe para responder UNA pregunta, y la disciplina consiste en no dejar que ninguna responda la de otra:

| Muestra | Composición (curso) | Pregunta que responde | Qué la invalida |
|---|---|---|---|
| **DEV** | 70% de cosechas 2024-07→2025-02 | ¿Qué patrones hay? (binning, selección, coeficientes) | Nada: es el arenero |
| **HO** | 30% restante, mismas cosechas | ¿Me sobreajusté al ruido de DEV? | Que un cliente de DEV se cuele; que se use para iterar tanto que se vuelva DEV |
| **OOT** | Cosechas 2025-03→2025-06 | ¿Generaliza EN EL TIEMPO? | Tocarla más de una vez |
| **TTD** | Solicitudes recientes sin ventana completa | ¿A quién scoreo HOY? (PSI, sin target) | Intentar sacarle target |

### 1.1 Por qué HO y OOT son preguntas distintas

Es el punto que más se pierde viniendo de ML general. HO responde una pregunta de **varianza estadística**: con datos del mismo período, ¿mis elecciones (bins, variables, coeficientes) capturaron señal o memoria del ruido? OOT responde una pregunta de **estabilidad temporal**: ¿la relación variable→default sobrevive al paso del tiempo, con el ciclo económico moviéndose? Un modelo puede pasar HO holgado y morir en OOT — típicamente cuando aprendió patrones específicos del período de desarrollo (una campaña comercial, un evento macro, una estacionalidad).

En riesgo, OOT es la validación que importa, porque la pregunta real de negocio es temporal: el modelo se entrena con 2024 y decidirá sobre 2027.

### 1.2 La regla de un solo disparo

*"La OOT se toca UNA vez, al validar."* La razón es información-teórica: cada vez que miras la OOT y ajustas algo en respuesta, filtras información de la OOT hacia el desarrollo. Tras suficientes iteraciones, la OOT deja de ser evidencia independiente y pasa a ser un segundo DEV con delay. Es el mismo fenómeno del *leaderboard overfitting* en Kaggle, con una diferencia: aquí nadie te da otra OOT fresca — el tiempo es el recurso no renovable.

Disciplina práctica: se congela el pipeline (variables, bins, coeficientes, cutoffs candidatos), se corre la OOT, se reporta lo que dio. Si el resultado obliga a rediseñar, la corrida OOT anterior se declara quemada y la nueva validación exige cosechas nuevas (o, como mínimo, declarar la degradación de independencia en el informe).

### 1.3 Variantes de esquema muestral en la industria

- **k-fold dentro de DEV:** legítimo para decisiones internas (¿5 o 6 bins?), siempre por cliente y siempre dentro del período de desarrollo. No sustituye OOT.
- **OOT múltiple (rolling):** carteras grandes validan sobre varias ventanas temporales sucesivas para ver la degradación como curva, no como punto.
- **Out-of-universe:** validar en un segmento o canal no usado en desarrollo (¿el modelo de sucursales sirve para el canal digital?). Responde una tercera pregunta: generalización poblacional.
- **Carteras chicas:** cuando no alcanza para cuatro muestras decentes, el orden de sacrificio es: primero se relaja HO (usando k-fold por cliente en DEV), nunca la OOT; y se compensa la incertidumbre con bootstrap (E3): intervalos de confianza en vez de puntos.

---

## 2. La regla de cliente único

### 2.1 El problema

Un cliente activo puede tener varias solicitudes en el período. Incluirlas todas parece "más datos", pero es **la misma evidencia repetida** (tu guion del curso lo dice igual): las filas comparten el mismo comportamiento subyacente, el mismo riesgo latente y, con ventanas que se solapan, hasta los mismos meses de desempeño. Dos filas del mismo cliente no son observaciones independientes.

Dos daños concretos:

1. **Métricas infladas por pseudo-replicación:** el n efectivo es menor que el n nominal; los errores estándar y la aparente estabilidad de los bins mienten.
2. **Fuga de identidad:** con split aleatorio por fila, el mismo cliente cae en DEV y en HO. El modelo lo "reconoce" en validación — el hold-out deja de ser datos no vistos. Es la fuga tipo 4, y es devastadora porque infla exactamente la métrica que debía ser honesta.

### 2.2 La regla y sus relajaciones

**Regla del curso:** cada cliente entra UNA vez, con su primera solicitud del período (la primera, no una al azar: elegir "la mejor" o "la más reciente" introduce sesgos de selección propios). En el ideal, nadie se repite ni siquiera entre desarrollo y validación.

**Relajación documentada para carteras chicas:** admitir múltiples solicitudes por cliente PERO particionando **por cliente** (todas las filas de un cliente van a la misma muestra) y, idealmente, corrigiendo la inferencia (errores estándar clusterizados, o al menos reportando el n de clientes junto al de filas). El bootstrap por cliente (E3) cierra el círculo.

**El equivalente en tu mundo:** es exactamente el *group split* (`GroupKFold` con `groups=id_cliente`). La diferencia cultural es que en riesgo la partición por grupo no es una opción avanzada: es la línea base, y saltársela invalida la entrega.

---

## 3. Taxonomía completa de fugas

Fuga = cualquier mecanismo por el cual información posterior a t₀ (o exterior al universo de producción) se filtra al entrenamiento o a la validación. Los cuatro tipos del curso, expandidos con sus formas de detección:

### Tipo 1 · Columnas ex-post
Estados "actuales", marcas de castigo/fraude, flags de cobranza, fecha de último pago "a hoy". Cualquier columna cuyo valor se consolidó **después** del desenlace.
- **Ejemplo canónico:** `estado_actual` con AUC 0.991 en la demo de C1. La columna literalmente codifica el desenlace.
- **Detección:** diccionario de datos con la semántica temporal de cada columna (¿cuándo se escribe este campo?); screening de IV/AUC univariado — todo lo que discrimina "demasiado bien" se audita (umbral IV > 0.5).
- **Sutileza:** hay ex-post disfrazados. `n_productos_vigentes` de la foto actual de la base (no de t₀) es ex-post; `motivo_cierre_cuenta` también. La pregunta discriminante: *¿este valor pudo cambiar después de t₀?*

### Tipo 2 · Fugas de ventana
La variable es conceptualmente legal pero su cálculo roza meses > t₀−lag: un `+1` en el slice, un `between` inclusivo de más, un join por mes calendario que arrastra el mes en curso, un backfill de fuente rezagada (M2 §2.2).
- **Detección:** tests de pipeline (ver §5) + el experimento del ancla: recalcular la variable con la ventana corrida un mes y comparar IV — si el IV "oficial" se parece al corrido y no al legal, hay contaminación.

### Tipo 3 · Fugas de población
Entrenar con clientes que no existirán así en producción: filtrar la población de desarrollo con condiciones que usan el futuro ("clientes que siguen activos hoy" — sobreviviente típico), o con reglas que producción no aplicará.
- **Caso clásico:** sesgo de supervivencia — construir la base desde la foto actual de clientes, perdiendo a los que se fueron o fueron castigados. La tasa de malos histórica queda subestimada y las variables quedan condicionadas a sobrevivir.
- **Detección:** reconciliar conteos contra registros de originación de la época (¿cuántas solicitudes hubo realmente en 2024-07 vs cuántas tiene mi base?).

### Tipo 4 · Fugas de identidad
El mismo cliente (o el mismo hogar, o el mismo RUT con productos distintos) repartido entre DEV y HO/OOT. §2 la desarrolló.
- **Detección trivial y obligatoria:** `assert set(dev.id_cliente).isdisjoint(ho.id_cliente)`.

### 3.1 El síntoma transversal

*"Un desempeño demasiado bueno."* AUC univariado 0.95+, IV > 0.5, un salto de Gini de 15 puntos respecto del modelo anterior. En riesgo de crédito los buenos modelos de admisión viven en Gini 40-60; todo lo que se salga por arriba se celebra **después** de auditarse, nunca antes. La frase de la lámina 19 para tatuarse: en producción la fuga se paga doble — el modelo colapsa Y la confianza del comité también. Y la confianza tarda años en reconstruirse.

---

## 4. Anatomía de la autopsia (el método, no solo el caso)

La demo de C1 (celda del AUC 0.991) modela el método general de autopsia de una variable sospechosa. Como procedimiento reutilizable:

1. **Congelar la evidencia:** IV/AUC univariado de la variable, en DEV y HO.
2. **Interrogar la semántica:** ¿qué significa la columna, cuándo se escribe, quién la escribe? (El diccionario de datos de la celda 4 de la demo — "nadie modela columnas que no sabe qué significan".)
3. **Cruzar contra el target:** tabla de contingencia variable × target. En una fuga ex-post, ciertas categorías predicen casi determinísticamente (todos los `castigado` son malos).
4. **Reconstruir la línea de tiempo:** ¿el valor de esta variable para este cliente pudo conocerse en t₀? Tomar 5 casos y seguirles la historia mes a mes. Es artesanal y es lo que convence a un comité.
5. **Veredicto documentado:** fuga → fuera de la matriz y a la lista de columnas prohibidas; señal legítima extraordinaria → explicación de negocio escrita (tercera pregunta de comité de C2).

---

## 5. La test-suite anti-fugas (donde tu experiencia vale doble)

La detección artesanal no escala; la defensa real es que el pipeline **no pueda** producir fugas sin que un test truene. El notebook implementa esta suite mínima; en un proyecto real vivirían como tests de CI:

```
T1  ventana_legal      : para cada variable declarada, fin_ventana <= t0 - lag(fuente)
T2  target_limpio      : el slice del target empieza en t0+1 y exige ventana completa
T3  sin_target_es_nan  : cosechas incompletas tienen target NaN, no 0
T4  cliente_unico      : (o partición por cliente) ids disjuntos entre DEV/HO/OOT
T5  columnas_prohibidas: lista negra explícita (estado_actual, marca_fraude, flags de
                         cobranza...) ausente de la matriz de modelación
T6  screening_alerta   : ninguna variable con IV > umbral_auditoria entra al modelo
                         sin un registro de auditoría aprobado
T7  poblacion_reconcilia: conteo de solicitudes por cosecha == registro de originación
```

Dos principios de diseño: (a) los tests corren **en cada ejecución** del pipeline, no una vez — las fugas se reintroducen con cada refactor; (b) T6 no bloquea automáticamente: exige el artefacto de auditoría, porque hay IVs altos legítimos y la decisión es humana y documentada.

---

## 6. El notebook: laboratorio de fugas

`M4_laboratorio_fugas.py` (Marimo) genera una cartera sintética y luego, con un selector, **fabrica cada fuga a propósito** y mide su precio:

| Escenario | Qué hace | Qué observarás |
|---|---|---|
| Base limpia | Variables legales, cliente único, split por cliente | El AUC honesto de referencia |
| Fuga 1: ex-post | Agrega `estado_actual` como predictor | AUC salta a ~0.99 |
| Fuga 2: ventana corrida | Variable de mora con fin en t₀+1 | IV se duplica; AUC sube "gratis" |
| Fuga 3: supervivencia | Filtra a los clientes que "siguen activos" al final del panel | Tasa de malos cae; el modelo aprende una población que no existe |
| Fuga 4: identidad | Duplica clientes y parte por fila | El gap DEV-HO desaparece artificialmente |

Cierra con la test-suite corriendo sobre cada escenario: la base limpia pasa los 7 tests; cada fuga hace fallar exactamente el suyo. Ese es el entregable reutilizable del módulo.

---

## 7. Preguntas de autoevaluación

1. ¿Qué pregunta responde HO que no responde OOT, y viceversa? Da un ejemplo de modelo que pasa una y falla la otra en cada dirección.
2. ¿Por qué "primera solicitud del período" y no "una solicitud al azar" o "la más reciente"?
3. Tu base viene de la foto actual del core. Nombra dos fugas de tipo 3 que probablemente ya tiene y cómo las reconciliarías.
4. Un colega defiende una variable con IV 0.8: "es que la utilización predice mucho". Diseña la autopsia de 5 pasos para esa variable.
5. ¿Por qué T6 (screening de IV alto) exige auditoría en lugar de descartar automáticamente? ¿Qué se pierde con el descarte automático?
6. La OOT dio Gini 12 puntos bajo HO y decides agregar una variable de estabilidad y revalidar. ¿Qué acabas de hacerle a la OOT y cómo lo declaras en el informe?

**Siguiente módulo:** M5 · La fábrica de variables como pipeline de ingeniería — la parte del proyecto donde tus 10 años de DS se convierten en ventaja competitiva directa.

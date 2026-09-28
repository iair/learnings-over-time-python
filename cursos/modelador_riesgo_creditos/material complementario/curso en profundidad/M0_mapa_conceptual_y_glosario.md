# M0 · Mapa conceptual y glosario

**Serie: Modelador de Riesgo en Profundidad** · Material complementario al curso *Modelador de Riesgo de Crédito* (Academia Bayes, ago-sep 2026)
Fase 1, módulo 0 de 8 · Documento de referencia

---

## 1. Cómo usar esta serie

Esta serie complementa las seis clases del curso con dos objetivos que el curso, por tiempo, no puede cubrir: **profundidad** (por qué cada técnica es como es, qué variaciones existen en la industria y cuándo usarlas) e **implementación de calidad de ingeniería** (cada mecanismo reconstruido desde cero en notebooks reproducibles, con la mentalidad de pipeline que ya traes de data science).

La serie tiene dos fases:

- **Fase 1 (M0–M7):** los conceptos de las clases 1 y 2, llevados a nivel de referencia. Cada módulo se puede leer de forma independiente, pero el orden sigue la lógica del pipeline.
- **Fase 2 (E1–E8):** todo lo que el curso declara como limitación, teaser o material de reserva — reject inference, calibración, bootstrap, bureau, ML, normativa, stress testing — más una bibliografía comentada que conecta cada módulo con el capítulo exacto donde profundizar.

Cada módulo combina hasta tres tipos de material:

| Tipo | Formato | Para qué sirve |
|---|---|---|
| Documento de referencia | `.md` (~3-4k palabras) | Teoría, variaciones de industria, criterios de decisión, errores comunes |
| Notebook interactivo | `.py` (Marimo) | Reconstruir el mecanismo desde cero con datos sintéticos autocontenidos; sliders para mover parámetros y ver el efecto |
| Planilla | `.csv`/`.xlsx` liviano (importable a Google Sheets) | Calculadoras, catálogos y cheatsheets de consulta rápida |

**Sugerencia de ritmo:** un módulo por semana en paralelo al curso funciona bien; M5–M7 conviene tenerlos leídos antes de cerrar la Entrega 1, porque cubren exactamente lo que la pauta del Lab 1 parte B evalúa.

---

## 2. El mapa: las seis clases son un solo pipeline

El curso insiste en algo que un data scientist experimentado reconoce de inmediato pero que en riesgo de crédito tiene consecuencias regulatorias: el scorecard no es un modelo, es un **pipeline con decisiones documentadas en cada etapa**. Cada clase construye una pieza, y cada pieza responde una pregunta distinta:

```
C1 · DISEÑO            ¿QUÉ predigo y con QUIÉN?
   │  target (90+ en 12m) · población · exclusiones · muestras DEV/HO/OOT/TTD
   ▼
C2 · DATOS             ¿CON QUÉ predigo?
   │  ancla temporal · fábrica de variables · calidad · binning/WoE/IV (screening)
   ▼
C3 · MODELO            ¿CÓMO combino las variables?
   │  correlación/VIF/PSI · regresión logística sobre WoE · scorecard en puntos
   ▼
C4 · CALIBRACIÓN       ¿QUÉ decisión tomo con el score?
   │  score → PD · tendencia central · cutoff · estrategia de negocio
   ▼
C5 · VALIDACIÓN        ¿FUNCIONA y SEGUIRÁ funcionando?
   │  discriminación · estabilidad · OOT (se toca UNA vez) · alertas · bootstrap
   ▼
C6 · GOBIERNO          ¿Quién responde por esto y cómo se audita?
      model card · audit trail · despliegue · monitoreo
```

Tres ideas transversales cruzan todo el pipeline y aparecen una y otra vez en esta serie:

1. **La regla de oro temporal.** Toda variable se calcula solo con información ≤ t₀; todo target se mide solo con información > t₀. Cruzar la línea en cualquier dirección es una fuga. Es la regla que hace la diferencia entre un modelo que funciona en el notebook y uno que funciona en producción (M2, M4, M7).
2. **Evidencia, no herencia.** El umbral 90+, la ventana de 12 meses, la exclusión de indeterminados: nada se hereda de "la industria" sin sustentarlo con los datos propios (roll rates, curvas de maduración). El comité no pregunta "¿qué usaste?" sino "¿por qué?" (M3).
3. **Explicabilidad como restricción de diseño, no como adorno.** El binning, el WoE, la monotonicidad y la regresión logística existen porque el resultado debe ser una tabla de puntos que un comité pueda leer y un regulador auditar. La pérdida de granularidad es un costo que se paga a cambio de estabilidad y gobernabilidad (M7, E5).

### 2.1 Dónde el 80% de los errores caros nace

El curso lo dice en la lámina 4 de C1 y vale la pena internalizarlo: los errores caros no nacen en el ajuste del modelo (C3), nacen en el **diseño** (C1) y en la **construcción de datos** (C2). Un target contaminado, una población mal definida o una variable con fuga producen un modelo que valida espectacular y colapsa en producción. Por eso la Fase 1 dedica seis de ocho módulos a C1 y C2, y solo después la Fase 2 avanza hacia calibración y validación.

---

## 3. Mapa de conceptos por clase

### Clase 1 · Diseño y supuestos → módulos M1–M4

| Concepto | Lámina(s) | Módulo donde se profundiza |
|---|---|---|
| Ciclo de riesgo: admisión / comportamiento / cobranza (+ provisiones) | C1-6, apuntes | M1 |
| Costo de equivocarse en ambas direcciones; pérdida esperada | C1-7 | M1 |
| Anatomía temporal: ventana de observación, t₀, ventana de desempeño | C1-8 | M2 |
| Máquina del tiempo: entrenar en un pasado que ya conoce su futuro | C1-9 | M2 |
| Ventanas escalonadas: cada crédito con SU ventana | C1-10 | M2 |
| Definición de default: umbral y su evidencia | C1-12 | M3 |
| Roll rates: matriz de transición, cura, punto de no retorno | C1-13 | M3 |
| Curvas de maduración y elección de la ventana de 12 meses | C1-14 | M3 |
| Indeterminados (zona gris 30-89), exclusiones, marca ex-post | C1-15, apuntes | M3 |
| Rechazados y sesgo de selección (limitación declarada) | C1-15 | E1 |
| Esquema muestral DEV / HO / OOT / TTD | C1-17 | M4 |
| Regla de cliente único y particiones por cliente | C1-18 | M4 |
| Los 4 tipos de fuga de información | C1-19 | M4 |
| Eventos extraordinarios (retiros AFP, crisis) | apuntes | E7 |
| Contexto normativo 90 días (reserva) | C1-28 | E6 |
| Por qué no ML desde el día uno (reserva) | C1-29 | E5 |
| Bootstrap para carteras chicas (teaser C5) | C1-18 | E3 |

### Clase 2 · Datos y selección I → módulos M5–M7

| Concepto | Lámina(s) | Módulo donde se profundiza |
|---|---|---|
| Ancla temporal aplicada a variables: [t₀−k, t₀−1] | C2-5/6 | M2, M5 |
| Lag de datos y anclas corridas (t₀−2 con bureau) | C2-6 | M2, E4 |
| Fábrica: familias × ventanas × agregadores | C2-8 | M5 |
| Convención de nombres como documentación | C2-9 | M5 |
| Agregadores: promedio, máximo, suma, tendencia, recencia | C2-10 | M5 |
| Recencia con censura (el valor 13) y missing estructural | C2-11 | M5, M6 |
| Ratios de negocio y el debate del denominador (¿qué ingreso?) | C2-12/13 | M5 |
| Missing informativo (MNAR): la renta de los independientes | C2-16 | M6 |
| Tres tipos de missing, tres tratamientos | C2-17 | M6 |
| Outliers, centinelas, imposibles de negocio, winsorización | C2-18 | M6 |
| Por qué binning; fine classing → coarse classing | C2-20/21 | M7 |
| WoE e IV: fórmulas, lectura, convención de signo | C2-22 | M7 |
| Monotonicidad: cuándo forzarla | C2-23 | M7 |
| Umbrales de Siddiqi y lectura del IV | C2-24 | M7 |
| Trampa 1: el IV premia el número de bins | C2-25 | M7 |
| Trampa 2: el IV es inestable con pocos malos | C2-26 | M7 |
| Trampa 3: IV altísimo = auditar (ventana corrida, desenlace) | C2-27 | M2, M7 |
| Screening ≠ selección | C2-29 | M7 (y puente a C3) |
| Feature engineering automático (reserva) | C2-37 | E5 |
| Variables de bureau (reserva) | C2-38 | E4 |

---

## 4. Estructura completa de la serie

### Fase 1 · El curso en profundidad

| # | Módulo | Materiales |
|---|---|---|
| M0 | Mapa conceptual y glosario (este documento) | .md |
| M1 | El problema de negocio y la economía del error | .md + Sheets (calculadora de trade-off) |
| M2 | Arquitectura temporal: t₀, ventanas, anclas y sus variaciones | .md + Marimo (simulador de anclas: mueve el ancla, mira el IV inflarse) |
| M3 | La definición de default: roll rates, cura, maduración, indeterminados | .md + Marimo (cadenas de Markov de mora; sensibilidad del umbral 30/60/90) |
| M4 | Esquema muestral y las 4 fugas de información | .md + Marimo (laboratorio de fugas + test-suite anti-fugas reutilizable) |
| M5 | La fábrica de variables como pipeline de ingeniería | .md + Marimo (fábrica declarativa con validador de ancla) + Sheets (catálogo de candidatas) |
| M6 | Calidad de datos: missing, outliers, centinelas | .md + Marimo (experimento: imputar vs bin propio, impacto medido) |
| M7 | Binning, WoE e IV: mecánica, derivación y trampas | .md + Marimo (binner interactivo; las 3 trampas reproducidas desde cero) |

### Fase 2 · Extensiones y limitaciones declaradas

| # | Módulo | Origen en el curso |
|---|---|---|
| E1 | Reject inference: augmentation, parceling, fuzzy, performance inference | Limitación declarada en C1-15 |
| E2 | Calibración a tendencia central: de score a PD | Teaser de C4 (mencionado en C1 ante el deterioro 2025) |
| E3 | Bootstrap e intervalos de confianza para carteras chicas | Teaser de C5 (C1-18) |
| E4 | Variables de bureau: rezago, costo y trade-off Gini/costo | Reserva C2-38 |
| E5 | Scorecard clásico vs ML; feature engineering automático | Reservas C1-29 y C2-37 |
| E6 | Contexto normativo: Basilea II/III, IFRS 9, CMF | Reserva C1-28 |
| E7 | Eventos extraordinarios y stress testing | Apuntes C2 (retiros AFP, crisis) |
| E8 | Bibliografía comentada: Siddiqi, Thomas/Edelman/Crook, Anderson | Referencia metodológica del curso |

---

## 5. Glosario bilingüe

Organizado por bloque temático. Cada entrada: **término ES** (*término EN*) — definición operativa y, cuando importa, la convención que usa el curso.

### 5.1 Arquitectura temporal

- **Punto de observación, t₀** (*observation point*) — El mes de la solicitud. Divide el tiempo en dos territorios: hacia atrás se construyen variables, hacia adelante se mide el target. Toda la disciplina anti-fugas se reduce a respetar esta frontera.
- **Ventana de observación** (*observation window*) — Período hacia atrás desde t₀ (típicamente 3, 6 o 12 meses) con el que se calculan las variables. Convención del curso: termina en t₀−1, el último cierre mensual consolidado antes de decidir.
- **Ventana de desempeño** (*performance/outcome window*) — Período hacia adelante desde t₀ (12 meses en el curso) en el que se observa si el crédito alcanza el evento de default. Territorio exclusivo del target: ninguna variable la toca.
- **Ancla temporal** (*temporal anchor*) — La regla operativa que fija dónde termina la ventana de observación: [t₀−k, t₀−1]. Con fuentes rezagadas (bureau que llega con 2 meses) el ancla se corre a t₀−2 y se documenta.
- **Lag de datos** (*data lag / reporting lag*) — Tiempo entre que un mes ocurre y su cierre queda consolidado y disponible. La razón por la que la variable no puede usar el cierre de t₀: cuando hay que decidir, ese cierre aún no existe.
- **Cosecha** (*vintage / cohort*) — Conjunto de créditos originados en el mismo mes. Unidad natural para analizar maduración, estabilidad de tasa de malos y particiones temporales.
- **Ventanas escalonadas** (*staggered windows*) — Cada solicitud se observa 12 meses desde SU fecha, no según año calendario. Permite comparar cosechas con la misma vara.
- **Máquina del tiempo** (*time machine framing*) — El encuadre que resuelve la confusión clásica "¿cómo sé si paga a 12 meses?": para entrenar nos paramos en el pasado, donde ese "futuro" ya ocurrió y está en los datos. En producción el futuro sí es desconocido y el modelo lo predice.
- **TTD** (*through-the-door*) — Solicitudes recientes sin ventana de desempeño completa. No tienen target y no entrenan; sirven para medir a quién se scorea HOY (estabilidad poblacional, PSI).

### 5.2 Target y definición de default

- **Default / malo** (*default / bad*) — En el curso: alcanzar 90+ días de mora dentro de los 12 meses posteriores a t₀. No es una opinión: es una definición operativa que se sustenta con evidencia de la propia cartera.
- **DPD** (*days past due*) — Días de mora. La métrica cruda sobre la que se definen bandas y umbrales.
- **Banda de mora** (*delinquency bucket*) — Discretización de los DPD: al día, 1-29, 30-59, 60-89, 90+. Las transiciones entre bandas son la materia prima de los roll rates.
- **Roll rate** (*roll rate / transition rate*) — Probabilidad de pasar de una banda a otra en un mes. La matriz completa es una matriz de transición; leerla fila por fila es la evidencia empírica del umbral de default.
- **Cura** (*cure*) — Volver a estar al día desde una banda de mora. En Banco Austral: 34% cura desde 1-29, 11% desde 30-59, 4% desde 60-89, 2% desde 90+. El desplome de la cura es el argumento del "punto de no retorno".
- **Punto de no retorno** (*point of no return*) — La banda desde la cual la cura es residual. Ahí se ancla el umbral de default: etiquetar antes mete ruido (muchos se recuperan solos), etiquetar después pierde señal (eventos tardíos y escasos).
- **Maduración** (*seasoning / maturation*) — El proceso por el cual una cosecha acumula defaults con el paso de los meses. La curva de maduración (% acumulado en 90+ vs meses desde el cursado) sustenta la elección de la ventana de desempeño.
- **Indeterminado / zona gris** (*indeterminate*) — Caso cuya peor mora en la ventana quedó entre 30 y 89 días: no cumple la definición de bueno ni la de malo. Se excluye del entrenamiento (no para balancear la muestra, sino para no contaminar las clases) y se documenta cuántos son. Igual recibe score después.
- **Exclusiones** (*exclusions*) — Casos que se sacan de la población de modelamiento con justificación documentada: fraude confirmado, sin ventana completa, sin historia mínima.
- **Marca ex-post** (*ex-post flag*) — Columna cuyo valor se conoce solo después del desenlace (marca de fraude, estado actual, flags de castigo/cobranza). Solo sirve para excluir o auditar; usarla como predictor es fuga tipo 1.
- **Unidad de análisis** (*unit of analysis*) — Qué es una fila del dataset. En el curso: la solicitud cursada (con regla de cliente único). Definirla mal invalida todo lo que sigue.
- **Tasa de malos** (*bad rate*) — Proporción de malos en una población o cosecha. Su estabilidad entre cosechas es un chequeo de diseño; su nivel es el insumo de la calibración (E2).

### 5.3 Esquema muestral

- **DEV** (*development sample*) — Muestra de desarrollo (70% de las cosechas 2024-07→2025-02 en el curso). Ahí se ajusta todo: binning, selección, coeficientes.
- **HO** (*hold-out*) — El 30% restante de las mismas cosechas, no visto durante el ajuste. Responde: ¿me sobreajusté?
- **OOT** (*out-of-time*) — Cosechas posteriores al período de desarrollo (2025-03→2025-06). La prueba honesta de generalización temporal. **Se toca UNA vez, al validar**; si se usa para elegir variables, deja de ser evidencia y pasa a ser desarrollo.
- **Cliente único** (*unique customer rule*) — Cada cliente entra una sola vez (su primera solicitud del período). Dos filas del mismo cliente no son independientes: inflan métricas y arriesgan al mismo cliente en DEV y HO.
- **Partición por cliente** (*customer-level split*) — Si se relaja la regla de cliente único (cartera chica), la partición DEV/HO se hace por cliente, nunca por fila.
- **Bootstrap** (*bootstrap*) — Remuestreo con reemplazo para construir intervalos de confianza de las métricas cuando la cartera es chica. Se profundiza en E3.
- **Semilla** (*seed*) — Valor que fija la aleatoriedad de la partición para reproducibilidad. En el curso, personal por estudiante.

### 5.4 Fugas de información

- **Fuga de información** (*data/information leakage*) — Cualquier variable que usa información posterior a t₀. Síntoma clásico: desempeño demasiado bueno (AUC 0.95+ univariado es alerta, no hallazgo).
- **Fuga tipo 1 — columnas ex-post** (*ex-post columns*) — Estados "actuales", marcas de castigo/fraude, flags de cobranza. El AUC 0.991 de `estado_actual` en la demo de C1 es el ejemplo canónico.
- **Fuga tipo 2 — fuga de ventana** (*window leakage*) — Una variable "histórica" cuya ventana roza meses posteriores a t₀. Un error de índice de UN mes ([t₀−k, t₀+1] en vez de [t₀−k, t₀−1]) duplica el IV aparente.
- **Fuga tipo 3 — fuga de población** (*population leakage*) — Entrenar con clientes que no existirán así en producción (p. ej., filtrar por condiciones futuras).
- **Fuga tipo 4 — fuga de identidad** (*identity leakage*) — El mismo cliente repartido entre DEV y HO. El hold-out deja de ser "datos no vistos".
- **AUC** (*area under the ROC curve*) — Métrica de discriminación (0.5 = azar, 1 = perfecto). En riesgo se reporta más el Gini = 2·AUC − 1.

### 5.5 Fábrica de variables

- **Fábrica de variables** (*feature factory*) — Convención de tres ejes (familias × ventanas × agregadores) que se cruzan sistemáticamente en vez de inventar variables de a una. Ventajas: cobertura, auditabilidad, reutilización.
- **Familia** (*variable family*) — Qué se mide: mora propia, utilización, deuda y saldos, pagos, ahorro/ingreso, bureau externo, consultas.
- **Agregador** (*aggregator*) — Cómo se resume la serie en la ventana. Cada uno responde una pregunta de negocio distinta: promedio (comportamiento habitual), máximo (peor momento), suma/conteo (actividad acumulada), tendencia Δ (¿mejora o empeora?), recencia (¿hace cuánto?).
- **Recencia** (*recency*) — Meses desde el último evento (p. ej., última mora) dentro de la ventana. Requiere tratar la censura.
- **Censura** (*censoring*) — Cuando el evento no ocurre dentro de la ventana observada. Convención del curso: recencia sin evento → 13 (fuera del rango 0-12), para que "hace 12 meses" y "nunca" no compartan valor. El binning lo aísla en su propio bin.
- **Tendencia / delta** (*trend / delta*) — Variación entre extremos de la ventana (t₀−k vs t₀−1). Necesita ambos extremos: si falta t₀−12, aparece missing estructural.
- **Ratio de negocio** (*business ratio*) — Hipótesis de negocio escrita en fórmula: utilización (saldo/cupo), carga financiera (deuda/ingreso), pago sobre facturación, cuota/ingreso, ahorro/ingreso. Donde vive el criterio del analista; ningún algoritmo los inventa.
- **Foto** (*snapshot*) — Valor al último cierre disponible (t₀−1), sin agregación.
- **Convención de nombres** (*naming convention*) — `<concepto>_<agregador>_<ventana>` (p. ej., `uso_linea_prom_6m`). El nombre ES la documentación: si hay que abrir el código para entenderlo, el nombre está mal.

### 5.6 Calidad de datos

- **Missing estructural** (*structural missing*) — El dato no existe por construcción, no falta: Δ12m en clientes con 6-11 meses de historia. Tratamiento: bin propio "sin historia suficiente". Nunca imputar.
- **MNAR** (*missing not at random*) — La ausencia depende del valor no observado o de características del cliente: 36% de los independientes no declara renta vs 7% de los dependientes. La ausencia ES señal (WoE propio del bin MISSING). Imputar por la media la borra y contamina la distribución.
- **MCAR** (*missing completely at random*) — Ausencia sin patrón (fallo puntual de carga). El único tipo donde imputar es defendible, si es marginal, documentando criterio y midiendo impacto.
- **Imputación** (*imputation*) — Rellenar valores faltantes. El error caro no es imputar: es imputar sin preguntarse a qué tipo de missing pertenece.
- **Centinela** (*sentinel value*) — Valor especial usado como "sin dato" (999999, −1, 0). Se detecta mirando la masa puntual (concentración en un valor exacto), no el promedio.
- **Imposible de negocio** (*business-impossible value*) — Valor que viola la lógica del dominio: utilización = 47 es error a revisar; utilización = 1.2 puede ser sobregiro pactado legítimo. Distinguirlos exige conocer el negocio.
- **Winsorización** (*winsorizing*) — Acotar los extremos al percentil 1 y 99. Defendible; borrar filas casi nunca lo es.
- **Masa puntual** (*point mass*) — Porcentaje de observaciones en el valor más frecuente. Parte del chequeo mínimo por variable: n, % missing, mín, p1, mediana, p99, máx, % en el valor más frecuente.

### 5.7 Binning, WoE e Information Value

- **Binning / classing** (*binning*) — Agrupar una variable en tramos (bins). Captura no linealidades sin polinomios, neutraliza outliers, absorbe el missing como categoría y produce la tabla de puntos del scorecard.
- **Fine classing** — Primera pasada: ~20 bins de igual tamaño para VER la forma de la relación con el target.
- **Coarse classing** — Fusión hasta que cada bin tenga masa suficiente (≥5% de la muestra) y sentido de negocio. Regla práctica: 4-6 bins finales. La fusión no es automática: es donde el modelador decide.
- **WoE** (*weight of evidence*) — WoE(bin) = ln(%buenos / %malos): cuánto se desvía el bin del perfil promedio. Convención del curso (target 1 = malo): WoE alto = bin bueno. **Otras librerías invierten el signo** — verificar siempre antes de interpretar.
- **IV** (*information value*) — IV = Σ (%buenos − %malos) × WoE. Un solo número por variable; un bin pesa por separar mucho Y por tener volumen. Es una divergencia simétrica entre las distribuciones de buenos y malos (se deriva en M7).
- **Umbrales de Siddiqi** — Lectura de industria del IV: <0.02 sin poder, 0.02-0.10 débil, 0.10-0.30 medio, 0.30-0.50 fuerte, >0.50 **auditar** (sospecha de fuga o proxy del target). Son señales de dónde mirar, no ley.
- **Monotonicidad** (*monotonicity*) — WoE que crece o decrece consistentemente a lo largo de los bins. Las no-monotonías con pocos malos casi siempre son ruido; se fusiona (el IV apenas cambia si era ruido). Excepción legítima: formas en U que el negocio explica (edad).
- **Screening vs selección** — El IV mira una variable a la vez: sirve para DESCARTAR lo inútil y PRIORIZAR la revisión manual, no para elegir el modelo (dos variables con IV 1.2 correlacionadas al 95% aportan una sola vez). La selección multivariada es C3.

### 5.8 Adelantos de C3–C6 y Fase 2 (para no perderse cuando aparezcan)

- **Scorecard** — El entregable final: tabla de puntos por bin, derivada de una regresión logística sobre WoE, con escala definida por PDO/factor/offset.
- **PD / LGD / EAD** (*probability of default / loss given default / exposure at default*) — Los tres componentes de la pérdida esperada = PD × LGD × EAD. El curso modela PD; LGD y EAD son territorio de provisiones (M1, E6).
- **Cutoff** — El punto de corte del score que separa aprobar de rechazar: equilibrio explícito entre apetito y pérdida (C4, M1).
- **Calibración / tendencia central** (*calibration / central tendency*) — Ajustar la PD del modelo al nivel de tasa de malos esperado de largo plazo, no al de la época del entrenamiento. La respuesta a "¿el deterioro 2025 no invalida entrenar con 2024?" (E2).
- **PSI** (*population stability index*) — Mide cuánto cambió la distribución de una variable o del score entre dos poblaciones (DEV vs TTD, p. ej.). El termómetro del monitoreo (C5).
- **Gini** — 2·AUC − 1. La métrica de discriminación estándar en riesgo.
- **VIF / stepwise** — Herramientas de selección multivariada de C3: colinealidad y construcción incremental del modelo.
- **Reason codes** — Las razones legibles de un rechazo ("línea usada sobre 56% → −18 puntos"). Salen gratis del scorecard; de un boosting, no (E5).
- **Reject inference** — Familia de técnicas para corregir el sesgo de selección de entrenar solo con aprobados: augmentation, parceling, fuzzy, performance inference (E1).
- **Model card / audit trail** — Documentación estandarizada del modelo y trazabilidad de sus decisiones (C6).
- **Stress testing** — Evaluar el modelo bajo escenarios adversos; los períodos de crisis excluidos del entrenamiento se conservan justamente para esto (E7).

---

## 6. Convenciones de toda la serie

Para que los 16 módulos sean consistentes entre sí y con el curso:

1. **Notación temporal:** t₀ = mes de la solicitud; ventanas de variables [t₀−k, t₀−1]; ventana de target [t₀+1, t₀+12].
2. **Target:** 1 = malo (90+ DPD en 12m), 0 = bueno; indeterminados fuera del entrenamiento.
3. **WoE:** ln(%buenos/%malos), es decir WoE alto = bin bueno (convención del curso y de nikodym). Cuando un módulo muestre la convención invertida (scorecardpy, algunos textos), lo señalará explícitamente.
4. **Muestras:** IV y binning se ajustan en DEV, se contrastan en HO; la OOT no se toca hasta el módulo de validación.
5. **Notebooks:** Marimo (`.py`), autocontenidos — cada uno incluye su generador de datos sintéticos y no depende de archivos externos. Dependencias: `marimo`, `numpy`, `pandas`, `matplotlib`.
6. **Planillas:** `.csv` o `.xlsx` sin macros, importables a Google Sheets sin pérdida.
7. **Idioma:** español, con el término en inglés entre paréntesis la primera vez que aparece (el inglés domina la literatura y las entrevistas técnicas del rubro).

---

## 7. Del data scientist al modelador de riesgo: el cambio de mentalidad

Cierre de este módulo con lo que, viniendo de 10 años en DS, más cuesta y más vale la pena internalizar. Son cinco desplazamientos de énfasis:

1. **De maximizar métricas a defender decisiones.** En DS general, un AUC más alto es mejor casi por definición. En riesgo, un AUC sospechosamente alto es una alerta de fuga, y un modelo con 2 puntos menos de Gini pero explicable ante un comité y estable en el tiempo gana. La pregunta del comité nunca es "¿cuánto discrimina?" sino "¿por qué debería creerte?".
2. **Del split aleatorio a la disciplina temporal.** El `train_test_split` por fila —reflejo automático en DS— es aquí la fuga tipo 4. La partición correcta respeta clientes y tiempo (DEV/HO dentro del período, OOT después), porque la pregunta real es de generalización temporal, no estadística.
3. **Del feature engineering exploratorio a la fábrica auditada.** Generar miles de features y dejar que el modelo elija funciona en Kaggle; en admisión bancaria produce variables imposibles de auditar contra el ancla temporal y basura correlacionada a escala industrial. La fábrica declarativa (pocas familias, convención clara, revisión humana) es menos "creativa" y más valiosa — y es exactamente donde tu experiencia en pipelines se convierte en ventaja: la fábrica ES un pipeline, con contratos, validaciones y tests.
4. **Del dato como insumo al dato como evidencia.** Cada exclusión, cada imputación, cada fusión de bins queda documentada porque el modelo será auditado por validación interna, auditoría y potencialmente el regulador. El notebook no es un borrador: es parte del expediente.
5. **Del modelo como producto al modelo como política de crédito.** El scorecard no termina en un `predict()`: termina en un cutoff que aprueba o rechaza personas, con costo asimétrico en ambas direcciones y consecuencias de negocio que se ven 12-18 meses después. Esa distancia temporal entre decisión y consecuencia es lo que hace al diseño (C1) más importante que al algoritmo (C3).

El resto de la serie desarrolla cada pieza. Siguiente parada: **M1 · El problema de negocio y la economía del error**.

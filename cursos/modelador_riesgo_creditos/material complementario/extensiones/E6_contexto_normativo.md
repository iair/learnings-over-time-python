# E6 · Contexto normativo: Basilea, IFRS 9 y la CMF

**Serie: Modelador de Riesgo en Profundidad** · Fase 2, extensión 6 de 8
Origen: reserva C1-28 ("¿tiene relación con el default de 90 días de la CMF?") · Nota: panorama conceptual a nivel de lo vigente al cierre de esta serie; las normas cambian — verificar siempre contra la fuente oficial vigente.

---

## 1. Por qué un modelador "gerencial" necesita el mapa regulatorio

El scorecard del curso es un modelo de **gestión** (decide admisión), no un modelo regulatorio. Pero vive en un banco donde conviven tres marcos que usan los mismos conceptos (PD, default, pérdida esperada) con definiciones y propósitos distintos, y el modelador que no conoce el mapa comete dos errores típicos: reinventar definiciones que ya existen reguladas (y luego no poder conciliar), o asumir que su modelo gerencial debe cumplir exigencias que no le aplican (y sobre-ingeniería). El mapa mínimo:

| Marco | Pregunta | Producto |
|---|---|---|
| **Basilea (capital)** | ¿Cuánto capital debe tener el banco para pérdidas INESPERADAS? | Activos ponderados por riesgo; PD/LGD/EAD regulatorias (si usa modelos internos) |
| **IFRS 9 (provisiones contables)** | ¿Cuánta pérdida ESPERADA reconozco hoy en el balance? | Provisiones por etapas (stages), PD lifetime, forward-looking |
| **Normativa local (CMF en Chile)** | Reglas de provisión, definición de mora/castigo, gobernanza de modelos | Normas de provisiones por tipo de cartera, matrices estándar, exigencias de gestión |

## 2. Basilea en dos páginas

- **Basilea II (2004)** introdujo la arquitectura de tres pilares (capital mínimo / revisión supervisora / disciplina de mercado) y, para riesgo de crédito, dos rutas: **estándar** (ponderadores fijos por tipo de exposición) e **IRB** (*internal ratings-based*: el banco estima PD —y en IRB avanzado también LGD y EAD— con modelos internos aprobados por el supervisor).
- **La definición de default de Basilea** es el ancestro del 90+ del curso: obligación con más de 90 días de atraso **o** *unlikeliness to pay* (indicadores de improbabilidad de pago: castigo, renegociación forzosa/distressed restructuring, quiebra). Nota el "o": la definición regulatoria es más ancha que el puro DPD — de ahí la discusión de renegociaciones de M3.
- **Exigencias IRB que iluminan la práctica aun sin ser IRB:** PD calibrada a promedio de largo plazo (la tendencia central de E2), horizonte de un año, downturn LGD, uso efectivo del modelo en la gestión (*use test*: el regulador no acepta modelos "de vitrina"), y validación independiente periódica. La fórmula de capital de Basilea (ASRF/Vasicek) convierte PD/LGD/EAD en capital por la pérdida inesperada a un percentil 99,9 — la PD del modelador es insumo directo de cuánto capital consume cada crédito, lo que conecta con el pricing por riesgo de M1.
- **Basilea III (post-2008, con implementación escalonada hasta los 2020s)** endureció cantidad y calidad del capital, agregó colchones, ratio de apalancamiento y liquidez, y con las reformas finales ("Basilea 3.1/endgame") limitó los beneficios de los modelos internos (pisos de output respecto del método estándar). En Chile, la LGB de 2019 alineó la exigencia de capital con Basilea III bajo supervisión de la CMF.

## 3. IFRS 9 en dos páginas

Reemplazó (2018) al modelo de "pérdida incurrida" de IAS 39 por **pérdida esperada** (ECL): se provisiona ANTES del evento. La maquinaria:

- **Tres etapas:** Stage 1 (riesgo no deteriorado significativamente: provisión = ECL a 12 meses), Stage 2 (aumento significativo del riesgo desde la originación, *SICR*: ECL de por vida/lifetime), Stage 3 (deteriorado/default: lifetime con reconocimiento de interés sobre neto).
- **Consecuencias de modelamiento:** PD a 12 meses Y curvas de PD lifetime; criterio de SICR (típicamente umbrales de deterioro relativo de PD + backstop de 30 días de mora); **forward-looking**: las ECL ponderan escenarios macroeconómicos (base/adverso/favorable) — la PD es PIT condicionada a escenarios (contraste directo con la TTC de capital: E2 §3.2).
- **El mismo cliente, tres números:** un banco sofisticado tiene PD gerencial (admisión, este curso), PD IFRS 9 (PIT, lifetime, por escenarios) y PD regulatoria de capital (TTC, floors). Conciliarlas —o al menos explicarlas— es trabajo recurrente del área de modelos.

## 4. El marco chileno (CMF) en lo que toca al curso

- **Provisiones:** la normativa de la CMF (RAN, en particular el capítulo B-1 del Compendio) define provisiones por riesgo de crédito con métodos estándar para carteras grupales (consumo/hipotecario, con matrices normativas que dependen fuertemente de la morosidad) y evaluación individual para deudores comerciales grandes. La cartera de consumo del curso cae típicamente en evaluación **grupal**, donde la mora de 90 días es un umbral operativo central (cartera en incumplimiento) — de ahí la respuesta a la pregunta de reserva: el 90+ del curso no es un invento del profesor: dialoga con la definición de incumplimiento del marco local e internacional, lo que hace las cifras conciliables entre gestión, contabilidad y regulación.
- **Castigos:** la normativa fija plazos máximos para castigar según tipo de crédito (consumo entre los más cortos) — el "default contable" tardío que M3 descartó como target por llegar tarde.
- **Gobernanza de modelos:** la CMF exige gestión de riesgo con modelos gobernados (roles, validación, documentación) en línea con la supervisión basada en riesgo; la lógica de model card/audit trail de C6 no es opcional cultural, es expectativa supervisora.
- **Datos:** la información de deuda consolidada del sistema (el "bureau CMF") y la ley de protección de datos personales acotan qué se consulta y usa (E4 §4).

## 5. Qué debe retener el modelador gerencial

1. **Alinear la definición de default con el marco (90+/12m) compra conciliabilidad gratis** entre el modelo de admisión, las provisiones y el reporte regulatorio. Apartarse es legítimo pero se paga en explicaciones (M3 §2.4).
2. **Las exigencias IRB son un estándar de calidad exportable:** tendencia central, use test, validación independiente, documentación — aplicarlas al modelo gerencial (proporcionalmente) es adoptar el estado del arte aunque nadie lo exija.
3. **PIT vs TTC no es jerga: es saber qué número entregas a quién.** Admisión y pricing quieren la mejor predicción condicional (PIT-ish); capital quiere estabilidad de ciclo (TTC); IFRS 9 quiere PIT con escenarios. Un solo scorecard, tres calibraciones (E2).
4. **La frontera gerencial/regulatorio es de gobernanza, no de matemática:** el mismo Gini, otro nivel de evidencia, aprobación y trazabilidad. Presupuestar esa diferencia al planificar un proyecto es criterio senior.

## 6. Para profundizar

- BCBS, *An Explanatory Note on the Basel II IRB Risk Weight Functions* — la derivación de la fórmula de capital, legible.
- IFRS 9 (norma) + las guías de implementación de los grandes auditores para ECL — el estándar y su práctica.
- CMF: Compendio de Normas Contables (B-1 provisiones) y normativa de Basilea III en Chile — leer la fuente, no resúmenes.
- BCBS WP14 (*Studies on the Validation of Internal Rating Systems*) — el puente entre este módulo y E2/E3.

## 7. Preguntas de autoevaluación

1. ¿Por qué "unlikeliness to pay" hace la definición regulatoria más ancha que el 90+ puro? Da dos ejemplos de default sin 90 días de mora.
2. Explica a un gerente por qué el banco tiene tres PDs para el mismo cliente y ninguna está "mala".
3. ¿Qué es el use test y por qué mata a los modelos "de vitrina"? ¿Cómo lo evidenciarías para el scorecard del curso?
4. La matriz normativa grupal de provisiones depende de la mora observada; tu scorecard predice mora futura. ¿Cómo conviven en la gestión de una cartera de consumo?
5. Stage 2 exige detectar "aumento significativo de riesgo". Propón un criterio SICR usando el score de admisión + el behavior, y discute sus falsos positivos.

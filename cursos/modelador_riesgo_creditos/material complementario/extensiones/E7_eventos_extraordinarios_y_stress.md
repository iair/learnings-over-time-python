# E7 · Eventos extraordinarios y stress testing

**Serie: Modelador de Riesgo en Profundidad** · Fase 2, extensión 7 de 8
Origen: tus apuntes de C2 (retiros de AFP, crisis: analizar aparte, decidir con documentación) y el deterioro 2025 del caso

---

## 1. Dos preguntas distintas que suelen confundirse

1. **¿Qué hago con los períodos anómalos en mis datos de desarrollo?** (mirando hacia atrás — un problema de M3/M6).
2. **¿Qué le pasaría a mi cartera y a mi modelo si viniera un shock?** (mirando hacia adelante — stress testing).

Están conectadas por un hecho: los períodos anómalos históricos son la materia prima empírica de los escenarios de estrés. Por eso la regla de M3 §5.2 — excluir del entrenamiento si distorsiona, pero **conservar siempre** — no es conservadurismo de archivo: es guardar el único experimento natural disponible.

## 2. Anatomía de un evento extraordinario (el caso AFP como ejemplo)

Los retiros de fondos previsionales en Chile (2020-2021) son el ejemplo perfecto de por qué estos períodos rompen los modelos:

- **Liquidez artificial masiva:** hogares que normalmente estarían al límite recibieron inyecciones de efectivo → la mora cayó a mínimos históricos → las cosechas 2020-21 son "demasiado buenas".
- **Relaciones variable→default distorsionadas:** la carga financiera alta dejó de predecir default (porque el retiro pagaba las cuotas) — el WoE de esa variable en ese período apunta al revés del estructural.
- **La trampa de entrenarlo:** un modelo desarrollado con esas cosechas aprende que el riesgo es bajo y las señales débiles; cuando la liquidez se agota (2022-23, y el deterioro que el caso del curso pone en 2025), el modelo subestima sistemáticamente. Es la versión macro de la fuga: información de un régimen usada en otro.

**El protocolo de manejo (formalizando tus apuntes):**
1. Detectar y delimitar el período (quiebres en la tasa de malos por cosecha, en los WoE por período, en el PSI de variables).
2. Medir la distorsión: WoE/IV por variable dentro vs fuera del período; tasa de malos por cosecha contra la curva de maduración normal.
3. Decidir con documentación: excluir / marcar con dummy de régimen / mantener — según cuánto distorsiona y si el fenómeno es repetible.
4. Archivar el período etiquetado como **escenario**: distribución de shocks observados sobre variables y sobre tasas, para el stress de la sección siguiente.

## 3. Stress testing: la caja de herramientas

### 3.1 Qué se estresa en un modelo de admisión

- **El nivel (calibración):** ¿qué pasa con la pérdida esperada de la cartera aprobada si la tasa de malos escala ×1.5, ×2, ×3 (o al nivel de la peor cosecha histórica)? Con la maquinaria de M1: mover la PD de cada tramo y recomputar pérdida/utilidad al cutoff vigente. Es el estrés mínimo que todo comité debería ver junto con la propuesta de cutoff.
- **El ordenamiento (discriminación):** ¿el Gini sobrevive al régimen? Evidencia: medir el Gini del modelo EN el período anómalo conservado (el modelo entrenado fuera, evaluado dentro). Si el ordenamiento aguanta (suele degradarse menos que el nivel, E2 §2), la respuesta de gestión es recalibrar y ajustar cutoff, no rediseñar.
- **Las variables (insumos):** shocks sobre variables concretas — desempleo sube → abonos observados caen → carga financiera sube → la distribución de score se corre. Trasladar un shock macro a las variables del modelo exige un modelo satélite (macro → variables) o supuestos documentados; la versión simple (mover directamente la distribución de score) es legítima como primera aproximación.
- **La estrategia (cutoff y apetito):** dado el estrés, ¿el cutoff vigente sigue dentro del apetito? ¿Cuál sería el cutoff contingente? Tener la respuesta ANTES del shock es la diferencia entre gestión y reacción.

### 3.2 Diseño de escenarios

- **Históricos:** replicar el peor episodio observado (la crisis 2008-9, el estallido/pandemia, el deterioro 2025 del caso). Ventaja: internamente coherentes (así se movieron las cosas de verdad). Desventaja: el próximo shock no repite el anterior.
- **Hipotéticos:** narrativas construidas (desempleo +X pp, tasas +Y pb) con la coherencia impuesta por un modelo macro o por el juicio del área de estudios. Los reguladores publican escenarios de referencia (los ejercicios de estrés supervisores) que sirven de ancla.
- **Sensibilidad univariada:** mover una perilla a la vez (solo tasa de malos, solo LGD). Menos realista, más diagnóstica: muestra a qué es sensible el resultado.
- **Reversa (reverse stress):** ¿qué tamaño de shock quiebra el apetito/el negocio? Se busca el escenario que produce el resultado inaceptable — poderosa para encontrar concentraciones escondidas.

### 3.3 El vínculo con IFRS 9 y capital (cierre del triángulo con E6)

El forward-looking de IFRS 9 ES un stress ponderado: ECL = Σ prob(escenario) × ECL(escenario). Y los ejercicios de estrés de capital preguntan si el banco resiste el escenario adverso con los colchones de Basilea III. El modelador de admisión aporta la pieza PD-por-tramo-bajo-escenario; entender el uso downstream evita producir números inconsistentes entre gestión, contabilidad y capital.

## 4. Señales de alerta temprana (el puente con el monitoreo de C5)

El estrés se conecta con la operación mediante indicadores que anticipan el régimen: mora temprana (1-29) por cosecha joven (la primera en moverse), roll rate 0→1-29 (el flujo de entrada a mora), PSI del score y de las variables macro-sensibles, % de utilización de líneas (los clientes estresados se copan la línea ANTES de caer — el patrón de M2). Un tablero que junte estos indicadores con gatillos predefinidos (si X supera Y, se activa el cutoff contingente) convierte el ejercicio de estrés en política operativa.

## 5. Para profundizar

- BCBS, *Stress testing principles* (2018) — los principios supervisores, cortos y legibles.
- Los informes de estabilidad financiera del banco central local (para Chile, el IEF del BCCh) — de dónde salen los escenarios de referencia y el análisis de los episodios (retiros incluidos).
- Breeden, *Reinventing Retail Lending Analytics* — vintage analysis, descomposición edad-período-cosecha (APC) y forecasting de cosechas bajo escenarios: la técnica que une maduración (M3) con estrés.

## 6. Preguntas de autoevaluación

1. ¿Por qué "excluir pero conservar" y no "excluir y listo"? ¿Qué se pierde exactamente si el período anómalo se borra?
2. Con las curvas de maduración de M3 y un shock que duplica la tasa de las cosechas jóvenes, proyecta qué verá el monitoreo a 3, 6 y 12 meses. ¿Cuál es el primer indicador en moverse?
3. Diseña el stress mínimo para acompañar la propuesta de cutoff del proyecto del curso: escenarios, perillas, y el formato de una lámina para comité.
4. El Gini en el período anómalo conservado cayó de 58 a 51, y la tasa se triplicó. ¿Recalibras, ajustas cutoff, rediseñas, o las tres? Ordena las acciones con su justificación.
5. Construye el reverse stress: ¿qué combinación de tasa de malos y LGD hace que la cartera aprobada al cutoff vigente destruya valor? ¿Qué tan lejos está del escenario histórico peor?

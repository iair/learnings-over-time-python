# E1 · Reject inference: modelar a los que nunca viste

**Serie: Modelador de Riesgo en Profundidad** · Fase 2, extensión 1 de 8
Origen: limitación declarada en C1-15 ("aprendemos solo de los aprobados")

---

## 1. El problema: sesgo de selección por diseño

Un modelo de admisión se entrena con solicitudes **cursadas**: solo de ellas se observa el desempeño. Pero en producción scorea a **todos los solicitantes**, incluidos los perfiles que la política anterior rechazaba sistemáticamente. El modelo aprende P(malo | X, aprobado), y se usa como si fuera P(malo | X). La brecha entre ambas es el sesgo de selección.

### 1.1 Por qué importa (y cuánto)

- **Extrapolación ciega:** en las zonas de X donde casi todo fue rechazado (score bajo del modelo anterior, thin files), el nuevo modelo tiene poquísimos datos, y los pocos que hay son **atípicos**: fueron aprobados por excepción (override comercial, garantías, error), así que ni siquiera representan a su zona.
- **Atenuación de pendientes:** las variables que la política anterior ya usaba para rechazar quedan con rango restringido en la muestra aprobada → sus coeficientes se atenúan → el nuevo modelo subestima el riesgo exactamente donde más importa.
- **Círculo vicioso de política:** cada generación de modelo hereda los sesgos de la política anterior. Si la política vieja rechazaba injustificadamente a un segmento, el nuevo modelo nunca aprende que era rechazo injustificado.

La magnitud depende de la **tasa de aprobación**: con 90% de aprobación el sesgo es menor (se observa casi todo el espectro); con 40%, la mitad del universo es terra incognita. Primera tarea profesional: **dimensionar** — comparar la distribución de score/variables de aprobados vs rechazados (un PSI entre ambos) antes de decidir si se corrige.

## 2. El marco conceptual: esto es un problema de missing data

El desempeño de los rechazados es un missing **MNAR por construcción**: falta precisamente porque la política predijo que sería malo. Toda técnica de reject inference es, en el fondo, una hipótesis sobre ese mecanismo de missing — y por eso ninguna es magia: **no crean información, redistribuyen supuestos**. La honestidad metodológica consiste en declarar el supuesto de cada técnica.

## 3. Las técnicas

### 3.1 Augmentation (re-ponderación)

**Idea:** re-ponderar a los aprobados para que representen a la población total. Se modela P(aprobado | X) (un "modelo de aceptación") y cada aprobado recibe peso 1/P(aprobado | X): los aprobados "raros" (que se parecen a los rechazados) pesan más.

**Supuesto:** MAR — condicional a X, los rechazados se comportan como los aprobados parecidos a ellos. Es exactamente el supuesto que NO se puede verificar (inverse probability weighting clásico, con el mismo talón de Aquiles).

**Riesgos:** pesos extremos en las zonas de poca aprobación → varianza explosiva; se estabiliza truncando pesos (y documentándolo).

### 3.2 Parceling (asignación por tramos)

**Idea:** scorear a los rechazados con un modelo preliminar (entrenado en aprobados), agruparlos por tramo de score, y asignarles desenlace **proporcional a la tasa de malos del tramo... ajustada**: la práctica estándar infla la tasa de los rechazados respecto de la de los aprobados del mismo tramo (típicamente ×1.5-×4, decreciente con el score), reconociendo que fueron rechazados por algo.

**Supuesto:** el multiplicador. Es un número inventado con criterio experto — el método es transparente sobre su arbitrariedad, lo que paradójicamente lo hace defendible: el supuesto está a la vista y se puede sensibilizar (correr el modelo con ×1.5, ×2, ×3 y mirar la estabilidad de los coeficientes).

### 3.3 Fuzzy augmentation (asignación parcial)

**Idea:** en lugar de asignar a cada rechazado un 0 o un 1, se **duplica**: entra como bueno con peso (1−p̂) y como malo con peso p̂, donde p̂ es su PD estimada por el modelo preliminar (ajustada al alza como en parceling). Es la versión suave de parceling: usa la información individual completa en vez de promedios por tramo.

**Supuesto:** que el modelo preliminar extrapola razonablemente a los rechazados — el mismo salto de fe, repartido con más elegancia. Es la técnica más usada en la práctica de scorecards (es la que implementan las suites comerciales clásicas).

### 3.4 Performance inference con datos externos (la única que agrega información)

**Idea:** observar el desempeño real de los rechazados **en otra parte**: el bureau permite ver si el cliente que rechazaste tomó crédito en otra institución y cómo le fue (mora en el sistema). También: cohortes de prueba (aprobar aleatoriamente una fracción pequeña bajo el cutoff — un experimento controlado, caro pero oro puro), o el desempeño de rechazados que luego fueron aprobados en una solicitud posterior.

**Supuestos:** el desempeño externo es proxy del que habría tenido contigo (distinto producto, distinto monto); el que consiguió crédito en otra parte no es igual al que no lo consiguió (otro sesgo de selección, un piso más abajo). Aun así: es la única familia que **añade observación** en vez de redistribuir supuestos, y por eso es el estándar de oro cuando el bureau lo permite.

## 4. Cómo se evalúa un ejercicio de reject inference

No hay ground truth, así que la evaluación es indirecta y de estabilidad:

1. **Sensibilidad:** correr el modelo sin RI, y con cada técnica/parámetro. Si los coeficientes y el ordenamiento apenas se mueven, el sesgo era menor y el modelo sin RI es defendible (con la limitación declarada). Si se mueven mucho, la zona de incertidumbre es real — y ninguna técnica la resuelve: la decisión pasa a ser de apetito (¿cuánto quiero crecer hacia lo desconocido?).
2. **Plausibilidad:** la tasa de malos inferida de los rechazados debe ser ≥ la de los aprobados del mismo tramo, y la tasa poblacional total reconstruida debe cuadrar con benchmarks (bureau, industria).
3. **Prueba del tiempo:** si el nuevo modelo aprueba zonas antes rechazadas, monitorear esas cosechas por separado (¡son el experimento!) y comparar contra lo inferido. Es la única validación real, y llega 12 meses tarde — razón de más para entrar a esas zonas gradualmente (cupos bajos, montos acotados).

## 5. Posición práctica (y la del curso)

Para un primer scorecard o un rediseño con tasa de aprobación alta, **declarar la limitación** (como hace el curso) es una posición metodológicamente seria: el modelo ordena bien DENTRO del universo tipo-aprobado, y la política mantiene prudencia fuera de él. Reject inference se vuelve importante cuando: la tasa de aprobación es baja, el objetivo del rediseño es justamente expandir aprobación, o el regulador/validador lo exige. Y cuando se hace, el orden de preferencia es: datos externos (bureau) > fuzzy/parceling con análisis de sensibilidad > augmentation, siempre reportando el modelo sin RI como referencia.

## 6. Para profundizar

- Siddiqi, *Intelligent Credit Scoring* (2ª ed.), cap. de reject inference — el tratamiento estándar de industria, con fuzzy y parceling paso a paso.
- Thomas, Edelman & Crook, *Credit Scoring and its Applications*, secciones de sample selection — el tratamiento más formal (conexión con Heckman).
- Hand & Henley (1993/1997) y Crook & Banasik, *Does reject inference really improve the performance of application scoring models?* — el escepticismo empírico documentado: con datos reales donde SÍ se observó a los rechazados, las técnicas puramente estadísticas mejoran poco o nada. Lectura obligatoria antes de prometerle milagros a un comité.

## 7. Preguntas de autoevaluación

1. ¿Por qué el desempeño de los rechazados es MNAR y no MAR? ¿Qué técnica asume MAR de todos modos?
2. Implementa mentalmente fuzzy augmentation: ¿qué filas nuevas aparecen en la matriz y con qué pesos? ¿Qué pasa con el cálculo del WoE con pesos fraccionales?
3. Tu banco aprueba el 85%. ¿Inviertes en reject inference? ¿Y con 45%? Justifica con el mecanismo del sesgo.
4. Diseña la cohorte de prueba (aprobación aleatoria bajo el cutoff): tamaño, límites de exposición, y el análisis a 12 meses.
5. ¿Por qué "la tasa inferida de los rechazados debe ser ≥ la de los aprobados del mismo tramo" es un chequeo de plausibilidad y no una verdad garantizada? Construye un contraejemplo (pista: overrides comerciales).

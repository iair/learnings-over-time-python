# V0 · Videoteca por módulo: videos de YouTube para reforzar la serie

**Serie: Modelador de Riesgo en Profundidad** · Complemento audiovisual de las Fases 1 y 2
Criterio de curaduría: solo videos verificados como existentes (búsqueda ago-2026), gratuitos, priorizando canales establecidos. La mayoría está en inglés — el material de scoring en español en YouTube es escaso y de menor calidad; esto además refuerza el vocabulario bilingüe de M0. Donde no existe buen video para el contenido específico del curso, se indica **[GUIÓN Gx]**: el guión correspondiente está en los archivos `G1`–`G5` para que lo transformes en video/podcast.

**Nota de mantenimiento:** los enlaces de YouTube mueren o cambian; si alguno falla, los títulos y canales indicados bastan para reencontrarlos con el buscador.

---

## Panorama general del pipeline (M0)

| Video | Qué refuerza |
|---|---|
| **Credit Risk Scorecard Models Masterclass — Part 1: Data Sources, EDA & Vintage Analysis** · youtube.com/watch?v=GpF7WqcWMRI | Recorrido de la metodología completa de scorecards (fuentes de datos, EDA, vintage, WoE, reject inference). Buen "mapa aéreo" alternativo al del curso; útil ver dónde coincide y dónde difiere en vocabulario. |
| **FRM: Basel II Overview** (Bionic Turtle) · youtube.com/watch?v=o2kGYUP7Vro | Contexto de dónde viven PD/LGD/EAD en el marco regulatorio; complementa el mapa de M0 §2 y adelanta E6. |

## M1 · Problema de negocio y economía del error

| Video | Qué refuerza |
|---|---|
| Masterclass Part 1 (ver M0) | La sección de objetivos de negocio y tipos de scorecard (admisión vs comportamiento vs cobranza). |
| **[GUIÓN G5]** | La economía del error específica del curso — PD de indiferencia, asimetría de costos, cutoff como política — no tiene video digno: el guión G5 la cubre. |

## M2 · Arquitectura temporal (t₀, ventanas, anclas)

| Video | Qué refuerza |
|---|---|
| **Data Leakage Explained Visually — How Models Cheat Without You Realizing** · youtube.com/watch?v=xzCXqYlV2HE | La intuición visual de por qué un modelo con fuga valida perfecto y muere en producción. |
| **Understanding the different kinds of Data Leakage in Machine Learning** · youtube.com/watch?v=2D9EfiEE06Y | Taxonomía general de fugas (target leakage, contaminación train-test, fuga temporal) — el paraguas del cual las 4 fugas del curso son casos. |
| **[GUIÓN G1]** | El ancla [t₀−k, t₀−1], el lag por fuente, la máquina del tiempo y la Trampa 3 (Δcupo IV 11) son contenido específico del curso sin video existente: guión G1. |

## M3 · Definición de default (roll rates, cura, maduración)

| Video | Qué refuerza |
|---|---|
| **Markov Chains Clearly Explained! Part 1** (Normalized Nerd) · youtube.com/watch?v=i3AkTO9HLXo — y la playlist completa: youtube.com/playlist?list=PLM8wYQRetTxBkdvBtz-gw8b9lcVkdXQKV | La maquinaria matemática de M3 §2.2: matrices de transición, estados, distribución estacionaria. Con esto, la matriz de roll rates se lee como lo que es: una cadena de Markov empírica. |
| **Prob & Stats — Markov Chains (serie de 38 videos, ilectureonline)** · youtube.com/watch?v=Uz3JIp6EvIg | Alternativa más pausada y completa, incluye estados absorbentes (la base de la "PD por banda" del notebook M3). |
| Masterclass Part 1 (ver M0) | Su tratamiento de vintage analysis y roll rate para definir el target: el mismo procedimiento del curso contado por otra voz. |

## M4 · Esquema muestral y las 4 fugas

| Video | Qué refuerza |
|---|---|
| **What is Data Leakage in Machine Learning?** (Krish Naik) · youtube.com/watch?v=n9jz7G68pVg | Fuga por contaminación del split y por preprocesamiento — la versión ML-general de las fugas tipo 2 y 4. |
| **Data Leakage Explained Visually** (ver M2) · youtube.com/watch?v=xzCXqYlV2HE | Repaso visual antes de correr el laboratorio `M4_laboratorio_fugas.py`. |
| **[GUIÓN G1]** | Las 4 fugas con nombres del curso (ex-post, ventana, población, identidad), el AUC 0.991 de `estado_actual` y la regla de un solo disparo de la OOT: guión G1. |

## M5 · Fábrica de variables + M6 · Calidad de datos

| Video | Qué refuerza |
|---|---|
| **[GUIÓN G2]** | La fábrica declarativa (familias × ventanas × agregadores), la recencia censurada → 13, el missing estructural del Δ12m y el MNAR de la renta son contenido específico sin video: guión G2. |
| Masterclass Part 1 (ver M0) | Su sección de fuentes de datos y EDA toca la lógica de familias de variables y calidad. |

## M7 · Binning, WoE e IV

| Video | Qué refuerza |
|---|---|
| **Weight of Evidence (WOE) and Information Value (IV) in Credit Scoring** · youtube.com/watch?v=98Zzr6PU19U | La mecánica de cálculo de WoE/IV con ejemplo numérico. **Ojo con la convención de signo** (M7 §3.2): verifica con un bin obvio antes de comparar con el curso. |
| **Weight of Evidence (WOE) & Information Value (IV) — Logistic Regression Scorecard** · youtube.com/watch?v=ntdiqgBFu4U | Fine/coarse classing y el uso del WoE como transformación para la logística (el puente a C3). |
| **[GUIÓN G3]** | Las tres trampas del IV (premia bins, inestable con pocos malos, IV altísimo = fuga) no aparecen en ningún video con ese nivel de detalle: guión G3. |

## E1 · Reject inference

| Video | Qué refuerza |
|---|---|
| **Rejected Inference in Credit Scoring** · youtube.com/watch?v=0rFed3LE6Rw | Qué es y cómo se aplica; introducción liviana. |
| **Credit Scoring & R: Reject inference, nested conditional models & joint scores** (charla técnica) · youtube.com/watch?v=9h8roMbiJEE | Tratamiento serio y escéptico, alineado con la posición de E1 §5. |
| **Active Learning for Reject Inference in Credit Scoring** (charla académica) · youtube.com/watch?v=sUM4B1YqkNk | La frontera de investigación: qué se intenta más allá de fuzzy/parceling. |
| **Reject inferencing in credit risk modeling** · youtube.com/watch?v=yXDNy-e-cdM | Repaso alternativo corto. |

## E2 · Calibración y tendencia central

| Video | Qué refuerza |
|---|---|
| **[GUIÓN G4]** | PIT vs TTC, tendencia central y el ajuste de intercepto en log-odds no tienen video accesible (la literatura es de papers técnicos): guión G4. Los videos de "model calibration / Platt scaling" de ML general sirven de precalentamiento pero no cubren el ancla de largo plazo. |

## E3 · Bootstrap e intervalos

| Video | Qué refuerza |
|---|---|
| **StatQuest: Confidence Intervals, Clearly Explained!!!** · youtube.com/watch?v=TqOeMYtOc1w | La derivación de intervalos VÍA bootstrap — exactamente el enfoque de E3. Corto y memorable. |
| **Bootstrap Confidence Interval with Examples** (MarinStatsLectures) · youtube.com/watch?v=-YgeLJRZQYY | La mecánica paso a paso del percentil bootstrap. |
| **Bootstrap confidence intervals — explained** (TileStats) · youtube.com/watch?v=AA7Jtuu9TaE | Incluye la pregunta "¿funcionan de verdad?" (cobertura), útil para el criterio de E3 §2.4. |

## E4 · Variables de bureau

| Video | Qué refuerza |
|---|---|
| **[GUIÓN incluido en G2, sección final]** | El rezago/backfill y el trade-off Gini/costo son específicos; los videos genéricos de "how credit bureaus work" (orientados a consumidor) no aportan al modelador. G2 cierra con la sección de bureau. |

## E5 · Scorecard vs ML

| Video | Qué refuerza |
|---|---|
| **SHAP values for beginners** · youtube.com/watch?v=MQ6fFDwjuco | La herramienta de explicabilidad post-hoc del lado ML; base para la discusión de reason codes de E5 §2. |
| **Shapley Values Explained — Interpretability for AI models** · youtube.com/watch?v=5-1lKFvV1i0 | La intuición de teoría de juegos detrás de SHAP. |
| WoE/IV videos de M7 | El contrapunto: la explicabilidad estructural (la tabla de puntos ES el modelo). Ver ambos lados es el ejercicio de E5. |

## E6 · Normativa

| Video | Qué refuerza |
|---|---|
| **Basel III in 10 minutes** · youtube.com/watch?v=KpWBf3s4NpI | El delta Basilea II→III en capital, comprimido. |
| **FRM: Basel II Overview** (Bionic Turtle) · youtube.com/watch?v=o2kGYUP7Vro | Los tres pilares y las rutas estándar/IRB — el esqueleto de E6 §2. |
| **Basel 3 Explained** · youtube.com/watch?v=M3PU-HPxgAg | Introducción de alto nivel alternativa. |
| **Basel III framework and its three pillars** · youtube.com/watch?v=0nlalO3ENAo | Repaso estructurado de pilares y ratios. |
| **IFRS 9 ECL Model Explained: Simple Walkthrough with Example** · youtube.com/watch?v=Rx_RzRcgvlk | El modelo de pérdida esperada con ejemplo numérico — E6 §3. |
| **IFRS 9 Impairment — Expected Credit Loss Model** (Farhat Lectures) · youtube.com/watch?v=x0OsNn-qm6c | Los stages y la lógica contable, estilo clase universitaria. |
| **Understanding IFRS 9 — Expected Credit Loss (ECL) Model** (AARO Academy) · youtube.com/watch?v=wwlqm0YYSHM | Versión ejecutiva corta. |

## E7 · Stress testing

| Video | Qué refuerza |
|---|---|
| **Stress Tests** (Reserve Bank of New Zealand) · youtube.com/watch?v=08OuDzDZZ5o | Qué es un stress test y para qué lo usa un supervisor — la vista regulatoria. |
| **What is stress testing?** (Bank of England) · youtube.com/watch?v=XtihtTHVXTE | Ídem, desde el BoE; los escenarios "what if". |
| **A brief explanation of stress testing under Basel rules with an Excel example** · youtube.com/watch?v=qWJjS5V0iTI | Un ejercicio numérico simple — el espíritu del stress mínimo de E7 §3.1. |
| **Introduction to Stress Testing (in Lending, Trading and Asset Management)** · youtube.com/watch?v=7SGLpWzRcDE | Panorama por tipo de negocio, incluye lending. |

## E8 · Bibliografía

Sin video: es el módulo de lectura. Como complemento audiovisual general, la playlist de Markov de Normalized Nerd (M3) y el canal de StatQuest (E3) son los dos canales que más rinden por minuto para este perfil.

---

## Orden de visionado sugerido (si quieres una "temporada" de estudio)

1. Masterclass Part 1 (mapa) → 2. Guión G5 (economía del error) → 3. Guión G1 (tiempo y fugas) + los 2 videos de leakage → 4. Markov Part 1 (roll rates con fundamento) → 5. Videos WoE/IV + Guión G3 (trampas) → 6. Guión G2 (fábrica y calidad) → 7. StatQuest de intervalos → 8. Videos de reject inference → 9. Guión G4 (calibración) → 10. Basilea/IFRS 9 → 11. Stress testing → 12. SHAP (para el debate ML).

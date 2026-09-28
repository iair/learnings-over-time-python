# BRIEFS POR MÓDULO — Serie 2

Cada brief dice: foco, contenidos mínimos obligatorios (puedes agregar), experimentos del notebook,
planilla y enlaces. Los contenidos mínimos son el piso, no el techo.

---
## M08 · Estabilidad poblacional: PSI, CSI y el orden del embudo  (`M08_estabilidad_psi_csi`)
Curso: clase 3 (PSI primero, 108→94, PSI no medible, variable categórica plantada en Andes, TTD vs
OOT) y clase 5 (PSI del score sobre las 8 bandas, CSI como canario, DEV congelado vs ventana móvil).
Mínimos:
- PSI como divergencia de Jeffreys = KL(a‖e)+KL(e‖a); derivación. Relación con χ²: para n grande y
  sin drift, PSI·(1/(1/n_e+1/n_a)) ≈ χ²_{B−1} → valor esperado del PSI sin drift ≈ (B−1)(1/n_e+1/n_a).
  Consecuencia: el umbral 0,10 significa cosas distintas con n=300 o n=30.000. Referencia:
  Yurdakul (2018) «Statistical properties of population stability index» (Western Michigan Univ.,
  tesis doctoral) — verificar; también Taplin & Hunt (2019) «The Population Accuracy Index» (Risks).
- Origen práctico de 0,10/0,25 (Siddiqi) = convención sin base inferencial.
- Sensibilidad del PSI a: número de bins, deciles de DEV vs bins del scorecard vs bandas de la master
  scale, el ε para ceros, masa de ceros (por qué el curso obtiene PSI NaN con moras cortas y cómo
  hacerlo medible: bins por masa + bin especial para la moda).
- Alternativas: KS de dos muestras, Jensen-Shannon, Wasserstein, test χ² de homogeneidad, Hellinger.
  Qué detecta cada una (cambio de ubicación vs forma vs colas).
- Taxonomía de cambio: covariate shift, prior shift, concept drift; por qué el PSI solo ve el primero.
- CSI vs PSI del score: cancelaciones (el caso deuda_interna_max_3m 0,138 con score 0,013);
  aporte direccional por bin.
- Umbrales ajustados por tamaño: crítico por percentil de la distribución nula (simulada o χ²).
- La regla del curso «sin PSI medible no sigue»: pros/contras, alternativa defendible.
Notebook: (1) psi numpy vs versión con scipy (entropy / chi2_contingency) coinciden; (2) simulador de
drift (shift de media, cambio de varianza, mezcla categórica) con sliders de magnitud, n y bins →
PSI, KS, JS, Wasserstein lado a lado; (3) distribución nula del PSI por simulación vs aproximación χ²
para varios n → tabla de «umbral 95% por n»; (4) caso canal (variable categórica que deriva en el
generador) y CSI de todas las variables DEV→OOT/TTD; (5) cancelación: dos variables que se mueven en
sentido opuesto y PSI del score casi 0.
Planilla: `M08_calculadora_psi.xlsx` — ingresas % esperado y % actual por bin (10 bins) y n de cada
muestra; calcula aporte por bin, PSI, KL en ambos sentidos, estadístico χ² equivalente, p-valor
(CHISQ.DIST.RT) y semáforo.

---
## M09 · Redundancia, colinealidad y familias de variables  (`M09_redundancia_colinealidad`)
Curso: clase 3 (corr WoE > 0,70 greedy por IV 63→26, VIF máx 2,97, clusterización conceptual 8–14
variables) y clase 4 v21 (familias: mora_max_3m/6m/12m sobreviven a la correlación y cuentan la
misma historia; redundancia condicional; reemplazo entre ventanas; cobertura de drivers).
Mínimos:
- Por qué correlación sobre WoE (y no crudos): Pearson sobre WoE ≈ relación en escala log-odds;
  Spearman como alternativa; correlación entre categóricas (V de Cramér) y mixto.
- El greedy: dependencia del orden y del umbral (0,6/0,7/0,8/0,9) → cuántas variables quedan y qué
  Gini final; no es óptimo (es un problema de conjunto independiente máximo ponderado).
- VIF: derivación VIF_j = 1/(1−R²_j) = [R⁻¹]_jj (diagonal de la inversa de la matriz de
  correlación). Índice de condición y descomposición de proporciones de varianza (Belsley, Kuh &
  Welsch 1980): detectan colinealidades de 3+ variables que el VIF por variable diluye. VIF
  generalizado (Fox & Monette 1992) para grupos. VIF en logística: versión ponderada por W de IRLS.
- Efecto de supresión: cómo dos variables correlacionadas producen un coeficiente de signo
  «equivocado» y muy significativo (el aviso del Lab 2). Derivar en el caso de 2 regresores.
- VARCLUS / clustering jerárquico de variables sobre 1−|ρ|, ratio 1−R²; representante por cluster.
- Familias temporales (misma serie, distintas ventanas): nivel vs tendencia; construir
  «nivel + delta» en vez de 3 ventanas; redundancia condicional (corr parcial).
- Alternativas: L1/elastic net sobre WoE; PCA (por qué no se usa en scorecards).
Notebook: numpy (VIF por inversa, condición, Belsley) vs statsmodels variance_inflation_factor;
experimento de supresión con slider de correlación; greedy con sliders de umbral/orden → nº variables
y Gini HO; dendrograma de clusters (scipy.cluster.hierarchy) + representante por cluster; familia
uso_tc 3m/12m: nivel+delta vs ambas.
Sin planilla obligatoria (opcional CSV de la matriz de correlación).

---
## M10 · La regresión logística sobre WoE, desde la verosimilitud  (`M10_logistica_woe`)
Curso: clase 3 (logit, β, p-valor, «no es un curso de econometría: investíguenlo»; ajustar sobre
WoE resuelve no linealidad, escalas, missing; intercepto absorbe tasa de DEV; HO 78 malos / 9
parámetros = 8,7 EPV «contrasta, no re-estima»).
Mínimos:
- Verosimilitud Bernoulli, log-verosimilitud, gradiente Xᵀ(y−p), Hessiano −XᵀWX; Newton-Raphson =
  IRLS; convergencia; errores estándar de la inversa de la información de Fisher.
- Ecuación de primer orden con intercepto ⇒ Σp̂ = Σy (por qué DEV «clava por construcción»).
- Tests: Wald, razón de verosimilitud (LR), score; por qué pueden discrepar (Hauck-Donner).
- **Resultado clave**: con UNA variable en WoE (convención ln(%B/%M)), el MLE da β = −1 y
  β₀ = ln(M/B) (log-odds de DEV) exactamente (sin el suavizado +0,5). Derivación: el WoE es el
  log-likelihood ratio del bin; modelo saturado por bins. Con suavizado, ≈ −1. En multivariado,
  β_j ≠ −1 mide cuánto hay que «descontar» la evidencia por redundancia ⇒ conexión con Naive Bayes
  (β = −1 para todas = Naive Bayes). Interpretación: |β|<1 redundancia, |β|>1 sinergia/supresión.
- Grados de libertad escondidos por el binning (los bins se eligieron mirando y) ⇒ p-valores y SE
  optimistas; experimento.
- Separación completa y cuasi-completa; Firth (penalización de Jeffreys) como remedio.
- EPV (eventos por variable): Peduzzi et al. (1996) regla 10; críticas (van Smeden et al. 2016/2019).
- WoE vs dummies vs crudo + splines: grados de libertad, monotonía, regularización.
- Regularización L2/L1 en scorecards; class imbalance y pesos (por qué no se balancea en riesgo o,
  si se hace, cómo corregir el intercepto: King & Zeng 2001).
Notebook: IRLS en numpy vs statsmodels Logit vs sklearn LogisticRegression(penalty=None) — mismos β
y SE; demostración β=−1 (con y sin suavizado); experimento Naive Bayes vs logística multivariada
(Gini y calibración); experimento de p-valores optimistas (variable de ruido binneada en la misma
muestra → tasa de rechazo real vs 5% nominal, con slider de nº de bins); separación con Firth
(implementar Firth en numpy; statsmodels no lo trae: comparar con resultado analítico o dejar solo
numpy explicándolo); EPV: variabilidad de β al re-estimar en muestras chicas.

---
## M11 · Selección de variables: stepwise y sus alternativas  (`M11_seleccion_variables`)
Curso: clase 3 y Lab 2 (stepwise forward con revisión backward, p<0,05, tope 14, política de signos:
el signo manda sobre la significancia; forzar variables pedidas por comité y medir costo; bitácora).
Mínimos:
- Variantes: forward, backward, bidireccional; criterios p-valor (Wald vs LR), AIC, BIC, y aporte a
  Gini/HO; por qué el curso usa log-verosimilitud + p.
- Problemas: inferencia post-selección (p-valores y β sesgados hacia afuera), paradoja de Freedman
  (1983): stepwise encuentra «significativas» en ruido puro; inestabilidad de la selección.
- Bootstrap del stepwise: frecuencia de inclusión por variable (Austin & Tu 2004); stability selection
  (Meinshausen & Bühlmann 2010).
- LASSO / elastic net sobre WoE con restricción de signo (coeficientes ≤ 0) como alternativa;
  camino de regularización.
- Política de signos formalizada; forzar variables: costo en Gini y en log-verosimilitud con LR test.
- Criterios de negocio: cobertura de drivers (conducta, endeudamiento, capacidad, recencia),
  clusterización conceptual, parsimonia 8–14 (gobierno, no estadística).
- Bitácora de decisiones como artefacto versionado (qué entró, por qué, en qué ronda).
Notebook: stepwise genérico parametrizable (criterio, umbral, tope) en numpy/IRLS propio + versión con
statsmodels; Freedman: 50 variables de ruido puro → cuántas «significativas»; mapa de frecuencia de
inclusión por bootstrap (slider B); LASSO con sklearn (liblinear/saga) y camino; comparación Gini HO
de los enfoques; experimento «forzar variable» y su costo.

---
## M12 · Poder discriminante: ROC, AUC, Gini, KS, CAP y su incertidumbre  (`M12_discriminacion`)
Curso: clase 3 (Gini por muestra, rangos típicos 0,35–0,55 admisión, >0,85 sospechoso, reglas de
caída 0,10/0,15, tabla de rendimiento por deciles) y clase 5 (AUC como probabilidad, KS 0,545 en
560, bootstrap B=1000 IC del Gini y de la caída, deciles/lift, caída relativa 20%/30%).
E3 de Serie 1 ya cubre bootstrap en general: aquí aplícalo a métricas de ranking y profundiza lo nuevo.
Mínimos:
- AUC = P(score_bueno > score_malo) = U de Mann-Whitney / (n_B·n_M); manejo de empates (½);
  Gini = 2AUC−1 = accuracy ratio de la curva CAP = D de Somers. Demostrar AR = Gini.
- KS: definición, relación con AUC (cotas), dónde se alcanza y por qué cerca del cutoff.
- Gini máximo alcanzable dada la PD verdadera (con el generador de verdad conocida: el Gini de la
  pd_verdadera es el techo); Gini «perfecto» imposible con eventos aleatorios.
- Varianza del AUC: Hanley & McNeil (1982), DeLong et al. (1988) (implementar DeLong en numpy);
  bootstrap percentil; comparación de dos AUC correlacionados (DeLong pareado) — la pregunta correcta
  para «¿el modelo nuevo es mejor?».
- Ruido de la caída DEV→HO dado nº de malos: ¿cuánto cae por azar? Tabla por nº de malos. Crítica a
  las reglas fijas 0,10/0,15 y a la caída relativa.
- Optimismo DEV→HO (selección) vs deterioro HO→OOT (tiempo); por qué se leen distinto.
- Tabla de deciles/ganancia, lift, curva de captura; métricas de negocio vs AUC.
- Relación aproximada IV ↔ Gini de una variable (para binormal) — mencionar con cautela.
- Por qué el Gini no cambia con δ (invarianza a transformaciones monótonas).
Notebook: AUC por conteo O(n²) vs rank (Mann-Whitney) vs sklearn roc_auc_score vs scipy mannwhitneyu;
KS numpy vs scipy ks_2samp; DeLong numpy (IC y test pareado) vs bootstrap; curvas ROC/CAP/KS;
simulador de «caída por azar» con slider de nº de malos; tabla de deciles.
Planilla: `M12_tabla_deciles_gini.xlsx` — pegas n y malos por decil (ordenado de peor a mejor) y
calcula tasa, captura acumulada, lift, KS por decil y Gini aproximado por trapecios (fórmulas).

---
## M13 · Del logit al scorecard: scaling y tabla de puntos  (`M13_scaling_scorecard`)
Curso: clase 3/4 (factor 28,85, offset 487,12, base 71,8 = offset/8 − β₀/8·factor, pendiente −β·factor,
uso_tc_prom_12m 10,2 pts por unidad de WoE, 41 tramos, S0030427 = 603,8, 604 ⇔ PD 1,7%, redondeo a
enteros en producción).
Mínimos:
- Derivación completa: score = offset + factor·ln(odds_buenos) con odds = (1−p)/p; por qué el signo;
  factor = PDO/ln2; offset; lectura 580/600/620.
- Descomposición del score en suma de puntos por variable; reparto del intercepto: partes iguales
  (curso), proporcional al rango, «puntos neutros» (WoE=0 → puntos fijos) — efecto en la lectura de
  cada tabla y en los reason codes (M14).
- Escalas alternativas: PDO 20/40/50, bases distintas, scores crecientes con el riesgo; conversión
  entre escalas; relación score ↔ PD (tabla).
- Redondeo a enteros: error máximo = n/2 puntos; efecto en PD y en decisiones cerca del cutoff
  (cuántos cambian de decisión); redondear puntos vs redondear score.
- Puntos negativos: cuándo aparecen y por qué no son un error; convención de desplazar.
- Recalibración y scaling: si δ se suma al logit, el score calibrado = score − δ·factor (todos se
  mueven igual) — ¿se re-escala la tabla o se mueve el cutoff? pros/contras de cada uno.
- Scorecard como artefacto de datos (tabla variable×bin×puntos) — preparación para M21.
Notebook: scorecard completo sobre el generador (pipeline corto: WoE → logística numpy → puntos), con
sliders PDO/score base/odds base/reparto del intercepto; verificación score = suma de puntos =
transformación del logit (assert); experimento de redondeo (decisiones que cambian en un cutoff);
comparación con optbinning Scorecard (opcional si corre; si optbinning da problemas, usar statsmodels
para β y dejar optbinning en el .md).
Planilla: `M13_scorecard_vivo.xlsx` — parámetros PDO, score base, odds base, β₀, n variables; tabla
de 3 variables × 5 bins con WoE y β; calcula factor, offset, puntos por bin, y un «simulador de
cliente» (eliges bin de cada variable con número de bin → score, odds, PD). Hoja score↔PD.

---
## M14 · Interpretabilidad: contribuciones, monotonía, binning óptimo y reason codes  (`M14_interpretabilidad_reason_codes`)
Curso: clase 3 (rango de puntos vs coeficiente; aportes |β|·σ(WoE) y |β|·IV; estabilidad de
coeficientes en HO con inversiones p 0,96/0,78 indistinguibles de cero; quiebres de monotonía en
meses_desde_mora y deuda_interna; reason codes por brecha contra el máximo; S0027717;
optbinning: 3 experimentos 0,757/0,698/0,675 vs 0,766/0,683/0,684 vs 66 vars 0,805/0,657/0,677;
¿puede la edad ser reason code?), clase 4 v21 (granularidad: min_event_rate_diff, max_n_bins; casos
1 y 2; missing −9/−99 mezclados en un 4,1% que esconde 2,5% y 17%; cobertura; «un Gini alto no
cierra la revisión»).
Mínimos:
- Medidas de importancia: rango de puntos, |β|·σ(WoE), |β|·IV, caída de log-verosimilitud / Gini al
  sacar la variable (drop-column), contribución tipo Shapley exacta para modelos aditivos (en un
  modelo aditivo en WoE, el SHAP de cada variable = puntos − E[puntos]) — derivarlo.
- Estabilidad de coeficientes DEV vs HO: test de diferencia (z con SE combinados), qué es una
  inversión significativa, potencia con pocos malos.
- Monotonía: por qué importa (explicabilidad, reason codes coherentes, robustez); diagnóstico;
  fusión de bins; binning monótono (Mironchyk & Tchistiakov 2017), binning óptimo por programación
  entera (Navas-Palencia 2020, optbinning): restricciones de tamaño mínimo, min_event_rate_diff,
  max_n_bins, monotonic_trend (ascending/descending/peak/valley/auto). Trade-off granularidad.
- Missing y valores especiales: separar códigos por mecanismo; demostrar el sesgo de agrupar
  (−9 vs −99 en el generador: el binner del curso los mezcla en un bin «(-inf, -9]»).
- Reason codes: métodos (brecha contra el máximo = curso; contra el promedio poblacional; contra
  puntos neutros/WoE=0), empates, mínimo de brecha para reportar, variables no permitidas o no
  accionables (edad, sexo), mapeo variable→frase, estabilidad de los códigos.
- Regulación: EE.UU. ECOA/Regulation B (adverse action notice, «principal reasons», típicamente
  hasta 4) y FCRA; UE (AI Act: scoring crediticio de alto riesgo; GDPR art. 22); Chile: Ley 19.628 y
  la nueva Ley 21.719 de protección de datos personales (decisiones automatizadas) — VERIFICA con
  WebSearch fechas y contenido antes de afirmar; si no puedes, formula con cautela.
- Equidad: edad como variable/reason code; proxies; disparate impact (mención, sin sermonear).
Notebook: medidas de importancia numpy vs statsmodels; diagnóstico de monotonía; binning monótono en
numpy (algoritmo de fusión de bins adyacentes tipo PAV/pool-adjacent-violators) vs optbinning
OptimalBinning(monotonic_trend=...) con sliders de max_n_bins/min_event_rate_diff/min_bin_size →
Gini HO y forma; experimento −9/−99; reason codes por los 3 métodos para 3 clientes (y cuánto
cambian); SHAP aditivo = puntos − media (assert).
Si optbinning falla al importar en el entorno, envuélvelo en try/except y deja la versión numpy
funcional (el check no debe depender de optbinning).

---
## M15 · Calibración: nivel vs ranking  (`M15_calibracion`)
Curso: clase 4 v1 (TC 5,42% de 12 cosechas, δ Siddiqi 0,092 vs exacto 0,111, Jensen, curva de
calibración 10 grupos) y v21 (**PIT**: muestra reciente y madura mar–jun 2025, 5,19%→5,94%, δ aprox
0,143 vs exacto 0,177, «OOT usado para calibrar no valida»), clase 5 (TC vs PIT: la muestra con que
se calibra y la que valida no pueden ser la misma; con TC el binomial OOT da 0,56; con PIT daría 1,00
por construcción; decisión casi igual: 78,3% vs 77,2% en 560). Serie 1 · E2 ya cubrió TC, PIT/TTC
conceptual: aquí la MECÁNICA y las métricas.
Mínimos:
- Por qué el intercepto de MLE clava la media en DEV (primer orden).
- Ajuste de intercepto: aprox Siddiqi δ ≈ logit(TC) − logit(PD̄) vs exacto (raíz de
  mean σ(logit p_i + δ) = TC); acotar el error de Jensen (desarrollo de Taylor de segundo orden:
  depende de la dispersión de las PD); cuándo importa.
- Corrección por sobremuestreo / prior shift: δ = ln[(π₁/π₀)·(ρ₀/ρ₁)] (King & Zeng 2001) —
  derivarla desde Bayes; equivalencia con el ajuste de intercepto.
- Recalibración logística (intercepto + pendiente: logit p* = a + b·logit p), «calibration slope»;
  Platt; isotónica (PAV) y por qué introduce empates y rompe la escala PDO; recalibración por bandas.
- Jerarquía de calibración (Van Calster et al. 2016/2019): media, débil (intercepto+pendiente),
  moderada (curva), fuerte.
- Métricas: calibration-in-the-large, razón O/E, pendiente, Brier y su descomposición de Murphy
  (confiabilidad − resolución + incertidumbre), log-loss, ECE; curvas con IC binomiales (Wilson,
  Jeffreys) — el grupo 5 de OOT (1,18% pred vs 4,00% obs, n=200) ¿es significativo?
- PIT vs TTC operativo: qué muestra calibra, qué muestra valida, cómo documentarlo; doble calibración
  (δ_PIT para provisiones, δ_TTC para capital) con el mismo ranking.
- Efecto de δ sobre el score: desplazamiento δ·factor puntos (enlace M13) y sobre el cutoff.
Notebook: δ aprox vs exacto (numpy bisección propia vs scipy.optimize.brentq) con slider de
dispersión de las PD → error de Jensen; prior shift por sobremuestreo con verdad conocida
(pd_verdadera del generador); recalibración intercepto+pendiente (statsmodels GLM con offset vs IRLS
numpy); isotónica numpy (PAV) vs sklearn IsotonicRegression; Brier/Murphy numpy; curvas de
calibración con IC de Wilson; experimento «calibrar y validar en la misma muestra» (p=1 por
construcción) vs muestras separadas.
Planilla: `M15_calibracion_intercepto.xlsx` — pegas PD por grupo (10 grupos: PD media y n) y la
TC objetivo; calcula δ aproximado (fórmula), δ exacto aproximado por tabla de búsqueda (columna de δ
candidatos con SUMPRODUCT) , PD calibrada por grupo, O/E si hay observados.

---
## M16 · Master scale  (`M16_master_scale`)
Curso: clase 4/5 (8 bandas A1…E ancladas al PDO: cada 20 pts duplica odds; 600 = 50:1 = PD 1,96%;
tabla Austral con n, PD cal. media, tasa observada, % modelación, % TTD; requisitos monótona, sin
bandas vacías ni concentradas, estable; 7–10 bandas corporativas).
Mínimos:
- Diseños: por PDO (geométrica en odds), por PD objetivo (límites de PD geométricos tipo escala de
  rating), por cuantiles de población, por optimización (minimizar pérdida de información / maximizar
  separación). Escalas de agencias y escalas internas bancarias (ej. 20+ grados en IRB): mención.
- Asignación de PD a la banda: media de PD calibradas, PD en el punto medio geométrico, tasa observada
  suavizada; por qué la media aritmética y la geométrica difieren; impacto en provisiones.
- Requisitos cuantitativos: monotonía (test), concentración (Herfindahl-Hirschman por banda; umbral
  usado en validación IRB), mínimo de n y de malos por banda para backtesting con potencia,
  separabilidad entre bandas adyacentes (test de proporciones / binomial).
- Granularidad vs estabilidad: más bandas = más precisión y más migración/ruido.
- Matriz de migración entre bandas (behavioral): estabilidad del rating; mención.
- Mapear varios modelos a la escala corporativa; qué pasa al recalibrar (las bandas se mueven δ·factor).
Notebook: construir las 4 variantes de escala sobre el generador (numpy), métricas de cada una (HHI,
monotonía, n y malos por banda, separabilidad con tests de proporciones: numpy vs statsmodels
proportions_ztest), slider de nº de bandas y de PDO; potencia del backtest por banda según n.
Planilla: `M16_master_scale.xlsx` — parámetros (PDO, score ancla, odds ancla, ancho de banda en pts,
nº bandas); calcula límites de score, odds y PD de cada banda, PD punto medio geométrico; hoja de
validación donde pegas n y malos por banda y calcula tasa, % población, HHI, check de monotonía.

---
## M17 · Estrategia: cutoff, pérdida esperada y rentabilidad  (`M17_estrategia_cutoff`)
Curso: clase 4/5 (EL = PD×LGD×EAD, LGD 45%, EAD = monto solicitado; tabla de estrategia OOT
500…640; apetito 2,5% mora + aprobación ≥ 75% ⇒ 560; frontera de estrategia; política de referencia
knock-outs 90,2% / 4,48%; cutoff por segmento, pricing por riesgo, montos/plazos; «overlay
silencioso»). El usuario pidió incluir un **caso de financiamiento de motos** con parámetros genéricos.
Mínimos:
- EL con supuestos explícitos; EAD amortizante vs monto; LGD con garantía: LGD = 1 − (valor de
  recupero neto de costos, descontado)/EAD; curva de depreciación de la moto y tiempo a recupero;
  la LGD depende del momento del default (default temprano vs tardío).
- Cutoff por breakeven: aprobar si margen esperado > pérdida esperada ⇒ odds de breakeven
  = pérdida_si_malo / ganancia_si_bueno ⇒ score de breakeven = offset + factor·ln(odds_BE)
  (relación con PDO). Derivar. Con costo de fondos, costo operativo y costo de adquisición.
- Rentabilidad: margen financiero, NPV simple por crédito, RAROC (mención: capital económico con
  fórmula IRB de Basilea como referencia, sin profundizar — ver Serie 1 · E6).
- Frontera eficiente aprobación vs mora (y vs EL, vs utilidad); dominancia; por qué se construye
  sobre OOT.
- Cutoff por segmento: condición de optimalidad (igualar el margen en el corte entre segmentos);
  costo de gobernarlo.
- Pricing por riesgo: tasa que iguala EL + costos + margen objetivo por banda; límites legales de
  tasa (Chile: tasa máxima convencional — verificar con WebSearch cómo se fija; mencionar sin cifra
  si no se verifica).
- Asignación de monto/plazo: EL como función del monto; acotar exposición en bandas D.
- Sensibilidad del cutoff óptimo a LGD, TC (δ) y margen; tornado.
- Caso motos: parámetros genéricos (ej. monto 3–6 MM CLP, plazo 24–48 meses, pie 10–30%, tasa,
  depreciación anual 15–25%, costo de recupero/remate 10–20%, tiempo a recupero 4–8 meses) —
  presentados como ilustrativos, con rangos; calcular LGD por mes de default y cutoff de breakeven.
Notebook: tabla de estrategia sobre el generador (numpy) con sliders LGD/margen/costo de fondos/TC;
frontera eficiente y punto óptimo de utilidad; cutoff de breakeven analítico vs numérico (scipy
optimize) — deben coincidir; caso motos: curva de LGD por mes de default y EL por banda; tornado de
sensibilidad.
Planilla: `M17_tabla_estrategia_motos.xlsx` — EL PRINCIPAL DE LA SERIE. Hoja Parámetros (LGD o
parámetros de moto: precio, pie, tasa, plazo, depreciación, costo recupero, meses a recupero; costo
de fondos; costo operativo; PDO/offset/factor); hoja Bandas (8 bandas con score, PD calibrada, % de
solicitudes, monto medio — valores de ejemplo editables); hoja Estrategia (para cada cutoff: aprobación,
PD media aprobada, EL, margen, utilidad esperada, EL/monto — todo con fórmulas); hoja LGD_motos
(cronograma de saldo, valor de la moto y LGD por mes de default); hoja Breakeven (odds y score de
breakeven).

---
## M18 · Swap-set y el problema contrafactual  (`M18_swap_set`)
Curso: clase 4/5 (2×2 a igual aprobación 90,2% en OOT: ambas aprueban 1.680·3,0%, swap-out
128·23,4%, swap-in 128·9,4%, ambas rechazan 68·38,2%; cartera 4,48%→3,48%; knock-outs = mora interna
vigente o ≥30d en sistema, rechazan 9,8%; reject inference como confesión; cohorte swap-in 9,4% vs
PD 5,1% p 0,04 en el tablero). Serie 1 · E1 cubre reject inference: enlazar, no repetir.
Mínimos:
- Formalización: descomposición de la tasa de malos de la cartera nueva vs vieja en términos de
  swap-in/swap-out; condición de mejora; iso-aprobación vs iso-riesgo (a igual mora, cuánta más
  aprobación) vs iso-EL.
- **La trampa sutil**: en el curso el swap-in tiene desempeño porque la política vieja es hipotética
  (los knock-outs se aplican a posteriori sobre aprobados). En la vida real el swap-in de un cambio de
  política son rechazados históricos SIN desempeño ⇒ su tasa se estima (reject inference) o se
  aprende (experimento). Cuantificar el sesgo con el generador (truncamiento por la política vieja).
- Cómo aprender el swap-in sin sesgo: champion/challenger, bandas de exploración aleatorizadas
  (approve a random small % below cutoff), costo de la exploración vs valor de la información;
  cohortes de monitoreo del swap-in (umbral + fecha, como el tablero de clase 5).
- IC para las tasas de cada celda (n chicos: 128 casos): binomial exacto; ¿es significativa la
  diferencia 9,4% vs 23,4%?
- Swap-set con segmentos y con montos (EL en vez de conteo).
Notebook: política vieja (knock-outs) vs scorecard sobre el generador; matriz 2×2 a iso-aprobación
con slider de aprobación; iso-riesgo; IC binomiales (numpy/scipy.stats.binomtest vs statsmodels
proportion_confint); experimento del sesgo: entrenar solo con aprobados por la política vieja y
evaluar el swap-in «real» (con pd_verdadera) vs el estimado; simulación de exploración aleatoria:
cuánto cuesta (EL) y cuánto reduce el error de estimación del swap-in.

---
## M19 · Backtesting de calibración: binomial, Hosmer-Lemeshow y más  (`M19_backtesting_calibracion`)
Curso: clase 5 (binomial por banda: B2 1,42%, 235, 8 malos, p 0,020 una cola «8 o más»… la tabla usa
bilateral exacto; semáforo 0,05/0,01; global p 0,56; HL: 10 grupos, χ² 29,0 con dos deciles
aportando 24,3; p χ² 0,0003 vs p simulado 0,012 — «un rojo que depende de una aproximación inválida
no es un rojo»; nikodym: HL OOT falla p<0,001; con 9 indicadores al 5%, 34% de ver ≥1 amarillo).
Mínimos:
- Test binomial exacto: una cola vs dos colas (métodos de p bilateral: doble cola mínima, método de
  probabilidades ≤ — el de scipy binomtest); aproximación normal y cuándo falla (np < 5);
  potencia: con n y PD de cada banda, qué desvío se puede detectar (tabla).
- Correlación de defaults: el binomial asume independencia; con factor común (Vasicek, ρ de activos)
  la varianza de la tasa de default se infla ⇒ el binomial rechaza de más en años malos. Test
  binomial ajustado por correlación (BCBS WP14 2005 «Studies on the Validation of Internal Rating
  Systems»; Tasche). Implementar vía simulación del modelo de un factor.
- Hosmer-Lemeshow: construcción, por qué χ²_{g−2} en desarrollo y χ²_g en validación externa;
  dependencia del agrupamiento (deciles vs bandas) y del nº de grupos; por qué la χ² es mala con
  esperados < 5 (celdas con 0,1 esperados); p-valor por simulación (parametric bootstrap bajo H0).
- Otros tests: Spiegelhalter (1986) z; test de Jeffreys (usado por el BCE en validación de PD: la
  PD está dentro del IC bayesiano Beta(D+½, n−D+½)); traffic light de Basilea para PD (Tasche 2003,
  «A traffic lights approach to PD validation») (verificar); test de Brier; calibration slope test.
- Multiplicidad: 8 bandas × tests ⇒ FWER; Bonferroni/Holm vs lectura por patrón (el curso: «el
  diagnóstico está en el patrón»); signos: rachas de subestimación en bandas buenas (test de signos).
- Validación con la muestra NO usada para calibrar (enlazar M15).
Notebook: binomial numpy (desde la pmf con scipy.special.gammaln o log-comb en numpy) vs
scipy.stats.binomtest (una y dos colas); HL numpy con p χ² y p simulado vs statsmodels? (no lo trae;
comparar contra implementación alternativa) ; Spiegelhalter; Jeffreys (scipy.stats.beta); simulador:
cartera perfectamente calibrada con correlación ρ (slider) → tasa de rechazo del binomial vs 5%
nominal; potencia por banda (slider de desvío); Holm vs Bonferroni sobre 8 bandas; reproducir la
tabla del curso de Austral (B2: 235, 1,42%, 8 → p) y el HL 29,0 con los números de la lámina.
Planilla: `M19_backtesting_bandas.xlsx` — pegas n, PD y malos por banda (8) → esperados, tasa, p
binomial una cola (1−BINOM.DIST(k−1,n,p,TRUE)), p dos colas aproximado (normal), IC de Jeffreys
(BETA.INV), semáforo; HL con CHISQ.DIST.RT.

---
## M20 · Monitoreo: tablero, semáforos, gatillos y diagnóstico  (`M20_monitoreo_tablero`)
Curso: clase 5 (tablero Austral 9 indicadores con valor, umbral, estado, frecuencia; 5 partes por
indicador: definición, umbral, frecuencia, responsable, acción; umbrales ANTES de mirar; diagnóstico
por patrón: amarillos de NIVEL y POBLACIÓN con ORDEN verde ⇒ recalibrar, no re-desarrollar;
jerarquía vigilancia→recalibración δ→re-desarrollo→contingencia; «un tablero todo verde el día 1 es
mala noticia»; cohorte swap-in con fecha ago-2027; mix D+E +3,6 pts) y clase 6 (gatillos con 5 partes:
condición, valor hoy, quién decide, acción, plazo).
Mínimos:
- Taxonomía de indicadores: ranking (Gini, KS), población (PSI score, CSI, mix por banda),
  nivel/forma (binomial, HL, O/E), negocio (aprobación, overrides, mora temprana), datos (contrato,
  missing, fuera de rango), cohortes (vintage curves, swap-in).
- Indicadores ADELANTADOS vs rezagados: el target maduro llega 12 meses tarde ⇒ proxies: mora temprana
  (30+ a 3/6 meses, «first payment default»), roll rates, curvas de cosecha (vintage) comparadas contra
  la curva esperada; cuánto adelantan y cuánto se equivocan.
- Diseño de umbrales: por convención vs por distribución nula (M08, M12, M19) vs por costo de
  decisión; falsa alarma vs detección tardía (curva operativa); multiplicidad del tablero (1−0,95^k).
- Monitoreo secuencial: gráficos de control (Shewhart, EWMA, CUSUM) aplicados a tasa de mora temprana
  o PSI mensual; ARL (average run length) — cuánto demora en detectar un deterioro de tamaño d.
- Diagnóstico por patrón: matriz síntoma→causa (orden roto vs nivel corrido vs población movida vs
  datos rotos) y acción proporcional; árbol de decisión.
- Gatillos bien escritos (5 partes) + RACI (enlace M22); overrides y su monitoreo (override rate,
  desempeño de overrides).
Notebook: simular 24 meses de producción con el generador donde a partir del mes k aparece un
deterioro (slider de mes, tipo: nivel / población / ranking / dato roto) → tablero mensual con
semáforos (tabla coloreada) → qué indicador se enciende primero; EWMA/CUSUM numpy vs statsmodels?
(no trae CUSUM de proporciones: comparar CUSUM numpy con implementación alternativa o con cálculo
analítico de ARL por simulación); curvas de cosecha con banda esperada; mora temprana como proxy
(correlación con el 90+ a 12m).
Planilla: `M20_tablero_monitoreo.xlsx` — plantilla de tablero: por indicador, definición, umbral
amarillo/rojo, dirección, frecuencia, responsable, acción; columnas de 12 meses donde se pegan
valores y el estado se calcula con fórmulas (IF) → semáforo textual; hoja de gatillos (5 partes).
Además CSV `M20_tablero_plantilla.csv`.

---
## M21 · Implementación: artefacto congelado, contrato de datos y paridad  (`M21_implementacion_artefacto`)
Curso: clase 6 y notebook demo_c6_bases (en producción NO existe DEV; congelar 6 cosas: variables,
cortes, WoE, β, calibración, escala; JSON estándar allow_nan=False, no pickle; motor `puntuar` que
no toca DEV y devuelve índice + score; bins no vistos → marcados para revisión; prueba de paridad
idéntica; bug A silencioso (re-binning del lote + mapeo por posición: 6,8% decisiones cambian, cero
alarmas) vs bug B ruidoso (25,4%, WoE neutro, se delata); contrato de datos bloquea/avisa según
magnitud; 3,1% de antiguedad fuera de rango en TTD; 3 desastres que no lanzan excepción:
×1000, feed a la mitad, NaN).
Mínimos:
- Anatomía del artefacto: esquema JSON (versión, hash, variables, cortes con bordes −inf/inf
  representados de forma estándar, tratamiento de missing/especiales, WoE/puntos, β, δ, factor/offset,
  master scale, cutoff, reason code map); por qué JSON y no pickle/joblib; JSON Schema para validarlo;
  PMML/ONNX como estándares (mención con pros/contras).
- Semántica de bordes (intervalos abiertos/cerrados: (a,b] del pd.cut) — fuente clásica de bugs de
  paridad; valores exactamente en el corte.
- Motor de scoring puro: función sin estado, determinista, vectorizada; manejo de categorías nuevas,
  NaN, especiales; idempotencia; el índice viaja con el score.
- Pruebas: paridad bit a bit (o tolerancia declarada), tests de propiedades (monotonía del score en
  una variable con β<0 y WoE monótono; invariancia a permutar filas; score de un lote = score fila a
  fila — el test que mata al bug A), golden files, tests de regresión por versión.
- Contrato de datos: esquema (tipos, nulos), rangos (derivados de DEV con margen), reglas de negocio,
  severidad por magnitud (% de filas afectadas), qué bloquea y qué avisa; herramientas (pandera,
  Great Expectations, pydantic — mención).
- Versionado semántico del modelo (cambio de δ = minor; nuevas variables = major), despliegue
  champion/challenger y shadow mode; rollback.
Notebook: construir el artefacto JSON desde un scorecard del generador; motor `puntuar` numpy puro;
reproducir bug A y bug B (porcentaje de decisiones que cambian) con slider de «drift del lote»;
test de invariancia lote vs fila que detecta bug A; contrato de datos en numpy/pandas puro vs
versión con pydantic o pandera si están disponibles (si no, solo numpy/pandas + jsonschema-like
propio; no agregues dependencias pesadas que no estén instaladas: puedes usar `pydantic` que suele
venir con marimo — verifícalo); los 3 desastres silenciosos y cómo el contrato los bloquea;
paridad artefacto vs notebook (assert idéntico).

---
## M22 · Gobierno de modelos: expediente, trazabilidad, model card y validación independiente  (`M22_gobierno_modelos`)
Curso: clase 6 y demo nikodym (expediente de 9 piezas; RACI: quien construye no valida ni aprueba;
gatillos; audit trail; hash_i = SHA-256(hash_{i−1} + evento_i); ataque torpe vs prolijo; sello
externo; lineage: run_id, config_hash, data_hash, root_seed, git_sha; model card con limitaciones
declaradas vs automáticas; informe técnico 8 secciones + 3 anexos; 4 errores que bajan nota;
«usos no previstos»; «reproducible y documentado no significa aprobado»).
Mínimos:
- Marco de riesgo de modelo: SR 11-7 (Fed/OCC 2011) — desarrollo, validación independiente
  («effective challenge»), gobierno; inventario de modelos y tiering por materialidad; EBA
  (GL/2017/16 sobre estimación de PD/LGD — verificar), guía del BCE para modelos internos (TRIM /
  ECB Guide to internal models — verificar); CMF Chile: normas relevantes a modelos de provisiones
  y gestión de riesgo de crédito (Compendio de Normas Contables para bancos, cap. B-1 — verificar
  con WebSearch; y lo aplicable a no bancarias/fintech: Ley Fintec 21.521 — verificar). Formula con
  cautela lo que no verifiques.
- Ciclo de vida: desarrollo → validación → aprobación → implementación → monitoreo → revisión
  anual → retiro. Qué evidencia produce cada etapa.
- Trazabilidad: lineage (datos, código, config, entorno, semilla); hashing de contenido
  (canonicalización JSON); cadena de hashes (tipo blockchain sin consenso) y por qué no basta
  (ataque prolijo); sello externo (WORM storage, timestamping RFC 3161, firma) — qué garantiza cada
  capa; Merkle trees como alternativa para verificar eventos individuales.
- Model card (Mitchell et al. 2019) adaptada a crédito; datasheets; informe técnico; resumen
  ejecutivo de 1 página (5 secciones del curso); limitaciones buenas vs genéricas (ejemplos).
- Validación independiente: qué revisa un validador (replicación, benchmarking con challenger,
  pruebas de sensibilidad, revisión conceptual); cómo prepararse (el expediente como producto).
- Overrides y excepciones: registro, límites, monitoreo.
- Uso de librerías (nikodym, optbinning) en gobierno: pinning de versión, validación de la
  herramienta (la herramienta también es un «modelo» a validar).
Notebook: pipeline mínimo del generador que emite audit trail JSONL; canonicalización y hash (hashlib)
; cadena de hashes, verificación, ataque torpe/prolijo, sello externo; Merkle tree numpy/hashlib con
prueba de inclusión; lineage bundle (config_hash, data_hash con hash determinista del DataFrame,
semilla, versiones de librerías); model card generado desde la corrida a Markdown (mostrar con
mo.md). «Dos implementaciones»: aquí comparar hash del DataFrame con pandas.util.hash_pandas_object
vs serialización canónica propia (y explicar por qué difieren / cuál es estable entre versiones).
Planilla: `M22_raci_gatillos.xlsx` — matriz RACI editable (actividades × roles, check: exactamente
una A por fila con COUNTIF) y hoja de gatillos (5 partes) + CSV plantilla del inventario de modelos.

---
## M23 · Numpy optimizado vs scipy / statsmodels / scikit-learn / optbinning / nikodym  (`M23_numpy_vs_librerias`)
Pedido explícito del usuario: «un módulo que explique las ventajas y desventajas de usar solo numpy
optimizado vs scipy, statsmodels y otros». Transversal a toda la serie.
Mínimos:
- Criterios de decisión: correctitud, estabilidad numérica, rendimiento, transparencia/auditabilidad,
  dependencias y superficie de riesgo (supply chain, pinning, CVEs), portabilidad a producción
  (motor no-Python), mantenibilidad, validación de la herramienta por el validador de modelos.
- Estabilidad numérica en riesgo: sigmoide y log-verosimilitud estables (expit, log1p, logsumexp,
  `np.logaddexp`), cancelación catastrófica en logit(p) con p→0/1, suma de muchas log-probabilidades,
  condicionamiento de XᵀWX, inversión vs solve vs Cholesky vs QR, float32 vs float64.
- Rendimiento: vectorización, broadcasting, evitar bucles en bootstrap (índices en matriz), einsum,
  np.searchsorted para binning (vs pd.cut), np.bincount para tablas WoE, memoria; benchmarks reales
  con timeit (tabla).
- Qué aporta cada librería: scipy (optimizadores, distribuciones, tests exactos), statsmodels
  (inferencia, SE, tests, fórmulas, GLM con offset), scikit-learn (API de pipelines, regularización,
  métricas, pero sin p-valores y con penalización L2 POR DEFECTO — la trampa clásica: LogisticRegression()
  regulariza con C=1), optbinning (binning óptimo con MIP/CP, scorecard, monitoreo), nikodym (pipeline
  gobernado del curso: config validado, audit trail, model card). Diferencias de convención
  documentadas (ddof, signo del WoE en optbinning = ln(%malos/%buenos)?? — VERIFICA la convención de
  optbinning en su documentación/código instalado antes de afirmarla; bilateral en binomtest; etc.).
- Recomendación por capa: exploración, desarrollo, validación, producción. Estrategia «núcleo numpy
  puro auditable + librerías como oráculo de pruebas» (tests de paridad).
Notebook: benchmarks (timeit) numpy vs pandas vs sklearn/scipy para: binning+WoE (bincount vs groupby
vs optbinning), AUC (rank vs sklearn), IRLS vs statsmodels vs sklearn (y la trampa C=1), bootstrap
vectorizado vs bucle; demostración de inestabilidad numérica (sigmoide ingenua con overflow, logit de
p=1e-17, float32) y su versión estable; tabla de convenciones verificadas empíricamente (asserts);
slider de n para ver cómo escala cada implementación.

---
## C2 · Integrador: el scorecard como pipeline declarativo sobre Financiera Andes  (`C2_integrador_andes`)
(Se asigna al final; brief detallado aparte.)

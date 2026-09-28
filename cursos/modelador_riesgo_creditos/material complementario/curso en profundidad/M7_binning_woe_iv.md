# M7 · Binning, WoE e Information Value: mecánica, derivación y trampas

**Serie: Modelador de Riesgo en Profundidad** · Fase 1, módulo 7 de 8
Profundiza: C2 láminas 19-29 (binning, fine→coarse, WoE/IV, monotonicidad, umbrales, las tres trampas, screening≠selección)
Material asociado: `M7_binner_interactivo.py` (notebook Marimo: binner fine→coarse en vivo + las tres trampas reproducidas desde cero)

---

## 1. Por qué el estándar de la industria discretiza a propósito

Perder granularidad suena a pecado estadístico. El scorecard lo hace deliberadamente porque compra cuatro cosas que en este dominio valen más que la granularidad:

1. **No linealidad sin artificios:** la relación deuda→riesgo real tiene forma (los tramos extremos se comportan distinto); los bins la capturan sin polinomios ni splines que nadie puede explicar en comité.
2. **Robustez a outliers:** el cliente con deuda 100× cae en el último bin y pesa lo que pesa su bin — no arrastra ningún coeficiente (M6 §3.3).
3. **El missing como categoría:** el bin MISSING integra la ausencia informativa al modelo sin imputación (M6 §2).
4. **Explicabilidad estructural:** "deuda entre 2 y 5 millones: −12 puntos" es una frase de comité y un reason code regulatorio. La tabla de puntos ES el modelo.

El costo real: dentro de cada bin, todos los clientes son iguales (se pierde el ordenamiento intra-bin), y los bordes son discontinuidades (un peso más de deuda te cambia de bin). La industria acepta ese costo; E5 discute cuándo un ML sin binning lo recupera y qué pierde a cambio.

---

## 2. La mecánica: fine → coarse

### 2.1 Fine classing: ver la forma

Primera pasada con ~20 bins de igual frecuencia (cuantiles). El objetivo NO es modelar: es **ver** — la curva de tasa de malos por bin revela si la relación es monótona, en U, plana o ruidosa. Con 20 bins de un dataset de 15.000 y 165 malos, cada bin tiene ~8 malos: el zigzag es esperable y todavía no significa nada.

### 2.2 Coarse classing: decidir

Fusionar bins adyacentes hasta que cada bin final cumpla tres condiciones:

- **Masa suficiente:** ≥5% de la muestra (regla práctica del curso). Con pocos malos, conviene el criterio adicional de industria: un mínimo de malos por bin (p. ej. ≥20-30) — la estabilidad del WoE depende del conteo de malos, no del total (§5, Trampa 2).
- **Orden razonable:** WoE monótono salvo excepción de negocio (§4).
- **Sentido de negocio:** los cortes caen donde el negocio reconoce fronteras (utilización 30/56/80% se puede narrar; 31,7/55,2/79,8% también, pero redondear a fronteras narrables cuesta poco IV y compra mucha comunicación).

**Quién decide:** el modelador. Existen algoritmos de coarse classing (árboles de decisión univariados sobre el target, chi-merge, optimización de IV con restricción de monotonía — lo que hacen `optbinning` o `scorecardpy`), y son excelentes **puntos de partida**; el punto final es humano porque las tres condiciones de arriba incluyen una que ningún algoritmo tiene (el negocio). El flujo profesional: binning automático como propuesta → revisión y ajuste manual documentado → tabla final versionada.

### 2.3 Casos especiales

- **Missing:** siempre su propio bin primero; si queda con masa ínfima Y su WoE se parece al de un bin vecino, se puede fusionar con él (documentado).
- **Masa puntual dominante** (el 60% de los clientes con 0 consultas): ese valor es su propio bin y el resto se binnea aparte.
- **Categóricas:** el "binning" es agrupar categorías por WoE similar + sentido de negocio (regiones con perfil parecido); las categorías raras (<1-2%) van a un bin "otras".
- **Recencia censurada:** el 13 (M5 §4.1) queda naturalmente aislado en su bin "sin evento".

---

## 3. WoE e IV: derivación y lectura

### 3.1 Las fórmulas y su lógica

Para el bin b, con %B_b = proporción de todos los buenos que caen en b, y %M_b = ídem malos:

```
WoE_b = ln(%B_b / %M_b)
IV    = Σ_b (%B_b − %M_b) × WoE_b
```

**Lectura del WoE:** compara la composición del bin contra el perfil global. WoE = 0: el bin luce como la población. WoE > 0: sobre-representa buenos (bin bueno, en la convención del curso). WoE < 0: sobre-representa malos. Es un log-odds ratio centrado: WoE_b = logit(global) − logit(bin) en términos de odds de malo — por eso encaja tan naturalmente con la regresión logística de C3: transformar la variable a su WoE deja "pre-digerida" la no linealidad y todos los coeficientes en la misma escala.

**Derivación del IV** (para tu formación, esto lo vuelve memorable): el IV es la **divergencia de Jeffreys** (la versión simetrizada de Kullback-Leibler) entre la distribución de buenos y la de malos a través de los bins:

```
IV = KL(B‖M) + KL(M‖B) = Σ (%B − %M) ln(%B/%M)
```

De ahí sus propiedades: es ≥ 0 siempre; es 0 si y solo si buenos y malos se distribuyen igual (la variable no separa nada); crece sin cota cuando algún bin se vuelve casi puro (ln → ∞) — el germen matemático de la Trampa 3: una variable que "sabe" el desenlace produce bins casi puros y el IV explota. Y también el germen de la Trampa 1: cada refinamiento de la partición solo puede mantener o **aumentar** la divergencia medida (la desigualdad de procesamiento de datos al revés: agrupar solo puede perder separación aparente).

### 3.2 La convención de signo (aviso de compatibilidad)

El curso define WoE = ln(buenos/malos) → WoE alto = bin bueno. **`scorecardpy` y varios textos usan ln(malos/buenos)**: mismo |WoE|, signo invertido, mismo IV (el producto (%B−%M)×WoE no cambia de signo). Antes de interpretar cualquier salida de librería: verificar con un bin obvio (el de mora alta debe salir "malo"). Los bugs de signo no rompen el IV pero sí rompen la lectura y, aguas abajo, el signo esperado de los coeficientes en C3.

### 3.3 Los umbrales de Siddiqi y su uso correcto

| IV | Lectura | Acción |
|---|---|---|
| < 0.02 | Sin poder | Descartar |
| 0.02–0.10 | Débil | Puede aportar en conjunto |
| 0.10–0.30 | Medio | Candidata sólida |
| 0.30–0.50 | Fuerte | Revisar que sea legítima |
| > 0.50 | Sospechosamente fuerte | **Auditar antes de celebrar** |

Tres aclaraciones de uso profesional: (a) son heurísticas de Siddiqi para admisión de consumo — en comportamiento, IVs de 0.5-1.0 legítimos son comunes (la información transaccional propia es así de buena) y el umbral de auditoría se recalibra; (b) el IV depende de la tasa de malos y del binning: comparar IVs entre datasets distintos es comparar peras con manzanas; (c) el IV se calcula en DEV y se **contrasta en HO**: la pareja (IV_DEV, IV_HO) informa más que cualquiera de los dos solos.

### 3.4 IV vs alternativas de screening

Gini univariado/AUC (mide ordenamiento, no forma; complementario), chi² (sensible a n, no acotado), information gain (la versión de árboles; pariente directo). La práctica madura usa IV + Gini univariado + estabilidad (PSI de la variable entre cosechas) como terna de screening. El IV domina la industria por su acople con el flujo WoE y su interpretabilidad por bin, no por superioridad estadística intrínseca.

---

## 4. Monotonicidad: cuándo forzarla y cuándo no

**Por qué se exige:** (a) estabilidad — un zigzag de WoE con pocos malos por bin es casi siempre ruido muestral, y forzar monotonía es una regularización barata; (b) explicabilidad — "más deuda, más riesgo, siempre" es narrable y auditable; un scorecard donde el tramo 3 de deuda es más seguro que el 2 invita una pregunta de comité sin buena respuesta; (c) el ordenamiento del score hereda el orden de sus componentes.

**El test práctico del curso:** fusionar los bins del zigzag; si el IV apenas cae, era ruido (fusionar = regularizar gratis). Si el IV cae mucho, hay estructura real → investigar.

**Excepciones legítimas:** formas en U con historia de negocio. Edad (jóvenes y mayores más riesgosos que la meseta central, por razones distintas), utilización (0% absoluto a veces es "cliente sin uso = sin información", distinto del 10% saludable), antigüedad laboral. La regla: la no-monotonía se acepta si (a) sobrevive en HO y (b) el negocio la explica **antes** de mirar el WoE — la explicación post-hoc es sobreajuste narrativo, el equivalente cualitativo de la Trampa 2.

---

## 5. Las tres trampas, en profundidad

### Trampa 1 · El IV premia el número de bins

**Mecanismo:** el IV es una divergencia medida sobre una partición: refinar la partición nunca la reduce y el ruido muestral la aumenta. Una variable **aleatoria** con 64 niveles alcanzó IV 0.32 en la demo (¡"fuerte" según la tabla!) — cada nivel con pocos casos fluctúa por azar, y cada fluctuación suma IV positivo.

**Corolario cuantitativo:** el sesgo del IV crece aproximadamente con (n_bins/n_malos): más bins o menos malos = más IV fantasma. De ahí las dos defensas mecánicas: masa mínima por bin (limita n_bins) y comparación **a igual número de bins** entre variables candidatas.

**Defensa definitiva:** contrastar en HO. El IV fantasma es memoria del ruido de DEV; en HO se desploma (el 0.32 de la variable aleatoria cae a ~0.01). Ese contraste DEV/HO es el test de sobreajuste univariado más barato que existe.

### Trampa 2 · El IV es inestable con pocos malos

**Mecanismo:** el WoE de cada bin depende de su conteo de **malos** (la clase escasa). Con 165 malos en DEV repartidos en 5 bins, cada WoE cuelga de ~33 malos: mover 5 malos de bin cambia el WoE en ~0.15 y el IV en decenas de puntos. El caso del curso: `uso_linea_max_12m` con IV 1.21 en DEV y 0.55 en HO — no hubo fuga; hubo varianza.

**Defensas:** mínimo de malos por bin (no solo masa total); suavizado de conteos (el +0.5 de Laplace que el notebook usa evita WoE infinitos con bins de 0 malos); intervalos de confianza del IV vía bootstrap (E3) — reportar IV 0.85 [0.55, 1.20] cambia la conversación; y sobre todo, decidir con la **pareja** DEV/HO: una variable con IV 0.4/0.38 vale más que una con 1.2/0.5.

### Trampa 3 · El IV altísimo es una alarma, no un trofeo

M2 §5 la diseccionó (ventana corrida: 0.39→0.90; Δcupo t₀+12: 11.13). La regla operativa aquí: el screening genera **tres listas** — descartadas (IV<0.02 en DEV o colapso en HO), candidatas (rango sano y estable), y auditoría (IV>0.5): estas últimas no entran ni se descartan hasta tener explicación escrita. La tercera pregunta de comité de C2 es exactamente esta: *"¿alguna variable con IV sospechosamente alto llegó al modelo sin explicación de negocio?"*

---

## 6. Screening ≠ selección (el puente a C3)

Lo que el IV **hace**: mirar cada variable a solas contra el target, descartar lo inútil, priorizar auditorías. Lo que **no hace**: ver redundancia (dos utilizaciones con IV 1.2 correlacionadas al 95% aportan una sola vez), ver complementariedad (una variable de IV 0.08 puede sumar mucho si es ortogonal al resto), ni garantizar estabilidad multivariada. Por eso el pipeline sigue en C3 con correlaciones/VIF (redundancia), PSI (estabilidad temporal de cada variable) y selección en el contexto del modelo (stepwise u otras). Quedarse en "las 15 de mayor IV" produce scorecards redundantes y frágiles: cinco sabores de la misma señal de mora.

Checklist de salida del screening (el estado en que C2 deja el proyecto): tabla con IV_DEV, IV_HO, n_bins, % missing, flag de auditoría y decisión (descartar/candidata/auditar) para las 108; tablas WoE de las candidatas; y las explicaciones escritas de todo IV > 0.5.

---

## 7. El notebook: binner interactivo

`M7_binner_interactivo.py` (Marimo) trae tres piezas:

1. **Binner fine→coarse en vivo:** slider de número de bins finos y control de fusión; tabla WoE, IV, y flag de monotonicidad recalculados al vuelo sobre una variable sintética con forma realista (incluye bin MISSING).
2. **Trampa 1 reproducida:** variable 100% aleatoria; slider de número de niveles (4→64) mostrando el IV fantasma crecer en DEV y morir en HO.
3. **Trampa 2 reproducida:** slider de número de malos en la muestra (80→2.000); el IV de la MISMA variable se reporta con su intervalo bootstrap — mira el intervalo angostarse con los malos.

---

## 8. Preguntas de autoevaluación

1. Deriva por qué IV = KL(B‖M) + KL(M‖B). ¿Qué propiedad de esta divergencia explica la Trampa 1? ¿Y la 3?
2. Un bin tiene 12% de los buenos y 4% de los malos. Calcula su WoE en ambas convenciones de signo y su aporte al IV. ¿Es un bin bueno o malo?
3. ¿Por qué el criterio de masa mínima debería contar malos y no solo casos totales? Muestra con números qué pasa con un bin de 800 casos y 3 malos.
4. Te llega un binning automático de `optbinning` con 7 bins y cortes en 31,7% / 55,2% / 79,8%. ¿Qué harías antes de adoptarlo y por qué?
5. `edad` muestra forma en U con WoE −0.3 / +0.2 / +0.35 / +0.1 / −0.25. ¿La fuerzas monótona? ¿Qué dos condiciones exigirías para conservar la U?
6. Diseña la tabla final del screening de las 108 candidatas: columnas, umbrales de decisión y las tres listas de salida.

**Cierra la Fase 1.** La Fase 2 (E1–E8) desarrolla las limitaciones declaradas: reject inference, calibración, bootstrap, bureau, ML, normativa, stress testing y la bibliografía para seguir.

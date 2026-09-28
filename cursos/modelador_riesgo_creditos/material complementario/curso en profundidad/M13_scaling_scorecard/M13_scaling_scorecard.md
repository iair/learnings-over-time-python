# M13 · Del logit al scorecard: scaling y tabla de puntos

> **Ficha**
> - **Clases que profundiza:** clase 3, parte 3 (láminas 22–33: scaling PDO, puntos por tramo, scorecard de 41 filas, rango de puntos) y clase 4 v21, láminas 4–7 (base común + aporte, S0030427, reason codes). Toca la lámina 25 de la clase 4 (δ de calibración PIT) desde el lado del score.
> - **Prerrequisitos:** Serie 1 · M7 (WoE/IV), Serie 1 · E2 (calibración y tendencia central), Serie 2 · M10 (logística sobre WoE desde la verosimilitud). Prepara M14 (reason codes), M15–M16 (calibración y master scale) y M21 (artefacto congelado).
> - **Archivos del módulo:** `M13_scaling_scorecard.md` (este documento) · `M13_scaling_scorecard.py` (notebook Marimo: `marimo edit --sandbox M13_scaling_scorecard.py`) · `M13_scorecard_vivo.xlsx` (calculadora con fórmulas vivas).
> - **Tiempo estimado:** 2,5–3 h (lectura 75 min, notebook 60 min, ejercicios 45 min).

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

El curso cerró la construcción del modelo de Banco Austral con una logística de 8 variables sobre WoE, ajustada en DEV (3.322 créditos, 4,97 % de malos), y la convirtió en un scorecard con tres parámetros «de negocio»: **score base 600**, **odds base 50:1** (buenos por malo) y **puntos para duplicar las odds (points to double the odds, PDO) = 20**. De ahí salen dos constantes:

- factor = PDO / ln 2 = 20 / 0,693147 = **28,8539**
- offset = 600 − 28,8539 · ln 50 = 600 − 112,8771 = **487,1229**

La lectura de la escala: 580 puntos ⇔ 25:1 (PD 3,8 %), 600 ⇔ 50:1 (PD 2,0 %, exactamente 1/51 = 1,96 %), 620 ⇔ 100:1 (PD 1,0 %, exactamente 0,99 %). Cada 20 puntos, las odds de malo se dividen por dos.

Con el intercepto del stepwise, β₀ = −3,0252, y el reparto **en partes iguales** entre las 8 variables, cada tramo recibe una base común:

$$\text{base} = \frac{\text{offset}}{8} - \frac{\beta_0}{8}\cdot\text{factor} = 60{,}89 + 10{,}91 = 71{,}80$$

y una pendiente por variable, $-\beta_v\cdot\text{factor}$. Para `uso_tc_prom_12m` (β = −0,3525): 10,17 puntos por unidad de WoE. El tramo «hasta 0,208» (WoE 4,26) vale 71,80 + 43,33 = **115,1**; el tramo «sobre 0,339» (WoE −1,23), 71,80 − 12,51 = **59,3**. Las pendientes van de 8,9 puntos/WoE (`uso_tc_prom_3m`) a 26,7 (`deuda_interna_max_3m`). Todo el scorecard son esas dos cuentas, 41 veces. El notebook del curso lo escribe en una línea:

$$\text{puntos}(v,b) = -\left(\beta_v\,\text{WoE}_{v,b} + \frac{\beta_0}{n}\right)\cdot\text{factor} + \frac{\text{offset}}{n}.$$

El score de una solicitud es la suma de sus 8 tramos. S0030427: 70,1 + 80,2 + 77,0 + 75,4 + 67,9 + 101,4 + 61,4 + 70,5 = **603,8 ≈ 604 ⇔ PD 1,7 %** (odds 57,0:1). S0027717: 457 puntos ⇔ PD 73,9 %. El **rango** por variable (máximo − mínimo de sus tramos) es lo que se discute en comité: de 55,8 puntos (`uso_tc_prom_12m`, β chico) a 21,8 (`carga_financiera`); por coeficiente, la primera sería `deuda_interna_max_3m` (β −0,926), que por rango es sexta (26,6). Score en DEV: p5 524 · mediana 612 · p95 705. «En producción la tabla se redondea a enteros: es el anexo que se firma.» optbinning, con `scaling_method="pdo_odds"` y los mismos parámetros, produjo un score DEV casi idéntico (522 · 610 · 705).

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **La derivación** de factor y offset (dos ecuaciones, dos incógnitas) y por qué el score se define sobre las odds de *buenos* y no de malos.
2. **El reparto del intercepto** en partes iguales se declaró «una convención»; no se mostraron las alternativas ni su efecto sobre la lectura de cada fila y sobre los reason codes.
3. **Escalas alternativas** (otros PDO, otras anclas, scores crecientes con el riesgo) y cómo convertir un score o un corte de una escala a otra.
4. **El redondeo a enteros**: cuánto error introduce, si tiene sesgo, cuántas decisiones cambia en el corte, y si conviene redondear puntos o score.
5. **Puntos negativos**: cuándo aparecen y por qué no son un error.
6. **Recalibración y escala**: la clase 4 suma δ = 0,177 al logit; qué le pasa al score (se mueve 5,11 puntos para todos) y si se re-escala la tabla o se mueve el corte.
7. **El ancla es una promesa de calibración**: «600 ⇔ 50:1» es verdad solo donde el modelo está calibrado; y el PDO es verdad solo si la pendiente de calibración es 1.
8. **El scorecard como artefacto de datos**: qué se congela, qué se versiona, cómo se testea (se retoma en M21).

---

## 2. Intuición

**El score es un log-odds con otras unidades.** La logística produce $\eta = \ln\frac{p}{1-p}$ (log-odds de malo). El scaling no agrega información: es una recta que convierte $\eta$ en un número que un comité puede leer. Como toda recta, tiene **dos** grados de libertad: pendiente (factor) y posición (offset). Los tres parámetros del curso son redundantes: «600 @ 50:1, PDO 20» define exactamente la misma escala que «580 @ 25:1, PDO 20» o que «487,12 @ 1:1, PDO 20». El par (score base, odds base) es **un punto** de la recta; el PDO, su pendiente. Elegir el ancla es elegir en qué número de la escala quiere uno leer un nivel de riesgo conocido; no cambia nada del modelo.

**Por qué la suma funciona.** La logística sobre WoE es aditiva en log-odds: $\eta = \beta_0 + \sum_v \beta_v\,\text{WoE}_v$. Una transformación afín de una suma es una suma de transformaciones afines más una constante. Esa constante (offset e intercepto, llevados a puntos) no pertenece a ninguna variable: hay que decidir dónde se imprime. El curso la reparte en partes iguales (71,80 por variable); se podría imprimir en una fila aparte o repartir de otro modo. **El score del cliente no cambia**; cambia qué número aparece en cada fila de la tabla.

**Por qué el signo sale «bien».** Con WoE = ln(%buenos/%malos), un bin bueno tiene WoE alto y β es negativo (más WoE ⇒ menos log-odds de malo). El score se define sobre las odds de **buenos**, $\ln\frac{1-p}{p} = -\eta$, así que la pendiente en puntos es $-\beta_v\cdot\text{factor} > 0$: el bin bueno **suma** puntos. Dos signos negativos que se cancelan: es lo que hace legible la tabla (clase 3). Si una librería usa WoE con el signo contrario, β cambia de signo y los puntos quedan iguales: el scaling es invariante a la convención de WoE siempre que β se ajuste con la misma convención.

**Cómo leer un rango.** Un rango de $R$ puntos equivale a $R/\text{PDO}$ duplicaciones de las odds. Los 55,8 puntos de `uso_tc_prom_12m` son 2,79 duplicaciones: entre el mejor y el peor tramo, a igualdad del resto, las odds cambian por un factor $2^{2{,}79} = 6{,}9$. Esta lectura es la que se lleva a un comité.

**Qué no hace el scaling.** No calibra (el ancla es verdad solo si la PD es verdad), no ordena mejor (Gini idéntico con cualquier escala) y no hace el modelo más estable. Rotula. Todo lo que vaya mal en nivel o pendiente se arrastra a la tabla, con la apariencia de precisión que dan los números enteros.

---

## 3. Formalización

### 3.1 Factor y offset

Se postula una escala afín en el log-odds de buenos:

$$s(p) = \text{offset} + \text{factor}\cdot\ln\frac{1-p}{p} = \text{offset} + \text{factor}\cdot\ln O,\qquad O = \frac{1-p}{p}.$$

Condición de PDO: duplicar las odds suma PDO puntos,

$$s(2O) - s(O) = \text{factor}\cdot\ln(2O) - \text{factor}\cdot\ln O = \text{factor}\cdot\ln 2 = \text{PDO}\ \Rightarrow\ \text{factor} = \frac{\text{PDO}}{\ln 2}.$$

Condición de ancla: en las odds base $O_0$ el score vale $s_0$,

$$s_0 = \text{offset} + \text{factor}\cdot\ln O_0\ \Rightarrow\ \text{offset} = s_0 - \text{factor}\cdot\ln O_0.$$

Con PDO 20, $s_0 = 600$, $O_0 = 50$: factor $= 28{,}8539$, offset $= 487{,}1229$. Nótese que el offset es el score de odds 1:1 (PD 50 %). La inversa:

$$O(s) = \exp\!\left(\frac{s-\text{offset}}{\text{factor}}\right),\qquad p(s) = \frac{1}{1+O(s)}.$$

Como $\eta = \ln\frac{p}{1-p} = -\ln O$:

$$\boxed{s = \text{offset} - \text{factor}\cdot\eta}$$

### 3.2 Descomposición en puntos por variable

Con $\eta = \beta_0 + \sum_{v=1}^{n}\beta_v\,\text{WoE}_{v,b(v)}$, donde $b(v)$ es el bin del cliente en la variable $v$:

$$s = \underbrace{\text{offset} - \beta_0\,\text{factor}}_{C} + \sum_{v=1}^{n}\underbrace{\left(-\beta_v\,\text{factor}\right)\text{WoE}_{v,b(v)}}_{a_{v,b(v)}}.$$

$C$ es la **constante a repartir**; $a_{v,b}$ es el **aporte** del bin. Cualquier familia de bases $\{\text{base}_v\}$ con una fila constante $K = C - \sum_v \text{base}_v$ define una tabla

$$\text{puntos}(v,b) = \text{base}_v + a_{v,b},\qquad s = K + \sum_v \text{puntos}(v,b(v)),$$

y el score es **idéntico** para todo cliente, porque $K + \sum_v(\text{base}_v + a_{v,b(v)}) = C + \sum_v a_{v,b(v)}$. El reparto del curso es $\text{base}_v = C/n$, $K = 0$:

$$\text{puntos}(v,b) = \frac{C}{n} + a_{v,b} = \frac{\text{offset}}{n} - \frac{\beta_0}{n}\text{factor} - \beta_v\,\text{factor}\,\text{WoE}_{v,b} = -\left(\beta_v\text{WoE}_{v,b} + \frac{\beta_0}{n}\right)\text{factor} + \frac{\text{offset}}{n},$$

que es exactamente la línea del notebook del curso. En Austral, $C = 487{,}12 + 3{,}0252\cdot 28{,}8539 = 574{,}41$ y $C/8 = 71{,}80$.

Cuatro repartos (todos implementados en el notebook, con assert de que el score no cambia):

| Reparto | $\text{base}_v$ | $K$ | Lectura de una fila |
|---|---|---|---|
| Partes iguales (curso) | $C/n$ | 0 | WoE = 0 ⇔ $C/n$ puntos; el «neutro» depende de $\beta_0$ y de $n$ |
| Neutro fijo $N$ | $N$ | $C - nN$ | WoE = 0 ⇔ $N$ puntos, siempre; con $N=0$ la fila es el aporte puro, puede ser negativa |
| Mínimo cero («intercept-based») | $-\min_b a_{v,b}$ | $C - \sum_v\text{base}_v$ | el peor bin de cada variable vale 0; los puntos son «lo que el cliente gana sobre el peor caso» |
| Proporcional al rango | $C\,R_v/\sum_u R_u$ | 0 | variables con más rango cargan más constante; filas más dispersas |

con $R_v = \max_b a_{v,b} - \min_b a_{v,b} = |\beta_v|\cdot\text{factor}\cdot(\max_b\text{WoE}_{v,b} - \min_b\text{WoE}_{v,b})$. **El rango no depende del reparto** (es una diferencia dentro de la variable), por eso es la magnitud correcta para discutir en comité.

### 3.3 Reason codes: qué es invariante al reparto

Sea un método de reason codes que ordena variables por la brecha $g_v = r_v - \text{puntos}(v,b(v))$ contra una referencia **de la misma variable** $r_v$: el máximo de la variable (curso), el promedio poblacional de la variable, o el promedio de los aprobados cerca del corte. Si cambio el reparto, $\text{puntos}(v,b) \to \text{puntos}(v,b) + \Delta_v$, y cualquier referencia construida con los puntos de la variable se mueve igual, $r_v \to r_v + \Delta_v$. Entonces $g_v$ no cambia y el orden de motivos tampoco. En cambio, un método que compara **puntos absolutos entre variables** («las 3 variables con menos puntos») depende de $\Delta_v - \Delta_u$ y cambia con el reparto. En el notebook, sobre las 4.848 solicitudes TTD: brecha al máximo y brecha a la media, 0 % de cambios; método ingenuo, 62 % de cambios al pasar a «mínimo cero» y 99 % con «proporcional al rango». El detalle de reason codes (etiquetas bin a bin, monotonía, requisitos legales) es materia de M14.

### 3.4 Conversión entre escalas

Dos escalas $A$ y $B$ sobre el mismo modelo: $s_A = o_A - f_A\,\eta$ y $s_B = o_B - f_B\,\eta$. Eliminando $\eta$:

$$s_B = o_B + \frac{f_B}{f_A}\,(s_A - o_A).$$

Es una recta con pendiente $f_B/f_A = \text{PDO}_B/\text{PDO}_A$. Un corte $c_A$ se convierte con la misma fórmula; un **ancho de banda** se escala por $\text{PDO}_B/\text{PDO}_A$. Una escala **creciente con el riesgo** es el caso $f_B < 0$: $s = o + |f|\,\eta$, en la que un bin bueno resta puntos. Nada de esto cambia la PD de un cliente ni el orden (el notebook lo verifica cliente a cliente con `assert np.allclose`). Tabla de referencia (misma PD en cuatro escalas):

| PD | odds | PDO 20 · 600 @ 50:1 (curso) | PDO 40 · 500 @ 20:1 | PDO 50 · 1000 @ 100:1 | riesgo ↑, PDO 20 · 400 @ 50:1 |
|---|---|---|---|---|---|
| 0,50 % | 199 | 639,9 | 632,6 | 1.049,6 | 360,1 |
| 1,00 % | 99 | 619,7 | 592,3 | 999,3 | 380,3 |
| 2,00 % | 49 | 599,4 | 551,7 | 948,5 | 400,6 |
| 4,00 % | 24 | 578,8 | 510,5 | 897,1 | 421,2 |
| 8,00 % | 11,5 | 557,6 | 468,1 | 844,0 | 442,4 |
| 15,00 % | 5,7 | 537,2 | 427,2 | 792,9 | 462,8 |
| 30,00 % | 2,3 | 511,6 | 376,0 | 728,9 | 488,4 |

(PD 2 % no es exactamente 600: 600 es odds 50:1, PD 1,96 %; con odds 49:1 se obtiene 599,4. El «600 ⇔ 2,0 %» del curso es un redondeo.)

### 3.5 Redondeo a enteros

Sea $e_{v,b} = \text{round}(\text{puntos}(v,b)) - \text{puntos}(v,b) \in [-\tfrac12, \tfrac12]$ el error de cada fila. El error del score de un cliente con tabla entera es

$$\varepsilon = \sum_{v=1}^{n} e_{v,b(v)}\quad(+\,e_K\text{ si hay fila constante}).$$

**Cota dura:** $|\varepsilon| \le n/2$ (en Austral, 4 puntos; en el notebook, 2,5). Una cota más fina para una tabla concreta es $\sum_v \max_b |e_{v,b}|$ (2,02 en el notebook).

**Tamaño típico.** Si los $e_{v,b}$ se comportaran como uniformes independientes en $[-\tfrac12,\tfrac12]$, $\varepsilon$ seguiría una distribución de Irwin–Hall centrada con

$$\text{Var}(\varepsilon) = \frac{n}{12},\qquad E|\varepsilon| \approx \sqrt{\frac{n}{12}}\sqrt{\frac{2}{\pi}}\quad(\text{aprox. normal}).$$

Para $n = 8$: sd 0,82, $E|\varepsilon| \approx 0{,}65$. **Pero los errores no son aleatorios por cliente**: cada bin tiene un error fijo y todos los clientes del bin lo heredan. Si los bins más poblados tienen errores del mismo signo, aparece un **sesgo**: $E[\varepsilon] = \sum_v\sum_b \pi_{v,b}\,e_{v,b} \ne 0$, con $\pi_{v,b}$ la proporción de clientes en el bin. En el notebook: media de $\varepsilon$ = −0,65 puntos, sd 0,55 y $E|\varepsilon|$ = 0,72, mayor que el 0,52 que predice el supuesto de independencia. La tabla entera, en promedio, puntúa 0,65 puntos más abajo que la exacta.

**Efecto en la PD.** Las odds se multiplican por $e^{\varepsilon/\text{factor}}$. Con factor 28,85: error máximo de 4 puntos (Austral) ⇒ ±14,9 % en odds; típico 0,65 ⇒ ±2,3 %. Con PDO 5 (factor 7,2) el mismo error típico ya es ±9 % en odds: **PDO chico hace caro redondear**.

**Decisiones que cambian en un corte $c$.** Con la regla «aprueba si $s \ge c$», la decisión cambia si $c$ queda entre $s$ y $s+\varepsilon$. Si $S$ tiene densidad $f_S$ suave cerca de $c$ y $\varepsilon$ es pequeño,

$$P(\text{cambia}) = E\big[\,|F_S(c) - F_S(c-\varepsilon)|\,\big] \approx f_S(c)\cdot E|\varepsilon|.$$

En el notebook (corte 543, 24.000 solicitudes): cambian 132 decisiones (0,55 %); la aproximación predice 179 (la diferencia viene de que $\varepsilon$ no es independiente de $S$: depende de los bins, que determinan $S$). Para Austral, con densidad TTD cerca de 560 de ≈ 0,54 % por punto (deciles 2 y 3 de la lámina 35), 8.585 × 0,0054 × 0,65 ≈ **30 decisiones (≈ 0,35 %)** — estimación, no cálculo del curso.

**Redondear puntos vs redondear score.** Redondear solo el score final tiene error máximo 0,5 y sin sesgo sistemático por bin; redondear la tabla tiene error hasta $n/2$ y sesgo. Pero ambos mueven decisiones respecto del score continuo (121 y 132 en el notebook) y **entre sí** discrepan en 225. La pregunta relevante no es cuál es «más exacto», sino **cuál es el artefacto firmado**: si la tabla entera es el anexo, el score oficial es la suma de enteros, y la master scale, la calibración y el corte deben construirse sobre ese score (no sobre el continuo del notebook de desarrollo).

### 3.6 Puntos negativos

Con partes iguales, la fila $(v,b)$ es negativa si

$$\frac{C}{n} + (-\beta_v\,\text{factor})\,\text{WoE}_{v,b} < 0 \iff \text{WoE}_{v,b} < -\frac{C}{n\,|\beta_v|\,\text{factor}},\qquad C = s_0 - \text{factor}\,(\ln O_0 + \beta_0).$$

Se vuelve más probable con **score base bajo** ($s_0$ chico), **PDO alto** (factor grande, que agranda el aporte y achica $C$ cuando $\ln O_0 + \beta_0 > 0$), **muchas variables** ($C/n$ chico) y WoE extremos (bins chicos, clases raras). En Austral la fila mínima es 27,1 (`meses_desde_mora_12m` «hasta 2»): lejos de 0. En el notebook, 8 de 16 combinaciones (PDO 20–80 × base 200–600 @ 50:1) producen filas negativas. **No son un error**: el score es exacto. Si el negocio exige puntos no negativos, se usa «mínimo cero» y se publica la constante; lo que no se hace es truncar a 0, que sí cambia scores y rompe la igualdad con el logit.

### 3.7 Recalibración y escala

Calibración de intercepto (clase 4): $\text{logit}(p^{cal}) = \eta + \delta$, con $\delta$ tal que la PD media calibrada iguala la tasa observada. En el score:

$$s^{cal} = \text{offset} - \text{factor}\,(\eta + \delta) = s - \delta\cdot\text{factor}.$$

Todos los clientes bajan lo mismo. En Austral, $\delta = 0{,}177$ ⇒ **5,11 puntos** (0,64 por variable si se repartiera). S0030427 pasa de PD 1,72 % a 2,05 %; el «600 ⇔ 50:1» pasa a «600 ⇔ 41,9:1» (PD 2,33 %).

Si la recalibración también corrige la **pendiente** (Platt / calibración logística): $\text{logit}(p^{cal}) = \delta_0 + \delta_1\eta$. Entonces

$$s^{cal} = \text{offset} - \text{factor}(\delta_0 + \delta_1\eta) = \delta_1\,s + (1-\delta_1)\,\text{offset} - \delta_0\,\text{factor},$$

otra transformación afín del score original: preserva el orden si $\delta_1 > 0$, y un corte en la escala calibrada $c^{cal}$ equivale a $c = \big(c^{cal} - (1-\delta_1)\text{offset} + \delta_0\text{factor}\big)/\delta_1$ en la original.

**Tres implementaciones:**

- **A. Mover el corte:** tabla intacta, corte $c + \delta\cdot\text{factor}$ sobre el score original. Un número cambia, una línea en la política. Decisiones idénticas al score calibrado, salvo el redondeo ya existente de la tabla.
- **B. Re-escalar la tabla:** restar $\delta\cdot\text{factor}/n$ a cada fila y re-redondear. Nuevo anexo, nueva firma, nuevo hash. Como $\delta\cdot\text{factor}/n$ no es entero, el re-redondeo reparte el desplazamiento de forma desigual entre combinaciones de bins. En el notebook ($\delta = 0{,}369$, 10,66 puntos, 2,13 por variable) A y B discrepan en 34 de 24.000 decisiones, sin ninguna razón de riesgo.
- **C. Cambiar solo el mapeo score → PD** (master scale, M16): tabla y corte en puntos intactos; la PD asociada a cada score cambia. Si el corte está definido **en PD** (apetito), se mueve solo; si está definido en puntos, hay que decidir explícitamente.

Recomendación: la tabla es un artefacto de **orden** y se congela; $\delta$ es un parámetro de **nivel** y vive en el mapeo score → PD, versionado aparte (A o C). Re-escalar la tabla (B) solo tiene sentido cuando se re-desarrolla el modelo.

### 3.8 El PDO efectivo y el ancla observada

Ajuste una logística de `malo` sobre el propio score: $\text{logit}(p) = a + b\,x$, con $x = (s - \text{offset})/\text{factor} = -\eta$. En DEV la solución es exactamente $a = 0$, $b = -1$. Demostración: las ecuaciones de verosimilitud del modelo completo son $\sum_i (y_i - p_i)\,x_{ij} = 0$ para toda columna $j$, incluida la constante. El candidato $(a,b) = (0,-1)$ reproduce las mismas $p_i$, y sus dos ecuaciones son $\sum_i (y_i - p_i) = 0$ (ya se cumple) y $\sum_i (y_i - p_i)\,\eta_i = \beta_0\sum_i(y_i-p_i) + \sum_j \beta_j \sum_i (y_i-p_i)x_{ij} = 0$. Como la log-verosimilitud es estrictamente cóncava, es el único máximo. Por lo tanto, en DEV:

$$\text{PDO}_{\text{efectivo}} = -\frac{\text{PDO}}{b} = \text{PDO},\qquad \text{ancla observada} = \text{ancla nominal}.$$

Fuera de DEV, $a$ y $b$ se estiman libremente: $a \ne 0$ es un sesgo de nivel (lo que corrige $\delta$) y $b \ne -1$ es una pendiente distinta (el PDO real ya no es 20). En el notebook, con el deterioro plantado de 0,35 en log-odds, OOT da PDO efectivo 18,6 y, en el score base, un log-odds de malo 0,21 más alto que el nominal (≈ 6 puntos): «600 ⇔ 50:1» es en realidad ≈ 40,5:1. El promedio de la cartera, en cambio, requiere $\delta = 0{,}369$: el desvío no es uniforme a lo largo de la escala.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| **PDO + ancla (score base, odds base)**, score ↑ = mejor | Escala legible y estable entre versiones; cada PDO puntos = ×2 odds | Ninguno técnico; exige documentar el ancla y no confundirla con calibración | Default en admisión con scorecards WoE | Estándar de los textos de scorecards (Siddiqi 2006/2017; Refaat 2011); `optbinning` `scaling_method="pdo_odds"` |
| **Min–max** (score en $[a,b]$ fijo) | Rango de puntajes predefinido (p. ej. 0–1000) | La relación puntos ↔ odds depende del modelo: cambia en cada re-desarrollo; PDO implícito no redondo | Cuando el sistema de decisión exige un rango fijo | `optbinning` `scaling_method="min_max"` (con redondeo por MIP opcional) |
| **Score creciente con el riesgo** | Alinear con la convención «más = peor» (algunas escalas de cobranza o de PD×1000) | Tablas con signo invertido; confusión si convive con una escala «más = mejor» | Solo si el ecosistema ya la usa | `reverse_scorecard=True` en optbinning; práctica de algunas instituciones (verificar caso a caso) |
| **Reparto en partes iguales** | Cada fila es «puntos totales» del atributo; sin fila constante | El neutro (WoE = 0) depende de β₀ y n; cambia al recalibrar el intercepto si se re-escala | Default del curso y de optbinning `pdo_odds` | Curso (clase 3); fórmula clásica de Siddiqi |
| **Mínimo cero + constante** («intercept-based») | Todos los puntos ≥ 0; fila = ganancia sobre el peor bin | Hay una constante que imprimir; lectura por fila distinta | Cuando el negocio no acepta puntos negativos | `intercept_based=True` en optbinning |
| **Neutro fijo $N$ + constante** | WoE = 0 ⇔ $N$ puntos, independiente de β₀: bins nuevos o no vistos tienen un valor estable | Puede producir negativos (N = 0); constante grande | Cuando se quiere que la tabla no cambie al mover el intercepto | Práctica de desarrollo interno; no conozco una norma que lo exija |
| **Proporcional al rango** | Variables importantes «se ven» con más puntos | Filas muy dispersas; engaña al método ingenuo de reason codes | Rara vez; presentaciones | Práctica ocasional; sin respaldo normativo conocido |
| **Redondeo de puntos a entero** | Anexo firmable, cálculo manual, reproducible en cualquier sistema | Error hasta $n/2$ con sesgo por bin; decisiones en el corte | Siempre que la tabla sea el artefacto productivo | Curso («en producción la tabla se redondea a enteros»); `rounding=True` en optbinning (`np.rint`) |
| **Redondeo solo del score final** | Error ≤ 0,5 sin sesgo por bin | La tabla publicada tiene decimales; hay que especificar la regla de redondeo en cada sistema | Motores de decisión que puntúan con la tabla continua | Práctica de implementación |
| **Redondeo óptimo (MIP)** | Minimiza el error agregado respetando restricciones (rango, monotonía) | Solver, reproducibilidad | Tablas con pocos bins y mucho peso en pocas filas | `optbinning` para `min_max` |
| **Recalibrar moviendo el corte / el mapeo a PD** | Cambio de nivel sin re-firmar la tabla | Hay que gobernar un segundo artefacto (política o master scale) | Calibración PIT/TTC periódica | Clase 4 (δ); M15–M16 |
| **Recalibrar re-escalando la tabla** | Mantener «600 ⇔ 50:1» verdadero en la tabla | Nuevo anexo, re-redondeo, decisiones que cambian sin razón de riesgo | Solo en re-desarrollo | — |

Sobre «puntajes de bureau» (FICO 300–850, etc.): su construcción interna no es pública en detalle y no se usa aquí como referencia técnica; lo único seguro es que son escalas «más = mejor» sobre un rango fijo.

---

## 5. Cuándo falla: trampas y modos de falla

**T1. El ancla no es una calibración.**
*Síntoma:* el comité lee «600 ⇔ PD 2 %» en la tabla y lo usa para pricing o provisiones. *Causa:* el scaling solo rotula el log-odds del modelo; si el nivel del modelo está corrido (otro ciclo, otro mix), el rótulo miente. *Detección:* tabla por bandas de score con PD nominal vs tasa observada en HO/OOT (notebook, sección 8: banda 560–580, nominal 5,4 %, DEV 5,2 %, OOT 8,2 %). *Qué hacer:* separar escala (orden) de mapeo score → PD (nivel); documentar la muestra de calibración (clase 4) y nunca publicar PD «de la escala» sin validación.

**T2. El PDO real no es el nominal.**
*Síntoma:* bandas de 20 puntos que no duplican las odds fuera de DEV; «20 puntos = mitad de riesgo» deja de ser cierto. *Causa:* pendiente de calibración ≠ 1 (sobreajuste la aplana; cambios de población pueden empinarla). *Detección:* logística de `malo` sobre el score en HO/OOT ⇒ PDO efectivo $= -\text{PDO}/b$ (en DEV es exactamente PDO, §3.8; en el OOT del notebook, 18,6). *Qué hacer:* reportarlo en el tablero de monitoreo (M20); si es material, la calibración debe incluir pendiente (Platt) y eso se gobierna en el mapeo a PD, no en la tabla.

**T3. Códigos especiales neutralizados en silencio.**
*Síntoma:* un segmento de riesgo alto (sin bureau) recibe puntos «normales». *Causa:* la implementación asigna WoE 0 al especial. En `optbinning` 1.0.0, `transform` usa `metric_special=0` y `metric_missing=0` por defecto: en el notebook, el código −99 (20 % de malos, WoE −0,68 en la **misma** tabla) recibe los puntos neutros 109,5 en vez de 94,7; el score medio de ese segmento sube 15,7 puntos y el Gini DEV baja de 0,551 a 0,543. La tabla impresa muestra el WoE empírico: el error solo se ve en la columna de puntos. *Detección:* test que compare, para cada fila, puntos contra $\text{base} + a_{v,b}$ con el WoE publicado. *Qué hacer:* `binning_transform_params={"metric_special": "empirical", "metric_missing": "empirical"}` o mapping explícito en el artefacto. Relacionado: `binear()` del curso (y optbinning con lista de especiales) junta −9 y −99 en un mismo bin; la clase 4 (lámina 16) exige separarlos.

**T4. Bin no visto: el neutro depende del reparto.**
*Síntoma:* una categoría nueva en producción recibe puntos distintos según quién implementó. *Causa:* «WoE 0» significa $C/n$ puntos con partes iguales (71,8 en Austral), $N$ con neutro fijo, y $-\min_b a_{v,b}$ con mínimo cero: no es un valor universal. *Detección:* test de contrato con una categoría sintética no vista. *Qué hacer:* fila `_NO_VISTO` explícita por variable en el artefacto (notebook, sección 10), con su justificación (neutro, conservador o rechazo a revisión).

**T5. Redondeo con sesgo y en lugares distintos.**
*Síntoma:* el score del motor de decisión difiere en 1–3 puntos del de validación; la tasa de aprobación en el corte no calza. *Causa:* un sistema redondea la tabla, otro el score; o uno usa redondeo a par (`np.round`, `np.rint`) y otro «.5 lejos de cero» (`ROUND` de Excel/Sheets). *Detección:* test de paridad fila a fila y cliente a cliente (M21); media de $\varepsilon$ en la población (−0,65 en el notebook). *Qué hacer:* la tabla entera es el artefacto; la regla de redondeo se declara; la master scale y el corte se construyen sobre el score entero.

**T6. Reason codes que dependen de una convención de impresión.**
*Síntoma:* motivos distintos para el mismo cliente tras «ordenar la tabla». *Causa:* método que compara puntos absolutos entre variables. *Detección:* re-calcular motivos con otro reparto (notebook: 62–99 % de cambios con el método ingenuo). *Qué hacer:* brecha contra una referencia de la misma variable (máximo o promedio); ver M14.

**T7. Recalibrar re-escalando la tabla.**
*Síntoma:* nuevo anexo cada trimestre; decisiones que cambian sin cambio de riesgo. *Causa:* confundir nivel con orden. *Detección:* diff de decisiones A vs B (34 en el notebook). *Qué hacer:* §3.7, opción A o C.

**T8. Cortes heredados sin convertir.**
*Síntoma:* al migrar de una escala antigua a la del modelo nuevo se conserva «el corte 480» por costumbre. *Causa:* el número del corte no tiene significado fuera de su escala. *Detección:* convertir el corte con §3.4 y comparar la PD implícita. *Qué hacer:* definir cortes en PD u odds y derivar el número en puntos para cada escala (ver §8.3).

**T9. Truncar negativos o «maquillar» filas.**
*Síntoma:* puntos mínimos 0 en una tabla con partes iguales. *Causa:* truncamiento a 0. *Detección:* assert «suma de puntos = transformación del logit». *Qué hacer:* desplazar con mínimo cero, nunca truncar.

**T10. Penalización por defecto en la librería.**
*Síntoma:* puntos que no coinciden con statsmodels. *Causa:* `LogisticRegression()` de scikit-learn usa L2 con C = 1 por defecto. En el notebook el efecto es pequeño (máximo 0,44 puntos de diferencia en score; el rango de `antiguedad_meses` baja de 28,2 a 27,8) porque n es grande y los WoE están acotados, pero con carteras chicas o WoE extremos no lo es. *Qué hacer:* declarar la penalización en la ficha del modelo; si no se quiere, `C` muy grande (o sin penalización según la versión de sklearn).

---

## 6. Puente con ingeniería

**El scorecard es un artefacto de datos, no un modelo.** Lo que se despliega es una tabla y unos metadatos; producción puntúa por búsqueda (lookup) y suma enteros, sin β ni exponenciales. Esto permite puntuar en SQL, en un motor de reglas o a mano, y hace trivial la paridad.

**Esquema mínimo del artefacto** (una fila por atributo):

```
variable        : str      # nombre canónico de la variable (contrato de datos, M21)
bin_id          : int      # orden estable del bin
tipo_bin        : enum     # intervalo | categoria | especial | missing | no_visto | constante
limite_inf      : float?   # intervalo: (limite_inf, limite_sup]
limite_sup      : float?
codigo_especial : float?   # -9, -99, ... un código por fila, nunca agrupados sin decisión documentada
woe             : float    # trazabilidad; producción no lo usa
puntos          : float    # continuo (desarrollo)
puntos_int      : int      # LO QUE SE FIRMA Y SE USA
```

**Metadatos versionados** (cabecera del artefacto): PDO, score base, odds base, factor, offset (10 decimales), reparto del intercepto, regla de redondeo (`half_even` / `half_away`), regla de bin no visto, versión del modelo, hash SHA-256 de la serialización canónica (el notebook lo calcula; cambia con cualquier parámetro), fecha y responsable. La calibración ($\delta$, o $\delta_0,\delta_1$) y el corte **no** van aquí: van en un artefacto de política/master scale con su propio ciclo de vida.

**Pipeline declarativo** (en el estilo de Serie 1 · M5):

```yaml
scorecard:
  fuente_modelo: modelo_v3.json        # β, mapas WoE, cortes (congelados en DEV)
  escala: {pdo: 20, score_base: 600, odds_base: 50, sentido: mas_es_mejor}
  reparto_intercepto: partes_iguales   # | minimo_cero | neutro_fijo:{N: 0} | proporcional_rango
  redondeo: {nivel: fila, regla: half_away}
  no_visto: neutro                     # | peor_bin | derivar_a_revision
  salida: scorecard_v3.csv + scorecard_v3.meta.json + sha256
politica:
  mapeo_pd: master_scale_v3_2026Q3.csv # δ vive aquí
  corte: {en: pd, valor: 0.04}         # el número en puntos se deriva
```

**Invariantes verificables (tests tipo CI):**

1. `score_por_lookup(artefacto, X) == suma(puntos_int)` para un set dorado de solicitudes (igualdad exacta de enteros).
2. `|score_continuo − (offset − factor·η)| < 1e-9` para todos los repartos (el notebook lo verifica en DEV/HO/OOT/TTD).
3. `|score_int − score_continuo| ≤ n/2` y, más fino, `≤ Σ_v max_b |e_vb|`.
4. Exactamente una fila `constante` y una fila `no_visto` por variable; todos los especiales declarados tienen fila propia.
5. Puntos por fila $= \text{base}_v + (-\beta_v\,\text{factor})\,\text{WoE}_{v,b}$ con el WoE publicado (atrapa T3).
6. Rango por variable invariante al reparto; signo de todas las pendientes > 0 (convención del curso).
7. Reason codes de un set dorado idénticos antes y después de un cambio de reparto o de un cambio de calibración.
8. Hash del artefacto en producción = hash registrado en el expediente (M22).

**Qué se congela y qué se versiona.** Congelado con el modelo: cortes de bins, WoE, β, reparto, escala, tabla entera. Versionado aparte, con cadencia propia: δ (calibración), master scale, corte, textos de motivos. Un cambio de nivel nunca debería cambiar el hash del scorecard.

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy desde cero (notebook) | Librería | Diferencias y convención | En producción |
|---|---|---|---|---|
| Logística sobre WoE | `irls_logit`: Newton-Raphson, $\beta \leftarrow \beta + (X^\top W X)^{-1}X^\top(y-p)$ | `statsmodels.Logit` (MLE sin penalización); `sklearn.LogisticRegression` | statsmodels = MLE (coinciden a 1e-7 en el notebook). sklearn penaliza L2 con C = 1 por defecto: β encogidos. Con `C=1e12` coincide a 1e-8 | statsmodels para el expediente (errores estándar, p-valores); IRLS propio solo como verificación |
| Factor, offset, puntos | Una línea: `-(β·WoE + β0/n)·factor + offset/n` y la versión «base + aporte» | `optbinning.scorecard.Scorecard(scaling_method="pdo_odds")` | Idéntica fórmula (verificado a 2,5e-7 puntos en optbinning 1.0.0). optbinning: WoE sin suavizado +0,5; especiales/missing con WoE 0 por defecto en `transform` (T3); `intercept_based`, `reverse_scorecard`, `rounding` disponibles | Tabla propia exportada como artefacto; la librería como herramienta de desarrollo |
| Logit / sigmoide | `1/(1+np.exp(-z))`, `np.log(p/(1-p))` | `scipy.special.expit`, `scipy.special.logit` | scipy es numéricamente más estable en colas (p cercano a 0 o 1); con PD de scorecard es irrelevante | scipy o equivalentes estables |
| δ de calibración | Aproximación `logit(tasa) − logit(PD media)` | `scipy.optimize.brentq` sobre $\bar p(\delta) - \text{tasa}$ | La aproximación subestima (0,322 vs 0,369 en el notebook; 0,143 vs 0,177 en Austral) porque $\sigma$ es convexa en la cola baja | brentq (exacto, clase 4) |
| Redondeo | `np.round` (a par) | `np.rint` (a par, usado por optbinning), `ROUND` de Excel/Sheets (lejos de cero), `round()` de Python (a par) | Solo difieren en valores exactamente x,5; con puntos continuos es raro, con tablas ya publicadas a un decimal no | Declarar la regla en el artefacto y testear paridad |
| Score por lookup | `map` de etiquetas a `puntos_int` + fila constante | SQL `JOIN`/`CASE`; `Scorecard.score()` | `score()` de optbinning no redondea salvo `rounding=True` | Lookup sobre el artefacto congelado |
| Hash del artefacto | `hashlib.sha256` sobre CSV canónico (orden fijo, 6 decimales, `\n`) | Mismo | La serialización canónica es la parte difícil: orden de filas, formato de floats, fin de línea | Igual; registrar en el expediente |

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral (curso)

**Reproducción de la tabla.** Con factor 28,8539, offset 487,1229, β₀ = −3,0252 y n = 8: base 71,80. Pendientes $-\beta_v\cdot\text{factor}$: `uso_linea_prom_12m` 14,57; `meses_desde_mora_12m` 22,14; `uso_tc_prom_12m` 10,17; `deuda_interna_max_3m` 26,71; `antiguedad_meses` 17,49; `deuda_otras_prom_12m` 22,78; `carga_financiera` 12,06; `uso_tc_prom_3m` 8,91. La planilla `M13_scorecard_vivo.xlsx` reproduce las filas de `uso_tc_prom_12m` (115,1 / 87,5 / 77,0 / 70,9 / 59,3) y las de `uso_linea_prom_12m` y `antiguedad_meses` a partir de WoE reconstruidos.

**Lectura de rangos en duplicaciones.** 55,8 puntos (`uso_tc_prom_12m`) = 2,79 duplicaciones = ×6,9 en odds; 21,8 (`carga_financiera`) = 1,09 duplicaciones = ×2,1. «Entre el mejor y el peor uso de tarjeta, a igualdad del resto, las odds de malo cambian casi 7 veces» es una frase que un comité entiende.

**S0030427 con redondeo y calibración.** Score 603,8 (PD 1,72 %). Con tabla entera, el score es 604 si todas las filas del cliente redondean «a favor»; la diferencia máxima posible para cualquier cliente es 4 puntos (odds ±14,9 %). Con la calibración PIT, su PD pasa a σ(logit(1,72 %) + 0,177) = 2,05 %, equivalente a un score calibrado de 603,8 − 5,11 = 598,7.

**El corte 560 y la calibración.** Un corte de 560 definido *antes* de calibrar significaba odds 12,5:1 (PD 7,4 %). Tras el δ PIT, el mismo número 560 significa PD σ(logit(7,4 %) + 0,177) = 8,7 %. Si el apetito está en PD, el corte en score original debe subir a 565,1 (opción A). Si se deja en 560, se está aceptando más riesgo por un cambio de calibración que nadie votó: es la trampa T1 hecha política. Con el ancla TTC que usa la clase 5 (δ = 0,1115) el efecto es menor pero del mismo signo: 560 pasa a PD 8,2 % y el corte equivalente sube a 563,2. La elección TTC vs PIT se discute en M15.

**Redondeo en la bandeja.** Con 8.585 solicitudes TTD y densidad ≈ 0,54 % por punto cerca de 560, redondear la tabla movería del orden de 30 decisiones (≈ 0,35 %). Es poco, pero no es cero: si validación usa el score continuo y producción el entero, el backtesting de la tasa de aprobación no calzará, y la diferencia tendrá que estar explicada en el expediente.

### 8.2 Banco Sintético (notebook)

Pipeline corto con 5 variables (`uso_linea_prom_12m`, `meses_desde_mora_12m`, `antiguedad_meses` forzada con IV 0,088, `carga_financiera`, `consultas_6m`), DEV de 10.065 créditos con 11,3 % de malos. β₀ = −2,063 (logit de la tasa de DEV: −2,063); los cinco β negativos. Con la escala del curso: $C = 546{,}65$, base 109,33 por variable; rangos 45,5 / 40,9 / 28,2 / 14,8 / 11,8 puntos. Score DEV: p5 503, mediana 562, p95 598 — unos 50 puntos por debajo de Austral porque la tasa de malos es más del doble (con 11 % de malos, las odds medias rondan 8:1, que en esta escala es ≈ 547).

Resultados que el notebook demuestra (con los controles en sus valores por defecto):

- **Score = suma de puntos = offset − factor·η** en DEV/HO/OOT/TTD para los 4 repartos: diferencia máxima 2,3e-13.
- **Repartos:** con neutro N = 0 aparecen 12 filas negativas (mínimo −29,9); con mínimo cero la constante es 472,5 y las filas van de 0 a 45,5; con proporcional al rango, de 37,9 a 201,4. Reason codes por brecha (al máximo o a la media): 0 % de cambios; método ingenuo: 62 % y 99 %.
- **Redondeo** (24.000 solicitudes, corte 543): max |ε| = 2,01 (cota 2,5; cota de la tabla 2,02), sesgo −0,65, 132 decisiones cambian (0,55 %).
- **Recalibración en OOT:** tasa 15,40 % vs PD media 11,65 %; δ exacto 0,369 (aprox. 0,322); desplazamiento 10,66 puntos; A vs B: 34 decisiones distintas. PD verdadera media en OOT (generador): 15,82 %.
- **PDO efectivo:** 20,000 en DEV (identidad de §3.8), 18,6 en OOT; el ancla «600 ⇔ 50:1» es ≈ 40,5:1 en OOT.
- **optbinning:** misma fórmula a 2,5e-7; trampa de especiales: −99 recibe 109,5 en vez de 94,7 puntos.

### 8.3 Crédito de motos: migrar una escala heredada

Caso típico en una fintech de financiamiento de motos: el motor de decisión usa un score antiguo en escala «PDO 40, 500 @ 20:1» con corte 480, y el nuevo modelo se entrega en la escala del curso. La tentación es poner «el corte que da la misma tasa de aprobación» o, peor, dejar 480.

1. **Qué significaba 480.** En la escala heredada: factor 57,71, offset 500 − 57,71·ln 20 = 327,12. $\ln O = (480 - 327{,}12)/57{,}71 = 2{,}649$ ⇒ odds 14,1:1 ⇒ **PD 6,6 %** (si la escala heredada estaba calibrada, cosa que hay que verificar primero).
2. **El mismo apetito en la escala nueva:** $s = 487{,}12 + 28{,}85\cdot 2{,}649 = $ **563,6**.
3. **Dos preguntas distintas.** «¿Mismo apetito?» ⇒ corte 563,6 sobre la PD calibrada del modelo nuevo. «¿Misma tasa de aprobación?» ⇒ el percentil correspondiente del score nuevo en la bandeja actual. Solo coinciden si ambos modelos están calibrados sobre la misma población; la diferencia entre ambas respuestas es información (swap-set, M18), no ruido.
4. **Elegir el ancla para la cartera.** Si la tasa de malos de motos ronda 10–15 %, con «600 @ 50:1» casi toda la cartera vive entre 500 y 580, y un PDO 20 con tabla entera deja pocos puntajes distintos en la zona del corte. Mover el ancla (p. ej. 600 @ 10:1) no cambia el modelo, pero centra la escala donde se decide; subir el PDO aumenta la resolución y reduce el costo relativo del redondeo. Ambas son decisiones de presentación que deben tomarse **una vez** y quedar fijas entre versiones, para que los usuarios del score no tengan que re-aprender la escala.

---

## 9. Preguntas de comité

**1. «¿Por qué 600, 50:1 y 20? ¿Quién lo decidió?»**
Son convenciones de presentación, no parámetros estadísticos: definen una recta sobre el log-odds (dos grados de libertad; el par score/odds base es un punto de ella). Se eligen para que la escala se parezca a la que la organización ya usa y para tener resolución suficiente en la zona del corte. Cambiarlas no cambia el orden, el Gini ni la PD de ningún cliente. Lo que sí debe estar documentado es que el ancla **no** es una afirmación de calibración.

**2. «La tabla dice que 600 es PD 2 %. ¿Lo podemos usar para provisionar?»**
No directamente. 600 ⇔ 50:1 es verdad en DEV por construcción; fuera de DEV depende de la calibración. En Austral, después del ajuste PIT (δ = 0,177), 600 corresponde a PD 2,33 %. La PD para provisiones sale del mapeo score → PD validado (master scale, M16), no del rótulo de la escala.

**3. «¿Por qué el intercepto aparece repartido en cada variable? ¿No distorsiona la lectura?»**
Es una convención de impresión: cualquier reparto da el mismo score. Lo que distorsiona es comparar puntos absolutos entre variables. Los rangos por variable (lo que se discute) y los reason codes por brecha son invariantes al reparto. Se declara el reparto en la ficha del scorecard.

**4. «En producción la tabla es entera. ¿Cuánto error introduce y a cuántos clientes afecta?»**
Error máximo n/2 puntos (4 en Austral, ≈ ±15 % en odds), típico ≈ 0,65 puntos (≈ ±2 % en odds), con un sesgo que depende de qué bins redondean en qué dirección. Afecta decisiones solo cerca del corte: ≈ densidad del score en el corte × error medio; en Austral, del orden de 30 de 8.585 solicitudes. La master scale y el corte se construyen sobre el score entero, que es el oficial.

**5. «Calibramos con δ. ¿Hay que re-firmar la tabla?»**
No. El score calibrado es el original menos δ·factor para todos (5,11 puntos en Austral). Se mueve el corte (o se define en PD) y se actualiza el mapeo score → PD. Re-escalar la tabla obliga a re-redondear y cambia decisiones sin razón de riesgo.

**6. «Hay puntos negativos en la tabla. ¿Es un error?»**
No, si la suma reproduce el logit. Aparecen con anclas bajas, PDO altos, muchas variables o bins extremos. Si el negocio no los acepta, se desplaza (mínimo cero + constante) y se documenta; truncar a 0 sí sería un error.

**7. «¿Qué pasa con un cliente cuyo valor cae en un bin que no existía en desarrollo, o con un código especial?»**
El artefacto debe tener una fila explícita para ese caso. «WoE 0» no es un valor universal: depende del reparto (71,8 puntos en Austral con partes iguales). Y algunas librerías asignan WoE 0 a los especiales por defecto aunque el especial tenga su propio WoE (T3): es un test obligatorio.

**8. «El validador dice que el PDO efectivo en OOT es 18,6. ¿Qué significa?»**
Que en OOT, 20 puntos ya no duplican exactamente las odds: la pendiente de calibración cambió. En DEV es exactamente 20 por construcción. Si el desvío es material, se recalibra con pendiente (δ₀, δ₁) en el mapeo a PD; la tabla no se toca salvo re-desarrollo.

---

## 10. Ejercicios

**E1. (Cálculo a mano)** Con PDO 20, 600 @ 50:1, β₀ = −3,0252 y n = 8, calcule factor, offset, base y los puntos del tramo «hasta 2» de `meses_desde_mora_12m` sabiendo que vale 27,1 puntos y β = −0,7675. ¿Cuál es su WoE?

<details><summary>Solución</summary>

factor = 20/ln 2 = 28,8539; offset = 600 − 28,8539·ln 50 = 487,1229; base = 487,1229/8 + 3,0252/8·28,8539 = 60,890 + 10,911 = 71,801. Pendiente = 0,7675·28,8539 = 22,145. WoE = (27,1 − 71,801)/22,145 = **−2,02**. Es el bin más riesgoso del scorecard: aporta −44,7 puntos respecto del neutro.
</details>

**E2. (Derivación)** Muestre que «600 @ 50:1, PDO 20» y «580 @ 25:1, PDO 20» definen la misma escala, y encuentre el score con odds 1:1.

<details><summary>Solución</summary>

El factor es el mismo (28,8539). Offset₁ = 600 − 28,8539·ln 50; offset₂ = 580 − 28,8539·ln 25 = 580 − 28,8539·(ln 50 − ln 2) = 580 − 28,8539·ln 50 + 20 = offset₁. Con odds 1:1, ln O = 0 ⇒ s = offset = 487,12.
</details>

**E3. (Conversión)** Un score heredado en escala «PDO 40, 500 @ 20:1» tiene corte 500. ¿Qué PD implica y qué corte equivalente tiene en la escala del curso?

<details><summary>Solución</summary>

Con odds base 20:1 en 500, el corte está justo en el ancla: odds 20:1, PD 1/21 = 4,76 %. En la escala del curso: $s = 487{,}12 + 28{,}85\cdot\ln 20 = 487{,}12 + 86{,}44 = 573{,}6$. Con la fórmula general: $s_A = o_A + (f_A/f_B)(s_B - o_B) = 487{,}12 + 0{,}5\cdot(500 - 327{,}12) = 573{,}6$.
</details>

**E4. (Reparto e invariancia)** Demuestre que el método de reason codes «brecha al promedio de la variable en la población» da los mismos motivos para cualquier reparto del intercepto, y dé un contraejemplo con el método «menores puntos obtenidos».

<details><summary>Solución</summary>

Con un cambio de reparto, $\text{puntos}(v,b) \to \text{puntos}(v,b) + \Delta_v$. El promedio de la variable $\bar r_v = \sum_b \pi_{v,b}\,\text{puntos}(v,b)$ pasa a $\bar r_v + \Delta_v$ (porque $\sum_b\pi_{v,b} = 1$). La brecha $\bar r_v - \text{puntos}(v,b(v))$ no cambia. Contraejemplo: dos variables, cliente con 60 puntos en A y 65 en B con partes iguales; con proporcional al rango, A recibe +20 de base y B −20: pasan a 80 y 45, y el motivo «menos puntos» cambia de A a B.
</details>

**E5. (Redondeo)** Un scorecard tiene 10 variables y PDO 20. (a) Cota del error del score por redondear la tabla. (b) Error típico bajo independencia. (c) Error relativo máximo y típico en odds. (d) Si la densidad del score en el corte es 0,6 % por punto y hay 50.000 solicitudes al mes, ¿cuántas decisiones cambian?

<details><summary>Solución</summary>

(a) 10/2 = 5 puntos. (b) sd = √(10/12) = 0,913; E|ε| ≈ 0,913·0,798 = 0,728. (c) Máximo e^{5/28,85} − 1 = 18,9 %; típico e^{0,728/28,85} − 1 = 2,6 %. (d) 50.000 × 0,006 × 0,728 ≈ 218 al mes (≈ 0,44 %). Es una estimación: si los errores por bin tienen sesgo, E|ε| puede ser mayor (en el notebook, 0,72 contra 0,52 teórico con 5 variables).
</details>

**E6. (Recalibración con pendiente)** Tras una calibración logística se obtiene $\text{logit}(p^{cal}) = 0{,}2 + 0{,}9\,\eta$. Con la escala del curso, escriba el score calibrado como función del score original y convierta el corte calibrado 560.

<details><summary>Solución</summary>

$s^{cal} = \delta_1 s + (1-\delta_1)\,\text{offset} - \delta_0\,\text{factor} = 0{,}9\,s + 0{,}1\cdot 487{,}12 - 0{,}2\cdot 28{,}85 = 0{,}9\,s + 48{,}71 - 5{,}77 = 0{,}9\,s + 42{,}94$. Corte: $0{,}9\,s + 42{,}94 = 560 \Rightarrow s = 574{,}5$ en la escala original. Nótese que con $\delta_1 < 1$ la escala calibrada tiene PDO efectivo 20/0,9 = 22,2 respecto del score original.
</details>

**E7. (Puntos negativos)** Con β₀ = −2,06, n = 5, odds base 50:1 y PDO 40, ¿qué score base mínimo garantiza que una fila con aporte −60 puntos no sea negativa con partes iguales?

<details><summary>Solución</summary>

Se requiere $C/n \ge 60 \Rightarrow C \ge 300$. $C = s_0 - \text{factor}(\ln 50 + \beta_0) = s_0 - 57{,}71\cdot(3{,}912 - 2{,}06) = s_0 - 106{,}9$. Entonces $s_0 \ge 406{,}9$. (Con los datos del notebook y PDO 40, la grilla muestra el mínimo en −1,1 con base 400 y 38,9 con base 600: consistente.)
</details>

**E8. (Código)** Escriba una función `pdo_efectivo(score, y, pdo, factor, offset)` que ajuste una logística de `y` sobre `(score − offset)/factor` y devuelva el PDO efectivo. Demuestre (a mano o con un test) que en la muestra de desarrollo devuelve exactamente `pdo`.

<details><summary>Solución</summary>

```python
def pdo_efectivo(score, y, pdo, factor, offset):
    x = (np.asarray(score) - offset) / factor        # x = -eta
    a, b = irls_logit(x[:, None], np.asarray(y))      # logit(y) = a + b·x
    return -pdo / b, a
```
En DEV, $x = -\eta$ y $(a,b) = (0,-1)$ satisface las ecuaciones de verosimilitud (ver §3.8), que por concavidad estricta tienen solución única ⇒ PDO efectivo = PDO. El notebook lo verifica con `assert np.isclose(pdo_dev, pdo_ui.value, rtol=1e-6)`.
</details>

**E9. (Diseño)** Diseñe los tests de CI que impedirían que la trampa T3 (especiales neutralizados) llegue a producción.

<details><summary>Solución</summary>

(1) Para cada fila del artefacto con `tipo_bin = especial`, verificar `puntos == base_v + pendiente_v·woe` con el WoE publicado (no 0) salvo que la ficha declare explícitamente «especial neutro». (2) Set dorado con un cliente sintético por código especial: su score por lookup debe igualar `offset − factor·η` con el WoE empírico. (3) Assert de que cada código especial declarado en el contrato de datos tiene una fila propia (sin agrupar −9 con −99 salvo decisión documentada). (4) Diff de puntos entre versiones: una fila especial que pasa a los puntos neutros exactos es una alerta.
</details>

---

## 11. Referencias

- **Siddiqi, N. (2006).** *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley. — La referencia clásica del scaling PDO/odds y de la fórmula con reparto en partes iguales; capítulo de «scorecard scaling».
- **Siddiqi, N. (2017).** *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2.ª ed.). Wiley. — Versión actualizada; útil para la práctica de implementación y el lado de negocio de las anclas.
- **Thomas, L. C., Edelman, D. B. y Crook, J. N. (2002).** *Credit Scoring and Its Applications.* SIAM (2.ª ed. con Thomas, Crook y Edelman, 2017 — verificar edición). — Fundamentos de log-odds scores y su relación con la PD; más formal que Siddiqi.
- **Anderson, R. (2007).** *The Credit Scoring Toolkit: Theory and Practice for Retail Credit Risk Management and Decision Automation.* Oxford University Press. — Muy completo en escalas, conversiones y operación de scorecards en motores de decisión.
- **Refaat, M. (2011).** *Credit Risk Scorecards: Development and Implementation Using SAS.* (verificar editorial). — Implementación paso a paso del scaling en código; útil como contraste de convenciones.
- **Baesens, B., Rösch, D. y Scheule, H. (2016).** *Credit Risk Analytics: Measurement Techniques, Applications, and Examples in SAS.* Wiley. — Contexto de calibración y uso de scores en PD regulatoria; puente a M15–M16.
- **Navas-Palencia, G. (2020).** «Optimal binning: mathematical programming formulation». arXiv:2001.08025. — El paper detrás de `optbinning`; la documentación de `Scorecard` (métodos `pdo_odds`, `min_max`, `intercept_based`, `rounding`) complementa lo visto en §4 y §7. Las afirmaciones sobre valores por defecto se verificaron en optbinning 1.0.0 leyendo el código; verificar en su versión.
- **CFPB — Regulation B, 12 CFR 1002.9 y su comentario oficial (Supplement I, 9(b)(2)).** — Para EE.UU.: los motivos deben referirse a factores efectivamente puntuados; más de cuatro «probablemente no ayuda»; acepta identificar los factores donde el solicitante quedó más lejos del puntaje promedio del factor (de todos los solicitantes, o de los aprobados en torno al corte). Base del método «brecha a la media» de §3.3. Para Chile no afirmo un requisito equivalente específico sin verificar la norma vigente (ver Serie 1 · E6).
- **Serie 1 · M7** (WoE/IV y convención de signo), **Serie 1 · E2** (calibración PIT/TTC), **Serie 2 · M10** (logística sobre WoE), **M14** (reason codes), **M15–M16** (calibración y master scale), **M21** (artefacto congelado y paridad).

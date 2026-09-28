# M12 · Poder discriminante: ROC, AUC, Gini, KS, CAP y su incertidumbre

> **Ficha**
> - **Clases que profundiza:** clase 3 (Gini por muestra, rangos típicos, reglas de caída 0,10/0,15, tabla de rendimiento por deciles) y clase 5 parte 1 (AUC como probabilidad, KS, bootstrap B = 1.000 del Gini y de su caída, deciles/lift, caída relativa 20%/30%).
> - **Prerrequisitos:** Serie 1 · M4 (esquema muestral), M7 (WoE/IV), E2 (calibración, PIT vs TTC), E3 (bootstrap para carteras chicas). Serie 2 · M10 (logística sobre WoE) y M11 (selección de variables).
> - **Archivos:** `M12_discriminacion.md` (este documento) · `M12_discriminacion.py` (notebook Marimo: `marimo edit --sandbox M12_discriminacion.py`) · `M12_tabla_deciles_gini.xlsx` (tabla de deciles, Gini por trapecios, Hanley–McNeil, caída y ruido por nº de malos).
> - **Tiempo estimado:** 3–4 h de lectura con derivaciones + 1,5 h de notebook + 1 h de ejercicios.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**Clase 3.** El modelo de Banco Austral (8 variables, logística sobre WoE, PDO 20) se midió por muestra con el Gini = 2·AUC − 1: **0,757 en DEV, 0,698 en HO y 0,675 en OOT**. La caída DEV→HO (0,059) se comparó con la regla de 0,10 («¿memorizó?») y la DEV→OOT (0,082) con la de 0,15 («¿envejece?»). Se dio una tabla de rangos típicos: admisión solo con bureau 0,35–0,55; admisión con historial interno 0,50–0,70; comportamiento 0,60–0,80; bajo 0,30 no sirve para decidir; sobre 0,85 es sospechoso (fuga o target trivial), con la advertencia correcta de que son órdenes de magnitud y que un Gini se compara contra el modelo anterior y el benchmark del bureau. La tabla de rendimiento por deciles con cortes de DEV mostró tasas de 28,2% en el decil 1 a 0% en los dos últimos; los deciles 1–2 concentran 75% de los malos, y las inversiones en HO/OOT (de 1 punto con 5–9 malos por decil) se leyeron como ruido.

**Clase 5.** AUC como probabilidad de que un malo al azar tenga peor score que un bueno al azar; KS = máx |F_malos − F_buenos| = **0,545 en OOT, alcanzado en el score 560 (banda C2), «cerca del cutoff: no es casualidad»**; las tres métricas son invariantes al δ de calibración de la clase 4. La caída se expresó en relativo: 7,8% en HO y 10,8% en OOT (1 − 0,675/0,757), con semáforo amarillo sobre 20% y rojo sobre 30%. El bootstrap (remuestrear la muestra de evaluación, mismo n, B = 1.000, semillas fijas y distintas por muestra, sin reajustar el modelo) dio:

| Muestra | n · malos | AUC | Gini | IC 95% Gini | KS | IC 95% KS |
|---|---|---|---|---|---|---|
| DEV | 3.322 · 165 | 0,8785 | 0,757 | [0,709; 0,799] | 0,597 | [0,565; 0,655] |
| HO | 1.397 · 78 | 0,8490 | 0,698 | [0,617; 0,769] | 0,553 | [0,487; 0,651] |
| OOT | 2.004 · 119 | 0,8375 | 0,675 | [0,600; 0,744] | 0,545 | [0,473; 0,627] |

Caída DEV − OOT +0,082, IC [−0,006; +0,167]; HO − OOT +0,023, IC [−0,083; +0,126]: ninguna concluyente, y ambas compatibles con caídas relevantes («no rechazo» no es «demostré estabilidad»). La tabla de deciles OOT (decil 1 con 57 de 119 malos: captura 47,9%, lift 4,8; al tercer decil, 79%) cerró el bloque.

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **Empates.** El AUC se calculó con `roc_auc_score` sin decir que trata los empates como ½, ni cuánto cambia el Gini si se mide sobre bandas en vez del score.
2. **Por qué AR = Gini.** Se mostró la CAP, pero la igualdad con 2·AUC − 1 se afirmó sin derivar.
3. **Dónde se alcanza el KS y por qué.** «No es casualidad» quedó como intuición. La respuesta formal (razón de verosimilitudes = 1, PD = tasa de malos) y su relación con el cutoff óptimo (que en general **no** coincide) quedaron fuera.
4. **El techo.** No se discutió que existe un Gini máximo alcanzable dado el riesgo real, ni que ese techo **depende de la tasa de malos**.
5. **Fórmulas cerradas de varianza.** Solo bootstrap. Ni Hanley–McNeil ni DeLong, que es el estándar para comparar dos modelos sobre la misma muestra.
6. **Comparación de modelos.** «¿El nuevo es mejor?» requiere un test pareado; comparar dos IC que se solapan es incorrecto.
7. **Las reglas 0,10 / 0,15 / 20% / 30% son convenciones** sin base inferencial: no dependen del número de malos ni del nivel del Gini.
8. **Optimismo vs deterioro** se nombró bien (DEV→HO vs HO→OOT), pero sin cuantificar el optimismo ni mostrar que un deterioro de nivel puede mover el Gini sin cambiar el ranking.
9. **Métricas de negocio** (tasa de malos a una tasa de aprobación dada) vs AUC; AUC parcial.
10. **Relación IV ↔ Gini.**

---

## 2. Intuición

**El AUC es un torneo.** Toma todos los pares (malo, bueno) posibles. En cada par, el modelo «gana» si le asigna más riesgo al malo, empata si les asigna lo mismo, pierde si se equivoca. AUC = fracción de victorias contando los empates como media victoria. El Gini es **victorias menos derrotas** sobre el total de pares: 0 es una moneda, 1 es no perder nunca. No mide si la PD está bien (nivel), solo si el orden es correcto; por eso el δ de la clase 4 no lo toca.

**El KS es el mejor corte único.** Si solo pudieras trazar una línea en el score, ¿dónde separarías mejor las distribuciones de buenos y malos? Ahí donde la proporción acumulada de malos le saca más ventaja a la de buenos. Esa línea está donde las dos densidades se cruzan: un cliente con ese score tiene exactamente la misma verosimilitud de venir de cualquiera de los dos grupos, y por tanto su PD es la tasa de malos promedio. No es el cutoff que maximiza la utilidad del negocio salvo coincidencia.

**La CAP es la vista del gerente.** «Si rechazo al 20% peor, ¿qué fracción de los malos me saco de encima?» El AR compara esa curva con la del modelo perfecto. Es el mismo número que el Gini.

**Hay un techo.** Aun conociendo la PD exacta de cada cliente, el default es una moneda cargada. Un cliente de PD 30% es bueno 7 de cada 10 veces; el modelo perfecto lo pondrá por encima de muchos buenos con PD 5% que luego resultan malos. El Gini alcanzable lo fija la **dispersión del riesgo real** de la población, no la técnica. Por eso un Gini > 0,85 en admisión es más probable que sea fuga que genio.

**Las métricas son variables aleatorias.** El Gini de OOT de Banco Austral no es 0,675: es 0,675 ± 0,07. Con 119 malos, cada malo «pesa» casi un punto de captura. Una regla que no mire cuántos malos hay es ciega al tamaño de su propio ruido.

**Comparar dos modelos no es comparar dos intervalos.** Si los dos modelos se evalúan sobre los mismos clientes, sus errores están correlacionados (los clientes difíciles son difíciles para ambos). La diferencia se conoce **mucho** mejor que cada AUC por separado.

---

## 3. Formalización

### 3.1 Notación

Sea $s$ un **score de riesgo** (más alto = peor; en el notebook, la PD). $M$ = malos ($y=1$), $B$ = buenos, $n_M$, $n_B$ sus tamaños, $\pi = n_M/(n_M+n_B)$. $F_M$, $F_B$ son las distribuciones de $s$ en cada grupo. Para un corte $t$ que rechaza $s \ge t$:

$$\text{TPR}(t) = P(S_M \ge t), \qquad \text{FPR}(t) = P(S_B \ge t).$$

La curva ROC es $\{(\text{FPR}(t), \text{TPR}(t))\}_t$. Con el score del curso (más puntos = menos riesgo) basta tomar $s = -\text{score}$; todas las métricas son idénticas.

### 3.2 AUC = probabilidad de concordancia

Caso continuo. Parametrizamos la ROC por $t$; cuando $t$ baja de $+\infty$ a $-\infty$, FPR sube de 0 a 1 con $d\,\text{FPR}(t) = f_B(t)\,dt$ (en valor absoluto). Entonces

$$\text{AUC} = \int_0^1 \text{TPR}\, d\text{FPR} = \int_{-\infty}^{\infty} P(S_M \ge t)\, f_B(t)\, dt = P(S_M \ge S_B) = P(S_M > S_B),$$

donde el último paso usa que en el caso continuo $P(S_M = S_B) = 0$ y que $S_M$, $S_B$ son independientes.

Caso discreto (empates). Ordena los valores distintos del score de mayor a menor riesgo. En un grupo de empate con $\Delta_M$ malos y $\Delta_B$ buenos, la ROC empírica avanza en diagonal de $(x_0, y_0)$ a $(x_0 + \Delta_B/n_B,\; y_0 + \Delta_M/n_M)$. El área del trapecio bajo ese tramo es

$$\frac{\Delta_B}{n_B}\Big(y_0 + \frac{1}{2}\frac{\Delta_M}{n_M}\Big) = \frac{1}{n_M n_B}\Big(\underbrace{n_M y_0\,\Delta_B}_{\text{pares malo más riesgoso que bueno}} + \tfrac12\underbrace{\Delta_M \Delta_B}_{\text{pares empatados}}\Big).$$

Sumando sobre grupos:

$$\boxed{\text{AUC} = \frac{1}{n_M n_B}\sum_{i\in M}\sum_{j\in B}\Big[\mathbb 1(s_i > s_j) + \tfrac12\,\mathbb 1(s_i = s_j)\Big]}$$

**La regla de trapecios sobre la ROC es exactamente la convención ½ para empates.** Contar empates como 0 equivale a la ROC «en escalera por abajo»; como 1, «por arriba». La diferencia entre ambas es $P(S_M = S_B)$, que con 8 bandas es grande: en el notebook (§2, OOT) el AUC va de 0,709 a 0,818 según la convención.

### 3.3 AUC = U de Mann–Whitney / $(n_M n_B)$

Asigna rangos medios (*midranks*) a las $n = n_M + n_B$ observaciones combinadas, de menor a mayor riesgo. El rango medio de la observación $i$ es

$$r_i = 1 + \#\{k: s_k < s_i\} + \tfrac12\big(\#\{k: s_k = s_i\} - 1\big).$$

Suma sobre los malos, $R_M = \sum_{i\in M} r_i$, y separa cada conteo en «contra malos» y «contra buenos»:

$$R_M = n_M + \sum_{i\in M}\Big[\#\{k\in M: s_k<s_i\} + \tfrac12(\#\{k\in M: s_k=s_i\}-1)\Big] + \sum_{i\in M}\Big[\#\{j\in B: s_j<s_i\} + \tfrac12\#\{j\in B: s_j=s_i\}\Big].$$

El segundo término cuenta cada **par de malos** exactamente una vez (el par estricto aporta 1 al mayor; el par empatado aporta ½ a cada uno): vale $\binom{n_M}{2}$. El tercero es, por definición, $U_M$. Luego

$$U_M = R_M - n_M - \frac{n_M(n_M-1)}{2} = R_M - \frac{n_M(n_M+1)}{2}, \qquad \text{AUC} = \frac{U_M}{n_M n_B}.$$

Esto da un algoritmo $O(n\log n)$ (ordenar) en vez de $O(n_M n_B)$. Es el resultado de Bamber (1975). El notebook lo implementa (`auc_rangos`) y verifica contra conteo de pares, `sklearn` y `scipy.stats.mannwhitneyu` hasta $10^{-12}$.

### 3.4 Gini = D de Somers

La D de Somers de $s$ respecto de $y$ es (pares concordantes − discordantes) / (pares con $y$ distinto). Con $p_> = P(S_M > S_B)$, $p_< = P(S_M < S_B)$, $p_= = P(S_M = S_B)$ y $p_> + p_< + p_= = 1$:

$$D = p_> - p_< = \big(p_> + \tfrac12 p_=\big) - \big(p_< + \tfrac12 p_=\big) = \text{AUC} - (1 - \text{AUC}) = 2\,\text{AUC} - 1.$$

El «Gini» de la industria de crédito **es** la D de Somers. Ojo: la tau-b de Kendall entre $s$ y $y$ binaria **no** es el Gini (normaliza distinto).

### 3.5 AR de la CAP = Gini (demostración)

La CAP pone en el eje horizontal la fracción de población rechazada y en el vertical la fracción de malos capturada: para cada corte $t$,

$$x(t) = \pi\,\text{TPR}(t) + (1-\pi)\,\text{FPR}(t), \qquad y(t) = \text{TPR}(t).$$

Área bajo la CAP (integral de Stieltjes, válida también con empates si ambas curvas interpolan linealmente):

$$A_{\text{CAP}} = \int y\, dx = \pi\int \text{TPR}\, d\text{TPR} + (1-\pi)\int \text{TPR}\, d\text{FPR} = \frac{\pi}{2} + (1-\pi)\,\text{AUC}.$$

(El primer término: $\int_0^1 u\,du = 1/2$.) El área entre la CAP y la diagonal es $a_R = A_{\text{CAP}} - \tfrac12 = (1-\pi)(\text{AUC} - \tfrac12)$. El modelo perfecto sube linealmente hasta $(\pi, 1)$ y sigue plano: su área es $1 - \pi/2$, luego $a_P = \tfrac12 - \tfrac{\pi}{2} = \tfrac{1-\pi}{2}$. Por tanto

$$\text{AR} = \frac{a_R}{a_P} = \frac{(1-\pi)(\text{AUC}-\tfrac12)}{(1-\pi)/2} = 2\,\text{AUC} - 1. \qquad\blacksquare$$

El resultado aparece en Engelmann, Hayden y Tasche (2003) y en el BCBS WP 14 (2005). La normalización por $(1-\pi)/2$ importa: quien normalice por ½ obtiene un número que depende de la tasa de malos.

### 3.6 KS: dónde se alcanza, qué optimiza y sus cotas

$$\text{KS} = \sup_t\,|F_M(t) - F_B(t)| = \sup_t\,\big(\text{TPR}(t) - \text{FPR}(t)\big)$$

para un score que ordena en el sentido correcto. Es el índice de Youden (1950) máximo.

**Dónde.** Con densidades, la condición de primer orden de $\text{TPR}(t) - \text{FPR}(t)$ es $-f_M(t) + f_B(t) = 0$, o sea $f_M(t^*) = f_B(t^*)$: la razón de verosimilitudes vale 1. Por Bayes,

$$P(\text{malo}\mid s = t^*) = \frac{\pi f_M(t^*)}{\pi f_M(t^*) + (1-\pi) f_B(t^*)} = \pi.$$

**El KS se alcanza donde la PD verdadera iguala la tasa de malos de la muestra.** En el notebook (§4): en DEV, donde la logística reproduce la tasa media por construcción, el KS está en PD = 0,123 con tasa 0,113. En OOT el KS está en PD del modelo 0,128 con tasa observada 0,154, porque el deterioro macro plantado hace que el modelo subestime el nivel. El punto sigue estando donde la PD **verdadera** vale π.

**Qué optimiza (y qué no).** Si rechazar a un bueno cuesta $c_B$ (margen perdido) y aprobar a un malo cuesta $c_M$ (pérdida), el costo esperado por solicitud de rechazar $s \ge t$ es

$$C(t) = \pi\, c_M\,(1 - \text{TPR}(t)) + (1-\pi)\, c_B\,\text{FPR}(t).$$

Derivando e igualando a 0: $\pi c_M f_M(t) = (1-\pi) c_B f_B(t)$, es decir, $P(\text{malo}\mid t) = c_B/(c_B + c_M)$: el **cutoff óptimo está donde la PD iguala la PD de equilibrio** $\text{PD}^* = c_B/(c_B+c_M)$ (ver M17). El corte del KS coincide con el óptimo solo si $\text{PD}^* = \pi$, es decir, si $c_M/c_B = (1-\pi)/\pi$. Con π = 6% eso exige que un malo cueste 15,7 veces lo que rinde un bueno: plausible en consumo de margen bajo, muy lejos en productos de margen alto. «KS cerca del cutoff» en Banco Austral es una coincidencia económica, no una ley. Lo que sí es cierto es que en la zona del KS la decisión es más «disputada» (densidades iguales).

**Cotas KS ↔ Gini.** Sea $D$ = KS alcanzado en $(x^*, x^* + D)$ de la ROC.

- *Cota superior (siempre).* Para todo $x$, $\text{ROC}(x) \le \min(1, x + D)$. Entonces
$$\text{AUC} \le \int_0^{1-D}(x+D)\,dx + \int_{1-D}^1 1\,dx = \frac{(1-D)^2}{2} + D(1-D) + D = \frac12 + D - \frac{D^2}{2},$$
luego $\text{Gini} \le 2D - D^2 = D(2 - D)$.
- *Cota inferior con ROC cóncava.* Una ROC cóncava queda sobre el polígono $(0,0) \to (x^*, x^*+D) \to (1,1)$, cuya área es
$$\frac{x^*(x^*+D)}{2} + \frac{(1-x^*)(x^*+D+1)}{2} = \frac{1+D}{2},$$
luego $\text{Gini} \ge D$.
- *Sin concavidad* solo vale $\text{AUC} \ge (1-x^*)(x^*+D) \ge D$, o sea $\text{Gini} \ge 2D - 1$.

Banco Austral OOT: KS 0,545 ⇒ Gini ∈ [0,545; 0,793] si la ROC es cóncava; el observado 0,675 cae dentro. La ROC es cóncava cuando el score es monótono en la razón de verosimilitudes, que es lo que un modelo bien especificado intenta. Si una muestra viola $\text{Gini} \ge \text{KS}$, hay un tramo del score que ordena al revés.

### 3.7 El techo: el Gini de la PD verdadera

Supón que conoces $p_i$ exacta y $y_i \sim \text{Bernoulli}(p_i)$ independientes. Para un par $(i, j)$, la probabilidad de que $i$ sea malo y $j$ bueno es $p_i(1-p_j)$. Ordenando por $p$, el número esperado de pares concordantes (+½ empates) y el de pares (malo, bueno) son

$$E[C] = \sum_{i \ne j} p_i(1-p_j)\big[\mathbb 1(p_i>p_j) + \tfrac12\mathbb 1(p_i=p_j)\big], \qquad E[n_M n_B] = \sum_{i\ne j} p_i(1-p_j).$$

El **AUC techo** es $\text{AUC}^* \approx E[C]/E[n_M n_B]$ (razón de esperanzas; converge a la esperanza de la razón cuando $n\to\infty$). Ningún modelo que use información disponible en $t_0$ lo supera en expectativa: ordenar por la PD verdadera es el orden óptimo (lema de Neyman–Pearson: la razón de verosimilitudes es monótona en $p$).

**Modelo logit-normal.** Si el log-odds verdadero es $\eta \sim N(\mu, \sigma^2)$ y los eventos son raros ($\mu \to -\infty$), $p \approx e^{\eta}$. La densidad de $\eta$ entre los malos es proporcional a $e^\eta\varphi\big((\eta-\mu)/\sigma\big)$; completando el cuadrado, eso es $N(\mu + \sigma^2, \sigma^2)$, mientras que los buenos siguen ≈ $N(\mu, \sigma^2)$. Dos normales de igual varianza separadas por $\sigma^2$:

$$\text{AUC}^* \to \Phi\Big(\frac{\sigma^2}{\sigma\sqrt2}\Big) = \Phi\Big(\frac{\sigma}{\sqrt2}\Big).$$

Con σ = 1, el techo de eventos raros es Gini 0,52; con σ = 2, 0,84. Fuera del límite el techo es **menor** y **baja cuando sube la tasa media**: con σ = 2, μ = −4 (tasa 6,7%) da 0,78, y μ = −1 (tasa 35%) da 0,73. La razón: cuando las PD suben, hay más buenos con PD alta, que «parecen» malos, y el peso $p_i(1-p_j)$ de los pares mal ordenados crece.

**Consecuencia práctica (demostrada en el notebook, §9b).** Un choque macro que suma una constante al log-odds de todos no reordena a ningún cliente, pero baja el Gini esperado. En la cartera sintética, con deterioro 0 → 1,0 la tasa OOT pasa de 12,1% a 24,0%, el techo esperado de 0,618 a 0,588 y el Gini del modelo de 0,571 a 0,523, mientras la **brecha modelo–techo** queda en 0,043–0,046. Comparar Gini entre periodos con tasas de malos muy distintas mezcla ranking con nivel.

### 3.8 La varianza del AUC

**AUC como U-estadístico de dos muestras.** Con $\psi(x, z) = \mathbb 1(x>z) + \tfrac12\mathbb 1(x=z)$, $\widehat{\text{AUC}} = \frac{1}{n_M n_B}\sum_i\sum_j \psi(X_i, Z_j)$, con $X$ los scores de malos y $Z$ los de buenos. La descomposición de Hoeffding da, a primer orden,

$$\operatorname{Var}(\widehat{\text{AUC}}) \approx \frac{\xi_{10}}{n_M} + \frac{\xi_{01}}{n_B}, \qquad \xi_{10} = \operatorname{Var}_X\big(E_Z[\psi(X,Z)]\big),\ \ \xi_{01} = \operatorname{Var}_Z\big(E_X[\psi(X,Z)]\big).$$

**DeLong, DeLong y Clarke-Pearson (1988)** estiman $\xi_{10}$ y $\xi_{01}$ con las *componentes estructurales* (valores de colocación):

$$V_{10}(i) = \frac{1}{n_B}\sum_{j}\psi(X_i, Z_j), \qquad V_{01}(j) = \frac{1}{n_M}\sum_i \psi(X_i, Z_j),$$

que son «la fracción de buenos que el malo $i$ supera» y «la fracción de malos que superan al bueno $j$». Entonces $\widehat{\text{AUC}} = \overline{V_{10}} = \overline{V_{01}}$ y

$$\widehat{\operatorname{Var}}(\widehat{\text{AUC}}) = \frac{S_{10}}{n_M} + \frac{S_{01}}{n_B}, \qquad S_{10} = \frac{1}{n_M-1}\sum_i (V_{10}(i) - \widehat{\text{AUC}})^2,\ \ S_{01}\ \text{análogo}.$$

**Versión rápida (Sun y Xu, 2014).** Con $T_Z$ = rangos medios en la muestra combinada, $T_X$ = rangos medios dentro de los malos y $T_Y$ = dentro de los buenos, por el mismo argumento de 3.3:

$$V_{10}(i) = \frac{T_Z(X_i) - T_X(X_i)}{n_B}, \qquad V_{01}(j) = 1 - \frac{T_Z(Z_j) - T_Y(Z_j)}{n_M}.$$

$T_Z(X_i) - T_X(X_i)$ es exactamente el número de buenos por debajo de $X_i$ más ½ de los empatados. Costo $O(n \log n)$. El notebook implementa ambas y verifica que coinciden a precisión de máquina.

**Dos modelos sobre las mismas observaciones.** Con $k$ scores, $V_{10}^{(r)}$ y $V_{01}^{(r)}$ para cada uno; la matriz de covarianzas es $\mathbf S = \mathbf S_{10}/n_M + \mathbf S_{01}/n_B$, con $\mathbf S_{10}$ la covarianza muestral entre los vectores $V_{10}^{(r)}$. Para $H_0: \text{AUC}_1 = \text{AUC}_2$,

$$z = \frac{\widehat{\text{AUC}}_1 - \widehat{\text{AUC}}_2}{\sqrt{S_{11} + S_{22} - 2S_{12}}} \ \overset{\cdot}{\sim}\ N(0,1).$$

**Hanley y McNeil (1982)** proponen una fórmula cerrada que solo necesita $A$, $n_M$, $n_B$:

$$\operatorname{SE}^2(A) = \frac{A(1-A) + (n_M-1)(Q_1 - A^2) + (n_B-1)(Q_2-A^2)}{n_M n_B}, \quad Q_1 = \frac{A}{2-A},\ \ Q_2 = \frac{2A^2}{1+A}.$$

$Q_1$ es la probabilidad de que dos malos al azar superen a un bueno; $Q_2$, la de que un malo supere a dos buenos. Las expresiones cerradas salen de suponer distribuciones exponenciales, y fuera de ese supuesto son una aproximación. En el notebook, HM **sobreestima** el SE en torno a 10% frente a DeLong/bootstrap en la cartera sintética. En Banco Austral la diferencia es mayor: HM da SE(Gini) = 0,035 / 0,055 / 0,046 (DEV/HO/OOT) contra ≈ 0,023 / 0,039 / 0,037 que se infieren de los IC bootstrap del curso. Sirve para órdenes de magnitud y diseño muestral, no para un informe.

**Bootstrap.** Percentil (el del curso) o BCa. Remuestrear toda la muestra deja $n_M$ aleatorio (refleja también la incertidumbre de la tasa); el bootstrap **estratificado** (remuestrear malos y buenos por separado) fija $n_M$ y $n_B$ y es el análogo exacto de DeLong. Con $n_M$ del orden de cientos, las diferencias son de tercer decimal.

**SE del Gini** = 2·SE(AUC). Regla de bolsillo: cuando $n_B \gg n_M$, domina $S_{10}/n_M$, luego $\operatorname{SE}(\text{Gini}) \approx c/\sqrt{n_M}$ con $c = 2\sqrt{S_{10}}$. En la cartera sintética (Gini ≈ 0,57, π ≈ 11,5%), $c \approx 0{,}45$–$0{,}50$. En Banco Austral, con los IC del curso, $c \approx 0{,}30$–$0{,}40$ (Gini más alto, varianza menor).

### 3.9 El ruido de la caída y la potencia

Para dos muestras **independientes** (DEV vs OOT, HO vs OOT), $\operatorname{Var}(G_1 - G_2) = \operatorname{Var}(G_1) + \operatorname{Var}(G_2)$. Con igual número de malos y la regla de bolsillo:

$$\operatorname{SD}(G_1 - G_2) \approx \frac{\sqrt2\,c}{\sqrt{n_M}}, \qquad \text{umbral 95\% unilateral} \approx \frac{1{,}645\sqrt2\,c}{\sqrt{n_M}} \approx \frac{1{,}05}{\sqrt{n_M}}\quad(c=0{,}45).$$

Con 120 malos: 0,096; con 1.000: 0,033. La simulación del notebook (§8) da 0,096 y 0,037. Para **detectar** una caída real $\Delta$ con potencia $1-\beta$ al nivel $\alpha$ unilateral:

$$n_M \approx \frac{2\,(z_{1-\alpha} + z_{1-\beta})^2\, c^2}{\Delta^2}.$$

Con $c = 0{,}45$, α = 5%, potencia 80% ($z$ = 1,645 + 0,842): **Δ = 0,10 exige ≈ 250 malos por muestra; Δ = 0,05 exige ≈ 1.000**. La regla fija de 0,10 es, entonces, demasiado estricta con 30–120 malos (dispara por azar entre 19% y 4% de las veces según la simulación) y demasiado laxa con miles (una caída real de 0,06 con 1.000 malos está muy por sobre el ruido y la regla no la ve).

**La caída relativa** (20%/30% del curso) no arregla nada: el umbral absoluto implícito es 0,2·$G_{\text{DEV}}$, que depende del nivel del Gini y no de $n_M$. Peor: a mayor Gini, menor varianza (la $c$ baja), así que la regla relativa es **más laxa justo donde el Gini se mide mejor**.

### 3.10 Optimismo

Define el optimismo como $\omega = E[G_{\text{DEV}} - G_{\text{nuevo}}]$ cuando el modelo se ajusta en DEV y se evalúa en datos nuevos de la misma población. Es un sesgo, no ruido: tiene signo positivo en expectativa y crece con (grados de libertad efectivos) / $n_M$, incluidos los grados de libertad «ocultos» del binning, del WoE y de la selección de variables. La corrección de Efron (1983), popularizada por Harrell para modelos clínicos, estima $\omega$ por bootstrap: ajustas **todo el pipeline** (binning incluido) en una réplica bootstrap, mides $G$ en la réplica y en la muestra original, promedias la diferencia y la restas al $G$ aparente.

Resultados del notebook (§9a, 20 repeticiones, SE ≈ 0,01):

| malos en DEV | 7 variables reales | 7 reales + 10 de ruido |
|---|---|---|
| 40 | 0,173 | 0,412 |
| 100 | 0,065 | 0,206 |
| 250 | 0,018 | 0,089 |
| 600 | 0,018 | 0,042 |

Con 165 malos (Banco Austral DEV) y 8 variables, el optimismo esperado es del orden de 0,02–0,05. La caída DEV→HO de 0,059 del curso es coherente con eso **más** ruido. En el notebook, el modelo base tiene Gini HO 0,588 > DEV 0,533: el optimismo existe en expectativa, pero una realización puede tener cualquier signo.

### 3.11 IV ↔ Gini en el mundo binormal

Supón una variable $x$ con buenos $\sim N(d, 1)$ y malos $\sim N(0, 1)$. El WoE continuo es

$$\text{WoE}(x) = \ln\frac{f_B(x)}{f_M(x)} = \frac{-(x-d)^2 + x^2}{2} = d\,x - \frac{d^2}{2},$$

lineal en $x$ (la logística sobre WoE es exacta aquí). El IV continuo es la divergencia de Jeffreys:

$$\text{IV} = E_B[\text{WoE}] - E_M[\text{WoE}] = \Big(d^2 - \frac{d^2}{2}\Big) - \Big(0 - \frac{d^2}{2}\Big) = d^2.$$

Y $\text{AUC} = P(X_B > X_M) = P(N(d, 2) > 0) = \Phi(d/\sqrt2)$. Luego

$$\boxed{\text{Gini} = 2\,\Phi\big(\sqrt{\text{IV}/2}\big) - 1}\qquad(\text{solo binormal, igual varianza, una variable}).$$

IV 0,10 → Gini 0,18; IV 0,30 → 0,30; IV 0,50 → 0,38; IV 1,0 → 0,52. **Cautelas:** (i) el IV binneado es menor que el continuo (con 5 bins y d = 1, el notebook da 0,81 vs 1,0) y además tiene sesgo positivo con pocos datos; (ii) con varianzas distintas o colas asimétricas la relación cambia; (iii) el Gini de un modelo no es ninguna suma de los Gini de sus variables, porque las variables están correlacionadas. Sirve para una sola cosa: detectar un IV implausible (una variable sola con IV > 1,5 promete Gini > 0,6: sospecha de fuga, ver Serie 1 · M7).

### 3.12 Invariancia monótona

AUC, Gini, KS y AR dependen de los scores solo a través de los signos de $s_i - s_j$. Si $g$ es **estrictamente** creciente, $\text{sign}(g(s_i) - g(s_j)) = \text{sign}(s_i - s_j)$ y todas las métricas son idénticas. Eso incluye calibrar con δ en el logit (clase 4), pasar a puntos PDO, cualquier $\exp$ o rango medio. Si $g$ es solo débilmente creciente (redondear, bandear, truncar con `clip`), algunos pares estrictos pasan a empate y el AUC cambia (en general baja). Romper empates arbitrariamente (p. ej. `argsort`) también lo cambia. El notebook (§12) lo verifica sobre OOT: redondear el score a entero cambia el Gini en −0,0001, truncar la PD en [2%, 30%] en −0,003, y 8 bandas en −0,015.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| AUC / Gini (= AR = D de Somers) | Ranking global, independiente del nivel y del corte | $O(n\log n)$ | Métrica principal de discriminación; comparación entre versiones | Universal en scoring; BCBS WP14; guía de reporte del BCE (AUC) |
| KS | Separación máxima con un corte | $O(n\log n)$ | Complemento; lectura intuitiva; cartas de riesgo en EE.UU. y LatAm | Muy usado en scoring de consumo (EE.UU., Latinoamérica) y en reportes de bureau |
| CAP / AR | Vista de captura para el negocio | igual que AUC | Presentaciones a comité y a riesgo de negocio | Agencias de rating y bancos IRB (AR) |
| AUC parcial (McClish, 1989) | Discriminación solo en la zona operativa de FPR/aprobación | bajo | Cuando la política aprueba 70–90% y lo demás no importa | Medicina, algo en crédito; `sklearn` con `max_fpr` (estandarizado) |
| Tasa de malos a aprobación fija; curva de estrategia | La métrica que se traduce a pérdida | bajo | Comparar modelos para una política concreta | Riesgo de negocio, pricing (ver M17) |
| Lift / captura en el k% peor | Eficiencia de cobranza y campañas | bajo | Modelos de cobranza, prevención, retención | Cobranza, marketing |
| Medida H (Hand, 2009) | El AUC pondera los costos de error de forma incoherente entre modelos; H fija una distribución de costos | medio | Comparación académica o cuando se conoce la distribución de costos | Poco adoptada en banca |
| Brier / log-loss | Nivel + ranking a la vez | bajo | Cuando la PD se usa como probabilidad (pricing, IFRS 9) | Complemento en validación de PD (ver M15, M19) |
| SE de Hanley–McNeil | IC del AUC sin datos individuales | nulo | Diseño muestral y órdenes de magnitud | Literatura clínica clásica |
| DeLong (un modelo o pareado) | IC no paramétrico y comparación de AUC correlacionados | $O(n\log n)$ | Estándar para «¿el nuevo es mejor?» sobre la misma muestra | pROC en R (`roc.test`); literatura de validación |
| Bootstrap (percentil, BCa, estratificado) | IC de cualquier métrica (KS, lift, caída entre muestras) | $B\times$ costo métrica | Métricas sin fórmula; diferencias entre muestras | Curso (B = 1.000); práctica de validación |
| Test AUC inicial vs actual del BCE | Monitoreo supervisor de la discriminación | bajo | Bancos IRB en la zona euro | BCE, *Instructions for reporting the validation results of internal models* (feb. 2019) |

Sobre el test del BCE, que es el ejemplo regulatorio más concreto. Según las instrucciones de febrero de 2019 (verificado en el documento del BCE): se compara el AUC del desarrollo inicial con el del periodo actual mediante $S = (\text{AUC}_{\text{init}} - \text{AUC}_{\text{curr}})/s$, con $s$ el error estándar estimado del AUC actual, y p-valor **unilateral** $1 - \Phi(S)$. Cuando la PD final se obtiene mapeando el score a grados, el AUC se calcula **sobre los grados** (con todos sus empates). El detalle de cómo se estima $s$ está en un anexo que no pude leer completo; según entiendo, es un estimador tipo Mann–Whitney/DeLong (verificar con el anexo vigente). Para la CMF chilena no encontré una regla cuantitativa equivalente sobre el Gini para modelos internos de consumo. No afirmo que exista ni que no; verificar con la normativa vigente (ver Serie 1 · E6).

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 El Gini «de la master scale» vs «del score».**
- *Síntoma:* el informe de validación dice Gini 0,53 y el de desarrollo, 0,54, para el mismo modelo y la misma muestra.
- *Causa:* uno se calculó sobre 8 bandas y el otro sobre el score continuo. Bandear convierte pares ordenados en empates (½).
- *Detección:* reporta siempre el número de valores distintos del score evaluado. Recalcula sobre ambos.
- *Qué hacer:* declara en el contrato de la métrica sobre qué se calcula. Si el regulador pide grados (BCE), reporta ambos. La pérdida por bandear es información: si es grande, la master scale tiene pocas bandas o están mal cortadas (M16).

**5.2 Comparar Gini entre poblaciones o periodos con distinta tasa o dispersión de riesgo.**
- *Síntoma:* «el modelo de motos (Gini 0,45) es peor que el de consumo (0,62)», o «el Gini cayó en la recesión».
- *Causa:* el techo depende de la dispersión del riesgo real y de la tasa media (§3.7). Una población más homogénea (clientes nuevos de un canal, thin file) tiene un techo más bajo. Un choque de nivel baja el Gini sin que cambie el orden.
- *Detección:* mide junto al Gini la tasa de malos y la dispersión de la PD (p. ej. el ratio de PD entre los deciles 9 y 1). En simulación, compara contra el techo esperado.
- *Qué hacer:* compara siempre contra un benchmark **sobre la misma muestra** (modelo anterior, score de bureau). Esa diferencia pareada sí es interpretable.

**5.3 Reglas fijas de caída sin mirar el número de malos.**
- *Síntoma:* semáforo rojo en el segmento chico que desaparece el mes siguiente; semáforo verde eterno en el segmento grande que se degrada lentamente.
- *Causa:* el ruido de la caída escala como $1/\sqrt{n_M}$ (§3.9).
- *Detección:* tabla de ruido por nº de malos (notebook §8, hoja `Ruido` de la planilla).
- *Qué hacer:* umbrales en función de $n_M$ (percentil de la distribución nula, o test DeLong/bootstrap), ventanas acumuladas hasta un mínimo de malos, y un umbral de **materialidad** separado del de significancia.

**5.4 Pruebas múltiples en el monitoreo.**
- *Síntoma:* «cada tanto salta el Gini de algún segmento».
- *Causa:* 12 meses × 5 segmentos = 60 tests. Con α = 5% por test y tests independientes, esperas 3 alarmas falsas al año.
- *Qué hacer:* controla el error por familia (Bonferroni o, mejor, FDR de Benjamini–Hochberg), usa ventanas móviles de 3–6 meses, y exige persistencia (dos cortes seguidos) para escalar.

**5.5 Comparar dos modelos mirando si sus IC se solapan.**
- *Síntoma:* «los IC se solapan, el modelo nuevo no es mejor».
- *Causa:* se ignora la covarianza. En el notebook (§7), la correlación entre los AUC de dos modelos anidados es 0,89–0,996, y el SE de la diferencia ignorándola es **9 veces** mayor.
- *Qué hacer:* DeLong pareado o bootstrap pareado (mismos índices para ambos modelos en cada réplica). Además, lo opuesto: con correlación 0,995 una diferencia de 0,006 en Gini sale significativa (`carga_financiera` en el notebook) y no vale nada económicamente. Separa significancia de materialidad.

**5.6 Gini sospechosamente alto.**
- *Síntoma:* Gini > 0,85 en admisión; una variable con IV > 1,5.
- *Causa:* fuga temporal (variable medida después de $t_0$, como un «días de mora actual» que ya incorpora el default), target trivial (la definición de malo se deduce de una variable) o población contaminada (malos ya castigados en el origen).
- *Detección:* compara con el techo plausible (§3.7): Gini 0,85 exige log-odds con σ ≈ 3. Revisa el Gini por variable y el de cada variable en OOT vs DEV. Revisa el linaje temporal (Serie 1 · M2, M4).
- *Qué hacer:* no se «corrige»: se busca la fuga.

**5.7 KS por deciles y KS «en el cutoff».**
- *Síntoma:* el KS de la tabla de deciles es menor que el del software; «el KS dice que el cutoff debe ser 560».
- *Causa:* la grilla de 10 cortes no ve el máximo entre cortes (planilla: 0,529 vs 0,545 exacto). El KS maximiza TPR − FPR, que solo es el óptimo económico si $\text{PD}^* = \pi$ (§3.6).
- *Qué hacer:* reporta el KS exacto y su posición. El cutoff sale de la economía (M17), no del KS.

**5.8 El AUC promedia regiones que nadie usa; las ROC se cruzan.**
- *Síntoma:* el modelo con mejor AUC da peor tasa de malos en la zona de aprobación real.
- *Causa:* el AUC integra sobre todos los cortes. Si las ROC se cruzan, el ranking por AUC puede invertir el ranking operativo.
- *Detección:* curva de estrategia (tasa de malos vs aprobación) de ambos modelos; AUC parcial en la zona de FPR relevante.
- *Qué hacer:* decide con la métrica operativa en la zona de la política, y usa el AUC como control global.

**5.9 Gini agregado sobre segmentos heterogéneos.**
- *Síntoma:* Gini global 0,60, pero dentro de cada segmento (motos nuevas / usadas, canal) no pasa de 0,45.
- *Causa:* el Gini global incluye la discriminación **entre** segmentos (tasas distintas). Si la política decide dentro del segmento (cutoffs por segmento), el Gini relevante es el intra-segmento.
- *Qué hacer:* reporta Gini global e intra-segmento (con su $n_M$). No vendas como poder del modelo lo que es composición de la cartera.

**5.10 Bootstrap mal especificado.**
- *Síntoma:* IC imposiblemente estrechos para la caída; IC que cambian al reejecutar.
- *Causa:* la misma semilla en las dos muestras (réplicas acopladas, advertencia del curso); réplicas degeneradas sin malos; reajustar el modelo en cada réplica cuando se quería la incertidumbre de la métrica (o no reajustar cuando se quería el optimismo); pocas réplicas para percentiles extremos.
- *Qué hacer:* semilla declarada y distinta por muestra; B ≥ 1.000 para percentiles 2,5/97,5 en informes; decide explícitamente **qué** incertidumbre mides: la de la métrica con el modelo fijo (curso) o la del procedimiento de ajuste (optimismo).

**5.11 Leer el optimismo como deterioro (y viceversa).**
- *Síntoma:* «cayó 0,08 de DEV a OOT: el modelo envejeció».
- *Causa:* DEV→OOT = optimismo + deterioro + ruido. Solo HO→OOT aísla el tiempo, y con 78 y 119 malos su SD es ≈ 0,05.
- *Qué hacer:* reporta HO→OOT como la caída temporal, con su IC. Estima el optimismo aparte (bootstrap de Efron) si quieres usar DEV como referencia.

---

## 6. Puente con ingeniería

Una métrica de discriminación en producción es una **función pura con contrato**, no una línea de notebook. En un pipeline declarativo, la validación y el monitoreo son nodos cuyo input es (predicciones congeladas, resultados maduros, metadatos de muestra) y cuyo output es un registro versionado.

**Contrato de la función de métrica** (lo que el nodo garantiza y lo que exige):

```yaml
metrica: auc_gini_ks
version: 1.2.0
entrada:
  y: {tipo: int8, valores: [0, 1], significado: "1 = malo (90+ a 12m)", nulos: prohibidos}
  s: {tipo: float64, sentido: "mayor = más riesgo", nulos: prohibidos}
precondiciones:
  - "0 < sum(y) < len(y)"            # ambas clases presentes
  - "len(y) == len(s)"
convenciones:
  empates: "1/2"                      # trapecios = Mann-Whitney
  sobre: "score continuo"             # o "grados de la master scale": se DECLARA
salida: {auc, gini, ks, ks_umbral, n, n_malos, n_valores_distintos, se_delong, ic95_delong}
determinista: true                    # DeLong no tiene semilla; el bootstrap declara semilla y B
```

**Tests tipo CI** (propiedades, no solo valores; el notebook tiene la mayoría en su celda final):

- `auc_rangos == auc_conteo == sklearn == scipy` sobre datos aleatorios con y sin empates (`hypothesis` o semillas fijas).
- Simetría: `auc(1 - y, -s) == auc(y, s)`; y `auc(y, -s) == 1 - auc(y, s)`.
- Invariancia: `auc(y, g(s)) == auc(y, s)` para $g$ estrictamente creciente (logit + δ, puntos PDO, exp).
- Identidades: `AR_CAP == 2*AUC - 1 == somersD`; `KS_numpy == ks_2samp == max(TPR - FPR)`; `KS <= Gini <= KS*(2-KS)` cuando la ROC es cóncava (warning si no se cumple, no error: indica un tramo que ordena al revés).
- DeLong: definición $O(n^2)$ = versión rápida en muestras chicas; SE DeLong / SE bootstrap ∈ [0,8; 1,25] (test estadístico con tolerancia).
- Permutación: reordenar filas no cambia nada; duplicar el dataset no cambia el AUC y reduce el SE en $\sqrt2$.
- Tabla de deciles: $\sum n$ = n, $\sum$ malos = malos; KS por decil ≤ KS exacto.

**Qué se congela y qué se versiona.**

- Se congelan: las predicciones del modelo sobre DEV/HO/OOT (con el hash del artefacto que las produjo, ver M21); los **cortes de deciles de DEV** cuando la tabla de rendimiento se usa para comparar muestras (clase 3), a diferencia de los cortes por cuantiles de la propia muestra (tabla OOT de la clase 5), que miden captura pero no son comparables entre periodos. Hay que declarar cuál se usa.
- Se versionan: la definición de la métrica (convención de empates, sobre qué score), las semillas y B del bootstrap, y los **umbrales como función de $n_M$**, no como constantes.

**Umbrales declarativos** (el monitoreo como configuración, no como código disperso):

```yaml
monitoreo_discriminacion:
  referencia: HO                       # no DEV: DEV trae optimismo
  ventana_minima_malos: 250            # acumula meses hasta alcanzarlo
  regla_estadistica:
    test: delong_independiente         # HO vs ventana actual
    alfa_unilateral: 0.05
    correccion_multiple: benjamini_hochberg
  regla_materialidad:
    caida_gini_amarilla: 0.05
    caida_gini_roja: 0.10
  semaforo: "rojo si (p < alfa y caida > roja); amarillo si (p < alfa y caida > amarilla) o dos cortes seguidos con p < 0.10"
  reportar: [gini, ic95, n_malos, tasa_malos, n_valores_distintos, gini_benchmark_bureau]
```

La idea de ingeniería: la regla del curso («caída relativa > 20%») es un umbral de materialidad disfrazado de test. Separar **significancia** (¿es ruido?) de **materialidad** (¿importa?) en dos campos del contrato evita las dos fallas de §5.3.

**Registro de salida.** Cada ejecución escribe un JSON inmutable: `{modelo_hash, muestra, corte_fecha, n, n_malos, auc, se_delong, ic95, ks, ks_umbral, metodo, semilla, B, version_metrica}`. El tablero (M20) lee de ahí; nadie recalcula a mano.

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy del notebook | Librería | Diferencias de convención | Recomendación producción |
|---|---|---|---|---|
| AUC | `auc_rangos` (rangos medios) y `auc_conteo` ($O(n^2)$, parámetro `peso_empate`) | `sklearn.metrics.roc_auc_score`; `scipy.stats.mannwhitneyu` | sklearn: trapecios = ½ empates; el orden de argumentos es `(y_true, y_score)` con clase positiva = 1 → con target 1 = malo, pasa la PD, no el score. `mannwhitneyu(x, y).statistic` es $U$ de la **primera** muestra; la corrección de continuidad (`use_continuity`) solo afecta al p-valor, no a $U$. | `roc_auc_score` para el punto; tu `auc_rangos` como referencia de tests |
| AUC parcial | — | `roc_auc_score(..., max_fpr=a)` | Devuelve el AUC parcial **estandarizado** de McClish (0,5 = azar), no el área cruda | Úsalo sabiendo que está reescalado |
| ROC | `curvas_roc_cap` (agrupa empates) | `sklearn.metrics.roc_curve` | `drop_intermediate=True` elimina puntos colineales; no cambia el AUC ni el máx(TPR − FPR) | Cualquiera |
| KS | `ks_numpy` (ECDF + `searchsorted`) | `scipy.stats.ks_2samp` | Mismo estadístico; el p-valor de `ks_2samp` supone distribuciones continuas (con empates masivos deja de ser exacto) y responde «¿son distintas las distribuciones?», no «¿discrimina lo suficiente?» | `ks_2samp` para el estadístico; no uses su p-valor como test de discriminación |
| Gini = D de Somers | `2*auc - 1`, `accuracy_ratio` | `scipy.stats.somersd(y, s)` | `somersd(x, y)` devuelve $D(Y\mid X)$: el orden de argumentos importa. Con `(y, s)` da el Gini | Solo como verificación |
| Rangos medios | `rangos_medios` | `scipy.stats.rankdata` (`method="average"`) | Idénticos; `rankdata` acepta `axis` (útil para simular muchas muestras) | `rankdata` |
| DeLong | `delong_definicion`, `delong_rapido` | No existe en sklearn / scipy / statsmodels | En R: `pROC::roc.test(method="delong")` | Implementación propia **con tests** contra la definición |
| Hanley–McNeil | `hanley_mcneil` | — | Supuesto exponencial; sobreestima en los ejemplos del módulo | Solo diseño muestral |
| Bootstrap | `bootstrap_auc` (percentil, remuestreo simple) | `scipy.stats.bootstrap(..., paired=True, method="percentile")` | El default de scipy es **BCa**, no percentil; sin `paired=True` remuestrea y y s por separado (¡destruye la asociación!); `n_resamples` default 9.999 | scipy con `paired=True`, método explícito, `rng` fijo |
| Deciles | `deciles_numpy` (cuantiles + `searchsorted(side="left")` = intervalos $(a, b]$) | `pandas.qcut(..., duplicates="drop")` | Mismos cortes; con empates masivos ambos colapsan deciles (menos de 10 grupos) | `qcut` y reporta cuántos grupos quedaron |
| Techo | `auc_esperado` $O(n\log n)$ | — | — | Solo en simulación (en datos reales no hay PD verdadera) |

Las funciones numpy del notebook son la especificación ejecutable, y las librerías, la implementación rápida y mantenida. En producción se usan las librerías donde existen y se testean contra la especificación. DeLong no tiene librería estándar en Python: la implementación propia vive en el repositorio de validación con sus tests.

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral (números del curso)

**Gini = 2·AUC − 1.** OOT: 2 × 0,8375 − 1 = 0,675. HO: 2 × 0,8490 − 1 = 0,698.

**Cotas KS.** OOT KS 0,545 ⇒ con ROC cóncava, Gini ∈ [0,545; 0,793]; observado 0,675 ✓. DEV: KS 0,597 ⇒ [0,597; 0,838]; observado 0,757 ✓.

**Hanley–McNeil vs bootstrap del curso.**

| Muestra | AUC | $n_M$ · $n_B$ | SE(Gini) HM | SE(Gini) implícito en IC bootstrap* | Ratio |
|---|---|---|---|---|---|
| DEV | 0,8785 | 165 · 3.157 | 0,035 | 0,023 | 1,52 |
| HO | 0,8490 | 78 · 1.319 | 0,055 | 0,039 | 1,43 |
| OOT | 0,8375 | 119 · 1.885 | 0,046 | 0,037 | 1,26 |

\* ancho del IC 95% / 3,92; aproximado porque el IC percentil no es simétrico.

En estos datos HM es conservador en 25–50%: sirve para dimensionar, no para concluir.

**Caída DEV − OOT.** Con el SE bootstrap del curso (IC de la diferencia [−0,006; +0,167], SE ≈ 0,173/3,92 = 0,044): $z$ = 0,082/0,044 ≈ 1,86. Bilateral: p ≈ 0,06, no concluyente, que es la lectura del curso. **Unilateral** (como el test del BCE): p ≈ 0,03, «significativa». La misma evidencia cambia de veredicto según el test que se haya declarado **antes** de mirar. Además, DEV trae optimismo, así que la caída desde DEV está sesgada hacia «significativa». La comparación temporal honesta es HO − OOT: +0,023 con SE ≈ 0,054 (bootstrap del curso), $z$ ≈ 0,43. Ahí no hay señal de deterioro, y tampoco evidencia de estabilidad: con 78 y 119 malos, la caída mínima detectable con potencia 80% es del orden de 0,13 (2,49 × 0,054).

**Tabla de deciles OOT en la planilla.** Con los n y malos del curso: Gini agrupado por trapecios 0,664 (vs 0,675 exacto); KS por decil 0,529 en el decil 2 (vs 0,545 exacto en el score 560, que cae justo después del corte 557 entre deciles 2 y 3); lift acumulado del decil 1: 4,78; 3 inversiones de tasa (4→5, 5→6, 7→8), todas con menos de 10 malos por decil.

**Ruido para Banco Austral** (hoja `Ruido`, Gini 0,70, π = 6%): con 120 malos por muestra, el umbral de caída al 95% es ≈ 0,10 (la regla de 0,10 del curso corresponde, sin buscarlo, a ~120 malos); con 60, ≈ 0,15; con 500, ≈ 0,05.

### 8.2 Cartera sintética (notebook)

Modelo de 7 variables sobre WoE: Gini DEV 0,533 / HO 0,588 / OOT 0,543, con 1.135 / 507 / 738 malos. Tasas 11,3% / 11,8% / 15,4%.

1. **Cuatro AUC iguales** (conteo, rangos, sklearn, scipy) hasta $10^{-12}$. A estos tamaños (≤ 4 millones de pares) ambos algoritmos toman milisegundos; el conteo escala como $n_M n_B$ y deja de ser viable con carteras de cientos de miles.
2. **Empates:** con 8 bandas, AUC OOT 0,709 (empate = 0) / 0,764 (½) / 0,818 (1). Gini 8 bandas 0,528 vs 0,543 continuo.
3. **Techo:** Gini esperado de la PD verdadera 0,608 en OOT; modelo evaluado contra la verdad 0,564: brecha 0,044, que es cota superior de lo que ML podría ganar (parte es heterogeneidad no observada del generador).
4. **Varianza (OOT):** SE(AUC) DeLong 0,0096 = definición; bootstrap numpy 0,0097; scipy 0,0093; HM 0,0106. IC 95% Gini: [0,505; 0,581] (DeLong) vs [0,503; 0,580] (bootstrap).
5. **DeLong pareado** (quitar una variable y reajustar): `meses_desde_mora_12m` Δ Gini = 0,035, $z$ = 3,7; `uso_linea_prom_12m` 0,024, $z$ = 4,4; `carga_financiera` 0,006, $z$ = 3,0 (significativa e irrelevante); `consultas_6m` 0,003, $z$ = 1,2. Correlación entre los AUC de los dos modelos 0,89–0,996; ignorarla infla el SE de la diferencia entre 3 y 16 veces.
6. **Caída por azar:** SD del Gini 0,090 (30 malos), 0,041 (120), 0,015 (1.000); P(caída > 0,10 | sin deterioro) = 19% / 4% / 0%.
7. **Nivel vs ranking:** deterioro de nivel 0 → 1,0 baja el Gini de 0,571 a 0,523 con la brecha al techo constante. En cambio, quitar el 60% del efecto de `uso_linea_prom_12m` en OOT (concept drift) baja el Gini a 0,460 con la tasa casi igual.

### 8.3 Crédito de motos

Considera una financiera de motos con 1.500 créditos al mes y 8% de malos a 12 meses: 120 malos por cohorte mensual. Supuestos: Gini ≈ 0,50 (admisión con bureau, población de ingresos medios-bajos y parte thin file) y $c \approx 0{,}45$–$0{,}50$ (una población más homogénea tiene el techo más bajo y la varianza algo mayor).

- **SE del Gini de una cohorte mensual** ≈ 0,47/√120 ≈ 0,043. Umbral de caída al 95% unilateral entre dos cohortes ≈ 1,645·√2·0,043 ≈ 0,10. Un monitoreo **mensual** con regla fija 0,10 dispara por azar ~5% de los meses. Si además se hace por 4 segmentos (nuevas / usadas × canal concesionario / digital), a lo largo de un año la probabilidad de al menos una falsa alarma es $1 - 0{,}95^{48} \approx 92\%$ (suponiendo independencia).
- **Segmento chico** (motos de alta cilindrada, 300 créditos/mes, 25 malos): SE(Gini) mensual ≈ 0,09. El Gini de ese segmento no se puede monitorear mes a mes. Hay que acumular: para detectar una caída de 0,10 con potencia 80% se necesitan ≈ 250 malos, o sea **unos 10 meses** de ese segmento. Esa es la ventana mínima y debe declararse.
- **Gini global vs por segmento:** si las motos usadas tienen tasa de malos del doble que las nuevas, el Gini global incluye la separación entre segmentos. Si la política usa cutoffs por segmento, el comité debe ver el Gini intra-segmento.
- **Techo:** una población de ingresos homogéneos con poca historia tiene σ del log-odds bajo. Un Gini de 0,45 puede estar cerca del techo, y la mejora posible con más técnica ser marginal. La mejora real suele venir de **información nueva** (comportamiento de pago de otros productos, datos de la concesionaria, pie inicial), que sube σ observable.
- **Métrica de negocio:** para decidir entre dos versiones del modelo, la pregunta del comité es «a 75% de aprobación, ¿cuántos puntos básicos de mora ahorro?», con un IC bootstrap **pareado** de esa diferencia.

---

## 9. Preguntas de comité

**1. «El Gini OOT es 0,675. ¿Qué tan seguros estamos?»**
Con 119 malos, el IC 95% es [0,60; 0,74] por bootstrap (curso), y DeLong daría un ancho similar. Ese ±0,07 es el precio de la muestra: 0,675 es una estimación, y cualquier conclusión que dependa de la segunda cifra decimal no está respaldada.

**2. «Cayó de 0,757 a 0,675: ¿hay que re-desarrollar?»**
No con esta evidencia. (i) La caída desde DEV mezcla optimismo de selección (esperado ≈ 0,02–0,05 con 165 malos y 8 variables) con deterioro. (ii) La comparación temporal es HO → OOT: 0,023 con SE ≈ 0,054, $z$ ≈ 0,4. (iii) La caída relativa de 10,8% está bajo el umbral de materialidad del 20%. Decisión: vigilar, acumular malos y declarar ahora el test y el umbral con que se decidirá en el próximo corte.

**3. «El modelo nuevo tiene Gini 0,69 y el actual 0,675; los IC se solapan. ¿Es mejor?»**
La pregunta está mal planteada con IC separados. Sobre la misma muestra, los AUC están correlacionados (típicamente 0,9 o más) y el SE de la diferencia es varias veces menor que el de cada Gini. Pido DeLong pareado o bootstrap pareado de la diferencia, y además la diferencia en tasa de malos a la aprobación objetivo, con IC. Si la diferencia es significativa pero de 0,005, no justifica el costo de cambiar de modelo.

**4. «¿Por qué el KS se alcanza cerca del cutoff?»**
El KS se alcanza donde la PD iguala la tasa de malos de la muestra (razón de verosimilitudes = 1). El cutoff óptimo está donde la PD iguala la PD de equilibrio $c_B/(c_B + c_M)$. Coinciden solo si esa PD de equilibrio se parece a la tasa de malos. En Banco Austral es una coincidencia económica y no justifica el cutoff; el cutoff se justifica con la curva de rentabilidad (M17).

**5. «El Gini de este modelo de motos es 0,45; el de consumo, 0,62. ¿El de motos es malo?»**
No se puede concluir. Son poblaciones con distinta dispersión de riesgo real y distinta tasa: el techo alcanzable es distinto. La comparación válida es contra el benchmark sobre la misma población (score de bureau, modelo anterior), de forma pareada.

**6. «Calibraron la PD con un δ nuevo. ¿Hay que revalidar la discriminación?»**
No: el δ es una transformación estrictamente creciente del score y el AUC, el Gini y el KS son idénticos (el notebook lo verifica). Sí hay que revalidar la calibración (M15, M19). Distinto sería si cambió la master scale (bandas): ahí el Gini sobre grados cambia por los empates y debe recalcularse.

**7. «¿Por qué no usan una regla fija de 0,10 como todo el mundo?»**
Porque el ruido depende del número de malos: con 60 malos el azar supera 0,10 en ~12% de los casos, y con 1.000 una caída real de 0,06 quedaría invisible. Proponemos dos reglas: un test (DeLong o bootstrap) con α declarado y ventana mínima de 250 malos para la significancia, y un umbral de materialidad para la acción.

**8. «En DEV el Gini es 0,53 y en HO 0,59. ¿Hay un error?»**
No necesariamente. El optimismo es positivo en expectativa, pero con ~500 malos en HO la SD del Gini es ≈ 0,02–0,03 y una realización con HO > DEV es perfectamente posible. Se revisa (partición realmente aleatoria, sin fuga de HO a DEV) antes de celebrar o acusar. Si se repite sistemáticamente en varias particiones, entonces sí hay algo raro.

---

## 10. Ejercicios

**Ejercicio 1 (AUC a mano con empates).** Malos con PD {0,30; 0,20; 0,20}; buenos con PD {0,20; 0,10; 0,05; 0,20}. Calcula el AUC por conteo de pares y por rangos medios. ¿Cuánto vale con empates = 0 y = 1?

<details><summary>Solución</summary>

Pares: $3 \times 4 = 12$.
- Malo 0,30: supera a los 4 buenos → 4.
- Malo 0,20 (×2): cada uno supera a 0,10 y 0,05 (2) y empata con dos buenos de 0,20 (2 × ½ = 1) → 3 cada uno.

Total = 4 + 3 + 3 = 10 → AUC = 10/12 = 0,833.

Rangos medios (7 obs. ordenadas ascendentes): 0,05 → 1; 0,10 → 2; 0,20 (cuatro: 2 malos + 2 buenos) → rangos 3–6, medio 4,5; 0,30 → 7. $R_M = 7 + 4{,}5 + 4{,}5 = 16$. $U = 16 - 3\cdot4/2 = 10$ ✓.

Pares empatados: 2 malos × 2 buenos = 4. Empates = 0: (10 − 2)/12 = 0,667; empates = 1: (10 + 2)/12 = 1,000. La convención mueve el AUC ±0,167: con muestras chicas y scores discretos, el ½ es imprescindible.
</details>

**Ejercicio 2 (AR = Gini con datos agrupados).** Con la tabla OOT del curso (deciles del peor al mejor, malos 57, 26, 11, 5, 7, 8, 2, 3, 0, 0; n ≈ 200 por decil; total 2.004 y 119 malos), explica por qué el AR calculado sobre la CAP agrupada coincide **exactamente** con el Gini calculado por trapecios sobre la ROC agrupada, y por qué ambos son menores que 0,675.

<details><summary>Solución</summary>

La derivación de §3.5 vale para cualquier curva lineal por tramos, siempre que CAP y ROC usen los mismos cortes: $A_{\text{CAP}} = \pi/2 + (1-\pi)\,\text{AUC}_{\text{agrupado}}$ tramo a tramo, porque en cada decil $\Delta x = \pi\Delta\text{TPR} + (1-\pi)\Delta\text{FPR}$ y el trapecio es lineal. Luego AR = 2·AUC_agrupado − 1 = 0,664 (celda de control de la planilla = 0). Son menores que 0,675 porque agrupar trata como empatados (½) a los pares malo-bueno dentro de un mismo decil, que en el score continuo estaban en su mayoría bien ordenados.
</details>

**Ejercicio 3 (cotas KS).** Un proveedor reporta KS = 0,40 y Gini = 0,35. ¿Es posible? ¿Y KS = 0,40 con Gini = 0,70?

<details><summary>Solución</summary>

Cota superior (siempre): Gini ≤ KS(2 − KS) = 0,40 × 1,60 = 0,64. **Gini 0,70 es imposible** con KS 0,40: hay un error de cálculo o las métricas vienen de muestras distintas.

Gini 0,35 < KS 0,40 viola la cota inferior **para ROC cóncava**, pero no la general (Gini ≥ 2·0,40 − 1 = −0,20). Es posible: indica una ROC no cóncava, es decir, un tramo del score que ordena mal (típicamente las colas, o una variable con signo invertido en un segmento). Pedir la ROC y la tabla de deciles para ubicar el tramo.
</details>

**Ejercicio 4 (Hanley–McNeil).** Calcula el SE del AUC y del Gini de HO de Banco Austral (AUC 0,8490, 78 malos, 1.319 buenos) con Hanley–McNeil, y compáralo con el IC bootstrap del curso [0,617; 0,769].

<details><summary>Solución</summary>

$Q_1 = 0{,}849/1{,}151 = 0{,}7376$; $Q_2 = 2 \cdot 0{,}7208/1{,}849 = 0{,}7797$; $A^2 = 0{,}7208$.

Numerador: $0{,}849 \cdot 0{,}151 = 0{,}1282$; $77 \cdot (0{,}7376 - 0{,}7208) = 77 \cdot 0{,}0168 = 1{,}294$; $1.318 \cdot (0{,}7797 - 0{,}7208) = 1.318 \cdot 0{,}0589 = 77{,}6$. Suma ≈ 79,0.

Denominador: $78 \cdot 1.319 = 102.882$. SE(AUC) = √(79,0/102.882) = √0,000768 ≈ 0,0277; SE(Gini) ≈ 0,055; IC ≈ 0,698 ± 0,109 = [0,589; 0,807].

El bootstrap da un ancho de 0,152, o sea un SE ≈ 0,039. HM es ~40% más ancho: su supuesto exponencial no se ajusta a estos datos. Para decidir, DeLong o bootstrap.
</details>

**Ejercicio 5 (tamaño muestral para monitoreo).** Quieres detectar una caída del Gini de 0,07 entre HO y una ventana de monitoreo, con α = 5% unilateral y potencia 80%. Con $c$ = 0,40 y el mismo número de malos en ambas muestras, ¿cuántos malos necesitas en cada una? Con 90 malos al mes, ¿cuántos meses de ventana?

<details><summary>Solución</summary>

$n_M = 2(1{,}645 + 0{,}842)^2 c^2/\Delta^2 = 2 \cdot 6{,}185 \cdot 0{,}16/0{,}0049 = 1{,}979/0{,}0049 \approx 404$ malos por muestra. Con 90 malos al mes, ≈ 4,5 meses: ventana móvil de 5 meses (y HO debe tener al menos ~400 malos; si tiene menos, la incertidumbre de HO domina y hay que recalcular con $n_M$ distintos: $\operatorname{Var} = c^2(1/n_1 + 1/n_2)$).
</details>

**Ejercicio 6 (IV ↔ Gini).** En el mundo binormal, (a) ¿qué Gini promete una variable con IV = 0,30? (b) ¿Qué IV necesita una variable para tener Gini 0,40 sola? (c) ¿Por qué no deberías usar esto para predecir el Gini de un modelo de 8 variables con IV 0,3 cada una?

<details><summary>Solución</summary>

(a) $2\Phi(\sqrt{0{,}15}) - 1 = 2\Phi(0{,}387) - 1 = 2(0{,}6507) - 1 = 0{,}301$.

(b) $\Phi(z) = 0{,}70 \Rightarrow z = 0{,}5244$; $\text{IV} = 2z^2 = 0{,}55$.

(c) Las variables están correlacionadas: la información no se suma. Incluso si fueran independientes, en el binormal las separaciones se suman en cuadratura ($d^2_{\text{total}} = \sum d_k^2$, es decir, los IV se suman) solo con independencia condicional exacta, y eso casi nunca ocurre en bureau (las familias de uso y mora comparten el factor latente, M09). Además el IV binneado tiene sesgo y la normalidad no se cumple.
</details>

**Ejercicio 7 (derivación del punto KS con costos).** Muestra que el corte que minimiza $C(t) = \pi c_M (1-\text{TPR}) + (1-\pi) c_B \text{FPR}$ se alcanza donde $P(\text{malo}\mid s = t) = c_B/(c_B + c_M)$, y que coincide con el punto KS si y solo si $c_M/c_B = (1-\pi)/\pi$. Para π = 6%, ¿qué razón pérdida/margen implica?

<details><summary>Solución</summary>

$dC/dt = \pi c_M f_M(t) - (1-\pi) c_B f_B(t)$ (porque $d\,\text{TPR}/dt = -f_M$ y $d\,\text{FPR}/dt = -f_B$). Igualando a 0: $\pi c_M f_M = (1-\pi) c_B f_B$. Por Bayes, $\text{PD}(t) = \pi f_M/(\pi f_M + (1-\pi) f_B)$; sustituyendo $(1-\pi) f_B = \pi c_M f_M/c_B$, queda $\text{PD}(t) = \pi f_M/(\pi f_M(1 + c_M/c_B)) = c_B/(c_B + c_M)$.

El KS está donde $f_M = f_B$, o sea $\text{PD} = \pi$. Coinciden si $c_B/(c_B+c_M) = \pi \iff c_M/c_B = (1-\pi)/\pi$. Con π = 0,06: $c_M/c_B = 15{,}7$. Un malo tendría que costar 15,7 veces el margen de vida de un bueno; con LGD·EAD ≈ 60% del monto, eso exige un margen ≈ 3,8% del monto. Es plausible en consumo bancario de margen bajo, no en crédito de alto margen.
</details>

**Ejercicio 8 (código: bootstrap estratificado).** Modifica `bootstrap_auc` del notebook para remuestrear malos y buenos **por separado** (manteniendo $n_M$ y $n_B$). Compara su SE con el del bootstrap simple y con DeLong en HO y OOT. ¿Cuál debería parecerse más a DeLong y por qué?

<details><summary>Solución</summary>

```python
def bootstrap_auc_estratificado(y, s, B, semilla):
    rng = np.random.default_rng(semilla)
    y = np.asarray(y); s = np.asarray(s)
    im, ib = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = np.empty(B)
    for b in range(B):
        idx = np.r_[rng.choice(im, len(im)), rng.choice(ib, len(ib))]
        out[b] = auc_rangos(y[idx], s[idx])
    return out
```

DeLong condiciona en $n_M$ y $n_B$ (varianza de un U-estadístico de dos muestras con tamaños fijos), así que el estratificado es su análogo exacto. El simple añade la variabilidad de $n_M$, que para el AUC es de segundo orden: las diferencias aparecen en el tercer decimal del SE con cientos de malos. Con 30 malos, el simple puede generar réplicas con 15 o 45 malos y su SE se aleja más.
</details>

**Ejercicio 9 (DeLong pareado a mano).** Dos modelos sobre la misma muestra: $\text{AUC}_1 = 0{,}780$, $\text{AUC}_2 = 0{,}760$, $\operatorname{Var}_1 = \operatorname{Var}_2 = 1{,}0\times10^{-4}$, $\operatorname{Cov} = 0{,}9\times10^{-4}$. Calcula $z$ pareado y $z$ ignorando la covarianza. Interpreta.

<details><summary>Solución</summary>

Pareado: $\operatorname{Var}(\Delta) = 1 + 1 - 1{,}8 = 0{,}2 \times 10^{-4}$; SE = 0,00447; $z = 0{,}020/0{,}00447 = 4{,}47$ (p < 0,0001).

Ignorando la covarianza: SE = √(2 × 10⁻⁴) = 0,01414; $z$ = 1,41 (p = 0,16).

Cada IC individual es 0,78 ± 0,0196 y 0,76 ± 0,0196: se solapan. Aun así, la diferencia es clarísima, porque los dos modelos se equivocan en los mismos clientes (correlación 0,9) y lo que varía entre réplicas es sobre todo común a ambos.
</details>

**Ejercicio 10 (diseño: regla de monitoreo).** Redacta la regla de semáforo para el Gini de un modelo de motos con 120 malos/mes en total y 4 segmentos, usando el YAML de §6 como base. Justifica cada número.

<details><summary>Solución (una propuesta defendible)</summary>

- Referencia: HO (no DEV). Ventana: móvil de 3 meses a nivel total (~360 malos) y de 12 meses por segmento (~360 malos si cada segmento tiene ~30/mes).
- Significancia: DeLong entre muestras independientes (HO vs ventana), α = 5% unilateral, Benjamini–Hochberg sobre los 5 tests del mes (total + 4 segmentos).
- Materialidad: caída ≥ 0,05 amarillo, ≥ 0,10 rojo, aplicada solo si el test rechaza.
- Persistencia: rojo requiere dos cortes consecutivos.
- Justificación: con ~360 malos, SE(Gini) ≈ 0,45/√360 ≈ 0,024 y el umbral de azar de la diferencia ≈ 1,645·√2·0,024 ≈ 0,055, del orden del umbral amarillo; la potencia para Δ = 0,10 es ≈ 90% ($0{,}10/(\sqrt2 \cdot 0{,}024) - 1{,}645 \approx 1{,}3$ desviaciones). Los segmentos se evalúan anualmente porque mensualmente su SE (~0,08) haría que la regla dispare por ruido.
- Reportar siempre: $n_M$, tasa de malos, Gini del score de bureau sobre la misma ventana (benchmark pareado), y dispersión de la PD (para separar deterioro de ranking de cambio de población, §3.7).
</details>

---

## 11. Referencias

- **Hanley, J. A. y McNeil, B. J. (1982).** «The meaning and use of the area under a receiver operating characteristic (ROC) curve». *Radiology*, 143(1), 29–36. — La lectura de AUC como probabilidad y la fórmula de SE que todavía se usa para dimensionar.
- **DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988).** «Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach». *Biometrics*, 44(3), 837–845. — El test pareado; imprescindible para comparar modelos.
- **Sun, X. y Xu, W. (2014).** «Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves». *IEEE Signal Processing Letters*, 21(11), 1389–1393. — El algoritmo con rangos medios que implementa el notebook.
- **Bamber, D. (1975).** «The area above the ordinal dominance graph and the area below the receiver operating characteristic graph». *Journal of Mathematical Psychology*, 12(4), 387–415. — La equivalencia AUC = Mann–Whitney.
- **Mann, H. B. y Whitney, D. R. (1947).** «On a test of whether one of two random variables is stochastically larger than the other». *Annals of Mathematical Statistics*, 18(1), 50–60. — El estadístico U original.
- **Engelmann, B., Hayden, E. y Tasche, D. (2003).** «Testing rating accuracy». *Risk*, 16(1), 82–86 (verificar paginación). — AR = 2·AUC − 1 y los IC del AR en el lenguaje de riesgo de crédito.
- **Basel Committee on Banking Supervision (2005).** *Studies on the Validation of Internal Rating Systems*, Working Paper No. 14 (revisado). — El capítulo de discriminación (CAP, AR, ROC, AUROC y su varianza) es la referencia supervisora clásica.
- **Tasche, D. (2006).** «Validation of internal rating systems and PD estimates». arXiv: physics/0606071 (publicado luego como capítulo en *The Analytics of Risk Model Validation*, 2008, verificar edición). — Síntesis matemática corta de métricas de discriminación y calibración para PD.
- **European Central Bank (2019).** *Instructions for reporting the validation results of internal models — IRB Pillar I models for credit risk* (febrero 2019). — Test unilateral AUC inicial vs actual y AUC sobre grados; leer el anexo para el estimador de varianza.
- **Youden, W. J. (1950).** «Index for rating diagnostic tests». *Cancer*, 3(1), 32–35. — El índice J = TPR − FPR, es decir, el KS.
- **McClish, D. K. (1989).** «Analyzing a portion of the ROC curve». *Medical Decision Making*, 9(3), 190–195. — El AUC parcial estandarizado que usa `sklearn` con `max_fpr`.
- **Fawcett, T. (2006).** «An introduction to ROC analysis». *Pattern Recognition Letters*, 27(8), 861–874. — Tutorial limpio de ROC, empates y promedios.
- **Hand, D. J. (2009).** «Measuring classifier performance: a coherent alternative to the area under the ROC curve». *Machine Learning*, 77(1), 103–123. — La crítica de fondo al AUC y la medida H.
- **Krzanowski, W. J. y Hand, D. J. (2009).** *ROC Curves for Continuous Data*. CRC Press. — Tratamiento completo: binormal, inferencia, comparación.
- **Pepe, M. S. (2003).** *The Statistical Evaluation of Medical Tests for Classification and Prediction*. Oxford University Press. — Inferencia sobre ROC con rigor; el origen de muchas ideas que el crédito adoptó tarde.
- **Newson, R. (2002).** «Parameters behind "nonparametric" statistics: Kendall's tau, Somers' D and median differences». *Stata Journal*, 2(1), 45–64. — Por qué el Gini es la D de Somers y cómo se relaciona con tau.
- **Efron, B. (1983).** «Estimating the error rate of a prediction rule: improvement on cross-validation». *JASA*, 78(382), 316–331. — El estimador bootstrap del optimismo.
- **Harrell, F. E. (2015).** *Regression Modeling Strategies* (2.ª ed.). Springer. — Validación con corrección de optimismo aplicada a todo el pipeline de ajuste.
- **Efron, B. y Tibshirani, R. J. (1993).** *An Introduction to the Bootstrap*. Chapman & Hall. — Percentil vs BCa, número de réplicas.
- **Robin, X. et al. (2011).** «pROC: an open-source package for R and S+ to analyze and compare ROC curves». *BMC Bioinformatics*, 12, 77. — La implementación de referencia de DeLong pareado (útil para contrastar la tuya).
- **Thomas, L. C., Crook, J. y Edelman, D. (2017).** *Credit Scoring and Its Applications* (2.ª ed.). SIAM. — Métricas de discriminación en el contexto de scoring (KS, Gini, curvas de estrategia).
- **Siddiqi, N. (2017).** *Intelligent Credit Scoring* (2.ª ed.). Wiley. — La práctica de industria (y el origen de muchas reglas de dedo que aquí se discuten).
- **Serie 1:** E3 (bootstrap para carteras chicas), E2 (calibración, PIT vs TTC), M7 (IV), M4 (esquema muestral). **Serie 2:** M08 (PSI y cancelaciones), M15 (calibración), M16 (master scale y empates por bandas), M17 (cutoff y PD de equilibrio), M19 (backtesting de nivel), M20 (tablero y semáforos).

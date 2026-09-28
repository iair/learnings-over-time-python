# M10 · La regresión logística sobre WoE, desde la verosimilitud

> **Ficha.** Profundiza la clase 3 (parte 2: «La regresión logística sobre WoE», stepwise, modelo final de Banco Austral, estabilidad de coeficientes en HO, intercepto). · **Prerrequisitos:** Serie 1 · M7 (binning, WoE e IV con derivación formal), Serie 1 · E3 (bootstrap en carteras chicas); Serie 2 · M09 (redundancia y colinealidad). · **Archivos:** `M10_logistica_woe.md` (este documento), `M10_logistica_woe.py` (notebook Marimo: `marimo edit --sandbox M10_logistica_woe.py`). · **Tiempo estimado:** 3 h de lectura con derivaciones + 1,5 h de notebook.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

En la clase 3 el embudo de Banco Austral llegó a 26 variables (108 → 94 → 63 → 26, VIF máximo 2,97) y un stepwise con tres criterios simultáneos (p < 0,05, signo esperado y mejora de la log-verosimilitud) dejó **8 variables**, sin que ninguna saliera después de entrar. La regresión se presentó en «dos minutos de nivelación»: se modela $\ln(\text{PD}/(1-\text{PD}))=\beta_0+\sum_j\beta_j\,\text{WoE}_j$; el signo de β da la dirección; el p-valor dice «qué tan probable sería observar ese coeficiente si la variable no aportara nada». Y la frase textual: *«esto no es un curso de econometría: para dictaminar basta leer signo, p-valor y rango de puntos. Si quieren el detalle, investíguenlo»*. Este módulo es ese detalle.

Lo que sí quedó establecido en clase, y que usaremos como ancla:

- **Por qué WoE y no crudo**: el WoE resuelve de antemano la no linealidad (queda en los bins), las escalas (todo queda en unidades de log-odds) y los missing/categóricas (son un bin más). Costo explícito: se pierde granularidad dentro del bin.
- **Convención de signo**: $\text{WoE}=\ln(\%\text{buenos}/\%\text{malos})$, target 1 = malo ⇒ los 8 coeficientes deben ser negativos. Los de Austral: uso_linea_prom_12m −0,505; meses_desde_mora_12m −0,768; uso_tc_prom_12m −0,353; deuda_interna_max_3m −0,926; antiguedad_meses −0,606; deuda_otras_prom_12m −0,789; carga_financiera −0,418; uso_tc_prom_3m −0,309.
- **Gini por muestra**: DEV 0,757, HO 0,698 (caída 0,059 < 0,10), OOT 0,675 (caída 0,082 < 0,15).
- **Estabilidad de coeficientes**: el mismo modelo re-ajustado solo en HO (1.397 créditos, **78 malos para 9 parámetros = 8,7 por parámetro**) «contrasta, no re-estima». Las fuertes cuentan la misma historia (uso_linea −0,51 → −0,57; mora −0,77 → −0,82); deuda_interna y deuda_otras «cambian de signo» a ≈ 0 con p 0,96 y 0,78: indistinguibles de cero, no una inversión.
- **El intercepto** (lámina de reserva) «absorbe la tasa de malos de la muestra de desarrollo (~5%)»; fija el nivel, no el orden.

Lo que el curso **simplificó, omitió o dejó como convención**:

1. Cómo se estima β (máxima verosimilitud, Newton-Raphson/IRLS) y de dónde sale el error estándar que produce el p-valor.
2. Por qué DEV reproduce exactamente la tasa de malos: es una identidad algebraica, no evidencia de calibración.
3. Que hay tres tests distintos (Wald, razón de verosimilitud, score) y que el que imprime `statsmodels` en la tabla (Wald) es el que peor se comporta en bins extremos.
4. **El resultado β = −1**: con una sola variable en WoE, el MLE es exactamente $\hat\beta_1=-1$, $\hat\beta_0=\ln(M/B)$. Todo coeficiente multivariado es, por lo tanto, un *factor de descuento* respecto de −1. Esto da una lectura de los β de Austral que el curso no hizo.
5. Que los p-valores de variables en WoE son **optimistas** porque los WoE se estimaron mirando $y$ en la misma muestra (grados de libertad escondidos).
6. Separación (bins sin malos) y la corrección de Firth; y que el suavizado +0,5 de `tabla_woe` es, literalmente, Firth.
7. El «8,7 por parámetro» aparece como regla 10–20 sin fuente ni crítica.
8. Alternativas: dummies, splines, regularización, pesos por clase y corrección del intercepto al submuestrear.

---

## 2. Intuición

Piense en el WoE de un bin como **el peso de la evidencia** que aporta ese bin a favor de «bueno», en el sentido de I. J. Good: cuántos log-odds mueve observar al cliente en ese tramo. Si un cliente cae en el bin de utilización baja con WoE +2,31, la evidencia dice «las odds buenas/malas de este cliente son $e^{2,31}\approx 10$ veces las de la cartera».

Con **una sola variable**, la mejor predicción posible dentro de ese binning es simplemente la tasa de malos observada en cada bin (el modelo saturado). Y resulta que esa tasa, en escala log-odds, es exactamente «log-odds de la cartera menos el WoE del bin». La logística con una variable en WoE no tiene nada que aprender: la respuesta ya está escrita en el WoE, con coeficiente −1.

Con **varias variables**, si fueran independientes dado el tipo de cliente (bueno o malo), la evidencia se sumaría: log-odds = log-odds de la cartera − WoE₁ − WoE₂ − … Eso es Naive Bayes. Pero uso de línea, uso de tarjeta a 12 meses y uso de tarjeta a 3 meses cuentan casi la misma historia; sumarlas es contar tres veces la misma evidencia. La logística multivariada aprende cuánto de cada evidencia es *nueva* dado el resto: el coeficiente de uso_tc_prom_3m en Austral (−0,309) dice «solo un 31% de lo que esta variable dice sola es información adicional».

Hay una sorpresa: una variable totalmente independiente de las demás puede terminar con |β| > 1. No es sinergia; es que el WoE univariado subestima el efecto cuando hay otra fuente fuerte de heterogeneidad que no controla (no colapsabilidad del odds-ratio).

Y hay una trampa: como el WoE se calculó mirando $y$, cada variable en WoE ya «gastó» $K-1$ grados de libertad antes de entrar a la regresión. La regresión le cobra solo uno. Los p-valores salen demasiado optimistas.

---

## 3. Formalización

Notación: $n$ créditos, $y_i\in\{0,1\}$ (1 = malo), $x_i\in\mathbb R^{k}$ con primera componente 1; $\eta_i=x_i^\top\beta$, $p_i=\sigma(\eta_i)=1/(1+e^{-\eta_i})$. Totales: $M=\sum_i y_i$ malos, $B=n-M$ buenos. Para una variable con bins $k=1,\dots,K$: $m_k$ malos, $b_k$ buenos, $n_k=m_k+b_k$.

### 3.1 Verosimilitud Bernoulli

$$L(\beta)=\prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i}.$$

Como $\ln p_i=\eta_i-\ln(1+e^{\eta_i})$ y $\ln(1-p_i)=-\ln(1+e^{\eta_i})$:

$$\ell(\beta)=\sum_i\Big[y_i\big(\eta_i-\ln(1+e^{\eta_i})\big)-(1-y_i)\ln(1+e^{\eta_i})\Big]=\sum_i\Big[y_i\eta_i-\ln(1+e^{\eta_i})\Big].$$

En el notebook se evalúa con `np.logaddexp(0, eta)` para no desbordar con $|\eta|$ grande.

### 3.2 Gradiente, Hessiano y concavidad

Usando $\frac{d}{d\eta}\ln(1+e^\eta)=\sigma(\eta)=p$:

$$\frac{\partial \ell}{\partial\beta}=\sum_i (y_i-p_i)\,x_i = X^\top(y-p)\equiv U(\beta).$$

Derivando otra vez, con $\frac{dp}{d\eta}=p(1-p)$:

$$\frac{\partial^2\ell}{\partial\beta\,\partial\beta^\top}=-\sum_i p_i(1-p_i)\,x_ix_i^\top=-X^\top W X,\qquad W=\operatorname{diag}\big(p_i(1-p_i)\big).$$

Si $X$ tiene rango completo y $0<p_i<1$, $X^\top WX$ es definida positiva ⇒ $\ell$ es **estrictamente cóncava** ⇒ si el máximo existe, es único. «Si existe» es la condición que falla con separación (§3.11).

### 3.3 Newton-Raphson = IRLS

Newton sobre la aproximación cuadrática de $\ell$ en $\beta^{(t)}$:

$$\beta^{(t+1)}=\beta^{(t)}+(X^\top WX)^{-1}X^\top(y-p).$$

Factorizando $X^\top W$:

$$\beta^{(t+1)}=(X^\top WX)^{-1}X^\top W\,z,\qquad z=X\beta^{(t)}+W^{-1}(y-p),$$

que es mínimos cuadrados ponderados de la «respuesta de trabajo» $z$ sobre $X$ con pesos $W$, re-calculados en cada iteración: **IRLS** (iteratively reweighted least squares). Como el logit es el enlace canónico de la Bernoulli, el Hessiano no depende de $y$ y Newton coincide con *Fisher scoring*.

Convergencia: cerca del óptimo es cuadrática (el error se eleva al cuadrado en cada paso). En el notebook, con el modelo de 8 variables sobre DEV, `max|Δβ|` pasa de $2{,}0\cdot10^{-3}$ a $2{,}3\cdot10^{-6}$ a $3{,}0\cdot10^{-12}$ en las iteraciones 5, 6 y 7. Criterio de parada razonable en producción: $\max|\Delta\beta|<10^{-8}$ **y** $\lVert U(\hat\beta)\rVert_\infty<10^{-6}$, más una bandera explícita de no convergencia (nunca aceptar silenciosamente el resultado de la iteración máxima).

### 3.4 Información de Fisher y errores estándar

$I(\beta)=-E[\partial^2\ell/\partial\beta\partial\beta^\top]=X^\top WX$. Bajo el modelo correcto, observaciones independientes y regresores **fijos**,

$$\hat\beta\ \dot\sim\ \mathcal N\big(\beta,\ I(\hat\beta)^{-1}\big),\qquad \text{SE}(\hat\beta_j)=\sqrt{\big[(X^\top\hat WX)^{-1}\big]_{jj}}.$$

Tres supuestos que en scorecards se violan con frecuencia: (i) regresores fijos: los WoE son *estimados* con el mismo $y$ (§3.10); (ii) independencia: varias operaciones del mismo cliente o del mismo dealer comparten shocks; (iii) especificación correcta: el binning es una aproximación. Si (ii) falla, los SE correctos son de tipo sándwich/cluster.

### 3.5 Ecuaciones de primer orden: por qué DEV «clava» la tasa

En el óptimo $U(\hat\beta)=X^\top(y-\hat p)=0$. La fila del intercepto (columna de unos) dice

$$\sum_i(y_i-\hat p_i)=0\quad\Longrightarrow\quad\sum_i\hat p_i=\sum_i y_i=M.$$

La PD media predicha en DEV es la tasa de DEV **por construcción**, para cualquier conjunto de variables, bueno o malo. En el notebook: $\sum\hat p_i=1.135{,}000000$ vs $M=1.135$. Análogamente, si el diseño incluye una dummy por bin, la tasa predicha de cada bin coincide con la observada. Con WoE la condición es más débil: $\sum_i\text{WoE}_{ij}(y_i-\hat p_i)=0$ para cada $j$ (una restricción por variable, no por bin), por eso un scorecard sobre WoE **puede** estar descalibrado por bin incluso en DEV.

Consecuencia práctica: «el modelo calibra en DEV» no es un hallazgo; la calibración se mide fuera de muestra (HO, OOT) y por bandas (M15, M19).

### 3.6 Tres tests: Wald, razón de verosimilitud y score

Para $H_0:\beta_j=0$ (o, en general, $q$ restricciones), con $\hat\beta$ el MLE libre y $\tilde\beta$ el restringido:

$$W=\frac{\hat\beta_j^2}{\widehat{\operatorname{Var}}(\hat\beta_j)},\qquad \text{LR}=2\big[\ell(\hat\beta)-\ell(\tilde\beta)\big],\qquad S=U(\tilde\beta)^\top I(\tilde\beta)^{-1}U(\tilde\beta).$$

Bajo $H_0$ los tres son asintóticamente $\chi^2_q$. Interpretación geométrica (Buse, 1982): LR mide la caída vertical de $\ell$; Wald, la distancia horizontal $\hat\beta-0$ escalada por la curvatura **en** $\hat\beta$; score, la pendiente en $\tilde\beta$ escalada por la curvatura **en** $\tilde\beta$. Si $\ell$ fuera exactamente cuadrática, coincidirían. En DEV del notebook coinciden a 1–2% en variables de efecto moderado, pero se separan en las de efecto grande: meses_desde_mora_12m tiene Wald 237,0, LR 228,5 y score 247,3.

**Efecto Hauck-Donner** (Hauck & Donner, 1977). Para un bin chico contra el resto (tabla 2×2 con celdas $a,b,c,d$), $\hat\beta$ es el log odds-ratio y $\operatorname{Var}(\hat\beta)=1/a+1/b+1/c+1/d$. Cuando el bin se vuelve extremo ($b\to 0$), $\hat\beta\sim\ln(1/b)$ crece logarítmicamente pero $\operatorname{Var}(\hat\beta)\approx1/b$ crece como $1/b$: $W\approx b\,[\ln(1/b)]^2\to0$. El Wald **cae** cuando la evidencia **sube**. En el experimento del notebook (bin de 60 contra 1.000 con 5% de malos), el Wald alcanza su máximo con 47 malos en el bin y luego baja: con 57 malos, Wald 93,2 vs LR 272,7. En crédito ocurre en el bin «excelente» con 0–2 malos: el Wald dice «no significativo» justo en el bin más informativo. **Regla**: para decisiones de entrada/salida, usar LR (o score, que no requiere ajustar el modelo grande).

### 3.7 El resultado clave: una variable en WoE ⇒ $\hat\beta_1=-1$, $\hat\beta_0=\ln(M/B)$

**Proposición.** Sea una variable con $K\ge2$ bins, $m_k>0$ y $b_k>0$ en todos, WoE no constante entre bins y

$$\text{WoE}_k=\ln\frac{b_k/B}{m_k/M}\qquad(\text{sin suavizar}).$$

Entonces el MLE de $\operatorname{logit}p_k=\beta_0+\beta_1\text{WoE}_k$ es exactamente $\hat\beta_1=-1$ y $\hat\beta_0=\ln(M/B)$.

**Demostración.** (1) Todas las observaciones de un bin comparten el regresor, así que la log-verosimilitud del modelo de 2 parámetros depende de $\beta$ solo a través de $p_k$:
$$\ell=\sum_{k=1}^K\big[m_k\ln p_k+b_k\ln(1-p_k)\big].$$
(2) Si dejamos $p_k$ libre por bin (modelo saturado), cada sumando se maximiza por separado: $\partial/\partial p_k=m_k/p_k-b_k/(1-p_k)=0\Rightarrow p_k^\star=m_k/n_k$. Este es el máximo global sobre **cualquier** vector de probabilidades constante por bin, en particular sobre todos los que el modelo de 2 parámetros puede generar.
(3) El log-odds del saturado es
$$\ln\frac{p_k^\star}{1-p_k^\star}=\ln\frac{m_k}{b_k}=\ln\frac{M}{B}+\ln\frac{m_k/M}{b_k/B}=\ln\frac{M}{B}-\text{WoE}_k.$$
(4) Luego $(\beta_0,\beta_1)=(\ln(M/B),-1)$ **alcanza** el máximo del modelo más grande; como pertenece al modelo restringido, es su maximizador. (5) Por concavidad estricta (§3.2, WoE no constante ⇒ rango completo) es el único. ∎

Observaciones:

- No depende del número de bins, de cómo se eligieron los cortes (cuantiles, óptimo, a mano), ni de que la variable sirva: el notebook obtiene $\hat\beta_1=-1$ a 10 decimales para las 11 variables del generador, incluidas `edad` y `renta_mm` con IV < 0,01. **El coeficiente univariado sobre WoE no mide poder**; el poder está en la dispersión del WoE (el IV) y se ve en el SE, no en β.
- Zeng (2014) demuestra el mismo resultado con la convención opuesta ($\text{WoE}=\ln(\%\text{malos}/\%\text{buenos})$ ⇒ pendiente +1) y lo propone como **condición necesaria** de un binning bien implementado. Es un test unitario excelente para un pipeline de WoE (§6).
- En Austral, con DEV de 3.322 y 4,97% de malos ($M=165$, $B=3.157$), cada variable sola daría $\hat\beta_0=\ln(165/3.157)\approx-2{,}951$: el «intercepto absorbe la tasa de DEV» de la lámina de reserva es literalmente esto.
- La log-verosimilitud del modelo WoE es la del saturado ⇒ el LR contra el modelo nulo es exactamente el $G^2$ de la tabla $K\times2$. Lo usaremos en §3.10.

### 3.8 Con el suavizado +0,5 del curso: aproximado, y por qué |β| tiende a ser > 1

`tabla_woe` del curso usa
$$\widetilde{\text{WoE}}_k=\ln\frac{(b_k+\tfrac12)/(B+\tfrac K2)}{(m_k+\tfrac12)/(M+\tfrac K2)}.$$
Reescribiendo el log-odds observado:
$$\ln\frac{m_k}{b_k}=\underbrace{\ln\frac{M+K/2}{B+K/2}}_{c}-\widetilde{\text{WoE}}_k+\varepsilon_k,\qquad \varepsilon_k=\ln\frac{m_k}{m_k+\tfrac12}-\ln\frac{b_k}{b_k+\tfrac12}\approx-\frac{1}{2m_k}+\frac{1}{2b_k}.$$
Como $\varepsilon_k$ varía entre bins, el modelo de 2 parámetros ya no alcanza al saturado y $\hat\beta_1\neq-1$. Aproximando el MLE por mínimos cuadrados ponderados de los logits observados sobre $\widetilde{\text{WoE}}$ (pesos $\approx n_kp_k(1-p_k)$):
$$\hat\beta_1\approx-1+\frac{\operatorname{Cov}_w(\varepsilon,\widetilde{\text{WoE}})}{\operatorname{Var}_w(\widetilde{\text{WoE}})}.$$
Los bins buenos (WoE alto) son los de pocos malos, donde $\varepsilon_k\approx-1/(2m_k)$ es más negativo ⇒ covarianza negativa ⇒ $\hat\beta_1<-1$. Intuición: el suavizado *encoge* los WoE extremos y el MLE los «des-encoge». El sesgo es $O(1/m_{\min})$: en DEV completo del notebook (≥ 48 malos por bin) la mayor desviación es 0,0029; con submuestras de 150 créditos (≈17 malos) la desviación media es 0,061 y la media de $\hat\beta_1$ es −1,034; en el juguete de separación del notebook (bin con 0 malos) el MLE sobre el WoE suavizado da −1,106.

**El suavizado +0,5 es Firth.** El logit por bin de Firth en el modelo saturado es $\ln\frac{m_k+1/2}{b_k+1/2}$ (demostración en §3.11), y
$$-\widetilde{\text{WoE}}_k+c=\ln\frac{m_k+\tfrac12}{M+\tfrac K2}-\ln\frac{b_k+\tfrac12}{B+\tfrac K2}+\ln\frac{M+\tfrac K2}{B+\tfrac K2}=\ln\frac{m_k+\tfrac12}{b_k+\tfrac12}.$$
Es decir, el WoE suavizado del curso es (salvo constante y signo) la estimación de Firth del log-odds de cada bin. El notebook lo verifica a $10^{-12}$. Lo que el curso presentó como un truco para evitar $\ln 0$ tiene una justificación de verosimilitud penalizada.

### 3.9 Multivariado: Naive Bayes como caso particular y los β como factores de descuento

Por Bayes, para un cliente con bins $(k_1,\dots,k_J)$ en $J$ variables:
$$\ln\frac{P(\text{malo}\mid x)}{P(\text{bueno}\mid x)}=\ln\frac{M}{B}+\ln\frac{P(k_1,\dots,k_J\mid\text{malo})}{P(k_1,\dots,k_J\mid\text{bueno})}.$$
Si las variables son **condicionalmente independientes dada la clase**, el cociente se factoriza y cada factor es $\frac{m_{k_j}/M}{b_{k_j}/B}=e^{-\text{WoE}_{j,k_j}}$:
$$\ln\frac{P(\text{malo}\mid x)}{P(\text{bueno}\mid x)}=\ln\frac MB-\sum_{j=1}^J\text{WoE}_{j}.$$
Esto es una logística sobre WoE con **todos los $\beta_j=-1$** y $\beta_0=\ln(M/B)$: Naive Bayes sobre bins (Hand & Yu, 2001, lo llaman «Idiot's Bayes» y discuten por qué funciona mejor de lo esperado en ranking). La logística multivariada relaja la restricción y estima cada $\beta_j$. Lectura:

- $\beta_j=-1$: la evidencia de $j$ entra completa, como si fuera independiente del resto.
- $-1<\beta_j<0$: **descuento por redundancia**; solo una fracción $|\beta_j|$ de la evidencia univariada es nueva dado el resto. En Austral los ocho factores van de 0,31 (uso_tc_prom_3m) a 0,93 (deuda_interna_max_3m). En el notebook, las tres de utilización quedan en 0,55, 0,47 y **−0,18**.
- $\beta_j>0$: la variable actúa como **supresor**: con su pariente en el modelo, lo que queda de ella tiene signo contrario (uso_tc_prom_12m, +0,181 con z = 1,39, al lado de uso_tc_prom_3m con correlación de WoE 0,91). El curso la habría eliminado antes por correlación > 0,70.
- $\beta_j<-1$: dos causas distintas. (a) **Sinergia/supresión** genuina: la evidencia condicional es más fuerte que la marginal. (b) **No colapsabilidad** del odds-ratio: aunque $X_1$ y $X_2$ sean independientes, si ambas afectan el riesgo, el log odds-ratio marginal de $X_1$ (que es lo que mide su WoE univariado) está atenuado hacia 0 respecto del condicional, porque promediar sigmoides no es la sigmoide del promedio. Al condicionar en $X_2$, el efecto de $X_1$ se «des-atenúa» y $|\beta_1|>1$. En el juguete del notebook ($X_1,X_2$ independientes, correlación de WoE 0,0015, logit verdadero $-2{,}5+X_1+2X_2$), $\hat\beta_1=-1{,}198$. En el modelo del notebook, antiguedad_meses (ortogonal al resto) da −1,139. **No lea |β| > 1 como sinergia sin descartar (b).**

Consecuencia empírica (notebook, HO): Naive Bayes ordena casi igual (Gini 0,565 vs 0,589) pero su PD media es 18,6% contra 11,8% observado y su pendiente de calibración es 0,45: sus log-odds son ~2,2 veces demasiado extremos porque suma tres veces la evidencia de utilización. La logística no es «mejor ranking»; es, sobre todo, **evidencia contada una sola vez**.

### 3.10 Grados de libertad escondidos: el WoE se estimó mirando $y$

La inferencia de §3.4 trata el regresor como fijo. Pero $\text{WoE}_k$ es función de $(m_k,b_k)$: la columna de WoE es una **codificación supervisada** (target encoding) estimada en la misma muestra. Consecuencia exacta, usando §3.7: el LR del modelo univariado en WoE es igual al LR del modelo saturado por bins,
$$\text{LR}_{\text{WoE}}=G^2_{K\times2}=2\sum_{k}\Big[m_k\ln\frac{m_k}{n_k M/n}+b_k\ln\frac{b_k}{n_kB/n}\Big]\ \overset{H_0}{\sim}\ \chi^2_{K-1},$$
pero el software lo compara con $\chi^2_1$. El tamaño real del test «p < 0,05» para una variable de puro ruido es $P(\chi^2_{K-1}>3{,}84)$: 0,05 con $K=2$; 0,15 con $K=3$; **0,43 con $K=5$**; 0,80 con $K=8$; 0,92 con $K=10$. La simulación del notebook (ruido, $n=3.322$, 5% malos, 200 réplicas) da 41,5% con 5 bins por cuantiles y 93% con 10. Si los cortes además se eligen maximizando IV (binning supervisado), la tasa sube a 96% con 5 bins y ni siquiera el LR contra $\chi^2_{K-1}$ la corrige (45%), porque la búsqueda de cortes consumió grados de libertad adicionales.

**El Wald de una variable en WoE es un test de IV.** Como $\hat\beta_1=-1$, $W=1/\operatorname{Var}(\hat\beta_1)=\sum_i\hat p_i(1-\hat p_i)(\text{WoE}_i-\overline{\text{WoE}}_w)^2\approx n\bar p(1-\bar p)\operatorname{Var}(\text{WoE})$. Para efectos chicos, $\text{IV}=\sum_k(\%b_k-\%m_k)\text{WoE}_k\approx\operatorname{Var}(\text{WoE})$ (ambos son $\approx\sum_k q_k\text{WoE}_k^2$ con $q_k$ la fracción del bin). Como $n\bar p(1-\bar p)=MB/n=1/(1/M+1/B)$:
$$W\ \approx\ \frac{\text{IV}}{1/M+1/B},\qquad E[\text{IV}\mid H_0]\approx (K-1)\Big(\frac1M+\frac1B\Big).$$
El notebook confirma el cociente (1,008 en promedio) y el IV medio del ruido (0,0246 simulado vs 0,0254 teórico). Es la misma estructura que el PSI (M08): **un IV sin tamaño muestral no es una cantidad inferencial**. En DEV de Austral ($M=165$), el umbral IV ≥ 0,10 equivale a $W\approx15{,}7$ ($z\approx3{,}96$), que contra la nula correcta $\chi^2_4$ es $p\approx0{,}003$. Con $n=30.000$ al 5%, el mismo IV = 0,10 sería $W\approx142$: el umbral de IV es de tamaño de efecto, no de significancia.

Remedios, de más barato a más limpio:

1. **Contrastar el LR contra $\chi^2_{K-1}$** (válido si los cortes no miraron $y$): 4,5% en la simulación.
2. **Cross-fitting**: cortes y WoE en una mitad, test en la otra (3,0% con cuantiles; 4,5% con binning supervisado). Costo: media muestra para cada paso.
3. **Permutación**: re-calcular *todo* el pipeline (binning + WoE + ajuste) con $y$ permutado; la distribución nula es exacta para ese pipeline. Costo computacional.
4. En el multivariado, el stepwise con p < 0,05 sobre WoE hereda el problema: el filtro de IV previo (M11) y la validación en HO/OOT son los que realmente protegen. Los p-valores del reporte final son descriptivos, no inferenciales.

### 3.11 Separación y la corrección de Firth

**Separación** (Albert & Anderson, 1984): completa si existe $\beta$ con $x_i^\top\beta>0$ para todo malo y $<0$ para todo bueno; cuasi-completa si se cumple con $\ge$ y algunas igualdades. En ambos casos $\ell$ no tiene máximo finito: $\sup\ell$ se alcanza solo cuando algún $|\beta_j|\to\infty$. En scorecards aparece al usar dummies por bin (o WoE sin suavizar) con un bin sin malos o sin buenos. Síntomas en el juguete del notebook (bin de 20 clientes, 0 malos): IRLS baja el β del bin ≈ 1 por iteración (−24 tras 25 iteraciones, SE ≈ $10^5$); `statsmodels` devuelve β = −21,3, SE = 2,8·10⁴ y solo un aviso de no convergencia. El Wald resultante tiene p ≈ 1 en el bin más limpio de la tabla: Hauck-Donner llevado al límite.

**Firth (1993)** maximiza la verosimilitud penalizada por el prior de Jeffreys:
$$\ell^*(\beta)=\ell(\beta)+\tfrac12\ln\lvert I(\beta)\rvert,\qquad U^*(\beta)=X^\top\big(y-p+h\odot(\tfrac12-p)\big),$$
con $h_i$ la diagonal de $H=W^{1/2}X(X^\top WX)^{-1}X^\top W^{1/2}$. Elimina el sesgo de orden $n^{-1}$ del MLE y siempre da estimaciones finitas (Heinze & Schemper, 2002).

**Caso saturado por bins.** Con una columna indicadora por bin, $X^\top WX=\operatorname{diag}(n_kw_k)$ y $h_i=w_k/(n_kw_k)=1/n_k$ para $i$ en el bin $k$. La componente $k$ del score modificado:
$$\sum_{i\in k}\Big(y_i-p_k+\tfrac1{n_k}(\tfrac12-p_k)\Big)=m_k-n_kp_k+\tfrac12-p_k=0\ \Longrightarrow\ \tilde p_k=\frac{m_k+\tfrac12}{n_k+1}.$$
**Firth en el modelo saturado = sumar 0,5 a cada celda** (malos y buenos). El notebook lo verifica a $2{,}5\cdot10^{-13}$ y obtiene para el bin sin malos β = −1,53 (SE 1,48). `statsmodels` no implementa Firth; en R está `logistf` (Heinze y coautores); en Python existen implementaciones de terceros (verificar mantenimiento antes de usarlas en producción). La implementación en numpy del notebook tiene 20 líneas.

**Precio de Firth**: el prior empuja las probabilidades hacia ½, así que se rompe $\sum\hat p=\sum y$ (65,26 vs 63 en el juguete). Puhr et al. (2017) proponen **FLIC**: pendientes de Firth, intercepto re-estimado por ML con las pendientes fijas como offset (restaura $\sum\hat p=\sum y$), y FLAC (aumentación de datos). Para PD, donde el nivel importa, FLIC es la variante sensata.

### 3.12 EPV: eventos por variable

Peduzzi et al. (1996) simularon logísticas con distintos EPV (malos por parámetro) y encontraron sesgo, cobertura pobre de intervalos y signos erróneos por debajo de ~10: de ahí la regla «10 EPV». Críticas (van Smeden et al., 2016): la regla no tiene base teórica sólida; lo que produce el sesgo es sobre todo la separación y el tamaño total, y el desempeño depende de la prevalencia, la fuerza de los predictores y su distribución. van Smeden et al. (2019) muestran que para desempeño predictivo el EPV explica poco; importan $n$, la fracción de eventos y el número de predictores por separado. Riley et al. (2019, 2020) proponen criterios de tamaño muestral basados en contracción esperada; el principal para binarios (fórmula de Riley et al. 2019, verificar en la fuente):
$$n\ \ge\ \frac{P}{(S-1)\ln\!\big(1-R^2_{\text{CS}}/S\big)},\qquad S=0{,}9,$$
con $P$ parámetros candidatos y $R^2_{\text{CS}}$ el $R^2$ de Cox-Snell anticipado. Con el modelo del notebook ($R^2_{\text{CS}}=0{,}095$, $P=8$) da $n\approx714$, ≈ 81 malos, EPV ≈ 10: coincide con la regla por casualidad de esta cartera, no por principio.

En Austral, HO con 78 malos y 9 parámetros (EPV 8,7) «contrasta, no re-estima». El experimento EPV del notebook (EPV 9, 150 submuestras) muestra el mecanismo: las variables de IV alto se estiman bien (meses_desde_mora: DE 0,19, nunca cambia de signo), pero deuda_otras_prom_12m (IV 0,015) tiene DE 1,16 para un β de −0,79 y cambia de signo en 20% de las réplicas. Eso es exactamente lo que el curso vio en HO. **El EPV correcto no es por variable sino por información**: una variable de IV bajo necesita muchos más malos que una de IV alto para el mismo SE, porque $\operatorname{Var}(\hat\beta_j)\approx1/[n\bar p(1-\bar p)\operatorname{Var}(\text{WoE}_j)(1-R^2_j)]$ con $R^2_j$ el de $\text{WoE}_j$ contra las demás (el VIF de M09).

### 3.13 Regularización: L2 hacia 0 o hacia −1

Penalizar $\ell(\beta)-\frac\lambda2\lVert\beta_{1:}-c\rVert^2$ equivale a un MAP con prior $\mathcal N(c,1/\lambda)$ para las pendientes. Newton: gradiente $X^\top(y-p)-\lambda(\beta-c)$, Hessiano $-(X^\top WX+\lambda I)$ (sin penalizar el intercepto). El ridge estándar ($c=0$) encoge hacia «la variable no aporta». Sobre WoE el centro natural es $c=-1$: «creo en la evidencia univariada completa» (Naive Bayes). Con muestras chicas (EPV 5, $n=355$, 80 réplicas), la log-verosimilitud media en HO pasa de −0,3123 (MLE) a −0,3068 (L2 hacia 0, mejor λ = 2) y a −0,3044 (L2 hacia −1, mejor λ = 5). Con DEV completo la diferencia es marginal. Dos cuidados: (i) la penalización no es invariante a escala y castiga más a las variables cuyo WoE se abre poco (deuda_otras_prom_12m cambia 0,063 con el default de sklearn); (ii) los SE del MLE ya no aplican.

L1 (lasso) hace selección, útil como alternativa al stepwise (M11), pero con WoE altamente correlacionados elige de forma inestable entre parientes.

### 3.14 Desbalance: submuestreo, pesos y corrección del intercepto

Si se muestrea con tasas $s_1$ (malos) y $s_0$ (buenos) que no dependen de $x$ dado $y$, por Bayes:
$$\operatorname{logit}P(y=1\mid x,\text{muestreado})=\operatorname{logit}P(y=1\mid x)+\ln\frac{s_1}{s_0}.$$
Si el modelo está bien especificado, solo cambia el intercepto. Con $\tau$ la tasa poblacional y $\bar y$ la de la muestra, $s_1/s_0=\frac{\bar y/(1-\bar y)}{\tau/(1-\tau)}$ y la **corrección previa** (*prior correction*, King & Zeng, 2001) es
$$\hat\beta_0^{\text{corr}}=\hat\beta_0-\ln\Big[\frac{1-\tau}{\tau}\cdot\frac{\bar y}{1-\bar y}\Big].$$
En el notebook, llevar DEV a 50% de malos da una corrección de 2,063; sin corregir, la PD media en HO es 42,2%; corregida, 11,7% (observado 11,8%). Los pesos `class_weight="balanced"` producen el mismo desplazamiento (42,0%) y requieren la misma corrección. Además, con pesos $(X^\top WX)^{-1}$ ya no es la varianza correcta (hace falta sándwich). En riesgo casi nunca conviene balancear: la logística no sufre por 5% de malos; sufre por **pocos malos en términos absolutos**, y submuestrear buenos bota información (87% de los buenos en el ejemplo).

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| Logística MLE sobre WoE (el curso) | No linealidad, escala y missing vía binning; 1 parámetro por variable; puntos por tramo | Pierde granularidad dentro del bin; p-valores optimistas (§3.10); forma fijada por el WoE univariado | Admisión con explicabilidad obligatoria; estándar de facto | Banca retail en general; supervisores esperan la estructura (ver Serie 1 · E6) |
| Logística sobre dummies por bin | Forma libre por bin (no restringida al WoE univariado) | $K-1$ parámetros por variable (31 vs 9 en el notebook); separación con bins chicos; monotonía no garantizada | Cuando el WoE univariado está muy confundido por otras variables y hay datos de sobra | Algunas implementaciones SAS de scorecards; poco común |
| Crudo + splines (lineales, restringidos) / GAM | Usa información dentro del bin; efectos suaves | Nudos, códigos especiales a mano, sin puntos por tramo; monotonía requiere restricciones | Modelos de comportamiento o de pricing con variables continuas limpias | Harrell (2015) lo recomienda en bioestadística; en crédito, uso interno/challenger |
| Firth / FLIC | Separación, sesgo en muestras chicas | Sesgo hacia ½ en predicciones (FLIC lo corrige); sin implementación en statsmodels | Carteras chicas, segmentos nuevos (p. ej., un producto de motos recién lanzado), bins extremos | Bioestadística; en crédito, low-default portfolios |
| L2 / L1 / elastic net | Varianza en muestras chicas; selección (L1) | Sesgo; inferencia clásica no aplica; no invariante a escala | Muchas candidatas o EPV bajo; challenger | ML en fintech; validadores piden justificar λ |
| Naive Bayes sobre WoE ($\beta_j=-1$) | Cero estimación; interpretable | Descalibrado con variables redundantes (pendiente 0,45 en el notebook) | Nunca en producción; útil como benchmark y como centro de regularización | Referencia académica (Hand & Yu 2001) |
| Logística bayesiana (priors informativos) | Incorporar conocimiento previo (p. ej., β del modelo anterior) | Elegir y defender priors | Recalibraciones con poca data nueva | Poco frecuente en banca; más en seguros |
| Probit / cloglog | Otro enlace (cloglog: asimetría, conexión con hazard) | Pierde la lectura WoE ↔ log-odds y el β = −1 | Cloglog para PD con horizonte y datos de tiempo discreto | Modelos de supervivencia en tiempo discreto |
| Boosting con restricciones monótonas | No linealidad e interacciones automáticas | Gobierno, explicación caso a caso, re-entrenamiento | Cobranza, fraude, campañas (clase 3, lámina 43) | Fintechs; en admisión, con SHAP y validación reforzada (ver Serie 1 · E5) |

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 Leer el β univariado sobre WoE como importancia.**
Síntoma: «todas las variables tienen β ≈ −1 en el univariado, todas pesan igual». Causa: §3.7, es −1 por construcción. Detección: mirar el SE o el IV, no β. Qué hacer: la importancia se lee en IV (univariado) y en el rango de puntos o $|\beta|\cdot\sigma(\text{WoE})$ (multivariado, clase 3 lámina 36; M14).

**5.2 Coeficiente positivo por supresión.**
Síntoma: β > 0 en una variable con WoE monótono y sentido de negocio claro (uso_tc_prom_12m +0,18 en el notebook). Causa: una pariente muy correlacionada absorbe la evidencia; la variable queda capturando el residuo con signo opuesto. Detección: correlación de WoE > 0,7 con otra variable del modelo; β univariado −1 y multivariado positivo; VIF. Qué hacer: sacar una de las dos (M09); nunca «forzar el signo» con restricciones sin entender el porqué.

**5.3 p-valores optimistas por grados de libertad escondidos.**
Síntoma: variables de ruido significativas; el stepwise deja entrar variables que no sobreviven en HO. Causa: §3.10, el WoE gasta $K-1$ grados de libertad y el binning supervisado aún más. Detección: simulación con $y$ permutado; comparar p-valor con LR contra $\chi^2_{K-1}$. Qué hacer: filtro de IV con umbral dependiente de $n$, cross-fitting o permutación; tratar los p-valores del reporte como descriptivos.

**5.4 Wald no significativo en el bin más extremo (Hauck-Donner).**
Síntoma: con dummies, el bin con 0–2 malos aparece con p > 0,05 y SE gigante. Causa: §3.6. Detección: comparar Wald vs LR; SE desproporcionado respecto del resto. Qué hacer: LR; fusionar bins (Serie 1 · M7); Firth.

**5.5 Separación silenciosa.**
Síntoma: β de −20 o más, SE de miles, un aviso de convergencia que nadie lee. Causa: bin sin malos o sin buenos con dummies o WoE crudo. Detección: `min(malos por bin) == 0`; bandera de convergencia; $\max|\beta|$ implausible. Qué hacer: suavizado (el +0,5 del curso), fusión de bins o Firth/FLIC. Test de CI que falle si hay bins con 0 malos en DEV.

**5.6 «El modelo calibra en DEV».**
Síntoma: PD media en DEV igual a la tasa de DEV, presentada como evidencia. Causa: §3.5, identidad del intercepto. Detección: la igualdad es exacta a $10^{-9}$ (una calibración real nunca es exacta). Qué hacer: calibración en HO/OOT y por bandas (M15, M19).

**5.7 Regularización accidental (sklearn).**
Síntoma: β de sklearn distintos a los de statsmodels. Causa: `LogisticRegression()` usa L2 con C = 1 por defecto. Detección: comparar motores (el notebook lo hace). Qué hacer: `C=np.inf` explícito (desde sklearn 1.8 `penalty=None` está deprecado) o usar statsmodels.

**5.8 Balancear sin corregir el intercepto.**
Síntoma: PD media 4 veces la tasa real; cortes de política mal puestos. Causa: §3.14. Detección: PD media en HO vs tasa observada. Qué hacer: no balancear; si se hizo, corrección de King & Zeng o re-estimar el intercepto en una muestra representativa (conecta con calibración, M15).

**5.9 «Cambio de signo» en el re-ajuste en HO con EPV bajo.**
Síntoma: la variable se invierte en HO. Causa: §3.12, DE comparable a β. Detección: el signo invertido no es significativo; la DE por bootstrap (Serie 1 · E3) es del orden de β. Qué hacer: documentar como «indistinguible de cero», como hizo el curso; descalificar solo una inversión significativa.

**5.10 Comparar β entre modelos anidados como si midieran lo mismo.**
Síntoma: «al agregar la variable X, el β de antigüedad subió de −1,0 a −1,14: hay interacción». Causa: no colapsabilidad (§3.9), el β condicional de la logística cambia al agregar cualquier predictor fuerte, aunque sea independiente. Detección: correlación entre WoE ≈ 0 y aun así el β se mueve. Qué hacer: comparar efectos en escala de probabilidad (efectos marginales promedio) o en puntos, no en β crudos.

**5.11 Firth sin FLIC en un modelo de PD.**
Síntoma: la PD media del modelo queda por encima de la tasa observada. Causa: el prior de Jeffreys empuja hacia ½. Qué hacer: FLIC (re-estimar el intercepto con offset).

---

## 6. Puente con ingeniería

En un pipeline declarativo, el ajuste es una etapa con **entradas congeladas** (bins y WoE de DEV, identificados por hash) y **salidas congeladas** (β, matriz de covarianza, bandera de convergencia). Un contrato de ejemplo:

```yaml
etapa: ajuste_logistico
entrada:
  datos: dev@sha256:…           # snapshot de DEV (M4)
  mapas_woe: woe_v3@sha256:…     # bins + WoE congelados (M7)
  variables: [uso_linea_prom_12m, meses_desde_mora_12m, …]
estimador:
  algoritmo: irls                 # o statsmodels.Logit(method="newton")
  penalizacion: ninguna           # explícito: evita el default de sklearn
  tol_beta: 1.0e-10
  tol_gradiente: 1.0e-6
  max_iter: 100
salida:
  beta: coeficientes_v3.json
  cov: cov_v3.npy
  convergio: true
invariantes: [convergencia, suma_p_igual_suma_y, sin_separacion, signos, paridad_motores, woe_unitario]
```

Invariantes verificables (tests de CI; todas están implementadas en el notebook):

1. **Convergencia real**: bandera `convergio` y $\lVert X^\top(y-\hat p)\rVert_\infty<10^{-6}$. Nunca aceptar el resultado de `max_iter`.
2. **Identidad del intercepto**: $|\sum\hat p_i-\sum y_i|<10^{-6}$ en DEV. Si falla, el modelo no tiene intercepto, no convergió o se aplicaron pesos sin declararlos.
3. **Test unitario del WoE (Zeng)**: para cada variable, re-calcular el WoE **sin suavizar** sobre DEV, ajustar el univariado y exigir $|\hat\beta_1+1|<10^{-8}$ y $|\hat\beta_0-\ln(M/B)|<10^{-8}$. Si falla, hay un bug de pipeline: bins mal alineados entre la tabla de WoE y los datos, un join que duplicó filas, un bin «MISSING» mapeado a 0 por error, o el WoE calculado sobre otra muestra. Es barato y detecta errores que ningún Gini detecta.
4. **Sin separación**: $\min_k m_k\ge m_{\min}$ y $\min_k b_k\ge b_{\min}$ en DEV (p. ej., 20; convención, no ley) y $\max|\beta|<5$.
5. **Contrato de signos**: todos los $\beta_j<0$ (con la convención del curso); una excepción debe estar declarada y justificada en la bitácora.
6. **Paridad de motores**: IRLS propio vs `statsmodels` a $10^{-8}$ en β y SE. Si difieren, uno de los dos no está resolviendo el mismo problema.
7. **Reproducibilidad**: versiones de numpy/statsmodels congeladas; el mismo input produce el mismo β bit a bit (o a $10^{-12}$ si hay BLAS multihilo).

Qué se **congela**: bins, WoE, β, covarianza, versión del estimador y tolerancias. Qué se **versiona**: el snapshot de DEV, la lista de variables y la bitácora de decisiones (variables forzadas o excluidas, clase 3 lámina 46). Qué se **recalcula** en cada corrida de monitoreo: nada de lo anterior; solo se aplica (clase 6: «en producción no existe DEV»).

Un detalle de diseño que Iair apreciará: la etapa de WoE y la de ajuste **no son separables estadísticamente** aunque lo sean en el DAG. Como el WoE mira $y$, cualquier inferencia sobre β debe hacerse re-ejecutando ambas etapas (permutación o bootstrap del pipeline completo, Serie 1 · E3), no solo la segunda.

---

## 7. Numpy desde cero vs librerías

| Cálculo | numpy (notebook) | Librería | Diferencias / convención | En producción |
|---|---|---|---|---|
| Ajuste MLE | `ajustar_logit`: IRLS con `np.linalg.solve` | `statsmodels.Logit(...).fit(method="newton")`; `sm.GLM(..., family=Binomial())` (IRLS) | Idénticos a $10^{-16}$ en β y SE | statsmodels (inferencia completa) o IRLS propio con test de paridad |
| Ajuste sin penalización en sklearn | — | `LogisticRegression(C=np.inf)` (L-BFGS) | Default **C = 1 = L2**; `penalty=None` deprecado en 1.8; coincide a ~$10^{-6}$; no entrega SE (el notebook los calcula con Fisher) | Solo con `C=np.inf` explícito y sin necesidad de inferencia |
| Errores estándar | $\sqrt{\operatorname{diag}((X^\top\hat WX)^{-1})}$ | `.bse` | Iguales; ninguno corrige por WoE estimado ni por clusters | `cov_type="cluster"` en statsmodels si hay dependencia |
| Wald / LR / score | `tests_drop_one` | `.bse`, dos `.llf`, `.score_test(exog_extra=…)` | Iguales a $10^{-6}$ relativo | LR para decisiones |
| Firth | `ajustar_firth` (score modificado, paso acotado) | No en statsmodels ni sklearn; R `logistf` | Validado contra el resultado analítico +0,5 | numpy propio con test analítico, o R |
| Ridge | `ajustar_ridge` (Newton penalizado, intercepto libre) | `LogisticRegression(C=1/λ)` | sklearn minimiza $C\sum\text{logloss}+\frac12\lVert w\rVert^2$ ⇒ $\lambda=1/C$; con `lbfgs` no penaliza el intercepto (verificado a $10^{-4}$); `liblinear` sí lo penaliza | sklearn con λ documentado; numpy si se necesita centro ≠ 0 |
| Pesos por clase | `w` en IRLS | `class_weight="balanced"` | Iguales; los SE del MLE ponderado no son válidos | Evitar; si se usan, sándwich + corrección de intercepto |
| Corrección de intercepto | una línea | — | King & Zeng (2001) | Siempre que la muestra no sea representativa |

`statsmodels` reporta separación con un aviso (`ConvergenceWarning` o `PerfectSeparationWarning` según el caso) y **devuelve igual los parámetros**: la bandera `mle_retvals["converged"]` debe leerse explícitamente.

---

## 8. Aplicación: casos y números

**8.1 Leer los β de Austral como descuentos.** Los ocho β van de −0,309 a −0,926. Con §3.9: deuda_interna_max_3m (−0,926) aporta 93% de su evidencia univariada (casi nada redundante: fuente propia, deuda con la institución); uso_tc_prom_3m (−0,309) aporta 31% (ya contada por uso_tc_prom_12m y uso_linea_prom_12m). Ninguno supera −1 ni es positivo: el filtro de correlación de 0,70 hizo su trabajo. El punto que el comité debe discutir no es el β, es el rango de puntos (clase 3), y el β = −1 explica por qué: el rango de puntos de una variable es $|\beta_j|\cdot\text{factor}\cdot(\text{WoE}_{\max}-\text{WoE}_{\min})$, producto del descuento y de la apertura del WoE.

**8.2 El intercepto de Austral.** En el multivariado el intercepto ya no es exactamente $\ln(M/B)$, pero queda cerca porque los WoE promedian cerca de 0 en la población (en el notebook: $\hat\beta_0=-2{,}0625$ vs $\ln(M/B)=-2{,}0628$). Para Austral, con 4,97% de malos en DEV, $\ln(M/B)\approx-2{,}95$; el curso reporta $\hat\beta_0=-3{,}03$ (clase 4): cerca, pero no igual, porque con 8 variables los WoE no promedian exactamente 0 y, por Jensen, la media de $\eta$ queda bajo $\operatorname{logit}(\bar y)$ aunque la media de $\hat p$ clave la tasa. En el scaling (M13), la base por variable depende de $\beta_0/n$: el intercepto no es decorativo.

**8.3 IV ≥ 0,10 como test.** Con $M=165$ y $B=3.157$, $1/M+1/B=0{,}00638$; IV = 0,10 ⇒ $W\approx15{,}7$. Contra $\chi^2_4$ (5 bins, cortes por cuantiles) es $p\approx0{,}003$. El filtro de IV del curso ya es un filtro de significancia estricto en DEV de Austral; en una cartera de 30.000 créditos no lo sería.

**8.4 El re-ajuste en HO (EPV 8,7).** Con 78 malos, la DE de un β de una variable de IV bajo es del orden del propio β (notebook: DE 1,16 para β −0,79 con EPV 9). «deuda_otras cambia de signo con p 0,78» es exactamente lo esperado si el β verdadero es −0,79 y la DE es ~1. La lectura del curso («indistinguible de cero, no una inversión») es la correcta; una regla operacional: descalificar solo si el β de HO es significativamente de signo contrario **y** fuera del intervalo de DEV.

**8.5 Crédito de motos (caso hipotético).** Una financiera lanza un producto de motos: 1.500 créditos maduros, 6% de malos (90 malos). Con 8 variables, EPV = 11: parece «cumplir la regla». Pero si dos variables tienen IV ≈ 0,05, su SE será comparable a sus β (§3.12), y si hay un bin «dealer premium» con 40 créditos y 0 malos, hay separación en dummies. Receta: (i) binning con mínimo de malos por bin (≥ 10–20), (ii) WoE suavizado (= Firth por bin), (iii) ajuste con FLIC o L2 hacia −1 con λ elegido por validación cruzada, (iv) reportar LR y no Wald, (v) usar Riley para decidir cuántas variables candidatas son defendibles con 90 malos. Con $R^2_{\text{CS}}\approx0{,}08$ (supuesto) y $S=0{,}9$, $n\ge P/(0{,}1\cdot\ln(1/(1-0{,}0889)))\approx P/0{,}0093$: con 1.500 créditos, $P\le13$ ($1.500\times0{,}0093=13{,}96$); con 8 variables sobra, pero cada dummy de canal o dealer cuenta como parámetro.

**8.6 Submuestreo en un proveedor.** Si un proveedor entrega un modelo entrenado «con clases balanceadas», su PD media será del orden de 40–50%, no del 5–15% de la cartera. Pida la tasa de la muestra de entrenamiento y aplique King & Zeng antes de cualquier comparación de PD, master scale o EL (M16, M17).

---

## 9. Preguntas de comité

**1. «¿Por qué todos sus coeficientes son negativos y cercanos a −1? ¿No debería cada variable tener su propia escala?»**
Porque las variables entran en WoE, que ya está en log-odds. Con una sola variable el MLE es exactamente −1; en el multivariado, cada β es la fracción de la evidencia univariada que sigue siendo nueva dado el resto. Un β de −0,31 no significa «variable débil», significa «redundante en un 69%». La importancia se discute en rango de puntos.

**2. «Su tabla dice p < 0,001 para las ocho. ¿Cuánta confianza tienen esos p-valores?»**
Son optimistas: los WoE se estimaron con el mismo target y cada variable gastó $K-1$ grados de libertad que el p-valor no cobra. Con 5 bins, una variable de ruido sale «significativa» ~43% de las veces. La protección real es la validación en HO/OOT y el filtro de IV previo; si se requiere inferencia formal, reportamos LR contra $\chi^2_{K-1}$ o p-valores por permutación del pipeline completo.

**3. «La PD media del modelo en desarrollo es exactamente la tasa observada. ¿Está bien calibrado?»**
Eso es una identidad de la ecuación del intercepto; ocurre con cualquier conjunto de variables. La calibración se evalúa en HO y OOT, y por bandas: ahí es donde mostramos la prueba binomial y Hosmer-Lemeshow.

**4. «uso_tc_prom_12m aparece con coeficiente positivo. ¿La tarjeta cargada reduce el riesgo?»**
No. Está en el modelo junto a uso_tc_prom_3m (correlación de WoE 0,91); la de 3 meses absorbe la evidencia y la de 12 meses queda como supresor con signo opuesto y no significativo (z = 1,39). Se excluye por redundancia; el Gini no cambia de forma material.

**5. «El re-ajuste en HO invierte el signo de deuda_otras. ¿Descalifica la variable?»**
Con 78 malos para 9 parámetros, la desviación estándar de ese β es del orden del propio β; la inversión tiene p 0,78: es ruido. Descalificaría una inversión significativa. Lo documentamos y lo ponemos en vigilancia.

**6. «¿Por qué no usan un modelo con dummies por tramo, que es más flexible?»**
Porque cuesta 31 parámetros en lugar de 9, no mejora el Gini fuera de muestra (0,587 vs 0,589 en nuestra réplica) y expone a separación en tramos chicos. El WoE es una restricción deliberada: los tramos de una variable se mueven juntos en proporción a su evidencia univariada.

**7. «El proveedor entrenó con 50/50 de buenos y malos. ¿Podemos usar sus PD?»**
No sin corregir el intercepto: el submuestreo desplaza el log-odds en $\ln(s_1/s_0)$. Con la corrección de King & Zeng el nivel se recupera si el modelo está bien especificado; lo verificamos contra la tasa observada en una muestra representativa.

**8. «¿Cuántos malos necesitan para re-desarrollar este modelo?»**
La regla de 10 por variable es orientativa y no tiene base sólida. Usamos el criterio de contracción de Riley et al.: con nuestro $R^2$ y 8 parámetros, del orden de 80 malos como mínimo; y exigimos más si hay variables de IV bajo, porque su error estándar depende de la información, no solo del conteo de malos.

---

## 10. Ejercicios

**Ejercicio 1 (a mano).** Una variable con 3 bins: (malos, buenos) = (30, 270), (15, 485), (5, 695). Calcule el WoE sin suavizar de cada bin, $\ln(M/B)$, y verifique que $\ln(m_k/b_k)=\ln(M/B)-\text{WoE}_k$ en los tres.

<details><summary>Solución</summary>

$M=50$, $B=1.450$, $\ln(M/B)=\ln(0{,}03448)=-3{,}3673$.
Bin 1: $\%b=270/1450=0{,}18621$, $\%m=30/50=0{,}6$; WoE $=\ln(0{,}31034)=-1{,}1701$. $\ln(30/270)=-2{,}1972$ y $-3{,}3673-(-1{,}1701)=-2{,}1972$. Coincide.
Bin 2: $\%b=0{,}33448$, $\%m=0{,}3$; WoE $=\ln(1{,}11494)=0{,}1088$. $\ln(15/485)=-3{,}4761=-3{,}3673-0{,}1088$. Coincide.
Bin 3: $\%b=0{,}47931$, $\%m=0{,}1$; WoE $=\ln(4{,}7931)=1{,}5672$. $\ln(5/695)=-4{,}9345=-3{,}3673-1{,}5672$. Coincide.
Por lo tanto $\hat\beta=(-3{,}3673,\,-1)$.
</details>

**Ejercicio 2 (derivación).** Demuestre que en una logística con intercepto y una dummy por bin (modelo saturado), $\hat p_k=m_k/n_k$ para cada bin, usando solo las ecuaciones de primer orden $X^\top(y-\hat p)=0$.

<details><summary>Solución</summary>

Reparametrice con una columna indicadora por bin (mismo espacio columna que intercepto + $K-1$ dummies). La ecuación de la columna del bin $k$ es $\sum_{i\in k}(y_i-\hat p_i)=0$. Todos los $i\in k$ tienen el mismo $\hat p_k$, así que $m_k-n_k\hat p_k=0$. La reparametrización no cambia el MLE de las probabilidades porque el conjunto de predictores lineales alcanzables es el mismo.
</details>

**Ejercicio 3 (Hauck-Donner).** Tabla 2×2: bin chico con $a$ malos y $b=60-a$ buenos; resto con 50 malos y 950 buenos. Calcule el Wald para $a=40$ y $a=58$. ¿Cuál tiene más evidencia contra $H_0$? ¿Qué dice el Wald?

<details><summary>Solución</summary>

$a=40$: $\hat\beta=\ln(40/20)-\ln(50/950)=0{,}693+2{,}944=3{,}638$; $\operatorname{Var}=1/40+1/20+1/50+1/950=0{,}0961$; $W=13{,}23/0{,}0961=137{,}7$.
$a=58$: $\hat\beta=\ln(29)+2{,}944=6{,}312$; $\operatorname{Var}=1/58+1/2+0{,}02+0{,}00105=0{,}5383$; $W=39{,}84/0{,}5383=74{,}0$.
La tasa 58/60 es mucho más extrema: la LR ($G^2$) sube de 142,6 a 283,4, pero el Wald cae casi a la mitad. Es el efecto Hauck-Donner: use LR.
</details>

**Ejercicio 4 (suavizado).** Con los datos del ejercicio 1 y el suavizado del curso (+0,5), ¿el MLE sobre el WoE suavizado dará $\hat\beta_1$ mayor o menor que −1? Argumente con §3.8 sin calcular el MLE y luego verifíquelo en el notebook.

<details><summary>Solución</summary>

$\varepsilon_k\approx-1/(2m_k)+1/(2b_k)$: $-0{,}0148$, $-0{,}0323$, $-0{,}0993$. El bin 3 (WoE más alto) tiene el $\varepsilon$ más negativo ⇒ $\operatorname{Cov}_w(\varepsilon,\widetilde{\text{WoE}})<0$ ⇒ $\hat\beta_1<-1$ (|β| > 1). La magnitud es del orden de $0{,}08/2{,}7\approx0{,}03$ (diferencia de $\varepsilon$ extremos sobre el rango de WoE), es decir $\hat\beta_1\approx-1{,}03$. El MLE exacto es $-1{,}025$ (con $\hat\beta_0=-3{,}379$ vs $\ln(M/B)=-3{,}367$). Para verificar: `ajustar_logit` sobre la columna de WoE suavizado.
</details>

**Ejercicio 5 (Naive Bayes).** Dos variables A y B, idénticas (B es una copia de A). Si el WoE de A en el bin del cliente es +1,2, ¿qué log-odds de malo da Naive Bayes? ¿Qué β estimará la logística para A y B? ¿Qué pasa con el solver?

<details><summary>Solución</summary>

NB: $\ln(M/B)-2{,}4$, cuenta dos veces la evidencia. La logística solo identifica $\beta_A+\beta_B=-1$ (con una sola variable el coeficiente es −1); $X$ no tiene rango completo, $X^\top WX$ es singular y Newton falla (`LinAlgError`) o, con un solver regularizado, reparte −0,5/−0,5. Es el caso extremo de «descuento por redundancia»: cada copia aporta 50% de su evidencia.
</details>

**Ejercicio 6 (grados de libertad).** Usted binea una variable de ruido en 8 bins por cuantiles, calcula WoE en DEV y el stepwise la testea con Wald. ¿Cuál es la probabilidad aproximada de que entre con p < 0,05? ¿Y si el umbral fuera p < 0,001?

<details><summary>Solución</summary>

La nula real del Wald ≈ LR es $\chi^2_7$. $P(\chi^2_7>3{,}84)\approx0{,}80$. Para $p<0{,}001$ el crítico es 10,83: $P(\chi^2_7>10{,}83)\approx0{,}15$. Aun con un umbral 50 veces más estricto, entra 15% de las veces. Por eso el filtro de IV previo y la validación fuera de muestra son indispensables.
</details>

**Ejercicio 7 (Firth).** En el juguete del notebook (bin 5 con 20 clientes y 0 malos), calcule a mano el log-odds de Firth del bin y la PD que implica. Compare con la PD del WoE suavizado del curso.

<details><summary>Solución</summary>

$\ln((0+0{,}5)/(20+0{,}5))=\ln(0{,}02439)=-3{,}7136$; PD $=0{,}5/21=2{,}38\%$. El WoE suavizado da exactamente el mismo log-odds tras restar la constante $c$ (§3.8): $-\widetilde{\text{WoE}}_5+c=-1{,}0503-2{,}6633=-3{,}7136$, con $c=\ln\frac{M+K/2}{B+K/2}=\ln(65{,}5/939{,}5)=-2{,}6633$.
</details>

**Ejercicio 8 (desbalance).** Población con 4% de malos; el modelo se entrenó con una muestra de 25% de malos y dio $\hat\beta_0=-0{,}9$. ¿Cuál es el intercepto corregido?

<details><summary>Solución</summary>

$\ln[(0{,}96/0{,}04)\cdot(0{,}25/0{,}75)]=\ln(24\cdot0{,}3333)=\ln 8=2{,}0794$. $\hat\beta_0^{\text{corr}}=-0{,}9-2{,}079=-2{,}979$.
</details>

**Ejercicio 9 (código).** Modifique `simular_gl_escondidos` para agregar la columna «rech permutación»: para cada réplica, estime la distribución nula del LR con 99 permutaciones de $y$ re-ejecutando binning + WoE + ajuste, y rechace si el LR observado supera el percentil 95. Verifique que la tasa de rechazo vuelve a ~5% incluso con binning supervisado.

<details><summary>Solución</summary>

Dentro del loop de réplicas: `lr_perm = [lr_de(x, rng.permutation(y)) for _ in range(99)]`, donde `lr_de` encapsula cortes (cuantiles o supervisados) + `_woe_codigos` + `ajustar_logit` + LR; rechazo si `lr > np.quantile(lr_perm, 0.95)`. Con 200 réplicas y 99 permutaciones son ~20.000 ajustes univariados (segundos con cuantiles; más con supervisado, reduzca la grilla). La tasa debe quedar en 5% ± 3 puntos (error de Monte Carlo con 200 réplicas: $\sqrt{0{,}05\cdot0{,}95/200}\approx1{,}5$ puntos).
</details>

**Ejercicio 10 (diseño).** Escriba el test de CI «woe_unitario» del §6 para un pipeline cuyo WoE se guarda suavizado. ¿Qué tolerancia usa y por qué?

<details><summary>Solución</summary>

El test no debe usar el WoE guardado (suavizado): re-calcula el WoE **crudo** desde los conteos $(m_k,b_k)$ de la tabla guardada, lo mapea a DEV con los mismos cortes y ajusta el univariado. Tolerancia $10^{-8}$ en $|\hat\beta_1+1|$ y $|\hat\beta_0-\ln(M/B)|$ (es una identidad exacta; el único error es de optimizador). Si algún bin tiene 0 malos o 0 buenos, el test debe fallar con mensaje explícito («separación: fusionar bin»). Adicionalmente, verificar que el WoE guardado coincide con el recalculado desde los conteos con el suavizado declarado (a $10^{-12}$): así se detecta tanto un bug de mapeo como una tabla de conteos desalineada.
</details>

---

## 11. Referencias

- **Albert, A. & Anderson, J. A. (1984).** «On the existence of maximum likelihood estimates in logistic regression models». *Biometrika*, 71(1), 1–10. — Definición formal de separación completa y cuasi-completa; por qué el MLE no existe.
- **Buse, A. (1982).** «The likelihood ratio, Wald, and Lagrange multiplier tests: an expository note». *The American Statistician*, 36(3), 153–157. — La geometría de los tres tests en una figura.
- **Firth, D. (1993).** «Bias reduction of maximum likelihood estimates». *Biometrika*, 80(1), 27–38. — La penalización de Jeffreys y el score modificado.
- **Good, I. J. (1950).** *Probability and the Weighing of Evidence*. Griffin. — El origen del «peso de la evidencia» como log-likelihood ratio.
- **Hand, D. J. & Yu, K. (2001).** «Idiot's Bayes: not so stupid after all?». *International Statistical Review*, 69(3), 385–398. — Por qué Naive Bayes ordena bien aunque la independencia sea falsa (y por qué calibra mal).
- **Harrell, F. E. (2015).** *Regression Modeling Strategies* (2.ª ed.). Springer. — Splines restringidos, «phantom degrees of freedom», contracción; la alternativa bioestadística al binning.
- **Hauck, W. W. & Donner, A. (1977).** «Wald's test as applied to hypotheses in logit analysis». *JASA*, 72(360), 851–853. — El Wald no monótono.
- **Heinze, G. & Schemper, M. (2002).** «A solution to the problem of separation in logistic regression». *Statistics in Medicine*, 21(16), 2409–2419. — Firth como remedio práctico a la separación, con intervalos por verosimilitud perfilada.
- **Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X. (2013).** *Applied Logistic Regression* (3.ª ed.). Wiley. — Referencia estándar de estimación, tests y diagnóstico.
- **King, G. & Zeng, L. (2001).** «Logistic regression in rare events data». *Political Analysis*, 9(2), 137–163. — Corrección previa del intercepto y corrección de sesgo en eventos raros.
- **McCullagh, P. & Nelder, J. A. (1989).** *Generalized Linear Models* (2.ª ed.). Chapman & Hall. — IRLS y Fisher scoring para toda la familia GLM.
- **Peduzzi, P., Concato, J., Kemper, E., Holford, T. R. & Feinstein, A. R. (1996).** «A simulation study of the number of events per variable in logistic regression analysis». *Journal of Clinical Epidemiology*, 49(12), 1373–1379. — El origen de la regla de 10 EPV.
- **Puhr, R., Heinze, G., Nold, M., Lusa, L. & Geroldinger, A. (2017).** «Firth's logistic regression with rare events: accurate effect estimates and predictions?». *Statistics in Medicine*, 36(14), 2302–2317 (verificar páginas). — FLIC y FLAC: cómo arreglar el sesgo de Firth en las predicciones.
- **Riley, R. D. et al. (2019).** «Minimum sample size for developing a multivariable prediction model: PART II – binary and time-to-event outcomes». *Statistics in Medicine*, 38(7), 1276–1296 (verificar páginas); y **Riley, R. D. et al. (2020).** «Calculating the sample size required for developing a clinical prediction model». *BMJ*, 368, m441. — Criterios de tamaño muestral por contracción que reemplazan al EPV.
- **Siddiqi, N. (2017).** *Intelligent Credit Scoring* (2.ª ed.). Wiley. — La práctica de industria de WoE + logística; útil para contrastar con la teoría de este módulo.
- **Thomas, L. C., Crook, J. & Edelman, D. (2017).** *Credit Scoring and Its Applications* (2.ª ed.). SIAM. — Fundamentos de scoring, incluida la relación entre logística y enfoques bayesianos.
- **van Smeden, M. et al. (2016).** «No rationale for 1 variable per 10 events criterion for binary logistic regression analysis». *BMC Medical Research Methodology*, 16, 163. — La crítica a la regla de 10 EPV.
- **van Smeden, M. et al. (2019).** «Sample size for binary logistic prediction models: Beyond events per variable criteria». *Statistical Methods in Medical Research*, 28(8), 2455–2474 (verificar páginas). — Lo que importa para predicción no es el EPV.
- **Zeng, G. (2014).** «A necessary condition for a good binning algorithm in credit scoring». *Applied Mathematical Sciences*, 8(65), 3229–3242. — Demuestra el resultado β = ±1, $\beta_0=\ln(\text{malos}/\text{buenos})$ con WoE univariado y lo usa como chequeo de binning.

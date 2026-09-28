# M14 · Interpretabilidad: contribuciones, monotonía, binning óptimo y reason codes

> **Ficha.** Profundiza la clase 3 (láminas 20–43: interpretación, estabilidad de coeficientes, rango de puntos, aportes, reason codes, optbinning, «¿puede la edad ser reason code?») y la clase 4 v2.1 (láminas 7–19: monotonicidad por contrato, granularidad del binning, missing −9/−99, cobertura, familias, «un Gini alto no cierra la revisión»).
> **Prerrequisitos:** Serie 1 · M7 (WoE/IV formal), E5 (scorecard vs ML), E6 (regulación); Serie 2 · M10 (logística sobre WoE), M12 (discriminación), M13 (scaling y tabla de puntos).
> **Archivos:** `M14_interpretabilidad_reason_codes.md` (este documento) · `M14_interpretabilidad_reason_codes.py` (notebook Marimo, ~30 s en CPU; `marimo edit --sandbox` instala dependencias).
> **Tiempo estimado:** 4–5 horas (2 de lectura, 2–3 de notebook y ejercicios).

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**Lo que se enseñó, con los números de Banco Austral.** Antes de firmar un scorecard se revisan signos, magnitudes, estabilidad y sentido de negocio. Los 8 coeficientes salieron negativos, como exige la convención WoE $=\ln(\%\text{buenos}/\%\text{malos})$. Se re-ajustó el mismo modelo en HO (1.397 créditos, 78 malos, 8,7 malos por parámetro, bajo la regla de 10–20): las variables fuertes contaron la misma historia (`uso_linea_prom_12m` −0,51 → −0,57; `meses_desde_mora_12m` −0,77 → −0,82; `antiguedad_meses` −0,61 → −0,90) y las dos que «cambiaron de signo» lo hicieron hacia cero (`deuda_interna_max_3m` −0,926 → +0,023, p = 0,96; `deuda_otras_prom_12m` −0,789 → +0,072, p = 0,78): **indistinguibles de cero, no inversiones**. Solo una inversión significativa descalifica.

La discusión de comité se hace sobre el **rango de puntos**, no sobre el coeficiente: `uso_tc_prom_12m` (β = −0,353) separa 55,8 puntos y es la primera; `deuda_interna_max_3m` (β = −0,926, el mayor en valor absoluto) es sexta con 26,6; `carga_financiera` (β = −0,418) es la última con 21,8. Se presentaron dos «aportes» normalizados, $|\beta|\cdot\sigma(\text{WoE})$ (7% a 20%) y $|\beta|\cdot\text{IV}$ (4% a 30%), con reglas de dedo: bajo 2% la variable sobra, sobre 60% hay concentración.

La tabla de puntos mostró **quiebres de monotonía**: `meses_desde_mora_12m` (75,3 → 52,7 → 64,7), `deuda_interna_max_3m` (73,6 → 58,9 → 71,8 → 85,4 → 75,4) y la forma casi plana con salto de `deuda_otras_prom_12m` (66,7 … 67,8 → 101,4, el *thin file* bancarizado). El «serrucho» de 1,4 puntos de `carga_financiera` se declaró ruido. La objeción se cerró con `optbinning` y `monotonic_trend`, en tres experimentos: embudo a mano (8 variables, Gini 0,757 / 0,698 / 0,675), optbinning con las mismas 8 (0,766 / 0,683 / 0,684: empate, «gana monotonía, no Gini») y optbinning sobre el pool de 94 con IV ≥ 0,10 (66 variables, 0,805 / 0,657 / 0,677: sobreajuste, y «nadie defiende 66 reason codes»). La clase 4 v2.1 agregó que una secuencia monótona puede ser **contraintuitiva** (deuda 69,0 → 76,3 puntos: más deuda, menos riesgo) y que eso se investiga, no se fuerza.

**Reason codes**: para cada cliente, brecha = máximo de la variable − puntos obtenidos; las tres mayores son los motivos. S0027717 (score 457, PD 73,9%): tarjeta cargada (−56), mora reciente (−53), utilización de la línea alta (−53). S0030427 (604, PD 1,7%): −38, −35, −22. S0018737 (740): un solo motivo de −7. El lab advirtió que la etiqueta es fija por variable y que en variables no monótonas puede mentir.

**Granularidad (clase 4 v2.1)**: `min_event_rate_diff` fija una diferencia absoluta entre tasas de bins contiguos; con tasa global 8%, exigir 4 pp puede dejar 2 bins (caso 1); con 0,2 pp y 8 bins aparecen tramos casi iguales (caso 2). **Missing**: un grupo «missing» con 4,1% de malos escondía −9 «nunca tuvo mora» (800 casos, 2,5%) y −99 «sin bureau» (100 casos, 17%). Cobertura de drivers, familias de variables y «un Gini alto no cierra la revisión».

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **Las medidas de importancia son convenciones.** Rango, $|\beta|\sigma$ y $|\beta|\text{IV}$ no se derivaron ni se compararon con medidas de interpretación estadística (caída de verosimilitud, Shapley). No se explicó que $|\beta|\cdot\text{IV}$ tiene una lectura exacta (sección 3.2).
2. **Estabilidad sin test.** «Indistinguible de cero» se leyó del p-valor de HO, sin un contraste formal DEV vs HO ni un cálculo de potencia: con 78 malos, *no detectar* una inversión dice poco.
3. **Monotonía como caja.** Se mostró que optbinning la garantiza, no cómo (PAV, programación entera) ni por qué una heurística puede quedar lejos del óptimo.
4. **Especiales.** El caso −9/−99 fue ilustrativo; no se mostró que el propio `binear()` del curso trata los códigos como números y los junta en `(-inf, -9]`.
5. **Un solo método de reason codes** (brecha contra el máximo), sin mencionar que la norma estadounidense describe otros dos, ni empates, umbrales, estabilidad o variables no permitidas.
6. **Regulación** como «legal en varias jurisdicciones»: aquí se precisa qué dice cada norma (verificado a septiembre de 2026).
7. **SHAP** se despachó como «aproximación sin lectura aditiva exacta». En un modelo aditivo sí es exacto, y coincide con uno de los métodos de reason codes (sección 3.3).

---

## 2. Intuición

**Toda explicación es una comparación.** Un scorecard es aditivo: $\text{score}=\sum_v \text{puntos}_v$. Preguntar «¿cuánto le restó `uso_linea` a este cliente?» no tiene respuesta hasta decir *contra qué*: contra el mejor bin posible, contra el cliente promedio, contra el cliente que apenas aprueba, contra un bin «neutro». Cada método de reason codes es una elección de **referencia**, y el módulo entero se ordena alrededor de esa idea.

**Tres preguntas distintas sobre «importancia».** (a) ¿Cuánto *puede* mover la variable? — rango de puntos, mejor vs peor bin, aunque el peor bin tenga al 3% de la población. (b) ¿Cuánto *mueve de hecho* en la población? — dispersión de sus puntos ($|\beta|\sigma$, SHAP medio) o separación entre buenos y malos ($|\beta|\cdot\text{IV}$). (c) ¿Cuánto *aporta dado el resto*? — lo que se pierde al sacarla y re-ajustar (drop-column). Las tres responden cosas diferentes y ordenan distinto; ninguna es «la» importancia. Para el comité importa (a) porque es lo que un cliente puede ganar o perder; para la parsimonia importa (c); para la concentración de riesgo, (b).

**La monotonía no es estética.** Un bin con menos riesgo que sus dos vecinos, en una variable ordinal, significa que el modelo cree que *empeorar* la variable mejora el score en ese tramo. Eso rompe tres cosas: la explicación al comité, la coherencia de los reason codes (la frase de la variable apunta en una dirección) y la robustez (un bin que se sale del patrón suele ser ruido de muestra, y el ruido no se repite en OOT). Pero hay quiebres reales (la relación deuda–riesgo con *thin file*) y quiebres fabricados (códigos especiales ordenados como números). El trabajo es distinguirlos antes de forzar nada.

**Juntar dos grupos distintos esconde a ambos.** Un bin con 900 clientes al 4,1% puede ser 800 al 2,5% y 100 al 17%. El promedio se ve razonable, el bin parece bueno, y el modelo le regala puntos a los 100 más riesgosos de la cartera. El error no aparece en ninguna métrica agregada porque los dos sesgos se compensan en el promedio.

**Un reason code es una estimación, no un hecho.** Si las dos brechas mayores de un cliente son 18 y 17 puntos, cambiar un β en su error estándar puede invertir el orden. El «motivo principal» de ese cliente depende de la muestra de desarrollo. Esto se mide, igual que se mide la estabilidad de los coeficientes.

---

## 3. Formalización

### 3.1 El scorecard como modelo aditivo

Con $n$ variables, WoE $w_v(x_v)$ (función escalonada del bin) y logística $\eta(x)=\ln\frac{p}{1-p}=\beta_0+\sum_v\beta_v w_v(x_v)$, el scaling del curso (ver M13) es

$$
\text{score}(x)=\text{offset}+\text{factor}\cdot\ln\frac{1-p}{p}=\text{offset}-\text{factor}\cdot\eta(x),
\qquad \text{factor}=\frac{\text{PDO}}{\ln 2},
$$

y el reparto del intercepto en partes iguales da

$$
\text{puntos}_v(x_v)=-\Big(\beta_v w_v(x_v)+\frac{\beta_0}{n}\Big)\text{factor}+\frac{\text{offset}}{n},
\qquad \sum_v \text{puntos}_v=\text{offset}-\text{factor}\Big(\beta_0+\sum_v\beta_v w_v\Big)=\text{score}.
$$

Definimos $g_v(x_v)=\text{puntos}_v(x_v)$. El score es $\sum_v g_v(x_v)$: **exactamente aditivo, sin interacciones**. Todo lo que sigue usa solo esa propiedad. Los **puntos neutros** (bin con WoE = 0) son $\text{puntos}^0=-\frac{\beta_0}{n}\text{factor}+\frac{\text{offset}}{n}$, iguales para todas las variables con este reparto.

### 3.2 Medidas de importancia

**Rango de puntos (curso).** $R_v=\text{factor}\cdot|\beta_v|\cdot(\max_b w_{v,b}-\min_b w_{v,b})$. Depende de los bins extremos, no de cuánta gente cae en ellos.

**Aporte $|\beta|\sigma(\text{WoE})$ (curso).** Si las $w_v$ fueran independientes, $\operatorname{Var}(\eta)=\sum_v\beta_v^2\operatorname{Var}(w_v)$ y la participación «natural» sería $\beta_v^2\sigma_v^2/\sum_u\beta_u^2\sigma_u^2$. El curso normaliza $|\beta_v|\sigma_v$ (desviaciones, no varianzas), que comprime las diferencias. Es un **coeficiente estandarizado**; la normalización a 100% es convención. Con WoE correlacionados (utilizaciones), la descomposición de la varianza tiene términos cruzados $2\beta_u\beta_v\operatorname{Cov}(w_u,w_v)$ que ninguna de las dos versiones reparte.

**Aporte $|\beta|\cdot\text{IV}$ — tiene una lectura exacta.** Sean $g_b=\%\text{buenos}$ y $m_b=\%\text{malos}$ del bin $b$. Entonces

$$
\text{IV}_v=\sum_b (g_b-m_b)\ln\frac{g_b}{m_b}=\sum_b g_b\,w_b-\sum_b m_b\,w_b
= E[w_v\mid\text{bueno}]-E[w_v\mid\text{malo}].
$$

El IV es la diferencia de WoE medio entre buenos y malos. Multiplicando por $\text{factor}\cdot|\beta_v|$:

$$
\text{factor}\cdot|\beta_v|\cdot\text{IV}_v=E[\text{puntos}_v\mid\text{bueno}]-E[\text{puntos}_v\mid\text{malo}],
$$

**cuántos puntos más obtiene, en promedio, un bueno que un malo por culpa de la variable $v$**. Sumando sobre $v$ se obtiene la diferencia de score medio entre buenos y malos, que es una medida de separación (pariente de la divergencia de M12). Es la versión más defendible de las dos convenciones del curso. En Austral: `uso_linea_prom_12m`, $0{,}505\times1{,}604\times28{,}85=23{,}4$ puntos de separación media; `deuda_interna_max_3m`, $0{,}926\times0{,}104\times28{,}85=2{,}8$.

**SHAP medio.** Con $\phi_v(x)=g_v(x_v)-E[g_v]$ (sección 3.3), $E|\phi_v|=\text{factor}\cdot|\beta_v|\cdot E|w_v-Ew_v|$: la desviación media absoluta de los puntos de la variable. Es $|\beta|\sigma$ con $E|\cdot|$ en vez de raíz de varianza: por eso ambas ordenan casi igual.

**Drop-column.** Re-ajustar sin $v$ y medir $\Delta\ell=\ell_{\text{completo}}-\ell_{-v}$. Bajo $H_0:\beta_v=0$, $\text{LR}=2\Delta\ell\sim\chi^2_1$ asintóticamente. Es la única de estas medidas que responde «¿qué pierdo si la saco?» y la que captura redundancia: si dos variables cuentan la misma historia, cada una por separado aporta poco aunque ambas tengan IV alto. La variante en discriminación es $\Delta\text{Gini}_{HO}$.

### 3.3 SHAP en un modelo aditivo: derivación

El valor de Shapley de la variable $v$ para la predicción $f(x)$ es

$$
\phi_v(x)=\sum_{S\subseteq N\setminus\{v\}}\frac{|S|!\,(M-|S|-1)!}{M!}\Big[\nu(S\cup\{v\})-\nu(S)\Big],
$$

con $M$ variables y una función de valor $\nu(S)$ que dice cuánto vale la predicción si solo se conocen las variables de $S$. La versión *interventional* (Janzing et al., 2020; es, por ejemplo, la de `shap.LinearExplainer` con `feature_perturbation="interventional"`) toma las variables ausentes de una distribución de fondo $\mathcal D$, independientemente de las presentes:

$$
\nu(S)=E_{X'\sim\mathcal D}\big[f(x_S,X'_{\bar S})\big].
$$

Con $f(x)=c+\sum_u g_u(x_u)$:

$$
\nu(S)=c+\sum_{u\in S}g_u(x_u)+\sum_{u\notin S}E_{\mathcal D}[g_u(X_u)].
$$

Entonces, para cualquier $S$ que no contiene a $v$,

$$
\nu(S\cup\{v\})-\nu(S)=g_v(x_v)-E_{\mathcal D}[g_v(X_v)],
$$

que **no depende de $S$**. Como los pesos de Shapley suman 1 sobre todos los $S\subseteq N\setminus\{v\}$ (hay $\binom{M-1}{s}$ subconjuntos de tamaño $s$, y $\sum_s\binom{M-1}{s}\frac{s!(M-s-1)!}{M!}=\sum_s\frac{1}{M}=1$),

$$
\boxed{\;\phi_v(x)=g_v(x_v)-E_{\mathcal D}[g_v]=\text{puntos}_v(x_v)-E_{\mathcal D}[\text{puntos}_v]\;}
$$

En log-odds, $\phi_v^{\eta}(x)=\beta_v\,(w_v(x_v)-E_{\mathcal D}w_v)$. La propiedad de eficiencia queda $\sum_v\phi_v=\text{score}(x)-E_{\mathcal D}[\text{score}]$. El notebook lo verifica enumerando las $2^8=256$ coaliciones con predicciones de `statsmodels`: diferencia máxima $\sim10^{-13}$ puntos.

Tres precisiones que importan en un comité:

1. **La escala manda.** En escala de probabilidad, $f=\sigma(\eta)$ ya no es aditiva: $\nu(S\cup\{v\})-\nu(S)$ depende de $S$ y $\phi_v^{PD}$ depende del nivel de las otras variables. En 20 rechazados del notebook, SHAP-PD y SHAP-puntos coinciden en el motivo principal en 85% de los casos y en el top-3 en 90%.
2. **El fondo manda.** Cambiar $\mathcal D$ (toda la población, solo aprobados, la banda del corte) cambia $\phi$ en una constante por variable, y por lo tanto el ranking. Es la misma elección de referencia de los reason codes.
3. **Interventional vs observacional.** La versión condicional $\nu(S)=E[f(X)\mid X_S=x_S]$ (Aas, Jullum y Løland, 2021) reparte crédito entre variables correlacionadas y ya no da $g_v-Eg_v$. Para explicar *qué hizo el modelo* con los datos de un cliente, la interventional es la natural; para explicar *qué dicen los datos*, la observacional. En reason codes se usa la primera, y hay que declararlo.

### 3.4 Estabilidad de coeficientes: test y potencia

Con DEV y HO independientes, bajo $H_0:\beta_v^{DEV}=\beta_v^{HO}$,

$$
z_v=\frac{\hat\beta_v^{DEV}-\hat\beta_v^{HO}}{\sqrt{se_{DEV}^2+se_{HO}^2}}\;\dot\sim\;N(0,1).
$$

Distinguimos: **inversión aparente** ($\operatorname{sign}\hat\beta^{HO}\neq\operatorname{sign}\hat\beta^{DEV}$), **inversión significativa** (además $|\hat\beta^{HO}/se_{HO}|>1{,}96$) y **diferencia significativa** ($|z_v|>1{,}96$). Solo la segunda descalifica una variable; la tercera obliga a investigar.

**¿Cuánta información trae HO?** La información de Fisher de la logística es $I(\beta)=X^\top WX$ con $W=\operatorname{diag}(p_i(1-p_i))$. Para una variable centrada y aproximadamente ortogonal al resto, $I_{vv}\approx\sum_i p_i(1-p_i)(w_{iv}-\bar w_v)^2\approx n\,\bar p(1-\bar p)\operatorname{Var}(w_v)$. Con $\bar p$ chico, $n\bar p\approx$ número de malos $B$:

$$
se(\hat\beta_v)\approx\frac{1}{\sqrt{B\,(1-\bar p)\operatorname{Var}(w_v)(1-R_v^2)}},
$$

donde $R_v^2$ es la colinealidad con las otras variables (el VIF de M09). El error estándar escala con $1/\sqrt{\text{malos}}$, no con $1/\sqrt{n}$: **los malos son la moneda de la estimación**. De ahí las dos cantidades que calcula el notebook:

$$
P(\text{inversión aparente})\approx\Phi\!\left(-\frac{|\beta_v|}{se_k}\right),\qquad
\text{potencia para detectar }\beta^{HO}=-\beta^{DEV}\approx 1-\Phi\!\left(1{,}96-\frac{2|\beta_v|}{se_k}\right),
$$

con $se_k=se_{HO}\sqrt{n_{HO}/n_k}$ al submuestrear HO a $k$ malos (la segunda ignora $se_{DEV}$, que es menor).

### 3.5 Monotonía y binning óptimo

**Pool-adjacent-violators (PAV).** Dadas tasas de pre-bins $r_1,\dots,r_K$ con pesos $n_k$, la regresión isotónica resuelve

$$
\min_{\theta_1\le\dots\le\theta_K}\sum_k n_k(r_k-\theta_k)^2 .
$$

PAV (Ayer et al., 1955; Barlow et al., 1972) la resuelve exactamente: recorre la secuencia y, cada vez que $\theta_k>\theta_{k+1}$, reemplaza ambos por su promedio ponderado y retrocede. El resultado son **bloques** de pre-bins contiguos con tasa común: un binning monótono. Dos propiedades usadas en el notebook:

- *Fusionar vecinos preserva la monotonía:* la tasa fusionada $\frac{n_k r_k+n_{k+1}r_{k+1}}{n_k+n_{k+1}}$ queda entre $r_k$ y $r_{k+1}$. Por eso las fusiones posteriores (tamaño mínimo, diferencia mínima, máximo de bins) no rompen lo que hizo PAV. Lo mismo vale para formas *peak/valley* (unimodales).
- *Fusionar nunca aumenta el IV.* El IV es la divergencia simétrica $D(g\Vert m)+D(m\Vert g)$ y, por la desigualdad log-sum, $(g_1+g_2)\ln\frac{g_1+g_2}{m_1+m_2}\le g_1\ln\frac{g_1}{m_1}+g_2\ln\frac{g_2}{m_2}$. Sin restricciones, el óptimo del IV es la partición más fina; las restricciones son las que eligen dónde perder.

**PAV no maximiza IV.** Minimiza error cuadrático con la restricción de orden; el problema del practicante es otro: elegir una partición de los pre-bins en intervalos contiguos que **maximice el IV** sujeto a monotonía, tamaño mínimo, separación mínima y número máximo de bins. Como el IV es separable por bin, se formula como partición de conjuntos (Navas-Palencia, 2020, con una formulación más compacta):

$$
\max_{x}\sum_{i\le j}V_{ij}\,x_{ij}\quad\text{s.a.}\quad
\sum_{i\le k\le j}x_{ij}=1\;\forall k,\quad
\sum_{i\le j}x_{ij}\le N_{\max},\quad
x_{ij}=0 \text{ si } n_{ij}<n_{\min},
$$

$$
x_{ij}+x_{j+1,l}\le 1\quad\text{si } r_{j+1,l}-r_{ij}<\delta\ (\text{ascendente}),\qquad x_{ij}\in\{0,1\},
$$

donde $x_{ij}=1$ si los pre-bins $i..j$ forman un bin, $V_{ij}=(g_{ij}-m_{ij})\ln(g_{ij}/m_{ij})$ su aporte al IV, $r_{ij}$ su tasa y $\delta$ = `min_event_rate_diff`. `optbinning` resuelve un modelo de este tipo con un solver CP-SAT/MIP (`solver="cp"` por defecto). Consecuencia verificable: **con los mismos pre-bins y restricciones, el IV de optbinning es ≥ el de cualquier heurística factible** (check del notebook). La heurística greedy del notebook (PAV + fusionar el par de menor pérdida de IV) empata en la mayoría de configuraciones, pero con `min_event_rate_diff = 0,04` en `uso_linea_prom_12m` queda en IV 0,593 contra 0,639 del óptimo: una fusión temprana miope le cierra el camino.

**Variantes de forma** (`monotonic_trend`): `ascending`/`descending` (tasa de malos creciente/decreciente en la variable), `peak`/`valley` (unimodal, con vértice elegido por el optimizador; en numpy se prueba cada posición), `auto`/`auto_asc_desc` (el algoritmo elige; `auto` de optbinning usa un clasificador de forma que puede devolver peak/valley).

### 3.6 Valores especiales: la aritmética de la mezcla

Si un bin contiene dos subgrupos con $(n_1,r_1)$ y $(n_2,r_2)$, su tasa es $r=\frac{n_1r_1+n_2r_2}{n_1+n_2}$ y su WoE $w=\ln\frac{G_1+G_2}{M_1+M_2}$ (en proporciones sobre totales). El sesgo para cada subgrupo es $w-w_k$: positivo (se le regalan puntos) para el de mayor riesgo, negativo para el de menor. En el generador: $w_{-9}=+0{,}42$, $w_{-99}=-0{,}68$, $w_{\text{mezcla}}=+0{,}22$. El sin bureau recibe $0{,}22+0{,}68=0{,}90$ unidades de WoE de más: con $\beta=-0{,}74$ y factor 28,85, son **19,3 puntos regalados** por la variable mora. Por la desigualdad log-sum, la mezcla además pierde IV.

### 3.7 Reason codes: un marco único

Para $\beta_v<0$ (convención del curso), la brecha del cliente $i$ en la variable $v$ contra una referencia $r_v$ es

$$
\text{brecha}_{i,v}=r_v-\text{puntos}_v(x_{iv})=\text{factor}\cdot|\beta_v|\cdot\big(w^{*}_v-w_{iv}\big),
$$

donde $w^*_v$ es el WoE que «vale» $r_v$. Los métodos solo difieren en $w^*_v$:

| método | $w^*_v$ | relación con SHAP |
|---|---|---|
| máximo (curso) | $\max_b w_{v,b}$ | $-\phi_v+\text{factor}\,\lvert\beta_v\rvert(\max_b w_{v,b}-Ew_v)$ |
| media poblacional (Reg B, método 2) | $E_{\text{pob}}[w_v]$ | exactamente $-\phi_v$ |
| media en el corte (Reg B, método 1) | $E[w_v\mid \text{score}\in[c,c+h)]$ | $-\phi_v$ con fondo = banda del corte |
| neutro | $0$ | $-\phi_v-\text{factor}\,\lvert\beta_v\rvert\,Ew_v$ |

Así, **todos los métodos son SHAP en puntos más una constante por variable**, $c_v=\text{factor}\,|\beta_v|(w_v^*-E w_v)$. El ranking cambia entre métodos solo por esas constantes. El método del máximo le suma a cada variable su «potencial al alza» (cuánto mejor es su mejor bin que el promedio): favorece a variables con un bin excepcional, aunque el cliente esté en el promedio en esa variable.

La constante del método neutro tiene signo conocido. Con proporción de malos $\bar p$, la población pondera los bins por $\pi_b=(1-\bar p)g_b+\bar p\,m_b$, así que

$$
E_{\text{pob}}[w_v]=(1-\bar p)\sum_b g_b\ln\frac{g_b}{m_b}+\bar p\sum_b m_b\ln\frac{g_b}{m_b}=(1-\bar p)\,D(g\Vert m)-\bar p\,D(m\Vert g),
$$

positiva salvo carteras muy malas: el cliente promedio tiene WoE > 0 y **la referencia neutra queda bajo la media** (en el notebook, 68,3 puntos vs medias de 68,5 a 72,4). El check del notebook verifica $r^{\text{media}}_v-r^{\text{neutro}}_v=-\beta_v\,\text{factor}\,E[w_v]$.

**Selección de motivos.** $\text{RC}_k(i)=$ las $k$ variables con mayor brecha sobre un umbral $\tau$, con desempate declarado. Tres parámetros de gobierno: $k$ (Reg B sugiere que más de 4 no ayuda), $\tau$ (una brecha de 2 puntos no es un motivo) y el desempate (con puntos enteros los empates son frecuentes).

**Estabilidad.** El orden entre las variables $u$ y $v$ para el cliente $i$ depende de $\Delta_{i}=\text{brecha}_{i,u}-\text{brecha}_{i,v}$, que hereda la incertidumbre de $\hat\beta$ y de los WoE. Aproximadamente $P(\text{el orden se invierte})\approx\Phi(-|\Delta_i|/sd(\hat\Delta_i))$: clientes con brechas parecidas tienen motivos inestables. La métrica práctica es la **retención del motivo principal bajo re-desarrollo bootstrap**.

**Coherencia de la etiqueta.** Una frase direccional («utilización alta») es correcta para todos los bins solo si los puntos son monótonos en la dirección que afirma la frase. En variables no monótonas la frase debe ser neutra («nivel de…») o depender del bin.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| Rango de puntos | Potencial de la variable en el score | Nulo | Discusión de comité, lectura de la tabla | Práctica estándar de scorecards (Siddiqi) |
| $\lvert\beta\rvert\sigma$, $\lvert\beta\rvert\cdot$IV normalizados | Contribución «realizada»; concentración | Nulo | Chequeo de que ninguna variable domine (>60%) o sobre (<2%) | Convención de industria (curso); umbrales sin base teórica |
| Drop-column (ΔLL, test LR, ΔGini HO) | Aporte marginal dado el resto; redundancia | K re-ajustes | Parsimonia, defensa de cada variable | Validación de modelos; estándar estadístico |
| Importancia por permutación | Lo mismo sin re-ajustar (modelos caros) | K×B predicciones | ML, cuando re-ajustar no es viable | Breiman (2001); Fisher, Rudin y Dominici (2019) |
| SHAP interventional en puntos | Contribución individual exacta en modelos aditivos | Cerrado: nulo | Reason codes, monitoreo de drivers | Equivale al método 2 de Reg B |
| SHAP (Tree/Kernel) en ML | Contribución individual aproximada en modelos no aditivos | Alto; depende de fondo y escala | Modelos de boosting en admisión (con validación aparte) | Práctica creciente; la CFPB exigió motivos específicos también con modelos complejos (circular 2022-03, retirada en 2025) |
| Binning por cuantiles + fusión manual | Simplicidad | Horas de modelador | Pocas variables, criterio experto | Curso (`binear`) |
| ChiMerge / árbol (CART) de pre-binning | Cortes supervisados | Bajo | Pre-binning | Pre-binning por defecto de optbinning |
| PAV / binning monótono heurístico | Monotonía garantizada | Bajo | Sin solver disponible; prototipos | Mironchyk y Tchistiakov (2017) |
| Binning óptimo por programación entera (optbinning) | Máximo IV con restricciones de forma, tamaño, separación | Solver CP/MIP; segundos | Producción; trazabilidad | Navas-Palencia (2020) |
| Especiales con bin propio | Separar mecanismos de ausencia | Nulo | Siempre que los códigos signifiquen cosas distintas | Clase 4 v2.1; `special_codes` en optbinning |
| RC contra el máximo | Motivos «cuánto le faltó para lo mejor» | Nulo | Tradición del curso y de muchos scorecards | No es uno de los dos métodos descritos por Reg B; aceptable solo si da resultados «sustancialmente similares» (verificar con asesoría legal) |
| RC contra la media poblacional | Motivos = −SHAP | Nulo | Por defecto razonable | Reg B, comentario 9(b)(2)-5, método 2 |
| RC contra la media en el corte | Motivos respecto de quien apenas aprueba | Requiere banda con n suficiente | Cuando el corte es estable | Reg B, 9(b)(2)-5, método 1 |
| RC contra WoE = 0 | Motivos respecto del «bin neutro» | Nulo | Pocas razones para preferirlo | Práctica de algunos proveedores (verificar) |
| Explicaciones contrafactuales | «Qué tendría que cambiar para aprobar» | Optimización por cliente | Comunicación al cliente, accionabilidad | Literatura de XAI; no reemplaza el aviso legal |

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 Rango inflado por un bin chico.** *Síntoma:* una variable encabeza el ranking por rango pero aporta poco ΔLL. *Causa:* un bin extremo con pocas observaciones y WoE extremo (en Austral, `deuda_otras` salta a 101,4 puntos arriba de 5,71 M). *Detección:* comparar rango con SHAP medio y ΔLL; reportar el tamaño del mejor y del peor bin junto al rango. *Qué hacer:* `min_bin_size`; discutir rango y ΔLL juntos.

**5.2 IV univariado engañoso en ambas direcciones.** *Síntoma:* en el generador, `uso_tc_prom_12m` tiene IV 0,42 pero aporta 2,0% de la ΔLL (lo que sabe ya lo dice `uso_linea`, correlación WoE 0,74); `deuda_otras_prom_12m` tiene IV 0,015 y test LR p = 0,011. *Causa:* el IV mira la variable sola; el modelo la usa condicional al resto. *Detección:* drop-column. *Qué hacer:* no usar IV como medida de importancia en el modelo final; usarlo solo como filtro de entrada (y entender que puede descartar efectos condicionales).

**5.3 Inversión aparente con pocos malos.** *Síntoma:* un β cambia de signo en HO. *Causa:* $se\propto1/\sqrt{\text{malos}}$. En el generador, con 78 malos (el tamaño de Austral), `deuda_otras_prom_12m` invierte su signo en 36,5% de las submuestras y `carga_financiera` en 24%, **sin ninguna inversión real** (HO sale del mismo proceso). Las inversiones significativas: 0% a 1%. *Detección:* test z y $P(\text{inversión})\approx\Phi(-|\beta|/se)$. *Qué hacer:* el criterio del curso (solo la inversión significativa descalifica) es correcto; reportar también la **potencia**: con 78 malos, la probabilidad de detectar una inversión completa en `deuda_otras` es ~26%. «No detecté inversión» con esa potencia no certifica estabilidad.

**5.4 Monotonía fabricada por códigos especiales.** *Síntoma:* en el generador, `meses_desde_mora_12m` con el binner del curso muestra un quiebre de 1,61 de WoE (34,6 puntos): el bin `(-inf, -9]` (9,3% de malos) queda antes que la mora reciente `(-9, 3]` (33,9%). *Causa:* −9 y −99 se ordenan como números. *Detección:* diagnóstico de monotonía que excluya especiales; inventario de códigos en el contrato de datos. *Qué hacer:* bins propios para cada código antes de cualquier restricción de monotonía.

**5.5 Monotonía forzada sobre una relación que no lo es.** *Síntoma:* forzar `ascending` en `deuda_otras_prom_12m` da IV 0,0166; `peak` da 0,0186, y el efecto real del generador (sobre 5 millones de pesos el riesgo *baja*) desaparece en el binning monótono. *Causa:* `auto_asc_desc` no permite formas unimodales. *Detección:* comparar IV y Gini HO entre formas; mirar la tasa por pre-bin. *Qué hacer:* elegir la forma por argumento de negocio documentado (`peak` para deuda si se entiende el mecanismo *thin file*), no por defecto.

**5.6 Monotonía correcta, dirección contraintuitiva.** *Síntoma:* clase 4 v2.1, lámina 9: puntos crecientes con la deuda (69,0 → 76,3). *Causa posible:* selección de aprobados (solo los mejores endeudados fueron aprobados), relación con el producto, definición imperfecta. *Detección:* revisar la relación en HO/OOT y en la población de solicitudes (TTD). *Qué hacer:* si no es defendible, imponer la tendencia acordada y re-estimar, o descartar; nunca forzar solo por estética.

**5.7 Restricciones de granularidad mal calibradas o heurística miope.** *Síntoma:* con `min_event_rate_diff = 0,04` y tasa global de 11%, `uso_linea` queda con 4 bins y la heurística pierde 7% de IV frente al óptimo (0,593 vs 0,639; Gini HO univariado 0,450 vs 0,485 con la configuración equilibrada). Con 0 y 10 bins, aparecen tramos de 3,3% y 4,0% (caso 2 del curso). *Detección:* reportar la separación mínima y el tamaño mínimo efectivos; comparar con el óptimo. *Qué hacer:* expresar `min_event_rate_diff` relativo a la tasa global (4 pp sobre 8% es la mitad de la tasa); usar el solver exacto en producción.

**5.8 Especiales mezclados.** *Síntoma:* en HO los sin bureau (−99) tienen tasa verdadera 19,4% (observada 24,1%, n = 133); el modelo con el bin mezclado les asigna PD media 8,6%; el separado, 15,0%. Los −9: verdad 7,7%, mezclado 9,1%, separado 7,9%. *Causa:* sección 3.6. *Detección:* calibración por código especial en HO/OOT. *Qué hacer:* bin propio por mecanismo; si el grupo es chico, bin propio igual (su WoE suavizado es ruidoso, pero no mezcla historias) y política externa si hace falta.

**5.9 Etiqueta direccional en variable no monótona.** *Síntoma:* en el generador, el bin de `deuda_otras` que más puntos pierde es (2,3; 4,5] millones (63,6 pts), mientras el de mayor deuda recibe 67,8. La frase «endeudamiento alto en otras instituciones» se la daría a quien debe 3 millones y no a quien debe 10. *Detección:* test de coherencia etiqueta–forma (sección 6). *Qué hacer:* frase neutra (Reg B, comentario 9(b)(2)-3 permite «nivel de…») o binning monótono.

**5.10 El método del máximo acusa lo que no es.** *Síntoma:* el cliente S000027 del notebook (score 522, rechazado con corte 538) tiene `uso_linea_prom_12m` = 0,44, *mejor que el promedio* (brecha contra la media −2,8 puntos). El método del máximo le da «utilización de la línea sostenidamente alta» como tercer motivo (brecha 14,3) porque la variable tiene un mejor bin muy bueno. *Causa:* la constante $c_v$ de la sección 3.7. *Detección:* comparar métodos; marcar motivos cuya brecha contra la media es negativa. *Qué hacer:* preferir referencias de media (Reg B) o exigir brecha positiva contra la media para reportar.

**5.11 Variables no accionables o sensibles como motivo.** *Síntoma:* `canal` aparece en el top-3 de 14% (máximo) a 48% (media en el corte) de los rechazados de OOT. *Causa:* con referencias de media, variables de rango chico pero con muchos clientes bajo el promedio suben al top-3. *Detección:* frecuencia de cada variable como motivo. *Qué hacer:* la decisión se toma **al diseñar el modelo**, no al imprimir la carta: bajo Reg B ningún motivo principal puede omitirse (comentario 9(b)(2)-4). Si una variable no se puede decir en voz alta, no entra.

**5.12 Empates.** *Síntoma:* con la tabla redondeada a enteros, 8,3% de los rechazados tiene empate exacto entre el motivo 3 y el 4. *Causa:* redondeo (M13). *Qué hacer:* regla de desempate determinista, documentada y testeada (aquí: mayor rango de puntos gana).

**5.13 Reason codes inestables.** *Síntoma:* re-desarrollando el modelo en 20 remuestras bootstrap, el motivo principal se mantiene en 87,4% de los rechazados con el método del máximo (p10: 78%) y en 91,8% a 93,0% con media o neutro. *Causa:* brechas cercanas; el método del máximo depende del mejor bin, que suele ser chico y ruidoso. *Qué hacer:* reportar la retención del top-1 como métrica de validación; umbral de brecha; binnings sin bins extremos minúsculos.

**5.14 SHAP sin escala ni fondo.** *Síntoma:* un proveedor entrega «SHAP» de un modelo de boosting. *Causa:* $\phi$ depende de la escala (log-odds vs PD) y de $\mathcal D$. *Qué hacer:* exigir ambas en la documentación y verificar eficiencia ($\sum\phi=f(x)-E f$).

**5.15 Atribuir el Gini a la herramienta equivocada.** *Síntoma:* en el generador, pasar a binning monótono sube el Gini HO de 0,582 a 0,603. *Causa real:* 0,017 de esos 0,022 vienen de separar −9/−99 (con el binner del curso y solo esa corrección, Gini HO 0,599). *Qué hacer:* ablación: cambiar una cosa a la vez antes de atribuir. La moraleja de Austral se mantiene: el binning óptimo compra forma, no Gini.

---

## 6. Puente con ingeniería

El scorecard y su capa de explicación son **artefactos de datos**, no código. Lo que se congela y versiona (con hash, ver M21 y clase 6) es:

```yaml
# scorecard_v3.yaml  (fragmento)
scaling: {pdo: 20, score_base: 600, odds_base: 50, reparto_intercepto: partes_iguales}
variables:
  meses_desde_mora_12m:
    especiales: {-9: "nunca tuvo mora", -99: "sin bureau"}   # contrato de datos
    tendencia: descending                                     # restricción declarada
    justificacion_tendencia: "mora más reciente = más riesgo"
    bins:   # [lim_inf, lim_sup) · woe · puntos (enteros, firmados)
      - {bin: "ESP -99", woe: -0.68, puntos: 53, frase: "sin información de bureau"}
      - {bin: "ESP -9",  woe:  0.42, puntos: 77, frase: "mora propia reciente"}
      - ...
    frase_por_defecto: "mora propia reciente"
reason_codes:
  metodo: media_poblacional        # Reg B 9(b)(2)-5, método 2
  referencia: {fuente: DEV, fecha: 2025-01-15, valores: {...}}   # congelada, no se recalcula en línea
  k: 4
  umbral_puntos: 3
  desempate: [rango_puntos_desc, nombre_asc]
  variables_no_reportables: []     # debe estar vacío: si no, la variable no puede estar en el modelo
```

**Invariantes que corren en CI** (tests que fallan el build):

1. `score == sum(puntos)` para un set de clientes de referencia (golden set), y `score == offset − factor·logit` si se conserva la versión continua.
2. `brecha == referencia − puntos`; los motivos son las `k` mayores sobre el umbral con el desempate declarado (test de propiedad con empates sintéticos).
3. **Coherencia etiqueta–forma:** para cada variable con frase direccional, los puntos de sus bins no especiales son monótonos en la dirección declarada. Si una variable no es monótona, debe tener frase por bin o frase neutra.
4. **Tendencias declaradas:** el binning congelado cumple `tendencia` en DEV; en HO/OOT se reportan violaciones (no se re-binea en producción: «en producción NO existe DEV», clase 6).
5. **Especiales:** todo código del contrato de datos tiene bin propio; un valor no visto va al bin que el contrato declara (no a «WoE = 0» por defecto silencioso, que es lo que hace `a_woe` del curso).
6. **Regla de edad** (si la edad estuviera en el modelo): el tramo ≥ 62 existe como bin propio y no tiene menos puntos que ningún tramo < 62 (lectura conservadora de §1002.6(b)(2); confirmar con asesoría legal).
7. `variables_no_reportables` vacío.

**Métricas de validación que se agregan al expediente** (M22): retención del top-1 bajo bootstrap, frecuencia de cada variable como motivo 1, acuerdo entre el método elegido y el método 2 de Reg B, fracción de empates.

**Auditoría por decisión:** se registra el score, los puntos por variable, las brechas y los motivos emitidos, con el hash del artefacto. Así una carta de rechazo se reproduce bit a bit años después; si la referencia se recalculara en línea, dos clientes idénticos en fechas distintas recibirían motivos distintos.

```python
def test_coherencia_etiquetas(scorecard):
    for v, spec in scorecard.variables.items():
        if spec.frase_direccional:
            pts = [b.puntos for b in spec.bins if not b.especial]
            d = np.diff(pts)
            assert (d >= 0).all() or (d <= 0).all(), f"{v}: frase direccional en variable no monótona"
```

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy en el notebook | Librería | Diferencias / convención | Producción |
|---|---|---|---|---|
| Logística | Newton-Raphson (IRLS) con información de Fisher | `statsmodels.Logit` | Coinciden en β, se y log-verosimilitud a $10^{-6}$ (assert). **Ojo:** `sklearn.LogisticRegression` usa penalización L2 con `C=1.0` por defecto; los β quedan contraídos. El Scorecard de optbinning del lab usa ese estimador por defecto. Usar `C=np.inf` para comparar (desde sklearn 1.8 el parámetro `penalty` está deprecado; ver M10 y M23) | statsmodels (inferencia) |
| AUC/Gini | Mann-Whitney con rangos promedio | `sklearn.metrics.roc_auc_score` | Idénticos; ambos tratan empates con 1/2 | sklearn |
| Test z DEV vs HO | `erf` vectorizada | `scipy.stats.norm.sf` | Idénticos | scipy |
| Test LR drop-column | $2\Delta\ell$ con numpy | `scipy.stats.chi2.sf` + `llf` de statsmodels | que yo sepa, statsmodels no trae un LR anidado genérico para Logit (su `llr` es contra el modelo nulo) | numpy + scipy |
| Binning monótono | PAV + fusiones greedy | `optbinning.OptimalBinning` (CP-SAT/MIP) | Con los mismos `user_splits`, IV óptimo ≥ heurístico (assert). optbinning usa intervalos **[a, b)** y el binner del curso `pd.cut` usa **(a, b]**: en variables enteras (meses de mora) cambian los bins. IV de optbinning sin suavizado; el curso suma 0,5 a cada celda (0,6532 vs 0,6521 en `uso_linea`). En optbinning 1.0.0, `binning_table` y `transform(metric="woe")` usan $\ln(\%\text{no evento}/\%\text{evento})$, el mismo signo del curso (verificado en el entorno; versiones antiguas: verificar) | optbinning |
| PAV puro | propio | `sklearn.isotonic.IsotonicRegression` | Misma solución (error cuadrático ponderado); sklearn no devuelve bloques, hay que reconstruirlos de los valores ajustados | cualquiera |
| SHAP | fórmula cerrada y enumeración de 256 coaliciones | `shap.LinearExplainer` (no instalado en el entorno) | Con `feature_perturbation="interventional"` explica el margen lineal (log-odds) como $\beta(x-\bar x)$ con $\bar x$ del fondo (verificar versión). Para puntos: multiplicar por −factor | fórmula cerrada (es exacta) |
| Scorecard | puntos con reparto en partes iguales | `optbinning.scorecard.Scorecard` | Parámetros `intercept_based`, `reverse_scorecard`, `rounding`; su reparto del intercepto puede diferir del curso: comparar scores, no puntos por bin | cualquiera, con test de paridad |

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral, releído

**Aportes.** Con la identidad $\text{factor}|\beta|\text{IV}=E[\text{puntos}\mid\text{bueno}]-E[\text{puntos}\mid\text{malo}]$, la tabla de la lámina 33 se lee en puntos: `uso_linea_prom_12m` 23,4; `uso_tc_prom_12m` $0{,}353\times1{,}713\times28{,}85=17{,}4$; `meses_desde_mora_12m` 12,2; `uso_tc_prom_3m` 8,9; `deuda_otras_prom_12m` 5,1; `carga_financiera` 4,5; `antiguedad_meses` 3,4; `deuda_interna_max_3m` 2,8. Suma: ~78 puntos de separación media entre buenos y malos (≈ 3,9 veces el PDO de 20). La variable con mayor β es la que menos separa.

**Reason codes de S0027717.** Cae en el peor bin de cinco variables, y en esas la brecha contra el máximo es el rango completo (55,8; 53,2; 52,9; 27,8; 21,8). Sin los puntos medios de DEV no podemos recalcular sus motivos por el método 2 de Reg B; con WoE fuertemente negativo en las tres primeras, el orden probablemente se mantiene. El caso donde los métodos divergen es el cliente medio (S0030427), cuyas brechas contra el máximo (−38, −35, −22) incluyen el «potencial al alza» de cada variable.

**Coeficientes en HO.** Con la aproximación $se\propto1/\sqrt{\text{malos}}$ y 78 malos, un β de −0,79 con $se_{HO}\approx0{,}26$ (valor ilustrativo del orden de los errores de Austral) tiene $P(\text{inversión aparente})\approx\Phi(-3)\approx0{,}1\%$; con $se\approx0{,}9$, ≈ 19%. La lectura del curso («−0,789 → +0,072 con p = 0,78 es ruido») es la correcta, pero hay que reportar además la potencia.

### 8.2 Banco Sintético (notebook)

**Muestras y modelo.** DEV 10.065 créditos (11,3% malos), HO 4.295 (507 malos), OOT 4.792 (15,4% por el deterioro plantado). 8 variables, todas con β < 0: `uso_linea` −0,590, `uso_tc_12m` −0,232, `meses_desde_mora` −0,743, `antiguedad` −1,130, `carga` −0,372, `consultas_6m` −0,344, `deuda_otras` −0,812, `canal` −1,187. Gini 0,533 / 0,582 / 0,546 (techo con la PD verdadera: 0,566 / 0,627 / 0,593). Score DEV: p5 503, mediana 562, p95 600.

**Importancia** (rango · $|\beta|\sigma$% · $|\beta|$IV% · SHAP medio · ΔLL% · ΔGini HO):

| variable | rango | $\lvert\beta\rvert\sigma$ | $\lvert\beta\rvert$IV | SHAP medio | ΔLL | ΔGini HO |
|---|---|---|---|---|---|---|
| meses_desde_mora_12m | 41,1 | 22,9% | 32,1% | 9,7 | 49,0% | 0,059 |
| uso_linea_prom_12m | 38,1 | 24,2% | 34,8% | 11,7 | 16,8% | 0,024 |
| antiguedad_meses | 28,0 | 18,3% | 9,7% | 9,2 | 21,0% | 0,018 |
| uso_tc_prom_12m | 12,7 | 7,9% | 9,4% | 3,6 | 2,0% | 0,002 |
| consultas_6m | 11,6 | 8,0% | 7,6% | 3,7 | 4,5% | 0,008 |
| canal | 10,7 | 7,1% | 1,5% | 3,2 | 3,5% | 0,001 |
| carga_financiera | 10,4 | 6,5% | 3,7% | 2,9 | 1,7% | 0,000 |
| deuda_otras_prom_12m | 8,1 | 5,1% | 1,2% | 2,2 | 1,4% | −0,000 |

`antiguedad_meses` (IV 0,088, bajo el umbral del curso) es la segunda en ΔLL: su efecto es lineal y fuerte en el generador, y su IV univariado lo subestima. Las ocho tienen test LR con p ≤ 0,011, pero tres aportan menos de 0,002 de Gini HO: significancia estadística y relevancia práctica son cosas distintas con 1.135 malos.

**Estabilidad.** Con HO completo no hay inversiones; el mayor $|z|$ es 1,10. Con 78 malos: inversión aparente en 36,5% (`deuda_otras`), 24% (`carga`), 21,5% (`canal`), 12% (`uso_tc_12m`); significativas ≤ 1%; potencia para detectar una inversión completa: 26%, 31%, 54% y 35%.

**Binning.** `uso_linea_prom_12m` con configuración equilibrada (5 bins, 1 pp, 5%): numpy y optbinning llegan a los mismos cortes (0,228 / 0,486 / 0,656 / 0,790), IV 0,653 vs 0,610 del binner del curso, Gini HO univariado 0,485 vs 0,477. `meses_desde_mora_12m` descendente con especiales aparte: numpy corta en 5 y 13 (IV 0,488), optbinning agrega un corte en 10 (IV 0,491).

**Especiales.** Bin mezclado en DEV: 2.241 créditos al 9,3% = (1.951 × 7,7% + 290 × 20,0%) / 2.241. Separar sube el Gini HO de 0,582 a 0,599 y acerca la PD del sin bureau a la verdad (8,6% → 15,0%, verdad 19,4%). En puntos de la variable mora (relativos al neutro), el sin bureau pasa de +4,6 a −15,1: mezclado, la mora *suma* y jamás podría ser su motivo de rechazo.

**Reason codes** (corte 538 = p25 de DEV; 1.319 rechazados en OOT; umbral 3 puntos):

- **S019904** (score 478, PD 57,7%, fue malo): los cuatro métodos dan mora reciente, utilización de la línea, antigüedad corta. Brechas contra el máximo 41 / 38 / 28; contra la media 33 / 21 / 11. Cliente malo en todo: los métodos coinciden.
- **S022516** (score 538, justo bajo el corte): máximo → línea, antigüedad, consultas; los otros tres → línea, consultas, antigüedad. Solo cambia el orden.
- **S000027** (score 522): máximo → mora, antigüedad, **utilización de la línea**; media → mora, **uso de tarjeta, canal**. Su `uso_linea` es 0,44, mejor que el promedio: el método del máximo le reprocha algo que no es un problema.

Acuerdo en el motivo principal entre métodos: máximo–media 80%, máximo–media en el corte 67%, media–media en el corte 87%, media–neutro 97%. Como motivo 1, el método del máximo elige `uso_linea` en 64% de los casos (la variable con el mejor bin más lejano de la media); el de la media poblacional reparte 47% línea y 46% mora.

**SHAP.** Fórmula cerrada = fuerza bruta a $10^{-13}$; SHAP-PD y SHAP-puntos coinciden en el top-1 en 85% de 20 rechazados.

**Edad.** IV 0,006, efecto verdadero cero; forzada, β = −0,82 (p = 0,06). El tramo que contiene a los 62+ (`(52, inf]`) recibe 60,4 puntos contra 63,7 del tramo 40–46: con la lectura conservadora de Reg B no pasa, y además 75,5% de ese tramo tiene menos de 62 años, así que la regla ni siquiera se puede verificar con ese binning.

### 8.3 Crédito de motos

Tres aplicaciones directas para una cartera de financiamiento de motos:

1. **Canal y concesionario.** El canal (concesionario, web, fuerza de venta) suele predecir porque proxy de selección y de fraude. Es una variable legítima para ranking, pero como motivo de rechazo es inaccionable («usted compró en el concesionario X»). El experimento de `canal` (hasta 48% de los rechazados con motivo «canal» según método) muestra la magnitud del problema. Opciones: sacarla del scorecard de admisión y usarla en la política de concesionarios, o aceptar que será motivo y redactar la frase.
2. **Edad del solicitante.** En motos la edad joven suele correlacionar con riesgo de siniestro y de mora. En Chile no conozco una regla equivalente a §1002.6(b)(2) (verificar), pero el principio del curso aplica: si no se puede decir en voz alta al cliente y al regulador, no entra. Si entra, el binning debe aislar los umbrales que importan.
3. **Especiales en bureau.** «Sin historia en bureau» es frecuente en primer crédito de moto (clientes jóvenes, informales). Es el caso −99: bin propio siempre, y política explícita si el volumen es chico (trampas de variables de bureau en Serie 1 · E4).

---

## 9. Preguntas de comité

**1. «¿Por qué la variable con mayor coeficiente es la que menos importa?»**
Porque el coeficiente multiplica WoE, y el WoE de esa variable casi no se mueve. Lo que el cliente gana o pierde es $\text{factor}\cdot|\beta|\cdot\Delta w$: el rango. Y lo que la variable separa en la población es $\text{factor}\cdot|\beta|\cdot\text{IV}$ = diferencia de puntos medios entre buenos y malos. En Austral, `deuda_interna_max_3m` (β = −0,926) separa 2,8 puntos en promedio; `uso_linea` (β = −0,505), 23,4.

**2. «Esta variable cambió de signo en HO. ¿La sacamos?»**
Solo si la inversión es significativa. Con 78 malos, una variable débil invierte su signo por azar en un 20%–35% de las muestras (lo mostramos con verdad conocida). Reportamos el test z de diferencia, el p-valor de β en HO y la potencia: si la potencia para detectar una inversión es 26%, el «no se invierte» tampoco certifica nada. La variable queda en vigilancia en OOT y en el monitoreo.

**3. «¿Por qué el binning monótono no mejoró el Gini?» / «¿Por qué sí lo mejoró?»**
Con el embudo bien hecho, el óptimo compra forma y trazabilidad, no Gini (Austral: 0,698 vs 0,683). Si mejora, se atribuye antes de celebrar: en nuestro caso, 0,017 de los 0,022 venían de separar códigos especiales, un error del binner, no un mérito de la monotonía.

**4. «¿Qué método de reason codes usan y por qué es aceptable?»**
Declaramos la referencia (media poblacional de DEV congelada), $k = 4$, umbral de 3 puntos y desempate por rango. Es el método 2 descrito por Reg B (comentario 9(b)(2)-5) y coincide con SHAP en puntos. Si se usa el del máximo (el del curso), hay que mostrar que los resultados son sustancialmente similares a los de un método descrito: en el notebook coinciden en el motivo principal en 80% de los rechazados, y hay casos donde el máximo reprocha una variable en que el cliente está sobre el promedio.

**5. «¿Puede la edad ser un reason code?»**
Bajo Reg B, si la edad está en un sistema empíricamente derivado y es un motivo principal, no se puede omitir (9(b)(2)-4); y solo se puede usar si el solicitante de 62+ no recibe un valor negativo (§1002.6(b)(2)). La respuesta de diseño: si no queremos que sea motivo, no puede estar en el modelo. En el generador no tiene señal (IV 0,006) y su tramo de 62+ recibiría menos puntos que otro tramo: no entra.

**6. «¿Qué pasa con los clientes sin información de bureau?»**
Tienen bin propio y lo validamos en HO/OOT. Mezclados con «nunca tuvo mora», el modelo les asignaba 8,6% de PD con una verdad de 19,4%, y les regalaba ~19 puntos. Si su volumen es chico, el WoE del bin propio es ruidoso; la respuesta es un bin propio más una política explícita, no juntarlos con otro grupo.

**7. «¿Cuán estables son los motivos que imprimimos?»**
Re-desarrollando el modelo en remuestras bootstrap de DEV, el motivo principal se mantiene en 92% de los rechazados con el método de la media (p10: 82%). Los que cambian son clientes con dos brechas casi iguales; para ellos el orden de los motivos 1 y 2 es arbitrario, y lo documentamos.

**8. «¿Esto nos pone bajo el AI Act?»**
Si operamos en la UE con personas naturales: la evaluación de solvencia está en el Anexo III como alto riesgo, pero un scorecard logístico podría quedar fuera de la definición de «sistema de IA» según las directrices de la Comisión (no vinculantes, debatido). Las obligaciones de alto riesgo se aplazaron al 2 de diciembre de 2027. El RGPD art. 22 aplica igual, y el TJUE (SCHUFA, 2023) consideró que el score puede ser una decisión automatizada. Para Chile, la Ley 21.719 (ver sección 11). Es una pregunta para el área legal con este módulo como insumo técnico.

---

## 10. Ejercicios

**E1. (Cálculo a mano)** Con los datos de la lámina 33 de Austral y factor 28,85, calcule la separación media en puntos entre buenos y malos que aporta cada variable ($\text{factor}|\beta|\text{IV}$) y su participación. ¿En qué variables cambia el ranking respecto de $|\beta|\cdot\text{IV}$ normalizado del curso?

<details><summary>Solución</summary>

Es la misma cantidad multiplicada por la constante 28,85, así que **el ranking y las participaciones no cambian** (30%, 22%, 16%, 11%, 7%, 6%, 4%, 4%): uso_linea 23,4; uso_tc_12m 17,4; mora 12,2; uso_tc_3m 8,9; deuda_otras 5,1; carga 4,5; antigüedad 3,4; deuda_interna 2,8. Lo que cambia es la **interpretación**: 23,4 puntos es una magnitud con unidades (más de un PDO), no un porcentaje. Suma ≈ 78 puntos: los buenos de DEV tienen, en promedio, ~78 puntos más que los malos. Esa suma es la diferencia de score medio entre buenos y malos, comparable con el PDO: 78/20 ≈ 3,9 duplicaciones de odds.
</details>

**E2. (Derivación)** Demuestre que $\text{IV}=E[w\mid\text{bueno}]-E[w\mid\text{malo}]$ y que fusionar dos bins nunca aumenta el IV.

<details><summary>Solución</summary>

$\text{IV}=\sum_b(g_b-m_b)w_b=\sum_b g_bw_b-\sum_b m_bw_b$; como $g_b=P(b\mid\text{bueno})$, $\sum_bg_bw_b=E[w\mid\text{bueno}]$, idem para malos. Para la fusión: $\text{IV}=\sum_b g_b\ln\frac{g_b}{m_b}+\sum_b m_b\ln\frac{m_b}{g_b}=D(g\Vert m)+D(m\Vert g)$. Por la desigualdad log-sum, $\sum_{k=1}^2 a_k\ln\frac{a_k}{b_k}\ge\big(\sum a_k\big)\ln\frac{\sum a_k}{\sum b_k}$, aplicada con $(a,b)=(g,m)$ y $(a,b)=(m,g)$: cada divergencia baja o se mantiene al fusionar; la igualdad vale si y solo si $g_1/m_1=g_2/m_2$ (mismo WoE). Por eso, sin restricciones, el máximo IV es la partición más fina.
</details>

**E3. (Derivación)** Muestre que en un scorecard el reason code «contra la media poblacional» es exactamente $-\phi_v$ (SHAP interventional con fondo = población) y que el método del máximo difiere en una constante por variable. ¿Cuándo dan el mismo ranking para todos los clientes?

<details><summary>Solución</summary>

Por la sección 3.3, $\phi_v=\text{puntos}_v-E[\text{puntos}_v]$, y la brecha contra la media es $E[\text{puntos}_v]-\text{puntos}_v=-\phi_v$. La brecha contra el máximo es $\max\text{puntos}_v-\text{puntos}_v=-\phi_v+c_v$ con $c_v=\max\text{puntos}_v-E[\text{puntos}_v]\ge0$. Los rankings coinciden para todos los clientes si y solo si $c_v$ es igual para todas las variables (si difieren, siempre existe un cliente con brechas contra la media lo bastante parecidas para que $c_u-c_v$ invierta el orden). En la práctica nunca son iguales: $c_v=\text{factor}|\beta_v|(\max w_v-Ew_v)$ depende del mejor bin de cada variable.
</details>

**E4. (Cálculo)** Un bin «missing» tiene 900 clientes al 4,1% = 800 al 2,5% (−9) + 100 al 17% (−99). La cartera tiene 10.000 clientes y 8% de malos. Calcule el WoE del bin mezclado y de cada subgrupo (sin suavizado), y cuántos puntos regala el bin mezclado al grupo −99 si $\beta=-0{,}75$ y factor 28,85.

<details><summary>Solución</summary>

Totales: 800 malos, 9.200 buenos. −9: 20 malos, 780 buenos → $w=\ln\frac{780/9200}{20/800}=\ln\frac{0{,}08478}{0{,}025}=1{,}221$. −99: 17 malos, 83 buenos → $w=\ln\frac{83/9200}{17/800}=\ln\frac{0{,}009022}{0{,}02125}=-0{,}857$. Mezcla: 37 malos, 863 buenos → $w=\ln\frac{863/9200}{37/800}=\ln\frac{0{,}09380}{0{,}04625}=0{,}707$. Al −99 le regala $0{,}707-(-0{,}857)=1{,}564$ unidades de WoE, es decir $0{,}75\times1{,}564\times28{,}85=33{,}8$ puntos (más de un PDO y medio: sus odds quedan sobreestimadas ~3,2 veces). Al −9 le quita $0{,}75\times0{,}514\times28{,}85=11{,}1$ puntos.
</details>

**E5. (Potencia)** En HO, con 507 malos, `carga_financiera` tiene $\hat\beta_{DEV}=-0{,}372$ y $se_{HO}=0{,}199$. Estime $P(\text{inversión aparente})$ y la potencia para detectar una inversión completa si HO tuviera solo 78 malos. Compare con la simulación del notebook.

<details><summary>Solución</summary>

$se_{78}\approx0{,}199\sqrt{507/78}=0{,}199\times2{,}55=0{,}507$. $P(\text{inversión})\approx\Phi(-0{,}372/0{,}507)=\Phi(-0{,}73)=0{,}23$. Potencia: $1-\Phi(1{,}96-2\times0{,}372/0{,}507)=1-\Phi(0{,}49)=0{,}31$. El notebook simula 24% de inversiones aparentes (analítico 23,2%) y reporta potencia 31%. Con 78 malos, en 7 de cada 10 muestras una inversión *real y completa* pasaría inadvertida para esta variable.
</details>

**E6. (Diseño)** Diseñe la regla de reason codes para una cartera de motos: método, $k$, umbral, desempate, tratamiento de variables no monótonas y de `canal`. Justifique cada elección con un número o una norma.

<details><summary>Solución</summary>

Una respuesta defendible: método 2 de Reg B (media poblacional de DEV, congelada con el artefacto), porque coincide con SHAP, es estable (92% de retención del top-1 en el notebook vs 87% del máximo) y está descrito en el comentario 9(b)(2)-5. $k=4$ (Reg B: más de 4 no ayuda; FCRA limita los key factors a 4). Umbral: 3 puntos (≈ 15% de un PDO; una brecha menor cambia las odds en menos de ~11%, pues $2^{3/20}=1{,}11$). Desempate: mayor rango, luego orden alfabético (determinista). Variables no monótonas: frase neutra o binning monótono (test de coherencia en CI). `canal`: fuera del scorecard de admisión y dentro de la política de concesionarios, porque sería motivo en hasta 48% de los rechazados y no es accionable. Todo queda en el YAML del artefacto y en el expediente.
</details>

**E7. (Código)** Modifique `binning_monotono` del notebook para que, al violar `min_event_rate_diff`, fusione el par con **menor diferencia de tasa** en vez del de menor pérdida de IV. Compare el IV con la versión original y con optbinning en `uso_linea_prom_12m` con `min_event_rate_diff = 0,04`.

<details><summary>Solución</summary>

Reemplazar la selección `i = cand[argmin(perd)]` por `i = cand[argmin(np.abs(np.diff(rr))[cand])]`. Resultado esperado: IV igual o menor que la versión por pérdida de IV (ambas ≤ 0,639 del óptimo). La lección: cualquier regla greedy puede quedar atrapada; el solver exacto explora combinaciones de fusiones que ninguna regla local ve. En producción se usa el exacto y la heurística solo como prueba de humo (su IV es una cota inferior).
</details>

**E8. (Código)** Extienda la celda de estabilidad bootstrap para reportar, por cliente, la probabilidad de que su motivo 1 cambie, y grafique esa probabilidad contra la diferencia entre sus dos brechas mayores. ¿Qué forma espera?

<details><summary>Solución</summary>

Guardar `_t[:, 0]` de cada réplica en una matriz (B × clientes), calcular `np.mean(mat != base[:,0], axis=0)` y graficar contra `brecha_1 − brecha_2` del modelo base. Se espera una curva decreciente tipo $\Phi(-\Delta/sd)$ (sección 3.7): cerca de 50% cuando $\Delta\approx0$ y cercana a 0 para $\Delta$ mayor que ~3 veces la desviación de la diferencia (típicamente 5–10 puntos). Sirve para fijar el umbral de «motivo principal confiable».
</details>

**E9. (Regulación aplicada)** Un proveedor le ofrece un modelo de boosting con «reason codes SHAP». Liste cinco preguntas que le haría antes de aceptar sus motivos en cartas de rechazo.

<details><summary>Solución</summary>

(1) ¿En qué escala se calcula SHAP (margen/log-odds o probabilidad)? (2) ¿Cuál es la distribución de fondo, y está congelada? (3) ¿Interventional u observacional? (4) ¿Cómo se agregan variables correlacionadas o derivadas en un motivo («familia») y cómo se mapea a frases que describan el factor realmente usado (Reg B 9(b)(2)-2)? (5) ¿Qué estabilidad tiene el motivo principal ante re-entrenamiento y ante semillas? Y, transversal: ¿cumplen eficiencia ($\sum\phi=f(x)-Ef$) en una muestra de auditoría?
</details>

---

## 11. Referencias

**Regulación (verificada en septiembre de 2026; no es asesoría legal).**

- **EE.UU., Regulation B (12 CFR 1002), §1002.9(b)(2) y comentario oficial.** El acreedor debe dar los motivos principales de una acción adversa; no se exige un número, pero «más de cuatro probablemente no ayuda» (9(b)(2)-1); los motivos deben describir los factores realmente considerados (-2); no hace falta decir cómo afectó el factor, p. ej. «tiempo de residencia» (-3); en scoring, ningún factor que sea motivo principal puede omitirse, aunque su relación con el riesgo no sea clara para el cliente («antigüedad del automóvil») (-4); dos métodos descritos para seleccionar motivos: brecha contra el promedio de quienes quedan en o apenas sobre el puntaje mínimo de aprobación, o contra el promedio de todos los solicitantes, y cualquier método con resultados «sustancialmente similares» (-5). Fuente: consumerfinance.gov, Regulation B, interpretaciones de §1002.9.
- **Regulation B, §1002.6(b)(2).** En un sistema de scoring «empíricamente derivado, demostrable y estadísticamente sólido» se puede usar la edad, siempre que a un solicitante de 62 años o más no se le asigne un factor o valor negativo.
- **FCRA, 15 U.S.C. §1681g(f)(1)(C) y (f)(9).** La divulgación del score incluye hasta 4 factores clave que lo afectaron negativamente; el número de consultas se agrega aunque exceda ese límite. Relación con el aviso de acción adversa (§1681m): verificar redacción vigente.
- **CFPB, Circulares 2022-03 y 2023-03** (motivos específicos con algoritmos complejos; uso de los formularios modelo). **Retiradas el 12 de mayo de 2025** en la retirada masiva de guías; los requisitos de Reg B siguen vigentes. Útiles como lectura de criterio, no como norma.
- **UE, Reglamento (UE) 2024/1689 (AI Act), Anexo III, punto 5(b).** Alto riesgo: sistemas para evaluar la solvencia de personas naturales o establecer su score crediticio, excepto detección de fraude. **Art. 86:** derecho a explicaciones claras y significativas del papel del sistema en la decisión. **Reglamento (UE) 2026/1744 («Digital Omnibus on AI»)**, en vigor desde el 27 de julio de 2026 según fuentes secundarias, aplaza las obligaciones del Anexo III al 2 de diciembre de 2027 (verificar en el Diario Oficial). Las **directrices de la Comisión sobre la definición de sistema de IA** (febrero de 2025, no vinculantes) sugieren que métodos de optimización tradicionales como la regresión logística pueden quedar fuera de la definición: su aplicación a scorecards es discutida.
- **RGPD art. 22** y **TJUE C-634/21 (SCHUFA, 7-12-2023):** el score puede constituir una decisión automatizada cuando es determinante para la decisión del tercero. **TJUE C-203/22 (Dun & Bradstreet Austria, 27-2-2025):** derecho a una explicación del procedimiento y los principios aplicados en decisiones automatizadas de solvencia.
- **Chile, Ley 21.719** (publicada en diciembre de 2024; modifica la Ley 19.628). Según fuentes secundarias, incorpora un art. 8° bis sobre **decisiones individuales automatizadas** (derecho a oponerse y a no ser objeto de decisiones basadas en tratamiento automatizado, incluida la elaboración de perfiles, que produzcan efectos jurídicos o afecten significativamente, con excepciones y salvaguardas como intervención humana y revisión); no pude acceder al texto en BCN: verificar redacción exacta, en particular si exige que la decisión sea «únicamente» automatizada. Entrada en vigencia: 1 de diciembre de 2026. El Ejecutivo ingresó el 31 de agosto de 2026 un proyecto (boletín 18.623-07) para postergarla al 1 de diciembre de 2027; al 24 de septiembre de 2026 seguía en primer trámite en el Senado, sin aprobación. No conozco una norma chilena que, como Reg B, obligue a entregar motivos de rechazo de crédito (verificar).

**Métodos.**

- Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2.ª ed. Wiley. — La referencia de práctica: rango de puntos, reason codes, revisión de monotonía.
- Thomas, L., Crook, J. y Edelman, D. (2017). *Credit Scoring and Its Applications*, 2.ª ed. SIAM. — Fundamentos estadísticos del scorecard y su interpretación.
- Navas-Palencia, G. (2020). «Optimal binning: mathematical programming formulation». arXiv:2001.08025. — La formulación detrás de optbinning; leer la sección de restricciones de monotonía.
- Mironchyk, P. y Tchistiakov, V. (2017). «Monotone optimal binning algorithm for credit risk modeling». Working paper (verificar edición). — Algoritmo monótono heurístico que motivó varias implementaciones.
- Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T. y Silverman, E. (1955). «An empirical distribution function for sampling with incomplete information». *Annals of Mathematical Statistics* 26(4). — Origen de PAV.
- Barlow, R. E., Bartholomew, D. J., Bremner, J. M. y Brunk, H. D. (1972). *Statistical Inference under Order Restrictions*. Wiley. — Regresión isotónica y sus propiedades.
- Lundberg, S. y Lee, S.-I. (2017). «A Unified Approach to Interpreting Model Predictions». *NeurIPS*. — SHAP.
- Štrumbelj, E. y Kononenko, I. (2014). «Explaining prediction models and individual predictions with feature contributions». *Knowledge and Information Systems* 41(3). — Shapley para explicación de predicciones, antes de SHAP.
- Janzing, D., Minorics, L. y Blöbaum, P. (2020). «Feature relevance quantification in explainable AI: A causal problem». *AISTATS*. — Por qué la versión interventional es la que explica al modelo.
- Aas, K., Jullum, M. y Løland, A. (2021). «Explaining individual predictions when features are dependent». *Artificial Intelligence* 298. — La alternativa condicional y cuándo importa.
- Fisher, A., Rudin, C. y Dominici, F. (2019). «All Models are Wrong, but Many are Useful». *JMLR* 20. — Importancia por permutación con fundamento (*model reliance*).
- Bracke, P., Datta, A., Jung, C. y Sen, S. (2019). «Machine learning explainability in finance: an application to default risk analysis». Bank of England Staff Working Paper 816. — Explicabilidad aplicada a default, desde un supervisor.
- Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T. y Walther, A. (2022). «Predictably Unequal? The Effects of Machine Learning on Credit Markets». *Journal of Finance* 77(1). — Cómo modelos más flexibles redistribuyen el acceso entre grupos: la base empírica de la discusión de equidad.
- Navas-Palencia, G. Documentación de `optbinning` (versión 1.0.0 usada aquí). — Parámetros `monotonic_trend`, `min_event_rate_diff`, `special_codes`, `user_splits`.

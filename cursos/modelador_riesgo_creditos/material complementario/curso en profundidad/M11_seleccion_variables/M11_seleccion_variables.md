# M11 · Selección de variables: stepwise y sus alternativas

> **Ficha.** Profundiza la clase 3 (paso 5 del embudo del Banco Austral: 26 → 8, lámina «Stepwise: cómo entra (y sale) cada variable», RESERVA «¿Y si el comité pide una variable que el stepwise dejó fuera?») y el Lab 2 §5 (Financiera Andes). **Prerrequisitos:** Serie 1 · M7 (WoE/IV), Serie 2 · M09 (redundancia y colinealidad) y M10 (logística sobre WoE desde la verosimilitud). **Archivos:** `M11_seleccion_variables.md` (este documento) y `M11_seleccion_variables.py` (notebook Marimo: `marimo edit --sandbox M11_seleccion_variables.py`). **Tiempo estimado:** 3 h de lectura + 2 h de notebook.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**El procedimiento.** En el Banco Austral, después de estabilidad (108 → 94), poder (IV ≥ 0,10: 94 → 63), redundancia (correlación de WoE > 0,70: 63 → 26) y VIF (máximo 2,97: 26 → 26), el último filtro es un **stepwise forward con revisión backward** sobre los WoE de DEV (3.322 solicitudes, 165 malos):

1. Partir sin variables; probar cada candidata; **entra** la que más mejora la log-verosimilitud, si su p-valor (Wald, el `pvalues` de `statsmodels`) es < 0,05.
2. Tras cada entrada, revisar las incluidas: si alguna perdió significancia, **sale la peor** y se re-estima antes de volver a mirar.
3. Parar cuando ninguna candidata aporta o al llegar a **14 variables** (tope de parsimonia del curso).

Tres criterios a la vez: significancia, **signo esperado** (con WoE = ln(%B/%M), todos los β deben ser negativos) y aporte marginal. «El signo manda sobre la significancia»: un coeficiente muy significativo con signo equivocado se descarta. En Austral entraron 8 y no salió ninguna; la trayectoria de log-verosimilitud fue −540,7 → −512,8 → −501,7 → −495,4 → −491,1 → −486,7 → −481,6 → −478,5, con Gini 0,757 / 0,698 / 0,675 en DEV/HO/OOT. En el Lab 2, algunas muestras personales dejaban entrar una variable con β positivo, que había que sacar a mano (`seleccion.remove(...)`) y declarar en la justificación 5c.

**Lo de gobierno.** La RESERVA de la clase 3 plantea el caso del comité que pide una variable excluida («¿y la renta?, ¿y la antigüedad laboral?»): diagnosticar si quedó fuera por redundancia o por aporte nulo; si aporta poco, se puede **forzar y medir el costo** («típicamente centésimas de Gini a cambio de paz institucional»). «Forzar una variable no es pecado; no documentarlo, sí. El modelo es tan defendible como su bitácora de decisiones.» La clusterización conceptual (1–3 variables por concepto: mora, utilización, bureau, demografía…) y el rango de 8–14 variables se justifican «por gobierno, no por estética».

**Lo que el curso simplificó, omitió o dejó como convención:**

- **Qué test.** Entrada por Wald, cuando la literatura y SAS usan el test *score* para entrar y Wald para salir; el LR es el más confiable en muestras chicas. No se discutió que Wald, LR y score pueden discrepar.
- **Por qué p < 0,05 y no AIC/BIC.** Sobre WoE los tres son el mismo test con distinto umbral (sección 3.3). En Austral, BIC habría parado en 7 variables.
- **El tamaño real del test.** El p-valor se calcula como si la variable tuviera 1 grado de libertad, pero su WoE se ajustó con los malos de DEV. Con 5 bins, una variable de **ruido puro** pasa el «test al 5%» con probabilidad ≈ 0,43 (sección 3.4). El filtro IV ≥ 0,10 es lo que, sin decirlo, protege al embudo.
- **Inferencia post-selección.** Los p-valores y los β del modelo final se reportan como si las variables se hubieran elegido de antemano. Están sesgados hacia afuera (*winner's curse*) y los intervalos cubren menos que lo nominal.
- **Inestabilidad.** No se midió cuánto cambiaría la selección con otra muestra. En el notebook, 40 réplicas bootstrap producen 39 modelos distintos.
- **Alternativas.** No se vieron LASSO/elastic net (y su versión con restricción de signo), stability selection, ni el bootstrap de la selección.
- **Política de signos.** Se aplicó a mano y después del stepwise, no como regla del algoritmo. Tampoco se discutió que puede sacar variables **verdaderas** (supresión en familias nivel/tendencia).
- **Forzar.** Se mencionó el costo, pero no cómo medirlo (test LR anidado, ΔGini con intervalo) ni que forzar al final y forzar al inicio producen modelos distintos.
- **El tope de 14** es convención de gobierno sin base estadística; cuando el tope es el que corta, la bitácora debe decirlo.

---

## 2. Intuición

**El stepwise es una búsqueda codiciosa, no un estimador.** Con $p$ candidatas hay $2^p$ modelos posibles (con las 26 de Austral, 67 millones). Forward elige en cada paso el mejor paso local; nada garantiza que el conjunto final sea el mejor de su tamaño, y dos variables que solo sirven juntas pueden no entrar nunca. Backward parte del modelo completo y saca; bidireccional (el del curso) alterna. Los tres son heurísticas sobre el mismo problema combinatorio.

**Cada decisión es un test, y se hacen decenas.** En la ronda $k$ se comparan todas las candidatas restantes y se elige la **máxima**. El p-valor de la ganadora no es el de un test aislado: es el mínimo de muchos. Con 26 candidatas en la primera ronda, aun si ninguna importara, la probabilidad de que la mejor tenga p < 0,05 sería $1-0{,}95^{26}\approx 0{,}74$ (bajo independencia). Esto es la **paradoja de Freedman** (1983): seleccione con los datos y luego haga inferencia con los mismos datos, y el ruido parece señal. En la réplica del notebook (regresión lineal, $n=100$, 50 regresores de ruido, tamizado a p < 0,25 y re-ajuste, 1.000 corridas) quedan en promedio 12,5 variables tras el tamizado, 5,6 de ellas «significativas» al 5%, un R² de 0,31 y un F global que rechaza en el 96% de las corridas. Nada de eso existe.

**El WoE ajustado en la misma muestra agrava todo.** El WoE de un bin es el log del cociente de proporciones de buenos y malos **observadas** en DEV. Para una variable de ruido, esas proporciones difieren por azar, y el WoE «aprende» ese azar. Luego la logística recibe una variable que ya viene orientada hacia el target: su β sale ≈ −1 con el signo «correcto» y un p-valor que se calcula con 1 grado de libertad cuando en realidad se usaron $B-1$ para construirla.

**Lo que se selecciona es una muestra de una nube.** Si se repite el procedimiento con otra muestra de la misma población, sale otro conjunto. Las variables fuertes aparecen siempre; las marginales entran y salen. Frecuencias de inclusión por bootstrap y stability selection miden esa nube en lugar de ignorarla.

**La regularización cambia la pregunta.** LASSO no pregunta «¿es significativa?» sino «¿cuánto baja la pérdida por unidad de complejidad?». Es convexo (no depende del orden), es continuo (una variable entra gradualmente) y admite restricciones de signo **dentro** de la optimización, en vez de como parche posterior.

**Selección no es lo mismo que discriminación.** En el notebook, entre enfoques razonables el Gini OOT varía en ±0,01. La selección rara vez compra Gini; lo que decide es **cuántas** variables quedan, **cuáles** (¿hay ruido?, ¿hay proxies?) y **cuántas explicaciones** habrá que dar al comité.

---

## 3. Formalización

### 3.1 El problema y la búsqueda codiciosa

Sea $\mathcal C$ el conjunto de $p$ candidatas (columnas WoE), $S\subseteq\mathcal C$ un modelo y $\hat\ell(S)$ la log-verosimilitud maximizada de la logística con intercepto y las columnas de $S$. Casi todos los criterios de selección son instancias de

$$
S^\star=\arg\min_{S\subseteq\mathcal C}\; -2\hat\ell(S) + c\,|S| ,
$$

con $c=2$ (AIC), $c=\ln n$ (BIC) o, implícitamente, $c=\chi^2_{1;1-\alpha}$ para un stepwise por p-valor. Resolverlo exactamente es combinatorio (*best subset*); hay algoritmos de ramificación y acotamiento (Furnival & Wilson, 1974; el `SELECTION=SCORE` de SAS `PROC LOGISTIC`) y formulaciones de optimización entera mixta (Bertsimas, King & Mazumder, 2016), viables para decenas de variables.

**Forward** construye $S_0=\varnothing$, $S_{k+1}=S_k\cup\{j^\star\}$ con $j^\star=\arg\max_{j\notin S_k}\hat\ell(S_k\cup\{j\})$ si pasa el umbral. **Revisión backward:** tras cada entrada, mientras exista $u\in S_{k+1}$ cuyo test de salida no pase, se retira el peor y se re-estima. Un detalle que el Lab 2 enfatiza y que tiene justificación: sacar de a uno, porque los p-valores de las demás cambian al re-estimar.

**Terminación.** Sin revisión backward, forward termina en a lo más $p$ pasos. Con revisión, puede ciclar (A entra, B sale, B entra, A sale…). Implementaciones serias detectan modelos repetidos (el notebook guarda los conjuntos visitados) o fijan $\alpha_{\text{salida}}\ge\alpha_{\text{entrada}}$, que impide que una variable recién entrada salga en la misma ronda por el mismo test.

### 3.2 Los tres tests para agregar una variable

Sea el modelo actual con matriz $X$ ($n\times q$, incluye unos), ajustado con $\hat p_0$, pesos $W=\operatorname{diag}(\hat p_0(1-\hat p_0))$, y una candidata $z$. El modelo ampliado es $\eta = X\beta + \gamma z$. Queremos testear $H_0:\gamma=0$.

**Wald.** Se ajusta el modelo ampliado y $\;W_z=\hat\gamma^2/\widehat{\operatorname{Var}}(\hat\gamma)$, con la varianza tomada de la inversa de la información $[\,[X\;z]^\top \hat W_1 [X\;z]\,]^{-1}$ evaluada en el ajuste **ampliado**.

**Razón de verosimilitudes.** $\;\text{LR}=2\,[\hat\ell(X,z)-\hat\ell(X)]=2\Delta\ell.$

**Score (Rao).** No requiere ajustar el ampliado. La derivada de la log-verosimilitud respecto de $\gamma$ en $\gamma=0$ es

$$
U = \frac{\partial\ell}{\partial\gamma}\Big|_{\gamma=0}= z^\top (y-\hat p_0).
$$

Su varianza bajo $H_0$ no es $z^\top W z$, porque $\beta$ también se estimó: hay que descontar la parte de $z$ explicada por $X$. Particionando la información de Fisher del modelo ampliado,

$$
\mathcal I=\begin{pmatrix} X^\top W X & X^\top W z\\ z^\top W X & z^\top W z\end{pmatrix},
$$

la varianza asintótica de $U$ ajustada por la estimación de $\beta$ es el **complemento de Schur** del bloque $X^\top W X$:

$$
I_{\gamma\cdot\beta}= z^\top W z - z^\top W X\,(X^\top W X)^{-1}X^\top W z ,
\qquad
S=\frac{U^2}{I_{\gamma\cdot\beta}}\;\overset{H_0}{\sim}\;\chi^2_1 .
$$

Interpretación: $I_{\gamma\cdot\beta}$ es la suma de cuadrados ponderada del residuo de regresar $z$ sobre $X$ con pesos $W$. Si $z$ es casi colineal con $X$, $I_{\gamma\cdot\beta}\to 0$ y cualquier $U$ pequeño da un $S$ grande: el test score es sensible a la colinealidad, igual que el VIF (Serie 2 · M09). Como $U$ e $I$ son lineales en $z$, se evalúan **todas las candidatas con dos productos de matrices**; por eso SAS usa score para entrar. El notebook implementa `score_todas` y lo contrasta con `statsmodels.GLM(...).score_test(exog_extra=...)` (diferencia ≈ 1e-13).

**Relación entre los tres.** Bajo $H_0$ y regularidad, $W_z$, LR y $S$ son asintóticamente equivalentes ($\chi^2_1$). En muestras finitas difieren; con alternativas fuertes el Wald puede incluso *disminuir* al crecer el efecto (Hauck–Donner, ver M10). En la trayectoria de Austral (paso 6, `deuda_otras_prom_12m`) el Wald reportado da p = 7,7·10⁻³ y el LR reconstruido desde las log-verosimilitudes da p ≈ 3,0·10⁻³: el mismo orden de magnitud, no el mismo número. Para decisiones al borde del umbral, el LR es el más defendible.

### 3.3 p-valor, AIC y BIC sobre WoE: un solo test, tres umbrales

Sobre WoE cada candidata aporta **un** parámetro. Entonces, para una candidata:

- **p-valor LR < α** $\iff 2\Delta\ell > \chi^2_{1;1-\alpha}$ $\iff \Delta\ell > 1{,}92$ (α = 0,05) o $\Delta\ell > 3{,}32$ (α = 0,01).
- **AIC baja** $\iff -2(\hat\ell+\Delta\ell)+2(k+1) < -2\hat\ell+2k \iff \Delta\ell>1$, es decir, α equivalente $=\Pr(\chi^2_1>2)=0{,}157$.
- **BIC baja** $\iff \Delta\ell > \tfrac12\ln n$, es decir, $\alpha(n)=\Pr(\chi^2_1>\ln n)$: 0,0086 con $n=1.000$; **0,0044 con $n=3.322$** (Austral); 0,0024 con $n=10.000$.

Tres consecuencias:

1. **Elegir criterio es elegir α.** AIC es un stepwise permisivo (α ≈ 0,16); BIC es un stepwise estricto cuyo α cae con $n$. «Usamos AIC» no es más riguroso que «usamos p < 0,05»: es otra perilla.
2. **BIC es consistente, AIC no.** Si el modelo verdadero está entre los candidatos, BIC lo selecciona con probabilidad → 1 al crecer $n$; AIC sobreajusta con probabilidad positiva, pero minimiza el error de predicción asintótico (Burnham & Anderson, 2002). En scorecards el modelo verdadero no está en el pool (el log-odds real no es lineal en WoE), así que la «consistencia» es un argumento débil; lo que importa es el costo de variables espurias frente al costo de omitir señales.
3. **El curso usa log-verosimilitud + p** porque ambos se leen directo de `statsmodels` y porque el p es el lenguaje del comité. Es legítimo; lo que no es legítimo es leer ese p como un error tipo I del 5% (sección 3.4) ni como una probabilidad de que la variable importe.

**Austral re-leído.** Con 165 malos en 3.322, el nulo es $\hat\ell_0=165\ln(165/3322)+3157\ln(3157/3322)=-656{,}2$. Las ganancias por paso son $\Delta\ell = 115{,}5;\;27{,}9;\;11{,}1;\;6{,}3;\;4{,}3;\;4{,}4;\;5{,}1;\;3{,}1$. Todas superan 1,92 (p < 0,05) y 1 (AIC). Con BIC (umbral $\tfrac12\ln 3322=4{,}05$) el paso 8 (`uso_tc_prom_3m`, Δℓ = 3,1) no entra: **BIC habría dejado 7 variables**; lo mismo α = 0,01. Los pasos 5 y 6 pasan BIC por 0,25 y 0,35 unidades de log-verosimilitud, con log-verosimilitudes publicadas redondeadas a 0,1: están en el borde.

### 3.4 Los grados de libertad escondidos del WoE

**Resultado.** Sea $x$ binneada en $B$ bins con cortes que **no** dependen de $y$ (cuantiles, como `binear`). Sea $m_b,g_b$ el número de malos y buenos en el bin $b$, $M=\sum m_b$, $G=\sum g_b$, y $\text{WoE}_b=\ln\!\big(\tfrac{g_b/G}{m_b/M}\big)$ calculado en la misma muestra (sin suavizado). Entonces la logística univariada sobre el WoE alcanza el máximo de verosimilitud del modelo saturado por bins, y

$$
\text{LR}=2[\hat\ell(\text{WoE})-\hat\ell_0]=G^2_{B\times2}\;\overset{H_0}{\longrightarrow}\;\chi^2_{B-1}.
$$

**Demostración.** El log-odds empírico del bin $b$ es

$$
\ln\frac{m_b}{g_b}=\ln\frac{M}{G}+\ln\frac{m_b/M}{g_b/G}=\ln\frac{M}{G}-\text{WoE}_b .
$$

Luego $\beta_0=\ln(M/G)$, $\beta_1=-1$ reproduce exactamente la tasa observada de cada bin, que es el MLE del modelo con un parámetro libre por bin (el saturado para una variable categórica de $B$ niveles). Como el modelo WoE está contenido en el saturado y lo alcanza, su máximo coincide. El LR contra el nulo es entonces el estadístico $G^2$ de independencia de la tabla $B\times 2$, que bajo $H_0$ (y cortes independientes de $y$) es asintóticamente $\chi^2_{B-1}$. Con el suavizado +0,5 del curso, $\hat\beta_1\approx-1$ y el LR se aproxima al $G^2$. $\square$

**Consecuencia.** El stepwise compara ese LR (o el Wald equivalente) con $\chi^2_{1;0,95}=3{,}84$. La probabilidad de que una variable de ruido pase la primera ronda es

$$
\Pr(\chi^2_{B-1}>3{,}84)=0{,}147\;(B=3),\quad 0{,}279\;(B=4),\quad \mathbf{0{,}428}\;(B=5),\quad 0{,}572\;(B=6),\quad 0{,}922\;(B=10).
$$

Con **binning supervisado** (óptimo, chi-merge, árboles, `optbinning`), los cortes sí dependen de $y$ y la distribución nula es peor que $\chi^2_{B-1}$. En etapas posteriores del stepwise el LR parcial de una variable de ruido (independiente de las demás) sigue aproximadamente la misma ley, porque su WoE sigue «alineado» con el residuo.

**El IV del ruido.** Con $\pi_b$ la proporción poblacional del bin, $\hat g_b$ y $\hat m_b$ las proporciones de buenos y malos en el bin. Bajo $H_0$, $\hat g_b-\hat m_b\approx 0$ y $\ln(\hat g_b/\hat m_b)\approx(\hat g_b-\hat m_b)/\pi_b$, de donde

$$
\text{IV}=\sum_b(\hat g_b-\hat m_b)\ln\frac{\hat g_b}{\hat m_b}\approx\sum_b\frac{(\hat g_b-\hat m_b)^2}{\pi_b}.
$$

Los vectores $\hat g$ y $\hat m$ son proporciones multinomiales independientes con la misma media $\pi$, así que $\hat g-\hat m$ tiene covarianza $(\tfrac1G+\tfrac1M)(\operatorname{diag}\pi-\pi\pi^\top)$ y la forma cuadrática da

$$
\frac{\text{IV}}{\tfrac1M+\tfrac1G}\;\overset{H_0}{\approx}\;\chi^2_{B-1},\qquad \mathbb E[\text{IV}_{\text{ruido}}]\approx(B-1)\Big(\tfrac1M+\tfrac1G\Big).
$$

Es el mismo argumento que para el PSI (Serie 2 · M08): IV y PSI son divergencias de Jeffreys. En Austral ($M=165$, $G=3.157$): $\mathbb E[\text{IV}]\approx 0{,}026$ y $\Pr(\text{IV}\ge 0{,}10)=\Pr(\chi^2_4>15{,}7)\approx 0{,}35\%$. Con 108 candidatas, eso es del orden de 0,4 variables de ruido que sobreviven al filtro de poder. **El filtro IV ≥ 0,10 del curso es un test de hipótesis al ≈ 0,3%, y es el que realmente protege contra el ruido**; el p < 0,05 del stepwise, aplicado a WoE in-sample, es un test al ≈ 43%.

**Remedio: WoE cruzado para seleccionar.** Partir DEV en $K$ pliegues y calcular el WoE de cada fila con cortes y tasas estimadas en los otros $K-1$ (como el *target encoding* con validación cruzada). El WoE de una variable de ruido deja de estar alineado con el $y$ de la fila y el test vuelve a su tamaño nominal. Se usa **solo para decidir qué entra**; el modelo final se re-ajusta con el WoE de DEV completo (lo que se congela). Alternativas equivalentes: usar $B-1$ grados de libertad en el test (LR contra $\chi^2_{B-1}$), o un test de permutación que re-binee en cada permutación.

### 3.5 Inferencia post-selección: *winner's curse* y cobertura

Supongamos $\hat\gamma\sim N(\gamma,\sigma^2)$ y que la variable se retiene si $\hat\gamma/\sigma>c$ (caso unilateral, $c=1{,}96$). Con $d=\gamma/\sigma$ y $a=c-d$, la media de una normal truncada da

$$
\mathbb E[\hat\gamma\mid\text{retenida}]=\gamma+\sigma\,\frac{\varphi(a)}{1-\Phi(a)}
\quad\Longrightarrow\quad
\frac{\mathbb E[\hat\gamma\mid\text{retenida}]}{\gamma}=1+\frac{1}{d}\,\frac{\varphi(c-d)}{1-\Phi(c-d)} .
$$

El sesgo depende de la **potencia**, no de la variable: con $d=4{,}2$ la inflación es 1,01; con $d=2{,}5$, 1,19; con $d=1{,}26$, **2,03**. El notebook lo verifica por simulación (tres señales con β = 0,30; 0,18; 0,09, $n=2.000$, 20 ruidos): inflación observada 1,02 / 1,28 / 2,11 y cobertura del IC ingenuo al 95%, condicional a la selección, 97% / 98% / **88%**. La selección condiciona la muestra de β̂ que uno ve; los intervalos que ignoran esa condición no cubren.

Remedios, de más simple a más sofisticado: (i) **división de muestra** (seleccionar en una mitad, estimar e inferir en la otra; en la simulación la inflación de la señal débil baja a 1,05 y la cobertura vuelve a 96%, a costa de potencia); (ii) **bootstrap de todo el procedimiento** (selección incluida) para intervalos; (iii) inferencia selectiva exacta (Lee, Sun, Sun & Taylor, 2016, para LASSO) o simultánea (PoSI, Berk et al., 2013). En scorecards, (i) es lo práctico: HO ya existe; el problema es que el curso usa HO para *contrastar*, no para re-estimar (78 malos en HO: «contrasta, no re-estima»).

### 3.6 Bootstrap de la selección y stability selection

**Frecuencia de inclusión** (Sauerbrei & Schumacher, 1992; Austin & Tu, 2004). Para $b=1,\dots,B$, remuestrear DEV con reemplazo, correr el procedimiento **completo** y registrar $S^{(b)}$. La frecuencia de inclusión es $\hat\pi_j=\tfrac1B\sum_b \mathbb 1[j\in S^{(b)}]$. Austin & Tu proponen retener las variables con $\hat\pi_j$ sobre un umbral (60% es un valor de trabajo frecuente, sin base teórica). También sirve contar **modelos distintos**: si el modal aparece en el 5% de las réplicas, hablar de «el modelo seleccionado» es una ficción.

Una sutileza que casi nunca se menciona: si el WoE está fijo (ajustado en DEV completo), el bootstrap re-muestrea filas cuyo WoE ya incorporó su propio $y$. El bootstrap **hereda la fuga**: una variable de ruido con WoE in-sample puede tener $\hat\pi_j$ alto. La versión honesta re-binea y re-calcula WoE dentro de cada réplica, o usa WoE cruzado.

**Stability selection** (Meinshausen & Bühlmann, 2010). Para subsamples $I_1,\dots,I_{2R}$ de tamaño $\lfloor n/2\rfloor$ (en pares complementarios, Shah & Samworth, 2013), se aplica un selector con tamaño promedio acotado (típicamente las primeras $q$ variables del camino LASSO) y $\hat\pi_j=\tfrac{1}{2R}\sum_r\mathbb 1[j\in\hat S_q(I_r)]$. Se retiene $\hat S_{\text{estable}}=\{j:\hat\pi_j\ge\pi_{\text{thr}}\}$. Bajo intercambiabilidad de las variables de ruido y un selector no peor que el azar, el número esperado de falsos positivos $V$ cumple

$$
\mathbb E[V]\le\frac{1}{2\pi_{\text{thr}}-1}\cdot\frac{q^2}{p},\qquad \pi_{\text{thr}}\in(1/2,1].
$$

Con $p=21$, $q=8$, $\pi_{\text{thr}}=0{,}75$ la cota es 6,1: casi inútil con pools chicos. La cota es informativa cuando $p\gg q^2$ (cientos de candidatas, como antes del embudo). Stability selection selecciona lo **estable**, que no es lo mismo que lo **verdadero**: señales débiles o redundantes con otra (en el notebook, `uso_tc_prom_12m` y `canal`) quedan fuera.

### 3.7 LASSO y elastic net con restricción de signo

Con WoE estandarizado $\tilde x_j=(x_j-\bar x_j)/s_j$ y $\ell_n(\beta_0,\beta)=\tfrac1n\sum_i[\log(1+e^{\eta_i})-y_i\eta_i]$:

$$
\min_{\beta_0,\beta}\;\ell_n(\beta_0,\beta)+\lambda\Big(\alpha\lVert\beta\rVert_1+\tfrac{1-\alpha}{2}\lVert\beta\rVert_2^2\Big)\quad\text{s.a.}\quad\beta_j\le0\;\;\forall j .
$$

**La restricción vuelve suave el problema.** En la región factible $|\beta_j|=-\beta_j$, así que el término L1 es **lineal**: $\lambda\alpha\sum_j|\beta_j|=-\lambda\alpha\sum_j\beta_j$. El problema es diferenciable con cotas simples y lo resuelve cualquier método con cotas (L-BFGS-B). Es convexo: no hay dependencia del orden de entrada.

**Condiciones de optimalidad (KKT).** Sea $g_j=\partial\ell_n/\partial\beta_j=\tfrac1n\tilde x_j^\top(\hat p-y)$. Para $\beta_j<0$: $g_j-\lambda\alpha+\lambda(1-\alpha)\beta_j=0$. Para $\beta_j=0$ (en la cota) la derivada direccional hacia valores negativos debe ser no negativa: $g_j\le\lambda\alpha$. En $\beta=0$ con intercepto $\operatorname{logit}\bar y$, $g_j=\tfrac1n\tilde x_j^\top(\bar y-y)$, y la primera variable entra cuando

$$
\lambda<\lambda_{\max}^{(-)}=\frac{1}{\alpha}\max_j\;\frac{\tilde x_j^\top(\bar y-y)}{n}.
$$

Una variable con asociación marginal «al revés» ($\tilde x_j^\top(y-\bar y)>0$, WoE alto asociado a más malos) **nunca** entra al inicio del camino. Con WoE, todas tienen asociación marginal negativa por construcción; la restricción actúa solo cuando la **supresión** multivariada invertiría el signo, que es exactamente la situación que la política de signos del curso quiere evitar. La diferencia es que aquí se resuelve **dentro** del estimador: la variable queda en $\beta_j=0$ en vez de ser sacada a mano y re-estimar.

**Algoritmo en numpy (FISTA).** Gradiente proximal acelerado (Beck & Teboulle, 2009) con paso $t=1/L$, $L=\lambda_{\max}(\tilde X^\top\tilde X/n)/4$ (más la curvatura L2). El operador proximal de $t\lambda[\alpha|b|+\tfrac{1-\alpha}{2}b^2]+\iota_{b\le0}$ en $z$ es

$$
\operatorname{prox}(z)=\frac{\min\{0,\;z+t\lambda\alpha\}}{1+t\lambda(1-\alpha)},
$$

que se deriva minimizando $\tfrac12(b-z)^2-t\lambda\alpha b+\tfrac{t\lambda(1-\alpha)}{2}b^2$ sobre $b\le0$: el mínimo sin restricción es $(z+t\lambda\alpha)/(1+t\lambda(1-\alpha))$ y se proyecta a $b\le 0$. Sin restricción, el conocido *soft-thresholding* $\operatorname{sign}(z)\max(|z|-t\lambda\alpha,0)/(1+t\lambda(1-\alpha))$.

**Elección de λ.** Opciones: validación cruzada (cuidado, sección 5, trampa 8), BIC del modelo re-ajustado sin penalizar sobre el conjunto activo (*relaxed lasso*, Meinshausen 2007; es lo que usa el notebook) o el λ que deja el número de variables que gobierno pide. Los grados de libertad del LASSO son, en esperanza, el número de coeficientes no nulos (Zou, Hastie & Tibshirani, 2007), lo que justifica el BIC con $|S|$.

**Por qué re-ajustar.** El LASSO encoge los β hacia 0; un scorecard con β encogidos distribuye puntos de forma distinta y la PD queda comprimida. En la práctica: LASSO para **seleccionar**, logística sin penalizar sobre el conjunto elegido para **estimar**, y calibración aparte (M15).

### 3.8 Forzar una variable: el costo, medido

Sea $S$ la selección y $v\notin S$ la variable que pide el comité. El costo estadístico de agregarla es el test anidado

$$
\text{LR}_v=2[\hat\ell(S\cup\{v\})-\hat\ell(S)]\sim\chi^2_1\quad(H_0:\gamma_v=0),
$$

y el costo predictivo es $\Delta\text{Gini}=\text{Gini}(S\cup\{v\})-\text{Gini}(S)$ en HO y OOT, con intervalo por bootstrap **pareado** (se remuestrean filas y se recalculan ambos Gini con las mismas filas: la varianza de la diferencia es mucho menor que la de cada Gini). Si el intervalo contiene 0, el costo es indistinguible de cero y la decisión es de gobierno.

Hay dos decisiones distintas que se confunden: **forzar al final** ($S\cup\{v\}$) y **forzar al inicio** (stepwise con $v$ incluida desde la ronda 0 y sin posibilidad de salir). La segunda puede desplazar variables que comparten información con $v$ y producir otro modelo. La bitácora debe decir cuál se tomó.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / marco |
|---|---|---|---|---|
| Stepwise por p-valor (entrada score, salida Wald) | Selección rápida, lenguaje de comité | Bajo; inferencia post-selección inválida; inestable | Pool ya filtrado por estabilidad, IV y redundancia (≤ 30 candidatas) | SAS `PROC LOGISTIC` (`SLENTRY`/`SLSTAY`, 0,05 por defecto); práctica bancaria de scorecards (Siddiqi) |
| Stepwise por LR / Wald con tope (el del curso) | Igual, con aporte marginal explícito | Un ajuste por candidata y ronda | Enseñanza, auditoría (cada paso es reproducible a mano) | Curso; `statsmodels` a mano |
| Stepwise por AIC / BIC | Un criterio de información en vez de un umbral arbitrario | Igual que p (es un α implícito: 0,157 o $\Pr(\chi^2_1>\ln n)$) | BIC cuando se quiere parsimonia y $n$ es grande | R `step()` / `MASS::stepAIC` (AIC por defecto, `k = log(n)` para BIC) |
| *Purposeful selection* | Mezcla univariado laxo (p < 0,25), multivariado, confusores y juicio clínico | Manual, lento | Modelos explicativos con variables de ajuste | Hosmer, Lemeshow & Sturdivant (2013); Bursac et al. (2008) |
| Best subset (ramificación y acotamiento, MIO) | Óptimo global para cada tamaño | Exponencial en el peor caso; viable hasta decenas | Cuando el pool final es chico y se quiere descartar que el greedy se perdió algo | SAS `SELECTION=SCORE` (Furnival–Wilson); Bertsimas et al. (2016) |
| Forward por Gini/KS en validación | Optimiza la métrica de negocio | Contamina la muestra de validación | Solo con una tercera muestra intacta | Práctica frecuente en herramientas comerciales (verificar por proveedor) |
| LASSO / elastic net | Selección + encogimiento en un problema convexo; estable ante colinealidad (EN) | Elegir λ; β encogidos (re-ajustar) | Pools grandes, colinealidad moderada | `glmnet` (R/Python), `scikit-learn` (`saga`) |
| LASSO con restricción de signo | Política de signos dentro del estimador | Requiere solver con cotas | Scorecards WoE donde el signo es contrato | `glmnet` (`upper.limits = 0`); `scipy.optimize` L-BFGS-B; numpy FISTA |
| Adaptive / relaxed lasso | Menos sesgo en coeficientes grandes; selección consistente | Dos etapas | Cuando se quiere LASSO para seleccionar y β poco sesgados | Zou (2006); Meinshausen (2007) |
| Bootstrap de la selección | Mide inestabilidad; frecuencia de inclusión | B × costo del stepwise | Siempre como diagnóstico en el expediente | Austin & Tu (2004); Sauerbrei & Schumacher (1992); Heinze et al. (2018) |
| Stability selection | Control aproximado de falsos positivos | $2R$ caminos LASSO | Pools grandes ($p\gg q^2$), p. ej. antes del embudo | Meinshausen & Bühlmann (2010); Shah & Samworth (2013) |
| Knockoffs | Control de FDR con garantía finita | Construir knockoffs (difícil con WoE discretos) | Investigación; pools muy grandes | Barber & Candès (2015) |
| Selección por concepto / juicio experto | Cobertura de drivers, explicabilidad, robustez ante caída de una fuente | Subjetivo; debe documentarse | Siempre, como capa final | EBA/GL/2017/16 (consulta a expertos de negocio y plausibilidad económica de los *risk drivers*, párrs. 35, 57–58, verificar numeración); SR 11-7, hoy reemplazada por SR 26-2 (documentar decisiones de desarrollo; ver M22) |
| WoE cruzado para seleccionar | Elimina los grados de libertad escondidos del binning | K binnings | Siempre que la selección use tests o CV sobre WoE | Práctica análoga al *target encoding* con CV (no estándar en scorecards) |

Una nota regulatoria que aparece en la RESERVA («¿puede edad ser un reason code?»): en EE.UU., la Regulation B (12 CFR §1002.6(b)(2)(ii)) permite usar la edad en un sistema de scoring «empíricamente derivado, demostrable y estadísticamente sólido», siempre que a un solicitante de edad avanzada no se le asigne un factor o valor negativo. Es decir, una variable puede ser estadísticamente elegible y legalmente restringida en su **signo por tramo**. Para Chile, revisar la norma vigente de la CMF y la Ley 19.628 sobre datos personales (ver Serie 1 · E6); no afirmo aquí una restricción específica.

---

## 5. Cuándo falla: trampas y modos de falla

**Trampa 1 · Ruido que entra por el WoE in-sample.**
*Síntoma:* variables de IV bajo (0,02–0,04) entran con p ≈ 10⁻³ y β ≈ −1. *Causa:* WoE ajustado con los malos de la misma muestra; el test usa 1 gl cuando la variable consumió $B-1$ (sección 3.4). *Detección:* inyectar variables canario de ruido y ver si entran; comparar el stepwise con WoE cruzado. En el notebook, el stepwise del curso elige 11 variables, **3 de ruido puro**; en la simulación con 50 variables de ruido y $n=2.000$, el stepwise sobre WoE in-sample deja en promedio **19,3** variables espurias (vs 2,1 con la variable cruda y 1,5 con WoE cruzado). *Qué hacer:* filtro IV previo (ya lo hace el curso), WoE cruzado para seleccionar, o LR contra $\chi^2_{B-1}$.

**Trampa 2 · Categóricas de alta cardinalidad.**
*Síntoma:* `concesionario`, `marca`, `comuna` con decenas de niveles entran siempre. *Causa:* cada nivel es un bin; $B-1$ grande. Con 30 niveles, $\Pr(\chi^2_{29}>3{,}84)\approx 1$ y $\mathbb E[\text{IV}_{\text{ruido}}]\approx 29(\tfrac1M+\tfrac1G)$, que supera 0,10 con 150 malos (sección 8). *Detección:* IV y p que crecen con el número de niveles; WoE de niveles con pocos casos. *Qué hacer:* agrupar niveles con reglas definidas sin mirar $y$ (volumen mínimo) o con binning supervisado **cruzado**; tratar la variable como una familia con test de $B-1$ gl.

**Trampa 3 · Leer p-valores y β post-selección como si fueran pre-registrados.**
*Síntoma:* el informe dice «todas significativas al 1%». *Causa:* las variables se eligieron *porque* tenían p chico. *Detección:* re-estimar en HO (el curso lo hace: en Austral `deuda_interna_max_3m` pasa de −0,926 a +0,023 con p = 0,96) o bootstrap del procedimiento. *Qué hacer:* reportar los p de DEV como descriptivos; la evidencia de que una variable aporta es su estabilidad (bootstrap, HO, OOT), no su p.

**Trampa 4 · Inestabilidad no medida.**
*Síntoma:* re-entrenar con 3 meses más cambia 3 de 10 variables. *Causa:* variables marginales, correlacionadas entre sí, con aportes parecidos. *Detección:* bootstrap de la selección: en el notebook, **39 modelos distintos en 40 réplicas**, con el modal en el 5%. *Qué hacer:* reportar frecuencias de inclusión; decidir las marginales por concepto y costo de datos, no por el azar de la muestra.

**Trampa 5 · La política de signos saca variables verdaderas.**
*Síntoma:* una variable con fuerte sentido de negocio sale por β > 0. *Causa:* supresión: con dos ventanas de la misma serie (nivel y tendencia), el coeficiente de una, condicional a la otra, puede invertirse **en la verdad**. En el generador, el riesgo depende de `uso_tc_prom_12m` y de `uso_tc_prom_3m − uso_tc_prom_12m`; el modelo **oráculo** (solo variables verdaderas) tiene β > 0 en `uso_tc_prom_12m`. *Detección:* signos que se invierten al agregar una variable de la misma familia. *Qué hacer:* re-expresar la familia como nivel + delta (M09) antes de seleccionar; la política de signos se aplica a variables re-expresadas, no a ventanas redundantes.

**Trampa 6 · Seleccionar mirando HO.**
*Síntoma:* Gini HO del modelo final más alto que el de alternativas, y OOT igual. *Causa:* HO usado para elegir variables (forward por Gini HO, o «probé 5 configuraciones y me quedé con la de mejor HO»). *Detección:* en el notebook, el forward por Gini HO tiene el mejor Gini HO (0,551) y en OOT (0,529) queda en el montón. *Qué hacer:* toda decisión de selección en DEV (o en validación cruzada dentro de DEV); HO y OOT solo se miran al final, una vez.

**Trampa 7 · El tope decide y nadie lo sabe.**
*Síntoma:* el stepwise se detuvo en 14 y la bitácora no lo dice. *Causa:* el tope es un corte de gobierno que actúa como criterio. *Detección:* motivo de término en la bitácora. En el notebook, el stepwise por AIC sobre WoE in-sample llega al tope de 14 con 5 variables de ruido. *Qué hacer:* registrar el motivo de término; si el tope corta, revisar el umbral o el pool, no celebrar las 14.

**Trampa 8 · Validación cruzada sobre WoE precalculado.**
*Síntoma:* CV elige un λ pequeño (muchas variables) y el HO decepciona. *Causa:* el WoE se calculó con todo DEV; cada pliegue de validación ya «vio» sus propios malos a través del WoE. *Detección:* comparar CV con WoE fijo vs WoE re-calculado dentro de cada pliegue. *Qué hacer:* re-binear y re-calcular WoE **dentro** del pliegue de entrenamiento (pipeline completo en la CV), o usar BIC.

**Trampa 9 · Filas que cambian entre modelos.**
*Síntoma:* LR tests negativos o inconsistentes en un stepwise sobre variables crudas. *Causa:* con missing, cada modelo candidato usa las filas completas de *sus* variables; los modelos anidados no se ajustan sobre las mismas filas y el LR no es válido. Con WoE no pasa (el missing es un bin), pero sí en stepwise sobre variables crudas o dummies. *Qué hacer:* fijar la muestra de análisis antes de seleccionar.

**Trampa 10 · Forzar sin rastro, o forzar «al inicio» cuando se dijo «al final».**
*Síntoma:* el modelo en producción tiene una variable que el stepwise documentado no eligió, o le faltan dos que sí eligió. *Causa:* forzado desde la ronda 0 desplazó variables. *Qué hacer:* bitácora con acción, momento, autor, justificación y costo (LR, ΔGini con IC).

**Trampa 11 · Carteras chicas.**
*Síntoma:* con 80–150 malos, el stepwise elige 3 variables en una corrida y 9 en otra. *Causa:* baja potencia + muchas candidatas; EPV bajo (Serie 2 · M10). *Qué hacer:* BIC o α = 0,01, pocas candidatas pre-seleccionadas por concepto, bootstrap obligatorio, y preferir un modelo más chico y estable.

---

## 6. Puente con ingeniería

Piense la selección como **una etapa ajustada del pipeline**, con entrada, salida, configuración y artefacto, igual que el binning.

**Contrato de la etapa `seleccion`:**

```yaml
etapa: seleccion
entrada:
  matriz: woe_dev            # solo DEV; el contrato prohíbe columnas de HO/OOT/TTD
  target: malo
  candidatas: salida_de(etapa: redundancia)
config:
  woe_para_seleccionar: cruzado   # {in_sample, cruzado}; K=5, semilla=7
  criterio: wald                  # {wald, lr, score, aic, bic}
  alfa_entrada: 0.05
  alfa_salida: 0.05
  tope: 14
  politica_signo: todas           # {todas, entrante, ninguna}
  forzadas: []                    # cada una con ticket de comité
  semilla_bootstrap: 2004
  B_bootstrap: 200
salida:
  seleccion: [..]                 # lista ordenada por entrada
  bitacora: bitacora_seleccion.json
  frecuencias_bootstrap: frec_inclusion.csv
artefacto_congelado: [seleccion, bitacora, hash_bitacora]
```

**Invariantes verificables en CI:**

1. **Paridad de implementaciones.** El stepwise con el ajustador numpy y con `statsmodels` produce la misma selección y las mismas log-verosimilitudes (tolerancia 1e-6). El notebook lo verifica.
2. **Determinismo.** Misma entrada + misma config + misma semilla ⇒ mismo hash de bitácora.
3. **Aislamiento de muestras.** La etapa no recibe HO/OOT (test de contrato sobre columnas y sobre la huella de filas).
4. **Canarios de ruido.** En CI se inyectan $k$ columnas de ruido (al estilo de las *shadow features* de Boruta, Kursa & Rudnicki 2010) y se exige que ninguna quede seleccionada (o menos de una fracción en bootstrap). Es la prueba más barata contra la trampa 1 y detecta regresiones cuando alguien cambia el binning.
5. **Signos.** Todos los β del modelo final ≤ 0 salvo variables forzadas con excepción registrada.
6. **Motivo de término explícito** en la bitácora (umbral, tope, ciclo).
7. **Monotonía de criterios.** Con los mismos datos, la selección por BIC ⊆ la de p < 0,05 ⊆ la de AIC no es una ley (el greedy puede tomar caminos distintos), pero el **número** de variables debe respetar el orden en la gran mayoría de corridas; una violación es una alerta, no un error.

**Qué se congela y qué se versiona.** Se congela la **lista seleccionada** y su orden, junto con los cortes y WoE (etapa anterior) y los β (etapa siguiente). Se versiona la bitácora completa: configuración, huella de datos (hash del target y de la matriz WoE de DEV), cada decisión automática (entra / sale / rechazo de signo / fin) y cada decisión humana (forzar, excluir por costo de dato, excluir por regulación), con autor y justificación. El SHA-256 de la bitácora entra a la cadena de hashes del expediente (clase 6). Cambiar una decisión cambia el hash, y eso es lo deseado: la trazabilidad no se discute, se verifica.

**Re-entrenamiento.** En un re-desarrollo, correr el procedimiento completo y el bootstrap, y comparar con la bitácora anterior: qué variables cambiaron y si estaban en la zona inestable (frecuencia 30–70%). Un cambio en una variable con frecuencia 100% es noticia; en una con 50% es ruido esperado.

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy en el notebook | Librería | Diferencias / convenciones | En producción |
|---|---|---|---|---|
| Logística (β, SE, ℓ) | `irls`: Newton-Raphson, $H=X^\top WX$, SE de $H^{-1}$ | `statsmodels.Logit(...).fit(method="newton")`; `sklearn.LogisticRegression(C=np.inf)` | numpy = statsmodels a 1e-15; sklearn (lbfgs) a ≈ 1e-5 por tolerancia de parada. En scikit-learn ≥ 1.8, `penalty` está deprecado: sin penalización es `C=np.inf` | `statsmodels` para inferencia; numpy si se necesita velocidad en bucles |
| Test score | `score_todas`: complemento de Schur, todas las candidatas a la vez | `statsmodels.GLM(...).fit().score_test(exog_extra=z)` | Idénticos (≈ 1e-13); GLM evalúa una candidata por llamada | numpy vectorizado dentro del stepwise |
| Wald | $(\hat\beta/\text{SE})^2$ | `results.pvalues`, `results.wald_test(R, scalar=True)` | `pvalues` es Wald bilateral con normal | cualquiera |
| LR anidado | $2(\hat\ell_1-\hat\ell_0)$ + `scipy.stats.chi2.sf` | `Logit` no trae `compare_lr_test` (sí `OLS`); se arma con `llf` | — | numpy/scipy |
| Stepwise | `stepwise(...)` genérico con bitácora | No hay en `statsmodels`; `sklearn.feature_selection.SequentialFeatureSelector` (por CV, sin p-valores); SAS `PROC LOGISTIC`; R `step()` | SAS: entrada por score, salida por Wald, `SLENTRY=SLSTAY=0,05` por defecto; R `step()`: AIC | propio, con bitácora (ninguna librería la entrega) |
| LASSO / EN | FISTA con prox cerrado | `LogisticRegression(l1_ratio=α, C=1/(nλ), solver="saga")` | sklearn minimiza $C\sum\text{pérdida}+r(w)$ ⇒ $\lambda=1/(nC)$. `liblinear` penaliza el intercepto; `saga` no. `glmnet` estandariza por defecto; sklearn no | `saga` o `glmnet` |
| LASSO con signo | FISTA con $\operatorname{prox}=\min(0,z+t\lambda\alpha)/(1+t\lambda(1-\alpha))$ | `scipy.optimize.minimize(..., method="L-BFGS-B", bounds=(None,0))`; R `glmnet(upper.limits=0)` | `LogisticRegression` de sklearn **no** admite restricción de signo | scipy o glmnet |
| AUC/Gini | Rangos con empates promediados | `sklearn.metrics.roc_auc_score` | Idénticos | sklearn |
| OLS (Freedman) | `lstsq` + t de Student | `statsmodels.OLS` | Idénticos (≈ 1e-15) | statsmodels |

Dos convenciones que muerden. (1) **Estandarización**: LASSO penaliza β en la escala de las columnas. Sobre WoE (ya en unidades de log-odds) hay dos elecciones defendibles: penalizar β en WoE (cada unidad de WoE «cuesta» lo mismo) o estandarizar (penalizar el efecto por desviación estándar, que es lo que hace `glmnet`). El notebook estandariza y reporta β en unidades de WoE. (2) **Escala de la pérdida**: sklearn suma, glmnet y el notebook promedian; el mismo λ no significa lo mismo en ambas.

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral (clase 3)

- **Trayectoria re-leída** (sección 3.3): las 8 entradas pasan p < 0,05 y AIC; **BIC y α = 0,01 habrían parado en 7**, sin `uso_tc_prom_3m` (Δℓ = 3,1). Es la variable de tendencia reciente de la familia de utilización de tarjeta; que sea la más débil es coherente con su rol de «delta» sobre `uso_tc_prom_12m`. Una decisión defendible es mantenerla por concepto (deterioro reciente) y documentar que un criterio más estricto la habría excluido.
- **EPV del proceso.** 165 malos para 26 candidatas evaluadas: 6,3 eventos por candidata en la primera ronda, bajo la regla de 10 (M10). Para los 9 parámetros del modelo final (8 β + intercepto): 18,3. La selección opera con menos información de la que sugiere el modelo final.
- **El filtro que protege.** $\Pr(\text{IV}_{\text{ruido}}\ge 0{,}10)\approx 0{,}35\%$ con 165 malos; el stepwise, con WoE in-sample, habría dejado pasar ruido con probabilidad ≈ 43% por variable. El embudo funciona **por el orden** de sus filtros, no por el stepwise.
- **Estabilidad en HO** (clase 3): `deuda_interna_max_3m` y `deuda_otras_prom_12m` pierden todo (p 0,96 y 0,78 en HO). Coherente con la sección 3.5: variables que entraron en los pasos 4 y 6 con Δℓ de 6,3 y 4,4 son las candidatas naturales a β inflados en DEV.

### 8.2 Banco Sintético (notebook, con verdad conocida)

DEV 3.409 solicitudes (398 malos, 11,7%), HO 1.408, OOT 1.595 (15,1% por el deterioro plantado). Pool: 9 señales, 2 proxies, 10 ruidos.

| Enfoque (selección en DEV) | WoE para seleccionar | nº var. | ruido | Gini HO | Gini OOT |
|---|---|---|---|---|---|
| Stepwise Wald 0,05 (curso) | in-sample | 11 | 3 | 0,494 | 0,523 |
| Stepwise BIC | in-sample | 9 | 2 | 0,505 | 0,526 |
| Stepwise AIC | in-sample | 14 (tope) | 5 | 0,485 | 0,523 |
| Stepwise Wald 0,05 | cruzado | 10 | 2 | 0,502 | 0,527 |
| Stepwise BIC | cruzado | 5 | 0 | 0,529 | 0,520 |
| LASSO con signo, λ por BIC | cruzado | 5 | 0 | 0,529 | 0,520 |
| Stability selection (q = 8, π = 0,75) | ambos | 5 | 0 | 0,529 | 0,520 |
| Forward por Gini HO | in-sample | 6 | 1 | **0,551** (optimista) | 0,529 |
| Modelo completo | — | 21 | 10 | 0,476 | 0,526 |
| Oráculo (solo señal) | — | 9 | 0 | 0,508 | 0,531 |

Lecturas: (i) el Gini OOT de los enfoques razonables varía en 0,010; (ii) el WoE cruzado reduce el ruido pero no lo elimina bajo p < 0,05 (con 10 ruidos y ~10 rondas, 0,5–1 falso positivo es lo esperable a un α honesto del 5%: el cruzado corrige el **tamaño** del test, no la **multiplicidad**); (iii) los enfoques estrictos (BIC cruzado, LASSO-BIC, stability) coinciden en las mismas 5 variables y ganan en HO, pero pierden señales débiles verdaderas (`canal`, `deuda_otras_prom_12m`, `carga_financiera`); (iv) el modelo completo con 10 ruidos tiene el peor HO; (v) el oráculo tiene un β > 0 (trampa 5).

**Forzar `edad`** (el ejemplo del notebook con la configuración por defecto): LR = 2,22 (p = 0,14), β = −1,22 (signo esperado), ΔGini HO con IC95% bootstrap pareado [−0,010; +0,002] y OOT [−0,005; +0,005]. Es el caso típico de la RESERVA: costo indistinguible de cero, decisión de gobierno. `edad` es un proxy (en el generador solo actúa vía antigüedad) y es una variable con restricciones legales en algunas jurisdicciones (sección 4): «no cuesta Gini» no es un argumento para incluirla.

### 8.3 Crédito de motos

Una cartera de financiamiento de motos típica de una fintech tiene tres rasgos que empeoran la selección:

1. **Pocos malos en DEV.** Supongamos 3.000 créditos con 150 malos. Entonces $\tfrac1M+\tfrac1G=0{,}0070$, $\mathbb E[\text{IV}_{\text{ruido}}]\approx 0{,}028$ con 5 bins y $\Pr(\text{IV}\ge 0{,}10)\approx 0{,}65\%$: el filtro IV todavía protege a las numéricas.
2. **Categóricas de alta cardinalidad** (concesionario, marca, modelo, comuna). Con 30 concesionarios como niveles, $\mathbb E[\text{IV}_{\text{ruido}}]\approx 29\times0{,}0070=0{,}20$ y $\Pr(\text{IV}\ge 0{,}10)\approx 0{,}99$: **un concesionario asignado al azar pasa el filtro IV y el stepwise casi con certeza.** Con 10 niveles, 11% y 92% respectivamente. Esta es la trampa 2 en su forma más cara: una variable de concesionario que «funciona» en DEV y se desordena apenas cambia la mezcla de concesionarios (lo cual además es frecuente, por campañas comerciales).
3. **Drivers de producto** (pie, plazo, cilindrada, relación cuota/renta) muy correlacionados entre sí y con la política comercial vigente: la selección entre ellos es inestable y a menudo refleja la política de originación del período, no el riesgo.

Recomendación operativa: (a) agrupar concesionarios por volumen mínimo **antes** de ver el target (p. ej., ≥ 5% de la cartera o «otros»), o usar WoE cruzado con suavizado fuerte; (b) seleccionar con BIC o α = 0,01 sobre WoE cruzado; (c) bootstrap de la selección con B ≥ 200 y reportar frecuencias; (d) forzar cobertura conceptual mínima (capacidad: cuota/renta; conducta: bureau; producto: pie/plazo; relación: antigüedad) y documentar las forzadas con LR y ΔGini; (e) si la cartera tiene menos de ~100 malos, preferir un modelo de 4–6 variables con signos restringidos (LASSO con signo + re-ajuste) a un stepwise de 10.

---

## 9. Preguntas de comité

**1. «¿Por qué estas 8 variables y no otras?»**
*Respuesta modelo (con cifras ilustrativas):* Por tres filtros en orden (estabilidad, poder, redundancia) y un stepwise con criterio, umbral y política de signos declarados en la bitácora. Adjuntamos la frecuencia de inclusión bootstrap: por ejemplo, las 6 primeras entran en más del 90% de las réplicas y las 2 últimas en 55–70%; esas dos las mantenemos por concepto (tendencia reciente de utilización, exposición externa). En Austral, un criterio más estricto (BIC) habría dejado 7.

**2. «Todas tienen p < 0,05, ¿entonces todas son significativas?»**
*Respuesta modelo:* Los p-valores del modelo final están condicionados a la selección: se eligieron porque tenían p chico. Son descriptivos. La evidencia de que aportan es su estabilidad en bootstrap, su signo y su comportamiento en HO/OOT, no el p de DEV.

**3. «¿Qué garantiza que ninguna variable sea ruido?»**
*Respuesta modelo:* Nada lo garantiza; lo controlamos. El filtro IV ≥ 0,10 equivale a un test al ~0,3% con nuestro número de malos; además seleccionamos con WoE cruzado (el WoE ajustado en la misma muestra hace pasar ruido con probabilidad ~43% por variable) y en CI inyectamos variables canario de ruido que el procedimiento no elige.

**4. «Pedimos incluir renta. ¿Cuánto cuesta?»**
*Respuesta modelo:* Se reporta el LR anidado con su p, y el ΔGini en HO y OOT con IC95% bootstrap pareado (en el ejemplo del notebook con `edad`: LR = 2,22, p = 0,14; HO [−0,010; +0,002]). Si el intervalo contiene 0, el costo en discriminación es indistinguible de cero. Queda registrada como forzada **al final** (no altera la selección), con autor y justificación. Si su β fuera positivo, no la incluiríamos sin una explicación escrita del signo.

**5. «Si re-entrenamos el próximo año, ¿cambiarán las variables?»**
*Respuesta modelo:* Las de frecuencia de inclusión cercana a 100% no deberían. Las de la zona 30–70% pueden cambiar sin que signifique deterioro. Lo documentamos ahora para que un cambio futuro se lea contra esta línea base y no como alarma.

**6. «¿Por qué no usar LASSO o un algoritmo automático?»**
*Respuesta modelo:* Lo corrimos: LASSO con restricción de signo y λ por BIC elige un subconjunto de nuestras variables con el mismo Gini OOT (±0,01). Lo usamos como contraste. La selección final agrega criterios que ningún algoritmo trae: cobertura de conceptos, costo y disponibilidad del dato, restricciones legales.

**7. «¿Qué variable, si falla la fuente, deja el modelo sin sustento?»**
*Respuesta modelo:* Mostramos la cobertura por concepto (utilización, conducta, relación, endeudamiento, capacidad) y el ΔGini de retirar cada una (un análisis *leave-one-out* en HO); el objetivo es que ninguna concentre una fracción dominante del Gini. Esa es la razón de la clusterización conceptual del curso.

**8. «¿El orden de entrada indica importancia?»**
*Respuesta modelo:* Solo parcialmente: el primer paso sí es la variable más informativa sola; los siguientes dependen de lo que ya entró. La importancia para el comité se discute en rango de puntos del scorecard (M13), no en orden de entrada.

---

## 10. Ejercicios

**E1 (cálculo).** Con 165 malos y 3.322 solicitudes, calcule $\hat\ell_0$ y el LR del primer paso de Austral ($\hat\ell_1=-540{,}7$).

<details><summary>Solución</summary>

$\hat\ell_0=165\ln(165/3322)+3157\ln(3157/3322)=165(-3{,}0024)+3157(-0{,}0509)=-495{,}4-160{,}8=-656{,}2$. LR $=2(656{,}2-540{,}7)=231$, con 1 gl: p ≈ 4·10⁻⁵² (el Wald reportado da 3·10⁻³⁴; ambos son «cero» a efectos prácticos, pero muestran que Wald y LR divergen lejos de $H_0$).
</details>

**E2 (derivación).** Muestre que, con 1 gl, «el AIC baja al agregar $x_j$» equivale a «p-valor LR < 0,157», y que para BIC el α equivalente con $n=10.000$ es ≈ 0,0024.

<details><summary>Solución</summary>

AIC $=-2\hat\ell+2k$. Agregar $x_j$: $\Delta\text{AIC}=-2\Delta\ell+2<0\iff 2\Delta\ell>2$. Como $2\Delta\ell\sim\chi^2_1$ bajo $H_0$, $\alpha=\Pr(\chi^2_1>2)=2[1-\Phi(\sqrt2)]=0{,}157$. BIC: $2\Delta\ell>\ln n=9{,}21$, $\alpha=\Pr(\chi^2_1>9{,}21)=2[1-\Phi(3{,}035)]\approx0{,}0024$.
</details>

**E3 (derivación).** Demuestre que la logística univariada sobre WoE in-sample sin suavizado da $\hat\beta_1=-1$, $\hat\beta_0=\ln(M/G)$, y que el LR contra el nulo es el $G^2$ de la tabla $B\times2$. ¿Por qué con binning supervisado la nula es peor que $\chi^2_{B-1}$?

<details><summary>Solución</summary>

Ver sección 3.4: $\ln(m_b/g_b)=\ln(M/G)-\text{WoE}_b$, así que esos parámetros reproducen la tasa de cada bin, que es el MLE del modelo saturado por bins; al coincidir el máximo, el LR es $2\sum_b[m_b\ln(m_b/\hat m_b^0)+g_b\ln(g_b/\hat g_b^0)]=G^2$. Con binning supervisado los cortes se eligen para maximizar la separación en la misma muestra: se suma una búsqueda sobre cortes (más grados de libertad efectivos) y $G^2$ ya no es $\chi^2_{B-1}$ bajo $H_0$ sino estocásticamente mayor.
</details>

**E4 (cálculo).** Una variable `concesionario` con 25 niveles, en un DEV con 120 malos y 2.400 buenos. Calcule $\mathbb E[\text{IV}]$ bajo $H_0$ y la probabilidad de que un concesionario de ruido supere IV ≥ 0,10.

<details><summary>Solución</summary>

$\tfrac1M+\tfrac1G=0{,}00833+0{,}00042=0{,}00875$. $\mathbb E[\text{IV}]\approx24\times0{,}00875=0{,}21$. $\Pr(\text{IV}\ge0{,}10)=\Pr(\chi^2_{24}>0{,}10/0{,}00875=11{,}4)\approx0{,}985$. El filtro IV no sirve para esta variable; hay que agrupar niveles sin mirar el target o usar WoE cruzado.
</details>

**E5 (winner's curse).** Una variable tiene $\gamma/\sigma=2$ y se retiene si $\hat\gamma/\sigma>1{,}96$. Calcule $\Pr(\text{retenida})$ y $\mathbb E[\hat\gamma\mid\text{retenida}]/\gamma$.

<details><summary>Solución</summary>

$a=1{,}96-2=-0{,}04$. $\Pr=1-\Phi(-0{,}04)=0{,}516$. $\varphi(-0{,}04)=0{,}3986$; razón de Mills $0{,}3986/0{,}516=0{,}772$. Inflación $=1+0{,}772/2=1{,}39$: una variable con potencia ~50% sale en promedio 39% más fuerte de lo que es cuando se la reporta.
</details>

**E6 (derivación).** Derive el operador proximal con restricción de signo de la sección 3.7 y muestre que con $\alpha=1$ es $\min(0,z+t\lambda)$.

<details><summary>Solución</summary>

Minimizar $\phi(b)=\tfrac12(b-z)^2-t\lambda\alpha b+\tfrac{t\lambda(1-\alpha)}2b^2$ en $b\le0$ ($|b|=-b$). $\phi$ es cuadrática convexa; su mínimo sin restricción cumple $b-z-t\lambda\alpha+t\lambda(1-\alpha)b=0\Rightarrow b^{\star}=(z+t\lambda\alpha)/(1+t\lambda(1-\alpha))$. Si $b^{\star}\le0$ es la solución; si no, por convexidad el mínimo en $b\le0$ está en la frontera $b=0$. Como el denominador es positivo, $\operatorname{sign}(b^{\star})=\operatorname{sign}(z+t\lambda\alpha)$ y la solución es $\min(0,z+t\lambda\alpha)/(1+t\lambda(1-\alpha))$. Con $\alpha=1$: $\min(0,z+t\lambda)$.
</details>

**E7 (diseño).** Escriba el esquema JSON de una entrada de bitácora de decisión humana (forzar/excluir) con los campos mínimos para que un validador independiente pueda reproducir su costo.

<details><summary>Solución</summary>

`{"accion": "forzar|excluir", "variable": str, "momento": "inicio|final", "autor": str, "fecha": ISO-8601, "ticket_comite": str, "justificacion": str, "config_base_hash": sha256, "llf_sin": float, "llf_con": float, "LR": float, "p_LR": float, "beta": float, "gini_ho_sin": float, "gini_ho_con": float, "ic95_delta_gini_ho": [float, float], "semilla_bootstrap": int, "B": int}`. Con `config_base_hash` y la semilla, el validador re-ejecuta y compara.
</details>

**E8 (código).** En el notebook, suba el número de bins del WoE del juguete de Freedman (modifique `woe_rapido(..., bins=10)`) y compare la media de variables de ruido elegidas con la predicción $K\cdot\Pr(\chi^2_9>3{,}84)$ para la primera ronda.

<details><summary>Solución</summary>

$\Pr(\chi^2_9>3{,}84)=0{,}92$: prácticamente todas las variables de ruido pasan la primera ronda. En rondas siguientes el umbral es el mismo pero el residuo cambia, y el stepwise sigue agregando hasta el tope. Con $K=50$, $n=2.000$ y 10 réplicas, la media observada es ≈ 39 de 50 variables de ruido en el modelo (con 5 bins era 19,3); no llega a 50 porque cada variable que entra consume parte del «azar» disponible del residuo. La lección: el daño crece con el número de bins, no con la «calidad» de la variable.
</details>

**E9 (código).** Modifique `stepwise` para que el test de **salida** también sea LR (re-ajustando el modelo sin cada variable) y compare el tiempo y la selección con la versión Wald en el pool del notebook.

<details><summary>Solución</summary>

En el bucle backward, para cada $u\in S$ ajustar $S\setminus\{u\}$ y calcular $2[\hat\ell(S)-\hat\ell(S\setminus u)]$; sale la de menor LR si es < umbral. El costo por ronda pasa de 0 ajustes extra a $|S|$ ajustes. Espere diferencias solo en variables cercanas al umbral: Wald y LR son asintóticamente equivalentes y con $n\approx3.400$ divergen poco, salvo en coeficientes grandes (Hauck–Donner) o en variables con pocos malos por bin. Si la selección cambia, esa variable es de la zona inestable y la bitácora debe registrarlo.
</details>

**E10 (diseño).** Proponga un test de CI que falle si alguien cambia el binning a uno supervisado sin cambiar el test de selección.

<details><summary>Solución</summary>

Test de canarios: generar 20 columnas de ruido con la misma distribución marginal que variables reales (permutando filas de variables reales, lo que preserva la marginal y destruye la relación con $y$), pasarlas por el pipeline completo (binning + WoE + selección) y exigir que la fracción de canarios seleccionados sea ≤ 5% (con un umbral de tolerancia por azar, p. ej. binomial con p = 0,05 al 99%). Con binning supervisado y WoE in-sample, la fracción se dispara y el test falla.
</details>

---

## 11. Referencias

- **Freedman, D. A. (1983).** «A note on screening regression equations». *The American Statistician*, 37(2), 152–155. — El experimento original: 100 observaciones, 50 regresores de ruido, tamizado a p < 0,25 y re-ajuste; la base de la sección 5 del notebook.
- **Lukacs, P. M., Burnham, K. P. & Anderson, D. R. (2010).** «Model selection bias and Freedman's paradox». *Annals of the Institute of Statistical Mathematics*, 62, 117–125 (verificar volumen/páginas). — Muestra que promediar modelos (AIC) mitiga la paradoja.
- **Austin, P. C. & Tu, J. V. (2004).** «Bootstrap methods for developing predictive models». *The American Statistician*, 58(2), 131–137. — Frecuencia de inclusión bootstrap aplicada a logística; referencia citada en el brief.
- **Sauerbrei, W. & Schumacher, M. (1992).** «A bootstrap resampling procedure for model building: application to the Cox regression model». *Statistics in Medicine*, 11, 2093–2109. — El origen de las frecuencias de inclusión.
- **Meinshausen, N. & Bühlmann, P. (2010).** «Stability selection». *JRSS B*, 72(4), 417–473. — Definición y cota de falsos positivos.
- **Shah, R. D. & Samworth, R. J. (2013).** «Variable selection with error control: another look at stability selection». *JRSS B*, 75(1), 55–80. — Pares complementarios y cotas más finas.
- **Heinze, G., Wallisch, C. & Dunkler, D. (2018).** «Variable selection – A review and recommendations for the practicing statistician». *Biometrical Journal*, 60(3), 431–449. — La mejor revisión práctica: cuándo usar (y no) stepwise, recomendaciones de bootstrap.
- **Harrell, F. E. (2015).** *Regression Modeling Strategies*, 2.ª ed. Springer. — La crítica clásica al stepwise (sesgos, SE, R²) y alternativas.
- **Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X. (2013).** *Applied Logistic Regression*, 3.ª ed. Wiley. — Cap. 4: *purposeful selection* y stepwise en logística. Complemento: Bursac, Z. et al. (2008), *Source Code for Biology and Medicine*, 3:17.
- **Siddiqi, N. (2017).** *Intelligent Credit Scoring*, 2.ª ed. Wiley. — La práctica de scorecards: selección con criterio de negocio, IV y stepwise.
- **Thomas, L. C., Crook, J. & Edelman, D. (2017).** *Credit Scoring and Its Applications*, 2.ª ed. SIAM. — Tratamiento más formal de la construcción de scorecards.
- **Tibshirani, R. (1996).** «Regression shrinkage and selection via the lasso». *JRSS B*, 58(1), 267–288. — El LASSO.
- **Zou, H. & Hastie, T. (2005).** «Regularization and variable selection via the elastic net». *JRSS B*, 67(2), 301–320. — Elastic net y colinealidad.
- **Zou, H., Hastie, T. & Tibshirani, R. (2007).** «On the "degrees of freedom" of the lasso». *Annals of Statistics*, 35(5), 2173–2192. — Justifica BIC con el número de coeficientes no nulos.
- **Meinshausen, N. (2007).** «Relaxed lasso». *Computational Statistics & Data Analysis*, 52(1), 374–393. — Seleccionar con LASSO, estimar con menos encogimiento.
- **Zou, H. (2006).** «The adaptive lasso and its oracle properties». *JASA*, 101(476), 1418–1429.
- **Friedman, J., Hastie, T. & Tibshirani, R. (2010).** «Regularization paths for generalized linear models via coordinate descent». *Journal of Statistical Software*, 33(1). — `glmnet` (incluye cotas `lower.limits`/`upper.limits`).
- **Beck, A. & Teboulle, M. (2009).** «A fast iterative shrinkage-thresholding algorithm for linear inverse problems». *SIAM J. Imaging Sciences*, 2(1), 183–202. — FISTA, el solver del notebook.
- **Berk, R., Brown, L., Buja, A., Zhang, K. & Zhao, L. (2013).** «Valid post-selection inference». *Annals of Statistics*, 41(2), 802–837. — PoSI: intervalos simultáneos válidos tras cualquier selección.
- **Lee, J. D., Sun, D. L., Sun, Y. & Taylor, J. E. (2016).** «Exact post-selection inference, with application to the lasso». *Annals of Statistics*, 44(3), 907–927. — Inferencia selectiva exacta.
- **Furnival, G. M. & Wilson, R. W. (1974).** «Regressions by leaps and bounds». *Technometrics*, 16(4), 499–511. — Best subset por ramificación y acotamiento.
- **Bertsimas, D., King, A. & Mazumder, R. (2016).** «Best subset selection via a modern optimization lens». *Annals of Statistics*, 44(2), 813–852. — Best subset con MIO.
- **Barber, R. F. & Candès, E. J. (2015).** «Controlling the false discovery rate via knockoffs». *Annals of Statistics*, 43(5), 2055–2085.
- **Burnham, K. P. & Anderson, D. R. (2002).** *Model Selection and Multimodel Inference*, 2.ª ed. Springer. — AIC, su justificación y el promedio de modelos.
- **Kursa, M. B. & Rudnicki, W. R. (2010).** «Feature selection with the Boruta package». *Journal of Statistical Software*, 36(11). — Las *shadow features*, origen de la idea de canarios de la sección 6.
- **SAS Institute.** *SAS/STAT User's Guide*, `PROC LOGISTIC`, «Effect-Selection Methods» y opciones `SLENTRY`/`SLSTAY` del `MODEL`. — Entrada por score, salida por Wald, 0,05 por defecto.
- **EBA (2017).** *Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures* (EBA/GL/2017/16). — Selección de *risk drivers* con amplitud de información, consulta a expertos de negocio y documentación del juicio humano (párrs. 35, 57–58; verificar numeración en la versión vigente).
- **Board of Governors of the Federal Reserve System & OCC (2011).** *SR 11-7: Supervisory Guidance on Model Risk Management*. — Documentación y justificación de las decisiones de desarrollo. Origen del vocabulario de riesgo de modelo; reemplazada el 17-abr-2026 por SR 26-2 (Fed; OCC Bulletin 2026-13), el marco actual (ver M22).
- **12 CFR §1002.6(b)(2) (Regulation B, EE.UU.).** — Uso de la edad en sistemas de scoring empíricamente derivados.

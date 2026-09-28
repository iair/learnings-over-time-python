# M23 · Numpy optimizado vs scipy, statsmodels, scikit-learn, optbinning y nikodym

> **Ficha.** Módulo transversal: toca el código de las clases 3 (logística, optbinning), 4 (δ con `brentq`), 5 (Gini con `roc_auc_score`, bootstrap B = 1.000, `binomtest`, Hosmer-Lemeshow) y 6 (nikodym, artefacto congelado, bug del binner). · **Prerrequisitos:** Serie 1 · M5 (fábrica de variables como pipeline declarativo) y E3 (bootstrap); Serie 2 · M10 (logística desde la verosimilitud), M12 (discriminación), M13 (scaling), M15 (calibración). La sección 7 de cada módulo M08–M15 es la materia prima de este. · **Archivos:** `M23_numpy_vs_librerias.md` (este documento) y `M23_numpy_vs_librerias.py` (notebook Marimo con benchmarks `timeit`, demostraciones de inestabilidad numérica, tabla de 18 convenciones verificadas con `assert` y un arnés de paridad). · **Tiempo estimado:** 3 h de lectura + 1 h de notebook.

Versiones con las que se verificó todo lo que este documento afirma sobre defaults: numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, statsmodels 0.15.0, scikit-learn 1.8.0, optbinning 1.0.0 (Python 3.11, OpenBLAS 0.3.31, 2 núcleos). Los defaults cambian entre versiones: lo que aquí se dice vale para esas versiones y el notebook falla si una actualización lo contradice.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

El curso usa, sin decirlo explícitamente, una **arquitectura mixta**:

| Cálculo | Clase | Implementación del curso |
|---|---|---|
| Binning, WoE, IV | 3, 4 | `binear()` y `tabla_woe()` escritas a mano con pandas (`np.nanquantile` + `pd.cut` + `groupby`), suavizado +0,5 |
| VIF | 3; Lab 2 | `statsmodels.stats.outliers_influence.variance_inflation_factor` |
| Logística | 3–5 | `sm.Logit(y, X).fit(disp=0, maxiter=200)` |
| Scorecard «automático» | 3, 5 | `optbinning.BinningProcess` + `Scorecard(estimator=LogisticRegression(max_iter=1000))` |
| Gini | 3–6 | `2 * roc_auc_score(y, pd) - 1` |
| δ exacto | 4 | `brentq(lambda d: expit(lp_dev + d).mean() - TC, -5, 5, xtol=1e-12)`: 0,177 vs 0,143 aproximado |
| Bootstrap del Gini | 5 | bucle a mano, B = 1.000, semillas 20260916/20260917 |
| Backtesting por banda | 5 | `stats.binomtest(malos, n, pd).pvalue` (default bilateral) |
| Hosmer-Lemeshow | 5 | a mano; χ² = 29,0 en OOT; «χ² diría 0,0003», p simulado 0,012 |
| Monitoreo | 5 | `optbinning.scorecard.ScorecardMonitoring` (CSI 0,004 en `deuda_interna_max_3m`) |
| Corrida gobernada | 6 | nikodym 1.11.0: config validado, contrato, `run`, trail, lineage, model card |

La clase 3 cerró con tres experimentos en Banco Austral: el embudo a mano (8 variables, Gini 0,757 / 0,698 / 0,675 en DEV/HO/OOT), optbinning con las mismas 8 (0,766 / 0,683 / 0,684: «empate, gana monotonía, no Gini») y optbinning con el pool de 94 (66 variables, 0,805 / 0,657 / 0,677: «sobreajuste»). La clase 6 corrió el mismo dataset en nikodym en 7,8 s: Gini 0,7831 / 0,6949 / 0,6938, «mismo dataset, dos pipelines; los Ginis se parecen, las variables no tienen por qué coincidir». Y dejó la frase que ordena este módulo: *«Todo lo que hoy corre solo lo escribimos a mano en las clases 3, 4 y 5: la librería no reemplaza el juicio, lo industrializa».*

**Lo que el curso no trató, y este módulo sí:**

1. **Los defaults como decisiones de modelamiento.** `LogisticRegression(max_iter=1000)` de las clases 3 y 5 regulariza con `C = 1`: el experimento 2 no es un MLE. `binomtest` es bilateral por defecto. `transform()` de optbinning asigna WoE 0 a los códigos especiales. Ninguno de estos defaults aparece en las láminas.
2. **Estabilidad numérica.** Con PD de consumo y 8 variables WoE nada explota, pero el mismo código aplicado a *low default portfolios*, a modelos con separación o a motores float32 sí falla, y falla en silencio.
3. **Rendimiento.** El bootstrap de la clase 5 es un bucle de Python. Para una muestra es irrelevante; para 24 meses × 8 segmentos × 3 métricas de monitoreo no lo es.
4. **Superficie de riesgo de las dependencias** (supply chain, *pinning*, CVE) y **portabilidad** a un motor que no corre Python.
5. **Validación de la herramienta.** Quién valida optbinning o nikodym cuando el modelo que se firma depende de ellos; cómo se demuestra que numpy, statsmodels y el motor de producción calculan lo mismo (tests de paridad).
6. **Convenciones que chocan**: intervalos `(a, b]` del curso vs `[a, b)` de optbinning; −9 y −99 que el binner del curso junta; `ddof` distinto entre numpy y pandas.

---

## 2. Intuición

### 2.1 Tres tipos de código en un modelo de crédito

Un scorecard tiene tres capas de código con requisitos distintos:

- **Especificación**: qué significa cada número (WoE con suavizado +0,5, intervalos `(a, b]`, PD = σ(η + δ), Gini con empates a ½). Aquí se necesita **legibilidad y control total** de la convención.
- **Estimación e inferencia**: optimización (MLE, binning óptimo), errores estándar, tests, colas de distribuciones. Aquí se necesita **corrección probada por miles de usuarios**: reimplementar la cola de una χ² no crea valor.
- **Aplicación**: el score en producción. Aquí se necesita **cero dependencias estadísticas, determinismo y portabilidad**: un lookup y una suma.

numpy es la herramienta natural para la primera y la tercera; scipy/statsmodels para la segunda; scikit-learn y optbinning son herramientas de **desarrollo y exploración** cuyas convenciones hay que traducir a la especificación; nikodym es una capa de **gobierno** encima de todo lo anterior.

### 2.2 Una librería es un conjunto de decisiones que no tomaste

Cada default es una decisión de modelamiento que alguien tomó por ti, razonable para su caso de uso: scikit-learn es una librería de *machine learning* predictivo y regularizar por defecto es sensato ahí; en un scorecard cuyo expediente dice «máxima verosimilitud», no. El riesgo no es el default: es la **brecha no documentada** entre el default y lo que el expediente dice. La tabla de convenciones del notebook (18 filas) es el inventario de esas brechas para las versiones instaladas.

### 2.3 Dónde se va el tiempo

Una llamada de Python a una función de librería cuesta microsegundos de sobrecarga fija (validar entradas, crear objetos, despachar tipos). Una operación vectorizada de numpy sobre 10.000 números cuesta microsegundos de trabajo real. Por eso:

- en llamadas **pequeñas y repetidas** (bootstrap, monitoreo por segmento, stepwise) domina la sobrecarga y numpy puro gana por 10–100×;
- en llamadas **grandes y únicas** domina el algoritmo ($O(n\log n)$ de ordenar) y la ventaja se diluye a 2–8×.

El notebook lo mide: la razón pandas/numpy en la tabla WoE pasa de ~70× con $n$ = 2.000 a 2,6× con $n$ = 100.000.

### 2.4 Dónde se pierden los dígitos

float64 tiene 53 bits de mantisa: ~16 dígitos decimales. Se pierden de tres maneras: **restando números casi iguales** ($1-p$ con $p\approx 10^{-12}$), **multiplicando o sumando muchos números** (el producto de 30.000 probabilidades de supervivencia es $10^{-807}$, que no existe en float64) y **elevando al cuadrado el condicionamiento** (formar $X^\top WX$). Las tres tienen remedio conocido (`log1p`, logaritmos/`logsumexp`, QR) y las tres son invisibles hasta que fallan.

### 2.5 Criterios de decisión

| Criterio | Pregunta | numpy puro | scipy | statsmodels | scikit-learn | optbinning | nikodym |
|---|---|---|---|---|---|---|---|
| Correctitud | ¿calcula lo que el expediente dice? | la que tú escribas (y testees) | alta; colas y optimizadores de referencia | alta para GLM/inferencia | alta, **con otra definición** (C = 1) | alta, con sus convenciones | la de su motor (no verificable aquí) |
| Estabilidad numérica | ¿resiste colas, separación, float32? | depende de ti (`logaddexp`, `log1p`) | `expit`, `log_expit`, `logsumexp` | buena; avisa separación | buena | buena | — |
| Rendimiento | ¿cuánto cuesta repetirlo 10⁴ veces? | el mejor en llamadas chicas | bueno | sobrecarga alta | sobrecarga alta | optimización costosa | corrida completa: 7,8 s |
| Transparencia | ¿lo puede leer un validador en una tarde? | 20–60 líneas por función | código grande, bien documentado | ídem | ídem | modelo MIP/CP; salida tabular clara | config + trail + model card |
| Dependencias | ¿qué arrastra? | 0 | 1 | 12 | 4 | 34 (ortools, cvxpy, protobuf…) | no medido |
| Portabilidad | ¿se traduce a SQL/Java? | trivial (tabla + suma) | no aplica (es desarrollo) | no aplica | ONNX/PMML con esfuerzo | exporta tabla | exporta artefacto (según demo) |
| Mantenibilidad | ¿quién la arregla en 5 años? | tu equipo | comunidad grande | comunidad mediana | comunidad grande | un mantenedor principal (verificar) | proveedor del curso |
| Validación de la herramienta | ¿cómo se demuestra que está bien? | tests contra oráculo | es el oráculo | es el oráculo | oráculo con `C=np.inf` | contra binning propio con `user_splits` | paridad contra corrida a mano (clase 6 lo hizo) |

La conclusión de la tabla no es «usar numpy»: es que **cada capa tiene un ganador distinto** (sección 7 y tabla F del notebook).

---

## 3. Formalización

### 3.1 El modelo de punto flotante

Para las operaciones básicas, la aritmética IEEE 754 garantiza

$$\mathrm{fl}(x \circ y) = (x\circ y)(1+\delta),\qquad |\delta|\le u,$$

con unidad de redondeo $u = 2^{-53}\approx 1{,}11\times10^{-16}$ en float64 y $u=2^{-24}\approx 5{,}96\times10^{-8}$ en float32. El épsilon de máquina ($\varepsilon=2u$, distancia de 1 al siguiente float) es $2{,}22\times10^{-16}$ y $1{,}19\times10^{-7}$. Todo lo que sigue se deriva de esta única ecuación.

### 3.2 Cancelación en $\ln(1-p)$ y en $1-P(\text{bueno})$

**$\ln(1-p)$ ingenuo.** Se calcula $q=\mathrm{fl}(1-p)=(1-p)(1+\delta)$ y luego $\ln q = \ln(1-p)+\ln(1+\delta)\approx \ln(1-p)+\delta$. Como $\ln(1-p)\approx -p$ para $p$ chico, el error relativo es

$$\frac{|\ln q - \ln(1-p)|}{|\ln(1-p)|}\approx\frac{|\delta|}{p}\le\frac{u}{p}.$$

Con $p=10^{-10}$ la cota es $1{,}1\times10^{-6}$ (el notebook observa $8{,}3\times10^{-8}$); con $p<u$, $\mathrm{fl}(1-p)=1$ y $\ln q = 0$: error relativo 100%. `np.log1p(-p)` calcula $\ln(1+x)$ sin formar $1+x$ y tiene error relativo $O(u)$ para todo $p$.

**PD por complemento.** Si un sistema calcula $P(\text{bueno})=\sigma(-\eta)$ y luego $\text{PD}=1-P(\text{bueno})$, el error **absoluto** de $P(\text{bueno})\approx 1$ es del orden de $u$ (el espaciado de los floats justo bajo 1 es $2^{-53}$), luego

$$\frac{|\widehat{\text{PD}}-\text{PD}|}{\text{PD}}\lesssim\frac{u}{\text{PD}}.$$

El notebook: error relativo $8{,}9\times10^{-5}$ con PD $=10^{-12}$, $8{,}0\times10^{-4}$ con $10^{-14}$ y 100% (PD = 0) desde $10^{-16}$. Calcular directamente $\sigma(\eta)$ no tiene el problema.

**Logit.** $\text{logit}(p)=\ln p-\ln(1-p)$ se calcula bien como `log(p) - log1p(-p)` (o `scipy.special.logit`) cuando $p$ es chico; cuando $p\to1$ ya se perdió información al almacenar $p$ y ninguna fórmula la recupera: por eso se propagan **log-odds**, no probabilidades. El notebook muestra `logit(1 - 1e-17) = inf` mientras $-\text{logit}(10^{-17}) = 39{,}14$.

### 3.3 Log-verosimilitud y sigmoide estables

Con $p=\sigma(z)=1/(1+e^{-z})$:

$$\ln\sigma(z)=-\ln(1+e^{-z}),\qquad \ln(1-\sigma(z))=\ln\sigma(-z)=-\ln(1+e^{z}).$$

Usando $\ln(1+e^{-z})=\ln(1+e^{z})-z$:

$$\ell_i=y_i\ln\sigma(z_i)+(1-y_i)\ln(1-\sigma(z_i)) = -y_i\big[\ln(1+e^{z_i})-z_i\big]-(1-y_i)\ln(1+e^{z_i}) = y_i z_i-\ln(1+e^{z_i}).$$

Y $\ln(1+e^{z})$ se evalúa sin overflow como

$$\text{logaddexp}(0,z)=\max(0,z)+\text{log1p}\big(e^{-|z|}\big),$$

porque $e^{-|z|}\in(0,1]$ nunca desborda. La forma ingenua tiene **dos** fallas distintas: (i) $\ln 0=-\infty$ cuando $\sigma(z)$ redondea a 0 o 1; (ii) $0\cdot(-\infty)=\text{NaN}$: con $y=0$ y $p$ redondeado a 0, el término $y\ln p$ es NaN y contamina la suma aunque «no debería contar». En float64 la sigmoide satura cuando $e^{-z}<u$, es decir $z> 53\ln 2 = 36{,}7$; en float32, $z>24\ln2=16{,}6$. El notebook lo muestra: 10 de 26 celdas fallan en float64 (todas con $|z|\ge 40$) y 17 de 26 en float32 (desde $|z|=17$).

La **sigmoide** estable usa dos ramas: $\sigma(z)=1/(1+e^{-z})$ si $z\ge0$ y $e^{z}/(1+e^{z})$ si $z<0$; en ambas el exponente es $\le 0$. Nota honesta: en numpy 2.4, `1/(1+np.exp(-z))` con $z=-800$ devuelve el 0 correcto con un `RuntimeWarning` de overflow; el problema no es la sigmoide sino **lo que se hace después con ella** (logaritmos, complementos).

El gradiente no necesita logaritmos: $\partial\ell/\partial\beta = X^\top(y-p)$, así que el IRLS es estable aunque la log-verosimilitud ingenua no lo sea; lo que falla es el **reporte** (log-verosimilitud, devianza, AIC, test LR).

### 3.4 Productos y sumas de exponenciales: `logsumexp`

**Probabilidad de cero defaults** en $n$ créditos independientes: $\ln P = \sum_i \text{log1p}(-p_i)$. Con 30.000 créditos y PD uniforme entre 2% y 10%, $\ln P = -1.858{,}4$, es decir $P=10^{-807}$, fuera del rango de float64 (mínimo subnormal $\approx 4{,}9\times10^{-324}$); `np.prod(1 - p)` devuelve 0.

**Log-sum-exp.** Para $a\in\mathbb{R}^S$ y $m=\max_s a_s$:

$$\text{LSE}(a)=\ln\sum_s e^{a_s}=m+\ln\sum_s e^{a_s-m}.$$

Todos los exponentes son $\le0$ y al menos uno es 0, luego la suma está en $[1,S]$: sin overflow ni underflow total. Los **pesos posteriores de escenarios** $w_s\propto\pi_s e^{\ell_s}$ son un *softmax*: $w_s=\exp(\ln\pi_s+\ell_s-\text{LSE}(\ln\pi+\ell))$. En el notebook, la cohorte OOT (deterioro plantado de +0,35 en log-odds) bajo tres desplazamientos del log-odds da $\ell = -1.750{,}9$, $-1.723{,}4$ y $-1.762{,}2$: $e^{\ell}$ es 0 en float64, la versión ingenua devuelve `nan` en los tres pesos y la versión LSE asigna ≈100% al escenario adverso (diferencia de 27,5 en log-verosimilitud: factor de Bayes $\approx e^{27{,}5}\approx 9\times10^{11}$).

### 3.5 Error de suma

Para la suma recursiva (secuencial) $\hat S$ de $n$ términos (Higham 2002, cap. 4):

$$|\hat S - S|\le (n-1)\,u\sum_i|x_i| + O(u^2).$$

Para la suma **por pares** (recursiva en mitades) la cota es $\lceil\log_2 n\rceil\,u\sum_i|x_i|$, y para Kahan (suma compensada) $2u\sum_i|x_i|$. Con float32 y $n=10^6$, las cotas son $6\%$ (secuencial) y $1{,}2\times10^{-6}$ (pares); el notebook observa $1{,}7\times10^{-4}$ y $2{,}8\times10^{-9}$ (las cotas son de peor caso; el error típico crece como $\sqrt n\,u$). La documentación de `np.sum` dice que numpy usa suma por pares **parcial** «en muchos casos» y siempre cuando no se da `axis`; no lo garantiza para todos los ejes. Un motor SQL que acumula fila a fila, un bucle de Python o `np.cumsum` son secuenciales. **Consecuencia para paridad**: dos implementaciones correctas de la misma log-verosimilitud pueden diferir en el cuarto dígito en float32 solo por el orden de suma.

### 3.6 IRLS, ecuaciones normales y condicionamiento

El paso de Newton de la logística es

$$d=(X^\top WX)^{-1}X^\top(y-p),\qquad W=\text{diag}\big(p_i(1-p_i)\big).$$

Es la solución de mínimos cuadrados ponderados

$$d=\arg\min_d\big\lVert W^{1/2}Xd - W^{-1/2}(y-p)\big\rVert^2,$$

porque sus ecuaciones normales son $X^\top W^{1/2}W^{1/2}Xd=X^\top W^{1/2}W^{-1/2}(y-p)=X^\top(y-p)$. Sea $\tilde X=W^{1/2}X=U\Sigma V^\top$ (SVD). Entonces $\tilde X^\top\tilde X=V\Sigma^2V^\top$ y

$$\kappa_2(X^\top WX)=\frac{\sigma_{\max}^2}{\sigma_{\min}^2}=\kappa_2(\tilde X)^2 .$$

Las cotas de error hacia adelante (Golub & Van Loan 2013, §5.3; Higham 2002, cap. 20) son, para $\tilde X\beta = b$ con residuo $r$:

- ecuaciones normales resueltas por Cholesky, LU (`solve`) o inversa: $\dfrac{\lVert\hat\beta-\beta\rVert}{\lVert\beta\rVert}\lesssim \kappa^2 u$;
- QR de Householder o SVD sobre $\tilde X$: $\dfrac{\lVert\hat\beta-\beta\rVert}{\lVert\beta\rVert}\lesssim \kappa u + \kappa^2 u\dfrac{\lVert r\rVert}{\lVert\tilde X\rVert\lVert\beta\rVert}$.

Con residuo nulo QR pierde la mitad de dígitos que las ecuaciones normales; con residuo grande (el caso típico en una logística, donde $W^{-1/2}(y-p)$ no es pequeño) ambas pagan $\kappa^2$, aunque QR con mejor constante. En el experimento del notebook ($2.000\times 6$, residuo nulo): con $\kappa=10^7$ las ecuaciones normales conservan 1,8 dígitos y QR 10,6; con $\kappa = 10^8\approx u^{-1/2}$ las ecuaciones normales no tienen ningún dígito correcto (error 64–70%) y QR conserva ~10.

**Inversa vs `solve`.** La inversa explícita cuesta ~3 veces los flops de una factorización LU y su error es similar o peor; no hay razón para usarla salvo que se necesite la matriz inversa misma (la covarianza de $\hat\beta$, que sí es $(X^\top WX)^{-1}$). En el experimento, `inv` y `solve` tienen errores casi idénticos: el salto de calidad es **ecuaciones normales vs QR**, no inversa vs solve.

**Equilibrado (van der Sluis).** Un $\kappa$ alto puede ser pura escala. Si $D$ es diagonal positiva, el error de Cholesky/LU con pivoteo sobre $A$ se comporta como el de $D^{-1}AD^{-1}$ (van der Sluis 1969; Higham 2002, §7.3): los algoritmos son casi invariantes a escalar columnas. El notebook lo demuestra: con la renta en pesos en lugar de millones, $\kappa(W^{1/2}X)$ pasa de 414 a $6{,}1\times10^{6}$, pero con columnas equilibradas ambos diseños tienen $\kappa = 9{,}3$, y el IRLS por ecuaciones normales da log-odds que difieren en $2{,}7\times10^{-15}$. Las *dummies* completas con intercepto, en cambio, tienen $\kappa\approx10^{16}$ también equilibradas: singularidad real.

**Por qué el WoE está bien condicionado.** Todas las columnas están en la misma unidad (log-odds), con magnitud ~1 y media ponderada cercana a 0; el filtro de correlación ($\le 0{,}70$) y de VIF ($\le 5$) del curso acota la colinealidad (y $\kappa(R)\ge\text{VIF}_{\max}$, M09). El diseño WoE del notebook tiene $\kappa = 9{,}9$ (3,0 equilibrado): $X^\top WX$ pierde 2 de 16 dígitos. Incluso agregando la otra ventana de la misma familia (`uso_tc_prom_3m`, correlación WoE 0,911), $\kappa$ sube solo a 10,8. **La colinealidad de familias es un problema estadístico (varianza de $\hat\beta$), no numérico.**

### 3.7 La penalización de scikit-learn como función de $n$

`LogisticRegression` minimiza (sin penalizar el intercepto con `lbfgs`)

$$J_C(\beta)=\tfrac12\lVert\beta_{-0}\rVert^2 - C\sum_{i=1}^n\ell_i(\beta).$$

Dividiendo por $Cn$: $-\frac1n\sum\ell_i+\frac{1}{2Cn}\lVert\beta_{-0}\rVert^2$, un ridge con peso $\lambda=1/(Cn)$ **por observación**: con $C=1$ fijo, la penalización relativa cae como $1/n$.

**Primer orden en el intercepto.** Como $\beta_0$ no se penaliza, $\partial J_C/\partial\beta_0=-C\sum_i(y_i-p_i)=0$, luego $\bar p=\bar y$ en la muestra de ajuste **exactamente como en el MLE**. Por eso la trampa no se ve en el primer control (PD media vs tasa): el notebook mide una brecha de $1{,}9\times10^{-5}$.

**Contracción por variable.** Sea $H=X^\top\hat WX$ (suma, no promedio) en el MLE $\hat\beta$ e $I_0$ la identidad con un 0 en la posición del intercepto. Un paso de Newton desde $\hat\beta$ sobre $J_C/C$ da

$$\nabla\!\Big(\tfrac{J_C}{C}\Big)(\hat\beta)=\tfrac1C I_0\hat\beta,\qquad \nabla^2\!\Big(\tfrac{J_C}{C}\Big)\approx H+\tfrac1C I_0\;\Longrightarrow\; \hat\beta_C\approx\hat\beta-\Big(H+\tfrac1C I_0\Big)^{-1}\tfrac1C I_0\hat\beta=\Big(H+\tfrac1C I_0\Big)^{-1}H\hat\beta .$$

Si $H$ fuera diagonal (después de centrar), $\hat\beta_{C,j}\approx \dfrac{h_j}{h_j+1/C}\hat\beta_j$ con $h_j=\sum_i\hat w_i(x_{ij}-\bar x_{w,j})^2\approx n\,\bar w\,\mathrm{Var}_w(\text{WoE}_j)$. La contracción es **mayor para variables con poca dispersión de WoE** (IV bajo) y **menor cuanto mayor sea $n$**. En el notebook con $n=1.000$ (101 malos): `canal` tiene $h=1{,}06$ y factor teórico 0,51; sklearn da 0,52. Las variables de alto IV ($h\approx 45$) casi no se tocan (factor 0,98). La aproximación de un paso reproduce la razón observada a menos de 0,015 en todas las variables. La norma de las pendientes de sklearn relativa al MLE es 0,52 con $n=300$, 0,66 con 500, 0,84 con 1.000, 0,89 con 2.000, 0,94 con 5.000 y 0,97 con 10.065.

Corolario: la penalización **no es invariante a la escala** de las columnas. Multiplicar una columna por 10 multiplica su $h_j$ por 100 y prácticamente elimina su contracción. Sobre WoE la escala es común a todas; sobre variables crudas, el resultado depende de las unidades.

### 3.8 AUC por conteos y bootstrap vectorizado

Con PD tomando $k$ valores distintos $v_1<\dots<v_k$, sea $m_j$ el número de malos y $b_j$ el de buenos con PD $=v_j$, y $B_{<j}=\sum_{i<j}b_i$. Un par (malo en $v_j$, bueno en $v_i$) aporta 1 si $i<j$ y ½ si $i=j$, así que

$$\text{AUC}=\frac{1}{n_m n_b}\sum_{j=1}^k m_j\Big(B_{<j}+\tfrac12 b_j\Big).$$

Con los códigos $c_i\in\{0,\dots,k-1\}$ precalculados, $m$ y $b$ salen de dos `np.bincount` y $B_{<j}$ de un `cumsum`: $O(n+k)$, sin ordenar. Para $B$ réplicas bootstrap con matriz de índices $I\in\{0..n-1\}^{B\times n}$, los códigos desplazados $c_{I_{rt}}+k\,r$ son únicos por (réplica, valor) y **un solo** `bincount` de largo $Bk$ entrega todas las tablas: tiempo $O(Bn)$ y memoria $8Bn$ bytes para los índices en int64 (7,7 MB con $B=200$ y $n=4.792$; 800 MB con $B=1.000$ y $n=100.000$).

### 3.9 Tres definiciones que las librerías eligen por ti

**Binomial bilateral.** `scipy.stats.binomtest(k, n, p)` con `alternative='two-sided'` (default) suma las probabilidades de todos los resultados **no más probables** que el observado:

$$p_{\text{bil}}=\sum_{j:\,P(X=j)\le P(X=k)}P(X=j),$$

no $2\min\{P(X\le k),P(X\ge k)\}$. Con $k=3$, $n=50$, $p=0{,}02$: 0,0784 vs 0,1569. Para validar PD, la pregunta de negocio suele ser unilateral («¿se subestima el riesgo?»): `alternative='greater'`.

**Hosmer-Lemeshow.** Con $g$ grupos, $\text{HL}=\sum_{j}\frac{(O_j-E_j)^2}{E_j(1-\bar p_j)}$. En la muestra de **ajuste** la distribución de referencia es $\chi^2_{g-2}$ (dos grados de libertad se consumen al estimar); en una muestra **independiente** con PD fijas, ningún parámetro se estimó con esos datos y la referencia natural es $\chi^2_g$ (así lo hace Stata para fuera de muestra, según la documentación de statsmodels). `statsmodels.stats.diagnostic_gen.test_chisquare_binning` usa $g-2$ por defecto. Con el χ² = 29,0 de Banco Austral OOT: $g-2$ da p = 0,0003 (el número de la clase 5), $g$ da 0,0012 y el p simulado de la clase, 0,012.

**Intervalo para una proporción.** `statsmodels.stats.proportion.proportion_confint(count, nobs)` usa `method='normal'` (Wald) por defecto. Con $k=1$, $n=200$ el Wald es $0{,}005\pm0{,}0098 = [-0{,}0048;\,0{,}0148]$, que statsmodels recorta a $[0;\,0{,}0148]$. Wilson o Clopper-Pearson (`'beta'`) son los recomendados para proporciones chicas (Brown, Cai & DasGupta 2001).

---

## 4. Variantes y alternativas de industria

| Opción | Qué resuelve | Costo | Cuándo usarla | Quién la usa / regulación |
|---|---|---|---|---|
| **numpy puro** (núcleo propio) | especificación ejecutable, scoring, métricas de monitoreo | escribir y testear cada función; responsabilidad total | artefacto de producción, validación independiente, bucles pesados | equipos con ingeniería fuerte; es lo que el validador puede leer completo |
| **scipy** | optimizadores (`brentq`, `minimize`, `milp`), distribuciones y tests exactos, `special` estable | 1 dependencia | colas de χ²/binomial/normal, raíces 1-D, `log_expit`/`logsumexp` | estándar de facto en Python científico |
| **statsmodels** | GLM/Logit con inferencia completa, offset, pesos, sándwich, tests | 12 dependencias; sobrecarga por llamada | el modelo que se firma y su tabla de inferencia | el curso (clases 3–5); análogo a `glm` de R y `PROC LOGISTIC` de SAS |
| **scikit-learn** | pipelines, regularización, métricas, CV | penaliza por defecto; sin SE ni p-valores | exploración, benchmarks ML, métricas (`roc_auc_score`) | equipos de ML; `Scorecard` de optbinning lo usa como estimador |
| **optbinning** | binning óptimo con restricciones (monotonía, tamaño mínimo) por programación entera/CP; scorecard; monitoreo | 34 dependencias (ortools, cvxpy…); convenciones propias | proponer cortes que luego se **congelan** en el artefacto | clases 3 y 5; Navas-Palencia (2020) |
| **nikodym** | corrida gobernada: config validado y con hash, contrato de datos, trail, lineage, model card | dependencia de un proveedor; motor numérico no inspeccionado aquí | industrializar el expediente cuando la especificación ya está clara | clase 6 (v1.11.0) |
| **R** (`glm`, `scorecard`, `pROC`, `logistf`) | ecosistema estadístico maduro; Firth y DeLong disponibles | segundo lenguaje en el stack | validación independiente con implementación distinta | común en validación y academia |
| **SAS** (`PROC LOGISTIC`, Enterprise Miner) | stack histórico de la banca | licencia; caja más cerrada | cuando el banco ya lo usa y el regulador lo conoce | banca tradicional (según entiendo, todavía frecuente en validación) |
| **SQL / motor de decisión** | scoring en producción sin Python | traducir convenciones (`<=` vs `<`, NULL, redondeo) | originación en línea, concesionarios, *batch* nocturno | producción típica de crédito de consumo |
| **ONNX / PMML** | serializar modelos entre lenguajes | cobertura parcial de transformaciones; precisión (float32 en ONNX es común) | cuando el motor consume ese formato | integraciones con plataformas de terceros |
| **numba / JAX** | compilar bucles o diferenciar automáticamente | otra dependencia; JAX usa float32 por defecto (verificar configuración) | simulaciones grandes, gradientes de funciones propias | investigación, equipos cuantitativos |

**Nota sobre regulación.** En EE.UU., SR 11-7 (2011) fue reemplazada el 17 de abril de 2026 por la guía interagencial **SR 26-2** (Fed, OCC, FDIC; ver M22). Según su texto, los productos de proveedores «pueden presentar desafíos únicos para la validación» y la buena práctica incluye «desarrollar un entendimiento del modelo del proveedor, incluida su solidez conceptual, diseño, datos de desarrollo y desempeño»; excluye de la definición de modelo los cálculos aritméticos simples y los procesos deterministas basados en reglas. No menciona explícitamente librerías de código abierto. En el Reino Unido, la SS1/23 de la PRA (vigente desde el 17 de mayo de 2024, para bancos con modelos internos aprobados) se aplica a modelos «desarrollados internamente o externamente (incluidos modelos de proveedores), independiente de la tecnología». Mi lectura, no una cita: una librería abierta cuyo default define el estimador (optbinning con `LogisticRegression()`) está más cerca de un «modelo de proveedor» que de una herramienta neutra, y el validador debe poder explicar sus convenciones. Para Chile/CMF, ver Serie 1 · E6 y verificar con la norma vigente.

---

## 5. Cuándo falla: trampas y modos de falla

**T1 · `LogisticRegression()` no es un MLE.** *Síntoma:* β distintos de statsmodels; PD extremas comprimidas; más en muestras chicas. *Causa:* `C = 1.0` por defecto (L2). *Detección:* comparar contra `sm.Logit` (el notebook: 0,089 de diferencia máxima con $n$ = 10.065; con $n$ = 1.000, la pendiente de `canal` queda en 52% del MLE y el score OOT cambia hasta 6,2 puntos; con $n$ = 200, hasta 38,9 puntos). *Qué hacer:* `C=np.inf` o statsmodels; si se quiere regularizar, declararlo en el expediente con el λ efectivo $1/(Cn)$.

**T2 · Código viejo con `penalty=`.** *Síntoma:* `FutureWarning` en scikit-learn 1.8 (`penalty` deprecado, se elimina en 1.10) y, con `C=np.inf`, un `UserWarning` «Setting penalty=None will ignore the C and l1_ratio parameters». *Causa:* cambio de API. *Detección:* correr los tests con `-W error::FutureWarning`. *Qué hacer:* usar `C` y `l1_ratio`; filtrar explícitamente el aviso esperado de `C=np.inf`. Nota: el M14 de esta serie recomienda `penalty=None`; en 1.8 eso funciona pero está deprecado.

**T3 · Especiales con puntos neutros.** *Síntoma:* el segmento sin bureau (20% de malos) recibe WoE 0. *Causa:* `OptimalBinning.transform()` usa `metric_special=0` y `metric_missing=0` por defecto; la tabla muestra el WoE empírico (−0,6765) pero la transformación no lo usa. *Detección:* test que compare, fila a fila, el WoE de la tabla publicada con el de `transform`. *Qué hacer:* `metric_special="empirical"` o mapeo explícito en el artefacto (M13).

**T4 · Especiales fusionados.** *Síntoma:* −9 (7,7% de malos) y −99 (20,0%) en el mismo bin. *Causa:* `special_codes=[-9, -99]` (lista) crea **un** bin «Special»; el `binear()` del curso los trata como números y los junta en `(-inf, -9.0]`. *Detección:* inventario de códigos del contrato de datos vs bins del artefacto. *Qué hacer:* diccionario (`{"nunca_mora": [-9], "sin_bureau": [-99]}`) y bins propios en el binner propio.

**T5 · `(a, b]` vs `[a, b)`.** *Síntoma:* dos implementaciones con los mismos cortes asignan bins distintos. *Causa:* `pd.cut` y `np.searchsorted(side="left")` usan `(a, b]`; optbinning y `np.digitize` usan `[a, b)`. *Detección:* test con x = cada corte. En `meses_desde_mora_12m` con los cortes del curso (−9, 3, 8), cambiar de convención mueve 2.280 de 10.065 filas de DEV (22,7%), y el código −9 pasa del bin de −99 al de mora reciente `[-9, 3)`. *Qué hacer:* declarar la convención en el artefacto y traducirla explícitamente (`<=` en SQL).

**T6 · WoE que «casi» coinciden.** *Síntoma:* diferencias de $10^{-3}$ entre el WoE del curso y el de optbinning con los mismos cortes (hasta 0,0048 en `uso_linea`). *Causa:* el curso suma 0,5 a cada celda; optbinning no suaviza. Signo: ambos usan $\ln(\%\text{no evento}/\%\text{evento})$, verificado. *Qué hacer:* tolerancias de paridad que reflejen la diferencia o, mejor, comparar conteos (que sí son exactos).

**T7 · `GLM.converged = True` con separación perfecta.** *Síntoma:* un modelo con pendiente 42,7 marcado como convergido. *Causa:* el IRLS de GLM para por cambio de devianza, que se estabiliza aunque el MLE no exista; `Logit` sí marca `converged = False` y emite `ConvergenceWarning`. *Detección:* convertir `PerfectSeparationWarning` en error. *Qué hacer:* Firth (M10) o fusionar bins; nunca aceptar un ajuste solo por la bandera.

**T8 · Backtesting bilateral.** *Síntoma:* una banda con **menos** malos de lo esperado sale en rojo. *Causa:* `binomtest` es bilateral por defecto, y su bilateral no es «2 × cola». *Detección:* revisar la dirección del desvío en las bandas rojas. *Qué hacer:* `alternative='greater'` para la pregunta de subestimación; si se quiere detectar también conservadurismo, declararlo.

**T9 · Intervalo de Wald por defecto.** *Síntoma:* IC con límite inferior 0 en bandas con 0–2 malos. *Causa:* `proportion_confint` usa `method='normal'`. *Qué hacer:* `'wilson'` o `'beta'`.

**T10 · `scipy.stats.bootstrap` con defaults.** *Síntoma:* IC distintos a los de la receta del curso, o absurdamente anchos. *Causa:* `method='BCa'`, `n_resamples=9999`, `paired=False` (sin `paired=True` remuestrea `y` y `s` por separado y destruye la asociación). *Qué hacer:* fijar los cuatro parámetros y `rng`.

**T11 · Hosmer-Lemeshow con grados de libertad de muestra de ajuste.** *Síntoma:* p-valores OOT más pequeños que los simulados (0,0003 vs 0,012 en Austral). *Causa:* $g-2$ por defecto y aproximación χ² con esperados chicos. *Qué hacer:* $g$ fuera de muestra, o p simulado (clase 5).

**T12 · Log-verosimilitud ingenua.** *Síntoma:* `-inf` o `NaN` en la log-verosimilitud, el AIC o el test LR, con β perfectamente razonables. *Causa:* $\ln(1-p)$ con $p$ redondeado a 1, o $0\cdot(-\infty)$. *Detección:* `np.isfinite` en todas las métricas del reporte. *Qué hacer:* $yz-\text{logaddexp}(0,z)$ o `scipy.special.log_expit`.

**T13 · Paridad float32 vs float64.** *Síntoma:* el test de paridad con el motor falla por $10^{-6}$. *Causa:* el motor usa float32 (el ajuste completo en float32 cambia el score en $2\times10^{-5}$ puntos en el notebook). *Qué hacer:* tolerancia justificada por la precisión del motor; referencia float32 si el motor es float32.

**T14 · Orden de acumulación.** *Síntoma:* la suma de log-verosimilitudes en SQL difiere de numpy en el cuarto dígito. *Causa:* suma secuencial en float32 (error $1{,}7\times10^{-4}$) vs por pares (error $2{,}8\times10^{-9}$). *Qué hacer:* acumular en float64 (DOUBLE) en ambos lados.

**T15 · La fórmula del libro traducida literal.** *Síntoma:* un IRLS que no corre con 100.000 filas. *Causa:* `X.T @ np.diag(w) @ X` materializa una matriz $n\times n$ (80 GB con $n$ = 100.000). *Qué hacer:* `X.T @ (X * w[:, None])`; con $n$ = 2.000 ya es entre 400× y 1.600× más rápido.

**T16 · Serialización de cortes.** *Síntoma:* el artefacto en SQL asigna bins distintos en valores exactamente iguales al corte. *Causa:* cortes escritos con 10 dígitos significativos (3 de 4 cortes de `uso_linea` no vuelven al mismo float) o, en numpy 2, `repr(np.float64(0.2))` que produce `'np.float64(0.2)'`. *Qué hacer:* `repr(float(c))` (ida y vuelta exacta) y test de lectura del artefacto.

**T17 · `ddof`.** *Síntoma:* desviaciones estándar que difieren en $\sqrt{n/(n-1)}$. *Causa:* `np.std`/`np.var` usan `ddof=0`; `pd.Series.std` usa `ddof=1`; `np.cov` usa $n-1$. Inconsistencia incluso **dentro** de numpy. *Qué hacer:* pasar `ddof` siempre.

**T18 · AUC al revés.** *Síntoma:* Gini negativo o igual a $1-$Gini esperado. *Causa:* `roc_auc_score(y_true, y_score)` espera un score creciente en la clase positiva (malo = 1); si se le pasa el **score de puntos** (alto = bueno), devuelve $1-\text{AUC}$. *Qué hacer:* pasar PD o `-score`; assert AUC > 0,5 en DEV.

**T19 · Artefacto en pickle.** *Síntoma:* el modelo de producción es un `.pkl` de un objeto sklearn/optbinning. *Causa:* comodidad. *Riesgo:* deserializar un pickle ejecuta código arbitrario (la documentación de persistencia de scikit-learn lo advierte; CVE-2019-6446 afectó a `numpy.load` con `allow_pickle`, cuyo default pasó a `False` en numpy 1.16.3), el objeto depende de la versión exacta de la librería y no es legible por un validador ni portable a SQL. *Qué hacer:* artefacto tabular (JSON/CSV canónico con hash, M13/M21).

**T20 · Benchmarks engañosos.** *Síntoma:* «numpy es 70× más rápido» (verdad con $n$ = 2.000) aplicado a $n$ = 10⁶ (donde es 2–3×). *Causa:* la sobrecarga fija domina en tamaños chicos. *Qué hacer:* medir en el tamaño real, reportar el mínimo de varias repeticiones y la variabilidad (en este entorno, el tiempo de `LogisticRegression` varió entre 20 y 110 ms entre corridas).

**T21 · Vectorizar hasta quedarse sin memoria.** *Síntoma:* el bootstrap vectorizado muere con `MemoryError` en producción. *Causa:* la matriz $B\times n$. *Qué hacer:* bloques de réplicas; índices int32; o remuestreo multinomial por pesos (`rng.multinomial(n, ...)` + `bincount` ponderado) si se quiere $O(B\,k)$ memoria.

---

## 6. Puente con ingeniería

### 6.1 Arquitectura: núcleo auditable + oráculos

```
scorecard/
├── nucleo/                 # numpy puro, sin scipy/statsmodels/sklearn
│   ├── binning.py          # aplicar_cortes(x, cortes, convencion="(a,b]", especiales={...})
│   ├── woe.py              # tabla_woe(n, malos, suavizado=0.5)
│   ├── score.py            # log_odds(artefacto, fila) -> puntos; pd = sigmoide_estable(eta + delta)
│   └── metricas.py         # auc_conteos, psi, ks, bootstrap en bloques
├── desarrollo/             # statsmodels (Logit/GLM), optbinning (propuesta de cortes), scipy (brentq)
├── artefacto/
│   ├── scorecard.json      # cortes (repr), WoE, β, δ, factor, offset, convenciones, hash
│   └── convenciones.yaml   # el contrato explícito (ver 6.2)
└── tests/
    ├── test_paridad.py     # núcleo vs oráculos, muchas carteras aleatorias, tolerancias declaradas
    ├── test_bordes.py      # x = corte, corte ± ε, NULL, −9, −99, fuera de rango
    ├── test_defaults.py    # la tabla de convenciones del notebook como test «canario»
    └── test_golden.py      # 20 filas fijas con score esperado (también contra el motor SQL)
```

El **núcleo** es la especificación ejecutable: lo lee el validador, lo traduce TI a SQL, no cambia con una actualización de sklearn. Las librerías viven en `desarrollo/` y en `tests/` como **oráculo**: si el IRLS del núcleo y `sm.Logit` difieren en más de $10^{-8}$ en alguna de 12 carteras aleatorias, algo está mal en uno de los dos.

### 6.2 El contrato de convenciones

```yaml
convenciones:
  target: "1 = malo (90+ DPD a 12 meses)"
  woe: {formula: "ln(%buenos/%malos)", suavizado: 0.5}
  intervalos: "(a, b]"            # SQL: x <= corte
  especiales:
    meses_desde_mora_12m: {-9: nunca_mora, -99: sin_bureau}   # bins propios, WoE empírico
  missing: "bin propio; si n < 50, WoE del bin más riesgoso (declarado)"
  estimador: "MLE sin penalización (statsmodels Logit, newton)"
  calibracion: {metodo: "delta exacto (brentq)", objetivo: "TC 5,42%"}
  redondeo_puntos: "round half to even por variable; score = suma de enteros"
  precision_motor: float64
  tolerancias_paridad: {conteos: 0, woe: 1e-12, beta: 1e-8, score_puntos: 1e-6}
versiones_validadas: {numpy: 2.4.4, scipy: 1.17.1, statsmodels: 0.15.0, scikit-learn: 1.8.0, optbinning: 1.0.0}
```

Cada línea responde una trampa de la sección 5. El archivo entra al hash del artefacto: cambiar una convención cambia la identidad del modelo, como el `config_hash` de nikodym.

### 6.3 Tipos de test

- **Paridad con oráculo** (sección E del notebook): 6 cálculos × 12 carteras, discrepancia máxima observada ≤ $5\times10^{-14}$ con tolerancias declaradas entre $10^{-12}$ y $10^{-8}$. Las tolerancias se justifican: $10^{-12}$ donde ambos lados hacen la misma aritmética (conteos, rangos), $10^{-8}$ donde hay un criterio de parada iterativo. Una tolerancia de $10^{-3}$ dejaría pasar la trampa `C = 1` en muestras grandes.
- **Invariantes** (tipo *property-based*): AUC ∈ [0, 1]; AUC(−s) = 1 − AUC(s); en el MLE $\sum_i(y_i-p_i)=0$; PD monótona en el score; IV ≥ 0; el δ exacto clava la media; los puntos por bin reconstruyen el score.
- **Bordes**: el notebook muestra que con los cortes de `uso_linea` (cuantiles interpolados) escribir `<` en vez de `<=` en SQL no cambia **ninguna** fila de DEV y sí los 4 de 4 casos sintéticos x = corte. Un test solo sobre datos reales no detecta ese error.
- **Canarios de defaults**: la tabla de convenciones como test. Al subir la versión de una librería, el PR corre la batería; si un default cambió, falla antes de producción.
- **Golden tests del motor**: las mismas 20 filas se puntúan en numpy y en el motor destino (SQL, Java); igualdad de enteros.

### 6.4 Qué se congela y qué se versiona

| Objeto | Congelado en el artefacto | Versionado |
|---|---|---|
| Cortes, WoE, β, δ, factor, offset, puntos | sí (con `repr` de float) | hash del artefacto |
| Convenciones (6.2) | sí | dentro del hash |
| Código del núcleo | — | git SHA |
| Librerías de desarrollo | — | *lockfile* con hashes (`uv.lock`, `pip-tools --generate-hashes`), SBOM |
| Resultados de paridad y canarios | — | adjuntos al expediente con la versión de cada librería |
| Hilos de BLAS | — | fijados (`threadpoolctl`, `OMP_NUM_THREADS`) si se exige reproducibilidad bit a bit; si no, tolerancias |

**Supply chain.** Fijar versiones exactas con hashes, escanear dependencias contra bases de vulnerabilidades (p. ej. `pip-audit` de PyPA) y actualizar por PR con la batería completa. La clausura transitiva medida en el notebook: numpy 0 dependencias, scipy 1, scikit-learn 4, statsmodels 12, optbinning 34 (ortools, cvxpy, protobuf, matplotlib, jinja2…). Un servicio de scoring que solo necesita numpy (o nada) tiene una superficie de ataque y de mantenimiento incomparablemente menor. El costo de arranque medido (importar en un proceso limpio, incluye iniciar Python): numpy 0,1–0,3 s, statsmodels 1,5–2,9 s, optbinning 2,2–3,0 s.

### 6.5 Dónde encaja nikodym

La demo de la clase 6 muestra lo que una capa de gobierno aporta: `NikodymConfig.model_validate(cfg)` (receta validada con hash), `check_dataset` + reglas de rango que bloquean la corrida (331 hallazgos de `antiguedad_meses ≤ 344`), `nikodym.run` que devuelve un `Study` con estado `failed` en vez de lanzar una excepción, `study.artifacts.get(dominio, clave)`, `read_trail` (398 eventos, 341 decisiones), `lineage_bundle()` y `ModelCardBuilder`. Lo que la demo **no** delega: la cadena de hashes y el sello externo (escritos en el notebook; «no son parte de la API de la librería en 1.11.0»), la recomendación y los gatillos. Para este módulo, lo relevante es que nikodym es una capa **encima** del núcleo numérico, no un reemplazo de la validación de ese núcleo. La propia clase 6 hizo el test de paridad correcto: «mismo dataset, dos pipelines» (Gini DEV 0,7570 a mano vs 0,7831 nikodym, con variables distintas). Las etiquetas del scorecard de la demo (`[-0.21, -0.13)`, `Special` con WoE 0,0000) tienen el formato de optbinning; no está verificado aquí si nikodym usa optbinning internamente, pero si lo hace hereda T3–T5, y el validador debe preguntarlo.

---

## 7. Numpy desde cero vs librerías

Este módulo consolida la sección 7 de M08–M15 y agrega lo que su notebook mide.

| Cálculo | Numpy (dónde) | Librería | Diferencia de convención verificada | En producción |
|---|---|---|---|---|
| Asignación a bins | `searchsorted(cortes[1:-1], x, side="left")` (M08, M23) | `pd.cut` (a,b]; `np.digitize` [a,b); `np.histogram` (último cerrado); optbinning [a,b) | 2.280 filas cambian en `meses_desde_mora_12m` | numpy + test de bordes |
| Tabla WoE / IV | `bincount` (M23: 0,2 ms) | `groupby` (2,4–2,7 ms), `tabla_woe` (19–21 ms), optbinning (2,5–3,0 ms con cortes fijos) | optbinning sin +0,5; mismo signo | numpy |
| Binning óptimo | PAV + fusiones (M14) | `OptimalBinning` (CP/MIP, 27–32 ms) | otro problema: elige cortes | optbinning en desarrollo, cortes congelados |
| Logística (β, SE) | IRLS con `solve` (M10, M11, M13, M23: 2–6 ms) | `sm.Logit` (20–27 ms), `sm.GLM` (28–260 ms), `LogisticRegression` (20–190 ms) | sklearn `C = 1` por defecto; `penalty` deprecado en 1.8 | statsmodels (inferencia) |
| Log-verosimilitud | $yz-\text{logaddexp}(0,z)$ | `scipy.special.log_expit`; `sklearn.metrics.log_loss` | la ingenua da −∞/NaN desde $|z|\ge 37$ (float64) | forma estable |
| δ de calibración | Newton 1-D (M15, M23) | `brentq`; `sm.GLM(offset=η)` | idénticos (0,330940 en OOT); aproximación 0,2865 | brentq o GLM (da SE) |
| AUC / Gini | rangos (0,4 ms), conteos (0,1 ms) (M12, M23) | `roc_auc_score` (3,3–4,6 ms), `mannwhitneyu` (2,0–2,3 ms) | orden de argumentos; U de la primera muestra | sklearn o conteos |
| Bootstrap | matriz de índices + `bincount` (M23: 13–19 ms con B = 200) | bucle + `roc_auc_score` (430–650 ms); `scipy.stats.bootstrap` (475–785 ms) | scipy: BCa, 9.999, `paired=False` por defecto | vectorizado en bloques |
| $X^\top WX$ | broadcasting + BLAS (0,09 ms) | `einsum` (0,47–0,49 ms; 0,15 con `optimize=True`) | `np.diag(w)`: $O(n^2)$ | broadcasting |
| Mínimos cuadrados | `lstsq`/QR | `scipy.linalg.cho_solve`, `solve_triangular` | QR pierde la mitad de dígitos que ecuaciones normales | QR si κ equilibrado > 10⁶ |
| PSI, KL, JS | numpy (M08) | `scipy.stats.entropy`, `jensenshannon` | `entropy` renormaliza; JS devuelve la distancia | numpy |
| χ² homogeneidad | numpy (M08) | `chi2_contingency` | Yates por defecto en 2×2 | `correction=False` |
| Binomial por banda | colas con `binom.sf/cdf` (M15) | `binomtest` | bilateral = suma de probabilidades ≤ P(k) | declarar `alternative` |
| Hosmer-Lemeshow | `array_split` del argsort (M23) | `diagnostic_gen.test_chisquare_binning` | df = g − 2 por defecto | df = g fuera de muestra o simulado |
| IC de proporción | Wilson cerrado (M15) | `proportion_confint` | Wald por defecto | Wilson/Jeffreys |
| VIF / condición | `diag(inv(R))`, SVD (M09) | `variance_inflation_factor` | la librería necesita constante | numpy |
| DeLong | propio (M12) | no existe en scipy/statsmodels/sklearn | — | propio con tests |
| Firth | propio (M10) | no en statsmodels/sklearn; R `logistf` | — | propio o R |
| Isotónica | PAV (M15) | `IsotonicRegression` | `increasing="auto"` puede invertir | sklearn con `increasing=True` |

**Regla que resume la tabla.** Donde existe una librería madura y la convención coincide con el expediente (colas de distribuciones, `brentq`, `roc_auc_score`, `sm.Logit`), se usa la librería y se testea contra una especificación numpy corta. Donde no existe (DeLong, Firth, bitácora de stepwise) o la convención no coincide (WoE con suavizado, intervalos `(a, b]`, especiales), manda el núcleo propio. En **producción**, nunca una librería de estimación: el artefacto congelado y un lookup.

---

## 8. Aplicación: casos y números

### 8.1 El experimento 2 de la clase 3 no era un MLE

`Scorecard(binning_process=proceso_8, estimator=LogisticRegression(max_iter=1000), ...)` en el lab de la clase 3 y en la demo de la clase 5 usa `C = 1`. Banco Austral tiene 3.322 filas y 165 malos en DEV (tasa 5%): la información por observación ($\bar w\approx 0{,}048$) es la mitad que en el generador del notebook ($\bar w\approx 0{,}10$), así que su contracción equivale a la del notebook con ~1.600 filas: del orden de 0,85–0,90 en la norma de las pendientes, más en las variables de bajo IV. Es una extrapolación con la fórmula $h_j\approx n\bar w\,\text{Var}(\text{WoE}_j)$, no una medición sobre Austral. Consecuencia: el «empate ±0,01» entre el embudo a mano (0,757/0,698/0,675) y optbinning (0,766/0,683/0,684) compara **dos cosas a la vez**: el binning óptimo y la penalización. Para aislar el efecto del binning hay que repetir con `LogisticRegression(C=np.inf)`. En el experimento 3 (66 variables, 165 malos: 2,5 malos por parámetro), la penalización probablemente **protegió** al modelo; el sobreajuste que la clase reportó (0,805 → 0,657) habría sido peor con un MLE. Regularizar no es el error; no declararlo sí.

### 8.2 El δ de la clase 4 por tres caminos

La clase 4 calculó δ = 0,177 con `brentq` vs 0,143 con la aproximación $\text{logit(TC)}-\text{logit}(\bar p)$. El notebook replica la estructura en OOT: GLM con offset, `brentq` y Newton numpy dan 0,330940 los tres; la aproximación, 0,2865 (subestima 13%; en Austral, 19%: 0,143 vs 0,177; mismo mecanismo, la convexidad de σ en la cola baja, M15). El GLM agrega el error estándar de δ: si el δ de recalibración que un gatillo dispara está dentro de 2 SE de cero, la recalibración no está justificada por los datos.

### 8.3 La clase 5: dos p-valores que dependen de la librería

- **Backtesting por banda.** `binomtest(malos, n, pd).pvalue` es bilateral. El semáforo 🔴/🟡 de la clase trata igual una banda que subestima y una que sobreestima el riesgo. Para el mandato típico (detectar subestimación), la prueba es unilateral `alternative='greater'`. Además, el bilateral de scipy no es «2 × cola»: en el ejemplo verificado (3 malos en 50 con PD 2%) da 0,078 contra 0,157, un factor 2 que puede cambiar el color de una banda entre 🟢 y 🟡.
- **Hosmer-Lemeshow OOT.** χ² = 29,0 con $g=10$: statsmodels (df = 8) da p = 0,0003 🔴; df = 10 da 0,0012 🔴; el p simulado de la clase, 0,012 🟡. **La misma evidencia cambia de color según la referencia**, y la clase eligió bien: el p simulado es el más honesto con esperados chicos (grupos 1–4 esperaban 0,1–1,2 malos).

### 8.4 Bootstrap B = 1.000 y el costo del monitoreo

Medido en el notebook con B = 200 y $n$ = 4.792 (OOT del generador): bucle con `roc_auc_score` 430–650 ms, bucle con AUC numpy 45–78 ms, vectorizado 13–19 ms, `scipy.stats.bootstrap` 475–785 ms. Extrapolando a la receta de la clase 5 (B = 1.000, OOT de Austral con 2.004 filas): ~1–2 s el bucle con sklearn (su costo por llamada tiene una parte fija de ~1 ms que no baja con $n$) vs ~0,03–0,04 s vectorizado. Para un tablero mensual con 8 segmentos × 3 métricas × 24 meses (576 bootstraps) son ~10–20 minutos vs ~20 segundos. Es una extrapolación; el costo real depende del número de valores únicos del score y de la máquina. El IC 95% del Gini OOT en el notebook, [0,516; 0,588], coincide exactamente entre el vectorizado y scipy: con la misma semilla, scipy 1.17 sortea la misma matriz de índices. Es un detalle de implementación, no un contrato.

### 8.5 Clase 6: el bug del binner y su primo, el borde de intervalo

El bug de la clase 6 (el binner re-ajustado con el lote del día cambia 587 de 8.585 decisiones, 6,8%) y la trampa T5 son la misma familia: **una convención de asignación a bins que no está congelada en el artefacto**. En el generador, cambiar `(a, b]` por `[a, b)` con los cortes congelados del curso mueve 22,7% de las filas de `meses_desde_mora_12m`, porque la variable es entera y los cortes caen sobre valores observados (−9: 1.951 filas; 3: 171; 8: 158). Ninguna línea falla; ninguna alerta salta. El test que lo detecta cuesta cinco líneas: puntuar x = cada corte en ambos motores.

En el scorecard de la demo de nikodym, el bin `Special` de `abonos_delta_12m` tiene WoE 0,0000 y 64 puntos. Puede ser un bin vacío (y el WoE 0 es entonces una política razonable) o el default `metric_special=0` aplicado a un bin con población: el expediente debe decir cuál.

### 8.6 Crédito de motos: cartera chica, motor sin Python

Una financiera de motos que lanza un producto nuevo (por ejemplo, motos eléctricas en una red de concesionarios) desarrolla con ~1.000 créditos y ~100 malos. Es exactamente el caso $n$ = 1.000, 101 malos del notebook: con `LogisticRegression()` por defecto, la pendiente de la variable de canal queda en 52% de su MLE, el score OOT se mueve hasta 6,2 puntos y la PD del percentil 95 baja de 34,4% a 33,3%. Si el cutoff de aprobación está en esa zona, el default de una librería decide aprobaciones. Con $n$ = 200 (un piloto en una región), la norma de las pendientes cae a 43% y el score cambia hasta 38,9 puntos.

El motor de originación en concesionarios suele ser un sistema transaccional (SQL, Java o un motor de reglas del proveedor), no Python. La sección E2 del notebook genera el `CASE` de SQL desde el núcleo numpy:

```sql
CASE
  WHEN uso_linea_prom_12m IS NULL THEN NULL  -- política de missing: declarar
  WHEN uso_linea_prom_12m <= 0.19672000000000006 THEN 98.7299
  WHEN uso_linea_prom_12m <= 0.31626000000000004 THEN 91.1564
  WHEN uso_linea_prom_12m <= 0.4488400000000001 THEN 84.7353
  WHEN uso_linea_prom_12m <= 0.60482 THEN 74.0472
  ELSE 61.5781
END AS pts_uso_linea_prom_12m
```

Los cortes aparecen con todos sus dígitos a propósito: con 10 dígitos significativos, 3 de 4 no vuelven al mismo float. Y el test de paridad incluye los cuatro valores x = corte, porque sobre los datos reales `<` y `<=` dan el mismo resultado.

---

## 9. Preguntas de comité

**P1. «¿Por qué el modelo que presentan no se ajustó con la misma herramienta que el modelo de la clase 3/la versión anterior?»**
*Respuesta modelo:* El estimador es el mismo (MLE logístico); cambió la implementación. Demostramos equivalencia con un test de paridad: statsmodels y nuestro IRLS coinciden a $10^{-12}$ en β y SE en 12 carteras de prueba y en la muestra de desarrollo. La versión anterior usaba `LogisticRegression` con `C = 1` (penalización L2 por defecto); la diferencia en β era de hasta X en la variable Y y la documentamos en el anexo de cambios.

**P2. «¿Quién validó optbinning?»**
*Respuesta modelo:* Nadie valida una librería en abstracto; validamos lo que usamos de ella. De optbinning usamos la **propuesta de cortes**; los cortes se congelan en el artefacto y el WoE, el scorecard y la transformación los calcula nuestro núcleo. Verificamos cinco convenciones de la versión 1.0.0 (signo del WoE, intervalos `[a, b)`, especiales con WoE 0 por defecto, fusión de especiales en lista, sin suavizado) y cómo las traducimos. La versión está fijada con hash; un test canario falla si un default cambia.

**P3. «¿El modelo en producción calcula lo mismo que el notebook?»**
*Respuesta modelo:* Sí, con evidencia: golden test de 20 filas con igualdad de enteros entre numpy y el motor; test de bordes en cada corte (x = corte, ± ε, NULL, códigos especiales); convención `(a, b]` traducida a `<=` y declarada en el artefacto; cortes serializados con ida y vuelta exacta. La tolerancia para la PD es $10^{-9}$ porque el motor trabaja en DOUBLE.

**P4. «¿Por qué no usan una sola librería que haga todo?»**
*Respuesta modelo:* Porque cada capa tiene requisitos distintos. El modelo que se firma necesita inferencia (statsmodels); el binning necesita optimización con restricciones (optbinning); producción necesita cero dependencias y portabilidad (tabla + suma). Una sola librería nos obligaría a aceptar sus convenciones en las tres capas o a llevar 34 dependencias al servidor de scoring.

**P5. «Su test binomial por banda, ¿es uni- o bilateral?»**
*Respuesta modelo:* El default de scipy es bilateral y además no es «2 × cola». Declaramos unilateral (`alternative='greater'`) porque el riesgo que el comité quiere controlar es la subestimación; reportamos aparte las bandas con sobreestimación material, como información de conservadurismo, sin semáforo rojo.

**P6. «¿Hay riesgo numérico en este modelo?»**
*Respuesta modelo:* Acotado y medido: el diseño WoE tiene número de condición 9,9 (pérdida de 2 de 16 dígitos en $X^\top WX$); los log-odds están en un rango donde float64 no satura; la log-verosimilitud se calcula en forma estable. El riesgo numérico relevante es de **implementación** (convenciones de borde, precisión y orden de suma del motor), y lo cubre la batería de paridad.

**P7. «¿Qué pasa si mañana sale scikit-learn 1.10?»**
*Respuesta modelo:* El modelo en producción no depende de scikit-learn. En desarrollo, las versiones están fijadas; la actualización entra por PR con la batería completa. Sabemos que 1.10 elimina el parámetro `penalty` (deprecado en 1.8); nuestro código no lo usa.

**P8. «nikodym genera el informe y el model card. ¿Eso es la validación?»**
*Respuesta modelo:* No. nikodym industrializa la evidencia (config con hash, contrato de datos, trail, lineage, model card), pero no emite el veredicto: el propio informe lo deja como campo «por completar» del validador. Y su motor numérico debe pasar la misma prueba que cualquier implementación: paridad contra una especificación independiente, como hizo la clase 6 comparando los Gini de la corrida a mano y de nikodym.

---

## 10. Ejercicios

**E1 (cálculo a mano).** En float64, ¿para qué log-odds $z$ la sigmoide ingenua devuelve exactamente 1? ¿Y en float32? ¿Qué PD mínima puede representar un sistema que guarda $P(\text{bueno})$ en float32 y calcula PD = 1 − P(bueno)?

<details><summary>Solución</summary>

$1+e^{-z}$ redondea a 1 cuando $e^{-z}$ es menor que la mitad del espaciado de los floats en 1, es decir $e^{-z}<u$. float64: $z>53\ln 2=36{,}74$; float32: $z>24\ln2=16{,}64$. Con $P(\text{bueno})$ en float32, el espaciado justo bajo 1 es $2^{-24}\approx 6\times10^{-8}$: la menor PD no nula representable como $1-P(\text{bueno})$ es $\approx 6\times10^{-8}$, y todas las PD entre 0 y ese valor se redondean a múltiplos de él. Una cartera prime con PD de $10^{-4}$ tendría solo ~1.700 niveles distintos de PD posibles; con $10^{-6}$, ~17. El notebook (A1 con float32) muestra la saturación en $z=17$.
</details>

**E2 (derivación).** Demuestre que $\ell_i = y_iz_i-\ln(1+e^{z_i})$ y que su gradiente respecto de $\beta$ es $x_i(y_i-\sigma(z_i))$. ¿Por qué el IRLS converge bien aunque la log-verosimilitud ingenua dé −∞?

<details><summary>Solución</summary>

Derivación en la sección 3.3. Gradiente: $\partial\ell_i/\partial z_i=y_i-\frac{e^{z_i}}{1+e^{z_i}}=y_i-\sigma(z_i)$, y por la regla de la cadena con $z_i=x_i^\top\beta$, $\nabla_\beta\ell_i=x_i(y_i-\sigma(z_i))$. El IRLS solo usa $\sigma(z)$ (acotada en [0, 1], nunca −∞) y $w=\sigma(1-\sigma)$. La log-verosimilitud aparece solo en el **reporte** (criterio de parada basado en devianza, AIC, LR). Si el criterio de parada usa la log-verosimilitud ingenua, un −∞ puede detener o desestabilizar el algoritmo; por eso el notebook para por $\max|d|$.
</details>

**E3 (cálculo).** Un scorecard tiene 10 variables WoE; la matriz $\tilde X=W^{1/2}X$ tiene $\kappa=40$. (a) ¿Cuántos dígitos pierde $X^\top WX$? (b) ¿Cuántos quedan en β resolviendo por Cholesky? (c) Un analista agrega una variable que es la suma de otras dos más ruido de desviación $10^{-7}$: $\kappa$ sube a $3\times10^{7}$. Repita (a) y (b), y diga qué harían QR y el filtro VIF del curso.

<details><summary>Solución</summary>

(a) $\kappa^2=1.600$: ~3,2 dígitos. (b) ~16 − 3,2 ≈ 12–13 dígitos: irrelevante. (c) $\kappa^2=9\times10^{14}$: se pierden ~15 dígitos; Cholesky deja ~1 dígito correcto (o falla por no definida positiva). QR sobre $\tilde X$ perdería ~7,5 dígitos y dejaría ~8 correctos (con residuo chico). Pero la respuesta correcta no es QR: el VIF de esa variable es del orden de $10^{14}$, muy sobre el umbral 5/10 del curso, y la variable no entra. La estabilidad numérica de un scorecard se compra en el diseño.
</details>

**E4 (cálculo).** Usando $h_j\approx n\,\bar w\,\text{Var}(\text{WoE}_j)$, estime la contracción de `LogisticRegression()` (C = 1) para una variable con desviación estándar de WoE 0,12 en (a) 1.000 créditos de motos con 10% de malos; (b) 20.000 créditos con 5% de malos. ¿Cuándo importa?

<details><summary>Solución</summary>

(a) $\bar w\approx \bar p(1-\bar p)\approx 0{,}09$ (aproximando $\bar w$ por $\bar p(1-\bar p)$; con PD heterogéneas $\bar w$ es algo menor); $h\approx 1.000\times0{,}09\times0{,}0144=1{,}30$; factor $1{,}30/2{,}30=0{,}56$: la pendiente queda en ~56% del MLE (el notebook mide 0,52 para `canal` con $h$ = 1,06). (b) $h\approx 20.000\times0{,}0475\times0{,}0144=13{,}7$; factor 0,93. Importa en carteras chicas, en variables de bajo IV y cuando el score está cerca del cutoff. Nunca importa «en promedio» porque la PD media en la muestra de ajuste es exacta (el intercepto no se penaliza).
</details>

**E5 (diseño).** Escriba el contrato de convenciones (formato 6.2) para `meses_desde_mora_12m` que evite T4 y T5 y el test que lo verifica.

<details><summary>Solución</summary>

```yaml
meses_desde_mora_12m:
  especiales: {-9: {bin: nunca_mora, woe: empirico}, -99: {bin: sin_bureau, woe: empirico}}
  moda: {13: {bin: sin_mora_12m}}
  cortes: [3.0, 8.0]          # sobre valores 1..12
  intervalos: "(a, b]"        # 1–3, 4–8, 9–12
  fuera_de_dominio: error      # 0, 14, -5 → rechazo en el contrato de datos
```
Test: `for x in [-99, -9, 1, 3, 4, 8, 9, 12, 13]: assert bin_nucleo(x) == bin_motor(x) == esperado[x]`, más `assert bin(-9) != bin(-99)` y un test que falle con 0 y 14. Con `[a, b)` el 3 caería en «4–8» y el test lo detecta; con lista de especiales, −9 y −99 compartirían bin y el assert de desigualdad lo detecta.
</details>

**E6 (código).** Implemente el bootstrap del AUC con remuestreo **multinomial** (sin matriz $B\times n$): para cada réplica, $\text{cnt}\sim\text{Multinomial}(n, \mathbf{1}/n)$, y las tablas $m_j$, $b_j$ salen de `bincount(codigos, weights=cnt*y)`. ¿Es la misma distribución que el bootstrap por índices? ¿Cuánta memoria usa?

<details><summary>Solución</summary>

```python
def bootstrap_auc_multinomial(cod, y, k, B, rng):
    n = len(y)
    out = np.empty(B)
    for r in range(B):                       # bucle en réplicas, pero O(n) vectorizado dentro
        cnt = rng.multinomial(n, np.full(n, 1.0 / n))
        m = np.bincount(cod, weights=cnt * y, minlength=k)
        t = np.bincount(cod, weights=cnt, minlength=k)
        b = t - m
        out[r] = np.sum(m * (np.cumsum(b) - b + 0.5 * b)) / (m.sum() * b.sum())
    return out
```
Sí: el número de veces que cada observación aparece en una réplica por índices tiene exactamente distribución Multinomial($n$, $1/n$); ambas definen el mismo bootstrap no paramétrico (las réplicas individuales difieren porque consumen el generador distinto). Memoria $O(n+k)$ por réplica en lugar de $O(Bn)$. Se puede vectorizar por bloques de réplicas con `rng.multinomial(n, p, size=bloque)`.
</details>

**E7 (análisis).** En el notebook, con la renta en pesos $\kappa(W^{1/2}X)=6{,}1\times10^6$, pero el IRLS por ecuaciones normales da las mismas predicciones que con la renta en millones. ¿Contradice la sección 3.6? Construya un ejemplo donde un $\kappa$ alto **sí** degrade la solución por `solve`.

<details><summary>Solución</summary>

No la contradice: la cota $\kappa^2u$ es de peor caso sobre todas las perturbaciones; el mal condicionamiento por escala de columnas es benigno para Cholesky/LU (van der Sluis), y lo relevante es el $\kappa$ del diseño equilibrado (9,3 en ambos). Ejemplo dañino: dos columnas **casi colineales en dirección**, no en escala, p. ej. $x_2=x_1+10^{-7}\,\varepsilon$ con $\varepsilon$ normal estándar. Equilibrar no ayuda porque el problema es el ángulo entre columnas, no su norma. El experimento A4 del notebook construye esa situación con valores singulares controlados.
</details>

**E8 (diseño de test).** El arnés de paridad del notebook usa tolerancia $10^{-8}$ para β. Proponga la tolerancia para comparar β de statsmodels contra `LogisticRegression(C=np.inf, tol=1e-10)` y justifíquela. ¿Qué tolerancia dejaría pasar la trampa `C = 1` con $n$ = 10.065?

<details><summary>Solución</summary>

L-BFGS para por norma del gradiente, no por tamaño del paso; con `tol=1e-10` el notebook observa $1{,}5\times10^{-6}$ de diferencia máxima. Una tolerancia defendible es $10^{-4}$ (dos órdenes sobre lo observado, cuatro bajo cualquier efecto material en puntos: $10^{-4}\times$ WoE máximo ~2 × factor 28,85 ≈ 0,006 puntos). La trampa `C = 1` con $n$ = 10.065 produce 0,089 de diferencia máxima: cualquier tolerancia ≥ 0,09 la deja pasar; una de $10^{-2}$ ya la detecta. Lección: la tolerancia se deriva del mecanismo (criterio de parada) y del impacto (puntos), no se copia.
</details>

**E9 (gobierno).** Redacte las tres filas del inventario de «herramientas del modelo» del expediente para statsmodels, optbinning y el núcleo numpy: uso, versión, convenciones relevantes, evidencia de verificación y dueño.

<details><summary>Solución</summary>

| Herramienta | Uso en el modelo | Versión | Convenciones relevantes | Evidencia | Dueño |
|---|---|---|---|---|---|
| statsmodels | ajuste MLE e inferencia (desarrollo) | 0.15.0 (hash en lockfile) | `Logit.fit` Newton, `maxiter=35` por defecto (se fija 200); `GLM.converged` no detecta separación | paridad con núcleo a $10^{-8}$; test de separación | modelador |
| optbinning | propuesta de cortes (desarrollo) | 1.0.0 | `[a, b)`; especiales con WoE 0 en `transform`; lista de especiales = un bin; sin suavizado | canarios de defaults; cortes congelados y re-tabulados por el núcleo | modelador |
| núcleo numpy | binning, WoE, score, PD, métricas de monitoreo (producción y validación) | git SHA | `(a, b]`, +0,5, especiales por código, `logaddexp` | golden tests, bordes, paridad con oráculos y con el motor SQL | modelador + TI (motor) |
</details>

---

## 11. Referencias

- **Higham, N. J. (2002).** *Accuracy and Stability of Numerical Algorithms*, 2.ª ed. SIAM. — La referencia para todo lo de las secciones 3.1–3.6: sumas (cap. 4), equilibrado (§7.3), mínimos cuadrados (cap. 20).
- **Golub, G. H. & Van Loan, C. F. (2013).** *Matrix Computations*, 4.ª ed. Johns Hopkins University Press. — Ecuaciones normales vs QR y sus cotas de error (cap. 5).
- **Goldberg, D. (1991).** «What every computer scientist should know about floating-point arithmetic». *ACM Computing Surveys* 23(1). — El modelo de punto flotante explicado para no especialistas.
- **van der Sluis, A. (1969).** «Condition numbers and equilibration of matrices». *Numerische Mathematik* 14. — Por qué un κ alto por escala de columnas es benigno (verificar páginas).
- **Blanchard, P., Higham, D. J. & Higham, N. J. (2021).** «Accurately computing the log-sum-exp and softmax functions». *IMA Journal of Numerical Analysis* 41(4). — Análisis de error de `logsumexp` y softmax.
- **Mächler, M. (2012).** «Accurately computing log(1 − exp(−|a|)), assessed by the Rmpfr package». Viñeta de CRAN. — Las variantes estables de `log1p`/`expm1` para probabilidades en las colas.
- **Kahan, W. (1965).** «Further remarks on reducing truncation errors». *Communications of the ACM* 8(1). — La suma compensada.
- **Harris, C. R. et al. (2020).** «Array programming with NumPy». *Nature* 585. — Diseño de numpy: vectorización, broadcasting, por qué los bucles van en C.
- **Virtanen, P. et al. (2020).** «SciPy 1.0: fundamental algorithms for scientific computing in Python». *Nature Methods* 17. — Qué garantiza scipy y cómo se prueba.
- **Seabold, S. & Perktold, J. (2010).** «Statsmodels: econometric and statistical modeling with Python». *Proceedings of the 9th Python in Science Conference*. — Filosofía de statsmodels (inferencia primero).
- **Pedregosa, F. et al. (2011).** «Scikit-learn: machine learning in Python». *Journal of Machine Learning Research* 12. — Filosofía de sklearn (predicción primero): contexto para leer sus defaults.
- **Navas-Palencia, G. (2020).** «Optimal binning: mathematical programming formulation». arXiv:2001.08025. — La formulación detrás de optbinning.
- **McCullagh, P. & Nelder, J. A. (1989).** *Generalized Linear Models*, 2.ª ed. Chapman & Hall. — IRLS, offset y devianza.
- **Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X. (2013).** *Applied Logistic Regression*, 3.ª ed. Wiley. — El test HL y sus grados de libertad.
- **Brown, L. D., Cai, T. T. & DasGupta, A. (2001).** «Interval estimation for a binomial proportion». *Statistical Science* 16(2). — Por qué no usar Wald.
- **Efron, B. & Tibshirani, R. J. (1993).** *An Introduction to the Bootstrap*. Chapman & Hall. — Percentil vs BCa (el default de scipy).
- **Firth, D. (1993).** «Bias reduction of maximum likelihood estimates». *Biometrika* 80(1). — La salida correcta ante separación (T7).
- **Siddiqi, N. (2017).** *Intelligent Credit Scoring*, 2.ª ed. Wiley. — El estándar de práctica del scorecard que las convenciones del curso siguen.
- **Board of Governors of the Federal Reserve System, OCC & FDIC (2026).** SR 26-2, *Model Risk Management Guidance* (17 de abril de 2026; reemplaza a SR 11-7 de 2011). — Modelos de proveedores y materialidad.
- **Prudential Regulation Authority (2023).** SS1/23, *Model risk management principles for banks* (vigente desde el 17 de mayo de 2024). — Aplica a modelos internos y de proveedores «independiente de la tecnología».
- **CVE-2019-6446** (NumPy < 1.16.3, `numpy.load` con pickle). — El caso de referencia de por qué un artefacto no debe ser un pickle.
- **Serie 1 · M5, E3, E6; Serie 2 · M08–M15, M21, M22.** — Pipeline declarativo, bootstrap, regulación; y la sección 7 de cada módulo, que esta tabla consolida.

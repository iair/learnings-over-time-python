# M15 · Calibración: nivel vs ranking

> **Ficha.** Profundiza: clase 4 v1 (láminas 11–19, 35) y v21 (láminas 20–33 y 49); clase 5 (lámina 4 y backtesting de la lámina 22). · Prerrequisitos: Serie 1 · E2 (tendencia central, PIT vs TTC conceptual), Serie 2 · M10 (logística sobre WoE), M13 (scaling). Prepara M16 (master scale), M17 (cutoff) y M19 (backtesting). · Archivos: `M15_calibracion.md` (este documento), `M15_calibracion.py` (notebook Marimo), `M15_calibracion_intercepto.xlsx` (calculadora de δ por grupos). · Tiempo estimado: 3 h de lectura + 2 h de notebook y ejercicios.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**El nivel en Banco Austral.** El scorecard de 8 variables (Gini 0,757 / 0,698 / 0,675) tiene intercepto β₀ = −3,03, que «es en la práctica el log-odds de la tasa de malos de DEV». La PD media del modelo por muestra fue: DEV 4,97% contra 4,97% observado («clava por construcción»), HO 5,17% vs 5,58% (−0,42 pp), OOT 5,19% vs 5,94% (−0,75 pp) y TTD 5,79% (aún no observable). El ranking sobrevive; el nivel queda anclado a DEV.

**La tendencia central (TC).** De la tabla de cosechas de la clase 1, 12 cosechas maduras (2024-07 a 2025-06) entre 3,98% y 6,97%. TC del curso = promedio simple 5,42% (ponderado 5,38%). El modelo «nació ~9% más optimista que el ciclo». La TC tiene dueño, se documenta, y usa también las cosechas de OOT: la mejora de OOT tras calibrar es «coherencia con el ancla, no validación independiente».

**El ajuste de intercepto (v1).** $\text{PD}^{cal}_i=\sigma(\text{logit}\,\text{PD}_i+\delta)$. Siddiqi: δ ≈ logit(TC) − logit(PD̄_DEV) = +0,092; exacto (raíz de la media, `brentq`): +0,111 (la clase 5 lo reporta como 0,1115). «La media de la sigmoide no es la sigmoide de la media (Jensen)». Después del δ: DEV 5,42% (+0,45 pp, intencional), HO 5,63% (+0,05), OOT 5,65% (−0,29), TTD 6,31%. El Gini queda intacto.

**La calibración PIT (v21).** Muestra reciente y madura: las 4 últimas cosechas (mar–jun 2025, n = 2.004, 119 malos, 5,94%). PD media 5,19% → δ aprox +0,143 vs exacto +0,177 (el usado). Resultado: DEV 5,70% (+0,74 pp), HO 5,92% (+0,34), OOT 5,94% (0,00 por construcción). Brecha absoluta media mensual 0,67 pp. Curva de 10 grupos: el grupo 5 de OOT tiene 1,18% predicho y 4,00% observado (n = 200). Lo que la calibración PIT no logra: corregir pendiente, recuperar señal, probar estabilidad fuera de muestra. «La mejora en OOT no es validación independiente, porque OOT fue usado para calibrar.»

**Clase 5, lámina 4.** El curso ancla al ciclo (TTC, δ = 0,1115); PIT daría δ = 0,177. En el cutoff 560 se aprueba 78,3% con TC y 77,2% con PIT: 1,9 puntos de score para todos. Si se calibra contra OOT, el binomial en OOT da p = 1,00 por construcción; con TC contesta 0,56. Regla: **la muestra con que se calibra y la que valida no pueden ser la misma**, y ambas se declaran.

**Lo que el curso simplificó, omitió o dejó como convención:**

1. *Por qué* el MLE clava la media: se afirmó, no se derivó; tampoco cuándo deja de valer (penalización, pesos).
2. El error de Siddiqi se atribuyó a Jensen «con PD dispersas», sin cuantificarlo. Aquí se demuestra que el error relativo es exactamente un promedio de $\kappa=\operatorname{Var}(p)/[\bar p(1-\bar p)]$.
3. Sobremuestreo / *prior shift* (King & Zeng): no se trató.
4. Solo se usó el ajuste de intercepto. Quedaron fuera la recalibración logística con pendiente, Platt, la isotónica y la recalibración por bandas.
5. Métricas: la curva se leyó «a ojo entrenado». La clase 5 agregó binomial y Hosmer-Lemeshow (M19), pero no CITL en escala logit, pendiente de calibración, Brier y su descomposición, log-loss, ECE ni IC de Wilson/Jeffreys.
6. La jerarquía de calibración (media, débil, moderada, fuerte).
7. Cómo documentar operativamente dos calibraciones sobre el mismo ranking y qué pasa con el cutoff.
8. La elección TC = promedio simple de 12 cosechas es una **convención** declarada, no una *long-run average* regulatoria (la EBA pide más de 5 años cuando haga falta para cubrir la variabilidad, con años buenos y malos; ver §4).

---

## 2. Intuición

**Dos perillas y una forma.** Escribe la PD de un modelo logístico como $\text{logit}\,p_i=\eta_i$. Todo lo que hace la calibración paramétrica cabe en $\text{logit}\,p^*_i=a+b\,\eta_i$:

- **a (nivel)** desplaza todas las log-odds lo mismo. No cambia el orden, así que el Gini es idéntico. Es la perilla del ajuste de intercepto.
- **b (confianza)** estira o comprime las log-odds alrededor de un punto. Con $b<1$ el modelo «grita demasiado»: sus PD extremas son demasiado extremas, el síntoma clásico del sobreajuste. Con $b>1$ es tímido. Tampoco cambia el orden si $b>0$.
- La **forma** (curvatura de la curva de calibración) es lo que ninguna de las dos perillas arregla. Para eso están los métodos no paramétricos (isotónica, bandas), que pagan en varianza y en pérdida de la escala PDO.

Un termómetro es la analogía exacta: *offset* (a), *ganancia* (b), no linealidad del sensor (forma). Ordenar bien es necesario para que exista la calibración (un score aleatorio calibrado da la tasa media a todos), pero no la garantiza.

**Por qué la media no se traslada con el logit.** La sigmoide es convexa bajo 0 (donde viven casi todas las PD de consumo) y cóncava sobre 0. Sumar δ al logit sube mucho la PD de los clientes riesgosos (donde σ es empinada) y casi nada la de los clientes muy buenos (donde σ es plana en términos absolutos). La media sube **menos** que lo que dice «sumar δ al logit de la media». Siddiqi razona con un cliente representativo; la cartera no es un cliente. Mientras más dispersas las PD, más se equivoca. Y las PD son más dispersas mientras **mejor discrimina** el modelo: la aproximación falla más en los modelos buenos.

**Sobremuestreo.** Si se arma DEV con 40% de malos cuando la población tiene 11%, el modelo aprende bien *qué distingue* a un malo (las distribuciones de X dentro de cada clase no cambian) pero aprende mal *cuántos malos hay*. Esa información vive completa en el intercepto, y Bayes dice exactamente cuánto moverlo.

**Calibrar es estimar.** δ es un parámetro estimado con una muestra. Si después se evalúa el nivel con esa misma muestra, la pregunta «¿coincide la PD media con lo observado?» está respondida por construcción. Si se evalúa en otra muestra, el test tiene que cargar con el error de estimación de δ además del ruido de la muestra de validación.

---

## 3. Formalización

Notación: $y_i\in\{0,1\}$ (1 = malo), $\eta_i=x_i^\top\beta$ el logit del modelo, $p_i=\sigma(\eta_i)=1/(1+e^{-\eta_i})$, $\bar p=\frac1n\sum p_i$, $T$ la tasa objetivo (TC o tasa PIT). Recordatorios: $\sigma'(z)=\sigma(z)(1-\sigma(z))$ y $\sigma''(z)=\sigma(z)(1-\sigma(z))(1-2\sigma(z))$.

### 3.1 Por qué el intercepto de máxima verosimilitud clava la media

La log-verosimilitud logística es

$$\ell(\beta)=\sum_{i=1}^n\big[y_i\,\eta_i-\ln(1+e^{\eta_i})\big],\qquad \eta_i=\beta_0+\sum_j\beta_j x_{ij}.$$

Derivando respecto de $\beta_j$ y usando $\frac{d}{d\eta}\ln(1+e^\eta)=\sigma(\eta)$:

$$\frac{\partial\ell}{\partial\beta_j}=\sum_i\big(y_i-p_i\big)\,x_{ij}.$$

Para el intercepto $x_{i0}\equiv1$, así que en el MLE $\hat\beta$:

$$\sum_i(y_i-\hat p_i)=0\quad\Longleftrightarrow\quad\frac1n\sum_i\hat p_i=\frac1n\sum_i y_i .$$

Es una **ecuación de momentos**, no una propiedad del modelo. Consecuencias:

- Las otras ecuaciones dicen $\sum_i x_{ij}(y_i-\hat p_i)=0$: la PD también clava la media ponderada por cada WoE. Si una variable entra como dummies, la PD clava la tasa de cada categoría en DEV.
- **Con pesos** $w_i$ la ecuación es $\sum_i w_i(y_i-\hat p_i)=0$: clava la media ponderada. En el notebook, ponderar buenos ×2 deja la media simple de la PD en 6,38% contra 11,28% observado, y la ponderada en 5,972% contra 5,975%.
- **Con penalización sobre el intercepto** la ecuación se rompe. `liblinear` de scikit-learn trata el intercepto como una columna más y lo penaliza. Con C = 0,01 la PD media en DEV queda en 13,09% contra 11,28% (+1,8 pp). `lbfgs` con L2 no penaliza el intercepto y conserva la media (±0,004 pp, tolerancia del optimizador).
- La media clava la media **de la muestra**, con su ruido incluido. En el generador, la PD real media de DEV es 11,80% y lo observado 11,28% (−0,52 pp, ≈ 1,6 errores estándar): el modelo hereda ese ruido en su intercepto.

### 3.2 El ajuste de intercepto y su solución exacta

Se busca $\delta$ tal que la PD media ajustada en la muestra de calibración sea $T$:

$$f(\delta)=\frac1n\sum_i\sigma(\eta_i+\delta)-T=0.$$

$f$ es continua y estrictamente creciente, porque $f'(\delta)=\frac1n\sum_i\sigma(\eta_i+\delta)(1-\sigma(\eta_i+\delta))>0$. Además $f(-\infty)=-T<0$ y $f(+\infty)=1-T>0$. Hay **raíz única** y bisección o Brent la encuentran siempre en un intervalo como $[-10,10]$. La bisección necesita $\lceil\log_2(20/\text{tol})\rceil$ iteraciones (45 para tol = 10⁻¹²). Brent combina bisección con interpolación y converge superlinealmente.

**Equivalencia con un GLM con offset.** Si $T=\bar y$ de la misma muestra, la ecuación es $\sum_i(y_i-\sigma(\eta_i+\delta))=0$: la ecuación de primer orden de una logística con **solo intercepto** y $\eta_i$ como *offset*. Calibrar al observado = MLE del intercepto con offset. El notebook lo verifica: δ_PIT = 0,40684 por `brentq`, por IRLS numpy y por `statsmodels.GLM(offset=η)`.

La **aproximación de Siddiqi** reemplaza la cartera por un cliente con PD $\bar p$:

$$\delta_S=\text{logit}(T)-\text{logit}(\bar p).$$

### 3.3 El error de Siddiqi: una identidad exacta

Define la log-odds de la media como función del desplazamiento:

$$g(\delta)=\text{logit}\big(m(\delta)\big),\qquad m(\delta)=\frac1n\sum_i\sigma(\eta_i+\delta).$$

El δ exacto cumple $g(\delta^{\star})=\text{logit}(T)$ y Siddiqi calcula $\delta_S=\text{logit}(T)-g(0)=g(\delta^{\star})-g(0)$. Por la regla de la cadena, con $\frac{d}{dm}\text{logit}(m)=\frac{1}{m(1-m)}$:

$$g'(\delta)=\frac{m'(\delta)}{m(1-m)}=\frac{\frac1n\sum_i p_i(\delta)(1-p_i(\delta))}{m(1-m)}.$$

El numerador se descompone usando $\frac1n\sum p_i^2=\operatorname{Var}(p)+m^2$ (varianza poblacional, ddof = 0):

$$\frac1n\sum_i p_i(1-p_i)=m-\frac1n\sum_i p_i^2=m-m^2-\operatorname{Var}(p)=m(1-m)-\operatorname{Var}(p).$$

Por lo tanto

$$\boxed{\,g'(\delta)=1-\kappa(\delta),\qquad \kappa(\delta)=\frac{\operatorname{Var}\big(p(\delta)\big)}{m(\delta)\big(1-m(\delta)\big)}\,}$$

Como $0\le p_i\le1$, $\operatorname{Var}(p)\le m(1-m)$ (la varianza de una variable en [0,1] con media $m$ no supera la de una Bernoulli con esa media). Entonces $\kappa\in[0,1]$, y $\kappa<1$ salvo que todas las PD sean 0 o 1. Integrando de 0 a $\delta^{\star}$:

$$\delta_S=\int_0^{\delta^{\star}}g'(u)\,du=\delta^{\star}\,(1-\bar\kappa),\qquad \bar\kappa=\frac{1}{\delta^{\star}}\int_0^{\delta^{\star}}\kappa(u)\,du.$$

**Lecturas:**

1. **Siddiqi siempre se queda corto**: mismo signo que el exacto y magnitud menor, en la proporción $1-\bar\kappa$. No es un error aleatorio, es un sesgo hacia cero.
2. El **error relativo** $1-\delta_S/\delta^{\star}$ es exactamente $\bar\kappa$, un promedio de $\kappa$ en el camino. Por el teorema del valor medio queda entre el mínimo y el máximo de $\kappa$ en $[0,\delta^{\star}]$.
3. **Newton en un paso**: $\delta_N=\delta_S/(1-\kappa(0))$ usa solo $\bar p$ y $\operatorname{Var}(p)$. En Banco Sintético: TTC 0,1240 vs exacto 0,1245 (Siddiqi 0,1092); PIT 0,4020 vs 0,4068 (Siddiqi 0,3542).
4. **κ es discriminación.** Si el modelo está calibrado en sentido fuerte ($E[y\mid p]=p$), entonces $E[p\mid y=1]=E[p^2]/\bar p$ y $E[p\mid y=0]=(\bar p-E[p^2])/(1-\bar p)$, y su diferencia es

$$D_{\text{Tjur}}=\frac{E[p^2]}{\bar p}-\frac{\bar p-E[p^2]}{1-\bar p}=\frac{E[p^2]-\bar p^2}{\bar p(1-\bar p)}=\kappa .$$

  κ es el **coeficiente de discriminación de Tjur** (2009) y también el cociente resolución/incertidumbre de Murphy para un pronóstico calibrado. En DEV (calibrado solo en la media, no en sentido fuerte) el notebook da κ = 0,1199 y Tjur = 0,1241. *Mientras mejor discrimina el modelo, peor funciona Siddiqi.*
5. **Contraste con Austral.** 0,0914/0,1115 implica $\bar\kappa\approx0{,}18$ (TTC) y 0,143/0,177 implica $\bar\kappa\approx0{,}19$ (PIT). En Banco Sintético (Gini ~0,55) κ ≈ 0,12 y el error es ~12–13%. El juguete del notebook con logit-normal de desviación $s=1{,}8$ reproduce Austral PIT casi al decimal: $\bar p=5{,}19\%\to T=5{,}94\%$ da δ_S = 0,1429 y δ = 0,1783.

**Taylor de segundo orden (para intuición).** Si $\eta\sim(\mu,s^2)$, expandiendo $\sigma$ alrededor de $\mu$:

$$\bar p\approx\sigma(\mu)+\tfrac12 s^2\,\sigma''(\mu)=\sigma(\mu)+\tfrac12 s^2\,\sigma(\mu)(1-\sigma(\mu))(1-2\sigma(\mu)).$$

Para $\sigma(\mu)<\tfrac12$, $\bar p>\sigma(\mu)$: la PD del logit medio subestima la PD media (Jensen en la región convexa). A primer orden $\operatorname{Var}(p)\approx[\sigma'(\mu)]^2s^2$, de donde $\kappa\approx\bar p(1-\bar p)\,s^2$. Ambas expansiones sirven para $s\lesssim0{,}7$. Con $s=1{,}8$ la Taylor da $\bar p\approx4{,}0\%$ cuando el verdadero es 5,19%, y $\kappa\approx0{,}154$ cuando el verdadero es 0,193. Para carteras reales ($s>1$) usa κ calculado, no la expansión.

**¿Cuándo importa?** El error absoluto en δ es $\bar\kappa\,\delta^{\star}$. En Austral TTC eso es 0,02 en logit, 0,6 puntos de score y ~0,1 pp de PD media: irrelevante para una decisión y relevante para una auditoría («la media calibrada no calza con la TC declarada»). En Banco Sintético PIT es 0,053 en logit (1,5 puntos). Importa cuando δ es grande (recalibraciones fuertes, corrección de sobremuestreo) o κ es alto (modelos muy discriminantes, PD de comportamiento). No hay razón para usar la aproximación: la solución exacta cuesta una línea.

### 3.4 Sobremuestreo y *prior shift*: la corrección de King & Zeng desde Bayes

Supuesto clave (muestreo por clase, o *case-control*): la muestra se arma eligiendo malos y buenos con probabilidades que dependen **solo de y**. Entonces las densidades condicionales $f(x\mid y)$ son iguales en la muestra y en la población, y solo cambian las proporciones: $\rho_1$ en la muestra, $\pi_1$ en la población. Por Bayes, en cada una:

$$P_{\text{mues}}(y=1\mid x)=\frac{\rho_1 f(x\mid1)}{\rho_1 f(x\mid1)+\rho_0 f(x\mid0)}\;\Rightarrow\;\text{odds}_{\text{mues}}(x)=\frac{\rho_1}{\rho_0}\cdot\frac{f(x\mid1)}{f(x\mid0)},$$

$$\text{odds}_{\text{pob}}(x)=\frac{\pi_1}{\pi_0}\cdot\frac{f(x\mid1)}{f(x\mid0)}.$$

Dividiendo, la razón de verosimilitud $f(x\mid1)/f(x\mid0)$ se cancela:

$$\text{logit}\,p_{\text{pob}}(x)=\text{logit}\,p_{\text{mues}}(x)+\underbrace{\ln\!\Big[\frac{\pi_1}{\pi_0}\cdot\frac{\rho_0}{\rho_1}\Big]}_{\delta_{KZ}}.$$

Es un **ajuste de intercepto con δ analítico** y no depende de x: el sobremuestreo no altera los β de pendiente (en el modelo correcto), solo $\beta_0$. King y Zeng (2001) llaman a esto *prior correction* y lo comparan con *weighting* (maximizar la verosimilitud ponderada con $w_1=\pi_1/\rho_1$, $w_0=\pi_0/\rho_0$). Con el modelo bien especificado la corrección a priori es más eficiente. Con mal especificación, la ponderación es más robusta porque estima el mejor modelo *para la población*.

Tres observaciones útiles:

- **El WoE es invariante al submuestreo aleatorio de buenos**: $\text{WoE}_b=\ln(\%\text{buenos}_b/\%\text{malos}_b)$ usa distribuciones *dentro* de cada clase, que el muestreo por clase no cambia (salvo ruido). Todo el prior vive en $\beta_0$.
- **Equivalencia con el ajuste de intercepto.** Si el modelo fuese perfecto, $\delta_{KZ}$ y el δ exacto que lleva la media poblacional a $\pi_1$ coincidirían en el límite. Con n finito y mal especificación difieren. En el notebook, con $\rho_1=40\%$: δ_KZ = −1,658 y δ exacto = −1,698. La PD media con δ_KZ queda en 11,63% contra π₁ = 11,28%; sin corregir, en 34,2%. El intercepto de la submuestra más δ_KZ da −2,058, contra −2,063 del modelo en DEV completo.
- **El mismo álgebra vale en el tiempo** si el cambio es de prior puro (*label shift*: $f(x\mid y)$ estable, cambia $\pi_1$). En ese caso un δ constante es la corrección **exacta** del logit. Si la tasa de malos sube porque cambió la mezcla de X (*covariate shift*), el modelo ya lo captura y no hace falta δ. Si lo que cambió es la relación $x\to y$ (*concept shift*), un δ constante no alcanza. Saerens, Latinne y Decaestecker (2002) estiman el nuevo prior con EM cuando no se observa. El generador de la serie planta el caso limpio: `deterioro` suma 0,35 al logit verdadero de todas las cohortes 2025.

### 3.5 Recalibración logística (intercepto + pendiente) y Platt

Modelo: $y_i\sim\text{Bernoulli}(\sigma(a+b\,\eta_i))$, con $\eta_i$ fijo (el score del modelo congelado). Es una logística con un regresor. Newton-Raphson, que para el enlace canónico coincide con IRLS:

$$\theta^{(t+1)}=\theta^{(t)}+(Z^\top W Z)^{-1}Z^\top(y-p),\quad Z=[\mathbf 1,\eta],\ W=\operatorname{diag}(p_i(1-p_i)).$$

- $b$ es la **pendiente de calibración** (Cox 1958). En validación, $b<1$ indica sobreajuste (PD demasiado extremas) y $b>1$ PD comprimidas. En el notebook: b = 1,059 (SE 0,055) en la ventana PIT, 1,166 en HO, y la verdad del generador da 1,103. El modelo es algo tímido (bins gruesos, variables omitidas), pero con ~500 malos un desvío de 6–10% en la pendiente **no es significativo** (z ≈ 1,1). La hipótesis por defecto debe ser $b=1$.
- **CITL en escala logit** (Van Calster): el $a$ del modelo con $b\equiv1$, o sea el δ con offset de §3.2. Mide el desnivel medio en log-odds.
- **Platt (1999)** ajusta $\sigma(c+d\cdot s_i)$ sobre el score $s_i$. Como $s_i=\text{offset}-\text{factor}\cdot\eta_i$, se tiene $c+d\,s_i=(c+d\cdot\text{offset})-d\cdot\text{factor}\cdot\eta_i$. Es la misma familia con $a=c+d\cdot\text{offset}$ y $b=-d\cdot\text{factor}$ (verificado en el notebook). Ojo: `CalibratedClassifierCV(method="sigmoid")` de scikit-learn usa los objetivos suavizados de Platt, $(N_+ +1)/(N_+ +2)$ y $1/(N_- +2)$, en vez de 0/1. No reproduce exactamente el MLE ni clava la media.
- *Temperature scaling* (Guo et al. 2017) es el caso $a=0$, solo $b$. Tiene poco sentido en crédito, porque el nivel es lo primero que se mueve.

**Compatibilidad con el scorecard.** Con (a, b) el score calibrado es $s^{\star}=\text{offset}-\text{factor}(a+b\eta)$, afín en el score original. Se puede re-escalar la tabla de puntos (multiplicar todos los puntos por $b$ y ajustar la base) y mantener la escala PDO. Con $b\neq1$, «20 puntos duplican las odds» vale en la escala *nueva*, no en la original.

### 3.6 Isotónica (PAV) y recalibración por bandas

**Isotónica.** Resuelve $\min_{m\ \text{no decreciente}}\sum_i(y_i-m(\eta_i))^2$. El algoritmo PAV recorre los puntos ordenados por η, agregando primero los empates en η, y **fusiona bloques adyacentes** mientras violen la monotonía, reemplazándolos por su media ponderada. Propiedades:

1. En la muestra de ajuste, cada bloque tiene PD igual a su tasa observada. Por eso $\sum\hat m=\sum y$: la isotónica también clava la media in-sample, y la curva de calibración in-sample es perfecta por construcción.
2. La salida es **escalonada**: en el notebook, 1.831 PD distintas se colapsan en 28 escalones.
3. Los extremos pueden quedar en **0 y 1**: bloques con pocos casos y ningún o todos malos. Para PD regulatoria o pricing se necesitan piso y techo. Referencias de piso: 0,03% en Basilea II; según entiendo, 0,05% para minoristas en Basilea III final (verificar con la norma aplicable).
4. **Empates y AUC.** Un par mal ordenado que pasa a empate sube de 0 a ½; uno bien ordenado baja de 1 a ½. In-sample el AUC puede **subir** (en el notebook el Gini PIT pasa de 0,531 a 0,543, sello de sobreajuste). Fuera de muestra tiende a bajar (TTD: 0,5496 → 0,5482).
5. **Rompe la escala PDO**: el mapa score → log-odds deja de ser una recta de pendiente −1/factor.

**Recalibración por bandas** (*histogram binning*, Zadrozny & Elkan 2001). La PD de cada banda es su tasa observada en la muestra de calibración. Es lo que hace una master scale cuando asigna la «tasa observada suavizada» a cada banda (M16). Requiere monotonía entre bandas (si no, se funden bandas, igual que en PAV) y suficientes malos por banda.

**Varianza vs sesgo.** Con verdad conocida, el notebook mide $\text{RMSE}_{verdad}=\sqrt{\frac1n\sum(\hat p_i-p_i^{real})^2}$ en TTD para calibraciones estimadas con $n_{cal}=600$ (25 réplicas): δ 0,0764; logística a+b 0,0764; isotónica 0,0894; bandas 0,0934; sin calibrar 0,0964. Con n chico, los métodos flexibles pagan varianza. El piso ~0,073 que ninguno baja es la falta de resolución del modelo base: **calibrar no agrega información**.

### 3.7 Métricas de calibración

Para la muestra de validación ($n$ créditos):

| Métrica | Definición | Ideal | Qué mide |
|---|---|---|---|
| CITL (diferencia) | $\bar y-\bar p$ | 0 | nivel, en pp |
| CITL (logit) | $a$ en $\text{logit}\,P(y{=}1)=a+\eta$ | 0 | nivel, en log-odds (= δ que faltó) |
| O/E | $\sum y_i/\sum p_i$ | 1 | nivel, relativo |
| Pendiente | $b$ en $a+b\eta$ | 1 | confianza (dispersión de las PD) |
| Brier | $\frac1n\sum(p_i-y_i)^2$ | ↓ | exactitud total (calibración + discriminación) |
| Log-loss | $-\frac1n\sum[y\ln p+(1-y)\ln(1-p)]$ | ↓ | igual, castiga fuerte las PD cerca de 0 o 1 |
| ECE | $\sum_k\frac{n_k}{n}\lvert\bar y_k-\bar p_k\rvert$ | 0 | desvío medio de la curva (depende de K) |

**Descomposición de Murphy (1973) con grupos.** Particiona los créditos en K grupos (deciles de PD). Sean $\bar p_k$ y $\bar y_k$ las medias del grupo. Escribe $p_i-y_i=(p_i-\bar p_k)+(\bar p_k-\bar y_k)+(\bar y_k-y_i)$ y eleva al cuadrado, sumando dentro del grupo $k$:

$$\sum_{i\in k}(p_i-y_i)^2=\sum_{i\in k}(p_i-\bar p_k)^2+n_k(\bar p_k-\bar y_k)^2+\sum_{i\in k}(\bar y_k-y_i)^2+2\sum_{i\in k}(p_i-\bar p_k)(\bar y_k-y_i),$$

porque los cruces con el término constante $(\bar p_k-\bar y_k)$ se anulan: $\sum_{i\in k}(p_i-\bar p_k)=0$ y $\sum_{i\in k}(\bar y_k-y_i)=0$. Como $y_i\in\{0,1\}$, $\sum_{i\in k}(y_i-\bar y_k)^2=n_k\bar y_k(1-\bar y_k)$. Además

$$\sum_k n_k\bar y_k(1-\bar y_k)=n\,\bar y(1-\bar y)-\sum_k n_k(\bar y_k-\bar y)^2$$

(descomposición de la varianza de $y$ en *entre* y *dentro* de grupos). Dividiendo por $n$:

$$\text{BS}=\underbrace{\sum_k\tfrac{n_k}{n}(\bar p_k-\bar y_k)^2}_{\text{REL}}-\underbrace{\sum_k\tfrac{n_k}{n}(\bar y_k-\bar y)^2}_{\text{RES}}+\underbrace{\bar y(1-\bar y)}_{\text{UNC}}+\underbrace{\tfrac1n\sum_k\sum_{i\in k}(p_i-\bar p_k)^2}_{\text{WBV}}-2\,\underbrace{\tfrac1n\sum_k\sum_{i\in k}(p_i-\bar p_k)(y_i-\bar y_k)}_{\text{WBC}}.$$

Los dos últimos términos (varianza y covarianza **dentro** de los grupos) son los «dos componentes extra» de Stephenson, Coelho y Jolliffe (2008). Desaparecen si la PD toma exactamente K valores (bandas de master scale). REL es calibración (↓), RES es resolución o discriminación (↑) y UNC es la dificultad intrínseca, que no depende del modelo.

Lecturas en el notebook (TTD, oráculo, tasa 16,2%): el Brier va de 0,1159 (sin calibrar) a 0,1128 (δ PIT), mientras UNC ≈ 0,136 domina. **Brier solo es una mala métrica de calibración**: el cambio relevante está en REL (0,0035 → 0,0004) y en O/E (1,36 → 1,005). RES es idéntica (0,0225) en todos los métodos monótonos paramétricos: la discriminación no se compra calibrando.

**Relación con la jerarquía de Van Calster et al. (2016):**

1. *Media* (calibration-in-the-large): $\bar p=\bar y$. El δ la garantiza en la muestra de calibración.
2. *Débil*: $a=0$ y $b=1$. Requiere al menos la recalibración logística, o que el modelo base tenga pendiente 1.
3. *Moderada*: $E[y\mid p]=p$ para todo $p$ (la curva sobre la diagonal). Es lo que se mira con la curva de 10 grupos y lo que testean binomial por banda y Hosmer-Lemeshow (M19).
4. *Fuerte*: calibración para cada patrón de covariables. Los autores la llaman utópica. En crédito equivale a pedir calibración en cada segmento cruzado (canal × producto × plazo), y es la razón práctica para testear calibración por segmento relevante (M20).

Los mismos autores sostienen que la calibración moderada es el objetivo realista y que perseguir la fuerte con n finito lleva a sobreajuste.

### 3.8 Intervalos para una tasa observada: Wilson y Jeffreys

**Wilson (1927).** Invierte el test score $|\hat p-p|\le z\sqrt{p(1-p)/n}$. Elevando al cuadrado queda una cuadrática en $p$, cuya solución es

$$p\in\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}.$$

A diferencia de Wald ($\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$), no colapsa a un punto con 0 malos y su cobertura es cercana a la nominal con $np$ chico (Brown, Cai y DasGupta 2001).

**Jeffreys.** Cuantiles $\alpha/2$ y $1-\alpha/2$ de la posterior $\text{Beta}(k+\tfrac12,\,n-k+\tfrac12)$ bajo el prior de Jeffreys. Brown et al. recomiendan fijar el límite inferior en 0 cuando $k=0$ (y el superior en 1 cuando $k=n$); `statsmodels` no hace ese ajuste.

**El grupo 5 de OOT de Austral** (8 malos en 200; PD PIT 1,18% ⇒ 2,36 esperados):

- Wilson: $\hat p=0{,}04$, $z^2=3{,}8415$, denominador $1{,}01921$, centro $(0{,}04+0{,}009604)/1{,}01921=0{,}04867$, semiancho $1{,}92306\cdot\sqrt{0{,}000192+0{,}0000240}=0{,}02826$ ⇒ **[2,04%; 7,69%]**. No contiene 1,18%.
- Jeffreys: [1,91%; 7,41%].
- Binomial de una cola: $P(X\ge8\mid n=200,p=0{,}0118)=0{,}0028$. Supera incluso Bonferroni por 10 grupos (0,005).

**Pero**: (i) el grupo se destacó *porque* se veía raro, y Bonferroni es lo mínimo cuando se elige el peor de 10; (ii) la partición es un grado de libertad del analista: con las 8 bandas de la clase 5 la misma zona es B2 (8 malos en 235 a 1,42%), p = 0,020, amarillo; (iii) el binomial supone independencia y con correlación de defaults el p real es mayor (M19). La lectura defendible: evidencia de subestimación en bandas buenas, coherente con el patrón A2/B2 de la clase 5, que se vigila y se testea como pendiente, **no** se corrige a mano en un grupo.

### 3.9 Efecto de δ sobre el score y el cutoff

Con el scaling del curso, $s=\text{offset}+\text{factor}\cdot\ln\frac{1-p}{p}=\text{offset}-\text{factor}\cdot\eta$. Sumar δ al logit da

$$s^{cal}=\text{offset}-\text{factor}(\eta+\delta)=s-\delta\cdot\text{factor}.$$

Todos los clientes se mueven **los mismos** $\delta\cdot28{,}85$ puntos. En Austral: TTC 0,1115 → −3,22 pts; PIT 0,177 → −5,11 pts; diferencia 1,89 pts (el «1,9» de la clase 5). En Banco Sintético: −3,6 (TTC) y −11,7 (PIT). Hay dos formas coherentes de gobernarlo (M13):

- **Cutoff en score, tabla sin re-escalar**: la aprobación no cambia, cambia la PD que se promete en el corte y en la cartera aprobada. En el notebook, con cutoff 530 se aprueba 79,6% en los tres casos. La PD media prometida de aprobados es 7,05% (sin calibrar), 7,88% (TTC) y 10,10% (PIT); la verdad es 10,00%.
- **Cutoff en PD (apetito)**: el score de corte se desplaza δ·factor. Con apetito PD ≤ 18,45%, la aprobación cae de 79,6% a 76,5% (TTC) y 69,4% (PIT).

En Austral la diferencia entre TTC y PIT fue de 1,1 pp de aprobación (78,3% vs 77,2%) porque δ_PIT − δ_TTC = 0,066. En Banco Sintético es de 0,28 y la decisión cambia mucho más. **La elección de calibración es una decisión de negocio cuando el apetito está escrito en PD.**

### 3.10 PIT vs TTC operativo

(Concepto en Serie 1 · E2.) Operativamente:

| | PIT | TTC |
|---|---|---|
| Objetivo T | tasa observada de un periodo **reciente y maduro** | TC: promedio de largo plazo (simple por cosecha, o *long-run average* regulatoria) |
| Muestra que calibra | cosechas recientes con 12 m cumplidos (Austral: mar–jun 2025) | toda la historia disponible (Austral: 12 cosechas) |
| Muestra que valida | una cosecha **posterior** madura (o el oráculo en el notebook) | OOT, con la advertencia de que OOT puede estar dentro de la TC |
| Uso típico | provisiones (IFRS 9 exige una medición insesgada que use información razonable y sustentable, incluida la prospectiva), pricing de corto plazo | capital IRB (PD de largo plazo por grado), apetito estable, límites |
| Riesgo | persigue ruido: recalibraciones frecuentes, prociclicidad | sesgo sistemático en años malos: O/E > 1 esperado y aceptado |

**Doble calibración.** El mismo ranking (β, WoE, tabla de puntos) con **dos parámetros declarados**, δ_PIT y δ_TTC, cada uno con su muestra, su fecha y su dueño. El backtesting de cada uno se hace contra lo que promete: δ_TTC no se rechaza porque O/E ≈ 1,24 en un año malo si la TC está bien fijada. En el notebook, la calibración TTC en TTD da O/E 1,24 y la PIT 1,005. Lo incorrecto es mezclar: calibrar PIT y reportar la PD como TTC, o validar PIT contra la muestra de calibración.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo / riesgo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| Ajuste de intercepto aprox. (Siddiqi) | nivel medio con dos números | sesgo sistemático $-\bar\kappa\delta$ | solo cálculo de servilleta | práctica tradicional de scorecards (Siddiqi 2006/2017) |
| Ajuste de intercepto exacto (raíz o GLM con offset) | nivel medio exacto en la muestra de calibración | 1 parámetro; no arregla pendiente ni forma | **por defecto** en scorecards logísticos | curso; práctica bancaria |
| Corrección a priori (King & Zeng) | sobremuestreo por clase | requiere π₁ conocido y muestreo que dependa solo de y | desarrollo con muestras balanceadas o *case-control* | eventos raros, fraude, bajo default |
| MLE ponderado | sobremuestreo, robusto a mal especificación | SE menos eficientes; el FOC clava la media ponderada | modelo dudoso, varias estratificaciones | King & Zeng (2001) lo discuten como alternativa |
| Recalibración logística (a, b) | nivel + confianza | 2 parámetros; la pendiente necesita n grande | validación muestra b ≠ 1 significativo | medicina clínica (Steyerberg; Van Calster), bancos |
| Platt | igual que (a, b), sobre el score o margen | la versión sklearn suaviza objetivos | clasificadores no probabilísticos (SVM, boosting) | ML general |
| *Beta calibration* | forma asimétrica con 3 parámetros | más varianza | salidas de ML con sesgo en los extremos | Kull et al. (2017) |
| Isotónica (PAV) | cualquier forma monótona | escalones, 0/1 en extremos, empates, rompe PDO, n grande | modelos ML con n de calibración ≥ miles de malos | ML general; poco en scorecards |
| Por bandas (*histogram binning*) | forma, con PD gobernable por banda | depende de la partición; monotonía entre bandas | master scale con PD = tasa observada suavizada | bancos (M16), EBA: calibración por grado |
| Calibración por grado a LRA | nivel por grado de rating al promedio de largo plazo | exige historia de ciclo | IRB | EBA GL/2017/16 (párr. 87–92); CRR art. 180 |
| *Quasi moment matching* (Tasche) | ajustar una curva de PD a una media objetivo **y** a un poder discriminante objetivo | supone una forma paramétrica | recalibrar la curva completa con TC y AR objetivo | Tasche (2013) |
| Conversión PIT↔TTC tipo Vasicek | mover la PD con un factor sistémico: $\Phi^{-1}(PD_{PIT})=\frac{\Phi^{-1}(PD_{TTC})-\sqrt\rho\,Z}{\sqrt{1-\rho}}$ | supuestos de un factor; ρ y Z estimados | puentes IFRS 9 / capital | práctica de modelación macro (ver Serie 1 · E2, E7) |

**Regulación, con cautela.**

- **EBA GL/2017/16** (aplicables desde el 1-1-2021): la tasa de default de largo plazo es el **promedio aritmético de las tasas anuales de default** (párr. 81). El periodo debe reflejar la «variabilidad probable» con una mezcla representativa de años buenos y malos, más de 5 años si hace falta (párr. 82–85). La calibración se testea por grado o pool y por segmento de calibración (párr. 87–92). Tiene sección propia sobre margen de conservadurismo (4.4).
- **CRR (art. 180)** pide, para minoristas, estimar PD por grado o pool a partir de promedios de largo plazo de tasas de default a un año (verificar numeral exacto).
- **IFRS 9 (5.5.17)** exige que la pérdida esperada refleje un monto **insesgado** y ponderado por probabilidad, el valor del dinero en el tiempo e información razonable y sustentable sobre hechos pasados, condiciones actuales y pronósticos. Por eso la PD de provisiones es PIT y prospectiva.
- **CMF (Chile)**: ver Serie 1 · E6. Según entiendo, las provisiones de consumo bancarias en Chile combinan métodos estándar y modelos internos bajo el Compendio de Normas Contables (cap. B-1). Verificar con la norma vigente antes de afirmar requisitos de calibración específicos.

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 Calibrar y validar en la misma muestra.**
- *Síntoma*: O/E = 1,000 y p binomial global = 1 en la muestra de «validación».
- *Causa*: el δ exacto iguala esperados y observados por construcción. $k$ es la moda de $\text{Bin}(n,k/n)$ (la moda es $\lfloor(n+1)k/n\rfloor=k$ para $k<n$), así que el p bilateral de «probabilidades ≤» es 1.
- *Detección*: contrato de datos que prohíba intersección entre las cohortes de calibración y de validación (§6). Buscar p = 1,000 exacto en tableros.
- *Qué hacer*: validar en una cosecha posterior madura. Si no existe todavía, declarar el nivel como «promesa» con fecha de verificación (el «6,31% de TTD» de Austral).

**5.2 Validar contra un δ estimado con una muestra de tamaño comparable.**
- *Síntoma*: demasiados rojos. En el notebook, calibrando en una mitad de la ventana PIT y testeando en la otra, **15,5%** de los p caen bajo 0,05 (no 5%).
- *Causa*: $\operatorname{Var}(O_B-n_B\hat p_A)=n_Bp(1-p)\,(1+n_B/n_A)$. El binomial ignora el segundo término. Con mitades iguales la tasa de rechazo nominal 5% es $2\Phi(-1{,}96/\sqrt2)=16{,}6\%$.
- *Detección*: comparar $n_B/n_A$. Si no es chico, el test está mal especificado.
- *Qué hacer*: test de diferencia de dos proporciones, o inflar la varianza por $(1+n_B/n_A)$.

**5.3 Siddiqi con PD dispersas.**
- *Síntoma*: tras «calibrar», la PD media en la muestra de calibración no es la TC declarada; la brecha es proporcional a δ.
- *Causa*: §3.3, sesgo relativo $\bar\kappa$.
- *Detección*: el test de CI `abs(mean(pd_cal) − T) < 1e-8`.
- *Qué hacer*: δ exacto.

**5.4 Olvidar la corrección de sobremuestreo o de ponderación.**
- *Síntoma*: PD media del modelo ≈ tasa de la muestra de desarrollo (p. ej. 34% con ρ₁ = 40%), desproporcionada frente a la tasa real.
- *Causa*: §3.4.
- *Detección*: comparar la PD media en una muestra poblacional (TTD, HO sin submuestrear) con la tasa histórica.
- *Qué hacer*: δ_KZ o δ exacto a π₁. Documentar ρ₁ y π₁ en el expediente.

**5.5 Cambio de solver o de regularización que mueve el nivel.**
- *Síntoma*: tras una «refactorización» del entrenamiento, la PD media en DEV ya no es la tasa de DEV.
- *Causa*: penalización del intercepto (`liblinear`), pesos por clase (`class_weight="balanced"`) o *early stopping*.
- *Detección*: invariante de CI $|\bar p_{DEV}-\bar y_{DEV}|<10^{-6}$ para el modelo base sin pesos.
- *Qué hacer*: fijar solver y penalización en el contrato del entrenamiento. Si se usan pesos, calibrar después, explícitamente.

**5.6 Un δ constante frente a un problema de pendiente o de forma.**
- *Síntoma*: la media calza, pero la curva cruza la diagonal: optimista en bandas buenas y pesimista en malas (o al revés). Austral: A2/B2 subestiman.
- *Causa*: sobreajuste ($b<1$), deriva que no es un desplazamiento uniforme del logit (*concept shift*, cambio de mezcla no capturado) o variables que perdieron señal.
- *Detección*: pendiente de calibración con su SE y su IC. Binomial por banda con patrón (M19).
- *Qué hacer*: recalibración (a, b) si $b$ es significativamente distinto de 1 y estable; si no, re-desarrollo. Nunca ajustar grupos a mano.

**5.7 Muestra de calibración inmadura.**
- *Síntoma*: T PIT sospechosamente baja; tras calibrar, las cosechas siguientes muestran O/E > 1 creciente.
- *Causa*: cosechas sin los 12 meses de ventana: los malos aún no maduran (curvas de maduración, Serie 1 · M3).
- *Detección*: `max(cohorte_calibración) + 12 meses ≤ fecha_de_corte_del_desempeño` como aserción del pipeline.
- *Qué hacer*: usar solo cosechas maduras. Si hace falta más recencia, extrapolar con curvas de maduración y declararlo como juicio.

**5.8 Isotónica con n chico.**
- *Síntoma*: PD de 0% o 100% en los extremos; muchos empates; AUC in-sample sube y fuera baja; la tabla de puntos ya no mapea a PD por PDO.
- *Causa*: §3.6.
- *Detección*: contar valores únicos, mínimos y máximos, y Gini in/out.
- *Qué hacer*: δ o (a, b) en scorecards. Si se necesita forma, bandas monótonas con mínimo de malos por banda, más piso y techo de PD.

**5.9 Leer la curva a ojo, o elegir la partición después de mirar.**
- *Síntoma*: «el grupo 5 está descalibrado», con p = 0,003 con deciles y p = 0,020 con bandas.
- *Causa*: comparaciones múltiples y grados de libertad del analista.
- *Detección*: fijar *ex ante* la partición (la master scale) y el umbral por comparación. Reportar el número esperado de falsos positivos (con 9 indicadores al 5%, 37% de ver al menos un amarillo por azar; la clase 5 cita 34% para su set de indicadores).
- *Qué hacer*: IC de Wilson o Jeffreys por banda, test global (HL con p simulado, M19) y patrón antes que puntos.

**5.10 TC de ventana corta tratada como TTC.**
- *Síntoma*: «TTC» que cambia cada año.
- *Causa*: 12 o 24 cosechas no cubren un ciclo. La TC queda sesgada al régimen de la ventana (Banco Sintético: 18 cosechas «buenas» y 6 «malas» ⇒ TC 12,4%).
- *Detección*: comparar con series macro o con la tasa de default de largo plazo del sistema.
- *Qué hacer*: declararla como «ancla de la ventana disponible», con plan de revisión. Para capital, seguir la guía de LRA (EBA párr. 82–85) o su equivalente local.

**5.11 Brier como métrica de calibración.**
- *Síntoma*: «calibrar mejoró el Brier de 0,1159 a 0,1128: poco».
- *Causa*: Brier = REL − RES + UNC + … y UNC domina.
- *Detección*: mirar REL, O/E, CITL y pendiente.
- *Qué hacer*: reportar la descomposición, no el total.

**5.12 Re-escalar la tabla de puntos en un sistema y mover el cutoff en otro.**
- *Síntoma*: después de recalibrar, la aprobación cambia sin que nadie lo haya decidido, o la PD reportada en riesgo no coincide con la del motor de decisión.
- *Causa*: δ aplicado dos veces (tabla re-escalada **y** cutoff movido) o ninguna.
- *Detección*: test de paridad de punta a punta, score → PD → decisión, en un set dorado de clientes (M21).
- *Qué hacer*: un único artefacto de calibración versionado, consumido por todos los sistemas.

**5.13 Calibrar en aprobados y aplicar a solicitantes.**
- *Síntoma*: nivel correcto en la cartera aprobada y subestimado en TTD.
- *Causa*: sesgo de selección: los aprobados son los buenos del score anterior.
- *Detección*: comparar la distribución de score de la muestra de calibración con TTD (PSI, M08).
- *Qué hacer*: inferencia de rechazados (Serie 1 · E1) o, al menos, declarar el alcance de la calibración.

---

## 6. Puente con ingeniería

La calibración es un **artefacto separado del modelo**. El modelo (binning, WoE, β, tabla de puntos) se congela en el desarrollo. La calibración es un parámetro de negocio que cambia más seguido, con su propio ciclo de vida, dueño y auditoría. En un pipeline declarativo:

```yaml
# calibracion/austral_consumo_v3_pit_2025q2.yaml
modelo_ref: austral_consumo_v3@sha256:9f2c…     # artefacto congelado (β, WoE, cortes)
tipo: PIT                                        # PIT | TTC
metodo: intercepto_exacto                        # intercepto_exacto | logistica_ab | bandas
objetivo:
  fuente: tasa_observada                         # tasa_observada | tendencia_central
  valor: 0.0594
  definicion_target: "90+ DPD a 12m; indeterminados 30-89 fuera"
muestra_calibracion:
  cohortes: ["2025-03", "2025-04", "2025-05", "2025-06"]
  madurez_requerida_meses: 12
  fecha_corte_desempeno: "2026-06-30"
  n: 2004
  malos: 119
  hash_pd_individuales: sha256:1ab7…
resultado:
  delta: 0.177
  delta_siddiqi_referencial: 0.143
  desplazamiento_score_pts: -5.11
muestra_validacion:
  cohortes: ["2025-07", "2025-08", "2025-09"]    # posteriores, madurarán 2026-09
  estado: pendiente
gobierno:
  dueno_tc: "Comité de Riesgo, acta 2025-10-14"
  vigencia_hasta: "2026-06-30"
  gatillo_recalibracion: "|O/E - 1| > 0.15 dos trimestres o binomial global p < 0.01"
```

**Invariantes verificables (tests tipo CI):**

```python
def test_calibracion(cfg, pd_cal_muestra, lp_muestra, cohortes_cal, cohortes_val, auc_antes, auc_despues):
    # 1. la media calibrada es el objetivo (δ exacto, no Siddiqi)
    assert abs(pd_cal_muestra.mean() - cfg.objetivo.valor) < 1e-8
    # 2. el ranking no cambia (δ, o a+b con b>0)
    assert auc_antes == auc_despues
    # 3. disjunción calibración / validación
    assert set(cohortes_cal).isdisjoint(cohortes_val)
    # 4. madurez: la última cohorte de calibración tiene 12 m de desempeño
    assert max(cohortes_cal) + meses(cfg.muestra_calibracion.madurez_requerida_meses) <= cfg.fecha_corte
    # 5. coherencia con el score: desplazamiento = -δ·factor
    assert abs(cfg.resultado.desplazamiento_score_pts + cfg.resultado.delta * 20 / np.log(2)) < 0.01
    # 6. δ_Siddiqi tiene el mismo signo y menor magnitud (sanidad de la identidad de §3.3)
    assert 0 < cfg.resultado.delta_siddiqi_referencial / cfg.resultado.delta < 1
    # 7. PD dentro de [piso, techo] tras calibrar
    assert pd_cal_muestra.min() >= PISO_PD and pd_cal_muestra.max() <= 1 - PISO_PD
```

Para el **modelo base** hay un invariante que corre en cada re-entrenamiento: $|\bar p_{DEV}-\bar y_{DEV}|<10^{-6}$ si el ajuste es sin pesos y con intercepto libre. Si falla, alguien cambió el solver, los pesos o la penalización.

**Qué se congela y qué se versiona.**
- *Congelado*: cortes de bins, WoE, β, factor y offset. Cambiarlos es un modelo nuevo.
- *Versionado aparte*: δ (o a, b), el objetivo T con su fuente, la muestra de calibración (lista de cohortes y hash de las PD individuales), fecha, dueño y muestra de validación prevista.
- *Derivado, nunca editado a mano*: tabla score → PD de la master scale y cutoff en score si el apetito está en PD. Se regenera desde el artefacto de calibración.

**Dos calibraciones = dos archivos**, `…_pit_2025q2.yaml` (provisiones) y `…_ttc_2025.yaml` (capital, apetito), con el mismo `modelo_ref`. El servicio de scoring expone `pd_pit` y `pd_ttc` como salidas con nombre. Nunca una «PD» a secas.

**Reproducibilidad del δ.** Guarda el δ con 6 decimales y el hash de las PD de entrada. Con el mismo input, bisección y Brent deben coincidir a 1e-9; el test lo exige. Evita recalcular δ en línea en producción: es un parámetro, no una función del tráfico.

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy (notebook) | Librería | Diferencias y trampas | Producción |
|---|---|---|---|---|
| δ exacto | `delta_biseccion` (45 iteraciones a 1e-12) | `scipy.optimize.brentq` | Mismo resultado a 1e-9. Brent converge más rápido. Ambos requieren un intervalo con cambio de signo (garantizado por §3.2). | brentq, o GLM con offset si ya estás en statsmodels |
| δ = GLM con offset | `irls_logistica(1, y, offset=η)` | `sm.GLM(..., offset=η)` | Idénticos. El GLM entrega además SE(δ), útil para §5.2. | statsmodels |
| Recalibración (a, b) | `irls_logistica([1, η], y)` | `sm.GLM`/`sm.Logit`; `sklearn.LogisticRegression(C=np.inf)` | sklearn: el `C` por defecto (=1) regulariza b hacia 0, y `liblinear` penaliza a. Usa `C=np.inf` (en sklearn 1.8 `penalty=None` está deprecado). | statsmodels (inferencia) |
| Platt | GLM sobre el score | `CalibratedClassifierCV(method="sigmoid")` | sklearn usa objetivos suavizados de Platt: no clava la media ni coincide con el MLE. | GLM explícito |
| Isotónica | `pav_numpy` + `np.interp` | `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` | sklearn agrega empates en x (`_make_unique`) igual que el PAV del notebook, interpola linealmente entre umbrales y recorta fuera de rango. `increasing="auto"` puede invertir el sentido en silencio: fija `increasing=True`. | sklearn + piso/techo |
| Curva / ECE | `grupos_cuantil` = percentiles + `searchsorted` | `sklearn.calibration.calibration_curve(strategy="quantile")` | sklearn no devuelve los n por grupo (ECE necesita pesos) y descarta grupos vacíos. El curso usa `pd.qcut(..., duplicates="drop")`, que con empates da **otros** grupos: declara la regla. | numpy propio (necesitas n) |
| Brier | `np.mean((p-y)**2)` | `sklearn.metrics.brier_score_loss` | Idénticos. sklearn no descompone: Murphy es tuyo. | propio |
| Log-loss | numpy con recorte | `sklearn.metrics.log_loss` | sklearn recorta a `eps` de máquina del dtype; si recortas a 1e-15, las diferencias aparecen solo con PD 0/1 (isotónica). | sklearn |
| Wilson | fórmula cerrada | `statsmodels.stats.proportion.proportion_confint(method="wilson")` | Idénticos. | cualquiera |
| Jeffreys | `scipy.special.betaincinv` | `proportion_confint(method="jeffreys")` | Ninguno fija 0 en k = 0 (Brown et al. sí). Documenta la convención. | statsmodels + ajuste de bordes |
| Binomial | `binom.sf`/`binom.cdf`, doble cola mínima | `scipy.stats.binomtest` | El bilateral de scipy suma las probabilidades ≤ P(k) y puede diferir del «2 × mínima cola». Detalle en M19 §3.2; ejemplo con la cohorte swap-in en M18 §3.8. | declarar el método |

Regla práctica: numpy para el **artefacto** (la función de calibración que corre en producción debe ser trivial y sin dependencias: `sigmoid(lp + delta)`). statsmodels para la **inferencia** (SE, IC de a y b). sklearn para isotónica y métricas estándar, con parámetros fijados explícitamente.

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral, de punta a punta

| Paso | TTC (curso) | PIT (clase 4 v21) |
|---|---|---|
| Muestra de calibración | DEV (3.322), PD media 4,97% | OOT mar–jun 2025 (2.004, 119 malos), PD media 5,19% |
| Objetivo T | TC 5,42% (12 cosechas, promedio simple) | 5,94% |
| δ Siddiqi | 0,092 | 0,143 |
| δ exacto | 0,111 (0,1115) | 0,177 |
| κ̄ implícito = 1 − δ_S/δ | ≈ 0,18 | ≈ 0,19 |
| Score | −3,22 pts | −5,11 pts |
| OOT tras calibrar | 5,65% vs 5,94% (−0,29 pp), binomial global p = 0,56 | 5,94% vs 5,94% (0,00, p = 1 por construcción) |
| Aprobación en cutoff 560 | 78,3% | 77,2% |

**Recalibrar TTC → PIT con 10 grupos (planilla).** La planilla viene cargada con la tabla OOT de la clase 4 v1 (PD TTC por grupo, n y malos). Con objetivo 5,94% calcula: PD media 5,65%, κ agrupado 0,154, Siddiqi 0,0531, Newton 0,0628 y exacto agrupado 0,0630. Como los δ se suman, el exacto individual sería 0,177 − 0,1115 = 0,0655. **Agrupar en 10 grupos subestima δ en ~4%** porque pierde la dispersión intra-grupo. Para una cifra de expediente, calcula con PD individuales. La planilla también marca el grupo 5 en rojo (p de una cola 0,003; bilateral por duplicación 0,006) y el total en verde con p = 1: es la misma muestra con que se calibró.

### 8.2 Banco Sintético (notebook)

| | DEV | HO | OOT | TTD (oráculo) |
|---|---|---|---|---|
| PD media del modelo | 11,28% | 11,36% | 11,47% | 11,93% |
| Tasa observada | 11,28% | 11,80% | 15,40% | 16,21% |
| PD real (verdad) | 11,80% | 11,87% | 16,10% | 16,74% |
| Gini | 0,532 | 0,588 | 0,545 | 0,550 |

- TC (24 cosechas, simple) 12,42% (ponderada 12,43%). Tasa PIT (mar–jun 2025) 15,72%.
- δ_TTC: Siddiqi 0,1092, exacto 0,1245 (error 12,3%, κ₀ = 0,120).
- δ_PIT: Siddiqi 0,3542, exacto 0,4068 (error 13,0%, κ₀ = 0,119).
- Validación en TTD con oráculo:

| Método | O/E | CITL logit | pendiente | REL | ECE | RMSE vs verdad |
|---|---|---|---|---|---|---|
| sin calibrar | 1,360 | 0,413 | 1,063 | 0,0035 | 0,043 | 0,0964 |
| δ TTC | 1,236 | 0,289 | 1,063 | 0,0021 | 0,031 | 0,0876 |
| δ PIT | 1,005 | 0,007 | 1,063 | 0,0004 | 0,014 | 0,0743 |
| logística a+b (PIT) | 1,003 | 0,004 | 1,004 | 0,0003 | 0,013 | 0,0730 |
| isotónica (PIT) | 0,999 | −0,001 | 0,890 | 0,0002 | 0,012 | 0,0764 |

  Lectura: en un año con deterioro plantado, la TTC subestima por diseño (O/E 1,24). La PIT clava. La logística corrige además la pendiente. La isotónica tiene el mejor REL y ECE (métricas por grupo, que premian su flexibilidad), pero peor error contra la verdad crédito a crédito y pendiente 0,89: sobreajusta.

### 8.3 Caso motos: desarrollo balanceado y dos usos de la PD

Parámetros genéricos (no son datos de ninguna empresa). Una financiera de motos desarrolla su scorecard con una muestra balanceada: todos los malos disponibles (1.500) y 1.500 buenos al azar, ρ₁ = 50%. La tasa poblacional histórica de malos es π₁ = 8%.

1. **Corrección de prior**: $\delta_{KZ}=\ln(0{,}08/0{,}92)-\ln(0{,}5/0{,}5)=-2{,}442$. Sin ella, la PD media del modelo en la población quedaría muy por encima del 8% (del orden de 30–50%, según la discriminación) y cualquier pricing por riesgo quedaría absurdo. Con PDO 20 eso son $-2{,}442\cdot28{,}85=-70{,}5$ puntos: la tabla de puntos del modelo balanceado está corrida 70 puntos respecto de una escala poblacional.
2. **TTC para apetito, PIT para provisiones.** Si la historia de 3 años promedia 8% (TC) y las últimas 4 cosechas maduras muestran 10%, δ_PIT − δ_TTC ≈ $\ln(0{,}10/0{,}90)-\ln(0{,}08/0{,}92)\approx0{,}246$ en versión Siddiqi. El exacto es mayor en $\bar\kappa$ %: con Gini ~0,6, κ ~0,12–0,18, el exacto queda cerca de 0,28–0,30. Son ~8 puntos de score. Si el cutoff está escrito en PD («aprobar si PD TTC ≤ 12%»), la aprobación no cambia al actualizar la PIT, porque la PIT solo alimenta provisiones. Si alguien usa la PIT en el motor de decisión, la aprobación cae varios puntos sin que el comité lo haya decidido.
3. **Segmentos**: si la mora sube solo en un canal (p. ej. concesionarios nuevos), un δ global no es la corrección correcta. Es *concept/covariate shift* por segmento: se testea calibración por canal antes de mover el nivel global.

---

## 9. Preguntas de comité

**1. «Esta PD, ¿a qué tasa está calibrada, con qué muestra, quién fijó el objetivo y cuándo se revisa?»**
*Respuesta modelo*: PIT a 5,94%, tasa observada de las cosechas mar–jun 2025, todas con 12 meses de ventana cumplidos a la fecha de corte; δ exacto +0,177 (score −5,1 pts). El objetivo lo aprobó el Comité de Riesgo (acta …). Se revisa semestralmente o si se activa el gatillo (|O/E−1| > 15% dos trimestres). Coexiste una calibración TTC a 5,42% (δ +0,1115) para apetito, documentada por separado.

**2. «¿Por qué no usaron la fórmula de Siddiqi, que es la estándar?»**
Porque es sesgada hacia cero por construcción. Su error relativo es exactamente $\bar\kappa=\overline{\operatorname{Var}(p)/[\bar p(1-\bar p)]}$, ≈ 19% en este modelo (0,143 vs 0,177). Con Siddiqi, la PD media en la muestra de calibración no habría sido la tasa declarada. El exacto cuesta una línea y es reproducible a 1e-9.

**3. «La calibración PIT da O/E = 1,00 en OOT. ¿Eso valida el nivel?»**
No. OOT es la muestra de calibración; el O/E = 1 y el p = 1 son identidades. La validación del nivel se hará con las cosechas jul–sep 2025 cuando maduren. Mientras tanto, la evidencia independiente disponible es la calibración TTC evaluada en OOT: p = 0,56.

**4. «Si la media calza, ¿por qué los grupos buenos subestiman?»**
Porque el δ corrige solo el nivel medio. El patrón (A2/B2 y el grupo 5 con O/E > 1 en bandas buenas) es compatible con una pendiente de calibración distinta de 1 o con deriva concentrada en perfiles buenos. Estimamos la pendiente con su SE: si es distinta de 1 con significancia y estable, se propone recalibración (a, b); si no, se vigila. No se corrige ningún grupo a mano.

**5. «El grupo 5: 4% observado contra 1,18% predicho. ¿Es descalibración?»**
Individualmente, p de una cola 0,003, e IC de Wilson [2,0%; 7,7%] que excluye 1,18%. Pero el grupo se destacó *ex post* entre 10, la partición es arbitraria (con las bandas de la master scale el mismo tramo da p = 0,02) y el binomial ignora la correlación de defaults. Es señal amarilla con patrón, no un rojo aislado.

**6. «¿Por qué no usan isotónica, que calibra mejor?»**
Calibra mejor *in-sample* y en métricas por grupo. Fuera de muestra, con nuestro n, su error contra la verdad es mayor que el de δ o (a, b). Además produce PD de 0% y 100% en los extremos, empates que alteran el Gini y un mapa score → PD escalonado incompatible con la escala PDO y con los reason codes por puntos. Para forma usamos la master scale por bandas, con monotonía y mínimo de malos por banda.

**7. «Al recalibrar, ¿cambió la política de aprobación?»**
Depende de dónde esté escrito el apetito. Si el cutoff está en score, la aprobación no cambia; cambia la PD prometida (se re-documenta la PD del corte). Si el apetito está en PD, el score de corte se mueve δ·factor = 5,1 puntos y la aprobación baja. Lo mostramos con la tabla de estrategia antes y después (M17). Cualquier cambio de aprobación queda en acta.

**8. «¿El modelo se desarrolló con alguna ponderación, sobremuestreo o regularización?»**
No. Logística sin pesos ni penalización (statsmodels), y el test de CI verifica que la PD media en DEV es la tasa de DEV a 1e-6. Si se hubiera usado muestra balanceada, se aplica la corrección de King-Zeng $\ln[\pi_1\rho_0/(\pi_0\rho_1)]$ y se documentan ρ₁ y π₁.

---

## 10. Ejercicios

**E1 (a mano).** Austral TTC: PD media DEV 4,97%, TC 5,42%. (a) Calcula δ_S. (b) Sabiendo que el exacto es 0,1115, calcula κ̄. (c) Si κ(0) = 0,17, ¿qué da el paso de Newton?

<details><summary>Solución</summary>

(a) logit(0,0542) = ln(0,0542/0,9458) = −2,8594; logit(0,0497) = ln(0,0497/0,9503) = −2,9508; δ_S = 0,0914.
(b) κ̄ = 1 − 0,0914/0,1115 = 0,180.
(c) δ_N = 0,0914/(1 − 0,17) = 0,1101: a 0,0014 del exacto, contra 0,0201 de Siddiqi.
</details>

**E2 (derivación).** Demuestra que $g'(\delta)=1-\kappa(\delta)$ y concluye que, si $T>\bar p$, entonces $0<\delta_S<\delta^{\star}$.

<details><summary>Solución</summary>

Ver §3.3. Como $\kappa\in[0,1)$, $g'\in(0,1]$: $g$ es creciente con pendiente menor o igual a 1. Si $T>\bar p$, entonces $\text{logit}\,T>g(0)$ y $\delta^{\star}>0$. Por lo tanto $\delta_S=g(\delta^{\star})-g(0)=\int_0^{\delta^{\star}}g'\le\delta^{\star}$, con desigualdad estricta si alguna $p_i$ difiere de las demás (Var > 0), y $\delta_S>0$ porque $g'>0$.
</details>

**E3 (King-Zeng).** DEV tiene π₁ = 11,28%. Submuestreas buenos hasta ρ₁ = 40%. Calcula δ_KZ y explica por qué no depende de x.

<details><summary>Solución</summary>

δ = ln(0,1128/0,8872) − ln(0,40/0,60) = −2,0624 − (−0,4055) = −1,657 (el notebook da −1,658 con la ρ₁ efectiva). No depende de x porque la razón de verosimilitud $f(x\mid1)/f(x\mid0)$ es la misma en muestra y población si el muestreo depende solo de y; se cancela en el cociente de odds (§3.4).
</details>

**E4 (IC a mano).** Calcula el IC de Wilson al 95% para 3 malos en 200 (grupo 3 de OOT de Austral, PD PIT 0,34%). ¿Contiene la PD?

<details><summary>Solución</summary>

p̂ = 0,015; z² = 3,8415; denominador 1 + 3,8415/200 = 1,01921; centro = (0,015 + 0,009604)/1,01921 = 0,02414; semiancho = (1,95996/1,01921)·√(0,015·0,985/200 + 3,8415/160000) = 1,92302·√(0,00007388 + 0,00002401) = 1,92302·0,009894 = 0,01903. IC ≈ [0,51%; 4,32%]. **No** contiene 0,34%: otro grupo bueno que subestima, coherente con el patrón.
</details>

**E5 (score).** Con PDO 20, ¿cuántos puntos mueve δ = 0,177? ¿Y la diferencia TTC vs PIT de Austral? Si el apetito es «PD ≤ 2,5%», ¿cuál es el score de corte bajo cada calibración, sabiendo que 600 ↔ odds 50:1 en la escala sin calibrar?

<details><summary>Solución</summary>

0,177·28,8539 = 5,11 pts; (0,177 − 0,1115)·28,8539 = 1,89 pts. PD 2,5% ⇒ odds buenos 39:1 ⇒ en la escala sin calibrar $s=487{,}12+28{,}854\ln39=592{,}83$. La PD calibrada es 2,5% cuando $\eta+\delta=\text{logit}(0{,}025)$, o sea cuando el score sin calibrar vale $592{,}83+\delta\cdot28{,}854$: 596,05 (TTC) y 597,94 (PIT). El corte sube: se aprueba menos.
</details>

**E6 (Murphy).** Demuestra que, si la PD toma exactamente un valor por grupo (bandas), BS = REL − RES + UNC. ¿Qué términos desaparecen y por qué?

<details><summary>Solución</summary>

Si $p_i=\bar p_k$ para todo $i\in k$, entonces $p_i-\bar p_k=0$: WBV = 0 y WBC = 0. Queda la descomposición clásica de Murphy (1973). Con PD continua y K grupos, omitir WBV − 2WBC hace que REL − RES + UNC no sume el Brier (Stephenson et al. 2008).
</details>

**E7 (misma muestra).** Prueba que si δ se calibra exacto a $\bar y$ en una muestra de n créditos con k malos ($0<k<n$), el p-valor bilateral de `binomtest(k, n, k/n)` es 1.

<details><summary>Solución</summary>

La moda de $\text{Bin}(n,p)$ es $\lfloor(n+1)p\rfloor$ (o dos modas si $(n+1)p$ es entero). Con $p=k/n$, $(n+1)k/n=k+k/n$ y $\lfloor k+k/n\rfloor=k$ porque $0<k/n<1$. Entonces $P(X=k)$ es la probabilidad máxima, y el p de «probabilidades ≤ P(k)» suma todas las probabilidades: 1. (Estrictamente, `binomtest` usa la PD media, que con δ exacto es k/n a precisión de máquina; scipy aplica una tolerancia relativa en la comparación de probabilidades.)
</details>

**E8 (tasa de rechazo).** Calibras δ en una muestra de $n_A$ y testeas el nivel con un binomial en $n_B$. Si el modelo está bien calibrado, ¿cuál es la tasa de rechazo real al 5% nominal? Evalúa $n_B/n_A$ = 1, 1/3 y 1/10.

<details><summary>Solución</summary>

$\operatorname{Var}(O_B-n_B\hat p_A)\approx n_Bp(1-p)(1+n_B/n_A)$, así que el estadístico del binomial está inflado por $\sqrt{1+r}$ y la tasa de rechazo es $2\Phi(-1{,}96/\sqrt{1+r})$: r = 1 → 16,6%; r = 1/3 → 9,0%; r = 1/10 → 6,2%. El notebook mide 15,5% con r = 1 (200 particiones). Implicación: la TC de Austral usa 12 cosechas (6.723 créditos) que *incluyen* las de OOT (2.004), así que el test OOT contra TC no es ni independiente ni «misma muestra»: está en medio.
</details>

**E9 (diseño).** Escribe los tests de CI para un artefacto de calibración PIT que exija: madurez, disjunción con validación, media exacta, piso de PD y coherencia score-δ. ¿Qué campo del YAML de §6 usa cada test?

<details><summary>Solución</summary>

Ver el bloque `test_calibracion` de §6: madurez → `muestra_calibracion.cohortes` + `madurez_requerida_meses` + `fecha_corte_desempeno`; disjunción → `muestra_calibracion.cohortes` vs `muestra_validacion.cohortes`; media → `objetivo.valor` + `hash_pd_individuales` (para reconstruir las PD); piso → constante de política; coherencia → `resultado.delta` y `resultado.desplazamiento_score_pts`.
</details>

**E10 (código).** Modifica `recal_bandas` del notebook para que (a) fusione bandas adyacentes no monótonas (PAV sobre las tasas de banda, ponderado por n) y (b) imponga un mínimo de 20 malos por banda. Compara su RMSE contra la verdad con n_cal = 600 y 3.000.

<details><summary>Solución</summary>

(a) Calcula tasas y n por banda y aplica `pav_numpy(np.arange(K), tasas, w=n)`. (b) Antes, fusiona iterativamente la banda con menos malos con su vecina de tasa más parecida hasta que todas tengan ≥ 20. Esperable: con n_cal = 600 (~95 malos) quedan 3–4 bandas, y el RMSE mejora respecto de 10 bandas sin restricción pero no alcanza a δ. Con 3.000 se acerca a la logística. La lección es la misma de §3.6: la flexibilidad necesita malos.
</details>

---

## 11. Referencias

- **Siddiqi, N. (2006).** *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley. **2ª ed. (2017)**: *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards*. Fuente del ajuste de intercepto «de servilleta» y de la práctica de scorecards.
- **King, G. & Zeng, L. (2001).** «Logistic Regression in Rare Events Data». *Political Analysis* 9(2), 137–163. Corrección a priori vs ponderación con muestras por clase; derivación limpia del δ analítico.
- **Saerens, M., Latinne, P. & Decaestecker, C. (2002).** «Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure». *Neural Computation* 14(1), 21–41. El mismo ajuste cuando el nuevo prior no se conoce (EM); base del *label shift*.
- **Tasche, D. (2013).** «The art of probability-of-default curve calibration». *Journal of Credit Risk* 9(4) (verificar número y páginas); arXiv:1212.3716. Calibración de curvas de PD en banca, *quasi moment matching* y la distinción entre calibrar la media y la discriminación.
- **Cox, D. R. (1958).** «Two further applications of a model for binary regression». *Biometrika* 45, 562–565. Origen de la recalibración intercepto + pendiente.
- **Platt, J. (1999).** «Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods». En *Advances in Large Margin Classifiers*, MIT Press. Escalado sigmoide y los objetivos suavizados que usa sklearn.
- **Zadrozny, B. & Elkan, C. (2001; 2002).** «Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers» (ICML 2001) y «Transforming classifier scores into accurate multiclass probability estimates» (KDD 2002). *Histogram binning* e isotónica como calibradores.
- **Niculescu-Mizil, A. & Caruana, R. (2005).** «Predicting good probabilities with supervised learning». ICML. Platt vs isotónica según tamaño de muestra: la isotónica sobreajusta con pocos datos.
- **Kull, M., Silva Filho, T. & Flach, P. (2017).** «Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers». AISTATS. Alternativa de 3 parámetros a Platt.
- **Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. (2017).** «On Calibration of Modern Neural Networks». ICML. *Temperature scaling* y popularización del ECE.
- **Naeini, M. P., Cooper, G. & Hauskrecht, M. (2015).** «Obtaining Well Calibrated Probabilities Using Bayesian Binning». AAAI. Definición del ECE usada en ML.
- **Van Calster, B., Nieboer, D., Vergouwe, Y., De Cock, B., Pencina, M. & Steyerberg, E. (2016).** «A calibration hierarchy for risk models was defined: from utopia to empirical data». *Journal of Clinical Epidemiology* 74, 167–176. Jerarquía media/débil/moderada/fuerte.
- **Van Calster, B., McLernon, D., van Smeden, M., Wynants, L. & Steyerberg, E. (2019).** «Calibration: the Achilles heel of predictive analytics». *BMC Medicine* 17, 230. Guía práctica: CITL, pendiente y curvas con IC.
- **Steyerberg, E. (2019).** *Clinical Prediction Models*, 2ª ed. Springer. Recalibración y actualización de modelos: el análogo clínico del ciclo de vida de un scorecard.
- **Brier, G. (1950).** «Verification of forecasts expressed in terms of probability». *Monthly Weather Review* 78, 1–3. La métrica.
- **Murphy, A. (1973).** «A new vector partition of the probability score». *Journal of Applied Meteorology* 12, 595–600. REL − RES + UNC.
- **Stephenson, D., Coelho, C. & Jolliffe, I. (2008).** «Two Extra Components in the Brier Score Decomposition». *Weather and Forecasting* 23(4), 752–757 (verificar páginas). Los términos WBV y WBC que hacen exacta la descomposición con grupos.
- **Tjur, T. (2009).** «Coefficients of Determination in Logistic Regression Models — A New Proposal: The Coefficient of Discrimination». *The American Statistician* 63(4), 366–372. El κ de §3.3.
- **Wilson, E. B. (1927).** «Probable inference, the law of succession, and statistical inference». *JASA* 22, 209–212. El intervalo.
- **Brown, L., Cai, T. & DasGupta, A. (2001).** «Interval Estimation for a Binomial Proportion». *Statistical Science* 16(2), 101–133. Por qué Wald falla y Wilson/Jeffreys no.
- **EBA (2017).** *Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures* (EBA/GL/2017/16), aplicables desde el 1-1-2021. Párr. 79–92: promedio de largo plazo y calibración por grado. La referencia regulatoria más detallada sobre calibración TTC.
- **IASB. IFRS 9 *Financial Instruments*, párr. 5.5.17.** Medición de la pérdida esperada insesgada, ponderada por probabilidad y con información prospectiva: la razón de la PD PIT para provisiones.
- **Serie 1 · E2** (tendencia central, PIT vs TTC), **E6** (regulación), **M3** (curvas de maduración), **E1** (reject inference); **Serie 2 · M13** (scaling), **M16** (master scale), **M19** (backtesting).

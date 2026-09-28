# M09 · Redundancia, colinealidad y familias de variables

> **Ficha**
> - **Profundiza:** clase 3 (paso 3 del embudo: correlación de WoE > 0,70 con greedy por IV, 63 → 26; paso 4: VIF máx 2,97; clusterización conceptual de 8–14 variables) y clase 4 v21, láminas 11 y 17–19 (familias de variables, redundancia condicional, reemplazo entre ventanas, cobertura de drivers).
> - **Prerrequisitos:** Serie 1 · M7 (binning, WoE, IV con derivación formal), Serie 1 · M5 (fábrica de variables como pipeline declarativo), Serie 2 · M08 (estabilidad: el filtro que va *antes* de este). Enlaza hacia M10 (logística sobre WoE desde la verosimilitud) y M11 (stepwise y alternativas).
> - **Archivos:** `M09_redundancia_colinealidad.md` (este documento) · `M09_redundancia_colinealidad.py` (notebook Marimo, ~25 s en CPU) · `M09_matriz_correlacion_woe.csv` (los 190 pares de las 20 candidatas del notebook: Pearson sobre WoE, Pearson y Spearman crudos, IV y rol verdadero).
> - **Tiempo estimado:** 3–4 h (lectura 1,5 h; notebook 1 h; ejercicios 1–1,5 h).

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**El embudo de Banco Austral.** 108 candidatas → 94 (PSI) → 63 (IV ≥ 0,10) → **26 (correlación de WoE > 0,70)** → 26 (VIF) → 8 (stepwise con signo y significancia). La clase 3 lo justificó así:

- La correlación se calcula **sobre el WoE, no sobre los crudos**: «es exactamente lo que el modelo va a ver». Las 63 candidatas se transforman al WoE ajustado en DEV (3.322 solicitudes, 5,0% de malos) y se calcula la matriz de $|\rho|$.
- Había **35 pares sobre 0,90**. Los dos ejemplos de clase: `dias_mora_prom_12m ~ dias_mora_max_12m` 0,989 (distinto agregador, misma información) y `abonos_prom_6m ~ abonos_prom_12m` 0,983 (misma serie, otra ventana). El mapa de calor agrupado por familia (regex `_(3|6|12)m` al final del nombre) mostraba bloques oscuros: «cada bloque es una idea contada varias veces».
- **Greedy por IV:** se recorren las candidatas de mayor a menor IV y cada una entra solo si su correlación con **todas** las ya elegidas es ≤ 0,70. «El algoritmo propone, el modelador dispone.» Resultado: 26.
- **VIF** con `statsmodels.variance_inflation_factor` sobre WoE estandarizados con constante: $\mathrm{VIF}=1/(1-R^2)$; alerta > 5, problema > 10. **Máximo 2,97** (las otras 25 explican el 66% de esa variable): no descartó nada. Se reporta igual, porque «ausencia de evidencia no es evidencia de ausencia» y porque la correlación es de a pares mientras el VIF caza la colinealidad múltiple.
- **Stepwise con signo:** con WoE alto = bin bueno, todos los $\beta$ deben ser negativos; «el signo manda sobre la significancia: un coeficiente muy significativo con el signo equivocado se descarta — está actuando como proxy de otra cosa». En Austral entraron 8 y nadie salió. El Lab 2 (Andes) advertía que con algunas muestras personales entraría una variable con coeficiente positivo.
- **Modelo final:** 8 variables, 5 conceptos. Entre ellas **`uso_tc_prom_12m` (−0,353, p 0,006) y `uso_tc_prom_3m` (−0,309, p 0,015)**: dos ventanas de la misma serie sobrevivieron al filtro (su $|\rho|$ de WoE debió quedar ≤ 0,70). Gini 0,757 / 0,698 / 0,675 (DEV/HO/OOT).
- **Estabilidad de coeficientes en HO:** `deuda_interna_max_3m` y `deuda_otras_prom_12m` «cambian de signo» a ≈ 0 con p 0,96 y 0,78; lectura del curso: indistinguible de cero, no una inversión (78 malos para 9 parámetros).
- **Clusterización conceptual:** agrupar por concepto de negocio (mora, utilización, ahorro, bureau, demografía) y llevar 1–3 de cada grupo; cobertura, robustez ante caída de una fuente, reason codes explicables.
- **Clase 4 v21, lámina 18:** «tres variables de mora (`mora_max_3m/6m/12m`) pueden sobrevivir al filtro y seguir contando casi la misma historia». Revisión conjunta (aporte incremental, estabilidad, disponibilidad, rol de negocio), decisión **a nivel de familia**, y «reestimar porque al sacar 12m puede entrar 6m o 3m». Cierre: «la correlación lineal o por pares no detecta toda la redundancia condicional ni el reemplazo entre ventanas». Lámina 17: pocas variables no es problema en sí; lo es dejar huecos en los drivers (conducta, endeudamiento, capacidad, recencia/intensidad).

**Lo que el curso simplificó, omitió o dejó como convención:**

1. *Por qué* Pearson sobre WoE es la medida correcta (y cuándo no): no se derivó. Tampoco qué hacer con categóricas nominales o mixtas.
2. El greedy se presentó como procedimiento, no como problema de optimización. No se discutió que es **dependiente del orden y del umbral**, ni qué problema combinatorio resuelve (conjunto independiente de peso máximo), ni si su objetivo ($\sum$ IV) tiene sentido.
3. Los umbrales 0,70 (correlación) y 5/10 (VIF) son **convenciones**. El VIF 10 se suele atribuir a Marquardt (1970); O'Brien (2007) muestra que ninguna regla de dedo tiene base inferencial.
4. $\mathrm{VIF}=[R^{-1}]_{jj}$ no se derivó; tampoco la relación entre VIF, autovalores e índice de condición, ni el diagnóstico de Belsley-Kuh-Welsch (BKW) que dice **qué variables** participan en cada dependencia.
5. Una variable con varias columnas (dummies, familia) necesita el **VIF generalizado** de Fox & Monette (1992); el curso no lo mencionó.
6. El VIF que se reporta es el de OLS; en logística la varianza es $(X^\top WX)^{-1}$ y el VIF relevante es el **ponderado por $W$**.
7. El «signo equivocado» se trató como síntoma de proxy. Falta la mecánica: **supresión** en el caso de 2 regresores, y la distinción clave entre volteo de signo *por varianza* (no significativo) y volteo *por efecto condicional real* (significativo).
8. VARCLUS / clustering jerárquico de variables y el criterio de representante $1-R^2$: fuera.
9. Familias temporales: el curso dice qué revisar, no cómo. Aquí: correlación parcial, test de razón de verosimilitud incremental y la re-parametrización **nivel + delta**.
10. Alternativas (L1/elastic net, PCA) y por qué la industria de scorecards no usa PCA.

---

## 2. Intuición

Una scorecard sobre WoE es una suma de columnas que ya están en la escala del riesgo. Cada WoE, sola, es una «opinión» sobre el log-odds del cliente. Dos columnas son redundantes cuando **opinan lo mismo sobre las mismas personas**. El modelo conjunto tiene que repartir el mérito entre ellas, y ese reparto es lo que se vuelve inestable:

- Si dos columnas dicen casi lo mismo, el dato solo identifica **su suma**, no cada sumando. Cualquier combinación $(\beta_1,\beta_2)$ con la misma suma ajusta parecido. Resultado: errores estándar grandes, coeficientes que bailan entre muestras (HO vs DEV) y que pueden cruzar el cero. Eso es **colinealidad**: un problema de *varianza*, no de sesgo.
- Si dos columnas dicen *casi* lo mismo pero su pequeña diferencia **sí** predice (por ejemplo, 3m − 12m = «¿subió el uso?»), el modelo construye esa diferencia con coeficientes de signos opuestos: uno sale «equivocado» y significativo. Eso es **supresión**: el modelo tiene razón, pero no se puede firmar como reason code.
- La redundancia **de a pares** es la más visible pero no la única. Una variable puede ser casi combinación lineal de otras tres sin parecerse mucho a ninguna (colinealidad múltiple), y una variable puede no agregar nada **dado** el resto del modelo aunque su correlación marginal con el líder sea moderada (redundancia condicional).

Tres observaciones que ordenan el módulo:

1. **La redundancia se mide en la escala del modelo.** Pearson sobre WoE es la medida de redundancia lineal en log-odds. Correlación de crudos puede decir «redundantes» cuando en riesgo no lo son (y al revés).
2. **Filtrar redundancia no compra Gini; compra coeficientes firmables.** En el notebook, con verdad conocida, pasar de 7 a 16 variables correlacionadas no mejora el Gini HO (0,593 → 0,602, dentro del ruido muestral) y produce 2 coeficientes positivos.
3. **Ningún filtro estadístico distingue causa de síntoma.** El greedy, el MWIS, el clustering y el lasso eligieron, en más de un caso, un proxy (`dias_mora_max_12m`) en vez del driver verdadero (`meses_desde_mora_12m`). Esa elección es de negocio y queda documentada.

---

## 3. Formalización

Notación: $w_{ij}$ es el WoE de la variable $j$ para el cliente $i$; $X$ la matriz de WoE (sin intercepto salvo que se diga); $R$ su matriz de correlación; $y_i=1$ = malo. Todo se ajusta en DEV.

### 3.1 El WoE es una coordenada de log-odds, y $\beta=-1$ univariado

Para el bin $b$ de la variable $X$, con $g_b, m_b$ buenos y malos del bin y $G, M$ los totales:

$$
\mathrm{WoE}_b=\ln\frac{g_b/G}{m_b/M}=\ln\frac{P(X\in b\mid \text{bueno})}{P(X\in b\mid \text{malo})}.
$$

Por Bayes, el log-odds de malo dentro del bin es

$$
\ln\frac{P(\text{malo}\mid X\in b)}{P(\text{bueno}\mid X\in b)}=\ln\frac{m_b}{g_b}=\ln\frac{M}{G}+\ln\frac{m_b/M}{g_b/G}=\ln\frac{M}{G}-\mathrm{WoE}_b .
$$

Consecuencia 1: la logística univariada $\mathrm{logit}\,p_i=\beta_0+\beta_1 w_i$ con $\beta_0=\ln(M/G)$, $\beta_1=-1$ reproduce **exactamente** la tasa de malos de cada bin. Entonces las ecuaciones de score $\sum_i (y_i-\hat p_i)=0$ y $\sum_i (y_i-\hat p_i)w_i=0$ se cumplen (dentro de cada bin $\sum_{i\in b}(y_i-\hat p_b)=0$ y $w_i$ es constante en el bin). Como la log-verosimilitud logística es estrictamente cóncava si hay al menos dos valores distintos de WoE, ese es **el** MLE: $\hat\beta_1=-1$ exacto sin suavizado. Con el suavizado +0,5 del código común queda en $[-1{,}003;\,-0{,}998]$ para las 16 variables del pool (notebook, §4).

Consecuencia 2: el coeficiente de una variable en el modelo conjunto se lee **contra −1**. Lo sano es $\beta_j\in(-1,0)$: la variable aporta la fracción no redundante de su opinión. Fuera de ese rango hay algo que explicar:

- $\beta_j>0$: supresión o colinealidad (§3.7).
- $\beta_j<-1$: puede ser supresión (una compañera con signo invertido «limpia» a esta, ver §3.7: amplificación ⇔ volteo), **o simplemente no-colapsabilidad** de la logística: al agregar predictores fuertes *no correlacionados*, los coeficientes de los demás crecen en magnitud (Gail, Wieand & Piantadosi 1984; Mood 2010). En el notebook, `antiguedad_meses` tiene correlación ≈ 0 con todo y sale −1,14. No es un problema.

### 3.2 Por qué la correlación sobre WoE (y cuál)

En el modelo $\eta=\beta_0+X\beta$ la matriz de información es $X^\top WX$. La inestabilidad de $\hat\beta$ depende de la **dependencia lineal entre las columnas de WoE** (ponderadas por $W$, §3.6). La correlación de Pearson entre columnas de WoE es exactamente el objeto cuya inversa da los VIF. Nada más ni nada menos. Las alternativas miden otras cosas:

| Medida | Qué mide | Problema como filtro de redundancia para la scorecard |
|---|---|---|
| Pearson crudo | asociación lineal en unidades originales | colas, escalas, **códigos especiales** (−9, −99, 13); no es la escala del modelo |
| Spearman crudo | asociación monótona (Pearson sobre rangos) | arregla colas y escala, no códigos cuyo orden numérico no es orden de riesgo, ni relaciones no monótonas |
| Pearson sobre WoE | asociación lineal en la escala de log-odds de cada variable | depende del binning; es la adecuada |
| Spearman sobre WoE | lo mismo sobre rangos de bins | el WoE tiene pocos valores (empates masivos); no es lo que entra al modelo |
| V de Cramér sobre bins | asociación entre etiquetas, $\sqrt{\chi^2/(n(\min(r,c)-1))}$ | ignora orden y dirección respecto del target; sesgada al alza en muestras chicas |
| Razón de correlación $\eta$ | categórica vs numérica: $\sqrt{SS_\text{entre}/SS_\text{total}}$ | asimétrica; innecesaria si todo se lleva a WoE |

**Ejemplos del notebook (DEV, Tabla 1):**

- `meses_desde_mora_12m ~ dias_mora_max_12m`: Pearson crudo **−0,107**, Spearman crudo −0,402, **Pearson WoE 0,939**. Es la misma información (recencia y severidad de la última mora), pero 13 = «sin mora» queda numéricamente lejos de 12 y −99 aún más. Un filtro sobre crudos las dejaría pasar juntas.
- `deuda_otras_prom_12m ~ deuda_total_mm`: Pearson crudo **0,993**, WoE **0,639**. El error inverso: la cola lognormal de la deuda externa domina ambos crudos, pero en riesgo son distintas porque `deuda_total_mm` hereda la señal de `uso_linea` (IV 0,144 vs 0,015). Un filtro sobre crudos habría descartado una por «redundante».
- Juguete no monótono (riesgo en U sobre $x$, $z\approx|x-0{,}5|$): Pearson crudo −0,006, Spearman −0,012, **Pearson WoE 0,887**.

**Orientación.** El WoE orienta todas las variables hacia «bueno». Por eso, entre variables de riesgo, las correlaciones de WoE son casi siempre positivas (en el pool, la mínima es −0,017). Una correlación de WoE **negativa y grande** significa que dos variables clasifican a las mismas personas en sentidos opuestos. Es rara y merece investigarse (suele ser un binning mal orientado o un proxy de segmento).

**V de Cramér y su sesgo.** Bajo independencia $E[\chi^2]\approx(r-1)(c-1)$, así que $V$ tiene un piso de ruido $\approx\sqrt{(r-1)(c-1)/(n(\min(r,c)-1))}$: 0,063 con 5×5 bins y $n=1000$; 0,134 con 10×10 y $n=500$. Bergsma (2013) propone una corrección de sesgo. Úsala para categóricas nominales *antes* del WoE (p. ej., para decidir agrupar categorías), no como filtro de redundancia del modelo.

### 3.3 VIF: derivación de $\mathrm{VIF}_j=1/(1-R_j^2)=[R^{-1}]_{jj}$

Modelo lineal $y=\beta_0+X\beta+\varepsilon$, $\mathrm{Var}(\varepsilon)=\sigma^2I$. Centra y escala las columnas de modo que $X^\top X=nR$. Entonces

$$
\mathrm{Var}(\hat\beta)=\sigma^2(X^\top X)^{-1}=\frac{\sigma^2}{n}R^{-1}.
$$

Si las columnas fueran ortogonales ($R=I$), $\mathrm{Var}(\hat\beta_j)=\sigma^2/n$. El factor de inflación es entonces $[R^{-1}]_{jj}$. Para calcularlo, ordena de modo que $j$ sea la primera columna y particiona:

$$
R=\begin{pmatrix}1 & r^\top\\ r & R_{-j}\end{pmatrix},\qquad r=\mathrm{Corr}(x_j,X_{-j}).
$$

Por la fórmula de inversa por bloques (complemento de Schur),

$$
[R^{-1}]_{11}=\frac{1}{1-r^\top R_{-j}^{-1}r}.
$$

Queda identificar $r^\top R_{-j}^{-1}r$. La regresión (estandarizada) de $x_j$ sobre $X_{-j}$ tiene coeficientes $\gamma=R_{-j}^{-1}r$ (ecuaciones normales $R_{-j}\gamma=r$). La varianza explicada es $\mathrm{Var}(X_{-j}\gamma)=\gamma^\top R_{-j}\gamma=r^\top R_{-j}^{-1}r$, y como $\mathrm{Var}(x_j)=1$, eso es $R_j^2$. Por lo tanto

$$
\boxed{\mathrm{VIF}_j=[R^{-1}]_{jj}=\frac{1}{1-R_j^2}},\qquad \mathrm{Var}(\hat\beta_j)=\frac{\sigma^2}{n\,s_j^2}\cdot\mathrm{VIF}_j .
$$

El error estándar se infla en $\sqrt{\mathrm{VIF}}$: el 2,97 de Austral significa EE ×1,72 y $R^2=0{,}66$.

**Cuatro consecuencias útiles.**

1. $\mathrm{VIF}_j\ge1$, con igualdad solo si $x_j$ es ortogonal al resto.
2. **Cota de a pares:** $R^2$ con todas las demás ≥ $R^2$ con una sola, así que $\mathrm{VIF}_j\ge 1/(1-\rho_{jk}^2)$ para todo $k$. Y al revés, *después* de filtrar $|\rho|\le0{,}70$, la contribución de cualquier par por sí solo es a lo más $1/(1-0{,}49)=1{,}96$. **Todo VIF > 1,96 después del filtro del curso es colinealidad múltiple.** El 2,97 de Austral no descartó nada, pero sí revela que había estructura de 3+ variables (leve).
3. **Forma espectral:** con $R=\sum_k\lambda_kv_kv_k^\top$, $R^{-1}=\sum_k v_kv_k^\top/\lambda_k$ y

$$
\mathrm{VIF}_j=\sum_k\frac{v_{jk}^2}{\lambda_k}\le\frac1{\lambda_{\min}}\le\frac{\lambda_{\max}}{\lambda_{\min}}=\kappa(R),
$$

usando $\sum_kv_{jk}^2=1$ y $\lambda_{\max}\ge\bar\lambda=1$ (la traza de $R$ es $p$). Entonces el número de condición acota a todos los VIF: $\kappa(R)\ge\mathrm{VIF}_{\max}$, o sea $\eta_{\max}=\sqrt{\kappa}\ge\sqrt{\mathrm{VIF}_{\max}}$ (§3.4). En el pool del notebook, $\kappa(R)=485{,}6\ge38{,}7$.
4. **El filtro de a pares no acota el VIF.** En el juguete del notebook ningún par supera $|\rho|=0{,}697$ y el VIF de $a_4\approx(a_1+a_2+a_3)/\sqrt3$ es **409**.

### 3.4 Índice de condición y descomposición de proporciones de varianza (BKW 1980)

El VIF dice *cuánto* se infla cada varianza, pero no cuántas dependencias hay ni quién participa en cada una. BKW trabajan con la SVD de la matriz de diseño **con intercepto y columnas escaladas a norma 1, sin centrar**: $\tilde X=UDV^\top$, $D=\mathrm{diag}(\mu_1,\dots,\mu_p)$. Entonces

$$
\mathrm{Var}(b)=\sigma^2(\tilde X^\top\tilde X)^{-1}=\sigma^2VD^{-2}V^\top\;\Rightarrow\;\mathrm{Var}(b_j)=\sigma^2\sum_k\frac{v_{jk}^2}{\mu_k^2}=\sigma^2\sum_k\phi_{jk}.
$$

Se definen:

$$
\eta_k=\frac{\mu_{\max}}{\mu_k}\ (\text{índice de condición}),\qquad \pi_{kj}=\frac{\phi_{jk}}{\sum_\ell\phi_{j\ell}}\ (\text{proporción de }\mathrm{Var}(b_j)\text{ asociada a la dimensión }k).
$$

Cada fila $k$ con $\eta_k$ grande es una casi-dependencia: $\tilde Xv_k=\mu_ku_k\approx0$, o sea $\sum_jv_{jk}\tilde x_j\approx0$. Las variables con $\pi_{kj}$ alto (BKW: > 0,5) en esa fila **son las que participan**, porque buena parte de su varianza viene de dividir por ese $\mu_k$ chico. Lectura de BKW: índices del orden de 5–10 acompañan dependencias débiles y de 30–100, dependencias moderadas a fuertes. La práctica usa **$\eta>30$ como alarma** (10–30 como zona gris), junto con **al menos dos** variables con $\pi>0{,}5$ en la misma fila. Estos umbrales vienen de los experimentos de BKW, no de una distribución.

**Por qué ve lo que el VIF diluye.** Juguete del notebook: dos dependencias separadas, $a_4\approx(a_1+a_2+a_3)/\sqrt3$ y $b_3\approx(b_1+b_2)/\sqrt2$, más $c_1,c_2$ independientes. VIF: 134–409 en el bloque $a$ y 13–26 en el bloque $b$. Todos «altos», sin estructura. BKW: una fila con $\eta=40{,}5$ y $\pi\ge0{,}99$ para $a_1..a_4$; otra con $\eta=10{,}3$ y $\pi\ge0{,}95$ para $b_1..b_3$; $c_1,c_2$ en ninguna. La acción correcta es **una variable (o una combinación) por dependencia**, no sacar todo lo que tenga VIF > 10. El caso inverso también existe: una dependencia que involucra muchas variables con cargas chicas reparte la inflación y deja VIF individuales moderados mientras $\eta$ ya es alto.

**Escalas distintas.** Por §3.3, $\eta_{\max}\ge\sqrt{\mathrm{VIF}_{\max}}$ (versión centrada). «$\eta>30$» permite VIF de hasta ~900 en el peor caso: es un umbral **mucho más permisivo** que «VIF > 10». Son convenciones de severidad distinta, no la misma regla.

**Centrar o no.** BKW no centran porque la colinealidad con el intercepto también infla varianzas (una variable casi constante es colineal con la constante). Otros autores (Marquardt; la literatura de econometría aplicada) centran para medir solo la colinealidad «entre regresores». Con WoE la diferencia es menor porque la media del WoE es cercana a 0. Declara cuál usas.

**Pool real del notebook (16 WoE sin filtro):** las cinco peores dimensiones son familias reconocibles: $\eta=23{,}5$ (`dias_mora_max_12m`, `n_meses_mora_12m`), 17,7 (`ratio_deuda_renta`, `carga_financiera`), 14,8 (las tres `uso_tc`), 12,9 (`uso_linea` 12m y 6m), 9,9 (`meses_desde_mora_12m`).

### 3.5 VIF generalizado (Fox & Monette 1992)

Cuando un concepto ocupa varias columnas (dummies de bins en una scorecard *dummy-coded*, o una familia completa), el VIF por columna depende de la parametrización (p. ej., de qué categoría es la referencia). Fox & Monette proponen medir la inflación del **volumen** de la región de confianza del bloque. Con el grupo 1 ($p_1$ columnas) y el resto 2, en escala estandarizada:

$$
\mathrm{Cov}(\hat\beta_1)\propto[R^{-1}]_{11}=(R_{11}-R_{12}R_{22}^{-1}R_{21})^{-1}.
$$

El volumen del elipsoide es proporcional a $\det(\cdot)^{1/2}$. Comparado con el diseño en que el grupo es ortogonal al resto ($\mathrm{Cov}\propto R_{11}^{-1}$):

$$
\mathrm{GVIF}=\frac{\det[R^{-1}]_{11}}{\det R_{11}^{-1}}=\frac{\det R_{11}}{\det(R_{11}-R_{12}R_{22}^{-1}R_{21})}=\frac{\det R_{11}\,\det R_{22}}{\det R},
$$

donde el último paso usa la fórmula del determinante por bloques $\det R=\det R_{22}\det(R_{11}-R_{12}R_{22}^{-1}R_{21})$.

**Segunda forma (correlaciones canónicas).** Los autovalores de $R_{11}^{-1/2}R_{12}R_{22}^{-1}R_{21}R_{11}^{-1/2}$ son las correlaciones canónicas al cuadrado $\rho_k^2$ entre el grupo y el resto. Entonces $\det(I-\cdot)=\prod_k(1-\rho_k^2)$ y

$$
\mathrm{GVIF}=\prod_{k=1}^{p_1}\frac{1}{1-\rho_k^2}.
$$

El notebook implementa ambas y verifica que coinciden. Propiedades: es invariante a re-parametrizaciones **dentro** del grupo; es simétrico (el GVIF del grupo contra el resto es igual al del resto contra el grupo); con $p_1=1$ se reduce al VIF. Para comparar grupos de distinto tamaño se usa $\mathrm{GVIF}^{1/(2p_1)}$, en la escala de inflación del error estándar (comparable con $\sqrt{\mathrm{VIF}}$).

**Trampa de lectura.** El GVIF de una familia contra el resto **cancela la redundancia interna** ($\det R_{11}$ aparece arriba). En el notebook, las tres `uso_tc` tienen VIF individuales de 10,6–18,6, pero su GVIF como familia contra el resto es 2,37: la familia completa se parece moderadamente a `uso_linea` (correlación canónica máxima 0,76), pero el problema está **dentro** de la familia. GVIF para «¿cuánto de esta familia ya está en el resto?»; VIF/BKW dentro de la familia para «¿cuántos miembros necesito?».

### 3.6 VIF en logística: la versión ponderada por $W$

El MLE de la logística se obtiene por IRLS: $\hat\beta=(X^\top WX)^{-1}X^\top Wz$, con $W=\mathrm{diag}\{\hat p_i(1-\hat p_i)\}$ y $z$ la respuesta de trabajo. La covarianza asintótica es $(X^\top WX)^{-1}$. Particionando con el intercepto, el bloque de pendientes es $(\tilde X^\top W\tilde X)^{-1}$, con $\tilde X$ centrada con **medias ponderadas** $\bar x^w_j=\sum_iw_ix_{ij}/\sum_iw_i$. Repitiendo §3.3 con el producto interno ponderado:

$$
\mathrm{VIF}^W_j=[R_W^{-1}]_{jj}=\frac1{1-R^2_{j,W}},\qquad \mathrm{VIF}^W_j=[(X^\top WX)^{-1}]_{jj}\cdot\sum_iw_i(x_{ij}-\bar x^w_j)^2,
$$

donde $R_W$ es la correlación ponderada y $R^2_{j,W}$ el de la regresión WLS de $x_j$ sobre las demás con pesos $w_i$. La segunda igualdad da la verificación con la covarianza de statsmodels (el notebook la hace con `cov_params()`). Si $\hat p$ fuera constante, $W\propto I$ y se recupera el VIF clásico. Lesaffre & Marx (1993) y Segerstedt & Nyquist (1992) discuten el mal condicionamiento de $X^\top WX$ en modelos lineales generalizados.

**Qué cambia en crédito.** $p(1-p)$ pesa más a los clientes con PD alta (con PD de 5–11%, el peso crece casi lineal con la PD). Si dos variables correlacionan distinto entre los riesgosos que en el promedio, el VIF clásico se equivoca. En el notebook, `deuda_total_mm` baja de 1,95 (clásico) a 1,65 (ponderado) y `ratio_deuda_renta` de 1,85 a 1,54: en la zona que pesa, esas dos variables correlacionan menos. El VIF ponderado depende del modelo ajustado ($\hat p$); se calcula al final, no como filtro previo.

### 3.7 Supresión con dos regresores

Trabaja en escala estandarizada (versión lineal; la logística se comporta igual en primera aproximación a través de las ecuaciones normales ponderadas de IRLS, más la no-colapsabilidad). Sean $r_1,r_2$ las correlaciones de $x_1,x_2$ con el target orientado (ambas positivas, como con WoE: las dos «indican bueno») y $\rho$ la correlación entre ellas. Las ecuaciones normales $R\beta=r$ dan

$$
\beta_1=\frac{r_1-\rho r_2}{1-\rho^2},\qquad \beta_2=\frac{r_2-\rho r_1}{1-\rho^2},\qquad R^2=\frac{r_1^2+r_2^2-2\rho r_1r_2}{1-\rho^2}.
$$

**Condición de volteo.** Con $\rho>0$, $\beta_2$ tiene signo opuesto a $r_2$ si y solo si

$$
r_2<\rho\,r_1 .
$$

Es decir, $x_2$ se voltea cuando su relación con el target es *menor que la que heredaría* de $x_1$ a través de $\rho$. Una variable débil muy correlacionada con una fuerte es la candidata natural.

**Amplificación ⇔ volteo.** $\beta_1/r_1>1\iff 1-\rho r_2/r_1>1-\rho^2\iff r_2<\rho r_1$. **Exactamente cuando $x_2$ se voltea, $x_1$ queda amplificado** por sobre su efecto univariado. En WoE: un $\beta<-1$ junto a un $\beta>0$ de una compañera correlacionada es la firma de la supresión.

**Supresión clásica** ($r_2=0$): $\beta_2=-\rho r_1/(1-\rho^2)\neq0$ y $R^2=r_1^2/(1-\rho^2)>r_1^2$. La variable que no predice nada sola mejora el ajuste porque le resta a $x_1$ su componente de ruido (Horst 1941; Conger 1974). Notebook: $x_1=(s+e)/\sqrt{1{,}64}$, $x_2=e/0{,}8$, riesgo solo en $s$. $x_2$ sola: $\beta=-0{,}011$ ($z=-0{,}5$). Con $x_1$: $\beta_2=+0{,}793$ ($z=26{,}8$; teórico +0,80) y $\beta_1=-1{,}293$ (teórico −1,28). **Signo «equivocado», enorme significancia y modelo correcto.** Con WoE esta forma pura casi nunca llega al modelo, porque el IV ≥ 0,10 filtra a $x_2$ antes. Lo que sí llega es la supresión **cooperativa/neta** entre dos proxies.

**Varianza vs sesgo: cuándo un signo equivocado es significativo.** $\mathrm{Var}(\hat\beta_j)=\sigma^2/(n(1-\rho^2))$ (en escala estandarizada). La colinealidad agranda la varianza pero **no sesga** $\hat\beta_2$. Si el efecto condicional verdadero es $\beta_2\le0$:

$$
P(\hat\beta_2>0\ \text{y significativo al nivel }\alpha)\le\frac\alpha2,
$$

sea cual sea $\rho$. La colinealidad sola produce volteos **no** significativos. Un volteo **significativo** indica, salvo un error de tipo I, que el efecto condicional verdadero **sí** tiene ese signo: supresión real, o una variable omitida que el par está reconstruyendo. Notebook (200 muestras de $n=2000$, $\beta_1=-0{,}8$):

| $\beta_2$ verdadero | $\rho$ | $\bar{\hat\beta}_2^{\,univ}$ | % $\hat\beta_2>0$ | % $>0$ y $p<0{,}05$ |
|---|---|---|---|---|
| −0,10 | 0,90 | −0,81 | 28% | 0% |
| 0 | 0,50 / 0,90 | −0,38 / −0,71 | 53,5% | 2,5% (= α/2) |
| +0,20 | 0,50 | −0,19 | 98,5% | 67% |
| +0,20 | 0,90 | −0,52 | 88% | 23,5% |

Con $\rho=0{,}9$ y $\beta_2=-0{,}1$, la desviación estándar de $\hat\beta_2$ es 2,18 veces la de $\rho=0$ (teoría $1/\sqrt{1-0{,}81}=2{,}29$). En la última fila, univariado $x_2$ «parece» fuertemente correcta (−0,52), y en el modelo conjunto sale positiva y significativa 1 de cada 4 veces: **es la situación del Lab 2**.

Esto matiza la regla del curso. «Un coeficiente significativo con signo equivocado se descarta» sigue siendo la **política correcta de gobierno** (no se puede firmar un reason code que da más puntos al bin peor). Pero el diagnóstico no es «ruido por colinealidad», es «hay un efecto condicional que el modelo necesita y que ninguna variable expresa directamente». La respuesta de ingeniería es **construir esa variable** (§3.9).

### 3.8 El greedy es una heurística para un conjunto independiente de peso máximo

Construye el grafo $G_u=(V,E_u)$: vértices = candidatas, arista $\{j,k\}$ si $|\rho_{jk}|>u$. Una selección admisible es un **conjunto independiente** (ningún par con arista). El curso busca el de mayor «valor» con pesos $w_j=\mathrm{IV}_j$:

$$
\max_{S\subseteq V}\sum_{j\in S}w_j\quad\text{s.a. } \{j,k\}\notin E_u\ \ \forall j,k\in S .
$$

Es el problema MWIS (*maximum weight independent set*). El conjunto independiente máximo está entre los problemas NP-completos de Karp (1972). Con 20–60 candidatas se resuelve exacto: por ramificación $\mathrm{MWIS}(G)=\max\{\mathrm{MWIS}(G-v),\,w_v+\mathrm{MWIS}(G-N[v])\}$ con memoria (numpy), o como programa entero $\max w^\top x$, $x_j+x_k\le1\ \forall\{j,k\}\in E_u$, $x\in\{0,1\}^p$ (`scipy.optimize.milp`). El notebook usa ambas y verifica que dan el mismo óptimo.

**Garantía del greedy por peso.** Sea $\Delta$ el grado máximo. Cada vértice de la solución óptima $O$ fue elegido por el greedy o descartado por un vecino elegido antes, que pesa al menos lo mismo. Cada vértice elegido $g$ puede «cubrir» como máximo $\Delta$ vértices de $O$ (sus vecinos, o él mismo si está en $O$). Entonces $w(O)\le\Delta\cdot w(\text{greedy})$. **Para grafos con estrellas (un líder correlacionado con varias candidatas que no se correlacionan entre sí) el greedy puede perder mucho.** El ejercicio 5 muestra un caso.

**Pero el objetivo está mal planteado.** $\sum\mathrm{IV}$ supone que la información es aditiva, y justamente entre variables correlacionadas no lo es. Notebook, Tabla 2:

| umbral | greedy: n / ΣIV / Gini HO / coef+ | MWIS: n / ΣIV / Gini HO / coef+ |
|---|---|---|
| 0,60 | 5 / 1,55 / 0,594 / 0 | 5 / 1,55 / 0,594 / 0 |
| 0,70 | 7 / 1,79 / 0,593 / 0 | 8 / 2,24 / 0,598 / 0 |
| 0,80 | 8 / 2,25 / 0,602 / 0 | 8 / 2,25 / 0,602 / 0 |
| 0,90 | 8 / 2,25 / 0,602 / 0 | 10 / 3,38 / 0,602 / 0 (VIF máx 7,2) |
| 0,95 | 13 / 4,85 / 0,604 / 1 | igual |
| 1,00 (sin filtro) | 16 / 5,88 / 0,602 / 2 (VIF máx 38,7) | igual |

El Gini HO se mueve en 0,593–0,604 entre umbrales, dentro del ruido muestral (el propio techo verdadero difiere 0,06 entre DEV y HO por azar, §8). **El umbral decide *cuáles* y *cuántas*, casi no *cuánto discrimina*.** Con 30 órdenes aleatorios a 0,70 el greedy deja 7 u 8 variables y el Gini HO va de 0,575 a 0,604: el orden importa tanto como el umbral. El MWIS gana en $\sum$IV y empata en Gini. Y para ganar, a 0,70 cambia `uso_linea_prom_12m` (driver) por `uso_linea_prom_3m` (proxy ruidoso), porque este último correlaciona 0,68 con `uso_tc_prom_3m` y así «caben» los dos: optimizar $\sum$IV premia elegir proxies que se parezcan menos a sus vecinos. **El greedy se defiende como heurística transparente y reproducible, no como óptimo**, y la decisión relevante (qué miembro de cada familia representa el concepto) sigue siendo del modelador.

### 3.9 Redundancia condicional, familias y «nivel + delta»

**Correlación parcial.** La redundancia que importa para entrar al modelo es condicional: ¿$x_v$ agrega **dado** lo que ya está ($Z$)? Con $P=R^{-1}$ la matriz de precisión de $(y,x_v,Z)$:

$$
\rho_{yv\cdot Z}=-\frac{P_{yv}}{\sqrt{P_{yy}P_{vv}}},
$$

que equivale a la correlación entre los residuos de regresar $y$ y $x_v$ sobre $Z$ (el notebook verifica ambas). Para el modelo logístico la prueba formal es el **test de razón de verosimilitud** incremental $LR=2(\ell_{Z+v}-\ell_Z)\sim\chi^2_1$ bajo $H_0$. En el notebook, dado el resto y `uso_tc_prom_3m` (líder por IV), `uso_tc_prom_12m` tiene correlación parcial 0,018 y $LR=2{,}0$ ($p=0{,}16$); `uso_tc_prom_6m`, $LR=0{,}6$ ($p=0{,}44$). Su $|\rho|$ marginal con el líder es 0,91 y 0,95, pero lo relevante es que **condicionalmente no agregan**.

**Reemplazo entre ventanas.** Con la familia completa, `uso_tc_prom_12m` sale **+0,415 (p 0,014)**. Sin el líder, entra `uso_tc_prom_6m` con −0,660 (p < 0,001) y 12m sigue positivo (+0,397). El Gini HO no se mueve (0,588 / 0,588 / 0,584). Es lo que advierte la lámina 18.

**Re-parametrización nivel + delta.** Sean $x_{12}$ y $x_3$ la misma serie en dos ventanas y $\Delta=x_3-x_{12}$. En un modelo lineal en los crudos,

$$
\eta=a\,x_{12}+b\,x_3=(a+b)\,x_{12}+b\,\Delta .
$$

Ambas parametrizaciones generan **el mismo espacio y el mismo ajuste**. Lo que cambia es la interpretación y la covarianza de los coeficientes. El «signo equivocado» de $x_{12}$ cuando entra junto a $x_3$ ($a>0$, $b<0$) es simplemente cómo se ve una **tendencia** en la parametrización de ventanas: nivel $a+b<0$ (más uso, más riesgo) y delta $b<0$ (si subió, más riesgo). Además, $\mathrm{Corr}(x_{12},\Delta)=(\rho\sigma_3-\sigma_{12})/\sigma_\Delta$ suele ser chica, mientras $\mathrm{Corr}(x_{12},x_3)$ es ~0,9. El VIF baja y los signos se vuelven firmables.

Con WoE hay una diferencia de fondo: cada variable se binea y transforma por separado, así que $\mathrm{WoE}(x_{12})+\mathrm{WoE}(x_3)$ y $\mathrm{WoE}(x_{12})+\mathrm{WoE}(\Delta)$ **no** generan el mismo espacio. Gana la parametrización cuya estructura aditiva se parece a la verdad. En el generador el riesgo es aditivo en el nivel y en $\max(\Delta,0)$: nivel + delta es la especificación correcta. Resultados (Tabla §6.2 del notebook, base de 5 drivers):

| especificación | $\beta_{12m}$ | $\beta$ otra | % bootstrap $\beta_{12m}>0$ | VIF máx | log-verosim. DEV | AIC | Gini HO |
|---|---|---|---|---|---|---|---|
| solo 12m | −0,225 | — | 0% | 2,57 | −3.051,7 | 6.117,4 | 0,581 |
| solo 3m | — | −0,327 | — | 2,40 | −3.045,2 | 6.104,4 | 0,588 |
| ambas | **+0,184** | −0,467 (3m) | **90%** | 6,47 | −3.044,2 | 6.104,4 | 0,590 |
| nivel + delta | −0,216 | −0,614 (Δ) | 0% | 2,58 | **−3.038,3** | **6.092,6** | 0,590 |

Correlación WoE: 3m–12m 0,911; nivel–delta 0,109. Nivel + delta mejora la log-verosimilitud en 5,9 puntos **con los mismos grados de libertad**, deja ambos signos correctos en el 100% del bootstrap y reduce el VIF a la mitad. El Gini HO es idéntico. **El Gini no decide; decide que el modelo se puede firmar**, y el delta es un reason code legible («su uso de tarjeta subió en los últimos 3 meses»).

**Otras re-parametrizaciones de familia:** ratio corto/largo ($x_3/x_{12}$, útil si el efecto es multiplicativo; cuidado con denominador 0), máximo vs promedio (severidad vs nivel), pendiente de una regresión en la ventana, número de meses sobre un umbral. La elección se hace en la fábrica de variables (Serie 1 · M5) y el embudo la evalúa como cualquier candidata.

### 3.10 Clustering de variables: VARCLUS y el ratio $1-R^2$

`PROC VARCLUS` de SAS es **divisivo**: parte con todas las variables en un cluster, calcula las dos primeras componentes principales del cluster, y lo divide si el segundo autovalor supera un umbral (MAXEIGEN; por defecto 1 con matriz de correlación, según la documentación de SAS: verificar en su versión). Asigna cada variable a la componente (rotada, oblicua) con la que más correlaciona y reasigna iterativamente. Nelson (2001, SUGI 26) popularizó en la industria el criterio de representante:

$$
\text{ratio }1-R^2=\frac{1-R^2_{\text{propio}}}{1-R^2_{\text{vecino}}},
$$

donde $R^2_{\text{propio}}$ es con la componente de su cluster y $R^2_{\text{vecino}}$ con la del cluster más cercano. Se elige el mínimo: la variable que mejor resume lo suyo y menos se parece a lo ajeno. Con la matriz de correlación basta: si el cluster $k$ tiene miembros $m$, autovector principal $v$ y autovalor $\lambda$,

$$
\mathrm{Corr}(x_j,\mathrm{PC}_k)=\frac{R_{j,m}\,v}{\sqrt\lambda}.
$$

El notebook usa la versión **aglomerativa** que se implementa en cualquier stack: distancia $d_{jk}=1-|\rho_{jk}|$ sobre WoE, enlace promedio (UPGMA; implementado desde cero y comparado con `scipy.cluster.hierarchy`). Cortar a altura $h$ con enlace promedio significa que **en promedio** los miembros tienen $|\rho|\ge1-h$, no cada par (eso sería enlace completo, que es lo que emula mejor al greedy). `Hmisc::varclus` en R (Harrell) usa por defecto $\rho^2$ de Spearman y enlace completo (verificar versión).

Resultado al corte 0,30 (≈ $|\rho|$ 0,70): 11 clusters. Mora (3 miembros) y un cluster de **7** que junta `uso_linea` y `uso_tc` (su $|\rho|$ entre familias es 0,70–0,74). Ratio e IV eligen el mismo representante en todos los clusters, y en mora ambos eligen `dias_mora_max_12m` (proxy), no el driver. El ratio $1-R^2$ es **sin target**: elige el centroide. Sirve para ordenar la discusión, no para decidirla.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / norma |
|---|---|---|---|---|
| Filtro de a pares sobre WoE + greedy por IV (curso) | redundancia bivariada evidente | trivial; depende del orden y umbral | siempre, como primer paso auditable | práctica estándar de scorecards (Siddiqi 2017; Anderson 2007) |
| MWIS exacto (ramificación / MILP) | óptimo del mismo objetivo | ms con < 60 variables | para mostrar que el greedy no deja valor evidente sobre la mesa | poco usado; útil en validación |
| VIF (OLS, sobre WoE) | inflación de varianza por colinealidad múltiple | trivial | reportar siempre; esperable ≤ 2–3 tras el filtro | lo espera cualquier validador; umbrales 5/10 son convención |
| VIF ponderado por $W$ | lo mismo en la métrica de la logística | requiere el modelo ajustado | modelo final; PD muy baja o segmentos | literatura GLM (Lesaffre & Marx 1993) |
| Índice de condición + BKW | cuántas dependencias y quién participa | SVD; lectura experta | VIF alto después del filtro; familias grandes | Belsley, Kuh & Welsch (1980); software econométrico |
| GVIF (Fox & Monette) | colinealidad de un grupo de columnas | trivial | scorecards con dummies; evaluar una familia entera | `car::vif` en R |
| Clustering de variables (VARCLUS / jerárquico) | reducir 100–3.000 candidatas a conceptos | bajo | preselección masiva; mapa para el comité | SAS VARCLUS muy extendido en banca |
| Correlación parcial / LR incremental | redundancia condicional | una logística por candidata | familias temporales; forzar variables pedidas por el comité | estándar estadístico |
| Nivel + delta (re-parametrización) | supresión dentro de familias temporales | una variable nueva por familia | cuando dos ventanas sobreviven o hay signo positivo significativo | práctica de modeladores de comportamiento |
| L1 / elastic net sobre WoE | selección y contracción simultáneas | ajuste por camino; elegir λ | exploración, benchmark del stepwise; muchas candidatas | común en ML; en scorecards como apoyo, no como modelo firmado |
| Ridge | estabiliza coeficientes sin seleccionar | sesgo explícito; p-valores no estándar | benchmark; raramente en producción regulada | literatura de colinealidad (Hoerl & Kennard 1970) |
| PCA / PCR / PLS | elimina colinealidad por construcción | pierde interpretabilidad y reason codes | casi nunca en scorecards de admisión; útil para análisis exploratorio o macro | econometría, no scorecards |
| Clusterización conceptual (curso) | cobertura de drivers y gobierno | juicio experto | siempre, sobre el resultado estadístico | práctica y expectativa de comités |

**L1 y elastic net sobre WoE.** El lasso (Tibshirani 1996) resuelve $\min_\beta -\ell(\beta)+\lambda\|\beta\|_1$ y elige variables al anularlas. Con variables muy correlacionadas tiende a quedarse con una de cada grupo, pero **cuál** puede cambiar entre muestras. Elastic net (Zou & Hastie 2005) agrega $\lambda_2\|\beta\|_2^2$ y tiene el «efecto de agrupación»: las correlacionadas entran juntas con coeficientes parecidos. Para scorecards eso es contraproducente (reparte el peso entre proxies). En el notebook, el camino L1 sobre las 16 WoE estandarizadas mete primero `uso_linea_prom_12m` y `dias_mora_max_12m` (de nuevo el proxy de mora antes que el driver), y **tercero `uso_linea_prom_3m`, de la misma familia**. `uso_tc_prom_12m` entra recién con $C\approx0{,}095$. El lasso no reemplaza la decisión de familia. Además, los coeficientes penalizados están sesgados hacia 0 y sus p-valores no son los de siempre: si se usa, es para **seleccionar** y luego re-estimar sin penalizar (relaxed lasso / post-selección), declarando que la inferencia post-selección no es estándar (ver M11).

**Por qué no PCA.** (i) Una componente mezcla conceptos: en el notebook PC1 carga 0,32–0,35 en uso de línea y tarjeta y 0,18 en mora («riesgo general»), y no hay reason code posible. (ii) El signo y la rotación de las componentes son arbitrarios y cambian en cada re-estimación, lo que rompe la trazabilidad del artefacto congelado (M21). (iii) La PCA no mira el target: las componentes de **menor** varianza pueden ser justo las predictivas (Jolliffe 1982). En el notebook, el delta 3m − 12m vive en una dirección de varianza chica. (iv) No hay control de monotonía ni de signo por variable. El Gini puede empatar (6 componentes: Gini HO 0,598 vs 0,593 del greedy), pero empatar en Gini no es el criterio de aprobación.

---

## 5. Cuándo falla: trampas y modos de falla

**T1 · Correlación sobre crudos.** *Síntoma:* dos variables de mora con códigos especiales pasan el filtro juntas; o dos deudas «redundantes» (Pearson 0,99) donde una tiene IV y la otra no. *Causa:* códigos −9/−99/13 y colas dominan el Pearson; la escala no es la del modelo. *Detección:* comparar Pearson crudo vs WoE (Tabla 1); pares con discrepancia > 0,3. *Qué hacer:* filtrar siempre sobre WoE de DEV; usar crudos solo como diagnóstico de la fábrica.

**T2 · El filtro se queda con el proxy, no con el driver.** *Síntoma:* sobrevive `dias_mora_max_12m` (proxy) y cae `meses_desde_mora_12m` (driver), porque el proxy tuvo 0,03 más de IV en DEV. *Causa:* el IV difiere por ruido muestral entre miembros casi equivalentes; el greedy no sabe de causalidad ni de disponibilidad. *Detección:* marcar familias antes de filtrar; mirar el ranking de IV dentro de cada familia con intervalo bootstrap (Serie 1 · E3): si se solapan, el orden es azar. *Qué hacer:* orden de negocio explícito dentro de familia (definición más simple, más estable, disponible en producción) y documentarlo; la lámina 18 lo pide.

**T3 · Dependencia del orden y del umbral no declarada.** *Síntoma:* el mismo pool da 7 u 8 variables según el orden de recorrido; el umbral 0,70 vs 0,80 cambia la familia `uso_tc` completa. *Causa:* el greedy es una heurística de MWIS. *Detección:* correr el greedy con 30 órdenes aleatorios y 3–4 umbrales (notebook, §2); reportar el rango. *Qué hacer:* fijar orden con desempate determinista (IV, luego nombre) en la configuración; reportar sensibilidad en el expediente.

**T4 · VIF «bajo» leído como ausencia de colinealidad.** *Síntoma:* VIF máx 2,97 y el comité concluye «no hay colinealidad». *Causa:* tras el filtro, el VIF de a pares está acotado en 1,96; el VIF por variable no ve cuántas dependencias hay. *Detección:* BKW sobre el modelo final; $\eta_{\max}$ y filas con dos o más $\pi>0{,}5$. *Qué hacer:* reportar VIF + $\eta_{\max}$ + lectura de BKW; explicar que 2,97 > 1,96 implica estructura múltiple leve.

**T5 · VIF mal calculado (sin constante).** *Síntoma:* VIF de 13 donde el correcto es 3 (notebook: `uso_linea_prom_12m` crudo 3,07 → 13,56). *Causa:* `variance_inflation_factor` regresa sobre las columnas tal como se pasan; sin constante ni centrado, $R^2$ no centrado. statsmodels 0.15 introdujo `standardize=True` por defecto; versiones anteriores no centraban (verificar en su entorno). *Detección:* test de CI: `diag(inv(corrcoef(X)))` debe coincidir con la librería. *Qué hacer:* implementar el VIF como $[R^{-1}]_{jj}$ y usar la librería solo como verificación.

**T6 · Signo equivocado significativo tratado como «colinealidad».** *Síntoma:* stepwise mete una variable con $\beta>0$, $p<0{,}01$; se saca y listo. *Causa:* §3.7: la colinealidad sola casi nunca produce volteos significativos (≤ α/2). Si es significativo, hay un efecto condicional real (tendencia, interacción, no linealidad) que el modelo está reconstruyendo con un supresor. *Detección:* identificar con quién se suprime (correlación parcial, BKW), probar la re-parametrización (nivel + delta, ratio). *Qué hacer:* excluir por política (curso) **y** construir la variable que expresa el efecto; medir si se recupera la log-verosimilitud (notebook: +5,9 con nivel + delta).

**T7 · Familia completa en el modelo «porque pasó el filtro».** *Síntoma:* 3m y 12m de la misma serie en el modelo final (Austral lo tiene: `uso_tc_prom_12m` y `_3m`). *Causa:* $|\rho|$ de WoE justo bajo 0,70; el filtro es binario. *Detección:* LR incremental de cada miembro dado el otro; bootstrap del signo; coeficientes en HO. *Qué hacer:* decidir a nivel de familia: un representante, o nivel + delta. Si ambos quedan, justificar con LR y estabilidad de signo, no solo con p-valor en DEV.

**T8 · Leer mal el GVIF de familia.** *Síntoma:* «la familia `uso_tc` tiene GVIF 2,4, no hay problema», con VIF internos de 10–18. *Causa:* el GVIF del grupo contra el resto cancela la redundancia interna. *Detección:* comparar GVIF de familia vs VIF/BKW internos. *Qué hacer:* usar GVIF para colinealidad **entre** conceptos y VIF/BKW para **dentro** del concepto.

**T9 · Redundancia que cambia en el tiempo.** *Síntoma:* dos variables con $|\rho|=0{,}55$ en DEV correlacionan 0,85 en OOT (p. ej., un cambio de política de líneas que alinea uso de línea y de tarjeta). *Causa:* la estructura de correlación es parte de la población y también deriva; el PSI por variable no lo ve. *Detección:* recalcular la matriz de correlación de WoE en OOT/producción y monitorear $\max|\Delta\rho|$ y $\eta_{\max}$ junto al CSI (M08, M20). *Qué hacer:* gatillo de revisión si un par cruza el umbral o $\eta_{\max}$ sube > 50%.

**T10 · Clustering con enlace o corte mal elegidos.** *Síntoma:* un cluster de 7 junta dos conceptos distintos (uso de línea y de tarjeta). *Causa:* enlace promedio o simple permite cadenas; el corte es arbitrario. *Detección:* revisar el dendrograma completo y la $|\rho|$ mínima intra-cluster. *Qué hacer:* enlace completo si se quiere emular el filtro de a pares; separar por concepto de negocio; declarar el corte.

**T11 · Correlaciones estimadas con pocos malos.** *Síntoma:* en carteras chicas el WoE de bins pequeños es ruidoso y las correlaciones de WoE se inflan o desinflan. *Causa:* la varianza de los WoE por bin es ~$1/m_b+1/g_b$; con $m_b<20$ el WoE de cada bin es poco más que ruido. *Detección:* bootstrap de la matriz de correlación (Serie 1 · E3). *Qué hacer:* bins más gruesos, umbral con banda de indiferencia (p. ej., 0,65–0,75 → decisión de negocio).

**T12 · Identidades contables no detectadas.** *Síntoma:* $R$ singular o casi, VIF infinito, `LinAlgError` o coeficientes absurdos. *Causa:* variables que son función exacta de otras (en crédito de motos: valor = pie + monto; pie% = 1 − LTV). *Detección:* $\eta_{\max}$ enorme, rango de $X$ < número de columnas; test de CI con `matrix_rank`. *Qué hacer:* eliminar identidades en la fábrica de variables (contrato de linaje: cada variable declara de qué columnas deriva).

---

## 6. Puente con ingeniería

El paso de redundancia es una **transformación declarativa con estado ajustado en DEV**, como el binning: recibe la tabla de WoE de DEV y una configuración, devuelve una lista de variables y un **registro de decisiones**, y se congela.

**Configuración declarativa (ejemplo):**

```yaml
redundancia:
  medida: pearson_woe            # sobre WoE de DEV; alternativas: pearson_woe_ponderada
  umbral_par: 0.70               # convención del curso; banda de indiferencia 0.65-0.75
  orden:                         # desempate determinista
    - prioridad_familia          # declarada abajo
    - iv_desc
    - nombre_asc
  familias:
    - patron: "^uso_tc_prom_(3|6|12)m$"
      politica: nivel_mas_delta  # nivel = 12m, delta = 3m - 12m
    - patron: "^(meses_desde_mora|dias_mora_max|n_meses_mora)_12m$"
      prioridad: [meses_desde_mora_12m]   # driver de negocio; justificar
      max_miembros: 1
  vif: {max: 5, tipo: ponderado_W, reportar_bkw: true, eta_max: 30}
  identidades_prohibidas: linaje   # rechaza variables con linaje algebraico redundante
```

**Invariantes verificables (tests tipo CI):**

1. `max |ρ_WoE|` entre seleccionadas ≤ `umbral_par` (el notebook lo verifica para cada umbral).
2. $\mathrm{VIF}$ implementado como $[R^{-1}]_{jj}$ coincide con la librería (con constante) a $10^{-8}$.
3. `rank(X_sel) == n_columnas` y $\eta_{\max}$ ≤ umbral declarado.
4. Máximo `max_miembros` por familia, salvo excepción registrada con LR y bootstrap.
5. Coeficientes del modelo final: todos < 0; cualquier $\beta<-1{,}3$ o $\beta>0$ en un modelo intermedio genera una entrada en el registro con la compañera supresora identificada (BKW / correlación parcial).
6. El resultado es **determinista**: mismo input + misma config → misma lista (hash).
7. $\Sigma$IV(greedy) ≤ $\Sigma$IV(MWIS) (sanidad de la implementación) y reporte de la brecha.

**Qué se congela y se versiona:** la matriz de correlación de WoE de DEV (el CSV del módulo es un ejemplo del formato), la lista seleccionada, el registro `variable → descartada_por → ρ → regla`, la configuración y los hashes de los mapas WoE de entrada. El registro responde la pregunta de comité «¿por qué no está la renta?» (clase 3, reserva R) con una línea: «descartada por correlación 0,82 con X» o «LR incremental p = 0,6 dado el modelo».

**Qué se monitorea:** la matriz de correlación de WoE en ventanas de producción (T9): $\max|\rho_t-\rho_{DEV}|$ entre variables del modelo, $\eta_{\max,t}$ y el VIF ponderado con los $\hat p$ del periodo. Se agrega al tablero de M20 como indicador secundario, con gatillo de revisión, no de recalibración.

**Paridad.** Si la implementación de producción recalcula variables (p. ej., el delta 3m − 12m), el delta **se calcula a partir de los crudos** y se binea con los cortes congelados, nunca como diferencia de WoE (M21, «en producción NO existe DEV»).

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy en el notebook | Librería | Diferencias / convención | Producción |
|---|---|---|---|---|
| Pearson / Spearman | producto punto centrado; rangos promedio vectorizados | `scipy.stats.pearsonr`, `spearmanr`; `DataFrame.corr` | manejo de NaN (pairwise vs listwise); empates → rango medio en ambos | pandas/numpy |
| V de Cramér | $\chi^2$ de Pearson sin corrección | `scipy.stats.contingency.association(method="cramer")` | exige tabla entera; sin corrección de Yates por defecto; sin corrección de sesgo de Bergsma | scipy |
| Razón $\eta$ | $SS_\text{entre}/SS_\text{total}$ con `bincount` | $\sqrt{R^2}$ de OLS sobre dummies (statsmodels) | idénticas | cualquiera |
| VIF | `diag(inv(corrcoef(X)))` | `statsmodels…variance_inflation_factor` | la librería necesita constante (o `standardize=True`, ≥ 0.15) | numpy + test contra librería |
| Índice de condición / BKW | SVD de $X$ con intercepto escalada a norma 1 | `np.linalg.cond` (solo el máximo); no hay BKW en statsmodels/scipy (en R: `olsrr`, `mctest`; verificar) | centrar o no cambia los números | numpy |
| GVIF | $\det R_{11}\det R_{22}/\det R$ con `slogdet` | correlaciones canónicas vía `scipy.linalg` (Cholesky + SVD); `car::vif` en R | usar `slogdet` para estabilidad numérica | numpy |
| VIF ponderado | correlación ponderada por $\hat p(1-\hat p)$ | `diag(cov_params())·SS_w` de statsmodels | depende del modelo ajustado | numpy |
| Logística | Newton-Raphson (= IRLS) | `statsmodels.Logit` | mismo MLE; statsmodels reporta EE, p-valores | statsmodels (inferencia) |
| MWIS | ramificación con memoria | `scipy.optimize.milp` | pueden elegir conjuntos distintos con igual objetivo (empates) | milp |
| Clustering jerárquico | UPGMA desde cero | `scipy.cluster.hierarchy.linkage/fcluster` | desempates con distancias iguales | scipy |
| Correlación parcial | $-P_{ij}/\sqrt{P_{ii}P_{jj}}$ | residuos de OLS (statsmodels) | idénticas | numpy |
| L1 | — | `LogisticRegression` (liblinear) | sklearn ≥ 1.8 depreca `penalty="l1"` en favor de `l1_ratio=1`; liblinear penaliza el intercepto (mitigado con `intercept_scaling`) | sklearn, solo para explorar |
| PCA | autovalores de $R$ | `sklearn.decomposition.PCA` | signo de componentes arbitrario | — |

Todas las parejas se verifican con `assert np.allclose` en la celda final del notebook, incluyendo las identidades: $\sum_k\pi_{kj}=1$, $\sum_k\phi_{jk}=[(\tilde X^\top\tilde X)^{-1}]_{jj}$, $\kappa(R)\ge\mathrm{VIF}_{\max}$, GVIF de una columna = VIF, y que L1 con $C\to\infty$ reproduce la log-verosimilitud del MLE.

---

## 8. Aplicación: casos y números

### 8.1 El laboratorio con verdad conocida (notebook)

Cartera `generar_cartera()` + `ampliar_pool()`: DEV 10.065, HO 4.295, OOT 4.792; malos DEV 11,3%. 20 candidatas, de las cuales 9 son drivers verdaderos y 9 son proxies construidos (otra ventana, otro agregador, identidad contable). Pool de trabajo: IV ≥ 0,02 → 16. Se usa 0,02 y no 0,10 para no perder `antiguedad_meses` (driver, IV 0,088) y no mezclar dos filtros.

**Techo de discriminación:** Gini de la PD verdadera DEV 0,566 · HO 0,627 · OOT 0,593. El techo de DEV es menor que el de HO con esta semilla. Por eso los modelos dan Gini DEV < HO sin que signifique nada, y por eso diferencias de ±0,01 entre estrategias de selección son ruido.

Hallazgos (todos reproducibles en el notebook):

1. **El greedy del curso a 0,70** deja 7 variables, solo 3 drivers verdaderos. Descarta **toda** la familia `uso_tc` (choca con `uso_linea_prom_12m` a $|\rho|$ 0,72–0,74) y cambia `meses_desde_mora_12m` por `dias_mora_max_12m` y `carga_financiera` por `ratio_deuda_renta`. VIF máx 1,95; Gini HO 0,593. A 0,80 recupera `uso_tc_prom_3m` (Gini HO 0,602).
2. **Sin filtro** (16 variables): VIF máx 38,7; `uso_tc_prom_12m` **+0,459 (p 0,007)**; el driver `uso_linea_prom_12m` queda en −0,06 (p 0,71) mientras su proxy ruidoso `uso_linea_prom_3m` se lleva −0,35 (p 0,001); el driver `meses_desde_mora_12m` −0,01 (p 0,94). **La colinealidad no solo infla varianzas: reparte el mérito arbitrariamente entre causa y proxies.** Gini HO 0,602: igual.
3. **BKW** separa las dependencias del pool en familias reconocibles (η 23,5 / 17,7 / 14,8 / 12,9) y, en el juguete, dos dependencias que el filtro de a pares deja pasar ($|\rho|_{\max}$ 0,697; VIF hasta 409).
4. **Familia `uso_tc`:** ambas ventanas → signo positivo en 90% del bootstrap; nivel + delta → signos correctos, +5,9 de log-verosimilitud, VIF 2,58.

### 8.2 Banco Austral (curso)

- **63 → 26 con 35 pares > 0,90.** Con 63 candidatas hay 1.953 pares; 35 sobre 0,90 es típico de una fábrica que genera ventanas 3/6/12 y agregadores prom/max. En el pool del notebook (20 candidatas, 190 pares) hay 10 pares > 0,90 y 21 > 0,70.
- **VIF máx 2,97.** Tras el filtro, un par solo puede aportar hasta VIF 1,96; el 2,97 ($R^2=0{,}66$) viene de combinaciones de 3+ variables. Leve, pero no «nada». La lectura correcta para el comité: «la colinealidad múltiple residual infla los errores estándar hasta 1,72×; no amerita descartar».
- **`uso_tc_prom_12m` y `uso_tc_prom_3m` juntas en el modelo final**, ambas negativas y significativas en DEV. Es el caso de la lámina 18. Preguntas de validación: (a) LR incremental de 3m dado 12m y el resto; (b) bootstrap del signo de cada una; (c) especificación nivel + delta y comparación de log-verosimilitud a igual número de parámetros; (d) coeficientes en HO. Con 78 malos en HO, (d) solo contrasta. Si nivel + delta mejora la verosimilitud o estabiliza signos, es la especificación a firmar. Si no, queda documentado que se probó.
- **Volteos en HO de `deuda_interna_max_3m` y `deuda_otras_prom_12m`** a ≈ 0 con p 0,96 y 0,78: consistente con §3.7. Con 78 malos, la varianza de $\hat\beta$ en HO es ~2,4 veces la de DEV (por tamaño) más la inflación VIF; un volteo no significativo es exactamente lo esperable bajo $H_0$: «el efecto es el de DEV».
- **Experimento 3 de optbinning** (66 variables, Gini 0,805 / 0,657 / 0,677): cuando se sacan los filtros de redundancia y parsimonia, DEV sube y HO baja. En el notebook, 16 correlacionadas vs 7 filtradas empatan en HO; en Austral con 66, HO pierde 4 puntos. Más variables no compra Gini; compra varianza.

### 8.3 Crédito de motos (aplicación a Galgo)

En financiamiento de motos, la fábrica de variables genera familias e **identidades contables** que no aparecen en consumo tradicional:

- **Identidad exacta:** `valor_moto = pie + monto_financiado`. Si las tres entran como candidatas, $R$ (en crudos) es singular; en WoE (cada una bineada por separado) la singularidad se vuelve casi-singularidad, con η muy alto y VIF enormes. `pie_pct = 1 − LTV` es una identidad de a pares: Pearson crudo −1, **Pearson WoE ≈ +1**, porque el WoE orienta ambas hacia «bueno». El filtro de a pares las caza. La triple identidad (valor, pie, monto) no siempre: cada par puede estar bajo 0,70 y la tríada ser exactamente colineal. Es el caso de BKW.
- **Familia de la cuota:** cuota ≈ f(monto, plazo, tasa). `carga = cuota / renta` y `monto / renta` son casi colineales si el plazo y la tasa varían poco en la cartera. Si el producto tiene pocos plazos, `plazo` actúa como segmentador más que como variable continua.
- **Familias temporales de comportamiento** (renovación, segunda moto, cross-sell): `mora_max_3m/6m/12m` de la cuota de la moto. La lámina 18 aplica literal. La re-parametrización recomendada es nivel (12m) + deterioro reciente (3m − 12m, o indicador «mora en últimos 3m sin mora previa»).
- **Proxy de segmento:** `cilindrada`, `marca` y `valor_moto` correlacionan con el perfil del cliente (edad, renta) y con el canal (concesionario). Un signo positivo significativo de `valor_moto` junto a `LTV` es típicamente supresión: a igual LTV, motos más caras = clientes con más renta. El remedio es modelar la capacidad directamente (renta, carga), no dejar que el valor de la moto la reconstruya.

Regla práctica para el pipeline de motos: declarar en la configuración las identidades (`linaje`) y rechazar en CI cualquier selección cuya matriz de WoE tenga $\eta_{\max}>30$ o rango deficiente.

---

## 9. Preguntas de comité

**1. «El VIF máximo es 2,97. ¿Por qué me lo reportan si no descartó nada?»**
Porque el filtro de a pares no garantiza ausencia de colinealidad múltiple: una variable puede ser combinación de otras tres sin correlacionar > 0,70 con ninguna. Después del filtro, un par solo aporta hasta 1,96. Un VIF de 2,97 indica estructura múltiple leve: errores estándar 1,72× los de un diseño ortogonal. Lo reportamos con el índice de condición y la descomposición de BKW para mostrar que no hay una dependencia concentrada.

**2. «Si el umbral fuera 0,80 en vez de 0,70, ¿el modelo sería otro?»**
Posiblemente en composición, no en discriminación. En nuestro análisis de sensibilidad (umbrales 0,6–0,95 y 30 órdenes de recorrido), el Gini HO se mantuvo dentro de ±0,015, que es el ruido muestral. Lo que cambia es qué representante queda por familia. Esa elección la fijamos con criterio de negocio (definición, disponibilidad, estabilidad) y está en el registro de decisiones.

**3. «Hay dos ventanas de uso de tarjeta en el modelo. ¿No es la misma variable dos veces?»**
Lo probamos a nivel de familia: LR incremental de 3m dado 12m, bootstrap de signos y la re-parametrización nivel + delta. Si el delta agrega verosimilitud con signos estables, se firma nivel + delta: el reason code es «su uso subió en los últimos 3 meses». Si no agrega, queda un solo representante. Dos ventanas con signos estables y LR significativo se pueden mantener, pero con esa evidencia documentada.

**4. «Un coeficiente salió positivo y muy significativo en una corrida. ¿Colinealidad?»**
No solo eso. La colinealidad infla la varianza pero no sesga: un volteo *significativo* por puro ruido ocurre a lo más con probabilidad α/2. Si es significativo, el modelo encontró un efecto condicional real (típicamente una tendencia o una diferencia entre dos variables) y lo está construyendo con un supresor. Por política la variable se excluye (no se puede firmar un atributo peor que da más puntos), y además construimos la variable que expresa ese efecto de forma directa.

**5. «¿Por qué no usan PCA o lasso, que resuelven la colinealidad automáticamente?»**
La PCA produce componentes que mezclan conceptos, sin reason codes, con signos arbitrarios que cambian en cada re-estimación, y sin mirar el target. El lasso lo usamos como benchmark de selección. No lo firmamos: dentro de familias elige arbitrariamente (en nuestra prueba eligió un proxy de mora antes que el driver) y sus coeficientes están contraídos. Ninguno reemplaza la decisión de familia ni la cobertura de drivers.

**6. «El comité quiere la renta y el stepwise la dejó fuera. ¿Por qué?»**
Mostramos el registro: si fue descartada por correlación, qué variable del modelo carga su información y con qué $\rho$. Si fue por aporte, su LR incremental dado el modelo. Si el comité igual la quiere, se fuerza y se mide el costo (típicamente centésimas de Gini y, si es redundante, un coeficiente inestable que hay que vigilar), documentado (clase 3, reserva R).

**7. «¿La matriz de correlación de DEV sigue valiendo en producción?»**
Es parte de la población y puede derivar sin que el PSI de cada variable lo detecte. Monitoreamos la correlación de WoE entre las variables del modelo y el índice de condición en cada ventana, con gatillo de revisión si un par cruza el umbral o el índice sube materialmente.

**8. «¿Qué es exactamente lo que el umbral 0,70 garantiza?»**
Que ningún par de variables del modelo comparte más de 49% de varianza lineal en la escala de WoE, y que la inflación atribuible a cualquier par aislado es ≤ 1,96. Es una convención de práctica sin base inferencial. No garantiza nada sobre combinaciones de 3+ variables ni sobre redundancia condicional; para eso están VIF/BKW y los tests de familia.

---

## 10. Ejercicios

**E1 (cálculo a mano).** Tres WoE con $R=\begin{pmatrix}1&0{,}6&0{,}5\\0{,}6&1&0{,}65\\0{,}5&0{,}65&1\end{pmatrix}$. (a) Calcula los tres VIF. (b) ¿Pasan el filtro de a pares a 0,70? (c) Verifica que $\kappa(R)\ge\mathrm{VIF}_{\max}$.

<details><summary>Solución</summary>

(a) $\det R=1-0{,}36-0{,}25-0{,}4225+2(0{,}6)(0{,}5)(0{,}65)=0{,}3575$. Cofactores diagonales: $C_{11}=1-0{,}4225=0{,}5775$; $C_{22}=1-0{,}25=0{,}75$; $C_{33}=1-0{,}36=0{,}64$. $\mathrm{VIF}=C_{jj}/\det R$ = **1,615; 2,098; 1,790**.
(b) Sí: el máximo $|\rho|$ es 0,65. Aun así el VIF de $x_2$ es 2,10 > 1,96: hay algo de estructura múltiple ($x_2$ se explica 52% por $x_1$ y $x_3$ juntas, vs 42% por $x_3$ sola).
(c) Autovalores: 0,327; 0,504; 2,169 → $\kappa=6{,}64\ge2{,}10$. ✓
</details>

**E2 (derivación).** Demuestra que $\mathrm{VIF}_j\ge1/(1-\rho_{jk}^2)$ para todo $k\ne j$, y concluye que tras un filtro de a pares a umbral $u$, un VIF mayor que $1/(1-u^2)$ implica colinealidad múltiple.

<details><summary>Solución</summary>

$R_j^2$ es el máximo de $\mathrm{Corr}^2(x_j,\,X_{-j}\gamma)$ sobre todos los $\gamma$; tomando $\gamma=e_k$ se obtiene $\rho_{jk}^2$. Entonces $R_j^2\ge\rho_{jk}^2$ y $1/(1-R_j^2)\ge1/(1-\rho_{jk}^2)$. Tras el filtro, cada par cumple $\rho_{jk}^2\le u^2$, así que un solo par puede explicar a lo más $R^2=u^2$. Si $\mathrm{VIF}_j>1/(1-u^2)$, entonces $R_j^2>u^2$, lo que ningún regresor individual puede lograr: la explicación requiere dos o más. Con $u=0{,}70$: 1,96.
</details>

**E3 (supresión).** $r_1=0{,}40$, $r_2=0{,}25$, $\rho=0{,}75$ (escala estandarizada, ambas orientadas a «bueno»). (a) Calcula $\beta_1,\beta_2,R^2$. (b) ¿Qué variable «se voltea» y por qué? (c) ¿Cuánto se infla la varianza de cada coeficiente?

<details><summary>Solución</summary>

(a) $1-\rho^2=0{,}4375$. $\beta_1=(0{,}40-0{,}1875)/0{,}4375=0{,}486$; $\beta_2=(0{,}25-0{,}30)/0{,}4375=-0{,}114$; $R^2=(0{,}16+0{,}0625-0{,}15)/0{,}4375=0{,}166$.
(b) $x_2$, porque $r_2=0{,}25<\rho r_1=0{,}30$. Simultáneamente $\beta_1=0{,}486>r_1=0{,}40$ (amplificación ⇔ volteo). $x_2$ agrega apenas 0,006 de $R^2$ sobre $x_1$ sola (0,160).
(c) VIF = $1/0{,}4375=2{,}29$ para ambas; EE ×1,51. En WoE, $x_2$ saldría con coeficiente positivo; si además fuera significativo, significaría que el efecto condicional de $x_2$ realmente es opuesto.
</details>

**E4 (GVIF).** Con $R=\begin{pmatrix}1&0{,}3&0{,}5\\0{,}3&1&0{,}4\\0{,}5&0{,}4&1\end{pmatrix}$, calcula el GVIF del grupo $\{x_1,x_2\}$ contra $x_3$ y compáralo con el VIF de $x_3$. Explica la coincidencia.

<details><summary>Solución</summary>

$\det R_{11}=1-0{,}09=0{,}91$; $\det R_{22}=1$; $\det R=1-0{,}09-0{,}25-0{,}16+2(0{,}3)(0{,}5)(0{,}4)=0{,}62$. GVIF $=0{,}91/0{,}62=1{,}468$; $\mathrm{GVIF}^{1/4}=1{,}101$. El VIF de $x_3$ es $[R^{-1}]_{33}=\det R_{11}/\det R=1{,}468$: coinciden porque la fórmula $\det R_{11}\det R_{22}/\det R$ es simétrica en los dos bloques. El GVIF del grupo contra el resto es igual al del resto contra el grupo, y cuando el resto es una sola columna, es su VIF. También: la correlación canónica es $\rho_c^2=1-1/1{,}468=0{,}319$, el $R^2$ de $x_3$ sobre $(x_1,x_2)$.
</details>

**E5 (greedy vs óptimo).** Cuatro candidatas: A (IV 0,50), B (0,45), C (0,40), D (0,30). $|\rho|$: A–B 0,80, A–C 0,78; el resto < 0,3. Umbral 0,70. (a) Resultado del greedy por IV. (b) MWIS. (c) Verifica la cota $w(O)\le\Delta\,w(\text{greedy})$. (d) ¿Cuál elegirías y qué más necesitarías saber?

<details><summary>Solución</summary>

(a) A entra; B y C chocan con A; D entra: {A, D}, ΣIV 0,80. (b) {B, C, D}: 1,15 (A excluye a B y C). (c) $\Delta=2$ (A tiene grado 2): $1{,}15\le2\times0{,}80=1{,}60$ ✓. (d) ΣIV no es aditiva: si B y C son dos mitades del concepto de A (p. ej., A = uso total, B = uso línea, C = uso tarjeta), {B, C, D} puede tener más información pero dos coeficientes que el comité debe entender. Hace falta: LR de {A, D} vs {B, C, D}, correlación B–C *condicional* a D, estabilidad en HO y el rol de negocio. La respuesta no sale del grafo.
</details>

**E6 (V de Cramér).** Dos variables categóricas independientes con 10 y 10 niveles, $n=500$. (a) ¿Qué V de Cramér esperas por puro azar? (b) ¿Y con 5×5 y $n=5000$? (c) ¿Por qué esto importa para usar V como filtro de redundancia?

<details><summary>Solución</summary>

(a) $E[\chi^2]\approx(r-1)(c-1)=81$ → $V\approx\sqrt{81/(500\cdot9)}=0{,}134$. (b) $\sqrt{16/(5000\cdot4)}=0{,}028$. (c) El piso de ruido depende de $n$ y del número de niveles: un umbral fijo de V significa cosas distintas en carteras chicas y grandes (el mismo problema del PSI en M08). Usar la corrección de Bergsma (2013) o comparar contra la distribución nula por permutación.
</details>

**E7 (nivel + delta).** El modelo con ambas ventanas da $\hat\beta_{12}=+0{,}18$, $\hat\beta_3=-0{,}47$ (crudos estandarizados, modelo lineal en log-odds). (a) ¿Qué coeficientes tendría el modelo equivalente en (nivel = 12m, $\Delta$ = 3m − 12m)? (b) ¿Por qué con WoE la equivalencia es solo aproximada? (c) ¿Qué reason codes produce cada parametrización?

<details><summary>Solución</summary>

(a) $\eta=a x_{12}+b x_3=(a+b)x_{12}+b\Delta$ → nivel $-0{,}29$, delta $-0{,}47$. Ambos negativos: la «inversión» era la forma de expresar una tendencia. (Si las variables están estandarizadas por separado, hay que reescalar $b$ por $\sigma_\Delta/\sigma_3$; la lógica de signos no cambia.) (b) Porque cada variable se binea por separado: $\mathrm{WoE}(x_{12})+\mathrm{WoE}(\Delta)$ genera funciones aditivas en $(x_{12},\Delta)$, distintas de las aditivas en $(x_{12},x_3)$. Si la verdad es aditiva en nivel y tendencia (como en el generador), nivel + delta ajusta mejor (notebook: +5,9 de log-verosimilitud). (c) Ambas: «uso 12m alto → menos puntos» *y* «uso 3m alto → menos puntos» pero «uso 12m alto → **más** puntos, dado el de 3m», un reason code imposible de explicar. Nivel + delta: «uso alto» y «uso en aumento», ambos legibles.
</details>

**E8 (diseño).** Diseña el test de CI que falla si una selección de variables viola la política de redundancia. Incluye al menos: umbral de a pares, VIF, índice de condición, política de familias y determinismo.

<details><summary>Solución</summary>

```python
def test_politica_redundancia(W_dev, seleccion, config, registro):
    C = W_dev[seleccion].corr().abs().values
    np.fill_diagonal(C, 0)
    assert C.max() <= config["umbral_par"] + 1e-12
    vif = np.diag(np.linalg.inv(np.corrcoef(W_dev[seleccion].values, rowvar=False)))
    assert vif.max() <= config["vif"]["max"]
    X = np.column_stack([np.ones(len(W_dev)), W_dev[seleccion].values])
    Xs = X / np.linalg.norm(X, axis=0)
    assert np.linalg.cond(Xs) <= config["vif"]["eta_max"]
    assert np.linalg.matrix_rank(X) == X.shape[1]
    for fam in config["familias"]:
        miembros = [v for v in seleccion if re.match(fam["patron"], v)]
        assert len(miembros) <= fam.get("max_miembros", 1) or registro.tiene_excepcion(fam)
    assert hash_seleccion(ejecutar_embudo(W_dev, config)) == hash_seleccion(seleccion)
```
Se agrega una prueba de identidad de la implementación (VIF numpy = statsmodels con constante) y el registro de decisiones como artefacto versionado.
</details>

**E9 (código, notebook).** En la sección 4 del notebook, fija $\rho=0{,}9$ y mueve $\beta_2$ de −0,4 a +0,4. (a) ¿Para qué valores la proporción de «positivo y significativo» supera 5%? (b) Explica por qué con $\beta_2=0$ se estabiliza en ≈ 2,5% para cualquier $\rho$.

<details><summary>Solución</summary>

(a) Solo para $\beta_2>0$ (con $n=2000$ y $\rho=0{,}9$, cerca de 9% en +0,1 y 23,5% en +0,2; más con $\rho$ menor, porque la potencia sube). Para $\beta_2\le0$ se mantiene ≤ 2,5%. (b) Con $\beta_2=0$, $\hat\beta_2/\mathrm{EE}\sim N(0,1)$ asintóticamente sea cual sea $\rho$: la colinealidad escala el numerador y el denominador por igual. La probabilidad de $z>1{,}96$ es 2,5% = α/2. La colinealidad no fabrica significancia; fabrica volteos no significativos.
</details>

**E10 (derivación, logística).** Muestra que en la logística univariada sobre WoE sin suavizado, $\hat\beta_0=\ln(M/G)$ y $\hat\beta_1=-1$, y explica por qué en el modelo multivariado un $\hat\beta<-1$ no implica necesariamente supresión.

<details><summary>Solución</summary>

Ver §3.1: con esos valores la probabilidad ajustada de cada bin es su tasa empírica de malos; las ecuaciones de score se cumplen bin a bin; la concavidad estricta da unicidad. En el multivariado, agregar predictores **no correlacionados** hace crecer en magnitud los coeficientes de la logística (no-colapsabilidad: el odds ratio marginal está atenuado respecto del condicional). Así, $\hat\beta<-1$ aparece sin supresión (notebook: `antiguedad_meses` −1,14 con correlación ≈ 0 con todo). La supresión se reconoce por el par: amplificación de una **y** volteo o debilitamiento anómalo de una compañera correlacionada.
</details>

---

## 11. Referencias

- **Belsley, D. A., Kuh, E. & Welsch, R. E. (1980).** *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity.* Wiley. — Origen del índice de condición y de la descomposición de proporciones de varianza (cap. 3); la fuente de «η > 30».
- **Belsley, D. A. (1991).** *Conditioning Diagnostics: Collinearity and Weak Data in Regression.* Wiley. — Versión ampliada y la defensa detallada de no centrar.
- **Fox, J. & Monette, G. (1992).** Generalized collinearity diagnostics. *JASA*, 87(417), 178–183. — El GVIF y su invariancia a la parametrización del grupo; implementado en `car::vif`.
- **O'Brien, R. M. (2007).** A caution regarding rules of thumb for variance inflation factors. *Quality & Quantity*, 41(5), 673–690. — Por qué VIF 5/10 no son umbrales inferenciales; qué más mirar.
- **Marquardt, D. W. (1970).** Generalized inverses, ridge regression, biased linear estimation, and nonlinear estimation. *Technometrics*, 12(3), 591–612. — Frecuentemente citado como origen del VIF 10 (verificar la atribución exacta en el texto).
- **Lesaffre, E. & Marx, B. D. (1993).** Collinearity in generalized linear regression. *Communications in Statistics – Theory and Methods*, 22(7), 1933–1952. — Colinealidad en GLM a través de $X^\top WX$.
- **Segerstedt, B. & Nyquist, H. (1992).** On the conditioning problem in generalized linear models. *Journal of Applied Statistics*, 19(4), 513–526. — Mal condicionamiento en GLM; complementa el VIF ponderado.
- **Conger, A. J. (1974).** A revised definition for suppressor variables: a guide to their identification and interpretation. *Educational and Psychological Measurement*, 34(1), 35–46. — Taxonomía de supresión (clásica, negativa, recíproca). El concepto se remonta a Horst (1941).
- **Friedman, L. & Wall, M. (2005).** Graphical views of suppression and multicollinearity in multiple linear regression. *The American Statistician*, 59(2), 127–136. — La geometría del caso de 2 regresores de §3.7, con gráficos.
- **Gail, M. H., Wieand, S. & Piantadosi, S. (1984).** Biased estimates of treatment effect in randomized experiments with nonlinear regressions and omitted covariates. *Biometrika*, 71(3), 431–444. — No-colapsabilidad: por qué $\beta<-1$ aparece sin supresión.
- **Mood, C. (2010).** Logistic regression: why we cannot do what we think we can do, and what we can do about it. *European Sociological Review*, 26(1), 67–82. — Lectura aplicada sobre comparar coeficientes logísticos entre modelos.
- **Bergsma, W. (2013).** A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society*, 42(3), 323–328. — Corrección del piso de ruido de V.
- **Nelson, B. D. (2001).** Variable reduction for modeling using PROC VARCLUS. *SUGI 26*, paper 261-26. — El uso industrial de VARCLUS y del ratio $1-R^2$.
- **SAS Institute.** *SAS/STAT User's Guide: The VARCLUS Procedure* (verificar edición). — Algoritmo divisivo, criterios de división y salidas.
- **Harrell, F. E. (2015).** *Regression Modeling Strategies*, 2.ª ed. Springer. — Clustering de variables (`varclus`) y reducción de datos sin mirar el target antes de modelar.
- **Jolliffe, I. T. (1982).** A note on the use of principal components in regression. *Applied Statistics*, 31(3), 300–303. — Ejemplos de componentes de baja varianza que son las predictivas: argumento contra PCR.
- **Tibshirani, R. (1996).** Regression shrinkage and selection via the lasso. *JRSS B*, 58(1), 267–288. — L1.
- **Zou, H. & Hastie, T. (2005).** Regularization and variable selection via the elastic net. *JRSS B*, 67(2), 301–320. — Efecto de agrupación entre correlacionadas.
- **Hoerl, A. E. & Kennard, R. W. (1970).** Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. — La respuesta clásica a la colinealidad por contracción.
- **Karp, R. M. (1972).** Reducibility among combinatorial problems. En *Complexity of Computer Computations*, Plenum. — NP-completitud de conjunto independiente/clique: por qué el greedy es heurística.
- **Siddiqi, N. (2017).** *Intelligent Credit Scoring*, 2.ª ed. Wiley. — Práctica de industria del embudo de variables y de la agrupación conceptual.
- **Anderson, R. (2007).** *The Credit Scoring Toolkit.* Oxford University Press. — Selección de características y colinealidad en scorecards, con perspectiva de gobierno.
- **Thomas, L. C., Edelman, D. B. & Crook, J. N. (2002; 2.ª ed. 2017).** *Credit Scoring and Its Applications.* SIAM. — Marco estadístico general de la scorecard.

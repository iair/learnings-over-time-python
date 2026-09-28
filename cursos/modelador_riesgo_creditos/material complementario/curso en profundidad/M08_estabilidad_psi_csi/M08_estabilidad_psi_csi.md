# M08 · Estabilidad poblacional: PSI, CSI y el orden del embudo

> **Ficha**
> - **Clases que profundiza:** clase 3 (PSI primero, embudo 108 → 94, «PSI no medible», variable categórica plantada en Andes, OOT vs TTD) y clase 5 (PSI del score sobre las 8 bandas de la master scale, CSI como canario, DEV congelado vs ventana móvil).
> - **Prerrequisitos:** Serie 1 · M2 (t₀ y ventanas), M4 (esquema muestral DEV/HO/OOT/TTD), M6 (masas puntuales y códigos especiales), M7 (binning y WoE). Estadística: KL, χ² de Pearson, teorema central del límite multivariado.
> - **Archivos del módulo:** `M08_estabilidad_psi_csi.md` (este documento) · `M08_estabilidad_psi_csi.py` (notebook Marimo: `marimo edit --sandbox M08_estabilidad_psi_csi.py`) · `M08_calculadora_psi.xlsx` (calculadora de PSI con p-valor y semáforos).
> - **Tiempo estimado:** 3–4 h (lectura 1,5 h; notebook 1,5 h; ejercicios 1 h).

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**Clase 3.** El embudo del Banco Austral partió de 108 candidatas y el primer filtro fue estabilidad, no poder: «una variable potentísima que se mueve es una bomba de tiempo en producción». El `psi()` de clase calcula cortes por **deciles de DEV**, los aplica a la muestra actual, suma $\varepsilon = 10^{-4}$ a las proporciones y devuelve $\sum_b (a_b - e_b)\ln(a_b/e_b)$. Se midió dos veces: contra OOT (¿cambió dentro de la ventana de desarrollo?) y contra TTD (¿cambió respecto de quién llega hoy?). El resultado: **9 variables con PSI DEV→TTD > 0,25** (familias `n_productos`, `pagos`, `facturacion`, `saldo_consumo`), ninguna con IV sobre 0,096; y **5 variables «no medibles»** (`dias_mora_ult`, `n_meses_mora_3m`, `dias_mora_prom_3m`, `dias_mora_max_3m`, `peor_mora_sistema_ult`) con ~90% de ceros, cuyos deciles colapsan y el PSI da `NaN`, pese a IV de 0,49 a 0,58. Regla del curso: **sin PSI medible no hay certificado de estabilidad → no siguen**. Embudo 108 → 94. El caso didáctico fue `n_productos_prom_3m`: PSI 0,000 contra OOT y 0,386 contra TTD; el cambio era posterior a la ventana de desarrollo y solo el TTD lo veía. En el score final, el PSI por deciles fue 0,003 (HO), 0,005 (OOT) y 0,015 (TTD). En el Lab 2 de Financiera Andes había una variable **categórica** plantada que debía caer en este paso.

**Clase 5.** El PSI del score se calculó sobre las **8 bandas de la master scale** («los mismos cortes con que se gobierna»): DEV→OOT 0,002, DEV→TTD 0,013, verde, pero con dirección (A1 se vacía de 21,43% a 18,43%, aporte 0,0045; E se llena de 12,25% a 14,47%, aporte 0,0037). El **CSI** (el mismo cálculo sobre los bins WoE de cada variable) mostró el «canario»: `deuda_interna_max_3m` con CSI DEV→TTD **0,138** 🟡 mientras el score seguía en 0,013, porque «el resto compensa». Se mencionó que la referencia puede ser **DEV congelado** (elección del curso, auditable) o una **ventana móvil**, y que los bins se congelan con el modelo. En el tablero, «CSI máximo DEV→TTD» es un indicador mensual con umbrales 0,10 / 0,25.

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **Los umbrales 0,10 / 0,25 son convención** (Siddiqi), sin error tipo I ni tipo II asociado. No dependen de $n$ ni del número de bins $B$, y deberían depender de ambos.
2. **No se reportó ningún p-valor.** El 0,013 «verde» del Austral es estadísticamente significativo (p ≈ 6·10⁻⁵; §8). El semáforo fijo mezcla dos preguntas: *¿hubo cambio?* y *¿importa?*
3. **El PSI no se presentó como lo que es**: la divergencia de Jeffreys, pariente directo del χ² de homogeneidad. De ahí sale todo lo demás.
4. **El ε es arbitrario** y, con bins vacíos, el aporte de ese bin depende de $\ln\varepsilon$.
5. **El `NaN` de las moras cortas es un artefacto del binning por deciles**, no una propiedad de los datos. El propio `binear()` del curso (bin de moda) las mide sin problema.
6. **El CSI no tiene signo** ni peso: no dice hacia dónde se movió la variable ni cuánto le importa al score. La «variante de industria» de ponderar por puntos quedó mencionada, no desarrollada.
7. **Qué tipo de cambio ve el PSI**: solo cambios en $p(x)$. El deterioro de $p(y|x)$ (concept drift) es invisible.
8. **Multiplicidad**: 108 variables a la vez generan ~5 falsas alarmas al 5% aunque nada cambie.
9. **DEV congelado vs ventana móvil** se mencionó como elección documentable, sin mostrar qué detecta cada una.

---

## 2. Intuición

El PSI responde una pregunta sin mirar el target: *¿la gente que llega hoy se distribuye igual que la gente con que se estimó el modelo?* Por eso se puede calcular sobre la bandeja TTD el mismo día, doce meses antes de que haya desempeño. Es el indicador más adelantado del tablero y, a la vez, el más limitado: dice que la población cambió, no que el modelo esté fallando.

Tres ideas organizan el módulo.

**Primera: el PSI es una distancia entre histogramas, y todo histograma tiene ruido.** Aunque DEV y TTD vinieran exactamente de la misma población, sus proporciones por bin diferirían por azar, y el PSI sería positivo. ¿Cuánto? Aproximadamente $(B-1)(1/n_e + 1/n_a)$. Con 10 bins y 300 casos por muestra eso da 0,06, y el percentil 95 es 0,113: el umbral 0,10 dispara en ~9% de los meses sin que haya pasado nada. Con 30.000 casos por muestra el percentil 95 es 0,0011: el 0,10 queda 90 veces por encima y cambios reales, detectables, pasan como «verdes». Un umbral fijo no puede servir para ambos casos.

**Segunda: el PSI solo ve $p(x)$.** Si la macro empeora y los mismos perfiles caen más (concept drift), el PSI no se mueve. Si cambia una variable que el modelo no usa pero que afecta el riesgo (el canal de originación, por ejemplo), para el modelo eso también es un cambio de $p(y|x)$ que el PSI de sus variables no ve. Un PSI verde certifica que la población se parece, no que el modelo esté sano.

**Tercera: el score agrega, y al agregar puede cancelar.** Si una variable empuja el score hacia abajo y otra hacia arriba, el PSI del score queda plano mientras dos CSI se encienden. Esto no es una curiosidad: es exactamente el caso `deuda_interna_max_3m` de la clase 5. Por eso se miran las dos capas, y por eso conviene agregar una tercera: el corrimiento **en puntos**, con signo, por variable.

Sobre el orden del embudo (estabilidad antes que poder): la lógica es que el IV de una variable inestable describe una población que ya no existe. Pero el orden tiene un costo que el curso no discutió: el filtro de estabilidad descarta sin mirar cuánto le importaría esa inestabilidad al score. Una variable con PSI 0,30 y un rango de 5 puntos en el scorecard es menos peligrosa que una con PSI 0,12 y un rango de 60. La versión madura del embudo usa la estabilidad como filtro para lo grosero y como **penalización ponderada por impacto** para lo fino (§4).

---

## 3. Formalización

Notación: $B$ bins fijados en DEV; $e_b$ y $a_b$ proporciones esperada (DEV) y actual por bin, $\sum_b e_b = \sum_b a_b = 1$; $n_e$, $n_a$ tamaños muestrales; $\delta_b = a_b - e_b$; $m_b = (a_b + e_b)/2$.

### 3.1 El PSI es la divergencia de Jeffreys

$$\text{PSI} = \sum_{b=1}^{B} (a_b - e_b)\ln\frac{a_b}{e_b}.$$

Separando el factor $(a_b - e_b)$:

$$(a_b - e_b)\ln\frac{a_b}{e_b} = a_b\ln\frac{a_b}{e_b} - e_b\ln\frac{a_b}{e_b} = a_b\ln\frac{a_b}{e_b} + e_b\ln\frac{e_b}{a_b}.$$

Sumando sobre $b$:

$$\text{PSI} = \underbrace{\sum_b a_b\ln\frac{a_b}{e_b}}_{\mathrm{KL}(a\|e)} + \underbrace{\sum_b e_b\ln\frac{e_b}{a_b}}_{\mathrm{KL}(e\|a)} = J(a,e).$$

$J$ es la divergencia simetrizada de Kullback-Leibler, que Jeffreys (1946) introdujo como medida invariante entre distribuciones. Consecuencias inmediatas:

- $\text{PSI} \ge 0$, con igualdad si y solo si $a = e$ (desigualdad de Gibbs aplicada a cada KL).
- Es simétrica en $(a,e)$: intercambiar DEV y actual no cambia el valor. Cada aporte por bin también es simétrico y no negativo, porque $(a_b - e_b)$ y $\ln(a_b/e_b)$ siempre tienen el mismo signo.
- **No es una métrica**: no cumple la desigualdad triangular. PSI(ene→mar) puede ser mayor que PSI(ene→feb) + PSI(feb→mar). Por eso los PSI de meses consecutivos no se «suman» a la distancia acumulada (§3.7 y §7.4 del notebook).
- **No está acotada**: si $a_b \to 0$ con $e_b > 0$, el aporte tiende a infinito. De ahí el ε.
- La identidad es exacta aun sumando ε sin renormalizar (es algebraica término a término), pero entonces $a+\varepsilon$ y $e+\varepsilon$ ya no son distribuciones; `scipy.stats.entropy` sí renormaliza y difiere en $O(\varepsilon)$.

### 3.2 Expansión local: PSI ≈ χ² de Pearson simetrizado

Escribamos $a_b = m_b + h_b$, $e_b = m_b - h_b$ (así $\delta_b = 2h_b$, $\sum_b h_b = 0$). Usando $\ln\frac{1+u}{1-u} = 2\,\mathrm{artanh}(u) = 2\left(u + \frac{u^3}{3} + \frac{u^5}{5} + \cdots\right)$ con $u = h_b/m_b$:

$$(a_b - e_b)\ln\frac{a_b}{e_b} = 2h_b\cdot 2\left(\frac{h_b}{m_b} + \frac{h_b^3}{3m_b^3} + \cdots\right) = \frac{4h_b^2}{m_b} + \frac{4h_b^4}{3m_b^3} + \cdots$$

Volviendo a $\delta_b = 2h_b$:

$$\boxed{\;\text{PSI} = \sum_b \frac{\delta_b^2}{m_b} + \sum_b\frac{\delta_b^4}{12\,m_b^3} + O(\delta^6)\;}$$

Todos los términos son positivos, así que $\text{PSI} \ge \sum_b \delta_b^2/m_b$, y el error del primer término es de **cuarto** orden: para drift chico, el PSI es el χ² de Pearson con denominador en el punto medio.

**Jensen-Shannon.** Con $m$ la mezcla, $\mathrm{KL}(a\|m) = \sum_b (m_b + h_b)\ln(1 + h_b/m_b)$. Expandiendo $\ln(1+u) = u - u^2/2 + O(u^3)$:

$$\mathrm{KL}(a\|m) = \sum_b (m_b + h_b)\left(\frac{h_b}{m_b} - \frac{h_b^2}{2m_b^2}\right) + O(h^3) = \sum_b h_b + \sum_b\frac{h_b^2}{2m_b} + O(h^3).$$

Como $\sum_b h_b = 0$, $\mathrm{KL}(a\|m) \approx \sum_b h_b^2/(2m_b)$, y lo mismo para $\mathrm{KL}(e\|m)$ (con $-h_b$; los términos cúbicos se cancelan al promediar). Entonces

$$\text{JS} = \tfrac12\mathrm{KL}(a\|m) + \tfrac12\mathrm{KL}(e\|m) \approx \sum_b\frac{h_b^2}{2m_b} = \sum_b\frac{\delta_b^2}{8m_b} \approx \frac{\text{PSI}}{8}.$$

**Hellinger.** $H^2 = \tfrac12\sum_b(\sqrt{a_b} - \sqrt{e_b})^2$. Con $\sqrt{m \pm h} = \sqrt m\,(1 \pm \tfrac{h}{2m} - \tfrac{h^2}{8m^2} + \cdots)$, la diferencia es $h_b/\sqrt{m_b} + O(h^3)$, y

$$H^2 \approx \tfrac12\sum_b \frac{h_b^2}{m_b} = \sum_b\frac{\delta_b^2}{8m_b} \approx \frac{\text{PSI}}{8}.$$

**Lectura.** PSI, JS, Hellinger y χ² son, para drift chico, **la misma medida** con distinta escala (todas son $f$-divergencias y todas se reducen localmente a la métrica de información de Fisher). Elegir entre ellas no cambia qué se detecta mientras el drift sea chico y no haya bins vacíos; difieren en el comportamiento con drift grande (JS ≤ ln 2 y $H \le 1$ están acotadas; el PSI no) y con ceros (JS y Hellinger los toleran sin ε). El notebook (§2) lo verifica: con corrimiento de 0,05 DE las cuatro coinciden a 4 decimales; con 1 DE, PSI 0,926 contra 8·JS 0,850.

### 3.3 Qué significa un PSI en unidades de la variable

Para dos normales, $\mathrm{KL}\big(N(\mu_a,\sigma_a^2)\,\|\,N(\mu_e,\sigma_e^2)\big) = \ln\frac{\sigma_e}{\sigma_a} + \frac{\sigma_a^2 + (\mu_a - \mu_e)^2}{2\sigma_e^2} - \frac12$.

- **Corrimiento de media** $\Delta$ (en DE, $\sigma_a = \sigma_e$): cada KL vale $\Delta^2/2$, así que $J = \Delta^2$.
- **Cambio de escala** $\sigma_a = k\sigma_e$ (misma media): $\mathrm{KL}(a\|e) = -\ln k + \frac{k^2}{2} - \frac12$ y $\mathrm{KL}(e\|a) = \ln k + \frac{1}{2k^2} - \frac12$. Sumando: $J = \frac{k^2 + k^{-2} - 2}{2} = \frac{(k - 1/k)^2}{2}$.

Agrupar en bins es una función determinista de $x$; por la **desigualdad de procesamiento de datos** (Cover y Thomas, cap. 2), la KL de las versiones agrupadas es menor o igual que la de las continuas, en ambos sentidos. Luego $\text{PSI}_B \le J$. El notebook muestra cuánto se pierde: con deciles, un corrimiento de media conserva ~96% de $J$, pero un cambio de escala $k=1{,}3$ conserva solo ~61% ($J = 0{,}141$, PSI por deciles $0{,}087$), porque la información sobre la varianza vive en las colas, que los deciles extremos agrupan.

Traducción de los umbrales del curso: PSI 0,10 ≈ corrimiento de media de **0,33 DE**; PSI 0,25 ≈ **0,5 DE**. Es la manera más honesta de explicarle a un comité qué significa «0,10».

### 3.4 Masa puntual y regla de la cadena

Sea un bin especial para la moda (el cero de las moras) con masas $a_0$, $e_0$, y el resto particionado en bins con proporciones **condicionales** $a_{c,b}$, $e_{c,b}$ (que suman 1 entre los no-cero). Entonces $a_b = (1-a_0)a_{c,b}$ y $e_b = (1-e_0)e_{c,b}$ para los bins no-cero, y

$$\sum_{b\neq 0} a_b\ln\frac{a_b}{e_b} = (1-a_0)\sum_b a_{c,b}\left[\ln\frac{1-a_0}{1-e_0} + \ln\frac{a_{c,b}}{e_{c,b}}\right] = (1-a_0)\ln\frac{1-a_0}{1-e_0} + (1-a_0)\,\mathrm{KL}(a_c\|e_c).$$

Agregando el bin de la moda, $\mathrm{KL}(a\|e) = \mathrm{KL}_{\text{Bern}}(a_0\|e_0) + (1-a_0)\,\mathrm{KL}(a_c\|e_c)$, y simétricamente para $\mathrm{KL}(e\|a)$. Sumando:

$$\boxed{\;J(a,e) = J_{\text{Bern}}(a_0, e_0) + (1-a_0)\,\mathrm{KL}(a_c\|e_c) + (1-e_0)\,\mathrm{KL}(e_c\|a_c)\;}$$

Es la regla de la cadena de la KL. Operativamente permite reportar **dos preguntas separadas**: ¿cambió la fracción con mora? (primer término, 1 grado de libertad) y ¿cambió la distribución de la mora entre quienes la tienen? (resto). El notebook la verifica a precisión de máquina.

### 3.5 Distribución nula: por qué $\text{PSI}/c \to \chi^2_{B-1}$

**Supuestos:** muestras independientes; bins fijos; conteos $A \sim \text{Mult}(n_a, p)$ y $E \sim \text{Mult}(n_e, p)$ con la **misma** $p$ (H0: sin drift); $\hat a = A/n_a$, $\hat e = E/n_e$.

*Paso 1 — momentos.* $\mathbb E[\hat a - \hat e] = 0$ y, por independencia,
$$\mathrm{Cov}(\hat a - \hat e) = \frac{1}{n_a}\Sigma_p + \frac{1}{n_e}\Sigma_p = c\,\Sigma_p,\qquad \Sigma_p = \mathrm{diag}(p) - pp^\top,\qquad c = \frac{1}{n_e} + \frac{1}{n_a}.$$

*Paso 2 — normalidad.* Por el TCL multivariado, $\delta/\sqrt c \xrightarrow{d} N(0, \Sigma_p)$.

*Paso 3 — estandarización.* Sea $z_b = \delta_b/\sqrt{c\,p_b}$, es decir $z = D^{-1/2}\delta/\sqrt c$ con $D = \mathrm{diag}(p)$. Su covarianza límite es
$$D^{-1/2}\Sigma_p D^{-1/2} = I - \sqrt p\,\sqrt p^{\,\top} =: P.$$
$P$ es simétrica e idempotente ($P^2 = I - 2\sqrt p\sqrt p^\top + \sqrt p(\sqrt p^\top\sqrt p)\sqrt p^\top = P$, porque $\sqrt p^\top\sqrt p = \sum p_b = 1$) con traza $B - 1$: es una proyección de rango $B-1$.

*Paso 4 — forma cuadrática.* Si $z \sim N(0,P)$ con $P$ proyección de rango $r$, diagonalizando $P = Q\,\mathrm{diag}(1,\dots,1,0)\,Q^\top$ se tiene $z^\top z = \sum_{i=1}^{r} u_i^2$ con $u_i$ normales estándar independientes: $z^\top z \sim \chi^2_r$. Luego $\sum_b \delta_b^2/(c\,p_b) \xrightarrow{d} \chi^2_{B-1}$.

*Paso 5 — del χ² al PSI.* Por §3.2, $\text{PSI}/c = \sum_b \delta_b^2/(c\,m_b) + \sum_b \delta_b^4/(12\,c\,m_b^3) + \cdots$. Bajo H0, $m_b \to p_b$ en probabilidad (Slutsky) y $\delta_b = O_p(\sqrt c)$, así que el término cuártico es $O_p(c) \to 0$. Por lo tanto

$$\frac{\text{PSI}}{c} \xrightarrow{d} \chi^2_{B-1},\qquad \mathbb E[\text{PSI}] \approx (B-1)\,c,\qquad \mathrm{Var}[\text{PSI}] \approx 2(B-1)\,c^2.$$

Este es el resultado de Yurdakul (2018) y Yurdakul y Naranjo (2020). El crítico de tamaño $\alpha$ es $\text{PSI}_{\text{crit}} = c\cdot\chi^2_{1-\alpha,\,B-1}$, o con la aproximación normal $c\,[(B-1) + z_{1-\alpha}\sqrt{2(B-1)}]$ (para $B = 10$, $\alpha = 5\%$: 16,92 exacto vs 15,98 normal).

**Conexión exacta con el χ² de homogeneidad.** La prueba de Pearson sobre la tabla $2\times B$ usa $\hat p_b = (n_a\hat a_b + n_e\hat e_b)/N$, $N = n_a + n_e$. El residuo de la fila actual es $n_a(\hat a_b - \hat p_b) = \frac{n_a n_e}{N}(\hat a_b - \hat e_b)$ y el de DEV es el mismo con signo opuesto. Entonces

$$X^2 = \sum_b \left(\frac{n_a n_e}{N}\right)^2\delta_b^2\left[\frac{1}{n_a\hat p_b} + \frac{1}{n_e\hat p_b}\right] = \frac{n_a n_e}{N}\sum_b\frac{\delta_b^2}{\hat p_b} = \frac{1}{c}\sum_b\frac{\delta_b^2}{\hat p_b}.$$

Comparado con $\text{PSI}/c \approx \frac1c\sum_b \delta_b^2/m_b$: la única diferencia de segundo orden es el denominador, punto medio simple ($m_b$) versus promedio ponderado por tamaño ($\hat p_b$). Si $n_a = n_e$ coinciden y la diferencia es solo el término cuártico. En el ancla Austral (DEV→TTD), $\text{PSI}/c = 31{,}08$ y $X^2 = 30{,}99$.

**Bins por cuantiles de DEV (lo que hace el curso).** Con cortes aleatorios, $\hat e_b = 1/B$ exactamente y el ruido de DEV «se muda» a los cortes: la masa poblacional $q_b$ de cada bin es un grupo de espaciamientos uniformes, con $\mathrm{Var}(q_b) = \frac{p_b(1-p_b)}{n_e+2}$ (distribución Dirichlet de los espaciamientos: $q_b$ es Beta con parámetros que suman $n_e+1$). Entonces $\hat a_b - 1/B = (\hat a_b - q_b) + (q_b - 1/B)$, dos piezas independientes con covarianzas $\Sigma_p/n_a$ y $\approx\Sigma_p/n_e$: el mismo $c$. La simulación del notebook confirma la aproximación desde $n \approx 300$ (percentil 95 simulado 0,106 vs 0,113 teórico con $n_e = n_a = 300$).

**Si DEV se trata como conocida** ($n_e \to \infty$, o se usa `scipy.stats.chisquare` contra las proporciones de DEV), $c = 1/n_a$. Con DEV de 3.322 y lotes mensuales de 800, ignorar el ruido de DEV subestima $c$ en ~20% y produce p-valores optimistas.

### 3.6 Bajo drift: sesgo y potencia

Con drift poblacional fijo, $\mathbb E[\text{PSI}_{\text{obs}}] \approx \text{PSI}_{\text{pob}} + (B-1)c$: el PSI observado está **sesgado hacia arriba** por el ruido. Un estimador corregido es $\text{PSI} - (B-1)c$. Para alternativas locales ($\delta$ del orden de $\sqrt c$), $\text{PSI}/c \to \chi^2_{B-1}(\lambda)$ no central con $\lambda \approx \text{PSI}_{\text{pob}}/c$. Esto da la potencia y permite construir un **intervalo de confianza** para $\text{PSI}_{\text{pob}}$ invirtiendo la χ² no central, que es la herramienta correcta para certificar estabilidad (§4).

Dos consecuencias para el diseño del número de bins:
- La señal $\lambda$ crece con $B$ solo si el drift tiene estructura fina (cola, forma). Para un corrimiento de media, $\lambda$ se satura con pocos bins mientras los grados de libertad siguen creciendo: la potencia **cae** con $B$ (notebook §5.1: con Δ = 0,08 DE y $n = 3.000$, potencia 0,69 con 3 bins, 0,51 con 10, 0,28 con 50).
- Para cambios en la cola ocurre lo contrario: más bins resuelven mejor (§3 del notebook: contaminación de cola, potencia 0,37 con 10 bins y 0,68 con 30).

### 3.7 Taxonomía de cambio

Con $p(x,y) = p(y\,|\,x)\,p(x) = p(x\,|\,y)\,p(y)$ (Moreno-Torres et al., 2012):

| Tipo | Qué cambia | Qué se mantiene | ¿Lo ve el PSI de $x$ o del score? |
|---|---|---|---|
| Covariate shift | $p(x)$ | $p(y\,|\,x)$ | Sí: es exactamente lo que mide |
| Prior shift | $p(y)$ | $p(x\,|\,y)$ | Indirectamente: $p(s) = \pi\,p(s|1) + (1-\pi)\,p(s|0)$ se mueve con $\pi$ |
| Concept drift | $p(y\,|\,x)$ | $p(x)$ | **No** |

Es más preciso decir que el PSI ve **cambios en la marginal de $x$**, cualquiera sea su origen, y no puede distinguir entre ellos. El notebook (§6) lo muestra con verdad conocida: prior shift de 10% a 15% de malos da PSI del score 0,011 (detectable con $n = 20.000$); concept drift de +0,6 en log-odds sube la tasa de malos a 13,7% con PSI 0,0009 (ruido).

Un caso intermedio importante en crédito: **covariate shift en una variable omitida**. Si $z$ afecta el riesgo pero no está en el modelo, $p(y\,|\,x_{\text{modelo}}) = \int p(y\,|\,x,z)\,p(z\,|\,x)\,dz$ cambia cuando cambia $p(z)$. Para el modelo, es concept drift; el PSI de sus variables no lo ve, pero el PSI de $z$ sí. Justifica monitorear variables **fuera** del modelo (canal, producto, sucursal, tipo de ingreso).

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo / limitación | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| PSI con umbrales fijos 0,10 / 0,25 | Lenguaje común, fácil de comunicar | Ignora $n$ y $B$; falsa alarma alta con $n$ chico, ceguera con $n$ grande | Comunicación; nunca como único criterio | Práctica de scorecards (Siddiqi); ninguna regulación que conozca prescribe estos valores |
| PSI con crítico por $n$ (χ² o simulación) | Controla el error tipo I | Con $n$ grande todo es significativo | Primera compuerta: *¿hubo cambio?* | Yurdakul y Naranjo (2020) |
| PSI + materialidad (dos compuertas) | Separa «cambió» de «importa» | Hay que definir la materialidad (negocio) | Tablero de producción | Recomendación de este módulo; cf. du Pisanie et al. (2023, tamaño de efecto) |
| IC del PSI / test de equivalencia | Certifica estabilidad cuando $n$ es chico («no sé» ≠ «estable») | χ² no central o bootstrap | Selección de variables con $n$ chico; certificado de estabilidad | Práctica avanzada |
| χ² de homogeneidad / G-test | Test clásico sobre la tabla $2\times B$ | Mismo problema de $n$ grande | Equivalente al PSI/c; útil para auditores que no conocen el PSI | Estadística clásica (Agresti) |
| KS de dos muestras | Ubicación, sin bins, invariante a transformaciones monótonas | Poco sensible en colas y a cambios de varianza | Variables continuas y score; más potente que el PSI para corrimientos de media | Muy usado en validación del score (KS de discriminación es otra cosa: buenos vs malos) |
| Jensen-Shannon | Acotada (≤ ln 2), tolera ceros sin ε | Localmente = PSI/8; no aporta información nueva con drift chico | Muchos bins o categorías raras | Librerías de monitoreo de ML |
| Hellinger | Acotada, métrica verdadera ($H$ cumple triangular) | Localmente $H^2$ = PSI/8 | Cuando se necesita desigualdad triangular | Literatura de detección de drift |
| Wasserstein-1 | Distancia en unidades de $x$; sensible a cuán lejos se fue la masa (colas) | No es invariante a escala; sin umbral universal; sensible a outliers | Colas y montos (deuda, renta) con escala interpretable | ML monitoring |
| Population Accuracy Index (PAI) | Mide si el cambio **afecta la precisión** del modelo (extrapolación) | Requiere el modelo y su matriz de covarianza | Decidir si un cambio poblacional importa para las predicciones | Taplin y Hunt (2019) |
| Δpuntos por variable (análisis de característica) | Dirección e impacto en el score; se suma exacto al Δ del score medio | No es un test; necesita la tabla de puntos | Siempre, junto al CSI | Manuales de scorecards; suites de monitoreo |
| Detección de drift multivariada (clasificador DEV vs actual, MMD) | Cambios en la dependencia entre variables | Menos interpretable; más costo | Modelos ML; auditoría anual | Rabanser et al. (2019) |

**Umbrales ajustados por tamaño.** La regla defendible para producción, en dos compuertas:

1. **¿Hubo cambio?** $p = P\big(\chi^2_{B-1} > \text{PSI}/c\big) < \alpha$, con $\alpha$ corregido por multiplicidad si se miran muchas variables (Benjamini-Hochberg sobre las $K$ variables del mes). Si las observaciones no son independientes (mismos clientes en meses sucesivos, muestras solapadas), el p-valor es optimista: calibrar la nula por simulación o bloque-bootstrap.
2. **¿Importa?** Materialidad declarada de antemano: PSI corregido $\text{PSI} - (B-1)c$ sobre un piso (p. ej. 0,05), o mejor, impacto en decisiones: Δpuntos del score, cambio en tasa de aprobación al cutoff vigente, cambio en PD media implícita.

La combinación produce cuatro estados: cambio real y material (escalar), cambio real no material (registrar tendencia), PSI alto sin significancia ($n$ chico: acumular datos, no concluir) y estable. La planilla del módulo implementa exactamente esta lógica.

**La regla «sin PSI medible no sigue».** *A favor:* conservadora, barata en el caso Austral (la información de mora sobrevive en `dias_mora_max_12m` y `meses_desde_mora_12m`), fácil de auditar. *En contra:* la causa del `NaN` es el binning por deciles, no la variable; descarta información con IV 0,49–0,58 por un artefacto técnico; y crea un incentivo perverso (una variable con 89% de ceros «pasa» y una con 91% no). *Alternativa defendible:* medir la estabilidad con el **mismo binning con que la variable entraría al modelo** (bin de moda + cuantiles del resto, bin MISSING, códigos especiales separados; ver Serie 1 · M6 y M7); reportar la descomposición de §3.4; y exigir un **certificado de estabilidad** en forma de cota superior: el límite superior del IC al 95% del PSI poblacional bajo el umbral de materialidad. Una variable sin masa suficiente fuera de la moda para ese certificado se monitorea como indicador binario. Así «no medible» deja de existir como categoría y queda «certificada», «no certificable con este $n$» o «inestable».

**Referencia: DEV congelado vs ventana móvil.** DEV congelado mide la distancia acumulada al mundo en que se estimaron los parámetros: es la pregunta de gobierno y es auditable. La ventana móvil (mes vs mes anterior) detecta quiebres abruptos (cambio de política, nuevo canal, bug de captura) pero es ciega a derivas lentas: cada mes se parece al anterior. En el notebook (§7.4), la tendencia lenta de `uso_linea_prom_12m` supera el crítico contra DEV en los 6 meses de TTD, mientras la comparación mes a mes queda casi siempre dentro del ruido. Ninguna supera 0,10. La recomendación es usar ambas, con roles distintos, declaradas en el plan de monitoreo.

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 Bins recalculados en la muestra actual.**
*Síntoma:* PSI ≈ 0 en todas las variables, todos los meses. *Causa:* alguien calculó deciles sobre la muestra actual (por construcción, 10% en cada bin) o re-ajustó el binner en producción (el bug de la clase 6: 587 de 8.585 decisiones cambiadas). *Detección:* test de contrato: los cortes usados en el monitoreo tienen el hash de los congelados en el artefacto. *Qué hacer:* los cortes, las etiquetas y $e_b$ son parte del artefacto versionado, no del código de monitoreo.

**5.2 Comparar PSI con distinto $n$ o distinto $B$.**
*Síntoma:* «la variable X empeoró de 0,03 a 0,06» cuando el lote pasó de 5.000 a 800 casos, o cuando otro analista usó 20 bins. *Causa:* $\mathbb E[\text{PSI}\,|\,H_0] = (B-1)c$ sube con $B$ y baja con $n$. *Detección:* reportar siempre $(B-1)c$ al lado del PSI, o directamente el p-valor. *Qué hacer:* comparar p-valores o PSI corregidos, no PSI crudos.

**5.3 Falsas alarmas por $n$ chico.**
*Síntoma:* semáforos amarillos que aparecen y desaparecen sin patrón en segmentos pequeños. *Causa:* con $n_a = 100$ y $B = 10$, $c \approx 0{,}01$ y $P(\text{PSI} > 0{,}10\,|\,H_0) \approx 36\%$ (con $n_e = 10.000$). *Detección:* p-valor alto con PSI sobre umbral. *Qué hacer:* agregar meses, reducir bins o reportar «no concluyente».

**5.4 Ceguera por $n$ grande.**
*Síntoma:* todo verde durante años y luego un deterioro que «nadie vio». *Causa:* con $n$ de decenas de miles, 0,10 está ~100 veces por encima del crítico; derivas reales y persistentes nunca llegan al umbral. *Detección:* series de p-valores sistemáticamente bajos con PSI verde (el Austral DEV→TTD). *Qué hacer:* compuerta estadística + tendencia (p. ej. PSI contra DEV creciente 6 meses seguidos) + materialidad.

**5.5 Bins vacíos y el ε.**
*Síntoma:* un bin con 0 casos domina el PSI. *Causa:* el aporte es $\approx e_b\ln(e_b/\varepsilon)$; con $e_b = 5\%$ vale 0,31 con ε = 10⁻⁴ y 0,54 con ε = 10⁻⁶. *Detección:* aporte de un solo bin > 50% del total; bin con $a_b = 0$. *Qué hacer:* al congelar, exigir masa mínima por bin (p. ej. ≥ 5% en DEV) y fusionar; usar suavizado de Laplace en conteos (+0,5), que depende de $n$ y no de una constante arbitraria; reportar JS como control.

**5.6 Masas puntuales.**
*Síntoma:* `NaN`, o PSI con menos bins de los que el lector cree (en el generador, `meses_desde_mora_12m` queda con 4 bins efectivos por deciles). *Causa:* `np.unique` sobre cuantiles repetidos. *Detección:* registrar $B_{\text{efectivo}}$ en cada cálculo. *Qué hacer:* bin de moda explícito; códigos especiales (−9, −99) como bins propios; descomposición de §3.4.

**5.7 Cancelación en el score.**
*Síntoma:* CSI amarillos con PSI del score verde (Austral: 0,138 vs 0,013). *Causa:* el score es una suma; corrimientos opuestos se compensan en la media. *Detección:* tabla de Δpuntos por variable: dos Δ grandes de signo opuesto con Δ total chico. *Qué hacer:* el tablero reporta CSI y Δpuntos; investigar la variable aunque el score esté verde, porque la PD por banda puede dejar de ser válida si la relación real no es exactamente aditiva en esa región.

**5.8 Concept drift invisible.**
*Síntoma:* PSI verde, tasa de malos de las cosechas nuevas subiendo. *Causa:* cambió $p(y|x)$ (macro, fraude, cambio de política de cobranza) o una variable omitida (canal). *Detección:* solo con desempeño: indicadores tempranos de mora (first payment default, 30+ a 3 meses por cosecha) contra lo esperado por el modelo. *Qué hacer:* no usar el PSI como evidencia de salud del modelo; en el tablero, el PSI es población, no calibración (ver M19 y M20).

**5.9 El TTD contaminado por la política.**
*Síntoma:* después de implementar el modelo, PSI creciente en las variables más importantes. *Causa:* si el «TTD» son créditos **cursados**, la política nueva cambia quién se cursa, y el PSI mide el efecto del propio modelo. En el curso, TTD son cursadas recientes; lo correcto para medir población es la bandeja de **solicitudes** (through-the-door real), antes de la decisión. *Detección:* comparar PSI de solicitudes vs PSI de cursadas. *Qué hacer:* definir TTD = solicitudes en el plan de monitoreo; si solo existen cursadas, declararlo.

**5.10 Multiplicidad y ganador de la maldición.**
*Síntoma:* en cada corrida aparecen «las 5 variables inestables del mes», distintas cada vez. *Causa:* 108 tests al 5% generan ~5,4 falsos positivos esperados. Y la otra cara: si el embudo selecciona variables porque su PSI contra *este* TTD fue bajo, el monitoreo posterior contra un TTD parecido está sesgado hacia el verde (selección sobre ruido). *Detección:* distribución de p-valores (uniforme bajo H0). *Qué hacer:* Benjamini-Hochberg por corte; para la selección, medir estabilidad en una ventana y certificarla en otra.

**5.11 Niveles categóricos nuevos.**
*Síntoma:* PSI enorme por un canal o producto que no existía en DEV. *Causa:* $e_b = 0$, el aporte es $a_b\ln(a_b/\varepsilon)$. *Detección:* nivel no visto en el diccionario congelado. *Qué hacer:* bin «OTRO» explícito en el artefacto, con WoE neutro documentado (el `a_woe` del curso asigna 0), y alerta de calidad de datos separada del PSI.

**5.12 Estacionalidad y dependencia serial.**
*Síntoma:* amarillos recurrentes cada diciembre. *Causa:* DEV mezcla 12+ meses; un mes aislado tiene su propio mix estacional. Además, clientes que aparecen en meses sucesivos hacen que las observaciones no sean independientes. *Qué hacer:* comparar contra el mismo mes de DEV cuando haya estacionalidad; calibrar la nula por bloque-bootstrap cuando haya dependencia.

**5.13 Cambios de captura que parecen cambios de población.**
*Síntoma:* PSI rojo súbito en una variable de bureau. *Causa:* cambio de formato, de código de missing (−99), de proveedor. *Detección:* el cambio es abrupto y concentrado en un bin (MISSING o código especial). *Qué hacer:* es un incidente de calidad de datos (Serie 1 · M6), no un fenómeno poblacional: se corrige en el pipeline, no en el modelo.

---

## 6. Puente con ingeniería

El monitoreo de estabilidad es un pipeline declarativo más, con un artefacto congelado de referencia y una especificación de lo que se calcula cada corte. Lo que debe quedar **congelado con el modelo** (junto al hash del artefacto del scorecard; ver M21):

```yaml
monitor_estabilidad:
  referencia:
    muestra: DEV                      # congelada; hash del snapshot
    snapshot_sha256: "…"
    n_referencia: 10065               # necesario para c = 1/n_e + 1/n_a
  variables:
    uso_linea_prom_12m:
      tipo: numerica
      cortes: [-.inf, 0.215, 0.26, 0.307, 0.364, .inf]   # los del binning WoE
      intervalos: derecha_cerrada     # (c_{k-1}, c_k], igual que pd.cut
      bins_especiales: {MISSING: null}
      proporciones_ref: [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
      conteos_ref: [2013, 2013, 2013, 2013, 2013, 0]
    canal:
      tipo: categorica
      niveles: [sucursal, web, app, fuerza_venta]
      nivel_no_visto: OTRO
  score:
    binning: master_scale_8_bandas    # los cortes de M16, congelados
  politica:
    suavizado: {tipo: aditivo, eps: 1.0e-4}
    compuerta_estadistica: {alfa: 0.05, multiplicidad: benjamini_hochberg}
    materialidad: {psi_corregido_min: 0.05, delta_puntos_min: 3}
    umbrales_comunicacion: {amarillo: 0.10, rojo: 0.25}   # convención del curso
    referencia_secundaria: {tipo: ventana_movil, meses: 1}
```

**Salidas por corte (append-only, versionadas):** conteos por bin (no solo proporciones: sin conteos no se puede recalcular el p-valor), $n_a$, $B_{\text{efectivo}}$, PSI, $(B-1)c$, p-valor, p ajustado, Δpuntos por variable, estado de cada compuerta, versión del artefacto y del código.

**Invariantes verificables (tests tipo CI):**

1. `psi(ref, ref) == 0` y `psi(e, a) == psi(a, e)` (simetría).
2. `psi == kl(a, e) + kl(e, a)` a tolerancia de máquina.
3. Invariancia a transformaciones monótonas: PSI con cuantiles de DEV no cambia si se aplica `log1p` a la variable en ambas muestras.
4. Las proporciones de referencia se reproducen exactamente desde el snapshot de DEV (hash).
5. Todo bin congelado tiene masa de referencia ≥ mínimo declarado; `B_efectivo` coincide con el declarado.
6. Niveles no vistos se mapean a `OTRO` y disparan un evento de calidad de datos, no un error.
7. **Calibración de la nula:** simular 1.000 lotes desde `proporciones_ref` con el $n_a$ típico y verificar que la tasa de alarmas de la compuerta estadística esté en $\alpha \pm 2\sqrt{\alpha(1-\alpha)/1000}$.
8. Identidad aditiva: $\sum_v \Delta\text{puntos}_v = \bar s_{\text{actual}} - \bar s_{\text{ref}}$ a tolerancia de máquina (§7.2 del notebook).
9. Cortes aplicados con la misma convención de intervalos que en entrenamiento (derecha cerrada); un test con valores exactamente iguales a los cortes lo verifica.

**Qué se versiona y qué no:** se versiona la especificación (YAML), el snapshot de referencia y el código del cálculo. No se «re-calibra» la referencia sin un evento de gobierno: cambiar DEV por una ventana reciente es un cambio de modelo de monitoreo y va al comité. La ventana móvil es un cálculo secundario sin poder de decisión propio.

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy (notebook) | Librería | Diferencias de convención |
|---|---|---|---|
| Asignación a bins | `np.searchsorted(cortes[1:-1], x, side="left")` → $(c_{k-1}, c_k]$ | `pd.cut(x, cortes)` (derecha cerrada); `np.histogram` (izquierda cerrada salvo el último); `np.digitize` (configurable) | Con variables discretas y valores iguales a los cortes, los conteos cambian. El curso usa `pd.cut`: replicar eso |
| Cuantiles | `np.nanquantile(..., method="linear")` | `pd.qcut`, `np.percentile` | Métodos de interpolación distintos mueven cortes en variables discretas |
| PSI | `sum((a-e)*log(a/e))` con ε sumado | `scipy.stats.entropy(a,e) + entropy(e,a)` | `entropy` **renormaliza** sus entradas; con ε sin renormalizar difiere en $O(\varepsilon)$. Coinciden exacto con ε = 0 |
| KL | `where(p>0, p*log(p/q), 0)` | `scipy.special.rel_entr(p,q).sum()` | Ambas usan $0\ln 0 = 0$; `rel_entr` devuelve `inf` si $q_b = 0 < p_b$ |
| Jensen-Shannon | $\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m)$ | `scipy.spatial.distance.jensenshannon(p,q)` | Devuelve la **distancia** (raíz de la divergencia); base $e$ por defecto (`base=2` la acota en 1) |
| Hellinger | `sqrt(0.5*sum((sqrt(p)-sqrt(q))**2))` | `euclidean(sqrt(p), sqrt(q))/sqrt(2)` | Algunas fuentes definen $H^2 = 1 - \sum\sqrt{pq}$ (idéntico) y otras omiten el ½ |
| χ² de homogeneidad | $\sum(O-E)^2/E$ sobre $2\times B$ | `scipy.stats.chi2_contingency(O, correction=False)` | Con 1 grado de libertad (2 bins, p. ej. el indicador de ceros) aplica **Yates** por defecto: pasar `correction=False` para coincidir. `lambda_="log-likelihood"` da el G-test |
| χ² contra DEV fija | — | `scipy.stats.chisquare(f_obs, f_exp)` | Trata DEV como conocida: $c = 1/n_a$, p-valores optimistas si $n_e$ no es ≫ $n_a$ |
| KS | `max|F_e - F_a|` sobre la unión de puntos | `scipy.stats.ks_2samp` | Estadístico idéntico; el p-valor usa distribución exacta o asintótica según $n$ (`method`) |
| Wasserstein-1 | $\sum |F_e - F_a|\,\Delta x$ entre puntos consecutivos | `scipy.stats.wasserstein_distance` | Idéntico; en unidades de $x$ |
| Cola de la χ² | — | `scipy.stats.chi2.sf`, `chi2.ppf` | En planilla: `CHISQ.DIST.RT`, `CHISQ.INV.RT` |
| Multiplicidad | — | `statsmodels.stats.multitest.multipletests(p, method="fdr_bh")` | — |
| Monitoreo empaquetado | — | `optbinning.scorecard.ScorecardMonitoring` | Usa sus propios bins (p. ej. CART) sobre la esperada; en la demo de clase 5 dio CSI 0,004 para `deuda_interna_max_3m` DEV→OOT: otra referencia, otros bins. Verificar en la documentación su tratamiento de ceros antes de comparar números |

Todas las coincidencias numpy/librería están en los checks del notebook (`assert np.isclose(...)`). **En producción**: el PSI y los conteos, en numpy puro (transparente, sin dependencias, fácil de auditar); las colas de la χ² y la corrección por multiplicidad, desde `scipy`/`statsmodels`. La calculadora `.xlsx` replica la lógica para quien no programa.

---

## 8. Aplicación: casos y números

### 8.1 Los números del Banco Austral, contra su nula

Tamaños: DEV 3.322, HO 1.397, OOT 2.004, TTD 8.585. $B$ = bins efectivos (para los CSI se supone $B = 5$; el resultado no depende de este supuesto).

| Caso (clase) | Valor | $B$ | $\mathbb E[\text{PSI}\,|\,H_0]$ | Crítico 95% | p aprox. | Lectura |
|---|---|---|---|---|---|---|
| Score deciles DEV→HO (3) | 0,003 | 10 | 0,0092 | 0,0172 | 0,97 | Ruido |
| Score deciles DEV→OOT (3) | 0,005 | 10 | 0,0072 | 0,0135 | 0,71 | Ruido |
| Score deciles DEV→TTD (3) | 0,015 | 10 | 0,0038 | 0,0071 | 4·10⁻⁵ | Cambio real, no material |
| Score 8 bandas DEV→OOT (5) | 0,002 | 8 | 0,0056 | 0,0113 | 0,93 | Ruido |
| Score 8 bandas DEV→TTD (5) | 0,013 | 8 | 0,0029 | 0,0059 | 6·10⁻⁵ | Cambio real, no material |
| CSI `deuda_interna_max_3m` DEV→TTD (5) | 0,138 | 5 | 0,0017 | 0,0040 | ≈ 0 | ~80 veces la nula: cambio real y cerca de material |
| CSI `carga_financiera` DEV→TTD (5) | 0,062 | 5 | 0,0017 | 0,0040 | ≈ 0 | Cambio real, verde por convención |
| PSI `uso_linea_prom_12m` DEV→TTD (3) | 0,020 | 10 | 0,0038 | 0,0071 | < 10⁻⁶ | Cambio real, verde por convención |

Tres lecturas. (1) La validación temporal (HO, OOT) es limpia en sentido estricto: los PSI están **por debajo** de su valor esperado sin drift. (2) La bandeja TTD es estadísticamente distinta de DEV en el score, en la mayoría de sus variables principales y en `carga_financiera`, un hecho que los semáforos no registran. El diagnóstico de la clase 5 («deriva incipiente de nivel en los tramos buenos») es coherente con esto y se habría podido afirmar con más fuerza. (3) El único CSI amarillo es el único cuyo PSI supera 0,10, pero `carga_financiera` (0,062, verde) también es ~37 veces su nula: con la compuerta estadística habría entrado al tablero como «cambio real, no material, vigilar».

### 8.2 Banco Sintético (generador de la serie, notebook §7)

DEV 10.065, OOT 4.792, TTD 4.848; ~11% de malos en DEV. Verdad conocida: drift de `canal`, tendencia lenta en `uso_linea_prom_12m` y deterioro macro de +0,35 log-odds en 2025.

- **`canal`**: PSI DEV→TTD **0,290** (teórico poblacional 0,287) 🔴 con IV 0,013; DEV→OOT 0,323. Cae en el paso 1, como la variable plantada de Andes. Para categóricas PSI = CSI.
- **`uso_linea_prom_12m`**: PSI DEV→TTD **0,027** 🟢, p ≈ 2·10⁻¹⁵, significativa después de Benjamini-Hochberg. Es la tendencia plantada: el semáforo fijo nunca la marca.
- Las otras 9 candidatas: sin cambio después de BH (la mínima, `edad`, p = 0,054 sin ajustar, 0,20 ajustada).
- **Score** (6 variables, sin `canal`; Gini DEV/HO/OOT 0,53/0,58/0,54): PSI DEV→TTD por quintiles 0,0055 (p 0,001), deciles 0,0073 (p 0,004), 20 bins 0,011 (p 0,009), 8 bandas 0,0046 (p 0,035). Cuatro PSI distintos para la misma población; los p-valores cuentan una historia consistente.
- **Dirección**: Δ score medio DEV→TTD = −1,87 puntos, del cual `uso_linea_prom_12m` explica −1,90 (el resto se compensa). El CSI de `uso_linea` es 0,023 y su rango de puntos 37: es la variable que mueve el score.
- **Lo que el PSI no ve**: con la misma semilla y `deterioro = 0`, $X$ es idéntica y el PSI del score DEV→OOT es el mismo (0,0028); la PD verdadera media en OOT es 12,5% sin deterioro y 15,8% con él, mientras el modelo predice 11,5% en ambos casos.
- **Ventana**: contra DEV congelado, 6 de 6 cohortes TTD superan el crítico al 95% (PSI 0,030–0,050 con crítico ~0,023); ninguna llega a 0,10.

### 8.3 Crédito de motos (ilustrativo)

En financiamiento de motos, las fuentes típicas de drift poblacional son fáciles de nombrar: mezcla de **concesionarios** y marcas (entrada de marcas nuevas de bajo precio cambia monto, pie y plazo), **estacionalidad** (la demanda sube en primavera-verano), y el segmento de **repartidores de aplicaciones**, cuya participación puede cambiar rápido y cuyo ingreso es variable. Tres cálculos con números hipotéticos, para dimensionar:

1. **Tipo de ingreso** (categórica): DEV [dependiente 60%, independiente 25%, repartidor app 15%] → actual [40%, 25%, 35%]. PSI $= (0{,}4-0{,}6)\ln\frac{0{,}4}{0{,}6} + 0 + (0{,}35-0{,}15)\ln\frac{0{,}35}{0{,}15} = 0{,}081 + 0{,}169 = 0{,}251$: rojo. Si el tipo de ingreso **no** está en el modelo, esta es exactamente la situación de §3.7: covariate shift en una variable omitida, que el PSI del score puede no ver.
2. **Monitoreo mensual de toda la cartera**: DEV $n_e = 10.000$, 1.500 solicitudes/mes, 10 bins: $c = 0{,}00077$, $\mathbb E[\text{PSI}\,|\,H_0] = 0{,}0069$, crítico 95% = 0,013. Un PSI de 0,02 mensual sostenido es un cambio real; el 0,10 no lo vería.
3. **Monitoreo por concesionario**: 100 solicitudes/mes en un concesionario: $c = 0{,}0101$, $P(\text{PSI} > 0{,}10\,|\,H_0) \approx 36\%$. Un tablero por concesionario con umbral 0,10 genera una falsa alarma cada tres meses por concesionario. Solución: agregar trimestres, reducir a 4–5 bins o usar el crítico por $n$ (0,17 a 95%).

---

## 9. Preguntas de comité

**1. «El PSI del score está en 0,013, verde. ¿Por qué me muestran una alerta?»**
Porque 0,013 no es un número bajo para este tamaño muestral: sin cambio, lo esperable sería 0,003, y la probabilidad de ver 0,013 por azar es de 6 en 100.000. La población sí cambió; lo que el verde dice es que el cambio es chico. Proponemos registrarlo como «cambio real, no material», con seguimiento de tendencia, y no actuar sobre el modelo.

**2. «¿De dónde salen 0,10 y 0,25?»**
Son convenciones de práctica de scorecards (Siddiqi), sin base inferencial ni tasa de error asociada. Equivalen aproximadamente a corrimientos de media de 0,33 y 0,5 desviaciones estándar en una variable normal. Los mantenemos para comunicar, pero la decisión se toma con una compuerta estadística ajustada por tamaño y una compuerta de materialidad declarada.

**3. «El CSI de una variable está en amarillo pero el score está verde. ¿Cuál manda?»**
Ninguno solo. El score puede estar verde porque otras variables compensan (cancelación). Miramos el Δpuntos por variable: si la variable amarilla mueve el score en una dirección y otra lo compensa, el modelo está ordenando a una población distinta de la de desarrollo, aunque la mezcla total se parezca. Se investiga la causa de negocio de la variable amarilla antes de que la compensación deje de ocurrir.

**4. «¿Un PSI verde garantiza que el modelo funciona?»**
No. El PSI mide solo la distribución de las variables. Un deterioro macro que hace caer más a los mismos perfiles (concept drift) no mueve el PSI. En el caso sintético, con la población idéntica, la PD real pasó de 12,5% a 15,8% sin que el PSI cambiara. La salud del modelo se verifica con desempeño: backtesting por banda y discriminación por cosecha (M19, M12).

**5. «¿Por qué se descartaron variables de mora con IV 0,5 por "no medibles"?»**
Porque el método de deciles no puede formar bins cuando el 90% de los casos es cero. Es una limitación del cálculo, no de la variable. La alternativa que proponemos es medir con el binning del propio modelo (un bin para el cero y cuantiles para el resto) y certificar estabilidad con una cota superior. En el caso Austral el descarte costó poco porque la información de mora sobrevive en variables de ventana larga, pero la regla general no es defendible.

**6. «¿Contra qué se compara: DEV o el mes pasado?»**
Contra DEV congelado como referencia oficial: mide la distancia acumulada al mundo en que se estimó el modelo y es auditable. El mes pasado se usa como detector secundario de quiebres abruptos. La comparación mes a mes sola no ve derivas lentas: en el caso sintético, una tendencia que supera el crítico contra DEV en 6 de 6 meses pasa desapercibida mes a mes.

**7. «Con 108 variables, ¿cuántas alarmas esperan por azar?»**
Al 5%, unas 5 por corte aunque nada cambie. Por eso el tablero usa corrección por multiplicidad (Benjamini-Hochberg) y se fija en patrones persistentes, no en alarmas aisladas.

**8. «Si el TTD está formado por créditos cursados, ¿el PSI no está midiendo nuestra propia política?»**
Sí, en parte. Después de implementar el modelo, la población cursada cambia por la política. Para medir la población que llega hay que usar las solicitudes antes de la decisión. Si solo hay cursadas, el PSI se reporta con esa salvedad explícita.

---

## 10. Ejercicios

**E1 (cálculo a mano).** Con la tabla de bandas del Austral (clase 5), calcula el aporte de la banda E y de la banda D al PSI DEV→TTD, y la fracción del PSI que explican juntas.

<details><summary>Solución</summary>

E: $(0{,}1447 - 0{,}1225)\ln(0{,}1447/0{,}1225) = 0{,}0222 \times 0{,}1666 = 0{,}0037$. D: $(0{,}1090 - 0{,}0942)\ln(0{,}1090/0{,}0942) = 0{,}0148 \times 0{,}1459 = 0{,}0022$. Juntas: 0,0059 de 0,0130, un 45%. Con A1 (0,0045) llegan al 80%: el PSI se concentra en los extremos, que es donde el corrimiento hacia el riesgo se ve. Los aportes no tienen signo: la columna «dirección» (se vacía / se llena) hay que agregarla aparte.
</details>

**E2 (derivación).** Demuestra que cada aporte $(a_b - e_b)\ln(a_b/e_b)$ es no negativo y que es cero solo si $a_b = e_b$. ¿Qué propiedad de la KL individual *no* se hereda término a término?

<details><summary>Solución</summary>

$\ln$ es creciente: si $a_b > e_b$, $\ln(a_b/e_b) > 0$; si $a_b < e_b$, $\ln(a_b/e_b) < 0$. El producto tiene el signo de $(a_b - e_b)^2 \ge 0$. En cambio, los términos de $\mathrm{KL}(a\|e) = \sum a_b\ln(a_b/e_b)$ **pueden ser negativos** (columna `kl_a_e_ttd` en el notebook: −0,028 para A1); solo la suma es no negativa (desigualdad de Gibbs). Por eso el PSI admite una lectura por bin y la KL no.
</details>

**E3 (nula).** Un monitor mensual compara DEV ($n_e = 5.000$) con lotes de $n_a = 600$ usando 10 bins. (a) ¿Cuál es el PSI esperado sin drift? (b) ¿El crítico al 95%? (c) ¿Qué fracción de los meses daría amarillo con la regla 0,10 sin ningún cambio?

<details><summary>Solución</summary>

$c = 1/5000 + 1/600 = 0{,}0002 + 0{,}001667 = 0{,}001867$. (a) $9c = 0{,}0168$. (b) $c\cdot\chi^2_{0{,}95;9} = 0{,}001867 \times 16{,}92 = 0{,}0316$. (c) $P(\chi^2_9 > 0{,}10/0{,}001867 = 53{,}6) \approx 2\cdot10^{-8}$: prácticamente nunca. Con $n_a = 600$ la regla 0,10 no produce falsas alarmas, pero es insensible: un PSI de 0,05 (p ≈ 0,0015) sería un cambio real y pasaría como verde. Compruébalo en la hoja `Umbral_por_n` poniendo $n_e = 5.000$.
</details>

**E4 (ε).** En un mes, el bin 10 de DEV (10% de la masa) queda vacío en la muestra actual por un bug de captura. Calcula el aporte de ese bin con ε = 10⁻⁴ y con ε = 10⁻⁶. ¿Qué dice esto del uso del PSI como métrica de calidad de datos?

<details><summary>Solución</summary>

Aporte $\approx e_b\ln(e_b/\varepsilon)$: con 10⁻⁴, $0{,}1\ln(1000) = 0{,}69$; con 10⁻⁶, $0{,}1\ln(10^5) = 1{,}15$ (la tabla del notebook da 0,691 y 1,151). El PSI rojo es correcto en ambos casos, pero su **valor** es arbitrario: depende de una constante sin significado. Para calidad de datos conviene un control directo (bin vacío, fracción de MISSING, niveles no vistos), no la magnitud del PSI.
</details>

**E5 (regla de la cadena).** Una variable de mora tiene 92% de ceros en DEV y 89% en la muestra actual; entre los no-cero, la distribución condicional no cambia. Calcula el PSI con un bin de moda y cualquier partición del resto.

<details><summary>Solución</summary>

Si $a_c = e_c$, los dos términos KL condicionales son 0 y $J = J_{\text{Bern}}(0{,}89;0{,}92) = (0{,}89 - 0{,}92)\ln\frac{0{,}89}{0{,}92} + (0{,}11 - 0{,}08)\ln\frac{0{,}11}{0{,}08} = 0{,}00099 + 0{,}00955 = 0{,}0105$. Todo el PSI está en un grado de libertad. Con $n_e = n_a = 4.000$, $c = 0{,}0005$, $\chi^2_1 = 21$, p ≈ 5·10⁻⁶. Si en cambio se reporta el PSI con 11 bins (moda + 10 deciles), el mismo 0,0105 se compara contra $\chi^2_{10}$: p ≈ 0,02, y con el ruido adicional de los bins condicionales puede no ser significativo (el notebook §5.3 muestra p = 0,13 contra p = 0,0005 del indicador). Separar las etapas gana potencia.
</details>

**E6 (cancelación y Δpuntos).** Un scorecard tiene dos variables. Entre DEV y TTD, $x_1$ pasa masa de su bin de 40 puntos a su bin de 10 puntos (8 puntos porcentuales) y $x_2$ pasa 12 puntos porcentuales de su bin de 5 puntos a su bin de 25 puntos. Calcula Δpuntos por variable y el Δ del score medio. ¿Qué CSI esperarías?

<details><summary>Solución</summary>

$\Delta_1 = 0{,}08 \times (10 - 40) = -2{,}4$; $\Delta_2 = 0{,}12 \times (25 - 5) = +2{,}4$; Δ score medio = 0. Los dos CSI son positivos (cualquier movimiento de masa lo es) y su magnitud depende de las masas iniciales; por ejemplo, si el bin de origen de $x_1$ tenía 30% y el de destino 20%, sus aportes son $(0{,}22-0{,}30)\ln(0{,}22/0{,}30) + (0{,}28-0{,}20)\ln(0{,}28/0{,}20) = 0{,}025 + 0{,}027 = 0{,}052$. El PSI del score puede ser ≈ 0 en media, aunque la forma cambie si los puntos no son lineales (ver el control «Puntos por variable» en §8 del notebook).
</details>

**E7 (diseño).** Diseña la regla de estabilidad para el embudo de un modelo de motos con 60 candidatas, DEV de 6.000 y TTD de 2.500. Debe reemplazar «PSI > 0,25 fuera; NaN fuera».

<details><summary>Solución (una propuesta)</summary>

(1) Binning de estabilidad = binning del modelo (bin de moda si > 35% de masa, MISSING y códigos especiales separados, masa mínima 5% por bin). (2) Compuerta estadística: p-valor χ² con $c = 1/6000 + 1/2500 = 0{,}000567$, BH al 5% sobre las 60. (3) Materialidad: PSI corregido $\text{PSI} - (B-1)c > 0{,}05$ **o** Δpuntos potencial > 5 (estimado con los puntos provisionales). (4) Fuera si pasa ambas compuertas; «vigilar» si solo la estadística. (5) Certificado: el límite superior del IC 95% del PSI poblacional (χ² no central o bootstrap) < 0,10; las variables que no lo logran por $n$ se marcan «no certificables» y requieren justificación de negocio para entrar. (6) Todo se documenta en el expediente con conteos por bin. Las variables categóricas de originación (concesionario, marca) se monitorean aunque no entren al modelo.
</details>

**E8 (código).** Implementa en numpy una función `ic_psi_bootstrap(xe, xa, cortes, B=500)` que devuelva un IC percentil al 95% del PSI y úsala para decidir si `uso_linea_prom_12m` DEV→TTD del Banco Sintético está certificada como estable bajo 0,10. ¿Qué sesgo tiene el IC percentil?

<details><summary>Solución</summary>

```python
def ic_psi_bootstrap(xe, xa, cortes, B=500, eps=1e-4, semilla=0):
    rng = np.random.default_rng(semilla)
    xe, xa = np.asarray(xe), np.asarray(xa)
    v = np.empty(B)
    for i in range(B):
        be = rng.choice(xe, len(xe)); ba = rng.choice(xa, len(xa))
        e = conteos(be, cortes) / len(be); a = conteos(ba, cortes) / len(ba)
        v[i] = psi_np(e, a, eps)
    return np.quantile(v, [0.025, 0.975]), v
```
El PSI observado (0,027) está sesgado hacia arriba en ~$(B-1)c$; el bootstrap agrega **otra** vez ese sesgo (cada réplica suma ruido), así que el IC percentil queda corrido a la derecha. Corrección simple: restar $(B-1)c$ a los extremos, o usar un IC de sesgo corregido. Con límite superior ≈ 0,04, la variable queda certificada bajo 0,10 aunque el cambio sea estadísticamente significativo: estable en sentido material, con deriva real.
</details>

**E9 (taxonomía).** Explica por qué un prior shift mueve el PSI del score y por qué un analista no puede distinguirlo de un covariate shift usando solo el PSI.

<details><summary>Solución</summary>

$p(s) = \pi\,p(s\,|\,1) + (1-\pi)\,p(s\,|\,0)$. Si $\pi$ sube, la mezcla carga más masa donde $p(s|1)$ es alta (scores bajos): la distribución marginal de $s$ cambia y el PSI lo registra (notebook §6: 0,011 con π de 10% a 15%). Pero una distribución marginal de $s$ se puede producir con muchas combinaciones de $(\pi, p(s|0), p(s|1))$: sin el target, el PSI no identifica el mecanismo. La distinción solo aparece con desempeño: si $p(s\,|\,y)$ se mantiene y cambia $\pi$, la curva ROC es la misma y la calibración falla en el nivel; si cambia $p(x)$ con $p(y|x)$ fija, la calibración por banda se mantiene.
</details>

---

## 11. Referencias

- **Siddiqi, N. (2006).** *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley. Segunda edición como *Intelligent Credit Scoring* (2017) (verificar edición). Fuente práctica de los umbrales 0,10/0,25 y del análisis de características; léela para la jerga, no para la inferencia.
- **Yurdakul, B. (2018).** *Statistical Properties of Population Stability Index.* Tesis doctoral, Western Michigan University (dir. J. D. Naranjo). Primer tratamiento sistemático de la distribución del PSI; señala que los umbrales de industria se usan «sin referencia a errores tipo I o II».
- **Yurdakul, B. y Naranjo, J. (2020).** «Statistical properties of the population stability index». *Journal of Risk Model Validation*, 14(4). Versión publicada: $\text{PSI}/(1/n + 1/m) \sim \chi^2_{B-1}$ y umbrales por $n$, $B$ y $\alpha$. La referencia para §3.5.
- **Taplin, R. y Hunt, C. (2019).** «The Population Accuracy Index: A New Measure of Population Stability for Model Monitoring». *Risks*, 7(2), 53. Muestra que el mismo PSI puede ser inocuo (interpolación) o peligroso (extrapolación) y propone medir el efecto en la varianza de las predicciones.
- **du Pisanie, J., Allison, J. S., Budde, C. y Visagie, J. (2023).** «A critical review of existing and new population stability testing procedures in credit risk scoring». arXiv:2303.01227. Revisión de PSI y tests clásicos, problema de muestras grandes y propuesta basada en tamaño de efecto.
- **du Pisanie, J., Allison, J. S. y Visagie, J. (2023).** «A Proposed Simulation Technique for Population Stability Testing in Credit Risk Scorecards». *Mathematics*, 11(2), 492. Calibración de la nula por simulación.
- **Jeffreys, H. (1946).** «An invariant form for the prior probability in estimation problems». *Proceedings of the Royal Society A*, 186. Origen de la divergencia simetrizada que el PSI es.
- **Kullback, S. y Leibler, R. A. (1951).** «On information and sufficiency». *Annals of Mathematical Statistics*, 22(1). La KL.
- **Lin, J. (1991).** «Divergence measures based on the Shannon entropy». *IEEE Transactions on Information Theory*, 37(1). Jensen-Shannon y sus cotas.
- **Cover, T. M. y Thomas, J. A. (2006).** *Elements of Information Theory*, 2.ª ed. Wiley. Regla de la cadena y desigualdad de procesamiento de datos (cap. 2), usadas en §3.3–3.4.
- **Agresti, A. (2013).** *Categorical Data Analysis*, 3.ª ed. Wiley (verificar edición). χ² de homogeneidad, G-test y su asintótica.
- **Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V. y Herrera, F. (2012).** «A unifying view on dataset shift in classification». *Pattern Recognition*, 45(1). Taxonomía covariate/prior/concept shift.
- **Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A. y Lawrence, N. D. (eds.) (2009).** *Dataset Shift in Machine Learning.* MIT Press. Marco general del problema.
- **Rabanser, S., Günnemann, S. y Lipton, Z. C. (2019).** «Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift». *NeurIPS*. Comparación de tests univariados con corrección por multiplicidad frente a tests multivariados.
- **Benjamini, Y. y Hochberg, Y. (1995).** «Controlling the false discovery rate». *JRSS B*, 57(1). La corrección usada en el tablero.
- **Board of Governors of the Federal Reserve System y OCC (2011).** *SR 11-7: Supervisory Guidance on Model Risk Management.* Origen del vocabulario de monitoreo continuo y análisis de resultados; no prescribe el PSI ni sus umbrales. Fue reemplazada el 17-abr-2026 por **SR 26-2** (Fed; OCC Bulletin 2026-13), que mantiene el monitoreo con un enfoque proporcional al riesgo y tampoco prescribe métricas: la elección y su justificación siguen siendo responsabilidad de la institución (ver M22). Para Chile (CMF), revisar la norma vigente sobre modelos de provisiones y gestión de riesgo de crédito antes de afirmar requisitos específicos de monitoreo (ver Serie 1 · E6).
- **Thomas, L. C., Crook, J. y Edelman, D. (2017).** *Credit Scoring and Its Applications*, 2.ª ed. SIAM. Contexto de monitoreo de scorecards y efectos de la política sobre la población.

# M20 · Monitoreo: tablero, semáforos, gatillos y diagnóstico

> **Ficha.** Profundiza: clase 5, parte 4 (láminas 26–30: el tablero de Banco Austral, «¿un modelo nuevo puede nacer con amarillos?», cómo se lee el tablero, jerarquía de acciones); demo de la clase 5, secciones 10–11 (tablero con semáforos y `ScorecardMonitoring`); Lab 3, sección 6 («TU tablero»); clase 6, láminas 2 y 9 (las dos deudas de la clase 5; gatillos de cinco partes) y la matriz RACI. · Prerrequisitos: Serie 1 · M2 (arquitectura temporal, t₀), M3 (definición de default y curvas de maduración), E3 (bootstrap), E7 (eventos extraordinarios); Serie 2 · M08 (PSI/CSI), M12 (Gini/KS y su incertidumbre), M15 (calibración y δ), M18 (swap-set: fila de la cohorte swap-in), M19 (backtesting: binomial, Hosmer-Lemeshow). Prepara M21 (contrato de datos) y M22 (gobierno, RACI, expediente). · Archivos: `M20_monitoreo_tablero.md` (este documento), `M20_monitoreo_tablero.py` (notebook Marimo: 24 meses de producción simulada con verdad conocida), `M20_tablero_monitoreo.xlsx` (plantilla de tablero con semáforos, diagnóstico y gatillos por fórmulas), `M20_tablero_plantilla.csv` (la misma especificación, sin valores). · Tiempo estimado: 3 h de lectura + 2 h de notebook y ejercicios.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**El tablero de Banco Austral (clase 5, lámina 26, corte agosto 2026).** Nueve filas, cada una con valor de hoy, umbral 🟡/🔴, estado y frecuencia (M mensual, T trimestral):

| Indicador | Valor hoy | Umbral 🟡 / 🔴 | Estado | Frec. |
|---|---|---|---|---|
| Gini OOT (caída relativa vs DEV) | 0,675 (−10,8%) | caída > 20% / > 30% | 🟢 | T |
| KS OOT | 0,545 | < 0,30 / < 0,20 | 🟢 | T |
| PSI del score DEV → TTD (8 bandas) | 0,013 | > 0,10 / > 0,25 | 🟢 | M |
| CSI máximo DEV → TTD (`deuda_interna_max_3m`) | 0,138 | > 0,10 / > 0,25 | 🟡 | M |
| Binomial global OOT (nivel) | p = 0,56 | p < 0,05 / < 0,01 | 🟢 | T |
| Bandas con binomial fuera (OOT) | 2 🟡 · 0 🔴 | alguna amarilla / alguna roja | 🟡 | T |
| Hosmer-Lemeshow OOT (forma; p simulado) | χ² 29,0 · p 0,012 (la χ² diría 0,0003) | p < 0,05 / < 0,01 | 🟡 | T |
| Mix D+E en la bandeja TTD | 25,4% (+3,6 pts) | > +3 / > +6 pts | 🟡 | M |
| Cohorte swap-in: mora vs PD calibrada | 9,4% vs 5,1% (p 0,04) · TTD: ago-2027 | p < 0,05 / < 0,01 | 🟡 | T |

Las reglas que el curso dejó escritas:

- **Cinco cosas por indicador**: definición, umbral, frecuencia, responsable y *acción si cruza*. «Sin la quinta es un adorno.»
- **Los umbrales se fijan antes de mirar el dato.** La fila del swap-in ya fija umbral y fecha (agosto 2027) para una cohorte que todavía no madura: un indicador definido hoy para el futuro. El Lab 3 hizo lo mismo con la «TC realizada vs ancla» (⏳, primer corte jul-2027).
- **«¿Un modelo nuevo puede nacer con amarillos?»** Sí: se ajustó con 2024 y se valida con 2025. Lo que decide es *qué* está amarillo: los indicadores de **orden** (Gini, KS) están verdes; los amarillos son de **nivel** y **población**. Recuperar el orden es re-desarrollar (meses, otro comité); corregir el nivel es mover una constante (δ), documentarla y firmarla.
- **Aritmética del amarillo**: «con 9 indicadores a un 5% de tolerancia esperaríamos ~1 amarillo por azar (1 − 0,95⁸ = 34%)». Cinco amarillos apuntando en la misma dirección no son azar. Y al revés: **un tablero todo verde el primer día sería mala noticia** (umbrales flojos o ventana de validación pegada a la de desarrollo).
- **Diagnóstico por patrón** (lámina 28): bandas buenas que subestiman (B2, A2), HL acusando la forma, una variable de deuda que se corre, el mix cargándose a D+E, el swap-in peor que su PD → deriva incipiente de nivel en los tramos buenos con ranking sano.
- **Jerarquía de acciones**: vigilancia reforzada → recalibración del δ (cambio de parámetro documentado) → re-desarrollo (proyecto) → contingencia (volver a la política experta). Acción proporcionada para Austral: vigilancia reforzada 2 meses y recalibración del δ *programada* si el patrón se sostiene. No re-desarrollo. Firma el comité de modelos; la función de monitoreo es dueña del tablero.
- **«Los semáforos valen lo que valga el test de atrás»**: con la χ² del HL a ciegas habría un rojo falso (p 0,0003). La demo lo repitió con `ScorecardMonitoring` de optbinning: el tramo [600, 613) marcaba p = 0,0001 porque en DEV tenía *event rate* 0,0000 — un test contra «esperado cero» revienta con el primer malo.

**Clase 6.** «El tablero vive en el expediente del modelo» (un tablero en el Excel de alguien es un favor personal) y «el gatillo de recalibración del δ queda definido HOY». Un gatillo bien escrito tiene cinco partes: **condición medible · valor de hoy · quién decide · qué acción dispara · en cuánto tiempo**. Saltarse escalones de la jerarquía «es tan malo como no tener gatillos». El caso de la corrida con `nikodym`: Gini HO 0,695 vs OOT 0,694, PSI 0,008, pero calibración OOT roja → diagnosticar por tramos y decidir si basta recalibrar δ. En la RACI, «monitorear el tablero mensual» es R del modelador y A del jefe de modelos; «decidir recalibrar o re-desarrollar» es A del comité. En el Lab 3 (tarea 11) la columna `disparado` no se escribe a mano (se calcula desde la condición), admite «no evaluable» y exige al menos dos responsables distintos.

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **El tiempo.** El tablero de Austral es una foto: validación OOT contra DEV. En producción el target (90+ a 12 meses) llega 12 meses tarde; el curso no trató los **indicadores adelantados** (mora temprana, *first payment default*, *roll rates*, curvas de cosecha) ni cuánto adelantan y cuánto se equivocan.
2. **Monitoreo secuencial.** El semáforo mira el último dato. No se vio Shewhart/EWMA/CUSUM, ni el **ARL** (cuántos meses tarda en saltar la alarma, con y sin deterioro).
3. **Diseño de umbrales.** Se usaron convenciones (Siddiqi para PSI; 0,05/0,01 para tests; 20%/30% para Gini). No se discutió el umbral por distribución nula ni por costo, ni la curva falsa alarma ↔ demora.
4. **Multiplicidad, cuantificada.** El 34% supone tests independientes; en un tablero están correlacionados, varias filas no son tests con α calibrado y la fila «bandas fuera» ya es un mínimo de 8 p-valores. Detalle menor pero útil: la lámina dice 9 indicadores y usa 0,95⁸ (con 9 sería 37%; con 8 tests de p-valor, 34%).
5. **Persistencia** como regla formal (dos meses seguidos, 2 de 3) y su efecto sobre falsas alarmas.
6. **El árbol de diagnóstico** quedó implícito («el diagnóstico está en el patrón»). Aquí se escribe y se prueba contra verdad conocida.
7. **Datos** como familia propia del tablero (contrato, missing, fuera de rango, valores centinela), que M21 formaliza.
8. **Overrides**: su tasa y su desempeño.
9. **Truncamiento**: en producción solo se observa el desempeño de los aprobados; el Gini observable es menor que el de desarrollo sobre la TTD completa.
10. **Calendario vs cosecha**: un shock macro golpea a todas las cosechas vivas a la vez; leerlo como problema «de las cosechas nuevas» lleva a conclusiones equivocadas.

---

## 2. Intuición

**Un tablero es un clasificador de estados con sensores lentos y ruidosos.** El estado verdadero del modelo es uno de pocos: sano, nivel corrido, población movida, orden roto, datos rotos, política cambiada. Cada indicador es un sensor con tres propiedades: **qué estado detecta**, **cuánto rezago** tiene (meses entre originar y poder medir) y **cuánto ruido** (su tasa de falsas alarmas y su potencia). Monitorear es inferir el estado a partir de sensores que llegan a destiempo.

**El embudo temporal.** Una solicitud originada en el mes $t$ produce información en capas:

| Llega en | Qué se puede medir | Familia |
|---|---|---|
| $t$ | contrato de datos, PSI del score, CSI, mix por banda, aprobación, overrides | datos, población, negocio |
| $t+3$ | mora 30+ a MOB 3, *first payment default* | adelantado de nivel |
| $t+6$ | 30+ o 90+ a MOB 6, Gini sobre mora temprana, curva de cosecha a MOB 6 | adelantado de nivel y de orden |
| $t+12$ | 90+ a 12 meses: Gini, KS, binomial, HL, bandas | rezagado (el target del modelo) |

Los indicadores de rezago cero **no usan el target**: detectan causas (la población cambió, el feed se rompió), no consecuencias. Los rezagados miden la consecuencia exacta que importa, pero un año tarde. Los adelantados son el compromiso: consecuencia aproximada con 6–9 meses de ventaja.

**Firmas.** Cada causa deja una huella distinta en las familias:

| Causa | Datos | Población | Negocio | Nivel | Orden |
|---|---|---|---|---|---|
| Feed roto (valor por defecto) | 🔴 inmediato | a veces | a veces | tarde, débil | tarde |
| Llega gente distinta (covariate shift) | — | 🔴 inmediato | aprobación se mueve | 🟢 si el modelo está bien especificado | 🟢 |
| Shock macro (prior/level shift) | — | — | — | 🟡→🔴 con rezago 3–6 | 🟢 |
| *Concept drift* (la relación cambia) | — | — | — | a veces | 🔴 con rezago 6–12 |
| Presión comercial (overrides) | — | — | 🔴 inmediato | 🟢 (O/E ≈ 1) | Gini **sube** |

El diagnóstico por patrón es leer esta tabla al revés. La razón del orden del árbol (datos → orden → nivel → población → negocio) es de costo y de lógica: si los datos están rotos, todos los demás sensores miden basura; si el orden está roto, recalibrar el nivel no sirve.

**Por qué el patrón pesa más que el color.** Un amarillo suelto de un test al 5% aparece, por construcción, uno de cada veinte meses. Cinco amarillos coherentes (todos diciendo «nivel corrido en tramos buenos») tienen una probabilidad conjunta bajo «todo sano» del orden de 10⁻⁴ si fueran independientes (§8.1). La coherencia es la evidencia; el color es solo el dato de entrada.

**Por qué acumular.** Un semáforo mensual compara el último mes contra un umbral. Si la mora temprana sube 20% y el error estándar mensual es del mismo orden, un mes suelto rara vez cruza. Un CUSUM suma los excesos mes a mes y descuenta un «peaje» $k$: el ruido se cancela, el exceso persistente se acumula. Es la diferencia entre mirar una foto y mirar el saldo de una cuenta.

---

## 3. Formalización

### 3.1 Notación

Para el indicador $j$ en el mes de reporte $M$, $I_j(M)$ es su valor (calculado con los datos que existen en $M$: la cosecha $M-L_j$ si su rezago es $L_j$). El semáforo es $s_j(M)=\mathbb 1[I_j(M)\in A_j]+\mathbb 1[I_j(M)\in R_j]\in\{0,1,2\}$, con $R_j\subset A_j$ las regiones roja y amarilla. Si el semáforo es un test, $\alpha_j=P(I_j\in A_j\mid H_0)$ es su tasa de falsas alarmas por mes y $1-\beta_j(\delta)=P(I_j\in A_j\mid \text{deterioro }\delta)$ su potencia.

### 3.2 Multiplicidad y persistencia

Con $m$ indicadores independientes bajo $H_0$:
$$P(\ge 1\ \text{amarillo})=1-\prod_{j=1}^m(1-\alpha_j)\;\overset{\alpha_j=\alpha}{=}\;1-(1-\alpha)^m .$$
Sin independencia valen las cotas $\max_j\alpha_j\le P(\ge1)\le\sum_j\alpha_j$ (la de la derecha es Bonferroni). Con correlación positiva la probabilidad real queda entre ambas y más cerca de la izquierda.

**La fila «alguna banda fuera»** es $\min_{b=1..8}p_b<0{,}05$. Si las bandas son independientes, $P(\min p_b<\alpha)=1-(1-\alpha)^8=0{,}337$: esa fila sola se enciende un tercio de los meses bajo $H_0$. Con Šidák (umbral $1-(1-\alpha)^{1/8}=0{,}0064$) o Bonferroni ($\alpha/8=0{,}00625$) vuelve a ~5%. El notebook lo confirma: R4 se enciende 34% de los meses en el escenario nulo.

**Persistencia.** Regla «amarillo dos meses seguidos». Si los meses son independientes, la falsa alarma por par de meses es $\alpha^2=0{,}0025$. Pero un indicador calculado sobre **ventanas móviles solapadas** (tres cosechas, de las cuales dos se repiten el mes siguiente) tiene autocorrelación. Para ventanas de $w$ cosechas de igual tamaño que avanzan de a una, $\operatorname{corr}(z_M,z_{M-1})=(w-1)/w$. Con $z$ normales bivariadas de correlación $\rho$:

| $\rho$ | $P(\lvert z_1\rvert>1{,}96,\ \lvert z_2\rvert>1{,}96)$ | razón a $\alpha$ |
|---|---|---|
| 0 | 0,0025 | 0,05 |
| 1/3 | 0,0054 | 0,11 |
| 2/3 | 0,0151 | 0,30 |
| 0,9 | 0,0296 | 0,59 |

Con $w=3$ ($\rho=2/3$) la persistencia filtra 3,3 veces, no 20. La regla «2 de 3 meses» con independencia da $3\alpha^2(1-\alpha)+\alpha^3=0{,}0073$.

### 3.3 Observado contra esperado: la suma de Bernoulli heterogéneas

Una cosecha de $n$ créditos con PD del modelo $p_i$ produce $O=\sum_i y_i$ malos. Bajo $H_0$ (el modelo dice la verdad y los defaults son independientes), $O$ es **Poisson-binomial**:
$$E=\sum_i p_i,\qquad V=\sum_i p_i(1-p_i).$$
Si en vez de eso se usa una binomial con la PD media $\bar p$, la varianza es $n\bar p(1-\bar p)$. La diferencia:
$$n\bar p(1-\bar p)-\sum_i p_i(1-p_i)=n\bar p-n\bar p^2-\sum_ip_i+\sum_ip_i^2=\sum_i p_i^2-n\bar p^2=\sum_i(p_i-\bar p)^2\ \ge 0.$$
La binomial con PD media **sobreestima** la varianza (es conservadora); la diferencia es grande cuando las PD son dispersas (cartera entera) y chica dentro de una banda. El notebook usa $z=(O-E)/\sqrt V$ con $p=2[1-\Phi(|z|)]$ para las filas A1, A2 y R2; el check del notebook compara contra la Poisson-binomial exacta (convolución) y difiere en menos de 0,03 cuando $E\gtrsim 30$. Con correlación de defaults (factor común) la varianza real es mayor que $V$: ver M19.

### 3.4 El modelo de tiempo al default y la curva esperada

Sea $p_i$ la PD a 12 meses. Se reparte el hazard acumulado $\Lambda_i=-\ln(1-p_i)$ sobre los meses de vida con pesos $w_m\ge0$, $\sum_{m=1}^{12}w_m=1$ (perfil de maduración, estimado en desarrollo), y un multiplicador de calendario $\mu(t)$:
$$\Lambda_i(j)=\Lambda_i\sum_{m=1}^{j}w_m\,\mu(c_i+m),\qquad P(T_i\le j)=1-e^{-\Lambda_i(j)}.$$
Sin shock ($\mu\equiv1$), $P(T_i\le 12)=1-e^{-\Lambda_i}=p_i$ exactamente, y con $F(j)=\sum_{m\le j}w_m$:
$$P(T_i\le j)=1-e^{\ln(1-p_i)F(j)}=1-(1-p_i)^{F(j)} .$$
Esa es la **curva de cosecha esperada** de un crédito; la de la cosecha es su promedio, con varianza Poisson-binomial $\sum_i q_i(1-q_i)/n^2$, $q_i=1-(1-p_i)^{F(j)}$. Un shock de calendario ($\mu(t)=e^{d}$ desde $t=k$) multiplica el hazard de *todas* las cosechas vivas desde $k$: para PD chicas, $p_i\mapsto\approx p_i e^{d\,\phi_i}$, con $\phi_i$ la fracción del hazard de la cosecha que cae después de $k$.

**Mora temprana.** Si el default ocurre en $T$ (90+), el crédito pasó por 30+ en $T-2$. Sumando los que tocan 30+ y se curan (en el simulador, con probabilidad $0{,}6\,p_i$ en los 6 primeros meses), la mora temprana esperada no es una función cerrada simple de $p_i$; se estima en desarrollo con un **modelo satélite** $\operatorname{logit}P(\text{30+ a MOB }3)=a+b\operatorname{logit}p_i$ y se congela con el artefacto.

### 3.5 Calidad del proxy de cohorte: fiabilidad y adelanto

Sea $r_c$ la tasa temprana y $y_c$ la tasa final de la cosecha $c$. Descomponga cada una en señal (el riesgo verdadero de la cosecha) más ruido binomial:
$$r_c=\rho_c+\varepsilon_c,\quad y_c=\eta_c+\nu_c,\quad \operatorname{Var}\varepsilon_c\approx\frac{\bar r(1-\bar r)}{n},\ \operatorname{Var}\nu_c\approx\frac{\bar y(1-\bar y)}{n}.$$
Si las señales están perfectamente correlacionadas y los ruidos son independientes,
$$\operatorname{corr}(r_c,y_c)=\sqrt{R_r R_y},\qquad R_x=\frac{\operatorname{Var}(\text{señal})}{\operatorname{Var}(\text{señal})+\operatorname{Var}(\text{ruido})}.$$
Con un shock por cosecha de $\sigma_d$ en log-odds, $\operatorname{Var}(\text{señal})\approx[\bar x(1-\bar x)\sigma_d]^2$ (delta method). En el notebook ($n=1.500$, $\sigma_d=0{,}25$, tasa final 6,1%, mora 30+ a MOB 3 de 3,05%): $R_y\approx0{,}84$, $R_r\approx0{,}74$ y la cota da 0,79; la simulación da 0,74. Para 30+ a MOB 6 la cota da 0,85 y la simulación **0,93**: los ruidos *no* son independientes, porque los que caen temprano son parte de los malos finales (57% de los malos pasa por 30+ antes del MOB 6). Consecuencias:

- **Sin variación entre cosechas no hay proxy**: con $\sigma_d\to0$, $R\to0$ y la correlación desaparece (el slider del notebook lo muestra).
- **Adelanto vs ruido**: el MOB 3 adelanta 9 meses pero captura 16% de los malos; el MOB 6 adelanta 6 y captura 57%.
- **El proxy es un modelo**: si el *timing* cambia (gracia, reprogramación, cobranza que cura más), $r_c/y_c$ cambia sin que cambie $y_c$. El notebook: retrasar 2 meses el default desde la cosecha 25 produce un sesgo de −2,5 pp (MOB 3) y −1,6 pp (MOB 6) en la predicción del malo final, sobre una tasa de ~6%.

### 3.6 Shewhart: el semáforo como gráfico de control

Con $z_t$ i.i.d. $N(\delta,1)$ y alarma si $|z_t|>L$, cada mes alarma con probabilidad $\pi(\delta)=1-\Phi(L-\delta)+\Phi(-L-\delta)$ y el tiempo a la alarma es geométrico:
$$\text{ARL}(\delta)=\frac1{\pi(\delta)},\qquad \operatorname{sd}(\text{RL})=\frac{\sqrt{1-\pi}}{\pi}.$$
El semáforo del curso ($p<0{,}05$ bilateral, $L=1{,}96$) tiene $\text{ARL}_0=20$ meses. Con $L=3$, $\text{ARL}_0=370$. La desviación estándar del run length es casi igual a su media: la demora es muy variable.

### 3.7 EWMA (Roberts 1959)

$W_t=\lambda z_t+(1-\lambda)W_{t-1}$, $W_0=0$. Desenrollando la recursión:
$$W_t=\lambda\sum_{i=0}^{t-1}(1-\lambda)^i z_{t-i}.$$
Con $z$ i.i.d. de varianza 1, usando $\sum_{i=0}^{t-1}q^i=\frac{1-q^t}{1-q}$ con $q=(1-\lambda)^2$:
$$\operatorname{Var}W_t=\lambda^2\sum_{i=0}^{t-1}(1-\lambda)^{2i}=\lambda^2\frac{1-(1-\lambda)^{2t}}{1-(1-\lambda)^2}=\lambda^2\frac{1-(1-\lambda)^{2t}}{\lambda(2-\lambda)}=\frac{\lambda}{2-\lambda}\bigl[1-(1-\lambda)^{2t}\bigr].$$
Límites: $\pm L\sqrt{\operatorname{Var}W_t}$. Con $\lambda=1$ es Shewhart; con $\lambda$ chico, memoria larga (vida media $\ln 2/(-\ln(1-\lambda))$: 3,1 meses con $\lambda=0{,}2$). Valores de referencia (Lucas & Saccucci 1990, reproducidos en Montgomery): $\lambda=0{,}2$, $L=2{,}962$ da $\text{ARL}_0\approx500$ y $\text{ARL}(1\sigma)\approx10{,}5$; el notebook lo reproduce por cadena de Markov (499,6 y 10,54).

### 3.8 CUSUM (Page 1954) desde la razón de verosimilitud

Para detectar un cambio de $N(0,1)$ a $N(\delta,1)$, la log-razón de verosimilitud de una observación es
$$\ell(z)=\ln\frac{\phi(z-\delta)}{\phi(z)}=-\frac{(z-\delta)^2}{2}+\frac{z^2}{2}=\delta z-\frac{\delta^2}{2}=\delta\Bigl(z-\frac{\delta}{2}\Bigr).$$
Si el cambio ocurrió en un $\tau$ desconocido, el estadístico de máxima verosimilitud sobre $\tau$ es
$$G_t=\max_{1\le\tau\le t+1}\sum_{s=\tau}^{t}\ell(z_s)=\delta\Bigl(S_t-\min_{0\le j\le t}S_j\Bigr),\qquad S_t=\sum_{s\le t}(z_s-k),\ k=\delta/2,\ S_0=0.$$
Y $S_t-\min_{j\le t}S_j$ satisface la recursión de Page: si $C_{t-1}=S_{t-1}-m_{t-1}$ con $m_{t-1}=\min_{j\le t-1}S_j$, entonces $m_t=\min(m_{t-1},S_t)$ y
$$C_t=S_t-m_t=\max(S_t-m_{t-1},0)=\max(C_{t-1}+z_t-k,\ 0).$$
Alarma si $C_t>h$. Este CUSUM minimiza la demora de detección en el peor caso entre todos los procedimientos con igual $\text{ARL}_0$ (Lorden 1971; optimalidad exacta en Moustakides 1986). El notebook implementa ambas formas (recursión y $S_t-\min S_j$) y verifica que coinciden.

### 3.9 ARL por cadena de Markov (Brook & Evans 1972)

Sea $L(c)$ el ARL arrancando en $C=c$. Por análisis del primer paso:
$$L(c)=1+\int_{0}^{h}L(c')\,f(c'\mid c)\,dc'+L(0)\,P(c+z-k\le0),$$
porque o alarma en este paso (aporta 1), o sigue en $(0,h]$, o vuelve a 0. Discretizando $[0,h]$ en $m$ estados con centros $c_i$, la matriz de transición entre estados no absorbentes $Q_{ij}=P(C_{t}\in\text{estado }j\mid C_{t-1}=c_i)$ da
$$\mathbf L=\mathbf 1+Q\mathbf L\ \Longrightarrow\ \mathbf L=(I-Q)^{-1}\mathbf 1 ,$$
y el ARL es la componente del estado 0. Para el EWMA se hace lo mismo sobre $[-H,H]$. El CUSUM bilateral combina los unilaterales con $1/\text{ARL}=1/\text{ARL}^+ +1/\text{ARL}^-$ (la combinación usual; ver Montgomery). **Aproximación de Siegmund** (unilateral):
$$\text{ARL}\approx\frac{e^{-2\Delta b}+2\Delta b-1}{2\Delta^2},\qquad \Delta=\delta-k,\ b=h+1{,}166;\qquad \Delta=0\Rightarrow \text{ARL}=b^2 .$$
Con $k=0{,}5$, $h=4$: $\text{ARL}_0\approx338$ unilateral y ~168 bilateral, $\text{ARL}(1\sigma)\approx8{,}3$, que es la tabla clásica (168; 26,6; 8,38; 3,34; 2,19 para $\delta=0;0{,}5;1;2;3$ con $h=4$, y 465; 38,0; 10,4; 4,01; 2,57 con $h=5$). El notebook reproduce la tabla por Markov (±1%), por simulación (dentro de su error estándar) y por Siegmund.

### 3.10 CUSUM ajustado por riesgo (Steiner, Cook, Farewell & Treasure 2000)

En crédito cada observación tiene su propia PD. Hipótesis: $H_0$: odds$_i$ = odds del modelo; $H_1$: odds$_i$ = $R\times$ odds del modelo. Entonces $p_{1i}=\frac{Rp_i}{1-p_i+Rp_i}$ y $1-p_{1i}=\frac{1-p_i}{1-p_i+Rp_i}$. La log-razón de verosimilitud del crédito $i$:
$$W_i=y_i\ln\frac{p_{1i}}{p_i}+(1-y_i)\ln\frac{1-p_{1i}}{1-p_i}=y_i\ln R-\ln(1-p_i+Rp_i).$$
Agregando por mes, $W_t=O_t\ln R-\sum_i\ln(1-p_i+Rp_i)$ y $S_t=\max(0,S_{t-1}+W_t)$. Es el CUSUM «óptimo» (en el sentido de §3.8) para un cambio de odds de tamaño exactamente $R$, sin necesidad de estandarizar ni de suponer normalidad. El umbral $h$ no tiene tabla: se calibra por simulación bajo $H_0$ con las PD de la cartera (el notebook lo hace para igualar el $\text{ARL}_0\approx336$ del CUSUM de $z$ y obtiene $h\approx4{,}1$ con $R=1{,}5$).

### 3.11 Umbral por costo

Sea $\pi$ la probabilidad mensual de que empiece un deterioro, $C_{FA}$ el costo de una falsa alarma y $C_d$ el costo por mes de deterioro no detectado. Aproximando (tiempo casi siempre en control, un deterioro a la vez):
$$c(h)=\frac{C_{FA}}{\text{ARL}_0(h)}+\pi\,C_d\,\text{ARL}_1(h).$$
La condición de primer orden, $-C_{FA}\,\text{ARL}_0'/\text{ARL}_0^2+\pi C_d\,\text{ARL}_1'=0$, usa que $\text{ARL}_0$ crece aproximadamente exponencial en $h$ (Siegmund: $\propto e^{2kb}$) y $\text{ARL}_1$ crece lineal (pendiente $\approx1/(\delta-k)$). El óptimo es interior solo si $\pi C_d/C_{FA}$ es chico: con $\pi=1/36$, $\delta=1{,}22\sigma$ y $C_d/C_{FA}=0{,}1$, $h^*=4$; con razón ≥ 1, $h^*$ se va al borde inferior. Es el espíritu del diseño económico de gráficos de control (Duncan 1956), muy simplificado. Su valor práctico es forzar a escribir **qué cuesta una falsa alarma** (la acción equivocada que dispara, no el informe) y **qué cuesta un mes tarde**.

### 3.12 El diagnóstico como clasificador

Sea $\theta\in\{\text{sano},\text{nivel},\text{población},\text{orden},\text{datos},\text{política}\}$ y $\mathbf s(M)$ el vector de semáforos. El diagnóstico óptimo es el MAP, $\hat\theta=\arg\max_\theta P(\mathbf s\mid\theta)P(\theta)$. El árbol del notebook es una aproximación escrita a mano: (i) agrupa indicadores en familias con firmas casi disjuntas (§2), (ii) exige ≥ 2 indicadores persistentes en las familias con tests ruidosos (nivel, población), que es pedir una razón de verosimilitud alta, y (iii) impone un orden que refleja $P(\mathbf s\mid\text{datos})$: con datos rotos cualquier otro patrón es posible, así que «datos» domina. Con verdad conocida, el notebook mide la matriz de confusión del árbol (§8.2).

### 3.13 Por qué el Gini de aprobados es más bajo

El AUC es la probabilidad de concordancia sobre todos los pares (malo, bueno). Al truncar por el cutoff se eliminan justamente los pares más fáciles (malos con PD muy alta contra buenos de PD baja). Si el score de riesgo es $s$ y la población aprobada es $\{s\le c\}$, el AUC de aprobados es la concordancia condicional a ese rango, que en general es menor. En el notebook: Gini a 12 m 0,58 en la TTD completa de DEV y 0,38 en los aprobados de DEV. **La referencia del tablero tiene que ser el Gini de DEV sobre la misma población que se observa en producción.** Corolario: todo cambio del cutoff o de la política de overrides cambia el Gini observado sin que el modelo cambie (§5, trampa 6).

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| Semáforo por umbral de convención (PSI 0,10/0,25; caída de Gini 20%/30%) | Comunicación simple, comparable entre modelos | Nulo; α desconocido y dependiente de $n$ | Indicadores descriptivos (población, mix) | Práctica de scorecards (Siddiqi); tablero del curso |
| Semáforo por p-valor (binomial, HL, Jeffreys, *traffic light*) | α controlado por fila | Tests con supuestos (independencia, esperados ≥ 5) | Nivel y forma de la PD (M19) | Validación de PD en banca (BCBS WP14 2005; Tasche) |
| Umbral por distribución nula (simulada o bootstrap) | α real con el $n$ real | Simular bajo $H_0$ con el artefacto | Cuando la convención no calza con el tamaño (M08) | Buena práctica de validación; no normado |
| Shewhart (p-chart) | Saltos grandes | Mínimo | Datos y contrato (un 15% de filas rotas es un salto) | Control estadístico de procesos (Montgomery) |
| EWMA | Desvíos chicos sostenidos; suaviza | Elegir λ; inercia tras recuperación | Mora temprana, PSI mensual | SPC; salud pública |
| CUSUM tabular | Desvíos chicos sostenidos; óptimo en peor caso | Elegir $k,h$; hay que reiniciar tras alarma | Series O/E de mora temprana y 90+ | SPC (Page 1954); monitoreo de fraude |
| CUSUM ajustado por riesgo / Bernoulli CUSUM | Monitoreo crédito a crédito con PD heterogéneas | Calibrar $h$ por simulación | Carteras chicas, overrides por gestor o concesionario | Cirugía cardíaca (Steiner et al. 2000); Reynolds & Stoumbos (1999) para proporciones |
| Curvas de cosecha contra curva esperada | Separa cosecha, madurez y calendario | Perfil de maduración estimado | Siempre en consumo; lectura en comité | Práctica estándar de retail; análisis edad-período-cohorte (Breeden) |
| *Roll rates* / *flow rates* | Adelanta el 90+ desde el tránsito entre tramos de mora | Datos mensuales de tramo por cuenta | Cobranza, provisiones, alerta temprana de cartera | Cobranza y provisiones (IFRS 9) |
| Mora temprana / *first payment default* (FPD) | Adelanto de 6–9 meses | Satélite a re-estimar si cambia el producto | Productos nuevos, canales nuevos, fraude | Originación de consumo y auto; FPD es la alarma clásica de fraude |
| Benchmarking / *challenger* | Detecta deterioro relativo a otro modelo | Mantener un segundo modelo | Revisión anual; cuando el tablero es ambiguo | Guías de riesgo de modelo (la SR 11-7 lo describía explícitamente; la SR 26-2 es menos prescriptiva) |
| `ScorecardMonitoring` (optbinning) | PSI de score y variables, tests por tramo | Bins CART automáticos; no sabe de tu negocio | Mecánica rápida y reproducible | Demo de la clase 5 |

Sobre regulación, con cautela: en Estados Unidos, la guía interagencias de riesgo de modelo **SR 11-7 (2011) fue reemplazada el 17 de abril de 2026 por la SR 26-2 («Revised Guidance on Model Risk Management»)**, orientada a organizaciones de más de USD 30 mil millones en activos; mantiene el monitoreo continuo («ongoing model monitoring… performing as expected») y el análisis de resultados, y dice que desvíos persistentes fuera de umbrales establecidos pueden justificar ajuste, recalibración o re-desarrollo, pero es menos prescriptiva que la SR 11-7 en técnicas concretas (comparación completa en M22). En el Reino Unido, la PRA SS1/23 (2023) pide monitoreo de desempeño de modelos como parte de su principio de validación (verificar el texto del principio correspondiente). En Chile, la CMF puso en consulta en agosto de 2026 un nuevo Capítulo 21-9 de la RAN sobre metodologías internas para provisiones y capital por riesgo de crédito, que incluye el procedimiento de seguimiento de esas metodologías; su texto final y sus exigencias concretas de monitoreo deben verificarse cuando se publique. Una fintech no bancaria no está sujeta a estas normas, pero su comité, sus financistas y sus auditores leen con ellas en la cabeza.

---

## 5. Cuándo falla: trampas y modos de falla

**1. El tablero de colores sin quinta parte.**
Síntoma: amarillos que se repiten mes tras mes y nadie hace nada. Causa: indicador sin acción ni dueño escritos. Detección: auditar que cada fila tenga acción, responsable y plazo; contar amarillos persistentes sin ticket. Qué hacer: toda fila del tablero se vincula a un gatillo (§6); si una fila no puede disparar ninguna acción, se saca del tablero.

**2. Umbrales elegidos después de mirar.**
Síntoma: los umbrales cambian justo cuando algo se pone amarillo. Causa: umbral contaminado por el dato (el curso: «todos en la sala lo saben»). Detección: fecha de aprobación del umbral en el control de versiones anterior al primer valor medido. Qué hacer: umbrales versionados y firmados; cambiarlos requiere comité y se aplica hacia adelante.

**3. Actuar sobre un amarillo suelto.**
Síntoma: recalibraciones o investigaciones frecuentes que no encuentran nada. Causa: multiplicidad (el tablero del notebook tiene al menos un amarillo en 51% de los meses sin deterioro). Detección: tasa de falsas alarmas por fila medida bajo $H_0$ simulada. Qué hacer: persistencia, familias, lectura por patrón; para filas que agregan muchos tests (bandas), Holm/Bonferroni o p mínimo corregido.

**4. Un semáforo que depende de un test inválido.**
Síntoma: rojo con muchos p-valores minúsculos en grupos con pocos esperados. Causa: aproximación χ² del HL con esperados < 5 (Austral: 0,0003 vs 0,012 simulado); test por tramo contra *event rate* esperado 0 (optbinning, tramo [600, 613)). Detección: revisar esperados por celda; comparar p asintótico contra simulado. Qué hacer: p por simulación, agrupar celdas, o test exacto (M19).

**5. El Gini «cae» porque se compara contra la población equivocada.**
Síntoma: caída de Gini de 30–40% desde el primer mes de producción. Causa: referencia = Gini de DEV sobre la TTD; producción solo observa aprobados (0,58 vs 0,38 en el notebook). Detección: recalcular la referencia en DEV restringido al cutoff vigente. Qué hacer: referencia congelada por población y por política; si cambia el cutoff, se recalcula la referencia (y se documenta).

**6. El Gini «sube» y nadie pregunta por qué.**
Síntoma: caída relativa negativa (mejora) coincidente con más overrides o un cutoff más laxo. Causa: la población aprobada se vuelve más heterogénea (entra gente de PD alta): el ranking parece mejor sin que el modelo cambie. En el notebook, con overrides al 35% de los rechazados, la «caída» del Gini a 12 m pasa a −35%. Detección: leer el Gini junto con la aprobación y el override rate. Qué hacer: medir el Gini también sobre una población de referencia fija (aprobados por score, sin overrides).

**7. El proxy temprano miente porque cambió el timing.**
Síntoma: la mora temprana mejora, el 90+ no. Causa: período de gracia, reprogramaciones, cambio en cobranza temprana o en la fecha de la primera cuota. Detección: la razón (malo final)/(mora temprana) por cosecha se desplaza; las curvas de cosecha cambian de forma, no solo de nivel. Qué hacer: re-estimar el satélite al cambiar producto o cobranza; monitorear también el MOB 6 y la forma de la curva.

**8. Shock de calendario leído como efecto de cosecha.**
Síntoma: «las cosechas nuevas son peores» y se endurece la originación, pero las viejas también se deterioran desde el mismo mes calendario. Causa: shock macro que multiplica el hazard de todas las cosechas vivas. Detección: en el gráfico de cosechas por calendario, las curvas se despegan en la misma fecha y no en el mismo MOB. En el notebook, R2/R3 (rezago 12) se encienden antes de lo esperable porque las cosechas previas a $k$ reciben el shock en sus últimos meses. Qué hacer: descomposición edad-período-cohorte o, al menos, comparar por mes calendario; la respuesta a un shock macro es de nivel (δ, provisiones), no de re-desarrollo.

**9. El dato roto silencioso.**
Síntoma: nada rojo en población ni en desempeño, pero la aprobación sube un poco. Causa: el feed entrega un valor por defecto *dentro del rango válido* (0 en `uso_linea`). En el notebook, con 15% de filas rotas, el CSI de esa variable queda en ~0,08 (verde) y el PSI del score en ~0,01; solo el contrato lo ve. Detección: contrato de datos con reglas distribucionales (masa en valores centinela contra la historia), no solo de rango (M21). Qué hacer: contingencia (filas afectadas a política experta), corregir, re-puntuar; no recalibrar.

**10. Umbrales de convención con tamaños muy distintos.**
Síntoma: PSI que nunca se mueve en la cartera grande y que salta todos los meses en la cartera chica. Causa: el PSI nulo escala como $(B-1)(1/n_e+1/n_a)$ (M08). En el notebook, con 1.500 solicitudes al mes, el p95 nulo del PSI es 0,012 contra un umbral de 0,10: la convención es ocho veces más holgada que la nula. Qué hacer: umbrales por percentil de la nula para el $n$ de cada cartera, o al menos reportar el p-valor junto al PSI.

**11. Persistencia ilusoria por ventanas solapadas.**
Síntoma: la regla «dos meses seguidos» dispara casi tanto como la de un mes. Causa: ventanas móviles de 3 cosechas con autocorrelación 2/3 (§3.2). Qué hacer: exigir persistencia en ventanas no solapadas, o usar un CUSUM (que es la forma correcta de «acumular»).

**12. El tablero todo verde el día 1.**
Síntoma: ningún amarillo durante meses. Causa: umbrales flojos, ventana de validación pegada a DEV, o indicadores sin potencia (carteras chicas). Detección: potencia de cada fila contra un deterioro de referencia (ARL₁). Qué hacer: el curso lo dijo: es mala noticia; revisar umbrales y $n$.

**13. La referencia que se mueve.**
Síntoma: deterioro lento que nunca cruza. Causa: comparar cada mes contra el mes anterior o contra una ventana móvil. Detección: el valor acumulado contra DEV congelado crece monotónicamente. Qué hacer: referencia congelada (DEV) para gobierno; la ventana móvil solo como diagnóstico adicional.

**14. Resolución insuficiente.**
Síntoma: con intensidad baja nada se enciende nunca. Causa: con $n$ mensual chico el error estándar es del orden del deterioro. En el notebook, un shock de nivel de odds × 1,13 (intensidad 0,2) no produce diagnóstico en 9 meses en 5 de 6 semillas. Qué hacer: acumular (CUSUM), agregar trimestralmente, usar eventos más frecuentes (30+ en vez de 90+), y declarar en la *model card* el deterioro mínimo detectable y en cuánto tiempo.

---

## 6. Puente con ingeniería

**El tablero es código declarativo.** Cada indicador es una entrada de una especificación versionada; el tablero es una vista materializada de un almacén de valores *append-only*. Un ejemplo de especificación (el CSV `M20_tablero_plantilla.csv` tiene las mismas columnas):

```yaml
modelo: consumo_v1.3.0            # hash del artefacto congelado (M21)
referencia: {muestra: DEV, hash: 9f2c…, poblacion: aprobados_cutoff_530}
indicadores:
  - id: A1
    familia: nivel
    definicion: mora30_mob3_vs_satelite
    funcion: monitoreo.o_vs_e(cosecha=M-3, evento="30+", mob=3, esperado="satelite_v1")
    direccion: menor            # p-valor: malo si baja
    umbral: {amarillo: 0.05, rojo: 0.01}
    rezago_meses: 3
    frecuencia: mensual
    persistencia: {regla: "2_seguidos"}
    secuencial: {tipo: cusum, k: 0.5, h: 4.0}
    responsable: modelador
    accion: {amarillo: vigilancia_reforzada, rojo: gatillo_recalibracion}
    aprobado: {por: comite_modelos, fecha: 2026-09-15, acta: CM-2026-031}
```

**Contratos.** Entrada: artefacto (hash), mes de reporte, cosechas usadas (lista explícita), población (aprobados, overrides aparte). Salida por indicador y mes: valor, estado, $n$, eventos, esperado, ventana, versión de la especificación, `run_id`. «SIN DATO» es un estado, no un verde (el Lab 3 lo llamó «no evaluable»).

**Tests tipo CI.**

1. *Umbrales antes del dato*: para cada indicador, `fecha_aprobacion_umbral < fecha_primer_valor`; un PR que cambia umbrales falla si no trae acta.
2. *Replay idempotente*: recalcular un mes pasado con la misma especificación y los mismos datos da exactamente los mismos valores (sin semillas implícitas; el p simulado del HL con semilla derivada del `run_id`).
3. *Monotonía del estado*: para dirección «mayor», valor₁ ≤ valor₂ ⇒ estado₁ ≤ estado₂ (test de propiedad).
4. *Tasa de falsas alarmas*: correr el tablero sobre $H_0$ simulada con el artefacto (el escenario «ninguno» del notebook) y verificar que cada fila de p-valor esté en su α ± tolerancia y que la fila de bandas esté marcada como «α real 34%».
5. *Tests de inyección*: los seis escenarios del notebook como **tests de regresión del detector**: «un feed con 15% de ceros debe dar DATOS ROTOS en el mismo mes»; «un shock de odds × 1,35 debe dar NIVEL antes de k + 9». Si un cambio de código empeora la detección, el CI lo ve.
6. *Rezagos*: el indicador de rezago $L$ calculado en $M$ solo puede leer cosechas $\le M-L$ con MOB completo (test contra fuga temporal, Serie 1 · M2).
7. *Máquina de estados de persistencia*: casos con meses sin dato, trimestrales y reinicio del CUSUM tras alarma atendida.

**Qué se congela y qué se versiona.** Congelado con el artefacto: referencia (distribuciones de DEV por banda y por variable, Gini de referencia por población), satélite de mora temprana, perfil de maduración $F(j)$, $k$ y $h$ de los CUSUM. Versionado con acta: umbrales, familias, reglas de persistencia, árbol de diagnóstico, gatillos. Append-only: valores y estados mensuales (el curso: «el histórico mensual, no solo el último mes»).

**Del gatillo al ticket.** Las cinco partes del gatillo son los campos de un ticket: condición (la regla que lo abrió), valor de hoy (el snapshot), quién decide (asignado), acción (plantilla), plazo (SLA). Un gatillo disparado abre el ticket automáticamente; cerrarlo exige la decisión firmada. Así el audit trail (M22) registra no solo que el indicador cruzó, sino qué se hizo y quién lo decidió.

**El DAG de datos.** Cada indicador depende de datos con distinta madurez: solicitudes del mes (día 1), desempeño a MOB 3 (cierre de mes + 3), etc. El orquestador calcula cada fila cuando su insumo está disponible y marca «SIN DATO» si no lo está, en vez de calcular con datos incompletos (una cosecha con MOB 5 contada como MOB 6 subestima la mora).

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy (notebook) | Librería | Diferencias y convención | En producción |
|---|---|---|---|---|
| PSI | $\sum(a-e)\ln(a/e)$ con ε para ceros y renormalización | `scipy.stats.entropy(a,e)+entropy(e,a)` (divergencia de Jeffreys) | `entropy` normaliza sus argumentos; con ε sin renormalizar hay una diferencia de orden ε. El check usa las mismas proporciones | numpy (trivial) con bins congelados |
| AUC / Gini | Rangos promedio (Mann–Whitney), empates ½ | `sklearn.metrics.roc_auc_score` | Idénticos incluidos empates (el check usa scores redondeados para forzar empates) | sklearn |
| Binomial bilateral | log-comb por sumas acumuladas; suma de probabilidades ≤ la observada con tolerancia $1+10^{-7}$ | `scipy.stats.binomtest(...).pvalue` | Es exactamente el método de scipy; la «doble cola mínima» ($2\min$) da otro número (M19) | scipy |
| O vs E Poisson-binomial | $z=(O-E)/\sqrt V$, normal | Sin función en scipy; exacto por convolución (numpy) | Diferencia < 0,03 en p con $E\gtrsim30$; con pocos eventos usar exacto | numpy exacto si $E<30$ |
| Hosmer-Lemeshow | Grupos por PD, p por simulación | `statsmodels.stats.diagnostic_gen.test_chisquare_binning` (solo p asintótico, df = g − 2 por defecto; M23) | Grupos con `array_split` (tamaños iguales, empates partidos) | numpy + semilla trazable |
| Satélite logístico | Newton-Raphson (IRLS) | `statsmodels.Logit` | Iguales a 1e-6 | statsmodels (errores estándar incluidos) |
| EWMA | Recursión con $W_0=0$ | `pandas.Series.ewm(alpha=λ, adjust=False).mean()` | pandas arranca en el primer dato; se antepone un 0 para igualar. Con `adjust=True` pondera distinto al inicio | numpy (5 líneas, sin sorpresas) |
| CUSUM | Recursión de Page | Forma $S_t-\min S_j$ (vectorizada) | Idénticas (identidad de §3.8). Ojo: `statsmodels…breaks_cusumolsresid` es un test de quiebre estructural sobre residuos OLS, **no** un gráfico de control | numpy |
| ARL | Cadena de Markov (Brook–Evans), simulación, Siegmund | Tabla de Montgomery; en R, el paquete `spc` (verificar nombres de funciones y versión) | Markov con $m=300$ estados reproduce la tabla a ±1% | Markov (determinista y rápido) |
| h del CUSUM ajustado por riesgo | Simulación bajo $H_0$ | Sin estándar | Depende de la distribución de PD de la cartera: recalibrar si cambia el mix | Simulación con semilla fija, versionada |

Regla general: la mecánica de cada indicador puede venir de librerías; la **lógica del tablero** (rezagos, persistencia, familias, árbol, gatillos) es código propio, y ahí es donde van los tests de §6.

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral, releído con las herramientas del módulo

**Familias.** Orden: Gini −10,8% y KS 0,545 → verdes. Población: PSI 0,013 (verde), CSI máximo 0,138 (amarillo), mix D+E +3,6 pts (amarillo). Nivel y forma: binomial global p 0,56 (verde), bandas con 2 amarillas (B2 p 0,020; A2 p 0,036), HL p 0,012 (amarillo), swap-in p 0,04 (amarillo). Datos y negocio: no estaban en el tablero (en producción deben estar).

**¿Cuán raro es el patrón?** Hay 11 tests de p-valor en el tablero (global, 8 bandas, HL, swap-in). Bajo independencia y $H_0$, $P(\ge1\ \text{amarillo})=1-0{,}95^{11}=43\%$: tener amarillos no dice nada. Pero $P(\ge4\ \text{de}\ 11)=0{,}0016$ y $P(\ge5)=0{,}0001$. Los tests no son independientes (HL, bandas y global usan los mismos 2.004 créditos), así que estas cifras son orientativas; aun así, cuatro amarillos de nivel que además apuntan en la misma dirección (tramos buenos que subestiman) no son azar. La lectura del curso es correcta.

**¿Persistente?** El tablero de Austral es una sola medición: con la regla de persistencia de este módulo, **ninguna familia de nivel está «encendida» todavía**; lo que hay es una **señal incipiente** (≥ 2 amarillos coherentes en una medición). La planilla del módulo reproduce esto con los valores de Austral en el mes 12: diagnóstico POBLACIÓN (CSI y mix amarillos en dos mediciones seguidas, con valores ilustrativos en el mes 11), señal incipiente de NIVEL, gatillo de vigilancia reforzada disparado y recalibración del δ **no** disparada. Es exactamente la acción del curso: vigilancia 2 meses y recalibración programada si el patrón se sostiene.

**¿Con qué rapidez se vería un deterioro en Austral?** OOT tiene 2.004 créditos en 6 cosechas: ~334 por mes, tasa de malos 5,94%. El error estándar mensual de la tasa a 12 m es $\sqrt{0{,}0594\cdot0{,}9406/334}=1{,}29$ pp. Un deterioro de 20% relativo (+1,19 pp) es $\delta=0{,}92\sigma$: el semáforo lo detecta en ~6,6 meses (con ARL₀ 20) y un CUSUM $h=4$ en ~9,5 meses, **más 12 de rezago**. Con mora 30+ a MOB 3 (supuesto: tasa base 2,5%), el mismo 20% es $\delta=0{,}59\sigma$: ~11 meses con el semáforo o ~20 con el CUSUM $h=4$, más 3 de rezago. El adelanto de 9 meses se come en parte por la menor potencia (menos eventos): en carteras de este tamaño, el proxy temprano gana pero menos de lo que su rezago promete. (La tasa de 2,5% es un supuesto; el curso no reportó mora temprana.)

**El rojo falso del HL.** Con la χ² asintótica (p 0,0003) el tablero tendría un 🔴 y, por el árbol, un gatillo de recalibración inmediato; con el p simulado (0,012) es 🟡. Un semáforo es una función de un test: si el test está mal especificado, la acción también.

### 8.2 Banco Sintético (notebook)

- **Artefacto**: 7 variables, cutoff 530 puntos, aprobación de referencia 80,1%, mix D+E de referencia 27,9% de las solicitudes; δ = +0,058 para anclar la PD media de DEV a la verdadera (DEV tuvo, por azar, 11,28% de malos contra 11,80% de PD verdadera). Gini a 12 m: 0,58 sobre la TTD de DEV, **0,38 sobre los aprobados** (la referencia correcta del tablero).
- **Mora temprana (aprobados de DEV)**: 30+ a MOB 3 = 3,05%; 30+ a MOB 6 = 7,35%; 90+ a 12 m = 6,10%. $P(\text{malo}\mid 30+\ \text{MOB3})=33\%$ (5,3 veces la tasa base) pero captura solo 16% de los malos (57% a MOB 6). φ a nivel de crédito = 0,20.
- **Proxy de cohorte** ($\sigma_d=0{,}25$): correlación con el malo final 0,74 (MOB 3; RMSE 1,0 pp) y 0,93 (MOB 6; RMSE 0,6 pp). Retrasando 2 meses el timing desde la cosecha 25: sesgo −2,5 pp y −1,6 pp.
- **Escenario nulo** (4 semillas × 24 meses): P(≥ 1 semáforo 🟡/🔴 en un mes) = 51%; R4 (bandas) se enciende 34% de los meses; P(≥ 1 indicador persistente) = 20%; el árbol da un diagnóstico falso en 2% de los meses. El p95 nulo del PSI es 0,012 y el del CSI máximo 0,009 (umbral 0,10); el de la caída relativa del Gini a 12 m es ~3% (umbral 20%): con este $n$, los umbrales del curso son muy holgados para población y orden, y justos para los tests.
- **¿Qué se enciende primero?** (intensidad 0,5, $k=8$, medianas de 6 semillas, primer mes con indicador persistente): *dato roto* → D1 en el mes 9 (es decir, $k$ y $k+1$); *población* → P2, P3 y N1 en el mes 9, P1 en el 13–14; *overrides* → N2 en el 9; *nivel* → R4 ~10–11, A2 en el 12, R2 en el 14, R3 en el 15 y A1 en el 18–19 (el semáforo de mora temprana parpadea); *ranking* → A3 (Gini sobre mora temprana) en el 16, R1 (Gini a 12 m) en el 22: **el adelantado de orden gana 6 meses**.
- **Diagnóstico** (árbol en el mes $k+9=17$): con intensidad 0,5, acierto 100% en 36 corridas (6 escenarios × 6 semillas). Primer mes con diagnóstico correcto (mediana): datos rotos, población y overrides en el 9; nivel en el 14; orden roto en el 16. Con intensidad 0,2: datos rotos y overrides siguen al 100%; nivel se diagnostica en 1 de 6 corridas; población y orden en 0 de 6 (quedan en SIN SEÑAL). Esa es la resolución del tablero mensual con 1.500 solicitudes.
- **ARL** (tabla del notebook, §7): Markov reproduce la tabla de Montgomery ($h=4$: 166,5 vs 168; 8,37 vs 8,38). A igual ARL₀ = 20 meses que el semáforo, un CUSUM ($h=2{,}04$) detecta 0,5σ en 9,8 meses y un EWMA ($L=1{,}60$) en 8,8, contra 12,6 del semáforo; a 2σ son equivalentes. En unidades de crédito (1.200 aprobados/mes, mora temprana 3%, +20% relativo = 1,22σ): semáforo 4,3 meses, CUSUM de igual ARL₀ 3,5, CUSUM $h=4$ 6,3 (con una falsa alarma cada ~28 años en vez de cada 20 meses).
- **Curva operativa**: con $C_d/C_{FA}=0{,}1$ y $\pi=1/36$, el $h$ de costo mínimo es 4 (ARL₀ 330, ARL₁ 6,3).
- **Overrides**: con overrides al azar, su O/E de mora temprana se parece al de los aprobados por score (ambos ~1,2–1,3 en el escenario de nivel por defecto), con malo a 12 m de ~41% contra ~9%: el modelo los ordena bien; que valgan la pena es una pregunta de precio.

### 8.3 Crédito de motos (ilustrativo, supuestos explícitos)

Supuestos: 400 créditos originados por mes, mora 30+ a MOB 3 base 6% (producto más riesgoso y con cuota temprana), 25 concesionarios con 5 a 40 créditos al mes.

- **Cartera total**: error estándar mensual $\sqrt{0{,}06\cdot0{,}94/400}=1{,}19$ pp. Un deterioro de +25% relativo (+1,5 pp) es $\delta=1{,}26\sigma$: semáforo ~4,1 meses, CUSUM de igual ARL₀ ~3,4, CUSUM $h=4$ ~6,0; más 3 meses de rezago. Monitorear la mora temprana mensual *sí* funciona a este tamaño.
- **Por concesionario**: con 40 créditos al mes el mismo +25% es $\delta=0{,}40\sigma$ (semáforo ~14,6 meses; CUSUM $h=4$ ~39); con 5 créditos es invisible mes a mes. Aquí el instrumento es el **CUSUM ajustado por riesgo acumulado crédito a crédito** (§3.10), que usa la PD individual y no necesita tamaño mensual; su $h$ se calibra por simulación con la mezcla de PD del concesionario.
- **First payment default**: en motos el FPD (no paga la primera cuota) es primero una alarma de **fraude** (identidad, «testaferro», concesionario que coloca a cualquiera) y solo después de riesgo de crédito. Su gatillo va a fraude y comercial, no a modelos: es la familia «negocio/política», no «nivel».
- **Timing**: si la primera cuota se difiere (promociones «primera cuota en 60 días») el satélite de mora temprana queda sesgado (trampa 7): se versiona por producto.
- **Recupero de la moto**: afecta la severidad, no la PD; los *roll rates* de 60 → 90 cambian cuando cambia la política de retiro del vehículo. Un tablero de PD no debe confundir un cambio de política de cobranza con un cambio de riesgo: la fila de *roll rate* 30 → 60 → 90 va con su propio dueño (cobranza).
- **Overrides**: el concesionario presiona por excepciones. Dos filas: override rate por concesionario y O/E de mora temprana de los overrides con CUSUM acumulado. Si los overrides de un concesionario pagan peor que su PD, se retira la facultad.

**Roll rates, en una línea formal.** Con tramos $b\in\{0,30,60,90+\}$, la matriz de transición mensual $P_{ab}(t)$ se estima con los conteos de cuentas; el pronóstico de 90+ a $h$ meses es $\mathbf e_0^\top\prod_{s=1}^{h}P(t+s)\,\mathbf e_{90}$. Un aumento sostenido en $P_{30\to60}$ adelanta el 90+ unos dos meses. El notebook no simula tramos mensuales (su simulador solo tiene el mes del 30+ y del 90+); es una extensión natural.

---

## 9. Preguntas de comité

**1. «Hay cinco amarillos. ¿Por qué no recalibramos ya?»**
Porque ninguno es persistente todavía y el tablero, por construcción, tiene amarillos la mitad de los meses sin que pase nada (51% en el escenario nulo del notebook). Lo que sí hay es coherencia: los cinco hablan de nivel en tramos buenos. Eso justifica vigilancia reforzada con un gatillo escrito hoy: si en la próxima medición al menos dos indicadores de nivel siguen amarillos (o uno pasa a rojo) y el orden sigue verde, se recalibra δ en 30 días. Recalibrar ahora sería actuar sobre una sola medición; no hacer nada sería ignorar un patrón con probabilidad ~10⁻³ bajo «todo sano».

**2. «¿Cuánto tardaríamos en enterarnos si el modelo se deteriora?»**
Depende del deterioro y del indicador: demora total = rezago + ARL₁. Para un aumento de 20% en la mora con nuestro volumen, la mora temprana lo detecta en ~3 + 4 a 6 meses; el Gini a 12 m, si el orden se rompe, en ~12 + 2 a 8 meses. Un deterioro menor que ~10% relativo no lo detectamos antes de un año con este $n$. Eso va escrito en la *model card* como limitación.

**3. «¿Por qué el Gini de producción es tanto más bajo que el de desarrollo?»**
Porque en producción solo vemos aprobados y el truncamiento elimina los pares más fáciles de ordenar. La referencia del tablero es el Gini de desarrollo sobre los aprobados con el mismo cutoff (0,38 contra 0,58 en el ejemplo del notebook). Si cambiamos el cutoff o crecen los overrides, la referencia se recalcula y se documenta.

**4. «La mora temprana mejoró. ¿Podemos relajar el cutoff?»**
No sin mirar el timing. Si cambió la cobranza temprana, la fecha de primera cuota o hubo reprogramaciones, la mora temprana baja sin que baje el 90+. Antes de relajar: verificar la razón malo/mora temprana de las cosechas maduras recientes y la forma de las curvas de cosecha a MOB 6.

**5. «¿Quién decide y en qué plazo?»**
Lo dice cada gatillo: la contingencia de datos la decide el dueño del dato con el jefe de modelos en 24–48 horas; la recalibración del δ la aprueba el comité de modelos a propuesta del jefe de modelos en 30 días; el re-desarrollo lo abre el comité de riesgo con plan en 60 días. Quien construye el modelo no valida ni aprueba (RACI, M22).

**6. «El PSI está en 0,013. ¿La población está estable?»**
El PSI del score está lejos de 0,10, pero eso no dice mucho: su percentil 95 bajo estabilidad depende del $n$ (con 1.500 solicitudes al mes es ~0,012, notebook §6), así que 0,013 está en el borde de lo que el azar produce. Y el CSI de una variable está en 0,138: el score no lo acusa porque otras variables compensan (M08). Estable en el score, no en las variables; por eso el CSI está en el tablero.

**7. «¿Por qué no usamos solo semáforos, que todos entienden?»**
Los usamos para comunicar. Para decidir, cada fila de desempeño tiene además un CUSUM: a igual tasa de falsas alarmas detecta desvíos chicos y sostenidos 20–30% antes, y a igual velocidad produce muchas menos falsas alarmas. El semáforo del curso implica aceptar una falsa alarma cada 20 meses por indicador de p-valor; con las 13 filas del notebook hay al menos un amarillo en la mitad de los meses sin deterioro.

**8. «¿Qué pasa si el tablero entero se pone rojo de un mes a otro?»**
Primero, datos: un cambio simultáneo en muchas familias casi siempre es un feed roto o un cambio de definición. Por eso el árbol mira datos primero y el gatillo de contingencia tiene plazo de horas. Solo descartado eso se lee el patrón.

---

## 10. Ejercicios

**E1. Multiplicidad del tablero del notebook.** El tablero tiene 4 filas de p-valor simples al 5% (A1, A2, R2, R3) y una fila R4 = mínimo de 8 p-valores al 5%. Suponiendo independencia, calcule la probabilidad de al menos un amarillo en un mes sin deterioro. Luego, la probabilidad de que una fila simple esté amarilla dos meses seguidos si su ventana solapa 2/3 (use la tabla de §3.2).

<details><summary>Solución</summary>

R4 equivale a 8 tests: en total $4+8=12$ tests independientes. $P=1-0{,}95^{12}=0{,}46$. La simulación del notebook da 51%: la independencia no se cumple exactamente y hay filas de convención que aportan poco. Persistencia con $\rho=2/3$: $0{,}0151$ por mes, es decir, 30% de $\alpha$; con independencia sería $0{,}0025$. La persistencia filtra, pero mucho menos de lo que la cuenta ingenua sugiere.
</details>

**E2. Varianza del EWMA.** Demuestre que $\operatorname{Var}W_t=\frac{\lambda}{2-\lambda}[1-(1-\lambda)^{2t}]$ y calcule el límite asintótico con $\lambda=0{,}2$ y $L=2{,}86$. ¿Cuántos meses tarda la varianza en llegar al 95% de su valor asintótico?

<details><summary>Solución</summary>

Ver §3.7. Asintótico: $\sqrt{0{,}2/1{,}8}=0{,}333$; límite $2{,}86\times0{,}333=0{,}953$. $1-(0{,}8)^{2t}\ge0{,}95\iff0{,}64^t\le0{,}05\iff t\ge\ln0{,}05/\ln0{,}64=6{,}7$: 7 meses. Por eso los primeros meses usan el límite exacto (más estrecho).
</details>

**E3. CUSUM a mano.** Con $k=0{,}5$ y $h=4$, calcule el CUSUM superior para $z=(1{,}1;\ 1{,}3;\ -1{,}6;\ 0{,}5;\ 2{,}2;\ 0{,}6;\ 4{,}0;\ 1{,}6)$ y el mes de alarma. Luego, con Siegmund, el ARL unilateral para $\delta=1$.

<details><summary>Solución</summary>

$C$: $0{,}6$; $1{,}4$; $0$ (1,4 − 1,6 − 0,5 < 0); $0$; $1{,}7$; $1{,}8$; $5{,}3$ → alarma en el mes 7. Siegmund: $\Delta=0{,}5$, $b=5{,}166$, $2\Delta b=5{,}166$; $\text{ARL}=(e^{-5{,}166}+5{,}166-1)/(2\cdot0{,}25)=(0{,}0057+4{,}166)/0{,}5=8{,}34$ meses (Markov: 8,37).
</details>

**E4. Fiabilidad del proxy.** Con $n=1.500$, tasa final 6,1%, mora 30+ a MOB 3 de 3,05% y shock por cosecha $\sigma_d=0{,}25$ en log-odds, calcule $R_y$, $R_r$ y la cota $\sqrt{R_rR_y}$. ¿Qué pasa con $\sigma_d=0{,}10$? ¿Por qué el notebook da una correlación mayor que la cota para el MOB 6?

<details><summary>Solución</summary>

Señal final: $0{,}061\cdot0{,}939\cdot0{,}25=1{,}43$ pp; ruido: $\sqrt{0{,}061\cdot0{,}939/1500}=0{,}62$ pp; $R_y=1{,}43^2/(1{,}43^2+0{,}62^2)=0{,}84$. Temprana: señal $0{,}0305\cdot0{,}9695\cdot0{,}25=0{,}74$ pp; ruido $0{,}44$ pp; $R_r=0{,}74$. Cota: $\sqrt{0{,}84\cdot0{,}74}=0{,}79$ (el notebook: 0,74; el shock de calendario no escala exactamente como $p(1-p)$). Con $\sigma_d=0{,}10$: señales 0,57 y 0,30 pp; $R_y=0{,}46$, $R_r=0{,}31$; cota 0,38. Para MOB 6 la cota asume ruidos independientes, pero los malos que ya cayeron a MOB 6 son los mismos que cuentan en el malo final: ruidos positivamente correlacionados, correlación mayor que la cota.
</details>

**E5. CUSUM ajustado por riesgo.** Derive $W_i=y_i\ln R-\ln(1-p_i+Rp_i)$. En un mes con 1.230 aprobados, $\sum_i\ln(1-p_i+1{,}5p_i)=19{,}0$ y $O=58$ malos tempranos, calcule $W_t$. Si $S_{t-1}=2{,}3$ y $h=4{,}1$, ¿hay alarma?

<details><summary>Solución</summary>

Derivación en §3.10. $W_t=58\ln1{,}5-19{,}0=58\cdot0{,}4055-19{,}0=23{,}52-19{,}0=4{,}52$. $S_t=\max(0;2{,}3+4{,}52)=6{,}82>4{,}1$: alarma. Nota: $\sum\ln(1-p+Rp)\approx(R-1)\sum p=0{,}5E$ para PD chicas, así que $E\approx38$ y el mes tuvo 58 contra 38 esperados.
</details>

**E6. Diagnóstico.** Mes de corte con estados persistentes: D1 🟢, P1 🟢, P2 🟡, P3 🔴, N1 🟡, N2 🟢, A1 🟢, A2 🟢, A3 🟢, R1 🟢, R2 🟡 (solo este mes), R3 🟢, R4 🟡. ¿Diagnóstico y acción? ¿Qué pregunta harías antes de firmar?

<details><summary>Solución</summary>

Datos verdes, orden verde. Nivel: R2 no es persistente y R4 sola es ruido esperado (34% de los meses): familia nivel apagada. Población: P2 y P3 persistentes → encendida. Negocio: N1 encendida, pero población va primero en el árbol. Diagnóstico: POBLACIÓN. Acción: el modelo sigue válido; revisar estrategia y cutoff (M17) y verificar la calibración por banda (que es lo que dirá si el corrimiento cae en zonas donde el modelo calibra mal). Pregunta: ¿de dónde viene la gente nueva (canal, campaña, concesionario) y hay variables fuera del modelo que se movieron con ella?
</details>

**E7. Varianza Poisson-binomial.** Demuestre $n\bar p(1-\bar p)-\sum p_i(1-p_i)=\sum(p_i-\bar p)^2$. Una banda de 235 créditos tiene PD entre 1,2% y 1,6% (media 1,42%); la cartera completa tiene PD entre 0,1% y 60% (media 6%, desviación estándar 9%). ¿En cuál importa usar Poisson-binomial?

<details><summary>Solución</summary>

Ver §3.3. Banda: $\sum(p_i-\bar p)^2\le235\cdot0{,}002^2\approx0{,}001$ contra $n\bar p(1-\bar p)=3{,}29$: irrelevante. Cartera de $n=2.000$: $\sum(p_i-\bar p)^2=2000\cdot0{,}0081=16{,}2$ contra $2000\cdot0{,}06\cdot0{,}94=112{,}8$: la binomial con PD media sobreestima la varianza 17% (el error estándar, 8%): es conservadora y pierde potencia en el test global.
</details>

**E8. Umbral por costo.** Con $\pi=1/36$ y $\delta=1{,}22$, use la curva del notebook (ARL₀ y ARL₁ por $h$) para encontrar el $h$ óptimo con $C_d/C_{FA}=0{,}1$ y con $1$. Explique por qué con razón 1 la solución «alarmar casi siempre» es absurda y qué costo falta en la fórmula.

<details><summary>Solución</summary>

Con 0,1: $c(3)=1/116+0{,}0028\cdot4{,}87=0{,}0222$; $c(4)=1/330+0{,}0028\cdot6{,}26=0{,}0204$; $c(5)=1/911+0{,}0028\cdot7{,}64=0{,}0223$ → $h^*\approx4$. Con 1: el término de demora pesa 10 veces más y $c$ es mínima en el borde ($h\le1$). Falta el costo de la **acción** que dispara la falsa alarma (recalibrar sin causa mueve precios y aprobación), el desgaste de la alarma y el costo de oportunidad del equipo. Si $C_{FA}$ incluye la recalibración innecesaria, la razón real es ≪ 1.
</details>

**E9. Código: persistencia «k de n».** Escriba una función numpy que, dada la matriz de estados (indicador × mes), marque persistencia «2 de los últimos 3 meses». Calcule su tasa de falsas alarmas con independencia y compárela con «2 seguidos».

<details><summary>Solución</summary>

```python
def persistente_k_de_n(est, k=2, n=3):
    a = (np.nan_to_num(est, nan=0.0) >= 1).astype(int)          # indicador × mes
    ventana = np.stack([np.roll(a, s, axis=1) for s in range(n)]).sum(axis=0)
    ventana[:, : n - 1] = 0                                       # sin historia suficiente
    return ventana >= k
```
Con independencia: $P(\ge2\ \text{de}\ 3)=3\alpha^2(1-\alpha)+\alpha^3=0{,}0073$ contra $\alpha^2=0{,}0025$ de «2 seguidos»: casi el triple de falsas alarmas, a cambio de tolerar un mes verde intermedio (útil cuando el indicador parpadea, como A1 en el escenario de nivel).
</details>

**E10. Diseño: tres gatillos para motos.** Escriba tres gatillos de cinco partes para la cartera de motos de §8.3: uno de datos, uno de nivel y uno de overrides por concesionario.

<details><summary>Solución</summary>

(a) *Contingencia de datos*: condición = % de solicitudes con renta, patente o RUT del vendedor fuera de contrato > 1% en el día; valor hoy = 0,3%; decide = dueño del dato + jefe de modelos; acción = solicitudes afectadas a evaluación manual, corregir integración con el concesionario; plazo = 24 h. (b) *Recalibración del δ*: condición = CUSUM de O/E de mora 30+ a MOB 3 ($k=0{,}5$, $h=4$) en alarma y 90+ a MOB 6 con p < 0,05 en dos cosechas seguidas, con Gini temprano sin caída > 20%; valor hoy = CUSUM 2,1, p 0,31; decide = comité de modelos; acción = re-anclar δ con cosechas maduras recientes, acta; plazo = 30 días. (c) *Overrides por concesionario*: condición = CUSUM ajustado por riesgo ($R=1{,}5$, $h$ calibrado por simulación) de los overrides del concesionario en alarma, o override rate > 20% de sus rechazos en el trimestre; valor hoy = S = 1,4, 12%; decide = comité de crédito; acción = suspender facultad de override del concesionario y auditar expedientes; plazo = 15 días.
</details>

---

## 11. Referencias

- **Page, E. S. (1954).** «Continuous inspection schemes». *Biometrika*, 41(1/2), 100–115. — El origen del CUSUM; la recursión de §3.8.
- **Roberts, S. W. (1959).** «Control chart tests based on geometric moving averages». *Technometrics*, 1(3), 239–250. — El origen del EWMA.
- **Lucas, J. M., & Saccucci, M. S. (1990).** «Exponentially weighted moving average control schemes: properties and enhancements». *Technometrics*, 32(1), 1–12. — Tablas de ARL del EWMA y elección de λ y L.
- **Brook, D., & Evans, D. A. (1972).** «An approach to the probability distribution of CUSUM run length». *Biometrika*, 59(3), 539–549. — El método de cadena de Markov del notebook.
- **Montgomery, D. C.** *Introduction to Statistical Quality Control* (8.ª ed., Wiley, 2019/2020; verificar edición). — Capítulos de CUSUM y EWMA: tablas de ARL, diseño de $k$, $h$, λ, L; referencia de trabajo.
- **Siegmund, D. (1985).** *Sequential Analysis: Tests and Confidence Intervals*. Springer. — La aproximación de ARL usada en §3.9 y en la planilla.
- **Lorden, G. (1971).** «Procedures for reacting to a change in distribution». *Annals of Mathematical Statistics*, 42(6), 1897–1908. — Optimalidad asintótica del CUSUM.
- **Moustakides, G. V. (1986).** «Optimal stopping times for detecting changes in distributions». *Annals of Statistics*, 14(4), 1379–1387. — Optimalidad exacta del CUSUM (criterio de Lorden).
- **Steiner, S. H., Cook, R. J., Farewell, V. T., & Treasure, T. (2000).** «Monitoring surgical performance using risk-adjusted cumulative sum charts». *Biostatistics*, 1(4), 441–452. — El CUSUM ajustado por riesgo; se traslada directo a crédito con PD por crédito.
- **Reynolds, M. R., & Stoumbos, Z. G. (1999).** «A CUSUM chart for monitoring a proportion when inspecting continuously». *Journal of Quality Technology*, 31(1), 87–108 (verificar páginas). — Bernoulli CUSUM para proporciones.
- **Duncan, A. J. (1956).** «The economic design of X̄ charts used to maintain current control of a process». *Journal of the American Statistical Association*, 51(274), 228–242 (verificar páginas). — Diseño económico de gráficos de control; la idea de §3.11.
- **Siddiqi, N. (2017).** *Intelligent Credit Scoring* (2.ª ed.). Wiley. — Informes de monitoreo de scorecards (estabilidad, características, desempeño); origen práctico de los umbrales PSI.
- **Anderson, R. (2007).** *The Credit Scoring Toolkit*. Oxford University Press. — Monitoreo, informes de *final score*, análisis de cosechas y *roll rates* en la práctica.
- **Thomas, L. C., Crook, J. N., & Edelman, D. B. (2017).** *Credit Scoring and Its Applications* (2.ª ed.). SIAM. — Marco general; monitoreo y modelos de comportamiento.
- **Breeden, J. L.** *Reinventing Retail Lending Analytics* (Riskbooks; verificar edición y año). — Descomposición edad-período-cohorte de curvas de cosecha: separa calendario de cosecha (trampa 8).
- **Basel Committee on Banking Supervision (2005).** *Studies on the Validation of Internal Rating Systems* (Working Paper 14). — Validación de PD y backtesting; base de M19.
- **Board of Governors of the Federal Reserve System, OCC, FDIC (2026).** *SR 26-2: Revised Guidance on Model Risk Management* (17 de abril de 2026), que reemplaza a SR 11-7 (2011) y SR 21-8. — Monitoreo continuo y análisis de resultados en el marco de riesgo de modelo.
- **Prudential Regulation Authority (2023).** *SS1/23 — Model risk management principles for banks*. — Principios de riesgo de modelo en el Reino Unido, incluido el monitoreo de desempeño (verificar el principio específico).
- **Comisión para el Mercado Financiero (2026).** Consulta pública del nuevo Capítulo 21-9 de la RAN sobre metodologías internas para provisiones y capital por riesgo de crédito (agosto de 2026). — Marco chileno en construcción; verificar texto final.
- **Serie 2 · M08, M12, M15, M19, M21, M22.** — PSI/CSI y su nula; Gini/KS y su incertidumbre; δ y calibración; backtesting; contrato de datos y artefacto; gobierno y RACI.

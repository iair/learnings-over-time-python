# M18 · Swap-set y el problema contrafactual

> **Ficha.** Profundiza: clase 4 v1 (láminas 28–31 y reserva 37), clase 4 v21 (láminas 41–45 y 51), clase 5 (lámina 9 y fila «cohorte swap-in» del tablero, lámina 27); notebooks `demo_c4_austral_v2` (secciones 8–9) y `demo_clase5_validacion_austral` (sección 9). · Prerrequisitos: Serie 1 · E1 (reject inference), Serie 1 · E3 (intervalos para carteras chicas), Serie 1 · M3 (curvas de maduración); Serie 2 · M15 (calibración) y M17 (estrategia y cutoff, en paralelo). Prepara M19 (backtesting), M20 (monitoreo: fila de la cohorte swap-in) y M22 (gobierno). · Archivos: `M18_swap_set.md` (este documento), `M18_swap_set.py` (notebook Marimo), `M18_swap_2x2_ic.xlsx` (matriz 2×2 con IC binomiales, test de la cohorte y costo de exploración). · Tiempo estimado: 3 h de lectura + 2 h de notebook y ejercicios.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**La pregunta.** «¿A QUIÉNES vamos a aprobar que hoy rechazamos, y a quiénes al revés?» Cambiar de política no mueve solo una tasa: intercambia clientes concretos. La receta del curso fija **la misma tasa de aprobación** para las dos políticas y las cruza en una matriz 2×2 sobre una muestra con desempeño (OOT). *Swap-in*: los que la nueva aprueba y la vieja rechazaba («riesgo que se asume»). *Swap-out*: los que la vieja aprobaba y la nueva rechaza («riesgo que se suelta»). El intercambio conviene si la tasa de malos del swap-in es menor que la del swap-out; a igual aprobación «no hay excusa comercial: el swap-set aísla el efecto puro del modelo».

**Los números de Banco Austral (OOT, iso-aprobación 90,2%).** La política vigente asumida para la demo son dos knock-outs: mora interna vigente (`dias_mora_ult > 0`) o mora ≥ 30 días en el sistema (`peor_mora_sistema_ult ≥ 30`); rechazan el 9,8% del flujo. El scorecard aprueba sus k = 1.808 mejores scores (desempate estable; cutoff equivalente 533):

| Política vigente ↓ · Scorecard → | Aprueba (score ≥ 533) | Rechaza (score < 533) |
|---|---|---|
| Knock-outs aprueban | 1.680 · 3,0% · ambas aprueban | 128 · 23,4% · **swap-out** |
| Knock-outs rechazan | 128 · 9,4% · **swap-in** | 68 · 38,2% · ambas rechazan |

En conteos: 51/1.680, 30/128, 12/128 y 26/68. Mora de la cartera aprobada 81/1.808 = 4,48% → 63/1.808 = 3,48% (−22%) con la misma aprobación. «Los que ambas rechazan traen 38,2%: los knock-outs no estaban locos, solo eran insuficientes». El notebook de clase agrega dos matices: (1) el 9,4% del swap-in es el triple del 3,0% de la cartera, pero el número relevante es el 23,4% del swap-out; (2) en producción KO y score suelen convivir (KO primero), y esa política combinada **no** es la del ejercicio, que midió el reemplazo puro.

**La confesión metodológica.** El 33,1% de las 32.301 solicitudes históricas fue rechazado y no tiene desempeño. El modelo aprende «cómo se comportan los que alguien ya filtró»; si el cutoff nuevo entra a población antes rechazada, ahí extrapola. El curso decide no hacer reject inference y declararlo como limitación: partir conservador en el territorio nuevo, monitorear el swap-in como cohorte aparte y recalibrar con la evidencia que genere.

**La cohorte en el tablero (clase 5).** Sobre OOT, la cohorte swap-in (n = 128) tiene PD calibrada media 5,1% (esperaba 6,5 malos) y llegaron 12 (9,4%): test binomial p = 0,040, 🟡. «En el territorio donde la política vieja no dejaba entrar a nadie, la PD sí subestima: el modelo extrapola». La fila del tablero fija umbral (p < 0,05 / < 0,01), frecuencia trimestral y fecha de lectura de la bandeja TTD (agosto 2027): «un indicador definido HOY para el futuro».

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **La política vieja era hipotética.** Los knock-outs se aplicaron *a posteriori* sobre créditos que el banco sí había cursado; por eso las cuatro celdas tienen desempeño. En un cambio de política real la vieja estuvo vigente: el swap-in son rechazados históricos **sin** desempeño. El curso lo insinúa («la letra chica») pero no cuantifica cuánto cambia la conclusión.
2. **La matriz se calcula dentro de las cursadas.** El 90,2% es aprobación *de cursadas*, no de la puerta completa (32.301 solicitudes). Los rechazados reales del banco no están en ninguna celda.
3. No hay **álgebra** de la mejora: por qué 4,48% → 3,48% es exactamente 128/1.808 × (9,4% − 23,4%), qué pasa fuera de iso-aprobación, ni la relación con iso-riesgo e iso-pérdida.
4. No hay **inferencia**: 128 casos por celda y ningún intervalo. ¿Es significativo 9,4% vs 23,4%? ¿Lo es la caída de un punto de la cartera? (Sí y sí, pero por razones distintas; §3.4–3.5.)
5. La cohorte swap-in se testea con p = 0,04 sin preguntar la **potencia**: con n = 128, un swap-in que de verdad tiene 9,4% contra PD 5,1% pasa como 🟢 casi la mitad de las veces (§3.8).
6. «Monitorear y recalibrar» no dice **cómo aprender** el swap-in sin sesgo: champion/challenger, bandas de exploración aleatorizadas, su costo y su valor.
7. El swap-set se cuenta en créditos; no en pesos (EAD, pérdida) ni por segmento.
8. El 22% de mejora se lee como efecto del modelo; es efecto del modelo **contra una política asumida**. El número depende de qué política vieja se asuma (§8.2).

Reject inference (augmentation, parceling, fuzzy, sus supuestos y por qué casi nunca se puede validar) está en Serie 1 · E1; aquí solo se usa lo que hace falta para entender el sesgo del swap-in.

---

## 2. Intuición

**Una política es un conjunto; cambiarla es una diferencia simétrica.** La cartera aprobada es un conjunto de solicitudes. Pasar de la política vieja a la nueva agrega unas (swap-in) y quita otras (swap-out); lo que ambas aprueban no cambia y, por lo tanto, **no puede explicar la diferencia**. Toda la variación de la mora de la cartera vive en dos celdas chicas. Eso tiene tres consecuencias que el resto del módulo formaliza:

- **Qué comparar.** A igual aprobación, la mora de la cartera baja si y solo si el que entra es mejor que el que sale. No importa que el que entra sea peor que el promedio de la cartera (9,4% vs 3,0%): el promedio no se va. Fuera de iso-aprobación, el umbral relevante del swap-in pasa a ser la mora media de la cartera vieja.
- **Cuánta evidencia hay.** El ruido de las 1.680 que ambas aprueban aparece en las dos carteras y se cancela en la diferencia. La comparación de carteras es un diseño **pareado**: mucho más preciso que comparar dos carteras independientes del mismo tamaño. Los intervalos de 4,48% y 3,48% se solapan y aun así la diferencia es claramente distinta de cero.
- **Dónde está la incertidumbre.** En la vida real, de las dos celdas que importan una es observable (el swap-out: la vieja los aprobó, tienen desempeño) y la otra no (el swap-in: la vieja los rechazó). Todo el argumento a favor del cambio depende de una tasa que no se ha observado nunca. Es un problema **contrafactual**: ¿qué habría pasado con quien no recibió crédito si lo hubiera recibido?

**Por qué el modelo subestima justo ahí.** El swap-in es, por definición, la región donde el modelo nuevo contradice a la política vieja: «estos que ustedes rechazaban, yo los veo buenos». Si el modelo se entrenó con aprobados de la vieja, en esa región tiene poca o ninguna evidencia (extrapola). Y aunque la tuviera, al elegir precisamente a los que él mismo puntúa bien entre los rechazados, selecciona los casos donde su error es más favorable: la **maldición del ganador** (*winner's curse*, u *optimizer's curse* en la versión de decisiones). Si además la política vieja usaba información que el modelo no ve (juicio del ejecutivo, documentos, una llamada), el swap-in queda adversamente seleccionado: entran justo los que alguien tenía buenas razones para rechazar.

**Cómo se sale.** Hay tres salidas honestas y ninguna gratis: (1) **cotas y punto de quiebre**: no estimar el swap-in, sino preguntar cuánto tendría que equivocarse la estimación para que la decisión cambie; (2) **supuestos** explícitos (reject inference, Heckman), que se declaran y se estresan; (3) **experimentar**: aprobar al azar una fracción de los que se rechazarían y observar. La tercera es la única que produce evidencia sin supuestos sobre el mecanismo de selección y se paga en pérdida esperada. El notebook cuantifica las tres.

---

## 3. Formalización

### 3.1 Notación y la identidad de la variación de mora

Solicitudes $i=1,\dots,N$; decisión vieja $V_i\in\{0,1\}$ y nueva $W_i\in\{0,1\}$ (1 = aprueba); resultado $Y_i\in\{0,1\}$ (1 = malo, 90+ a 12 meses), definido como resultado **potencial**: lo que habría pasado si se le otorgaba el crédito. Las cuatro celdas:

$$
C_{jk}=\{i: V_i=j,\ W_i=k\},\qquad n_{jk}=|C_{jk}|,\qquad m_{jk}=\sum_{i\in C_{jk}}Y_i,\qquad b_{jk}=m_{jk}/n_{jk}.
$$

$C_{11}$ ambas aprueban, $C_{10}$ swap-out, $C_{01}$ swap-in, $C_{00}$ ambas rechazan. Carteras aprobadas: $K_v=n_{11}+n_{10}$, $K_n=n_{11}+n_{01}$; moras

$$
\text{BR}_v=\frac{m_{11}+m_{10}}{K_v},\qquad \text{BR}_n=\frac{m_{11}+m_{01}}{K_n}.
$$

**Identidad general.** Despejando $m_{11}=K_v\text{BR}_v-m_{10}$ y reemplazando en $\text{BR}_n$:

$$
K_n\,\text{BR}_n = K_v\,\text{BR}_v - n_{10}b_{10} + n_{01}b_{01}.
$$

Restando $K_n\text{BR}_v$ a ambos lados y usando $K_v-K_n=n_{10}-n_{01}$:

$$
K_n(\text{BR}_n-\text{BR}_v)=(n_{10}-n_{01})\text{BR}_v - n_{10}b_{10}+n_{01}b_{01}
= n_{01}(b_{01}-\text{BR}_v)-n_{10}(b_{10}-\text{BR}_v),
$$

$$
\boxed{\ \text{BR}_n-\text{BR}_v=\frac{n_{01}\,(b_{01}-\text{BR}_v)\;-\;n_{10}\,(b_{10}-\text{BR}_v)}{K_n}\ }
$$

Cada celda de intercambio aporta su **exceso de riesgo respecto de la cartera vieja**, ponderado por su tamaño. $C_{11}$ no aparece. La condición de mejora es

$$
\text{BR}_n<\text{BR}_v\iff n_{01}(b_{01}-\text{BR}_v)<n_{10}(b_{10}-\text{BR}_v).
$$

**Caso iso-aprobación** ($n_{01}=n_{10}=s$, $K_n=K_v=K$): los términos en $\text{BR}_v$ se cancelan y

$$
\text{BR}_n-\text{BR}_v=\frac{s}{K}\,(b_{01}-b_{10}),\qquad \text{mejora}\iff b_{01}<b_{10}.
$$

Banco Austral: $\tfrac{128}{1808}(0{,}09375-0{,}234375)=0{,}0708\times(-0{,}1406)=-0{,}00996$, es decir 4,48% → 3,48%. El «−22%» es $-0{,}00996/0{,}0448$.

**Caso expansión pura** ($n_{10}=0$: la nueva aprueba todo lo que aprobaba la vieja y más): $\text{BR}_n-\text{BR}_v=n_{01}(b_{01}-\text{BR}_v)/K_n$. La mora sube si y solo si el swap-in es peor que la **cartera vieja**. Aquí sí la comparación con 3,0% o 4,48% es la correcta; a iso-aprobación no.

**Caso contracción pura** ($n_{01}=0$): baja si y solo si $b_{10}>\text{BR}_v$, lo que casi siempre ocurre si la nueva ordena bien. Por eso «subir el cutoff baja la mora» no es evidencia de nada: la evidencia del modelo está en el intercambio.

**En pesos.** Con pérdida $L_i=\text{LGD}\cdot\text{EAD}_i\cdot Y_i$ (o su esperanza con PD) y $L_{jk}=\sum_{C_{jk}}L_i$, la identidad para totales no necesita normalización:

$$
L_n-L_v=L_{01}-L_{10}.
$$

Para tasas sobre monto, $\ell=L/V$ con volumen $V=\sum \text{EAD}$, la identidad es la misma que la de conteos cambiando $n$ por volumen: $\ell_n-\ell_v=[V_{01}(\ell_{01}-\ell_v)-V_{10}(\ell_{10}-\ell_v)]/V_n$. Iso-aprobación en conteo no implica iso-volumen: si el swap-in tiene montos mayores que el swap-out, $V_{01}\neq V_{10}$ y reaparece el término de nivel.

### 3.2 Iso-aprobación, iso-riesgo, iso-pérdida: la frontera

Ordenemos las solicitudes por score nuevo de mejor a peor, $y_{(1)},y_{(2)},\dots$. Aprobar las $k$ mejores define la **frontera** del modelo:

$$
\text{BR}_n(k)=\frac1k\sum_{j\le k}y_{(j)},\quad L_n(k)=\sum_{j\le k}L_{(j)},\quad V_n(k)=\sum_{j\le k}\text{EAD}_{(j)}.
$$

La política vieja es un punto $(K_v,\text{BR}_v,L_v,V_v)$. Cinco preguntas, cinco $k$:

| Criterio | Definición | Pregunta que responde |
|---|---|---|
| iso-aprobación | $k=K_v$ | ¿cuánta mora ahorro sin tocar el volumen de créditos? |
| iso-riesgo | $\max\{k:\text{BR}_n(k)\le\text{BR}_v\}$ | ¿cuántos créditos más apruebo con la misma mora? |
| iso-tasa de pérdida | $\max\{k: L_n(k)/V_n(k)\le L_v/V_v\}$ | lo mismo, en pesos por peso colocado |
| iso-pérdida total | $\max\{k: L_n(k)\le L_v\}$ | ¿cuánto más apruebo gastando el mismo presupuesto de pérdida? |
| iso-volumen | $\min\{k: V_n(k)\ge V_v\}$ | ¿con cuántos créditos coloco los mismos pesos? |

Tres observaciones.

**(a) La frontera empírica no es monótona.** $\text{BR}_n(k)$ es un promedio acumulado con ruido; en tramos puede bajar. Por eso el criterio usa «el último $k$ que cumple» (una convención; el notebook la implementa igual en numpy y con `pandas.expanding`). Con la frontera suavizada (PD calibrada en vez de $y$) el problema desaparece, pero entonces la frontera es la del modelo, no la de los datos.

**(b) Cuánta aprobación compra una mejora de mora.** En continuo, sea $a$ la tasa de aprobación y $b(a)$ la tasa de malos **marginal** del crédito que está en el cutoff. La mora media es $\text{BR}(a)=\frac1a\int_0^a b(u)\,du$, así que

$$
\text{BR}'(a)=\frac{b(a)-\text{BR}(a)}{a}.
$$

Linealizando alrededor de $a_v$, la ganancia de aprobación a iso-riesgo es

$$
\Delta a\approx\frac{\text{BR}_v-\text{BR}_n(a_v)}{\text{BR}'(a_v)}=\frac{a_v\,[\text{BR}_v-\text{BR}_n(a_v)]}{b(a_v)-\text{BR}_n(a_v)}.
$$

La misma mejora iso-aprobación se traduce en poca aprobación extra si el marginal es muy malo respecto del promedio (curva empinada) y en mucha si el marginal se parece al promedio (curva plana). Por eso iso-riesgo reporta ganancias tan distintas a distintos niveles de aprobación (§8.2: +4,7 pp a 92,5% de aprobación, +14,5 pp a 80%).

**(c) Iso-pérdida no es iso-riesgo.** Si el score nuevo reordena hacia montos grandes de bajo riesgo, la tasa de pérdida sobre monto baja más que la tasa de malos, y el presupuesto de pérdida total permite más aprobación. Si reordena hacia montos grandes de alto riesgo (típico cuando la renta entra al monto y no al score), pasa lo contrario. No hay un criterio «correcto»: el comité debe decir si su restricción es la mora (apetito en tasa), la pérdida (presupuesto en pesos) o el capital (que depende de PD, LGD y EAD por separado).

### 3.3 Inferencia por celda: intervalos binomiales exactos

**Modelo.** Una vez fijadas las dos políticas (el score se ajustó en DEV, los knock-outs no miran $Y$, y $k$ sale de la política vieja), la pertenencia a cada celda **no depende de los $Y$ de la muestra** donde se evalúa. Condicional a las celdas, $m_{jk}\sim\text{Binomial}(n_{jk},\pi_{jk})$ independientes, con $\pi_{jk}$ la tasa de malos de la subpoblación. (Supuesto de independencia entre créditos: razonable dentro de una cohorte; con un shock macro común, los defaults están correlacionados y todos los intervalos de este módulo son optimistas; ver Serie 1 · E3.)

**Clopper-Pearson.** El intervalo exacto a nivel $1-\alpha$ es el conjunto de $\pi$ que no se rechazan con ninguna de las dos colas:

$$
\pi_L=\inf\{\pi:\ P_\pi(X\ge m)>\alpha/2\},\qquad \pi_U=\sup\{\pi:\ P_\pi(X\le m)>\alpha/2\}.
$$

Para calcularlo sin iterar se usa la identidad binomial–beta. Derivemos $\frac{d}{d\pi}P_\pi(X\ge m)$ término a término:

$$
\frac{d}{d\pi}\binom{n}{x}\pi^x(1-\pi)^{n-x}=n\left[\binom{n-1}{x-1}\pi^{x-1}(1-\pi)^{n-x}-\binom{n-1}{x}\pi^{x}(1-\pi)^{n-1-x}\right],
$$

usando $x\binom{n}{x}=n\binom{n-1}{x-1}$ y $(n-x)\binom{n}{x}=n\binom{n-1}{x}$. Al sumar desde $x=m$ hasta $n$ la serie telescopea y queda solo el primer término:

$$
\frac{d}{d\pi}P_\pi(X\ge m)=n\binom{n-1}{m-1}\pi^{m-1}(1-\pi)^{n-m}=\frac{\pi^{m-1}(1-\pi)^{n-m}}{B(m,\,n-m+1)}.
$$

Integrando desde 0 (donde la probabilidad vale 0): $P_\pi(X\ge m)=I_\pi(m,n-m+1)$, la beta incompleta regularizada. Por lo tanto

$$
\pi_L=B^{-1}\!\left(\tfrac{\alpha}{2};\,m,\,n-m+1\right),\qquad \pi_U=B^{-1}\!\left(1-\tfrac{\alpha}{2};\,m+1,\,n-m\right),
$$

con $\pi_L=0$ si $m=0$ y $\pi_U=1$ si $m=n$. Es lo que hacen `statsmodels.proportion_confint(method="beta")` y la planilla con `BETA.INV`; el notebook lo calcula además invirtiendo las colas por bisección en numpy, sin la identidad, y comprueba que coinciden.

Clopper-Pearson es **conservador**: su cobertura real es $\ge 1-\alpha$ para todo $\pi$ y, por la discreción de la binomial, suele ser bastante mayor. Brown, Cai y DasGupta (2001) recomiendan Wilson o Jeffreys cuando se busca cobertura *promedio* cercana a la nominal. En validación de riesgo, donde el error caro es declarar «la PD está bien» cuando no lo está, lo exacto-conservador es defendible para la celda individual; lo que no es defendible es usar Wald ($\hat b\pm z\sqrt{\hat b(1-\hat b)/n}$) con 12 malos.

**Banco Austral (95%).** Swap-in 12/128 = 9,4%, IC [4,9%; 15,8%]. Swap-out 30/128 = 23,4%, IC [16,4%; 31,7%]. Ambas aprueban 51/1.680 = 3,0%, IC [2,3%; 4,0%]. Ambas rechazan 26/68 = 38,2%, IC [26,7%; 50,8%]. Los IC de swap-in y swap-out no se tocan; con otras 128 observaciones en cada celda, el swap-in podría razonablemente estar entre 5% y 16%.

### 3.4 ¿Es significativo 9,4% vs 23,4%? Fisher, Newcombe y compañía

**Fisher exacto.** Tabla $\begin{pmatrix}12&116\\30&98\end{pmatrix}$ (malos, buenos por fila). Bajo $H_0:\pi_{01}=\pi_{10}$ y condicionando en los márgenes (42 malos en 256), el número de malos del swap-in es hipergeométrico:

$$
P(X=x)=\frac{\binom{128}{x}\binom{128}{42-x}}{\binom{256}{42}}.
$$

El p-valor bilateral suma las probabilidades de las tablas tan o menos probables que la observada: $p=0{,}0037$. Barnard y Boschloo (que no condicionan en ambos márgenes) son más potentes: `scipy.stats.boschloo_exact` da $p=0{,}0023$. La conclusión no cambia: la diferencia es real.

**IC de la diferencia (Newcombe 1998, método híbrido de Wilson).** Con $(l_1,u_1)$ y $(l_2,u_2)$ los IC de Wilson de $\hat p_1=b_{01}$ y $\hat p_2=b_{10}$, y $d=\hat p_1-\hat p_2$:

$$
L=d-\sqrt{(\hat p_1-l_1)^2+(u_2-\hat p_2)^2},\qquad U=d+\sqrt{(u_1-\hat p_1)^2+(\hat p_2-l_2)^2}.
$$

La lógica: cada semiancho de Wilson es la «distancia de error» de su proporción en la dirección relevante, y para una diferencia de independientes se combinan en cuadratura. Austral: $d=-14{,}1$ pp, IC 95% [−23,0; −5,0] pp. Agresti-Caffo (sumar un malo y un bueno a cada celda y usar Wald) da [−22,8; −4,9] pp. Fagerland, Lydersen y Laake (2015) comparan estos métodos y recomiendan Newcombe o Agresti-Caffo para uso general; Wald sin corrección no.

### 3.5 La diferencia de carteras es pareada

El error frecuente: «la cartera vieja es 4,48% [3,57; 5,54] y la nueva 3,48% [2,69; 4,44]; los intervalos se solapan, la mejora no es significativa». Es falso por construcción. En iso-aprobación,

$$
\Delta=\text{BR}_n-\text{BR}_v=\frac{m_{01}-m_{10}}{K},\qquad
\operatorname{Var}(\Delta)=\frac{n_{01}\pi_{01}(1-\pi_{01})+n_{10}\pi_{10}(1-\pi_{10})}{K^2}.
$$

$m_{11}$ aparece en las dos carteras con el mismo signo y se cancela: su varianza **no entra**. Si se trataran como independientes se sumaría $2\,n_{11}\pi_{11}(1-\pi_{11})/K^2$ de más. Austral:

- EE pareado: $\tfrac{1}{1808}\sqrt{128(0{,}094)(0{,}906)+128(0{,}234)(0{,}766)}=0{,}32$ pp.
- EE «independiente»: $\sqrt{0{,}0448(0{,}9552)/1808+0{,}0348(0{,}9652)/1808}=0{,}65$ pp, el doble.

IC 95% de Δ escalando el de Newcombe por $s/K=128/1808$: **[−1,63; −0,35] pp**. Excluye cero, mientras los intervalos de las carteras se solapan. Es el mismo fenómeno que el test t pareado frente al de dos muestras.

**Fuera de iso-aprobación** la celda compartida no se cancela del todo. Con $A=m_{11}$:

$$
\Delta=\frac{A+m_{01}}{K_n}-\frac{A+m_{10}}{K_v},\qquad
\operatorname{Var}(\Delta)=\operatorname{Var}(A)\left(\tfrac{1}{K_n}-\tfrac{1}{K_v}\right)^2+\frac{\operatorname{Var}(m_{01})}{K_n^2}+\frac{\operatorname{Var}(m_{10})}{K_v^2},
$$

que la planilla implementa como IC de Wald por método delta (fila «EE de Δ»). Con $K_n\approx K_v$ el primer término es despreciable.

**Qué NO cubre este intervalo.** Es condicional a las celdas y a la muestra OOT. No incluye la incertidumbre de haber elegido la política vieja de referencia, ni el ruido del ajuste del scorecard en DEV (para eso: bootstrap de todo el pipeline, Serie 1 · E3), ni el hecho de que el swap-in, en la vida real, no se observa (§3.6). Es el IC de un ejercicio retrospectivo, no de la política futura.

### 3.6 El problema contrafactual: qué se identifica y qué no

En la vida real, $Y_i$ solo se observa si el crédito se otorgó, es decir si $V_i=1$ en la política histórica. La celda que decide la conveniencia del cambio, $b_{01}=E[Y\mid V=0,W=1]$, está en la región no observada. Tres niveles de supuestos:

**(i) Sin supuestos: cotas de Manski.** Lo único cierto es $0\le b_{01}\le 1$. Reemplazando en la identidad:

$$
\frac{m_{11}}{K_n}\ \le\ \text{BR}_n\ \le\ \frac{m_{11}+n_{01}}{K_n}.
$$

Austral: $\text{BR}_n\in[2{,}82\%;\ 9{,}90\%]$, y la vieja es 4,48%: **sin supuestos, ni el signo de la mejora está identificado**. Más útil es el **punto de quiebre** (análisis de *tipping point*): la tasa del swap-in que haría $\text{BR}_n=\text{BR}_v$,

$$
b_{01}^{*}=\frac{K_n\,\text{BR}_v-m_{11}}{n_{01}}\quad\overset{\text{iso}}{=}\quad b_{10}.
$$

Austral: $b^*_{01}=23{,}4\%$, es decir 2,5 veces el 9,4% estimado. La pregunta para el comité deja de ser «¿cuánto es el swap-in?» y pasa a ser «¿creemos que el swap-in puede ser 2,5 veces peor que lo que dice el modelo?». Con un margen así, la decisión es robusta a casi cualquier sesgo razonable de la estimación; con un múltiplo de 1,2 no lo sería.

**(ii) Selección sobre observables (MAR / ignorabilidad condicional).** Supuesto: $Y\perp V\mid X$ (la política vieja decidió solo con variables que el modelo ve). Entonces $E[Y\mid X,V=0]=E[Y\mid X,V=1]=f(X)$ y $b_{01}=E[f(X)\mid V=0,W=1]$, identificado **si hay solapamiento**: $P(V=1\mid X=x)>0$ para los $x$ del swap-in. Los knock-outs son reglas deterministas: $P(V=1\mid X)=0$ en la región KO. No hay solapamiento, y $f$ en esa región se obtiene **por forma funcional** (el binning, la linealidad en el WoE, la logística). Incluso bajo MAR, el swap-in de un knock-out se identifica solo por extrapolación. Con binning, la extrapolación es literal: el valor nunca visto cae en el bin vecino y hereda su WoE (§8.2 lo muestra con `meses_desde_mora_12m`).

**(iii) Selección sobre no observables (MNAR).** Si la política vieja usó información $U$ correlacionada con $Y$ que el modelo no tiene (juicio del analista, verificación telefónica, documentos), $E[Y\mid X,V=0]\neq E[Y\mid X,V=1]$ incluso con solapamiento. El modelo clásico es el de selección de Heckman (1979): $Y^*=X\beta+\varepsilon$, $V^*=Z\gamma+u$, $(\varepsilon,u)$ normal bivariada con correlación $\rho$. Para los seleccionados, $E[\varepsilon\mid V=1]=\rho\,\sigma_\varepsilon\,\lambda(Z\gamma)$ con $\lambda=\phi/\Phi$ (inversa de Mills); para los rechazados, el signo se invierte. En crédito, con $Y$ binario, la versión es el probit bivariado con selección (Boyes, Hoffman y Low, 1989). Identificarlo bien exige una **restricción de exclusión**: una variable que mueva la aprobación y no el default (por ejemplo, un cambio administrativo de política o de sucursal). Sin ella, la corrección depende solo de la normalidad, y eso es tan frágil como suena.

**Por qué el modelo subestima incluso con todos los datos.** Sea $\hat\eta$ el log-odds estimado y $\eta$ el verdadero. Si $\hat\eta=\eta+e$ con $e\perp\eta$, $e\sim N(0,\tau^2)$, $\eta\sim N(\mu,\sigma^2)$, entonces por normal bivariada

$$
E[\eta\mid\hat\eta]=\mu+\lambda(\hat\eta-\mu),\qquad \lambda=\frac{\sigma^2}{\sigma^2+\tau^2}<1,
$$

y para una celda elegida por tener $\hat\eta$ bajo ($\hat\eta\le c$):

$$
E[\eta-\hat\eta\mid\hat\eta\le c]=(1-\lambda)\big(\mu-E[\hat\eta\mid\hat\eta\le c]\big)>0.
$$

El riesgo verdadero de lo que el modelo eligió como bueno es mayor que el estimado: regresión a la media aplicada a una decisión (Smith y Winkler, 2006, lo llaman *optimizer's curse*). Si $\hat\eta$ fuera exactamente $E[\eta\mid X]$ (un modelo perfectamente calibrado condicional a $X$), no habría sesgo; el problema es que el error de estimación y de especificación **no es uniforme**, y el swap-in concentra justamente la región donde el modelo tiene más error: la que la política vieja nunca dejó entrar. En el notebook, el modelo entrenado **con todos** los datos (algo imposible en la vida real) subestima el swap-in en ≈ 4 pp; el truncamiento agrega 1,4 pp más (§8.2).

### 3.7 Aprender el swap-in: exploración aleatoria, costo y valor de la información

**Diseño.** Cada solicitud que la política vigente rechaza se aprueba al azar con probabilidad conocida $\varepsilon$ (banda de exploración), o solo las que caen en una región candidata $R$ (exploración focalizada; por ejemplo, rechazados por KO con score sobre el cutoff). Sea $Z_i\sim\text{Bernoulli}(\varepsilon)$ independiente de todo. El estimador de Horvitz-Thompson/Hájek de $\theta=E[Y\mid R]$:

$$
\hat\theta=\frac{\sum_{i\in R}Z_iY_i/\varepsilon}{\sum_{i\in R}Z_i/\varepsilon}=\frac{\sum_{i\in R}Z_iY_i}{\sum_{i\in R}Z_i}.
$$

Como $Z\perp Y$, condicional a $n_R=\sum_R Z_i>0$ los explorados son una muestra aleatoria simple de $R$ y $E[\hat\theta\mid n_R]=\theta$. Varianza:

$$
\operatorname{Var}(\hat\theta)\approx\frac{\theta(1-\theta)}{\varepsilon\,N_R},
$$

con $N_R$ el número de solicitudes de la región en el periodo de exploración. Con propensión $\varepsilon_i$ variable (por ejemplo, más exploración cerca del cutoff) se pondera por $1/\varepsilon_i$, y la misma lógica permite **reentrenar** el modelo con los explorados ponderados: es la única fuente de reject inference sin supuestos sobre el mecanismo de selección.

**Costo.** Explorar un rechazado cuesta su pérdida esperada menos el margen que genera:

$$
C(\varepsilon)=\varepsilon\sum_{i\in P}\text{EAD}_i\,\big(\text{LGD}\cdot p_i-m\big),
$$

con $P$ la población elegible (todos los rechazados o solo $R$) y $m$ el margen neto de vida sobre EAD. Explorar todos los rechazados para aprender sobre $R$ paga también a los rechazados fuera de $R$, que suelen ser los peores (en el notebook, 66% de malos). El **costo por observación útil** es $\text{EAD}\,(\text{LGD}\,\bar p_P-m)/s_R$, con $s_R$ la fracción de $P$ que cae en $R$: focalizar baja a la vez el numerador (explorados menos malos) y el denominador (todos útiles).

**Valor de la información para una decisión.** Supongamos que la decisión es abrir la región $R$ (aprobarla en adelante) o no. Conviene si $\text{LGD}\cdot\theta<m$, es decir si $\theta<b^{*}=m/\text{LGD}$. El valor en juego durante la vida de la política es

$$
V=N_R^{\text{fut}}\cdot\overline{\text{EAD}}\cdot|m-\text{LGD}\,\theta|.
$$

Con un estimador aproximadamente normal $\hat\theta\sim N(\theta,se^2)$, la probabilidad de decidir mal es $\Phi\!\left(-|b^*-\theta|/se\right)$ y el arrepentimiento esperado $R(\varepsilon)=V\,\Phi\!\left(-|b^*-\theta|/se(\varepsilon)\right)$. La exploración óptima minimiza $C(\varepsilon)+R(\varepsilon)$. Dos lecturas:

- Si $\theta$ está lejos del quiebre, se necesita poca información para decidir bien: el valor de explorar es bajo *para esta decisión*.
- Si está cerca, cada punto de error estándar vale mucho, pero también hay que aceptar que ningún tamaño razonable elimina el riesgo de decidir mal.

$\theta$ es desconocido ex ante; la versión rigurosa pone un prior sobre $\theta$ (por ejemplo, centrado en la estimación del modelo con varianza inflada por el sesgo esperado) y calcula el valor esperado de la información muestral (*expected value of sample information*). El notebook usa la verdad del generador para mostrar el mecanismo; en producción, se hace el cálculo con el prior y se declara.

**El estimador sin explorar (E0) tiene error cero de varianza y sesgo fijo.** Con muy poca exploración, el directo (E1) tiene más error cuadrático que el modelo sesgado: con 10 explorados útiles, el RMSE de E1 es 14 pp contra 5 pp de sesgo de E0. Un estimador intermedio (E2) recalibra el modelo con un δ propio estimado sobre **todos** los rechazados explorados: usa más datos (menos varianza) a cambio de suponer que el desvío es uniforme en log-odds (algo de sesgo). Es la versión exploratoria del ajuste de intercepto de M15.

**Champion/challenger** es la variante de política completa: una fracción aleatoria del flujo se decide con la política challenger. Cada brazo observa el desempeño de sus propios aprobados, así que el challenger revela su swap-in sin supuestos; el costo es la pérdida extra del challenger si es peor y la operación de dos políticas. Para detectar una diferencia de mora entre brazos se usa el tamaño de muestra de dos proporciones independientes (§3.8).

### 3.8 La cohorte swap-in en monitoreo: test, potencia y fecha

**Test.** La fila del tablero compara la mora observada de la cohorte swap-in con su PD calibrada media $\bar p_0$: $H_0:\pi=\bar p_0$. El curso usó el binomial exacto bilateral de `scipy`, que suma las probabilidades de los resultados tan o menos probables que el observado (*minlike*): 12/128 con $\bar p_0=5{,}1\%$ da $p=0{,}041$. Otras definiciones dan otro número con los mismos datos: unilateral $P(X\ge12)=0{,}031$; bilateral por duplicación de la cola $0{,}062$. En monitoreo la hipótesis de interés es unilateral («la PD subestima»), y un validador preferiría el unilateral y declararlo; lo inaceptable es elegir la definición después de ver el número.

**Potencia exacta.** Para $H_0:\pi=p_0$ a nivel $\alpha$, la región de rechazo es $\mathcal R=\{x:\ p\text{-valor}(x)\le\alpha\}$ y la potencia contra $\pi=p_1$ es $\sum_{x\in\mathcal R}\binom{n}{x}p_1^x(1-p_1)^{n-x}$. Por la discreción, la potencia **no es monótona** en $n$ (dientes de sierra; en el notebook, n = 128 y n = 150 tienen casi la misma potencia). Aproximación normal para el $n$ necesario:

$$
n\approx\left(\frac{z_{1-\alpha/2}\sqrt{p_0(1-p_0)}+z_{1-\beta}\sqrt{p_1(1-p_1)}}{p_1-p_0}\right)^2.
$$

Austral, $p_0=5{,}1\%$, $p_1=9{,}4\%$, $\alpha=5\%$, potencia 80%: $n\approx(1{,}96\cdot0{,}220+0{,}842\cdot0{,}292)^2/0{,}043^2=248$ (exacto: 250). Son cifras del test bilateral del curso; con el unilateral que recomendamos arriba, la normal da $n\approx200$ y el exacto cruza 80% por primera vez en $n\approx215$ (con dientes de sierra hasta ~240). A n = 128 ambas versiones tienen la misma potencia. Con n = 128 la potencia es 55%: **un swap-in que de verdad tiene casi el doble de mora que su PD (9,4% vs 5,1%) pasa como 🟢 el 45% de las veces**. El amarillo del curso no es «evidencia débil de un problema pequeño»; es una lectura afortunada de un test sin potencia.

**Fecha de lectura.** Si el swap-in origina $f$ créditos al mes, la cohorte alcanza $n$ en $\lceil n/f\rceil$ meses y su tasa de 12 meses se lee 12 meses después del último originado. Con 25 al mes (Austral tuvo 128 en 6 meses de OOT, ≈ 21 al mes): 10 meses de originación + 12 de maduración. Por eso la fila del tablero lleva **fecha**, además de umbral: antes de esa fecha el verde no significa nada. Dos atajos legítimos: indicadores tempranos (30+ a 3 y 6 meses, convertidos a 90+ a 12 meses con la curva de maduración de Serie 1 · M3) y tests secuenciales (Wald, 1945), que permiten mirar cada mes sin inflar el error tipo I si el plan de mirada se fija antes. Mirar cada mes con el test de muestra fija sí infla el α.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| Swap-set retrospectivo sobre cursadas (el del curso) | Efecto puro del modelo a igual aprobación, con desempeño en las 4 celdas | Bajo | Siempre, como primera evidencia; solo vale para políticas **hipotéticas** sobre aprobados | Práctica estándar de desarrollo de scorecards (Siddiqi; Anderson) |
| Swap-set con swap-in inferido (reject inference: augmentation, parceling, fuzzy) | Llena la celda sin desempeño | Medio | Cambio real de política sin datos de rechazados | Muy común; supuestos MAR casi nunca verificables (Hand y Henley; Crook y Banasik). Ver Serie 1 · E1 |
| Cotas de Manski y punto de quiebre | Robustez sin supuestos: ¿cuán equivocado puede estar el swap-in antes de que cambie la decisión? | Bajo | Siempre que la decisión dependa de una celda no observada | Buena práctica de validación; poco institucionalizada |
| Desempeño de rechazados en otras instituciones (bureau) | Proxy observable del swap-in | Medio (datos; definición de malo distinta) | Mercados con información positiva consolidada | Común en EE.UU./RU; en Chile, la Ley 21.680 crea el Registro de Deuda Consolidada (NCG 540 de la CMF, 2025): verificar fecha de operación y alcance de consulta |
| Modelos de selección (Heckman, probit bivariado) | Corrige MNAR si hay restricción de exclusión | Alto (supuestos, estimación) | Investigación; rara vez en producción | Académico (Boyes et al. 1989; Banasik et al. 2003) |
| Champion/challenger aleatorizado | Desempeño insesgado de la política completa, incluido su swap-in | Alto (pérdida del challenger, operación) | Cambios grandes de política con flujo suficiente | Práctica en tarjetas y consumo masivo; exige aprobación de gobierno y revisión legal |
| Banda de exploración (aprobar al azar ε bajo el corte o dentro de KO) | Observar la región rechazada sin supuestos | Pérdida neta de los explorados | Como mecanismo permanente y pequeño (1–5%) | Fintech y emisores de alto volumen; se documenta como política |
| Exploración focalizada / estratificada | Mismo objetivo, menor costo por observación útil | Menor | Cuando la región swap-in es identificable ex ante | Recomendable sobre la banda uniforme |
| Bandits (ε-greedy, Thompson) | Exploración adaptativa | Alto de gobierno; target con 12 meses de retraso | Alto volumen y proxies tempranos del target | Uso creciente en fintech; difícil de defender ante un validador sin diseño previo |
| Rampa de cutoff con cohortes | Limita exposición mientras se aprende | Bajo | Siempre que se entra a territorio nuevo | El «partir conservador» del curso |

Tres notas de contexto:

- **Regulación.** En EE.UU., la guía supervisora de riesgo de modelos SR 11-7 (2011), que exigía análisis de resultados (*outcomes analysis*), fue reemplazada el 17-abr-2026 por SR 26-2 (Fed) y el Bulletin 2026-13 (OCC) (ver M22). Verificar el texto vigente antes de citarlo. Ninguna de las dos prescribe un método de swap-set. Aprobar al azar bajo el corte es tratar distinto a solicitantes iguales; en jurisdicciones con normas de crédito justo (ECOA/Reg B en EE.UU.) y protección al consumidor (en Chile, Ley 20.555 del Sernac Financiero), el diseño debe pasar por cumplimiento. En general se acepta mejor aprobar al azar a quien se habría rechazado que rechazar al azar a quien se habría aprobado. Verificar con la norma vigente.
- **Por qué casi nadie usa Heckman en producción.** Porque la restricción de exclusión rara vez existe, porque el resultado depende de la normalidad bivariada y porque la literatura empírica (Banasik, Crook y Thomas, 2003, con datos de un banco que aprobó a todos) encontró ganancias pequeñas de corregir el sesgo muestral.
- **La evidencia del «aprobar a todos».** Los pocos estudios con datos donde nadie fue rechazado (Banasik et al. 2003) y los marcos recientes de evaluación con sesgo muestral (Kozodoi et al., 2025) coinciden en algo: el sesgo afecta más a la **evaluación** del modelo (y a decisiones como el swap-set) que a su ranking. Justamente lo que este módulo mide.

---

## 5. Cuándo falla: trampas y modos de falla

**T1 · Comparar el swap-in con la cartera.**
Síntoma: «el swap-in trae 9,4%, el triple de la cartera (3,0%); el cambio empeora». Causa: confundir expansión con intercambio. Detección: mirar si $n_{01}=n_{10}$. Qué hacer: a iso-aprobación el comparador es el swap-out (§3.1); en expansión, la cartera vieja.

**T2 · Tratar el swap-in retrospectivo como si fuera el real.**
Síntoma: la 2×2 del desarrollo muestra desempeño observado en las cuatro celdas y el informe dice «el swap-in tendrá 9,4%». Causa: la política vieja del ejercicio era hipotética (aplicada a posteriori sobre aprobados). Detección: preguntar qué política estuvo **vigente** cuando se originaron los créditos de la muestra; si es la «vieja» del ejercicio, el swap-in no puede tener desempeño. Qué hacer: rotular el ejercicio como «reemplazo retrospectivo sobre cursadas» y reportar aparte el swap-in real con su método de estimación y su punto de quiebre.

**T3 · Circularidad: el modelo nuevo valida su propio swap-in.**
Síntoma: la tasa del swap-in se estima con la PD del scorecard nuevo y se presenta como evidencia de que el scorecard nuevo es mejor. Causa: el mismo modelo que eligió la celda estima su riesgo, y lo hace con sesgo a la baja (§3.6). Detección: la cohorte swap-in, cuando madura, sale sobre su PD (Austral: 9,4% vs 5,1%; notebook: 22,7% vs 17,6%). Qué hacer: reportar punto de quiebre y cotas; castigar la estimación con un múltiplo de estrés; planificar exploración.

**T4 · Comparar intervalos de carteras en vez de la diferencia.**
Síntoma: «los IC se solapan, la mejora no es significativa». Causa: ignorar que las carteras comparten $C_{11}$. Detección: EE pareado ≪ EE ingenuo. Qué hacer: IC de $\Delta$ desde las celdas de intercambio (§3.5).

**T5 · Iso-aprobación en créditos, no en pesos ni en segmentos.**
Síntoma: la mora baja pero la pérdida en pesos no baja tanto, o un canal pierde 3 pp de aprobación. Causa: el swap-in tiene montos distintos o se concentra en otros segmentos. Detección: la matriz en pesos y por segmento (§8.2). Qué hacer: reportar las cinco iso-métricas y la mezcla; si hay cupos por canal, hacer el swap-set por canal.

**T6 · Elegir el nivel de comparación mirando el resultado.**
Síntoma: el swap-set se presenta a la aprobación donde el modelo luce mejor. Causa: con un slider de aprobación siempre hay un punto favorable (comparaciones múltiples). Detección: preguntar por qué esa aprobación. Qué hacer: fijar la aprobación de la política vigente (la real) antes de mirar y mostrar la curva completa (la frontera), no un punto.

**T7 · Medir en DEV o calibrar y evaluar en la misma muestra.**
Síntoma: swap-set más favorable que en OOT; cohorte swap-in «perfecta». Causa: sobreajuste o, más a menudo, que DEV no trae el cambio de nivel. Detección: repetir en HO y OOT. En el notebook, el optimismo por sobreajuste es chico (−14% relativo en DEV, −20% en HO, −13% en OOT) y lo que cambia mucho es el nivel (swap-in 16,2% en DEV, 21,6% en OOT). Qué hacer: decidir en OOT, calibrar en una muestra y validar en otra (M15).

**T8 · Swap-in vacío en producción.**
Síntoma: la cohorte swap-in del tablero nunca se llena. Causa: los KO siguen vinculantes (KO primero, score después): la política implementada no es la que se evaluó. Detección: contar la cohorte al primer mes. Qué hacer: decidir explícitamente reemplazo vs combinación y rehacer el swap-set de la política que de verdad se implementa.

**T9 · Cohorte sin potencia leída como verde.**
Síntoma: «el swap-in está bajo control, p = 0,3». Causa: n chico. Detección: calcular la potencia **antes** de leer (§3.8). Qué hacer: fecha de lectura con potencia objetivo; mientras tanto, reportar «sin evidencia suficiente», no 🟢.

**T10 · Exploración sin propensión registrada.**
Síntoma: «aprobamos algunos bajo el corte» pero no se sabe cuáles fueron al azar y cuáles excepciones comerciales. Causa: exploración mezclada con overrides. Detección: auditar la asignación: si el ejecutivo pudo elegir, hay selección. Qué hacer: aleatorizar por hash determinista del id, registrar ε y la versión de política en el log de decisiones; las excepciones van por otro canal y no entran al estimador.

**T11 · Leer la cohorte antes de madurar.**
Síntoma: swap-in a 6 meses con 3% de malos, «mejor que lo esperado». Causa: la tasa de 90+ a 12 meses aún no se completa. Detección: curva de maduración (Serie 1 · M3). Qué hacer: comparar contra la PD **a la misma madurez** o esperar la fecha.

**T12 · Reject inference validada con sus propios supuestos.**
Síntoma: «el parceling mejora el Gini en los rechazados inferidos». Causa: se evalúa sobre etiquetas que el propio método inventó. Detección: preguntar con qué desempeño **observado** se validó. Qué hacer: solo exploración o datos externos validan la celda (Hand y Henley; Serie 1 · E1).

---

## 6. Puente con ingeniería

El swap-set deja de ser un análisis ad hoc cuando la política es una **función pura versionada** y cada decisión deja un rastro contrafactual.

**Contrato de la política.** `decidir(solicitud, config_politica) -> Decision(aprueba, razones, score, version, propension)`. Sin efectos laterales, determinista dado el config. Consecuencia: cualquier política pasada o candidata se puede re-ejecutar sobre cualquier solicitud histórica, y el swap-set retrospectivo es un `join`.

**Modo sombra (shadow scoring).** En producción, cada solicitud se evalúa con la política vigente **y** con la candidata; se registra ambas decisiones, solo la vigente se ejecuta. El log de decisiones contrafactuales construye la 2×2 sola, mes a mes, sin reprocesar. Cuando se aprueba el cambio, la cohorte swap-in queda etiquetada en el origen (`celda_swap = "in"`, `version_vieja`, `version_nueva`, `fecha_cambio`), no reconstruida después.

**Exploración auditable.** La aleatorización no usa `random()`: usa un hash determinista, reproducible y auditable,

```python
def en_exploracion(id_solicitud: str, semilla: str, eps: float) -> bool:
    h = hashlib.sha256(f"{semilla}:{id_solicitud}".encode()).hexdigest()
    return int(h[:15], 16) / 16**15 < eps
```

y la propensión $\varepsilon_i$ se guarda en el registro de la decisión. Sin propensión registrada no hay estimador insesgado (T10).

**Config declarativo** (lo que se congela y versiona con el modelo):

```yaml
politica: v2026.10-scorecard
reemplaza: v2024.03-knockouts
modo: reemplazo            # reemplazo | combinacion (KO vinculantes)
iso: aprobacion            # aprobacion | riesgo | perdida_total | volumen
muestra_evaluacion: OOT_2025H1
exploracion:
  region: "ko_rechaza & score >= cutoff"
  eps: 0.05
  semilla: "m18-2026"
  presupuesto_perdida_MM: 5.0
monitoreo_swap_in:
  test: binomial_unilateral
  umbrales: {amarillo: 0.05, rojo: 0.01}
  potencia_objetivo: 0.80
  n_requerido: 250
  fecha_lectura: 2028-07
```

**Invariantes verificables (tests de CI):**

```python
def test_swap_set(tabla, politica_vieja, politica_nueva, y):
    c = matriz_swap(politica_vieja, politica_nueva, y)
    assert c.n.sum() == len(tabla)                                   # particion
    if cfg.iso == "aprobacion":
        assert c.loc["swap-in", "n"] == c.loc["swap-out", "n"]       # iso exacto (top-k, no cuantil)
        s, K = c.loc["swap-in", "n"], politica_vieja.sum()
        assert np.isclose(br(politica_nueva) - br(politica_vieja),
                          s / K * (c.loc["swap-in", "tasa"] - c.loc["swap-out", "tasa"]))
    if cfg.modo == "combinacion":
        assert c.loc["swap-in", "n"] == 0                              # KO vinculantes => sin swap-in
    assert not tabla.loc[tabla.celda == "swap-in", "desempeno_observado"].any() \
        or cfg.politica_vieja_hipotetica                               # T2: el swap-in real no tiene Y
    assert (tabla.loc[tabla.explorado, "propension"] > 0).all()        # T10
```

**Qué se versiona.** El config de ambas políticas; la muestra de evaluación (hash); la tabla 2×2 con IC, punto de quiebre y cotas; el método de estimación del swap-in y su múltiplo de estrés; la potencia y la fecha de lectura de la cohorte. La fila del tablero se genera desde este registro, no a mano.

**Qué NO se hace en el pipeline.** Recalcular el $k$ de iso-aprobación en producción (se fija el cutoff equivalente y se congela); mezclar explorados con excepciones; reentrenar con explorados sin ponderar por $1/\varepsilon$.

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy (notebook) | Librería | Diferencias / convención | En producción |
|---|---|---|---|---|
| Matriz 2×2 | Máscaras booleanas + sumas | `pandas.groupby` / `crosstab` | Ninguna; el assert compara conteos | Pandas o SQL, con el test de partición |
| Top-k a iso-aprobación | `argsort(-score, kind="stable")[:k]` | `np.quantile` como cutoff | Un cuantil **no** garantiza $k$ exacto con empates (el binning genera muchos) | Top-k estable, reportar cutoff equivalente |
| pmf binomial | log-combinatorios por suma acumulada, `exp` | `scipy.stats.binom.pmf` | Idénticas a 1e−12; la versión log no desborda con n grande | scipy |
| Clopper-Pearson | Inversión de colas por bisección | `statsmodels.proportion_confint(method="beta")`, `scipy.stats.binomtest(...).proportion_ci("exact")`, `BETA.INV` en planilla | Coinciden a 1e−8 | statsmodels |
| Wilson / Newcombe | Fórmulas cerradas | `statsmodels.confint_proportions_2indep(method="newcomb")` | Coinciden; statsmodels también ofrece `agresti-caffo`, `score` | statsmodels |
| Fisher exacto | Hipergeométrica en log-espacio, suma de tablas con pmf ≤ observada (tolerancia 1e−7) | `scipy.stats.fisher_exact` | Coinciden; `boschloo_exact` es más potente (p 0,0023 vs 0,0037) | scipy; Boschloo si n chico |
| Test binomial | Suma de pmf ≤ pmf observada (*minlike*) | `scipy.stats.binomtest` | Mismo p; la planilla muestra además unilateral (0,031) y por duplicación (0,062) | Declarar la definición antes de mirar |
| Potencia exacta | Región de rechazo vectorizada (sort + searchsorted) | Bucle con `binomtest` + `binom.pmf`; `statsmodels` tiene potencias normales (`NormalIndPower`) | Exacta vs normal difieren en n chicos y en dientes de sierra | Exacta para decidir n; normal para órdenes de magnitud |
| Logística | IRLS (Newton) | `statsmodels.GLM(Binomial)` | Coinciden a 1e−6 | statsmodels / sklearn sin penalización |
| δ de calibración | Newton sobre la media | `scipy.optimize.brentq` | Coinciden a 1e−9 | brentq (acotado, robusto) |
| Exploración | Monte Carlo propio | — (no hay librería estándar) | — | Simulación propia, versionada con semilla |

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral, con todo lo que el curso no dijo

| Magnitud | Valor | Fuente |
|---|---|---|
| Celdas (malos/n) | 51/1.680 · 30/128 · 12/128 · 26/68 | curso |
| Mora vieja → nueva | 4,48% → 3,48% (Δ = −1,00 pp, −22%) | identidad §3.1 |
| IC 95% swap-in / swap-out | [4,9%; 15,8%] / [16,4%; 31,7%] | Clopper-Pearson |
| Fisher (swap-in vs swap-out) | p = 0,0037 (Boschloo 0,0023) | §3.4 |
| IC 95% de $b_{01}-b_{10}$ | [−23,0; −5,0] pp | Newcombe |
| IC 95% de Δ cartera | [−1,63; −0,35] pp (EE 0,32 pp vs 0,65 pp «independiente») | §3.5 |
| IC de las carteras | [3,57; 5,54] vs [2,69; 4,44]: se solapan | irrelevante (T4) |
| Punto de quiebre del swap-in | 23,4% = 2,5 × 9,4% | §3.6 |
| Cotas de Manski de la cartera nueva | [2,82%; 9,90%] vs 4,48% vieja | §3.6 |
| Cohorte swap-in vs PD 5,1% | p minlike 0,041 · unilateral 0,031 · duplicación 0,062 | §3.8 |
| Potencia con n = 128 (real 9,4%) | 55% | §3.8 |
| n para potencia 80% | 250 (normal: 248) | §3.8 |

Lectura para el comité: en el ejercicio retrospectivo, la mejora es real y estadísticamente sólida, y el margen de seguridad es amplio (2,5×). Pero es el reemplazo de una política **hipotética**. Si Austral de verdad hubiera operado con esos knock-outs, el 9,4% sería una estimación del modelo, y la propia cohorte del tablero sugiere que el modelo subestima en esa región por un factor de 1,8 (9,4/5,1). Un 1,8× de sesgo cabe dentro del 2,5× de margen: la decisión seguiría en pie, con menos holgura de la que sugiere la tabla.

### 8.2 Banco Sintético (notebook)

Cartera de 24.000 solicitudes del generador común, scorecard de 7 variables ajustado en DEV (Gini OOT 0,549; la PD verdadera logra 0,593, el techo). Política vieja «base»: knock-outs de mora hace ≤ 2 meses, ≥ 6 consultas en 6 meses o renta no acreditada; rechazan 8,0% del flujo histórico (26,8% de malos entre rechazados vs 11,2% entre los que pasan). La tasa de malos del generador es alta (12,4% global, 15,4% en OOT por el deterioro plantado): los niveles no son los de Austral, la mecánica sí.

**Swap-set retrospectivo (OOT, iso-aprobación 92,5%, cutoff equivalente 510).** Ambas aprueban 4.181 · 11,4%; swap-out 250 · 53,6%; swap-in 250 · 21,6%; ambas rechazan 111 · 64,9%. Mora 13,81% → 12,01% (−1,81 pp). Con la política vieja bajada a 80% por el tope de carga: 12,73% → 9,05%; el tope de carga es mala política y el scorecard la supera con más holgura.

**Las cinco iso-métricas (OOT, política vieja a 92,5%).** Iso-aprobación: 12,01% de mora. Iso-riesgo: 97,2% de aprobación (+4,7 pp). Iso-pérdida total: 95,9%. Iso-tasa de pérdida: 97,2%. Iso-volumen: 92,2% (menos créditos para colocar los mismos pesos, porque el score favorece montos algo mayores). A 80% de aprobación, iso-riesgo concede +14,5 pp, coherente con §3.2(b).

**En pesos.** Swap-in 645 MM CLP colocados (2,58 MM CLP medio) vs swap-out 616 MM CLP (2,46 MM CLP): el intercambio es 1:1 en créditos y no en pesos. Pérdida realizada (LGD 45%): swap-in 60,9 MM CLP, swap-out 146,6 MM CLP; cartera 681,7 → 596,0 MM CLP (−85,7 = 60,9 − 146,6, la identidad en pesos), tasa sobre monto 6,05% → 5,27%.

**Por segmento.** La aprobación global es la misma, pero sucursal sube de 91,9% a 93,9%, app baja de 92,3% a 90,6% y fuerza de venta de 91,7% a 88,8%. El swap-in de sucursal tiene 7,2% de malos contra 32,3% en fuerza de venta. Si fuerza de venta tiene metas propias, el iso global no le sirve a su dueño.

**La trampa sutil: la política vieja estuvo vigente.** Se reentrena el scorecard solo con los aprobados históricos de DEV, se calibra su nivel con un δ en los aprobados de OOT (lo único observable; δ = +0,317) y se estima el swap-in:

| Mecanismo de selección histórico | Swap-in estimado (modelo truncado) | Modelo con todos los datos | Verdad (PD media) | Observado | Sesgo total | del cual truncamiento |
|---|---|---|---|---|---|---|
| Solo knock-outs (MAR) | 17,6% | 19,0% | 23,0% | 22,7% | −5,3 pp | −1,4 pp |
| KO + juicio del ejecutivo (MNAR) | 14,9% | 18,0% | 25,8% | 25,6% | −10,9 pp | −3,1 pp |

Tres lecturas:

1. **Incluso bajo MAR** la estimación subestima un 23% relativo (17,6 vs 23,0). La mayor parte (−4,0 pp) no viene del truncamiento sino de la selección por el propio score (§3.6): el modelo con todos los datos también se equivoca justo en la celda que eligió. El truncamiento agrega −1,4 pp. La tabla de bins lo muestra: con KO de mora ≤ 2 meses, «mora hace 1 mes» (49% de malos reales en OOT) cae en el bin (−9, 3] cuyo WoE se estimó solo con clientes de 3 meses: −1,320 truncado vs −1,396 con todos. El modelo no sabe que no sabe.
2. **Bajo MNAR el sesgo se duplica** y la mejora estimada de la cartera (11,20% → 10,10%, −1,10 pp) es 3,5 veces la real (11,20% → 10,89%, −0,31 pp). La decisión sigue siendo favorable, pero por una fracción de lo que se prometió.
3. **Robustez.** Punto de quiebre MAR: 52,7% (3,0× la estimación); MNAR: 29,8% (2,0×). Cotas de Manski de la cartera nueva MAR: [10,7%; 16,6%] contra 13,8% de la vieja: sin supuestos no se sabe ni el signo.

**Exploración (MAR, 3.000 solicitudes/mes, 6 meses, 1.356 rechazados en el periodo, margen 9%, LGD 45%).** Verdad del swap-in 22,7%; el modelo dice 17,6% (sesgo 5,0 pp).

| ε | Explorados útiles | RMSE directo (E1) | RMSE modelo + δ (E2) | Costo neto, todos los rechazados | Costo neto, focalizada | P(decisión errada, E1) |
|---|---|---|---|---|---|---|
| 1% | 10 | 14,2 pp | 10,2 pp | 1,9 MM CLP | 0,4 MM CLP | 45% |
| 5% | 49 | 5,8 pp | 5,2 pp | 10,4 MM CLP | 1,4 MM CLP | 30% |
| 10% | 97 | 4,5 pp | 3,7 pp | 20,4 MM CLP | 2,8 MM CLP | 26% |
| 30% | 292 | 2,3 pp | 2,5 pp | 61,1 MM CLP | 8,9 MM CLP | 10% |

- Con menos de ~50 explorados útiles, el estimador directo es **peor** que el modelo sesgado (RMSE > 5 pp). E2 domina con poca exploración; E1 con mucha (E2 converge a su sesgo de ~2,2 pp).
- **Focalizar** reduce el costo por dato útil unas 7 veces (2,8 vs 20,4 MM CLP a ε = 10%), porque no paga explorar a los rechazados fuera del swap-in (66% de malos).
- **Decisión «abrir el swap-in»** con margen 9%: el quiebre es $b^*=9\%/45\%=20\%$. El modelo (17,6%) dice abrir; la verdad (22,7%) dice no abrir. La región mueve ≈ 5 MM CLP de valor por mes de política (163 créditos/mes × 2,58 MM CLP × |9% − 45%·22,7%|). La exploración focalizada al 10% (2,8 MM CLP) se paga con menos de un mes de política bien decidida, aunque todavía deja 26% de probabilidad de decidir mal porque la verdad está cerca del quiebre (≈ 0,6 EE).

**Cohorte swap-in.** Con los números de Austral y 25 créditos swap-in al mes, se necesitan 250 casos (10 meses) + 12 de maduración: si la política parte en octubre de 2026, la fila se lee con potencia 80% en **julio de 2028**. Antes de eso, cualquier verde es falta de evidencia.

### 8.3 Caso motos: reemplazar el knock-out «sin renta acreditada»

Contexto típico de una fintech de financiamiento de motos: una fracción relevante de solicitantes son trabajadores independientes (repartidores de plataformas, comercio informal) sin renta acreditable, y la política vieja los rechaza por knock-out. El scorecard nuevo usa comportamiento de pago, uso de líneas y consultas, y aprobaría a una parte de ellos. Supuestos de ilustración (declarados, no de mercado): 3.000 solicitudes/mes; el KO «sin renta» rechaza 4%; el scorecard aprobaría 70% de esos 120; EAD 2,5 MM CLP; LGD 45% (convención del curso; con prenda sobre la moto la LGD puede ser menor, pero la depreciación y el costo de recuperación de motos usadas son altos: se modela aparte); margen neto de vida 9% del EAD.

1. **Swap-set retrospectivo:** imposible para este segmento. Nunca se les otorgó crédito; no hay cursadas que re-filtrar.
2. **Estimación por modelo:** el modelo nunca vio a un «sin renta» aprobado; si la renta no está en el score, la estimación será la de clientes con igual comportamiento y renta acreditada. Bajo MNAR (el independiente sin renta tiene riesgo no capturado por su comportamiento bancario, por ejemplo volatilidad de ingresos), el notebook sugiere sesgos de 5–11 pp.
3. **Punto de quiebre:** si el swap-in total está a iso-aprobación contra un swap-out del 40%, el segmento puede equivocarse mucho sin dañar la cartera; pero la decisión de **precio** del segmento depende de su tasa absoluta contra el quiebre de margen $b^*=20\%$, y ahí el sesgo importa.
4. **Exploración focalizada:** aprobar al azar el 20% de los «sin renta» con score sobre el cutoff durante 6 meses da ≈ 100 explorados útiles (84 al mes × 20% × 6), EE ≈ 4 pp con una tasa del 20%. Si el segmento tiene 20% de malos, el costo neto esperado es ≈ $100\times2{,}5\times(0{,}45\cdot0{,}20-0{,}09)=0$ MM CLP: en el quiebre, explorar no cuesta nada en esperanza; si tiene 30%, cuesta ≈ 11 MM CLP. El valor: ≈ 84 créditos/mes × 2,5 MM CLP × |9% − 45%·θ| durante la vida de la política.
5. **Cohorte:** etiquetada en origen, test unilateral contra su PD, fecha de lectura escrita en el config; indicador temprano de 30+ a 3 meses (en motos la mora temprana es muy informativa por el riesgo de abandono del bien).

---

## 9. Preguntas de comité

**1. «El swap-in tiene 9,4% de malos, el triple de nuestra cartera. ¿Por qué lo aprobaríamos?»**
Porque a igual aprobación no se agrega el swap-in a la cartera: se reemplaza al swap-out, que tiene 23,4%. La mora cambia en $\tfrac{s}{K}(b_{in}-b_{out})=\tfrac{128}{1808}(9{,}4\%-23{,}4\%)=-1{,}0$ pp. La comparación con la cartera solo es la correcta si el cambio es una expansión (más aprobación).

**2. «Los intervalos de 4,48% y 3,48% se solapan. ¿No es ruido?»**
No. Las dos carteras comparten 1.680 créditos y su ruido se cancela; la diferencia depende solo de las 256 del intercambio. IC 95% de la diferencia: [−1,63; −0,35] pp. Fisher para 12/128 vs 30/128: p = 0,004.

**3. «¿Esos 128 del swap-in tienen desempeño observado? Si la política vieja los rechazaba, ¿cómo?»**
En el ejercicio del curso, sí, porque la política vieja era hipotética: se aplicó a créditos que el banco cursó. Si la política vieja hubiera estado vigente, no tendrían desempeño y el 9,4% sería una estimación del modelo, sesgada a la baja (§3.6). Por eso el informe separa «reemplazo retrospectivo» de «swap-in real».

**4. «¿Cuánto se puede equivocar la estimación del swap-in antes de que el cambio deje de convenir?»**
El punto de quiebre a iso-aprobación es la tasa del swap-out: 23,4%, es decir 2,5 veces la estimación. La propia cohorte sugiere un sesgo de ~1,8×. Cabe, con menos holgura de la que muestra la tabla.

**5. «La cohorte swap-in dio p = 0,04. ¿Es grave?»**
Es amarillo con un test de poca potencia: con n = 128, un swap-in con casi el doble de mora que su PD (9,4% vs 5,1%) se detecta solo el 55% de las veces. La acción es la del tablero (vigilancia reforzada, recalibración programada si persiste), más una fecha de lectura con potencia 80% (n ≈ 250). También hay que declarar si el test es uni o bilateral: con los mismos datos da 0,031, 0,041 o 0,062.

**6. «¿Por qué no aprobamos al azar algunos rechazados para saber de verdad?»**
Es el único método sin supuestos, y se paga en pérdida. Focalizado en la región swap-in, cuesta unas 7 veces menos por dato útil que explorar todos los rechazados. Su valor depende de qué tan cerca esté la tasa del quiebre de la decisión. Requiere aleatorización auditable, propensión registrada y revisión de cumplimiento.

**7. «¿Los knock-outs se eliminan?»**
El ejercicio midió reemplazo. Los que ambas políticas rechazan tienen 38,2%: las reglas no eran absurdas, eran insuficientes. Si los KO siguen vinculantes, el swap-in desaparece y la ganancia es otra: hay que rehacer el swap-set de la política que de verdad se implementa (T8).

**8. «¿El −22% se sostiene si medimos en pesos o por canal?»**
Hay que mostrarlo: iso-aprobación en créditos no es iso-volumen ni iso-mezcla. En el sintético, el intercambio es 1:1 en créditos pero 645 vs 616 MM CLP en pesos, y fuerza de venta pierde casi 3 pp de aprobación (91,7% → 88,8%).

---

## 10. Ejercicios

**E1 (cálculo).** Una política nueva aprueba 2.000 de 2.200 solicitudes; la vieja aprobaba 1.900. Ambas aprueban 1.850 con 2,8% de malos; swap-out 50 con 20%; swap-in 150 con 7%. (a) Mora vieja y nueva. (b) Verifica la identidad general. (c) ¿Con qué tasa del swap-in la mora nueva igualaría la vieja?

<details><summary>Solución</summary>

(a) $m_{11}=51{,}8$, $m_{10}=10$, $m_{01}=10{,}5$. $\text{BR}_v=61{,}8/1900=3{,}253\%$; $\text{BR}_n=62{,}3/2000=3{,}115\%$.
(b) $[150(0{,}07-0{,}03253)-50(0{,}20-0{,}03253)]/2000=[5{,}62-8{,}37]/2000=-0{,}138$ pp $=3{,}115\%-3{,}253\%$. ✓
(c) $b^*_{01}=(2000\cdot0{,}03253-51{,}8)/150=(65{,}05-51{,}8)/150=8{,}8\%$. El swap-in solo puede estar 1,26× sobre su estimación: decisión frágil, aunque sea «favorable».
</details>

**E2 (derivación).** Demuestra que a iso-aprobación la varianza de $\Delta$ no depende de $n_{11}$ ni de $\pi_{11}$, y que fuera de iso depende de ellos solo a través de $(1/K_n-1/K_v)^2$.

<details><summary>Solución</summary>

$\Delta=(A+m_{01})/K_n-(A+m_{10})/K_v=A(1/K_n-1/K_v)+m_{01}/K_n-m_{10}/K_v$, con $A,m_{01},m_{10}$ independientes. $\operatorname{Var}(\Delta)=\operatorname{Var}(A)(1/K_n-1/K_v)^2+\operatorname{Var}(m_{01})/K_n^2+\operatorname{Var}(m_{10})/K_v^2$. En iso, $K_n=K_v$ y el primer término vale cero; $\operatorname{Var}(A)=n_{11}\pi_{11}(1-\pi_{11})$ solo entra por ese término.
</details>

**E3 (IC a mano).** Para el swap-in de Austral (12/128), calcula el IC de Wilson al 95% y compáralo con Clopper-Pearson [4,9%; 15,8%]. ¿Por qué Wilson es más angosto?

<details><summary>Solución</summary>

$\hat p=0{,}09375$, $z^2=3{,}8416$. Centro $=(0{,}09375+3{,}8416/256)/(1+3{,}8416/128)=(0{,}09375+0{,}01501)/1{,}03001=0{,}10559$. Semiancho $=1{,}96\sqrt{0{,}09375\cdot0{,}90625/128+3{,}8416/(4\cdot128^2)}/1{,}03001=1{,}96\sqrt{0{,}000664+0{,}0000586}/1{,}03001=0{,}05116$. IC: [5,44%; 15,67%]. Clopper-Pearson garantiza cobertura ≥ 95% para todo π (conservador por la discreción); Wilson apunta a cobertura cercana a 95% en promedio y puede quedar algo bajo en algunos π.
</details>

**E4 (potencia).** Con $p_0=5{,}1\%$, ¿qué mora real $p_1$ detecta con potencia 80% una cohorte de n = 128 (α = 5% bilateral, aproximación normal)? Interpreta.

<details><summary>Solución</summary>

Resolver $\sqrt{128}(p_1-p_0)=1{,}96\sqrt{p_0q_0}+0{,}842\sqrt{p_1q_1}$. Con $\sqrt{p_0q_0}=0{,}220$: $11{,}31(p_1-0{,}051)=0{,}4313+0{,}842\sqrt{p_1(1-p_1)}$. Probando $p_1=0{,}11$: izq. 0,667; der. 0,4313+0,842·0,313=0,695. $p_1=0{,}115$: izq. 0,724; der. 0,4313+0,842·0,319=0,700. Solución ≈ 11,3%. La fila del tablero con n = 128 solo detecta con confianza un swap-in con más del doble de su PD. Una subestimación de 1,5× pasa inadvertida la mayoría de las veces.
</details>

**E5 (diseño).** Tu política vigente son knock-outs más juicio del analista para montos altos. Diseña el swap-set de un scorecard que reemplazaría ambos: qué celdas son observables, qué estimas y cómo, qué reportas como robustez y qué exploración propones.

<details><summary>Solución</summary>

Observables: ambas aprueban y swap-out (fueron aprobados). No observables: swap-in y ambas rechazan. El swap-in contiene rechazos por KO (MAR, sin solapamiento: extrapolación) y por juicio (MNAR: selección adversa). Estimación: modelo truncado calibrado en aprobados recientes, más un múltiplo de estrés separado por tipo de rechazo (mayor para los rechazados por juicio). Robustez: punto de quiebre $b^*_{01}$ y cotas de Manski. Si el múltiplo de quiebre es menor que ~2, no firmar sin evidencia. Exploración focalizada en los rechazados por juicio con score sobre el cutoff (la región con más incertidumbre), ε por hash, propensión registrada, presupuesto de pérdida y fecha de lectura. Rampa de cutoff en montos altos mientras madura la evidencia.
</details>

**E6 (código).** Escribe en numpy una función que, dados `score`, `aprueba_vieja`, `y` y `monto`, devuelva el $k$ iso-riesgo e iso-pérdida total, y un test que falle si la frontera no es monótona en expectativa (usa la PD en vez de `y`).

<details><summary>Solución</summary>

```python
def k_iso(score, ap_v, y, monto, lgd=0.45):
    o = np.argsort(-score, kind="stable")
    ys, ms = y[o], monto[o]
    k = np.arange(1, len(y) + 1)
    br = np.cumsum(ys) / k
    perd = np.cumsum(lgd * ys * ms)
    br_v = y[ap_v].mean(); perd_v = (lgd * y[ap_v] * monto[ap_v]).sum()
    ultimo = lambda c: int(np.flatnonzero(c)[-1] + 1)
    return ultimo(br <= br_v), ultimo(perd <= perd_v)

def test_frontera_monotona(score, pd_cal):
    o = np.argsort(-score, kind="stable")
    br_esp = np.cumsum(pd_cal[o]) / np.arange(1, len(o) + 1)
    assert np.all(np.diff(br_esp) >= -1e-12)   # si score y PD ordenan igual
```
El test pasa si la PD es monótona en el score (lo es por construcción con scaling PDO); falla si se mezclan scores y PD de versiones distintas, que es justamente lo que se quiere detectar.
</details>

**E7 (exploración).** Rechazados por mes: 240; fracción en la región swap-in: 70%; θ supuesto 23%; resto de rechazados 55%; EAD 2,5 MM CLP; LGD 45%; margen 9%. Compara, para 6 meses y un EE objetivo de 4 pp, el ε y el costo neto de explorar todos los rechazados frente a explorar solo la región.

<details><summary>Solución</summary>

EE 4 pp exige $n_R=0{,}23\cdot0{,}77/0{,}04^2=111$ explorados útiles. $N_R=240\cdot0{,}7\cdot6=1008$ ⇒ ε = 11%. Todos: explorados $=0{,}11\cdot1440=158$, de ellos 47 fuera de la región. Costo $=111\cdot2{,}5(0{,}45\cdot0{,}23-0{,}09)+47\cdot2{,}5(0{,}45\cdot0{,}55-0{,}09)=111\cdot0{,}034+47\cdot0{,}394=3{,}8+18{,}5=22{,}3$ MM CLP. Focalizada: solo los 111 ⇒ 3,8 MM CLP. La diferencia (18,5 MM CLP) es pérdida pagada por aprender algo que no se necesitaba. La hoja `Exploracion` de la planilla hace esta cuenta.
</details>

**E8 (crítica).** Un proveedor presenta: «Nuestro modelo con reject inference aumenta el Gini sobre la población completa de 0,55 a 0,61». ¿Qué preguntas haces?

<details><summary>Solución</summary>

¿Con qué desempeño se midió el Gini en los rechazados? Si con etiquetas inferidas por el mismo método, la mejora es circular (T12). ¿Hay una muestra de exploración o de otra institución con desempeño observado de rechazados? ¿Qué supuesto de selección usa (MAR, MNAR, parceling con qué factor) y cómo cambia el resultado si el factor se estresa? ¿Qué pasa con la estimación del swap-in de la política propuesta, que es lo que decide el comité, y cuál es su punto de quiebre? Sin evidencia observada, el número correcto es «no validado», no 0,61.
</details>

**E9 (numpy).** Implementa el p-valor bilateral *minlike* en numpy y explica por qué necesita una tolerancia relativa (1e−7) al comparar probabilidades.

<details><summary>Solución</summary>

`pmf = binom_pmf_np(n, p0); p = pmf[pmf <= pmf[k] * (1 + 1e-7)].sum()`. Resultados con probabilidad teóricamente igual a la observada pueden diferir en el último bit por redondeo (sobre todo en el lado opuesto de la distribución) y quedar fuera de la suma. La tolerancia los incluye. `scipy.stats.binomtest` usa la misma idea, y el notebook comprueba que ambos dan igual.
</details>

---

## 11. Referencias

- **Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2.ª ed.). Wiley.** Swap-set, estrategia de cutoff y reject inference desde la práctica. Es la fuente de la receta del curso.
- **Anderson, R. (2007). *The Credit Scoring Toolkit*. Oxford University Press.** Capítulos de estrategia y de reject inference con la visión de implementación (swap sets, champion/challenger).
- **Thomas, L. C., Crook, J. y Edelman, D. (2017). *Credit Scoring and Its Applications* (2.ª ed.). SIAM.** Tratamiento formal del sesgo muestral y de las decisiones de aceptación.
- **Hand, D. J. y Henley, W. E. (1993). «Can reject inference ever work?» *IMA Journal of Mathematics Applied in Business and Industry*, 5(1), 45–55 (verificar volumen/año).** Por qué la reject inference sin supuestos externos no puede funcionar: el argumento de fondo de §3.6.
- **Banasik, J., Crook, J. y Thomas, L. (2003). «Sample selection bias in credit scoring models». *Journal of the Operational Research Society*, 54(8), 822–832.** Datos de un banco que aprobó a todos: miden cuánto importa de verdad el sesgo muestral.
- **Crook, J. y Banasik, J. (2004). «Does reject inference really improve the performance of application scoring models?» *Journal of Banking & Finance*, 28(4), 857–874.** Respuesta empírica escéptica. Complemento de Hand y Henley.
- **Kozodoi, N., Lessmann, S. et al. (2025). «Fighting sampling bias: A framework for training and evaluating credit scoring models». *European Journal of Operational Research* (preprint arXiv:2407.13009).** Marco reciente que separa el sesgo en entrenamiento y en evaluación. Lo más útil para el problema del swap-in.
- **Boyes, W. J., Hoffman, D. L. y Low, S. A. (1989). «An econometric analysis of the bank credit scoring problem». *Journal of Econometrics*, 40(1), 3–14.** Probit bivariado con selección aplicado a crédito.
- **Heckman, J. J. (1979). «Sample selection bias as a specification error». *Econometrica*, 47(1), 153–161.** El modelo de selección y la inversa de Mills.
- **Manski, C. F. (1989). «Anatomy of the selection problem». *Journal of Human Resources*, 24(3), 343–360.** Cotas sin supuestos: la base de §3.6(i).
- **Smith, J. E. y Winkler, R. L. (2006). «The optimizer's curse: Skepticism and postdecision surprise in decision analysis». *Management Science*, 52(3), 311–322.** Por qué lo que elegiste como mejor rinde peor de lo estimado.
- **Clopper, C. J. y Pearson, E. S. (1934). «The use of confidence or fiducial limits illustrated in the case of the binomial». *Biometrika*, 26(4), 404–413.** El intervalo exacto.
- **Brown, L. D., Cai, T. T. y DasGupta, A. (2001). «Interval estimation for a binomial proportion». *Statistical Science*, 16(2), 101–133.** Por qué Wald falla y cuándo preferir Wilson/Jeffreys.
- **Newcombe, R. G. (1998). «Interval estimation for the difference between independent proportions: comparison of eleven methods». *Statistics in Medicine*, 17(8), 873–890.** El IC híbrido de §3.4.
- **Fagerland, M. W., Lydersen, S. y Laake, P. (2015). «Recommended confidence intervals for two independent binomial proportions». *Statistical Methods in Medical Research*, 24(2), 224–254.** Guía práctica para elegir el IC de una diferencia.
- **Wald, A. (1945). «Sequential tests of statistical hypotheses». *Annals of Mathematical Statistics*, 16(2), 117–186.** Monitoreo continuo de la cohorte sin inflar el error.
- **Kohavi, R., Tang, D. y Xu, Y. (2020). *Trustworthy Online Controlled Experiments*. Cambridge University Press.** Ingeniería de experimentos aleatorizados (asignación por hash, sesgos de implementación). Traducible directo a champion/challenger.
- **Board of Governors of the Federal Reserve System (2026). SR 26-2, «Revised Guidance on Model Risk Management» (17-abr-2026), que reemplaza SR 11-7 (2011); OCC Bulletin 2026-13.** Marco supervisor de EE.UU. para validación y análisis de resultados; verificar texto.
- **Ley 21.680 (Chile, 2024) y CMF, NCG 540 (2025): Registro de Deuda Consolidada.** Relevante para usar desempeño de rechazados en otras instituciones. Verificar fecha de operación y reglas de consulta.
- **Serie 1 · E1** (reject inference), **E3** (bootstrap e intervalos para carteras chicas), **M3** (curvas de maduración); **Serie 2 · M15** (calibración), **M17** (estrategia y cutoff), **M19** (backtesting).

# M17 · Estrategia: cutoff, pérdida esperada y rentabilidad

> **Ficha.** Profundiza: clase 4 v1 (parte 3, láminas 24–28 y reserva 36–37) y v21 (láminas 37–42 y 50); clase 5 (láminas 7–8). · Prerrequisitos: Serie 1 · M1 (economía del error de crédito), Serie 1 · E6 (regulación), Serie 2 · M13 (scaling), M15 (calibración), M16 (master scale). Prepara M18 (swap-set) y M20 (monitoreo). · Archivos: `M17_estrategia_cutoff.md` (este documento), `M17_estrategia_cutoff.py` (notebook Marimo), `M17_tabla_estrategia_motos.xlsx` (**planilla principal de la serie**: tabla de estrategia del caso motos con fórmulas vivas). · Tiempo estimado: 3,5 h de lectura + 2,5 h de notebook, planilla y ejercicios.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**El cutoff es una decisión comercial informada por riesgo.** La frase de la clase 4 resume la postura: «el modelo pone la curva; el negocio elige el punto». El cutoff lo firma un comité (riesgo + comercial + finanzas) y el modelador trae la evidencia. Esa evidencia es la **tabla de estrategia**, construida sobre **OOT** («la muestra con desempeño más parecida a la cartera que viene»). La nota de método del curso: elegir el cutoff mirando DEV está «inflado por construcción».

**La pérdida esperada.** $\text{EL}=\text{PD}\times\text{LGD}\times\text{EAD}$, «proxy didáctico», con dos convenciones declaradas: LGD = 45% fija y EAD = monto solicitado. Ejemplo de lámina: 1 MM\$ con PD 2% y LGD 45% ⇒ 9 mil pesos. «Cambiarlas mueve los pesos, no la forma de la decisión.» La PD es la calibrada a la tendencia central (TC 5,42%, δ = 0,1115; ver M15).

**La tabla de Banco Austral (OOT, LGD 45%, EAD = monto):**

| Cutoff | Aprobación | Mora aprobados | PD cal. media | Monto aprobado | EL (MM\$) | EL/monto |
|---|---|---|---|---|---|---|
| 500 | 97,8% | 4,90% | 4,53% | 5.966 MM\$ | 122,0 | 2,05% |
| 520 | 94,8% | 4,21% | 3,73% | 5.776 MM\$ | 95,8 | 1,66% |
| 540 | 86,9% | 3,16% | 2,44% | 5.315 MM\$ | 58,3 | 1,10% |
| 560 | 78,3% | 2,10% | 1,59% | 4.797 MM\$ | 34,4 | 0,72% |
| 580 | 67,3% | 1,78% | 0,96% | 4.145 MM\$ | 18,6 | 0,45% |
| 600 | 55,4% | 1,44% | 0,56% | 3.375 MM\$ | 8,8 | 0,26% |
| 620 | 43,7% | 0,91% | 0,34% | 2.620 MM\$ | 4,0 | 0,15% |
| 640 | 31,5% | 0,63% | 0,19% | 1.881 MM\$ | 1,6 | 0,09% |

**La recomendación.** Con un apetito de mora ≤ 2,5% y aprobación ≥ 75% (supuestos declarados), 540 no pasa (3,16%), 580 cede volumen de más (67,3%) y **560** es «el corte más generoso que lo respeta». La **frontera de estrategia** (un punto por cutoff) deja a la política de referencia, que solo aplica knock-outs (mora interna vigente o ≥ 30 días en el sistema; aprueba 90,2% de las cursadas con 4,48% de mora), **sobre** la curva. A igual aprobación, el scorecard logra 3,48%: −1,00 pp, el valor del modelo en puntos de mora. Si comercial no cede volumen, el corte 533 mantiene el 90,2% y baja la mora ~22% (swap-set, M18). En la clase 5, calibrar PIT en lugar de TC mueve la aprobación en 560 de 78,3% a 77,2%: 1,9 puntos de score para todos (M15 §3.9).

**Reserva (clase 4, lámina 36; v21, lámina 50).** Cuando comercial quiere aprobar más sin subir la mora: cutoff por segmento («exige gobernarlos todos»), pricing por riesgo («la mora extra la paga el margen; requiere PD calibrada»), cupos y plazos diferenciados («aprobar la banda D con monto menor acota la EL»). Y lo que no se hace: mover el cutoff «por esta semana» sin acta. Es un **overlay silencioso**, y «en auditoría no existe la palabra temporal».

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **La tabla mide riesgo, no plata.** Tiene EL pero no margen, costo de fondos, costo operativo ni de adquisición. Sin ellos no existe «utilidad» y no se puede preguntar cuál es el cutoff **óptimo**. Solo cabe preguntar cuál cumple un apetito.
2. **El apetito no se derivó.** El 2,5% es un supuesto. Tampoco se calculó lo que cuesta respetarlo (su **precio sombra**) ni se distinguió una restricción sobre el **promedio** de la cartera de una condición sobre el **cliente marginal**.
3. **El cutoff de breakeven** $s_{BE}=\text{offset}+\text{factor}\ln(l/g)$ y su relación con el PDO no aparecen.
4. **EL sin horizonte ni descuento.** PD a 12 meses por monto total: no distingue un crédito de 12 meses de uno de 48. «EAD = monto» ignora la amortización. La LGD fija ignora que, con garantía, la pérdida depende del **mes** en que se cae.
5. **La EL usa la PD de la TC y la tabla muestra mora observada**: 560 → PD 1,59% vs mora 2,10%. La brecha se mostró y no se discutió como **sesgo de la EL** en un periodo de deterioro.
6. **Frontera y dominancia.** Se dibujó la frontera aprobación–mora. No se mostró que en el plano utilidad–riesgo los cortes más laxos que el breakeven están **dominados**.
7. **Cutoff por segmento, pricing y cupos** quedaron como viñetas, sin condición de optimalidad, sin fórmula de tasa y sin el **tope legal** chileno (tasa máxima convencional).
8. **Sensibilidad.** No se cuantificó cuánto se mueve el corte si cambian la LGD, la TC o el margen.
9. **Productos con garantía** (motos, autos) no se trataron. Es el caso del lector.

---

## 2. Intuición

**El cutoff es una decisión marginal; el apetito es una restricción promedio.** Al bajar el corte en un punto entra un grupo de clientes con una PD específica: la PD **en el corte**, no la media de los aprobados. Conviene que entren si lo que dejan los buenos paga lo que cuestan los malos *de ese grupo*. El apetito («mora de la cartera aprobada ≤ 2,5%») mira otra cosa: el promedio de todos los aprobados. Como la PD media de aprobados siempre es menor que la PD del cliente marginal, un apetito de 2,5% de mora media puede convivir con clientes marginales de 5–7% de PD. Los dos criterios pueden coincidir, pero no tienen por qué. Cuando difieren, la utilidad que se deja sobre la mesa por respetar el apetito es su precio.

**Odds contra odds.** Un cliente con PD $p$ tiene odds $(1-p)/p$: buenos por cada malo. Si cada bueno deja $g$ y cada malo quita $l$, conviene aprobarlo cuando hay más de $l/g$ buenos por malo. El scorecard ya está escrito en log-odds (M13), así que el umbral se traduce directo a puntos: **cada vez que $l/g$ se duplica, el corte sube un PDO**. Con PDO 20, pasar de LGD 45% a 60% con costos de 12% y margen neto de 12% mueve el corte $28{,}85\cdot\ln(0{,}72/0{,}57)=6{,}7$ puntos.

**$g$ es una diferencia chica entre números grandes.** $g$ = intereses − costo de fondos − operación − adquisición. En consumo, 24% − 6% − 6% = 12%. Un cambio de 3 pp en la tasa mueve $g$ un 25%, mientras que un cambio de 10 pp en la LGD mueve $l$ ≈ 18%. Por eso el corte suele ser **más sensible a la economía del margen que a la LGD**, al revés de lo que sugiere la atención que recibe cada supuesto en un comité.

**Con garantía, la pérdida depende del cuándo.** En un crédito de moto el saldo baja con cada cuota y la moto pierde valor con el tiempo. Si el cliente cae en el mes 4, se debe casi todo y la moto ya perdió el castigo de salida de tienda. Si cae en el mes 30, el saldo es chico y la moto (si aparece) lo cubre. La LGD es entonces una **curva** en el mes de default, y los defaults se concentran en los primeros meses: el default más frecuente es también el más caro. La constante del curso (45%) puede acertar el promedio de una cartera y equivocarse en cada segmento que tenga otra distribución de tiempos.

**La PD del scorecard no es la PD del crédito.** El target es 90+ en 12 meses. Un crédito a 36 meses puede caer en el mes 20. Para decidir un crédito de 36 meses se necesita la PD de **vida**, y hay que convertirla con un supuesto explícito sobre la forma de la curva de default.

**La TMC es un cutoff de facto.** Si el riesgo de una banda exige una tasa sobre el máximo legal, no hay precio que la haga rentable. Quedan tres caminos: rechazarla, cambiar la estructura (más pie, menos plazo) o bajar su pérdida (más recupero). El pricing por riesgo tiene techo, y en Chile ese techo lo fija la ley.

---

## 3. Formalización

Notación: $p$ = PD (a 12 meses salvo que se diga), $m$ = monto, $r$ = tasa anual del modelo de un período, $c_f$ = costo de fondos, $c_{op}$ y $c_{adq}$ = costos operativo y de adquisición (en % del monto en el modelo simple), $k=c_f+c_{op}+c_{adq}$. Escala del curso: $s=\text{offset}+\text{factor}\cdot\ln\frac{1-p}{p}$, factor = 20/ln 2 = 28,8539, offset = 487,1229 (M13).

### 3.1 Pérdida esperada: qué horizonte, qué EAD

La definición general es la esperanza de la pérdida descontada:

$$\text{EL}=\mathbb{E}\Big[\sum_{d}\mathbb{1}\{\tau=d\}\,\frac{\text{LGD}(d)\cdot\text{EAD}(d)}{(1+i)^{d}}\Big]=\sum_{d}P(\tau=d)\,\frac{\text{LGD}(d)\,\text{EAD}(d)}{(1+i)^d},$$

con $\tau$ el mes de default y $i$ la tasa de descuento mensual. La fórmula del curso $\text{PD}\times\text{LGD}\times\text{EAD}$ es el caso particular de **un período**, sin descuento, con LGD y EAD constantes. Es exacta solo si $\text{LGD}(d)\,\text{EAD}(d)$ no depende de $d$. Para un crédito que amortiza no lo es.

**EAD amortizante.** Crédito de monto $M_0$ a $N$ cuotas fijas con tasa mensual $r_m$: la cuota es $C=M_0\,r_m/(1-(1+r_m)^{-N})$ y el saldo tras la cuota $k$ es

$$B_k=M_0\,\frac{(1+r_m)^N-(1+r_m)^k}{(1+r_m)^N-1}.$$

*Derivación.* La recursión $B_k=B_{k-1}(1+r_m)-C$ con $B_0=M_0$ da $B_k=M_0(1+r_m)^k-C\frac{(1+r_m)^k-1}{r_m}$. Reemplazando $C$:

$$B_k=M_0\Big[(1+r_m)^k-\frac{(1+r_m)^k-1}{1-(1+r_m)^{-N}}\Big]=M_0\,\frac{(1+r_m)^k\big(1-(1+r_m)^{-N}\big)-(1+r_m)^k+1}{1-(1+r_m)^{-N}}.$$

El numerador es $1-(1+r_m)^{k-N}$. Multiplicando numerador y denominador por $(1+r_m)^N$ se obtiene la fórmula, con $B_N=0$. El notebook verifica fórmula cerrada = recursión.

**Convención de default.** Con default = 90+ días de mora, si se alcanza en el mes $d$ la última cuota pagada fue la $d-4$: la cuota $d-3$ lleva 90 días impaga. Entonces $\text{EAD}(d)=B_{d-4}$ (capital insoluto), con $4\le d\le N+3$. Incluir intereses devengados sube la EAD y **baja** la LGD en porcentaje sin cambiar la pérdida en pesos. La convención del denominador se declara, porque cambia el número que el comité ve.

### 3.2 LGD con garantía

Sea $V(t)=P\,(1-h_0)\,(1-\text{dep})^{t/12}$ el valor de mercado de la moto a $t$ meses del otorgamiento (precio $P$, castigo inicial $h_0$, depreciación geométrica anual). Si hay default en $d$ y la moto se recupera (probabilidad $\pi$), se vende en $d+T$ a un valor neto de costos $V(d+T)(1-c)$. El acreedor cobra como máximo su acreencia, $A(d)=\text{EAD}(d)(1+r_m)^{3+T}$ (capital más el interés pactado hasta la venta); el excedente se devuelve al deudor. Descontado a la fecha de default:

$$R(d)=\frac{\min\{V(d+T)(1-c),\,A(d)\}}{(1+i)^T}.$$

La LGD **económica** en el mes $d$ es la esperanza sobre recuperar o no la moto:

$$\boxed{\text{LGD}(d)=\pi\cdot\max\Big(0,\,1-\frac{R(d)}{\text{EAD}(d)}\Big)+(1-\pi)}$$

Tres propiedades:

1. **Piso $1-\pi$.** Si la moto no aparece se pierde toda la EAD, así que $\text{LGD}(d)\ge 1-\pi$ para todo $d$. Ni el pie ni la depreciación bajan ese piso; solo lo baja subir $\pi$ (GPS, seguro contra robo, cobranza temprana).
2. **Decrece con $d$ mientras el saldo cae más rápido que la moto.** El cociente $V(d+T)/B_{d-4}$ crece si la amortización relativa del saldo supera la depreciación mensual. Al inicio del crédito francés se amortiza poco capital (con 2,2% mensual, 54% de la primera cuota es interés), así que la curva es plana los primeros meses y cae después.
3. **Mes de cruce.** Desde el primer $d^{\ast}$ con $V(d^{\ast})(1-c)\ge B_{d^{\ast}-4}$, la LGD condicional a recuperar es ≈ 0 (salvo descuento) y la LGD queda en su piso. Con los supuestos base del caso (§8.3), $d^{\ast}=24$.

**Por qué importa la esperanza y no «LGD con π dentro».** Escribir $\text{LGD}=\max(0,1-\pi R/\text{EAD})$ no es lo mismo que la fórmula de arriba cuando el $\max$ se activa: $\mathbb{E}[\max(\cdot)]\ne\max(\mathbb{E}[\cdot])$ (Jensen). La planilla y el notebook usan la versión correcta.

**Descuento.** La tasa de descuento de la LGD económica es una elección. Aquí se usa el costo de fondos (coherente con el VPN de §3.6). Las guías europeas de LGD (EBA/GL/2017/16) fijan una tasa de descuento regulatoria que no es la del contrato (verificar el detalle vigente). En Chile, las provisiones de consumo bajo el estándar de la CMF siguen su propia metodología (Serie 1 · E6). Se declara cuál se usa.

### 3.3 El cutoff de breakeven

**Un período.** Por peso prestado, un bueno paga $r$ y le cuesta al prestamista $k$; un malo no paga intereses, devuelve $1-\text{LGD}$ del capital y cuesta igual $k$:

$$\text{utilidad si bueno}=g=r-k,\qquad \text{utilidad si malo}=-l=-(\text{LGD}+k).$$

Utilidad esperada del cliente $i$: $u_i=m_i\big[(1-p_i)\,g-p_i\,l\big]$. Conviene aprobar si $u_i>0$:

$$(1-p)g>p\,l\iff\frac{1-p}{p}>\frac{l}{g}\equiv O_{BE}\iff p<p_{BE}=\frac{g}{g+l}.$$

En la escala del score, $s>s_{BE}$ con

$$\boxed{s_{BE}=\text{offset}+\text{factor}\cdot\ln\frac{\text{LGD}+k}{r-k}}$$

y si $r\le k$ no hay breakeven: ni un cliente sin riesgo paga los costos.

**Descomposición margen − EL.** $u_i=m_i[(1-p_i)r-k]-p_i\,\text{LGD}\,m_i$. El primer término es el **margen** (intereses esperados menos costos) y el segundo la **EL** del curso. La tabla de estrategia del notebook y de la planilla reporta ambos y su diferencia.

**Relación con el PDO.** $\partial s_{BE}/\partial\ln O_{BE}=\text{factor}=\text{PDO}/\ln 2$. Duplicar $O_{BE}$ sube el corte exactamente un PDO. Para un parámetro $\theta$:

$$\frac{\partial s_{BE}}{\partial\theta}=\text{factor}\Big(\frac{\partial\ln l}{\partial\theta}-\frac{\partial\ln g}{\partial\theta}\Big).$$

Para la LGD: $\partial s_{BE}/\partial\text{LGD}=\text{factor}/l$. Para la tasa: $\partial s_{BE}/\partial r=-\text{factor}/g$. Para los costos: $\partial s_{BE}/\partial k=\text{factor}(1/l+1/g)$, que es el mayor de los tres porque los costos atacan a los dos lados. Con $g=0{,}12$ y $l=0{,}57$: 1 pp de LGD mueve el corte 0,51 pts, 1 pp de tasa 2,40 pts y 1 pp de costos 2,91 pts.

**Marginal vs total: el breakeven maximiza la utilidad.** Sea $f(s)$ la densidad de scores (en masa de monto) y $\bar u(s)=\mathbb{E}[u\mid s]$. La utilidad total de un corte $c$ es $U(c)=\int_c^\infty \bar u(s)f(s)\,ds$. Por el teorema fundamental del cálculo,

$$U'(c)=-\bar u(c)\,f(c).$$

Si la PD en uso es función decreciente del score (lo es por construcción: $p=\sigma(-(s-\text{offset})/\text{factor})$), $\bar u(s)$ cambia de signo **una sola vez**, en $s_{BE}$: negativa abajo, positiva arriba. Entonces $U'(c)>0$ para $c<s_{BE}$ y $U'(c)<0$ para $c>s_{BE}$, y $s_{BE}$ es el máximo global. En una muestra finita, $U(c)$ es escalonada y su argmax es el primer score observado sobre $s_{BE}$. El notebook lo confirma: analítico 532,08; bisección y `brentq` idénticos; `minimize_scalar` sobre la utilidad suavizada 532,09; argmax en los datos 532,13.

**Apetito y precio sombra.** Maximizar $U(c)$ sujeto a $\bar p_{ap}(c)\le\rho$ (PD o mora media de aprobados). Como $\bar p_{ap}$ es creciente al bajar el corte, la restricción equivale a $c\ge c_\rho$ y el óptimo restringido es $\max(s_{BE},c_\rho)$. El precio sombra es $U(s_{BE})-U(c_\rho)$ cuando $c_\rho>s_{BE}$, y cero si no.

**RAROC.** Si se exige que cada cliente pague además el costo del capital que consume, la condición es $u(s)\ge h\cdot K(p(s))\cdot m$, con $h$ la tasa de retorno exigida y $K$ el capital por peso. Como referencia se usa la fórmula IRB de «otras exposiciones minoristas» del Comité de Basilea:

$$K(p)=\text{LGD}\Big[\Phi\Big(\frac{\Phi^{-1}(p)+\sqrt{R(p)}\,\Phi^{-1}(0{,}999)}{\sqrt{1-R(p)}}\Big)-p\Big],\quad R(p)=0{,}03\,\frac{1-e^{-35p}}{1-e^{-35}}+0{,}16\Big(1-\frac{1-e^{-35p}}{1-e^{-35}}\Big).$$

El corte RAROC queda **sobre** el breakeven. En el notebook, con $h$ = 15%, sube de 532,1 a 535,4. Es una referencia de capital económico, no un requisito para una fintech no bancaria (ver Serie 1 · E6). Según entiendo, los bancos chilenos calculan los activos ponderados por riesgo de crédito con el método estándar de la CMF; verificar con la norma vigente si hay autorizaciones de modelos internos.

### 3.4 Descalibración: cuánto se corre el corte y cuánto cuesta

Supongamos que la PD en uso es $p(s)$ y la verdad es $p^{\ast}(s)=\sigma(\text{logit}\,p(s)+\Delta)$, con el mismo $\Delta$ para todos. El breakeven verdadero resuelve $\text{logit}\,p(s^{\ast})+\Delta=\text{logit}\,p_{BE}$. Como $\text{logit}\,p(s)=-(s-\text{offset})/\text{factor}$:

$$-\frac{s^{\ast}-\text{offset}}{\text{factor}}+\Delta=-\frac{s_{BE}-\text{offset}}{\text{factor}}\;\Longrightarrow\;\boxed{s^{\ast}=s_{BE}+\text{factor}\cdot\Delta.}$$

Con $\Delta>0$ (la PD en uso subestima, típico de una PD de ciclo en un periodo malo), el corte calculado queda demasiado laxo por $\text{factor}\cdot\Delta$ puntos.

**Costo de segundo orden.** Expandiendo $U$ en torno a $s^{\ast}$ (su máximo verdadero, con $U'(s^{\ast})=0$):

$$U(s^{\ast})-U(s_{BE})\approx-\tfrac12U''(s^{\ast})(s_{BE}-s^{\ast})^2=\tfrac12\,\bar u^{\ast\prime}(s^{\ast})\,f(s^{\ast})\,(\text{factor}\cdot\Delta)^2,$$

usando $U''(c)=-\bar u'(c)f(c)-\bar u(c)f'(c)$ y $\bar u^{\ast}(s^{\ast})=0$. La pérdida crece con el **cuadrado** del error y con la densidad de clientes en el corte. En el notebook: $\Delta$ = 0,327 ⇒ corrimiento predicho +9,4 pts (532,1 → 541,5), óptimo verdadero en los datos 539,7. Usar la PD de la TC cuesta 10,5 MM\$ (2,0% de la utilidad máxima). Sobrecorregir con Δδ = 0,80 cuesta 56,7 MM\$. Corregir la mitad del sesgo (Δδ = 0,15) deja la pérdida en 1,8 MM\$.

### 3.5 Cutoff por segmento: la condición de optimalidad

Segmentos $j=1..J$ con densidades $f_j$, utilidades marginales $\bar u_j(s)$ y cortes $c_j$.

**Sin restricción.** $\max\sum_j U_j(c_j)$ se separa: cada segmento corta en su breakeven $\bar u_j(c_j)=0$.

**Con volumen fijo** $\sum_j A_j(c_j)=A^{\ast}$, con $A_j(c)=\int_c^\infty f_j$. Lagrangiano $\mathcal{L}=\sum_jU_j(c_j)-\lambda\big(\sum_jA_j(c_j)-A^{\ast}\big)$. Con $U_j'=-\bar u_jf_j$ y $A_j'=-f_j$:

$$\frac{\partial\mathcal L}{\partial c_j}=-\bar u_j(c_j)f_j(c_j)+\lambda f_j(c_j)=0\;\Longrightarrow\;\bar u_j(c_j)=\lambda\quad\forall j.$$

La **utilidad marginal se iguala** entre segmentos. En la práctica equivale a ordenar a todos los solicitantes por utilidad esperada $u_i$ (no por score) y aprobar los $A^{\ast}$ primeros.

**Con apetito de mora** $\sum_jB_j(c_j)\le\rho\sum_jA_j(c_j)$, con $B_j'=-p_jf_j$:

$$-\bar u_j f_j+\mu\,(p_j-\rho)\,f_j=0\;\Longrightarrow\;\bar u_j(c_j)-\mu\,(p_j(c_j)-\rho)=0\quad\forall j.$$

Se iguala la utilidad marginal **ajustada** por el costo de la mora que aporta el cliente marginal.

**Corolario.** Si la PD está calibrada **por segmento** y la economía $(g,l)$ es común, $\bar u_j(s)$ depende de $s$ solo a través de $p_j(s)$. La condición se cumple con la misma **PD** de corte en todos los segmentos. Un cutoff por segmento **en score** solo se justifica por dos razones:

- (a) el mismo score significa distinta PD por segmento. Es una calibración por segmento que el modelo no hizo, y es mejor hacerla en la PD (con su gobierno) que esconderla en la política;
- (b) la economía difiere: LGD por tipo de garantía, costo de adquisición por canal, tasa por convenio.

Con PD $p_j(s)=\sigma(-(s-\text{offset})/\text{factor}+\delta_j)$ y $l_j,g_j$ propios:

$$c_j=\text{offset}+\text{factor}\Big(\ln\frac{l_j}{g_j}+\delta_j\Big).$$

**Costo del ruido.** $\hat\delta_j$ se estima con $n_j$ casos. Su error estándar es aproximadamente $1/\sqrt{n_j\,\bar p_j(1-\bar p_j)}$ (información de Fisher del intercepto). El error del corte es factor veces eso. Con $n_j$ = 150 y $\bar p$ = 12%: $28{,}85/\sqrt{150\cdot0{,}106}=7{,}2$ puntos, del orden de la ganancia que se busca. El notebook lo muestra: con toda la muestra de calibración (14.360 casos) los cortes por canal ganan +7,0 MM\$ (+1,3%) con desviación de 0,8–1,9 pts. Con el 4% (574 casos) la ganancia media es −6,8 MM\$ y en 68% de los bootstraps rinden **menos** que el corte único.

### 3.6 Vida del crédito: PD de vida, VPN y breakeven del caso motos

**Riesgos proporcionales.** Intensidad mensual de default en la banda $b$: $h_b(d)=1-e^{-\lambda_b s(d)}$, con $s(d)\ge0$ una forma común (aquí $s(d)=x\,e^{1-x}$, $x=(d-3)/(d_{pico}-3)$). Sea $S(d)=\sum_{j\le d}s(j)$. La supervivencia es $\text{Sv}_b(d)=e^{-\lambda_bS(d)}$ y

$$\text{PD}_{12,b}=1-e^{-\lambda_bS_{12}}\;\Rightarrow\;\lambda_b=\frac{-\ln(1-\text{PD}_{12,b})}{S_{12}},$$
$$\text{PD}^{vida}_b=1-e^{-\lambda_bS_N}=1-(1-\text{PD}_{12,b})^{\kappa},\qquad\kappa=\frac{S_N}{S_{12}},$$

con $S_N=S(N+3)$. Es una **identidad** del modelo. Toda la incertidumbre está en la forma $s(d)$, que en la práctica se estima con las curvas de maduración de las propias cosechas (Serie 1 · M3). Con $d_{pico}=7$ y $N=36$, $\kappa=1{,}459$: una PD12 de 10% es una PD de vida de 14,2%.

**VPN por crédito.** Con $i$ la tasa mensual de descuento (costo de fondos) y $F_k=C-\frac{c_{op}}{12}B_{k-1}$ el flujo neto del mes $k$:

$$G=\text{VPN}_{bueno}=-M_0-c_{adq}+\sum_{k=1}^{N}\frac{F_k}{(1+i)^k},\qquad
\text{VPN}_{malo}(d)=-M_0-c_{adq}+\sum_{k=1}^{d-4}\frac{F_k}{(1+i)^k}+\pi\,\frac{\min\{V(d+T)(1-c),A(d)\}}{(1+i)^{d+T}}.$$

La probabilidad de default en $d$ para la banda $b$ es $f_b(d)=\text{Sv}_b(d-1)-\text{Sv}_b(d)$ y

$$\mathbb{E}[\text{VPN}_b]=\text{Sv}_b(N+3)\,G+\sum_d f_b(d)\,\text{VPN}_{malo}(d),\qquad \text{EL}_b=\sum_df_b(d)\,\frac{\text{LGD}(d)\,\text{EAD}(d)}{(1+i)^d}.$$

**Breakeven de vida.** Si los tiempos de default fueran los mismos en todas las bandas ($f_b(d)\approx\text{PD}^{vida}_b\,w(d)$ con $w=s/S_N$, el límite $\lambda\to0$), entonces $\mathbb{E}[\text{VPN}_b]=(1-\text{PD}^{vida}_b)G-\text{PD}^{vida}_bL$, con $L=-\sum_dw(d)\text{VPN}_{malo}(d)$, y el breakeven es la fórmula de §3.3 con $(G,L)$:

$$\text{PD}^{vida}_{BE}=\frac{G}{G+L},\qquad \text{PD}_{12,BE}=1-(1-\text{PD}^{vida}_{BE})^{1/\kappa},\qquad s_{BE}=\text{offset}+\text{factor}\ln\frac{1-\text{PD}_{12,BE}}{\text{PD}_{12,BE}}.$$

La aproximación queda **laxa**. En una banda riesgosa la supervivencia se agota antes, los malos se concentran en los primeros meses y esos son los defaults más caros (más saldo, menos cuotas cobradas, LGD más alta). El breakeven exacto se obtiene resolviendo $\mathbb{E}[\text{VPN}](\text{PD}_{12})=0$ con `brentq`. En el caso base: 533,6 exacto vs 533,0 aproximado. El error en E[VPN] es de 0,3 \$ en A1 y de 19.605 \$ en E.

### 3.7 Pricing por riesgo y el techo legal

**Un período.** La tasa que deja una utilidad esperada $m^{\ast}$ por peso:

$$(1-p)(1+r)+p(1-\text{LGD})=1+k+m^{\ast}\;\Longrightarrow\;\boxed{r(p)=\frac{k+m^{\ast}+p\,\text{LGD}}{1-p}.}$$

(Expandiendo: $(1-p)+(1-p)r+p-p\,\text{LGD}=1+(1-p)r-p\,\text{LGD}$.) Es convexa y creciente en $p$. Con un techo legal $T$ (tasa máxima), la PD más alta que se puede preciar es

$$T=\frac{k+m^{\ast}+p\,\text{LGD}}{1-p}\;\Longrightarrow\;p_{max}=\frac{T-k-m^{\ast}}{T+\text{LGD}}.$$

Con $T$ = 34,5%, $k$ = 12%, $m^{\ast}$ = 0 y LGD 45%: $p_{max}$ = 28,3%, score 513,9. Bajo ese score **no existe tasa legal** que pague el riesgo. La TMC induce un cutoff.

**Vida del crédito.** La tasa mensual $r_b$ resuelve $\mathbb{E}[\text{VPN}_b](r_b)=m^{\ast}M_0$. Es una ecuación escalar monótona en $r$ que se resuelve con bisección o `brentq`. La planilla usa una **secante** entre la tasa base $r_0$ y la TMC mensual $r_T$:

$$r_b\approx r_0+\big(m^{\ast}M_0-E_0\big)\frac{r_T-r_0}{E_T-E_0}.$$

Es exacta en los extremos y su error en el caso base es ≤ 0,022 pp mensuales (mayor en las bandas que caen fuera del intervalo, donde extrapola).

**Cómo se fija la TMC en Chile (verificado).** La Ley 18.010 define el interés corriente como el promedio ponderado por montos de las tasas cobradas por los bancos, por tipo de operación. En general, el interés máximo convencional es el mayor entre 1,5 veces el corriente y el corriente más 2 pp. Para operaciones no reajustables en moneda nacional de 90 días o más y hasta 200 UF, la Ley 20.715 (2013) fijó otro régimen: el interés corriente del tramo > 200 y ≤ 5.000 UF más **14 pp** para operaciones > 50 y ≤ 200 UF, o más **21 pp** para operaciones ≤ 50 UF. La CMF certifica las tasas **cada mes**, expresadas en forma lineal anual, base 360 días. En el certificado vigente desde el 14-08-2026, el corriente del tramo 200–5.000 UF es 20,50%, así que la TMC es 34,50% para > 50–200 UF (20,50 + 14) y 41,50% para ≤ 50 UF (20,50 + 21). Un crédito de moto de 3,2 MM\$ ≈ 78 UF cae en el tramo > 50–200 UF. Si el pacto excede el máximo, la ley lo tiene por no escrito y los intereses se reducen al corriente (art. 8). Qué cuenta como interés (comisiones, gastos) es materia legal. Los números cambian cada mes: el código lee el certificado vigente, no lo fija.

### 3.8 Monto, plazo y pie como palancas

Con pie fijo, $\text{EAD}(d)$, $V$, el recupero y los flujos escalan linealmente con $M_0$, salvo el costo de adquisición, que es fijo por crédito:

$$\mathbb{E}[\text{VPN}_b](M)=\frac{M}{M_0}\big(\mathbb{E}[\text{VPN}_b](M_0)+c_{adq}\big)-c_{adq}.$$

Consecuencias:

- **Monto mínimo.** Si $\mathbb{E}[\text{VPN}_b]$ por peso es positivo, existe un monto bajo el cual el costo fijo no se paga: $M_{min}=c_{adq}M_0/(\mathbb{E}[\text{VPN}_b](M_0)+c_{adq})$.
- **Monto no arregla una banda con VPN por peso negativo.** Reducir el cupo en la banda D acota la EL en pesos, pero si el VPN por peso es negativo solo reduce la pérdida. La palanca efectiva es la **estructura**: pie (baja EAD y LGD a la vez), plazo (reduce $\kappa$ y acelera la amortización frente a la depreciación) o recupero ($\pi$).
- **El pie tiene un límite: el piso $1-\pi$.** En el caso base, la banda E no llega a $\mathbb{E}[\text{VPN}]\ge0$ con ningún pie hasta 80% a la tasa base: con π = 60% la LGD nunca baja de 40%. A la TMC, en cambio, el E[VPN] es positivo en todo el rango de pie evaluado (0–60%), aunque no alcanza el margen objetivo.

### 3.9 Sensibilidad (tornado)

Con $s_{BE}=\text{offset}+\text{factor}\ln(L/G)$ y un parámetro $\theta$:

$$\Delta s_{BE}\approx\text{factor}\Big(\frac{\Delta L}{L}-\frac{\Delta G}{G}\Big).$$

El tornado evalúa exactamente cada extremo. Una fila especial es la **TC**: si la tendencia central verdadera difiere en $\Delta\delta$ log-odds de la calibrada, el corte en el score del modelo debe moverse $\text{factor}\cdot\Delta\delta$ (±7,2 pts para ±0,25), por §3.4. Ese movimiento es independiente de la economía.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| **Cutoff por apetito** (mora o EL media ≤ ρ, aprobación ≥ a) | Límite de riesgo de la cartera | Bajo; no mide utilidad | Siempre como restricción; el directorio fija ρ | Práctica universal; el «apetito de riesgo» es exigencia de gobierno corporativo |
| **Cutoff de breakeven** (odds ≥ $l/g$) | Maximiza la utilidad esperada | Requiere PD calibrada y economía por cliente | Producto con margen conocido; referencia obligada para el comité | Thomas, Edelman y Crook (2002); Lewis (1992) |
| **Hurdle de RAROC / EVA** | Paga el costo del capital | Requiere un modelo de capital | Bancos; productos con consumo de capital heterogéneo | Bancos con capital económico; IRB como referencia |
| **VPN por cliente con vida del crédito** (este módulo) | Horizonte, amortización, garantía | Supuestos de curva de default, LGD(d), prepago | Créditos a plazo con garantía (autos, motos) | Financieras automotrices; se alinea con la EL de vida de IFRS 9 |
| **Cutoff por segmento** | Economía o calibración distinta por segmento | Más parámetros, más monitoreo y ruido de estimación | Segmentos grandes con $(g,l)$ o PD claramente distintos | Clase 4 reserva; exige gobernarlos todos |
| **Matriz score × monto/plazo** (cupos) | Acota la exposición sin cerrar la puerta | Complejidad del motor | Bandas marginales; clientes nuevos | Tarjetas (línea por banda), autos (pie mínimo por banda) |
| **Pricing por riesgo** | La mora extra la paga el margen | Selección adversa y elasticidad; tope legal | PD calibrada y demanda poco sensible al precio | Limitado en Chile por la TMC; en EE.UU. con avisos de *risk-based pricing* |
| **Optimización con restricciones** (LP/MIP sobre acciones por segmento) | Asigna decisiones (aprobar, monto, tasa) maximizando la utilidad sujeta a apetito y volumen | Modelos de respuesta (aceptación, elasticidad) y de riesgo por acción | Carteras grandes, muchas palancas | Herramientas comerciales de optimización de decisiones; literatura de *profit scoring* |
| **Champion/challenger** | Mide el efecto real de una política nueva | Costo de explorar; tiempo de maduración | Siempre que cambie el corte o se amplíe el swap-in | M18 |

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 Tabla de estrategia construida en DEV.**
- *Síntoma*: la mora prometida es menor que la que se observa después.
- *Causa*: DEV es la muestra de ajuste (optimista) y suele ser más antigua.
- *Detección*: comparar las fronteras DEV y OOT. En el notebook, la curva DEV queda bajo la de OOT en prácticamente todo el rango (≈ 1,8 pp de mora menos en promedio).
- *Qué hacer*: tabla en OOT, siempre, y declararlo.

**5.2 EL con PD de ciclo en un periodo malo.**
- *Síntoma*: la mora observada de aprobados supera la PD media en todos los cortes (Austral 560: 2,10% vs 1,59%; Sintético OOT: 15,4% vs 12,6%).
- *Causa*: la PD está anclada a la TC y el periodo es peor que el ciclo.
- *Detección*: O/E por muestra (M15, M19).
- *Qué hacer*: para decidir el corte, usar la PD que representa el **periodo en que la cartera va a vivir** (PIT o escenario), o al menos un Δδ de sensibilidad. Para provisiones y capital, la calibración que corresponda a cada uso, documentadas por separado.

**5.3 Confundir apetito con breakeven.**
- *Síntoma*: se defiende 560 «porque es rentable».
- *Causa*: el apetito restringe el promedio y no dice nada del cliente marginal.
- *Detección*: calcular la utilidad marginal por tramo de score (notebook §4).
- *Qué hacer*: presentar ambos cortes y el **precio sombra** del apetito. En el ejemplo de Austral de §8.1, son ≈ 18 MM\$ en el periodo OOT.

**5.4 LGD constante en un producto con garantía.**
- *Síntoma*: EL correcta en total y sesgada por plazo, pie o banda.
- *Causa*: la LGD depende del mes de default y la distribución de tiempos cambia entre segmentos.
- *Detección*: LGD realizada por mes de default y por plazo, con los casos de cobranza.
- *Qué hacer*: LGD(d) (§3.2) y tiempos por banda (§3.6).

**5.5 PD a 12 meses usada como PD de vida.**
- *Síntoma*: los créditos largos parecen más rentables que los cortos.
- *Causa*: se ignora el riesgo después del mes 12 (κ ≈ 1,46 a 36 meses en el caso base; más a 48).
- *Qué hacer*: convertir con una curva de default estimada de las cosechas (Serie 1 · M3) y declarar la forma.

**5.6 Optimizar sobre bandas.**
- *Síntoma*: el mejor corte «de la tabla» deja utilidad sobre la mesa.
- *Causa*: el óptimo continuo cae dentro de una banda. En el caso motos cae dentro de E (533,6 < 540); E en promedio es no rentable, pero su parte alta sí lo es.
- *Qué hacer*: evaluar la utilidad marginal por tramos finos de score o partir la banda.

**5.7 Olvidar los costos fijos.**
- *Síntoma*: montos chicos aprobados masivamente en bandas buenas.
- *Causa*: la utilidad «por peso» esconde el costo de adquisición fijo.
- *Qué hacer*: monto mínimo por banda (§3.8).

**5.8 Tasa sobre la TMC o TMC desactualizada.**
- *Síntoma*: tabla de pricing con bandas a 37% anual en un tramo con máximo 34,5%.
- *Causa*: pricing sin tope, o tope fijado en el código hace meses.
- *Detección*: test automático contra el certificado vigente (§6).
- *Qué hacer*: la TMC es un dato versionado con fecha; las bandas que no caben se rechazan o se reestructuran.

**5.9 Cutoffs por segmento con muestras chicas.**
- *Síntoma*: los cortes saltan varios puntos entre recalibraciones.
- *Causa*: error estándar de $\hat\delta_j$ ∝ $1/\sqrt{n_j\bar p_j(1-\bar p_j)}$.
- *Detección*: bootstrap de los cortes (notebook §6).
- *Qué hacer*: segmentar solo donde la ganancia esperada supera claramente el ruido; usar contracción hacia el corte común (δ_j con *shrinkage*).

**5.10 Pricing que cambia la población.**
- *Síntoma*: tras subir la tasa de una banda, su mora sube más de lo previsto.
- *Causa*: selección adversa. A tasa mayor aceptan proporcionalmente más los que no tienen alternativa, y la PD estimada a la tasa vieja ya no aplica.
- *Detección*: tasa de aceptación y PD por tasa ofrecida (requiere variación de precios, idealmente experimental).
- *Qué hacer*: pilotear, monitorear la cohorte y no extrapolar la PD fuera del rango de tasas observado.

**5.11 Overlay silencioso.**
- *Síntoma*: la aprobación real no coincide con la del corte firmado.
- *Causa*: corte o excepciones movidos sin acta.
- *Detección*: el motor registra versión de política por decisión; conciliación mensual entre aprobación esperada y real.
- *Qué hacer*: toda desviación es un cambio de política con dueño, fecha y fecha de término.

**5.12 δ aplicado dos veces.** Ver M15 §5.12: tabla re-escalada **y** corte movido. En estrategia se ve como una caída inexplicada de aprobación de $\text{factor}\cdot\delta$ puntos.

**5.13 El swap-in no tiene desempeño.** Si el corte nuevo es más laxo que la política con la que se construyó la muestra, la tabla extrapola. Ver M18.

---

## 6. Puente con ingeniería

La estrategia es **configuración**, no código. Un pipeline serio la trata como un artefacto declarativo derivado del modelo congelado (M21) y de supuestos económicos versionados:

```yaml
# estrategia_motos_v2026_09.yaml
modelo: scorecard_motos@3.1.0          # hash del artefacto congelado (M21)
calibracion: {tipo: TTC, delta: 0.1245, tc: 0.1242, fecha: 2026-09-01}
escala: {pdo: 20, ancla: 600, odds_ancla: 50}
economia:
  tasa_mensual_base: 0.022
  costo_fondos_anual: 0.10            # dueño: tesorería; revisión trimestral
  costo_operativo_anual_saldo: 0.04
  costo_adquisicion_clp: 200000
  lgd: {modelo: lgd_motos@1.2, prob_recupero: 0.60, depreciacion: 0.20, meses_venta: 6}
  curva_default: {forma: gamma_x_exp, mes_pico: 7, fuente: cosechas_2023_2025}
legal:
  tmc: {valor_anual_lineal: 0.3450, tramo: "50-200UF", certificado: "CMF 08/2026", vigente_desde: 2026-08-14}
apetito: {pd12_media_max: 0.045, aprobacion_min: 0.45, aprobado_por: comite_riesgo, acta: CR-2026-17}
politica:
  cutoff_score: 560                     # decisión del comité (puede diferir del breakeven)
  cutoff_breakeven_referencia: 533.6    # derivado, no editable
  reglas: [{banda: D, pie_min: 0.25}, {banda: E, accion: rechazar}]
```

**Qué se congela y qué se versiona.** Se congelan el artefacto del modelo, el δ y la tabla score → PD de la master scale. Se versionan con dueño y fecha los supuestos económicos (cambian con la tesorería y la cobranza), la TMC (cambia **cada mes**), el apetito (lo firma el comité) y la política (cortes, reglas por banda). Todo lo **derivado** se regenera y nunca se edita a mano: tabla de estrategia, breakeven, tasas requeridas.

**Invariantes verificables (tests tipo CI):**

```python
def test_breakeven_maximiza_utilidad(tabla, economia):
    s_be = OFFSET + FACTOR * np.log(economia.l / economia.g)
    assert abs(argmax_utilidad(tabla) - s_be) <= brecha_entre_scores(tabla)

def test_pdo_duplica_odds():
    assert np.isclose(s_be(2 * l, g) - s_be(l, g), PDO)

def test_tasas_bajo_tmc(tabla_pricing, certificado_cmf_vigente):
    tmc_m = certificado_cmf_vigente.tmc(tramo(tabla_pricing.monto_uf)) / 12
    assert (tabla_pricing.tasa_ofrecida <= tmc_m + 1e-12).all()

def test_estrategia_reproducible(config):
    assert hash(construir_tabla(config)) == config.hash_tabla_publicada

def test_frontera_monotona(tabla):
    assert tabla.sort_index().pd_media.is_monotonic_decreasing

def test_lgd_en_rango_y_piso(lgd_por_mes, prob_recupero):
    assert ((lgd_por_mes >= 1 - prob_recupero - 1e-12) & (lgd_por_mes <= 1)).all()
```

**Contratos.** La tabla de estrategia consume un DataFrame OOT con contrato explícito: `score`, `pd_cal`, `malo`, `monto`, `muestra == "OOT"`, sin nulos en `score`. Produce una tabla con esquema fijo (cutoff, aprobación, mora_obs, pd_media, monto, EL, margen, utilidad). El motor de decisión registra, por solicitud, la versión de política, el score, la banda, la PD, la regla que decidió y cualquier excepción con su código. Esa es la materia prima del monitoreo del swap-in (M20) y de la auditoría del overlay.

**Implementación de la tabla.** La versión numpy ordena una vez y usa sumas acumuladas por la cola: una pasada en O(n log n) para cualquier número de cortes. Con millones de solicitudes y miles de cortes, la versión con una máscara por corte es O(n·cortes) y se vuelve lenta. El notebook verifica que ambas coinciden al decimal.

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy (notebook) | Librería | Diferencias / convención | Producción |
|---|---|---|---|---|
| δ exacto de calibración | Bisección sobre la PD media | `scipy.optimize.brentq` | Idénticos a 1e-9; brentq converge más rápido | brentq, con tolerancia declarada |
| Tabla de estrategia | Orden + `cumsum` + `searchsorted` | pandas con máscaras | Idénticas; ojo con `side="left"` (score ≥ corte) | numpy/polars vectorizado |
| Breakeven | Fórmula; bisección sobre π(s) | `brentq`; `minimize_scalar(method="bounded")` sobre U suavizada | minimize_scalar busca un óptimo **local**; aquí U es unimodal (§3.3), en general no | Fórmula cerrada + test de argmax |
| Capital IRB | — | `scipy.stats.norm` vs `statistics.NormalDist` | Idénticos a 1e-10 | scipy |
| Cronograma francés | Fórmula cerrada vs recursión | `numpy_financial.pmt/ppmt` (no usado) | npf usa convención de **signo** de flujo de caja (pmt negativo); Excel `PMT` igual | Fórmula cerrada con test contra el core |
| Tasa requerida | Bisección | `brentq`; secante (planilla) | Secante: error ≤ 0,022 pp mensuales en el caso base; fuera de [r₀, TMC] extrapola | brentq |
| Pie mínimo | Barrido + `brentq` | — | $\mathbb{E}[\text{VPN}](\text{pie})$ **no es monótona** (costo fijo y piso de LGD): brentq solo sirve tras localizar el cambio de signo | Barrido + refinamiento |

**Convenciones de tasa que muerden.** La CMF expresa la TMC en forma **lineal anual base 360**: 34,50% anual = 2,875% mensual (dividir por 12, no $(1{,}345)^{1/12}-1$). El costo de fondos suele venir **efectivo anual** y se convierte con $(1+c_f)^{1/12}-1$. Convertir la TMC como si fuera efectiva da 2,50% mensual en vez de 2,875%: 37,5 pb de error, más que la diferencia de tasa requerida entre la mayoría de las bandas contiguas.

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral: el precio del apetito

La tabla del curso no trae margen. Supongamos, **ilustrativamente**, una tasa anual de 18% y costos totales $k$ = 8% (fondos, operación y adquisición). Aproximando el margen de cada corte con la PD media ($\sum m_i(1-p_i)\approx(1-\bar p)\sum m_i$, razonable si PD y monto no están correlacionados):

| Cutoff | Margen (MM\$) | EL (MM\$) | Utilidad (MM\$) |
|---|---|---|---|
| 500 | 548,0 | 122,0 | 426,0 |
| 520 | 538,9 | 95,8 | 443,1 |
| **540** | 508,2 | 58,3 | **449,9** |
| 560 | 466,0 | 34,4 | 431,6 |
| 580 | 407,3 | 18,6 | 388,7 |

Con esa economía, $g$ = 10%, $l$ = 53%, $O_{BE}$ = 5,3, PD de breakeven 15,9% y $s_{BE}$ = **535,2**. El óptimo de la grilla es 540 y el corte por apetito (560) sacrifica ≈ **18 MM\$** en el periodo OOT. Es el precio de mantener la mora bajo 2,5%. El número depende enteramente del margen supuesto. Para que 560 fuera el breakeven, la PD en 560 (7,4%, odds 12,5:1) exigiría $l/g$ = 12,5; con $k$ = 8% y LGD 45%, eso es una tasa de solo 12,2%. Lectura para el comité: **560 no es el corte rentable, es el corte prudente**, y la diferencia se paga. Si el directorio la acepta, queda en el acta como decisión, no como consecuencia del modelo.

### 8.2 Banco Sintético (notebook)

Generador de la serie, scorecard de 6 variables, TC = 12,42% (δ = +0,1245). En OOT la tasa observada es 15,40%, la PD real media 16,09% y la PD calibrada media 12,63%: el deterioro plantado de 2025. Con LGD 45%, tasa 24% y costos totales 12%:

- **Breakeven**: $l/g$ = 4,75, PD 17,39%, $s_{BE}$ = 532,08. Aprueba 76,5% con mora observada 8,57%. Los cuatro métodos numéricos coinciden (diferencias ≤ 0,05 pts).
- **RAROC ≥ 15%**: 535,4 (aprobación 73,6%). **Apetito de mora ≤ 8%**: 536,9 (72,3%). Precio sombra en utilidad esperada: 3,7 MM\$.
- **Knock-outs** (mora en los últimos 6 meses): aprueban 90,3% con 12,75% de mora. El scorecard a igual aprobación: 11,46% (−10%).
- **Descalibración**: δ que clava la PD real de OOT = +0,327 ⇒ el corte debería subir 9,4 pts. El óptimo verdadero está en 539,7 y usar la PD de la TC cuesta 10,5 MM\$ (2% del máximo). Por la forma cuadrática, sobrecorregir cuesta más que no corregir.
- **Cortes por canal**: δ relativos sucursal −0,06, web −0,11, app +0,23, fuerza de venta +0,12. Cortes 530,4 / 526,1 / 535,9 / 545,3 (fuerza de venta además paga 5% de adquisición). Ganancia en utilidad verdadera: +7,0 MM\$ (+1,3%). A igual aprobación, ordenar por utilidad esperada da +3,6 MM\$. Con 10% de la muestra de calibración la ganancia media cae a +1,3 MM\$ y en 35% de los bootstraps se pierde; con 4%, −6,8 MM\$ (68%).

### 8.3 Caso motos (notebook y planilla; parámetros genéricos e ilustrativos)

**Supuestos base** (rango entre paréntesis): precio 4,0 MM\$ (3–6), pie 20% (10–30), tasa 2,2% mensual = 26,4% lineal anual (1,9–2,6%), plazo 36 meses (24–48), depreciación 20% anual (15–25), castigo inicial 10% (0–20), costo de recupero y remate 15% (10–20), 6 meses de default a venta (4–8), probabilidad de recuperar la moto 60% (40–85), costo de fondos 10% anual (7–14), costo operativo 4% anual del saldo (3–6), adquisición 200.000 \$ (100–300 mil), mes pico de default 7 (5–10). Como referencia de depreciación, las tablas fiscales españolas de valoración de vehículos usados asignan ≈ 84% del valor a 1 año y 67% a 2 años, ≈ 16–20% anual (es una tabla fiscal, no un mercado, y no es chilena). Para Chile, la tasación fiscal anual del SII incluye motos y sirve para construir una curva propia; la depreciación de **remate** es la que importa y solo sale de los casos propios.

**Resultados base.** Monto financiado 3,2 MM\$ (≈ 78 UF ⇒ tramo > 50–200 UF, TMC 34,50%). Cuota 129.613 \$. VPN si bueno $G$ = 442.019 \$.

- **LGD por mes de default**: 54,6% en d = 4, 53,4% en d = 12, piso 40% desde d ≈ 26. La moto neta cubre el saldo desde el mes 24. La LGD ponderada por tiempos y EAD es **53,2%**: cerca del 45% del curso en orden de magnitud, pero con forma.
- **κ** = 1,459. PD de vida: D 10,16% → 14,47%; E 26,75% → 36,50%.
- **Por banda** (PD12 de Austral como ejemplo): E[VPN] A1 439.032 \$, C2 295.648 \$, D 169.949 \$, **E −257.041 \$**. EL/monto (VP): D 6,1%, E 15,7%.
- **Breakeven**: PD12 16,66% ⇒ **533,6** exacto (533,0 con la fórmula cerrada de la planilla). Cae dentro de E.
- **Tabla de estrategia** (1.000 solicitudes/mes, mezcla ilustrativa desplazada a bandas bajas): la utilidad es máxima aprobando **hasta D** (67% de aprobación, PD12 media 5,15%, 203,4 MM\$/mes). Aprobar todo da 118,6 MM\$. Con apetito PD12 media ≤ 4,5% y aprobación ≥ 45%, el corte es C2 (47%, 169,4 MM\$) y el precio sombra 34,0 MM\$/mes.
- **Pricing** con margen objetivo 5%: tasa requerida de 1,75% mensual (A1) a 2,18% (D) y **3,10% (E) > TMC 2,875%**. E no cabe bajo el máximo legal.
- **Tornado** del breakeven (puntos de rango): costo de fondos 27,1 · tasa 26,3 · plazo 20,8 · adquisición 16,3 · TC ±0,25 14,4 · mes pico 13,6 · prob. recupero 11,3 · costo operativo 10,8 · depreciación 3,7 · costo de remate 2,6 · meses a venta 2,4 · pie 1,6. La economía del margen domina a la del recupero. Donde conviene invertir en estimación es en la tasa efectivamente cobrada (descuentos, prepago), el costo de fondos marginal y la curva de default, antes que en afinar la depreciación.

Para una fintech de motos, el caso deja tres lecturas. (1) El corte óptimo no sale del Gini ni de la mora media: sale de $G$ y $L$, y $G$ es frágil. (2) La TMC convierte la banda E en un problema de **estructura** (pie, plazo, recupero), no de precio. (3) La probabilidad de recuperar la moto fija el piso de la LGD y es el supuesto con peor evidencia. Medirla con los casos propios de cobranza vale más que cualquier refinamiento del scorecard en la zona del corte.

---

## 9. Preguntas de comité

**1. «¿Por qué 560 y no el corte que maximiza la utilidad?»**
Porque el comité fijó un apetito de mora de 2,5% sobre la cartera aprobada, y el corte de máxima utilidad (≈ 535–540 con un margen ilustrativo de 10%) no lo cumple. La diferencia de utilidad es el precio de ese apetito: ≈ 18 MM\$ en el periodo OOT con los supuestos de §8.1. Traemos ambos números para que el comité decida el trade-off explícitamente.

**2. «¿La EL de la tabla es la pérdida que vamos a ver?»**
No necesariamente. Usa la PD calibrada al ciclo (TC), y OOT muestra más mora que la PD en todos los cortes (560: 2,10% vs 1,59%). Si el periodo que viene se parece a OOT, la EL está subestimada. Presentamos una sensibilidad con Δδ: cada +0,1 en log-odds sube la EL ≈ 10% y el corte de breakeven 2,9 pts.

**3. «Comercial pide 540 porque la meta de colocación no se negocia. ¿Qué opciones hay?»**
(a) 540 con acta, declarando que excede el apetito (3,16% vs 2,5%) y su EL adicional. (b) 560 más una matriz de cupos: aprobar la banda D (540–560) con monto menor o más pie, lo que acota la EL sin cerrar la puerta. (c) Cutoff por segmento si hay segmentos con economía distinta. (d) Precio mayor para D si cabe bajo la TMC. Lo que no se hace es mover el corte «esta semana» sin acta.

**4. «¿Qué supuesto, si está mal, más nos mueve el corte?»**
En el caso motos, el costo de fondos y la tasa efectiva (±13 pts cada uno en su rango), luego el plazo y el costo de adquisición. La TC mueve ±7,2 pts por cada ±0,25 log-odds. Depreciación y costo de remate pesan poco. El tornado está en el notebook §8.

**5. «¿Por qué no usar cortes distintos por canal?»**
Solo vale la pena si el canal cambia la economía (adquisición) o la PD a igual score. En Banco Sintético gana +1,3% de utilidad con muestras grandes. Con ~600 casos de calibración por el total, el ruido de estimar δ por canal hace que pierda en 2 de cada 3 escenarios. Si la PD difiere por canal, lo correcto es calibrarla por canal en la PD, no esconderla en la política.

**6. «Esta banda necesita 37% anual. ¿La podemos cobrar?»**
No en el tramo de 50–200 UF: la TMC vigente (certificado CMF desde el 14-08-2026) es 34,50%, y el pacto sobre el máximo se tiene por no escrito. Para esa banda: rechazo, más pie o menos plazo. Con π = 60% el pie no basta (la LGD tiene piso de 40%), así que la palanca real es subir la probabilidad de recupero.

**7. «¿Por qué construyeron la tabla en OOT y no en toda la historia?»**
Porque la decisión aplica a la cartera que viene y OOT es la muestra con desempeño más parecida. DEV está sesgada a favor del modelo (ajuste) y es más antigua. En el notebook, la frontera DEV promete ≈ 1,8 pp menos de mora que la OOT al mismo corte.

**8. «¿Esta tabla vale para los clientes que antes rechazábamos?»**
Solo con cautela. Si el corte nuevo aprueba población que la política anterior rechazaba, la tabla extrapola: no hay desempeño de esos clientes. Se monitorean como cohorte aparte (swap-in, M18) con umbral y fecha de revisión.

---

## 10. Ejercicios

**E1. Breakeven a mano.** LGD 45%, tasa 24%, costo de fondos 6%, costos operativos y de adquisición 6%. Calcule $g$, $l$, las odds y la PD de breakeven y $s_{BE}$ en la escala del curso. ¿En qué banda de la master scale cae?

<details><summary>Solución</summary>

$k$ = 12%, $g$ = 12%, $l$ = 57%. $O_{BE}$ = 4,75; PD = 0,12/0,69 = 17,39%. $s_{BE}$ = 487,12 + 28,854·ln 4,75 = 487,12 + 44,96 = **532,1**: banda E (< 540). Es el número del notebook §4.
</details>

**E2. Sensibilidad analítica.** Con los datos de E1, sin recalcular la fórmula completa, ¿cuánto sube el corte si la LGD pasa a 60%? ¿Y si los costos suben 1 pp? Compare con el cálculo exacto.

<details><summary>Solución</summary>

$\partial s/\partial\text{LGD}=\text{factor}/l$ = 28,85/0,57 = 50,6 pts por unidad ⇒ +15 pp ≈ +7,6 pts. Exacto: 487,12 + 28,85·ln(0,72/0,12) = 538,8, es decir +6,7 pts (la derivada sobreestima porque $\ln$ es cóncavo). Costos: $\text{factor}(1/l+1/g)$ = 28,85·(1,754 + 8,333) = 291 pts por unidad ⇒ +1 pp ≈ +2,9 pts. Los costos pesan 5,7 veces más que la LGD por punto porcentual.
</details>

**E3. Derivación: el breakeven maximiza la utilidad.** Demuestre que si $p(s)$ es estrictamente decreciente y $m_i>0$, $U(c)=\sum_{s_i\ge c}u_i$ alcanza su máximo en el menor score observado $\ge s_{BE}$. ¿Qué pasa si la PD en uso **no** es monótona en el score (p. ej. una PD de ML con overlays por segmento)?

<details><summary>Solución</summary>

$u_i=m_i[(1-p_i)g-p_il]$ tiene el signo de $g-p_i(g+l)$, positivo sii $p_i<p_{BE}$ sii $s_i>s_{BE}$. Ordenando scores, al bajar $c$ de un score al siguiente se suma un $u_i$: positivo mientras $s_i>s_{BE}$ y negativo después. $U$ crece y luego decrece, con máximo en el último $s_i\ge s_{BE}$ incluido. Si $p$ no es monótona en $s$, el conjunto óptimo $\{i: u_i>0\}$ **no es un intervalo de score**. Ningún cutoff único lo alcanza: hay que decidir por PD (o por $u_i$), no por score. Por eso, cuando hay overlays, el motor debe aplicar el corte a la PD final.
</details>

**E4. Precio sombra en Austral.** Con la tabla de §1 y tasa 20%, $k$ = 8%, recalcule la utilidad aproximada de los cortes 520, 540 y 560. ¿Cambia el óptimo? ¿Cuánto cuesta el apetito?

<details><summary>Solución</summary>

Margen ≈ monto·((1 − PD)·0,20 − 0,08). 520: 5.776·(0,19254 − 0,08) − 95,8 = 650,0 − 95,8 = 554,2. 540: 5.315·(0,19512 − 0,08) − 58,3 = 611,9 − 58,3 = 553,6. 560: 4.797·(0,19682 − 0,08) − 34,4 = 560,4 − 34,4 = 526,0. Con más margen el óptimo se corre hacia 520–540 (breakeven: $l/g$ = 0,53/0,12 = 4,42 ⇒ 530,0). El apetito (560) cuesta ≈ 28 MM\$: sube con el margen, porque el apetito deja fuera clientes cada vez más rentables.
</details>

**E5. LGD por mes a mano.** Moto de 4,0 MM\$, pie 20%, tasa 2,2% mensual, 36 cuotas. Default en d = 10 (última cuota pagada: 6). Castigo inicial 10%, depreciación 20% anual, venta 6 meses después, costo de remate 15%, π = 60%, descuento mensual 0,797%. Calcule EAD, valor neto de venta, LGD si se recupera y LGD esperada.

<details><summary>Solución</summary>

$M_0$ = 3,2 MM\$. $B_6=3{,}2\cdot\frac{1{,}022^{36}-1{,}022^{6}}{1{,}022^{36}-1}$. $1{,}022^{36}$ = 2,1885; $1{,}022^{6}$ = 1,1395 ⇒ $B_6$ = 3,2·1,0490/1,1885 = **2,824 MM\$**. Venta en t = 16: $V$ = 4,0·0,9·0,8^{16/12} = 3,6·0,7427 = 2,674; neto 0,85·2,674 = **2,273 MM\$** (bajo la acreencia). Descontado 6 meses: 2,273/1,0489 = 2,167. LGD si se recupera = 1 − 2,167/2,824 = **23,3%**. LGD esperada = 0,6·0,233 + 0,4 = **54,0%** (el notebook da 54,0% en d = 10).
</details>

**E6. PD de vida.** Con la forma $s(d)=x e^{1-x}$, $x=(d-3)/4$, calcule κ para plazo 24 y 48 (use la aproximación continua $\int_0^X xe^{1-x}dx=e[1-(1+X)e^{-X}]$ con $X_{12}=9/4$, $X_{24}=24/4$, $X_{48}=48/4$). ¿Cuál es la PD de vida de una PD12 de 10% en cada plazo?

<details><summary>Solución</summary>

$I(X)=e[1-(1+X)e^{-X}]$. $I(2{,}25)$ = e·(1 − 3,25·0,1054) = e·0,6575. $I(6)$ = e·(1 − 7·0,00248) = e·0,9826. $I(12)$ = e·(1 − 13·6,1e−6) ≈ e·0,99992. κ₂₄ ≈ 1,494, κ₄₈ ≈ 1,521 (continuo; el discreto del notebook para 36 da 1,459). PD de vida: 1 − 0,9^{1,494} = 14,6% y 1 − 0,9^{1,521} = 14,8%. Con esta forma la cola después del mes 24 aporta poco. Una forma con cola más pesada (p. ej. un piso de intensidad) separaría mucho más los plazos: la forma es el supuesto, y sale de las cosechas.
</details>

**E7. Techo legal.** Modelo de un período: $k$ = 12%, LGD 45%, margen objetivo 3%. ¿Qué PD máxima se puede preciar bajo una TMC de 34,5%? ¿Y en el tramo ≤ 50 UF (41,5%)? Tradúzcalo a score.

<details><summary>Solución</summary>

$p_{max}=(T-k-m^{\ast})/(T+\text{LGD})$. Con 34,5%: 0,195/0,795 = 24,5% ⇒ $s$ = 487,12 + 28,85·ln(0,755/0,245) = 519,6. Con 41,5%: 0,265/0,865 = 30,6% ⇒ $s$ = 487,12 + 28,85·ln(0,694/0,306) = 510,7. El tramo chico admite ≈ 9 puntos más de riesgo. Ojo: ese tramo tiene también más costo fijo relativo al monto, así que $k$ en % del monto es mayor.
</details>

**E8. Cortes por segmento con ruido.** Dos canales con igual economía. Canal A: n = 8.000, tasa 10%; canal B: n = 400, tasa 14%, y δ_B − δ_A estimado = 0,20. (a) ¿Cuánto difieren los cortes? (b) ¿Cuál es el error estándar aproximado de la diferencia en puntos? (c) ¿Recomendaría segmentar?

<details><summary>Solución</summary>

(a) factor·0,20 = 5,8 pts. (b) $se(\hat\delta)\approx1/\sqrt{n\bar p(1-\bar p)}$: A: 1/√(8.000·0,09) = 0,037; B: 1/√(400·0,1204) = 0,144; se de la diferencia ≈ √(0,037² + 0,144²) = 0,149 ⇒ 4,3 pts. (c) La diferencia es 1,3 errores estándar: no se distingue del ruido. Mejor un corte común y calibrar B con contracción (δ_B* = ω·0,20 con ω = τ²/(τ² + se²)), o acumular casos antes de segmentar. Si se segmenta, que sea con fecha de revisión y en acta.
</details>

**E9. Código.** Modifique `tabla_estrategia_np` para que el corte se aplique sobre la **utilidad esperada** $u_i$ en lugar del score y verifique que, con PD monótona y una sola economía, la tabla por utilidad coincide con la tabla por score. Luego introduzca costos de adquisición por canal (notebook §6) y muestre que ya no coinciden.

<details><summary>Solución</summary>

Basta reemplazar `score` por `u_i` en el ordenamiento y usar cortes en pesos (por ejemplo, cuantiles de $u_i$). Con una economía, $u_i/m_i$ es función decreciente de $p_i$ y creciente de $s_i$: el orden por $u_i/m_i$ coincide con el orden por score, y a igual aprobación los conjuntos son idénticos. Con $u_i$ total (no por peso) difieren levemente, porque el monto pesa. Con $k$ por canal, $u_i/m_i$ depende del canal a igual score: el orden cambia y la tabla por utilidad domina a la tabla por score a igual aprobación (+3,6 MM\$ en el notebook).
</details>

**E10. Diseño.** Escriba el contrato (campos, tipos, dueño, frecuencia) del objeto «supuestos económicos» de la estrategia de motos y tres tests que fallen si alguien actualiza la TMC sin actualizar la tabla de pricing.

<details><summary>Solución</summary>

Campos: `tasa_base` (float, mensual, dueño comercial), `costo_fondos` (float, efectivo anual, tesorería, trimestral), `costo_operativo` (float, % anual saldo, finanzas, anual), `costo_adquisicion` (CLP, comercial, semestral), `lgd_modelo` (versión, riesgo), `prob_recupero` (float, cobranza, semestral, con n de casos), `curva_default` (versión, riesgo), `tmc` (float lineal anual, tramo, n.º de certificado, vigente_desde, legal/cumplimiento, **mensual**). Tests: (1) `tabla_pricing.certificado == supuestos.tmc.certificado`; (2) todas las tasas ofrecidas ≤ TMC mensual vigente; (3) la fecha de la tabla de pricing ≥ `tmc.vigente_desde`. Opcional: un test que rechace el despliegue si el certificado vigente publicado por la CMF difiere del configurado.
</details>

---

## 11. Referencias

- **Thomas, L. C., Edelman, D. B. y Crook, J. N. (2002).** *Credit Scoring and Its Applications.* SIAM (2.ª ed. 2017 con Thomas, Crook y Edelman — verificar edición). — Capítulos sobre cutoff, odds y decisiones basadas en utilidad; la derivación clásica del breakeven.
- **Thomas, L. C. (2009).** *Consumer Credit Models: Pricing, Profit and Portfolios.* Oxford University Press. — El tratamiento más completo de pricing por riesgo, utilidad y selección adversa en crédito de consumo; base de §3.7 y §5.10.
- **Lewis, E. M. (1992).** *An Introduction to Credit Scoring.* Fair, Isaac & Co. (verificar editorial). — Clásico de práctica: odds, cutoffs y la lógica de «cuántos buenos por malo» que usa la industria.
- **Siddiqi, N. (2017).** *Intelligent Credit Scoring* (2.ª ed.). Wiley. — Capítulos de estrategia: tablas de ganancias, cutoffs, políticas y overrides; lenguaje de implementación.
- **Anderson, R. (2007).** *The Credit Scoring Toolkit.* Oxford University Press. — Estrategia, matrices de decisión score × monto y gobierno de overrides en motores de decisión.
- **Crook, J. N., Edelman, D. B. y Thomas, L. C. (2007).** «Recent developments in consumer credit risk assessment». *European Journal of Operational Research* 183(3), 1447–1465. — Panorama de scoring por utilidad y temas abiertos.
- **Finlay, S. (2010).** «Credit scoring for profitability objectives». *European Journal of Operational Research* 202(2), 528–537 (verificar páginas). — Construir el score para maximizar utilidad en vez de discriminar buenos/malos.
- **Verbraken, T., Bravo, C., Weber, R. y Baesens, B. (2014).** «Development and application of consumer credit scoring models using profit-based classification measures». *EJOR* 238(2), 505–513 (verificar páginas). — Métrica EMP: evaluar modelos por la utilidad que generan en el corte.
- **Baesens, B., Rösch, D. y Scheule, H. (2016).** *Credit Risk Analytics.* Wiley. — Modelación de LGD y EAD, incluido el descuento y la LGD de exposiciones con garantía.
- **Schuermann, T. (2004).** «What do we know about Loss Given Default?». Wharton Financial Institutions Center, working paper (publicado en *Credit Risk Models and Management*, Risk Books — verificar edición). — Por qué la LGD es bimodal, cíclica y dependiente de la garantía.
- **Basel Committee on Banking Supervision (2005).** *An Explanatory Note on the Basel II IRB Risk Weight Functions.* BIS. — Derivación de la fórmula de capital de §3.3 y de la correlación de «otras minoristas».
- **Basel Committee on Banking Supervision (2006).** *International Convergence of Capital Measurement and Capital Standards* (Basilea II, versión integral), sección de exposiciones minoristas (verificar numeración de párrafos). — Fuente normativa de la fórmula IRB minorista.
- **IASB. IFRS 9 *Financial Instruments*, párr. 5.5.17.** — La EL como esperanza ponderada, descontada y con información prospectiva: el marco de la EL de vida de §3.1.
- **Ley N.º 18.010** sobre operaciones de crédito de dinero (arts. 6, 6 bis y 8) y **Ley N.º 20.715** (2013). — Definición de interés corriente y máximo convencional, régimen de operaciones ≤ 200 UF y sanción del exceso.
- **CMF (2026).** *Certificado mensual de tasas de interés corriente y máxima convencional*, agosto 2026 (vigente desde el 14-08-2026). — Fuente de los valores de §3.7; consultar el certificado vigente en cada uso.
- **Biblioteca del Congreso Nacional (2022).** *Tasa Máxima Convencional, tasa de interés corriente y su…* (informe de asesoría técnica parlamentaria, mayo 2022; título abreviado, verificar el completo). — Resumen claro del régimen por tramos y su historia.
- **Serie 1 · M1** (economía del error), **M3** (curvas de maduración, para la forma $s(d)$), **E1** (reject inference), **E2** (PIT vs TTC), **E6** (regulación y capital); **Serie 2 · M13** (scaling y PDO), **M15** (calibración y δ), **M16** (master scale), **M18** (swap-set), **M20** (monitoreo), **M21** (artefacto congelado).

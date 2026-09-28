# M19 · Backtesting de calibración: binomial, Hosmer-Lemeshow y más

> **Ficha.** Profundiza: clase 5, parte 3 (láminas 19–25: binomial por banda, Hosmer-Lemeshow) y láminas 27–29 (tablero, «¿un modelo nuevo puede nacer con amarillos?», lectura por patrón); demo de clase 5, sección 8; clase 6, lámina 9 y demo nikodym, paso 4 (HL de OOT que «falla» con p < 0,001). · Prerrequisitos: Serie 1 · E2 (tendencia central, PIT vs TTC), E3 (bootstrap e intervalos en carteras chicas), E6 (regulación), E7 (eventos extraordinarios); Serie 2 · M15 (calibración, Wilson/Jeffreys, métricas), M16 (master scale). Prepara M20 (tablero y gatillos) y M22 (gobierno). · Archivos: `M19_backtesting_calibracion.md` (este documento), `M19_backtesting_calibracion.py` (notebook Marimo), `M19_backtesting_bandas.xlsx` (calculadora por banda con fórmulas vivas). · Tiempo estimado: 3,5 h de lectura + 2,5 h de notebook, planilla y ejercicios.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**El test binomial por banda.** La clase 5 planteó la pregunta en su forma más limpia: a la banda B2 se le prometió una PD de 1,42%; tiene 235 créditos en OOT y llegaron 8 malos. Esperados: 235 × 1,42% = 3,3. «Ver 8 malos o más pasa el 2,0% de las veces»: p = 0,020, amarillo con el semáforo fijado antes de mirar (p ≥ 0,05 🟢 · 0,01–0,05 🟡 · < 0,01 🔴). La tabla completa de OOT (lámina 22):

| Banda | n | PD calibrada | Esperados | Malos | Tasa obs. | p (curso) | Estado |
|---|---|---|---|---|---|---|---|
| A1 | 432 | 0,11% | 0,5 | 1 | 0,23% | 0,373 | 🟢 |
| A2 | 199 | 0,36% | 0,7 | 3 | 1,51% | 0,036 | 🟡 |
| B1 | 244 | 0,72% | 1,8 | 4 | 1,64% | 0,100 | 🟢 |
| B2 | 235 | 1,42% | 3,3 | 8 | 3,40% | 0,020 | 🟡 |
| C1 | 239 | 2,80% | 6,7 | 8 | 3,35% | 0,554 | 🟢 |
| C2 | 220 | 5,44% | 12,0 | 9 | 4,09% | 0,458 | 🟢 |
| D | 172 | 10,21% | 17,6 | 22 | 12,79% | 0,257 | 🟢 |
| E | 263 | 26,91% | 70,8 | 64 | 24,33% | 0,367 | 🟢 |
| GLOBAL | 2.004 | 5,65% | 113,2 | 119 | 5,94% | 0,562 | 🟢 |

La nota al pie dice «p-valor binomial exacto, bilateral» (en el código, `scipy.stats.binomtest` con la alternativa por defecto). El global confirmó que el residuo de −0,29 pp de la clase 4 era ruido (p 0,56). La sensibilidad de re-anclar la tendencia central solo con las 8 cosechas pre-OOT llevó el global a p 0,28 y a **A2 de 0,036 a 0,006 (rojo)**. Lectura del curso: el patrón pesa más que el color; las dos bandas que fallan son **buenas** y ambas **subestiman**.

**Hosmer-Lemeshow.** Diez grupos por PD; en cada uno $(O-E)^2/[E(1-\bar p)]$; suma = 29,0 en OOT. La autopsia (lámina 24): el grupo 3 (esperaba 0,641, observó 3) aporta 8,7 y el grupo 5 (esperaba 2,191, observó 8) aporta 15,6; entre los dos, 24,3 de 29,0. La tabla χ² daba p = 0,0003 (rojo); el p **simulado** bajo $H_0$ (10.000 réplicas con las PD individuales) dio 0,012 (amarillo). Por muestra: DEV 0,80 · HO 0,27 · OOT 0,012. Re-anclado pre-OOT: HL 32,0, p simulado 0,0096. Frase de la clase: «un rojo que depende de una aproximación inválida no es un rojo».

**El tablero y el patrón.** Nueve indicadores, cinco amarillos coherentes (B2/A2, HL, CSI de `deuda_interna_max_3m` 0,138, mix D+E +3,6 pts, swap-in 9,4% vs 5,1% con p 0,04). «Con 9 indicadores a un 5% de tolerancia, esperaríamos ~1 amarillo por puro azar (1 − 0,95⁸ = 34% de que salga al menos uno).» Diagnóstico: deriva incipiente de **nivel** en los tramos buenos con el ranking sano; acción: vigilancia reforzada y recalibración de δ programada, no re-desarrollo.

**nikodym (clase 6).** La corrida validada de Banco Austral con la librería: Gini OOT 0,694, PSI 0,008, y calibración ❌: PD esperada 6,08% vs mora 5,94% en OOT, «HL p < 0,001: falla por tramos». Los estadísticos del artefacto `validation/calibration` son DEV 8,96 (p 0,345), HO 15,15 (p 0,056) y OOT 49,33 (p ≈ 5·10⁻⁸). Esos tres p coinciden con $\chi^2_8$, es decir, $G-2$ grados de libertad **también en HO y OOT** (§3.6).

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **Una cola vs dos colas.** La lámina 20 explica una cola («8 o más») y la 22 dice «bilateral exacto». En B2 dan lo mismo (0,020), y no es casualidad (§3.2). Con la definición bilateral «doble» que usan muchas planillas, B2 sería 0,040 y A2 0,072 (verde). La definición cambia colores (la misma ambigüedad aparece en la cohorte swap-in: 0,031 / 0,041 / 0,062, ver M18 §3.8).
2. **La aproximación normal** y por qué no sirve con $np<5$ (A2 daría rojo falso: p 0,007).
3. **Potencia.** Un verde en A1 no dice nada: para detectar con 80% de probabilidad hace falta que la PD real de A1 sea ~9 veces la calibrada.
4. **Correlación de defaults.** Todo el backtesting del curso supone independencia entre deudores. Con un factor común (Vasicek) la varianza de la tasa de default se infla y el binomial rechaza de más en años malos. El curso no lo mencionó.
5. **Grados de libertad del HL**: $G-2$ en desarrollo vs $G$ en validación externa; dependencia del agrupamiento y de $G$.
6. **Otros tests**: Spiegelhalter, Jeffreys (el que exige el BCE), semáforo de Tasche, pendiente de calibración.
7. **Multiplicidad.** El «34%» es $1-0{,}95^8$ (8 tests), no 9; con 9 sería 37%. Además los tests exactos discretos tienen tamaño real < 5% y los indicadores no son independientes. Ninguna de las dos bandas amarillas sobrevive a Holm.
8. **El p simulado tiene error Monte Carlo**: 0,012 ± 0,002 con 10.000 réplicas, pegado al umbral de 0,01.
9. **La PD de banda es un promedio**: el conteo real es Poisson-binomial, no binomial. La demo lo comentó; aquí se cuantifica el sentido del sesgo.

---

## 2. Intuición

**Una promesa por banda, un solo año de evidencia.** El backtesting compara lo que la master scale prometió con lo que pasó en *una* ventana de desempeño. Tres fuentes de discrepancia se mezclan en el número observado: (i) el **ruido binomial**, porque aun con la PD exacta cada crédito es una moneda cargada; (ii) el **año**, porque todos los deudores de la ventana vivieron la misma macro, y un año malo mueve todas las bandas juntas; (iii) el **error del modelo**, que es lo único que queremos detectar. El test binomial estándar solo conoce la fuente (i). Si (ii) existe y no se modela, el test la atribuye a (iii).

**Asimetría de información entre bandas.** Una banda con 0,5 malos esperados no puede demostrar casi nada. Con 1 malo observado su tasa es el doble de la prometida, y aun así el p es 0,38. Con 3 malos (A2) la tasa es 4 veces la prometida y el p apenas cruza 0,05. Las bandas buenas son ciegas a desvíos *relativos* enormes, justamente donde un error de nivel duele poco en pérdida absoluta pero mucho en pricing y en capital. Por eso un tablero sano mira el patrón a través de las bandas y no el color de cada una.

**Nivel, forma y ruido son preguntas distintas.** El binomial global contesta «¿la PD media calza?». El binomial por banda y el HL contestan «¿la curva calza tramo a tramo?». La pendiente de calibración contesta «¿las PD son demasiado extremas o demasiado tímidas?». En Austral OOT el global dice verde (p 0,56) y el HL amarillo (p 0,012). No se contradicen: la media calza porque los excesos de las bandas buenas se compensan con los déficits de C2 y E.

**Los tests del validador también tienen supuestos.** La χ² del HL supone esperados grandes; el binomial supone independencia; el p simulado supone que el simulador es la $H_0$ correcta; la corrección de Holm supone que se quiere controlar la probabilidad de *un* falso positivo. Validar el modelo incluye validar el test.

---

## 3. Formalización

### 3.1 Planteamiento

Sea una master scale con $B$ bandas. En la banda $b$ hay $n_b$ créditos con PD asignadas $p_{bi}$ y media $\text{PD}_b=\bar p_b$, y se observan $D_b=\sum_i y_{bi}$ malos al cierre de la ventana (12 meses, 90+). La hipótesis de calibración **moderada** (M15 §3.7) por banda es

$$H_0:\ E[D_b]=\sum_i p_{bi}=n_b\,\text{PD}_b\quad\text{para todo } b .$$

Para que $H_0$ determine una distribución hace falta un supuesto de dependencia. El estándar es **independencia condicional a las PD**: $y_{bi}\sim\text{Bernoulli}(p_{bi})$ independientes. Entonces $D_b$ es **Poisson-binomial**. Si además se reemplazan las $p_{bi}$ por su media, $D_b\sim\text{Bin}(n_b,\text{PD}_b)$. El efecto de ese reemplazo tiene signo conocido:

$$\operatorname{Var}_{\text{PB}}(D_b)=\sum_i p_{bi}(1-p_{bi})=n_b\bar p_b(1-\bar p_b)-\sum_i(p_{bi}-\bar p_b)^2\le n_b\bar p_b(1-\bar p_b).$$

(Se obtiene desarrollando $\sum_i p_{bi}^2=n_b\bar p_b^2+\sum_i(p_{bi}-\bar p_b)^2$.) El binomial con PD media sobreestima la varianza, así que es **levemente conservador**: da p algo mayores que los exactos. En bandas angostas de una master scale la diferencia es despreciable; en un decil que va de 18% a 90% de PD (el grupo 10 de Austral), no.

Todo lo que sigue supone que las PD se fijaron **antes** y con **otros datos** (validación externa). Si la muestra de validación se usó para calibrar (el ancla TC que incluye las cosechas OOT, o la calibración PIT sobre OOT), el test pierde grados de libertad o queda trivial: la calibración PIT sobre OOT da p = 1,00 por construcción (M15 §1 y §5).

### 3.2 El binomial exacto: una cola y tres formas de «dos colas»

**Una cola.** Si solo importa la subestimación (la mirada del supervisor y de capital):

$$p^{\uparrow}=P(D\ge k\mid n,\text{PD})=\sum_{j=k}^{n}\binom{n}{j}\text{PD}^j(1-\text{PD})^{n-j}.$$

Identidad útil, que conecta con Jeffreys (§3.8). Derivando respecto de $p$ la suma telescopa (cada término positivo se cancela con el negativo del siguiente) y queda

$$\frac{d}{dp}P(D\ge k)=n\binom{n-1}{k-1}p^{k-1}(1-p)^{n-k}=\frac{p^{k-1}(1-p)^{n-k}}{B(k,n-k+1)}.$$

Como $P(D\ge k)=0$ en $p=0$, integrando de 0 a $p$:

$$P(D\ge k\mid n,p)=I_p(k,\,n-k+1),$$

donde $I_p(a,b)$ es la beta incompleta regularizada, es decir, la cdf de una $\text{Beta}(a,b)$ evaluada en $p$. La planilla la usa implícitamente vía `BINOM.DIST`; el notebook la usa para verificar Jeffreys.

**Dos colas.** No hay una única definición, y las tres que circulan dan números distintos:

1. **«Doble»**: $p^{\pm}_{\text{doble}}=\min\{1,\,2\min(P(D\ge k),P(D\le k))\}$. Es la de la mayoría de las planillas y la de la planilla de este módulo (con aviso).
2. **«Minlike»** (probabilidades ≤ la observada): $p^{\pm}_{\text{min}}=\sum_{j:\,P(D=j)\le P(D=k)}P(D=j)$. Es la de `scipy.stats.binomtest` y de R `binom.test`, y la que produjo la tabla del curso. `scipy` compara con una tolerancia relativa $1+10^{-7}$ para no perder empates numéricos. El notebook la replica.
3. **Mid-p**: $P(D>k)+\tfrac12P(D=k)$ (por cola, luego se dobla). Reduce el conservadurismo de la discreción; no controla el tamaño en forma exacta.

**Por qué en B2 «bilateral» = «una cola».** La pmf binomial es unimodal. Con $n=235$ y $\text{PD}=0{,}0142$:

$$P(D=0)=(1-0{,}0142)^{235}=e^{235\ln 0{,}9858}=e^{-3{,}361}=0{,}0347,\qquad P(D=8)=0{,}0132 .$$

El resultado menos probable de la cola izquierda ($D=0$) es **más probable** que el observado. Ningún punto de la izquierda entra a la suma minlike y $p^{\pm}_{\text{min}}=P(D\ge8)=0{,}0201$. Las láminas 20 («8 o más») y 22 («bilateral exacto») dicen lo mismo en B2, y también en A1, A2 y B1, donde $np<2$. En C1 ya no: minlike 0,554 contra una cola 0,355. La condición general: el minlike coincide con la cola superior si $P(D=0)>P(D=k)$. Eso ocurre siempre que $np$ es chico y $k$ está en la cola derecha.

**Reproducción completa** (notebook §2): minlike numpy = scipy en las 8 bandas, y reproduce la lámina al tercer decimal salvo A1 (0,378 vs 0,373) y B1 (0,101 vs 0,100). La diferencia viene de las PD impresas redondeadas: en A1 (0,11%) la PD media que reproduce 0,3733 es 0,108%. Global: PD media ponderada 5,6527%, 113,28 esperados, p = 0,562.

**Tamaño real y discreción.** Un test exacto discreto no alcanza su nivel nominal. Con test de una cola al 5% el valor crítico de B2 es $k^*=8$ ($P(D\ge8)=0{,}0201\le0{,}05<P(D\ge7)$). Su tamaño real es 2,0%, no 5%. Los tamaños por banda van de 1,25% (A1) a 4,5% (D). El binomial exacto es **conservador**, y eso importa para multiplicidad (§3.10).

### 3.3 La aproximación normal y cuándo falla

$z=(k-np)/\sqrt{np(1-p)}$, con $p^{\pm}=2[1-\Phi(|z|)]$. Es la que aparece en muchos manuales y planillas («test normal» o «z-test» por grado). Falla por **asimetría**. La binomial tiene coeficiente de asimetría

$$\gamma_1=\frac{1-2p}{\sqrt{np(1-p)}},$$

que en Austral vale 1,45 (A1), 1,18 (A2), 0,54 (B2), 0,37 (C1) y 0,06 (E). La normal es simétrica, así que con $\gamma_1$ grande **subestima la cola derecha lejana**. En A2: $z=(3-0{,}7164)/\sqrt{0{,}7138}=2{,}70$, $p^{\pm}_{\text{normal}}=0{,}0069$ (🔴) contra 0,036 exacto (🟡). Es un rojo falso fabricado por la aproximación. En B2 da 0,0101, pegado al rojo, contra 0,020.

Reglas de pulgar: $np\ge5$ y $n(1-p)\ge5$ (algunos textos piden $np(1-p)\ge9$). Son convenciones, no teoremas. La corrección de continuidad ($|k-np|-\tfrac12$) mejora el centro pero no arregla la asimetría. En bandas de PD bajas la solución no es corregir la normal: es no usarla. El costo computacional del exacto es nulo.

### 3.4 Potencia: lo que cada banda puede ver

Con test de una cola al nivel $\alpha$, $k^*=\min\{k:P(D\ge k\mid\text{PD}_0)\le\alpha\}$. Si la PD verdadera es $\text{PD}_1=m\,\text{PD}_0$, la potencia es $\pi(m)=P(D\ge k^*\mid n,\,m\,\text{PD}_0)$. Aproximación normal, útil para diseñar (no para decidir): se rechaza con probabilidad $1-\beta$ si

$$\sqrt n\,(\text{PD}_1-\text{PD}_0)\ \ge\ z_{1-\alpha}\sqrt{\text{PD}_0(1-\text{PD}_0)}+z_{1-\beta}\sqrt{\text{PD}_1(1-\text{PD}_1)} .$$

Para B2 ($\alpha=5\%$, $\beta=20\%$) da $\text{PD}_1\approx3{,}7\%$. El exacto del notebook da 4,3% ($m_{80}=3{,}05$): la normal es optimista porque ignora la asimetría. Tabla exacta (notebook §3):

| Banda | n | PD | $k^*$ | Tamaño real | Potencia con $m=2$ | $m_{80}$ | PD detectable al 80% |
|---|---|---|---|---|---|---|---|
| A1 | 432 | 0,11% | 3 | 1,25% | 7% | 8,98 | 0,99% |
| A2 | 199 | 0,36% | 3 | 3,59% | 17% | 5,94 | 2,14% |
| B1 | 244 | 0,72% | 5 | 3,28% | 28% | 3,80 | 2,74% |
| B2 | 235 | 1,42% | 8 | 2,01% | 35% | 3,05 | 4,32% |
| C1 | 239 | 2,80% | 12 | 3,84% | 69% | 2,19 | 6,13% |
| C2 | 220 | 5,44% | 19 | 3,25% | 88% | 1,86 | 10,14% |
| D | 172 | 10,21% | 25 | 4,54% | 98% | 1,63 | 16,65% |
| E | 263 | 26,91% | 84 | 4,01% | 100% | 1,27 | 34,22% |

Lecturas: (i) si la PD de todas las bandas se **duplicara**, A1 lo detectaría el 7% de las veces; (ii) el verde de A1 es ausencia de evidencia, no evidencia de calibración; (iii) para ganar potencia en bandas buenas hay que **acumular** (varios períodos, o bandas agrupadas) o pasar a tests que usan toda la curva (HL, pendiente). Todos pagan en supuestos (§3.5 y §3.6).

### 3.5 Correlación de defaults: el modelo de un factor

**El modelo.** Vasicek (2002), el de las fórmulas IRB de Basilea: el deudor $i$ cae si su variable latente cruza un umbral,

$$A_i=\sqrt\rho\,Z+\sqrt{1-\rho}\,\varepsilon_i<c=\Phi^{-1}(\text{PD}),\qquad Z,\varepsilon_i\stackrel{iid}{\sim}N(0,1).$$

$Z$ es el factor común (el año, la macro) y $\rho$ la **correlación de activos**. Condicional a $Z=z$ los defaults son independientes con

$$p(z)=P(A_i<c\mid Z=z)=\Phi\!\left(\frac{c-\sqrt\rho\,z}{\sqrt{1-\rho}}\right),$$

y $E[p(Z)]=\text{PD}$ (el umbral se eligió para eso). La distribución no condicional del conteo es una **mezcla de binomiales**:

$$P(D\ge k)=\int_{-\infty}^{\infty}P\big(\text{Bin}(n,p(z))\ge k\big)\,\phi(z)\,dz .$$

**Cuánto se infla la varianza.** Por varianza total,

$$\operatorname{Var}(D)=E[\operatorname{Var}(D\mid Z)]+\operatorname{Var}(E[D\mid Z])=n\,E[p(1-p)]+n^2\operatorname{Var}(p(Z)).$$

Además $E[p(Z)^2]=P(A_1<c,A_2<c)=\Phi_2(c,c;\rho)$, porque dos deudores distintos comparten $Z$ y sus latentes tienen correlación $\rho$. Sustituyendo $E[p(1-p)]=\text{PD}-\Phi_2$ y $\operatorname{Var}(p)=\Phi_2-\text{PD}^2$:

$$\operatorname{Var}(D)=n\,\text{PD}(1-\text{PD})\big[1+(n-1)\rho_D\big],\qquad \rho_D=\frac{\Phi_2(c,c;\rho)-\text{PD}^2}{\text{PD}(1-\text{PD})}.$$

$\rho_D$ es la **correlación de defaults** (entre indicadores), mucho menor que $\rho$. El factor $1+(n-1)\rho_D$ es el *design effect* del muestreo por conglomerados. Con $\rho=5\%$:

| Banda | A1 | A2 | B1 | B2 | C1 | C2 | D | E | Global (n=2.004, PD 5,65%) |
|---|---|---|---|---|---|---|---|---|---|
| $\rho_D$ | 0,0008 | 0,0019 | 0,0032 | 0,0053 | 0,0083 | 0,0126 | 0,0180 | 0,0280 | 0,0129 |
| $1+(n-1)\rho_D$ | 1,34 | 1,38 | 1,79 | 2,23 | 2,98 | 3,76 | 4,08 | 8,33 | 26,8 |

El efecto crece con $n$ y con la PD. En E la varianza es 8 veces la binomial. En el **global** es 27 veces: la desviación estándar de la tasa global pasa de 0,52 pp a 2,67 pp. Con correlación el binomial global deja de medir el modelo y pasa a medir el año.

**Consecuencia para el test.** Con una cartera **perfectamente calibrada** y $\rho=5\%$, el binomial independiente de una cola al 5% rechaza (notebook §4, exacto por cuadratura y confirmado por simulación con 4.000 años):

| Banda | A1 | A2 | B1 | B2 | C1 | C2 | D | E |
|---|---|---|---|---|---|---|---|---|
| Tamaño bajo independencia | 1,3% | 3,6% | 3,3% | 2,0% | 3,8% | 3,3% | 4,5% | 4,0% |
| Rechazo real con $\rho=5\%$ | 2,7% | 5,8% | 7,5% | 7,7% | 13,1% | 15,0% | 18,1% | 25,9% |

Al menos una banda enciende el 34% de los años (vs ~23% bajo independencia con tamaños exactos). El ejemplo de Tasche (2006) se reproduce exacto: $n=1.000$, PD 1%, 19 defaults da p = 0,0069 bajo independencia y **0,1113** con $\rho=5\%$.

**Test binomial ajustado por correlación.** Se reemplaza $P(D\ge k)$ binomial por la mezcla. El notebook integra con la regla del trapecio en una grilla uniforme de $z$ (paso 0,05). Para integrandos suaves contra $\phi$ converge exponencialmente y coincide con `scipy.integrate.quad` a $10^{-7}$. Gauss-Hermite con 80 nodos, la elección «obvia», se equivoca hasta en 0,01 con $n$ grande, porque el integrando es casi un escalón en $z$. B2 pasa de 0,020 a 0,033 ($\rho=1\%$), 0,046 ($\rho=2\%$), **0,058** ($\rho=3\%$) y 0,077 ($\rho=5\%$). A2 pasa de 0,036 a 0,058 con $\rho=5\%$.

Dos sutilezas que el notebook deja a la vista:

- **El ajuste no siempre sube el p.** En A1 (1 malo, 0,48 esperados) el p **baja** de 0,378 a 0,339 con $\rho=5\%$. La mezcla engorda también $P(D=0)$ (desigualdad de Jensen: $E[(1-p(Z))^n]\ge(1-\text{PD})^n$), así que $P(D\ge1)$ cae. En C1, pegada a la media, casi no cambia. El ajuste ensancha la distribución, y eso sube la cola lejana pero no cualquier cola.
- **Qué ρ usar.** Basilea fija la correlación de activos de minoristas «otros» entre 3% y 16% según la PD, y en 15% para hipotecas y 4% para revolventes calificadas (Basilea II, párrafos 328–330; verificar en la versión consolidada). Esas cifras son **supervisoras y conservadoras para capital**; no son estimaciones de la dependencia dentro de un año. Tasche (2003) sugiere $\rho\approx5\%$ para Alemania. Lo defendible es estimar $\rho$ con la serie de tasas de default de las cosechas (método de momentos con $\operatorname{Var}(\text{DR})$, o máxima verosimilitud de Vasicek; Serie 1 · E7). Con 12 cosechas el error de estimación es grande: se reporta un **rango** de $\rho$ y se muestra si el color cambia dentro del rango.

**PIT vs TTC: qué absuelve y qué no el ajuste.** Si la PD pretende ser **TTC** (promedio del ciclo), un año malo *debe* producir más defaults que los prometidos, y el test tiene que tolerarlo: ahí el ajuste es obligatorio. Si la PD es **PIT** (condicional a las condiciones actuales, M15 §3.10), el factor común del año ya debería estar dentro de la PD y el $\rho$ relevante es el **residual**, mucho menor. Usar el $\rho$ de Basilea para absolver una PD PIT es hacer trampa: convierte cualquier deriva macro en «ruido». La calibración de Austral es un híbrido (TC del ciclo + δ), así que el rango honesto de $\rho$ para su backtesting es chico (1–3%). Con eso B2 queda entre 0,033 y 0,058, en el borde del amarillo.

**Semáforo de Tasche.** Tasche (2003) propone fijar dos cuantiles de $D$ bajo el modelo de un factor: verde si $D<c_{95\%}$, amarillo si $c_{95\%}\le D<c_{99,9\%}$, rojo si $D\ge c_{99,9\%}$. El paper aproxima los cuantiles con ajuste de granularidad o con una Beta por momentos; el notebook los calcula exactos por cuadratura. Con $\rho=5\%$: B2 pasa de $(c_{95},c_{99,9})=(8,11)$ a $(9,18)$ y A2 de $(3,5)$ a $(4,7)$. Las 8 bandas de Austral quedan verdes. La elección del 99,9% para el rojo no tiene base en la teoría de tests. Viene de la fórmula de capital, y es otra convención.

### 3.6 Hosmer-Lemeshow

**Construcción.** Se ordenan los créditos por PD y se forman $G$ grupos (deciles en la versión original de Hosmer y Lemeshow 1980). Con $O_g=\sum_{i\in g}y_i$, $n_g$, $\bar p_g$ y $E_g=n_g\bar p_g$:

$$\text{HL}=\sum_{g=1}^G\frac{(O_g-E_g)^2}{E_g(1-\bar p_g)} .$$

Es el Pearson χ² de la tabla $2\times G$ (malos y buenos por grupo), porque

$$\frac{(O-E)^2}{E}+\frac{\big((n-O)-(n-E)\big)^2}{n-E}=(O-E)^2\frac{n}{E(n-E)}=\frac{(O-E)^2}{E(1-\bar p)} .$$

`statsmodels` no trae un HL con ese nombre; su `stats.diagnostic_gen.test_chisquare_binning` es un χ² «tipo Hosmer-Lemeshow» que por defecto usa $G-2$ grados de libertad (ver M23 §3.9). Esta identidad permite verificarlo además con `scipy.stats.chisquare` sobre las $2G$ celdas, que es la «implementación alternativa» del notebook. Reproducción de la lámina 24 con PD media $=E_g/n_g$: **HL = 29,04**, grupos 3 y 5 = 24,28.

**¿$\chi^2_{G-2}$ o $\chi^2_G$?** Si las PD vienen de **afuera** (validación externa: el modelo se ajustó en DEV y se evalúa en OOT), cada término $Z_g=(O_g-E_g)/\sqrt{\operatorname{Var}(O_g)}$ es asintóticamente $N(0,1)$ e independiente de los otros grupos (los grupos no comparten créditos). La suma de $G$ normales estándar al cuadrado es $\chi^2_G$; no se consumió ningún grado de libertad. En **desarrollo** las PD se estimaron con los mismos datos. Con intercepto, las ecuaciones de verosimilitud imponen $\sum_i(y_i-p_i)=0$, lo que restringe la suma de los $O_g-E_g$. Los grupos además se forman con las PD ajustadas. Hosmer y Lemeshow (1980) mostraron por simulación que entonces la distribución se aproxima por $\chi^2_{G-2}$ (no es un conteo exacto de restricciones lineales, porque los grupos son aleatorios). Para validación externa el texto de Hosmer, Lemeshow y Sturdivant (2013) usa $G$ grados de libertad (verificar sección). En Austral OOT: $\chi^2_8$ p = 0,00031 y $\chi^2_{10}$ p = 0,0012. nikodym usa $G-2$ en todas las particiones. Es un detalle de implementación que conviene saber: en OOT exagera la evidencia.

**Por qué la χ² falla con esperados chicos.** Para un grupo con $O\sim\text{Poisson}(\lambda)$ (aproximación de una binomial con $p$ chico), el cuarto momento central es $\lambda+3\lambda^2$. Con $Z=(O-\lambda)/\sqrt\lambda$:

$$E[Z^2]=1,\qquad E[Z^4]=\frac{\lambda+3\lambda^2}{\lambda^2}=3+\frac1\lambda,\qquad \operatorname{Var}(Z^2)=2+\frac1\lambda .$$

Un $\chi^2_1$ tiene varianza 2. Con $\lambda=0{,}106$ (grupo 1 de Austral) la varianza del aporte es $2+9{,}4=11{,}4$: la media es la correcta, pero toda la masa está en 0 y hay saltos enormes. Ese grupo aporta 0,106 si no hay malos, 7,5 con uno, 33,9 con dos. Sumando los diez grupos de la lámina, $\sum 1/E_g=16{,}3$, así que $\operatorname{Var}(\text{HL})\approx36$ contra 20 de una $\chi^2_{10}$ y 16 de una $\chi^2_8$. La media real es ~10 y la cola derecha es mucho más pesada. La $\chi^2_8$ se equivoca dos veces: media 8 en vez de ~10, y cola demasiado liviana.

**El «Austral de laboratorio»** (notebook §5.1): 2.004 PD individuales que reproducen las medias de los 10 grupos, y 10.000 mundos donde la PD es verdad. El test «al 1%» con $\chi^2_8$ rechaza el **5,1%** de las veces; al 5%, el 13,5%; con $\chi^2_{10}$ al 1%, el 2,8%. El cuantil 99% real es 30,5, no 20,1. El p simulado del 29,04 bajo esas PD es 0,012, el mismo que reportó el curso.

**p-valor por simulación (bootstrap paramétrico bajo $H_0$).** Se simulan $S$ vectores $y^{(s)}_i\sim\text{Bernoulli}(p_i)$ con las PD **individuales**, se recalcula HL con los **mismos grupos y los mismos denominadores**, y $\hat p=\#\{\text{HL}^{(s)}\ge\text{HL}_{obs}\}/S$ (o $(1+\#)/(S+1)$, que nunca da 0 y es un p válido para cualquier $S$). Tres detalles:

- El error Monte Carlo es $\sqrt{\hat p(1-\hat p)/S}$: con $\hat p=0{,}012$ y $S=10.000$, 0,0011. «0,012» significa «entre ~0,010 y ~0,014» y el umbral rojo cae dentro. Si el p va a decidir un color cerca del umbral, se sube $S$ hasta que el intervalo no toque el umbral, o se reporta «en el borde».
- Con PD **homogéneas** por grupo (lo único que permite la tabla agregada de la lámina) y 200.000 réplicas el p es 0,013. Es levemente mayor que con PD individuales por lo de §3.1: la Poisson-binomial tiene menos varianza.
- El p simulado **no** corrige la dependencia entre deudores. Simula bajo independencia. Con correlación hay que simular el modelo de un factor, igual que en §3.5.

**Dependencia del agrupamiento.** HL no es un número del modelo: es un número del modelo **y** de una partición. En Austral, el mismo OOT da 29,0 con deciles (χ² p 0,0003; simulado 0,012) y **20,6 con las 8 bandas** de la master scale (χ² p 0,0084; simulado 0,019). En la cartera sintética (notebook §6), el HO del modelo da p simulados entre 0,034 (deciles) y 0,31 (50 grupos) según el agrupamiento; verde o amarillo según una decisión del analista. Por eso $G$ y la regla de corte se fijan en la política antes de mirar. Hosmer et al. (1997) documentaron además que distintos paquetes dan HL distintos con los mismos datos por el manejo de empates en los cortes.

**Con $n$ muy grande, el problema inverso.** La potencia del HL crece con $n$: en carteras de cientos de miles de créditos rechaza desviaciones económicamente irrelevantes. Paul, Pennell y Lemeshow (2013) proponen ajustar $G$ al tamaño muestral y Nattino, Pennell y Lemeshow (2020) una versión modificada para muestras grandes. En crédito masivo el HL se acompaña siempre de una medida de **tamaño del efecto**: O/E por grupo, diferencia máxima en pp, ECE (M15 §3.7).

### 3.7 Spiegelhalter (1986)

Bajo $H_0$ (independencia y $P(y_i=1)=p_i$), cada término del Brier $(y_i-p_i)^2$ vale $(1-p_i)^2$ con probabilidad $p_i$ y $p_i^2$ con probabilidad $1-p_i$. Entonces

$$E_0[(y_i-p_i)^2]=p_i(1-p_i)^2+(1-p_i)p_i^2=p_i(1-p_i),$$
$$\operatorname{Var}_0[(y_i-p_i)^2]=p_i(1-p_i)\big[(1-p_i)^2-p_i^2\big]^2=p_i(1-p_i)(1-2p_i)^2 .$$

Con $\text{BS}=\frac1n\sum(y_i-p_i)^2$, el estadístico

$$z_S=\frac{\text{BS}-\frac1n\sum p_i(1-p_i)}{\frac1n\sqrt{\sum p_i(1-p_i)(1-2p_i)^2}}=\frac{\sum_i(y_i-p_i)(1-2p_i)}{\sqrt{\sum_i(1-2p_i)^2p_i(1-p_i)}}$$

es asintóticamente $N(0,1)$. La segunda forma sale de la identidad $(y-p)^2-p(1-p)=(y-p)(1-2p)$, válida para $y\in\{0,1\}$ (se verifica caso a caso). Lecturas:

- Usa las PD **individuales**. No agrupa, así que no depende de cortes (ventaja sobre HL).
- Pondera cada residuo por $(1-2p_i)$. Con PD chicas el peso es ~1 y $z_S\approx(D-\sum p_i)/\sqrt{\sum p_i(1-p_i)}$: **en carteras de consumo Spiegelhalter es casi un test global de nivel** (la versión normal del binomial global con Poisson-binomial). Deudores con $p\approx0{,}5$ no aportan. El signo de $z_S$ puede diferir del de O/E: en el HO sintético, O/E = 1,039 y $z_S=-0{,}42$, porque el exceso de malos está en PD altas, donde el peso es menor.
- No distingue nivel de forma: un $z_S$ grande no dice qué se rompió. Se reporta junto al global y a la pendiente.
- Con PD chicas el TCL necesita muchos malos esperados; con pocos, la normal de $z_S$ tiene el mismo problema que en §3.3.

Tasche (2006) lo recomienda para PD **PIT**, porque su supuesto de independencia condicional a los scores es plausible ahí.

### 3.8 Test de Jeffreys (BCE)

Las *Instructions for reporting the validation results of internal models* del BCE (febrero de 2019, §2.5.3.1) piden, por grado y a nivel cartera, un test **de una cola** de la PD contra la tasa observada basado en la distribución $\text{Beta}(D+\tfrac12,\,N-D+\tfrac12)$. La hipótesis nula es que la PD aplicada es mayor que la verdadera (no hay subestimación). El p-valor es la cdf de esa Beta evaluada en la PD:

$$p_J=F_{\text{Beta}(D+\frac12,\,N-D+\frac12)}(\text{PD}) .$$

**De dónde sale.** Con prior de Jeffreys $\theta\sim\text{Beta}(\tfrac12,\tfrac12)$ y $D\mid\theta\sim\text{Bin}(N,\theta)$, la posterior es $\text{Beta}(D+\tfrac12,N-D+\tfrac12)$. $p_J=P(\theta\le\text{PD}\mid\text{datos})$ es la probabilidad **posterior** de que la tasa verdadera esté por debajo de la PD prometida. Si es chica, los datos dicen que la PD subestima.

**Relación con el binomial.** Por §3.2, $P(D\ge k)=I_{\text{PD}}(k,N-k+1)$ y $P(D\ge k+1)=I_{\text{PD}}(k+1,N-k)$. $I_x(a,b)$ decrece en $a$ y crece en $b$, así que

$$P(D\ge k+1)=I_{\text{PD}}(k+1,N-k)\ \le\ I_{\text{PD}}(k+\tfrac12,N-k+\tfrac12)=p_J\ \le\ I_{\text{PD}}(k,N-k+1)=P(D\ge k).$$

Jeffreys queda estrictamente entre las dos colas binomiales. Se comporta como un **mid-p**: menos conservador que el exacto. En Austral: B2 $p_J=0{,}0120$ (exacto 0,0201; mid-p 0,0135), A2 0,0152 (exacto 0,0359), B1 0,059 (exacto 0,101). Mismo color en todas, pero B1 queda a un malo del amarillo. Global: $p_J=0{,}287$.

**IC de Jeffreys** (M15 §3.8): cuantiles 2,5% y 97,5% de la misma Beta, con el límite inferior en 0 si $D=0$ (Brown, Cai y DasGupta 2001). A2: [0,43%; 3,97%], que no contiene 0,36%. B2: [1,62%; 6,32%], que no contiene 1,42%. La planilla trae ambos.

### 3.9 Pendiente de calibración (Cox 1958)

Regresión logística de $y$ sobre el logit de la PD a validar:

$$\text{logit}\,P(y_i=1)=a+b\,\text{logit}(p_i).$$

Calibración débil (M15 §3.7) equivale a $a=0,b=1$. Dos tests: LR de $H_0:(a,b)=(0,1)$ contra el modelo ajustado, con $\chi^2_2$, y Wald de $H_0:b=1$. Interpretación: $b<1$, PD demasiado extremas (sobreajuste típico de desarrollo); $b>1$, PD demasiado tímidas. El notebook implementa Newton-Raphson en numpy y lo compara con `statsmodels.GLM`. En la cartera sintética el HO del modelo pasa el global (p 0,36) pero da $\hat b=1{,}17$ con LR p = 0,005. La pendiente **verdadera** del modelo contra la PD real del generador es 1,09: el scorecard de 6 variables comprime las PD. El test encontró un defecto real que el binomial global no puede ver. Costo: supone linealidad en el logit; una curvatura en forma de S no la captura (para eso, HL o curva suavizada con IC).

### 3.10 Multiplicidad

Con $m$ tests independientes, cada uno con tamaño exacto $\alpha$ y todas las nulas ciertas, la tasa de error por familia (FWER) es

$$P(\ge1\text{ rechazo})=1-(1-\alpha)^m ,$$

que con $m=8$ y $\alpha=5\%$ es 33,7% (el «34%» de la lámina 28 corresponde a 8, no a 9 indicadores; con 9 sería 37%). Tres correcciones al número:

1. **Discreción.** Con los tamaños reales del minlike al 5% (1,3%–4,7% por banda), la FWER exacta bajo independencia es $1-\prod_b(1-\alpha_b^{\text{real}})=24{,}1\%$.
2. **Correlación.** Con $\rho=5\%$ los tests de las 8 bandas dejan de ser independientes (comparten $Z$) y cada uno se infla. El rechazo de al menos una banda (una cola) sube a 34%.
3. **El tablero no son 8 tests del mismo tipo**: mezcla binomiales, HL, PSI y CSI, con dependencias entre sí. La cuenta $1-0{,}95^9$ es un orden de magnitud, no una probabilidad.

**Correcciones.** Bonferroni: $\tilde p_b=\min(1,m\,p_b)$. Controla FWER bajo cualquier dependencia (desigualdad de Boole). **Holm (1979)**: ordena $p_{(1)}\le\dots\le p_{(m)}$ y ajusta $\tilde p_{(r)}=\max_{j\le r}\min\{1,(m-j+1)p_{(j)}\}$. También controla FWER bajo cualquier dependencia y es uniformemente más potente que Bonferroni. No hay razón para usar Bonferroni si se puede usar Holm. **Benjamini-Hochberg (1995)** controla la tasa de falsos descubrimientos (FDR), no la FWER; es válido bajo independencia o dependencia positiva. En Austral: B2 0,020 → Holm 0,161; A2 0,036 → 0,251; BH: ambas 0,144. **Ninguna banda sobrevive a la corrección.**

**¿Entonces el curso se equivocó al leer amarillo?** No, pero conviene decir qué pregunta contesta cada lectura. La corrección controla «¿hay *alguna* banda mal calibrada?» con error 5%, y la respuesta es: no se puede afirmar con esta muestra. La lectura por **patrón** contesta otra cosa: «¿hay una desviación *sistemática* en una dirección?». Un test de signos formaliza la parte más simple. Bajo $H_0$ cada banda tiene probabilidad ~½ de caer sobre sus esperados (aproximadamente: la binomial es asimétrica y con $np$ chico $P(D>np)<\tfrac12$). En Austral 6 de 8 bandas tienen más malos que los esperados, $P(\text{Bin}(8,\tfrac12)\ge6)=37/256=0{,}145$. No es concluyente. La evidencia del curso es **multi-fuente**: bandas buenas que subestiman + HL por los deciles buenos + CSI de una variable de deuda + mix que se corre a D+E + swap-in peor que su PD. Son fuentes parcialmente independientes que apuntan al mismo mecanismo, y ningún test individual la contiene. La práctica defendible:

- declarar **ex ante** qué manda el semáforo (por banda sin corrección como *alerta*, con Holm como *hallazgo*; o global + pendiente + HL como *hallazgo* y bandas como *diagnóstico*);
- tratar los amarillos sueltos como **vigilancia**, no como hallazgo;
- escalar solo cuando el patrón es coherente y tiene mecanismo, como hizo el curso.

### 3.11 La muestra de validación no puede ser la de calibración

M15 lo desarrolla. Aquí basta la aritmética del backtesting. Si $\delta$ se ajusta de modo que $\sum_i\sigma(\eta_i+\delta)=D$ en la muestra $S$, el binomial global en $S$ tiene $k=n\bar p$ exacto y $p=1$. El test no tiene información. En la cartera sintética, con $\delta=+0{,}328$ ajustado en OOT ene–mar, el global en ene–mar da p = 1,000; en abr–jun (fuera de muestra) da O/E 1,090, p global 0,066 y Jeffreys 0,032. La verdad del generador muestra que ene–mar tuvo una tasa realizada **bajo** su PD real (PD real media / PD calibrada = 1,093, la columna «O/E_verdad» del notebook): el δ heredó ese ruido y solo la muestra siguiente lo reveló. En Austral la TC que ancló δ incluye las 4 cosechas de OOT, así que el backtesting OOT del curso no es 100% independiente. La sensibilidad pre-OOT (global 0,276; A2 0,006; HL 32,0 con p 0,0096) es la versión honesta, y la clase la hizo.

---

## 4. Variantes y alternativas de industria

| Método | Qué resuelve | Costo / supuestos | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| Binomial exacto por banda (una cola) | ¿La banda subestima? | Independencia; PD homogénea; potencia baja en bandas buenas | Siempre, como base; la cola que importa al capital | CRR art. 185(b) exige comparar tasa observada vs PD por grado y analizar las desviaciones «fuera del rango esperado»; la forma del test la elige el banco |
| Binomial bilateral (minlike / doble) | Sub y sobreestimación | Igual; la definición cambia el p | Cuando sobreestimar también cuesta (pricing, rechazo de buenos) | scipy/R (minlike); planillas (doble) |
| Test normal (z) por banda | Rapidez, fórmula cerrada | Falla con $np<5$ (rojos falsos) | Solo con $np(1-p)\ge9$ | Manuales antiguos, muchas suites |
| Binomial global | Nivel de la cartera | Independencia; con correlación mide el año | Pregunta de nivel; junto con TC | Universal |
| Jeffreys (una cola) | Subestimación, sin conservadurismo excesivo | Prior de Jeffreys; es casi un mid-p | Reporte al BCE; bandas con $np$ chico | BCE, instrucciones de reporte de validación (2019), por grado y cartera |
| Binomial ajustado por correlación (Vasicek) | Rechazo excesivo en años malos | Hay que elegir $\rho$ (y justificarlo) | PD TTC; bandas grandes; test global | BCBS WP14 (2005); Tasche (2003, 2006) |
| Semáforo de Tasche (95% / 99,9%) | Umbrales con dependencia | Cuantiles de la mezcla; la elección de niveles es convención | Tableros regulatorios de PD TTC | Propuesto por Tasche (2003) en el Bundesbank |
| Semáforo extendido multi-período (Blochwitz et al.) | Acumular evidencia en el tiempo | Supone independencia entre años o la modela | Bandas buenas con pocos defaults por año | Blochwitz, Martin y Wehn (2006) |
| Hosmer-Lemeshow con $\chi^2$ | Forma de toda la curva en un número | Esperados ≥ 5; depende de $G$ y cortes | Carteras con PD altas y $n$ moderado | Clínica; suites de scorecards; nikodym (con $G-2$) |
| Hosmer-Lemeshow con p simulado | Idem, sin la aproximación | Costo computacional (trivial); error MC; sigue suponiendo independencia | PD bajas / esperados chicos (casi siempre en consumo) | Curso (clase 5) |
| Spiegelhalter | Calibración con PD individuales, sin agrupar | Normal asintótica; con PD chicas ≈ test de nivel | PD PIT; complemento del HL | Tasche (2006); clínica |
| Pendiente e intercepto (Cox) | Nivel y dispersión por separado | Linealidad en el logit | Siempre en validación de desarrollo/HO; diagnostica sobreajuste | Clínica (Van Calster et al.); cada vez más en crédito |
| Brier / ECE / O/E por grupo | Tamaño del efecto | No son tests | Siempre junto a un p (sobre todo con $n$ grande) | Universal |
| Corrección Holm / BH | Multiplicidad entre bandas | Pierde potencia (Holm) o controla otra cosa (BH) | Cuando el hallazgo es «alguna banda falla» | Estadística general; poco usado en tableros de crédito |

Notas:

- **BCBS WP14 (2005)**, *Studies on the Validation of Internal Rating Systems* (versión revisada de mayo de 2005), dedica una sección a la validación de la calibración: binomial, χ² (HL), test normal y semáforos, con la advertencia de que los tests suponen independencia, que la correlación de defaults los vuelve demasiado exigentes en años malos, y que ninguno tiene gran potencia con los tamaños usuales. No pude acceder al PDF durante la redacción para citar textual; la síntesis de Tasche (2006) sobre el mismo material es consistente con esto (ver referencias).
- **CMF (Chile).** Para modelos internos de provisiones, el Compendio de Normas Contables para bancos (capítulo B-1) exige que los bancos validen y revisen periódicamente sus modelos. Según entiendo no prescribe un test de backtesting específico; verificar con la norma vigente y con las guías de la CMF sobre metodologías estándar vs internas (Serie 1 · E6).
- **EE.UU.** SR 11-7 (Fed/OCC, 2011) instaló el *outcomes analysis* (backtesting) como pilar de la validación, sin prescribir tests. Fue reemplazada el 17-abr-2026 por SR 26-2 (OCC Bulletin 2026-13), que mantiene el análisis de resultados con un enfoque proporcional al riesgo y tampoco prescribe tests (ver M22).

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 Rojo por aproximación normal en bandas buenas.**
Síntoma: A1/A2 en rojo con 2–3 malos. Causa: $z$ con $np<1$, cola derecha subestimada por la asimetría. Detección: columna «aviso np» de la planilla; comparar con el exacto. Qué hacer: binomial exacto (o Jeffreys) siempre que $np<5$. Nunca decidir con $z$ en bandas de PD bajo ~2%.

**5.2 Rojo de HL por la tabla χ² con esperados chicos.**
Síntoma: HL «p < 0,001» en OOT con 2.000 créditos y PD de 0,05% en el primer decil (nikodym, clase 6; la χ² del curso). Causa: $\operatorname{Var}(Z_g^2)=2+1/\lambda_g$; los grupos con esperados < 1 dominan. Detección: contar grupos con $E_g<5$ (la planilla lo hace); comparar $\chi^2$ con simulado. Qué hacer: p simulado con PD individuales y $S\ge10.000$; reportar estadístico, método del p y supuesto. Si el p simulado vive en el borde, reportar «en el borde» con su error MC.

**5.3 $G-2$ grados de libertad en validación externa.**
Síntoma: la librería reporta el mismo tipo de p en DEV y en OOT. Causa: implementación genérica. Detección: recomputar $\chi^2_{G-2}$ y $\chi^2_G$; si uno reproduce el p de la librería, ya sabes cuál usa (nikodym: $G-2$). Qué hacer: $\chi^2_G$ para OOT, o directamente simulado.

**5.4 El color depende del agrupamiento.**
Síntoma: el HL cambia de color al pasar de deciles a bandas o a ventiles. Causa: HL es función de la partición. Detección: correr el HL con 2–3 agrupamientos (notebook §6: HO del modelo, p simulado 0,034–0,31). Qué hacer: fijar $G$ y la regla de corte en la política; reportar la sensibilidad como diagnóstico, no elegir el más cómodo.

**5.5 El binomial rechaza «todo» en un año malo.**
Síntoma: global y bandas grandes (D, E) en rojo el mismo trimestre en que sube el desempleo; al año siguiente vuelven a verde sin tocar el modelo. Causa: factor común; el binomial trata un shock sistemático como error del modelo. Detección: correlación de las desviaciones entre bandas (todas del mismo signo, proporcionales a la PD); comparación con la serie de tasas de cosechas. Qué hacer: si la PD es TTC, test ajustado con un $\rho$ justificado y semáforo tipo Tasche. Si es PIT, el rojo es real: la PD no siguió al ciclo y hay que recalibrar el nivel (δ), no absolver.

**5.6 Absolver con un ρ de Basilea.**
Síntoma: «con $\rho=15\%$ todo es verde». Causa: se usó una correlación supervisora pensada para capital sobre una PD que se declara PIT. Detección: preguntar qué es la PD (PIT/TTC) y de dónde sale el $\rho$. Qué hacer: $\rho$ estimado con la serie de cosechas, reportado como rango. Para PIT, $\rho$ residual.

**5.7 Verde sin potencia.**
Síntoma: A1 verde trimestre tras trimestre. Causa: 0,5 malos esperados; potencia de 7% contra una PD dos veces mayor. Detección: tabla de potencia (planilla, hoja «Potencia»). Qué hacer: no reportar el verde como evidencia; acumular períodos (con cuidado por la correlación entre años), agrupar bandas buenas para el test, o usar tests de curva completa. En el comité, decir «no evaluable con este n» es más honesto que un verde.

**5.8 Leer colores sueltos en un tablero con muchos tests.**
Síntoma: re-desarrollo propuesto por un amarillo. Causa: FWER ~24–34% por trimestre con un modelo perfecto. En la cartera sintética, la PD **verdadera** del generador enciende un amarillo por banda y un HL amarillo (p 0,039) en HO. Detección: contar cuántos amarillos se esperan por azar dada la batería. Qué hacer: jerarquía de lectura escrita ex ante; Holm para hallazgos; patrón con mecanismo para escalar.

**5.9 Calibrar y validar con la misma muestra.**
Síntoma: global p = 1,00, O/E = 1,000. Causa: δ ajustado en la muestra testeada. Detección: el O/E exacto 1 es la huella. Qué hacer: separar calibración y validación en el tiempo (M15); si el ancla usa las cosechas de validación, correr la sensibilidad sin ellas, como el curso.

**5.10 Sustituir PD individuales por la media de banda en un grupo heterogéneo.**
Síntoma: p de HL o binomial algo mayores que lo que da la simulación con PD individuales (Austral: 0,013 vs 0,012). Causa: varianza binomial ≥ Poisson-binomial. Detección: comparar $n\bar p(1-\bar p)$ con $\sum p_i(1-p_i)$ por grupo. Qué hacer: en bandas angostas es despreciable. En grupos anchos (último decil), simular con PD individuales.

**5.11 Error Monte Carlo ignorado.**
Síntoma: el color de un p simulado cambia con la semilla. Causa: $S$ chico cerca de un umbral. Detección: IC binomial del p simulado. Qué hacer: $S$ tal que el IC no toque el umbral; fijar la semilla en el contrato (como la demo: 20260908) para reproducibilidad, sin creer que la semilla elimina el error.

**5.12 Una PD de banda «esperada = 0».**
Síntoma: un test contra una tasa de referencia 0 da p = 0,0001 con el primer malo (el `ScorecardMonitoring` de optbinning en la demo de clase 5, tramo [600, 613) con tasa DEV 0). Causa: PD de referencia estimada en DEV sin suavizar. Detección: esperados = 0. Qué hacer: nunca testear contra una PD empírica cero. La master scale asigna PD calibradas positivas (M16), o se usa un piso.

---

## 6. Puente con ingeniería

El backtesting es un **job reproducible** que consume tres artefactos congelados y produce un reporte versionado. En un pipeline declarativo:

```yaml
# backtesting/austral_consumo_v3_2025q3.yaml
modelo:        {id: austral_consumo_v3, hash_artefacto: "sha256:…"}      # el scorer congelado (M21)
master_scale:  {id: ms_austral_v3, bandas: [A1, A2, B1, B2, C1, C2, D, E], pd_banda: pd_calibrada_media}
muestra:
  nombre: OOT_2025Q1
  ventana_desempeno: {meses: 12, malo: "90+"}
  excluir: [indeterminados_30_89]
  usada_para_calibrar: false            # invariante: si es true, el job falla
calibracion_referencia: {tc: 0.0542, delta: 0.1115, cosechas_ancla: ["2024-07", "2025-06"]}
tests:
  binomial_banda: {cola: superior, bilateral: minlike, alfa: [0.05, 0.01]}
  binomial_global: {bilateral: minlike}
  jeffreys: {cola: superior}
  hosmer_lemeshow: {grupos: deciles_pd, gl: G, p: simulado, S: 20000, semilla: 20260908}
  pendiente: {test: LR_a0_b1}
  correlacion: {rho_rango: [0.01, 0.03], modelo: vasicek_un_factor}
multiplicidad: {hallazgo: holm, alerta: sin_ajuste}
semaforo: {verde: ">= 0.05", amarillo: "[0.01, 0.05)", rojo: "< 0.01"}
lectura_por_patron: {test_signos: true, direccion_esperada: subestima}
```

**Contratos e invariantes (tests tipo CI).**

- *Conservación*: $\sum_b n_b = n$, $\sum_b D_b = D$, $\sum_b E_b=\sum_i p_i$ (a $10^{-9}$). Ningún crédito queda sin banda; bandas vacías se reportan como «—», **nunca** como verde (la demo lo hace).
- *Independencia de muestras*: la muestra de validación no se intersecta con la de calibración (por id y por cohorte). Si `usada_para_calibrar: true`, el job no emite semáforos de nivel.
- *Madurez*: todas las cohortes tienen 12 meses de desempeño cumplidos al corte; si no, el job falla (Serie 1 · M2, M3).
- *Rangos*: $0<\text{PD}_b<1$; esperados > 0; p ∈ [0,1].
- *Coherencia numérica*: la implementación propia coincide con la de referencia (`scipy.stats.binomtest`, `scipy.stats.beta.cdf`, `chisquare` 2×G) en una batería fija de casos (los checks del notebook son exactamente eso).
- *Reproducción del expediente*: el reporte regenera los números del informe; un diff distinto de cero bloquea la publicación (clase 6: «generar los documentos DESDE la corrida»).
- *Estabilidad del p simulado*: se reporta con su error MC y el job falla si el IC del p cruza un umbral de color con el $S$ configurado (obliga a subir $S$ antes que a publicar un color ambiguo).

**Qué se congela y qué se versiona.** Congelado: artefacto del modelo, master scale (cortes y PD por banda), δ y TC con sus cosechas, definición de malo. Versionado por corrida: muestra (hash de ids), configuración de tests, semilla, versiones de librerías, salida completa (tablas por banda, HL con aportes, p con método). Los umbrales y la jerarquía de lectura se versionan **aparte** y con fecha anterior a la corrida: son política, no parámetro del job (clase 5: «los umbrales se fijan ANTES de mirar el dato»).

**Implementación eficiente.** El p minlike de todas las $k$ de una banda se precalcula una vez (vector de $n+1$ p-valores), y la simulación de rechazo se vuelve un *lookup*. El HL simulado se vectoriza por bloques de réplicas ($S\times n$ booleanos por bloque) y es trivial en CPU para $n$ de decenas de miles. El test ajustado por correlación usa una grilla fija de $z$ que se comparte entre bandas.

---

## 7. Numpy desde cero vs librerías

| Cálculo | Numpy (notebook) | Librería | Diferencias / convenciones | Producción |
|---|---|---|---|---|
| pmf binomial | log-combinatorio por suma acumulada de logs, `exp` | `scipy.stats.binom.pmf` | Idénticas a $10^{-12}$; scipy usa rutinas más estables en colas extremas ($<10^{-300}$ scipy da 0, numpy un subnormal) | scipy |
| p una cola / minlike | suma de pmf; minlike con tolerancia $1+10^{-7}$ | `scipy.stats.binomtest(k, n, p, alternative=…)` | scipy «two-sided» = minlike, **no** doble cola; R `binom.test` igual | scipy, declarando el método |
| p bilateral doble | `2·min(colas)` | no existe como opción en scipy | Puede duplicar el p de minlike | Solo si la política lo define |
| Crítico y potencia | cola acumulada invertida, `argmax` | `binom.ppf(1−α)+1`, `binom.sf` | Idénticos | scipy |
| Mezcla Vasicek | trapecio en grilla uniforme de $z$ | `scipy.integrate.quad` | Coinciden a $10^{-7}$; Gauss-Hermite 80 nodos falla hasta 0,01 con $n$ grande | grilla (vectorizable, determinista) |
| HL | agregados con `bincount` | `scipy.stats.chisquare` sobre tabla $2\times G$ | Idénticos (identidad algebraica); `statsmodels` lo trae como `diagnostic_gen.test_chisquare_binning` (df = $G-2$ por defecto, M23); `chisquare` exige que $\sum O=\sum E$ | propia + test de paridad |
| p HL simulado | Bernoulli por bloques | — | Error MC; semilla fija | propia |
| Spiegelhalter | forma $(y-p)(1-2p)$ | forma Brier con `sklearn.metrics.brier_score_loss` | Idénticos (identidad) | cualquiera |
| Jeffreys | `scipy.special.betainc(k+½, n−k+½, PD)` | `scipy.stats.beta.cdf` | Idénticos; planilla: `BETA.DIST` | cualquiera |
| IC Jeffreys | — | `stats.beta.ppf`; `statsmodels.proportion_confint(method="jeffreys")` | statsmodels no fuerza 0 cuando $D=0$ (M15) | ajustar el borde |
| Pendiente | Newton-Raphson 2×2 | `statsmodels.GLM(Binomial)` | Idénticos a $10^{-7}$ | statsmodels |
| Bonferroni/Holm/BH | `argsort` + `maximum/minimum.accumulate` | `statsmodels.stats.multitest.multipletests` | Idénticos | statsmodels |
| Planilla | `BINOM.DIST`, `BETA.DIST`, `BETA.INV`, `CHISQ.DIST.RT`, `NORM.S.DIST`, `BINOM.INV` | — | Bilateral = doble (no minlike); en el archivo se guardan con prefijo `_xlfn.` (estándar OOXML) | para comités y chequeo manual |

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral OOT, releído

| Banda | Esperados | Malos | Una cola sup | Minlike (curso) | Doble | Normal | Jeffreys | Holm (minlike) | Ajustado $\rho=5\%$ |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 0,48 | 1 | 0,378 | 0,378 | 0,757 | 0,446 | 0,187 | 1,000 | 0,339 |
| A2 | 0,72 | 3 | 0,036 🟡 | 0,036 🟡 | 0,072 | **0,007 🔴** | 0,015 🟡 | 0,251 | 0,058 |
| B1 | 1,76 | 4 | 0,101 | 0,101 | 0,202 | 0,089 | 0,059 | 0,607 | 0,141 |
| B2 | 3,34 | 8 | 0,020 🟡 | 0,020 🟡 | 0,040 🟡 | 0,010 🟡 | 0,012 🟡 | 0,161 | 0,077 |
| C1 | 6,69 | 8 | 0,355 | 0,554 | 0,710 | 0,608 | 0,288 | 1,000 | 0,353 |
| C2 | 11,97 | 9 | 0,850 | 0,458 | 0,476 | 0,378 | 0,809 | 1,000 | 0,663 |
| D | 17,56 | 22 | 0,160 | 0,257 | 0,320 | 0,264 | 0,133 | 1,000 | 0,276 |
| E | 70,77 | 64 | 0,844 | 0,367 | 0,385 | 0,346 | 0,827 | 1,000 | 0,612 |
| Global | 113,28 | 119 | 0,303 | 0,562 | 0,606 | 0,580 | 0,287 | — | — |

(Holm 1,000 = ajuste topado en 1. Normal B2 = 0,0101: amarillo por 0,0001.)

Lectura de validador: (i) los dos amarillos son robustos a la definición del p **salvo** en la convención «doble» (A2 pasa a verde) y a la corrección por multiplicidad (ambos pasan a verde); (ii) con correlación modesta B2 deja de ser amarillo a partir de $\rho\approx3\%$; (iii) el global está lejos de cualquier umbral con cualquier método; (iv) HL sobre deciles 29,0 (simulado 0,012) y sobre bandas 20,6 (simulado 0,019): amarillo con ambas particiones, rojo solo con la χ². El diagnóstico del curso (deriva incipiente de nivel en tramos buenos; vigilancia, no re-desarrollo) sobrevive a todas las variantes. Lo que **no** sobrevive es cualquier lectura de un color individual como hallazgo.

### 8.2 nikodym, el HL que «falla»

nikodym reporta HL OOT = 49,33, p ≈ 5·10⁻⁸ con $G-2=8$ gl, sobre deciles cuyos primeros grupos tienen PD media 0,08%–0,36% (deciles 10, 9 y 8 de su tabla: 0, 0 y 5 malos observados). Con 200 créditos por decil son 0,16–0,72 malos esperados. El octavo decil de nikodym (PD media 0,36%, 5 malos en 200) aporta por sí solo $(5-0{,}72)^2/(0{,}72\cdot0{,}9964)\approx25{,}5$, y el cuarto (PD 3,44%, 16 malos donde esperaba 6,9) otros 12,4. Recalculado desde la tabla de deciles publicada (PD media por decil, que es lo único disponible), el HL da 47,9 (nikodym: 49,3 con PD individuales): $\chi^2_8$ p = 1·10⁻⁷, $\chi^2_{10}$ p = 6·10⁻⁷, y **p simulado ≈ 0,001** (PD homogéneas por decil, 400.000 réplicas; con PD individuales sería algo menor, §3.1). Aquí, a diferencia de la demo de la clase 5, **el rojo sobrevive a la corrección**: la magnitud «p ≈ 5·10⁻⁸» la infla la tabla χ², pero un p simulado de ~0,001 sigue siendo rojo. La lectura de la clase 6 («falla por tramos; no pasar a producción sin diagnóstico») es correcta. El diagnóstico debe nombrar los dos deciles que la producen: uno bueno que subestima (como en la demo) y uno intermedio. La lección general: el p simulado no está para absolver, está para medir bien.

### 8.3 Cartera sintética con verdad conocida (notebook §6–§7)

Scorecard de 6 variables sobre `generar_cartera()`; tasas: DEV 11,3%, HO 11,8%, OOT 15,4% (deterioro plantado de +0,35 en log-odds en 2025).

| Caso | O/E | p global | Bandas 🟡/🔴 | HL | p HL sim | $\hat b$ | p LR(a=0,b=1) | O/E verdad | Pendiente verdad |
|---|---|---|---|---|---|---|---|---|---|
| HO / modelo | 1,039 | 0,361 | 0/1 | 19,0 | 0,043 | 1,166 | 0,005 | 1,044 | 1,093 |
| HO / oráculo | 0,995 | 0,925 | 1/0 | 19,8 | 0,039 | 1,057 | 0,465 | 1 | 1 |
| OOT / modelo | 1,343 | < 10⁻¹⁵ | 2/4 | 91,0 | 0,000 | 1,078 | < 10⁻¹⁵ | 1,403 | 1,108 |
| OOT / oráculo | 0,957 | 0,195 | 0/0 | 8,0 | 0,604 | 0,992 | 0,329 | 1 | 1 |
| OOT ene–mar, δ ahí | 1,000 | 1,000 | 1/0 | 10,7 | 0,387 | 1,083 | 0,461 | 1,093 | 1,106 |
| OOT abr–jun, δ de ene–mar | 1,090 | 0,066 | 1/0 | 14,5 | 0,114 | 1,072 | 0,071 | 1,094 | 1,109 |

Cuatro lecciones con verdad conocida: (1) el deterioro de nivel se detecta con todo; (2) la **verdad misma** enciende amarillos (HO/oráculo: HL p 0,039 y una banda amarilla); (3) el defecto de forma real del modelo (pendiente verdadera 1,09) lo ve la prueba de pendiente y casi no lo ve el global; (4) calibrar y validar en la misma muestra da O/E = 1 y p = 1 sin información, mientras la muestra siguiente revela el sesgo que el δ heredó.

### 8.4 Crédito de motos: bandas chicas y un solo año

Ilustración con números supuestos (no son datos de Galgo). Una financiera de motos con cohortes trimestrales, banda intermedia con PD calibrada 4%, 180 créditos por trimestre y 13 malos a 12 meses: una cola p = 0,030 (🟡), minlike 0,035. Aquí la cola izquierda **sí** entra ($np=7{,}2$), y por eso el bilateral es mayor. La potencia de esa banda al 5% contra una PD real 1,5 veces mayor (6%) es 29%. Con 2 veces, 69%. Si se juntan los cuatro trimestres del **mismo año** (720 créditos, 52 malos, 7,2%), el binomial independiente da p = 4·10⁻⁵. Con $\rho=5\%$ da 0,083. Los cuatro trimestres comparten el año: juntar cohortes del mismo año multiplica la precisión binomial, pero no la precisión frente al factor común. En motos el factor común es fuerte y concreto (desempleo de repartidores y conductores de aplicaciones, precio del combustible, estacionalidad de ventas). Ganar potencia de verdad pide **años** independientes. Mientras tanto, lo correcto es declarar el backtesting de bandas intermedias como de baja potencia y apoyarse en el patrón (signos, pendiente) y en indicadores adelantados (roll rates tempranos, Serie 1 · M3).

---

## 9. Preguntas de comité

**1. «El HL dio p < 0,001 en OOT. ¿Rechazamos el modelo?»**
No con ese número. El p viene de una $\chi^2$ aplicada a grupos que esperan 0,1–0,7 malos, donde la aproximación sobrerrechaza (en nuestro laboratorio, el test «al 1%» rechaza el 5% de las veces bajo $H_0$) y con $G-2$ gl en una validación externa. Con p simulado el mismo dato da 0,012: amarillo en el borde. La señal (dos deciles buenos que subestiman) es real y se vigila; la magnitud «p < 0,001» es un artefacto.

**2. «¿Por qué el curso dice bilateral y la lámina explica una cola?»**
Porque en B2 coinciden: con 3,3 esperados, $P(D=0)=0{,}035>P(D=8)=0{,}013$, así que la definición bilateral de scipy/R (minlike) no suma nada de la cola izquierda. Con la convención «doble» sería 0,040. La política debe fijar cuál se usa, y para capital la relevante es la de una cola superior.

**3. «A1 lleva cuatro trimestres en verde. ¿Está bien calibrada?»**
No lo sabemos. Con 0,5 malos esperados, el test tiene 7% de potencia contra una PD real del doble y necesita una PD ~9 veces mayor para detectar con 80%. El verde es «no evaluable». Para opinar sobre A1 hay que acumular años o testearla agrupada con A2.

**4. «Si el año fue malo, ¿no es injusto culpar al modelo?»**
Depende de lo que la PD promete. Si es TTC, sí: el test debe permitir la variación del ciclo, y se usa un binomial ajustado por correlación (Vasicek) o el semáforo de Tasche con un $\rho$ justificado. Si es PIT, la PD debió incorporar el año; un rojo en año malo es precisamente el hallazgo. Lo que no se acepta es elegir el $\rho$ después de ver el resultado.

**5. «Dos bandas amarillas de ocho. ¿Es señal o azar?»**
Por azar esperaríamos al menos un amarillo en ~24% de los trimestres (tamaños exactos, independencia) y más con correlación. Con Holm ninguna sobrevive (B2 0,16). La señal está en el patrón: ambas son bandas buenas, ambas subestiman, y coinciden con el HL, el CSI de deuda, el mix y el swap-in. Eso justifica vigilancia reforzada y recalibración programada, no re-desarrollo.

**6. «El global da 0,56 y el HL amarillo. ¿Se contradicen?»**
No. El global mira el nivel medio (119 vs 113 malos); el HL suma desviaciones por tramo, que se compensan en la media. Es la firma de un problema de **forma** (o de nivel solo en un segmento), que un δ global no arregla: el δ mueve todas las bandas juntas.

**7. «¿Qué test pide el BCE?»**
Las instrucciones de reporte de validación (2019) piden un test de Jeffreys de una cola por grado y a nivel cartera: p = cdf de $\text{Beta}(D+\tfrac12,N-D+\tfrac12)$ en la PD. Es algo menos conservador que el binomial exacto (queda entre $P(D\ge k+1)$ y $P(D\ge k)$). En Austral no cambia ningún color, pero B1 queda cerca del amarillo (0,059).

**8. «¿Podemos validar la calibración en la misma muestra con que se calibró?»**
Para el nivel, no: el binomial global da p = 1 por construcción. Se valida en una muestra posterior y madura. Si el ancla usa las cosechas de validación (como la TC de Austral), se corre la sensibilidad sin ellas (global 0,276, A2 0,006, HL 32,0 con p 0,0096) y se reporta.

---

## 10. Ejercicios

**Ejercicio 1 (a mano).** Banda A2: $n=199$, PD 0,36%, 3 malos. Calcula $P(D\le2)$ y el p de una cola. Verifica si la definición minlike incluye algún punto de la cola izquierda.

<details><summary>Solución</summary>

$\lambda\approx np=0{,}716$. Exacto: $P(0)=(0{,}9964)^{199}=e^{199\ln0{,}9964}=e^{-0{,}7177}=0{,}4879$; $P(1)=199\cdot0{,}0036\cdot(0{,}9964)^{198}=0{,}7164\cdot0{,}4897=0{,}3508$; $P(2)=\binom{199}{2}0{,}0036^2(0{,}9964)^{197}=19.701\cdot1{,}296\cdot10^{-5}\cdot0{,}4915=0{,}1255$. Suma 0,9642 ⇒ $P(D\ge3)=0{,}0358$ (notebook: 0,0359; la diferencia es redondeo). Minlike: $P(3)\approx0{,}0298$ y todos los puntos de la izquierda (0, 1, 2) tienen probabilidad mayor; no entra ninguno. Bilateral minlike = 0,0359.
</details>

**Ejercicio 2 (derivación).** Demuestra que $\frac{d}{dp}P(D\ge k)=n\binom{n-1}{k-1}p^{k-1}(1-p)^{n-k}$ y deduce $P(D\ge k)=I_p(k,n-k+1)$.

<details><summary>Solución</summary>

$\frac{d}{dp}\binom{n}{j}p^j(1-p)^{n-j}=\binom{n}{j}[jp^{j-1}(1-p)^{n-j}-(n-j)p^j(1-p)^{n-j-1}]$. Usando $\binom{n}{j}j=n\binom{n-1}{j-1}$ y $\binom{n}{j}(n-j)=n\binom{n-1}{j}$, el término $j$ es $n[b_{j-1}-b_j]$ con $b_j=\binom{n-1}{j}p^j(1-p)^{n-1-j}$ ($b_{n}=0$). Sumando de $j=k$ a $n$, telescopa a $n\,b_{k-1}$. Como $n\binom{n-1}{k-1}=1/B(k,n-k+1)$ y $P(D\ge k)|_{p=0}=0$ para $k\ge1$, integrando: $P(D\ge k)=\int_0^p t^{k-1}(1-t)^{n-k}dt/B(k,n-k+1)=I_p(k,n-k+1)$.
</details>

**Ejercicio 3 (a mano).** Varianza de la tasa de default de la banda E ($n=263$, PD 26,91%) con $\rho=5\%$, sabiendo que $\rho_D=0{,}0280$. ¿Cuánto vale la desviación estándar de la tasa observada con y sin correlación?

<details><summary>Solución</summary>

Sin correlación: $\sqrt{0{,}2691\cdot0{,}7309/263}=\sqrt{0{,}000748}=2{,}73$ pp. Design effect $1+262\cdot0{,}0280=8{,}33$ ⇒ $\text{DE}=2{,}73\cdot\sqrt{8{,}33}=7{,}89$ pp. La tasa de E puede estar ±8 pp de su PD en un año normal sin que el modelo esté mal en promedio del ciclo. Observado: 24,33% vs 26,91% (−2,6 pp).
</details>

**Ejercicio 4 (HL con esperados chicos).** Grupo con $n=201$, $\bar p=0{,}000527$. Calcula el aporte al HL si se observan 0, 1 y 2 malos, y la probabilidad de cada caso bajo $H_0$ (Poisson). ¿Cuál es la probabilidad de que este grupo **solo** aporte más que el cuantil 99% de una $\chi^2_8$ (20,09)?

<details><summary>Solución</summary>

$E=0{,}106$, denominador $0{,}106\cdot0{,}99947=0{,}10594$. Aportes: 0 → $0{,}0112/0{,}10594=0{,}106$; 1 → $0{,}7992/0{,}10594=7{,}54$; 2 → $3{,}587/0{,}10594=33{,}9$. Poisson(0,106): $P(0)=0{,}899$, $P(1)=0{,}0953$, $P(\ge2)=1-0{,}899-0{,}0953=0{,}0052$. Con probabilidad 0,5% este grupo solo supera 20,09. Sumado a los otros grupos chicos, es la razón por la que el test $\chi^2_8$ «al 1%» rechaza ~5% de las veces.
</details>

**Ejercicio 5 (Jeffreys).** Demuestra que $P(D\ge k+1)\le p_J\le P(D\ge k)$ y calcula $p_J$ para el global de Austral (119 de 2.004, PD 5,6527%) con la aproximación normal de la Beta.

<details><summary>Solución</summary>

Por el ejercicio 2 las colas son $I_p(k+1,n-k)$ e $I_p(k,n-k+1)$, y $I_x(a,b)$ decrece en $a$ y crece en $b$ (§3.8). Numérico: Beta(119,5; 1.885,5) tiene media $119{,}5/2005=0{,}05960$ y DE $\sqrt{0{,}0596\cdot0{,}9404/2006}=0{,}005285$. $z=(0{,}056527-0{,}05960)/0{,}005285=-0{,}582$ ⇒ $p_J\approx\Phi(-0{,}582)=0{,}280$ (exacto 0,287; la Beta es levemente asimétrica). Verde.
</details>

**Ejercicio 6 (multiplicidad).** Con los p minlike de Austral, aplica Holm a mano y di cuántas bandas quedan amarillas. ¿Y con BH?

<details><summary>Solución</summary>

Ordenados: 0,0201 (B2), 0,0359 (A2), 0,1012 (B1), 0,2568 (D), 0,3667 (E), 0,3784 (A1), 0,4578 (C2), 0,5539 (C1). Holm: $8\cdot0{,}0201=0{,}161$; $\max(0{,}161,7\cdot0{,}0359=0{,}251)=0{,}251$; $6\cdot0{,}1012=0{,}607$; el resto, ≥ 0,607. Ninguna < 0,05. BH: $0{,}0201\cdot8/1=0{,}161$, $0{,}0359\cdot8/2=0{,}144$ ⇒ monotonizando desde arriba, ambas 0,144. Ninguna. Cero amarillas en los dos casos.
</details>

**Ejercicio 7 (diseño).** Escribe la regla de semáforo de backtesting por banda para una PD declarada PIT de un producto de motos con cohortes trimestrales: test, cola, tratamiento de correlación, multiplicidad y qué se hace ante un amarillo aislado vs un patrón.

<details><summary>Solución</summary>

Una propuesta defendible: binomial exacto de una cola superior por banda (y Jeffreys si se reporta a un supervisor que lo pida); bilateral minlike solo como diagnóstico de sobreestimación. Por ser PIT, sin ajuste por el $\rho$ de Basilea: sensibilidad con $\rho$ residual estimado en la serie de cosechas trimestrales (p. ej. 0,5–2%) y se reporta si cambia el color. Hallazgo = p Holm < 0,05 en alguna banda o pendiente con LR p < 0,01 o global p < 0,01; alerta = amarillo sin ajuste. Amarillo aislado: vigilancia al trimestre siguiente, sin acción. Patrón (≥ 3 bandas en la misma dirección, test de signos o pendiente significativa, coherente con PSI/CSI o roll rates): recalibración de δ programada con muestra posterior y madura, documentada. Umbrales y regla firmados antes del primer corte.
</details>

**Ejercicio 8 (código).** Con el notebook: construye un vector de PD que reproduzca los deciles OOT de nikodym (clase 6: PD medias 0,0008 … 0,3496, 200–201 créditos por decil, malos 0, 0, 5, 0, 3, 2, 16, 11, 28, 54 del decil 10 al 1) y calcula HL, $\chi^2_8$, $\chi^2_{10}$ y p simulado.

<details><summary>Solución</summary>

Pasos: `austral_hl` con esas columnas (ordenadas de PD baja a alta), `pd_lab` con el mismo generador log-uniforme por grupo, `hl_individual` y `hl_p_simulado` con S = 20.000. Con PD medias por decil (en vez de la PD de cada crédito) el HL no reproduce exacto el 49,3 de nikodym. Desde los agregados: HL = 47,9 (aportes 25,5 en el decil de PD 0,36% y 12,4 en el de 3,44%), $\chi^2_8$ p = 1,0·10⁻⁷, $\chi^2_{10}$ p = 6,4·10⁻⁷, p simulado con PD homogéneas ≈ 0,0009. Con el generador log-uniforme el p simulado queda en el mismo orden (algo menor). Moraleja: el p simulado no siempre absuelve. Aquí el rojo es real y la corrección solo cambia su magnitud en cuatro órdenes. (Sin las PD individuales de nikodym el número exacto no es reproducible.)
</details>

**Ejercicio 9 (derivación).** Muestra que con PD homogénea $p$ en un grupo el término del HL tiene esperanza exacta 1 bajo $H_0$, y que con PD heterogéneas tiene esperanza $<1$.

<details><summary>Solución</summary>

$E[(O-E)^2]=\operatorname{Var}(O)$. Homogénea: $\operatorname{Var}(O)=np(1-p)=E(1-\bar p)$ ⇒ esperanza 1. Heterogénea: $\operatorname{Var}(O)=\sum p_i(1-p_i)=n\bar p(1-\bar p)-\sum(p_i-\bar p)^2$ (§3.1) ⇒ esperanza $1-\sum(p_i-\bar p)^2/[n\bar p(1-\bar p)]<1$. El HL con PD heterogéneas es conservador en media. Es el mismo $\kappa$ de Tjur de M15 §3.3, calculado dentro del grupo.
</details>

**Ejercicio 10 (código, correlación).** En el notebook, lleva ρ a 0,10 y mira la tasa de rechazo del global. Luego estima ρ por momentos con las 12 tasas de cosecha de Austral (3,98%–6,97%; TC 5,42%): $\widehat{\operatorname{Var}}(\text{DR})\approx\text{PD}(1-\text{PD})/\bar n+(\Phi_2(c,c;\rho)-\text{PD}^2)$. ¿Qué te dice el resultado?

<details><summary>Solución</summary>

Con $\rho=0{,}10$ el binomial global de una cola al 5% rechaza ~32% de los años (con $\rho=0{,}05$, ~31%; con $\rho=0{,}03$, ~30%). Con $n$ grande la tasa se satura en $P(p(Z)>\text{PD})$, que para $\rho=0{,}10$ y PD 5,65% vale $\Phi\big(c(1-\sqrt{1-\rho})/\sqrt\rho\big)=\Phi(-0{,}257)\approx0{,}40$: el test global pasa a responder «¿el año fue peor que la mediana?». Para estimar: si la DE entre cosechas es ~0,9 pp y el $\bar n$ mensual ~500, la parte binomial es $0{,}0542\cdot0{,}9458/500=0{,}000103$ (DE 1,0 pp). La varianza observada ($\approx0{,}000081$) **no supera** la binomial, así que $\hat\rho\approx0$. Con cosechas mensuales chicas el ruido binomial tapa la correlación y $\rho$ no es identificable con 12 puntos. Hace falta una serie larga de tasas anuales, o benchmarks externos. Hay que reportarlo así, con rango, y no inventar un $\rho$ preciso. (Los números de DE y $\bar n$ son supuestos; recalcula con los de tu tabla de cosechas.)
</details>

---

## 11. Referencias

- **Hosmer, D. W. & Lemeshow, S. (1980).** «Goodness of fit tests for the multiple logistic regression model». *Communications in Statistics – Theory and Methods* 9(10), 1043–1069. El test original y la aproximación $\chi^2_{G-2}$ por simulación.
- **Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X. (2013).** *Applied Logistic Regression*, 3ª ed. Wiley. Capítulo de evaluación del ajuste: construcción del HL, empates, y validación externa (verificar sección para los $G$ gl).
- **Hosmer, D. W., Hosmer, T., Le Cessie, S. & Lemeshow, S. (1997).** «A comparison of goodness-of-fit tests for the logistic regression model». *Statistics in Medicine* 16(9), 965–980. Por qué distintos paquetes dan HL distintos y alternativas al HL.
- **Paul, P., Pennell, M. L. & Lemeshow, S. (2013).** «Standardizing the power of the Hosmer–Lemeshow goodness of fit test in large data sets». *Statistics in Medicine* 32(1), 67–80. HL en muestras grandes: elegir $G$ según $n$.
- **Nattino, G., Pennell, M. L. & Lemeshow, S. (2020).** «Assessing the goodness of fit of logistic regression models in large samples: A modification of the Hosmer-Lemeshow test». *Biometrics* 76(2), 549–560 (verificar páginas). Versión para $n$ grande.
- **Spiegelhalter, D. J. (1986).** «Probabilistic prediction in patient management and clinical trials». *Statistics in Medicine* 5(5), 421–433. El $z$ del Brier.
- **Cox, D. R. (1958).** «Two further applications of a model for binary regression». *Biometrika* 45, 562–565. Intercepto y pendiente de calibración.
- **Vasicek, O. (2002).** «The distribution of loan portfolio value». *Risk* 15(12), 160–162. El modelo de un factor detrás de la mezcla binomial y de las fórmulas IRB.
- **Tasche, D. (2003).** «A traffic lights approach to PD validation». Deutsche Bundesbank; arXiv:cond-mat/0305038. Semáforo con cuantiles 95%/99,9% bajo el modelo de un factor; ajuste de granularidad y Beta por momentos; sugiere $\rho\approx5\%$ para Alemania.
- **Tasche, D. (2006).** «Validation of internal rating systems and PD estimates». arXiv:physics/0606071 (publicado luego en *The Analytics of Risk Model Validation*, Christodoulakis & Satchell eds., Academic Press, 2008; verificar). Síntesis de binomial, HL, Spiegelhalter y Brier con sus supuestos; el ejemplo 1.000 / 1% / 19 defaults (0,7% vs 11,1%) que el notebook reproduce.
- **Basel Committee on Banking Supervision (2005).** *Studies on the Validation of Internal Rating Systems* (Working Paper 14, revisado mayo 2005). La referencia regulatoria clásica sobre validación de discriminación y calibración; incluye binomial, χ², test normal y semáforos, y discute la correlación.
- **Blochwitz, S., Martin, M. R. W. & Wehn, C. S. (2006).** «Statistical Approaches to PD Validation». En Engelmann, B. & Rauhmeier, R. (eds.), *The Basel II Risk Parameters*, Springer, 289–306. Tests multi-período y semáforo extendido.
- **European Central Bank (2019).** *Instructions for reporting the validation results of internal models – IRB Pillar I models for credit risk* (febrero 2019), §2.5.3.1. El test de Jeffreys de una cola por grado y cartera.
- **Reglamento (UE) 575/2013 (CRR), art. 185(b).** Obligación de comparar tasas observadas con PD por grado y analizar las desviaciones fuera del rango esperado.
- **Board of Governors of the Federal Reserve System & OCC (2011).** *SR 11-7: Supervisory Guidance on Model Risk Management*. El backtesting como *outcomes analysis* dentro de la validación; origen del vocabulario, reemplazada en abril de 2026 por SR 26-2 (ver M22).
- **Holm, S. (1979).** «A simple sequentially rejective multiple test procedure». *Scandinavian Journal of Statistics* 6(2), 65–70. Corrección escalonada que domina a Bonferroni.
- **Benjamini, Y. & Hochberg, Y. (1995).** «Controlling the false discovery rate: a practical and powerful approach to multiple testing». *JRSS B* 57(1), 289–300. FDR.
- **Brown, L., Cai, T. & DasGupta, A. (2001).** «Interval Estimation for a Binomial Proportion». *Statistical Science* 16(2), 101–133. Jeffreys y Wilson frente a Wald; el ajuste en $D=0$.
- **Van Calster, B. et al. (2016; 2019)**, ver M15. Jerarquía de calibración y la pendiente como prueba estándar.
- **Serie 1 · E2** (PIT/TTC), **E3** (bootstrap en carteras chicas), **E6** (regulación), **E7** (eventos extraordinarios y correlación macro); **Serie 2 · M15** (calibración, Wilson/Jeffreys), **M16** (master scale), **M20** (tablero y gatillos), **M22** (gobierno).

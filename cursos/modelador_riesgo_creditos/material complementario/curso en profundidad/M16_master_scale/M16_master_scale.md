# M16 · Master scale

> **Ficha.** Profundiza: clase 4 v1 (parte 2, «La master scale», láminas 21–22) y v21 (láminas 34–36); clase 5 (nivelación, láminas 5–6, y backtest por banda de la lámina 22); demo `demo_c4_austral_v2.ipynb`, sección 6. · Prerrequisitos: Serie 1 · E2 (PIT vs TTC), E3 (intervalos para carteras chicas), E6 (regulación); Serie 2 · M13 (scaling PDO), M15 (calibración). Prepara M17 (cutoff y estrategia), M19 (backtesting) y M20 (monitoreo). · Archivos: `M16_master_scale.md` (este documento), `M16_master_scale.py` (notebook Marimo), `M16_master_scale.xlsx` (calculadora de escala y hoja de validación). · Tiempo estimado: 3 h de lectura + 2 h de notebook y ejercicios.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**La definición.** «Nadie gobierna 10.000 PD distintas. Se gobiernan 8 bandas con nombre.» Una master scale es una partición del score en bandas de riesgo con PD de referencia: el idioma común entre modelo y negocio. El comité discute «banda C1», no «PD 3,7%»; pricing, cupos, provisiones y monitoreo se definen por banda. En la v21: «La banda ayuda a decidir y comunicar. La PD sigue siendo el número que mide el riesgo.»

**La construcción.** Con PDO = 20, una banda de 20 puntos duplica las odds exactamente: «la escala sale del scaling de la clase 3, gratis». Ancla: 600 puntos = odds 50:1 = PD 1,96%. Ocho bandas, de A1 (≥ 660, odds 400:1 en el piso) a E (< 540). La demo asigna con `pd.cut(..., right=False)`: un score exactamente en un corte cae en la banda superior, coherente con `score >= cutoff`.

**Los requisitos.** Monótona (la tasa observada escala banda a banda, sin inversiones); sin bandas vacías ni concentradas; estable en el tiempo (PSI del score, clase 5). En la práctica las instituciones definen 7–10 bandas «maestras» corporativas y cada modelo se mapea a ellas. «La validación de la escala no es teórica: se mira la tasa observada por banda.»

**La tabla de Banco Austral** (DEV+HO+OOT, 6.723 créditos; score calibrado TTC):

| Banda | Score | Odds en el piso | n | PD calibrada media | Tasa observada | % modelación | % TTD |
|---|---|---|---|---|---|---|---|
| A1 | ≥ 660 | 400:1 | 1.446 | 0,11% | 0,14% | 21,5% | 18,4% |
| A2 | 640–660 | 200:1 | 655 | 0,36% | 0,61% | 9,7% | 8,9% |
| B1 | 620–640 | 100:1 | 774 | 0,72% | 1,03% | 11,5% | 10,8% |
| B2 | 600–620 | 50:1 | 829 | 1,41% | 2,05% | 12,3% | 12,2% |
| C1 | 580–600 | 25:1 | 810 | 2,79% | 3,70% | 12,0% | 12,3% |
| C2 | 560–580 | 12,5:1 | 744 | 5,43% | 4,44% | 11,1% | 12,0% |
| D | 540–560 | 6,25:1 | 628 | 10,16% | 9,55% | 9,3% | 10,9% |
| E | < 540 | — | 837 | 26,75% | 24,85% | 12,4% | 14,5% |

Lectura del curso: monótona de punta a punta (0,14% → 24,85%), sin bandas vacías, ninguna banda sobre 22%; la bandeja TTD pesa más en D y E, «primer aviso de deriva». La clase 5 retoma la escala para el PSI del score (0,013 DEV → TTD sobre estas 8 bandas) y para el backtest binomial por banda en OOT: B2 (n = 235, PD 1,42%, 8 malos, 3,3 esperados, p = 0,020) y A2 quedan en amarillo; «las que fallan son bandas buenas y ambas subestiman».

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **Un solo diseño.** La escala PDO se presentó como «gratis». No se discutieron las alternativas (límites geométricos en PD tipo escala de rating, cuantiles, optimización) ni qué pasa cuando el rango del score no alcanza para 8 bandas de 20 puntos.
2. **La PD de la banda.** Se usó la media de PD calibradas sin discutir la alternativa del punto medio geométrico, la tasa observada suavizada ni qué hacer con las bandas abiertas (A1 y E no tienen punto medio).
3. **Los requisitos quedaron cualitativos.** «Sin bandas concentradas»: ¿medido cómo? (HHI, índice del BCE). «Monótona»: ¿con qué test, y con qué potencia? «Sin bandas vacías»: ¿cuántos malos mínimos para que el backtest por banda signifique algo?
4. **Separabilidad.** Monótona no implica que bandas adyacentes sean estadísticamente distintas. En Austral, C2 vs C1 (4,44% vs 3,70%) tiene p = 0,23 (§8.1).
5. **Granularidad vs estabilidad.** «7–10 bandas» es una convención de industria que no se justificó.
6. **Migración, recalibración y mapeo.** La matriz de migración, el efecto de recalibrar (las bandas se mueven δ·factor) y el mapeo de varios modelos a una escala corporativa se mencionaron («cada modelo se mapea») sin mecánica.
7. **Requisitos regulatorios.** No se mencionó que Basilea exige un mínimo de grados para carteras no minoristas ni cómo los supervisores miden concentración.

---

## 2. Intuición

**Una master scale es un cuantizador con contrato.** El modelo produce un número continuo (log-odds, PD o score). La escala lo redondea a K niveles con nombre, y a cada nivel le adjunta una **promesa** (una PD) que después se audita. Toda la teoría de este módulo sale de pensar en las dos caras de ese redondeo:

- **Lo que se pierde al redondear**: resolución. Dos clientes con PD 2,1% y 3,7% quedan juntos en C1 y reciben el mismo trato. Más bandas, menos pérdida.
- **Lo que se gana al redondear**: gobernabilidad y *poder estadístico*. Una promesa por banda se puede verificar con los malos de la banda; una promesa por cliente, no. Menos bandas, más malos por banda, más potencia para detectar que la promesa era falsa, menos clientes que cambian de etiqueta por ruido.

El óptimo está donde la resolución marginal de una banda extra ya no paga la potencia y la estabilidad que cuesta. Por eso la granularidad correcta **depende de los datos** (cuántos malos, cuánta dispersión de PD), no de una preferencia estética.

**Por qué PDO da una escala natural.** Si el score es lineal en log-odds, una banda de ancho fijo en puntos es una banda de ancho fijo en log-odds: cada banda multiplica las odds por el mismo factor. Es la misma lógica de las escalas de rating de agencias, donde cada escalón representa aproximadamente un múltiplo constante del riesgo. Pero la escala PDO fija el **ancho** y deja libre la **ocupación**: si la cartera vive en 125 puntos, no caben más de 6 o 7 bandas de 20 puntos útiles. Banco Sintético, con 11% de malos, lo muestra: la escala del curso (540…660) dejaría A1, A2 y B1 casi vacías.

**Por qué la PD de la banda no es trivial.** Dentro de una banda la PD varía por un factor 2 (con bandas de una duplicación). Asignar un solo número exige elegir qué preservar: la suma de malos esperados (media aritmética), la simetría en escala logarítmica (punto medio geométrico) o lo que se observó (tasa suavizada). Las tres coinciden al 2–7% en bandas cerradas; en bandas abiertas pueden diferir en un factor 2.

**Monótona, separable y con potencia son tres cosas distintas.** Una escala puede ser monótona en la muestra por azar (con n chico, las tasas de A1 y A2 salen en cualquier orden) o puede tener bandas monótonas que no se distinguen estadísticamente (dos bandas que el comité trata distinto pero que la evidencia no separa). Y un backtest 🟢 en una banda de PD 0,11% con 432 créditos no certifica nada: la probabilidad de detectar que la PD real es el doble es 7%.

**Recalibrar mueve la escala o mueve las promesas.** Si el nivel cambia (δ), o bien los cortes de las bandas se desplazan δ·factor puntos (las bandas están definidas en PD) o bien las bandas quedan fijas en puntos y cambia la PD que prometen. No hay tercera opción, y elegir es una decisión de gobierno.

---

## 3. Formalización

Notación: score calibrado $s$, con $s=\text{offset}+\text{factor}\cdot\ln o$, $o=(1-p)/p$ las odds buenos:malos, $\eta=\operatorname{logit}p=-\ln o$. Con el scaling del curso, $\text{factor}=\text{PDO}/\ln 2=28{,}8539$ y $\text{offset}=600-\text{factor}\ln 50=487{,}1229$. Las bandas se indexan $k=1,\dots,K$ de mejor a peor en las tablas; en el código, $0$ es la peor. Una banda es el intervalo $[c_{k},c_{k-1})$ en score (cerrado abajo: `right=False`).

### 3.1 La escala PDO: geométrica en odds

Las odds en un score $s$ son

$$o(s)=o_a\cdot 2^{(s-s_a)/\text{PDO}},$$

con ancla $(s_a,o_a)=(600,50)$. Una banda de ancho $w$ puntos tiene, en log-odds, ancho

$$h=\frac{w}{\text{factor}}=\frac{w\ln 2}{\text{PDO}},$$

es decir $w/\text{PDO}$ duplicaciones. Las odds en el techo son las del piso multiplicadas por $2^{w/\text{PDO}}$. La PD de la banda está acotada por la PD en sus límites:

$$p_{\text{techo}}=\frac{1}{1+o(c_{k-1})}\ \le\ p\ \le\ \frac{1}{1+o(c_k)}=p_{\text{piso}}.$$

En Austral, B2 = [600, 620): $o\in[50,100)$, $p\in(0{,}990\%;\,1{,}961\%]$.

**Invariancia al PDO del scaling.** Si el mismo modelo se expresa con otro PDO' (misma ancla), $s'-600=(\text{PDO}'/20)(s-600)$. Una escala de ancho $w'$ en $s'$ es idéntica a una de ancho $w=w'\cdot 20/\text{PDO}'$ en $s$: la partición solo depende de $h$. El notebook lo verifica: con PDO 40 y ancho 40, la asignación es la misma que con PDO 20 y ancho 20. **El PDO es cosmético; el ancho en log-odds no.**

**Cuántas bandas caben.** Si el score útil (p. ej. p1–p99) cubre un rango $R$ en puntos, caben $\approx R/w$ bandas interiores. En Banco Sintético, p5–p95 = 503–599 (96 puntos): con $w=20$ caben ~5 interiores más dos abiertas. Pedir K = 10 con $w=20$ produce bandas vacías (lo muestra el slider del notebook).

**Elección del desplazamiento.** Los cortes son $600+w\cdot j$ para $j$ enteros consecutivos; el $j$ inicial no lo fija la teoría. El notebook elige el que minimiza el HHI en DEV y lo declara. Austral eligió 3 cortes bajo el ancla (540, 560, 580) y 3 sobre él (620, 640, 660).

### 3.2 La escala por PD objetivo: geométrica en PD

Las escalas de rating definen límites de PD en progresión geométrica: $p_j=p_{\max}\,r^{-(j-1)}$, $j=1,\dots,K-1$, con $r=(p_{\max}/p_{\min})^{1/(K-2)}$. Convertida a score, $c_j=\text{offset}+\text{factor}\ln\frac{1-p_j}{p_j}$. El ancho en log-odds entre dos límites consecutivos es

$$\Delta\ln o=\ln\frac{(1-p_{j+1})/p_{j+1}}{(1-p_j)/p_j}=\ln r+\ln\frac{1-p_j/r}{1-p_j}.$$

El segundo término es positivo y crece con $p_j$: **las bandas de PD alta son más anchas en score** que las de PD baja. Para PD bajas ($p_j\ll1$) el término se anula y la escala geométrica en PD coincide con la geométrica en odds (PDO). Con $p_{\min}=0{,}05\%$, $p_{\max}=30\%$, K = 8 ($r=2{,}90$): el primer escalón (30% → 10,3%) mide 37,9 puntos (1,90 duplicaciones) y el último (0,145% → 0,05%) 30,8 puntos (1,54 duplicaciones). La planilla (`Diseno_PD`) lo calcula.

Cuándo importa: en carteras de consumo riesgosas (motos, Banco Sintético) la mitad de la escala vive sobre PD 10%. Con $r\approx2{,}9$, el término extra agrega 7% al ancho de la banda que parte en PD 10%, 23% a la que parte en 30% y 47% a la que parte en 50%.

### 3.3 Cuantiles

Cortes $c_k=F^{-1}(k/K)$ en la muestra de diseño. Por construcción, $\text{HHI}=1/K$ (el mínimo) si el score es continuo. Con score discreto (suma de puntos enteros, M13) hay empates en los cortes y la igualdad es aproximada; además la convención de borde importa: `pd.qcut` usa intervalos cerrados a la derecha y el curso usa `right=False`, así que los clientes exactamente en el corte cambian de lado según la herramienta. Los cuantiles igualan **población**, no riesgo: en la zona densa del score producen bandas angostas con tasas casi iguales (no separables) y en las colas bandas anchas con PD muy heterogénea.

### 3.4 Escala óptima: pérdida de información y programación dinámica

Se pre-binea el score en $M$ bloques finos ordenados, con $(n_f,b_f)$ créditos y malos. Una escala es una partición contigua en $K$ bandas $B_1,\dots,B_K$ con tasa $r_B=\sum_{f\in B}b_f/\sum_{f\in B}n_f$. La log-verosimilitud binomial con PD constante por banda es

$$\ell_K=\sum_{B}\big[b_B\ln r_B+(n_B-b_B)\ln(1-r_B)\big].$$

**Pérdida de información.** Con PD constante por bloque fino, $\ell_M=\sum_f[b_f\ln r_f+(n_f-b_f)\ln(1-r_f)]$. La diferencia es

$$\ell_M-\ell_K=\sum_B\sum_{f\in B}n_f\left[r_f\ln\frac{r_f}{r_B}+(1-r_f)\ln\frac{1-r_f}{1-r_B}\right]=\sum_f n_f\,\mathrm{KL}\big(\mathrm{Ber}(r_f)\,\|\,\mathrm{Ber}(r_{B(f)})\big)\ \ge 0.$$

Paso a paso: dentro de cada banda, $\sum_{f\in B}b_f\ln r_B=b_B\ln r_B$ porque $r_B$ es constante; se suma y resta, y se agrupa por bloque fino. Maximizar $\ell_K$ es **minimizar la divergencia KL ponderada** entre la escala fina y la agrupada, que es la información sobre $Y$ que se destruye al agrupar. Equivalentemente, como $-\ell_K/N$ es la entropía condicional empírica $\hat H(Y\mid B)$, maximizar $\ell_K$ es maximizar la información mutua $\hat I(Y;B)=\hat H(Y)-\hat H(Y\mid B)$. (El IV es otra medida de divergencia, simétrica, entre las distribuciones de buenos y malos por banda; M7 de la Serie 1. Los dos criterios ordenan particiones de forma parecida, no idéntica.)

**Programación dinámica.** El objetivo es aditivo por banda, así que la partición óptima cumple el principio de Bellman (Fisher, 1958, lo resolvió para la suma de varianzas intra-grupo). Sin restricciones entre bandas:

$$D[k,j]=\max_{i<j}\big\{D[k-1,i]+v(i,j)\big\},\qquad v(i,j)=\ell\big(\text{bloques }[i,j)\big),$$

con $v=-\infty$ si el bloque viola mínimos (≥ 3% de población y ≥ 10 malos, por ejemplo). Costo $O(KM^2)$.

**Con separabilidad.** Se exige que cada banda sea significativamente peor que la siguiente: $z(B_{k},B_{k+1})>z_{1-\alpha}$ (test de §3.6). La restricción involucra dos bandas adyacentes, así que el estado debe recordar dónde empieza la última banda:

$$D[k,i,j]=v(i,j)+\max_{h<i:\ z([h,i),[i,j))>z_{1-\alpha}}D[k-1,h,i].$$

Costo $O(KM^3)$: con $M=30$ y $K\le 14$, décimas de segundo. La separabilidad adyacente implica monotonía. El notebook verifica la DP contra enumeración exhaustiva de todas las particiones (`itertools.combinations`, $M=12$, $K=3,4,5$): mismo valor y mismos cortes.

**Una consecuencia no obvia.** Sin restricciones, $\max\ell_K$ es no decreciente en $K$ (una partición de $K$ bandas es un caso de $K+1$ con un corte redundante). Con la restricción de separabilidad, **no**: el corte redundante crea dos bandas iguales, que violan la restricción. En Banco Sintético la log-verosimilitud restringida sube de −3.082,3 (K = 4) a −3.048,7 (K = 9) y baja a −3.052,6 en K = 11; K = 12 no es factible. El máximo de bandas separables lo dictan los datos.

### 3.5 La PD de la banda

Sea una banda con límites de log-odds $\eta_{lo}<\eta_{hi}$ (el piso de score corresponde a $\eta_{hi}$) y densidad $g(\eta)$ de los clientes dentro de ella.

**(a) Media aritmética de las PD calibradas** (curso): $\bar p_B=\frac1{n_B}\sum_{i\in B}p_i$. Propiedad clave: $n_B\bar p_B=\sum_{i\in B}p_i$, es decir, **preserva los malos esperados** de la muestra de diseño en cada banda. Si las PD individuales están calibradas, la pérdida esperada por banda $\text{EL}_B=n_B\bar p_B\cdot\text{LGD}\cdot\text{EAD}$ es insesgada en esa muestra.

**(b) Punto medio geométrico:** $p^{mid}_B=\sigma\big(\tfrac{\eta_{lo}+\eta_{hi}}{2}\big)$, que corresponde a la media geométrica de las odds de los límites, $o^{mid}=\sqrt{o_{piso}\,o_{techo}}$. Depende solo de la geometría de la escala, no de la población.

**(c) Media bajo logit uniforme.** Si $g$ es uniforme en $[\eta_{lo},\eta_{hi}]$ y $h=\eta_{hi}-\eta_{lo}$, como $\int\sigma(\eta)\,d\eta=\ln(1+e^\eta)$,

$$\bar p^{unif}_B=\frac{\ln(1+e^{\eta_{hi}})-\ln(1+e^{\eta_{lo}})}{h}=\frac{\ln(1+1/o_{piso})-\ln(1+1/o_{techo})}{\ln(o_{techo}/o_{piso})}.$$

**Por qué (a)/(c) y (b) difieren: Jensen.** Para PD bajas, $\sigma(\eta)\approx e^\eta$. Entonces, con $\eta_m$ el punto medio,

$$\frac{\bar p^{unif}}{p^{mid}}\approx\frac{\frac1h\int_{-h/2}^{h/2}e^{\eta_m+u}\,du}{e^{\eta_m}}=\frac{e^{h/2}-e^{-h/2}}{h}=\frac{\sinh(h/2)}{h/2}=1+\frac{h^2}{24}+O(h^4).$$

Con una duplicación por banda ($h=\ln2$): 1,020; con dos ($h=2\ln2$): 1,082; con tres: 1,190. La media aritmética queda **sobre** el punto medio porque $\sigma$ es convexa para $p<1/2$ ($\sigma''=\sigma(1-\sigma)(1-2\sigma)>0$). Para densidad general, desarrollando alrededor de la media $\bar\eta_B$ de la banda:

$$\bar p_B\approx\sigma(\bar\eta_B)+\tfrac12\,\sigma(\bar\eta_B)(1-\sigma(\bar\eta_B))(1-2\sigma(\bar\eta_B))\,\mathrm{Var}_B(\eta).$$

El primer término depende de dónde está la masa dentro de la banda (si $g$ crece hacia el piso, $\bar\eta_B>\eta_m$ y la media sube); el segundo es el sesgo de Jensen. En Banco Sintético, con la escala PDO de 8 bandas, la razón media/punto medio va de 0,963 a 1,070: la forma de la densidad pesa más que la convexidad.

Austral lo confirma desde la geometría: B2 tiene punto medio 1,394%, media uniforme 1,421% y media calibrada 1,41%; C2, 5,354% / 5,445% / 5,43%; D, 10,16% / 10,31% / 10,16%. La media calibrada del curso cae entre las dos referencias geométricas.

**(d) Tasa observada suavizada.** La tasa cruda de cada banda es ruidosa (y puede ser 0 en bandas buenas). Suavizar impone estructura: una logística ponderada $\operatorname{logit} r_B=a+b\,\operatorname{logit}\bar p_B$ (2 parámetros, monótona si $b>0$), isotónica sobre bandas, o un modelo bayesiano beta-binomial que contrae la tasa de cada banda hacia la PD del modelo. Para bandas de muy baja PD sin malos, el enfoque conservador de Pluto y Tasche (2005) acota la PD por arriba con intervalos de confianza. La tasa suavizada es la que usa una master scale cuando la PD por banda se «re-ancla» a lo observado (recalibración por bandas o *histogram binning*, M15 §3.6).

**Bandas abiertas.** A1 y E no tienen punto medio ni media uniforme. Solo (a) y (d) están definidas. En Austral, E tiene PD en su techo de 13,8% y media 26,75%; A1, PD en su piso de 0,25% y media 0,11%. Cualquier «PD de la banda abierta» es una convención que se declara.

**Impacto en provisiones.** $\text{EL}=\sum_B n_B\,p_B\,\text{LGD}\,\text{EAD}_B$. Usar (b) en vez de (a) en bandas cerradas cambia la EL por el factor $p^{mid}/\bar p$ (−2% con densidad plana y una duplicación). En Banco Sintético TTD (LGD 45%, EAD 1): individual 5,43% del saldo · media 5,39% · punto medio 5,46% · logit uniforme 5,51% · suavizada 5,33% · **verdad 7,53%**. La regla por banda mueve la EL en décimas de punto; el nivel (el deterioro de 2025 que la tendencia central no incluye) la mueve en dos puntos. **Primero el nivel, después la regla.**

### 3.6 Requisitos cuantitativos

**Concentración.** Con cuotas $s_k=n_k/N$:

$$\text{HHI}=\sum_k s_k^2\in[1/K,\,1],\qquad \text{HHI}^*=\frac{\text{HHI}-1/K}{1-1/K}\in[0,1],\qquad K_{eq}=1/\text{HHI}.$$

$K_{eq}$ es el «número equivalente de bandas igualmente pobladas». El BCE, en sus instrucciones de reporte de validación de modelos IRB (2019), usa el coeficiente de variación de las cuotas y un índice normalizado:

$$\text{CV}=\sqrt{K\sum_k(s_k-1/K)^2},\qquad \text{HI}=1+\frac{\ln\big((\text{CV}^2+1)/K\big)}{\ln K}.$$

Como $\sum(s_k-1/K)^2=\text{HHI}-1/K$, se tiene $\text{CV}^2+1=K\cdot\text{HHI}$ y por tanto $\text{HI}=1+\ln\text{HHI}/\ln K$: el índice del BCE es el HHI en escala logarítmica normalizada, 0 si la población es uniforme y 1 si está toda en un grado. El test que acompaña compara la concentración actual con la del desarrollo, con estadístico $S=\sqrt{K-1}\,(\text{CV}_{act}-\text{CV}_{ini})/\sqrt{\text{CV}_{act}^2(0{,}5+\text{CV}_{act}^2)}$ y $p=1-\Phi(S)$ (verificar la versión vigente del documento). Dos advertencias: el test trata a los $K$ grados como si fueran $K-1$ observaciones, así que tiene poquísima potencia; y **no existe un umbral regulatorio numérico** de HHI. Lo que sí existe es la exigencia cualitativa (§4) y umbrales internos por convención (p. ej. ninguna banda sobre 20–30% de la población), que conviene declarar como tales.

**Monotonía.** La hipótesis es un orden: $\pi_1<\pi_2<\dots<\pi_K$ (de mejor a peor). Tres niveles de exigencia:

1. *Descriptiva*: sin inversiones en las tasas observadas. Es lo que hizo el curso. Falla por azar con n chico (abajo).
2. *Pares adyacentes*: rechazar $\pi_k\ge\pi_{k+1}$ para cada par (lo que sigue, «separabilidad»).
3. *Tendencia global*: test de Cochran–Armitage (lineal en un puntaje de banda) o razón de verosimilitudes isotónica. Detectan que hay orden en conjunto, no que cada par esté ordenado.

**Probabilidad de inversión espuria.** Aunque $\pi_k<\pi_{k+1}$ sea cierto, con $X_k\sim\text{Bin}(n_k,\pi_k)$ independientes

$$P(\text{inversión}_k)=P\!\left(\frac{X_k}{n_k}\ge\frac{X_{k+1}}{n_{k+1}}\right)=\sum_{x=0}^{n_k}P(X_k=x)\,P\!\left(X_{k+1}\le\left\lfloor x\,\tfrac{n_{k+1}}{n_k}\right\rfloor\right).$$

Con las PD de Austral y el n de OOT de la clase 5: A1–A2 49,2%, A2–B1 37,6%, B1–B2 17,7%, B2–C1 17,9%, C1–C2 8,2%, C2–D 4,0%; **al menos una inversión: 81%** (Monte Carlo del notebook; con 10 veces más n, 9%). La monotonía observada en OOT de Austral en las bandas buenas es, en parte, suerte.

**Separabilidad entre bandas adyacentes.** Test de dos proporciones con varianza agrupada bajo $H_0:\pi_k=\pi_{k+1}$:

$$z=\frac{\hat\pi_{k+1}-\hat\pi_k}{\sqrt{\bar\pi(1-\bar\pi)\left(\frac1{n_k}+\frac1{n_{k+1}}\right)}},\qquad\bar\pi=\frac{x_k+x_{k+1}}{n_k+n_{k+1}},$$

unilateral ($H_1$: la banda peor tiene más tasa), $p=1-\Phi(z)$. Es lo que calcula `statsmodels.stats.proportion.proportions_ztest(..., alternative="larger")` con su varianza agrupada por defecto. Con pocos malos conviene el test exacto condicional (Fisher) o intervalos de Jeffreys; la superposición de intervalos al 90% **no** es un test (dos intervalos pueden solaparse y la diferencia ser significativa), pero es la lectura visual habitual (Hanson y Schuermann, 2006, muestran que los intervalos de PD de grados adyacentes de agencias se superponen con frecuencia, sobre todo en grado de inversión).

**Mínimo de n y de malos por banda.** Se deriva de la potencia (§3.7). La regla no es «n ≥ 100», es **malos esperados** en la ventana de backtesting.

### 3.7 Potencia del backtest por banda

Test binomial unilateral: $H_0:\pi=p_0$ (la PD prometida) contra $H_1:\pi=m\,p_0$ con $m>1$ (la banda subestima). Con $\lambda_0=np_0$ y $p_0$ pequeño, $D\approx\text{Poisson}$ y, por aproximación normal, $D\mid H_0\approx N(\lambda_0,\lambda_0)$, $D\mid H_1\approx N(m\lambda_0,m\lambda_0)$. Se rechaza si $D>\lambda_0+z_{1-\alpha}\sqrt{\lambda_0}$. La potencia es

$$1-\beta=\Phi\!\left(\frac{m\lambda_0-\lambda_0-z_{1-\alpha}\sqrt{\lambda_0}}{\sqrt{m\lambda_0}}\right).$$

Igualando el argumento a $z_{1-\beta}$: $(m-1)\lambda_0-z_{1-\alpha}\sqrt{\lambda_0}=z_{1-\beta}\sqrt{m\lambda_0}$. Dividiendo por $\sqrt{\lambda_0}$:

$$\sqrt{\lambda_0}\,(m-1)=z_{1-\alpha}+z_{1-\beta}\sqrt m\quad\Longrightarrow\quad \boxed{\lambda_0=\left(\frac{z_{1-\alpha}+z_{1-\beta}\sqrt m}{m-1}\right)^2}.$$

$\lambda_0$ no depende de $p_0$: **la potencia la compran los malos esperados**. Con $\alpha=5\%$, potencia 80%:

| m (PD real / prometida) | 3 | 2 | 1,5 | 1,25 |
|---|---|---|---|---|
| malos esperados $\lambda_0$ | 2,4 | 8,0 | 28,6 | 107 |

El test exacto es conservador por la discreción: su tamaño real queda bajo α y su potencia bajo la aproximación. Para $p_0=1{,}42\%$ y $m=2$, con 8 malos esperados la potencia exacta es 0,73 (normal: 0,80); hacen falta ≈ 12 para 0,88.

Austral OOT (clase 5), potencia exacta para detectar $m=2$: A1 7,1%, A2 17,4%, B1 27,6%, B2 35,2%, C1 69,1%, C2 88,3%, D 98,1%, E ≈ 100%. El n mínimo para $\lambda_0=8$: A1 ≈ 7.310 créditos, A2 ≈ 2.230, B1 ≈ 1.120, B2 ≈ 570. Los 🟢 de A1 y B1 en la clase 5 son «sin evidencia», no «validado». El amarillo de B2 (p = 0,020) apareció con potencia 35%: cuando un test de baja potencia rechaza, el efecto probablemente es grande (tasa observada 3,40% vs 1,42%, $m\approx2{,}4$).

### 3.8 Granularidad y migración

**Resolución.** Con una escala de K bandas, el AUC del score bandeado es $P(\text{concordancia})+\tfrac12P(\text{empate})$; los pares que caen en la misma banda aportan ½. El Gini de la escala crece con K y converge al del score continuo. En Banco Sintético (cuantiles): K = 4 → 0,481; K = 8 → 0,520; K = 16 → 0,528; continuo 0,532.

**Migración por ruido.** Sea $s'$ el score de comportamiento seis meses después, con la misma marginal y correlación $\rho$: $s'=\mu+\rho(s-\mu)+\sqrt{1-\rho^2}\,\sigma\varepsilon$. El cambio $\Delta=s'-s=-(1-\rho)(s-\mu)+\sqrt{1-\rho^2}\,\sigma\varepsilon$ tiene

$$\mathrm{Var}(\Delta)=(1-\rho)^2\sigma^2+(1-\rho^2)\sigma^2=2(1-\rho)\sigma^2.$$

Si la posición dentro de una banda de ancho $w$ es uniforme e independiente de $\Delta$, y $\Delta\sim N(0,\sigma_\Delta^2)$, la probabilidad de quedarse es $E[(1-|\Delta|/w)^+]$. Con $a=w/\sigma_\Delta$ e integrando la semi-normal:

$$P(\text{misma banda})=\big(2\Phi(a)-1\big)-\frac{2\sigma_\Delta}{w}\big(\varphi(0)-\varphi(a)\big).$$

En Banco Sintético, $\sigma=28{,}9$ puntos, $\rho=0{,}85$ ⇒ $\sigma_\Delta=15{,}9$; con $w=20$, $a=1{,}26$ y la fórmula da 44,7%. La simulación del notebook sobre la escala PDO de 8 bandas da 45,3%. Con cuantiles (bandas centrales más angostas) la permanencia es 58,6% con K = 4, 36,8% con K = 8 y 20,6% con K = 16: la migración pasa de 41% a 63% y a 79%. **Cada duplicación de K sube fuerte la migración**, mientras el Gini gana 3,9 puntos de 4 a 8 y menos de uno de 8 a 16.

**Matriz de migración.** $P_{ij}=N_{ij}/N_i$ entre la banda $i$ hoy y la $j$ en el horizonte. Se lee: diagonal (estabilidad), masa sobre y bajo la diagonal (mejoras y deterioros), salto medio. Con la misma marginal los flujos son aproximadamente simétricos; en datos reales, una asimetría persistente es deriva o ciclo. Las instrucciones del BCE incluyen tests sobre esta matriz: monotonía de las probabilidades fuera de la diagonal (que la probabilidad de migrar decrezca con la distancia) y el *matrix weighted bandwidth* (MWB), un salto medio normalizado; M20 los usa en monitoreo.

**PIT vs TTC.** Una escala cuyas PD siguen el ciclo (PIT) hace migrar a toda la cartera en la recesión (el rating baja porque la PD sube); una TTC la mantiene estable y concentra el ciclo en la tasa observada por banda (Serie 1 · E2). La matriz de migración es la huella de la filosofía de rating.

### 3.9 Recalibrar mueve las bandas δ·factor

Recalibrar el intercepto (M15) cambia el logit de todos: $\eta\to\eta+\delta$, y el score $s\to s-\delta\cdot\text{factor}$. Dos formas coherentes de gobernar la escala:

- **Bandas definidas en PD** (o en el score calibrado): los límites en PD no cambian; en el score sin recalibrar, los cortes suben $\delta\cdot\text{factor}$ puntos. Cambian de banda los clientes en la franja $[c_k,\,c_k+\delta\cdot\text{factor})$ sobre cada corte:

$$P(\text{cambia de banda})=\sum_k\int_{c_k}^{c_k+\delta\,\text{factor}}f(s)\,ds\ \approx\ \delta\cdot\text{factor}\sum_k f(c_k).$$

- **Bandas fijas en puntos**: nadie migra, pero la PD que promete cada banda pasa a $\sigma(\operatorname{logit}p_B+\delta)$ y todo lo escrito en PD (apetito, pricing, provisiones) cambia.

Austral: TC $\delta=0{,}1115$ ⇒ 3,22 puntos; PIT $\delta=0{,}177$ ⇒ 5,11; la diferencia TC → PIT es 1,89 puntos: con bandas de 20 puntos y densidad plana, ≈ 9% de cada banda interior bajaría una banda. Banco Sintético, recalibrando a OOT (deterioro plantado): $\delta_{PIT}-\delta_{TC}=0{,}376$ ⇒ 10,85 puntos ⇒ 55,9% de la bandeja TTD baja una banda (A1 pasa de 4,4% a 0,5%). Si las bandas quedan fijas en puntos, la banda 540–560 de Banco Sintético pasa a prometer 14,1% en vez de 10,2%.

### 3.10 Varios modelos, una escala

La escala corporativa se define en **PD** (límites $p_1>p_2>\dots$). Cada modelo $m$ con su propio scaling $(\text{PDO}_m,s_{a,m},o_{a,m})$ y su propia calibración se mapea así:

$$\ln o=\ln o_{a,m}+\frac{(s_m-s_{a,m})\ln 2}{\text{PDO}_m}\ \Rightarrow\ s^{corp}=\text{offset}+\text{factor}\cdot\ln o\ \Rightarrow\ \text{banda}.$$

Condiciones: (i) cada modelo calibrado a su propia tendencia central antes de mapear; (ii) la misma banda debe significar la misma PD en todos los modelos, lo que se valida con backtest **por modelo y por banda**; (iii) un modelo menos discriminante ocupará menos bandas (más concentrado en el centro): es información sobre el modelo, no defecto de la escala. Copiar los cortes numéricos de un modelo a otro con distinto scaling es el error típico: en el notebook, el modelo «motos» (PDO 40, 500 ⇔ 20:1) con los cortes 480…600 copiados pone 55,2% de la cartera en E y 0% en A1.

---

## 4. Variantes y alternativas de industria

**Diseños de escala.**

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| PDO (geométrica en odds) | Bandas de riesgo relativo constante, directo desde el scaling | Ocupación libre; bandas vacías si el rango del score es estrecho | Scorecards de originación con scaling PDO; escalas por producto | Práctica de scoring (Siddiqi); el curso |
| PD objetivo (geométrica en PD) | Escala comparable entre modelos y con escalas externas | Más anchas en score las bandas de PD alta; ocupación libre | Escala corporativa multi-modelo; mapeo a ratings de agencia | Escalas maestras de bancos IRB; agencias (AAA…C, ~20 escalones) |
| Cuantiles | Concentración mínima; misma población por banda | Bandas no separables en el centro; PD heterogénea en las colas | Tablas de deciles para validación (M12), no para gobernar | Validación; reporting |
| Óptima (DP / programación entera) | Pérdida de información mínima con restricciones explícitas | Depende de la muestra (se congela); riesgo de sobreajuste | Cuando hay muchos datos y se quiere la mayor granularidad que resiste validación | Binning óptimo (`optbinning` resuelve el problema análogo por variable con MIP/CP) |
| Escala regulatoria fija | PD de referencia uniforme para provisiones | No refleja la discriminación del modelo; saltos no geométricos | Provisiones por categoría | CMF, cap. B-1: evaluación individual A1–A6 (normal), B1–B4 (subestándar), C1–C6 (incumplimiento) |
| Escala continua (estimación directa) | Evita el redondeo | No hay «PD por grado» que validar; se valida por agrupaciones | Modelos con PD individual usada directamente | CRR art. 169(3): la estimación directa se entiende como escala continua (EBA/GL/2017/16) |

**Asignación de PD a la banda.**

| Método | Qué preserva | Problema | Uso típico |
|---|---|---|---|
| Media de PD calibradas | Malos esperados por banda en la muestra | Hereda cualquier sesgo de nivel del modelo | El curso; práctica corriente |
| Punto medio geométrico | Simetría logarítmica; independiente de la población | Sesgo de Jensen (−2% por duplicación); no existe en bandas abiertas | Escalas de rating con PD de referencia fija por grado |
| Tasa observada cruda | Lo observado | Ruidosa; ceros en bandas buenas; no monótona | Nunca sola |
| Tasa suavizada (logística, isotónica, beta-binomial) | Lo observado con estructura | Requiere muestra madura; decisión de modelado extra | Recalibración por grado (EBA: calibración a nivel de grado o de segmento) |
| Cota conservadora (Pluto–Tasche) | Prudencia con pocos o cero malos | Conservadora por diseño | Bandas de muy baja PD, carteras de bajo default |

**Requisitos regulatorios (verificados, con cautela).** Basilea II (2006), requisitos mínimos IRB (hoy capítulo CRE36 del marco consolidado; verificar numeración): «una distribución significativa de exposiciones entre grados, sin concentraciones excesivas»; **mínimo de siete grados de deudor para no incumplidos y uno para incumplidos** (carteras soberanas, bancos y empresas); las concentraciones significativas en un grado deben respaldarse con evidencia empírica de que el grado cubre una banda de PD razonablemente estrecha. Para **minoristas** no hay número mínimo de grados o *pools*: se exige una distribución significativa entre *pools* y que ningún *pool* concentre indebidamente la exposición minorista. En la UE, el CRR (Reglamento 575/2013) art. 170 recoge lo mismo (170(1)(b): mínimo 7 grados más 1 de incumplimiento en no minoristas). Es frecuente que los bancos IRB grandes usen escalas internas de 15–25 grados, a menudo alineadas con los escalones de agencia (práctica, no requisito). El BCE mide concentración con el HI de §3.6 y exige tests sobre la matriz de migración. En Chile, el capítulo B-1 del Compendio de Normas Contables (CMF) fija, para evaluación individual, una escala de 10 categorías con PD de referencia (versión consultada: A1 0,04% · A2 0,10% · A3 0,25% · A4 2,00% · A5 4,75% · A6 10,00% · B1 15,00% · B2 22,00% · B3 33,00% · B4 45,00%; verificar vigencia): una escala **no geométrica** (razones entre 1,4 y 8), definida por provisiones y no por discriminación. El crédito de consumo se provisiona con métodos grupales (y mínimos estándar), así que la master scale de un scorecard de consumo es un artefacto de gestión, no regulatorio.

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 Copiar los cortes de otra cartera o de otro modelo.**
*Síntoma:* bandas extremas vacías o con la mitad de la cartera. *Causa:* los cortes 540…660 codifican odds bajo un scaling y un nivel de riesgo concretos. *Detección:* ocupación por banda en DEV; HHI; rango de score p1–p99 contra los cortes. *Qué hacer:* copiar la regla (límites en PD o ancho en log-odds), recalcular cortes; mapear modelos por PD (§3.10). En el notebook: cortes copiados → 55% en E.

**5.2 Bandas de ancho fijo que no caben.**
*Síntoma:* la escala PDO con K = 8 tiene una banda con 39 casos (0,4%) y 7 malos (Banco Sintético). *Causa:* el rango del score es ~6 duplicaciones; ocho bandas de una duplicación no caben con ocupación razonable. *Detección:* mínimo de n y de malos por banda, bandas vacías. *Qué hacer:* ensanchar las bandas extremas (abiertas), reducir K o usar ancho de 1,5 duplicaciones. La restricción es de la información, no del diseño.

**5.3 Monotonía «verde» por azar.**
*Síntoma:* la escala es monótona en DEV y OOT, se da por validada. *Causa:* con pocos malos, el orden observado es casi aleatorio en las bandas buenas (81% de probabilidad de al menos una inversión con el n OOT de Austral aunque la verdad sea monótona; el complemento es que también puede salir monótona con una escala mal ordenada). *Detección:* tests de separabilidad; probabilidad de inversión bajo la escala supuesta; intervalos de Jeffreys. *Qué hacer:* reportar monotonía con potencia; fusionar bandas para el test cuando no alcance; acumular cosechas.

**5.4 Bandas monótonas pero no separables.**
*Síntoma:* el comité asigna precios o cupos distintos a C1 y C2, pero sus tasas (3,70% vs 4,44%) no son estadísticamente distintas (p = 0,23). *Causa:* bandas demasiado angostas para el n disponible, o densidad alta en esa zona. *Detección:* z adyacentes; DP restringida para ver cuántas bandas separables soporta la muestra. *Qué hacer:* fusionar para validación manteniendo la etiqueta comercial, o rediseñar. Documentar qué pares no se separan.

**5.5 Concentración escondida en una banda abierta.**
*Síntoma:* HHI bajo, pero A1 tiene 21,5% de Austral con PD entre 0 y 0,25%. *Causa:* la banda abierta junta clientes con PD que difieren en órdenes de magnitud. *Detección:* dispersión de la PD dentro de la banda (p1/p99 de PD calibrada); cuota de la banda abierta. *Qué hacer:* partir la banda abierta si su PD interna es heterogénea y hay malos para validarla; si no, declarar la PD de la banda como media y vigilar su cuota (en Austral TTD bajó a 18,4%).

**5.6 PD por banda inconsistente con sus límites.**
*Síntoma:* Austral: A2 observa 0,61% con límites de PD [0,25%; 0,50%]; B1 1,03% con [0,50%; 0,99%]; B2 2,05% con [0,99%; 1,96%]. *Causa:* nivel del modelo subestimado en las bandas buenas (no es un problema de la escala sino de la calibración, que la escala hace visible). *Detección:* columna «¿tasa dentro del rango de PD de la banda?» y test binomial por banda. *Qué hacer:* recalibrar (con pendiente si el sesgo depende del nivel, M15 §3.5) o recalibrar por bandas con tasa suavizada. No mover los cortes para «hacer calzar».

**5.7 Punto medio en bandas abiertas o regla de PD no declarada.**
*Síntoma:* dos equipos calculan distinta EL con la misma escala. *Causa:* uno usa media, otro punto medio, y cada uno resuelve distinto las bandas abiertas. *Detección:* test de paridad de la tabla banda → PD entre sistemas. *Qué hacer:* la regla es parte del artefacto (§6), con la convención explícita para A1 y E.

**5.8 Recalibrar sin decidir qué se mueve.**
*Síntoma:* tras la recalibración anual, la ocupación por banda salta (las bandas estaban en PD y nadie lo anticipó) o las PD por banda dejan de calzar con el apetito (estaban en puntos). *Causa:* la escala no declaraba si sus límites viven en PD o en puntos. *Detección:* diff de la ocupación TTD antes/después; fracción que cambia de banda ≈ δ·factor·Σ densidad en cortes. *Qué hacer:* declarar en el artefacto «límites en PD» o «límites en score»; simular el efecto antes de firmar (Banco Sintético: 56% de migración con 10,9 puntos).

**5.9 Optimizar cortes sin restricciones o sobre la muestra de validación.**
*Síntoma:* escala «óptima» con 14 bandas que en HO tiene 2 inversiones y 8 pares no separables. *Causa:* maximizar verosimilitud sin exigir monotonía ni separabilidad corta donde el ruido sube; con mínimos laxos, cada banda tiene pocos malos. *Detección:* inversiones y separabilidad en HO/OOT. *Qué hacer:* DP restringida (separabilidad, mínimos de n y malos), diseñada en DEV y validada fuera; la restringida con K = 9 da 0 inversiones en HO, aunque 3 pares no separables (menos n en HO). La restricción regulariza, no garantiza.

**5.10 Convención de borde distinta entre sistemas.**
*Síntoma:* el motor de decisión y el reporte difieren en la banda de algunos clientes. *Causa:* `pd.cut` por defecto usa `right=True` (el corte pertenece a la banda de abajo); el curso usa `right=False`; SQL con `BETWEEN` incluye ambos bordes. Con score entero (M13), la masa exactamente en cada corte es del orden de la densidad por punto (≈ 1–2% de la cartera en la zona densa). *Detección:* test de paridad con scores exactamente en los cortes. *Qué hacer:* declarar el borde en el artefacto y probarlo en CI (§6).

**5.11 Escala construida sobre el score equivocado.**
*Síntoma:* la ocupación del motor no calza con la del documento. *Causa:* escala sobre el score continuo del notebook y producción con score entero de la tabla redondeada, o escala sobre el score sin calibrar y PD de la calibrada. *Detección:* paridad score oficial vs score de desarrollo. *Qué hacer:* la escala se construye sobre el score oficial (M13 §3.5) y se declara si es calibrado.

**5.12 Granularidad por gusto.**
*Síntoma:* 12 bandas «porque el banco tiene 12»; migración mensual del 60%; mitad de las bandas sin potencia. *Causa:* K elegido sin mirar malos por banda ni estabilidad. *Detección:* curva Gini-vs-K, permanencia vs K, mínimo de malos esperados por banda en la ventana de backtest. *Qué hacer:* elegir K en el codo; si la escala corporativa tiene más grados que los que el modelo soporta, el modelo ocupa menos grados (y se valida en agrupaciones).

---

## 6. Puente con ingeniería

La master scale es un **artefacto de política**, separado del scorecard (M13 §3.7): el scorecard (bins, WoE, β, tabla de puntos) se congela con el modelo; la escala tiene su propio ciclo de vida, su hash y su dueño. Un cambio de nivel nunca debería cambiar el hash del scorecard; puede cambiar el de la escala.

**Declaración (YAML versionado):**

```yaml
master_scale:
  id: ms_consumo_corporativa
  version: 3.1.0
  vigente_desde: 2026-10-01
  dueño: riesgo_de_credito/modelos
  definicion_limites: pd              # pd | score  (qué queda fijo al recalibrar)
  borde: right_false                  # score == corte → banda superior
  bandas:                             # de mejor a peor; límites en PD (techo, piso]
    - {nombre: A1, pd_hasta: 0.002494}
    - {nombre: A2, pd_hasta: 0.004975}
    - {nombre: B1, pd_hasta: 0.009901}
    - {nombre: B2, pd_hasta: 0.019608}
    - {nombre: C1, pd_hasta: 0.038462}
    - {nombre: C2, pd_hasta: 0.074074}
    - {nombre: D,  pd_hasta: 0.137931}
    - {nombre: E,  pd_hasta: 1.0}
  pd_por_banda:
    regla: media_pd_calibrada          # media | punto_medio | suavizada_logistica
    bandas_abiertas: media_pd_calibrada
    muestra: {nombre: DEV+HO+OOT, cohortes: 2023-07..2025-06}
    valores: {A1: 0.0011, A2: 0.0036, B1: 0.0072, B2: 0.0141, C1: 0.0279, C2: 0.0543, D: 0.1016, E: 0.2675}
  validacion:
    alfa_separabilidad: 0.05
    backtest: {alfa: 0.05, potencia: 0.80, m: 2.0, malos_esperados_min: 8}
    concentracion: {max_pct_banda: 0.25}   # convención interna, no regulatoria
  mapeos:                              # un bloque por modelo; todos por PD
    - {modelo: scorecard_austral_v1, pdo: 20, score_ancla: 600, odds_ancla: 50, calibracion: cal_tc_2025q2}
    - {modelo: motos_v2, pdo: 40, score_ancla: 500, odds_ancla: 20, calibracion: cal_motos_2025q2}
  hash: sha256:…                        # de la serialización canónica de todo lo anterior
```

**Contratos e invariantes verificables (CI):**

```python
def test_escala(ms, score_oficial_dev, pd_cal_dev, y_dev):
    lim = np.array([b["pd_hasta"] for b in ms["bandas"]])
    assert np.all(np.diff(lim) > 0) and lim[-1] == 1.0            # límites crecientes, cubren (0, 1]
    pdb = np.array(list(ms["pd_por_banda"]["valores"].values()))
    assert np.all(np.diff(pdb) > 0)                                # PD por banda estrictamente monótona
    techo = np.r_[0, lim[:-1]]
    cerradas = slice(1, -1)
    assert np.all((pdb[cerradas] > techo[cerradas]) & (pdb[cerradas] <= lim[cerradas]))  # PD dentro de sus límites
    # paridad del borde: un score exactamente en el corte cae en la banda superior
    cortes = pd_a_score(lim[:-1])[::-1]
    assert (asignar(cortes, cortes) == np.arange(1, len(cortes) + 1)).all()
    # paridad con la otra implementación (SQL / motor): mismos conteos por banda
    assert np.array_equal(conteo_motor(score_oficial_dev), np.bincount(asignar(score_oficial_dev, cortes)))
    # EL de la muestra de diseño preservada si la regla es la media
    if ms["pd_por_banda"]["regla"] == "media_pd_calibrada":
        b = asignar(score_oficial_dev, cortes)
        assert np.isclose(pdb[::-1][b].sum(), pd_cal_dev.sum(), rtol=1e-6)
```

**Qué se congela, qué se versiona, qué se recalcula:**

- *Congelado con la escala*: límites (en PD o en score, según `definicion_limites`), borde, nombres, regla de PD por banda y convención de bandas abiertas.
- *Versionado aparte*: valores de PD por banda (cambian con una recalibración), mapeos por modelo (cambian si cambia un modelo o su calibración), umbrales de validación.
- *Derivado, nunca editado a mano*: cortes en score de cada modelo (desde los límites en PD y el scaling del modelo), tabla score → banda del motor, corte de aprobación si el apetito está en PD.
- *Recalculado en cada corrida de monitoreo*: ocupación, HHI, PSI sobre las bandas, tasas, separabilidad, binomial/Jeffreys por banda **con su potencia**, matriz de migración (M20).

**Pipeline declarativo.** La escala es un nodo del DAG con entradas explícitas: `score_oficial` (del artefacto del scorecard), `calibracion` (δ versionado), `muestra_diseño`. Salidas: `master_scale.yaml`, `tabla_validacion.parquet`, `reporte.html`. Un cambio de δ dispara el recálculo de `mapeos` y del corte en score, y un *diff* de ocupación TTD antes/después que debe aprobarse (5.8). Un cambio en los límites dispara re-aprobación de comité, porque cambia el significado de las etiquetas.

**Diseñar con reglas explícitas.** Si se usa la DP, sus parámetros (M fino, mínimos, α, K) son parte del artefacto, y la escala resultante se congela: re-optimizar cortes cada trimestre convierte la escala en un objetivo móvil y destruye la comparabilidad de la matriz de migración.

---

## 7. Numpy desde cero vs librerías

| Cálculo | numpy en el notebook | Librería | Diferencias de convención | En producción |
|---|---|---|---|---|
| Asignación a banda | `np.searchsorted(cortes, s, side="right")` | `pd.cut(..., right=False)` | `pd.cut` por defecto es `right=True`: el corte cae abajo | `searchsorted` (sin categorías), borde declarado |
| Cortes por cuantiles | `np.quantile` | `pd.Series.quantile`, `pd.qcut` | `qcut` cierra a la derecha y falla con empates duplicados (`duplicates=`) | `np.quantile` + borde explícito |
| HHI, índice BCE | fórmulas directas | — | El BCE define CV con $\sqrt{K\sum(s-1/K)^2}$; identidad $\text{CV}^2+1=K\cdot\text{HHI}$ | numpy |
| z de dos proporciones | fórmula con varianza agrupada | `statsmodels.stats.proportion.proportions_ztest` | `statsmodels` agrupa por defecto (`prop_var=False`); sin corrección de continuidad | `statsmodels` (o exacto de Fisher con pocos malos) |
| IC de Jeffreys | `scipy.stats.beta.ppf(q, x+½, n−x+½)` | `statsmodels proportion_confint(method="jeffreys")` | Idénticos para $0<x<n$; en los bordes `statsmodels` fija 0 o 1 | `statsmodels` |
| Cola binomial y potencia | log-pmf con `lgamma` y suma estable | `scipy.stats.binom.sf` | `binom.sf(k−1)` = P(D ≥ k); cuidado con el −1 | `scipy` |
| Media bajo logit uniforme | fórmula cerrada con `log1p(exp(·))` | `scipy.integrate.quad` | Fórmula exacta; `quad` solo verifica | fórmula |
| Tasa suavizada | IRLS ponderado (2 parámetros) | `statsmodels.GLM(Binomial, var_weights=n)` | Con tasas y `var_weights`, mismos β que con conteos | `statsmodels` |
| Escala óptima | DP $O(KM^3)$ con separabilidad | Enumeración (`itertools`); `optbinning` resuelve el problema análogo (binning óptimo por programación entera/restricciones) | `optbinning` admite restricciones parecidas (`min_bin_size`, `monotonic_trend`, `min_event_rate_diff`, `max_pvalue` entre bins consecutivos), pero su objetivo por defecto es el IV, no la verosimilitud; verificar documentación | DP propia testeada contra fuerza bruta, u `optbinning` sobre el score con sus parámetros congelados |
| Migración | `np.add.at` sobre la matriz | `pd.crosstab(normalize="index")` | `crosstab` omite filas/columnas vacías: reindexar | cualquiera, con reindex |
| Gini de la escala | AUC desde conteos por banda, empates = ½ | `sklearn.metrics.roc_auc_score` | Idénticos con empates | `sklearn` |

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral, re-auditado

Datos de la tabla de la clase 4 (malos reconstruidos como round(tasa × n): 2, 4, 8, 17, 30, 33, 60, 208; total 362 = 5,38%). La planilla `Validacion` los trae por defecto.

- **Concentración.** HHI 0,1352 (mínimo 0,125; $K_{eq}=7{,}40$; $\text{HHI}^*=0{,}012$); índice del BCE 0,038 en modelación y 0,022 en TTD. El test de aumento de concentración da $p=0{,}88$: la TTD está *menos* concentrada (A1 bajó de 21,5% a 18,4%). Sin problema de concentración; la única banda sobre 20% es la abierta A1.
- **Geometría vs PD asignada.** La PD calibrada media de cada banda cerrada cae entre el punto medio geométrico y la media uniforme (B2: 1,394% / 1,41% / 1,421%). La escala está bien construida sobre un score bien escalado.
- **Tasa fuera del rango.** A2 (0,61% vs techo de PD 0,50%), B1 (1,03% vs 0,99%) y B2 (2,05% vs 1,96%) observan más que la PD máxima de su banda (la del piso de score). Es la calibración TTC subestimando las bandas buenas, la misma firma del backtest OOT de la clase 5.
- **Separabilidad (unilateral, 5%).** E–D y D–C2: $p<0{,}001$; C2–C1: $p=0{,}23$; C1–B2: 0,022; B2–B1: 0,050; B1–A2: 0,19; A2–A1: 0,030. Tres pares no separables: C2–C1, B2–B1 (en el límite) y B1–A2. Con el n agregado de 6.723 créditos, Austral soporta estadísticamente 5–6 bandas separadas, no 8. La escala sirve para gobernar, pero sus 8 etiquetas prometen más resolución de la que la evidencia distingue.
- **Backtest OOT.** Potencias para $m=2$: A1 7%, A2 17%, B1 28%, B2 35%, C1 69%, C2 88%, D 98%, E ≈ 100%. Lectura correcta del tablero de la clase 5: D y E validados; C1–C2 razonablemente; A1–B1 **sin evidencia en ninguna dirección**; el amarillo de B2 es real y grande. Para validar A1 con $m=2$ harían falta ≈ 7.300 créditos en la ventana: unas 17 veces lo que había.
- **Recalibración.** TC → PIT desplaza 1,89 puntos; con bandas en PD, ≈ 9% de cada banda interior baja un escalón. La clase 5 decidió que la aprobación casi no cambia (78,3% vs 77,2% en el corte 560): coherente con el desplazamiento chico.

### 8.2 Banco Sintético (notebook, verdad conocida)

Tendencia central (DEV+HO) 11,43%, δ = 0,015. Score DEV p5–p95: 503–599, desviación estándar 28,9 puntos. Gini continuo DEV 0,532.

- **Cuatro diseños, K = 8:**

| Diseño | Cortes | HHI | Máx. % | Mín. malos | Inversiones | No separables | Gini escala |
|---|---|---|---|---|---|---|---|
| PDO (w = 20) | 480, 500, …, 600 | 0,193 | 28,7% | 7 | 0 | 0 | 0,521 |
| PD objetivo (1,5%–40%) | 498,8 … 607,9 | 0,169 | 22,7% | 3 | 1 | 1 | 0,520 |
| Cuantiles | 520,8 … 588,8 | 0,125 | 12,6% | 25 | 0 | 1 | 0,520 |
| Óptimo (DP restringida) | 496,9 … 582,4 | 0,160 | 20,0% | 47 | 0 | 0 | 0,525 |

La DP gana a los tres en resolución y es la única que cumple todas las restricciones a la vez, pero sus cortes no son «redondos» ni comparables entre modelos. La PDO tiene una banda E con 39 casos (0,4%): no cumple el mínimo de 3%.

- **Máximo de bandas separables** (≥ 3%, ≥ 10 malos, α = 5%): K = 11; la verosimilitud restringida alcanza su máximo en K = 9.
- **PD por banda (PDO, 8 bandas):** media/punto medio entre 0,963 y 1,070; EL TTD 5,39% (media) a 5,51% (uniforme) vs verdad 7,53%.
- **Potencia:** con $p_0=1{,}42\%$ y $m=2$, 8 malos esperados dan potencia exacta 0,73; 12 dan 0,88.
- **Granularidad (ρ = 0,85):** Gini 0,481 → 0,520 → 0,528 con K = 4, 8, 16; permanencia 58,6% → 36,8% → 20,6%; mínimo de malos esperados OOT con K = 16: 5,0.
- **Migración (PDO, 8 bandas):** diagonal 45,3% (fórmula de §3.8: 44,7%), 26,9% mejora, 27,9% empeora, salto medio 1,14 bandas.
- **Recalibración a OOT:** +10,85 puntos, 55,9% de TTD baja una banda.
- **Mapeo del modelo motos:** HHI 0,208 vs 0,193 (más concentrado, E vacía), tasas comparables por banda; con cortes copiados, 55% en E.

### 8.3 Crédito de motos: una escala corporativa para dos carteras

Supuestos ilustrativos (no datos de Galgo): consumo general con tasa de malos ~5% y motos con ~12%, un scorecard por producto, ambos con scaling PDO 20 / 600 ⇔ 50:1 y calibración propia.

1. **Rango.** Motos vive en PD 1%–50% (como Banco Sintético); consumo en 0,05%–30%. Una escala corporativa en PD de 0,05% a 50% con límites geométricos cubre ambas. Motos ocupará las 5–6 bandas peores; consumo, las 6–7 mejores. Es correcto: la misma etiqueta, la misma PD.
2. **Diseño por PD, no por PDO.** En la zona de PD alta (10%–50%) una escala geométrica en odds y una geométrica en PD difieren entre 7% y 47% de ancho por banda (§3.2). Para que «D» signifique lo mismo en ambos productos, la escala se define en PD y cada modelo se mapea (§3.10).
3. **Potencia.** Supón 800 créditos de motos por mes, 15% en la banda de PD 3%. Cada cosecha aporta $800\times0{,}15\times0{,}03=3{,}6$ malos esperados en esa banda: para detectar $m=1{,}5$ (28,6 malos) hacen falta **8 cosechas**; para $m=2$, 3 cosechas. En la banda de PD 1% (supón 5% de la cartera), 0,4 malos por cosecha: 20 cosechas para $m=2$. El backtest por banda en motos se hace sobre ventanas acumuladas o bandas agregadas, y así se escribe en la política.
4. **Recalibración.** Motos es más cíclico (desempleo, precio del combustible). Si la escala está en PD y se recalibra PIT, la ocupación de motos se moverá varios puntos por ciclo; si el pricing está atado a la banda, el precio también. Alternativa: escala TTC para gobernar y pricing con un ajuste PIT explícito.
5. **Concentración.** Un scorecard de motos con Gini ~0,5 concentrará más en el centro (como el modelo 2 del notebook). Un HHI alto en motos no es falla de la escala: es la discriminación disponible. El umbral interno de concentración se evalúa por modelo.

---

## 9. Preguntas de comité

**1. ¿Por qué 8 bandas y no 12?**
Porque 8 es lo que la evidencia puede sostener y gobernar. Criterios: (i) resolución: el Gini de la escala está a menos de un punto del continuo desde K ≈ 8; (ii) potencia: cada banda necesita ~8 malos esperados en la ventana de backtest para detectar que su PD real duplica la prometida, y con más bandas las buenas no llegan; (iii) estabilidad: pasar de 8 a 16 bandas sube la migración a 6 meses de 63% a 79% en nuestro ejemplo; (iv) separabilidad: la DP restringida muestra cuántas bandas separables soporta la muestra. «7–10» es convención de industria; nuestro número sale de estos cuatro cálculos. Para no minoristas, Basilea exige al menos 7 grados de no incumplidos más 1 de incumplidos; para minoristas no hay mínimo.

**2. La escala es monótona. ¿No basta?**
No. Con el n de OOT de Austral, aunque la escala verdadera sea monótona, la probabilidad de ver al menos una inversión es 81%; recíprocamente, una escala mal ordenada puede salir monótona. Además, monótona no implica separable: C2 vs C1 (4,44% vs 3,70%) no se distingue ($p=0{,}23$). Reportamos monotonía con separabilidad por par y potencia por banda.

**3. ¿Qué PD usan para provisionar por banda y por qué?**
La media de PD calibradas de la banda en la muestra de diseño, porque preserva los malos esperados (EL insesgada si la calibración lo es). El punto medio geométrico subestima ~2% por duplicación de ancho con densidad plana y no existe en bandas abiertas. La tasa observada se usa solo suavizada y en una muestra madura distinta a la de validación. La regla y la convención para A1 y E están en el artefacto.

**4. A1 tiene 🟢 en el backtest. ¿Está validada?**
No. Con 432 créditos y PD 0,11% se esperaba medio malo; la potencia para detectar que la PD real sea el doble es 7%. 🟢 significa «sin evidencia en contra», no «validado». Para validar A1 a ese nivel hacen falta ~7.300 créditos en la ventana, o agregar A1+A2, o usar una cota conservadora (Pluto–Tasche).

**5. Recalibramos δ. ¿Cambian las bandas?**
Depende de lo declarado. Si los límites viven en PD, los cortes en score suben δ·factor puntos y la fracción de clientes en la franja cambia de banda (Austral TC → PIT: 1,89 puntos, ~9% por banda interior; Sintético: 10,9 puntos, 56%). Si viven en puntos, nadie migra y cambia la PD prometida por banda. Nuestra escala declara límites en PD; presentamos el diff de ocupación antes de firmar.

**6. ¿Cómo se asegura que la banda C1 del modelo de motos signifique lo mismo que la del modelo de consumo?**
Cada modelo se calibra a su tendencia central y se mapea por PD (score propio → log-odds → banda), nunca copiando cortes. Luego se valida por modelo y por banda: la tasa observada de C1 en motos debe ser consistente con la PD de C1 (binomial/Jeffreys, con su potencia). Que motos ocupe menos bandas es esperable.

**7. ¿Por qué no usar la escala óptima que maximiza la separación?**
La usamos como referencia de cuánta granularidad soportan los datos, con restricciones de mínimo de n, de malos y de separabilidad. Como escala de gobierno preferimos límites en PD geométricos: comparables entre modelos, estables y explicables. Una escala óptima sin restricciones sobreajusta (14 bandas, 8 pares no separables en HO).

**8. El HHI es 0,135. ¿Es bueno?**
Es cercano al mínimo (0,125 con 8 bandas; 7,4 bandas equivalentes). No hay umbral regulatorio numérico; el BCE compara la concentración actual con la del desarrollo (aquí bajó). Nuestro límite interno (ninguna banda sobre 25%) es una convención declarada. Lo que sí vigilamos es la banda abierta A1 (21,5%), cuya PD interna es heterogénea.

---

## 10. Ejercicios

**E1 (cálculo).** Con PDO 20, 600 ⇔ 50:1, calcula para la banda C1 = [580, 600): odds y PD en piso y techo, PD en el punto medio geométrico y PD media bajo logit uniforme. Compara con la PD calibrada media de Austral (2,79%).

<details><summary>Solución</summary>

Piso 580: $o=50\cdot2^{-1}=25$, $p=1/26=3{,}846\%$. Techo 600: $o=50$, $p=1/51=1{,}961\%$. Punto medio: $o=\sqrt{25\cdot50}=35{,}36$, $p=1/36{,}36=2{,}751\%$. Media uniforme: $[\ln(1+1/25)-\ln(1+1/50)]/\ln 2=(0{,}039221-0{,}019803)/0{,}693147=2{,}801\%$. Austral: 2,79%, entre ambas, más cerca de la uniforme: la densidad dentro de C1 es casi plana. Razón uniforme/punto medio = 1,018 (la aproximación $\sinh(h/2)/(h/2)$ da 1,020; la diferencia es el término $1-\sigma$ que la aproximación ignora).
</details>

**E2 (derivación).** Demuestra que $\text{CV}^2+1=K\cdot\text{HHI}$ y que el índice del BCE es $\text{HI}=1+\ln\text{HHI}/\ln K$. ¿Qué valor toma HI con 2 bandas de 50% y 6 vacías (K = 8)?

<details><summary>Solución</summary>

$\sum_k(s_k-1/K)^2=\sum s_k^2-\frac2K\sum s_k+\frac{K}{K^2}=\text{HHI}-\frac1K$. Entonces $\text{CV}^2=K\,\text{HHI}-1$. Sustituyendo, $(\text{CV}^2+1)/K=\text{HHI}$ y $\text{HI}=1+\ln\text{HHI}/\ln K$. Con dos bandas de 50%: HHI = 0,5, HI = $1+\ln0{,}5/\ln8=1-1/3=0{,}667$. $K_{eq}=2$.
</details>

**E3 (potencia).** Una banda promete PD 0,8%. ¿Cuántos créditos hacen falta para detectar con 80% de potencia (α = 5% unilateral) que la PD real es 1,2%? ¿Y si fuese 1,6%?

<details><summary>Solución</summary>

$m=1{,}5$: $\lambda_0=((1{,}645+0{,}8416\sqrt{1{,}5})/0{,}5)^2=((1{,}645+1{,}031)/0{,}5)^2=28{,}6$; $n=28{,}6/0{,}008\approx3.580$. $m=2$: $\lambda_0=8{,}04$, $n\approx1.005$. Con el test exacto, algo más (≈ 10–25%) por la discreción. Si la banda tiene 5% de una originación de 2.000 créditos/mes, son 100 por mes: 36 cosechas para $m=1{,}5$, 10 para $m=2$.
</details>

**E4 (separabilidad).** Austral, B1 (774 créditos, 8 malos) vs B2 (829, 17). Calcula el z unilateral agrupado y decide al 5%. ¿Qué cambia si el test fuera bilateral?

<details><summary>Solución</summary>

$\hat\pi_{B1}=1{,}034\%$, $\hat\pi_{B2}=2{,}051\%$, $\bar\pi=25/1.603=1{,}560\%$. $\text{se}=\sqrt{0{,}01560\cdot0{,}98440\,(1/774+1/829)}=\sqrt{0{,}015356\cdot0{,}0024982}=0{,}006194$. $z=(0{,}02051-0{,}01034)/0{,}006194=1{,}642$; $p$ unilateral = 0,0503: no separable al 5% (en el límite). Bilateral: $p=0{,}101$. La elección de unilateral es correcta (la hipótesis es de orden), pero hay que fijarla ex ante.
</details>

**E5 (recalibración).** Una escala de 8 bandas PDO de 20 puntos, con límites en PD. Se recalibra con δ = +0,25. ¿Cuántos puntos se mueven los cortes en el score sin recalibrar? Si la densidad del score es aproximadamente 1,2% por punto en los 7 cortes, ¿qué fracción de la cartera cambia de banda?

<details><summary>Solución</summary>

$0{,}25\times28{,}854=7{,}21$ puntos. Como $7{,}21<20$ (el desplazamiento es menor que el ancho de banda), cada cliente cruza a lo más un corte, y cambia de banda la masa de las 7 franjas de 7,21 puntos sobre cada corte: $7\times7{,}21\times1{,}2\%\approx60\%$. Cota de cordura: 7 franjas cubren 50 puntos de un score cuya zona densa mide ~80, así que 60% es plausible. Lección: con bandas de 20 puntos, un δ de 0,2–0,4 mueve a la mitad de la cartera (el notebook: 0,376 → 56%). Por eso el diff de ocupación se aprueba antes de firmar una recalibración.
</details>

**E6 (diseño).** Diseña una escala por PD objetivo de 10 bandas entre 0,03% y 35%. Calcula la razón $r$ y el ancho en puntos de la primera y la última banda cerrada. ¿Por qué difieren?

<details><summary>Solución</summary>

$K-1=9$ límites, $r=(0{,}35/0{,}0003)^{1/8}=1.166{,}7^{1/8}=2{,}418$. Límites: 35%, 14,47%, 5,99%, 2,48%, 1,02%, 0,42%, 0,176%, 0,073%, 0,030%. Ancho primera banda cerrada (35% → 14,47%): $\Delta\ln o=\ln(5{,}911/1{,}857)=1{,}158$ → 33,4 puntos. Última (0,073% → 0,030%): $\Delta\ln o=0{,}883$ → 25,5 puntos, prácticamente $\ln r=0{,}883$. Difieren por el término $\ln\frac{1-p_j/r}{1-p_j}$, grande con PD alta y nulo con PD baja (§3.2).
</details>

**E7 (código).** Escribe una función que, dada la tabla (n, malos) por banda de mejor a peor, fusione iterativamente el par adyacente menos separable hasta que todos los pares tengan $p<\alpha$. Aplícala a Austral.

<details><summary>Solución</summary>

```python
def fusionar_hasta_separar(n, malos, nombres, alfa=0.05):
    n, m, nom = list(n), list(malos), list(nombres)
    while len(n) > 1:
        nn, mm = np.array(n, float), np.array(m, float)
        pp = (mm[:-1] + mm[1:]) / (nn[:-1] + nn[1:])
        z = (mm[1:] / nn[1:] - mm[:-1] / nn[:-1]) / np.sqrt(pp * (1 - pp) * (1 / nn[:-1] + 1 / nn[1:]))
        p = norm.sf(z)
        k = int(np.argmax(p))
        if p[k] < alfa:
            break
        n[k:k + 2] = [n[k] + n[k + 1]]; m[k:k + 2] = [m[k] + m[k + 1]]
        nom[k:k + 2] = [nom[k] + "+" + nom[k + 1]]
    return nom, n, m
```

Austral: primero fusiona C1+C2 ($p=0{,}23$), luego A2+B1 ($p=0{,}19$) y se recalcula. Resultado: A1 · A2+B1 · B2 · C1+C2 · D · E (6 grupos); tras las dos fusiones todos los pares quedan con $p\le0{,}007$ (B2 vs A2+B1: 0,005). La escala comercial conserva 8 etiquetas; el backtest y la separabilidad se reportan sobre los 6 grupos.
</details>

**E8 (migración).** Con $\sigma=30$ puntos y bandas de 20 puntos, ¿qué ρ a 6 meses hace falta para que el 60% de los clientes se quede en su banda? Usa la fórmula de §3.8.

<details><summary>Solución</summary>

Se busca $a=w/\sigma_\Delta$ con $(2\Phi(a)-1)-\frac{2}{a}(\varphi(0)-\varphi(a))=0{,}60$. Probando: $a=2$: $0{,}9545-(0{,}39894-0{,}05399)=0{,}6096$; $a=1{,}95$: $0{,}9488-1{,}0256(0{,}39894-0{,}05959)=0{,}6008$. Entonces $\sigma_\Delta\approx20/1{,}95=10{,}3$ y $2(1-\rho)\cdot900=105{,}2$ ⇒ $\rho\approx0{,}94$. Un score conductual con correlación 0,94 a seis meses es exigente; con ρ = 0,85 la permanencia es ~43%. Para una escala de gestión estable, o bandas más anchas o menos bandas.
</details>

**E9 (provisiones).** Una cartera de 10.000 créditos (EAD 1, LGD 45%) está repartida uniformemente en score dentro de la banda B2 = [600, 620). Calcula la EL con PD = punto medio y con PD = media uniforme. ¿Cuál es la correcta y por qué?

<details><summary>Solución</summary>

Punto medio 1,394% ⇒ EL = 10.000 × 0,01394 × 0,45 = 62,7. Media uniforme 1,421% ⇒ 64,0. Con densidad uniforme en score (= uniforme en log-odds), la media de las PD individuales es la uniforme; el punto medio subestima en 1,9% por Jensen. La correcta es la que preserva la suma de PD individuales: la media.
</details>

**E10 (diseño con DP).** Modifica `dp_escala` del notebook para exigir, además, que la PD asignada (media de la banda) caiga dentro de un rango de PD fijo por banda de una escala corporativa dada. ¿Qué cambia en el estado de la DP?

<details><summary>Solución</summary>

Si los límites corporativos son fijos en PD, la banda de cada bloque ya está determinada (no hay nada que optimizar): la DP pasa a ser un chequeo. La versión interesante es otra: bandas propias del modelo que deben *anidarse* en las corporativas (cada banda del modelo contenida en una corporativa). Se agrega a $v(i,j)$ la condición «los bloques $[i,j)$ caen en la misma banda corporativa» (−∞ si no), sin cambiar el estado. La separabilidad sigue en el estado (k, i, j).
</details>

---

## 11. Referencias

- **Siddiqi, N. (2017).** *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards*, 2.ª ed. Wiley. — Scaling PDO, bandas de score y estrategia; la escala del curso sale de aquí.
- **Anderson, R. (2007).** *The Credit Scoring Toolkit: Theory and Practice for Retail Credit Risk Management and Decision Automation.* Oxford University Press. — Capítulos de calibración y agrupamiento de scores en grados; visión de gestión retail.
- **Thomas, L. C., Crook, J. y Edelman, D. (2017).** *Credit Scoring and Its Applications*, 2.ª ed. SIAM. — Fundamento de la relación score–log-odds y de la validación por grupos.
- **Baesens, B., Rösch, D. y Scheule, H. (2016).** *Credit Risk Analytics: Measurement Techniques, Applications, and Examples in SAS.* Wiley. — Mapeo a escalas de rating, backtesting por grado y matrices de migración.
- **Engelmann, B. y Rauhmeier, R. (eds.) (2011).** *The Basel II Risk Parameters: Estimation, Validation, Stress Testing*, 2.ª ed. Springer. — Capítulos de validación de sistemas de rating (tests binomiales, calibración por grado) (verificar edición).
- **Basel Committee on Banking Supervision (2005).** *Studies on the Validation of Internal Rating Systems*, Working Paper No. 14. — Referencia clásica sobre tests de calibración por grado y sus limitaciones de potencia.
- **Basel Committee on Banking Supervision (2006).** *International Convergence of Capital Measurement and Capital Standards: A Revised Framework* (Basilea II), requisitos mínimos IRB; hoy CRE36 del marco consolidado. — Mínimo de 7+1 grados en no minoristas, concentraciones, *pools* minoristas (verificar numeración de párrafos).
- **Reglamento (UE) 575/2013 (CRR), arts. 169–170.** — Estructura de sistemas de rating; escala continua en estimación directa.
- **EBA (2017).** *Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures* (EBA/GL/2017/16). — Calibración a nivel de grado o de segmento; MoC sobre la master scale.
- **ECB (2019).** *Instructions for reporting the validation results of internal models – IRB Pillar I models for credit risk.* — Índice de Herfindahl normalizado y su test, test de Jeffreys por grado, tests sobre la matriz de migración y MWB (verificar versión vigente).
- **CMF, Compendio de Normas Contables para bancos, capítulo B-1.** — Categorías regulatorias A1–A6, B1–B4, C1–C6 con PD de referencia para evaluación individual (verificar vigencia de los valores).
- **Fisher, W. D. (1958).** «On grouping for maximum homogeneity». *Journal of the American Statistical Association*, 53(284), 789–798. — La programación dinámica para particiones óptimas de datos ordenados.
- **Hanson, S. y Schuermann, T. (2006).** «Confidence intervals for probabilities of default». *Journal of Banking & Finance*, 30(8), 2281–2301. — Intervalos por grado de rating; los de grados adyacentes se superponen con frecuencia.
- **Pluto, K. y Tasche, D. (2005).** «Thinking positively». *Risk*, 18(8), 72–78. — Estimación conservadora de PD en grados sin o con pocos defaults.
- **Tasche, D. (2013).** «The art of probability-of-default curve calibration». *Journal of Credit Risk*, 9(4) (verificar páginas). — Calibración de la curva PD por score y por grado; alternativas a la media.
- **Serie 1 · E2** (PIT vs TTC), **E3** (intervalos para carteras chicas), **E6** (regulación); **Serie 2 · M13** (scaling), **M15** (calibración), **M17** (cutoff sobre la escala), **M19** (backtesting), **M20** (monitoreo y migración).

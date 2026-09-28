# M21 · Implementación: artefacto congelado, contrato de datos y paridad

> **Ficha**
> - **Clases que profundiza:** clase 6, láminas 4–6 («En producción NO existe DEV», «El bug de implementación más caro de la industria», límites de la cadena de hashes); notebook `demo_c6_bases_austral`, secciones 2–7 (congelar el binning, artefacto JSON, motor `puntuar` y paridad, bug A / bug B, contrato de datos, corrida de producción); Lab 3 de Financiera Andes, tareas 7–9.
> - **Prerrequisitos:** Serie 1 · M5 (fábrica de variables como pipeline declarativo), Serie 1 · M6 (missing, especiales, outliers), Serie 1 · M7 (binning/WoE); Serie 2 · M10 (logística sobre WoE), M13 (scaling y scorecard como tabla de datos), M14 (reason codes), M15–M16 (δ y master scale), M18 (swap-set). Prepara M22 (expediente, audit trail, model card).
> - **Archivos del módulo:** `M21_implementacion_artefacto.md` (este documento) · `M21_implementacion_artefacto.py` (notebook Marimo: `marimo edit --sandbox M21_implementacion_artefacto.py`) · `M21_artefacto_v1.0.0.json` (el artefacto que produce el notebook) · `M21_artefacto.schema.json` (su JSON Schema, Draft 2020-12).
> - **Tiempo estimado:** 3–3,5 h (lectura 90 min, notebook 75 min, ejercicios 45 min).

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

La clase 6 abrió con la frase que ordena todo este módulo: **en producción NO existe DEV**. El `binear(x, ref=dev[v])` que usamos desde la clase 2 recalcula los cortes mirando la muestra de desarrollo en cada llamada; en el servidor de scoring esas 3.322 filas de 2024 no están, no deben estar y no van a estar. La regla: *si producción tiene que CALCULAR algo que dependa de la población, ya está mal; producción solo APLICA*. El artefacto congela seis cosas: variables, cortes de binning, WoE, coeficientes, calibración y escala de puntos.

El notebook `demo_c6_bases` lo hizo concreto sobre Banco Austral:

- **Congelar** con `ajustar_bins` (la misma partición que `binear(ref=dev)`, como datos) y `aplicar_bins`. La verificación: 32 de 32 comparaciones idénticas (8 variables × 4 muestras). Los ±∞ se guardan como `null` porque `Infinity` no es JSON válido; `clave_discreta` normaliza «0» y «0.0» para que un NaN que promueve `int64` a `float64` no neutralice una variable discreta.
- **Artefacto**: ~5.463 caracteres de JSON (`allow_nan=False`, `sort_keys=True`) con 8 variables y 41 bins con WoE, coeficientes, escalado (PDO 20, 600 @ 50:1, factor 28,8539, offset 487,1229), calibración (TC 5,42 %, δ = +0,1115), master scale (cortes 540…660, etiquetas E…A1, `right=False`), política (cutoff 560), contrato de datos y `woe_faltante = 0`. Hash SHA-256 `98845d69…`. «Un artefacto que necesita su versión exacta de scikit-learn para abrirse no es un artefacto, es una dependencia disfrazada» (contra `pickle`).
- **Motor `puntuar`**: no toca DEV, no reajusta, no mira el target; **el índice viaja con el score** y los casos con un bin que DEV no vio salen marcados «revisar». Paridad contra el notebook: diferencia máxima de score 1,1·10⁻¹³, de PD 2,2·10⁻¹⁶, bandas idénticas en DEV/HO/OOT/TTD. Segunda prueba: el caso #7 puntuado solo, en un lote de 10 y en los 8.585 de TTD da 632,1990 las tres veces.
- **Bug A vs bug B.** El bug A re-ajusta los cortes con el lote del día y mapea el WoE **por posición**: 587 de 8.585 solicitudes (6,84 %) cambian de decisión, Δ medio +4,33 puntos, máximo 73,2, **0 % de WoE sin mapa**. El bug B re-ajusta y mapea por **etiqueta**: 2.178 (25,37 %) cambian, Δ medio −25,2, 89,3 % de WoE sin mapa. «El bug que se ve se arregla. El que no se ve se paga.» Los 587 del bug A estaban, en promedio, a 10,0 puntos del cutoff; el resto, a 60,7.
- **Contrato de datos**: dos severidades fijadas por la **magnitud**: 🔴 bloquea (la corrida aborta y queda registrado) y 🟡 avisa. Tolerancias: fuera de rango 1 % avisa y 10 % bloquea; tope de missing = 3·(%missing DEV) + 1 %, bloquea sobre min(4·tope, 25 %); categorías nuevas por el mismo 10 %. En la bandeja TTD real apareció un **3,1 % de `antiguedad_meses` fuera de [6, 344]** (🟡), a pesar de un CSI bajo (0,002): contrato y CSI miran cosas distintas. El lote roto de laboratorio (`uso_linea × 1000`, feed a la mitad, columna que no llegó) produjo 3 🔴 + 1 🟡: **ninguno de los tres desastres lanza una excepción de Python**.
- **Corrida sellada**: 8 eventos encadenados, aprobación TTD 74,6 %, PD media calibrada 6,31 %, 0 casos a revisión; el aborto del lote roto también queda en el trail (la cadena de hashes, el sello externo y el lineage son materia de M22).

El Lab 3 (tareas 7–9) replica todo sobre Financiera Andes y pide tres justificaciones que este módulo responde con más herramientas: 7d, *¿por qué la paridad debe ser exacta y qué buscaría primero si diera 0,3 puntos?*; 8c, *si la bandeja no tiene hallazgos, ¿la población no se movió o los topes son muy anchos?*; 9b, *qué NO puede prometer un trail sin ancla externa*.

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **La anatomía completa del artefacto**: el del curso no trae la tabla de puntos, ni el mapa de reason codes, ni la convención de cierre de los intervalos, ni la política explícita para lo no visto, ni su propio hash dentro, ni versión de esquema. Tampoco se discutió un **JSON Schema** para validarlo ni los estándares de intercambio (PMML, ONNX).
2. **La semántica de bordes** $(a,b]$ vs $[a,b)$: se heredó de `pd.cut` sin declararla. En datos con cuantiles que caen sobre valores observados (enteros, datos redondeados) es la fuente de bugs de paridad más barata de cometer.
3. **Por qué la paridad «exacta» no es igualdad de bits** entre caminos de cómputo distintos, y dónde sí debe serlo.
4. **Por qué el bug A es silencioso** en un sentido formal: qué propiedad del motor viola, por qué la prueba de permutación no lo ve, y cuánto daño hace en función de la densidad del score en el corte.
5. **Tests de propiedades, golden files y versionado semántico**: el curso hizo dos pruebas puntuales (paridad y caso #7); no hubo batería de CI, ni regla para decidir si un cambio es patch, minor o major, ni *shadow mode*, ni *rollback*.
6. **La base estadística de los umbrales del contrato**: 1 %/10 %/25 % son convenciones. No se vio qué fracción fuera de rango es *esperable sin drift* (2/(n+1) para el min–máx de DEV) ni que un rango min–máx es ciego a desplazamientos del cuerpo de la distribución.
7. **Dominio de negocio vs rango observado**, fallas **de fila** vs fallas **de lote**, identidad duplicada, reglas cruzadas entre campos.
8. **Paridad de la fábrica de variables** (*training-serving skew*): el artefacto congela el modelo, no el cálculo de `uso_linea_prom_12m`. Se menciona aquí y se retoma en el integrador C2.

---

## 2. Intuición

**El modelo es un archivo; el motor es una función.** Todo lo que el curso llamó «modelo» durante cinco clases — cortes, WoE, β, δ, factor, offset, bandas, cutoff — es una colección de números que se estimaron una vez. Desplegar es separar dos cosas que en el notebook estaban mezcladas: **los parámetros** (datos, versionados, con hash) y **la función que los aplica** (código, sin estado, testeable). Si el motor necesita algo que no está en el archivo, el diseño está mal: o el motor está estimando (bug A) o el archivo está incompleto.

**Por qué el bug A no lo ve nadie.** Todos los controles habituales miran propiedades *agregadas* del lote: nulos, tipos, WoE sin mapa, distribución del score. El bug A produce un lote perfectamente normal: cada cliente cae en algún bin, cada bin tiene un WoE, el score tiene una distribución verosímil. Lo que está mal es una propiedad **de cada cliente**: su score depende de quién más vino ese día. Para verlo hay que hacer la pregunta al revés: *¿qué score le doy a este cliente si viene solo?* Esa es la prueba de subconjunto, y es la única de la batería estándar que lo mata.

**Por qué el daño se concentra en el corte.** Un error de ±5 puntos en un cliente con score 620 no cambia nada si el cutoff es 530. El mismo error en un cliente de 532 lo rechaza. El porcentaje de decisiones que cambia es, a primer orden, *densidad del score en el cutoff × tamaño medio del error*. Por eso los 587 de Austral estaban a 10 puntos del corte: el bug se come exactamente la franja donde el modelo aporta más valor.

**Los bordes son parte del modelo.** Un corte en `consultas_6m = 1` no dice si el cliente con exactamente una consulta está en el bin de arriba o el de abajo; lo dice la convención de cierre. Con datos enteros, la mitad de la cartera puede estar exactamente sobre un corte. Dos implementaciones «correctas» del mismo artefacto — una en pandas con `pd.cut`, otra en SQL con `>=` — pueden diferir en el 12 % de las decisiones.

**El contrato no es un test del modelo, es un test del mundo.** El modelo no distingue un dato bueno de uno malo. `uso_linea × 1000` es aritmética válida; la mitad del feed en NaN es un DataFrame válido. El contrato traduce «lo que el modelo vio en desarrollo» y «lo que el negocio considera posible» a reglas con severidad, y la severidad la decide la **magnitud**: un 3 % fuera de rango es deriva; un 100 %, que alguien cambió la unidad.

**La versión la decide la prueba, no la opinión.** Quien redondea los cortes «para que el anexo se lea bien» cree que hizo un cambio cosmético (patch). El golden file dice otra cosa: el orden de los clientes cambió, y eso es un cambio mayor. Versionado semántico aplicado a modelos = derivar el incremento del diff de la salida observable.

---

## 3. Formalización

### 3.1 El artefacto y el motor

Sea $\theta$ el artefacto. Para cada variable $v=1,\dots,k$: una partición $\mathcal{P}_v$ del dominio en bins, un WoE $w_{v,b}$ por bin, un coeficiente $\beta_v$; además $\beta_0$, $\delta$, factor $f=\text{PDO}/\ln 2$, offset $o=s_0-f\ln(\text{odds}_0)$, cortes de master scale $m_1<\dots<m_{J}$ y cutoff $c$. El motor es

$$
S_\theta(x)=o-f\,\eta_\theta(x),\qquad \eta_\theta(x)=\beta_0+\sum_{v=1}^{k}\beta_v\,w_{v,\,b_v(x_v)}+\delta,
$$

$$
\text{PD}_\theta(x)=\frac{1}{1+e^{-\eta_\theta(x)}},\qquad D_\theta(x)=\mathbb{1}[S_\theta(x)\ge c],
$$

donde $b_v(\cdot)$ asigna el bin. Con la tabla de puntos de M13, $p_{v,b}=-(\beta_v w_{v,b}+\beta_0/k)f+o/k$, se tiene la identidad

$$
\sum_{v}p_{v,b_v(x_v)}=o-f\Big(\beta_0+\sum_v\beta_v w_{v,b_v}\Big)=S_\theta(x)+f\,\delta ,
$$

es decir $S_\theta=\sum_v p_{v,b_v}-f\delta$: producción puede puntuar sumando la tabla (el notebook lo verifica con error $2{,}3\cdot10^{-13}$). Nótese que el motor calcula el score **desde $\eta$**, no desde la PD: $o-f\,\text{logit}(\text{PD})$ exige recortar la PD a $[10^{-12},1-10^{-12}]$ (como hace el curso) para no producir $\pm\infty$, lo que acota el score a $o\pm f\cdot 27{,}6\approx[-310,\,1284]$; desde $\eta$ no hay recorte.

**Definición (motor de lote).** Un motor es una función $M$ que recibe un lote $L=(x_1,\dots,x_m)$ y devuelve $(s_1,\dots,s_m)$. Decimos que $M$ es:

- *equivariante a permutaciones* si $M(L_\pi)=M(L)_\pi$ para toda permutación $\pi$;
- *local* si existe $g$ tal que $M(L)_i=g(x_i)$ para todo $L$ e $i$.

**Proposición 1.** $M$ es local si y solo si es *invariante a subconjuntos*: $M(S)_i=M(L)_i$ para todo $S\subseteq L$ con $x_i\in S$.

*Demostración.* (⇒) Si $M(L)_i=g(x_i)$, el valor no depende de $L$. (⇐) Tomando $S=\{x_i\}$ se define $g(x):=M(\{x\})$, y la invariancia a subconjuntos da $M(L)_i=M(\{x_i\})=g(x_i)$. ∎

**Proposición 2 (por qué el bug A es invisible a la prueba de permutación).** El bug A es $M_A(L)_i=S_{\theta(\hat Q_L)}(x_i)$, donde $\hat Q_L$ son los cuantiles empíricos del lote. Como $\hat Q_L$ es una función simétrica de la distribución empírica de $L$, $\hat Q_{L_\pi}=\hat Q_L$ y $M_A$ es equivariante a permutaciones. Pero no es local: si para algún $S\subset L$ se tiene $\hat Q_S\neq\hat Q_L$ y algún $x_i\in S$ cae en bins distintos bajo ambos, $M_A(S)_i\ne M_A(L)_i$. La prueba de permutación tiene **potencia cero** contra el bug A; la de subconjunto (incluido el subconjunto de tamaño 1, «puntuar fila a fila») tiene potencia uno en cuanto los cuantiles cambian. En el notebook: el bug A pasa permutación y pureza, y falla subconjunto con 27,2 puntos de diferencia para el mismo cliente.

### 3.2 Semántica de bordes

Sean $c_1<\dots<c_{K-1}$ los cortes interiores y $c_0=-\infty$, $c_K=+\infty$. Las dos convenciones son

$$
b^{]}(x)=\#\{j\ge 1: c_j<x\}\quad\text{(intervalos }(c_{j-1},c_j]\text{)},\qquad
b^{[}(x)=\#\{j\ge 1: c_j\le x\}\quad\text{(intervalos }[c_{j-1},c_j)\text{)}.
$$

$b^{]}$ es exactamente `np.searchsorted(c, x, side="left")` y `pd.cut(..., right=True)`; $b^{[}$ es `side="right"`, `np.digitize(x, c)` (por defecto, `right=False`) y `pd.cut(..., right=False)`. Las dos asignaciones difieren **si y solo si** $x\in\{c_1,\dots,c_{K-1}\}$, así que la fracción de casos afectados es

$$
\Pr\big(b^{]}(X)\ne b^{[}(X)\big)=\sum_{j=1}^{K-1}\Pr(X=c_j).
$$

Para una variable continua esto es cero; para una variable entera o redondeada no lo es. ¿Por qué un corte coincide con un valor observado? El cuantil «lineal» de numpy y pandas (tipo 7 de Hyndman y Fan) con $n$ observaciones ordenadas $x_{(1)}\le\dots\le x_{(n)}$ es

$$
\hat Q(q)=x_{(\lfloor h\rfloor)}+\big(h-\lfloor h\rfloor\big)\big(x_{(\lfloor h\rfloor+1)}-x_{(\lfloor h\rfloor)}\big),\qquad h=(n-1)q+1 .
$$

Si $x_{(\lfloor h\rfloor)}=x_{(\lfloor h\rfloor+1)}$ (empate, casi seguro con datos enteros o redondeados a 4 decimales y $n$ grande), $\hat Q(q)$ **es** un valor observado, sin importar la parte fraccionaria. En el generador, `consultas_6m` (Poisson) tiene cortes en 1 y 2 y $\Pr(X=1)+\Pr(X=2)\approx 51{,}4\,\%$; `meses_desde_mora_12m`, 22,2 %; `antiguedad_meses`, 3,7 %. Cambiar la convención de cierre cambia el **11,6 % de las decisiones** del TTD.

**Redondeo de cortes.** Si un corte $c$ se publica como $\tilde c$, cambian de bin los casos con $x$ entre ambos: para una variable continua con densidad $f_X$, la fracción es $\approx f_X(c)\,|c-\tilde c|\le f_X(c)\cdot\tfrac12 10^{-k}$ al redondear a $k$ decimales. Para datos en una grilla de paso $\Delta$, el daño es exactamente cero cuando $\tilde c$ y $c$ quedan en la misma celda de la grilla — por eso en el notebook desaparece desde $k=4$: es una propiedad de *estos datos*, no del artefacto. Obsérvese además que el corte que un reporte muestra como «0,19672» es en realidad `0.19672000000000006` (artefacto de la interpolación en coma flotante): quien lo transcriba a 5 decimales mueve de bin a cualquier cliente con exactamente 0,19672.

**Precisión de coma flotante.** Un `float64` tiene 53 bits de significando; 17 dígitos decimales significativos bastan para que la ida y vuelta texto→número sea exacta, y Python (desde 3.1) escribe con `repr` la cadena **más corta** que hace la ida y vuelta. Por eso el round-trip JSON del artefacto es bit a bit idéntico (el notebook lo verifica). Un `float32` tiene 24 bits: error relativo hasta $2^{-24}\approx 6\cdot10^{-8}$. Un campo de hoja de cálculo muestra 15 dígitos significativos. Ambos pueden mover de bin un caso que esté exactamente sobre un corte.

### 3.3 Cuánto daño hace un error de score: la fórmula del bug A

Sea $S$ el score correcto, con densidad $f_S$, y $S+\Delta$ el score con error. La decisión cambia si y solo si $(S-c)$ y $(S+\Delta-c)$ tienen distinto signo (con la convención $S\ge c$ aprueba). Condicional a $\Delta=\delta>0$, eso ocurre para $S\in[c-\delta,c)$:

$$
\Pr(\text{cambia}\mid\Delta=\delta)=\int_{c-\delta}^{c}f_S(s)\,ds\;\approx\;f_S(c)\,\delta ,
$$

y simétricamente para $\delta<0$ con $S\in[c,c+|\delta|)$. Si $\Delta$ es aproximadamente independiente de $S$ cerca del corte y $f_S$ es casi constante en la franja $[c-|\Delta|,c+|\Delta|]$,

$$
\boxed{\;\Pr(\text{cambia})\approx f_S(c)\;\mathbb{E}|\Delta|\;}
$$

Además, condicional a $\Delta=\delta$ y a que cambie, $S$ es aproximadamente uniforme en la franja, así que $\mathbb{E}[|S-c|\mid\text{cambia},\Delta=\delta]=|\delta|/2$; ponderando por la probabilidad de cambio ($\propto|\delta|$):

$$
\mathbb{E}\big[\,|S-c|\;\big|\;\text{cambia}\big]\approx\frac{\mathbb{E}[\Delta^2]}{2\,\mathbb{E}|\Delta|}.
$$

En el notebook (TTD completo, bug A): $f_S(c)\approx0{,}0086$ por punto, $\mathbb{E}|\Delta|=2{,}17$ ⇒ predicción 1,86 % vs 2,23 % observado; distancia media predicha 4,5 puntos vs 4,6 observada. En Austral, la distancia media de 10,0 puntos y $\mathbb{E}|\Delta|$ más grande son coherentes con un 6,8 %. La fórmula explica tres hechos: (i) el daño es proporcional a cuánta cartera vive en el corte (una política con cutoff en la moda del score sufre más), (ii) los errores grandes y raros hacen menos daño por punto que los chicos y frecuentes, (iii) cualquier error sistemático de score — un δ mal aplicado, un redondeo — se traduce en decisiones con la misma fórmula (§3.5).

### 3.4 El contrato: qué es esperable sin drift

**Rango min–máx de DEV.** Si $X_1,\dots,X_n$ (DEV) y $X_{n+1}$ (nuevo) son intercambiables y continuas, cada una tiene la misma probabilidad de ser el máximo, así que $\Pr(X_{n+1}>\max_{i\le n}X_i)=1/(n+1)$ y, por simetría, $\Pr(X_{n+1}\notin[\min,\max])=2/(n+1)$. Con $n=3.322$ (Austral) es 0,06 %; con $n=10.065$ (generador), 0,02 %. El 3,1 % de Austral es ~50 veces lo esperado; el TTD del generador (0,00–0,06 %) está dentro.

**Distribución del conteo.** Para un lote de $m$ casos, el conteo $K$ fuera de rango **no** es binomial: todos comparten el mismo mín y máx de DEV. Condicional a DEV, cada caso cae fuera con probabilidad $p=F(X_{(1)})+1-F(X_{(n)})$. Con $U_{(i)}=F(X_{(i)})$ estadísticos de orden uniformes, los $n+1$ espaciamientos siguen una Dirichlet$(1,\dots,1)$ y la suma de dos de ellos es $\text{Beta}(2,n-1)$. Entonces $K\sim$ beta-binomial$(m,2,n-1)$, con

$$
\mathbb{E}K=\frac{2m}{n+1},\qquad \operatorname{Var}K=m\,\bar p(1-\bar p)\Big[1+\frac{m-1}{n+2}\Big],\quad \bar p=\frac{2}{n+1}.
$$

Con $m=4.848$ y $n=10.065$ la varianza es 1,48 veces la binomial: el p-valor binomial es **anti-conservador**. El notebook muestra ambos.

**Rango por cuantiles.** Si el rango es $[\hat Q(\alpha),\hat Q(1-\alpha)]$, la fracción esperada fuera es $\approx2\alpha$ (menos si hay empates en un extremo: `antiguedad_meses` tiene su p0,5 en el mínimo, 6). La lección del notebook: con el min–máx del curso, envejecer la antigüedad 60 meses no dispara nada (el generador recorta la antigüedad en 360 y el máximo de DEV *es* el borde del dominio); con cuantiles 0,5–99,5 %, a los 36 meses el 1,9 % queda fuera contra ~0,5 % esperado y el contrato avisa. **Un min–máx detecta roturas de cola; un rango por cuantiles detecta desplazamientos del cuerpo**, pero exige medir el *exceso sobre lo esperado*, no la fracción bruta. El notebook implementa la severidad como exceso: avisa si $\hat p-p_0>1\,\%$, bloquea si $\hat p-p_0>10\,\%$.

**Missing.** Con el tope del curso $\tau=3p_{\text{DEV}}+0{,}01$ y bajo la hipótesis de que el lote tiene la misma tasa, $\hat p\sim\text{Bin}(m,p)/m$ con desviación $\sqrt{p(1-p)/m}$. Para `renta_mm` ($p=3{,}98\,\%$, $\tau=12{,}95\,\%$) y $m=4.848$ la desviación es 0,28 %: el tope está a 32 desviaciones. El contrato del curso está diseñado para **fallas gruesas**, no para detectar deriva sutil del missing (eso es trabajo del tablero, M20). Para lotes de 50 casos la desviación es 2,8 % y el mismo tope ya no es tan holgado: los umbrales deberían depender de $m$.

**Severidad por magnitud, formalizada.** Para una regla $r$ con tasa observada $\hat p_r$ y tasa esperada $p_{0,r}$, la severidad es una función escalonada del exceso $e_r=\hat p_r-p_{0,r}$: 🟢 si $e_r\le a_r$, 🟡 si $a_r<e_r\le b_r$, 🔴 si $e_r>b_r$. Los umbrales $a_r,b_r$ son **decisiones de negocio** (costo de abortar una corrida vs costo de puntuar basura); lo que la estadística aporta es $p_{0,r}$ y la dispersión de $\hat p_r$ bajo la nula, para que $a_r$ no dispare todos los días por ruido.

### 3.5 Versionado semántico como clase de transformaciones

Sea $S$ el score de la versión $A$ y $S'$ el de la versión $B$ sobre el mismo lote (golden + referencia). La convención de la serie:

- **PATCH** ⇔ $S'=S$ bit a bit y bandas, decisiones y reason codes idénticos.
- **MINOR** ⇔ $S'=S$ pero bandas/decisiones distintas (cutoff, master scale), **o** $S'=S+\kappa$ con $\kappa$ constante (recalibración de δ).
- **MAJOR** ⇔ cualquier otra cosa.

La recalibración es MINOR porque $S'=o-f(\eta+\delta')=S-f(\delta'-\delta)$: una traslación. Toda métrica de ranking (AUC, Gini, KS, curva CAP, M12) es invariante a transformaciones estrictamente crecientes, en particular a traslaciones; cambian el nivel de PD y las decisiones en la franja $[c,\,c+f(\delta'-\delta))$, cuya masa, por el mismo argumento de §3.3, es $\approx f_S(c)\,f\,|\delta'-\delta|$. En el notebook: δ pasa de 0,1248 a 0,3375 (recalibración a la tasa OOT de 15,40 %), los scores bajan 6,14 puntos, y la predicción $0{,}0086\times6{,}14\approx5{,}3\,\%$ calza con el swap-out observado de 272/4.848 = 5,6 %.

Un cambio de PDO es una transformación creciente pero **no** una traslación: el ranking se conserva y, sin embargo, cambia el significado de cada punto para todo consumidor del score (cutoff, reportes, reglas de negocio). Por eso la convención lo trata como MAJOR: es un cambio incompatible de la «API» del modelo. Lo mismo un cambio de ancla (600 @ 50:1). Esto es convención, no teorema; lo importante es que la regla esté escrita y que la aplique una prueba.

**Cobertura.** La clasificación es tan buena como el lote que la decide. Si un bin no aparece en el golden, un cambio en su WoE produce $S'=S$ en todo el golden y pasa como PATCH. Invariante: **cobertura de bins = 100 %** (cada bin de cada variable con al menos un caso). El notebook construye el golden así (202 casos: 15 casos borde + uno por bin + 150 al azar).

### 3.6 Identidad del artefacto: hash canónico

El hash identifica el contenido si la serialización es una función **canónica**: igual contenido ⇒ igual texto. `json.dumps` sin `sort_keys` depende del orden de inserción de las claves; con separadores por defecto (`", "`, `": "`) depende del estilo. La canónica del notebook es `sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False`, y el hash se calcula sobre el artefacto **sin** su campo `integridad` (un hash no puede contenerse a sí mismo). Para que un motor en otro lenguaje recalcule el mismo hash hace falta además fijar la escritura de números: Python escribe `1e-07` y `1e+16` donde ECMAScript escribe `1e-7` y `10000000000000000`. El RFC 8785 (JSON Canonicalization Scheme, JCS) estandariza exactamente eso; si el hash debe verificarse fuera de Python, conviene adoptarlo o, más simple, verificar el hash de los **bytes del archivo** tal como se publicó, no de su re-serialización.

### 3.7 Paridad: qué significa «exacta»

La suma en coma flotante no es asociativa. Para $k$ términos, el error de la suma calculada satisface $|\text{fl}(\sum x_i)-\sum x_i|\le (k-1)\,u\sum|x_i|+O(u^2)$ con $u=2^{-53}\approx1{,}1\cdot10^{-16}$. Con 9 términos de magnitud total ~700 puntos, la cota es ~$7\cdot10^{-13}$: la diferencia de $1{,}1\cdot10^{-13}$ entre el motor (suma variable a variable desde $\eta$) y el notebook (statsmodels hace `X @ β` y luego `logit(expit(·))`) es exactamente el tamaño esperado de **dos caminos de cómputo distintos**. De ahí la regla:

- Entre caminos distintos (notebook vs motor, Python vs SQL): **tolerancia declarada** (aquí $10^{-9}$ puntos) **y** decisiones, bandas y reason codes idénticos. Una diferencia de 0,3 puntos (Lab 3, 7d) no es redondeo: es 10¹² veces la cota; es un bug (orden δ↔PDO, un bin distinto, un WoE redondeado) y no se despliega.
- Dentro del mismo camino (mismo artefacto, mismo motor, otra máquina u otro día; artefacto en memoria vs releído del JSON; numpy vs `pd.cut` sobre los mismos WoE): **igualdad de bits**, verificable con un hash de la salida.

---

## 4. Variantes y alternativas de industria

### 4.1 Formatos del artefacto

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién lo usa / regulación |
|---|---|---|---|---|
| **JSON propio + JSON Schema** (este módulo, curso) | Legible por humanos y máquinas; diff en git; hash; validable por esquema; cualquier lenguaje lo lee | El motor hay que escribirlo (y testearlo) en cada plataforma; semántica (bordes, no vistos) solo en la documentación y los tests | Scorecards y modelos lineales; equipos que controlan su motor | Práctica común en fintech; el expediente del curso lo exige como pieza 5 |
| **PMML `Scorecard`** (DMG) | Estándar XML con `Characteristic`/`Attribute`, predicados (`lessOrEqual`, `isMissing`), `baselineScore`, `reasonCode`, `reasonCodeAlgorithm` (`pointsBelow`/`pointsAbove`) | Los atributos se evalúan **en orden** y gana el primero que calza; si ninguno calza el score es inválido; soporte desigual entre motores; XML verboso | Motores de decisión que ya consumen PMML; intercambio con proveedores | Estándar abierto del Data Mining Group (versiones 4.1+ incluyen Scorecard) |
| **ONNX / ONNX-ML** | Grafo de cómputo portable con runtimes optimizados (ONNX Runtime) | Opaco para un comité (protobuf binario); el binning hay que expresarlo con operadores genéricos y su semántica de bordes depende del conversor | Modelos no lineales (GBM, redes) que deben correr fuera de Python | Industria ML general; poco habitual para scorecards |
| **`pickle` / `joblib`** | Cero trabajo: se guarda el objeto Python | Dependiente de la versión exacta de las librerías; ejecutar un pickle es ejecutar código arbitrario; ilegible; no hay diff | Nunca como artefacto de producción de un modelo regulado; a lo sumo caché local | El curso lo desaconseja explícitamente |
| **SQL generado** (`CASE WHEN`) | Puntúa dentro del data warehouse, sin servicio | El generador es un segundo motor que también hay que testear (bordes `<` vs `<=`, NULL) | Scoring batch masivo, cartera vigente | Frecuente en banca para modelos de comportamiento |
| **Tabla en motor de reglas** (p. ej., motores comerciales de decisión — verificar capacidades de cada producto) | El negocio edita cutoffs y políticas sin despliegue | Riesgo de editar también los puntos; trazabilidad depende del motor | Originación con políticas cambiantes | Bancos grandes; exige control de cambios |
| **Registro de modelos** (MLflow u otro) | Versionado, etapas (staging/producción), linaje | Otra plataforma que operar; el «modelo» sigue siendo un pickle salvo que se registre el JSON | Equipos con muchos modelos | Industria ML |

### 4.2 Estrategias de despliegue

| Método | Qué resuelve | Costo | Cuándo usarlo | Quién / regulación |
|---|---|---|---|---|
| **Big bang** | Simplicidad | Sin evidencia previa en producción; rollback con clientes ya decididos | Nunca para un modelo nuevo; aceptable para PATCH | — |
| **Shadow mode** | Evidencia en datos reales sin riesgo: el challenger puntúa, se registra, no decide | Infraestructura doble; el challenger no genera desempeño propio (solo swap-set) | Toda versión MINOR/MAJOR antes de promover | Buena práctica de gestión de riesgo de modelos |
| **Champion/challenger** (A/B) | Mide desempeño del challenger en una fracción aleatoria | Clientes decididos por un modelo no aprobado para el 100 %; espera de 12 meses para el target | Estrategias y políticas (cutoff, oferta), más que el score | Tarjetas y consumo en banca desde hace décadas |
| **Canary** | Limita el daño de un error de implementación | Muestra chica: no mide desempeño de riesgo, solo operación | Cambios de motor/infraestructura | Ingeniería de software |
| **Blue-green + puntero** | Rollback instantáneo (mover el puntero a un hash anterior) | Exige artefactos inmutables y verificación de hash al cargar | Siempre, como mecanismo de rollback | — |

### 4.3 Herramientas de contrato de datos

| Herramienta | Qué resuelve | Costo | Cuándo usarla |
|---|---|---|---|
| **numpy/pandas propio** (notebook) | Control total; reglas de lote (tasas, excesos) y de fila; cero dependencias | Hay que escribir y testear cada regla | Contratos chicos, motores críticos, reglas estadísticas |
| **pydantic** | Validación por **registro** con tipos y validadores; errores estructurados; genera JSON Schema de los modelos | Fila a fila: no expresa reglas de lote; más lento en lotes grandes | APIs de originación en línea (un cliente por request) |
| **jsonschema** | Valida documentos JSON (el artefacto, o cada solicitud) contra un estándar independiente del lenguaje | Solo estructura; no expresa invariantes cruzados | El artefacto; contratos entre equipos/lenguajes |
| **pandera** | Esquemas declarativos para DataFrames con checks por columna y por DataFrame, incluidos estadísticos | Dependencia adicional (no instalada en este entorno) | Pipelines batch en pandas |
| **Great Expectations** | «Expectativas» declarativas, documentación generada, perfiles | Pesado de operar | Plataformas de datos con muchos equipos |
| **TFDV / Deequ** | Validación a escala (esquemas inferidos, detección de skew/drift) | Ecosistemas TensorFlow / Spark | Volúmenes grandes |

---

## 5. Cuándo falla: trampas y modos de falla

**5.1 Bug A: re-ajustar el binner en producción.** *Síntoma:* ninguno en los logs; tasa de aprobación que «respira» día a día sin cambio de población; clientes idénticos con decisiones distintas según la fecha. *Causa:* el motor estima cortes con el lote (Proposición 2). *Detección:* prueba de subconjunto (incluido tamaño 1) en CI; en producción, re-puntuar una muestra fija diaria («clientes centinela») y exigir scores idénticos. *Qué hacer:* cortes solo desde el artefacto. En el generador: 2,2 % de decisiones en TTD completo, 1,7 % con lotes de 1.000 sin drift, 3,4 % con lotes de 250, hasta 29 % con drift fuerte.

**5.2 Bug B: mapear WoE por etiqueta de texto.** *Síntoma:* WoE sin mapa (64,5 % en el generador, 89 % en Austral), scores concentrados, alertas. *Causa:* etiquetas de `pd.cut` regeneradas con otros cortes (y con `precision=3`). *Detección:* contador de no vistos por variable. *Qué hacer:* bins por **id** entero, nunca por etiqueta.

**5.3 Convención de cierre no declarada.** *Síntoma:* paridad falla en un 5–15 % de casos concentrados en valores redondos. *Causa:* $(a,b]$ en Python, `>=`/`<` en SQL, `np.digitize` por defecto ($[a,b)$). *Detección:* casos borde exactamente en cada corte en el golden. *Qué hacer:* `"cierre"` como campo del artefacto; el test de casos borde en ambos motores.

**5.4 Cortes truncados.** *Síntoma:* diferencias en pocos casos, solo en variables continuas. *Causa:* cortes copiados de un reporte, redondeados, pasados por `float32` o por una planilla (15 dígitos). *Detección:* validación semántica (cortes del motor = cortes del artefacto con `==`), golden. *Qué hacer:* publicar cortes con `repr` completo; el anexo para humanos se genera del artefacto, nunca al revés.

**5.5 Códigos especiales tratados como números.** *Síntoma:* −99 («sin bureau») comparte bin con «mora hace 1–3 meses»; un cambio de código (−99 → −1) cambia el bin sin error. *Causa:* binning por cuantiles sobre la columna cruda (el binner del curso lo hace). *Detección:* casos borde con cada código especial; contrato con `especiales` declarados. *Qué hacer:* bins `valor` propios para cada código (M14 muestra el efecto en Gini); nunca dejar que un código caiga en un intervalo.

**5.6 Claves de tipo inestable.** *Síntoma:* una variable discreta queda entera en WoE 0. *Causa:* `int64` en DEV, `float64` en producción (basta un NaN) ⇒ «0» vs «0.0» (el `clave_discreta` del curso). *Detección:* % no vistos por variable. *Qué hacer:* normalizar claves en el artefacto y en el motor; contrato de tipos.

**5.7 Categorías que no son las mismas.** *Síntoma:* un canal entero a revisión (motor con marcas) o a WoE 0 en silencio (motor del curso). *Causa:* `'App'` vs `'app'`, espacios, tildes, renombres en el front. *Detección:* regla `categoria_nueva` por magnitud. *Qué hacer:* normalización declarada en el contrato (minúsculas, `strip`) aplicada antes del motor, y mapa de sinónimos versionado.

**5.8 WoE «neutro» no es neutro.** *Síntoma:* un caso con categoría no vista **sube** 5 puntos (en el notebook, `canal='App'` pasa de 582,5 a 587,6). *Causa:* WoE 0 corresponde al riesgo promedio de DEV, no al riesgo de ese cliente; si su categoría real era de las malas, WoE 0 lo mejora. *Detección:* casos borde. *Qué hacer:* no visto ⇒ revisión, no aprobación automática (el curso lo hace); si el volumen lo impide, política explícita (peor bin) aprobada por comité.

**5.9 NaN e Infinity en el JSON.** *Síntoma:* el motor Java rechaza el artefacto, o lo lee con un parser permisivo que convierte `NaN` en otra cosa. *Causa:* `json.dumps` por defecto escribe `NaN`; `json.loads` por defecto lo lee. *Detección:* `allow_nan=False` al escribir, `parse_constant` que falla al leer. *Qué hacer:* ambas cosas, más el JSON Schema, que por sí solo **no** atrapa un NaN en el objeto Python.

**5.10 Artefacto como pickle.** *Síntoma:* el modelo «no abre» tras actualizar scikit-learn; o abre y da otros números. *Causa:* el pickle guarda el objeto, no los parámetros. *Detección:* imposible de auditar a simple vista. *Qué hacer:* JSON (o PMML) con los parámetros; el pickle, si existe, es caché.

**5.11 Identidad perdida.** *Síntoma:* scores correctos asignados a clientes equivocados; métricas agregadas intactas. *Causa:* el motor devuelve un array y alguien lo asigna por posición a un DataFrame reordenado o filtrado. *Detección:* test «salida.index == lote.index» y permutación con re-alineación por índice; contrato de identidad única (`id_duplicado` 🔴). *Qué hacer:* el índice viaja con el score (curso) y los joins son por id, nunca por posición.

**5.12 Cambio de unidad aguas arriba.** *Síntoma:* aprobación cae de 75,4 % a 62,8 % (notebook, `uso_linea × 1000`) sin excepción. *Causa:* porcentaje vs fracción, pesos vs miles de pesos, UF vs pesos. *Detección:* dominio de negocio (fila) y rango de DEV (lote). *Qué hacer:* bloquear; nunca «reescalar automáticamente».

**5.13 Feed parcial con missing «legítimo».** *Síntoma:* 0,19 % de decisiones cambian en silencio (notebook, `renta_mm` con la mitad del feed caído). *Causa:* el missing de esa variable existía en DEV, así que el motor le asigna, con toda legitimidad, el WoE de MISSING. *Detección:* **solo** el contrato (50 % vs tope 12,95 %). *Qué hacer:* bloquear por magnitud; este es el caso que justifica el contrato aunque el motor sea impecable.

**5.14 Paridad del modelo sin paridad de las variables.** *Síntoma:* paridad perfecta en el golden y, sin embargo, PSI del score alto desde el primer día. *Causa:* *training-serving skew*: producción calcula `uso_linea_prom_12m` con otra ventana (t₀ en vez de t₀−1), otra fuente o otro tratamiento de meses faltantes. *Detección:* recalcular las variables de una muestra de solicitudes con el pipeline de desarrollo y comparar con las que recibió el motor (paridad de la fábrica). *Qué hacer:* la fábrica de variables también es un artefacto versionado (Serie 1 · M5); el contrato registra su versión.

**5.15 Versión declarada a mano.** *Síntoma:* un «patch» cambia decisiones. *Causa:* el incremento lo decide quien hace el cambio. *Detección:* clasificación por diff de salida (§3.5). *Qué hacer:* el gate de CI bloquea si el incremento declarado es menor que el requerido (notebook: la «v1.0.2» con cortes redondeados, 32 campos distintos, Δ máximo 11,1 puntos, declarada PATCH, requerida MAJOR).

**5.16 Golden sin cobertura.** *Síntoma:* un cambio en un bin raro pasa todos los tests. *Causa:* golden elegido al azar (los bins raros tienen probabilidad baja de aparecer). *Detección:* métrica de cobertura de bins. *Qué hacer:* un caso por bin por variable, más casos borde.

**5.17 Recalcular en producción «para mantenerlo al día».** *Síntoma:* δ o cutoff que cambian solos cada mes. *Causa:* un job que «recalibra automáticamente» con la mora reciente. *Detección:* hash del artefacto en cada corrida (M22). *Qué hacer:* todo parámetro nuevo es una versión nueva, con su gatillo, su dueño y su aprobación (clase 6, lámina 9).

---

## 6. Puente con ingeniería

**Dos cosas versionadas, no una.** El **motor** (código: `puntuar`, validadores, carga) y el **artefacto** (datos). Cada uno con su semver. El artefacto declara `formato.version_esquema`; el motor declara qué versiones de esquema sabe leer. Un motor nuevo se valida corriendo **todos** los golden de todos los artefactos vigentes: si cambia un bit, es un cambio de motor que exige revisión (y no un cambio de modelo).

**Estructura de repositorio (un modelo):**

```
modelos/consumo-scorecard/
  artefactos/
    1.0.0.json            # inmutable; hash en el nombre del objeto en el registro
    1.1.0.json
  golden/
    1.0.0.parquet         # entrada + salida exacta (score, pd, banda, decision, rc1..rc3)
  contrato/
    dominios.yaml         # lo POSIBLE (lo firma el dueño del dato)
    umbrales.yaml         # severidades (lo firma riesgo)
  schema/
    scorecard-artefacto-1.0.0.schema.json
  CHANGELOG.md            # versión, incremento requerido por la prueba, aprobación, gatillo
```

**Contrato declarativo** (lo que el notebook implementa en `derivar_contrato` + `validar_lote`):

```yaml
contrato:
  identidad: {columna: id_solicitud, unica: true}            # id_duplicado → 🔴
  variables:
    uso_linea_prom_12m:
      tipo: numerico
      dominio: {min: 0.0, max: 1.5}                          # POSIBLE: fila fuera → revisión; >1% → 🔴
      rango_dev: {modo: cuantiles, p005: 0.0347, p995: 0.9137, esperado_fuera: 0.0100}
      missing: {dev: 0.0, tope: 0.01, bloqueo: "min(4*tope, 0.25)"}
    meses_desde_mora_12m:
      tipo: numerico
      dominio: {min: 1, max: 13, entero: true, especiales: [-99, -9]}
    canal:
      tipo: categorico
      normalizar: [minusculas, strip]
      categorias: [app, fuerza_venta, sucursal, web]         # nueva: 🟡 ; >10% → 🔴
  reglas:
    - {id: antiguedad_vs_edad, expr: "antiguedad_meses <= 12 * edad", nivel: fila}
  severidad: {medida: exceso_sobre_esperado, aviso: 0.01, bloqueo: 0.10}
```

**Pipeline de CI (cada PR que toca un artefacto, el motor o el contrato):**

1. **Serialización estricta**: el artefacto se relee con `parse_constant` que falla; round-trip bit a bit.
2. **JSON Schema** (Draft 2020-12).
3. **Validación semántica**: contigüidad de bins, extremos `null`, cortes crecientes, `factor = PDO/ln 2`, `offset` consistente con el ancla, etiquetas de master scale = cortes + 1, WoE finitos, puntos = fórmula M13, todo reason code con texto, **hash = contenido**. Avisos (no bloquean): WoE no monótono respecto de la tendencia declarada (el notebook encuentra dos reales: `antiguedad_meses` y `renta_mm`).
4. **Paridad** contra el notebook de desarrollo en DEV/HO/OOT/TTD: $\max|\Delta S|<10^{-9}$, decisiones/bandas idénticas; numpy vs implementación alternativa bit a bit.
5. **Propiedades**: subconjunto (cientos de subconjuntos aleatorios, incluido tamaño 1), permutación con re-alineación por índice, pureza (no muta, determinista), monotonía por variable con tendencia declarada.
6. **Casos borde** con bin esperado escrito a mano (valor en cada corte, siguiente float, especiales, missing visto y no visto, ±inf, categoría con mayúscula y nueva).
7. **Golden**: cobertura de bins 100 %; clasificación del incremento por diff de salida; **gate**: declarado ≥ requerido.
8. **Publicación**: el artefacto se sube al registro con su hash como clave; el golden se guarda junto.

**Invariantes en tiempo de ejecución:**

- Al cargar: `sha256(canónico(artefacto sin integridad)) == integridad.hash == hash registrado para esa versión`. Si no, **no carga** (el notebook lo demuestra con un artefacto editado en el registro).
- Por corrida: contrato → (si 🔴: abortar y registrar) → motor → filas con falla de fila a revisión → registrar hash del artefacto, hash del lote, versión del motor, conteos por decisión (el audit trail de M22).
- Por día: clientes centinela re-puntuados con igualdad de bits.

**Qué se congela y qué no.** Congelado en el artefacto: todo lo que depende de DEV (cortes, WoE, β, β₀, δ, escala, bandas) y la política vigente (cutoff). Fuera del artefacto, con su propio ciclo: los textos largos de reason codes para cartas al cliente (si se editan a menudo), los umbrales operativos del contrato (si riesgo quiere ajustarlos sin nueva versión del modelo — pero entonces con su propio versionado y aprobación). La regla práctica: **si cambia la decisión de algún cliente, cambia la versión del artefacto**.

**Shadow y rollback.** El challenger corre con el mismo lote, después del contrato del champion, y su salida se escribe en una tabla de sombra con la versión y el hash; nunca en la tabla de decisiones. El rollback es mover el puntero de producción al hash anterior; por eso los artefactos son inmutables (se crea una versión nueva, nunca se edita una existente).

---

## 7. Numpy desde cero vs librerías

| Cálculo | Notebook (numpy desde cero) | Librería | Diferencias de convención | En producción |
|---|---|---|---|---|
| Asignar bin $(a,b]$ | `np.searchsorted(cortes, x, side="left")` | `pd.cut(x, bordes, right=True, labels=False)` | **Defaults que no coinciden**: `pd.cut` cierra a la derecha; `np.digitize` por defecto (`right=False`) cierra a la izquierda; `np.histogram` cierra el último bin por ambos lados; `BETWEEN` en SQL es cerrado en ambos extremos. NaN: `searchsorted` lo manda al final (hay que enmascararlo); `pd.cut` devuelve NaN | `searchsorted` (explícito, vectorizado, sin etiquetas); el notebook verifica igualdad bit a bit con `pd.cut` |
| Cuantiles para los cortes (solo desarrollo) | `np.nanquantile(..., method="linear")` | `pd.Series.quantile` (lineal por defecto) | Hay 9 definiciones de cuantil (Hyndman–Fan); cambiar de método cambia cortes y bins. Irrelevante en producción **si** los cortes vienen del artefacto | Nunca se calcula en producción |
| Categorías | `np.unique(..., return_inverse=True)` + diccionario | `Series.map` | Tipos: `'1'` vs `1` vs `1.0`; NaN no es igual a NaN | Normalizar antes; el notebook verifica igualdad |
| Score y PD | $o-f\eta$ y `expit` | `statsmodels.predict` + `logit`/`expit` | Orden de operaciones ⇒ diferencias de $10^{-13}$; *clip* de la PD en el camino del curso | Desde $\eta$; tolerancia declarada contra el notebook |
| Serialización | `json.dumps(sort_keys, separators, allow_nan=False)` | `orjson`, `simplejson`, `ujson` | Según su documentación, `orjson` escribe NaN/Infinity como `null` (verificar versión): un NaN se convierte en «no acotado» en silencio. Formato de exponentes distinto entre lenguajes | `json` estándar o canónico RFC 8785; validar al leer |
| Hash | `hashlib.sha256` sobre la canónica | — | Hash de un DataFrame con `pd.util.hash_pandas_object`: estable en una versión de pandas, sin garantía documentada entre versiones | Hash de bytes publicados para el artefacto; para datos, hash de un formato de archivo fijado (p. ej., Parquet + versión) |
| Contrato por lote | numpy/pandas (`validar_lote`) | pandera, Great Expectations | Las librerías expresan bien reglas por columna; las reglas de **exceso sobre lo esperado** hay que escribirlas igual | Propio o pandera con checks personalizados |
| Contrato por fila | máscaras numpy | **pydantic** (`create_model` + `field_validator`) | pydantic acepta NaN en `Optional[float]` si no se valida; `Optional[str]` rechaza el NaN de pandas (hay que convertirlo a `None`) | pydantic en la API de originación en línea; el notebook verifica que ambos cuentan exactamente lo mismo por (variable, regla) |
| Artefacto | validador semántico propio | `jsonschema.Draft202012Validator` | JSON Schema valida el **objeto** Python: un `float('nan')` es `number` y pasa | Las tres capas |
| Tests de propiedades | generador numpy con semilla | Hypothesis (Python), QuickCheck (Haskell) | Hypothesis agrega *shrinking* (reduce el contraejemplo a uno mínimo); el notebook hace un *shrinking* rudimentario (el par más cercano) | Hypothesis en CI |

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral (curso)

La corrida de la clase dejó: 8 variables, 41 bins, JSON de 5.463 caracteres, paridad 1,1·10⁻¹³, aprobación TTD 74,6 %, bug A 6,84 % (587/8.585) vs bug B 25,37 % (2.178), contrato con un 3,1 % de `antiguedad_meses` fuera de [6, 344]. Tres lecturas con las herramientas de este módulo:

1. **El 6,8 % del bug A es densidad en el corte.** El curso no reporta $\mathbb{E}|\Delta|$, pero la distancia media de los que cambian (10,0 puntos $\approx\mathbb{E}[\Delta^2]/2\mathbb{E}|\Delta|$) sugiere errores típicos del orden de 20 puntos entre los casos relevantes; con $\Pr\approx f_S(c)\,\mathbb{E}|\Delta|=6{,}8\,\%$, eso implica una densidad del score en 560 de algunas décimas de punto porcentual por punto de score, plausible para un score con desviación de unos 50 puntos (estimación de orden de magnitud; verificable en el notebook de clase con `np.abs(d_a).mean()`). Con la densidad ≈ 0,54 % por punto cerca de 560 que M13 §8.1 lee de los deciles 2–3 de la lámina 35, la fórmula da $\mathbb{E}|\Delta|\approx12{,}6$ puntos; como $\mathbb{E}[\Delta^2]/\mathbb{E}|\Delta|\approx20>12{,}6$, el error del bug A es disperso (muchos casos con Δ chico y una cola de Δ grandes), no un corrimiento uniforme. Un cutoff en una zona menos densa del score habría escondido parte del problema — no lo habría resuelto.
2. **El 3,1 % es 50 veces lo esperable.** Con $n=3.322$, $2/(n+1)=0{,}06\,\%$; en 8.585 solicitudes se esperan ~5 fuera de rango y hay ~269. Incluso con la sobredispersión beta-binomial ($\operatorname{Var}$ ×3,6) es una desviación de ~60 σ: no es ruido. Que sea 🟡 y no 🔴 es una decisión de negocio razonable (rotura de cola, no de unidad), pero debe ir al informe con su hipótesis: una variable que crece con el calendario — la antigüedad de clientes que siguen siéndolo — y que el modelo nunca vio por encima de 344.
3. **La justificación 7d del Lab 3.** Una diferencia de 0,3 puntos es $10^{12}$ veces la cota de redondeo (§3.7): se busca primero el orden δ→PDO (¿se sumó δ al score en vez de al logit?), luego un bin distinto en algún caso (bordes, especiales, claves de tipo), luego WoE o β redondeados en el artefacto. No se despliega porque cada punto de diferencia sistemática cambia decisiones con la fórmula de §3.3.

### 8.2 Banco Sintético (notebook)

| Resultado | Valor |
|---|---|
| DEV / TC / δ / cutoff | 10.065 créditos (11,28 % malos) · 12,42 % · 0,1248 · 530 |
| Artefacto v1.0.0 | 11.320 caracteres canónicos, 8 variables, 38 bins, contrato y procedencia incluidos |
| Paridad motor vs notebook | $1{,}1\cdot10^{-13}$ en score, $2{,}2\cdot10^{-16}$ en PD; bandas y decisiones idénticas; numpy = `pd.cut` y JSON releído **bit a bit** |
| Tabla de puntos | $\sum p-f\delta$ reproduce el score con error $2{,}3\cdot10^{-13}$ |
| Cierre $[a,b)$ en vez de $(a,b]$ | 51,4 % de `consultas_6m` y 22,2 % de `meses_desde_mora_12m` cambian de bin; **11,6 % de decisiones** |
| Cortes a 1 / 2 / 3 decimales | 1,07 % / 0,19 % / 0,02 % de decisiones; 0 desde 4 (datos con 4 decimales) |
| Bug A (TTD completo) | 108 decisiones (2,23 %), 0 % WoE sin mapa; distancia al cutoff 4,6 vs 31,6; predicción 1,86 % y 4,5 puntos |
| Bug B (TTD completo) | 1.109 decisiones (22,9 %), 64,5 % WoE sin mapa |
| Bug A por tamaño de lote (sin drift) | 3,4 % (250) · 1,7 % (1.000) · 1,7 % (4.000) |
| Bug A con drift (lote 1.000) | 10,7 % (γ = −1,5) … 29,4 % (γ = +1,5); el bug B daña más salvo con γ ≤ −1 |
| Propiedades | bug A: permutación ✔, pureza ✔, **subconjunto ✘ (27,2 puntos)** |
| Monotonía | `antiguedad_meses`: 33 de 400 pares violan (46 → 53 meses: −0,06 puntos) |
| Semver | v1.0.1 PATCH ✔ · «v1.0.2» declarada PATCH, requerida MAJOR ✘ · v1.1.0 MINOR (traslado −6,14) ✔ · v1.2.0 (cutoff) MINOR ✔ · v2.0.0 MAJOR ✔ |
| Shadow v1.1.0 | 272 swap-out (5,6 %), 0 swap-in; predicción $f_S(c)\,f\,\Delta\delta\approx5{,}3\,\%$ |
| Desastres sin contrato | ① aprobación 75,4 % → 62,8 % · ② 0,19 % de decisiones cambian en silencio · ③ 3,0 % en silencio (motor ingenuo) o 100 % a revisión (motor con marcas). **Cero excepciones.** Con contrato: los tres 🔴 |

### 8.3 Crédito de motos

El financiamiento de motos tiene tres rasgos que agravan exactamente los modos de falla de este módulo:

- **Lotes chicos por originador.** Buena parte de la originación llega por concesionarios, cada uno con su integración y su volumen diario. Si alguien «normaliza por concesionario» o re-ajusta tramos con el lote del día, el bug A se amplifica: en el generador, pasar de lotes de 1.000 a 250 duplica el daño sin drift (1,7 % → 3,4 %). Y la mezcla de clientes de un concesionario de motos deportivas no es la de uno de motos de trabajo: el γ de un lote puede ser grande. La prueba de subconjunto es obligatoria; la de «clientes centinela» por concesionario, recomendable.
- **Unidades.** Montos en pesos, en miles de pesos o en UF (una UF equivale a varias decenas de miles de pesos; verificar el valor del día) según el sistema del concesionario; pie como monto o como porcentaje; cilindrada en cc o en «clase». Son cambios de unidad por **subpoblación**, no del lote entero: el 100 % fuera de rango que delata el ×1000 del curso se convierte en un 8 % si solo un concesionario lo hace. Por eso el contrato debe tener dominio de negocio **por fila** (la fila imposible va a revisión aunque el lote pase) y reportar severidad **por originador**, no solo por lote.
- **Categorías vivas.** Marcas, modelos y concesionarios nuevos aparecen cada mes. Una variable como «marca» o «tipo de moto» genera `categoria_nueva` estructuralmente; la política para lo no visto (revisión, peor bin, agrupación «otras» con WoE propio estimado en DEV) es una decisión de comité que debe estar **escrita en el artefacto**, no en el código.

Reglas de contrato naturales para motos: `pie ≤ precio`, `monto_financiado = precio − pie (± gastos)`, `plazo ∈ {12, 18, 24, 36, 48}`, `año_modelo ≤ año_solicitud + 1`, `cilindrada > 0`, y un identificador de concesionario obligatorio y único por solicitud.

---

## 9. Preguntas de comité

**1. «¿Cómo sabemos que lo que corre en producción es el modelo que aprobamos?»**
Por tres igualdades verificables: el hash del artefacto cargado es el del expediente (el motor lo verifica al cargar y rechaza uno editado); el motor pasa la paridad contra el notebook de desarrollo en DEV/HO/OOT/TTD con tolerancia declarada ($10^{-9}$ puntos) y decisiones idénticas; y el golden de la versión aprobada reproduce su salida bit a bit en el entorno productivo. Si cualquiera falla, lo que corre no es lo que se aprobó.

**2. «La paridad dio $10^{-13}$, no cero. ¿Por qué aceptarlo?»**
Porque son dos caminos de cómputo distintos (statsmodels y el motor suman en otro orden), y la cota teórica del error de redondeo para esta suma es del orden de $10^{-12}$. Lo que no se acepta es cualquier diferencia de decisión, banda o reason code, ni diferencias entre dos corridas del **mismo** motor con el **mismo** artefacto: ahí se exige igualdad de bits.

**3. «¿Qué pasa con un cliente que el modelo nunca vio (un canal nuevo, un dato faltante que en desarrollo no existía)?»**
No se aprueba automáticamente. El motor lo marca y la decisión es «revisar». Asignarle WoE 0 no es neutral: en el notebook, un canal mal escrito sube el score de un cliente 5 puntos. Si el volumen de no vistos supera el umbral del contrato (10 % para categorías), la corrida entera se bloquea porque ya no es un caso raro sino un cambio del mundo.

**4. «¿Por qué el contrato bloqueó la corrida si el modelo podía puntuar igual?»**
Porque el modelo puntúa cualquier cosa: los tres desastres de laboratorio no producen ninguna excepción, y uno de ellos (la mitad del feed de renta caído) cambia decisiones en silencio incluso con el mejor motor posible, porque el missing de renta existía en desarrollo. La severidad la fija la magnitud: 50 % de missing contra un tope de 13 % no es deriva, es un feed roto.

**5. «¿Cómo decidieron que este cambio es una versión menor y no requiere re-validación completa?»**
No lo decidimos: lo decidió la prueba. Sobre un golden que cubre el 100 % de los bins, la versión nueva desplaza todos los scores en exactamente la misma constante ($-f\,\Delta\delta=-6{,}14$ puntos) y conserva el orden de todos los clientes. Por construcción, Gini, KS y curvas de captura son idénticos; lo que cambia es el nivel de PD y las decisiones en una franja de 6 puntos sobre el cutoff (272 solicitudes, 5,6 % del TTD), que se revisan con el swap-set y el backtesting de calibración.

**6. «¿Qué NO cubre esta batería de pruebas?»**
Tres cosas. (i) La paridad de la fábrica de variables: el artefacto congela el modelo, no el cálculo de `uso_linea_prom_12m`; eso exige su propia prueba de paridad. (ii) La calidad del modelo: todas estas pruebas pasan con un modelo malo (el del notebook incluye dos variables con IV 0,01 a propósito). (iii) La integridad del log de corridas y su custodia: eso es el audit trail con sello externo (M22).

**7. «Si mañana hay que volver atrás, ¿cuánto demora y qué garantiza que volvemos al mismo modelo?»**
Lo que demora mover un puntero: el artefacto anterior sigue en el registro, inmutable, identificado por su hash. La garantía es la verificación del hash al cargar: si alguien editó el archivo anterior (el notebook lo simula cambiando el cutoff), la carga falla. Lo que el rollback no deshace son las decisiones ya tomadas con la versión retirada; esas quedan en el trail con la versión y el hash que las produjo.

---

## 10. Ejercicios

**E1 (cálculo a mano: bordes).** Una variable entera tiene cortes interiores {1, 2} y la distribución en TTD es Poisson(1,2). ¿Qué fracción de casos cambia de bin si un implementador usa $[a,b)$ en vez de $(a,b]$?

<details><summary>Solución</summary>

Cambian exactamente los casos con $x\in\{1,2\}$: $\Pr(X=1)+\Pr(X=2)=e^{-1{,}2}(1{,}2+1{,}2^2/2)=0{,}3012\cdot(1{,}2+0{,}72)=0{,}578$. Un 57,8 % de los casos cambia de bin. (El 51,4 % de `consultas_6m` en el notebook viene de una Poisson con media heterogénea.)
</details>

**E2 (derivación: rango min–máx).** Demuestre que, para $X_1,\dots,X_{n+1}$ intercambiables y continuas, $\Pr(X_{n+1}\notin[\min_{i\le n}X_i,\max_{i\le n}X_i])=2/(n+1)$, y calcule el número esperado de casos fuera de rango en un lote de 8.585 con $n=3.322$.

<details><summary>Solución</summary>

Por intercambiabilidad y continuidad (sin empates), cada una de las $n+1$ variables tiene probabilidad $1/(n+1)$ de ser la mayor; $X_{n+1}$ queda sobre el máximo de las otras $n$ exactamente cuando es la mayor de las $n+1$. Igual para el mínimo; ambos eventos son disjuntos para $n\ge1$. Total $2/(n+1)$. Con $n=3.322$: $6{,}02\cdot10^{-4}$; en 8.585 casos, $\approx5{,}2$ esperados. Austral observó ~269 (3,1 %).
</details>

**E3 (derivación: daño de un error de score).** Un motor suma por error $\delta$ al **score** en vez de al logit (en puntos: +0,1115 en vez de $-0{,}1115\cdot28{,}85=-3{,}22$). Con $f_S(c)=0{,}006$ por punto, ¿qué fracción de decisiones cambia?

<details><summary>Solución</summary>

El score correcto es $S$; el erróneo es $S+3{,}22+0{,}11=S+3{,}33$ (se omitió $-3{,}22$ y se sumó $+0{,}11$). El error es constante, así que cambia la franja $[c-3{,}33,c)$: $\approx0{,}006\times3{,}33=2{,}0\,\%$ de las decisiones, todas de rechazo a aprobación. Un error de 3 puntos en el orden δ↔PDO es exactamente el tipo de diferencia que la paridad debe cazar.
</details>

**E4 (diseño: propiedades).** Un colega propone probar el motor con «el Gini del lote puntuado por producción debe ser igual al del notebook». ¿Detecta el bug A? ¿Y la prueba de permutación? Proponga la prueba mínima que sí lo detecte.

<details><summary>Solución</summary>

El Gini agregado no necesariamente: el bug A mueve scores dentro de una misma franja y puede dejar el Gini casi igual (además, TTD no tiene target). La permutación no lo detecta nunca (Proposición 2). La prueba mínima: puntuar un caso solo y dentro del lote y exigir igualdad (el caso #7 del curso); en CI, cientos de subconjuntos aleatorios incluyendo tamaño 1.
</details>

**E5 (cálculo: semver).** La v1.3.0 cambia la master scale (cortes 490…610 → 495…615) y deja todo lo demás igual. ¿Qué incremento exige la regla de §3.5? ¿Y si además cambia el PDO de 20 a 25?

<details><summary>Solución</summary>

Solo master scale: el score es idéntico bit a bit y cambian las bandas ⇒ MINOR. Con PDO 25: el score cambia como $S'=o'-f'\eta$, una transformación creciente pero no una traslación ($f'\ne f$) ⇒ MAJOR por la convención (cambia el significado de cada punto para todos los consumidores del score; el cutoff 530 deja de significar lo mismo).
</details>

**E6 (código: contrato).** Escriba en numpy la regla «exceso sobre lo esperado» para el rango min–máx: dados `x_lote`, `n_dev`, `mn`, `mx`, devuelva la severidad con umbrales 1 %/10 %.

<details><summary>Solución</summary>

```python
def severidad_rango(x_lote, n_dev, mn, mx, aviso=0.01, bloqueo=0.10):
    obs = x_lote[~np.isnan(x_lote)]
    if obs.size == 0:
        return "🟢", 0.0
    p_hat = np.mean((obs < mn) | (obs > mx))
    exceso = p_hat - 2.0 / (n_dev + 1)
    sev = "🔴" if exceso > bloqueo else "🟡" if exceso > aviso else "🟢"
    return sev, float(p_hat)
```

Nótese que el denominador son los **observados** (los NaN se evalúan en la regla de missing), como corrigió el notebook del curso.
</details>

**E7 (diseño: artefacto).** Liste qué campos del artefacto debería cambiar, y qué incremento de versión corresponde, para separar `meses_desde_mora_12m = −99` en su propio bin (como propone M14).

<details><summary>Solución</summary>

Nuevo bin `{"tipo": "valor", "valor": -99.0, ...}` con su WoE estimado en DEV, WoE del intervalo `(-inf, -9]` re-estimado (ya no contiene los −99), y — como cambian los WoE de una variable — **re-estimación de todos los β** (la logística es conjunta), de β₀, de δ y de los puntos de todas las variables. Cambian el ranking y los scores ⇒ MAJOR, con re-validación completa y golden nuevo. El contrato agrega −99 a `especiales` (ya estaba) y conviene un caso borde con el bin esperado nuevo.
</details>

**E8 (cálculo: beta-binomial).** Con $n=10.065$ y un lote de $m=4.848$, calcule $\mathbb{E}K$ y la desviación estándar de $K$ (casos fuera del min–máx de DEV) bajo la binomial y bajo la beta-binomial. ¿Cuántos casos fuera haría falta observar para una alarma a 3σ en cada caso?

<details><summary>Solución</summary>

$\bar p=2/10.066=1{,}987\cdot10^{-4}$; $\mathbb{E}K=0{,}963$. Binomial: $\sigma=\sqrt{0{,}963(1-\bar p)}=0{,}981$. Beta-binomial: factor $1+(m-1)/(n+2)=1+4.847/10.067=1{,}481$, $\sigma=0{,}981\sqrt{1{,}481}=1{,}194$. Umbral a 3σ: $0{,}963+2{,}94\approx3{,}9$ (binomial) vs $0{,}963+3{,}58\approx4{,}5$ (beta-binomial): 4 vs 5 casos. Con conteos tan chicos la normal es mala aproximación: mejor usar la cola exacta (el notebook reporta el p-valor beta-binomial). Y en cualquier caso, 4–5 casos (≈ 0,1 %) están muy por debajo del umbral de aviso del curso (1 %): el umbral del curso no está pensado para detectar deriva estadísticamente significativa sino para separar deriva material de ruido.
</details>

**E9 (código: motor).** Modifique `_woe_numerico` del notebook para soportar `"cierre": "izquierda"` y verifique con `variante_bordes(art, "cierre_izquierdo")` que `consultas_6m = 1` cae en el bin 1.

<details><summary>Solución</summary>

El notebook ya lo soporta: `_lado = "left" if var["cierre"] == "derecha" else "right"`. Con `side="right"`, `np.searchsorted([1.0, 2.0], 1.0, side="right") = 1`: el valor 1 cae en el bin $[1,2)$, id 1. La prueba: `asignar_bins(casos_borde, variante_bordes(art_v100, "cierre_izquierdo"))` y mirar la fila `BORDE03`.
</details>

---

## 11. Referencias

- **Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards*, 2.ª ed. Wiley.** El capítulo de implementación (pruebas pre-despliegue, reportes de estabilidad) es la referencia de industria para scorecards.
- **Thomas, L. C., Crook, J. N. y Edelman, D. B. (2017). *Credit Scoring and Its Applications*, 2.ª ed. SIAM.** Contexto de despliegue, champion/challenger y monitoreo en crédito de consumo.
- **Federal Reserve / OCC (2011). SR 11-7 / OCC 2011-12, *Supervisory Guidance on Model Risk Management*.** Implementación, pruebas de proceso y control de cambios del código como parte de la gestión de riesgo de modelo. Fue reemplazada el 17-abr-2026 por la guía interagencial revisada SR 26-2 / OCC Bulletin 2026-13, el marco actual, que mantiene gobierno, validación y documentación con un enfoque proporcional al riesgo (detalle en M22). Para Chile, verificar los requisitos de la CMF sobre gestión de riesgo de modelos (ver Serie 1 · E6).
- **Sculley, D. et al. (2015). «Hidden Technical Debt in Machine Learning Systems». *NeurIPS*.** El catálogo clásico de deudas de sistemas de ML: *training-serving skew*, dependencias de datos, configuraciones.
- **Breck, E., Cai, S., Nielsen, E., Salib, M. y Sculley, D. (2017). «The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction». *IEEE Big Data*.** Lista de pruebas de datos, modelo e infraestructura; útil para auditar un pipeline de scoring.
- **Breck, E., Polyzotis, N., Roy, S., Whang, S. E. y Zinkevich, M. (2019). «Data Validation for Machine Learning». *SysML/MLSys*.** Esquemas de datos y detección de *skew* entre entrenamiento y servicio (base de TFDV).
- **Claessen, K. y Hughes, J. (2000). «QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs». *ICFP*.** El origen de los tests de propiedades.
- **MacIver, D. R., Hatfield-Dodds, Z. et al. (2019). «Hypothesis: A new approach to property-based testing». *Journal of Open Source Software*.** La herramienta de tests de propiedades en Python, con *shrinking*.
- **Goldberg, D. (1991). «What Every Computer Scientist Should Know About Floating-Point Arithmetic». *ACM Computing Surveys*.** Por qué la paridad entre caminos de cómputo no es igualdad de bits.
- **Hyndman, R. J. y Fan, Y. (1996). «Sample Quantiles in Statistical Packages». *The American Statistician*.** Las nueve definiciones de cuantil; por qué dos librerías producen cortes distintos.
- **Bray, T. (ed.) (2017). RFC 8259, *The JavaScript Object Notation (JSON) Data Interchange Format*.** Establece que NaN e Infinity no son valores JSON.
- **Rundgren, A., Jordan, B. y Erdtman, S. (2020). RFC 8785, *JSON Canonicalization Scheme (JCS)*.** Serialización canónica para hashes y firmas verificables entre lenguajes.
- **JSON Schema, Draft 2020-12 (json-schema.org).** La especificación usada por `M21_artefacto.schema.json`.
- **Preston-Werner, T. *Semantic Versioning 2.0.0* (semver.org).** La convención MAJOR.MINOR.PATCH que la §3.5 adapta a modelos.
- **Data Mining Group. *PMML 4.4 — Scorecard* (dmg.org).** El estándar de intercambio de scorecards: atributos evaluados en orden, `isMissing`, reason codes `pointsBelow`/`pointsAbove`.
- **ONNX (onnx.ai) y ONNX-ML.** Estándar de grafos de cómputo; la alternativa para modelos no lineales.
- **Bantilan, N. (2020). «pandera: Statistical Data Validation of Pandas Dataframes». *Proceedings of the 19th Python in Science Conference (SciPy)*.** Contratos declarativos para DataFrames.
- **Documentación de pydantic v2 y de `jsonschema` (Python).** Las dos librerías usadas en el notebook para el contrato por fila y el esquema del artefacto.
- **Serie 1 · M5, M6, M7; Serie 2 · M13, M14, M15, M16, M18, M20, M22.** Fábrica de variables, especiales, WoE, tabla de puntos, reason codes, δ y master scale, swap-set, tablero y gobierno.

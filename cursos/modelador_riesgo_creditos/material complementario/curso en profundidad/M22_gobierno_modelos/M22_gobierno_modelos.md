# M22 · Gobierno de modelos: expediente, trazabilidad, model card y validación independiente

> **Ficha.** Profundiza la clase 6 («El modelo que no está documentado no existe»: expediente de 9 piezas, RACI, gatillos, audit trail, cadena de hashes, sello externo, lineage, model card, informe técnico, los 4 errores de la E3), la demo `demo_c6_nikodym_austral` (pasos 5–9), la demo `demo_c6_bases_austral` (secciones 7–12) y el Lab 3 · Financiera Andes (tareas 9–12 y Anexo 2 con las plantillas de informe técnico, model card y resumen ejecutivo).
> **Prerrequisitos:** Serie 1 · E6 (Basilea, IFRS 9, CMF: aquí no se repite), E1 (reject inference), E2 (PIT/TTC); Serie 2 · M19 (backtesting), M20 (monitoreo y semáforos), M21 (artefacto congelado, contrato de datos, paridad).
> **Archivos:** `M22_gobierno_modelos.md` (este documento) · `M22_gobierno_modelos.py` (notebook Marimo, ~15 s en CPU; `marimo edit --sandbox` instala dependencias) · `M22_raci_gatillos.xlsx` (RACI con chequeos COUNTIF, gatillos de cinco partes, tiering y fatiga de alarmas, con fórmulas vivas) · `M22_inventario_modelos.csv` (plantilla de inventario con seis filas de ejemplo).
> **Tiempo estimado:** 5–6 horas (2½ de lectura, 2–3 de notebook y planilla, 1 de ejercicios).
> **Estado regulatorio:** verificado con fuentes primarias o secundarias serias a septiembre de 2026. Lo que no se pudo verificar está marcado *(verificar)*.

---

## 1. Lo que vimos en el curso (y lo que quedó fuera)

**La tesis de la clase 6.** Cinco clases produjeron evidencia; la sexta preguntó si el modelo sobrevive a que su autor se vaya de vacaciones. Un notebook «es la memoria de cómo se construyó un modelo», no un modelo en producción. Lo que sobrevive son cuatro objetos: un **artefacto** (el JSON congelado que puntúa: variables, cortes, WoE, β, δ, escala; tema de M21), un **contrato** (qué datos se aceptan, qué avisa y qué bloquea; M21), un **expediente** y un **gobierno**. Este módulo trata los dos últimos.

**El expediente de nueve piezas** (lámina 7): informe técnico, model card, resumen ejecutivo, notebook reproducible, artefacto versionado, audit trail, tablero de monitoreo, actas de comité e informe de validación. La lámina marcaba con ✓ lo que la corrida deja como objeto trazable y con ✎ lo que exige autoría humana o independencia: el resumen, el notebook ejecutado, el tablero histórico, las actas y el informe de validación. «El informe generado es una base: la recomendación, los gatillos y los usos no previstos todavía se escriben.»

**La RACI** (lámina 8) asignó ocho actividades a cinco roles (Modelador, Jefe de Modelos, Validación, TI, Comité) con dos reglas: una sola A por fila, y **quien construye no valida ni aprueba**. Los **gatillos** (lámina 9) tienen cinco partes —condición medible, valor de hoy, quién decide, qué acción, en cuánto tiempo— y una jerarquía: vigilancia reforzada → recalibración del δ → re-desarrollo → contingencia. «El umbral se escribe ANTES de mirar el dato.» El caso Austral: ranking estable (Gini HO 0,695 vs OOT 0,694), PSI 0,008, calibración OOT roja → diagnosticar por tramos y decidir si basta recalibrar δ; no re-desarrollar sin evidencia de deterioro.

**Trazabilidad** (lámina 6). El audit trail registra evento, hora, paso y decisión, pero «por sí solo puede editarse». El hash de cada evento es su huella, pero «no demuestra que el log esté completo». La cadena $\text{hash}_i=\text{SHA-256}(\text{hash}_{i-1}+\text{evento}_i)$ detecta edición, borrado y reordenamiento, pero «quien reescribe el archivo puede recalcular toda la cola». El sello externo (hash terminal + número de eventos, archivado fuera del log) cierra ese hueco, y «su fuerza depende de que el archivo externo tenga custodia separada». Ataque torpe: la cadena lo detecta; ataque prolijo: la cadena recalculada cuadra y solo el sello lo rechaza. «La librería registra; la institución encadena y custodia el sello.»

**Los números de la corrida de clase** (`nikodym 1.11.0`, Banco Austral): matriz de 15.665 filas y 108 candidatas, `config_hash` `8bdf3d7b…`; una regla de rango deliberadamente falsa (`antiguedad_meses ≤ 344`) bloqueó la corrida con 331 hallazgos; la corrida válida (`run_id` `ab724122…`) dejó **398 eventos, 341 de ellos decisiones** (más 55 artefactos, un inicio y un fin). Gini DEV/HO/OOT 0,7831/0,6949/0,6938; KS 0,6354/0,5671/0,5616; PSI del score DEV→OOT 0,0081; Hosmer-Lemeshow OOT 49,33 con p < 0,001 (PD esperada 6,08% vs mora 5,94%: el nivel medio calza, la **forma** por tramos no) → `overall_status: fail`. Lineage: `data_hash` `7e3b19ce…`, `root_seed` 20240706, `git_sha` None, cinco paquetes registrados. Model card: una limitación declarada («modelado sobre aprobados») y tres automáticas (git no disponible, lineage parcial sin SHA, sin hash de `uv.lock`). Informe: 8 secciones + 3 anexos, 126 tablas. Lectura final: «reproducible y documentado no significa aprobado; el rojo de calibración sigue abierto».

**La versión a mano** (demo de bases): artefacto de 5.463 caracteres (8 variables, 41 bins, δ = +0,1115, cutoff 560, hash `98845d69…`); trail de **8 eventos** con reloj lógico y sello `b92e0003…`; aprobación TTD 74,6% con PD media calibrada 6,31%; el ataque torpe («subir la aprobación del 74,6% al 99%») se detectó en el evento 7; truncamiento y trail vacío también se detectan. El model card manual traía cuatro usos no previstos (IFRS 9, cobranza, precios, otros segmentos) y seis limitaciones con número (33,1% de rechazados no observados; TC anclada con cosechas que incluyen OOT; HL OOT p simulado 0,012; swap-in n = 128 con mora 9,4% vs PD 5,1%; TTD madura en ago-2027; la cadena no detecta reescritura completa). Un detalle que conviene no perder: la corrida a mano y la de `nikodym` son **dos modelos distintos** sobre el mismo dataset (Gini OOT 0,675 vs 0,694, HL p 0,012 vs < 0,001), y el Lab 3 declara invalidante mezclarlos en un mismo informe.

**Documentos.** El resumen ejecutivo tiene cinco bloques en orden: qué se pide aprobar · qué gana la institución · qué se sabe que no funciona · qué se vigila y con qué gatillo · qué se pide de vuelta («si no se sabe qué decidir, es un abstract»). Los **cuatro errores que más bajan nota**: informe que no corresponde a los números del notebook (invalidante), limitaciones genéricas, model card sin usos no previstos, notebook que no corre de punta a punta. Antídoto: generar los documentos **desde** la corrida.

**Lo que el curso simplificó, omitió o dejó como convención:**

1. **El marco regulatorio de riesgo de modelo.** La clase habló de «el regulador» en genérico. No se revisó SR 11-7 —que además fue **reemplazada en abril de 2026** por SR 26-2—, ni EBA/ECB, ni lo que la CMF exige hoy a modelos de provisiones, ni el hecho de que una fintech no bancaria chilena no tiene un marco de riesgo de modelo específico.
2. **Inventario y *tiering*.** No se habló de la lista institucional de modelos ni de cómo la materialidad determina cuánto gobierno recibe cada uno.
3. **El ciclo de vida completo.** El curso terminó en «aprobar»; no trató cambios post-aprobación (qué es un cambio menor), revisión periódica ni retiro.
4. **Validación independiente.** Apareció como pieza ✎ del expediente y como fila de la RACI, sin decir qué hace un validador ni cómo prepararse.
5. **Overrides.** La RACI asignó «autorizar excepciones al cutoff» al Comité; no se vio cómo se registran, se limitan y se validan.
6. **Canonicalización.** El curso usó `sort_keys=True` y lo justificó; no discutió números (`1` vs `1.0`), NaN, Unicode ni interoperabilidad entre lenguajes (RFC 8785).
7. **Qué garantiza exactamente cada capa.** Se mostraron dos ataques. Faltan: borrado, inserción, reordenamiento, truncamiento de la cola (que el HMAC tampoco detecta), el sello «externo» guardado donde el atacante escribe, y los modelos de amenaza correspondientes.
8. **Sello externo real.** Se dijo «append-only, custodio, firma». No se explicó RFC 3161 ni qué prueba un token de sellado de tiempo.
9. **Árboles de Merkle** como alternativa para verificar un evento sin recorrer ni revelar el log.
10. **Hash de un DataFrame.** El `hash_df` del curso usa `pandas.util.hash_pandas_object`: rápido, pero identifica la *representación física*, no el *contenido*, y no está documentado como estable entre versiones.
11. **La herramienta como modelo.** `nikodym` se fijó con `==1.11.0` y un `assert` sobre la versión: correcto, pero la validación de la librería misma quedó implícita.
12. **Inconsistencias menores del material:** la demo de bases dice «7 secciones + 3 anexos» y la de `nikodym` «8 secciones» (la diferencia es contar o no el resumen ejecutivo); el Lab 3 rotula como «Clase 7» la implementación que el curso dictó en la clase 6.

---

## 2. Intuición

**El riesgo de modelo es un riesgo de decisión, no de código.** SR 11-7 lo definió como «las potenciales consecuencias adversas de decisiones basadas en salidas o reportes de modelos incorrectos o mal usados», con dos fuentes: el modelo tiene errores fundamentales (datos, supuestos, implementación) o el modelo está bien y **se usa para lo que no fue hecho**. La segunda fuente es la que explica por qué el curso insiste en los *usos no previstos*: un scorecard de admisión a 12 meses que alguien reutiliza para provisionar cartera vigente o para fijar precios no falla en ninguna métrica del tablero, porque el tablero mide lo que el modelo sabe hacer.

**Gobernar es separar funciones para que el error de uno lo vea otro.** Quien construye tiene incentivos (plazos, bonos, orgullo técnico) y puntos ciegos (lo que no pensó no lo va a probar). La validación independiente existe porque «el autor no ve su propio error» es un hecho empírico, no una sospecha moral. La RACI es la forma de escribir esa separación para que sea **verificable**: una A por fila y el modelador fuera de las filas de validar y aprobar. Todo lo demás del gobierno —gatillos, actas, inventario— es memoria institucional: decisiones tomadas antes, por escrito, con dueño, para que el día en que el indicador se ponga rojo la respuesta sea una acción y no una reunión.

**La evidencia tiene capas, y cada capa responde a un atacante distinto.** Un log plano responde «qué pasó» si nadie lo tocó. El hash de cada evento detecta que *ese* evento cambió. La cadena detecta ediciones descuidadas. Una clave (HMAC) o un ancla externa (sello) detectan al que reescribe todo. Un sellado de tiempo de un tercero prueba que esos bytes existían antes de una fecha. Ninguna capa dice si la decisión registrada era **correcta**: integridad no es calidad. El error típico es comprar una capa («audit trail inmutable») sin preguntar contra qué atacante protege. La pregunta útil es la del curso: *¿dónde está el ancla?*

**Reproducible no es aprobado, y documentado no es verdadero.** El lineage (`config_hash`, `data_hash`, semilla, versiones) permite que un tercero **re-obtenga** los números; no dice que los números sean buenos. El model card declara límites; no garantiza que se respeten. La secuencia correcta es: reproducibilidad → validación (¿los números son buenos para este uso?) → aprobación (¿la institución acepta el riesgo residual?) → monitoreo (¿sigue siendo cierto?). Saltarse un escalón porque el anterior «salió verde» es exactamente el error que el curso cerró con el caso Austral: todo reproducible, todo documentado, calibración OOT roja.

**Los gatillos son hipótesis preregistradas.** Escribir el umbral después de ver el dato es el «jardín de senderos que se bifurcan» del análisis de datos aplicado al gobierno: cualquier umbral elegido con el dato a la vista se puede racionalizar. Pero preregistrar no basta: con muchos indicadores, alguno se prende por azar todos los meses. La jerarquía (vigilar → recalibrar → re-desarrollar) es la respuesta a esa fatiga: el primer escalón es barato y confirma persistencia antes de gastar.

---

## 3. Formalización

### 3.1 Qué es un modelo y cuánto importa: definición y materialidad

SR 11-7 (2011) definía modelo como un «método, sistema o enfoque cuantitativo que aplica teorías y técnicas estadísticas, económicas, financieras o matemáticas y supuestos para procesar datos de entrada en estimaciones cuantitativas». SR 26-2 (2026) lo reescribió como «un método, sistema o enfoque cuantitativo **complejo** que aplica teorías estadísticas, económicas o financieras para procesar datos de entrada en estimaciones cuantitativas» y, según los análisis publicados, excluye expresamente cálculos aritméticos simples y procesos deterministas basados en reglas. La consecuencia práctica para un área de riesgo: una política de *knock-outs* o una planilla de pricing puede quedar fuera de la definición, pero **no fuera del inventario** (sección 4.3).

Para cuantificar materialidad conviene expresar el daño de un error del modelo en la decisión. En un scorecard con escala $\text{score}=\text{offset}-\text{factor}\cdot(\eta+\delta)$ y regla «aprobar si score $\ge c$», un error $\Delta\delta$ en el ajuste de nivel desplaza **todos** los scores en $-\text{factor}\cdot\Delta\delta$ puntos. Si $f_S$ es la densidad del score en el lote alrededor del cutoff, la tasa de aprobación cambia en

$$
\Delta A \;=\; \Pr(S-\text{factor}\,\Delta\delta\ge c)-\Pr(S\ge c)\;\approx\; -f_S(c)\cdot\text{factor}\cdot\Delta\delta .
$$

Con PDO 20, $\text{factor}=20/\ln 2=28{,}8539$; $\Delta\delta=0{,}10$ son 2,89 puntos. En el notebook (sección 10), la aprobación TTD pasa de 66,9% a 65,1% con $\Delta\delta=+0{,}10$ y a 70,0% con $-0{,}10$: una densidad local de ≈0,6 pp por punto hacia arriba y ≈1,1 pp por punto hacia abajo (asimétrica: el cutoff no está en el centro de la distribución y la aproximación lineal solo vale para desplazamientos chicos). Multiplicado por el volumen mensual y por la pérdida esperada marginal de los aprobados en esa franja, eso es la **materialidad de un error de calibración** en pesos. Con esa cuenta se defiende por qué un rojo de calibración con Gini sano es tema de comité.

### 3.2 Canonicalización

Una función de hash criptográfica $H:\{0,1\}^*\to\{0,1\}^{256}$ opera sobre **bytes**. Para hashear objetos se necesita una serialización $c:\mathcal{O}\to\{0,1\}^*$. Sea $\equiv$ la equivalencia que declaramos sobre objetos (por ejemplo: dos diccionarios son equivalentes si tienen las mismas claves con valores equivalentes, sin importar el orden de inserción). Pedimos

$$
a\equiv b \iff c(a)=c(b). \tag{C}
$$

La dirección $\Rightarrow$ es la que falla sin canonicalización: `json.dumps({"b":1,"a":2})` y `json.dumps({"a":2,"b":1})` son bytes distintos para objetos equivalentes, y sus hashes difieren. `sort_keys=True` la arregla para claves. La dirección $\Leftarrow$ se cumple si $c$ es inyectiva sobre clases: JSON lo es mientras no se pierda información (por eso NaN, que no es JSON válido, debe **rechazarse** con `allow_nan=False` en vez de serializarse como el literal `NaN`).

Tres fuentes de violación de (C) que `sort_keys` no resuelve:

- **Números.** ¿Es `1` equivalente a `1.0`? Python serializa `1` y `1.0`; el esquema RFC 8785 (*JSON Canonicalization Scheme*, JCS, 2020) serializa números como ECMAScript, y ambos quedan `1`. Dos sistemas que dicen «SHA-256 del JSON canónico» producen hashes distintos para el mismo evento si uno usa `json.dumps` y el otro JCS. Para doubles, Python usa `repr(float)`, la representación decimal **más corta que reconstruye exactamente** el mismo double (garantía de CPython desde 3.1); no hay pérdida, pero `0.1+0.2` se serializa `0.30000000000000004` —correcto: es otro número—.
- **Unicode.** «Concepción» puede venir compuesta (NFC: `ó` = U+00F3) o descompuesta (NFD: `o` + U+0301). Son la misma cadena para un humano y bytes distintos. JCS **no** normaliza; la normalización NFC es una política que se declara y se aplica **antes** del hash.
- **Tipos físicos.** `int64` vs `float64`, `-0.0` vs `0.0` (iguales en IEEE-754 bajo `==`, con bits distintos). La política debe decir si son equivalentes.

En el notebook, un serializador escrito desde cero (reglas explícitas de escape y `repr` para doubles) coincide byte a byte con `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False)` en el artefacto, los 28 eventos y una batería de casos borde.

### 3.3 La cadena de hashes y lo que puede prometer

Sean $e_1,\dots,e_n$ los eventos, $h_0=0^{256}$ y

$$
h_i \;=\; H\big(h_{i-1}\,\|\,c(e_i)\big),\qquad i=1,\dots,n.
$$

**Proposición 1 (propagación).** Si se reemplaza $e_k$ por $e_k'\not\equiv e_k$ y se recalcula, entonces $h_j'\neq h_j$ para todo $j\ge k$, salvo una colisión de $H$.

*Demostración.* Por (C), $c(e_k')\neq c(e_k)$, luego las entradas $h_{k-1}\|c(e_k')$ y $h_{k-1}\|c(e_k)$ son distintas. Si $h_k'=h_k$, tenemos una colisión de $H$. Supongamos $h_k'\neq h_k$. Para $j>k$, las entradas $h_{j-1}'\|c(e_j)$ y $h_{j-1}\|c(e_j)$ difieren en el prefijo de 64 caracteres (longitud fija), luego son distintas; si $h_j'=h_j$, de nuevo hay colisión. Por inducción, $h_j'\neq h_j$ para todo $j\ge k$ salvo colisión. $\square$

La probabilidad de colisión es despreciable: por el argumento del cumpleaños, con $q$ hashes calculados, $\Pr(\text{colisión})\le q^2/2^{257}$; con $q=10^{12}$ eventos, $\approx 4\cdot10^{-54}$. El riesgo real no es criptográfico; es de **modelo de amenaza**.

**Modelo de amenaza.** Clasificamos al adversario por capacidades:

- $\mathcal{A}_1$ (torpe): edita $e_k$ y deja los $h_i$ guardados;
- $\mathcal{A}_2$ (prolijo): edita, borra, inserta o reordena eventos y **recalcula** la cola; renumera si hace falta;
- $\mathcal{A}_3$: como $\mathcal{A}_2$ y además reescribe el archivo del sello si está a su alcance;
- $\mathcal{A}_T$ (truncador): elimina los últimos $m$ eventos y sus hashes, sin recalcular nada.

**Proposición 2 (qué detecta cada defensa).** (i) La cadena sola detecta $\mathcal{A}_1$ y no detecta $\mathcal{A}_2$ ni $\mathcal{A}_T$. (ii) La estructura (numeración $1..n$ sin huecos, `run_id` único, último evento de tipo `run_end`) detecta $\mathcal{A}_T$ salvo que el truncador agregue un cierre falso y recalcule —lo que lo convierte en $\mathcal{A}_2$—. (iii) Un sello $(h_n,n)$ fuera del alcance del adversario detecta $\mathcal{A}_1$, $\mathcal{A}_2$ y $\mathcal{A}_T$ —siempre que se compare contra la cadena **recalculada desde los eventos**—, y no detecta $\mathcal{A}_3$.

*Demostración.* (i) Contra $\mathcal{A}_1$, la Proposición 1 aplicada a la verificación (que recalcula) da $h_k^{\text{recalc}}\neq h_k^{\text{guardado}}$. Contra $\mathcal{A}_2$, el adversario presenta una lista $(e',h')$ con $h'=\text{encadenar}(e')$ por construcción: la verificación recalcula exactamente $h'$. Contra $\mathcal{A}_T$, el prefijo $(e_{1..n-m},h_{1..n-m})$ satisface la recursión porque cada $h_i$ solo depende del pasado. (ii) Es directo de la definición. (iii) Tras $\mathcal{A}_2$ o $\mathcal{A}_1$, la cadena recalculada termina en $h_n'\neq h_n$ por la Proposición 1 (o tiene otro largo si hubo borrado o inserción); tras $\mathcal{A}_T$, el largo es $n-m\neq n$. Si en cambio el sello se compara con los $h_i$ **guardados**, $\mathcal{A}_1$ pasa, porque dejó intacto $h_n$. $\square$

La nota final de (iii) no es un tecnicismo: al construir el notebook, la primera versión comparaba el sello con la lista de hashes guardada y el ataque torpe **pasaba** el sello. El control correcto recalcula siempre desde los eventos.

### 3.4 HMAC: mover el problema a una clave

Con una clave secreta $K$, $h_i=\text{HMAC}_K(h_{i-1}\|c(e_i))$ (RFC 2104). Si $\mathcal{A}_2$ no conoce $K$, recalcular la cola exige forjar un MAC, lo que es computacionalmente inviable si HMAC-SHA256 es un PRF seguro. Pero:

- $\mathcal{A}_T$ **sigue sin detectarse**: el prefijo trae MACs auténticos (el notebook lo verifica en la matriz de ataques);
- quien tiene $K$ puede reescribir todo: la seguridad se reduce a la custodia de la clave (KMS/HSM, rotación, separación entre quien opera el pipeline y quien administra la clave);
- verificar exige la clave: el auditor externo necesita acceso a $K$ o un servicio de verificación. Una firma asimétrica evita esto: se verifica con la clave pública.

Esquemas de logs seguros más elaborados (Schneier y Kelsey, 1999) evolucionan la clave hacia adelante ($K_{i+1}=H(K_i)$, borrando $K_i$) para que comprometer el sistema en el tiempo $t$ no permita reescribir el pasado.

### 3.5 El sello externo y RFC 3161

El sello del curso es $s=(h_n,n)$. Su garantía depende de la custodia: si $\mathcal{A}_3$ escribe donde se guarda $s$, fabrica $s'=(h_n',n')$ coherente con su log adulterado. Una **autoridad de sellado de tiempo** (TSA, RFC 3161, 2001; actualizado por RFC 5816) resuelve la custodia con un tercero: el solicitante envía solo $\text{messageImprint}=H(s)$; la TSA devuelve un token firmado que contiene, entre otros campos, `policy`, `messageImprint`, `serialNumber` y `genTime`. La TSA no examina el contenido: por diseño solo ve el hash. Formalmente, el token es $\tau=\text{Sign}_{sk_{\text{TSA}}}(\text{TSTInfo})$ con $\text{TSTInfo}\ni(H(s),t)$, y cualquiera con el certificado de la TSA verifica $\text{Verify}_{pk}(\tau)$.

**Qué prueba:** que los bytes cuyo hash es $H(s)$ existían **antes** de $t$. Si $\mathcal{A}_3$ fabrica $s'$ después, no puede obtener un token con fecha anterior (salvo corromper a la TSA). **Qué no prueba:** que el log estuviera completo al sellarse, que su contenido sea verdadero, ni que la decisión fuera correcta. El notebook simula la TSA con un HMAC para mostrar las capas; la simulación **no** tiene la propiedad de verificación pública ni de no repudio de una firma real.

Almacenamiento WORM (*write once, read many*) es otra forma de custodia: el medio impide sobrescribir durante un período de retención. La regla 17a-4 de la SEC para *broker-dealers* es el ejemplo regulatorio clásico; sus enmiendas de octubre de 2022 admiten, como alternativa al WORM, un sistema que conserve un *audit trail* completo de modificaciones. Para un banco o una fintech chilena no es norma aplicable; es una referencia de diseño.

### 3.6 Árboles de Merkle

Con hojas $d_0,\dots,d_{n-1}$ (los eventos canónicos), RFC 6962 define

$$
\text{MTH}(\{\})=H(),\qquad
\text{MTH}(\{d_0\})=H(\texttt{0x00}\,\|\,d_0),\qquad
\text{MTH}(D_n)=H\big(\texttt{0x01}\,\|\,\text{MTH}(D_{0:k})\,\|\,\text{MTH}(D_{k:n})\big),
$$

con $k$ la mayor potencia de 2 estrictamente menor que $n$.

**Prueba de inclusión.** $\text{PATH}(m,D_n)$ es la lista de hashes hermanos desde la hoja $m$ hasta la raíz:
$\text{PATH}(m,D_n)=\text{PATH}(m,D_{0:k})\,:\,\text{MTH}(D_{k:n})$ si $m<k$, y $\text{PATH}(m-k,D_{k:n})\,:\,\text{MTH}(D_{0:k})$ si $m\ge k$.

**Proposición 3.** $|\text{PATH}(m,D_n)|\le\lceil\log_2 n\rceil$.

*Demostración.* Por inducción fuerte en $n$. Para $n=1$ el camino es vacío. Para $n>1$, el camino agrega un hash y recurre sobre un subárbol de tamaño $k$ o $n-k$, ambos $\le k<n$, con $k$ la mayor potencia de 2 menor que $n$, de modo que $k\ge n/2$ y $n-k\le k$. Luego el subárbol tiene a lo más $k=2^{\lceil\log_2 n\rceil-1}$ hojas y, por hipótesis, un camino de largo $\le\lceil\log_2 n\rceil-1$. Sumando el hash agregado, $\le\lceil\log_2 n\rceil$. $\square$

Para los 398 eventos de la corrida `nikodym`, 9 hashes; para los 28 del notebook, 5. La verificación (RFC 9162 §2.1.3.2) recorre el camino combinando a izquierda o derecha según los bits del índice $m$ y del tamaño $n-1$, y compara con la raíz sellada.

**Por qué los prefijos.** Sin `0x00`/`0x01`, un nodo interno $H(a\|b)$ es indistinguible del hash de una hoja cuyo contenido fuera $a\|b$: un atacante podría presentar un árbol más corto con una «hoja» que en realidad es un nodo (ataque de segunda preimagen sobre la estructura). Los prefijos separan dominios.

**La mutación de duplicar.** Si el nivel impar se completa **duplicando** el último nodo (como en el árbol de transacciones de Bitcoin), los logs $[a,b,c]$ y $[a,b,c,c]$ producen la misma raíz: en el primer nivel el primero se completa a $[\,\ell_a,\ell_b,\ell_c,\ell_c\,]$, que es exactamente el segundo. Un auditor que solo compara raíces aceptaría un log con un evento repetido. Este defecto se documentó en Bitcoin como CVE-2012-2459. RFC 6962 **promueve** el nodo impar sin duplicarlo, y la implementación iterativa que promueve coincide con la recursiva para todo $n$ (el notebook lo verifica para $n=0,\dots,129$).

**Cadena vs Merkle.** La cadena responde «¿se tocó la historia?» y exige recorrerla entera. Merkle responde «¿este evento está en el log sellado?» con $O(\log n)$ hashes y **sin revelar los demás eventos**: útil cuando el auditor no puede ver datos de clientes. La propiedad *append-only* entre dos raíces sucesivas se prueba con **pruebas de consistencia** (RFC 6962 §2.1.2). Los *transparency logs* (Certificate Transparency) combinan ambas ideas con publicación pública de raíces firmadas.

### 3.7 Lineage como identificación de una función

Una corrida es $y=f(\text{código},\text{config},\text{datos},\text{entorno},\text{semilla})$. El lineage registra un identificador por argumento: `git_sha` (y `git_dirty`), `config_hash`, `data_hash`, versiones de librerías y hash del archivo de bloqueo, `root_seed`. Reproducibilidad es la implicación

$$
(\text{git\_sha},\text{config\_hash},\text{data\_hash},\text{lock\_hash},\text{root\_seed})\ \text{iguales}\ \Rightarrow\ y\ \text{igual},
$$

que exige que $f$ sea **determinista** dado ese vector (sin reloj de pared, sin orden de iteración no determinista, sin paralelismo con reducciones en orden variable, sin llamadas a red). El `run_id` identifica la *ejecución* y por eso cambia; comparar corridas exige una **huella de contenido** que lo excluya. En el notebook, dos corridas con distinto `run_id` producen la misma huella del trail, el mismo artefacto y el mismo δ bit a bit. Cuando un eslabón falta (`git_sha=None`, `lock_hash=None`), la implicación deja de ser verificable y el sistema debe decirlo: las «limitaciones automáticas» de `nikodym` son exactamente eso.

### 3.8 Dos hashes de un DataFrame y sus clases de equivalencia

Toda función de hash de datos induce una equivalencia: $D\sim_h D' \iff h(D)=h(D')$. Elegir $h$ es elegir qué cuenta como «los mismos datos». Se comparan dos:

- $h_{\text{pd}}$ (la del curso): SHA-256 de un esquema (posición, nombre, dtype) concatenado con `pandas.util.hash_pandas_object(df, index=True)`. Induce la equivalencia **física**: mismos dtypes, mismo orden de filas y columnas, mismo índice, mismos bits.
- $h_{\text{can}}$ (canónica): filas ordenadas por `id`, columnas por nombre, números como double con `repr` y $-0{,}0\to0{,}0$, NaN/None → `null`, texto en NFC. Induce la equivalencia de **contenido** bajo una política declarada.

Se definen dos errores respecto de la política de contenido: **falsa alarma** ($D\equiv D'$ pero $h(D)\neq h(D')$) y **omisión** ($D\not\equiv D'$ pero $h(D)=h(D')$). Las omisiones son colisiones (despreciables) o pérdidas de información de la serialización (por ejemplo, convertir a double un entero mayor que $2^{53}$). En el notebook, frente a nueve perturbaciones, $h_{\text{can}}$ acierta en las nueve y $h_{\text{pd}}$ da **siete falsas alarmas** (permutar filas, reordenar columnas, `float64→int64` con los mismos valores, `str→object`, $-0{,}0$, cambiar el índice, NFD). Ambos detectan el cambio real ($+10^{-9}$ en un valor).

¿Cuál es mejor? Depende de la pregunta. «¿Es exactamente el archivo que se usó?» → $h_{\text{pd}}$ (o el SHA-256 del archivo Parquet). «¿Son los mismos datos, aunque se hayan re-exportado?» → $h_{\text{can}}$. Para un `data_hash` que debe sobrevivir años en un expediente, importa además la **estabilidad entre versiones**: $h_{\text{can}}$ depende de `repr` de Python, JSON y NFC, todo especificado; `hash_pandas_object` depende de una implementación interna (SipHash con una clave fija por defecto y una combinación propia de columnas) cuya estabilidad entre versiones la documentación no promete, y además es sensible a cambios de dtype por defecto, como el tipo `str` de pandas 3 *(la ausencia de garantía es una lectura de la documentación; verificar para su versión)*.

### 3.9 Overrides: tasa, desempeño y potencia

Sea $R$ el conjunto de rechazados por el score y $O\subseteq R$ los aprobados por excepción (*low-side overrides*). Tres cantidades:

$$
\text{tasa de override}=\frac{|O|}{|A|+|O|},\qquad
\hat p_O=\frac{1}{|O|}\sum_{i\in O}y_i,\qquad
\bar\pi_O=\frac{1}{|O|}\sum_{i\in O}\text{PD}_i ,
$$

con $A$ los aprobados por score, $y_i$ el malo observado y $\text{PD}_i$ la PD calibrada del modelo. La hipótesis «la excepción no agrega información» es $H_0:\ \mathbb{E}[\hat p_O]=\bar\pi_O$; con $|O|$ pequeño y PD parecidas dentro de $O$, $X=\sum_{i\in O}y_i\sim\text{Bin}(|O|,\bar\pi_O)$ bajo $H_0$ (aproximación: la suma exacta es Poisson-binomial, M19). El p-valor exacto de dos colas es

$$
p=\sum_{k:\ \Pr(X=k)\le\Pr(X=x)}\Pr(X=k),
$$

que es el criterio de `scipy.stats.binomtest` (con tolerancia relativa $10^{-7}$ en la comparación); el notebook lo implementa con `math.lgamma` y coincide.

**Potencia.** Para detectar que los overrides tienen mora $p_1$ cuando el modelo les asigna $p_0$, con test unilateral de nivel $\alpha$ y potencia $1-\beta$, la aproximación normal da

$$
n \;\approx\;\left(\frac{z_{1-\alpha}\sqrt{p_0(1-p_0)}+z_{1-\beta}\sqrt{p_1(1-p_1)}}{p_1-p_0}\right)^2 .
$$

Con $p_0=10\%$, $p_1=15\%$, $\alpha=5\%$ y potencia 80%: $n\approx\big((1{,}645\cdot0{,}300+0{,}842\cdot0{,}357)/0{,}05\big)^2=(0{,}794/0{,}05)^2\approx 252$ overrides **maduros**. Si la maduración es de 12 meses, se aprende del override un año después de otorgarlo. Por eso el registro tiene que existir desde el primer día.

**Lo que no se puede validar.** Los *high-side overrides* (rechazar a quien el score aprueba) no tienen desempeño observable: es el mismo problema contrafactual del swap-set (M18) y del reject inference (Serie 1 · E1). Se gobiernan por tasa, motivo y concentración, no por desempeño.

### 3.10 RACI y gatillos como restricciones

Una RACI es una matriz $M\in\{\text{R},\text{A},\text{R/A},\text{C},\text{I},\varnothing\}^{a\times r}$ (actividades × roles). Con $\mathbb{1}_A(x)=[x\in\{\text{A},\text{R/A}\}]$ y $\mathbb{1}_R(x)=[x\in\{\text{R},\text{R/A}\}]$, las reglas del curso son

$$
\sum_j \mathbb{1}_A(M_{ij})=1,\qquad \sum_j\mathbb{1}_R(M_{ij})\ge1,\qquad
\iota_i=1\Rightarrow \mathbb{1}_A(M_{i,\text{mod}})=\mathbb{1}_R(M_{i,\text{mod}})=0,
$$

donde $\iota_i$ marca las actividades que exigen independencia. Son **tests**: se chequean con `COUNTIF` en la planilla y con dos implementaciones en el notebook.

Un gatillo es una tupla $g=(\text{indicador},\ \diamond,\ \theta,\ \text{dueño},\ \text{acción},\ \text{plazo})$ con $\diamond\in\{>,\ge,<,\le\}$, y su estado es

$$
\text{disparado}(g)=\begin{cases}\text{no evaluable} & \text{si el valor de hoy no existe}\\ \text{SÍ} & \text{si } x\diamond\theta\\ \text{no} & \text{si no.}\end{cases}
$$

**Fatiga de alarmas.** Con $k$ indicadores independientes, cada uno con probabilidad $\alpha$ de alarma bajo estabilidad, la probabilidad de al menos una alarma falsa en un mes es $1-(1-\alpha)^k$: con $k=9$ y $\alpha=5\%$, 37,0%; en un año, $1-(1-\alpha)^{12k}\approx 99{,}6\%$, con $12k\alpha=5{,}4$ alarmas falsas esperadas. (Si los indicadores están correlacionados —PSI del score y CSI de sus variables lo están—, la probabilidad real es menor.) Exigir **persistencia** reduce el problema: con dos trimestres seguidos y trimestres independientes, $\Pr=\alpha^2=0{,}25\%$ por indicador. Esa es la razón matemática de la jerarquía del curso: el primer escalón (vigilancia) es barato y está diseñado para prenderse seguido; la recalibración exige persistencia; el re-desarrollo exige deterioro de ranking.

---

## 4. Variantes y alternativas de industria

### 4.1 Marcos de riesgo de modelo (estado a septiembre de 2026)

| Marco | Qué resuelve | Costo / exigencia | Cuándo usarlo como referencia | Quién lo aplica |
|---|---|---|---|---|
| **SR 11-7** (Fed, 4-abr-2011; OCC 2011-12; FDIC la adoptó en 2017) | Definió riesgo de modelo (errores fundamentales y **mal uso**), las tres piezas de la validación (solidez conceptual, monitoreo continuo con verificación de procesos y *benchmarking*, análisis de resultados con *backtesting*), el *effective challenge* («análisis crítico por partes objetivas e informadas») apoyado en incentivos, competencia e influencia, revisión periódica «al menos anual», inventario y análisis de overrides | Alto: programa completo de MRM | Sigue siendo el vocabulario de la industria y de casi toda la literatura. **Rescindida el 17-abr-2026** | Bancos de EE.UU. (hasta 2026); de facto, bancos globales |
| **SR 26-2** / **OCC Bulletin 2026-13** (17-abr-2026, interagencial Fed-OCC-FDIC) | Reemplaza SR 11-7 y SR 21-8. Enfoque basado en riesgo: **riesgo inherente** (complejidad, supuestos, datos) × **materialidad** (exposición y propósito). Definición de modelo restringida a métodos «complejos». Excluye IA generativa y agéntica. Vendor models con sección propia. *Effective challenge* por personas con pericia, independencia suficiente para objetividad y posición para provocar cambios. La frecuencia de validación depende del modelo (sin «anual» fijo) | Proporcional; **no vinculante**: «no establece estándares exigibles… el incumplimiento no dará lugar a crítica supervisora» | Para diseñar un *tiering* defendible y un programa proporcional en una institución pequeña | Instituciones con más de USD 30.000 millones en activos reguladas por la Fed (umbral principal); las menores pueden usarla como referencia |
| **PRA SS1/23** (Reino Unido, vigente desde 17-may-2024) | Cinco principios: identificación y clasificación de modelos, gobierno, desarrollo-implementación-uso, validación independiente, mitigantes de riesgo de modelo | Alto; incluye responsabilidad de un alto ejecutivo designado | Referencia moderna y explícita de «model risk mitigants» (ajustes post-modelo, restricciones de uso) | Bancos del Reino Unido con modelos internos aprobados *(verificar alcance exacto)* |
| **EBA/GL/2017/16** (publicada nov-2017, aplicable desde 1-ene-2021) | Estimación de PD y LGD en IRB: representatividad de datos, filosofía de rating (PIT/TTC), tasa de default de largo plazo (con más de 5 años si hace falta para capturar el ciclo), **margen de conservadurismo** por categorías A (deficiencias de datos/método), B (cambios de política o de originación) y C (error de estimación general), revisión anual de estimaciones con umbrales predefinidos, juicio humano y overrides documentados | Alto; exige MoC cuantificado | Cuando el modelo alimenta capital IRB, o como estándar de calidad para calibración y overrides | Bancos UE con IRB |
| **ECB Guide to internal models** (origen: TRIM; revisiones feb-2024, jul-2025 con CRR3 y *machine learning*, y jun-2026 que retiró la guía sobre CCF) | Capítulo de temas generales: documentación, gobierno de datos, marco de MRM, validación interna inicial y **anual**, separación entre validación y desarrollo, auditoría interna, registro de modelos (dueño, ámbito, materialidad, fecha de aprobación), cambios de modelo, terceros; ML «adecuadamente explicable» y con desempeño que justifique su complejidad | Alto | Estándar operativo más detallado para un área de validación | Bancos significativos de la eurozona con modelos internos |
| **CMF Chile — Compendio de Normas Contables, cap. B-1** | Provisiones por riesgo de crédito. Método estándar de consumo vigente desde enero de 2025 (PD a 12 meses por mora propia y del sistema; PDI por producto); las provisiones son el **máximo** entre el método estándar y el interno, y la presencia del estándar no elimina la obligación de desarrollar metodologías internas | Medio–alto | Todo modelo que toque provisiones en un banco | Bancos |
| **CMF — consulta RAN 21-9** (3-ago-2026, abierta hasta 26-oct-2026) | Consolida requisitos para **metodologías internas** de provisiones y capital: primera solicitud con ≥ 20% de las exposiciones y plan de cobertura total en 5 años; modalidades fundacional y avanzada; modifica B-1 y RAN 21-1/21-6. Carteras grupales: vigencia inmediata tras publicación; individuales desde 2028 | En consulta | Seguirla si se trabaja con bancos: será el marco chileno de aprobación de modelos internos | Bancos (propuesta) |
| **Ley Fintec 21.521** (publicada 4-ene-2023) + **NCG 502** (CMF, 12-ene-2024) | Regula servicios financieros tecnológicos; «asesoría crediticia» = «evaluaciones o recomendaciones **a terceros** respecto de la capacidad o probabilidad de pago». NCG 502 exige directorio responsable, políticas, función de riesgos independiente, auditoría interna, ciberseguridad y continuidad, con proporcionalidad | Proporcional | Si la fintech ofrece scoring a terceros o intermedia financiamiento | Prestadores inscritos. Según entiendo, **prestar con recursos propios** (una financiera de motos) no es por sí mismo un servicio Fintec *(verificar con asesoría legal)* |
| **Ley 21.719** (protección de datos; vigencia 1-dic-2026, con una postergación en evaluación a agosto de 2026) | Art. 8 bis: derecho a oponerse a decisiones individuales **automatizadas** (incluida la elaboración de perfiles), a pedir intervención humana y a expresar su punto de vista; excepciones: contrato, consentimiento explícito o ley | Medio | Todo scoring que decide en automático sobre personas | Todo responsable de datos en Chile *(verificar fecha de publicación y texto final)* |
| **EU AI Act**, Anexo III 5(b) | Sistemas para evaluar solvencia o puntuar crédito de personas naturales son de **alto riesgo** (excepción: detección de fraude): gestión de riesgos, gobernanza de datos, documentación técnica, registro de eventos, supervisión humana | Alto | Si se opera en la UE; como estándar de documentación | Proveedores y usuarios en la UE. Obligaciones del Anexo III movidas de ago-2026 a **2-dic-2027** por el *Digital Omnibus* (aprobación final del Consejo 29-jun-2026; publicación en el DOUE pendiente a julio *(verificar)*) |

Tres lecturas. Primera: el centro de gravedad se movió de «un programa uniforme» (SR 11-7) a «intensidad proporcional al riesgo» (SR 26-2, SS1/23). Segunda: en Chile **no encontré** una norma CMF específica de riesgo de modelo equivalente a SR 26-2 para bancos, ni una para prestamistas no bancarios; lo exigible hoy viene por B-1 (provisiones), por el proceso de autorización de metodologías internas (en consulta) y por normas generales de gestión de riesgos *(verificar con la normativa vigente de su institución)*. Tercera: para una fintech de crédito de motos, el estándar exigible es bajo, pero el **financiador** (un banco, un fondo, una securitización) sí puede exigir un expediente al nivel de SR 11-7; tenerlo es un activo comercial.

### 4.2 Técnicas de trazabilidad

| Técnica | Qué garantiza | Costo | Cuándo usarla | Quién la usa |
|---|---|---|---|---|
| Log plano (JSONL) | Historia legible si nadie la tocó | Mínimo | Siempre, como base | Toda librería (`nikodym` registra 398 eventos) |
| Hash por evento | Detecta cambio de *ese* evento, si el hash se guardó aparte | Mínimo | Cuando cada evento se referencia desde otro documento | — |
| Cadena de hashes | Detecta edición sin recalcular (atacante torpe) | Bajo | Siempre: el «arreglito» de última hora es el caso frecuente | Curso (clase 6) |
| HMAC encadenado | Detecta reescritura por quien no tiene la clave; **no** truncamiento | Bajo + gestión de claves | Cuando operadores del pipeline no deben poder reescribir | Logs seguros (Schneier–Kelsey) |
| Sello $(h_n,n)$ con custodia separada | Detecta reescritura y truncamiento si el custodio es independiente | Bajo + proceso | Siempre que exista un custodio (auditoría, riesgo) | Curso |
| Almacenamiento WORM / retención inmutable | Impide sobrescribir durante la retención | Medio | Expedientes con obligación de retención | Regla 17a-4 (SEC), servicios de *object lock* |
| Sellado de tiempo RFC 3161 | Existencia de los bytes antes de `genTime`, verificable por terceros | Bajo por token; requiere TSA confiable | Sellos de corridas aprobadas, actas | Firma electrónica avanzada, facturación |
| Árbol de Merkle + pruebas de inclusión | Verificar un evento con $O(\log n)$ hashes sin revelar el resto | Bajo–medio | Auditor externo que no puede ver datos personales; logs muy grandes | Certificate Transparency (RFC 6962/9162) |
| Firma digital del artefacto | Autoría y no repudio del JSON aprobado | Medio (PKI) | Paso a producción | Cadenas de suministro de software |
| Anclaje en blockchain pública | Sello sin TSA de confianza | Medio; latencia | Rara vez necesario en banca: una TSA acreditada basta | — |

### 4.3 Documentación y registro

| Documento | Qué resuelve | Costo | Cuándo | Referencia |
|---|---|---|---|---|
| **Model card** | Una a dos páginas: uso previsto y no previsto, datos, desempeño, limitaciones, gobierno | Bajo si se genera desde la corrida | Siempre | Mitchell et al. (2019); plantilla del Lab 3 (7 secciones) |
| **Datasheet** del dataset | Motivación, composición, recolección, preprocesamiento, usos, mantenimiento de los **datos** | Bajo | Matrices reutilizadas por varios modelos | Gebru et al. (2021) |
| **FactSheet** | Declaración de conformidad del proveedor de un servicio de IA | Medio | Modelos comprados o vendidos | Arnold et al. (2019) |
| **Informe técnico** | Razonamiento completo, reproducible por un tercero | Alto | Aprobación y revalidación | Plantilla del curso: resumen + 7 capítulos + 3 anexos |
| **Resumen ejecutivo** | Qué se firma y qué pasa si se firma | Bajo | Comité | Cinco bloques del curso |
| **Informe de validación** | Opinión independiente con hallazgos y severidad | Alto | Antes de aprobar; revisión periódica | SR 11-7 / SR 26-2, EGIM |
| **Inventario** | Qué es modelo, qué no (y por qué), tier, dueño, validador, estado, próxima revisión | Bajo por fila | Siempre, incluidos EUC y modelos de proveedor | EGIM («registro»), SR 26-2; `M22_inventario_modelos.csv` |

**Tiering.** Ninguna norma revisada fija una fórmula. La que usa este módulo (convención, en la hoja `Tiering` y en el notebook) combina materialidad por exposición (1/2/3 según < 1.000, 1.000–10.000 o > 10.000 MM CLP), uso (1 informativo, 2 apoya decisión humana, 3 decide en automático o es regulatorio) y complejidad (1 regla, 2 estadístico interpretable, 3 ML o caja negra): puntaje $=0{,}4\,m+0{,}3\,u+0{,}3\,c$, tier 1 si $\ge2{,}4$, tier 2 si $\ge1{,}7$. Con ella, el scorecard de Austral y el score de bureau del proveedor son tier 1; un scorecard de motos de 4.500 MM CLP es tier 2; la planilla de pricing (EUC) es tier 2, no «nada».

### 4.4 Ciclo de vida: qué evidencia produce cada etapa

| Etapa | Pregunta que cierra | Evidencia que debe quedar | Quién firma | Qué la invalida |
|---|---|---|---|---|
| **Desarrollo** | ¿El modelo es razonable para este uso? | `config` sellado, trail con decisiones del embudo, artefacto con hash, métricas en DEV/HO/OOT, informe técnico base, model card con usos no previstos | Jefe de Modelos (A) | Métricas solo en DEV; fuga temporal; números que no salen de la corrida |
| **Validación** | ¿Otro llega a lo mismo y está de acuerdo? | Replicación desde el expediente (mismos hashes), revisión conceptual, *challenger*, sensibilidad, hallazgos con severidad, veredicto | Validación (R/A) | Validador que desarrolló o que reporta al desarrollador |
| **Aprobación** | ¿La institución acepta el riesgo residual? | Acta: uso aprobado, condiciones, límites, gatillos sellados, vigencia | Comité (R/A) | Aprobar sin gatillos o sin plan para los hallazgos abiertos |
| **Implementación** | ¿Producción puntúa lo mismo que el notebook? | Prueba de paridad, contrato de datos, hash del artefacto desplegado = aprobado (M21) | TI (R/A) | Re-ajuste de binning en producción (bug de los 587 casos) |
| **Monitoreo** | ¿Sigue siendo cierto? | Tablero histórico en el expediente, gatillos evaluados con fecha, overrides con desempeño | Modelador (R), Jefe (A) | Indicadores sin dueño ni acción |
| **Revisión periódica** | ¿Sigue sirviendo para lo aprobado? | Revalidación proporcional al tier (anual en tier 1 con la convención del módulo; EGIM exige validación anual para modelos internos) | Validación / Comité | «Se revisó» sin nueva evidencia |
| **Cambio** | ¿Qué cambió y cuánto? | Clasificación *semver* (δ = *minor*; variables = *major*), paridad, acta | según magnitud | Cambios sin versión nueva |
| **Retiro** | ¿Qué lo reemplaza y qué se conserva? | Fecha de baja, sucesor, expediente archivado (retención), consumidores desconectados | Comité | Modelo «apagado» que sigue alimentando otro sistema |

La columna «qué la invalida» es la lista de lo que un auditor busca primero. El retiro es la etapa más olvidada: un scorecard reemplazado que sigue escribiendo su score en una tabla que lee el motor de pricing es un uso no previsto que nadie aprobó.

### 4.5 Qué revisa un validador (y cómo preparar el expediente)

Las tres piezas de la validación de SR 11-7 siguen siendo la mejor lista de verificación, aunque SR 26-2 las formule con más flexibilidad:

1. **Solidez conceptual.** ¿El target, la población, la ventana, el esquema muestral y la elección de variables responden al uso? ¿Los supuestos están escritos? (Clases 1–3; Serie 1 · M2–M4.) El validador lee el informe técnico **y** el trail de decisiones: si el embudo del informe no coincide con las decisiones registradas, hay un problema.
2. **Monitoreo y verificación de procesos.** ¿Lo que corre es lo que se aprobó? Replicación desde el expediente (en el notebook: cuatro hashes iguales), paridad del artefacto (M21), contrato de datos, *benchmarking* contra un *challenger* (en el notebook: un GBM que no mejora de forma concluyente, IC de la diferencia [−0,012; +0,030]).
3. **Análisis de resultados.** *Backtesting* de discriminación y calibración fuera de desarrollo (M12, M19), por tramos y no solo global, y sensibilidad de las decisiones a los supuestos (en el notebook: ±0,10 en δ mueve la aprobación TTD entre 65,1% y 70,0%).

**Cómo prepararse: el expediente como producto.** Un validador tiene poco tiempo y una pregunta: *¿puedo llegar a los mismos números sin llamar al desarrollador?* El expediente debe permitirlo: (i) un `README` con el comando único que reproduce la corrida desde el `config` sellado; (ii) el lineage completo y declarado lo que falta; (iii) cada número del informe con su origen; (iv) las decisiones del embudo en el trail; (v) las limitaciones ya escritas, para que el validador no las «descubra»; (vi) la lista de lo que **no** se probó. Un expediente así acorta la validación de semanas a días y cambia el tono del informe: de «hallazgos» a «observaciones».

### 4.6 Librerías de modelamiento dentro del gobierno

`nikodym`, `optbinning`, `statsmodels` o `scikit-learn` hacen cálculos que terminan en decisiones: son **componentes del modelo**. Tres prácticas: (a) *pin* exacto en el archivo de bloqueo con hash (sin él, `nikodym` declara «lineage parcial: sin hash de uv.lock»); (b) *golden test* por librería: un dataset pequeño, fijo, con salidas selladas (bins de `optbinning`, IV, β de `statsmodels`), que corre en cada actualización y cuyo fallo bloquea el despliegue; (c) contraste con una implementación independiente de los cálculos centrales (Gini, IV, binomial, δ), como hace cada notebook de esta serie (catálogo de defaults y discrepancias en M23). Una actualización de librería que cambia una salida es un **cambio de modelo** con su versión y su acta, aunque nadie haya tocado el código propio. En `optbinning`, por ejemplo, un cambio del solver o de sus parámetros por defecto puede mover cortes; el *golden test* lo detecta, la lectura del *changelog* no siempre.

### 4.7 Limitaciones: genéricas vs útiles

| Genérica (no sirve) | Útil (número + qué no se puede concluir + consecuencia) |
|---|---|
| «El modelo tiene supuestos.» | «Default = 90+ DPD a 12 meses con indeterminados (30–89) excluidos: la PD no aplica a quien hoy está en 30–89 días; no usar en cobranza temprana.» |
| «Los datos podrían no ser representativos.» | «Modelado sobre aprobados: no observa al 33,1% rechazado; en el territorio que la política rechazaba la mora fue 9,4% contra 5,1% previsto (n = 128, p 0,04): no bajar el cutoff sin reject inference.» |
| «La calibración puede variar en el tiempo.» | «En OOT, PD esperada 6,08% y mora 5,94%, pero HL p < 0,001: falla por tramos; recalibrar δ no lo corrige; no pasar a producción sin diagnóstico.» |
| «Se recomienda monitorear.» | «La cadena del trail detecta edición sin resellado, no la reescritura completa: la evidencia exige el sello externo con custodia de Auditoría.» |
| «El modelo podría usarse mal.» | «No validado para provisiones: horizonte de 12 meses sobre solicitudes, no PD lifetime sobre cartera vigente.» |


---

## 5. Cuándo falla: trampas y modos de falla

**5.1 El informe no corresponde al notebook.** *Síntoma:* el informe dice Gini 0,66 y el notebook 0,631. *Causa:* documentos escritos a mano desde una corrida anterior. *Detección:* cada número del documento debe llevar al lado el `run_id` o el hash de la corrida que lo produjo; un test de CI que extrae los números del documento generado y los compara con los del JSON de métricas. *Qué hacer:* generar los documentos desde la corrida (plantillas con campos, no texto libre para los números) y sellar documento y corrida juntos. Es el invalidante n.º 1 del curso por una razón: destruye la credibilidad de **todo** el expediente, no solo de ese número.

**5.2 Limitaciones genéricas.** *Síntoma:* «el modelo tiene supuestos», «los datos podrían no ser representativos». *Causa:* la sección se escribe al final, por cumplir. *Detección:* un *lint* mínimo (número + consecuencia + extensión) atrapa las vacías; el notebook rechaza las tres genéricas de ejemplo y acepta las cuatro declaradas. El lint no reemplaza al validador: una limitación con número puede seguir siendo irrelevante. *Qué hacer:* cada limitación dice qué **no** se puede concluir y qué consecuencia práctica tiene («OOT: PD 12,48% vs mora 15,00%, binomial p = 2,9·10⁻⁷: no pasar a producción sin diagnóstico por tramos»).

**5.3 Model card sin usos no previstos (*use creep*).** *Síntoma:* el scorecard de admisión aparece como insumo de provisiones, de pricing o de cobranza. *Causa:* «total, da una probabilidad». *Detección:* el inventario registra consumidores del modelo (qué sistemas leen su salida); una auditoría de linaje de salida (quién consume la tabla de scores). *Qué hacer:* cada uso no previsto con su **razón** (horizonte, población, efecto de selección); todo nuevo uso pasa por validación como si fuera un modelo nuevo. Es la segunda fuente de riesgo de modelo de SR 11-7 —mal uso de un modelo correcto— y ninguna métrica del tablero la detecta.

**5.4 Validación que no es independiente.** *Síntoma:* el validador trabaja en el mismo equipo, revisa su propio código o solo relee el notebook. *Causa:* falta de dotación. *Detección:* la RACI con `COUNTIF` (el Modelador con R o A en «validar» o «aprobar» es una violación verificable); el informe de validación no trae **replicación** ni *challenger*. *Qué hacer:* en instituciones pequeñas, validación externa o revisión cruzada entre equipos con reporte distinto; como mínimo, que quien valida no reporte a quien desarrolló (EGIM pide unidades separadas con reporte a distintos miembros de la alta gerencia).

**5.5 Hash no canónico.** *Síntoma:* dos corridas idénticas dan `config_hash` distinto, o el mismo evento tiene un hash en Python y otro en el servicio Java. *Causa:* orden de claves, `1` vs `1.0`, NaN serializado como `NaN`, texto en NFD. *Detección:* test de idempotencia (serializar–deserializar–serializar da los mismos bytes); test cruzado con la otra implementación. *Qué hacer:* una sola función de canonicalización versionada (o RFC 8785 en todos los lenguajes), `allow_nan=False`, NFC como política declarada.

**5.6 `data_hash` que no se reproduce tras actualizar pandas.** *Síntoma:* el validador re-ejecuta y el `data_hash` difiere, aunque nadie tocó los datos. *Causa:* hash de representación física (dtypes, orden, índice) o implementación interna de la librería. *Detección:* la tabla de perturbaciones del notebook: 7 falsas alarmas de `hash_pandas_object` en 9 perturbaciones que no cambian el contenido. *Qué hacer:* hash canónico de contenido para el lineage; hash físico (del archivo) como verificación complementaria; ambos definidos por escrito en el expediente.

**5.7 Cadena sin ancla (falsa seguridad).** *Síntoma:* se vende como «audit trail inmutable» un JSONL encadenado en el mismo bucket que opera el pipeline. *Causa:* confundir detección de edición descuidada con prueba de integridad. *Detección:* la pregunta del curso: ¿dónde está el ancla y quién puede escribir ahí? *Qué hacer:* sello con custodia separada; idealmente token RFC 3161 o almacenamiento con retención inmutable.

**5.8 El sello se compara con lo que no corresponde.** *Síntoma:* el verificador acepta un log editado. *Causa:* comparar el sello con el último hash **guardado** en vez de recalcular la cadena desde los eventos: el atacante torpe deja intacto el hash terminal y pasa. *Detección:* test adversarial en CI que ejecuta los seis ataques de la matriz y exige que el sello los rechace todos. *Qué hacer:* la verificación nunca confía en hashes que vinieron dentro del objeto verificado.

**5.9 Truncamiento silencioso.** *Síntoma:* el log termina «antes» y todo verifica. *Causa:* un prefijo de una cadena válida es una cadena válida, también con HMAC. *Detección:* exigir evento de cierre (`run_end`/`corrida_abortada`) y comparar $n$ con el sello. *Qué hacer:* el sello incluye siempre el número de eventos.

**5.10 Merkle con duplicación.** *Síntoma:* dos logs distintos con la misma raíz. *Causa:* completar niveles impares duplicando el último nodo. *Detección:* test que compara `[a,b,c]` con `[a,b,c,c]`. *Qué hacer:* RFC 6962 (promover sin duplicar, prefijos de dominio).

**5.11 Gatillos escritos después del dato.** *Síntoma:* el umbral «coincide» con el valor observado, justo por encima o por debajo. *Causa:* el jardín de senderos que se bifurcan. *Detección:* la tabla de gatillos está sellada (hash + fecha) **antes** del primer mes de monitoreo; el acta de aprobación la referencia. *Qué hacer:* aprobar los gatillos junto con el modelo; cambiarlos exige acta y justificación que no dependa del valor observado.

**5.12 Fatiga de alarmas.** *Síntoma:* el tablero siempre tiene algo amarillo y nadie actúa. *Causa:* $k$ indicadores al 5% dan 37% de meses con alguna alarma falsa ($k=9$). *Detección:* la hoja `Fatiga_alarmas`; la historia del tablero (¿cuántos amarillos se resolvieron solos?). *Qué hacer:* jerarquía con persistencia (dos trimestres), Bonferroni u otro control de multiplicidad para gatillos caros, y distinguir indicadores de **diagnóstico** (muchos, sin acción automática) de **gatillos** (pocos, con acción).

**5.13 Overrides sin registro (*override creep*).** *Síntoma:* la mora sube y el modelo «funciona»; o el modelo es culpado de algo que decidió la mesa. *Causa:* las excepciones no se registran con motivo ni se evalúan. *Detección:* tasa de overrides por emisor y por segmento; desempeño de la cohorte de overrides contra su PD. *Qué hacer:* motivo codificado obligatorio, límite de tasa como gatillo, revisión de concentración. Recordar la potencia: ~250 overrides maduros para detectar 15% vs 10%.

**5.14 La herramienta no validada.** *Síntoma:* un cambio de versión de la librería cambia el binning o el IV, y nadie lo nota. *Causa:* dependencia sin *pin*, o *pin* sin prueba. *Detección:* `assert libreria.__version__ == PIN` (como el curso con `nikodym 1.11.0`) + un *golden test* (dataset fijo con resultado esperado sellado) que corre en cada actualización. *Qué hacer:* tratar la librería como un modelo de proveedor: inventario, pin, pruebas de regresión, benchmarking contra una implementación independiente (el notebook compara el Gini propio con `sklearn` y la binomial propia con `scipy`).

**5.15 Inventario incompleto.** *Síntoma:* la planilla de pricing, el score del bureau o el modelo de fraude no están en el inventario. *Causa:* «no es un modelo» o «no es nuestro». *Detección:* barrido de consumidores de datos y de decisiones automáticas. *Qué hacer:* inventariar también lo que se decidió que **no** es modelo, con la razón; los modelos de proveedor con validación de salidas y monitoreo.

**5.16 «Reproducible, luego aprobado».** *Síntoma:* un expediente impecable en lineage y trail sobre un modelo con calibración fallida. *Causa:* confundir evidencia de proceso con evidencia de desempeño. *Detección:* la última línea del paso 9 de la demo: «reproducible y documentado no significa aprobado». *Qué hacer:* el veredicto de validación es un campo estructural que marca y firma el validador; la herramienta no lo calcula.

---

## 6. Puente con ingeniería

Piensa el expediente como el **producto de build** de un pipeline declarativo, no como una carpeta que alguien arma al final.

**Contrato del pipeline.** Entrada: `config.yaml` (receta completa: fuente de datos y su hash esperado, semilla, candidatas, umbrales de selección, escala, ancla de calibración, política, gatillos, bloque de gobierno con propósito, usuarios y usos no previstos). Salida: un directorio inmutable por corrida:

```
expediente/<modelo_id>/<version>/<run_id>/
  config.json                 # canónico; su SHA-256 = config_hash
  lineage.json                # run_id, config_hash, data_hash (canónico y físico), root_seed,
                              # git_sha, git_dirty, lock_hash, versiones, creado_en
  audit_trail.jsonl           # eventos canónicos, uno por línea
  trail_hashes.json           # cadena h_1..h_n (o HMAC)
  sello.json + sello.tsr      # (h_n, n, run_id) y token RFC 3161 de la TSA
  artefacto.json              # lo que puntúa (M21); su hash va en el trail
  metricas.json               # Gini/KS/PSI/backtesting por muestra
  model_card.{json,md}        # generado desde metricas + config + lineage
  informe_tecnico.{qmd,html}  # base generada; secciones ✎ quedan marcadas
  gatillos.json               # tabla de gatillos sellada con la aprobación
  MANIFEST.json               # SHA-256 de cada archivo anterior + raíz de Merkle
```

**Qué se congela y qué se versiona.** Se congela todo lo que depende de la población (M21) y la tabla de gatillos. Se versiona el modelo con *semver*: cambio de δ o de cutoff = *minor* con acta; nuevas variables, nuevos cortes o nuevo target = *major* con validación completa; corrección de implementación sin cambio de salida = *patch* con prueba de paridad. El expediente de cada versión no se sobrescribe nunca.

**Tests tipo CI (fallan el build):**

1. `test_canon_idempotente`: `canon(json.loads(canon(x))) == canon(x)` para config, artefacto y eventos.
2. `test_canon_rechaza_nan`: cualquier NaN o infinito en payloads falla al registrar (no al final).
3. `test_trail_integro`: `verificar_cadena`, `verificar_estructura` y `verificar_sello` (recalculando desde eventos) pasan.
4. `test_ataques`: los seis ataques del notebook son rechazados por el sello; el torpe también por la cadena. Es un test de la **verificación**, no del log.
5. `test_determinismo`: dos corridas con el mismo config dan la misma huella de contenido del trail y el mismo `hash_artefacto`.
6. `test_lineage_completo`: si `git_sha`, `git_dirty=False` o `lock_hash` faltan, la corrida se marca `no_reproducible` y el model card lo declara (no se puede aprobar una corrida así en producción).
7. `test_model_card`: siete secciones; ≥ 4 usos no previstos; limitaciones pasan el lint; cada número del card existe en `metricas.json`.
8. `test_raci`: exactamente una A por actividad, ≥ 1 R, modelador fuera de validar/aprobar.
9. `test_gatillos`: cada gatillo tiene las cinco partes; `disparado` calculado; al menos dos responsables distintos.
10. `test_golden_libreria`: el dataset dorado reproduce el scorecard y las métricas selladas con la versión *pineada* de la librería.

**Invariantes verificables (para el validador, no solo para CI):** todo número de un documento es trazable a `metricas.json` de un `run_id`; el `hash_artefacto` en producción es el del expediente aprobado; `n` del sello = líneas del JSONL; la raíz de Merkle del `MANIFEST` cubre todos los archivos; el token TSA verifica contra el hash del sello.

**Dónde vive cada control.** La librería (`nikodym` o la propia) **registra**. La plataforma **encadena, sella y custodia**: el job de CI que cierra la corrida calcula la cadena, pide el token a la TSA y escribe el expediente en un almacenamiento con retención inmutable al que el equipo de modelos no tiene permiso de borrado. Separar esos permisos es la RACI llevada a IAM.

**Eventos: esquema mínimo.** `{n, run_id, t (reloj lógico), tipo ∈ {run_start, artifact, decision, run_end, run_aborted}, paso, evento, payload}`. El reloj lógico hace el trail determinista; la hora de pared va una sola vez en `lineage.creado_en`. Los eventos de decisión llevan `regla`, `umbral`, `valor`, `accion`: es lo que permite a un validador reconstruir el embudo (en el notebook: 11 decisiones de IV, 6 de correlación, signos, δ y backtesting).

---

## 7. Numpy desde cero vs librerías

| Cálculo | Desde cero en el notebook | Librería estándar | Diferencias / convención | En producción |
|---|---|---|---|---|
| Canonicalización JSON | `canon_desde_cero` (escapes y `repr` explícitos) | `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False)` | Idénticas byte a byte en Python; **no** idénticas a RFC 8785 (números) | Una función versionada; JCS si hay más de un lenguaje |
| SHA-256 / HMAC | — (no se implementa criptografía a mano) | `hashlib.sha256`, `hmac.new(..., hashlib.sha256)` | — | Nunca implementar primitivas propias |
| Cadena de hashes | `encadenar`, `verificar_cadena`, `verificar_estructura`, `verificar_sello` | `nikodym.audit` registra; la cadena es institucional en 1.11.0 | La verificación debe recalcular desde eventos | Propia, pequeña y testeada, o un servicio de logs con verificación |
| Merkle | `mth` recursivo (RFC 6962) y `raiz_iterativa` | Librerías de *transparency logs* *(no se evaluaron aquí)* | Duplicar vs promover el nodo impar cambia la raíz | RFC 6962/9162 con pruebas de consistencia |
| Hash de DataFrame | `hash_df_canonico` | `pandas.util.hash_pandas_object` (+ esquema, como el curso) | Contenido vs representación física; estabilidad entre versiones | Canónico para lineage; físico como verificación intra-entorno |
| Gini | `gini_numpy` (Mann-Whitney con rangos promedio) | `sklearn.metrics.roc_auc_score` | Idénticos (tolerancia $10^{-12}$) | `sklearn` o el propio con test contra `sklearn` |
| Binomial exacta | `binom_dos_colas_numpy` con `math.lgamma` | `scipy.stats.binomtest` | Mismo criterio de dos colas (suma de P ≤ P(x), tolerancia $10^{-7}$) | `scipy` |
| RACI | `chequear_raci_numpy` (matriz de strings) | `chequear_raci_pandas` (`isin`) | Idénticos | Cualquiera, dentro de CI |
| Model card | `card_markdown` | `nikodym.governance.ModelCardBuilder` | La librería agrega limitaciones automáticas del entorno | Librería + bloque humano obligatorio |

Dos observaciones. Primera: en gobierno, «desde cero» no es un ejercicio académico, porque la verificación la tiene que poder hacer **un tercero** sin instalar la herramienta del desarrollador; una cadena de hashes en 20 líneas de Python estándar es auditable, un binario propietario no. Segunda: la librería del desarrollador **también** es un modelo a validar (sección 5.14): el *pin* de versión (`nikodym[scoring]==1.11.0`) sin *golden test* solo garantiza que se instaló lo mismo, no que lo mismo haga lo correcto.

---

## 8. Aplicación: casos y números

### 8.1 Banco Austral: el expediente de la clase 6, auditado

**Estado de las nueve piezas** tras la corrida `nikodym` (run `ab724122…`):

| Pieza | Estado | Lo que falta para un validador |
|---|---|---|
| Informe técnico | ✓ base (8 secciones + 3 anexos, 126 tablas) | Introducción, contexto, veredicto, recomendación, usos no previstos, gatillos |
| Model card | ✓ (identidad, 341 decisiones, 4 limitaciones) | Tabla de desempeño, usos no previstos, gobierno (el Lab 3 lo pide agregar) |
| Resumen ejecutivo | ✎ | Cinco bloques con números de **esta** corrida |
| Notebook reproducible | ✎ | Ejecutado en orden; `git_sha` ausente (Colab) |
| Artefacto versionado | ✓ interno al estudio | Hash del artefacto exportable (la versión a mano: `98845d69…`) |
| Audit trail | ✓ 398 eventos | Cadena y sello **institucionales**; custodia separada |
| Tablero de monitoreo | ✎ | Histórico mensual, no solo el último mes |
| Actas de comité | ✎ | Aprobación, límites de uso, excepciones |
| Informe de validación | ✎ | Replicación, *challenger*, sensibilidad, veredicto firmado |

**Gatillos con los números de Austral** (hoja `Gatillos` de la planilla): vigilancia reforzada **SÍ** (5 indicadores en 🟡 en el tablero de la clase 5); diagnóstico de calibración **SÍ** (HL OOT p < 0,001); recalibración de δ **no evaluable** (hay un solo trimestre OOT); re-desarrollo **no** en sus tres ramas (caída relativa de Gini DEV→OOT $1-0{,}6938/0{,}7831=11{,}4\%<30\%$; KS OOT 0,5616 > 0,20; PSI 0,0081 < 0,25); contingencia **no**; overrides **no evaluable** (no hay registro). Nivel más alto disparado: 2. Esto es exactamente la lectura del curso («diagnosticar por tramos y decidir si basta recalibrar δ; no re-desarrollar sin evidencia de deterioro»), ahora con cada paso calculado y con dueño.

**Una limitación que el curso dejó implícita.** La PD media esperada en OOT (6,08%) está **por encima** de la mora observada (5,94%) y aun así HL rechaza con p < 0,001: el nivel medio calza y la forma no. Un binomial global no lo detectaría. Por eso el diagnóstico es «por tramos» y el gatillo de recalibración de δ (que corrige nivel, no forma) no es la respuesta automática: si el problema es de pendiente, recalibrar δ no lo arregla (M15 y M19). Esa frase, con sus números, es una limitación bien escrita.

### 8.2 El notebook: la misma historia con verdad conocida

La corrida del notebook sobre la cartera sintética (24.000 solicitudes, `root_seed` 20240706) deja **28 eventos** (21 decisiones): de 11 candidatas, 6 pasan IV ≥ 0,10 y la correlación de WoE descarta `uso_tc_prom_3m` (0,714) y `uso_tc_prom_12m` (0,735), quedando 4 variables con β todos negativos (−0,701, −0,760, −0,395, −0,569) y 243,6 malos por parámetro. TC anclada en DEV+HO 11,81%, δ = −0,0251. Gini DEV/HO/OOT 0,564/0,505/0,534; PSI del score DEV→OOT 0,0096 y DEV→TTD 0,0178; pero en OOT la PD media es 12,48% y la mora 15,00% (binomial p = 2,9·10⁻⁷): el deterioro de 2025 **plantado** por el generador. `config_hash` `80960678…`, `data_hash` canónico `e1de84d7…`, artefacto `c13aaca0…` (valores con numpy 2.4.4 / pandas 3.0.2; otra versión de numpy podría generar otra cartera y, con ella, otros hashes: es la razón de registrar versiones).

**Lo que el notebook demuestra:**

- Matriz de ataques: la cadena solo detecta al torpe; la estructura detecta el truncamiento; HMAC detecta todo lo que recalcula pero **no** el truncamiento; el sello detecta los seis.
- Un sello local reescrito junto con el log «cuadra»; el token (simulado) de la TSA lo rechaza.
- Merkle: prueba de inclusión de 5 hashes para 28 eventos; recursiva = iterativa; la variante que duplica acepta `[a,b,c]` y `[a,b,c,c]` con la misma raíz.
- Lineage: dos corridas con distinto `run_id` dan huella de contenido, artefacto y δ idénticos; el validador replica `data_hash`, `config_hash`, `hash_artefacto` y el hash de la salida TTD.
- Hash de DataFrame: canónico 9/9; `hash_pandas_object` con 7 falsas alarmas de 9.
- Model card generado: 4 limitaciones declaradas con números, 5 automáticas (git, archivo de bloqueo, backtesting fallido, CSI de `canal` DEV→TTD = 0,311, sello no custodiado).
- *Challenger* (GBM sobre variables crudas): Gini OOT 0,544 vs 0,534 del campeón; IC 95% de la diferencia [−0,012; +0,030] contiene el cero → la complejidad no se justifica con esta evidencia.
- Overrides al 10% de los 1.487 rechazados (149 casos, 4,3% de los aprobados finales): con información blanda, mora 15,4% vs PD del modelo 19,3% (verdad 11,2%), p = 0,25 —la mesa sabe algo, pero con 149 casos **no alcanza para probarlo**—; con política comercial, mora 32,9% vs 26,6% (p = 0,095): el canal `fuerza_venta` tiene riesgo que el modelo no ve porque el embudo descartó `canal` por IV bajo en DEV. En los tres casos la mora de la cartera sube (7,13% → 7,49% / 8,25% / 8,08%).

### 8.3 Una fintech de crédito de motos: gobierno proporcional

Supuestos ilustrativos (no son datos de ninguna empresa): 1.500 solicitudes al mes en concesionarios, aprobación 60%, cartera vigente de 4.500 MM CLP, equipo de datos de tres personas, financiamiento de un fondo que exige reportes trimestrales.

**Tiering.** Con la convención del módulo, el scorecard de admisión es tier 2 (materialidad 2, uso 3, complejidad 2 → 2,3). El score de bureau que compra es tier 1 si se usa en toda la cartera. La planilla de pricing es EUC tier 2. Eso define la intensidad: validación completa cada dos años y monitoreo trimestral para el scorecard; para el bureau, monitoreo de salidas y *benchmarking*.

**RACI con tres personas.** La regla «quien construye no valida» no desaparece por ser pocos. Opciones, de mejor a peor: validador externo contratado por el directorio o por el financiador; revisión cruzada con el área de finanzas o de riesgo del financiador; como mínimo, que la aprobación la haga un comité con un director que no reporte al líder de datos. La planilla `M22_raci_gatillos.xlsx` marca la violación si el Modelador aparece con R o A en «Validar independientemente».

**Overrides en el concesionario.** El canal de motos tiene un incentivo estructural: el concesionario vende la moto si el crédito se aprueba. Si el vendedor puede «pedir excepción», el *override creep* está garantizado. Mínimo operativo: excepción solo por la mesa central, con motivo codificado (documentos adicionales, pie mayor, aval, cliente recurrente), límite de tasa (gatillo 5, p. ej. 5% de los aprobados) y reporte por concesionario. Con 900 aprobaciones al mes y un 5% de excepciones, se acumulan 45 overrides mensuales: los ~250 casos maduros necesarios para detectar 15% vs 10% se completan en ~6 meses de originación y se leen **12 meses después**. Hasta entonces, el gobierno se hace por tasa y concentración, no por desempeño.

**Expediente mínimo viable.** Config en el repositorio, corrida que emite JSONL con cadena, sello enviado por correo al directorio (custodia separada barata) o sellado RFC 3161 con un proveedor, model card de dos páginas con usos no previstos (refinanciación, cobranza, seguros asociados) y un inventario de 6–10 filas. Todo cabe en un repositorio y un job semanal; es la parte del trabajo que un financiador o un comprador de cartera va a pedir primero.

**Regulación.** Hoy, según entiendo, una financiera de motos que presta con recursos propios no está bajo la Ley Fintec por ese solo hecho (sí lo estaría si ofreciera «asesoría crediticia» a terceros o una plataforma de financiamiento), ni bajo el cap. B-1 (que es para bancos). Sí le aplicará la Ley 21.719 si decide en automático sobre personas: derecho a oposición e intervención humana (art. 8 bis), lo que en la práctica exige **reason codes** (M14) y un procedimiento de revisión humana documentado *(verificar con asesoría legal; fechas de vigencia en evaluación)*.

---

## 9. Preguntas de comité

**1. «Muéstrenme la corrida que produjo este número.»**
*Respuesta modelo:* «El Gini OOT de 0,6938 está en `metricas.json` de la corrida `ab724122…`, con `config_hash` `8bdf3d7b…` y `data_hash` `7e3b19ce…`. El trail tiene 398 eventos, su hash terminal está sellado fuera del log y el informe cita esos mismos identificadores. Validación la replicó desde el expediente sin nuestra ayuda.» Si la respuesta requiere abrir el computador de alguien, el modelo está guardado, no implementado.

**2. «¿Qué NO puede hacer este modelo?»**
*Respuesta modelo:* los cuatro usos no previstos con su razón (provisiones: horizonte y población distintos; cobranza: es de admisión; pricing: no validado el efecto de selección; otros productos: otra definición de default) y las limitaciones con número: «no observa al 33,1% rechazado; en el territorio que la política rechazaba subestima (swap-in 9,4% vs 5,1%, p 0,04); en OOT la forma de la calibración falla (HL p < 0,001) aunque el nivel medio calce».

**3. «¿Quién decide recalibrar, con qué gatillo y en qué plazo?»**
*Respuesta modelo:* «Propone el Jefe de Modelos, decide el Comité con acta, si el binomial global queda en amarillo o rojo dos trimestres seguidos; plazo 60 días desde la detección. Hoy ese gatillo es *no evaluable*: hay un trimestre. Lo que sí está disparado es la vigilancia reforzada y el diagnóstico por tramos, con dueño Jefe de Modelos y plazo 30 días.»

**4. «Su audit trail es inmutable, ¿verdad?»**
*Respuesta modelo:* «No existe el log inmutable; existe el log cuya alteración es detectable frente a cierto atacante. La cadena detecta ediciones descuidadas; el sello de $(h_n,n)$ con custodia en Auditoría detecta reescrituras y truncamientos; el token RFC 3161 prueba que el sello existía en la fecha de aprobación. Nada de eso prueba que las decisiones fueran correctas: eso lo dice el informe de validación.»

**5. «¿Por qué no aprobar ya, si el Gini es estable y el PSI es 0,008?»**
*Respuesta modelo:* «Porque ranking y nivel son preguntas distintas. El Gini dice que ordenamos bien; la calibración dice que las PD que usaremos para el cutoff y la pérdida esperada no calzan por tramos. Un error de 0,10 en δ mueve el score 2,9 puntos y la aprobación entre 1,7 y 3,2 pp en el lote actual (sección 3.1): es un riesgo de decisión, no de estadística.»

**6. «El modelo lo validó la misma consultora que lo desarrolló. ¿Sirve?»**
*Respuesta modelo:* «Sirve como revisión de calidad, no como validación independiente: quien valida no puede tener incentivo en el resultado ni haber tomado las decisiones que evalúa. SR 26-2 pide pericia, independencia suficiente para objetividad y posición para provocar cambios. Si no hay alternativa interna, que el validador lo contrate y le reporte el directorio.»

**7. «¿Cuántos overrides hubo y cómo les fue?»**
*Respuesta modelo:* «Tasa sobre aprobados, desglose por emisor y motivo, y mora de la cohorte madura contra la PD que el modelo le asignaba. Si los overrides rinden **mejor** que su PD de forma consistente, no es un éxito de la mesa: es información que el modelo no tiene y entra al re-desarrollo. SR 11-7 lo decía así: una tasa alta de overrides, o overrides que mejoran sistemáticamente el desempeño, suelen indicar que el modelo necesita revisión.»

**8. «¿La librería que usaron está validada?»**
*Respuesta modelo:* «Está *pineada* (`nikodym[scoring]==1.11.0`), con `assert` de versión y un *golden test* que reproduce métricas selladas; sus cálculos centrales (Gini, binomial, IV) se contrastaron con implementaciones independientes. Una actualización de la librería es un cambio de modelo: pasa por paridad y por acta.»

---

## 10. Ejercicios

**Ejercicio 1 (fatiga, a mano).** Un tablero tiene 6 indicadores con semáforo amarillo al 5%, independientes. (a) Probabilidad de al menos un amarillo en un mes con el modelo perfectamente estable. (b) Número esperado de amarillos falsos en un año. (c) Si el gatillo de recalibración exige el mismo indicador en amarillo dos trimestres seguidos, ¿cuál es la probabilidad anual de dispararlo en falso para un indicador trimestral?

<details><summary>Solución</summary>

(a) $1-0{,}95^6=1-0{,}7351=26{,}5\%$. (b) $6\cdot12\cdot0{,}05=3{,}6$ amarillos falsos al año. (c) Con 4 trimestres hay 3 pares consecutivos; la probabilidad de al menos un par con ambos amarillos es $1-\Pr(\text{ningún par})$. Por la recursión de rachas (o enumerando las 16 secuencias), las secuencias de 4 trimestres sin dos amarillos seguidos son: ninguno ($q^4$), uno en cualquiera de 4 posiciones ($4pq^3$) o dos no adyacentes en las posiciones (1,3), (1,4) o (2,4) ($3p^2q^2$); con $p=0{,}05$, $q=0{,}95$: $0{,}8145+0{,}1715+0{,}0068=0{,}9927$. Probabilidad de disparo en falso ≈ 0,73% al año por indicador, contra 18,5% ($1-0{,}95^4$) si bastara un trimestre.
</details>

**Ejercicio 2 (Merkle, a mano).** Log de 8 eventos $d_0..d_7$. (a) Escribe $\text{PATH}(5,D_8)$ en términos de MTH de sub-rangos. (b) ¿Cuántos hashes tiene la prueba para $n=398$ y para $n=341$? (c) ¿Qué pasaría con la verificación si el auditor recibe $d_5$ con un byte cambiado?

<details><summary>Solución</summary>

(a) $n=8$, $k=4$, $m=5\ge4$: $\text{PATH}(1,D_{4:8})\,:\,\text{MTH}(D_{0:4})$. Dentro de $D_{4:8}$ ($n=4$, $k=2$, $m=1<2$): $\text{PATH}(1,D_{4:6})\,:\,\text{MTH}(D_{6:8})$. Dentro de $D_{4:6}$ ($n=2$, $k=1$, $m=1\ge1$): $\text{PATH}(0,\{d_5\})\,:\,\text{MTH}(\{d_4\})=[\,]\,:\,\text{MTH}(\{d_4\})$. Resultado: $[\text{MTH}(\{d_4\}),\ \text{MTH}(D_{6:8}),\ \text{MTH}(D_{0:4})]$, 3 hashes $=\log_2 8$. (b) $\lceil\log_2 398\rceil=9$ y $\lceil\log_2 341\rceil=9$ (el camino exacto puede ser más corto para hojas en el subárbol derecho incompleto). (c) La hoja recalculada $H(\texttt{0x00}\|d_5')$ difiere, y por la Proposición 1 aplicada al camino, la raíz recalculada difiere de la sellada salvo colisión: la verificación falla (el notebook lo muestra con «hoja alterada en un byte: False»).
</details>

**Ejercicio 3 (modelos de amenaza).** Para cada acción, di qué defensas la detectan (cadena, cadena + estructura, HMAC, sello): (a) borrar el último evento sin recalcular nada; (b) borrar el evento 3 y recalcular la cola con SHA-256; (c) borrar el evento 3, recalcular con la clave HMAC (el atacante es un operador que tiene la clave) y reescribir el sello local; (d) igual que (c) pero existe un token RFC 3161 del sello original.

<details><summary>Solución</summary>

(a) Truncamiento: la cadena y el HMAC **no** (el prefijo es válido); estructura sí (falta `run_end`); sello sí ($n$ distinto). (b) Cadena no, estructura no (si renumera), HMAC sí (no hay clave), sello sí. (c) Ninguna de las cuatro: el atacante tiene la clave y el sello está a su alcance ($\mathcal{A}_3$ con clave). (d) El token detecta: firma el hash del sello **original** con fecha anterior; el sello reescrito no verifica contra él. Moraleja: la custodia de claves y sellos es parte del diseño, no un detalle operativo.
</details>

**Ejercicio 4 (canonicalización).** ¿Qué pares tienen el mismo `canon` (el del notebook)? (a) `{"a":1,"b":[1,2]}` vs `{"b":[1,2],"a":1}`; (b) `{"x":1}` vs `{"x":1.0}`; (c) `{"x":[1,2]}` vs `{"x":[2,1]}`; (d) `"Concepción"` NFC vs NFD; (e) `{"x": float("nan")}` vs `{"x": None}`.

<details><summary>Solución</summary>

(a) Sí (sort_keys). (b) No (`1` vs `1.0`; con JCS serían iguales). (c) No: el orden de una lista es contenido. (d) No: `canon` no normaliza Unicode; `hash_df_canonico` sí (política NFC). (e) El primero lanza `ValueError` (`allow_nan=False`); si la política es tratar NaN como ausencia, hay que convertirlo a `None` **antes** de serializar, y declararlo.
</details>

**Ejercicio 5 (RACI).** En la hoja `RACI`, escribe «R» en Modelador × «Validar independientemente» y «A» en Comité × «Monitorear el tablero mensual». ¿Qué columnas cambian y cuál es el contador final? Luego propón una RACI para un equipo de tres personas (líder de datos, analista, gerente de riesgo) más un directorio, que pase los tres chequeos.

<details><summary>Solución</summary>

La primera fila pasa a «VIOLA: el modelador ejecuta o aprueba» en «Chequeo independencia» (y además tiene 1 A en Validación, así que la A sigue única); la segunda a «ERROR: 2 A». «Filas a revisar» sube de 0 a 2 (el notebook reproduce exactamente esta matriz rota y la marca en esas dos filas). Una RACI viable: construir (analista R, líder A); validar (validador externo R/A contratado por el directorio; gerente de riesgo C); aprobar (directorio o comité con un director independiente R/A); monitorear (analista R, líder A, gerente C); recalibrar (líder R, comité A); excepciones (gerente de riesgo R, comité A; el líder de datos solo I).
</details>

**Ejercicio 6 (potencia de overrides).** El modelo asigna PD media 8% a los overrides; se sospecha que su mora real es 12%. ¿Cuántos overrides maduros se necesitan para detectarlo con α = 5% unilateral y potencia 80%? Con 30 overrides al mes y maduración de 12 meses, ¿cuándo se tiene la respuesta?

<details><summary>Solución</summary>

$n\approx\big((1{,}645\sqrt{0{,}08\cdot0{,}92}+0{,}842\sqrt{0{,}12\cdot0{,}88})/0{,}04\big)^2=\big((1{,}645\cdot0{,}2713+0{,}842\cdot0{,}3250)/0{,}04\big)^2=\big((0{,}4463+0{,}2736)/0{,}04\big)^2=(17{,}997)^2\approx 324$. Con 30 al mes, 11 meses de originación; con 12 meses de maduración, la respuesta llega ~23 meses después del primer override. Por eso el gobierno de overrides en los primeros dos años es por tasa, motivo y concentración.
</details>

**Ejercicio 7 (limitaciones y usos no previstos).** Escribe tres limitaciones para el modelo del notebook que pasen el lint **y** que un comité considere útiles, y cuatro usos no previstos para un scorecard de admisión de motos.

<details><summary>Solución</summary>

Limitaciones: (1) «En OOT la PD media es 12,48% y la mora 15,00% (binomial p = 2,9·10⁻⁷): el nivel queda corto en 2,5 pp si el deterioro de 2025 persiste; no usar la PD para pérdida esperada sin recalibrar.» (2) «El CSI de `canal` DEV→TTD es 0,311 y el modelo no usa `canal`; la política comercial de overrides en `fuerza_venta` muestra mora 32,9% vs PD 26,6%: el modelo subestima ese canal.» (3) «El *challenger* GBM no mejora el Gini OOT de forma concluyente (IC de la diferencia [−0,012; +0,030]); la estructura no lineal no justifica hoy un modelo más complejo, pero la conclusión descansa en 4.767 casos OOT.» Usos no previstos (motos): refinanciación o reprogramación (población y horizonte distintos); cobranza (sin variables de comportamiento); cotización de seguros asociados (otro evento); aprobación de clientes sin licencia o de flotas/empresas (otro segmento y otra definición de default).
</details>

**Ejercicio 8 (código: el bug del sello).** Escribe `verificar_sello_malo(hashes_guardados, sello)` que compare el sello con el último hash guardado y muestra, con el ataque torpe del notebook, que lo deja pasar. Luego escribe un test de CI que falle si alguna función de verificación acepta alguno de los seis ataques.

<details><summary>Solución</summary>

```python
def verificar_sello_malo(hashes_guardados, sello):
    return hashes_guardados[-1] == sello["hash_terminal"] and len(hashes_guardados) == sello["n_eventos"]

ev, hs = atacar(EVENTOS, HASHES, "torpe: editar sin recalcular", 12)
assert verificar_sello_malo(hs, SELLO)          # ¡pasa! el torpe no tocó h_n
assert not verificar_sello(ev, SELLO)           # la versión correcta recalcula y lo rechaza

def test_verificacion_rechaza_ataques():
    for tipo in TIPOS_ATAQUE:
        ev, hs = atacar(EVENTOS, HASHES, tipo, 12)
        assert not verificar_sello(ev, SELLO), tipo
```
El test es sobre la **función verificadora**: protege contra regresiones del control, que es donde estaba el error.
</details>

**Ejercicio 9 (tiering).** Un modelo de fraude de proveedor, usado para bloquear transacciones en automático sobre una cartera de 800 MM CLP. Calcula materialidad, puntaje y tier con la convención del módulo. ¿Estás de acuerdo con el resultado? ¿Qué ajustarías de la regla?

<details><summary>Solución</summary>

Materialidad 1 (< 1.000), uso 3, complejidad 3: puntaje $0{,}4+0{,}9+0{,}9=2{,}2$ → tier 2. Es discutible: el daño de un bloqueo erróneo no escala con la exposición de la cartera sino con el volumen de transacciones y el daño reputacional/regulatorio (Ley 21.719: decisión automatizada). La regla debería admitir un **mínimo de tier** por uso («decide en automático sobre personas» → tier ≤ 2 siempre, y tier 1 si además es caja negra de proveedor), o medir materialidad por número de decisiones, no solo por monto.
</details>

**Ejercicio 10 (diseño de gatillo).** Escribe, con sus cinco partes, un gatillo para el CSI de una variable **no** usada por el modelo (como `canal`) y justifica por qué no debe disparar re-desarrollo por sí solo.

<details><summary>Solución</summary>

Condición: CSI DEV→lote de `canal` > 0,25 dos meses seguidos. Valor de hoy: 0,311 (un mes: no disparado por persistencia). Quién decide: Jefe de Modelos. Acción: análisis de mora por canal en la última cohorte madura y del desempeño de overrides por canal; si la mora por canal difiere de la PD del modelo con p < 0,01, se abre evaluación de re-desarrollo (que decide el Comité). Plazo: 30 días. No debe disparar re-desarrollo solo porque un cambio de mezcla en una variable no usada cambia la población, pero no necesariamente la relación score–riesgo; lo que importa es si la calibración por canal se rompe. Es un indicador de **diagnóstico** que alimenta a un gatillo, no un gatillo de acción cara.
</details>

---

## 11. Referencias

**Regulación y supervisión (verificadas a septiembre de 2026):**

- Board of Governors of the Federal Reserve System & OCC (2011). *SR 11-7: Supervisory Guidance on Model Risk Management* (OCC Bulletin 2011-12). — El vocabulario de la disciplina: riesgo de modelo, *effective challenge*, tres piezas de la validación, overrides. Rescindida en abril de 2026, pero sigue siendo la referencia que cita toda la literatura.
- Federal Reserve (17-abr-2026). *SR 26-2: Revised Guidance on Model Risk Management*; OCC (2026). *Bulletin 2026-13: Model Risk Management: Revised Guidance*. — El reemplazo: enfoque por riesgo inherente × materialidad, definición «compleja» de modelo, exclusión de IA generativa, carácter no vinculante. Leer junto con análisis de firmas (p. ej. Orrick, abril de 2026) para el detalle de lo rescindido.
- OCC (6-oct-2025). *Bulletin 2025-26: Model Risk Management: Clarification for Community Banks*. — Antecedente directo de la proporcionalidad de 2026.
- Prudential Regulation Authority (2023). *SS1/23 — Model risk management principles for banks* (vigente desde 17-may-2024). — Cinco principios y «model risk mitigants»; la formulación moderna más clara.
- European Banking Authority (2017). *EBA/GL/2017/16 — Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures* (aplicables desde 1-ene-2021). — Margen de conservadurismo, tasa de largo plazo, revisión anual de estimaciones.
- European Central Bank (2025/2026). *ECB guide to internal models* (revisión de julio de 2025; actualización de junio de 2026). — Capítulo de temas generales: registro de modelos, validación anual, separación desarrollo/validación, ML explicable.
- CMF Chile. *Compendio de Normas Contables*, cap. B-1 (método estándar de consumo vigente desde enero de 2025) y consulta de agosto de 2026 sobre el nuevo **RAN 21-9** (metodologías internas). — Lo exigible a modelos de provisiones en bancos chilenos y hacia dónde va.
- Ley 21.521 (Fintec, 2023) y NCG 502 (CMF, 2024); Ley 21.719 (protección de datos, art. 8 bis). — Marco chileno aplicable a fintech y a decisiones automatizadas; verificar con asesoría legal para cada caso.
- Reglamento (UE) 2024/1689 (AI Act), Anexo III 5(b), y el *Digital Omnibus* (2026). — Calificación de alto riesgo del scoring de personas y su nuevo calendario.

**Trazabilidad y criptografía aplicada:**

- Adams, C., Cain, P., Pinkas, D. & Zuccherato, R. (2001). *RFC 3161: Time-Stamp Protocol (TSP)*; actualizado por RFC 5816. — Qué es un token de sellado de tiempo y qué prueba.
- Laurie, B., Langley, A. & Kasper, E. (2013). *RFC 6962: Certificate Transparency*; RFC 9162 (2021, CT 2.0). — Definición de MTH, pruebas de inclusión y de consistencia; los *transparency logs*.
- Rundgren, A., Jordan, B. & Erdtman, S. (2020). *RFC 8785: JSON Canonicalization Scheme (JCS)*. — Canonicalización interoperable; muestra por qué `json.dumps` no basta entre lenguajes.
- Krawczyk, H., Bellare, M. & Canetti, R. (1997). *RFC 2104: HMAC*. — El MAC que se usa para encadenar con clave.
- Haber, S. & Stornetta, W. S. (1991). How to time-stamp a digital document. *Journal of Cryptology*, 3(2). — El origen del encadenamiento de hashes para sellar documentos.
- Merkle, R. (1988). A digital signature based on a conventional encryption function. *CRYPTO '87*, LNCS 293. — Los árboles de hashes.
- Schneier, B. & Kelsey, J. (1999). Secure audit logs to support computer forensics. *ACM TISSEC*, 2(2). — Logs con claves que evolucionan: protección del pasado ante compromiso del presente.
- Crosby, S. & Wallach, D. (2009). Efficient data structures for tamper-evident logging. *USENIX Security Symposium*. — Logs basados en árboles con pruebas eficientes; el diseño que sigue a la cadena.

**Documentación de modelos:**

- Mitchell, M. et al. (2019). Model Cards for Model Reporting. *Proceedings of FAT\* '19*. — El origen del model card; leer la sección de usos previstos y desempeño desagregado.
- Gebru, T. et al. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12). — La ficha de los **datos**; útil para la matriz modelable.
- Arnold, M. et al. (2019). FactSheets: Increasing trust in AI services through supplier's declarations of conformity. *IBM Journal of Research and Development*, 63(4/5). — Cuando el modelo se compra o se vende.

**Riesgo de modelo y validación en crédito:**

- Basel Committee on Banking Supervision (2005). *Studies on the Validation of Internal Rating Systems* (Working Paper 14). — Clásico sobre validación de sistemas de rating: conceptual, cuantitativa y de uso.
- Morini, M. (2011). *Understanding and Managing Model Risk*. Wiley. — Riesgo de modelo desde la práctica de mercados; útil para la noción de mal uso.
- Scandizzo, S. (2016). *The Validation of Risk Models*. Palgrave Macmillan *(verificar edición)*. — Marco de validación aplicado a modelos de riesgo.
- Siddiqi, N. (2017). *Intelligent Credit Scoring*, 2.ª ed. Wiley. — Capítulos de implementación, overrides y monitoreo de scorecards.
- Anderson, R. (2007). *The Credit Scoring Toolkit*. Oxford University Press. — Overrides, políticas y gobierno del scoring en retail.
- Tasche, D. (2012). Bounds for rating override rates. arXiv:1203.2287. — Formaliza cuántos overrides son «demasiados» y cita el pasaje de SR 11-7 sobre overrides.

**Del curso:** clase 6 (láminas 4–13 y «Para el comité»); `demo_c6_nikodym_austral.ipynb` (pasos 5–9); `demo_c6_bases_austral.ipynb` (secciones 7–12); `lab3_andes.ipynb` (tareas 9–12, Anexos 1 y 2). **De la serie:** Serie 1 · E1 (reject inference), E6 (regulación); Serie 2 · M14 (reason codes), M15 (calibración), M18 (swap-set), M19 (backtesting), M20 (monitoreo), M21 (artefacto y contrato).

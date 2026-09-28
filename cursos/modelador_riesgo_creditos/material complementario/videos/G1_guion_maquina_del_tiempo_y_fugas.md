# G1 · Guión: "La máquina del tiempo y las cuatro fugas"

**Formato objetivo:** video/podcast de 12-15 minutos · Cubre: M2 (arquitectura temporal) + M4 (fugas)
**Tono:** narrador único, conversacional pero preciso. Los [APOYOS VISUALES] son opcionales para versión video; en podcast se omiten.

---

## COLD OPEN (0:00–0:45)

Imagina que entrenas un modelo de riesgo de crédito y una variable te da un AUC de 0.991. Casi perfecto. La tentación es celebrar. La reacción correcta es alarmarse. Porque en riesgo de crédito, un desempeño demasiado bueno casi nunca es un descubrimiento: es una filtración. Hoy te cuento cómo un modelo puede hacer trampa sin que nadie se dé cuenta, por qué la frontera entre lo legal y lo ilegal es una fecha, y cuánto cuesta —medido en métricas concretas— cruzar esa frontera aunque sea por un mes.

## BLOQUE 1 · La regla de oro (0:45–3:00)

Todo modelo de admisión de crédito vive alrededor de un instante: t cero, el mes en que el cliente pide el crédito y el banco tiene que decidir. Ese instante parte el tiempo en dos territorios.

Hacia atrás está la ventana de observación: los últimos 3, 6 o 12 meses de comportamiento del cliente. De ahí salen todas las variables: su mora pasada, cuánto usó la línea, cómo evolucionó su deuda.

Hacia adelante está la ventana de desempeño: los 12 meses siguientes. De ahí sale una sola cosa: el target. ¿Llegó o no llegó a 90 días de mora?

Y la regla de oro, la única que no admite excepción: las variables solo miran hacia atrás, el target solo mira hacia adelante. [APOYO VISUAL: línea de tiempo con t₀ al centro, flechas en colores distintos.]

Suena obvio. El problema es que implementarlo sobre datos reales tiene tres complicaciones que hacen que gente muy competente lo viole sin querer.

Primera: el mes de la solicitud está en curso. Su cierre contable se consolida días o semanas después. Si tu variable usa el cierre de t cero, en desarrollo funciona —los datos históricos ya lo tienen— pero en producción esa información no existe todavía cuando hay que decidir. Por eso la ventana legal termina en t cero menos uno: el último mes ya cerrado. Ni un mes más.

Segunda: no todas las fuentes llegan a la misma velocidad. Tus datos internos cierran en días. El archivo del bureau externo llega con uno o dos meses de rezago. Entonces la ventana legal para variables de bureau no termina en t cero menos uno: termina en t cero menos dos. El ancla temporal es por fuente, no global.

Y tercera, la más traicionera: las bases históricas suelen tener el dato del bureau "backfilled", puesto en el mes al que se refiere, no en el mes en que llegó. Si desarrollas con esa base sin corregir el ancla, entrenas con información un mes más fresca de la que producción tendrá jamás. El modelo valida bien y rinde peor en la calle, y nadie entiende por qué.

## BLOQUE 2 · La máquina del tiempo (3:00–5:00)

Aquí aparece la confusión que todo el mundo trae la primera vez: "¿cómo voy a saber si el cliente paga a 12 meses, si no tengo esa información?".

La respuesta es un cambio de sistema de referencia. Para entrenar, no te paras en el presente: te paras en el pasado. Digamos, octubre de 2024. Desde ahí, "el futuro" —noviembre de 2024 a octubre de 2025— ya ocurrió. Está escrito en tus datos. Ves el futuro entre comillas, porque para ti, hoy, todo eso es historia.

Es una máquina del tiempo: retrocedes a un punto donde el desenlace ya se conoce, reconstruyes qué sabías en ese momento —solo eso—, y le enseñas al modelo a conectar lo uno con lo otro. En producción, con solicitudes de hoy, el futuro sí es desconocido: ahí el modelo predice, y doce meses después el monitoreo dirá si tenía razón.

Y un detalle que importa: cada solicitud viaja con su propio reloj. El crédito de enero se observa hasta el enero siguiente; el de junio, hasta el junio siguiente. Ventanas escalonadas: todas las cosechas medidas con la misma vara de 12 meses. Las solicitudes recientes que no alcanzan a completar su ventana no son "buenas por defecto": quedan sin target, fuera del entrenamiento. Tratarlas como buenas es regalarse una tasa de malos artificialmente baja justo en las cosechas más recientes.

## BLOQUE 3 · Las cuatro fugas (5:00–10:00)

Con ese marco, las fugas de información se dejan clasificar. Son cuatro, y conviene saberlas de memoria porque las vas a encontrar todas.

**Fuga uno: columnas ex-post.** Columnas cuyo valor se escribió después del desenlace. El estado "actual" del cliente. La marca de castigo. El flag de fraude confirmado. La demo canónica: una variable llamada estado_actual con AUC 0.991. Claro que predice: es el desenlace con otro nombre. En producción no existe, porque el desenlace de una solicitud nueva está en el futuro. Detección: pregúntale a cada columna cuándo se escribe. Si la respuesta es "se actualiza", sospecha.

**Fuga dos: fuga de ventana.** La variable es conceptualmente legítima —mora máxima en 6 meses— pero su cálculo roza meses posteriores a t cero. Un error de índice de UN mes. ¿Y cuánto puede costar un mes? En el caso que estudiamos: la variable bien anclada tiene un IV de 0.39. Con la ventana corrida un mes hacia adelante: 0.90. Se duplica. ¿Por qué? Porque ese mes extra incluye la primera cuota del crédito nuevo, y quien parte atrasándose en la primera cuota es casi seguro malo. Un mes de contaminación, el doble de poder predictivo aparente. Y todo falso.

Y la versión extrema: una variable que mide el cambio del cupo del cliente 12 meses después de la solicitud. IV: once punto uno. Un número absurdo. ¿Qué está midiendo? La reacción del banco: cuando un cliente se deteriora, el banco le recorta el cupo. Esa variable no predice el default: lo fotografía. Las fugas más peligrosas son estas, las de acción institucional: columnas que registran decisiones del banco tomadas porque el cliente ya estaba cayendo.

**Fuga tres: fuga de población.** Entrenar con una población que no existirá en producción. El caso clásico: construir la base desde la foto actual de clientes. Los que se fueron, los castigados, los que cerraron la cuenta... no están. Sesgo de supervivencia: tu historia queda contada solo por los que sobrevivieron, la tasa de malos histórica se subestima, y el modelo aprende un mundo que no es. Detección: reconciliar conteos contra los registros de originación de la época. ¿Cuántas solicitudes hubo de verdad en julio de 2024, y cuántas tiene tu base?

**Fuga cuatro: fuga de identidad.** El mismo cliente repartido entre entrenamiento y validación. Pasa cuando el split es aleatorio por fila y un cliente tiene varias solicitudes. El modelo lo "reconoce" en el hold-out, y la validación deja de ser honesta. La defensa tiene dos capas: regla de cliente único —cada cliente entra una vez, con su primera solicitud— y, si hay que relajarla, partición por cliente, nunca por fila. Si vienes de machine learning: es el group split de toda la vida, solo que aquí no es opcional.

## BLOQUE 4 · Las defensas (10:00–12:30)

¿Cómo se defiende un equipo profesional? Con tres capas.

Capa uno: el screening con alarma. Toda variable con IV sobre 0.5, o AUC univariado sobre 0.9, se audita antes de celebrarse. Las fugas graves se delatan solas: son demasiado buenas.

Capa dos: la autopsia. Para la variable sospechosa: qué significa, cuándo se escribe, cruce contra el target, y reconstrucción de la línea de tiempo de cinco casos concretos. ¿Este valor pudo conocerse el día de la decisión? Si dudas, es no.

Capa tres, la definitiva: tests automáticos en el pipeline. Cada variable declara su ventana y su fuente; un test verifica mecánicamente que el fin de la ventana sea menor o igual a t cero menos el rezago de la fuente. Otro test verifica que los ids de clientes de desarrollo y validación sean conjuntos disjuntos. Otro, que las columnas prohibidas no estén en la matriz. Corren en cada ejecución, porque las fugas se reintroducen con cada refactor. La disciplina temporal no es un ritual de una vez: es un invariante del sistema.

## CIERRE (12:30–13:30)

Recapitulando: la frontera es una fecha. Variables hacia atrás, target hacia adelante, y el ancla ajustada al rezago real de cada fuente. Las fugas vienen en cuatro sabores —ex-post, ventana, población, identidad— y comparten un síntoma: métricas demasiado buenas. En este dominio, la sospecha ante los buenos resultados no es pesimismo: es competencia profesional. Porque una fuga en producción se paga doble: el modelo colapsa, y la confianza del comité colapsa con él. Y de las dos, la segunda tarda más en recuperarse.

La próxima vez: cómo se fabrica el número que resume cuánto separa una variable —el Information Value— y las tres trampas que lo convierten en el número más malinterpretado del riesgo de crédito.

---

**Notas de producción:** los números citados (AUC 0.991, IV 0.39→0.90, IV 11.1) provienen del caso Banco Austral del curso; si el video será público, generalizarlos como "en un caso real de un banco de consumo". Duración estimada leída a ritmo natural: ~13 min.

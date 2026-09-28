# G4 · Guión: "De score a probabilidad: calibración y la tendencia central"

**Formato objetivo:** video/podcast de 10-12 minutos · Cubre: E2 (calibración, tendencia central, PIT vs TTC)

---

## COLD OPEN (0:00–0:45)

Tu modelo de crédito tiene un Gini espectacular: ordena a los clientes de mejor a peor casi sin errores. Y aun así puede estar mintiéndote en lo más importante: el número. Dice "probabilidad de default 4 por ciento" donde la realidad es 7. Hoy: por qué ordenar bien y acertar el nivel son dos propiedades distintas, por qué todo scorecard nace descalibrado, y qué es esa "tendencia central" a la que los bancos anclan sus probabilidades — incluyendo la respuesta a una pregunta que aparece en todo proyecto: "si la economía se deterioró, ¿no es inútil entrenar con datos del año pasado?".

## BLOQUE 1 · Dos propiedades, no una (0:45–3:00)

Un modelo de riesgo hace dos afirmaciones distintas, y conviene separarlas quirúrgicamente.

La primera: "este cliente es más riesgoso que aquel". Eso es **discriminación** — ordenamiento. Se mide con Gini, AUC, KS. Y tiene una propiedad clave: es invariante a transformaciones monótonas. Puedes estirar, comprimir o desplazar la escala del score y el orden no cambia.

La segunda: "en este grupo, caerá el 4 por ciento". Eso es **calibración** — nivel. Se verifica comparando probabilidad predicha contra frecuencia observada, tramo por tramo.

Son independientes. Un modelo puede ordenar perfecto y errar el nivel al doble. ¿Importa? Depende de para qué usas el número. Para priorizar cobranza — puro orden — la calibración da lo mismo. Pero para fijar el punto de corte de aprobación, la cuenta económica se hace en probabilidades: apruebo mientras la pérdida esperada del marginal no supere su margen. Si la probabilidad está corrida, tu cutoff "del 8 por ciento" es en realidad uno del 14, y nadie lo decidió. Y para provisiones y pricing, la probabilidad ES el producto. Ahí calibrar no es opcional: es el trabajo.

## BLOQUE 2 · Por qué el scorecard nace descalibrado (3:00–5:00)

Tres fuentes, casi siempre las tres a la vez.

Uno: la época. El modelo aprendió la tasa de malos de sus cosechas de entrenamiento — digamos, un año relativamente benigno. Va a decidir sobre los próximos años, que pueden ser otra cosa. El nivel que lleva incrustado es una foto del pasado.

Dos: la muestra. Excluiste indeterminados —los casos grises entre bueno y malo—, quizás re-balanceaste clases. La tasa de TU muestra ya no es la de la población.

Tres: la escala. Tras convertir el modelo a puntos de score, el output ni siquiera es una probabilidad. La relación puntos-a-probabilidad hay que construirla explícitamente.

Y aquí la observación empírica que ordena todo el problema, y que responde la pregunta del deterioro económico: cuando el ciclo se mueve, **el ordenamiento sobrevive mucho mejor que el nivel**. En una recesión, todos los perfiles caen más — pero los relativamente peores siguen siendo relativamente peores. El Gini se despeina; el nivel se dispara. Conclusión operativa: el deterioro no invalida el modelo entrenado con datos previos. Invalida su nivel. Y el nivel se corrige por fuera, en una etapa aparte: la calibración. Esa separación —ordenar con el modelo, nivelar con la calibración— es de las ideas más elegantes de esta disciplina.

## BLOQUE 3 · La maquinaria (5:00–8:00)

¿Cómo se calibra? La receta estándar tiene tres pasos.

Paso uno: agrupar el score en tramos y medir, en una muestra honesta — que el ajuste nunca vio —, la tasa de malos real de cada tramo. Esa tabla es tu calibración empírica cruda.

Paso dos: suavizarla con una regresión logística del desenlace contra el score. Dos parámetros: intercepto y pendiente. Si la pendiente sale cerca de uno, la cirugía mínima es tocar solo el intercepto: todos los scores se desplazan en paralelo en la escala log-odds, el nivel se corrige y el ordenamiento queda intacto, exactamente como queríamos.

Paso tres, el que le da nombre al episodio: decidir a qué nivel promedio anclar. Y aquí hay tres candidatos.

¿La tasa del período de desarrollo? No: es justo lo que queremos corregir. ¿La tasa más reciente? Mejor, pero es la foto de UN punto del ciclo: si calibras en el peor momento, sobre-estimarás durante toda la recuperación. La respuesta madura es la tercera: la **tendencia central** — el promedio de la tasa de malos a través de un ciclo económico completo, idealmente cinco o más años que incluyan al menos una recesión. Un ancla estable, prudente, que no persigue al ciclo.

La mecánica del ajuste es una línea de álgebra: el desplazamiento del intercepto es la diferencia entre el log-odds de la tendencia central y el log-odds de la tasa de tu muestra. Una suma en la escala correcta, y toda la curva de probabilidades queda anclada al largo plazo.

## BLOQUE 4 · PIT, TTC y los tres números del mismo cliente (8:00–10:00)

Esto abre la distinción más citada del riesgo regulatorio: PIT contra TTC.

Una probabilidad **point-in-time** refleja el momento del ciclo: en recesión sube, en expansión baja. Es la mejor predicción condicional de lo que viene. La contabilidad de provisiones moderna —IFRS 9— pide exactamente eso, incluso condicionado a escenarios macroeconómicos hacia adelante.

Una probabilidad **through-the-cycle** refleja el promedio del ciclo: estable, suave, anclada a la tendencia central. La gestión de capital la prefiere, porque el capital no puede andar oscilando con cada trimestre.

Y aquí la escena que confunde a todo recién llegado: el mismo cliente, en el mismo banco, tiene legítimamente varias probabilidades de default. La gerencial del scorecard de admisión. La PIT con escenarios de IFRS 9. La TTC de capital. Ninguna está "mala": responden preguntas distintas, con anclas distintas. Y lo notable es que un mismo motor de ordenamiento —un mismo scorecard— puede alimentar a todas, cambiando solo la calibración. El modelo ordena; cada marco pone su nivel.

## CIERRE (10:00–11:00)

Para llevarse: discriminar y calibrar son afirmaciones distintas, y se auditan por separado. El scorecard nace descalibrado por época, muestra y escala — es lo esperado, no un defecto. La corrección estándar es quirúrgica: intercepto en log-odds, anclado a la tendencia central de largo plazo, verificado en muestras honestas con la curva de calibración. Y la vigilancia continúa en producción: la calibración es lo primero que el ciclo degrada — mucho antes que el Gini —, así que el tablero de monitoreo necesita una luz propia para "predicho contra observado, cosecha a cosecha".

La pregunta del gerente — "¿el deterioro no invalida el modelo?" — ya tiene respuesta de dos líneas: el ordenamiento sobrevive, el nivel se recalibra. Y saber exactamente cuál de las dos cosas se rompió, con evidencia, es la diferencia entre rediseñar seis meses... y ajustar un intercepto en una tarde.

---

**Notas de producción:** episodio más conceptual que los anteriores; funciona especialmente bien como podcast puro. Si es video: la única figura imprescindible es la curva de calibración (predicho vs observado con la diagonal). Duración estimada: ~10-11 min.

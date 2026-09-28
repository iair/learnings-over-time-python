# G5 · Guión: "La economía del error: por qué un banco aprueba gente que probablemente no pagará"

**Formato objetivo:** video/podcast de 10-12 minutos · Cubre: M1 (economía del error, PD de indiferencia, cutoff)

---

## COLD OPEN (0:00–0:45)

Pregunta de entrevista: un solicitante tiene 20 por ciento de probabilidad de no pagar. Uno de cada cinco como él deja al banco con la pérdida. ¿Lo apruebas? La intuición grita que no. La respuesta correcta, en la mayoría de los créditos de consumo, es que sí — y no por generosidad, sino por aritmética. Hoy: los dos errores de todo modelo de crédito, por qué cuestan distinto y se ven distinto, la fórmula de una línea que define dónde cortar, y por qué "el modelo rechaza mucho" no es una crítica al modelo.

## BLOQUE 1 · Los dos errores y su asimetría (0:45–4:00)

Todo clasificador de crédito comete dos errores, y no son simétricos ni en magnitud ni en visibilidad.

**Error A: aprobar a un malo.** El costo es la pérdida del crédito: la exposición al momento de la caída, multiplicada por la severidad — la fracción que no se recupera ni con cobranza ni con garantías. Para un crédito de consumo de dos millones con severidad del 70 por ciento, cada malo aprobado cuesta del orden de un millón cuatrocientos. Y ese costo es **visible**: aparece en la cartera vencida, en las provisiones, en el estado de resultados. Tiene nombre, fecha y un informe que lo recuerda.

**Error B: rechazar a un bueno.** El costo es el margen que ese cliente habría pagado durante la vida del crédito — para el mismo crédito, quizás 400 o 500 mil — más el valor del cliente que se fue a la competencia. Y ese costo es **invisible**: no existe ningún estado financiero donde aparezcan los buenos rechazados. Nadie provisiona por la venta que no ocurrió.

Fíjate en la doble asimetría. De magnitud: el malo aprobado cuesta, en este ejemplo, tres veces lo que deja el bueno. Y de visibilidad: uno se audita, el otro se evapora. Esa segunda asimetría tiene una consecuencia organizacional que hay que conocer: el área de riesgo es castigada por los malos aprobados y jamás por los buenos rechazados, así que su incentivo natural empuja al conservadurismo. El área comercial empuja exactamente al revés. El scorecard no zanja esa pelea: la vuelve explícita y cuantificable. Esa es su verdadera función política dentro del banco.

## BLOQUE 2 · La fórmula de una línea (4:00–7:00)

Pongamos números a la decisión del margen — el solicitante que está justo en el borde.

Si lo apruebo y paga —probabilidad uno menos p— gano el margen. Si lo apruebo y cae —probabilidad p— pierdo severidad por exposición. El banco está indiferente cuando ambas esperanzas se igualan:

p por la pérdida, igual a, uno menos p por el margen.

Despejando p, la **probabilidad de indiferencia**: margen dividido por margen más pérdida. Una línea.

Con los números del ejemplo: margen 450 mil, pérdida 1 millón 400. p-estrella igual a 450 sobre 1.850: 24 por ciento. Todo solicitante con probabilidad estimada bajo 24 genera valor esperado positivo. Ahí está la respuesta a la pregunta del inicio: el del 20 por ciento se aprueba, porque el margen de los cuatro que pagan cubre con holgura la pérdida del que no. El negocio de crédito masivo es exactamente eso: la mayoría que paga subsidia a la minoría que no, y la frontera rentable está mucho más adentro de lo que la intuición tolera.

Y mira lo que la fórmula te regala: estática comparativa gratis. ¿Mejora la cobranza y la severidad baja de 70 a 60 por ciento? La pérdida baja, p-estrella sube, puedes aprobar más profundo. ¿Sube tu costo de fondeo y el margen se comprime? p-estrella baja, el corte se endurece solo. El punto de corte deja de ser una opinión y pasa a ser una función de tres números del negocio.

## BLOQUE 3 · Del break-even a la política (7:00–9:30)

Ahora, tres refinamientos que separan la versión de pizarra de la versión de comité.

Uno: **el óptimo económico no siempre manda.** El banco puede fijar el corte por tasa de aprobación objetivo —está en modo crecimiento—, por pérdida máxima tolerada —el apetito de riesgo que declaró al directorio—, o por retorno sobre el capital que cada tramo consume. El modelo no elige el punto: entrega la curva completa — para cada corte posible, cuánta aprobación, cuánta pérdida, cuánta utilidad — y el negocio elige dónde pararse, con los trade-offs a la vista.

Dos: **el corte único es la versión de juguete.** La práctica real usa zonas: aprobación automática arriba, rechazo automático abajo, y una zona gris al medio que va a revisión manual. Y el score no solo decide la entrada: decide cupo y precio. Al más riesgoso, menos línea y más tasa. La decisión binaria se convierte en una superficie de decisión.

Tres, y este es el que responde la crítica clásica: **"el modelo rechaza mucho" confunde dos cosas.** El modelo ORDENA — pone a los solicitantes en fila del más seguro al más riesgoso. Dónde cortar la fila es una decisión de negocio, una perilla. Si el orden es bueno, aprobar más o menos es cuestión de mover el corte, con su costo calculable. La crítica legítima al modelo es "ordena mal" — y eso se mide. Confundir el instrumento con la política es el error de encuadre más frecuente en las discusiones de crédito. Su gemelo: pedir "bajar la definición de default a 60 días para vender más" — cambiar la definición del target no aprueba a nadie; el corte sí.

## CIERRE (9:30–11:00)

Un último encuadre, el del tiempo. En la mayoría de los negocios digitales, un mal modelo se descubre en semanas. En crédito, el error de hoy madura en la cartera durante 12 o 18 meses antes de volverse visible — y para entonces hay miles de créditos cursados bajo la política equivocada. Esa distancia entre decisión y consecuencia es la razón profunda por la que en este oficio el diseño pesa más que el algoritmo, y por la que cada supuesto — la definición de malo, la población, el corte — se escribe, se sustenta con evidencia y se le pone fecha de verificación.

Para llevarse: dos errores, asimétricos en costo y en visibilidad. Una fórmula de una línea — margen sobre margen más pérdida — que convierte el corte en aritmética del negocio. Y la separación mental clave: el modelo ordena, la política corta. Quien domina esa separación puede defender un scorecard ante cualquier comité. Quien no, termina discutiendo de estadística cuando la pregunta era de plata.

---

**Notas de producción:** los montos ($2M, LGD 70%, margen $450k) son ilustrativos y coherentes con la calculadora `M1_calculadora_tradeoff.xlsx` de la serie; puedes regrabar los números con los de tu propio producto usando la planilla. Duración estimada: ~10-11 min.

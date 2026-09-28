# G3 · Guión: "Las tres trampas del Information Value"

**Formato objetivo:** video/podcast de 10-12 minutos · Cubre: M7 (binning, WoE, IV y sus trampas)
**Prerrequisito sugerido para la audiencia:** saber qué es WoE/IV a nivel de fórmula (los videos de la videoteca lo cubren); este episodio va de cómo se malinterpreta.

---

## COLD OPEN (0:00–0:45)

El Information Value es el número más usado del riesgo de crédito para responder una pregunta simple: ¿cuánto separa esta variable a los buenos de los malos? Un solo número por variable. Una tabla de umbrales famosa para leerlo. Y tres maneras clásicas de engañarse con él. Hoy: por qué una variable cien por ciento aleatoria puede marcar IV "fuerte", por qué un IV de 1.2 puede desplomarse a la mitad en validación sin que haya trampa alguna, y por qué el IV más alto de tu proyecto es probablemente el más falso.

## BLOQUE 1 · El instrumento, en dos minutos (0:45–3:00)

Repaso mínimo. Discretizas la variable en tramos —bins—. Para cada bin comparas dos porcentajes: qué fracción de todos los BUENOS cayó ahí, y qué fracción de todos los MALOS. El logaritmo de ese cociente es el Weight of Evidence del bin: cero si el bin luce como la población, positivo si sobre-representa buenos, negativo si sobre-representa malos — al menos en la convención buenos-sobre-malos; hay libros y librerías que usan la inversa, así que antes de comparar números entre fuentes, verifica el signo con un bin obvio: el de mora alta tiene que salir "malo".

El IV suma, sobre los bins, la diferencia de porcentajes por el WoE. Propiedades: siempre positivo; cero solo si buenos y malos se distribuyen idéntico; y crece sin techo cuando algún bin se vuelve casi puro. Para los que gustan del fundamento: es la divergencia de Kullback-Leibler simetrizada entre la distribución de buenos y la de malos. Guarda ese dato, porque las tres trampas salen de ahí.

Y la tabla de lectura estándar, la de Siddiqi: bajo 0.02, sin poder. Hasta 0.1, débil. Hasta 0.3, medio. Hasta 0.5, fuerte. Y sobre 0.5... ojo: no dice "excelente". Dice: auditar.

## BLOQUE 2 · Trampa uno: el IV premia el número de bins (3:00–5:30)

Experimento real de clase: se genera una variable completamente aleatoria — ruido puro, sin relación alguna con el default — pero con 64 niveles distintos. Se calcula su IV. Resultado: 0.32. "Fuerte", según la tabla. Ruido puro, calificado de fuerte.

¿El mecanismo? Cada nivel tiene pocos casos. Con pocos casos, la proporción de malos fluctúa por puro azar. Y aquí está la maldad matemática: cada fluctuación —para arriba o para abajo— suma IV positivo, porque el IV es una suma de términos no negativos. Más particiones, más fluctuaciones, más IV fantasma. El sesgo crece con el cociente entre número de bins y número de malos.

Defensas, tres. Masa mínima por bin — la regla práctica: ningún bin con menos del 5 por ciento de la muestra —, que limita cuántos bins puedes tener. Comparar variables a igual número de bins, porque comparar el IV de una variable de 5 bins contra una de 60 es comparar peras con manzanas infladas. Y la definitiva: contrastar en una muestra que el binning nunca vio. El IV fantasma es memoria del ruido de la muestra de desarrollo; en hold-out se desploma. El 0.32 de la variable aleatoria cae a prácticamente cero. Ese contraste desarrollo-contra-holdout es el test de sobreajuste más barato que existe.

## BLOQUE 3 · Trampa dos: el IV cuelga de los malos (5:30–8:00)

Segundo experimento, este duele más porque no hay ruido deliberado: una variable legítima — uso máximo de la línea en 12 meses — marca IV 1.21 en desarrollo. Espectacular. Se recalcula en hold-out: 0.55. Menos de la mitad. ¿Fuga? No. ¿Error? Tampoco. Varianza.

El asunto es que el WoE de cada bin depende del conteo de la clase escasa: los malos. En el caso de clase había 165 malos en toda la muestra de desarrollo. Repártelos en cinco bins: 33 malos por bin, en promedio. El peso de evidencia de cada bin cuelga de 33 observaciones. Mueve cinco malos de un bin a otro —nada, azar muestral puro— y el IV se sacude en décimas. Un IV calculado con pocos malos no es un número: es una nube, y reportar solo el centro de la nube es mentir por omisión.

Defensas. Mínimo de MALOS por bin, no solo de casos totales — un bin con 800 casos y 3 malos es una ruleta. Suavizado de conteos, para que un bin sin malos no dispare logaritmos infinitos. Reportar el IV con su intervalo de confianza vía bootstrap: decir "1.2" es una cosa; decir "1.2, con intervalo entre 0.7 y 1.6" cambia la conversación completa. Y sobre todo: decidir siempre con la pareja desarrollo-holdout. Una variable 0.40 y 0.38 vale más que una 1.2 y 0.55. Estabilidad le gana a espectacularidad.

## BLOQUE 4 · Trampa tres: el IV altísimo es una alarma (8:00–10:00)

Tercera escena. En el screening aparece una variable con IV 3. Otra con 11. Aplausos en la sala equivocada.

Recuerda la propiedad matemática: el IV explota cuando un bin se vuelve casi puro — casi todos malos o casi todos buenos. ¿Y qué tipo de variable produce bins casi puros? Una que ya sabe el desenlace. Una columna de estado posterior. Una ventana de cálculo que cruzó la fecha de la solicitud. Una variable que registra la reacción del banco al deterioro — el recorte de cupo después de que el cliente empezó a caer, IV once, que no predice el default: lo fotografía.

Por eso el umbral de 0.5 dice "auditar" y no "celebrar". En admisión de consumo, la señal honesta vive en IVs de 0.1 a 0.5. Sobre eso, la probabilidad de que sea filtración supera por mucho a la de que sea descubrimiento. El protocolo profesional: la variable sospechosa no entra al modelo NI se descarta hasta tener una explicación de negocio escrita. Porque también existen los IVs altos legítimos —en modelos de comportamiento, con datos transaccionales ricos, son comunes— y el descarte automático tira señal real. La decisión es humana, y documentada.

## CIERRE (10:00–11:30)

El cuadro completo del screening profesional, entonces. El IV se calcula en desarrollo y se contrasta en hold-out, siempre en pareja. Los bins respetan masa mínima y mínimo de malos. Los umbrales de la tabla son señales de dónde mirar, no ley. Y la salida del screening no es un ranking: son tres listas. Descartadas: las que no separan o colapsan fuera de muestra. Candidatas: rango sano, estables. Y auditoría: las sospechosamente fuertes, congeladas hasta tener explicación.

Y la advertencia final, que es la más importante: el IV mira UNA variable a la vez. Sirve para descartar lo inútil y priorizar auditorías. No sirve para elegir el modelo: dos variables con IV enorme pero correlacionadas al 95 por ciento aportan una sola vez, y una variable modesta pero ortogonal al resto puede valer oro. Screening no es selección. La selección es un problema multivariado, y esa es otra historia.

---

**Notas de producción:** cifras del caso Banco Austral (IV 0.32 aleatoria, 1.21→0.55, 165 malos, IV 11 del Δcupo); para difusión pública, "en un caso real de banca de consumo". Duración estimada: ~11 min.

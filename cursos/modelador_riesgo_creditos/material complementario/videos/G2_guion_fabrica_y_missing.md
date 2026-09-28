# G2 · Guión: "La fábrica de variables y el missing que informa"

**Formato objetivo:** video/podcast de 12-14 minutos · Cubre: M5 (fábrica) + M6 (calidad de datos) + cierre de E4 (bureau)
**Tono:** narrador único, dirigido a un data scientist que llega a riesgo.

---

## COLD OPEN (0:00–0:40)

Hay dos maneras de crear variables para un modelo de crédito. La del principiante: sentarse a inventar. "Se me ocurre que el uso de la tarjeta podría predecir". La del profesional: no inventar ninguna variable... y construir una fábrica que las produzca todas. Hoy: cómo se diseña esa fábrica, por qué un valor faltante puede ser la variable más predictiva de tu dataset, y por qué imputar por la media —el reflejo de todo data scientist— puede ser la forma más educada de destruir información.

## BLOQUE 1 · La fábrica (0:40–4:30)

La idea central es una convención de tres ejes.

Eje uno: **familias**. Qué se mide. Mora propia. Uso de la línea. Deuda y saldos. Pagos. Ahorro. Comportamiento en el resto del sistema —el bureau—. Consultas de crédito. Siete, ocho familias: el mapa de lo que un banco sabe de un cliente.

Eje dos: **ventanas**. Desde cuándo. Tres meses: la foto reciente, reactiva pero ruidosa. Doce meses: el patrón estructural, estable pero lento. Seis: el compromiso.

Eje tres: **agregadores**. Cómo se resume la serie mensual en un número. Y aquí está la finura, porque cada agregador responde una pregunta de negocio distinta. El promedio: ¿cómo se comporta habitualmente? El máximo: ¿cuál fue su peor momento? Y ojo: uso promedio 40 por ciento con máximo 95 cuenta una historia muy distinta que 40 con 45 — por eso promedio y máximo de la misma serie no son redundantes. La suma o el conteo: ¿cuánta actividad acumuló? El delta: ¿está mejorando o empeorando? Y la recencia: ¿hace cuánto que no pasa algo? — porque una mora de hace dos meses y una de hace once no pesan igual.

Cruzas los tres ejes y la fábrica produce, sistemáticamente, del orden de cien variables candidatas. Con tres ventajas que ningún proceso artesanal tiene.

Cobertura: el producto cartesiano no se olvida de combinaciones. Auditoría: el que revisa entiende cien variables leyendo UNA convención. Y reutilización: el próximo modelo hereda la fábrica entera.

Y hay una cuarta, para los que venimos de ingeniería de datos: la fábrica es un pipeline declarativo. Una especificación —datos— separada de un motor —código—. Cada variable declara su familia, su agregador, su ventana y el rezago de su fuente. De esa declaración se genera automáticamente el nombre —uso_linea_promedio_6m, imposible que el nombre mienta sobre la ventana porque ambos salen de la misma tupla—, se genera el diccionario para el equipo de producción, y se valida mecánicamente que ninguna ventana cruce la fecha de la solicitud. La regla anti-fugas del episodio anterior, convertida en test automático.

Falta un ingrediente que la fábrica no produce: los **ratios de negocio**. Utilización: saldo sobre cupo. Carga financiera: deuda sobre ingreso. Cuota sobre ingreso. Cada ratio es una hipótesis de negocio escrita en fórmula, y ningún algoritmo los inventa por ti. Ahí vive el criterio del modelador... y las decisiones escondidas. Porque "deuda sobre ingreso" obliga a decidir: ¿qué ingreso? ¿La renta que el cliente declaró en un formulario hace tres años, sin verificar y que además falta en el 13 por ciento de los casos? ¿O los abonos que el banco VE entrar a la cuenta mes a mes? La decisión profesional estándar: el ratio usa lo observable —los abonos—, y la renta declarada entra como variable aparte, con su ausencia incluida. Que la ausencia entre al modelo no es un descuido. Es el punto del siguiente bloque.

## BLOQUE 2 · El missing que informa (4:30–9:30)

En machine learning general, el valor faltante es un estorbo: se imputa y se sigue. En crédito, el valor faltante es un personaje con motivaciones. Hay tres tipos, y confundirlos cuesta caro.

**Tipo uno: missing estructural.** El dato no falta: no existe. La variable "cambio de deuda en 12 meses" necesita saber la deuda de hace 12 meses. Un cliente con 8 meses de antigüedad no la tiene, por aritmética pura. ¿Imputarle un valor? Sería inventarle una tendencia a alguien cuya característica real es ser nuevo. Y ser nuevo ES información: los clientes cortos tienen su propio perfil de riesgo. Tratamiento: una categoría propia, "sin historia suficiente". Nunca imputación.

**Tipo dos: el importante. MNAR** — missing not at random. La ausencia depende de quién es el cliente. El caso estrella: la renta declarada. En un caso real de banca de consumo, el 7 por ciento de los empleados dependientes no declara renta... contra el 36 por ciento de los trabajadores independientes. Eso no es azar: es informalidad de ingresos. Y la informalidad correlaciona con riesgo. Cuando agrupas a todos los que no declararon y mides su tasa de mora, es peor que el promedio. La ausencia ES la señal.

Ahora mira lo que hace la imputación por la media. Toma a ese grupo — más riesgoso que el promedio — y lo disfraza del cliente promedio. Doble daño: borra la señal, porque los que no declararon quedan camuflados en el centro de la distribución, y contamina la distribución, porque le plantas un pico artificial de 13 por ciento de masa en la media, deformando los cuantiles de todos los demás. La imputación no fue neutral: fue una decisión de modelamiento, tomada sin darse cuenta, en la dirección equivocada.

El tratamiento correcto es de una simpleza hermosa: el missing es su propia categoría, con su propio peso en el modelo. El binning —la discretización estándar de los scorecards— lo hace natural: un bin llamado MISSING, con su tasa de malos y su peso de evidencia propios. La ausencia entra al modelo como lo que es: información.

**Tipo tres: MCAR** — completamente al azar. El batch de febrero que no cargó. Es el único caso donde imputar es defendible... y es el caso menos frecuente. De ahí la regla de bolsillo: la imputación por la media es la respuesta correcta a la pregunta menos frecuente. Antes de imputar, diagnostica: ¿la ausencia se concentra en un segmento? ¿la tasa de malos de los ausentes difiere? Si cualquiera da sí, no imputes: dale su bin.

Y un párrafo para los outliers, que siguen la misma lógica de diagnóstico antes que tratamiento. Tres especies. Los centinelas: el 999999, el menos uno — códigos de "sin dato" de algún sistema; se detectan por masa puntual, un pico anormal de casos en un valor exacto, y se convierten a missing. Los imposibles de negocio: utilización de 47 veces el cupo — un error a corregir; pero cuidado, utilización 1.2 puede ser un sobregiro pactado perfectamente legítimo, y distinguirlos exige conocer el negocio, no la estadística. Y los extremos reales: el cliente con deuda 20 veces la mediana. Esos son señal, y suelen ser señal de riesgo. La buena noticia: en el flujo con binning, el extremo cae en el bin de los extremos y no arrastra ningún coeficiente. El binning es un amortiguador de outliers gratis.

## BLOQUE 3 · El mundo exterior: el bureau (9:30–12:00)

Un cierre sobre la familia de variables más valiosa y más traicionera: las del bureau de crédito, el comportamiento del cliente en todo el sistema financiero.

Valiosa, porque resuelve el punto ciego del banco: el cliente nuevo, del que internamente no sabes nada, puede tener años de historia en otra parte. Y porque contiene la señal que tus datos internos no pueden ver por definición: el credit hunger — el racimo de consultas de crédito recientes en varias instituciones, un clásico predictor de estrés financiero.

Traicionera, por dos razones. Uno: el rezago. El archivo del bureau llega uno o dos meses tarde, y las bases históricas suelen tenerlo puesto en el mes al que se refiere, no en el que llegó. Si desarrollas sin correr el ancla, entrenas con frescura que producción no tendrá. Dos: el costo. Cada consulta se paga. Y eso convierte la selección de variables en una cuenta de valor neto: ¿cuántos puntos de discriminación agrega este bloque de datos, y cuánto cuesta al año? Esa cuenta —Gini extra traducido a pérdida evitada, contra la factura del proveedor— la produce el modelador, y es de las que más impresionan a un comité, porque habla el idioma del negocio.

## CIERRE (12:00–13:00)

Tres ideas para llevarse. Uno: las variables no se inventan, se fabrican — con una convención de familias, ventanas y agregadores que hace el trabajo exhaustivo, auditable y reutilizable. Dos: el valor faltante es información hasta que se demuestre lo contrario; el diagnóstico va antes que el tratamiento, y el bin propio le gana a la imputación en casi todos los casos que importan. Tres: cada ratio y cada fuente externa esconden decisiones de negocio — el denominador, el rezago, el costo — y el trabajo del modelador es sacarlas de la sombra y documentarlas.

La próxima: el número que resume todo esto — el Information Value — y por qué es a la vez la herramienta más usada y la más malinterpretada del screening de variables.

---

**Notas de producción:** las cifras (7% vs 36%, 13% missing de renta) son del caso Banco Austral del curso; para difusión pública, presentarlas como "en un caso real de banca de consumo". Duración estimada: ~12-13 min.

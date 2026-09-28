# E4 · Variables de bureau: el mundo exterior, con rezago y con precio

**Serie: Modelador de Riesgo en Profundidad** · Fase 2, extensión 4 de 8
Origen: reserva C2-38 ("¿por qué no usamos más variables de bureau si discriminan tan bien?")

---

## 1. Qué es el bureau y por qué es oro para admisión

El bureau (en Chile: el sistema de deuda consolidada de la CMF, más burós privados tipo Equifax/DICOM; en otros mercados: Experian, TransUnion, etc.) agrega el comportamiento del cliente **en todo el sistema financiero**: deuda total y por tipo, morosidades vigentes e históricas, número de acreedores, consultas recientes. Para admisión es especialmente valioso porque resuelve el punto ciego estructural del banco: el cliente nuevo, del que internamente no se sabe nada, puede tener años de historia... en otra parte. Las familias típicas:

| Familia bureau | Ejemplos | Señal |
|---|---|---|
| Endeudamiento sistema | deuda total, n° acreedores, deuda/ingreso sistema | Capacidad comprometida real (no solo la propia) |
| Morosidad sistema | mora vigente, peor mora 12m, recencia de mora externa | El comportamiento de pago "verdadero" |
| Búsqueda de crédito | n° consultas 3/6m, recencia de consulta | El *credit hunger*: el racimo de consultas recientes es un clásico predictor de estrés |
| Historia/perfil | antigüedad en el sistema, mix de productos | Thin file vs perfil establecido |

La fábrica de M5 las procesa igual que a las internas (familias × ventanas × agregadores), con dos diferencias operativas que son el corazón de esta extensión: **rezago** y **costo**.

## 2. El rezago: la trampa del backfill (reprise de M2, ahora en serio)

El archivo de bureau se genera con la foto de fin de mes y se distribuye después: cuando decides una solicitud de marzo, el último archivo disponible suele ser el de enero (lag 2). El error de desarrollo clásico: la base histórica tiene el archivo de enero **etiquetado en enero** (backfilled), y el modelador ancla la variable a t₀−1 como todas las internas. Resultado: el modelo se entrena con información un mes más fresca de la que producción tendrá — fuga de ventana silenciosa, degradación estructural al desplegar.

Reglas prácticas:

1. **Ancla por fuente en la especificación de la fábrica** (`lag_fuente=2` en la tupla): la ventana bureau de "6 meses" es [t₀−7, t₀−2]. El validador de M5 la verifica mecánicamente.
2. **Verificar el lag real, no el nominal:** medir en producción la distribución de la antigüedad del archivo al momento de decidir (a veces es 1 mes, a veces 3 — feriados, atrasos del proveedor). Desarrollar con el lag del percentil conservador.
3. **Consulta en línea vs archivo batch:** algunos burós ofrecen consulta online con datos más frescos y estructura distinta. Si producción consultará online, el desarrollo debe replicar ESE contenido y ESE rezago — no el del archivo mensual. La discrepancia archivo/online es una fuente clásica de "el modelo rinde menos de lo validado".

## 3. El costo: cada consulta se paga

A diferencia de las variables internas (gratis en el margen), la consulta al bureau cuesta dinero por solicitud, y algunos atributos premium cuestan más. Esto convierte la selección de variables en un problema de **valor neto**, no solo de IV:

1. **Trade-off Gini/costo explícito:** medir el Gini del modelo con y sin cada bloque de bureau. Si el bloque premium agrega 1,5 puntos de Gini, la cuenta es: puntos de Gini → mejora en pérdida esperada a cutoff constante (la calculadora de M1 da el orden de magnitud) vs costo anual de consultas. Es una decisión de negocio con números, y el modelador es quien los produce.
2. **Arquitecturas de dos etapas:** una respuesta común al costo — un filtro barato con variables internas + demografía decide los casos claros, y solo la zona gris paga la consulta completa al bureau. (Cuidado: la política de a quién consultar cambia la población sobre la que las variables bureau se observan — sesgo de selección de segundo orden que hay que documentar.)
3. **Dependencia del proveedor:** un scorecard cargado de variables de un buró queda expuesto a cambios de formato, de precio o de disponibilidad del proveedor. La resiliencia (¿qué pasa con el score si el bureau no responde en línea?) es parte del diseño: definir el tratamiento "sin bureau" (bin propio, política conservadora) desde el desarrollo.

## 4. Detalles de calidad específicos del bureau

- **Missing con tres mecanismos distintos:** (a) thin file real (el cliente no existe en el sistema — estructural e informativo: su propio bin, típicamente de riesgo alto en consumo); (b) fallo de match (RUT mal escrito, homónimos — operacional, MCAR-ish); (c) el proveedor no reporta ese atributo para ese tipo de cliente (estructural). El diagnóstico de M6 aplica: la regla que genera el missing es el mecanismo.
- **Definiciones que cambian bajo tus pies:** los burós redefinen atributos (qué cuenta como "consulta", qué productos entran en "deuda"). Cada cambio de definición es un quiebre estructural en la serie de la variable → monitorear PSI por variable bureau con especial atención y guardar los diccionarios de cada versión.
- **Normativa de uso:** qué se puede consultar, con qué consentimiento, y qué se puede usar para decidir está regulado (protección de datos, en Chile la ley de deuda consolidada y la reciente ley de datos personales). El "se puede técnicamente" no implica "se puede legalmente" — coordinar con legal es parte del proyecto, no un trámite posterior.

## 5. Cuánto aporta (calibración de expectativas)

Regla de pulgar de industria para admisión de consumo: un modelo solo-interno para clientes con historia rinde razonable; agregar bureau suma típicamente entre 3 y 10 puntos de Gini, con el aporte concentrado en (a) clientes nuevos/thin-internal (donde el bureau es casi toda la información) y (b) el *credit hunger* (consultas recientes), que la información interna no ve por definición. Corolario de diseño: si el mix de solicitantes carga hacia clientes nuevos, el bureau pasa de "mejora" a "columna vertebral" — y viceversa.

## 6. Para profundizar

- Anderson, *The Credit Scoring Toolkit* — el tratamiento más completo del ecosistema de burós, atributos y su economía.
- Siddiqi, secciones de fuentes de datos y de implementación (rezagos y consistencia desarrollo/producción).
- Documentación pública de la CMF sobre el sistema de obligaciones (para el caso chileno) y los data dictionaries de los burós del mercado donde trabajes: leer el diccionario del proveedor es la única forma de saber qué significa cada atributo.

## 7. Preguntas de autoevaluación

1. Reconstruye la fuga del backfill: qué contiene la base histórica, qué contendrá producción, y cómo se ve la degradación en el monitoreo.
2. El bloque premium del buró agrega 2 puntos de Gini y cuesta $X por consulta con 100.000 solicitudes/año. Arma la cuenta de valor neto con la calculadora de M1 (¿qué supuestos necesitas?).
3. Diseña el tratamiento "sin respuesta del bureau" para producción en línea: variable por variable, ¿bin propio, valor conservador o rechazo a revisión manual?
4. ¿Por qué el racimo de consultas recientes predice estrés? ¿Qué riesgo de círculo vicioso tiene usarla (pista: tu propia consulta también cuenta)?
5. El buró anuncia que desde julio "consulta" incluirá también las consultas de pre-aprobados. ¿Qué le pasa a tu variable, a su PSI y a tu scorecard?

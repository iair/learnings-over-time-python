# ESPECIFICACIÓN — Serie 2 «Del embudo al gobierno» (M8–M23 + integrador C2)

Lee este archivo COMPLETO antes de escribir nada. Es el contrato de calidad de la serie.

## 1. Para quién y para qué

- Lector: **Iair**, líder de data science en una fintech chilena (crédito de consumo, incluido
  financiamiento de motos), con **10 años de experiencia en data science** y fuerte en ingeniería de
  pipelines. Cursa «Modelador de Riesgo de Crédito» (Academia Bayes + Nikodym Advisory, prof. Camilo
  González). Ya aprobó/entregó los labs; su objetivo NO es aprobar: es convertirse en **modelador de
  riesgo experto**, entendiendo en profundidad cada concepto, sus derivaciones, sus variantes de
  industria y cuándo fallan.
- No le expliques qué es un DataFrame, un p-valor básico o una regresión. Sí explícale TODO lo
  específico de riesgo de crédito y la matemática que el curso deja fuera («esto no es un curso de
  econometría: investíguenlo»). Rigor cuantitativo, supuestos explícitos, rangos de sensibilidad,
  caveats honestos. Directo y preciso, nada de relleno motivacional.
- Existe una Serie 1 (NO repetir; enlazar cuando corresponda): M0 mapa y glosario · M1 economía del
  error de crédito · M2 arquitectura temporal (t₀) · M3 definición de default (cadenas de Markov,
  curvas de maduración) · M4 esquema muestral y fuga · M5 fábrica de variables como pipeline
  declarativo · M6 calidad de datos, missing y outliers · M7 binning/WoE/IV con derivación formal ·
  E1 reject inference · E2 calibración y tendencia central / PIT vs TTC · E3 bootstrap e intervalos
  para carteras chicas · E4 variables de bureau y trampas · E5 scorecard vs ML y feature engineering
  automático · E6 regulación (Basilea II/III, IFRS 9, CMF) · E7 eventos extraordinarios y stress
  testing · E8 bibliografía y plan de lectura 90 días.
  Cita así: «(ver Serie 1 · M7)». Si tu módulo toca algo ya cubierto, resume en ≤ 1 párrafo y
  profundiza SOLO en lo nuevo.

## 2. Material del curso (fuente de verdad sobre qué se enseñó)

En `/home/claude/serie2/_spec/curso/`:
- `clase3_modelo_scorecard.txt` — PSI, IV, correlación WoE, VIF, stepwise, logística sobre WoE, Gini
  por muestra, estabilidad de coeficientes en HO, scaling PDO, scorecard, rango de puntos, aportes,
  reason codes, optbinning (3 experimentos), por qué la banca sigue con logística.
- `clase4_calibracion_estrategia_v1.txt` y `clase4_calibracion_estrategia_v21.txt` (versión mejorada:
  granularidad de binning `min_event_rate_diff`, missing −9/−99, familias de variables, cobertura,
  calibración **PIT** con δ exacto 0,177 vs aprox 0,143, muestra de calibración reciente y madura).
- `clase5_validacion_monitoreo.txt` — PIT vs TTC (TC 5,42%, δ 0,1115), master scale, EL, estrategia,
  swap-set; AUC/Gini/KS, bootstrap B=1000, deciles/lift, PSI del score sobre 8 bandas, CSI,
  binomial por banda, Hosmer-Lemeshow con p simulado, tablero con semáforos, diagnóstico por patrón.
- `clase6_implementacion_gobierno.txt` — «en producción NO existe DEV», artefacto congelado, bug del
  binner re-ajustado (587 de 8.585 = 6,8% decisiones cambian), audit trail, cadena de hashes, sello
  externo, expediente de 9 piezas, RACI, gatillos, model card, nikodym.
- Notebooks `.ipynb` (Banco Austral y labs de Financiera Andes) con el código de clase.
Lee los que correspondan a tu módulo. Usa SUS números (Banco Austral) como anclas en los documentos:
el lector los reconoce. Convenciones del curso que DEBES respetar:
- t₀; target **1 = malo** (90+ DPD a 12 meses); indeterminados 30–89 fuera.
- **WoE = ln(%buenos / %malos)** → WoE alto = bin bueno → coeficientes **negativos**.
- Muestras DEV / HO / OOT / TTD. Todo lo que se ajusta (bins, WoE, β, δ, cortes) se ajusta en DEV (o
  en la muestra de calibración declarada) y se APLICA al resto.
- Scaling: **PDO 20, score 600 a odds 50:1** → factor = 20/ln2 = 28,8539; offset = 600 − factor·ln50 = 487,1229.
  puntos(v,b) = −(β_v·WoE_{v,b} + β₀/n)·factor + offset/n.
- Umbrales del curso: PSI/CSI 0,10 / 0,25; IV ≥ 0,10; corr WoE ≤ 0,70; VIF 5/10; 8–14 variables;
  caída Gini DEV→HO < 0,10 y DEV→OOT < 0,15 (o relativa 20%/30% en clase 5); binomial p 0,05/0,01.

## 3. Idioma y estilo

- Español neutro-chileno. Término técnico en inglés entre paréntesis la primera vez:
  «puntos para duplicar las odds (points to double the odds, PDO)».
- En prosa, decimales con coma (0,10); en código y fórmulas LaTeX, punto.
- Frases directas. Nada de «¡Excelente pregunta!», «en resumen, es fundamental…». Nada de emojis
  salvo los semáforos 🟢🟡🔴 cuando se hable de tableros.
- Cuando algo sea convención y no ley, dilo. Cuando un umbral no tenga base teórica, dilo.
- **Honestidad de fuentes**: cita solo referencias que sabes que existen (autor, año, título). Si no
  estás seguro de edición/año, escribe «(verificar edición)». No inventes URLs. Para afirmaciones
  regulatorias (Chile/CMF, EE.UU., UE) usa WebSearch para verificar antes de afirmarlas; si no puedes
  verificar, formula con cautela («según entiendo… verificar con la norma vigente»).

## 4. Entregables por módulo (en `/home/claude/serie2/MXX_slug/`)

1. `MXX_slug.md` — documento de referencia.
2. `MXX_slug.py` — notebook **Marimo** autocontenido.
3. Opcional (solo si el brief lo pide o aporta de verdad): `MXX_slug.xlsx` (calculadora con fórmulas)
   y/o `MXX_*.csv` (tablas de datos). Deben importarse limpio a Google Sheets.

### 4.1 Documento `.md` (4.000–8.000 palabras; más solo si el tema lo exige)

Estructura obligatoria (puedes renombrar levemente, no omitir):

```
# MXX · Título
> Ficha: clases del curso que profundiza · prerrequisitos (Serie 1/2) · archivos del módulo · tiempo estimado

## 1. Lo que vimos en el curso (y lo que quedó fuera)
   Resumen fiel con los números de Banco Austral/Andes. Luego: lista explícita de lo que el curso
   simplificó, omitió o dejó como convención.
## 2. Intuición
## 3. Formalización
   Derivaciones completas en LaTeX ($...$ y $$...$$). Paso a paso, sin saltos «es fácil ver que».
## 4. Variantes y alternativas de industria
   Tabla comparativa (método · qué resuelve · costo · cuándo usarlo · quién lo usa/regulación).
## 5. Cuándo falla: trampas y modos de falla
   Cada trampa: síntoma → causa → cómo se detecta → qué hacer.
## 6. Puente con ingeniería
   Cómo se implementa en un pipeline serio: contratos, tests (tipo CI), invariantes verificables,
   qué se congela, qué se versiona. Iair piensa en pipelines declarativos: úsalo.
## 7. Numpy desde cero vs librerías
   Qué función de numpy implementa el notebook y qué librería (scipy/statsmodels/sklearn/optbinning)
   hace lo mismo; diferencias numéricas o de convención (p. ej. signo de WoE, ddof, corrección de
   continuidad) y cuál usar en producción.
## 8. Aplicación: casos y números
   Con Banco Austral/Andes y, donde el brief lo pida, crédito de motos.
## 9. Preguntas de comité
   5–8 preguntas que haría un comité o un validador, con la respuesta modelo.
## 10. Ejercicios
   6–10 ejercicios (cálculo a mano, derivación, diseño, código) con soluciones en
   <details><summary>Solución</summary> … </details>.
## 11. Referencias
   Comentadas (1 línea: por qué leerla).
```

### 4.2 Notebook Marimo `.py`

- Empieza con el bloque PEP 723 (ver `plantilla_marimo.py`) declarando dependencias reales usadas
  (marimo, numpy, pandas, matplotlib + las librerías de la opción 2: scipy, statsmodels,
  scikit-learn; optbinning solo si el brief lo pide). Así `marimo edit --sandbox archivo.py` instala todo.
- Autocontenido: SIN red (salvo el integrador C2), SIN leer archivos externos. Usa el generador
  `generar_cartera()` y las herramientas `binear/tabla_woe/a_woe` de `_spec/comun.py` pegados
  VERBATIM en una celda propia titulada «Código común de la serie». Puedes agregar tus propios
  generadores de juguete para experimentos específicos.
- **Dos implementaciones** para cada cálculo central: (1) numpy desde cero, legible y comentada;
  (2) la librería estándar; y un `assert np.allclose(...)` (o comparación explicada si difieren por
  convención) que demuestre que coinciden. El lector quiere ver ambas opciones.
- Interactividad: al menos 3 controles `mo.ui` (slider, dropdown, number…) que cambien un
  experimento y muestren el efecto (tabla o gráfico). Cada sección: celda `mo.md` que explique qué
  mirar → código → salida → celda `mo.md` con la lectura (puede ser dinámica con f-strings).
- Experimentos que muestren **cuándo falla** la técnica (con verdad conocida del generador).
- Reglas Marimo: cada nombre global se define UNA sola vez en todo el notebook; variables locales
  de celda con prefijo `_`; la última expresión de la celda es lo que se muestra; no mutar objetos
  definidos en otra celda (copiar). Funciones reutilizables definidas en celdas y retornadas.
- Rendimiento: el notebook completo debe correr en < 90 s en CPU modesta (usa n moderados, B de
  bootstrap ≤ 500 en modo script; puedes permitir más vía slider).
- Última celda: «Checks del módulo» con asserts (coincidencias numpy vs librería, invariantes
  teóricas) — si falla uno, el notebook falla.
- Gráficos matplotlib limpios: título, ejes con unidades, leyenda si hay >1 serie, `figsize` moderado,
  sin estilos raros. Devuelve la figura (no `plt.show()`).

### 4.3 Planillas

- `.xlsx` generado con openpyxl: hoja `LEEME` (qué es, cómo usarla, supuestos) + hojas de cálculo con
  **fórmulas vivas** (no valores pegados) para que el lector cambie parámetros (celdas de parámetros
  con relleno amarillo claro). Formato mínimo (encabezados en negrita, formatos de número/porcentaje).
  Solo funciones que existen en Google Sheets y Excel (LN, EXP, SUMPRODUCT, INDEX/MATCH, IF, MAX,
  MIN, BINOM.DIST, NORM.S.INV, CHISQ.DIST.RT, etc.). Nada de macros, tablas dinámicas, LET/LAMBDA,
  referencias estructuradas ni validaciones complejas.
- `.csv` UTF-8, separador coma, punto decimal, encabezados en snake_case.

## 5. Verificación OBLIGATORIA antes de terminar

```
bash /home/claude/serie2/_spec/verificar.sh /home/claude/serie2/MXX_slug
```
Debe imprimir `TODO OK` (ejecuta el notebook como script, `marimo check`, export html sin celdas con
error, y recalcula cada .xlsx con LibreOffice buscando errores). Itera hasta lograrlo.
Además relee tu `.md`: fórmulas LaTeX balanceadas, números consistentes con el notebook y con el
curso, sin secciones vacías, sin «TODO».

## 6. Informe final al orquestador

Responde con: archivos creados (ruta + tamaño/palabras), resultado de verificar.sh, 3–5 hallazgos
numéricos clave que el notebook demuestra, y cualquier afirmación que no pudiste verificar.

# Serie 2 · Del embudo al gobierno

**Material complementario para «Modelador de Riesgo de Crédito» (Academia Bayes · Nikodym Advisory), clases 3 a 6 y labs 2 y 3.**

Continúa la Serie 1 (M0–M7 y E1–E8), que cubrió población, target, fábrica de variables, binning, WoE, IV y los temas de extensión. Esta serie empieza en el embudo multivariado y termina en el gobierno del modelo en producción. En total son 16 módulos de profundización (M08–M23) y un proyecto integrador (C2) sobre los datos reales de Financiera Andes.

---

## 1. Mapa: clase del curso → módulo

| Clase del curso | Tema en clase | Módulos que lo profundizan |
|---|---|---|
| **Clase 3 · Modelo e interpretación** (Lab 2 parte A) | PSI primero, IV ≥ 0,10, correlación WoE, VIF, stepwise, Gini por muestra, scaling PDO, scorecard, reason codes, optbinning | M08 · M09 · M10 · M11 · M12 · M13 · M14 |
| **Clase 4 · Calibración y estrategia** (Lab 2 parte B) | Nivel vs ranking, TC/PIT, ajuste de δ, curva de calibración, master scale, EL, cutoff, swap-set, granularidad del binning, missing −9/−99, familias de variables | M15 · M16 · M17 · M18 (+ M09 y M14 para las láminas nuevas de la v21) |
| **Clase 5 · Validación y monitoreo** (Lab 3 parte A) | AUC/Gini/KS con bootstrap, deciles, PSI del score y CSI, binomial por banda, Hosmer-Lemeshow, tablero con semáforos, diagnóstico por patrón | M12 · M08 · M19 · M20 |
| **Clase 6 · Implementación y gobierno** (Lab 3 parte B) | «En producción no existe DEV», artefacto, bugs del binner, contrato de datos, audit trail, cadena de hashes, sello, lineage, model card, RACI, gatillos, nikodym | M21 · M22 |
| Transversal (pedido tuyo) | Numpy optimizado vs scipy/statsmodels/sklearn/optbinning/nikodym | M23 |
| Todo el curso | El ciclo completo sobre Financiera Andes, dirigido por una configuración y con análisis de sensibilidad | **C2** |

## 2. Inventario

| Módulo | Documento | Notebook Marimo | Planilla / datos |
|---|---|---|---|
| **M08** Estabilidad: PSI, CSI y el orden del embudo | `M08_estabilidad_psi_csi.md` | `.py` | `M08_calculadora_psi.xlsx` |
| **M09** Redundancia, colinealidad y familias de variables | `.md` | `.py` | `M09_matriz_correlacion_woe.csv` |
| **M10** Regresión logística sobre WoE, desde la verosimilitud | `.md` | `.py` | — |
| **M11** Selección de variables: stepwise y alternativas | `.md` | `.py` | — |
| **M12** Poder discriminante: ROC, AUC, Gini, KS, CAP e incertidumbre | `.md` | `.py` | `M12_tabla_deciles_gini.xlsx` |
| **M13** Del logit al scorecard: scaling y tabla de puntos | `.md` | `.py` | `M13_scorecard_vivo.xlsx` |
| **M14** Interpretabilidad: contribuciones, monotonía, binning óptimo, reason codes | `.md` | `.py` | — |
| **M15** Calibración: nivel vs ranking | `.md` | `.py` | `M15_calibracion_intercepto.xlsx` |
| **M16** Master scale | `.md` | `.py` | `M16_master_scale.xlsx` |
| **M17** Estrategia: cutoff, pérdida esperada, rentabilidad y **caso motos** | `.md` | `.py` | **`M17_tabla_estrategia_motos.xlsx`** (la planilla principal) |
| **M18** Swap-set y el problema contrafactual | `.md` | `.py` | `M18_swap_2x2_ic.xlsx` |
| **M19** Backtesting de calibración: binomial, HL, Jeffreys, correlación | `.md` | `.py` | `M19_backtesting_bandas.xlsx` |
| **M20** Monitoreo: tablero, semáforos, gatillos y diagnóstico | `.md` | `.py` | `M20_tablero_monitoreo.xlsx`, `M20_tablero_plantilla.csv` |
| **M21** Implementación: artefacto congelado, contrato de datos y paridad | `.md` | `.py` | `M21_artefacto_v1.0.0.json`, `M21_artefacto.schema.json` |
| **M22** Gobierno: expediente, trazabilidad, model card, validación independiente | `.md` | `.py` | `M22_raci_gatillos.xlsx`, `M22_inventario_modelos.csv` |
| **M23** Numpy optimizado vs librerías | `.md` | `.py` | — |
| **C2** Integrador: pipeline declarativo sobre Financiera Andes | `.md` | `.py` (descarga los datos del curso) | — |

Cada documento sigue la misma estructura: qué vio el curso y qué dejó fuera, intuición, formalización con derivaciones, variantes de industria, modos de falla, puente con ingeniería, numpy vs librerías, aplicación con los números de Austral y Andes, preguntas de comité, ejercicios con solución y referencias comentadas.

## 3. Cómo ejecutar los notebooks

Los `.py` son notebooks **Marimo** autocontenidos. Cada uno declara sus dependencias en un bloque PEP 723 al inicio del archivo. Hay dos formas de abrirlos:

```bash
# Opción A: entorno aislado; Marimo instala las dependencias declaradas (requiere uv)
uvx marimo edit --sandbox M08_estabilidad_psi_csi.py

# Opción B: en tu entorno
pip install marimo numpy pandas matplotlib scipy statsmodels scikit-learn
pip install optbinning jsonschema pydantic   # solo M13, M14, M21 y M23 los usan (con respaldo en numpy)
marimo edit M08_estabilidad_psi_csi.py
```

- **Datos:** todos usan el generador `generar_cartera()` («Banco Sintético», con la PD verdadera conocida). Así se puede medir cada técnica contra la verdad, cosa imposible con datos reales. La única excepción es **C2**, que descarga Financiera Andes desde la URL del curso con el commit fijado. En C2, si ingresas tu correo del curso en `ID_ALUMNO`, reproduce tu muestra personal de los labs; vacío, usa la población completa.
- **Dos implementaciones:** cada cálculo central está hecho en numpy desde cero y con la librería estándar, con un `assert` que prueba que coinciden (o explica por qué difieren por convención).
- **Verificación:** todos los notebooks se ejecutaron de punta a punta como script, pasan `marimo check` y exportan a HTML sin celdas con error. Cada uno termina en una celda de checks. Versiones usadas: numpy 2.4 · pandas 3.0 · scipy 1.17 · statsmodels 0.15 · scikit-learn 1.8 · optbinning 1.0 · marimo 0.25.

**Planillas:** son `.xlsx` sin formato especial, con fórmulas vivas. Las celdas amarillas son parámetros editables. Se importan a Google Sheets manteniendo las fórmulas (Archivo → Importar); todas se recalcularon sin errores. Los `.csv` son plantillas de datos.

## 4. Rutas de lectura

- **Completa (recomendada, unas 6–8 semanas a 5 h por semana):** M08 → M09 → M10 → M11 → M12 → M13 → M14 → M15 → M16 → M17 → M18 → M19 → M20 → M21 → M22 → M23 → C2.
- **Corta, para defender un modelo ante un comité:** M12 → M13 → M15 → M17 → M19 → M20 → M22.
- **De ingeniería, para llevarlo a producción:** M13 → M21 → M22 → M23 → C2.
- **Para tu trabajo en crédito de motos:** M17 (LGD con garantía, breakeven, TMC, pricing) → M15 → M16 → M18 (exploración para aprender el swap-in) → M20.

## 5. Hallazgos transversales que vale la pena llevarse

Todos salen de los notebooks; los números son del generador sintético salvo que se indique otra cosa.

1. **Los umbrales del curso son convenciones sin base inferencial, y su significado cambia con n.** Umbrales que sí escalan con n y con el número de malos:

   | Umbral del curso | Qué pasa en realidad | Alternativa que escala | Módulo |
   |---|---|---|---|
   | PSI 0,10 | Sin drift da alarma el 83% de las veces con n = 100; con n = 30.000 el crítico real es 0,0011 | Umbral por la distribución nula del PSI, que se aproxima con una χ² | M08 |
   | Caída de Gini 0,10 | Con 30 malos aparece por azar el 19% de las veces | Umbral ≈ 1,05/√(nº de malos) | M12 |
   | IV ≥ 0,10 | Equivale a un test al 0,35% con 165 malos | — | M11 |

2. **El p < 0,05 del stepwise sobre WoE ajustado en el mismo DEV no filtra ruido.** En la práctica es un test al ~43%, porque el WoE esconde K−1 grados de libertad. Lo que protege el embudo es el filtro de IV y la validación fuera de muestra (M10, M11).
3. **Con una sola variable en WoE, el MLE da β = −1 exacto.** La logística multivariada es Naive Bayes corregido por redundancia. Un |β| < 1 indica evidencia compartida con otras variables (M10).
4. **Un signo «equivocado» y significativo casi nunca sale de la colinealidad sola.** Aparece cuando el efecto condicional real tiene ese signo. Reconstruir la familia como «nivel + delta» deja los signos negativos sin perder Gini (M09).
5. **El atajo de Siddiqi para δ siempre se queda corto, y el sesgo se puede predecir.** Cumple δ_S = δ_exacto·(1−κ̄), así que mientras mejor discrimina el modelo, peor funciona el atajo (M15). Calibrar y validar en la misma muestra da p = 1 por construcción (M15, clase 5).
6. **Los 🟢 del backtest en bandas buenas casi no validan nada.** La potencia para detectar una PD real del doble es 7–35%; hacen falta unos 8 malos esperados por banda (M16). Con correlación entre deudores de 5%, el binomial rechaza de más en años malos (M19). El HL con χ² da rojos falsos; el p simulado es el correcto (M19).
7. **Un tablero sin deterioro tiene algo amarillo en el 51% de los meses.** La lectura correcta es el patrón por familia de indicadores, no el color suelto (M20).
8. **El cutoff óptimo en utilidad es el score de breakeven:** offset + factor·ln(pérdida/ganancia). Cada vez que la razón pérdida/ganancia se duplica, el corte sube un PDO. En el caso motos, el margen pesa más que el recupero de la garantía (M17).
9. **El swap-in real no tiene desempeño.** El modelo entrenado con aprobados lo subestima en unos 5 pp, y explorar aleatoriamente solo esa región es unas 7 veces más barato que explorar toda la zona bajo el corte (M18).
10. **El bug silencioso del binner** cambia decisiones con cero alarmas. Lo detecta solo el test «lote vs subconjunto». El cierre de los intervalos (a,b] vs [a,b) cambia el 11,6% de las decisiones (M21) y mueve de bin el 22,7% de las filas de `meses_desde_mora_12m` (M23).
11. **`LogisticRegression()` de sklearn regulariza por defecto (C = 1).** Los labs del curso lo usan dentro de `optbinning.Scorecard`, así que ese experimento no es un MLE. Además, optbinning 1.0 da WoE 0 al código −99 al transformar (M13, M23).
12. **En Andes (C2), las convenciones mueven las variables elegidas pero casi no el Gini.** En 48 combinaciones el Gini OOT queda en 0,626–0,663, menos que el ancho del intervalo bootstrap. Las variables cambian (8 a 14) y el cutoff se reparte entre 530 y 540: la decisión de negocio es más sensible a las convenciones que la métrica.

## 6. Caveats honestos

- **Regulación:** los puntos se verificaron en la web a septiembre de 2026 y cada módulo marca con «(verificar)» lo que no se pudo confirmar. Destacados: **SR 11-7 fue reemplazada por SR 26-2** (Fed, 17-abr-2026) · CMF en consulta de la RAN 21-9 (agosto de 2026) · Ley 21.719 vigente desde el 1-dic-2026 salvo que se apruebe una postergación · tasa máxima convencional según el certificado CMF vigente. Revisa la norma vigente antes de usar cualquiera de estos datos en un informe.
- **Parámetros ilustrativos:** los del caso motos (precio, pie, depreciación, costo de remate, margen) son genéricos, con rangos, y no corresponden a ninguna empresa. El apetito de riesgo de C2 (mora ≤ 8%, aprobación ≥ 60%) es un supuesto.
- **Generador sintético:** tiene efectos no lineales, códigos −9/−99, deriva de canal y deterioro macro plantados. Sirve para medir contra la verdad; sus niveles (tasa de malos ~11%) no son los de Austral.
- **Referencias:** cada una se citó solo si se conocía o se verificó. Las páginas o ediciones dudosas van marcadas.

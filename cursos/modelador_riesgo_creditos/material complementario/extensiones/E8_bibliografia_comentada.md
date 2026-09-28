# E8 · Bibliografía comentada: qué leer, en qué orden y para qué

**Serie: Modelador de Riesgo en Profundidad** · Fase 2, extensión 8 de 8
Cierra la serie: el mapa de lectura para pasar de "curso terminado" a "modelador experto"

---

## 1. Los tres libros canónicos (los del curso)

### Siddiqi — *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2ª ed., 2017)

**Qué es:** EL manual de práctica del scorecard clásico, escrito desde la trinchera (SAS). Cubre el pipeline completo del curso con el mismo vocabulario: definición de target y ventanas, muestras, fine/coarse classing, WoE/IV (sus umbrales son los de C2-24), regresión sobre WoE, escalamiento a puntos (PDO/offset), reject inference (fuzzy y parceling paso a paso), estrategia y cutoffs, monitoreo.

**Cómo leerlo:** es un libro de RECETAS con criterio, no de fundamentos: ideal como segunda pasada después del curso, capítulo a capítulo en paralelo a tu proyecto. Sus fortalezas: los detalles operativos que nadie más escribe (cómo documentar, qué mira un comité, el proceso organizacional). Sus límites: poca formalización estadística (para eso, Thomas) y una era pre-ML (para eso, la literatura de E5).

**Mapa a esta serie:** M1 (caps. de business plan del scorecard), M3-M4 (development sample, definiciones), M7 (el corazón del libro), E1 (reject inference), E2 (scaling/calibración), C4-C6 del curso (estrategia, implementación, monitoreo).

### Thomas, Edelman & Crook — *Credit Scoring and its Applications* (2ª ed., 2017, SIAM)

**Qué es:** el tratamiento académico riguroso: la estadística DEBAJO del scorecard. Modelos (logística, discriminante, árboles, y métodos de supervivencia para tiempo-al-default), medidas de desempeño con sus propiedades (ROC/AUC/KS/Gini formalizados), sample selection y reject inference con la conexión econométrica (Heckman), scoring de comportamiento y cadenas de Markov (¡la formalización de M3!), pricing y rentabilidad.

**Cómo leerlo:** como referencia por temas, no de corrido. Con tu formación cuantitativa es el libro que responde los "¿pero por qué funciona?" que Siddiqi deja abiertos. Especialmente valiosos: el capítulo de Markov para cobranza/comportamiento (M3 §2.2 en serio), y el tratamiento de las métricas (E3).

**Mapa a esta serie:** M3 (cadenas de Markov), M4 (muestras y selección), M7 (fundamentos de las medidas univariadas), E1 (la versión formal), E2-E3 (calibración y varianza de métricas).

### Anderson — *The Credit Scoring Toolkit: Theory and Practice for Retail Credit Risk Management and Decision Automation* (2007)

**Qué es:** la enciclopedia del ECOSISTEMA: historia del scoring, burós y sus datos (el mejor tratamiento escrito: E4), el ciclo de crédito completo (originación/comportamiento/cobranza: M1), la operación (motores de decisión, overrides, estrategias champion/challenger), regulación y ética, y la organización del área de riesgo.

**Cómo leerlo:** por partes según necesidad; es enorme (800+ pp) y algo datado en tecnología, pero insuperable en contexto de negocio e industria. Es el libro que te hace sonar senior en las conversaciones que no son de estadística.

**Mapa a esta serie:** M1 (el ciclo y la economía), E4 (burós), E6 (el marco regulatorio en perspectiva), C6 del curso (gobernanza y operación).

**Orden sugerido para ti:** Siddiqi de corrido durante el curso (refuerza cada clase) → Thomas por temas cuando quieras el fundamento (empezando por Markov y métricas) → Anderson como consulta permanente de contexto.

## 2. Segunda capa: los complementos por tema

- **Baesens, Rösch & Scheule — *Credit Risk Analytics* (2016):** el pipeline completo con código; bueno como puente práctica-teoría y para LGD/EAD (lo que el curso no cubre).
- **Efron & Tibshirani — *An Introduction to the Bootstrap*:** E3; los primeros capítulos bastan.
- **Breeden — *Reinventing Retail Lending Analytics*:** vintage/APC y forecasting bajo escenarios; el complemento de M3+E7.
- **Lessmann et al. (2015), *Benchmarking state-of-the-art classification algorithms for credit scoring* (EJOR):** el paper de referencia para la conversación scorecard vs ML (E5) con números en vez de opiniones.
- **Rudin (2019), *Stop explaining black box machine learning models for high stakes decisions...* (Nature MI):** la posición pro-interpretabilidad estructural, bien argumentada; contrapunto necesario a la fiebre SHAP.
- **Hand & Henley (1997), *Statistical classification methods in consumer credit scoring: a review* (JRSS-A):** el survey clásico; corto y todavía lúcido sobre qué importa y qué no.
- **BCBS WP14, *Studies on the Validation of Internal Rating Systems*:** el documento que une E2, E3 y E6: qué significa "validar" para un supervisor.
- **Normativa local (para Chile):** Compendio de Normas Contables de la CMF (B-1) y el marco de Basilea III local — leer la fuente al menos una vez.

## 3. Tercera capa: para el perfil DS→riesgo

- **Documentación de `optbinning` (Navas-Palencia):** la librería de binning óptimo de referencia en Python; su paper/manual explica la optimización con restricciones de monotonía — la versión automatizada de M7 §2.2 (úsala como propuesta, no como veredicto).
- **`scorecardpy` (Xie):** el port de scorecard a Python más usado; ojo con la convención de signo del WoE (M7 §3.2).
- **Kuhn & Johnson — *Feature Engineering and Selection*:** la mirada ML-general sobre lo que M5 hace a la manera de riesgo; útil para el diálogo entre ambos mundos.
- **Molnar — *Interpretable Machine Learning* (online):** el manual de SHAP/PDP/contrafactuales para la ruta E5.

## 4. Cómo estudiar esto (método, no solo lista)

1. **Regla del proyecto ancla:** cada lectura se digiere aplicándola al proyecto del curso (Financiera Andes) o a los notebooks de esta serie: leer el capítulo de reject inference de Siddiqi Y correr fuzzy sobre datos sintéticos vale por diez lecturas pasivas.
2. **Regla de la pregunta de comité:** al cerrar cada tema, formula las 3 preguntas que un comité/validador haría y escribe las respuestas. Las tres preguntas con que cierran C1 y C2 son el modelo del género.
3. **Regla del vocabulario bilingüe:** las entrevistas y la literatura son en inglés; el glosario de M0 es la base — al leer, anota el término inglés de cada concepto del curso.
4. **Secuencia de 90 días sugerida:** mes 1 = curso + Siddiqi + Fase 1 de esta serie; mes 2 = proyecto propio punta a punta + Thomas por temas + E1-E3; mes 3 = E4-E7, un challenger ML sobre tu propio proyecto (E5), y el simulacro de defensa ante comité con las preguntas de autoevaluación de toda la serie.

---

**Fin de la serie.** El pipeline sigue en las clases C3-C6 del curso (modelo, calibración, validación, gobernanza) — y todos los módulos de esta serie dejaron tendidos los puentes hacia ellas: la selección multivariada (M7 §6), la calibración (E2), la validación con intervalos (E3) y la gobernanza (E6). Buen modelaje.

# E3 · Bootstrap e incertidumbre: métricas con intervalos, no con puntos

**Serie: Modelador de Riesgo en Profundidad** · Fase 2, extensión 3 de 8
Origen: teaser de C5 en C1-18 ("si la cartera es chica... el bootstrap ayuda a medir la incertidumbre")

---

## 1. El problema: 165 malos no dan para certezas

Las métricas de un scorecard (Gini, KS, IV por variable, tasa de malos por tramo) son **estimaciones** calculadas sobre una muestra, y su varianza depende sobre todo del conteo de la clase escasa. Con los números del curso (15.665 clientes, 165 malos en DEV tras exclusiones), el Gini puntual de HO puede moverse ±5-8 puntos solo por azar muestral. Consecuencias prácticas de ignorarlo:

- Declarar "deterioro" cuando el Gini OOT cae 4 puntos... dentro del ruido.
- Elegir la variable A (IV 0.42) sobre la B (IV 0.38) como si la diferencia significara algo.
- Prometerle al comité un Gini de 58 que era 58±7.

La solución barata y general: **bootstrap** — reportar cada métrica con su intervalo de confianza y tomar decisiones con los intervalos.

## 2. La mecánica

### 2.1 Bootstrap no paramétrico básico

Remuestrear con reemplazo n filas de la muestra, recalcular la métrica, repetir B veces (500-2.000), y tomar percentiles de la distribución resultante (el IC 95% = percentiles 2,5 y 97,5). La intuición: la variabilidad entre remuestras imita la variabilidad que veríamos entre muestras reales de la misma población.

### 2.2 Los tres refinamientos obligatorios en crédito

1. **Bootstrap por cliente, no por fila.** Si hay múltiples filas por cliente (o cualquier estructura de grupo), se remuestrean **clientes completos** — el equivalente bootstrap de la partición por cliente de M4. Remuestrear filas subestima la varianza (las filas del mismo cliente no son independientes).
2. **Bootstrap por cosecha (block/cluster temporal) cuando la pregunta es temporal.** Para la incertidumbre de la estabilidad entre cosechas, remuestrear cosechas enteras: la correlación intra-cosecha (todos comparten el mismo mes del ciclo) es justamente lo que se quiere preservar.
3. **Estratificar el target si los malos son muy pocos:** remuestrear buenos y malos por separado (manteniendo los conteos) estabiliza cuando B remuestras podrían dejar tramos sin malos. Alternativa con mejores propiedades para métricas de tasas: intervalos binomiales exactos (Clopper-Pearson) o de Wilson para la tasa de malos por tramo/cosecha — más baratos que bootstrap y correctos para proporciones.

### 2.3 Dónde aplicarlo en el pipeline del curso

| Métrica | Uso del intervalo |
|---|---|
| IV por variable (DEV y HO) | La Trampa 2 de M7 con números: IV 0.85 [0.55, 1.20] no es "casi 1". Descartar variables cuyo IC en HO cubre 0.02. |
| Gini/KS del modelo (HO, OOT) | ¿La caída DEV→OOT es señal o ruido? Si los IC se solapan generosamente, es ruido. |
| Tasa de malos por tramo de score | Los tramos altos tienen poquísimos malos: sus tasas necesitan IC para calibración (E2) y para el cutoff. |
| PSI | También es una estimación; con muestras chicas el "0.12, amarillo" puede ser [0.05, 0.22]. |
| Comparación de dos modelos | Bootstrap **pareado**: remuestrear una vez y evaluar ambos modelos en la misma remuestra; el IC de la DIFERENCIA de Ginis es la prueba correcta (las métricas están correlacionadas al compartir muestra). |

### 2.4 Qué no arregla el bootstrap

El bootstrap estima la varianza **dentro del proceso generador de la muestra**: no ve el sesgo (una fuga infla el Gini y su IC completo), no ve el cambio de régimen (el IC de DEV no informa sobre 2027), y no sustituye la OOT. Es un instrumento de honestidad sobre el ruido, no una defensa contra el error sistemático.

## 3. Recetas concretas

**IC del Gini (por cliente, pareado si comparas):**
```
B = 1000
para b en 1..B:
    ids_b = muestreo con reemplazo de los id_cliente
    filas_b = todas las filas de esos ids
    gini_b = gini(target_b, score_b)
IC95 = percentiles 2.5 y 97.5 de {gini_b}
```

**Decisión de variables con la pareja (IV_DEV, IV_HO) + IC:** candidata sólida = IC de HO enteramente sobre 0.05 y solapado con el de DEV (estable); sospechosa = IC de DEV muy por encima del de HO (Trampa 2); auditable = IC de HO sobre 0.5 (Trampa 3).

**Presupuesto de cómputo:** el bootstrap es vergonzosamente paralelizable, y con métricas vectorizadas (el AUC por rangos de M4) 1.000 réplicas de una muestra de 15k corren en segundos. No hay excusa de costo.

## 4. Alternativas y complementos

- **Fórmulas asintóticas:** el error estándar del AUC (DeLong) evita el bootstrap para el Gini; útil como verificación cruzada.
- **Validación cruzada repetida (por cliente, dentro de DEV):** estima la varianza del *procedimiento* completo (binning+selección+ajuste), no solo de la métrica final — más caro y más honesto cuando el pipeline tiene muchas decisiones dependientes de datos.
- **Jackknife por cosecha:** dejar fuera una cosecha a la vez; barato y muestra qué cosecha "sostiene" el resultado.

## 5. Para profundizar

- Efron & Tibshirani, *An Introduction to the Bootstrap* — el clásico; caps. iniciales bastan.
- DeLong et al. (1988) para la varianza del AUC; Hanley & McNeil (1982) como antecedente.
- En scoring: los capítulos de validación de Thomas/Edelman/Crook y las guías de validación de Basilea (WP14) piden exactamente esto: significancia y estabilidad de las métricas, no puntos.

## 6. Preguntas de autoevaluación

1. ¿Por qué remuestrear filas cuando hay varias por cliente subestima la varianza? Conecta con la fuga de identidad de M4.
2. Gini HO 61 [55, 67], Gini OOT 54 [45, 62]. ¿Reportas deterioro? ¿Qué análisis adicional pides antes de rediseñar?
3. Diseña el bootstrap correcto para el PSI entre DEV y TTD. ¿Qué se remuestrea?
4. ¿Cuándo preferirías un intervalo de Wilson a un bootstrap para la tasa de malos de un tramo con 900 casos y 6 malos?
5. Explica a un comité en tres frases por qué el informe trae intervalos y no números "limpios", sin usar la palabra "varianza".

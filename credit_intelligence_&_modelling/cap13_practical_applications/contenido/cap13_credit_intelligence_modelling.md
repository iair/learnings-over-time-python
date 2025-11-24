# Capítulo 13: Aplicación Práctica
## Credit Intelligence and Modelling - Raymond Anderson

---

# 13.1 Transformaciones de Características

Las transformaciones de características son técnicas fundamentales para preparar variables antes del modelado. En el contexto de conversión de leads y scoring crediticio, transformar las variables permite mejorar la interpretabilidad, estabilidad y poder predictivo del modelo.

---

## 13.1.1 Rescale (Reescalado)

### Definición Formal

El reescalado es una técnica de transformación que ajusta el rango de valores de una variable a una escala específica, típicamente [0,1] o [-1,1]. Las dos técnicas más comunes son:

**Min-Max Scaling (Normalización):**
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Z-Score Standardization (Estandarización):**
$$X_{standardized} = \frac{X - \mu}{\sigma}$$

Donde:
- $X$ = valor original
- $X_{min}$, $X_{max}$ = valores mínimo y máximo de la variable
- $\mu$ = media de la variable
- $\sigma$ = desviación estándar

### Analogía

Imagina que estás comparando el desempeño de vendedores de motos en diferentes ciudades. En Ciudad A, un vendedor cierra entre 5 y 50 ventas mensuales, mientras que en Ciudad B el rango es de 100 a 500 ventas. Sin reescalar, los números de Ciudad B dominarían cualquier análisis. El reescalado es como convertir ambos a "porcentaje de su potencial máximo", permitiendo una comparación justa.

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Escenario:** Galgo quiere predecir qué leads se convertirán en clientes. Tienen dos variables:
- **Ingreso mensual:** Rango de $3,000 a $50,000 MXN
- **Tiempo en empleo actual:** Rango de 0 a 360 meses

| Lead | Ingreso (MXN) | Tiempo Empleo (meses) |
|------|---------------|----------------------|
| A    | 15,000        | 24                   |
| B    | 35,000        | 180                  |
| C    | 8,000         | 6                    |

**Aplicando Min-Max Scaling:**

Para Ingreso:
$$Ingreso_{A} = \frac{15,000 - 3,000}{50,000 - 3,000} = \frac{12,000}{47,000} = 0.255$$

$$Ingreso_{B} = \frac{35,000 - 3,000}{47,000} = 0.681$$

$$Ingreso_{C} = \frac{8,000 - 3,000}{47,000} = 0.106$$

| Lead | Ingreso Escalado | Tiempo Escalado |
|------|------------------|-----------------|
| A    | 0.255            | 0.067           |
| B    | 0.681            | 0.500           |
| C    | 0.106            | 0.017           |

### Interpretación

- **0 a 1:** Después del reescalado, todas las variables están en la misma escala, lo que permite que el modelo trate cada variable de manera equitativa
- **Lead B** tiene los valores más altos en ambas dimensiones después del reescalado (0.681 y 0.500), indicando mayor estabilidad financiera y laboral
- **El reescalado NO cambia la distribución** de los datos ni elimina outliers; solo ajusta el rango
- Es especialmente útil cuando se usan algoritmos sensibles a la magnitud (redes neuronales, SVM, K-means)

---

## 13.1.2 Discretize (Discretización/Binning)

### Definición Formal

La discretización es el proceso de transformar variables continuas en categorías o "bins" discretos. En credit scoring, esta técnica es fundamental para crear scorecards interpretables.

**Fórmula de asignación a bins:**
$$Bin_i = \{x : c_{i-1} < x \leq c_i\}$$

Donde $c_0, c_1, ..., c_k$ son los puntos de corte que definen los k bins.

**Métodos comunes de discretización:**
1. **Equal Width (Amplitud igual):** Divide el rango en intervalos de igual tamaño
2. **Equal Frequency (Frecuencia igual):** Cada bin contiene aproximadamente el mismo número de observaciones
3. **Chi-Merge:** Combina bins adyacentes que no son estadísticamente diferentes
4. **Optimal Binning:** Maximiza el poder predictivo (Information Value)

### Analogía

La discretización es como clasificar estudiantes por rangos de calificación en lugar de usar el puntaje exacto. En lugar de decir "Juan tiene 87.3 y María tiene 87.5", los agrupamos como "ambos están en el rango B+ (85-89)". Esto simplifica el análisis sin perder información esencial sobre el desempeño.

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Escenario:** Discretizar el ingreso mensual de leads para predecir conversión

**Datos de 1,000 leads:**

| Rango Ingreso (MXN) | Total Leads | Convertidos | No Convertidos | Tasa Conversión |
|---------------------|-------------|-------------|----------------|-----------------|
| 0 - 8,000           | 150         | 15          | 135            | 10.0%           |
| 8,001 - 15,000      | 300         | 60          | 240            | 20.0%           |
| 15,001 - 25,000     | 280         | 84          | 196            | 30.0%           |
| 25,001 - 40,000     | 180         | 72          | 108            | 40.0%           |
| 40,001+             | 90          | 54          | 36             | 60.0%           |
| **Total**           | **1,000**   | **285**     | **715**        | **28.5%**       |

**Calculando Weight of Evidence (WOE) para cada bin:**

$$WOE_i = \ln\left(\frac{\%\text{ No Convertidos}_i}{\%\text{ Convertidos}_i}\right)$$

| Bin | % Convertidos | % No Convertidos | WOE |
|-----|---------------|------------------|-----|
| 0-8K | 15/285 = 5.26% | 135/715 = 18.88% | ln(18.88/5.26) = 1.278 |
| 8K-15K | 60/285 = 21.05% | 240/715 = 33.57% | ln(33.57/21.05) = 0.467 |
| 15K-25K | 84/285 = 29.47% | 196/715 = 27.41% | ln(27.41/29.47) = -0.072 |
| 25K-40K | 72/285 = 25.26% | 108/715 = 15.10% | ln(15.10/25.26) = -0.514 |
| 40K+ | 54/285 = 18.95% | 36/715 = 5.03% | ln(5.03/18.95) = -1.327 |

### Interpretación

- **WOE Positivo:** El bin tiene más "no convertidos" que "convertidos" proporcionalmente → Menor probabilidad de conversión
- **WOE Negativo:** El bin tiene más "convertidos" proporcionalmente → Mayor probabilidad de conversión
- **WOE cercano a 0:** La proporción de buenos y malos es similar al promedio general
- **Patrón monotónico:** El WOE decrece consistentemente a medida que aumenta el ingreso, lo cual es un buen indicador de una variable predictiva estable
- **Regla del 5%:** Cada bin debe contener al menos 5% de las observaciones para asegurar estabilidad estadística

---

# 13.2 Evaluación de Características

Las métricas de evaluación de características ayudan a determinar qué variables incluir en el modelo y cómo monitorear su estabilidad a lo largo del tiempo.

---

## 13.2.1 Information Value (IV) - Valor de Información

### Definición Formal

El Information Value (IV) mide el poder predictivo de una variable independiente respecto a una variable dependiente binaria. Se deriva de la teoría de la información y es esencialmente una divergencia de Kullback-Leibler simetrizada (también conocida como divergencia J).

$$IV = \sum_{i=1}^{n} (\%\text{Buenos}_i - \%\text{Malos}_i) \times WOE_i$$

Donde:
$$WOE_i = \ln\left(\frac{\%\text{Buenos}_i}{\%\text{Malos}_i}\right)$$

**Interpretación estándar del IV:**

| IV | Poder Predictivo |
|----|------------------|
| < 0.02 | No predictivo |
| 0.02 - 0.1 | Débil |
| 0.1 - 0.3 | Medio |
| 0.3 - 0.5 | Fuerte |
| > 0.5 | Sospechoso (posible overfitting o data leakage) |

### Analogía

El IV es como medir qué tan "chismosa" es una variable. Una variable con alto IV es como ese amigo que siempre sabe quién va a comprar y quién no — tiene información valiosa que distingue claramente entre los dos grupos. Una variable con IV bajo es como preguntar "¿de qué color es tu carro?" para predecir si comprarás una moto — probablemente irrelevante.

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Continuando con los datos de ingreso:**

| Bin | % Convertidos | % No Convertidos | Diferencia | WOE | Componente IV |
|-----|---------------|------------------|------------|-----|---------------|
| 0-8K | 5.26% | 18.88% | -13.62% | 1.278 | -0.136 × 1.278 = -0.174 |
| 8K-15K | 21.05% | 33.57% | -12.52% | 0.467 | -0.125 × 0.467 = -0.058 |
| 15K-25K | 29.47% | 27.41% | 2.06% | -0.072 | 0.021 × -0.072 = -0.002 |
| 25K-40K | 25.26% | 15.10% | 10.16% | -0.514 | 0.102 × -0.514 = -0.052 |
| 40K+ | 18.95% | 5.03% | 13.92% | -1.327 | 0.139 × -1.327 = -0.185 |

**Nota:** El IV siempre es positivo porque $(a-b) \times \ln(a/b)$ siempre tiene el mismo signo.

**Cálculo correcto:**
$$IV = |(-0.136) \times 1.278| + |(-0.125) \times 0.467| + ... $$

$$IV_{Ingreso} \approx 0.174 + 0.058 + 0.002 + 0.052 + 0.185 = 0.471$$

**Comparación con otras variables de Galgo:**

| Variable | IV | Poder Predictivo |
|----------|-----|------------------|
| Ingreso mensual | 0.471 | Fuerte |
| Antigüedad laboral | 0.285 | Medio-Fuerte |
| Edad | 0.156 | Medio |
| Tipo de empleo | 0.089 | Débil |
| Estado civil | 0.023 | Muy débil |
| Género | 0.008 | No predictivo |

### Interpretación

- **Ingreso mensual (IV=0.471):** Es la variable más predictiva. Los leads de mayor ingreso tienen significativamente mayor probabilidad de conversión
- **Variables con IV < 0.02** deberían excluirse del modelo final
- **IV muy alto (>0.5)** puede indicar problemas como:
  - Data leakage (la variable "ve el futuro")
  - Definición circular con el target
  - Debe investigarse antes de usar
- El IV es **aditivo por bins**, permitiendo entender qué segmentos de la variable aportan más poder predictivo

---

## 13.2.2 Population Stability Index (PSI) - Índice de Estabilidad Poblacional

### Definición Formal

El PSI mide cuánto ha cambiado la distribución de una variable (o score) entre dos momentos en el tiempo. Es fundamental para el monitoreo de modelos en producción.

$$PSI = \sum_{i=1}^{n} (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$

Donde:
- $A_i$ = Proporción actual en el bin i
- $E_i$ = Proporción esperada (baseline/desarrollo) en el bin i

**Interpretación estándar:**

| PSI | Interpretación | Acción |
|-----|----------------|--------|
| < 0.10 | Sin cambio significativo | Continuar monitoreo normal |
| 0.10 - 0.25 | Cambio moderado | Investigar causas |
| > 0.25 | Cambio significativo | Recalibrar o redevelopar modelo |

### Analogía

El PSI es como un termómetro para tu modelo. Imagina que entrenas un modelo de conversión usando datos de clientes durante una época de bonanza económica. Si la economía cambia, los leads que llegan pueden ser muy diferentes (más jóvenes, menores ingresos, diferentes empleos). El PSI detecta este "cambio de clima" en tu población antes de que el modelo falle.

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Escenario:** Comparar la distribución de scores entre el período de desarrollo (Q1 2024) y producción (Q4 2024)

| Rango Score | % Desarrollo (E) | % Producción (A) | (A-E) | ln(A/E) | Componente PSI |
|-------------|------------------|------------------|-------|---------|----------------|
| 0-200 | 5% | 8% | 3% | 0.470 | 0.0141 |
| 201-300 | 12% | 15% | 3% | 0.223 | 0.0067 |
| 301-400 | 25% | 28% | 3% | 0.113 | 0.0034 |
| 401-500 | 30% | 27% | -3% | -0.105 | 0.0032 |
| 501-600 | 18% | 14% | -4% | -0.251 | 0.0100 |
| 601+ | 10% | 8% | -2% | -0.223 | 0.0045 |
| **Total** | **100%** | **100%** | | | **PSI = 0.042** |

**Análisis por variable individual:**

| Variable | PSI | Interpretación |
|----------|-----|----------------|
| Score total | 0.042 | Estable ✓ |
| Ingreso | 0.087 | Estable ✓ |
| Edad | 0.034 | Estable ✓ |
| Antigüedad laboral | 0.156 | ⚠️ Moderado |
| Canal de captación | 0.312 | 🚨 Significativo |

### Interpretación

- **PSI del score = 0.042:** El modelo es estable en general
- **Canal de captación (PSI=0.312):** Hay un cambio significativo en cómo llegan los leads:
  - Posible nueva campaña de marketing
  - Cambio en el mix de canales digitales vs. presenciales
  - Requiere investigación inmediata
- **Acciones recomendadas:**
  1. Investigar qué cambió en el canal de captación
  2. Evaluar si el modelo sigue performando bien en los nuevos canales
  3. Considerar recalibración si el performance se degrada
- El PSI es **simétrico conceptualmente** pero **no en la práctica** porque los puntos de corte se definen sobre la población base

---

## 13.2.3 Chi-Square (Chi-Cuadrado)

### Definición Formal

La prueba Chi-Cuadrado evalúa si existe una asociación estadísticamente significativa entre dos variables categóricas. En credit scoring, se usa principalmente para:
1. Validar si los bins de una variable discriminan significativamente
2. Selección de variables
3. Comparar distribuciones entre poblaciones

$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Donde:
- $O_{ij}$ = Frecuencia observada en celda (i,j)
- $E_{ij}$ = Frecuencia esperada = (Total fila i × Total columna j) / Total general
- Grados de libertad = (r-1) × (c-1)

### Analogía

Chi-Cuadrado es como un árbitro que evalúa si un dado está cargado. Si lanzas un dado 60 veces, esperarías aproximadamente 10 veces cada número. Si el "1" sale 25 veces y el "6" solo 2 veces, Chi-Cuadrado te dice qué tan probable es que esa diferencia sea solo por azar. En nuestro contexto, evalúa si la diferencia en tasas de conversión entre bins es real o solo ruido.

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Escenario:** Evaluar si el tipo de empleo tiene asociación significativa con la conversión

**Tabla de contingencia observada:**

| Tipo Empleo | Convertidos | No Convertidos | Total |
|-------------|-------------|----------------|-------|
| Asalariado | 180 | 320 | 500 |
| Independiente | 60 | 240 | 300 |
| Jubilado | 30 | 70 | 100 |
| Otro | 15 | 85 | 100 |
| **Total** | **285** | **715** | **1,000** |

**Cálculo de frecuencias esperadas (si no hubiera relación):**

$$E_{Asalariado,Conv} = \frac{500 \times 285}{1000} = 142.5$$

| Tipo Empleo | E(Convertidos) | E(No Convertidos) |
|-------------|----------------|-------------------|
| Asalariado | 142.5 | 357.5 |
| Independiente | 85.5 | 214.5 |
| Jubilado | 28.5 | 71.5 |
| Otro | 28.5 | 71.5 |

**Cálculo del estadístico Chi-Cuadrado:**

$$\chi^2 = \frac{(180-142.5)^2}{142.5} + \frac{(320-357.5)^2}{357.5} + \frac{(60-85.5)^2}{85.5} + ...$$

$$\chi^2 = 9.87 + 3.93 + 7.60 + 3.03 + 0.08 + 0.03 + 6.39 + 2.55 = 33.48$$

**Grados de libertad:** (4-1) × (2-1) = 3

**Valor crítico** (α=0.05, gl=3): 7.815

**p-valor:** < 0.0001

### Interpretación

- **χ² = 33.48 >> 7.815:** Rechazamos la hipótesis nula de independencia
- **Conclusión:** El tipo de empleo está **significativamente asociado** con la conversión
- **Análisis por celda:**
  - Asalariados: Observados (180) > Esperados (142.5) → **Sobre-representados** en conversiones
  - Independientes: Observados (60) < Esperados (85.5) → **Sub-representados** en conversiones
  - "Otro": Observados (15) < Esperados (28.5) → Mayor riesgo de no conversión
- **Aplicación en binning:** Chi-Merge usa esta prueba para decidir si dos bins adyacentes deben combinarse (si χ² es bajo, los bins son similares y pueden fusionarse)

---

# 13.3 Evaluación de Modelos

Las métricas de evaluación de modelos miden qué tan bien el modelo completo discrimina entre las clases objetivo.

---

## 13.3.1 Curva de Lorenz y Coeficiente de Gini

### Definición Formal

**Curva de Lorenz:** Gráfico que muestra la distribución acumulada de "malos" (eje Y) vs. la distribución acumulada de la población ordenada por score (eje X), de menor a mayor score.

**Coeficiente de Gini:** Mide el poder discriminatorio del modelo. Es el doble del área entre la curva del modelo y la línea diagonal (modelo aleatorio).

$$Gini = \frac{A_{modelo} - A_{aleatorio}}{A_{perfecto} - A_{aleatorio}} = 2 \times AUC - 1$$

Donde AUC = Área bajo la curva ROC

**Relación con AUC:**
$$Gini = 2 \times AUC - 1$$

| Gini | Interpretación |
|------|----------------|
| < 0.20 | Pobre |
| 0.20 - 0.40 | Aceptable |
| 0.40 - 0.60 | Bueno |
| 0.60 - 0.80 | Muy bueno |
| > 0.80 | Excelente (verificar overfitting) |

### Analogía

Imagina que tienes una lista de 1,000 leads y debes contactarlos para venderles motos. Sin modelo, contactarías al azar. Con un modelo perfecto, contactarías primero a todos los que van a comprar. El Gini mide qué tan cerca está tu modelo de ese "orden perfecto". Un Gini de 0.60 significa que tu modelo captura el 60% de la mejora posible sobre el azar.

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Datos ordenados por score (de menor a mayor probabilidad de conversión):**

| Decil | % Acum. Población | % Acum. No Convertidos | % Acum. Convertidos |
|-------|-------------------|------------------------|---------------------|
| 1 | 10% | 3.2% | 17.9% |
| 2 | 20% | 7.4% | 33.7% |
| 3 | 30% | 13.1% | 47.4% |
| 4 | 40% | 20.3% | 58.9% |
| 5 | 50% | 29.5% | 68.4% |
| 6 | 60% | 40.8% | 76.5% |
| 7 | 70% | 54.2% | 83.2% |
| 8 | 80% | 69.7% | 89.5% |
| 9 | 90% | 86.0% | 95.1% |
| 10 | 100% | 100% | 100% |

**Cálculo del Gini usando la fórmula trapezoidal:**

$$Gini = 1 - 2 \times \sum_{i=1}^{n} \frac{(x_i - x_{i-1})(y_i + y_{i-1})}{2}$$

Para nuestros datos: **Gini ≈ 0.52**

**Interpretación gráfica:**

```
    100% |                              ●
    No   |                         ●
    Conv |                    ●
    (%)  |               ●       Modelo
         |          ●          Actual
         |     ●
         | ●              Línea de
         |           igualdad (azar)
         |________________________
         0%                    100%
              % Población
```

### Interpretación

- **Gini = 0.52:** El modelo tiene un poder discriminatorio **bueno**
- **Lectura práctica:** Al ordenar leads por el score del modelo:
  - El 20% de leads con menor score contiene el 33.7% de los no convertidos
  - El 30% de leads con mayor score (deciles 8-10) contiene solo el 10.5% de los no convertidos
- **Comparación con benchmarks de industria:**
  - Modelos de application scoring: Gini típico 0.40-0.60
  - Modelos de behavior scoring: Gini típico 0.50-0.70
  - Modelos de collection scoring: Gini típico 0.30-0.50

---

## 13.3.2 Cumulative Accuracy Profile (CAP), Accuracy Ratio (AR) y Lift

### Definición Formal

**Cumulative Accuracy Profile (CAP):** Similar a la curva de Lorenz pero ordenando de mayor a menor score (los más riesgosos primero). Muestra qué proporción de "malos" se captura al seleccionar una proporción de la población ordenada por score.

**Accuracy Ratio (AR):**
$$AR = \frac{A_R}{A_P}$$

Donde:
- $A_R$ = Área entre la curva del modelo y el modelo aleatorio
- $A_P$ = Área entre el modelo perfecto y el modelo aleatorio

**Nota importante:** AR = Gini (son matemáticamente equivalentes)

**Lift:**
$$Lift_i = \frac{\text{Tasa de respuesta en decil } i}{\text{Tasa de respuesta global}}$$

**Lift Acumulado:**
$$Lift_{acum,i} = \frac{\text{Tasa de respuesta acumulada hasta decil } i}{\text{Tasa de respuesta global}}$$

### Analogía

Imagina que eres un buscador de oro. El modelo aleatorio es como cavar en cualquier lugar al azar. El modelo perfecto es tener un mapa exacto de dónde está todo el oro. Tu modelo de scoring es como un detector de metales imperfecto pero útil. El CAP muestra qué tan eficiente eres encontrando oro comparado con el azar y con el detector perfecto. El Lift te dice "usando mi detector, encuentro 3 veces más oro que cavando al azar en el primer 10% del terreno".

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Escenario:** Evaluar qué tan bien el modelo identifica leads que NO van a convertir (para optimizar esfuerzos de remarketing)

**Ordenando de mayor a menor riesgo de NO conversión:**

| Decil | % Población | No Conv en Decil | No Conv Acum | Tasa No Conv | Lift | Lift Acum |
|-------|-------------|------------------|--------------|--------------|------|-----------|
| 1 (peor) | 10% | 143 | 143 | 20.0% | 2.80 | 2.80 |
| 2 | 10% | 122 | 265 | 17.1% | 2.39 | 2.60 |
| 3 | 10% | 100 | 365 | 14.0% | 1.96 | 2.38 |
| 4 | 10% | 86 | 451 | 12.0% | 1.68 | 2.21 |
| 5 | 10% | 72 | 523 | 10.1% | 1.41 | 2.04 |
| 6 | 10% | 64 | 587 | 9.0% | 1.26 | 1.91 |
| 7 | 10% | 57 | 644 | 8.0% | 1.12 | 1.80 |
| 8 | 10% | 50 | 694 | 7.0% | 0.98 | 1.70 |
| 9 | 10% | 36 | 730 | 5.0% | 0.70 | 1.59 |
| 10 (mejor) | 10% | 21 | 751 | 2.9% | 0.41 | 1.47 |

**Tasa global de no conversión:** 715/1000 = 71.5%

**Cálculo del Lift para Decil 1:**
$$Lift_1 = \frac{20\%}{7.15\%} = 2.80$$

**Accuracy Ratio:**
$$AR = \frac{\text{Área CAP modelo} - \text{Área aleatorio}}{\text{Área perfecto} - \text{Área aleatorio}} \approx 0.52$$

### Interpretación

- **Lift Decil 1 = 2.80:** En el 10% de leads con peor score, hay 2.8 veces más no-convertidos que el promedio
- **Aplicación práctica en Galgo:**
  - Si solo puedes contactar al 30% de leads (por recursos limitados), contacta los deciles 8-10
  - El modelo captura el 50.4% de las conversiones contactando solo el 30% de leads
  - Evita gastar recursos en el decil 1, donde solo el 14.3% convertirá (vs. 28.5% promedio)
- **Lift acumulado de 2.04 en el decil 5:** Al filtrar el 50% de leads con peor score, la tasa de no conversión es el doble del promedio
- **AR = 0.52:** Confirma el poder discriminatorio bueno, consistente con el Gini calculado previamente

---

# 13.4 Odds and Sods (Métricas Adicionales)

Esta sección cubre métricas especializadas menos comunes pero útiles en contextos específicos.

---

## 13.4.1 Deviance (Devianza)

### Definición Formal

La Devianza mide la bondad de ajuste de un modelo de regresión logística. Es análoga a la suma de cuadrados residuales en regresión lineal.

$$Deviance = -2 \times (\log L_{modelo} - \log L_{saturado})$$

Donde:
- $\log L_{modelo}$ = Log-verosimilitud del modelo ajustado
- $\log L_{saturado}$ = Log-verosimilitud del modelo saturado (perfecto)

**Tipos de Devianza:**

1. **Null Deviance ($D_0$):** Devianza del modelo solo con intercepto (sin predictores)
2. **Residual Deviance ($D$):** Devianza del modelo con predictores

**Pseudo R² de McFadden:**
$$R^2_{McFadden} = 1 - \frac{D}{D_0} = 1 - \frac{\text{Deviance residual}}{\text{Deviance nula}}$$

**Prueba de Likelihood Ratio:**
$$LR = D_0 - D \sim \chi^2_{p}$$

Donde p = número de predictores añadidos

### Analogía

La Devianza es como medir cuánto "error" comete tu modelo. El modelo saturado es como tener una respuesta perfecta para cada observación (Devianza = 0). El modelo nulo es como predecir siempre la probabilidad promedio sin usar ninguna información (la peor opción informada). Tu modelo está en algún punto intermedio. Cuanto más cercana esté la Devianza a 0, mejor ajusta el modelo.

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Comparación de modelos con diferente número de variables:**

| Modelo | Variables | Deviance | Δ Deviance | p-valor | Pseudo R² |
|--------|-----------|----------|------------|---------|-----------|
| Nulo | (intercepto) | 1,178.5 | - | - | 0.000 |
| M1 | + Ingreso | 1,098.2 | 80.3 | <0.001 | 0.068 |
| M2 | + Antigüedad | 1,056.8 | 41.4 | <0.001 | 0.103 |
| M3 | + Edad | 1,044.1 | 12.7 | <0.001 | 0.114 |
| M4 | + Canal | 1,032.5 | 11.6 | <0.001 | 0.124 |
| M5 | + Estado Civil | 1,031.2 | 1.3 | 0.254 | 0.125 |

**Output típico de regresión logística:**

```
Null deviance: 1178.5 on 999 degrees of freedom
Residual deviance: 1032.5 on 995 degrees of freedom
AIC: 1042.5

Number of Fisher Scoring iterations: 4
```

### Interpretación

- **Reducción de Deviance:** Cada variable añadida reduce la devianza, mejorando el ajuste
- **M1 (+ Ingreso):** Δ Deviance = 80.3, p < 0.001 → Ingreso es significativo
- **M5 (+ Estado Civil):** Δ Deviance = 1.3, p = 0.254 → Estado civil NO es significativo; no incluir en el modelo final
- **Pseudo R² = 0.124:** El modelo explica aproximadamente el 12.4% de la variabilidad
  - En credit scoring, valores de 0.10-0.20 son típicos
  - No es directamente comparable con R² de regresión lineal
- **Regla práctica:** Añadir una variable solo si la reducción en Deviance es estadísticamente significativa (p < 0.05)
- **AIC = 1042.5:** Útil para comparar modelos no anidados; menor es mejor

---

## 13.4.2 Calinski-Harabasz Statistic (Estadístico de Calinski-Harabasz)

### Definición Formal

El índice de Calinski-Harabasz (también llamado Variance Ratio Criterion) evalúa la calidad de una segmentación o clustering. Mide qué tan bien separados y compactos son los clusters.

$$CH = \frac{SS_B / (k-1)}{SS_W / (n-k)} = \frac{BCSS / (k-1)}{WCSS / (n-k)}$$

Donde:
- $SS_B$ (BCSS) = Suma de cuadrados entre clusters (Between-Cluster Sum of Squares)
- $SS_W$ (WCSS) = Suma de cuadrados dentro de clusters (Within-Cluster Sum of Squares)
- $k$ = Número de clusters
- $n$ = Número de observaciones

$$SS_B = \sum_{j=1}^{k} n_j \times ||\bar{x}_j - \bar{x}||^2$$

$$SS_W = \sum_{j=1}^{k} \sum_{i \in C_j} ||x_i - \bar{x}_j||^2$$

### Analogía

Imagina que estás organizando leads en grupos para campañas de marketing personalizadas. El índice Calinski-Harabasz es como evaluar qué tan bien "ordenaste las canicas por color":
- **Alta separación (alto SS_B):** Los grupos de diferentes colores están muy alejados entre sí
- **Alta compacidad (bajo SS_W):** Las canicas de cada color están muy juntas
- **Un buen CH significa:** Grupos claramente distinguibles internamente homogéneos

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Escenario:** Segmentar leads para estrategias diferenciadas de seguimiento

**Variables de segmentación:** Ingreso, Edad, Antigüedad Laboral (normalizadas)

| Número de Clusters | CH Index | Interpretación |
|--------------------|----------|----------------|
| 2 | 312.5 | Buena separación básica |
| 3 | 458.2 | **Mejor segmentación** |
| 4 | 421.8 | Comienza a fragmentar |
| 5 | 385.4 | Demasiados clusters |
| 6 | 342.1 | Clusters muy pequeños |

**Caracterización de los 3 clusters óptimos:**

| Cluster | n | Ingreso Prom | Edad Prom | Antig. Prom | Tasa Conv |
|---------|---|--------------|-----------|-------------|-----------|
| A: "Jóvenes Estables" | 350 | $18,500 | 28 años | 36 meses | 35.7% |
| B: "Maduros Premium" | 280 | $42,000 | 45 años | 120 meses | 48.2% |
| C: "Emergentes Riesgo" | 370 | $9,500 | 32 años | 8 meses | 14.3% |

**Análisis de varianza:**

```
BCSS (Varianza entre clusters): 485,230
WCSS (Varianza dentro de clusters): 1,052,480

CH = (485,230 / 2) / (1,052,480 / 997)
CH = 242,615 / 1,055.6
CH = 229.8 (para k=3)
```

### Interpretación

- **CH máximo en k=3:** Tres clusters es la segmentación óptima para estos datos
- **Aplicación en Galgo:**
  - **Cluster A:** Campaña de engagement digital, ofertas de entrada
  - **Cluster B:** Atención personalizada, productos premium
  - **Cluster C:** Precalificación estricta, remarketing cauteloso
- **Por qué importa CH en credit scoring:**
  - Segmentación para desarrollar scorecards específicos por segmento
  - Cada segmento puede tener diferentes drivers de riesgo
  - Mejora la precisión al adaptar el modelo a cada población
- **Limitaciones:**
  - Favorece clusters convexos y esféricos
  - Sensible a outliers
  - No hay valor absoluto "bueno"; comparar entre diferentes k

---

## 13.4.3 Gini Variance (Varianza del Gini)

### Definición Formal

La varianza del Gini cuantifica la incertidumbre en la estimación del coeficiente de Gini. Es crucial para:
1. Construir intervalos de confianza
2. Comparar estadísticamente dos modelos
3. Evaluar la estabilidad del modelo

**Fórmula aproximada (DeLong):**

$$Var(Gini) \approx 4 \times Var(AUC)$$

$$Var(AUC) = \frac{AUC(1-AUC) + (n_1 - 1)(Q_1 - AUC^2) + (n_0 - 1)(Q_2 - AUC^2)}{n_1 \times n_0}$$

Donde:
- $n_1$ = número de eventos (malos/no convertidos)
- $n_0$ = número de no eventos (buenos/convertidos)
- $Q_1 = AUC / (2 - AUC)$
- $Q_2 = 2 \times AUC^2 / (1 + AUC)$

**Error estándar del Gini:**
$$SE(Gini) = \sqrt{Var(Gini)}$$

**Intervalo de confianza (95%):**
$$IC_{95\%} = Gini \pm 1.96 \times SE(Gini)$$

### Analogía

Imagina que mides tu altura con una regla. Cada medición tendrá pequeñas variaciones por el ángulo, postura, etc. La varianza del Gini es como el "error de medición" de qué tan bueno es tu modelo. Un Gini de 0.50 con varianza pequeña es más confiable que un Gini de 0.55 con varianza grande. Te ayuda a saber si la diferencia entre dos modelos es "real" o solo "ruido".

### Ejemplo Aplicado: Conversión de Leads en Galgo

**Escenario:** Comparar el modelo actual vs. un modelo challenger

**Datos del modelo:**
- n = 1,000 leads
- Convertidos ($n_0$) = 285
- No convertidos ($n_1$) = 715

**Modelo Actual:**
- AUC = 0.76
- Gini = 0.52

**Cálculo de varianza:**

$$Q_1 = \frac{0.76}{2 - 0.76} = 0.613$$

$$Q_2 = \frac{2 \times 0.76^2}{1 + 0.76} = 0.656$$

$$Var(AUC) = \frac{0.76(1-0.76) + (714)(0.613 - 0.576) + (284)(0.656 - 0.576)}{715 \times 285}$$

$$Var(AUC) = \frac{0.182 + 26.42 + 22.72}{203,775} = 0.000242$$

$$SE(AUC) = \sqrt{0.000242} = 0.0156$$

$$SE(Gini) = 2 \times SE(AUC) = 0.0312$$

**Intervalo de confianza:**
$$IC_{95\%}(Gini) = 0.52 \pm 1.96 \times 0.0312 = [0.459, 0.581]$$

**Comparación con modelo challenger:**

| Métrica | Modelo Actual | Modelo Challenger | Diferencia |
|---------|---------------|-------------------|------------|
| Gini | 0.520 | 0.548 | 0.028 |
| SE(Gini) | 0.031 | 0.029 | - |
| IC 95% | [0.459, 0.581] | [0.491, 0.605] | - |

**Test de significancia:**

$$Z = \frac{Gini_{challenger} - Gini_{actual}}{\sqrt{SE^2_{challenger} + SE^2_{actual}}}$$

$$Z = \frac{0.548 - 0.520}{\sqrt{0.029^2 + 0.031^2}} = \frac{0.028}{0.042} = 0.67$$

p-valor = 0.503 (no significativo)

### Interpretación

- **IC 95% del Gini actual: [0.459, 0.581]:** El Gini "real" del modelo está en este rango con 95% de confianza
- **Comparación de modelos:**
  - Los intervalos de confianza se **superponen** considerablemente
  - Z = 0.67, p = 0.503 > 0.05
  - **No hay diferencia estadísticamente significativa** entre los modelos
- **Decisión de negocio:**
  - El modelo challenger no ofrece mejora significativa
  - Considerar otros factores: simplicidad, explicabilidad, costo de implementación
  - Si se mantiene el modelo actual, se evitan costos de migración
- **Importancia en validación:**
  - Siempre reportar Gini con su intervalo de confianza
  - Especialmente importante con muestras pequeñas (<1,000)
  - El regulador puede requerir demostrar significancia estadística de mejoras

---

# Resumen de Métricas

| Categoría | Métrica | Propósito | Rango/Threshold |
|-----------|---------|-----------|-----------------|
| **Transformación** | Rescale | Normalizar rangos | [0,1] o [-1,1] |
| | Discretize | Crear bins interpretables | 5-20 bins |
| **Característica** | IV | Poder predictivo | 0.1-0.5 ideal |
| | PSI | Estabilidad poblacional | <0.10 estable |
| | Chi-Square | Significancia de asociación | p<0.05 significativo |
| **Modelo** | Gini | Discriminación global | 0.40-0.60 bueno |
| | CAP/AR | Discriminación visual | AR = Gini |
| | Lift | Eficiencia de targeting | >2.0 en top decil |
| **Adicionales** | Deviance | Bondad de ajuste | Menor es mejor |
| | Calinski-Harabasz | Calidad de segmentación | Maximizar |
| | Gini Variance | Incertidumbre del Gini | Para IC y tests |

---

# Referencias

- Anderson, R. (2019). *Credit Intelligence and Modelling: Many Paths Through the Forest*. Oxford University Press.
- Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards*. Wiley.
- Engelmann, B., Hayden, E., & Tasche, D. (2003). *Measuring the Discriminative Power of Rating Systems*. Deutsche Bundesbank Discussion Paper.
- Yurdakul, B., & Naranjo, J. (2021). *Statistical Properties of the Population Stability Index*. Journal of Risk Model Validation.


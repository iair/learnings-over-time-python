# M5 · La fábrica declarativa de variables, con validador de ancla y diccionario auto-generado
# Serie: Modelador de Riesgo en Profundidad · Ejecutar: marimo edit M5_fabrica_declarativa.py
# Dependencias: marimo, numpy, pandas, matplotlib
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
        # M5 · La fábrica de variables como pipeline declarativo

        Arquitectura: **especificación** (lista de tuplas) + **motor** (una función por agregador)
        + **contratos** (validador de ancla, nombre generado, diccionario auto-producido).

        Incluye una variable-trampa con ventana ilegal para ver al validador rechazarla.
        """
    )
    return


@app.cell
def _(np):
    # ---------------- Panel sintético: cliente × mes ----------------
    RNG = np.random.default_rng(23)
    N, T, T0 = 5000, 24, 18            # t0 en el mes 18 (deja 6 meses "futuros" para la trampa)
    riesgo = RNG.beta(2, 8, N)
    antiguedad = RNG.integers(3, T0 + 1, N)   # meses de historia de cada cliente

    def panel(base, vol, trend=0.0):
        m = np.clip(RNG.normal(base[:, None] + trend * np.arange(T)[None, :], vol, (N, T)), 0, None)
        return m

    mora = np.zeros((N, T))
    for t in range(1, T):
        entra = RNG.uniform(size=N) < (0.02 + 0.22 * riesgo)
        persiste = (mora[:, t - 1] > 0) & (RNG.uniform(size=N) < 0.6 + 0.3 * riesgo)
        mora[:, t] = np.clip(np.where(persiste, mora[:, t - 1] + 30, np.where(entra, 30, 0)), 0, 180)

    METRICAS = {
        "dias_mora": mora,
        "uso_linea": np.clip(panel(0.30 + 0.5 * riesgo, 0.10), 0, 1.3),
        "deuda_total": panel(1_500_000 + 2_000_000 * riesgo, 220_000),
        "n_consultas": (RNG.poisson(0.3 + 2.2 * riesgo[:, None], (N, T))).astype(float),
        "abonos": panel(900_000 - 250_000 * riesgo, 130_000),
    }
    # borrar la historia previa a la fecha de alta de cada cliente (missing real)
    for nombre_m in METRICAS:
        Mx = METRICAS[nombre_m]
        for i in range(N):
            Mx[i, : T0 - antiguedad[i]] = np.nan

    LAGS = {"dias_mora": 1, "uso_linea": 1, "deuda_total": 1, "abonos": 1, "n_consultas": 2}
    return LAGS, METRICAS, N, RNG, T, T0, antiguedad, mora, riesgo


@app.cell
def _(np):
    # ---------------- Motor: agregadores como funciones puras ----------------
    def ag_prom(M, ini, fin):
        return np.nanmean(M[:, ini : fin + 1], axis=1)

    def ag_max(M, ini, fin):
        return np.nanmax(np.where(np.isnan(M[:, ini : fin + 1]), -np.inf, M[:, ini : fin + 1]), axis=1)

    def ag_suma(M, ini, fin):
        return np.nansum(M[:, ini : fin + 1], axis=1)

    def ag_delta(M, ini, fin):
        # necesita AMBOS extremos: si falta uno, missing ESTRUCTURAL (NaN)
        return M[:, fin] - M[:, ini]

    def ag_recencia(M, ini, fin, umbral=30):
        # meses desde el último mes con evento (>= umbral); sin evento -> 13 (censurado)
        ventana = M[:, ini : fin + 1] >= umbral
        k = ventana.shape[1]
        out = np.full(M.shape[0], 13.0)
        for j in range(k):                      # j=0 es el mes más antiguo
            col = ventana[:, j]
            out = np.where(col, k - j, out)     # meses desde el evento más reciente
        # si TODO el tramo era NaN, no hay información -> NaN (estructural)
        todo_nan = np.isnan(M[:, ini : fin + 1]).all(axis=1)
        return np.where(todo_nan, np.nan, out)

    AGREGADORES = {"prom": ag_prom, "max": ag_max, "suma": ag_suma,
                   "delta": ag_delta, "recencia": ag_recencia}
    return (AGREGADORES,)


@app.cell
def _(mo):
    mo.md(
        """
        ## La especificación (datos, no código)

        Cada tupla: `(concepto, métrica_fuente, agregador, ventana_meses, lag_override)`.
        El **nombre**, la **ventana exacta** y el **diccionario** se generan de aquí — es imposible
        que el nombre mienta sobre la ventana porque ambos salen de la misma tupla.
        """
    )
    return


@app.cell
def _(AGREGADORES, LAGS, METRICAS, T0, mo, np, pd):
    SPEC = [
        ("dias_mora",  "dias_mora",   "max",      12, None),
        ("dias_mora",  "dias_mora",   "max",      6,  None),
        ("meses_desde_mora", "dias_mora", "recencia", 12, None),
        ("uso_linea",  "uso_linea",   "prom",     6,  None),
        ("uso_linea",  "uso_linea",   "prom",     12, None),
        ("uso_linea",  "uso_linea",   "max",      12, None),
        ("deuda_total","deuda_total", "delta",    12, None),
        ("deuda_total","deuda_total", "delta",    6,  None),
        ("consultas",  "n_consultas", "suma",     3,  None),
        ("consultas",  "n_consultas", "suma",     6,  None),
        ("abonos",     "abonos",      "prom",     6,  None),
        # -------- VARIABLE-TRAMPA: ventana que cruza t0 (fin en t0+2) --------
        ("dias_mora_TRAMPA", "dias_mora", "max", 6, -2),  # lag negativo = mira el futuro (fin en t0+2)
    ]

    def construir(spec, t0=T0):
        matriz, diccionario, rechazadas = {}, [], []
        for concepto, metrica, agg, ventana, lag_override in spec:
            lag = LAGS[metrica] if lag_override is None else lag_override
            fin = t0 - lag                      # índice del último mes usado
            ini = fin - ventana + 1
            nombre = f"{concepto}_{agg}_{ventana}m"
            # ---------- CONTRATO DE ANCLA ----------
            if fin > t0 - 1:
                rechazadas.append((nombre, f"fin de ventana t0{fin - t0:+d} > t0-1: ILEGAL"))
                continue
            valores = AGREGADORES[agg](METRICAS[metrica], ini, fin)
            valores = np.where(np.isinf(valores), np.nan, valores)
            matriz[nombre] = valores
            diccionario.append({
                "variable": nombre, "familia": concepto, "agregador": agg,
                "ventana": f"[t0{ini - t0:+d}, t0{fin - t0:+d}]",
                "fuente": metrica, "lag_fuente": lag,
                "pct_missing": float(np.mean(np.isnan(valores))),
                "nota": "missing estructural esperado (historia < ventana)"
                        if agg in ("delta", "recencia") else "",
            })
        return pd.DataFrame(matriz), pd.DataFrame(diccionario), rechazadas

    X, DICCIONARIO, RECHAZADAS = construir(SPEC)

    rech = "\n".join(f"- 🚨 `{n}` — {motivo}" for n, motivo in RECHAZADAS) or "(ninguna)"
    mo.md(
        f"""
        ## Resultado de la construcción

        Variables construidas: **{X.shape[1]}** · Variables **rechazadas por el validador de ancla**:

        {rech}

        ### Diccionario auto-generado (primeras filas)

        {DICCIONARIO.head(12).round(3).to_markdown(index=False)}
        """
    )
    return DICCIONARIO, X


@app.cell
def _(mo):
    antig_min = mo.ui.slider(start=3, stop=12, step=1, value=6,
                             label="Antigüedad mínima exigida (meses)")
    antig_min
    return (antig_min,)


@app.cell
def _(X, antig_min, antiguedad, mo, np, plt):
    # ---------------- Trade-off: población incluida vs missing estructural ----------------
    umbrales = list(range(3, 13))
    pobl, miss = [], []
    delta_col = "deuda_total_delta_12m"
    for u in umbrales:
        inc = antiguedad >= u
        pobl.append(inc.mean() * 100)
        miss.append(np.mean(np.isnan(X[delta_col].to_numpy()[inc])) * 100)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(umbrales, pobl, marker="o", color="#1f4e9c", label="% población incluida")
    ax1.set_xlabel("antigüedad mínima exigida (meses)")
    ax1.set_ylabel("% población incluida", color="#1f4e9c")
    ax2 = ax1.twinx()
    ax2.plot(umbrales, miss, marker="s", color="#b03030", label="% missing estructural en Δ12m")
    ax2.set_ylabel("% missing estructural Δ12m", color="#b03030")
    ax1.axvline(antig_min.value, color="gray", ls="--")
    ax1.set_title("El trade-off de la antigüedad mínima (línea gris = tu slider)")
    ax1.grid(alpha=0.3)
    fig
    return


@app.cell
def _(antig_min, antiguedad, mo, np):
    incl = antiguedad >= antig_min.value
    mo.md(
        f"""
        Con antigüedad mínima **{antig_min.value} meses**: población incluida
        **{incl.mean():.1%}**. Subir el requisito limpia el missing estructural pero recorta
        población — la decisión de C1 (≥6 meses) es este trade-off, elegido y documentado.

        ### Ejercicios
        1. Agrega el agregador `volatilidad` (desviación estándar en la ventana) y el ratio
           corto/largo `uso_prom_3m / uso_prom_12m` como variables derivadas de la matriz.
        2. Agrega al diccionario la columna `disponible_en_produccion` (bool) calculada desde el
           lag de cada fuente y un lag de producción configurable.
        3. Conecta la test-suite de M4: T1 ya está implementado como contrato; agrega T5
           (lista negra) y T6 (IV > 0.5 exige auditoría) tras el screening de M7.
        """
    )
    return


if __name__ == "__main__":
    app.run()

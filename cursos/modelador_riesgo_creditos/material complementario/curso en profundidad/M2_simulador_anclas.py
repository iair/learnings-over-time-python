# M2 · Simulador de anclas temporales — Serie: Modelador de Riesgo en Profundidad
# Ejecutar con:  marimo edit M2_simulador_anclas.py   (o `marimo run` para modo app)
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
        # M2 · Simulador de anclas: cuánto «paga» cruzar la línea de t₀

        Este notebook genera una cartera sintética con dinámica realista y mide el **IV** de una
        variable de mora según dónde termina su ventana de cálculo. La regla del curso: la ventana
        legal es **[t₀−k, t₀−1]**. Aquí puedes violarla a propósito y ver el precio.

        La dinámica sintética incluye tres ingredientes del mundo real:
        1. La mora es **persistente** (quien está en mora tiende a seguir).
        2. Los futuros malos se **deterioran gradualmente** después de t₀.
        3. El banco **reacciona**: recorta el cupo del que se deteriora (la variable-trampa Δcupo).
        """
    )
    return


@app.cell
def _(np):
    # ----------------- Generador sintético autocontenido -----------------
    RNG = np.random.default_rng(42)
    N = 8000          # solicitudes
    T = 30            # meses de panel: t0 está en el mes 12; quedan 17 meses hacia adelante
    T0 = 12

    # Riesgo latente de cada cliente
    riesgo = RNG.beta(2, 9, N)                     # media ~0.18
    es_malo = (RNG.uniform(size=N) < riesgo * 0.45).astype(int)  # ~8% de malos

    # Panel de dias de mora: proceso persistente + deterioro post-t0 de los malos
    mora = np.zeros((N, T))
    for t in range(1, T):
        base = 0.03 + 0.25 * riesgo                          # prob de entrar en mora
        # el deterioro de los malos se acelera después de t0
        acel = np.where((es_malo == 1) & (t > T0), 0.10 + 0.02 * (t - T0), 0.0)
        entra = RNG.uniform(size=N) < (base + acel)
        persiste = (mora[:, t - 1] > 0) & (RNG.uniform(size=N) < 0.55 + 0.35 * es_malo)
        mora[:, t] = np.where(persiste, mora[:, t - 1] + 30, np.where(entra, 30, 0))
    mora = np.clip(mora, 0, 180)

    # Cupo: parte en ~2M y el banco lo recorta cuando ve mora reciente (reacción institucional)
    cupo = np.full((N, T), 2_000_000.0)
    for t in range(1, T):
        recorta = mora[:, t - 1] >= 30
        cupo[:, t] = np.where(recorta, cupo[:, t - 1] * 0.75, cupo[:, t - 1] * 1.002)

    # Target honesto: 90+ en (t0, t0+12]
    peor_futuro = mora[:, T0 + 1 : T0 + 13].max(axis=1)
    target = (peor_futuro >= 90).astype(int)
    return N, T, T0, cupo, mora, target


@app.cell
def _(np):
    # ----------------- IV con binning por cuantiles (5 bins) -----------------
    def iv_de(x, y, bins=5):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        ok = ~np.isnan(x)
        x, y = x[ok], y[ok]
        # cortes por cuantiles (colapsa duplicados: variables con masa puntual)
        qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        if len(qs) < 3:
            qs = np.array([x.min() - 1, np.median(x), x.max() + 1])
        idx = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
        iv = 0.0
        nb, ng = max(y.sum(), 1), max((1 - y).sum(), 1)
        for b in range(len(qs) - 1):
            m = idx == b
            if m.sum() == 0:
                continue
            pb = max(y[m].sum(), 0.5) / nb          # % malos del bin (suavizado 0.5)
            pg = max((1 - y[m]).sum(), 0.5) / ng    # % buenos del bin
            iv += (pg - pb) * np.log(pg / pb)
        return iv
    return (iv_de,)


@app.cell
def _(mo):
    ancla = mo.ui.slider(
        start=-3, stop=12, step=1, value=-1,
        label="Fin de la ventana de `dias_mora_max_6m` (meses respecto de t₀; legal: ≤ −1)",
    )
    k_cupo = mo.ui.slider(
        start=1, stop=12, step=1, value=3,
        label="Variable-trampa: Δ cupo a t₀+k vs t₀−1 (k meses vista)",
    )
    mo.vstack([ancla, k_cupo])
    return ancla, k_cupo


@app.cell
def _(T0, ancla, cupo, iv_de, k_cupo, mo, mora, np, target):
    # Variable de mora con ancla movil: ventana de 6 meses que TERMINA en t0 + desplazamiento
    fin = T0 + ancla.value
    ini = fin - 5
    var_mora = mora[:, max(ini, 0) : fin + 1].max(axis=1)
    iv_mora = iv_de(var_mora, target)

    # Variable-trampa: delta de cupo mirando k meses despues de t0
    var_cupo = cupo[:, T0 + k_cupo.value] - cupo[:, T0 - 1]
    iv_cupo = iv_de(var_cupo, target)

    legal = "✅ LEGAL (≤ t₀−1)" if ancla.value <= -1 else "🚨 FUGA: la ventana cruza t₀"
    mo.md(
        f"""
        ### Resultado con los sliders actuales

        | Variable | Ventana | IV | Estado |
        |---|---|---|---|
        | `dias_mora_max_6m` | [t₀{ini - T0:+d}, t₀{ancla.value:+d}] | **{iv_mora:.2f}** | {legal} |
        | `delta_cupo` | t₀+{k_cupo.value} vs t₀−1 | **{iv_cupo:.2f}** | 🚨 siempre fuga: mide la reacción del banco |
        """
    )
    return


@app.cell
def _(T0, cupo, iv_de, mora, np, plt, target):
    # ----------------- Curva completa: IV vs posicion del ancla -----------------
    desplazamientos = list(range(-3, 13))
    ivs = []
    for d in desplazamientos:
        fin_d = T0 + d
        var = mora[:, max(fin_d - 5, 0) : fin_d + 1].max(axis=1)
        ivs.append(iv_de(var, target))

    ks = list(range(1, 13))
    ivs_cupo = [iv_de(cupo[:, T0 + k] - cupo[:, T0 - 1], target) for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.axvspan(-3.5, -0.5, color="#DDEBF7", label="zona legal")
    ax1.plot(desplazamientos, ivs, marker="o", color="#1f4e9c")
    ax1.axvline(-1, color="green", ls="--", lw=1)
    ax1.set_title("IV de dias_mora_max_6m según el fin de su ventana")
    ax1.set_xlabel("fin de la ventana (meses respecto de t₀)")
    ax1.set_ylabel("IV")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar([str(k) for k in ks], ivs_cupo, color="#b03030")
    ax2.set_title("IV de Δcupo t₀+k vs t₀−1 (desenlace disfrazado)")
    ax2.set_xlabel("k (meses después de t₀)")
    ax2.set_ylabel("IV")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Lecturas

        1. **Meseta honesta:** con el ancla en la zona legal el IV es estable — es la señal real
           de la mora pasada.
        2. **Explosión al cruzar t₀:** un solo mes de contaminación sube el IV de forma
           discontinua; la variable «ve» el inicio del deterioro que define al target.
        3. **La variable de reacción del banco** crece con k hasta volverse casi el target mismo:
           el banco recortó el cupo PORQUE el cliente cayó. En producción no existe.

        **Ejercicio:** modifica el generador para que el banco reaccione con 2 meses de rezago y
        verifica que la curva roja se desplaza. Después escribe el assert que rechazaría ambas
        variables ilegales en un pipeline real (pista: cada variable debe declarar su ventana y
        el assert compara `fin_ventana <= t0 - lag_fuente`).
        """
    )
    return


if __name__ == "__main__":
    app.run()

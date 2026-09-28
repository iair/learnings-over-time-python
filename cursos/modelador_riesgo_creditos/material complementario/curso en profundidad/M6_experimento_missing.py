# M6 · El experimento del daño medido: imputar vs bin propio en missing MNAR
# Serie: Modelador de Riesgo en Profundidad · Ejecutar: marimo edit M6_experimento_missing.py
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
        # M6 · Cuánta señal destruye una imputación bienintencionada

        Generamos una variable de **renta** cuyo missing es MNAR: se concentra en un segmento
        (independientes) que además es más riesgoso — el caso de la lámina 16 de C2. Después
        comparamos cuatro tratamientos midiendo el **IV** de la variable binneada y la forma de
        la distribución. El slider controla qué tan informativo es el missing.
        """
    )
    return


@app.cell
def _(mo):
    intensidad = mo.ui.slider(
        start=0.0, stop=3.0, step=0.25, value=1.5,
        label="Intensidad MNAR: cuánto más riesgosos son los que NO declaran (0 = MCAR)",
    )
    intensidad
    return (intensidad,)


@app.cell
def _(intensidad, np):
    # ---------------- Generador ----------------
    RNG = np.random.default_rng(31)
    N = 20000
    independiente = RNG.uniform(size=N) < 0.30
    # renta verdadera (log-normal); independientes algo más volátiles
    renta = np.exp(RNG.normal(13.6, 0.5, N))
    # riesgo: sube con baja renta, con ser independiente, y con un factor latente
    lat = RNG.normal(size=N)
    logit = -2.4 - 0.9 * (np.log(renta) - 13.6) + 0.5 * independiente + 0.6 * lat
    pd_true = 1 / (1 + np.exp(-logit))
    target = (RNG.uniform(size=N) < pd_true).astype(int)

    # MNAR: no declara con prob mayor si es independiente Y si su riesgo latente es alto
    p_miss = 0.05 + 0.28 * independiente + 0.04 * intensidad.value * np.clip(lat, 0, None)
    missing = RNG.uniform(size=N) < np.clip(p_miss, 0, 0.9)
    renta_obs = np.where(missing, np.nan, renta)
    return N, RNG, independiente, missing, renta, renta_obs, target


@app.cell
def _(np):
    # ---------------- WoE / IV con bins por cuantiles + bin MISSING opcional ----------------
    def woe_iv(x, y, bins=5, bin_missing=True):
        x = np.asarray(x, dtype=float); y = np.asarray(y)
        nb, ng = max(y.sum(), 1), max((1 - y).sum(), 1)
        iv, filas = 0.0, []
        nanmask = np.isnan(x)

        def aporte(mask, etiqueta):
            nonlocal iv
            if mask.sum() == 0:
                return
            pb = max(y[mask].sum(), 0.5) / nb
            pg = max((1 - y[mask]).sum(), 0.5) / ng
            w = np.log(pg / pb)
            iv += (pg - pb) * w
            filas.append((etiqueta, int(mask.sum()), float(y[mask].mean()), float(w)))

        if bin_missing and nanmask.any():
            aporte(nanmask, "MISSING")
        xv = x[~nanmask]
        qs = np.unique(np.quantile(xv, np.linspace(0, 1, bins + 1)))
        idx = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
        for b in range(len(qs) - 1):
            aporte((~nanmask) & (idx == b), f"[{qs[b]:,.0f}, {qs[b+1]:,.0f}]")
        return iv, filas
    return (woe_iv,)


@app.cell
def _(independiente, missing, mo, np, renta, renta_obs, target, woe_iv):
    # ---------------- Los cuatro tratamientos ----------------
    media = np.nanmean(renta_obs)
    mediana = np.nanmedian(renta_obs)

    t1 = renta_obs                                            # bin MISSING propio
    t2 = np.where(np.isnan(renta_obs), media, renta_obs)      # media global
    t3 = np.where(np.isnan(renta_obs), mediana, renta_obs)    # mediana global
    # "inteligente": media por segmento
    m_dep = np.nanmean(renta_obs[~independiente]); m_ind = np.nanmean(renta_obs[independiente])
    t4 = np.where(np.isnan(renta_obs), np.where(independiente, m_ind, m_dep), renta_obs)

    iv1, filas1 = woe_iv(t1, target, bin_missing=True)
    iv2, _ = woe_iv(t2, target, bin_missing=False)
    iv3, _ = woe_iv(t3, target, bin_missing=False)
    iv4, _ = woe_iv(t4, target, bin_missing=False)

    tasa_miss = target[missing].mean()
    tasa_no = target[~missing].mean()

    tabla_bins = "\n".join(
        f"| {e} | {n:,} | {tm:.1%} | {w:+.2f} |" for e, n, tm, w in filas1
    )
    mo.md(
        f"""
        ### Diagnóstico del missing

        % missing: **{missing.mean():.1%}** · tasa de malos de los MISSING: **{tasa_miss:.1%}**
        vs no-missing: **{tasa_no:.1%}** → la ausencia informa (MNAR).

        ### IV según tratamiento

        | Tratamiento | IV |
        |---|---|
        | **Bin MISSING propio** | **{iv1:.3f}** |
        | Imputar media global | {iv2:.3f} |
        | Imputar mediana | {iv3:.3f} |
        | Imputar media por segmento | {iv4:.3f} |

        ### Tabla WoE del tratamiento correcto

        | Bin | n | tasa de malos | WoE |
        |---|---|---|---|
        {tabla_bins}

        Nota el WoE del bin MISSING: esa es la señal que la imputación borra. Sube el slider
        de intensidad y mira la brecha de IV crecer.
        """
    )
    return media, t1, t2


@app.cell
def _(media, np, plt, renta_obs, t2):
    # ---------------- La deformación de la distribución ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    ax1.hist(renta_obs[~np.isnan(renta_obs)], bins=60, color="#1f4e9c")
    ax1.set_title("Distribución observada (missing aparte)")
    ax2.hist(t2, bins=60, color="#b03030")
    ax2.axvline(media, color="black", ls="--", lw=1)
    ax2.set_title("Tras imputar por la media: el pico artificial")
    for ax in (ax1, ax2):
        ax.set_xlabel("renta"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo, np):
    mo.md("## Detector de centinelas por masa puntual")
    return


@app.cell
def _(RNG, mo, np, pd):
    # ---------------- Centinelas plantados y su detección ----------------
    n = 15000
    deuda = np.exp(RNG.normal(14.2, 0.7, n))
    idx9 = RNG.choice(n, int(0.06 * n), replace=False)
    deuda[idx9] = 999999          # centinela clásico
    consultas = RNG.poisson(1.2, n).astype(float)
    idxm1 = RNG.choice(n, int(0.04 * n), replace=False)
    consultas[idxm1] = -1         # otro centinela

    def masa_puntual(x, top=3):
        s = pd.Series(x).value_counts(normalize=True).head(top)
        return [(v, f"{p:.1%}") for v, p in s.items()]

    def chequeo(nombre, x):
        x = np.asarray(x, float)
        return {
            "variable": nombre, "n": len(x), "% missing": f"{np.isnan(x).mean():.1%}",
            "min": round(np.nanmin(x), 1), "p1": round(np.nanpercentile(x, 1), 1),
            "mediana": round(np.nanmedian(x), 1), "p99": round(np.nanpercentile(x, 99), 1),
            "max": round(np.nanmax(x), 1),
            "masa puntual (top)": str(masa_puntual(x, 2)),
        }

    reporte = pd.DataFrame([chequeo("deuda_total", deuda), chequeo("n_consultas_6m", consultas)])
    mo.md(
        f"""
        El chequeo mínimo por variable, auto-generado. Las alertas saltan a la vista: masa
        puntual anómala en un valor exacto (999999, −1) y máximos fuera del rango de negocio.

        {reporte.to_markdown(index=False)}

        **Ejercicios:** (1) convierte los centinelas a NaN y vuelve a diagnosticar el missing
        resultante: ¿MCAR o MNAR? (aquí fue plantado al azar: compara la tasa de malos);
        (2) agrega la regla «masa puntual > 3% en un valor que no es 0 → alerta» al reporte de
        la fábrica de M5; (3) planta un centinela en 0 sobre una variable donde 0 es legítimo y
        diseña cómo distinguirlos (pista: cruzar contra tenencia del producto).
        """
    )
    return


if __name__ == "__main__":
    app.run()

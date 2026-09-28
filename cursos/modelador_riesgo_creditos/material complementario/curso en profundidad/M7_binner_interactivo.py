# M7 · Binner interactivo: fine→coarse, WoE/IV en vivo y las tres trampas reproducidas
# Serie: Modelador de Riesgo en Profundidad · Ejecutar: marimo edit M7_binner_interactivo.py
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
        # M7 · Binning, WoE e IV con las manos en la masa

        Tres piezas: **(1)** binner fine→coarse sobre una variable realista, con tabla WoE, IV y
        chequeo de monotonicidad en vivo; **(2)** Trampa 1: el IV fantasma de una variable
        aleatoria crece con el número de bins... solo en DEV; **(3)** Trampa 2: el IV de la misma
        variable con su intervalo bootstrap según cuántos malos hay.

        Convención del curso: WoE = ln(%buenos/%malos) → **WoE alto = bin bueno**.
        """
    )
    return


@app.cell
def _(np):
    RNG = np.random.default_rng(5)

    def woe_tabla(x, y, cortes, suavizado=0.5):
        """Tabla WoE/IV dado un vector x (con NaN), target y, y lista de cortes."""
        x = np.asarray(x, float); y = np.asarray(y)
        nb, ng = max(y.sum(), 1), max((1 - y).sum(), 1)
        filas, iv = [], 0.0
        nan = np.isnan(x)
        grupos = [("MISSING", nan)] if nan.any() else []
        idx = np.clip(np.searchsorted(cortes, x, side="right") - 1, 0, len(cortes) - 2)
        for b in range(len(cortes) - 1):
            grupos.append((f"[{cortes[b]:.2f}, {cortes[b+1]:.2f})", (~nan) & (idx == b)))
        for etiqueta, m in grupos:
            if m.sum() == 0:
                continue
            pb = (y[m].sum() + suavizado) / nb
            pg = ((1 - y[m]).sum() + suavizado) / ng
            w = float(np.log(pg / pb))
            aporte = (pg - pb) * w
            iv += aporte
            filas.append({"bin": etiqueta, "n": int(m.sum()), "% muestra": m.mean(),
                          "tasa de malos": float(y[m].mean()), "WoE": w, "aporte IV": aporte})
        return filas, float(iv)

    def monotono(filas):
        ws = [f["WoE"] for f in filas if f["bin"] != "MISSING"]
        if len(ws) < 3:
            return True
        d = np.diff(ws)
        return bool(np.all(d <= 1e-12) or np.all(d >= -1e-12))
    return RNG, monotono, woe_tabla


@app.cell
def _(RNG, np):
    # ---------------- Variable sintética realista: utilización de línea ----------------
    N = 15000
    riesgo = RNG.beta(2, 8, N)
    uso = np.clip(RNG.normal(0.25 + 0.55 * riesgo, 0.13), 0, 1.25)
    uso[RNG.choice(N, int(0.05 * N), replace=False)] = np.nan   # 5% missing
    pd_true = 1 / (1 + np.exp(-(-3.1 + 2.6 * np.nan_to_num(uso, nan=0.45) + 0.8 * riesgo)))
    y_all = (RNG.uniform(size=N) < pd_true).astype(int)
    # DEV / HO por cliente (aquí 1 fila = 1 cliente)
    perm = RNG.permutation(N)
    DEV, HO = perm[: int(0.7 * N)], perm[int(0.7 * N):]
    return DEV, HO, N, uso, y_all


@app.cell
def _(mo):
    n_finos = mo.ui.slider(start=4, stop=20, step=2, value=10, label="Bins finos (cuantiles)")
    fusion = mo.ui.slider(start=1, stop=4, step=1, value=2,
                          label="Factor de fusión coarse (agrupa cada k bins finos)")
    mo.vstack([n_finos, fusion])
    return fusion, n_finos


@app.cell
def _(DEV, HO, fusion, mo, monotono, n_finos, np, pd, uso, woe_tabla, y_all):
    xv = uso[DEV]; yv = y_all[DEV]
    qs = np.unique(np.nanquantile(xv, np.linspace(0, 1, n_finos.value + 1)))
    # coarse: tomar 1 de cada k cortes (siempre conservando extremos)
    cortes = list(qs[:: fusion.value])
    if cortes[-1] != qs[-1]:
        cortes.append(qs[-1])
    cortes[0], cortes[-1] = -np.inf, np.inf

    filas_dev, iv_dev = woe_tabla(xv, yv, np.array(cortes))
    filas_ho, iv_ho = woe_tabla(uso[HO], y_all[HO], np.array(cortes))
    ok_mono = monotono(filas_dev)
    masa_min = min(f["% muestra"] for f in filas_dev)

    tabla = pd.DataFrame(filas_dev).round(3)
    mo.md(
        f"""
        ## 1 · Binner fine→coarse: `uso_linea` (5% missing)

        Bins finales: **{len(filas_dev)}** · IV DEV: **{iv_dev:.3f}** · IV HO: **{iv_ho:.3f}** ·
        Monótono (sin contar MISSING): {"✅" if ok_mono else "🚨 zigzag: ¿ruido o forma real?"} ·
        Masa mínima por bin: {masa_min:.1%} {"✅" if masa_min >= 0.05 else "🚨 < 5%"}

        {tabla.to_markdown(index=False)}

        Sube la fusión y observa: el IV baja un poco, la monotonía y la masa mejoran — la
        fusión es regularización. Si el IV se desploma al fusionar, había estructura real.
        """
    )
    return


@app.cell
def _(mo):
    niveles = mo.ui.slider(start=4, stop=64, step=4, value=32,
                           label="Trampa 1 · Niveles de la variable ALEATORIA")
    niveles
    return (niveles,)


@app.cell
def _(DEV, HO, RNG, mo, niveles, np, woe_tabla, y_all):
    # ---------------- Trampa 1: variable aleatoria con k niveles ----------------
    z = RNG.integers(0, niveles.value, len(y_all)).astype(float)
    cortes_z = np.array([-np.inf] + [i + 0.5 for i in range(niveles.value - 1)] + [np.inf])
    _, iv_dev_z = woe_tabla(z[DEV], y_all[DEV], cortes_z)
    _, iv_ho_z = woe_tabla(z[HO], y_all[HO], cortes_z)
    mo.md(
        f"""
        ## 2 · Trampa 1: el IV premia el número de bins

        Variable **100% aleatoria** con {niveles.value} niveles:
        IV en DEV = **{iv_dev_z:.3f}** (¡"{'fuerte' if iv_dev_z > 0.3 else 'media' if iv_dev_z > 0.1 else 'débil'}" según la tabla!)
        · IV en HO = **{iv_ho_z:.3f}** → el fantasma muere fuera de muestra.

        Mueve el slider: el IV fantasma crece ~linealmente con los niveles. Defensas: masa
        mínima por bin, comparar a igual número de bins, y SIEMPRE contrastar en HO.
        """
    )
    return


@app.cell
def _(mo):
    n_malos_ui = mo.ui.slider(start=80, stop=2000, step=80, value=160,
                              label="Trampa 2 · Número de malos en la muestra")
    n_malos_ui
    return (n_malos_ui,)


@app.cell
def _(RNG, mo, n_malos_ui, np, plt, woe_tabla):
    # ---------------- Trampa 2: inestabilidad del IV con pocos malos + bootstrap ----------------
    n_tot = 12000
    n_malos = n_malos_ui.value
    riesgo2 = RNG.beta(2, 8, n_tot)
    x2 = np.clip(RNG.normal(0.3 + 0.5 * riesgo2, 0.12), 0, 1.2)
    # forzar exactamente n_malos (los de mayor riesgo tienen más chance)
    p = riesgo2 / riesgo2.sum()
    idx_malos = RNG.choice(n_tot, n_malos, replace=False, p=p)
    y2 = np.zeros(n_tot, int); y2[idx_malos] = 1

    cortes2 = np.array([-np.inf, 0.25, 0.45, 0.65, 0.85, np.inf])
    _, iv_punto = woe_tabla(x2, y2, cortes2)

    ivs_boot = []
    for _ in range(200):
        bidx = RNG.integers(0, n_tot, n_tot)
        _, ivb = woe_tabla(x2[bidx], y2[bidx], cortes2)
        ivs_boot.append(ivb)
    lo, hi = np.percentile(ivs_boot, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(ivs_boot, bins=30, color="#1f4e9c")
    ax.axvline(iv_punto, color="crimson", ls="--", label=f"IV puntual {iv_punto:.2f}")
    ax.set_title(f"Bootstrap del IV con {n_malos} malos · IC95% [{lo:.2f}, {hi:.2f}]")
    ax.set_xlabel("IV"); ax.legend(); ax.grid(alpha=0.3)

    mo.vstack([
        mo.md(
            f"""
            ## 3 · Trampa 2: el IV cuelga del conteo de malos

            Con **{n_malos} malos**: IV = **{iv_punto:.2f}**, IC95% bootstrap
            **[{lo:.2f}, {hi:.2f}]**. Mueve el slider hacia 2.000 malos y mira el intervalo
            angostarse. Reportar el intervalo (E3) cambia la conversación: un IV 0.85
            [0.55, 1.20] no es «casi 1», es «entre medio y fuerte, con suerte».
            """
        ),
        fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Ejercicios

        1. Implementa la restricción «mínimo 25 malos por bin» en el coarse y compárala con la
           de masa 5%: ¿cuál protege mejor el IV en HO?
        2. Agrega una variable con forma en U real (edad sintética) y verifica que fusionarla
           hasta la monotonía SÍ destruye IV — el test práctico de la excepción legítima.
        3. Reproduce la Trampa 3 trayendo la variable `mora_max_6m_corrida` del laboratorio de
           M4 y pásala por este binner: ¿dónde queda su IV y qué lista del screening la recibe?
        """
    )
    return


if __name__ == "__main__":
    app.run()

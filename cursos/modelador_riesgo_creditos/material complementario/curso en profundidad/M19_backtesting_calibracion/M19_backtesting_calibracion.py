# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
#     "statsmodels",
#     "scikit-learn",
# ]
# ///
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy import stats, integrate
    from scipy.optimize import brentq
    from scipy.special import betainc
    from sklearn.metrics import brier_score_loss
    from statsmodels.stats.multitest import multipletests
    return (
        betainc,
        brentq,
        brier_score_loss,
        integrate,
        mo,
        multipletests,
        plt,
        sm,
        stats,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # M19 · Backtesting de calibración: binomial, Hosmer-Lemeshow y más

    **Serie 2 · Del embudo al gobierno.** Notebook de `M19_backtesting_calibracion.md`.

    La calibración (M15) es una promesa: «en esta banda caerá el 1,42%». El backtesting la contrasta con
    lo que pasó. Este notebook arma cada test **dos veces** (numpy desde cero y la librería estándar),
    reproduce los números de Banco Austral de la clase 5 y muestra, con verdad conocida, **cuándo
    mienten los tests**:

    1. Los datos de la lámina (8 bandas OOT y los 10 grupos del Hosmer-Lemeshow).
    2. Binomial exacto: una cola, dos colas (tres definiciones) y la aproximación normal.
    3. Potencia: qué desvío se puede detectar con el n de cada banda.
    4. Correlación de defaults (Vasicek): el binomial rechaza de más; test ajustado y semáforo de Tasche.
    5. Hosmer-Lemeshow: construcción, $\chi^2_{g-2}$ vs $\chi^2_g$, p simulado y dependencia del agrupamiento.
    6. Spiegelhalter, Jeffreys (BCE) y test de pendiente de calibración.
    7. Batería completa sobre una cartera sintética con verdad conocida.
    8. Multiplicidad: Bonferroni, Holm, BH y la lectura por patrón (test de signos).
    9. Checks del módulo.

    Convenciones: target **1 = malo**; semáforo del curso p ≥ 0,05 🟢 · 0,01–0,05 🟡 · < 0,01 🔴.
    """)
    return


@app.cell
def _():
    # ===========================================================================
    # Código común de la serie
    # ===========================================================================
    # ===========================================================================
    # CÓDIGO COMÚN DE LA SERIE 2 — pegar VERBATIM en la celda "comun" de cada
    # notebook Marimo (los notebooks son autocontenidos: no se importa entre ellos).
    # Solo depende de numpy y pandas.
    # ===========================================================================
    import numpy as np
    import pandas as pd


    def _sigmoide(z):
        return 1.0 / (1.0 + np.exp(-z))


    def generar_cartera(n=24_000, semilla=2026, deterioro=0.35, drift_canal=True):
        """Cartera sintética «Banco Sintético» con VERDAD CONOCIDA.

        - 24 cohortes mensuales (2023-07 … 2025-06) + 6 de TTD (2025-07 … 2025-12).
        - Target 1 = malo (90+ a 12 meses). `pd_verdadera` es la PD real del
          generador: permite medir calibración y sesgos contra la verdad, cosa que
          en datos reales nunca se puede.
        - Muestras como el curso: DEV (≤ 2024-12, 70% al azar) · HO (resto de esos
          meses) · OOT (2025-01 … 2025-06) · TTD (sin desempeño, malo = NaN).
        - `deterioro` sube el log-odds de las cohortes 2025 (macro plantada).
        - `drift_canal`: la mezcla de `canal` cambia en OOT/TTD (drift categórico).
        - `meses_desde_mora` usa 13 = «sin mora en 12m» (como el curso) y trae dos
          códigos especiales: -9 (nunca tuvo mora registrada) y -99 (sin bureau).
        """
        rng = np.random.default_rng(semilla)
        meses_hist = pd.period_range("2023-07", "2025-06", freq="M")
        meses_ttd = pd.period_range("2025-07", "2025-12", freq="M")
        meses = meses_hist.append(meses_ttd)
        k = len(meses)
        t = rng.integers(0, k, size=n)
        t_frac = t / (k - 1)

        antig = np.clip(rng.gamma(2.2, 30, n), 6, 360).round()
        edad = np.clip(rng.normal(41, 11, n) + antig / 40, 19, 80).round()
        renta = np.exp(rng.normal(13.6, 0.55, n)) / 1e6            # MM$
        z_uso = rng.normal(0, 1, n)                                  # factor latente de presión
        uso_linea = np.clip(_sigmoide(z_uso * 1.1 - 0.6 + 0.3 * t_frac), 0, 1.2)
        uso_tc_12m = np.clip(_sigmoide(0.8 * z_uso + rng.normal(0, 0.6, n) - 0.5), 0, 1.2)
        uso_tc_3m = np.clip(uso_tc_12m + rng.normal(0, 0.07, n) + 0.04 * (z_uso > 1), 0, 1.3)
        deuda_otras = np.round(np.exp(rng.normal(0.8, 1.0, n)) * (rng.random(n) < 0.82), 3)
        carga = np.round((deuda_otras + uso_linea * 2.5) / np.maximum(renta, 0.2), 3)
        consultas_6m = rng.poisson(np.exp(0.2 + 0.5 * z_uso.clip(-2, 2)))

        # mora propia: recencia en meses (1..12) o 13 = sin mora en 12m
        p_mora = _sigmoide(-1.6 + 0.9 * z_uso)
        tuvo = rng.random(n) < p_mora
        meses_desde_mora = np.where(tuvo, rng.integers(1, 13, n), 13).astype(float)

        canal_niveles = np.array(["sucursal", "web", "app", "fuerza_venta"])
        p_base = np.array([0.45, 0.25, 0.15, 0.15])
        p_nuevo = np.array([0.25, 0.25, 0.35, 0.15])
        tardio = meses[t] >= pd.Period("2025-01", "M") if drift_canal else np.zeros(n, bool)
        canal = np.where(tardio,
                         rng.choice(canal_niveles, n, p=p_nuevo),
                         rng.choice(canal_niveles, n, p=p_base))

        # --- log-odds verdadero (efectos no lineales a propósito) ---
        efecto_mora = np.where(meses_desde_mora == 13, -0.55,
                               np.interp(meses_desde_mora, [1, 3, 6, 12], [1.4, 0.9, 0.4, 0.0]))
        eta = (-3.45
               + 2.4 * (uso_linea - 0.35)
               + 1.3 * np.maximum(uso_tc_12m - 0.4, 0)
               + 0.9 * np.maximum(uso_tc_3m - uso_tc_12m, 0) * 4
               + efecto_mora
               - 0.009 * (antig - 60)
               + 0.45 * np.log1p(carga)
               - 0.50 * (deuda_otras > 5)
               + 0.12 * np.minimum(consultas_6m, 6)
               + np.select([canal == "app", canal == "web", canal == "fuerza_venta"],
                           [0.30, 0.15, 0.25], 0.0)
               + deterioro * (meses[t] >= pd.Period("2025-01", "M"))
               + rng.normal(0, 0.35, n))                              # heterogeneidad no observada
        pd_verdadera = _sigmoide(eta)
        malo = (rng.random(n) < pd_verdadera).astype(float)

        # códigos especiales en meses_desde_mora (se asignan DESPUÉS del target:
        # -99 = sin bureau → a propósito con más riesgo real; -9 = nunca tuvo mora)
        sin_bureau = rng.random(n) < 0.03
        nunca = (~tuvo) & (rng.random(n) < 0.25)
        malo = np.where(sin_bureau & (rng.random(n) < 0.10), 1.0, malo)
        meses_desde_mora = np.where(nunca, -9.0, meses_desde_mora)
        meses_desde_mora = np.where(sin_bureau, -99.0, meses_desde_mora)
        renta = np.where(rng.random(n) < 0.04, np.nan, renta)       # missing de captura

        df = pd.DataFrame({
            "id": [f"S{i:06d}" for i in range(n)],
            "cohorte": meses[t].astype(str),
            "uso_linea_prom_12m": uso_linea.round(4),
            "uso_tc_prom_12m": uso_tc_12m.round(4),
            "uso_tc_prom_3m": uso_tc_3m.round(4),
            "meses_desde_mora_12m": meses_desde_mora,
            "antiguedad_meses": antig,
            "edad": edad,
            "renta_mm": renta,
            "deuda_otras_prom_12m": deuda_otras,
            "carga_financiera": carga,
            "consultas_6m": consultas_6m.astype(float),
            "canal": canal,
            "pd_verdadera": pd_verdadera,
            "malo": malo,
        })
        es_ttd = df["cohorte"] >= "2025-07"
        es_oot = (df["cohorte"] >= "2025-01") & ~es_ttd
        azar = rng.random(n)
        df["muestra"] = np.select([es_ttd, es_oot, azar < 0.70], ["TTD", "OOT", "DEV"], "HO")
        df.loc[es_ttd, "malo"] = np.nan
        return df


    # --- herramientas del curso (idénticas en lógica a binear/tabla_woe de clase) ---
    def binear(x, bins=5, umbral_moda=0.35, ref=None):
        """Binning del curso. Con `ref`, los cortes se calculan en `ref` y se aplican a `x`."""
        x = pd.Series(x).reset_index(drop=True)
        base_b = x if ref is None else pd.Series(ref).reset_index(drop=True)
        if not pd.api.types.is_numeric_dtype(base_b):
            return x.astype(str).where(x.notna(), "MISSING"), None
        if base_b.nunique(dropna=True) <= bins:
            return x.astype(str).where(x.notna(), "MISSING"), None
        moda = base_b.mode().iloc[0]
        frac_moda = (base_b == moda).mean()
        resto = base_b[base_b != moda] if frac_moda > umbral_moda else base_b
        cortes = np.unique(np.nanquantile(resto.dropna(), np.linspace(0, 1, bins + 1)))
        if len(cortes) < 3:
            return x.astype(str).where(x.notna(), "MISSING"), None
        cortes[0], cortes[-1] = -np.inf, np.inf
        categorias = pd.cut(x, cortes)
        orden = [str(c) for c in categorias.cat.categories]
        etiquetas = categorias.astype(str)
        if frac_moda > umbral_moda:
            etiquetas = etiquetas.where(x != moda, f"= {moda:.4g}")
            orden = [f"= {moda:.4g}"] + orden
        return etiquetas.where(x.notna(), "MISSING"), orden + ["MISSING"]


    def tabla_woe(x, y, bins=5, ref=None):
        """Tabla WoE de una variable y su IV (convención: target 1 = malo, WoE alto = bin bueno)."""
        etiquetas, orden = binear(x, bins, ref=ref)
        tab = pd.DataFrame({"bin": etiquetas.values, "y": pd.Series(y).values}) \
                .groupby("bin")["y"].agg(["count", "sum"])
        tab.columns = ["n", "malos"]
        if orden:
            tab = tab.reindex([o for o in orden if o in tab.index])
        tab["buenos"] = tab["n"] - tab["malos"]
        tab["tasa_malos"] = tab["malos"] / tab["n"]
        p_malos = (tab["malos"] + 0.5) / (tab["malos"].sum() + 0.5 * len(tab))
        p_buenos = (tab["buenos"] + 0.5) / (tab["buenos"].sum() + 0.5 * len(tab))
        tab["woe"] = np.log(p_buenos / p_malos)
        tab["iv_aporte"] = (p_buenos - p_malos) * tab["woe"]
        return tab, float(tab["iv_aporte"].sum())


    def a_woe(df, variables, dev, mapas):
        """Transforma `df` al WoE de DEV (cortes y WoE de DEV; bin no visto → 0 = neutro)."""
        F = pd.DataFrame(index=range(len(df)))
        for v in variables:
            etiquetas, _ = binear(df[v], ref=dev[v])
            F[v] = etiquetas.map(mapas[v]).fillna(0.0).values
        return F
    return a_woe, generar_cartera, np, pd, tabla_woe


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Banco Austral: los números de la lámina

    Dos tablas de la clase 5 (láminas 22 y 24), tal como se presentaron. Son **agregados**: n, PD
    calibrada media y malos por banda; n, PD media, esperados y observados por decil de PD. Con eso
    basta para reproducir casi todo. Lo único que no se puede reproducir exacto es lo que depende de
    las PD individuales (el p simulado del HL), y ahí se explica la diferencia.
    """)
    return


@app.cell
def _(np, pd):
    austral_bandas = pd.DataFrame(
        {
            "banda": ["A1", "A2", "B1", "B2", "C1", "C2", "D", "E"],
            "n": [432, 199, 244, 235, 239, 220, 172, 263],
            "pd": [0.0011, 0.0036, 0.0072, 0.0142, 0.0280, 0.0544, 0.1021, 0.2691],
            "malos": [1, 3, 4, 8, 8, 9, 22, 64],
            "p_curso": [0.373, 0.036, 0.100, 0.020, 0.554, 0.458, 0.257, 0.367],
        }
    ).set_index("banda")
    austral_bandas["esperados"] = austral_bandas["n"] * austral_bandas["pd"]
    austral_bandas["tasa_obs"] = austral_bandas["malos"] / austral_bandas["n"]

    austral_hl = pd.DataFrame(
        {
            "grupo": np.arange(1, 11),
            "n": [201, 200, 200, 201, 200, 200, 201, 200, 200, 201],
            "pd_media": [0.0005, 0.0014, 0.0032, 0.0062, 0.0110, 0.0193, 0.0340, 0.0621, 0.1211, 0.3056],
            "esperados": [0.106, 0.289, 0.641, 1.246, 2.191, 3.856, 6.839, 12.419, 24.229, 61.430],
            "observados": [0, 0, 3, 2, 8, 7, 5, 11, 26, 57],
        }
    ).set_index("grupo")
    austral_bandas.round(4)
    return austral_bandas, austral_hl


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. El test binomial exacto, desde la pmf

    Bajo $H_0$ los malos de la banda son $D\sim\text{Bin}(n,\text{PD})$. La pmf se calcula en escala
    logarítmica para no desbordar con n grandes:
    $\ln P(D=k)=\ln\binom{n}{k}+k\ln p+(n-k)\ln(1-p)$, con $\ln\binom{n}{k}$ armado desde la suma
    acumulada de logaritmos (numpy puro, sin `gammaln`).

    Cinco p-valores para el mismo dato:

    - **una cola superior** $P(D\ge k)$: la pregunta del regulador (¿la PD **subestima**?);
    - **una cola inferior** $P(D\le k)$;
    - **dos colas «minlike»** $\sum_{j:\,P(j)\le P(k)}P(j)$: la definición de `scipy.stats.binomtest`
      (y de R `binom.test`), la que usó el curso;
    - **dos colas «doble»** $\min\{1,\,2\min(P(D\ge k),P(D\le k))\}$ (la de muchas planillas);
    - **mid-p superior** $P(D>k)+\tfrac12P(D=k)$;
    - y la **aproximación normal** bilateral con $z=(k-np)/\sqrt{np(1-p)}$.

    Qué mirar: si la columna `minlike` reproduce la tabla del curso y por qué en B2 coincide con la
    **una cola** (0,020).
    """)
    return


@app.cell
def _(np, stats):
    def log_comb_np(n):
        """ln C(n, k) para k = 0..n con numpy puro (suma acumulada de logaritmos)."""
        _lf = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])  # ln k!
        return _lf[n] - _lf[: n + 1] - _lf[n::-1]

    def pmf_binom_np(n, p):
        """pmf completa P(D = k), k = 0..n, en escala log y luego exp."""
        _k = np.arange(n + 1)
        with np.errstate(divide="ignore"):
            _lp = log_comb_np(n) + _k * np.log(p) + (n - _k) * np.log1p(-p)
        return np.exp(_lp)

    def pvalores_binom_np(k, n, p):
        """Todas las variantes de p-valor binomial para k malos en n con PD p."""
        _pmf = pmf_binom_np(n, p)
        _sup = float(_pmf[k:].sum())
        _inf = float(_pmf[: k + 1].sum())
        # minlike: suma de los resultados tan o menos probables que el observado.
        # scipy usa una tolerancia relativa 1 + 1e-7 para empates numéricos.
        _minlike = float(_pmf[_pmf <= _pmf[k] * (1 + 1e-7)].sum())
        _z = (k - n * p) / np.sqrt(n * p * (1 - p))
        return {
            "sup": min(_sup, 1.0),
            "inf": min(_inf, 1.0),
            "dos_minlike": min(_minlike, 1.0),
            "dos_doble": min(1.0, 2 * min(_sup, _inf)),
            "mid_sup": float(_pmf[k + 1:].sum() + 0.5 * _pmf[k]),
            "z": float(_z),
            "normal_dos": float(2 * stats.norm.sf(abs(_z))),
        }

    return pmf_binom_np, pvalores_binom_np


@app.cell
def _(austral_bandas, pd, pvalores_binom_np, stats):
    _filas = []
    for _b, _f in austral_bandas.iterrows():
        _k, _n, _p = int(_f["malos"]), int(_f["n"]), float(_f["pd"])
        _np_ = pvalores_binom_np(_k, _n, _p)
        _filas.append(
            {
                "banda": _b,
                "np": _n * _p,
                "p_curso": _f["p_curso"],
                "sup_numpy": _np_["sup"],
                "sup_scipy": stats.binomtest(_k, _n, _p, alternative="greater").pvalue,
                "inf_numpy": _np_["inf"],
                "inf_scipy": stats.binomtest(_k, _n, _p, alternative="less").pvalue,
                "minlike_numpy": _np_["dos_minlike"],
                "minlike_scipy": stats.binomtest(_k, _n, _p).pvalue,
                "doble": _np_["dos_doble"],
                "mid_sup": _np_["mid_sup"],
                "normal_dos": _np_["normal_dos"],
            }
        )
    tabla_binom = pd.DataFrame(_filas).set_index("banda")

    def semaforo(p):
        return "🔴" if p < 0.01 else ("🟡" if p < 0.05 else "🟢")

    tabla_binom["color_minlike"] = tabla_binom["minlike_numpy"].map(semaforo)
    tabla_binom["color_doble"] = tabla_binom["doble"].map(semaforo)
    tabla_binom["color_normal"] = tabla_binom["normal_dos"].map(semaforo)
    tabla_binom.round(4)
    return semaforo, tabla_binom


@app.cell
def _(austral_bandas, brentq, stats):
    # A1: el curso reporta 0,373 y aquí sale 0,378. ¿Qué PD media reproduce 0,3733?
    pd_a1_implicita = brentq(
        lambda _p: stats.binom.sf(0, 432, _p) - 0.3733, 1e-5, 0.01, xtol=1e-12
    )
    # y el global: 119 malos en 2.004 contra la PD media ponderada
    pd_global_austral = float(
        (austral_bandas["n"] * austral_bandas["pd"]).sum() / austral_bandas["n"].sum()
    )
    p_global_austral = stats.binomtest(119, 2004, pd_global_austral).pvalue
    return p_global_austral, pd_a1_implicita, pd_global_austral


@app.cell
def _(mo, p_global_austral, pd_a1_implicita, pd_global_austral, tabla_binom):
    _b2 = tabla_binom.loc["B2"]
    _a2 = tabla_binom.loc["A2"]
    mo.md(rf"""
    **Lectura.**

    - La columna `minlike` (numpy = scipy) reproduce la tabla del curso banda por banda. B2:
      **{_b2['minlike_numpy']:.4f}**. La lámina 20 dice «8 o más pasa el 2,0% de las veces» (una cola) y
      la lámina 22 dice «bilateral exacto». **Las dos cosas son ciertas a la vez**: con $np=3{{,}}34$ el
      resultado menos probable de la cola izquierda es $P(D=0)=0{{,}}0347$, que es **mayor** que
      $P(D=8)=0{{,}}0132$; ningún punto de la cola izquierda entra en la suma «minlike» y el p bilateral
      coincide con el de una cola. Pasa lo mismo en A1, A2, B1 y B2 (bandas con $np$ chico). En C1
      no: minlike {tabla_binom.loc['C1', 'minlike_numpy']:.3f} vs una cola superior
      {tabla_binom.loc['C1', 'sup_numpy']:.3f}.
    - La convención «doble» **duplica** el p en las bandas asimétricas: B2 pasaría a
      {_b2['doble']:.3f} (sigue 🟡) y A2 a {_a2['doble']:.3f} (**verde**). La elección de la definición
      bilateral cambia colores: se declara en la política, no se elige después.
    - La aproximación normal es la peor: A2 ({_a2['np']:.2f} esperados) da {_a2['normal_dos']:.4f} → 🔴
      **falso**; B2 da {_b2['normal_dos']:.4f}, pegado al rojo. Con $np<5$ la normal no sirve.
    - A1 da {tabla_binom.loc['A1', 'minlike_numpy']:.3f} y no 0,373 porque la tabla muestra la PD
      **redondeada** (0,11%); la PD media que reproduce 0,3733 es {pd_a1_implicita:.5%}. Cualquier
      reproducción desde una tabla impresa hereda ese redondeo.
    - Global: PD media ponderada {pd_global_austral:.4%}, 119 malos vs {2004 * pd_global_austral:.1f}
      esperados, p = **{p_global_austral:.3f}** (curso 0,562).
    """)
    return


@app.cell
def _(austral_bandas, mo):
    selector_banda = mo.ui.dropdown(
        options=list(austral_bandas.index), value="B2", label="Banda a inspeccionar"
    )
    selector_banda
    return (selector_banda,)


@app.cell
def _(austral_bandas, np, plt, pmf_binom_np, pvalores_binom_np, selector_banda):
    _b = selector_banda.value
    _n, _p, _k = int(austral_bandas.loc[_b, "n"]), float(austral_bandas.loc[_b, "pd"]), int(austral_bandas.loc[_b, "malos"])
    _pmf = pmf_binom_np(_n, _p)
    _kmax = int(max(_k + 4, _n * _p + 5 * np.sqrt(_n * _p * (1 - _p)) + 3))
    _kmax = min(_kmax, _n)
    _x = np.arange(_kmax + 1)
    _en_minlike = _pmf[: _kmax + 1] <= _pmf[_k] * (1 + 1e-7)
    _colores = np.where(_x >= _k, "#c0392b", np.where(_en_minlike, "#e67e22", "#95a5a6"))
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.bar(_x, _pmf[: _kmax + 1], color=_colores)
    _ax.axvline(_n * _p, color="k", ls="--", lw=1, label=f"esperados = {_n * _p:.1f}")
    _pv = pvalores_binom_np(_k, _n, _p)
    _ax.set_title(
        f"Banda {_b}: n={_n}, PD={_p:.2%}, observados={_k} · "
        f"p sup {_pv['sup']:.3f} · minlike {_pv['dos_minlike']:.3f}"
    )
    _ax.set_xlabel("malos en la banda (k)")
    _ax.set_ylabel("P(D = k) bajo H0")
    _ax.legend(title="rojo: cola ≥ k · naranjo: cola izq. que entra al minlike", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Potencia: ¿qué desvío puede detectar cada banda?

    Un test que no rechaza no «valida» nada si no tenía potencia. Para cada banda, con test de una cola
    al 5%, el valor crítico es $k^*=\min\{k: P(D\ge k\mid \text{PD})\le 0{,}05\}$ y la potencia contra
    una PD verdadera $m\cdot\text{PD}$ es $P(D\ge k^*\mid m\cdot\text{PD})$. Mueve el multiplicador $m$.

    Dos columnas extra: el **tamaño real** del test (por discreción es menor que 5%) y el multiplicador
    $m_{80}$ que haría falta para detectar con 80% de probabilidad.
    """)
    return


@app.cell
def _(mo):
    slider_m = mo.ui.slider(1.0, 4.0, step=0.1, value=2.0, label="Multiplicador m de la PD verdadera")
    slider_m
    return (slider_m,)


@app.cell
def _(austral_bandas, brentq, np, pd, pmf_binom_np, slider_m, stats):
    def critico_sup_np(n, p, alfa=0.05):
        """Menor k con P(D >= k) <= alfa (numpy, desde la pmf)."""
        _cola = np.cumsum(pmf_binom_np(n, p)[::-1])[::-1]  # P(D >= k)
        return int(np.argmax(_cola <= alfa))

    def potencia_np(n, p, m, alfa=0.05):
        _k = critico_sup_np(n, p, alfa)
        return float(pmf_binom_np(n, min(m * p, 1 - 1e-12))[_k:].sum())

    _filas = []
    for _b, _f in austral_bandas.iterrows():
        _n, _p = int(_f["n"]), float(_f["pd"])
        _k = critico_sup_np(_n, _p)
        _k_scipy = int(stats.binom.ppf(0.95, _n, _p)) + 1
        _m80 = brentq(lambda _m: potencia_np(_n, _p, _m) - 0.80, 1.0, 0.999 / _p)
        _filas.append(
            {
                "banda": _b,
                "n": _n,
                "pd": _p,
                "k_critico": _k,
                "k_critico_scipy": _k_scipy,
                "tamano_real": float(stats.binom.sf(_k - 1, _n, _p)),
                "potencia_m": potencia_np(_n, _p, slider_m.value),
                "potencia_m_scipy": float(stats.binom.sf(_k - 1, _n, min(slider_m.value * _p, 1))),
                "m80": _m80,
                "pd_detectable_80": _m80 * _p,
            }
        )
    tabla_potencia = pd.DataFrame(_filas).set_index("banda")
    tabla_potencia.round(4)
    return critico_sup_np, potencia_np, tabla_potencia


@app.cell
def _(austral_bandas, np, plt, potencia_np, slider_m):
    _ms = np.linspace(1, 6, 60)
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    for _b, _f in austral_bandas.iterrows():
        _ax.plot(_ms, [potencia_np(int(_f["n"]), float(_f["pd"]), _m) for _m in _ms], label=_b)
    _ax.axhline(0.8, color="k", ls=":", lw=1)
    _ax.axvline(slider_m.value, color="grey", ls="--", lw=1)
    _ax.set_xlabel("m = PD verdadera / PD calibrada")
    _ax.set_ylabel("potencia (una cola, 5%)")
    _ax.set_title("Potencia del binomial por banda (n de Austral OOT)")
    _ax.legend(ncol=4, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, slider_m, tabla_potencia):
    _t = tabla_potencia
    mo.md(rf"""
    **Lectura.** Con $m={slider_m.value:.1f}$ la potencia va de {_t['potencia_m'].min():.0%}
    ({_t['potencia_m'].idxmin()}) a {_t['potencia_m'].max():.0%} ({_t['potencia_m'].idxmax()}).
    Para detectar con 80% hace falta que la PD verdadera de A1 sea **{_t.loc['A1', 'm80']:.1f} veces** la
    calibrada; en B2 {_t.loc['B2', 'm80']:.2f} veces (PD {_t.loc['B2', 'pd_detectable_80']:.2%} en vez de 1,42%);
    en E basta {_t.loc['E', 'm80']:.2f}. Las bandas buenas son **ciegas** a desvíos relativos grandes: un
    verde en A1 no es evidencia de calibración, es falta de datos. Curiosidad de B2: el valor crítico es
    exactamente **{int(_t.loc['B2', 'k_critico'])}** malos, lo observado. Y el tamaño real del test es
    {_t['tamano_real'].min():.1%}–{_t['tamano_real'].max():.1%}, no 5%: el binomial exacto es conservador
    por discreción.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Correlación de defaults: el binomial rechaza de más

    El binomial supone que los defaults son independientes. En el modelo de un factor de Vasicek (el de
    las fórmulas de capital de Basilea) el deudor $i$ cae si
    $\sqrt{\rho}\,Z+\sqrt{1-\rho}\,\varepsilon_i<\Phi^{-1}(\text{PD})$, con $Z$ común a todos (el año) y
    $\varepsilon_i$ idiosincrático. Condicional a $Z=z$, los defaults **sí** son independientes con
    $$p(z)=\Phi\!\left(\frac{\Phi^{-1}(\text{PD})-\sqrt{\rho}\,z}{\sqrt{1-\rho}}\right),$$
    y la distribución de $D$ es la mezcla $P(D\ge k)=\int P(\text{Bin}(n,p(z))\ge k)\,\phi(z)\,dz$.

    Implementación numpy: regla del trapecio sobre una grilla uniforme de $z$ (paso 0,05). Librería:
    `scipy.integrate.quad` (adaptativa). Check externo: el ejemplo de Tasche (2006) — n = 1.000, PD 1%, 19 defaults:
    p = 0,7% independiente vs 11,1% con ρ = 5%.

    El slider de ρ alimenta (a) la simulación de una cartera **perfectamente calibrada** con los n y PD
    de Austral y (b) el p ajustado de cada banda.
    """)
    return


@app.cell
def _(integrate, np, stats):
    # Grilla uniforme en z con regla del trapecio (paso 0,05 en [-8,5; 8,5]). Para integrandos suaves
    # contra la densidad normal converge exponencialmente; Gauss-Hermite con 80 nodos NO sirve aquí:
    # con n grande el integrando es casi un escalón en z y GH se equivoca hasta en ~0,01.
    _h = 0.05
    _nodos = np.arange(-8.5, 8.5 + _h / 2, _h)
    _pesos = stats.norm.pdf(_nodos) * _h
    _pesos[[0, -1]] *= 0.5
    _pesos = _pesos / _pesos.sum()

    def pd_condicional(pd_, rho, z):
        return stats.norm.cdf((stats.norm.ppf(pd_) - np.sqrt(rho) * z) / np.sqrt(1 - rho))

    def p_sup_vasicek_np(k, n, pd_, rho):
        """P(D >= k) bajo Vasicek: mezcla de binomiales integrada con trapecio en z."""
        if rho <= 0:
            return float(stats.binom.sf(k - 1, n, pd_))
        _pz = pd_condicional(pd_, rho, _nodos)
        return float((_pesos * stats.binom.sf(k - 1, n, _pz)).sum())

    def p_sup_vasicek_quad(k, n, pd_, rho):
        _f = lambda z: stats.binom.sf(k - 1, n, pd_condicional(pd_, rho, z)) * stats.norm.pdf(z)
        return integrate.quad(_f, -10, 10, limit=200, epsabs=1e-13)[0]

    tasche_indep = float(stats.binom.sf(18, 1000, 0.01))
    tasche_rho5 = p_sup_vasicek_np(19, 1000, 0.01, 0.05)
    tasche_rho5_quad = p_sup_vasicek_quad(19, 1000, 0.01, 0.05)
    return (
        p_sup_vasicek_np,
        p_sup_vasicek_quad,
        pd_condicional,
        tasche_indep,
        tasche_rho5,
        tasche_rho5_quad,
    )


@app.cell
def _(mo):
    slider_rho = mo.ui.slider(0.0, 0.20, step=0.01, value=0.05, label="Correlación de activos ρ")
    slider_rho
    return (slider_rho,)


@app.cell
def _(austral_bandas, critico_sup_np, np, p_sup_vasicek_np, pd, pd_condicional, slider_rho):
    _rho = slider_rho.value
    _R = 4000
    _rng = np.random.default_rng(20260928)
    _z = _rng.standard_normal(_R)
    _filas = []
    _rech_global = np.zeros(_R, dtype=bool)
    _D_tot = np.zeros(_R)
    for _b, _f in austral_bandas.iterrows():
        _n, _p = int(_f["n"]), float(_f["pd"])
        _pz = pd_condicional(_p, _rho, _z) if _rho > 0 else np.full(_R, _p)
        _D = _rng.binomial(_n, _pz)
        _D_tot += _D
        _k = critico_sup_np(_n, _p)
        _rech = _D >= _k
        _rech_global |= _rech
        _filas.append(
            {
                "banda": _b,
                "tamano_indep": p_sup_vasicek_np(_k, _n, _p, 0.0),
                "rechazo_sim": _rech.mean(),
                "rechazo_exacto": p_sup_vasicek_np(_k, _n, _p, _rho),
                "p_obs_indep": p_sup_vasicek_np(int(_f["malos"]), _n, _p, 0.0),
                "p_obs_ajustado": p_sup_vasicek_np(int(_f["malos"]), _n, _p, _rho),
            }
        )
    tabla_rho = pd.DataFrame(_filas).set_index("banda")
    rechazo_alguna_banda = float(_rech_global.mean())
    error_mc_rho = float(np.max(np.abs(tabla_rho["rechazo_sim"] - tabla_rho["rechazo_exacto"])))
    tabla_rho.round(4)
    return error_mc_rho, rechazo_alguna_banda, tabla_rho


@app.cell
def _(austral_bandas, critico_sup_np, np, p_sup_vasicek_np, plt, slider_rho):
    _rhos = np.linspace(0, 0.20, 21)
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    for _b in ["A1", "B2", "C2", "E"]:
        _n, _p = int(austral_bandas.loc[_b, "n"]), float(austral_bandas.loc[_b, "pd"])
        _k = critico_sup_np(_n, _p)
        _ax.plot(_rhos, [p_sup_vasicek_np(_k, _n, _p, _r) for _r in _rhos], label=_b)
    # la cartera completa como una sola «banda» (test global)
    _k_glob = critico_sup_np(2004, 0.0565)
    _ax.plot(_rhos, [p_sup_vasicek_np(_k_glob, 2004, 0.0565, _r) for _r in _rhos], "k--", label="global (n=2.004)")
    _ax.axhline(0.05, color="grey", ls=":", lw=1)
    _ax.axvline(slider_rho.value, color="grey", ls="--", lw=1)
    _ax.set_xlabel("ρ (correlación de activos)")
    _ax.set_ylabel("tasa de rechazo real (nominal 5%)")
    _ax.set_title("Binomial independiente sobre una cartera bien calibrada con correlación")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, rechazo_alguna_banda, slider_rho, tabla_rho, tasche_indep, tasche_rho5):
    _t = tabla_rho
    mo.md(rf"""
    **Lectura (ρ = {slider_rho.value:.2f}).** Check de Tasche: independiente {tasche_indep:.4f}, con
    ρ = 5% {tasche_rho5:.4f} (el paper: 0,7% y 11,1%). En la cartera de Austral **perfectamente
    calibrada**, el binomial independiente de una cola al 5% rechaza a B2 el
    {_t.loc['B2', 'rechazo_exacto']:.1%} de los años y a E el {_t.loc['E', 'rechazo_exacto']:.1%};
    al menos una banda en rojo-o-amarillo el {rechazo_alguna_banda:.0%} de los años. El exceso crece con
    $n\cdot$PD: en las bandas chicas domina el ruido binomial y la correlación casi no se nota; en las
    grandes (y en el test **global**) domina el factor común y el binomial se vuelve un detector del
    ciclo, no del modelo.

    El p observado de B2 pasa de {_t.loc['B2', 'p_obs_indep']:.4f} a **{_t.loc['B2', 'p_obs_ajustado']:.4f}**
    con el ajuste; A2 de {_t.loc['A2', 'p_obs_indep']:.4f} a {_t.loc['A2', 'p_obs_ajustado']:.4f}.
    Ojo con el signo de la lección: el ajuste por correlación **no absuelve** al modelo. Dice que un año
    malo puede producir esa desviación sin que el modelo esté mal *en promedio del ciclo*; si la PD
    pretende ser PIT (condicional al año), el factor común ya debería estar dentro de la PD y el ρ
    relevante es el residual, mucho menor.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.1 Semáforo de Tasche (2003)

    Tasche propone fijar dos cuantiles de la distribución de $D$ **bajo el modelo de un factor**: verde
    si $D < c_{95\%}$, amarillo si $c_{95\%}\le D < c_{99{,}9\%}$, rojo si $D\ge c_{99{,}9\%}$ (los niveles
    de confianza son los del paper; 99,9% es el de la fórmula de capital). Aquí los cuantiles se calculan
    exactos (mezcla por cuadratura) con el ρ del slider y se comparan con los de la binomial
    independiente.
    """)
    return


@app.cell
def _(austral_bandas, p_sup_vasicek_np, pd, slider_rho):
    def _cuantil_mezcla(n, pd_, rho, conf):
        """Menor c tal que P(D >= c) <= 1 - conf bajo Vasicek (o binomial si rho = 0)."""
        _c = 1
        while _c <= n and p_sup_vasicek_np(_c, n, pd_, rho) > 1 - conf:
            _c += 1
        return _c

    _filas = []
    for _b, _f in austral_bandas.iterrows():
        _n, _p, _k = int(_f["n"]), float(_f["pd"]), int(_f["malos"])
        _c95_i, _c999_i = _cuantil_mezcla(_n, _p, 0.0, 0.95), _cuantil_mezcla(_n, _p, 0.0, 0.999)
        _c95, _c999 = _cuantil_mezcla(_n, _p, slider_rho.value, 0.95), _cuantil_mezcla(_n, _p, slider_rho.value, 0.999)

        def _color(k, a, r):
            return "🔴" if k >= r else ("🟡" if k >= a else "🟢")

        _filas.append(
            {
                "banda": _b, "malos": _k,
                "c95_indep": _c95_i, "c999_indep": _c999_i, "color_indep": _color(_k, _c95_i, _c999_i),
                "c95_rho": _c95, "c999_rho": _c999, "color_rho": _color(_k, _c95, _c999),
            }
        )
    tabla_tasche = pd.DataFrame(_filas).set_index("banda")
    tabla_tasche
    return (tabla_tasche,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Hosmer-Lemeshow

    $$\text{HL}=\sum_{g=1}^{G}\frac{(O_g-E_g)^2}{E_g(1-\bar p_g)},\qquad E_g=n_g\bar p_g .$$

    Dos implementaciones: (1) numpy directo; (2) `scipy.stats.chisquare` sobre la tabla $2\times G$
    (malos y buenos por grupo). Son **idénticas** algebraicamente:
    $\frac{(O-E)^2}{E}+\frac{(O-E)^2}{n-E}=\frac{(O-E)^2\,n}{E(n-E)}=\frac{(O-E)^2}{E(1-\bar p)}$.
    `statsmodels` solo lo trae como `diagnostic_gen.test_chisquare_binning` (df = $G-2$ por defecto);
    esta identidad con `scipy` es la «implementación alternativa».

    Tres p-valores: $\chi^2_{G-2}$ (desarrollo: las PD se estimaron con esos datos), $\chi^2_G$ (validación
    externa: las PD vienen de afuera, sin grados de libertad consumidos) y **simulado** (bootstrap
    paramétrico bajo $H_0$: $y_i\sim\text{Bernoulli}(p_i)$, mismo agrupamiento, mismo estadístico).
    """)
    return


@app.cell
def _(np, pd, stats):
    def hl_desde_agregados(n, pbar, obs):
        """HL desde (n_g, PD media_g, observados_g). Devuelve estadístico y aportes."""
        n, pbar, obs = (np.asarray(a, dtype=float) for a in (n, pbar, obs))
        _esp = n * pbar
        _aportes = (obs - _esp) ** 2 / (_esp * (1 - pbar))
        return float(_aportes.sum()), _aportes

    def hl_scipy_2xg(n, pbar, obs):
        """Misma cosa como Pearson χ² sobre la tabla 2×G (malos y buenos)."""
        n, pbar, obs = (np.asarray(a, dtype=float) for a in (n, pbar, obs))
        _esp = n * pbar
        _f_obs = np.concatenate([obs, n - obs])
        _f_esp = np.concatenate([_esp, n - _esp])
        return float(stats.chisquare(_f_obs, _f_esp).statistic)

    def grupos_hl(p, g=10, cortes=None):
        """Etiqueta de grupo: cuantiles de p (deciles si g=10) o cortes fijos (bandas)."""
        if cortes is None:
            return pd.qcut(pd.Series(p).rank(method="first"), g, labels=False).to_numpy()
        return np.searchsorted(np.asarray(cortes), p, side="right")

    def hl_individual(p, y, grupos):
        """HL numpy desde PD individuales y grupos; devuelve estadístico, tabla y arrays."""
        p, y = np.asarray(p, float), np.asarray(y, float)
        _gs = np.unique(grupos)
        _idx = np.searchsorted(_gs, grupos)
        _n = np.bincount(_idx).astype(float)
        _o = np.bincount(_idx, weights=y)
        _pbar = np.bincount(_idx, weights=p) / _n
        _stat, _ap = hl_desde_agregados(_n, _pbar, _o)
        _tabla = pd.DataFrame({"n": _n, "pd_media": _pbar, "esperados": _n * _pbar, "observados": _o, "aporte": _ap})
        return _stat, _tabla, _idx, _n, _pbar

    def hl_p_simulado(p, idx, n_g, pbar_g, stat_obs, S=2000, semilla=20260908, bloque=500):
        """Bootstrap paramétrico bajo H0: y_i ~ Bernoulli(p_i), mismo agrupamiento."""
        _rng = np.random.default_rng(semilla)
        _esp = n_g * pbar_g
        _den = _esp * (1 - pbar_g)
        _G = len(n_g)
        _sims = []
        for _ini in range(0, S, bloque):
            _s = min(bloque, S - _ini)
            _y = (_rng.random((_s, len(p))) < p).astype(float)
            _o = np.zeros((_s, _G))
            for _g in range(_G):
                _o[:, _g] = _y[:, idx == _g].sum(axis=1)
            _sims.append(((_o - _esp) ** 2 / _den).sum(axis=1))
        _sims = np.concatenate(_sims)
        return float((_sims >= stat_obs).mean()), _sims

    return grupos_hl, hl_desde_agregados, hl_individual, hl_p_simulado, hl_scipy_2xg


@app.cell
def _(austral_bandas, austral_hl, hl_desde_agregados, hl_scipy_2xg, np, stats):
    # Reproducción de la lámina 24: PD media = esperados / n (más decimales que la columna impresa)
    _pbar = austral_hl["esperados"].to_numpy() / austral_hl["n"].to_numpy()
    hl_austral, aportes_austral = hl_desde_agregados(austral_hl["n"], _pbar, austral_hl["observados"])
    hl_austral_scipy = hl_scipy_2xg(austral_hl["n"], _pbar, austral_hl["observados"])
    p_hl_austral_g2 = float(stats.chi2.sf(hl_austral, 8))
    p_hl_austral_g = float(stats.chi2.sf(hl_austral, 10))
    # p simulado con PD homogénea dentro de cada grupo (lo único que permite la tabla agregada)
    _rng = np.random.default_rng(20260908)
    _sim = _rng.binomial(austral_hl["n"].to_numpy(), _pbar, size=(200_000, 10))
    _esp = austral_hl["n"].to_numpy() * _pbar
    _hl_sim = ((_sim - _esp) ** 2 / (_esp * (1 - _pbar))).sum(axis=1)
    p_hl_austral_sim_homog = float((_hl_sim >= hl_austral).mean())
    aporte_g3_g5 = float(aportes_austral[2] + aportes_austral[4])
    # El mismo HL sobre las 8 bandas de la master scale (lámina 22): otro agrupamiento, otro número
    _nb, _pb, _kb = (austral_bandas[_c].to_numpy() for _c in ("n", "pd", "malos"))
    hl_bandas_austral, _ = hl_desde_agregados(_nb, _pb, _kb)
    p_hl_bandas_chi2 = float(stats.chi2.sf(hl_bandas_austral, 8))
    _simb = _rng.binomial(_nb, _pb, size=(200_000, 8))
    p_hl_bandas_sim = float((((_simb - _nb * _pb) ** 2 / (_nb * _pb * (1 - _pb))).sum(axis=1) >= hl_bandas_austral).mean())
    return (
        hl_bandas_austral,
        p_hl_bandas_chi2,
        p_hl_bandas_sim,
        aporte_g3_g5,
        aportes_austral,
        hl_austral,
        hl_austral_scipy,
        p_hl_austral_g,
        p_hl_austral_g2,
        p_hl_austral_sim_homog,
    )


@app.cell
def _(
    aporte_g3_g5,
    hl_austral,
    hl_austral_scipy,
    hl_bandas_austral,
    mo,
    p_hl_bandas_chi2,
    p_hl_bandas_sim,
    p_hl_austral_g,
    p_hl_austral_g2,
    p_hl_austral_sim_homog,
):
    mo.md(rf"""
    **Reproducción de la lámina 24.** HL numpy = **{hl_austral:.2f}**, scipy (tabla 2×10) =
    {hl_austral_scipy:.2f}; grupos 3 y 5 aportan {aporte_g3_g5:.1f} (curso: 24,3 de 29,0).

    | referencia | p |
    |---|---|
    | $\chi^2_8$ (lo que reportó el curso y lo que usa nikodym, incluso en OOT) | {p_hl_austral_g2:.4f} |
    | $\chi^2_{{10}}$ (validación externa: lo correcto para OOT con PD fijadas antes) | {p_hl_austral_g:.4f} |
    | simulado, PD homogénea por grupo (200.000 réplicas) | {p_hl_austral_sim_homog:.4f} |
    | simulado del curso, PD individuales (10.000 réplicas) | 0,012 |

    La diferencia entre {p_hl_austral_sim_homog:.3f} y 0,012 no es error: con PD homogénea en el grupo la
    varianza de $O_g$ es $n\bar p(1-\bar p)$; con PD individuales es $\sum p_i(1-p_i)$, **menor**
    (varianza de una Poisson-binomial), así que la distribución nula es más angosta y el p baja. La
    diferencia pesa sobre todo en el grupo 10 (PD de ~18% a ~90%). Y notar: incluso con $\chi^2_{{10}}$,
    la aproximación asintótica sigue diciendo rojo. El problema no son los grados de libertad, son los
    **esperados chicos**.

    **Mismo dato, otro agrupamiento.** El HL sobre las 8 bandas de la lámina 22 da
    {hl_bandas_austral:.2f}: $\chi^2_8$ p = {p_hl_bandas_chi2:.4f} (🔴) y simulado p = {p_hl_bandas_sim:.3f} (🟡).
    Cuatro de las ocho bandas esperan menos de 5 malos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.1 ¿Cuánto miente la χ² con esperados chicos? Un «Austral de laboratorio»

    Construimos 2.004 PD individuales que reproducen las PD medias de los 10 grupos de la lámina (PD
    log-uniformes dentro de cada grupo, reescaladas a la media exacta). Luego simulamos 10.000 mundos
    donde **la PD es verdad** y miramos la distribución del HL. Si la $\chi^2_8$ fuera buena, el 1% de los
    mundos pasaría el cuantil 99% de la $\chi^2_8$.
    """)
    return


@app.cell
def _(austral_hl, hl_individual, hl_p_simulado, np, stats):
    _rng = np.random.default_rng(7)
    _medias = austral_hl["esperados"].to_numpy() / austral_hl["n"].to_numpy()
    _fronteras = np.sqrt(_medias[:-1] * _medias[1:])
    _lo = np.concatenate([[_medias[0] / 2.5], _fronteras])
    _hi = np.concatenate([_fronteras, [0.90]])
    _pds = []
    for _g in range(10):
        _u = np.exp(_rng.uniform(np.log(_lo[_g]), np.log(_hi[_g]), austral_hl["n"].iloc[_g]))
        _u = np.clip(_u * _medias[_g] / _u.mean(), 1e-6, 0.95)
        _pds.append(np.sort(_u))
    pd_lab = np.concatenate(_pds)
    _grupo_lab = np.repeat(np.arange(10), austral_hl["n"].to_numpy())
    _stat0, _tabla_lab, _idx_lab, _n_lab, _pbar_lab = hl_individual(pd_lab, np.zeros_like(pd_lab), _grupo_lab)
    # p simulado del 29,04 observado bajo estas PD individuales
    p_lab_29, hl_nulo_lab = hl_p_simulado(pd_lab, _idx_lab, _n_lab, _pbar_lab, 29.04, S=10_000, semilla=20260908)
    tamano_chi2_8_1 = float((hl_nulo_lab >= stats.chi2.isf(0.01, 8)).mean())
    tamano_chi2_8_5 = float((hl_nulo_lab >= stats.chi2.isf(0.05, 8)).mean())
    tamano_chi2_10_1 = float((hl_nulo_lab >= stats.chi2.isf(0.01, 10)).mean())
    q99_nulo_lab = float(np.quantile(hl_nulo_lab, 0.99))
    return (
        hl_nulo_lab,
        p_lab_29,
        pd_lab,
        q99_nulo_lab,
        tamano_chi2_10_1,
        tamano_chi2_8_1,
        tamano_chi2_8_5,
    )


@app.cell
def _(hl_nulo_lab, np, plt, stats):
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _x = np.linspace(0, 45, 400)
    _ax.hist(hl_nulo_lab, bins=np.arange(0, 60, 1), density=True, color="#bdc3c7", label="HL simulado bajo H0")
    _ax.plot(_x, stats.chi2.pdf(_x, 8), label="χ² con 8 gl")
    _ax.plot(_x, stats.chi2.pdf(_x, 10), label="χ² con 10 gl")
    _ax.axvline(29.04, color="#c0392b", ls="--", label="observado Austral 29,0")
    _ax.set_xlim(0, 45)
    _ax.set_xlabel("estadístico HL")
    _ax.set_ylabel("densidad")
    _ax.set_title("Distribución nula del HL con esperados chicos (Austral de laboratorio)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, p_lab_29, q99_nulo_lab, stats, tamano_chi2_10_1, tamano_chi2_8_1, tamano_chi2_8_5):
    mo.md(rf"""
    **Lectura.** Bajo $H_0$ verdadera, el test $\chi^2_8$ «al 1%» rechaza el **{tamano_chi2_8_1:.1%}** de
    las veces (al 5%: {tamano_chi2_8_5:.1%}); con $\chi^2_{{10}}$ al 1%: {tamano_chi2_10_1:.1%}. El
    cuantil 99% real es {q99_nulo_lab:.1f}, no {stats.chi2.isf(0.01, 8):.1f}. La cola derecha es pesada
    porque los grupos 1–3 esperan 0,1–0,6 malos: un solo malo en un grupo que esperaba 0,106 aporta
    $(1-0{{,}}106)^2/(0{{,}}106\cdot0{{,}}9995)\approx7{{,}}5$ puntos. El p simulado del 29,0 con estas PD
    individuales es **{p_lab_29:.4f}** (curso 0,012 con las PD reales). Conclusión del curso confirmada:
    «un rojo que depende de una aproximación inválida no es un rojo».
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Cartera sintética con verdad conocida

    Scorecard corto sobre `generar_cartera()` (6 variables, WoE del curso, logística en DEV). Master
    scale sintética: 8 bandas con cortes en los octiles del score de DEV (en esta cartera la tasa de malos
    es ~11%, no ~5%, así que los cortes 540/560/… del curso dejarían bandas vacías). Tres «modelos» a
    validar, con verdad conocida:

    - **HO / modelo**: mismo período que DEV. Calibrado por construcción (salvo ruido): el test debería
      aceptar.
    - **OOT / modelo**: el generador planta `deterioro = 0,35` en el log-odds de 2025. Descalibración de
      **nivel** real: el test debería rechazar.
    - **OOT / oráculo**: la PD verdadera completa del generador (incluye el +10% de malos de los «sin
      bureau», que `pd_verdadera` omite). Verdad: el test debería aceptar.
    """)
    return


@app.cell
def _(a_woe, generar_cartera, np, pd, sm, tabla_woe):
    cartera = generar_cartera()
    _vars = ["uso_linea_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m",
             "antiguedad_meses", "carga_financiera", "consultas_6m"]
    _dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    _mapas = {_v: tabla_woe(_dev[_v], _dev["malo"])[0]["woe"].to_dict() for _v in _vars}
    _X = sm.add_constant(a_woe(_dev, _vars, _dev, _mapas))
    _modelo = sm.Logit(_dev["malo"].to_numpy(), _X).fit(disp=0)
    _sin_bureau = cartera["meses_desde_mora_12m"] == -99
    cartera["pd_real"] = np.where(_sin_bureau, cartera["pd_verdadera"] + (1 - cartera["pd_verdadera"]) * 0.10,
                                  cartera["pd_verdadera"])
    cartera["pd_modelo"] = np.nan
    for _m in ["DEV", "HO", "OOT"]:
        _mask = cartera["muestra"] == _m
        _d = cartera[_mask].reset_index(drop=True)
        cartera.loc[_mask, "pd_modelo"] = _modelo.predict(sm.add_constant(a_woe(_d, _vars, _dev, _mapas))).to_numpy()
    cartera["score"] = 487.1229 + 28.8539 * np.log((1 - cartera["pd_modelo"]) / cartera["pd_modelo"])
    # cortes de banda en PD (equivalentes a los octiles del score en DEV), de PD baja a alta
    cortes_pd = np.quantile(cartera.loc[cartera["muestra"] == "DEV", "pd_modelo"], np.linspace(0, 1, 9)[1:-1])
    _resumen = cartera[cartera["muestra"] != "TTD"].groupby("muestra").agg(
        n=("malo", "size"), tasa_obs=("malo", "mean"), pd_modelo=("pd_modelo", "mean"), pd_real=("pd_real", "mean"))
    _resumen.round(4)
    return cartera, cortes_pd


@app.cell
def _(mo):
    selector_grupos = mo.ui.dropdown(
        options=["deciles (10)", "quintiles (5)", "ventiles (20)", "bandas de la master scale (8)", "50 grupos"],
        value="deciles (10)",
        label="Agrupamiento del HL",
    )
    selector_grupos
    return (selector_grupos,)


@app.cell
def _(cartera, cortes_pd, grupos_hl, hl_individual, hl_p_simulado, pd, selector_grupos, stats):
    _conf = {
        "deciles (10)": (10, None), "quintiles (5)": (5, None), "ventiles (20)": (20, None),
        "bandas de la master scale (8)": (8, cortes_pd), "50 grupos": (50, None),
    }[selector_grupos.value]
    _filas = []
    for _nombre, _muestra, _col in [("HO / modelo", "HO", "pd_modelo"), ("OOT / modelo", "OOT", "pd_modelo"),
                                     ("OOT / oráculo", "OOT", "pd_real")]:
        _d = cartera[cartera["muestra"] == _muestra]
        _p, _y = _d[_col].to_numpy(), _d["malo"].to_numpy()
        _g = grupos_hl(_p, _conf[0], _conf[1])
        _stat, _tab, _idx, _n, _pbar = hl_individual(_p, _y, _g)
        _G = len(_n)
        _psim, _ = hl_p_simulado(_p, _idx, _n, _pbar, _stat, S=1000, semilla=11)
        _filas.append({"caso": _nombre, "G": _G, "HL": _stat, "p_chi2_G-2": stats.chi2.sf(_stat, _G - 2),
                       "p_chi2_G": stats.chi2.sf(_stat, _G), "p_simulado": _psim,
                       "min_esperados": float(_tab["esperados"].min())})
    tabla_hl_sint = pd.DataFrame(_filas).set_index("caso")
    tabla_hl_sint.round(4)
    return (tabla_hl_sint,)


@app.cell
def _(cartera, cortes_pd, grupos_hl, hl_individual, hl_p_simulado, pd, stats):
    # Todas las opciones de agrupamiento a la vez, para HO/modelo y OOT/oráculo
    _opciones = [("quintiles", 5, None), ("bandas (8)", 8, cortes_pd), ("deciles", 10, None),
                 ("ventiles", 20, None), ("50 grupos", 50, None)]
    _filas = []
    for _caso, _m, _col in [("HO / modelo", "HO", "pd_modelo"), ("OOT / oráculo", "OOT", "pd_real")]:
        _d = cartera[cartera["muestra"] == _m]
        _p, _y = _d[_col].to_numpy(), _d["malo"].to_numpy()
        for _nom, _g, _c in _opciones:
            _stat, _tab, _idx, _n, _pb = hl_individual(_p, _y, grupos_hl(_p, _g, _c))
            _ps, _ = hl_p_simulado(_p, _idx, _n, _pb, _stat, S=500, semilla=13)
            _filas.append({"caso": _caso, "agrupamiento": _nom, "HL": _stat,
                           "p_chi2_G": float(stats.chi2.sf(_stat, len(_n))), "p_simulado": _ps,
                           "min_esperados": float(_tab["esperados"].min())})
    tabla_agrupamientos = pd.DataFrame(_filas).set_index(["caso", "agrupamiento"])
    tabla_agrupamientos.round(4)
    return (tabla_agrupamientos,)


@app.cell
def _(mo, selector_grupos, tabla_agrupamientos, tabla_hl_sint):
    _t = tabla_hl_sint
    _a = tabla_agrupamientos.loc["HO / modelo", "p_simulado"]
    mo.md(rf"""
    **Lectura ({selector_grupos.value}).** El modelo en OOT da p simulado {_t.loc['OOT / modelo', 'p_simulado']:.3f}:
    el HL detecta el deterioro plantado. El oráculo en OOT da {_t.loc['OOT / oráculo', 'p_simulado']:.3f}.
    El caso interesante es **HO / modelo**: según cómo se agrupe, el p simulado va de {_a.min():.3f} a
    {_a.max():.3f} (tabla de arriba). Mismo modelo, mismos datos, el color depende de una decisión del
    analista. Por eso el agrupamiento (y G) se fija en la política **antes** de mirar. Con 50 grupos los
    esperados mínimos caen a ~1 y la $\chi^2$ vuelve a ser dudosa; con 5 se pierde resolución de forma.
    En esta cartera (tasa ~11–15%) los esperados de los deciles son grandes y $\chi^2_G$ y el simulado
    casi coinciden: la aproximación falla por esperados chicos, no por el test en sí.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Spiegelhalter, Jeffreys y pendiente de calibración

    **Spiegelhalter (1986).** Estandariza el Brier bajo $H_0$:
    $E_0[\text{BS}]=\frac1n\sum p_i(1-p_i)$, $\operatorname{Var}_0[\text{BS}]=\frac1{n^2}\sum(1-2p_i)^2p_i(1-p_i)$,
    $z=(\text{BS}-E_0)/\sqrt{\operatorname{Var}_0}$. Forma equivalente: $z=\sum(y_i-p_i)(1-2p_i)/\sqrt{\sum(1-2p_i)^2p_i(1-p_i)}$,
    porque con $y\in\{0,1\}$, $(y-p)^2-p(1-p)=(y-p)(1-2p)$. Implementación 2: el BS de `sklearn`.

    **Jeffreys (instrucciones de reporte de validación del BCE).** p = $F_{\text{Beta}(D+\frac12,\,n-D+\frac12)}(\text{PD})$,
    una cola; p chico ⇒ PD subestima. Numpy: `betainc` (función beta incompleta regularizada). Librería:
    `scipy.stats.beta.cdf`. Propiedad: queda **entre** $P(D\ge k+1)$ y $P(D\ge k)$ (es casi un mid-p).

    **Pendiente de calibración (Cox 1958).** Regresión logística $\text{logit}\,P(y=1)=a+b\,\text{logit}(p)$;
    $H_0: a=0,b=1$ (LR con 2 gl) y $H_0: b=1$ (Wald). Newton-Raphson numpy vs `statsmodels` GLM.
    """)
    return


@app.cell
def _(betainc, brier_score_loss, np, stats):
    def spiegelhalter_np(p, y):
        p, y = np.asarray(p, float), np.asarray(y, float)
        _num = ((y - p) * (1 - 2 * p)).sum()
        _den = np.sqrt(((1 - 2 * p) ** 2 * p * (1 - p)).sum())
        _z = _num / _den
        return float(_z), float(2 * stats.norm.sf(abs(_z)))

    def spiegelhalter_brier(p, y):
        p, y = np.asarray(p, float), np.asarray(y, float)
        _n = len(p)
        _bs = brier_score_loss(y, p)
        _e0 = (p * (1 - p)).mean()
        _v0 = ((1 - 2 * p) ** 2 * p * (1 - p)).sum() / _n**2
        return float((_bs - _e0) / np.sqrt(_v0))

    def jeffreys_np(k, n, pd_):
        """p de Jeffreys del BCE (una cola) con la beta incompleta regularizada."""
        return float(betainc(k + 0.5, n - k + 0.5, pd_))

    def jeffreys_ic(k, n, conf=0.95):
        _a = (1 - conf) / 2
        _lo = 0.0 if k == 0 else float(stats.beta.ppf(_a, k + 0.5, n - k + 0.5))
        _hi = 1.0 if k == n else float(stats.beta.ppf(1 - _a, k + 0.5, n - k + 0.5))
        return _lo, _hi

    return jeffreys_ic, jeffreys_np, spiegelhalter_brier, spiegelhalter_np


@app.cell
def _(austral_bandas, jeffreys_ic, jeffreys_np, pd, semaforo, stats, tabla_binom):
    _filas = []
    for _b, _f in austral_bandas.iterrows():
        _k, _n, _p = int(_f["malos"]), int(_f["n"]), float(_f["pd"])
        _lo, _hi = jeffreys_ic(_k, _n)
        _filas.append({
            "banda": _b, "pd": _p, "tasa_obs": _k / _n,
            "jeffreys_numpy": jeffreys_np(_k, _n, _p),
            "jeffreys_scipy": float(stats.beta.cdf(_p, _k + 0.5, _n - _k + 0.5)),
            "binom_sup_k": tabla_binom.loc[_b, "sup_numpy"],
            "binom_sup_k+1": float(stats.binom.sf(_k, _n, _p)),
            "ic95_lo": _lo, "ic95_hi": _hi,
        })
    tabla_jeffreys = pd.DataFrame(_filas).set_index("banda")
    tabla_jeffreys["color"] = tabla_jeffreys["jeffreys_numpy"].map(semaforo)
    tabla_jeffreys.round(4)
    return (tabla_jeffreys,)


@app.cell
def _(np, sm, stats):
    def pendiente_calibracion_np(p, y, iters=50):
        """Newton-Raphson para logit P(y=1) = a + b·logit(p). Devuelve (a, b), EE y tests."""
        p, y = np.asarray(p, float), np.asarray(y, float)
        _x = np.log(p / (1 - p))
        _X = np.column_stack([np.ones_like(_x), _x])
        _beta = np.array([0.0, 1.0])
        for _ in range(iters):
            _mu = 1 / (1 + np.exp(-_X @ _beta))
            _W = _mu * (1 - _mu)
            _H = _X.T @ (_X * _W[:, None])
            _paso = np.linalg.solve(_H, _X.T @ (y - _mu))
            _beta = _beta + _paso
            if np.max(np.abs(_paso)) < 1e-12:
                break
        _mu = 1 / (1 + np.exp(-_X @ _beta))
        _cov = np.linalg.inv(_X.T @ (_X * (_mu * (1 - _mu))[:, None]))
        _ee = np.sqrt(np.diag(_cov))
        _ll = lambda m: float((y * np.log(m) + (1 - y) * np.log(1 - m)).sum())
        _lr = 2 * (_ll(_mu) - _ll(p))  # H0: a=0, b=1 ⇒ la PD original
        return {
            "a": float(_beta[0]), "b": float(_beta[1]), "ee_b": float(_ee[1]),
            "p_wald_b1": float(2 * stats.norm.sf(abs((_beta[1] - 1) / _ee[1]))),
            "p_lr_a0b1": float(stats.chi2.sf(_lr, 2)),
        }

    def pendiente_calibracion_sm(p, y):
        _x = np.log(np.asarray(p) / (1 - np.asarray(p)))
        _fit = sm.GLM(np.asarray(y, float), sm.add_constant(_x), family=sm.families.Binomial()).fit()
        return np.asarray(_fit.params), np.asarray(_fit.bse)

    return pendiente_calibracion_np, pendiente_calibracion_sm


@app.cell
def _(
    cartera,
    cortes_pd,
    grupos_hl,
    hl_individual,
    hl_p_simulado,
    jeffreys_np,
    np,
    pd,
    pendiente_calibracion_np,
    pvalores_binom_np,
    semaforo,
    spiegelhalter_np,
    stats,
):
    def _bateria(p, y, verdad, S=1000):
        """Todos los tests de calibración sobre un vector de PD y su desempeño.
        `verdad` (PD real del generador) solo se usa para las dos columnas de referencia."""
        p, y, verdad = np.asarray(p, float), np.asarray(y, float), np.asarray(verdad, float)
        _n, _D = len(y), int(y.sum())
        _pbar = float(p.mean())
        _bandas = grupos_hl(p, cortes=cortes_pd)
        _colores = []
        for _b in range(8):
            _m = _bandas == _b
            if _m.sum() == 0:
                continue
            _colores.append(semaforo(pvalores_binom_np(int(y[_m].sum()), int(_m.sum()), float(p[_m].mean()))["dos_minlike"]))
        _stat, _t, _idx, _ng, _pg = hl_individual(p, y, grupos_hl(p, 10))
        _psim, _ = hl_p_simulado(p, _idx, _ng, _pg, _stat, S=S, semilla=5)
        _sp = spiegelhalter_np(p, y)
        _sl = pendiente_calibracion_np(p, y)
        return {
            "n": _n, "tasa_obs": _D / _n, "pd_media": _pbar, "O/E": _D / p.sum(),
            "p_global": float(stats.binomtest(_D, _n, _pbar).pvalue),
            "p_jeffreys": jeffreys_np(_D, _n, _pbar),
            "bandas 🟡/🔴": f"{_colores.count('🟡')}/{_colores.count('🔴')}",
            "HL": _stat, "p_HL_chi2_10": float(stats.chi2.sf(_stat, 10)), "p_HL_sim": _psim,
            "z_spiegel": _sp[0], "p_spiegel": _sp[1],
            "pendiente_b": _sl["b"], "p_LR_a0b1": _sl["p_lr_a0b1"],
            # referencias con verdad conocida (imposibles en datos reales)
            "O/E_verdad": verdad.sum() / p.sum(),
            "pendiente_verdad": float(np.polyfit(np.log(p / (1 - p)), np.log(verdad / (1 - verdad)), 1)[0]),
        }

    _oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)
    _ho = cartera[cartera["muestra"] == "HO"]
    # Recalibración de nivel con OOT ene–mar 2025 y validación en abr–jun 2025 (muestra NO usada)
    _cal = _oot["cohorte"] <= "2025-03"
    _lp = np.log(_oot["pd_modelo"] / (1 - _oot["pd_modelo"]))
    _tasa_cal = _oot.loc[_cal, "malo"].mean()
    _lo_d, _hi_d = -3.0, 3.0
    for _ in range(100):  # bisección para δ exacto (media de PD = tasa de calibración)
        _mid = 0.5 * (_lo_d + _hi_d)
        if (1 / (1 + np.exp(-(_lp[_cal] + _mid)))).mean() < _tasa_cal:
            _lo_d = _mid
        else:
            _hi_d = _mid
    delta_recal = 0.5 * (_lo_d + _hi_d)
    _pd_recal = 1 / (1 + np.exp(-(_lp + delta_recal)))
    casos_bateria = {
        "HO / modelo": (_ho["pd_modelo"], _ho["malo"], _ho["pd_real"]),
        "HO / oráculo": (_ho["pd_real"], _ho["malo"], _ho["pd_real"]),
        "OOT / modelo": (_oot["pd_modelo"], _oot["malo"], _oot["pd_real"]),
        "OOT / oráculo": (_oot["pd_real"], _oot["malo"], _oot["pd_real"]),
        "OOT ene–mar / δ calibrado ahí (misma muestra)": (_pd_recal[_cal], _oot.loc[_cal, "malo"], _oot.loc[_cal, "pd_real"]),
        "OOT abr–jun / δ de ene–mar (fuera de muestra)": (_pd_recal[~_cal], _oot.loc[~_cal, "malo"], _oot.loc[~_cal, "pd_real"]),
    }
    tabla_bateria = pd.DataFrame({_k: _bateria(*_v) for _k, _v in casos_bateria.items()}).T
    tabla_bateria
    return casos_bateria, delta_recal, tabla_bateria


@app.cell
def _(delta_recal, mo, tabla_bateria):
    _t = tabla_bateria
    _ho, _hoo, _oo, _ooo = (_t.loc[_c] for _c in ["HO / modelo", "HO / oráculo", "OOT / modelo", "OOT / oráculo"])
    _mm = _t.loc["OOT ene–mar / δ calibrado ahí (misma muestra)"]
    _fm = _t.loc["OOT abr–jun / δ de ene–mar (fuera de muestra)"]
    mo.md(rf"""
    **Lectura de la batería** (las dos últimas columnas usan la verdad del generador; en la vida real no existen).

    - **OOT / modelo**: O/E {_oo['O/E']:.3f} (verdad: {_oo['O/E_verdad']:.3f}). Todos los tests de nivel
      explotan (p global {_oo['p_global']:.1e}) y el HL también. Diagnóstico: problema de **nivel**, que es
      el deterioro plantado (desplazamiento constante del log-odds).
    - **HO / modelo**: el nivel pasa (O/E {_ho['O/E']:.3f}, p global {_ho['p_global']:.2f}), pero el HL
      simulado da {_ho['p_HL_sim']:.3f} y la pendiente {_ho['pendiente_b']:.3f} (LR p {_ho['p_LR_a0b1']:.3f}).
      ¿Falsa alarma? No: la pendiente **verdadera** del modelo contra la PD real es
      {_ho['pendiente_verdad']:.3f}. El scorecard de 6 variables comprime las PD (no captura todas las no
      linealidades del generador) y la prueba de pendiente lo ve con n = {int(_ho['n']):,}.
    - **HO / oráculo** (la verdad misma): p global {_hoo['p_global']:.2f}, pendiente p {_hoo['p_LR_a0b1']:.2f},
      pero HL simulado **{_hoo['p_HL_sim']:.3f}** y una banda {_hoo['bandas 🟡/🔴']} (🟡/🔴). La verdad también
      enciende amarillos: al 5%, 1 de cada 20 veces. Con una batería de ~6 tests × 8 bandas, algún
      amarillo en un modelo perfecto es la norma (§8).
    - **OOT / oráculo**: pasa todo (p global {_ooo['p_global']:.2f}, HL sim {_ooo['p_HL_sim']:.3f}).
    - **Misma muestra vs fuera de muestra**: con δ = {delta_recal:+.3f} ajustado en ene–mar, el binomial
      global en ene–mar da p = {_mm['p_global']:.3f} **por construcción** (M15). En abr–jun, la prueba
      honesta: O/E {_fm['O/E']:.3f}, p global {_fm['p_global']:.3f}, Jeffreys {_fm['p_jeffreys']:.3f}.
      El δ se estimó en un trimestre cuya tasa realizada quedó bajo la verdadera (O/E verdad en ene–mar
      {_mm['O/E_verdad']:.3f}; en abr–jun {_fm['O/E_verdad']:.3f}): la calibración hereda el ruido de su
      muestra y solo la muestra siguiente lo revela.
    - Spiegelhalter mezcla nivel y forma en un solo $z$; se reporta al lado de la pendiente y del
      binomial global, no en vez de ellos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Multiplicidad: 8 bandas, un tablero

    Con 8 tests independientes al 5% y $H_0$ cierta en todos, $P(\ge1 \text{ amarillo})=1-0{,}95^8=33{,}7\%$
    si cada test tuviera tamaño exacto 5%. Con tests binomiales exactos (discretos) el tamaño real es menor
    y la FWER real también. Correcciones: Bonferroni ($\min(1,m\,p)$), Holm (escalonada, controla FWER,
    uniformemente más potente que Bonferroni), Benjamini-Hochberg (controla FDR, no FWER).
    El curso propone otra vía: **leer el patrón**. Un test de signos formaliza una parte: bajo $H_0$ cada
    banda subestima o sobreestima con probabilidad ~½.
    """)
    return


@app.cell
def _(mo):
    selector_ajuste = mo.ui.dropdown(
        options={"sin ajuste": "none", "Bonferroni": "bonferroni", "Holm": "holm", "Benjamini-Hochberg": "fdr_bh"},
        value="Holm",
        label="Corrección por multiplicidad",
    )
    selector_ajuste
    return (selector_ajuste,)


@app.cell
def _(austral_bandas, multipletests, np, pd, pmf_binom_np, selector_ajuste, semaforo, stats, tabla_binom):
    def ajustar_np(p, metodo):
        """Bonferroni / Holm / BH desde cero (numpy)."""
        p = np.asarray(p, float)
        _m = len(p)
        if metodo == "none":
            return p.copy()
        if metodo == "bonferroni":
            return np.minimum(1.0, _m * p)
        _o = np.argsort(p)
        _ps = p[_o]
        if metodo == "holm":
            _adj = np.maximum.accumulate((_m - np.arange(_m)) * _ps)
        elif metodo == "fdr_bh":
            _adj = np.minimum.accumulate((_m / np.arange(1, _m + 1) * _ps)[::-1])[::-1]
        else:
            raise ValueError(metodo)
        _res = np.empty(_m)
        _res[_o] = np.minimum(1.0, _adj)
        return _res

    _p = tabla_binom["minlike_numpy"].to_numpy()
    _met = selector_ajuste.value
    _adj_np = ajustar_np(_p, _met)
    _adj_sm = _p.copy() if _met == "none" else multipletests(_p, method=_met)[1]
    tabla_multi = pd.DataFrame({"p": _p, "p_ajustado_numpy": _adj_np, "p_ajustado_statsmodels": _adj_sm},
                               index=tabla_binom.index)
    tabla_multi["color"] = tabla_multi["p_ajustado_numpy"].map(semaforo)

    # tamaño real de cada test (minlike al 5%) y FWER exacta bajo H0 independiente
    _tam = []
    for _b, _f in austral_bandas.iterrows():
        _pmf = pmf_binom_np(int(_f["n"]), float(_f["pd"]))
        _pv = np.array([_pmf[_pmf <= _pmf[_j] * (1 + 1e-7)].sum() for _j in range(len(_pmf))])
        _tam.append(float(_pmf[_pv <= 0.05].sum()))
    tamanos_reales = pd.Series(_tam, index=austral_bandas.index)
    fwer_exacta = 1 - float(np.prod(1 - tamanos_reales.to_numpy()))
    # test de signos: bandas con observados > esperados
    _sub = int((austral_bandas["malos"] > austral_bandas["esperados"]).sum())
    signos_sub = _sub
    p_signos = float(stats.binomtest(_sub, 8, 0.5, alternative="greater").pvalue)
    tabla_multi.round(4)
    return ajustar_np, fwer_exacta, p_signos, signos_sub, tamanos_reales


@app.cell
def _(fwer_exacta, mo, p_signos, selector_ajuste, signos_sub, tamanos_reales):
    mo.md(rf"""
    **Lectura ({selector_ajuste.value}).** Con Holm o Bonferroni **ninguna** banda de Austral queda
    amarilla (B2 ajustado ≈ 0,16). Tamaños reales del minlike al 5%: de {tamanos_reales.min():.3f} a
    {tamanos_reales.max():.3f}; la FWER exacta bajo $H_0$ es **{fwer_exacta:.1%}**, no 33,7%. Test de
    signos: {signos_sub} de 8 bandas con más malos que los esperados, p = {p_signos:.3f} (una cola). Tampoco
    concluyente por sí solo. El patrón del curso (las dos amarillas son bandas **buenas** que
    **subestiman**, más el HL, más el CSI, más el swap-in) es evidencia acumulada de varias fuentes
    parcialmente independientes; ningún test individual la contiene. La lección operativa: se declara
    *antes* qué lectura manda (por banda con corrección, o patrón con test de signos/pendiente) y se
    reporta el resto como diagnóstico.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Checks del módulo

    Si algún `assert` falla, el notebook falla.
    """)
    return


@app.cell
def _(
    ajustar_np,
    aporte_g3_g5,
    austral_bandas,
    casos_bateria,
    critico_sup_np,
    error_mc_rho,
    hl_austral,
    hl_austral_scipy,
    hl_desde_agregados,
    hl_scipy_2xg,
    jeffreys_np,
    mo,
    multipletests,
    np,
    p_global_austral,
    p_hl_austral_g2,
    p_hl_austral_sim_homog,
    p_lab_29,
    p_sup_vasicek_np,
    p_sup_vasicek_quad,
    pendiente_calibracion_np,
    pendiente_calibracion_sm,
    pvalores_binom_np,
    spiegelhalter_brier,
    spiegelhalter_np,
    stats,
    tabla_bateria,
    tabla_binom,
    tabla_jeffreys,
    tabla_potencia,
    tabla_rho,
    tamano_chi2_8_1,
    tasche_indep,
    tasche_rho5,
    tasche_rho5_quad,
):
    _checks = []

    # 1. Binomial numpy = scipy (tres alternativas) en Austral y en casos al azar
    assert np.allclose(tabla_binom["sup_numpy"], tabla_binom["sup_scipy"], rtol=1e-9)
    assert np.allclose(tabla_binom["inf_numpy"], tabla_binom["inf_scipy"], rtol=1e-9)
    assert np.allclose(tabla_binom["minlike_numpy"], tabla_binom["minlike_scipy"], rtol=1e-9)
    _rng = np.random.default_rng(0)
    for _ in range(40):
        _n = int(_rng.integers(5, 800)); _p = float(_rng.uniform(0.001, 0.4)); _k = int(_rng.integers(0, _n + 1))
        _v = pvalores_binom_np(_k, _n, _p)
        assert np.isclose(_v["sup"], stats.binomtest(_k, _n, _p, alternative="greater").pvalue, rtol=1e-8, atol=1e-12)
        assert np.isclose(_v["dos_minlike"], stats.binomtest(_k, _n, _p).pvalue, rtol=1e-8, atol=1e-12)
    _checks.append("binomial numpy = scipy.binomtest (greater, less, two-sided)")

    # 2. Reproduce la tabla del curso (tolerancia por PD redondeadas en la lámina)
    assert np.allclose(tabla_binom["minlike_numpy"], austral_bandas["p_curso"], atol=0.006)
    assert np.isclose(tabla_binom.loc["B2", "minlike_numpy"], 0.020, atol=0.0006)
    assert np.isclose(tabla_binom.loc["B2", "minlike_numpy"], tabla_binom.loc["B2", "sup_numpy"])
    assert abs(p_global_austral - 0.562) < 0.002
    assert tabla_binom.loc["A2", "color_normal"] == "🔴" and tabla_binom.loc["A2", "color_minlike"] == "🟡"
    _checks.append("tabla Austral reproducida; B2 bilateral = una cola; normal da rojo falso en A2")

    # 3. Potencia: crítico numpy = scipy; tamaño real ≤ 5%; B2 crítico = 8
    assert (tabla_potencia["k_critico"] == tabla_potencia["k_critico_scipy"]).all()
    assert (tabla_potencia["tamano_real"] <= 0.05).all()
    assert np.allclose(tabla_potencia["potencia_m"], tabla_potencia["potencia_m_scipy"], rtol=1e-8)
    assert tabla_potencia.loc["B2", "k_critico"] == 8
    _checks.append("potencia: crítico numpy = scipy; tamaño ≤ 5%")

    # 4. Vasicek: Gauss-Hermite = quad; Tasche (2006) reproducido; ρ=0 ⇒ binomial; monotonía en ρ
    assert np.isclose(tasche_rho5, tasche_rho5_quad, rtol=1e-6)
    for _n, _p, _k, _r in [(2004, 0.0565, 119, 0.05), (263, 0.2691, 64, 0.2), (235, 0.0142, 8, 0.1)]:
        assert abs(p_sup_vasicek_np(_k, _n, _p, _r) - p_sup_vasicek_quad(_k, _n, _p, _r)) < 1e-7
    assert abs(tasche_indep - 0.007) < 0.0005 and abs(tasche_rho5 - 0.111) < 0.001
    for _b in ["A2", "B2", "E"]:
        _n, _p, _k = int(austral_bandas.loc[_b, "n"]), float(austral_bandas.loc[_b, "pd"]), int(austral_bandas.loc[_b, "malos"])
        assert np.isclose(p_sup_vasicek_np(_k, _n, _p, 1e-9), stats.binom.sf(_k - 1, _n, _p), rtol=1e-5)
        _kc = critico_sup_np(_n, _p)
        _r = [p_sup_vasicek_np(_kc, _n, _p, _x) for _x in (0.0, 0.02, 0.05, 0.1)]
        assert all(np.diff(_r) > 0)
    assert error_mc_rho < 0.03
    # en las bandas con exceso claro de malos, la cola superior engorda con ρ (en A1 y C1, pegadas
    # a la media, no: la mezcla también engorda P(D = 0) y puede bajar P(D >= k))
    _sel = ["A2", "B1", "B2", "D"]
    assert (tabla_rho.loc[_sel, "p_obs_ajustado"] >= tabla_rho.loc[_sel, "p_obs_indep"] - 1e-12).all()
    _checks.append("Vasicek: trapecio = quad; Tasche 0,7% / 11,1%; rechazo crece con ρ; simulación = exacto")

    # 5. HL: numpy = scipy 2×G; lámina reproducida; χ² sobre-rechaza con esperados chicos
    assert np.isclose(hl_austral, hl_austral_scipy, rtol=1e-10)
    assert abs(hl_austral - 29.0) < 0.1 and abs(aporte_g3_g5 - 24.3) < 0.05
    assert abs(p_hl_austral_g2 - 0.0003) < 0.00005
    assert 0.005 < p_hl_austral_sim_homog < 0.03 and 0.003 < p_lab_29 < 0.03
    assert tamano_chi2_8_1 > 0.015
    _rng2 = np.random.default_rng(1)
    for _ in range(10):
        _n = _rng2.integers(20, 300, 6); _pb = _rng2.uniform(0.01, 0.5, 6); _o = _rng2.binomial(_n, _pb)
        assert np.isclose(hl_desde_agregados(_n, _pb, _o)[0], hl_scipy_2xg(_n, _pb, _o), rtol=1e-10)
    _checks.append("HL numpy = scipy χ² 2×G; 29,0 y 24,3 reproducidos; p sim ≫ p χ²")

    # 6. Spiegelhalter: dos formas = ; Jeffreys numpy = scipy y entre las colas binomiales
    for _p_, _y_, _v_ in casos_bateria.values():
        assert np.isclose(spiegelhalter_np(_p_, _y_)[0], spiegelhalter_brier(_p_, _y_), rtol=1e-8)
    assert np.allclose(tabla_jeffreys["jeffreys_numpy"], tabla_jeffreys["jeffreys_scipy"], rtol=1e-10)
    assert (tabla_jeffreys["binom_sup_k+1"] <= tabla_jeffreys["jeffreys_numpy"]).all()
    assert (tabla_jeffreys["jeffreys_numpy"] <= tabla_jeffreys["binom_sup_k"]).all()
    assert np.isclose(jeffreys_np(8, 235, 0.0142), tabla_jeffreys.loc["B2", "jeffreys_numpy"])
    _checks.append("Spiegelhalter (2 formas) y Jeffreys (betainc = beta.cdf; entre colas)")

    # 7. Pendiente de calibración: Newton numpy = statsmodels GLM
    for _p_, _y_, _v_ in list(casos_bateria.values())[:4]:
        _r = pendiente_calibracion_np(_p_, _y_)
        _par, _ee = pendiente_calibracion_sm(_p_, _y_)
        assert np.allclose([_r["a"], _r["b"]], _par, atol=1e-7) and np.isclose(_r["ee_b"], _ee[1], rtol=1e-5)
    _checks.append("pendiente: Newton numpy = statsmodels GLM")

    # 8. Verdad conocida: el oráculo pasa, el modelo en OOT falla, misma muestra da p≈1
    _t = tabla_bateria
    assert _t.loc["OOT / oráculo", "p_global"] > 0.05 and _t.loc["OOT / modelo", "p_global"] < 1e-6
    assert _t.loc["OOT / modelo", "O/E"] > 1.15
    assert _t.loc["OOT ene–mar / δ calibrado ahí (misma muestra)", "p_global"] > 0.95
    assert np.isclose(_t.loc["OOT / oráculo", "pendiente_verdad"], 1.0)
    _checks.append("oráculo pasa; deterioro plantado detectado; misma muestra ⇒ p≈1")

    # 9. Multiplicidad: numpy = statsmodels para los tres métodos
    _pp = tabla_binom["minlike_numpy"].to_numpy()
    for _m in ["bonferroni", "holm", "fdr_bh"]:
        assert np.allclose(ajustar_np(_pp, _m), multipletests(_pp, method=_m)[1])
    _pr = np.random.default_rng(3).uniform(0, 0.2, 12)
    for _m in ["bonferroni", "holm", "fdr_bh"]:
        assert np.allclose(ajustar_np(_pr, _m), multipletests(_pr, method=_m)[1])
    assert np.isclose(1 - 0.95**8, 0.3366, atol=1e-4)
    _checks.append("Bonferroni/Holm/BH numpy = statsmodels")

    mo.md("**Todos los checks pasaron.**\n\n" + "\n".join(f"- {c}" for c in _checks))
    return


if __name__ == "__main__":
    app.run()

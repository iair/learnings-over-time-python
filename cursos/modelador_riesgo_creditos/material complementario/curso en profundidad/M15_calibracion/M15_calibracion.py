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
    from scipy.optimize import brentq
    from scipy.special import expit, logit, betaincinv
    from scipy.stats import binomtest, binom
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.calibration import calibration_curve
    from statsmodels.stats.proportion import proportion_confint
    return (
        IsotonicRegression,
        LogisticRegression,
        betaincinv,
        binom,
        binomtest,
        brentq,
        brier_score_loss,
        calibration_curve,
        expit,
        log_loss,
        logit,
        mo,
        plt,
        proportion_confint,
        roc_auc_score,
        sm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # M15 · Calibración: nivel vs ranking

    **Serie 2 · Del embudo al gobierno.** Este notebook acompaña a `M15_calibracion.md`.

    Un score puede **ordenar** perfecto y aun así **mentir en el nivel**. Aquí se construye toda la
    mecánica de calibración sobre una cartera sintética con **verdad conocida** (`pd_verdadera` del
    generador), algo que en datos reales nunca se tiene:

    1. Pipeline mínimo y por qué el intercepto de máxima verosimilitud clava la media en DEV.
    2. Tendencia central (TC) y ventana PIT desde la tabla de cosechas.
    3. Ajuste de intercepto: δ aproximado (Siddiqi) vs exacto (bisección numpy vs `brentq`) y la
       fórmula del error: **δ_S = δ·(1 − κ̄)**, con κ = Var(p)/[p̄(1−p̄)].
    4. Experimento de Jensen: dispersión de las PD vs error de la aproximación.
    5. Corrección por sobremuestreo (*prior shift*, King & Zeng 2001).
    6. Recalibración logística intercepto + pendiente (IRLS numpy vs `statsmodels` GLM) y Platt.
    7. Isotónica (PAV numpy vs `sklearn`) y por qué rompe la escala PDO.
    8. Métricas: CITL, O/E, pendiente, Brier y su descomposición de Murphy, log-loss, ECE.
    9. Curvas de calibración con IC de Wilson/Jeffreys y el grupo 5 de OOT de Banco Austral.
    10. PIT vs TTC operativo y el efecto de δ sobre el score y el cutoff.
    11. «Calibrar y validar en la misma muestra»: p = 1 por construcción.
    12. Checks del módulo.

    Convenciones del curso: target **1 = malo**; WoE = ln(%buenos/%malos); PDO 20, score 600 a
    odds 50:1 (factor 28,8539; offset 487,1229).
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
    ## 1. Pipeline mínimo y el intercepto que clava la media

    Construimos un scorecard corto (6 variables, binning y WoE del curso, logística sobre WoE) y
    calculamos la PD en DEV/HO/OOT/TTD. Agregamos dos columnas que la vida real no tiene:

    - `pd_real`: la PD verdadera **completa** del generador. `pd_verdadera` no incluye el 10% de
      malos extra que el generador planta en los «sin bureau» (−99); para ellos
      $p^{real} = p + (1-p)\cdot 0{,}10$.
    - `malo_futuro`: para TTD, un desempeño simulado desde `pd_real` (el «oráculo» de lo que se
      observará en 12 meses). Es la muestra **posterior y madura** que el curso pide para validar
      una calibración PIT, y que en la clase no existía todavía.

    Qué mirar: en DEV la PD media es **igual** a la tasa observada hasta 10 decimales. No es mérito:
    es la ecuación de primer orden del intercepto, $\sum_i (y_i - p_i) = 0$.
    """)
    return


@app.cell
def _(a_woe, generar_cartera, logit, np, pd, sm, tabla_woe):
    cartera = generar_cartera()
    cartera["pd_real"] = np.where(
        cartera["meses_desde_mora_12m"] == -99,
        cartera["pd_verdadera"] + (1 - cartera["pd_verdadera"]) * 0.10,
        cartera["pd_verdadera"],
    )
    _rng = np.random.default_rng(1515)
    cartera["malo_futuro"] = np.where(
        cartera["muestra"] == "TTD",
        (_rng.random(len(cartera)) < cartera["pd_real"]).astype(float),
        np.nan,
    )

    VARIABLES = ["uso_linea_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m",
                 "antiguedad_meses", "carga_financiera", "consultas_6m"]
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    mapas_woe = {v: tabla_woe(dev[v], dev["malo"])[0]["woe"].to_dict() for v in VARIABLES}
    X_dev = sm.add_constant(a_woe(dev, VARIABLES, dev, mapas_woe))
    modelo = sm.Logit(dev["malo"].values, X_dev).fit(disp=0)

    def pd_del_modelo(df_):
        """PD cruda (sin calibrar) del scorecard para cualquier muestra."""
        _X = sm.add_constant(a_woe(df_, VARIABLES, dev, mapas_woe), has_constant="add")
        return np.asarray(modelo.predict(_X), dtype=float)

    muestras = {}
    for _m in ["DEV", "HO", "OOT", "TTD"]:
        _d = cartera[cartera["muestra"] == _m].reset_index(drop=True).copy()
        _d["pd_modelo"] = pd_del_modelo(_d)
        _d["lp_modelo"] = logit(_d["pd_modelo"].values)
        muestras[_m] = _d

    coeficientes = pd.DataFrame({"beta": modelo.params, "se": modelo.bse}).round(4)
    coeficientes
    return VARIABLES, X_dev, cartera, coeficientes, dev, modelo, muestras


@app.cell
def _(X_dev, dev, muestras, np, pd, roc_auc_score):
    _filas = []
    for _m, _d in muestras.items():
        _obs = _d["malo_futuro"].mean() if _m == "TTD" else _d["malo"].mean()
        _y = _d["malo_futuro"] if _m == "TTD" else _d["malo"]
        _filas.append({
            "muestra": _m + (" (oráculo)" if _m == "TTD" else ""),
            "n": len(_d),
            "pd_media_modelo": _d["pd_modelo"].mean(),
            "tasa_observada": _obs,
            "pd_real_media": _d["pd_real"].mean(),
            "sesgo_pp": 100 * (_d["pd_modelo"].mean() - _obs),
            "gini": 2 * roc_auc_score(_y, _d["pd_modelo"]) - 1,
        })
    tabla_nivel = pd.DataFrame(_filas).set_index("muestra")

    # ecuaciones de primer orden (score equations) evaluadas en el MLE, en numpy puro
    _p = muestras["DEV"]["pd_modelo"].values
    _y = dev["malo"].values
    gradiente_foc = X_dev.values.T @ (_y - _p)          # un componente por parámetro
    brecha_media_dev = abs(_p.mean() - _y.mean())
    tabla_nivel.round(4)
    return brecha_media_dev, gradiente_foc, tabla_nivel


@app.cell
def _(X_dev, brecha_media_dev, gradiente_foc, mo, np, tabla_nivel):
    _t = tabla_nivel
    mo.md(f"""
    **Lectura.** En DEV: PD media {_t.loc['DEV','pd_media_modelo']:.4%} = tasa {_t.loc['DEV','tasa_observada']:.4%}
    (brecha {brecha_media_dev:.1e}). El gradiente de la log-verosimilitud en el MLE es
    ≈ 0 en **todas** las coordenadas (máx |∂ℓ/∂β| = {np.max(np.abs(gradiente_foc)):.1e}); la de la constante es
    exactamente $\\sum_i (y_i-p_i)$. Las otras {X_dev.shape[1]-1} dicen algo más fuerte: la PD también clava la media
    **ponderada por el WoE** de cada variable. Y un detalle que solo la verdad revela: la PD real media de DEV es
    {_t.loc['DEV','pd_real_media']:.2%}, {100*(_t.loc['DEV','pd_real_media']-_t.loc['DEV','tasa_observada']):.2f} pp sobre lo observado
    (≈ {(_t.loc['DEV','pd_real_media']-_t.loc['DEV','tasa_observada'])/np.sqrt(_t.loc['DEV','tasa_observada']*(1-_t.loc['DEV','tasa_observada'])/_t.loc['DEV','n']):.1f}
    errores estándar: ruido de muestreo). El intercepto clava la media **de la muestra**, ruido incluido.

    Fuera de DEV el modelo subestima: OOT {_t.loc['OOT','pd_media_modelo']:.2%} predicho vs {_t.loc['OOT','tasa_observada']:.2%}
    observado ({_t.loc['OOT','sesgo_pp']:+.2f} pp). La verdad del generador confirma que no es ruido: la PD real media de OOT es
    {_t.loc['OOT','pd_real_media']:.2%} (el generador sube el log-odds en 0,35 desde 2025-01). El Gini casi no se mueve
    entre muestras: **el ranking sobrevive, el nivel no**. Es el mismo patrón de Banco Austral (4,97% → 5,94%), con otra
    escala: Banco Sintético tiene tasa de malos ~11%, no ~5%.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.1 Cuándo la media NO clava en DEV

    La igualdad $\bar p = \bar y$ requiere tres cosas: intercepto libre, **sin penalizar**, y
    verosimilitud **sin ponderar**. Tres ajustes comunes la rompen:

    - `liblinear` de scikit-learn **penaliza el intercepto** (lo trata como una columna más, vía
      `intercept_scaling`): con regularización fuerte la media se aleja de la tasa.
    - `lbfgs` con L2 no penaliza el intercepto: la media se mantiene aunque los β se encojan.
    - Con pesos (p. ej. ponderar buenos ×2 para deshacer un submuestreo), la igualdad vale para la
      media **ponderada**, no la simple.
    """)
    return


@app.cell
def _(LogisticRegression, X_dev, dev, np, pd):
    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    _X = X_dev.drop(columns="const").values
    _y = dev["malo"].values
    _casos = {
        "sklearn sin penalizar (C=∞)": LogisticRegression(C=np.inf, max_iter=2000),
        "sklearn lbfgs, L2 C=0,01 (intercepto libre)": LogisticRegression(C=0.01, max_iter=2000),
        "sklearn liblinear, L2 C=0,01 (penaliza intercepto)": LogisticRegression(solver="liblinear", C=0.01),
    }
    _filas = []
    for _nombre, _clf in _casos.items():
        _clf.fit(_X, _y)
        _p = _clf.predict_proba(_X)[:, 1]
        _filas.append({"ajuste": _nombre, "pd_media": _p.mean(), "tasa_dev": _y.mean(),
                       "brecha_pp": 100 * (_p.mean() - _y.mean())})
    _w = np.where(_y == 1, 1.0, 2.0)
    _clf = LogisticRegression(C=np.inf, max_iter=2000).fit(_X, _y, sample_weight=_w)
    _p = _clf.predict_proba(_X)[:, 1]
    _filas.append({"ajuste": "pesos buenos×2: media simple", "pd_media": _p.mean(),
                   "tasa_dev": _y.mean(), "brecha_pp": 100 * (_p.mean() - _y.mean())})
    _filas.append({"ajuste": "pesos buenos×2: media ponderada", "pd_media": np.average(_p, weights=_w),
                   "tasa_dev": np.average(_y, weights=_w),
                   "brecha_pp": 100 * (np.average(_p, weights=_w) - np.average(_y, weights=_w))})
    tabla_foc_falla = pd.DataFrame(_filas).set_index("ajuste")
    tabla_foc_falla.round(4)
    return (tabla_foc_falla,)


@app.cell
def _(mo, tabla_foc_falla):
    _b = tabla_foc_falla["brecha_pp"]
    mo.md(f"""
    **Lectura.** `liblinear` con C=0,01 descuadra la media en {_b.iloc[2]:+.2f} pp: un pipeline que cambia de
    solver «por velocidad» cambia el nivel de todas las PD sin tocar una línea de calibración. Con pesos, la media
    simple queda en {_b.iloc[3]:+.2f} pp y la ponderada en {_b.iloc[4]:+.4f} pp: el intercepto clava la media
    **de la verosimilitud que se maximizó**. Regla de pipeline: el nivel implícito del modelo base se declara
    (solver, penalización, pesos) antes de calibrar.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. La tendencia central y la ventana PIT, desde la tabla de cosechas

    - **TC (TTC del curso)**: promedio **simple** de la tasa de malos de las cosechas maduras (aquí 24,
      de 2023-07 a 2025-06). Cada cosecha pesa igual: mirada de ciclo.
    - **PIT**: la tasa observada del periodo **reciente y maduro**: las últimas 4 cosechas maduras
      (2025-03 a 2025-06), igual que el curso con Banco Austral (mar–jun 2025).

    El gráfico agrega lo que en la vida real no existe: la PD real media de cada cosecha
    (línea gris) y la PD del modelo sin calibrar.
    """)
    return


@app.cell
def _(cartera, muestras, pd):
    _hist = cartera[cartera["muestra"] != "TTD"]
    tabla_cosechas = _hist.groupby("cohorte").agg(
        n=("malo", "size"), malos=("malo", "sum"), tasa=("malo", "mean"),
        pd_real_media=("pd_real", "mean"))
    _pdm = pd.concat([muestras[m][["cohorte", "pd_modelo"]] for m in ["DEV", "HO", "OOT"]])
    tabla_cosechas["pd_modelo_media"] = _pdm.groupby("cohorte")["pd_modelo"].mean()

    TC = float(tabla_cosechas["tasa"].mean())                         # promedio simple (curso)
    tc_ponderada = float(_hist["malo"].mean())
    COSECHAS_PIT = ["2025-03", "2025-04", "2025-05", "2025-06"]
    _oot = muestras["OOT"]
    es_pit = _oot["cohorte"].isin(COSECHAS_PIT).values
    TASA_PIT = float(_oot.loc[es_pit, "malo"].mean())
    tasa_dev = float(muestras["DEV"]["malo"].mean())
    return COSECHAS_PIT, TASA_PIT, TC, es_pit, tabla_cosechas, tasa_dev, tc_ponderada


@app.cell
def _(TASA_PIT, TC, plt, tabla_cosechas, tasa_dev):
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _x = range(len(tabla_cosechas))
    _ax.plot(_x, 100 * tabla_cosechas["tasa"], "o-", color="#1f4e79", ms=4, label="tasa observada por cosecha")
    _ax.plot(_x, 100 * tabla_cosechas["pd_real_media"], "-", color="0.55", lw=2, label="PD real media (verdad)")
    _ax.plot(_x, 100 * tabla_cosechas["pd_modelo_media"], "--", color="#b3261e", label="PD del modelo sin calibrar")
    _ax.axhline(100 * TC, color="#2e7d32", ls=":", label=f"TC simple = {TC:.2%}")
    _ax.axhline(100 * TASA_PIT, color="#ef6c00", ls=":", label=f"tasa PIT mar–jun 2025 = {TASA_PIT:.2%}")
    _ax.axhline(100 * tasa_dev, color="#b3261e", ls=(0, (1, 3)), lw=1)
    _ax.set_xticks(list(_x)[::3], tabla_cosechas.index[::3], rotation=0, fontsize=8)
    _ax.set_ylabel("tasa de malos (%)")
    _ax.set_xlabel("cosecha (mes de solicitud)")
    _ax.set_title("Tabla de cosechas: TC, ventana PIT y PD del modelo")
    _ax.legend(fontsize=7, ncol=2, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(TASA_PIT, TC, mo, tasa_dev, tc_ponderada):
    mo.md(f"""
    **Lectura.** TC simple {TC:.2%} vs ponderada {tc_ponderada:.2%} (casi iguales porque los volúmenes mensuales son
    parecidos, como en Austral: 5,42% vs 5,38%). DEV ancla el modelo en {tasa_dev:.2%}; el ciclo pide {TC:.2%}; el
    presente (PIT) pide {TASA_PIT:.2%}. El salto de 2025 es el `deterioro=0.35` plantado en el generador: la TC mezcla
    18 cosechas «buenas» con 6 «malas», por eso queda entre ambos regímenes. Una TC con 2 años de historia no es una
    *long-run average* en sentido regulatorio (la EBA pide un periodo representativo de años buenos y malos); se
    declara como «el ancla que la ventana permite».
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Ajuste de intercepto: aproximado vs exacto

    $$\text{PD}^{cal}_i=\sigma(\eta_i+\delta),\qquad \eta_i=\text{logit}(\text{PD}_i).$$

    - **Siddiqi (aprox.)**: $\delta_S=\text{logit}(T)-\text{logit}(\bar p)$.
    - **Exacto**: la raíz de $f(\delta)=\frac1n\sum_i\sigma(\eta_i+\delta)-T$. Es monótona creciente
      ($f'(\delta)=\overline{p(1-p)}>0$), así que tiene raíz única y la bisección converge siempre.
    - **Resultado clave del módulo** (derivado en el `.md`, §3.3): definiendo
      $g(\delta)=\text{logit}\big(\overline{\sigma(\eta+\delta)}\big)$,
      $$g'(\delta)=1-\kappa(\delta),\qquad \kappa=\frac{\operatorname{Var}(p)}{\bar p(1-\bar p)}\in[0,1),$$
      de modo que $\delta_S=\delta\,(1-\bar\kappa)$: **Siddiqi siempre se queda corto** (mismo signo, menor
      magnitud) y el error relativo es κ promedio en el camino. Un paso de Newton desde 0 da
      $\delta_N=\delta_S/(1-\kappa_0)$, calculable con solo $\bar p$ y $\operatorname{Var}(p)$.
      Para un modelo calibrado, κ es el **coeficiente de discriminación de Tjur**: mejor modelo ⇒ peor aproximación.

    Implementación 1: bisección en numpy. Implementación 2: `scipy.optimize.brentq`.
    """)
    return


@app.cell
def _(brentq, expit, logit, np):
    def sigmoide_np(z):
        """σ(z) en numpy puro, estable para |z| grande."""
        z = np.asarray(z, dtype=float)
        _ez = np.exp(-np.abs(z))
        return np.where(z >= 0, 1.0 / (1.0 + _ez), _ez / (1.0 + _ez))

    def logit_np(p):
        p = np.asarray(p, dtype=float)
        return np.log(p) - np.log1p(-p)

    def delta_siddiqi(p, objetivo):
        return float(logit_np(objetivo) - logit_np(np.mean(p)))

    def kappa(p, w=None):
        """κ = Var(p) / [p̄(1−p̄)]  (varianza poblacional, ddof=0)."""
        _pb = np.average(p, weights=w)
        _var = np.average((np.asarray(p) - _pb) ** 2, weights=w)
        return float(_var / (_pb * (1 - _pb)))

    def delta_biseccion(lp, objetivo, tol=1e-12, lo=-10.0, hi=10.0):
        """Raíz de mean σ(lp+δ) − objetivo por bisección (numpy puro). f es creciente en δ."""
        _f = lambda d: sigmoide_np(lp + d).mean() - objetivo
        assert _f(lo) < 0 < _f(hi), "el intervalo inicial no encierra la raíz"
        _it = 0
        while hi - lo > tol and _it < 200:
            _mid = 0.5 * (lo + hi)
            if _f(_mid) < 0:
                lo = _mid
            else:
                hi = _mid
            _it += 1
        return 0.5 * (lo + hi), _it

    def delta_brentq(lp, objetivo):
        return float(brentq(lambda d: expit(lp + d).mean() - objetivo, -10, 10, xtol=1e-14))

    def delta_newton1(p, objetivo):
        """Un paso de Newton en escala logit desde δ=0: δ_S / (1 − κ₀)."""
        return delta_siddiqi(p, objetivo) / (1 - kappa(p))

    # sanity: nuestras σ/logit coinciden con scipy
    _z = np.linspace(-30, 30, 7)
    assert np.allclose(sigmoide_np(_z), expit(_z)) and np.allclose(logit_np(expit(_z[1:-1])), logit(expit(_z[1:-1])))
    return (
        delta_biseccion,
        delta_brentq,
        delta_newton1,
        delta_siddiqi,
        kappa,
        logit_np,
        sigmoide_np,
    )


@app.cell
def _(
    TASA_PIT,
    TC,
    delta_biseccion,
    delta_brentq,
    delta_newton1,
    delta_siddiqi,
    es_pit,
    kappa,
    muestras,
    pd,
):
    _casos = {
        "TTC: DEV → TC": (muestras["DEV"]["pd_modelo"].values, TC),
        "PIT: mar–jun 2025 → tasa PIT": (muestras["OOT"]["pd_modelo"].values[es_pit], TASA_PIT),
    }
    _filas = []
    for _nombre, (_p, _T) in _casos.items():
        _lp = muestras["DEV"]["lp_modelo"].values if _nombre.startswith("TTC") else \
            muestras["OOT"]["lp_modelo"].values[es_pit]
        _db, _it = delta_biseccion(_lp, _T)
        _de = delta_brentq(_lp, _T)
        _ds = delta_siddiqi(_p, _T)
        _filas.append({"caso": _nombre, "n": len(_p), "pd_media": _p.mean(), "objetivo": _T,
                       "delta_siddiqi": _ds, "delta_biseccion": _db, "iter_bisec": _it,
                       "delta_brentq": _de, "delta_newton1": delta_newton1(_p, _T),
                       "kappa0": kappa(_p), "error_rel_siddiqi": 1 - _ds / _de})
    tabla_deltas = pd.DataFrame(_filas).set_index("caso")
    DELTA_TTC = float(tabla_deltas.loc["TTC: DEV → TC", "delta_brentq"])
    DELTA_PIT = float(tabla_deltas.loc["PIT: mar–jun 2025 → tasa PIT", "delta_brentq"])
    tabla_deltas.round(5)
    return DELTA_PIT, DELTA_TTC, tabla_deltas


@app.cell
def _(mo, tabla_deltas):
    _a = tabla_deltas.iloc[0]
    _b = tabla_deltas.iloc[1]
    mo.md(f"""
    **Lectura.** TTC: Siddiqi {_a.delta_siddiqi:+.4f} vs exacto {_a.delta_brentq:+.4f} (se queda corto un
    {_a.error_rel_siddiqi:.1%}); κ₀ = {_a.kappa0:.3f}, y el paso de Newton ({_a.delta_newton1:+.4f}) cierra casi todo el hueco.
    PIT: {_b.delta_siddiqi:+.4f} vs {_b.delta_brentq:+.4f} (corto un {_b.error_rel_siddiqi:.1%}). Bisección y `brentq`
    coinciden a 1e-10; la bisección necesitó {int(_a.iter_bisec)} iteraciones (log₂(20/1e-12) ≈ 44), `brentq` bastante menos.

    **Contraste con Banco Austral**: 0,092 vs 0,111 (TTC) y 0,143 vs 0,177 (PIT) implican κ̄ ≈ 1 − 0,0914/0,1115 ≈ 0,18 y
    1 − 0,143/0,177 ≈ 0,19. Austral discrimina más (Gini 0,75 vs ~0,55 aquí), sus PD son más dispersas, κ es mayor y
    la aproximación falla más. Es la misma ley.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Experimento de Jensen: la dispersión de las PD controla el error

    Juguete: $\eta\sim\mathcal N(\mu,s^2)$ con μ elegido para que $\bar p=5{,}19\%$ (Austral PIT antes de
    calibrar) y objetivo $T=5{,}94\%$. Las esperanzas se calculan con cuadratura de Gauss-Hermite (80 nodos,
    `numpy.polynomial.hermite_e`), sin simulación. Mueve $s$: con $s\to0$ todas las PD son iguales y Siddiqi es
    exacto; con $s$ grande (modelo muy discriminante) el error crece como κ.

    También se muestra la aproximación de Taylor de 2º orden
    $\bar p\approx\sigma(\mu)+\tfrac12 s^2\,\sigma(\mu)(1-\sigma(\mu))(1-2\sigma(\mu))$ y su límite de validez.
    """)
    return


@app.cell
def _(mo):
    slider_s = mo.ui.slider(0.05, 3.0, step=0.05, value=1.8, label="Dispersión s del logit de las PD")
    slider_s
    return (slider_s,)


@app.cell
def _(brentq, np, sigmoide_np):
    _nodos, _pesos = np.polynomial.hermite_e.hermegauss(80)
    _pesos = _pesos / _pesos.sum()                       # E[h(Z)] ≈ Σ w_k h(z_k), Z ~ N(0,1)

    def esperanza_normal(h, mu, s):
        return float(np.sum(_pesos * h(mu + s * _nodos)))

    def juguete_jensen(s, p0=0.0519, T=0.0594):
        """Devuelve δ_S, δ exacto, δ Newton, κ₀ y la Taylor de p̄ para η ~ N(μ, s²) con E[σ(η)] = p0."""
        mu = brentq(lambda m: esperanza_normal(sigmoide_np, m, s) - p0, -15, 5)
        m1 = esperanza_normal(sigmoide_np, mu, s)
        m2 = esperanza_normal(lambda z: sigmoide_np(z) ** 2, mu, s)
        k0 = (m2 - m1 ** 2) / (m1 * (1 - m1))
        ds = np.log(T / (1 - T)) - np.log(m1 / (1 - m1))
        de = brentq(lambda d: esperanza_normal(sigmoide_np, mu + d, s) - T, -5, 5, xtol=1e-13)
        sm_ = sigmoide_np(mu)
        taylor = sm_ + 0.5 * s ** 2 * sm_ * (1 - sm_) * (1 - 2 * sm_)
        return {"s": s, "mu": mu, "p_media": m1, "sigma_mu": float(sm_), "taylor_p": float(taylor),
                "kappa0": k0, "delta_siddiqi": ds, "delta_exacto": de,
                "delta_newton1": ds / (1 - k0), "error_rel": 1 - ds / de}
    return (juguete_jensen,)


@app.cell
def _(juguete_jensen, np, pd, plt, slider_s):
    grilla_jensen = pd.DataFrame([juguete_jensen(_s) for _s in np.linspace(0.05, 3.0, 40)])
    _sel = juguete_jensen(slider_s.value)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _a1.plot(grilla_jensen["s"], 100 * grilla_jensen["error_rel"], label="error rel. Siddiqi = 1 − δ_S/δ")
    _a1.plot(grilla_jensen["s"], 100 * grilla_jensen["kappa0"], "--", label="κ₀ = Var(p)/[p̄(1−p̄)]")
    _a1.plot(grilla_jensen["s"], 100 * (1 - grilla_jensen["delta_newton1"] / grilla_jensen["delta_exacto"]),
             ":", label="error rel. Newton 1 paso")
    _a1.axvline(slider_s.value, color="0.6", lw=1)
    _a1.set_xlabel("s (desv. estándar del logit de las PD)")
    _a1.set_ylabel("%")
    _a1.set_title("Error de la aproximación de Siddiqi")
    _a1.legend(fontsize=8)
    _a2.plot(grilla_jensen["s"], 100 * grilla_jensen["p_media"], label="p̄ exacta (= 5,19% fija)")
    _a2.plot(grilla_jensen["s"], 100 * grilla_jensen["sigma_mu"], "--", label="σ(μ): la PD del logit medio")
    _a2.plot(grilla_jensen["s"], 100 * grilla_jensen["taylor_p"], ":", label="Taylor 2º orden de p̄")
    _a2.axvline(slider_s.value, color="0.6", lw=1)
    _a2.set_ylim(0, 8)
    _a2.set_xlabel("s")
    _a2.set_ylabel("PD (%)")
    _a2.set_title("Jensen: la media de σ no es σ de la media")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    seleccion_jensen = _sel
    _fig
    return grilla_jensen, seleccion_jensen


@app.cell
def _(mo, seleccion_jensen):
    _j = seleccion_jensen
    mo.md(f"""
    **Lectura (s = {_j['s']:.2f}).** δ_S = {_j['delta_siddiqi']:+.4f}, δ exacto = {_j['delta_exacto']:+.4f},
    Newton 1 paso = {_j['delta_newton1']:+.4f}. κ₀ = {_j['kappa0']:.3f} y el error relativo de Siddiqi es
    {_j['error_rel']:.1%}: la identidad δ_S = δ(1 − κ̄) se ve en el gráfico (las dos primeras curvas casi se
    superponen; κ̄ es un promedio de κ en el camino de 0 a δ, por eso no calzan al decimal).
    Con s ≈ 1,8–2 se reproduce el κ ≈ 0,18 que implica Austral. La Taylor de 2º orden de p̄ es buena para
    s ≲ 0,7 y después se despega: para carteras reales (s > 1) no sirve como corrección; κ sí.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Sobremuestreo y *prior shift*: King & Zeng (2001)

    Si la muestra de desarrollo tiene una proporción de malos $\rho_1$ distinta de la poblacional
    $\pi_1$ (p. ej. se submuestrearon buenos), Bayes da (derivación en el `.md`, §3.4):
    $$\text{logit}\,p_{pob}(x)=\text{logit}\,p_{mues}(x)+\ln\!\Big[\frac{\pi_1}{\pi_0}\cdot\frac{\rho_0}{\rho_1}\Big].$$
    Es un **ajuste de intercepto con δ analítico**. Además, el WoE es invariante al submuestreo aleatorio de
    buenos (la distribución *dentro* de los buenos no cambia): solo el intercepto absorbe el prior.

    Experimento: submuestreamos buenos de DEV hasta una tasa de malos $\rho_1$ (slider), reajustamos WoE y
    logística **en la submuestra**, y corregimos con King-Zeng. Comparamos contra: el modelo en DEV completo,
    el δ exacto de la sección 3 y la verdad del generador.
    """)
    return


@app.cell
def _(mo):
    slider_rho = mo.ui.slider(0.15, 0.60, step=0.05, value=0.40, label="Tasa de malos ρ₁ en la submuestra")
    slider_rho
    return (slider_rho,)


@app.cell
def _(
    VARIABLES,
    a_woe,
    delta_brentq,
    dev,
    modelo,
    muestras,
    np,
    pd,
    sigmoide_np,
    slider_rho,
    sm,
    tabla_woe,
):
    def experimento_prior_shift(rho1, semilla=7):
        _rng = np.random.default_rng(semilla)
        _y = dev["malo"].values
        _pi1 = _y.mean()
        _idx_m = np.flatnonzero(_y == 1)
        _idx_b = np.flatnonzero(_y == 0)
        _n_b = int(round(len(_idx_m) * (1 - rho1) / rho1))
        _idx = np.concatenate([_idx_m, _rng.choice(_idx_b, size=min(_n_b, len(_idx_b)), replace=False)])
        _sub = dev.iloc[_idx].reset_index(drop=True)
        _rho1 = _sub["malo"].mean()
        _mapas = {v: tabla_woe(_sub[v], _sub["malo"])[0]["woe"].to_dict() for v in VARIABLES}
        _m = sm.Logit(_sub["malo"].values, sm.add_constant(a_woe(_sub, VARIABLES, _sub, _mapas))).fit(disp=0)
        # aplicar a DEV completo con los cortes y WoE de la submuestra
        _Xd = sm.add_constant(a_woe(dev, VARIABLES, _sub, _mapas), has_constant="add")
        _lp = np.asarray(_Xd.values @ _m.params.values, dtype=float)
        _d_kz = float(np.log(_pi1 / (1 - _pi1)) - np.log(_rho1 / (1 - _rho1)))
        _d_ex = delta_brentq(_lp, _pi1)
        return {
            "rho1_efectiva": _rho1, "pi1_dev": _pi1,
            "pd_media_sin_corregir": sigmoide_np(_lp).mean(),
            "delta_king_zeng": _d_kz, "delta_exacto": _d_ex,
            "pd_media_kz": sigmoide_np(_lp + _d_kz).mean(),
            "intercepto_sub": float(_m.params["const"]),
            "intercepto_sub_mas_kz": float(_m.params["const"]) + _d_kz,
            "intercepto_dev_completo": float(modelo.params["const"]),
            "betas_sub": _m.params.drop("const").values,
            "pd_real_media": float(dev["pd_real"].mean()),
            "corr_logit_con_modelo_completo": float(np.corrcoef(_lp, muestras["DEV"]["lp_modelo"].values)[0, 1]),
        }

    resultado_prior = experimento_prior_shift(slider_rho.value)
    _r = resultado_prior
    tabla_prior = pd.DataFrame({
        "valor": [_r["rho1_efectiva"], _r["pi1_dev"], _r["pd_media_sin_corregir"], _r["delta_king_zeng"],
                  _r["delta_exacto"], _r["pd_media_kz"], _r["intercepto_sub"], _r["intercepto_sub_mas_kz"],
                  _r["intercepto_dev_completo"], _r["corr_logit_con_modelo_completo"]],
    }, index=["ρ₁ (submuestra)", "π₁ (DEV completo)", "PD media en DEV sin corregir",
              "δ King-Zeng (analítico)", "δ exacto (brentq a π₁)", "PD media en DEV con δ_KZ",
              "β₀ submuestra", "β₀ submuestra + δ_KZ", "β₀ modelo DEV completo",
              "corr(logit submuestra, logit DEV completo)"])
    tabla_prior_betas = pd.DataFrame({"beta_submuestra": _r["betas_sub"],
                                      "beta_dev_completo": modelo.params.drop("const").values},
                                     index=VARIABLES)
    tabla_prior.round(4)
    return experimento_prior_shift, resultado_prior, tabla_prior, tabla_prior_betas


@app.cell
def _(mo, resultado_prior, tabla_prior_betas):
    _r = resultado_prior
    mo.vstack([
        tabla_prior_betas.round(3),
        mo.md(f"""
    **Lectura.** Con ρ₁ = {_r['rho1_efectiva']:.1%}, la PD media sin corregir en DEV es
    {_r['pd_media_sin_corregir']:.1%} (≈ ρ₁, por la ecuación de primer orden aplicada a la submuestra).
    King-Zeng: δ = {_r['delta_king_zeng']:+.3f} lleva la media a {_r['pd_media_kz']:.2%} vs π₁ = {_r['pi1_dev']:.2%};
    el δ exacto sería {_r['delta_exacto']:+.3f}. La diferencia existe porque King-Zeng es exacto **para el modelo
    verdadero** y el nuestro está mal especificado (WoE en 5 bins) y estimado con menos buenos; no hay Jensen aquí:
    el δ analítico no depende de la dispersión de las PD. Los β de pendiente se parecen a los del modelo completo
    (difieren por ruido: se botaron buenos) y β₀ + δ_KZ ≈ β₀ del modelo completo. **El sobremuestreo solo mueve el
    intercepto**; olvidarse de corregirlo produce PD infladas en un factor de odds de {(_r['rho1_efectiva']/(1-_r['rho1_efectiva']))/(_r['pi1_dev']/(1-_r['pi1_dev'])):.1f}×.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Recalibración logística: intercepto + pendiente (y Platt)

    $$\text{logit}\,p^*_i=a+b\,\eta_i .$$

    - $b$ es la **pendiente de calibración** (*calibration slope*, Cox 1958): $b<1$ ⇒ PD demasiado extremas
      (sobreajuste típico); $b>1$ ⇒ PD demasiado comprimidas.
    - Con $b\equiv1$ (η como *offset*) el MLE de $a$ resuelve $\sum_i(y_i-\sigma(\eta_i+a))=0$, que es
      **exactamente** la ecuación del δ exacto de la sección 3 con $T=\bar y$: calibrar al observado de una
      muestra = GLM con offset.
    - **Platt** (1999) = logística sobre el score: como $\text{score}=\text{offset}-\text{factor}\cdot\eta$, es la
      misma familia reparametrizada.

    Implementación 1: IRLS/Newton-Raphson en numpy (con offset). Implementación 2: `statsmodels` GLM Binomial.
    Muestra de calibración: la ventana PIT (OOT mar–jun 2025).
    """)
    return


@app.cell
def _(np, sigmoide_np):
    def irls_logistica(X, y, offset=None, tol=1e-12, max_iter=100):
        """MLE logístico por Newton-Raphson (= IRLS con enlace canónico). Devuelve (β, SE, iteraciones)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
        beta = np.zeros(X.shape[1])
        for it in range(1, max_iter + 1):
            p = sigmoide_np(X @ beta + off)
            W = p * (1 - p)
            grad = X.T @ (y - p)                      # score
            H = X.T @ (X * W[:, None])                # información observada (= esperada, enlace canónico)
            paso = np.linalg.solve(H, grad)
            beta = beta + paso
            if np.max(np.abs(paso)) < tol:
                break
        p = sigmoide_np(X @ beta + off)
        H = X.T @ (X * (p * (1 - p))[:, None])
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        return beta, se, it
    return (irls_logistica,)


@app.cell
def _(DELTA_PIT, es_pit, irls_logistica, muestras, np, pd, sm):
    FACTOR = 20 / np.log(2)
    OFFSET = 600 - FACTOR * np.log(50)
    _oot = muestras["OOT"]
    lp_pit = _oot["lp_modelo"].values[es_pit]
    y_pit = _oot["malo"].values[es_pit]
    _n = len(y_pit)

    # (a) solo intercepto con offset = δ exacto
    _b_off, _se_off, _ = irls_logistica(np.ones((_n, 1)), y_pit, offset=lp_pit)
    _g_off = sm.GLM(y_pit, np.ones((_n, 1)), family=sm.families.Binomial(), offset=lp_pit).fit()
    # (b) intercepto + pendiente
    _Xab = np.column_stack([np.ones(_n), lp_pit])
    coef_ab_numpy, se_ab_numpy, iter_irls = irls_logistica(_Xab, y_pit)
    _g_ab = sm.GLM(y_pit, _Xab, family=sm.families.Binomial()).fit()
    coef_ab_glm = np.asarray(_g_ab.params)
    # (c) Platt sobre el score
    _score = OFFSET - FACTOR * lp_pit
    _g_platt = sm.GLM(y_pit, sm.add_constant(_score), family=sm.families.Binomial()).fit()
    _c, _dd = np.asarray(_g_platt.params)
    platt_a_b = np.array([_c + _dd * OFFSET, -_dd * FACTOR])     # (a, b) implícitos

    delta_offset_numpy = float(_b_off[0])
    delta_offset_glm = float(np.asarray(_g_off.params)[0])
    tabla_recal = pd.DataFrame({
        "numpy IRLS": [delta_offset_numpy, coef_ab_numpy[0], coef_ab_numpy[1]],
        "statsmodels GLM": [delta_offset_glm, coef_ab_glm[0], coef_ab_glm[1]],
        "Platt (score) → (a,b)": [np.nan, platt_a_b[0], platt_a_b[1]],
        "SE (GLM)": [float(np.asarray(_g_off.bse)[0]), float(_g_ab.bse[0]), float(_g_ab.bse[1])],
        "δ exacto brentq (§3)": [DELTA_PIT, np.nan, np.nan],
    }, index=["δ (offset, b≡1)", "a (intercepto)", "b (pendiente)"])
    tabla_recal.round(5)
    return (
        FACTOR,
        OFFSET,
        coef_ab_glm,
        coef_ab_numpy,
        delta_offset_glm,
        delta_offset_numpy,
        iter_irls,
        lp_pit,
        platt_a_b,
        tabla_recal,
        y_pit,
    )


@app.cell
def _(coef_ab_glm, iter_irls, mo, muestras, np, sm, tabla_recal):
    _ho = muestras["HO"]
    _g = sm.GLM(_ho["malo"].values, sm.add_constant(_ho["lp_modelo"].values), family=sm.families.Binomial()).fit()
    _tt = muestras["TTD"]
    _ver = np.polyfit(_tt["lp_modelo"].values, np.log(_tt["pd_real"] / (1 - _tt["pd_real"])).values, 1)
    _b = coef_ab_glm[1]
    _se = tabla_recal.loc["b (pendiente)", "SE (GLM)"]
    pendiente_ho = float(_g.params[1])
    pendiente_verdad = float(_ver[0])
    mo.md(f"""
    **Lectura.** El δ con offset (IRLS y GLM) coincide con el δ exacto de `brentq`: son la misma ecuación.
    IRLS convergió en {iter_irls} iteraciones (convergencia cuadrática de Newton). La pendiente estimada en PIT es
    b = {_b:.3f} (SE {_se:.3f}; z = {(_b-1)/_se:.2f} contra b = 1: **no significativa** con n = 3.245); en HO,
    b = {pendiente_ho:.3f}. La verdad (regresión de logit(pd_real) sobre η en TTD) dice {pendiente_verdad:.3f}: el modelo
    es algo **conservador en el ordenamiento** (b > 1, PD comprimidas hacia la media por bins gruesos y variables
    omitidas), el caso contrario al sobreajuste. Moraleja: una desviación de pendiente real del 10% no se detecta con
    ~500 malos; por eso la hipótesis nula por defecto es b = 1 y se recalibra la pendiente solo con evidencia. Platt
    sobre el score entrega los mismos (a, b): {tabla_recal.loc['a (intercepto)','Platt (score) → (a,b)']:.4f},
    {tabla_recal.loc['b (pendiente)','Platt (score) → (a,b)']:.4f}.
    """)
    return pendiente_ho, pendiente_verdad


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Isotónica (PAV): flexible, con empates, y sin escala PDO

    La regresión isotónica busca la función **no decreciente** $m$ que minimiza $\sum_i(y_i-m(\eta_i))^2$.
    El algoritmo *pool-adjacent-violators* (PAV) la resuelve exacto en $O(n)$ tras ordenar. Consecuencias:

    - la salida es una **función escalonada**: muchos clientes quedan con la misma PD (empates);
    - los empates **cambian el AUC**: un par mal ordenado que pasa a empate sube de 0 a ½ (en la muestra de
      ajuste el AUC puede *subir*: sobreajuste), un par bien ordenado que pasa a empate baja de 1 a ½;
    - el mapa score → PD deja de ser $\text{logit}\,p=(\text{offset}-\text{score})/\text{factor}$: «20 puntos
      duplican las odds» deja de ser cierto.

    Implementación 1: PAV en numpy (con pila de bloques y agregación de empates). Implementación 2:
    `sklearn.isotonic.IsotonicRegression`. Ajuste en la ventana PIT; predicción en TTD.
    """)
    return


@app.cell
def _(np):
    def pav_numpy(x, y, w=None):
        """Regresión isotónica creciente por PAV. Devuelve (x_únicos, ajuste en x_únicos)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
        xu, inv = np.unique(x, return_inverse=True)           # 1) agregar empates en x
        wu = np.bincount(inv, weights=w)
        yu = np.bincount(inv, weights=w * y) / wu
        val, peso, largo = [], [], []                        # 2) pila de bloques
        for yi, wi in zip(yu, wu):
            val.append(yi)
            peso.append(wi)
            largo.append(1)
            while len(val) > 1 and val[-2] > val[-1]:        # violación de monotonía → fusionar
                wt = peso[-2] + peso[-1]
                vt = (val[-2] * peso[-2] + val[-1] * peso[-1]) / wt
                lt = largo[-2] + largo[-1]
                del val[-1], peso[-1], largo[-1]
                val[-1], peso[-1], largo[-1] = vt, wt, lt
        return xu, np.repeat(val, largo)

    def pav_predecir(xu, ajuste, x_nuevo):
        """Interpolación lineal entre umbrales y recorte en los extremos (misma convención que sklearn)."""
        return np.interp(np.asarray(x_nuevo, dtype=float), xu, ajuste)
    return pav_numpy, pav_predecir


@app.cell
def _(
    FACTOR,
    IsotonicRegression,
    OFFSET,
    lp_pit,
    muestras,
    np,
    pav_numpy,
    pav_predecir,
    roc_auc_score,
    y_pit,
):
    xu_iso, ajuste_iso = pav_numpy(lp_pit, y_pit)
    iso_sk = IsotonicRegression(out_of_bounds="clip").fit(lp_pit, y_pit)
    _lp_ttd = muestras["TTD"]["lp_modelo"].values
    pd_iso_ttd_numpy = pav_predecir(xu_iso, ajuste_iso, _lp_ttd)
    pd_iso_ttd_sklearn = iso_sk.predict(_lp_ttd)
    _pit_numpy = pav_predecir(xu_iso, ajuste_iso, lp_pit)

    _y_ttd = muestras["TTD"]["malo_futuro"].values
    resumen_iso = {
        "valores_unicos_pd_modelo_pit": int(len(np.unique(lp_pit))),
        "valores_unicos_isotonica_pit": int(len(np.unique(np.round(_pit_numpy, 12)))),
        "gini_pit_modelo": 2 * roc_auc_score(y_pit, lp_pit) - 1,
        "gini_pit_isotonica": 2 * roc_auc_score(y_pit, _pit_numpy) - 1,
        "gini_ttd_modelo": 2 * roc_auc_score(_y_ttd, _lp_ttd) - 1,
        "gini_ttd_isotonica": 2 * roc_auc_score(_y_ttd, pd_iso_ttd_numpy) - 1,
        "pd_iso_min": float(ajuste_iso.min()),
        "pd_iso_max": float(ajuste_iso.max()),
    }
    _score_grid = np.linspace(440, 640, 400)
    curva_iso_score = (_score_grid, pav_predecir(xu_iso, ajuste_iso, (OFFSET - _score_grid) / FACTOR))
    resumen_iso
    return (
        ajuste_iso,
        curva_iso_score,
        iso_sk,
        pd_iso_ttd_numpy,
        pd_iso_ttd_sklearn,
        resumen_iso,
        xu_iso,
    )


@app.cell
def _(
    DELTA_PIT,
    FACTOR,
    OFFSET,
    coef_ab_glm,
    curva_iso_score,
    np,
    plt,
    sigmoide_np,
):
    _s, _piso = curva_iso_score
    _eta = (OFFSET - _s) / FACTOR
    _fig, _ax = plt.subplots(figsize=(7.5, 3.6))
    _ax.semilogy(_s, 100 * sigmoide_np(_eta + DELTA_PIT), label="δ PIT (recta en log-odds)")
    _ax.semilogy(_s, 100 * sigmoide_np(coef_ab_glm[0] + coef_ab_glm[1] * _eta), "--", label="logística a + b·η")
    _ax.semilogy(_s, 100 * np.clip(_piso, 1e-4, None), drawstyle="steps-post", label="isotónica (PAV)")
    _ax.set_xlabel("score (PDO 20, 600 = 50:1, antes de calibrar)")
    _ax.set_ylabel("PD calibrada (%, escala log)")
    _ax.set_title("Mapa score → PD según el método de calibración")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, resumen_iso):
    _r = resumen_iso
    mo.md(f"""
    **Lectura.** En la ventana PIT el modelo tiene {_r['valores_unicos_pd_modelo_pit']} PD distintas; la isotónica las
    colapsa a {_r['valores_unicos_isotonica_pit']} escalones. Gini PIT {_r['gini_pit_modelo']:.4f} → {_r['gini_pit_isotonica']:.4f};
    en TTD (oráculo) {_r['gini_ttd_modelo']:.4f} → {_r['gini_ttd_isotonica']:.4f}. La PD isotónica va de {_r['pd_iso_min']:.2%}
    a {_r['pd_iso_max']:.2%}: los bloques extremos tienen pocos casos y quedan en 0 (ningún malo) o en 1 (todos malos).
    Una PD de 0% o de 100% es inaceptable para provisiones o pricing: hacen falta piso y techo (p. ej. el piso de PD de
    0,03% de Basilea II para IRB; Basilea III lo sube a 0,05% para minoristas: verificar con la norma aplicable). El
    Gini sube dentro de la muestra de ajuste y baja fuera: es el sello del sobreajuste. En el gráfico, δ y la logística
    son rectas en log-odds (compatibles con la escala PDO); la isotónica es una escalera: 20 puntos ya no duplican las odds.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.1 ¿Cuánta flexibilidad compra el tamaño de la muestra de calibración?

    Con verdad conocida podemos medir el **error de calibración contra la verdad**
    $\text{RMSE}_{verdad}=\sqrt{\tfrac1n\sum_i(\hat p_i-p^{real}_i)^2}$ en TTD, algo imposible con datos reales.
    Tomamos submuestras de tamaño $n_{cal}$ de la ventana PIT (con reemplazo, R réplicas), calibramos con cuatro
    métodos y evaluamos en TTD. Los métodos con más parámetros ganan solo si hay datos para pagarlos.
    """)
    return


@app.cell
def _(mo):
    slider_ncal = mo.ui.slider(200, 3000, step=100, value=600, label="n de la muestra de calibración")
    slider_ncal
    return (slider_ncal,)


@app.cell
def _(
    delta_brentq,
    irls_logistica,
    lp_pit,
    muestras,
    np,
    pav_numpy,
    pav_predecir,
    pd,
    sigmoide_np,
    slider_ncal,
    y_pit,
):
    def recal_bandas(lp_cal, y_cal, lp_nuevo, k=10):
        """Recalibración por bandas: PD = tasa observada de la banda (bandas por cuantiles de η en calibración)."""
        _bordes = np.unique(np.quantile(lp_cal, np.linspace(0, 1, k + 1)))
        _bc = np.clip(np.searchsorted(_bordes[1:-1], lp_cal, side="right"), 0, len(_bordes) - 2)
        _tasas = np.array([y_cal[_bc == j].mean() if np.any(_bc == j) else np.nan
                           for j in range(len(_bordes) - 1)])
        _tasas = np.where(np.isnan(_tasas), np.nanmean(_tasas), _tasas)
        _bn = np.clip(np.searchsorted(_bordes[1:-1], lp_nuevo, side="right"), 0, len(_bordes) - 2)
        return _tasas[_bn]

    def comparar_metodos(n_cal, R=25, semilla=11):
        _rng = np.random.default_rng(semilla)
        _tt = muestras["TTD"]
        _lp_t = _tt["lp_modelo"].values
        _verdad = _tt["pd_real"].values
        _res = {"sin calibrar": [], "δ (intercepto)": [], "logística a+b": [],
                "bandas (10)": [], "isotónica": []}
        for _r in range(R):
            _i = _rng.integers(0, len(y_pit), n_cal)
            _lc, _yc = lp_pit[_i], y_pit[_i]
            _preds = {
                "sin calibrar": sigmoide_np(_lp_t),
                "δ (intercepto)": sigmoide_np(_lp_t + delta_brentq(_lc, _yc.mean())),
            }
            _ab, _, _ = irls_logistica(np.column_stack([np.ones(n_cal), _lc]), _yc)
            _preds["logística a+b"] = sigmoide_np(_ab[0] + _ab[1] * _lp_t)
            _preds["bandas (10)"] = recal_bandas(_lc, _yc, _lp_t)
            _xu, _aj = pav_numpy(_lc, _yc)
            _preds["isotónica"] = pav_predecir(_xu, _aj, _lp_t)
            for _k, _p in _preds.items():
                _res[_k].append(np.sqrt(np.mean((_p - _verdad) ** 2)))
        return pd.DataFrame({"rmse_vs_verdad_media": {k: np.mean(v) for k, v in _res.items()},
                             "rmse_vs_verdad_p90": {k: np.quantile(v, 0.9) for k, v in _res.items()}})

    tabla_metodos_ncal = comparar_metodos(slider_ncal.value)
    tabla_metodos_ncal.round(5)
    return comparar_metodos, recal_bandas, tabla_metodos_ncal


@app.cell
def _(mo, slider_ncal, tabla_metodos_ncal):
    _t = tabla_metodos_ncal["rmse_vs_verdad_media"]
    _mejor = _t.idxmin()
    mo.md(f"""
    **Lectura (n_cal = {slider_ncal.value}).** Mejor método contra la verdad: **{_mejor}**
    (RMSE {_t.min():.4f}). δ: {_t['δ (intercepto)']:.4f} · logística: {_t['logística a+b']:.4f} ·
    bandas: {_t['bandas (10)']:.4f} · isotónica: {_t['isotónica']:.4f} · sin calibrar: {_t['sin calibrar']:.4f}.
    Mueve el slider: con n chico la isotónica y las bandas pagan varianza (cada escalón se estima con pocos
    malos); con n grande la logística a+b aprovecha que b ≠ 1. Ninguno recupera el piso de error que viene
    de la falta de resolución del modelo (bins gruesos, variables omitidas): **calibrar no agrega información**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Métricas de calibración

    Para una PD $p_i$ y un resultado $y_i$ en una muestra de validación:

    | métrica | fórmula | ideal |
    |---|---|---|
    | CITL (diferencia) | $\bar y-\bar p$ | 0 |
    | CITL (logit, Van Calster) | $a$ en $\text{logit}\,P(y=1)=a+\eta$ (offset) | 0 |
    | razón O/E | $\sum y_i/\sum p_i$ | 1 |
    | pendiente | $b$ en $a+b\,\eta$ | 1 |
    | Brier | $\frac1n\sum(p_i-y_i)^2$ | bajo |
    | log-loss | $-\frac1n\sum[y\ln p+(1-y)\ln(1-p)]$ | bajo |
    | ECE | $\sum_k\frac{n_k}{n}\lvert\bar y_k-\bar p_k\rvert$ | 0 |

    **Murphy con K grupos** (exacta, Stephenson et al. 2008):
    $\text{BS}=\text{REL}-\text{RES}+\text{UNC}+\text{WBV}-2\,\text{WBC}$, con
    $\text{REL}=\sum_k\frac{n_k}{n}(\bar p_k-\bar y_k)^2$, $\text{RES}=\sum_k\frac{n_k}{n}(\bar y_k-\bar y)^2$,
    $\text{UNC}=\bar y(1-\bar y)$, $\text{WBV}=\frac1n\sum_k\sum_{i\in k}(p_i-\bar p_k)^2$ y
    $\text{WBC}=\frac1n\sum_k\sum_{i\in k}(p_i-\bar p_k)(y_i-\bar y_k)$.

    Implementación 1: numpy. Implementación 2: `sklearn` (`brier_score_loss`, `log_loss`, `calibration_curve`).
    Los grupos usan la convención de `calibration_curve(strategy="quantile")` para que ambas coincidan.
    """)
    return


@app.cell
def _(np):
    def grupos_cuantil(p, k):
        """Asignación de grupo con la misma regla que sklearn.calibration_curve(strategy='quantile')."""
        _bordes = np.percentile(p, np.linspace(0, 100, k + 1))
        return np.searchsorted(_bordes[1:-1], p)

    def curva_numpy(p, y, k):
        g = grupos_cuantil(p, k)
        _n = np.bincount(g, minlength=k)
        _sp = np.bincount(g, weights=p, minlength=k)
        _sy = np.bincount(g, weights=y, minlength=k)
        _nz = _n > 0
        return _sy[_nz] / _n[_nz], _sp[_nz] / _n[_nz], _n[_nz], g

    def murphy(p, y, k=10):
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        obs_k, pred_k, n_k, g = curva_numpy(p, y, k)
        _ids = np.unique(g)
        _map = {gid: j for j, gid in enumerate(_ids)}
        _j = np.array([_map[v] for v in g])
        yb = y.mean()
        n = len(y)
        rel = np.sum(n_k * (pred_k - obs_k) ** 2) / n
        res = np.sum(n_k * (obs_k - yb) ** 2) / n
        unc = yb * (1 - yb)
        wbv = np.sum((p - pred_k[_j]) ** 2) / n
        wbc = np.sum((p - pred_k[_j]) * (y - obs_k[_j])) / n
        bs = np.mean((p - y) ** 2)
        return {"brier": bs, "REL": rel, "RES": res, "UNC": unc, "WBV": wbv, "WBC": wbc,
                "suma": rel - res + unc + wbv - 2 * wbc}

    def logloss_numpy(p, y, eps=1e-15):
        p = np.clip(np.asarray(p, float), eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))

    def ece_numpy(p, y, k=10):
        obs_k, pred_k, n_k, _ = curva_numpy(p, y, k)
        return float(np.sum(n_k * np.abs(obs_k - pred_k)) / n_k.sum())
    return curva_numpy, ece_numpy, grupos_cuantil, logloss_numpy, murphy


@app.cell
def _(
    DELTA_PIT,
    DELTA_TTC,
    coef_ab_glm,
    delta_brentq,
    ece_numpy,
    logit_np,
    logloss_numpy,
    muestras,
    murphy,
    np,
    pd,
    pd_iso_ttd_numpy,
    sigmoide_np,
    sm,
):
    _tt = muestras["TTD"]
    y_val = _tt["malo_futuro"].values
    _lp = _tt["lp_modelo"].values
    predicciones_ttd = {
        "sin calibrar": sigmoide_np(_lp),
        "δ TTC (DEV→TC)": sigmoide_np(_lp + DELTA_TTC),
        "δ PIT (mar–jun 2025)": sigmoide_np(_lp + DELTA_PIT),
        "logística a+b (PIT)": sigmoide_np(coef_ab_glm[0] + coef_ab_glm[1] * _lp),
        "isotónica (PIT)": pd_iso_ttd_numpy,
    }
    _filas = []
    for _k, _p in predicciones_ttd.items():
        _pc = np.clip(_p, 1e-6, 1 - 1e-6)
        _g = sm.GLM(y_val, sm.add_constant(logit_np(_pc)), family=sm.families.Binomial()).fit()
        _m = murphy(_p, y_val)
        _filas.append({
            "método": _k, "pd_media": _p.mean(), "obs": y_val.mean(),
            "citl_dif": y_val.mean() - _p.mean(),
            "citl_logit": delta_brentq(logit_np(_pc), y_val.mean()),
            "o_e": y_val.sum() / _p.sum(), "pendiente": float(_g.params[1]),
            "brier": _m["brier"], "REL": _m["REL"], "RES": _m["RES"],
            "log_loss": logloss_numpy(_pc, y_val), "ece": ece_numpy(_p, y_val),
            "rmse_vs_verdad": float(np.sqrt(np.mean((_p - _tt["pd_real"].values) ** 2))),
        })
    tabla_metricas = pd.DataFrame(_filas).set_index("método")
    tabla_metricas.round(4)
    return predicciones_ttd, tabla_metricas, y_val


@app.cell
def _(mo, tabla_metricas):
    _t = tabla_metricas
    mo.md(f"""
    **Lectura (validación en TTD con el oráculo; tasa {_t['obs'].iloc[0]:.2%}).** Sin calibrar, O/E =
    {_t.loc['sin calibrar','o_e']:.2f}: por cada 100 malos esperados llegan {100*_t.loc['sin calibrar','o_e']:.0f}.
    El δ TTC deja O/E {_t.loc['δ TTC (DEV→TC)','o_e']:.2f} (ancla de ciclo en un año malo: **subestima por diseño**);
    el δ PIT, {_t.loc['δ PIT (mar–jun 2025)','o_e']:.2f}. La pendiente en TTD (~{_t.loc['δ PIT (mar–jun 2025)','pendiente']:.2f}) no cambia con δ:
    un intercepto no corrige pendiente; la logística a+b sí la acerca a 1 ({_t.loc['logística a+b (PIT)','pendiente']:.2f}).
    La **resolución (RES)** es casi idéntica en todos los métodos monótonos: la discriminación no se compra calibrando.
    Brier se mueve poco entre métodos (está dominado por UNC = p̄(1−p̄)); por eso Brier solo no sirve para
    comparar calibraciones: mira REL, O/E y pendiente.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Curvas de calibración con intervalos de confianza

    Para cada grupo: PD predicha media, tasa observada con IC al 95% y (privilegio del generador) la PD real
    media. **Wilson** (inversión del test score) y **Jeffreys** (cuantiles de la posterior Beta(k+½, n−k+½))
    tienen cobertura cercana a la nominal incluso con pocos eventos; Wald no (Brown, Cai y DasGupta 2001).

    Implementación 1: numpy/`scipy.special.betaincinv`. Implementación 2: `statsmodels.proportion_confint`.
    """)
    return


@app.cell
def _(mo):
    dropdown_metodo = mo.ui.dropdown(
        options=["sin calibrar", "δ TTC (DEV→TC)", "δ PIT (mar–jun 2025)", "logística a+b (PIT)", "isotónica (PIT)"],
        value="δ TTC (DEV→TC)", label="Calibración")
    dropdown_ic = mo.ui.dropdown(options=["Wilson", "Jeffreys"], value="Wilson", label="Intervalo")
    slider_grupos = mo.ui.slider(5, 20, step=1, value=10, label="Nº de grupos")
    mo.hstack([dropdown_metodo, dropdown_ic, slider_grupos])
    return dropdown_ic, dropdown_metodo, slider_grupos


@app.cell
def _(betaincinv, np):
    def ic_wilson(k, n, z=1.959963984540054):
        k = np.asarray(k, float)
        n = np.asarray(n, float)
        ph = k / n
        centro = (ph + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        semi = z / (1 + z ** 2 / n) * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2))
        return centro - semi, centro + semi

    def ic_jeffreys(k, n, alfa=0.05):
        """Cuantiles de Beta(k+½, n−k+½). Sin el ajuste de Brown et al. en k=0/k=n (igual que statsmodels)."""
        k = np.asarray(k, float)
        n = np.asarray(n, float)
        return betaincinv(k + 0.5, n - k + 0.5, alfa / 2), betaincinv(k + 0.5, n - k + 0.5, 1 - alfa / 2)
    return ic_jeffreys, ic_wilson


@app.cell
def _(
    binom,
    curva_numpy,
    dropdown_ic,
    dropdown_metodo,
    grupos_cuantil,
    ic_jeffreys,
    ic_wilson,
    muestras,
    np,
    pd,
    plt,
    predicciones_ttd,
    slider_grupos,
    y_val,
):
    _p = predicciones_ttd[dropdown_metodo.value]
    _k = slider_grupos.value
    _obs, _pred, _n, _g = curva_numpy(_p, y_val, _k)
    _malos = np.round(_obs * _n).astype(int)
    _verdad = pd.Series(muestras["TTD"]["pd_real"].values).groupby(_g).mean().values
    _lo, _hi = (ic_wilson if dropdown_ic.value == "Wilson" else ic_jeffreys)(_malos, _n)
    # p-valor binomial de dos colas por «doble cola mínima» (convención simple; M19 discute alternativas)
    _p_sup = binom.sf(_malos - 1, _n, _pred)
    _p_inf = binom.cdf(_malos, _n, _pred)
    _p2 = np.minimum(1, 2 * np.minimum(_p_sup, _p_inf))
    tabla_curva = pd.DataFrame({"n": _n, "malos": _malos, "pd_predicha": _pred, "tasa_obs": _obs,
                                "ic_inf": _lo, "ic_sup": _hi, "pd_real_media": _verdad,
                                "pred_dentro_ic": (_pred >= _lo) & (_pred <= _hi), "p_binomial_2c": _p2},
                               index=pd.Index(range(1, len(_n) + 1), name="grupo"))
    _fig, _ax = plt.subplots(figsize=(6.5, 5))
    _lim = max(_hi.max(), _pred.max()) * 1.05
    _ax.plot([0, _lim], [0, _lim], ls="--", color="0.5", lw=1, label="predicho = observado")
    _ax.errorbar(_pred, _obs, yerr=[_obs - _lo, _hi - _obs], fmt="o", color="#1f4e79", ms=4, capsize=2,
                 label=f"observado ± IC 95% {dropdown_ic.value}")
    _ax.plot(_pred, _verdad, "x", color="#b3261e", label="PD real media del grupo (verdad)")
    _ax.set_xlim(0, _lim)
    _ax.set_ylim(0, _lim)
    _ax.set_xlabel("PD predicha media del grupo")
    _ax.set_ylabel("tasa de malos")
    _ax.set_title(f"Curva de calibración en TTD (oráculo) · {dropdown_metodo.value}")
    _ax.legend(fontsize=8, loc="upper left")
    _fig.tight_layout()
    _fig
    return (tabla_curva,)


@app.cell
def _(dropdown_metodo, mo, tabla_curva):
    _t = tabla_curva
    _fuera = int((~_t["pred_dentro_ic"]).sum())
    _sig = int((_t["p_binomial_2c"] < 0.05).sum())
    mo.vstack([
        _t.round(4),
        mo.md(f"""
    **Lectura ({dropdown_metodo.value}).** {_fuera} de {len(_t)} grupos tienen la PD predicha fuera del IC 95%;
    {_sig} con p binomial < 0,05. Con {len(_t)} grupos y calibración perfecta esperaríamos ~{0.05*len(_t):.1f} falsos
    positivos al 5%. Las cruces rojas (verdad) muestran cuánto del zigzag es ruido de muestreo: cuando la cruz está
    sobre el punto predicho y la tasa observada se aleja, es ruido; cuando la cruz se aleja del predicho, es
    descalibración real. Prueba «sin calibrar» y «δ TTC» (todo el patrón sobre la diagonal: subestimación
    sistemática) contra «δ PIT».
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 9.1 El grupo 5 de OOT de Banco Austral: ¿es significativo?

    Lámina 32 (clase 4 v21): grupo 5 de OOT, PD PIT 1,18%, observado 4,00%, n = 200 ⇒ 8 malos donde se
    esperaban 2,36. Y el análogo de la clase 5 con bandas: B2, 235 créditos, PD 1,42%, 8 malos, p 0,020.
    """)
    return


@app.cell
def _(binom, binomtest, ic_jeffreys, ic_wilson, np, pd, proportion_confint):
    _casos = [("Austral OOT grupo 5 (PIT)", 8, 200, 0.0118), ("Austral OOT grupo 5 (TTC, v1)", 8, 200, 0.0110),
              ("Austral OOT banda B2 (clase 5)", 8, 235, 0.0142)]
    _filas = []
    for _nom, _k, _n, _p0 in _casos:
        _w = ic_wilson(_k, _n)
        _j = ic_jeffreys(_k, _n)
        _filas.append({
            "caso": _nom, "malos": _k, "n": _n, "pd": _p0, "esperados": _n * _p0,
            "wilson_inf": float(_w[0]), "wilson_sup": float(_w[1]),
            "jeffreys_inf": float(_j[0]), "jeffreys_sup": float(_j[1]),
            "p_una_cola_(≥k)": float(binom.sf(_k - 1, _n, _p0)),
            "p_2c_scipy": float(binomtest(_k, _n, _p0).pvalue),
            "umbral_bonferroni_10": 0.005,
        })
    tabla_austral = pd.DataFrame(_filas).set_index("caso")
    # numpy vs statsmodels
    _sw = proportion_confint(8, 200, alpha=0.05, method="wilson")
    _sj = proportion_confint(8, 200, alpha=0.05, method="jeffreys")
    ic_coinciden = bool(np.allclose(ic_wilson(8, 200), _sw) and np.allclose(ic_jeffreys(8, 200), _sj))
    tabla_austral.round(4)
    return ic_coinciden, tabla_austral


@app.cell
def _(ic_coinciden, mo, tabla_austral):
    _g = tabla_austral.iloc[0]
    _b = tabla_austral.iloc[2]
    mo.md(f"""
    **Lectura.** IC Wilson de 8/200: [{_g.wilson_inf:.2%}; {_g.wilson_sup:.2%}] — no contiene 1,18%. p de una cola
    P(X ≥ 8 | 1,18%) = {_g['p_una_cola_(≥k)']:.4f}; incluso con Bonferroni por 10 grupos (0,005) rechaza. Pero tres
    matices de validador: (i) el grupo se miró **porque** se veía raro (si se miran 10 grupos y se elige el peor,
    Bonferroni es lo mínimo); (ii) la partición es arbitraria: con las 8 bandas de la clase 5 la misma zona da
    B2 con p = {_b.p_2c_scipy:.3f} (amarillo, no rojo); (iii) el binomial asume independencia; con correlación de
    defaults el p real es mayor (M19). Numpy y statsmodels coinciden en ambos IC: {ic_coinciden}.
    La respuesta de comité: «es evidencia de subestimación en bandas buenas, consistente con el patrón B2/A2;
    no se corrige a mano; se vigila y se testea la pendiente».
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. PIT vs TTC operativo: mismo ranking, dos niveles, dos scores

    Sumar δ al logit resta $\delta\cdot\text{factor}$ puntos a **todos** los scores
    ($\text{score}=\text{offset}-\text{factor}\cdot\eta$). Dos formas de gobernarlo:
    (a) el cutoff se define en **score** y no se toca ⇒ la aprobación no cambia pero la PD que se promete en el
    corte sí; (b) el cutoff se define en **PD** (apetito) ⇒ el score de corte se mueve δ·factor puntos.

    Slider: cutoff en score (calculado sobre el score **sin calibrar**, como en producción cuando la tabla de
    puntos no se re-escala).
    """)
    return


@app.cell
def _(mo):
    slider_cutoff = mo.ui.slider(480, 600, step=5, value=530, label="Cutoff de score (sin calibrar)")
    slider_cutoff
    return (slider_cutoff,)


@app.cell
def _(
    DELTA_PIT,
    DELTA_TTC,
    FACTOR,
    OFFSET,
    muestras,
    np,
    pd,
    sigmoide_np,
    slider_cutoff,
):
    _tt = muestras["TTD"]
    _lp = _tt["lp_modelo"].values
    _score = OFFSET - FACTOR * _lp
    _c = slider_cutoff.value
    _aprob = _score >= _c
    _eta_c = (OFFSET - _c) / FACTOR
    _filas = []
    for _nom, _d in [("sin calibrar", 0.0), ("TTC", DELTA_TTC), ("PIT", DELTA_PIT)]:
        _pdc = sigmoide_np(_lp + _d)
        _filas.append({
            "calibración": _nom, "delta": _d, "desplazamiento_pts": -_d * FACTOR,
            "pd_en_el_cutoff": float(sigmoide_np(_eta_c + _d)),
            "aprobación_ttd": _aprob.mean(),
            "pd_media_prometida_aprobados": _pdc[_aprob].mean(),
            "pd_real_aprobados": _tt["pd_real"].values[_aprob].mean(),
            "malos_oráculo_aprobados": _tt["malo_futuro"].values[_aprob].mean(),
        })
    tabla_pit_ttc = pd.DataFrame(_filas).set_index("calibración")
    # cutoff definido en PD: la PD sin calibrar en el cutoff elegido como apetito fijo
    _pd_apetito = float(sigmoide_np(_eta_c))
    _cut_en_pd = {}
    for _nom, _d in [("sin calibrar", 0.0), ("TTC", DELTA_TTC), ("PIT", DELTA_PIT)]:
        _s_c = OFFSET - FACTOR * (np.log(_pd_apetito / (1 - _pd_apetito)) - _d)
        _cut_en_pd[_nom] = {"score_de_corte_equivalente": _s_c, "aprobación_ttd": float((_score >= _s_c).mean())}
    tabla_cutoff_pd = pd.DataFrame(_cut_en_pd).T
    pd_apetito = _pd_apetito
    tabla_pit_ttc.round(4)
    return pd_apetito, tabla_cutoff_pd, tabla_pit_ttc


@app.cell
def _(DELTA_PIT, DELTA_TTC, FACTOR, mo, pd_apetito, tabla_cutoff_pd, tabla_pit_ttc):
    _t = tabla_pit_ttc
    _c = tabla_cutoff_pd
    mo.vstack([
        mo.md(f"**Cutoff definido en PD** (apetito = PD máx. {pd_apetito:.2%}, la del cutoff sin calibrar):"),
        _c.round(4),
        mo.md(f"""
    **Lectura.** δ_TTC = {DELTA_TTC:+.3f} ⇒ {DELTA_TTC*FACTOR:.1f} puntos; δ_PIT = {DELTA_PIT:+.3f} ⇒ {DELTA_PIT*FACTOR:.1f}
    puntos; la distancia entre ambas calibraciones es {(DELTA_PIT-DELTA_TTC)*FACTOR:.1f} puntos para todos los clientes
    (en Austral: (0,177 − 0,1115)·28,85 = 1,9 puntos). Con cutoff en score la aprobación es idéntica
    ({_t['aprobación_ttd'].iloc[0]:.1%}), pero la PD prometida de los aprobados cambia: TTC promete
    {_t.loc['TTC','pd_media_prometida_aprobados']:.2%}, PIT {_t.loc['PIT','pd_media_prometida_aprobados']:.2%};
    la verdad es {_t.loc['PIT','pd_real_aprobados']:.2%}. Con cutoff en PD (apetito fijo), la aprobación cae de
    {_c['aprobación_ttd'].iloc[0]:.1%} a {_c.loc['TTC','aprobación_ttd']:.1%} (TTC) y {_c.loc['PIT','aprobación_ttd']:.1%} (PIT).
    En Banco Sintético el deterioro 2025 es grande (δ_PIT − δ_TTC ≈ {DELTA_PIT-DELTA_TTC:.2f}), por eso la diferencia
    de decisión es mucho mayor que el 78,3% vs 77,2% de Austral. **La calibración elegida es una decisión de negocio
    que cambia la aprobación si el apetito está escrito en PD.**
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Calibrar y validar en la misma muestra: p = 1 por construcción

    Si δ se ajusta para que $\bar p=\bar y$ en una muestra y luego se testea el nivel en **esa misma**
    muestra, el número esperado de malos es exactamente el observado: $k$ es la moda de
    $\text{Bin}(n,k/n)$, así que el p bilateral de `binomtest` es 1. El test no tiene potencia: no puede
    rechazar nunca.

    Experimento: (a) PIT calibrado y testeado en mar–jun 2025; (b) 200 particiones aleatorias de la ventana
    PIT en mitades: calibrar en una, testear en la otra; (c) la calibración TTC testeada en la ventana PIT
    (una hipótesis falsa: el test **debe** rechazar).
    """)
    return


@app.cell
def _(
    DELTA_PIT,
    DELTA_TTC,
    binomtest,
    delta_brentq,
    lp_pit,
    np,
    plt,
    sigmoide_np,
    y_pit,
):
    def p_global(pd_, y_):
        """Binomial global: malos observados vs n·PD media (bilateral, método de scipy)."""
        return float(binomtest(int(y_.sum()), len(y_), float(np.mean(pd_))).pvalue)

    p_misma_muestra = p_global(sigmoide_np(lp_pit + DELTA_PIT), y_pit)
    p_ttc_en_pit = p_global(sigmoide_np(lp_pit + DELTA_TTC), y_pit)
    _rng = np.random.default_rng(2024)
    _ps = []
    for _b in range(200):
        _perm = _rng.permutation(len(y_pit))
        _A, _B = _perm[: len(_perm) // 2], _perm[len(_perm) // 2:]
        _d = delta_brentq(lp_pit[_A], y_pit[_A].mean())
        _ps.append(p_global(sigmoide_np(lp_pit[_B] + _d), y_pit[_B]))
    p_particiones = np.array(_ps)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.hist(p_particiones, bins=20, range=(0, 1), color="#1f4e79", alpha=0.8,
             label="calibrar en mitad A, testear en mitad B")
    _ax.axvline(p_misma_muestra, color="#b3261e", lw=2, label=f"misma muestra: p = {p_misma_muestra:.3f}")
    _ax.axvline(p_ttc_en_pit, color="#ef6c00", lw=2, ls="--", label=f"TTC testeada en PIT: p = {p_ttc_en_pit:.1e}")
    _ax.set_xlabel("p-valor binomial global (bilateral)")
    _ax.set_ylabel("frecuencia (de 200)")
    _ax.set_title("La misma muestra no puede calibrar y validar")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return p_global, p_misma_muestra, p_particiones, p_ttc_en_pit


@app.cell
def _(mo, np, p_misma_muestra, p_particiones, p_ttc_en_pit):
    from scipy.stats import norm as _norm
    _norm_cdf = _norm.cdf
    mo.md(f"""
    **Lectura.** Misma muestra: p = {p_misma_muestra:.4f} (≡ 1). Mitades independientes: los p se reparten en
    [0, 1] (mediana {np.median(p_particiones):.2f}), pero **{np.mean(p_particiones < 0.05):.1%} caen bajo 0,05**, no 5%. No es un
    error: el δ de la mitad A trae su propio error de muestreo, del mismo tamaño que el de la mitad B, y el binomial
    lo ignora. La varianza de (observado − esperado) se duplica, el estadístico real es √2 veces más ancho y la tasa
    de rechazo nominal 5% pasa a 2Φ(−1,96/√2) = {2*_norm_cdf(-1.959964/np.sqrt(2)):.1%}. Lección: cuando la PD se calibró con una
    muestra de tamaño comparable a la de validación, el test correcto es de **diferencia de dos proporciones**. La
    calibración TTC evaluada en el periodo PIT da p = {p_ttc_en_pit:.1e}: el test **sí** tiene potencia cuando el nivel
    está mal. Es el argumento de la clase 5: con TC el binomial OOT de Austral dio 0,56 (informativo); con PIT daría 1,00.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Checks del módulo

    Si falla un `assert`, el notebook falla. Cubren las coincidencias numpy vs librería y las invariantes
    teóricas del módulo.
    """)
    return


@app.cell
def _(
    DELTA_PIT,
    DELTA_TTC,
    TC,
    ajuste_iso,
    brecha_media_dev,
    brier_score_loss,
    calibration_curve,
    coef_ab_glm,
    coef_ab_numpy,
    curva_numpy,
    delta_biseccion,
    delta_newton1,
    delta_offset_glm,
    delta_offset_numpy,
    delta_siddiqi,
    ece_numpy,
    es_pit,
    grilla_jensen,
    ic_coinciden,
    iso_sk,
    kappa,
    log_loss,
    logloss_numpy,
    mo,
    muestras,
    murphy,
    np,
    p_misma_muestra,
    p_ttc_en_pit,
    pd_iso_ttd_numpy,
    pd_iso_ttd_sklearn,
    platt_a_b,
    predicciones_ttd,
    resultado_prior,
    roc_auc_score,
    sigmoide_np,
    tabla_deltas,
    tabla_metricas,
    xu_iso,
    y_pit,
    y_val,
):
    _checks = []

    # 1. Primer orden: la PD media de DEV es la tasa de DEV
    assert brecha_media_dev < 1e-8
    _checks.append("FOC: PD media DEV = tasa DEV")

    # 2. δ: bisección numpy = brentq; δ exacto clava la media; Siddiqi se queda corto; Newton mejora
    _lp_dev = muestras["DEV"]["lp_modelo"].values
    _p_dev = muestras["DEV"]["pd_modelo"].values
    _db, _ = delta_biseccion(_lp_dev, TC)
    assert np.isclose(_db, DELTA_TTC, atol=1e-9)
    assert abs(sigmoide_np(_lp_dev + DELTA_TTC).mean() - TC) < 1e-10
    for _, _r in tabla_deltas.iterrows():
        assert np.isclose(_r.delta_biseccion, _r.delta_brentq, atol=1e-9)
        assert 0 < _r.delta_siddiqi < _r.delta_brentq            # mismo signo, menor magnitud
        assert abs(_r.delta_newton1 - _r.delta_brentq) < abs(_r.delta_siddiqi - _r.delta_brentq)
    _checks.append("δ bisección = brentq; Siddiqi corto; Newton mejor")

    # 3. Identidad g'(δ) = 1 − κ(δ) por diferencias finitas
    _h = 1e-5
    _g = lambda d: np.log(sigmoide_np(_lp_dev + d).mean() / (1 - sigmoide_np(_lp_dev + d).mean()))
    for _d0 in (0.0, DELTA_TTC, 0.5):
        _deriv = (_g(_d0 + _h) - _g(_d0 - _h)) / (2 * _h)
        assert np.isclose(_deriv, 1 - kappa(sigmoide_np(_lp_dev + _d0)), atol=1e-6)
    # y δ_S = δ·(1 − κ̄) con κ̄ entre κ(0) y κ(δ)
    _kbar = 1 - delta_siddiqi(_p_dev, TC) / DELTA_TTC
    _k0, _k1 = kappa(_p_dev), kappa(sigmoide_np(_lp_dev + DELTA_TTC))
    assert min(_k0, _k1) - 1e-9 <= _kbar <= max(_k0, _k1) + 1e-9
    # κ = coeficiente de Tjur en DEV (modelo calibrado en DEV)
    _y_dev = muestras["DEV"]["malo"].values
    _tjur = _p_dev[_y_dev == 1].mean() - _p_dev[_y_dev == 0].mean()
    assert abs(_tjur - _k0) < 0.01          # igualdad exacta solo si E[y|p] = p; aquí ≈
    _checks.append("g'(δ) = 1 − κ; κ̄ acotado; κ ≈ Tjur en DEV")

    # 4. Jensen: el error relativo crece con la dispersión y es ~0 con s→0
    assert grilla_jensen["error_rel"].is_monotonic_increasing
    assert grilla_jensen["error_rel"].iloc[0] < 1e-3
    _checks.append("Jensen: error monótono en s, nulo en s→0")

    # 5. Ranking intacto con δ: Gini igual
    _yo = muestras["OOT"]["malo"].values
    _lo = muestras["OOT"]["lp_modelo"].values
    assert np.isclose(roc_auc_score(_yo, sigmoide_np(_lo)), roc_auc_score(_yo, sigmoide_np(_lo + DELTA_PIT)))
    _checks.append("δ no cambia el AUC")

    # 6. Prior shift: la corrección KZ acerca la media a π₁ mucho más que no corregir
    _r = resultado_prior
    assert abs(_r["pd_media_kz"] - _r["pi1_dev"]) < 0.25 * abs(_r["pd_media_sin_corregir"] - _r["pi1_dev"])
    _checks.append("King-Zeng corrige el nivel")

    # 7. Recalibración: IRLS numpy = GLM; offset = δ exacto; Platt = (a, b)
    assert np.allclose(coef_ab_numpy, coef_ab_glm, atol=1e-8)
    assert np.isclose(delta_offset_numpy, delta_offset_glm, atol=1e-8)
    assert np.isclose(delta_offset_numpy, DELTA_PIT, atol=1e-8)
    assert np.allclose(platt_a_b, coef_ab_glm, atol=1e-6)
    _checks.append("IRLS = GLM; offset = δ; Platt ≡ logística")

    # 8. Isotónica: PAV numpy = sklearn (entrenamiento y predicción) y es monótona
    assert np.allclose(pd_iso_ttd_numpy, pd_iso_ttd_sklearn, atol=1e-12)
    assert np.allclose(np.interp(xu_iso, xu_iso, ajuste_iso), iso_sk.predict(xu_iso), atol=1e-12)
    assert np.all(np.diff(ajuste_iso) >= -1e-15)
    # PAV conserva la media en la muestra de ajuste
    _lp_pit = muestras["OOT"]["lp_modelo"].values[es_pit]
    assert np.isclose(iso_sk.predict(_lp_pit).mean(), y_pit.mean(), atol=1e-12)
    _checks.append("PAV numpy = sklearn; monótona; conserva la media")

    # 9. Métricas: Murphy exacta y = sklearn; log-loss y curva = sklearn
    for _k, _p in predicciones_ttd.items():
        _m = murphy(_p, y_val)
        assert np.isclose(_m["suma"], _m["brier"], atol=1e-12)
        assert np.isclose(_m["brier"], brier_score_loss(y_val, _p), atol=1e-12)
        _pc = np.clip(_p, 1e-15, 1 - 1e-15)
        assert np.isclose(logloss_numpy(_pc, y_val), log_loss(y_val, _pc), atol=1e-10)
        _pt, _pp = calibration_curve(y_val, _p, n_bins=10, strategy="quantile")
        _o, _q, _, _ = curva_numpy(_p, y_val, 10)
        assert np.allclose(_pt, _o) and np.allclose(_pp, _q)
        assert ece_numpy(_p, y_val) >= 0
    # un modelo recalibrado en el mismo dato tiene O/E = 1
    assert np.isclose(y_pit.sum() / sigmoide_np(_lp_pit + DELTA_PIT).sum(), 1.0)
    _checks.append("Murphy exacta; Brier/log-loss/curva = sklearn")

    # 10. IC Wilson/Jeffreys numpy = statsmodels
    assert ic_coinciden
    _checks.append("Wilson y Jeffreys = statsmodels")

    # 11. Misma muestra: p = 1; hipótesis TTC falsa en PIT: se rechaza
    assert np.isclose(p_misma_muestra, 1.0)
    assert p_ttc_en_pit < 0.01
    _checks.append("misma muestra p = 1; TTC en PIT rechaza")

    # 12. Orden de niveles plantado por el generador: DEV < TC < PIT
    assert muestras["DEV"]["malo"].mean() < TC < y_pit.mean()
    assert 0 < DELTA_TTC < DELTA_PIT
    assert tabla_metricas.loc["δ PIT (mar–jun 2025)", "o_e"] < tabla_metricas.loc["δ TTC (DEV→TC)", "o_e"]
    _checks.append("DEV < TC < PIT; δ_TTC < δ_PIT")

    mo.md("**Todos los checks pasaron.**\n\n" + "\n".join(f"- {c}" for c in _checks))
    return


if __name__ == "__main__":
    app.run()

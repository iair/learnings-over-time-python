# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
#     "statsmodels",
#     "scikit-learn>=1.8",
# ]
# ///
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import time
    import json
    import hashlib
    import warnings
    from collections import Counter
    from scipy import stats, optimize
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    return (
        Counter,
        LogisticRegression,
        hashlib,
        json,
        mo,
        optimize,
        plt,
        roc_auc_score,
        sm,
        stats,
        time,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # M11 · Selección de variables: stepwise y sus alternativas

    **Serie 2 «Del embudo al gobierno»** · profundiza la clase 3 (paso 5 del embudo: 26 → 8) y el Lab 2 §5.

    Este notebook construye, desde cero y con librerías, todo lo que hay detrás de la frase
    «stepwise forward con revisión backward, p < 0,05, signo esperado, tope 14»:

    1. Logística por IRLS en numpy, contra `statsmodels` y `scikit-learn` (base de todo lo demás).
    2. Un **stepwise genérico** (criterio Wald / LR / score / AIC / BIC, umbral, tope, política de signos,
       variables forzadas) con **bitácora** de decisiones, en numpy y con `statsmodels`.
    3. **Por qué entra el ruido** cuando el WoE se ajusta en la misma muestra en que se selecciona, y el
       remedio (WoE cruzado).
    4. Equivalencia entre criterios (p-valor, AIC, BIC) y la trayectoria real del Banco Austral.
    5. La **paradoja de Freedman** (1983) y la **inferencia post-selección** con verdad conocida.
    6. **Bootstrap del stepwise** (frecuencia de inclusión) y **stability selection**.
    7. **LASSO / elastic net con restricción de signo** y su camino de regularización.
    8. Comparación de enfoques en HO/OOT y el costo de **forzar** una variable (LR test).
    9. La bitácora como artefacto versionado (hash).

    Convenciones del curso: target 1 = malo; WoE = ln(%buenos/%malos) ⇒ coeficientes **negativos**;
    todo se ajusta en DEV y se aplica a HO/OOT.
    """)
    return


@app.cell
def _():
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
    ## 0. La cartera y el pool de candidatas (con verdad conocida)

    Usamos `generar_cartera()` de la serie. Sabemos **exactamente** qué variables entran al log-odds
    verdadero, y eso permite algo imposible con datos reales: medir cuántas veces un método de selección
    elige **ruido**. El pool tiene tres tipos de candidatas:

    - **señal directa**: aparecen en el log-odds del generador (utilización, mora, antigüedad, deuda,
      carga, consultas, canal);
    - **proxy**: `edad` y `renta_mm` no tienen efecto directo, pero están correlacionadas con variables
      que sí (edad ↔ antigüedad; renta → carga financiera);
    - **ruido puro**: `ruido_k ~ N(0,1)`, independientes de todo (el slider fija cuántas).

    Con `n = 8.000` el DEV queda en ≈ 3.400 solicitudes, el tamaño del DEV del Banco Austral (3.322),
    aunque con más malos (la tasa del generador es ≈ 11–12%, no 5%).
    """)
    return


@app.cell
def _(mo):
    ui_n = mo.ui.dropdown(
        options={"4.000": 4000, "8.000 (DEV ≈ Austral)": 8000, "24.000": 24000},
        value="8.000 (DEV ≈ Austral)", label="Tamaño de la cartera")
    ui_ruido = mo.ui.slider(0, 30, value=10, step=1, label="Nº de variables de ruido puro")
    mo.hstack([ui_n, ui_ruido], justify="start")
    return ui_n, ui_ruido


@app.cell
def _(a_woe, generar_cartera, np, pd, tabla_woe, ui_n, ui_ruido):
    _base = generar_cartera(n=ui_n.value)
    _rng = np.random.default_rng(1983)
    RUIDO = [f"ruido_{k + 1:02d}" for k in range(ui_ruido.value)]
    cartera = _base.assign(**{v: _rng.normal(size=len(_base)) for v in RUIDO})

    SENAL = ["uso_linea_prom_12m", "uso_tc_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m",
             "antiguedad_meses", "deuda_otras_prom_12m", "carga_financiera", "consultas_6m", "canal"]
    PROXY = ["edad", "renta_mm"]
    CANDIDATAS = SENAL[:5] + ["edad", "renta_mm"] + SENAL[5:] + RUIDO
    VERDAD = {**{v: "señal" for v in SENAL}, **{v: "proxy" for v in PROXY},
              **{v: "ruido" for v in RUIDO}}
    CONCEPTO = {"uso_linea_prom_12m": "utilización", "uso_tc_prom_12m": "utilización",
                "uso_tc_prom_3m": "utilización", "meses_desde_mora_12m": "conducta de pago",
                "antiguedad_meses": "relación", "edad": "demografía", "renta_mm": "capacidad",
                "deuda_otras_prom_12m": "endeudamiento", "carga_financiera": "capacidad",
                "consultas_6m": "apetito de crédito", "canal": "originación",
                **{v: "—" for v in RUIDO}}

    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    ho = cartera[cartera["muestra"] == "HO"].reset_index(drop=True)
    oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)
    y_dev = dev["malo"].to_numpy(float)
    y_ho = ho["malo"].to_numpy(float)
    y_oot = oot["malo"].to_numpy(float)

    mapas, IV = {}, {}
    for _v in CANDIDATAS:
        _tab, IV[_v] = tabla_woe(dev[_v], dev["malo"])
        mapas[_v] = _tab["woe"].to_dict()
    W_dev = a_woe(dev, CANDIDATAS, dev, mapas)      # WoE ajustado y aplicado en DEV (curso)
    W_ho = a_woe(ho, CANDIDATAS, dev, mapas)
    W_oot = a_woe(oot, CANDIDATAS, dev, mapas)

    tabla_pool = pd.DataFrame({"IV_DEV": pd.Series(IV), "verdad": pd.Series(VERDAD),
                               "concepto": pd.Series(CONCEPTO)}).loc[CANDIDATAS]
    tabla_pool = tabla_pool.sort_values("IV_DEV", ascending=False).round(4)
    resumen_muestras = (f"DEV = {len(dev):,} ({int(y_dev.sum()):,} malos, {y_dev.mean():.1%}) · "
                f"HO = {len(ho):,} ({int(y_ho.sum()):,} malos) · "
                f"OOT = {len(oot):,} ({int(y_oot.sum()):,} malos, {y_oot.mean():.1%})").replace(",", ".")
    return (
        CANDIDATAS,
        CONCEPTO,
        IV,
        RUIDO,
        VERDAD,
        W_dev,
        W_ho,
        W_oot,
        dev,
        tabla_pool,
        y_dev,
        y_ho,
        y_oot,
        resumen_muestras,
    )


@app.cell
def _(mo, resumen_muestras, tabla_pool):
    mo.vstack([mo.md(f"**Muestras:** {resumen_muestras}"), tabla_pool])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### WoE cruzado (*out-of-fold*): la herramienta que vamos a necesitar

    El curso ajusta cortes y WoE en DEV y **selecciona en el mismo DEV**. Eso es correcto para el modelo
    final, pero contamina la **selección**: el WoE de una variable de ruido ya está «acomodado» a los
    malos de DEV. La alternativa es el WoE cruzado: se parte DEV en $K$ pliegues y el WoE de cada fila se
    calcula con cortes y tasas de los **otros** $K-1$ pliegues (la misma idea que el *target encoding*
    con validación cruzada). Solo se usa **para decidir qué entra**; el modelo final se re-ajusta con el
    WoE de DEV completo, como en el curso.
    """)
    return


@app.cell
def _(CANDIDATAS, a_woe, dev, np, pd, tabla_woe, time):
    def woe_cruzado(df, variables, K=5, semilla=7):
        """WoE out-of-fold: cortes y WoE de cada pliegue salen de los otros K-1."""
        _rng = np.random.default_rng(semilla)
        _pl = _rng.integers(0, K, len(df))
        _W = pd.DataFrame(0.0, index=range(len(df)), columns=list(variables))
        for _k in range(K):
            _otros = df[_pl != _k].reset_index(drop=True)
            _este = df[_pl == _k].reset_index(drop=True)
            _mp = {v: tabla_woe(_otros[v], _otros["malo"])[0]["woe"].to_dict() for v in variables}
            _W.iloc[np.where(_pl == _k)[0], :] = a_woe(_este, variables, _otros, _mp).to_numpy()
        return _W

    _t0 = time.time()
    W_cruz = woe_cruzado(dev, CANDIDATAS, K=5)
    t_cruz = time.time() - _t0
    return W_cruz, t_cruz


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. El motor: logística por IRLS (numpy) contra statsmodels y scikit-learn

    Todo método de selección es un bucle que ajusta cientos de logísticas. Implementamos Newton-Raphson
    (= IRLS) con gradiente $X^\top(y-p)$ y Hessiano $-X^\top W X$ (derivación en Serie 2 · M10), y el
    **test score de Rao** vectorizado para *todas* las candidatas a la vez:

    $$U_j = z_j^\top (y-\hat p_0), \qquad
    I_j = z_j^\top W z_j - z_j^\top W X\,(X^\top W X)^{-1} X^\top W z_j, \qquad S_j = U_j^2 / I_j \sim \chi^2_1 .$$

    El score solo necesita el modelo **actual** (no ajusta cada candidata): es lo que usa SAS
    `PROC LOGISTIC` para la entrada en stepwise.
    """)
    return


@app.cell
def _(np, sm):
    def irls(X, y, max_iter=60, tol=1e-10):
        """Logística por Newton-Raphson/IRLS. X incluye la columna de unos."""
        _p = X.shape[1]
        _b = np.zeros(_p)
        _yb = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        _b[0] = np.log(_yb / (1 - _yb))
        _it = 0
        for _it in range(max_iter):
            _mu = 1.0 / (1.0 + np.exp(-(X @ _b)))
            _w = _mu * (1 - _mu)
            _H = (X * _w[:, None]).T @ X                 # información de Fisher = -Hessiano
            _paso = np.linalg.solve(_H, X.T @ (y - _mu))
            _b = _b + _paso
            if np.max(np.abs(_paso)) < tol:
                break
        _eta = X @ _b
        _mu = 1.0 / (1.0 + np.exp(-_eta))
        _w = _mu * (1 - _mu)
        _H = (X * _w[:, None]).T @ X
        _cov = np.linalg.inv(_H)
        _llf = float(np.sum(y * _eta - np.logaddexp(0.0, _eta)))
        return {"beta": _b, "se": np.sqrt(np.diag(_cov)), "llf": _llf, "mu": _mu, "w": _w,
                "H": _H, "iter": _it + 1}

    def ajustar_sm(X, y):
        """Mismo contrato que `irls`, pero estimado con statsmodels.Logit (Newton)."""
        _r = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100, tol=1e-12)
        _mu = np.asarray(_r.predict(X))
        _w = _mu * (1 - _mu)
        return {"beta": np.asarray(_r.params), "se": np.asarray(_r.bse), "llf": float(_r.llf),
                "mu": _mu, "w": _w, "H": (X * _w[:, None]).T @ X,
                "iter": _r.mle_retvals["iterations"]}

    def score_todas(X, ajuste, Z, y):
        """Test score de Rao (1 gl) para agregar cada columna de Z al modelo X ya ajustado."""
        _U = Z.T @ (y - ajuste["mu"])
        _A = (X * ajuste["w"][:, None]).T @ Z
        _I = (Z ** 2 * ajuste["w"][:, None]).sum(0) - np.sum(_A * np.linalg.solve(ajuste["H"], _A), 0)
        return _U ** 2 / _I

    def auc_np(y, s):
        """AUC por rangos (Mann-Whitney) con empates promediados; s alto = más riesgo."""
        _o = np.argsort(s, kind="mergesort")
        _, _inv, _cnt = np.unique(s[_o], return_inverse=True, return_counts=True)
        _fin = np.cumsum(_cnt)
        _r = np.empty(len(s))
        _r[_o] = ((_fin - _cnt + 1 + _fin) / 2.0)[_inv]
        _n1 = y.sum()
        _n0 = len(y) - _n1
        return float((_r[y == 1].sum() - _n1 * (_n1 + 1) / 2) / (_n1 * _n0))

    def disenar(W, variables):
        return np.column_stack([np.ones(len(W))] + [W[v].to_numpy(float) for v in variables])
    return ajustar_sm, auc_np, disenar, irls, score_todas


@app.cell
def _(
    CANDIDATAS,
    LogisticRegression,
    W_dev,
    ajustar_sm,
    disenar,
    irls,
    np,
    pd,
    score_todas,
    sm,
    warnings,
    y_dev,
):
    _X = disenar(W_dev, CANDIDATAS)
    _a = irls(_X, y_dev)
    _s = ajustar_sm(_X, y_dev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(_X[:, 1:], y_dev)
    _b_sk = np.r_[_sk.intercept_, _sk.coef_[0]]
    # test score: agregar 'edad' a un modelo con uso_linea + mora
    _X0 = disenar(W_dev, ["uso_linea_prom_12m", "meses_desde_mora_12m"])
    _a0 = irls(_X0, y_dev)
    _S_np = score_todas(_X0, _a0, W_dev[["edad"]].to_numpy(float), y_dev)[0]
    _g0 = sm.GLM(y_dev, _X0, family=sm.families.Binomial()).fit()
    _S_sm = float(np.ravel(_g0.score_test(exog_extra=W_dev[["edad"]].to_numpy(float)).statistic)[0])

    chk_irls = {
        "max|Δβ| numpy vs statsmodels": float(np.max(np.abs(_a["beta"] - _s["beta"]))),
        "max|ΔSE| numpy vs statsmodels": float(np.max(np.abs(_a["se"] - _s["se"]))),
        "|Δ log-verosimilitud|": abs(_a["llf"] - _s["llf"]),
        "max|Δβ| numpy vs sklearn (C=inf)": float(np.max(np.abs(_a["beta"] - _b_sk))),
        "|Δ score| numpy vs GLM.score_test": abs(_S_np - _S_sm),
    }
    tabla_modelo_completo = pd.DataFrame(
        {"beta_numpy": _a["beta"], "beta_statsmodels": _s["beta"], "beta_sklearn": _b_sk,
         "se_numpy": _a["se"], "se_statsmodels": _s["se"]},
        index=["const"] + CANDIDATAS).round(5)
    iter_irls = _a["iter"]
    return chk_irls, iter_irls, tabla_modelo_completo


@app.cell
def _(chk_irls, iter_irls, mo, pd, tabla_modelo_completo):
    mo.vstack([
        mo.md(f"IRLS convergió en **{iter_irls} iteraciones** (modelo con todas las candidatas). "
              "Diferencias máximas contra las librerías:"),
        pd.Series(chk_irls, name="diferencia").to_frame().map(lambda v: f"{v:.2e}"),
        mo.md("Modelo **con todas** las candidatas (referencia; fíjense en los signos positivos que "
              "aparecen por supresión y en los β ≈ −1 del ruido):"),
        tabla_modelo_completo,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Stepwise genérico con bitácora

    Parámetros: **criterio** de entrada (Wald = el del curso vía `pvalues` de statsmodels; LR; score;
    AIC; BIC), **umbral** α, **tope** de variables, **política de signos** y variables **forzadas**.
    Entre las candidatas que superan el umbral entra la de **mayor log-verosimilitud** (como el curso;
    con score, la de mayor $S_j$, como SAS). Tras cada entrada, **revisión backward**: sale **la peor**
    (primero cualquier signo positivo si la política es «todas»; luego el menor Wald bajo el umbral de
    salida) y se re-estima antes de volver a mirar. Para AIC/BIC la salida usa el Wald contra el mismo
    umbral en escala χ² (2 o ln n). Se detiene si ninguna entra, si se alcanza el tope o si un modelo ya
    visitado se repite (ciclo).

    Cada decisión queda en la **bitácora** (ronda, acción, variable, estadístico, p, Δ log-verosimilitud).
    """)
    return


@app.cell
def _(np, pd, score_todas, stats, irls):
    def umbral_chi2(criterio, alfa, n):
        """Umbral en escala χ²(1 gl) que implica cada criterio."""
        if criterio == "aic":
            return 2.0
        if criterio == "bic":
            return float(np.log(n))
        return float(stats.chi2.ppf(1 - alfa, 1))

    def stepwise(W, y, candidatas, criterio="wald", alfa_entrada=0.05, alfa_salida=0.05, tope=14,
                 politica_signo="todas", forzadas=(), ajustador=irls, max_rondas=80):
        """Stepwise forward con revisión backward. Devuelve (elegidas, bitácora, ajuste final)."""
        _n = len(y)
        _M = W.to_numpy(float)
        _col = {v: i for i, v in enumerate(W.columns)}
        _uno = np.ones((_n, 1))

        def _dis(vs):
            return np.hstack([_uno, _M[:, [_col[v] for v in vs]]])

        def _signo_ok(aj, vs, v):
            _b = aj["beta"][1:]
            if politica_signo == "ninguna":
                return True, ""
            if politica_signo == "entrante":
                return bool(_b[-1] < 0), f"coef. de {v} = {_b[-1]:+.3f} > 0"
            _pos = [u for u, bb in zip(vs, _b) if bb > 0 and u not in forzadas]
            return (not _pos), ("signo positivo en: " + ", ".join(_pos)) if _pos else ""

        _c_in = umbral_chi2(criterio, alfa_entrada, _n)
        _c_out = umbral_chi2(criterio, alfa_salida, _n) if criterio in ("wald", "lr", "score") else _c_in
        elegidas = list(forzadas)
        restantes = [v for v in candidatas if v not in elegidas]
        aj = ajustador(_dis(elegidas), y)
        bit = [dict(ronda=0, accion="forzada", variable=v, estadistico=np.nan, p_valor=np.nan,
                    delta_llf=np.nan, llf=aj["llf"], n_vars=len(elegidas),
                    motivo="decisión de comité (no puede salir)") for v in forzadas]
        vistos = {frozenset(elegidas)}
        ronda = 0
        motivo_fin = "sin candidatas restantes"
        while restantes and len(elegidas) < tope and ronda < max_rondas:
            ronda += 1
            elegida = None
            if criterio == "score":
                _S = score_todas(_dis(elegidas), aj, _M[:, [_col[v] for v in restantes]], y)
                for _k in np.argsort(-_S):
                    if _S[_k] <= _c_in:
                        break
                    _v = restantes[_k]
                    _a1 = ajustador(_dis(elegidas + [_v]), y)
                    _ok, _mot = _signo_ok(_a1, elegidas + [_v], _v)
                    if _ok:
                        elegida = (_v, float(_S[_k]), _a1)
                        break
                    bit.append(dict(ronda=ronda, accion="rechazo_signo", variable=_v,
                                    estadistico=float(_S[_k]), p_valor=stats.chi2.sf(_S[_k], 1),
                                    delta_llf=_a1["llf"] - aj["llf"], llf=_a1["llf"],
                                    n_vars=len(elegidas), motivo=_mot))
            else:
                _ops = []
                for _v in restantes:
                    _a1 = ajustador(_dis(elegidas + [_v]), y)
                    if criterio == "wald":
                        _s = float((_a1["beta"][-1] / _a1["se"][-1]) ** 2)
                    else:
                        _s = float(2 * (_a1["llf"] - aj["llf"]))
                    if _s > _c_in:
                        _ops.append((_v, _s, _a1))
                _ops.sort(key=lambda t: -t[2]["llf"])
                for _v, _s, _a1 in _ops:
                    _ok, _mot = _signo_ok(_a1, elegidas + [_v], _v)
                    if _ok:
                        elegida = (_v, _s, _a1)
                        break
                    bit.append(dict(ronda=ronda, accion="rechazo_signo", variable=_v, estadistico=_s,
                                    p_valor=stats.chi2.sf(_s, 1), delta_llf=_a1["llf"] - aj["llf"],
                                    llf=_a1["llf"], n_vars=len(elegidas), motivo=_mot))
            if elegida is None:
                motivo_fin = "ninguna candidata supera el umbral con signo admisible"
                break
            _v, _s, _a1 = elegida
            bit.append(dict(ronda=ronda, accion="entra", variable=_v, estadistico=_s,
                            p_valor=stats.chi2.sf(_s, 1), delta_llf=_a1["llf"] - aj["llf"],
                            llf=_a1["llf"], n_vars=len(elegidas) + 1, motivo=f"mejor candidata ({criterio})"))
            elegidas.append(_v)
            restantes.remove(_v)
            aj = _a1
            while True:                                   # revisión backward: sale LA PEOR
                _b = aj["beta"][1:]
                _z2 = (_b / aj["se"][1:]) ** 2
                _malas = []
                for _i, _u in enumerate(elegidas):
                    if _u in forzadas:
                        continue
                    if politica_signo == "todas" and _b[_i] > 0:
                        _malas.append((np.inf, _i, _u, "signo positivo tras re-estimar"))
                    elif _z2[_i] <= _c_out:
                        _malas.append((1.0 / max(_z2[_i], 1e-300), _i, _u, "perdió significancia (Wald)"))
                if not _malas:
                    break
                _, _i, _u, _mot = max(_malas)
                _prev = aj["llf"]
                elegidas.remove(_u)
                restantes.append(_u)
                aj = ajustador(_dis(elegidas), y)
                bit.append(dict(ronda=ronda, accion="sale", variable=_u, estadistico=float(_z2[_i]),
                                p_valor=stats.chi2.sf(_z2[_i], 1), delta_llf=aj["llf"] - _prev,
                                llf=aj["llf"], n_vars=len(elegidas), motivo=_mot))
            _clave = frozenset(elegidas)
            if _clave in vistos and bit[-1]["accion"] == "sale":
                motivo_fin = "ciclo: el modelo ya había sido visitado"
                break
            vistos.add(_clave)
            if len(elegidas) >= tope:
                motivo_fin = f"tope de {tope} variables"
        bit.append(dict(ronda=ronda, accion="fin", variable="", estadistico=np.nan, p_valor=np.nan,
                        delta_llf=np.nan, llf=aj["llf"], n_vars=len(elegidas), motivo=motivo_fin))
        return elegidas, pd.DataFrame(bit), aj
    return stepwise, umbral_chi2


@app.cell
def _(W_dev, W_ho, W_oot, auc_np, disenar, irls, np, y_dev, y_ho, y_oot):
    def evaluar(variables):
        """Re-ajusta en DEV con el WoE de DEV (como producción) y mide Gini en DEV/HO/OOT."""
        _vs = list(variables)
        _a = irls(disenar(W_dev, _vs), y_dev)
        _out = {"vars": _vs, "beta": _a["beta"], "se": _a["se"], "llf": _a["llf"]}
        for _nom, _W, _y in [("dev", W_dev, y_dev), ("ho", W_ho, y_ho), ("oot", W_oot, y_oot)]:
            _p = 1.0 / (1.0 + np.exp(-(disenar(_W, _vs) @ _a["beta"])))
            _out[f"p_{_nom}"] = _p
            _out[f"gini_{_nom}"] = 2 * auc_np(_y, _p) - 1
        return _out
    return (evaluar,)


@app.cell
def _(mo):
    ui_crit = mo.ui.dropdown(
        options={"Wald p (curso)": "wald", "LR p": "lr", "Score p (SAS)": "score", "AIC": "aic",
                 "BIC": "bic"}, value="Wald p (curso)", label="Criterio de entrada")
    ui_alfa = mo.ui.slider(0.001, 0.20, value=0.05, step=0.001, label="α entrada = α salida")
    ui_tope = mo.ui.slider(3, 25, value=14, step=1, label="Tope de variables")
    ui_signo = mo.ui.dropdown(
        options={"todas negativas (curso)": "todas", "solo la entrante": "entrante",
                 "sin política de signos": "ninguna"},
        value="todas negativas (curso)", label="Política de signos")
    ui_modo = mo.ui.dropdown(
        options={"WoE ajustado en DEV (curso)": "in", "WoE cruzado K=5": "cruz"},
        value="WoE ajustado en DEV (curso)", label="WoE usado para SELECCIONAR")
    mo.vstack([mo.hstack([ui_crit, ui_alfa, ui_tope], justify="start"),
               mo.hstack([ui_signo, ui_modo], justify="start")])
    return ui_alfa, ui_crit, ui_modo, ui_signo, ui_tope


@app.cell
def _(
    CANDIDATAS,
    CONCEPTO,
    VERDAD,
    W_cruz,
    W_dev,
    ajustar_sm,
    evaluar,
    np,
    pd,
    stats,
    stepwise,
    time,
    ui_alfa,
    ui_crit,
    ui_modo,
    ui_signo,
    ui_tope,
    y_dev,
):
    W_sel = W_dev if ui_modo.value == "in" else W_cruz
    _cfg = dict(criterio=ui_crit.value, alfa_entrada=ui_alfa.value, alfa_salida=ui_alfa.value,
                tope=ui_tope.value, politica_signo=ui_signo.value)
    _t0 = time.time()
    sel_main, bit_main, _ = stepwise(W_sel, y_dev, CANDIDATAS, **_cfg)
    t_step_np = time.time() - _t0
    _t0 = time.time()
    sel_main_sm, bit_main_sm, _ = stepwise(W_sel, y_dev, CANDIDATAS, ajustador=ajustar_sm, **_cfg)
    t_step_sm = time.time() - _t0
    cfg_main = _cfg | {"modo_woe": ui_modo.value}

    modelo_main = evaluar(sel_main)
    _z = modelo_main["beta"][1:] / modelo_main["se"][1:]
    tabla_main = pd.DataFrame({
        "beta": modelo_main["beta"][1:], "se": modelo_main["se"][1:],
        "p_wald": 2 * stats.norm.sf(np.abs(_z)),
        "verdad": [VERDAD[v] for v in sel_main], "concepto": [CONCEPTO[v] for v in sel_main]},
        index=sel_main).round(4)
    return (
        W_sel,
        bit_main,
        bit_main_sm,
        cfg_main,
        modelo_main,
        sel_main,
        sel_main_sm,
        t_step_np,
        t_step_sm,
        tabla_main,
    )


@app.cell
def _(
    bit_main,
    mo,
    modelo_main,
    sel_main,
    sel_main_sm,
    t_step_np,
    t_step_sm,
    tabla_main,
):
    _bit = bit_main.copy()
    _bit["p_valor"] = _bit["p_valor"].map(lambda v: "" if v != v else f"{v:.1e}")
    mo.vstack([
        mo.md(f"**Bitácora del stepwise** (numpy: {t_step_np:.2f} s · statsmodels: {t_step_sm:.2f} s · "
              f"misma selección: **{sel_main == sel_main_sm}**)"),
        _bit.round(3),
        mo.md(f"**Modelo final** (re-ajustado con WoE de DEV) · Gini DEV "
              f"{modelo_main['gini_dev']:.3f} · HO {modelo_main['gini_ho']:.3f} · OOT "
              f"{modelo_main['gini_oot']:.3f}"),
        tabla_main,
    ])
    return


@app.cell
def _(VERDAD, mo, sel_main, ui_modo):
    _n_r = sum(VERDAD[v] == "ruido" for v in sel_main)
    _n_s = sum(VERDAD[v] == "señal" for v in sel_main)
    _n_p = sum(VERDAD[v] == "proxy" for v in sel_main)
    _modo = "ajustado en DEV (curso)" if ui_modo.value == "in" else "cruzado"
    mo.md(f"""
    **Lectura.** Con WoE {_modo}, el stepwise eligió **{len(sel_main)}** variables:
    **{_n_s}** de señal directa, **{_n_p}** proxy y **{_n_r}** de ruido puro. Si hay ruido con p < 0,05
    y signo «correcto», no es mala suerte: la sección 3 muestra que con WoE ajustado en la misma muestra
    **cada** variable de ruido de 5 bins pasa un test Wald/LR al 5% con probabilidad cercana a 0,43.
    Cambien el selector a «WoE cruzado» y vuelvan a mirar la bitácora.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. ¿Por qué entra el ruido? Grados de libertad escondidos en el WoE

    Con una sola variable en WoE (convención $\ln(\%B/\%M)$ y sin suavizado), la logística reproduce el
    modelo **saturado por bins**: $\hat\beta=-1$ y la verosimilitud es la del modelo con un parámetro por
    bin (Serie 2 · M10). Por lo tanto el estadístico LR «de 1 gl» es en realidad el $G^2$ de la tabla
    $B\times 2$:

    $$\text{LR}_{\text{WoE in-sample}} \;\approx\; G^2_{B\times 2} \;\overset{H_0}{\sim}\; \chi^2_{B-1}
    \quad\Longrightarrow\quad
    \Pr(\text{entra} \mid \text{ruido}) \approx \Pr\!\left(\chi^2_{B-1} > \chi^2_{1;\,0.95}=3{,}84\right).$$

    Y el IV del ruido: $\text{IV}\cdot\left(\tfrac{1}{M}+\tfrac{1}{G}\right)^{-1} \approx \chi^2_{B-1}$, de
    donde $\mathbb E[\text{IV}_{\text{ruido}}]\approx (B-1)\left(\tfrac{1}{M}+\tfrac{1}{G}\right)$ (mismo
    argumento que el PSI en M08). El filtro IV ≥ 0,10 del curso es, sin decirlo, el que protege contra el
    ruido; el p < 0,05 del stepwise casi no lo hace.
    """)
    return


@app.cell
def _(
    CANDIDATAS,
    IV,
    RUIDO,
    VERDAD,
    W_cruz,
    W_dev,
    np,
    pd,
    stats,
    stepwise,
    y_dev,
):
    _c1 = stats.chi2.ppf(0.95, 1)
    tabla_gl = pd.DataFrame({
        "B_bins": list(range(2, 11)),
        "P(entra | ruido), test nominal 5%": [stats.chi2.sf(_c1, b - 1) for b in range(2, 11)],
    }).set_index("B_bins").round(3)

    _M = y_dev.sum()
    _G = len(y_dev) - _M
    _escala = 1 / _M + 1 / _G
    _M_aus, _G_aus = 165, 3322 - 165
    _escala_aus = 1 / _M_aus + 1 / _G_aus
    tabla_iv_nulo = pd.DataFrame({
        "muestra": ["DEV generador", "DEV Austral (165 malos)"],
        "E[IV ruido] ≈ 4·(1/M+1/G)": [4 * _escala, 4 * _escala_aus],
        "P(IV ruido ≥ 0,10)": [stats.chi2.sf(0.10 / _escala, 4), stats.chi2.sf(0.10 / _escala_aus, 4)],
        "P(IV ruido ≥ 0,02)": [stats.chi2.sf(0.02 / _escala, 4), stats.chi2.sf(0.02 / _escala_aus, 4)],
    }).set_index("muestra")
    iv_ruido_obs = float(np.mean([IV[v] for v in RUIDO])) if RUIDO else float("nan")
    iv_ruido_teo = 4 * _escala

    _filas = []
    for _nom, _W in [("WoE ajustado en DEV (curso)", W_dev), ("WoE cruzado K=5", W_cruz)]:
        _s, _, _ = stepwise(_W, y_dev, CANDIDATAS)
        _filas.append({"modo": _nom, "n_vars": len(_s),
                       "señal": sum(VERDAD[v] == "señal" for v in _s),
                       "proxy": sum(VERDAD[v] == "proxy" for v in _s),
                       "ruido": sum(VERDAD[v] == "ruido" for v in _s),
                       "variables": ", ".join(_s)})
    tabla_modos = pd.DataFrame(_filas).set_index("modo")
    return iv_ruido_obs, iv_ruido_teo, tabla_gl, tabla_iv_nulo, tabla_modos


@app.cell
def _(iv_ruido_obs, iv_ruido_teo, mo, tabla_gl, tabla_iv_nulo, tabla_modos):
    mo.vstack([
        mo.md("**Tamaño real del test p < 0,05 sobre WoE in-sample** (una variable de ruido, B bins):"),
        tabla_gl.T,
        mo.md(f"**IV del ruido.** Promedio observado en el pool: **{iv_ruido_obs:.4f}** · teórico "
              f"{iv_ruido_teo:.4f}. Probabilidad de que una variable de ruido pase el filtro IV:"),
        tabla_iv_nulo.map(lambda v: f"{v:.2e}" if v < 1e-3 else f"{v:.4f}"),
        mo.md("**Stepwise del curso (Wald 0,05, signo, tope 14) con cada WoE:**"),
        tabla_modos,
        mo.md("""
    **Lectura.** Con 5 bins, el «test al 5%» es un test al ≈ 43% para una variable de ruido. En Austral
    no se notó porque el filtro IV ≥ 0,10 ya había sacado el ruido (con 165 malos, P(IV ruido ≥ 0,10)
    ≈ 0,4%). Con WoE cruzado el ruido deja de entrar, pero también se pierden señales débiles: el test
    se vuelve honesto y, por lo tanto, más conservador.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. p-valor, AIC y BIC son el mismo test con distinto umbral

    Sobre WoE cada variable ocupa **1 grado de libertad**. Agregar $x_j$ cambia la log-verosimilitud en
    $\Delta\ell_j$ y el LR es $2\Delta\ell_j\sim\chi^2_1$. Entonces:

    | criterio | entra si | α equivalente |
    |---|---|---|
    | LR p < α | $2\Delta\ell > \chi^2_{1;1-\alpha}$ | α |
    | AIC baja | $2\Delta\ell > 2$ | $\Pr(\chi^2_1>2)=0{,}157$ |
    | BIC baja | $2\Delta\ell > \ln n$ | $\Pr(\chi^2_1>\ln n)$ (depende de n) |

    Abajo: los umbrales para el DEV del generador y del Banco Austral, y la trayectoria **real** del
    stepwise de Austral (log-verosimilitudes de la clase 3) re-leída con cada criterio.
    """)
    return


@app.cell
def _(np, pd, stats, y_dev):
    def alfa_equivalente(n):
        return {"AIC": stats.chi2.sf(2.0, 1), "BIC": stats.chi2.sf(np.log(n), 1),
                "umbral Δℓ AIC": 1.0, "umbral Δℓ BIC": np.log(n) / 2,
                "umbral Δℓ p<0,05": stats.chi2.ppf(0.95, 1) / 2,
                "umbral Δℓ p<0,01": stats.chi2.ppf(0.99, 1) / 2}

    tabla_equiv = pd.DataFrame({f"DEV generador (n={len(y_dev)})": alfa_equivalente(len(y_dev)),
                                "DEV Austral (n=3322)": alfa_equivalente(3322)}).round(4)

    # Trayectoria del stepwise del Banco Austral (clase 3, lab_clase3_austral §6)
    _vars = ["uso_linea_prom_12m", "meses_desde_mora_12m", "uso_tc_prom_12m", "deuda_interna_max_3m",
             "antiguedad_meses", "deuda_otras_prom_12m", "carga_financiera", "uso_tc_prom_3m"]
    _ll = np.array([-540.7, -512.8, -501.7, -495.4, -491.1, -486.7, -481.6, -478.5])
    _p_wald = np.array([3.0e-34, 2.3e-15, 1.1e-05, 4.2e-04, 4.3e-03, 7.7e-03, 1.4e-03, 1.5e-02])
    _M, _N = 165, 3322                               # malos DEV (3+4+11+37+110) y tamaño DEV
    ll0_austral = _M * np.log(_M / _N) + (_N - _M) * np.log(1 - _M / _N)
    _dll = np.diff(np.r_[ll0_austral, _ll])
    _lr = 2 * _dll
    trayectoria_austral = pd.DataFrame({
        "variable": _vars, "log_vero": _ll, "Δℓ": _dll, "LR=2Δℓ": _lr,
        "p_LR": stats.chi2.sf(_lr, 1), "p_Wald (curso)": _p_wald,
        "p<0,05": _dll > stats.chi2.ppf(0.95, 1) / 2, "p<0,01": _dll > stats.chi2.ppf(0.99, 1) / 2,
        "AIC": _dll > 1.0, "BIC": _dll > np.log(_N) / 2}, index=range(1, 9))
    return alfa_equivalente, ll0_austral, tabla_equiv, trayectoria_austral


@app.cell
def _(ll0_austral, mo, tabla_equiv, trayectoria_austral):
    _t = trayectoria_austral.copy()
    _t["p_LR"] = _t["p_LR"].map(lambda v: f"{v:.1e}")
    _t["p_Wald (curso)"] = _t["p_Wald (curso)"].map(lambda v: f"{v:.1e}")
    _n_bic = int(trayectoria_austral["BIC"].cumprod().sum())
    mo.vstack([
        tabla_equiv,
        mo.md(f"**Banco Austral.** Log-verosimilitud del modelo nulo (165 malos en 3.322): "
              f"ℓ₀ = {ll0_austral:.1f}. Trayectoria de la clase 3:"),
        _t.round(3),
        mo.md(f"""
    **Lectura.** Las 8 entradas pasan p < 0,05 y AIC. Con **BIC** (umbral Δℓ > ln(3.322)/2 = 4,05) el
    stepwise se habría detenido en **{_n_bic}** variables: `uso_tc_prom_3m` entra con Δℓ ≈ 3,1. Lo mismo
    con α = 0,01. Noten además que Wald y LR no coinciden (paso 6: Wald 7,7e-3 vs LR ≈ 3e-3): son
    asintóticamente equivalentes, no idénticos. Las log-verosimilitudes del curso están redondeadas a
    0,1, así que Δℓ tiene ±0,1 de error.
    """),
    ])
    return


@app.cell
def _(
    CANDIDATAS,
    VERDAD,
    W_cruz,
    W_dev,
    evaluar,
    pd,
    stepwise,
    y_dev,
):
    _config = [("Wald p<0,05 (curso)", "wald", 0.05), ("LR p<0,05", "lr", 0.05),
               ("Score p<0,05 (SAS)", "score", 0.05), ("Wald p<0,01", "wald", 0.01),
               ("AIC", "aic", 0.05), ("BIC", "bic", 0.05)]
    _filas = []
    for _nm, _W in [("in", W_dev), ("cruz", W_cruz)]:
        for _et, _cr, _al in _config:
            _s, _, _ = stepwise(_W, y_dev, CANDIDATAS, criterio=_cr, alfa_entrada=_al, alfa_salida=_al)
            _e = evaluar(_s)
            _filas.append({"WoE": _nm, "criterio": _et, "n_vars": len(_s),
                           "señal": sum(VERDAD[v] == "señal" for v in _s),
                           "proxy": sum(VERDAD[v] == "proxy" for v in _s),
                           "ruido": sum(VERDAD[v] == "ruido" for v in _s),
                           "gini_dev": _e["gini_dev"], "gini_ho": _e["gini_ho"],
                           "gini_oot": _e["gini_oot"]})
    tabla_criterios = pd.DataFrame(_filas).set_index(["WoE", "criterio"]).round(4)
    tabla_criterios
    return (tabla_criterios,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. La paradoja de Freedman (1983): «significativas» en ruido puro

    **(a) Réplica del experimento original** (regresión lineal, $n=100$, 50 regresores de ruido
    independientes de $y$): primera pasada con todas, se retienen las de p < 0,25; segunda pasada solo
    con ellas y se cuentan las «significativas» al 5%. Numpy (`lstsq` + t de Student) contra
    `statsmodels.OLS` en una réplica, luego 1.000 réplicas.

    **(b) Versión scorecard**: stepwise logístico (entrada por score, salida por Wald, α = 0,05) sobre
    $K$ variables de **ruido puro** con $y\sim$ Bernoulli(0,12), en tres modos: variable cruda continua,
    WoE ajustado en la misma muestra (5 bins, como el curso) y WoE cruzado.
    """)
    return


@app.cell
def _(np, pd, sm, stats):
    def ols_np(X, y):
        _b, *_ = np.linalg.lstsq(X, y, rcond=None)
        _r = y - X @ _b
        _gl = X.shape[0] - X.shape[1]
        _s2 = _r @ _r / _gl
        _se = np.sqrt(np.diag(_s2 * np.linalg.inv(X.T @ X)))
        _p = 2 * stats.t.sf(np.abs(_b / _se), _gl)
        _r2 = 1 - (_r @ _r) / np.sum((y - y.mean()) ** 2)
        _k = X.shape[1] - 1
        _F = (_r2 / _k) / ((1 - _r2) / _gl) if _k > 0 else np.nan
        return _b, _p, _r2, stats.f.sf(_F, _k, _gl) if _k > 0 else np.nan

    def freedman_una(rng, n=100, p=50, alfa1=0.25):
        _X = rng.normal(size=(n, p))
        _y = rng.normal(size=n)
        _, _p1, _, _ = ols_np(np.column_stack([np.ones(n), _X]), _y)
        _ret = np.where(_p1[1:] < alfa1)[0]
        _X2 = np.column_stack([np.ones(n), _X[:, _ret]])
        _, _p2, _r2, _pF = ols_np(_X2, _y)
        return len(_ret), int((_p2[1:] < 0.05).sum()), _r2, _pF, (_X2, _y)

    _rng = np.random.default_rng(1983)
    _k1, _s1, _r21, _pF1, (_X2, _y2) = freedman_una(_rng)
    _o = sm.OLS(_y2, _X2).fit()
    _b_np, _p_np, _r2_np, _pF_np = ols_np(_X2, _y2)
    chk_ols = max(float(np.max(np.abs(_b_np - _o.params))), float(np.max(np.abs(_p_np - _o.pvalues))),
                  abs(_r2_np - _o.rsquared), abs(_pF_np - _o.f_pvalue))
    _res = np.array([freedman_una(_rng)[:4] for _ in range(1000)], dtype=float)
    tabla_freedman_lineal = pd.DataFrame({
        "retenidas en 1ª pasada (p<0,25)": _res[:, 0], "significativas al 5% (2ª pasada)": _res[:, 1],
        "R²": _res[:, 2], "F global p<0,05": (_res[:, 3] < 0.05).astype(float),
    }).describe().loc[["mean", "25%", "50%", "75%"]].round(3)
    return chk_ols, tabla_freedman_lineal


@app.cell
def _(chk_ols, mo, tabla_freedman_lineal):
    mo.vstack([
        mo.md(f"Numpy vs `statsmodels.OLS` (máx. diferencia en β, p, R², p del F): **{chk_ols:.1e}**"),
        tabla_freedman_lineal,
        mo.md("""
    **Lectura.** Partiendo de ruido puro, la segunda pasada típicamente muestra una regresión con
    decenas de variables, R² de un tercio, varias «significativas» al 5% y un F global que rechaza casi
    siempre. Nada de eso existe: es selección seguida de inferencia como si no hubiera habido selección.
    """),
    ])
    return


@app.cell
def _(mo):
    ui_fr_n = mo.ui.slider(500, 6000, value=2000, step=250, label="n (filas)")
    ui_fr_k = mo.ui.slider(5, 60, value=50, step=5, label="K variables de ruido")
    ui_fr_R = mo.ui.slider(10, 200, value=30, step=10, label="Réplicas")
    mo.hstack([ui_fr_n, ui_fr_k, ui_fr_R], justify="start")
    return ui_fr_R, ui_fr_k, ui_fr_n


@app.cell
def _(W_dev, dev, np):
    def woe_rapido(X_aj, y_aj, X_ap, bins=5):
        """WoE por quintiles ajustado en (X_aj, y_aj) y aplicado a X_ap (misma regla y suavizado
        +0,5 que tabla_woe para variables continuas sin moda dominante ni missing)."""
        _q = np.quantile(X_aj, np.linspace(0, 1, bins + 1)[1:-1], axis=0)
        _out = np.empty_like(X_ap, dtype=float)
        _M = y_aj.sum()
        _G = len(y_aj) - _M
        for _j in range(X_aj.shape[1]):
            _ia = np.searchsorted(_q[:, _j], X_aj[:, _j], side="left")
            _m = np.bincount(_ia, weights=y_aj, minlength=bins)
            _t = np.bincount(_ia, minlength=bins)
            _pm = (_m + 0.5) / (_M + 0.5 * bins)
            _pb = (_t - _m + 0.5) / (_G + 0.5 * bins)
            _w = np.log(_pb / _pm)
            _out[:, _j] = _w[np.searchsorted(_q[:, _j], X_ap[:, _j], side="left")]
        return _out

    # coincidencia con la herramienta del curso en una variable continua
    _x = dev["uso_linea_prom_12m"].to_numpy(float)
    _y = dev["malo"].to_numpy(float)
    _w_rap = woe_rapido(_x[:, None], _y, _x[:, None])[:, 0]
    chk_woe_rapido = float(np.max(np.abs(_w_rap - W_dev["uso_linea_prom_12m"].to_numpy())))
    return chk_woe_rapido, woe_rapido


@app.cell
def _(np, pd, stepwise, ui_fr_R, ui_fr_k, ui_fr_n, woe_rapido):
    def woe_cruzado_np(X, y, K=5, semilla=0):
        _pl = np.random.default_rng(semilla).integers(0, K, len(y))
        _W = np.empty_like(X, dtype=float)
        for _k in range(K):
            _m = _pl == _k
            _W[_m] = woe_rapido(X[~_m], y[~_m], X[_m])
        return _W

    _rng = np.random.default_rng(83)
    _cols = [f"r{j:02d}" for j in range(ui_fr_k.value)]
    _cuenta = {"cruda continua": [], "WoE en la misma muestra": [], "WoE cruzado": []}
    for _r in range(ui_fr_R.value):
        _X = _rng.normal(size=(ui_fr_n.value, ui_fr_k.value))
        _y = (_rng.random(ui_fr_n.value) < 0.12).astype(float)
        _mats = {"cruda continua": (_X, "ninguna"),
                 "WoE en la misma muestra": (woe_rapido(_X, _y, _X), "todas"),
                 "WoE cruzado": (woe_cruzado_np(_X, _y, semilla=_r), "todas")}
        for _nom, (_Z, _pol) in _mats.items():
            _s, _, _ = stepwise(pd.DataFrame(_Z, columns=_cols), _y, _cols, criterio="score",
                                politica_signo=_pol, tope=60)
            _cuenta[_nom].append(len(_s))
    conteo_freedman = pd.DataFrame(_cuenta)
    tabla_freedman_logit = pd.DataFrame({
        "media de variables de ruido elegidas": conteo_freedman.mean(),
        "P(al menos una)": (conteo_freedman > 0).mean(),
        "máximo": conteo_freedman.max()}).round(3)
    return conteo_freedman, tabla_freedman_logit, woe_cruzado_np


@app.cell
def _(conteo_freedman, mo, plt, tabla_freedman_logit, ui_fr_k, ui_fr_n):
    _fig, _ax = plt.subplots(figsize=(7.5, 3.4))
    _mx = int(conteo_freedman.to_numpy().max())
    _bins = range(0, _mx + 2)
    for _c in conteo_freedman.columns:
        _ax.hist(conteo_freedman[_c], bins=_bins, alpha=0.55, label=_c, align="left")
    _ax.set_xlabel("nº de variables de ruido que el stepwise deja en el modelo")
    _ax.set_ylabel("réplicas")
    _ax.set_title(f"Stepwise sobre ruido puro (n={ui_fr_n.value}, K={ui_fr_k.value}, α=0,05)")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([tabla_freedman_logit, _fig, mo.md("""
    **Lectura.** Con la variable cruda, el stepwise se comporta aproximadamente como un test múltiple:
    del orden de 1 − 0,95^K de réplicas con al menos una variable espuria. Con **WoE ajustado en la
    misma muestra** el problema se multiplica (cada variable tiene ≈ 43% de pasar la primera ronda). El
    WoE cruzado devuelve el comportamiento al de la variable cruda.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Inferencia post-selección: β inflados y p-valores que mienten

    Juguete con verdad exacta: $n=2.000$, 20 variables de ruido y 3 señales $N(0,1)$ independientes con
    $\beta = b\cdot(1;\,0{,}6;\,0{,}3)$ e intercepto $-2{,}2$ (≈ 11% de malos). En cada réplica se corre el
    stepwise (score/Wald, 0,05). Medimos, **condicional a que la señal fue elegida**: el sesgo
    $\mathbb E[\hat\beta\mid\text{elegida}]/\beta$ (*winner's curse*) y la cobertura del IC ingenuo al 95%.
    Remedio clásico: **dividir la muestra** (seleccionar en una mitad, estimar e inferir en la otra).
    """)
    return


@app.cell
def _(mo):
    ui_ps_b = mo.ui.slider(0.05, 0.50, value=0.30, step=0.01, label="b (fuerza de la señal)")
    ui_ps_R = mo.ui.slider(50, 400, value=150, step=50, label="Réplicas")
    mo.hstack([ui_ps_b, ui_ps_R], justify="start")
    return ui_ps_R, ui_ps_b


@app.cell
def _(irls, np, pd, stepwise, ui_ps_R, ui_ps_b):
    _n, _p_r = 2000, 20
    _beta = ui_ps_b.value * np.array([1.0, 0.6, 0.3])
    _cols = ["s1", "s2", "s3"] + [f"r{j:02d}" for j in range(_p_r)]
    _rng = np.random.default_rng(2004)
    _reg = {m: {c: [] for c in ["s1", "s2", "s3"]} for m in ["ingenuo", "división"]}
    _n_ruido = []
    for _r in range(ui_ps_R.value):
        _X = _rng.normal(size=(_n, 3 + _p_r))
        _y = (_rng.random(_n) < 1 / (1 + np.exp(-(-2.2 + _X[:, :3] @ _beta)))).astype(float)
        _D = pd.DataFrame(_X, columns=_cols)
        _s, _, _aj = stepwise(_D, _y, _cols, criterio="score", politica_signo="ninguna", tope=40)
        _n_ruido.append(sum(c.startswith("r") for c in _s))
        for _j, _c in enumerate(["s1", "s2", "s3"]):
            if _c in _s:
                _k = _s.index(_c) + 1
                _reg["ingenuo"][_c].append((_aj["beta"][_k], _aj["se"][_k]))
        # división de muestra: selecciona en A, estima en B
        _A = np.arange(_n) < _n // 2
        _sA, _, _ = stepwise(_D[_A].reset_index(drop=True), _y[_A], _cols, criterio="score",
                             politica_signo="ninguna", tope=40)
        if _sA:
            _aB = irls(np.column_stack([np.ones((~_A).sum()), _D.loc[~_A, _sA].to_numpy()]), _y[~_A])
            for _c in ["s1", "s2", "s3"]:
                if _c in _sA:
                    _k = _sA.index(_c) + 1
                    _reg["división"][_c].append((_aB["beta"][_k], _aB["se"][_k]))
    _filas = []
    for _m in ["ingenuo", "división"]:
        for _j, _c in enumerate(["s1", "s2", "s3"]):
            _v = np.array(_reg[_m][_c]).reshape(-1, 2)
            _cub = np.mean(np.abs(_v[:, 0] - _beta[_j]) <= 1.96 * _v[:, 1]) if len(_v) else np.nan
            _filas.append({"método": _m, "señal": _c, "β verdadero": _beta[_j],
                           "P(elegida)": len(_v) / ui_ps_R.value,
                           "E[β̂ | elegida] / β": _v[:, 0].mean() / _beta[_j] if len(_v) else np.nan,
                           "cobertura IC95 | elegida": _cub})
    tabla_postsel = pd.DataFrame(_filas).set_index(["método", "señal"]).round(3)
    ruido_postsel = float(np.mean(_n_ruido))
    return ruido_postsel, tabla_postsel


@app.cell
def _(mo, ruido_postsel, tabla_postsel):
    _t = tabla_postsel.loc["ingenuo"]
    mo.vstack([tabla_postsel, mo.md(f"""
    **Lectura.** La señal débil (s3) se elige en {_t.loc['s3', 'P(elegida)']:.0%} de las réplicas y,
    **cuando se elige**, su β̂ está inflado ×{_t.loc['s3', 'E[β̂ | elegida] / β']:.2f} y el IC «95%» cubre
    el verdadero solo {_t.loc['s3', 'cobertura IC95 | elegida']:.0%} de las veces. La señal fuerte (s1)
    entra en {_t.loc['s1', 'P(elegida)']:.0%} de las réplicas y casi no sufre (×{_t.loc['s1', 'E[β̂ | elegida] / β']:.2f},
    cobertura {_t.loc['s1', 'cobertura IC95 | elegida']:.0%}): cuando la selección no condiciona, no hay
    sesgo de selección. El sesgo es función de la **potencia**, no de la variable. Además entran en promedio
    {ruido_postsel:.2f} variables de ruido por réplica, todas con p < 0,05 **por construcción**. Con
    división de muestra la cobertura vuelve a ≈ 95% a cambio de estimar con la mitad de los datos.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Bootstrap del stepwise: frecuencia de inclusión

    Austin & Tu (2004), siguiendo a Sauerbrei & Schumacher (1992): se repite **todo** el procedimiento de
    selección en $B$ remuestras bootstrap de DEV y se cuenta en qué fracción entra cada variable. Una
    variable que entra en el 55% de las remuestras no es «significativa»: es una moneda. Por costo, por
    defecto se usa entrada por score (SAS); el selector permite el criterio Wald del curso (más lento).
    El WoE queda fijo (no se re-binea en cada réplica): con WoE in-sample la réplica **hereda** el
    sobreajuste del WoE, y eso es parte de lo que se quiere ver.
    """)
    return


@app.cell
def _(mo):
    ui_B = mo.ui.slider(10, 300, value=40, step=10, label="B (réplicas bootstrap)")
    ui_boot_crit = mo.ui.dropdown(options={"score/Wald (SAS, rápido)": "score", "Wald (curso, lento)": "wald"},
                                  value="score/Wald (SAS, rápido)", label="Criterio")
    mo.hstack([ui_B, ui_boot_crit], justify="start")
    return ui_B, ui_boot_crit


@app.cell
def _(
    CANDIDATAS,
    Counter,
    VERDAD,
    W_cruz,
    W_dev,
    np,
    pd,
    stepwise,
    time,
    ui_B,
    ui_boot_crit,
    y_dev,
):
    _t0 = time.time()
    _rng = np.random.default_rng(2004)
    _frec = {}
    modelos_distintos = {}
    modelo_modal = {}
    for _nm, _W in [("in", W_dev), ("cruz", W_cruz)]:
        _cnt = Counter()
        _mods = Counter()
        _M = _W.to_numpy()
        for _b in range(ui_B.value):
            _i = _rng.integers(0, len(y_dev), len(y_dev))
            _s, _, _ = stepwise(pd.DataFrame(_M[_i], columns=CANDIDATAS), y_dev[_i], CANDIDATAS,
                                criterio=ui_boot_crit.value)
            _cnt.update(_s)
            _mods[frozenset(_s)] += 1
        _frec[_nm] = pd.Series({v: _cnt[v] / ui_B.value for v in CANDIDATAS})
        modelos_distintos[_nm] = len(_mods)
        _top = _mods.most_common(1)[0]
        modelo_modal[_nm] = (sorted(_top[0]), _top[1] / ui_B.value)
    frec_inclusion = pd.DataFrame({"WoE DEV (curso)": _frec["in"], "WoE cruzado": _frec["cruz"],
                                   "verdad": pd.Series(VERDAD)}).loc[CANDIDATAS]
    t_boot = time.time() - _t0
    return frec_inclusion, modelo_modal, modelos_distintos, t_boot


@app.cell
def _(frec_inclusion, mo, modelo_modal, modelos_distintos, np, plt, t_boot, ui_B):
    _f = frec_inclusion.sort_values("WoE DEV (curso)")
    _r = _f[_f["verdad"] == "ruido"]
    _mx_in = _r["WoE DEV (curso)"].max() if len(_r) else 0.0
    _mx_cz = _r["WoE cruzado"].max() if len(_r) else 0.0
    _fig, _ax = plt.subplots(figsize=(8, 0.28 * len(_f) + 1.2))
    _y = np.arange(len(_f))
    _ax.barh(_y - 0.2, _f["WoE DEV (curso)"], height=0.4, label="WoE ajustado en DEV (curso)")
    _ax.barh(_y + 0.2, _f["WoE cruzado"], height=0.4, label="WoE cruzado K=5")
    _ax.set_yticks(_y, [f"{v} [{_f.loc[v, 'verdad']}]" for v in _f.index], fontsize=8)
    _ax.axvline(0.6, ls="--", lw=1, color="gray")
    _ax.set_xlabel("frecuencia de inclusión en el bootstrap")
    _ax.set_title(f"Bootstrap del stepwise (B={ui_B.value})")
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"""
    **Lectura** ({t_boot:.1f} s). Modelos **distintos** encontrados en {ui_B.value} réplicas:
    **{modelos_distintos['in']}** con WoE de DEV y **{modelos_distintos['cruz']}** con WoE cruzado. El
    modelo más frecuente aparece en {modelo_modal['in'][1]:.0%} y {modelo_modal['cruz'][1]:.0%} de las
    réplicas, respectivamente. El stepwise no elige «el» modelo: elige uno de una nube. Las señales
    fuertes están cerca de 100% en ambos modos. La variable de ruido más frecuente entra en
    **{_mx_in:.0%}** de las réplicas con WoE de DEV y **{_mx_cz:.0%}** con WoE cruzado: el bootstrap con
    WoE fijo hereda la fuga, así que una frecuencia alta **no** certifica una variable. La línea
    punteada (60%) es un umbral de trabajo frecuente, no una regla con base teórica.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. LASSO y elastic net con restricción de signo

    Sobre WoE estandarizado ($\tilde x_j = (x_j-\bar x_j)/s_j$) se resuelve

    $$\min_{\beta_0,\beta}\; \frac1n\sum_i \Big[\log(1+e^{\eta_i}) - y_i\eta_i\Big]
      + \lambda\Big(\alpha\lVert\beta\rVert_1 + \tfrac{1-\alpha}{2}\lVert\beta\rVert_2^2\Big)
      \quad\text{s.a. } \beta_j\le 0 .$$

    Con la restricción, $|\beta_j|=-\beta_j$: la penalización L1 se vuelve **lineal** y el problema es
    suave con cotas, que `scipy.optimize` (L-BFGS-B) resuelve directo. En numpy usamos gradiente
    proximal acelerado (FISTA) con el operador proximal cerrado
    $\operatorname{prox}(z)=\min\{0,\,z+t\lambda\alpha\}/(1+t\lambda(1-\alpha))$; sin signo, el
    *soft-thresholding* habitual, que comparamos con `scikit-learn` (`saga`, $C = 1/(n\lambda)$). El
    λ se elige por **BIC del modelo re-ajustado sin penalización** sobre el conjunto activo (*relaxed
    lasso*).
    """)
    return


@app.cell
def _(mo):
    ui_l1 = mo.ui.slider(0.1, 1.0, value=1.0, step=0.05, label="α (mezcla L1; 1 = LASSO)")
    ui_lsigno = mo.ui.checkbox(value=True, label="Restricción de signo (β ≤ 0)")
    ui_lmodo = mo.ui.dropdown(options={"WoE ajustado en DEV (curso)": "in", "WoE cruzado K=5": "cruz"},
                              value="WoE cruzado K=5", label="WoE para seleccionar")
    mo.hstack([ui_l1, ui_lsigno, ui_lmodo], justify="start")
    return ui_l1, ui_lmodo, ui_lsigno


@app.cell
def _(irls, np):
    def enet_logit(X, y, lam, l1=1.0, signo=True, b0=None, b=None, max_iter=5000, tol=1e-9):
        """Elastic net logístico por FISTA (intercepto sin penalizar). X ya estandarizada."""
        _n, _p = X.shape
        _L = (np.linalg.eigvalsh(X.T @ X / _n).max() + 1.0) / 4.0 + lam * (1 - l1)
        _t = 1.0 / _L
        if b is None:
            b = np.zeros(_p)
        if b0 is None:
            b0 = float(np.log(y.mean() / (1 - y.mean())))
        _zb, _z0, _tk = b.copy(), b0, 1.0
        for _it in range(max_iter):
            _r = 1.0 / (1.0 + np.exp(-(_z0 + X @ _zb))) - y
            _b0n = _z0 - _t * _r.mean()
            _zz = _zb - _t * (X.T @ _r / _n)
            _a = _t * lam * l1
            _c = 1.0 + _t * lam * (1 - l1)
            _bn = (np.minimum(0.0, _zz + _a) if signo else np.sign(_zz) * np.maximum(np.abs(_zz) - _a, 0)) / _c
            _tn = (1 + np.sqrt(1 + 4 * _tk * _tk)) / 2
            _dif = max(abs(_b0n - b0), float(np.max(np.abs(_bn - b))) if _p else 0.0)
            _zb = _bn + ((_tk - 1) / _tn) * (_bn - b)
            _z0 = _b0n + ((_tk - 1) / _tn) * (_b0n - b0)
            b0, b, _tk = _b0n, _bn, _tn
            if _dif < tol:
                break
        return b0, b

    def estandarizar(W):
        _Z = W.to_numpy(float)
        _mu, _sd = _Z.mean(0), _Z.std(0)
        _sd = np.where(_sd > 0, _sd, 1.0)
        return (_Z - _mu) / _sd, _mu, _sd

    def camino_enet(W, y, l1=1.0, signo=True, n_lam=40, razon=1e-3, tol=1e-8):
        """Camino de regularización con warm starts; BIC del re-ajuste sin penalizar por λ."""
        _Xs, _mu, _sd = estandarizar(W)
        _n = len(y)
        _lmax = np.max(np.abs(_Xs.T @ (y - y.mean()))) / (_n * l1)
        _lams = _lmax * np.logspace(0, np.log10(razon), n_lam)
        _b0, _b = None, None
        _B, _bic, _act = [], [], []
        _cache = {}
        for _l in _lams:
            _b0, _b = enet_logit(_Xs, y, _l, l1, signo, _b0, _b, tol=tol)
            _B.append(_b / _sd)                          # β en unidades de WoE
            _A = tuple(np.where(_b != 0)[0])
            if _A not in _cache:
                _X = np.column_stack([np.ones(_n)] + [W.iloc[:, j].to_numpy(float) for j in _A])
                _cache[_A] = -2 * irls(_X, y)["llf"] + (len(_A) + 1) * np.log(_n)
            _bic.append(_cache[_A])
            _act.append(_A)
        _i = int(np.argmin(_bic))
        return {"lams": _lams, "B": np.array(_B), "bic": np.array(_bic), "activos": _act,
                "i_bic": _i, "sel": [W.columns[j] for j in _act[_i]]}
    return camino_enet, enet_logit, estandarizar


@app.cell
def _(
    CANDIDATAS,
    LogisticRegression,
    W_cruz,
    W_dev,
    camino_enet,
    enet_logit,
    estandarizar,
    np,
    optimize,
    time,
    ui_l1,
    ui_lmodo,
    ui_lsigno,
    warnings,
    y_dev,
):
    _W = W_dev if ui_lmodo.value == "in" else W_cruz
    _t0 = time.time()
    camino = camino_enet(_W, y_dev, l1=ui_l1.value, signo=ui_lsigno.value)
    t_camino = time.time() - _t0

    # --- comparación con librerías en un λ intermedio ---
    _Xs, _, _ = estandarizar(_W)
    _n = len(y_dev)
    _lam = float(camino["lams"][12])
    _l1 = ui_l1.value
    _b0f, _bf = enet_logit(_Xs, y_dev, _lam, _l1, False, max_iter=50_000, tol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _sk = LogisticRegression(l1_ratio=_l1, C=1.0 / (_n * _lam), solver="saga", tol=1e-12,
                                 max_iter=200_000).fit(_Xs, y_dev)
    _b0s, _bs = enet_logit(_Xs, y_dev, _lam, _l1, True, max_iter=50_000, tol=1e-12)

    def _obj(th):
        _eta = th[0] + _Xs @ th[1:]
        _r = 1 / (1 + np.exp(-_eta)) - y_dev
        _f = (np.mean(np.logaddexp(0, _eta) - y_dev * _eta) - _lam * _l1 * np.sum(th[1:])
              + _lam * (1 - _l1) / 2 * np.sum(th[1:] ** 2))
        return _f, np.r_[_r.mean(), _Xs.T @ _r / _n - _lam * _l1 + _lam * (1 - _l1) * th[1:]]

    _res = optimize.minimize(_obj, np.zeros(len(CANDIDATAS) + 1), jac=True, method="L-BFGS-B",
                             bounds=[(None, None)] + [(None, 0.0)] * len(CANDIDATAS),
                             options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20_000})
    chk_lasso = {"sin signo: numpy FISTA vs sklearn saga": float(np.max(np.abs(np.r_[_b0f - _sk.intercept_[0], _bf - _sk.coef_[0]]))),
                 "con signo: numpy FISTA vs scipy L-BFGS-B": float(np.max(np.abs(np.r_[_b0s - _res.x[0], _bs - _res.x[1:]]))),
                 "λ de la comparación": _lam}
    return camino, chk_lasso, t_camino


@app.cell
def _(
    CANDIDATAS,
    VERDAD,
    camino,
    chk_lasso,
    evaluar,
    mo,
    np,
    pd,
    plt,
    t_camino,
    ui_lsigno,
):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _col = {"señal": "tab:blue", "proxy": "tab:orange", "ruido": "tab:gray"}
    _x = np.log10(camino["lams"])
    for _j, _v in enumerate(CANDIDATAS):
        _ax1.plot(_x, camino["B"][:, _j], color=_col[VERDAD[_v]], lw=1.2 if VERDAD[_v] != "ruido" else 0.7)
    _ax1.axvline(_x[camino["i_bic"]], ls="--", color="k", lw=1)
    _ax1.invert_xaxis()
    _ax1.set_xlabel("log10 λ (← más penalización)")
    _ax1.set_ylabel("β (unidades de WoE)")
    _ax1.set_title("Camino de regularización")
    for _k, _c in _col.items():
        _ax1.plot([], [], color=_c, label=_k)
    _ax1.legend(fontsize=8)
    _nv = [len(a) for a in camino["activos"]]
    _ax2.plot(_nv, camino["bic"], "o-", ms=3)
    _ax2.axvline(_nv[camino["i_bic"]], ls="--", color="k", lw=1)
    _ax2.set_xlabel("nº de variables activas")
    _ax2.set_ylabel("BIC del re-ajuste")
    _ax2.set_title("Elección de λ por BIC (relaxed)")
    _fig.tight_layout()
    modelo_lasso = evaluar(camino["sel"])
    _n_pos = int((camino["B"] > 1e-12).any(axis=0).sum())
    mo.vstack([_fig, pd.Series(chk_lasso).map(lambda v: f"{v:.2e}").to_frame("valor"), mo.md(f"""
    **Lectura** (camino en {t_camino:.1f} s). λ por BIC deja **{len(camino['sel'])}** variables:
    {', '.join(camino['sel'])}. Re-ajustado sin penalización: Gini HO {modelo_lasso['gini_ho']:.3f},
    OOT {modelo_lasso['gini_oot']:.3f}. Variables que alguna vez tuvieron β > 0 en el camino:
    **{_n_pos}** {'(cero por construcción: la restricción de signo)' if ui_lsigno.value else '(sin restricción, la supresión aparece como β positivos)'}.
    """)])
    return (modelo_lasso,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Stability selection (Meinshausen & Bühlmann, 2010)

    Se toman $2R$ submuestras de tamaño $n/2$ (pares complementarios, Shah & Samworth 2013), en cada una
    se recorre el camino LASSO-con-signo y se registran las **primeras $q$ variables** en entrar. La
    probabilidad de selección $\hat\pi_j$ es la fracción de submuestras en que $j$ está entre ellas. Se
    retienen las de $\hat\pi_j\ge\pi_{thr}$. Bajo intercambiabilidad del ruido, el número esperado de
    falsos positivos cumple $\mathbb E[V]\le \dfrac{q^2}{(2\pi_{thr}-1)\,p}$.
    """)
    return


@app.cell
def _(mo):
    ui_q = mo.ui.slider(3, 15, value=8, step=1, label="q (primeras variables del camino)")
    ui_pi = mo.ui.slider(0.55, 0.95, value=0.75, step=0.05, label="π umbral")
    ui_pares = mo.ui.slider(5, 50, value=15, step=5, label="R pares de submuestras")
    mo.hstack([ui_q, ui_pi, ui_pares], justify="start")
    return ui_pares, ui_pi, ui_q


@app.cell
def _(
    CANDIDATAS,
    VERDAD,
    W_cruz,
    W_dev,
    enet_logit,
    estandarizar,
    np,
    pd,
    time,
    ui_pares,
    ui_pi,
    ui_q,
    y_dev,
):
    def primeras_q(X, y, q, n_lam=30):
        """Índices de las primeras q variables en entrar al camino LASSO con signo."""
        _lmax = np.max(np.abs(X.T @ (y - y.mean()))) / len(y)
        _orden = []
        _b0, _b = None, None
        for _l in _lmax * np.logspace(0, -2.5, n_lam):
            _b0, _b = enet_logit(X, y, _l, 1.0, True, _b0, _b, max_iter=3000, tol=1e-6)
            _nuevas = [j for j in np.argsort(_b) if _b[j] != 0 and j not in _orden]
            _orden += _nuevas[: max(q - len(_orden), 0)]
            if len(_orden) >= q:
                break
        return _orden

    _t0 = time.time()
    _rng = np.random.default_rng(2010)
    _pi = {}
    for _nm, _W in [("in", W_dev), ("cruz", W_cruz)]:
        _Xs, _, _ = estandarizar(_W)
        _cnt = np.zeros(len(CANDIDATAS))
        for _r in range(ui_pares.value):
            _perm = _rng.permutation(len(y_dev))
            for _mitad in (_perm[: len(y_dev) // 2], _perm[len(y_dev) // 2:]):
                _cnt[primeras_q(_Xs[_mitad], y_dev[_mitad], ui_q.value)] += 1
        _pi[_nm] = _cnt / (2 * ui_pares.value)
    prob_estab = pd.DataFrame({"π̂ WoE DEV (curso)": _pi["in"], "π̂ WoE cruzado": _pi["cruz"],
                               "verdad": [VERDAD[v] for v in CANDIDATAS]}, index=CANDIDATAS)
    sel_estab = {m: [v for v, p in zip(CANDIDATAS, _pi[m]) if p >= ui_pi.value] for m in _pi}
    cota_EV = ui_q.value ** 2 / ((2 * ui_pi.value - 1) * len(CANDIDATAS))
    t_estab = time.time() - _t0
    return cota_EV, prob_estab, sel_estab, t_estab


@app.cell
def _(VERDAD, cota_EV, mo, prob_estab, sel_estab, t_estab, ui_pi, ui_q):
    _txt = {m: f"{len(s)} variables ({sum(VERDAD[v] == 'ruido' for v in s)} de ruido)" for m, s in sel_estab.items()}
    _es_r = prob_estab["verdad"] == "ruido"
    _mx = {m: prob_estab.loc[_es_r, c].max() if _es_r.any() else 0.0
           for m, c in [("in", "π̂ WoE DEV (curso)"), ("cruz", "π̂ WoE cruzado")]}
    mo.vstack([prob_estab.sort_values("π̂ WoE cruzado", ascending=False).round(3), mo.md(f"""
    **Lectura** ({t_estab:.1f} s). Con q = {ui_q.value} y π = {ui_pi.value:.2f}, la cota de
    Meinshausen-Bühlmann es **E[V] ≤ {cota_EV:.2f}** falsos positivos — con un pool chico la cota es
    poco informativa, y lo honesto es decirlo. Seleccionadas: WoE de DEV → {_txt['in']}; WoE cruzado →
    {_txt['cruz']}. La π̂ máxima de una variable de ruido es {_mx['in']:.2f} (WoE de DEV) y
    {_mx['cruz']:.2f} (cruzado). Lo que protege aquí es q: con solo las primeras q variables del camino,
    las señales fuertes ocupan los cupos. Stability selection **no** corrige la fuga del WoE ajustado en
    DEV (re-muestrea filas, pero el WoE de cada fila ya «vio» su propio target): suban q y el ruido
    aparece. Tampoco rescata señales débiles o redundantes (`uso_tc_prom_12m`, `canal`): selecciona lo
    **estable**, que no es lo mismo que lo **verdadero**.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Comparación de enfoques en HO y OOT

    Todas las selecciones se **re-ajustan** con el WoE de DEV (como en producción) y se evalúan en HO y
    OOT. Se agregan dos referencias: el **modelo completo** (todas las candidatas) y el **oráculo** (solo
    las variables de señal directa del generador, algo que nunca se sabe en la vida real). El enfoque
    «forward por Gini HO» selecciona mirando HO: su Gini HO deja de ser una estimación fuera de muestra.
    """)
    return


@app.cell
def _(
    CANDIDATAS,
    CONCEPTO,
    VERDAD,
    W_cruz,
    W_dev,
    W_ho,
    auc_np,
    camino_enet,
    disenar,
    evaluar,
    irls,
    pd,
    sel_estab,
    stepwise,
    y_dev,
    y_ho,
):
    def forward_gini_ho(W, tol=0.002, tope=14):
        """Forward que agrega la variable que más sube el Gini en HO (ajustando en DEV)."""
        _sel, _g_act = [], 0.0
        while len(_sel) < tope:
            _mejor = None
            for _v in [c for c in CANDIDATAS if c not in _sel]:
                _a = irls(disenar(W, _sel + [_v]), y_dev)
                _p = disenar(W_ho, _sel + [_v]) @ _a["beta"]
                _g = 2 * auc_np(y_ho, _p) - 1
                if _mejor is None or _g > _mejor[1]:
                    _mejor = (_v, _g)
            if _mejor is None or _mejor[1] - _g_act < tol:
                break
            _sel.append(_mejor[0])
            _g_act = _mejor[1]
        return _sel

    _enf = {}
    for _nm, _W in [("in", W_dev), ("cruz", W_cruz)]:
        _enf[(_nm, "stepwise Wald 0,05 (curso)")] = stepwise(_W, y_dev, CANDIDATAS)[0]
        _enf[(_nm, "stepwise BIC")] = stepwise(_W, y_dev, CANDIDATAS, criterio="bic")[0]
        _enf[(_nm, "stepwise AIC")] = stepwise(_W, y_dev, CANDIDATAS, criterio="aic")[0]
        _enf[(_nm, "LASSO con signo, λ BIC")] = camino_enet(_W, y_dev, 1.0, True)["sel"]
        _enf[(_nm, "stability selection")] = sel_estab[_nm]
        _enf[(_nm, "forward por Gini HO")] = forward_gini_ho(_W)
    _enf[("—", "modelo completo")] = list(CANDIDATAS)
    _enf[("—", "oráculo (solo señal)")] = [v for v in CANDIDATAS if VERDAD[v] == "señal"]
    _filas = []
    for (_nm, _et), _s in _enf.items():
        _e = evaluar(_s)
        _pos = int((_e["beta"][1:] > 0).sum())
        _filas.append({"WoE selección": _nm, "enfoque": _et, "n_vars": len(_s),
                       "señal": sum(VERDAD[v] == "señal" for v in _s),
                       "proxy": sum(VERDAD[v] == "proxy" for v in _s),
                       "ruido": sum(VERDAD[v] == "ruido" for v in _s),
                       "β>0": _pos,
                       "conceptos": len({CONCEPTO[v] for v in _s if CONCEPTO[v] != "—"}),
                       "gini_dev": _e["gini_dev"], "gini_ho": _e["gini_ho"], "gini_oot": _e["gini_oot"]})
    tabla_enfoques = pd.DataFrame(_filas).set_index(["WoE selección", "enfoque"]).round(4)
    selecciones = _enf
    return selecciones, tabla_enfoques


@app.cell
def _(mo, plt, tabla_enfoques):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _mk = {"in": "o", "cruz": "s", "—": "*"}
    for (_nm, _et), _r in tabla_enfoques.iterrows():
        _ax.scatter(_r["n_vars"], _r["gini_oot"], marker=_mk[_nm], s=60 if _nm != "—" else 140)
        _ax.annotate(_et.split(",")[0][:22], (_r["n_vars"], _r["gini_oot"]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    for _nm, _m in _mk.items():
        _ax.scatter([], [], marker=_m, color="k", label={"in": "WoE DEV", "cruz": "WoE cruzado", "—": "referencia"}[_nm])
    _ax.set_xlabel("nº de variables")
    _ax.set_ylabel("Gini OOT")
    _ax.set_title("Parsimonia vs discriminación fuera de tiempo")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _g = tabla_enfoques["gini_oot"]
    mo.vstack([tabla_enfoques, _fig, mo.md(f"""
    **Lectura.** El rango de Gini OOT entre enfoques razonables es de apenas
    **{_g.drop(index=('—', 'modelo completo')).max() - _g.drop(index=('—', 'modelo completo')).min():.3f}**:
    la selección casi no mueve el Gini; mueve **cuántas** variables, **cuáles** (ruido, proxies) y
    **cuántos signos positivos** hay que defender. Eso es lo que decide el comité. El «forward por Gini
    HO» muestra el Gini HO más alto ({tabla_enfoques.loc[('in', 'forward por Gini HO'), 'gini_ho']:.3f})
    porque **eligió mirando HO**; en OOT ({tabla_enfoques.loc[('in', 'forward por Gini HO'), 'gini_oot']:.3f})
    vuelve al montón. Y el oráculo tiene un β > 0: `uso_tc_prom_12m` junto a `uso_tc_prom_3m` (nivel vs
    tendencia; ver Serie 2 · M09) — la política de signos habría sacado una variable **verdadera**.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Forzar una variable: costo en log-verosimilitud y en Gini

    El comité pide una variable que el stepwise dejó fuera («¿y la renta?», RESERVA de la clase 3).
    Se mide: (i) el **test LR** anidado $2(\ell_{+v}-\ell)\sim\chi^2_1$ y el Wald; (ii) ΔGini en DEV, HO y
    OOT con **IC bootstrap pareado** en HO/OOT; (iii) qué pasa si se fuerza **desde el inicio** del
    stepwise (el camino puede cambiar: la forzada desplaza a otras).
    """)
    return


@app.cell
def _(CANDIDATAS, mo, sel_main):
    _fuera = [v for v in CANDIDATAS if v not in sel_main]
    _def = "renta_mm" if "renta_mm" in _fuera else ("edad" if "edad" in _fuera else (_fuera[0] if _fuera else None))
    ui_forzar = mo.ui.dropdown(options=_fuera if _fuera else ["(ninguna)"], value=_def or "(ninguna)",
                               label="Variable a forzar")
    ui_forzar
    return (ui_forzar,)


@app.cell
def _(
    CANDIDATAS,
    W_sel,
    auc_np,
    cfg_main,
    disenar,
    evaluar,
    np,
    pd,
    sel_main,
    sm,
    stats,
    stepwise,
    ui_forzar,
    W_dev,
    y_dev,
    y_ho,
    y_oot,
):
    v_forzada = ui_forzar.value
    _ok = v_forzada in CANDIDATAS
    if _ok:
        _base_f = evaluar(sel_main)
        _con_f = evaluar(sel_main + [v_forzada])
        _lr_np = 2 * (_con_f["llf"] - _base_f["llf"])
        _p_lr_np = float(stats.chi2.sf(_lr_np, 1))
        _r1 = sm.Logit(y_dev, disenar(W_dev, sel_main + [v_forzada])).fit(disp=0, method="newton", tol=1e-12)
        _r0 = sm.Logit(y_dev, disenar(W_dev, sel_main)).fit(disp=0, method="newton", tol=1e-12)
        _lr_sm = 2 * (_r1.llf - _r0.llf)
        _R = np.zeros((1, len(_r1.params)))
        _R[0, -1] = 1
        _wald_sm = float(np.squeeze(_r1.wald_test(_R, scalar=True).statistic))
        _wald_np = float((_con_f["beta"][-1] / _con_f["se"][-1]) ** 2)
        _rng = np.random.default_rng(11)
        _ic = {}
        for _nm, _y in [("ho", y_ho), ("oot", y_oot)]:
            _pb, _pf = _base_f[f"p_{_nm}"], _con_f[f"p_{_nm}"]
            _d = []
            for _b in range(300):
                _i = _rng.integers(0, len(_y), len(_y))
                _d.append(2 * (auc_np(_y[_i], _pf[_i]) - auc_np(_y[_i], _pb[_i])))
            _ic[_nm] = np.percentile(_d, [2.5, 97.5])
        sel_forzada_inicio, bit_forzada, _ = stepwise(
            W_sel, y_dev, CANDIDATAS, forzadas=(v_forzada,),
            **{k: v for k, v in cfg_main.items() if k != "modo_woe"})
        tabla_forzar = pd.DataFrame({
            "base (stepwise)": [len(sel_main), _base_f["llf"], _base_f["gini_dev"], _base_f["gini_ho"], _base_f["gini_oot"]],
            f"+ {v_forzada}": [len(sel_main) + 1, _con_f["llf"], _con_f["gini_dev"], _con_f["gini_ho"], _con_f["gini_oot"]]},
            index=["n_vars", "log-verosimilitud", "Gini DEV", "Gini HO", "Gini OOT"])
        tabla_forzar["Δ"] = tabla_forzar.iloc[:, 1] - tabla_forzar.iloc[:, 0]
        resumen_forzar = {"LR numpy": _lr_np, "LR statsmodels": _lr_sm, "p LR": _p_lr_np,
                          "Wald numpy": _wald_np, "Wald statsmodels": _wald_sm,
                          "β forzada": float(_con_f["beta"][-1]),
                          "IC95 ΔGini HO": _ic["ho"], "IC95 ΔGini OOT": _ic["oot"]}
    else:
        tabla_forzar, resumen_forzar, sel_forzada_inicio, bit_forzada = None, {}, [], pd.DataFrame()
    return bit_forzada, resumen_forzar, sel_forzada_inicio, tabla_forzar, v_forzada


@app.cell
def _(mo, resumen_forzar, sel_forzada_inicio, sel_main, tabla_forzar, v_forzada):
    if tabla_forzar is None:
        _out = mo.md("No hay variables fuera del modelo para forzar.")
    else:
        _r = resumen_forzar
        _desplazadas = [v for v in sel_main if v not in sel_forzada_inicio]
        _nuevas = [v for v in sel_forzada_inicio if v not in sel_main and v != v_forzada]
        _out = mo.vstack([tabla_forzar.round(4), mo.md(f"""
    **Test anidado.** LR = {_r['LR numpy']:.3f} (numpy) = {_r['LR statsmodels']:.3f} (statsmodels),
    p = {_r['p LR']:.3g}; Wald = {_r['Wald numpy']:.3f} = {_r['Wald statsmodels']:.3f}.
    β de `{v_forzada}` = **{_r['β forzada']:+.3f}** {'ATENCIÓN, signo positivo: forzarla viola la política de signos y exige explicación escrita' if _r['β forzada'] > 0 else '(signo esperado)'}.

    **ΔGini con IC95% bootstrap pareado:** HO [{_r['IC95 ΔGini HO'][0]:+.4f}; {_r['IC95 ΔGini HO'][1]:+.4f}] ·
    OOT [{_r['IC95 ΔGini OOT'][0]:+.4f}; {_r['IC95 ΔGini OOT'][1]:+.4f}]. Si el intervalo contiene 0,
    el «costo» (o beneficio) de forzar es indistinguible de cero: se fuerza por gobierno, no por Gini, y
    se deja escrito.

    **Forzada desde el inicio del stepwise:** salen {', '.join(_desplazadas) or 'ninguna'}; entran
    {', '.join(_nuevas) or 'ninguna nueva'}. Forzar al final y forzar al inicio **no** son la misma
    decisión: la bitácora debe decir cuál se tomó.
    """)])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. La bitácora como artefacto versionado

    La bitácora no es un log de consola: es un **artefacto** con la configuración, la huella de los
    datos (hash del target y de la matriz WoE de DEV) y cada decisión automática y humana. Su hash
    SHA-256 se registra en el expediente del modelo (clase 6: cadena de hashes). Cambiar cualquier
    decisión o dato cambia el hash.
    """)
    return


@app.cell
def _(W_sel, bit_forzada, bit_main, cfg_main, hashlib, json, mo, np, sel_main, v_forzada, y_dev):
    def _registros(df):
        return json.loads(df.replace({np.nan: None}).to_json(orient="records", double_precision=10))

    _huella = hashlib.sha256(np.ascontiguousarray(y_dev).tobytes()
                             + np.ascontiguousarray(W_sel.to_numpy()).tobytes()).hexdigest()
    bitacora = {
        "modulo": "M11", "version_bitacora": 1,
        "configuracion": cfg_main,
        "huella_datos_sha256": _huella,
        "decisiones_automaticas": _registros(bit_main),
        "seleccion_estadistica": list(sel_main),
        "decisiones_humanas": [
            {"accion": "forzar", "variable": v_forzada, "momento": "al final (sobre la selección)",
             "autor": "comité de riesgo (ejemplo)",
             "justificacion": "variable histórica de política; costo medido con LR test y ΔGini"}],
        "seleccion_forzada_desde_inicio": _registros(bit_forzada) if len(bit_forzada) else [],
    }
    bitacora_json = json.dumps(bitacora, sort_keys=True, ensure_ascii=False, indent=1)
    hash_bitacora = hashlib.sha256(bitacora_json.encode("utf-8")).hexdigest()
    mo.vstack([mo.md(f"**SHA-256 de la bitácora:** `{hash_bitacora}` · huella de datos `{_huella[:16]}…` · "
                     f"{len(bitacora_json):,} bytes".replace(",", ".")),
               mo.md("```json\n" + bitacora_json[:1500] + "\n…\n```")])
    return bitacora_json, hash_bitacora


@app.cell
def _(mo):
    mo.md(r"""
    ## Checks del módulo
    """)
    return


@app.cell
def _(
    CANDIDATAS,
    RUIDO,
    W_dev,
    auc_np,
    bitacora_json,
    chk_irls,
    chk_lasso,
    chk_ols,
    chk_woe_rapido,
    hash_bitacora,
    hashlib,
    modelo_main,
    np,
    resumen_forzar,
    roc_auc_score,
    sel_main,
    sel_main_sm,
    stats,
    stepwise,
    tabla_gl,
    tabla_modos,
    trayectoria_austral,
    umbral_chi2,
    y_ho,
    y_dev,
):
    # 1) IRLS numpy = statsmodels = sklearn; score numpy = statsmodels
    assert chk_irls["max|Δβ| numpy vs statsmodels"] < 1e-6
    assert chk_irls["max|ΔSE| numpy vs statsmodels"] < 1e-6
    assert chk_irls["|Δ log-verosimilitud|"] < 1e-6
    assert chk_irls["max|Δβ| numpy vs sklearn (C=inf)"] < 1e-3
    assert chk_irls["|Δ score| numpy vs GLM.score_test"] < 1e-6
    # 2) stepwise numpy y statsmodels eligen lo mismo
    assert sel_main == sel_main_sm
    # 3) AUC numpy = sklearn
    assert np.isclose(auc_np(y_ho, modelo_main["p_ho"]), roc_auc_score(y_ho, modelo_main["p_ho"]))
    # 4) OLS numpy = statsmodels; WoE rápido = tabla_woe del curso
    assert chk_ols < 1e-8
    assert chk_woe_rapido < 1e-10
    # 5) LASSO: numpy = sklearn (sin signo) y = scipy L-BFGS-B (con signo)
    assert chk_lasso["sin signo: numpy FISTA vs sklearn saga"] < 1e-4
    assert chk_lasso["con signo: numpy FISTA vs scipy L-BFGS-B"] < 1e-5
    # 6) teoría: equivalencias de criterio y tamaño real del test sobre WoE in-sample
    assert np.isclose(umbral_chi2("wald", 0.05, 100), 3.841458820694124)
    assert np.isclose(stats.chi2.sf(umbral_chi2("aic", 0.05, 100), 1), 0.1572992070502851)
    assert abs(tabla_gl.loc[5].iloc[0] - 0.428) < 1e-3
    # 7) Austral: 8 entradas pasan p<0,05 y AIC; BIC corta antes de uso_tc_prom_3m
    assert trayectoria_austral["p<0,05"].all() and trayectoria_austral["AIC"].all()
    assert not trayectoria_austral.loc[8, "BIC"]
    # 8) con WoE cruzado entra menos ruido que con WoE in-sample (si hay ruido en el pool)
    if RUIDO:
        assert tabla_modos.iloc[1]["ruido"] <= tabla_modos.iloc[0]["ruido"]
    # 9) univariado: β = -1 exacto sin suavizado ⇒ con suavizado, cerca de -1
    _s, _, _aj = stepwise(W_dev[[CANDIDATAS[0]]], y_dev, [CANDIDATAS[0]])
    assert abs(_aj["beta"][1] + 1) < 0.05
    # 10) forzar: LR y Wald numpy = statsmodels
    if resumen_forzar:
        assert np.isclose(resumen_forzar["LR numpy"], resumen_forzar["LR statsmodels"], atol=1e-6)
        assert np.isclose(resumen_forzar["Wald numpy"], resumen_forzar["Wald statsmodels"], rtol=1e-6)
    # 11) la bitácora es reproducible: mismo contenido ⇒ mismo hash
    assert hashlib.sha256(bitacora_json.encode("utf-8")).hexdigest() == hash_bitacora
    assert len(sel_main) <= 25
    "OK: todos los checks del módulo pasan"
    return


if __name__ == "__main__":
    app.run()

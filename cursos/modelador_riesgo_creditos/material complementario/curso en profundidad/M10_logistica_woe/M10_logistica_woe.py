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
    import warnings
    import matplotlib.pyplot as plt
    from scipy import stats
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    return LogisticRegression, mo, plt, roc_auc_score, sm, stats, warnings


@app.cell
def _(mo):
    mo.md(r"""
    # M10 · La regresión logística sobre WoE, desde la verosimilitud

    Este notebook acompaña a `M10_logistica_woe.md`. Todo corre sobre la cartera sintética de la serie
    («Banco Sintético», con **PD verdadera conocida**) y sobre juguetes construidos para aislar un efecto.

    Convenciones del curso: target **1 = malo**; $\text{WoE}=\ln(\%\text{buenos}/\%\text{malos})$, así que
    WoE alto = bin bueno y los coeficientes esperados son **negativos**. Todo se ajusta en DEV.

    Mapa de secciones:

    1. IRLS en numpy vs `statsmodels.Logit` vs `sklearn` (mismos β y errores estándar).
    2. Wald, razón de verosimilitud y score; el efecto Hauck-Donner.
    3. El resultado $\beta=-1$, $\beta_0=\ln(M/B)$ con una variable en WoE (exacto sin suavizado).
    4. Naive Bayes ($\beta_j=-1$ para todas) vs logística multivariada; no colapsabilidad.
    5. Grados de libertad escondidos: p-valores optimistas cuando el WoE se aprende en la misma muestra.
    6. Separación y Firth (implementado en numpy; `statsmodels` no lo trae).
    7. EPV: variabilidad de β en muestras chicas.
    8. WoE vs dummies vs crudo con splines lineales.
    9. Regularización L2: encoger hacia 0 o hacia −1.
    10. Desbalance, submuestreo y corrección del intercepto (King & Zeng).
    11. Checks del módulo.
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
    return a_woe, binear, generar_cartera, np, pd, tabla_woe


@app.cell
def _(mo):
    mo.md(r"""
    ## 0 · Datos: la cartera y las 8 variables del modelo

    Usamos 8 variables que imitan al modelo final de Banco Austral (utilización de línea, recencia de mora,
    uso de tarjeta 12m y 3m, antigüedad, deuda en otras instituciones, carga financiera y consultas).
    **A propósito no aplicamos el filtro de correlación de WoE > 0,70** (ver M09): `uso_tc_prom_12m` y
    `uso_tc_prom_3m` están correlacionadas ~0,9 y eso nos sirve para ver cómo la logística "descuenta"
    evidencia redundante. Bins y WoE se calculan en DEV con las herramientas del curso y se aplican a HO/OOT.
    """)
    return


@app.cell
def _(a_woe, generar_cartera, np, pd, tabla_woe):
    cartera = generar_cartera()
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    ho = cartera[cartera["muestra"] == "HO"].reset_index(drop=True)
    oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)
    VARS = ["uso_linea_prom_12m", "meses_desde_mora_12m", "uso_tc_prom_12m", "uso_tc_prom_3m",
            "antiguedad_meses", "deuda_otras_prom_12m", "carga_financiera", "consultas_6m"]
    _tablas = {v: tabla_woe(dev[v], dev["malo"]) for v in VARS}
    MAPAS = {v: _tablas[v][0]["woe"] for v in VARS}
    IV = pd.Series({v: _tablas[v][1] for v in VARS}, name="IV")
    W_dev = a_woe(dev, VARS, dev, MAPAS)
    W_ho = a_woe(ho, VARS, dev, MAPAS)
    W_oot = a_woe(oot, VARS, dev, MAPAS)
    y_dev = dev["malo"].to_numpy()
    y_ho = ho["malo"].to_numpy()
    y_oot = oot["malo"].to_numpy()
    X_dev = np.column_stack([np.ones(len(dev)), W_dev.to_numpy()])
    X_ho = np.column_stack([np.ones(len(ho)), W_ho.to_numpy()])
    X_oot = np.column_stack([np.ones(len(oot)), W_oot.to_numpy()])
    NOMBRES = ["const"] + VARS
    resumen_muestras = pd.DataFrame({
        "n": [len(dev), len(ho), len(oot)],
        "malos": [int(y_dev.sum()), int(y_ho.sum()), int(y_oot.sum())],
        "tasa_malos": [y_dev.mean(), y_ho.mean(), y_oot.mean()],
    }, index=["DEV", "HO", "OOT"])
    return (IV, MAPAS, NOMBRES, VARS, W_dev, X_dev, X_ho, X_oot, dev, ho,
            oot, resumen_muestras, y_dev, y_ho, y_oot)


@app.cell
def _(IV, W_dev, mo, resumen_muestras):
    mo.vstack([
        mo.md("**Muestras** (DEV ajusta todo; HO y OOT contrastan)"),
        resumen_muestras.style.format({"tasa_malos": "{:.2%}"}),
        mo.md("**IV en DEV de las 8 variables** y **|correlación| entre sus WoE**"),
        IV.round(3).to_frame().T,
        W_dev.corr().abs().round(2),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Máxima verosimilitud: IRLS desde cero vs statsmodels vs scikit-learn

    Log-verosimilitud Bernoulli con $\eta = X\beta$ y $p=\sigma(\eta)$:
    $\ell(\beta)=\sum_i \left[y_i\eta_i-\ln(1+e^{\eta_i})\right]$, gradiente $X^\top(y-p)$, Hessiano
    $-X^\top W X$ con $W=\operatorname{diag}(p_i(1-p_i))$. Newton-Raphson:
    $\beta^{(t+1)}=\beta^{(t)}+(X^\top W X)^{-1}X^\top(y-p)$, que es exactamente mínimos cuadrados
    ponderados iterados (IRLS). Errores estándar: raíz de la diagonal de $(X^\top W X)^{-1}$ en el óptimo.

    Qué mirar: (a) la columna `max|Δβ|` cae cuadráticamente (el número de decimales correctos se duplica por
    iteración); (b) los tres motores dan los mismos β; (c) $\sum_i \hat p_i = \sum_i y_i$ exacto.
    """)
    return


@app.cell
def _(np, pd):
    def ajustar_logit(X, y, w=None, offset=None, tol=1e-10, max_iter=100):
        """Logística por IRLS / Newton-Raphson en numpy.

        X incluye la columna de unos si se quiere intercepto. `w` son pesos por
        observación (frecuencia o 'class_weight'); `offset` entra al predictor
        lineal con coeficiente fijo 1. Devuelve dict con beta, se, cov, ll, p,
        historia de iteraciones y bandera de convergencia.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, k = X.shape
        w = np.ones(n) if w is None else np.asarray(w, float)
        off = np.zeros(n) if offset is None else np.asarray(offset, float)
        beta = np.zeros(k)
        historia = []
        convergio = False
        for it in range(max_iter):
            eta = X @ beta + off
            p = 1.0 / (1.0 + np.exp(-eta))
            ll = float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))
            grad = X.T @ (w * (y - p))                       # X'(y - p)
            info = (X * (w * p * (1 - p))[:, None]).T @ X    # X'WX = -Hessiano
            paso = np.linalg.solve(info, grad)               # Newton = IRLS
            beta = beta + paso
            historia.append((it + 1, ll, float(np.max(np.abs(paso)))))
            if np.max(np.abs(paso)) < tol:
                convergio = True
                break
        eta = X @ beta + off
        p = 1.0 / (1.0 + np.exp(-eta))
        info = (X * (w * p * (1 - p))[:, None]).T @ X
        cov = np.linalg.inv(info)
        return {
            "beta": beta, "se": np.sqrt(np.diag(cov)), "cov": cov, "p": p,
            "ll": float(np.sum(w * (y * eta - np.logaddexp(0.0, eta)))),
            "historia": pd.DataFrame(historia, columns=["iteración", "log_vero_antes", "max|Δβ|"]),
            "convergio": convergio,
        }

    def ll_bernoulli(beta, X, y):
        """Log-verosimilitud media (por observación) de un β fijo en otra muestra."""
        _eta = X @ beta
        return float(np.mean(y * _eta - np.logaddexp(0.0, _eta)))

    return ajustar_logit, ll_bernoulli


@app.cell
def _(LogisticRegression, NOMBRES, X_dev, ajustar_logit, np, pd, sm, warnings, y_dev):
    fit_np = ajustar_logit(X_dev, y_dev)
    fit_sm = sm.Logit(y_dev, X_dev).fit(disp=0, method="newton", tol=1e-12, maxiter=100)
    # sklearn >= 1.8: sin penalización es C=np.inf (penalty=None quedó deprecado)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # sklearn avisa que C=inf equivale a sin penalización
        _sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000).fit(X_dev[:, 1:], y_dev)
    beta_sk = np.r_[_sk.intercept_, _sk.coef_.ravel()]
    # sklearn no entrega errores estándar: se calculan con la información de Fisher en SU β
    _p = 1 / (1 + np.exp(-X_dev @ beta_sk))
    se_sk = np.sqrt(np.diag(np.linalg.inv((X_dev * (_p * (1 - _p))[:, None]).T @ X_dev)))
    comparacion_motores = pd.DataFrame({
        "β numpy": fit_np["beta"], "β statsmodels": fit_sm.params, "β sklearn": beta_sk,
        "SE numpy": fit_np["se"], "SE statsmodels": fit_sm.bse, "SE sklearn*": se_sk,
        "z": fit_np["beta"] / fit_np["se"],
    }, index=NOMBRES)
    suma_p_vs_y = (float(fit_np["p"].sum()), float(y_dev.sum()))
    return beta_sk, comparacion_motores, fit_np, fit_sm, suma_p_vs_y


@app.cell
def _(comparacion_motores, fit_np, fit_sm, mo, np, suma_p_vs_y):
    _dif_sm = np.max(np.abs(fit_np["beta"] - fit_sm.params))
    mo.vstack([
        comparacion_motores.round(4),
        mo.md("**Historia de IRLS** (convergencia cuadrática):"),
        fit_np["historia"].style.format({"log_vero_antes": "{:.6f}", "max|Δβ|": "{:.2e}"}),
        mo.md(f"""
    Lectura: numpy y statsmodels coinciden a {_dif_sm:.1e} (mismo algoritmo); sklearn (L-BFGS, otro
    optimizador) coincide a la tolerancia de su optimizador. `SE sklearn*` se calcula a mano porque sklearn
    no reporta inferencia. Ecuación del intercepto: $\\sum\\hat p_i$ = {suma_p_vs_y[0]:.6f} vs
    $\\sum y_i$ = {suma_p_vs_y[1]:.0f}. **Por eso DEV "clava" la tasa de malos por construcción**: no es
    evidencia de calibración, es la condición de primer orden del intercepto.

    Fíjese en el signo de `uso_tc_prom_12m`: **positivo** ({fit_np['beta'][3]:+.3f}, z =
    {fit_np['beta'][3] / fit_np['se'][3]:.2f}). Con `uso_tc_prom_3m` (correlación ~0,9) en el modelo, la de 12m
    queda como variable de supresión. El filtro de correlación del curso (M09) la habría sacado antes.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Tres tests para la misma hipótesis: Wald, razón de verosimilitud (LR) y score

    Para $H_0:\beta_j=0$: **Wald** usa solo el modelo completo, $z^2=\hat\beta_j^2/\widehat{\text{Var}}(\hat\beta_j)$;
    **LR** compara log-verosimilitudes, $2(\ell_1-\ell_0)$; **score** (Rao, multiplicador de Lagrange) usa solo
    el modelo restringido, $U(\tilde\beta)^\top I(\tilde\beta)^{-1}U(\tilde\beta)$. Los tres son asintóticamente
    $\chi^2_1$ y coinciden cuando el efecto es chico respecto de su precisión. Abajo: numpy vs statsmodels
    (`.bse`, dos `.llf` y `.score_test`).
    """)
    return


@app.cell
def _(VARS, X_dev, ajustar_logit, fit_np, fit_sm, np, pd, sm, stats, y_dev):
    def tests_drop_one(X, y, j, ajuste_completo):
        """Wald, LR y score para H0: beta_j = 0 (numpy)."""
        _Xr = np.delete(X, j, axis=1)
        _fr = ajustar_logit(_Xr, y)
        wald = (ajuste_completo["beta"][j] / ajuste_completo["se"][j]) ** 2
        lr = 2 * (ajuste_completo["ll"] - _fr["ll"])
        _b = np.insert(_fr["beta"], j, 0.0)                  # β restringido en el espacio completo
        _p = 1 / (1 + np.exp(-X @ _b))
        _U = X.T @ (y - _p)
        _I = (X * (_p * (1 - _p))[:, None]).T @ X
        score = float(_U @ np.linalg.solve(_I, _U))
        return wald, lr, score

    _filas = []
    for _j, _v in enumerate(VARS, start=1):
        _w, _lr, _sc = tests_drop_one(X_dev, y_dev, _j, fit_np)
        _r0 = sm.Logit(y_dev, np.delete(X_dev, _j, axis=1)).fit(disp=0, method="newton", tol=1e-12)
        _sc_sm = float(np.asarray(_r0.score_test(exog_extra=X_dev[:, [_j]]).statistic).ravel()[0])
        _filas.append({
            "variable": _v, "Wald": _w, "LR": _lr, "score": _sc,
            "Wald sm": (fit_sm.params[_j] / fit_sm.bse[_j]) ** 2,
            "LR sm": 2 * (fit_sm.llf - _r0.llf), "score sm": _sc_sm,
            "p Wald": stats.chi2.sf(_w, 1), "p LR": stats.chi2.sf(_lr, 1), "p score": stats.chi2.sf(_sc, 1),
        })
    tabla_tests = pd.DataFrame(_filas).set_index("variable")
    return tabla_tests, tests_drop_one


@app.cell
def _(mo, tabla_tests):
    mo.vstack([
        tabla_tests.style.format({c: "{:.3f}" for c in tabla_tests.columns[:6]} |
                                 {c: "{:.2e}" for c in tabla_tests.columns[6:]}),
        mo.md(r"""
    Lectura: con n ≈ 10.000 y efectos moderados, los tres estadísticos están cerca. Las diferencias crecen
    donde el efecto es grande respecto de su precisión: ahí la curvatura de la log-verosimilitud no es
    cuadrática y el Wald (que la aproxima por una parábola en $\hat\beta$) se degrada primero.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.1 · Efecto Hauck-Donner: cuando el Wald *baja* aunque la evidencia *sube*

    Un bin pequeño (60 clientes) con $m$ malos contra el resto de la cartera (1.000 clientes, 50 malos = 5%).
    El modelo es logit = β₀ + β₁·1[bin]; $\hat\beta_1$ es el log odds-ratio y
    $\text{SE}=\sqrt{1/a+1/b+1/c+1/d}$. Mueva $m$: cerca de 0 o de 60 (casi separación) el Wald se desploma
    mientras la LR sigue subiendo. En crédito esto aparece en los bins extremos (el tramo "excelente" con 0–2
    malos): el Wald dice "no significativo" justo en el bin más informativo.
    """)
    return


@app.cell
def _(mo):
    hd_m = mo.ui.slider(0, 60, value=57, step=1, label="m = malos en el bin chico (de 60)")
    hd_m
    return (hd_m,)


@app.cell
def _(np, stats):
    def wald_lr_2x2(m, n_a=60, c=50, n_b=1000):
        """Wald y LR (G²) para un bin de n_a con m malos vs resto n_b con c malos."""
        a, b, d = m, n_a - m, n_b - c
        if a == 0 or b == 0:
            wald = 0.0                         # β infinito, SE infinito: Wald → 0
        else:
            beta = np.log(a / b) - np.log(c / d)
            wald = beta ** 2 / (1 / a + 1 / b + 1 / c + 1 / d)
        obs = np.array([[a, b], [c, d]], float)
        esp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = 2 * np.nansum(np.where(obs > 0, obs * np.log(obs / esp), 0.0))
        return float(wald), float(lr), float(stats.chi2.sf(wald, 1)), float(stats.chi2.sf(lr, 1))

    return (wald_lr_2x2,)


@app.cell
def _(hd_m, mo, np, plt, wald_lr_2x2):
    _ms = np.arange(0, 61)
    _res = np.array([wald_lr_2x2(_m) for _m in _ms])
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.plot(_ms, _res[:, 0], label="Wald z²", color="#c0392b")
    _ax.plot(_ms, _res[:, 1], label="LR (G²)", color="#2c3e50")
    _ax.axvline(hd_m.value, color="grey", ls=":")
    _ax.axhline(3.84, color="grey", lw=0.8, ls="--", label="crítico χ²₁ 5%")
    _ax.set_xlabel("malos en el bin chico (de 60)")
    _ax.set_ylabel("estadístico χ²₁")
    _ax.set_title("Hauck-Donner: el Wald no es monótono en la evidencia")
    _ax.legend(loc="upper left")
    _fig.tight_layout()
    _w, _lr, _pw, _plr = wald_lr_2x2(hd_m.value)
    _m_pico = int(_ms[np.argmax(_res[:, 0])])
    mo.vstack([_fig, mo.md(
        f"Con m = {hd_m.value}: Wald = {_w:.1f} (p = {_pw:.2g}), LR = {_lr:.1f} (p = {_plr:.2g}). "
        f"El Wald alcanza su máximo en m = {_m_pico} y luego **cae** aunque el bin sea cada vez más extremo; "
        f"la LR es monótona. En m = 0 o m = 60 el MLE no existe (separación) y el Wald reportado por "
        f"software es ≈ 0.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · El resultado clave: una variable en WoE ⇒ $\hat\beta_1=-1$ y $\hat\beta_0=\ln(M/B)$

    Con WoE **sin suavizar**, el log-odds observado de malo en el bin $k$ es
    $\ln(m_k/b_k)=\ln(M/B)-\text{WoE}_k$: exactamente lineal en el WoE con pendiente −1. El modelo de 2
    parámetros reproduce el modelo saturado por bins, y el MLE del saturado es la tasa observada de cada bin.
    Con el suavizado +0,5 del curso la igualdad es aproximada. Elija la variable y el tamaño de muestra.
    """)
    return


@app.cell
def _(mo):
    uni_var = mo.ui.dropdown(
        options=["uso_linea_prom_12m", "meses_desde_mora_12m", "uso_tc_prom_12m", "uso_tc_prom_3m",
                 "antiguedad_meses", "deuda_otras_prom_12m", "carga_financiera", "consultas_6m",
                 "canal", "renta_mm", "edad"],
        value="meses_desde_mora_12m", label="Variable")
    uni_n = mo.ui.slider(150, 10000, value=10000, step=50, label="n de la submuestra de DEV")
    uni_semilla = mo.ui.number(start=1, stop=999, value=11, label="semilla")
    mo.hstack([uni_var, uni_n, uni_semilla])
    return uni_n, uni_semilla, uni_var


@app.cell
def _(ajustar_logit, binear, np, pd, sm, tabla_woe):
    def woe_crudo(x, y, bins=5):
        """WoE sin suavizar, ln(%buenos/%malos), con los bins del curso. Devuelve (woe_por_obs, tabla)."""
        _et, _ = binear(x, bins)
        _t = pd.DataFrame({"bin": _et.values, "y": np.asarray(y, float)}).groupby("bin")["y"].agg(["count", "sum"])
        _t.columns = ["n", "malos"]
        _t["buenos"] = _t["n"] - _t["malos"]
        with np.errstate(divide="ignore"):
            _t["woe"] = np.log((_t["buenos"] / _t["buenos"].sum()) / (_t["malos"] / _t["malos"].sum()))
        return _et.map(_t["woe"]).to_numpy(float), _t

    def beta_univariado(x, y):
        """β (MLE) de logit ~ WoE para WoE crudo y WoE suavizado del curso."""
        y = np.asarray(y, float)
        _wc, _tc = woe_crudo(x, y)
        _M, _B = y.sum(), len(y) - y.sum()
        out = {"ln(M/B)": np.log(_M / _B), "bins": len(_tc),
               "min malos/bin": int(_tc["malos"].min()), "min buenos/bin": int(_tc["buenos"].min())}
        if np.all(np.isfinite(_wc)):
            _f = ajustar_logit(np.column_stack([np.ones(len(y)), _wc]), y)
            out["β0 crudo"], out["β1 crudo"] = _f["beta"]
            _s = sm.Logit(y, np.column_stack([np.ones(len(y)), _wc])).fit(disp=0, method="newton", tol=1e-12)
            out["β1 crudo (sm)"] = _s.params[1]
        else:
            out["β0 crudo"] = out["β1 crudo"] = out["β1 crudo (sm)"] = np.nan   # bin con 0 malos o 0 buenos
        _ts, _ = tabla_woe(pd.Series(np.asarray(x)), y)
        _et, _ = binear(pd.Series(np.asarray(x)))
        _ws = _et.map(_ts["woe"]).to_numpy(float)
        _g = ajustar_logit(np.column_stack([np.ones(len(y)), _ws]), y)
        out["β0 suav"], out["β1 suav"] = _g["beta"]
        return out

    return beta_univariado, woe_crudo


@app.cell
def _(beta_univariado, dev, pd, y_dev):
    _todas = ["uso_linea_prom_12m", "meses_desde_mora_12m", "uso_tc_prom_12m", "uso_tc_prom_3m",
              "antiguedad_meses", "deuda_otras_prom_12m", "carga_financiera", "consultas_6m",
              "canal", "renta_mm", "edad"]
    tabla_beta_uni = pd.DataFrame({_v: beta_univariado(dev[_v], y_dev) for _v in _todas}).T
    return (tabla_beta_uni,)


@app.cell
def _(mo, tabla_beta_uni):
    mo.vstack([
        mo.md("**Todas las variables, DEV completo** (n = 10.065):"),
        tabla_beta_uni[["bins", "ln(M/B)", "β0 crudo", "β1 crudo", "β1 crudo (sm)", "β0 suav", "β1 suav"]]
        .style.format("{:.10f}", subset=["ln(M/B)", "β0 crudo", "β1 crudo", "β1 crudo (sm)"])
        .format("{:.4f}", subset=["β0 suav", "β1 suav"]),
        mo.md(f"""
    Sin suavizar, $\\hat\\beta_1=-1$ y $\\hat\\beta_0=\\ln(M/B)$ a 10 decimales para **todas** las variables, incluidas
    `edad` y `renta_mm`, que casi no discriminan (IV < 0,01): el coeficiente univariado sobre WoE **no mide poder**;
    es −1 por construcción. Con el suavizado del curso la máxima desviación es
    {(tabla_beta_uni['β1 suav'] + 1).abs().max():.4f}.
    """),
    ])
    return


@app.cell
def _(beta_univariado, dev, mo, np, plt, uni_n, uni_semilla, uni_var, woe_crudo):
    _rng = np.random.default_rng(int(uni_semilla.value))
    _idx = _rng.choice(len(dev), size=min(int(uni_n.value), len(dev)), replace=False)
    _x = dev[uni_var.value].iloc[_idx].reset_index(drop=True)
    _y = dev["malo"].to_numpy()[_idx]
    _r = beta_univariado(_x, _y)
    _, _t = woe_crudo(_x, _y)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _fin = np.isfinite(_t["woe"]) & (_t["malos"] > 0) & (_t["buenos"] > 0)
    _lo = np.log(_t.loc[_fin, "malos"] / _t.loc[_fin, "buenos"])
    _ax.scatter(_t.loc[_fin, "woe"], _lo, s=_t.loc[_fin, "n"] / _t["n"].max() * 300, alpha=0.6,
                label="bins (área ∝ n)")
    _xx = np.linspace(_t.loc[_fin, "woe"].min() - 0.2, _t.loc[_fin, "woe"].max() + 0.2, 10)
    _ax.plot(_xx, _r["ln(M/B)"] - _xx, color="k", lw=1, label="recta ln(M/B) − WoE")
    _ax.set_xlabel("WoE crudo del bin = ln(%buenos/%malos)")
    _ax.set_ylabel("log-odds observado de malo")
    _ax.set_title(f"{uni_var.value}: log-odds por bin vs WoE (n = {len(_y):,})".replace(",", "."))
    _ax.legend()
    _fig.tight_layout()
    if np.isfinite(_r["β1 crudo"]):
        _txt = (f"WoE crudo: β₁ = {_r['β1 crudo']:.10f}, β₀ = {_r['β0 crudo']:.6f} vs ln(M/B) = "
                f"{_r['ln(M/B)']:.6f}. ")
    else:
        _txt = ("WoE crudo: **hay un bin con 0 malos o 0 buenos** → WoE = ±∞ y el MLE no existe "
                "(separación cuasi-completa del modelo saturado). ")
    mo.vstack([_fig, mo.md(
        _txt + f"WoE suavizado (+0,5 del curso): β₁ = {_r['β1 suav']:.4f}, β₀ = {_r['β0 suav']:.4f}. "
        f"Mínimo de malos en un bin: {_r['min malos/bin']}. Achique n: la desviación del suavizado crece "
        "cuando hay bins con pocos malos, y el signo típico es |β₁| > 1: el MLE 'des-encoge' el suavizado.")])
    return


@app.cell
def _(beta_univariado, dev, np, pd, uni_var):
    # desviación del β suavizado respecto de −1 según n (40 réplicas por tamaño)
    _rng = np.random.default_rng(7)
    _filas = []
    for _n in [150, 300, 600, 1200, 2500, 5000, 10000]:
        _devs = []
        for _r in range(40 if _n < 10000 else 1):
            _idx = _rng.choice(len(dev), size=_n, replace=False)
            _o = beta_univariado(dev[uni_var.value].iloc[_idx].reset_index(drop=True),
                                 dev["malo"].to_numpy()[_idx])
            _devs.append((_o["β1 suav"], np.isfinite(_o["β1 crudo"]), _o["β1 crudo"]))
        _a = np.array(_devs, float)
        _filas.append({"n": _n, "malos esperados": round(_n * dev["malo"].mean()),
                       "β1 suav (media)": _a[:, 0].mean(), "|β1 suav + 1| (media)": np.abs(_a[:, 0] + 1).mean(),
                       "% réplicas con WoE crudo finito": _a[:, 1].mean(),
                       "máx |β1 crudo + 1|": np.nanmax(np.abs(_a[:, 2] + 1)) if _a[:, 1].any() else np.nan})
    tabla_suavizado_n = pd.DataFrame(_filas).set_index("n")
    return (tabla_suavizado_n,)


@app.cell
def _(mo, tabla_suavizado_n, uni_var):
    mo.vstack([
        mo.md(f"**{uni_var.value}: desviación de −1 según tamaño** (40 submuestras por n; binning y WoE "
              "re-calculados en cada submuestra)"),
        tabla_suavizado_n.style.format({"β1 suav (media)": "{:.4f}", "|β1 suav + 1| (media)": "{:.4f}",
                                        "% réplicas con WoE crudo finito": "{:.0%}",
                                        "máx |β1 crudo + 1|": "{:.1e}"}),
        mo.md("Cuando el WoE crudo existe, el β crudo es −1 a precisión de máquina en **todas** las réplicas; "
              "el suavizado introduce un sesgo que es despreciable con cientos de malos por bin y visible con "
              "decenas."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Naive Bayes = logística con todos los $\beta_j=-1$

    Si las variables fueran condicionalmente independientes dado la clase, el log-odds posterior sería
    $\ln(M/B)-\sum_j\text{WoE}_j$ (ver `.md` §3.9). Eso es **Naive Bayes** sobre bins. La logística
    multivariada estima cuánto "descontar" cada evidencia. Comparamos ambos en DEV, HO y OOT: discriminación
    (Gini) y calibración (PD media vs tasa, pendiente de calibración = coeficiente de regresar $y$ sobre el
    logit predicho; 1 es perfecto, < 1 significa predicciones demasiado extremas).
    """)
    return


@app.cell
def _(NOMBRES, X_dev, X_ho, X_oot, ajustar_logit, fit_np, np, pd, roc_auc_score, y_dev, y_ho, y_oot):
    beta_nb = np.r_[np.log(y_dev.sum() / (len(y_dev) - y_dev.sum())), -np.ones(X_dev.shape[1] - 1)]

    def resumen_calibracion(beta, X, y):
        _eta = X @ beta
        _p = 1 / (1 + np.exp(-_eta))
        _pend = ajustar_logit(np.column_stack([np.ones(len(y)), _eta]), y)["beta"][1]
        return {"Gini": 2 * roc_auc_score(y, _p) - 1, "PD media": _p.mean(), "tasa obs": y.mean(),
                "pendiente calib": _pend, "log-vero media": float(np.mean(y * _eta - np.logaddexp(0, _eta)))}

    _filas = []
    for _m, _X, _y in [("DEV", X_dev, y_dev), ("HO", X_ho, y_ho), ("OOT", X_oot, y_oot)]:
        for _nom, _b in [("logística", fit_np["beta"]), ("Naive Bayes", beta_nb)]:
            _filas.append({"muestra": _m, "modelo": _nom} | resumen_calibracion(_b, _X, _y))
    tabla_nb = pd.DataFrame(_filas).set_index(["muestra", "modelo"])
    tabla_descuento = pd.DataFrame({"β logística": fit_np["beta"], "β Naive Bayes": beta_nb,
                                    "factor (β/−1)": -fit_np["beta"]}, index=NOMBRES).iloc[1:]
    return beta_nb, resumen_calibracion, tabla_descuento, tabla_nb


@app.cell
def _(X_ho, beta_nb, fit_np, mo, np, pd, plt, tabla_descuento, tabla_nb, y_ho):
    _fig, _ax = plt.subplots(figsize=(5.5, 4))
    for _nom, _b, _c in [("logística", fit_np["beta"], "#2c3e50"), ("Naive Bayes", beta_nb, "#c0392b")]:
        _p = 1 / (1 + np.exp(-X_ho @ _b))
        _dec = pd.qcut(pd.Series(_p).rank(method="first"), 10, labels=False)
        _g = pd.DataFrame({"p": _p, "y": y_ho, "d": _dec}).groupby("d").mean()
        _ax.plot(_g["p"], _g["y"], "o-", label=_nom, color=_c)
    _ax.plot([0, 0.6], [0, 0.6], color="grey", ls="--", lw=0.8, label="calibración perfecta")
    _ax.set_xlabel("PD media predicha del decil")
    _ax.set_ylabel("tasa de malos observada")
    _ax.set_title("HO: calibración por deciles")
    _ax.legend()
    _fig.tight_layout()
    _t = tabla_nb
    mo.vstack([
        mo.hstack([tabla_descuento.round(3), _fig]),
        _t.style.format({"Gini": "{:.3f}", "PD media": "{:.2%}", "tasa obs": "{:.2%}",
                         "pendiente calib": "{:.3f}", "log-vero media": "{:.4f}"}),
        mo.md(f"""
    Lectura: Naive Bayes pierde poco en ranking (Gini HO {_t.loc[('HO', 'Naive Bayes'), 'Gini']:.3f} vs
    {_t.loc[('HO', 'logística'), 'Gini']:.3f}) pero es **pésimo en nivel**: PD media en HO
    {_t.loc[('HO', 'Naive Bayes'), 'PD media']:.1%} vs {_t.loc[('HO', 'logística'), 'tasa obs']:.1%}
    observado y pendiente de calibración {_t.loc[('HO', 'Naive Bayes'), 'pendiente calib']:.2f}: cuenta dos veces la
    evidencia redundante (las tres de utilización) y sus log-odds quedan ~{1 / _t.loc[('HO', 'Naive Bayes'), 'pendiente calib']:.1f} veces
    demasiado extremos. Los β de la logística son el "descuento": en las tres de utilización el factor va de
    {tabla_descuento['factor (β/−1)'].iloc[[0, 2, 3]].min():.2f} a {tabla_descuento['factor (β/−1)'].iloc[[0, 2, 3]].max():.2f}.
    Las de |β| > 1 (`antiguedad_meses`) no son sinergia: ver el experimento siguiente.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.1 · ¿Por qué |β| > 1 en una variable independiente? No colapsabilidad del odds-ratio

    Juguete: $X_1\sim\text{Bern}(0{,}3)$ y $X_2\sim\text{Bern}(0{,}5)$ **independientes**, logit verdadero
    $-2{,}5+1{,}0\,X_1+\gamma X_2$. Calculamos el WoE univariado de cada una y ajustamos la logística sobre
    ambos WoE. Sin redundancia alguna, $\hat\beta_1$ se aleja de −1 a medida que $\gamma$ crece: el WoE
    univariado de $X_1$ es un log-odds-ratio *marginal*, atenuado por la heterogeneidad que $X_2$ no controla;
    al condicionar en $X_2$ el efecto se "des-atenúa". Es la no colapsabilidad del odds-ratio, no sinergia.
    """)
    return


@app.cell
def _(mo):
    nc_gamma = mo.ui.slider(0.0, 3.0, value=2.0, step=0.25, label="γ = efecto de X₂ (independiente de X₁)")
    nc_gamma
    return (nc_gamma,)


@app.cell
def _(ajustar_logit, mo, nc_gamma, np):
    def no_colapsabilidad(gamma, n=200_000, semilla=5):
        _rng = np.random.default_rng(semilla)
        _x1 = (_rng.random(n) < 0.3).astype(float)
        _x2 = (_rng.random(n) < 0.5).astype(float)
        _y = (_rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 1.0 * _x1 + gamma * _x2)))).astype(float)

        def _woe(x):
            _b = np.array([np.sum((x == k) & (_y == 0)) for k in (0, 1)], float)
            _m = np.array([np.sum((x == k) & (_y == 1)) for k in (0, 1)], float)
            _w = np.log((_b / _b.sum()) / (_m / _m.sum()))
            return _w[x.astype(int)]

        _w1, _w2 = _woe(_x1), _woe(_x2)
        _uni = ajustar_logit(np.column_stack([np.ones(n), _w1]), _y)["beta"][1]
        _f = ajustar_logit(np.column_stack([np.ones(n), _w1, _w2]), _y) if gamma > 0 else None
        _b1 = _f["beta"][1] if _f is not None else _uni
        _b2 = _f["beta"][2] if _f is not None else np.nan
        return {"β1 univariado": _uni, "β1 bivariado": _b1, "β2 bivariado": _b2,
                "corr(WoE1,WoE2)": float(np.corrcoef(_w1, _w2)[0, 1])}

    nc_res = no_colapsabilidad(nc_gamma.value)
    mo.md(f"γ = {nc_gamma.value}: β₁ univariado = {nc_res['β1 univariado']:.4f} · "
          f"**β₁ bivariado = {nc_res['β1 bivariado']:.4f}** · β₂ bivariado = {nc_res['β2 bivariado']:.4f} · "
          f"corr(WoE₁, WoE₂) = {nc_res['corr(WoE1,WoE2)']:.4f}. Con variables independientes, |β| > 1 es la "
          "norma cuando otra variable fuerte entra al modelo.")
    return (no_colapsabilidad,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Grados de libertad escondidos: el WoE se aprendió mirando $y$

    Una variable de **ruido puro** (independiente del target), n = 3.322 y 5% de malos (el DEV de Austral).
    Se binea, se calcula su WoE **en la misma muestra** y se testea $\beta=0$ en la logística univariada.
    Como $\hat\beta=-1$ por construcción, el Wald solo mide cuánto se abrió el WoE por azar: su distribución
    nula es $\chi^2_{K-1}$, no $\chi^2_1$ (se gastaron $K-1$ parámetros al estimar los WoE). Con binning
    supervisado (cortes elegidos maximizando IV) es peor aún. Remedios: LR contra $\chi^2_{K-1}$ (solo si los
    cortes no miraron $y$) o **cross-fitting** (cortes y WoE en una mitad, test en la otra).
    """)
    return


@app.cell
def _(mo):
    gl_bins = mo.ui.slider(2, 10, value=5, step=1, label="K = número de bins")
    gl_tipo = mo.ui.dropdown(options=["cuantiles (no mira y)", "supervisado (maximiza IV)"],
                             value="cuantiles (no mira y)", label="Binning")
    gl_reps = mo.ui.slider(100, 1000, value=200, step=100, label="réplicas")
    mo.hstack([gl_bins, gl_tipo, gl_reps])
    return gl_bins, gl_reps, gl_tipo


@app.cell
def _(ajustar_logit, np, pd, stats):
    def _woe_codigos(codigos, y, K, suav=0.5):
        _m = np.bincount(codigos, weights=y, minlength=K)
        _n = np.bincount(codigos, minlength=K)
        _b = _n - _m
        _pb = (_b + suav) / (_b.sum() + suav * K)
        _pm = (_m + suav) / (_m.sum() + suav * K)
        return np.log(_pb / _pm), float(np.sum((_pb - _pm) * np.log(_pb / _pm)))

    def _cortes_cuantiles(x, K):
        return np.quantile(x, np.linspace(0, 1, K + 1)[1:-1])

    def _cortes_supervisados(x, y, K, grilla=16):
        """Greedy tipo árbol: agrega el corte (de una grilla de cuantiles) que más sube el IV."""
        _cand = list(np.quantile(x, np.linspace(0, 1, grilla + 1)[1:-1]))
        _cortes = []
        for _ in range(K - 1):
            _mejor = None
            for _c in _cand:
                if _c in _cortes:
                    continue
                _cc = np.sort(_cortes + [_c])
                _iv = _woe_codigos(np.searchsorted(_cc, x, side="right"), y, len(_cc) + 1)[1]
                if _mejor is None or _iv > _mejor[0]:
                    _mejor = (_iv, _c)
            _cortes.append(_mejor[1])
        return np.sort(_cortes)

    def simular_gl_escondidos(K, supervisado, reps, n=3322, pi=0.05, semilla=0):
        """Tasa de rechazo al 5% de H0: β=0 para una variable de ruido en WoE."""
        _rng = np.random.default_rng(semilla)
        _filas = []
        for _r in range(reps):
            _x = _rng.random(n)
            _y = (_rng.random(n) < pi).astype(float)
            _cortes = _cortes_supervisados(_x, _y, K) if supervisado else _cortes_cuantiles(_x, K)
            _cod = np.searchsorted(_cortes, _x, side="right")
            _w, _iv = _woe_codigos(_cod, _y, K)
            if np.ptp(_w[_cod]) < 1e-12:
                continue
            _f = ajustar_logit(np.column_stack([np.ones(n), _w[_cod]]), _y)
            _ll0 = float(np.sum(_y * np.log(_y.mean()) + (1 - _y) * np.log(1 - _y.mean())))
            _lr = 2 * (_f["ll"] - _ll0)
            _z2 = (_f["beta"][1] / _f["se"][1]) ** 2
            # cross-fitting: cortes + WoE en la mitad A, test (unilateral, β<0) en la mitad B
            _A = _rng.random(n) < 0.5
            _cA = _cortes_supervisados(_x[_A], _y[_A], K) if supervisado else _cortes_cuantiles(_x[_A], K)
            _wA, _ = _woe_codigos(np.searchsorted(_cA, _x[_A], side="right"), _y[_A], K)
            _xb = _wA[np.searchsorted(_cA, _x[~_A], side="right")]
            if np.ptp(_xb) < 1e-12:
                _rech_cf = False
            else:
                _fb = ajustar_logit(np.column_stack([np.ones(len(_xb)), _xb]), _y[~_A])
                _rech_cf = (_fb["beta"][1] / _fb["se"][1]) < stats.norm.ppf(0.05)
            _M = _y.sum()
            _filas.append({"β": _f["beta"][1], "Wald": _z2, "LR": _lr, "IV": _iv,
                           "z2 / [IV/(1/M+1/B)]": _z2 / (_iv / (1 / _M + 1 / (n - _M))),
                           "rech Wald χ²₁": stats.chi2.sf(_z2, 1) < 0.05,
                           "rech LR χ²₁": stats.chi2.sf(_lr, 1) < 0.05,
                           f"rech LR χ²_(K−1)": stats.chi2.sf(_lr, K - 1) < 0.05,
                           "rech cross-fit": bool(_rech_cf)})
        return pd.DataFrame(_filas)

    return (simular_gl_escondidos,)


@app.cell
def _(gl_bins, gl_reps, gl_tipo, simular_gl_escondidos):
    sim_gl = simular_gl_escondidos(int(gl_bins.value), gl_tipo.value.startswith("supervisado"),
                                   int(gl_reps.value))
    return (sim_gl,)


@app.cell
def _(gl_bins, gl_tipo, mo, np, plt, sim_gl, stats):
    _K = int(gl_bins.value)
    _tasas = sim_gl.filter(like="rech").mean()
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _ax.hist(sim_gl["LR"], bins=30, density=True, alpha=0.5, label="LR simulado (ruido)")
    _xx = np.linspace(0.05, max(sim_gl["LR"].quantile(0.99), 8), 200)
    _ax.plot(_xx, stats.chi2.pdf(_xx, 1), label="χ²₁ (lo que asume el p-valor)", color="#c0392b")
    if _K > 1:
        _ax.plot(_xx, stats.chi2.pdf(_xx, _K - 1), label=f"χ²_{_K - 1}", color="#2c3e50")
    _ax.set_xlabel("estadístico LR")
    _ax.set_ylabel("densidad")
    _ax.set_title(f"Ruido puro en WoE, K = {_K}, {gl_tipo.value}")
    _ax.set_ylim(0, 1.0)
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([
        mo.hstack([_tasas.rename("tasa de rechazo (nominal 5%)").to_frame().style.format("{:.1%}"), _fig]),
        mo.md(f"""
    Lectura: β medio = {sim_gl['β'].mean():.3f} (≈ −1 aunque la variable sea ruido). IV medio del ruido =
    {sim_gl['IV'].mean():.4f} vs teoría $(K-1)(1/M+1/B)$ = {(_K - 1) * (1 / 166 + 1 / 3156):.4f} (con M ≈ 166).
    Cociente medio $z^2/[\\text{{IV}}/(1/M+1/B)]$ = {sim_gl['z2 / [IV/(1/M+1/B)]'].mean():.3f} (≈ 1): **el Wald de
    una variable en WoE es un test de IV disfrazado**. Con K = {_K} el p-valor naive rechaza
    {_tasas.iloc[0]:.0%} de las veces una variable que no aporta nada. El cross-fitting devuelve la tasa al
    nominal incluso con binning supervisado, al costo de usar media muestra.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Separación cuasi-completa y Firth

    Tabla de juguete de 5 bins de una variable; el último bin (20 clientes, antigüedad muy alta) tiene
    **0 malos**. Con dummies por bin, el MLE del coeficiente de ese bin es −∞: IRLS "converge" a números
    enormes con SE gigantes. Firth (1993) maximiza $\ell(\beta)+\tfrac12\ln|I(\beta)|$ (prior de Jeffreys):
    score modificado $X^\top\!\left(y-p+h\odot(\tfrac12-p)\right)$ con $h$ la diagonal de la matriz sombrero
    ponderada. `statsmodels` no trae Firth: la validación es contra el **resultado analítico** (en un modelo
    saturado por bins, Firth = sumar 0,5 a cada celda).
    """)
    return


@app.cell
def _(np):
    def ajustar_firth(X, y, tol=1e-10, max_iter=200, paso_max=5.0):
        """Logística de Firth (Jeffreys) en numpy: score modificado + Newton con paso acotado."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        beta = np.zeros(X.shape[1])
        for it in range(max_iter):
            p = 1 / (1 + np.exp(-(X @ beta)))
            wv = p * (1 - p)
            info = (X * wv[:, None]).T @ X
            info_inv = np.linalg.inv(info)
            h = np.einsum("ij,jk,ik->i", X, info_inv, X) * wv          # diag de W^½X(X'WX)⁻¹X'W^½
            U = X.T @ (y - p + h * (0.5 - p))
            paso = info_inv @ U
            _mx = np.max(np.abs(paso))
            if _mx > paso_max:
                paso = paso * paso_max / _mx
            beta = beta + paso
            if np.max(np.abs(paso)) < tol:
                break
        p = 1 / (1 + np.exp(-(X @ beta)))
        info = (X * (p * (1 - p))[:, None]).T @ X
        return {"beta": beta, "se": np.sqrt(np.diag(np.linalg.inv(info))), "p": p, "iter": it + 1}

    return (ajustar_firth,)


@app.cell
def _(ajustar_firth, ajustar_logit, np, pd, sm, warnings):
    sep_n = np.array([400, 300, 200, 80, 20])
    sep_m = np.array([40, 15, 6, 2, 0])            # malos por bin: el último no tiene malos
    _grupo = np.repeat(np.arange(5), sep_n)
    sep_y = np.concatenate([np.r_[np.ones(_m), np.zeros(_n - _m)] for _n, _m in zip(sep_n, sep_m)])
    sep_X = np.column_stack([np.ones(len(sep_y))] + [(_grupo == _k).astype(float) for _k in range(1, 5)])
    sep_mle = ajustar_logit(sep_X, sep_y, max_iter=25, tol=1e-14)
    with warnings.catch_warnings(record=True) as _avisos:
        warnings.simplefilter("always")
        _sm = sm.Logit(sep_y, sep_X).fit(disp=0, maxiter=100)
    sep_sm_beta, sep_sm_bse = np.asarray(_sm.params), np.asarray(_sm.bse)
    sep_avisos = sorted({str(_a.message)[:90] for _a in _avisos})
    sep_firth = ajustar_firth(sep_X, sep_y)
    # log-odds por bin: Firth vs analítico (+0,5 a cada celda) vs WoE suavizado del curso
    _eta_f = np.array([(sep_X @ sep_firth["beta"])[_grupo == _k][0] for _k in range(5)])
    _b = sep_n - sep_m
    sep_logit_analitico = np.log((sep_m + 0.5) / (_b + 0.5))
    _K = 5
    sep_woe_curso = np.log(((_b + 0.5) / (_b.sum() + 0.5 * _K)) / ((sep_m + 0.5) / (sep_m.sum() + 0.5 * _K)))
    sep_const = np.log((sep_m.sum() + 0.5 * _K) / (_b.sum() + 0.5 * _K))
    tabla_sep = pd.DataFrame({
        "n": sep_n, "malos": sep_m, "logit obs": np.log(np.where(sep_m > 0, sep_m, np.nan) / _b),
        "logit MLE (25 it)": [(sep_X @ sep_mle["beta"])[_grupo == _k][0] for _k in range(5)],
        "logit Firth": _eta_f, "ln((m+½)/(b+½))": sep_logit_analitico,
        "WoE curso (+0,5)": sep_woe_curso, "−WoE curso + c": -sep_woe_curso + sep_const,
    }, index=[f"bin {_k + 1}" for _k in range(5)])
    sep_eta_firth = _eta_f
    return (sep_X, sep_avisos, sep_const, sep_eta_firth, sep_firth, sep_logit_analitico, sep_m,
            sep_mle, sep_n, sep_sm_beta, sep_sm_bse, sep_woe_curso, sep_y, tabla_sep)


@app.cell
def _(ajustar_logit, mo, np, sep_X, sep_avisos, sep_firth, sep_mle, sep_sm_beta, sep_sm_bse, sep_y, tabla_sep):
    # FLIC (Puhr et al. 2017): Firth para las pendientes, intercepto re-estimado por ML con offset
    _off = sep_X[:, 1:] @ sep_firth["beta"][1:]
    _flic = ajustar_logit(sep_X[:, :1], sep_y, offset=_off)
    _p_flic = 1 / (1 + np.exp(-(_flic["beta"][0] + _off)))
    mo.vstack([
        tabla_sep.round(4),
        mo.md(f"""
    - **MLE (numpy, 25 iteraciones)**: β del bin 5 = {sep_mle['beta'][4]:.1f}, SE = {sep_mle['se'][4]:.3g}. Cada
      iteración baja ese β en ≈ 1 (último max|Δβ| = {sep_mle['historia']['max|Δβ|'].iloc[-1]:.2f}) y la
      log-verosimilitud mejora en cantidades del orden de $e^{{\\beta}}$: no hay óptimo finito, solo un supremo.
    - **statsmodels**: β = {sep_sm_beta[4]:.1f}, SE = {sep_sm_bse[4]:.3g}. Avisos: {'; '.join(sep_avisos) or 'ninguno'}.
      Un Wald con ese SE da p ≈ 1 para el bin más limpio de la tabla (Hauck-Donner extremo).
    - **Firth (numpy)**: β = {sep_firth['beta'][4]:.3f}, SE = {sep_firth['se'][4]:.3f}, en {sep_firth['iter']}
      iteraciones. Los log-odds por bin coinciden con $\\ln((m+½)/(b+½))$ (columna analítica) a
      {np.max(np.abs(tabla_sep['logit Firth'] - tabla_sep['ln((m+½)/(b+½))'])):.1e}.
    - **Conexión con el curso**: la última columna muestra que $-\\text{{WoE}}_{{+0,5}}+c$ es *idéntico* al logit
      de Firth por bin. El suavizado +0,5 del curso **es** Firth sobre el modelo saturado de la variable.
    - Precio de Firth: $\\sum\\hat p$ = {sep_firth['p'].sum():.2f} vs $\\sum y$ = {sep_y.sum():.0f} (el prior empuja
      hacia 0,5 y rompe la ecuación del intercepto). FLIC re-estima solo el intercepto: $\\sum\\hat p$ =
      {_p_flic.sum():.4f}.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · EPV: cuánto se mueve β con pocos malos

    Re-estimamos el modelo de 8 variables (WoE fijo de DEV, como el re-ajuste en HO del curso) en submuestras
    de DEV con $\text{EPV}\times 8$ malos esperados. Referencia del curso: HO de Austral tenía 78 malos para 9
    parámetros, **EPV ≈ 8,7** (o 9,75 por variable sin contar el intercepto). Mire la dispersión, las
    inversiones de signo y el sesgo (MLE vs Firth) respecto del β de DEV completo.
    """)
    return


@app.cell
def _(mo):
    epv_valor = mo.ui.slider(2, 40, value=9, step=1, label="EPV (malos por variable)")
    epv_reps = mo.ui.slider(50, 500, value=150, step=50, label="réplicas")
    mo.hstack([epv_valor, epv_reps])
    return epv_reps, epv_valor


@app.cell
def _(VARS, X_dev, ajustar_firth, ajustar_logit, fit_np, np, pd, y_dev):
    def experimento_epv(epv, reps, semilla=3):
        _rng = np.random.default_rng(semilla)
        _n = int(round(epv * len(VARS) / y_dev.mean()))
        _B, _F, _fallas = [], [], 0
        for _r in range(reps):
            _i = _rng.choice(len(y_dev), _n, replace=False)
            try:
                _f = ajustar_logit(X_dev[_i], y_dev[_i], max_iter=50)
                if not _f["convergio"] or np.max(np.abs(_f["beta"])) > 15:
                    _fallas += 1
                    continue
            except np.linalg.LinAlgError:
                _fallas += 1
                continue
            _B.append(_f["beta"][1:])
            _F.append(ajustar_firth(X_dev[_i], y_dev[_i])["beta"][1:])
        _B, _F = np.array(_B), np.array(_F)
        _ref = fit_np["beta"][1:]
        tabla = pd.DataFrame({
            "β DEV completo": _ref, "media MLE": _B.mean(0), "DE MLE": _B.std(0, ddof=1),
            "media Firth": _F.mean(0), "DE Firth": _F.std(0, ddof=1),
            "% signo invertido": (np.sign(_B) != np.sign(_ref)).mean(0),
        }, index=VARS)
        return tabla, _B, _n, _fallas

    return (experimento_epv,)


@app.cell
def _(IV, VARS, epv_reps, epv_valor, experimento_epv, fit_np, mo, np, plt):
    tabla_epv, _B, _n, _fallas = experimento_epv(int(epv_valor.value), int(epv_reps.value))
    _fig, _ax = plt.subplots(figsize=(7.5, 3.8))
    _ax.boxplot(_B, showfliers=False)
    _ax.set_xticks(range(1, len(VARS) + 1))
    _ax.set_xticklabels([v.replace("_prom", "").replace("_meses", "") for v in VARS], rotation=30, ha="right",
                        fontsize=8)
    _ax.scatter(range(1, len(VARS) + 1), fit_np["beta"][1:], color="#c0392b", zorder=3, label="β en DEV completo")
    _ax.axhline(0, color="grey", lw=0.8)
    _ax.set_ylabel("β re-estimado")
    _ax.set_title(f"EPV = {epv_valor.value}: n = {_n}, ~{int(epv_valor.value) * 8} malos")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, tabla_epv.style.format("{:.3f}").format("{:.0%}", subset=["% signo invertido"]),
               mo.md(f"Réplicas descartadas por no convergencia/separación: {_fallas}. Con EPV ≈ 9 la DE de "
                     f"`deuda_otras_prom_12m` (IV {IV['deuda_otras_prom_12m']:.3f}) es del orden de su propio β: ese es el "
                     "'cambio de signo a ≈ 0 con p 0,96' que el curso vio en HO. Las variables de IV alto "
                     "(utilización, mora) se estiman bien incluso con EPV bajo: el EPV por sí solo no basta; "
                     "importa la información por parámetro.")])
    return (tabla_epv,)


@app.cell
def _(VARS, fit_np, mo, np, y_dev):
    # Criterio de contracción de Riley et al. (2019): n = P / ((S − 1)·ln(1 − R²_CS / S)), S = 0,9
    _n = len(y_dev)
    _yb = y_dev.mean()
    _ll0 = _n * (_yb * np.log(_yb) + (1 - _yb) * np.log(1 - _yb))
    r2_cs = 1 - np.exp(2 * (_ll0 - fit_np["ll"]) / _n)
    n_riley = len(VARS) / ((0.9 - 1) * np.log(1 - r2_cs / 0.9))
    mo.md(f"""
    **Tamaño muestral por contracción (Riley et al. 2019)**: con el $R^2$ de Cox-Snell del modelo de DEV
    ({r2_cs:.4f}), P = {len(VARS)} parámetros y contracción esperada S = 0,9, se necesitan n ≈ {n_riley:.0f}
    créditos, es decir ≈ {n_riley * _yb:.0f} malos (EPV ≈ {n_riley * _yb / len(VARS):.1f}) a la tasa de DEV. El
    requisito depende de cuánto explica el modelo, no solo de contar malos. Supuesto: el $R^2$ anticipado es el
    de DEV (optimista); con un $R^2$ más bajo el n requerido sube.
    """)
    return n_riley, r2_cs


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · WoE vs dummies vs crudo con splines lineales

    Tres codificaciones de las **mismas 8 variables**: (a) WoE (1 parámetro por variable, forma fijada por el
    binning), (b) dummies por bin (un parámetro por bin menos uno), (c) valor crudo + splines lineales con
    nudos en los cuartiles de DEV (y dummies para los códigos especiales −99/−9/13 de la mora). Comparamos
    parámetros, AIC en DEV, y Gini y log-verosimilitud fuera de muestra.
    """)
    return


@app.cell
def _(VARS, X_dev, X_ho, X_oot, ajustar_logit, binear, dev, ho, np, oot, pd, roc_auc_score, y_dev, y_ho, y_oot):
    def _dummies(df):
        _cols = []
        for _v in VARS:
            _et_dev, _orden = binear(dev[_v])
            _niv = _orden if _orden else sorted(_et_dev.unique())
            _niv = [c for c in _niv if (_et_dev == c).any()]
            _et, _ = binear(df[_v], ref=dev[_v])
            for _c in _niv[1:]:
                _cols.append((_et == _c).to_numpy(float))
        return np.column_stack([np.ones(len(df))] + _cols)

    def _splines(df):
        _cols = []
        for _v in VARS:
            _x = df[_v].to_numpy(float)
            _xd = dev[_v].to_numpy(float)
            if _v == "meses_desde_mora_12m":
                for _cod in (-99.0, -9.0, 13.0):
                    _cols.append((_x == _cod).astype(float))
                _ok = (_x >= 1) & (_x <= 12)
                _x = np.where(_ok, _x, 0.0)
                _xd = _xd[(_xd >= 1) & (_xd <= 12)]
            _mu, _sd = np.mean(_xd), np.std(_xd) + 1e-12
            _cols.append((_x - _mu) / _sd)
            for _q in np.unique(np.quantile(_xd, [0.25, 0.5, 0.75])):
                _cols.append(np.maximum(_x - _q, 0) / _sd)
        return np.column_stack([np.ones(len(df))] + _cols)

    _disenos = {"WoE": (X_dev, X_ho, X_oot), "dummies por bin": (_dummies(dev), _dummies(ho), _dummies(oot)),
                "crudo + splines lineales": (_splines(dev), _splines(ho), _splines(oot))}
    _filas = []
    for _nom, (_Xd, _Xh, _Xo) in _disenos.items():
        _f = ajustar_logit(_Xd, y_dev)
        _fila = {"codificación": _nom, "parámetros": _Xd.shape[1], "AIC DEV": 2 * _Xd.shape[1] - 2 * _f["ll"]}
        for _m, _X, _y in [("DEV", _Xd, y_dev), ("HO", _Xh, y_ho), ("OOT", _Xo, y_oot)]:
            _eta = _X @ _f["beta"]
            _fila[f"Gini {_m}"] = 2 * roc_auc_score(_y, _eta) - 1
            if _m != "DEV":
                _fila[f"log-vero media {_m}"] = float(np.mean(_y * _eta - np.logaddexp(0, _eta)))
        _filas.append(_fila)
    tabla_codificaciones = pd.DataFrame(_filas).set_index("codificación")
    return (tabla_codificaciones,)


@app.cell
def _(mo, tabla_codificaciones):
    _t = tabla_codificaciones
    mo.vstack([
        _t.style.format("{:.4f}").format("{:.0f}", subset=["parámetros"]).format("{:.1f}", subset=["AIC DEV"]),
        mo.md(f"""
    Lectura: las dummies gastan {_t.loc['dummies por bin', 'parámetros']:.0f} parámetros contra
    {_t.loc['WoE', 'parámetros']:.0f} del WoE y no ganan fuera de muestra (Gini HO
    {_t.loc['dummies por bin', 'Gini HO']:.3f} vs {_t.loc['WoE', 'Gini HO']:.3f}): el WoE es una codificación por
    bins con la forma **fijada** (los bins de una variable se mueven juntos, en proporción a su WoE univariado).
    Los splines **sí ganan** aquí (Gini HO {_t.loc['crudo + splines lineales', 'Gini HO']:.3f}, +
    {_t.loc['crudo + splines lineales', 'Gini HO'] - _t.loc['WoE', 'Gini HO']:.3f}): el generador tiene efectos
    suaves (la utilización entra lineal en el logit verdadero) y 5 bins pierden la variación dentro del bin. Es el
    costo honesto del WoE que el curso menciona («se pierde granularidad»). Los splines exigen decidir nudos, tratar
    códigos especiales a mano y no garantizan monotonía ni puntos por tramo. Ojo: el AIC del WoE es optimista
    porque no cuenta los $K-1$ parámetros por variable gastados al estimar los WoE (§5).
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 · Regularización L2: ¿hacia 0 o hacia −1?

    Ridge estándar encoge β hacia 0 = "esta variable no aporta". Sobre WoE el centro natural es **−1** (Naive
    Bayes: "creo en la evidencia univariada completa"). Implementamos en numpy
    $\max_\beta \ \ell(\beta)-\tfrac{\lambda}{2}\lVert\beta_{1:}-c\rVert^2$ con $c\in\{0,-1\}$ (intercepto sin
    penalizar), validamos $c=0$ contra `sklearn` ($C=1/\lambda$), y medimos la log-verosimilitud en HO de
    modelos entrenados con pocas observaciones.
    """)
    return


@app.cell
def _(mo):
    reg_epv = mo.ui.slider(2, 30, value=5, step=1, label="EPV de la muestra de entrenamiento")
    reg_lambda = mo.ui.dropdown(options=["0.5", "1", "2", "5", "10", "20", "50"], value="5",
                                label="λ para la tabla de β")
    mo.hstack([reg_epv, reg_lambda])
    return reg_epv, reg_lambda


@app.cell
def _(np):
    def ajustar_ridge(X, y, lam, centro=0.0, tol=1e-10, max_iter=100):
        """Logística con penalización (λ/2)·||β_1: − centro||² (intercepto libre), Newton en numpy."""
        k = X.shape[1]
        P = np.eye(k) * lam
        P[0, 0] = 0.0
        c = np.r_[0.0, np.full(k - 1, centro)]
        beta = np.r_[np.log(y.mean() / (1 - y.mean())), np.full(k - 1, centro)]
        for _ in range(max_iter):
            p = 1 / (1 + np.exp(-(X @ beta)))
            g = X.T @ (y - p) - P @ (beta - c)
            H = (X * (p * (1 - p))[:, None]).T @ X + P
            paso = np.linalg.solve(H, g)
            beta = beta + paso
            if np.max(np.abs(paso)) < tol:
                break
        return beta

    return (ajustar_ridge,)


@app.cell
def _(LogisticRegression, X_dev, X_ho, ajustar_logit, ajustar_ridge, ll_bernoulli, np, pd, reg_epv, y_dev, y_ho):
    _lams = [0.0, 0.5, 1, 2, 5, 10, 20, 50]
    _rng = np.random.default_rng(21)
    _n = int(round(reg_epv.value * 8 / y_dev.mean()))
    _res = {(_c, _l): [] for _c in (0.0, -1.0) for _l in _lams}
    _mle = []
    for _r in range(80):
        _i = _rng.choice(len(y_dev), _n, replace=False)
        _Xs, _ys = X_dev[_i], y_dev[_i]
        for _c in (0.0, -1.0):
            for _l in _lams:
                _b = ajustar_ridge(_Xs, _ys, _l, _c) if _l > 0 else ajustar_logit(_Xs, _ys)["beta"]
                _res[(_c, _l)].append(ll_bernoulli(_b, X_ho, y_ho))
    tabla_ridge = pd.DataFrame({
        "λ": _lams,
        "hacia 0": [np.mean(_res[(0.0, _l)]) for _l in _lams],
        "hacia −1": [np.mean(_res[(-1.0, _l)]) for _l in _lams],
    }).set_index("λ")
    # validación numpy vs sklearn (c = 0): sklearn minimiza C·Σlogloss + ½||w||²  ⇒  λ = 1/C
    _lam = 5.0
    _b_np = ajustar_ridge(X_dev, y_dev, _lam, 0.0)
    _sk = LogisticRegression(C=1 / _lam, tol=1e-12, max_iter=10_000).fit(X_dev[:, 1:], y_dev)
    ridge_np_vs_sk = (_b_np, np.r_[_sk.intercept_, _sk.coef_.ravel()])
    # la trampa del default: C = 1.0 penaliza aunque nadie lo haya pedido
    _sk_def = LogisticRegression(max_iter=10_000).fit(X_dev[:, 1:], y_dev)
    sklearn_default_dif = float(np.max(np.abs(np.r_[_sk_def.intercept_, _sk_def.coef_.ravel()]
                                              - ajustar_logit(X_dev, y_dev)["beta"])))
    ridge_n = _n
    return ridge_n, ridge_np_vs_sk, sklearn_default_dif, tabla_ridge


@app.cell
def _(NOMBRES, X_dev, ajustar_ridge, fit_np, mo, pd, plt, reg_lambda, ridge_n, ridge_np_vs_sk,
      sklearn_default_dif, tabla_ridge, y_dev):
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _l = tabla_ridge.index.to_numpy()[1:]
    _ax.semilogx(_l, tabla_ridge["hacia 0"].to_numpy()[1:], "o-", label="L2 hacia 0", color="#c0392b")
    _ax.semilogx(_l, tabla_ridge["hacia −1"].to_numpy()[1:], "o-", label="L2 hacia −1 (Naive Bayes)",
                 color="#2c3e50")
    _ax.axhline(tabla_ridge.loc[0.0, "hacia 0"], color="grey", ls="--", lw=0.8, label="MLE sin penalizar")
    _ax.set_xlabel("λ (escala log)")
    _ax.set_ylabel("log-verosimilitud media en HO")
    _ax.set_title(f"Entrenando con n = {ridge_n} (80 submuestras)")
    _ax.legend()
    _fig.tight_layout()
    _lam = float(reg_lambda.value)
    _tb = pd.DataFrame({"MLE DEV completo": fit_np["beta"],
                        f"L2→0, λ={_lam:g}": ajustar_ridge(X_dev, y_dev, _lam, 0.0),
                        f"L2→−1, λ={_lam:g}": ajustar_ridge(X_dev, y_dev, _lam, -1.0),
                        "sklearn C=1/5 (valida λ=5)": ridge_np_vs_sk[1]}, index=NOMBRES)
    mo.vstack([mo.hstack([_fig, tabla_ridge.round(4)]), _tb.round(4), mo.md(f"""
    Lectura: con pocos malos ambas penalizaciones mejoran al MLE fuera de muestra; encoger hacia −1 suele ser
    igual o mejor que hacia 0 porque parte de una creencia más razonable sobre WoE. Con DEV completo (n ≈ 10.000)
    λ = {_lam:g} mueve poco. Trampa: `LogisticRegression()` de sklearn con su default **C = 1** ya es L2; en DEV
    completo difiere del MLE en hasta {sklearn_default_dif:.3f} (en `deuda_otras_prom_12m`, la de WoE menos disperso:
    la penalización L2 no es invariante a la escala y castiga más a las variables cuyo WoE se abre poco). En
    muestras chicas es un sesgo silencioso que ningún reporte de p-valores menciona.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · Desbalance: submuestrear buenos y corregir el intercepto

    Submuestreamos los buenos de DEV hasta una tasa de malos objetivo. Si el modelo está bien especificado,
    el submuestreo por clase solo desplaza el intercepto en $\ln\!\left[\frac{1-\tau}{\tau}\frac{\bar
    y}{1-\bar y}\right]$ ($\tau$ = tasa poblacional, $\bar y$ = tasa de la muestra): *prior correction* de
    King & Zeng (2001). También comparamos `class_weight="balanced"` (pesos) en sklearn vs IRLS ponderado en numpy.
    """)
    return


@app.cell
def _(mo):
    des_tasa = mo.ui.slider(0.15, 0.5, value=0.5, step=0.05, label="tasa de malos objetivo tras submuestrear")
    des_tasa
    return (des_tasa,)


@app.cell
def _(LogisticRegression, NOMBRES, X_dev, X_ho, ajustar_logit, des_tasa, fit_np, mo, np, pd, warnings, y_dev, y_ho):
    _rng = np.random.default_rng(13)
    _malos = np.flatnonzero(y_dev == 1)
    _buenos = np.flatnonzero(y_dev == 0)
    _nb = int(round(len(_malos) * (1 - des_tasa.value) / des_tasa.value))
    _idx = np.r_[_malos, _rng.choice(_buenos, size=min(_nb, len(_buenos)), replace=False)]
    _f = ajustar_logit(X_dev[_idx], y_dev[_idx])
    _tau, _ybar = y_dev.mean(), y_dev[_idx].mean()
    _corr = np.log((1 - _tau) / _tau * _ybar / (1 - _ybar))
    _b_corr = _f["beta"].copy()
    _b_corr[0] -= _corr
    # pesos 'balanced': numpy vs sklearn
    _w = np.where(y_dev == 1, len(y_dev) / (2 * y_dev.sum()), len(y_dev) / (2 * (len(y_dev) - y_dev.sum())))
    _fw = ajustar_logit(X_dev, y_dev, w=_w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _skw = LogisticRegression(C=np.inf, class_weight="balanced", tol=1e-12,
                                  max_iter=10_000).fit(X_dev[:, 1:], y_dev)
    pesos_np_vs_sk = (_fw["beta"], np.r_[_skw.intercept_, _skw.coef_.ravel()])
    _pm = lambda b: float(np.mean(1 / (1 + np.exp(-X_ho @ b))))
    tabla_desbalance = pd.DataFrame({
        "DEV completo": fit_np["beta"], "submuestreado": _f["beta"], "submuestreado + corrección": _b_corr,
        "pesos balanced (numpy)": _fw["beta"], "pesos balanced (sklearn)": pesos_np_vs_sk[1],
    }, index=NOMBRES)
    _pms = pd.Series({c: _pm(tabla_desbalance[c].to_numpy()) for c in tabla_desbalance.columns} |
                     {"tasa observada HO": float(y_ho.mean())}, name="PD media en HO")
    mo.vstack([tabla_desbalance.round(4), _pms.to_frame().T.style.format("{:.2%}"), mo.md(f"""
    Submuestra: {len(_idx)} créditos, tasa {_ybar:.1%}; corrección del intercepto = {_corr:.4f}. Las pendientes
    no cambian en esperanza, pero se mueven por ruido muestral: se descartó el {1 - _nb / len(_buenos):.0%} de los
    buenos, es decir, información. El intercepto sin corregir infla la PD media en HO a {_pms.iloc[1]:.1%}; con la
    corrección vuelve a {_pms.iloc[2]:.1%} (observado {_pms.iloc[-1]:.1%}). Los pesos 'balanced' conservan toda la
    muestra (pendientes más cercanas a DEV completo) pero desplazan igual el intercepto (PD media
    {_pms.iloc[3]:.1%}) y requieren la misma corrección; además $(X^\\top WX)^{{-1}}$ con pesos no es la varianza
    correcta del estimador (hace falta sándwich). En riesgo de crédito rara vez hay razón para
    balancear: la logística no "sufre" con 5% de malos; sufre con **pocos malos en absoluto**.
    """)])
    return (pesos_np_vs_sk,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 11 · Checks del módulo

    Si alguno falla, el notebook falla: coincidencia numpy vs librerías e invariantes teóricas.
    """)
    return


@app.cell
def _(beta_sk, comparacion_motores, fit_np, fit_sm, mo, no_colapsabilidad, np, pesos_np_vs_sk,
      ridge_np_vs_sk, sep_const, sep_eta_firth, sep_logit_analitico, sep_woe_curso, sim_gl,
      suma_p_vs_y, tabla_beta_uni, tabla_nb, tabla_tests, wald_lr_2x2):
    # 1. IRLS numpy = statsmodels (β y SE) y ≈ sklearn
    assert np.allclose(fit_np["beta"], fit_sm.params, atol=1e-8)
    assert np.allclose(fit_np["se"], fit_sm.bse, atol=1e-8)
    assert np.allclose(fit_np["beta"], beta_sk, atol=1e-4)
    assert np.allclose(comparacion_motores["SE numpy"], comparacion_motores["SE sklearn*"], atol=1e-4)
    # 2. condición de primer orden del intercepto: Σp̂ = Σy
    assert abs(suma_p_vs_y[0] - suma_p_vs_y[1]) < 1e-6
    # 3. tests numpy = statsmodels
    assert np.allclose(tabla_tests["Wald"], tabla_tests["Wald sm"], rtol=1e-6)
    assert np.allclose(tabla_tests["LR"], tabla_tests["LR sm"], rtol=1e-6)
    assert np.allclose(tabla_tests["score"], tabla_tests["score sm"], rtol=1e-6)
    # 4. Hauck-Donner: Wald no monótono, LR sí (m = 30..59)
    _hd = np.array([wald_lr_2x2(m)[:2] for m in range(30, 60)])
    assert np.any(np.diff(_hd[:, 0]) < 0) and np.all(np.diff(_hd[:, 1]) > 0)
    # 5. β = −1 y β0 = ln(M/B) exactos sin suavizado (numpy y statsmodels)
    assert np.allclose(tabla_beta_uni["β1 crudo"].astype(float), -1.0, atol=1e-9)
    assert np.allclose(tabla_beta_uni["β1 crudo (sm)"].astype(float), -1.0, atol=1e-8)
    assert np.allclose(tabla_beta_uni["β0 crudo"].astype(float), tabla_beta_uni["ln(M/B)"].astype(float), atol=1e-9)
    assert np.all(np.abs(tabla_beta_uni["β1 suav"].astype(float) + 1) < 0.02)
    # 6. Naive Bayes: peor calibración que la logística en HO (pendiente lejos de 1)
    assert tabla_nb.loc[("HO", "Naive Bayes"), "pendiente calib"] < 0.7
    assert tabla_nb.loc[("HO", "logística"), "log-vero media"] > tabla_nb.loc[("HO", "Naive Bayes"), "log-vero media"]
    # 7. no colapsabilidad: con γ = 2, |β1| bivariado > 1 con variables independientes
    _nc = no_colapsabilidad(2.0)
    assert _nc["β1 bivariado"] < -1.05 and abs(_nc["corr(WoE1,WoE2)"]) < 0.01
    # 8. grados de libertad escondidos: β ≈ −1 en ruido y z² ≈ IV/(1/M+1/B)
    assert abs(sim_gl["β"].mean() + 1) < 0.1
    assert abs(sim_gl["z2 / [IV/(1/M+1/B)]"].mean() - 1) < 0.1
    # 9. Firth saturado = +0,5 por celda; WoE suavizado del curso = −logit Firth + c
    assert np.allclose(sep_eta_firth, sep_logit_analitico, atol=1e-7)
    assert np.allclose(-sep_woe_curso + sep_const, sep_logit_analitico, atol=1e-12)
    # 10. ridge numpy = sklearn (λ = 1/C) y pesos numpy = sklearn
    assert np.allclose(ridge_np_vs_sk[0], ridge_np_vs_sk[1], atol=1e-4)
    assert np.allclose(pesos_np_vs_sk[0], pesos_np_vs_sk[1], atol=1e-4)
    mo.md("**Todos los checks pasaron** (IRLS = statsmodels = sklearn; Σp̂ = Σy; Wald/LR/score = statsmodels; "
          "Hauck-Donner; β = −1 exacto; Naive Bayes descalibrado; no colapsabilidad; z² ≈ IV·n·π(1−π); "
          "Firth = +0,5; ridge y pesos = sklearn).")
    return


if __name__ == "__main__":
    app.run()

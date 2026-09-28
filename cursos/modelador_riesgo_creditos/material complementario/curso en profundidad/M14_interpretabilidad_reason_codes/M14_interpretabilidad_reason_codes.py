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
#     "optbinning",
# ]
# ///
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import warnings as _warnings
    from itertools import combinations
    from math import factorial, erf, sqrt
    from scipy import stats
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    _warnings.filterwarnings("ignore")
    return combinations, erf, factorial, mo, plt, roc_auc_score, sm, sqrt, stats


@app.cell
def _(mo):
    mo.md(r"""
    # M14 · Interpretabilidad: contribuciones, monotonía, binning óptimo y reason codes

    **Serie 2 · Modelador de Riesgo de Crédito.** Notebook compañero de `M14_interpretabilidad_reason_codes.md`.

    Todo corre sobre el generador `generar_cartera()` de la serie («Banco Sintético»), que tiene **verdad
    conocida**: sabemos la PD real de cada crédito y cómo se construyó cada variable. Eso permite algo que
    con Banco Austral no se puede: comprobar si una medida de importancia, un binning o un reason code
    dice la verdad.

    Recorrido:

    1. Scorecard base (WoE del curso → logística numpy vs statsmodels → puntos PDO 20 / 600 / 50:1).
    2. Siete medidas de importancia y por qué no ordenan igual.
    3. Estabilidad de coeficientes DEV vs HO: test z y potencia con pocos malos (slider).
    4. Monotonía: diagnóstico, binning monótono en numpy (PAV + restricciones) vs `optbinning` (sliders).
    5. Valores especiales −9 / −99: el sesgo de agruparlos.
    6. Reason codes por cuatro métodos, empates, umbral, variables no accionables y estabilidad bootstrap.
    7. SHAP aditivo: en un scorecard, SHAP = puntos − E[puntos] (derivado y verificado por fuerza bruta).
    8. Edad como variable: la regla de Regulation B sobre solicitantes de 62+ años.
    9. Checks del módulo.

    Convenciones del curso: target 1 = malo; WoE = ln(%buenos/%malos) → coeficientes negativos;
    todo se ajusta en DEV y se aplica al resto.
    """)
    return


@app.cell
def _():
    # ===================== Código común de la serie =====================
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
def _():
    # optbinning es opcional: si no importa, el notebook sigue con la versión numpy.
    import io as _io
    import contextlib as _ctx
    try:
        with _ctx.redirect_stderr(_io.StringIO()), _ctx.redirect_stdout(_io.StringIO()):
            import optbinning as _ob
            from optbinning import OptimalBinning
        OPTB_OK = True
        OPTB_VERSION = _ob.__version__
    except Exception as _e:  # noqa: BLE001
        OptimalBinning = None
        OPTB_OK = False
        OPTB_VERSION = f"no disponible ({type(_e).__name__})"
    return OPTB_OK, OPTB_VERSION, OptimalBinning


@app.cell
def _(generar_cartera):
    cartera = generar_cartera()
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    ho = cartera[cartera["muestra"] == "HO"].reset_index(drop=True)
    oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)
    ttd = cartera[cartera["muestra"] == "TTD"].reset_index(drop=True)
    # 8 variables del scorecard de juguete (ver lectura abajo)
    VARS = ["uso_linea_prom_12m", "uso_tc_prom_12m", "meses_desde_mora_12m",
            "antiguedad_meses", "carga_financiera", "consultas_6m",
            "deuda_otras_prom_12m", "canal"]
    resumen_muestras = (cartera.groupby("muestra")
                        .agg(n=("id", "size"), malos=("malo", "sum"),
                             tasa_malos=("malo", "mean"), pd_verdadera_media=("pd_verdadera", "mean"))
                        .reindex(["DEV", "HO", "OOT", "TTD"]).round(4))
    resumen_muestras
    return VARS, cartera, dev, ho, oot, resumen_muestras, ttd


@app.cell
def _(coma, mo, pct, resumen_muestras):
    mo.md(f"""
    **Lectura.** DEV tiene {coma(resumen_muestras.loc['DEV','n'], 0)} créditos y una tasa de malos de
    {pct(resumen_muestras.loc['DEV','tasa_malos'])} (más alta que el 5% de Austral: el generador es una cartera
    más riesgosa, parecida a consumo no bancario). OOT sube a {pct(resumen_muestras.loc['OOT','tasa_malos'])}
    por el deterioro macro plantado en 2025. Las 8 variables incluyen tres que el embudo del curso habría
    descartado por IV < 0,10 (`antiguedad_meses`, `deuda_otras_prom_12m`, `canal`): se fuerzan a propósito
    porque en el generador **tienen efecto real** y cada una ilustra un problema de interpretabilidad
    (efecto condicional, forma no monótona, variable no accionable).
    """)
    return


@app.cell
def _(np, pd):
    # ---------------- utilidades propias del módulo ----------------
    def sigmoide(z):
        return 1.0 / (1.0 + np.exp(-z))

    def logit_irls(X, y, tol=1e-10, max_iter=100):
        """Máxima verosimilitud de la logística por Newton-Raphson (= IRLS).
        X debe incluir la columna de unos. Devuelve (beta, se, loglik)."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        beta = np.zeros(X.shape[1])
        for _it in range(max_iter):
            p = sigmoide(X @ beta)
            w = p * (1 - p)
            H = X.T @ (X * w[:, None])          # información de Fisher = -Hessiano
            g = X.T @ (y - p)                   # gradiente (score)
            paso = np.linalg.solve(H, g)
            beta = beta + paso
            if np.max(np.abs(paso)) < tol:
                break
        p = sigmoide(X @ beta)
        cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
        return beta, np.sqrt(np.diag(cov)), ll

    def auc_np(y, s):
        """AUC por Mann-Whitney con rangos promedio (empates) — sin scipy ni sklearn."""
        y = np.asarray(y, float)
        s = np.asarray(s, float)
        _u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        rango_unico = np.cumsum(cnt) - (cnt - 1) / 2.0
        r = rango_unico[inv]
        n1 = y.sum()
        n0 = len(y) - n1
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def gini_np(y, s):
        return 2 * auc_np(y, s) - 1

    def phi_normal(z):
        """CDF normal estándar con numpy (aprox. de Abramowitz-Stegun 7.1.26 no basta: usamos erf vectorizada)."""
        from math import erf as _erf
        return 0.5 * (1 + np.vectorize(_erf)(np.asarray(z, float) / np.sqrt(2)))

    def woe_de_etiquetas(etq, y, orden=None):
        """Tabla WoE desde etiquetas de bin, con el MISMO suavizado +0,5 de tabla_woe del curso."""
        tab = (pd.DataFrame({"bin": np.asarray(etq), "y": np.asarray(y, float)})
               .groupby("bin")["y"].agg(["count", "sum"]))
        tab.columns = ["n", "malos"]
        if orden is not None:
            tab = tab.reindex([o for o in orden if o in tab.index])
        tab["buenos"] = tab["n"] - tab["malos"]
        tab["tasa_malos"] = tab["malos"] / tab["n"]
        pm = (tab["malos"] + 0.5) / (tab["malos"].sum() + 0.5 * len(tab))
        pb = (tab["buenos"] + 0.5) / (tab["buenos"].sum() + 0.5 * len(tab))
        tab["woe"] = np.log(pb / pm)
        tab["iv_aporte"] = (pb - pm) * tab["woe"]
        return tab, float(tab["iv_aporte"].sum())

    def iv_sin_suavizar(etq, y):
        """IV sin suavizado (la convención de optbinning)."""
        tab = pd.DataFrame({"bin": np.asarray(etq), "y": np.asarray(y, float)}).groupby("bin")["y"].agg(["count", "sum"])
        m = tab["sum"].values
        b = (tab["count"] - tab["sum"]).values
        pm, pb = m / m.sum(), b / b.sum()
        ok = (pm > 0) & (pb > 0)
        return float(np.sum((pb[ok] - pm[ok]) * np.log(pb[ok] / pm[ok])))

    def etiquetar(x, cortes, especiales=()):
        """Bins [c_{k-1}, c_k) (cerrados a la izquierda, como optbinning). Especiales y NaN aparte."""
        x = np.asarray(x, float)
        cortes = np.asarray(cortes, float)
        bordes = np.concatenate([[-np.inf], cortes, [np.inf]])
        nombres = [f"[{bordes[k]:.4g}, {bordes[k + 1]:.4g})" for k in range(len(bordes) - 1)]
        k = np.searchsorted(cortes, x, side="right")
        etq = np.array(nombres, dtype=object)[np.clip(k, 0, len(nombres) - 1)]
        for c in especiales:
            etq = np.where(x == c, f"ESP {c:g}", etq)
        etq = np.where(np.isnan(x), "MISSING", etq)
        orden = [f"ESP {c:g}" for c in especiales] + nombres + ["MISSING"]
        return etq, orden

    def clave_orden(etiqueta):
        """Posición numérica de una etiqueta del binner del curso ('= v', '(a, b]'); None si no es numérica."""
        s = str(etiqueta)
        if s.startswith("= "):
            return float(s[2:])
        if s.startswith("(") or s.startswith("["):
            a = s[1:].split(",")[0].strip()
            return -1e18 if a == "-inf" else float(a)
        return None

    def quiebres_monotonia(valores):
        """Número de cambios de signo en la secuencia (ignorando diferencias < 1e-9)."""
        d = np.diff(np.asarray(valores, float))
        sg = np.sign(d[np.abs(d) > 1e-9])
        return int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0

    def coma(x, d=2):
        return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def pct(x, d=1):
        return coma(100 * x, d) + "%"

    return (auc_np, clave_orden, coma, etiquetar, gini_np, iv_sin_suavizar, logit_irls,
            pct, phi_normal, quiebres_monotonia, sigmoide, woe_de_etiquetas)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Scorecard base: WoE → logística → puntos

    **Qué mirar.** Dos implementaciones de la máxima verosimilitud (Newton-Raphson en numpy y
    `statsmodels.Logit`) deben dar los mismos $\beta$, errores estándar y log-verosimilitud. Luego el
    scaling del curso: $\text{factor}=20/\ln 2$, $\text{offset}=600-\text{factor}\cdot\ln 50$ y
    $\text{puntos}(v,b) = -(\beta_v\,\text{WoE}_{v,b} + \beta_0/n)\cdot\text{factor} + \text{offset}/n$.
    """)
    return


@app.cell
def _(VARS, a_woe, dev, ho, oot, pd, tabla_woe, ttd):
    tablas_woe = {}
    iv_dev = {}
    for _v in VARS:
        tablas_woe[_v], iv_dev[_v] = tabla_woe(dev[_v], dev["malo"])
    iv_dev = pd.Series(iv_dev)
    mapas = {_v: tablas_woe[_v]["woe"] for _v in VARS}
    W_dev = a_woe(dev, VARS, dev, mapas)
    W_ho = a_woe(ho, VARS, dev, mapas)
    W_oot = a_woe(oot, VARS, dev, mapas)
    W_ttd = a_woe(ttd, VARS, dev, mapas)
    return W_dev, W_ho, W_oot, W_ttd, iv_dev, mapas, tablas_woe


@app.cell
def _(VARS, W_dev, dev, logit_irls, np, pd, sm):
    y_dev = dev["malo"].values
    X_dev = np.column_stack([np.ones(len(W_dev)), W_dev[VARS].values])
    beta_np, se_np, ll_np = logit_irls(X_dev, y_dev)
    modelo_sm = sm.Logit(y_dev, sm.add_constant(W_dev[VARS])).fit(disp=0)
    beta = pd.Series(beta_np, index=["const"] + VARS)
    se_beta = pd.Series(se_np, index=["const"] + VARS)
    comparacion_ajuste = pd.DataFrame({
        "beta_numpy": beta_np, "beta_statsmodels": modelo_sm.params.values,
        "se_numpy": se_np, "se_statsmodels": modelo_sm.bse.values,
        "p_valor": modelo_sm.pvalues.values}, index=["const"] + VARS).round(4)
    comparacion_ajuste
    return X_dev, beta, comparacion_ajuste, ll_np, modelo_sm, se_beta, y_dev


@app.cell
def _(VARS, W_dev, W_ho, W_oot, W_ttd, beta, mapas, np, pd):
    PDO, SCORE_BASE, ODDS_BASE = 20, 600, 50
    FACTOR = PDO / np.log(2)
    OFFSET = SCORE_BASE - FACTOR * np.log(ODDS_BASE)
    N_VARS = len(VARS)

    def puntos_de(W, b=beta):
        """Puntos por variable (una columna por variable) con reparto del intercepto en partes iguales."""
        return pd.DataFrame({v: -(b[v] * W[v].values + b["const"] / N_VARS) * FACTOR + OFFSET / N_VARS
                             for v in VARS})

    scorecard = pd.DataFrame([
        {"variable": _v, "bin": _b, "woe": _w,
         "puntos": -(beta[_v] * _w + beta["const"] / N_VARS) * FACTOR + OFFSET / N_VARS}
        for _v in VARS for _b, _w in mapas[_v].items()])
    pts_dev, pts_ho, pts_oot, pts_ttd = (puntos_de(_W) for _W in (W_dev, W_ho, W_oot, W_ttd))
    score_dev, score_ho, score_oot, score_ttd = (_p.sum(axis=1).values for _p in (pts_dev, pts_ho, pts_oot, pts_ttd))
    PUNTOS_NEUTROS = -(beta["const"] / N_VARS) * FACTOR + OFFSET / N_VARS   # puntos de un bin con WoE = 0
    return (FACTOR, N_VARS, OFFSET, PUNTOS_NEUTROS, pts_dev, pts_ho, pts_oot, pts_ttd,
            puntos_de, score_dev, score_ho, score_oot, score_ttd, scorecard)


@app.cell
def _(FACTOR, OFFSET, VARS, W_dev, W_ho, W_oot, beta, dev, gini_np, ho, modelo_sm, np, oot, pd,
      roc_auc_score, score_dev, sigmoide, sm):
    # score = suma de puntos = transformación del logit (invariante del scaling)
    _pd_dev = sigmoide(beta["const"] + W_dev[VARS].values @ beta[VARS].values)
    _score_logit = OFFSET + FACTOR * np.log((1 - _pd_dev) / _pd_dev)
    assert np.allclose(score_dev, _score_logit), "score ≠ offset + factor·ln(odds)"
    gini_base = {}
    for _n, _d, _W in [("DEV", dev, W_dev), ("HO", ho, W_ho), ("OOT", oot, W_oot)]:
        _p = modelo_sm.predict(sm.add_constant(_W[VARS], has_constant="add"))
        _g_np = gini_np(_d["malo"].values, _p)
        _g_sk = 2 * roc_auc_score(_d["malo"].values, _p) - 1
        assert np.isclose(_g_np, _g_sk)
        gini_base[_n] = _g_np
    gini_verdad = {_n: gini_np(_d["malo"].values, _d["pd_verdadera"].values)
                   for _n, _d in [("DEV", dev), ("HO", ho), ("OOT", oot)]}
    tabla_gini_base = pd.DataFrame({"gini_modelo": gini_base, "gini_pd_verdadera": gini_verdad}).round(3)
    tabla_gini_base
    return gini_base, tabla_gini_base


@app.cell
def _(coma, mo, np, score_dev, tabla_gini_base):
    mo.md(f"""
    **Lectura.** numpy y statsmodels coinciden (assert). El score cumple la identidad
    *score = Σ puntos = offset + factor·ln(odds buenos)* para los {coma(len(score_dev), 0)} créditos de DEV.
    Gini DEV / HO / OOT = {coma(tabla_gini_base.loc['DEV','gini_modelo'],3)} /
    {coma(tabla_gini_base.loc['HO','gini_modelo'],3)} / {coma(tabla_gini_base.loc['OOT','gini_modelo'],3)};
    el techo (Gini de la PD verdadera) es {coma(tabla_gini_base.loc['HO','gini_pd_verdadera'],3)} en HO: el
    generador tiene ruido no observable (σ = 0,35 en el log-odds), así que ningún modelo llega a 1.
    Score DEV: p5 {np.percentile(score_dev,5):.0f} · mediana {np.median(score_dev):.0f} · p95 {np.percentile(score_dev,95):.0f}
    (más bajo que Austral —p5 524 · mediana 612 · p95 705— porque las odds de esta cartera son ~8:1, no ~19:1).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Importancia de variables: siete medidas que no ordenan igual

    **Qué mirar.** Para cada variable: $\beta$, IV univariado, **rango de puntos** (curso), aporte
    $|\beta|\,\sigma(\text{WoE})$ y $|\beta|\cdot\text{IV}$ (curso, normalizados a 100%), **SHAP medio**
    $\text{factor}\cdot|\beta|\,E|\text{WoE}-E\,\text{WoE}|$, y dos medidas *drop-column*: caída de
    log-verosimilitud en DEV al re-ajustar sin la variable (con su test de razón de verosimilitud) y caída
    de Gini en HO. El drop-column se calcula dos veces (numpy y statsmodels).
    """)
    return


@app.cell
def _(FACTOR, VARS, W_dev, W_ho, X_dev, beta, gini_base, gini_np, ho, iv_dev, ll_np, logit_irls,
      np, pd, scorecard, sigmoide, sm, stats, y_dev):
    _filas = []
    for _j, _v in enumerate(VARS):
        _w = W_dev[_v].values
        _cols = [0] + [k + 1 for k in range(len(VARS)) if k != _j]
        _b_red, _se_red, _ll_red = logit_irls(X_dev[:, _cols], y_dev)
        _sm_red = sm.Logit(y_dev, X_dev[:, _cols]).fit(disp=0)
        assert np.isclose(_ll_red, _sm_red.llf, atol=1e-6), _v
        _otras = [u for u in VARS if u != _v]
        _p_ho_red = sigmoide(_b_red[0] + W_ho[_otras].values @ _b_red[1:])
        _pts_v = scorecard.loc[scorecard["variable"] == _v, "puntos"]
        _filas.append({
            "variable": _v, "beta": beta[_v], "iv": iv_dev[_v],
            "rango_pts": _pts_v.max() - _pts_v.min(),
            "b_sigma_woe": abs(beta[_v]) * _w.std(ddof=0),
            "b_iv": abs(beta[_v]) * iv_dev[_v],
            "shap_medio_pts": FACTOR * abs(beta[_v]) * np.mean(np.abs(_w - _w.mean())),
            "delta_ll_dev": ll_np - _ll_red,
            "p_lr": stats.chi2.sf(2 * (ll_np - _ll_red), df=1),
            "delta_gini_ho": gini_base["HO"] - gini_np(ho["malo"].values, _p_ho_red),
        })
    importancia = pd.DataFrame(_filas).set_index("variable")
    for _c in ["b_sigma_woe", "b_iv", "shap_medio_pts", "delta_ll_dev"]:
        importancia[_c + "_%"] = 100 * importancia[_c] / importancia[_c].sum()
    importancia = importancia.sort_values("rango_pts", ascending=False)
    importancia.round(3)
    return (importancia,)


@app.cell
def _(importancia, np, plt):
    _medidas = {"rango de puntos": importancia["rango_pts"] / importancia["rango_pts"].sum() * 100,
                "|β|·σ(WoE)": importancia["b_sigma_woe_%"], "|β|·IV": importancia["b_iv_%"],
                "SHAP medio": importancia["shap_medio_pts_%"], "Δ log-verosimilitud": importancia["delta_ll_dev_%"]}
    _fig, _ax = plt.subplots(figsize=(8.5, 4.2))
    _y = np.arange(len(importancia))
    _h = 0.16
    for _k, (_nom, _s) in enumerate(_medidas.items()):
        _ax.barh(_y + (_k - 2) * _h, _s.values, height=_h, label=_nom)
    _ax.set_yticks(_y)
    _ax.set_yticklabels(importancia.index)
    _ax.invert_yaxis()
    _ax.set_xlabel("participación en el total de la medida (%)")
    _ax.set_title("Importancia por variable: cinco medidas normalizadas a 100%")
    _ax.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(coma, importancia, mo, pct):
    _r_rango = importancia["rango_pts"].rank(ascending=False)
    _r_ll = importancia["delta_ll_dev"].rank(ascending=False)
    _peor = (_r_rango - _r_ll).abs().idxmax()
    mo.md(f"""
    **Lectura.** El rango de puntos pone primero a `{importancia.index[0]}`; la caída de log-verosimilitud
    (la medida con interpretación estadística directa) pone primero a `{_r_ll.idxmin()}`. La mayor
    discrepancia de ranking entre rango y ΔLL está en `{_peor}` (puesto {int(_r_rango[_peor])} por rango vs
    {int(_r_ll[_peor])} por ΔLL): el rango mide lo que la variable **puede** mover (mejor vs peor bin, aunque
    el peor bin tenga 3% de la población), ΔLL mide lo que **aporta dado el resto**. `deuda_otras_prom_12m`
    tiene IV univariado casi nulo ({coma(importancia.loc['deuda_otras_prom_12m','iv'], 3)}) pero su test LR da
    p = {coma(importancia.loc['deuda_otras_prom_12m','p_lr'], 3)}: su efecto es condicional (en el generador la
    deuda externa entra en la carga y, sobre 5 millones de pesos, *baja* el riesgo: el cliente bancarizado, como el bin «sobre
    5,71 M» de Austral). Al revés, `uso_tc_prom_12m` tiene IV {coma(importancia.loc['uso_tc_prom_12m','iv'], 2)}
    pero aporta {coma(importancia.loc['uso_tc_prom_12m','delta_ll_dev_%'], 1)}% de la ΔLL total: casi todo lo que
    sabe ya lo dice `uso_linea_prom_12m`. Rangos de aporte
    |β|·σ(WoE): de {pct(importancia['b_sigma_woe_%'].min()/100)} a {pct(importancia['b_sigma_woe_%'].max()/100)}.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Estabilidad de coeficientes DEV vs HO

    **Qué mirar.** Re-ajustamos el mismo modelo (WoE de DEV) en HO y comparamos con
    $z=(\hat\beta_{DEV}-\hat\beta_{HO})/\sqrt{se_{DEV}^2+se_{HO}^2}$. Distinguimos **inversión aparente**
    (el signo de $\hat\beta_{HO}$ cambia) de **inversión significativa** ($\hat\beta_{HO}$ con signo opuesto
    *y* distinto de cero al 5%). Como HO sale de los mismos meses que DEV, aquí la verdad es «no hay ninguna
    inversión real»: todo lo que aparezca es ruido de muestra. El slider reduce los malos de HO para
    reproducir la situación de Austral (78 malos para 9 parámetros).
    """)
    return


@app.cell
def _(VARS, W_ho, beta, ho, logit_irls, np, pd, phi_normal, se_beta, stats):
    X_ho = np.column_stack([np.ones(len(W_ho)), W_ho[VARS].values])
    y_ho = ho["malo"].values
    beta_ho_np, se_ho_np, _ll = logit_irls(X_ho, y_ho)
    _z = (beta.values - beta_ho_np) / np.sqrt(se_beta.values ** 2 + se_ho_np ** 2)
    estabilidad = pd.DataFrame({
        "beta_dev": beta.values, "se_dev": se_beta.values, "beta_ho": beta_ho_np, "se_ho": se_ho_np,
        "z_diferencia": _z,
        "p_diferencia_numpy": 2 * (1 - phi_normal(np.abs(_z))),
        "p_diferencia_scipy": 2 * stats.norm.sf(np.abs(_z)),
        "p_beta_ho_cero": 2 * stats.norm.sf(np.abs(beta_ho_np / se_ho_np)),
    }, index=["const"] + VARS)
    estabilidad["inversion_aparente"] = np.sign(estabilidad["beta_ho"]) != np.sign(estabilidad["beta_dev"])
    estabilidad["inversion_significativa"] = estabilidad["inversion_aparente"] & (estabilidad["p_beta_ho_cero"] < 0.05)
    estabilidad.round(3)
    return X_ho, estabilidad, se_ho_np, y_ho


@app.cell
def _(mo, y_ho):
    malos_ho = mo.ui.slider(40, int(y_ho.sum()), value=78, step=2, label="Malos disponibles en HO (78 = Austral)")
    malos_ho
    return (malos_ho,)


@app.cell
def _(VARS, X_ho, beta, logit_irls, malos_ho, np, pd, phi_normal, se_ho_np, stats, y_ho):
    # Monte Carlo: submuestras de HO con ~k malos. Verdad: NO hay inversión real.
    _rng = np.random.default_rng(14)
    _k = malos_ho.value
    _tasa = y_ho.mean()
    _n_sub = int(round(_k / _tasa))
    _R = 200
    _flip, _flip_sig, _zdiff = [], [], []
    for _r in range(_R):
        _idx = _rng.choice(len(y_ho), size=_n_sub, replace=False)
        try:
            _b, _se, _ = logit_irls(X_ho[_idx], y_ho[_idx])
        except np.linalg.LinAlgError:
            continue
        _flip.append(np.sign(_b[1:]) != np.sign(beta.values[1:]))
        _flip_sig.append((np.sign(_b[1:]) != np.sign(beta.values[1:])) & (np.abs(_b[1:] / _se[1:]) > 1.96))
        _zdiff.append(np.abs(_b[1:] - beta.values[1:]) / _se[1:] > 1.96)   # contraste con β_DEV fijo
    # aproximación analítica: se escala con 1/sqrt(n) → P(flip) ≈ Φ(-|β_DEV| / se_k)
    _se_k = se_ho_np[1:] * np.sqrt(len(y_ho) / _n_sub)
    potencia = pd.DataFrame({
        "beta_dev": beta.values[1:],
        "se_esperado_con_k_malos": _se_k,
        "P_inversion_aparente_sim": np.mean(_flip, axis=0),
        "P_inversion_aparente_analitica": phi_normal(-np.abs(beta.values[1:]) / _se_k),
        "P_inversion_significativa_sim": np.mean(_flip_sig, axis=0),
        "P_rechazo_falso_z_sim": np.mean(_zdiff, axis=0),
        "potencia_detectar_inversion_total": stats.norm.sf(1.96 - 2 * np.abs(beta.values[1:]) / _se_k),
    }, index=VARS)
    potencia_meta = {"k": _k, "n_sub": _n_sub, "R": len(_flip)}
    potencia.round(3)
    return potencia, potencia_meta


@app.cell
def _(coma, estabilidad, mo, pct, potencia, potencia_meta):
    _debil = potencia["P_inversion_aparente_sim"].idxmax()
    _inv = list(estabilidad.index[estabilidad["inversion_aparente"]])
    _zmax = estabilidad["z_diferencia"].abs().idxmax()
    mo.md(f"""
    **Lectura.** Con HO completo, inversiones aparentes: {", ".join(f"`{v}`" for v in _inv) if _inv else "ninguna"};
    inversiones significativas: {int(estabilidad["inversion_significativa"].sum())}; mayor |z| de diferencia:
    `{_zmax}` ({coma(estabilidad.loc[_zmax, 'z_diferencia'], 2)}). Con **{potencia_meta['k']} malos**
    (≈ {coma(potencia_meta['n_sub'], 0)} créditos, {potencia_meta['R']} réplicas) la variable más frágil es `{_debil}`:
    su signo se invierte en {pct(potencia.loc[_debil,'P_inversion_aparente_sim'])} de las submuestras
    (analítico ≈ {pct(potencia.loc[_debil,'P_inversion_aparente_analitica'])}) **sin que exista ninguna inversión
    real**. Las inversiones *significativas* quedan cerca o bajo el 2,5% nominal (una cola del test al 5%): el
    criterio del curso —solo descalifica una inversión significativa— es el correcto. La última columna es la
    potencia para detectar una inversión *completa* (β_HO = −β_DEV): con pocos malos, para las variables
    débiles es baja; «no detecté inversión» no es evidencia de estabilidad.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Monotonía

    ### 4.1 Diagnóstico sobre el binning del curso

    **Qué mirar.** Ordenamos los bins de cada variable por su posición numérica (el binner del curso pone la
    moda primero) y contamos cambios de signo de la secuencia de WoE. Un quiebre de 0,02 de WoE es ruido;
    uno de 0,5 es una historia que habrá que explicar en el comité (o fusionar).
    """)
    return


@app.cell
def _(FACTOR, VARS, beta, clave_orden, np, pd, quiebres_monotonia, tablas_woe):
    _filas = []
    for _v in VARS:
        _t = tablas_woe[_v]
        _claves = [(clave_orden(b), b) for b in _t.index]
        _num = sorted([c for c in _claves if c[0] is not None])
        if len(_num) < 3:
            _filas.append({"variable": _v, "n_bins": len(_t), "secuencia_woe": "(categórica)",
                           "quiebres": np.nan, "max_violacion_woe": np.nan, "max_violacion_pts": np.nan})
            continue
        _w = _t.loc[[b for _, b in _num], "woe"].values
        _d = np.diff(_w)
        _dir = np.sign(np.sum(_d))
        _viol = np.max(np.clip(-_dir * _d, 0, None)) if _dir != 0 else 0.0
        _filas.append({"variable": _v, "n_bins": len(_t),
                       "secuencia_woe": " → ".join(f"{x:+.2f}" for x in _w),
                       "quiebres": quiebres_monotonia(_w), "max_violacion_woe": _viol,
                       "max_violacion_pts": _viol * abs(beta[_v]) * FACTOR})
    diagnostico_monotonia = pd.DataFrame(_filas).set_index("variable")
    diagnostico_monotonia.round(3)
    return (diagnostico_monotonia,)


@app.cell
def _(coma, diagnostico_monotonia, mo, pct, tablas_woe):
    _t = tablas_woe["meses_desde_mora_12m"]
    _b1 = [b for b in _t.index if str(b).startswith("(-inf")][0]
    _b2 = [b for b in _t.index if str(b).startswith("(-9")][0]
    mo.md(f"""
    **Lectura.** `meses_desde_mora_12m` muestra el caso del curso: en orden numérico el bin `{_b1}`
    (tasa {pct(_t.loc[_b1,'tasa_malos'])}) va **antes** que la mora reciente `{_b2}`
    ({pct(_t.loc[_b2,'tasa_malos'])}): el binner trata los códigos −9 / −99 como números, y eso fabrica
    un quiebre que no existe en la relación económica. `deuda_otras_prom_12m` tiene
    {int(diagnostico_monotonia.loc['deuda_otras_prom_12m','quiebres'])} quiebres: ahí el efecto real *no es*
    monótono (máx. {coma(diagnostico_monotonia.loc['deuda_otras_prom_12m','max_violacion_pts'], 1)} puntos de
    violación). `antiguedad_meses` tiene {int(diagnostico_monotonia.loc['antiguedad_meses','quiebres'])} «quiebres» por
    {coma(diagnostico_monotonia.loc['antiguedad_meses','max_violacion_pts'], 2)} puntos: dos bins empatados, ruido; se
    fusionan (como el «serrucho» de 1,4 puntos de carga_financiera en Austral). Contar quiebres sin medir su
    tamaño en puntos confunde ambos casos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.2 Binning monótono: numpy (PAV + restricciones) vs `optbinning` (programación entera)

    **Algoritmo numpy.** (1) pre-bins por cuantiles de DEV (máx. 20), con los códigos especiales fuera;
    (2) *pool-adjacent-violators* sobre la tasa de malos ponderada por $n$ (para *peak/valley*: se prueba cada
    posición del vértice y se queda la de mayor IV); (3) se fusiona el bin más chico con su vecino más parecido
    hasta cumplir `min_bin_size`; (4) mientras haya pares adyacentes con diferencia de tasa < `min_event_rate_diff`
    o más de `max_n_bins` bins, se fusiona el par adyacente **que menos IV pierde**. Fusionar vecinos de una
    secuencia monótona la mantiene monótona, así que (3)–(4) no rompen lo que hizo (2).

    **optbinning.** Recibe los **mismos pre-bins** (`user_splits`) y resuelve el problema exacto (maximizar IV
    sujeto a las mismas restricciones) con un solver CP/MIP. Si la solución numpy es factible, el IV de
    optbinning debe ser ≥ (es óptimo); la heurística solo puede empatar o perder.
    """)
    return


@app.cell
def _(mo):
    var_bin = mo.ui.dropdown(options=["uso_linea_prom_12m", "carga_financiera", "antiguedad_meses",
                                      "meses_desde_mora_12m", "deuda_otras_prom_12m"],
                             value="uso_linea_prom_12m", label="Variable")
    tendencia = mo.ui.dropdown(options=["auto_asc_desc", "ascending", "descending", "peak", "valley"],
                               value="auto_asc_desc", label="monotonic_trend (tasa de malos)")
    max_n_bins = mo.ui.slider(2, 10, value=5, label="max_n_bins")
    min_event_rate_diff = mo.ui.slider(0.0, 0.05, value=0.01, step=0.0025, label="min_event_rate_diff")
    min_bin_size = mo.ui.slider(0.01, 0.20, value=0.05, step=0.01, label="min_bin_size (fracción)")
    mo.vstack([mo.hstack([var_bin, tendencia]), max_n_bins, min_event_rate_diff, min_bin_size])
    return max_n_bins, min_bin_size, min_event_rate_diff, tendencia, var_bin


@app.cell
def _(np):
    def _pav(tasa, peso, creciente=True):
        """Pool-adjacent-violators. Devuelve lista de bloques (listas de índices consecutivos)."""
        v = list(np.asarray(tasa, float) * (1 if creciente else -1))
        w = list(np.asarray(peso, float))
        bloques = [[i] for i in range(len(v))]
        i = 0
        while i < len(v) - 1:
            if v[i] > v[i + 1] + 1e-15:          # violación: se agrupan y se promedia ponderado
                v[i] = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
                w[i] += w[i + 1]
                bloques[i] = bloques[i] + bloques[i + 1]
                del v[i + 1], w[i + 1], bloques[i + 1]
                i = max(i - 1, 0)                # el nuevo bloque puede violar hacia atrás
            else:
                i += 1
        return bloques

    def binning_monotono(x, y, tendencia="ascending", max_n_bins=None, min_event_rate_diff=0.0,
                         min_bin_size=0.0, especiales=(), max_n_prebins=20):
        """Binning monótono heurístico. Devuelve (cortes, tendencia_usada, prebins)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        N = len(x)
        ok = ~np.isin(x, especiales) & ~np.isnan(x)
        pre = np.unique(np.quantile(x[ok], np.linspace(0, 1, max_n_prebins + 1))[1:-1])
        idx = np.searchsorted(pre, x[ok], side="right")
        n = np.bincount(idx, minlength=len(pre) + 1).astype(float)
        e = np.bincount(idx, weights=y[ok], minlength=len(pre) + 1)
        E, G = y.sum(), N - y.sum()

        def st(g):
            nn, ee = n[g].sum(), e[g].sum()
            return nn, ee, ee / nn

        def iv_de(gs):
            s = 0.0
            for g in gs:
                nn, ee, _t = st(g)
                pb, pm = (nn - ee) / G, ee / E
                if pb > 0 and pm > 0:
                    s += (pb - pm) * np.log(pb / pm)
            return s

        def ajustar(tend):
            gs = [[i] for i in range(len(n)) if n[i] > 0]
            r = np.array([st(g)[2] for g in gs])
            w = np.array([st(g)[0] for g in gs])
            if tend in ("ascending", "descending"):
                bl = _pav(r, w, creciente=(tend == "ascending"))
            else:                                  # peak / valley: probar cada vértice
                mejor, bl = -np.inf, None
                for m in range(len(gs)):
                    izq = _pav(r[:m + 1], w[:m + 1], creciente=(tend == "peak"))
                    der = _pav(r[m + 1:], w[m + 1:], creciente=(tend != "peak"))
                    cand = izq + [[j + m + 1 for j in b] for b in der]
                    val = iv_de([sum((gs[j] for j in b), []) for b in cand])
                    if val > mejor:
                        mejor, bl = val, cand
            gs = [sum((gs[j] for j in b), []) for b in bl]
            # (3) tamaño mínimo
            while len(gs) > 1:
                tam = [st(g)[0] for g in gs]
                j = int(np.argmin(tam))
                if tam[j] >= min_bin_size * N:
                    break
                if j == 0:
                    k = 0
                elif j == len(gs) - 1:
                    k = j - 1
                else:
                    rr = [st(g)[2] for g in gs]
                    k = j - 1 if abs(rr[j] - rr[j - 1]) <= abs(rr[j] - rr[j + 1]) else j
                gs = gs[:k] + [gs[k] + gs[k + 1]] + gs[k + 2:]
            # (4) separación mínima y máximo de bins: fusionar el par de menor pérdida de IV
            while len(gs) > 1:
                rr = np.array([st(g)[2] for g in gs])
                viola = np.abs(np.diff(rr)) < min_event_rate_diff
                sobra = max_n_bins is not None and len(gs) > max_n_bins
                if not viola.any() and not sobra:
                    break
                cand = np.where(viola)[0] if viola.any() else np.arange(len(gs) - 1)
                base = iv_de(gs)
                perd = [base - iv_de(gs[:i] + [gs[i] + gs[i + 1]] + gs[i + 2:]) for i in cand]
                i = int(cand[int(np.argmin(perd))])
                gs = gs[:i] + [gs[i] + gs[i + 1]] + gs[i + 2:]
            return np.array([pre[g[-1]] for g in gs[:-1]]), iv_de(gs)

        if tendencia == "auto_asc_desc":
            (c1, iv1), (c2, iv2) = ajustar("ascending"), ajustar("descending")
            cortes, usada = (c1, "ascending") if iv1 >= iv2 else (c2, "descending")
        else:
            cortes, _iv = ajustar(tendencia)
            usada = tendencia
        return cortes, usada, pre

    return (binning_monotono,)


@app.cell
def _(OPTB_OK, OptimalBinning, binning_monotono, dev, etiquetar, gini_np, ho, iv_sin_suavizar, max_n_bins,
      min_bin_size, min_event_rate_diff, np, pd, tablas_woe, tendencia, var_bin, woe_de_etiquetas, binear):
    _v = var_bin.value
    _esp = (-9.0, -99.0) if _v == "meses_desde_mora_12m" else ()
    _x, _y = dev[_v].values, dev["malo"].values
    cortes_np, tend_np, prebins = binning_monotono(
        _x, _y, tendencia=tendencia.value, max_n_bins=max_n_bins.value,
        min_event_rate_diff=min_event_rate_diff.value, min_bin_size=min_bin_size.value, especiales=_esp)
    cortes_ob, estado_ob, iv_ob_libreria = None, "optbinning no disponible", np.nan
    if OPTB_OK:
        try:
            _ob = OptimalBinning(name=_v, dtype="numerical", monotonic_trend=tendencia.value,
                                 max_n_bins=max_n_bins.value, min_event_rate_diff=min_event_rate_diff.value,
                                 min_bin_size=min_bin_size.value, user_splits=prebins,
                                 special_codes=list(_esp) if _esp else None, time_limit=20)
            _ob.fit(_x, _y.astype(int))
            cortes_ob, estado_ob = np.asarray(_ob.splits, float), _ob.status
            iv_ob_libreria = float(_ob.binning_table.build().loc["Totals", "IV"])
        except Exception as _e:  # noqa: BLE001
            estado_ob = f"falló: {type(_e).__name__}"

    def _evaluar(etq_dev, etq_ho, orden, nombre):
        _t, _iv = woe_de_etiquetas(etq_dev, _y, orden)
        _mapa = _t["woe"]
        _s_ho = -pd.Series(etq_ho).map(_mapa).fillna(0.0).values      # WoE alto = bueno → score de riesgo = −WoE
        _tasas = _t.loc[[b for b in _t.index if not str(b).startswith(("ESP", "MISSING", "= "))], "tasa_malos"].values
        _d = np.diff(_tasas)
        return {"método": nombre, "n_bins": len(_t), "iv_dev": _iv, "iv_sin_suavizar": iv_sin_suavizar(etq_dev, _y),
                "gini_ho_univariado": gini_np(ho["malo"].values, _s_ho),
                "monótona": bool(np.all(_d >= -1e-12) or np.all(_d <= 1e-12)),
                "min_bin_%": _t["n"].min() / len(_y), "min_dif_tasa": np.min(np.abs(_d)) if len(_d) else np.nan}

    _e_dev, _ord = etiquetar(_x, cortes_np, _esp)
    _e_ho, _ = etiquetar(ho[_v].values, cortes_np, _esp)
    _filas = [_evaluar(_e_dev, _e_ho, _ord, f"numpy PAV ({tend_np})")]
    if cortes_ob is not None:
        _e_dev2, _ord2 = etiquetar(_x, cortes_ob, _esp)
        _e_ho2, _ = etiquetar(ho[_v].values, cortes_ob, _esp)
        _filas.append(_evaluar(_e_dev2, _e_ho2, _ord2, f"optbinning ({estado_ob})"))
    _c_dev, _c_ord = binear(dev[_v])
    _c_ho, _ = binear(ho[_v], ref=dev[_v])
    _filas.append(_evaluar(_c_dev.values, _c_ho.values, _c_ord, "binner del curso (5 cuantiles)"))
    comparacion_binning = pd.DataFrame(_filas).set_index("método")
    tabla_bins_np = woe_de_etiquetas(_e_dev, _y, _ord)[0]
    comparacion_binning.round(4)
    return comparacion_binning, cortes_np, cortes_ob, estado_ob, iv_ob_libreria, prebins, tabla_bins_np


@app.cell
def _(cortes_np, cortes_ob, dev, etiquetar, np, plt, prebins, var_bin, woe_de_etiquetas):
    _v = var_bin.value
    _esp = (-9.0, -99.0) if _v == "meses_desde_mora_12m" else ()
    _x, _y = dev[_v].values, dev["malo"].values
    _ok = ~np.isin(_x, _esp) & ~np.isnan(_x)
    _lo, _hi = np.quantile(_x[_ok], [0.005, 0.995])
    _fig, _ax = plt.subplots(figsize=(8.5, 3.8))
    _e, _o = etiquetar(_x[_ok], prebins)
    _t = woe_de_etiquetas(_e, _y[_ok], _o)[0]
    _bordes = np.concatenate([[_lo], prebins, [_hi]])
    _mid = (_bordes[:-1] + _bordes[1:]) / 2
    _ax.scatter(_mid[:len(_t)], _t["tasa_malos"].values, s=_t["n"].values / 15, color="0.6",
                label="pre-bins (tamaño ∝ n)", zorder=3)
    for _cortes, _col, _nom, _ls in [(cortes_np, "C0", "numpy PAV", "-"), (cortes_ob, "C1", "optbinning", "--")]:
        if _cortes is None:
            continue
        _e2, _o2 = etiquetar(_x[_ok], _cortes)
        _t2 = woe_de_etiquetas(_e2, _y[_ok], _o2)[0]
        _b2 = np.clip(np.concatenate([[_lo], _cortes, [_hi]]), _lo, _hi)
        for _k, _r in enumerate(_t2["tasa_malos"].values):
            _ax.hlines(_r, _b2[_k], _b2[_k + 1], color=_col, lw=2.2, ls=_ls, label=_nom if _k == 0 else None)
    _ax.set_xlabel(f"{_v} (unidades de la variable; especiales fuera)")
    _ax.set_ylabel("tasa de malos en DEV")
    _ax.set_title(f"Binning monótono de {_v}: pre-bins y solución")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(coma, comparacion_binning, mo, var_bin):
    _c = comparacion_binning
    _fil = list(_c.index)
    _txt_ob = ""
    if len(_fil) == 3:
        _txt_ob = (f"optbinning logra IV {coma(_c.iloc[1]['iv_sin_suavizar'], 4)} vs "
                   f"{coma(_c.iloc[0]['iv_sin_suavizar'], 4)} de la heurística (diferencia "
                   f"{coma(_c.iloc[1]['iv_sin_suavizar'] - _c.iloc[0]['iv_sin_suavizar'], 4)}; nunca negativa si ambos "
                   f"usan los mismos pre-bins y restricciones). ")
    mo.md(f"""
    **Lectura para `{var_bin.value}`.** {_txt_ob}El binner del curso (5 cuantiles, sin restricción) tiene
    IV {coma(_c.iloc[-1]['iv_dev'], 4)} y {'es' if _c.iloc[-1]['monótona'] else '**no es**'} monótono. Gini HO univariado:
    {" · ".join(f"{i.split(' (')[0]} {coma(r['gini_ho_univariado'], 3)}" for i, r in _c.iterrows())}.
    Mueva `min_event_rate_diff` a 0,04: con una tasa global de ~11%, exigir 4 pp entre bins deja 2–3 bins
    (caso 1 de la clase 4). Llévelo a 0 con `max_n_bins` = 10: aparecen bins casi iguales (caso 2). En
    `deuda_otras_prom_12m` compare *ascending* con *peak*: forzar monotonía destruye la forma real (U invertida).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 El experimento 2 del curso, con verdad conocida

    Re-hacemos el scorecard reemplazando el binner del curso por binning monótono (`auto_asc_desc`,
    `max_n_bins=6`, `min_event_rate_diff=0.005`, `min_bin_size=0.05`, especiales −9/−99 aparte) en las siete
    variables numéricas — primero con la heurística numpy, luego con optbinning con su pre-binning por
    defecto (CART), como lo usaría un practicante. `canal` queda igual.
    """)
    return


@app.cell
def _(OPTB_OK, OptimalBinning, VARS, binning_monotono, dev, etiquetar, gini_np, ho, logit_irls, np, oot, pd,
      sigmoide, woe_de_etiquetas, W_dev, W_ho, W_oot, gini_base):
    _num = [v for v in VARS if v != "canal"]

    def _modelo_con_cortes(cortes_por_var):
        _Wd, _Wh, _Wo = W_dev.copy(), W_ho.copy(), W_oot.copy()
        _formas = {}
        for _v, (_c, _esp) in cortes_por_var.items():
            _ed, _o = etiquetar(dev[_v].values, _c, _esp)
            _t = woe_de_etiquetas(_ed, dev["malo"].values, _o)[0]
            _formas[_v] = _t
            for _W, _d in [(_Wd, dev), (_Wh, ho), (_Wo, oot)]:
                _W[_v] = pd.Series(etiquetar(_d[_v].values, _c, _esp)[0]).map(_t["woe"]).fillna(0.0).values
        _X = np.column_stack([np.ones(len(_Wd)), _Wd[VARS].values])
        _b, _se, _ll = logit_irls(_X, dev["malo"].values)
        _g = {}
        for _n, _W, _d in [("DEV", _Wd, dev), ("HO", _Wh, ho), ("OOT", _Wo, oot)]:
            _g[_n] = gini_np(_d["malo"].values, sigmoide(_b[0] + _W[VARS].values @ _b[1:]))
        return _g, _b, _formas

    _cortes_np = {}
    for _v in _num:
        _esp = (-9.0, -99.0) if _v == "meses_desde_mora_12m" else ()
        _c, _t, _p = binning_monotono(dev[_v].values, dev["malo"].values, "auto_asc_desc", 6, 0.005, 0.05, _esp)
        _cortes_np[_v] = (_c, _esp)
    gini_mono_np, beta_mono_np, formas_mono_np = _modelo_con_cortes(_cortes_np)
    _filas = {"binner del curso": gini_base, "monótono numpy": gini_mono_np}
    if OPTB_OK:
        try:
            _cortes_ob = {}
            for _v in _num:
                _esp = (-9.0, -99.0) if _v == "meses_desde_mora_12m" else ()
                _ob = OptimalBinning(name=_v, monotonic_trend="auto_asc_desc", max_n_bins=6,
                                     min_event_rate_diff=0.005, min_bin_size=0.05,
                                     special_codes=list(_esp) if _esp else None, time_limit=20)
                _ob.fit(dev[_v].values, dev["malo"].values.astype(int))
                _cortes_ob[_v] = (np.asarray(_ob.splits, float), _esp)
            _filas["monótono optbinning (CART)"], _b_ob, _f_ob = _modelo_con_cortes(_cortes_ob)
        except Exception:  # noqa: BLE001
            pass
    experimento_2 = pd.DataFrame(_filas).T[["DEV", "HO", "OOT"]]
    experimento_2["caída_DEV_HO"] = experimento_2["DEV"] - experimento_2["HO"]
    experimento_2.round(3)
    return experimento_2, formas_mono_np


@app.cell
def _(coma, experimento_2, formas_mono_np, gini_separado, mo):
    _d = experimento_2
    _dif = _d.loc["monótono numpy", "HO"] - _d.loc["binner del curso", "HO"]
    _sep = gini_separado.iloc[1]["HO"] - gini_separado.iloc[0]["HO"]
    _f = formas_mono_np["deuda_otras_prom_12m"]
    mo.md(f"""
    **Lectura.** Gini HO: curso {coma(_d.loc['binner del curso','HO'], 3)} vs monótono numpy
    {coma(_d.loc['monótono numpy','HO'], 3)} (diferencia {coma(_dif, 3)}). A diferencia de Austral (0,698 vs 0,683,
    empate), aquí el binning monótono sí sube el Gini — pero la sección 5 muestra que **{coma(_sep, 3)} de esa
    diferencia viene solo de separar −9/−99**, un error del binner del curso que Austral no tenía en sus 8
    variables. Descontado eso, la monotonía compra ~{coma(_dif - _sep, 3)}: la moraleja del curso se mantiene
    (el binning óptimo compra forma y trazabilidad, no Gini), y el Gini extra que aparezca hay que atribuirlo a
    una causa concreta antes de celebrarlo. El costo está donde la verdad no es monótona:
    `deuda_otras_prom_12m` queda con {len(_f)} bins monótonos y el efecto real (sobre 5 millones de pesos el riesgo baja) se
    diluye.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Valores especiales: −9 (nunca tuvo mora) y −99 (sin bureau)

    **Qué mirar.** El binner del curso ordena los códigos como números y los junta en `(-inf, -9.0]`.
    Comparamos ese modelo con uno que les da bin propio, contra la **verdad del generador**: para −99,
    $P(\text{malo}) = 0{,}1 + 0{,}9\,\text{pd\_verdadera}$ (el generador fuerza 10% de malos adicionales);
    para el resto, $P(\text{malo}) = \text{pd\_verdadera}$. Medimos en HO (mismo periodo que DEV, sin drift).
    """)
    return


@app.cell
def _(FACTOR, VARS, W_dev, W_ho, W_oot, beta, binear, dev, gini_base, gini_np, ho, logit_irls, mo, np, oot, pd,
      sigmoide, woe_de_etiquetas):
    _v = "meses_desde_mora_12m"

    def etiquetas_especiales(x, ref, especiales=(-9.0, -99.0)):
        """Binner del curso sobre los valores normales + un bin propio por código especial."""
        _x = pd.Series(x, dtype=float).reset_index(drop=True)
        _r = pd.Series(ref, dtype=float).reset_index(drop=True)
        _e, _o = binear(_x.where(~_x.isin(especiales)), ref=_r.where(~_r.isin(especiales)))
        for _c in especiales:
            _e = _e.where(_x != _c, f"ESP {_c:g}")
        return _e.values, [f"ESP {c:g}" for c in especiales] + [o for o in _o if o != "MISSING"]

    _e_dev, _o = etiquetas_especiales(dev[_v], dev[_v])
    tabla_mora_sep = woe_de_etiquetas(_e_dev, dev["malo"].values, _o)[0]
    _Wd, _Wh = W_dev.copy(), W_ho.copy()
    _Wd[_v] = pd.Series(_e_dev).map(tabla_mora_sep["woe"]).values
    _Wh[_v] = pd.Series(etiquetas_especiales(ho[_v], dev[_v])[0]).map(tabla_mora_sep["woe"]).fillna(0.0).values
    _Wo = W_oot.copy()
    _Wo[_v] = pd.Series(etiquetas_especiales(oot[_v], dev[_v])[0]).map(tabla_mora_sep["woe"]).fillna(0.0).values
    _b_sep, _, _ = logit_irls(np.column_stack([np.ones(len(_Wd)), _Wd[VARS].values]), dev["malo"].values)
    gini_separado = pd.DataFrame({
        "binner del curso (−9/−99 mezclados)": gini_base,
        "especiales separados": {_n: gini_np(_d["malo"].values, sigmoide(_b_sep[0] + _W[VARS].values @ _b_sep[1:]))
                                 for _n, _d, _W in [("DEV", dev, _Wd), ("HO", ho, _Wh), ("OOT", oot, _Wo)]}}).T
    _pd_mix = sigmoide(beta["const"] + W_ho[VARS].values @ beta[VARS].values)
    _pd_sep = sigmoide(_b_sep[0] + _Wh[VARS].values @ _b_sep[1:])
    _m = ho[_v].values
    _verdad = np.where(_m == -99, 0.1 + 0.9 * ho["pd_verdadera"].values, ho["pd_verdadera"].values)
    _filas = []
    for _nom, _msk in [("−99 sin bureau", _m == -99), ("−9 nunca tuvo mora", _m == -9),
                       ("−9 y −99 juntos", np.isin(_m, [-9, -99])), ("resto", ~np.isin(_m, [-9, -99]))]:
        _filas.append({"grupo": _nom, "n_HO": int(_msk.sum()), "tasa_observada": ho["malo"].values[_msk].mean(),
                       "tasa_verdadera": _verdad[_msk].mean(), "pd_modelo_mezclado": _pd_mix[_msk].mean(),
                       "pd_modelo_separado": _pd_sep[_msk].mean(),
                       "pts_mora_vs_neutro_mezclado": np.mean(-beta[_v] * W_ho[_v].values[_msk] * FACTOR),
                       "pts_mora_vs_neutro_separado": np.mean(-_b_sep[1 + VARS.index(_v)] * _Wh[_v].values[_msk] * FACTOR)})
    especiales = pd.DataFrame(_filas).set_index("grupo")
    _mdev = dev[_v].values
    composicion_mezcla = pd.DataFrame({
        "n_DEV": [int((_mdev == -9).sum()), int((_mdev == -99).sum())],
        "tasa_DEV": [dev["malo"].values[_mdev == -9].mean(), dev["malo"].values[_mdev == -99].mean()]},
        index=["−9", "−99"])
    mo.vstack([especiales.round(4), mo.md("**Gini por muestra**"), gini_separado.round(3)])
    return composicion_mezcla, especiales, gini_separado, tabla_mora_sep


@app.cell
def _(coma, composicion_mezcla, especiales, mo, pct, tabla_mora_sep, tablas_woe):
    _tm = tablas_woe["meses_desde_mora_12m"]
    _mix = _tm.loc[[b for b in _tm.index if str(b).startswith("(-inf")][0]]
    _c = composicion_mezcla
    _p = (_c["n_DEV"] * _c["tasa_DEV"]).sum() / _c["n_DEV"].sum()
    _e = especiales
    mo.md(f"""
    **Lectura.** En DEV el bin mezclado tiene {coma(_mix['n'], 0)} créditos y tasa {pct(_mix['tasa_malos'])}
    (WoE {coma(_mix['woe'], 2)}) = ({coma(_c.loc['−9','n_DEV'], 0)} × {pct(_c.loc['−9','tasa_DEV'])} +
    {int(_c.loc['−99','n_DEV'])} × {pct(_c.loc['−99','tasa_DEV'])}) / {coma(_c['n_DEV'].sum(), 0)} = {pct(_p)} —
    la misma aritmética del caso 3 de la clase 4 (800 × 2,5% + 100 × 17%)/900 = 4,1%. Separados: WoE −9 =
    {coma(tabla_mora_sep.loc['ESP -9','woe'], 2)}, WoE −99 = {coma(tabla_mora_sep.loc['ESP -99','woe'], 2)}. En HO, para
    los sin bureau la tasa verdadera es {pct(_e.loc['−99 sin bureau','tasa_verdadera'])}; el modelo mezclado les
    asigna {pct(_e.loc['−99 sin bureau','pd_modelo_mezclado'])} y el separado
    {pct(_e.loc['−99 sin bureau','pd_modelo_separado'])} (observada: {pct(_e.loc['−99 sin bureau','tasa_observada'])}
    con n = {int(_e.loc['−99 sin bureau','n_HO'])}). En puntos de la variable mora (relativos al bin neutro), el sin
    bureau pasa de {coma(_e.loc['−99 sin bureau','pts_mora_vs_neutro_mezclado'], 1)} a
    {coma(_e.loc['−99 sin bureau','pts_mora_vs_neutro_separado'], 1)}: mezclado, la mora le *suma* puntos y nunca
    puede ser su reason code. Agrupar **subestima el riesgo del grupo peligroso y sobreestima el del grupo sano**:
    dos errores que en el promedio se compensan y por eso no se ven. Y no es gratis en discriminación: Gini HO
    {coma(gini_separado.iloc[0]['HO'], 3)} → {coma(gini_separado.iloc[1]['HO'], 3)} solo por separar los códigos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Reason codes

    Todos los métodos son la misma fórmula con distinta **referencia** $r_v$:
    $$\text{brecha}_{i,v} = r_v - \text{puntos}_v(x_i),$$
    y los motivos son las $k$ brechas mayores (sobre un umbral). Referencias:

    | método | $r_v$ | origen |
    |---|---|---|
    | máximo | $\max_b \text{puntos}_{v,b}$ | curso (clase 3) |
    | media poblacional | $E_{DEV}[\text{puntos}_v]$ | Reg B, comentario 9(b)(2)-5, método 2 |
    | media en el corte | $E[\text{puntos}_v \mid \text{score}\in[c, c+10)]$ | Reg B, 9(b)(2)-5, método 1 |
    | neutro | puntos de un bin con WoE = 0 | práctica de algunos proveedores |

    Empates: se desempata por rango de puntos de la variable (determinista y documentado). Controles: punto de
    corte, umbral mínimo de brecha y cliente.
    """)
    return


@app.cell
def _(mo, np, score_dev):
    corte = mo.ui.slider(int(np.percentile(score_dev, 5)), int(np.percentile(score_dev, 60)),
                         value=int(round(np.percentile(score_dev, 25))), label="Punto de corte (score)")
    umbral_brecha = mo.ui.slider(0.0, 15.0, value=3.0, step=0.5, label="Brecha mínima para reportar (puntos)")
    mo.vstack([corte, umbral_brecha])
    return corte, umbral_brecha


@app.cell
def _(PUNTOS_NEUTROS, VARS, corte, importancia, np, pd, pts_dev, score_dev, scorecard):
    MOTIVOS = {
        "uso_linea_prom_12m": "utilización de la línea sostenidamente alta el último año",
        "uso_tc_prom_12m": "tarjeta de crédito cargada de forma sostenida",
        "meses_desde_mora_12m": "mora propia reciente o sin información de bureau",
        "antiguedad_meses": "relación aún corta con la institución",
        "carga_financiera": "carga financiera elevada para el ingreso observado",
        "consultas_6m": "número de consultas de crédito recientes",
        "deuda_otras_prom_12m": "nivel de deuda en otras instituciones",
        "canal": "canal de originación",
    }
    _c = corte.value
    _banda = (score_dev >= _c) & (score_dev < _c + 10)
    referencias = pd.DataFrame({
        "máximo": scorecard.groupby("variable")["puntos"].max().reindex(VARS),
        "media poblacional": pts_dev.mean(),
        "media en el corte": pts_dev[_banda].mean() if _banda.sum() > 0 else pts_dev.mean(),
        "neutro": pd.Series(PUNTOS_NEUTROS, index=VARS),
    })
    DESEMPATE = importancia["rango_pts"].reindex(VARS).rank().values   # mayor rango gana el empate

    def top_k(P, ref, k=3, umbral=0.0, excluir=()):
        """Top-k motivos por fila. P: (n × V) puntos; ref: (V,). Devuelve índices (-1 = sin motivo)."""
        G = np.asarray(ref)[None, :] - np.asarray(P)
        G = np.where(G > umbral, G, -np.inf)
        for v in excluir:
            G[:, VARS.index(v)] = -np.inf
        clave = G + 1e-9 * DESEMPATE[None, :]
        orden = np.argsort(-clave, axis=1)[:, :k]
        valido = np.take_along_axis(G, orden, axis=1) > -np.inf
        return np.where(valido, orden, -1)

    n_banda_corte = int(_banda.sum())
    return MOTIVOS, n_banda_corte, referencias, top_k


@app.cell
def _(corte, np, pts_oot, score_oot, top_k, referencias, umbral_brecha):
    # rechazados de OOT (la población más cercana a producción con desempeño observado)
    rechazados = np.where(score_oot < corte.value)[0]
    top3_por_metodo = {m: top_k(pts_oot.values[rechazados], referencias[m].values, 3, umbral_brecha.value)
                       for m in referencias.columns}
    # tres clientes: muy bajo, justo bajo el corte y el de máximo desacuerdo entre métodos
    _orden = rechazados[np.argsort(score_oot[rechazados])]
    _cerca = _orden[-1]
    _bajo = _orden[max(0, int(0.02 * len(_orden)))]
    _sets = [np.array([set(t[j][t[j] >= 0]) for j in range(len(rechazados))]) for t in top3_por_metodo.values()]
    _acuerdo = np.array([min(len(_sets[0][j] & s[j]) for s in _sets[1:]) for j in range(len(rechazados))])
    _desac = rechazados[int(np.argmin(_acuerdo))]
    clientes_rc = {"score muy bajo": int(_bajo), "justo bajo el corte": int(_cerca),
                   "máximo desacuerdo entre métodos": int(_desac)}
    return clientes_rc, rechazados, top3_por_metodo


@app.cell
def _(clientes_rc, mo):
    cliente_rc = mo.ui.dropdown(options=list(clientes_rc.keys()), value="score muy bajo", label="Cliente (OOT, rechazado)")
    cliente_rc
    return (cliente_rc,)


@app.cell
def _(FACTOR, MOTIVOS, OFFSET, VARS, cliente_rc, clientes_rc, np, oot, pct, pd, pts_oot, referencias,
      score_oot, top_k, umbral_brecha):
    _i = clientes_rc[cliente_rc.value]
    _p = pts_oot.values[_i]
    detalle_cliente = pd.DataFrame({"valor": [oot.loc[_i, v] for v in VARS], "puntos": _p}, index=VARS)
    for _m in referencias.columns:
        detalle_cliente[f"brecha_{_m}"] = referencias[_m].values - _p
    _lineas = []
    for _m in referencias.columns:
        _t = top_k(_p[None, :], referencias[_m].values, 3, umbral_brecha.value)[0]
        _mot = [f"{MOTIVOS[VARS[j]]} (−{referencias[_m].values[j] - _p[j]:.0f} pts)" for j in _t if j >= 0]
        _lineas.append(f"- **{_m}**: " + ("; ".join(_mot) if _mot else "sin motivos sobre el umbral"))
    _pdi = 1 / (1 + np.exp((score_oot[_i] - OFFSET) / FACTOR))
    resumen_cliente = (f"**{oot.loc[_i, 'id']}** · score {score_oot[_i]:.0f} · PD {pct(_pdi)} · "
                       f"malo observado = {int(oot.loc[_i, 'malo'])}\n\n" + "\n".join(_lineas))
    detalle_cliente.round(1)
    return detalle_cliente, resumen_cliente


@app.cell
def _(mo, resumen_cliente):
    mo.md(resumen_cliente)
    return


@app.cell
def _(VARS, np, pd, rechazados, referencias, top3_por_metodo):
    _M = list(referencias.columns)
    _top1 = pd.DataFrame(index=_M, columns=_M, dtype=float)
    _jac = pd.DataFrame(index=_M, columns=_M, dtype=float)
    for _a in _M:
        for _b in _M:
            _A, _B = top3_por_metodo[_a], top3_por_metodo[_b]
            _top1.loc[_a, _b] = np.mean(_A[:, 0] == _B[:, 0])
            _jac.loc[_a, _b] = np.mean([len(set(_A[j][_A[j] >= 0]) & set(_B[j][_B[j] >= 0])) /
                                        max(1, len(set(_A[j][_A[j] >= 0]) | set(_B[j][_B[j] >= 0])))
                                        for j in range(len(rechazados))])
    acuerdo_top1 = _top1.astype(float)
    acuerdo_jaccard = _jac.astype(float)
    frecuencia_motivo1 = pd.DataFrame({m: pd.Series(top3_por_metodo[m][:, 0]).map(
        dict(enumerate(VARS))).value_counts(normalize=True) for m in _M}).fillna(0.0)
    canal_en_top3 = pd.Series({m: np.mean((top3_por_metodo[m] == VARS.index("canal")).any(axis=1)) for m in _M})
    return acuerdo_jaccard, acuerdo_top1, canal_en_top3, frecuencia_motivo1


@app.cell
def _(acuerdo_jaccard, acuerdo_top1, frecuencia_motivo1, mo):
    mo.vstack([mo.md("**Acuerdo en el motivo principal** (fracción de rechazados OOT con el mismo motivo 1)"),
               acuerdo_top1.round(3),
               mo.md("**Acuerdo en el conjunto top-3** (Jaccard medio)"), acuerdo_jaccard.round(3),
               mo.md("**Frecuencia de cada variable como motivo 1**"), frecuencia_motivo1.round(3)])
    return


@app.cell
def _(VARS, np, pts_oot, rechazados, referencias, scorecard, top3_por_metodo):
    # (a) empates con la tabla redondeada a enteros (lo que se firma en producción), método máximo
    _P = np.round(pts_oot.values[rechazados])
    _R = np.round(referencias["máximo"].values)
    _G = np.sort(_R[None, :] - _P, axis=1)[:, ::-1]
    # (b) coherencia de la etiqueta en una variable NO monótona (deuda_otras_prom_12m):
    #     el bin que más puntos pierde NO es el de mayor deuda
    _sc = scorecard[scorecard["variable"] == "deuda_otras_prom_12m"].reset_index(drop=True)
    _j = VARS.index("deuda_otras_prom_12m")
    info_empates = {
        "frac_empate_frontera": float(np.mean((_G[:, 2] == _G[:, 3]) & (_G[:, 2] > 0))),
        "deuda_bin_peor": str(_sc.loc[_sc["puntos"].idxmin(), "bin"]),
        "deuda_pts_peor": float(_sc["puntos"].min()),
        "deuda_bin_alto": str(_sc["bin"].iloc[-1]),
        "deuda_pts_alto": float(_sc["puntos"].iloc[-1]),
        "frac_deuda_otras_top3_media": float(np.mean((top3_por_metodo["media poblacional"] == _j).any(axis=1))),
    }
    # la referencia «media» y la «neutra» difieren en una constante por variable (ver check)
    constante_media_vs_neutro = (referencias["media poblacional"] - referencias["neutro"]).values
    return constante_media_vs_neutro, info_empates


@app.cell
def _(acuerdo_top1, canal_en_top3, coma, corte, info_empates, mo, n_banda_corte, pct, rechazados):
    mo.md(f"""
    **Lectura.** Con corte {corte.value} hay {coma(len(rechazados), 0)} rechazados en OOT. Los métodos no son
    intercambiables: *máximo* y *media poblacional* coinciden en el motivo principal en
    {pct(acuerdo_top1.loc['máximo','media poblacional'])} de los casos; *media poblacional* y *media en el corte*,
    en {pct(acuerdo_top1.loc['media poblacional','media en el corte'])}; *máximo* y *neutro*, en
    {pct(acuerdo_top1.loc['máximo','neutro'])}. La referencia «máximo» premia a las variables con un bin
    excepcionalmente bueno (brecha grande aunque el cliente esté en el promedio). La referencia «media en el corte»
    usa {n_banda_corte} créditos DEV en la banda [{corte.value}, {corte.value + 10}): con bandas chicas es ruidosa.
    `canal` aparece en el top-3 de hasta {pct(canal_en_top3.max())} de los rechazados (según método): si el canal
    está en el modelo y es motivo principal, Reg B no permite omitirlo (comentario 9(b)(2)-4) — la decisión de
    gobierno se toma **antes**, al decidir si la variable entra.

    Con la tabla redondeada a enteros, {pct(info_empates['frac_empate_frontera'])} de los rechazados tiene un
    empate exacto entre el motivo 3 y el 4 (método máximo): la regla de desempate no es un detalle. Y en
    `deuda_otras_prom_12m` (no monótona) el bin que más puntos pierde es `{info_empates['deuda_bin_peor']}`
    ({coma(info_empates['deuda_pts_peor'], 1)} pts), mientras el de mayor deuda `{info_empates['deuda_bin_alto']}`
    recibe {coma(info_empates['deuda_pts_alto'], 1)}: una frase «endeudamiento alto en otras instituciones» se la
    daría a quien debe 3 millones y no a quien debe 10 millones. Por eso la frase es neutra («nivel de deuda en otras
    instituciones», lo que el comentario 9(b)(2)-3 permite) o el binning se hace monótono. (Con el método de la
    media poblacional aparece en el top-3 de {pct(info_empates['frac_deuda_otras_top3_media'])} de los rechazados.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.1 Estabilidad de los reason codes (bootstrap del desarrollo completo)

    **Qué mirar.** Re-hacemos *todo* el desarrollo (cortes de bins, WoE y β) sobre remuestras bootstrap de DEV,
    re-puntuamos a los mismos rechazados de OOT y medimos cuántas veces conservan su motivo principal y su top-3.
    Un reason code que cambia con una remuestra no es una propiedad del cliente sino del ruido de estimación.
    """)
    return


@app.cell
def _(mo):
    B_boot = mo.ui.slider(10, 100, value=20, step=10, label="Réplicas bootstrap B")
    B_boot
    return (B_boot,)


@app.cell
def _(B_boot, FACTOR, N_VARS, OFFSET, VARS, a_woe, dev, logit_irls, np, oot, pd, rechazados, tabla_woe,
      top3_por_metodo, top_k, umbral_brecha):
    _rng = np.random.default_rng(2026)
    _M = ["máximo", "media poblacional", "neutro"]
    _base = {m: top3_por_metodo[m] for m in _M}
    _oot_r = oot.iloc[rechazados].reset_index(drop=True)
    _res = {m: {"top1": [], "jac": []} for m in _M}
    for _b in range(B_boot.value):
        _d = dev.iloc[_rng.integers(0, len(dev), len(dev))].reset_index(drop=True)
        _mapas = {v: tabla_woe(_d[v], _d["malo"])[0]["woe"] for v in VARS}
        _Wd = a_woe(_d, VARS, _d, _mapas)
        _bb, _, _ = logit_irls(np.column_stack([np.ones(len(_Wd)), _Wd[VARS].values]), _d["malo"].values)
        _bs = pd.Series(_bb, index=["const"] + VARS)
        _Wo = a_woe(_oot_r, VARS, _d, _mapas)
        _pt = lambda W: np.column_stack([-(_bs[v] * W[v].values + _bs["const"] / N_VARS) * FACTOR + OFFSET / N_VARS
                                         for v in VARS])
        _Po, _Pd = _pt(_Wo), _pt(_Wd)
        _pmax = np.array([max(-(_bs[v] * w + _bs["const"] / N_VARS) * FACTOR + OFFSET / N_VARS
                              for w in _mapas[v].values) for v in VARS])
        _refs = {"máximo": _pmax, "media poblacional": _Pd.mean(axis=0),
                 "neutro": np.full(len(VARS), -(_bs["const"] / N_VARS) * FACTOR + OFFSET / N_VARS)}
        for _m in _M:
            _t = top_k(_Po, _refs[_m], 3, umbral_brecha.value)
            _res[_m]["top1"].append(np.mean(_t[:, 0] == _base[_m][:, 0]))
            _res[_m]["jac"].append(np.mean([len(set(_t[j][_t[j] >= 0]) & set(_base[_m][j][_base[_m][j] >= 0])) /
                                            max(1, len(set(_t[j][_t[j] >= 0]) | set(_base[_m][j][_base[_m][j] >= 0])))
                                            for j in range(len(_t))]))
    estabilidad_rc = pd.DataFrame({m: {"top1_igual_media": np.mean(_res[m]["top1"]),
                                       "top1_igual_p10": np.percentile(_res[m]["top1"], 10),
                                       "jaccard_top3_media": np.mean(_res[m]["jac"])} for m in _M}).T
    estabilidad_rc.round(3)
    return (estabilidad_rc,)


@app.cell
def _(estabilidad_rc, mo, pct):
    _mej = estabilidad_rc["top1_igual_media"].idxmax()
    _peo = estabilidad_rc["top1_igual_media"].idxmin()
    mo.md(f"""
    **Lectura.** Re-desarrollando el modelo en remuestras de DEV, el motivo principal se mantiene en
    {pct(estabilidad_rc.loc[_mej,'top1_igual_media'])} de los rechazados con el método *{_mej}* y en
    {pct(estabilidad_rc.loc[_peo,'top1_igual_media'])} con *{_peo}*. La inestabilidad se concentra en clientes
    cuyas dos brechas mayores son parecidas: la diferencia entre ellas es menor que el error de estimación de
    |β_v|·(r_v − WoE_i). Esto es un argumento para (a) reportar el motivo con umbral, (b) documentar
    la estabilidad del top-1 como métrica de validación y (c) preferir binnings con menos bins extremos chicos
    (el método «máximo» depende del mejor bin, que suele ser pequeño y ruidoso).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. SHAP aditivo: en un scorecard, $\phi_v = \text{puntos}_v - E[\text{puntos}_v]$

    **Qué mirar.** El valor de Shapley *interventional* con distribución de fondo $\mathcal D$ se calcula por
    **definición** (enumerando las $2^{8}=256$ coaliciones y promediando la predicción de `statsmodels` sobre una
    muestra de fondo) y se compara con la fórmula cerrada. En escala de puntos (lineal en WoE) deben coincidir
    exactamente. En escala de **probabilidad** también se puede calcular, pero no es la resta de puntos y puede
    ordenar distinto las variables.
    """)
    return


@app.cell
def _(FACTOR, OFFSET, VARS, W_dev, W_oot, clientes_rc, combinations, factorial, modelo_sm, np, pd, pts_oot,
      sm):
    _rng = np.random.default_rng(7)
    fondo = W_dev[VARS].values[_rng.choice(len(W_dev), 300, replace=False)]
    _M = len(VARS)

    def f_puntos(Wm):
        _p = modelo_sm.predict(sm.add_constant(pd.DataFrame(Wm, columns=VARS), has_constant="add"))
        return OFFSET + FACTOR * np.log((1 - _p) / _p)

    def f_pd(Wm):
        return np.asarray(modelo_sm.predict(sm.add_constant(pd.DataFrame(Wm, columns=VARS), has_constant="add")))

    def shapley_fuerza_bruta(x, f):
        """Shapley exacto por enumeración de coaliciones (interventional, fondo = `fondo`)."""
        cache = {}

        def valor(S):
            if S not in cache:
                Z = fondo.copy()
                if S:
                    Z[:, list(S)] = x[list(S)]
                cache[S] = float(np.mean(f(Z)))
            return cache[S]

        phi = np.zeros(_M)
        for j in range(_M):
            otros = [k for k in range(_M) if k != j]
            for s in range(_M):
                for S in combinations(otros, s):
                    peso = factorial(s) * factorial(_M - s - 1) / factorial(_M)
                    phi[j] += peso * (valor(tuple(sorted(S + (j,)))) - valor(S))
        return phi, valor(tuple()), valor(tuple(range(_M)))

    resultados_shap = {}
    for _nom, _i in clientes_rc.items():
        _x = W_oot[VARS].values[_i]
        _phi_bf, _v0, _vN = shapley_fuerza_bruta(_x, f_puntos)
        # fórmula cerrada: puntos del cliente − puntos medios del FONDO
        _pts_x = pts_oot.values[_i]
        _pts_fondo_media = np.array([
            np.mean(-(modelo_sm.params[v] * fondo[:, k] + modelo_sm.params["const"] / _M) * FACTOR + OFFSET / _M)
            for k, v in enumerate(VARS)])
        _phi_cf = _pts_x - _pts_fondo_media
        _phi_pd, _p0, _pN = shapley_fuerza_bruta(_x, f_pd)
        resultados_shap[_nom] = pd.DataFrame({"phi_puntos_formula": _phi_cf, "phi_puntos_fuerza_bruta": _phi_bf,
                                              "phi_pd_pp": 100 * _phi_pd}, index=VARS)
        resultados_shap[_nom].attrs.update({"v0": _v0, "vN": _vN, "p0": _p0, "pN": _pN})
    tabla_shap = pd.concat(resultados_shap, names=["cliente", "variable"])
    tabla_shap.round(3)
    return f_pd, fondo, resultados_shap, shapley_fuerza_bruta, tabla_shap


@app.cell
def _(coma, mo, np, resultados_shap):
    _lin = []
    for _nom, _t in resultados_shap.items():
        _r_pts = _t["phi_puntos_formula"].rank()
        _r_pd = (-_t["phi_pd_pp"]).rank(ascending=False)
        _top_pts = _t["phi_puntos_formula"].idxmin()
        _top_pd = _t["phi_pd_pp"].idxmax()
        _lin.append(f"- *{_nom}*: Σφ(puntos) = {coma(_t['phi_puntos_formula'].sum(), 1)} = score − E[score] "
                    f"({coma(_t.attrs['vN'], 1)} − {coma(_t.attrs['v0'], 1)}); variable más negativa en puntos "
                    f"`{_top_pts}`, en PD `{_top_pd}`; Σφ(PD) = {coma(_t['phi_pd_pp'].sum(), 2)} pp.")
    _maxdif = max(np.max(np.abs(t["phi_puntos_formula"] - t["phi_puntos_fuerza_bruta"])) for t in resultados_shap.values())
    mo.md("**Lectura.** Diferencia máxima fórmula vs fuerza bruta: " + f"{_maxdif:.1e}".replace(".", ",") + " puntos (error de redondeo).\n\n" + "\n".join(_lin) +
          "\n\nEl método de reason codes «media poblacional» es exactamente −SHAP en puntos (check al final). "
          "SHAP en escala PD suma lo mismo que la diferencia de PD, pero reparte distinto: la no linealidad de la "
          "sigmoide hace que el aporte de una variable dependa del nivel del resto.")
    return


@app.cell
def _(FACTOR, VARS, W_oot, beta, f_pd, fondo, np, pd, rechazados, shapley_fuerza_bruta):
    # ¿Ordena igual SHAP en escala PD que SHAP en puntos? 20 rechazados de OOT al azar.
    _rng = np.random.default_rng(11)
    _idx = _rng.choice(rechazados, size=min(20, len(rechazados)), replace=False)
    _t1, _t3 = [], []
    for _i in _idx:
        _x = W_oot[VARS].values[_i]
        _phi_pts = -beta[VARS].values * FACTOR * (_x - fondo.mean(axis=0))      # fórmula cerrada
        _phi_pd = shapley_fuerza_bruta(_x, f_pd)[0]
        _t1.append(np.argmin(_phi_pts) == np.argmax(_phi_pd))
        _t3.append(set(np.argsort(_phi_pts)[:3]) == set(np.argsort(-_phi_pd)[:3]))
    shap_pd_vs_puntos = pd.Series({"clientes": len(_idx), "top1_igual": float(np.mean(_t1)),
                                   "top3_igual": float(np.mean(_t3))})
    shap_pd_vs_puntos
    return (shap_pd_vs_puntos,)


@app.cell
def _(mo, pct, shap_pd_vs_puntos):
    mo.md(f"""
    **Lectura.** En {int(shap_pd_vs_puntos['clientes'])} rechazados, SHAP en escala PD y SHAP en puntos coinciden
    en el motivo principal en {pct(shap_pd_vs_puntos['top1_igual'])} de los casos y en el conjunto top-3 en
    {pct(shap_pd_vs_puntos['top3_igual'])}. Si un proveedor entrega «SHAP» sin decir la escala (log-odds, PD) ni
    la distribución de fondo, sus reason codes no son comparables con los del scorecard.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Edad: ¿variable, reason code, o ninguna?

    En el generador la edad **no tiene efecto** en el log-odds verdadero; solo se correlaciona débilmente con la
    antigüedad. Regulation B (EE.UU., §1002.6(b)(2)) permite usar edad en un sistema de scoring «empíricamente
    derivado, demostrable y estadísticamente sólido» **siempre que a un solicitante de 62 años o más no se le
    asigne un factor o valor negativo**. Forzamos la edad en el modelo y revisamos esa condición.
    """)
    return


@app.cell
def _(FACTOR, N_VARS, OFFSET, VARS, W_dev, a_woe, dev, logit_irls, np, pd, stats, tabla_woe):
    _t_edad, iv_edad = tabla_woe(dev["edad"], dev["malo"])
    _W = W_dev.copy()
    _W["edad"] = a_woe(dev, ["edad"], dev, {"edad": _t_edad["woe"]})["edad"].values
    _V = VARS + ["edad"]
    _b, _se, _ = logit_irls(np.column_stack([np.ones(len(_W)), _W[_V].values]), dev["malo"].values)
    _n = len(_V)
    tabla_edad = _t_edad[["n", "tasa_malos", "woe"]].copy()
    tabla_edad["puntos"] = -(_b[-1] * tabla_edad["woe"] + _b[0] / _n) * FACTOR + OFFSET / _n
    beta_edad = {"beta": _b[-1], "se": _se[-1], "p": 2 * stats.norm.sf(abs(_b[-1] / _se[-1]))}
    _ult = tabla_edad.index[-1]
    edad_62_en_bin = _ult
    edad_62_mezcla = float(((dev["edad"] > float(_ult.split(",")[0][1:])) & (dev["edad"] < 62)).mean() /
                           max(1e-9, (dev["edad"] > float(_ult.split(",")[0][1:])).mean()))
    tabla_edad.round(3)
    return beta_edad, edad_62_en_bin, edad_62_mezcla, iv_edad, tabla_edad


@app.cell
def _(beta_edad, coma, edad_62_en_bin, edad_62_mezcla, iv_edad, mo, pct, tabla_edad):
    _p_ult = tabla_edad["puntos"].iloc[-1]
    _p_max = tabla_edad["puntos"].iloc[:-1].max()
    mo.md(f"""
    **Lectura.** IV de edad en DEV = {coma(iv_edad, 3)}; forzada en el modelo, β = {coma(beta_edad['beta'], 3)}
    (p = {coma(beta_edad['p'], 2)}), sobre un WoE que casi no se mueve: en el generador el efecto verdadero es
    **cero**, y un p de ese orden aparece por azar. Aun así el binning le asigna puntos distintos por tramo, y el tramo
    que contiene a los 62+ (`{edad_62_en_bin}`) recibe {coma(_p_ult, 1)} puntos vs un máximo de {coma(_p_max, 1)} en los
    tramos menores: {"**menos puntos que otro tramo menor de 62**: con la lectura conservadora de §1002.6(b)(2), eso no pasa" if _p_ult < _p_max - 1e-9 else "no queda por debajo de ningún tramo menor"}.
    Peor aún, ese tramo mezcla {pct(edad_62_mezcla)} de personas de menos de 62 con las de 62+: un binning que
    no aísla el umbral legal no permite ni siquiera verificar la regla. Lectura de comité: una variable sin
    señal, legalmente delicada y que además podría convertirse en reason code («edad») no entra.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Checks del módulo

    Si alguno falla, el notebook falla. Verifican coincidencias numpy vs librerías e invariantes teóricas.
    """)
    return


@app.cell
def _(FACTOR, OFFSET, OPTB_OK, PUNTOS_NEUTROS, VARS, W_dev, beta, comparacion_ajuste, comparacion_binning,
      composicion_mezcla, constante_media_vs_neutro, estabilidad, iv_ob_libreria, ll_np, modelo_sm, np,
      pts_dev, pts_oot, referencias, resultados_shap, score_dev, tablas_woe, especiales, mo):
    _checks = []
    # 1. numpy IRLS == statsmodels
    assert np.allclose(comparacion_ajuste["beta_numpy"], comparacion_ajuste["beta_statsmodels"], atol=1e-6)
    assert np.allclose(comparacion_ajuste["se_numpy"], comparacion_ajuste["se_statsmodels"], atol=1e-6)
    assert np.isclose(ll_np, modelo_sm.llf)
    _checks.append("IRLS numpy = statsmodels (β, se, log-verosimilitud)")
    # 2. todos los β negativos (convención WoE del curso)
    assert (beta[VARS] < 0).all()
    _checks.append("los 8 β son negativos (WoE = ln(%buenos/%malos))")
    # 3. score = suma de puntos = offset + factor·ln(odds)
    _p = 1 / (1 + np.exp(-(beta["const"] + W_dev[VARS].values @ beta[VARS].values)))
    assert np.allclose(pts_dev.sum(axis=1).values, OFFSET + FACTOR * np.log((1 - _p) / _p))
    assert np.allclose(score_dev, pts_dev.sum(axis=1).values)
    _checks.append("score = Σ puntos = offset + factor·ln(odds)")
    # 4. SHAP: fórmula cerrada = fuerza bruta; eficiencia (Σφ = f(x) − E f)
    for _t in resultados_shap.values():
        assert np.allclose(_t["phi_puntos_formula"], _t["phi_puntos_fuerza_bruta"], atol=1e-6)
        assert np.isclose(_t["phi_puntos_fuerza_bruta"].sum(), _t.attrs["vN"] - _t.attrs["v0"], atol=1e-6)
        assert np.isclose(_t["phi_pd_pp"].sum() / 100, _t.attrs["pN"] - _t.attrs["p0"], atol=1e-9)
    _checks.append("SHAP aditivo: fórmula cerrada = enumeración de 256 coaliciones; eficiencia en puntos y en PD")
    # 5. reason code «media poblacional» = −SHAP (fondo = DEV completo)
    _phi_dev = pts_oot.values - pts_dev.mean().values
    assert np.allclose(referencias["media poblacional"].values - pts_oot.values, -_phi_dev)
    _checks.append("brecha contra la media poblacional = −SHAP en puntos")
    # 6. media vs neutro difieren en una constante por variable = −β·factor·E_DEV[WoE]
    assert np.allclose(constante_media_vs_neutro, -beta[VARS].values * FACTOR * W_dev[VARS].mean().values)
    assert np.isclose(referencias["neutro"].iloc[0], PUNTOS_NEUTROS)
    _checks.append("referencia media − referencia neutra = −β·factor·E[WoE] (constante por variable)")
    # 7. binning monótono numpy: monótono; optbinning ≥ heurística en IV (misma pre-binning, si ambos existen)
    assert bool(comparacion_binning.iloc[0]["monótona"]) or "peak" in comparacion_binning.index[0] or "valley" in comparacion_binning.index[0]
    if OPTB_OK and len(comparacion_binning) == 3 and "OPTIMAL" in comparacion_binning.index[1]:
        assert comparacion_binning.iloc[1]["iv_sin_suavizar"] >= comparacion_binning.iloc[0]["iv_sin_suavizar"] - 1e-3
        assert np.isclose(comparacion_binning.iloc[1]["iv_sin_suavizar"], iv_ob_libreria, atol=1e-6)
        _checks.append("optbinning: IV propio = IV de la librería; IV óptimo ≥ IV heurístico")
    else:
        _checks.append("optbinning no disponible u no óptimo: se omite la comparación (la versión numpy corre igual)")
    # 8. la tasa del bin mezclado es el promedio ponderado de −9 y −99
    _tm = tablas_woe["meses_desde_mora_12m"]
    _mix = _tm.loc[[b for b in _tm.index if str(b).startswith("(-inf")][0], "tasa_malos"]
    _c = composicion_mezcla
    assert np.isclose(_mix, (_c["n_DEV"] * _c["tasa_DEV"]).sum() / _c["n_DEV"].sum())
    assert especiales.loc["−99 sin bureau", "tasa_verdadera"] > especiales.loc["−9 nunca tuvo mora", "tasa_verdadera"]
    assert (abs(especiales.loc["−99 sin bureau", "pd_modelo_separado"] - especiales.loc["−99 sin bureau", "tasa_verdadera"])
            < abs(especiales.loc["−99 sin bureau", "pd_modelo_mezclado"] - especiales.loc["−99 sin bureau", "tasa_verdadera"]))
    _checks.append("bin mezclado = promedio ponderado; separar −99 acerca la PD a la verdad")
    # 9. test z: numpy = scipy
    assert np.allclose(estabilidad["p_diferencia_numpy"], estabilidad["p_diferencia_scipy"], atol=1e-10)
    _checks.append("p-valor del test z: numpy (erf) = scipy")
    mo.md("**Todos los checks pasan:**\n\n" + "\n".join(f"- ✓ {c}" for c in _checks))
    return


if __name__ == "__main__":
    app.run()

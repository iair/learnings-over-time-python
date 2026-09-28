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
    import statsmodels.api as sm
    from scipy.optimize import brentq
    import hashlib
    import warnings
    return brentq, hashlib, mo, plt, sm, warnings


@app.cell
def _(mo):
    mo.md(r"""
    # M13 · Del logit al scorecard: scaling y tabla de puntos

    Serie 2 «Del embudo al gobierno» · profundiza la clase 3 (parte 3) y la clase 4 v21 (láminas 4–7).

    El curso dejó una línea: $\text{puntos}(v,b) = -\left(\beta_v\,\text{WoE}_{v,b} + \beta_0/n\right)\cdot\text{factor} + \text{offset}/n$,
    con PDO 20 y 600 puntos a odds 50:1 (factor 28,8539; offset 487,1229; base 71,8 en Banco Austral).
    Este notebook la desarma:

    1. Pipeline corto sobre el generador de la serie: WoE → logística (numpy **y** statsmodels) → puntos.
    2. Scaling con controles: PDO, score base, odds base y **cuatro repartos del intercepto**.
    3. Verificación: score = suma de puntos = transformación afín del logit (assert, para todos los repartos).
    4. Escalas alternativas y conversión entre escalas; tabla score ↔ PD.
    5. Reparto del intercepto y *reason codes*: qué es invariante y qué no.
    6. Redondeo a enteros: error máximo $n/2$, error típico $\sqrt{n/12}$, decisiones que cambian en el corte.
    7. Puntos negativos; recalibración ($\delta$ en el logit ⇒ $-\delta\cdot$factor en el score): ¿mover la tabla o el corte?
    8. Cuándo falla: el ancla «600 ⇔ 50:1» es una promesa de calibración, y el PDO una promesa de pendiente.
    9. optbinning `Scorecard` contra la fórmula del curso (coincidencia exacta).
    10. El scorecard como artefacto de datos (tabla + hash) — puente a M21.

    Convenciones del curso: target 1 = malo; WoE = ln(%buenos/%malos) ⇒ coeficientes negativos; todo se ajusta en DEV.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Código común de la serie
    Pegado verbatim desde `_spec/comun.py` (generador «Banco Sintético» con verdad conocida y las herramientas `binear / tabla_woe / a_woe` del curso).
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
def _():
    def coma(x, d=1):
        """Formatea un número con coma decimal y punto de miles (prosa en español)."""
        s = f"{x:,.{d}f}"
        return s.replace(",", "§").replace(".", ",").replace("§", ".")

    def pct(x, d=1):
        return coma(100 * x, d) + " %"
    return coma, pct


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Pipeline corto: WoE → logística → β

    No repetimos el embudo (M8–M11): tomamos 5 variables del generador con correlación WoE ≤ 0,70; cuatro con IV ≥ 0,10 y
    `antiguedad_meses` (IV 0,088) **forzada** como variable de relación, documentada (el [R] de la clase 3: forzar no es pecado, no documentarlo sí).
    `uso_tc_prom_12m` queda fuera a propósito: su WoE correlaciona 0,74 con `uso_linea_prom_12m` (regla del curso).
    Bins y WoE se calculan en DEV y se **aplican** a HO/OOT/TTD. Qué mirar: IV por variable y que los 5 β sean negativos.
    """)
    return


@app.cell
def _(generar_cartera):
    cartera = generar_cartera()
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    ho = cartera[cartera["muestra"] == "HO"].reset_index(drop=True)
    oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)
    ttd = cartera[cartera["muestra"] == "TTD"].reset_index(drop=True)
    return cartera, dev, ho, oot, ttd


@app.cell
def _(a_woe, binear, dev, ho, oot, pd, tabla_woe, ttd):
    VARIABLES = ["uso_linea_prom_12m", "meses_desde_mora_12m", "antiguedad_meses",
                 "carga_financiera", "consultas_6m"]
    tablas_woe, mapas = {}, {}
    _filas = []
    for _v in VARIABLES:
        _t, _iv = tabla_woe(dev[_v], dev["malo"])
        tablas_woe[_v] = _t
        mapas[_v] = _t["woe"].to_dict()
        _filas.append({"variable": _v, "bins": len(_t), "IV_DEV": round(_iv, 3),
                       "WoE_min": round(_t["woe"].min(), 3), "WoE_max": round(_t["woe"].max(), 3)})
    resumen_iv = pd.DataFrame(_filas)

    W_dev = a_woe(dev, VARIABLES, dev, mapas)
    W_ho = a_woe(ho, VARIABLES, dev, mapas)
    W_oot = a_woe(oot, VARIABLES, dev, mapas)
    W_ttd = a_woe(ttd, VARIABLES, dev, mapas)

    def etiquetas_bins(df):
        """Etiqueta de bin (cortes de DEV) de cada variable: lo que usa el scorecard en producción."""
        return pd.DataFrame({_x: binear(df[_x], ref=dev[_x])[0].values for _x in VARIABLES})

    B_dev, B_ho, B_oot, B_ttd = (etiquetas_bins(_d) for _d in (dev, ho, oot, ttd))
    _corr = W_dev.corr().abs().where(lambda m: m < 0.999).max().max()
    resumen_iv
    return (B_dev, B_ho, B_oot, B_ttd, VARIABLES, W_dev, W_ho, W_oot, W_ttd,
            etiquetas_bins, mapas, resumen_iv, tablas_woe)


@app.cell
def _(VARIABLES, W_dev, dev, np, pd, sm):
    def irls_logit(X, y, tol=1e-12, max_iter=100):
        """Logística por Newton-Raphson (= IRLS). Devuelve [β0, β1..βk].
        Paso: β ← β + (XᵀWX)⁻¹ Xᵀ(y − p), con W = diag(p(1−p))."""
        X1 = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        b = np.zeros(X1.shape[1])
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-(X1 @ b)))
            H = X1.T @ (X1 * (p * (1 - p))[:, None])
            paso = np.linalg.solve(H, X1.T @ (y - p))
            b = b + paso
            if np.max(np.abs(paso)) < tol:
                break
        return b

    y_dev = dev["malo"].to_numpy()
    beta_np = irls_logit(W_dev[VARIABLES].to_numpy(), y_dev)
    modelo_sm = sm.Logit(y_dev, sm.add_constant(W_dev[VARIABLES])).fit(disp=0, tol=1e-12)
    assert np.allclose(beta_np, modelo_sm.params.to_numpy(), atol=1e-7), "IRLS numpy ≠ statsmodels"
    beta = pd.Series(beta_np, index=["const"] + VARIABLES)
    tabla_beta = pd.DataFrame({"β numpy (IRLS)": beta_np, "β statsmodels": modelo_sm.params.to_numpy(),
                               "p-valor": modelo_sm.pvalues.to_numpy()}, index=beta.index)
    tabla_beta.round(6)
    return beta, irls_logit, modelo_sm, tabla_beta, y_dev


@app.cell
def _(beta, coma, dev, mo, np, pct):
    _tasa = dev["malo"].mean()
    mo.md(f"""
    **Lectura.** IRLS en numpy y `statsmodels.Logit` coinciden a 1e-7 (el assert lo exige).
    Los 5 coeficientes son negativos (convención WoE del curso). El intercepto
    β₀ = {coma(beta['const'], 4)} es casi exactamente el log-odds de la tasa de malos de DEV
    ({pct(_tasa, 2)} ⇒ ln(p/(1−p)) = {coma(np.log(_tasa / (1 - _tasa)), 4)}): con WoE centrado cerca de 0,
    el intercepto carga el **nivel**. En Banco Austral: β₀ = −3,025 con 4,97 % de malos (logit −2,95).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Scaling: de log-odds a puntos

    El score es una función afín del log-odds de **buenos**:
    $$\text{score} = \text{offset} + \text{factor}\cdot\ln\frac{1-p}{p} = \text{offset} - \text{factor}\cdot\eta,\qquad \eta=\beta_0+\sum_v\beta_v\,\text{WoE}_v .$$
    Dos condiciones fijan las dos incógnitas: $\text{score}(\text{odds base}) = \text{score base}$ y
    $\text{score}(2\cdot\text{odds}) - \text{score}(\text{odds}) = \text{PDO}$ ⇒ factor = PDO/ln 2 y
    offset = score base − factor·ln(odds base). Al distribuir $-\text{factor}\cdot\eta$ entre variables queda una
    **constante** $C = \text{offset} - \beta_0\cdot\text{factor}$ que hay que repartir. Cuatro repartos:

    | reparto | base de la variable $v$ | fila «constante» |
    |---|---|---|
    | partes iguales (curso) | $C/n$ | 0 |
    | neutro fijo $N$ | $N$ (WoE = 0 ⇒ $N$ puntos) | $C - nN$ |
    | mínimo cero | $-\min_b a_{v,b}$ (peor bin = 0) | $C-\sum_v \text{base}_v$ |
    | proporcional al rango | $C\cdot R_v/\sum_u R_u$ | 0 |

    con $a_{v,b} = -\beta_v\cdot\text{factor}\cdot\text{WoE}_{v,b}$ y $R_v = \max_b a_{v,b}-\min_b a_{v,b}$. Mueve los controles.
    """)
    return


@app.cell
def _(mo):
    pdo_ui = mo.ui.slider(5, 80, step=1, value=20, label="PDO (puntos para duplicar las odds)")
    base_ui = mo.ui.number(start=100, stop=1000, step=10, value=600, label="Score base")
    odds_ui = mo.ui.number(start=1, stop=1000, step=1, value=50, label="Odds base (buenos:malos)")
    reparto_ui = mo.ui.dropdown(
        options={"Partes iguales (curso)": "partes_iguales", "Neutro fijo N + constante": "neutro_fijo",
                 "Mínimo cero + constante": "minimo_cero", "Proporcional al rango": "proporcional_rango"},
        value="Partes iguales (curso)", label="Reparto del intercepto")
    neutro_ui = mo.ui.number(start=-100, stop=200, step=5, value=0, label="N (puntos con WoE = 0, reparto neutro fijo)")
    mo.vstack([mo.hstack([pdo_ui, base_ui, odds_ui]), mo.hstack([reparto_ui, neutro_ui])])
    return base_ui, neutro_ui, odds_ui, pdo_ui, reparto_ui


@app.cell
def _(np, pd):
    REPARTOS = ["partes_iguales", "neutro_fijo", "minimo_cero", "proporcional_rango"]

    def parametros_scaling(pdo, score_base, odds_base):
        """factor = PDO/ln2 ; offset = score_base − factor·ln(odds_base)."""
        factor = pdo / np.log(2)
        return factor, score_base - factor * np.log(odds_base)

    def tabla_puntos(mapas, beta, variables, factor, offset, reparto="partes_iguales", neutro=0.0):
        """Scorecard largo: una fila por (variable, bin) + fila '_constante'.
        puntos = base_v + a_vb, a_vb = −β_v·factor·WoE_vb ; Σ_v base_v + constante = C."""
        n = len(variables)
        C = offset - beta["const"] * factor
        aportes = {v: {b: -beta[v] * factor * w for b, w in mapas[v].items()} for v in variables}
        if reparto == "partes_iguales":
            base = {v: C / n for v in variables}
        elif reparto == "neutro_fijo":
            base = {v: float(neutro) for v in variables}
        elif reparto == "minimo_cero":
            base = {v: -min(aportes[v].values()) for v in variables}
        elif reparto == "proporcional_rango":
            R = {v: max(aportes[v].values()) - min(aportes[v].values()) for v in variables}
            base = {v: C * R[v] / sum(R.values()) for v in variables}
        else:
            raise ValueError(reparto)
        filas = [{"variable": v, "bin": b, "woe": mapas[v][b], "beta": beta[v],
                  "pendiente": -beta[v] * factor, "base": base[v], "aporte": a,
                  "puntos": base[v] + a}
                 for v in variables for b, a in aportes[v].items()]
        constante = C - sum(base.values())
        filas.append({"variable": "_constante", "bin": "—", "woe": np.nan, "beta": np.nan,
                      "pendiente": np.nan, "base": constante, "aporte": 0.0, "puntos": constante})
        return pd.DataFrame(filas)

    def puntuar(etiquetas, tabla, columna="puntos"):
        """Score por BÚSQUEDA en la tabla (sin modelo). Bin no visto ⇒ puntos con WoE = 0 (= base_v)."""
        puntos = pd.DataFrame(index=etiquetas.index)
        for v in etiquetas.columns:
            sub = tabla[tabla["variable"] == v]
            mapa = dict(zip(sub["bin"], sub[columna]))
            neutro = sub["base"].iloc[0] if columna == "puntos" else np.round(sub["base"].iloc[0])
            puntos[v] = etiquetas[v].map(mapa).fillna(neutro).astype(float)
        const = tabla.loc[tabla["variable"] == "_constante", columna].iloc[0]
        return puntos, puntos.sum(axis=1).to_numpy() + const

    def score_desde_eta(eta, factor, offset):
        """Transformación afín del logit: score = offset − factor·η = offset + factor·ln((1−p)/p)."""
        return offset - factor * np.asarray(eta)

    def pd_desde_score(score, factor, offset):
        return 1.0 / (1.0 + np.exp((np.asarray(score) - offset) / factor))
    return REPARTOS, parametros_scaling, pd_desde_score, puntuar, score_desde_eta, tabla_puntos


@app.cell
def _(VARIABLES, base_ui, beta, mapas, neutro_ui, odds_ui, parametros_scaling, pdo_ui, reparto_ui,
      tabla_puntos):
    factor, offset = parametros_scaling(pdo_ui.value, base_ui.value, odds_ui.value)
    scorecard = tabla_puntos(mapas, beta, VARIABLES, factor, offset, reparto_ui.value, neutro_ui.value)
    scorecard.round(3)
    return factor, offset, scorecard


@app.cell
def _(base_ui, beta, coma, factor, mo, np, odds_ui, offset, pdo_ui, scorecard):
    _C = offset - beta["const"] * factor
    _pb = 1 / (1 + odds_ui.value)
    _lect = " · ".join(
        f"{coma(base_ui.value + k * pdo_ui.value, 0)} pts = odds {coma(odds_ui.value * 2 ** k, 1)}:1 (PD {coma(100 / (1 + odds_ui.value * 2 ** k), 2)} %)"
        for k in (-1, 0, 1))
    _neg = int((scorecard.loc[scorecard["variable"] != "_constante", "puntos"] < 0).sum())
    mo.md(f"""
    **Parámetros.** factor = PDO/ln 2 = **{coma(factor, 4)}** · offset = **{coma(offset, 4)}** ·
    constante a repartir C = offset − β₀·factor = {coma(offset, 2)} + {coma(-beta['const'] * factor, 2)} = **{coma(_C, 2)}**
    (con 5 variables en partes iguales: base {coma(_C / 5, 2)} por variable).
    Lectura de la escala: {_lect}. Cada {pdo_ui.value} puntos las odds se duplican.
    Filas con puntos negativos en esta tabla: **{_neg}**. PD en el score base: {coma(100 * _pb, 2)} %.
    (Con los parámetros del curso y el β₀ de Austral: factor 28,8539 · offset 487,1229 · base 60,89 + 10,91 = 71,80.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Verificación: score = suma de puntos = transformación del logit

    Calculamos el score de **dos formas independientes** en DEV, HO, OOT y TTD: (a) buscando la etiqueta de bin de cada
    cliente en la tabla y sumando (lo que hace producción); (b) $\text{offset}-\text{factor}\cdot\eta$ con la PD del modelo.
    Deben coincidir a precisión de máquina **para los cuatro repartos** (el reparto no cambia el score, solo cómo se lee cada fila).
    """)
    return


@app.cell
def _(B_dev, B_ho, B_oot, B_ttd, REPARTOS, VARIABLES, W_dev, W_ho, W_oot, W_ttd, beta, factor, mapas,
      neutro_ui, np, offset, pd, puntuar, score_desde_eta, tabla_puntos):
    def eta_de(W):
        return beta["const"] + W[VARIABLES].to_numpy() @ beta[VARIABLES].to_numpy()

    _filas = []
    for _rep in REPARTOS:
        _tab = tabla_puntos(mapas, beta, VARIABLES, factor, offset, _rep, neutro_ui.value)
        for _nom, _B, _W in [("DEV", B_dev, W_dev), ("HO", B_ho, W_ho), ("OOT", B_oot, W_oot), ("TTD", B_ttd, W_ttd)]:
            _, _s_suma = puntuar(_B, _tab)
            _s_logit = score_desde_eta(eta_de(_W), factor, offset)
            _filas.append({"reparto": _rep, "muestra": _nom, "n": len(_B),
                           "max |suma − logit|": float(np.max(np.abs(_s_suma - _s_logit)))})
    verif_suma = pd.DataFrame(_filas)
    assert verif_suma["max |suma − logit|"].max() < 1e-9, "score por suma de puntos ≠ transformación del logit"
    verif_suma.pivot(index="reparto", columns="muestra", values="max |suma − logit|")
    return eta_de, verif_suma


@app.cell
def _(B_dev, coma, mo, np, plt, puntuar, scorecard):
    _, score_dev = puntuar(B_dev, scorecard)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.hist(score_dev, bins=50, color="#4C72B0", alpha=0.85)
    _ax.set_title("Distribución del score en DEV (escala elegida)")
    _ax.set_xlabel("score (puntos)")
    _ax.set_ylabel("créditos")
    _p = np.percentile(score_dev, [5, 50, 95])
    for _q, _lab in zip(_p, ("p5", "mediana", "p95")):
        _ax.axvline(_q, ls="--", color="gray", lw=1)
        _ax.text(_q, _ax.get_ylim()[1] * 0.92, f" {_lab}", fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Score DEV: p5 **{coma(_p[0], 0)}** · mediana **{coma(_p[1], 0)}** · p95 **{coma(_p[2], 0)}** "
                           f"(Austral: 524 · 612 · 705). Nuestra cartera sintética tiene 11 % de malos en DEV, "
                           f"contra 5 % en Austral: por eso la masa cae ~50 puntos más abajo con la misma escala.")])
    return (score_dev,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Escalas alternativas y conversión entre escalas

    Dos escalas $A$ y $B$ que miden el mismo log-odds están ligadas por una recta:
    $$s_B = \text{offset}_B + \frac{\text{factor}_B}{\text{factor}_A}\,(s_A-\text{offset}_A).$$
    Una escala **creciente con el riesgo** es el caso $\text{factor}_B<0$ (se escribe $s = \text{offset}' + f\cdot\ln\frac{p}{1-p}$).
    La tabla muestra la misma PD en tres escalas; el assert verifica que convertir el score de DEV de la escala
    del curso a cada otra escala da la misma PD cliente a cliente.
    """)
    return


@app.cell
def _(np, parametros_scaling, pd, pd_desde_score, score_dev, factor, offset):
    ESCALAS = {"Curso: PDO 20 · 600 @ 50:1": (20, 600, 50),
               "PDO 40 · 500 @ 20:1": (40, 500, 20),
               "PDO 50 · 1000 @ 100:1": (50, 1000, 100),
               "Riesgo ↑: PDO −20 · 400 @ 50:1": (-20, 400, 50)}

    def convertir(s_a, fa, oa, fb, ob):
        return ob + fb / fa * (np.asarray(s_a) - oa)

    _pds = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30]
    _t = {"PD": [f"{100 * p:.2f} %" for p in _pds], "odds buenos:malos": [round((1 - p) / p, 1) for p in _pds]}
    for _nom, (_pdo, _sb, _ob) in ESCALAS.items():
        _f, _o = parametros_scaling(_pdo, _sb, _ob)
        _t[_nom] = [round(_o + _f * np.log((1 - p) / p), 1) for p in _pds]
    tabla_escalas = pd.DataFrame(_t)

    # Conversión cliente a cliente desde la escala elegida arriba (factor, offset) a cada escala
    _pd_a = pd_desde_score(score_dev, factor, offset)
    for _nom, (_pdo, _sb, _ob) in ESCALAS.items():
        _f, _o = parametros_scaling(_pdo, _sb, _ob)
        _s_b = convertir(score_dev, factor, offset, _f, _o)
        assert np.allclose(pd_desde_score(_s_b, _f, _o), _pd_a, rtol=1e-10), _nom
    tabla_escalas
    return ESCALAS, convertir, tabla_escalas


@app.cell
def _(mo):
    mo.md(r"""
    **Lectura.** Cambiar PDO/base/odds es un cambio de **unidades**, no de modelo: el orden, el Gini y la PD de cada
    cliente no cambian (assert). Lo que sí cambia: la **resolución** (con PDO 50 una unidad de log-odds vale 72 puntos;
    con PDO 20, 29) y, por eso, el costo relativo de redondear (sección 5). En la escala creciente con el riesgo los
    signos de la tabla se invierten: un bin bueno **resta** puntos. Mezclar escalas en un mismo sistema (p. ej. corte
    definido en una, reporte en otra) es la trampa operativa típica.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Reparto del intercepto: qué cambia en la lectura y qué no

    El reparto suma una constante distinta a cada variable y compensa en la fila «constante»: el **score no cambia**.
    Cambia (i) el número impreso en cada fila, (ii) si hay puntos negativos, (iii) cualquier regla que compare
    **puntos absolutos entre variables**. Los *reason codes* del curso usan la **brecha al máximo de la variable**
    ($\max_b p_{v,b} - p_{v,\text{obtenido}}$): la constante de la variable se cancela ⇒ son invariantes al reparto.
    Lo mismo vale para la brecha contra el **promedio** de la variable. Un método ingenuo («las 3 variables con menos puntos») sí depende del reparto. Lo medimos sobre TTD.
    """)
    return


@app.cell
def _(B_ttd, REPARTOS, VARIABLES, beta, factor, mapas, neutro_ui, np, offset, pd, puntuar, tabla_puntos):
    def top3_brecha(puntos, tab):
        maximos = tab[tab["variable"] != "_constante"].groupby("variable")["puntos"].max()
        brechas = maximos[VARIABLES].to_numpy()[None, :] - puntos[VARIABLES].to_numpy()
        return np.argsort(-brechas, axis=1, kind="stable")[:, :3]

    def top3_media(puntos):
        # Reg B (EE.UU.), comentario oficial 9(b)(2): brecha contra el puntaje PROMEDIO de cada factor
        brechas = puntos[VARIABLES].mean().to_numpy()[None, :] - puntos[VARIABLES].to_numpy()
        return np.argsort(-brechas, axis=1, kind="stable")[:, :3]

    def top3_ingenuo(puntos):
        return np.argsort(puntos[VARIABLES].to_numpy(), axis=1, kind="stable")[:, :3]

    _tabs = {r: tabla_puntos(mapas, beta, VARIABLES, factor, offset, r, neutro_ui.value) for r in REPARTOS}
    _pts = {r: puntuar(B_ttd, _tabs[r])[0] for r in REPARTOS}
    _ref_b, _ref_i = top3_brecha(_pts["partes_iguales"], _tabs["partes_iguales"]), top3_ingenuo(_pts["partes_iguales"])
    _ref_m = top3_media(_pts["partes_iguales"])
    _filas = []
    for _r in REPARTOS:
        _t = _tabs[_r]
        _sin = _t[_t["variable"] != "_constante"]
        _filas.append({
            "reparto": _r,
            "constante": round(_t.loc[_t["variable"] == "_constante", "puntos"].iloc[0], 2),
            "puntos mín. fila": round(_sin["puntos"].min(), 2),
            "puntos máx. fila": round(_sin["puntos"].max(), 2),
            "filas negativas": int((_sin["puntos"] < 0).sum()),
            "% TTD top-3 brecha ≠ curso": round(100 * np.mean(np.any(top3_brecha(_pts[_r], _t) != _ref_b, axis=1)), 2),
            "% TTD top-3 brecha a la media ≠ curso": round(100 * np.mean(np.any(top3_media(_pts[_r]) != _ref_m, axis=1)), 2),
            "% TTD top-3 ingenuo ≠ curso": round(100 * np.mean(np.any(top3_ingenuo(_pts[_r]) != _ref_i, axis=1)), 2),
        })
    comparacion_repartos = pd.DataFrame(_filas)
    rangos_variable = (_tabs["partes_iguales"].query("variable != '_constante'")
                       .groupby("variable")["puntos"].agg(["min", "max"]).assign(rango=lambda d: d["max"] - d["min"])
                       .sort_values("rango", ascending=False).round(2))
    assert (comparacion_repartos["% TTD top-3 brecha ≠ curso"] == 0).all(), "reason codes por brecha deben ser invariantes"
    assert (comparacion_repartos["% TTD top-3 brecha a la media ≠ curso"] == 0).all()
    comparacion_repartos
    return comparacion_repartos, rangos_variable, top3_brecha, top3_ingenuo, top3_media


@app.cell
def _(comparacion_repartos, mo, rangos_variable):
    _mx = comparacion_repartos["% TTD top-3 ingenuo ≠ curso"].max()
    mo.vstack([mo.md(f"""
    **Lectura.** Con reason codes por brecha al máximo (curso) o por brecha a la media de cada variable (uno de los métodos
    que acepta el comentario oficial de la Regulation B de EE.UU.), **0 %** de las solicitudes TTD cambia de motivos al cambiar el reparto
    (assert): toda brecha contra una referencia **de la misma variable** cancela la constante. Con el método ingenuo, hasta **{str(_mx).replace('.', ',')} %** cambia: el motivo dependería de una
    convención de presentación, no del riesgo. Los **rangos** por variable (abajo) tampoco dependen del reparto —
    son $|\\beta_v|\\cdot$factor·(rango de WoE) — y son lo que se discute en comité (clase 3, lámina 32).
    """), rangos_variable])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Redondeo a enteros

    En producción la tabla se firma en enteros. Si cada fila tiene error $e_{v,b}=\text{round}(p_{v,b})-p_{v,b}\in[-\tfrac12,\tfrac12]$,
    el score de un cliente acumula $\varepsilon=\sum_v e_{v,b(v)}$ (+ el de la constante si existe):
    **cota dura** $|\varepsilon|\le n/2$; si los $e$ se comportan como uniformes independientes, $\text{sd}(\varepsilon)\approx\sqrt{n/12}$ y
    $E|\varepsilon|\approx\sqrt{n/12}\sqrt{2/\pi}$. En la PD: las odds se multiplican por $e^{\varepsilon/\text{factor}}$.
    Decisiones que cambian en un corte $c$ ≈ densidad del score en $c$ × $E|\varepsilon|$. Comparamos **redondear puntos** (tabla entera)
    contra **redondear el score** (tabla con decimales y redondeo final) sobre las 24.000 solicitudes.
    """)
    return


@app.cell
def _(mo, score_dev, np):
    corte_ui = mo.ui.slider(int(np.percentile(score_dev, 5)), int(np.percentile(score_dev, 95)), step=1,
                            value=int(np.percentile(score_dev, 30)), label="Corte (aprueba si score ≥ corte)")
    corte_ui
    return (corte_ui,)


@app.cell
def _(VARIABLES, cartera, corte_ui, etiquetas_bins, factor, np, pd, puntuar, scorecard):
    B_todas = etiquetas_bins(cartera)
    tabla_entera = scorecard.assign(puntos_int=np.round(scorecard["puntos"]))
    _, s_exacto = puntuar(B_todas, tabla_entera, "puntos")
    _, s_pts_int = puntuar(B_todas, tabla_entera, "puntos_int")
    s_score_int = np.round(s_exacto)
    eps = s_pts_int - s_exacto
    _c = corte_ui.value
    _ap_ex, _ap_pts, _ap_sc = s_exacto >= _c, s_pts_int >= _c, s_score_int >= _c
    _n_filas_const = int(abs(tabla_entera.loc[tabla_entera["variable"] == "_constante", "puntos"].iloc[0]) > 1e-9)
    _n_eff = len(VARIABLES) + _n_filas_const
    _dens = np.mean(np.abs(s_exacto - _c) <= 2.0) / 4.0
    _e_tab = (tabla_entera["puntos_int"] - tabla_entera["puntos"]).abs()
    _cota_tabla = float(tabla_entera.assign(e=_e_tab).groupby("variable")["e"].max().sum())
    resumen_redondeo = pd.DataFrame({
        "métrica": ["n filas que redondean (variables + constante≠0)", "cota teórica n/2", "cota de ESTA tabla Σ max|e|",
                    "max |ε| observado", "media(ε) observada (sesgo)", "sd(ε) observada", "sd teórica √(n/12)", "E|ε| observado", "E|ε| teórico",
                    "max error relativo en odds, exp(max|ε|/factor) − 1",
                    "decisiones que cambian: puntos enteros vs exacto", "decisiones que cambian: score redondeado vs exacto",
                    "decisiones que cambian: puntos enteros vs score redondeado",
                    "aproximación densidad(c)·E|ε| (n esperado)", "solicitudes"],
        "valor": [_n_eff, _n_eff / 2, _cota_tabla, np.abs(eps).max(), eps.mean(), eps.std(), np.sqrt(_n_eff / 12),
                  np.abs(eps).mean(), np.sqrt(_n_eff / 12) * np.sqrt(2 / np.pi),
                  np.exp(np.abs(eps).max() / factor) - 1,
                  int(np.sum(_ap_ex != _ap_pts)), int(np.sum(_ap_ex != _ap_sc)), int(np.sum(_ap_pts != _ap_sc)),
                  _dens * np.abs(eps).mean() * len(s_exacto), len(s_exacto)]})
    assert np.abs(eps).max() <= _n_eff / 2 + 1e-9
    assert np.abs(eps).max() <= _cota_tabla + 1e-9
    resumen_redondeo.round(4)
    return B_todas, eps, resumen_redondeo, s_exacto, s_pts_int, s_score_int, tabla_entera


@app.cell
def _(coma, corte_ui, eps, mo, np, plt, resumen_redondeo):
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.hist(eps, bins=np.linspace(-2.5, 2.5, 51), color="#8172B2")
    _ax.set_title("Error de score por redondear la tabla a enteros (24.000 solicitudes)")
    _ax.set_xlabel("ε = score con puntos enteros − score exacto (puntos)")
    _ax.set_ylabel("solicitudes")
    _fig.tight_layout()
    _v = dict(zip(resumen_redondeo["métrica"], resumen_redondeo["valor"]))
    mo.vstack([_fig, mo.md(f"""
    **Lectura (corte {corte_ui.value}).** El histograma **no** es una campana suave: cada cliente hereda los errores fijos
    de sus bins, así que ε toma pocos valores (uno por combinación de bins). max |ε| = {coma(_v['max |ε| observado'], 2)}
    (cota n/2 = {coma(_v['cota teórica n/2'], 1)}; cota de esta tabla {coma(_v['cota de ESTA tabla Σ max|e|'], 2)}),
    sd {coma(_v['sd(ε) observada'], 3)} vs √(n/12) = {coma(_v['sd teórica √(n/12)'], 3)}, pero con **sesgo medio**
    {coma(_v['media(ε) observada (sesgo)'], 3)}: los bins más poblados (p. ej. «sin mora») tienen errores fijos que no se cancelan,
    así que E|ε| = {coma(_v['E|ε| observado'], 3)} supera al teórico {coma(_v['E|ε| teórico'], 3)} de errores independientes.
    En el corte cambian **{int(_v['decisiones que cambian: puntos enteros vs exacto'])}** de 24.000 decisiones
    ({coma(100 * _v['decisiones que cambian: puntos enteros vs exacto'] / 24000, 2)} %) al pasar a tabla entera; la aproximación
    densidad × E|ε| predice {coma(_v['aproximación densidad(c)·E|ε| (n esperado)'], 0)}. Redondear el **score** en vez de los puntos
    también mueve decisiones ({int(_v['decisiones que cambian: score redondeado vs exacto'])}): la pregunta no es «exacto vs redondeado»
    sino **cuál es el artefacto firmado** — y que desarrollo, validación y producción usen ese mismo.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Puntos negativos

    Con partes iguales, una fila es negativa si $C/n + (-\beta_v\,\text{factor})\,\text{WoE}_{v,b} < 0$, es decir, si
    el bin es tan malo que su aporte supera la base. Como $C = \text{score base} - \text{factor}(\ln\text{odds base}+\beta_0)$,
    aparecen con **score base bajo, PDO alto, muchas variables** o con repartos «neutro fijo» (N = 0). No son un error:
    el score sigue siendo exacto. La convención de «desplazar» (mínimo cero) solo mueve constantes entre filas.
    La grilla muestra el mínimo de la tabla (partes iguales) para varias escalas con este modelo.
    """)
    return


@app.cell
def _(VARIABLES, beta, mapas, pd, parametros_scaling, tabla_puntos):
    _filas = []
    for _pdo in (20, 40, 60, 80):
        _fila = {"PDO": _pdo}
        for _sb in (200, 300, 400, 600):
            _f, _o = parametros_scaling(_pdo, _sb, 50)
            _t = tabla_puntos(mapas, beta, VARIABLES, _f, _o, "partes_iguales")
            _fila[f"base {_sb} @ 50:1"] = round(_t.loc[_t["variable"] != "_constante", "puntos"].min(), 1)
        _filas.append(_fila)
    grilla_negativos = pd.DataFrame(_filas).set_index("PDO")
    grilla_negativos
    return (grilla_negativos,)


@app.cell
def _(grilla_negativos, mo):
    _n = int((grilla_negativos < 0).sum().sum())
    mo.md(f"""
    **Lectura.** {_n} de 16 combinaciones producen al menos una fila negativa (mínimo &lt; 0 en la grilla). En la escala del curso
    (PDO 20, 600 @ 50:1) el mínimo es holgadamente positivo. Si el negocio exige «puntos ≥ 0», se desplaza con mínimo cero
    y se publica la constante; lo que **no** se hace es truncar a 0 (eso sí cambia scores y rompe la igualdad con el logit).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Recalibración y scaling: ¿se mueve la tabla o el corte?

    Calibración PIT del curso: $\text{PD}^{cal}_i=\sigma(\text{logit}(\text{PD}_i)+\delta)$. En el score:
    $s^{cal}=\text{offset}-\text{factor}(\eta+\delta)=s-\delta\cdot\text{factor}$ — **todos se mueven igual**.
    Calculamos δ exacto (brentq) en OOT como en la clase 4 (en Austral: 0,177 exacto vs 0,143 aproximado ⇒ 5,11 puntos) y comparamos
    tres implementaciones: **A** mover el corte a $c+\delta\cdot$factor con la tabla entera intacta; **B** restar
    $\delta\cdot$factor$/n$ a cada fila y re-redondear; **C** dejar tabla y corte y cambiar solo el mapeo score → PD (el corte expresado en PD se mueve solo).
    """)
    return


@app.cell
def _(B_todas, VARIABLES, W_oot, brentq, corte_ui, eta_de, factor, np, oot, pd, puntuar, s_exacto, s_pts_int,
      tabla_entera):
    _eta = eta_de(W_oot)
    _pd = 1 / (1 + np.exp(-_eta))
    tasa_oot = oot["malo"].mean()
    delta_exacto = brentq(lambda d: np.mean(1 / (1 + np.exp(-(_eta + d)))) - tasa_oot, -5, 5, xtol=1e-12)
    _lg = lambda p: np.log(p / (1 - p))
    delta_aprox = _lg(tasa_oot) - _lg(_pd.mean())
    desplazamiento = delta_exacto * factor
    _n = len(VARIABLES)
    # B: re-escalar la tabla (restar δ·factor/n por fila de variable) y re-redondear
    _tab_b = tabla_entera.copy()
    _es_var = _tab_b["variable"] != "_constante"
    _tab_b.loc[_es_var, "puntos"] = _tab_b.loc[_es_var, "puntos"] - desplazamiento / _n
    _tab_b["puntos_int"] = np.round(_tab_b["puntos"])
    _, _s_b = puntuar(B_todas, _tab_b, "puntos_int")
    _c = corte_ui.value
    _ap_a = s_pts_int >= _c + desplazamiento          # A: tabla intacta, corte movido
    _ap_b = _s_b >= _c                                 # B: tabla re-escalada, corte intacto
    _ap_exacto = (s_exacto - desplazamiento) >= _c     # referencia: score calibrado exacto (sin redondeo)
    recalib = pd.DataFrame({
        "concepto": ["tasa observada OOT", "PD media modelo en OOT", "δ exacto (brentq)", "δ aproximado (logit de medias)",
                     "desplazamiento del score δ·factor (puntos)", "por variable δ·factor/n",
                     "A vs referencia: decisiones distintas", "B vs referencia: decisiones distintas",
                     "B vs A: decisiones distintas", "PD media calibrada OOT", "PD verdadera media OOT (generador)"],
        "valor": [tasa_oot, _pd.mean(), delta_exacto, delta_aprox, desplazamiento, desplazamiento / _n,
                  int(np.sum(_ap_a != _ap_exacto)), int(np.sum(_ap_b != _ap_exacto)), int(np.sum(_ap_b != _ap_a)),
                  np.mean(1 / (1 + np.exp(-(_eta + delta_exacto)))), oot["pd_verdadera"].mean()]})
    assert abs(np.mean(1 / (1 + np.exp(-(_eta + delta_exacto)))) - tasa_oot) < 1e-10
    recalib.round(5)
    return delta_aprox, delta_exacto, desplazamiento, recalib, tasa_oot


@app.cell
def _(coma, delta_aprox, delta_exacto, desplazamiento, mo, recalib):
    _v = dict(zip(recalib["concepto"], recalib["valor"]))
    mo.md(f"""
    **Lectura.** δ exacto = {coma(delta_exacto, 4)} (aproximado {coma(delta_aprox, 4)}: la aproximación por promedios subestima, igual que
    en Austral, porque σ es convexa en la cola baja donde vive casi toda la cartera). El score calibrado es el score menos
    **{coma(desplazamiento, 2)} puntos** para todos ({coma(_v['por variable δ·factor/n'], 2)} por variable).
    Contra el score calibrado exacto, **A** difiere en {int(_v['A vs referencia: decisiones distintas'])} solicitudes (solo el redondeo
    de la tabla ya firmada) y **B** en {int(_v['B vs referencia: decisiones distintas'])}; entre sí, A y B discrepan en
    {int(_v['B vs A: decisiones distintas'])}. Restar {coma(_v['por variable δ·factor/n'], 2)} a cada fila y
    re-redondear reparte el desplazamiento de forma desigual entre combinaciones de bins. Además B obliga a re-firmar
    la tabla, re-generar reason codes (no cambian) y re-versionar el artefacto por un cambio que es de **nivel**, no de orden.
    Recomendación: tabla congelada; δ vive en el mapeo score → PD y el corte se gobierna en PD o se mueve explícitamente.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Cuándo falla: el ancla y el PDO son promesas de calibración

    El scaling **define** que 600 puntos ⇔ odds 50:1. Eso es verdad en la población donde el modelo está calibrado.
    Si la cartera se deteriora (el generador suma `deterioro` al log-odds de las cohortes 2025) la tabla sigue diciendo
    50:1 y la realidad dice otra cosa. Además, el **PDO efectivo** (−ln 2 / pendiente de una logística de `malo` sobre el score)
    es exactamente el nominal en DEV (ecuaciones de verosimilitud) y puede derivar fuera. Mueve el deterioro: DEV no cambia
    (misma semilla), OOT sí. El scorecard está **congelado** (ajustado arriba).
    """)
    return


@app.cell
def _(mo):
    deterioro_ui = mo.ui.slider(0.0, 1.0, step=0.05, value=0.35, label="Deterioro macro plantado en 2025 (log-odds)")
    deterioro_ui
    return (deterioro_ui,)


@app.cell
def _(dev, deterioro_ui, etiquetas_bins, factor, generar_cartera, irls_logit, np, offset, pd, pd_desde_score,
      pdo_ui, puntuar, scorecard):
    cartera_f = generar_cartera(deterioro=deterioro_ui.value)
    _dev_f = cartera_f[cartera_f["muestra"] == "DEV"].reset_index(drop=True)
    assert np.array_equal(_dev_f["malo"].to_numpy(), dev["malo"].to_numpy()), "DEV debe ser idéntico (misma semilla)"
    _oot_f = cartera_f[cartera_f["muestra"] == "OOT"].reset_index(drop=True)
    _, _s_dev = puntuar(etiquetas_bins(_dev_f), scorecard)
    _, _s_oot = puntuar(etiquetas_bins(_oot_f), scorecard)

    def pdo_efectivo(s, y):
        _b = irls_logit(((s - offset) / factor)[:, None], y)   # logit(malo) = a + b·(s−offset)/factor
        return -pdo_ui.value / _b[1], _b

    pdo_dev, coef_dev = pdo_efectivo(_s_dev, _dev_f["malo"].to_numpy())
    pdo_oot, coef_oot = pdo_efectivo(_s_oot, _oot_f["malo"].to_numpy())
    _bordes = np.arange(np.floor(np.percentile(_s_dev, 2) / 20) * 20, np.percentile(_s_dev, 98) + 20, 20)
    _filas = []
    for _lo, _hi in zip(_bordes[:-1], _bordes[1:]):
        _md, _mt = (_s_dev >= _lo) & (_s_dev < _hi), (_s_oot >= _lo) & (_s_oot < _hi)
        _filas.append({"banda": f"{_lo:.0f}–{_hi:.0f}",
                       "PD nominal (escala)": pd_desde_score(_s_oot[_mt], factor, offset).mean() if _mt.any() else np.nan,
                       "DEV observada": _dev_f.loc[_md, "malo"].mean(),
                       "OOT observada": _oot_f.loc[_mt, "malo"].mean(),
                       "OOT PD verdadera": _oot_f.loc[_mt, "pd_verdadera"].mean(), "n OOT": int(_mt.sum())})
    bandas_falla = pd.DataFrame(_filas)
    bandas_falla.round(4)
    return bandas_falla, cartera_f, coef_dev, coef_oot, pdo_dev, pdo_efectivo, pdo_oot


@app.cell
def _(bandas_falla, base_ui, coef_oot, coma, deterioro_ui, factor, mo, np, odds_ui, pdo_dev, pdo_oot, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _x = np.arange(len(bandas_falla))
    _lo = lambda p: np.log(np.clip(p, 1e-4, 1) / (1 - np.clip(p, 1e-4, 0.9999)))
    _ax.plot(_x, _lo(bandas_falla["PD nominal (escala)"]), "k-", label="nominal (tabla de puntos)")
    _ax.plot(_x, _lo(bandas_falla["DEV observada"]), "o", color="#4C72B0", label="DEV observada")
    _ax.plot(_x, _lo(bandas_falla["OOT observada"]), "s", color="#C44E52", label="OOT observada")
    _ax.plot(_x, _lo(bandas_falla["OOT PD verdadera"]), "--", color="#C44E52", label="OOT verdad del generador")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(bandas_falla["banda"], rotation=45, fontsize=7)
    _ax.set_xlabel("banda de score (puntos)")
    _ax.set_ylabel("log-odds de malo")
    _ax.set_title(f"Score congelado vs realidad (deterioro = {deterioro_ui.value:.2f})")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _gap = coef_oot[0] + (coef_oot[1] + 1) * np.log(odds_ui.value)   # log-odds real − nominal en el score base
    _odds_real = np.exp(-(coef_oot[0] + coef_oot[1] * np.log(odds_ui.value)))
    mo.vstack([_fig, mo.md(f"""
    **Lectura.** PDO efectivo en DEV = **{coma(pdo_dev, 3)}** (idéntico al nominal: la logística de `malo` sobre su propio score
    devuelve intercepto 0 y pendiente −1/factor, assert en checks). En OOT: PDO efectivo **{coma(pdo_oot, 2)}** y, en el score base,
    log-odds de malo {coma(_gap, 3)} más alto que el nominal (≈ {coma(_gap * factor, 1)} puntos): el ancla de la
    tabla ({base_ui.value} ⇔ {odds_ui.value}:1) corresponde en OOT a odds **{coma(_odds_real, 1)}:1**. El ranking (la pendiente) aguanta mucho mejor que el
    nivel: exactamente lo que la clase 4 corrige con δ. El scaling no calibra; solo **rotula** un log-odds.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. optbinning `Scorecard` vs la fórmula del curso

    `optbinning` hace binning óptimo propio (otros cortes), así que no comparamos puntos con nuestra tabla. Comparamos
    **la mecánica**: sobre SU matriz WoE ajustamos β con nuestro IRLS, aplicamos la fórmula del curso y exigimos que
    coincida con `Scorecard.table()` y con `Scorecard.score()`. Si optbinning no está instalado, la celda lo informa y sigue.
    """)
    return


@app.cell
def _(VARIABLES, dev, irls_logit, np, pd, warnings):
    def ajustar_optbinning(empirico):
        """Scorecard de optbinning con el scaling del curso. `empirico=False` = valores por defecto de la librería."""
        from optbinning import BinningProcess
        from optbinning.scorecard import Scorecard
        from sklearn.linear_model import LogisticRegression
        _tp = ({v: {"metric_special": "empirical", "metric_missing": "empirical"} for v in VARIABLES}
               if empirico else None)
        _bp = BinningProcess(variable_names=VARIABLES, binning_transform_params=_tp,
                             special_codes={"nunca_mora": [-9], "sin_bureau": [-99]})
        _sc = Scorecard(binning_process=_bp,
                        estimator=LogisticRegression(C=1e12, max_iter=10_000, tol=1e-12),   # ≈ sin penalización
                        scaling_method="pdo_odds",
                        scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600})
        _sc.fit(dev[VARIABLES], dev["malo"].astype(int).to_numpy())
        return _sc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.metrics import roc_auc_score
            _f = 20 / np.log(2)
            _o = 600 - _f * np.log(50)
            _n = len(VARIABLES)
            _y = dev["malo"].astype(int).to_numpy()
            _sc_emp, _sc_def = ajustar_optbinning(True), ajustar_optbinning(False)
            # (1) β: nuestro IRLS sobre la matriz WoE de optbinning vs sklearn
            _Xw = _sc_emp.binning_process_.transform(dev[VARIABLES], metric="woe")
            _b = irls_logit(_Xw[VARIABLES].to_numpy(), dev["malo"].to_numpy())
            _b_sk = np.r_[_sc_emp.estimator_.intercept_, _sc_emp.estimator_.coef_.ravel()]
            # (2) puntos: fórmula del curso fila a fila vs Points
            _td = _sc_emp.table(style="detailed").reset_index(drop=True)
            _coef = dict(zip(VARIABLES, _b[1:]))
            _pts_np = -(_td["Variable"].map(_coef) * _td["WoE"] + _b[0] / _n) * _f + _o / _n
            # (3) score: offset + factor·ln((1−p)/p)
            _p = _sc_emp.predict_proba(dev[VARIABLES])[:, 1]
            _s_ob = _sc_emp.score(dev[VARIABLES])
            # (4) la trampa: valores por defecto ⇒ especiales con WoE 0 (puntos neutros)
            _td_def = _sc_def.table(style="detailed").reset_index(drop=True)
            _mora = lambda t: t[t["Variable"] == "meses_desde_mora_12m"].set_index("Bin")
            _sb = dev["meses_desde_mora_12m"] == -99
            ob_resultado = {
                "disponible": True,
                "max |β IRLS − β sklearn|": float(np.max(np.abs(_b - _b_sk))),
                "max |puntos fórmula curso − Points|": float(np.max(np.abs(_pts_np - _td["Points"]))),
                "max |score() − (offset + factor·ln((1−p)/p))|": float(np.max(np.abs(_s_ob - (_o + _f * np.log((1 - _p) / _p))))),
                "WoE del bin de menor tasa": float(_td.loc[_td.loc[_td["Count"] > 0, "Event rate"].idxmin(), "WoE"]),
                "sin_bureau: tasa de malos": float(_mora(_td)["Event rate"]["sin_bureau"]),
                "sin_bureau: WoE en la tabla": float(_mora(_td)["WoE"]["sin_bureau"]),
                "sin_bureau: puntos (empirical)": float(_mora(_td)["Points"]["sin_bureau"]),
                "sin_bureau: puntos (por defecto)": float(_mora(_td_def)["Points"]["sin_bureau"]),
                "puntos neutros (WoE = 0) por defecto": float(_mora(_td_def)["Points"]["Missing"]),
                "Gini DEV (empirical)": float(2 * roc_auc_score(_y, _sc_emp.predict_proba(dev[VARIABLES])[:, 1]) - 1),
                "Gini DEV (por defecto)": float(2 * roc_auc_score(_y, _sc_def.predict_proba(dev[VARIABLES])[:, 1]) - 1),
                "score medio sin_bureau (empirical)": float(_s_ob[_sb].mean()),
                "score medio sin_bureau (por defecto)": float(_sc_def.score(dev[VARIABLES])[_sb].mean()),
            }
            ob_tabla = (_mora(_td)[["Count", "Event rate", "WoE", "Points"]]
                        .join(_mora(_td_def)[["Points"]].rename(columns={"Points": "Points (por defecto)"})).round(3))
    except Exception as _e:  # optbinning ausente o incompatible: el notebook sigue
        ob_resultado = {"disponible": False, "error": repr(_e)[:200]}
        ob_tabla = pd.DataFrame()
    pd.DataFrame({"valor": ob_resultado})
    return ajustar_optbinning, ob_resultado, ob_tabla


@app.cell
def _(coma, mo, ob_resultado, ob_tabla):
    if ob_resultado.get("disponible"):
        _r = ob_resultado
        _txt = mo.md(f"""
    **Lectura (1) — misma fórmula.** Con `C=1e12` (penalización despreciable), β de nuestro IRLS vs sklearn:
    {_r['max |β IRLS − β sklearn|']:.1e}; puntos de la fórmula del curso vs `Points`: {_r['max |puntos fórmula curso − Points|']:.1e};
    `score()` vs offset + factor·ln((1−p)/p): {_r['max |score() − (offset + factor·ln((1−p)/p))|']:.1e}. Es el reparto en partes iguales.
    Convenciones: el WoE de optbinning no usa el suavizado +0,5 del curso, y su signo coincide con el del curso en esta versión
    (bin de menor tasa: WoE {coma(_r['WoE del bin de menor tasa'], 3)} &gt; 0). Con el `LogisticRegression()` por defecto (C = 1, L2) los β se
    encogen y los puntos **no** coinciden con statsmodels.

    **Lectura (2) — la trampa de los especiales.** Con los valores por defecto, `transform` usa `metric_special=0`: el código −99
    (sin bureau, {coma(100 * _r['sin_bureau: tasa de malos'], 1)} % de malos, WoE {coma(_r['sin_bureau: WoE en la tabla'], 3)} en la **misma** tabla)
    recibe los puntos neutros {coma(_r['sin_bureau: puntos (por defecto)'], 1)} en vez de {coma(_r['sin_bureau: puntos (empirical)'], 1)}:
    el score medio de esos clientes sube de {coma(_r['score medio sin_bureau (empirical)'], 1)} a {coma(_r['score medio sin_bureau (por defecto)'], 1)}
    y el Gini DEV baja de {coma(_r['Gini DEV (empirical)'], 3)} a {coma(_r['Gini DEV (por defecto)'], 3)}. La tabla impresa **no** delata el problema
    (muestra el WoE empírico); solo la columna de puntos. Es la lámina 15 de la clase 4 hecha bug: el mapping del especial debe ser
    explícito en el artefacto. Nota: `binear()` del curso junta −9 y −99 en el bin (−inf, −9]; aquí se separan por diccionario.
    """)
    else:
        _txt = mo.md(f"optbinning no disponible en este entorno ({ob_resultado.get('error')}). La comparación queda descrita en el .md.")
    mo.vstack([_txt, ob_tabla])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. El scorecard como artefacto de datos

    Lo que se firma y se despliega no es el modelo: es una **tabla** `variable × bin × puntos_int` (+ cortes, + fila constante,
    + regla para bin no visto, + parámetros de escala) con un hash. Producción puntúa por **búsqueda**, sin β ni logit.
    La celda serializa el artefacto de forma canónica, calcula SHA-256 y verifica que puntuar por búsqueda en el artefacto
    reproduce el score entero del notebook.
    """)
    return


@app.cell
def _(B_todas, VARIABLES, base_ui, factor, hashlib, np, odds_ui, offset, pd, pdo_ui, reparto_ui, s_pts_int,
      tabla_entera):
    artefacto = (tabla_entera[["variable", "bin", "woe", "beta", "puntos", "puntos_int"]]
                 .assign(puntos=lambda d: d["puntos"].round(6), woe=lambda d: d["woe"].round(6),
                         beta=lambda d: d["beta"].round(8))
                 .sort_values(["variable", "bin"]).reset_index(drop=True))
    _neutros = (tabla_entera[tabla_entera["variable"] != "_constante"].groupby("variable")["base"].first()
                .round().astype(int))
    _otros = pd.DataFrame({"variable": _neutros.index, "bin": "_NO_VISTO", "woe": 0.0, "beta": np.nan,
                           "puntos": np.nan, "puntos_int": _neutros.to_numpy().astype(float)})
    artefacto = pd.concat([artefacto, _otros], ignore_index=True)
    _meta = f"pdo={pdo_ui.value};score_base={base_ui.value};odds_base={odds_ui.value};reparto={reparto_ui.value};" \
            f"factor={factor:.10f};offset={offset:.10f};version=M13-demo"
    _canon = _meta + "\n" + artefacto.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    hash_artefacto = hashlib.sha256(_canon.encode("utf-8")).hexdigest()

    def puntuar_artefacto(etiquetas, art):
        """Producción: solo búsqueda en la tabla entera; bin desconocido ⇒ fila _NO_VISTO."""
        total = np.zeros(len(etiquetas))
        for v in VARIABLES:
            sub = art[art["variable"] == v]
            mapa = dict(zip(sub["bin"], sub["puntos_int"]))
            total += etiquetas[v].map(mapa).fillna(mapa["_NO_VISTO"]).to_numpy()
        return total + art.loc[art["variable"] == "_constante", "puntos_int"].iloc[0]

    s_artefacto = puntuar_artefacto(B_todas, artefacto)
    assert np.array_equal(s_artefacto, s_pts_int), "artefacto ≠ score entero del notebook"
    assert np.all(s_artefacto == np.round(s_artefacto)), "score del artefacto debe ser entero"
    artefacto
    return artefacto, hash_artefacto, puntuar_artefacto, s_artefacto


@app.cell
def _(artefacto, hash_artefacto, mo):
    mo.md(f"""
    **Lectura.** Artefacto de {len(artefacto)} filas; SHA-256 = `{hash_artefacto[:16]}…` (cambia con cualquier control de escala
    o reparto: es la huella que se registra en el expediente, M21–M22). Invariantes que un CI debe exigir: suma de puntos = score entero;
    |score entero − transformación del logit| ≤ cota de redondeo; una fila `_NO_VISTO` por variable; una única fila `_constante`;
    los rangos por variable no dependen del reparto.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Checks del módulo
    Si falla uno, el notebook falla.
    """)
    return


@app.cell
def _(B_dev, REPARTOS, VARIABLES, W_dev, beta, coef_dev, comparacion_repartos, eta_de, eps, irls_logit, mapas,
      modelo_sm, np, ob_resultado, parametros_scaling, pd_desde_score, pdo_dev, pdo_ui, puntuar, s_artefacto,
      s_pts_int, score_desde_eta, tabla_puntos, verif_suma, y_dev, delta_exacto, tasa_oot, factor, offset):
    # 1. numpy vs statsmodels
    assert np.allclose(beta.to_numpy(), modelo_sm.params.to_numpy(), atol=1e-7)
    assert (beta[VARIABLES] < 0).all(), "convención WoE: β negativos"
    # 2. constantes del curso
    _f, _o = parametros_scaling(20, 600, 50)
    assert np.isclose(_f, 28.8539, atol=1e-4) and np.isclose(_o, 487.1229, atol=1e-4)
    _base = _o / 8 - (-3.025210 / 8) * _f
    assert np.isclose(_base, 71.80, atol=0.01)
    assert np.isclose(_base + 0.352515 * _f * 4.26, 115.1, atol=0.05)          # uso_tc_prom_12m, WoE 4,26
    assert np.isclose(_base + 0.352515 * _f * -1.23, 59.3, atol=0.05)
    assert np.isclose(1 / (1 + np.exp((603.8 - _o) / _f)), 0.0172, atol=5e-4)  # S0030427: 604 ⇔ PD 1,7 %
    assert np.isclose(0.177 * _f, 5.107, atol=1e-3)                            # δ PIT Austral en puntos
    # 3. score = suma de puntos = transformación del logit, todos los repartos y muestras
    assert verif_suma["max |suma − logit|"].max() < 1e-9
    # 4. la lectura 580/600/620 de la escala del curso
    for _s, _pd in [(580, 1 / 26), (600, 1 / 51), (620, 1 / 101)]:
        assert np.isclose(pd_desde_score(_s, _f, _o), _pd)
    # 5. invariancia de reason codes por brecha
    assert (comparacion_repartos["% TTD top-3 brecha ≠ curso"] == 0).all()
    # 6. redondeo: cota n/2 y artefacto
    assert np.abs(eps).max() <= (len(VARIABLES) + 1) / 2
    assert np.array_equal(s_artefacto, s_pts_int)
    # 7. PDO efectivo en DEV = nominal y δ exacto calza la tasa
    assert np.isclose(pdo_dev, pdo_ui.value, rtol=1e-6) and abs(coef_dev[0]) < 1e-6
    assert np.isfinite(delta_exacto) and 0 < tasa_oot < 1
    # 8. Gini invariante a la escala (score y −η ordenan igual)
    _s = score_desde_eta(eta_de(W_dev), factor, offset)
    assert np.all(np.diff(_s[np.argsort(-eta_de(W_dev), kind="stable")]) >= -1e-9)
    # 9. optbinning (si está): misma fórmula
    if ob_resultado.get("disponible"):
        assert ob_resultado["max |β IRLS − β sklearn|"] < 1e-4
        assert ob_resultado["max |puntos fórmula curso − Points|"] < 1e-3
        assert ob_resultado["max |score() − (offset + factor·ln((1−p)/p))|"] < 1e-6
    "✅ Todos los checks del módulo pasan"
    return


if __name__ == "__main__":
    app.run()

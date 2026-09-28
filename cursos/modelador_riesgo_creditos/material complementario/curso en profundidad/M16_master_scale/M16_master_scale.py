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
    from itertools import combinations
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import expit, logit
    from scipy.stats import beta as beta_dist
    from scipy.stats import binom, norm
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.proportion import proportion_confint, proportions_ztest
    return (
        beta_dist,
        binom,
        brentq,
        combinations,
        expit,
        logit,
        mo,
        norm,
        plt,
        proportion_confint,
        proportions_ztest,
        quad,
        roc_auc_score,
        sm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # M16 · Master scale

    **Serie 2 · Del embudo al gobierno.** Profundiza la clase 4 (parte 2, láminas 34–36 de la v21) y la
    nivelación de la clase 5 (láminas 5–6). En clase: 8 bandas A1…E ancladas al PDO (cada 20 puntos
    duplica las odds; 600 = 50:1 = PD 1,96%), una tabla contra los datos y tres requisitos
    (monótona, sin bandas vacías ni concentradas, estable). Aquí se construye la escala como un
    **problema de diseño con restricciones medibles**.

    Qué hace este notebook:

    1. Pipeline mínimo sobre Banco Sintético (verdad conocida) y la escala de Banco Austral como ancla.
    2. **Cuatro diseños**: por PDO (geométrica en odds), por PD objetivo (geométrica en PD), por cuantiles
       y por optimización (programación dinámica con restricciones), con sus métricas.
    3. **Asignación de PD a la banda**: media aritmética, punto medio geométrico, media exacta bajo logit
       uniforme y tasa observada suavizada; efecto en la pérdida esperada.
    4. **Requisitos cuantitativos**: monotonía, concentración (HHI y el índice del BCE), separabilidad
       (test de proporciones, numpy vs `statsmodels`) y **potencia del backtest por banda**.
    5. Granularidad vs estabilidad, matriz de migración, recalibración (las bandas se mueven δ·factor) y
       mapeo de un segundo modelo a la escala corporativa.
    6. Experimentos de falla con verdad conocida y checks automáticos.

    Convenciones del curso: target **1 = malo**; PDO 20, 600 ⇔ 50:1, factor = 28,8539, offset = 487,1229;
    un score exactamente en un corte cae en la banda **superior** (`right=False`, como en la demo).
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
    ## 1. Pipeline mínimo: scorecard, calibración y score

    Scorecard corto (6 variables, binning y WoE del curso, logística sobre WoE) ajustado en DEV. Se
    calibra el intercepto (M15) a la tasa de DEV+HO, que hace de **tendencia central** del periodo
    pre-2025: $p^{cal}_i=\sigma(\eta_i+\delta)$ con δ exacto. El score oficial es el **calibrado**:
    $s_i = \text{offset} - \text{factor}\,(\eta_i+\delta)$, igual que `SCORE_CAL` en la demo de la clase 4.

    Columnas que la vida real no tiene: `pd_real` (la PD verdadera completa del generador, incluyendo el
    10% extra de malos que se planta a los «sin bureau») y, en TTD, `malo_futuro` simulado desde
    `pd_real` (el desempeño que se observará). OOT trae el deterioro plantado (+0,35 en log-odds).
    """)
    return


@app.cell
def _(a_woe, brentq, expit, generar_cartera, np, pd, sm, tabla_woe):
    FACTOR = 20 / np.log(2)
    OFFSET = 600 - FACTOR * np.log(50)

    cartera = generar_cartera()
    cartera["pd_real"] = np.where(
        cartera["meses_desde_mora_12m"] == -99,
        cartera["pd_verdadera"] + (1 - cartera["pd_verdadera"]) * 0.10,
        cartera["pd_verdadera"],
    )
    _rng = np.random.default_rng(1616)
    cartera["malo_futuro"] = np.where(
        cartera["muestra"] == "TTD",
        (_rng.random(len(cartera)) < cartera["pd_real"]).astype(float),
        cartera["malo"],
    )
    VARIABLES = ["uso_linea_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m",
                 "antiguedad_meses", "carga_financiera", "consultas_6m"]
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    mapas_woe = {v: tabla_woe(dev[v], dev["malo"])[0]["woe"].to_dict() for v in VARIABLES}
    modelo = sm.Logit(dev["malo"].values, sm.add_constant(a_woe(dev, VARIABLES, dev, mapas_woe))).fit(disp=0)

    def _lp(df_):
        _X = sm.add_constant(a_woe(df_, VARIABLES, dev, mapas_woe), has_constant="add")
        return np.asarray(_X @ modelo.params, dtype=float)

    _partes = {m: cartera[cartera["muestra"] == m].reset_index(drop=True).copy()
               for m in ["DEV", "HO", "OOT", "TTD"]}
    for _m, _d in _partes.items():
        _d["lp"] = _lp(_d)
    _cal = pd.concat([_partes["DEV"], _partes["HO"]])
    TC_SINT = float(_cal["malo"].mean())
    DELTA_TC = brentq(lambda d: expit(_cal["lp"].values + d).mean() - TC_SINT, -5, 5)
    muestras = {}
    for _m, _d in _partes.items():
        _d["pd_cal"] = expit(_d["lp"].values + DELTA_TC)
        _d["score"] = OFFSET - FACTOR * (_d["lp"].values + DELTA_TC)
        muestras[_m] = _d
    resumen_muestras = pd.DataFrame({
        m: {"n": len(d), "pd_cal_media": d["pd_cal"].mean(), "tasa_observada": d["malo_futuro"].mean(),
            "pd_real_media": d["pd_real"].mean(), "score_p05": np.percentile(d["score"], 5),
            "score_p50": np.percentile(d["score"], 50), "score_p95": np.percentile(d["score"], 95)}
        for m, d in muestras.items()}).T
    resumen_muestras.round(4)
    return (
        DELTA_TC,
        FACTOR,
        OFFSET,
        TC_SINT,
        VARIABLES,
        cartera,
        dev,
        mapas_woe,
        muestras,
        resumen_muestras,
    )


@app.cell
def _(DELTA_TC, FACTOR, TC_SINT, coma, mo, resumen_muestras):
    _r = resumen_muestras
    mo.md(f"""
    **Lectura.** Tendencia central (DEV+HO) = {coma(100 * TC_SINT, 2)}%; δ exacto = {coma(DELTA_TC, 4)}
    ({coma(DELTA_TC * FACTOR, 2)} puntos). Banco Sintético es **mucho más riesgoso que Austral** (≈ 11–12% vs 5,4%)
    y su score vive entre ≈ {_r.loc['DEV', 'score_p05']:.0f} (p5) y {_r.loc['DEV', 'score_p95']:.0f} (p95):
    la escala del curso (cortes 540…660) dejaría A1, A2 y B1 casi vacías. Primera lección: **los cortes de
    una master scale no se copian entre carteras**; lo que se copia es la *regla* (ancho en log-odds, ancla).
    En OOT la tasa observada ({coma(100 * _r.loc['OOT', 'tasa_observada'], 2)}%) supera la PD calibrada
    ({coma(100 * _r.loc['OOT', 'pd_cal_media'], 2)}%): el deterioro plantado, que la escala heredará.
    """)
    return


@app.cell
def _(FACTOR, OFFSET, norm, np, pd, roc_auc_score):
    # ---------------------------------------------------------------------------
    # Herramientas de master scale (numpy puro salvo roc_auc para el contraste)
    # Convención interna: banda 0 = PEOR (score más bajo) … K-1 = MEJOR.
    # ---------------------------------------------------------------------------
    NOMBRES_AUSTRAL = ["E", "D", "C2", "C1", "B2", "B1", "A2", "A1"]   # peor → mejor

    def coma(x, dec=2):
        """Formato de prosa: decimales con coma."""
        return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def nombres_bandas(K):
        return NOMBRES_AUSTRAL if K == 8 else [f"R{K - k}" for k in range(K)]

    def asignar(score, cortes):
        """Banda de cada score. right=False del curso: en el corte exacto cae en la banda SUPERIOR."""
        return np.searchsorted(np.asarray(cortes, dtype=float), np.asarray(score, dtype=float), side="right")

    def score_a_pd(s):
        return 1.0 / (1.0 + np.exp((np.asarray(s, dtype=float) - OFFSET) / FACTOR))

    def pd_a_score(p):
        p = np.asarray(p, dtype=float)
        return OFFSET + FACTOR * np.log((1 - p) / p)

    def hhi(shares):
        s = np.asarray(shares, dtype=float)
        return float(np.sum(s ** 2))

    def indice_bce(shares):
        """Índice de Herfindahl normalizado del BCE: HI = 1 + ln(HHI)/ln(K) ∈ [0, 1]; CV = sqrt(K Σ(s−1/K)²)."""
        s = np.asarray(shares, dtype=float)
        K = len(s)
        cv = np.sqrt(K * np.sum((s - 1 / K) ** 2))
        hi = 1 + np.log((cv ** 2 + 1) / K) / np.log(K)
        return float(hi), float(cv)

    def ztest_adyacentes(n, malos):
        """z de dos proporciones (pooled) entre bandas adyacentes; H1: la banda PEOR tiene tasa mayor.
        Devuelve z y p unilateral para cada par (k, k+1), con k la banda peor."""
        n = np.asarray(n, dtype=float)
        m = np.asarray(malos, dtype=float)
        n1, n2, x1, x2 = n[:-1], n[1:], m[:-1], m[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            pp = (x1 + x2) / (n1 + n2)
            se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
            z = (x1 / n1 - x2 / n2) / se
        z = np.where(np.isfinite(z), z, 0.0)
        return z, norm.sf(z)

    def gini_bandas(banda, y):
        """Gini exacto de un score discreto (empates = ½), desde conteos. Banda baja = más riesgo."""
        banda = np.asarray(banda)
        y = np.asarray(y, dtype=float)
        K = int(banda.max()) + 1
        b = np.bincount(banda, weights=y, minlength=K)
        g = np.bincount(banda, weights=1 - y, minlength=K)
        g_mejores = np.concatenate([np.cumsum(g[::-1])[::-1][1:], [0.0]])   # buenos en bandas mejores
        auc = (np.sum(b * g_mejores) + 0.5 * np.sum(b * g)) / (b.sum() * g.sum())
        return 2 * auc - 1

    def tabla_escala(score, y, pd_cal, cortes, pd_real=None):
        """Tabla por banda (orden interno peor → mejor)."""
        K = len(cortes) + 1
        b = asignar(score, cortes)
        n = np.bincount(b, minlength=K).astype(float)
        m = np.bincount(b, weights=y, minlength=K)
        spd = np.bincount(b, weights=pd_cal, minlength=K)
        lo = np.concatenate([[-np.inf], cortes])
        hi = np.concatenate([cortes, [np.inf]])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = pd.DataFrame({
                "banda": nombres_bandas(K), "score_desde": lo, "score_hasta": hi, "n": n, "malos": m,
                "pct_pob": n / n.sum(), "tasa_obs": m / n, "pd_cal_media": spd / n,
                "pd_en_piso": score_a_pd(lo), "pd_en_techo": score_a_pd(hi)})
            if pd_real is not None:
                t["pd_real_media"] = np.bincount(b, weights=pd_real, minlength=K) / n
        return t

    def metricas_escala(score, y, pd_cal, cortes, alfa=0.05):
        t = tabla_escala(score, y, pd_cal, cortes)
        K = len(t)
        _, p_adj = ztest_adyacentes(t["n"], t["malos"])
        tasas = t["tasa_obs"].to_numpy()
        llenas = t["n"].to_numpy() > 0
        inv = int(np.sum(np.diff(tasas[llenas]) >= 0))            # debe BAJAR al subir de banda
        hi_, _ = indice_bce(t["pct_pob"])
        return {
            "K": K, "bandas_vacias": int((~llenas).sum()), "HHI": hhi(t["pct_pob"]), "HHI_min=1/K": 1 / K,
            "HI_BCE": hi_, "max_%_pob": float(t["pct_pob"].max()), "min_n": int(t["n"].min()),
            "min_malos": int(t["malos"].min()), "inversiones": inv,
            "pares_no_separables": int(np.sum(p_adj[llenas[:-1] & llenas[1:]] > alfa)),
            "gini_bandas": gini_bandas(asignar(score, cortes), y),
        }

    def gini_continuo(y, s):
        return 2 * roc_auc_score(y, -np.asarray(s)) - 1

    return (
        asignar,
        coma,
        gini_bandas,
        gini_continuo,
        hhi,
        indice_bce,
        metricas_escala,
        nombres_bandas,
        pd_a_score,
        score_a_pd,
        tabla_escala,
        ztest_adyacentes,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. La escala de Banco Austral, re-auditada

    La tabla de la clase 4 (DEV+HO+OOT, 6.723 créditos) y el backtest OOT de la clase 5. Los malos por banda
    se reconstruyen como `round(tasa × n)` (2, 4, 8, 17, 30, 33, 60, 208 = 362 malos, 5,38%): reproducen
    las tasas publicadas al centésimo. Qué mirar:

    - **Concentración**: HHI y el índice normalizado del BCE contra el mínimo 1/K.
    - **La tasa observada contra el rango de PD que la banda promete** (PD en el piso y en el techo).
    - **Separabilidad**: ¿cada banda es estadísticamente peor que la de arriba? (z de dos proporciones,
      numpy vs `statsmodels.proportions_ztest`).
    - **Potencia** del backtest binomial OOT para detectar que la PD real sea el doble de la prometida.
    """)
    return


@app.cell
def _(beta_dist, binom, hhi, indice_bce, norm, np, pd, proportion_confint, proportions_ztest, score_a_pd,
      ztest_adyacentes):
    # --- Datos del curso (clase 4, lámina 36 v21 / clase 5, lámina 6), orden MEJOR → PEOR como en clase ---
    austral = pd.DataFrame({
        "banda": ["A1", "A2", "B1", "B2", "C1", "C2", "D", "E"],
        "score_desde": [660, 640, 620, 600, 580, 560, 540, -np.inf],
        "score_hasta": [np.inf, 660, 640, 620, 600, 580, 560, 540],
        "n": [1446, 655, 774, 829, 810, 744, 628, 837],
        "pd_cal_media": [0.0011, 0.0036, 0.0072, 0.0141, 0.0279, 0.0543, 0.1016, 0.2675],
        "tasa_publicada": [0.0014, 0.0061, 0.0103, 0.0205, 0.0370, 0.0444, 0.0955, 0.2485],
        "pct_modelacion": [0.2151, 0.0974, 0.1151, 0.1233, 0.1205, 0.1107, 0.0934, 0.1245],
        "pct_ttd": [0.1843, 0.0895, 0.1075, 0.1216, 0.1231, 0.1203, 0.1090, 0.1447],
    })
    austral["malos"] = np.round(austral["tasa_publicada"] * austral["n"]).astype(int)
    austral["tasa_obs"] = austral["malos"] / austral["n"]
    austral["pd_en_piso"] = score_a_pd(austral["score_desde"])
    austral["pd_en_techo"] = score_a_pd(austral["score_hasta"])
    austral["dentro_del_rango"] = (austral["tasa_obs"] >= austral["pd_en_techo"]) & (
        austral["tasa_obs"] <= austral["pd_en_piso"])
    _jl, _ju = proportion_confint(austral["malos"], austral["n"], alpha=0.10, method="jeffreys")
    austral["jeffreys90_lo"], austral["jeffreys90_hi"] = _jl, _ju
    _jl_np = beta_dist.ppf(0.05, austral["malos"] + 0.5, austral["n"] - austral["malos"] + 0.5)
    _ju_np = beta_dist.ppf(0.95, austral["malos"] + 0.5, austral["n"] - austral["malos"] + 0.5)

    # --- concentración ---
    hhi_austral_mod = hhi(austral["pct_modelacion"])
    hhi_austral_ttd = hhi(austral["pct_ttd"])
    hi_mod, cv_mod = indice_bce(austral["pct_modelacion"])
    hi_ttd, cv_ttd = indice_bce(austral["pct_ttd"])
    # test del BCE (instrucciones de reporte de validación, 2019): H0 «la concentración no subió»
    _K = 8
    s_bce = np.sqrt(_K - 1) * (cv_ttd - cv_mod) / np.sqrt(cv_ttd ** 2 * (0.5 + cv_ttd ** 2))
    p_bce = float(norm.sf(s_bce))

    # --- separabilidad: pares adyacentes, en orden interno PEOR → MEJOR ---
    _n_pm = austral["n"].to_numpy()[::-1]
    _m_pm = austral["malos"].to_numpy()[::-1]
    _z, _p = ztest_adyacentes(_n_pm, _m_pm)
    _z_sm, _p_sm = [], []
    for _k in range(7):
        _zz, _pp = proportions_ztest(count=[_m_pm[_k], _m_pm[_k + 1]], nobs=[_n_pm[_k], _n_pm[_k + 1]],
                                     alternative="larger")
        _z_sm.append(_zz)
        _p_sm.append(_pp)
    _nb = austral["banda"].to_numpy()[::-1]
    separabilidad_austral = pd.DataFrame({
        "par (peor vs mejor)": [f"{_nb[k]} vs {_nb[k + 1]}" for k in range(7)],
        "tasa_peor": _m_pm[:-1] / _n_pm[:-1], "tasa_mejor": _m_pm[1:] / _n_pm[1:],
        "z_numpy": _z, "p_numpy": _p, "z_statsmodels": _z_sm, "p_statsmodels": _p_sm})

    # --- backtest OOT de la clase 5 y su potencia ---
    oot_austral = pd.DataFrame({
        "banda": ["A1", "A2", "B1", "B2", "C1", "C2", "D", "E"],
        "n": [432, 199, 244, 235, 239, 220, 172, 263],
        "pd_cal": [0.0011, 0.0036, 0.0072, 0.0142, 0.0280, 0.0544, 0.1021, 0.2691],
        "malos": [1, 3, 4, 8, 8, 9, 22, 64]})
    oot_austral["esperados"] = oot_austral["n"] * oot_austral["pd_cal"]
    oot_austral["p_cola_sup"] = binom.sf(oot_austral["malos"] - 1, oot_austral["n"], oot_austral["pd_cal"])

    def potencia_binomial(n, p0, mult, alfa=0.05):
        """Potencia EXACTA del test binomial unilateral superior (nivel ≤ alfa) si la PD real es mult·p0."""
        n = int(n)
        kc = int(binom.isf(alfa, n, p0)) + 1           # menor k con P(D ≥ k | p0) ≤ alfa
        while kc > 0 and binom.sf(kc - 2, n, p0) <= alfa:
            kc -= 1
        while binom.sf(kc - 1, n, p0) > alfa:
            kc += 1
        return float(binom.sf(kc - 1, n, min(mult * p0, 1.0))), kc

    oot_austral["potencia_x2"] = [potencia_binomial(r.n, r.pd_cal, 2.0)[0] for r in oot_austral.itertuples()]
    oot_austral["potencia_x1.5"] = [potencia_binomial(r.n, r.pd_cal, 1.5)[0] for r in oot_austral.itertuples()]
    oot_austral["n_min_x2"] = np.ceil(8.04 / oot_austral["pd_cal"]).astype(int)
    austral_jeffreys_np = (np.asarray(_jl_np), np.asarray(_ju_np))
    austral.drop(columns=["score_desde", "score_hasta"]).round(4)
    return (
        austral,
        austral_jeffreys_np,
        cv_mod,
        cv_ttd,
        hhi_austral_mod,
        hhi_austral_ttd,
        hi_mod,
        hi_ttd,
        oot_austral,
        p_bce,
        potencia_binomial,
        separabilidad_austral,
    )


@app.cell
def _(separabilidad_austral):
    separabilidad_austral.round(4)
    return


@app.cell
def _(oot_austral):
    oot_austral.round(4)
    return


@app.cell
def _(austral, coma, hhi_austral_mod, hhi_austral_ttd, hi_mod, hi_ttd, mo, oot_austral, p_bce,
      separabilidad_austral):
    _fuera = austral.loc[~austral["dentro_del_rango"], "banda"].tolist()
    _nosep = separabilidad_austral.loc[separabilidad_austral["p_numpy"] > 0.05, "par (peor vs mejor)"].tolist()
    _o = oot_austral.set_index("banda")
    mo.md(f"""
    **Lectura.**

    - **Concentración.** HHI modelación = {coma(hhi_austral_mod, 4)} (mínimo posible 1/8 = 0,125; «número
      equivalente de bandas» 1/HHI = {coma(1 / hhi_austral_mod, 2)}); TTD = {coma(hhi_austral_ttd, 4)}. Índice del
      BCE: {coma(hi_mod, 4)} → {coma(hi_ttd, 4)}. El test del BCE de «¿subió la concentración?» da p = {coma(p_bce, 3)}:
      la TTD está *menos* concentrada (A1 perdió peso). La escala no tiene problema de concentración; A1 con
      21,5% es la banda más cargada, y es abierta.
    - **Tasa fuera del rango prometido**: {', '.join(_fuera)}. En A2, B1 y B2 la tasa observada supera la PD
      del *piso* de la banda (p. ej. B2: 2,05% observado vs PD de 1,96% en 600). La escala es monótona, pero el
      nivel de las bandas buenas está subestimado: es la misma huella que la clase 5 encontró en OOT.
    - **Separabilidad** (α = 5%, unilateral): pares NO separables = {', '.join(_nosep) if _nosep else 'ninguno'}.
      Monótona no es lo mismo que separable: C2 vs C1 (4,44% vs 3,70%) tiene p = {coma(separabilidad_austral.iloc[2]['p_numpy'], 3)}.
      numpy y `statsmodels` coinciden (check al final).
    - **Potencia OOT**: para detectar que la PD real de A1 sea el **doble** de 0,11% con 432 créditos, el test
      binomial tiene potencia {coma(100 * _o.loc['A1', 'potencia_x2'], 1)}%; A2 {coma(100 * _o.loc['A2', 'potencia_x2'], 1)}%,
      B2 {coma(100 * _o.loc['B2', 'potencia_x2'], 1)}%, C2 {coma(100 * _o.loc['C2', 'potencia_x2'], 1)}%. Hace falta ≈ 8
      malos *esperados* por banda (§6): en A1 eso son ≈ {coma(_o.loc['A1', 'n_min_x2'], 0)} créditos. Un 🟢 en A1 no
      certifica nada.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Cuatro diseños de escala

    Los cuatro producen **cortes en el score calibrado**; lo que cambia es el criterio:

    | Diseño | Regla | Qué fija | Qué queda libre |
    |---|---|---|---|
    | PDO | cortes en $600 + w\cdot j$ | ancho constante en log-odds ($w/\text{PDO}$ duplicaciones) | ocupación, nº de bandas útiles |
    | PD objetivo | límites de PD en progresión geométrica entre $PD_{min}$ y $PD_{max}$ | razón constante de PD entre límites | ocupación |
    | Cuantiles | igual % de población | concentración mínima (HHI = 1/K) | anchos de PD, separabilidad |
    | Óptimo | máx. verosimilitud por programación dinámica con restricciones | pérdida de información mínima | forma de los cortes |

    En el diseño PDO el desplazamiento $j$ inicial no viene dado: se elige el que minimiza el HHI en DEV
    (declarado). Los controles de abajo cambian K (todos los diseños), el ancho en puntos y el PDO **de la
    escala en que se expresa el score**: verás que solo importa el cociente $w/\text{PDO}$.
    """)
    return


@app.cell
def _(mo):
    k_ui = mo.ui.slider(4, 14, step=1, value=8, label="Número de bandas K")
    ancho_ui = mo.ui.slider(10, 40, step=5, value=20, label="Ancho de banda (puntos de la escala)")
    pdo_ui = mo.ui.slider(10, 60, step=5, value=20, label="PDO de la escala del score")
    alfa_ui = mo.ui.dropdown({"0,10": 0.10, "0,05": 0.05, "0,01": 0.01}, value="0,05",
                             label="α de separabilidad (unilateral)")
    mo.vstack([mo.hstack([k_ui, alfa_ui]), mo.hstack([ancho_ui, pdo_ui])])
    return alfa_ui, ancho_ui, k_ui, pdo_ui


@app.cell
def _(asignar, np, pd_a_score):
    def cortes_pdo(score, K, ancho, pdo=20.0, ancla=600.0):
        """Escala geométrica en odds. El score se re-expresa en una escala con `pdo` (misma ancla 600 ⇔ 50:1):
        s' = 600 + (pdo/20)(s − 600). Cortes en 600 + ancho·j; j0 elegido para minimizar el HHI.
        Devuelve (cortes en la escala ORIGINAL, cortes en la escala re-expresada)."""
        s2 = ancla + (pdo / 20.0) * (np.asarray(score) - ancla)
        mejor = None
        for j0 in range(-60, 60):
            c2 = ancla + ancho * (np.arange(K - 1) + j0)
            sh = np.bincount(asignar(s2, c2), minlength=K) / len(s2)
            h = float(np.sum(sh ** 2))
            if mejor is None or h < mejor[0] - 1e-15:
                mejor = (h, c2)
        c2 = mejor[1]
        return ancla + (20.0 / pdo) * (c2 - ancla), c2

    def cortes_pd_geometrica(K, pd_min=0.015, pd_max=0.40):
        """Límites de PD en progresión geométrica (tipo escala de rating), convertidos a score."""
        limites_pd = np.geomspace(pd_max, pd_min, K - 1)       # de la banda peor a la mejor
        return pd_a_score(limites_pd), limites_pd

    def cortes_cuantil(score, K):
        return np.quantile(np.asarray(score, dtype=float), np.arange(1, K) / K)

    return cortes_cuantil, cortes_pd_geometrica, cortes_pdo


@app.cell
def _(asignar, norm, np):
    def _ll(n, b):
        """Log-verosimilitud binomial de un bloque con PD = su tasa (0·log0 = 0)."""
        n = np.asarray(n, dtype=float)
        b = np.asarray(b, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = b / n
            v = b * np.log(r) + (n - b) * np.log1p(-r)
        return np.where((b > 0) & (b < n), v, 0.0)

    def bloques_finos(score, y, m_fino):
        q = np.unique(np.quantile(np.asarray(score, dtype=float), np.arange(1, m_fino) / m_fino))
        f = asignar(score, q)
        mf = len(q) + 1
        nf = np.bincount(f, minlength=mf).astype(float)
        bf = np.bincount(f, weights=y, minlength=mf)
        return q, nf, bf

    def dp_escala(score, y, K, m_fino=30, min_pct=0.03, min_malos=10, alfa=0.05, restringida=True):
        """Partición ÓPTIMA de los bloques finos (ordenados por score) en K bandas contiguas que maximiza
        la log-verosimilitud binomial (= minimiza la pérdida de información, ver §3 del .md), sujeta a:
        cada banda con ≥ min_pct de la población y ≥ min_malos; y, si `restringida`, cada banda
        significativamente PEOR que la siguiente (z unilateral > z_{1−α}), lo que implica monotonía.
        Estado: D[k, i, j] = mejor valor con k bandas cubriendo [0, j) y la última = [i, j).
        Devuelve (cortes, valor) o (None, -inf) si no existe partición factible."""
        q, nf, bf = bloques_finos(score, y, m_fino)
        mf = len(nf)
        Cn = np.concatenate([[0.0], np.cumsum(nf)])
        Cb = np.concatenate([[0.0], np.cumsum(bf)])
        N = Cn[-1]
        NB = Cn[None, :] - Cn[:, None]
        BB = Cb[None, :] - Cb[:, None]
        ok = (NB >= min_pct * N) & (BB >= min_malos)
        V = np.where(ok, _ll(np.where(NB > 0, NB, 1.0), BB), -np.inf)
        zc = norm.isf(alfa) if restringida else -np.inf
        D = np.full((K + 1, mf + 1, mf + 1), -np.inf)
        P = np.full((K + 1, mf + 1, mf + 1), -1, dtype=int)
        D[1, 0, 1:] = V[0, 1:]
        for k in range(2, K + 1):
            for i in range(1, mf):
                h = np.arange(0, i)
                prev = D[k - 1, h, i]
                if not np.isfinite(prev).any():
                    continue
                for j in range(i + 1, mf + 1):
                    if not ok[i, j]:
                        continue
                    n1, b1, n2, b2 = NB[h, i], BB[h, i], NB[i, j], BB[i, j]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pp = (b1 + b2) / (n1 + n2)
                        z = (b1 / n1 - b2 / n2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
                    cand = np.where(np.isfinite(prev) & (np.nan_to_num(z, nan=-np.inf) > zc), prev, -np.inf)
                    a = int(np.argmax(cand))
                    if np.isfinite(cand[a]):
                        D[k, i, j] = cand[a] + V[i, j]
                        P[k, i, j] = h[a]
        if K == 1:
            return np.array([]), float(V[0, mf])
        i = int(np.argmax(D[K, :, mf]))
        if not np.isfinite(D[K, i, mf]):
            return None, -np.inf
        valor = float(D[K, i, mf])
        lim, j, k = [], mf, K
        while k > 1:
            lim.append(i)
            h = P[k, i, j]
            j, i, k = i, h, k - 1
        return q[np.array(sorted(lim)) - 1], valor

    def fuerza_bruta_escala(score, y, K, m_fino=10, min_pct=0.03, min_malos=10, alfa=0.05, restringida=True,
                            combinaciones=None):
        """Misma optimización por enumeración exhaustiva de todas las particiones (solo para m_fino chico)."""
        q, nf, bf = bloques_finos(score, y, m_fino)
        mf = len(nf)
        N = nf.sum()
        zc = norm.isf(alfa) if restringida else -np.inf
        mejor = (-np.inf, None)
        for comb in combinaciones(range(1, mf), K - 1):
            bordes = [0, *comb, mf]
            n = np.array([nf[bordes[t]:bordes[t + 1]].sum() for t in range(K)])
            b = np.array([bf[bordes[t]:bordes[t + 1]].sum() for t in range(K)])
            if (n < min_pct * N).any() or (b < min_malos).any():
                continue
            pp = (b[:-1] + b[1:]) / (n[:-1] + n[1:])
            z = (b[:-1] / n[:-1] - b[1:] / n[1:]) / np.sqrt(pp * (1 - pp) * (1 / n[:-1] + 1 / n[1:]))
            if (z <= zc).any():
                continue
            v = float(_ll(n, b).sum())
            if v > mejor[0]:
                mejor = (v, q[np.array(comb) - 1])
        return mejor[1], mejor[0]

    return dp_escala, fuerza_bruta_escala


@app.cell
def _(
    alfa_ui,
    ancho_ui,
    asignar,
    cortes_cuantil,
    cortes_pd_geometrica,
    cortes_pdo,
    dp_escala,
    k_ui,
    metricas_escala,
    muestras,
    np,
    pd,
    pdo_ui,
):
    _d = muestras["DEV"]
    s_dev, y_dev, p_dev = _d["score"].to_numpy(), _d["malo"].to_numpy(), _d["pd_cal"].to_numpy()
    K_SEL = k_ui.value
    cortes_pdo_sel, cortes_pdo_escala = cortes_pdo(s_dev, K_SEL, ancho_ui.value, pdo_ui.value)
    # invariancia: re-expresar el score con otro PDO y usar ancho proporcional da LA MISMA partición
    _c_ref, _ = cortes_pdo(s_dev, K_SEL, ancho_ui.value * 20.0 / pdo_ui.value, 20.0)
    invariancia_pdo = bool(np.array_equal(asignar(s_dev, cortes_pdo_sel), asignar(s_dev, _c_ref)))
    _dp_c, _ = dp_escala(s_dev, y_dev, K_SEL, alfa=alfa_ui.value)
    dp_factible = _dp_c is not None
    disenos = {
        "PDO": cortes_pdo_sel,
        "PD objetivo": cortes_pd_geometrica(K_SEL)[0],
        "Cuantiles": cortes_cuantil(s_dev, K_SEL),
        "Óptimo (DP)": _dp_c if dp_factible else cortes_cuantil(s_dev, K_SEL),
    }
    comparacion_disenos = pd.DataFrame(
        {k: metricas_escala(s_dev, y_dev, p_dev, c, alfa=alfa_ui.value) for k, c in disenos.items()}).T
    comparacion_disenos
    return (
        K_SEL,
        comparacion_disenos,
        cortes_pdo_escala,
        disenos,
        dp_factible,
        invariancia_pdo,
        p_dev,
        s_dev,
        y_dev,
    )


@app.cell
def _(K_SEL, asignar, disenos, np, plt, tabla_escala, y_dev, s_dev, p_dev):
    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.8))
    _x = np.arange(K_SEL)
    for _i, (_nom, _c) in enumerate(disenos.items()):
        _t = tabla_escala(s_dev, y_dev, p_dev, _c).iloc[::-1]       # mejor → peor
        _axs[0].plot(_x, 100 * _t["tasa_obs"], marker="o", label=_nom)
        _axs[1].plot(_x, 100 * _t["pct_pob"], marker="s", label=_nom)
    _axs[0].set_yscale("log")
    _axs[0].set_title("Tasa de malos observada por banda (DEV)")
    _axs[0].set_xlabel("banda (0 = mejor)")
    _axs[0].set_ylabel("tasa de malos (%, escala log)")
    _axs[1].axhline(100 / K_SEL, color="grey", ls=":", lw=1, label="1/K")
    _axs[1].set_title("Ocupación por banda (DEV)")
    _axs[1].set_xlabel("banda (0 = mejor)")
    _axs[1].set_ylabel("% de la población")
    _axs[1].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(K_SEL, ancho_ui, coma, comparacion_disenos, cortes_pdo_escala, disenos, dp_factible, gini_continuo,
      invariancia_pdo, mo, pdo_ui, s_dev, y_dev):
    _c = comparacion_disenos
    _dup = ancho_ui.value / pdo_ui.value
    mo.md(f"""
    **Lectura (K = {K_SEL}).** Gini del score continuo en DEV = {coma(gini_continuo(y_dev, s_dev), 4)}; bandear
    cuesta Gini (empates): PDO {coma(_c.loc['PDO', 'gini_bandas'], 4)}, PD objetivo {coma(_c.loc['PD objetivo', 'gini_bandas'], 4)},
    cuantiles {coma(_c.loc['Cuantiles', 'gini_bandas'], 4)}, óptimo {coma(_c.loc['Óptimo (DP)', 'gini_bandas'], 4)}.

    - **PDO**: ancho = {ancho_ui.value} pts en una escala de PDO {pdo_ui.value} = {coma(_dup, 2)} duplicaciones de odds por banda.
      Cortes en la escala re-expresada: {', '.join(f'{x:.0f}' for x in cortes_pdo_escala)}; en la escala oficial:
      {', '.join(coma(x, 1) for x in disenos['PDO'])}. Invariancia (misma partición con ancho·20/PDO en la escala
      original): **{'sí' if invariancia_pdo else 'NO'}**. El PDO del scaling es cosmético; el ancho en log-odds no.
      Bandas vacías: {int(_c.loc['PDO', 'bandas_vacias'])}. El rango del score limita cuántas bandas de ancho fijo caben.
    - **PD objetivo** (1,5% a 40%): razón constante entre límites de PD, no en odds. En log-odds el ancho es
      $\\ln r + \\ln\\frac{{1-p_j/r}}{{1-p_j}} > \\ln r$: las bandas de PD alta son **más anchas** en score (cortes
      {', '.join(coma(x, 1) for x in disenos['PD objetivo'])}). Bandas vacías: {int(_c.loc['PD objetivo', 'bandas_vacias'])}.
    - **Cuantiles**: HHI ≈ 1/K por construcción (exacto salvo empates del score, que es discreto), pero pares
      adyacentes no separables: {int(_c.loc['Cuantiles', 'pares_no_separables'])}. Iguala población, no riesgo.
    - **Óptimo (DP)**: {'factible' if dp_factible else '**NO factible** con estas restricciones (se muestra cuantiles como reemplazo)'};
      por construcción todos los pares son separables al α elegido y cada banda tiene ≥ 3% y ≥ 10 malos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. El diseño óptimo: programación dinámica, verificada por fuerza bruta

    Se pre-binea el score en $M$ bloques finos (cuantiles) y se busca la partición contigua en K bandas que
    **maximiza la log-verosimilitud binomial** con PD constante por banda (equivalente a minimizar la
    divergencia KL entre la escala fina y la agrupada, §3.4 del documento). Restricciones: ≥ 3% de la
    población, ≥ 10 malos y z unilateral significativo contra la banda adyacente. La restricción de
    separabilidad depende solo de las dos últimas bandas, por eso cabe en una DP con estado (k, inicio de la
    última banda, fin): costo $O(K M^3)$.

    Implementación 2 (la «librería» de este problema): enumeración exhaustiva de todas las particiones con
    `itertools.combinations` en una grilla chica. Deben coincidir el valor y los cortes.
    """)
    return


@app.cell
def _(combinations, dp_escala, fuerza_bruta_escala, np, pd, s_dev, y_dev):
    _filas = []
    for _K in [3, 4, 5]:
        for _restr in [True, False]:
            _c_dp, _v_dp = dp_escala(s_dev, y_dev, _K, m_fino=12, restringida=_restr)
            _c_bf, _v_bf = fuerza_bruta_escala(s_dev, y_dev, _K, m_fino=12, restringida=_restr,
                                               combinaciones=combinations)
            _filas.append({"K": _K, "restringida": _restr, "loglik_DP": _v_dp, "loglik_fuerza_bruta": _v_bf,
                           "mismos_cortes": bool(np.allclose(_c_dp, _c_bf))})
    verificacion_dp = pd.DataFrame(_filas)

    # ¿cuántas bandas separables soporta DEV?
    _fact = []
    for _K in range(4, 16):
        _c, _v = dp_escala(s_dev, y_dev, _K, alfa=0.05)
        _fact.append({"K": _K, "factible": _c is not None, "loglik": _v})
    factibilidad_dp = pd.DataFrame(_fact)
    K_MAX_SEPARABLE = int(factibilidad_dp.loc[factibilidad_dp["factible"], "K"].max())
    verificacion_dp
    return K_MAX_SEPARABLE, factibilidad_dp, verificacion_dp


@app.cell
def _(K_MAX_SEPARABLE, coma, factibilidad_dp, mo, verificacion_dp):
    _f = factibilidad_dp.set_index("K")
    _g = _f.loc[_f["factible"], "loglik"]
    _k_mejor = int(_g.idxmax())
    mo.md(f"""
    **Lectura.** DP y fuerza bruta coinciden en valor y cortes en los {len(verificacion_dp)} casos
    ({int(verificacion_dp['mismos_cortes'].sum())} de {len(verificacion_dp)} con los mismos cortes).
    Con n = 10.065 en DEV, ≥ 3% y ≥ 10 malos por banda y separabilidad al 5%, **el máximo de bandas factibles
    es K = {K_MAX_SEPARABLE}**: la granularidad la dicta la información (malos), no el gusto del comité. Más
    fino aún: la log-verosimilitud restringida sube de {coma(_g.loc[4], 1)} (K = 4) a {coma(_g.loc[_k_mejor], 1)}
    (K = {_k_mejor}) y después **baja** ({coma(_g.loc[K_MAX_SEPARABLE], 1)} con K = {K_MAX_SEPARABLE}): con la
    restricción de separabilidad, el conjunto factible de K + 1 bandas no contiene al de K, y forzar bandas
    extra obliga a cortes peores. Entre K = 7 y K = {_k_mejor} la ganancia es de
    {coma(_g.loc[_k_mejor] - _g.loc[7], 1)} unidades de log-verosimilitud: rendimientos decrecientes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Qué PD se le asigna a la banda

    Cuatro reglas sobre la escala PDO de 8 bandas (cortes 480…600, ancho 20 = una duplicación):

    1. **Media aritmética** de las PD calibradas de la banda (lo que hizo el curso): preserva la suma de
       malos esperados de la muestra de diseño.
    2. **Punto medio geométrico**: la PD en el score medio, $\sigma$ del logit medio. Es la media geométrica
       de las odds de los límites.
    3. **Media exacta bajo logit uniforme** dentro de la banda:
       $\bar p = \dfrac{\ln(1+e^{\eta_{hi}})-\ln(1+e^{\eta_{lo}})}{\eta_{hi}-\eta_{lo}}$ (numpy) vs
       `scipy.integrate.quad`.
    4. **Tasa observada suavizada**: logística ponderada de la tasa de cada banda sobre el logit de su PD
       media (2 parámetros, monótona por construcción); IRLS en numpy vs `statsmodels.GLM`.

    Las bandas abiertas (la peor y la mejor) no tienen punto medio: se les asigna la media (convención
    declarada). Luego se mide la **pérdida esperada** en TTD (LGD 45%, EAD 1) contra la verdad del generador.
    """)
    return


@app.cell
def _(FACTOR, OFFSET, asignar, expit, logit, muestras, np, p_dev, pd, quad, s_dev, sm, tabla_escala, y_dev):
    CORTES_REF = np.arange(480.0, 601.0, 20.0)                  # 8 bandas PDO para esta sección
    _t = tabla_escala(s_dev, y_dev, p_dev, CORTES_REF)
    _eta = lambda s: (OFFSET - s) / FACTOR                        # logit de la PD en el score s
    _lo, _hi = _t["score_desde"].to_numpy(), _t["score_hasta"].to_numpy()
    _cerrada = np.isfinite(_lo) & np.isfinite(_hi)
    with np.errstate(invalid="ignore"):
        _e_alto, _e_bajo = _eta(_lo), _eta(_hi)                   # piso de score = logit más alto
        pd_punto_medio = np.where(_cerrada, expit(_eta((_lo + _hi) / 2)), np.nan)
        pd_unif_np = np.where(_cerrada, (np.log1p(np.exp(_e_alto)) - np.log1p(np.exp(_e_bajo))) / (_e_alto - _e_bajo),
                              np.nan)
    pd_unif_quad = np.array([quad(lambda e: expit(e), _e_bajo[k], _e_alto[k])[0] / (_e_alto[k] - _e_bajo[k])
                             if _cerrada[k] else np.nan for k in range(len(_t))])

    # tasa suavizada: logística ponderada tasa_b ~ a + b·logit(pd_media_b). IRLS numpy vs statsmodels
    _X = np.column_stack([np.ones(len(_t)), logit(_t["pd_cal_media"].to_numpy())])
    _yb, _w = _t["tasa_obs"].to_numpy(), _t["n"].to_numpy()
    _beta = np.zeros(2)
    for _ in range(50):
        _p = expit(_X @ _beta)
        _W = _w * _p * (1 - _p)
        _paso = np.linalg.solve(_X.T @ (_X * _W[:, None]), _X.T @ (_w * (_yb - _p)))
        _beta = _beta + _paso
        if np.max(np.abs(_paso)) < 1e-12:
            break
    beta_suav_np = _beta
    beta_suav_sm = sm.GLM(_yb, _X, family=sm.families.Binomial(), var_weights=_w).fit().params
    pd_suav = expit(_X @ beta_suav_np)

    asignacion = _t[["banda", "score_desde", "score_hasta", "n", "tasa_obs", "pd_cal_media"]].copy()
    asignacion["pd_punto_medio"] = pd_punto_medio
    asignacion["pd_logit_uniforme"] = pd_unif_np
    asignacion["pd_suavizada"] = pd_suav
    asignacion["media/punto_medio"] = asignacion["pd_cal_media"] / asignacion["pd_punto_medio"]

    # pérdida esperada (LGD 45%, EAD 1) en TTD, por regla, contra la verdad
    _reglas = {
        "individual (PD de cada cliente)": None,
        "media aritmética": asignacion["pd_cal_media"].to_numpy(),
        "punto medio geométrico": np.where(_cerrada, pd_punto_medio, asignacion["pd_cal_media"]),
        "logit uniforme": np.where(_cerrada, pd_unif_np, asignacion["pd_cal_media"]),
        "tasa suavizada": pd_suav,
    }
    _filas = []
    for _m in ["HO", "OOT", "TTD"]:
        _d = muestras[_m]
        _b = asignar(_d["score"], CORTES_REF)
        for _nom, _v in _reglas.items():
            _pd = _d["pd_cal"].to_numpy() if _v is None else _v[_b]
            _filas.append({"muestra": _m, "regla": _nom, "EL_%_saldo": 45 * _pd.mean(),
                           "EL_verdad_%": 45 * _d["pd_real"].mean()})
    el_reglas = pd.DataFrame(_filas).pivot(index="regla", columns="muestra", values="EL_%_saldo")
    el_verdad = {m: 45 * muestras[m]["pd_real"].mean() for m in ["HO", "OOT", "TTD"]}
    asignacion.round(4)
    return (
        CORTES_REF,
        asignacion,
        beta_suav_np,
        beta_suav_sm,
        el_reglas,
        el_verdad,
        pd_unif_np,
        pd_unif_quad,
    )


@app.cell
def _(el_reglas):
    el_reglas.round(3)
    return


@app.cell
def _(asignacion, coma, el_reglas, el_verdad, mo, np):
    _a = asignacion.set_index("banda")
    _cer = _a.iloc[1:-1]
    _r = el_reglas
    mo.md(f"""
    **Lectura.**

    - En bandas cerradas de una duplicación, media aritmética / punto medio va de
      {coma(_cer['media/punto_medio'].min(), 3)} a {coma(_cer['media/punto_medio'].max(), 3)}. Con densidad
      uniforme en logit la teoría da $\\sinh(h/2)/(h/2)$ = 1,020 para $h=\\ln 2$ (§3.5): la media aritmética
      queda **sobre** el punto medio por Jensen (σ convexa bajo PD 50%), pero la **forma de la densidad dentro
      de la banda** puede empujar hacia cualquier lado (en la banda de 480–500, con PD ≈ 47%, σ ya es casi
      lineal).
    - Las bandas abiertas son el problema real: el «punto medio» no existe. La banda peor (< 480) tiene PD media
      {coma(100 * _a.iloc[0]['pd_cal_media'], 1)}% y su único límite finito corresponde a PD
      {coma(100 / (1 + np.exp((480 - 487.1229) / 28.8539)), 1)}%; la mejor (≥ 600) tiene media
      {coma(100 * _a.iloc[-1]['pd_cal_media'], 2)}% contra 1,96% en su piso. Cualquier «PD de la banda» abierta es
      una convención que hay que declarar (aquí: la media).
    - **Pérdida esperada TTD (% del saldo, LGD 45%)**: individual {coma(_r.loc['individual (PD de cada cliente)', 'TTD'], 3)} ·
      media {coma(_r.loc['media aritmética', 'TTD'], 3)} · punto medio {coma(_r.loc['punto medio geométrico', 'TTD'], 3)} ·
      suavizada {coma(_r.loc['tasa suavizada', 'TTD'], 3)} · **verdad {coma(el_verdad['TTD'], 3)}**. La regla de
      asignación mueve la EL en décimas; la calibración (el deterioro de 2025 no está en la TC) la mueve en
      puntos. Orden de magnitud de las prioridades: primero el nivel, después la regla por banda.
    - HO (mismo periodo que el diseño): media {coma(_r.loc['media aritmética', 'HO'], 3)} vs verdad {coma(el_verdad['HO'], 3)}.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Potencia del backtest por banda: cuántos malos hacen falta

    Bajo $H_0$ la banda tiene PD $p_0$; la alternativa relevante es $p_1 = m\,p_0$ (la banda subestima por un
    factor $m$). Con aproximación normal a Poisson (§3.7 del documento) el número de malos **esperados**
    que se necesita es

    $$\lambda_0 = n\,p_0 \approx \left(\frac{z_{1-\alpha}+z_{1-\beta}\sqrt{m}}{m-1}\right)^2 ,$$

    casi independiente de $p_0$: la potencia la compran **malos**, no créditos. Se compara con la potencia
    **exacta** del binomial (cola de la binomial en numpy con `lgamma` vs `scipy.stats.binom`).
    """)
    return


@app.cell
def _(mo):
    mult_ui = mo.ui.slider(1.2, 3.0, step=0.1, value=2.0, label="Factor m de subestimación a detectar")
    pd0_ui = mo.ui.dropdown({"0,11% (A1)": 0.0011, "1,42% (B2)": 0.0142, "5,44% (C2)": 0.0544,
                             "26,91% (E)": 0.2691}, value="1,42% (B2)", label="PD prometida p₀")
    mo.hstack([mult_ui, pd0_ui])
    return mult_ui, pd0_ui


@app.cell
def _(norm, np):
    from math import lgamma

    def cola_binomial_np(k, n, p):
        """P(D ≥ k) para D ~ Bin(n, p), en numpy puro (log-pmf con lgamma y suma estable)."""
        if k <= 0:
            return 1.0
        j = np.arange(k, n + 1)
        lg = np.array([lgamma(n + 1) - lgamma(t + 1) - lgamma(n - t + 1) for t in j])
        logpmf = lg + j * np.log(p) + (n - j) * np.log1p(-p)
        mx = logpmf.max()
        return float(np.exp(mx) * np.exp(logpmf - mx).sum())

    def lambda_requerido(mult, alfa=0.05, potencia=0.80):
        return ((norm.isf(alfa) + norm.isf(1 - potencia) * np.sqrt(mult)) / (mult - 1)) ** 2

    def potencia_normal(lam0, mult, alfa=0.05):
        """Aproximación normal a Poisson: P(D > λ0 + z_{1−α}√λ0 | media m·λ0)."""
        return norm.sf((lam0 + norm.isf(alfa) * np.sqrt(lam0) - mult * lam0) / np.sqrt(mult * lam0))

    return cola_binomial_np, lambda_requerido, potencia_normal


@app.cell
def _(binom, cola_binomial_np, lambda_requerido, mult_ui, np, pd, pd0_ui, plt, potencia_binomial,
      potencia_normal):
    _p0, _m = pd0_ui.value, mult_ui.value
    _lams = np.array([0.5, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    _filas = []
    for _l in _lams:
        _n = int(round(_l / _p0))
        _pot, _kc = potencia_binomial(_n, _p0, _m)
        _filas.append({"malos_esperados": _l, "n": _n, "k_critico": _kc,
                       "tamaño_real": float(binom.sf(_kc - 1, _n, _p0)),
                       "tamaño_numpy": cola_binomial_np(_kc, _n, _p0),
                       "potencia_exacta": _pot, "potencia_numpy": cola_binomial_np(_kc, _n, min(_m * _p0, 1)),
                       "potencia_normal": float(potencia_normal(_l, _m))})
    tabla_potencia = pd.DataFrame(_filas)
    LAMBDA_REQ = float(lambda_requerido(_m))
    _fig, _ax = plt.subplots(figsize=(7.5, 3.6))
    _ax.plot(tabla_potencia["malos_esperados"], tabla_potencia["potencia_exacta"], marker="o",
             label="binomial exacta (dientes: k entero)")
    _ax.plot(tabla_potencia["malos_esperados"], tabla_potencia["potencia_normal"], ls="--",
             label="aproximación normal")
    _ax.axhline(0.8, color="grey", ls=":", lw=1)
    _ax.axvline(LAMBDA_REQ, color="grey", ls=":", lw=1)
    _ax.set_xscale("log")
    _ax.set_xlabel("malos esperados bajo H0, n·p0 (escala log)")
    _ax.set_ylabel("potencia")
    _ax.set_title(f"Potencia del binomial unilateral (α = 5%) para detectar PD real = {_m:.1f}·p0")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return LAMBDA_REQ, tabla_potencia


@app.cell
def _(LAMBDA_REQ, coma, mo, mult_ui, pd0_ui, tabla_potencia):
    _t = tabla_potencia
    mo.md(f"""
    **Lectura.** Para detectar m = {coma(mult_ui.value, 1)} con potencia 80% y α = 5% hacen falta
    ≈ **{coma(LAMBDA_REQ, 1)} malos esperados** (n ≈ {coma(LAMBDA_REQ / pd0_ui.value, 0)} con p₀ = {coma(100 * pd0_ui.value, 2)}%).
    Con m = 2 son ≈ 8; con m = 1,5, ≈ 29; con m = 1,25, ≈ 107. La tabla muestra además que el test exacto es
    **conservador**: su tamaño real (columna `tamaño_real`) queda bajo 5% por la discreción, y con pocos malos
    esperados puede ser muy inferior. numpy (`lgamma`) y `scipy.stats.binom` coinciden (check).
    Regla de diseño que se desprende: **mínimo de malos esperados por banda en la ventana de backtesting**,
    no mínimo de créditos. Si una banda buena no puede juntarlos, se agrega (A1+A2) para el backtest o se
    acumulan cosechas (M19).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Granularidad vs estabilidad

    Más bandas = más resolución (Gini de la escala, menos pérdida de información) y, a la vez, menos malos por
    banda (menos potencia) y más **migración** entre bandas por ruido. La migración se simula como un score
    conductual a 6 meses: $s' = \mu + \rho(s-\mu) + \sqrt{1-\rho^2}\,\sigma_s\,\varepsilon$ (misma distribución
    marginal, correlación ρ). Diseño de cuantiles para aislar el efecto de K.
    """)
    return


@app.cell
def _(mo):
    rho_ui = mo.ui.slider(0.60, 0.98, step=0.02, value=0.85, label="Correlación ρ del score a 6 meses")
    rho_ui
    return (rho_ui,)


@app.cell
def _(
    asignar,
    cortes_cuantil,
    gini_bandas,
    lambda_requerido,
    muestras,
    np,
    pd,
    rho_ui,
    s_dev,
    y_dev,
):
    _rng = np.random.default_rng(16)
    _mu, _sd = s_dev.mean(), s_dev.std()
    s_dev_6m = _mu + rho_ui.value * (s_dev - _mu) + np.sqrt(1 - rho_ui.value ** 2) * _sd * _rng.standard_normal(
        len(s_dev))
    _oot = muestras["OOT"]
    _filas = []
    for _K in range(3, 21):
        _c = cortes_cuantil(s_dev, _K)
        _b0, _b1 = asignar(s_dev, _c), asignar(s_dev_6m, _c)
        _bo = asignar(_oot["score"], _c)
        _esp_oot = np.bincount(_bo, weights=_oot["pd_cal"], minlength=_K)
        _filas.append({"K": _K, "gini_bandas_DEV": gini_bandas(_b0, y_dev),
                       "%_misma_banda_6m": np.mean(_b0 == _b1),
                       "salto_medio_si_migra": np.abs(_b0 - _b1)[_b0 != _b1].mean(),
                       "min_malos_esperados_OOT": _esp_oot.min(),
                       "bandas_OOT_con_potencia(m=1,5)": int(np.sum(_esp_oot >= lambda_requerido(1.5)))})
    granularidad = pd.DataFrame(_filas)
    granularidad.round(4)
    return granularidad, s_dev_6m


@app.cell
def _(granularidad, plt, rho_ui):
    _fig, _axs = plt.subplots(1, 3, figsize=(12, 3.4))
    _g = granularidad
    _axs[0].plot(_g["K"], _g["gini_bandas_DEV"], marker="o")
    _axs[0].set_title("Resolución: Gini de la escala (DEV)")
    _axs[0].set_xlabel("K bandas")
    _axs[0].set_ylabel("Gini")
    _axs[1].plot(_g["K"], 100 * _g["%_misma_banda_6m"], marker="o", color="tab:red")
    _axs[1].set_title(f"Estabilidad: % en la misma banda (ρ = {rho_ui.value:.2f})")
    _axs[1].set_xlabel("K bandas")
    _axs[1].set_ylabel("% de clientes")
    _axs[2].plot(_g["K"], _g["min_malos_esperados_OOT"], marker="o", color="tab:green")
    _axs[2].axhline(28.6, color="grey", ls=":", lw=1, label="λ requerido m = 1,5")
    _axs[2].axhline(8.0, color="grey", ls="--", lw=1, label="λ requerido m = 2")
    _axs[2].set_title("Potencia: mín. malos esperados por banda (OOT)")
    _axs[2].set_xlabel("K bandas")
    _axs[2].set_ylabel("malos esperados")
    _axs[2].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(coma, granularidad, mo, rho_ui):
    _g = granularidad.set_index("K")
    mo.md(f"""
    **Lectura (ρ = {coma(rho_ui.value, 2)}).** De K = 4 a K = 8 el Gini de la escala sube de
    {coma(_g.loc[4, 'gini_bandas_DEV'], 4)} a {coma(_g.loc[8, 'gini_bandas_DEV'], 4)}; de 8 a 16 solo a
    {coma(_g.loc[16, 'gini_bandas_DEV'], 4)}. La fracción que se queda en su banda cae de
    {coma(100 * _g.loc[4, '%_misma_banda_6m'], 1)}% a {coma(100 * _g.loc[8, '%_misma_banda_6m'], 1)}% y
    {coma(100 * _g.loc[16, '%_misma_banda_6m'], 1)}%. Con cuantiles, el mínimo de malos esperados en OOT es
    {coma(_g.loc[8, 'min_malos_esperados_OOT'], 1)} con K = 8 y {coma(_g.loc[16, 'min_malos_esperados_OOT'], 1)} con K = 16.
    El codo típico está entre 7 y 10 bandas: por eso el rango corporativo del curso (7–10) es razonable
    **como convención**, no como teorema. Para una cartera más chica o de menor PD, el codo se corre a la izquierda.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Matriz de migración entre bandas

    La matriz $P_{ij} = N_{ij}/N_i$ (de la banda $i$ hoy a la $j$ en 6 meses) resume la estabilidad del rating
    conductual. Implementación numpy (`np.add.at`) vs `pd.crosstab(normalize="index")`. Métricas: diagonal,
    % que sube / baja, y salto medio. Escala de 8 bandas PDO (§5).
    """)
    return


@app.cell
def _(CORTES_REF, asignar, nombres_bandas, np, pd, s_dev, s_dev_6m):
    _K = len(CORTES_REF) + 1
    _b0, _b1 = asignar(s_dev, CORTES_REF), asignar(s_dev_6m, CORTES_REF)
    _N = np.zeros((_K, _K))
    np.add.at(_N, (_b0, _b1), 1)
    migracion_np = _N / _N.sum(axis=1, keepdims=True)
    migracion_pd = pd.crosstab(_b0, _b1, normalize="index").reindex(index=range(_K), columns=range(_K),
                                                                     fill_value=0.0)
    _nom = nombres_bandas(_K)
    migracion = pd.DataFrame(migracion_np[::-1, ::-1], index=[f"hoy {x}" for x in _nom[::-1]],
                             columns=[f"6m {x}" for x in _nom[::-1]])
    _sube = np.sum(_b1 > _b0) / len(_b0)
    _baja = np.sum(_b1 < _b0) / len(_b0)
    resumen_migracion = {"diagonal_%": float(np.mean(_b0 == _b1)), "mejora_%": float(_sube),
                         "empeora_%": float(_baja), "salto_medio": float(np.abs(_b0 - _b1)[_b0 != _b1].mean())}
    migracion.round(3)
    return migracion, migracion_np, migracion_pd, resumen_migracion


@app.cell
def _(coma, mo, resumen_migracion):
    _r = resumen_migracion
    mo.md(f"""
    **Lectura.** {coma(100 * _r['diagonal_%'], 1)}% se queda en su banda, {coma(100 * _r['mejora_%'], 1)}% mejora y
    {coma(100 * _r['empeora_%'], 1)}% empeora; quien migra salta en promedio {coma(_r['salto_medio'], 2)} bandas.
    Como el score a 6 meses se simuló con la **misma marginal**, la matriz es aproximadamente simétrica en flujos:
    una asimetría persistente en datos reales es señal de deriva (o de un ciclo, si la escala es PIT). En IRB,
    las instrucciones de reporte del BCE incluyen tests sobre esta matriz (monotonía de las filas fuera de la
    diagonal, *matrix weighted bandwidth*); M20 los retoma para monitoreo.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Recalibrar mueve las bandas δ·factor

    Si las bandas se definen en **PD** (o, equivalente, en el score *calibrado*), una recalibración del
    intercepto con δ desplaza **todos** los scores calibrados en $-\delta\cdot\text{factor}$; visto desde el score
    sin recalibrar, los cortes suben $\delta\cdot\text{factor}$ puntos. Se recalibra PIT a OOT (la tasa con
    deterioro) y se mide cuántos clientes de TTD cambian de banda. La alternativa (bandas fijas en puntos) deja
    la ocupación intacta y cambia la PD que cada banda promete.
    """)
    return


@app.cell
def _(CORTES_REF, DELTA_TC, FACTOR, asignar, asignacion, brentq, expit, muestras, nombres_bandas, np, pd):
    _oot = muestras["OOT"]
    # δ PIT absoluto (sobre el logit crudo del modelo) para que la PD media de OOT iguale su tasa
    DELTA_PIT = brentq(lambda d: expit(_oot["lp"].values + d).mean() - _oot["malo"].mean(), -5, 5)
    _ttd = muestras["TTD"]
    _delta_extra = DELTA_PIT - DELTA_TC                   # lo que se agrega sobre la calibración vigente
    DESPLAZAMIENTO_PTS = _delta_extra * FACTOR
    _s = _ttd["score"].to_numpy()
    _s_pit = _s - DESPLAZAMIENTO_PTS                      # score recalibrado: todos bajan δ·factor
    _b0, _b1 = asignar(_s, CORTES_REF), asignar(_s_pit, CORTES_REF)
    _K = len(CORTES_REF) + 1
    _pdm = asignacion["pd_cal_media"].to_numpy()
    recal_bandas = pd.DataFrame({
        "banda": nombres_bandas(_K),
        "%_TTD_antes": np.bincount(_b0, minlength=_K) / len(_b0),
        "%_TTD_despues (bandas en PD)": np.bincount(_b1, minlength=_K) / len(_b1),
        "pd_prometida (media DEV)": _pdm,
        "pd_si_bandas_fijas_en_puntos": expit(np.log(_pdm / (1 - _pdm)) + _delta_extra),
    }).iloc[::-1].reset_index(drop=True)
    FRAC_CAMBIA_BANDA = float(np.mean(_b0 != _b1))
    # verificación: cambia de banda quien está en [corte, corte + δ·factor) de algún corte
    FRAC_FRANJA = float(np.mean(np.any((_s[:, None] >= CORTES_REF[None, :])
                                       & (_s[:, None] - DESPLAZAMIENTO_PTS < CORTES_REF[None, :]), axis=1)))
    recal_bandas.round(4)
    return DELTA_PIT, DESPLAZAMIENTO_PTS, FRAC_CAMBIA_BANDA, FRAC_FRANJA, recal_bandas


@app.cell
def _(DELTA_PIT, DELTA_TC, DESPLAZAMIENTO_PTS, FRAC_CAMBIA_BANDA, coma, mo):
    mo.md(f"""
    **Lectura.** δ PIT (a OOT) = {coma(DELTA_PIT, 4)} vs δ TC = {coma(DELTA_TC, 4)}: la recalibración agrega
    {coma(DELTA_PIT - DELTA_TC, 4)} en log-odds = **{coma(DESPLAZAMIENTO_PTS, 2)} puntos** para todos. Con bandas
    definidas en PD, {coma(100 * FRAC_CAMBIA_BANDA, 1)}% de la bandeja TTD baja una banda: exactamente quienes
    estaban a menos de {coma(DESPLAZAMIENTO_PTS, 1)} puntos sobre un corte. En Austral, TC → PIT eran 1,89 puntos
    (δ 0,1115 → 0,177) y con bandas de 20 puntos moverían ≈ 1,89/20 ≈ 9% de cada banda interior si la densidad
    fuera plana. Alternativa: bandas fijas en puntos; nadie migra, pero la PD de cada banda sube (última columna)
    y todo lo que se fijó en PD (apetito, pricing, provisiones) cambia. Cuál de las dos se usa es una **decisión
    de gobierno que se escribe**, no un detalle de implementación (M13 §3.7, M15 §3.9).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Mapear un segundo modelo a la escala corporativa

    «Modelo motos» heredado: 3 variables, escala propia **PDO 40, 500 ⇔ 20:1**, calibrado a la misma TC. Se
    mapea a la escala corporativa (8 bandas PDO, §5) de dos formas: (a) **por PD** (score propio → log-odds →
    score corporativo → banda) y (b) el error típico: copiar los cortes numéricos 480…600 sobre el score propio.
    """)
    return


@app.cell
def _(CORTES_REF, FACTOR, OFFSET, TC_SINT, a_woe, asignar, brentq, dev, expit, mapas_woe, muestras,
      nombres_bandas, np, pd, sm):
    _v2 = ["uso_linea_prom_12m", "meses_desde_mora_12m", "antiguedad_meses"]
    _m2 = sm.Logit(dev["malo"].values, sm.add_constant(a_woe(dev, _v2, dev, mapas_woe))).fit(disp=0)
    _lp2 = {m: np.asarray(sm.add_constant(a_woe(d, _v2, dev, mapas_woe), has_constant="add") @ _m2.params)
            for m, d in muestras.items()}
    _lcal = np.concatenate([_lp2["DEV"], _lp2["HO"]])
    _d2 = brentq(lambda d: expit(_lcal + d).mean() - TC_SINT, -5, 5)
    FACTOR_2 = 40 / np.log(2)
    OFFSET_2 = 500 - FACTOR_2 * np.log(20)
    _K = len(CORTES_REF) + 1
    _filas = []
    for _m in ["DEV", "OOT"]:
        _d = muestras[_m]
        _s2 = OFFSET_2 - FACTOR_2 * (_lp2[_m] + _d2)                      # score propio del modelo motos
        _s_corp = OFFSET + FACTOR * (_s2 - OFFSET_2) / FACTOR_2            # mismo log-odds, escala corporativa
        for _nom_map, _b in [("modelo 1 (corporativo)", asignar(_d["score"], CORTES_REF)),
                             ("modelo 2 mapeado por PD", asignar(_s_corp, CORTES_REF)),
                             ("modelo 2 con cortes copiados", asignar(_s2, CORTES_REF))]:
            _n = np.bincount(_b, minlength=_K)
            _mal = np.bincount(_b, weights=_d["malo"], minlength=_K)
            for _k in range(_K):
                _filas.append({"muestra": _m, "mapeo": _nom_map, "banda": nombres_bandas(_K)[_k],
                               "orden": _k, "pct": _n[_k] / _n.sum(),
                               "tasa": _mal[_k] / _n[_k] if _n[_k] > 0 else np.nan})
    mapeo_modelos = pd.DataFrame(_filas)
    _dev_map = mapeo_modelos[mapeo_modelos["muestra"] == "DEV"]
    mapeo_pct = _dev_map.pivot(index="banda", columns="mapeo", values="pct").reindex(nombres_bandas(_K)[::-1])
    mapeo_tasa = _dev_map.pivot(index="banda", columns="mapeo", values="tasa").reindex(nombres_bandas(_K)[::-1])
    HHI_MAPEO = {k: float(np.sum(mapeo_pct[k].to_numpy() ** 2)) for k in mapeo_pct.columns}
    mapeo_tasa.round(4)
    return HHI_MAPEO, mapeo_pct, mapeo_tasa


@app.cell
def _(mapeo_pct):
    mapeo_pct.round(4)
    return


@app.cell
def _(HHI_MAPEO, coma, mapeo_pct, mo):
    _c = mapeo_pct
    mo.md(f"""
    **Lectura.** Mapeado por PD, el modelo motos usa la escala corporativa con el **mismo significado** de banda
    (tasas comparables banda a banda con el modelo 1, tabla de tasas), pero **concentra más** población en las
    bandas centrales porque discrimina menos: HHI {coma(HHI_MAPEO['modelo 2 mapeado por PD'], 4)} vs
    {coma(HHI_MAPEO['modelo 1 (corporativo)'], 4)} del modelo 1. Un modelo débil no «llena» los extremos: eso es
    información, no un defecto de la escala. Con los cortes copiados sobre su score propio (500 ⇔ 20:1, no 1,5:1)
    el resultado es absurdo: {coma(100 * _c.loc['E', 'modelo 2 con cortes copiados'], 1)}% cae en E y
    {coma(100 * _c.loc['A1', 'modelo 2 con cortes copiados'], 1)}% en A1: la banda ya no significa una PD.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Experimentos de falla con verdad conocida

    **F1 · Inversión espuria.** Con las PD *verdaderas* de Austral (estrictamente monótonas) y el n de OOT de la
    clase 5, ¿qué probabilidad hay de observar al menos una inversión entre bandas adyacentes? Exacta (doble
    suma de binomiales, `scipy`) vs Monte Carlo (numpy).

    **F2 · Sobreajuste del óptimo sin restricciones.** DP sin la restricción de separabilidad y con mínimos
    laxos (1% de población, 3 malos, 14 bandas) diseñada en DEV: inversiones en DEV vs en HO (mismo periodo,
    otra muestra).
    """)
    return


@app.cell
def _(binom, np, oot_austral, pd):
    def prob_inversion_exacta(n1, p1, n2, p2):
        """P(tasa_obs banda mejor ≥ tasa_obs banda peor), p1 < p2 (1 = mejor)."""
        k1 = np.arange(n1 + 1)
        pmf1 = binom.pmf(k1, n1, p1)
        # para cada k1, P(X2/n2 ≤ k1/n1) = P(X2 ≤ floor(k1·n2/n1))
        lim = np.floor(k1 * n2 / n1 + 1e-12)
        return float(np.sum(pmf1 * binom.cdf(lim, n2, p2)))

    _rng = np.random.default_rng(2016)
    _o = oot_austral
    _filas = []
    _B = 20_000
    _inv_any = np.zeros(_B, dtype=bool)
    for _k in range(7):                                   # par (mejor k, peor k+1)
        _n1, _p1 = int(_o.loc[_k, "n"]), float(_o.loc[_k, "pd_cal"])
        _n2, _p2 = int(_o.loc[_k + 1, "n"]), float(_o.loc[_k + 1, "pd_cal"])
        _x1 = _rng.binomial(_n1, _p1, _B) / _n1
        _x2 = _rng.binomial(_n2, _p2, _B) / _n2
        _inv = _x1 >= _x2
        _inv_any |= _inv
        _filas.append({"par": f"{_o.loc[_k, 'banda']}–{_o.loc[_k + 1, 'banda']}",
                       "p_inversion_exacta": prob_inversion_exacta(_n1, _p1, _n2, _p2),
                       "p_inversion_MC": _inv.mean()})
    inversion_espuria = pd.DataFrame(_filas)
    P_ALGUNA_INVERSION = float(_inv_any.mean())
    P_ALGUNA_INVERSION_x10 = None
    _inv10 = np.zeros(_B, dtype=bool)
    for _k in range(7):
        _n1, _p1 = int(_o.loc[_k, "n"]) * 10, float(_o.loc[_k, "pd_cal"])
        _n2, _p2 = int(_o.loc[_k + 1, "n"]) * 10, float(_o.loc[_k + 1, "pd_cal"])
        _inv10 |= (_rng.binomial(_n1, _p1, _B) / _n1) >= (_rng.binomial(_n2, _p2, _B) / _n2)
    P_ALGUNA_INVERSION_x10 = float(_inv10.mean())
    inversion_espuria.round(4)
    return P_ALGUNA_INVERSION, P_ALGUNA_INVERSION_x10, inversion_espuria


@app.cell
def _(asignar, dp_escala, muestras, np, s_dev, y_dev, ztest_adyacentes):
    _c_libre, _ = dp_escala(s_dev, y_dev, 14, m_fino=40, min_pct=0.01, min_malos=3, restringida=False)
    _c_restr, _ = dp_escala(s_dev, y_dev, 9, m_fino=30, min_pct=0.03, min_malos=10, restringida=True)

    def _inversiones(score, y, cortes):
        _b = asignar(score, cortes)
        _K = len(cortes) + 1
        _t = np.bincount(_b, weights=y, minlength=_K) / np.maximum(np.bincount(_b, minlength=_K), 1)
        return int(np.sum(np.diff(_t) >= 0))

    def _no_separables(score, y, cortes):
        _b = asignar(score, cortes)
        _K = len(cortes) + 1
        _, _p = ztest_adyacentes(np.bincount(_b, minlength=_K), np.bincount(_b, weights=y, minlength=_K))
        return int(np.sum(_p > 0.05))

    _ho = muestras["HO"]
    sobreajuste_dp = {
        "libre_K14_inv_DEV": _inversiones(s_dev, y_dev, _c_libre),
        "libre_K14_inv_HO": _inversiones(_ho["score"], _ho["malo"], _c_libre),
        "restringida_K9_inv_DEV": _inversiones(s_dev, y_dev, _c_restr),
        "restringida_K9_inv_HO": _inversiones(_ho["score"], _ho["malo"], _c_restr),
        "libre_K14_no_sep_DEV": _no_separables(s_dev, y_dev, _c_libre),
        "libre_K14_no_sep_HO": _no_separables(_ho["score"], _ho["malo"], _c_libre),
        "restringida_K9_no_sep_DEV": _no_separables(s_dev, y_dev, _c_restr),
        "restringida_K9_no_sep_HO": _no_separables(_ho["score"], _ho["malo"], _c_restr),
    }
    sobreajuste_dp
    return (sobreajuste_dp,)


@app.cell
def _(P_ALGUNA_INVERSION, P_ALGUNA_INVERSION_x10, coma, inversion_espuria, mo, sobreajuste_dp):
    _i = inversion_espuria.set_index("par")
    _s = sobreajuste_dp
    mo.md(f"""
    **Lectura F1.** Aunque la escala verdadera sea perfectamente monótona, con el n de OOT de Austral la
    probabilidad de ver al menos una inversión es **{coma(100 * P_ALGUNA_INVERSION, 1)}%** (A1–A2:
    {coma(100 * _i.loc['A1–A2', 'p_inversion_exacta'], 1)}%, A2–B1: {coma(100 * _i.loc['A2–B1', 'p_inversion_exacta'], 1)}%).
    Con 10 veces más n, {coma(100 * P_ALGUNA_INVERSION_x10, 1)}%. Una inversión en las bandas buenas con n chico
    no prueba que la escala esté rota; la monotonía se evalúa con un test (y, si hace falta, fusionando bandas),
    no con el signo de la diferencia.

    **Lectura F2.** La DP sin restricción de separabilidad y con mínimos laxos (K = 14) ya tiene
    {_s['libre_K14_inv_DEV']} inversiones y {_s['libre_K14_no_sep_DEV']} pares no separables **en DEV** (maximizar
    verosimilitud no pide monotonía: corta donde el ruido la hace subir) y {_s['libre_K14_inv_HO']} inversiones y
    {_s['libre_K14_no_sep_HO']} pares no separables en HO. La restringida (K = 9): {_s['restringida_K9_inv_DEV']}
    inversiones y {_s['restringida_K9_no_sep_DEV']} no separables en DEV; en HO, {_s['restringida_K9_inv_HO']} inversiones
    y **{_s['restringida_K9_no_sep_HO']} pares no separables**: la separabilidad se diseña en DEV y se *valida* fuera,
    con menos n y por tanto menos potencia. La restricción es la regularización, no una garantía.
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
def _(
    FACTOR,
    OFFSET,
    austral,
    austral_jeffreys_np,
    beta_suav_np,
    beta_suav_sm,
    cola_binomial_np,
    comparacion_disenos,
    cv_mod,
    disenos,
    dp_factible,
    gini_bandas,
    hhi_austral_mod,
    hi_mod,
    invariancia_pdo,
    inversion_espuria,
    lambda_requerido,
    migracion_np,
    migracion_pd,
    np,
    pd,
    pd_unif_np,
    pd_unif_quad,
    roc_auc_score,
    s_dev,
    asignar,
    separabilidad_austral,
    tabla_potencia,
    verificacion_dp,
    y_dev,
    FRAC_CAMBIA_BANDA,
    FRAC_FRANJA,
    score_a_pd,
    cortes_cuantil,
):
    # 1. constantes del curso y la escala PDO: 600 ⇔ 50:1, cada 20 pts duplica
    assert np.isclose(FACTOR, 28.8539, atol=1e-4) and np.isclose(OFFSET, 487.1229, atol=1e-4)
    assert np.isclose(score_a_pd(600), 1 / 51) and np.isclose(score_a_pd(660), 1 / 401)
    assert np.isclose((1 - score_a_pd(620)) / score_a_pd(620), 2 * (1 - score_a_pd(600)) / score_a_pd(600))
    # 2. Austral: malos reconstruidos reproducen las tasas publicadas; HHI e índice BCE coherentes
    assert np.allclose(austral["tasa_obs"], austral["tasa_publicada"], atol=6e-5)
    assert int(austral["malos"].sum()) == 362 and int(austral["n"].sum()) == 6723
    assert np.isclose(cv_mod ** 2 + 1, 8 * hhi_austral_mod)                         # identidad CV² + 1 = K·HHI
    assert np.isclose(hi_mod, 1 + np.log(hhi_austral_mod) / np.log(8))
    assert (np.diff(austral["tasa_obs"].to_numpy()) > 0).all()                       # monótona (mejor → peor)
    # 3. z de proporciones: numpy = statsmodels
    assert np.allclose(separabilidad_austral["z_numpy"], separabilidad_austral["z_statsmodels"])
    assert np.allclose(separabilidad_austral["p_numpy"], separabilidad_austral["p_statsmodels"])
    # 4. Jeffreys: scipy.beta = statsmodels
    assert np.allclose(austral_jeffreys_np[0], austral["jeffreys90_lo"]) and np.allclose(
        austral_jeffreys_np[1], austral["jeffreys90_hi"])
    # 5. DP = fuerza bruta
    assert verificacion_dp["mismos_cortes"].all()
    assert np.allclose(verificacion_dp["loglik_DP"], verificacion_dp["loglik_fuerza_bruta"])
    # 6. cuantiles: np.quantile = pandas.quantile (misma interpolación lineal). No se compara con pd.qcut:
    #    el score es DISCRETO (suma de puntos por bin) y qcut usa right=True, así que los empates en un
    #    corte caen del otro lado. Con empates el HHI de «cuantiles» queda ≥ 1/K, no igual.
    _K = 8
    _c = cortes_cuantil(s_dev, _K)
    assert np.allclose(_c, pd.Series(s_dev).quantile(np.arange(1, _K) / _K).to_numpy())
    assert comparacion_disenos.loc["Cuantiles", "HHI"] >= 1 / comparacion_disenos.loc["Cuantiles", "K"] - 1e-12
    # 7. asignación con right=False = pd.cut(right=False) del curso
    _cortes = np.array([540, 560, 580, 600, 620, 640, 660.0])
    _s_test = np.array([539.9, 540.0, 559.99, 600.0, 660.0, 700.0])
    _pdcut = pd.cut(_s_test, np.r_[-np.inf, _cortes, np.inf], right=False, labels=False)
    assert np.array_equal(asignar(_s_test, _cortes), np.asarray(_pdcut))
    # 8. Gini de bandas por conteos = sklearn con empates
    _b = asignar(s_dev, disenos["PDO"])
    assert np.isclose(gini_bandas(_b, y_dev), 2 * roc_auc_score(y_dev, -_b) - 1)
    # 9. invariancia del diseño PDO al PDO de la escala
    assert invariancia_pdo
    # 10. media bajo logit uniforme: fórmula cerrada = integración numérica
    _m = np.isfinite(pd_unif_np)
    assert np.allclose(pd_unif_np[_m], pd_unif_quad[_m], rtol=1e-8)
    # 11. IRLS numpy = statsmodels GLM (tasa suavizada)
    assert np.allclose(beta_suav_np, np.asarray(beta_suav_sm), atol=1e-6)
    # 12. potencia: cola binomial numpy = scipy; λ requerido ≈ 8 (m=2) y ≈ 28,6 (m=1,5)
    assert np.allclose(tabla_potencia["tamaño_numpy"], tabla_potencia["tamaño_real"], rtol=1e-8)
    assert np.allclose(tabla_potencia["potencia_numpy"], tabla_potencia["potencia_exacta"], rtol=1e-8)
    assert (tabla_potencia["tamaño_real"] <= 0.05 + 1e-12).all()
    assert np.isclose(lambda_requerido(2.0), 8.04, atol=0.01) and np.isclose(lambda_requerido(1.5), 28.64, atol=0.01)
    assert np.isclose(cola_binomial_np(8, 235, 0.0142), 0.0201, atol=5e-4)          # p = 0,020 de B2 (clase 5)
    # 13. migración: numpy = crosstab
    assert np.allclose(migracion_np, migracion_pd.to_numpy())
    # 14. inversión espuria: exacta ≈ Monte Carlo (error MC ~ 0,004)
    assert np.allclose(inversion_espuria["p_inversion_exacta"], inversion_espuria["p_inversion_MC"], atol=0.015)
    # 15. recalibración: cambia de banda exactamente la franja de ancho δ·factor bajo cada corte
    assert np.isclose(FRAC_CAMBIA_BANDA, FRAC_FRANJA)
    # 16. la DP restringida, si es factible, no tiene pares no separables
    if dp_factible:
        assert comparacion_disenos.loc["Óptimo (DP)", "pares_no_separables"] == 0
        assert comparacion_disenos.loc["Óptimo (DP)", "inversiones"] == 0
    "✅ Todos los checks del módulo pasan"
    return


if __name__ == "__main__":
    app.run()

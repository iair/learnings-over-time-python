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
    import time
    import marimo as mo
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy import stats
    from sklearn.metrics import roc_auc_score, roc_curve
    return mo, plt, roc_auc_score, roc_curve, sm, stats, time


@app.cell
def _(mo):
    mo.md(r"""
    # M12 · Poder discriminante: ROC, AUC, Gini, KS, CAP y su incertidumbre

    Notebook del módulo M12 de la Serie 2. Profundiza la clase 3 (Gini por muestra, reglas de caída
    0,10/0,15, tabla de deciles) y la clase 5 (AUC como probabilidad, KS, bootstrap del Gini y de su
    caída, caída relativa 20%/30%).

    Convenciones: target **1 = malo**; la PD del modelo es el «score de riesgo» (más PD = peor). El
    AUC se calcula siempre como $P(\text{PD}_{\text{malo}} > \text{PD}_{\text{bueno}}) + \tfrac12
    P(\text{empate})$, que es lo mismo que $P(\text{score}_{\text{bueno}} > \text{score}_{\text{malo}})$
    con el score PDO del curso (más puntos = menos riesgo).

    | § | Qué se demuestra |
    |---|---|
    | 1 | Modelo base sobre la cartera sintética (verdad conocida) |
    | 2 | AUC de cuatro formas: conteo $O(n^2)$, rangos (Mann–Whitney), sklearn, scipy; el ½ de los empates |
    | 3 | Gini = AR de la CAP = D de Somers |
    | 4 | KS: tres implementaciones, dónde se alcanza, cotas KS ↔ Gini |
    | 5 | El techo: Gini de la PD verdadera |
    | 6 | Varianza del AUC: Hanley–McNeil, DeLong (definición y rápido), bootstrap numpy y scipy |
    | 7 | DeLong pareado: «¿el modelo nuevo es mejor?» |
    | 8 | Caída por azar según nº de malos; crítica a 0,10/0,15 y a la caída relativa |
    | 9 | Optimismo (DEV→HO) vs deterioro (HO→OOT): nivel no es ranking |
    | 10 | Deciles, lift, captura y una métrica de negocio |
    | 11 | IV ↔ Gini en el mundo binormal (con cautela) |
    | 12 | Invariancia monótona (el δ no toca el Gini) y cuándo se rompe |
    | ✔ | Checks del módulo |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Código común de la serie

    Pegado verbatim desde `_spec/comun.py`: generador `generar_cartera()` con verdad conocida
    (`pd_verdadera`) y las herramientas `binear / tabla_woe / a_woe` del curso.
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
    ## 1. El modelo base: logística sobre WoE ajustada en DEV

    Siete variables (la octava candidata natural, `uso_tc_prom_12m`, sale con signo cambiado por su
    colinealidad con `uso_tc_prom_3m`; ver Serie 2 · M09). Bins y WoE de DEV, β de DEV, y se APLICAN
    a HO y OOT. El score usa el scaling del curso (PDO 20, 600 a odds 50:1). Además del modelo,
    guardamos `pd_verdadera`: la PD real del generador, que en la vida real nunca se conoce.
    """)
    return


@app.cell
def _(a_woe, generar_cartera, np, sm, tabla_woe):
    cartera = generar_cartera()
    VARS = ["uso_linea_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m",
            "antiguedad_meses", "carga_financiera", "deuda_otras_prom_12m", "consultas_6m"]
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    MAPAS = {v: tabla_woe(dev[v], dev["malo"])[0]["woe"] for v in VARS}

    def ajustar_logit(datos_dev, variables):
        """Logit sobre WoE de DEV (statsmodels). Devuelve el resultado ajustado."""
        _X = sm.add_constant(a_woe(datos_dev, variables, dev, MAPAS), has_constant="add")
        return sm.Logit(datos_dev["malo"].values, _X).fit(disp=0, maxiter=200)

    def pd_de(modelo, datos, variables):
        """PD del modelo sobre `datos` con cortes y WoE de DEV (como producción)."""
        _X = sm.add_constant(a_woe(datos, variables, dev, MAPAS), has_constant="add")
        return np.asarray(modelo.predict(_X))

    FACTOR = 20 / np.log(2)
    OFFSET = 600 - FACTOR * np.log(50)

    def score_de_pd(p):
        """Score PDO del curso (más puntos = menos riesgo)."""
        _p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
        return OFFSET + FACTOR * np.log((1 - _p) / _p)

    modelo_a = ajustar_logit(dev, VARS)
    MUESTRAS = {m: cartera[cartera["muestra"] == m].reset_index(drop=True)
                for m in ("DEV", "HO", "OOT")}
    Y = {m: MUESTRAS[m]["malo"].to_numpy().astype(int) for m in MUESTRAS}
    PD_A = {m: pd_de(modelo_a, MUESTRAS[m], VARS) for m in MUESTRAS}
    PD_V = {m: MUESTRAS[m]["pd_verdadera"].to_numpy() for m in MUESTRAS}
    SCORE_A = {m: score_de_pd(PD_A[m]) for m in MUESTRAS}
    return (
        FACTOR,
        MAPAS,
        MUESTRAS,
        OFFSET,
        PD_A,
        PD_V,
        SCORE_A,
        VARS,
        Y,
        ajustar_logit,
        cartera,
        dev,
        modelo_a,
        pd_de,
        score_de_pd,
    )


@app.cell
def _(MUESTRAS, PD_A, PD_V, SCORE_A, Y, modelo_a, pd, roc_auc_score):
    _filas = []
    for _m in MUESTRAS:
        _auc = roc_auc_score(Y[_m], PD_A[_m])
        _filas.append({"muestra": _m, "n": len(Y[_m]), "malos": int(Y[_m].sum()),
                       "tasa_malos": Y[_m].mean(), "auc": _auc, "gini": 2 * _auc - 1,
                       "gini_pd_verdadera": 2 * roc_auc_score(Y[_m], PD_V[_m]) - 1,
                       "score_medio": SCORE_A[_m].mean()})
    tabla_base = pd.DataFrame(_filas).set_index("muestra")
    _coefs = modelo_a.params
    assert (_coefs.drop("const") < 0).all(), "convención WoE: todos los β deben ser negativos"
    mo_tabla = tabla_base.round(4)
    mo_tabla
    return (tabla_base,)


@app.cell
def _(mo, tabla_base):
    mo.md(f"""
    **Lectura.** Gini DEV {tabla_base.loc['DEV','gini']:.3f} · HO {tabla_base.loc['HO','gini']:.3f} ·
    OOT {tabla_base.loc['OOT','gini']:.3f}, con {tabla_base.loc['DEV','malos']:,} / {tabla_base.loc['HO','malos']:,} /
    {tabla_base.loc['OOT','malos']:,} malos. Tres cosas que el resto del notebook explica:

    1. Caída DEV→HO = {tabla_base.loc['DEV','gini'] - tabla_base.loc['HO','gini']:+.3f}
       {"(**HO supera a DEV**)" if tabla_base.loc['HO','gini'] > tabla_base.loc['DEV','gini'] else ""}.
       El optimismo de selección existe *en expectativa*; una realización puntual puede tener
       cualquier signo sin que nada esté mal (§8 y §9).
    2. La columna `gini_pd_verdadera` es el **techo**: ni conociendo la PD real se ordena mejor
       ({tabla_base.loc['OOT','gini_pd_verdadera']:.3f} en OOT). Nuestro modelo está a
       {tabla_base.loc['OOT','gini_pd_verdadera'] - tabla_base.loc['OOT','gini']:.3f} de ese techo (§5).
    3. La tasa de malos sube en OOT ({tabla_base.loc['OOT','tasa_malos']:.1%} vs {tabla_base.loc['DEV','tasa_malos']:.1%}) por el
       deterioro macro plantado. El modelo no reordena a nadie, pero el Gini esperado baja algo porque
       **el techo depende de la tasa de malos** (§5 y §9): deterioro de nivel ≠ deterioro de ranking.

    El Gini sintético (~0,55) es de «admisión con bureau», más bajo que el 0,757 de Banco Austral.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. El AUC de cuatro formas

    $$\text{AUC}=\frac{1}{n_M n_B}\sum_{i\in M}\sum_{j\in B}\Big[\mathbb 1(s_i>s_j)+\tfrac12\mathbb 1(s_i=s_j)\Big]
    =\frac{U_M}{n_M n_B},\qquad U_M=R_M-\frac{n_M(n_M+1)}{2}$$

    donde $s$ es la PD, $M$ los malos, $B$ los buenos y $R_M$ la suma de **rangos medios** de los malos en
    la muestra combinada. Cuatro implementaciones: (1) conteo de pares $O(n_M n_B)$ en numpy;
    (2) rangos medios en numpy, $O(n\log n)$; (3) `sklearn.metrics.roc_auc_score` (trapecios sobre la ROC);
    (4) `scipy.stats.mannwhitneyu`. Deben coincidir a precisión de máquina.
    """)
    return


@app.cell
def _(np):
    def rangos_medios(x):
        """Rangos 1..n con empates promediados (midranks), en numpy puro."""
        x = np.asarray(x, float)
        n = len(x)
        orden = np.argsort(x, kind="mergesort")
        xs = x[orden]
        cambio = np.r_[True, xs[1:] != xs[:-1]]          # inicio de cada grupo de empates
        grupo = np.cumsum(cambio) - 1
        inicio = np.flatnonzero(cambio)                   # posición 0-based del inicio
        fin = np.r_[inicio[1:], n]                        # fin exclusivo
        rango_grupo = (inicio + 1 + fin) / 2              # promedio de rangos inicio+1 … fin
        r = np.empty(n)
        r[orden] = rango_grupo[grupo]
        return r

    def auc_rangos(y, s):
        """AUC = U de Mann-Whitney / (n_M n_B), con ½ para empates (vía rangos medios)."""
        y = np.asarray(y).astype(bool)
        m, n = y.sum(), (~y).sum()
        r = rangos_medios(s)
        return (r[y].sum() - m * (m + 1) / 2) / (m * n)

    def auc_conteo(y, s, peso_empate=0.5, bloque=256):
        """AUC por definición: recorre todos los pares malo-bueno (O(n_M n_B))."""
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, float)
        sm_, sb_ = s[y], s[~y]
        mayor, igual = 0.0, 0.0
        for k in range(0, len(sm_), bloque):
            bloque_m = sm_[k:k + bloque, None]
            mayor += (bloque_m > sb_).sum()
            igual += (bloque_m == sb_).sum()
        return (mayor + peso_empate * igual) / (len(sm_) * len(sb_))

    def gini(y, s):
        return 2 * auc_rangos(y, s) - 1
    return auc_conteo, auc_rangos, gini, rangos_medios


@app.cell
def _(PD_A, Y, auc_conteo, auc_rangos, np, pd, roc_auc_score, stats, time):
    _filas = []
    for _m in ("DEV", "HO", "OOT"):
        _y, _s = Y[_m], PD_A[_m]
        _t = time.perf_counter(); _a1 = auc_conteo(_y, _s); _t1 = time.perf_counter() - _t
        _t = time.perf_counter(); _a2 = auc_rangos(_y, _s); _t2 = time.perf_counter() - _t
        _a3 = roc_auc_score(_y, _s)
        _u = stats.mannwhitneyu(_s[_y == 1], _s[_y == 0], alternative="greater").statistic
        _a4 = _u / ((_y == 1).sum() * (_y == 0).sum())
        _filas.append({"muestra": _m, "conteo_O(n2)": _a1, "rangos_numpy": _a2,
                       "sklearn": _a3, "scipy_MW": _a4,
                       "ms_conteo": 1e3 * _t1, "ms_rangos": 1e3 * _t2})
    tabla_auc4 = pd.DataFrame(_filas).set_index("muestra")
    for _c in ("rangos_numpy", "sklearn", "scipy_MW"):
        assert np.allclose(tabla_auc4["conteo_O(n2)"], tabla_auc4[_c], atol=1e-12)
    tabla_auc4.round(6)
    return (tabla_auc4,)


@app.cell
def _(mo):
    mo.md(r"""
    **Empates: por qué el ½ importa.** La PD de un scorecard es discreta (combinaciones de bins), y
    el score entero lo es aún más. Si los empates cuentan 0 (convención pesimista) o 1 (optimista), el
    AUC cambia. Elige cómo se agrupa el score y mira cuánto se mueve. La guía de reporte del BCE pide
    calcular el AUC **sobre los grados de la escala** cuando la PD final sale de mapear el score a
    grados: ahí los empates son masivos.
    """)
    return


@app.cell
def _(mo):
    dd_agrupacion = mo.ui.dropdown(
        options=["PD continua", "score entero", "score de a 10 puntos", "8 bandas (cuantiles DEV)",
                 "4 bandas (cuantiles DEV)"],
        value="8 bandas (cuantiles DEV)", label="Agrupación del score")
    dd_agrupacion
    return (dd_agrupacion,)


@app.cell
def _(SCORE_A, Y, auc_conteo, dd_agrupacion, np, pd):
    def agrupar_score(s, modo, ref):
        """Aplica una agrupación (cortes definidos en `ref` = DEV) y devuelve un score «de riesgo»
        (más alto = peor) para calcular el AUC con la convención malo-positivo."""
        if modo == "PD continua":
            return -s
        if modo == "score entero":
            return -np.round(s)
        if modo == "score de a 10 puntos":
            return -np.floor(s / 10)
        _k = 8 if modo.startswith("8") else 4
        _cortes = np.quantile(ref, np.linspace(0, 1, _k + 1)[1:-1])
        return -np.searchsorted(_cortes, s, side="right").astype(float)

    _filas = []
    for _m in ("DEV", "OOT"):
        _g = agrupar_score(SCORE_A[_m], dd_agrupacion.value, SCORE_A["DEV"])
        _filas.append({"muestra": _m, "valores_distintos": len(np.unique(_g)),
                       "AUC empate=0": auc_conteo(Y[_m], _g, 0.0),
                       "AUC empate=½": auc_conteo(Y[_m], _g, 0.5),
                       "AUC empate=1": auc_conteo(Y[_m], _g, 1.0),
                       "AUC continuo": auc_conteo(Y[_m], -SCORE_A[_m], 0.5)})
    tabla_empates = pd.DataFrame(_filas).set_index("muestra")
    tabla_empates["gini_½"] = 2 * tabla_empates["AUC empate=½"] - 1
    tabla_empates["gini_continuo"] = 2 * tabla_empates["AUC continuo"] - 1
    tabla_empates.round(4)
    return agrupar_score, tabla_empates


@app.cell
def _(dd_agrupacion, mo, tabla_empates):
    _o = tabla_empates.loc["OOT"]
    mo.md(f"""
    **Lectura ({dd_agrupacion.value}, OOT).** Con {int(_o['valores_distintos'])} valores distintos, el AUC
    va de {_o['AUC empate=0']:.4f} (empates = 0) a {_o['AUC empate=1']:.4f} (empates = 1): una banda de
    {_o['AUC empate=1'] - _o['AUC empate=0']:.3f} en AUC ({2*(_o['AUC empate=1'] - _o['AUC empate=0']):.3f} en Gini)
    que es **pura convención**. Con ½ (la única que hace el AUC simétrico y coherente con Mann–Whitney)
    el Gini agrupado es {_o['gini_½']:.3f} contra {_o['gini_continuo']:.3f} continuo: agrupar en bandas
    **cuesta Gini** porque convierte pares ordenados en empates. Reportar el Gini «de la master scale»
    y el «del score» sin decir cuál es comparar peras con manzanas.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Gini = accuracy ratio (CAP) = D de Somers

    La curva CAP (perfil de exactitud acumulada, *cumulative accuracy profile*) pone en $x$ la
    fracción de población ordenada de peor a mejor y en $y$ la fracción de malos capturada. El
    *accuracy ratio* es $\text{AR}=a_R/a_P$: área entre CAP y diagonal, dividida por la del modelo
    perfecto $(1-\pi)/2$. En el `.md` se demuestra $\text{AR}=2\,\text{AUC}-1$; aquí se verifica con
    numpy (trapecios con interpolación lineal dentro de los empates, que es el ½) y con
    `scipy.stats.somersd` ($D_{S|Y}$ = (concordantes − discordantes)/pares con distinto $y$).
    """)
    return


@app.cell
def _(np):
    def curvas_roc_cap(y, s):
        """Puntos de ROC y CAP agrupando empates (s = riesgo: más alto = peor).
        Devuelve fpr, tpr, x_cap, y_cap (con el (0,0) inicial)."""
        y = np.asarray(y).astype(int)
        s = np.asarray(s, float)
        orden = np.argsort(-s, kind="mergesort")
        ys, ss = y[orden], s[orden]
        fin_grupo = np.r_[ss[1:] != ss[:-1], True]        # último elemento de cada valor
        malos_acum = np.cumsum(ys)[fin_grupo]
        n_acum = np.cumsum(np.ones_like(ys))[fin_grupo]
        buenos_acum = n_acum - malos_acum
        M, N = ys.sum(), len(ys)
        tpr = np.r_[0, malos_acum / M]
        fpr = np.r_[0, buenos_acum / (N - M)]
        x_cap = np.r_[0, n_acum / N]
        return fpr, tpr, x_cap, tpr.copy()

    def area_trapecio(x, y):
        return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))

    def accuracy_ratio(y, s):
        """AR de la CAP: (área bajo CAP − ½) / ((1 − π)/2)."""
        _, _, xc, yc = curvas_roc_cap(y, s)
        pi = np.mean(y)
        return (area_trapecio(xc, yc) - 0.5) / ((1 - pi) / 2)
    return accuracy_ratio, area_trapecio, curvas_roc_cap


@app.cell
def _(PD_A, SCORE_A, Y, accuracy_ratio, agrupar_score, gini, np, pd, stats):
    _filas = []
    for _m in ("DEV", "HO", "OOT"):
        for _nombre, _s in (("PD continua", PD_A[_m]),
                            ("8 bandas", agrupar_score(SCORE_A[_m], "8 bandas (cuantiles DEV)",
                                                       SCORE_A["DEV"]))):
            _filas.append({"muestra": _m, "score": _nombre, "gini_2AUC-1": gini(Y[_m], _s),
                           "AR_CAP_numpy": accuracy_ratio(Y[_m], _s),
                           "somersD_scipy": stats.somersd(Y[_m], _s).statistic})
    tabla_ar = pd.DataFrame(_filas).set_index(["muestra", "score"])
    assert np.allclose(tabla_ar["gini_2AUC-1"], tabla_ar["AR_CAP_numpy"], atol=1e-10)
    assert np.allclose(tabla_ar["gini_2AUC-1"], tabla_ar["somersD_scipy"], atol=1e-10)
    tabla_ar.round(6)
    return (tabla_ar,)


@app.cell
def _(mo):
    dd_muestra = mo.ui.dropdown(options=["DEV", "HO", "OOT"], value="OOT", label="Muestra para las curvas")
    dd_muestra
    return (dd_muestra,)


@app.cell
def _(PD_A, PD_V, Y, curvas_roc_cap, dd_muestra, gini, plt):
    _m = dd_muestra.value
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for _nombre, _s, _c in (("modelo", PD_A[_m], "#1f5fbf"), ("PD verdadera (techo)", PD_V[_m], "#b3261e")):
        _f, _t, _xc, _yc = curvas_roc_cap(Y[_m], _s)
        _g = gini(Y[_m], _s)
        _a1.plot(_f, _t, color=_c, label=f"{_nombre}: Gini {_g:.3f}")
        _a2.plot(_xc, _yc, color=_c, label=f"{_nombre}: AR {_g:.3f}")
    _pi = Y[_m].mean()
    _a1.plot([0, 1], [0, 1], ls="--", color="grey", label="azar")
    _a2.plot([0, 1], [0, 1], ls="--", color="grey", label="azar")
    _a2.plot([0, _pi, 1], [0, 1, 1], ls=":", color="black", label="perfecto")
    _a1.set_xlabel("FPR: fracción de buenos rechazados"); _a1.set_ylabel("TPR: fracción de malos capturados")
    _a1.set_title(f"ROC · {_m}"); _a1.legend(fontsize=8)
    _a2.set_xlabel("fracción de la población (peor primero)"); _a2.set_ylabel("fracción de malos capturados")
    _a2.set_title(f"CAP · {_m} (π = {_pi:.1%})"); _a2.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    La CAP perfecta sube hasta 1 en $x=\pi$: con π ≈ 11–15% su «rodilla» está cerca del eje; por eso
    el AR se normaliza por $(1-\pi)/2$ y no por ½. La curva roja (PD verdadera) no llega al perfecto:
    los eventos son aleatorios aun conociendo la PD (§5).

    ## 4. KS: tres implementaciones, dónde se alcanza y sus cotas con el Gini

    $\text{KS}=\max_t |F_M(t)-F_B(t)| = \max_t(\text{TPR}(t)-\text{FPR}(t))$ (el índice J de Youden
    máximo). Implementaciones: ECDF en numpy, `scipy.stats.ks_2samp`, y máx(TPR−FPR) de `roc_curve`.
    En el `.md` se deriva que el máximo se alcanza donde las densidades de malos y buenos se cruzan,
    es decir, donde **la PD bien calibrada iguala la tasa de malos de la muestra**; y que con ROC
    cóncava $\text{KS}\le \text{Gini}\le \text{KS}(2-\text{KS})$.
    """)
    return


@app.cell
def _(np):
    def ks_numpy(y, s):
        """KS = max_t |F_M(t) − F_B(t)| sobre todos los valores observados. Devuelve (KS, t*)."""
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, float)
        malos, buenos = np.sort(s[y]), np.sort(s[~y])
        t = np.unique(s)
        f_m = np.searchsorted(malos, t, side="right") / len(malos)
        f_b = np.searchsorted(buenos, t, side="right") / len(buenos)
        dif = np.abs(f_m - f_b)
        k = int(np.argmax(dif))
        return float(dif[k]), float(t[k])
    return (ks_numpy,)


@app.cell
def _(PD_A, SCORE_A, Y, gini, ks_numpy, np, pd, roc_curve, stats):
    _filas = []
    for _m in ("DEV", "HO", "OOT"):
        _y, _p = Y[_m], PD_A[_m]
        _ks_np, _t = ks_numpy(_y, _p)
        _ks_sc = stats.ks_2samp(_p[_y == 1], _p[_y == 0]).statistic
        _f, _tp, _ = roc_curve(_y, _p)
        _ks_roc = float(np.max(_tp - _f))
        _g = gini(_y, _p)
        _filas.append({"muestra": _m, "KS_numpy": _ks_np, "KS_scipy": _ks_sc, "KS_roc": _ks_roc,
                       "PD_en_KS": _t, "tasa_malos": _y.mean(),
                       "score_en_KS": float(np.min(SCORE_A[_m][_p <= _t])) if np.any(_p <= _t) else np.nan,
                       "gini": _g, "cota_inf_KS": _ks_np, "cota_sup_KS(2-KS)": _ks_np * (2 - _ks_np)})
    tabla_ks = pd.DataFrame(_filas).set_index("muestra")
    assert np.allclose(tabla_ks["KS_numpy"], tabla_ks["KS_scipy"], atol=1e-12)
    assert np.allclose(tabla_ks["KS_numpy"], tabla_ks["KS_roc"], atol=1e-12)
    tabla_ks.round(4)
    return (tabla_ks,)


@app.cell
def _(PD_A, Y, dd_muestra, np, plt, tabla_ks):
    _m = dd_muestra.value
    _y, _p = Y[_m], PD_A[_m]
    _t = np.unique(_p)
    _fm = np.searchsorted(np.sort(_p[_y == 1]), _t, side="right") / (_y == 1).sum()
    _fb = np.searchsorted(np.sort(_p[_y == 0]), _t, side="right") / (_y == 0).sum()
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _ax.plot(_t, _fb, color="#1f5fbf", label="F buenos")
    _ax.plot(_t, _fm, color="#b3261e", label="F malos")
    _ax.plot(_t, _fb - _fm, color="black", lw=1, label="diferencia")
    _ax.axvline(tabla_ks.loc[_m, "PD_en_KS"], ls="--", color="grey",
                label=f"KS = {tabla_ks.loc[_m, 'KS_numpy']:.3f} en PD {tabla_ks.loc[_m, 'PD_en_KS']:.3f}")
    _ax.axvline(_y.mean(), ls=":", color="#b3261e", label=f"tasa de malos {_y.mean():.3f}")
    _ax.set_xscale("log"); _ax.set_xlabel("PD del modelo (escala log)"); _ax.set_ylabel("proporción acumulada")
    _ax.set_title(f"KS · {_m}"); _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, tabla_ks):
    _d = tabla_ks.loc["DEV"]
    mo.md(f"""
    **Lectura.** Las tres implementaciones del KS coinciden exacto (es la misma estadística). En DEV,
    donde el modelo está calibrado por construcción (la logística con intercepto reproduce la tasa
    media), el KS se alcanza en PD = {_d['PD_en_KS']:.3f}, muy cerca de la tasa de malos
    {_d['tasa_malos']:.3f}: el punto donde la razón de verosimilitudes vale 1. En OOT, con el nivel
    desplazado por el deterioro macro, la PD del punto KS ({tabla_ks.loc['OOT','PD_en_KS']:.3f}) queda por
    debajo de la tasa observada ({tabla_ks.loc['OOT','tasa_malos']:.3f}): el KS se sigue alcanzando donde
    la PD **verdadera** iguala la tasa, pero nuestro modelo la subestima. Las cotas
    KS ≤ Gini ≤ KS(2−KS) se cumplen en las tres muestras (la ROC empírica de un modelo razonable es
    casi cóncava). El «KS en el cutoff» del curso (560, banda C2) coincide solo si la PD de
    equilibrio del negocio se parece a la tasa de malos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. El techo: el Gini de la PD verdadera

    Aun conociendo la PD exacta de cada cliente, el resultado es una Bernoulli: un cliente con PD 30%
    es bueno el 70% de las veces. El AUC **esperado** de la PD verdadera es

    $$\text{AUC}^*=\frac{\sum_{i\ne j} p_i(1-p_j)\,[\mathbb 1(p_i>p_j)+\tfrac12\mathbb 1(p_i=p_j)]}
    {\sum_{i\ne j}p_i(1-p_j)}$$

    (razón de esperanzas; aproxima la esperanza de la razón para $n$ grande). Se calcula en
    $O(n\log n)$ ordenando por $p$. Luego, un experimento de juguete: log-odds verdadero
    $\eta\sim N(\mu,\sigma^2)$. El techo depende **solo de la dispersión del riesgo real** σ (y un poco
    de μ): ninguna técnica de modelado lo supera.
    """)
    return


@app.cell
def _(np):
    def auc_esperado(p):
        """AUC esperado de ordenar por la PD verdadera p (razón de esperanzas, O(n log n))."""
        p = np.sort(np.asarray(p, float))
        q = 1 - p
        # para cada i: suma de q_j con p_j < p_i (estrictamente) y con p_j = p_i (j ≠ i)
        q_acum = np.r_[0, np.cumsum(q)]
        izq = np.searchsorted(p, p, side="left")
        der = np.searchsorted(p, p, side="right")
        q_menor = q_acum[izq]
        q_igual = q_acum[der] - q_acum[izq] - q          # excluye j = i
        num = np.sum(p * (q_menor + 0.5 * q_igual))
        den = p.sum() * q.sum() - np.sum(p * q)
        return num / den

    def auc_esperado_conteo(p):
        """Misma cantidad por definición O(n²) (para verificar)."""
        p = np.asarray(p, float)
        w = p[:, None] * (1 - p)[None, :]
        np.fill_diagonal(w, 0)
        ind = (p[:, None] > p[None, :]) + 0.5 * (p[:, None] == p[None, :])
        return (w * ind).sum() / w.sum()
    return auc_esperado, auc_esperado_conteo


@app.cell
def _(PD_A, PD_V, Y, auc_esperado, gini, np, pd):
    def auc_esperado_modelo(s, p):
        """AUC esperado de ordenar por s cuando la verdad es p (misma lógica, pares ponderados)."""
        orden = np.argsort(s, kind="mergesort")
        s_, p_ = np.asarray(s)[orden], np.asarray(p)[orden]
        q_ = 1 - p_
        q_acum = np.r_[0, np.cumsum(q_)]
        izq = np.searchsorted(s_, s_, side="left")
        der = np.searchsorted(s_, s_, side="right")
        q_igual = q_acum[der] - q_acum[izq] - q_
        num = np.sum(p_ * (q_acum[izq] + 0.5 * q_igual))
        return num / (p_.sum() * q_.sum() - np.sum(p_ * q_))
    _filas = []
    for _m in ("DEV", "HO", "OOT"):
        _filas.append({"muestra": _m, "gini_techo_esperado": 2 * auc_esperado(PD_V[_m]) - 1,
                       "gini_pd_verdadera_obs": gini(Y[_m], PD_V[_m]),
                       "gini_modelo": gini(Y[_m], PD_A[_m]),
                       "gini_modelo_esperado": 2 * auc_esperado_modelo(PD_A[_m], PD_V[_m]) - 1})
    tabla_techo = pd.DataFrame(_filas).set_index("muestra")
    tabla_techo["brecha_modelo_vs_techo"] = tabla_techo["gini_techo_esperado"] - tabla_techo["gini_modelo_esperado"]

    tabla_techo.round(4)
    return auc_esperado_modelo, tabla_techo


@app.cell
def _(mo):
    sl_sigma = mo.ui.slider(0.25, 3.0, step=0.05, value=1.0, label="σ del log-odds verdadero")
    sl_mu = mo.ui.slider(-4.0, -0.5, step=0.1, value=-2.2, label="μ del log-odds (fija la tasa media)")
    mo.hstack([sl_sigma, sl_mu])
    return sl_mu, sl_sigma


@app.cell
def _(auc_esperado, gini, np, plt, sl_mu, sl_sigma):
    _rng = np.random.default_rng(12)
    _grilla = np.linspace(0.25, 3.0, 23)
    _techo = [2 * auc_esperado(1 / (1 + np.exp(-(sl_mu.value + _sg * _rng.standard_normal(6000))))) - 1
              for _sg in _grilla]
    _eta = sl_mu.value + sl_sigma.value * _rng.standard_normal(20000)
    _p = 1 / (1 + np.exp(-_eta))
    _y = (_rng.random(20000) < _p).astype(int)
    techo_juguete = {"sigma": sl_sigma.value, "tasa": _p.mean(),
                     "gini_techo": 2 * auc_esperado(_p) - 1, "gini_realizado": gini(_y, _p)}
    _fig, _ax = plt.subplots(figsize=(7.5, 3.6))
    _ax.plot(_grilla, _techo, color="#1f5fbf", label="Gini techo esperado")
    _ax.scatter([sl_sigma.value], [techo_juguete["gini_realizado"]], color="#b3261e", zorder=3,
                label="Gini realizado (una muestra, n = 20.000)")
    _ax.axhline(1, ls=":", color="grey")
    _ax.set_xlabel("σ del log-odds verdadero"); _ax.set_ylabel("Gini")
    _ax.set_title(f"Techo de Gini vs dispersión del riesgo real (μ = {sl_mu.value})")
    _ax.legend(fontsize=8); _fig.tight_layout()
    _fig
    return (techo_juguete,)


@app.cell
def _(mo, tabla_techo, techo_juguete):
    mo.md(f"""
    **Lectura.** En la cartera sintética el techo esperado en OOT es
    {tabla_techo.loc['OOT','gini_techo_esperado']:.3f}; el modelo, evaluado contra la misma verdad, llega a
    {tabla_techo.loc['OOT','gini_modelo_esperado']:.3f} (brecha {tabla_techo.loc['OOT','brecha_modelo_vs_techo']:.3f}).
    **Esa brecha es una cota superior de lo que cualquier modelo —ML incluido— podría ganar**, y
    ni siquiera es alcanzable: parte es heterogeneidad no observada (el término $N(0;\\,0{{,}}35^2)$ del
    log-odds del generador), que ninguna variable disponible captura. En el juguete, con σ = {techo_juguete['sigma']:.2f} y tasa {techo_juguete['tasa']:.1%}, el techo
    es {techo_juguete['gini_techo']:.3f}. Para un techo > 0,85 con μ = −2,2 necesitas σ ≈ 3: un cuarto de la
    cartera con PD bajo 1,5% y otro cuarto sobre 45%. Mueve μ: con σ fijo, **el techo baja cuando sube
    la tasa media** (en el límite de eventos raros, $\\text{{AUC}}^* \\to \\Phi(\\sigma/\\sqrt2)$). En admisión eso casi no ocurre; por eso un Gini > 0,85
    es sospechoso de fuga (el target «se ve» en las variables) o de un target trivial.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. La varianza del AUC: Hanley–McNeil, DeLong y bootstrap

    - **Hanley & McNeil (1982)**: fórmula cerrada que supone una forma (exponencial) para las
      distribuciones: $Q_1=A/(2-A)$, $Q_2=2A^2/(1+A)$.
    - **DeLong et al. (1988)**: no paramétrico, vía componentes estructurales (valores de colocación)
      $V_{10}(i)$ y $V_{01}(j)$. Se implementa dos veces: por definición $O(n_M n_B)$ y rápido con rangos
      medios $O(n\log n)$ (Sun & Xu, 2014).
    - **Bootstrap percentil**: numpy (como el curso) y `scipy.stats.bootstrap(paired=True)`.

    Ninguna librería estándar de Python (sklearn, scipy, statsmodels) trae DeLong: por eso la
    comparación «librería» es contra el bootstrap de scipy, que estima lo mismo por otra vía.
    """)
    return


@app.cell
def _(np, rangos_medios):
    def hanley_mcneil(auc, n_malos, n_buenos):
        """Error estándar del AUC según Hanley & McNeil (1982)."""
        q1 = auc / (2 - auc)
        q2 = 2 * auc ** 2 / (1 + auc)
        var = (auc * (1 - auc) + (n_malos - 1) * (q1 - auc ** 2)
               + (n_buenos - 1) * (q2 - auc ** 2)) / (n_malos * n_buenos)
        return float(np.sqrt(var))

    def delong_definicion(y, s):
        """DeLong por definición: matriz ψ de pares (n_M × n_B). Devuelve (AUC, var)."""
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, float)
        x_, z_ = s[y][:, None], s[~y][None, :]
        psi = (x_ > z_) + 0.5 * (x_ == z_)
        v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
        return float(psi.mean()), float(v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01))

    def delong_rapido(y, S):
        """DeLong rápido (Sun & Xu 2014) para k scores sobre las mismas observaciones.
        S: array (k, n). Devuelve aucs (k,) y matriz de covarianzas (k, k)."""
        y = np.asarray(y).astype(bool)
        S = np.atleast_2d(np.asarray(S, float))
        m, n = y.sum(), (~y).sum()
        k = S.shape[0]
        v10 = np.empty((k, m))
        v01 = np.empty((k, n))
        for r in range(k):
            xm, xb = S[r, y], S[r, ~y]
            tz = rangos_medios(np.r_[xm, xb])
            tx, ty = rangos_medios(xm), rangos_medios(xb)
            v10[r] = (tz[:m] - tx) / n               # fracción de buenos bajo cada malo (+½ empates)
            v01[r] = 1 - (tz[m:] - ty) / m           # fracción de malos sobre cada bueno (+½ empates)
        aucs = v10.mean(axis=1)
        cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
        return aucs, cov
    return delong_definicion, delong_rapido, hanley_mcneil


@app.cell
def _(mo):
    sl_B = mo.ui.slider(100, 2000, step=100, value=400, label="B (réplicas bootstrap)")
    sl_B
    return (sl_B,)


@app.cell
def _(
    PD_A,
    Y,
    auc_rangos,
    delong_definicion,
    delong_rapido,
    hanley_mcneil,
    np,
    pd,
    sl_B,
    stats,
):
    def bootstrap_auc(y, s, B, semilla):
        """Bootstrap percentil del AUC (numpy, remuestreo simple como el curso)."""
        rng = np.random.default_rng(semilla)
        y = np.asarray(y); s = np.asarray(s)
        out = np.empty(B)
        b = 0
        while b < B:
            idx = rng.integers(0, len(y), len(y))
            if 0 < y[idx].sum() < len(y):
                out[b] = auc_rangos(y[idx], s[idx]); b += 1
        return out

    _filas = []
    BOOT_AUC = {}
    for _k, _m in enumerate(("DEV", "HO", "OOT")):
        _y, _s = Y[_m], PD_A[_m]
        _nm, _nb = int(_y.sum()), int((1 - _y).sum())
        _a_def, _v_def = delong_definicion(_y, _s)
        _a_rap, _c_rap = delong_rapido(_y, _s)
        BOOT_AUC[_m] = bootstrap_auc(_y, _s, sl_B.value, 20260908 + _k)
        _res = stats.bootstrap((_y, _s), lambda a, b: auc_rangos(a, b), paired=True,
                               vectorized=False, n_resamples=sl_B.value, method="percentile",
                               rng=np.random.default_rng(777 + _k))
        _auc = float(_a_rap[0])
        _se_dl = float(np.sqrt(_c_rap[0, 0]))
        _filas.append({
            "muestra": _m, "malos": _nm, "AUC": _auc,
            "SE_HM": hanley_mcneil(_auc, _nm, _nb),
            "SE_DeLong_def": np.sqrt(_v_def), "SE_DeLong_rapido": _se_dl,
            "SE_boot_numpy": BOOT_AUC[_m].std(ddof=1),
            "SE_boot_scipy": float(_res.standard_error),
            "IC_gini_DeLong": f"[{2*(_auc-1.96*_se_dl)-1:.3f}; {2*(_auc+1.96*_se_dl)-1:.3f}]",
            "IC_gini_boot": "[{:.3f}; {:.3f}]".format(*(2 * np.percentile(BOOT_AUC[_m], [2.5, 97.5]) - 1)),
            "IC_gini_scipy": f"[{2*_res.confidence_interval.low-1:.3f}; {2*_res.confidence_interval.high-1:.3f}]",
        })
        assert np.isclose(_a_def, _auc, atol=1e-12) and np.isclose(_v_def, _c_rap[0, 0], rtol=1e-10)
    tabla_var = pd.DataFrame(_filas).set_index("muestra")
    tabla_var
    return BOOT_AUC, bootstrap_auc, tabla_var


@app.cell
def _(mo, sl_B, tabla_var):
    _o = tabla_var.loc["OOT"]
    mo.md(f"""
    **Lectura (B = {sl_B.value}).** DeLong por definición y rápido coinciden a precisión de máquina (la
    matriz de pares y los rangos medios son la misma cuenta). DeLong y bootstrap estiman lo mismo por
    caminos distintos: en OOT, SE del AUC {_o['SE_DeLong_rapido']:.4f} (DeLong) vs {_o['SE_boot_numpy']:.4f}
    (bootstrap numpy) vs {_o['SE_boot_scipy']:.4f} (scipy). Hanley–McNeil da {_o['SE_HM']:.4f}: su supuesto
    exponencial es una aproximación y aquí **sobreestima** el SE en torno a
    {100*(_o['SE_HM']/_o['SE_DeLong_rapido']-1):.0f}%. Para el Gini, SE × 2: el IC 95% del Gini OOT es
    {_o['IC_gini_DeLong']} por DeLong y {_o['IC_gini_boot']} por bootstrap. DeLong cuesta
    milisegundos y es determinista (no hay semilla que declarar); el bootstrap generaliza a KS,
    lift o cualquier métrica sin fórmula.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. DeLong pareado: la pregunta correcta para «¿el modelo nuevo es mejor?»

    Dos modelos evaluados **sobre los mismos clientes** tienen AUC correlacionados: los clientes
    difíciles son difíciles para ambos. La varianza de la diferencia es
    $\operatorname{Var}(A_1-A_2)=\operatorname{Var}(A_1)+\operatorname{Var}(A_2)-2\operatorname{Cov}(A_1,A_2)$,
    y la covarianza es grande y positiva. Ignorarla (comparar dos IC que se solapan) es el error más
    común de los comités. Elige qué variable le quitamos al modelo A para construir el «modelo B»,
    reajustado en DEV y evaluado en OOT.
    """)
    return


@app.cell
def _(VARS, mo):
    dd_quitar = mo.ui.dropdown(options=VARS, value="consultas_6m", label="Variable que se quita (modelo B)")
    dd_quitar
    return (dd_quitar,)


@app.cell
def _(
    MUESTRAS,
    PD_A,
    VARS,
    Y,
    ajustar_logit,
    auc_rangos,
    dd_quitar,
    delong_rapido,
    dev,
    np,
    pd,
    pd_de,
    stats,
):
    VARS_B = [v for v in VARS if v != dd_quitar.value]
    modelo_b = ajustar_logit(dev, VARS_B)
    PD_B = {m: pd_de(modelo_b, MUESTRAS[m], VARS_B) for m in ("DEV", "HO", "OOT")}

    def delong_pareado(y, s1, s2):
        """Test de DeLong para AUC1 − AUC2 sobre las mismas observaciones."""
        aucs, cov = delong_rapido(y, np.vstack([s1, s2]))
        dif = aucs[0] - aucs[1]
        var_dif = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        z = dif / np.sqrt(var_dif)
        return {"auc1": aucs[0], "auc2": aucs[1], "dif_gini": 2 * dif,
                "se_dif_gini_pareado": 2 * np.sqrt(var_dif),
                "se_dif_gini_ignorando_cov": 2 * np.sqrt(cov[0, 0] + cov[1, 1]),
                "correlacion": cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
                "z": z, "p_bilateral": 2 * stats.norm.sf(abs(z))}

    _rng = np.random.default_rng(4242)
    _y, _s1, _s2 = Y["OOT"], PD_A["OOT"], PD_B["OOT"]
    _dif_boot = np.empty(400)
    for _b in range(400):
        _i = _rng.integers(0, len(_y), len(_y))
        _dif_boot[_b] = 2 * (auc_rangos(_y[_i], _s1[_i]) - auc_rangos(_y[_i], _s2[_i]))
    resultado_pareado = delong_pareado(_y, _s1, _s2)
    resultado_pareado["se_dif_gini_boot_pareado"] = _dif_boot.std(ddof=1)
    resultado_pareado["IC95_boot"] = np.percentile(_dif_boot, [2.5, 97.5]).round(4).tolist()
    pd.Series(resultado_pareado)
    return PD_B, delong_pareado, resultado_pareado


@app.cell
def _(dd_quitar, mo, resultado_pareado):
    _r = resultado_pareado
    _veredicto = "significativa al 5%" if _r["p_bilateral"] < 0.05 else "NO significativa al 5%"
    mo.md(f"""
    **Lectura (quitando `{dd_quitar.value}`).** Gini A − Gini B en OOT = {_r['dif_gini']:+.4f}, SE pareado
    {_r['se_dif_gini_pareado']:.4f} (bootstrap pareado {_r['se_dif_gini_boot_pareado']:.4f}), z = {_r['z']:.2f},
    p = {_r['p_bilateral']:.4f}: diferencia {_veredicto}. Correlación entre los dos AUC:
    {_r['correlacion']:.3f}. Si se ignora la covarianza, el SE de la diferencia sería
    {_r['se_dif_gini_ignorando_cov']:.4f}, **{_r['se_dif_gini_ignorando_cov']/_r['se_dif_gini_pareado']:.1f} veces** mayor: la
    comparación «¿se solapan los IC?» declara empate diferencias que son reales. Prueba quitar
    `meses_desde_mora_12m` o `uso_linea_prom_12m` (diferencias grandes y significativas),
    `carga_financiera` (diferencia de ~0,006 en Gini, **significativa** porque la correlación es 0,995:
    estadísticamente real, económicamente irrelevante) y `deuda_otras_prom_12m` (B sale mejor en OOT).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. ¿Cuánto cae el Gini por puro azar?

    Experimento con verdad conocida: fijamos el modelo A y la población (DEV+HO, mismos meses, sin
    deterioro). Sacamos **dos** muestras independientes con el mismo número esperado de malos,
    **regeneramos los resultados** $y\sim\text{Bernoulli}(\text{pd\_verdadera})$ y medimos el Gini de cada una.
    No hay optimismo ni deterioro: toda diferencia es ruido muestral. ¿Con qué frecuencia la
    «caída» supera 0,10 (regla DEV→HO) o 0,15 (DEV→OOT)?
    """)
    return


@app.cell
def _(MUESTRAS, PD_A, np, stats):
    POOL_S = np.r_[PD_A["DEV"], PD_A["HO"]]
    POOL_P = np.r_[MUESTRAS["DEV"]["pd_verdadera"].to_numpy(), MUESTRAS["HO"]["pd_verdadera"].to_numpy()]

    def ginis_simulados(n_malos, R, semilla, s=POOL_S, p=POOL_P):
        """R Ginis de muestras de tamaño n = n_malos/π con resultados regenerados de la PD verdadera."""
        rng = np.random.default_rng(semilla)
        n = int(round(n_malos / p.mean()))
        idx = rng.integers(0, len(s), (R, n))
        y = rng.random((R, n)) < p[idx]
        r = stats.rankdata(s[idx], axis=1)               # rangos medios por fila
        m = y.sum(axis=1)
        u = (r * y).sum(axis=1) - m * (m + 1) / 2
        return 2 * u / (m * (n - m)) - 1
    return POOL_P, POOL_S, ginis_simulados


@app.cell
def _(mo):
    sl_malos = mo.ui.slider(20, 2000, step=10, value=120, label="Nº esperado de malos por muestra")
    sl_malos
    return (sl_malos,)


@app.cell
def _(ginis_simulados, np, plt, sl_malos):
    _g1 = ginis_simulados(sl_malos.value, 600, 1)
    _g2 = ginis_simulados(sl_malos.value, 600, 2)
    _caida = _g1 - _g2
    caida_slider = {"sd_gini": _g1.std(ddof=1), "sd_caida": _caida.std(ddof=1),
                    "p_caida_010": float(np.mean(_caida > 0.10)),
                    "p_caida_015": float(np.mean(_caida > 0.15)),
                    "p_rel_20": float(np.mean(1 - _g2 / _g1 > 0.20)),
                    "q95": float(np.quantile(_caida, 0.95))}
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _ax.hist(_caida, bins=40, color="#9db7e0", edgecolor="white")
    for _v, _c, _l in ((0.10, "#e0a100", "regla 0,10"), (0.15, "#b3261e", "regla 0,15"),
                       (caida_slider["q95"], "black", "percentil 95 del azar")):
        _ax.axvline(_v, color=_c, ls="--", label=_l)
    _ax.set_xlabel("Gini muestra 1 − Gini muestra 2 (sin ningún deterioro)")
    _ax.set_ylabel("réplicas")
    _ax.set_title(f"Caída por azar con ~{sl_malos.value} malos por muestra")
    _ax.legend(fontsize=8); _fig.tight_layout()
    _fig
    return (caida_slider,)


@app.cell
def _(caida_slider, mo, sl_malos):
    _c = caida_slider
    mo.md(f"""
    **Lectura (~{sl_malos.value} malos).** SD del Gini = {_c['sd_gini']:.3f}; SD de la diferencia =
    {_c['sd_caida']:.3f} (≈ √2 veces, porque las muestras son independientes). Sin deterioro alguno, la
    caída supera 0,10 en {_c['p_caida_010']:.1%} de los casos y 0,15 en {_c['p_caida_015']:.1%}; la caída
    relativa supera 20% en {_c['p_rel_20']:.1%}. El percentil 95 del ruido es {_c['q95']:.3f}: una regla
    que no dependa del número de malos es demasiado laxa con carteras grandes y dispara falsas
    alarmas con carteras chicas (motos en una sucursal, un canal nuevo).
    """)
    return


@app.cell
def _(POOL_P, ginis_simulados, hanley_mcneil, np, pd):
    _filas = []
    for _k, _nm in enumerate((30, 60, 120, 250, 500, 1000, 2000)):
        _g1 = ginis_simulados(_nm, 400, 100 + _k)
        _g2 = ginis_simulados(_nm, 400, 200 + _k)
        _d = _g1 - _g2
        _auc = (np.mean(_g1) + 1) / 2
        _nb = _nm * (1 - POOL_P.mean()) / POOL_P.mean()
        _filas.append({"malos": _nm, "gini_medio": _g1.mean(), "sd_gini": _g1.std(ddof=1),
                       "sd_gini_HM": 2 * hanley_mcneil(_auc, _nm, _nb),
                       "sd_caida": _d.std(ddof=1),
                       "umbral_95_unilateral": np.quantile(_d, 0.95),
                       "P(caida>0,10)": np.mean(_d > 0.10), "P(caida>0,15)": np.mean(_d > 0.15),
                       "P(caida_rel>20%)": np.mean(1 - _g2 / _g1 > 0.20)})
    tabla_caida = pd.DataFrame(_filas).set_index("malos")
    tabla_caida.round(3)
    return (tabla_caida,)


@app.cell
def _(mo, tabla_caida):
    mo.md(f"""
    **La tabla que reemplaza a la regla fija.** El ruido de la caída escala como $1/\\sqrt{{n_M}}$:
    umbral 95% ≈ {tabla_caida.loc[60,'umbral_95_unilateral']:.2f} con 60 malos, ≈ {tabla_caida.loc[250,'umbral_95_unilateral']:.2f} con 250 y
    ≈ {tabla_caida.loc[2000,'umbral_95_unilateral']:.3f} con 2.000. Con 30 malos, la regla 0,10 salta por azar
    {tabla_caida.loc[30,'P(caida>0,10)']:.0%} de las veces; con 1.000 malos, una caída real de 0,06 pasaría
    inadvertida bajo 0,10 aunque esté **muy** por sobre el ruido. Nota: la SD de Hanley–McNeil
    (columna `sd_gini_HM`) sobreestima levemente la simulada, igual que en §6.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Optimismo (DEV→HO) vs deterioro (HO→OOT)

    **(a) Optimismo.** Ajustamos el modelo en un DEV chico (n de malos variable) con 7 variables reales
    y, en una segunda versión, con 10 variables de **puro ruido** adicionales (WoE incluido, que
    también sobreajusta). Medimos Gini en DEV (in-sample) y en una muestra de prueba independiente
    con resultados regenerados. El optimismo es sesgo **sistemático** que crece con la complejidad y
    decrece con $n_M$.

    **(b) Deterioro de nivel.** Regeneramos la cartera con `deterioro` = 0 / 0,35 / 1,0: la macro sube
    el log-odds de todos los clientes de 2025 en una constante. ¿Cae el Gini OOT?

    **(c) Deterioro de ranking** (concept drift). En OOT, reescribimos la verdad quitándole una fracción
    λ del efecto de `uso_linea_prom_12m`: la variable más importante del modelo deja de discriminar.
    """)
    return


@app.cell
def _(MUESTRAS, gini, np, pd, sm):
    _pool = pd.concat([MUESTRAS["DEV"], MUESTRAS["HO"]], ignore_index=True)
    _REALES = ["uso_linea_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m", "antiguedad_meses",
               "carga_financiera", "deuda_otras_prom_12m", "consultas_6m"]

    def woe_rapido(x_dev, y_dev, x_aplicar, bins=5, umbral_moda=0.35):
        """Versión numpy simplificada de binear + tabla_woe (misma lógica: cuantiles de DEV, bin
        propio para la moda si pesa > 35%, bin MISSING, suavizado +0,5). ~100× más rápida; se usa
        solo en este experimento de simulación, que ajusta decenas de modelos."""
        xd = np.asarray(x_dev, float)
        vals, cnt = np.unique(xd[~np.isnan(xd)], return_counts=True)
        moda = vals[np.argmax(cnt)]
        con_moda = cnt.max() / len(xd) > umbral_moda
        resto = xd[~np.isnan(xd) & ((xd != moda) if con_moda else True)]
        cortes = np.unique(np.quantile(resto, np.linspace(0, 1, bins + 1)))[1:-1]

        def indice(x):
            x = np.asarray(x, float)
            k = np.searchsorted(cortes, x, side="left")          # intervalos (a, b] como pd.cut
            if con_moda:
                k = np.where(x == moda, len(cortes) + 1, k)
            return np.where(np.isnan(x), len(cortes) + 2, k)

        kd = indice(xd)
        nb = len(cortes) + 3
        malos = np.bincount(kd, weights=y_dev, minlength=nb)
        buenos = np.bincount(kd, minlength=nb) - malos
        presente = (malos + buenos) > 0
        L = presente.sum()
        p_m = (malos + 0.5) / (malos.sum() + 0.5 * L)
        p_b = (buenos + 0.5) / (buenos.sum() + 0.5 * L)
        woe = np.where(presente, np.log(p_b / p_m), 0.0)
        return woe[kd], woe[indice(x_aplicar)]

    def _optimismo(n_malos, con_ruido, semilla, n_prueba=5000):
        """Gini in-sample (DEV chico) y fuera de muestra, con resultados regenerados."""
        rng = np.random.default_rng(semilla)
        datos = _pool.copy()
        ruido = [f"ruido_{k}" for k in range(10)] if con_ruido else []
        for c in ruido:
            datos[c] = rng.standard_normal(len(datos))
        datos["malo"] = (rng.random(len(datos)) < datos["pd_verdadera"]).astype(float)
        perm = rng.permutation(len(datos))
        n_dev = int(round(n_malos / datos["pd_verdadera"].mean()))
        d_dev = datos.iloc[perm[:n_dev]]
        d_pr = datos.iloc[perm[n_dev:n_dev + n_prueba]]
        y_dev = d_dev["malo"].to_numpy()
        cols = [woe_rapido(d_dev[v], y_dev, d_pr[v]) for v in _REALES + ruido]
        Xd = sm.add_constant(np.column_stack([c[0] for c in cols]), has_constant="add")
        Xp = sm.add_constant(np.column_stack([c[1] for c in cols]), has_constant="add")
        try:
            fit = sm.Logit(y_dev, Xd).fit(disp=0, maxiter=100)
        except Exception:                                     # separación en muestras minúsculas
            return np.nan, np.nan
        return gini(y_dev, fit.predict(Xd)), gini(d_pr["malo"].to_numpy(), fit.predict(Xp))

    _filas = []
    for _nm in (40, 100, 250, 600):
        for _ruido in (False, True):
            _res = np.array([_optimismo(_nm, _ruido, 1000 * _nm + _r) for _r in range(20)])
            _filas.append({"malos_DEV": _nm, "variables": "7 reales + 10 ruido" if _ruido else "7 reales",
                           "gini_DEV": np.nanmean(_res[:, 0]), "gini_prueba": np.nanmean(_res[:, 1]),
                           "optimismo": np.nanmean(_res[:, 0] - _res[:, 1]),
                           "se_optimismo": np.nanstd(_res[:, 0] - _res[:, 1], ddof=1) / np.sqrt(20)})
    tabla_optimismo = pd.DataFrame(_filas).set_index(["malos_DEV", "variables"])
    tabla_optimismo.round(3)
    return tabla_optimismo, woe_rapido


@app.cell
def _(
    MUESTRAS,
    PD_A,
    VARS,
    auc_esperado,
    auc_esperado_modelo,
    generar_cartera,
    gini,
    modelo_a,
    np,
    pd,
    pd_de,
):
    _filas = []
    for _det in (0.0, 0.35, 1.0):
        _c = generar_cartera(deterioro=_det)
        _o = _c[_c["muestra"] == "OOT"].reset_index(drop=True)
        assert np.allclose(_o[VARS].fillna(-1).to_numpy(), MUESTRAS["OOT"][VARS].fillna(-1).to_numpy())
        _p = pd_de(modelo_a, _o, VARS)
        _y = _o["malo"].to_numpy()
        _filas.append({"deterioro": _det, "tasa_malos_OOT": _y.mean(), "pd_media_modelo": _p.mean(),
                       "gini_modelo_OOT": gini(_y, _p),
                       "gini_pd_verdadera_OOT": gini(_y, _o["pd_verdadera"].to_numpy()),
                       "techo_esperado": 2 * auc_esperado(_o["pd_verdadera"].to_numpy()) - 1,
                       "modelo_esperado": 2 * auc_esperado_modelo(_p, _o["pd_verdadera"].to_numpy()) - 1})
    tabla_nivel = pd.DataFrame(_filas).set_index("deterioro")
    tabla_nivel["brecha"] = tabla_nivel["techo_esperado"] - tabla_nivel["modelo_esperado"]
    _filas = None
    assert np.allclose(pd_de(modelo_a, MUESTRAS["OOT"], VARS), PD_A["OOT"])
    tabla_nivel.round(4)
    return (tabla_nivel,)


@app.cell
def _(mo):
    sl_lambda = mo.ui.slider(0.0, 1.0, step=0.05, value=0.6,
                             label="λ: fracción del efecto de uso_linea que desaparece en OOT")
    sl_lambda
    return (sl_lambda,)


@app.cell
def _(MUESTRAS, PD_A, gini, np, sl_lambda):
    _o = MUESTRAS["OOT"]
    _eta = np.log(_o["pd_verdadera"] / (1 - _o["pd_verdadera"])).to_numpy()
    _eta_nuevo = _eta - sl_lambda.value * 2.4 * (_o["uso_linea_prom_12m"].to_numpy() - 0.35)
    _p_nuevo = 1 / (1 + np.exp(-_eta_nuevo))
    _rng = np.random.default_rng(99)
    _y_nuevo = (_rng.random(len(_o)) < _p_nuevo).astype(int)
    drift_concepto = {"lambda": sl_lambda.value, "tasa_malos": _y_nuevo.mean(),
                      "gini_modelo": gini(_y_nuevo, PD_A["OOT"]),
                      "gini_techo_nuevo": gini(_y_nuevo, _p_nuevo)}
    drift_concepto
    return (drift_concepto,)


@app.cell
def _(drift_concepto, mo, tabla_nivel, tabla_optimismo):
    _t = tabla_optimismo
    mo.md(f"""
    **Lectura.**

    - **Optimismo**: con 40 malos en DEV y 7 variables reales, el Gini in-sample excede al de prueba
      en {_t.loc[(40,'7 reales'),'optimismo']:.3f} en promedio; con 10 variables de ruido, en
      {_t.loc[(40,'7 reales + 10 ruido'),'optimismo']:.3f}. Con 600 malos baja a
      {_t.loc[(600,'7 reales'),'optimismo']:.3f} / {_t.loc[(600,'7 reales + 10 ruido'),'optimismo']:.3f}. Es un sesgo que se predice
      (y se corrige: bootstrap de Efron para el optimismo, validación cruzada) y **siempre** tiene signo
      positivo en expectativa. Por eso DEV→HO se lee como «¿cuánto memorizó?», no como deterioro.
    - **Nivel**: con deterioro 0 → 1,0 la tasa de malos OOT pasa de {tabla_nivel.loc[0.0,'tasa_malos_OOT']:.1%} a
      {tabla_nivel.loc[1.0,'tasa_malos_OOT']:.1%} y la PD media del modelo no se mueve ({tabla_nivel.loc[1.0,'pd_media_modelo']:.1%}).
      El orden de los clientes es idéntico, y aun así el Gini OOT va de {tabla_nivel.loc[0.0,'gini_modelo_OOT']:.3f} a
      {tabla_nivel.loc[1.0,'gini_modelo_OOT']:.3f}. No es el modelo: el **techo** esperado baja de
      {tabla_nivel.loc[0.0,'techo_esperado']:.3f} a {tabla_nivel.loc[1.0,'techo_esperado']:.3f} (con más PD alta, más buenos
      «parecen» malos), y la brecha modelo–techo queda en {tabla_nivel['brecha'].min():.3f}–{tabla_nivel['brecha'].max():.3f}.
      Moraleja: comparar Gini de periodos con tasas de malos muy distintas mezcla ranking con nivel.
      El nivel lo detectan el backtesting de calibración y el binomial (M19), no el Gini.
    - **Ranking**: con λ = {drift_concepto['lambda']:.2f} el Gini OOT del modelo cae a
      {drift_concepto['gini_modelo']:.3f} (el techo con la nueva verdad es {drift_concepto['gini_techo_nuevo']:.3f}). Esto sí es
      deterioro HO→OOT: la relación variable→riesgo cambió.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Deciles, lift, captura y una métrica de negocio

    Deciles de la muestra evaluada (decil 1 = peor score), igual que la tabla OOT de la clase 5.
    Implementación numpy (cuantiles + `searchsorted`, intervalos cerrados a la derecha) vs
    `pandas.qcut`. La planilla `M12_tabla_deciles_gini.xlsx` hace la misma cuenta con fórmulas.
    """)
    return


@app.cell
def _(SCORE_A, Y, np, pd):
    def deciles_numpy(score, y, k=10):
        """Tabla de deciles: decil 1 = score más bajo (peor). Cortes = cuantiles de la muestra."""
        score = np.asarray(score, float); y = np.asarray(y, float)
        cortes = np.unique(np.quantile(score, np.linspace(0, 1, k + 1)))
        d = np.searchsorted(cortes[1:-1], score, side="left")       # intervalos (a, b]
        n = np.bincount(d); malos = np.bincount(d, weights=y)
        buenos = n - malos
        t = pd.DataFrame({"n": n, "malos": malos.astype(int), "tasa": malos / n,
                          "score_min": [score[d == j].min() for j in range(len(n))],
                          "score_max": [score[d == j].max() for j in range(len(n))]})
        t.index = pd.Index(np.arange(1, len(n) + 1), name="decil")
        t["pob_acum"] = n.cumsum() / n.sum()
        t["captura_malos"] = malos.cumsum() / malos.sum()
        t["captura_buenos"] = buenos.cumsum() / buenos.sum()
        t["lift_decil"] = t["tasa"] / (malos.sum() / n.sum())
        t["lift_acum"] = t["captura_malos"] / t["pob_acum"]
        t["ks_decil"] = t["captura_malos"] - t["captura_buenos"]
        return t

    def gini_agrupado(t):
        """Gini por trapecios sobre la ROC de la tabla agrupada (empates dentro del decil = ½)."""
        x = np.r_[0, t["captura_buenos"].to_numpy()]
        y_ = np.r_[0, t["captura_malos"].to_numpy()]
        return 2 * np.sum(np.diff(x) * (y_[1:] + y_[:-1]) / 2) - 1

    tabla_deciles = deciles_numpy(SCORE_A["OOT"], Y["OOT"])
    _q = pd.qcut(SCORE_A["OOT"], 10, labels=False, duplicates="drop") + 1
    _n_pd = pd.Series(Y["OOT"]).groupby(_q).agg(["size", "sum"])
    assert np.array_equal(_n_pd["size"].to_numpy(), tabla_deciles["n"].to_numpy())
    assert np.array_equal(_n_pd["sum"].to_numpy(), tabla_deciles["malos"].to_numpy())
    tabla_deciles.round(3)
    return deciles_numpy, gini_agrupado, tabla_deciles


@app.cell
def _(PD_A, SCORE_A, Y, gini, gini_agrupado, ks_numpy, mo, tabla_deciles):
    GINI_DECILES_OOT = gini_agrupado(tabla_deciles)
    _g = gini(Y["OOT"], PD_A["OOT"])
    _ks = ks_numpy(Y["OOT"], PD_A["OOT"])[0]
    _t = tabla_deciles
    mo.md(f"""
    **Lectura (OOT).** El peor decil concentra {_t.loc[1,'captura_malos']:.1%} de los malos (lift
    {_t.loc[1,'lift_acum']:.2f}); al tercero, {_t.loc[3,'captura_malos']:.1%}. KS máximo por decil
    {_t['ks_decil'].max():.3f} ≤ KS exacto {_ks:.3f} (la grilla gruesa nunca ve el máximo exacto,
    salvo que caiga en un corte). Gini por trapecios sobre deciles {GINI_DECILES_OOT:.3f} vs exacto
    {_g:.3f}: agrupar en 10 pierde {_g - GINI_DECILES_OOT:.3f}. Los scores van de {SCORE_A['OOT'].min():.0f}
    a {SCORE_A['OOT'].max():.0f} puntos.
    """)
    return (GINI_DECILES_OOT,)


@app.cell
def _(mo):
    sl_aprob = mo.ui.slider(0.40, 0.98, step=0.01, value=0.80, label="Tasa de aprobación")
    sl_aprob
    return (sl_aprob,)


@app.cell
def _(PD_A, PD_B, Y, dd_quitar, mo, np, pd, plt, sl_aprob):
    def negocio(y, p, aprob):
        """Aprueba el `aprob` de menor PD. Devuelve tasa de malos aprobados y % malos rechazados."""
        corte = np.quantile(p, aprob)
        ok = p <= corte
        return y[ok].mean(), 1 - y[ok].sum() / y.sum(), ok.mean()

    _y = Y["OOT"]
    _filas = []
    for _nombre, _p in (("A (7 variables)", PD_A["OOT"]), (f"B (sin {dd_quitar.value})", PD_B["OOT"])):
        _tm, _cap, _ap = negocio(_y, _p, sl_aprob.value)
        _filas.append({"modelo": _nombre, "aprobación_real": _ap, "tasa_malos_aprobados": _tm,
                       "malos_rechazados": _cap})
    tabla_negocio = pd.DataFrame(_filas).set_index("modelo")
    _grid = np.linspace(0.3, 1.0, 71)
    _fig, _ax = plt.subplots(figsize=(7.5, 3.6))
    for _nombre, _p, _c in (("A", PD_A["OOT"], "#1f5fbf"), ("B", PD_B["OOT"], "#e0a100")):
        _ax.plot(_grid, [negocio(_y, _p, _a)[0] for _a in _grid], color=_c, label=f"modelo {_nombre}")
    _ax.axvline(sl_aprob.value, ls="--", color="grey", label="aprobación elegida")
    _ax.set_xlabel("tasa de aprobación"); _ax.set_ylabel("tasa de malos entre aprobados")
    _ax.set_title("Curva de estrategia (OOT)"); _ax.legend(fontsize=8); _fig.tight_layout()
    _out = mo.vstack([tabla_negocio.round(4), _fig])
    _out
    return negocio, tabla_negocio


@app.cell
def _(mo, sl_aprob, tabla_negocio):
    _a, _b = tabla_negocio.iloc[0], tabla_negocio.iloc[1]
    mo.md(f"""
    **Lectura.** Aprobando el {sl_aprob.value:.0%}, la tasa de malos entre aprobados es
    {_a['tasa_malos_aprobados']:.2%} con A y {_b['tasa_malos_aprobados']:.2%} con B
    ({1e4*(_b['tasa_malos_aprobados']-_a['tasa_malos_aprobados']):+.0f} pb). Eso —no el AUC— es lo que se
    traduce a pérdida esperada (M17). El AUC promedia sobre **todos** los cortes, incluidos los que
    nadie usa (aprobar 5% o 99%); dos modelos con igual AUC pueden diferir en la zona operativa.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. IV ↔ Gini en el mundo binormal (con cautela)

    Si la variable es normal en buenos y malos con igual varianza y medias separadas por $d$
    desviaciones, el WoE es lineal en $x$ y (se deriva en el `.md`) $\text{IV}=d^2$ y
    $\text{AUC}=\Phi(d/\sqrt2)$, luego $\text{Gini}=2\Phi\big(\sqrt{\text{IV}/2}\big)-1$. Con datos reales la
    relación es solo orientativa: el IV binneado subestima el continuo, y fuera del binormal la
    relación cambia.
    """)
    return


@app.cell
def _(mo):
    dd_bins_iv = mo.ui.dropdown(options={"5 bins (curso)": 5, "10 bins": 10, "20 bins": 20},
                                value="5 bins (curso)", label="Bins para el IV empírico")
    dd_bins_iv
    return (dd_bins_iv,)


@app.cell
def _(dd_bins_iv, gini, np, pd, stats, tabla_woe):
    _rng = np.random.default_rng(31)
    _filas = []
    for _d in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        _n, _pi = 20000, 0.10
        _y = (_rng.random(_n) < _pi).astype(float)
        _x = np.where(_y == 1, 0.0, _d) + _rng.standard_normal(_n)      # buenos con media más alta
        _iv_emp = tabla_woe(_x, _y, bins=dd_bins_iv.value)[1]
        _filas.append({"d": _d, "IV_teorico": _d ** 2, "IV_binneado": _iv_emp,
                       "gini_teorico": 2 * stats.norm.cdf(_d / np.sqrt(2)) - 1,
                       "gini_desde_IV_binneado": 2 * stats.norm.cdf(np.sqrt(_iv_emp / 2)) - 1,
                       "gini_empirico": gini(_y, -_x)})
    tabla_iv_gini = pd.DataFrame(_filas).set_index("d")
    tabla_iv_gini.round(4)
    return (tabla_iv_gini,)


@app.cell
def _(mo, tabla_iv_gini):
    mo.md(f"""
    **Lectura.** Con d = 1 (IV teórico 1,0), el Gini teórico es {tabla_iv_gini.loc[1.0,'gini_teorico']:.3f} y el
    empírico {tabla_iv_gini.loc[1.0,'gini_empirico']:.3f}. El IV binneado ({tabla_iv_gini.loc[1.0,'IV_binneado']:.3f}) es menor que el
    teórico: binear pierde información, y la pérdida crece con la separación (con d = 2, IV 4,0 vs
    {tabla_iv_gini.loc[2.0,'IV_binneado']:.2f}). Regla de bolsillo: IV 0,10 ≈ Gini 0,18; IV 0,30 ≈ Gini 0,30;
    IV 0,50 ≈ Gini 0,38 — **una variable**, en el mundo binormal. El Gini de un modelo no es la suma de
    los Gini de sus variables.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Invariancia monótona: el δ no toca el Gini (y cuándo sí)

    El AUC depende solo del **orden**. Cualquier transformación estrictamente creciente del score
    (calibrar con δ en el logit, pasar a puntos PDO, exponenciar, rankear) deja el AUC idéntico. Lo que
    **no** es estrictamente creciente —redondear, bandear, truncar (`clip`)— crea empates y lo cambia.
    """)
    return


@app.cell
def _(PD_A, Y, gini, np, pd, score_de_pd, stats):
    _y, _p = Y["OOT"], PD_A["OOT"]
    _logit = np.log(_p / (1 - _p))
    _transf = {
        "PD cruda": _p,
        "PD calibrada δ = +0,177 (PIT curso)": 1 / (1 + np.exp(-(_logit + 0.177))),
        "PD calibrada δ = −0,5": 1 / (1 + np.exp(-(_logit - 0.5))),
        "−score PDO": -score_de_pd(_p),
        "exp(−score/10)": np.exp(-score_de_pd(_p) / 10),
        "rango medio (rankdata)": stats.rankdata(_p),
        "rango con desempate arbitrario (argsort)": np.argsort(np.argsort(_p, kind="mergesort"), kind="mergesort").astype(float),
        "−score redondeado a entero": -np.round(score_de_pd(_p)),
        "PD truncada en [2%, 30%]": np.clip(_p, 0.02, 0.30),
    }
    tabla_invariancia = pd.DataFrame({"gini": {k: gini(_y, v) for k, v in _transf.items()}})
    tabla_invariancia["dif_vs_cruda"] = tabla_invariancia["gini"] - tabla_invariancia.loc["PD cruda", "gini"]
    tabla_invariancia.round(6)
    return (tabla_invariancia,)


@app.cell
def _(mo, tabla_invariancia):
    _t = tabla_invariancia
    mo.md(f"""
    **Lectura.** Las transformaciones estrictamente crecientes (δ, puntos PDO, exponencial, rango
    medio) dan el mismo Gini hasta el último decimal. El rango con desempate arbitrario (`argsort`)
    **no**: difiere en {_t.loc['rango con desempate arbitrario (argsort)','dif_vs_cruda']:+.6f}, porque la PD de un
    scorecard tiene empates masivos y romperlos en el orden de la tabla convierte pares ½ en 0 o 1.
    Redondear a entero cambia {_t.loc['−score redondeado a entero','dif_vs_cruda']:+.5f}
    y truncar en [2%, 30%] cambia {_t.loc['PD truncada en [2%, 30%]','dif_vs_cruda']:+.4f}: los extremos truncados
    se empatan. Por eso el Gini de la clase 5 es «el de la clase 3, punto por punto»: el δ de la
    clase 4 es monótono.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ✔ Checks del módulo

    Si alguno falla, el notebook falla. Coincidencias numpy vs librería e invariantes teóricas.
    """)
    return


@app.cell
def _(
    GINI_DECILES_OOT,
    MAPAS,
    PD_A,
    PD_V,
    VARS,
    Y,
    a_woe,
    auc_conteo,
    auc_esperado,
    auc_esperado_conteo,
    auc_esperado_modelo,
    auc_rangos,
    bootstrap_auc,
    deciles_numpy,
    delong_definicion,
    delong_pareado,
    delong_rapido,
    dev,
    gini,
    ks_numpy,
    negocio,
    np,
    rangos_medios,
    roc_auc_score,
    stats,
    tabla_ar,
    tabla_auc4,
    tabla_caida,
    tabla_deciles,
    tabla_iv_gini,
    tabla_ks,
    tabla_nivel,
    tabla_optimismo,
    tabla_techo,
    tabla_var,
    woe_rapido,
):
    _rng = np.random.default_rng(0)
    # 1. rangos medios = scipy.rankdata
    _x = _rng.integers(0, 20, 500).astype(float)
    assert np.allclose(rangos_medios(_x), stats.rankdata(_x))
    # 2. AUC: cuatro implementaciones (ya verificadas) y AUC(1-y, -s) = AUC(y, s)
    assert np.allclose(tabla_auc4.iloc[:, :4].to_numpy(), tabla_auc4[["sklearn"]].to_numpy(), atol=1e-12)
    assert np.isclose(auc_rangos(1 - Y["OOT"], -PD_A["OOT"]), auc_rangos(Y["OOT"], PD_A["OOT"]))
    # 3. AUC con empates: 0 ≤ AUC(empate 0) ≤ AUC(½) ≤ AUC(1) y ½ = promedio
    _s = np.round(PD_A["OOT"], 2)
    _a0, _ah, _a1 = (auc_conteo(Y["OOT"], _s, w) for w in (0.0, 0.5, 1.0))
    assert _a0 <= _ah <= _a1 and np.isclose(_ah, (_a0 + _a1) / 2)
    assert np.isclose(_ah, roc_auc_score(Y["OOT"], _s))
    # 4. Gini = AR = Somers
    assert np.allclose(tabla_ar.iloc[:, 0], tabla_ar.iloc[:, 1]) and np.allclose(tabla_ar.iloc[:, 0], tabla_ar.iloc[:, 2])
    # 5. KS: tres implementaciones y cotas KS ≤ Gini ≤ KS(2−KS)
    assert np.allclose(tabla_ks["KS_numpy"], tabla_ks["KS_scipy"]) and np.allclose(tabla_ks["KS_numpy"], tabla_ks["KS_roc"])
    assert (tabla_ks["cota_inf_KS"] <= tabla_ks["gini"] + 1e-12).all()
    assert (tabla_ks["gini"] <= tabla_ks["cota_sup_KS(2-KS)"] + 1e-12).all()
    # 6. Techo: O(n log n) = O(n²); el techo supera al modelo evaluado contra la verdad
    _p = PD_V["OOT"][:800]
    assert np.isclose(auc_esperado(_p), auc_esperado_conteo(_p))
    assert np.isclose(auc_esperado_modelo(_p, _p), auc_esperado(_p))
    assert (tabla_techo["gini_techo_esperado"] > tabla_techo["gini_modelo_esperado"]).all()
    # 7. DeLong definición = rápido; DeLong ≈ bootstrap (±20%)
    _a, _v = delong_definicion(Y["HO"], PD_A["HO"])
    _ar, _c = delong_rapido(Y["HO"], PD_A["HO"])
    assert np.isclose(_a, _ar[0]) and np.isclose(_v, _c[0, 0], rtol=1e-10)
    _r = tabla_var["SE_DeLong_rapido"] / tabla_var["SE_boot_numpy"]
    assert ((_r > 0.8) & (_r < 1.25)).all(), _r
    _r2 = tabla_var["SE_boot_scipy"] / tabla_var["SE_boot_numpy"]
    assert ((_r2 > 0.8) & (_r2 < 1.25)).all(), _r2
    # 8. DeLong pareado de un modelo consigo mismo: diferencia 0; covarianza simétrica
    _a2, _c2 = delong_rapido(Y["OOT"], np.vstack([PD_A["OOT"], np.exp(PD_A["OOT"])]))
    assert np.isclose(_a2[0], _a2[1]) and np.allclose(_c2, _c2[0, 0])
    _dp = delong_pareado(Y["OOT"], PD_A["OOT"], PD_V["OOT"])
    assert _dp["dif_gini"] < 0 and 0 < _dp["correlacion"] < 1
    # 9. Caída por azar: la SD decrece con los malos (≈ 1/√n)
    _sd = tabla_caida["sd_caida"].to_numpy()
    assert np.all(np.diff(_sd) < 0)
    _pend = np.polyfit(np.log(tabla_caida.index.to_numpy()), np.log(_sd), 1)[0]
    assert -0.65 < _pend < -0.35, _pend
    # 10. Optimismo positivo en promedio y mayor con ruido cuando hay pocos malos
    assert (tabla_optimismo["optimismo"] > 0).all()
    assert tabla_optimismo.loc[(40, "7 reales + 10 ruido"), "optimismo"] > tabla_optimismo.loc[(40, "7 reales"), "optimismo"]
    # 11. Deterioro de nivel: tasa sube, el techo baja y la brecha modelo-techo casi no cambia
    assert tabla_nivel["tasa_malos_OOT"].is_monotonic_increasing
    assert tabla_nivel["techo_esperado"].is_monotonic_decreasing
    assert np.ptp(tabla_nivel["brecha"]) < 0.01
    # 12. Deciles: suma de n y malos; KS por decil ≤ KS exacto; Gini agrupado ≤ exacto (aquí)
    assert tabla_deciles["n"].sum() == len(Y["OOT"]) and tabla_deciles["malos"].sum() == Y["OOT"].sum()
    assert tabla_deciles["ks_decil"].max() <= tabla_ks.loc["OOT", "KS_numpy"] + 1e-12
    assert GINI_DECILES_OOT <= gini(Y["OOT"], PD_A["OOT"]) + 1e-12
    # 13. Binormal: Gini teórico ≈ empírico (n = 20.000) y IV binneado < teórico (d ≥ 0,75)
    assert np.allclose(tabla_iv_gini["gini_teorico"], tabla_iv_gini["gini_empirico"], atol=0.03)
    assert (tabla_iv_gini.loc[0.75:, "IV_binneado"] < tabla_iv_gini.loc[0.75:, "IV_teorico"]).all()
    # 14. Hanley-McNeil a mano (AUC OOT del curso 0,8375 con 119 malos y 1.885 buenos)
    _q1, _q2 = 0.8375 / 1.1625, 2 * 0.8375 ** 2 / 1.8375
    _se = np.sqrt((0.8375 * 0.1625 + 118 * (_q1 - 0.8375 ** 2) + 1884 * (_q2 - 0.8375 ** 2)) / (119 * 1885))
    assert 0.020 < _se < 0.026
    # 15. woe_rapido (numpy) reproduce a_woe/tabla_woe del curso en DEV, variable por variable
    _F = a_woe(dev, VARS, dev, MAPAS)
    for _v in VARS:
        assert np.allclose(woe_rapido(dev[_v], dev["malo"].to_numpy(), dev[_v])[0], _F[_v]), _v
    _ = (bootstrap_auc, deciles_numpy, ks_numpy, negocio)
    mo_checks = "✔ Todos los checks del módulo pasaron"
    mo_checks
    return


if __name__ == "__main__":
    app.run()

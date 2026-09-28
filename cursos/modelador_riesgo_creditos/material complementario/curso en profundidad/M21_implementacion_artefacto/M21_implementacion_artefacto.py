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
#     "jsonschema>=4.18",
#     "pydantic>=2",
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
    import json
    import hashlib
    import copy
    import math
    from typing import Optional
    from scipy.special import expit, logit
    from scipy.optimize import brentq
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    import jsonschema
    from pydantic import ValidationError, create_model, field_validator
    return (ValidationError, Optional, brentq, copy, create_model, expit, field_validator, hashlib, json,
            jsonschema, logit, math, mo, plt, roc_auc_score, sm, stats)


@app.cell
def _(mo):
    mo.md(r"""
    # M21 · Implementación: artefacto congelado, contrato de datos y paridad

    Serie 2 «Del embudo al gobierno» · profundiza la clase 6 (láminas 4–6), el notebook `demo_c6_bases`
    (secciones 2–7) y el Lab 3 de Financiera Andes (tareas 7–9).

    La regla de la clase: **producción no CALCULA, producción APLICA**. Este notebook la convierte en código
    verificable sobre el generador de la serie (verdad conocida):

    1. Pipeline de desarrollo (el «notebook»): binning del curso, WoE, logística, δ, escala PDO 20 / 600 @ 50:1.
    2. **Congelar**: el artefacto JSON (bins con bordes explícitos, WoE, puntos, β, δ, escala, master scale,
       cutoff, reason codes, contrato), su hash canónico y su **JSON Schema** + validación semántica.
    3. **Motor `puntuar`**: función pura, vectorizada, numpy desde cero (`searchsorted`) **y** versión
       `pd.cut`; paridad contra el notebook en DEV/HO/OOT/TTD.
    4. **Semántica de bordes**: $(a,b]$ vs $[a,b)$, cortes redondeados, cortes en `float32`.
    5. **Bug A (silencioso) vs bug B (ruidoso)** con slider de *drift* del lote.
    6. **Tests de propiedades**: subconjunto (el que mata al bug A), permutación, idempotencia, monotonía.
    7. **Golden files y versionado semántico** derivado de la prueba; *shadow mode* y *rollback*.
    8. **Contrato de datos**: numpy/pandas puro **y** pydantic; severidad por magnitud; los 3 desastres que
       no lanzan excepción.
    9. Checks del módulo.

    Convenciones del curso: target 1 = malo; WoE = ln(%buenos/%malos) ⇒ β negativos; todo se ajusta en DEV
    y se **aplica** al resto.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Código común de la serie
    Pegado verbatim desde `_spec/comun.py` (generador «Banco Sintético» con verdad conocida y las herramientas
    `binear / tabla_woe / a_woe` del curso).
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
    def fmt_num(x, d=2):
        """Número con coma decimal y punto de miles (prosa en español)."""
        s = f"{x:,.{d}f}"
        return s.replace(",", "·").replace(".", ",").replace("·", ".")

    def fmt_pct(x, d=1):
        return fmt_num(100 * x, d) + " %"
    return fmt_num, fmt_pct


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. El pipeline de desarrollo (lo que vive en el notebook)

    Replicamos la receta del curso sobre el generador: 8 variables, binning del curso con `ref=dev`,
    WoE de DEV, logística (statsmodels), calibración PIT por desplazamiento del intercepto
    $\text{PD}_{cal}=\text{logistic}(\eta+\delta)$ con $\delta$ que iguala la PD media de DEV a la tendencia
    central, y escala PDO 20 / 600 @ 50:1.

    Aviso honesto: `canal` (IV ≈ 0,01) y `renta_mm` (IV ≈ 0,01) **no pasarían** el filtro IV ≥ 0,10 del curso.
    Están aquí a propósito: son el camino categórico y el camino *missing* del motor. Este scorecard no es un
    buen modelo; es un buen banco de pruebas de implementación.

    Todo lo de esta sección es **desarrollo**: usa DEV. La pregunta del módulo es cómo llevarlo a un lugar
    donde DEV no existe.
    """)
    return


@app.cell
def _(generar_cartera):
    cartera = generar_cartera()
    # el índice ES la identidad del caso: id de solicitud, no 0..n-1
    MUESTRAS = {_m: cartera[cartera["muestra"] == _m].set_index("id", drop=False)
                for _m in ("DEV", "HO", "OOT", "TTD")}
    dev, ho, oot, ttd = (MUESTRAS[_m] for _m in ("DEV", "HO", "OOT", "TTD"))
    return MUESTRAS, cartera, dev, ho, oot, ttd


@app.cell
def _(a_woe, brentq, cartera, dev, expit, logit, np, pd, sm, tabla_woe):
    VARIABLES = ["uso_linea_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m", "antiguedad_meses",
                 "carga_financiera", "consultas_6m", "canal", "renta_mm"]

    def desarrollar(variables, base):
        """Desarrollo: tablas WoE y logística, todo ajustado sobre `base` (DEV)."""
        _tablas = {_v: tabla_woe(base[_v], base["malo"])[0] for _v in variables}
        _mapas = {_v: _tablas[_v]["woe"] for _v in variables}
        _X = sm.add_constant(a_woe(base, variables, base, _mapas), has_constant="add")
        _mod = sm.Logit(base["malo"].to_numpy(), _X).fit(disp=0, maxiter=200)
        return _tablas, _mapas, _mod

    def calibrar_delta(lp, tasa):
        """δ tal que mean(logistic(lp + δ)) = tasa (brentq, tolerancia 1e-12)."""
        return float(brentq(lambda _d: expit(lp + _d).mean() - tasa, -5, 5, xtol=1e-12))

    tablas_dev, mapas_dev, modelo_sm = desarrollar(VARIABLES, dev)
    beta_dev = modelo_sm.params.copy()

    # tendencia central: promedio simple de tasas mensuales con desempeño (como la clase 4/5)
    TC = float(cartera[cartera["malo"].notna()].groupby("cohorte")["malo"].mean().mean())
    _lp_dev = logit(np.clip(modelo_sm.predict(sm.add_constant(a_woe(dev, VARIABLES, dev, mapas_dev),
                                                                has_constant="add")), 1e-12, 1 - 1e-12))
    delta_dev = calibrar_delta(np.asarray(_lp_dev), TC)

    PDO, SCORE_BASE, ODDS_BASE = 20, 600, 50
    FACTOR = PDO / np.log(2)
    OFFSET = SCORE_BASE - FACTOR * np.log(ODDS_BASE)
    CORTES_MS = [490, 510, 530, 550, 570, 590, 610]          # master scale adaptada al generador
    ETIQUETAS_MS = ["E", "D", "C2", "C1", "B2", "B1", "A2", "A1"]   # peor → mejor, [a, b)
    CUTOFF = 530

    def pd_notebook(df):
        """PD cruda del notebook: binear(ref=dev) + WoE de DEV + statsmodels.predict."""
        _X = sm.add_constant(a_woe(df, VARIABLES, dev, mapas_dev), has_constant="add")
        return np.asarray(modelo_sm.predict(_X))

    def score_notebook(df):
        """(score, PD calibrada) como lo hace el notebook del curso (clip bilateral 1e-12)."""
        _p = np.clip(pd_notebook(df), 1e-12, 1 - 1e-12)
        _pc = expit(logit(_p) + delta_dev)
        _s = OFFSET - FACTOR * logit(np.clip(_pc, 1e-12, 1 - 1e-12))
        return _s, _pc

    resumen_modelo = pd.DataFrame({
        "beta": beta_dev.drop("const").round(4),
        "IV_DEV": [round(float(tablas_dev[_v]["iv_aporte"].sum()), 3) for _v in VARIABLES],
        "bins": [len(tablas_dev[_v]) for _v in VARIABLES],
    })
    assert (beta_dev.drop("const") < 0).all(), "convención WoE: β negativos"
    resumen_modelo
    return (CORTES_MS, CUTOFF, ETIQUETAS_MS, FACTOR, ODDS_BASE, OFFSET, PDO, SCORE_BASE, TC, VARIABLES, beta_dev,
            calibrar_delta, delta_dev, desarrollar, mapas_dev, modelo_sm, pd_notebook, resumen_modelo,
            score_notebook, tablas_dev)


@app.cell
def _(FACTOR, OFFSET, TC, beta_dev, delta_dev, dev, fmt_num, fmt_pct, mo, roc_auc_score, score_notebook, ttd,
      CUTOFF):
    _s_dev, _pc_dev = score_notebook(dev)
    _s_ttd, _ = score_notebook(ttd)
    _gini = 2 * roc_auc_score(dev["malo"], _pc_dev) - 1
    mo.md(f"""
    **Lectura.** DEV = {fmt_num(len(dev), 0)} créditos ({fmt_pct(dev['malo'].mean())} malos); intercepto
    β₀ = {fmt_num(beta_dev['const'], 4)}; Gini DEV {fmt_num(_gini, 3)}. Tendencia central
    TC = {fmt_pct(TC, 2)} ⇒ δ = {fmt_num(delta_dev, 4)} (en puntos: −δ·factor = {fmt_num(-delta_dev * FACTOR, 2)}).
    Escala: factor {fmt_num(FACTOR, 4)}, offset {fmt_num(OFFSET, 4)} (los del curso). Cutoff {CUTOFF}
    (aprobación TTD {fmt_pct((_s_ttd >= CUTOFF).mean())}). Todos los β son negativos, como exige la convención
    de WoE del curso.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Congelar: el artefacto

    `binear(x, ref=dev[v])` recalcula los cortes **mirando DEV** en cada llamada. En el servidor de scoring DEV
    no existe. Congelar = convertir la partición en **datos**. El artefacto guarda por variable una lista de
    bins explícitos:

    - `intervalo` con `inf`/`sup` (`null` = no acotado; así se representan $-\infty$ y $+\infty$ en JSON
      estándar, que **no** admite `Infinity`) y `cierre: "derecha"` ⇒ $(\text{inf}, \text{sup}]$, la
      semántica de `pd.cut` por defecto;
    - `valor` (la moda aparte del binning del curso, p. ej. `meses_desde_mora_12m = 13`);
    - `categoria` y `missing`;
    - `no_visto`: qué hacer con lo que DEV nunca vio (WoE 0 **y** marca para revisión).

    Más: β por variable e intercepto, escala (PDO, ancla, factor, offset), calibración (δ y TC), master scale
    con su convención de cierre, cutoff, mapa de reason codes, contrato de datos (§8), procedencia (hash de
    DEV) e integridad (hash SHA-256 de la serialización canónica).
    """)
    return


@app.cell
def _(CORTES_MS, CUTOFF, ETIQUETAS_MS, FACTOR, ODDS_BASE, OFFSET, PDO, SCORE_BASE, copy, hashlib, json, np, pd):
    REASON_TEXTOS = {
        "uso_linea_prom_12m": ("RC01", "Uso alto de la línea de crédito"),
        "uso_tc_prom_3m": ("RC02", "Uso alto de la tarjeta en los últimos 3 meses"),
        "meses_desde_mora_12m": ("RC03", "Mora propia reciente o sin información de bureau"),
        "antiguedad_meses": ("RC04", "Poca antigüedad como cliente"),
        "carga_financiera": ("RC05", "Carga financiera alta respecto de la renta"),
        "consultas_6m": ("RC06", "Muchas consultas de crédito recientes"),
        "canal": ("RC07", "Canal de originación"),
        "renta_mm": ("RC08", "Renta baja o no informada"),
        "deuda_otras_prom_12m": ("RC09", "Deuda alta en otras instituciones"),
    }
    # tendencia de negocio DECLARADA (no estimada): lo que el comité espera ver
    TENDENCIAS = {"uso_linea_prom_12m": "decreciente", "uso_tc_prom_3m": "decreciente",
                  "meses_desde_mora_12m": None, "antiguedad_meses": "creciente",
                  "carga_financiera": "decreciente", "consultas_6m": "decreciente",
                  "canal": None, "renta_mm": "creciente", "deuda_otras_prom_12m": "decreciente"}
    # dominio de negocio (valores POSIBLES, no observados): lo fija el dueño del dato
    DOMINIOS = {"uso_linea_prom_12m": {"min": 0.0, "max": 1.5},
                "uso_tc_prom_3m": {"min": 0.0, "max": 1.5},
                "meses_desde_mora_12m": {"min": 1.0, "max": 13.0, "especiales": [-99.0, -9.0], "entero": True},
                "antiguedad_meses": {"min": 0.0, "max": 600.0, "entero": True},
                "carga_financiera": {"min": 0.0, "max": 500.0},
                "consultas_6m": {"min": 0.0, "max": 100.0, "entero": True},
                "renta_mm": {"min": 0.01, "max": 100.0},
                "deuda_otras_prom_12m": {"min": 0.0, "max": 10000.0},
                "edad": {"min": 18.0, "max": 100.0, "entero": True}}

    def ajustar_cortes(base, bins=5, umbral_moda=0.35):
        """MISMA partición que `binear(x, ref=base)`, pero devuelta como datos."""
        _b = pd.Series(base).reset_index(drop=True)
        if not pd.api.types.is_numeric_dtype(_b):
            return {"tipo": "categorico"}
        if _b.nunique(dropna=True) <= bins:
            return {"tipo": "discreto"}
        _moda = _b.mode().iloc[0]
        _frac = float((_b == _moda).mean())
        _resto = _b[_b != _moda] if _frac > umbral_moda else _b
        _c = np.unique(np.nanquantile(_resto.dropna(), np.linspace(0, 1, bins + 1)))
        if len(_c) < 3:
            return {"tipo": "discreto"}
        return {"tipo": "numerico", "cortes": [float(_x) for _x in _c[1:-1]],
                "moda_aparte": _frac > umbral_moda, "moda": float(_moda)}

    def congelar_variable(nombre, base, tabla, beta, b0, n_var):
        """Spec congelada de UNA variable: bins explícitos con WoE y puntos (fórmula M13)."""
        _spec = ajustar_cortes(base)
        _woe, _n = tabla["woe"], tabla["n"]

        def _pts(w):
            return float(-(beta * w + b0 / n_var) * FACTOR + OFFSET / n_var)

        def _bin(k, tipo, etiqueta, **extra):
            _w = float(_woe.get(etiqueta, 0.0))
            return {"id": k, "tipo": tipo, **extra, "woe": _w, "puntos": _pts(_w),
                    "n_dev": int(_n.get(etiqueta, 0)), "etiqueta_dev": etiqueta}

        _bins = []
        if _spec["tipo"] == "numerico":
            _c = _spec["cortes"]
            _bordes = [None] + _c + [None]
            _et = [str(_i) for _i in pd.cut(pd.Series([np.nan]),
                                             np.array([-np.inf, *_c, np.inf])).cat.categories]
            for _k in range(len(_c) + 1):
                _bins.append(_bin(_k, "intervalo", _et[_k], inf=_bordes[_k], sup=_bordes[_k + 1]))
            if _spec["moda_aparte"]:
                _bins.append(_bin(len(_bins), "valor", f"= {_spec['moda']:.4g}", valor=_spec["moda"]))
        elif _spec["tipo"] == "categorico":
            for _cat in [str(_i) for _i in tabla.index if _i != "MISSING"]:
                _bins.append(_bin(len(_bins), "categoria", _cat, valor=_cat))
        else:
            raise NotImplementedError("discreto: fuera del alcance de este notebook (ver clave_discreta del curso)")
        if "MISSING" in tabla.index:
            _bins.append(_bin(len(_bins), "missing", "MISSING"))
        _codigo, _ = REASON_TEXTOS[nombre]
        _var = {"nombre": nombre, "tipo": "numerico" if _spec["tipo"] == "numerico" else "categorico",
                "beta": float(beta), "bins": _bins,
                "no_visto": {"woe": 0.0, "accion": "revisar"},
                "reason_code": _codigo, "tendencia_declarada": TENDENCIAS[nombre]}
        if _var["tipo"] == "numerico":
            _var["cierre"] = "derecha"
        return _var

    def derivar_contrato(base, variables):
        """Contrato de datos (§8): dominio de negocio + rango observado en DEV + tope de missing (curso)."""
        _cv = {}
        for _v in variables:
            _s = base[_v]
            _pm = float(_s.isna().mean())
            _c = {"pct_missing_dev": _pm, "pct_missing_max": min(1.0, 3.0 * _pm + 0.01)}
            if pd.api.types.is_numeric_dtype(_s):
                _c.update({"tipo": "numerico", "dominio": DOMINIOS[_v],
                           "rango_dev": {"min": float(np.nanmin(_s)), "max": float(np.nanmax(_s)),
                                         "p005": float(np.nanquantile(_s, 0.005)),
                                         "p995": float(np.nanquantile(_s, 0.995))},
                           "fuera_p005_p995_dev": float(((_s < np.nanquantile(_s, 0.005))
                                                         | (_s > np.nanquantile(_s, 0.995)))[_s.notna()].mean()),
                           "n_dev_observados": int(_s.notna().sum())})
            else:
                _c.update({"tipo": "categorico", "categorias": sorted(_s.dropna().astype(str).unique().tolist())})
            _cv[_v] = _c
        return {"variables": _cv,
                "reglas": [{"id": "antiguedad_vs_edad", "expresion": "antiguedad_meses <= 12 * edad",
                            "columnas": ["antiguedad_meses", "edad"]}],
                "umbrales": {"fuera_rango_aviso": 0.01, "fuera_rango_bloqueo": 0.10,
                             "missing_bloqueo_abs": 0.25, "missing_factor_bloqueo": 4.0,
                             "categoria_nueva_bloqueo": 0.10, "fila_invalida_bloqueo": 0.01}}

    def canonico(obj):
        """Serialización canónica: claves ordenadas, sin espacios, UTF-8, sin NaN/Infinity."""
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    def hash_contenido(art):
        _sin = {_k: _v for _k, _v in art.items() if _k != "integridad"}
        return hashlib.sha256(canonico(_sin).encode("utf-8")).hexdigest()

    def sellar(art):
        _a = copy.deepcopy(art)
        _a["integridad"] = {"algoritmo": "sha256",
                            "canonicalizacion": "json sort_keys, separators=(',',':'), UTF-8, allow_nan=False",
                            "hash": hash_contenido(_a)}
        return _a

    def construir_artefacto(variables, base, tablas, beta, delta, tc, version, descripcion,
                            cortes_ms=CORTES_MS, cutoff=CUTOFF, fecha="2026-09-28"):
        _n = len(variables)
        _b0 = float(beta["const"])
        _hdev = __import__("hashlib").sha256(
            pd.util.hash_pandas_object(base[variables], index=True).values.tobytes()).hexdigest()
        _art = {
            "formato": {"nombre": "scorecard-artefacto", "version_esquema": "1.0.0"},
            "modelo": {"id": "sintetico-consumo-scorecard", "version": version, "fecha_construccion": fecha,
                       "descripcion": descripcion, "target": "1 = malo (90+ DPD a 12 meses)"},
            "variables": [congelar_variable(_v, base[_v], tablas[_v], float(beta[_v]), _b0, _n)
                          for _v in variables],
            "intercepto": _b0,
            "escalado": {"pdo": PDO, "score_base": SCORE_BASE, "odds_base": ODDS_BASE,
                         "factor": float(FACTOR), "offset": float(OFFSET), "sentido": "mayor_es_mejor"},
            "calibracion": {"metodo": "desplazamiento_logit", "delta": float(delta),
                            "tendencia_central": float(tc),
                            "fuente": "media de PD en DEV = TC (brentq, xtol 1e-12)"},
            "master_scale": {"cortes": [float(_c) for _c in cortes_ms], "etiquetas": list(ETIQUETAS_MS),
                             "cierre": "izquierda"},
            "politica": {"cutoff": float(cutoff), "regla": "aprobar si score >= cutoff y sin marcas",
                         "no_visto": "revisar"},
            "reason_codes": {REASON_TEXTOS[_v][0]: REASON_TEXTOS[_v][1] for _v in variables},
            "contrato_datos": derivar_contrato(base, variables),
            "procedencia": {"muestra_ajuste": "DEV (cohortes <= 2024-12, 70% al azar)", "n_dev": int(len(base)),
                            "hash_dev": _hdev, "semilla_generador": 2026},
        }
        return sellar(_art)
    return (DOMINIOS, REASON_TEXTOS, TENDENCIAS, ajustar_cortes, canonico, congelar_variable,
            construir_artefacto, derivar_contrato, hash_contenido, sellar)


@app.cell
def _(TC, VARIABLES, beta_dev, construir_artefacto, delta_dev, dev, tablas_dev):
    art_v100 = construir_artefacto(VARIABLES, dev, tablas_dev, beta_dev, delta_dev, TC, "1.0.0",
                                   "Scorecard de consumo, 8 variables (banco de pruebas M21)")
    return (art_v100,)


@app.cell
def _(art_v100, binear, canonico, dev, fmt_num, json, mo, np, pd):
    # prueba de congelamiento: la partición congelada es IDÉNTICA a binear(ref=dev) (bin a bin)
    def _particion_congelada(x, var):
        _x = pd.Series(x).to_numpy()
        _out = np.full(len(_x), "", dtype=object)
        for _b in var["bins"]:
            if _b["tipo"] == "intervalo":
                _lo = -np.inf if _b["inf"] is None else _b["inf"]
                _hi = np.inf if _b["sup"] is None else _b["sup"]
                with np.errstate(invalid="ignore"):
                    _m = (_x > _lo) & (_x <= _hi)
                _out[_m] = _b["etiqueta_dev"]
        for _b in var["bins"]:
            if _b["tipo"] == "valor":
                _out[_x == _b["valor"]] = _b["etiqueta_dev"]
            if _b["tipo"] == "categoria":
                _out[_x == _b["valor"]] = _b["etiqueta_dev"]
        _na = pd.isna(_x)
        _out[_na] = "MISSING"
        return _out

    comparaciones_congelado = {}
    for _var in art_v100["variables"]:
        _v = _var["nombre"]
        _cong = _particion_congelada(dev[_v], _var)
        _desa = binear(dev[_v], ref=dev[_v])[0].to_numpy()
        comparaciones_congelado[_v] = bool((_cong == _desa).all())

    _txt = canonico(art_v100)
    _ejemplo = json.dumps(art_v100["variables"][2], indent=1, ensure_ascii=False)
    mo.md(f"""
    **Congelar no cambió nada:** partición idéntica a `binear(ref=dev)` en
    {sum(comparaciones_congelado.values())} de {len(comparaciones_congelado)} variables (bin a bin, DEV).

    Artefacto v{art_v100['modelo']['version']}: **{fmt_num(len(_txt), 0)} caracteres** de JSON canónico ·
    {len(art_v100['variables'])} variables · {sum(len(_x['bins']) for _x in art_v100['variables'])} bins ·
    hash `{art_v100['integridad']['hash'][:16]}…` (el curso: ~5.463 caracteres, 41 bins, hash `98845d69…`).

    Una variable completa (`meses_desde_mora_12m`: nótese la moda 13 aparte y que **−99 y −9 caen en el
    intervalo `(-inf, -9]`** — el binning del curso no separa especiales; ver M14):

    ```json
    {_ejemplo}
    ```
    """)
    return (comparaciones_congelado,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Por qué JSON estándar (y por qué `allow_nan=False` no es estética)

    RFC 8259 no admite `NaN` ni `Infinity`. Python **sí** los escribe por defecto (`json.dumps(float('nan'))`
    → `NaN`) y **sí** los lee por defecto (`json.loads('NaN')` funciona). Resultado: un artefacto que Python
    escribe y relee sin quejarse y que el motor Java/Go/SQL del banco rechaza — o peor, interpreta distinto.
    Dos candados: `allow_nan=False` al escribir y `parse_constant` que falla al leer. Y el hash: sin
    canonicalización (orden de claves, separadores), el mismo contenido da hashes distintos.
    """)
    return


@app.cell
def _(art_v100, canonico, copy, hashlib, json, math, pd):
    def cargar_estricto(texto):
        """json.loads que RECHAZA NaN/Infinity (Python los acepta por defecto)."""
        def _rechazar(c):
            raise ValueError(f"constante no estándar en el artefacto: {c}")
        return json.loads(texto, parse_constant=_rechazar)

    _filas = []
    # 1) escribir NaN
    try:
        json.dumps({"woe": float("nan")}, allow_nan=False)
        _r = "aceptado"
    except ValueError as _e:
        _r = f"rechazado ({type(_e).__name__})"
    _filas.append({"prueba": "json.dumps({'woe': nan}) por defecto",
                   "resultado": json.dumps({"woe": float("nan")})})
    _filas.append({"prueba": "json.dumps(..., allow_nan=False)", "resultado": _r})
    # 2) leer NaN
    _leido = json.loads('{"sup": Infinity, "woe": NaN}')
    _filas.append({"prueba": "json.loads('{\"sup\": Infinity, \"woe\": NaN}') por defecto",
                   "resultado": f"aceptado: sup={_leido['sup']}, woe es NaN={math.isnan(_leido['woe'])}"})
    try:
        cargar_estricto('{"sup": Infinity}')
        _r2 = "aceptado"
    except ValueError as _e:
        _r2 = f"rechazado: {_e}"
    _filas.append({"prueba": "cargar_estricto('{\"sup\": Infinity}')", "resultado": _r2})
    # 3) hash: orden de claves y separadores
    _a = copy.deepcopy(art_v100)
    _b = dict(reversed(list(_a.items())))                         # mismo contenido, otro orden
    _h_ing_a = hashlib.sha256(json.dumps(_a, ensure_ascii=False).encode()).hexdigest()
    _h_ing_b = hashlib.sha256(json.dumps(_b, ensure_ascii=False).encode()).hexdigest()
    _h_can_a = hashlib.sha256(canonico(_a).encode()).hexdigest()
    _h_can_b = hashlib.sha256(canonico(_b).encode()).hexdigest()
    _filas.append({"prueba": "hash ingenuo (json.dumps sin sort_keys), dos órdenes de claves",
                   "resultado": f"{'iguales' if _h_ing_a == _h_ing_b else 'DISTINTOS'}: {_h_ing_a[:10]}… vs {_h_ing_b[:10]}…"})
    _filas.append({"prueba": "hash canónico, dos órdenes de claves",
                   "resultado": f"{'iguales' if _h_can_a == _h_can_b else 'DISTINTOS'}: {_h_can_a[:10]}…"})
    # 4) round-trip exacto de floats (repr más corto de Python)
    _rt = cargar_estricto(canonico(art_v100))
    _cortes_orig = [_b2["sup"] for _v in art_v100["variables"] for _b2 in _v["bins"] if _b2.get("sup") is not None]
    _cortes_rt = [_b2["sup"] for _v in _rt["variables"] for _b2 in _v["bins"] if _b2.get("sup") is not None]
    roundtrip_exacto = _cortes_orig == _cortes_rt and _rt == art_v100
    _filas.append({"prueba": "round-trip JSON de todos los floats del artefacto",
                   "resultado": "bit a bit idéntico" if roundtrip_exacto else "DIFIERE"})
    hash_orden_invariante = (_h_can_a == _h_can_b) and (_h_ing_a != _h_ing_b)
    tabla_json = pd.DataFrame(_filas)
    tabla_json
    return cargar_estricto, hash_orden_invariante, roundtrip_exacto, tabla_json


@app.cell
def _(mo):
    mo.md(r"""
    ### JSON Schema: estructura sí, semántica no

    El esquema (Draft 2020-12) valida **forma**: campos obligatorios, tipos, enumeraciones, `inf`/`sup`
    numérico o `null`. No puede expresar invariantes que cruzan elementos: contigüidad de bins, cortes
    crecientes, `factor = PDO/ln 2`, etiquetas de master scale = cortes + 1, WoE finitos. Y valida el objeto
    **Python**, donde `float('nan')` es un `number`: el NaN pasa el esquema. Por eso son tres capas:
    serialización estricta → JSON Schema → validación semántica.
    """)
    return


@app.cell
def _():
    ESQUEMA_ARTEFACTO = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ejemplo.invalid/serie2/M21/scorecard-artefacto-1.0.0.schema.json",
        "title": "Artefacto congelado de scorecard (Serie 2 · M21)",
        "type": "object",
        "required": ["formato", "modelo", "variables", "intercepto", "escalado", "calibracion",
                     "master_scale", "politica", "reason_codes", "contrato_datos", "integridad"],
        "additionalProperties": False,
        "properties": {
            "formato": {"type": "object", "required": ["nombre", "version_esquema"],
                        "properties": {"nombre": {"const": "scorecard-artefacto"},
                                       "version_esquema": {"type": "string"}}},
            "modelo": {"type": "object", "required": ["id", "version", "fecha_construccion"],
                       "properties": {
                           "id": {"type": "string", "minLength": 1},
                           "version": {"type": "string",
                                       "pattern": r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$"},
                           "fecha_construccion": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}}},
            "variables": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/variable"}},
            "intercepto": {"type": "number"},
            "escalado": {"type": "object",
                         "required": ["pdo", "score_base", "odds_base", "factor", "offset", "sentido"],
                         "properties": {"pdo": {"type": "number", "exclusiveMinimum": 0},
                                        "odds_base": {"type": "number", "exclusiveMinimum": 0},
                                        "factor": {"type": "number"}, "offset": {"type": "number"},
                                        "sentido": {"enum": ["mayor_es_mejor", "mayor_es_peor"]}}},
            "calibracion": {"type": "object", "required": ["metodo", "delta"],
                            "properties": {"metodo": {"enum": ["desplazamiento_logit", "ninguna"]},
                                           "delta": {"type": "number"}}},
            "master_scale": {"type": "object", "required": ["cortes", "etiquetas", "cierre"],
                             "properties": {"cortes": {"type": "array", "items": {"type": "number"}},
                                            "etiquetas": {"type": "array", "items": {"type": "string"}},
                                            "cierre": {"enum": ["izquierda", "derecha"]}}},
            "politica": {"type": "object", "required": ["cutoff", "no_visto"],
                         "properties": {"cutoff": {"type": "number"},
                                        "no_visto": {"enum": ["revisar", "rechazar", "neutro"]}}},
            "reason_codes": {"type": "object", "patternProperties": {"^RC\\d{2}$": {"type": "string"}},
                             "additionalProperties": False},
            "contrato_datos": {"type": "object", "required": ["variables", "umbrales"]},
            "procedencia": {"type": "object"},
            "integridad": {"type": "object", "required": ["algoritmo", "hash"],
                           "properties": {"algoritmo": {"const": "sha256"},
                                          "hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}},
        },
        "$defs": {
            "borde": {"type": ["number", "null"]},
            "bin": {"type": "object", "required": ["id", "tipo", "woe", "puntos"],
                    "properties": {"id": {"type": "integer", "minimum": 0},
                                   "tipo": {"enum": ["intervalo", "valor", "categoria", "missing"]},
                                   "inf": {"$ref": "#/$defs/borde"}, "sup": {"$ref": "#/$defs/borde"},
                                   "woe": {"type": "number"}, "puntos": {"type": "number"},
                                   "n_dev": {"type": "integer", "minimum": 0}},
                    "allOf": [{"if": {"properties": {"tipo": {"const": "intervalo"}}},
                               "then": {"required": ["inf", "sup"]}},
                              {"if": {"properties": {"tipo": {"enum": ["valor", "categoria"]}}},
                               "then": {"required": ["valor"]}}]},
            "variable": {"type": "object",
                         "required": ["nombre", "tipo", "beta", "bins", "no_visto", "reason_code"],
                         "properties": {"nombre": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                                        "tipo": {"enum": ["numerico", "categorico"]},
                                        "beta": {"type": "number"},
                                        "cierre": {"enum": ["derecha", "izquierda"]},
                                        "bins": {"type": "array", "minItems": 1,
                                                 "items": {"$ref": "#/$defs/bin"}},
                                        "no_visto": {"type": "object", "required": ["woe", "accion"]},
                                        "reason_code": {"type": "string", "pattern": "^RC\\d{2}$"},
                                        "tendencia_declarada": {"enum": ["creciente", "decreciente", None]}},
                         "if": {"properties": {"tipo": {"const": "numerico"}}},
                         "then": {"required": ["cierre"]}},
        },
    }
    return (ESQUEMA_ARTEFACTO,)


@app.cell
def _(ESQUEMA_ARTEFACTO, art_v100, canonico, cargar_estricto, copy, hash_contenido, jsonschema, math, np, pd):
    _validador = jsonschema.Draft202012Validator(ESQUEMA_ARTEFACTO)

    def validar_semantica(art, tol=1e-9):
        """Invariantes que JSON Schema no puede expresar. Devuelve (errores, avisos)."""
        _err, _avi = [], []
        _e = art["escalado"]
        if not math.isclose(_e["factor"], _e["pdo"] / math.log(2), rel_tol=0, abs_tol=tol):
            _err.append("factor != pdo/ln2")
        if not math.isclose(_e["offset"], _e["score_base"] - _e["factor"] * math.log(_e["odds_base"]),
                            rel_tol=0, abs_tol=tol):
            _err.append("offset inconsistente con el ancla")
        _ms = art["master_scale"]
        if len(_ms["etiquetas"]) != len(_ms["cortes"]) + 1:
            _err.append("master scale: etiquetas != cortes + 1")
        if list(_ms["cortes"]) != sorted(set(_ms["cortes"])):
            _err.append("master scale: cortes no estrictamente crecientes")
        _n = len(art["variables"])
        for _v in art["variables"]:
            _nom = _v["nombre"]
            if not all(math.isfinite(_b["woe"]) and math.isfinite(_b["puntos"]) for _b in _v["bins"]):
                _err.append(f"{_nom}: WoE o puntos no finitos")
            if sorted(_b["id"] for _b in _v["bins"]) != list(range(len(_v["bins"]))):
                _err.append(f"{_nom}: ids de bin no son 0..k-1")
            if _v["reason_code"] not in art["reason_codes"]:
                _err.append(f"{_nom}: reason code sin texto")
            _esp = -(_v["beta"] * np.array([_b["woe"] for _b in _v["bins"]]) + art["intercepto"] / _n) \
                * _e["factor"] + _e["offset"] / _n
            if np.abs(_esp - np.array([_b["puntos"] for _b in _v["bins"]])).max() > 1e-6:
                _err.append(f"{_nom}: puntos inconsistentes con β, WoE y escala")
            if _v["tipo"] == "numerico":
                _iv = [_b for _b in _v["bins"] if _b["tipo"] == "intervalo"]
                if not _iv or _iv[0]["inf"] is not None or _iv[-1]["sup"] is not None:
                    _err.append(f"{_nom}: los extremos deben ser no acotados (null)")
                for _a, _b in zip(_iv[:-1], _iv[1:]):
                    if _a["sup"] is None or _b["inf"] is None or _a["sup"] != _b["inf"]:
                        _err.append(f"{_nom}: bins no contiguos ({_a['sup']} vs {_b['inf']})")
                    elif not _a["sup"] > (-math.inf if _a["inf"] is None else _a["inf"]):
                        _err.append(f"{_nom}: cortes no crecientes")
                _t = _v.get("tendencia_declarada")
                if _t and len(_iv) > 1:
                    _d = np.diff([_b["woe"] for _b in _iv])
                    if (_t == "creciente" and (_d < 0).any()) or (_t == "decreciente" and (_d > 0).any()):
                        _avi.append(f"{_nom}: WoE no monótono respecto de la tendencia declarada "
                                    f"({_t}; Δ mín {_d.min():+.4f}, máx {_d.max():+.4f})")
        if hash_contenido(art) != art.get("integridad", {}).get("hash"):
            _err.append("hash de integridad no coincide con el contenido")
        return _err, _avi

    def validar_artefacto(art):
        """Tres capas: serialización estricta, JSON Schema, semántica."""
        try:
            _obj = cargar_estricto(canonico(art))
            _ser = "ok"
        except ValueError as _ex:
            return {"serializacion": f"FALLA: {str(_ex)[:60]}", "json_schema": "—", "semantica": "—"}
        _es = sorted(_validador.iter_errors(_obj), key=lambda _x: list(_x.path))
        _sch = "ok" if not _es else f"FALLA: {_es[0].message[:70]}"
        _se, _ = validar_semantica(_obj)
        return {"serializacion": _ser, "json_schema": _sch, "semantica": "ok" if not _se else f"FALLA: {_se[0]}"}

    def _mutar(f):
        _a = copy.deepcopy(art_v100)
        f(_a)
        return _a

    def _sin_delta(a):
        del a["calibracion"]["delta"]

    def _sup_texto(a):
        a["variables"][0]["bins"][4]["inf"] = "inf"

    def _cutoff_texto(a):
        a["politica"]["cutoff"] = "530"

    def _woe_nan(a):
        a["variables"][0]["bins"][1]["woe"] = float("nan")

    def _hueco(a):
        a["variables"][0]["bins"][1]["sup"] = a["variables"][0]["bins"][1]["sup"] + 0.01

    def _factor_malo(a):
        a["escalado"]["pdo"] = 25

    def _cutoff_editado(a):
        a["politica"]["cutoff"] = 500.0

    casos_validacion = {
        "artefacto v1.0.0 íntegro": art_v100,
        "falta calibracion.delta": _mutar(_sin_delta),
        "borde como texto \"inf\"": _mutar(_sup_texto),
        "cutoff como texto \"530\"": _mutar(_cutoff_texto),
        "WoE = NaN": _mutar(_woe_nan),
        "bins con hueco (no contiguos)": _mutar(_hueco),
        "PDO editado sin recalcular factor": _mutar(_factor_malo),
        "cutoff editado a mano (sin resellar)": _mutar(_cutoff_editado),
    }
    tabla_validacion = pd.DataFrame([{"caso": _k, **validar_artefacto(_a)} for _k, _a in casos_validacion.items()])
    avisos_v100 = validar_semantica(art_v100)[1]
    tabla_validacion
    return avisos_v100, casos_validacion, tabla_validacion, validar_artefacto, validar_semantica


@app.cell
def _(avisos_v100, mo):
    mo.md(f"""
    **Lectura.** Cada mutación la caza **una** capa distinta: el NaN solo lo caza la serialización estricta
    (el esquema lo habría dejado pasar), el hueco entre bins y el PDO editado solo la capa semántica (el
    esquema los deja pasar), el cutoff editado a mano sin re-sellar solo el hash. Ninguna capa sola basta.

    Avisos semánticos del artefacto v1.0.0 (no bloquean; van al informe de validación):
    {chr(10).join('- ' + _a for _a in avisos_v100) if avisos_v100 else '- ninguno'}

    El aviso sobre `antiguedad_meses` es real: el WoE baja de −0,135 a −0,137 entre dos bins centrales.
    El test de monotonía del motor (§6) lo va a encontrar por el otro lado.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. El motor `puntuar`: función pura, vectorizada, sin DEV

    Firma: `puntuar(lote, art) -> DataFrame` con **el mismo índice** que el lote. Sin estado, sin leer nada
    más que el lote y el artefacto, sin mirar el target. Dos implementaciones:

    1. **numpy desde cero**: `np.searchsorted(cortes, x, side="left")` asigna $(a,b]$ (con `side="right"`
       sería $[a,b)$); valores aparte, missing, categorías y no vistos con máscaras.
    2. **librería**: `pd.cut(x, bordes, right=True, labels=False)` → códigos enteros (nunca etiquetas de
       texto: ahí vive el bug B).

    El score se calcula desde $\eta_{cal}=\beta_0+\sum_v\beta_v\,\text{WoE}_v+\delta$ como
    $\text{offset}-\text{factor}\cdot\eta_{cal}$, sin pasar por la PD (no necesita *clip*). Reason codes: los 3
    con más puntos perdidos respecto del máximo de su variable (método «max», PMML `pointsBelow`).
    """)
    return


@app.cell
def _(expit, np, pd):
    def _woe_numerico(x, var):
        """numpy puro: bin id, WoE y marca de no visto para una variable numérica."""
        _iv = [_b for _b in var["bins"] if _b["tipo"] == "intervalo"]
        _cortes = np.array([_b["sup"] for _b in _iv[:-1]], dtype=float)
        _lado = "left" if var["cierre"] == "derecha" else "right"      # (a,b] ↔ left ; [a,b) ↔ right
        _idx = np.searchsorted(_cortes, x, side=_lado)
        _w = np.array([_b["woe"] for _b in _iv])[_idx]
        _id = np.array([_b["id"] for _b in _iv])[_idx]
        for _b in var["bins"]:
            if _b["tipo"] == "valor":
                _m = x == _b["valor"]
                _w = np.where(_m, _b["woe"], _w)
                _id = np.where(_m, _b["id"], _id)
        _nv = ~np.isfinite(x)                                           # NaN e ±inf
        _miss = [_b for _b in var["bins"] if _b["tipo"] == "missing"]
        _nan = np.isnan(x)
        if _miss:
            _w = np.where(_nan, _miss[0]["woe"], _w)
            _id = np.where(_nan, _miss[0]["id"], _id)
            _nv = _nv & ~_nan
        _w = np.where(_nv, var["no_visto"]["woe"], _w)
        _id = np.where(_nv, -1, _id)
        return _id, _w, _nv

    def _woe_categorico(serie, var):
        _vals = pd.Series(serie).to_numpy(dtype=object)
        _nan = pd.isna(_vals)
        _claves = np.where(_nan, "\x00MISSING", _vals.astype(str))
        _uniq, _inv = np.unique(_claves, return_inverse=True)
        _mapa = {_b["valor"]: (_b["id"], _b["woe"]) for _b in var["bins"] if _b["tipo"] == "categoria"}
        _miss = [_b for _b in var["bins"] if _b["tipo"] == "missing"]
        if _miss:
            _mapa["\x00MISSING"] = (_miss[0]["id"], _miss[0]["woe"])
        _ids_u = np.array([_mapa.get(_u, (-1, var["no_visto"]["woe"]))[0] for _u in _uniq], dtype=int)
        _w_u = np.array([_mapa.get(_u, (-1, var["no_visto"]["woe"]))[1] for _u in _uniq], dtype=float)
        _id = _ids_u[_inv]
        return _id, _w_u[_inv], _id == -1

    def asignar_bins(lote, art):
        """Matrices (n × k): bin id (−1 = no visto), WoE y marca de no visto. numpy puro."""
        _n, _k = len(lote), len(art["variables"])
        _B = np.empty((_n, _k), dtype=int)
        _W = np.empty((_n, _k), dtype=float)
        _NV = np.empty((_n, _k), dtype=bool)
        for _j, _var in enumerate(art["variables"]):
            _col = lote[_var["nombre"]]                 # KeyError si falta: ruidoso a propósito
            if _var["tipo"] == "numerico":
                _x = _col.to_numpy(dtype=float)          # ValueError si llega texto: ruidoso a propósito
                _B[:, _j], _W[:, _j], _NV[:, _j] = _woe_numerico(_x, _var)
            else:
                _B[:, _j], _W[:, _j], _NV[:, _j] = _woe_categorico(_col, _var)
        return _B, _W, _NV

    def asignar_bins_pdcut(lote, art):
        """Implementación 2 (librería): pd.cut con labels=False + map. Mismas matrices."""
        _n, _k = len(lote), len(art["variables"])
        _B = np.empty((_n, _k), dtype=int)
        _W = np.empty((_n, _k), dtype=float)
        _NV = np.empty((_n, _k), dtype=bool)
        for _j, _var in enumerate(art["variables"]):
            _s = lote[_var["nombre"]]
            _nvw = _var["no_visto"]["woe"]
            if _var["tipo"] == "numerico":
                _iv = [_b for _b in _var["bins"] if _b["tipo"] == "intervalo"]
                _bordes = [-np.inf] + [_b["sup"] for _b in _iv[:-1]] + [np.inf]
                _cod = pd.cut(_s.astype(float), _bordes, right=(_var["cierre"] == "derecha"),
                              labels=False, include_lowest=True)
                _id = pd.Series(_cod, index=_s.index).map(dict(enumerate(_b["id"] for _b in _iv)))
                for _b in _var["bins"]:
                    if _b["tipo"] == "valor":
                        _id = _id.mask(_s == _b["valor"], _b["id"])
                _miss = [_b for _b in _var["bins"] if _b["tipo"] == "missing"]
                if _miss:
                    _id = _id.mask(_s.isna(), _miss[0]["id"])
                _id = _id.mask(~np.isfinite(_s.astype(float)) & _id.notna() & ~_s.isna(), np.nan)
            else:
                _mapa = {_b["valor"]: _b["id"] for _b in _var["bins"] if _b["tipo"] == "categoria"}
                _id = _s.astype(object).map(_mapa)
                _miss = [_b for _b in _var["bins"] if _b["tipo"] == "missing"]
                if _miss:
                    _id = _id.mask(_s.isna(), _miss[0]["id"])
            _woe_de_id = {_b["id"]: _b["woe"] for _b in _var["bins"]}
            _NV[:, _j] = _id.isna().to_numpy()
            _B[:, _j] = _id.fillna(-1).astype(int).to_numpy()
            _W[:, _j] = _id.map(_woe_de_id).fillna(_nvw).astype(float).to_numpy()
        return _B, _W, _NV

    def puntuar(lote, art, asignador=None, marcar_no_vistos=True, razones=True):
        """El motor: SOLO lote + artefacto. Devuelve un DataFrame con el índice del lote."""
        _B, _W, _NV = (asignador or asignar_bins)(lote, art)
        _lp = np.full(len(lote), art["intercepto"], dtype=float)
        for _j, _var in enumerate(art["variables"]):
            _lp = _lp + _var["beta"] * _W[:, _j]        # orden fijo de variables ⇒ determinista
        _eta = _lp + art["calibracion"]["delta"]
        _e = art["escalado"]
        _score = _e["offset"] - _e["factor"] * _eta
        _ms = art["master_scale"]
        _lado = "right" if _ms["cierre"] == "izquierda" else "left"
        _banda = np.array(_ms["etiquetas"])[np.searchsorted(np.array(_ms["cortes"]), _score, side=_lado)]
        _nnv = _NV.sum(axis=1)
        _marca = (_nnv > 0) if marcar_no_vistos else np.zeros(len(lote), bool)
        _dec = np.where(_marca, "revisar",
                        np.where(_score >= art["politica"]["cutoff"], "aprobar", "rechazar"))
        _out = pd.DataFrame({"score": _score, "pd": expit(_eta), "banda": _banda,
                             "n_no_vistos": _nnv, "decision": _dec}, index=lote.index)
        if razones:
            _n = len(art["variables"])
            _P = -(np.array([_v["beta"] for _v in art["variables"]]) * _W + art["intercepto"] / _n) \
                * _e["factor"] + _e["offset"] / _n
            _Pmax = np.array([max(_b["puntos"] for _b in _v["bins"]) for _v in art["variables"]])
            _perd = _Pmax - _P
            _ord = np.argsort(-_perd, axis=1, kind="stable")
            _cod = np.array([_v["reason_code"] for _v in art["variables"]])
            for _r in range(3):
                _col = _ord[:, _r]
                _ok = _perd[np.arange(len(lote)), _col] > 1e-9
                _out[f"rc{_r + 1}"] = np.where(_ok, _cod[_col], "")
        _out.attrs["hash_artefacto"] = art["integridad"]["hash"]
        return _out
    return asignar_bins, asignar_bins_pdcut, puntuar


@app.cell
def _(FACTOR, MUESTRAS, art_v100, asignar_bins, asignar_bins_pdcut, canonico, cargar_estricto, np, pd,
      puntuar, score_notebook):
    _art_rt = cargar_estricto(canonico(art_v100))        # el artefacto tal como lo lee producción
    _filas = []
    for _m, _df in MUESTRAS.items():
        _s_nb, _pc_nb = score_notebook(_df)
        _o = puntuar(_df, art_v100)
        _o_lib = puntuar(_df, art_v100, asignador=asignar_bins_pdcut)
        _o_rt = puntuar(_df, _art_rt)
        _B1, _W1, _N1 = asignar_bins(_df, art_v100)
        _B2, _W2, _N2 = asignar_bins_pdcut(_df, art_v100)
        _banda_nb = pd.cut(_s_nb, [-np.inf] + art_v100["master_scale"]["cortes"] + [np.inf],
                           labels=art_v100["master_scale"]["etiquetas"], right=False).astype(str)
        _dec_nb = np.where(_s_nb >= art_v100["politica"]["cutoff"], "aprobar", "rechazar")
        # score desde la TABLA de puntos: Σ puntos − δ·factor
        _P = sum(np.array([_b["puntos"] for _b in _v["bins"]])[
            np.searchsorted([_b["id"] for _b in _v["bins"]], _B1[:, _j])]
            for _j, _v in enumerate(art_v100["variables"]))
        _s_tabla = _P - art_v100["calibracion"]["delta"] * FACTOR
        _filas.append({
            "muestra": _m, "n": len(_df),
            "max|Δscore| motor vs notebook": float(np.abs(_o["score"].to_numpy() - _s_nb).max()),
            "max|ΔPD| motor vs notebook": float(np.abs(_o["pd"].to_numpy() - _pc_nb).max()),
            "bandas iguales": bool((_o["banda"].to_numpy() == _banda_nb.to_numpy()).all()),
            "decisiones iguales": bool((_o["decision"].to_numpy() == _dec_nb).all()),
            "numpy == pd.cut (bit a bit)": bool(np.array_equal(_B1, _B2) and np.array_equal(_W1, _W2)
                                                and np.array_equal(_o["score"].to_numpy(),
                                                                   _o_lib["score"].to_numpy())),
            "JSON round-trip (bit a bit)": bool(np.array_equal(_o["score"].to_numpy(), _o_rt["score"].to_numpy())),
            "max|Σpuntos − δ·factor − score|": float(np.abs(_s_tabla - _o["score"].to_numpy()).max()),
            "índice preservado": bool(_o.index.equals(_df.index)),
            "no vistos": int((_o["n_no_vistos"] > 0).sum()),
        })
    tabla_paridad = pd.DataFrame(_filas).set_index("muestra")
    tabla_paridad
    return (tabla_paridad,)


@app.cell
def _(fmt_num, mo, tabla_paridad):
    _mx = tabla_paridad["max|Δscore| motor vs notebook"].max()
    mo.md(f"""
    **Paridad.** Diferencia máxima de score motor vs notebook: **{_mx:.1e} puntos** en las cuatro muestras
    (el curso: 1,1e-13); bandas y decisiones idénticas. No es cero porque son **dos caminos de cómputo**
    distintos (statsmodels hace `X @ β` y el curso pasa por `logit(expit(·))`; el motor suma variable a variable
    desde η): el error es de redondeo IEEE 754 — un ulp de un número entre 512 y 1024 es 1,1e-13. La **tolerancia se
    declara** (aquí 1e-9 puntos y decisiones idénticas) en vez de exigir igualdad de bits entre caminos
    distintos. Donde el camino es el mismo — numpy vs `pd.cut`, artefacto en memoria vs releído del JSON —
    exigimos y obtenemos **igualdad bit a bit**. Y la tabla de puntos reproduce el score con error
    {fmt_num(tabla_paridad['max|Σpuntos − δ·factor − score|'].max() * 1e12, 1)}e-12: producción podría puntuar
    sumando la tabla.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Casos borde: la semántica que se congela junto con los números

    Un golden de casos borde construido a mano. Cada fila parte de una misma solicitud TTD y cambia **un**
    campo. La columna `bin_esperado` es la especificación (escrita antes de correr): si un cambio del motor la
    altera, el build falla.
    """)
    return


@app.cell
def _(art_v100, asignar_bins, np, pd, puntuar, ttd):
    _base = ttd.iloc[[0]].copy()
    _var = {_v["nombre"]: _v for _v in art_v100["variables"]}
    _c0 = _var["uso_linea_prom_12m"]["bins"][0]["sup"]
    _casos = [
        ("base (sin cambios)", {}, None, None),
        ("uso_linea = corte exacto c₁", {"uso_linea_prom_12m": _c0}, "uso_linea_prom_12m", 0),
        ("uso_linea = siguiente float después de c₁", {"uso_linea_prom_12m": float(np.nextafter(_c0, 2))},
         "uso_linea_prom_12m", 1),
        ("consultas_6m = 1 (corte entero)", {"consultas_6m": 1.0}, "consultas_6m", 0),
        ("consultas_6m = 2 (corte entero)", {"consultas_6m": 2.0}, "consultas_6m", 1),
        ("antiguedad_meses = 28 (corte entero)", {"antiguedad_meses": 28.0}, "antiguedad_meses", 0),
        ("meses_desde_mora = 13 (moda aparte)", {"meses_desde_mora_12m": 13.0}, "meses_desde_mora_12m", 4),
        ("meses_desde_mora = −99 (sin bureau)", {"meses_desde_mora_12m": -99.0}, "meses_desde_mora_12m", 0),
        ("meses_desde_mora = −9 (nunca mora)", {"meses_desde_mora_12m": -9.0}, "meses_desde_mora_12m", 0),
        ("renta_mm = NaN (missing visto en DEV)", {"renta_mm": np.nan}, "renta_mm", 5),
        ("uso_linea = NaN (missing NO visto)", {"uso_linea_prom_12m": np.nan}, "uso_linea_prom_12m", -1),
        ("carga = +inf", {"carga_financiera": np.inf}, "carga_financiera", -1),
        ("canal = 'App' (mayúscula)", {"canal": "App"}, "canal", -1),
        ("canal = 'marketplace' (nuevo)", {"canal": "marketplace"}, "canal", -1),
        ("canal = 'app'", {"canal": "app"}, "canal", 0),
    ]
    _filas_df = []
    for _k, (_nombre, _cambios, _vv, _esp) in enumerate(_casos):
        _r = _base.copy()
        for _c, _x in _cambios.items():
            _r[_c] = _x
        _r.index = [f"BORDE{_k:02d}"]
        _filas_df.append(_r)
    casos_borde = pd.concat(_filas_df)
    _B, _, _ = asignar_bins(casos_borde, art_v100)
    _o = puntuar(casos_borde, art_v100)
    _nombres_var = [_v["nombre"] for _v in art_v100["variables"]]
    tabla_bordes_golden = pd.DataFrame({
        "caso": [_c[0] for _c in _casos],
        "variable": [_c[2] or "—" for _c in _casos],
        "bin_esperado": [_c[3] if _c[3] is not None else "—" for _c in _casos],
        "bin_asignado": [(_B[_i, _nombres_var.index(_c[2])] if _c[2] else "—") for _i, _c in enumerate(_casos)],
        "score": _o["score"].round(3).to_numpy(),
        "decision": _o["decision"].to_numpy(),
        "rc1": _o["rc1"].to_numpy(),
    }, index=casos_borde.index)
    bordes_ok = bool(all(_c[3] is None or _B[_i, _nombres_var.index(_c[2])] == _c[3]
                         for _i, _c in enumerate(_casos)))
    tabla_bordes_golden
    return bordes_ok, casos_borde, tabla_bordes_golden


@app.cell
def _(bordes_ok, mo):
    mo.md(f"""
    **Lectura.** Semántica verificada: {'✔ todas las asignaciones coinciden con la especificación' if bordes_ok else '✘ HAY DIFERENCIAS'}.
    Tres cosas para el comité: (1) el valor **exactamente** en el corte va al bin de abajo, $(a,b]$, y el
    siguiente float representable ya va al de arriba; (2) **−99 y −9 comparten bin** con los meses de mora
    bajos: el binning del curso trata el código «sin bureau» como un número pequeño — una decisión que el
    artefacto hace visible y que M14 corrige separando especiales; (3) `'App'` no es `'app'`: sin normalizar
    categorías en el contrato, un cambio de mayúsculas en el front convierte al canal en «no visto» y manda la
    solicitud a revisión (o, en el motor del curso, a WoE 0 en silencio).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Semántica de bordes: el bug de paridad más barato de cometer

    Los cortes del binning son **cuantiles de DEV**; con datos redondeados (4 decimales) o enteros, un cuantil
    cae exactamente sobre valores observados. En este generador, **51 % de las solicitudes TTD tienen
    `consultas_6m` exactamente en un corte** y 22 % `meses_desde_mora_12m`. Tres implementaciones «casi
    iguales» del artefacto:

    - **cierre izquierdo** $[a,b)$ (quien reimplementa en SQL con `>=`/`<`, o `np.digitize` con `right=False`);
    - **cortes redondeados** a $k$ decimales («para que se lea bien en el anexo»);
    - **cortes en `float32`** (una columna `REAL`, un `float` de Java).
    """)
    return


@app.cell
def _(mo):
    decimales_ui = mo.ui.slider(1, 17, value=2, step=1, label="Decimales al redondear los cortes (k)",
                                show_value=True)
    decimales_ui
    return (decimales_ui,)


@app.cell
def _(CUTOFF, art_v100, asignar_bins, copy, decimales_ui, np, pd, plt, puntuar, ttd):
    def variante_bordes(art, tipo, k=2):
        """Copia del artefacto con otra semántica/precisión de bordes (sin re-sellar: es un experimento)."""
        _a = copy.deepcopy(art)
        for _v in _a["variables"]:
            if _v["tipo"] != "numerico":
                continue
            if tipo == "cierre_izquierdo":
                _v["cierre"] = "izquierda"
            for _b in _v["bins"]:
                for _campo in ("inf", "sup"):
                    if _b.get("tipo") == "intervalo" and _b[_campo] is not None:
                        if tipo == "redondeo":
                            _b[_campo] = round(_b[_campo], k)
                        elif tipo == "float32":
                            _b[_campo] = float(np.float32(_b[_campo]))
        return _a

    _B0, _, _ = asignar_bins(ttd, art_v100)
    _d0 = puntuar(ttd, art_v100, razones=False)["decision"].to_numpy()
    danio_por_decimales = {_k: float((puntuar(ttd, variante_bordes(art_v100, "redondeo", _k), razones=False)
                                      ["decision"].to_numpy() != _d0).mean()) for _k in range(1, 18)}
    _nombres = [_v["nombre"] for _v in art_v100["variables"]]
    _filas = []
    for _tipo, _et in [("cierre_izquierdo", "cierre [a,b)"), ("redondeo", f"cortes a {decimales_ui.value} decimales"),
                       ("float32", "cortes en float32")]:
        _a = variante_bordes(art_v100, _tipo, decimales_ui.value)
        _B, _, _ = asignar_bins(ttd, _a)
        _o = puntuar(ttd, _a, razones=False)
        _fila = {"variante": _et}
        _fila.update({_n: float((_B[:, _j] != _B0[:, _j]).mean()) for _j, _n in enumerate(_nombres)})
        _fila["% decisiones que cambian"] = float((_o["decision"].to_numpy() != _d0).mean())
        _filas.append(_fila)
    tabla_bordes = pd.DataFrame(_filas).set_index("variante")

    _num = [_n for _j, _n in enumerate(_nombres) if art_v100["variables"][_j]["tipo"] == "numerico"]
    fig_bordes, _ax = plt.subplots(figsize=(8.5, 3.6))
    _x = np.arange(len(_num))
    _colores = ["#2a78d6", "#eb6834", "#1baf7a"]
    for _i, (_var_nombre, _fila) in enumerate(tabla_bordes.iterrows()):
        _ax.bar(_x + (_i - 1) * 0.27, 100 * _fila[_num].to_numpy(dtype=float), width=0.25,
                color=_colores[_i], label=_var_nombre)
    _ax.set_xticks(_x, [_n.replace("_prom", "").replace("_12m", "").replace("_meses", "") for _n in _num],
                   rotation=20, fontsize=8)
    _ax.set_ylabel("% de solicitudes TTD que cambian de bin")
    _ax.set_title(f"Semántica de bordes (CUTOFF {CUTOFF}): mismo modelo, otra implementación")
    _ax.legend(fontsize=8, frameon=False)
    _ax.grid(axis="y", alpha=0.3)
    fig_bordes.tight_layout()
    fig_bordes
    return danio_por_decimales, fig_bordes, tabla_bordes, variante_bordes


@app.cell
def _(danio_por_decimales, decimales_ui, fmt_pct, mo, tabla_bordes):
    _t = tabla_bordes.copy()
    _izq, _red, _f32 = _t.iloc[0], _t.iloc[1], _t.iloc[2]
    _k_seguro = min(_k for _k in danio_por_decimales if all(danio_por_decimales[_j] == 0
                                                             for _j in danio_por_decimales if _j >= _k))
    mo.vstack([
        mo.md(f"""
    **Lectura.** Cerrar a la izquierda mueve de bin al {fmt_pct(_izq['consultas_6m'])} de las solicitudes en
    `consultas_6m` y al {fmt_pct(_izq['meses_desde_mora_12m'])} en `meses_desde_mora_12m`, y cambia
    **{fmt_pct(_izq['% decisiones que cambian'], 2)} de las decisiones** sin una sola excepción. Redondear los cortes
    a {decimales_ui.value} decimales cambia {fmt_pct(_red['% decisiones que cambian'], 2)} de las decisiones; en este
    lote el daño desaparece desde k = {_k_seguro} decimales (los datos vienen con ≤ 4 decimales salvo `renta_mm`),
    pero ese umbral es una propiedad **de estos datos**, no del artefacto: el único formato seguro es el `repr`
    completo (≤ 17 dígitos significativos, round-trip exacto). `float32` cambia {fmt_pct(_f32['% decisiones que cambian'], 2)}
    aquí; no es una garantía: basta un valor observado entre el corte en float64 y su redondeo a float32 para que
    un caso cambie de bin, y la paridad exigida es cero. La convención de cierre es parte del artefacto
    (`"cierre": "derecha"`), no del código de quien lo implementa.
    """),
        _t.style.format("{:.2%}"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Bug A (silencioso) vs bug B (ruidoso)

    - **Bug A**: producción re-ajusta los cortes con el lote del día y mapea el WoE **por posición** (el bin
      k-ésimo del lote recibe el WoE del bin k-ésimo de DEV). Todo encuentra WoE: cero nulos, cero alarmas.
    - **Bug B**: re-ajusta los cortes y mapea por **etiqueta de texto**; las etiquetas nuevas no existen en el
      mapa y caen a WoE 0 (neutro). Hace más daño y se delata solo.

    El lote se arma desde HO+OOT+TTD con **inclinación exponencial** $w_i\propto e^{\gamma z_i}$, con $z$ el uso
    de línea estandarizado: $\gamma>0$ trae un lote más riesgoso **sin cambiar a ningún cliente**. Esa es la
    esencia del bug: el score de un cliente pasa a depender de con quién vino.
    """)
    return


@app.cell
def _(mo):
    drift_ui = mo.ui.slider(-1.5, 1.5, value=0.5, step=0.25, label="Drift del lote γ (inclinación hacia riesgo)",
                            show_value=True)
    tamano_ui = mo.ui.dropdown(options={"250 solicitudes": 250, "1.000 solicitudes": 1000,
                                        "4.000 solicitudes": 4000},
                               value="1.000 solicitudes", label="Tamaño del lote diario")
    mo.hstack([drift_ui, tamano_ui], justify="start", gap=2)
    return drift_ui, tamano_ui


@app.cell
def _(art_v100, binear, expit, ho, mapas_dev, np, oot, pd, ttd):
    POOL_LOTES = pd.concat([ho, oot, ttd])

    def generar_lote(pool, m, gamma, semilla):
        """Muestra m solicitudes sin reemplazo con pesos ∝ exp(γ·z_uso): cambia la POBLACIÓN, no los clientes."""
        _rng = np.random.default_rng(semilla)
        _z = pool["uso_linea_prom_12m"].to_numpy()
        _z = (_z - _z.mean()) / _z.std()
        _w = np.exp(gamma * _z)
        _idx = _rng.choice(len(pool), size=m, replace=False, p=_w / _w.sum())
        return pool.iloc[np.sort(_idx)]

    def _cerrar(lp, art, idx):
        _eta = lp + art["calibracion"]["delta"]
        _s = art["escalado"]["offset"] - art["escalado"]["factor"] * _eta
        return pd.DataFrame({"score": _s, "pd": expit(_eta)}, index=idx)

    def puntuar_bug_a(lote, art):
        """BUG A: cortes re-ajustados en el lote; WoE de DEV asignado POR POSICIÓN. Devuelve (salida, n_sin_mapa)."""
        _lp = np.full(len(lote), art["intercepto"])
        _sin = 0
        for _var in art["variables"]:
            _v = _var["nombre"]
            _et, _orden = binear(lote[_v])                       # ← cortes del LOTE
            _woe_dev = mapas_dev[_v]
            if _orden is not None and _var["tipo"] == "numerico":
                _int_dev = [_b["woe"] for _b in _var["bins"] if _b["tipo"] == "intervalo"]
                _int_lote = [_o for _o in _orden if _o.startswith("(")]
                _mapa = {_e: _int_dev[min(_k, len(_int_dev) - 1)] for _k, _e in enumerate(_int_lote)}
                _moda_dev = [_b for _b in _var["bins"] if _b["tipo"] == "valor"]
                _moda_lote = [_o for _o in _orden if _o.startswith("=")]
                if _moda_dev and _moda_lote:
                    _mapa[_moda_lote[0]] = _moda_dev[0]["woe"]
                if "MISSING" in _woe_dev.index:
                    _mapa["MISSING"] = float(_woe_dev["MISSING"])
            else:
                _mapa = _woe_dev.to_dict()
            _w = _et.map(_mapa).astype(float)
            _sin += int(_w.isna().sum())
            _lp = _lp + _var["beta"] * _w.fillna(0.0).to_numpy()
        return _cerrar(_lp, art, lote.index), _sin

    def puntuar_bug_b(lote, art):
        """BUG B: cortes re-ajustados en el lote; WoE por ETIQUETA de texto (no calza → 0)."""
        _lp = np.full(len(lote), art["intercepto"])
        _sin = 0
        for _var in art["variables"]:
            _v = _var["nombre"]
            _et, _ = binear(lote[_v])
            _w = _et.map(mapas_dev[_v].to_dict()).astype(float)
            _sin += int(_w.isna().sum())
            _lp = _lp + _var["beta"] * _w.fillna(0.0).to_numpy()
        return _cerrar(_lp, art, lote.index), _sin
    return POOL_LOTES, generar_lote, puntuar_bug_a, puntuar_bug_b


@app.cell
def _(CUTOFF, POOL_LOTES, art_v100, drift_ui, generar_lote, np, pd, puntuar, puntuar_bug_a, puntuar_bug_b,
      tamano_ui, ttd):
    def comparar_bugs(lote, art):
        _ok = puntuar(lote, art, razones=False)["score"].to_numpy()
        _filas = []
        for _nom, _f in [("A · cortes del lote, WoE por posición", puntuar_bug_a),
                         ("B · etiquetas que ya no calzan", puntuar_bug_b)]:
            _s, _sin = _f(lote, art)
            _d = _s["score"].to_numpy() - _ok
            _cambia = (_s["score"].to_numpy() >= CUTOFF) != (_ok >= CUTOFF)
            _filas.append({"bug": _nom, "Δ medio (pts)": _d.mean(), "máx |Δ| (pts)": np.abs(_d).max(),
                           "cambian decisión": int(_cambia.sum()), "% cambian": _cambia.mean(),
                           "% WoE sin mapa": _sin / (len(lote) * len(art["variables"])),
                           "dist. media al cutoff (cambian)": np.abs(_ok[_cambia] - CUTOFF).mean()
                           if _cambia.any() else np.nan,
                           "dist. media al cutoff (resto)": np.abs(_ok[~_cambia] - CUTOFF).mean()})
        return pd.DataFrame(_filas).set_index("bug"), _ok

    lote_actual = generar_lote(POOL_LOTES, tamano_ui.value, drift_ui.value, semilla=7)
    tabla_bugs_actual, _ = comparar_bugs(lote_actual, art_v100)
    tabla_bugs_ttd, score_ok_ttd = comparar_bugs(ttd, art_v100)     # el análogo exacto del curso: TTD completo
    tabla_bugs_actual
    return comparar_bugs, lote_actual, score_ok_ttd, tabla_bugs_actual, tabla_bugs_ttd


@app.cell
def _(CUTOFF, fmt_num, fmt_pct, mo, np, puntuar_bug_a, score_ok_ttd, tabla_bugs_ttd, ttd, art_v100):
    # aproximación de primer orden: % que cambia ≈ f_S(c) · E|Δ|
    _sa, _ = puntuar_bug_a(ttd, art_v100)
    _d = _sa["score"].to_numpy() - score_ok_ttd
    _h = 5.0
    _f_c = float(((score_ok_ttd > CUTOFF - _h) & (score_ok_ttd <= CUTOFF + _h)).mean() / (2 * _h))
    aprox_flip_a = _f_c * float(np.abs(_d).mean())
    # distancia media al cutoff de los que cambian, predicha: E[Δ²] / (2·E|Δ|)
    dist_pred_a = float((_d ** 2).mean() / (2 * np.abs(_d).mean()))
    _obs = tabla_bugs_ttd.iloc[0]["% cambian"]
    _b = tabla_bugs_ttd.iloc[1]
    mo.md(f"""
    **TTD completo ({fmt_num(len(ttd), 0)} solicitudes, el análogo del caso Austral):** bug A cambia
    **{fmt_pct(_obs, 2)}** de las decisiones con **{fmt_pct(tabla_bugs_ttd.iloc[0]['% WoE sin mapa'], 1)} de WoE sin
    mapa** (cero alarmas); bug B cambia {fmt_pct(_b['% cambian'], 2)} con {fmt_pct(_b['% WoE sin mapa'], 1)} de WoE
    sin mapa (se delata). En Banco Austral: 6,8 % vs 25,4 %, con 0 % vs 89 % sin mapa.

    Los que cambian con el bug A están a {fmt_num(tabla_bugs_ttd.iloc[0]['dist. media al cutoff (cambian)'], 1)} puntos
    del cutoff en promedio; el resto, a {fmt_num(tabla_bugs_ttd.iloc[0]['dist. media al cutoff (resto)'], 1)}
    (Austral: 10,0 vs 60,7). La aproximación de primer orden $f_S(c)\,\mathbb{{E}}|\Delta|$ =
    {fmt_num(_f_c, 4)} × {fmt_num(float(np.abs(_d).mean()), 2)} = **{fmt_pct(aprox_flip_a, 2)}** vs observado
    {fmt_pct(_obs, 2)}: el daño es densidad en el corte por tamaño medio del error (§3.3 del documento). La misma
    derivación predice la distancia media al cutoff de los que cambian, $\mathbb{{E}}[\Delta^2]/(2\,\mathbb{{E}}|\Delta|)$ =
    {fmt_num(dist_pred_a, 1)} puntos, vs {fmt_num(tabla_bugs_ttd.iloc[0]['dist. media al cutoff (cambian)'], 1)} observado.
    """)
    return aprox_flip_a, dist_pred_a


@app.cell
def _(CUTOFF, POOL_LOTES, art_v100, comparar_bugs, drift_ui, generar_lote, np, plt, tamano_ui):
    _grid = np.round(np.arange(-1.5, 1.51, 0.25), 2)
    _res_a, _res_b = [], []
    for _g in _grid:
        _pa, _pb = [], []
        for _rep in range(3):
            _lote = generar_lote(POOL_LOTES, tamano_ui.value, float(_g), semilla=100 + _rep)
            _t, _ = comparar_bugs(_lote, art_v100)
            _pa.append(_t.iloc[0]["% cambian"])
            _pb.append(_t.iloc[1]["% cambian"])
        _res_a.append(np.mean(_pa))
        _res_b.append(np.mean(_pb))
    curva_bugs = {"gamma": _grid, "bug_a": np.array(_res_a), "bug_b": np.array(_res_b)}

    fig_bugs, _ax = plt.subplots(figsize=(7.5, 3.6))
    _ax.plot(_grid, 100 * curva_bugs["bug_a"], marker="o", ms=4, lw=2, color="#2a78d6",
             label="Bug A (silencioso: 0 % WoE sin mapa)")
    _ax.plot(_grid, 100 * curva_bugs["bug_b"], marker="s", ms=4, lw=2, color="#eb6834",
             label="Bug B (ruidoso: WoE sin mapa)")
    _ax.axvline(drift_ui.value, color="#52514e", lw=1, ls="--", label=f"γ elegido = {drift_ui.value}")
    _ax.set_xlabel("Drift del lote γ (0 = población sin inclinar)")
    _ax.set_ylabel("% de decisiones que cambian")
    _ax.set_title(f"Re-ajustar el binner en producción · lote de {tamano_ui.value} · cutoff {CUTOFF} (media de 3 lotes)")
    _ax.set_ylim(bottom=0)
    _ax.grid(alpha=0.3)
    _ax.legend(fontsize=8, frameon=False)
    fig_bugs.tight_layout()
    fig_bugs
    return curva_bugs, fig_bugs


@app.cell
def _(curva_bugs, fmt_pct, mo, np, tamano_ui):
    _i0 = int(np.argmin(np.abs(curva_bugs["gamma"])))
    _a_gana = curva_bugs["gamma"][curva_bugs["bug_a"] > curva_bugs["bug_b"]]
    mo.md(f"""
    **Lectura.** Con lotes de {tamano_ui.value}, el bug A cambia {fmt_pct(curva_bugs['bug_a'][_i0], 1)} de las
    decisiones **incluso sin drift** (γ = 0): los cuantiles de un lote chico son ruidosos y cada día redibujan
    la partición. El mínimo no es cero en ningún γ. Con drift, el daño crece en ambos sentidos: si el lote viene
    más riesgoso, los cortes suben y cada cliente parece «menos extremo» de lo que es. El bug B hace más
    daño en casi todo el rango, pero no siempre{(": con γ ∈ {" + ", ".join(f"{_g:+.2f}" for _g in _a_gana) + "} el bug A daña más") if len(_a_gana) else ""}.
    Lo que no cambia es la huella: B siempre deja WoE sin mapa; A nunca. Pruebe con 250: el bug A de un
    originador pequeño (o de un canal con lotes chicos, como un concesionario de motos) es peor que el del banco
    grande.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Tests de propiedades: el que mata al bug A

    Un test de propiedades no compara contra un número esperado: afirma una **ley** que debe cumplirse para
    cualquier entrada, y la prueba contra muchas entradas generadas al azar (estilo QuickCheck/Hypothesis;
    aquí con un generador numpy para no agregar dependencias).

    | Propiedad | Enunciado | ¿Mata al bug A? |
    |---|---|---|
    | Permutación | `puntuar(lote[π])` = `puntuar(lote)[π]` | **No**: los cuantiles no dependen del orden |
    | **Subconjunto** | para todo $S\subseteq L$ e $i\in S$: `puntuar(S)[i] == puntuar(L)[i]` | **Sí** |
    | Idempotencia / pureza | no muta el lote; dos llamadas → mismo hash | No |
    | Monotonía | con β<0 y WoE monótono, el score es monótono en la variable | No (caza otra cosa) |
    """)
    return


@app.cell
def _(POOL_LOTES, art_v100, generar_lote, hashlib, np, pd, puntuar, puntuar_bug_a, puntuar_bug_b):
    def _score(f, lote, art):
        _o = f(lote, art)
        return (_o[0] if isinstance(_o, tuple) else _o)["score"]

    def prop_subconjunto(f, lote, art, n_casos=25, semilla=11):
        """∀ S ⊆ L: score_S(i) == score_L(i). Devuelve (pasa, máx |Δ|, tamaño del contraejemplo)."""
        _rng = np.random.default_rng(semilla)
        _full = _score(f, lote, art)
        _peor, _tam = 0.0, None
        for _ in range(n_casos):
            _m = int(_rng.integers(1, len(lote) + 1))
            _S = lote.iloc[np.sort(_rng.choice(len(lote), _m, replace=False))]
            _d = float(np.abs(_score(f, _S, art).to_numpy() - _full.loc[_S.index].to_numpy()).max())
            if _d > _peor:
                _peor, _tam = _d, _m
        return _peor == 0.0, _peor, _tam

    def prop_permutacion(f, lote, art, n_casos=10, semilla=12):
        _rng = np.random.default_rng(semilla)
        _full = _score(f, lote, art)
        _peor = 0.0
        for _ in range(n_casos):
            _P = lote.iloc[_rng.permutation(len(lote))]
            _peor = max(_peor, float(np.abs(_score(f, _P, art).loc[lote.index].to_numpy() - _full.to_numpy()).max()))
        return _peor == 0.0, _peor, None

    def huella_df(df):
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()

    def prop_pureza(f, lote, art):
        _antes = huella_df(lote)
        _s1 = _score(f, lote, art)
        _s2 = _score(f, lote, art)
        _ok = huella_df(lote) == _antes and np.array_equal(_s1.to_numpy(), _s2.to_numpy())
        return _ok, 0.0 if _ok else float("nan"), None

    lote_props = generar_lote(POOL_LOTES, 600, 0.5, semilla=3)
    _filas = []
    for _nom, _f in [("motor correcto (artefacto)", puntuar), ("bug A", puntuar_bug_a), ("bug B", puntuar_bug_b)]:
        for _pn, _p in [("permutación", prop_permutacion), ("subconjunto", prop_subconjunto),
                        ("pureza/idempotencia", prop_pureza)]:
            _ok, _d, _t = _p(_f, lote_props, art_v100)
            _filas.append({"implementación": _nom, "propiedad": _pn, "pasa": "✔" if _ok else "✘",
                           "máx |Δscore|": _d, "tamaño del subconjunto que la rompe": _t})
    tabla_props = pd.DataFrame(_filas).pivot(index="implementación", columns="propiedad", values="pasa")
    detalle_props = pd.DataFrame(_filas)
    tabla_props
    return detalle_props, huella_df, lote_props, prop_permutacion, prop_pureza, prop_subconjunto, tabla_props


@app.cell
def _(detalle_props, fmt_num, mo):
    _a = detalle_props[(detalle_props["implementación"] == "bug A") & (detalle_props["propiedad"] == "subconjunto")].iloc[0]
    mo.md(f"""
    **Lectura.** El bug A **pasa** la prueba de permutación (reordenar el lote no cambia sus cuantiles) y la de
    pureza (es determinista y no muta nada): es un bug «bien educado». Solo lo mata la propiedad de subconjunto:
    el mismo cliente, puntuado dentro de un sub-lote de {_a['tamaño del subconjunto que la rompe']} solicitudes,
    cambia hasta {fmt_num(_a['máx |Δscore|'], 1)} puntos. La prueba del curso («caso #7 solo, en 10 y en 8.585»)
    es la instancia mínima de esta ley; en CI se corre con cientos de subconjuntos aleatorios, incluido el de
    tamaño 1 (puntuar fila a fila).
    """)
    return


@app.cell
def _(mo):
    var_monotonia_ui = mo.ui.dropdown(
        options=["uso_linea_prom_12m", "uso_tc_prom_3m", "antiguedad_meses", "carga_financiera",
                 "consultas_6m", "renta_mm"],
        value="antiguedad_meses", label="Variable para la propiedad de monotonía")
    var_monotonia_ui
    return (var_monotonia_ui,)


@app.cell
def _(art_v100, dev, fmt_num, mo, np, pd, puntuar, ttd, var_monotonia_ui):
    def prop_monotonia(art, variable, base, ref, n_pares=400, semilla=5):
        """Para filas al azar y pares x1 < x2 (de la distribución de DEV), el score debe moverse en el
        sentido declarado. Devuelve (violaciones, contraejemplo con el par más cercano)."""
        _rng = np.random.default_rng(semilla)
        _var = next(_v for _v in art["variables"] if _v["nombre"] == variable)
        _sentido = {"creciente": 1, "decreciente": -1}[_var["tendencia_declarada"]]
        _vals = ref[variable].dropna().to_numpy()
        _filas = base.iloc[_rng.integers(0, len(base), n_pares)].copy()
        _x = np.sort(_rng.choice(_vals, (n_pares, 2)), axis=1)
        _l1, _l2 = _filas.copy(), _filas.copy()
        _l1[variable], _l2[variable] = _x[:, 0], _x[:, 1]
        _l1.index = _l2.index = [f"P{_i}" for _i in range(n_pares)]
        _ds = puntuar(_l2, art, razones=False)["score"].to_numpy() - puntuar(_l1, art, razones=False)["score"].to_numpy()
        _viol = (_sentido * _ds) < -1e-9
        _cx = None
        if _viol.any():
            _i = np.flatnonzero(_viol)[np.argmin((_x[_viol, 1] - _x[_viol, 0]))]
            _cx = {"x1": _x[_i, 0], "x2": _x[_i, 1], "Δscore": _ds[_i]}
        return int(_viol.sum()), n_pares, _cx

    resultado_monotonia = prop_monotonia(art_v100, var_monotonia_ui.value, ttd, dev)
    _nv, _np_, _cx = resultado_monotonia
    _var = next(_v for _v in art_v100["variables"] if _v["nombre"] == var_monotonia_ui.value)
    _woes = [round(_b["woe"], 3) for _b in _var["bins"] if _b["tipo"] == "intervalo"]
    mo.md(f"""
    **Monotonía de `{var_monotonia_ui.value}`** (tendencia declarada: *{_var['tendencia_declarada']}*; WoE por
    intervalo en DEV: {_woes}): {_nv} violaciones en {_np_} pares.
    {f"Contraejemplo mínimo: x₁ = {fmt_num(_cx['x1'], 4)} → x₂ = {fmt_num(_cx['x2'], 4)} y el score se mueve "
     f"{fmt_num(_cx['Δscore'], 3)} puntos en el sentido contrario al declarado." if _cx else "La propiedad se cumple."}

    En `antiguedad_meses` la violación es de centésimas de punto (el WoE baja 0,002 entre los bins (28, 46] y
    (46, 67]): irrelevante para el riesgo, **relevante para el comité**, que aprobó «más antigüedad, más puntos»
    y ante un reclamo tendría que explicar por qué 53 meses puntúa menos que 46. El test no juzga si importa; obliga a que alguien lo decida por escrito (fusionar bins
    o documentar la excepción).
    """)
    return prop_monotonia, resultado_monotonia


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Golden files y versionado semántico derivado de la prueba

    El *golden file* es un lote congelado (casos borde + **un caso por cada bin de cada variable** + muestra
    aleatoria) con la salida exacta de la versión aprobada. En un repositorio es un CSV/Parquet versionado junto
    al artefacto; aquí se materializa en memoria.

    Convención de la serie (convención, no ley): la **salida observable** decide el incremento de versión.

    | Resultado en golden + lote de referencia | Incremento | Ejemplo |
    |---|---|---|
    | salida idéntica (score, PD, banda, decisión, reason codes) | PATCH | textos, descripción |
    | score idéntico; bandas/decisiones distintas | MINOR | cutoff, master scale |
    | score trasladado por una constante, ranking idéntico | MINOR | recalibración de δ |
    | cualquier otra cosa | MAJOR | variables, cortes, WoE, β, escala |
    """)
    return


@app.cell
def _(art_v100, asignar_bins, casos_borde, np, pd, ttd):
    def construir_golden(art, lote_base, bordes, n_azar=150, semilla=21):
        """Casos borde + 1 caso real por (variable, bin) + muestra aleatoria. Mide la cobertura de bins."""
        _rng = np.random.default_rng(semilla)
        _B, _, _ = asignar_bins(lote_base, art)
        _elegidos = []
        for _j, _v in enumerate(art["variables"]):
            for _b in _v["bins"]:
                _cand = np.flatnonzero(_B[:, _j] == _b["id"])
                if len(_cand):
                    _elegidos.append(int(_rng.choice(_cand)))
        _elegidos += list(_rng.choice(len(lote_base), n_azar, replace=False))
        _g = pd.concat([bordes, lote_base.iloc[sorted(set(_elegidos))]])
        _Bg, _, _ = asignar_bins(_g, art)
        _cub = {(_v["nombre"], _b["id"]) for _j, _v in enumerate(art["variables"]) for _b in _v["bins"]
                if (_Bg[:, _j] == _b["id"]).any()}
        _tot = sum(len(_v["bins"]) for _v in art["variables"])
        return _g, len(_cub) / _tot

    golden_lote, cobertura_bins = construir_golden(art_v100, ttd, casos_borde)
    return cobertura_bins, construir_golden, golden_lote


@app.cell
def _(FACTOR, TC, VARIABLES, a_woe, art_v100, beta_dev, calibrar_delta, construir_artefacto, copy, desarrollar,
      dev, logit, mapas_dev, modelo_sm, np, oot, sellar, sm, tablas_dev):
    def nueva_version(art, version, **cambios):
        """Copia + cambios + versión + re-sello. `cambios` son funciones que mutan la copia."""
        _a = copy.deepcopy(art)
        for _f in cambios.values():
            _f(_a)
        _a["modelo"]["version"] = version
        return sellar(_a)

    # v1.0.1 — PATCH: textos de reason codes y descripción
    def _textos(a):
        a["reason_codes"]["RC04"] = "Antigüedad como cliente menor a la de clientes de menor riesgo"
        a["modelo"]["descripcion"] += " · textos RC revisados por Legal"
    art_v101 = nueva_version(art_v100, "1.0.1", textos=_textos)

    # v1.1.0 — MINOR: recalibración de δ a la tasa observada en OOT (deterioro 2025)
    _lp_oot = np.asarray(logit(np.clip(modelo_sm.predict(sm.add_constant(a_woe(oot, VARIABLES, dev, mapas_dev),
                                                                            has_constant="add")), 1e-12, 1 - 1e-12)))
    tasa_oot = float(oot["malo"].mean())
    delta_oot = calibrar_delta(_lp_oot, tasa_oot)

    def _recal(a):
        a["calibracion"].update({"delta": delta_oot, "tendencia_central": tasa_oot,
                                 "fuente": "media de PD en OOT = tasa OOT (recalibración 2025)"})
    art_v110 = nueva_version(art_v100, "1.1.0", recal=_recal)

    # v1.2.0 — MINOR: cutoff de política
    def _cutoff(a):
        a["politica"]["cutoff"] = 535.0
    art_v120 = nueva_version(art_v100, "1.2.0", cutoff=_cutoff)

    # «v1.0.2» — declarado PATCH por quien lo hizo: cortes redondeados a 2 decimales «para el anexo»
    def _redondeo(a):
        for _v in a["variables"]:
            for _b in _v["bins"]:
                if _b["tipo"] == "intervalo":
                    _b["inf"] = None if _b["inf"] is None else round(_b["inf"], 2)
                    _b["sup"] = None if _b["sup"] is None else round(_b["sup"], 2)
    art_v102_falso = nueva_version(art_v100, "1.0.2", redondeo=_redondeo)

    # v2.0.0 — MAJOR: se agrega deuda_otras_prom_12m (re-desarrollo en DEV)
    VARIABLES_V2 = VARIABLES + ["deuda_otras_prom_12m"]
    _tab2, _map2, _mod2 = desarrollar(VARIABLES_V2, dev)
    _lp2 = np.asarray(logit(np.clip(_mod2.predict(sm.add_constant(a_woe(dev, VARIABLES_V2, dev, _map2),
                                                                     has_constant="add")), 1e-12, 1 - 1e-12)))
    art_v200 = construir_artefacto(VARIABLES_V2, dev, _tab2, _mod2.params, calibrar_delta(_lp2, TC), TC, "2.0.0",
                                   "Scorecard de consumo, 9 variables (agrega deuda en otras instituciones)")
    beta_v2_negativos = bool((_mod2.params.drop("const") < 0).all())
    traslado_v110 = -(delta_oot - art_v100["calibracion"]["delta"]) * FACTOR
    _ = (beta_dev, tablas_dev)
    return (VARIABLES_V2, art_v101, art_v102_falso, art_v110, art_v120, art_v200, beta_v2_negativos, delta_oot,
            nueva_version, tasa_oot, traslado_v110)


@app.cell
def _(art_v100, art_v101, art_v102_falso, art_v110, art_v120, art_v200, golden_lote, np, pd, puntuar, ttd):
    _ORDEN = {"PATCH": 0, "MINOR": 1, "MAJOR": 2}

    def rutas_distintas(a, b, prefijo=""):
        """Diff estructural: rutas de campos que cambian (ignora integridad y versión)."""
        _ign = {"integridad", "modelo.version"}
        if prefijo in _ign:
            return []
        if isinstance(a, dict) and isinstance(b, dict):
            _out = []
            for _k in sorted(set(a) | set(b)):
                _p = f"{prefijo}.{_k}" if prefijo else _k
                _out += rutas_distintas(a.get(_k), b.get(_k), _p) if _k in a and _k in b else ([_p] if _p not in _ign else [])
            return _out
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            _out = []
            for _i, (_x, _y) in enumerate(zip(a, b)):
                _out += rutas_distintas(_x, _y, f"{prefijo}[{_i}]")
            return _out
        return [] if a == b else [prefijo]

    def incremento_requerido(art_a, art_b, lote):
        """La prueba decide: compara salidas en el lote (golden + referencia)."""
        _sa, _sb = puntuar(lote, art_a), puntuar(lote, art_b)
        _cat = ["banda", "decision", "rc1", "rc2", "rc3", "n_no_vistos"]
        _score_igual = np.array_equal(_sa["score"].to_numpy(), _sb["score"].to_numpy())
        if _score_igual and _sa[_cat].equals(_sb[_cat]):
            return "PATCH", 0.0
        if _score_igual:
            return "MINOR", 0.0
        _d = _sb["score"].to_numpy() - _sa["score"].to_numpy()
        _mismo_orden = np.array_equal(np.argsort(_sa["score"].to_numpy(), kind="stable"),
                                      np.argsort(_sb["score"].to_numpy(), kind="stable"))
        if np.ptp(_d) < 1e-9 and _mismo_orden:
            return "MINOR", float(_d.mean())
        return "MAJOR", float(np.abs(_d).max())

    def declarado(v_a, v_b):
        _a, _b = [list(map(int, _v.split("."))) for _v in (v_a, v_b)]
        return "MAJOR" if _b[0] > _a[0] else "MINOR" if _b[1] > _a[1] else "PATCH"

    _lote_ref = pd.concat([golden_lote, ttd.loc[~ttd.index.isin(golden_lote.index)]])
    _filas = []
    for _art in (art_v101, art_v102_falso, art_v110, art_v120, art_v200):
        _req, _mag = incremento_requerido(art_v100, _art, _lote_ref)
        _dec = declarado(art_v100["modelo"]["version"], _art["modelo"]["version"])
        _rutas = rutas_distintas(art_v100, _art)
        _filas.append({"versión": _art["modelo"]["version"], "declarado": _dec, "requerido por la prueba": _req,
                       "Δscore (traslado o máx)": round(_mag, 4), "campos que cambian": len(_rutas),
                       "ejemplo de campo": _rutas[0] if _rutas else "—",
                       "CI": "✔ merge" if _ORDEN[_dec] >= _ORDEN[_req] else "✘ BLOQUEADO",
                       "hash": _art["integridad"]["hash"][:12]})
    tabla_semver = pd.DataFrame(_filas).set_index("versión")
    tabla_semver
    return declarado, incremento_requerido, rutas_distintas, tabla_semver


@app.cell
def _(cobertura_bins, fmt_num, fmt_pct, golden_lote, mo, tabla_semver, traslado_v110):
    mo.md(f"""
    **Lectura.** Golden de {fmt_num(len(golden_lote), 0)} casos, cobertura de bins **{fmt_pct(cobertura_bins, 0)}**
    (sin cobertura total, un cambio en el WoE de un bin raro pasaría como PATCH). La v1.1.0 traslada **todos** los
    scores exactamente {fmt_num(traslado_v110, 3)} puntos $=-\Delta\delta\cdot\text{{factor}}$: el ranking no se
    mueve, el nivel sí — MINOR. La «v1.0.2» se declaró PATCH («solo redondeé los cortes para el anexo») y la prueba
    exige MAJOR: `{tabla_semver.loc['1.0.2', 'ejemplo de campo']}` cambió y con él el orden de los clientes. CI la
    bloquea. La versión no la decide la opinión de quien hizo el cambio; la decide el diff de la salida.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### *Shadow mode*, champion/challenger y *rollback*

    La v1.1.0 (δ recalibrado) corre en **sombra**: puntúa el mismo lote que la v1.0.0, se registra, **no
    decide**. Tras el periodo de sombra se compara (swap-set, M18) y el comité promueve. El *rollback* es
    mover un puntero a un hash que ya existía; exige que los artefactos sean inmutables y que el motor
    **verifique el hash al cargar**.
    """)
    return


@app.cell
def _(art_v100, art_v110, canonico, cargar_estricto, hash_contenido, pd, puntuar, ttd):
    _champ = puntuar(ttd, art_v100, razones=False)
    _chall = puntuar(ttd, art_v110, razones=False)
    tabla_shadow = pd.crosstab(_champ["decision"].rename("champion v1.0.0 (decide)"),
                               _chall["decision"].rename("challenger v1.1.0 (sombra)"), margins=True)

    REGISTRO_ARTEFACTOS = {_a["integridad"]["hash"]: canonico(_a) for _a in (art_v100, art_v110)}
    PUNTERO_PRODUCCION = {"historial": [art_v100["integridad"]["hash"], art_v110["integridad"]["hash"]]}

    def cargar_desde_registro(registro, h):
        """Carga por hash y VERIFICA: si el contenido no calza con su hash, no se carga."""
        _a = cargar_estricto(registro[h])
        if hash_contenido(_a) != h or _a["integridad"]["hash"] != h:
            raise ValueError(f"artefacto {h[:12]}… adulterado: hash no coincide, carga rechazada")
        return _a

    _eventos = []
    _h_prod = PUNTERO_PRODUCCION["historial"][-1]
    _eventos.append({"paso": "producción actual", "versión": cargar_desde_registro(REGISTRO_ARTEFACTOS, _h_prod)["modelo"]["version"]})
    _h_rb = PUNTERO_PRODUCCION["historial"][-2]
    _eventos.append({"paso": "rollback (puntero ← hash anterior)",
                     "versión": cargar_desde_registro(REGISTRO_ARTEFACTOS, _h_rb)["modelo"]["version"]})
    _adulterado = dict(REGISTRO_ARTEFACTOS)
    _adulterado[_h_rb] = _adulterado[_h_rb].replace('"cutoff":530.0', '"cutoff":500.0')
    try:
        cargar_desde_registro(_adulterado, _h_rb)
        _r = "cargó (MAL)"
        rollback_rechaza_adulterado = False
    except ValueError as _e:
        _r = str(_e)
        rollback_rechaza_adulterado = True
    _eventos.append({"paso": "rollback a un artefacto editado en el registro", "versión": _r})
    tabla_rollback = pd.DataFrame(_eventos)
    return (PUNTERO_PRODUCCION, REGISTRO_ARTEFACTOS, cargar_desde_registro, rollback_rechaza_adulterado,
            tabla_rollback, tabla_shadow)


@app.cell
def _(fmt_num, mo, tabla_rollback, tabla_shadow, traslado_v110):
    _sw_out = int(tabla_shadow.loc["aprobar", "rechazar"]) if "rechazar" in tabla_shadow.columns else 0
    mo.vstack([
        mo.md(f"""
    **Sombra.** Con δ recalibrado a OOT, todos los scores bajan {fmt_num(-traslado_v110, 2)} puntos; **{_sw_out}**
    solicitudes que hoy se aprueban pasarían a rechazo (swap-out). Nada de eso se ejecuta hasta que el comité lo
    apruebe: la sombra produce evidencia, no decisiones.
    """),
        tabla_shadow,
        tabla_rollback,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Contrato de datos: qué bloquea y qué solo avisa

    El motor puntúa lo que le den. El contrato es lo único entre la cartera y un lote con basura. Capas:

    1. **Esquema**: columnas presentes, tipo (numérico/categórico), identidad única.
    2. **Dominio de negocio** (valores *posibles*, fijados por el dueño del dato): fila fuera de dominio →
       esa fila a revisión; si son > 1 % del lote → bloquea (es sistémico).
    3. **Rango de DEV** (valores *vistos*): % de observados fuera de rango, medido como **exceso sobre lo
       esperado sin drift** → avisa > 1 %, bloquea > 10 % (umbrales del curso).
    4. **Missing**: tope = 3·%missing DEV + 1 %; bloquea sobre min(4·tope, 25 %) (curso).
    5. **Categorías nuevas**: avisa; bloquea > 10 %. **Reglas cruzadas**: `antiguedad_meses ≤ 12·edad`.

    **La severidad la fija la magnitud, no la regla.** Dos implementaciones: numpy/pandas puro y pydantic
    (validación por fila, agregada por variable). Deben contar exactamente lo mismo.
    """)
    return


@app.cell
def _(mo):
    modo_rango_ui = mo.ui.dropdown(options={"min–máx de DEV (curso)": "minmax",
                                            "min–máx ± 10 % del rango": "margen",
                                            "cuantiles 0,5 %–99,5 % de DEV": "cuantiles"},
                                   value="cuantiles 0,5 %–99,5 % de DEV", label="Rango de DEV")
    meses_ui = mo.ui.slider(0, 60, value=36, step=6,
                            label="Meses transcurridos desde DEV (antigüedad que envejece)", show_value=True)
    mo.hstack([modo_rango_ui, meses_ui], justify="start", gap=2)
    return meses_ui, modo_rango_ui


@app.cell
def _(np, pd):
    def validar_lote(lote, contrato, modo="minmax"):
        """Contrato en numpy/pandas puro. Devuelve (hallazgos, filas_a_revision, conteos por (variable, regla)).

        modo del rango de DEV: "minmax" (curso) · "margen" (min–max ± 10 % del rango) · "cuantiles" (p0,5–p99,5).
        La severidad se mide como EXCESO sobre lo esperado sin drift: 2/(n+1) para min–max, la fracción de
        DEV fuera del rango para cuantiles."""
        _u = contrato["umbrales"]
        _h, _cont = [], {}
        _fila_mala = np.zeros(len(lote), dtype=bool)

        def _anotar(v, sev, regla, detalle, valor):
            _h.append({"variable": v, "severidad": sev, "regla": regla, "detalle": detalle, "valor": valor})

        if lote.index.has_duplicates:
            _anotar("(índice)", "🔴", "id_duplicado", "la identidad del caso no es única", int(lote.index.duplicated().sum()))
        for _v, _c in contrato["variables"].items():
            if _v not in lote.columns:
                _anotar(_v, "🔴", "columna_faltante", "la columna no viene en el lote", None)
                continue
            _s = lote[_v]
            _es_num = pd.api.types.is_numeric_dtype(_s) and not pd.api.types.is_bool_dtype(_s)
            if _es_num != (_c["tipo"] == "numerico"):
                _anotar(_v, "🔴", "tipo_incompatible", f"esperado {_c['tipo']}, llegó {_s.dtype}", str(_s.dtype))
                continue
            _pm = float(_s.isna().mean())
            if _pm > _c["pct_missing_max"]:
                _grave = _pm > min(_c["pct_missing_max"] * _u["missing_factor_bloqueo"], _u["missing_bloqueo_abs"])
                _anotar(_v, "🔴" if _grave else "🟡", "missing_excesivo",
                        f"{_pm:.1%} > tope {_c['pct_missing_max']:.1%}", _pm)
            _obs_mask = _s.notna().to_numpy()
            if _c["tipo"] == "numerico":
                _x = _s.to_numpy(dtype=float)
                _d = _c["dominio"]
                with np.errstate(invalid="ignore"):
                    _en_dom = (_x >= _d["min"]) & (_x <= _d["max"])
                    if _d.get("especiales"):
                        _en_dom |= np.isin(_x, _d["especiales"])
                    if _d.get("entero"):
                        _en_dom &= (np.mod(_x, 1) == 0) | np.isin(_x, _d.get("especiales", []))
                _fuera_dom = _obs_mask & ~_en_dom
                _cont[(_v, "fuera_dominio")] = int(_fuera_dom.sum())
                if _fuera_dom.any():
                    _p = float(_fuera_dom.mean())
                    _anotar(_v, "🔴" if _p > _u["fila_invalida_bloqueo"] else "🟡", "fuera_dominio",
                            f"{_p:.2%} de las filas fuera de [{_d['min']:g}, {_d['max']:g}]", _p)
                    _fila_mala |= _fuera_dom
                _r = _c["rango_dev"]
                if modo == "cuantiles":
                    _lo, _hi, _esp = _r["p005"], _r["p995"], _c["fuera_p005_p995_dev"]
                else:
                    _ancho = 0.10 * (_r["max"] - _r["min"]) if modo == "margen" else 0.0
                    _lo, _hi = _r["min"] - _ancho, _r["max"] + _ancho
                    _esp = 2.0 / (_c["n_dev_observados"] + 1)
                # sobre los OBSERVADOS en dominio (si ninguno lo está, sobre todos los observados)
                _obs = _x[_obs_mask & _en_dom] if (_obs_mask & _en_dom).any() else _x[_obs_mask]
                if len(_obs):
                    _pf = float(((_obs < _lo) | (_obs > _hi)).mean())
                    _exceso = _pf - _esp
                    if _exceso > _u["fuera_rango_aviso"]:
                        _anotar(_v, "🔴" if _exceso > _u["fuera_rango_bloqueo"] else "🟡", "fuera_rango_dev",
                                f"{_pf:.1%} de los observados fuera de [{_lo:.4g}, {_hi:.4g}] (esperado {_esp:.2%})", _pf)
            else:
                _vals = _s.astype(object).where(_s.notna(), None).to_numpy()
                _nueva = _obs_mask & ~np.isin(_vals.astype(str), _c["categorias"])
                _cont[(_v, "categoria_nueva")] = int(_nueva.sum())
                if _nueva.any():
                    _p = float(_nueva[_obs_mask].mean())
                    _anotar(_v, "🔴" if _p > _u["categoria_nueva_bloqueo"] else "🟡", "categoria_nueva",
                            f"{_p:.1%} en categorías nuevas: {sorted(set(_vals[_nueva].astype(str)))[:3]}", _p)
                    _fila_mala |= _nueva
        for _regla in contrato.get("reglas", []):
            if all(_c in lote.columns for _c in _regla["columnas"]):
                _viola = (lote["antiguedad_meses"] > 12 * lote["edad"]).to_numpy()
                _cont[("antiguedad_meses", _regla["id"])] = int(_viola.sum())
                if _viola.any():
                    _p = float(_viola.mean())
                    _anotar(_regla["id"], "🔴" if _p > _u["fila_invalida_bloqueo"] else "🟡", "regla_negocio",
                            f"{_p:.2%} de las filas violan {_regla['expresion']}", _p)
                    _fila_mala |= _viola
        _tab = pd.DataFrame(_h, columns=["variable", "severidad", "regla", "detalle", "valor"])
        return _tab, pd.Series(_fila_mala, index=lote.index), _cont
    return (validar_lote,)


@app.cell
def _(Optional, ValidationError, create_model, field_validator, math, pd):
    def modelo_pydantic(contrato):
        """Implementación 2: el contrato por fila como modelo pydantic (tipos + dominio + categorías)."""
        _cv = contrato["variables"]
        _campos = {_v: ((Optional[float], None) if _c["tipo"] == "numerico" else (Optional[str], None))
                   for _v, _c in _cv.items()}

        def _validar(cls, valor, info):
            _c = _cv[info.field_name]
            if valor is None:
                return valor
            if _c["tipo"] == "categorico":
                if valor not in _c["categorias"]:
                    raise ValueError("categoria_nueva")
                return valor
            if math.isnan(valor):
                return None
            _d = _c["dominio"]
            if valor in _d.get("especiales", []):
                return valor
            if not (_d["min"] <= valor <= _d["max"]) or (_d.get("entero") and valor != math.floor(valor)):
                raise ValueError("fuera_dominio")
            return valor

        return create_model("SolicitudContrato",
                            __validators__={"dominio": field_validator(*_campos.keys())(_validar)}, **_campos)

    def validar_lote_pydantic(lote, contrato):
        """Valida fila a fila y cuenta errores por (variable, regla). Mismo conteo que la versión numpy."""
        _M = modelo_pydantic(contrato)
        _cols = [_c for _c in contrato["variables"] if _c in lote.columns]
        _registros = lote[_cols].astype(object).where(lote[_cols].notna(), None).to_dict("records")
        _cont = {}
        _malas = []
        for _i, _r in zip(lote.index, _registros):
            try:
                _M(**_r)
            except ValidationError as _e:
                _malas.append(_i)
                for _er in _e.errors():
                    _clave = (_er["loc"][0], _er["msg"].replace("Value error, ", ""))
                    _cont[_clave] = _cont.get(_clave, 0) + 1
        return _cont, pd.Index(_malas)
    return modelo_pydantic, validar_lote_pydantic


@app.cell
def _(art_v100, meses_ui, modo_rango_ui, np, pd, stats, ttd, validar_lote, validar_lote_pydantic):
    contrato_v100 = art_v100["contrato_datos"]
    # TTD con la antigüedad «envejecida»: la misma cartera, meses después (mecanismo plausible del 3,1 % de Austral)
    ttd_envejecido = ttd.copy()
    ttd_envejecido["antiguedad_meses"] = ttd_envejecido["antiguedad_meses"] + meses_ui.value
    bandeja_hallazgos, _filas_malas, conteo_numpy_ttd = validar_lote(ttd_envejecido, contrato_v100, modo_rango_ui.value)
    # la misma bandeja envejecida con los tres modos, para comparar sensibilidad
    sensibilidad_rango = pd.DataFrame([
        {"modo": _mo, **{f"+{_k} meses": ", ".join(
            f"{_r.severidad} {_r.regla}" for _r in validar_lote(
                ttd.assign(antiguedad_meses=ttd["antiguedad_meses"] + _k), contrato_v100, _mo)[0]
            .query("variable == 'antiguedad_meses'").itertuples()) or "—" for _k in (0, 24, 36, 60)}}
        for _mo in ("minmax", "margen", "cuantiles")]).set_index("modo")
    conteo_pyd_ttd, _ = validar_lote_pydantic(ttd_envejecido, contrato_v100)

    # expectativa bajo intercambiabilidad: P(nueva obs fuera de [min, max] de DEV) = 2/(n+1)
    _filas = []
    for _v, _c in contrato_v100["variables"].items():
        if _c["tipo"] != "numerico":
            continue
        _n = _c["n_dev_observados"]
        _x = ttd[_v].dropna().to_numpy()
        _k = int(((_x < _c["rango_dev"]["min"]) | (_x > _c["rango_dev"]["max"])).sum())
        _p0 = 2 / (_n + 1)
        _filas.append({"variable": _v, "n DEV": _n, "esperado 2/(n+1)": _p0, "observado TTD": _k / len(_x),
                       "fuera (n)": _k, "p-valor binomial (cola sup.)": float(stats.binom.sf(_k - 1, len(_x), _p0)),
                       # exacta: dado DEV, p = F(mín) + 1 − F(máx) ~ Beta(2, n − 1) ⇒ K ~ beta-binomial
                       "p-valor beta-binomial": float(stats.betabinom.sf(_k - 1, len(_x), 2, _n - 1))})
    tabla_intercambiable = pd.DataFrame(_filas).set_index("variable")
    _ = np
    bandeja_hallazgos
    return (bandeja_hallazgos, conteo_numpy_ttd, conteo_pyd_ttd, contrato_v100, sensibilidad_rango,
            tabla_intercambiable, ttd_envejecido)


@app.cell
def _(bandeja_hallazgos, fmt_pct, meses_ui, mo, modo_rango_ui, sensibilidad_rango, tabla_intercambiable):
    _ant = bandeja_hallazgos[bandeja_hallazgos["variable"] == "antiguedad_meses"]
    _txt = (f"`antiguedad_meses` sale {_ant.iloc[0]['severidad']} con {_ant.iloc[0]['detalle']}"
            if len(_ant) else "`antiguedad_meses` no dispara nada")
    mo.vstack([
        mo.md(f"""
    **Bandeja TTD** con la antigüedad envejecida {meses_ui.value} meses y rango «{modo_rango_ui.selected_key}»:
    {_txt}. Hallazgos totales: {len(bandeja_hallazgos)}.

    La tabla de sensibilidad muestra la lección: con el **min–máx** del curso, envejecer la antigüedad 60 meses
    no dispara nada — el generador recorta la antigüedad en 360 y el máximo de DEV **es** el borde del dominio;
    un rango min–máx solo detecta roturas de **cola** (extremos nuevos). El rango por **cuantiles** detecta
    desplazamientos del **cuerpo** de la distribución, pero hay que medir contra lo esperado (≈ 1 % fuera por
    construcción), no contra cero. En Banco Austral el 3,1 % de `antiguedad_meses` fuera de [6, 344] con min–máx
    es una rotura de cola: plausiblemente clientes cuya antigüedad siguió creciendo con el calendario (hipótesis:
    verificar en los datos).

    **¿Es mucho un 0,06 %?** Bajo intercambiabilidad (sin drift), la probabilidad de que una observación nueva
    caiga fuera del [mín, máx] de $n$ observaciones de DEV es exactamente $2/(n+1)$ ≈
    {fmt_pct(tabla_intercambiable['esperado 2/(n+1)'].iloc[0], 3)} aquí. En el generador, TTD está dentro de lo
    esperado (p-valores altos); en Austral, 3,1 % con n = 3.322 (esperado 0,06 %) es ~50 veces lo esperado.
    El p-valor binomial es **aproximado y anti-conservador**: todas las observaciones nuevas comparten el mismo
    mín/máx de DEV, así que el conteo es beta-binomial (sobredispersión $1+(m-1)/(n+2)$); la columna exacta está al
    lado. El umbral de aviso del curso (1 %) es una convención razonable, no un resultado.
    """),
        sensibilidad_rango,
        tabla_intercambiable.style.format({"esperado 2/(n+1)": "{:.4%}", "observado TTD": "{:.4%}",
                                           "p-valor binomial (cola sup.)": "{:.3g}",
                                           "p-valor beta-binomial": "{:.3g}"}),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Los tres desastres que no lanzan excepción

    Sobre la bandeja TTD (sin envejecer): (1) `uso_linea_prom_12m × 1000` (cambió la unidad aguas arriba);
    (2) `renta_mm` con la mitad del feed caído (NaN); (3) `carga_financiera` entera en NaN (el *join* falló).
    Se puntúan **sin contrato** con dos motores — el ingenuo del curso (no visto → WoE 0 en silencio) y el de
    este módulo (no visto → revisar) — y luego se pasan por el contrato.
    """)
    return


@app.cell
def _(art_v100, contrato_v100, np, pd, puntuar, ttd, validar_lote, validar_lote_pydantic):
    _limpia = puntuar(ttd, art_v100, razones=False)
    _d1, _d2, _d3 = ttd.copy(), ttd.copy(), ttd.copy()
    _d1["uso_linea_prom_12m"] = _d1["uso_linea_prom_12m"] * 1000
    _d2.loc[_d2.index[: len(_d2) // 2], "renta_mm"] = np.nan
    _d3["carga_financiera"] = np.nan
    LOTES_DESASTRE = {"lote limpio": ttd, "① uso_linea × 1000": _d1, "② renta: feed a la mitad": _d2,
                      "③ carga: columna entera NaN": _d3}
    _filas = []
    conteos_desastres = {}
    for _nom, _l in LOTES_DESASTRE.items():
        _excepciones = 0
        try:
            _ing = puntuar(_l, art_v100, marcar_no_vistos=False, razones=False)
            _mar = puntuar(_l, art_v100, razones=False)
        except Exception:
            _excepciones += 1
            continue
        _h, _fm, _cn = validar_lote(_l, contrato_v100)
        _cp, _ = validar_lote_pydantic(_l, contrato_v100)
        conteos_desastres[_nom] = (_cn, _cp)
        _filas.append({
            "lote": _nom, "excepciones Python": _excepciones,
            "aprobación (motor ingenuo)": (_ing["decision"] == "aprobar").mean(),
            "% decisiones distintas vs limpio (ingenuo)": (_ing["decision"] != _limpia["decision"]).mean(),
            "% a revisión (motor con marcas)": (_mar["decision"] == "revisar").mean(),
            "contrato: 🔴": int((_h["severidad"] == "🔴").sum()), "contrato: 🟡": int((_h["severidad"] == "🟡").sum()),
            "reglas": ", ".join(sorted(set(_h["regla"]))) or "—",
            "veredicto": "BLOQUEA" if (_h["severidad"] == "🔴").any() else ("puntúa y avisa" if len(_h) else "puntúa"),
        })
    tabla_desastres = pd.DataFrame(_filas).set_index("lote")
    tabla_desastres
    return LOTES_DESASTRE, conteos_desastres, tabla_desastres


@app.cell
def _(fmt_pct, mo, tabla_desastres):
    _t = tabla_desastres
    mo.md(f"""
    **Lectura.** Cero excepciones en los tres. Sin contrato, el lote ① pasa de {fmt_pct(_t.loc['lote limpio', 'aprobación (motor ingenuo)'])}
    a {fmt_pct(_t.loc['① uso_linea × 1000', 'aprobación (motor ingenuo)'])} de aprobación (todos caen en el último
    bin de uso), el ② cambia {fmt_pct(_t.loc['② renta: feed a la mitad', '% decisiones distintas vs limpio (ingenuo)'], 2)}
    de las decisiones **en silencio en ambos motores** (el missing de renta existía en DEV: el motor le asigna el
    WoE de MISSING con toda legitimidad), y el ③ en el motor ingenuo cambia
    {fmt_pct(_t.loc['③ carga: columna entera NaN', '% decisiones distintas vs limpio (ingenuo)'], 1)} de las decisiones
    sin avisar, mientras el motor con marcas manda {fmt_pct(_t.loc['③ carga: columna entera NaN', '% a revisión (motor con marcas)'], 0)}
    a revisión (ruidoso, pero sigue siendo un lote inservible). El contrato bloquea los tres **antes** de puntuar;
    el ② solo lo puede bloquear el contrato, porque ningún motor sabe que 50 % de missing no es la cartera.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### La corrida de producción: contrato → artefacto verificado → motor

    El orden importa: se verifica el hash del artefacto, se valida el lote, y solo entonces se puntúa. Las filas
    con fallas de fila (dominio, categoría nueva, regla cruzada) van a revisión aunque el lote pase.
    """)
    return


@app.cell
def _(LOTES_DESASTRE, REGISTRO_ARTEFACTOS, art_v100, cargar_desde_registro, pd, puntuar, validar_lote):
    def corrida_produccion(lote, registro, h):
        _art = cargar_desde_registro(registro, h)
        _hall, _filas_malas, _ = validar_lote(lote, _art["contrato_datos"])
        if (_hall["severidad"] == "🔴").any():
            return {"estado": "abortada", "motivo": "; ".join(_hall.loc[_hall["severidad"] == "🔴", "regla"]),
                    "n": len(lote), "aprobadas": 0, "a revisión": 0}, None
        _s = puntuar(lote, _art)
        _s.loc[_filas_malas.to_numpy(), "decision"] = "revisar"
        return {"estado": "ok", "motivo": f"{len(_hall)} aviso(s)", "n": len(lote),
                "aprobadas": int((_s["decision"] == "aprobar").sum()),
                "a revisión": int((_s["decision"] == "revisar").sum())}, _s

    _h = art_v100["integridad"]["hash"]
    _lote_fila = LOTES_DESASTRE["lote limpio"].copy()
    _lote_fila.iloc[:5, _lote_fila.columns.get_loc("uso_linea_prom_12m")] = 7.0      # 5 filas imposibles
    _lote_fila.iloc[5:8, _lote_fila.columns.get_loc("canal")] = "marketplace"       # 3 categorías nuevas
    _res = []
    for _nom, _l in {**LOTES_DESASTRE, "limpio + 8 filas malas": _lote_fila}.items():
        _r, _ = corrida_produccion(_l, REGISTRO_ARTEFACTOS, _h)
        _res.append({"lote": _nom, **_r})
    tabla_corridas = pd.DataFrame(_res).set_index("lote")
    tabla_corridas
    return corrida_produccion, tabla_corridas


@app.cell
def _(mo):
    mo.md(r"""
    ## Checks del módulo
    Si falla uno, el notebook falla.
    """)
    return


@app.cell
def _(FACTOR, OFFSET, VARIABLES, art_v100, art_v200, aprox_flip_a, dist_pred_a, avisos_v100, beta_v2_negativos, bordes_ok,
      cobertura_bins, comparaciones_congelado, conteo_numpy_ttd, conteo_pyd_ttd, conteos_desastres, curva_bugs,
      detalle_props, hash_orden_invariante, jsonschema, ESQUEMA_ARTEFACTO, np, resultado_monotonia,
      rollback_rechaza_adulterado, roundtrip_exacto, tabla_bordes, tabla_bugs_ttd, tabla_corridas,
      tabla_desastres, tabla_paridad, tabla_semver, tabla_validacion, traslado_v110, delta_oot, var_monotonia_ui):
    # 1. constantes del curso
    assert np.isclose(FACTOR, 28.8539, atol=1e-4) and np.isclose(OFFSET, 487.1229, atol=1e-4)
    assert np.isclose(0.1115 * FACTOR, 3.217, atol=1e-3)                  # δ Austral en puntos
    # 2. congelar = misma partición; artefacto válido; JSON estándar y hash canónico
    assert all(comparaciones_congelado.values())
    jsonschema.Draft202012Validator(ESQUEMA_ARTEFACTO).validate(art_v100)
    jsonschema.Draft202012Validator(ESQUEMA_ARTEFACTO).validate(art_v200)
    assert roundtrip_exacto and hash_orden_invariante
    assert (tabla_validacion.iloc[0, 1:] == "ok").all()
    assert (tabla_validacion.iloc[1:, 1:] != "ok").any(axis=1).all(), "cada mutación la caza al menos una capa"
    assert tabla_validacion.set_index("caso").loc["WoE = NaN", "serializacion"].startswith("FALLA")
    assert any("antiguedad_meses" in _a for _a in avisos_v100)
    # 3. paridad motor vs notebook (tolerancia declarada) y bit a bit donde el camino es el mismo
    assert tabla_paridad["max|Δscore| motor vs notebook"].max() < 1e-9
    assert tabla_paridad["max|ΔPD| motor vs notebook"].max() < 1e-12
    assert tabla_paridad["bandas iguales"].all() and tabla_paridad["decisiones iguales"].all()
    assert tabla_paridad["numpy == pd.cut (bit a bit)"].all() and tabla_paridad["JSON round-trip (bit a bit)"].all()
    assert tabla_paridad["max|Σpuntos − δ·factor − score|"].max() < 1e-9
    assert tabla_paridad["índice preservado"].all() and (tabla_paridad["no vistos"] == 0).all()
    assert bordes_ok
    # 4. semántica de bordes: cerrar a la izquierda SÍ cambia decisiones; float32 no es inocuo en bins
    assert tabla_bordes.iloc[0]["% decisiones que cambian"] > 0.01
    assert tabla_bordes.iloc[0]["consultas_6m"] > 0.3
    # 5. bugs: A silencioso (0 WoE sin mapa) y B ruidoso; B hace más daño; aproximación de 1er orden razonable
    _a, _b = tabla_bugs_ttd.iloc[0], tabla_bugs_ttd.iloc[1]
    assert _a["% WoE sin mapa"] == 0 and _a["cambian decisión"] > 0
    assert _b["% WoE sin mapa"] > 0.3 and _b["% cambian"] > _a["% cambian"]
    assert _a["dist. media al cutoff (cambian)"] < _a["dist. media al cutoff (resto)"]
    assert 0.5 < aprox_flip_a / _a["% cambian"] < 2.0
    assert 0.5 < dist_pred_a / _a["dist. media al cutoff (cambian)"] < 2.0
    assert (curva_bugs["bug_a"] > 0).all()
    # 6. propiedades: el motor pasa todo; el bug A pasa permutación y FALLA subconjunto
    _p = detalle_props.set_index(["implementación", "propiedad"])["pasa"]
    assert (_p.loc["motor correcto (artefacto)"] == "✔").all()
    assert _p.loc[("bug A", "permutación")] == "✔" and _p.loc[("bug A", "subconjunto")] == "✘"
    assert _p.loc[("bug B", "subconjunto")] == "✘"
    if var_monotonia_ui.value == "antiguedad_meses":
        assert resultado_monotonia[0] > 0                               # la violación real de monotonía
    # 7. golden y semver
    assert cobertura_bins == 1.0
    _sv = tabla_semver
    assert _sv.loc["1.0.1", "requerido por la prueba"] == "PATCH"
    assert _sv.loc["1.1.0", "requerido por la prueba"] == "MINOR"
    assert np.isclose(_sv.loc["1.1.0", "Δscore (traslado o máx)"], traslado_v110, atol=1e-4)
    assert _sv.loc["1.2.0", "requerido por la prueba"] == "MINOR"
    assert _sv.loc["1.0.2", "requerido por la prueba"] == "MAJOR" and _sv.loc["1.0.2", "CI"].startswith("✘")
    assert _sv.loc["2.0.0", "requerido por la prueba"] == "MAJOR" and beta_v2_negativos
    assert delta_oot > art_v100["calibracion"]["delta"] and rollback_rechaza_adulterado
    assert len(art_v200["variables"]) == len(VARIABLES) + 1
    # 8. contrato: numpy y pydantic cuentan lo mismo; los 3 desastres bloquean; el limpio no
    def _norm(c):
        return {(_k[0], _k[1]): _v for _k, _v in c.items() if _v > 0 and _k[1] in ("fuera_dominio", "categoria_nueva")}
    assert _norm(conteo_numpy_ttd) == _norm(conteo_pyd_ttd)
    for _nom, (_cn, _cp) in conteos_desastres.items():
        assert _norm(_cn) == _norm(_cp), _nom
    assert tabla_desastres.loc["lote limpio", "veredicto"] != "BLOQUEA"
    assert (tabla_desastres.drop("lote limpio")["veredicto"] == "BLOQUEA").all()
    assert (tabla_desastres["excepciones Python"] == 0).all()
    assert tabla_desastres.loc["② renta: feed a la mitad", "% decisiones distintas vs limpio (ingenuo)"] > 0
    assert tabla_corridas.loc["limpio + 8 filas malas", "estado"] == "ok"
    assert tabla_corridas.loc["limpio + 8 filas malas", "a revisión"] >= 8
    print("✔ todos los checks del módulo M21 pasan")
    return


if __name__ == "__main__":
    app.run()

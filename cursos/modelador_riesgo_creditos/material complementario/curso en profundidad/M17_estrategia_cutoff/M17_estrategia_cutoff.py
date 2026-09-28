# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
#     "statsmodels",
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
    from scipy.optimize import brentq, minimize_scalar
    from scipy.stats import norm
    from statistics import NormalDist
    return NormalDist, brentq, minimize_scalar, mo, norm, plt, sm


@app.cell
def _(mo):
    mo.md(r"""
    # M17 · Estrategia: cutoff, pérdida esperada y rentabilidad

    Notebook del módulo M17 de la Serie 2. Recorre, con **verdad conocida** del generador:

    1. Pipeline mínimo: scorecard de 6 variables, calibración a la tendencia central (TC) y score en
       la escala del curso (PDO 20, 600 @ 50:1).
    2. Pérdida esperada (EL) y **tabla de estrategia** sobre OOT, con controles de LGD, tasa, costo de
       fondos, costos y desplazamiento de la TC.
    3. **Frontera** aprobación–mora y aprobación–utilidad, dominancia, DEV vs OOT, política de
       referencia por *knock-outs*.
    4. **Cutoff de breakeven**: fórmula analítica vs bisección numpy vs `scipy.optimize` (deben coincidir);
       apetito y RAROC como restricciones adicionales.
    5. Cuándo falla: **descalibración** ⇒ el corte se corre $\text{factor}\cdot\Delta$ puntos.
    6. **Cutoff por segmento**: cuándo gana, cuánto gana y cuándo el ruido se come la ganancia.
    7. **Caso motos** (parámetros genéricos e ilustrativos): cronograma, valor de la moto, LGD por mes de
       default, PD de vida, EL y utilidad por banda, pricing con tope TMC, pie mínimo.
    8. **Tornado** de sensibilidad del cutoff de breakeven.

    Convenciones del curso: target 1 = malo; WoE = ln(%buenos/%malos); todo lo que se ajusta se ajusta en
    DEV (o en la muestra de calibración declarada) y se aplica al resto; la estrategia se mide en **OOT**.
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
    ## 1. Pipeline mínimo: scorecard, calibración a la TC y score

    Igual que M15: 6 variables, binning y WoE del curso, logística sobre WoE en DEV. La PD se calibra a la
    **tendencia central** (promedio simple de la tasa de malos de las 24 cosechas con desempeño) con el
    **δ exacto** (bisección numpy vs `brentq`). El score calibrado es

    $$s=\text{offset}+\text{factor}\cdot\ln\frac{1-p}{p},\qquad \text{factor}=\frac{20}{\ln 2}=28{,}8539,\quad
    \text{offset}=600-\text{factor}\cdot\ln 50=487{,}1229.$$

    Agregamos tres columnas propias de este módulo:

    - `monto_mm`: monto solicitado (MM\$), log-normal y creciente en la renta (generador propio, semilla 1717).
      Es la EAD del modelo simple del curso («EAD = monto solicitado»).
    - `pd_real`: la PD verdadera **completa** (incluye el 10% de malos extra que el generador planta en «sin
      bureau», −99).
    - `score` y `pd_cal`: score y PD calibrados a la TC.

    Qué mirar: en OOT la tasa observada y la PD real media están **por encima** de la PD calibrada. Es el
    `deterioro=0.35` plantado en 2025. La TC mira el ciclo; OOT es el presente. Eso va a importar en §5.
    """)
    return


@app.cell
def _(a_woe, brentq, generar_cartera, np, pd, sm, tabla_woe):
    FACTOR = 20 / np.log(2)
    OFFSET = 600 - FACTOR * np.log(50)

    def sigmoide_np(z):
        """σ(z) estable para |z| grande."""
        z = np.asarray(z, dtype=float)
        _ez = np.exp(-np.abs(z))
        return np.where(z >= 0, 1.0 / (1.0 + _ez), _ez / (1.0 + _ez))

    def logit_np(p):
        p = np.asarray(p, dtype=float)
        return np.log(p / (1.0 - p))

    def pd_de_score(s):
        """PD implícita por la escala: s = offset + factor·ln((1-p)/p)."""
        return sigmoide_np(-(np.asarray(s, dtype=float) - OFFSET) / FACTOR)

    def score_de_pd(p):
        return OFFSET + FACTOR * np.log((1.0 - np.asarray(p, dtype=float)) / np.asarray(p, dtype=float))

    def biseccion(f, lo, hi, tol=1e-12, max_iter=300):
        """Raíz de f en [lo, hi] por bisección (numpy puro). Exige cambio de signo."""
        f_lo = f(lo)
        assert f_lo * f(hi) <= 0, "sin cambio de signo en el intervalo"
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    cartera = generar_cartera()
    cartera["pd_real"] = np.where(
        cartera["meses_desde_mora_12m"] == -99,
        cartera["pd_verdadera"] + (1 - cartera["pd_verdadera"]) * 0.10,
        cartera["pd_verdadera"],
    )
    _rng = np.random.default_rng(1717)
    _renta = cartera["renta_mm"].fillna(cartera["renta_mm"].median()).values
    cartera["monto_mm"] = np.round(np.clip(np.exp(
        np.log(2.0) + 0.6 * (np.log(_renta) - np.log(0.8)) + _rng.normal(0, 0.45, len(cartera))), 0.3, 15), 3)

    VARIABLES = ["uso_linea_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m",
                 "antiguedad_meses", "carga_financiera", "consultas_6m"]
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    _mapas = {v: tabla_woe(dev[v], dev["malo"])[0]["woe"].to_dict() for v in VARIABLES}
    modelo = sm.Logit(dev["malo"].values, sm.add_constant(a_woe(dev, VARIABLES, dev, _mapas))).fit(disp=0)

    def _pd_modelo(df_):
        _X = sm.add_constant(a_woe(df_, VARIABLES, dev, _mapas), has_constant="add")
        return np.asarray(modelo.predict(_X), dtype=float)

    TC = float(cartera[cartera["muestra"] != "TTD"].groupby("cohorte")["malo"].mean().mean())
    _lp_dev = logit_np(_pd_modelo(dev))
    DELTA_TC = biseccion(lambda d: sigmoide_np(_lp_dev + d).mean() - TC, -5.0, 5.0)
    delta_tc_brentq = brentq(lambda d: sigmoide_np(_lp_dev + d).mean() - TC, -5.0, 5.0, xtol=1e-14)

    muestras = {}
    for _m in ["DEV", "HO", "OOT", "TTD"]:
        _d = cartera[cartera["muestra"] == _m].reset_index(drop=True).copy()
        _d["lp"] = logit_np(_pd_modelo(_d)) + DELTA_TC          # logit calibrado a la TC
        _d["pd_cal"] = sigmoide_np(_d["lp"].values)
        _d["score"] = OFFSET - FACTOR * _d["lp"].values
        muestras[_m] = _d

    tabla_nivel = pd.DataFrame({
        _m: {"n": len(_d), "tasa_observada": _d["malo"].mean(), "pd_cal_media": _d["pd_cal"].mean(),
             "pd_real_media": _d["pd_real"].mean(), "monto_medio_mm": _d["monto_mm"].mean(),
             "score_p10": _d["score"].quantile(0.10), "score_p50": _d["score"].median(),
             "score_p90": _d["score"].quantile(0.90)}
        for _m, _d in muestras.items()}).T
    tabla_nivel.round(4)
    return (
        DELTA_TC,
        FACTOR,
        OFFSET,
        TC,
        biseccion,
        cartera,
        delta_tc_brentq,
        logit_np,
        muestras,
        pd_de_score,
        score_de_pd,
        sigmoide_np,
        tabla_nivel,
    )


@app.cell
def _(DELTA_TC, FACTOR, TC, mo, tabla_nivel):
    _t = tabla_nivel
    mo.md(rf"""
    **Lectura.** TC = {TC:.2%}; δ exacto = {DELTA_TC:+.4f} (todos los clientes bajan
    {DELTA_TC * FACTOR:.2f} puntos). En OOT la tasa observada es {_t.loc['OOT', 'tasa_observada']:.2%} y la PD real
    media {_t.loc['OOT', 'pd_real_media']:.2%}, contra una PD calibrada media de {_t.loc['OOT', 'pd_cal_media']:.2%}.
    Banco Sintético tiene más del doble de riesgo que Banco Austral (TC 5,42%) y su score vive más abajo:
    mediana OOT {_t.loc['OOT', 'score_p50']:.0f} puntos. Por eso los cortes de este notebook están entre 480 y 600,
    no entre 500 y 640 como en la clase.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Pérdida esperada y tabla de estrategia (OOT)

    Modelo económico **de un período** (el del curso, completado con márgenes y costos), por peso de monto:

    - si el cliente es bueno: gana $g=r-k$, con $r$ la tasa anual y $k=c_f+c_{op}+c_{adq}$ (costo de fondos,
      operativo y de adquisición, en % del monto);
    - si es malo: pierde $l=\text{LGD}+k$ (no cobra intereses, recupera $1-\text{LGD}$ del capital y pagó igual
      los costos).

    Utilidad esperada del crédito $i$: $u_i=m_i[(1-p_i)g-p_i l]=\underbrace{m_i[(1-p_i)r-k]}_{\text{margen}}-
    \underbrace{p_i\,\text{LGD}\,m_i}_{\text{EL}}$. La tabla agrega, para cada cutoff, la aprobación, la mora
    **observada** de aprobados, la PD calibrada media, el monto, la EL, el margen y la utilidad (esperada con
    $p_i$ y **realizada** ex post con el malo observado).

    El control «desplazamiento de TC» suma $\Delta\delta$ al logit de todos: simula calibrar a otra
    tendencia central (p. ej. PIT). Con el cutoff escrito en **score del modelo**, la aprobación no cambia pero sí la EL
    y la utilidad que se prometen.

    Dos implementaciones: (1) numpy con orden + sumas acumuladas (una pasada, O(n log n)); (2) pandas con
    máscaras por cutoff. Deben coincidir.
    """)
    return


@app.cell
def _(mo):
    ctl_lgd = mo.ui.slider(0.20, 0.80, step=0.05, value=0.45, label="LGD")
    ctl_tasa = mo.ui.slider(0.10, 0.40, step=0.01, value=0.24, label="Tasa anual r")
    ctl_cf = mo.ui.slider(0.02, 0.15, step=0.005, value=0.06, label="Costo de fondos anual c_f")
    ctl_costos = mo.ui.slider(0.00, 0.12, step=0.005, value=0.06, label="Costos operativos + adquisición (% monto)")
    ctl_delta_extra = mo.ui.slider(-0.50, 0.80, step=0.05, value=0.0, label="Desplazamiento de TC Δδ (log-odds)")
    mo.vstack([mo.hstack([ctl_lgd, ctl_tasa]), mo.hstack([ctl_cf, ctl_costos]), ctl_delta_extra])
    return ctl_cf, ctl_costos, ctl_delta_extra, ctl_lgd, ctl_tasa


@app.cell
def _(np, pd):
    def tabla_estrategia_np(score, pd_, malo, monto, cortes, lgd, tasa, k):
        """Tabla de estrategia en numpy: ordenar una vez y usar sumas acumuladas por la cola."""
        _o = np.argsort(score, kind="stable")
        _s = score[_o]
        _cols = {
            "n": np.ones_like(_s), "malos": malo[_o], "pd": pd_[_o], "monto": monto[_o],
            "el": pd_[_o] * lgd * monto[_o],
            "margen": monto[_o] * ((1 - pd_[_o]) * tasa - k),
            "u_real": monto[_o] * ((1 - malo[_o]) * tasa - k - malo[_o] * lgd),
        }
        _pre = {c: np.r_[0.0, np.cumsum(v)] for c, v in _cols.items()}
        _idx = np.searchsorted(_s, np.asarray(cortes, float), side="left")   # primer score >= corte
        _sum = {c: v[-1] - v[_idx] for c, v in _pre.items()}                  # suma de la cola aprobada
        _n = _sum["n"]
        _seguro = np.maximum(_n, 1)
        _t = pd.DataFrame({
            "cutoff": np.asarray(cortes, float),
            "aprobacion": _n / len(score),
            "mora_obs": np.where(_n > 0, _sum["malos"] / _seguro, np.nan),
            "pd_cal_media": np.where(_n > 0, _sum["pd"] / _seguro, np.nan),
            "monto_mm": _sum["monto"],
            "el_mm": _sum["el"],
            "el_sobre_monto": np.where(_sum["monto"] > 0, _sum["el"] / np.maximum(_sum["monto"], 1e-12), np.nan),
            "margen_mm": _sum["margen"],
            "utilidad_mm": _sum["margen"] - _sum["el"],
            "utilidad_real_mm": _sum["u_real"],
        })
        return _t.set_index("cutoff")

    def tabla_estrategia_pd(df_, cortes, lgd, tasa, k, col_pd="pd_uso", col_score="score"):
        """Misma tabla con pandas: una máscara por cutoff (legible, O(n·cortes))."""
        _filas = []
        for _c in cortes:
            _a = df_[df_[col_score] >= _c]
            _el = (_a[col_pd] * lgd * _a["monto_mm"]).sum()
            _mg = (_a["monto_mm"] * ((1 - _a[col_pd]) * tasa - k)).sum()
            _ur = (_a["monto_mm"] * ((1 - _a["malo"]) * tasa - k - _a["malo"] * lgd)).sum()
            _filas.append({"cutoff": float(_c), "aprobacion": len(_a) / len(df_),
                           "mora_obs": _a["malo"].mean() if len(_a) else np.nan,
                           "pd_cal_media": _a[col_pd].mean() if len(_a) else np.nan,
                           "monto_mm": _a["monto_mm"].sum(), "el_mm": _el,
                           "el_sobre_monto": _el / _a["monto_mm"].sum() if len(_a) else np.nan,
                           "margen_mm": _mg, "utilidad_mm": _mg - _el, "utilidad_real_mm": _ur})
        return pd.DataFrame(_filas).set_index("cutoff")
    return tabla_estrategia_np, tabla_estrategia_pd


@app.cell
def _(
    FACTOR,
    OFFSET,
    ctl_cf,
    ctl_costos,
    ctl_delta_extra,
    ctl_lgd,
    ctl_tasa,
    muestras,
    np,
    sigmoide_np,
    tabla_estrategia_np,
    tabla_estrategia_pd,
):
    economia = {"lgd": ctl_lgd.value, "tasa": ctl_tasa.value,
                "k": ctl_cf.value + ctl_costos.value, "delta_extra": ctl_delta_extra.value}
    economia["g"] = economia["tasa"] - economia["k"]
    economia["l"] = economia["lgd"] + economia["k"]

    def preparar(df_, delta_extra):
        """Copia de la muestra con la PD en uso (TC + Δδ) y el score correspondiente."""
        _d = df_.copy()
        _d["lp_uso"] = _d["lp"] + delta_extra
        _d["pd_uso"] = sigmoide_np(_d["lp_uso"].values)
        _d["score_uso"] = OFFSET - FACTOR * _d["lp_uso"].values
        return _d

    oot = preparar(muestras["OOT"], economia["delta_extra"])
    CORTES_TABLA = np.arange(480, 601, 10)
    tabla_estr = tabla_estrategia_np(oot["score"].values, oot["pd_uso"].values, oot["malo"].values,
                                     oot["monto_mm"].values, CORTES_TABLA,
                                     economia["lgd"], economia["tasa"], economia["k"])
    tabla_estr_pandas = tabla_estrategia_pd(oot, CORTES_TABLA, economia["lgd"], economia["tasa"], economia["k"])
    estrategia_coincide = bool(np.allclose(tabla_estr.values, tabla_estr_pandas.values, equal_nan=True))
    assert estrategia_coincide
    tabla_estr.style.format({
        "aprobacion": "{:.1%}", "mora_obs": "{:.2%}", "pd_cal_media": "{:.2%}", "monto_mm": "{:,.0f}",
        "el_mm": "{:,.1f}", "el_sobre_monto": "{:.2%}", "margen_mm": "{:,.1f}", "utilidad_mm": "{:,.1f}",
        "utilidad_real_mm": "{:,.1f}"})
    return CORTES_TABLA, economia, estrategia_coincide, oot, preparar, tabla_estr


@app.cell
def _(economia, estrategia_coincide, mo, tabla_estr):
    _t = tabla_estr
    _best = _t["utilidad_mm"].idxmax()
    _best_r = _t["utilidad_real_mm"].idxmax()
    mo.md(rf"""
    **Lectura** (numpy = pandas: {estrategia_coincide}). Con LGD {economia['lgd']:.0%}, tasa {economia['tasa']:.0%} y
    costos totales {economia['k']:.1%}, en la grilla de 10 puntos la utilidad **esperada** es máxima en el corte
    **{_best:.0f}** y la **realizada** (con el malo observado en OOT) en **{_best_r:.0f}**. Tres observaciones:

    - La mora observada de aprobados supera a la PD calibrada media en todos los cortes. La PD está anclada
      al ciclo (TC) y OOT vive en el deterioro de 2025. La EL de la tabla usa la PD calibrada, así que
      **subestima** la pérdida de este periodo. El curso tuvo lo mismo en Austral (560: mora 2,10% vs PD 1,59%).
    - La utilidad no es monótona: sube mientras los clientes que se agregan tienen $u_i>0$ y baja después.
      Su máximo es el **cutoff de breakeven** (§4), no el corte que cumple el apetito.
    - La EL/monto cae rápido al subir el corte, pero la utilidad cae también: menos riesgo no es más plata.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Frontera de estrategia, dominancia y la política de referencia

    Cada cutoff es un punto (aprobación, mora). La curva que forman es la **frontera alcanzable** con este
    scorecard. La política de referencia del curso eran solo *knock-outs*. Aquí usamos «rechazar si hubo mora en los
    últimos 6 meses» (`meses_desde_mora_12m` entre 1 y 6), que en OOT aprueba ≈ 90% (el curso tenía 90,2%).
    Una política aleatoria que aprueba una fracción $a$ tiene mora constante igual a la tasa de la muestra: es la
    frontera de un modelo sin poder.

    Qué mirar:

    - **Distancia vertical** del rombo a la curva, a igual aprobación: el valor del modelo en puntos de mora.
    - **DEV vs OOT**: la curva de DEV queda más abajo (optimista). Por eso la tabla se construye en OOT.
    - En el panel de utilidad, todo corte **más laxo que el breakeven** queda dominado: más mora y menos
      utilidad que el propio breakeven. Solo se justifica por volumen, y eso se escribe en el acta.
    """)
    return


@app.cell
def _(mo):
    ctl_apetito = mo.ui.slider(0.04, 0.16, step=0.005, value=0.08, label="Apetito: mora máxima observada de aprobados")
    ctl_apetito
    return (ctl_apetito,)


@app.cell
def _(
    OFFSET,
    FACTOR,
    ctl_apetito,
    economia,
    muestras,
    np,
    oot,
    plt,
    preparar,
    tabla_estrategia_np,
):
    _dev = preparar(muestras["DEV"], economia["delta_extra"])
    _grilla = np.quantile(oot["score"], np.linspace(0.0, 0.97, 120))
    frontera_oot = tabla_estrategia_np(oot["score"].values, oot["pd_uso"].values, oot["malo"].values,
                                       oot["monto_mm"].values, _grilla, economia["lgd"], economia["tasa"], economia["k"])
    frontera_dev = tabla_estrategia_np(_dev["score"].values, _dev["pd_uso"].values, _dev["malo"].values,
                                       _dev["monto_mm"].values, _grilla, economia["lgd"], economia["tasa"], economia["k"])
    _ko = oot["meses_desde_mora_12m"].between(1, 6).values
    politica_ko = {"aprobacion": float((~_ko).mean()), "mora": float(oot["malo"].values[~_ko].mean())}
    # scorecard a igual aprobación que los knock-outs (iso-aprobación)
    _c_iso = float(np.quantile(oot["score"], 1 - politica_ko["aprobacion"]))
    _ap_iso = oot["score"].values >= _c_iso
    politica_ko["cutoff_iso"] = _c_iso
    politica_ko["mora_scorecard_iso"] = float(oot["malo"].values[_ap_iso].mean())

    # corte por apetito: el más laxo cuya mora observada de aprobados cumple
    _cumple = frontera_oot[frontera_oot["mora_obs"] <= ctl_apetito.value]
    corte_apetito = float(_cumple.index.min()) if len(_cumple) else np.nan

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4))
    _ax[0].plot(100 * frontera_oot["aprobacion"], 100 * frontera_oot["mora_obs"], "-", color="#1f4e79", label="scorecard · OOT")
    _ax[0].plot(100 * frontera_dev["aprobacion"], 100 * frontera_dev["mora_obs"], "--", color="0.5", label="scorecard · DEV")
    _ax[0].axhline(100 * oot["malo"].mean(), color="0.75", lw=1, ls=":", label="política aleatoria (OOT)")
    _ax[0].plot(100 * politica_ko["aprobacion"], 100 * politica_ko["mora"], "D", color="#b3261e", ms=8, label="knock-outs (referencia)")
    _ax[0].axhline(100 * ctl_apetito.value, color="#ef6c00", lw=1, ls="-.", label=f"apetito {ctl_apetito.value:.1%}")
    _ax[0].set_xlabel("aprobación (%)")
    _ax[0].set_ylabel("mora observada de aprobados (%)")
    _ax[0].set_title("Frontera aprobación–mora")
    _ax[0].legend(fontsize=7)
    _ax[1].plot(100 * frontera_oot["aprobacion"], frontera_oot["utilidad_mm"], "-", color="#1f4e79", label="esperada (PD en uso)")
    _ax[1].plot(100 * frontera_oot["aprobacion"], frontera_oot["utilidad_real_mm"], "-", color="#2e7d32", label="realizada (malo observado)")
    _ax[1].plot(100 * frontera_dev["aprobacion"], frontera_dev["utilidad_real_mm"] * len(oot) / len(_dev), "--", color="0.5",
                label="realizada DEV (reescalada a n OOT)")
    _s_be = OFFSET + FACTOR * np.log(economia["l"] / economia["g"]) if economia["g"] > 0 else np.nan
    _ap_be = float((oot["score"].values >= _s_be).mean()) if np.isfinite(_s_be) else np.nan
    if np.isfinite(_ap_be):
        _ax[1].axvline(100 * _ap_be, color="#b3261e", ls=":", lw=1, label="breakeven analítico")
    _ax[1].set_xlabel("aprobación (%)")
    _ax[1].set_ylabel("utilidad (MM$)")
    _ax[1].set_title("Frontera aprobación–utilidad")
    _ax[1].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return corte_apetito, frontera_dev, frontera_oot, politica_ko


@app.cell
def _(corte_apetito, ctl_apetito, frontera_oot, mo, np, politica_ko):
    _p = politica_ko
    _txt_ap = (f"el corte más laxo que cumple mora ≤ {ctl_apetito.value:.1%} es **{corte_apetito:.1f}** "
               f"(aprobación {float(frontera_oot.loc[corte_apetito, 'aprobacion']):.1%})"
               if np.isfinite(corte_apetito) else "ningún corte de la grilla cumple ese apetito")
    mo.md(rf"""
    **Lectura.** Los knock-outs aprueban {_p['aprobacion']:.1%} con mora {_p['mora']:.2%}. El scorecard, a **igual
    aprobación** (corte {_p['cutoff_iso']:.1f}), logra {_p['mora_scorecard_iso']:.2%}: una reducción de
    {(1 - _p['mora_scorecard_iso'] / _p['mora']):.0%} de la mora. En el curso fue 4,48% → 3,48% (−22%). Con el apetito del slider,
    {_txt_ap}. La curva de DEV queda por debajo de OOT en prácticamente todo el rango: el mismo corte medido en DEV promete menos
    mora de la que se va a ver, porque DEV es la muestra de ajuste y además es previa al deterioro de 2025.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Cutoff de breakeven: analítico vs numérico

    Aprobar al cliente marginal conviene si $(1-p)g-p\,l>0\iff \frac{1-p}{p}>\frac{l}{g}$. En la escala del curso:

    $$s_{BE}=\text{offset}+\text{factor}\cdot\ln\frac{l}{g}=\text{offset}+\text{factor}\cdot
    \ln\frac{\text{LGD}+k}{r-k}.$$

    Si el monto $m_i>0$, el signo de $u_i$ depende **solo** de si $s_i\gtrless s_{BE}$. La utilidad total
    $U(c)=\sum_{s_i\ge c}u_i$ es máxima justo en $s_{BE}$ (derivación en el `.md`, §3.3). Cuatro caminos:

    1. **Analítico**: la fórmula.
    2. **Bisección numpy** sobre la utilidad marginal $\pi(s)=(1-p(s))g-p(s)l$, con $p(s)$ de la escala.
    3. **`scipy.optimize.brentq`** sobre la misma $\pi(s)$.
    4. **`scipy.optimize.minimize_scalar`** sobre $-U_\tau(c)$, con $U_\tau(c)=\sum_i\sigma\big((s_i-c)/\tau\big)u_i$
       (indicador suavizado, τ = 0,25 pts), sobre los datos de OOT. Además, el **argmax exacto** de $U(c)$ en la
       grilla de scores observados.

    Agregamos dos reglas que el breakeven no ve: el **apetito** (mora máxima) y un **RAROC** mínimo con capital
    IRB de minorista (solo como referencia, ver Serie 1 · E6): aprobar si $u(s)\ge h\cdot K(p(s))$, con
    $K(p)=\text{LGD}\,[\Phi(\frac{\Phi^{-1}(p)+\sqrt{R}\,\Phi^{-1}(0{,}999)}{\sqrt{1-R}})-p]$, $R$ de «otras
    exposiciones minoristas». $\Phi$ se calcula con `scipy.stats.norm` y con `statistics.NormalDist`.
    """)
    return


@app.cell
def _(
    FACTOR,
    NormalDist,
    OFFSET,
    biseccion,
    brentq,
    corte_apetito,
    economia,
    minimize_scalar,
    norm,
    np,
    oot,
    pd,
    pd_de_score,
    sigmoide_np,
    tabla_estrategia_np,
):
    _g, _l = economia["g"], economia["l"]
    assert _g > 0, "con r ≤ k ningún cliente es rentable: el cutoff es +∞"
    s_be_analitico = OFFSET + FACTOR * np.log(_l / _g)

    def utilidad_marginal(s):
        _p = pd_de_score(s)
        return (1 - _p) * _g - _p * _l

    s_be_biseccion = biseccion(utilidad_marginal, 300.0, 900.0)
    s_be_brentq = brentq(utilidad_marginal, 300.0, 900.0, xtol=1e-12)

    _s = oot["score"].values
    _m = oot["monto_mm"].values
    _p = oot["pd_uso"].values
    u_i = _m * ((1 - _p) * _g - _p * _l)

    def _menos_u_suave(c, tau=0.25):
        return -np.sum(sigmoide_np((_s - c) / tau) * u_i)

    _res = minimize_scalar(_menos_u_suave, bounds=(float(_s.min()), float(_s.max())), method="bounded",
                           options={"xatol": 1e-6})
    s_be_scipy_opt = float(_res.x)
    # argmax exacto de U(c) sobre los scores observados (c = score del último aprobado)
    _o = np.argsort(_s)
    _cola = np.cumsum(u_i[_o][::-1])[::-1]          # U si el corte es s_(j)
    _j = int(np.argmax(_cola))
    s_be_grilla = float(_s[_o][_j])
    brecha_grilla = float(_s[_o][_j] - _s[_o][_j - 1]) if _j > 0 else np.inf

    # RAROC: capital IRB «otras minoristas» con dos implementaciones de Φ
    def correlacion_minorista(p):
        _f = (1 - np.exp(-35 * p)) / (1 - np.exp(-35))
        return 0.03 * _f + 0.16 * (1 - _f)

    def k_irb_scipy(p, lgd):
        _R = correlacion_minorista(p)
        return lgd * (norm.cdf((norm.ppf(p) + np.sqrt(_R) * norm.ppf(0.999)) / np.sqrt(1 - _R)) - p)

    _nd = NormalDist()

    def k_irb_stdlib(p, lgd):
        _R = correlacion_minorista(p)
        return lgd * (_nd.cdf((_nd.inv_cdf(p) + np.sqrt(_R) * _nd.inv_cdf(0.999)) / np.sqrt(1 - _R)) - p)

    _pp = np.array([0.005, 0.02, 0.05, 0.10, 0.20, 0.40])
    irb_coincide = bool(np.allclose(k_irb_scipy(_pp, 0.45), [k_irb_stdlib(float(x), 0.45) for x in _pp], atol=1e-10))
    HURDLE = 0.15
    s_raroc = brentq(lambda s: utilidad_marginal(s) - HURDLE * k_irb_scipy(pd_de_score(s), economia["lgd"]),
                     300.0, 900.0, xtol=1e-10)

    _cortes = [s_be_analitico, s_raroc] + ([corte_apetito] if np.isfinite(corte_apetito) else [])
    _t = tabla_estrategia_np(_s, _p, oot["malo"].values, _m, _cortes, economia["lgd"], economia["tasa"], economia["k"])
    _t.index = ["breakeven (u ≥ 0)", f"RAROC ≥ {HURDLE:.0%}"] + (["apetito de mora"] if np.isfinite(corte_apetito) else [])
    tabla_reglas = _t[["aprobacion", "mora_obs", "pd_cal_media", "el_sobre_monto", "utilidad_mm", "utilidad_real_mm"]]
    tabla_reglas.insert(0, "cutoff", _cortes)

    comparacion_be = pd.DataFrame({
        "método": ["analítico", "bisección numpy", "scipy brentq", "scipy minimize_scalar (U suave)", "argmax exacto en datos"],
        "cutoff": [s_be_analitico, s_be_biseccion, s_be_brentq, s_be_scipy_opt, s_be_grilla]})
    comparacion_be["diferencia_vs_analitico"] = comparacion_be["cutoff"] - s_be_analitico
    comparacion_be
    return (
        HURDLE,
        brecha_grilla,
        comparacion_be,
        irb_coincide,
        k_irb_scipy,
        s_be_analitico,
        s_be_biseccion,
        s_be_brentq,
        s_be_grilla,
        s_be_scipy_opt,
        s_raroc,
        tabla_reglas,
        u_i,
        utilidad_marginal,
    )


@app.cell
def _(tabla_reglas):
    tabla_reglas.style.format({"cutoff": "{:.1f}", "aprobacion": "{:.1%}", "mora_obs": "{:.2%}",
                               "pd_cal_media": "{:.2%}", "el_sobre_monto": "{:.2%}",
                               "utilidad_mm": "{:,.1f}", "utilidad_real_mm": "{:,.1f}"})
    return


@app.cell
def _(economia, np, oot, plt, s_be_analitico, s_raroc, u_i):
    _s = oot["score"].values
    _bins = np.arange(np.floor(_s.min() / 5) * 5, _s.max() + 5, 5)
    _idx = np.digitize(_s, _bins)
    _u_bin = np.array([u_i[_idx == j].sum() for j in range(1, len(_bins))])
    _grid = np.linspace(_s.min(), _s.max(), 300)
    _U = np.array([u_i[_s >= c].sum() for c in _grid])
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.8))
    _ax[0].bar(_bins[:-1] + 2.5, _u_bin, width=4.5, color=np.where(_u_bin >= 0, "#2e7d32", "#b3261e"))
    _ax[0].axvline(s_be_analitico, color="k", ls=":", label=f"s_BE = {s_be_analitico:.1f}")
    _ax[0].set_xlabel("score calibrado (puntos)")
    _ax[0].set_ylabel("utilidad esperada del tramo (MM$)")
    _ax[0].set_title("Utilidad marginal por tramo de 5 puntos")
    _ax[0].legend(fontsize=8)
    _ax[1].plot(_grid, _U, color="#1f4e79")
    _ax[1].axvline(s_be_analitico, color="k", ls=":", label="breakeven")
    _ax[1].axvline(s_raroc, color="#ef6c00", ls="--", label="RAROC ≥ 15%")
    _ax[1].set_xlabel("cutoff (puntos)")
    _ax[1].set_ylabel("utilidad esperada total (MM$)")
    _ax[1].set_title(f"U(c) con LGD {economia['lgd']:.0%}, r {economia['tasa']:.0%}, k {economia['k']:.1%}")
    _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    FACTOR,
    brecha_grilla,
    economia,
    irb_coincide,
    mo,
    s_be_analitico,
    s_be_grilla,
    s_be_scipy_opt,
    s_raroc,
    tabla_reglas,
):
    _pd_be = economia["g"] / (economia["g"] + economia["l"])
    mo.md(rf"""
    **Lectura.** Odds de breakeven $l/g$ = {economia['l'] / economia['g']:.3f} ⇒ PD de breakeven
    $g/(g+l)$ = {_pd_be:.2%} ⇒ $s_{{BE}}$ = **{s_be_analitico:.2f}**. `minimize_scalar` sobre la utilidad suavizada da
    {s_be_scipy_opt:.2f} y el argmax exacto en los datos {s_be_grilla:.2f} (la brecha entre scores vecinos ahí es
    {brecha_grilla:.3f} pts). Coinciden porque, con PD monótona en el score, **maximizar la utilidad total equivale
    a aprobar a todo cliente con utilidad marginal positiva**.

    Cada vez que $l/g$ se duplica, el corte sube exactamente un PDO (20 puntos): $\partial s_{{BE}}/\partial\ln(l/g)$ =
    factor = {FACTOR:.2f}. El RAROC ≥ 15% corta en {s_raroc:.1f}, más arriba que el breakeven, porque exige que el
    cliente marginal pague además el costo del capital que consume (Φ scipy = stdlib: {irb_coincide}). El apetito de
    mora es una restricción sobre el **promedio** de la cartera, no sobre el cliente marginal. Por eso puede
    quedar lejos del breakeven en cualquiera de los dos sentidos. La utilidad que se sacrifica al respetarlo es el
    **precio sombra** del apetito: en esta configuración,
    {(tabla_reglas['utilidad_mm'].iloc[0] - tabla_reglas['utilidad_mm'].iloc[-1]):,.1f} MM\$ de utilidad esperada.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Cuándo falla (1): descalibración ⇒ el corte se corre $\text{factor}\cdot\Delta$ puntos

    El breakeven usa la PD **en uso**. Si la verdad es $p^{real}(s)=\sigma(\text{logit}\,p(s)+\Delta)$, el corte
    óptimo verdadero es $s^*=s_{BE}+\text{factor}\cdot\Delta$. Con la PD anclada al ciclo y OOT en deterioro,
    $\Delta>0$: el corte calculado es **demasiado laxo**. La pérdida de utilidad es de **segundo orden**:
    $U(s^*)-U(s_{BE})\approx\tfrac12\,|u'(s^*)|\,f(s^*)\,(\text{factor}\cdot\Delta)^2$. Un error chico casi no cuesta
    y uno grande cuesta cuadráticamente (derivación en el `.md`, §3.4).

    Evaluamos la utilidad **verdadera** con `pd_real` sobre OOT para cortes calculados con distintos
    $\Delta\delta$ (el mismo desplazamiento del control de §2). La columna `delta_real` es el δ que haría que la PD
    en uso clavara la PD real media de OOT.
    """)
    return


@app.cell
def _(FACTOR, OFFSET, biseccion, economia, np, oot, pd, sigmoide_np):
    _s = oot["score"].values                       # score del modelo calibrado a la TC (sin Δδ)
    _m = oot["monto_mm"].values
    _pr = oot["pd_real"].values
    _g, _l = economia["g"], economia["l"]
    _u_real = _m * ((1 - _pr) * _g - _pr * _l)
    _o = np.argsort(_s)
    _cola = np.cumsum(_u_real[_o][::-1])[::-1]

    def utilidad_verdadera(c):
        return float(_u_real[_s >= c].sum())

    s_opt_verdad = float(_s[_o][int(np.argmax(_cola))])
    delta_real = biseccion(lambda d: sigmoide_np(oot["lp"].values + d).mean() - _pr.mean(), -3.0, 3.0)
    _s_be0 = OFFSET + FACTOR * np.log(_l / _g)        # breakeven con la PD de la TC (Δδ = 0)
    _filas = []
    for _dd in [-0.30, 0.0, 0.15, delta_real, 0.45, 0.80]:
        _c = _s_be0 + FACTOR * _dd                   # corte calculado si se calibrara con TC + Δδ
        _filas.append({"delta_usado": _dd, "cutoff_en_score_TC": _c, "aprobacion": float((_s >= _c).mean()),
                       "utilidad_verdadera_mm": utilidad_verdadera(_c)})
    tabla_descal = pd.DataFrame(_filas)
    tabla_descal["perdida_vs_optimo_mm"] = utilidad_verdadera(s_opt_verdad) - tabla_descal["utilidad_verdadera_mm"]
    tabla_descal["cutoff_optimo_predicho"] = _s_be0 + FACTOR * delta_real
    tabla_descal.round(3)
    return delta_real, s_opt_verdad, tabla_descal, utilidad_verdadera


@app.cell
def _(FACTOR, delta_real, economia, mo, s_opt_verdad, tabla_descal, utilidad_verdadera):
    _s_be0 = tabla_descal.loc[tabla_descal["delta_usado"] == 0.0, "cutoff_en_score_TC"].iloc[0]
    _perd0 = tabla_descal.loc[tabla_descal["delta_usado"] == 0.0, "perdida_vs_optimo_mm"].iloc[0]
    _pred = _s_be0 + FACTOR * delta_real
    mo.md(rf"""
    **Lectura.** δ que clava la PD real de OOT: {delta_real:+.3f} ⇒ corrimiento predicho del corte
    factor·Δ = {FACTOR * delta_real:+.1f} pts: de {_s_be0:.1f} a **{_pred:.1f}**. El óptimo verdadero en los datos es
    **{s_opt_verdad:.1f}**. No coinciden al decimal porque la descalibración real no es un desplazamiento uniforme
    del logit: hay heterogeneidad, canal omitido y el −99 plantado. El sentido y el orden de magnitud sí coinciden.
    Usar la PD de la TC cuesta {_perd0:,.1f} MM\$ de utilidad verdadera frente al óptimo. Es
    {100 * _perd0 / utilidad_verdadera(s_opt_verdad):.1f}% de la utilidad máxima:
    poco, por la forma cuadrática. Sobrecorregir (Δδ = 0,80) cuesta más que no corregir. Con el control de §2
    (Δδ = {economia['delta_extra']:+.2f}) puede reproducir cualquier fila.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Cutoff por segmento: cuándo gana y cuánto cuesta gobernarlo

    Sin restricciones, cada segmento corta en **su** breakeven. Con una restricción agregada (volumen o mora), el
    lagrangiano da la condición de optimalidad: la **utilidad marginal ajustada** se iguala entre segmentos en el
    corte (§3.5 del `.md`). Consecuencia importante: si la PD está **calibrada por segmento** y la economía $(g,l)$
    es la misma, el corte óptimo es la misma **PD** en todos los segmentos. En ese caso el cutoff por segmento solo
    corrige una calibración que el modelo no hizo. Gana de verdad cuando (a) el mismo score significa distinta PD por
    segmento o (b) la economía difiere.

    Experimento: el scorecard **no usa canal**, pero el generador le asigna efecto (app +0,30, fuerza de venta +0,25,
    web +0,15 en log-odds) y en OOT la mezcla migra hacia app. Además, los costos de adquisición difieren: fuerza de
    venta paga comisión. Se estima un δ relativo por canal en la muestra de calibración (DEV+HO) y se evalúa la
    utilidad **verdadera** en OOT. El control reduce la muestra de calibración para ver cuándo el ruido de estimar
    δ por segmento se come la ganancia (bootstrap, B = 60).
    """)
    return


@app.cell
def _(mo):
    ctl_frac = mo.ui.slider(0.02, 1.0, step=0.02, value=1.0, label="Fracción de DEV+HO usada para calibrar por canal")
    ctl_frac
    return (ctl_frac,)


@app.cell
def _(
    FACTOR,
    OFFSET,
    biseccion,
    ctl_frac,
    economia,
    muestras,
    np,
    oot,
    pd,
    sigmoide_np,
):
    CANALES = ["sucursal", "web", "app", "fuerza_venta"]
    ADQ_CANAL = {"sucursal": 0.020, "web": 0.010, "app": 0.010, "fuerza_venta": 0.050}   # supuesto ilustrativo
    _calib = pd.concat([muestras["DEV"], muestras["HO"]], ignore_index=True)
    _k_base = economia["k"]          # incluye un costo de adquisición promedio implícito de ≈2%

    def deltas_por_canal(df_):
        """δ relativo por canal: corrección de nivel del canal respecto del total, en la muestra dada."""
        _lp = df_["lp"].values
        _d_tot = biseccion(lambda d: sigmoide_np(_lp + d).mean() - df_["malo"].mean(), -4, 4, tol=1e-9)
        _out = {}
        for _c in CANALES:
            _mk = (df_["canal"] == _c).values
            _obj = df_["malo"].values[_mk].mean()
            _out[_c] = biseccion(lambda d: sigmoide_np(_lp[_mk] + d).mean() - _obj, -4, 4, tol=1e-9) - _d_tot
        return _out

    def cortes_por_canal(deltas):
        _res = {}
        for _c in CANALES:
            _k = _k_base - 0.02 + ADQ_CANAL[_c]
            _g, _l = economia["tasa"] - _k, economia["lgd"] + _k
            _res[_c] = OFFSET + FACTOR * (np.log(_l / _g) + deltas[_c])
        return _res

    _s = oot["score"].values
    _m = oot["monto_mm"].values
    _pr = oot["pd_real"].values
    _canal = oot["canal"].values
    _k_i = _k_base - 0.02 + np.array([ADQ_CANAL[c] for c in _canal])
    _u_real = _m * ((1 - _pr) * (economia["tasa"] - _k_i) - _pr * (economia["lgd"] + _k_i))

    def utilidad_con_cortes(cortes):
        _c_i = np.array([cortes[c] for c in _canal])
        return float(_u_real[_s >= _c_i].sum()), float((_s >= _c_i).mean())

    _s_unico = OFFSET + FACTOR * np.log(economia["l"] / economia["g"])
    u_unico, ap_unico = utilidad_con_cortes({c: _s_unico for c in CANALES})
    deltas_canal = deltas_por_canal(_calib)
    cortes_canal = cortes_por_canal(deltas_canal)
    u_seg, ap_seg = utilidad_con_cortes(cortes_canal)

    # a igual aprobación que el corte único: ordenar por utilidad esperada con PD y economía del canal
    _p_seg = sigmoide_np(oot["lp"].values + np.array([deltas_canal[c] for c in _canal]))
    _u_esp = _m * ((1 - _p_seg) * (economia["tasa"] - _k_i) - _p_seg * (economia["lgd"] + _k_i))
    _n_ap = int(round(ap_unico * len(_s)))
    _top = np.argsort(-_u_esp)[:_n_ap]
    u_iso_ranking = float(_u_real[_top].sum())

    # ruido: bootstrap de la muestra de calibración con fracción f
    _rng = np.random.default_rng(77)
    _n_sub = max(int(ctl_frac.value * len(_calib)), 200)
    _boot_u, _boot_c = [], []
    for _b in range(60):
        _sub = _calib.iloc[_rng.integers(0, len(_calib), _n_sub)]
        try:
            _cc = cortes_por_canal(deltas_por_canal(_sub))
        except AssertionError:
            continue                                  # un canal sin malos: no se puede calibrar
        _boot_c.append([_cc[c] for c in CANALES])
        _boot_u.append(utilidad_con_cortes(_cc)[0])
    _boot_c = np.array(_boot_c)
    ruido_segmentos = {"n_calibracion": _n_sub, "replicas_validas": len(_boot_u),
                       "ganancia_media_mm": float(np.mean(_boot_u) - u_unico),
                       "prob_ganancia_negativa": float(np.mean(np.array(_boot_u) < u_unico)),
                       "sd_cortes_pts": dict(zip(CANALES, _boot_c.std(axis=0).round(2)))}

    tabla_segmentos = pd.DataFrame({
        "delta_rel": deltas_canal, "adq": ADQ_CANAL, "cutoff_segmento": cortes_canal,
        "cutoff_unico": {c: _s_unico for c in CANALES},
        "mix_oot": oot["canal"].value_counts(normalize=True).reindex(CANALES).to_dict()})
    tabla_segmentos.round(3)
    return (
        ap_seg,
        ap_unico,
        cortes_canal,
        deltas_canal,
        ruido_segmentos,
        tabla_segmentos,
        u_iso_ranking,
        u_seg,
        u_unico,
    )


@app.cell
def _(ap_seg, ap_unico, ctl_frac, mo, ruido_segmentos, u_iso_ranking, u_seg, u_unico):
    _r = ruido_segmentos
    mo.md(rf"""
    **Lectura.** Utilidad **verdadera** en OOT: corte único {u_unico:,.1f} MM\$ (aprobación {ap_unico:.1%}); cortes por
    canal {u_seg:,.1f} MM\$ (aprobación {ap_seg:.1%}), una ganancia de {u_seg - u_unico:+,.1f} MM\$
    ({(u_seg / u_unico - 1):+.1%}). A **igual aprobación** que el corte único, ordenar por utilidad esperada del canal
    (igualar la utilidad marginal) da {u_iso_ranking:,.1f} MM\$ ({u_iso_ranking - u_unico:+,.1f}): es la forma de
    ganar sin ceder volumen.

    **Ruido** (fracción {ctl_frac.value:.0%} de DEV+HO, n = {_r['n_calibracion']:,}): ganancia media
    {_r['ganancia_media_mm']:+,.1f} MM\$, probabilidad de que los cortes por canal rindan **menos** que el único
    {_r['prob_ganancia_negativa']:.0%}. Desviación estándar de cada corte (pts): {', '.join(f'{c} {v:.1f}' for c, v in _r['sd_cortes_pts'].items())}. Baje el
    control a 0,04–0,10: la desviación de los cortes crece como $1/\sqrt{{n\,\bar p(1-\bar p)}}$ y la ganancia se
    diluye. A eso se suma el costo de gobierno: cada corte es un parámetro más que monitorear, documentar y defender.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Caso motos: LGD con garantía, PD de vida, EL por banda y pricing

    Crédito de moto con pie, cuota fija (francesa) y la moto como garantía recuperable. **Todos los parámetros
    son genéricos e ilustrativos** (no son datos de ninguna empresa). Van con rango plausible y se declaran como
    supuestos:

    | Parámetro | Base | Rango | Nota |
    |---|---|---|---|
    | Precio de la moto | 4,0 MM\$ | 3–6 MM\$ | monto financiado = precio·(1 − pie) |
    | Pie | 20% | 10–30% | |
    | Tasa mensual | 2,2% | 1,9–2,6% | 26,4% lineal anual; tope: TMC del tramo |
    | Plazo | 36 meses | 24–48 | |
    | Depreciación anual | 20% | 15–25% | geométrica |
    | Castigo inicial (salida de tienda / liquidación) | 10% | 0–20% | |
    | Costo de recupero y remate | 15% del valor | 10–20% | |
    | Meses de default a venta | 6 | 4–8 | |
    | Probabilidad de recuperar la moto | 60% | 40–85% | el supuesto más incierto |
    | Costo de fondos | 10% anual efectivo | 7–14% | |
    | Costo operativo | 4% anual sobre saldo | 3–6% | |
    | Costo de adquisición | 200.000 \$ por crédito | 100–300 mil | comisión dealer, originación |
    | Mes de máxima intensidad de default | 7 | 5–10 | forma de la curva de default |

    **Default** = 90+ días de mora. Si se declara en el mes $d$, la última cuota pagada fue la $d-4$:
    $\text{EAD}(d)=B_{d-4}$, el capital insoluto. Si se recupera la moto (probabilidad $\pi$), se vende en
    $d+T$ a $V(d+T)(1-c)$, con tope en la acreencia. Si no, se pierde todo:

    $$\text{LGD}(d)=\pi\cdot\max\Big(0,\,1-\frac{\min\{V(d+T)(1-c),\,\text{acreencia}\}/(1+i)^{T}}{\text{EAD}(d)}\Big)+(1-\pi).$$

    Los controles mueven los supuestos del caso. La curva de LGD por mes de default es la pieza central.
    """)
    return


@app.cell
def _(mo):
    ctl_pie = mo.ui.slider(0.10, 0.40, step=0.05, value=0.20, label="Pie")
    ctl_dep = mo.ui.slider(0.10, 0.30, step=0.01, value=0.20, label="Depreciación anual")
    ctl_trec = mo.ui.slider(3, 10, step=1, value=6, label="Meses de default a venta")
    ctl_prec = mo.ui.slider(0.40, 0.90, step=0.05, value=0.60, label="Prob. de recuperar la moto")
    ctl_plazo = mo.ui.dropdown(options=["24", "36", "48"], value="36", label="Plazo (meses)")
    ctl_tasa_m = mo.ui.slider(0.015, 0.030, step=0.001, value=0.022, label="Tasa mensual")
    mo.vstack([mo.hstack([ctl_pie, ctl_dep, ctl_trec]), mo.hstack([ctl_prec, ctl_plazo, ctl_tasa_m])])
    return ctl_dep, ctl_pie, ctl_plazo, ctl_prec, ctl_tasa_m, ctl_trec


@app.cell
def _(np):
    PARAM_MOTOS_BASE = {"precio": 4_000_000, "pie": 0.20, "r": 0.022, "N": 36, "dep": 0.20, "h0": 0.10,
                        "c_rec": 0.15, "T_rec": 6, "p_rec": 0.60, "cf": 0.10, "op": 0.04, "adq": 200_000,
                        "pico": 7}
    TMC_ANUAL_50_200 = 0.3450       # CMF, vigente desde el 14-08-2026, tramo > 50 y ≤ 200 UF (lineal anual)

    def cronograma_cerrado(M0, r, N):
        """Saldo B_k tras la cuota k (k = 0..N) por fórmula cerrada de la anualidad."""
        _k = np.arange(N + 1)
        _C = M0 * r / (1 - (1 + r) ** -N)
        return _C, M0 * ((1 + r) ** N - (1 + r) ** _k) / ((1 + r) ** N - 1)

    def cronograma_recursivo(M0, r, N):
        """Mismo saldo por recursión B_k = B_{k-1}(1+r) − C (lo que hace un core bancario)."""
        _C = M0 * r / (1 - (1 + r) ** -N)
        _B = np.empty(N + 1)
        _B[0] = M0
        for _j in range(1, N + 1):
            _B[_j] = _B[_j - 1] * (1 + r) - _C
        return _C, _B

    def economia_motos(par, r=None):
        """Flujos, EAD, LGD y VPN por mes de default de un crédito de moto (pesos)."""
        _r = par["r"] if r is None else r
        _N = int(par["N"])
        _M0 = par["precio"] * (1 - par["pie"])
        _C, _B = cronograma_cerrado(_M0, _r, _N)
        _i = (1 + par["cf"]) ** (1 / 12) - 1                       # tasa de descuento mensual
        _k = np.arange(_N + 1)
        _flujo = np.r_[0.0, _C - par["op"] / 12 * _B[:-1]]        # cuota menos costo operativo del mes
        _cum = np.cumsum(_flujo / (1 + _i) ** _k)                  # VP acumulado de lo cobrado
        _npv_bueno = -_M0 - par["adq"] + _cum[_N]
        _d = np.arange(4, _N + 4)                                  # mes en que se alcanza 90+ DPD
        _ead = _B[_d - 4]
        _t = _d + par["T_rec"]
        _V = par["precio"] * (1 - par["h0"]) * (1 - par["dep"]) ** (_t / 12)
        _acreencia = _ead * (1 + _r) ** (3 + par["T_rec"])        # capital + interés pactado hasta la venta
        _rec = np.minimum(_V * (1 - par["c_rec"]), _acreencia)
        _lgd_si_recupera = np.maximum(0.0, 1 - _rec / (1 + _i) ** par["T_rec"] / _ead)
        _lgd = par["p_rec"] * _lgd_si_recupera + (1 - par["p_rec"])
        _npv_malo = -_M0 - par["adq"] + _cum[_d - 4] + par["p_rec"] * _rec / (1 + _i) ** _t
        _x = (_d - 3) / (par["pico"] - 3)
        _forma = _x * np.exp(1 - _x)                               # intensidad relativa de default
        _S = np.cumsum(_forma)
        return {"M0": _M0, "C": _C, "B": _B, "i": _i, "d": _d, "ead": _ead, "V": _V, "rec": _rec,
                "lgd": _lgd, "npv_bueno": _npv_bueno, "npv_malo": _npv_malo, "forma": _forma,
                "S12": float(_forma[_d <= 12].sum()), "SN": float(_forma.sum()), "S": _S,
                "kv": float(_forma.sum() / _forma[_d <= 12].sum()), "w": _forma / _forma.sum()}

    def banda_exacta(eco, pd12):
        """Riesgos proporcionales: PD de vida, VPN esperado y EL (VP) con la distribución de
        tiempos PROPIA de la banda (los malos de bandas riesgosas caen antes)."""
        _lam = -np.log(1 - pd12) / eco["S12"]
        _S_prev = np.r_[0.0, eco["S"][:-1]]
        _f = np.exp(-_lam * _S_prev) - np.exp(-_lam * eco["S"])  # P(default en d)
        _sobrev = np.exp(-_lam * eco["SN"])
        _e = _sobrev * eco["npv_bueno"] + float(np.sum(_f * eco["npv_malo"]))
        _el = float(np.sum(_f * eco["lgd"] * eco["ead"] / (1 + eco["i"]) ** eco["d"]))
        return {"pd_vida": 1 - _sobrev, "e_npv": _e, "el_vp": _el, "f": _f}

    def banda_aprox(eco, pd12):
        """Aproximación de tiempos comunes (límite PD → 0): la que usa la planilla para el breakeven."""
        _pdv = 1 - (1 - pd12) ** eco["kv"]
        _e = (1 - _pdv) * eco["npv_bueno"] + _pdv * float(np.sum(eco["w"] * eco["npv_malo"]))
        return {"pd_vida": _pdv, "e_npv": _e}
    return (
        PARAM_MOTOS_BASE,
        TMC_ANUAL_50_200,
        banda_aprox,
        banda_exacta,
        cronograma_cerrado,
        cronograma_recursivo,
        economia_motos,
    )


@app.cell
def _(
    PARAM_MOTOS_BASE,
    cronograma_cerrado,
    cronograma_recursivo,
    ctl_dep,
    ctl_pie,
    ctl_plazo,
    ctl_prec,
    ctl_tasa_m,
    ctl_trec,
    economia_motos,
    np,
    plt,
):
    par_motos = dict(PARAM_MOTOS_BASE)
    par_motos.update(pie=ctl_pie.value, dep=ctl_dep.value, T_rec=int(ctl_trec.value), p_rec=ctl_prec.value,
                     N=int(ctl_plazo.value), r=ctl_tasa_m.value)
    eco_motos = economia_motos(par_motos)
    _C1, _B1 = cronograma_cerrado(eco_motos["M0"], par_motos["r"], par_motos["N"])
    _C2, _B2 = cronograma_recursivo(eco_motos["M0"], par_motos["r"], par_motos["N"])
    cronograma_coincide = bool(np.isclose(_C1, _C2) and np.allclose(_B1, _B2, atol=1e-6) and abs(_B1[-1]) < 1e-6)

    _e = eco_motos
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.8))
    _ax[0].plot(_e["d"], _e["ead"] / 1e6, color="#1f4e79", label="EAD(d) = saldo capital")
    _ax[0].plot(_e["d"], _e["V"] * (1 - par_motos["c_rec"]) / 1e6, color="#2e7d32",
                label="valor neto de la moto al vender")
    _ax[0].set_xlabel("mes de default d (90+ DPD)")
    _ax[0].set_ylabel("MM$")
    _ax[0].set_title("Exposición vs garantía")
    _ax[0].legend(fontsize=8)
    _ax[1].plot(_e["d"], 100 * _e["lgd"], color="#b3261e", label="LGD(d) esperada")
    _ax[1].axhline(100 * (1 - par_motos["p_rec"]), color="0.5", ls=":", label="piso = 1 − prob. recupero")
    _ax2 = _ax[1].twinx()
    _ax2.bar(_e["d"], 100 * _e["w"], color="0.85", width=0.8)
    _ax2.set_ylabel("% de los defaults (tiempos)", color="0.45")
    _ax[1].set_zorder(_ax2.get_zorder() + 1)
    _ax[1].patch.set_visible(False)
    _ax[1].set_xlabel("mes de default d")
    _ax[1].set_ylabel("LGD (%)")
    _ax[1].set_title("LGD por mes de default")
    _ax[1].legend(fontsize=8, loc="upper right")
    _fig.tight_layout()
    _fig
    return cronograma_coincide, eco_motos, par_motos


@app.cell
def _(cronograma_coincide, eco_motos, mo, np, par_motos):
    _e = eco_motos
    _lgd_pond = float(np.sum(_e["w"] * _e["lgd"] * _e["ead"]) / np.sum(_e["w"] * _e["ead"]))
    _cruce = _e["d"][np.argmax(_e["V"] * (1 - par_motos["c_rec"]) >= _e["ead"])] if np.any(
        _e["V"] * (1 - par_motos["c_rec"]) >= _e["ead"]) else None
    _txt_cruce = (f"Desde el mes **{_cruce}**, el valor neto de la moto supera el saldo. Ahí la LGD cae a su piso "
                  f"{1 - par_motos['p_rec']:.0%}: solo se pierde cuando la moto no aparece."
                  if _cruce is not None else "Con estos supuestos el saldo nunca queda bajo el valor neto de la moto.")
    mo.md(rf"""
    **Lectura** (cronograma cerrado = recursivo: {cronograma_coincide}). Monto financiado
    {_e['M0'] / 1e6:.2f} MM\$, cuota {_e['C']:,.0f} \$. LGD si el default es temprano (d = 4):
    **{_e['lgd'][0]:.1%}**; en d = 12: {_e['lgd'][8]:.1%}; al final: {_e['lgd'][-1]:.1%}. {_txt_cruce} La LGD
    ponderada por la distribución de tiempos y por EAD es **{_lgd_pond:.1%}**. El 45% fijo del curso es una
    constante. Aquí la LGD depende del mes, y el default temprano, que es el más frecuente, es también el más caro.
    Una constante bien elegida puede acertar el promedio de la cartera y equivocarse en cada segmento que
    cambie la distribución de tiempos (plazos más largos, bandas más riesgosas). Por eso se modela LGD(d).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.2 De la PD a 12 meses a la PD de vida, y EL por banda

    El scorecard predice $\text{PD}_{12}$ (90+ en 12 meses). Un crédito a 36 meses puede caer después. Con
    **riesgos proporcionales**, $h_b(d)=1-e^{-\lambda_b s(d)}$, la forma $s(d)$ es común a todas las bandas y solo
    cambia el nivel $\lambda_b$. Entonces

    $$\text{PD}^{vida}_b=1-(1-\text{PD}_{12,b})^{\kappa},\qquad \kappa=\frac{\sum_{d\le N+3}s(d)}{\sum_{d\le 12}s(d)}.$$

    Es una identidad exacta del modelo, no una aproximación. La aproximación está en otra parte: usar la **misma**
    distribución condicional de tiempos para todas las bandas. En las bandas riesgosas, los malos caen antes (la
    supervivencia se agota), el default temprano es más caro y la aproximación **subestima** la pérdida. Lo
    medimos.

    Bandas: la master scale del curso (A1 ≥ 660 … E < 540), con la PD calibrada media de Banco Austral como PD a
    12 meses **de ejemplo** y una mezcla de solicitudes de motos desplazada hacia las bandas bajas (ilustrativa).
    """)
    return


@app.cell
def _(FACTOR, OFFSET, banda_aprox, banda_exacta, brentq, eco_motos, k_irb_scipy, np, pd):
    BANDAS = pd.DataFrame({
        "banda": ["A1", "A2", "B1", "B2", "C1", "C2", "D", "E"],
        "score_min": [660, 640, 620, 600, 580, 560, 540, np.nan],
        "pd12": [0.0011, 0.0036, 0.0072, 0.0141, 0.0279, 0.0543, 0.1016, 0.2675],   # Austral (clase 5)
        "mix": [0.02, 0.03, 0.05, 0.08, 0.12, 0.17, 0.20, 0.33],                  # ilustrativo motos
    }).set_index("banda")
    _e = eco_motos
    _filas = {}
    for _b, _row in BANDAS.iterrows():
        _ex = banda_exacta(_e, _row["pd12"])
        _ap = banda_aprox(_e, _row["pd12"])
        _filas[_b] = {"pd12": _row["pd12"], "pd_vida": _ex["pd_vida"], "el_vp": _ex["el_vp"],
                      "el_vp_sobre_monto": _ex["el_vp"] / _e["M0"], "e_npv": _ex["e_npv"],
                      "e_npv_aprox": _ap["e_npv"], "error_aprox": _ap["e_npv"] - _ex["e_npv"],
                      "k_irb_por_peso": float(k_irb_scipy(_row["pd12"], 0.45))}
    tabla_bandas_motos = pd.DataFrame(_filas).T
    tabla_bandas_motos["mix"] = BANDAS["mix"]

    # breakeven de motos: aproximado (fórmula cerrada, límite PD→0) vs exacto (brentq en PD12)
    _G = _e["npv_bueno"]
    _L = -float(np.sum(_e["w"] * _e["npv_malo"]))
    if _G > 0:
        _pdv_be = _G / (_G + _L)
        pd12_be_motos_aprox = 1 - (1 - _pdv_be) ** (1 / _e["kv"])
        pd12_be_motos_exacto = brentq(lambda p: banda_exacta(_e, p)["e_npv"], 1e-6, 0.95, xtol=1e-12)
    else:
        pd12_be_motos_aprox = pd12_be_motos_exacto = np.nan
    s_be_motos_aprox = OFFSET + FACTOR * np.log((1 - pd12_be_motos_aprox) / pd12_be_motos_aprox)
    s_be_motos_exacto = OFFSET + FACTOR * np.log((1 - pd12_be_motos_exacto) / pd12_be_motos_exacto)

    # tabla de estrategia de motos por banda (1.000 solicitudes)
    _t = tabla_bandas_motos
    _filas_e = []
    for _j in range(1, len(_t) + 1):
        _ap = _t.iloc[:_j]
        _n = 1000 * _ap["mix"]
        _filas_e.append({"aprueba_hasta": _t.index[_j - 1], "aprobacion": _ap["mix"].sum(),
                         "pd12_media": float((_n * _ap["pd12"]).sum() / _n.sum()),
                         "pd_vida_media": float((_n * _ap["pd_vida"]).sum() / _n.sum()),
                         "monto_mm": float(_n.sum() * _e["M0"] / 1e6),
                         "el_vp_mm": float((_n * _ap["el_vp"]).sum() / 1e6),
                         "utilidad_mm": float((_n * _ap["e_npv"]).sum() / 1e6)})
    estrategia_motos = pd.DataFrame(_filas_e).set_index("aprueba_hasta")
    estrategia_motos["el_sobre_monto"] = estrategia_motos["el_vp_mm"] / estrategia_motos["monto_mm"]
    estrategia_motos["margen_mm"] = estrategia_motos["utilidad_mm"] + estrategia_motos["el_vp_mm"]
    tabla_bandas_motos.style.format({"pd12": "{:.2%}", "pd_vida": "{:.2%}", "el_vp": "{:,.0f}",
                                     "el_vp_sobre_monto": "{:.2%}", "e_npv": "{:,.0f}", "e_npv_aprox": "{:,.0f}",
                                     "error_aprox": "{:,.0f}", "k_irb_por_peso": "{:.2%}", "mix": "{:.0%}"})
    return (
        BANDAS,
        estrategia_motos,
        pd12_be_motos_aprox,
        pd12_be_motos_exacto,
        s_be_motos_aprox,
        s_be_motos_exacto,
        tabla_bandas_motos,
    )


@app.cell
def _(estrategia_motos):
    estrategia_motos.style.format({"aprobacion": "{:.0%}", "pd12_media": "{:.2%}", "pd_vida_media": "{:.2%}",
                                   "monto_mm": "{:,.0f}", "el_vp_mm": "{:,.1f}", "utilidad_mm": "{:,.1f}",
                                   "el_sobre_monto": "{:.2%}", "margen_mm": "{:,.1f}"})
    return


@app.cell
def _(
    BANDAS,
    eco_motos,
    estrategia_motos,
    mo,
    np,
    pd12_be_motos_aprox,
    pd12_be_motos_exacto,
    s_be_motos_aprox,
    s_be_motos_exacto,
    tabla_bandas_motos,
):
    _t = tabla_bandas_motos
    _mejor = estrategia_motos["utilidad_mm"].idxmax()
    _lims = BANDAS["score_min"].fillna(-np.inf)
    _banda_be = next(b for b in BANDAS.index if s_be_motos_exacto >= _lims[b])
    mo.md(rf"""
    **Lectura.** κ = {eco_motos['kv']:.3f}: la PD de vida es $1-(1-\text{{PD}}_{{12}})^{{{eco_motos['kv']:.2f}}}$
    (banda D: {_t.loc['D', 'pd12']:.2%} → {_t.loc['D', 'pd_vida']:.2%}). VPN si bueno: {eco_motos['npv_bueno']:,.0f} \$.
    Breakeven: PD12 **{pd12_be_motos_exacto:.2%}** (exacto, `brentq`) vs {pd12_be_motos_aprox:.2%} (fórmula
    cerrada con tiempos comunes) ⇒ score **{s_be_motos_exacto:.1f}** vs {s_be_motos_aprox:.1f}. La aproximación
    queda más laxa porque ignora que en bandas riesgosas los malos caen antes y cuestan más. El error en E[VPN] de
    la banda E es {_t.loc['E', 'error_aprox']:,.0f} \$ por crédito; en A1, {_t.loc['A1', 'error_aprox']:,.1f} \$.

    Con el corte en las fronteras de banda, la utilidad es máxima aprobando **hasta {_mejor}**. El breakeven
    ({s_be_motos_exacto:.1f}) cae dentro de la banda **{_banda_be}**, así que esa banda es un promedio de clientes sobre y
    bajo el umbral. Partirla en dos sub-bandas capturaría la parte rentable. Es el costo de decidir con bandas y no con el score.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.3 Pricing por banda con tope legal, y pie mínimo

    **Tasa requerida** por banda: la tasa mensual $r_b$ que hace $E[\text{VPN}_b](r_b)=m^*\cdot M_0$ (margen objetivo
    $m^*$ = 5% del monto en VP). Se resuelve con bisección numpy y con `brentq`. La planilla usa una **secante**
    entre la tasa base y la tasa TMC; aquí medimos su error. Se compara con la **tasa máxima convencional** del
    tramo > 50 UF y ≤ 200 UF: 34,50% lineal anual (certificado CMF vigente desde el 14-08-2026) ⇒ 2,875% mensual.
    La TMC cambia cada mes: en producción se lee del certificado vigente, no se fija en el código.

    **Pie mínimo**: a la tasa base y a la TMC, el pie que hace $E[\text{VPN}_b]=0$. Subir el pie baja la EAD **y** la LGD a la
    vez (la moto cubre una fracción mayor del saldo). Es la palanca de «monto/plazo» del curso para no cerrar la
    puerta a una banda.
    """)
    return


@app.cell
def _(
    BANDAS,
    TMC_ANUAL_50_200,
    banda_exacta,
    biseccion,
    brentq,
    economia_motos,
    np,
    par_motos,
    pd,
):
    MARGEN_OBJ = 0.05
    _r_tmc = TMC_ANUAL_50_200 / 12
    _r0 = par_motos["r"]

    def pie_minimo(pd12, r):
        """Pie mínimo que hace E[VPN] ≥ 0 a la tasa r. E[VPN](pie) no es monótona (la adquisición es un costo
        fijo y la LGD tiene piso 1 − π), así que se barre una grilla y se refina con brentq."""
        def _g(pie):
            _p = dict(par_motos)
            _p["pie"], _p["r"] = pie, r
            return banda_exacta(economia_motos(_p), pd12)["e_npv"]
        _grid_pie = np.arange(0.0, 0.801, 0.05)
        _vals = np.array([_g(x) for x in _grid_pie])
        if _vals[0] >= 0:
            return 0.0
        if np.any(_vals >= 0):
            _j = int(np.argmax(_vals >= 0))
            return brentq(_g, _grid_pie[_j - 1], _grid_pie[_j], xtol=1e-10)
        return np.nan                          # ningún pie hasta 80% la hace rentable a esa tasa
    _filas = []
    for _b, _row in BANDAS.iterrows():
        _pd12 = _row["pd12"]

        def _f(r, _pd12=_pd12):
            _e = economia_motos(par_motos, r=r)
            return banda_exacta(_e, _pd12)["e_npv"] - MARGEN_OBJ * _e["M0"]

        _lo, _hi = 0.001, 0.20
        _r_bis = biseccion(_f, _lo, _hi, tol=1e-12) if _f(_lo) * _f(_hi) < 0 else np.nan
        _r_brq = brentq(_f, _lo, _hi, xtol=1e-14) if _f(_lo) * _f(_hi) < 0 else np.nan
        _f0, _f1 = _f(_r0), _f(_r_tmc)
        _r_sec = _r0 + (0 - _f0) * (_r_tmc - _r0) / (_f1 - _f0)
        _pie_min = pie_minimo(_pd12, _r0)
        _pie_min_tmc = pie_minimo(_pd12, _r_tmc)
        _filas.append({"banda": _b, "pd12": _pd12, "tasa_req_bisec": _r_bis, "tasa_req_brentq": _r_brq,
                       "tasa_req_secante": _r_sec, "tasa_req_anual_lineal": 12 * _r_brq,
                       "cabe_bajo_TMC": bool(_r_brq <= _r_tmc) if np.isfinite(_r_brq) else False,
                       "pie_min_tasa_base": _pie_min, "pie_min_a_TMC": _pie_min_tmc})
    tabla_pricing = pd.DataFrame(_filas).set_index("banda")
    _ok = tabla_pricing["tasa_req_brentq"].notna()
    pricing_coincide = bool(np.allclose(tabla_pricing.loc[_ok, "tasa_req_bisec"], tabla_pricing.loc[_ok, "tasa_req_brentq"], atol=1e-9))
    tabla_pricing.style.format({"pd12": "{:.2%}", "tasa_req_bisec": "{:.3%}", "tasa_req_brentq": "{:.3%}",
                                "tasa_req_secante": "{:.3%}", "tasa_req_anual_lineal": "{:.1%}", "pie_min_tasa_base": "{:.1%}",
                                "pie_min_a_TMC": "{:.1%}"})
    return MARGEN_OBJ, pie_minimo, pricing_coincide, tabla_pricing


@app.cell
def _(MARGEN_OBJ, TMC_ANUAL_50_200, mo, np, pricing_coincide, tabla_pricing):
    _t = tabla_pricing
    _fuera = [b for b in _t.index if not _t.loc[b, "cabe_bajo_TMC"]]
    _err = (_t["tasa_req_secante"] - _t["tasa_req_brentq"]).abs()
    _err_dentro = _err[_t["tasa_req_brentq"] <= TMC_ANUAL_50_200 / 12]
    mo.md(rf"""
    **Lectura** (bisección = brentq: {pricing_coincide}). Con margen objetivo {MARGEN_OBJ:.0%}, la tasa requerida va de
    {_t['tasa_req_brentq'].min():.2%} mensual (A1) a {_t['tasa_req_brentq'].max():.2%} (E). Bandas que **no caben** bajo la
    TMC ({TMC_ANUAL_50_200:.2%} anual = {TMC_ANUAL_50_200 / 12:.3%} mensual): **{', '.join(_fuera) if _fuera else 'ninguna'}**.
    Para ellas no hay precio legal que pague el riesgo con el margen objetivo. Quedan tres caminos: rechazar, exigir
    más pie o acortar plazo. En la banda E, el pie mínimo para $E[\text{{VPN}}]\ge0$ a la tasa base es
    **{'inexistente (ni con 80%)' if not np.isfinite(_t.loc['E', 'pie_min_tasa_base']) else format(_t.loc['E', 'pie_min_tasa_base'], '.0%')}**
    y a la TMC {'inexistente' if not np.isfinite(_t.loc['E', 'pie_min_a_TMC']) else format(_t.loc['E', 'pie_min_a_TMC'], '.0%')}.
    La razón: la LGD tiene piso $1-\pi$ (la moto que no aparece) y el pie no lo baja. La palanca que sí lo baja es
    $\pi$ (GPS, seguro contra robo, gestión de cobranza temprana). La TMC funciona entonces como un
    **cutoff de facto**. Error máximo de la secante de la planilla dentro de la TMC: {10000 * _err_dentro.max():.2f} pb mensuales.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Tornado: sensibilidad del cutoff de breakeven (motos)

    Cada barra mueve **un** supuesto a su extremo bajo y alto del rango, con el resto en su valor base, y recalcula el
    score de breakeven exacto. La fila «TC (Δδ ± 0,25)» no cambia la economía. Mide cuánto debería moverse el corte
    **en el score del modelo** si la tendencia central verdadera difiere en ±0,25 log-odds de la calibrada:
    $\pm\text{factor}\cdot0{,}25=\pm7{,}2$ puntos, exacto e independiente de todo lo demás.
    """)
    return


@app.cell
def _(
    FACTOR,
    OFFSET,
    PARAM_MOTOS_BASE,
    banda_exacta,
    brentq,
    economia_motos,
    np,
    pd,
    plt,
):
    def score_be_motos(par):
        _e = economia_motos(par)
        _f = lambda p: banda_exacta(_e, p)["e_npv"]
        if _f(1e-6) <= 0:
            return np.inf                      # ni el mejor cliente es rentable
        if _f(0.95) > 0:
            return -np.inf
        _p = brentq(_f, 1e-6, 0.95, xtol=1e-12)
        return OFFSET + FACTOR * np.log((1 - _p) / _p)

    s_be_base_motos = score_be_motos(PARAM_MOTOS_BASE)
    RANGOS = {"dep": (0.15, 0.25), "c_rec": (0.10, 0.20), "T_rec": (4, 8), "p_rec": (0.45, 0.75),
              "pie": (0.10, 0.30), "r": (0.019, 0.025), "cf": (0.07, 0.14), "op": (0.03, 0.06),
              "adq": (100_000, 300_000), "N": (24, 48), "pico": (5, 10)}
    ETIQ = {"dep": "depreciación 15–25%", "c_rec": "costo recupero 10–20%", "T_rec": "meses a venta 4–8",
            "p_rec": "prob. recupero 45–75%", "pie": "pie 10–30%", "r": "tasa 1,9–2,5% mensual",
            "cf": "costo fondos 7–14%", "op": "costo operativo 3–6%", "adq": "adquisición 100–300 mil",
            "N": "plazo 24–48", "pico": "mes pico default 5–10"}
    _filas = []
    for _k, (_lo, _hi) in RANGOS.items():
        _res = []
        for _v in (_lo, _hi):
            _p = dict(PARAM_MOTOS_BASE)
            _p[_k] = _v
            _res.append(score_be_motos(_p))
        _filas.append({"supuesto": ETIQ[_k], "score_bajo": _res[0], "score_alto": _res[1]})
    _filas.append({"supuesto": "TC (Δδ ± 0,25)", "score_bajo": s_be_base_motos - FACTOR * 0.25,
                   "score_alto": s_be_base_motos + FACTOR * 0.25})
    tornado = pd.DataFrame(_filas)
    tornado["rango"] = (tornado["score_alto"] - tornado["score_bajo"]).abs()
    tornado = tornado.sort_values("rango").reset_index(drop=True)

    _fig, _ax = plt.subplots(figsize=(8, 4.6))
    _y = np.arange(len(tornado))
    _ax.barh(_y, tornado["score_bajo"] - s_be_base_motos, color="#1f4e79", label="extremo bajo del supuesto")
    _ax.barh(_y, tornado["score_alto"] - s_be_base_motos, color="#ef6c00", alpha=0.8, label="extremo alto del supuesto")
    _ax.set_yticks(_y, tornado["supuesto"], fontsize=8)
    _ax.axvline(0, color="k", lw=0.8)
    _ax.set_xlabel(f"cambio del cutoff de breakeven (puntos; base = {s_be_base_motos:.1f})")
    _ax.set_title("Tornado: cutoff de breakeven del caso motos")
    _ax.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return RANGOS, s_be_base_motos, score_be_motos, tornado


@app.cell
def _(mo, s_be_base_motos, tornado):
    _top = tornado.iloc[::-1].head(4)
    _lineas = "\n".join(f"- {r.supuesto}: de {r.score_bajo:.1f} a {r.score_alto:.1f} ({r.rango:.1f} pts)"
                        for r in _top.itertuples())
    mo.md(rf"""
    **Lectura.** Base: {s_be_base_motos:.1f}. Los cuatro supuestos que más mueven el corte:

    {_lineas}

    El corte es $s_{{BE}}=\text{{offset}}+\text{{factor}}\ln(L/G)$: la misma variación **relativa** de $G$ (VPN si
    bueno) o de $L$ (pérdida si malo) lo mueve lo mismo. Pero $G$ es una **diferencia chica entre números grandes**
    (intereses menos fondos, operación y adquisición), así que los supuestos del margen lo mueven en proporción
    mucho más que los de recupero a $L$. Por eso tasa, costo de fondos, plazo y adquisición dominan, y
    depreciación o costo de remate pesan poco. La regla práctica es gastar el esfuerzo de estimación donde el tornado
    es más ancho: la tasa efectivamente cobrada (descuentos, prepago) y el costo de fondos marginal, antes que
    afinar la curva de depreciación. La TC entra con ±7,2 pts, del orden de los supuestos económicos grandes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Checks del módulo

    Si alguno falla, el notebook falla.
    """)
    return


@app.cell
def _(
    DELTA_TC,
    FACTOR,
    OFFSET,
    PARAM_MOTOS_BASE,
    banda_aprox,
    banda_exacta,
    brecha_grilla,
    cronograma_coincide,
    delta_tc_brentq,
    eco_motos,
    economia,
    estrategia_coincide,
    frontera_oot,
    irb_coincide,
    mo,
    np,
    oot,
    par_motos,
    pd12_be_motos_exacto,
    pricing_coincide,
    s_be_analitico,
    s_be_base_motos,
    s_be_biseccion,
    s_be_brentq,
    s_be_grilla,
    s_be_scipy_opt,
    score_be_motos,
    tabla_pricing,
    tornado,
    u_i,
    utilidad_marginal,
):
    _checks = []
    # 1. δ exacto: bisección numpy = brentq
    assert abs(DELTA_TC - delta_tc_brentq) < 1e-9
    _checks.append("δ exacto: bisección numpy = brentq")
    # 2. Tabla de estrategia: numpy = pandas
    assert estrategia_coincide
    _checks.append("tabla de estrategia numpy = pandas")
    # 3. Breakeven: analítico = bisección = brentq; optimizadores sobre datos a < 0,5 pts
    assert abs(s_be_analitico - s_be_biseccion) < 1e-8 and abs(s_be_analitico - s_be_brentq) < 1e-8
    assert abs(s_be_scipy_opt - s_be_analitico) < 0.5
    assert abs(s_be_grilla - s_be_analitico) <= max(brecha_grilla, 1e-9) + 1e-9
    assert abs(utilidad_marginal(s_be_analitico)) < 1e-12
    _checks.append("breakeven analítico = bisección = brentq ≈ minimize_scalar ≈ argmax en datos")
    # 4. Signo de u_i determinado por el score (con Δδ = 0 la PD en uso es la de la escala)
    if economia["delta_extra"] == 0.0:
        _s = oot["score"].values
        assert np.sum((u_i > 0) != (_s > s_be_analitico)) == 0
        _checks.append("u_i > 0 ⇔ s_i > s_BE")
    # 5. Frontera: la PD media de aprobados sube al bajar el corte (monótona por construcción)
    _pdm = frontera_oot["pd_cal_media"].values
    assert np.all(np.diff(_pdm) <= 1e-12)
    _checks.append("frontera monótona en PD esperada")
    # 6. Duplicar l/g sube el corte exactamente un PDO
    assert np.isclose((OFFSET + FACTOR * np.log(2 * 3.0)) - (OFFSET + FACTOR * np.log(3.0)), 20.0)
    _checks.append("duplicar l/g ⇒ +20 puntos (PDO)")
    # 7. Cronograma: cerrado = recursivo; saldo final 0
    assert cronograma_coincide
    _checks.append("cronograma cerrado = recursivo")
    # 8. LGD en [0,1]; LGD temprana ≥ LGD tardía; piso = 1 − p_rec
    _l = eco_motos["lgd"]
    assert np.all((_l >= 0) & (_l <= 1)) and _l[0] >= _l[-1]
    assert np.all(_l >= 1 - par_motos["p_rec"] - 1e-12)
    _checks.append("LGD ∈ [0,1] y decreciente de temprana a tardía")
    # 9. Riesgos proporcionales: PD vida exacta = 1 − (1 − PD12)^κ; exacto → aprox cuando PD → 0
    for _p in [0.001, 0.05, 0.25]:
        assert np.isclose(banda_exacta(eco_motos, _p)["pd_vida"], banda_aprox(eco_motos, _p)["pd_vida"])
    _a, _b = banda_exacta(eco_motos, 1e-6)["e_npv"], banda_aprox(eco_motos, 1e-6)["e_npv"]
    assert abs(_a - _b) < 1e-3 * abs(_a)
    _checks.append("PD vida = 1 − (1 − PD12)^κ exacto; aprox → exacto si PD → 0")
    # 10. Breakeven motos exacto: E[VPN] = 0 en PD12_BE; tornado base consistente
    assert abs(banda_exacta(eco_motos, pd12_be_motos_exacto)["e_npv"]) < 1e-3
    assert np.isclose(score_be_motos(PARAM_MOTOS_BASE), s_be_base_motos)
    _fila_tc = tornado[tornado["supuesto"].str.startswith("TC")].iloc[0]
    assert np.isclose(_fila_tc["rango"], 2 * FACTOR * 0.25)
    _checks.append("breakeven motos: E[VPN] = 0; fila TC del tornado = 2·factor·0,25")
    # 11. Pricing: bisección = brentq; tasa requerida monótona en PD
    assert pricing_coincide
    _tr = tabla_pricing["tasa_req_brentq"].dropna().values
    assert np.all(np.diff(_tr) > 0)
    _checks.append("pricing bisección = brentq y creciente en PD")
    # 12. Capital IRB: scipy = statistics.NormalDist
    assert irb_coincide
    _checks.append("K IRB: scipy = NormalDist")
    mo.md("**Todos los checks pasaron.**\n\n" + "\n".join(f"- {c}" for c in _checks))
    return


if __name__ == "__main__":
    app.run()

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
    from matplotlib.colors import ListedColormap
    import statsmodels.api as sm
    from scipy import optimize, stats
    from sklearn.metrics import roc_auc_score
    import re as _re

    def coma(texto):
        """Decimales con coma en la prosa (convención de la serie): 0.25 → 0,25."""
        return _re.sub(r"(?<=\d)\.(?=\d)", ",", texto)
    return ListedColormap, coma, mo, optimize, plt, roc_auc_score, sm, stats


@app.cell
def _(mo):
    mo.md(r"""
    # M20 · Monitoreo: tablero, semáforos, gatillos y diagnóstico

    Notebook del módulo M20 de la Serie 2. Profundiza la clase 5 (parte 4: el tablero de Banco
    Austral con 9 indicadores, umbrales fijados antes de mirar, diagnóstico por patrón, jerarquía
    vigilancia → recalibración δ → re-desarrollo → contingencia) y la clase 6 (gatillos de cinco
    partes, RACI).

    La idea central: **simular 24 meses de producción con verdad conocida**, plantar un deterioro de
    tipo conocido a partir del mes $k$ y ver qué indicador se enciende primero, cuánto demora, cuántas
    falsas alarmas cuesta y si el diagnóstico por patrón recupera la causa.

    | § | Qué se demuestra |
    |---|---|
    | 1 | El artefacto congelado: modelo, bandas, cutoff y valores de referencia |
    | 2 | Desempeño en el tiempo: hazard mensual, mora temprana, modelo satélite de «mora esperada» |
    | 3 | Mora temprana como proxy del 90+ a 12 m: cuánto adelanta y cuánto se equivoca |
    | 4 | Simulador de producción (6 escenarios) → tablero mensual de 13 indicadores con semáforos |
    | 5 | Diagnóstico por patrón: árbol de decisión y acción proporcional |
    | 6 | Multiplicidad y falsas alarmas: el tablero bajo «ningún deterioro» |
    | 7 | Monitoreo secuencial: Shewhart, EWMA, CUSUM y CUSUM ajustado por riesgo; ARL |
    | 8 | Umbral por costo: la curva operativa de un CUSUM |
    | 9 | Curvas de cosecha (vintage) contra la curva esperada |
    | 10 | ¿El patrón recupera la causa? Matriz escenario × diagnóstico |
    | 11 | Overrides y su monitoreo |
    | 12 | Gatillos de cinco partes, evaluados sobre el mes de corte |
    | ✔ | Checks del módulo |

    Convenciones del curso: target **1 = malo** (90+ a 12 meses), WoE = ln(%buenos/%malos), PDO 20 y
    600 puntos a odds 50:1, umbrales PSI/CSI 0,10/0,25, binomial p 0,05/0,01, caída relativa de Gini
    20%/30%.
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
    ## 1. El artefacto congelado

    Todo lo que se ajusta, se ajusta en DEV y se congela: cortes de binning, WoE, β, el δ de
    calibración, las 8 bandas y el cutoff. Producción solo **aplica**. El monitoreo compara contra
    estos objetos congelados (la referencia es DEV, no una ventana móvil: ver M08).

    Dos decisiones de diseño del experimento, declaradas:

    - La cartera de desarrollo se genera **sin** el deterioro macro plantado del generador
      (`deterioro=0`, `drift_canal=False`): así el escenario «ninguno» es de verdad estable y cualquier
      semáforo encendido en él es una falsa alarma.
    - El δ se ancla a la **PD verdadera media de DEV** (en la vida real sería la tendencia central de
      M15). DEV tuvo, por azar, menos malos que su PD verdadera; sin este ancla el escenario «ninguno»
      nacería con un sesgo de nivel de ~4% relativo que el tablero detectaría (correctamente) como
      nivel corrido.
    - La **población de producción** se remuestrea de un universo independiente del mismo generador
      (otra semilla, cohortes 2023-07 a 2024-12). La PD real de cada solicitante es conocida.
    """)
    return


@app.cell
def _(a_woe, binear, generar_cartera, np, optimize, pd, sm, tabla_woe):
    _cartera = generar_cartera(deterioro=0.0, drift_canal=False)
    VARS = ["uso_linea_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m",
            "antiguedad_meses", "carga_financiera", "deuda_otras_prom_12m", "consultas_6m"]
    dev = _cartera[_cartera["muestra"] == "DEV"].reset_index(drop=True)
    MAPAS = {v: tabla_woe(dev[v], dev["malo"])[0]["woe"] for v in VARS}
    modelo_dev = sm.Logit(dev["malo"].to_numpy(),
                          sm.add_constant(a_woe(dev, VARS, dev, MAPAS))).fit(disp=0)

    def pd_real_de(df):
        """PD verdadera del generador (incluye el +10% de riesgo de los «sin bureau» = -99)."""
        _p = df["pd_verdadera"].to_numpy()
        _sb = df["meses_desde_mora_12m"].to_numpy() == -99
        return np.clip(_p + 0.10 * (1 - _p) * _sb, 1e-6, 1 - 1e-6)

    W_DEV = a_woe(dev, VARS, dev, MAPAS).to_numpy()
    P_REAL_DEV = pd_real_de(dev)
    _eta = modelo_dev.params.to_numpy()[0] + W_DEV @ modelo_dev.params.to_numpy()[1:]
    DELTA = optimize.brentq(lambda d: (1 / (1 + np.exp(-(_eta + d)))).mean() - P_REAL_DEV.mean(), -2, 2)
    BETA = modelo_dev.params.to_numpy().copy()
    BETA[0] += DELTA                                   # δ congelado en el artefacto
    FACTOR = 20 / np.log(2)
    OFFSET = 600 - FACTOR * np.log(50)

    def pd_de_woe(W):
        """PD calibrada del artefacto a partir de la matriz WoE (n × 7)."""
        return 1 / (1 + np.exp(-(BETA[0] + W @ BETA[1:])))

    def score_de_pd(p):
        """Score PDO del curso (más puntos = menos riesgo)."""
        return OFFSET + FACTOR * np.log((1 - p) / p)

    CORTES = np.array([520, 540, 550, 560, 570, 580, 590.0])
    BANDAS = ["E", "D", "C2", "C1", "B2", "B1", "A2", "A1"]
    CUTOFF = 530.0

    def banda_idx(s):
        """0 = E (peor) … 7 = A1 (mejor)."""
        return np.digitize(s, CORTES)

    def codigos(df, v):
        """Código entero del bin de DEV de la variable v (cortes congelados)."""
        _et, _ = binear(df[v], ref=dev[v])
        return pd.Categorical(_et, categories=list(MAPAS[v].index)).codes

    PD_DEV = pd_de_woe(W_DEV)
    S_DEV = score_de_pd(PD_DEV)
    REF_BANDAS = np.bincount(banda_idx(S_DEV), minlength=8) / len(S_DEV)
    REF_CSI = {v: np.bincount(codigos(dev, v), minlength=len(MAPAS[v])) / len(dev) for v in VARS}
    REF_APROB = float((S_DEV >= CUTOFF).mean())
    REF_MIX_DE = float(REF_BANDAS[:2].sum())

    # universo de producción (otra semilla) y sus versiones «dato roto»
    _u = generar_cartera(n=100_000, semilla=77, deterioro=0.0, drift_canal=False)
    universo = _u[_u["cohorte"] <= "2024-12"].reset_index(drop=True)
    POOL_P = pd_real_de(universo)
    POOL_ETA = np.log(POOL_P / (1 - POOL_P))
    POOL_W = a_woe(universo, VARS, dev, MAPAS).to_numpy()
    _roto = universo.copy()
    _roto["uso_linea_prom_12m"] = 0.0                  # el feed entrega 0 por defecto
    POOL_WR = a_woe(_roto, VARS, dev, MAPAS).to_numpy()
    POOL_COD = np.column_stack([codigos(universo, v) for v in VARS])
    POOL_COD_ROTO = codigos(_roto, "uso_linea_prom_12m")
    _z = universo["uso_linea_prom_12m"].to_numpy()
    POOL_Z = (_z - _z.mean()) / _z.std()
    return (
        BANDAS,
        BETA,
        CORTES,
        CUTOFF,
        DELTA,
        FACTOR,
        MAPAS,
        OFFSET,
        PD_DEV,
        POOL_COD,
        POOL_COD_ROTO,
        POOL_ETA,
        POOL_P,
        POOL_W,
        POOL_WR,
        POOL_Z,
        P_REAL_DEV,
        REF_APROB,
        REF_BANDAS,
        REF_CSI,
        REF_MIX_DE,
        S_DEV,
        VARS,
        W_DEV,
        banda_idx,
        dev,
        modelo_dev,
        pd_de_woe,
        score_de_pd,
        universo,
    )


@app.cell
def _(BANDAS, CORTES, CUTOFF, PD_DEV, P_REAL_DEV, REF_BANDAS, S_DEV, banda_idx, dev, np, pd):
    _b = banda_idx(S_DEV)
    _lim = np.r_[-np.inf, CORTES, np.inf]
    tabla_bandas_dev = pd.DataFrame({
        "banda": BANDAS,
        "score": [f"[{_lim[i]:.0f}, {_lim[i+1]:.0f})" for i in range(8)],
        "pct_dev": REF_BANDAS,
        "pd_media_modelo": [PD_DEV[_b == i].mean() for i in range(8)],
        "pd_verdadera_media": [P_REAL_DEV[_b == i].mean() for i in range(8)],
        "tasa_malos_dev": [dev["malo"].to_numpy()[_b == i].mean() for i in range(8)],
        "aprobada": [(_lim[i] >= CUTOFF) for i in range(8)],
    })
    tabla_bandas_dev.round(4)
    return (tabla_bandas_dev,)


@app.cell
def _(
    CUTOFF,
    DELTA,
    REF_APROB,
    REF_MIX_DE,
    coma,
    dev,
    mo,
    modelo_dev,
    tabla_bandas_dev,
):
    mo.md(coma(f"""
    **Lectura.** Siete variables, β todos negativos (convención WoE: {'sí' if (modelo_dev.params.drop('const') < 0).all() else 'NO'}),
    δ = {DELTA:+.3f} para anclar la PD media a la verdadera. Tasa de malos de DEV {dev['malo'].mean():.2%}
    (esta cartera sintética es más riesgosa que Austral: úsala por su verdad conocida, no por sus niveles).
    Cutoff {CUTOFF:.0f} puntos → aprobación de referencia **{REF_APROB:.1%}**; el mix D+E de referencia es
    **{REF_MIX_DE:.1%}** de las solicitudes. Las 8 bandas se congelan con el modelo; la banda E tiene PD media
    {tabla_bandas_dev['pd_media_modelo'].iloc[0]:.1%} y A1 {tabla_bandas_dev['pd_media_modelo'].iloc[-1]:.2%}.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Desempeño en el tiempo: hazard mensual, mora temprana y «mora esperada»

    El target (90+ a 12 meses) madura 12 meses después de la originación. Para tener indicadores
    **adelantados** hay que modelar *cuándo* ocurre el default dentro de la ventana. El simulador reparte
    el hazard acumulado de cada crédito sobre los meses de vida (MOB, *months on book*):

    $$\Lambda_i(j)=-\ln(1-p_i)\sum_{m\le j} w_m\,\mu(c_i+m),\qquad \sum_{m=1}^{12} w_m=1,$$

    con $p_i$ la PD real a 12 meses, $w_m$ un perfil de maduración (nadie llega a 90+ antes del MOB 4) y
    $\mu(t)$ un multiplicador de **calendario** (1 salvo en el escenario de nivel: un shock macro golpea a
    todas las cosechas vivas a la vez, no solo a las nuevas). Sin shock, $1-e^{-\Lambda_i(12)}=p_i$
    exactamente. El default ocurre en el primer MOB $T$ con $\Lambda_i(T)\ge E$, $E\sim\text{Exp}(1)$.

    **Mora temprana 30+.** Quien llega a 90+ en el MOB $T$ pasó por 30+ en $T-2$. Además hay «curas»:
    buenos que tocan 30+ en los primeros 6 meses y se normalizan, con probabilidad $0{,}6\,p_i$. Por eso la
    mora temprana es un proxy **ruidoso** del malo.

    **Modelo satélite de mora esperada.** Para comparar la mora temprana observada contra algo, en DEV
    se ajusta $\text{logit}\,P(\text{30+ a MOB 3})=a+b\,\text{logit}(\text{PD})$. Se congela con el artefacto.
    Dos implementaciones: IRLS en numpy y `statsmodels.Logit`.
    """)
    return


@app.cell
def _(CUTOFF, PD_DEV, P_REAL_DEV, S_DEV, np, pd, roc_auc_score, sm, stats):
    W_MOB = np.array([0, 0, 0, 6, 9, 11, 12, 12, 11, 10, 9, 8.0])
    W_MOB = W_MOB / W_MOB.sum()
    F_MOB = np.cumsum(W_MOB)                           # fracción del hazard acumulada al MOB j

    def sin_shock(cal):
        """Multiplicador de calendario neutro."""
        return np.ones(np.shape(cal))

    def simular_desempeno(p12, origen, mult, rng, w=W_MOB, q_cura=0.6):
        """Devuelve (T, ep): MOB del 90+ (99 = no cae) y MOB de un episodio 30+ curado (99 = no hay)."""
        _n = len(p12)
        _cal = origen[:, None] + np.arange(1, 13)[None, :]
        _lam = -np.log1p(-p12)[:, None] * w[None, :] * mult(_cal)
        _L = np.cumsum(_lam, axis=1)
        _e = rng.exponential(size=_n)
        _hit = _L >= _e[:, None]
        _T = np.where(_hit.any(axis=1), _hit.argmax(axis=1) + 1, 99)
        _ep_mob = rng.integers(1, 7, _n)
        _q = np.minimum(1, q_cura * p12 * mult(origen + _ep_mob))
        _ep = np.where((_T == 99) & (rng.random(_n) < _q), _ep_mob, 99)
        return _T, _ep

    def mora30(T, ep, mob):
        """¿Tocó 30+ a más tardar en el MOB `mob`? (camino al default o episodio curado)."""
        return ((T < 99) & (T - 2 <= mob)) | (ep <= mob)

    def auc_numpy(y, s):
        """AUC = P(s_malo > s_bueno) + ½ P(empate), por rangos promedio (Mann–Whitney)."""
        _y = np.asarray(y).astype(bool)
        _u, _inv, _cnt = np.unique(s, return_inverse=True, return_counts=True)
        _rk = (np.cumsum(_cnt) - (_cnt - 1) / 2)[_inv]
        _n1 = _y.sum()
        _n0 = len(_y) - _n1
        return (_rk[_y].sum() - _n1 * (_n1 + 1) / 2) / (_n1 * _n0)

    def gini_np(y, s):
        return 2 * auc_numpy(y, s) - 1

    def logit_irls(X, y, iters=30):
        """Regresión logística por Newton-Raphson (IRLS) en numpy."""
        _b = np.zeros(X.shape[1])
        for _ in range(iters):
            _p = 1 / (1 + np.exp(-X @ _b))
            _H = X.T @ (X * (_p * (1 - _p))[:, None])
            _b = _b + np.linalg.solve(_H, X.T @ (y - _p))
        return _b

    _rng = np.random.default_rng(11)
    T_DEV, EP_DEV = simular_desempeno(P_REAL_DEV, np.zeros(len(P_REAL_DEV), int), sin_shock, _rng)
    BOOK_DEV = S_DEV >= CUTOFF
    Y12_DEV = (T_DEV < 99).astype(float)
    E3_DEV = mora30(T_DEV, EP_DEV, 3).astype(float)
    E6_DEV = mora30(T_DEV, EP_DEV, 6).astype(float)
    REF_GINI12 = gini_np(Y12_DEV[BOOK_DEV], PD_DEV[BOOK_DEV])
    REF_GINI_E6 = gini_np(E6_DEV[BOOK_DEV], PD_DEV[BOOK_DEV])
    _X = np.column_stack([np.ones(BOOK_DEV.sum()), np.log(PD_DEV / (1 - PD_DEV))[BOOK_DEV]])
    SAT3 = logit_irls(_X, E3_DEV[BOOK_DEV])
    SAT3_SM = sm.Logit(E3_DEV[BOOK_DEV], _X).fit(disp=0).params
    _Xa = np.column_stack([np.ones(len(PD_DEV)), np.log(PD_DEV / (1 - PD_DEV))])
    SAT6 = logit_irls(_Xa, E6_DEV)
    assert np.allclose(SAT3, SAT3_SM, atol=1e-6)

    _b = BOOK_DEV
    tabla_timing = {
        "tasa 30+ a MOB3 (aprobados)": E3_DEV[_b].mean(),
        "tasa 30+ a MOB6": E6_DEV[_b].mean(),
        "tasa 90+ a MOB6": (T_DEV[_b] <= 6).mean(),
        "tasa 90+ a 12m (target)": Y12_DEV[_b].mean(),
        "P(malo | 30+ a MOB3)": Y12_DEV[_b][E3_DEV[_b] == 1].mean(),
        "P(30+ a MOB3 | malo) = captura": E3_DEV[_b][Y12_DEV[_b] == 1].mean(),
        "P(30+ a MOB6 | malo) = captura": E6_DEV[_b][Y12_DEV[_b] == 1].mean(),
        "phi(30+ MOB3, malo) numpy": np.corrcoef(E3_DEV[_b], Y12_DEV[_b])[0, 1],
        "phi(30+ MOB3, malo) scipy": stats.pearsonr(E3_DEV[_b], Y12_DEV[_b])[0],
        "Gini 12m aprobados (referencia)": REF_GINI12,
        "Gini 12m TTD completa": gini_np(Y12_DEV, PD_DEV),
        "Gini 12m sklearn (aprobados)": 2 * roc_auc_score(Y12_DEV[_b], PD_DEV[_b]) - 1,
        "Gini sobre 30+ MOB6 (referencia)": REF_GINI_E6,
    }
    tabla_timing_df = pd.DataFrame({"valor": tabla_timing}).round(4)
    tabla_timing_df
    return (
        BOOK_DEV,
        E3_DEV,
        F_MOB,
        REF_GINI12,
        REF_GINI_E6,
        SAT3,
        SAT3_SM,
        SAT6,
        T_DEV,
        W_MOB,
        Y12_DEV,
        auc_numpy,
        gini_np,
        logit_irls,
        mora30,
        simular_desempeno,
        sin_shock,
        tabla_timing,
    )


@app.cell
def _(SAT3, coma, mo, tabla_timing):
    mo.md(coma(f"""
    **Lectura.** Entre los aprobados, la mora 30+ a MOB 3 es {tabla_timing['tasa 30+ a MOB3 (aprobados)']:.2%} y el
    malo a 12 m {tabla_timing['tasa 90+ a 12m (target)']:.2%}. La mora temprana es **precisa pero de baja
    captura**: {tabla_timing['P(malo | 30+ a MOB3)']:.0%} de los que tocan 30+ a MOB 3 termina en malo
    (más de 5 veces la tasa base), pero solo {tabla_timing['P(30+ a MOB3 | malo) = captura']:.0%} de los malos se
    ve a MOB 3 ({tabla_timing['P(30+ a MOB6 | malo) = captura']:.0%} a MOB 6). La correlación φ a nivel de crédito es
    {tabla_timing['phi(30+ MOB3, malo) numpy']:.2f}: como predictor individual es pobre; como **indicador de cohorte**
    (promedios de miles) es otra historia (§3).

    Segundo hallazgo que el tablero debe respetar: el Gini a 12 m de los **aprobados** es
    {tabla_timing['Gini 12m aprobados (referencia)']:.3f}, mientras que sobre la TTD completa es
    {tabla_timing['Gini 12m TTD completa']:.3f}. El truncamiento por el cutoff baja el Gini observable; la referencia
    del tablero tiene que ser el Gini de DEV **en la misma población** (aprobados), no el de desarrollo completo.
    Modelo satélite: logit P(30+ MOB3) = {SAT3[0]:.3f} + {SAT3[1]:.3f}·logit(PD) (IRLS numpy = statsmodels).
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Mora temprana como proxy: cuánto adelanta y cuánto se equivoca

    Experimento de cohortes: 36 cosechas de 1.500 aprobados, cada una con un shock macro propio
    $d_c\sim N(0,\sigma_d)$ en log-odds (la variación entre cosechas que el proxy debe seguir). Para cada
    cosecha se mide la tasa 30+ a MOB 3 (disponible 3 meses después de originar), la 30+ a MOB 6
    (6 meses) y el malo a 12 m (12 meses). Se ajusta en las primeras 24 cosechas la recta
    $\text{malo}_{12} = a + b\cdot\text{mora}_{30+}$ y se predice el resto.

    **El modo de falla.** Desde la cosecha 25 el producto cambia de *timing* (período de gracia, cuotas
    que parten más tarde, estacionalidad de la moto de reparto): el default ocurre $r$ meses más tarde
    con la misma PD a 12 m. El proxy temprano cae, la predicción del malo cae con él y el tablero se ve
    **mejor justo cuando no lo está**.
    """)
    return


@app.cell
def _(mo):
    retraso = mo.ui.slider(0, 4, value=2, step=1, label="Retraso del timing desde la cosecha 25 (meses)")
    sd_macro = mo.ui.slider(0.05, 0.50, value=0.25, step=0.05, label="σ del shock macro por cosecha (log-odds)")
    mo.hstack([retraso, sd_macro])
    return retraso, sd_macro


@app.cell
def _(BOOK_DEV, PD_DEV, P_REAL_DEV, W_MOB, mo, mora30, np, pd, plt, retraso, sd_macro, simular_desempeno, stats):
    _rng = np.random.default_rng(20260928)
    _idx_book = np.flatnonzero(BOOK_DEV)
    _filas = []
    for _c in range(36):
        _i = _rng.choice(_idx_book, 1500)
        _d = _rng.normal(0, sd_macro.value)
        _p = P_REAL_DEV[_i]
        _p = 1 / (1 + np.exp(-(np.log(_p / (1 - _p)) + _d)))
        _w = W_MOB
        if _c >= 24 and retraso.value > 0:
            _w = np.r_[np.zeros(retraso.value), W_MOB[:-retraso.value]]
            _w = _w / _w.sum()
        _T, _ep = simular_desempeno(_p, np.zeros(1500, int), lambda cal: np.ones(np.shape(cal)), _rng, w=_w)
        _filas.append({"cosecha": _c + 1, "shock": _d, "pd_modelo": PD_DEV[_i].mean(),
                       "mora30_mob3": mora30(_T, _ep, 3).mean(), "mora30_mob6": mora30(_T, _ep, 6).mean(),
                       "malo12": (_T < 99).mean()})
    cosechas_proxy = pd.DataFrame(_filas)
    _tr = cosechas_proxy.iloc[:24]
    _te = cosechas_proxy.iloc[24:]
    resumen_proxy = {}
    for _v in ("mora30_mob3", "mora30_mob6"):
        _b, _a = np.polyfit(_tr[_v], _tr["malo12"], 1)
        _r = stats.pearsonr(_tr[_v], _tr["malo12"])[0]
        _pred_tr = _a + _b * _tr[_v]
        _pred_te = _a + _b * _te[_v]
        resumen_proxy[_v] = {
            "adelanto_meses": 12 - (3 if _v.endswith("3") else 6),
            "corr_24_cosechas": _r,
            "R2": _r ** 2,
            "RMSE_pp_dentro": 100 * np.sqrt(np.mean((_pred_tr - _tr["malo12"]) ** 2)),
            "sesgo_pp_tras_cambio": 100 * np.mean(_pred_te - _te["malo12"]),
            "RMSE_pp_tras_cambio": 100 * np.sqrt(np.mean((_pred_te - _te["malo12"]) ** 2)),
        }
    resumen_proxy_df = pd.DataFrame(resumen_proxy).T

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.8))
    _ax[0].plot(cosechas_proxy["cosecha"], 100 * cosechas_proxy["malo12"], "k-o", ms=3, label="malo 90+ a 12 m (real)")
    for _v, _col in (("mora30_mob3", "tab:blue"), ("mora30_mob6", "tab:orange")):
        _b, _a = np.polyfit(_tr[_v], _tr["malo12"], 1)
        _ax[0].plot(cosechas_proxy["cosecha"], 100 * (_a + _b * cosechas_proxy[_v]), "--", color=_col,
                    label=f"predicho desde {_v}")
    _ax[0].axvline(24.5, color="grey", ls=":")
    _ax[0].set_xlabel("cosecha (mes de originación)")
    _ax[0].set_ylabel("tasa (%)")
    _ax[0].set_title("Proxy temprano vs malo a 12 m")
    _ax[0].legend(fontsize=7)
    _ax[1].scatter(100 * _tr["mora30_mob6"], 100 * _tr["malo12"], s=14, label="cosechas 1–24")
    _ax[1].scatter(100 * _te["mora30_mob6"], 100 * _te["malo12"], s=14, color="tab:red", label="cosechas 25–36")
    _ax[1].set_xlabel("mora 30+ a MOB 6 (%)")
    _ax[1].set_ylabel("malo 90+ a 12 m (%)")
    _ax[1].set_title("La relación se desplaza si cambia el timing")
    _ax[1].legend(fontsize=7)
    _fig.tight_layout()
    mo_proxy = mo.vstack([resumen_proxy_df.round(3), _fig])
    mo_proxy
    return cosechas_proxy, resumen_proxy


@app.cell
def _(coma, mo, resumen_proxy, retraso, sd_macro):
    _r3 = resumen_proxy["mora30_mob3"]
    _r6 = resumen_proxy["mora30_mob6"]
    mo.md(coma(f"""
    **Lectura.** Con σ del shock = {sd_macro.value:.2f}, la mora 30+ a MOB 3 adelanta **9 meses** con correlación de
    cohorte {_r3['corr_24_cosechas']:.2f} (RMSE {_r3['RMSE_pp_dentro']:.2f} pp); la de MOB 6 adelanta **6 meses** con
    correlación {_r6['corr_24_cosechas']:.2f} (RMSE {_r6['RMSE_pp_dentro']:.2f} pp). Es el trade-off de siempre: **más
    adelanto, más ruido**. Baja σ a 0,05 y la correlación se desploma: si las cosechas no varían, no hay señal
    que seguir y el proxy solo mide ruido binomial.

    Con un retraso de timing de {retraso.value} meses desde la cosecha 25, el sesgo de la predicción pasa a
    {_r3['sesgo_pp_tras_cambio']:+.2f} pp (MOB 3) y {_r6['sesgo_pp_tras_cambio']:+.2f} pp (MOB 6): el proxy **subestima** el malo.
    Regla operativa: la relación proxy → malo es un modelo más (satélite) y se re-estima cuando cambia el
    producto, el calendario de cuotas o la política de cobranza; un cambio de cobranza que «cura» más
    30+ sin cambiar el 90+ produce el mismo sesgo en sentido contrario.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Simulador de producción y tablero mensual

    Se simulan cohortes mensuales de 1.500 solicitudes desde el mes −13 hasta el 24 (las 14 previas dan
    historia madura a los indicadores rezagados desde el mes 1). Desde el mes $k$ aparece un deterioro:

    | Escenario | Qué cambia en la verdad | Parámetro (intensidad $I\in[0,1]$) |
    |---|---|---|
    | ninguno | nada (control: todo semáforo encendido es falsa alarma) | — |
    | nivel | shock macro de calendario: hazard × $e^{d}$ para todas las cosechas vivas | $d = 0{,}6\,I$ |
    | población | llegan solicitantes con más uso de línea (remuestreo con peso $e^{\gamma z}$); la PD de cada perfil no cambia | $\gamma = 0{,}8\,I$ |
    | ranking | *concept drift*: la PD real se desacopla de las variables, $\eta' = \bar\eta + (1-a)(\eta-\bar\eta) + \sqrt{1-(1-a)^2}\,s_\eta\,\varepsilon$ | $a = 0{,}8\,I$ |
    | dato roto | una fracción de solicitudes llega con `uso_linea_prom_12m = 0` (valor por defecto del feed); el comportamiento real no cambia | fracción $= 0{,}3\,I$ |
    | overrides | presión comercial: se aprueba por override una fracción mayor de los rechazados por score | $10\% \to 10\% + 50\%\,I$ |

    El desempeño solo se observa en los **aprobados** (score ≥ cutoff u override). Los 13 indicadores,
    con su familia y su rezago (meses entre originar y poder medir):

    | id | familia | indicador | rezago | 🟡 / 🔴 |
    |---|---|---|---|---|
    | D1 | datos | % de solicitudes que violan el contrato (fuera de rango, centinela, missing) | 0 | > 1% / > 3% |
    | P1 | población | PSI del score, 8 bandas, vs DEV | 0 | > 0,10 / > 0,25 |
    | P2 | población | CSI máximo de las 7 variables vs DEV | 0 | > 0,10 / > 0,25 |
    | P3 | población | mix D+E de solicitudes, pts vs DEV | 0 | > +3 / > +6 |
    | N1 | negocio | tasa de aprobación, desvío absoluto en pts vs DEV | 0 | > 5 / > 10 |
    | N2 | negocio | override rate (% de rechazados por score que se aprueban) | 0 | > 15% / > 25% |
    | A1 | adelantado | mora 30+ a MOB 3 vs satélite: p bilateral | 3 | < 0,05 / < 0,01 |
    | A2 | adelantado | 90+ a MOB 6 vs curva esperada: p bilateral | 6 | < 0,05 / < 0,01 |
    | A3 | adelantado | Gini sobre 30+ a MOB 6 (3 cosechas), caída relativa | 6–8 | > 20% / > 30% |
    | R1 | rezagado | Gini 90+ a 12 m (3 cosechas), caída relativa | 12–14 | > 20% / > 30% |
    | R2 | rezagado | binomial global 12 m (3 cosechas): p bilateral | 12–14 | < 0,05 / < 0,01 |
    | R3 | rezagado | Hosmer-Lemeshow 12 m, p simulado | 12–14 | < 0,05 / < 0,01 |
    | R4 | rezagado | binomial por banda: p mínimo de las 8 | 12–14 | < 0,05 / < 0,01 |

    R4 es la fila del curso «alguna banda amarilla / alguna roja», escrita como p mínimo. Los umbrales
    son los del curso donde existen; D1, N1 y N2 son convenciones de este notebook (no tienen base
    teórica y se declaran como tales).
    """)
    return


@app.cell
def _(mo):
    escenario = mo.ui.dropdown(["ninguno", "nivel", "población", "ranking", "dato roto", "overrides"],
                               value="nivel", label="Tipo de deterioro")
    mes_k = mo.ui.slider(2, 18, value=8, step=1, label="Mes k de inicio")
    intensidad = mo.ui.slider(0.1, 1.0, value=0.5, step=0.1, label="Intensidad I")
    semilla = mo.ui.number(1, 99_999, value=2026, step=1, label="Semilla")
    mes_corte = mo.ui.slider(1, 24, value=24, step=1, label="Mes de corte (diagnóstico y gatillos)")
    mo.vstack([mo.hstack([escenario, mes_k, intensidad]), mo.hstack([semilla, mes_corte])])
    return escenario, intensidad, mes_corte, mes_k, semilla


@app.cell
def _(
    CUTOFF,
    F_MOB,
    POOL_COD,
    POOL_COD_ROTO,
    POOL_ETA,
    POOL_P,
    POOL_W,
    POOL_WR,
    POOL_Z,
    REF_APROB,
    REF_BANDAS,
    REF_CSI,
    REF_GINI12,
    REF_GINI_E6,
    REF_MIX_DE,
    SAT3,
    VARS,
    banda_idx,
    gini_np,
    mora30,
    np,
    pd,
    pd_de_woe,
    score_de_pd,
    simular_desempeno,
    sin_shock,
    stats,
):
    ESCENARIOS = ["ninguno", "nivel", "población", "ranking", "dato roto", "overrides"]

    def simular_produccion(tipo, k, inten, seed, n_mes=1500, m0=-13, m1=24):
        """Cohortes mensuales m0..m1 con deterioro `tipo` desde el mes k. Verdad conocida."""
        _rng = np.random.default_rng(seed)
        _d = 0.6 * inten
        _mult = (lambda cal: np.where(cal >= k, np.exp(_d), 1.0)) if tipo == "nivel" else sin_shock
        _out = []
        for _m in range(m0, m1 + 1):
            _post = _m >= k
            if tipo == "población" and _post:
                _w = np.exp(0.8 * inten * POOL_Z)
                _idx = _rng.choice(len(POOL_P), n_mes, p=_w / _w.sum())
            else:
                _idx = _rng.integers(0, len(POOL_P), n_mes)
            _p = POOL_P[_idx].copy()
            _W = POOL_W[_idx].copy()
            _cod = POOL_COD[_idx].copy()
            _viol = np.zeros(n_mes, bool)
            if tipo == "dato roto" and _post:
                _viol = _rng.random(n_mes) < 0.3 * inten
                _W[_viol] = POOL_WR[_idx[_viol]]
                _cod[_viol, 0] = POOL_COD_ROTO[_idx[_viol]]
            if tipo == "ranking" and _post:
                _a = 0.8 * inten
                _mu, _sd = POOL_ETA.mean(), POOL_ETA.std()
                _e2 = (_mu + (1 - _a) * (POOL_ETA[_idx] - _mu)
                       + np.sqrt(1 - (1 - _a) ** 2) * _sd * _rng.standard_normal(n_mes))
                _p = 1 / (1 + np.exp(-_e2))
            _pdm = pd_de_woe(_W)
            _s = score_de_pd(_pdm)
            _tasa_ov = 0.10 + (0.5 * inten if (tipo == "overrides" and _post) else 0.0)
            _ap = _s >= CUTOFF
            _ov = (~_ap) & (_rng.random(n_mes) < _tasa_ov)
            _T, _ep = simular_desempeno(_p, np.full(n_mes, _m), _mult, _rng)
            _df = pd.DataFrame({"cohorte": _m, "pd_real": _p, "pd_modelo": _pdm, "score": _s,
                                "banda": banda_idx(_s), "aprueba": _ap, "override": _ov,
                                "T": _T, "ep": _ep, "viol": _viol})
            for _j in range(len(VARS)):
                _df[f"c{_j}"] = _cod[:, _j]
            _out.append(_df)
        return pd.concat(_out, ignore_index=True)

    def psi_numpy(a, e, eps=1e-6):
        """PSI = Σ (a−e)·ln(a/e) sobre proporciones (ε solo para ceros, luego renormaliza)."""
        _a = np.clip(np.asarray(a, float), eps, None)
        _e = np.clip(np.asarray(e, float), eps, None)
        _a, _e = _a / _a.sum(), _e / _e.sum()
        return float(np.sum((_a - _e) * np.log(_a / _e)))

    def psi_scipy(a, e, eps=1e-6):
        """PSI = KL(a‖e) + KL(e‖a) (divergencia de Jeffreys) con scipy.stats.entropy."""
        _a = np.clip(np.asarray(a, float), eps, None)
        _e = np.clip(np.asarray(e, float), eps, None)
        return float(stats.entropy(_a, _e) + stats.entropy(_e, _a))

    def z_poisson_binomial(obs, p):
        """(z, E, V) para una suma de Bernoulli independientes con probabilidades p."""
        _E = float(np.sum(p))
        _V = float(np.sum(p * (1 - p)))
        return (obs - _E) / np.sqrt(_V), _E, _V

    def p_bilateral_normal(obs, p):
        _z = z_poisson_binomial(obs, p)[0]
        return float(2 * stats.norm.sf(abs(_z)))

    def p_binomial_numpy(k, n, p):
        """p bilateral exacto por el método de «probabilidades ≤ la observada» (el de scipy.binomtest)."""
        _j = np.arange(n + 1)
        _lc = np.concatenate([[0.0], np.cumsum(np.log(np.arange(n, 0, -1)) - np.log(np.arange(1, n + 1)))])
        _lp = _lc + _j * np.log(p) + (n - _j) * np.log1p(-p)
        _pmf = np.exp(_lp - _lp.max())
        _pmf = _pmf / _pmf.sum()
        return float(min(1.0, _pmf[_pmf <= _pmf[int(k)] * (1 + 1e-7)].sum()))

    def hosmer_lemeshow(y, p, rng, g=10, B=200):
        """HL con g grupos por PD; p simulado bajo H0 (bootstrap paramétrico)."""
        _o = np.argsort(p, kind="stable")
        _grupos = np.array_split(_o, g)
        _E = np.array([p[i].sum() for i in _grupos])
        _den = np.array([p[i].sum() * (1 - p[i].mean()) for i in _grupos])
        _h = float(np.sum((np.array([y[i].sum() for i in _grupos]) - _E) ** 2 / _den))
        _sims = rng.random((B, len(p))) < p[None, :]
        _O = np.column_stack([_sims[:, i].sum(axis=1) for i in _grupos])
        _hs = ((_O - _E[None, :]) ** 2 / _den[None, :]).sum(axis=1)
        return _h, float((1 + (_hs >= _h).sum()) / (B + 1)), float(stats.chi2.sf(_h, g))

    INDICADORES = pd.DataFrame([
        ("D1", "datos", "Violaciones de contrato (%)", 0, "mayor", 1.0, 3.0),
        ("P1", "población", "PSI score 8 bandas", 0, "mayor", 0.10, 0.25),
        ("P2", "población", "CSI máximo", 0, "mayor", 0.10, 0.25),
        ("P3", "población", "Mix D+E (pts vs DEV)", 0, "mayor", 3.0, 6.0),
        ("N1", "negocio", "Aprobación |desvío| (pts)", 0, "mayor", 5.0, 10.0),
        ("N2", "negocio", "Override rate (%)", 0, "mayor", 15.0, 25.0),
        ("A1", "adelantado", "Mora 30+ MOB3: p", 3, "menor", 0.05, 0.01),
        ("A2", "adelantado", "90+ a MOB6: p", 6, "menor", 0.05, 0.01),
        ("A3", "adelantado", "Gini 30+ MOB6: caída %", 6, "mayor", 20.0, 30.0),
        ("R1", "rezagado", "Gini 12m: caída %", 12, "mayor", 20.0, 30.0),
        ("R2", "rezagado", "Binomial global 12m: p", 12, "menor", 0.05, 0.01),
        ("R3", "rezagado", "HL 12m: p simulado", 12, "menor", 0.05, 0.01),
        ("R4", "rezagado", "Bandas: p mínimo", 12, "menor", 0.05, 0.01),
    ], columns=["id", "familia", "indicador", "rezago", "direccion", "amarillo", "rojo"]).set_index("id")

    def estado(valor, direccion, amarillo, rojo):
        """0 verde · 1 amarillo · 2 rojo · NaN sin dato."""
        if valor is None or not np.isfinite(valor):
            return np.nan
        if direccion == "mayor":
            return 2.0 if valor > rojo else (1.0 if valor > amarillo else 0.0)
        return 2.0 if valor < rojo else (1.0 if valor < amarillo else 0.0)

    def valores_mes(sim, M, rng, B_hl=200):
        """Los 13 indicadores del mes de reporte M (cada uno con los datos que ya existen en M)."""
        _v = {}
        _book = (sim["aprueba"] | sim["override"]).to_numpy()
        _coh = sim["cohorte"].to_numpy()
        _a = sim[_coh == M]
        _v["D1"] = 100 * _a["viol"].mean()
        _v["P1"] = psi_numpy(np.bincount(_a["banda"], minlength=8) / len(_a), REF_BANDAS)
        _v["P2"] = max(psi_numpy(np.bincount(_a[f"c{j}"], minlength=len(REF_CSI[x])) / len(_a), REF_CSI[x])
                       for j, x in enumerate(VARS))
        _v["P3"] = 100 * ((_a["banda"] <= 1).mean() - REF_MIX_DE)
        _v["N1"] = 100 * abs(_a["aprueba"].mean() - REF_APROB)
        _v["N2"] = 100 * _a["override"].sum() / max(1, (~_a["aprueba"]).sum())
        _b = sim[(_coh == M - 3) & _book]
        _l = np.log(_b["pd_modelo"] / (1 - _b["pd_modelo"])).to_numpy()
        _pe = 1 / (1 + np.exp(-(SAT3[0] + SAT3[1] * _l)))
        _v["A1"] = p_bilateral_normal(mora30(_b["T"].to_numpy(), _b["ep"].to_numpy(), 3).sum(), _pe)
        _b = sim[(_coh == M - 6) & _book]
        _pe = 1 - (1 - _b["pd_modelo"].to_numpy()) ** F_MOB[5]
        _v["A2"] = p_bilateral_normal((_b["T"].to_numpy() <= 6).sum(), _pe)
        _b = sim[(_coh >= M - 8) & (_coh <= M - 6) & _book]
        _v["A3"] = 100 * (1 - gini_np(mora30(_b["T"].to_numpy(), _b["ep"].to_numpy(), 6),
                                      _b["pd_modelo"].to_numpy()) / REF_GINI_E6)
        _b = sim[(_coh >= M - 14) & (_coh <= M - 12) & _book]
        _y = (_b["T"].to_numpy() < 99).astype(float)
        _p = _b["pd_modelo"].to_numpy()
        _v["R1"] = 100 * (1 - gini_np(_y, _p) / REF_GINI12)
        _v["R2"] = p_bilateral_normal(_y.sum(), _p)
        _v["R3"] = hosmer_lemeshow(_y, _p, rng, B=B_hl)[1]
        _bd = _b["banda"].to_numpy()
        _v["R4"] = min(p_binomial_numpy(_y[_bd == j].sum(), int((_bd == j).sum()), _p[_bd == j].mean())
                       for j in range(8) if (_bd == j).sum() > 0)
        return _v

    def calcular_tablero(sim, meses, seed=0, B_hl=200):
        """Devuelve (valores, estados): DataFrames indicador × mes."""
        _rng = np.random.default_rng(seed)
        _val = pd.DataFrame({M: valores_mes(sim, M, _rng, B_hl) for M in meses}).loc[INDICADORES.index]
        _est = _val.copy()
        for _i in INDICADORES.index:
            _r = INDICADORES.loc[_i]
            _est.loc[_i] = [estado(x, _r["direccion"], _r["amarillo"], _r["rojo"]) for x in _val.loc[_i]]
        return _val, _est.astype(float)
    return (
        ESCENARIOS,
        INDICADORES,
        calcular_tablero,
        estado,
        hosmer_lemeshow,
        p_bilateral_normal,
        p_binomial_numpy,
        psi_numpy,
        psi_scipy,
        simular_produccion,
        z_poisson_binomial,
    )


@app.cell
def _(calcular_tablero, escenario, intensidad, mes_k, semilla, simular_produccion):
    sim_actual = simular_produccion(escenario.value, mes_k.value, intensidad.value, int(semilla.value))
    val_actual, est_actual = calcular_tablero(sim_actual, range(1, 25), seed=int(semilla.value))
    return est_actual, sim_actual, val_actual


@app.cell
def _(INDICADORES, ListedColormap, escenario, est_actual, intensidad, mes_k, np, plt):
    _cmap = ListedColormap(["#2e9e44", "#f2c230", "#d7301f"])
    _fig, _ax = plt.subplots(figsize=(11, 4.6))
    _M = np.ma.masked_invalid(est_actual.to_numpy())
    _ax.imshow(_M, cmap=_cmap, vmin=0, vmax=2, aspect="auto")
    _ax.set_yticks(range(len(INDICADORES)))
    _ax.set_yticklabels([f"{i} · {INDICADORES.loc[i, 'indicador']}" for i in INDICADORES.index], fontsize=8)
    _ax.set_xticks(range(24))
    _ax.set_xticklabels(range(1, 25), fontsize=8)
    _ax.set_xlabel("mes de reporte")
    _ax.axvline(mes_k.value - 1.5, color="k", lw=2)
    for _f in (5.5, 8.5):
        _ax.axhline(_f, color="white", lw=2)
    _ax.set_title(f"Tablero mensual · escenario «{escenario.value}» desde el mes {mes_k.value} (I = {intensidad.value:.1f})"
                  " · verde / amarillo / rojo")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(INDICADORES, est_actual, mes_k, mo, np, pd, val_actual):
    _emoji = {0.0: "🟢", 1.0: "🟡", 2.0: "🔴"}
    tablero_emoji = est_actual.apply(lambda col: col.map(lambda x: _emoji.get(x, "⚪")))
    tablero_emoji.index = [f"{i} {INDICADORES.loc[i, 'indicador']}" for i in est_actual.index]
    tablero_emoji.columns = [f"m{c}" for c in est_actual.columns]

    _filas = []
    for _i in INDICADORES.index:
        _s = est_actual.loc[_i]
        _post = _s[_s.index >= mes_k.value]
        _pre = _s[_s.index < mes_k.value]
        _pers = (_s >= 1) & (_s.shift(1) >= 1)
        _pp = _pers[_pers.index >= mes_k.value]
        _filas.append({
            "id": _i, "familia": INDICADORES.loc[_i, "familia"], "rezago": INDICADORES.loc[_i, "rezago"],
            "primer_🟡+": int(_post.index[_post >= 1][0]) if (_post >= 1).any() else np.nan,
            "primer_🔴": int(_post.index[_post >= 2][0]) if (_post >= 2).any() else np.nan,
            "primer_persistente": int(_pp.index[_pp][0]) if _pp.any() else np.nan,
            "falsas_alarmas_antes_de_k": int((_pre >= 1).sum()),
            "valor_mes_24": val_actual.loc[_i, 24],
        })
    primeros = pd.DataFrame(_filas).set_index("id")
    primeros["demora_🟡"] = primeros["primer_🟡+"] - mes_k.value
    primeros["demora_persistente"] = primeros["primer_persistente"] - mes_k.value
    mo.vstack([
        mo.md("**Semáforos por mes** (⚪ = sin dato):"),
        tablero_emoji,
        mo.md("**¿Qué se enciende primero?** Demora = meses desde k hasta el primer amarillo (o el primer par de meses seguidos):"),
        primeros.sort_values(["demora_persistente", "demora_🟡"]).round(4),
    ])
    return primeros, tablero_emoji


@app.cell
def _(coma, escenario, mes_k, mo, np, primeros):
    _ok = primeros.dropna(subset=["primer_persistente"]).sort_values("primer_persistente")
    _pre = int(primeros["falsas_alarmas_antes_de_k"].sum())
    if escenario.value == "ninguno":
        _txt = (f"Escenario de control: no hay deterioro, así que **todo** amarillo es falsa alarma. Antes de k hubo {_pre} "
                f"semáforos 🟡/🔴 sueltos y después {int(np.nansum(primeros['primer_🟡+'].notna()))} indicadores tuvieron al menos uno. "
                "Mira qué filas los producen: casi siempre las de p-valor (A1, A2, R2–R4), que por construcción se encienden "
                "~5% de los meses cada una. Los de convención con n grande (PSI, CSI) casi nunca.")
    elif len(_ok) == 0:
        _txt = (f"Con esta intensidad ningún indicador se enciende dos meses seguidos: el deterioro plantado en k = {mes_k.value} "
                "es invisible para el tablero mensual. Sube la intensidad o mira el CUSUM de §7.")
    else:
        _prim = _ok.index[0]
        _orden = ", ".join(f"{i} (+{int(_ok.loc[i, 'demora_persistente'])})" for i in _ok.index)
        _txt = (f"Primero en encenderse de forma persistente: **{_prim}** ({_ok.loc[_prim, 'familia']}) en el mes "
                f"{int(_ok.loc[_prim, 'primer_persistente'])}, {int(_ok.loc[_prim, 'demora_persistente'])} mes(es) después de k. "
                f"Orden completo: {_orden}. "
                f"Antes de k hubo {_pre} semáforos sueltos (falsas alarmas).")
    mo.md(coma(f"""
    **Lectura.** {_txt}

    Pruebas que vale la pena hacer con los controles: (i) «dato roto» — D1 se enciende **el mismo mes k** y los
    indicadores de desempeño casi no se mueven con 15% de filas rotas: solo el contrato ve el problema;
    (ii) «nivel» — A1/A2 (rezago 3–6) y, antes de lo que uno esperaría, R2/R3: el shock es de **calendario** y
    golpea los últimos meses de vida de cosechas viejas que ya estaban por madurar; (iii) «ranking» — A3
    (Gini sobre mora temprana) adelanta ~6 meses a R1; (iv) «overrides» — el Gini observado **sube** (A3 y
    R1 con caída negativa) porque los aprobados se vuelven más heterogéneos: un Gini que mejora no es
    un modelo que mejora; (v) «población» — PSI/CSI/mix y aprobación rojos con nivel y orden verdes.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Diagnóstico por patrón: árbol de decisión

    Un semáforo suelto no es diagnóstico. El árbol lee **familias**, con persistencia (un indicador
    cuenta si está 🟡/🔴 este mes *y* el anterior) y con un orden deliberado:

    1. **Datos** primero: si el contrato está roto, todos los demás indicadores miden basura. Acción:
       contingencia de datos, no recalibrar.
    2. **Orden** (A3 o R1 persistente): el ranking se rompió. Acción: re-desarrollo, *después* de
       descartar datos y política.
    3. **Nivel** (≥ 2 de A1, A2, R2, R3, R4 persistentes) con orden verde: recalibrar δ.
    4. **Población** (≥ 2 de P1, P2, P3 persistentes) con nivel verde: el modelo sigue válido;
       revisar estrategia y cutoff (M17). Si además hay nivel: recalibrar y buscar variables fuera del
       modelo que se movieron.
    5. **Negocio** (N1 o N2 persistente) sin nada más: es la política, no el modelo.
    6. Nada: vigilancia normal.

    El «≥ 2» en nivel y población es la regla del curso hecha operativa: *amarillos coherentes* pesan;
    un amarillo solitario de un test al 5% es ruido esperado.
    """)
    return


@app.cell
def _(np, pd):
    FAMILIAS = {"datos": ["D1"], "orden": ["A3", "R1"], "nivel": ["A1", "A2", "R2", "R3", "R4"],
                "población": ["P1", "P2", "P3"], "negocio": ["N1", "N2"]}
    MINIMO_FAMILIA = {"datos": 1, "orden": 1, "nivel": 2, "población": 2, "negocio": 1}
    ACCION = {
        "DATOS ROTOS": "contingencia de datos: bloquear scoring automático de filas con violación, corregir feed; NO recalibrar",
        "ORDEN ROTO": "re-desarrollo (tras descartar datos y política); mientras, recalibración provisional y cutoff conservador",
        "NIVEL + POBLACIÓN": "recalibrar δ y buscar variables fuera del modelo que se movieron",
        "NIVEL": "recalibrar δ con muestra reciente madura declarada; vigilancia reforzada",
        "POBLACIÓN": "modelo válido: revisar estrategia/cutoff y capacidad; vigilancia reforzada",
        "POLÍTICA / NEGOCIO": "revisar overrides y aprobación con el dueño comercial; no tocar el modelo",
        "SIN SEÑAL": "vigilancia normal",
    }

    def familias_encendidas(est, M):
        """Familias con al menos MINIMO_FAMILIA indicadores 🟡/🔴 en M y en M−1."""
        _pers = (est[M].fillna(0) >= 1) & (est[M - 1].fillna(0) >= 1) if (M - 1) in est.columns \
            else (est[M].fillna(0) >= 1)
        return {f: int(_pers[ids].sum()) >= MINIMO_FAMILIA[f] for f, ids in FAMILIAS.items()}

    def diagnosticar(est, M):
        _f = familias_encendidas(est, M)
        if _f["datos"]:
            return "DATOS ROTOS"
        if _f["orden"]:
            return "ORDEN ROTO"
        if _f["nivel"] and _f["población"]:
            return "NIVEL + POBLACIÓN"
        if _f["nivel"]:
            return "NIVEL"
        if _f["población"]:
            return "POBLACIÓN"
        if _f["negocio"]:
            return "POLÍTICA / NEGOCIO"
        return "SIN SEÑAL"

    def diagnostico_serie(est):
        return pd.DataFrame({"diagnóstico": [diagnosticar(est, M) for M in est.columns]},
                            index=pd.Index(est.columns, name="mes"))
    _ = np
    return ACCION, FAMILIAS, diagnosticar, diagnostico_serie, familias_encendidas


@app.cell
def _(
    ACCION,
    coma,
    diagnosticar,
    diagnostico_serie,
    est_actual,
    familias_encendidas,
    mes_corte,
    mo,
):
    diag_actual = diagnostico_serie(est_actual)
    _d = diagnosticar(est_actual, mes_corte.value)
    _f = familias_encendidas(est_actual, mes_corte.value)
    mo.vstack([
        mo.md(coma(f"""**Mes de corte {mes_corte.value}.** Familias encendidas (persistentes):
        {', '.join(k for k, v in _f.items() if v) or 'ninguna'} → diagnóstico **{_d}** → acción: {ACCION[_d]}.""")),
        diag_actual.T.rename(columns=lambda c: f"m{c}"),
    ])
    return (diag_actual,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Multiplicidad y falsas alarmas

    La cuenta del curso: con $m$ tests independientes a nivel $\alpha$, $P(\ge 1\ \text{amarillo})=1-(1-\alpha)^m$
    (la lámina 28 escribe $1-0{,}95^8=34\%$ con 9 indicadores en el tablero: con 9 sería 37%; con 8 tests
    de p-valor, 34%). En un tablero real hay tres complicaciones: (i) no todos los indicadores son tests
    con $\alpha$ calibrado (el PSI con umbral 0,10 y $n$ grande tiene $\alpha\approx 0$), (ii) los indicadores
    comparten datos y están correlacionados, (iii) R4 ya es un «mínimo de 8 p-valores», cuyo $\alpha$ real
    es $1-0{,}95^8$. Aquí se mide todo por simulación: el tablero bajo «ninguno» con varias semillas.
    """)
    return


@app.cell
def _(mo):
    n_semillas_nulas = mo.ui.slider(2, 12, value=4, step=1, label="Semillas del escenario nulo")
    n_semillas_nulas
    return (n_semillas_nulas,)


@app.cell
def _(INDICADORES, calcular_tablero, diagnosticar, n_semillas_nulas, np, pd, simular_produccion):
    _vals, _ests, _diags = [], [], []
    for _s in range(n_semillas_nulas.value):
        _sim = simular_produccion("ninguno", 99, 0.0, 5000 + _s)
        _v, _e = calcular_tablero(_sim, range(1, 25), seed=_s, B_hl=100)
        _vals.append(_v)
        _ests.append(_e)
        _diags += [diagnosticar(_e, M) for M in range(2, 25)]
    _E = np.stack([e.to_numpy() for e in _ests])          # semilla × indicador × mes
    _V = np.stack([v.to_numpy() for v in _vals])
    _pers = (_E[:, :, 1:] >= 1) & (_E[:, :, :-1] >= 1)
    nulo = pd.DataFrame({
        "tasa_🟡+_mensual": (_E >= 1).mean(axis=(0, 2)),
        "tasa_🔴_mensual": (_E >= 2).mean(axis=(0, 2)),
        "tasa_persistente": _pers.mean(axis=(0, 2)),
        "p95_nulo": np.nanpercentile(_V, 95, axis=(0, 2)),
        "p5_nulo": np.nanpercentile(_V, 5, axis=(0, 2)),
        "umbral_🟡": INDICADORES["amarillo"].to_numpy(),
    }, index=INDICADORES.index)
    FALSA_ALARMA_MES = float((_E >= 1).any(axis=1).mean())
    FALSA_ALARMA_PERSIST = float(_pers.any(axis=1).mean())
    DIAG_FALSO = float(np.mean([d != "SIN SEÑAL" for d in _diags]))
    ALFA_R4_TEORICO = 1 - 0.95 ** 8
    nulo.round(4)
    return ALFA_R4_TEORICO, DIAG_FALSO, FALSA_ALARMA_MES, FALSA_ALARMA_PERSIST, nulo


@app.cell
def _(
    ALFA_R4_TEORICO,
    DIAG_FALSO,
    FALSA_ALARMA_MES,
    FALSA_ALARMA_PERSIST,
    coma,
    mo,
    nulo,
):
    mo.md(coma(f"""
    **Lectura.** Bajo «ninguno»:

    - **P(≥ 1 semáforo 🟡/🔴 en un mes) = {FALSA_ALARMA_MES:.0%}.** La fórmula ingenua con los 4 tests de p-valor
      simples más R4 (que vale por 8) da $1-0{{,}}95^{{12}}$ = {1 - 0.95 ** 12:.0%}; la simulación queda cerca porque los tests
      rezagados comparten cohortes y no son independientes. El tablero **todos los meses** tiene algo amarillo: por eso
      el curso dice que un tablero todo verde es sospechoso, y por eso nadie debe actuar sobre un color suelto.
    - R4 (p mínimo de 8 bandas) se enciende {nulo.loc['R4', 'tasa_🟡+_mensual']:.0%} de los meses; su α teórico es
      {ALFA_R4_TEORICO:.0%}. Con corrección de Bonferroni (umbral 0,05/8) volvería a ~5%. Es la fila más ruidosa del tablero del curso.
    - **Persistencia** (dos meses seguidos): P(≥ 1 indicador persistente) = {FALSA_ALARMA_PERSIST:.0%}; el árbol de
      diagnóstico (que exige familias) da un diagnóstico falso en {DIAG_FALSO:.0%} de los meses.
    - **Umbral por convención vs por distribución nula.** El p95 nulo del PSI es {nulo.loc['P1', 'p95_nulo']:.3f} y el
      del CSI máximo {nulo.loc['P2', 'p95_nulo']:.3f}, contra el umbral 0,10: con 1500 solicitudes al mes la convención es
      muy holgada (casi sin falsas alarmas, pero ciega a corrimientos chicos; ver M08). La caída relativa del Gini a
      12 m tiene p95 nulo de {nulo.loc['R1', 'p95_nulo']:.1f}% contra el umbral 20%: aquí convención y nula casi coinciden,
      por azar del tamaño de esta cartera. Con una cartera de motos de 300 créditos al mes no coincidirían.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Monitoreo secuencial: Shewhart, EWMA y CUSUM

    El semáforo mensual es un gráfico de Shewhart con límites a 1,96σ: mira **solo el último mes**.
    Un deterioro chico y sostenido se acumula y un test de un mes no lo ve. Sea
    $z_t=(O_t-E_t)/\sqrt{V_t}$ la mora temprana estandarizada contra el modelo satélite
    (Poisson-binomial: $E_t=\sum p_i$, $V_t=\sum p_i(1-p_i)$).

    - **Shewhart**: alarma si $z_t > L$. ARL $=1/P(z>L)$. Con $L=1{,}96$ bilateral, ARL₀ = 20 meses.
    - **EWMA** (Roberts 1959): $W_t=\lambda z_t+(1-\lambda)W_{t-1}$, $W_0=0$; alarma si
      $|W_t|>L\sqrt{\lambda/(2-\lambda)\,[1-(1-\lambda)^{2t}]}$.
    - **CUSUM** (Page 1954): $C_t=\max(0,\,C_{t-1}+z_t-k)$; alarma si $C_t>h$. Con $k=\delta/2$ es la
      aproximación del test de razón de verosimilitud secuencial para un salto de $\delta$ σ.
    - **CUSUM ajustado por riesgo** (Steiner et al. 2000): con cada crédito su PD, el incremento es la
      log-razón de verosimilitud de «odds × R» contra «odds × 1»:
      $W_t=O_t\ln R-\sum_i\ln(1-p_i+Rp_i)$, $S_t=\max(0,S_{t-1}+W_t)$.

    **ARL** (*average run length*): meses promedio hasta la alarma. ARL₀ (sin cambio) mide falsas
    alarmas; ARL₁ (con cambio δ) mide demora de detección. Se calcula de tres formas: cadena de Markov
    (Brook & Evans 1972), simulación y, para CUSUM, la aproximación de Siegmund. Se contrasta con la
    tabla clásica de Montgomery ($k=0{,}5$; $h=4$ y $h=5$, bilateral).
    """)
    return


@app.cell
def _(np, pd, stats):
    def ewma_numpy(z, lam):
        _w = np.zeros(len(z))
        _prev = 0.0
        for _t, _x in enumerate(z):
            _prev = lam * _x + (1 - lam) * _prev
            _w[_t] = _prev
        return _w

    def ewma_pandas(z, lam):
        """pandas.ewm con adjust=False arranca en el primer dato: se antepone W0 = 0."""
        return pd.Series(np.r_[0.0, z]).ewm(alpha=lam, adjust=False).mean().to_numpy()[1:]

    def cusum_numpy(z, k):
        _c = np.zeros(len(z))
        _prev = 0.0
        for _t, _x in enumerate(z):
            _prev = max(0.0, _prev + _x - k)
            _c[_t] = _prev
        return _c

    def cusum_page(z, k):
        """Forma original de Page: C_t = S_t − min(0, min_{j≤t} S_j), con S_t = Σ (z − k)."""
        _S = np.cumsum(np.asarray(z) - k)
        return _S - np.minimum(0.0, np.minimum.accumulate(_S))

    def arl_cusum_markov(delta, k, h, m=300):
        """ARL del CUSUM superior (arranque en 0) por cadena de Markov de Brook & Evans."""
        _w = h / m
        _c = np.arange(m) * _w
        _up = (np.arange(m) + 0.5) * _w
        _P = stats.norm.cdf(_up[None, :] - _c[:, None] + k - delta)
        _P[:, 1:] = _P[:, 1:] - stats.norm.cdf(_up[None, :-1] - _c[:, None] + k - delta)
        return float(np.linalg.solve(np.eye(m) - _P, np.ones(m))[0])

    def arl_cusum_bilateral(delta, k, h, m=300):
        """1/ARL = 1/ARL⁺ + 1/ARL⁻ (combinación usual de los dos CUSUM unilaterales)."""
        return 1 / (1 / arl_cusum_markov(delta, k, h, m) + 1 / arl_cusum_markov(-delta, k, h, m))

    def arl_siegmund(delta, k, h):
        _D = delta - k
        _b = h + 1.166
        return _b ** 2 if abs(_D) < 1e-9 else (np.exp(-2 * _D * _b) + 2 * _D * _b - 1) / (2 * _D ** 2)

    def arl_ewma_markov(delta, lam, L, m=301):
        """ARL del EWMA bilateral con límites asintóticos (cadena de Markov)."""
        _H = L * np.sqrt(lam / (2 - lam))
        _ed = np.linspace(-_H, _H, m + 1)
        _c = (_ed[:-1] + _ed[1:]) / 2
        _mu = (1 - lam) * _c
        _P = (stats.norm.cdf((_ed[None, 1:] - _mu[:, None]) / lam - delta)
              - stats.norm.cdf((_ed[None, :-1] - _mu[:, None]) / lam - delta))
        return float(np.linalg.solve(np.eye(m) - _P, np.ones(m))[np.argmin(np.abs(_c))])

    def arl_shewhart(delta, L, bilateral=True):
        _p = stats.norm.sf(L - delta) + (stats.norm.cdf(-L - delta) if bilateral else 0.0)
        return float(1 / _p)

    def arl_simulado(delta, k, h, rng, n_rutas=2000, t_max=4000, bilateral=True):
        """ARL por simulación del CUSUM (bilateral si se pide), vectorizado por bloques."""
        _rl = np.full(n_rutas, t_max, float)
        _cp = np.zeros(n_rutas)
        _cm = np.zeros(n_rutas)
        _vivo = np.ones(n_rutas, bool)
        for _t in range(1, t_max + 1):
            _x = rng.standard_normal(n_rutas) + delta
            _cp = np.maximum(0, _cp + _x - k)
            _cm = np.maximum(0, _cm - _x - k) if bilateral else _cm
            _alarma = _vivo & ((_cp > h) | (_cm > h))
            _rl[_alarma] = _t
            _vivo &= ~_alarma
            if not _vivo.any():
                break
        return float(_rl.mean()), float(_rl.std(ddof=1) / np.sqrt(n_rutas))
    return (
        arl_cusum_bilateral,
        arl_cusum_markov,
        arl_ewma_markov,
        arl_shewhart,
        arl_siegmund,
        arl_simulado,
        cusum_numpy,
        cusum_page,
        ewma_numpy,
        ewma_pandas,
    )


@app.cell
def _(arl_cusum_bilateral, arl_cusum_markov, arl_ewma_markov, arl_shewhart, arl_siegmund, arl_simulado, np, optimize, pd):
    # Tabla clásica (Montgomery, CUSUM tabular k = 1/2, bilateral): δ → (h = 4, h = 5)
    MONTGOMERY = {0.0: (168, 465), 0.5: (26.6, 38.0), 1.0: (8.38, 10.4), 2.0: (3.34, 4.01), 3.0: (2.19, 2.57)}
    _rng = np.random.default_rng(1954)
    # a IGUAL tasa de falsas alarmas que el semáforo (ARL0 = 20 meses, bilateral)
    H20 = optimize.brentq(lambda h: arl_cusum_bilateral(0.0, 0.5, h, m=200) - 20, 0.3, 4)
    L20 = optimize.brentq(lambda x: arl_ewma_markov(0.0, 0.2, x, m=201) - 20, 1.0, 2.9)
    _filas = []
    for _d, (_t4, _t5) in MONTGOMERY.items():
        _sim4 = arl_simulado(_d, 0.5, 4, _rng, n_rutas=1500) if _d > 0 else (np.nan, np.nan)
        _filas.append({"δ (σ)": _d, "tabla h=4": _t4, "Markov h=4": arl_cusum_bilateral(_d, 0.5, 4),
                       "simulado h=4": _sim4[0], "±EE": _sim4[1],
                       "Siegmund unilateral h=4": arl_siegmund(_d, 0.5, 4),
                       "Markov unilateral h=4": arl_cusum_markov(_d, 0.5, 4),
                       "tabla h=5": _t5, "Markov h=5": arl_cusum_bilateral(_d, 0.5, 5),
                       "EWMA λ=0,2 L=2,962": arl_ewma_markov(_d, 0.2, 2.962),
                       "Shewhart 3σ": arl_shewhart(_d, 3.0),
                       "semáforo 1,96σ": arl_shewhart(_d, 1.96),
                       "CUSUM ARL0=20": arl_cusum_bilateral(_d, 0.5, H20, m=200),
                       "EWMA ARL0=20": arl_ewma_markov(_d, 0.2, L20, m=201)})
    tabla_arl = pd.DataFrame(_filas).set_index("δ (σ)")
    tabla_arl.round(2)
    return H20, L20, MONTGOMERY, tabla_arl


@app.cell
def _(H20, L20, coma, mo, tabla_arl):
    mo.md(coma(f"""
    **Lectura.** La cadena de Markov reproduce la tabla de Montgomery (h = 4: ARL₀ {tabla_arl.loc[0.0, 'Markov h=4']:.0f} vs 168;
    δ = 1: {tabla_arl.loc[1.0, 'Markov h=4']:.2f} vs 8,38) y la simulación coincide dentro de su error estándar. La tabla es
    **bilateral**; para crédito lo que importa es el CUSUM superior (deterioro), cuyo ARL₀ es el doble
    ({tabla_arl.loc[0.0, 'Markov unilateral h=4']:.0f} meses) y cuyo ARL₁ es prácticamente igual.
    Ahora la columna que incomoda: el **semáforo mensual a 1,96σ tiene ARL₀ = 20 meses**, una falsa alarma cada año y
    medio *por indicador*. Es rápido ({tabla_arl.loc[0.5, 'semáforo 1,96σ']:.1f} meses para 0,5σ, contra
    {tabla_arl.loc[0.5, 'Markov h=4']:.1f} del CUSUM h = 4) porque compra velocidad con falsas alarmas. La comparación
    justa es a **igual ARL₀**: un CUSUM con h = {H20:.2f} o un EWMA con L = {L20:.2f} también tienen ARL₀ = 20 y detectan 0,5σ en
    {tabla_arl.loc[0.5, 'CUSUM ARL0=20']:.1f} y {tabla_arl.loc[0.5, 'EWMA ARL0=20']:.1f} meses (≈ 20–30% antes que el semáforo); para
    saltos grandes (2–3σ) el semáforo es igual o mejor. Conclusión honesta: acumular gana en desvíos **chicos y sostenidos**
    (el deterioro de nivel típico); la palanca más grande no es el tipo de gráfico sino **qué ARL₀ se está dispuesto a pagar**,
    y el semáforo del curso lo fija implícitamente en 20 meses por indicador.
    """))
    return


@app.cell
def _(mo):
    n_book = mo.ui.slider(200, 3000, value=1200, step=100, label="Créditos aprobados por mes")
    p0_temprana = mo.ui.slider(0.01, 0.10, value=0.03, step=0.005, label="Tasa base de mora 30+ a MOB3")
    deterioro_rel = mo.ui.slider(5, 60, value=20, step=5, label="Deterioro relativo a detectar (%)")
    mo.hstack([n_book, p0_temprana, deterioro_rel])
    return deterioro_rel, n_book, p0_temprana


@app.cell
def _(H20, arl_cusum_bilateral, arl_cusum_markov, arl_ewma_markov, arl_shewhart, deterioro_rel, mo, n_book, np, p0_temprana, pd, plt):
    _sd = np.sqrt(p0_temprana.value * (1 - p0_temprana.value) / n_book.value)
    _rels = np.arange(0, 65, 5) / 100
    _deltas = p0_temprana.value * _rels / _sd
    _curvas = pd.DataFrame({
        "deterioro_%": 100 * _rels, "δ_sigmas": _deltas,
        "semáforo 1,96σ (bilateral)": [arl_shewhart(d, 1.96) for d in _deltas],
        "Shewhart 3σ": [arl_shewhart(d, 3.0) for d in _deltas],
        "EWMA λ=0,2 L=2,86": [arl_ewma_markov(d, 0.2, 2.86, m=151) for d in _deltas],
        "CUSUM k=0,5 h=4 (sup.)": [arl_cusum_markov(d, 0.5, 4, m=150) for d in _deltas],
        "CUSUM a ARL₀=20 (h=H20)": [arl_cusum_bilateral(d, 0.5, H20, m=150) for d in _deltas],
    }).set_index("deterioro_%")
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    for _c in _curvas.columns[1:]:
        _ax.plot(_curvas.index, _curvas[_c], marker="o", ms=3, label=_c)
    _ax.set_yscale("log")
    _ax.axvline(deterioro_rel.value, color="grey", ls=":")
    _ax.set_xlabel("deterioro relativo de la mora temprana (%)")
    _ax.set_ylabel("ARL (meses, escala log)")
    _ax.set_title(f"Meses hasta la alarma · n = {n_book.value}/mes, tasa base {p0_temprana.value:.1%}")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _d_sel = p0_temprana.value * deterioro_rel.value / 100 / _sd
    arl_credito = {"δ": _d_sel, "sd_pp": 100 * _sd,
                   "semaforo": arl_shewhart(_d_sel, 1.96), "cusum": arl_cusum_markov(_d_sel, 0.5, 4, m=150),
                   "cusum20": arl_cusum_bilateral(_d_sel, 0.5, H20, m=150),
                   "ewma": arl_ewma_markov(_d_sel, 0.2, 2.86, m=151)}
    mo.vstack([_fig, _curvas.round(2)])
    return (arl_credito,)


@app.cell
def _(arl_credito, coma, deterioro_rel, mo, n_book, p0_temprana):
    mo.md(coma(f"""
    **Lectura en unidades de crédito.** Con {n_book.value} aprobados al mes y mora temprana base {p0_temprana.value:.1%}, el
    error estándar mensual es {arl_credito['sd_pp']:.2f} pp; un deterioro relativo de {deterioro_rel.value}% equivale a
    δ = {arl_credito['δ']:.2f} σ. El semáforo lo ve en {arl_credito['semaforo']:.1f} meses en promedio (con una falsa alarma
    cada 20 meses); el CUSUM de igual ARL₀ en {arl_credito['cusum20']:.1f}; el EWMA (λ = 0,2, L = 2,86) en {arl_credito['ewma']:.1f} y el
    CUSUM h = 4 en {arl_credito['cusum']:.1f}, estos dos con una falsa alarma cada ~30 años. Baja n a 300 (una cartera de motos chica): δ cae a la
    mitad y los ARL se disparan. Con carteras chicas el monitoreo mensual de tasas es casi ciego y la única salida es
    **acumular** (CUSUM, ventanas trimestrales) o monitorear indicadores con más eventos (30+ en vez de 90+).
    Recuerda sumar el rezago: la mora a MOB 3 se mide 3 meses después de originar, así que la demora total es
    rezago + ARL.
    """))
    return


@app.cell
def _(SAT3, cusum_numpy, cusum_page, ewma_numpy, ewma_pandas, mes_k, mo, mora30, np, pd, plt, sim_actual, stats):
    # serie mensual de mora 30+ a MOB3 (aprobados) del escenario elegido, con E y V del satélite
    _book = sim_actual["aprueba"] | sim_actual["override"]
    _filas = []
    for _M in range(1, 25):
        _b = sim_actual[(sim_actual["cohorte"] == _M - 3) & _book]
        _l = np.log(_b["pd_modelo"] / (1 - _b["pd_modelo"])).to_numpy()
        _p = 1 / (1 + np.exp(-(SAT3[0] + SAT3[1] * _l)))
        _O = int(mora30(_b["T"].to_numpy(), _b["ep"].to_numpy(), 3).sum())
        _filas.append({"mes": _M, "n": len(_b), "O": _O, "E": _p.sum(), "V": (_p * (1 - _p)).sum(),
                       "WRA": _O * np.log(1.5) - np.log(1 - _p + 1.5 * _p).sum()})
    serie_temprana = pd.DataFrame(_filas).set_index("mes")
    serie_temprana["z"] = (serie_temprana["O"] - serie_temprana["E"]) / np.sqrt(serie_temprana["V"])
    _z = serie_temprana["z"].to_numpy()
    serie_temprana["EWMA"] = ewma_numpy(_z, 0.2)
    serie_temprana["CUSUM"] = cusum_numpy(_z, 0.5)
    serie_temprana["RA_CUSUM"] = cusum_numpy(serie_temprana["WRA"].to_numpy(), 0.0)
    assert np.allclose(serie_temprana["EWMA"], ewma_pandas(_z, 0.2))
    assert np.allclose(serie_temprana["CUSUM"], cusum_page(_z, 0.5))

    # h del CUSUM ajustado por riesgo: calibrado por simulación bajo H0 para ARL0 ≈ 336 (= CUSUM z, h = 4, unilateral)
    _rng = np.random.default_rng(1959)
    _E, _n = serie_temprana["E"].mean(), int(round(serie_temprana["n"].mean()))
    _pbar = _E / _n
    _cte = serie_temprana["WRA"].mean() - serie_temprana["O"].mean() * np.log(1.5)
    _rutas, _T = 800, 1500
    _O = _rng.binomial(_n, _pbar, size=(_rutas, _T))
    _S = np.zeros(_rutas)
    _max = np.zeros((_rutas, _T))
    for _t in range(_T):
        _S = np.maximum(0, _S + _O[:, _t] * np.log(1.5) + _cte)
        _max[:, _t] = _S
    _runmax = np.maximum.accumulate(_max, axis=1)
    _hs = np.linspace(1, 12, 111)
    _arl0 = np.array([np.where((_runmax > h).any(axis=1), (_runmax > h).argmax(axis=1) + 1, _T).mean() for h in _hs])
    H_RA = float(_hs[np.argmin(np.abs(_arl0 - 336))])

    _lim_ewma = 2.86 * np.sqrt(0.2 / 1.8 * (1 - 0.8 ** (2 * np.arange(1, 25))))

    def _primera(mask):
        _m = np.flatnonzero(mask)
        return int(serie_temprana.index[_m[0]]) if len(_m) else np.nan

    detecciones = pd.Series({
        "semáforo A1 (|z| > 1,96)": _primera(np.abs(_z) > 1.96),
        "Shewhart 3σ": _primera(_z > 3),
        "EWMA λ=0,2 L=2,86": _primera(serie_temprana["EWMA"].to_numpy() > _lim_ewma),
        "CUSUM k=0,5 h=4": _primera(serie_temprana["CUSUM"].to_numpy() > 4),
        f"CUSUM ajustado por riesgo R=1,5 h={H_RA:.1f}": _primera(serie_temprana["RA_CUSUM"].to_numpy() > H_RA),
    }, name="primer mes con alarma")

    _fig, _ax = plt.subplots(1, 3, figsize=(12, 3.4))
    _ax[0].bar(serie_temprana.index, _z, color=["#d7301f" if abs(x) > 1.96 else "#999999" for x in _z])
    _ax[0].axhline(1.96, color="k", ls=":")
    _ax[0].axhline(-1.96, color="k", ls=":")
    _ax[0].set_title("z mensual (semáforo A1)")
    _ax[1].plot(serie_temprana.index, serie_temprana["EWMA"], "o-", ms=3, label="EWMA")
    _ax[1].plot(serie_temprana.index, _lim_ewma, "k:", label="límite")
    _ax[1].set_title("EWMA de z")
    _ax[1].legend(fontsize=7)
    _ax[2].plot(serie_temprana.index, serie_temprana["CUSUM"], "o-", ms=3, label="CUSUM z (h=4)")
    _ax[2].plot(serie_temprana.index, serie_temprana["RA_CUSUM"], "s-", ms=3, label=f"CUSUM riesgo (h={H_RA:.1f})")
    _ax[2].axhline(4, color="tab:blue", ls=":")
    _ax[2].axhline(H_RA, color="tab:orange", ls=":")
    _ax[2].set_title("CUSUM superior")
    _ax[2].legend(fontsize=7)
    for _a in _ax:
        _a.axvline(mes_k.value, color="grey", lw=1)
        _a.set_xlabel("mes de reporte")
    _ax[0].set_ylabel("σ / estadístico")
    _fig.tight_layout()
    _ = stats
    mo.vstack([_fig, detecciones.to_frame()])
    return H_RA, detecciones, serie_temprana


@app.cell
def _(coma, detecciones, escenario, mes_k, mo):
    mo.md(coma(f"""
    **Lectura** (escenario «{escenario.value}», k = {mes_k.value}; la mora a MOB 3 de la cosecha k se ve en el mes k + 3,
    aunque un shock de calendario se asoma antes en cosechas previas). Primeras alarmas:
    {', '.join(f'{i}: {v:.0f}' if v == v else f'{i}: —' for i, v in detecciones.items())}.
    En «nivel» el semáforo suele encenderse primero (es el más sensible: paga ARL₀ = 20) pero **parpadea**: se prende y se
    apaga, y la regla de persistencia lo retrasa; los acumuladores, una vez que cruzan, se quedan arriba. En «ninguno»
    cualquier alarma es falsa: el semáforo a 1,96σ produce, en promedio, una cada 20 meses; el CUSUM, una cada ~28 años.
    El CUSUM ajustado por riesgo usa la PD de cada crédito: es el test óptimo (en el sentido de Moustakides 1986) para
    detectar un cambio de odds de magnitud R. Su h se calibró por simulación a igual ARL₀ que el CUSUM de z.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Umbral por costo: la curva operativa del CUSUM

    Elegir $h$ es elegir un punto de la curva ARL₀ ↔ ARL₁. Un criterio explícito (en el espíritu del diseño
    económico de gráficos de control de Duncan 1956, muy simplificado): costo por mes
    $$c(h)=\frac{C_{FA}}{\text{ARL}_0(h)}+\pi\,C_{d}\,\text{ARL}_1(h),$$
    con $C_{FA}$ el costo de investigar una falsa alarma, $C_d$ el costo de cada mes de deterioro no
    detectado y $\pi$ la probabilidad mensual de que ocurra un deterioro (aquí 1/36). Es un modelo de
    juguete: su valor es obligar a escribir los dos costos, no el $h$ que entrega.
    """)
    return


@app.cell
def _(mo):
    razon_costos = mo.ui.slider(steps=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], value=0.1,
                                label="C_d / C_FA (costo de un mes sin detectar ÷ costo de una falsa alarma)")
    razon_costos
    return (razon_costos,)


@app.cell
def _(arl_credito, arl_cusum_markov, np, pd, plt, razon_costos):
    _hs = np.arange(0.5, 10.01, 0.5)
    _d = max(arl_credito["δ"], 0.25)
    _a0 = np.array([arl_cusum_markov(0.0, 0.5, h, m=120) for h in _hs])
    _a1 = np.array([arl_cusum_markov(_d, 0.5, h, m=120) for h in _hs])
    _costo = 1 / _a0 + (1 / 36) * razon_costos.value * _a1
    curva_operativa = pd.DataFrame({"h": _hs, "ARL0": _a0, "ARL1": _a1, "costo_rel": _costo}).set_index("h")
    H_OPT = float(_hs[np.argmin(_costo)])
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.4))
    _ax[0].plot(_a0, _a1, "o-")
    for _h, _x, _y in zip(_hs[::2], _a0[::2], _a1[::2]):
        _ax[0].annotate(f"h={_h:g}", (_x, _y), fontsize=7)
    _ax[0].set_xscale("log")
    _ax[0].set_xlabel("ARL₀: meses entre falsas alarmas (log)")
    _ax[0].set_ylabel(f"ARL₁ a δ = {_d:.2f}σ (meses)")
    _ax[0].set_title("Curva operativa del CUSUM (k = 0,5)")
    _ax[1].plot(_hs, _costo, "o-")
    _ax[1].axvline(H_OPT, color="grey", ls=":")
    _ax[1].set_xlabel("h")
    _ax[1].set_ylabel("costo por mes (unidades de C_FA)")
    _ax[1].set_title(f"Costo mínimo en h = {H_OPT:g}")
    _fig.tight_layout()
    _fig
    return H_OPT, curva_operativa


@app.cell
def _(H_OPT, coma, curva_operativa, mo, razon_costos):
    mo.md(coma(f"""
    **Lectura.** Con C_d/C_FA = {razon_costos.value}, el h de costo mínimo es **{H_OPT:g}** (ARL₀ {curva_operativa.loc[H_OPT, 'ARL0']:.0f}
    meses, ARL₁ {curva_operativa.loc[H_OPT, 'ARL1']:.1f}). Sube la razón a 1 o más y el óptimo se va al borde inferior
    (h ≤ 1: alarma casi todos los meses): con π = 1/36, si un mes de deterioro cuesta lo mismo que una falsa alarma, a
    este modelo le conviene alarmar siempre. Que eso sea absurdo en la práctica revela lo que la fórmula omite: el
    costo de una falsa alarma no es revisar un informe, es la **acción equivocada** que dispara (recalibrar sin causa
    mueve precios y aprobación) y el **desgaste de la alarma** (el comité que aprende a ignorar amarillos). Escribir
    ambos costos es lo que justifica un h de 4–5 en vez de heredarlo de una tabla.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Curvas de cosecha (vintage) contra la curva esperada

    Para cada cosecha aprobada, la tasa acumulada de 90+ por MOB, observada hasta donde el mes de
    corte lo permite, contra la curva **esperada** que implica el artefacto:
    $\hat v_c(j)=\frac1{n_c}\sum_i\bigl[1-(1-\text{PD}_i)^{F(j)}\bigr]$, con $F(j)=\sum_{m\le j}w_m$ el perfil de
    maduración estimado en desarrollo, y banda de ±1,96 desviaciones estándar Poisson-binomiales. Una
    cosecha que sale de su banda a MOB 5–6 anticipa el malo a 12 m de esa cosecha, no del modelo en
    general: la curva de cosecha mezcla efecto **cosecha** (quién entró), **madurez** (MOB) y **calendario**
    (el shock que golpea a todas a la vez; en el gráfico se ve como diagonal).
    """)
    return


@app.cell
def _(F_MOB, mes_corte, mes_k, np, plt, sim_actual):
    _book = sim_actual["aprueba"] | sim_actual["override"]
    _coh = [c for c in (mes_k.value - 6, mes_k.value - 2, mes_k.value, mes_k.value + 3, mes_k.value + 6)
            if -13 <= c <= mes_corte.value - 2] or [c for c in (mes_corte.value - 10, mes_corte.value - 6,
                                                             mes_corte.value - 2) if c >= -13]
    _fig, _axs = plt.subplots(1, max(1, len(_coh)), figsize=(2.6 * max(1, len(_coh)), 3.2), sharey=True)
    _axs = np.atleast_1d(_axs)
    vintage_fuera = {}
    for _a, _c in zip(_axs, _coh):
        _b = sim_actual[(sim_actual["cohorte"] == _c) & _book]
        _pd = _b["pd_modelo"].to_numpy()
        _T = _b["T"].to_numpy()
        _mobs = np.arange(1, 13)
        _q = 1 - (1 - _pd[:, None]) ** F_MOB[None, :]
        _esp = _q.mean(axis=0)
        _sd = np.sqrt((_q * (1 - _q)).sum(axis=0)) / len(_pd)
        _max_mob = min(12, mes_corte.value - _c)
        _obs = np.array([(_T <= j).mean() for j in _mobs])[:_max_mob]
        _a.fill_between(_mobs, 100 * (_esp - 1.96 * _sd), 100 * (_esp + 1.96 * _sd), color="grey", alpha=0.3,
                        label="esperada ±1,96σ")
        _a.plot(_mobs, 100 * _esp, "k--", lw=1)
        _a.plot(_mobs[:_max_mob], 100 * _obs, "o-", ms=3, color="tab:red" if _c >= mes_k.value else "tab:blue",
                label="observada")
        _a.set_title(f"cosecha {_c}" + (" (post k)" if _c >= mes_k.value else ""), fontsize=9)
        _a.set_xlabel("MOB")
        _fuera = _obs > (_esp + 1.96 * _sd)[:_max_mob]
        vintage_fuera[_c] = int(_mobs[:_max_mob][_fuera][0]) if _fuera.any() else None
    _axs[0].set_ylabel("90+ acumulado (%)")
    _axs[0].legend(fontsize=7)
    _fig.suptitle(f"Curvas de cosecha al mes de corte {mes_corte.value}", fontsize=10)
    _fig.tight_layout()
    _fig
    return (vintage_fuera,)


@app.cell
def _(coma, mo, vintage_fuera):
    mo.md(coma(f"""
    **Lectura.** Primer MOB en que cada cosecha supera su banda superior:
    {'; '.join(f'cosecha {c}: ' + (f'MOB {m}' if m else 'nunca') for c, m in vintage_fuera.items())}. En «nivel» las cosechas
    previas a k también se despegan, pero desde el MOB en que su calendario cruza k (efecto de calendario, no de
    cosecha); en «población» las curvas post-k suben **y** la curva esperada sube con ellas (el modelo sabe que
    entró gente más riesgosa): observada dentro de la banda, sin alarma de nivel. Esa es la diferencia entre
    «más malos» y «más malos de los prometidos».
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. ¿El patrón recupera la causa? Matriz escenario × diagnóstico

    Para cada escenario (inicio en el mes 8, intensidad elegida abajo) y varias semillas, se aplica el árbol de §5
    en el mes $k+9$ (tiempo para que los adelantados con rezago 6–8 tengan datos post-$k$). La diagonal
    esperada: nivel → NIVEL, población → POBLACIÓN, ranking → ORDEN ROTO, dato roto → DATOS ROTOS,
    overrides → POLÍTICA / NEGOCIO, ninguno → SIN SEÑAL.
    """)
    return


@app.cell
def _(mo):
    n_semillas_diag = mo.ui.slider(1, 8, value=3, step=1, label="Semillas por escenario")
    intensidad_diag = mo.ui.slider(0.1, 1.0, value=0.5, step=0.1, label="Intensidad del deterioro")
    mo.hstack([n_semillas_diag, intensidad_diag])
    return intensidad_diag, n_semillas_diag


@app.cell
def _(ESCENARIOS, calcular_tablero, diagnosticar, intensidad_diag, n_semillas_diag, pd, simular_produccion):
    _k, _M = 8, 17
    _filas = []
    for _esc in ESCENARIOS:
        for _s in range(n_semillas_diag.value):
            _sim = simular_produccion(_esc, _k, intensidad_diag.value, 700 + _s, m1=_M)
            _v, _e = calcular_tablero(_sim, [_M - 1, _M], seed=_s, B_hl=100)
            _filas.append({"escenario": _esc, "diagnóstico": diagnosticar(_e, _M)})
    _df = pd.DataFrame(_filas)
    matriz_diag = pd.crosstab(_df["escenario"], _df["diagnóstico"]).reindex(ESCENARIOS).fillna(0).astype(int)
    _correcto = {"ninguno": "SIN SEÑAL", "nivel": "NIVEL", "población": "POBLACIÓN", "ranking": "ORDEN ROTO",
                 "dato roto": "DATOS ROTOS", "overrides": "POLÍTICA / NEGOCIO"}
    ACIERTO_DIAG = float((_df["diagnóstico"] == _df["escenario"].map(_correcto)).mean())
    matriz_diag
    return ACIERTO_DIAG, matriz_diag


@app.cell
def _(ACIERTO_DIAG, coma, intensidad_diag, matriz_diag, mo, np):
    _sin = int(matriz_diag["SIN SEÑAL"].drop("ninguno").sum()) if "SIN SEÑAL" in matriz_diag.columns else 0
    _odds = float(np.exp(0.6 * 0.2))
    mo.md(coma(f"""
    **Lectura.** Con intensidad {intensidad_diag.value:.1f}, acierto global del árbol en el mes k + 9: **{ACIERTO_DIAG:.0%}**;
    {_sin} corridas con deterioro real quedaron en SIN SEÑAL. Baja la intensidad a 0,2: datos rotos y overrides se
    siguen diagnosticando (sus indicadores tienen rezago 0 y casi no tienen ruido), mientras que nivel, población y
    orden caen a SIN SEÑAL, porque ningún par de indicadores de la familia se sostiene dos meses. El tablero tiene
    **resolución finita**: un deterioro de nivel de odds × {_odds:.2f} no se distingue del ruido
    en 9 meses con 1500 solicitudes al mes. La decisión honesta con señales débiles es vigilancia reforzada y un
    acumulador (§7), no una acción.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Overrides y su monitoreo

    Dos indicadores distintos: **cuántos** (override rate, N2) y **cómo pagan** (mora de los overrides contra
    la PD que el modelo les asignó). Si los overrides pagan mejor que su PD, el juicio experto agrega
    información que el modelo no tiene (candidata a variable nueva); si pagan igual, el override solo
    compra volumen al precio que el modelo anunciaba; si pagan peor, hay un problema de gobierno.
    Se compara la mora 30+ a MOB 6 (satélite de DEV) de overrides y de aprobados por score.
    """)
    return


@app.cell
def _(SAT6, mes_corte, mo, mora30, np, pd, sim_actual, z_poisson_binomial):
    _lim = mes_corte.value - 6
    _filas = []
    for _nombre, _mask in (("aprobados por score", sim_actual["aprueba"]), ("overrides", sim_actual["override"])):
        _b = sim_actual[_mask & (sim_actual["cohorte"] <= _lim) & (sim_actual["cohorte"] >= 1)]
        if len(_b) == 0:
            continue
        _l = np.log(_b["pd_modelo"] / (1 - _b["pd_modelo"])).to_numpy()
        _pe = 1 / (1 + np.exp(-(SAT6[0] + SAT6[1] * _l)))
        _O = int(mora30(_b["T"].to_numpy(), _b["ep"].to_numpy(), 6).sum())
        _z, _E, _V = z_poisson_binomial(_O, _pe)
        _filas.append({"grupo": _nombre, "n": len(_b), "PD_modelo_media": _b["pd_modelo"].mean(),
                       "PD_real_media": _b["pd_real"].mean(), "mora30_MOB6_obs": _O, "esperada": _E,
                       "O/E": _O / _E, "z": _z,
                       "malo12_real_%": 100 * (_b["T"] < 99).mean()})
    tabla_overrides = pd.DataFrame(_filas, columns=["grupo", "n", "PD_modelo_media", "PD_real_media",
                                                    "mora30_MOB6_obs", "esperada", "O/E", "z",
                                                    "malo12_real_%"]).set_index("grupo")
    _ov_mes = (sim_actual[sim_actual["cohorte"] >= 1].groupby("cohorte")
               .apply(lambda d: 100 * d["override"].sum() / max(1, (~d["aprueba"]).sum()), include_groups=False))
    mo.vstack([tabla_overrides.round(3),
               mo.md(f"Override rate mensual (%): {', '.join(f'{x:.0f}' for x in _ov_mes)}")])
    return (tabla_overrides,)


@app.cell
def _(coma, mo, tabla_overrides):
    _t = tabla_overrides
    _ov = _t.loc["overrides"] if "overrides" in _t.index else None
    _ap = _t.loc["aprobados por score"] if "aprobados por score" in _t.index else None
    _txt = ("Al mes de corte todavía no hay cosechas con MOB 6 observado: sube el mes de corte. " if _ap is None else
            "" if _ov is None else
            f"Overrides: PD media del modelo {_ov['PD_modelo_media']:.1%}, malo real a 12 m {_ov['malo12_real_%']:.1f}%, "
            f"O/E de mora temprana {_ov['O/E']:.2f} (z = {_ov['z']:+.2f}); aprobados por score: O/E {_ap['O/E']:.2f}. ")
    mo.md(coma(f"""
    **Lectura.** {_txt}
    En este simulador los overrides se eligen **al azar** entre los rechazados, así que pagan lo que el modelo
    anunciaba **relativo al resto**: su O/E se parece al de los aprobados por score (ambos ≈ 1 salvo en el escenario de
    nivel, donde ambos suben), pero con una mora varias veces mayor. Moraleja: un O/E de overrides ≈ 1 no justifica los
    overrides; solo dice que el modelo los ordenaba bien. La pregunta de negocio es si esa PD cabe en el precio (M17).
    En la vida real los overrides no son al azar y conviene medirlos por gestor, sucursal y motivo, con n chicos:
    acumular (CUSUM) antes de concluir.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Gatillos de cinco partes, evaluados en el mes de corte

    Cada gatillo: **condición** medible · **valor de hoy** · **quién decide** · **qué acción** · **plazo**.
    La columna `disparado` se calcula desde la condición (no se escribe a mano), como en el Lab 3.
    """)
    return


@app.cell
def _(FAMILIAS, est_actual, familias_encendidas, mes_corte, pd, val_actual):
    _M = mes_corte.value
    _f = familias_encendidas(est_actual, _M)
    _pers = [i for i in est_actual.index if est_actual.loc[i, _M] >= 1 and (_M - 1 not in est_actual.columns
                                                                           or est_actual.loc[i, _M - 1] >= 1)]
    _rojos = [i for i in est_actual.index if est_actual.loc[i, _M] >= 2]

    def _hoy(ids):
        return "; ".join(f"{i}={val_actual.loc[i, _M]:.3g}" for i in ids)

    gatillos = pd.DataFrame([
        {"gatillo": "Vigilancia reforzada",
         "condicion": "≥1 indicador 🟡/🔴 dos meses seguidos, o cualquier 🔴",
         "valor_hoy": f"persistentes: {_pers or 'ninguno'}; rojos: {_rojos or 'ninguno'}",
         "disparado": "SÍ" if (_pers or _rojos) else "no",
         "quien_decide": "Jefe de Modelos", "accion": "análisis por segmento y canal; CUSUM diario/semanal de mora temprana",
         "plazo": "informe en 10 días hábiles; dura 2 meses"},
        {"gatillo": "Contingencia de datos",
         "condicion": "D1 > 1% (violaciones de contrato)",
         "valor_hoy": _hoy(FAMILIAS["datos"]), "disparado": "SÍ" if _f["datos"] or est_actual.loc["D1", _M] >= 1 else "no",
         "quien_decide": "Dueño del dato (TI) + Jefe de Modelos",
         "accion": "filas con violación a revisión manual/política experta; corregir feed; re-puntuar afectados",
         "plazo": "24–48 horas"},
        {"gatillo": "Recalibración del δ",
         "condicion": "familia NIVEL encendida (≥2 de A1, A2, R2, R3, R4 persistentes) con ORDEN verde y DATOS verde",
         "valor_hoy": _hoy(FAMILIAS["nivel"]),
         "disparado": "SÍ" if (_f["nivel"] and not _f["orden"] and not _f["datos"]) else "no",
         "quien_decide": "Comité de modelos (A), a propuesta del Jefe de Modelos (R)",
         "accion": "re-anclar δ con muestra reciente y madura declarada; acta; nueva versión menor del artefacto",
         "plazo": "propuesta en 30 días; vigencia al mes siguiente"},
        {"gatillo": "Re-desarrollo",
         "condicion": "A3 o R1 con caída relativa > 20% persistente, o > 30% en un mes, descartados datos y política",
         "valor_hoy": _hoy(FAMILIAS["orden"]),
         "disparado": "SÍ" if (_f["orden"] and not _f["datos"]) or (val_actual.loc[["A3", "R1"], _M] > 30).any() else "no",
         "quien_decide": "Comité de Riesgo",
         "accion": "abrir proyecto; mientras, recalibración provisional y cutoff conservador",
         "plazo": "plan en 60 días; modelo nuevo en 6–9 meses"},
        {"gatillo": "Revisión de estrategia (población)",
         "condicion": "familia POBLACIÓN encendida (≥2 de P1, P2, P3 persistentes)",
         "valor_hoy": _hoy(FAMILIAS["población"]), "disparado": "SÍ" if _f["población"] else "no",
         "quien_decide": "Gerencia de Riesgo de Crédito",
         "accion": "re-evaluar cutoff y límites por banda (M17); verificar calibración por banda antes de tocar el modelo",
         "plazo": "60 días"},
        {"gatillo": "Revisión de política comercial",
         "condicion": "N2 > 15% o |N1| > 5 pts, dos meses seguidos",
         "valor_hoy": _hoy(FAMILIAS["negocio"]), "disparado": "SÍ" if _f["negocio"] else "no",
         "quien_decide": "Comité de Crédito (dueño comercial + riesgo)",
         "accion": "auditar overrides por gestor y motivo; medir O/E de overrides; límite de overrides por sucursal",
         "plazo": "30 días"},
    ]).set_index("gatillo")
    gatillos
    return (gatillos,)


@app.cell
def _(coma, gatillos, mes_corte, mo):
    _disp = list(gatillos.index[gatillos["disparado"] == "SÍ"])
    mo.md(coma(f"""
    **Lectura.** En el mes {mes_corte.value} están disparados: {', '.join(_disp) if _disp else 'ninguno'}. Los gatillos tienen
    **dueños distintos** (quien construye no valida ni aprueba; RACI de la clase 6 y M22) y **plazos**: un gatillo
    sin plazo termina en una reunión, no en una acción. La jerarquía se respeta por construcción: la
    recalibración exige orden y datos verdes; el re-desarrollo exige datos verdes.
    """))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ✔ Checks del módulo

    Coincidencias numpy vs librería e invariantes teóricas. Corren sobre simulaciones con semilla fija
    (no dependen de los controles).
    """)
    return


@app.cell
def _(
    CUTOFF,
    MONTGOMERY,
    P_REAL_DEV,
    PD_DEV,
    REF_BANDAS,
    SAT3,
    SAT3_SM,
    arl_cusum_bilateral,
    arl_cusum_markov,
    arl_ewma_markov,
    arl_siegmund,
    auc_numpy,
    calcular_tablero,
    cusum_numpy,
    cusum_page,
    ewma_numpy,
    ewma_pandas,
    hosmer_lemeshow,
    np,
    p_binomial_numpy,
    psi_numpy,
    psi_scipy,
    roc_auc_score,
    simular_desempeno,
    simular_produccion,
    sin_shock,
    stats,
    z_poisson_binomial,
):
    _rng = np.random.default_rng(123)
    # 1. PSI numpy = Jeffreys con scipy.entropy
    _a = _rng.dirichlet(np.ones(8) * 30)
    assert np.isclose(psi_numpy(_a, REF_BANDAS), psi_scipy(_a, REF_BANDAS))
    # 2. AUC numpy (rangos, con empates) = sklearn
    _y = (_rng.random(3000) < 0.1).astype(int)
    _s = np.round(_rng.normal(_y, 1.0), 1)
    assert np.isclose(auc_numpy(_y, _s), roc_auc_score(_y, _s))
    # 3. binomial bilateral numpy = scipy.binomtest (B2 del curso: 235 créditos, PD 1,42%, 8 malos)
    for _k, _n, _p in ((8, 235, 0.0142), (3, 199, 0.0036), (64, 263, 0.2691), (0, 50, 0.02)):
        assert np.isclose(p_binomial_numpy(_k, _n, _p), stats.binomtest(_k, _n, _p).pvalue, rtol=1e-6)
    # 4. satélite IRLS numpy = statsmodels
    assert np.allclose(SAT3, SAT3_SM, atol=1e-6)
    # 5. EWMA numpy = pandas.ewm; CUSUM recursivo = forma de Page
    _z = _rng.standard_normal(200)
    assert np.allclose(ewma_numpy(_z, 0.2), ewma_pandas(_z, 0.2))
    assert np.allclose(cusum_numpy(_z, 0.5), cusum_page(_z, 0.5))
    # 6. ARL: Markov reproduce la tabla de Montgomery (±3%) y Siegmund ≈ Markov unilateral (±5%)
    for _d, (_t4, _t5) in MONTGOMERY.items():
        assert abs(arl_cusum_bilateral(_d, 0.5, 4) / _t4 - 1) < 0.03, _d
        assert abs(arl_cusum_bilateral(_d, 0.5, 5) / _t5 - 1) < 0.03, _d
        if _d <= 1:
            assert abs(arl_siegmund(_d, 0.5, 4) / arl_cusum_markov(_d, 0.5, 4) - 1) < 0.05, _d
    assert abs(arl_ewma_markov(0.0, 0.2, 2.962) / 500 - 1) < 0.03          # Lucas & Saccucci
    # 7. Poisson-binomial: normal ≈ exacto (convolución numpy) cuando E es grande
    _pp = _rng.uniform(0.01, 0.08, 1500)
    _pmf = np.array([1.0])
    for _pi in _pp:
        _pmf = np.convolve(_pmf, [1 - _pi, _pi])
    _O = int(_pp.sum() + 12)
    _p_exacto = min(1.0, 2 * min(_pmf[:_O + 1].sum(), _pmf[_O:].sum()))
    _p_normal = 2 * stats.norm.sf(abs(z_poisson_binomial(_O, _pp)[0]))
    assert abs(_p_exacto - _p_normal) < 0.03, (_p_exacto, _p_normal)
    # 8. HL: bajo H0 el p simulado no es sistemáticamente chico
    _ph = [hosmer_lemeshow((_rng.random(2000) < PD_DEV[:2000]).astype(float), PD_DEV[:2000], _rng, B=100)[1]
           for _ in range(20)]
    assert 0.25 < np.mean(_ph) < 0.75
    # 9. Invariantes del simulador (semillas fijas)
    _v0, _e0 = calcular_tablero(simular_produccion("ninguno", 99, 0.0, 42), range(1, 25), seed=1, B_hl=100)
    assert (_v0.loc["D1"] == 0).all()                                   # sin dato roto, contrato limpio
    assert (_v0.loc["P1"] < 0.05).all()                                 # PSI nulo lejos de 0,10 con n = 1.500
    _vd, _ed = calcular_tablero(simular_produccion("dato roto", 8, 0.5, 42), range(6, 12), seed=1, B_hl=50)
    assert _ed.loc["D1", 8] == 2 and (_ed.loc["D1", [6, 7]] == 0).all()  # el contrato lo ve el mismo mes k
    _vr, _er = calcular_tablero(simular_produccion("ranking", 8, 0.7, 42), [18, 23], seed=1, B_hl=50)
    assert _vr.loc["A3", 18] > 20 and _vr.loc["R1", 23] > 20             # el orden se rompe y se detecta
    _vo, _eo = calcular_tablero(simular_produccion("overrides", 8, 0.5, 42), [23, 24], seed=1, B_hl=50)
    assert (_vo.loc["N2"] > 25).all() and (_vo.loc["R1"] < 0).all()      # más overrides → Gini observado SUBE
    # 10. δ ancla la PD media del artefacto a la verdadera; sin shock el hazard reproduce la PD a 12 m
    assert abs(PD_DEV.mean() - P_REAL_DEV.mean()) < 1e-6
    _T, _ = simular_desempeno(np.full(200_000, 0.08), np.zeros(200_000, int), sin_shock, _rng)
    assert abs((_T < 99).mean() - 0.08) < 0.003
    assert CUTOFF == 530.0
    mo_checks = "✔ Todos los checks del módulo pasaron"
    mo_checks
    return


if __name__ == "__main__":
    app.run()

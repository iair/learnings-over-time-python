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
    from scipy import stats
    from scipy.optimize import brentq
    from statsmodels.stats.proportion import (
        confint_proportions_2indep,
        proportion_confint,
    )
    return brentq, confint_proportions_2indep, mo, plt, proportion_confint, sm, stats


@app.cell
def _(mo):
    mo.md(r"""
    # M18 · Swap-set y el problema contrafactual

    **Serie 2 · Del embudo al gobierno.** Acompaña a `M18_swap_set.md`.

    El curso cerró la estrategia con una matriz 2×2: a igual aprobación (90,2% en OOT), el scorecard
    **suelta** 128 créditos con 23,4% de malos (*swap-out*) y **toma** 128 con 9,4% (*swap-in*), y la
    mora de la cartera aprobada baja de 4,48% a 3,48%. Este notebook reconstruye esa mecánica sobre
    una cartera sintética con **verdad conocida** y después la rompe a propósito:

    1. Política vieja (knock-outs + tope de carga) vs scorecard; la matriz 2×2 a **iso-aprobación**,
       con slider de aprobación, severidad de los knock-outs y muestra (DEV infla).
    2. La frontera aprobación–mora: **iso-aprobación, iso-riesgo, iso-pérdida** (con montos).
    3. Intervalos binomiales exactos por celda (Clopper-Pearson numpy vs `scipy`/`statsmodels`),
       Fisher exacto y Newcombe para 9,4% vs 23,4%, y por qué la diferencia de carteras es **pareada**.
    4. **La trampa sutil**: en la vida real el swap-in no tiene desempeño. Entrenamos solo con los
       aprobados por la política vieja y medimos cuánto se equivoca la estimación del swap-in contra
       la verdad (selección sobre observables vs sobre información privada).
    5. **Exploración aleatoria** bajo el corte: cuánto cuesta (pérdida) y cuánto error de estimación
       compra; valor de la información para una decisión concreta.
    6. La **cohorte swap-in** del tablero: test binomial, potencia y fecha de lectura.
    7. Swap-set por **segmento** y por **montos** (pérdida en pesos en vez de conteo).
    8. Checks del módulo.

    Convenciones del curso: target **1 = malo** (90+ a 12 meses); WoE = ln(%buenos/%malos);
    PDO 20, score 600 a odds 50:1 (factor 28,8539; offset 487,1229). Todo lo que se ajusta se ajusta
    en DEV y se aplica al resto.
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
        renta = np.exp(rng.normal(13.6, 0.55, n)) / 1e6            # MM CLP
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
    ## 0. Herramientas propias del módulo

    Encima del código común: el scorecard (WoE de DEV + logística; IRLS en numpy **y** GLM de
    `statsmodels`), las políticas (knock-outs, tope de carga, top-k por score) y la aritmética
    binomial en numpy puro (log-combinatorios por suma acumulada, sin `scipy.special`). Las
    versiones de librería se usan en cada sección para comparar.
    """)
    return


@app.cell
def _(a_woe, np, pd, sm, tabla_woe):
    FACTOR = 20 / np.log(2)
    OFFSET = 600 - FACTOR * np.log(50)
    LGD = 0.45
    VARIABLES = ["uso_linea_prom_12m", "uso_tc_prom_3m",
                 "meses_desde_mora_12m", "antiguedad_meses", "carga_financiera",
                 "consultas_6m", "canal"]


    def sigmoide(z):
        return 1.0 / (1.0 + np.exp(-z))


    def logit_np(p):
        return np.log(p / (1.0 - p))


    def pct(x, d=1):
        """Porcentaje con coma decimal para la prosa."""
        return f"{100 * x:.{d}f}%".replace(".", ",")


    def num(x, d=2):
        """Número con coma decimal y punto de miles (prosa en español)."""
        return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


    def sgn(x, d=2):
        """Número con signo explícito y coma decimal."""
        return ("+" if x >= 0 else "−") + num(abs(x), d)


    def irls_numpy(X, y, iters=50):
        """Logística por IRLS (Newton-Raphson) desde cero; X sin columna de unos."""
        Xc = np.column_stack([np.ones(len(X)), X])
        b = np.zeros(Xc.shape[1])
        for _ in range(iters):
            p = sigmoide(Xc @ b)
            W = p * (1 - p)
            H = Xc.T @ (Xc * W[:, None])
            g = Xc.T @ (y - p)
            paso = np.linalg.solve(H, g)
            b = b + paso
            if np.max(np.abs(paso)) < 1e-11:
                break
        return b


    def ajustar_scorecard(dev, variables=VARIABLES):
        """Bins y WoE en `dev`, logística sobre WoE (GLM statsmodels + IRLS numpy)."""
        dev = dev.reset_index(drop=True)
        mapas, ivs = {}, {}
        for v in variables:
            tab, iv = tabla_woe(dev[v], dev["malo"])
            mapas[v] = tab["woe"].to_dict()
            ivs[v] = iv
        X = a_woe(dev, variables, dev, mapas).to_numpy()
        y = dev["malo"].to_numpy()
        glm = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
        return {"dev": dev, "mapas": mapas, "vars": variables, "iv": ivs,
                "beta": np.asarray(glm.params), "beta_irls": irls_numpy(X, y)}


    def puntuar(sc, d):
        """PD del modelo y score PDO (score = offset + factor·ln(odds buenos))."""
        X = a_woe(d.reset_index(drop=True), sc["vars"], sc["dev"], sc["mapas"]).to_numpy()
        eta = sc["beta"][0] + X @ sc["beta"][1:]
        return sigmoide(eta), OFFSET - FACTOR * eta


    SEVERIDADES = {
        "suave (mora ≤ 1 m)": {"meses": 1, "consultas": 99, "renta": False},
        "base (mora ≤ 2 m · consultas ≥ 6 · sin renta)": {"meses": 2, "consultas": 6, "renta": True},
        "dura (mora ≤ 3 m · consultas ≥ 5 · sin renta)": {"meses": 3, "consultas": 5, "renta": True},
    }


    def knock_outs(d, severidad):
        """Reglas duras de la política vieja: flags por regla (True = rechaza)."""
        c = SEVERIDADES[severidad]
        m = d["meses_desde_mora_12m"].to_numpy()
        return pd.DataFrame({
            "ko_mora_reciente": (m >= 1) & (m <= c["meses"]),
            "ko_consultas": d["consultas_6m"].to_numpy() >= c["consultas"],
            "ko_sin_renta": d["renta_mm"].isna().to_numpy() & c["renta"],
        })


    def politica_vieja(d, ko_any, aprob_obj):
        """Knock-outs + tope de carga: entre los que pasan KO, aprueba los de menor carga
        hasta llegar a `aprob_obj` (si aprob_obj supera el paso de KO, aprueba a todos)."""
        n = len(d)
        k = min(int(round(aprob_obj * n)), int((~ko_any).sum()))
        carga = np.where(ko_any, np.inf, d["carga_financiera"].to_numpy())
        orden = np.argsort(carga, kind="stable")
        ap = np.zeros(n, dtype=bool)
        ap[orden[:k]] = True
        return ap


    def aprobar_top_k(score, k):
        """Los k mejores scores con desempate estable (un cuantil no garantiza k exacto)."""
        orden = np.argsort(-score, kind="stable")
        ap = np.zeros(len(score), dtype=bool)
        ap[orden[:k]] = True
        return ap, float(score[orden[k - 1]])


    CELDAS = ["ambas aprueban", "swap-out (sale)", "swap-in (entra)", "ambas rechazan"]


    def matriz_swap_numpy(ap_vieja, ap_nueva, y, monto=None):
        mascaras = [ap_vieja & ap_nueva, ap_vieja & ~ap_nueva,
                    ~ap_vieja & ap_nueva, ~ap_vieja & ~ap_nueva]
        filas = []
        for nombre, m in zip(CELDAS, mascaras):
            fila = {"celda": nombre, "n": int(m.sum()), "malos": int(y[m].sum()),
                    "tasa_malos": float(y[m].mean()) if m.any() else np.nan}
            if monto is not None:
                fila["monto_MM"] = float(monto[m].sum())
                fila["perdida_MM"] = float((LGD * y[m] * monto[m]).sum())
            filas.append(fila)
        return pd.DataFrame(filas).set_index("celda")


    def matriz_swap_pandas(ap_vieja, ap_nueva, y):
        g = (pd.DataFrame({"vieja": ap_vieja, "nueva": ap_nueva, "y": y})
               .groupby(["vieja", "nueva"])["y"].agg(["size", "sum"]))
        mapa = {(True, True): CELDAS[0], (True, False): CELDAS[1],
                (False, True): CELDAS[2], (False, False): CELDAS[3]}
        g.index = [mapa[i] for i in g.index]
        return g.reindex(CELDAS).fillna(0)


    # ---------- aritmética binomial en numpy puro ----------
    def log_comb_tabla(n):
        """log C(n, k) para k = 0..n por suma acumulada de log((n−j)/(j+1))."""
        j = np.arange(n)
        return np.concatenate([[0.0], np.cumsum(np.log((n - j) / (j + 1)))])


    def binom_pmf_np(n, p):
        """Vector pmf(k; n, p), k = 0..n, en log-espacio (no desborda con n grande)."""
        k = np.arange(n + 1)
        return np.exp(log_comb_tabla(n) + k * np.log(p) + (n - k) * np.log1p(-p))


    def biseccion(f, lo, hi, iters=80):
        """Raíz de f monótona en [lo, hi] por bisección."""
        f_lo = f(lo)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if np.sign(f(mid)) == np.sign(f_lo):
                lo, f_lo = mid, f(mid)
            else:
                hi = mid
        return 0.5 * (lo + hi)


    def clopper_pearson_np(k, n, alfa=0.05):
        """IC exacto: invierte las dos colas de la binomial (sin fórmula beta)."""
        def cola_sup(p):   # P(X ≥ k | p) − α/2, creciente en p
            return binom_pmf_np(n, p)[k:].sum() - alfa / 2

        def cola_inf(p):   # P(X ≤ k | p) − α/2, decreciente en p
            return binom_pmf_np(n, p)[:k + 1].sum() - alfa / 2

        lo = 0.0 if k == 0 else biseccion(cola_sup, 1e-12, 1 - 1e-12)
        hi = 1.0 if k == n else biseccion(cola_inf, 1e-12, 1 - 1e-12)
        return lo, hi


    def wilson_np(k, n, alfa=0.05):
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1 - alfa / 2)
        ph = k / n
        centro = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
        semi = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return centro - semi, centro + semi


    def newcombe_np(k1, n1, k2, n2, alfa=0.05):
        """IC híbrido de Newcombe (1998, método 10) para p1 − p2, desde Wilson."""
        p1, p2 = k1 / n1, k2 / n2
        l1, u1 = wilson_np(k1, n1, alfa)
        l2, u2 = wilson_np(k2, n2, alfa)
        d = p1 - p2
        return (d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
                d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


    def binomtest_np(k, n, p0):
        """p-valor bilateral exacto «minlike»: suma de pmf(x) ≤ pmf(k)."""
        pmf = binom_pmf_np(n, p0)
        return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-7)].sum()))


    def fisher_np(a, b, c, d):
        """Fisher exacto bilateral para [[a, b], [c, d]] vía hipergeométrica en numpy."""
        r1, r2, c1 = a + b, c + d, a + c
        n = r1 + r2
        lc1, lc2, lcn = log_comb_tabla(r1), log_comb_tabla(r2), log_comb_tabla(n)
        xs = np.arange(max(0, c1 - r2), min(r1, c1) + 1)
        pmf = np.exp(lc1[xs] + lc2[c1 - xs] - lcn[c1])
        p_obs = pmf[xs == a][0]
        return float(min(1.0, pmf[pmf <= p_obs * (1 + 1e-7)].sum()))


    def potencia_binomial_np(n, p0, p1, alfa=0.05):
        """Potencia exacta del test binomial bilateral (minlike) de H0: p = p0 cuando p = p1."""
        pmf0 = binom_pmf_np(n, p0)
        orden = np.sort(pmf0)
        acum = np.cumsum(orden)
        # p-valor(x) = suma de pmf0 ≤ pmf0[x]  → búsqueda en la distribución ordenada
        idx = np.searchsorted(orden, pmf0 * (1 + 1e-7), side="right") - 1
        pval = acum[idx]
        rechaza = pval <= alfa
        return float(binom_pmf_np(n, p1)[rechaza].sum())
    return (
        CELDAS,
        FACTOR,
        LGD,
        OFFSET,
        SEVERIDADES,
        VARIABLES,
        ajustar_scorecard,
        aprobar_top_k,
        binom_pmf_np,
        binomtest_np,
        clopper_pearson_np,
        fisher_np,
        knock_outs,
        logit_np,
        matriz_swap_numpy,
        matriz_swap_pandas,
        newcombe_np,
        num,
        pct,
        politica_vieja,
        sgn,
        potencia_binomial_np,
        puntuar,
        sigmoide,
        wilson_np,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. La cartera, el scorecard y la política vieja

    **Qué mirar.** Generamos la cartera (24.000 solicitudes, verdad conocida) y le agregamos un
    **monto** sintético de crédito de moto (EAD, MM CLP) que crece con la renta: los montos grandes van a
    clientes de menor carga, así que contar créditos y contar pesos no dan lo mismo (sección 7).
    El scorecard se ajusta en DEV con las 8 variables. La **política vieja** son tres knock-outs
    (análogos a «mora interna vigente o ≥ 30 días en el sistema» del curso) más un tope de carga
    financiera que fija la aprobación.

    En esta vista «del curso», los knock-outs son **hipotéticos**: todos tienen desempeño, igual
    que las cursadas de Banco Austral a las que el curso aplicó los knock-outs a posteriori.
    """)
    return


@app.cell
def _(ajustar_scorecard, generar_cartera, np):
    cartera = generar_cartera()
    _rng = np.random.default_rng(18)
    _renta = cartera["renta_mm"].fillna(cartera["renta_mm"].median()).to_numpy()
    # monto de moto (MM CLP): ~2,4 MM en la renta mediana, elasticidad 0,45, ruido log-normal
    cartera["monto_mm"] = np.clip(
        2.4 * (_renta / 0.8) ** 0.45 * np.exp(_rng.normal(0, 0.25, len(cartera))), 0.8, 8.0
    ).round(3)
    sc_completo = ajustar_scorecard(cartera[cartera["muestra"] == "DEV"])
    return cartera, sc_completo


@app.cell
def _(SEVERIDADES, cartera, knock_outs, mo, np, pct, pd):
    _hist = cartera[cartera["muestra"] != "TTD"]
    _filas = []
    for _sev in SEVERIDADES:
        _ko = knock_outs(_hist, _sev)
        _any = _ko.any(axis=1).to_numpy()
        _fila = {"severidad": _sev, "rechazo KO": pct(_any.mean()),
                 "malos rechazados KO": pct(_hist["malo"].to_numpy()[_any].mean()),
                 "malos que pasan KO": pct(_hist["malo"].to_numpy()[~_any].mean())}
        for _r in _ko.columns:
            _fila[_r] = pct(_ko[_r].mean())
        _filas.append(_fila)
    _tabla_ko = pd.DataFrame(_filas).set_index("severidad")
    mo.vstack([
        mo.md("**Knock-outs sobre la historia con desempeño (DEV+HO+OOT).** "
              f"Tasa de malos global: {pct(np.nanmean(_hist['malo']))}."),
        mo.ui.table(_tabla_ko.reset_index(), selection=None),
    ])
    return


@app.cell
def _(
    VARIABLES,
    cartera,
    mo,
    np,
    num,
    pd,
    puntuar,
    sc_completo,
    stats,
):
    _oot = cartera[cartera["muestra"] == "OOT"]
    _pd, _ = puntuar(sc_completo, _oot)
    _auc = stats.mannwhitneyu(_pd[_oot["malo"] == 1], _pd[_oot["malo"] == 0]).statistic / (
        (_oot["malo"] == 1).sum() * (_oot["malo"] == 0).sum())
    _auc_v = stats.mannwhitneyu(_oot["pd_verdadera"][_oot["malo"] == 1],
                                _oot["pd_verdadera"][_oot["malo"] == 0]).statistic / (
        (_oot["malo"] == 1).sum() * (_oot["malo"] == 0).sum())
    gini_oot_completo = 2 * _auc - 1
    _coef = pd.DataFrame({"β (GLM)": sc_completo["beta"][1:], "β (IRLS numpy)": sc_completo["beta_irls"][1:],
                          "IV (DEV)": [sc_completo["iv"][v] for v in VARIABLES]}, index=VARIABLES).round(4)
    assert np.allclose(sc_completo["beta"], sc_completo["beta_irls"], atol=1e-6)
    mo.vstack([
        mo.md(f"**Scorecard (DEV completo).** Intercepto {num(sc_completo['beta'][0], 3)}; Gini OOT "
              f"{num(gini_oot_completo, 3)} (la PD verdadera logra {num(2 * _auc_v - 1, 3)}: techo del generador). "
              "IRLS numpy y GLM coinciden (assert)."),
        mo.ui.table(_coef.reset_index(names="variable"), selection=None),
    ])
    return (gini_oot_completo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. La matriz 2×2 a iso-aprobación

    **Qué mirar.** Elige la severidad de los knock-outs, la aprobación de la política vieja (el tope
    de carga la baja por debajo del paso de KO) y la muestra. El scorecard aprueba sus **k mejores**
    scores con k = aprobados de la vieja (desempate estable; el «cutoff equivalente» es el score del
    k-ésimo). Con k igual, swap-in y swap-out tienen **el mismo n** por construcción, y la variación de
    mora de la cartera es exactamente $\Delta=\tfrac{s}{K}(b_{\text{in}}-b_{\text{out}})$.
    Compara DEV, HO y OOT: la regla del curso es medir fuera del desarrollo; aquí verás cuánto
    pesa el sobreajuste (poco, con este modelo) frente al cambio de nivel entre periodos (mucho).
    """)
    return


@app.cell
def _(SEVERIDADES, mo):
    sel_severidad = mo.ui.dropdown(options=list(SEVERIDADES), value="base (mora ≤ 2 m · consultas ≥ 6 · sin renta)",
                                   label="Knock-outs de la política vieja")
    sl_aprob = mo.ui.slider(0.60, 0.95, step=0.01, value=0.95, label="Aprobación objetivo de la política vieja")
    sel_muestra = mo.ui.dropdown(options=["DEV", "HO", "OOT"], value="OOT", label="Muestra")
    mo.hstack([sel_severidad, sl_aprob, sel_muestra], justify="start")
    return sel_muestra, sel_severidad, sl_aprob


@app.cell
def _(
    aprobar_top_k,
    cartera,
    knock_outs,
    matriz_swap_numpy,
    matriz_swap_pandas,
    np,
    politica_vieja,
    puntuar,
    sc_completo,
    sel_muestra,
    sel_severidad,
    sl_aprob,
):
    datos_swap = cartera[cartera["muestra"] == sel_muestra.value].reset_index(drop=True)
    _ko = knock_outs(datos_swap, sel_severidad.value).any(axis=1).to_numpy()
    ap_vieja = politica_vieja(datos_swap, _ko, sl_aprob.value)
    pd_swap, score_swap = puntuar(sc_completo, datos_swap)
    k_swap = int(ap_vieja.sum())
    ap_nueva, cutoff_equiv = aprobar_top_k(score_swap, k_swap)
    y_swap = datos_swap["malo"].to_numpy()
    matriz = matriz_swap_numpy(ap_vieja, ap_nueva, y_swap, datos_swap["monto_mm"].to_numpy())
    _mp = matriz_swap_pandas(ap_vieja, ap_nueva, y_swap)
    assert np.array_equal(matriz["n"].to_numpy(), _mp["size"].to_numpy().astype(int))
    assert np.array_equal(matriz["malos"].to_numpy(), _mp["sum"].to_numpy().astype(int))
    assert matriz.loc["swap-in (entra)", "n"] == matriz.loc["swap-out (sale)", "n"]
    br_vieja = float(y_swap[ap_vieja].mean())
    br_nueva = float(y_swap[ap_nueva].mean())
    return (
        ap_nueva,
        ap_vieja,
        br_nueva,
        br_vieja,
        cutoff_equiv,
        datos_swap,
        k_swap,
        matriz,
        pd_swap,
        score_swap,
        y_swap,
    )


@app.cell
def _(
    br_nueva,
    br_vieja,
    cutoff_equiv,
    datos_swap,
    k_swap,
    matriz,
    mo,
    np,
    num,
    pct,
    proportion_confint,
    sgn,
):
    _t = matriz.copy()
    _ic = [proportion_confint(r.malos, r.n, method="beta") if r.n > 0 else (np.nan, np.nan)
           for r in _t.itertuples()]
    _t["IC95 exacto"] = [f"[{pct(a)} ; {pct(b)}]" for a, b in _ic]
    _t["tasa_malos"] = _t["tasa_malos"].map(pct)
    _t["% del flujo"] = (matriz["n"] / len(datos_swap)).map(pct)
    _s = int(matriz.loc["swap-in (entra)", "n"])
    _bi = matriz.loc["swap-in (entra)", "tasa_malos"]
    _bo = matriz.loc["swap-out (sale)", "tasa_malos"]
    _delta_identidad = _s / k_swap * (_bi - _bo) if _s > 0 else 0.0
    assert np.isclose(br_nueva - br_vieja, _delta_identidad)
    mo.vstack([
        mo.md(f"Aprobación: **{pct(k_swap / len(datos_swap))}** ({num(k_swap, 0)} de {num(len(datos_swap), 0)}) · "
              f"cutoff equivalente del scorecard: **{cutoff_equiv:.0f}** puntos."),
        mo.ui.table(_t[["n", "% del flujo", "malos", "tasa_malos", "IC95 exacto", "monto_MM"]]
                    .round(1).reset_index(), selection=None),
        mo.md(f"Mora de la cartera aprobada: vieja **{pct(br_vieja, 2)}** → scorecard **{pct(br_nueva, 2)}** "
              f"({sgn((br_nueva / br_vieja - 1) * 100, 0)}% relativo). Identidad "
              f"$\\Delta = \\tfrac{{s}}{{K}}(b_{{in}}-b_{{out}})$ = {sgn(_delta_identidad * 100, 2)} pp (assert)."),
    ])
    return


@app.cell
def _(br_nueva, br_vieja, matriz, mo, pct, sel_muestra):
    _bi = matriz.loc["swap-in (entra)", "tasa_malos"]
    _bo = matriz.loc["swap-out (sale)", "tasa_malos"]
    _br = matriz.loc["ambas rechazan", "tasa_malos"]
    _txt = (f"**Lectura ({sel_muestra.value}).** Entran {int(matriz.loc['swap-in (entra)', 'n'])} créditos con "
            f"{pct(_bi)} de malos y salen otros tantos con {pct(_bo)}: el intercambio "
            f"{'mejora' if _bi < _bo else 'EMPEORA'} la cartera ({pct(br_vieja, 2)} → {pct(br_nueva, 2)}). "
            f"Los que ambas rechazan traen {pct(_br)}: los knock-outs no estaban locos, eran insuficientes. ")
    if sel_muestra.value == "DEV":
        _txt += ("En **DEV** el modelo se evalúa con los mismos datos con que se ajustó. Con 7 variables "
                 "agrupadas y ~10.000 casos el optimismo por sobreajuste es chico (compara la variación "
                 "relativa con HO): lo que DEV sí distorsiona es el **nivel**, porque no trae el deterioro "
                 "macro de 2025. La estrategia se decide en OOT (pregunta 2 del comité de la clase 4).")
    mo.md(_txt)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Iso-aprobación, iso-riesgo e iso-pérdida: la frontera

    **Qué mirar.** La curva azul es la frontera del scorecard en OOT: para cada k (aprobar los k
    mejores) su tasa de malos. La curva gris es la familia de la política vieja (mismos knock-outs,
    tope de carga variable). El punto negro es la política vieja elegida con el slider. Tres
    proyecciones: **vertical** (misma aprobación, menos mora: el swap-set del curso), **horizontal**
    (misma mora, más aprobación: iso-riesgo) e **iso-pérdida** en pesos (misma pérdida total o misma
    pérdida sobre monto, con LGD 45% y EAD = monto). La búsqueda del k se hace en numpy (sumas
    acumuladas) y con `pandas.expanding` (assert).
    """)
    return


@app.cell
def _(
    LGD,
    cartera,
    knock_outs,
    np,
    pd,
    politica_vieja,
    puntuar,
    sc_completo,
    sel_severidad,
    sl_aprob,
):
    oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)
    _, score_oot = puntuar(sc_completo, oot)
    y_oot = oot["malo"].to_numpy()
    monto_oot = oot["monto_mm"].to_numpy()
    _ko = knock_outs(oot, sel_severidad.value).any(axis=1).to_numpy()
    _ap_v = politica_vieja(oot, _ko, sl_aprob.value)
    _orden = np.argsort(-score_oot, kind="stable")
    _ys, _ms = y_oot[_orden], monto_oot[_orden]
    _k = np.arange(1, len(oot) + 1)
    curva_br = np.cumsum(_ys) / _k
    _perd_acum = np.cumsum(LGD * _ys * _ms)
    _monto_acum = np.cumsum(_ms)
    _curva_tasa_perdida = _perd_acum / _monto_acum

    _br_v = y_oot[_ap_v].mean()
    _perd_v = (LGD * y_oot[_ap_v] * monto_oot[_ap_v]).sum()
    _monto_v = monto_oot[_ap_v].sum()
    _k_v = int(_ap_v.sum())


    def _ultimo_k(cond):
        idx = np.flatnonzero(cond)
        return int(idx[-1] + 1) if len(idx) else 0


    k_iso = {
        "iso-aprobación": _k_v,
        "iso-riesgo (tasa de malos)": _ultimo_k(curva_br <= _br_v),
        "iso-pérdida / monto": _ultimo_k(_curva_tasa_perdida <= _perd_v / _monto_v),
        "iso-pérdida total ($)": _ultimo_k(_perd_acum <= _perd_v),
        "iso-volumen ($ colocado)": int(np.searchsorted(_monto_acum, _monto_v) + 1),
    }
    # misma búsqueda con pandas (expanding) para iso-riesgo
    _exp = pd.Series(_ys).expanding().mean()
    assert k_iso["iso-riesgo (tasa de malos)"] == int(_exp[_exp <= _br_v].index.max() + 1)

    _filas = [{"política": "vieja (KO + tope de carga)", "aprobación": _k_v / len(oot), "tasa malos": _br_v,
               "pérdida/monto": _perd_v / _monto_v, "pérdida MM CLP": _perd_v, "monto MM CLP": _monto_v}]
    for _nom, _kk in k_iso.items():
        _filas.append({"política": f"scorecard {_nom}", "aprobación": _kk / len(oot), "tasa malos": curva_br[_kk - 1],
                       "pérdida/monto": _curva_tasa_perdida[_kk - 1], "pérdida MM CLP": _perd_acum[_kk - 1],
                       "monto MM CLP": _monto_acum[_kk - 1]})
    tabla_iso = pd.DataFrame(_filas).set_index("política")

    # familia de la política vieja (para la curva gris)
    familia_vieja = []
    for _a in np.linspace(0.55, 0.97, 22):
        _apx = politica_vieja(oot, _ko, _a)
        familia_vieja.append((_apx.mean(), y_oot[_apx].mean()))
    familia_vieja = np.array(familia_vieja)
    return curva_br, familia_vieja, k_iso, monto_oot, oot, score_oot, tabla_iso, y_oot


@app.cell
def _(curva_br, familia_vieja, k_iso, mo, np, oot, pct, plt, tabla_iso):
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    _x = np.arange(1, len(oot) + 1) / len(oot)
    _ax.plot(_x * 100, curva_br * 100, color="tab:blue", label="scorecard (top-k)")
    _ax.plot(familia_vieja[:, 0] * 100, familia_vieja[:, 1] * 100, color="gray", marker=".", label="política vieja (KO + tope de carga)")
    _a0 = tabla_iso.iloc[0]
    _ax.scatter([_a0["aprobación"] * 100], [_a0["tasa malos"] * 100], color="black", zorder=5, label="vieja elegida")
    for _nom, _mk in [("iso-aprobación", "v"), ("iso-riesgo (tasa de malos)", ">"), ("iso-pérdida total ($)", "s")]:
        _kk = k_iso[_nom]
        _ax.scatter([_kk / len(oot) * 100], [curva_br[_kk - 1] * 100], marker=_mk, s=60, zorder=6, label=_nom)
    _ax.set_xlim(50, 100)
    _ax.set_ylim(0, max(25, curva_br[-1] * 110))
    _ax.set_xlabel("Aprobación (% de solicitudes OOT)")
    _ax.set_ylabel("Tasa de malos de la cartera aprobada (%)")
    _ax.set_title("Frontera aprobación–mora en OOT")
    _ax.legend(fontsize=8, loc="upper left")
    _ax.grid(alpha=0.3)
    _t = tabla_iso.copy()
    for _c in ["aprobación", "tasa malos", "pérdida/monto"]:
        _t[_c] = _t[_c].map(lambda v: pct(v, 2))
    _t = _t.round(1)
    _gan = tabla_iso.iloc[2]["aprobación"] - tabla_iso.iloc[0]["aprobación"]
    mo.vstack([_fig, mo.ui.table(_t.reset_index(), selection=None),
               mo.md(f"**Lectura.** A igual mora (iso-riesgo) el scorecard aprueba **{pct(_gan)}** más del flujo. "
                     "Iso-pérdida total en pesos concede aún más aprobación que iso-riesgo cuando el scorecard "
                     "reordena hacia montos grandes de bajo riesgo; iso-volumen necesita menos créditos para colocar "
                     "los mismos pesos. Son cinco preguntas distintas y cada una tiene su cutoff: el comité tiene "
                     "que decir cuál está respondiendo.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. ¿Es significativo 9,4% vs 23,4%? Intervalos exactos con los números del curso

    **Qué mirar.** Las celdas de Banco Austral (OOT, iso-aprobación 90,2%) reconstruidas a conteos:
    ambas aprueban 51/1.680, swap-out 30/128, swap-in 12/128, ambas rechazan 26/68. Puedes cambiar
    los malos de las dos celdas de intercambio. Tres implementaciones del intervalo exacto
    (Clopper-Pearson): numpy (inversión de colas por bisección), `scipy.stats.binomtest(...).proportion_ci`
    y `statsmodels.proportion_confint(method="beta")`. Para la diferencia: Fisher exacto (numpy vs
    `scipy.stats.fisher_exact`) y el IC de Newcombe (numpy vs `statsmodels.confint_proportions_2indep`).
    """)
    return


@app.cell
def _(mo):
    num_malos_in = mo.ui.number(start=0, stop=128, step=1, value=12, label="Malos en swap-in (n = 128)")
    num_malos_out = mo.ui.number(start=0, stop=128, step=1, value=30, label="Malos en swap-out (n = 128)")
    sel_conf = mo.ui.dropdown(options={"90%": 0.10, "95%": 0.05, "99%": 0.01}, value="95%", label="Confianza")
    mo.hstack([num_malos_in, num_malos_out, sel_conf], justify="start")
    return num_malos_in, num_malos_out, sel_conf


@app.cell
def _(
    clopper_pearson_np,
    confint_proportions_2indep,
    fisher_np,
    newcombe_np,
    np,
    num_malos_in,
    num_malos_out,
    pd,
    proportion_confint,
    sel_conf,
    stats,
):
    alfa_ic = sel_conf.value
    _celdas_curso = {"ambas aprueban": (51, 1680), "swap-out (sale)": (int(num_malos_out.value), 128),
                    "swap-in (entra)": (int(num_malos_in.value), 128), "ambas rechazan": (26, 68)}
    _filas = []
    for _nom, (_k, _n) in _celdas_curso.items():
        _np_ic = clopper_pearson_np(_k, _n, alfa_ic)
        _sp = stats.binomtest(_k, _n).proportion_ci(confidence_level=1 - alfa_ic, method="exact")
        _sm = proportion_confint(_k, _n, alpha=alfa_ic, method="beta")
        assert np.allclose(_np_ic, (_sp.low, _sp.high), atol=1e-8)
        assert np.allclose(_np_ic, _sm, atol=1e-8)
        _filas.append({"celda": _nom, "malos": _k, "n": _n, "tasa": _k / _n,
                       "CP inf (numpy)": _np_ic[0], "CP sup (numpy)": _np_ic[1],
                       "CP inf (scipy)": _sp.low, "CP sup (statsmodels)": _sm[1]})
    tabla_ic_curso = pd.DataFrame(_filas).set_index("celda")

    _ki, _ko = _celdas_curso["swap-in (entra)"][0], _celdas_curso["swap-out (sale)"][0]
    p_fisher_np = fisher_np(_ki, 128 - _ki, _ko, 128 - _ko)
    _p_fisher_sp = float(stats.fisher_exact([[_ki, 128 - _ki], [_ko, 128 - _ko]]).pvalue)
    ic_newc_np = newcombe_np(_ki, 128, _ko, 128, alfa_ic)
    _ic_newc_sm = confint_proportions_2indep(_ki, 128, _ko, 128, method="newcomb", compare="diff", alpha=alfa_ic)
    assert np.isclose(p_fisher_np, _p_fisher_sp, rtol=1e-6)
    assert np.allclose(ic_newc_np, _ic_newc_sm, atol=1e-8)
    # cartera: vieja vs nueva, IC por separado vs IC de la diferencia pareada
    _A = 51
    br_curso_vieja = (_A + _ko) / 1808
    br_curso_nueva = (_A + _ki) / 1808
    ic_vieja = proportion_confint(_A + _ko, 1808, alpha=alfa_ic, method="beta")
    ic_nueva = proportion_confint(_A + _ki, 1808, alpha=alfa_ic, method="beta")
    ic_delta = (128 / 1808 * ic_newc_np[0], 128 / 1808 * ic_newc_np[1])
    return (
        alfa_ic,
        br_curso_nueva,
        br_curso_vieja,
        ic_delta,
        ic_newc_np,
        ic_nueva,
        ic_vieja,
        p_fisher_np,
        tabla_ic_curso,
    )


@app.cell
def _(
    alfa_ic,
    br_curso_nueva,
    br_curso_vieja,
    ic_delta,
    ic_newc_np,
    ic_nueva,
    ic_vieja,
    mo,
    np,
    num,
    p_fisher_np,
    pct,
    plt,
    sgn,
    tabla_ic_curso,
):
    _fig, _ax = plt.subplots(figsize=(7.5, 3.6))
    _etq = list(tabla_ic_curso.index) + ["cartera vieja", "cartera nueva"]
    _p = list(tabla_ic_curso["tasa"]) + [br_curso_vieja, br_curso_nueva]
    _lo = list(tabla_ic_curso["CP inf (numpy)"]) + [ic_vieja[0], ic_nueva[0]]
    _hi = list(tabla_ic_curso["CP sup (numpy)"]) + [ic_vieja[1], ic_nueva[1]]
    _y = np.arange(len(_etq))[::-1]
    _ax.errorbar(np.array(_p) * 100, _y, xerr=[(np.array(_p) - _lo) * 100, (np.array(_hi) - _p) * 100],
                 fmt="o", capsize=4, color="tab:blue")
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_etq)
    _ax.set_xlabel(f"Tasa de malos (%) · IC {int((1 - alfa_ic) * 100)}% Clopper-Pearson")
    _ax.set_title("Banco Austral OOT: celdas del swap-set y carteras")
    _ax.grid(alpha=0.3, axis="x")
    _t = tabla_ic_curso.copy()
    for _c in _t.columns[2:]:
        _t[_c] = _t[_c].map(lambda v: pct(v, 2))
    _solapan = ic_vieja[0] < ic_nueva[1]
    mo.vstack([
        _fig, mo.ui.table(_t.reset_index(), selection=None),
        mo.md(f"**Lectura.** Swap-in vs swap-out: Fisher exacto p = {num(p_fisher_np, 4)} (numpy = scipy); IC de "
              f"Newcombe de la diferencia [{pct(ic_newc_np[0])} ; {pct(ic_newc_np[1])}]. "
              f"Los IC de las **carteras** {'se solapan' if _solapan else 'no se solapan'} "
              f"({pct(ic_vieja[0], 2)}–{pct(ic_vieja[1], 2)} vs {pct(ic_nueva[0], 2)}–{pct(ic_nueva[1], 2)}), "
              f"pero eso no dice nada: las dos carteras comparten 1.680 créditos y ese ruido se cancela. "
              f"El IC correcto de Δ es el de la diferencia de las celdas escalado por s/K = 128/1.808: "
              f"[{sgn(ic_delta[0] * 100, 2)} ; {sgn(ic_delta[1] * 100, 2)}] pp.")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. La trampa sutil: en la vida real el swap-in no tiene desempeño

    En el curso la política vieja era **hipotética**: los knock-outs se aplicaron a posteriori sobre
    créditos que el banco sí había cursado, así que las cuatro celdas tenían desempeño. En un cambio de
    política real, la vieja **estuvo vigente**: los que rechazó nunca recibieron crédito. El swap-in
    son rechazados históricos sin desempeño y su tasa de malos se **estima** con un modelo entrenado
    solo con aprobados.

    **Qué mirar.** Reentrenamos el scorecard solo con los aprobados históricos de DEV
    (truncamiento), lo calibramos con un δ en los aprobados de OOT (lo único observable) y
    estimamos la mora del swap-in. El generador da la verdad. Dos mecanismos de selección:

    - **Solo knock-outs**: la política vieja rechazó por variables que el modelo ve (selección sobre
      observables: *missing at random* condicional a X). El problema es la **extrapolación**: el
      tramo de mora reciente nunca se vio en entrenamiento.
    - **Knock-outs + juicio del ejecutivo**: además rechazó al 5% con peor señal **privada**
      (correlacionada con el riesgo verdadero y no contenida en X). Selección sobre no observables
      (*missing not at random*): el swap-in queda **adversamente seleccionado**.
    """)
    return


@app.cell
def _(mo):
    sel_mecanismo = mo.ui.dropdown(options=["solo knock-outs (MAR)", "knock-outs + juicio del ejecutivo (MNAR)"],
                                   value="solo knock-outs (MAR)", label="Mecanismo de selección histórico")
    sel_mecanismo
    return (sel_mecanismo,)


@app.cell
def _(cartera, knock_outs, logit_np, np, sel_mecanismo):
    _ko = knock_outs(cartera, "base (mora ≤ 2 m · consultas ≥ 6 · sin renta)").any(axis=1).to_numpy()
    _rng = np.random.default_rng(1818)
    # señal privada del ejecutivo: log-odds verdadero + ruido (ve parte de lo que el modelo no ve)
    _senal = logit_np(cartera["pd_verdadera"].to_numpy()) + _rng.normal(0, 0.5, len(cartera))
    if sel_mecanismo.value.startswith("knock-outs + juicio"):
        _umbral = np.quantile(_senal[~_ko], 0.95)
        rechazo_hist = _ko | (_senal > _umbral)
    else:
        rechazo_hist = _ko.copy()
    return (rechazo_hist,)


@app.cell
def _(ajustar_scorecard, cartera, rechazo_hist):
    _es_dev = (cartera["muestra"] == "DEV").to_numpy()
    sc_truncado = ajustar_scorecard(cartera[_es_dev & ~rechazo_hist])
    return (sc_truncado,)


@app.cell
def _(
    aprobar_top_k,
    brentq,
    cartera,
    logit_np,
    np,
    pd,
    puntuar,
    rechazo_hist,
    sc_completo,
    sc_truncado,
    sigmoide,
):
    _es_oot = (cartera["muestra"] == "OOT").to_numpy()
    oot_r = cartera[_es_oot].reset_index(drop=True)
    ap_hist_oot = ~rechazo_hist[_es_oot]
    y_r = oot_r["malo"].to_numpy()
    pv_r = oot_r["pd_verdadera"].to_numpy()
    _pd_tr, score_tr = puntuar(sc_truncado, oot_r)
    _pd_co, _ = puntuar(sc_completo, oot_r)


    def calibrar_delta_np(lp, y, iters=50):
        """δ exacto por Newton en numpy: media de σ(lp + δ) = media de y."""
        d = 0.0
        for _ in range(iters):
            p = sigmoide(lp + d)
            paso = (p.sum() - y.sum()) / (p * (1 - p)).sum()
            d -= paso
            if abs(paso) < 1e-13:
                break
        return d


    # calibración PIT solo donde se observa: los aprobados históricos de OOT
    _lp_tr = logit_np(_pd_tr)
    delta_tr = calibrar_delta_np(_lp_tr[ap_hist_oot], y_r[ap_hist_oot])
    _delta_brent = brentq(lambda d: sigmoide(_lp_tr[ap_hist_oot] + d).mean() - y_r[ap_hist_oot].mean(), -5, 5, xtol=1e-14)
    assert np.isclose(delta_tr, _delta_brent, atol=1e-9)
    pdcal_tr = sigmoide(_lp_tr + delta_tr)
    _lp_co = logit_np(_pd_co)
    _delta_co = calibrar_delta_np(_lp_co[ap_hist_oot], y_r[ap_hist_oot])
    pdcal_co = sigmoide(_lp_co + _delta_co)

    k_r = int(ap_hist_oot.sum())
    ap_nueva_r, cutoff_r = aprobar_top_k(score_tr, k_r)
    celdas_r = {"ambas aprueban": ap_hist_oot & ap_nueva_r, "swap-out (sale)": ap_hist_oot & ~ap_nueva_r,
                "swap-in (entra)": ~ap_hist_oot & ap_nueva_r, "ambas rechazan": ~ap_hist_oot & ~ap_nueva_r}
    _filas = []
    for _nom, _m in celdas_r.items():
        _obs = _nom in ("ambas aprueban", "swap-out (sale)")
        _filas.append({"celda": _nom, "n": int(_m.sum()), "¿observable en la vida real?": "sí" if _obs else "NO",
                       "PD cal. modelo truncado": pdcal_tr[_m].mean(),
                       "PD cal. modelo sin truncar": pdcal_co[_m].mean(),
                       "PD verdadera (media)": pv_r[_m].mean(), "tasa observada (generador)": y_r[_m].mean()})
    tabla_realidad = pd.DataFrame(_filas).set_index("celda")
    m_in_r = celdas_r["swap-in (entra)"]
    return (
        ap_hist_oot,
        ap_nueva_r,
        calibrar_delta_np,
        celdas_r,
        cutoff_r,
        delta_tr,
        k_r,
        m_in_r,
        oot_r,
        pdcal_co,
        pdcal_tr,
        pv_r,
        score_tr,
        tabla_realidad,
        y_r,
    )


@app.cell
def _(
    ap_hist_oot,
    celdas_r,
    delta_tr,
    k_r,
    m_in_r,
    mo,
    num,
    pct,
    pdcal_co,
    pdcal_tr,
    pv_r,
    sel_mecanismo,
    sgn,
    tabla_realidad,
    y_r,
):
    _t = tabla_realidad.copy()
    for _c in _t.columns[2:]:
        _t[_c] = _t[_c].map(pct)
    est_in = float(pdcal_tr[m_in_r].mean())
    verdad_in = float(pv_r[m_in_r].mean())
    obs_in = float(y_r[m_in_r].mean())
    sesgo_total = est_in - verdad_in
    sesgo_truncamiento = est_in - float(pdcal_co[m_in_r].mean())
    _sesgo_especificacion = float(pdcal_co[m_in_r].mean()) - verdad_in
    _out = celdas_r["swap-out (sale)"]
    _s = int(m_in_r.sum())
    b_out_r = float(y_r[_out].mean())
    _A = float(y_r[celdas_r["ambas aprueban"]].sum())
    _br_vieja_r = float(y_r[ap_hist_oot].mean())
    br_nueva_est = (_A + est_in * _s) / k_r
    br_nueva_real = (_A + y_r[m_in_r].sum()) / k_r
    manski = (_A / k_r, (_A + _s) / k_r)
    mo.vstack([
        mo.md(f"**{sel_mecanismo.value}** · δ de calibración en aprobados OOT = {sgn(delta_tr, 3)} · "
              f"swap-in n = {_s}"),
        mo.ui.table(_t.reset_index(), selection=None),
        mo.md(f"""
    **Lectura.** El comité vería una mora estimada del swap-in de **{pct(est_in)}**; la verdad es
    **{pct(verdad_in)}** (observada en el generador: {pct(obs_in)}). Sesgo total {sgn(sesgo_total * 100, 1)} pp, que se
    descompone en truncamiento {sgn(sesgo_truncamiento * 100, 1)} pp (modelo entrenado solo con aprobados vs con
    todos) y especificación/selección por el score {sgn(_sesgo_especificacion * 100, 1)} pp (incluso el modelo con
    todos los datos subestima en la celda que él mismo eligió como buena).

    Cartera nueva **estimada** {pct(br_nueva_est, 2)} vs **real** {pct(br_nueva_real, 2)} (vieja {pct(_br_vieja_r, 2)}).
    Sin supuestos sobre el swap-in (cotas de Manski: su tasa ∈ [0, 1]) la cartera nueva solo se sabe en
    [{pct(manski[0], 2)} ; {pct(manski[1], 2)}]. El punto de quiebre es b_in = b_out = {pct(b_out_r)}: el
    swap-in podría ser {num(b_out_r / est_in, 1)}× peor que lo estimado antes de que el intercambio deje de convenir.
    Ese cociente, no la estimación puntual, es lo que se lleva al comité.
    """),
    ])
    return b_out_r, br_nueva_est, br_nueva_real, est_in, manski, obs_in, sesgo_total, sesgo_truncamiento, verdad_in


@app.cell
def _(mo):
    mo.md(r"""
    **Dónde nace el sesgo de truncamiento.** La tabla compara los bins de `meses_desde_mora_12m`
    del modelo con todos los datos y del modelo truncado. Con knock-out de mora ≤ 2 meses, los
    valores 1–2 nunca aparecen en el entrenamiento truncado: los cortes se recalculan sin ellos y,
    al aplicar el modelo, esos clientes caen en un bin cuyo WoE se estimó con clientes de 3–4 meses.
    El modelo **no sabe que no sabe**: asigna un WoE finito a un territorio que nunca vio.
    """)
    return


@app.cell
def _(
    binear,
    cartera,
    mo,
    pct,
    pd,
    rechazo_hist,
    sc_completo,
    sc_truncado,
    sgn,
):
    _v = "meses_desde_mora_12m"
    _oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)
    _es_oot = (cartera["muestra"] == "OOT").to_numpy()
    _rech = rechazo_hist[_es_oot]
    _filas = []
    for _nom, _sc in [("sin truncar", sc_completo), ("truncado", sc_truncado)]:
        _et, _ = binear(_oot[_v], ref=_sc["dev"][_v])
        _woe = _et.map(_sc["mapas"][_v]).fillna(0.0).to_numpy()
        for _r in [1, 2, 3, 6, 13]:
            _m = (_oot[_v].to_numpy() == _r)
            _filas.append({"modelo": _nom, "meses desde mora": _r, "bin asignado": _et[_m].iloc[0],
                           "WoE": round(float(_woe[_m][0]), 3),
                           "tasa real OOT": pct(float(_oot["malo"].to_numpy()[_m].mean())),
                           "% rechazado hist.": pct(float(_rech[_m].mean()))})
    tabla_bins_mora = pd.DataFrame(_filas)
    _w1 = tabla_bins_mora.query("modelo == 'truncado' and `meses desde mora` == 1")["WoE"].iloc[0]
    _w1c = tabla_bins_mora.query("modelo == 'sin truncar' and `meses desde mora` == 1")["WoE"].iloc[0]
    mo.vstack([mo.ui.table(tabla_bins_mora, selection=None),
               mo.md(f"WoE de «mora hace 1 mes»: {sgn(_w1c, 3)} con todos los datos vs {sgn(_w1, 3)} truncado "
                     f"(WoE menos negativo = el modelo truncado lo cree {'mejor' if _w1 > _w1c else 'peor'} de lo que es).")])
    return (tabla_bins_mora,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Aprender el swap-in: exploración aleatoria bajo la política

    Única forma de **observar** el swap-in sin supuestos: aprobar al azar una fracción ε de los que la
    política vieja rechaza (banda de exploración) o asignar al azar parte del flujo a la política
    challenger. Con propensión conocida ε, la tasa de malos de los explorados que caen en la región
    swap-in es un estimador **insesgado** (ponderación 1/ε: Horvitz-Thompson).

    **Qué mirar.** Tres estimadores de la mora del swap-in: (E0) el modelo truncado calibrado,
    sin exploración; (E1) tasa directa de los explorados en la región swap-in; (E2) modelo truncado
    recalibrado con un δ propio estimado sobre **todos** los rechazados explorados (más n, algo de
    sesgo si el desvío no es uniforme). Diseño alternativo: **exploración focalizada**, que solo
    aprueba al azar dentro de la región swap-in candidata (menos costo por dato útil). El flujo se
    escala remuestreando los rechazados de OOT. Costo = pérdida neta esperada de los explorados:
    LGD·malo·EAD − margen·EAD. El **valor de la información** se mide contra una decisión: «abrir el
    swap-in» conviene si LGD·b_in < margen.
    """)
    return


@app.cell
def _(mo):
    sl_eps = mo.ui.slider(0.01, 0.50, step=0.01, value=0.10, label="ε: fracción de rechazados explorada")
    num_flujo = mo.ui.number(start=500, stop=20000, step=500, value=3000, label="Solicitudes por mes")
    num_meses_exp = mo.ui.slider(1, 12, step=1, value=6, label="Meses de exploración")
    sl_margen = mo.ui.slider(0.04, 0.16, step=0.005, value=0.09, label="Margen neto de vida (% EAD)")
    mo.vstack([mo.hstack([sl_eps, num_meses_exp], justify="start"),
               mo.hstack([num_flujo, sl_margen], justify="start")])
    return num_flujo, num_meses_exp, sl_eps, sl_margen


@app.cell
def _(
    LGD,
    ap_hist_oot,
    logit_np,
    m_in_r,
    np,
    num_flujo,
    num_meses_exp,
    oot_r,
    pdcal_tr,
    sigmoide,
    y_r,
):
    # población de rechazados históricos de OOT (con y sin swap-in), escalada al flujo pedido
    _rech = ~ap_hist_oot
    _en_in = m_in_r[_rech]
    _y = y_r[_rech]
    _lp = logit_np(pdcal_tr[_rech])
    _monto = oot_r["monto_mm"].to_numpy()[_rech]
    tasa_rechazo_r = float(_rech.mean())
    n_rech_periodo = int(round(num_flujo.value * num_meses_exp.value * tasa_rechazo_r))
    theta_in = float(_y[_en_in].mean())                 # verdad a estimar (tasa del swap-in)
    E0 = float(sigmoide(_lp[_en_in]).mean())            # sin exploración
    _REPS = 200
    grilla_eps = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])


    def simular_exploracion(eps, focalizada=False, reps=_REPS, semilla=7):
        """Monte Carlo de la banda de exploración sobre la superpoblación de rechazados.

        Cada solicitud rechazada del periodo se aprueba al azar con prob. ε; equivale a
        m ~ Binomial(N, ε) explorados, cada uno una extracción iid de la población de
        rechazados (o solo de la región swap-in si la exploración es focalizada).
        Devuelve por réplica: E1, E2, n explorados en swap-in, pérdida y monto colocado.
        """
        rng = np.random.default_rng(semilla)
        pool = np.flatnonzero(_en_in) if focalizada else np.arange(len(_y))
        N = int(round(n_rech_periodo * (_en_in.mean() if focalizada else 1.0)))
        e1, e2 = np.empty(reps), np.full(reps, np.nan)
        n_in, perdida, colocado = np.empty(reps), np.empty(reps), np.empty(reps)
        for r in range(reps):
            idx = pool[rng.integers(0, len(pool), rng.binomial(N, eps))]
            yi, ini, lpi, mi = _y[idx], _en_in[idx], _lp[idx], _monto[idx]
            n_in[r] = ini.sum()
            e1[r] = yi[ini].mean() if ini.any() else E0
            if not focalizada and len(idx) > 0:
                d = 0.0                      # δ propio de los explorados (Newton)
                for _ in range(30):
                    p = sigmoide(lpi + d)
                    fp = (p * (1 - p)).sum()
                    paso = (p.sum() - yi.sum()) / fp
                    d = float(np.clip(d - paso, -8, 8))
                    if abs(paso) < 1e-10:
                        break
                e2[r] = sigmoide(_lp[_en_in] + d).mean()
            perdida[r] = (LGD * yi * mi).sum()
            colocado[r] = mi.sum()
        return e1, e2, n_in, perdida, colocado
    return E0, grilla_eps, n_rech_periodo, simular_exploracion, tasa_rechazo_r, theta_in


@app.cell
def _(
    E0,
    LGD,
    grilla_eps,
    np,
    pd,
    simular_exploracion,
    sl_eps,
    sl_margen,
    theta_in,
):
    _filas = []
    for _foc in (False, True):
        for _e in list(grilla_eps) + [sl_eps.value]:
            _e1, _e2, _nin, _perd, _col = simular_exploracion(_e, _foc)
            _costo = (_perd - sl_margen.value * _col).mean()
            _filas.append({"diseño": "focalizada" if _foc else "todos los rechazados", "eps": _e,
                           "n explorados en swap-in (media)": _nin.mean(),
                           "RMSE E0": abs(E0 - theta_in),
                           "RMSE E1": np.sqrt(((_e1 - theta_in) ** 2).mean()),
                           "sesgo E1": _e1.mean() - theta_in,
                           "RMSE E2": np.sqrt(np.nanmean((_e2 - theta_in) ** 2)) if not _foc else np.nan,
                           "P(decisión errada) E1": np.mean((LGD * _e1 < sl_margen.value) != (LGD * theta_in < sl_margen.value)),
                           "costo neto MM CLP": _costo, "pérdida MM CLP": _perd.mean()})
    tabla_exploracion = pd.DataFrame(_filas)
    return (tabla_exploracion,)


@app.cell
def _(
    E0,
    LGD,
    m_in_r,
    mo,
    n_rech_periodo,
    num,
    num_flujo,
    oot_r,
    pct,
    plt,
    sl_eps,
    sl_margen,
    tabla_exploracion,
    tasa_rechazo_r,
    theta_in,
):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for _dis, _col in [("todos los rechazados", "tab:blue"), ("focalizada", "tab:orange")]:
        _t = tabla_exploracion[tabla_exploracion["diseño"] == _dis].sort_values("eps").drop_duplicates("eps")
        _a1.plot(_t["eps"] * 100, _t["RMSE E1"] * 100, marker="o", color=_col, label=f"E1 directo · {_dis}")
        _a2.plot(_t["costo neto MM CLP"], _t["RMSE E1"] * 100, marker="o", color=_col, label=_dis)
    _t0 = tabla_exploracion[tabla_exploracion["diseño"] == "todos los rechazados"].sort_values("eps").drop_duplicates("eps")
    _a1.plot(_t0["eps"] * 100, _t0["RMSE E2"] * 100, ls="--", color="tab:green", label="E2 modelo + δ explorado")
    _a1.axhline(abs(E0 - theta_in) * 100, color="gray", ls=":", label="E0 sin explorar (sesgo)")
    _a1.set_xlabel("ε (% de rechazados aprobados al azar)")
    _a1.set_ylabel("RMSE de la mora del swap-in (pp)")
    _a1.set_title("Error de estimación vs ε")
    _a1.legend(fontsize=7)
    _a1.grid(alpha=0.3)
    _a2.axhline(abs(E0 - theta_in) * 100, color="gray", ls=":", label="E0 sin explorar")
    _a2.set_xlabel("Costo neto esperado de explorar (MM CLP)")
    _a2.set_ylabel("RMSE E1 (pp)")
    _a2.set_title("Qué compra cada peso de exploración")
    _a2.legend(fontsize=7)
    _a2.grid(alpha=0.3)
    _sel = tabla_exploracion[(tabla_exploracion["eps"] == sl_eps.value)].drop_duplicates("diseño")
    _r = _sel[_sel["diseño"] == "todos los rechazados"].iloc[0]
    _f = _sel[_sel["diseño"] == "focalizada"].iloc[0]
    _decide_E0 = "abrir" if LGD * E0 < sl_margen.value else "no abrir"
    _decide_v = "abrir" if LGD * theta_in < sl_margen.value else "no abrir"
    _n_in_mes = num_flujo.value * m_in_r.mean()
    _monto_in = float(oot_r["monto_mm"].to_numpy()[m_in_r].mean())
    _valor_mes = _n_in_mes * _monto_in * abs(sl_margen.value - LGD * theta_in)
    _t = _sel.copy()
    for _c in ["RMSE E0", "RMSE E1", "sesgo E1", "RMSE E2", "P(decisión errada) E1"]:
        _t[_c] = _t[_c].map(lambda v: pct(v, 2))
    mo.vstack([
        _fig,
        mo.ui.table(_t.round(2), selection=None),
        mo.md(f"""
    **Lectura (ε = {pct(sl_eps.value, 0)}).** Rechazados en el periodo: {num(n_rech_periodo, 0)} (tasa de rechazo
    {pct(tasa_rechazo_r)}). Verdad del swap-in {pct(theta_in)}; el modelo sin explorar dice {pct(E0)}.
    Explorando todos los rechazados, E1 queda con RMSE {_t.iloc[0]['RMSE E1']}
    a un costo neto de {num(_r['costo neto MM CLP'], 1)} MM CLP; la versión focalizada logra RMSE {_t.iloc[1]['RMSE E1']}
    con {num(_f['costo neto MM CLP'], 1)} MM CLP.

    **Decisión «abrir el swap-in»** (margen {pct(sl_margen.value)}; umbral de quiebre b* = margen/LGD =
    {pct(sl_margen.value / LGD)}): con E0 se decidiría **{_decide_E0}**; con la verdad, **{_decide_v}**.
    Cada mes de política en esa región mueve ≈ {num(_valor_mes, 1)} MM CLP de valor (≈ {_n_in_mes:.0f} créditos
    de {num(_monto_in, 2)} MM CLP). Si E0 y la verdad caen a lados distintos del umbral, la exploración
    se paga con pocos meses de política; si caen del mismo lado, la información vale poco *para esta
    decisión* (aunque sirva para recalibrar).
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. La cohorte swap-in en el tablero: test, potencia y fecha

    **Qué mirar.** El tablero de la clase 5 tiene la fila «cohorte swap-in: mora vs PD calibrada,
    9,4% vs 5,1% (p 0,04) · TTD: ago-2027». Reproducimos el test (numpy vs `scipy.stats.binomtest`)
    y calculamos la **potencia exacta** del test para detectar la brecha observada con cada n
    (numpy vs `scipy.stats.binom`). Con el flujo mensual del swap-in se obtiene cuántos meses de
    originación hacen falta y, sumando 12 meses de maduración, **la fecha en que la fila se puede
    leer con potencia 80%**.
    """)
    return


@app.cell
def _(mo):
    num_flujo_in = mo.ui.number(start=5, stop=2000, step=5, value=25, label="Créditos swap-in originados por mes")
    sl_p1 = mo.ui.slider(0.06, 0.15, step=0.005, value=0.094, label="Mora real del swap-in que se quiere detectar")
    mo.hstack([num_flujo_in, sl_p1], justify="start")
    return num_flujo_in, sl_p1


@app.cell
def _(binomtest_np, np, num_flujo_in, pd, potencia_binomial_np, sl_p1, stats):
    p_curso_np = binomtest_np(12, 128, 0.051)
    _p_curso_sp = float(stats.binomtest(12, 128, 0.051).pvalue)
    assert np.isclose(p_curso_np, _p_curso_sp, rtol=1e-9)
    grilla_n = np.array([50, 100, 128, 150, 200, 250, 300, 400, 500, 700, 1000])
    pot_np = np.array([potencia_binomial_np(int(_n), 0.051, sl_p1.value) for _n in grilla_n])


    def _potencia_scipy(n, p0, p1, alfa=0.05):
        xs = np.arange(n + 1)
        rechaza = np.array([stats.binomtest(int(x), n, p0).pvalue <= alfa for x in xs])
        return float(stats.binom.pmf(xs[rechaza], n, p1).sum())


    for _n in (128, 300):
        assert np.isclose(potencia_binomial_np(_n, 0.051, sl_p1.value), _potencia_scipy(_n, 0.051, sl_p1.value), atol=1e-10)
    _n80 = next((int(n) for n in range(20, 5001, 5) if potencia_binomial_np(n, 0.051, sl_p1.value) >= 0.80), None)
    n_80 = _n80
    _meses = int(np.ceil(n_80 / num_flujo_in.value)) if n_80 else None
    inicio = pd.Period("2026-10", "M")
    fecha_lectura = (inicio + _meses - 1 + 12) if _meses else None
    meses_originacion = _meses
    return (
        fecha_lectura,
        grilla_n,
        inicio,
        meses_originacion,
        n_80,
        p_curso_np,
        pot_np,
    )


@app.cell
def _(
    fecha_lectura,
    grilla_n,
    inicio,
    meses_originacion,
    mo,
    n_80,
    num,
    num_flujo_in,
    p_curso_np,
    pct,
    plt,
    pot_np,
    sl_p1,
):
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.plot(grilla_n, pot_np * 100, marker="o")
    _ax.axhline(80, color="gray", ls=":")
    _ax.axvline(128, color="tab:red", ls="--", label="n del curso (128)")
    _ax.set_xlabel("n de la cohorte swap-in")
    _ax.set_ylabel("Potencia (%)")
    _ax.set_title(f"Test binomial bilateral α=5%: H0 PD=5,1% vs real {pct(sl_p1.value)}")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _pot128 = pot_np[list(grilla_n).index(128)]
    mo.vstack([_fig, mo.md(f"""
    **Lectura.** El p-valor del curso se reproduce: binomial exacto 12/128 vs 5,1% → p = {num(p_curso_np, 3)}
    (numpy = scipy). Pero la potencia con n = 128 para una mora real de {pct(sl_p1.value)} es solo
    **{pct(_pot128, 0)}**: el {pct(1 - _pot128, 0)} de las veces un swap-in así de malo pasaría como 🟢. Para 80% se
    necesitan **n ≈ {n_80}**; con {num_flujo_in.value} créditos swap-in al mes son {meses_originacion} meses de
    originación desde {inicio} y, con 12 meses de maduración, la fila se lee con potencia en
    **{fecha_lectura}**. Esa fecha, el umbral y la acción se escriben antes de mirar el dato.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Swap-set por segmento y por montos

    **Qué mirar.** La matriz de la sección 2 (misma severidad, aprobación y muestra) abierta por
    `canal`: iso-aprobación **global** no es iso-aprobación **por segmento**, así que el cambio de
    política también mueve la mezcla de canales. Y la misma matriz en **pesos**: monto colocado y
    pérdida realizada (LGD 45% × malo × EAD) por celda. El intercambio puede ser 1:1 en créditos y
    no en pesos.
    """)
    return


@app.cell
def _(
    LGD,
    ap_nueva,
    ap_vieja,
    datos_swap,
    matriz,
    mo,
    np,
    num,
    pct,
    pd,
    y_swap,
):
    _filas = []
    _canal = datos_swap["canal"].to_numpy()
    for _c in ["sucursal", "web", "app", "fuerza_venta"]:
        _m = _canal == _c
        _in = _m & ~ap_vieja & ap_nueva
        _out = _m & ap_vieja & ~ap_nueva
        _filas.append({"canal": _c, "n": int(_m.sum()),
                       "aprob. vieja": pct(ap_vieja[_m].mean()), "aprob. nueva": pct(ap_nueva[_m].mean()),
                       "swap-in n": int(_in.sum()), "swap-in malos": pct(y_swap[_in].mean()) if _in.any() else "—",
                       "swap-out n": int(_out.sum()), "swap-out malos": pct(y_swap[_out].mean()) if _out.any() else "—",
                       "mora vieja": pct(y_swap[_m & ap_vieja].mean(), 2),
                       "mora nueva": pct(y_swap[_m & ap_nueva].mean(), 2)})
    tabla_segmentos = pd.DataFrame(_filas)
    _mm = matriz[["n", "monto_MM", "perdida_MM"]].copy()
    _mm["monto medio"] = _mm["monto_MM"] / _mm["n"].clip(lower=1)
    _mm["pérdida/monto"] = (_mm["perdida_MM"] / _mm["monto_MM"]).map(lambda v: pct(v, 2))
    _mon = datos_swap["monto_mm"].to_numpy()
    _perd = LGD * y_swap * _mon
    perdida_vieja = float(_perd[ap_vieja].sum())
    perdida_nueva = float(_perd[ap_nueva].sum())
    _monto_vieja = float(_mon[ap_vieja].sum())
    _monto_nueva = float(_mon[ap_nueva].sum())
    assert np.isclose(perdida_nueva - perdida_vieja,
                      matriz.loc["swap-in (entra)", "perdida_MM"] - matriz.loc["swap-out (sale)", "perdida_MM"])
    mo.vstack([
        mo.md("**Por canal**"), mo.ui.table(tabla_segmentos, selection=None),
        mo.md("**En pesos (MM CLP)**"), mo.ui.table(_mm.round(2).reset_index(), selection=None),
        mo.md(f"Pérdida realizada: vieja {num(perdida_vieja, 1)} MM CLP sobre {num(_monto_vieja, 0)} MM CLP colocados "
              f"({pct(perdida_vieja / _monto_vieja, 2)}) → nueva {num(perdida_nueva, 1)} MM CLP sobre {num(_monto_nueva, 0)} MM CLP "
              f"({pct(perdida_nueva / _monto_nueva, 2)}). Identidad en pesos: ΔPérdida = pérdida(swap-in) − "
              "pérdida(swap-out) (assert). La mezcla por canal cambia aunque la aprobación global sea la misma: "
              "si un canal tiene cupo o comisión propia, el iso-aprobación global no le sirve a su dueño."),
    ])
    return perdida_nueva, perdida_vieja, tabla_segmentos


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Checks del módulo

    Asserts de coincidencia numpy vs librería e invariantes teóricas. Si uno falla, el notebook falla.
    """)
    return


@app.cell
def _(
    E0,
    b_out_r,
    binom_pmf_np,
    br_nueva_real,
    clopper_pearson_np,
    confint_proportions_2indep,
    est_in,
    fisher_np,
    ic_delta,
    manski,
    matriz,
    mo,
    newcombe_np,
    np,
    p_curso_np,
    perdida_nueva,
    perdida_vieja,
    proportion_confint,
    sc_completo,
    sc_truncado,
    sesgo_total,
    simular_exploracion,
    stats,
    theta_in,
    verdad_in,
):
    _checks = []
    # 1. Números del curso: 4,48% → 3,48% y p ≈ 0,04
    assert round(81 / 1808, 4) == 0.0448 and round(63 / 1808, 4) == 0.0348
    assert np.isclose(63 / 1808 - 81 / 1808, 128 / 1808 * (12 / 128 - 30 / 128))
    assert round(p_curso_np, 2) == 0.04
    _checks.append("curso: 4,48% → 3,48% = s/K·(b_in − b_out); binomial 12/128 vs 5,1% p = 0,04")
    # 2. Identidad general (no iso): BR_n − BR_o = [n01(b01 − BR_o) − n10(b10 − BR_o)]/K_n
    _rng = np.random.default_rng(3)
    for _ in range(50):
        _n11, _n10, _n01 = _rng.integers(50, 500, 3)
        _b11, _b10, _b01 = _rng.random(3) * 0.4
        _bro = (_n11 * _b11 + _n10 * _b10) / (_n11 + _n10)
        _brn = (_n11 * _b11 + _n01 * _b01) / (_n11 + _n01)
        assert np.isclose(_brn - _bro, (_n01 * (_b01 - _bro) - _n10 * (_b10 - _bro)) / (_n11 + _n01))
    _checks.append("identidad general de la variación de mora (50 casos aleatorios)")
    # 3. Binomial numpy: pmf suma 1 y = scipy; CP = scipy = statsmodels en casos borde
    for _n, _p in [(128, 0.094), (1808, 0.03), (5, 0.5)]:
        assert np.isclose(binom_pmf_np(_n, _p).sum(), 1.0)
        assert np.allclose(binom_pmf_np(_n, _p), stats.binom.pmf(np.arange(_n + 1), _n, _p), atol=1e-12)
    for _k, _n in [(0, 50), (50, 50), (1, 7), (12, 128)]:
        assert np.allclose(clopper_pearson_np(_k, _n), proportion_confint(_k, _n, method="beta"), atol=1e-8)
    _checks.append("pmf y Clopper-Pearson numpy = scipy/statsmodels (incluye k = 0 y k = n)")
    # 4. Fisher y Newcombe numpy = librerías en varias tablas
    for _a, _b, _c, _d in [(12, 116, 30, 98), (0, 20, 5, 15), (3, 9, 4, 8)]:
        assert np.isclose(fisher_np(_a, _b, _c, _d), stats.fisher_exact([[_a, _b], [_c, _d]]).pvalue, rtol=1e-6)
        assert np.allclose(newcombe_np(_a, _a + _b, _c, _c + _d),
                           confint_proportions_2indep(_a, _a + _b, _c, _c + _d, method="newcomb"), atol=1e-8)
    _checks.append("Fisher exacto y Newcombe numpy = scipy/statsmodels")
    # 5. Diferencia pareada del curso significativa aunque los IC de las carteras se solapen
    assert ic_delta[1] < 0
    assert proportion_confint(63, 1808, method="beta")[1] > proportion_confint(81, 1808, method="beta")[0]
    _checks.append("Δ del curso: IC excluye 0 con IC de carteras solapados (diseño pareado)")
    # 6. IRLS numpy = GLM en ambos scorecards
    assert np.allclose(sc_completo["beta"], sc_completo["beta_irls"], atol=1e-6)
    assert np.allclose(sc_truncado["beta"], sc_truncado["beta_irls"], atol=1e-6)
    _checks.append("IRLS numpy = statsmodels GLM (completo y truncado)")
    # 7. Swap-set de la sección 2: iso y en pesos
    assert matriz.loc["swap-in (entra)", "n"] == matriz.loc["swap-out (sale)", "n"]
    assert np.isclose(perdida_nueva - perdida_vieja,
                      matriz.loc["swap-in (entra)", "perdida_MM"] - matriz.loc["swap-out (sale)", "perdida_MM"])
    _checks.append("iso-aprobación exacta; identidad de pérdida en pesos")
    # 8. La trampa: la estimación del swap-in subestima la verdad; Manski contiene lo real
    assert sesgo_total < 0 and est_in < verdad_in
    assert manski[0] <= br_nueva_real <= manski[1]
    assert b_out_r > est_in
    _checks.append("swap-in estimado < verdad; cotas de Manski contienen la cartera real")
    # 9. Exploración: E1 insesgado (|sesgo| < 3 EE Monte Carlo) y RMSE decreciente en ε
    _e1a, _, _, _, _ = simular_exploracion(0.05)
    _e1b, _, _, _, _ = simular_exploracion(0.30)
    _ee = _e1b.std() / np.sqrt(len(_e1b))
    assert abs(_e1b.mean() - theta_in) < 3 * _ee + 1e-3
    assert np.sqrt(((_e1b - theta_in) ** 2).mean()) < np.sqrt(((_e1a - theta_in) ** 2).mean())
    assert np.sqrt(((_e1b - theta_in) ** 2).mean()) < abs(E0 - theta_in)
    _checks.append("exploración: E1 insesgado, RMSE baja con ε y vence al modelo sin explorar")
    mo.md("**Todos los checks pasaron.**\n\n" + "\n".join(f"- {c}" for c in _checks))
    return


if __name__ == "__main__":
    app.run()

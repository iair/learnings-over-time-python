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
    import scipy.stats as stats
    import scipy.linalg as sla
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    from scipy.optimize import milp, LinearConstraint, Bounds
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    return (
        Bounds,
        LinearConstraint,
        LogisticRegression,
        PCA,
        milp,
        mo,
        plt,
        roc_auc_score,
        sch,
        sla,
        sm,
        squareform,
        stats,
        variance_inflation_factor,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # M09 · Redundancia, colinealidad y familias de variables

    **Serie 2 «Del embudo al gobierno»** · profundiza la clase 3 (correlación de WoE > 0,70 greedy
    por IV 63 → 26, VIF máx 2,97, clusterización conceptual) y la clase 4 v21 (familias de variables,
    redundancia condicional, reemplazo entre ventanas).

    Este notebook trabaja con **verdad conocida**: el generador `generar_cartera()` define el
    log-odds verdadero, y a esa cartera le agregamos *proxies* (misma serie con otra ventana, otro
    agregador, o una identidad contable) cuya relación con la verdad conocemos. Así podemos
    preguntar algo que con datos reales es imposible: **¿el filtro se quedó con el driver
    verdadero o con un proxy?**

    Secciones: (0) datos y pool · (1) ¿correlación sobre qué? · (2) el greedy y su óptimo ·
    (3) VIF, índice de condición, Belsley-Kuh-Welsch, GVIF y VIF ponderado ·
    (4) supresión y signos «equivocados» · (5) clustering de variables tipo VARCLUS ·
    (6) familias temporales: nivel + delta · (7) alternativas: L1 y PCA · checks.
    """)
    return


@app.cell
def _():
    # === Código común de la serie (pegado VERBATIM desde _spec/comun.py) ===
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
    ## Herramientas del módulo

    Cada cálculo central tiene una versión **numpy desde cero** (esta celda) y una versión de
    librería (en la sección correspondiente); la celda final verifica que coincidan.
    """)
    return


@app.cell
def _(np, pd):
    def fmt(x, d=3):
        """Número con coma decimal para la prosa."""
        return f"{x:.{d}f}".replace(".", ",")

    def sigm(z):
        return 1.0 / (1.0 + np.exp(-z))

    def logit_np(X, y, max_iter=60, tol=1e-10):
        """Regresión logística por Newton-Raphson (= IRLS). X ya trae la columna de unos.
        Devuelve (beta, cov, loglik). cov = (X'WX)^{-1} evaluada en el óptimo."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        beta = np.zeros(X.shape[1])
        for _ in range(max_iter):
            p = sigm(X @ beta)
            w = p * (1 - p)
            H = X.T @ (X * w[:, None])
            paso = np.linalg.solve(H, X.T @ (y - p))
            beta = beta + paso
            if np.max(np.abs(paso)) < tol:
                break
        p = sigm(X @ beta)
        w = p * (1 - p)
        cov = np.linalg.inv(X.T @ (X * w[:, None]))
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return beta, cov, ll

    def rangos_np(x):
        """Rangos promedio (empates → rango medio), 1-based, vectorizado."""
        x = np.asarray(x, float)
        o = np.argsort(x, kind="mergesort")
        xs = x[o]
        nuevo = np.r_[True, xs[1:] != xs[:-1]]
        grupo = np.cumsum(nuevo) - 1
        inicio = np.flatnonzero(nuevo)
        fin = np.r_[inicio[1:], len(x)]
        rango_medio = (inicio + 1 + fin) / 2.0
        r = np.empty(len(x))
        r[o] = rango_medio[grupo]
        return r

    def gini_np(y, s):
        """Gini = 2·AUC − 1, AUC por Mann-Whitney con rangos promedio. s alto = más riesgo."""
        y = np.asarray(y, float)
        r = rangos_np(s)
        n1 = y.sum()
        n0 = len(y) - n1
        auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        return 2 * auc - 1

    def pearson_np(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        ok = ~(np.isnan(a) | np.isnan(b))
        a = a[ok] - a[ok].mean()
        b = b[ok] - b[ok].mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    def spearman_np(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        ok = ~(np.isnan(a) | np.isnan(b))
        return pearson_np(rangos_np(a[ok]), rangos_np(b[ok]))

    def tabla_contingencia(a, b):
        ia, _ua = pd.factorize(pd.Series(a).astype(str))
        ib, _ub = pd.factorize(pd.Series(b).astype(str))
        T = np.zeros((ia.max() + 1, ib.max() + 1))
        np.add.at(T, (ia, ib), 1)
        return T

    def cramer_v_np(a, b):
        """V de Cramér = sqrt(χ² / (n·(min(r,c)−1))), χ² de Pearson sin corrección."""
        T = tabla_contingencia(a, b)
        n = T.sum()
        E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / n
        chi2 = ((T - E) ** 2 / E).sum()
        return float(np.sqrt(chi2 / (n * (min(T.shape) - 1))))

    def razon_eta_np(cat, x):
        """Razón de correlación η: sqrt(SS_entre / SS_total) de x por categorías."""
        x = np.asarray(x, float)
        g, _u = pd.factorize(pd.Series(cat).astype(str))
        m = x.mean()
        ss_tot = ((x - m) ** 2).sum()
        sumas = np.bincount(g, weights=x)
        cuentas = np.bincount(g)
        ss_entre = (cuentas * (sumas / cuentas - m) ** 2).sum()
        return float(np.sqrt(ss_entre / ss_tot))

    def vif_np(X):
        """VIF_j = [R^{-1}]_jj, R = matriz de correlación de las columnas de X."""
        R = np.corrcoef(np.asarray(X, float), rowvar=False)
        return np.diag(np.linalg.inv(R))

    def bkw_np(X):
        """Belsley-Kuh-Welsch: índices de condición y proporciones de descomposición de
        varianza. X SIN constante; se agrega intercepto y se escala cada columna a
        norma 1 (sin centrar, como recomiendan BKW). Devuelve (indices, props, phi)
        con props[k, j] = proporción de Var(b_j) asociada a la dimensión k."""
        X = np.asarray(X, float)
        Xc = np.column_stack([np.ones(len(X)), X])
        Xs = Xc / np.linalg.norm(Xc, axis=0)
        _u, d, Vt = np.linalg.svd(Xs, full_matrices=False)
        indices = d.max() / d
        phi = (Vt.T ** 2) / d ** 2            # phi[j, k] = v_jk² / μ_k²
        props = (phi / phi.sum(1, keepdims=True)).T
        return indices, props, phi, Xs

    def gvif_np(R, idx):
        """VIF generalizado de Fox & Monette (1992): det(R11)·det(R22)/det(R)."""
        idx = np.asarray(idx)
        resto = np.setdiff1d(np.arange(R.shape[0]), idx)
        ld = lambda M: np.linalg.slogdet(M)[1]
        return float(np.exp(ld(R[np.ix_(idx, idx)]) + ld(R[np.ix_(resto, resto)]) - ld(R)))

    def vif_ponderado_np(X, w):
        """VIF con los pesos W = p(1−p) de IRLS: diagonal de la inversa de la
        matriz de correlación ponderada (centrada con medias ponderadas)."""
        X = np.asarray(X, float)
        w = np.asarray(w, float)
        mu = (w[:, None] * X).sum(0) / w.sum()
        Xc = X - mu
        S = Xc.T @ (Xc * w[:, None])
        D = 1 / np.sqrt(np.diag(S))
        Rw = S * D[:, None] * D[None, :]
        return np.diag(np.linalg.inv(Rw))

    def greedy_corr(orden, C, umbral):
        """Greedy del curso: recorre `orden`; entra si |ρ| ≤ umbral con TODAS las ya elegidas."""
        sel, desc = [], {}
        for v in orden:
            choque = next((s for s in sel if abs(C.loc[v, s]) > umbral), None)
            if choque is None:
                sel.append(v)
            else:
                desc[v] = choque
        return sel, desc

    def mwis_np(nombres, C, umbral, pesos):
        """Conjunto independiente de peso máximo (exacto) por ramificación con memoria:
        MWIS(G) = max( MWIS(G − v), w_v + MWIS(G − N[v]) )."""
        n = len(nombres)
        A = np.abs(C.loc[nombres, nombres].values) > umbral
        np.fill_diagonal(A, False)
        w = np.array([float(pesos[v]) for v in nombres])
        vecinos = [frozenset(np.flatnonzero(A[i]).tolist()) for i in range(n)]
        memo = {}

        def rec(cand):
            if not cand:
                return 0.0, ()
            if cand in memo:
                return memo[cand]
            v = max(cand, key=lambda i: (len(vecinos[i] & cand), w[i], -i))
            if not (vecinos[v] & cand):
                val, s = rec(cand - {v})
                res = (val + w[v], s + (v,))
            else:
                v1, s1 = rec(cand - {v})
                v2, s2 = rec(cand - {v} - vecinos[v])
                res = (v1, s1) if v1 >= v2 + w[v] else (v2 + w[v], s2 + (v,))
            memo[cand] = res
            return res

        val, s = rec(frozenset(range(n)))
        return [nombres[i] for i in sorted(s)], val

    def aglomerativo_np(D):
        """Clustering jerárquico UPGMA (average linkage) desde cero.
        Distancia entre grupos = promedio de las distancias originales entre sus miembros.
        Devuelve lista de fusiones (id_a, id_b, altura, miembros)."""
        n = D.shape[0]
        activos = {i: [i] for i in range(n)}
        dist = {(i, j): float(D[i, j]) for i in range(n) for j in range(i + 1, n)}
        fusiones = []
        nuevo_id = n
        while len(activos) > 1:
            (a, b), h = min(dist.items(), key=lambda kv: kv[1])
            miembros = activos.pop(a) + activos.pop(b)
            dist = {k: v for k, v in dist.items() if a not in k and b not in k}
            for c, mc in activos.items():
                dist[(c, nuevo_id)] = float(D[np.ix_(mc, miembros)].mean())
            activos[nuevo_id] = miembros
            fusiones.append((a, b, h, miembros))
            nuevo_id += 1
        return fusiones

    def cortar_np(fusiones, n, altura):
        """Partición que resulta de aplicar las fusiones con altura ≤ `altura`."""
        etiqueta = np.arange(n)
        for _a, _b, h, miembros in fusiones:
            if h <= altura:
                etiqueta[miembros] = min(etiqueta[miembros])
        _u, canon = np.unique(etiqueta, return_inverse=True)
        return canon

    def particion_canonica(etiquetas):
        """Renombra etiquetas por orden de primera aparición (para comparar particiones)."""
        mapa = {}
        return np.array([mapa.setdefault(e, len(mapa)) for e in etiquetas])

    def corr_parcial_np(M):
        """Correlaciones parciales desde la matriz de precisión: −P_ij / sqrt(P_ii P_jj)."""
        P = np.linalg.inv(np.corrcoef(np.asarray(M, float), rowvar=False))
        d = 1 / np.sqrt(np.diag(P))
        Rp = -P * d[:, None] * d[None, :]
        np.fill_diagonal(Rp, 1.0)
        return Rp

    return (
        bkw_np,
        corr_parcial_np,
        cortar_np,
        cramer_v_np,
        fmt,
        gini_np,
        greedy_corr,
        gvif_np,
        logit_np,
        mwis_np,
        aglomerativo_np,
        particion_canonica,
        pearson_np,
        rangos_np,
        razon_eta_np,
        sigm,
        spearman_np,
        tabla_contingencia,
        vif_np,
        vif_ponderado_np,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Datos: la cartera sintética y un pool de candidatas con familias

    `generar_cartera()` trae 11 variables; su log-odds verdadero usa `uso_linea_prom_12m`,
    `uso_tc_prom_12m`, `uso_tc_prom_3m` (vía $\max(3m-12m,0)$, una **tendencia**), la recencia de
    mora, antigüedad, carga, deuda externa, consultas y canal. `edad` y `renta_mm` no entran
    directo (renta entra solo a través de `carga_financiera`).

    `ampliar_pool()` agrega lo que en un banco real aparece solo: la **misma serie con otra
    ventana** (`uso_linea_prom_3m/6m`, `uso_tc_prom_6m`), **otro agregador** (`uso_linea_max_12m`,
    `dias_mora_max_12m`, `n_meses_mora_12m`), un **submuestreo** (`consultas_3m`) y dos
    **identidades contables** (`deuda_total_mm`, `ratio_deuda_renta` ≈ `carga_financiera`).
    Ninguna de ellas agrega información sobre el riesgo que no esté ya en las originales: son
    proxies ruidosos. La columna `rol` lo declara.
    """)
    return


@app.cell
def _(np):
    def ampliar_pool(df, semilla=909):
        """Agrega proxies con estructura conocida (no agregan información sobre η)."""
        rng = np.random.default_rng(semilla)
        n = len(df)
        d = df.copy()
        u = d["uso_linea_prom_12m"].values
        d["uso_linea_prom_6m"] = np.clip(u + rng.normal(0, 0.045, n), 0, 1.2).round(4)
        d["uso_linea_prom_3m"] = np.clip(u + rng.normal(0, 0.085, n), 0, 1.2).round(4)
        d["uso_linea_max_12m"] = np.clip(u + np.abs(rng.normal(0, 0.12, n)), 0, 1.3).round(4)
        t3 = d["uso_tc_prom_3m"].values
        t12 = d["uso_tc_prom_12m"].values
        d["uso_tc_prom_6m"] = np.clip(0.5 * (t3 + t12) + rng.normal(0, 0.02, n), 0, 1.3).round(4)
        r = d["meses_desde_mora_12m"].values
        con_mora = (r >= 1) & (r <= 12)
        d["dias_mora_max_12m"] = np.where(
            con_mora, np.clip(np.round(15 + 75 * (13 - r) / 12 + rng.normal(0, 15, n)), 1, 120), 0.0)
        d.loc[r == -99, "dias_mora_max_12m"] = np.nan
        d["n_meses_mora_12m"] = np.where(
            con_mora, rng.binomial(np.clip(13 - r, 1, 12).astype(int), 0.35) + 1, 0).astype(float)
        d.loc[r == -99, "n_meses_mora_12m"] = np.nan
        d["consultas_3m"] = rng.binomial(d["consultas_6m"].values.astype(int), 0.55).astype(float)
        d["deuda_total_mm"] = (d["deuda_otras_prom_12m"] + 2.5 * u).round(3)
        d["ratio_deuda_renta"] = (d["deuda_total_mm"] / d["renta_mm"]).round(3)
        d["delta_uso_tc"] = (d["uso_tc_prom_3m"] - d["uso_tc_prom_12m"]).round(4)
        return d

    ROL = {
        "uso_linea_prom_12m": "driver verdadero",
        "uso_tc_prom_12m": "driver verdadero (nivel)",
        "uso_tc_prom_3m": "driver verdadero (vía tendencia 3m−12m)",
        "meses_desde_mora_12m": "driver verdadero",
        "antiguedad_meses": "driver verdadero",
        "deuda_otras_prom_12m": "driver verdadero",
        "carga_financiera": "driver verdadero",
        "consultas_6m": "driver verdadero",
        "canal": "driver verdadero",
        "edad": "sin efecto directo",
        "renta_mm": "indirecto (vía carga)",
        "uso_linea_prom_6m": "proxy: otra ventana de uso_linea",
        "uso_linea_prom_3m": "proxy: otra ventana de uso_linea",
        "uso_linea_max_12m": "proxy: otro agregador de uso_linea",
        "uso_tc_prom_6m": "proxy: promedio de 3m y 12m",
        "dias_mora_max_12m": "proxy: otro agregador de la mora",
        "n_meses_mora_12m": "proxy: otro agregador de la mora",
        "consultas_3m": "proxy: submuestra de consultas_6m",
        "deuda_total_mm": "proxy: identidad contable",
        "ratio_deuda_renta": "proxy: ≈ carga_financiera",
        "delta_uso_tc": "construida: 3m − 12m (tendencia)",
    }
    return ROL, ampliar_pool


@app.cell
def _(ROL, a_woe, ampliar_pool, generar_cartera, pd, tabla_woe):
    df_all = ampliar_pool(generar_cartera())
    dev = df_all[df_all["muestra"] == "DEV"].reset_index(drop=True)
    ho = df_all[df_all["muestra"] == "HO"].reset_index(drop=True)
    oot = df_all[df_all["muestra"] == "OOT"].reset_index(drop=True)
    candidatas = [v for v in ROL if v != "delta_uso_tc"]
    tabs_woe = {v: tabla_woe(dev[v], dev["malo"]) for v in ROL}
    mapas = {v: tabs_woe[v][0]["woe"] for v in ROL}
    iv = pd.Series({v: tabs_woe[v][1] for v in candidatas}).sort_values(ascending=False)
    W_dev = a_woe(dev, list(ROL), dev, mapas)
    W_ho = a_woe(ho, list(ROL), dev, mapas)
    W_oot = a_woe(oot, list(ROL), dev, mapas)
    y_dev = dev["malo"].values
    y_ho = ho["malo"].values
    y_oot = oot["malo"].values
    IV_MIN = 0.02
    pool = list(iv[iv >= IV_MIN].index)
    C_woe = W_dev[candidatas].corr()
    tabla_pool = pd.DataFrame({"iv_dev": iv.round(3), "rol": [ROL[v] for v in iv.index],
                               "en_pool": [v in pool for v in iv.index]})
    return (
        C_woe,
        IV_MIN,
        W_dev,
        W_ho,
        W_oot,
        candidatas,
        dev,
        ho,
        iv,
        mapas,
        oot,
        pool,
        tabla_pool,
        tabs_woe,
        y_dev,
        y_ho,
        y_oot,
    )


@app.cell
def _(IV_MIN, dev, fmt, gini_np, ho, mo, oot, pool, tabla_pool, y_dev):
    _g_dev = gini_np(y_dev, dev["pd_verdadera"].values)
    _g_ho = gini_np(ho["malo"].values, ho["pd_verdadera"].values)
    _g_oot = gini_np(oot["malo"].values, oot["pd_verdadera"].values)
    mo.vstack([
        mo.md(f"""
    DEV {len(dev)} · HO {len(ho)} · OOT {len(oot)} · tasa de malos DEV {fmt(100*y_dev.mean(),1)}%.
    **Techo de referencia** (Gini de la PD verdadera): DEV {fmt(_g_dev)} · HO {fmt(_g_ho)} ·
    OOT {fmt(_g_oot)}. Ningún modelo lo supera salvo por azar muestral. Ojo: el techo de DEV es
    *menor* que el de HO con esta semilla; por eso aquí el Gini DEV de los modelos sale bajo el de
    HO sin que eso signifique nada. Y da la escala del ruido: diferencias de ±0,01 de Gini entre
    estrategias de selección están dentro del azar muestral.

    Pool de trabajo: IV ≥ {fmt(IV_MIN,2)} → **{len(pool)} candidatas**. Se usa 0,02 y no el 0,10
    del curso a propósito: con 0,10 se caería `antiguedad_meses` (driver verdadero, IV ≈ 0,09) y el
    experimento de redundancia quedaría mezclado con otro filtro.
    """),
        tabla_pool,
    ])
    return


@app.cell
def _(W_dev, W_ho, W_oot, gini_np, np, pd, sm, y_dev, y_ho, y_oot):
    def evaluar_modelo(variables):
        """Logística (statsmodels) sobre WoE de DEV; Gini en DEV/HO/OOT y signos."""
        variables = list(variables)

        def _X(Wm):
            return sm.add_constant(Wm[variables].values, has_constant="add") if variables \
                else np.ones((len(Wm), 1))

        m = sm.Logit(y_dev, _X(W_dev)).fit(disp=0, maxiter=200)
        coef = pd.Series(m.params[1:], index=variables, dtype=float)
        pval = pd.Series(m.pvalues[1:], index=variables, dtype=float)
        return {
            "n": len(variables),
            "gini_dev": gini_np(y_dev, m.predict(_X(W_dev))),
            "gini_ho": gini_np(y_ho, m.predict(_X(W_ho))),
            "gini_oot": gini_np(y_oot, m.predict(_X(W_oot))),
            "n_pos": int((coef > 0).sum()),
            "coef": coef,
            "pval": pval,
            "llf": float(m.llf),
            "aic": float(m.aic),
            "modelo": m,
        }

    return (evaluar_modelo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. ¿Correlación sobre qué? Crudos, rangos, WoE y categóricas

    El curso calcula la correlación **sobre el WoE**, «exactamente lo que el modelo va a ver».
    La razón formal: el modelo es lineal en el WoE, así que la redundancia que importa es la
    redundancia *lineal en la escala de log-odds*. Pearson sobre crudos mide otra cosa, y se
    rompe con códigos especiales (−9, −99, 13), colas pesadas (ratios) y relaciones no monótonas.

    Qué mirar en la tabla: la columna `pearson_crudo` contra `pearson_woe`. Cuando difieren mucho,
    el filtro sobre crudos habría tomado una decisión distinta de la que el modelo necesita.
    """)
    return


@app.cell
def _(
    W_dev,
    binear,
    cramer_v_np,
    dev,
    np,
    pd,
    pearson_np,
    spearman_np,
    stats,
    tabla_contingencia,
):
    pares_corr = [
        ("meses_desde_mora_12m", "dias_mora_max_12m"),
        ("carga_financiera", "ratio_deuda_renta"),
        ("uso_tc_prom_12m", "uso_tc_prom_3m"),
        ("uso_linea_prom_12m", "consultas_6m"),
        ("consultas_6m", "consultas_3m"),
        ("deuda_otras_prom_12m", "deuda_total_mm"),
        ("carga_financiera", "deuda_total_mm"),
    ]
    _filas = []
    _chk = []
    for _a, _b in pares_corr:
        _xa = dev[_a].values.astype(float)
        _xb = dev[_b].values.astype(float)
        _ba = binear(dev[_a])[0]
        _bb = binear(dev[_b])[0]
        _ok = ~(np.isnan(_xa) | np.isnan(_xb))
        _filas.append({
            "par": f"{_a} ~ {_b}",
            "pearson_crudo": pearson_np(_xa, _xb),
            "spearman_crudo": spearman_np(_xa, _xb),
            "pearson_woe": pearson_np(W_dev[_a], W_dev[_b]),
            "spearman_woe": spearman_np(W_dev[_a], W_dev[_b]),
            "cramer_v_bins": cramer_v_np(_ba, _bb),
        })
        # librería: scipy
        _chk.append((
            _filas[-1]["pearson_crudo"], stats.pearsonr(_xa[_ok], _xb[_ok])[0],
            _filas[-1]["spearman_crudo"], stats.spearmanr(_xa[_ok], _xb[_ok])[0],
            _filas[-1]["cramer_v_bins"],
            stats.contingency.association(tabla_contingencia(_ba, _bb).astype(int), method="cramer"),
        ))
    tabla_corr = pd.DataFrame(_filas).set_index("par").round(3)
    chk_corr = np.array(_chk)
    return chk_corr, pares_corr, tabla_corr


@app.cell
def _(mo, tabla_corr):
    mo.vstack([mo.md("**Tabla 1.** Correlaciones en DEV (numpy; verificadas contra scipy en los checks)."),
               tabla_corr])
    return


@app.cell
def _(fmt, mo, tabla_corr):
    _t = tabla_corr
    _m = _t.loc["meses_desde_mora_12m ~ dias_mora_max_12m"]
    _c = _t.loc["deuda_otras_prom_12m ~ deuda_total_mm"]
    mo.md(f"""
    **Lectura.** `meses_desde_mora_12m ~ dias_mora_max_12m`: Pearson crudo {fmt(_m.pearson_crudo)}
    (los códigos −9/−99/13 lo destruyen: 13 = «sin mora» queda numéricamente *lejos* de 12 y −99
    aún más), mientras que en WoE es {fmt(_m.pearson_woe)}: **es la misma información** y el
    modelo la vería duplicada. Un filtro sobre crudos las dejaría pasar juntas.
    El error inverso: `deuda_otras_prom_12m ~ deuda_total_mm` da {fmt(_c.pearson_crudo)} crudo
    (la cola lognormal de la deuda externa domina ambas sumas) pero {fmt(_c.pearson_woe)} en WoE:
    en la escala del riesgo **no** son la misma variable, porque `deuda_total_mm` hereda la señal
    de `uso_linea` (2,5·uso) y la deuda externa casi no tiene IV. Un filtro sobre crudos habría
    descartado una de las dos por «redundante». Spearman crudo corrige escala y colas, pero no
    los códigos especiales cuyo orden numérico no es orden de riesgo.

    La V de Cramér sobre los bins no usa el orden ni el target: mide asociación entre
    etiquetas. Sirve para categóricas nominales, pero no dice si la asociación *apunta al riesgo
    en la misma dirección*, que es lo que infla varianzas en la regresión sobre WoE.
    """)
    return


@app.cell
def _(binear, np, pd, pearson_np, sigm, spearman_np, tabla_woe):
    # Juguete no monótono: el riesgo depende de |x − 0,5|; z es un proxy de |x − 0,5|.
    _rng = np.random.default_rng(11)
    _n = 20_000
    _x = _rng.random(_n)
    _z = np.abs(_x - 0.5) + _rng.normal(0, 0.04, _n)
    _y = (_rng.random(_n) < sigm(-2.6 + 7 * np.abs(_x - 0.5))).astype(float)
    _tx = tabla_woe(_x, _y)[0]["woe"]
    _tz = tabla_woe(_z, _y)[0]["woe"]
    _wx = binear(_x)[0].map(_tx).values
    _wz = binear(_z)[0].map(_tz).values
    toy_u = pd.Series({
        "pearson_crudo": pearson_np(_x, _z),
        "spearman_crudo": spearman_np(_x, _z),
        "pearson_woe": pearson_np(_wx, _wz),
    }).round(3)
    return (toy_u,)


@app.cell
def _(dev, fmt, mo, pd, razon_eta_np, sm, toy_u):
    _eta = razon_eta_np(dev["canal"], dev["uso_linea_prom_12m"])
    _Xd = sm.add_constant(pd.get_dummies(dev["canal"], drop_first=True).astype(float).values)
    _r2 = sm.OLS(dev["uso_linea_prom_12m"].values, _Xd).fit().rsquared
    eta_canal = (_eta, _r2 ** 0.5)
    mo.md(f"""
    **Juguete no monótono** (riesgo en U sobre $x$; $z \\approx |x-0{{,}}5|$): Pearson crudo
    {fmt(toy_u.pearson_crudo)}, Spearman crudo {fmt(toy_u.spearman_crudo)}, **Pearson sobre WoE
    {fmt(toy_u.pearson_woe)}**. Ni Pearson ni Spearman ven que $x$ y $z$ cuentan la misma historia
    de riesgo; el WoE sí, porque ambas quedan expresadas en la escala del target.

    **Mixto categórica–numérica**: razón de correlación $\\eta$ entre `canal` y
    `uso_linea_prom_12m` = {fmt(_eta)} (numpy) = $\\sqrt{{R^2}}$ de la OLS sobre dummies
    {fmt(eta_canal[1])} (statsmodels). En la regresión sobre WoE no hace falta: `canal` también
    es una columna de WoE y entra a la misma matriz de Pearson.
    """)
    return (eta_canal,)


@app.cell
def _(mo, pares_corr):
    par_corr = mo.ui.dropdown(options=[f"{a} ~ {b}" for a, b in pares_corr],
                              value=f"{pares_corr[0][0]} ~ {pares_corr[0][1]}",
                              label="Par a graficar")
    par_corr
    return (par_corr,)


@app.cell
def _(W_dev, dev, np, par_corr, pearson_np, plt):
    _a, _b = par_corr.value.split(" ~ ")
    _rng = np.random.default_rng(0)
    _i = _rng.choice(len(dev), 3000, replace=False)
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.8))
    _ax[0].scatter(dev[_a].values[_i], dev[_b].values[_i], s=4, alpha=0.3)
    _ax[0].set_xlabel(f"{_a} (crudo)")
    _ax[0].set_ylabel(f"{_b} (crudo)")
    _ax[0].set_title(f"Crudo: Pearson {pearson_np(dev[_a], dev[_b]):.3f}")
    _j = lambda v: v + _rng.normal(0, 0.03, len(v))
    _ax[1].scatter(_j(W_dev[_a].values[_i]), _j(W_dev[_b].values[_i]), s=4, alpha=0.3,
                   color="tab:orange")
    _ax[1].set_xlabel(f"WoE {_a} (con jitter)")
    _ax[1].set_ylabel(f"WoE {_b}")
    _ax[1].set_title(f"WoE: Pearson {pearson_np(W_dev[_a], W_dev[_b]):.3f}")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. El greedy del curso: dependencia del umbral y del orden, y su óptimo

    El greedy recorre las candidatas en un orden (el curso: IV descendente) y acepta cada una si
    su $|\rho|$ con **todas** las ya elegidas es ≤ umbral. Es un algoritmo para el problema de
    **conjunto independiente de peso máximo** (MWIS) en el grafo cuyas aristas son los pares con
    $|\rho|>u$: elegir vértices sin aristas entre ellos maximizando la suma de pesos (IV). El
    MWIS es NP-difícil en general; con 16–30 variables se resuelve exacto en milisegundos
    (ramificación en numpy, o programación entera con `scipy.optimize.milp`).

    Mueve el umbral y el orden. Mira: cuántas variables quedan, si quedan los **drivers
    verdaderos** o sus proxies, el Gini HO/OOT y cuántos coeficientes salen positivos (signo
    «equivocado» con WoE alto = bin bueno).
    """)
    return


@app.cell
def _(mo):
    umbral_greedy = mo.ui.slider(0.5, 1.0, step=0.05, value=0.7, label="Umbral |ρ| de WoE",
                                 show_value=True)
    orden_greedy = mo.ui.dropdown(
        options=["IV descendente (curso)", "Gini univariado DEV", "IV ascendente (peor caso)",
                 "Negocio: ventana 12m y variables base primero", "Aleatorio (semilla)"],
        value="IV descendente (curso)", label="Orden de recorrido")
    semilla_orden = mo.ui.number(start=1, stop=9999, step=1, value=7, label="Semilla (orden aleatorio)")
    mo.hstack([umbral_greedy, orden_greedy, semilla_orden])
    return orden_greedy, semilla_orden, umbral_greedy


@app.cell
def _(W_dev, gini_np, iv, pool, y_dev):
    gini_uni = {v: gini_np(y_dev, -W_dev[v].values) for v in pool}

    def ordenar_pool(criterio, semilla):
        import numpy as _np
        if criterio.startswith("IV descendente"):
            return list(pool)
        if criterio.startswith("Gini"):
            return sorted(pool, key=lambda v: -gini_uni[v])
        if criterio.startswith("IV ascendente"):
            return list(pool)[::-1]
        if criterio.startswith("Negocio"):
            base = {"meses_desde_mora_12m", "carga_financiera", "consultas_6m", "antiguedad_meses"}
            return sorted(pool, key=lambda v: (0 if (v.endswith("_12m") and "max" not in v
                                                     and "dias" not in v and "n_meses" not in v)
                                               or v in base else 1, -iv[v]))
        return list(_np.random.default_rng(int(semilla)).permutation(pool))

    return gini_uni, ordenar_pool


@app.cell
def _(
    C_woe,
    ROL,
    evaluar_modelo,
    greedy_corr,
    iv,
    mwis_np,
    orden_greedy,
    ordenar_pool,
    pd,
    pool,
    semilla_orden,
    umbral_greedy,
):
    _u = umbral_greedy.value
    orden_actual = ordenar_pool(orden_greedy.value, semilla_orden.value)
    sel_greedy, desc_greedy = greedy_corr(orden_actual, C_woe, _u)
    sel_mwis, _val = mwis_np(pool, C_woe, _u, iv)
    res_greedy = evaluar_modelo(sel_greedy)
    res_mwis = evaluar_modelo(sel_mwis)
    comp_greedy = pd.DataFrame({
        "greedy (orden elegido)": [res_greedy["n"], iv[sel_greedy].sum(), res_greedy["gini_dev"],
                                   res_greedy["gini_ho"], res_greedy["gini_oot"], res_greedy["n_pos"],
                                   sum(ROL[v].startswith("driver") for v in sel_greedy)],
        "MWIS exacto (máx Σ IV)": [res_mwis["n"], iv[sel_mwis].sum(), res_mwis["gini_dev"],
                                   res_mwis["gini_ho"], res_mwis["gini_oot"], res_mwis["n_pos"],
                                   sum(ROL[v].startswith("driver") for v in sel_mwis)],
    }, index=["n variables", "Σ IV", "Gini DEV", "Gini HO", "Gini OOT", "coef. positivos",
              "drivers verdaderos"]).round(3)
    detalle_greedy = pd.DataFrame({
        "iv": [iv[v] for v in sel_greedy],
        "rol": [ROL[v] for v in sel_greedy],
        "coef": res_greedy["coef"].values,
        "p_valor": res_greedy["pval"].values,
    }, index=sel_greedy).round(3)
    return comp_greedy, desc_greedy, detalle_greedy, res_greedy, sel_greedy, sel_mwis


@app.cell
def _(comp_greedy, desc_greedy, detalle_greedy, mo, orden_greedy, sel_mwis, umbral_greedy):
    _desc = ", ".join(f"`{k}`→`{v}`" for k, v in desc_greedy.items()) or "ninguna"
    mo.vstack([
        mo.md(f"**Umbral {umbral_greedy.value:.2f} · orden «{orden_greedy.value}»**".replace(".", ",", 1)),
        comp_greedy,
        mo.md("**Seleccionadas por el greedy** (coeficientes del modelo conjunto en DEV):"),
        detalle_greedy,
        mo.md(f"Descartadas (variable → con quién chocó): {_desc}"),
        mo.md(f"MWIS eligió: {', '.join('`'+v+'`' for v in sel_mwis)}"),
    ])
    return


@app.cell
def _(C_woe, evaluar_modelo, greedy_corr, np, ordenar_pool, plt, pool, umbral_greedy):
    # Distribución sobre 30 órdenes aleatorios al umbral elegido
    _res = []
    for _s in range(30):
        _sel, _d = greedy_corr(ordenar_pool("Aleatorio", 1000 + _s), C_woe, umbral_greedy.value)
        _r = evaluar_modelo(_sel)
        _res.append((_r["n"], _r["gini_ho"], _r["n_pos"]))
    dist_ordenes = np.array(_res)
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.2))
    _ax[0].hist(dist_ordenes[:, 0], bins=np.arange(dist_ordenes[:, 0].min() - 0.5,
                                                   dist_ordenes[:, 0].max() + 1.5), rwidth=0.8)
    _ax[0].set_xlabel("n° de variables que sobreviven")
    _ax[0].set_ylabel("n° de órdenes (de 30)")
    _ax[0].set_title("Greedy con orden aleatorio")
    _ax[1].hist(dist_ordenes[:, 1], bins=12, color="tab:green")
    _ax[1].set_xlabel("Gini HO del modelo con todas las sobrevivientes")
    _ax[1].set_ylabel("n° de órdenes")
    _ax[1].set_title(f"Rango Gini HO: {dist_ordenes[:,1].min():.3f} – {dist_ordenes[:,1].max():.3f}")
    _fig.tight_layout()
    _fig
    return (dist_ordenes,)


@app.cell
def _(C_woe, ROL, evaluar_modelo, greedy_corr, iv, mwis_np, np, pd, pool, vif_np, W_dev):
    _filas = []
    for _u in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        for _nom, _sel in [("greedy IV", greedy_corr(pool, C_woe, _u)[0]),
                           ("MWIS", mwis_np(pool, C_woe, _u, iv)[0])]:
            _r = evaluar_modelo(_sel)
            _filas.append({"umbral": _u, "método": _nom, "n": _r["n"],
                           "sum_iv": iv[_sel].sum(), "gini_ho": _r["gini_ho"],
                           "gini_oot": _r["gini_oot"], "coef_pos": _r["n_pos"],
                           "vif_max": vif_np(W_dev[_sel].values).max() if len(_sel) > 1 else 1.0,
                           "drivers": sum(ROL[v].startswith("driver") for v in _sel)})
    tabla_umbrales = pd.DataFrame(_filas).round(3)
    sel_07 = greedy_corr(pool, C_woe, 0.7)[0]
    res_07 = evaluar_modelo(sel_07)
    res_pool = evaluar_modelo(pool)
    return res_07, res_pool, sel_07, tabla_umbrales


@app.cell
def _(fmt, mo, res_07, res_pool, tabla_umbrales):
    _t = tabla_umbrales
    _g = _t[_t["método"] == "greedy IV"]
    mo.vstack([
        mo.md("**Tabla 2.** Greedy por IV vs MWIS exacto, por umbral (fijo, no depende de los controles)."),
        _t,
        mo.md(f"""
    **Lectura.** Entre 0,6 y 0,9 el Gini HO del greedy se mueve en
    {fmt(_g.gini_ho.min())}–{fmt(_g.gini_ho.max())}: **el umbral cambia *qué* variables y *cuántas*,
    casi no el Gini**. Sin filtro (umbral 1,0) el modelo con las {res_pool['n']} candidatas
    tiene {res_pool['n_pos']} coeficientes positivos y Gini HO {fmt(res_pool['gini_ho'])}, contra
    {fmt(res_07['gini_ho'])} con las {res_07['n']} del greedy a 0,70: la redundancia no compra
    discriminación, compra signos rotos. El MWIS maximiza $\\sum IV$, pero $\\sum IV$ de
    variables correlacionadas **no es aditivo** en información: más IV total no garantiza más
    Gini. El óptimo del problema combinatorio no es el óptimo del modelo; por eso el greedy
    se defiende como heurística razonable y no como solución.

    Fíjate además en *quién* sobrevive: con IV descendente el greedy toma
    `dias_mora_max_12m` (proxy) antes que `meses_desde_mora_12m` (driver verdadero), porque el
    proxy tiene IV algo mayor en DEV. El filtro de redundancia no distingue causa de síntoma;
    eso lo hace el modelador con criterio de negocio (y la lámina 18: disponibilidad,
    estabilidad, definición).
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. VIF, índice de condición, Belsley-Kuh-Welsch, GVIF y VIF ponderado

    **VIF.** $\mathrm{VIF}_j = 1/(1-R_j^2) = [R^{-1}]_{jj}$ (derivación en el `.md`, §3.3).
    El curso usa `variance_inflation_factor` de statsmodels con WoE estandarizados **y
    constante**. Sin constante, statsmodels calcula un $R^2$ *no centrado* y el VIF sale mal: lo
    mostramos.
    """)
    return


@app.cell
def _(W_dev, dev, np, pd, pool, sel_07, sm, variance_inflation_factor, vif_np):
    def vif_statsmodels(X, constante=True):
        X = np.asarray(X, float)
        if constante:
            Z = sm.add_constant((X - X.mean(0)) / X.std(0), has_constant="add")
            return np.array([variance_inflation_factor(Z, k) for k in range(1, Z.shape[1])])
        # comportamiento clásico (statsmodels < 0.15 no estandarizaba): R² no centrado
        try:
            return np.array([variance_inflation_factor(X, k, standardize=False)
                             for k in range(X.shape[1])])
        except TypeError:
            return np.array([variance_inflation_factor(X, k) for k in range(X.shape[1])])

    vif_07_np = vif_np(W_dev[sel_07].values)
    vif_07_sm = vif_statsmodels(W_dev[sel_07].values)
    _crudas = ["uso_linea_prom_12m", "antiguedad_meses", "edad", "consultas_6m", "uso_tc_prom_12m"]
    _Xc = dev[_crudas].values
    tabla_vif_trampa = pd.DataFrame({
        "vif_numpy (correcto)": vif_np(_Xc),
        "statsmodels con constante": vif_statsmodels(_Xc),
        "statsmodels SIN constante ni estandarizar": vif_statsmodels(_Xc, constante=False),
    }, index=_crudas).round(2)
    vif_pool_np = vif_np(W_dev[pool].values)
    vif_pool_sm = vif_statsmodels(W_dev[pool].values)
    tabla_vif = pd.DataFrame({
        "vif_numpy": vif_07_np, "vif_statsmodels": vif_07_sm,
    }, index=sel_07).round(3)
    tabla_vif_pool = pd.Series(vif_pool_np, index=pool, name="vif_pool").sort_values(ascending=False).round(2)
    return (
        tabla_vif,
        tabla_vif_pool,
        tabla_vif_trampa,
        vif_07_np,
        vif_07_sm,
        vif_pool_np,
        vif_pool_sm,
        vif_statsmodels,
    )


@app.cell
def _(fmt, mo, tabla_vif, tabla_vif_pool, tabla_vif_trampa, vif_07_np):
    mo.vstack([
        mo.md(f"""**Tabla 3.** VIF de las {len(vif_07_np)} sobrevivientes del greedy a 0,70
    (máximo {fmt(vif_07_np.max(),2)}, como el 2,97 de Austral: tras el filtro de a pares,
    el VIF casi nunca encuentra nada)."""),
        tabla_vif,
        mo.md("**VIF del pool completo** (sin filtro de correlación): las familias explotan."),
        tabla_vif_pool.to_frame().T,
        mo.md("""**La trampa de `variance_inflation_factor` sin constante**, sobre variables *crudas*
    (medias lejos de 0). La función regresa la columna $j$ contra las demás columnas *tal como
    se las pasas*; sin columna de unos y sin centrar, el $R^2$ es no centrado y el VIF mide la
    distancia al origen, no la colinealidad. statsmodels 0.15 agregó `standardize=True` por
    defecto (centra y escala), lo que neutraliza la trampa; en versiones anteriores había que
    agregar la constante a mano, como hace el código de clase. Con WoE casi no se nota porque su
    media es cercana a 0; con crudos, sí:"""),
        tabla_vif_trampa,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.2 Índice de condición y descomposición de proporciones de varianza (BKW 1980)

    El VIF dice *cuánto* se infla cada varianza, no *con quién* está enredada cada variable, ni
    cuántas dependencias distintas hay. BKW descomponen $\mathrm{Var}(b_j)$ en la SVD de $X$
    (escalada a columnas de norma 1): cada dimensión $k$ con índice de condición
    $\eta_k=\mu_{\max}/\mu_k$ alto (la práctica usa > 30 como alarma y 10–30 como zona gris) es una casi-dependencia, y
    las variables con proporción $\pi_{kj}>0{,}5$ en esa fila son las que participan.

    **Experimento con verdad conocida**: 8 variables con **dos** casi-dependencias separadas
    ($a_4 \approx (a_1+a_2+a_3)/\sqrt3$ y $b_3\approx(b_1+b_2)/\sqrt2$) y dos independientes. Ningún par
    supera $|\rho|=0{,}70$: **el filtro del curso las deja pasar a todas**.
    """)
    return


@app.cell
def _(bkw_np, np, pd, vif_np):
    _rng = np.random.default_rng(42)
    _n = 5000
    _a = _rng.normal(size=(_n, 3))
    _a4 = _a.sum(1) / np.sqrt(3) + _rng.normal(0, 0.05, _n)
    _b = _rng.normal(size=(_n, 2))
    _b3 = _b.sum(1) / np.sqrt(2) + _rng.normal(0, 0.2, _n)
    _c = _rng.normal(size=(_n, 2))
    X_bkw = np.column_stack([_a, _a4, _b, _b3, _c])
    nombres_bkw = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "c1", "c2"]
    _R = np.corrcoef(X_bkw, rowvar=False)
    max_par_bkw = np.max(np.abs(_R - np.eye(len(_R))))
    vif_bkw = pd.Series(vif_np(X_bkw), index=nombres_bkw)
    ind_bkw, props_bkw, phi_bkw, Xs_bkw = bkw_np(X_bkw)
    tabla_bkw = pd.DataFrame(props_bkw, columns=["const"] + nombres_bkw)
    tabla_bkw.insert(0, "indice_condicion", ind_bkw)
    tabla_bkw = tabla_bkw.sort_values("indice_condicion", ascending=False).round(2)
    return X_bkw, ind_bkw, max_par_bkw, phi_bkw, tabla_bkw, vif_bkw, Xs_bkw


@app.cell
def _(fmt, max_par_bkw, mo, tabla_bkw, vif_bkw):
    mo.vstack([
        mo.md(f"""Máximo $|\\rho|$ de a pares: **{fmt(max_par_bkw)}** (< 0,70). VIF:
    {", ".join(f"{k} {fmt(v,1)}" for k, v in vif_bkw.items())}."""),
        mo.md("**Tabla 4.** Proporciones de descomposición de varianza (filas = dimensiones, ordenadas por índice de condición)."),
        tabla_bkw.head(4),
        mo.md(r"""
    **Lectura.** Los VIF de $a_1..a_4$ y de $b_1..b_3$ son todos altos, pero el VIF no dice que
    hay **dos** problemas distintos. BKW sí: la fila de mayor índice de condición (> 30, alarma)
    concentra la varianza de $a_1..a_4$ ($\pi>0{,}5$); la segunda (10–30, zona gris) la de
    $b_1..b_3$; $c_1, c_2$ no aparecen. Nota de escala: $\eta_{\max}\ge\sqrt{\mathrm{VIF}_{\max}}$
    (derivación en el `.md`), así que «η > 30» es mucho más permisivo que «VIF > 10»: los dos
    umbrales son convenciones de severidad distinta, no la misma regla. La
    acción correcta es sacar (o combinar) **una variable por dependencia**, no todas las de VIF
    alto. Y a la inversa: una dependencia que involucra muchas variables con cargas chicas
    reparte la inflación entre ellas, y cada VIF individual puede quedar bajo el umbral 5
    mientras el índice de condición ya es alto.
    """),
    ])
    return


@app.cell
def _(W_dev, bkw_np, np, pd, pool):
    ind_pool, props_pool, _phi, _Xs = bkw_np(W_dev[pool].values)
    _orden = np.argsort(-ind_pool)
    _filas = []
    for _k in _orden[:5]:
        _vars = [(["const"] + pool)[j] for j in np.flatnonzero(props_pool[_k] > 0.3)]
        _filas.append({"indice_condicion": round(float(ind_pool[_k]), 1),
                       "variables con π > 0,3": ", ".join(_vars)})
    tabla_bkw_pool = pd.DataFrame(_filas)
    return ind_pool, tabla_bkw_pool


@app.cell
def _(mo, tabla_bkw_pool):
    mo.vstack([
        mo.md("""**BKW sobre el pool real (16 WoE, sin filtro).** Las cinco dimensiones peores y
    quién carga en cada una: cada fila es una **familia** reconocible."""),
        tabla_bkw_pool,
        mo.md("""Con WoE hay un matiz: el WoE no está centrado en 0 exacto y la constante participa
    en las dependencias (BKW no centran a propósito: la colinealidad con el intercepto también
    infla varianzas). El umbral 30 viene de la experiencia de BKW, no de una distribución."""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.3 VIF generalizado (Fox & Monette 1992) para grupos

    Cuando una variable ocupa **varias columnas** (dummies de bins, una familia completa),
    el VIF por columna no tiene sentido. El GVIF mide cuánto se infla el volumen de la región de
    confianza del *grupo*: $\mathrm{GVIF}=\det R_{11}\det R_{22}/\det R$. Para comparar entre grupos
    de distinto tamaño se usa $\mathrm{GVIF}^{1/(2\,df)}$ (escala de «inflación del error estándar»).
    Segunda implementación: vía correlaciones canónicas, $\mathrm{GVIF}=\prod_k 1/(1-\rho_k^2)$.
    """)
    return


@app.cell
def _(W_dev, gvif_np, np, pd, pool, sla):
    familias = {
        "uso_linea": ["uso_linea_prom_12m", "uso_linea_prom_6m", "uso_linea_prom_3m", "uso_linea_max_12m"],
        "uso_tc": ["uso_tc_prom_12m", "uso_tc_prom_6m", "uso_tc_prom_3m"],
        "mora": ["meses_desde_mora_12m", "dias_mora_max_12m", "n_meses_mora_12m"],
        "consultas": ["consultas_6m", "consultas_3m"],
        "deuda_carga": ["carga_financiera", "ratio_deuda_renta", "deuda_total_mm"],
    }

    def gvif_cca(X, idx):
        """GVIF vía correlaciones canónicas entre el grupo y el resto (scipy.linalg)."""
        X = np.asarray(X, float)
        idx = np.asarray(idx)
        resto = np.setdiff1d(np.arange(X.shape[1]), idx)
        S = np.cov(X, rowvar=False)
        Laa = sla.cholesky(S[np.ix_(idx, idx)], lower=True)
        Lbb = sla.cholesky(S[np.ix_(resto, resto)], lower=True)
        M = sla.solve_triangular(Laa, S[np.ix_(idx, resto)], lower=True)
        M = sla.solve_triangular(Lbb, M.T, lower=True).T
        rho = np.clip(sla.svdvals(M), 0, 1 - 1e-15)
        return float(np.prod(1 / (1 - rho ** 2))), rho

    _R = np.corrcoef(W_dev[pool].values, rowvar=False)
    _filas = []
    for _f, _m in familias.items():
        _idx = [pool.index(v) for v in _m if v in pool]
        _g1 = gvif_np(_R, _idx)
        _g2, _rho = gvif_cca(W_dev[pool].values, _idx)
        _filas.append({"familia": _f, "df": len(_idx), "gvif_det": _g1, "gvif_cca": _g2,
                       "gvif^(1/2df)": _g1 ** (1 / (2 * len(_idx))),
                       "corr_canonica_max": _rho.max()})
    tabla_gvif = pd.DataFrame(_filas).set_index("familia")
    return familias, gvif_cca, tabla_gvif


@app.cell
def _(mo, tabla_gvif):
    mo.vstack([
        mo.md("**Tabla 5.** GVIF de cada familia contra el resto del pool (fórmula de determinantes = correlaciones canónicas)."),
        tabla_gvif.round(3),
        mo.md(r"""**Lectura.** El GVIF de familia responde «¿cuánta información de esta familia ya está
    en el resto del modelo?». La correlación canónica máxima es el mejor $R$ que se puede lograr
    combinando linealmente la familia *y* el resto. Cuidado: el GVIF de familia mezcla la
    redundancia *hacia afuera* (con otras familias) con ninguna de la redundancia *interna*
    (el $\det R_{11}$ se cancela). Para la interna, VIF o BKW dentro de la familia."""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.4 VIF en logística: la versión ponderada por $W$ de IRLS

    En la logística $\widehat{\mathrm{Var}}(\hat\beta)=(X^\top W X)^{-1}$ con $W=\mathrm{diag}\{\hat p_i(1-\hat p_i)\}$.
    La inflación relevante es la de la correlación **ponderada** por $W$: pesa más a quien tiene
    $p$ cerca de 0,5 (en crédito, los riesgosos). La identidad exacta
    $\mathrm{VIF}^W_j=[(X^\top WX)^{-1}]_{jj}\cdot\sum_i w_i(x_{ij}-\bar x^w_j)^2$ nos da la segunda
    implementación: la covarianza de statsmodels.
    """)
    return


@app.cell
def _(W_dev, np, pd, res_07, sel_07, vif_07_np, vif_ponderado_np):
    _m = res_07["modelo"]
    _p = _m.predict()
    _w = _p * (1 - _p)
    _X = W_dev[sel_07].values
    vifw_np = vif_ponderado_np(_X, _w)
    _mu = (_w[:, None] * _X).sum(0) / _w.sum()
    _ss = ((_X - _mu) ** 2 * _w[:, None]).sum(0)
    vifw_sm = np.diag(np.asarray(_m.cov_params()))[1:] * _ss
    tabla_vifw = pd.DataFrame({"vif_clasico": vif_07_np, "vif_ponderado_numpy": vifw_np,
                               "vif_ponderado_desde_cov_statsmodels": vifw_sm},
                              index=sel_07).round(3)
    return tabla_vifw, vifw_np, vifw_sm


@app.cell
def _(fmt, mo, tabla_vifw):
    _d = (tabla_vifw["vif_ponderado_numpy"] - tabla_vifw["vif_clasico"])
    mo.vstack([
        tabla_vifw,
        mo.md(f"""**Lectura.** Diferencia máxima ponderado − clásico: {fmt(_d.abs().max())}
    (`{_d.abs().idxmax()}`: {fmt(tabla_vifw['vif_clasico'][_d.abs().idxmax()])} →
    {fmt(tabla_vifw['vif_ponderado_numpy'][_d.abs().idxmax()])}). Con PD promedio ~11% el peso
    $p(1-p)$ se concentra en el tramo riesgoso; si dos variables correlacionan distinto *entre los
    riesgosos* que en el promedio, el VIF clásico se equivoca en la dirección correspondiente.
    Aquí la deuda total y el ratio correlacionan *menos* en la zona que pesa, y el ponderado es
    menor. El ponderado es el que corresponde reportar si alguien pregunta «¿el VIF de una
    logística?»; depende de $\hat p$, o sea, del modelo ajustado."""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Supresión: signos «equivocados» y muy significativos

    **Referencia univariada.** Con una sola variable WoE sin suavizar, el MLE de la logística
    da exactamente $\beta=-1$ (derivación en el `.md`, §3.5). Con el suavizado 0,5 del código común,
    casi −1. En el modelo conjunto el coeficiente mide la parte **no redundante**: lo sano es
    $\beta_j\in(-1,0)$ aprox.; fuera de ese rango hay supresión, colinealidad o
    no-colapsabilidad (un $\beta$ algo menor que −1 aparece aun sin correlación, porque la
    logística no es colapsable).

    **El experimento.** $x_1,x_2$ son WoE estandarizados con correlación $\rho$; el log-odds
    verdadero es $\eta=a+\beta_1x_1+\beta_2x_2$ con $\beta_1=-0{,}8$. Elige $\rho$ y el
    coeficiente verdadero $\beta_2$ (negativo = signo esperado con WoE). 200 muestras de
    $n=2000$ por configuración.
    """)
    return


@app.cell
def _(W_dev, logit_np, np, pd, pool, y_dev):
    _filas = {}
    for _v in pool:
        _b, _c, _l = logit_np(np.column_stack([np.ones(len(y_dev)), W_dev[_v].values]), y_dev)
        _filas[_v] = _b[1]
    beta_univ = pd.Series(_filas, name="beta_univariado")
    return (beta_univ,)


@app.cell
def _(mo):
    rho_sup = mo.ui.slider(0.0, 0.98, step=0.02, value=0.9, label="ρ entre x1 y x2", show_value=True)
    b2_sup = mo.ui.slider(-0.4, 0.4, step=0.05, value=-0.1,
                          label="β2 verdadero de x2 (negativo = signo esperado)",
                          show_value=True)
    mo.hstack([rho_sup, b2_sup])
    return b2_sup, rho_sup


@app.cell
def _(logit_np, np):
    def simular_supresion(rho, b2, n=2000, R=200, b1=0.8, semilla=5):
        """Devuelve arrays (R,) de β̂1, β̂2, z2 (multivariado) y β̂2 univariado.
        Convención WoE: x alto = bueno, efecto 'correcto' = coeficiente negativo."""
        rng = np.random.default_rng(semilla)
        a = np.log(0.1 / 0.9)
        L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]) + 1e-12 * np.eye(2))
        out = np.zeros((R, 4))
        for r in range(R):
            Z = rng.normal(size=(n, 2)) @ L.T
            eta = a - b1 * Z[:, 0] - b2 * Z[:, 1]
            y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
            X = np.column_stack([np.ones(n), Z])
            beta, cov, _ll = logit_np(X, y)
            bu, _cu, _lu = logit_np(X[:, [0, 2]], y)
            out[r] = [beta[1], beta[2], beta[2] / np.sqrt(cov[2, 2]), bu[1]]
        return out

    return (simular_supresion,)


@app.cell
def _(b2_sup, fmt, mo, np, plt, rho_sup, simular_supresion):
    sim_sup = simular_supresion(rho_sup.value, -b2_sup.value)
    _b2hat, _z2, _bu = sim_sup[:, 1], sim_sup[:, 2], sim_sup[:, 3]
    _p_pos = (_b2hat > 0).mean()
    _p_pos_sig = ((_b2hat > 0) & (_z2 > 1.96)).mean()
    _se_ratio = np.std(_b2hat) / np.std(simular_supresion(0.0, -b2_sup.value, R=100)[:, 1])
    _fig, _ax = plt.subplots(figsize=(8, 3.4))
    _ax.hist(_bu, bins=30, alpha=0.6, label="β̂2 univariado (solo x2)")
    _ax.hist(_b2hat, bins=30, alpha=0.6, label="β̂2 en el modelo con x1 y x2")
    _ax.axvline(b2_sup.value, color="k", ls="--", label="valor verdadero")
    _ax.axvline(0, color="grey", lw=0.8)
    _ax.set_xlabel("coeficiente de x2 (negativo = signo esperado con WoE)")
    _ax.set_ylabel("n° de muestras (de 200)")
    _ax.set_title(f"ρ = {rho_sup.value:.2f}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"""
    Univariado, $x_2$ parece fuerte: $\\bar\\beta_2^{{univ}}$ = {fmt(_bu.mean())} (carga el efecto
    de $x_1$ vía $\\rho$). En el modelo conjunto: media {fmt(_b2hat.mean())}, desviación
    {fmt(_b2hat.std())} = **{fmt(_se_ratio,2)}×** la de $\\rho=0$ (teoría: $1/\\sqrt{{1-\\rho^2}}$ =
    {fmt(1/np.sqrt(1-rho_sup.value**2),2)}). **{fmt(100*_p_pos,1)}%** de las muestras dan signo positivo
    («equivocado») y **{fmt(100*_p_pos_sig,1)}%** positivo *y* significativo al 5%.

    La distinción que importa: la colinealidad **infla la varianza** (volteos de signo *no*
    significativos, el intervalo cruza el cero), pero **no sesga** $\hat\beta_2$. Un signo
    «equivocado» **y significativo** aparece cuando el efecto *condicional* verdadero tiene ese
    signo (mueve el control a $\beta_2>0$): supresión real o una variable omitida que el par
    está reconstruyendo (el caso `uso_tc` 3m/12m de la sección 6).
    """)])
    return (sim_sup,)


@app.cell
def _(b2_sup, np, plt, simular_supresion):
    # Curva: P(signo positivo y significativo) vs ρ, para el b2 elegido
    _rhos = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    curva_sup = []
    for _r in _rhos:
        _s = simular_supresion(_r, -b2_sup.value, R=100, semilla=9)
        curva_sup.append([(_s[:, 1] > 0).mean(), ((_s[:, 1] > 0) & (_s[:, 2] > 1.96)).mean()])
    curva_sup = np.array(curva_sup)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_rhos, curva_sup[:, 0], "o-", label="P(β̂2 > 0)")
    _ax.plot(_rhos, curva_sup[:, 1], "s-", label="P(β̂2 > 0 y p < 0,05)")
    _ax.set_xlabel("ρ entre x1 y x2")
    _ax.set_ylabel("probabilidad (100 muestras)")
    _ax.set_title(f"Signo 'equivocado' vs correlación (b2 verdadero = {b2_sup.value:.2f})")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return (curva_sup,)


@app.cell
def _(logit_np, np, pd, sigm):
    # Supresor clásico: x2 no tiene relación marginal con y, pero "limpia" el ruido de x1.
    _rng = np.random.default_rng(21)
    _n = 20_000
    _s = _rng.normal(size=_n)
    _se = 0.8
    _e = _rng.normal(0, _se, _n)
    _x1 = (_s + _e) / np.sqrt(1 + _se ** 2)
    _x2 = _e / _se
    _y = (_rng.random(_n) < sigm(np.log(0.1 / 0.9) - 1.0 * _s)).astype(float)
    _one = np.ones(_n)
    _b_u2, _c_u2, _ = logit_np(np.column_stack([_one, _x2]), _y)
    _b_12, _c_12, _ = logit_np(np.column_stack([_one, _x1, _x2]), _y)
    supresor_clasico = pd.DataFrame({
        "coef": [_b_u2[1], _b_12[1], _b_12[2]],
        "z": [_b_u2[1] / np.sqrt(_c_u2[1, 1]), _b_12[1] / np.sqrt(_c_12[1, 1]),
              _b_12[2] / np.sqrt(_c_12[2, 2])],
        "teórico (sin atenuación)": [0.0, -np.sqrt(1 + _se ** 2), _se],
    }, index=["x2 solo", "x1 (con x2)", "x2 (con x1)"]).round(3)
    return (supresor_clasico,)


@app.cell
def _(beta_univ, fmt, mo, res_pool, supresor_clasico):
    _pos = res_pool["coef"][res_pool["coef"] > 0]
    _txt_pos = ", ".join(f"`{v}` {fmt(c)} (p {fmt(res_pool['pval'][v])})" for v, c in _pos.items())
    mo.vstack([
        mo.md("**Supresor clásico** ($x_1=s+e$, $x_2=e$, el riesgo depende solo de $s$):"),
        supresor_clasico,
        mo.md(f"""
    $x_2$ solo no predice nada (IV ≈ 0), pero en el modelo conjunto sale **positivo y con z enorme**:
    le resta a $x_1$ su ruido. El modelo es *correcto* y el signo «equivocado» es real. Con WoE, esta
    forma pura casi nunca llega al modelo porque el IV ≥ 0,10 la filtra antes; lo que sí llega es
    la supresión **cooperativa/neta** del experimento anterior (dos proxies de lo mismo).

    **En el pool real** (16 WoE sin filtro de correlación) los coeficientes positivos son:
    {_txt_pos}. Coeficientes univariados de referencia (deberían ser ≈ −1):
    mín {fmt(beta_univ.min())}, máx {fmt(beta_univ.max())}.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Clustering de variables: dendrograma sobre $1-|\rho|$ y representante por cluster

    Versión aglomerativa del espíritu de `PROC VARCLUS` de SAS (que es divisivo y basado en
    componentes principales; ver `.md` §4). Distancia $d_{ij}=1-|\rho_{ij}|$ entre WoE,
    enlace promedio (UPGMA). Cortar a altura $h$ equivale a pedir que, *en promedio*, los
    miembros de un cluster tengan $|\rho|\ge 1-h$ (no cada par: eso sería enlace completo).

    Representante: **mínimo ratio $1-R^2$** $=\dfrac{1-R^2_{\text{propio}}}{1-R^2_{\text{vecino}}}$, donde
    $R^2_{\text{propio}}$ es con la 1.ª componente principal de su cluster y $R^2_{\text{vecino}}$ con
    la del cluster más cercano. Lo comparamos con «mayor IV» y con el rol verdadero.
    """)
    return


@app.cell
def _(mo):
    corte_cluster = mo.ui.slider(0.05, 0.8, step=0.05, value=0.3,
                                 label="Altura de corte (1 − |ρ| promedio)", show_value=True)
    corte_cluster
    return (corte_cluster,)


@app.cell
def _(C_woe, candidatas, np, sch, squareform):
    D_var = 1 - np.abs(C_woe.loc[candidatas, candidatas].values)
    np.fill_diagonal(D_var, 0.0)
    D_var = (D_var + D_var.T) / 2
    Z_link = sch.linkage(squareform(D_var, checks=False), method="average")
    return D_var, Z_link


@app.cell
def _(C_woe, ROL, Z_link, candidatas, corte_cluster, iv, np, pd, plt, sch):
    etiquetas_cl = sch.fcluster(Z_link, t=corte_cluster.value, criterion="distance")
    _R = C_woe.loc[candidatas, candidatas].values
    _pcs = {}
    for _k in np.unique(etiquetas_cl):
        _m = np.flatnonzero(etiquetas_cl == _k)
        _lam, _vec = np.linalg.eigh(_R[np.ix_(_m, _m)])
        _pcs[_k] = (_m, _vec[:, -1], _lam[-1])

    def _r2(j, k):
        _m, _v, _l = _pcs[k]
        return float((_R[j, _m] @ _v) ** 2 / _l)

    _filas = []
    for _j, _v in enumerate(candidatas):
        _k = etiquetas_cl[_j]
        _own = _r2(_j, _k)
        _otros = [_r2(_j, k2) for k2 in _pcs if k2 != _k]
        _next = max(_otros) if _otros else 0.0
        _filas.append({"variable": _v, "cluster": int(_k), "r2_propio": _own, "r2_vecino": _next,
                       "ratio_1_r2": (1 - _own) / (1 - _next), "iv": iv[_v], "rol": ROL[_v]})
    tabla_clusters = pd.DataFrame(_filas).sort_values(["cluster", "ratio_1_r2"]).round(3)
    _rep = tabla_clusters.groupby("cluster").agg(
        miembros=("variable", "size"),
        rep_ratio=("variable", "first"),
        rep_iv=("variable", lambda s: tabla_clusters.loc[s.index].sort_values("iv").iloc[-1]["variable"]))
    representantes = _rep
    _fig, _ax = plt.subplots(figsize=(10, 4.2))
    sch.dendrogram(Z_link, labels=candidatas, color_threshold=corte_cluster.value, ax=_ax,
                   leaf_rotation=90, leaf_font_size=8)
    _ax.axhline(corte_cluster.value, color="k", ls="--", lw=1)
    _ax.set_ylabel("distancia 1 − |ρ| (enlace promedio)")
    _ax.set_title(f"Dendrograma de las {len(candidatas)} candidatas · {len(_pcs)} clusters al corte")
    _fig.tight_layout()
    fig_dendro = _fig
    return etiquetas_cl, fig_dendro, representantes, tabla_clusters


@app.cell
def _(ROL, fig_dendro, mo, representantes, tabla_clusters):
    _multi = representantes[representantes["miembros"] > 1]
    _difieren = int((_multi["rep_ratio"] != _multi["rep_iv"]).sum())
    _drv_ratio = int(sum(ROL[v].startswith("driver") for v in _multi["rep_ratio"]))
    mo.vstack([
        fig_dendro,
        mo.md("**Representante por cluster**: por mínimo ratio $1-R^2$ vs por mayor IV."),
        representantes,
        mo.accordion({"Tabla completa (R² propio, vecino, ratio)": tabla_clusters}),
        mo.md(f"""
    **Lectura.** Al corte elegido hay {len(_multi)} clusters con más de un miembro; en
    {_difieren} de ellos el representante por ratio $1-R^2$ difiere del de mayor IV, y en
    {_drv_ratio} de {len(_multi)} el representante por ratio es un driver verdadero. El ratio elige
    a la variable **más central** de su cluster (la que mejor lo resume y menos se parece a los
    vecinos): es un criterio *sin target*. El IV elige a la más predictiva. En el cluster de mora
    ambos criterios eligen `dias_mora_max_12m` (proxy) y no `meses_desde_mora_12m` (driver):
    ningún criterio estadístico garantiza elegir la causa. Mueve el corte: a 0,3 los usos de
    línea y de tarjeta caen en **un solo** cluster (enlace promedio ≈ |ρ| 0,7 entre familias),
    que el modelador probablemente querría separar por concepto de negocio.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Familias temporales: redundancia condicional y «nivel + delta»

    La lámina 18 de la clase 4: tres ventanas de la misma serie pueden sobrevivir al filtro y
    seguir contando la misma historia; al sacar una, entra la otra. Aquí:

    1. para la familia elegida, el aporte **condicional** de cada miembro (correlación parcial con
       el target dado el líder y el resto del modelo, y test de razón de verosimilitud);
    2. el **reemplazo entre ventanas**: qué pasa con los coeficientes si se saca al líder.
    """)
    return


@app.cell
def _(mo):
    familia_sel = mo.ui.dropdown(options=["uso_tc", "uso_linea", "mora", "consultas", "deuda_carga"],
                                 value="uso_tc", label="Familia")
    familia_sel
    return (familia_sel,)


@app.cell
def _(
    W_dev,
    corr_parcial_np,
    evaluar_modelo,
    familia_sel,
    familias,
    iv,
    np,
    pd,
    sm,
    stats,
    y_dev,
):
    BASE = ["uso_linea_prom_12m", "meses_desde_mora_12m", "uso_tc_prom_12m", "consultas_6m",
            "carga_financiera", "antiguedad_meses"]
    _fam = familias[familia_sel.value]
    _lider = max(_fam, key=lambda v: iv[v])
    _base = [v for v in BASE if v not in _fam]
    _r_base_lider = evaluar_modelo(_base + [_lider])
    _r_todas = evaluar_modelo(_base + _fam)
    _r_sin_lider = evaluar_modelo(_base + [v for v in _fam if v != _lider])
    _filas = []
    for _v in _fam:
        _cond = _base + ([] if _v == _lider else [_lider])
        _M = np.column_stack([y_dev, W_dev[_v].values, W_dev[_cond].values])
        _pc_np = corr_parcial_np(_M)[0, 1]
        # librería: residuos de OLS (statsmodels)
        _Xc = sm.add_constant(W_dev[_cond].values)
        _ry = sm.OLS(y_dev, _Xc).fit().resid
        _rv = sm.OLS(W_dev[_v].values, _Xc).fit().resid
        _pc_sm = float(np.corrcoef(_ry, _rv)[0, 1])
        _r0 = evaluar_modelo(_cond)
        _r1 = evaluar_modelo(_cond + [_v])
        _lr = 2 * (_r1["llf"] - _r0["llf"])
        _filas.append({
            "variable": _v, "iv": iv[_v], "condiciona_en": "base" if _v == _lider else "base + líder",
            "corr_parcial_numpy": _pc_np, "corr_parcial_ols": _pc_sm,
            "LR_incremental": _lr, "p_LR": stats.chi2.sf(_lr, 1),
            "coef_con_toda_familia": _r_todas["coef"][_v],
            "p_con_toda_familia": _r_todas["pval"][_v],
            "coef_sin_lider": np.nan if _v == _lider else _r_sin_lider["coef"][_v],
            "p_sin_lider": np.nan if _v == _lider else _r_sin_lider["pval"][_v],
        })
    tabla_familia = pd.DataFrame(_filas).set_index("variable")
    lider_familia = _lider
    resumen_familia = pd.DataFrame({
        "base + líder": [_r_base_lider["gini_ho"], _r_base_lider["gini_oot"], _r_base_lider["n_pos"]],
        "base + toda la familia": [_r_todas["gini_ho"], _r_todas["gini_oot"], _r_todas["n_pos"]],
        "base + familia sin líder": [_r_sin_lider["gini_ho"], _r_sin_lider["gini_oot"], _r_sin_lider["n_pos"]],
    }, index=["Gini HO", "Gini OOT", "coef. positivos"])
    return BASE, lider_familia, resumen_familia, tabla_familia


@app.cell
def _(familia_sel, lider_familia, mo, resumen_familia, tabla_familia):
    mo.vstack([
        mo.md(f"**Familia `{familia_sel.value}`** · líder por IV: `{lider_familia}`"),
        tabla_familia.round(4),
        resumen_familia.round(3),
        mo.md(r"""
    **Lectura.** Una correlación parcial chica y un LR no significativo dicen que el miembro
    **no agrega** una vez que está el líder: es redundancia condicional, aunque su $|\rho|$
    marginal con el líder haya quedado bajo 0,70. Con toda la familia dentro aparecen coeficientes
    inestables o positivos; al sacar al líder, **otro miembro toma su lugar** con coeficiente
    fuerte y el Gini casi no cambia: eso es el reemplazo entre ventanas. Decisión a nivel de
    familia, no de variable (lámina 18).
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.2 `uso_tc` 3m y 12m: ambas vs «nivel + delta»

    En el generador el riesgo depende del **nivel** (12m) y de la **tendencia**
    $\max(3m-12m,0)$. Meter 3m y 12m juntas obliga al modelo a reconstruir la tendencia como
    diferencia de dos coeficientes casi colineales. Re-parametrizar como nivel + delta
    ($\Delta = 3m-12m$, con su propio binning y WoE en DEV) separa los dos conceptos. Base: los
    otros drivers. Bootstrap de DEV (B = 200) con logística en numpy.
    """)
    return


@app.cell
def _(W_dev, W_ho, W_oot, gini_np, logit_np, np, pd, vif_np, y_dev, y_ho, y_oot):
    _base = ["uso_linea_prom_12m", "meses_desde_mora_12m", "antiguedad_meses", "consultas_6m",
             "carga_financiera"]
    especs_familia = {
        "ninguna": [],
        "solo 12m": ["uso_tc_prom_12m"],
        "solo 3m": ["uso_tc_prom_3m"],
        "ambas (3m + 12m)": ["uso_tc_prom_12m", "uso_tc_prom_3m"],
        "nivel + delta (12m + Δ)": ["uso_tc_prom_12m", "delta_uso_tc"],
    }
    _rng = np.random.default_rng(2026)
    _B = 200
    _idx_boot = [_rng.integers(0, len(y_dev), len(y_dev)) for _ in range(_B)]
    _filas = []
    boot_familia = {}
    for _nom, _s in especs_familia.items():
        _vars = _base + _s
        _X = np.column_stack([np.ones(len(y_dev)), W_dev[_vars].values])
        _b, _cov, _ll = logit_np(_X, y_dev)
        _ph = 1 / (1 + np.exp(-np.column_stack([np.ones(len(y_ho)), W_ho[_vars].values]) @ _b))
        _po = 1 / (1 + np.exp(-np.column_stack([np.ones(len(y_oot)), W_oot[_vars].values]) @ _b))
        _bs = np.array([logit_np(_X[i], y_dev[i])[0] for i in _idx_boot]) if _s else None
        boot_familia[_nom] = _bs
        _fila = {"especificación": _nom, "gini_ho": gini_np(y_ho, _ph), "gini_oot": gini_np(y_oot, _po),
                 "loglik_dev": _ll, "aic": -2 * _ll + 2 * _X.shape[1],
                 "vif_max": vif_np(W_dev[_vars].values).max()}
        for _i, _v in enumerate(_s):
            _k = len(_base) + 1 + _i
            _fila[f"β[{_v}]"] = _b[_k]
            _fila[f"sd_boot[{_v}]"] = _bs[:, _k].std()
            _fila[f"%boot>0[{_v}]"] = 100 * (_bs[:, _k] > 0).mean()
        _filas.append(_fila)
    tabla_nivel_delta = pd.DataFrame(_filas).set_index("especificación")
    corr_3m_12m = float(np.corrcoef(W_dev["uso_tc_prom_12m"], W_dev["uso_tc_prom_3m"])[0, 1])
    corr_nivel_delta = float(np.corrcoef(W_dev["uso_tc_prom_12m"], W_dev["delta_uso_tc"])[0, 1])
    return corr_3m_12m, corr_nivel_delta, especs_familia, tabla_nivel_delta


@app.cell
def _(corr_3m_12m, corr_nivel_delta, fmt, mo, tabla_nivel_delta):
    _t = tabla_nivel_delta
    _a = _t.loc["ambas (3m + 12m)"]
    _d = _t.loc["nivel + delta (12m + Δ)"]
    mo.vstack([
        _t.T.round(3),
        mo.md(f"""
    **Lectura.** Correlación WoE 3m–12m: {fmt(corr_3m_12m)}; nivel–delta: {fmt(corr_nivel_delta)}.
    Con **ambas**, $\\beta$(12m) = {fmt(_a['β[uso_tc_prom_12m]'])}: **signo positivo** (el modelo
    usa 12m como supresor para construir la tendencia), VIF máx {fmt(_a['vif_max'],2)} y
    {fmt(_a['%boot>0[uso_tc_prom_12m]'],0)}% de los bootstrap con signo positivo. Con **nivel + delta**
    ambos coeficientes negativos, VIF máx {fmt(_d['vif_max'],2)}, log-verosimilitud
    {fmt(_d['loglik_dev'],1)} vs {fmt(_a['loglik_dev'],1)} (mejor ajuste con los mismos grados de
    libertad) y Gini HO {fmt(_d['gini_ho'])} vs {fmt(_a['gini_ho'])}: **el Gini casi no se mueve; lo
    que cambia es que el modelo se puede firmar**. El delta es además un reason code legible
    («su uso de tarjeta subió en los últimos 3 meses»).
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Alternativas: L1 sobre WoE y PCA

    **L1 (lasso)** sobre WoE estandarizados: el camino de regularización muestra el **orden de
    entrada**; dentro de una familia el lasso tiende a quedarse con *uno* (y cuál, puede cambiar
    con la muestra). **PCA**: comprime sin mirar el target y produce componentes que mezclan
    familias; el Gini puede ser parecido, pero no hay reason codes ni control de signo por
    variable.
    """)
    return


@app.cell
def _(LogisticRegression, W_dev, np, pd, pool, res_pool, y_dev):
    _X = W_dev[pool].values
    _mu, _sd = _X.mean(0), _X.std(0)
    _Xs = (_X - _mu) / _sd
    Cs_l1 = np.logspace(-3.3, 0, 30)
    # API de L1 compatible con scikit-learn < 1.8 (penalty="l1") y ≥ 1.8 (l1_ratio=1)
    import sklearn as _sk
    _ver = tuple(int(p) for p in _sk.__version__.split(".")[:2])
    _l1 = {"l1_ratio": 1.0} if _ver >= (1, 8) else {"penalty": "l1"}
    _coefs = []
    for _C in Cs_l1:
        _m = LogisticRegression(**_l1, C=_C, solver="liblinear", tol=1e-6,
                                intercept_scaling=100, max_iter=2000)
        _m.fit(_Xs, y_dev)
        _coefs.append(_m.coef_[0])
    camino_l1 = pd.DataFrame(np.array(_coefs), index=Cs_l1, columns=pool)
    _entrada = {v: (camino_l1.index[np.flatnonzero(camino_l1[v].abs().values > 1e-8)[0]]
                    if (camino_l1[v].abs() > 1e-8).any() else np.nan) for v in pool}
    orden_entrada_l1 = pd.Series(_entrada, name="C_de_entrada").sort_values()
    # L1 con C muy grande ≈ MLE sin penalizar → debe reproducir la log-verosimilitud de statsmodels
    _mbig = LogisticRegression(**_l1, C=1e6, solver="liblinear", tol=1e-8,
                               intercept_scaling=1000, max_iter=10000).fit(_Xs, y_dev)
    _p = np.clip(_mbig.predict_proba(_Xs)[:, 1], 1e-12, 1 - 1e-12)
    ll_l1_grande = float(np.sum(y_dev * np.log(_p) + (1 - y_dev) * np.log(1 - _p)))
    ll_mle = res_pool["llf"]
    return camino_l1, ll_l1_grande, ll_mle, orden_entrada_l1


@app.cell
def _(camino_l1, mo, np, orden_entrada_l1, plt):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    for _v in camino_l1.columns:
        _ax.plot(np.log10(camino_l1.index), camino_l1[_v], lw=1.2,
                 label=_v if camino_l1[_v].abs().max() > 0.15 else None)
    _ax.axhline(0, color="grey", lw=0.8)
    _ax.set_xlabel("log10(C)  (C = 1/λ; a la derecha, menos penalización)")
    _ax.set_ylabel("coeficiente (WoE estandarizado)")
    _ax.set_title("Camino L1 sobre las 16 WoE del pool")
    _ax.legend(fontsize=7, ncol=2)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("**Orden de entrada al camino L1** (C más chico = entra antes):"),
               orden_entrada_l1.to_frame().T.round(4)])
    return


@app.cell
def _(PCA, W_dev, W_ho, familias, gini_np, np, pd, pool, sm, y_dev, y_ho):
    _X = W_dev[pool].values
    _mu, _sd = _X.mean(0), _X.std(0)
    _Xs = (_X - _mu) / _sd
    _Xh = (W_ho[pool].values - _mu) / _sd
    # numpy: autovalores de la matriz de correlación
    _lam, _V = np.linalg.eigh(np.corrcoef(_Xs, rowvar=False))
    _o = np.argsort(_lam)[::-1]
    var_exp_np = _lam[_o] / _lam.sum()
    _pca = PCA().fit(_Xs)
    var_exp_sk = _pca.explained_variance_ratio_
    k_pca = int(np.searchsorted(np.cumsum(var_exp_np), 0.90) + 1)
    _T = _pca.transform(_Xs)[:, :k_pca]
    _Th = _pca.transform(_Xh)[:, :k_pca]
    _m = sm.Logit(y_dev, sm.add_constant(_T)).fit(disp=0)
    gini_pca_ho = gini_np(y_ho, _m.predict(sm.add_constant(_Th)))
    _fam_de = {v: f for f, ms in familias.items() for v in ms}
    cargas_pca = pd.DataFrame(_pca.components_[:3].T, index=pool,
                              columns=["PC1", "PC2", "PC3"]).round(2)
    cargas_pca.insert(0, "familia", [_fam_de.get(v, "—") for v in pool])
    return cargas_pca, gini_pca_ho, k_pca, var_exp_np, var_exp_sk


@app.cell
def _(cargas_pca, fmt, gini_pca_ho, k_pca, mo, res_07):
    mo.vstack([
        mo.md(f"""**PCA**: {k_pca} componentes explican 90% de la varianza de las 16 WoE; la logística
    sobre ellas da Gini HO {fmt(gini_pca_ho)} (vs {fmt(res_07['gini_ho'])} del greedy a 0,70 con
    {res_07['n']} variables). Cargas de las 3 primeras:"""),
        cargas_pca,
        mo.md("""PC1 carga en casi todo (uso de línea, tarjeta, mora, consultas): es «riesgo general»,
    no un concepto. Un reason code «su PC1 es bajo» no se le puede decir a un cliente, las cargas
    cambian en cada re-estimación (y su signo es arbitrario), y la PCA no mira el target: puede
    descartar en las últimas componentes justo la dirección que discrimina."""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Checks del módulo

    Coincidencias numpy vs librería e invariantes teóricas. Si alguno falla, el notebook falla.
    """)
    return


@app.cell
def _(
    C_woe,
    D_var,
    W_dev,
    X_bkw,
    Xs_bkw,
    Z_link,
    aglomerativo_np,
    beta_univ,
    bkw_np,
    candidatas,
    chk_corr,
    cortar_np,
    corte_cluster,
    etiquetas_cl,
    eta_canal,
    gini_np,
    greedy_corr,
    gvif_cca,
    gvif_np,
    ind_bkw,
    iv,
    ll_l1_grande,
    ll_mle,
    logit_np,
    milp,
    LinearConstraint,
    Bounds,
    mo,
    mwis_np,
    np,
    particion_canonica,
    phi_bkw,
    pool,
    props_bkw,
    res_07,
    roc_auc_score,
    sch,
    sel_07,
    sel_greedy,
    sm,
    tabla_familia,
    tabla_nivel_delta,
    tabla_umbrales,
    tabla_vif_trampa,
    umbral_greedy,
    var_exp_np,
    var_exp_sk,
    vif_07_np,
    vif_07_sm,
    vif_np,
    vif_pool_np,
    vif_pool_sm,
    vifw_np,
    vifw_sm,
    y_dev,
):
    _ok = []
    # 1. Pearson / Spearman / Cramér: numpy == scipy
    assert np.allclose(chk_corr[:, 0], chk_corr[:, 1])
    assert np.allclose(chk_corr[:, 2], chk_corr[:, 3])
    assert np.allclose(chk_corr[:, 4], chk_corr[:, 5])
    assert np.isclose(eta_canal[0], eta_canal[1])
    _ok.append("Pearson, Spearman, V de Cramér y η: numpy = scipy/statsmodels")
    # 2. logística numpy == statsmodels; Gini numpy == sklearn
    _X = np.column_stack([np.ones(len(y_dev)), W_dev[sel_07].values])
    _b, _cov, _ll = logit_np(_X, y_dev)
    _m = sm.Logit(y_dev, _X).fit(disp=0)
    assert np.allclose(_b, _m.params, atol=1e-6) and np.isclose(_ll, _m.llf)
    assert np.allclose(_cov, _m.cov_params(), rtol=1e-5)
    _s = _m.predict(_X)
    assert np.isclose(gini_np(y_dev, _s), 2 * roc_auc_score(y_dev, _s) - 1)
    _ok.append("Logística Newton-numpy = statsmodels (β, cov, loglik); Gini numpy = sklearn")
    # 3. VIF: diag(R^-1) == statsmodels con constante; VIF >= 1
    assert np.allclose(vif_07_np, vif_07_sm) and np.allclose(vif_pool_np, vif_pool_sm)
    assert (vif_pool_np >= 1 - 1e-12).all()
    # identidad VIF = 1/(1-R²) con una regresión auxiliar explícita
    _j = 0
    _Xa = W_dev[pool].values
    _r2 = sm.OLS(_Xa[:, _j], sm.add_constant(np.delete(_Xa, _j, axis=1))).fit().rsquared
    assert np.isclose(vif_pool_np[_j], 1 / (1 - _r2))
    _ok.append("VIF: [R⁻¹]_jj = 1/(1−R²_j) = statsmodels (con constante)")
    # 4. BKW: proporciones suman 1 por variable y reconstruyen diag((Xs'Xs)^-1)
    assert np.allclose(props_bkw.sum(0), 1)
    assert np.allclose(phi_bkw.sum(1), np.diag(np.linalg.inv(Xs_bkw.T @ Xs_bkw)))
    assert np.isclose(ind_bkw.max(), np.linalg.cond(Xs_bkw))
    _ok.append("BKW: Σ_k π_kj = 1, Σ_k φ_jk = [(X'X)⁻¹]_jj, máx índice = cond(X) de numpy")
    # 5. GVIF: determinantes == canónicas; GVIF de 1 columna == VIF
    _R = np.corrcoef(W_dev[pool].values, rowvar=False)
    for _idx in ([0, 1, 2], [4, 5], [3]):
        assert np.isclose(gvif_np(_R, _idx), gvif_cca(W_dev[pool].values, _idx)[0], rtol=1e-6)
    assert np.isclose(gvif_np(_R, [3]), vif_pool_np[3])
    _ok.append("GVIF: det(R11)det(R22)/det(R) = Π 1/(1−ρ²_canónicas); GVIF(1 col) = VIF")
    # 6. VIF ponderado numpy == desde cov de statsmodels
    assert np.allclose(vifw_np, vifw_sm, rtol=1e-5)
    _ok.append("VIF ponderado por W de IRLS = diag((X'WX)⁻¹)·SS_w de statsmodels")
    # 7. MWIS: ramificación numpy == scipy milp (objetivo), MWIS >= greedy, greedy respeta umbral
    for _u in [0.6, 0.7, 0.8, 0.9]:
        _sel, _val = mwis_np(pool, C_woe, _u, iv)
        _A = []
        for _i in range(len(pool)):
            for _k in range(_i + 1, len(pool)):
                if abs(C_woe.loc[pool[_i], pool[_k]]) > _u:
                    _r = np.zeros(len(pool))
                    _r[[_i, _k]] = 1
                    _A.append(_r)
        _res = milp(-iv[pool].values, constraints=LinearConstraint(np.array(_A), -np.inf, 1),
                    integrality=np.ones(len(pool)), bounds=Bounds(0, 1))
        assert np.isclose(_val, -_res.fun, atol=1e-6)
        _g = greedy_corr(pool, C_woe, _u)[0]
        assert _val >= iv[_g].sum() - 1e-12
        assert all(abs(C_woe.loc[a, b]) <= _u for i, a in enumerate(_g) for b in _g[i + 1:])
    assert all(abs(C_woe.loc[a, b]) <= umbral_greedy.value + 1e-12
               for i, a in enumerate(sel_greedy) for b in sel_greedy[i + 1:])
    _ok.append("MWIS: ramificación numpy = scipy.milp; Σ IV(MWIS) ≥ Σ IV(greedy); greedy respeta el umbral")
    # 8. Clustering: UPGMA numpy == scipy (alturas y partición al corte)
    _fus = aglomerativo_np(D_var)
    assert np.allclose(sorted(f[2] for f in _fus), np.sort(Z_link[:, 2]))
    for _h in [0.1, 0.3, 0.5, corte_cluster.value]:
        _p_np = particion_canonica(cortar_np(_fus, len(candidatas), _h))
        _p_sc = particion_canonica(sch.fcluster(Z_link, t=_h, criterion="distance"))
        assert (_p_np == _p_sc).all()
    _ok.append("Clustering UPGMA numpy = scipy.cluster.hierarchy (alturas y particiones)")
    # 9. Correlación parcial: precisión == residuos OLS
    assert np.allclose(tabla_familia["corr_parcial_numpy"], tabla_familia["corr_parcial_ols"])
    _ok.append("Correlación parcial: −P_ij/√(P_ii P_jj) = corr de residuos OLS")
    # 10. PCA numpy == sklearn; L1 con C enorme ≈ MLE
    assert np.allclose(var_exp_np, var_exp_sk)
    assert abs(ll_l1_grande - ll_mle) / abs(ll_mle) < 1e-4
    _ok.append("PCA: autovalores de R = sklearn; L1 con C→∞ reproduce la loglik del MLE")
    # 11. Invariantes del experimento
    assert np.all(np.abs(beta_univ + 1) < 0.1), "β univariado sobre WoE debe ser ≈ −1"
    assert np.max(np.abs(np.corrcoef(X_bkw, rowvar=False) - np.eye(X_bkw.shape[1]))) < 0.70
    assert (vif_np(X_bkw)[:7] > 5).all()
    # κ(R) = λmax/λmin ≥ VIF_max  ⇒  η_max ≥ √VIF_max (versión centrada)
    assert np.linalg.cond(np.corrcoef(X_bkw, rowvar=False)) >= vif_np(X_bkw).max() - 1e-9
    assert np.linalg.cond(np.corrcoef(W_dev[pool].values, rowvar=False)) >= vif_pool_np.max() - 1e-9
    assert np.allclose(tabla_vif_trampa.iloc[:, 0], tabla_vif_trampa.iloc[:, 1], atol=0.005)
    assert (tabla_vif_trampa.iloc[:, 2] > tabla_vif_trampa.iloc[:, 0]).all()
    _tn = tabla_nivel_delta
    assert _tn.loc["ambas (3m + 12m)", "β[uso_tc_prom_12m]"] > 0
    assert _tn.loc["nivel + delta (12m + Δ)", "β[uso_tc_prom_12m]"] < 0
    assert _tn.loc["nivel + delta (12m + Δ)", "β[delta_uso_tc]"] < 0
    assert _tn.loc["nivel + delta (12m + Δ)", "vif_max"] < _tn.loc["ambas (3m + 12m)", "vif_max"]
    assert tabla_umbrales.query("umbral == 1.0 and `método` == 'greedy IV'")["coef_pos"].iloc[0] > 0
    assert res_07["n_pos"] == 0
    _ok.append("Invariantes: β univariado ≈ −1; BKW pasa el filtro de a pares con VIF > 5; "
               "'ambas' da signo positivo y nivel+delta no; sin filtro hay signos positivos")
    mo.md("**Todos los checks pasan:**\n\n" + "\n".join(f"- ✓ {t}" for t in _ok))
    return


if __name__ == "__main__":
    app.run()

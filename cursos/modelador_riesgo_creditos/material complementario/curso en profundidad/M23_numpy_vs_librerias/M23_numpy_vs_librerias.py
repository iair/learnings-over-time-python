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
    import contextlib
    import importlib.metadata as importlib_md
    import io
    import math
    import subprocess
    import sys
    import time
    import timeit
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy import linalg, special, stats
    from scipy.optimize import brentq
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.diagnostic_gen import test_chisquare_binning
    return (
        LogisticRegression,
        brentq,
        contextlib,
        importlib_md,
        io,
        linalg,
        math,
        mo,
        plt,
        roc_auc_score,
        sm,
        special,
        stats,
        subprocess,
        sys,
        test_chisquare_binning,
        time,
        timeit,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # M23 · Numpy optimizado vs scipy, statsmodels, scikit-learn, optbinning y nikodym

    **Serie 2 · Del embudo al gobierno.** Acompaña a `M23_numpy_vs_librerias.md`.

    La pregunta no es «¿numpy o librerías?», sino **qué capa del pipeline** se escribe en numpy
    puro, cuál se delega y cómo se demuestra que ambas dicen lo mismo. Este notebook mide las
    cinco dimensiones que se pueden medir en un notebook:

    1. **Estabilidad numérica**: sigmoide y log-verosimilitud (`logaddexp`, `log1p`,
       `logsumexp`), logit en las colas, sumas largas, condicionamiento de $X^\top WX$
       (inversa vs `solve` vs Cholesky vs QR) y float32 vs float64.
    2. **Rendimiento** con `timeit`: binning + WoE (`searchsorted` + `bincount` vs `pd.cut` +
       `groupby` vs `tabla_woe` del curso vs optbinning), AUC (rangos vs conteos vs sklearn vs
       scipy), IRLS vs statsmodels vs sklearn, bootstrap en bucle vs vectorizado vs
       `scipy.stats.bootstrap`, `einsum` y escalamiento con $n$.
    3. **Convenciones** verificadas con `assert`: `C=1` de sklearn, signo del WoE y cortes
       `[a, b)` de optbinning, `metric_special=0`, `ddof`, `binomtest` bilateral, Yates,
       BCa por defecto, grados de libertad del Hosmer-Lemeshow, `converged` de GLM con
       separación.
    4. **Dependencias**: clausura transitiva y costo de importación de cada librería.
    5. **Núcleo numpy + oráculo**: arnés de paridad sobre carteras aleatorias y exportación
       del artefacto a SQL.

    Convenciones del curso: target **1 = malo**; WoE = ln(%buenos/%malos) (WoE alto = bin
    bueno → β negativos); PDO 20, score 600 a odds 50:1 (factor 28,8539; offset 487,1229).

    > Los tiempos dependen de la máquina, de la versión de BLAS y de la carga del momento.
    > Lo que se sostiene entre máquinas son los **órdenes de magnitud** y las razones, no los
    > milisegundos.
    """)
    return


@app.cell
def _():
    # ===========================================================================
    # Código común de la serie (pegado VERBATIM desde _spec/comun.py)
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
    return a_woe, binear, generar_cartera, np, pd, tabla_woe


@app.cell
def _(contextlib, io, time):
    # optbinning 1.0.0 imprime un aviso de HIGHS al importar (inofensivo): se silencia.
    # Si no está instalado, el notebook sigue y omite las comparaciones con optbinning.
    _t0 = time.perf_counter()
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            import optbinning as _ob
            from optbinning import OptimalBinning
        OPTB_OK = True
        OPTB_VERSION = _ob.__version__
    except Exception as _e:  # noqa: BLE001
        OptimalBinning = None
        OPTB_OK = False
        OPTB_VERSION = f"no disponible ({type(_e).__name__})"
    SEG_IMPORT_OPTB = time.perf_counter() - _t0
    return OPTB_OK, OPTB_VERSION, OptimalBinning, SEG_IMPORT_OPTB


@app.cell
def _(np, time, timeit):
    FACTOR = 20 / np.log(2)           # 28.8539 puntos por unidad de log-odds
    OFFSET = 600 - FACTOR * np.log(50)  # 487.1229


    def coma(x, d=2):
        """Número con coma decimal para la prosa."""
        return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


    def miles(n):
        """Entero con punto de miles (convención chilena)."""
        return f"{int(round(n)):,}".replace(",", ".")


    def medir(fn, objetivo=0.04, repeticiones=3):
        """Segundos por llamada: calibra `number` para que cada repetición dure ≈ `objetivo`
        y devuelve el MÍNIMO de las repeticiones (el menos contaminado por ruido del sistema)."""
        _t0 = time.perf_counter()
        fn()
        _t1 = max(time.perf_counter() - _t0, 1e-7)
        _num = max(1, int(objetivo / _t1))
        return min(timeit.repeat(fn, number=_num, repeat=repeticiones)) / _num
    return FACTOR, OFFSET, coma, medir, miles


@app.cell
def _(a_woe, generar_cartera, np, tabla_woe):
    cartera = generar_cartera()
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    ho = cartera[cartera["muestra"] == "HO"].reset_index(drop=True)
    oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)

    VARIABLES = ["uso_linea_prom_12m", "uso_tc_prom_12m", "meses_desde_mora_12m",
                 "antiguedad_meses", "carga_financiera", "consultas_6m", "canal"]
    # WoE de DEV (binning del curso), aplicado a DEV y OOT
    mapas_woe = {v: tabla_woe(dev[v], dev["malo"])[0]["woe"].to_dict() for v in VARIABLES}
    X_dev = np.column_stack([np.ones(len(dev)), a_woe(dev, VARIABLES, dev, mapas_woe).to_numpy()])
    X_oot = np.column_stack([np.ones(len(oot)), a_woe(oot, VARIABLES, dev, mapas_woe).to_numpy()])
    y_dev = dev["malo"].to_numpy()
    y_oot = oot["malo"].to_numpy()
    return VARIABLES, X_dev, X_oot, cartera, dev, ho, mapas_woe, oot, y_dev, y_oot


@app.cell
def _(X_dev, dev, ho, miles, mo, oot, y_dev, y_oot):
    mo.md(f"""
    **Cartera sintética** (`generar_cartera()`, semilla 2026): DEV {miles(len(dev))} filas ·
    {miles(y_dev.sum())} malos · HO {miles(len(ho))} · OOT {miles(len(oot))} filas · {miles(y_oot.sum())}
    malos. Modelo de trabajo: logística sobre el WoE de DEV de {X_dev.shape[1] - 1} variables
    (binning del curso), la misma estructura de las clases 3–4.
    """)
    return


@app.cell
def _(np):
    # ======================================================================
    # NÚCLEO NUMPY: las funciones que el resto del notebook compara con librerías
    # ======================================================================
    def sigmoide_ingenua(z):
        """σ(z) = 1/(1+e^{-z}) tal cual: desborda e^{-z} para z ≪ 0 (con aviso)."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoide_estable(z):
        """σ(z) sin overflow: usa e^{-|z|} ≤ 1 en ambas ramas."""
        z = np.asarray(z)
        e = np.exp(-np.abs(z))
        return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

    def log_verosimilitud_ingenua(z, y):
        """ℓ = Σ y·ln p + (1−y)·ln(1−p) calculando p primero: pierde todo cuando p redondea a 0 o 1."""
        p = sigmoide_ingenua(z)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    def log_verosimilitud_estable(z, y):
        """ln p = −ln(1+e^{−z}) = −logaddexp(0, −z);  ln(1−p) = −logaddexp(0, z).
        Equivalente: ℓ_i = y·z − logaddexp(0, z)."""
        return y * z - np.logaddexp(0.0, z)

    def irls_logit(X, y, offset=None, metodo="solve", dtype=np.float64, tol=1e-10, max_iter=50):
        """Máxima verosimilitud logística por Newton-Raphson (= IRLS).
        metodo: 'solve' (LU sobre XᵀWX), 'inv' (inversa explícita), 'cholesky' o 'qr'
        (mínimos cuadrados sobre W^{1/2}X, sin formar XᵀWX)."""
        X = np.asarray(X, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
        o = np.zeros(len(y), dtype=dtype) if offset is None else np.asarray(offset, dtype=dtype)
        b = np.zeros(X.shape[1], dtype=dtype)
        for it in range(1, max_iter + 1):
            eta = X @ b + o
            p = sigmoide_estable(eta).astype(dtype)
            w = p * (1 - p)
            g = X.T @ (y - p)                          # gradiente (score)
            if metodo == "qr":
                sw = np.sqrt(w)
                d = np.linalg.lstsq(X * sw[:, None], (y - p) / sw, rcond=None)[0]
            else:
                H = X.T @ (X * w[:, None])             # información de Fisher XᵀWX
                if metodo == "inv":
                    d = np.linalg.inv(H) @ g
                elif metodo == "cholesky":
                    L = np.linalg.cholesky(H)
                    d = np.linalg.solve(L.T, np.linalg.solve(L, g))
                else:
                    d = np.linalg.solve(H, g)
            b = b + d
            if np.max(np.abs(d)) < tol:
                break
        eta = X @ b + o
        p = sigmoide_estable(eta)
        H = X.T @ (X * (p * (1 - p))[:, None])
        return {"beta": b, "iter": it, "H": H,
                "se": np.sqrt(np.diag(np.linalg.inv(H.astype(np.float64)))),
                "loglik": float(np.sum(log_verosimilitud_estable(eta.astype(np.float64), y)))}

    def woe_numpy(x, y, cortes):
        """Tabla WoE con cortes dados. Intervalos (c_{k-1}, c_k] como pd.cut:
        searchsorted(side='left') sobre los cortes INTERIORES. Suavizado +0,5 del curso."""
        k = len(cortes) - 1
        idx = np.searchsorted(cortes[1:-1], x, side="left")
        n = np.bincount(idx, minlength=k).astype(float)
        m = np.bincount(idx, weights=y, minlength=k)
        b = n - m
        pm = (m + 0.5) / (m.sum() + 0.5 * k)
        pb = (b + 0.5) / (b.sum() + 0.5 * k)
        woe = np.log(pb / pm)
        return woe, float(np.sum((pb - pm) * woe)), n

    def auc_rangos(y, s):
        """AUC = (Σ rangos de malos − n_m(n_m+1)/2)/(n_m·n_b), rangos medios en empates.
        Convención: s = PD (mayor = más riesgoso)."""
        u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        rango_medio = np.cumsum(cnt) - cnt + (cnt + 1) / 2.0
        r = rango_medio[inv]
        nm = y.sum()
        nb = len(y) - nm
        return float((r[y == 1].sum() - nm * (nm + 1) / 2) / (nm * nb))

    def auc_conteos(codigos, y, k):
        """AUC con conteos por valor único (códigos 0..k−1 en orden ascendente de s):
        AUC = Σ_j m_j (B_{<j} + ½ b_j) / (n_m n_b). O(n + k), sin ordenar."""
        m = np.bincount(codigos, weights=y, minlength=k)
        b = np.bincount(codigos, minlength=k) - m
        b_menor = np.cumsum(b) - b
        return float(np.sum(m * (b_menor + 0.5 * b)) / (m.sum() * b.sum()))

    def hosmer_lemeshow_numpy(y, p, g=10):
        """HL con g grupos de igual tamaño por PD (np.array_split del argsort, como statsmodels)."""
        orden = np.argsort(p, kind="quicksort")
        grupos = np.array_split(orden, g)
        o = np.array([y[i].sum() for i in grupos])
        e = np.array([p[i].sum() for i in grupos])
        n = np.array([len(i) for i in grupos])
        return float(np.sum((o - e) ** 2 / (e * (1 - e / n))))

    def delta_newton(lp, objetivo, tol=1e-12):
        """δ tal que mean(σ(lp + δ)) = objetivo, por Newton 1-D (la función es monótona y convexa en la cola)."""
        d = 0.0
        for _ in range(100):
            p = sigmoide_estable(lp + d)
            paso = (p.mean() - objetivo) / np.mean(p * (1 - p))
            d -= paso
            if abs(paso) < tol:
                break
        return d
    return (
        auc_conteos,
        auc_rangos,
        delta_newton,
        hosmer_lemeshow_numpy,
        irls_logit,
        log_verosimilitud_estable,
        log_verosimilitud_ingenua,
        sigmoide_estable,
        sigmoide_ingenua,
        woe_numpy,
    )


@app.cell
def _(X_dev, X_oot, irls_logit, sigmoide_estable, y_dev):
    modelo_dev = irls_logit(X_dev, y_dev)
    beta_mle = modelo_dev["beta"]
    lp_dev = X_dev @ beta_mle
    lp_oot = X_oot @ beta_mle
    pd_dev = sigmoide_estable(lp_dev)
    pd_oot = sigmoide_estable(lp_oot)
    return beta_mle, lp_dev, lp_oot, modelo_dev, pd_dev, pd_oot


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## A. Estabilidad numérica

    ### A1 · Sigmoide y log-verosimilitud: la resta que no se ve

    La log-verosimilitud logística se puede escribir de dos maneras algebraicamente idénticas:

    $$\ell_i = y_i\ln p_i + (1-y_i)\ln(1-p_i) \quad\text{con } p_i=\sigma(z_i)
    \qquad\Longleftrightarrow\qquad \ell_i = y_i z_i - \ln(1+e^{z_i}).$$

    La primera calcula $p$ y **después** toma logaritmos: si $p$ redondea a 1 (en float64 ocurre
    para $z \gtrsim 37$; en float32 para $z \gtrsim 17$), $\ln(1-p) = \ln 0 = -\infty$. La segunda
    nunca forma $p$: `np.logaddexp(0, z)` calcula $\ln(1+e^{z})$ sin overflow. **Qué mirar:** las
    filas marcadas ✗, y cómo con float32 el problema aparece con log-odds que un scorecard sí
    produce (una PD de $10^{-8}$ es $z=-18{,}4$).
    """)
    return


@app.cell
def _(mo):
    tipo_flotante = mo.ui.dropdown(options=["float64", "float32"], value="float64",
                                   label="Precisión de punto flotante")
    tipo_flotante
    return (tipo_flotante,)


@app.cell
def _(log_verosimilitud_estable, log_verosimilitud_ingenua, np, pd, special, tipo_flotante):
    _dt = np.dtype(tipo_flotante.value)
    _z = np.array([-800, -100, -40, -20, -17, -10, 0, 10, 17, 20, 40, 100, 800], dtype=_dt)
    with np.errstate(all="ignore"):
        _p = (1 / (1 + np.exp(-_z))).astype(_dt)
        _l0_ing = log_verosimilitud_ingenua(_z, np.zeros_like(_z))   # buen pagador: ln(1−p)
        _l1_ing = log_verosimilitud_ingenua(_z, np.ones_like(_z))    # malo: ln p
    _l0_est = log_verosimilitud_estable(_z, np.zeros_like(_z))
    _l1_est = log_verosimilitud_estable(_z, np.ones_like(_z))
    _l0_sc = special.log_expit(-_z)
    _l1_sc = special.log_expit(_z)
    # referencia exacta en float64 (para medir el error relativo de la versión ingenua)
    _z64 = _z.astype(np.float64)
    _ref0 = -np.logaddexp(0, _z64)
    _ref1 = -np.logaddexp(0, -_z64)

    def _falla(a, ref):
        with np.errstate(all="ignore"):
            rel = np.abs(a.astype(np.float64) - ref) / np.maximum(np.abs(ref), 1e-300)
        return np.where(~np.isfinite(a) | (rel > 1e-3), "✗", "")

    tabla_sigmoide = pd.DataFrame({
        "z (log-odds)": _z,
        "p = σ(z) ingenua": _p,
        "ln(1−p) ingenuo": _l0_ing, "": _falla(_l0_ing, _ref0),
        "ln(1−p) estable": _l0_est,
        "log_expit(−z) scipy": _l0_sc,
        "ln p ingenuo": _l1_ing, " ": _falla(_l1_ing, _ref1),
        "ln p estable": _l1_est,
    })
    fallas_sigmoide = int((tabla_sigmoide[""] == "✗").sum() + (tabla_sigmoide[" "] == "✗").sum())
    assert np.allclose(_l0_est, _l0_sc, rtol=1e-5 if _dt == np.float32 else 1e-12)
    tabla_sigmoide
    return fallas_sigmoide, tabla_sigmoide


@app.cell
def _(fallas_sigmoide, mo, tipo_flotante):
    mo.md(f"""
    **Lectura ({tipo_flotante.value}).** La versión ingenua falla en **{fallas_sigmoide}** de 26
    celdas (−∞, `NaN`, 0 donde debía haber un número pequeño, o error relativo > 0,1%). Los `NaN` son
    la trampa menos obvia: con $y=0$ y $p$ redondeado a 0, el término $y\ln p = 0\cdot(-\infty)$ es
    `NaN` y contamina la suma **aunque ese término no debería contar**. La estable y
    `scipy.special.log_expit` coinciden en todas. En float64 las fallas están en |z| ≥ 40, fuera de
    lo que produce un scorecard sano… pero **no** fuera de lo que produce un árbol potenciado, un
    modelo con separación casi perfecta o un bin con cero malos sin suavizado (WoE = ±∞). En
    float32 las fallas empiezan en |z| = 17. Regla: *nunca* calcular `log(p)` o `log(1-p)` después
    de `p`; usar la forma $yz - \\text{{logaddexp}}(0, z)$ o `log_expit`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A2 · Logit y complementos en las colas: $\ln(1-p)$, $1-P(\text{bueno})$, $\text{logit}(1-p)$

    Tres operaciones con **cancelación catastrófica** cuando $p\to 0$: (i) $\ln(1-p)$ calculado
    como `log(1 - p)` pierde los dígitos de $p$ al formar $1-p$; `np.log1p(-p)` no los pierde.
    (ii) Una PD obtenida como $1 - P(\text{bueno})$ hereda el error absoluto de
    $P(\text{bueno})\approx 1$, que es $\approx 10^{-16}$: para PD $< 10^{-12}$ el error relativo es
    enorme. (iii) $\text{logit}(1-p)$ debería ser $-\text{logit}(p)$ exactamente; calculado sobre
    $1-p$ ya redondeado no lo es. **Qué mirar:** la columna de error relativo.
    """)
    return


@app.cell
def _(np, pd, special):
    _p = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-17])
    with np.errstate(all="ignore"):
        _log1m_ing = np.log(1 - _p)
        _log1m_est = np.log1p(-_p)
        _pd_por_complemento = 1 - special.expit(-special.logit(_p))   # 1 − P(bueno)
        _pd_directa = special.expit(special.logit(_p))
        _logit_c_ing = special.logit(1 - _p)
        _logit_c_est = -special.logit(_p)
        tabla_colas = pd.DataFrame({
            "p (PD)": _p,
            "log(1−p)": _log1m_ing,
            "log1p(−p)": _log1m_est,
            "err. rel. log(1−p)": np.abs(_log1m_ing - _log1m_est) / np.abs(_log1m_est),
            "PD = 1 − P(bueno)": _pd_por_complemento,
            "err. rel. complemento": np.abs(_pd_por_complemento - _p) / _p,
            "PD = σ(logit p)": _pd_directa,
            "logit(1−p)": _logit_c_ing,
            "−logit(p)": _logit_c_est,
        })
    err_log1m_1e10 = float(tabla_colas.loc[tabla_colas["p (PD)"] == 1e-10, "err. rel. log(1−p)"].iloc[0])
    tabla_colas
    return err_log1m_1e10, tabla_colas


@app.cell
def _(err_log1m_1e10, mo, tabla_colas):
    _e = tabla_colas.set_index("p (PD)")["err. rel. complemento"]
    mo.md(f"""
    **Lectura.** Con PD = $10^{{-10}}$, `log(1-p)` ya tiene error relativo {err_log1m_1e10:.1e}
    (y en $10^{{-17}}$ devuelve 0: el pagador «no aporta» a la verosimilitud). La PD reconstruida como
    $1-P(\\text{{bueno}})$ tiene error relativo {_e.loc[1e-12]:.1e} en $10^{{-12}}$, {_e.loc[1e-14]:.1e} en
    $10^{{-14}}$ y es **cero** desde $10^{{-16}}$ (el error absoluto es siempre ~$10^{{-16}}$, el ε de
    máquina alrededor de 1). En un scorecard de consumo las PD viven en $[10^{{-4}}, 0{{,}}5]$ y el problema no
    muerde; muerde en (a) modelos de *low default portfolios*, (b) probabilidades de supervivencia
    acumuladas a varios años, (c) productos de muchas probabilidades (A3). Regla de
    implementación: guardar y propagar **log-odds**, no probabilidades; convertir a PD al final.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A3 · Sumas y productos largos: verosimilitud de cartera y pesos de escenarios

    Tres casos de riesgo:

    1. **Probabilidad de cero defaults** en $n$ créditos independientes: $\prod_i(1-p_i)$ hace
       *underflow* a 0 con pocos miles de créditos; $\exp\big(\sum_i \texttt{log1p}(-p_i)\big)$ se
       queda en escala logarítmica.
    2. **Pesos posteriores de escenarios macro.** Con escenarios $s$ que desplazan el log-odds
       ($\eta+\Delta_s$), el peso es $w_s\propto\pi_s\,e^{\ell_s}$ con $\ell_s\approx -1.500$: $e^{\ell_s}$
       es 0 en float64 y la normalización da `0/0 = nan`. Con `logsumexp`:
       $w_s = \exp\big(\ln\pi_s + \ell_s - \text{LSE}_s(\ln\pi_s+\ell_s)\big)$.
    3. **Acumular un millón de log-verosimilitudes en float32**: `np.sum` usa suma por pares
       (*pairwise*, error $O(\varepsilon\log n)$); una suma secuencial (`cumsum`, un bucle de
       Python, un `UPDATE ... SET acc = acc + x` o un motor que acumula fila a fila) tiene error
       $O(\varepsilon n)$.
    """)
    return


@app.cell
def _(lp_oot, math, np, pd, special, y_oot):
    # (1) cero defaults en una cartera de 30.000 créditos con PD entre 2% y 10%
    _rng = np.random.default_rng(23)
    _pd_cartera = _rng.uniform(0.02, 0.10, 30_000)
    with np.errstate(all="ignore"):
        prob_cero_ingenua = float(np.prod(1 - _pd_cartera))
    log_prob_cero = float(np.sum(np.log1p(-_pd_cartera)))

    # (2) ¿qué escenario macro explica la cohorte OOT? Desplazamientos del log-odds
    _desplaz = np.array([0.0, 0.35, 0.70])
    _prior = np.array([0.6, 0.3, 0.1])
    _loglik = np.array([np.sum(y_oot * (lp_oot + d) - np.logaddexp(0, lp_oot + d)) for d in _desplaz])
    with np.errstate(all="ignore"):
        _w_ing = _prior * np.exp(_loglik)
        pesos_ingenuos = _w_ing / _w_ing.sum()
    _lw = np.log(_prior) + _loglik
    pesos_lse = np.exp(_lw - special.logsumexp(_lw))
    tabla_escenarios = pd.DataFrame({
        "escenario": ["base (Δ=0)", "adverso (Δ=+0,35)", "severo (Δ=+0,70)"],
        "prior": _prior, "log-verosimilitud OOT": _loglik.round(2),
        "peso ingenuo": pesos_ingenuos, "peso con logsumexp": pesos_lse.round(6)})

    # (3) un millón de términos de log-verosimilitud en float32
    _terminos = np.tile(y_oot * lp_oot - np.logaddexp(0, lp_oot), 1_000_000 // len(y_oot) + 1)[:1_000_000]
    _t32 = _terminos.astype(np.float32)
    suma_exacta = math.fsum(_t32.astype(np.float64))
    tabla_sumas = pd.DataFrame({
        "método": ["math.fsum (exacta)", "np.sum float32 (pares)", "cumsum float32 (secuencial)",
                   "np.sum float64"],
        "suma": [suma_exacta, float(np.sum(_t32)), float(np.cumsum(_t32)[-1]),
                 float(np.sum(_t32, dtype=np.float64))]})
    tabla_sumas["error relativo"] = np.abs(tabla_sumas["suma"] - suma_exacta) / abs(suma_exacta)
    err_cumsum32 = float(tabla_sumas["error relativo"].iloc[2])
    err_pairwise32 = float(tabla_sumas["error relativo"].iloc[1])
    return (
        err_cumsum32,
        err_pairwise32,
        log_prob_cero,
        pesos_ingenuos,
        pesos_lse,
        prob_cero_ingenua,
        tabla_escenarios,
        tabla_sumas,
    )


@app.cell
def _(
    coma,
    err_cumsum32,
    err_pairwise32,
    log_prob_cero,
    mo,
    pesos_lse,
    prob_cero_ingenua,
    tabla_escenarios,
    tabla_sumas,
):
    mo.vstack([
        mo.md(f"""
    **(1) Cero defaults en 30.000 créditos.** `np.prod(1 − p)` = {prob_cero_ingenua} (underflow);
    en logaritmos: ln P = {coma(log_prob_cero, 1)}, es decir P = 10^{coma(log_prob_cero / 2.302585, 1)}.
    El número es inútil como probabilidad, pero su logaritmo es exactamente lo que necesita una
    razón de verosimilitudes o un test.

    **(2) Escenarios macro sobre OOT** (la cohorte OOT del generador tiene un deterioro plantado de
    +0,35 en log-odds):
    """),
        tabla_escenarios,
        mo.md(f"""
    La versión ingenua da `nan` en los tres pesos; `logsumexp` asigna {coma(pesos_lse[1] * 100, 1)}%
    al escenario adverso, el que coincide con el deterioro plantado.

    **(3) Un millón de términos en float32:**
    """),
        tabla_sumas,
        mo.md(f"""
    La suma secuencial en float32 se equivoca en {coma(err_cumsum32 * 100, 3)}% (el acumulador crece y
    cada término pequeño pierde dígitos); `np.sum` en float32 se equivoca en {err_pairwise32:.1e}
    gracias a la suma por pares. **Consecuencia práctica**: la paridad numpy ↔ motor de producción
    (SQL, Java, Spark) se rompe en sumas largas aunque ambos «usen float»: el orden de acumulación
    importa. Acumular en float64 lo resuelve en ambos lados.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A4 · Condicionamiento de $X^\top WX$: inversa vs `solve` vs Cholesky vs QR

    Cada paso de IRLS resuelve $(X^\top WX)\,d = X^\top(y-p)$. Si $\kappa(\tilde X)$ es el número
    de condición de $\tilde X = W^{1/2}X$, entonces $\kappa(X^\top WX)=\kappa(\tilde X)^2$: formar
    las **ecuaciones normales eleva al cuadrado** el condicionamiento. La cota de error relativo
    hacia adelante es $\approx \kappa^2\varepsilon$ para inversa, LU (`solve`) y Cholesky, y
    $\approx\kappa\varepsilon$ para QR/SVD sobre $\tilde X$ (con residuo cero; con residuo grande
    QR también paga un término $\kappa^2$). El experimento construye $X$ de $2.000\times 6$ con
    valores singulares $1\ldots10^{-\log_{10}\kappa}$ y $\beta$ conocido. **Qué mirar:** a partir de
    $\kappa\approx10^{8}$ ($=\varepsilon^{-1/2}$) las ecuaciones normales no tienen ni un dígito
    correcto y QR todavía conserva unos diez.
    """)
    return


@app.cell
def _(mo):
    log_kappa = mo.ui.slider(1, 12, value=7, step=1, label="log₁₀ κ(X)")
    log_kappa
    return (log_kappa,)


@app.cell
def _(linalg, np, warnings):
    def errores_solvers(logk, n=2000, p=6, semilla=0):
        """Error relativo ‖β̂−β‖/‖β‖ de cinco formas de resolver mínimos cuadrados."""
        _rng = np.random.default_rng(semilla)
        _U, _ = np.linalg.qr(_rng.normal(size=(n, p)))
        _V, _ = np.linalg.qr(_rng.normal(size=(p, p)))
        _A = _U @ np.diag(np.logspace(0, -logk, p)) @ _V.T
        _beta = np.ones(p)
        _b = _A @ _beta
        _M = _A.T @ _A
        _r = _A.T @ _b
        _res = {}
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            for _nombre, _f in [
                ("inv(XᵀX)·Xᵀy", lambda: np.linalg.inv(_M) @ _r),
                ("solve(XᵀX, Xᵀy)", lambda: np.linalg.solve(_M, _r)),
                ("Cholesky", lambda: linalg.cho_solve(linalg.cho_factor(_M), _r)),
                ("QR de X", lambda: linalg.solve_triangular(*(lambda Q, R: (R, Q.T @ _b))(*np.linalg.qr(_A)))),
                ("lstsq (SVD)", lambda: np.linalg.lstsq(_A, _b, rcond=None)[0]),
            ]:
                try:
                    _x = _f()
                    _res[_nombre] = float(np.linalg.norm(_x - _beta) / np.linalg.norm(_beta))
                except Exception:  # noqa: BLE001  (Cholesky falla si XᵀX deja de ser definida positiva)
                    _res[_nombre] = np.nan
        return _res

    grilla_solvers = {k: errores_solvers(k) for k in range(1, 13)}
    return errores_solvers, grilla_solvers


@app.cell
def _(grilla_solvers, log_kappa, mo, np, pd, plt):
    _ks = np.array(sorted(grilla_solvers))
    _metodos = list(grilla_solvers[1])
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    for _m, _mk in zip(_metodos, ["o", "s", "^", "D", "v"]):
        _e = np.array([grilla_solvers[k][_m] for k in _ks])
        _ax.semilogy(_ks, np.clip(np.nan_to_num(_e, nan=10.0), 1e-17, 10), marker=_mk, label=_m)
    _eps = np.finfo(float).eps
    _ax.semilogy(_ks, np.minimum(10.0 ** _ks * _eps, 10), "k:", lw=1, label="κ·ε")
    _ax.semilogy(_ks, np.minimum(10.0 ** (2 * _ks) * _eps, 10), "k--", lw=1, label="κ²·ε")
    _ax.axvline(log_kappa.value, color="grey", alpha=0.4)
    _ax.set_xlabel("log₁₀ κ(X)")
    _ax.set_ylabel("error relativo de β (escala log)")
    _ax.set_title("Ecuaciones normales pierden el doble de dígitos que QR")
    _ax.legend(fontsize=7, ncol=2)
    _fig.tight_layout()
    tabla_solver_actual = pd.DataFrame({"método": list(grilla_solvers[log_kappa.value]),
                                        "error relativo": list(grilla_solvers[log_kappa.value].values())})
    tabla_solver_actual["dígitos correctos ≈"] = (-np.log10(tabla_solver_actual["error relativo"]
                                                            .clip(1e-17, 1))).round(1)
    mo.hstack([_fig, tabla_solver_actual], widths=[3, 2])
    return (tabla_solver_actual,)


@app.cell
def _(mo):
    mo.md(r"""
    **¿Y en un scorecard real?** El experimento anterior es de laboratorio. La tabla siguiente
    calcula $\kappa$ en diseños que un modelador de crédito sí arma sobre DEV, **en bruto** y
    **con columnas equilibradas** (cada columna de $W^{1/2}X$ escalada a norma 1). La distinción
    importa: Cholesky y LU con pivoteo son (casi) invariantes a escalar columnas (van der Sluis,
    1969), así que un $\kappa$ alto que desaparece al equilibrar es **mal condicionamiento de
    escala**, benigno. **Qué mirar:** la última columna y el experimento de reescalar la renta.
    """)
    return


@app.cell
def _(VARIABLES, X_dev, a_woe, dev, irls_logit, np, pd, pd_dev, tabla_woe, y_dev):
    def _kappa(X, w, equilibrar=False):
        _Xt = X * np.sqrt(w)[:, None]
        if equilibrar:
            _Xt = _Xt / np.linalg.norm(_Xt, axis=0)
        _s = np.linalg.svd(_Xt, compute_uv=False)
        return float(_s[0] / _s[-1]) if _s[-1] > 0 else np.inf

    _w = pd_dev * (1 - pd_dev)
    # 2) crudas con renta en millones vs en pesos
    _num = ["uso_linea_prom_12m", "antiguedad_meses", "carga_financiera", "consultas_6m"]
    _renta = dev["renta_mm"].fillna(dev["renta_mm"].median()).to_numpy()
    _X_mm = np.column_stack([np.ones(len(dev)), dev[_num].to_numpy(), _renta])
    _X_pesos = _X_mm.copy()
    _X_pesos[:, -1] = _X_pesos[:, -1] * 1e6
    # 3) dummies completas de canal + intercepto (trampa de variables ficticias)
    _X_dum = np.column_stack([np.ones(len(dev)), pd.get_dummies(dev["canal"]).to_numpy(dtype=float)])
    # 4) WoE + la misma familia dos veces (uso_tc 12m y 3m)
    _m3 = {"uso_tc_prom_3m": tabla_woe(dev["uso_tc_prom_3m"], dev["malo"])[0]["woe"].to_dict()}
    _w3 = a_woe(dev, ["uso_tc_prom_3m"], dev, _m3).to_numpy()
    _X_fam = np.column_stack([X_dev, _w3])
    corr_familia = float(np.corrcoef(X_dev[:, VARIABLES.index("uso_tc_prom_12m") + 1], _w3[:, 0])[0, 1])
    _disenos = {"WoE del modelo (8 col.)": X_dev, "crudas, renta en MM$": _X_mm, "crudas, renta en $": _X_pesos,
                "dummies completas de canal + intercepto": _X_dum, "WoE + uso_tc 3m (misma familia)": _X_fam}
    tabla_kappa = pd.DataFrame({
        "diseño": list(_disenos),
        "κ(W½X) bruto": [_kappa(X, _w) for X in _disenos.values()],
        "κ(W½X) equilibrado": [_kappa(X, _w, True) for X in _disenos.values()],
    })
    tabla_kappa["dígitos perdidos en XᵀWX (bruto)"] = np.minimum(2 * np.log10(tabla_kappa["κ(W½X) bruto"].clip(1, 1e300)), 16).round(1)
    tabla_kappa["ídem (equilibrado)"] = np.minimum(2 * np.log10(tabla_kappa["κ(W½X) equilibrado"].clip(1, 1e300)), 16).round(1)
    kappa_woe = float(tabla_kappa["κ(W½X) bruto"].iloc[0])
    kappa_pesos = float(tabla_kappa["κ(W½X) bruto"].iloc[2])
    kappa_pesos_eq = float(tabla_kappa["κ(W½X) equilibrado"].iloc[2])
    # experimento: el MISMO modelo en MM$ y en $ con solve (ecuaciones normales): ¿cambian las predicciones?
    _a = irls_logit(_X_mm, y_dev, metodo="solve")
    _b = irls_logit(_X_pesos, y_dev, metodo="solve")
    dif_pred_escala = float(np.max(np.abs(_X_mm @ _a["beta"] - _X_pesos @ _b["beta"])))
    tabla_kappa
    return corr_familia, dif_pred_escala, kappa_pesos, kappa_pesos_eq, kappa_woe, tabla_kappa


@app.cell
def _(coma, corr_familia, dif_pred_escala, kappa_pesos, kappa_pesos_eq, kappa_woe, mo, np):
    mo.md(f"""
    **Lectura.** El diseño WoE tiene κ ≈ {coma(kappa_woe, 1)}: $X^\\top WX$ pierde ~{coma(2 * np.log10(kappa_woe), 1)}
    de los 16 dígitos; cualquier método sirve. Con la renta en pesos el κ bruto sube a {kappa_pesos:.1e},
    pero equilibrado vuelve a {coma(kappa_pesos_eq, 1)}: es condicionamiento de escala. La prueba empírica:
    el mismo modelo ajustado con `solve` sobre las ecuaciones normales en MM$ y en $ da log-odds
    que difieren en {dif_pred_escala:.1e}. Las dummies completas con intercepto son singulares **también**
    equilibradas (κ del orden de $10^{{16}}$ o infinito): ese sí es un problema, y lo resuelve el diseño
    (una categoría de referencia), no el algoritmo. La familia de uso de tarjeta (corr. WoE
    {coma(corr_familia, 3)}) apenas mueve κ: la colinealidad de una familia es un problema
    **estadístico** (varianza de β, M09), no numérico. Moraleja: en un scorecard la estabilidad
    numérica se compra en el diseño; `lstsq` vs `solve` importa en regresiones polinomiales, splines
    sin base B, o cuando se ajustan miles de modelos automáticos sin revisar el diseño.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A5 · float32 vs float64 en el ajuste completo

    GPUs, Spark ML con ciertos *backends*, ONNX y muchos motores de *scoring* trabajan en float32
    para ahorrar memoria. ¿Cuánto cambia el scorecard? Se ajusta el mismo IRLS en ambas precisiones
    y se compara en **puntos de score** (factor 28,85 por unidad de log-odds).
    """)
    return


@app.cell
def _(FACTOR, X_dev, irls_logit, np, pd, sm, y_dev):
    _m64 = irls_logit(X_dev, y_dev, dtype=np.float64)
    _m32 = irls_logit(X_dev, y_dev, dtype=np.float32, tol=1e-5)
    _lp64 = X_dev @ _m64["beta"]
    _lp32 = (X_dev.astype(np.float32) @ _m32["beta"]).astype(np.float64)
    dif_beta_32 = float(np.max(np.abs(_m32["beta"].astype(np.float64) - _m64["beta"])))
    dif_puntos_32 = float(np.max(np.abs(_lp32 - _lp64)) * FACTOR)
    _r = sm.Logit(y_dev.astype(np.float32), X_dev.astype(np.float32)).fit(disp=0)
    dtype_statsmodels_32 = str(_r.params.dtype)
    tabla_f32 = pd.DataFrame({
        "precisión": ["float64", "float32"],
        "iteraciones": [_m64["iter"], _m32["iter"]],
        "β₀": [_m64["beta"][0], float(_m32["beta"][0])],
        "máx |Δβ| vs float64": [0.0, dif_beta_32],
        "máx |Δ score| (puntos)": [0.0, dif_puntos_32],
    })
    tabla_f32
    return dif_beta_32, dif_puntos_32, dtype_statsmodels_32, tabla_f32


@app.cell
def _(coma, dif_beta_32, dif_puntos_32, dtype_statsmodels_32, mo):
    mo.md(f"""
    **Lectura.** float32 cambia los β en hasta {dif_beta_32:.1e} y el score en hasta
    {dif_puntos_32:.1e} puntos: **irrelevante para decidir** (los puntos se redondean a entero),
    **relevante para un test de paridad** con tolerancia $10^{{-8}}$, que fallaría. Por eso la
    tolerancia de un test numpy ↔ motor debe declararse en función de la precisión del motor, no
    copiarse. statsmodels promueve la entrada float32 a `{dtype_statsmodels_32}` en silencio;
    sklearn (lbfgs) también. Si el motor de producción usa float32, la paridad debe probarse
    contra una referencia float32, no contra el notebook.
    """)
    return



@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## B. Rendimiento: benchmarks con `timeit`

    Cada tiempo es el **mínimo** de 3 repeticiones (función `medir`), la estimación menos
    contaminada por otros procesos. Todas las parejas se comparan también en **resultado**: un
    benchmark donde las dos versiones no calculan lo mismo no mide nada.

    ### B1 · Binning + WoE: `searchsorted` + `bincount` vs `pd.cut` + `groupby` vs `tabla_woe` vs optbinning

    Misma variable (`uso_linea_prom_12m`, DEV), mismos cortes (quintiles de DEV), mismo
    suavizado +0,5. optbinning se mide dos veces: con los cortes fijados (`user_splits`, solo
    tabula) y en modo óptimo (pre-binning CART + optimización CP), que resuelve **otro problema**.
    """)
    return


@app.cell
def _(OPTB_OK, OptimalBinning, dev, medir, np, pd, tabla_woe, woe_numpy):
    _x = dev["uso_linea_prom_12m"].to_numpy()
    _y = dev["malo"].to_numpy()
    cortes_uso = np.unique(np.nanquantile(_x, np.linspace(0, 1, 6)))
    cortes_uso[0], cortes_uso[-1] = -np.inf, np.inf

    def woe_pandas(x, y, cortes):
        """Misma tabla con pd.cut (intervalos (a, b]) + groupby."""
        _c = pd.cut(pd.Series(x), cortes)
        _t = pd.DataFrame({"c": _c, "y": y}).groupby("c", observed=False)["y"].agg(["count", "sum"])
        _m = _t["sum"].to_numpy()
        _b = _t["count"].to_numpy() - _m
        _k = len(_m)
        _pm = (_m + 0.5) / (_m.sum() + 0.5 * _k)
        _pb = (_b + 0.5) / (_b.sum() + 0.5 * _k)
        _w = np.log(_pb / _pm)
        return _w, float(np.sum((_pb - _pm) * _w))

    woe_np_uso, iv_np_uso, n_np_uso = woe_numpy(_x, _y, cortes_uso)
    woe_pd_uso, iv_pd_uso = woe_pandas(_x, _y, cortes_uso)
    _tab_curso, iv_curso_uso = tabla_woe(_x, _y)
    woe_curso_uso = _tab_curso["woe"].to_numpy()

    _filas = [
        ("numpy: searchsorted + bincount", medir(lambda: woe_numpy(_x, _y, cortes_uso)), iv_np_uso),
        ("pandas: pd.cut + groupby", medir(lambda: woe_pandas(_x, _y, cortes_uso)), iv_pd_uso),
        ("curso: tabla_woe (cuantiles + etiquetas str)", medir(lambda: tabla_woe(_x, _y)), iv_curso_uso),
    ]
    woe_optb_uso = None
    if OPTB_OK:
        _yi = _y.astype(int)
        _fijo = lambda: OptimalBinning(dtype="numerical", user_splits=cortes_uso[1:-1],  # noqa: E731
                                       user_splits_fixed=[True] * 4, monotonic_trend=None).fit(_x, _yi)
        _ob = _fijo()
        _bt = _ob.binning_table.build()
        woe_optb_uso = _bt["WoE"].iloc[:5].to_numpy(dtype=float)
        _filas.append(("optbinning: cortes fijos (user_splits)", medir(_fijo, repeticiones=2),
                       float(_bt.loc["Totals", "IV"])))
        _opt = lambda: OptimalBinning(dtype="numerical").fit(_x, _yi)  # noqa: E731
        _ob2 = _opt()
        _filas.append(("optbinning: binning óptimo (CART + CP)", medir(_opt, repeticiones=2),
                       float(_ob2.binning_table.build().loc["Totals", "IV"])))
    bench_woe = pd.DataFrame(_filas, columns=["implementación", "segundos", "IV"])
    bench_woe["ms"] = bench_woe["segundos"] * 1e3
    bench_woe["× vs numpy"] = bench_woe["segundos"] / bench_woe["segundos"].iloc[0]
    assert np.allclose(woe_np_uso, woe_pd_uso, atol=1e-12)
    assert np.allclose(woe_np_uso, woe_curso_uso, atol=1e-12)
    bench_woe[["implementación", "ms", "× vs numpy", "IV"]].round(4)
    return (
        bench_woe,
        cortes_uso,
        iv_np_uso,
        n_np_uso,
        woe_curso_uso,
        woe_np_uso,
        woe_optb_uso,
        woe_pandas,
        woe_pd_uso,
    )


@app.cell
def _(OPTB_OK, bench_woe, coma, mo, np, woe_np_uso, woe_optb_uso):
    _txt_ob = ""
    if OPTB_OK and woe_optb_uso is not None:
        _txt_ob = (f" Con los mismos cortes, optbinning da los mismos conteos pero WoE distintos en hasta "
                   f"{coma(float(np.max(np.abs(woe_optb_uso - woe_np_uso))), 4)}: **no suaviza** (el curso suma 0,5 "
                   f"a cada celda). Y su binning óptimo encuentra otro IV porque elige otros cortes.")
    mo.md(f"""
    **Lectura.** Los tres primeros calculan **exactamente** el mismo WoE (`assert` a $10^{{-12}}$).
    numpy es {coma(bench_woe['× vs numpy'].iloc[1], 0)}× más rápido que pandas y
    {coma(bench_woe['× vs numpy'].iloc[2], 0)}× más rápido que `tabla_woe`: el costo de la versión
    del curso no es la aritmética sino construir etiquetas de texto por fila. En la práctica, en
    tiempo absoluto todas son rápidas para una variable; la diferencia importa cuando el cálculo
    se repite (100 candidatas × 1.000 réplicas bootstrap × 24 meses de monitoreo).{_txt_ob}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B2 · AUC: rangos numpy vs conteos (`bincount`) vs sklearn vs Mann-Whitney de scipy

    $\text{AUC} = P(\text{PD}_\text{malo} > \text{PD}_\text{bueno}) + \tfrac12 P(\text{empate})$.
    Tres algoritmos: rangos medios ($O(n\log n)$ por el ordenamiento), conteos por valor único
    ($O(n+k)$ si los códigos ya existen; es el caso de un scorecard con puntajes enteros),
    `roc_auc_score` (construye la curva ROC completa) y `mannwhitneyu` (calcula además el p-valor).
    """)
    return


@app.cell
def _(auc_conteos, auc_rangos, medir, np, pd, pd_dev, roc_auc_score, stats, y_dev):
    _u, codigos_dev = np.unique(pd_dev, return_inverse=True)
    k_dev = len(_u)
    _nm = y_dev.sum()
    _nb = len(y_dev) - _nm
    auc_np_dev = auc_rangos(y_dev, pd_dev)
    auc_cont_dev = auc_conteos(codigos_dev, y_dev, k_dev)
    auc_sk_dev = float(roc_auc_score(y_dev, pd_dev))
    auc_mw_dev = float(stats.mannwhitneyu(pd_dev[y_dev == 1], pd_dev[y_dev == 0]).statistic / (_nm * _nb))
    bench_auc = pd.DataFrame([
        ("numpy: rangos medios (np.unique)", medir(lambda: auc_rangos(y_dev, pd_dev)), auc_np_dev),
        ("numpy: conteos con bincount (códigos dados)", medir(lambda: auc_conteos(codigos_dev, y_dev, k_dev)), auc_cont_dev),
        ("sklearn.metrics.roc_auc_score", medir(lambda: roc_auc_score(y_dev, pd_dev)), auc_sk_dev),
        ("scipy.stats.mannwhitneyu (U + p-valor)", medir(lambda: stats.mannwhitneyu(pd_dev[y_dev == 1], pd_dev[y_dev == 0])), auc_mw_dev),
    ], columns=["implementación", "segundos", "AUC"])
    bench_auc["ms"] = bench_auc["segundos"] * 1e3
    bench_auc["× vs rangos"] = bench_auc["segundos"] / bench_auc["segundos"].iloc[0]
    assert np.allclose([auc_cont_dev, auc_sk_dev, auc_mw_dev], auc_np_dev, atol=1e-12)
    bench_auc[["implementación", "ms", "× vs rangos", "AUC"]].round(6)
    return auc_np_dev, auc_sk_dev, bench_auc, codigos_dev, k_dev


@app.cell
def _(auc_np_dev, bench_auc, coma, k_dev, miles, mo):
    mo.md(f"""
    **Lectura.** Las cuatro dan AUC = {coma(auc_np_dev, 6)} (Gini {coma(2 * auc_np_dev - 1, 4)}) a
    $10^{{-12}}$; la PD del modelo toma {miles(k_dev)} valores distintos (hay empates: el WoE es discreto).
    `roc_auc_score` es {coma(bench_auc['× vs rangos'].iloc[2], 0)}× más lento que los rangos numpy porque
    valida entradas, detecta el tipo de problema y arma la curva completa. Eso no es desperdicio:
    es la validación que el código propio no hace (etiquetas que no son 0/1, NaN, una sola clase).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B3 · Logística: IRLS numpy vs statsmodels (Logit y GLM) vs scikit-learn

    Mismo diseño (intercepto + 7 WoE, DEV). **Qué mirar:** la columna de diferencia con el MLE; la
    fila `LogisticRegression()` con los valores por defecto **no** estima el mismo modelo.
    """)
    return


@app.cell
def _(LogisticRegression, X_dev, beta_mle, irls_logit, medir, np, pd, sm, warnings, y_dev):
    def _sk(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _m = LogisticRegression(max_iter=1000, **kw).fit(X_dev[:, 1:], y_dev)
        return np.r_[_m.intercept_, _m.coef_[0]]

    _r_logit = sm.Logit(y_dev, X_dev).fit(disp=0)
    _r_glm = sm.GLM(y_dev, X_dev, family=sm.families.Binomial()).fit()
    beta_sk_defecto = _sk()
    beta_sk_inf = _sk(C=np.inf, tol=1e-10)
    beta_sm = _r_logit.params
    bench_logit = pd.DataFrame([
        ("numpy: IRLS con solve", medir(lambda: irls_logit(X_dev, y_dev)), beta_mle, "β, SE (Fisher)"),
        ("statsmodels Logit (Newton)", medir(lambda: sm.Logit(y_dev, X_dev).fit(disp=0)), beta_sm,
         "β, SE, p-valores, LR, AIC/BIC, sándwich"),
        ("statsmodels GLM Binomial (IRLS)", medir(lambda: sm.GLM(y_dev, X_dev, family=sm.families.Binomial()).fit()),
         _r_glm.params, "ídem + offset, pesos, devianza"),
        ("sklearn LogisticRegression() [C=1]", medir(lambda: _sk()), beta_sk_defecto, "β PENALIZADOS (L2)"),
        ("sklearn LogisticRegression(C=np.inf)", medir(lambda: _sk(C=np.inf, tol=1e-10)), beta_sk_inf, "β (sin SE)"),
    ], columns=["implementación", "segundos", "beta", "entrega"])
    bench_logit["ms"] = bench_logit["segundos"] * 1e3
    bench_logit["máx |β − β_MLE|"] = [float(np.max(np.abs(b - beta_mle))) for b in bench_logit["beta"]]
    assert np.allclose(beta_sm, beta_mle, atol=1e-8)
    assert np.allclose(_r_glm.params, beta_mle, atol=1e-8)
    assert np.allclose(beta_sk_inf, beta_mle, atol=1e-4)
    assert np.max(np.abs(beta_sk_defecto - beta_mle)) > 1e-3        # C=1 NO es el MLE
    bench_logit[["implementación", "ms", "máx |β − β_MLE|", "entrega"]]
    return beta_sk_defecto, beta_sk_inf, beta_sm, bench_logit


@app.cell
def _(bench_logit, coma, mo):
    mo.md(f"""
    **Lectura.** IRLS numpy, `Logit` y `GLM` coinciden a $10^{{-12}}$ o mejor. `C=np.inf` coincide a
    {bench_logit['máx |β − β_MLE|'].iloc[4]:.0e} (tolerancia de L-BFGS). `LogisticRegression()` se aleja
    {coma(bench_logit['máx |β − β_MLE|'].iloc[3], 4)} en el peor coeficiente: con 10.065 filas la
    penalización es chica; la sección C1 muestra cuándo no lo es. El IRLS de 20 líneas es
    {coma(bench_logit['segundos'].iloc[1] / bench_logit['segundos'].iloc[0], 0)}× más rápido que `Logit` y
    {coma(bench_logit['segundos'].iloc[4] / bench_logit['segundos'].iloc[0], 0)}× más rápido que sklearn: con
    7 columnas el costo está dominado por la sobrecarga (validación, objetos de resultados), no
    por el álgebra. Esa sobrecarga compra la tabla de inferencia de statsmodels, que el expediente
    necesita y que escrita a mano es fuente de errores.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B4 · Bootstrap del AUC: bucle vs matriz de índices vs `scipy.stats.bootstrap`

    La receta del curso (clase 5): remuestrear la muestra de **evaluación** con reemplazo, recalcular
    el Gini con las **mismas** predicciones, $B$ réplicas, percentiles 2,5 y 97,5. Vectorizado: se
    genera una matriz de índices $B\times n$ y se cuentan malos/buenos por (réplica, valor único)
    con un solo `bincount` sobre códigos desplazados $c + k\cdot r$. **Qué mirar:** el mismo
    resultado réplica a réplica (mismos índices) y el costo en memoria de la matriz.
    """)
    return


@app.cell
def _(mo):
    B_boot = mo.ui.slider(100, 1000, value=200, step=100, label="Réplicas bootstrap B")
    B_boot
    return (B_boot,)


@app.cell
def _(B_boot, auc_rangos, medir, np, pd, pd_oot, roc_auc_score, stats, time, y_oot):
    _u, _cod = np.unique(pd_oot, return_inverse=True)
    _k = len(_u)
    _n = len(y_oot)
    _B = int(B_boot.value)
    idx_boot = np.random.default_rng(20260917).integers(0, _n, size=(_B, _n))

    def bootstrap_auc_bucle(y, s, idx):
        return np.array([auc_rangos(y[i], s[i]) for i in idx])

    def bootstrap_auc_vectorizado(cod, y, k, idx):
        """AUC de B réplicas con un solo bincount sobre códigos desplazados k·r."""
        _B = idx.shape[0]
        _c = (cod[idx] + k * np.arange(_B)[:, None]).ravel()
        _m = np.bincount(_c, weights=y[idx].ravel(), minlength=_B * k).reshape(_B, k)
        _b = np.bincount(_c, minlength=_B * k).reshape(_B, k) - _m
        _bmenor = np.cumsum(_b, axis=1) - _b
        return (_m * (_bmenor + 0.5 * _b)).sum(1) / (_m.sum(1) * _b.sum(1))

    aucs_bucle = bootstrap_auc_bucle(y_oot, pd_oot, idx_boot)
    aucs_vec = bootstrap_auc_vectorizado(_cod, y_oot, _k, idx_boot)
    assert np.allclose(aucs_bucle, aucs_vec, atol=1e-12)

    _t0 = time.perf_counter()
    _res_scipy = stats.bootstrap((y_oot, pd_oot), lambda a, b: roc_auc_score(a, b), paired=True,
                                 vectorized=False, n_resamples=_B, method="percentile",
                                 rng=np.random.default_rng(20260917))
    _t_scipy = time.perf_counter() - _t0
    _t_sk = medir(lambda: [roc_auc_score(y_oot[i], pd_oot[i]) for i in idx_boot], objetivo=0.0, repeticiones=1)
    bench_boot = pd.DataFrame([
        ("bucle Python + auc_rangos numpy", medir(lambda: bootstrap_auc_bucle(y_oot, pd_oot, idx_boot), objetivo=0.0, repeticiones=2)),
        ("bucle Python + roc_auc_score", _t_sk),
        ("vectorizado: matriz B×n + bincount", medir(lambda: bootstrap_auc_vectorizado(_cod, y_oot, _k, idx_boot), objetivo=0.0, repeticiones=2)),
        ("scipy.stats.bootstrap(paired=True, percentile)", _t_scipy),
    ], columns=["implementación", "segundos"])
    bench_boot["× vs vectorizado"] = bench_boot["segundos"] / bench_boot["segundos"].iloc[2]
    ic_gini_vec = 2 * np.percentile(aucs_vec, [2.5, 97.5]) - 1
    ic_gini_scipy = 2 * np.array([_res_scipy.confidence_interval.low, _res_scipy.confidence_interval.high]) - 1
    mem_idx_mb = idx_boot.nbytes / 1e6
    bench_boot.round(4)
    return (
        aucs_vec,
        bench_boot,
        bootstrap_auc_bucle,
        bootstrap_auc_vectorizado,
        ic_gini_scipy,
        ic_gini_vec,
        idx_boot,
        mem_idx_mb,
    )


@app.cell
def _(B_boot, bench_boot, coma, ic_gini_scipy, ic_gini_vec, mem_idx_mb, miles, mo, y_oot):
    mo.md(f"""
    **Lectura (B = {B_boot.value}, n OOT = {miles(len(y_oot))}).** Bucle y vectorizado dan las mismas
    {B_boot.value} réplicas a $10^{{-12}}$ (mismos índices). El vectorizado es
    {coma(bench_boot['× vs vectorizado'].iloc[0], 0)}× más rápido que el bucle numpy y
    {coma(bench_boot['× vs vectorizado'].iloc[1], 0)}× más rápido que el bucle con `roc_auc_score`.
    IC 95% del Gini: vectorizado [{coma(ic_gini_vec[0], 3)}; {coma(ic_gini_vec[1], 3)}], scipy
    [{coma(ic_gini_scipy[0], 3)}; {coma(ic_gini_scipy[1], 3)}]. {"Coinciden **exactamente**: con la misma semilla, scipy 1.17 sortea la misma matriz `rng.integers(0, n, (B, n))`. Es un detalle de implementación, no un contrato documentado: no lo uses como test de paridad (una versión futura puede sortear por bloques)." if abs(ic_gini_vec[0] - ic_gini_scipy[0]) < 1e-12 else "Difieren por azar Monte Carlo (otras réplicas, mismo método)."} **Costo:** la matriz de índices ocupa
    {coma(mem_idx_mb, 1)} MB en int64; con B = 1.000 y n = 100.000 serían 800 MB. Solución: procesar en
    bloques de réplicas (p. ej. 100) o usar índices int32. `scipy.stats.bootstrap` por defecto usa
    **BCa** y **9.999** réplicas y, sin `paired=True`, remuestrea `y` y `s` por separado (destruye la
    asociación): fijar `method`, `n_resamples`, `paired` y `rng` explícitamente.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B5 · $X^\top WX$: broadcasting vs `einsum` vs `np.diag(w)`

    Cuatro formas de la misma matriz. `np.diag(w)` materializa una matriz $n\times n$ (con
    $n=10.065$ serían 810 MB): se mide solo con $n = 2.000$.
    """)
    return


@app.cell
def _(X_dev, medir, np, pd, pd_dev):
    _w = pd_dev * (1 - pd_dev)
    _ref = X_dev.T @ (X_dev * _w[:, None])
    _Xs, _ws = X_dev[:2000], _w[:2000]
    _variantes = [
        ("X.T @ (X * w[:, None])  (broadcasting + BLAS)", lambda: X_dev.T @ (X_dev * _w[:, None]), X_dev.shape[0]),
        ("np.einsum('ni,n,nj->ij')", lambda: np.einsum("ni,n,nj->ij", X_dev, _w, X_dev), X_dev.shape[0]),
        ("np.einsum(..., optimize=True)", lambda: np.einsum("ni,n,nj->ij", X_dev, _w, X_dev, optimize=True), X_dev.shape[0]),
        ("Z = X·√w ; Z.T @ Z", lambda: (lambda Z: Z.T @ Z)(X_dev * np.sqrt(_w)[:, None]), X_dev.shape[0]),
        ("X.T @ np.diag(w) @ X  (n = 2.000)", lambda: _Xs.T @ np.diag(_ws) @ _Xs, 2000),
        ("X.T @ (X * w[:, None])  (n = 2.000)", lambda: _Xs.T @ (_Xs * _ws[:, None]), 2000),
    ]
    for _nombre, _f, _nn in _variantes[:4]:
        assert np.allclose(_f(), _ref, rtol=1e-12)
    bench_xtwx = pd.DataFrame([(a, n, medir(f) * 1e3) for a, f, n in _variantes],
                              columns=["variante", "n", "ms"])
    bench_xtwx["× vs broadcasting"] = bench_xtwx["ms"] / bench_xtwx["ms"].iloc[0]
    razon_diag = float(bench_xtwx["ms"].iloc[4] / bench_xtwx["ms"].iloc[5])
    razon_einsum = float(bench_xtwx["ms"].iloc[1] / bench_xtwx["ms"].iloc[0])
    bench_xtwx.round(4)
    return bench_xtwx, razon_diag, razon_einsum


@app.cell
def _(coma, mo, razon_diag, razon_einsum):
    mo.md(f"""
    **Lectura.** `einsum` sin optimizar es {coma(razon_einsum, 1)}× más **lento** que broadcasting +
    matmul: sin `optimize=True` hace un bucle en C sin BLAS. `einsum` es una herramienta de
    **legibilidad** (la fórmula se lee como el índice de la derivación); para velocidad, verificar
    siempre. `np.diag(w)` es {coma(razon_diag, 0)}× más lento incluso con n = 2.000 y su memoria crece
    como $n^2$: es el antipatrón clásico de traducir la fórmula de libro literalmente.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### B6 · Escalamiento con $n$

    ¿Se mantienen las razones cuando $n$ crece? Datos de juguete (PD redondeada a 3 decimales →
    empates, como un score) para $n$ en una grilla geométrica hasta el valor del slider.
    **Qué mirar:** las pendientes en escala log-log (≈ 1 = lineal) y si las curvas se cruzan.
    """)
    return


@app.cell
def _(mo):
    n_max = mo.ui.slider(20_000, 400_000, value=100_000, step=20_000, label="n máximo de la grilla")
    n_max
    return (n_max,)


@app.cell
def _(auc_rangos, medir, n_max, np, pd, plt, roc_auc_score, woe_numpy, woe_pandas):
    _ns = np.unique(np.geomspace(2_000, n_max.value, 5).astype(int))
    _rng = np.random.default_rng(7)
    _filas = []
    for _n in _ns:
        _x = _rng.normal(size=_n)
        _p = np.round(1 / (1 + np.exp(-(-2.2 + 1.1 * _x))), 3)
        _y = (_rng.random(_n) < _p).astype(float)
        _c = np.r_[-np.inf, np.quantile(_x, [0.2, 0.4, 0.6, 0.8]), np.inf]
        _filas.append({
            "n": int(_n),
            "WoE numpy": medir(lambda: woe_numpy(_x, _y, _c), objetivo=0.02, repeticiones=2),
            "WoE pandas": medir(lambda: woe_pandas(_x, _y, _c), objetivo=0.02, repeticiones=2),
            "AUC numpy": medir(lambda: auc_rangos(_y, _p), objetivo=0.02, repeticiones=2),
            "AUC sklearn": medir(lambda: roc_auc_score(_y, _p), objetivo=0.02, repeticiones=2),
        })
    tabla_escala = pd.DataFrame(_filas)
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    for _col, _mk in zip(["WoE numpy", "WoE pandas", "AUC numpy", "AUC sklearn"], ["o", "s", "^", "D"]):
        _ax.loglog(tabla_escala["n"], tabla_escala[_col] * 1e3, marker=_mk, label=_col)
    _ax.set_xlabel("n (filas)")
    _ax.set_ylabel("tiempo por llamada (ms, escala log)")
    _ax.set_title("Escalamiento: la brecha de sobrecarga se cierra con n")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _pend = {c: float(np.polyfit(np.log(tabla_escala["n"]), np.log(tabla_escala[c]), 1)[0])
             for c in ["WoE numpy", "WoE pandas", "AUC numpy", "AUC sklearn"]}
    pendientes_escala = _pend
    _fig
    return pendientes_escala, tabla_escala


@app.cell
def _(coma, miles, mo, pendientes_escala, tabla_escala):
    _r0 = tabla_escala.iloc[0]
    _r1 = tabla_escala.iloc[-1]
    mo.vstack([tabla_escala.assign(**{c: tabla_escala[c] * 1e3 for c in tabla_escala.columns[1:]}).round(3)
               .rename(columns=lambda c: c if c == "n" else f"{c} (ms)"),
               mo.md(f"""
    **Lectura.** Pendientes log-log: WoE numpy {coma(pendientes_escala['WoE numpy'])}, pandas
    {coma(pendientes_escala['WoE pandas'])}, AUC numpy {coma(pendientes_escala['AUC numpy'])}, sklearn
    {coma(pendientes_escala['AUC sklearn'])}. Una pendiente < 1 revela **sobrecarga fija** que se amortiza:
    la razón pandas/numpy pasa de {coma(_r0['WoE pandas'] / _r0['WoE numpy'], 1)}× con n = {miles(_r0['n'])} a
    {coma(_r1['WoE pandas'] / _r1['WoE numpy'], 1)}× con n = {miles(_r1['n'])}; sklearn/numpy en AUC pasa de
    {coma(_r0['AUC sklearn'] / _r0['AUC numpy'], 1)}× a {coma(_r1['AUC sklearn'] / _r1['AUC numpy'], 1)}×. La
    ventaja de numpy es grande en llamadas **pequeñas y repetidas** (bootstrap, monitoreo por
    segmento) y se diluye en llamadas grandes únicas, donde lo que manda es el algoritmo
    ($O(n\\log n)$ del ordenamiento) y no el lenguaje.
    """)])
    return



@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## C. Qué aporta cada librería y dónde muerde

    ### C1 · La trampa clásica: `LogisticRegression()` regulariza con `C = 1`

    scikit-learn minimiza $\tfrac12\lVert w\rVert^2 + C\sum_i \text{logloss}_i$ (sin penalizar el
    intercepto con `lbfgs`). Dividiendo por $Cn$: es el MLE con una penalización ridge de peso
    $\lambda = 1/(Cn)$ **por observación**. Con $C=1$ fijo, la contracción depende de $n$: es
    despreciable con 10.000 filas y seria con 500. Una cartera de motos de una sucursal, un
    segmento nuevo o un modelo de cobranza temprana viven en el rango donde sí importa.
    **Qué mirar:** la razón β_sklearn/β_MLE por variable y el cambio de score en puntos.
    """)
    return


@app.cell
def _(mo):
    n_sub = mo.ui.slider(200, 10_000, value=1_000, step=200, label="n de la submuestra de DEV")
    n_sub
    return (n_sub,)


@app.cell
def _(FACTOR, LogisticRegression, VARIABLES, X_dev, X_oot, auc_rangos, irls_logit, n_sub, np, pd, warnings, y_dev, y_oot):
    def comparar_c1(n, semilla=11):
        _rng = np.random.default_rng(semilla)
        _i = _rng.choice(len(y_dev), size=min(n, len(y_dev)), replace=False)
        _mle = irls_logit(X_dev[_i], y_dev[_i])["beta"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _m = LogisticRegression(max_iter=2000).fit(X_dev[_i, 1:], y_dev[_i])
        _sk = np.r_[_m.intercept_, _m.coef_[0]]
        return _i, _mle, _sk

    _i, beta_c1_mle, beta_c1_sk = comparar_c1(n_sub.value)
    _lp_mle_oot = X_oot @ beta_c1_mle
    _lp_sk_oot = X_oot @ beta_c1_sk
    tabla_c1 = pd.DataFrame({"parámetro": ["intercepto"] + VARIABLES,
                             "β MLE": beta_c1_mle, "β sklearn C=1": beta_c1_sk})
    tabla_c1["razón sk/MLE"] = tabla_c1["β sklearn C=1"] / tabla_c1["β MLE"]

    def contraccion_teorica(i, beta):
        """Un paso de Newton desde el MLE para el objetivo penalizado (C = 1):
        β_C ≈ (H + I₀/C)⁻¹ H β̂, con H = XᵀŴX (suma, no promedio) e I₀ = identidad sin el intercepto.
        Versión diagonal: factor_j ≈ h_j / (h_j + 1/C), h_j = Σ ŵ_i (x_ij − x̄_w)²."""
        _p = 1 / (1 + np.exp(-(X_dev[i] @ beta)))
        _w = _p * (1 - _p)
        _H = X_dev[i].T @ (X_dev[i] * _w[:, None])
        _I0 = np.eye(X_dev.shape[1])
        _I0[0, 0] = 0.0
        _aprox = np.linalg.solve(_H + _I0, _H @ beta)
        _Xc = X_dev[i] - np.average(X_dev[i], axis=0, weights=_w)
        _h = np.sum(_w[:, None] * _Xc ** 2, axis=0)
        return _aprox, _h

    _aprox, _h = contraccion_teorica(_i, beta_c1_mle)
    tabla_c1["razón 1 paso (teoría)"] = _aprox / beta_c1_mle
    tabla_c1["h_j"] = _h
    tabla_c1["h_j/(h_j+1)"] = _h / (_h + 1)
    tabla_c1.loc[0, ["h_j", "h_j/(h_j+1)"]] = np.nan
    _i1000, _m1000, _s1000 = comparar_c1(1000)
    _a1000, _ = contraccion_teorica(_i1000, _m1000)
    err_aprox_c1_1000 = float(np.max(np.abs(_a1000[1:] / _m1000[1:] - _s1000[1:] / _m1000[1:])))
    _p_sk_sub = 1 / (1 + np.exp(-(X_dev[_i] @ beta_c1_sk)))
    resumen_c1 = {
        "n": len(_i), "malos": int(y_dev[_i].sum()),
        "contraccion_norma": float(np.linalg.norm(beta_c1_sk[1:]) / np.linalg.norm(beta_c1_mle[1:])),
        "max_dif_puntos_oot": float(np.max(np.abs(_lp_sk_oot - _lp_mle_oot)) * FACTOR),
        "gini_oot_mle": 2 * auc_rangos(y_oot, _lp_mle_oot) - 1,
        "gini_oot_sk": 2 * auc_rangos(y_oot, _lp_sk_oot) - 1,
        "brecha_media_sk": float(abs(_p_sk_sub.mean() - y_dev[_i].mean())),
        "pd_p95_mle": float(np.percentile(1 / (1 + np.exp(-_lp_mle_oot)), 95)),
        "pd_p95_sk": float(np.percentile(1 / (1 + np.exp(-_lp_sk_oot)), 95)),
    }
    # curva de contracción vs n (promedio de 3 submuestras por n)
    _grid = [300, 500, 1000, 2000, 5000, len(y_dev)]
    curva_c1 = pd.DataFrame([{
        "n": g,
        "‖β_sk‖/‖β_MLE‖": float(np.mean([np.linalg.norm(comparar_c1(g, s)[2][1:]) / np.linalg.norm(comparar_c1(g, s)[1][1:])
                                        for s in (1, 2, 3)]))} for g in _grid])
    tabla_c1.round(4)
    return (
        beta_c1_mle,
        beta_c1_sk,
        comparar_c1,
        contraccion_teorica,
        curva_c1,
        err_aprox_c1_1000,
        resumen_c1,
        tabla_c1,
    )


@app.cell
def _(coma, curva_c1, miles, mo, plt, resumen_c1, tabla_c1):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.semilogx(curva_c1["n"], curva_c1["‖β_sk‖/‖β_MLE‖"], "o-", label="LogisticRegression() vs MLE")
    _ax.axhline(1, color="k", lw=0.8, ls=":")
    _ax.axvline(resumen_c1["n"], color="grey", alpha=0.4)
    _ax.set_xlabel("n de desarrollo (escala log)")
    _ax.set_ylabel("‖β_sklearn‖ / ‖β_MLE‖")
    _ax.set_title("C = 1 por defecto: la contracción depende de n")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"""
    **Lectura (n = {miles(resumen_c1['n'])}, {resumen_c1['malos']} malos).** La norma de las pendientes de
    sklearn es {coma(resumen_c1['contraccion_norma'] * 100, 1)}% de la del MLE. El score OOT cambia hasta
    {coma(resumen_c1['max_dif_puntos_oot'], 1)} puntos; el Gini OOT pasa de {coma(resumen_c1['gini_oot_mle'], 4)} a
    {coma(resumen_c1['gini_oot_sk'], 4)} (la contracción puede incluso **ayudar** al ranking fuera de muestra:
    es regularización legítima). El problema no es que sea malo: es que **no es el modelo que el
    expediente dice** («logística por máxima verosimilitud»), no trae errores estándar y comprime las PD
    extremas (percentil 95 de PD OOT: {coma(resumen_c1['pd_p95_mle'] * 100, 2)}% MLE vs
    {coma(resumen_c1['pd_p95_sk'] * 100, 2)}% sklearn). La PD media en la muestra de ajuste sí coincide
    (brecha {resumen_c1['brecha_media_sk']:.1e}) porque el intercepto no se penaliza: por eso la trampa no
    se ve en el primer control que uno hace. En el curso, `Scorecard(estimator=LogisticRegression(max_iter=1000))`
    de las clases 3 y 5 usa este default.

    **Por qué unas variables se contraen más que otras.** Un paso de Newton desde el MLE da
    $\hat\beta_C\approx(H+I_0/C)^{{-1}}H\hat\beta$ (columna «razón 1 paso», que reproduce la de sklearn), y su
    versión diagonal, $h_j/(h_j+1/C)$ con $h_j\approx n\,\bar w\,\mathrm{{Var}}(\text{{WoE}}_j)$. La penalización
    es la misma para todas; lo que cambia es la **información** de cada columna. `canal` tiene WoE de poca
    dispersión (IV bajo): $h$ = {coma(tabla_c1['h_j'].iloc[-1], 1)} y factor
    {coma(tabla_c1['h_j/(h_j+1)'].iloc[-1], 2)}. Las variables de mayor IV casi no se tocan.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C2 · Lo que statsmodels entrega y numpy no (sin esfuerzo): inferencia, offset, diagnóstico

    Tres cosas: (1) la tabla de inferencia (SE, z, p-valores) que el IRLS numpy también puede
    dar, pero que hay que programar y testear; (2) **GLM con offset**, que es exactamente la
    calibración de intercepto δ de la clase 4 cuando el objetivo es la tasa de la propia muestra;
    (3) el comportamiento ante **separación**, donde las dos APIs de statsmodels no dicen lo mismo.
    """)
    return


@app.cell
def _(VARIABLES, X_dev, brentq, delta_newton, lp_oot, modelo_dev, np, pd, sm, special, warnings, y_dev, y_oot):
    _r = sm.Logit(y_dev, X_dev).fit(disp=0)
    tabla_inferencia = pd.DataFrame({
        "parámetro": ["intercepto"] + VARIABLES, "β": _r.params, "SE statsmodels": _r.bse,
        "SE numpy (Fisher)": modelo_dev["se"], "p-valor (Wald)": _r.pvalues})
    assert np.allclose(_r.bse, modelo_dev["se"], rtol=1e-8)

    # δ: GLM con offset = brentq = Newton numpy (objetivo: tasa observada de OOT)
    _tasa = float(y_oot.mean())
    delta_glm = float(sm.GLM(y_oot, np.ones((len(y_oot), 1)), family=sm.families.Binomial(), offset=lp_oot).fit().params[0])
    delta_brentq = float(brentq(lambda d: special.expit(lp_oot + d).mean() - _tasa, -5, 5, xtol=1e-14))
    delta_np = float(delta_newton(lp_oot, _tasa))
    delta_aprox = float(special.logit(_tasa) - special.logit(special.expit(lp_oot).mean()))

    # separación perfecta: Logit avisa y marca no convergido; GLM «converge»
    _x = np.r_[np.arange(10.0), np.arange(10.0)]
    _y = (_x > 4).astype(float)
    _Xs = sm.add_constant(_x)
    with warnings.catch_warnings(record=True) as _w1:
        warnings.simplefilter("always")
        _rl = sm.Logit(_y, _Xs).fit(disp=0)
    with warnings.catch_warnings(record=True) as _w2:
        warnings.simplefilter("always")
        _rg = sm.GLM(_y, _Xs, family=sm.families.Binomial()).fit()
    separacion = {
        "logit_converged": bool(_rl.mle_retvals["converged"]), "logit_beta": float(_rl.params[1]),
        "logit_avisos": sorted({w.category.__name__ for w in _w1}),
        "glm_converged": bool(_rg.converged), "glm_beta": float(_rg.params[1]),
        "glm_avisos": sorted({w.category.__name__ for w in _w2}),
    }
    tabla_inferencia.round(5)
    return delta_aprox, delta_brentq, delta_glm, delta_np, separacion, tabla_inferencia


@app.cell
def _(coma, delta_aprox, delta_brentq, delta_glm, delta_np, mo, separacion):
    mo.md(f"""
    **Lectura.** SE numpy = SE statsmodels a $10^{{-8}}$ relativo. **δ para llevar la PD media OOT a
    la tasa observada OOT:** GLM con offset {coma(delta_glm, 6)}, `brentq` {coma(delta_brentq, 6)}, Newton
    numpy {coma(delta_np, 6)}: el mismo número por tres caminos; la aproximación
    logit(tasa) − logit(PD media) da {coma(delta_aprox, 4)} (subestima, como el 0,143 vs 0,177 de Banco
    Austral). El GLM entrega además el SE de δ, útil para decidir si recalibrar.

    **Separación perfecta** (x = 0…9, malo si x > 4): `Logit` devuelve pendiente
    {coma(separacion['logit_beta'], 1)} con `converged = {separacion['logit_converged']}` y avisos
    {', '.join(separacion['logit_avisos'])}. `GLM` devuelve pendiente {coma(separacion['glm_beta'], 1)} con
    **`converged = {separacion['glm_converged']}`** y solo {', '.join(separacion['glm_avisos'])}: su criterio de
    parada (cambio de devianza) se cumple aunque el MLE no exista. Un pipeline que solo lee
    `.converged` del GLM aprueba un modelo sin MLE. Los avisos son la única señal: el pipeline debe
    convertirlos en errores (`warnings.simplefilter("error", PerfectSeparationWarning)`).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C3 · optbinning: signo del WoE, intervalos `[a, b)`, especiales y el default `metric_special=0`

    Se verifican sobre `meses_desde_mora_12m` (entera, con 13 = sin mora y los códigos −9 y −99),
    la variable donde las convenciones chocan.
    """)
    return


@app.cell
def _(OPTB_OK, OptimalBinning, binear, dev, np, pd):
    _x = dev["meses_desde_mora_12m"].to_numpy()
    _y = dev["malo"].astype(int).to_numpy()
    conv_optb = {}
    if OPTB_OK:
        _ob_lista = OptimalBinning(dtype="numerical", special_codes=[-9, -99]).fit(_x, _y)
        _bt_l = _ob_lista.binning_table.build()
        _ob_dict = OptimalBinning(dtype="numerical", special_codes={"nunca_mora": [-9], "sin_bureau": [-99]}).fit(_x, _y)
        _bt_d = _ob_dict.binning_table.build()
        # signo: WoE de optbinning vs ln(%no evento / %evento) calculado a mano, sin suavizado
        _fila = _bt_d.iloc[0]
        _pct_ne = (_fila["Count"] - _fila["Event"]) / (_bt_d.loc["Totals", "Count"] - _bt_d.loc["Totals", "Event"])
        _pct_e = _fila["Event"] / _bt_d.loc["Totals", "Event"]
        conv_optb["woe_manual"] = float(np.log(_pct_ne / _pct_e))
        conv_optb["woe_optb"] = float(_fila["WoE"])
        conv_optb["tasa_bin0"] = float(_fila["Event rate"])
        conv_optb["tasa_total"] = float(_bt_d.loc["Totals", "Event rate"])
        # intervalos [a, b): un valor igual al corte cae en el bin SUPERIOR
        _s = float(_ob_dict.splits[0])
        conv_optb["corte"] = _s
        conv_optb["bins_en_corte"] = list(_ob_dict.transform(np.array([_s - 1e-9, _s]), metric="bins"))
        # especiales: WoE empírico en la tabla vs 0 en transform por defecto
        _woe_sb_tabla = float(_bt_d.loc[_bt_d["Bin"].astype(str) == "sin_bureau", "WoE"].iloc[0])
        conv_optb["woe_sin_bureau_tabla"] = _woe_sb_tabla
        conv_optb["woe_sin_bureau_transform"] = float(_ob_dict.transform(np.array([-99.0]), metric="woe")[0])
        conv_optb["woe_sin_bureau_empirico"] = float(_ob_dict.transform(np.array([-99.0]), metric="woe",
                                                                     metric_special="empirical")[0])
        conv_optb["n_bins_especiales_lista"] = int((_bt_l["Bin"].astype(str) == "Special").sum())
        conv_optb["n_bins_especiales_dict"] = int(_bt_d["Bin"].astype(str).isin(["nunca_mora", "sin_bureau"]).sum())
        tabla_optb = _bt_d[["Bin", "Count", "Event", "Event rate", "WoE", "IV"]].copy()
    else:
        tabla_optb = pd.DataFrame({"aviso": ["optbinning no disponible"]})

    # binner del curso: ¿separa −9 de −99?
    _et, _orden = binear(dev["meses_desde_mora_12m"])
    _et = _et.to_numpy()
    conv_optb["etiqueta_curso_m9"] = str(_et[_x == -9][0])
    conv_optb["etiqueta_curso_m99"] = str(_et[_x == -99][0])
    conv_optb["tasa_m9"] = float(_y[_x == -9].mean())
    conv_optb["tasa_m99"] = float(_y[_x == -99].mean())
    tabla_optb
    return conv_optb, tabla_optb


@app.cell
def _(dev, np):
    # ¿cuántas filas cambian de bin si los mismos cortes se aplican como [a, b) en vez de (a, b]?
    _x = dev["meses_desde_mora_12m"].to_numpy()
    _no13 = _x[_x != 13]
    _cortes = np.unique(np.nanquantile(_no13, np.linspace(0, 1, 6)))[1:-1]   # cortes interiores del curso
    _izq = np.searchsorted(_cortes, _x, side="left")    # (a, b]  = pd.cut del curso
    _der = np.searchsorted(_cortes, _x, side="right")   # [a, b)  = optbinning / np.digitize
    _cambia = (_izq != _der) & (_x != 13)
    filas_cambian_borde = int(np.sum(_cambia))
    desglose_borde = {float(v): int(np.sum(_cambia & (_x == v))) for v in _cortes}
    cortes_mora_curso = _cortes
    return cortes_mora_curso, desglose_borde, filas_cambian_borde


@app.cell
def _(OPTB_OK, OPTB_VERSION, coma, conv_optb, cortes_mora_curso, desglose_borde, dev, filas_cambian_borde, miles, mo):
    if OPTB_OK:
        _t = f"""
    **Lectura (optbinning {OPTB_VERSION}).**

    - **Signo del WoE.** Primer bin: tasa de malos {coma(conv_optb['tasa_bin0'] * 100, 1)}% (vs
      {coma(conv_optb['tasa_total'] * 100, 1)}% total), WoE optbinning {coma(conv_optb['woe_optb'], 4)} =
      ln(%no evento/%evento) a mano {coma(conv_optb['woe_manual'], 4)}. **Mismo signo que el curso**
      (bin malo → WoE negativo), sin suavizado.
    - **Intervalos `[a, b)`.** El valor {coma(conv_optb['corte'], 2)} (igual al corte) cae en
      `{conv_optb['bins_en_corte'][1]}`, no en `{conv_optb['bins_en_corte'][0]}`. `pd.cut` del curso usa `(a, b]`.
    - **Especiales.** Con lista `[-9, -99]` optbinning crea {conv_optb['n_bins_especiales_lista']} bin
      «Special» (los junta); con diccionario, {conv_optb['n_bins_especiales_dict']} bins. `sin_bureau`
      tiene WoE {coma(conv_optb['woe_sin_bureau_tabla'], 4)} en la tabla, pero `transform()` por
      defecto devuelve **{coma(conv_optb['woe_sin_bureau_transform'], 1)}** (`metric_special=0`): puntos neutros para el
      segmento más riesgoso. Con `metric_special="empirical"`: {coma(conv_optb['woe_sin_bureau_empirico'], 4)}.
    """
    else:
        _t = "optbinning no disponible: se omiten sus verificaciones."
    mo.md(_t + f"""
    - **Binner del curso.** −9 recibe la etiqueta `{conv_optb['etiqueta_curso_m9']}` y −99 la etiqueta
      `{conv_optb['etiqueta_curso_m99']}`: el mismo bin, aunque sus tasas de malos son
      {coma(conv_optb['tasa_m9'] * 100, 1)}% y {coma(conv_optb['tasa_m99'] * 100, 1)}%. La clase 4 (v21) pide
      separarlos; el código de clase no lo hace.
    - **Borde de intervalo.** Con los cortes del curso para esta variable
      ({', '.join(coma(c, 1) for c in cortes_mora_curso)}), aplicar `[a, b)` en vez de `(a, b]` mueve de bin a
      **{miles(filas_cambian_borde)}** de {miles(len(dev))} filas de DEV ({coma(100 * filas_cambian_borde / len(dev), 1)}%): en una
      variable entera, los cortes caen *sobre* valores observados
      ({'; '.join(f"valor {coma(k, 0)}: {miles(v)} filas" for k, v in desglose_borde.items())}). Lo peor: con
      `[a, b)` el código −9 («nunca tuvo mora», {coma(conv_optb['tasa_m9'] * 100, 1)}% de malos) deja el bin de −99 y pasa al
      bin de mora reciente `[-9, 3)`. Es el mismo tipo de discrepancia silenciosa que el bug del
      binner re-ajustado de la clase 6.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C4 · Tabla de convenciones verificadas empíricamente

    Cada fila se verifica con un `assert` en la celda: si una versión futura cambia el default,
    el notebook falla. Esta tabla es el tipo de artefacto que conviene versionar junto al
    `requirements.lock`: documenta **qué supuestos del código dependen de defaults de terceros**.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    OPTB_OK,
    X_dev,
    auc_np_dev,
    beta_mle,
    beta_sk_defecto,
    conv_optb,
    hosmer_lemeshow_numpy,
    np,
    pd,
    pd_dev,
    pd_oot,
    roc_auc_score,
    stats,
    test_chisquare_binning,
    warnings,
    y_dev,
    y_oot,
):
    import inspect as _inspect
    from statsmodels.stats.proportion import proportion_confint as _pc
    _filas = []

    def _fila(tema, funcion, verificado, implicancia):
        _filas.append({"tema": tema, "función": funcion, "comportamiento verificado": verificado,
                       "qué hacer": implicancia})

    # 1-3 sklearn
    assert LogisticRegression().C == 1.0 and np.max(np.abs(beta_sk_defecto - beta_mle)) > 1e-3
    _fila("penalización", "sklearn LogisticRegression()", "C = 1.0 → L2; β ≠ MLE",
          "C=np.inf explícito o statsmodels")
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        LogisticRegression(penalty=None, max_iter=1000).fit(X_dev[:, 1:], y_dev)
    _cats_pen = {w.category.__name__ for w in _w}
    assert "FutureWarning" in _cats_pen
    _fila("API", "sklearn 1.8 LogisticRegression(penalty=…)", "FutureWarning: `penalty` deprecado (se elimina en 1.10)",
          "usar C y l1_ratio; no copiar código viejo")
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        LogisticRegression(C=np.inf, max_iter=1000).fit(X_dev[:, 1:], y_dev)
    _msgs_inf = [str(w.message) for w in _w]
    assert any("penalty=None" in m for m in _msgs_inf)
    _fila("API", "sklearn LogisticRegression(C=np.inf)", "UserWarning «Setting penalty=None will ignore…»; β = MLE",
          "aviso esperado: filtrarlo explícitamente")
    # 4 AUC: orden de argumentos
    assert np.isclose(roc_auc_score(y_dev, -pd_dev), 1 - auc_np_dev)
    _fila("signo", "roc_auc_score(y_true, y_score)", "espera score creciente en la clase 1 (malo): pasar PD, no puntos",
          "con puntos: roc_auc_score(y, -score)")
    # 5-8 optbinning
    if OPTB_OK:
        assert np.sign(conv_optb["woe_optb"]) == np.sign(conv_optb["woe_manual"]) and conv_optb["tasa_bin0"] > conv_optb["tasa_total"]
        assert np.isclose(conv_optb["woe_optb"], conv_optb["woe_manual"]) and conv_optb["woe_optb"] < 0
        _fila("signo WoE", "optbinning binning_table / transform", "WoE = ln(%no evento/%evento): mismo signo que el curso; sin +0,5",
              "diferencias de 1e-3 vs curso por suavizado")
        assert conv_optb["bins_en_corte"][1] != conv_optb["bins_en_corte"][0]
        _fila("intervalos", "optbinning", "[a, b) (valor = corte va arriba)", "curso/pd.cut: (a, b]; declarar en el artefacto")
        assert conv_optb["woe_sin_bureau_transform"] == 0.0 and conv_optb["woe_sin_bureau_tabla"] != 0.0
        _fila("especiales", "OptimalBinning.transform()", "metric_special=0, metric_missing=0 por defecto",
              "metric_special='empirical' o mapeo explícito")
        assert conv_optb["n_bins_especiales_lista"] == 1 and conv_optb["n_bins_especiales_dict"] == 2
        _fila("especiales", "special_codes=[-9, -99]", "lista → UN bin «Special»; dict → un bin por clave", "usar dict")
    assert conv_optb["etiqueta_curso_m9"] == conv_optb["etiqueta_curso_m99"]
    _fila("especiales", "binear() del curso", "−9 y −99 caen en el mismo bin numérico", "bins propios por código")
    # 9-10 ddof
    _v = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(np.std(_v), np.sqrt(1.25)) and np.isclose(pd.Series(_v).std(), np.sqrt(5 / 3))
    _fila("ddof", "np.std / np.var vs pandas .std/.var", "numpy ddof=0; pandas ddof=1", "fijar ddof siempre")
    assert np.isclose(np.cov(_v, _v)[0, 1], np.var(_v, ddof=1))
    _fila("ddof", "np.cov vs np.var", "np.cov usa n−1 (bias=False); np.var usa n", "inconsistencia DENTRO de numpy")
    # 11 binomtest
    _pb = stats.binomtest(3, 50, 0.02).pvalue
    _p2 = 2 * min(stats.binom.cdf(3, 50, 0.02), stats.binom.sf(2, 50, 0.02))
    assert _inspect.signature(stats.binomtest).parameters["alternative"].default == "two-sided" and not np.isclose(_pb, _p2)
    _fila("bilateral", "scipy.stats.binomtest", f"two-sided = Σ P(X=j) con P(X=j) ≤ P(X=k): {_pb:.4f} vs 2×cola {_p2:.4f}",
          "backtesting de PD: alternative='greater' si solo preocupa subestimar")
    # 12 proportion_confint
    _lo, _hi = _pc(1, 200)
    assert _inspect.signature(_pc).parameters["method"].default == "normal"
    _fila("IC", "statsmodels proportion_confint", f"method='normal' (Wald) por defecto; k=1, n=200 → [{_lo:.4f}; {_hi:.4f}]",
          "method='wilson' o 'beta' en carteras chicas")
    # 13 Yates
    _t = np.array([[30, 170], [45, 155]])
    assert stats.chi2_contingency(_t)[0] < stats.chi2_contingency(_t, correction=False)[0]
    _fila("continuidad", "scipy chi2_contingency", "correction=True (Yates) en tablas 2×2", "correction=False para igualar la fórmula")
    # 14 bootstrap
    _sig = _inspect.signature(stats.bootstrap).parameters
    assert _sig["method"].default == "BCa" and _sig["n_resamples"].default == 9999 and _sig["paired"].default is False
    _fila("bootstrap", "scipy.stats.bootstrap", "method='BCa', n_resamples=9999, paired=False", "fijar los tres + rng")
    # 15 Mann-Whitney
    _fila("AUC", "scipy mannwhitneyu(x, y).statistic", "U de la PRIMERA muestra; U/(n₁n₂) = AUC si x = PD de malos",
          "use_continuity solo afecta al p-valor")
    # 16 Hosmer-Lemeshow
    _hl_np = hosmer_lemeshow_numpy(y_oot, pd_oot)
    _res_hl = test_chisquare_binning(np.column_stack([1 - y_oot, y_oot]), np.column_stack([1 - pd_oot, pd_oot]),
                                     sort_var=pd_oot, bins=10)
    assert np.isclose(_hl_np, _res_hl.statistic, rtol=1e-10) and _res_hl.df == 8
    assert abs(stats.chi2.sf(29.0, 8) - 0.0003) < 5e-5
    _fila("HL", "statsmodels test_chisquare_binning", "df = g−2 por defecto (in-sample); grupos por array_split",
          "out-of-sample: df = g (Stata) o p simulado (clase 5)")
    # 17 bordes numpy
    assert list(np.searchsorted([0, 1, 2, 3], [1, 2])) == [1, 2] and list(np.digitize([1, 2], [0, 1, 2, 3])) == [2, 3]
    assert list(np.histogram([0, 1, 2, 3], bins=[0, 1, 2, 3])[0]) == [1, 1, 2]
    _fila("intervalos", "searchsorted / digitize / histogram / pd.cut",
          "searchsorted(side='left') ≙ (a,b]; digitize ≙ [a,b); histogram: último bin cerrado; pd.cut: (a,b]",
          "un test por borde en el artefacto")
    tabla_convenciones = pd.DataFrame(_filas)
    hl_oot_np = _hl_np
    tabla_convenciones
    return hl_oot_np, tabla_convenciones


@app.cell
def _(mo, tabla_convenciones):
    mo.md(f"""
    **Lectura.** {len(tabla_convenciones)} convenciones verificadas en este entorno. Ninguna es un
    bug: son decisiones de diseño razonables de cada librería. El riesgo está en la **brecha entre la
    convención de la librería y la del expediente** (curso: WoE con +0,5, intervalos `(a, b]`, MLE sin
    penalizar, especiales con bin propio, HL con p simulado). Esa brecha no la detecta ningún test
    unitario de la librería: la detecta un test de paridad propio contra una especificación propia.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### C5 · Dependencias y costo de arranque

    La «superficie de riesgo» de una dependencia se aproxima por su **clausura transitiva** (cuántos
    paquetes arrastra) y su costo de importación (latencia de arranque en un *worker* de scoring o
    una función *serverless*). Se lee de los metadatos instalados (`importlib.metadata`, sin red) y
    se mide importando cada librería en un proceso limpio.
    """)
    return


@app.cell
def _(importlib_md, pd, subprocess, sys, time):
    import re as _re

    def _deps_directas(nombre):
        try:
            _d = importlib_md.distribution(nombre)
        except importlib_md.PackageNotFoundError:
            return None
        _out = []
        for _r in _d.requires or []:
            _req, _, _marca = _r.partition(";")
            if "extra" in _marca:
                continue
            if _marca.strip():
                try:
                    from packaging.markers import Marker as _Marker
                    if not _Marker(_marca).evaluate():
                        continue
                except Exception:  # noqa: BLE001
                    pass
            _out.append(_re.split(r"[\s<>=!~\[(]", _req.strip(), maxsplit=1)[0].lower().replace("_", "-"))
        return _out

    def clausura_dependencias(nombre):
        _vistos, _pila = set(), [nombre.lower()]
        while _pila:
            _a = _pila.pop()
            if _a in _vistos:
                continue
            _vistos.add(_a)
            _pila.extend(_deps_directas(_a) or [])
        _vistos.discard(nombre.lower())
        return sorted(_vistos)

    _filas = []
    for _paq, _mod in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy.stats"),
                       ("statsmodels", "statsmodels.api"), ("scikit-learn", "sklearn.linear_model"),
                       ("optbinning", "optbinning")]:
        try:
            _ver = importlib_md.version(_paq)
        except importlib_md.PackageNotFoundError:
            continue
        _cl = clausura_dependencias(_paq)
        _t0 = time.perf_counter()
        _ok = subprocess.run([sys.executable, "-c", f"import {_mod}"], capture_output=True).returncode == 0
        _seg = time.perf_counter() - _t0
        _filas.append({"paquete": _paq, "versión": _ver, "dependencias transitivas": len(_cl),
                       "import en proceso limpio (s)": round(_seg, 2) if _ok else None,
                       "ejemplos": ", ".join(_cl[:8]) + ("…" if len(_cl) > 8 else "")})
    tabla_dependencias = pd.DataFrame(_filas)
    tabla_dependencias
    return clausura_dependencias, tabla_dependencias


@app.cell
def _(coma, mo, tabla_dependencias):
    _d = tabla_dependencias.set_index("paquete")
    _ob = _d.loc["optbinning"] if "optbinning" in _d.index else None
    mo.md(f"""
    **Lectura.** numpy no arrastra nada ({_d.loc['numpy', 'dependencias transitivas']} dependencias);
    scipy, {_d.loc['scipy', 'dependencias transitivas']}; scikit-learn, {_d.loc['scikit-learn', 'dependencias transitivas']};
    statsmodels, {_d.loc['statsmodels', 'dependencias transitivas']}{'' if _ob is None else f"; optbinning, **{_ob['dependencias transitivas']}** (incluye ortools, cvxpy, protobuf, matplotlib)"}.
    Importar statsmodels cuesta ~{coma(_d.loc['statsmodels', 'import en proceso limpio (s)'], 1)} s y numpy
    ~{coma(_d.loc['numpy', 'import en proceso limpio (s)'], 1)} s (incluye arrancar Python). Para un
    notebook es irrelevante; para un servicio de scoring que escala a cero o para una imagen que
    un equipo de seguridad debe escanear, cada dependencia es un paquete que auditar, fijar y
    parchar. **La función de scoring de producción no necesita ninguna de estas librerías:** es una
    tabla de búsqueda y una suma (sección E).
    """)
    return



@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## D. nikodym: la capa de gobierno (descrita, no ejecutada)

    nikodym (versión 1.11.0 en la clase 6) no está instalado en este entorno y el notebook no lo
    importa. Lo que sigue se lee del notebook de clase `demo_c6_nikodym_austral.ipynb`:

    | Pieza | API en la demo | Qué resuelve que numpy/scipy no |
    |---|---|---|
    | Receta declarativa | `standard_preset()` → dict → `NikodymConfig.model_validate(cfg)` | config validado (tipos, rangos) y `config_hash` |
    | Contrato de datos | `check_dataset(config, columns=…)`; reglas `le`/`ge` en el esquema | bloquea la corrida (`DataValidationError`: 331 hallazgos) antes del binning |
    | Corrida | `nikodym.run(config)` → `Study` (`status done/failed`, `run_id`) | la falla queda registrada, no es una excepción suelta |
    | Artefactos | `study.artifacts.get("scorecard", "scorecard")`, `("performance", …)`, `("validation", …)` | objetos publicados por etapa, sin recalcular |
    | Audit trail | `read_trail(TRAIL)` → 398 eventos, 341 decisiones | registro cronológico de reglas aplicadas |
    | Lineage | `study.lineage_bundle()`: `config_hash`, `data_hash`, `root_seed`, versiones | reproducibilidad identificable |
    | Model card | `ModelCardBuilder(GovernanceConfig).build(study, trail_path=…)` | ficha JSON + Markdown ligada al `run_id` |

    Lo que la demo **no** delega a la librería: la cadena de hashes y el sello externo (funciones
    `encadenar`/`verificar` escritas en el notebook, «no son parte de la API de la librería en
    1.11.0»), la recomendación, los gatillos y el veredicto de validación. Las etiquetas del
    scorecard de la demo (`(-inf, -0.21)`, `[-0.21, -0.13)`, `Special`, `Missing`, WoE 0 para
    `Special`) tienen el formato de optbinning; **no está verificado** aquí si el motor de binning
    de nikodym usa optbinning internamente, pero el lector debe revisar si hereda sus defaults
    (intervalos `[a, b)`, `metric_special`).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## E. Núcleo numpy + librerías como oráculo de pruebas

    La estrategia que recomienda el módulo: el **artefacto y las métricas de monitoreo** en numpy
    puro (auditables, sin dependencias, portables); las librerías, como **oráculo** en un test de
    paridad que corre en CI sobre muchas carteras aleatorias. El arnés de abajo es ese test: 12
    carteras sintéticas con semillas distintas, cinco cálculos centrales, la máxima discrepancia
    observada y la tolerancia declarada.
    """)
    return


@app.cell
def _(
    auc_rangos,
    brentq,
    delta_newton,
    generar_cartera,
    hosmer_lemeshow_numpy,
    irls_logit,
    np,
    pd,
    roc_auc_score,
    sm,
    special,
    test_chisquare_binning,
    woe_numpy,
    woe_pandas,
):
    def arnes_paridad(semillas=range(100, 112), n=4_000):
        """Test de paridad numpy ↔ librería sobre carteras aleatorias. Devuelve la máxima
        discrepancia por cálculo. En CI: una fila por (cálculo, semilla) y assert por tolerancia."""
        _vars = ["uso_linea_prom_12m", "uso_tc_prom_12m", "antiguedad_meses", "carga_financiera"]
        _max = {"WoE (numpy vs pandas)": 0.0, "AUC (numpy vs sklearn)": 0.0, "β (IRLS vs statsmodels)": 0.0,
                "SE (Fisher vs statsmodels)": 0.0, "HL (numpy vs statsmodels)": 0.0, "δ (Newton vs brentq)": 0.0}
        for _s in semillas:
            _c = generar_cartera(n=n, semilla=int(_s))
            _d = _c[_c["muestra"] == "DEV"]
            _y = _d["malo"].to_numpy()
            _cols = [np.ones(len(_d))]
            for _v in _vars:
                _x = _d[_v].to_numpy()
                _ct = np.unique(np.quantile(_x, np.linspace(0, 1, 6)))
                _ct[0], _ct[-1] = -np.inf, np.inf
                _w1, _, _ = woe_numpy(_x, _y, _ct)
                _w2, _ = woe_pandas(_x, _y, _ct)
                _max["WoE (numpy vs pandas)"] = max(_max["WoE (numpy vs pandas)"], float(np.max(np.abs(_w1 - _w2))))
                _cols.append(_w1[np.searchsorted(_ct[1:-1], _x, side="left")])
            _X = np.column_stack(_cols)
            _m = irls_logit(_X, _y)
            _r = sm.Logit(_y, _X).fit(disp=0)
            _max["β (IRLS vs statsmodels)"] = max(_max["β (IRLS vs statsmodels)"], float(np.max(np.abs(_m["beta"] - _r.params))))
            _max["SE (Fisher vs statsmodels)"] = max(_max["SE (Fisher vs statsmodels)"], float(np.max(np.abs(_m["se"] - _r.bse))))
            _lp = _X @ _m["beta"]
            _p = special.expit(_lp)
            _max["AUC (numpy vs sklearn)"] = max(_max["AUC (numpy vs sklearn)"], abs(auc_rangos(_y, _p) - roc_auc_score(_y, _p)))
            _hl = test_chisquare_binning(np.column_stack([1 - _y, _y]), np.column_stack([1 - _p, _p]), sort_var=_p, bins=10)
            _max["HL (numpy vs statsmodels)"] = max(_max["HL (numpy vs statsmodels)"], abs(hosmer_lemeshow_numpy(_y, _p) - _hl.statistic))
            _obj = 0.05
            _max["δ (Newton vs brentq)"] = max(_max["δ (Newton vs brentq)"], abs(
                delta_newton(_lp, _obj) - brentq(lambda d: special.expit(_lp + d).mean() - _obj, -10, 10, xtol=1e-14)))
        _tol = {"WoE (numpy vs pandas)": 1e-12, "AUC (numpy vs sklearn)": 1e-12, "β (IRLS vs statsmodels)": 1e-8,
                "SE (Fisher vs statsmodels)": 1e-8, "HL (numpy vs statsmodels)": 1e-8, "δ (Newton vs brentq)": 1e-10}
        _t = pd.DataFrame({"cálculo": list(_max), "máx |discrepancia|": list(_max.values()),
                           "tolerancia declarada": [_tol[k] for k in _max]})
        _t["pasa"] = _t["máx |discrepancia|"] <= _t["tolerancia declarada"]
        return _t

    tabla_paridad = arnes_paridad()
    tabla_paridad
    return arnes_paridad, tabla_paridad


@app.cell
def _(mo, tabla_paridad):
    mo.md(f"""
    **Lectura.** {int(tabla_paridad['pasa'].sum())} de {len(tabla_paridad)} cálculos pasan en las 12
    carteras. Las tolerancias no son arbitrarias: $10^{{-12}}$ donde ambos lados hacen la misma
    aritmética (conteos, rangos), $10^{{-8}}$ donde hay un criterio de parada iterativo (Newton) y la
    diferencia es el `tol` de cada optimizador. Un test con tolerancia $10^{{-3}}$ «pasa» con la
    trampa `C=1` en carteras grandes: la tolerancia es parte del contrato y se justifica.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### E2 · Portabilidad: del núcleo numpy a un motor sin Python

    Un scorecard congelado es una tabla: cortes, WoE, β, puntos. Si el núcleo está en numpy con
    cortes explícitos, exportarlo a SQL (o Java, o una regla de un motor de decisión) es
    mecánico, y el test de paridad puede **emular la semántica del motor** en numpy. Abajo, la
    variable `uso_linea_prom_12m` del modelo de trabajo, con la convención `(a, b]` traducida a
    `<=`.
    """)
    return


@app.cell
def _(FACTOR, OFFSET, VARIABLES, beta_mle, cortes_uso, dev, np, woe_np_uso):
    _nvar = len(VARIABLES)
    _j = VARIABLES.index("uso_linea_prom_12m") + 1
    puntos_uso = -(beta_mle[_j] * woe_np_uso + beta_mle[0] / _nvar) * FACTOR + OFFSET / _nvar

    def sql_case(variable, cortes, puntos, decimales=4):
        """CASE de SQL para cortes (a, b] → `<=`. La rama NULL va primero y es explícita."""
        _l = [f"CASE", f"  WHEN {variable} IS NULL THEN NULL  -- política de missing: declarar"]
        for _c, _p in zip(cortes[1:-1], puntos[:-1]):
            _l.append(f"  WHEN {variable} <= {float(_c)!r} THEN {_p:.{decimales}f}")   # repr de float: ida y vuelta exacta
        _l.append(f"  ELSE {puntos[-1]:.{decimales}f}")
        _l.append(f"END AS pts_{variable}")
        return "\n".join(_l)

    def emular_sql(x, cortes, puntos):
        """Semántica del CASE: la primera condición verdadera gana."""
        _conds = [x <= c for c in cortes[1:-1]]
        return np.select(_conds, puntos[:-1], default=puntos[-1])

    _x = dev["uso_linea_prom_12m"].to_numpy()
    _lookup = puntos_uso[np.searchsorted(cortes_uso[1:-1], _x, side="left")]
    _sql = emular_sql(_x, cortes_uso, puntos_uso)
    _sql_mal = np.select([_x < c for c in cortes_uso[1:-1]], puntos_uso[:-1], default=puntos_uso[-1])
    assert np.array_equal(_lookup, _sql)
    filas_distintas_sql_mal = int(np.sum(_lookup != _sql_mal))
    # casos de borde sintéticos: x = cada corte exacto
    _bordes = cortes_uso[1:-1]
    _lk_b = puntos_uso[np.searchsorted(cortes_uso[1:-1], _bordes, side="left")]
    bordes_distintos_sql_mal = int(np.sum(_lk_b != np.select([_bordes < c for c in cortes_uso[1:-1]],
                                                            puntos_uso[:-1], default=puntos_uso[-1])))
    assert np.array_equal(_lk_b, emular_sql(_bordes, cortes_uso, puntos_uso))
    # serialización con 10 dígitos significativos: ¿el corte vuelve idéntico?
    cortes_no_ida_vuelta_10g = int(sum(float(f"{c:.10g}") != c for c in cortes_uso[1:-1]))
    assert repr(np.float64(0.2)) == "np.float64(0.2)" and repr(float(np.float64(0.2))) == "0.2"   # numpy ≥ 2
    texto_sql = sql_case("uso_linea_prom_12m", cortes_uso, puntos_uso)
    return (
        bordes_distintos_sql_mal,
        cortes_no_ida_vuelta_10g,
        emular_sql,
        filas_distintas_sql_mal,
        puntos_uso,
        sql_case,
        texto_sql,
    )


@app.cell
def _(bordes_distintos_sql_mal, cortes_no_ida_vuelta_10g, cortes_uso, filas_distintas_sql_mal, mo, texto_sql):
    mo.vstack([mo.md("```sql\n" + texto_sql + "\n```"), mo.md(f"""
    **Lectura.** El lookup numpy (`searchsorted`) y la emulación del `CASE` coinciden fila a fila
    (`assert`). Si el ingeniero escribe `<` en vez de `<=`, en esta variable continua cambian
    **{filas_distintas_sql_mal}** filas de DEV: los cortes son cuantiles interpolados que no coinciden con
    ningún valor observado, así que **un test de paridad solo sobre datos reales no detecta el error**.
    Sobre los {len(cortes_uso) - 2} casos de borde sintéticos (x = cada corte exacto) cambian
    **{bordes_distintos_sql_mal}** de {len(cortes_uso) - 2}. Regla: el test de paridad incluye siempre x = corte, corte ± ε,
    NULL y códigos especiales. Segundo detalle: los cortes se escriben con `repr(float(c))` (el
    mínimo de dígitos que garantiza ida y vuelta exacta); con `:.10g`, {cortes_no_ida_vuelta_10g} de {len(cortes_uso) - 2}
    cortes no vuelven al mismo float al leerse. Y ojo con numpy 2: `repr(np.float64(0.2))` es
    `'np.float64(0.2)'`, no `'0.2'`; hay que convertir a `float` antes de serializar. Con variables
    enteras (C3) el error de `<` vs `<=` mueve miles de filas.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## F. Recomendación por capa

    | Capa | Qué se usa | Por qué |
    |---|---|---|
    | Exploración | pandas + optbinning + sklearn, sin restricciones | velocidad de iteración; el costo de un default mal entendido es bajo |
    | Desarrollo | statsmodels (`Logit`/`GLM`) para el modelo; optbinning como propuesta de cortes que se **congelan** en un artefacto propio | inferencia completa para el expediente; binning óptimo como insumo, no como caja negra |
    | Validación | numpy desde cero (especificación ejecutable) **y** librería (oráculo): test de paridad | independencia real: dos implementaciones que coinciden por construcción distinta |
    | Producción | numpy puro o el motor destino (SQL/Java) sobre la tabla congelada; cero dependencias estadísticas | superficie mínima, portabilidad, paridad demostrable |
    | Monitoreo | numpy vectorizado (PSI, CSI, AUC con `bincount`, bootstrap en bloques) + scipy para colas de distribuciones | se repite miles de veces; las colas exactas no se reimplementan |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Checks del módulo

    Si falla un `assert`, el notebook falla. Cubren las coincidencias numpy vs librería, las
    invariantes teóricas y las convenciones de las librerías instaladas.
    """)
    return


@app.cell
def _(
    OPTB_OK,
    aucs_vec,
    auc_np_dev,
    auc_sk_dev,
    bench_auc,
    bench_boot,
    bench_logit,
    bench_woe,
    beta_mle,
    beta_sk_defecto,
    beta_sk_inf,
    beta_sm,
    bootstrap_auc_bucle,
    bordes_distintos_sql_mal,
    conv_optb,
    cortes_uso,
    delta_aprox,
    delta_brentq,
    delta_glm,
    delta_np,
    dif_pred_escala,
    dif_puntos_32,
    err_aprox_c1_1000,
    errores_solvers,
    filas_cambian_borde,
    grilla_solvers,
    idx_boot,
    kappa_woe,
    log_prob_cero,
    mo,
    np,
    pd_oot,
    pesos_ingenuos,
    pesos_lse,
    prob_cero_ingenua,
    resumen_c1,
    separacion,
    special,
    tabla_convenciones,
    tabla_paridad,
    woe_curso_uso,
    woe_np_uso,
    woe_pd_uso,
    y_oot,
):
    _checks = []
    # --- estabilidad numérica
    with np.errstate(all="ignore"):
        assert not np.isfinite(np.log(1 - 1 / (1 + np.exp(-40.0))))
    assert np.isclose(-np.logaddexp(0, 40.0), special.log_expit(-40.0))
    _checks.append("log(1−σ(40)) ingenuo = −∞; logaddexp = log_expit")
    assert np.log(1 - 1e-17) == 0.0 and np.log1p(-1e-17) == -1e-17
    _checks.append("log1p conserva p = 1e-17; log(1−p) lo pierde")
    assert 1 - special.expit(39.0) == 0.0 and special.expit(-39.0) > 1e-17
    _checks.append("PD = 1 − P(bueno) colapsa a 0; σ(−z) no")
    assert prob_cero_ingenua < 1e-300 and np.isfinite(log_prob_cero)
    assert np.all(np.isnan(pesos_ingenuos)) and np.isclose(pesos_lse.sum(), 1) and np.argmax(pesos_lse) == 1
    _checks.append("underflow de productos y logsumexp en pesos de escenarios")
    _e = grilla_solvers[8]
    assert _e["QR de X"] < 1e-6 and _e["solve(XᵀX, Xᵀy)"] > 1e-3
    assert errores_solvers(2)["solve(XᵀX, Xᵀy)"] < 1e-10
    _checks.append("κ = 1e8: QR exacto, ecuaciones normales sin dígitos")
    assert kappa_woe < 100 and dif_pred_escala < 1e-9
    _checks.append("diseño WoE bien condicionado; reescalar la renta no cambia las predicciones")
    assert dif_puntos_32 < 0.1
    _checks.append("float32 cambia el score < 0,1 puntos")
    # --- paridad numpy vs librerías
    assert np.allclose(woe_np_uso, woe_pd_uso) and np.allclose(woe_np_uso, woe_curso_uso)
    assert np.isclose(auc_np_dev, auc_sk_dev, atol=1e-12)
    assert np.allclose(beta_sm, beta_mle, atol=1e-8) and np.allclose(beta_sk_inf, beta_mle, atol=1e-4)
    assert np.max(np.abs(beta_sk_defecto - beta_mle)) > 1e-3
    _checks.append("WoE, AUC y β: numpy = librería; sklearn C=1 ≠ MLE")
    assert np.allclose(bootstrap_auc_bucle(y_oot, pd_oot, idx_boot[:20]), aucs_vec[:20], atol=1e-12)
    _checks.append("bootstrap vectorizado = bucle")
    assert abs(delta_glm - delta_brentq) < 1e-7 and abs(delta_np - delta_brentq) < 1e-10
    assert abs(delta_aprox) < abs(delta_brentq)
    _checks.append("δ: GLM offset = brentq = Newton; aproximación subestima")
    assert separacion["glm_converged"] and not separacion["logit_converged"]
    _checks.append("separación: GLM converged=True, Logit converged=False")
    assert resumen_c1["contraccion_norma"] < 1 and resumen_c1["brecha_media_sk"] < 1e-3
    assert err_aprox_c1_1000 < 0.03
    _checks.append("C=1 contrae pendientes, conserva la media; aproximación de 1 paso a < 0,03")
    if OPTB_OK:
        assert conv_optb["woe_sin_bureau_transform"] == 0.0 and conv_optb["woe_optb"] < 0
        _checks.append("optbinning: metric_special=0 y signo WoE del curso")
    assert filas_cambian_borde > 0 and bordes_distintos_sql_mal == len(cortes_uso) - 2
    _checks.append("(a,b] vs [a,b) cambia filas en variable entera y todos los casos de borde en SQL")
    assert len(tabla_convenciones) >= 12
    assert tabla_paridad["pasa"].all()
    _checks.append("arnés de paridad: todas las tolerancias")
    # --- rendimiento: solo invariantes robustas (orden de magnitud), no milisegundos
    assert bench_woe["segundos"].iloc[0] < bench_woe["segundos"].iloc[2]
    assert bench_boot["segundos"].iloc[2] < bench_boot["segundos"].iloc[1]
    assert bench_logit["segundos"].iloc[0] < bench_logit["segundos"].iloc[3]
    assert bench_auc["segundos"].iloc[0] < bench_auc["segundos"].iloc[2]
    _checks.append("benchmarks: numpy vectorizado más rápido en llamadas medianas")
    mo.md("**Checks OK** (" + str(len(_checks)) + "):\n\n" + "\n".join(f"- ✓ {c}" for c in _checks))
    return


if __name__ == "__main__":
    app.run()

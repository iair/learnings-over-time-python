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
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    return mo, multipletests, plt, sm, stats


@app.cell
def _(mo):
    mo.md(r"""
    # M08 · Estabilidad poblacional: PSI, CSI y el orden del embudo

    **Serie 2 «Del embudo al gobierno»** · profundiza clase 3 (PSI primero, 108 → 94, PSI no medible)
    y clase 5 (PSI del score sobre 8 bandas, CSI como canario).

    Qué demuestra este notebook, con verdad conocida:

    1. El PSI es la divergencia de Jeffreys, $\mathrm{KL}(a\|e)+\mathrm{KL}(e\|a)$: implementación numpy
       y `scipy.stats.entropy` coinciden a precisión de máquina; y $\text{PSI}/(1/n_e+1/n_a)$ es, a segundo
       orden, el χ² de homogeneidad de `chi2_contingency`.
    2. Simulador de drift (media, varianza, cola, mezcla categórica): PSI, KS, Jensen-Shannon, Hellinger y
       Wasserstein lado a lado, **comparados a igual tasa de falsa alarma** (potencia), no por su valor crudo.
    3. Distribución nula del PSI por simulación vs aproximación $\chi^2_{B-1}$ escalada: el umbral 0,10
       significa cosas distintas con $n=300$ o $n=30.000$.
    4. Sensibilidad a bins, $\varepsilon$ y masa de ceros (por qué el curso obtiene `NaN` y cómo medirlo).
    5. Taxonomía de cambio (covariate / prior / concept shift): lo que el PSI ve y lo que no.
    6. Caso «Banco Sintético»: CSI de todas las variables DEV→OOT/TTD, el canal que deriva, PSI del score por
       deciles/bandas, descomposición **direccional** en puntos y DEV congelado vs ventana móvil.
    7. Cancelación: dos variables que se mueven en sentido opuesto con PSI del score ≈ 0.

    Convenciones del curso: target 1 = malo; $\text{WoE}=\ln(\%\text{buenos}/\%\text{malos})$; bins y
    puntos se ajustan en DEV y se aplican al resto; PDO 20, 600 a odds 50:1; umbrales PSI/CSI 0,10 / 0,25.
    """)
    return


@app.cell
def _():
    # =====================================================================
    # Código común de la serie (pegado VERBATIM desde _spec/comun.py)
    # =====================================================================
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
    ## 1. Las piezas desde cero (numpy) y su equivalente de librería

    Todas las medidas de esta sección comparan dos vectores de proporciones por bin, $e$ (esperado, DEV)
    y $a$ (actual), salvo KS y Wasserstein, que trabajan sobre los valores crudos.

    | Medida | Fórmula (numpy) | Librería |
    |---|---|---|
    | PSI | $\sum_b (a_b-e_b)\ln(a_b/e_b)$ | `scipy.stats.entropy(a,e)+entropy(e,a)` |
    | KL | $\sum_b a_b\ln(a_b/e_b)$ | `scipy.special.rel_entr(a,e).sum()` |
    | Jensen-Shannon | $\tfrac12\mathrm{KL}(a\|m)+\tfrac12\mathrm{KL}(e\|m)$, $m=\tfrac{a+e}{2}$ | `scipy.spatial.distance.jensenshannon(a,e)**2` (¡devuelve la raíz!) |
    | Hellinger | $\sqrt{\tfrac12\sum_b(\sqrt{a_b}-\sqrt{e_b})^2}$ | `euclidean(√a, √e)/√2` |
    | χ² homogeneidad | $\sum_{s,b}(O_{sb}-E_{sb})^2/E_{sb}$ | `scipy.stats.chi2_contingency` |
    | KS 2 muestras | $\max_x |F_e(x)-F_a(x)|$ | `scipy.stats.ks_2samp` |
    | Wasserstein-1 | $\int |F_e(x)-F_a(x)|\,dx$ | `scipy.stats.wasserstein_distance` |

    Ojo con dos convenciones: (i) el curso **suma** $\varepsilon=10^{-4}$ a las proporciones sin renormalizar,
    mientras `scipy.stats.entropy` y `jensenshannon` **renormalizan** sus entradas; (ii) los bins numpy aquí
    son cerrados a la derecha, $(c_{k-1},c_k]$, igual que `pd.cut` en el `psi()` del curso (`np.histogram`
    los cierra a la izquierda: con variables discretas eso cambia los conteos).
    """)
    return


@app.cell
def _(binear, np, pd, stats):
    # ---------------- (1) numpy desde cero ----------------
    def cortes_cuantil(ref, B):
        """Cortes por cuantiles de la referencia (DEV), con ±inf en los extremos.
        np.unique colapsa cortes repetidos: con masas puntuales quedan MENOS de B bins."""
        _c = np.unique(np.nanquantile(np.asarray(ref, float), np.linspace(0, 1, B + 1)))
        _c[0], _c[-1] = -np.inf, np.inf
        return _c

    def conteos(x, cortes):
        """Conteos por bin, intervalos (c_{k-1}, c_k] como pd.cut."""
        _x = np.asarray(x, float)
        _idx = np.searchsorted(cortes[1:-1], _x, side="left")
        return np.bincount(_idx, minlength=len(cortes) - 1)

    def proporciones(x, cortes):
        _k = conteos(x, cortes)
        return _k / _k.sum()

    def aportes_psi(e, a, eps=0.0):
        """Aporte por bin (a-e)·ln(a/e). eps se SUMA a las proporciones (convención del curso)."""
        _e = np.asarray(e, float) + eps
        _a = np.asarray(a, float) + eps
        return (_a - _e) * np.log(_a / _e)

    def psi_np(e, a, eps=0.0):
        return float(np.sum(aportes_psi(e, a, eps)))

    def kl_np(p, q):
        """KL(p||q) con la convención 0·ln0 = 0."""
        _p, _q = np.asarray(p, float), np.asarray(q, float)
        _t = np.where(_p > 0, _p * np.log(np.where(_p > 0, _p, 1.0) / _q), 0.0)
        return float(_t.sum())

    def js_np(p, q):
        _m = (np.asarray(p, float) + np.asarray(q, float)) / 2
        return 0.5 * kl_np(p, _m) + 0.5 * kl_np(q, _m)

    def hellinger_np(p, q):
        return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

    def chi2_homog_np(k_e, k_a):
        """χ² de homogeneidad 2×B desde conteos; descarta bins vacíos en ambas muestras."""
        _O = np.vstack([k_e, k_a]).astype(float)
        _O = _O[:, _O.sum(axis=0) > 0]
        _E = _O.sum(axis=1, keepdims=True) * _O.sum(axis=0, keepdims=True) / _O.sum()
        _x2 = float(((_O - _E) ** 2 / _E).sum())
        _gl = _O.shape[1] - 1
        return _x2, _gl, float(stats.chi2.sf(_x2, _gl))

    def ks_np(xe, xa):
        """KS de dos muestras: sup |F_e - F_a| evaluado en todos los puntos observados."""
        _se, _sa = np.sort(xe), np.sort(xa)
        _g = np.concatenate([_se, _sa])
        _Fe = np.searchsorted(_se, _g, side="right") / len(_se)
        _Fa = np.searchsorted(_sa, _g, side="right") / len(_sa)
        return float(np.max(np.abs(_Fe - _Fa)))

    def wasserstein_np(xe, xa):
        """W1 = ∫|F_e - F_a| dx, integrando la diferencia de CDF empíricas entre puntos consecutivos."""
        _se, _sa = np.sort(xe), np.sort(xa)
        _g = np.sort(np.concatenate([_se, _sa]))
        _d = np.diff(_g)
        _Fe = np.searchsorted(_se, _g[:-1], side="right") / len(_se)
        _Fa = np.searchsorted(_sa, _g[:-1], side="right") / len(_sa)
        return float(np.sum(np.abs(_Fe - _Fa) * _d))

    def psi_curso(esperado, actual, bins=10):
        """Réplica EXACTA del psi() de la clase 3: deciles de DEV, +1e-4, NaN si colapsan."""
        esperado, actual = pd.Series(esperado), pd.Series(actual)
        if not pd.api.types.is_numeric_dtype(esperado):
            _e = esperado.fillna("MISSING").astype(str)
            _a = actual.fillna("MISSING").astype(str)
        else:
            _cortes = np.unique(np.nanquantile(esperado.dropna(), np.linspace(0, 1, bins + 1)))
            if len(_cortes) < 3:
                return np.nan
            _cortes[0], _cortes[-1] = -np.inf, np.inf
            _e = pd.cut(esperado, _cortes).astype(str).where(esperado.notna(), "MISSING")
            _a = pd.cut(actual, _cortes).astype(str).where(actual.notna(), "MISSING")
        _cats = sorted(set(_e) | set(_a))
        _pe = _e.value_counts(normalize=True).reindex(_cats).fillna(0) + 1e-4
        _pa = _a.value_counts(normalize=True).reindex(_cats).fillna(0) + 1e-4
        return float(((_pa - _pe) * np.log(_pa / _pe)).sum())

    def tabla_psi_binear(xe, xa, bins=5, eps=1e-4):
        """CSI del curso (clase 5): bins de binear() ajustados en DEV (bin de moda + MISSING
        incluidos). Devuelve tabla por bin con % esperado, % actual y aporte."""
        _le, _orden = binear(xe, bins)
        _la, _ = binear(xa, bins, ref=xe)
        _cats = list(dict.fromkeys((_orden or []) + sorted(set(_le) | set(_la))))
        _cats = [c for c in _cats if c in set(_le) | set(_la)]
        _e = _le.value_counts(normalize=True).reindex(_cats).fillna(0).to_numpy()
        _a = _la.value_counts(normalize=True).reindex(_cats).fillna(0).to_numpy()
        return pd.DataFrame({"bin": _cats, "pct_esp": _e, "pct_act": _a,
                             "aporte": aportes_psi(_e, _a, eps)})

    def csi_curso(xe, xa, bins=5, eps=1e-4):
        return float(tabla_psi_binear(xe, xa, bins, eps)["aporte"].sum())

    def c_n(n_e, n_a):
        """Factor de escala de la nula: 1/n_e + 1/n_a (n_e = inf si DEV se trata como conocida)."""
        return 1.0 / n_e + 1.0 / n_a

    def umbral_psi(n_e, n_a, B, alfa=0.05):
        """Crítico del PSI bajo H0 (sin drift): c·χ²_{1-α, B-1}  (Yurdakul y Naranjo 2020)."""
        return c_n(n_e, n_a) * float(stats.chi2.ppf(1 - alfa, B - 1))

    def p_psi(psi, n_e, n_a, B):
        """p-valor aproximado del PSI: P(χ²_{B-1} > PSI / c)."""
        return float(stats.chi2.sf(psi / c_n(n_e, n_a), B - 1))

    def semaforo(v, amarillo=0.10, rojo=0.25):
        if v != v:
            return "⚪ no medible"
        return "🔴" if v > rojo else ("🟡" if v > amarillo else "🟢")

    return (aportes_psi, c_n, chi2_homog_np, conteos, cortes_cuantil, csi_curso,
            hellinger_np, js_np, kl_np, ks_np, p_psi, proporciones, psi_curso, psi_np,
            semaforo, tabla_psi_binear, umbral_psi, wasserstein_np)


@app.cell
def _(np, stats):
    # ---------------- (2) librerías estándar ----------------
    from scipy.stats import entropy, chi2_contingency, ks_2samp, wasserstein_distance
    from scipy.special import rel_entr
    from scipy.spatial.distance import jensenshannon, euclidean

    def psi_scipy(e, a):
        """PSI = KL(a||e) + KL(e||a). OJO: entropy() renormaliza sus entradas a suma 1."""
        return float(entropy(a, e) + entropy(e, a))

    def js_scipy(p, q):
        """jensenshannon devuelve la DISTANCIA (raíz de la divergencia), base e por defecto."""
        return float(jensenshannon(p, q) ** 2)

    def hellinger_scipy(p, q):
        return float(euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2))

    def chi2_scipy(k_e, k_a):
        _O = np.vstack([k_e, k_a])
        _O = _O[:, _O.sum(axis=0) > 0]
        _r = chi2_contingency(_O, correction=False)
        return float(_r.statistic), int(_r.dof), float(_r.pvalue)

    def ks_scipy(xe, xa):
        _r = ks_2samp(xe, xa)
        return float(_r.statistic), float(_r.pvalue)

    def w_scipy(xe, xa):
        return float(wasserstein_distance(xe, xa))

    def kl_scipy(p, q):
        return float(rel_entr(p, q).sum())

    _ = stats  # (scipy.stats también se usa para la cola de la χ²)
    return chi2_scipy, hellinger_scipy, js_scipy, kl_scipy, ks_scipy, psi_scipy, w_scipy


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.1 El ancla: el PSI del score del Banco Austral (clase 5)

    Tomamos la tabla de 8 bandas de la clase 5 (DEV $n=3.322$, OOT $n=2.004$, TTD $n=8.585$) y
    reproducimos el PSI con las dos implementaciones. Además convertimos las proporciones a conteos para
    obtener el χ² de homogeneidad y verificar la relación $\text{PSI}/c \approx X^2$ con $c=1/n_e+1/n_a$.
    """)
    return


@app.cell
def _(aportes_psi, c_n, chi2_homog_np, chi2_scipy, kl_np, kl_scipy, np, p_psi, pd, psi_np, psi_scipy, umbral_psi):
    bandas_austral = pd.DataFrame({
        "banda": ["A1", "A2", "B1", "B2", "C1", "C2", "D", "E"],
        "dev": [21.43, 9.75, 11.65, 12.22, 12.40, 10.87, 9.42, 12.25],
        "oot": [21.56, 9.93, 12.18, 11.73, 11.93, 10.98, 8.58, 13.12],
        "ttd": [18.43, 8.95, 10.75, 12.16, 12.31, 12.03, 10.90, 14.47],
    })
    N_AUSTRAL = {"DEV": 3322, "HO": 1397, "OOT": 2004, "TTD": 8585}
    _e = bandas_austral["dev"].to_numpy() / 100
    _e = _e / _e.sum()                               # la tabla suma 99,99%: renormalizamos
    _filas = []
    for _m in ["oot", "ttd"]:
        _a = bandas_austral[_m].to_numpy() / 100
        _a = _a / _a.sum()
        _n_a = N_AUSTRAL[_m.upper()]
        _k_e = np.round(_e * N_AUSTRAL["DEV"]).astype(int)
        _k_a = np.round(_a * _n_a).astype(int)
        _x2, _gl, _p = chi2_homog_np(_k_e, _k_a)
        _x2s, _gls, _ps = chi2_scipy(_k_e, _k_a)
        _psi0 = psi_np(_e, _a)
        _filas.append({
            "comparación": f"DEV→{_m.upper()}",
            "PSI curso (ε=1e-4)": psi_np(_e, _a, 1e-4),
            "PSI numpy (ε=0)": _psi0,
            "PSI scipy entropy": psi_scipy(_e, _a),
            "KL(a‖e)": kl_np(_a, _e), "KL(e‖a)": kl_np(_e, _a),
            "KL(a‖e) scipy": kl_scipy(_a, _e),
            "c = 1/n_e+1/n_a": c_n(N_AUSTRAL["DEV"], _n_a),
            "PSI/c": _psi0 / c_n(N_AUSTRAL["DEV"], _n_a),
            "X² numpy": _x2, "X² scipy": _x2s, "gl": _gl,
            "p (PSI/c ~ χ²)": p_psi(_psi0, N_AUSTRAL["DEV"], _n_a, 8),
            "p (X² scipy)": _ps,
            "crítico 95%": umbral_psi(N_AUSTRAL["DEV"], _n_a, 8),
        })
    anclas_austral = pd.DataFrame(_filas).set_index("comparación")
    _a_ttd = bandas_austral["ttd"].to_numpy() / 100
    bandas_austral["aporte_ttd"] = aportes_psi(bandas_austral["dev"] / 100, _a_ttd, 1e-4)
    bandas_austral["kl_a_e_ttd"] = _a_ttd * np.log(_a_ttd / (bandas_austral["dev"] / 100))
    anclas_austral.T.round(5)
    return N_AUSTRAL, anclas_austral, bandas_austral


@app.cell
def _(anclas_austral, bandas_austral, mo):
    _t = anclas_austral.loc["DEV→TTD"]
    _o = anclas_austral.loc["DEV→OOT"]
    mo.vstack([
        mo.md(rf"""
    **Lectura.** Se reproduce el 0,0130 de la clase (DEV→TTD) y el 0,0020 (DEV→OOT). La versión
    numpy y `entropy(a,e)+entropy(e,a)` coinciden a precisión de máquina: el PSI **es** la divergencia
    de Jeffreys. La diferencia entre ε=1e-4 y ε=0 es de orden $10^{{-5}}$ aquí (no hay bins vacíos).

    Lo que la clase no dijo: con $n_e=3.322$ y $n_a=8.585$, el ruido de muestreo por sí solo produce un
    PSI esperado de $7c \approx$ {7 * _t['c = 1/n_e+1/n_a']:.4f} y un crítico al 95% de
    {_t['crítico 95%']:.4f}. El 0,013 «verde» equivale a $X^2\approx$ {_t['PSI/c']:.1f} con 7 gl,
    **p ≈ {_t['p (PSI/c ~ χ²)']:.1e}**: la población SÍ se movió (A1 se vacía, E se llena), aunque no sea
    material. En cambio DEV→OOT da $X^2\approx$ {_o['PSI/c']:.1f}, p ≈ {_o['p (PSI/c ~ χ²)']:.2f}:
    indistinguible del ruido. `PSI/c` y el $X^2$ exacto difieren en
    {abs(_t['PSI/c'] / _t['X² scipy'] - 1):.1%}: son iguales a segundo orden, no idénticos (sección 2).
    """),
        bandas_austral.round(4),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Geometría local: todas las f-divergencias son la misma para drift chico

    Resultado del documento (§3): con $m_b=(a_b+e_b)/2$,
    $$\text{PSI}=\sum_b\frac{(a_b-e_b)^2}{m_b}+O(\delta^4),\qquad
    \text{JS}\approx\frac{\text{PSI}}{8},\qquad H^2\approx\frac{\text{PSI}}{8}.$$
    Y para dos normales (continuo, sin bins): corrimiento de media $\Delta$ (en DE) da $J=\Delta^2$;
    cambio de escala $\sigma_a=k\sigma_e$ da $J=(k-1/k)^2/2$. El PSI con bins es **menor o igual** que $J$
    (desigualdad de procesamiento de datos: agrupar pierde información).

    Tabla determinista (sin muestreo): bins = deciles poblacionales de $N(0,1)$, proporciones exactas por
    la CDF normal.
    """)
    return


@app.cell
def _(hellinger_np, js_np, np, pd, psi_np, stats):
    def props_normal(B, mu=0.0, sd=1.0):
        """Proporciones EXACTAS de N(mu, sd) en los B cuantiles poblacionales de N(0,1)."""
        _c = stats.norm.ppf(np.linspace(0, 1, B + 1))
        return np.diff(stats.norm.cdf(_c, loc=mu, scale=sd))

    _filas = []
    _e10 = props_normal(10)
    for _tipo, _vals in [("media Δ (DE)", [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]),
                         ("escala k", [1.05, 1.1, 1.2, 1.3, 1.5, 2.0])]:
        for _v in _vals:
            _a = props_normal(10, mu=_v) if _tipo.startswith("media") else props_normal(10, sd=_v)
            _m = (_a + _e10) / 2
            _psi = psi_np(_e10, _a)
            _J = _v**2 if _tipo.startswith("media") else (_v - 1 / _v) ** 2 / 2
            _filas.append({"cambio": _tipo, "valor": _v, "PSI deciles": _psi,
                           "Σδ²/m": float(np.sum((_a - _e10) ** 2 / _m)),
                           "8·JS": 8 * js_np(_a, _e10), "8·H²": 8 * hellinger_np(_a, _e10) ** 2,
                           "J continuo": _J, "PSI/J": _psi / _J,
                           "PSI 50 bins": psi_np(props_normal(50), props_normal(50, mu=_v) if _tipo.startswith("media") else props_normal(50, sd=_v))})
    equivalencia_local = pd.DataFrame(_filas)
    equivalencia_local.round(4)
    return equivalencia_local, props_normal


@app.cell
def _(equivalencia_local, mo):
    _m = equivalencia_local[equivalencia_local["cambio"].str.startswith("media")]
    _s = equivalencia_local[equivalencia_local["cambio"].str.startswith("escala")]
    _psi010 = float(_m.loc[_m["valor"] == 0.3, "PSI deciles"].iloc[0])
    mo.md(rf"""
    **Lectura.**

    - Para cambios chicos, PSI, $\sum\delta^2/m$, $8\,\text{{JS}}$ y $8H^2$ son prácticamente el mismo número:
      elegir entre ellas es cosmético **mientras el drift sea chico y no haya bins vacíos**. Divergen con
      drift grande (Δ=1: PSI {float(_m['PSI deciles'].iloc[-1]):.3f} vs 8·JS {float(_m['8·JS'].iloc[-1]):.3f}),
      porque JS y Hellinger están acotadas y el PSI no.
    - Calibración de los umbrales: un corrimiento de media de 0,3 DE da PSI por deciles ≈ {_psi010:.3f};
      el 0,10 del curso equivale a ≈ 0,33 DE y el 0,25 a ≈ 0,5 DE. Eso es lo que «significan» en unidades
      de la variable.
    - Deciles capturan casi todo el corrimiento de **media** (PSI/J ≈ {float(_m['PSI/J'].iloc[2]):.2f}) pero
      solo una parte del cambio de **escala** (PSI/J ≈ {float(_s['PSI/J'].iloc[3]):.2f} con k=1,3): la
      información del cambio de varianza vive en las colas, que los deciles extremos agrupan. Con 50 bins
      se recupera más (columna «PSI 50 bins»), a costa de más ruido de muestreo (§4).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Simulador de drift: cinco métricas lado a lado

    Elige el tipo de cambio, su magnitud, el tamaño muestral y los bins. La tabla de arriba muestra el valor
    de cada métrica en **una** realización. La de abajo es la comparación honesta: para cada métrica se
    estima por simulación su crítico al 5% bajo H0 (sin drift, mismo $n$ y $B$) y luego la **potencia**
    (fracción de réplicas con drift que superan ese crítico). Comparar valores crudos entre métricas no
    tiene sentido: viven en escalas distintas.

    Tipos: **media** (Δ = magnitud, en DE) · **varianza** ($\sigma_a = 1+$ magnitud) · **cola** (una
    fracción $\pi = 0{,}10\times$magnitud se va a $N(3{,}5;\,0{,}5)$) · **categórica** (la mezcla del
    `canal` del generador se mueve una fracción «magnitud» del camino de [45, 25, 15, 15]% a [25, 25, 35, 15]%).
    """)
    return


@app.cell
def _(mo):
    tipo_drift = mo.ui.dropdown(["media", "varianza", "cola", "categórica"], value="media",
                                label="Tipo de cambio")
    magnitud = mo.ui.slider(0.0, 1.0, step=0.05, value=0.1, label="Magnitud")
    n_sim = mo.ui.dropdown(["300", "1000", "3000", "10000", "30000"], value="1000",
                           label="n por muestra")
    bins_sim = mo.ui.slider(4, 40, step=1, value=10, label="Bins (cuantiles de DEV)")
    mo.hstack([tipo_drift, magnitud, n_sim, bins_sim], justify="start")
    return bins_sim, magnitud, n_sim, tipo_drift


@app.cell
def _(chi2_homog_np, conteos, cortes_cuantil, hellinger_np, js_np, ks_np, np, psi_np, wasserstein_np):
    P_CANAL_BASE = np.array([0.45, 0.25, 0.15, 0.15])
    P_CANAL_NUEVO = np.array([0.25, 0.25, 0.35, 0.15])

    def muestra_drift(tipo, mag, n, rng, drift=True):
        """Devuelve (x_esperado, x_actual). Con drift=False ambas vienen de la misma distribución."""
        if tipo == "categórica":
            _xe = rng.choice(4, n, p=P_CANAL_BASE).astype(float)
            _p = P_CANAL_BASE + (mag if drift else 0.0) * (P_CANAL_NUEVO - P_CANAL_BASE)
            return _xe, rng.choice(4, n, p=_p).astype(float)
        _xe = rng.standard_normal(n)
        if not drift or mag == 0:
            return _xe, rng.standard_normal(n)
        if tipo == "media":
            return _xe, rng.normal(mag, 1.0, n)
        if tipo == "varianza":
            return _xe, rng.normal(0.0, 1.0 + mag, n)
        _xa = rng.standard_normal(n)                     # cola: contaminación
        _cola = rng.random(n) < 0.10 * mag
        _xa[_cola] = rng.normal(3.5, 0.5, _cola.sum())
        return _xe, _xa

    def metricas_drift(xe, xa, B, categorica, eps=1e-4):
        """Las 5 métricas + χ²; bins = cuantiles de DEV (o niveles si es categórica)."""
        _c = np.array([-np.inf, 0.5, 1.5, 2.5, np.inf]) if categorica else cortes_cuantil(xe, B)
        _ke, _ka = conteos(xe, _c), conteos(xa, _c)
        _e, _a = _ke / _ke.sum(), _ka / _ka.sum()
        _x2, _gl, _p = chi2_homog_np(_ke, _ka)
        return {"PSI": psi_np(_e, _a, eps), "JS": js_np(_a, _e), "Hellinger": hellinger_np(_a, _e),
                "χ² homog.": _x2,
                "KS": np.nan if categorica else ks_np(xe, xa),
                "Wasserstein": np.nan if categorica else wasserstein_np(xe, xa),
                "_gl": _gl, "_p_chi2": _p}

    return P_CANAL_BASE, P_CANAL_NUEVO, metricas_drift, muestra_drift


@app.cell
def _(bins_sim, c_n, magnitud, metricas_drift, muestra_drift, n_sim, np, p_psi, pd, stats, tipo_drift, umbral_psi):
    _n, _B, _mag, _tipo = int(n_sim.value), int(bins_sim.value), float(magnitud.value), tipo_drift.value
    _cat = _tipo == "categórica"
    _Bef = 4 if _cat else _B
    _rng = np.random.default_rng(8)
    xe_demo, xa_demo = muestra_drift(_tipo, _mag, _n, _rng)
    _m1 = metricas_drift(xe_demo, xa_demo, _B, _cat)
    _R = 200
    _nombres = ["PSI", "JS", "Hellinger", "χ² homog.", "KS", "Wasserstein"]
    _H0 = {k: [] for k in _nombres}
    _H1 = {k: [] for k in _nombres}
    for _r in range(_R):
        for _dic, _drift in [(_H0, False), (_H1, True)]:
            _x_e, _x_a = muestra_drift(_tipo, _mag, _n, _rng, drift=_drift)
            _mm = metricas_drift(_x_e, _x_a, _B, _cat)
            for _k in _nombres:
                _dic[_k].append(_mm[_k])
    _filas = []
    for _k in _nombres:
        _h0, _h1 = np.array(_H0[_k]), np.array(_H1[_k])
        if np.all(np.isnan(_h0)):
            _filas.append({"métrica": _k, "valor (1 réplica)": np.nan, "crítico 95% simulado": np.nan,
                           "potencia (α=5%)": np.nan})
            continue
        _crit = float(np.quantile(_h0, 0.95))
        _filas.append({"métrica": _k, "valor (1 réplica)": _m1[_k], "crítico 95% simulado": _crit,
                       "potencia (α=5%)": float(np.mean(_h1 > _crit))})
    tabla_simulador = pd.DataFrame(_filas).set_index("métrica")
    potencia_umbral_010 = float(np.mean(np.array(_H1["PSI"]) > 0.10))
    falsa_alarma_010 = float(np.mean(np.array(_H0["PSI"]) > 0.10))
    info_simulador = {
        "n": _n, "B": _Bef, "psi": _m1["PSI"], "p_psi": p_psi(_m1["PSI"], _n, _n, _Bef),
        "crit_chi2": umbral_psi(_n, _n, _Bef), "E_nulo": (_Bef - 1) * c_n(_n, _n),
        "pot_010": potencia_umbral_010, "fa_010": falsa_alarma_010, "tipo": _tipo, "mag": _mag,
        "crit_sim_psi": float(tabla_simulador.loc["PSI", "crítico 95% simulado"]),
        "ks_p": float(stats.ks_2samp(xe_demo, xa_demo).pvalue) if not _cat else np.nan,
    }
    tabla_simulador.round(4)
    return info_simulador, tabla_simulador, xa_demo, xe_demo


@app.cell
def _(info_simulador, mo, np, plt, tabla_simulador, xa_demo, xe_demo):
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.4))
    if info_simulador["tipo"] == "categórica":
        _lv = np.arange(4)
        _ax[0].bar(_lv - 0.2, [np.mean(xe_demo == k) for k in _lv], 0.4, label="esperado (DEV)")
        _ax[0].bar(_lv + 0.2, [np.mean(xa_demo == k) for k in _lv], 0.4, label="actual")
        _ax[0].set_xticks(_lv, ["sucursal", "web", "app", "f. venta"])
        _ax[0].set_ylabel("proporción")
    else:
        _bins = np.linspace(-4, 5.5, 60)
        _ax[0].hist(xe_demo, _bins, density=True, alpha=0.5, label="esperado (DEV)")
        _ax[0].hist(xa_demo, _bins, density=True, alpha=0.5, label="actual")
        _ax[0].set_xlabel("x (unidades de DE de DEV)")
        _ax[0].set_ylabel("densidad")
    _ax[0].set_title(f"Una réplica · {info_simulador['tipo']} · magnitud {info_simulador['mag']:.2f}")
    _ax[0].legend()
    _pot = tabla_simulador["potencia (α=5%)"].dropna()
    _ax[1].barh(_pot.index, _pot.values, color="tab:gray")
    _ax[1].axvline(0.05, color="tab:red", ls="--", lw=1, label="α = 5%")
    _ax[1].set_xlim(0, 1)
    _ax[1].set_xlabel("potencia a igual falsa alarma (5%)")
    _ax[1].set_title(f"n = {info_simulador['n']:,} por muestra, R = 200")
    _ax[1].legend(loc="lower right")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(rf"""
    **Lectura dinámica.** PSI de esta réplica = {info_simulador['psi']:.4f}; bajo H0 su valor esperado es
    $(B-1)c$ = {info_simulador['E_nulo']:.4f} y el crítico χ² al 95% es {info_simulador['crit_chi2']:.4f}
    (simulado: {info_simulador['crit_sim_psi']:.4f}); p ≈ {info_simulador['p_psi']:.3g}.
    Con la regla fija «PSI > 0,10» la potencia sería {info_simulador['pot_010']:.0%} y la falsa alarma
    {info_simulador['fa_010']:.0%}.

    Qué mirar al mover los controles (con n = 1.000, R = 200, se obtiene aproximadamente):
    (i) **media** 0,10: KS ≈ 0,5 y Wasserstein ≈ 0,55 de potencia contra ≈ 0,27 de PSI/JS/Hellinger/χ² con 10
    bins, y ≈ 0,12 con 30 bins: para un corrimiento de ubicación las métricas que usan el **orden** completo
    ganan, y agregar bins solo agrega grados de libertad de ruido; (ii) **varianza** 0,10: PSI con 10 bins
    (≈ 0,42) supera a KS (≈ 0,19), porque las CDF se cruzan en el centro y el máximo de su diferencia es chico;
    (iii) **cola** 0,40: Wasserstein ≈ 0,97 (pesa la *distancia* a la que se fue la masa), KS ≈ 0,34 y PSI
    sube de ≈ 0,37 con 10 bins a ≈ 0,68 con 30 (más bins resuelven la cola); (iv) **categórica**: KS y
    Wasserstein no aplican (no hay orden) y las cuatro f-divergencias tienen **la misma** potencia.
    Las cuatro columnas binned (PSI, JS, Hellinger, χ²) casi siempre empatan: con drift chico son la misma
    medida (§2). Con $n$ grande todas detectan todo: la potencia deja de discriminar y lo que importa es la
    **materialidad** (§4 y §7).
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Distribución nula del PSI: el umbral depende de $n$

    Sin drift, $\text{PSI}/c \xrightarrow{d} \chi^2_{B-1}$ con $c=1/n_e+1/n_a$ (Yurdakul y Naranjo, 2020;
    derivación en el documento). Simulamos réplicas **con el mismo procedimiento del curso** (cortes por
    deciles de la muestra esperada, aplicados a la actual; ε = 1e-4) y comparamos con la aproximación.
    El esquema «DEV fijo» usa $n_e = 3.322$ (DEV del Banco Austral) y varía el tamaño del lote actual.
    """)
    return


@app.cell
def _(mo):
    bins_nulo = mo.ui.slider(4, 20, step=1, value=10, label="Bins B")
    esquema_nulo = mo.ui.dropdown(["n_e = n_a = n", "DEV fijo n_e = 3.322"], value="n_e = n_a = n",
                                  label="Esquema")
    R_nulo = mo.ui.slider(100, 2000, step=100, value=300, label="Réplicas R")
    mo.hstack([bins_nulo, esquema_nulo, R_nulo], justify="start")
    return R_nulo, bins_nulo, esquema_nulo


@app.cell
def _(conteos, cortes_cuantil, np, psi_np):
    def simular_nulo_psi(n_e, n_a, B, R, rng, eps=1e-4):
        """R valores de PSI sin drift: cortes por cuantiles de la esperada (como el curso)."""
        _v = np.empty(R)
        for _r in range(R):
            _xe = rng.standard_normal(n_e)
            _xa = rng.standard_normal(n_a)
            _c = cortes_cuantil(_xe, B)
            _ke, _ka = conteos(_xe, _c), conteos(_xa, _c)
            _v[_r] = psi_np(_ke / n_e, _ka / n_a, eps)
        return _v

    return (simular_nulo_psi,)


@app.cell
def _(R_nulo, bins_nulo, c_n, esquema_nulo, np, pd, simular_nulo_psi, stats):
    _B, _R = int(bins_nulo.value), int(R_nulo.value)
    _rng = np.random.default_rng(2026)
    TAMANOS_NULO = [100, 300, 1000, 3000, 10000, 30000]
    nulos_psi = {}
    _filas = []
    for _n in TAMANOS_NULO:
        _ne = _n if esquema_nulo.value.startswith("n_e = n_a") else 3322
        _v = simular_nulo_psi(_ne, _n, _B, _R, _rng)
        nulos_psi[_n] = _v
        _c = c_n(_ne, _n)
        _filas.append({
            "n_e": _ne, "n_a": _n,
            "E[PSI] sim": _v.mean(), "E[PSI] ≈ (B-1)c": (_B - 1) * _c,
            "p95 sim": np.quantile(_v, 0.95), "p95 χ²": _c * stats.chi2.ppf(0.95, _B - 1),
            "p99 χ²": _c * stats.chi2.ppf(0.99, _B - 1),
            "P(PSI>0,10) sim": np.mean(_v > 0.10),
            "P(PSI>0,10) χ²": stats.chi2.sf(0.10 / _c, _B - 1),
            "P(PSI>0,25) χ²": stats.chi2.sf(0.25 / _c, _B - 1),
        })
    tabla_nula = pd.DataFrame(_filas).set_index("n_a")
    tabla_nula.round(5)
    return TAMANOS_NULO, nulos_psi, tabla_nula


@app.cell
def _(bins_nulo, c_n, esquema_nulo, mo, np, nulos_psi, plt, stats, tabla_nula):
    _B = int(bins_nulo.value)
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for _ax, _n in zip(_axs, [300, 3000]):
        _ne = _n if esquema_nulo.value.startswith("n_e = n_a") else 3322
        _c = c_n(_ne, _n)
        _v = nulos_psi[_n]
        _ax.hist(_v, bins=40, density=True, alpha=0.6, label="PSI simulado (H0)")
        _xs = np.linspace(1e-6, _v.max() * 1.1, 300)
        _ax.plot(_xs, stats.chi2.pdf(_xs / _c, _B - 1) / _c, "k-", lw=1.2, label=r"$c\cdot\chi^2_{B-1}$")
        _ax.axvline(_c * stats.chi2.ppf(0.95, _B - 1), color="tab:orange", ls="--", label="crítico 95%")
        if _v.max() > 0.08:
            _ax.axvline(0.10, color="tab:red", ls=":", label="umbral 0,10")
        _ax.set_title(f"n_e = {_ne:,}, n_a = {_n:,}, B = {_B}")
        _ax.set_xlabel("PSI")
        _ax.set_ylabel("densidad")
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _t = tabla_nula
    mo.vstack([_fig, mo.md(rf"""
    **Lectura.** Con B = {_B}: a $n=300$ el crítico al 95% es {_t.loc[300, 'p95 χ²']:.3f} (simulado
    {_t.loc[300, 'p95 sim']:.3f}) y la regla «0,10» dispara en {_t.loc[300, 'P(PSI>0,10) sim']:.1%} de
    los meses **sin ningún cambio**; a $n=100$ la cifra sube a {_t.loc[100, 'P(PSI>0,10) sim']:.0%}.
    A $n=30.000$ el crítico es {_t.loc[30000, 'p95 χ²']:.4f}: el 0,10 queda ~{0.10 / _t.loc[30000, 'p95 χ²']:.0f}
    veces por encima, y cualquier cambio estadísticamente detectable pasa como «verde».
    La aproximación χ² funciona bien desde $n\approx 300$ incluso con cortes aleatorios (cuantiles de la
    muestra esperada); con $n$ muy chico aparecen bins vacíos y el ε domina (ver la fila $n=100$).
    Con «DEV fijo», el piso $(B-1)/n_e$ no desaparece aunque el lote actual crezca: el ruido de DEV también
    cuenta.
    """)])
    return


@app.cell
def _(N_AUSTRAL, c_n, pd, stats):
    # Los números del curso (Banco Austral), leídos con la nula. B = bins efectivos (supuesto declarado).
    _casos = [
        ("PSI score deciles DEV→HO (clase 3)", 0.003, N_AUSTRAL["DEV"], N_AUSTRAL["HO"], 10),
        ("PSI score deciles DEV→OOT (clase 3)", 0.005, N_AUSTRAL["DEV"], N_AUSTRAL["OOT"], 10),
        ("PSI score deciles DEV→TTD (clase 3)", 0.015, N_AUSTRAL["DEV"], N_AUSTRAL["TTD"], 10),
        ("PSI score 8 bandas DEV→OOT (clase 5)", 0.002, N_AUSTRAL["DEV"], N_AUSTRAL["OOT"], 8),
        ("PSI score 8 bandas DEV→TTD (clase 5)", 0.013, N_AUSTRAL["DEV"], N_AUSTRAL["TTD"], 8),
        ("CSI deuda_interna_max_3m DEV→TTD (supuesto B=5)", 0.138, N_AUSTRAL["DEV"], N_AUSTRAL["TTD"], 5),
        ("CSI carga_financiera DEV→TTD (supuesto B=5)", 0.062, N_AUSTRAL["DEV"], N_AUSTRAL["TTD"], 5),
        ("PSI uso_linea_prom_12m DEV→TTD (clase 3, 10 bins)", 0.020, N_AUSTRAL["DEV"], N_AUSTRAL["TTD"], 10),
    ]
    _f = []
    for _nom, _v, _ne, _na, _B in _casos:
        _c = c_n(_ne, _na)
        _f.append({"caso": _nom, "valor": _v, "B": _B, "E[PSI|H0]": (_B - 1) * _c,
                   "crítico 95%": _c * stats.chi2.ppf(0.95, _B - 1), "PSI/c": _v / _c,
                   "p aprox": stats.chi2.sf(_v / _c, _B - 1)})
    austral_nula = pd.DataFrame(_f).set_index("caso")
    austral_nula.round(5)
    return (austral_nula,)


@app.cell
def _(mo):
    mo.md(r"""
    **Lectura de la tabla Austral.** Los números del curso, puestos contra su propia nula: las
    comparaciones DEV→HO y DEV→OOT son ruido puro (p > 0,5), como debe ser; DEV→TTD es **estadísticamente
    significativa** en los deciles, en las 8 bandas y en `uso_linea_prom_12m` (p < 0,001), aunque todas son
    «verdes». La bandeja de hoy es distinta de DEV, pero poco. El CSI 0,138 de `deuda_interna_max_3m` está a
    ~80 veces su valor esperado bajo H0. La lectura correcta combina dos preguntas separadas: *¿hay
    cambio?* (p-valor, depende de $n$) y *¿importa?* (magnitud del PSI o, mejor, impacto en puntos y en PD:
    §7). El semáforo fijo mezcla las dos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Sensibilidad: bins, ε y masa de ceros

    ### 5.1 Número de bins
    Mismo drift (media 0,08 DE), $n=3.000$ por muestra, 150 réplicas por $B$. El PSI **crece con $B$
    incluso sin drift** (su esperanza nula es $(B-1)c$): un PSI de 0,02 con 5 bins y uno de 0,02 con 50 bins
    no son comparables. Lo que se puede comparar es el p-valor (o la potencia).
    """)
    return


@app.cell
def _(c_n, conteos, cortes_cuantil, np, pd, plt, psi_np, stats):
    _rng = np.random.default_rng(55)
    _n, _R, _mu = 3000, 150, 0.08
    _f = []
    for _B in [3, 5, 8, 10, 15, 20, 30, 50]:
        _v0, _v1 = [], []
        for _r in range(_R):
            _xe = _rng.standard_normal(_n)
            _c = cortes_cuantil(_xe, _B)
            _v0.append(psi_np(conteos(_xe, _c) / _n, conteos(_rng.standard_normal(_n), _c) / _n, 1e-4))
            _v1.append(psi_np(conteos(_xe, _c) / _n, conteos(_rng.normal(_mu, 1, _n), _c) / _n, 1e-4))
        _crit = c_n(_n, _n) * stats.chi2.ppf(0.95, _B - 1)
        _f.append({"B": _B, "PSI medio H0": np.mean(_v0), "(B-1)c": (_B - 1) * c_n(_n, _n),
                   "PSI medio drift": np.mean(_v1), "crítico 95%": _crit,
                   "potencia (crítico χ²)": np.mean(np.array(_v1) > _crit)})
    sens_bins = pd.DataFrame(_f).set_index("B")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.plot(sens_bins.index, sens_bins["PSI medio H0"], "o-", label="PSI medio sin drift")
    _ax.plot(sens_bins.index, sens_bins["PSI medio drift"], "s-", label="PSI medio con Δ = 0,08 DE")
    _ax.plot(sens_bins.index, sens_bins["crítico 95%"], "k--", lw=1, label="crítico 95% (χ²)")
    _ax.set_xlabel("número de bins B")
    _ax.set_ylabel("PSI")
    _ax.set_title("El PSI depende de B aun sin drift (n = 3.000 por muestra)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig_bins = _fig
    sens_bins.round(4)
    return (sens_bins,)


@app.cell
def _(mo, sens_bins):
    mo.md(rf"""
    **Lectura.** Sin drift, el PSI medio pasa de {sens_bins.loc[3, 'PSI medio H0']:.4f} (3 bins) a
    {sens_bins.loc[50, 'PSI medio H0']:.4f} (50 bins), siguiendo $(B-1)c$. Con drift de media, la potencia
    es máxima con pocos bins ({sens_bins['potencia (crítico χ²)'].idxmax()} aquí) y cae lentamente al subir
    $B$: los grados de libertad adicionales agregan ruido sin agregar señal (un corrimiento de media es un
    fenómeno de «un grado de libertad»). Para cambios de forma o de cola la conclusión se invierte (probar
    en §3 con «varianza»/«cola» y B = 30). No existe un $B$ óptimo universal; existe un $B$ declarado y
    congelado con el modelo.
    """)
    return


@app.cell
def _(mo):
    eps_ceros = mo.ui.dropdown(["1e-06", "1e-04", "1e-03", "1e-02"], value="1e-04",
                               label="ε sumado a las proporciones")
    mo.vstack([mo.md(r"""
    ### 5.2 El ε: un bin vacío pesa lo que tú decidas
    Si en la muestra actual un bin queda vacío ($a_b=0$), su aporte es
    $(\varepsilon-e_b-\varepsilon)\ln\frac{\varepsilon}{e_b+\varepsilon}\approx e_b\ln(e_b/\varepsilon)$:
    **depende del logaritmo de ε**. Pasar de 1e-4 a 1e-6 casi duplica el aporte de ese bin.
    """), eps_ceros])
    return (eps_ceros,)


@app.cell
def _(aportes_psi, eps_ceros, np, pd):
    _eps = float(eps_ceros.value)
    _f = []
    for _eb in [0.005, 0.01, 0.02, 0.05, 0.10]:
        _fila = {"e_b (DEV)": _eb}
        for _ee in [1e-6, 1e-4, 1e-3, 1e-2]:
            _fila[f"aporte ε={_ee:g}"] = float(aportes_psi(np.array([_eb]), np.array([0.0]), _ee)[0])
        _fila["aproximación e·ln(e/ε) (ε elegido)"] = _eb * np.log(_eb / _eps)
        _fila["Laplace (+0,5 conteo, n=500)"] = float(aportes_psi(np.array([(_eb * 500 + 0.5) / 500.5]),
                                                                  np.array([0.5 / 500.5]))[0])
        _f.append(_fila)
    tabla_eps = pd.DataFrame(_f).set_index("e_b (DEV)")
    tabla_eps.round(4)
    return (tabla_eps,)


@app.cell
def _(mo):
    masa_ceros = mo.ui.slider(0.70, 0.97, step=0.01, value=0.92, label="Masa de ceros en DEV")
    cambio_ceros = mo.ui.slider(-0.10, 0.05, step=0.01, value=-0.03,
                                label="Cambio en la masa de ceros (actual − DEV)")
    mo.vstack([mo.md(r"""
    ### 5.3 Masa de ceros: por qué el curso obtiene `NaN` y cómo hacerlo medible
    Variable de juguete tipo `dias_mora_ult`: cero con probabilidad $p_0$, y si no, días de mora
    $\sim$ Gamma (1–89). Con $p_0 \ge 0{,}9$ los cuantiles 0 a 0,9 son todos 0: `np.unique` deja 2 cortes
    y el `psi()` del curso devuelve `NaN`. Alternativas: (a) bin especial para la moda + deciles del resto
    (lo que ya hace `binear()` del curso); (b) descomposición en dos etapas: PSI del indicador
    $\mathbb{1}[x>0]$ + PSI de la distribución condicional entre los no-cero.
    """), mo.hstack([masa_ceros, cambio_ceros], justify="start")])
    return cambio_ceros, masa_ceros


@app.cell
def _(cambio_ceros, cortes_cuantil, kl_np, masa_ceros, np, p_psi, pd, proporciones, psi_curso, psi_np, tabla_psi_binear):
    def mora_corta(n, p0, rng):
        _x = np.round(rng.gamma(1.5, 18, n)).clip(1, 89)
        return np.where(rng.random(n) < p0, 0.0, _x)

    _rng = np.random.default_rng(90)
    _p0 = float(masa_ceros.value)
    _p0a = float(np.clip(_p0 + cambio_ceros.value, 0.01, 0.995))
    _ne, _na = 4000, 4000
    x_mora_dev = mora_corta(_ne, _p0, _rng)
    x_mora_act = mora_corta(_na, _p0a, _rng)
    _psi_c = psi_curso(x_mora_dev, x_mora_act)
    _n_cortes = len(np.unique(np.quantile(x_mora_dev, np.linspace(0, 1, 11))))
    _t_moda = tabla_psi_binear(x_mora_dev, x_mora_act, bins=10, eps=0.0)
    _psi_moda = float(_t_moda["aporte"].sum())
    # dos etapas (chain rule exacta de KL, ver documento §3.4)
    _e0, _a0 = np.mean(x_mora_dev == 0), np.mean(x_mora_act == 0)
    _eb, _ab = np.array([_e0, 1 - _e0]), np.array([_a0, 1 - _a0])
    _nz_e, _nz_a = x_mora_dev[x_mora_dev > 0], x_mora_act[x_mora_act > 0]
    _c = cortes_cuantil(_nz_e, 10)
    _ec, _ac = proporciones(_nz_e, _c), proporciones(_nz_a, _c)
    _psi_bin = psi_np(_eb, _ab)
    _psi_cond = psi_np(_ec, _ac, 1e-4)
    _cadena = _psi_bin + (1 - _a0) * kl_np(_ac, _ec) + (1 - _e0) * kl_np(_ec, _ac)
    _B_moda = int((_t_moda[["pct_esp", "pct_act"]].sum(axis=1) > 0).sum())
    tabla_ceros = pd.DataFrame([
        {"método": "curso: deciles de DEV (+1e-4)", "PSI": _psi_c,
         "bins efectivos": _n_cortes - 1 if _n_cortes >= 3 else 0,
         "p aprox": p_psi(_psi_c, _ne, _na, _n_cortes - 1) if _psi_c == _psi_c else np.nan},
        {"método": "binear(): bin de moda + deciles del resto", "PSI": _psi_moda,
         "bins efectivos": _B_moda, "p aprox": p_psi(_psi_moda, _ne, _na, _B_moda)},
        {"método": "etapa 1: indicador x>0 (2 bins)", "PSI": _psi_bin, "bins efectivos": 2,
         "p aprox": p_psi(_psi_bin, _ne, _na, 2)},
        {"método": "etapa 2: deciles entre no-cero", "PSI": _psi_cond, "bins efectivos": len(_c) - 1,
         "p aprox": p_psi(_psi_cond, len(_nz_e), len(_nz_a), len(_c) - 1)},
        {"método": "regla de la cadena (etapa 1 + KL ponderadas)", "PSI": _cadena,
         "bins efectivos": np.nan, "p aprox": np.nan},
    ]).set_index("método")
    check_cadena = (_psi_moda, _cadena, _t_moda)
    tabla_ceros.round(5)
    return check_cadena, mora_corta, tabla_ceros, x_mora_act, x_mora_dev


@app.cell
def _(masa_ceros, mo, tabla_ceros):
    _pc = tabla_ceros.loc["curso: deciles de DEV (+1e-4)", "PSI"]
    _txt = ("`NaN`: los deciles colapsan (menos de 3 cortes únicos)" if _pc != _pc
            else f"{_pc:.4f}, pero con solo {int(tabla_ceros.iloc[0, 1])} bins efectivos y la moda mezclada con valores positivos")
    mo.md(rf"""
    **Lectura.** Con masa de ceros {masa_ceros.value:.0%}, el PSI del curso da {_txt}. El binning con
    bin de moda mide sin problemas, y la regla de la cadena lo descompone **exactamente**:
    $$J(a,e) = J(a_0,e_0) + (1-a_0)\,\mathrm{{KL}}(a_{{\cdot|x>0}}\|e_{{\cdot|x>0}}) +
    (1-e_0)\,\mathrm{{KL}}(e_{{\cdot|x>0}}\|a_{{\cdot|x>0}})$$
    (fila 5 = fila 2 a precisión de máquina con ε = 0). Así el reporte dice *qué* se movió: ¿hay más gente
    con mora (etapa 1) o los que tienen mora la tienen distinta (etapa 2)? Y mira los p-valores: la etapa 1
    (p ≈ {tabla_ceros.iloc[2, 2]:.3g}) detecta el cambio de masa de ceros con más potencia que el binning
    completo (p ≈ {tabla_ceros.iloc[1, 2]:.3g}), que diluye esa señal entre ~10 grados de libertad
    adicionales. La regla «sin PSI medible no
    sigue» descarta variables por un artefacto del binning, no por una propiedad de los datos: la regla
    defendible es «se mide con el binning declarado del modelo (con bin de moda); si ni así hay ≥ 2 bins
    con masa suficiente, se monitorea el indicador binario».
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Taxonomía de cambio: qué ve el PSI y qué no

    Con $p(x,y)$ la distribución conjunta:
    **covariate shift** cambia $p(x)$ con $p(y|x)$ fija; **prior shift** cambia $p(y)$ con $p(x|y)$ fija;
    **concept drift** cambia $p(y|x)$. El PSI mira solo $p(x)$ (o $p(s)$ para el score): ve el primero,
    ve el segundo **indirectamente** (porque $p(s)=\pi p(s|1)+(1-\pi)p(s|0)$ se mueve con $\pi$) y es
    **ciego** al tercero cuando $p(x)$ no cambia. Experimento de juguete con score en log-odds:
    $s\,|\,\text{bueno}\sim N(1{,}5;1)$, $s\,|\,\text{malo}\sim N(-0{,}5;1)$, $n=20.000$.
    """)
    return


@app.cell
def _(conteos, cortes_cuantil, np, pd, psi_np):
    def gini_np(s, y):
        """Gini = 2·AUC − 1 con score alto = bueno (AUC por rangos, empates promediados)."""
        _s, _y = np.asarray(s, float), np.asarray(y, float)
        _orden = np.argsort(-_s, kind="mergesort")
        _r = np.empty(len(_s))
        _r[_orden] = np.arange(1, len(_s) + 1)
        _r = pd.Series(_r).groupby(pd.Series(-_s)).transform("mean").to_numpy()
        _n1, _n0 = _y.sum(), len(_y) - _y.sum()
        _auc = (_r[_y == 1].sum() - _n1 * (_n1 + 1) / 2) / (_n1 * _n0)
        return 2 * _auc - 1

    _rng = np.random.default_rng(6)
    _n = 20000

    def _mundo(pi, mu_malo=-0.5, concepto=False):
        _y = (_rng.random(_n) < pi).astype(float)
        _s = np.where(_y == 1, _rng.normal(mu_malo, 1, _n), _rng.normal(1.5, 1, _n))
        if concepto:     # misma p(s) que la base; cambia p(y|s): +0,6 en log-odds de malo
            _y0 = (_rng.random(_n) < 0.10).astype(float)
            _s = np.where(_y0 == 1, _rng.normal(-0.5, 1, _n), _rng.normal(1.5, 1, _n))
            _lo = np.log(0.10 / 0.90) + np.log(
                np.exp(-0.5 * (_s + 0.5) ** 2) / np.exp(-0.5 * (_s - 1.5) ** 2))
            _y = (_rng.random(_n) < 1 / (1 + np.exp(-(_lo + 0.6)))).astype(float)
        return _s, _y

    _s0, _y0 = _mundo(0.10)
    _c = cortes_cuantil(_s0, 10)
    _e = conteos(_s0, _c) / _n
    _f = []
    for _nom, (_s, _y), _que in [
        ("base (sin cambio)", _mundo(0.10), "nada"),
        ("covariate shift en x omitida*", _mundo(0.10, mu_malo=-0.5), "ver caso canal §7"),
        ("prior shift π: 10% → 15%", _mundo(0.15), "p(y) y por ende p(s)"),
        ("prior shift π: 10% → 20%", _mundo(0.20), "p(y) y por ende p(s)"),
        ("concept drift: +0,6 log-odds", _mundo(0.10, concepto=True), "p(y|s); p(s) intacta"),
    ]:
        _f.append({"escenario": _nom, "qué cambia": _que,
                   "PSI score (deciles)": psi_np(_e, conteos(_s, _c) / _n, 1e-4),
                   "tasa de malos": _y.mean(), "Gini": gini_np(_s, _y)})
    taxonomia = pd.DataFrame(_f).set_index("escenario").drop(index="covariate shift en x omitida*")
    taxonomia.round(4)
    return gini_np, taxonomia


@app.cell
def _(mo, taxonomia):
    _cd = taxonomia.loc["concept drift: +0,6 log-odds"]
    _p15 = taxonomia.loc["prior shift π: 10% → 15%"]
    mo.md(rf"""
    **Lectura.** El concept drift sube la tasa de malos de ~10% a {_cd['tasa de malos']:.1%} con un PSI
    del score de {_cd['PSI score (deciles)']:.4f} (ruido): **el PSI no puede verlo**. El prior shift de 10%
    a 15% sí mueve el PSI ({_p15['PSI score (deciles)']:.4f}), pero el número no dice si el cambio es de
    mezcla de clases o de covariables. Consecuencia de gobierno: el PSI/CSI es un indicador *adelantado* y
    *parcial*; la verificación de $p(y|x)$ solo llega con desempeño maduro (backtesting binomial, HL,
    Gini por cosecha: M19, M12 y M20 de esta serie). En §7 se ve la versión realista con el generador.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Caso Banco Sintético: CSI de todas las variables, canal, score y dirección

    Generador común de la serie (`generar_cartera`): 24 cohortes con desempeño + 6 de TTD. Verdad conocida:
    (i) la mezcla de `canal` cambia desde 2025-01 de [45, 25, 15, 15]% a [25, 25, 35, 15]% (PSI poblacional
    teórico = 0,287); (ii) `uso_linea_prom_12m` tiene una tendencia lenta en el tiempo; (iii) las cohortes 2025
    tienen +0,35 en log-odds (deterioro macro, **concept drift** puro: no toca $X$). Primero, la tabla de
    estabilidad del embudo (clase 3) extendida con p-valor, corrección por multiplicidad (Benjamini-Hochberg
    vía `statsmodels`) y CSI con los bins de `binear()`.
    """)
    return


@app.cell
def _(P_CANAL_BASE, P_CANAL_NUEVO, csi_curso, generar_cartera, multipletests, np, p_psi, pd, psi_curso, psi_np, semaforo, tabla_woe):
    cartera = generar_cartera()
    dev = cartera[cartera["muestra"] == "DEV"].reset_index(drop=True)
    ho = cartera[cartera["muestra"] == "HO"].reset_index(drop=True)
    oot = cartera[cartera["muestra"] == "OOT"].reset_index(drop=True)
    ttd = cartera[cartera["muestra"] == "TTD"].reset_index(drop=True)
    CANDIDATAS = ["uso_linea_prom_12m", "uso_tc_prom_12m", "uso_tc_prom_3m", "meses_desde_mora_12m",
                  "antiguedad_meses", "edad", "renta_mm", "deuda_otras_prom_12m", "carga_financiera",
                  "consultas_6m", "canal"]
    PSI_CANAL_TEORICO = psi_np(P_CANAL_BASE, P_CANAL_NUEVO)

    def _bins_efectivos(xe, xa, bins=10):
        _xe = pd.Series(xe)
        if not pd.api.types.is_numeric_dtype(_xe):
            return int(len(set(_xe.fillna("MISSING").astype(str)) | set(pd.Series(xa).fillna("MISSING").astype(str))))
        _c = np.unique(np.nanquantile(_xe.dropna(), np.linspace(0, 1, bins + 1)))
        return int(len(_c) - 1 + (_xe.isna().any() or pd.Series(xa).isna().any()))

    _f = []
    for _v in CANDIDATAS:
        _B = _bins_efectivos(dev[_v], ttd[_v])
        _pt = psi_curso(dev[_v], ttd[_v])
        _f.append({"variable": _v, "iv": tabla_woe(dev[_v], dev["malo"])[1],
                   "psi_oot": psi_curso(dev[_v], oot[_v]), "psi_ttd": _pt, "B": _B,
                   "p_ttd": p_psi(_pt, len(dev), len(ttd), _B),
                   "csi_oot": csi_curso(dev[_v], oot[_v]), "csi_ttd": csi_curso(dev[_v], ttd[_v])})
    estabilidad = pd.DataFrame(_f).set_index("variable")
    estabilidad["p_ttd_bh"] = multipletests(estabilidad["p_ttd"], method="fdr_bh")[1]
    estabilidad["semáforo curso"] = estabilidad["psi_ttd"].map(semaforo)
    estabilidad["cambio estadístico (BH 5%)"] = np.where(estabilidad["p_ttd_bh"] < 0.05, "sí", "no")
    estabilidad.sort_values("psi_ttd", ascending=False).round(4)
    return CANDIDATAS, PSI_CANAL_TEORICO, cartera, dev, estabilidad, ho, oot, ttd


@app.cell
def _(PSI_CANAL_TEORICO, dev, estabilidad, mo, oot, pd, ttd):
    _mix = pd.DataFrame({m: d["canal"].value_counts(normalize=True) for m, d in
                         [("DEV", dev), ("OOT", oot), ("TTD", ttd)]}).round(3)
    _sig = list(estabilidad.index[estabilidad["cambio estadístico (BH 5%)"] == "sí"])
    mo.vstack([mo.md(rf"""
    **Lectura.** `canal` es la variable plantada: PSI DEV→TTD {estabilidad.loc['canal', 'psi_ttd']:.3f}
    (teórico poblacional {PSI_CANAL_TEORICO:.3f}) → 🔴, se cae en el paso 1 del embudo aunque su IV sea
    bajo ({estabilidad.loc['canal', 'iv']:.3f}). Para categóricas, PSI y CSI coinciden (los niveles son los
    bins). Con corrección BH al 5%, las variables con cambio estadístico DEV→TTD son: {', '.join(_sig)}.
    `uso_linea_prom_12m` es **significativa pero verde** (PSI {estabilidad.loc['uso_linea_prom_12m', 'psi_ttd']:.3f}):
    es la tendencia lenta plantada; el semáforo fijo la ignora, el test la detecta.
    `meses_desde_mora_12m` se mide con solo {int(estabilidad.loc['meses_desde_mora_12m', 'B'])} bins efectivos por
    deciles (masa en 13 y en −9): medible aquí, pero con menos resolución de la que el lector cree.
    Mezcla de canal por muestra:
    """), _mix])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.1 El scorecard y el PSI del score: deciles vs bins vs bandas

    Scorecard de 6 variables (sin `canal`, descartada por estabilidad) sobre WoE de `binear()` ajustado en
    DEV, logística con `statsmodels`, escalado PDO 20 / 600 / 50:1. La master scale sintética usa 8 bandas
    de score con cortes fijos (el generador tiene ~11% de malos, así que los cortes del Austral no sirven).
    El mismo score, tres binnings, tres PSI distintos.
    """)
    return


@app.cell
def _(a_woe, dev, np, pd, sm, tabla_woe):
    SEL = ["uso_linea_prom_12m", "uso_tc_prom_12m", "meses_desde_mora_12m", "antiguedad_meses",
           "carga_financiera", "consultas_6m"]
    MAPAS = {v: tabla_woe(dev[v], dev["malo"])[0]["woe"] for v in SEL}
    _W = a_woe(dev, SEL, dev, MAPAS)
    modelo = sm.Logit(dev["malo"].to_numpy(), sm.add_constant(_W)).fit(disp=0)
    FACTOR = 20 / np.log(2)
    OFFSET = 600 - FACTOR * np.log(50)

    def puntos_por_variable(df):
        """Matriz n×k de puntos: puntos(v,b) = −(β_v·WoE_vb + β0/k)·factor + offset/k."""
        _Wd = a_woe(df, SEL, dev, MAPAS)
        _k = len(SEL)
        _b0 = modelo.params["const"]
        return pd.DataFrame({v: -(modelo.params[v] * _Wd[v] + _b0 / _k) * FACTOR + OFFSET / _k
                             for v in SEL})

    def score_de(df):
        return puntos_por_variable(df).sum(axis=1).to_numpy()

    CORTES_BANDAS = np.array([-np.inf, 500, 515, 530, 545, 560, 575, 590, np.inf])
    ETIQ_BANDAS = ["E", "D", "C2", "C1", "B2", "B1", "A2", "A1"]
    assert (modelo.params[SEL] < 0).all(), "coeficientes WoE deben ser negativos"
    modelo.params.round(3).to_frame("β")
    return CORTES_BANDAS, ETIQ_BANDAS, FACTOR, MAPAS, OFFSET, SEL, modelo, puntos_por_variable, score_de


@app.cell
def _(CORTES_BANDAS, conteos, cortes_cuantil, dev, gini_np, ho, np, oot, p_psi, pd, psi_np, score_de, ttd):
    SCORES = {"DEV": score_de(dev), "HO": score_de(ho), "OOT": score_de(oot), "TTD": score_de(ttd)}
    _esquemas = {"deciles de DEV (10)": cortes_cuantil(SCORES["DEV"], 10),
                 "quintiles de DEV (5)": cortes_cuantil(SCORES["DEV"], 5),
                 "20 bins de DEV": cortes_cuantil(SCORES["DEV"], 20),
                 "8 bandas master scale": CORTES_BANDAS}
    _f = []
    for _nom, _c in _esquemas.items():
        _e = conteos(SCORES["DEV"], _c) / len(SCORES["DEV"])
        for _m in ["HO", "OOT", "TTD"]:
            _a = conteos(SCORES[_m], _c) / len(SCORES[_m])
            _psi = psi_np(_e, _a, 1e-4)
            _f.append({"binning": _nom, "muestra": _m, "PSI": _psi,
                       "p aprox": p_psi(_psi, len(dev), len(SCORES[_m]), len(_c) - 1)})
    psi_score = pd.DataFrame(_f).pivot(index="binning", columns="muestra", values=["PSI", "p aprox"])
    gini_muestras = {m: gini_np(SCORES[m], d["malo"]) for m, d in [("DEV", dev), ("HO", ho), ("OOT", oot)]}
    _masas = pd.DataFrame({m: conteos(SCORES[m], CORTES_BANDAS) / len(SCORES[m]) for m in SCORES},
                          index=["E", "D", "C2", "C1", "B2", "B1", "A2", "A1"])
    mix_bandas = _masas
    _ = np
    psi_score.round(4)
    return SCORES, gini_muestras, mix_bandas, psi_score


@app.cell
def _(gini_muestras, mix_bandas, mo, psi_score):
    mo.vstack([mo.md(rf"""
    **Lectura.** Gini DEV/HO/OOT = {gini_muestras['DEV']:.3f} / {gini_muestras['HO']:.3f} /
    {gini_muestras['OOT']:.3f}. El PSI del score DEV→TTD cambia con el binning
    (quintiles {psi_score.loc['quintiles de DEV (5)', ('PSI', 'TTD')]:.4f} · deciles
    {psi_score.loc['deciles de DEV (10)', ('PSI', 'TTD')]:.4f} · 20 bins
    {psi_score.loc['20 bins de DEV', ('PSI', 'TTD')]:.4f} · 8 bandas
    {psi_score.loc['8 bandas master scale', ('PSI', 'TTD')]:.4f}); los p-valores cuentan la misma historia
    con menos variación, porque ya descuentan los grados de libertad. Las **bandas** tienen ventaja de
    gobierno (cada aporte se lee como «la banda E se llenó») y desventaja estadística si tienen masas
    muy desiguales (bandas con poca masa aportan ruido). Mezcla por banda (proporción de cada muestra):
    """), mix_bandas.round(3)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.2 Aporte direccional: del CSI (sin signo) al corrimiento en puntos (con signo)

    El CSI dice *cuánto* se movió una variable pero no *hacia dónde* ni *cuánto le importa al score*.
    Como el score es aditivo, el cambio en el score medio se descompone **exactamente**:
    $$\Delta\bar s = \sum_v \underbrace{\sum_b (a_{vb}-e_{vb})\,\text{puntos}_{vb}}_{\Delta\text{puntos}_v}.$$
    Esta es la versión cuantitativa del «análisis de características» de los manuales de scorecards
    (Siddiqi) y la «variante de industria» que la clase 5 mencionó: ponderar el corrimiento por puntos.
    """)
    return


@app.cell
def _(SCORES, SEL, csi_curso, dev, np, pd, puntos_por_variable, ttd):
    _P_dev = puntos_por_variable(dev)
    _P_ttd = puntos_por_variable(ttd)
    _f = []
    for _v in SEL:
        # distribución por bin (valor de puntos distinto = bin distinto) × puntos del bin
        _e = _P_dev[_v].round(8).value_counts(normalize=True)
        _a = _P_ttd[_v].round(8).value_counts(normalize=True)
        _idx = _e.index.union(_a.index)
        _de = (_a.reindex(_idx, fill_value=0) - _e.reindex(_idx, fill_value=0))
        _f.append({"variable": _v, "CSI DEV→TTD": csi_curso(dev[_v], ttd[_v]),
                   "Δpuntos (+ = mejor)": float((_de * _idx.to_numpy()).sum()),
                   "rango de puntos": float(_P_dev[_v].max() - _P_dev[_v].min())})
    direccional = pd.DataFrame(_f).set_index("variable").sort_values("CSI DEV→TTD", ascending=False)
    delta_score_medio = float(SCORES["TTD"].mean() - SCORES["DEV"].mean())
    suma_delta_puntos = float(direccional["Δpuntos (+ = mejor)"].sum())
    _ = np
    direccional.round(4)
    return delta_score_medio, direccional, suma_delta_puntos


@app.cell
def _(delta_score_medio, direccional, mo, suma_delta_puntos):
    _top = direccional["Δpuntos (+ = mejor)"].abs().idxmax()
    mo.md(rf"""
    **Lectura.** $\sum_v\Delta\text{{puntos}}_v$ = {suma_delta_puntos:+.4f} = Δ score medio
    DEV→TTD = {delta_score_medio:+.4f} (identidad exacta, verificada en los checks). La variable que más
    mueve el score es `{_top}` ({direccional.loc[_top, 'Δpuntos (+ = mejor)']:+.3f} puntos). Una variable
    con CSI alto y rango de puntos chico mueve poco el score; una con CSI moderado y rango grande puede
    moverlo mucho. Por eso el tablero debería reportar las dos columnas: CSI (¿cambió la población?) y
    Δpuntos (¿cuánto le importa al modelo y hacia dónde?).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.3 Lo que el PSI no ve: el deterioro macro y el canal omitido

    Regeneramos la cartera con la misma semilla y `deterioro = 0`. El generador consume los números
    aleatorios en el mismo orden, así que $X$ es **idéntica**; solo cambia el target (concept drift puro).
    Luego repetimos con `drift_canal = False`: el canal no está en el modelo, así que su covariate shift es,
    para el modelo, un cambio de $p(y\,|\,x_\text{modelo})$.
    """)
    return


@app.cell
def _(cartera, generar_cartera, modelo, np, oot, pd, psi_curso, score_de, SEL, a_woe, dev, MAPAS, sm):
    _sin_det = generar_cartera(deterioro=0.0)
    _sin_canal = generar_cartera(drift_canal=False)
    x_identica = bool(np.allclose(_sin_det[SEL].to_numpy(dtype=float), cartera[SEL].to_numpy(dtype=float),
                                  equal_nan=True))
    _oot0 = _sin_det[_sin_det["muestra"] == "OOT"].reset_index(drop=True)
    _dev0 = _sin_det[_sin_det["muestra"] == "DEV"].reset_index(drop=True)
    _ootc = _sin_canal[_sin_canal["muestra"] == "OOT"].reset_index(drop=True)
    _pd_oot = modelo.predict(sm.add_constant(a_woe(oot, SEL, dev, MAPAS), has_constant="add"))
    _filas = []
    for _nom, _d, _o in [("real (deterioro 0,35 + canal)", dev, oot),
                         ("sin deterioro (canal sí deriva)", _dev0, _oot0),
                         ("sin drift de canal (deterioro sí)", _sin_canal[_sin_canal["muestra"] == "DEV"], _ootc)]:
        _filas.append({"escenario": _nom,
                       "PSI score deciles DEV→OOT": psi_curso(score_de(_d.reset_index(drop=True)),
                                                               score_de(_o)),
                       "PD media del modelo (OOT)": float(np.mean(_pd_oot)),
                       "PD verdadera media (OOT)": float(_o["pd_verdadera"].mean()),
                       "tasa de malos OOT": float(_o["malo"].mean())})
    ciegas = pd.DataFrame(_filas).set_index("escenario")
    ciegas.round(4)
    return ciegas, x_identica


@app.cell
def _(ciegas, mo, x_identica):
    _r = ciegas.iloc[0]
    _s = ciegas.iloc[1]
    _c = ciegas.iloc[2]
    mo.md(rf"""
    **Lectura.** $X$ idéntica entre escenarios: **{x_identica}**. El PSI del score DEV→OOT es el mismo con o sin
    deterioro ({_r['PSI score deciles DEV→OOT']:.4f} vs {_s['PSI score deciles DEV→OOT']:.4f}), pero la PD
    verdadera media en OOT pasa de {_s['PD verdadera media (OOT)']:.2%} a {_r['PD verdadera media (OOT)']:.2%},
    mientras el modelo sigue prediciendo {_r['PD media del modelo (OOT)']:.2%}. Sin drift de canal la PD
    verdadera es {_c['PD verdadera media (OOT)']:.2%}: el canal omitido que deriva (más `app`, +0,30 log-odds)
    también sube el riesgo sin que ninguna variable del modelo se mueva. Moraleja: **PSI verde no es modelo
    sano**; es población parecida. El CSI de variables *fuera* del modelo (como `canal`) es un monitor
    legítimo: avisa de un posible cambio de $p(y|x_\text{{modelo}})$ antes de que madure el desempeño.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 7.4 DEV congelado vs ventana móvil

    Por cohorte mensual ($n\approx 800$): PSI contra DEV congelado y contra el mes anterior (ventana móvil de
    un mes), con la banda de ruido al 95% de cada comparación. La tendencia lenta de `uso_linea_prom_12m` es el
    caso de la «rana hervida».
    """)
    return


@app.cell
def _(cartera, cortes_cuantil, conteos, dev, np, pd, plt, psi_np, umbral_psi):
    _var = "uso_linea_prom_12m"
    _c = cortes_cuantil(dev[_var], 10)
    _e_dev = conteos(dev[_var], _c) / len(dev)
    _meses = sorted(cartera["cohorte"].unique())
    _f = []
    _prev = None
    for _m in _meses:
        _x = cartera.loc[cartera["cohorte"] == _m, _var].to_numpy()
        _a = conteos(_x, _c) / len(_x)
        _fila = {"cohorte": _m, "n": len(_x), "PSI vs DEV": psi_np(_e_dev, _a, 1e-4),
                 "crítico vs DEV": umbral_psi(len(dev), len(_x), 10)}
        if _prev is not None:
            _fila["PSI vs mes anterior"] = psi_np(_prev[0], _a, 1e-4)
            _fila["crítico móvil"] = umbral_psi(_prev[1], len(_x), 10)
        _prev = (_a, len(_x))
        _f.append(_fila)
    ventana = pd.DataFrame(_f).set_index("cohorte")
    _fig, _ax = plt.subplots(figsize=(10, 3.4))
    _xx = np.arange(len(ventana))
    _ax.plot(_xx, ventana["PSI vs DEV"], "o-", ms=3, label="PSI vs DEV congelado")
    _ax.plot(_xx, ventana["PSI vs mes anterior"], "s-", ms=3, label="PSI vs mes anterior")
    _ax.plot(_xx, ventana["crítico vs DEV"], "k--", lw=1, label="crítico 95% vs DEV")
    _ax.plot(_xx, ventana["crítico móvil"], "k:", lw=1, label="crítico 95% móvil")
    _ax.set_xticks(_xx[::3], ventana.index[::3], rotation=45, fontsize=8)
    _ax.set_xlabel("cohorte")
    _ax.set_ylabel("PSI (10 bins de DEV)")
    _ax.set_title(f"{_var}: deriva lenta")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    fig_ventana = _fig
    fig_ventana
    return fig_ventana, ventana


@app.cell
def _(mo, ventana):
    _ult = ventana.iloc[-6:]
    _fuera_dev = int((_ult["PSI vs DEV"] > _ult["crítico vs DEV"]).sum())
    _fuera_mov = int((ventana["PSI vs mes anterior"] > ventana["crítico móvil"]).sum())
    mo.md(rf"""
    **Lectura.** En los 6 meses de TTD, {_fuera_dev} de 6 cohortes superan el crítico contra DEV congelado;
    en toda la serie, solo {_fuera_mov} de {len(ventana) - 1} comparaciones mes a mes superan su crítico
    (lo esperado por azar al 5% es ~{0.05 * (len(ventana) - 1):.1f}). La ventana móvil es ciega a la deriva
    lenta: cada mes se parece al anterior. Sirve para cambios **abruptos** (un cambio de política, un nuevo
    canal, un bug de captura). Regla defendible: DEV congelado como referencia oficial (auditable, mide
    distancia acumulada al mundo en que se estimó el modelo) + ventana móvil como detector de quiebres.
    Y ninguna de las dos pasa de 0,10: el semáforo fijo no habría dicho nada.
    """)
    return


@app.cell
def _(mo):
    mu_cancel = mo.ui.slider(0.0, 1.0, step=0.05, value=0.4, label="Corrimiento μ (DE)")
    peso_x2 = mo.ui.slider(0.0, 2.0, step=0.1, value=1.0, label="Peso relativo de x₂ en el score")
    modo_puntos = mo.ui.dropdown(["lineales (bins muy finos)", "10 bins", "5 bins"],
                                 value="lineales (bins muy finos)", label="Puntos por variable")
    mo.vstack([mo.md(r"""
    ## 8. Cancelación: CSI altos, PSI del score ≈ 0

    Dos variables independientes $x_1, x_2 \sim N(0,1)$ en DEV; el score es aditivo,
    $\text{puntos}_v = -20\,w_v\,g_v(x_v)$ (más alto = mejor), con $w_1=1$, $w_2$ ajustable y $g_v$ lineal
    ($g(x)=x$, el límite de bins muy finos) o escalonada (media de $x$ en su bin de DEV, 10 o 5 bins). El CSI
    se mide siempre con 5 bins de DEV. En la muestra actual $x_1\sim N(\mu,1)$ (empeora) y $x_2\sim N(-\mu,1)$
    (mejora). Con $w_2 = 1$ los efectos se cancelan en el score: es el caso `deuda_interna_max_3m` (CSI 0,138)
    con PSI del score 0,013 de la clase 5, en versión de laboratorio.
    """), mo.hstack([mu_cancel, peso_x2, modo_puntos], justify="start")])
    return modo_puntos, mu_cancel, peso_x2


@app.cell
def _(c_n, conteos, cortes_cuantil, modo_puntos, mu_cancel, np, p_psi, pd, peso_x2, psi_np, umbral_psi):
    _rng = np.random.default_rng(31)
    _n = 5000
    _mu, _w2 = float(mu_cancel.value), float(peso_x2.value)
    _X_e = _rng.standard_normal((_n, 2))
    _X_a = np.column_stack([_rng.normal(_mu, 1, _n), _rng.normal(-_mu, 1, _n)])
    _w = np.array([1.0, _w2])
    _pts_e, _pts_a, _f = [], [], []
    _Bp = {"10 bins": 10, "5 bins": 5}.get(modo_puntos.value, 0)
    for _j in range(2):
        if _Bp:
            _cp = cortes_cuantil(_X_e[:, _j], _Bp)
            _ie = np.searchsorted(_cp[1:-1], _X_e[:, _j])
            _ia = np.searchsorted(_cp[1:-1], _X_a[:, _j])
            _medias = np.array([_X_e[_ie == b, _j].mean() for b in range(_Bp)])
            _pe_j, _pa_j = -20 * _w[_j] * _medias[_ie], -20 * _w[_j] * _medias[_ia]
        else:
            _pe_j, _pa_j = -20 * _w[_j] * _X_e[:, _j], -20 * _w[_j] * _X_a[:, _j]
        _pts_e.append(_pe_j)
        _pts_a.append(_pa_j)
        _c = cortes_cuantil(_X_e[:, _j], 5)
        _e, _a = conteos(_X_e[:, _j], _c) / _n, conteos(_X_a[:, _j], _c) / _n
        _psi = psi_np(_e, _a, 1e-4)
        _f.append({"variable": f"x{_j + 1}", "CSI (5 bins)": _psi, "p": p_psi(_psi, _n, _n, 5),
                   "Δpuntos": float(_pa_j.mean() - _pe_j.mean())})
    _s_e = np.sum(_pts_e, axis=0)
    _s_a = np.sum(_pts_a, axis=0)
    _cs = cortes_cuantil(_s_e, 10)
    _psi_s = psi_np(conteos(_s_e, _cs) / _n, conteos(_s_a, _cs) / _n, 1e-4)
    _B_s = len(_cs) - 1
    _f.append({"variable": "score (deciles)", "CSI (5 bins)": _psi_s, "p": p_psi(_psi_s, _n, _n, _B_s),
               "Δpuntos": float(_s_a.mean() - _s_e.mean())})
    cancelacion = pd.DataFrame(_f).set_index("variable")
    info_cancel = {"crit_s": umbral_psi(_n, _n, _B_s), "E0": (_B_s - 1) * c_n(_n, _n), "mu": _mu, "w2": _w2,
                   "modo": modo_puntos.value, "sd_e": float(_s_e.std()), "sd_a": float(_s_a.std())}
    cancelacion.round(4)
    return cancelacion, info_cancel


@app.cell
def _(cancelacion, info_cancel, mo):
    _c1 = cancelacion.loc["x1", "CSI (5 bins)"]
    _c2 = cancelacion.loc["x2", "CSI (5 bins)"]
    _ps = cancelacion.loc["score (deciles)", "CSI (5 bins)"]
    mo.md(rf"""
    **Lectura dinámica.** Con μ = {info_cancel['mu']:.2f} y $w_2$ = {info_cancel['w2']:.1f}: CSI de
    $x_1$ = {_c1:.3f}, de $x_2$ = {_c2:.3f}; PSI del score = {_ps:.4f} (esperado bajo H0
    {info_cancel['E0']:.4f}, crítico {info_cancel['crit_s']:.4f}). La columna Δpuntos muestra por qué:
    {cancelacion.loc['x1', 'Δpuntos']:+.2f} y {cancelacion.loc['x2', 'Δpuntos']:+.2f} se compensan
    (Δ score medio {cancelacion.loc['score (deciles)', 'Δpuntos']:+.2f}). Con puntos lineales y $w_2=1$ la
    cancelación es exacta en distribución: $(x_1+\mu)+(x_2-\mu)=x_1+x_2$. Con puntos escalonados
    (modo actual: {info_cancel['modo']}) se cancela la media pero no la **forma**: los bins extremos saturan y la
    desviación estándar del score pasa de {info_cancel['sd_e']:.1f} a {info_cancel['sd_a']:.1f} puntos, así que
    el PSI del score recupera parte de la señal (pruébalo con 5 bins y μ = 0,4). Mueve $w_2$ lejos de 1 y la
    cancelación se rompe. ¿Es un problema si se cancela? Sí: el score ordena a
    una población distinta, los reason codes cambian de frecuencia y la PD por banda puede dejar de ser
    válida si la relación $p(y|x)$ no es exactamente la del modelo (no lineal, interacciones).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Checks del módulo
    Si alguno falla, el notebook falla. Coincidencias numpy vs librería e invariantes teóricas.
    """)
    return


@app.cell
def _(
    anclas_austral, check_cadena, chi2_homog_np, chi2_scipy, conteos, cortes_cuantil, delta_score_medio,
    equivalencia_local, estabilidad, hellinger_np, hellinger_scipy, js_np, js_scipy, kl_np, kl_scipy,
    ks_np, ks_scipy, np, psi_curso, psi_np, psi_scipy, suma_delta_puntos, tabla_nula, taxonomia,
    w_scipy, wasserstein_np, x_identica, ciegas, cancelacion, info_cancel, mora_corta, props_normal,
):
    _rng = np.random.default_rng(123)
    _xe, _xa = _rng.standard_normal(3000), _rng.normal(0.2, 1.2, 2500)
    _c = cortes_cuantil(_xe, 10)
    _ke, _ka = conteos(_xe, _c), conteos(_xa, _c)
    _e, _a = _ke / _ke.sum(), _ka / _ka.sum()
    # 1. PSI = Jeffreys = entropy(a,e)+entropy(e,a)
    assert np.isclose(psi_np(_e, _a), psi_scipy(_e, _a), rtol=1e-12)
    assert np.isclose(psi_np(_e, _a), kl_np(_a, _e) + kl_np(_e, _a), rtol=1e-12)
    assert np.isclose(kl_np(_a, _e), kl_scipy(_a, _e), rtol=1e-12)
    # 2. JS y Hellinger numpy vs scipy
    assert np.isclose(js_np(_a, _e), js_scipy(_a, _e), rtol=1e-9)
    assert np.isclose(hellinger_np(_a, _e), hellinger_scipy(_a, _e), rtol=1e-12)
    # 3. χ² homogeneidad numpy vs chi2_contingency
    assert np.allclose(chi2_homog_np(_ke, _ka), chi2_scipy(_ke, _ka), rtol=1e-10)
    # 4. KS y Wasserstein numpy vs scipy
    assert np.isclose(ks_np(_xe, _xa), ks_scipy(_xe, _xa)[0], rtol=1e-12)
    assert np.isclose(wasserstein_np(_xe, _xa), w_scipy(_xe, _xa), rtol=1e-9)
    # 5. psi_np con bins (c_{k-1}, c_k] reproduce el psi() del curso (pd.cut)
    assert np.isclose(psi_np(_e, _a, 1e-4), psi_curso(_xe, _xa), rtol=1e-12)
    # 6. Austral: se reproducen 0,013 y 0,002 de la clase 5
    assert abs(anclas_austral.loc["DEV→TTD", "PSI curso (ε=1e-4)"] - 0.0130) < 5e-5
    assert abs(anclas_austral.loc["DEV→OOT", "PSI curso (ε=1e-4)"] - 0.0020) < 1e-4
    # 7. PSI/c ≈ X² (segundo orden) cuando el drift es chico
    assert abs(anclas_austral.loc["DEV→TTD", "PSI/c"] / anclas_austral.loc["DEV→TTD", "X² scipy"] - 1) < 0.05
    # 8. Equivalencia local y procesamiento de datos: PSI con bins ≤ J continuo
    _chico = equivalencia_local[equivalencia_local["valor"].isin([0.05, 1.05])]
    assert np.allclose(_chico["PSI deciles"], _chico["8·JS"], rtol=0.01)
    assert np.allclose(_chico["PSI deciles"], _chico["8·H²"], rtol=0.01)
    assert (equivalencia_local["PSI deciles"] <= equivalencia_local["J continuo"] + 1e-12).all()
    assert (equivalencia_local["PSI deciles"] <= equivalencia_local["PSI 50 bins"] + 1e-12).all()
    _e50 = props_normal(50)
    assert np.isclose(_e50.sum(), 1.0)
    # 9. Nula: la aproximación χ² acierta el p95 dentro de ±20% desde n = 300
    _t = tabla_nula.loc[[300, 1000, 3000, 10000, 30000]]
    assert np.all(np.abs(_t["p95 sim"] / _t["p95 χ²"] - 1) < 0.20)
    assert np.all(np.abs(_t["E[PSI] sim"] / _t["E[PSI] ≈ (B-1)c"] - 1) < 0.15)
    # 10. Regla de la cadena exacta para el bin de moda
    assert np.isclose(check_cadena[0], check_cadena[1], rtol=1e-10)
    # 11. Masa de ceros ≥ 90%: el psi() del curso devuelve NaN; binear() sí mide
    _r2 = np.random.default_rng(1)
    _z_e, _z_a = mora_corta(3000, 0.93, _r2), mora_corta(3000, 0.90, _r2)
    assert np.isnan(psi_curso(_z_e, _z_a))
    # 12. canal: PSI observado cerca del teórico 0,287 y en rojo; concept drift invisible al PSI
    assert estabilidad.loc["canal", "psi_ttd"] > 0.25
    assert taxonomia.loc["concept drift: +0,6 log-odds", "PSI score (deciles)"] < 0.01
    assert taxonomia.loc["concept drift: +0,6 log-odds", "tasa de malos"] > 0.12
    assert x_identica
    assert np.isclose(ciegas.iloc[0, 0], ciegas.iloc[1, 0], rtol=1e-12)
    assert ciegas.iloc[0]["PD verdadera media (OOT)"] > ciegas.iloc[1]["PD verdadera media (OOT)"] + 0.02
    # 13. Descomposición aditiva exacta del Δ score medio
    assert np.isclose(delta_score_medio, suma_delta_puntos, atol=1e-8)
    # 14. Cancelación (con los valores por defecto μ=0,4, w2=1): CSI > 0,10 y score bajo 0,10
    if np.isclose(info_cancel["mu"], 0.4) and np.isclose(info_cancel["w2"], 1.0) and info_cancel["modo"].startswith("lineales"):
        assert cancelacion.loc["x1", "CSI (5 bins)"] > 0.10 and cancelacion.loc["x2", "CSI (5 bins)"] > 0.10
        assert cancelacion.loc["score (deciles)", "CSI (5 bins)"] < 0.10
    "✅ Todos los checks del módulo pasan"
    return


if __name__ == "__main__":
    app.run()

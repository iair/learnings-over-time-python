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
    import copy
    import hashlib
    import hmac
    import json
    import math
    import platform
    import unicodedata
    import uuid
    import scipy
    import sklearn
    import statsmodels
    import statsmodels.api as sm
    from scipy.optimize import brentq
    from scipy.special import expit
    from scipy.stats import binomtest
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    return (
        HistGradientBoostingClassifier,
        binomtest,
        brentq,
        copy,
        expit,
        hashlib,
        hmac,
        json,
        math,
        mo,
        platform,
        plt,
        roc_auc_score,
        scipy,
        sklearn,
        sm,
        statsmodels,
        unicodedata,
        uuid,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # M22 · Gobierno de modelos: expediente, trazabilidad, model card y validación independiente

    **Serie 2 · Del embudo al gobierno.** Este notebook acompaña a `M22_gobierno_modelos.md`.

    La clase 6 cerró el curso con una frase: *el modelo que no está documentado no existe*. Aquí se
    construye, sobre la cartera sintética con **verdad conocida**, todo lo que convierte una corrida en
    **evidencia auditable**, y se mide qué garantiza (y qué no) cada capa:

    1. Una corrida mínima declarativa (config → datos → WoE → logística → δ → score) que **emite su
       propio audit trail** en JSONL.
    2. Canonicalización JSON: un serializador escrito desde cero vs `json.dumps` (y por qué `1` y `1.0`
       rompen hashes entre lenguajes).
    3. Cadena de hashes $h_i=\text{SHA-256}(h_{i-1}\,\|\,c(e_i))$, verificación y **seis ataques**
       contra cuatro defensas (cadena, cierre + numeración, HMAC, sello externo).
    4. Un sello externo al estilo RFC 3161 (simulado) y por qué el sello local no basta.
    5. Árbol de Merkle RFC 6962: dos implementaciones (recursiva e iterativa), prueba de inclusión y la
       mutación de Bitcoin (duplicar la última hoja).
    6. Lineage bundle y determinismo; **hash de un DataFrame**: `pandas.util.hash_pandas_object` vs
       serialización canónica propia, frente a nueve perturbaciones.
    7. Model card generado **desde la corrida** (limitaciones declaradas vs automáticas; usos no previstos).
    8. RACI como restricción verificable; gatillos con `disparado` calculado.
    9. Overrides: tres políticas de excepción y qué dice su desempeño.
    10. Validación independiente mínima: replicación, *challenger* y sensibilidad.
    11. Inventario y *tiering*; checks del módulo.

    Convenciones del curso: target **1 = malo**; WoE = ln(%buenos/%malos) → β negativos; PDO 20,
    score 600 a odds 50:1 (factor 28,8539; offset 487,1229).
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
    ## 1. La corrida como función declarativa que deja rastro

    La receta vive en un **config** (datos, semilla, candidatas, umbrales, escala, calibración,
    política). La corrida es una función pura `config → (artefacto, métricas, trail)`: lo único que
    cambia entre dos ejecuciones con el mismo config es el `run_id`. Cada etapa **registra** eventos en un
    trail (`run_start`, `artifact`, `decision`, `run_end`) igual que `nikodym`; la librería registra y la
    institución **encadena y sella** después (secciones 3–5), como en el paso 6 de la demo de clase.

    Qué mirar: el `config_hash` identifica la receta; el `data_hash` los datos efectivos; el número de
    eventos y decisiones es la historia de la corrida. Nada de esto dice todavía si el modelo es bueno.
    """)
    return


@app.cell
def _(hashlib, json):
    def canon(obj):
        """Serialización canónica: mismas claves, mismo orden, mismos bytes. Rechaza NaN/Infinity."""
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    def sha256_hex(texto):
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()

    def hash_obj(obj):
        return sha256_hex(canon(obj))

    CONFIG = {
        "modelo_id": "sintetico-scorecard-consumo",
        "version": "1.0.0",
        "root_seed": 20240706,
        "datos": {"generador": "generar_cartera", "n": 24_000, "deterioro": 0.35,
                  "drift_canal": True},
        "candidatas": ["uso_linea_prom_12m", "uso_tc_prom_12m", "uso_tc_prom_3m",
                       "meses_desde_mora_12m", "antiguedad_meses", "edad", "renta_mm",
                       "deuda_otras_prom_12m", "carga_financiera", "consultas_6m", "canal"],
        "seleccion": {"iv_min": 0.10, "corr_woe_max": 0.70},
        "escalado": {"pdo": 20, "score_base": 600, "odds_base": 50},
        "calibracion": {"ancla": "DEV+HO (cohortes <= 2024-12)",
                        "metodo": "brentq: media PD DEV = TC"},
        "politica": {"cutoff": 540, "regla": "aprobar si score >= cutoff"},
        "gobierno": {
            "proposito": "Admisión de crédito de consumo (solicitudes nuevas), PD a 12 meses.",
            "usuarios": "Motor de originación (automático) y mesa de excepciones (manual).",
            "usos_no_previstos": [
                "Provisiones (Compendio B-1 / IFRS 9): el horizonte y la población no son los de "
                "una provisión; el modelo no fue validado para ese uso.",
                "Cobranza o gestión de cartera vigente: es un modelo de admisión, sin variables "
                "de comportamiento post-desembolso.",
                "Precios diferenciados por riesgo: no se validó la calibración por tramo fino ni el "
                "efecto de selección que induce el precio.",
                "Otros productos (hipotecario, pymes, motos con prenda): población y definición de "
                "default distintas.",
            ],
            "responsable": "Equipo de Modelos (desarrollo)",
            "revision_meses": 12,
        },
    }
    return CONFIG, canon, hash_obj, sha256_hex


@app.cell
def _(canon, hashlib, math, np, pd, unicodedata):
    def hash_df_pandas(df):
        """Hash de la clase 6 (demo bases, celda 18): esquema + hash_pandas_object(index=True)."""
        esquema = [{"pos": i, "nombre": str(c), "dtype": str(df[c].dtype)}
                   for i, c in enumerate(df.columns)]
        h = hashlib.sha256()
        h.update(canon(esquema).encode("utf-8"))
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
        return h.hexdigest()

    def _token_num(v):
        # float(v) + 0.0 convierte -0.0 en 0.0 (IEEE-754, redondeo al más cercano);
        # repr(float) es la representación más corta que reconstruye el mismo double.
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "null"
        f = float(v) + 0.0
        if math.isinf(f):
            raise ValueError("Infinity no es un valor canónico")
        return repr(f)

    def _token_txt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
            return "null"
        return canon(unicodedata.normalize("NFC", str(v)))

    def hash_df_canonico(df, clave="id"):
        """Hash por CONTENIDO: filas ordenadas por `clave`, columnas por nombre, números como
        double (repr), -0.0 → 0.0, NaN/None → null, texto en NFC. Ignora índice y dtype físico."""
        d = df.sort_values(clave, kind="mergesort") if clave else df
        columnas = sorted(map(str, d.columns))
        h = hashlib.sha256()
        h.update(canon({"columnas": columnas, "n_filas": int(len(d))}).encode("utf-8"))
        for c in columnas:
            s = d[c]
            es_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if es_num:
                vals = s.to_numpy(dtype=float, na_value=np.nan)
                fichas = [_token_num(v) for v in vals.tolist()]
            else:
                fichas = [_token_txt(v) for v in s.astype(object).tolist()]
            h.update(("\n" + canon(c) + ":[" + ",".join(fichas) + "]").encode("utf-8"))
        return h.hexdigest()
    return hash_df_canonico, hash_df_pandas


@app.cell
def _(canon):
    class Trail:
        """Registro de la corrida (lo que hace la librería): eventos con reloj lógico, sin hashes."""

        def __init__(self, run_id):
            self.run_id = run_id
            self.eventos = []

        def registrar(self, tipo, paso, evento, **payload):
            e = {"n": len(self.eventos) + 1, "run_id": self.run_id, "t": len(self.eventos),
                 "tipo": tipo, "paso": paso, "evento": evento, "payload": payload}
            canon(e)  # falla aquí si el payload trae NaN/inf o tipos no serializables
            self.eventos.append(e)
            return e

        def jsonl(self):
            return "\n".join(canon(e) for e in self.eventos)
    return (Trail,)


@app.cell
def _(np):
    def gini_numpy(y, s):
        """Gini = 2·AUC − 1 con AUC de Mann-Whitney por rangos promedio (s alto = más riesgo)."""
        y = np.asarray(y, float)
        s = np.asarray(s, float)
        orden = np.argsort(s, kind="mergesort")
        s_ord = s[orden]
        rangos = np.empty(len(s))
        i = 0
        while i < len(s):                       # rangos promedio para empates
            j = i
            while j + 1 < len(s) and s_ord[j + 1] == s_ord[i]:
                j += 1
            rangos[orden[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        n1 = y.sum()
        n0 = len(y) - n1
        auc = (rangos[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        return 2 * auc - 1

    def ks_numpy(y, s):
        y = np.asarray(y, float)
        orden = np.argsort(-np.asarray(s, float), kind="mergesort")
        cm = np.cumsum(y[orden]) / y.sum()
        cb = np.cumsum(1 - y[orden]) / (1 - y).sum()
        return float(np.max(np.abs(cm - cb)))

    def psi_numpy(esperado, actual, bins=10):
        cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, bins + 1)))
        cortes[0], cortes[-1] = -np.inf, np.inf
        e = np.histogram(esperado, cortes)[0] / len(esperado) + 1e-6
        a = np.histogram(actual, cortes)[0] / len(actual) + 1e-6
        return float(np.sum((a - e) * np.log(a / e)))
    return gini_numpy, ks_numpy, psi_numpy


@app.cell
def _(
    Trail,
    a_woe,
    binomtest,
    brentq,
    expit,
    generar_cartera,
    gini_numpy,
    hash_df_canonico,
    hash_df_pandas,
    hash_obj,
    ks_numpy,
    np,
    pd,
    psi_numpy,
    sm,
    tabla_woe,
):
    def correr_pipeline(config, run_id):
        """Corrida completa. Determinista dado `config`; solo `run_id` varía entre ejecuciones."""
        tr = Trail(run_id)
        cfg_hash = hash_obj(config)
        tr.registrar("run_start", "inicio", "corrida_iniciada", modelo=config["modelo_id"],
                     version=config["version"], config_hash=cfg_hash)
        d = config["datos"]
        df = generar_cartera(n=d["n"], semilla=config["root_seed"], deterioro=d["deterioro"],
                             drift_canal=d["drift_canal"])
        data_hash = hash_df_canonico(df)
        tr.registrar("artifact", "datos", "datos_cargados", n=int(len(df)),
                     n_columnas=int(df.shape[1]), data_hash=data_hash,
                     data_hash_pandas=hash_df_pandas(df))
        muestras = {m: df[df["muestra"] == m].reset_index(drop=True)
                    for m in ("DEV", "HO", "OOT", "TTD")}
        tr.registrar("decision", "particion", "particion_aplicada", regla="cohorte + azar 70/30",
                     n={m: int(len(v)) for m, v in muestras.items()},
                     tasa_malos={m: round(float(v["malo"].mean()), 6)
                                 for m, v in muestras.items() if m != "TTD"})
        dev = muestras["DEV"]
        # --- IV mínimo ---
        mapas, ivs = {}, {}
        for v in config["candidatas"]:
            tab, iv = tabla_woe(dev[v], dev["malo"])
            mapas[v] = {str(k): float(w) for k, w in tab["woe"].items()}
            ivs[v] = iv
            accion = "retener" if iv >= config["seleccion"]["iv_min"] else "descartar"
            tr.registrar("decision", "seleccion", "iv_minimo", variable=v, iv=round(iv, 6),
                         umbral=config["seleccion"]["iv_min"], accion=accion)
        pool = [v for v in sorted(ivs, key=ivs.get, reverse=True)
                if ivs[v] >= config["seleccion"]["iv_min"]]
        # --- correlación de WoE (voraz por IV) ---
        W = a_woe(dev, pool, dev, mapas)
        elegidas = []
        for v in pool:
            corr = [abs(float(np.corrcoef(W[v], W[k])[0, 1])) for k in elegidas]
            cmax = max(corr) if corr else 0.0
            accion = "retener" if cmax <= config["seleccion"]["corr_woe_max"] else "descartar"
            tr.registrar("decision", "seleccion", "correlacion_woe", variable=v,
                         corr_max=round(cmax, 6), umbral=config["seleccion"]["corr_woe_max"],
                         accion=accion)
            if accion == "retener":
                elegidas.append(v)
        # --- logística sobre WoE ---
        X = sm.add_constant(W[elegidas])
        modelo = sm.Logit(dev["malo"].values, X).fit(disp=0)
        beta = {k: float(b) for k, b in modelo.params.items()}
        n_malos = int(dev["malo"].sum())
        tr.registrar("artifact", "modelo", "modelo_ajustado", coeficientes=beta,
                     n_malos_dev=n_malos, malos_por_parametro=round(n_malos / (len(elegidas) + 1), 2))
        signos_ok = all(beta[v] < 0 for v in elegidas)
        tr.registrar("decision", "modelo", "signos_woe", regla="beta < 0 con WoE = ln(%b/%m)",
                     todos_negativos=bool(signos_ok),
                     accion="continuar" if signos_ok else "revisar")
        # --- escala y calibración ---
        e = config["escalado"]
        factor = e["pdo"] / np.log(2)
        offset = e["score_base"] - factor * np.log(e["odds_base"])

        def pred_lineal(d_):
            return beta["const"] + a_woe(d_, elegidas, dev, mapas).values @ np.array(
                [beta[v] for v in elegidas])

        lp = {m: pred_lineal(v) for m, v in muestras.items()}
        ancla = pd.concat([muestras["DEV"], muestras["HO"]])
        tc = float(ancla["malo"].mean())
        delta = float(brentq(lambda x: expit(lp["DEV"] + x).mean() - tc, -5, 5, xtol=1e-14))
        tr.registrar("decision", "calibracion", "delta_calibrado", tc=round(tc, 8),
                     delta=round(delta, 10), ancla=config["calibracion"]["ancla"])
        artefacto = {
            "modelo_id": config["modelo_id"], "version": config["version"],
            "variables": elegidas,
            "woe": {v: mapas[v] for v in elegidas},
            "coeficientes": beta,
            "escalado": {"factor": float(factor), "offset": float(offset), **e},
            "calibracion": {"tc": tc, "delta": delta},
            "politica": config["politica"],
            "nota": "cortes de binning: se aplican con binear(ref=DEV) en este módulo; "
                    "el congelado explícito de cortes es M21",
        }
        hash_artefacto = hash_obj(artefacto)
        tr.registrar("artifact", "modelo", "artefacto_congelado", hash_artefacto=hash_artefacto,
                     n_variables=len(elegidas))
        # --- desempeño y estabilidad ---
        pdc = {m: expit(lp[m] + delta) for m in muestras}
        score = {m: offset - factor * (lp[m] + delta) for m in muestras}
        met = {}
        for m in ("DEV", "HO", "OOT"):
            y = muestras[m]["malo"].values
            met[m] = {"n": int(len(y)), "malos": int(y.sum()),
                      "gini": float(gini_numpy(y, lp[m])), "ks": float(ks_numpy(y, lp[m])),
                      "pd_media": float(pdc[m].mean()), "tasa_obs": float(y.mean())}
        psi = {"DEV_OOT": psi_numpy(score["DEV"], score["OOT"]),
               "DEV_TTD": psi_numpy(score["DEV"], score["TTD"])}
        tr.registrar("artifact", "validacion", "metricas", gini={m: round(met[m]["gini"], 6)
                     for m in met}, ks={m: round(met[m]["ks"], 6) for m in met},
                     psi={k: round(v, 6) for k, v in psi.items()})
        y_oot = muestras["OOT"]["malo"].values
        bt = binomtest(int(y_oot.sum()), len(y_oot), float(pdc["OOT"].mean()))
        decision_bt = "pass" if bt.pvalue >= 0.05 else "fail"
        tr.registrar("decision", "validacion", "backtesting_oot", test="binomial global",
                     pd_media=round(float(pdc["OOT"].mean()), 6),
                     tasa_obs=round(float(y_oot.mean()), 6), p_value=float(bt.pvalue),
                     decision=decision_bt)
        # --- scoring del lote TTD ---
        cut = config["politica"]["cutoff"]
        salida = pd.DataFrame({"id": muestras["TTD"]["id"].values,
                               "score": np.round(score["TTD"], 6),
                               "pd_calibrada": np.round(pdc["TTD"], 8)})
        salida["decision"] = np.where(salida["score"] >= cut, "aprobar", "rechazar")
        tr.registrar("artifact", "scoring", "lote_ttd_puntuado", n=int(len(salida)),
                     tasa_aprobacion=round(float((salida["decision"] == "aprobar").mean()), 6),
                     hash_salida=hash_df_canonico(salida))
        tr.registrar("run_end", "fin", "corrida_terminada", estado="done")
        return {"trail": tr, "df": df, "muestras": muestras, "mapas": mapas, "ivs": ivs,
                "elegidas": elegidas, "beta": beta, "modelo": modelo, "factor": factor,
                "offset": offset, "tc": tc, "delta": delta, "lp": lp, "pdc": pdc,
                "score": score, "metricas": met, "psi": psi, "backtesting_p": float(bt.pvalue),
                "backtesting": decision_bt, "salida": salida, "artefacto": artefacto,
                "hash_artefacto": hash_artefacto, "config_hash": cfg_hash,
                "data_hash": data_hash}
    return (correr_pipeline,)


@app.cell
def _(CONFIG, correr_pipeline, uuid):
    RUN_ID = uuid.uuid4().hex
    corrida = correr_pipeline(CONFIG, RUN_ID)
    return RUN_ID, corrida


@app.cell
def _(corrida, mo, pd):
    _tr = corrida["trail"]
    _conteo = pd.Series([e["tipo"] for e in _tr.eventos]).value_counts()
    _dec = pd.DataFrame([{"n": e["n"], "evento": e["evento"],
                          "variable": e["payload"].get("variable", "—"),
                          "accion": e["payload"].get("accion", e["payload"].get("decision", "—"))}
                         for e in _tr.eventos if e["tipo"] == "decision"])
    _met = pd.DataFrame(corrida["metricas"]).T[["n", "malos", "gini", "ks", "pd_media", "tasa_obs"]]
    _lineas = _tr.jsonl().splitlines()
    mo.vstack([
        mo.md(f"""
    **Corrida** `{corrida['trail'].run_id}` · `config_hash` `{corrida['config_hash'][:16]}…` ·
    `data_hash` `{corrida['data_hash'][:16]}…` · artefacto `{corrida['hash_artefacto'][:16]}…`

    **{len(_tr.eventos)} eventos** ({', '.join(f'{k}: {v}' for k, v in _conteo.items())}).
    Variables elegidas: {', '.join(corrida['elegidas'])} · TC = {corrida['tc']:.4%} ·
    δ = {corrida['delta']:+.4f} · cutoff {corrida['artefacto']['politica']['cutoff']}.

    Primeras dos líneas del JSONL (cada línea es un evento canónico):
    """),
        mo.md("```\n" + "\n".join(l[:230] + ("…" if len(l) > 230 else "") for l in _lineas[:2])
              + "\n```"),
        mo.md("**Decisiones registradas** (lo que un validador lee para reconstruir el embudo):"),
        _dec,
        mo.md("**Objetos de comité** (Gini y KS sobre el predictor lineal; PD calibrada vs tasa observada):"),
        _met.round(4),
    ])
    return


@app.cell
def _(corrida, mo):
    _m = corrida["metricas"]
    mo.md(f"""
    **Lectura.** El ranking se sostiene fuera de desarrollo (Gini DEV {_m['DEV']['gini']:.3f} ·
    HO {_m['HO']['gini']:.3f} · OOT {_m['OOT']['gini']:.3f}) y el PSI del score DEV→OOT es
    {corrida['psi']['DEV_OOT']:.4f}, pero el **nivel** falla en OOT: PD media
    {_m['OOT']['pd_media']:.2%} contra mora observada {_m['OOT']['tasa_obs']:.2%}, binomial global
    p = {corrida['backtesting_p']:.2g} → `{corrida['backtesting']}`. Es el mismo patrón que Banco
    Austral en la clase 6 (Gini HO 0,695 vs OOT 0,694, PSI 0,008, calibración OOT roja), aquí
    **plantado** por el generador (`deterioro` = 0,35 en el log-odds de 2025). La corrida terminó
    (`done`), está documentada y es reproducible — **y eso no la aprueba**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Canonicalización: el hash es de *bytes*, no de *objetos*

    SHA-256 no sabe qué es un diccionario. Dos objetos «iguales» producen el mismo hash **solo si se
    serializan a los mismos bytes**. Una canonicalización es una función $c$ de objetos a bytes con la
    propiedad $c(a)=c(b) \iff a \equiv b$ para la equivalencia que declaramos (mismas claves y valores,
    sin importar el orden de inserción).

    Dos implementaciones: (1) un serializador **desde cero** (recursivo, reglas explícitas de escape y
    de números); (2) `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    allow_nan=False)`. Se comparan byte a byte sobre el artefacto, todo el trail y una batería de casos
    borde. Después, cuatro trampas: orden de claves, `1` vs `1.0`, NaN y Unicode.
    """)
    return


@app.cell
def _(math):
    _ESCAPES = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t",
                "\b": "\\b", "\f": "\\f"}

    def _cadena(s):
        partes = []
        for ch in s:
            if ch in _ESCAPES:
                partes.append(_ESCAPES[ch])
            elif ord(ch) < 0x20:
                partes.append("\\u%04x" % ord(ch))
            else:
                partes.append(ch)          # ensure_ascii=False: UTF-8 tal cual
        return '"' + "".join(partes) + '"'

    def canon_desde_cero(obj):
        """Serializador canónico escrito a mano (mismas reglas que json.dumps canónico)."""
        if obj is None:
            return "null"
        if obj is True:
            return "true"
        if obj is False:
            return "false"
        if isinstance(obj, int):            # bool ya salió arriba: bool es subclase de int
            return str(obj)
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                raise ValueError("NaN/Infinity no son JSON válido")
            return repr(obj)                # representación más corta que reconstruye el double
        if isinstance(obj, str):
            return _cadena(obj)
        if isinstance(obj, (list, tuple)):
            return "[" + ",".join(canon_desde_cero(x) for x in obj) + "]"
        if isinstance(obj, dict):
            claves = sorted(obj)            # orden por punto de código (como sort_keys)
            return "{" + ",".join(_cadena(k) + ":" + canon_desde_cero(obj[k]) for k in claves) + "}"
        raise TypeError(f"tipo no canonicalizable: {type(obj).__name__}")
    return (canon_desde_cero,)


@app.cell
def _(canon, canon_desde_cero, corrida, json, mo, sha256_hex, unicodedata):
    _casos = [corrida["artefacto"]] + corrida["trail"].eventos + [
        {"b": 1, "a": [1.5, -0.0, 1e21, 1e-7, 0.1 + 0.2], "ñ": "Concepción\n\t\"x\"\\", "c": None,
         "d": True, "e": {"z": "\x01", "y": []}},
    ]
    coincide_canon = all(canon(x) == canon_desde_cero(x) for x in _casos)

    _d1 = {"variable": "uso_linea", "iv": 0.80}
    _d2 = {"iv": 0.80, "variable": "uso_linea"}
    _nfc = unicodedata.normalize("NFC", "Concepción")
    _nfd = unicodedata.normalize("NFD", "Concepción")
    trampas_canon = [
        {"trampa": "mismo dict, otro orden de inserción",
         "sin canonicalizar": sha256_hex(json.dumps(_d1))[:12] + " vs " + sha256_hex(json.dumps(_d2))[:12],
         "canónico": sha256_hex(canon(_d1))[:12] + " vs " + sha256_hex(canon(_d2))[:12],
         "¿iguales en canónico?": canon(_d1) == canon(_d2)},
        {"trampa": "1 (int) vs 1.0 (float)",
         "sin canonicalizar": json.dumps(1) + " vs " + json.dumps(1.0),
         "canónico": canon(1) + " vs " + canon(1.0),
         "¿iguales en canónico?": canon(1) == canon(1.0)},
        {"trampa": "NaN en el payload",
         "sin canonicalizar": json.dumps(float("nan")),
         "canónico": "ValueError (allow_nan=False)",
         "¿iguales en canónico?": None},
        {"trampa": "Unicode NFC vs NFD («Concepción»)",
         "sin canonicalizar": f"{len(_nfc.encode())} vs {len(_nfd.encode())} bytes",
         "canónico": sha256_hex(canon(_nfc))[:12] + " vs " + sha256_hex(canon(_nfd))[:12],
         "¿iguales en canónico?": canon(_nfc) == canon(_nfd)},
    ]
    _nan_rechazado = False
    try:
        canon({"x": float("nan")})
    except ValueError:
        _nan_rechazado = True
    mo.vstack([
        mo.md(f"**Desde cero = `json.dumps` canónico, byte a byte, en {len(_casos)} objetos "
              f"(artefacto + trail + casos borde):** `{coincide_canon}` · NaN rechazado: "
              f"`{_nan_rechazado}`"),
        trampas_canon,
    ])
    return coincide_canon, trampas_canon


@app.cell
def _(mo):
    mo.md(r"""
    **Lectura.** El orden de claves se resuelve con `sort_keys`. Las otras tres trampas **no** las
    resuelve `json.dumps`: (i) Python escribe `1` y `1.0` distinto, mientras el esquema RFC 8785 (JCS)
    serializa números como ECMAScript y ambos serían `1` — un hash calculado en Python y otro en
    JavaScript/Java sobre «el mismo» evento **no coinciden** salvo que ambos implementen el mismo
    esquema; (ii) NaN no es JSON: `allow_nan=False` convierte un problema silencioso en un error;
    (iii) JCS tampoco normaliza Unicode: si el texto viene de dos sistemas (NFC y NFD), la
    normalización es una **política** que hay que declarar y aplicar antes del hash. La regla práctica:
    **el hash de identidad se calcula siempre con una sola función, versionada, y su definición viaja
    en el expediente**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Cadena de hashes, seis ataques y cuatro defensas

    Regla del curso: $h_0 = 0^{256}$, $\;h_i=\text{SHA-256}\big(h_{i-1}\,\|\,c(e_i)\big)$. Cambiar $e_k$
    cambia $h_k$ y, por arrastre, **todos** los $h_j$ con $j\ge k$. Pero quien puede reescribir el
    archivo puede **recalcular la cola**. Se modelan seis ataques y cuatro defensas:

    | Defensa | Qué verifica |
    |---|---|
    | Cadena | cada $h_i$ se recalcula desde los eventos |
    | Cadena + estructura | además: numeración $1..n$ sin huecos, un solo `run_id`, termina en `run_end` |
    | HMAC | $h_i=\text{HMAC}_K(h_{i-1}\,\|\,c(e_i))$ con clave $K$ que el atacante **no** tiene |
    | Sello externo | $(h_n, n)$ archivado fuera del log, con custodia separada |

    Elige el ataque y el evento atacado; la tabla de abajo se calcula para **todos** los ataques.
    """)
    return


@app.cell
def _(canon, hashlib, hmac):
    CERO = "0" * 64

    def encadenar(eventos, clave=None):
        """Lista de hashes encadenados. Con `clave`, HMAC-SHA256 en vez de SHA-256."""
        prev, hs = CERO, []
        for e in eventos:
            msg = (prev + canon(e)).encode("utf-8")
            prev = (hmac.new(clave, msg, hashlib.sha256).hexdigest() if clave
                    else hashlib.sha256(msg).hexdigest())
            hs.append(prev)
        return hs

    def verificar_cadena(eventos, hashes, clave=None):
        """(íntegra, primera posición 1-based donde falla)."""
        rehechos = encadenar(eventos, clave)
        if len(rehechos) != len(hashes):
            return False, min(len(rehechos), len(hashes)) + 1
        for i, (a, b) in enumerate(zip(rehechos, hashes)):
            if a != b:
                return False, i + 1
        return True, None

    def verificar_estructura(eventos):
        if not eventos:
            return False, "trail vacío"
        for k, e in enumerate(eventos, start=1):
            if e["n"] != k:
                return False, f"numeración rota en {k}"
            if e["run_id"] != eventos[0]["run_id"]:
                return False, f"evento {k} de otra corrida"
        if eventos[-1]["tipo"] != "run_end":
            return False, "no termina en run_end (truncado)"
        return True, None

    def verificar_sello(eventos, sello):
        """Recalcula la cadena DESDE LOS EVENTOS y compara (hash terminal, n) con el sello."""
        hs = encadenar(eventos)
        return bool(hs) and hs[-1] == sello["hash_terminal"] and len(hs) == sello["n_eventos"]
    return CERO, encadenar, verificar_cadena, verificar_estructura, verificar_sello


@app.cell
def _(corrida, encadenar):
    EVENTOS = corrida["trail"].eventos
    HASHES = encadenar(EVENTOS)
    CLAVE_HMAC = b"clave-institucional-custodiada-por-seguridad"   # didáctica: en producción, un KMS/HSM
    HASHES_HMAC = encadenar(EVENTOS, CLAVE_HMAC)
    SELLO = {"hash_terminal": HASHES[-1], "n_eventos": len(HASHES), "run_id": EVENTOS[0]["run_id"]}
    return CLAVE_HMAC, EVENTOS, HASHES, HASHES_HMAC, SELLO


@app.cell
def _(EVENTOS, mo):
    selector_ataque = mo.ui.dropdown(
        options=["torpe: editar sin recalcular", "prolijo: editar y recalcular",
                 "borrar un evento y recalcular", "truncar la cola (sin recalcular)",
                 "reordenar dos eventos y recalcular", "insertar un evento y recalcular"],
        value="prolijo: editar y recalcular", label="Ataque")
    slider_evento = mo.ui.slider(2, len(EVENTOS) - 1, value=min(12, len(EVENTOS) - 1),
                                 label="Evento atacado (posición)")
    mo.hstack([selector_ataque, slider_evento])
    return selector_ataque, slider_evento


@app.cell
def _(copy, encadenar):
    def atacar(eventos, hashes, tipo, pos):
        """Devuelve (eventos, hashes) adulterados. `pos` es 1-based. El atacante no tiene la
        clave HMAC: cuando «recalcula», recalcula SHA-256 (lo único que puede hacer)."""
        ev = copy.deepcopy(eventos)
        k = pos - 1
        if tipo.startswith("torpe"):
            ev[k]["payload"]["_adulterado"] = "aprobar_sin_control"
            return ev, list(hashes)
        if tipo.startswith("prolijo"):
            ev[k]["payload"]["_adulterado"] = "aprobar_sin_control"
        elif tipo.startswith("borrar"):
            del ev[k]
        elif tipo.startswith("truncar"):
            return ev[:pos], list(hashes[:pos])  # conserva los primeros `pos` eventos
        elif tipo.startswith("reordenar"):
            ev[k], ev[k + 1] = ev[k + 1], ev[k]
        elif tipo.startswith("insertar"):
            nuevo = copy.deepcopy(ev[k])
            nuevo["evento"] = "decision_fabricada"
            nuevo["payload"] = {"accion": "excepcion_aprobada"}
            ev.insert(k, nuevo)
        for i, e in enumerate(ev, start=1):       # el atacante prolijo también renumera
            e["n"], e["t"] = i, i - 1
        return ev, encadenar(ev)
    return (atacar,)


@app.cell
def _(
    CLAVE_HMAC,
    EVENTOS,
    HASHES,
    HASHES_HMAC,
    SELLO,
    atacar,
    pd,
    verificar_cadena,
    verificar_estructura,
    verificar_sello,
):
    def evaluar_ataque(tipo, pos):
        ev, hs = atacar(EVENTOS, HASHES, tipo, pos)
        ok_c, donde = verificar_cadena(ev, hs)
        ok_e, _motivo = verificar_estructura(ev)
        # despliegue con HMAC: el log guarda HASHES_HMAC; el atacante, sin clave, solo sabe
        # recalcular SHA-256
        ev_h, hs_h = atacar(EVENTOS, HASHES_HMAC, tipo, pos)
        ok_h, _ = verificar_cadena(ev_h, hs_h, CLAVE_HMAC)
        return {"ataque": tipo,
                "cadena": "✅ pasa" if ok_c else f"❌ detecta (evento {donde})",
                "cadena + estructura": "✅ pasa" if (ok_c and ok_e) else "❌ detecta",
                "HMAC (sin clave)": "✅ pasa" if ok_h else "❌ detecta",
                "sello externo": "✅ pasa" if verificar_sello(ev, SELLO) else "❌ detecta"}

    TIPOS_ATAQUE = ["torpe: editar sin recalcular", "prolijo: editar y recalcular",
                    "borrar un evento y recalcular", "truncar la cola (sin recalcular)",
                    "reordenar dos eventos y recalcular", "insertar un evento y recalcular"]
    matriz_ataques = pd.DataFrame([evaluar_ataque(t, 12) for t in TIPOS_ATAQUE]).set_index("ataque")
    return evaluar_ataque, matriz_ataques


@app.cell
def _(evaluar_ataque, matriz_ataques, mo, selector_ataque, slider_evento):
    _r = evaluar_ataque(selector_ataque.value, slider_evento.value)
    mo.vstack([
        mo.md(f"**Ataque elegido** «{selector_ataque.value}» sobre el evento {slider_evento.value}: "
              f"cadena {_r['cadena']} · estructura {_r['cadena + estructura']} · "
              f"HMAC {_r['HMAC (sin clave)']} · sello {_r['sello externo']}"),
        mo.md("**Matriz completa** (evento 12; ✅ pasa = el ataque NO se detecta con esa defensa):"),
        matriz_ataques,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Lectura.** La cadena sola solo caza al atacante **torpe**. El truncamiento de la cola deja un
    prefijo que es una cadena perfectamente válida —también con HMAC: los hashes del prefijo son
    auténticos—; lo cazan la **estructura** (falta el `run_end`) y el **sello** (el número de eventos).
    Todo ataque que recalcula la cola pasa la cadena y la estructura (el atacante prolijo renumera) y
    solo lo detectan (a) una **clave** que el atacante no tiene (HMAC) o (b) un **ancla** fuera de su
    alcance (el sello, que se compara contra la cadena **recalculada desde los eventos**, no contra los
    hashes guardados: si no, el ataque torpe lo pasaría). Ninguna de las cuatro defensas dice
    si la decisión registrada era **correcta**: prueban integridad del registro, no calidad del modelo.
    Y el HMAC no elimina el problema de custodia: lo **traslada** a la clave (quien la tenga puede
    reescribir todo); por eso se combina con el sello.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. El sello externo y el sello que no sirve

    El sello del curso es $(h_n, n)$ archivado **fuera** del log. Si el atacante prolijo puede escribir
    también el archivo del sello, lo reescribe y todo cuadra. La defensa real es que el sello quede
    **firmado por un tercero** que el equipo no controla. RFC 3161 (Time-Stamp Protocol, 2001) define
    exactamente eso: el solicitante envía solo el **hash** (`messageImprint`), la autoridad de sellado
    (TSA) devuelve un token firmado con `genTime`, número de serie, política y el mismo hash. La TSA no
    ve el contenido.

    Aquí se **simula** una TSA con HMAC (clave de la TSA). Diferencia importante: una TSA real firma con
    clave **asimétrica** (certificado X.509), así cualquiera verifica con la clave pública y la TSA no
    puede negar haber firmado; un HMAC exige compartir la clave para verificar. La simulación sirve para
    mostrar **qué capa detecta qué**, no como implementación.
    """)
    return


@app.cell
def _(EVENTOS, HASHES, SELLO, atacar, canon, hashlib, hmac):
    _CLAVE_TSA = b"clave-privada-de-la-TSA (simulada)"

    def emitir_token(hash_hex, serial, gen_time="2026-09-28T12:00:00Z"):
        tst_info = {"version": 1, "policy": "1.3.6.1.4.1.99999.1 (ficticia)",
                    "messageImprint": {"hashAlgorithm": "sha256", "hashedMessage": hash_hex},
                    "serialNumber": serial, "genTime": gen_time, "tsa": "TSA simulada"}
        firma = hmac.new(_CLAVE_TSA, canon(tst_info).encode(), hashlib.sha256).hexdigest()
        return {"tstInfo": tst_info, "firma": firma}

    def verificar_token(token, hash_hex):
        ok_firma = hmac.compare_digest(
            token["firma"],
            hmac.new(_CLAVE_TSA, canon(token["tstInfo"]).encode(), hashlib.sha256).hexdigest())
        return ok_firma and token["tstInfo"]["messageImprint"]["hashedMessage"] == hash_hex

    TOKEN_TSA = emitir_token(hashlib.sha256(canon(SELLO).encode()).hexdigest(), serial=1)

    # El atacante prolijo reescribe el trail y decide qué sello presentar.
    _ev_p, _hs_p = atacar(EVENTOS, HASHES, "prolijo: editar y recalcular", 12)
    _sello_falso = {"hash_terminal": _hs_p[-1], "n_eventos": len(_hs_p), "run_id": SELLO["run_id"]}

    def _fila(nombre, hs, sello_presentado):
        cuadra_local = hs[-1] == sello_presentado["hash_terminal"] and len(hs) == sello_presentado["n_eventos"]
        cuadra_tsa = verificar_token(TOKEN_TSA, hashlib.sha256(canon(sello_presentado).encode()).hexdigest())
        return {"escenario": nombre, "trail ↔ sello presentado": cuadra_local,
                "sello presentado ↔ token TSA": cuadra_tsa,
                "resultado": "aceptado" if (cuadra_local and cuadra_tsa) else "RECHAZADO"}

    capas_sello = [
        _fila("trail íntegro · sello original", HASHES, SELLO),
        _fila("prolijo · presenta el sello original", _hs_p, SELLO),
        _fila("prolijo · reescribe también el sello local", _hs_p, _sello_falso),
    ]
    return TOKEN_TSA, capas_sello, emitir_token, verificar_token


@app.cell
def _(TOKEN_TSA, capas_sello, mo, pd):
    mo.vstack([
        mo.md("**Token (simulado) que devuelve la TSA** — solo contiene el hash del sello, no el trail:"),
        mo.md("```\n" + str(TOKEN_TSA["tstInfo"]) + "\n```"),
        pd.DataFrame(capas_sello).set_index("escenario"),
        mo.md(r"""
    **Lectura.** Si el atacante reescribe el trail **y** el sello local, el sello local «cuadra»
    (tercera fila): un sello guardado donde el mismo equipo puede escribir es decoración. El token de
    la TSA no cuadra porque firma el hash **original** en una fecha y el atacante no puede emitir otro
    token con la fecha antigua. Qué **no** prueba el token: que el log esté completo antes de sellarse,
    que el contenido sea verdadero o que la decisión sea correcta. Prueba *existencia de esos bytes
    antes de `genTime`*.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Árbol de Merkle: verificar **un** evento sin mostrar los demás

    La cadena obliga a recorrer los $n$ eventos para verificar cualquiera. Un árbol de Merkle
    (RFC 6962, *Certificate Transparency*) resume $n$ hojas en una raíz, y una **prueba de inclusión**
    de $\lceil\log_2 n\rceil$ hashes basta para demostrar que el evento $m$ está en el log sellado:

    $$\text{MTH}(\{d_0\})=H(\texttt{0x00}\,\|\,d_0),\qquad
    \text{MTH}(D_n)=H\big(\texttt{0x01}\,\|\,\text{MTH}(D_{0:k})\,\|\,\text{MTH}(D_{k:n})\big),$$

    con $k$ la mayor potencia de 2 **menor** que $n$. Los prefijos `0x00`/`0x01` separan hojas de nodos
    internos (evitan que un nodo interno se haga pasar por hoja).

    Dos implementaciones: (1) la definición **recursiva** de RFC 6962; (2) una **iterativa por niveles**
    que promueve el nodo impar sin duplicarlo. Deben coincidir para todo $n$. Una tercera variante —
    **duplicar** la última hoja cuando el nivel es impar, como el árbol de transacciones de Bitcoin —
    produce otra raíz y admite una **mutación**: dos logs distintos con la misma raíz.
    """)
    return


@app.cell
def _(hashlib):
    def _h(b):
        return hashlib.sha256(b).digest()

    def hoja(d):
        return _h(b"\x00" + d)

    def nodo(izq, der):
        return _h(b"\x01" + izq + der)

    def _k(n):
        k = 1
        while k * 2 < n:
            k *= 2
        return k

    def mth(datos):
        """Merkle Tree Hash recursivo (RFC 6962 §2.1)."""
        n = len(datos)
        if n == 0:
            return _h(b"")
        if n == 1:
            return hoja(datos[0])
        k = _k(n)
        return nodo(mth(datos[:k]), mth(datos[k:]))

    def raiz_iterativa(datos):
        """Por niveles; el nodo impar sube sin duplicarse (equivale a RFC 6962)."""
        if not datos:
            return _h(b"")
        nivel = [hoja(d) for d in datos]
        while len(nivel) > 1:
            sig = [nodo(nivel[i], nivel[i + 1]) for i in range(0, len(nivel) - 1, 2)]
            if len(nivel) % 2 == 1:
                sig.append(nivel[-1])
            nivel = sig
        return nivel[0]

    def raiz_duplicando(datos):
        """Variante estilo Bitcoin: si el nivel es impar, se DUPLICA el último nodo."""
        nivel = [hoja(d) for d in datos]
        while len(nivel) > 1:
            if len(nivel) % 2 == 1:
                nivel = nivel + [nivel[-1]]
            nivel = [nodo(nivel[i], nivel[i + 1]) for i in range(0, len(nivel), 2)]
        return nivel[0]

    def prueba_inclusion(datos, m):
        """PATH(m, D[n]) de RFC 6962 §2.1.1: hashes hermanos desde la hoja hasta la raíz."""
        n = len(datos)
        if n <= 1:
            return []
        k = _k(n)
        if m < k:
            return prueba_inclusion(datos[:k], m) + [mth(datos[k:])]
        return prueba_inclusion(datos[k:], m - k) + [mth(datos[:k])]

    def verificar_inclusion(d, m, n, camino, raiz):
        """Algoritmo de verificación de RFC 9162 §2.1.3.2 (índice m, tamaño n)."""
        if m >= n:
            return False
        fn, sn, r = m, n - 1, hoja(d)
        for p in camino:
            if sn == 0:
                return False
            if (fn & 1) or fn == sn:
                r = nodo(p, r)
                if not (fn & 1):
                    while not (fn & 1) and fn != 0:
                        fn >>= 1
                        sn >>= 1
            else:
                r = nodo(r, p)
            fn >>= 1
            sn >>= 1
        return sn == 0 and r == raiz
    return (
        mth,
        prueba_inclusion,
        raiz_duplicando,
        raiz_iterativa,
        verificar_inclusion,
    )


@app.cell
def _(mo):
    slider_hojas = mo.ui.slider(1, 400, value=341, label="n.º de hojas (eventos del log)")
    slider_hoja = mo.ui.slider(0, 399, value=200, label="hoja a probar (índice m)")
    mo.hstack([slider_hojas, slider_hoja])
    return slider_hoja, slider_hojas


@app.cell
def _(
    canon,
    corrida,
    math,
    mo,
    mth,
    prueba_inclusion,
    raiz_duplicando,
    raiz_iterativa,
    slider_hoja,
    slider_hojas,
    verificar_inclusion,
):
    # hojas: los eventos reales del trail, completados con eventos sintéticos hasta n
    _base = [canon(e).encode() for e in corrida["trail"].eventos]
    _n = slider_hojas.value
    hojas_demo = [_base[i] if i < len(_base) else canon({"n": i + 1, "evento": "relleno"}).encode()
                  for i in range(_n)]
    _m = min(slider_hoja.value, _n - 1)
    _raiz = mth(hojas_demo)
    _camino = prueba_inclusion(hojas_demo, _m)
    _ok = verificar_inclusion(hojas_demo[_m], _m, _n, _camino, _raiz)
    _falso = verificar_inclusion(hojas_demo[_m] + b" ", _m, _n, _camino, _raiz)
    _coinciden = _raiz == raiz_iterativa(hojas_demo)
    _dup = raiz_duplicando(hojas_demo)
    mo.md(f"""
    n = {_n} hojas · raíz RFC 6962 `{_raiz.hex()[:16]}…` · iterativa = recursiva: `{_coinciden}`

    Prueba de inclusión de la hoja m = {_m}: **{len(_camino)} hashes** (⌈log₂ n⌉ = {math.ceil(math.log2(_n)) if _n > 1 else 0}).
    Verifica: `{_ok}` · con la hoja alterada en un byte: `{_falso}`.
    Raíz «duplicando» (Bitcoin): `{_dup.hex()[:16]}…` → {'igual' if _dup == _raiz else 'distinta'} a RFC 6962
    ({'n es potencia de 2: ambas convenciones coinciden' if _n & (_n - 1) == 0 else 'n no es potencia de 2'}).

    Con los {len(_base)} eventos de nuestra corrida, un auditor que quiere verificar **una** decisión
    recibe {math.ceil(math.log2(len(_base)))} hashes en vez de los {len(_base)} eventos; con los 398
    eventos de la corrida nikodym de clase serían 9.
    """)
    return


@app.cell
def _(mth, raiz_duplicando, raiz_iterativa):
    # La mutación de la variante que duplica: [a, b, c] y [a, b, c, c] comparten raíz.
    _a, _b, _c = b"evento-a", b"evento-b", b"evento-c"
    mutacion_duplicando = raiz_duplicando([_a, _b, _c]) == raiz_duplicando([_a, _b, _c, _c])
    mutacion_rfc6962 = mth([_a, _b, _c]) == mth([_a, _b, _c, _c])
    iterativa_igual_recursiva = all(
        raiz_iterativa([bytes([i % 251]) * 3 for i in range(n)]) == mth([bytes([i % 251]) * 3 for i in range(n)])
        for n in range(0, 130))
    return iterativa_igual_recursiva, mutacion_duplicando, mutacion_rfc6962


@app.cell
def _(iterativa_igual_recursiva, mo, mutacion_duplicando, mutacion_rfc6962):
    mo.md(f"""
    **Mutación.** Con la variante que duplica, el log `[a, b, c]` y el log `[a, b, c, c]` tienen la
    **misma raíz**: `{mutacion_duplicando}`. Con RFC 6962: `{mutacion_rfc6962}`. Un auditor que solo
    compara raíces aceptaría un log con un evento repetido (por ejemplo, una aprobación contada dos
    veces). Recursiva = iterativa para n = 0…129: `{iterativa_igual_recursiva}`.

    **Lectura.** Merkle no reemplaza a la cadena: resuelve **otra** pregunta. La cadena responde
    «¿alguien tocó la historia?» recorriéndola entera; el árbol responde «¿este evento está en el log
    que se selló?» con una prueba corta y sin revelar los demás eventos (útil cuando el auditor no
    puede ver datos de clientes). La propiedad *append-only* entre dos raíces sucesivas se prueba con
    **pruebas de consistencia** (RFC 6962 §2.1.2), que aquí no se implementan.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Lineage: la identidad de la corrida, y el hash de un DataFrame

    El lineage es la ficha de procedencia: `run_id` (esta ejecución), `config_hash` (la receta),
    `data_hash` (los datos efectivos), `root_seed`, versiones de librerías, `git_sha` y hash del archivo
    de bloqueo de dependencias. Reproducir **no** es obtener el mismo `run_id`: es obtener los mismos
    hashes de contenido con la misma receta, datos, semilla y entorno.

    Primero, determinismo: se corre la receta **dos veces** con distinto `run_id` y se comparan las
    huellas de contenido del trail (eventos sin `run_id`).
    """)
    return


@app.cell
def _(
    CONFIG,
    corrida,
    correr_pipeline,
    hash_obj,
    mo,
    platform,
    scipy,
    sklearn,
    statsmodels,
    np,
    pd,
    uuid,
):
    def huella_contenido(eventos):
        """Hash del trail sin run_id: lo que debe repetirse entre corridas equivalentes."""
        return hash_obj([{k: v for k, v in e.items() if k != "run_id"} for e in eventos])

    corrida_2 = correr_pipeline(CONFIG, uuid.uuid4().hex)
    determinista = (huella_contenido(corrida["trail"].eventos)
                    == huella_contenido(corrida_2["trail"].eventos))

    lineage = {
        "run_id": corrida["trail"].run_id,
        "config_hash": corrida["config_hash"],
        "data_hash": corrida["data_hash"],
        "hash_artefacto": corrida["hash_artefacto"],
        "root_seed": CONFIG["root_seed"],
        "git_sha": None,            # este notebook es autocontenido: no consulta git
        "git_dirty": None,
        "lock_hash": None,          # no hay uv.lock / requirements con hashes en el entorno
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__,
                             "scikit-learn": sklearn.__version__,
                             "statsmodels": statsmodels.__version__,
                             "marimo": mo.__version__},
    }
    advertencias_determinismo = []
    if lineage["git_sha"] is None:
        advertencias_determinismo.append("git no disponible: la corrida no tiene SHA de origen")
    if lineage["lock_hash"] is None:
        advertencias_determinismo.append("lineage parcial: sin hash de archivo de bloqueo de dependencias")
    return advertencias_determinismo, corrida_2, determinista, huella_contenido, lineage


@app.cell
def _(advertencias_determinismo, corrida, corrida_2, determinista, lineage, mo):
    mo.vstack([
        mo.md(f"""
    **Dos corridas, mismo config:** `run_id` {corrida['trail'].run_id[:8]}… vs
    {corrida_2['trail'].run_id[:8]}… · huella de contenido idéntica: `{determinista}` ·
    artefacto idéntico: `{corrida['hash_artefacto'] == corrida_2['hash_artefacto']}` ·
    δ idéntico bit a bit: `{corrida['delta'] == corrida_2['delta']}`.

    **Advertencias automáticas** (las escribe el código, no el modelador):
    """ + "\n".join(f"- {a}" for a in advertencias_determinismo)),
        mo.md("```\n" + "\n".join(f"{k:18s} {v}" for k, v in lineage.items()) + "\n```"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.1 Dos maneras de «hashear un DataFrame»

    (1) **La del curso** (`hash_df` de la demo de clase 6): esquema (posición, nombre, dtype) +
    `pandas.util.hash_pandas_object(df, index=True)` → SHA-256. Rápida y vectorizada; identifica la
    **representación física** (dtypes, orden de filas y columnas, índice).
    (2) **Canónica propia**: filas ordenadas por `id`, columnas por nombre, números como double
    (`repr`), `-0.0 → 0.0`, NaN/None → `null`, texto en NFC; ignora índice y dtype físico. Identifica
    el **contenido** bajo una equivalencia declarada.

    No deben coincidir entre sí (son funciones distintas): lo que se compara es **qué cambios detecta
    cada una**. Elige una perturbación; la tabla evalúa las nueve.
    """)
    return


@app.cell
def _(mo):
    selector_perturbacion = mo.ui.dropdown(
        options=["ninguna (regenerar con la misma semilla)", "un valor + 1e-9",
                 "permutar filas", "reordenar columnas", "edad float64 → int64 (mismos valores)",
                 "texto: str → object", "0.0 → -0.0 en deuda_otras", "índice distinto (0..n-1 → 1000..)",
                 "canal: NFC → NFD (texto con tilde)"],
        value="permutar filas", label="Perturbación")
    selector_perturbacion
    return (selector_perturbacion,)


@app.cell
def _(np, pd, unicodedata):
    def perturbar(df, tipo):
        d = df.copy()
        if tipo.startswith("ninguna"):
            return d
        if tipo.startswith("un valor"):
            d.loc[d.index[5], "uso_linea_prom_12m"] = d.loc[d.index[5], "uso_linea_prom_12m"] + 1e-9
        elif tipo.startswith("permutar"):
            d = d.sample(frac=1.0, random_state=7)
        elif tipo.startswith("reordenar"):
            d = d[list(reversed(d.columns))]
        elif tipo.startswith("edad"):
            d["edad"] = d["edad"].astype("int64")
        elif tipo.startswith("texto"):
            d["canal"] = d["canal"].astype(object)
        elif tipo.startswith("0.0"):
            d["deuda_otras_prom_12m"] = np.where(d["deuda_otras_prom_12m"] == 0.0, -0.0,
                                                 d["deuda_otras_prom_12m"])
        elif tipo.startswith("índice"):
            d.index = pd.RangeIndex(1000, 1000 + len(d))
        elif tipo.startswith("canal"):
            d["canal"] = d["canal"].str.replace("sucursal", "sucursal Concepción")
            d["canal"] = d["canal"].map(lambda s: unicodedata.normalize("NFD", s))
        return d
    return (perturbar,)


@app.cell
def _(corrida, corrida_2, hash_df_canonico, hash_df_pandas, pd, perturbar, unicodedata):
    # Base de comparación para la perturbación NFD: el mismo texto, pero en NFC
    _base = corrida["df"]
    _base_nfc = _base.copy()
    _base_nfc["canal"] = _base_nfc["canal"].str.replace("sucursal", "sucursal Concepción").map(
        lambda s: unicodedata.normalize("NFC", s))
    _POL = {  # ¿es «el mismo dato» bajo la política de identidad de contenido?
        "ninguna (regenerar con la misma semilla)": True, "un valor + 1e-9": False,
        "permutar filas": True, "reordenar columnas": True,
        "edad float64 → int64 (mismos valores)": True, "texto: str → object": True,
        "0.0 → -0.0 en deuda_otras": True, "índice distinto (0..n-1 → 1000..)": True,
        "canal: NFC → NFD (texto con tilde)": True}
    _hp0, _hc0 = hash_df_pandas(_base), hash_df_canonico(_base)
    _hp0n, _hc0n = hash_df_pandas(_base_nfc), hash_df_canonico(_base_nfc)
    _filas = []
    for _t, _mismo in _POL.items():
        _d = corrida_2["df"] if _t.startswith("ninguna") else perturbar(_base, _t)
        _ref_p, _ref_c = (_hp0n, _hc0n) if _t.startswith("canal") else (_hp0, _hc0)
        _cp = hash_df_pandas(_d) != _ref_p
        _cc = hash_df_canonico(_d) != _ref_c
        _filas.append({"perturbación": _t, "¿mismo contenido?": _mismo,
                       "cambia hash pandas": _cp, "cambia hash canónico": _cc,
                       "pandas acierta": _cp != _mismo, "canónico acierta": _cc != _mismo})
    tabla_perturbaciones = pd.DataFrame(_filas).set_index("perturbación")
    return (tabla_perturbaciones,)


@app.cell
def _(corrida, hash_df_canonico, hash_df_pandas, mo, pd, perturbar, selector_perturbacion, tabla_perturbaciones):
    _d = perturbar(corrida["df"], selector_perturbacion.value)
    mo.vstack([
        mo.md(f"**{selector_perturbacion.value}** → pandas `{hash_df_pandas(_d)[:16]}…` · "
              f"canónico `{hash_df_canonico(_d)[:16]}…` (base: pandas "
              f"`{hash_df_pandas(corrida['df'])[:16]}…`, canónico `{corrida['data_hash'][:16]}…`)"),
        tabla_perturbaciones,
        mo.md(f"""
    **Lectura.** Ambos detectan el cambio de contenido real (1e-9 en un valor). El hash de pandas
    **también** cambia con {int((tabla_perturbaciones['cambia hash pandas'] & tabla_perturbaciones['¿mismo contenido?']).sum())}
    perturbaciones que no cambian el contenido (orden de filas o columnas, int vs float, −0,0, índice,
    NFD, incluso `str` → `object`, porque el esquema del `hash_df` del curso registra el dtype): útil
    como huella **física** («este es exactamente el objeto que se usó»), ruidoso como identidad de
    **contenido** («estos son los mismos datos»). Además, `hash_pandas_object` no está documentada
    como estable entre versiones de pandas ({pd.__version__} aquí; usa SipHash con clave fija y una
    combinación propia de columnas; cambios de dtype por defecto —como el `str` de pandas 3— también
    la afectan), así que un `data_hash` archivado con pandas podría no reproducirse tras
    actualizar la librería **aunque los datos sean idénticos**. El canónico depende solo de `repr` de
    Python para doubles, de JSON y de NFC: más lento, pero su definición cabe en un párrafo del
    expediente. En producción: canónico para el `data_hash` del lineage; pandas como verificación
    rápida intra-entorno. Precaución del canónico: convierte todo número a double, así que enteros
    mayores que 2⁵³ (IDs largos) deben tratarse como texto.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. El model card, generado desde la corrida

    Mitchell et al. (2019) propusieron la ficha de modelo para que cada modelo publicado declare uso
    previsto, desempeño desagregado y limitaciones. Adaptada a crédito (clase 6 y plantilla del Lab 3),
    tiene siete secciones y dos reglas: **se genera desde la corrida** (cada número sale de `corrida`,
    nada se tipea) y separa lo **declarado** por personas de lo **automático** (lo que el entorno y los
    datos detectan solos, como hacía `nikodym` con «git no disponible»).

    Las limitaciones automáticas se disparan con reglas escritas antes de mirar el resultado:
    backtesting OOT fallido, CSI de una variable elegida > 0,10 contra TTD, menos de 10 malos por
    parámetro, advertencias de determinismo. Un *lint* mínimo separa limitaciones específicas de
    genéricas (no reemplaza el juicio: solo atrapa las vacías).
    """)
    return


@app.cell
def _(CONFIG, SELLO, advertencias_determinismo, corrida, lineage, np, pd, psi_numpy):
    def csi_variable(v):
        dev_v = corrida["muestras"]["DEV"][v]
        ttd_v = corrida["muestras"]["TTD"][v]
        if not pd.api.types.is_numeric_dtype(dev_v):
            cats = sorted(set(dev_v.dropna()) | set(ttd_v.dropna()))
            e = np.array([(dev_v == c).mean() for c in cats]) + 1e-6
            a = np.array([(ttd_v == c).mean() for c in cats]) + 1e-6
            return float(np.sum((a - e) * np.log(a / e)))
        return psi_numpy(dev_v.fillna(-1).values, ttd_v.fillna(-1).values)

    _m = corrida["metricas"]
    csi_ttd = {v: csi_variable(v) for v in corrida["elegidas"] + ["canal"]}
    _mpp = _m["DEV"]["malos"] / (len(corrida["elegidas"]) + 1)

    limitaciones_declaradas = [
        "Modelado sobre solicitudes aprobadas del generador: no observa rechazados; la PD de "
        "quien la política vigente rechazaba no está validada (0 rechazados observados; ver Serie 1 · E1).",
        f"La tendencia central ({corrida['tc']:.2%}) se ancló en DEV+HO (cohortes ≤ 2024-12): si el "
        f"deterioro de 2025 persiste, el nivel queda corto en {(_m['OOT']['tasa_obs'] - _m['OOT']['pd_media']) * 100:.1f} pp "
        f"(OOT: PD {_m['OOT']['pd_media']:.2%} vs mora {_m['OOT']['tasa_obs']:.2%}).",
        f"El desempeño del lote TTD ({len(corrida['salida']):,} solicitudes) no está verificado: sus ".replace(",", ".")
        + "cohortes 2025-07…12 maduran 12 meses después; la aprobación proyectada no es mora observada.",
        f"El binning se aplica con los cortes de DEV recalculados por binear(ref=DEV); el congelado "
        f"explícito de cortes (M21) no forma parte de este artefacto: no desplegar este JSON tal cual.",
    ]
    limitaciones_automaticas = list(advertencias_determinismo)
    if corrida["backtesting"] == "fail":
        limitaciones_automaticas.append(
            f"Backtesting OOT fallido: binomial global p = {corrida['backtesting_p']:.2g} "
            f"(PD {_m['OOT']['pd_media']:.2%} vs mora {_m['OOT']['tasa_obs']:.2%}); no autoriza producción "
            f"sin diagnóstico por tramos.")
    for _v, _c in csi_ttd.items():
        if _c > 0.10:
            limitaciones_automaticas.append(
                f"CSI DEV→TTD de «{_v}» = {_c:.3f} > 0,10: la población cambió en esa variable"
                + (" (no está en el modelo, pero su mezcla altera la población)" if _v not in corrida["elegidas"] else "") + ".")
    if _mpp < 10:
        limitaciones_automaticas.append(f"Solo {_mpp:.1f} malos por parámetro en DEV (< 10).")
    limitaciones_automaticas.append(
        "El sello del trail de este notebook está junto al log (demostración): sin custodia "
        "separada o sello RFC 3161 no prueba ausencia de reescritura.")

    model_card = {
        "identidad": {"modelo": CONFIG["modelo_id"], "version": CONFIG["version"],
                      "tipo": "Scorecard logístico de admisión (PD a 12 meses)",
                      "run_id": lineage["run_id"], "config_hash": lineage["config_hash"],
                      "data_hash": lineage["data_hash"], "hash_artefacto": lineage["hash_artefacto"],
                      "sello_trail": f"{SELLO['hash_terminal']} ({SELLO['n_eventos']} eventos)"},
        "proposito": {"uso_previsto": CONFIG["gobierno"]["proposito"]
                      + f" Cutoff {CONFIG['politica']['cutoff']}.",
                      "usuarios": CONFIG["gobierno"]["usuarios"],
                      "usos_no_previstos": CONFIG["gobierno"]["usos_no_previstos"]},
        "datos": {"poblacion": "Cartera sintética «Banco Sintético», cohortes 2023-07 … 2025-12.",
                  "target": "1 = malo (90+ DPD a 12 meses)",
                  "muestras": {m: _m[m]["n"] for m in _m} | {"TTD": int(len(corrida["salida"]))},
                  "tendencia_central": round(corrida["tc"], 6),
                  "variables_finales": corrida["elegidas"]},
        "desempeno": {"gini": {m: round(_m[m]["gini"], 4) for m in _m},
                      "ks": {m: round(_m[m]["ks"], 4) for m in _m},
                      "psi_score": {k: round(v, 4) for k, v in corrida["psi"].items()},
                      "backtesting_oot": f"binomial p = {corrida['backtesting_p']:.2g} → {corrida['backtesting']}"},
        "limitaciones": {"declaradas": limitaciones_declaradas, "automaticas": limitaciones_automaticas},
        "gobierno": {"estado": "en validación independiente", "responsable": CONFIG["gobierno"]["responsable"],
                     "revision": f"cada {CONFIG['gobierno']['revision_meses']} meses o ante gatillo",
                     "monitoreo": "mensual (ranking y estabilidad) · trimestral (calibración)"},
        "entorno": lineage["library_versions"],
    }
    return csi_ttd, limitaciones_automaticas, limitaciones_declaradas, model_card


@app.cell
def _():
    def card_markdown(card):
        titulos = {"identidad": "1. Identidad", "proposito": "2. Propósito y uso previsto",
                   "datos": "3. Datos", "desempeno": "4. Desempeño", "limitaciones": "5. Limitaciones",
                   "gobierno": "6. Gobierno", "entorno": "7. Entorno de construcción"}
        L = [f"### Model card — {card['identidad']['modelo']} v{card['identidad']['version']}", ""]
        for k, t in titulos.items():
            L += [f"**{t}**", ""]
            for k2, v2 in card[k].items():
                et = k2.replace("_", " ").capitalize()
                if isinstance(v2, dict):
                    L.append(f"- **{et}**: " + " · ".join(f"{a} `{b}`" for a, b in v2.items()))
                elif isinstance(v2, list):
                    L.append(f"- **{et}**:")
                    L += [f"    - {x}" for x in v2]
                else:
                    L.append(f"- **{et}**: {v2}")
            L.append("")
        return "\n".join(L)

    def lint_limitacion(texto):
        """Heurística: específica = trae un número, nombra una consecuencia y tiene > 60 caracteres."""
        import re
        tiene_numero = bool(re.search(r"\d", texto))
        consecuencia = any(p in texto.lower() for p in
                           ("no ", "sin ", "subestima", "sobreestima", "queda", "falla", "cambió",
                            "exige", "no autoriza", "no desplegar"))
        return tiene_numero and consecuencia and len(texto) > 60
    return card_markdown, lint_limitacion


@app.cell
def _(card_markdown, limitaciones_declaradas, lint_limitacion, mo, model_card, pd):
    _genericas = ["El modelo tiene supuestos.",
                  "Los datos podrían no ser representativos en el futuro.",
                  "Se recomienda monitorear el modelo periódicamente."]
    _lint = pd.DataFrame(
        [{"limitación": t[:90] + ("…" if len(t) > 90 else ""), "origen": "genérica (ejemplo)",
          "pasa lint": lint_limitacion(t)} for t in _genericas]
        + [{"limitación": t[:90] + "…", "origen": "declarada (esta corrida)", "pasa lint": lint_limitacion(t)}
           for t in limitaciones_declaradas])
    mo.vstack([mo.md(card_markdown(model_card)),
               mo.md("**Lint de limitaciones** (heurístico: número + consecuencia + extensión):"),
               _lint])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. RACI y gatillos: el gobierno como restricciones verificables

    **RACI** (clase 6, lámina 8): R ejecuta, A responde (**una sola por fila**), C consultado,
    I informado; *quien construye no valida ni aprueba*. Escrito como matriz, eso son restricciones que
    se verifican con código igual que un test: (i) exactamente una A (contando `R/A`) por actividad;
    (ii) al menos una R; (iii) en actividades que exigen independencia, el Modelador no tiene R ni A.
    Dos implementaciones del chequeo: numpy (matriz de strings) y pandas; deben coincidir.

    **Gatillos** (clase 6, lámina 9): cinco partes — condición medible, valor de hoy, quién decide,
    acción, plazo. `disparado` **se calcula** desde la condición; si falta el insumo, es
    «no evaluable» (no es lo mismo saber que no está disparado que no saberlo).
    """)
    return


@app.cell
def _(np, pd):
    ROLES = ["Modelador", "Jefe Modelos", "Validación", "TI", "Comité"]
    _filas = [  # actividad, requiere independencia, R/A por rol (clase 6, lámina 8)
        ("Construir el modelo", False, ["R", "A", "C", "I", "I"]),
        ("Validar independientemente", True, ["I", "I", "R/A", "C", "I"]),
        ("Aprobar el uso en producción", True, ["I", "C", "C", "I", "R/A"]),
        ("Desplegar y operar el scoring", False, ["C", "I", "I", "R/A", "I"]),
        ("Monitorear el tablero mensual", False, ["R", "A", "C", "C", "I"]),
        ("Decidir recalibrar o re-desarrollar", True, ["C", "R", "C", "I", "A"]),
        ("Autorizar excepciones al cutoff", True, ["I", "C", "I", "I", "R/A"]),
        ("Archivar el expediente del modelo", False, ["R", "A", "C", "I", "I"]),
    ]
    raci = pd.DataFrame([r[2] for r in _filas], columns=ROLES, index=[r[0] for r in _filas])
    raci_indep = pd.Series([r[1] for r in _filas], index=raci.index)

    def chequear_raci_numpy(M, indep, col_modelador=0):
        M = np.asarray(M, dtype=str)
        es_a = (M == "A") | (M == "R/A")
        es_r = (M == "R") | (M == "R/A")
        n_a = es_a.sum(axis=1)
        n_r = es_r.sum(axis=1)
        viola = np.asarray(indep) & (es_a[:, col_modelador] | es_r[:, col_modelador])
        return (n_a == 1) & (n_r >= 1) & ~viola

    def chequear_raci_pandas(df, indep):
        n_a = df.isin(["A", "R/A"]).sum(axis=1)
        n_r = df.isin(["R", "R/A"]).sum(axis=1)
        viola = indep & df["Modelador"].isin(["R", "A", "R/A"])
        return ((n_a == 1) & (n_r >= 1) & ~viola).to_numpy()

    raci_roto = raci.copy()
    raci_roto.loc["Validar independientemente", "Modelador"] = "R"      # el modelador «se valida»
    raci_roto.loc["Monitorear el tablero mensual", "Comité"] = "A"      # dos A en una fila
    return chequear_raci_numpy, chequear_raci_pandas, raci, raci_indep, raci_roto


@app.cell
def _(chequear_raci_numpy, chequear_raci_pandas, mo, raci, raci_indep, raci_roto):
    raci_ok_np = chequear_raci_numpy(raci.values, raci_indep.values)
    raci_roto_np = chequear_raci_numpy(raci_roto.values, raci_indep.values)
    raci_coinciden = bool((raci_ok_np == chequear_raci_pandas(raci, raci_indep)).all()
                          and (raci_roto_np == chequear_raci_pandas(raci_roto, raci_indep)).all())
    _t = raci.copy()
    _t["¿independencia?"] = raci_indep.map({True: "Sí", False: "No"})
    _t["chequeo"] = ["✓" if x else "✗" for x in raci_ok_np]
    _t["chequeo (RACI roto)"] = ["✓" if x else "✗" for x in raci_roto_np]
    mo.vstack([_t, mo.md(f"numpy = pandas en ambas matrices: `{raci_coinciden}` · la matriz rota "
                         f"falla en: {', '.join(raci.index[~raci_roto_np])}.")])
    return raci_coinciden, raci_ok_np, raci_roto_np


@app.cell
def _(corrida, pd):
    def evaluar(valor, operador, umbral):
        if valor is None:
            return "no evaluable"
        ok = {">": valor > umbral, ">=": valor >= umbral, "<": valor < umbral,
              "<=": valor <= umbral}[operador]
        return "SÍ" if ok else "no"

    _m = corrida["metricas"]
    _caida = 1 - _m["OOT"]["gini"] / _m["DEV"]["gini"]
    _gat = [
        ("1 · Vigilancia reforzada", "indicadores en 🟡/🔴 (backtesting OOT)", ">", 0,
         1 if corrida["backtesting"] == "fail" else 0, "Jefe Modelos",
         "duplicar frecuencia de medición por 2 meses", 30),
        ("2 · Recalibración del δ", "trimestres seguidos con binomial global 🟡/🔴", ">=", 2,
         None, "Comité (propone Jefe Modelos)", "re-anclar δ a la TC actualizada, con acta", 60),
        ("3a · Re-desarrollo", "caída relativa de Gini DEV→OOT", ">", 0.30, _caida,
         "Comité", "abrir proyecto de re-desarrollo", 90),
        ("3b · Re-desarrollo", "KS OOT", "<", 0.20, _m["OOT"]["ks"], "Comité",
         "abrir proyecto de re-desarrollo", 90),
        ("3c · Re-desarrollo", "PSI del score DEV→TTD", ">", 0.25, corrida["psi"]["DEV_TTD"],
         "Comité", "abrir proyecto de re-desarrollo", 90),
        ("4 · Contingencia", "% del lote con hallazgo 🔴 del contrato", ">", 0.0, 0.0,
         "TI / Producción", "política de knock-outs de respaldo + aviso al comité", 1),
        ("5 · Overrides", "tasa de overrides sobre aprobados", ">", 0.05, None,
         "Comité", "revisar política de excepciones y a quién se le otorgan", 30),
    ]
    gatillos = pd.DataFrame([{"gatillo": g[0], "condición": f"{g[1]} {g[2]} {g[3]}",
                              "valor_hoy": ("sin insumo" if g[4] is None else round(float(g[4]), 4)),
                              "disparado": evaluar(g[4], g[2], g[3]), "decide": g[5],
                              "acción": g[6], "plazo_días": g[7]} for g in _gat]).set_index("gatillo")
    return evaluar, gatillos


@app.cell
def _(gatillos, mo):
    mo.vstack([gatillos, mo.md(f"""
    **Lectura.** {int((gatillos['disparado'] == 'SÍ').sum())} disparado(s),
    {int((gatillos['disparado'] == 'no evaluable').sum())} sin insumo. El patrón es el de Austral: el
    ranking y la población no gatillan re-desarrollo; la calibración gatilla **vigilancia y
    diagnóstico**, y la recalibración queda «no evaluable» porque exige dos trimestres y hay uno.
    Saltar directo a re-desarrollo sería desproporcionado; recalibrar δ hoy, sin diagnóstico por
    tramos, sería decidir con un solo dato. El gatillo 5 no se puede evaluar sin **registro de
    overrides**: la sección siguiente muestra por qué ese registro no es burocracia.
    """)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Overrides: la excepción también es un modelo

    Un *override* bajo (*low-side*) aprueba a quien el score rechaza; uno alto (*high-side*) rechaza a
    quien el score aprueba. Del alto nunca se observa el desempeño (no hay crédito); del bajo sí, y
    eso permite **validar la excepción**. Aquí, sobre OOT (con desempeño y con la verdad del
    generador), la mesa de excepciones aprueba una fracción de los rechazados con tres políticas:

    - **información blanda**: la mesa ve una señal ruidosa del riesgo verdadero (`pd_verdadera` ×
      ruido log-normal) que el modelo no tiene;
    - **comercial**: prioriza el canal `fuerza_venta` (incentivo de colocación), sin información de riesgo;
    - **al azar**.

    Se contrasta la mora observada de los overrides con la PD que el **modelo** les asignaba (binomial
    exacta, dos implementaciones: `math.lgamma` desde cero vs `scipy.stats.binomtest`).
    """)
    return


@app.cell
def _(mo):
    slider_override = mo.ui.slider(0.01, 0.30, step=0.01, value=0.10,
                                   label="Fracción de rechazados que la mesa aprueba")
    selector_politica = mo.ui.dropdown(options=["información blanda", "comercial", "al azar"],
                                       value="información blanda", label="Política de override")
    mo.hstack([slider_override, selector_politica])
    return selector_politica, slider_override


@app.cell
def _(math, np):
    def binom_dos_colas_numpy(x, n, p):
        """p-valor exacto de dos colas (mismo criterio que scipy: suma de P(k) ≤ P(x))."""
        logf = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                + (k * math.log(p) if k else 0.0) + ((n - k) * math.log1p(-p) if n - k else 0.0)
                for k in range(n + 1)]
        pmf = np.exp(np.array(logf))
        return float(min(1.0, pmf[pmf <= pmf[x] * (1 + 1e-7)].sum()))
    return (binom_dos_colas_numpy,)


@app.cell
def _(CONFIG, binom_dos_colas_numpy, binomtest, corrida, np, pd):
    def simular_overrides(fraccion, politica, semilla=11):
        rng = np.random.default_rng(semilla)
        oot = corrida["muestras"]["OOT"]
        sc = corrida["score"]["OOT"]
        pdm = corrida["pdc"]["OOT"]
        rech = np.flatnonzero(sc < CONFIG["politica"]["cutoff"])
        k = max(1, int(round(fraccion * len(rech))))
        if politica == "información blanda":
            senal = oot["pd_verdadera"].to_numpy()[rech] * np.exp(rng.normal(0, 0.5, len(rech)))
            elegidos = rech[np.argsort(senal)[:k]]
        elif politica == "comercial":
            prioridad = (oot["canal"].to_numpy(dtype=object)[rech] == "fuerza_venta").astype(float) + rng.random(len(rech)) * 0.5
            elegidos = rech[np.argsort(-prioridad)[:k]]
        else:
            elegidos = rng.choice(rech, k, replace=False)
        y = oot["malo"].values
        aprob = np.flatnonzero(sc >= CONFIG["politica"]["cutoff"])
        x_ov, n_ov, p_mod = int(y[elegidos].sum()), len(elegidos), float(pdm[elegidos].mean())
        return {
            "n_rechazados": len(rech), "n_overrides": n_ov,
            "tasa_override_sobre_aprobados": n_ov / (len(aprob) + n_ov),
            "mora_aprobados_score": float(y[aprob].mean()),
            "mora_overrides": x_ov / n_ov, "pd_modelo_overrides": p_mod,
            "pd_verdadera_overrides": float(oot["pd_verdadera"].values[elegidos].mean()),
            "mora_rechazados_restantes": float(np.delete(y, np.concatenate([aprob, elegidos])).mean()),
            "p_numpy": binom_dos_colas_numpy(x_ov, n_ov, p_mod),
            "p_scipy": float(binomtest(x_ov, n_ov, p_mod).pvalue),
            "mora_cartera_sin": float(y[aprob].mean()),
            "mora_cartera_con": float(y[np.concatenate([aprob, elegidos])].mean()),
        }

    tabla_overrides = pd.DataFrame({pol: simular_overrides(0.10, pol)
                                    for pol in ("información blanda", "comercial", "al azar")}).T
    return simular_overrides, tabla_overrides


@app.cell
def _(mo, plt, selector_politica, simular_overrides, slider_override, tabla_overrides):
    ov = simular_overrides(slider_override.value, selector_politica.value)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _et = ["aprobados\npor score", "overrides:\nmora observada", "overrides:\nPD del modelo",
           "rechazados\nrestantes"]
    _vals = [ov["mora_aprobados_score"], ov["mora_overrides"], ov["pd_modelo_overrides"],
             ov["mora_rechazados_restantes"]]
    _ax.bar(_et, [v * 100 for v in _vals], color=["#4c72b0", "#dd8452", "#aaaaaa", "#c44e52"])
    _ax.set_ylabel("tasa de malos (%)")
    _ax.set_title(f"OOT · override «{selector_politica.value}» sobre {slider_override.value:.0%} de los rechazados")
    for _i, _v in enumerate(_vals):
        _ax.text(_i, _v * 100 + 0.3, f"{_v:.1%}", ha="center", fontsize=9)
    _ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"""
    {ov['n_overrides']} overrides ({ov['tasa_override_sobre_aprobados']:.1%} de los aprobados finales).
    Mora de los overrides **{ov['mora_overrides']:.1%}** vs PD que el modelo les asignaba
    {ov['pd_modelo_overrides']:.1%} (verdad del generador: {ov['pd_verdadera_overrides']:.1%}) ·
    binomial p = {ov['p_numpy']:.3g} (numpy) / {ov['p_scipy']:.3g} (scipy) · mora de la cartera
    {ov['mora_cartera_sin']:.2%} → {ov['mora_cartera_con']:.2%}.
    """),
        mo.md("**Las tres políticas al 10%:**"),
        tabla_overrides[["n_overrides", "mora_overrides", "pd_modelo_overrides",
                         "pd_verdadera_overrides", "p_scipy", "mora_cartera_con"]].astype(float).round(4),
        mo.md(f"""
    **Lectura.** Con información blanda (al 10%) los overrides tienen mora
    {tabla_overrides.loc['información blanda', 'mora_overrides']:.1%} contra una PD de modelo de
    {tabla_overrides.loc['información blanda', 'pd_modelo_overrides']:.1%} (verdad:
    {tabla_overrides.loc['información blanda', 'pd_verdadera_overrides']:.1%}): la mesa sabe algo que el
    modelo no. Pero con {int(tabla_overrides.loc['información blanda', 'n_overrides'])} casos la binomial
    da p = {tabla_overrides.loc['información blanda', 'p_scipy']:.2f}: **no alcanza para probarlo** (sube
    el slider y mira cómo cae el p-valor). Con la política comercial la mora de los overrides
    ({tabla_overrides.loc['comercial', 'mora_overrides']:.1%}) **supera** la PD del modelo
    ({tabla_overrides.loc['comercial', 'pd_modelo_overrides']:.1%}; p =
    {tabla_overrides.loc['comercial', 'p_scipy']:.2f}): el canal `fuerza_venta` tiene riesgo que el
    modelo no ve porque el embudo descartó `canal` por IV bajo en DEV. En los tres casos la mora de
    la cartera **sube** (se aprueba desde la zona rechazada). Lecturas: (i) un override que funciona
    es **insumo de re-desarrollo** (¿qué ve la mesa?), no un éxito de la mesa; (ii) uno que falla por
    segmento revela una variable omitida; (iii) sin **registro** (quién, motivo codificado, dato
    usado) estas tres políticas son indistinguibles en el tablero agregado. SR 11-7 ya pedía analizar
    overrides dentro del monitoreo. Mínimo operativo: motivo codificado, límite de tasa (gatillo 5),
    desempeño de la cohorte de overrides contra su PD, y alerta si un emisor concentra excepciones.
    """),
    ])
    return (ov,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Validación independiente mínima: replicar, retar, estresar

    El validador no «revisa el notebook»: **reproduce** desde el expediente, **reta** con un modelo
    alternativo y **estresa** supuestos. Tres pruebas:

    1. **Replicación**: con el lineage (config + semilla) vuelve a generar datos y artefacto; exige
       igualdad de `data_hash`, `config_hash` y hash del artefacto.
    2. **Benchmark con *challenger***: un `HistGradientBoostingClassifier` sobre las variables crudas
       (sin WoE). Diferencia de Gini OOT con bootstrap pareado (B = 200). Si el challenger gana por
       mucho, la pregunta es qué estructura se está perdiendo; si empata, la complejidad no se justifica.
    3. **Sensibilidad**: cuánto cambia la aprobación del lote TTD si δ se mueve ±0,10 (el orden de
       magnitud del error de nivel que vimos en OOT).
    """)
    return


@app.cell
def _(
    CONFIG,
    HistGradientBoostingClassifier,
    corrida,
    correr_pipeline,
    gini_numpy,
    hash_df_canonico,
    np,
    roc_auc_score,
):
    # 1. Replicación desde el expediente
    _rep = correr_pipeline(CONFIG, "validador")
    replicacion = {
        "data_hash": _rep["data_hash"] == corrida["data_hash"],
        "config_hash": _rep["config_hash"] == corrida["config_hash"],
        "hash_artefacto": _rep["hash_artefacto"] == corrida["hash_artefacto"],
        "hash_salida_ttd": hash_df_canonico(_rep["salida"]) == hash_df_canonico(corrida["salida"]),
    }

    # 2. Challenger
    _cols = [c for c in CONFIG["candidatas"] if c != "canal"]

    def _X(d):
        X = d[_cols].to_numpy(dtype=float)
        return np.column_stack([X, (d["canal"].to_numpy(dtype=object)[:, None] == np.array(
            ["sucursal", "web", "app", "fuerza_venta"])).astype(float)])

    _dev, _oot = corrida["muestras"]["DEV"], corrida["muestras"]["OOT"]
    _gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=0)
    _gb.fit(_X(_dev), _dev["malo"].values)
    _s_ch = _gb.predict_proba(_X(_oot))[:, 1]
    _s_cp = corrida["lp"]["OOT"]
    _y = _oot["malo"].values
    gini_campeon = 2 * roc_auc_score(_y, _s_cp) - 1
    gini_challenger = 2 * roc_auc_score(_y, _s_ch) - 1
    gini_numpy_igual_sklearn = bool(np.isclose(gini_numpy(_y, _s_cp), gini_campeon, atol=1e-12))
    _rng = np.random.default_rng(3)
    _dif = []
    for _ in range(200):
        _i = _rng.integers(0, len(_y), len(_y))
        if _y[_i].min() == _y[_i].max():
            continue
        _dif.append((2 * roc_auc_score(_y[_i], _s_ch[_i]) - 1) - (2 * roc_auc_score(_y[_i], _s_cp[_i]) - 1))
    ic_dif_gini = np.percentile(_dif, [2.5, 97.5])

    # 3. Sensibilidad a δ (el score incluye δ: moverlo desplaza todos los scores factor·Δδ puntos)
    _lp_ttd = corrida["lp"]["TTD"]
    _cut = CONFIG["politica"]["cutoff"]

    def aprobacion(dd):
        sc = corrida["offset"] - corrida["factor"] * (_lp_ttd + corrida["delta"] + dd)
        return float((sc >= _cut).mean())

    sensibilidad_delta = {dd: aprobacion(dd) for dd in (-0.10, 0.0, 0.10)}
    return (
        gini_campeon,
        gini_challenger,
        gini_numpy_igual_sklearn,
        ic_dif_gini,
        replicacion,
        sensibilidad_delta,
    )


@app.cell
def _(
    corrida,
    gini_campeon,
    gini_challenger,
    ic_dif_gini,
    mo,
    replicacion,
    sensibilidad_delta,
):
    mo.md(f"""
    **Replicación:** {', '.join(f'{k} `{v}`' for k, v in replicacion.items())}.

    **Challenger:** Gini OOT campeón {gini_campeon:.4f} · challenger (GBM sobre crudas)
    {gini_challenger:.4f} · IC 95% bootstrap de la diferencia [{ic_dif_gini[0]:+.4f}; {ic_dif_gini[1]:+.4f}].
    {'El IC contiene el cero: la complejidad adicional no se justifica con esta evidencia.' if ic_dif_gini[0] <= 0 <= ic_dif_gini[1] else ('El challenger gana de forma concluyente: hay estructura que el scorecard no captura (el generador tiene efectos no lineales y el canal, que el embudo descartó por IV).' if ic_dif_gini[0] > 0 else 'El campeón gana de forma concluyente.')}

    **Sensibilidad:** aprobación TTD con δ − 0,10 / δ / δ + 0,10 =
    {sensibilidad_delta[-0.10]:.1%} / {sensibilidad_delta[0.0]:.1%} / {sensibilidad_delta[0.10]:.1%}
    (±0,10 en δ = ∓{corrida['factor'] * 0.10:.2f} puntos en todo el score). Un error de nivel del
    tamaño observado en OOT mueve la aprobación en esa magnitud **sin cambiar el ranking**: por eso la
    calibración es un tema de comité aunque el Gini esté sano.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Inventario y *tiering* por materialidad

    El inventario es la lista de **todo** lo que la institución trata como modelo (y de lo que decidió
    que no lo es, con su razón). El *tiering* asigna intensidad de gobierno por materialidad y riesgo
    inherente. La regla de abajo es una **convención** de este módulo (ninguna norma la prescribe con
    estos números) y es la misma que implementa la hoja `Tiering` de `M22_raci_gatillos.xlsx` y la
    columna `tier` de `M22_inventario_modelos.csv`.

    $$\text{puntaje}=0{,}4\cdot\text{materialidad}+0{,}3\cdot\text{uso}+0{,}3\cdot\text{complejidad},
    \quad \text{tier}=\begin{cases}1 & \text{puntaje}\ge 2{,}4\\ 2 & 1{,}7\le\text{puntaje}<2{,}4\\
    3 & \text{si no}\end{cases}$$

    con materialidad = 1/2/3 según exposición < 1.000 / 1.000–10.000 / > 10.000 MM CLP.
    """)
    return


@app.cell
def _(pd):
    def tier(exposicion_mm, uso, complejidad, u1=1_000, u2=10_000):
        mat = 1 if exposicion_mm < u1 else (2 if exposicion_mm <= u2 else 3)
        puntaje = 0.4 * mat + 0.3 * uso + 0.3 * complejidad
        return mat, round(puntaje, 4), (1 if puntaje >= 2.4 else (2 if puntaje >= 1.7 else 3))

    _inv = [  # modelo_id, exposición MM CLP, uso (1-3), complejidad (1-3) — mismos valores que el CSV
        ("austral-scorecard-consumo", 180_000, 3, 2),
        ("sintetico-scorecard-consumo", 25_000, 3, 2),
        ("motos-admision-scorecard", 4_500, 3, 2),
        ("provisiones-consumo-metodo-estandar", 180_000, 3, 1),
        ("bureau-score-proveedor", 180_000, 2, 3),
        ("pricing-motos-planilla", 4_500, 2, 1),
    ]
    inventario_tiers = pd.DataFrame(
        [{"modelo_id": m, "exposicion_mm": e, "uso": u, "complejidad": c,
          **dict(zip(["materialidad", "puntaje", "tier"], tier(e, u, c)))} for m, e, u, c in _inv]
    ).set_index("modelo_id")
    inventario_tiers
    return inventario_tiers, tier


@app.cell
def _(mo):
    mo.md(r"""
    **Lectura.** El scorecard de motos de una fintech pequeña (4.500 MM CLP) cae en tier 2 aunque decide
    en automático, y el score de bureau del proveedor queda en tier 1 aunque nadie en la casa lo
    desarrolló: *comprar* un modelo no transfiere su riesgo. El método estándar de provisiones es una
    fórmula normativa (tier 1 por materialidad): bajo la definición de SR 26-2 —«método cuantitativo
    **complejo**»— podría no ser «modelo», pero se inventaría igual con esa justificación escrita.
    La planilla de pricing es una EUC (*end-user computing*): el inventario existe precisamente para
    que estas no queden fuera.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Checks del módulo
    """)
    return


@app.cell
def _(
    CLAVE_HMAC,
    EVENTOS,
    HASHES,
    HASHES_HMAC,
    SELLO,
    capas_sello,
    coincide_canon,
    corrida,
    determinista,
    encadenar,
    gini_numpy_igual_sklearn,
    inventario_tiers,
    iterativa_igual_recursiva,
    limitaciones_automaticas,
    limitaciones_declaradas,
    lint_limitacion,
    matriz_ataques,
    mo,
    model_card,
    mutacion_duplicando,
    mutacion_rfc6962,
    np,
    ov,
    raci_coinciden,
    raci_ok_np,
    raci_roto_np,
    replicacion,
    tabla_overrides,
    tabla_perturbaciones,
    trampas_canon,
    verificar_cadena,
    verificar_estructura,
    verificar_sello,
    gatillos,
):
    _c = []
    # 1. Canonicalización: desde cero = json.dumps; orden de claves resuelto; 1 ≠ 1.0; NFC ≠ NFD
    assert coincide_canon
    assert trampas_canon[0]["¿iguales en canónico?"] is True
    assert trampas_canon[1]["¿iguales en canónico?"] is False
    assert trampas_canon[3]["¿iguales en canónico?"] is False
    _c.append("canon desde cero = json.dumps canónico; trampas documentadas")
    # 2. Cadena íntegra, estructura y sello; HMAC distinto de SHA
    assert verificar_cadena(EVENTOS, HASHES)[0] and verificar_estructura(EVENTOS)[0]
    assert verificar_sello(EVENTOS, SELLO)
    assert verificar_cadena(EVENTOS, HASHES_HMAC, CLAVE_HMAC)[0] and HASHES_HMAC[-1] != HASHES[-1]
    assert encadenar(EVENTOS) == HASHES
    _c.append("cadena, estructura y sello íntegros")
    # 3. Matriz de ataques: la cadena solo caza al torpe; el sello caza todo; HMAC no caza truncado
    _m = matriz_ataques
    assert _m["cadena"].str.startswith("❌").tolist() == [True, False, False, False, False, False]
    assert _m["sello externo"].str.startswith("❌").all()
    assert _m.loc["truncar la cola (sin recalcular)", "cadena + estructura"].startswith("❌")
    assert _m.loc["truncar la cola (sin recalcular)", "HMAC (sin clave)"].startswith("✅")
    assert _m.drop("truncar la cola (sin recalcular)")["HMAC (sin clave)"].str.startswith("❌").all()
    _c.append("ataques: cadena solo caza torpe; sello caza los seis; HMAC no caza truncamiento")
    # 4. Sello local reescrito cuadra; el token TSA lo rechaza
    assert [f["resultado"] for f in capas_sello] == ["aceptado", "RECHAZADO", "RECHAZADO"]
    assert capas_sello[2]["trail ↔ sello presentado"] and not capas_sello[2]["sello presentado ↔ token TSA"]
    _c.append("sello local reescrito cuadra; token TSA lo rechaza")
    # 5. Merkle: recursiva = iterativa; mutación solo en la variante que duplica
    assert iterativa_igual_recursiva and mutacion_duplicando and not mutacion_rfc6962
    _c.append("Merkle RFC 6962: recursiva = iterativa; duplicar admite mutación")
    # 6. Determinismo y replicación
    assert determinista and all(replicacion.values())
    _c.append("dos corridas: misma huella; el validador replica los cuatro hashes")
    # 7. Hash de DataFrame: el canónico acierta en las 9; pandas solo en contenido real/regeneración/texto
    assert tabla_perturbaciones["canónico acierta"].all()
    assert tabla_perturbaciones.loc["un valor + 1e-9", "cambia hash pandas"]
    assert not tabla_perturbaciones.loc["ninguna (regenerar con la misma semilla)", "cambia hash pandas"]
    assert tabla_perturbaciones["cambia hash pandas"].sum() >= 6
    _c.append("hash canónico acierta 9/9; pandas es huella física")
    # 8. Model card: 7 secciones, ≥ 4 usos no previstos, limitaciones con lint, automáticas presentes
    assert list(model_card) == ["identidad", "proposito", "datos", "desempeno", "limitaciones",
                                "gobierno", "entorno"]
    assert len(model_card["proposito"]["usos_no_previstos"]) >= 4
    assert all(lint_limitacion(t) for t in limitaciones_declaradas)
    assert not lint_limitacion("El modelo tiene supuestos.")
    assert any("git no disponible" in t for t in limitaciones_automaticas)
    assert corrida["backtesting"] == "fail" and any("Backtesting OOT" in t for t in limitaciones_automaticas)
    _c.append("model card completo; limitaciones específicas; automáticas disparadas")
    # 9. RACI: la del curso pasa; la rota falla exactamente en las dos filas alteradas
    assert raci_coinciden and raci_ok_np.all() and (~raci_roto_np).sum() == 2
    _c.append("RACI del curso válida; numpy = pandas")
    # 10. Gatillos: valores calculados, no tipeados; patrón Austral (vigilancia sí, re-desarrollo no)
    assert gatillos.loc["1 · Vigilancia reforzada", "disparado"] == "SÍ"
    assert (gatillos.loc[["3a · Re-desarrollo", "3b · Re-desarrollo", "3c · Re-desarrollo"], "disparado"] == "no").all()
    assert gatillos.loc["2 · Recalibración del δ", "disparado"] == "no evaluable"
    _c.append("gatillos: vigilancia SÍ, re-desarrollo no, recalibración no evaluable")
    # 11. Overrides: binomial numpy = scipy; información blanda < PD del modelo; comercial no
    assert np.isclose(ov["p_numpy"], ov["p_scipy"], rtol=1e-6, atol=1e-12)
    for _pol, _r in tabla_overrides.iterrows():
        assert np.isclose(_r["p_numpy"], _r["p_scipy"], rtol=1e-6, atol=1e-12)
    assert tabla_overrides.loc["información blanda", "mora_overrides"] < tabla_overrides.loc["información blanda", "pd_modelo_overrides"]
    assert tabla_overrides.loc["información blanda", "mora_overrides"] < tabla_overrides.loc["comercial", "mora_overrides"]
    _c.append("overrides: binomial numpy = scipy; la información blanda se delata")
    # 12. Métricas: Gini numpy = sklearn; tiers coherentes
    assert gini_numpy_igual_sklearn
    assert inventario_tiers.loc["motos-admision-scorecard", "tier"] == 2
    assert inventario_tiers.loc["austral-scorecard-consumo", "tier"] == 1
    _c.append("Gini numpy = sklearn; tiering reproducible")
    mo.md("**Todos los checks pasaron.**\n\n" + "\n".join(f"- {x}" for x in _c))
    return


if __name__ == "__main__":
    app.run()

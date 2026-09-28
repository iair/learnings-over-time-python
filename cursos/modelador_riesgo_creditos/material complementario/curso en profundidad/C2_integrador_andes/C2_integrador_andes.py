# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25",
#     "numpy",
#     "pandas",
#     "pyarrow",
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

    return (mo,)


@app.cell
def _():
    import copy
    import hashlib
    import json
    import math
    import operator
    import platform
    import time
    import urllib.error
    import urllib.request
    import warnings
    from pathlib import Path
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import statsmodels
    import statsmodels.api as sm
    from scipy import stats
    from scipy.optimize import brentq
    from scipy.special import expit, rel_entr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    T0_NOTEBOOK = time.perf_counter()
    return (
        LogisticRegression,
        Path,
        T0_NOTEBOOK,
        brentq,
        copy,
        expit,
        hashlib,
        json,
        math,
        matplotlib,
        np,
        operator,
        pd,
        platform,
        plt,
        rel_entr,
        roc_auc_score,
        scipy,
        sklearn,
        sm,
        stats,
        statsmodels,
        time,
        urllib,
        variance_inflation_factor,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # C2 · El scorecard como pipeline declarativo sobre Financiera Andes

    **Proyecto integrador de la Serie 2.** Este notebook ejecuta el ciclo completo del curso —datos →
    embudo → scorecard → calibración → estrategia → validación → implementación y gobierno— sobre los
    **datos reales de Financiera Andes** (versión fijada por commit), dirigido por **una sola
    configuración declarativa** con huella (`config_hash`).

    - Cada etapa es una **función pura**: recibe `config` + insumos y devuelve artefactos + eventos para
      el audit trail. Nada se ajusta fuera de la muestra declarada (DEV o la ventana de calibración).
    - Dos **motores** intercambiables: `numpy` (implementación propia, auditable) y `librerias`
      (statsmodels / scipy / scikit-learn). Un **arnés de paridad** corre ambos y exige que coincidan.
    - Al final, una **grilla de sensibilidad** re-ejecuta el embudo variando convenciones (umbral PSI,
      IV, correlación, bins, α) para medir cuánto de «el modelo» es dato y cuánto es convención.

    Documento de referencia: `C2_integrador_andes.md` (arquitectura, contrato de config, resultados,
    lectura de la sensibilidad, preguntas de comité). Módulos que profundizan cada etapa: M08–M23.
    """)
    return


@app.cell
def _():
    # ============================================================================
    # CONFIGURACIÓN DECLARATIVA (la receta completa de la corrida)
    # Todo umbral, ventana, semilla y supuesto vive aquí. Los controles de la celda
    # siguiente solo SOBREESCRIBEN algunos campos; el config_hash cubre el resultado.
    # ============================================================================
    CONFIG_BASE = {
        "version_config": "C2-andes-1.0.0",
        "motor": "numpy",                       # numpy | librerias
        "datos": {
            "base_url": "https://raw.githubusercontent.com/nexolabs-gh/datos-riesgo-credito",
            "commit": "feaa1968d0808c44ff228506523b706b677e3020",   # el mismo del Lab 2
            "prefijo": "andes",
            "tablas": ["clientes", "solicitudes", "comportamiento", "bureau"],
            "sha256": {                         # contrato de la fuente: si no calza, no se corre
                "clientes": "b4bdfee9eeccaac4d9981b1239d6e9c8902b4153c4cfaf0f175db3d6373a3e53",
                "solicitudes": "1c6fd9d7d41d11d8616d41082d3bbb64cff7fa6f7c0a6545dd9144edcdfbc23c",
                "comportamiento": "aea1becc4b3f7775d6b0b03f5b0de5a8d201186b4e3802972956296f3e824b56",
                "bureau": "28cdd6b6214059f467946d483b959891a0c07fbb7ad37730d9475661f4f0bb91",
            },
            "cache_dir": "_cache_andes",
            "id_alumno": "",                    # vacío = población completa, sin submuestreo
            "sal_semilla": "ANDES-C1-2026-B",   # semilla = sha256(sal|correo)[:8] mod 1e6 (Lab 1-3)
            "frac_muestra_personal": 0.60,
            "periodo_modelacion": ["2024-09", "2025-06"],
            "ultima_cosecha_dev_ho": "2025-01",
            "frac_dev": 0.75,
            "inicio_ttd": "2025-07",
            "antiguedad_min_meses": 6,
            "horizonte_meses": 12,
            "dpd_malo": 90,
            "dpd_indeterminado": 30,
        },
        "embudo": {
            "psi_bins": 10, "psi_eps": 1e-4, "psi_max_ttd": 0.25,
            "psi_no_medible": "excluir",        # regla del curso: sin PSI medible no sigue
            "iv_min": 0.10, "bins_woe": 5, "umbral_moda": 0.35,
            "corr_max": 0.70, "vif_max": 10.0,
            "alpha_entrada": 0.05, "alpha_salida": 0.05, "max_variables": 14,
            "politica_signos": "excluir_y_reiniciar",   # o "reportar"
        },
        "scorecard": {"pdo": 20, "score_base": 600, "odds_base": 50, "n_reason_codes": 3},
        "calibracion": {
            "ancla": "ttc",                     # ttc | pit
            "ttc_cosechas": ["2024-09", "2025-01"],     # promedio simple de tasas por cosecha
            "pit_cosechas": ["2025-02", "2025-06"],     # tasa agregada del periodo reciente
            "muestra_validacion": "OOT",
        },
        "estrategia": {
            "cortes_banda": [540, 560, 580, 600, 620, 640, 660],
            "etiquetas_banda": ["E", "D", "C2", "C1", "B2", "B1", "A2", "A1"],   # peor → mejor
            "muestra": "OOT", "lgd": 0.45, "ead": "monto_solicitado",
            "grilla_cutoff": list(range(500, 661, 10)),
            "mora_max": 0.08, "aprobacion_min": 0.60,     # apetito (supuesto declarado)
            "knockouts": [["dias_mora_ult", ">", 0], ["peor_mora_sistema_ult", ">=", 30]],
        },
        "validacion": {
            "B_bootstrap": 1000, "semillas_bootstrap": {"HO": 20260916, "OOT": 20260917},
            "hl_grupos": 10, "hl_simulaciones": 10000, "hl_semilla": 20260908,
            "umbrales": {"caida_gini_rel": [0.20, 0.30], "ks": [0.30, 0.20], "psi": [0.10, 0.25],
                         "p_valor": [0.05, 0.01], "mix_de_pts": [3.0, 6.0]},
        },
        "implementacion": {
            "modelo_id": "andes-scorecard-consumo", "version": "1.0.0",
            "fecha_construccion": "2026-09-28",
            "tolerancia_missing": 3.0, "fuera_rango_aviso": 0.01, "fuera_rango_bloqueo": 0.10,
            "missing_bloqueo": 0.25, "factor_bloqueo": 4.0, "tol_paridad": 1e-9,
        },
        "sensibilidad": {
            "psi_max_ttd": [0.10, 0.25], "iv_min": [0.02, 0.10], "corr_max": [0.60, 0.70, 0.80],
            "bins_woe": [5, 10], "alpha": [0.01, 0.05],
        },
    }
    return (CONFIG_BASE,)


@app.cell
def _(
    hashlib,
    json,
    math,
    matplotlib,
    mo,
    np,
    pd,
    platform,
    scipy,
    sklearn,
    statsmodels,
):
    def c2(x, d=2):
        """Número con coma decimal y punto de miles (para prosa)."""
        if x is None:
            return "—"
        try:
            if not np.isfinite(float(x)):
                return "—"
        except (TypeError, ValueError):
            return str(x)
        return f"{float(x):,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".").replace("-", "−")


    def pct(x, d=1):
        return "—" if x is None or not np.isfinite(float(x)) else c2(100 * float(x), d) + "%"


    def limpiar(o):
        """Convierte a tipos JSON puros (numpy → python, NaN → None) para hashear y serializar."""
        if isinstance(o, dict):
            return {str(k): limpiar(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [limpiar(v) for v in o]
        if isinstance(o, (set, frozenset)):
            return [limpiar(v) for v in sorted(o, key=str)]
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (int, np.integer)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            f = float(o)
            return f if math.isfinite(f) else None
        if isinstance(o, np.ndarray):
            return limpiar(o.tolist())
        if isinstance(o, pd.Series):
            return limpiar(o.to_dict())
        if o is None or isinstance(o, str):
            return o
        return str(o)


    def json_canonico(obj):
        return json.dumps(limpiar(obj), sort_keys=True, ensure_ascii=False, allow_nan=False)


    def hash_texto(t):
        return hashlib.sha256(t.encode("utf-8")).hexdigest()


    def hash_json(obj):
        return hash_texto(json_canonico(obj))


    def hash_df(df):
        """Huella de un DataFrame: esquema (nombre, dtype, posición) + contenido con índice."""
        esquema = [{"pos": i, "nombre": str(c), "dtype": str(df[c].dtype)} for i, c in enumerate(df.columns)]
        h = hashlib.sha256()
        h.update(json.dumps(esquema, sort_keys=True).encode("utf-8"))
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
        return h.hexdigest()


    def a_yaml(d, sangria=0):
        """Render tipo YAML de un dict (solo para mostrar la config)."""
        lineas = []
        for k, v in d.items():
            pref = "  " * sangria + f"{k}:"
            if isinstance(v, dict):
                lineas.append(pref)
                lineas.append(a_yaml(v, sangria + 1))
            else:
                lineas.append(f"{pref} {json.dumps(limpiar(v), ensure_ascii=False)}")
        return "\n".join(lineas)


    def fmt_p(p, S=None):
        """p-valor legible; un p simulado 0 se reporta como cota (< 1/S)."""
        if p is None or not np.isfinite(p):
            return "—"
        if S is not None and p == 0:
            return f"< {c2(1 / S, 4)}"
        if p < 1e-4:
            return f"{p:.1e}".replace(".", ",")
        return c2(p, 4)


    def ev(paso, evento, **payload):
        """Evento para el audit trail: lo que cada etapa decide queda como dato."""
        return {"paso": paso, "evento": evento, "payload": limpiar(payload)}


    def semaforo_mayor(v, amarillo, rojo):
        if v is None or not np.isfinite(v):
            return "⚪"
        return "🔴" if v > rojo else ("🟡" if v > amarillo else "🟢")


    def semaforo_menor(v, amarillo, rojo):
        if v is None or not np.isfinite(v):
            return "⚪"
        return "🔴" if v < rojo else ("🟡" if v < amarillo else "🟢")


    def snapshot_entorno():
        return {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
                "matplotlib": matplotlib.__version__, "marimo": mo.__version__}

    return (
        a_yaml,
        c2,
        ev,
        fmt_p,
        hash_df,
        hash_json,
        hash_texto,
        limpiar,
        pct,
        semaforo_mayor,
        semaforo_menor,
        snapshot_entorno,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Controles y configuración efectiva

    Los controles sobreescriben campos de `CONFIG_BASE`; la configuración efectiva y su huella se
    muestran debajo. **`ID_ALUMNO` vacío = población completa** (sin el submuestreo de 60% del lab);
    si escribes el correo con el que hiciste los labs, se aplica la misma semilla
    (`sha256("ANDES-C1-2026-B|correo")`) y reproduces tu muestra personal. Cambiar solo umbrales no
    vuelve a descargar ni a reconstruir la matriz: la etapa de datos depende solo de su sección.
    """)
    return


@app.cell
def _(mo):
    ui_id = mo.ui.text(value="", placeholder="correo de los labs (vacío = población completa)",
                       label="ID_ALUMNO", full_width=True).form(submit_button_label="Aplicar")
    ui_motor = mo.ui.dropdown(options=["numpy", "librerias"], value="numpy", label="Motor")
    ui_ancla = mo.ui.dropdown(options={"TTC · cosechas DEV+HO": "ttc", "PIT · cosechas OOT": "pit"},
                              value="TTC · cosechas DEV+HO", label="Ancla de calibración")
    ui_psi = mo.ui.dropdown(options={"0,10": 0.10, "0,25": 0.25}, value="0,25", label="PSI máx. vs TTD")
    ui_iv = mo.ui.dropdown(options={"0,02": 0.02, "0,05": 0.05, "0,10": 0.10}, value="0,10",
                           label="IV mínimo")
    ui_corr = mo.ui.dropdown(options={"0,60": 0.60, "0,70": 0.70, "0,80": 0.80}, value="0,70",
                             label="|corr WoE| máx.")
    ui_bins = mo.ui.dropdown(options={"5": 5, "10": 10}, value="5", label="Bins WoE")
    ui_alpha = mo.ui.dropdown(options={"0,01": 0.01, "0,05": 0.05}, value="0,05", label="α stepwise")
    ui_mora = mo.ui.number(start=0.01, stop=0.30, step=0.005, value=0.08, label="Mora máx. (apetito)")
    ui_aprob = mo.ui.number(start=0.10, stop=1.00, step=0.05, value=0.60, label="Aprobación mín.")
    mo.vstack([
        ui_id,
        mo.hstack([ui_motor, ui_ancla, ui_psi, ui_iv], justify="start"),
        mo.hstack([ui_corr, ui_bins, ui_alpha, ui_mora, ui_aprob], justify="start"),
    ])
    return (
        ui_alpha,
        ui_ancla,
        ui_aprob,
        ui_bins,
        ui_corr,
        ui_id,
        ui_iv,
        ui_mora,
        ui_motor,
        ui_psi,
    )


@app.cell
def _(CONFIG_BASE, copy, ui_id):
    # La sección de datos depende SOLO de ID_ALUMNO: cambiar umbrales no reconstruye la matriz.
    cfg_datos = copy.deepcopy(CONFIG_BASE["datos"])
    cfg_datos["id_alumno"] = (ui_id.value or "").strip()
    return (cfg_datos,)


@app.cell
def _(
    CONFIG_BASE,
    a_yaml,
    cfg_datos,
    copy,
    hash_json,
    mo,
    ui_alpha,
    ui_ancla,
    ui_aprob,
    ui_bins,
    ui_corr,
    ui_iv,
    ui_mora,
    ui_motor,
    ui_psi,
):
    CONFIG = copy.deepcopy(CONFIG_BASE)
    CONFIG["motor"] = ui_motor.value
    CONFIG["datos"] = copy.deepcopy(cfg_datos)
    CONFIG["embudo"].update(psi_max_ttd=float(ui_psi.value), iv_min=float(ui_iv.value),
                            corr_max=float(ui_corr.value), bins_woe=int(ui_bins.value),
                            alpha_entrada=float(ui_alpha.value), alpha_salida=float(ui_alpha.value))
    CONFIG["calibracion"]["ancla"] = ui_ancla.value
    CONFIG["estrategia"].update(mora_max=float(ui_mora.value), aprobacion_min=float(ui_aprob.value))
    config_hash = hash_json(CONFIG)
    mo.md(f"""
    **Configuración efectiva** · `config_hash = {config_hash[:16]}…` · motor `{CONFIG['motor']}` ·
    {'población completa' if not CONFIG['datos']['id_alumno'] else 'muestra personal del correo ingresado'}

    ```yaml
    {a_yaml({k: v for k, v in CONFIG.items() if k not in ('datos',)})}
    datos:
    {a_yaml({k: v for k, v in CONFIG['datos'].items() if k not in ('sha256', 'id_alumno')}, 1)}
      id_alumno: {'"(vacío)"' if not CONFIG['datos']['id_alumno'] else '"(definido; no se imprime)"'}
    ```
    """)
    return CONFIG, config_hash


@app.cell
def _(np, pd):
    # === Herramientas del curso (Lab 2, VERBATIM) ===============================
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
        """Tabla WoE de una variable y su IV (convención: target 1 = malo)."""
        etiquetas, orden = binear(x, bins, ref=ref)
        tab = pd.DataFrame({"bin": etiquetas.values, "y": pd.Series(y).values}) \
                .groupby("bin")["y"].agg(["count", "sum"])
        tab.columns = ["n", "malos"]
        if orden:
            tab = tab.reindex([o for o in orden if o in tab.index])
        tab["buenos"] = tab["n"] - tab["malos"]
        tab["tasa_malos"] = tab["malos"] / tab["n"]
        p_malos  = (tab["malos"] + 0.5) / (tab["malos"].sum() + 0.5 * len(tab))
        p_buenos = (tab["buenos"] + 0.5) / (tab["buenos"].sum() + 0.5 * len(tab))
        tab["woe"] = np.log(p_buenos / p_malos)
        tab["iv_aporte"] = (p_buenos - p_malos) * tab["woe"]
        return tab, float(tab["iv_aporte"].sum())


    # === Extensiones mínimas (parametrizan lo que el lab fijaba) =================
    def a_woe_b(df, variables, dev, mapas, bins):
        """a_woe del Lab 2 con `bins` explícito: cortes y WoE de DEV; bin no visto → 0."""
        F = pd.DataFrame(index=range(len(df)))
        for v in variables:
            etiquetas, _ = binear(df[v], bins, ref=dev[v])
            F[v] = etiquetas.map(mapas[v]).fillna(0.0).values
        return F


    def psi_curso(esperado, actual, bins, eps, f_psi):
        """psi() de la clase 3: deciles de DEV, +eps, NaN si los deciles colapsan (masa de ceros)."""
        esperado, actual = pd.Series(esperado), pd.Series(actual)
        if not pd.api.types.is_numeric_dtype(esperado):
            e_ = esperado.fillna("MISSING").astype(str)
            a_ = actual.fillna("MISSING").astype(str)
        else:
            cortes = np.unique(np.nanquantile(esperado.dropna(), np.linspace(0, 1, bins + 1)))
            if len(cortes) < 3:
                return np.nan
            cortes[0], cortes[-1] = -np.inf, np.inf
            e_ = pd.cut(esperado, cortes).astype(str).where(esperado.notna(), "MISSING")
            a_ = pd.cut(actual, cortes).astype(str).where(actual.notna(), "MISSING")
        cats = sorted(set(e_) | set(a_))
        pe = e_.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy() + eps
        pa = a_.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy() + eps
        return f_psi(pe, pa)


    def csi_curso(base, actual, bins, eps, f_psi):
        """CSI del Lab 3: PSI sobre los bins WoE del scorecard (cortes de DEV)."""
        e_lab, _ = binear(base, bins)
        a_lab, _ = binear(actual, bins, ref=base)
        e = e_lab.value_counts(normalize=True)
        a = a_lab.value_counts(normalize=True)
        cats = sorted(set(e.index) | set(a.index))
        pe = e.reindex(cats).fillna(0).to_numpy() + eps
        pa = a.reindex(cats).fillna(0).to_numpy() + eps
        return f_psi(pe, pa)


    def con_constante(df):
        X = df.reset_index(drop=True).copy()
        X.insert(0, "const", 1.0)
        return X

    return a_woe_b, con_constante, csi_curso, psi_curso, tabla_woe


@app.cell
def _(
    brentq,
    expit,
    math,
    np,
    pd,
    rel_entr,
    roc_auc_score,
    sm,
    stats,
    variance_inflation_factor,
):
    # ============================================================================
    # MOTORES: dos implementaciones de cada cálculo central, misma interfaz.
    # ============================================================================
    def sig(z):
        """Sigmoide estable (sin overflow)."""
        z = np.asarray(z, dtype=float)
        e = np.exp(-np.abs(z))
        return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


    def logit_np_(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))


    # --- motor numpy -------------------------------------------------------------
    def np_logit(Xdf, y, tol=1e-10, max_iter=100):
        """MLE logística por Newton-Raphson (IRLS). p-valores de Wald con la normal (como statsmodels)."""
        cols = list(Xdf.columns)
        X = Xdf.to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        b = np.zeros(X.shape[1])
        convergio = False
        for _ in range(max_iter):
            eta = X @ b
            p = sig(eta)
            H = X.T @ (X * (p * (1 - p))[:, None])
            paso = np.linalg.solve(H, X.T @ (y - p))
            b = b + paso
            if np.max(np.abs(paso)) < tol:
                convergio = True
                break
        eta = X @ b
        p = sig(eta)
        H = X.T @ (X * (p * (1 - p))[:, None])
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        pv = np.array([math.erfc(abs(t) / math.sqrt(2.0)) for t in b / se])
        return {"params": pd.Series(b, index=cols), "bse": pd.Series(se, index=cols),
                "pvalues": pd.Series(pv, index=cols),
                "llf": float(np.sum(y * eta - np.logaddexp(0.0, eta))), "convergio": convergio}


    def np_auc(y, s):
        """AUC de Mann-Whitney con rangos medios en empates (s = PD: mayor = más riesgo)."""
        y = np.asarray(y, dtype=float)
        _, inv, cnt = np.unique(np.asarray(s, dtype=float), return_inverse=True, return_counts=True)
        r = (np.cumsum(cnt) - cnt + (cnt + 1) / 2.0)[inv]
        nm = y.sum()
        nb = len(y) - nm
        return float((r[y == 1].sum() - nm * (nm + 1) / 2) / (nm * nb))


    def np_ks(y, s):
        """KS = sup |F_malos − F_buenos| evaluado en todos los valores observados."""
        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)
        a, b = np.sort(s[y == 1]), np.sort(s[y == 0])
        g = np.concatenate([a, b])
        return float(np.max(np.abs(np.searchsorted(a, g, side="right") / len(a)
                                   - np.searchsorted(b, g, side="right") / len(b))))


    def np_vif(W):
        """VIF_j = [R⁻¹]_jj con R la matriz de correlación (invariante a estandarizar)."""
        if W.shape[1] == 1:
            return pd.Series([1.0], index=W.columns)
        R = np.corrcoef(W.to_numpy(dtype=float), rowvar=False)
        return pd.Series(np.diag(np.linalg.inv(R)), index=W.columns)


    def np_delta(lp, objetivo, tol=1e-13):
        """δ tal que mean(σ(lp + δ)) = objetivo, por Newton 1-D."""
        lp = np.asarray(lp, dtype=float)
        d = 0.0
        for _ in range(200):
            p = sig(lp + d)
            paso = (p.mean() - objetivo) / np.mean(p * (1 - p))
            d -= paso
            if abs(paso) < tol:
                break
        return float(d)


    def np_binom_p(k, n, p):
        """Binomial exacta bilateral, método «minlike» (el de scipy.stats.binomtest)."""
        k, n = int(k), int(n)
        if n == 0:
            return float("nan")
        i = np.arange(n + 1)
        log_comb = np.concatenate([[0.0], np.cumsum(np.log(np.arange(n, 0, -1)) - np.log(np.arange(1, n + 1)))])
        pmf = np.exp(log_comb + i * np.log(p) + (n - i) * np.log1p(-p))
        if k == p * n:
            return 1.0
        return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-7)].sum()))


    def _gamma_q(a, x):
        """Gamma incompleta regularizada superior Q(a, x) (serie / fracción continua de Lentz)."""
        if x <= 0:
            return 1.0
        gln = math.lgamma(a)
        if x < a + 1:
            ap, s = a, 1.0 / a
            d = s
            for _ in range(10_000):
                ap += 1
                d *= x / ap
                s += d
                if abs(d) < abs(s) * 1e-16:
                    break
            return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
        tiny = 1e-300
        b = x + 1 - a
        c = 1 / tiny
        d = 1 / b
        h = d
        for i in range(1, 10_000):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            d = tiny if abs(d) < tiny else d
            c = b + an / c
            c = tiny if abs(c) < tiny else c
            d = 1 / d
            h *= d * c
            if abs(d * c - 1) < 1e-16:
                break
        return math.exp(-x + a * math.log(x) - gln) * h


    def np_chi2_sf(x, gl):
        return float(_gamma_q(gl / 2.0, x / 2.0))


    def np_psi(pe, pa):
        pe, pa = np.asarray(pe, dtype=float), np.asarray(pa, dtype=float)
        return float(np.sum((pa - pe) * np.log(pa / pe)))


    # --- motor librerías -------------------------------------------------------------
    def lib_logit(Xdf, y):
        r = sm.Logit(np.asarray(y, dtype=float), Xdf).fit(disp=0, maxiter=200)
        return {"params": r.params, "bse": r.bse, "pvalues": r.pvalues, "llf": float(r.llf),
                "convergio": bool(r.mle_retvals["converged"])}


    def lib_ks(y, s):
        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)
        return float(stats.ks_2samp(s[y == 1], s[y == 0]).statistic)


    def lib_vif(W):
        if W.shape[1] == 1:
            return pd.Series([1.0], index=W.columns)
        Z = (W - W.mean()) / W.std()
        Xc = sm.add_constant(Z, has_constant="add").to_numpy()
        return pd.Series([variance_inflation_factor(Xc, j + 1) for j in range(W.shape[1])], index=W.columns)


    def lib_delta(lp, objetivo):
        lp = np.asarray(lp, dtype=float)
        return float(brentq(lambda d: expit(lp + d).mean() - objetivo, -5, 5, xtol=1e-13))


    MOTORES = {
        "numpy": {"nombre": "numpy", "logit": np_logit, "auc": np_auc, "ks": np_ks, "vif": np_vif,
                  "delta": np_delta, "binom_p": np_binom_p, "chi2_sf": np_chi2_sf, "psi": np_psi},
        "librerias": {"nombre": "librerias", "logit": lib_logit,
                      "auc": lambda y, s: float(roc_auc_score(y, s)), "ks": lib_ks, "vif": lib_vif,
                      "delta": lib_delta,
                      "binom_p": lambda k, n, p: float(stats.binomtest(int(k), int(n), p).pvalue),
                      "chi2_sf": lambda x, gl: float(stats.chi2.sf(x, gl)),
                      "psi": lambda pe, pa: float(np.sum(rel_entr(pa, pe) + rel_entr(pe, pa)))},
    }
    return MOTORES, logit_np_, sig


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Datos: fuente fijada, caché y la matriz del Lab 2

    La etapa 1 es la única con efectos (red y disco): descarga los 4 parquet del **commit fijado** a
    `./_cache_andes/`, verifica su SHA-256 contra la config y, si no hay red ni caché, se detiene con un
    mensaje (sin traceback). Luego reproduce **exactamente** la preparación del Lab 2: población
    (cursadas 2024-09…2025-06, primera por cliente, sin fraude, antigüedad ≥ 6 meses), target 90+ a 12
    meses, indeterminados 30–89 fuera, DEV/HO al azar 75/25 hasta 2025-01, OOT 2025-02…06, TTD desde
    2025-07, fábrica de 22 familias × ventanas, fotos «_ult» y ratios. Profundiza: Serie 1 · M2–M5.
    """)
    return


@app.cell
def _(Path, ev, hash_df, hashlib, np, pct, pd, urllib, warnings):
    def descargar_crudas(cfg_d, carpeta):
        """Lee los parquet desde caché o red. Devuelve (tablas, info). Lanza OSError/ValueError."""
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        tablas, info = {}, {}
        for t in cfg_d["tablas"]:
            nombre = f"{cfg_d['prefijo']}_{t}.parquet"
            destino = carpeta / nombre
            origen = "cache"
            if not destino.exists():
                url = f"{cfg_d['base_url']}/{cfg_d['commit']}/{nombre}"
                with urllib.request.urlopen(url, timeout=60) as r:
                    contenido = r.read()
                tmp = destino.with_suffix(".tmp")
                tmp.write_bytes(contenido)
                tmp.replace(destino)
                origen = "red"
            crudo = destino.read_bytes()
            h = hashlib.sha256(crudo).hexdigest()
            if h != cfg_d["sha256"][t]:
                destino.unlink(missing_ok=True)
                raise ValueError(f"{nombre}: SHA-256 {h[:12]}… no coincide con el contrato "
                                 f"{cfg_d['sha256'][t][:12]}… (se borró la copia local)")
            tablas[t] = pd.read_parquet(destino)
            info[t] = {"sha256": h, "bytes": len(crudo), "origen": origen,
                       "filas": int(len(tablas[t]))}
        return tablas, info


    def etapa_datos(cfg_d, crudas):
        """Población, target, muestras y fábrica de variables: el Lab 2 literal, parametrizado."""
        eventos = []
        clientes, solicitudes = crudas["clientes"], crudas["solicitudes"]
        comportamiento, bureau = crudas["comportamiento"], crudas["bureau"]
        id_al = cfg_d["id_alumno"].strip().lower()
        semilla = int(hashlib.sha256(f"{cfg_d['sal_semilla']}|{id_al}".encode()).hexdigest()[:8], 16) % 10**6
        if id_al:
            rng_p = np.random.default_rng(semilla)
            ids = np.sort(clientes["id_cliente"].unique())
            mis = set(rng_p.choice(ids, size=int(len(ids) * cfg_d["frac_muestra_personal"]), replace=False))
            clientes = clientes[clientes["id_cliente"].isin(mis)].reset_index(drop=True)
            solicitudes = solicitudes[solicitudes["id_cliente"].isin(mis)].reset_index(drop=True)
            comportamiento = comportamiento[comportamiento["id_cliente"].isin(mis)].reset_index(drop=True)
            bureau = bureau[bureau["id_cliente"].isin(mis)].reset_index(drop=True)
            modo = f"muestra personal ({pct(cfg_d['frac_muestra_personal'], 0)} de clientes, semilla del lab)"
        else:
            modo = "población completa (sin submuestreo)"
        eventos.append(ev("datos", "muestra_definida", modo=modo, semilla=semilla,
                          clientes=len(clientes), solicitudes=len(solicitudes)))

        # ---- Lab 1/2 · parte A: target, población y muestras ----
        mora_w = comportamiento.pivot_table(index="id_cliente", columns="mes",
                                            values="dias_mora", aggfunc="first")
        meses = list(mora_w.columns)
        pos_mes = {m: k for k, m in enumerate(meses)}
        M = mora_w.to_numpy()
        fila_de = {c: k for k, c in enumerate(mora_w.index)}
        H = cfg_d["horizonte_meses"]
        cursadas = solicitudes[solicitudes["aprobada"]].copy()
        cursadas["_t"] = cursadas["fecha_solicitud"].map(pos_mes)
        cursadas["_i"] = cursadas["id_cliente"].map(fila_de)
        peor = np.full(len(cursadas), np.nan)
        for k, (i, t0) in enumerate(zip(cursadas["_i"], cursadas["_t"])):
            if t0 + H <= len(meses) - 1:
                peor[k] = M[i, t0 + 1: t0 + H + 1].max()
        cursadas["peor_dpd_12m"] = peor
        cursadas["malo"] = np.where(np.isnan(peor), np.nan, (peor >= cfg_d["dpd_malo"]).astype(float))
        cursadas["indeterminado"] = (peor >= cfg_d["dpd_indeterminado"]) & (peor < cfg_d["dpd_malo"])

        antig = clientes.set_index("id_cliente")["fecha_alta_cliente"].map(
            lambda s: (pd.Period(s, freq="M") - pd.Period(meses[0], freq="M")).n)
        p0, p1 = cfg_d["periodo_modelacion"]
        periodo = cursadas[cursadas["fecha_solicitud"].between(p0, p1)]
        periodo = periodo.sort_values("fecha_solicitud", kind="stable").drop_duplicates(
            "id_cliente", keep="first").copy()
        periodo["antiguedad_meses"] = periodo["_t"] - periodo["id_cliente"].map(antig)
        base = periodo[periodo["malo"].notna() & ~periodo["marca_fraude"]
                       & (periodo["antiguedad_meses"] >= cfg_d["antiguedad_min_meses"])].copy()
        rng = np.random.default_rng(semilla)
        base["muestra"] = np.where(base["fecha_solicitud"] <= cfg_d["ultima_cosecha_dev_ho"],
                                   np.where(rng.random(len(base)) < cfg_d["frac_dev"], "DEV", "HO"), "OOT")
        ttd_pob = cursadas[cursadas["fecha_solicitud"] >= cfg_d["inicio_ttd"]].copy()
        ttd_pob["antiguedad_meses"] = ttd_pob["_t"] - ttd_pob["id_cliente"].map(antig)
        ttd_pob = ttd_pob[ttd_pob["antiguedad_meses"] >= cfg_d["antiguedad_min_meses"]]
        ttd_pob["muestra"] = "TTD"
        poblacion = pd.concat([base, ttd_pob], ignore_index=True)

        # ---- Lab 1/2 · parte B: panel, fábrica, fotos y ratios ----
        def pivotar(df, columna):
            ancho = df.pivot_table(index="id_cliente", columns="mes", values=columna, aggfunc="first")
            return ancho.reindex(index=mora_w.index, columns=meses).to_numpy()

        PANEL = {c: pivotar(comportamiento, c) for c in
                 ["dias_mora", "saldo_linea", "cupo_linea", "saldo_tc", "cupo_tc", "saldo_consumo",
                  "monto_pagado", "monto_facturado", "abonos_cuenta", "saldo_ahorro", "n_productos"]}
        PANEL.update({c: pivotar(bureau, c) for c in
                      ["deuda_otras_inst", "n_otras_inst", "consultas_mes", "peor_mora_sistema"]})
        with np.errstate(divide="ignore", invalid="ignore"):
            PANEL["uso_linea"] = np.where(PANEL["cupo_linea"] > 0,
                                          PANEL["saldo_linea"] / PANEL["cupo_linea"], np.nan)
            PANEL["uso_tc"] = np.where(PANEL["cupo_tc"] > 0,
                                       PANEL["saldo_tc"] / PANEL["cupo_tc"], np.nan)
        PANEL["deuda_interna"] = PANEL["saldo_linea"] + PANEL["saldo_tc"] + PANEL["saldo_consumo"]
        PANEL["deuda_total"] = PANEL["deuda_interna"] + PANEL["deuda_otras_inst"]
        PANEL["en_mora"] = np.where(np.isnan(PANEL["dias_mora"]), np.nan,
                                    (PANEL["dias_mora"] > 0).astype(float))
        PANEL["en_mora_sistema"] = np.where(np.isnan(PANEL["peor_mora_sistema"]), np.nan,
                                            (PANEL["peor_mora_sistema"] > 0).astype(float))

        def prom(V):
            return np.nanmean(V, axis=1)

        def vmax(V):
            return np.nanmax(V, axis=1)

        def suma(V):
            return np.nansum(V, axis=1)

        def delta(V):
            return V[:, -1] / np.maximum(V[:, 0], 1) - 1

        def recencia(V):
            evento = np.nan_to_num(V, nan=0.0) > 0
            k = evento.shape[1]
            ultimo = k - 1 - np.argmax(evento[:, ::-1], axis=1)
            return np.where(evento.any(axis=1), k - ultimo, k + 1)

        AGREGADORES = {"prom": prom, "max": vmax, "suma": suma, "delta": delta, "recencia": recencia}

        def agregar_ventana(matriz, filas, t0s, k, funcion):
            salida = np.full(len(filas), np.nan)
            for t0 in np.unique(t0s):
                sel = t0s == t0
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    salida[sel] = funcion(matriz[filas[sel], max(t0 - k, 0):t0])   # t0 NO entra
            return salida

        FABRICA = [
            ("dias_mora", "max", (3, 6, 12), "dias_mora_max_{k}m"),
            ("dias_mora", "prom", (3, 6, 12), "dias_mora_prom_{k}m"),
            ("en_mora", "suma", (3, 6, 12), "n_meses_mora_{k}m"),
            ("en_mora", "recencia", (12,), "meses_desde_mora_12m"),
            ("uso_linea", "prom", (3, 6, 12), "uso_linea_prom_{k}m"),
            ("uso_linea", "max", (3, 6, 12), "uso_linea_max_{k}m"),
            ("uso_tc", "prom", (3, 6, 12), "uso_tc_prom_{k}m"),
            ("uso_tc", "max", (3, 6, 12), "uso_tc_max_{k}m"),
            ("saldo_consumo", "prom", (3, 6, 12), "saldo_consumo_prom_{k}m"),
            ("deuda_interna", "prom", (3, 6, 12), "deuda_interna_prom_{k}m"),
            ("deuda_total", "prom", (3, 6, 12), "deuda_total_prom_{k}m"),
            ("deuda_total", "delta", (6, 12), "deuda_total_delta_{k}m"),
            ("monto_pagado", "suma", (3, 6, 12), "pagos_{k}m"),
            ("monto_facturado", "suma", (3, 6, 12), "facturacion_{k}m"),
            ("abonos_cuenta", "prom", (3, 6, 12), "abonos_prom_{k}m"),
            ("saldo_ahorro", "prom", (3, 6, 12), "saldo_ahorro_prom_{k}m"),
            ("deuda_otras_inst", "prom", (3, 6, 12), "deuda_otras_prom_{k}m"),
            ("deuda_otras_inst", "delta", (6,), "delta_deuda_otras_{k}m"),
            ("n_otras_inst", "prom", (3, 6, 12), "n_otras_inst_prom_{k}m"),
            ("peor_mora_sistema", "max", (3, 6, 12), "peor_mora_sistema_{k}m"),
            ("consultas_mes", "suma", (3, 6, 12), "consultas_{k}m"),
            ("consultas_mes", "recencia", (12,), "meses_desde_consulta_12m"),
        ]
        META = ["id_solicitud", "id_cliente", "fecha_solicitud", "_t", "_i",
                "muestra", "malo", "indeterminado", "peor_dpd_12m"]
        filas_p, t0s_p = poblacion["_i"].to_numpy(), poblacion["_t"].to_numpy()
        X = poblacion[META + ["antiguedad_meses", "monto_solicitado", "plazo_meses",
                              "destino", "canal"]].copy()
        for serie, agg, ventanas, plantilla in FABRICA:
            for k in ventanas:
                X[plantilla.format(k=k)] = agregar_ventana(PANEL[serie], filas_p, t0s_p, k,
                                                           AGREGADORES[agg])
        t_ult = np.maximum(t0s_p - 1, 0)
        for serie in ["dias_mora", "saldo_linea", "cupo_linea", "saldo_tc", "cupo_tc",
                      "saldo_consumo", "saldo_ahorro", "deuda_otras_inst", "n_productos",
                      "peor_mora_sistema", "uso_linea", "uso_tc"]:
            X[f"{serie}_ult"] = PANEL[serie][filas_p, t_ult]
        ingreso = X["abonos_prom_6m"].to_numpy()
        deuda_ult = (X["saldo_linea_ult"] + X["saldo_tc_ult"]
                     + X["saldo_consumo_ult"] + X["deuda_otras_inst_ult"])
        X["carga_financiera"] = deuda_ult / np.maximum(ingreso, 1)
        X["pago_sobre_fact_6m"] = X["pagos_6m"] / np.maximum(X["facturacion_6m"], 1)
        X["pago_sobre_fact_12m"] = X["pagos_12m"] / np.maximum(X["facturacion_12m"], 1)
        X["deuda_int_sobre_sistema"] = (X["saldo_linea_ult"] + X["saldo_tc_ult"]
                                        + X["saldo_consumo_ult"]) / np.maximum(deuda_ult, 1)
        X["ahorro_sobre_ingreso"] = X["saldo_ahorro_ult"] / np.maximum(ingreso, 1)
        X["monto_sobre_ingreso"] = X["monto_solicitado"] / np.maximum(ingreso, 1)
        X["cuota_est_sobre_ingreso"] = (X["monto_solicitado"] / X["plazo_meses"]) / np.maximum(ingreso, 1)
        X = X.merge(clientes[["id_cliente", "edad", "sexo", "region", "estado_civil",
                              "nivel_educacional", "tipo_empleo", "renta_liquida",
                              "n_dependientes"]], on="id_cliente", how="left")
        X["renta_vs_abonos"] = X["renta_liquida"] / np.maximum(ingreso, 1)

        predictoras = [c for c in X.columns if c not in META]
        filtro = X["malo"].notna() & ~X["indeterminado"]
        muestras = {"DEV": X[(X["muestra"] == "DEV") & filtro],
                    "HO": X[(X["muestra"] == "HO") & filtro],
                    "OOT": X[(X["muestra"] == "OOT") & filtro],
                    "TTD": X[X["muestra"] == "TTD"]}
        resumen = pd.DataFrame({
            "n": {m: len(d) for m, d in muestras.items()},
            "malos": {m: (int(d["malo"].sum()) if m != "TTD" else np.nan) for m, d in muestras.items()},
            "tasa_malos": {m: (float(d["malo"].mean()) if m != "TTD" else np.nan) for m, d in muestras.items()},
            "cosechas": {m: f"{d['fecha_solicitud'].min()} … {d['fecha_solicitud'].max()}" for m, d in muestras.items()},
        })
        n_indet = int((X["indeterminado"] & (X["muestra"] != "TTD")).sum())
        eventos.append(ev("datos", "matriz_construida", filas=len(X), candidatas=len(predictoras),
                          n_por_muestra={m: len(d) for m, d in muestras.items()},
                          indeterminados_excluidos=n_indet,
                          hash_matriz=hash_df(X[predictoras + ["malo", "muestra"]])))
        return {"X": X, "predictoras": predictoras, "muestras": muestras, "semilla": semilla,
                "modo": modo, "resumen": resumen, "indeterminados": n_indet,
                "meses": (meses[0], meses[-1]), "eventos": eventos}

    return descargar_crudas, etapa_datos


@app.cell
def _(
    Path,
    c2,
    cfg_datos,
    descargar_crudas,
    etapa_datos,
    hash_json,
    mo,
    time,
    urllib,
):
    _carpeta_cache = Path(mo.notebook_dir() or ".") / cfg_datos["cache_dir"]
    _t = time.perf_counter()
    try:
        _crudas, info_fuentes = descargar_crudas(cfg_datos, _carpeta_cache)
        _error_datos = None
    except (OSError, urllib.error.URLError, ValueError) as _e:
        _crudas, info_fuentes, _error_datos = None, None, f"{type(_e).__name__}: {_e}"
    mo.stop(_error_datos is not None, mo.md(f"""
    ### ⛔ Sin datos: la corrida se detiene aquí
    No se pudieron obtener los parquet de Financiera Andes (commit fijado) ni desde la red ni desde
    `{_carpeta_cache}`.

    `{_error_datos}`

    Qué hacer: verifica la conexión a `raw.githubusercontent.com` o copia los 4 archivos
    `andes_*.parquet` a esa carpeta. Ninguna etapa posterior corre sin la fuente certificada.
    """))
    datos = etapa_datos(cfg_datos, _crudas)
    data_hash = hash_json({t: i["sha256"] for t, i in info_fuentes.items()})
    t_datos = time.perf_counter() - _t
    mo.vstack([
        mo.md(f"""
    **Fuente**: commit `{cfg_datos['commit'][:10]}` · `data_hash = {data_hash[:16]}…` ·
    origen {', '.join(sorted({i['origen'] for i in info_fuentes.values()}))} · {c2(t_datos, 1)} s.
    **Modo**: {datos['modo']} · semilla {'(fórmula del lab con ID vacío) ' if not cfg_datos['id_alumno'] else ''}{datos['semilla']}
    (se usa en la partición DEV/HO; con ID vacío no hay submuestreo, pero el azar DEV/HO igual necesita semilla y se declara).
    **Matriz**: {c2(len(datos['X']), 0)} filas × {len(datos['predictoras'])} candidatas · {c2(datos['indeterminados'], 0)} indeterminados (30–89 DPD) fuera
    de DEV/HO/OOT · panel {datos['meses'][0]} … {datos['meses'][1]}.
    """),
        datos["resumen"],
    ])
    return data_hash, datos, info_fuentes


@app.cell
def _(mo):
    mo.md(r"""
    ## 2–6. Las etapas como funciones puras

    Cada función recibe `(config, insumos, motor)` y devuelve un dict de artefactos con una lista
    `eventos`. Orden del embudo (clase 3): **PSI vs TTD** (regla del curso: PSI no medible = fuera) →
    **IV ≥ umbral** (en DEV) → **correlación WoE greedy** por IV → **VIF** iterativo → **stepwise forward
    con revisión** + **política de signos** (β > 0 ⇒ se excluye la variable y se re-corre). Luego
    scorecard (PDO 20 / 600 / 50:1), calibración (TTC y PIT, ambas calculadas; la config elige el
    ancla), master scale y estrategia, y validación completa. Profundizan: M08 (PSI), M09 (corr/VIF),
    M10–M11 (logística y stepwise), M13–M14 (scorecard y reason codes), M15 (calibración), M16–M18
    (bandas, cutoff, swap-set), M12/M19/M20 (validación y tablero).
    """)
    return


@app.cell
def _(a_woe_b, con_constante, ev, np, pd, psi_curso, tabla_woe):
    def insumos_woe(datos, bins):
        """IV, mapas WoE y matrices WoE (DEV/HO/OOT) de TODAS las candidatas. No depende del motor."""
        dev = datos["muestras"]["DEV"]
        iv, mapas = {}, {}
        for v in datos["predictoras"]:
            tab, iv_v = tabla_woe(dev[v], dev["malo"], bins=bins)
            iv[v], mapas[v] = iv_v, tab["woe"]
        W = {m: a_woe_b(datos["muestras"][m], datos["predictoras"], dev, mapas, bins)
             for m in ("DEV", "HO", "OOT")}
        return {"bins": bins, "iv": pd.Series(iv), "mapas": mapas, "W": W}


    def tabla_psi(cfg, datos, motor):
        """PSI de todas las candidatas DEV→OOT y DEV→TTD con la función del motor."""
        E = cfg["embudo"]
        m_ = datos["muestras"]
        filas = {v: {"psi_oot": psi_curso(m_["DEV"][v], m_["OOT"][v], E["psi_bins"], E["psi_eps"], motor["psi"]),
                     "psi_ttd": psi_curso(m_["DEV"][v], m_["TTD"][v], E["psi_bins"], E["psi_eps"], motor["psi"])}
                 for v in datos["predictoras"]}
        return pd.DataFrame(filas).T


    def stepwise(W, y, candidatas, motor, a_in, a_out, max_vars):
        """Forward con revisión (clase 3 / Lab 2 §5). Devuelve (elegidas, bitácora)."""
        elegidas, restantes, bitacora, vistos = [], list(candidatas), [], set()

        def ajustar(cols):
            return motor["logit"](con_constante(W[cols]), y)

        while restantes and len(elegidas) < max_vars:
            mejor = None
            for v in restantes:
                m = ajustar(elegidas + [v])
                if m["pvalues"][v] < a_in and (mejor is None or m["llf"] > mejor[1]):
                    mejor = (v, m["llf"], float(m["pvalues"][v]))
            if mejor is None:
                bitacora.append({"accion": "fin", "motivo": "ninguna candidata con p < α de entrada"})
                break
            elegidas.append(mejor[0])
            restantes.remove(mejor[0])
            bitacora.append({"accion": "entra", "variable": mejor[0], "p": mejor[2], "llf": mejor[1]})
            while True:                                  # revisión: sale LA peor y se re-estima
                m = ajustar(elegidas)
                pv = m["pvalues"][elegidas]
                if pv.max() < a_out:
                    break
                peor = pv.idxmax()
                elegidas.remove(peor)
                restantes.append(peor)
                bitacora.append({"accion": "sale", "variable": peor, "p": float(pv.max())})
            estado = frozenset(elegidas)
            if estado in vistos:                         # guarda anti-ciclo (declarada)
                bitacora.append({"accion": "fin", "motivo": "ciclo entra/sale detectado"})
                break
            vistos.add(estado)
        else:
            bitacora.append({"accion": "fin", "motivo": "tope de variables o sin candidatas"})
        return elegidas, bitacora


    def etapa_embudo(cfg, datos, motor, ins, psi_tab):
        E = cfg["embudo"]
        eventos = []
        dev, ho = datos["muestras"]["DEV"], datos["muestras"]["HO"]
        y_dev, y_ho = dev["malo"].to_numpy(), ho["malo"].to_numpy()
        tab = psi_tab.copy()
        tab["iv"] = ins["iv"].reindex(tab.index)
        tab["estado"] = np.select([tab["psi_ttd"].isna(), tab["psi_ttd"] > E["psi_max_ttd"]],
                                  ["no_medible", "fuera_psi"], "estable")
        fuera = tab.index[tab["estado"] == "fuera_psi"].tolist()
        no_med = tab.index[tab["estado"] == "no_medible"].tolist()
        estables = [v for v in datos["predictoras"] if tab.loc[v, "estado"] == "estable"]
        eventos.append(ev("embudo", "psi", regla=f"psi_ttd > {E['psi_max_ttd']} fuera; NaN fuera",
                          fuera={v: tab.loc[v, "psi_ttd"] for v in fuera},
                          no_medibles={v: tab.loc[v, "iv"] for v in no_med}, estables=len(estables)))
        cand = (tab.loc[estables].loc[lambda d: d["iv"] >= E["iv_min"]]
                .sort_values("iv", ascending=False, kind="stable").index.tolist())
        eventos.append(ev("embudo", "iv", regla=f"iv >= {E['iv_min']} (DEV)", pasan=len(cand),
                          descartadas=len(estables) - len(cand)))
        Wd = ins["W"]["DEV"]
        corr = Wd[cand].corr().abs() if cand else pd.DataFrame()
        pares_090 = int((np.triu(corr.to_numpy(), 1) > 0.90).sum()) if cand else 0
        seleccion, descartes_corr = [], {}
        for v in cand:
            if seleccion:
                c = corr.loc[v, seleccion]
                if c.max() > E["corr_max"]:
                    descartes_corr[v] = {"por": c.idxmax(), "rho": float(c.max())}
                    continue
            seleccion.append(v)
        eventos.append(ev("embudo", "correlacion", regla=f"|rho WoE| > {E['corr_max']} con una ya elegida",
                          pasan=len(seleccion), descartes=descartes_corr, pares_sobre_090=pares_090))
        sel_vif, vif_fuera = list(seleccion), {}
        while len(sel_vif) > 1:
            vifs = motor["vif"](Wd[sel_vif])
            if vifs.max() <= E["vif_max"]:
                break
            vif_fuera[vifs.idxmax()] = float(vifs.max())
            sel_vif.remove(vifs.idxmax())
        vif_final = motor["vif"](Wd[sel_vif]).sort_values(ascending=False) if sel_vif else pd.Series(dtype=float)
        eventos.append(ev("embudo", "vif", regla=f"VIF > {E['vif_max']} sale (el mayor, iterativo)",
                          fuera=vif_fuera, vif_max=float(vif_final.max()) if len(vif_final) else None))
        excluidas_signo, corridas_sw = [], 0
        while True:
            base_sw = [v for v in sel_vif if v not in excluidas_signo]
            elegidas, bitacora = stepwise(Wd, y_dev, base_sw, motor, E["alpha_entrada"],
                                          E["alpha_salida"], E["max_variables"])
            corridas_sw += 1
            modelo = motor["logit"](con_constante(Wd[elegidas]), y_dev)
            positivos = [v for v in elegidas if modelo["params"][v] > 0]
            if not positivos or E["politica_signos"] != "excluir_y_reiniciar":
                break
            peor = max(positivos, key=lambda v: modelo["params"][v])
            excluidas_signo.append(peor)
            eventos.append(ev("embudo", "signo_excluida", variable=peor,
                              beta=float(modelo["params"][peor]),
                              regla="WoE alto = bin bueno ⇒ β < 0; se excluye y se re-corre"))
        for paso in bitacora:
            eventos.append(ev("embudo", f"stepwise_{paso['accion']}", **paso))
        m_ho = motor["logit"](con_constante(ins["W"]["HO"][elegidas]), y_ho)
        coef = pd.DataFrame({"beta": modelo["params"], "se": modelo["bse"], "p": modelo["pvalues"],
                             "beta_HO": m_ho["params"], "p_HO": m_ho["pvalues"]})
        coef["signo_invertido_HO"] = np.sign(coef["beta"]) != np.sign(coef["beta_HO"])
        coef["iv"] = ins["iv"].reindex(coef.index)
        eventos.append(ev("modelo", "modelo_ajustado", variables=elegidas,
                          beta={k: float(v) for k, v in modelo["params"].items()},
                          llf=modelo["llf"], convergio=modelo["convergio"],
                          invertidos_HO=coef.index[coef["signo_invertido_HO"]].tolist()))
        conteo = [("candidatas", len(datos["predictoras"])),
                  ("PSI medible y ≤ umbral", len(estables)), ("IV ≥ umbral", len(cand)),
                  ("corr WoE ≤ umbral", len(seleccion)), ("VIF ≤ umbral", len(sel_vif)),
                  ("stepwise + signos", len(elegidas))]
        return {"tabla": tab, "fuera_psi": fuera, "no_medibles": no_med, "estables": estables,
                "candidatas_iv": cand, "seleccion": seleccion, "descartes_corr": descartes_corr,
                "pares_090": pares_090, "sel_vif": sel_vif, "vif": vif_final, "vif_fuera": vif_fuera,
                "elegidas": elegidas, "bitacora": bitacora, "excluidas_signo": excluidas_signo,
                "corridas_stepwise": corridas_sw, "modelo": modelo, "coef": coef, "conteo": conteo,
                "eventos": eventos}

    return etapa_embudo, insumos_woe, tabla_psi


@app.cell
def _(a_woe_b, c2, ev, np, pd, sig):
    TEXTOS_MOTIVO = [
        ("meses_desde_mora", "mora propia reciente"), ("meses_desde_consulta", "consultas de crédito recientes"),
        ("n_meses_mora", "meses con mora propia"), ("dias_mora", "días de mora propia"),
        ("peor_mora_sistema", "mora en el sistema financiero"), ("uso_linea", "línea de crédito muy utilizada"),
        ("uso_tc", "tarjeta de crédito cargada"), ("cupo", "cupo de crédito bajo"),
        ("consultas", "muchas consultas de crédito"), ("antiguedad", "poca antigüedad como cliente"),
        ("edad", "edad (revisar: variable protegida en algunas jurisdicciones)"),
        ("deuda_otras", "deuda en otras instituciones"), ("delta_deuda_otras", "alza de deuda en otras instituciones"),
        ("n_otras_inst", "número de acreedores"), ("deuda", "nivel de endeudamiento"),
        ("carga_financiera", "carga financiera alta"), ("pago_sobre_fact", "pagos bajos respecto de lo facturado"),
        ("ahorro", "bajo nivel de ahorro"), ("abonos", "abonos bajos en cuenta"), ("saldo", "saldos altos"),
        ("monto", "monto alto respecto del ingreso"), ("cuota", "cuota alta respecto del ingreso"),
        ("renta", "renta declarada"), ("plazo", "plazo solicitado"), ("tipo_empleo", "tipo de empleo"),
        ("nivel_educacional", "nivel educacional"), ("estado_civil", "estado civil"), ("region", "región"),
        ("n_dependientes", "número de dependientes"), ("destino", "destino del crédito"), ("canal", "canal de venta"),
        ("n_productos", "pocos productos con la institución"), ("facturacion", "facturación"), ("pagos", "pagos"),
    ]


    def texto_motivo(v):
        for pref, txt in TEXTOS_MOTIVO:
            if v.startswith(pref):
                return txt
        return f"puntaje bajo en {v}"


    def etapa_scorecard(cfg, datos, emb, ins, motor):
        S = cfg["scorecard"]
        bins = cfg["embudo"]["bins_woe"]
        factor = S["pdo"] / np.log(2)
        offset = S["score_base"] - factor * np.log(S["odds_base"])
        el = emb["elegidas"]
        beta = emb["modelo"]["params"]
        n, b0 = len(el), float(beta["const"])
        dev = datos["muestras"]["DEV"]
        mapas = {v: ins["mapas"][v] for v in el}
        filas = []
        for v in el:
            for b, w in mapas[v].items():
                filas.append({"variable": v, "bin": b, "woe": float(w),
                              "puntos": -(float(beta[v]) * float(w) + b0 / n) * factor + offset / n})
        puntos = pd.DataFrame(filas)
        rango = (puntos.groupby("variable")["puntos"].agg(["min", "max"])
                 .assign(rango=lambda d: d["max"] - d["min"]).sort_values("rango", ascending=False))
        Wm = {m: a_woe_b(df, el, dev, mapas, bins) for m, df in datos["muestras"].items()}
        bvec = beta[el].to_numpy(dtype=float)
        LP = {m: b0 + Wm[m].to_numpy(dtype=float) @ bvec for m in Wm}
        PD_RAW = {m: sig(LP[m]) for m in LP}
        SCORE_RAW = {m: offset - factor * LP[m] for m in LP}
        # puntos por caso: suma de puntos == score crudo (invariante)
        Pt = {m: pd.DataFrame({v: -(float(beta[v]) * Wm[m][v].to_numpy() + b0 / n) * factor + offset / n
                               for v in el}) for m in ("DEV", "TTD")}
        dif_suma = max(float(np.max(np.abs(Pt[m].sum(axis=1).to_numpy() - SCORE_RAW[m]))) for m in Pt)
        # reason codes (clase 3): brecha = máximo de la variable − puntos obtenidos
        maximo = puntos.groupby("variable")["puntos"].max()
        brecha = maximo[el].to_numpy()[None, :] - Pt["TTD"].to_numpy()
        k = S["n_reason_codes"]
        orden = np.argsort(-brecha, axis=1, kind="stable")[:, :k]
        s_ttd = SCORE_RAW["TTD"]
        pos = np.argsort(s_ttd, kind="stable")
        ejemplos = []
        for q in (0.01, 0.50, 0.99):
            i = int(pos[int(q * (len(pos) - 1))])
            ejemplos.append({"caso_ttd": datos["muestras"]["TTD"]["id_solicitud"].iloc[i],
                             "score_crudo": float(s_ttd[i]), "pd_cruda": float(PD_RAW["TTD"][i]),
                             **{f"motivo_{j + 1}": f"{el[orden[i, j]]} (−{c2(brecha[i, orden[i, j]], 1)} pts): "
                                                   f"{texto_motivo(el[orden[i, j]])}" for j in range(min(k, n))}})
        peor20 = s_ttd <= np.quantile(s_ttd, 0.20)
        freq_m1 = pd.Series([el[j] for j in orden[peor20, 0]]).value_counts(normalize=True)
        eventos = [ev("scorecard", "escalado", pdo=S["pdo"], score_base=S["score_base"],
                      odds_base=S["odds_base"], factor=factor, offset=offset,
                      bins_con_puntos=len(puntos), max_dif_suma_puntos=dif_suma)]
        return {"factor": factor, "offset": offset, "puntos": puntos, "rango": rango, "LP": LP,
                "PD_RAW": PD_RAW, "SCORE_RAW": SCORE_RAW, "dif_suma_puntos": dif_suma,
                "reason_codes": pd.DataFrame(ejemplos), "freq_motivo1_peor20": freq_m1,
                "mapas": mapas, "eventos": eventos}

    return (etapa_scorecard,)


@app.cell
def _(ev, logit_np_, np, pd, sig):
    def score_desde_pd(p, factor, offset):
        """Score PDO desde una PD, con clip bilateral (evita ±inf en los bordes)."""
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
        return offset + factor * np.log((1 - p) / p)


    def etapa_calibracion(cfg, datos, sc, motor):
        K = cfg["calibracion"]
        marco = pd.concat([pd.DataFrame({"muestra": m, "cosecha": datos["muestras"][m]["fecha_solicitud"].to_numpy(),
                                         "malo": datos["muestras"][m]["malo"].to_numpy(),
                                         "lp": sc["LP"][m]}) for m in ("DEV", "HO", "OOT")],
                          ignore_index=True)
        marco["pd_cruda"] = sig(marco["lp"])
        cos = marco.groupby("cosecha").agg(n=("malo", "size"), tasa=("malo", "mean"),
                                           pd_cruda=("pd_cruda", "mean"))
        res = {}
        for ancla, (a, b) in (("ttc", K["ttc_cosechas"]), ("pit", K["pit_cosechas"])):
            en = marco["cosecha"].between(a, b)
            if ancla == "ttc":
                tc = float(cos.loc[a:b, "tasa"].mean())           # promedio SIMPLE de cosechas
            else:
                tc = float(marco.loc[en, "malo"].mean())         # tasa agregada del periodo
            lp = marco.loc[en, "lp"].to_numpy()
            d = motor["delta"](lp, tc)
            d_aprox = float(logit_np_(tc) - logit_np_(sig(lp).mean()))
            res[ancla] = {"ventana": [a, b], "tc": tc, "delta": d, "delta_aprox": d_aprox,
                          "n_casos": int(en.sum()), "pd_cruda_media": float(sig(lp).mean()),
                          "muestras": sorted(marco.loc[en, "muestra"].unique().tolist()),
                          "n_cosechas": int(cos.loc[a:b].shape[0])}
        ancla = K["ancla"]
        delta = res[ancla]["delta"]
        circular = K["muestra_validacion"] in res[ancla]["muestras"]
        PD_CAL = {m: sig(logit_np_(sc["PD_RAW"][m]) + delta) for m in sc["PD_RAW"]}
        SCORE_CAL = {m: score_desde_pd(PD_CAL[m], sc["factor"], sc["offset"]) for m in PD_CAL}
        resumen = pd.DataFrame({
            "tasa_observada": {m: (float(datos["muestras"][m]["malo"].mean()) if m != "TTD" else np.nan) for m in PD_CAL},
            "pd_cruda_media": {m: float(sc["PD_RAW"][m].mean()) for m in PD_CAL},
            "pd_calibrada_media": {m: float(PD_CAL[m].mean()) for m in PD_CAL}})
        marco["pd_cal"] = sig(marco["lp"] + delta)
        cos["pd_calibrada"] = marco.groupby("cosecha")["pd_cal"].mean()
        eventos = [ev("calibracion", "delta_estimado", ancla=ancla, tc=res[ancla]["tc"], delta=delta,
                      ventana=res[ancla]["ventana"], muestras_calibra=res[ancla]["muestras"],
                      muestra_valida=K["muestra_validacion"], circular=circular,
                      alternativa={k: {"tc": v["tc"], "delta": v["delta"]} for k, v in res.items() if k != ancla})]
        return {"anclas": res, "ancla": ancla, "delta": delta, "tc": res[ancla]["tc"], "circular": circular,
                "PD_CAL": PD_CAL, "SCORE_CAL": SCORE_CAL, "resumen": resumen, "cosechas": cos,
                "eventos": eventos}

    return etapa_calibracion, score_desde_pd


@app.cell
def _(ev, np, operator, pd):
    OPERADORES = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}


    def banda_de(score, cortes, etiquetas):
        """right=False: el corte cae en la banda superior (convención de la clase 4)."""
        return pd.cut(np.asarray(score, dtype=float), [-np.inf] + list(cortes) + [np.inf],
                      labels=etiquetas, right=False)


    def tabla_impacto_cutoff(df, score, pd_cal, cortes, lgd, col_ead):
        """Tabla de estrategia del Lab 3 (EL = PD × LGD × EAD)."""
        monto = df[col_ead].to_numpy(dtype=float)
        y = df["malo"].to_numpy(dtype=float)
        score, pd_cal = np.asarray(score), np.asarray(pd_cal)
        filas = []
        for corte in cortes:
            ap = score >= corte
            n_ap = int(ap.sum())
            filas.append({"cutoff": int(corte), "aprobacion": float(ap.mean()),
                          "mora_observada": float(y[ap].mean()) if n_ap else np.nan,
                          "pd_media": float(pd_cal[ap].mean()) if n_ap else np.nan,
                          "monto_aprobado_mm": float(monto[ap].sum() / 1e6),
                          "perdida_esperada_mm": float((pd_cal[ap] * lgd * monto[ap]).sum() / 1e6)})
        t = pd.DataFrame(filas).set_index("cutoff")
        t["el_sobre_monto"] = t["perdida_esperada_mm"] / t["monto_aprobado_mm"]
        return t


    def etapa_estrategia(cfg, datos, sc, cal, motor):
        T = cfg["estrategia"]
        cortes, etiq = T["cortes_banda"], T["etiquetas_banda"]
        SC, PDC = cal["SCORE_CAL"], cal["PD_CAL"]
        mod = ("DEV", "HO", "OOT")
        s_mod = np.concatenate([SC[m] for m in mod])
        p_mod = np.concatenate([PDC[m] for m in mod])
        y_mod = np.concatenate([datos["muestras"][m]["malo"].to_numpy() for m in mod])
        ms = (pd.DataFrame({"banda": banda_de(s_mod, cortes, etiq), "pd": p_mod, "malo": y_mod})
              .groupby("banda", observed=False)
              .agg(n=("malo", "size"), pd_calibrada=("pd", "mean"), tasa_observada=("malo", "mean")))
        ms["pct_modelacion"] = ms["n"] / ms["n"].sum()
        ms["pct_ttd"] = pd.Series(banda_de(SC["TTD"], cortes, etiq)).value_counts(normalize=True).reindex(etiq).fillna(0).to_numpy()
        lim_inf = [-np.inf] + list(cortes)
        ms["score_desde"] = lim_inf
        ms["odds_piso"] = [np.nan if not np.isfinite(l) else
                           cfg["scorecard"]["odds_base"] * 2 ** ((l - cfg["scorecard"]["score_base"]) / cfg["scorecard"]["pdo"])
                           for l in lim_inf]
        obs = ms["tasa_observada"].dropna().to_numpy()
        monotona = bool(np.all(np.diff(obs) <= 1e-12))          # de E (peor) a A1 (mejor): no crece
        # --- tabla de estrategia y recomendación según apetito ---
        m_e = T["muestra"]
        df_e = datos["muestras"][m_e]
        imp = tabla_impacto_cutoff(df_e, SC[m_e], PDC[m_e], T["grilla_cutoff"], T["lgd"], T["ead"])
        imp["cumple_mora"] = imp["mora_observada"] <= T["mora_max"]
        imp["cumple_aprobacion"] = imp["aprobacion"] >= T["aprobacion_min"]
        factibles = imp.index[imp["cumple_mora"] & imp["cumple_aprobacion"]]
        if len(factibles):
            cutoff, estado = int(min(factibles)), "factible: el corte más generoso que respeta el apetito"
        elif imp["cumple_mora"].any():
            cutoff = int(imp.index[imp["cumple_mora"]].min())
            estado = "CONFLICTO de apetito: ningún corte cumple ambos; se prioriza la mora (aprobación bajo el mínimo)"
        else:
            cutoff, estado = int(max(imp.index)), "CONFLICTO: ningún corte cumple la mora máxima; se toma el más estricto"
        # --- swap-set a igual aprobación contra knock-outs (receta de clase 4/5) ---
        rechaza_ko = np.zeros(len(df_e), dtype=bool)
        for var, op, val in T["knockouts"]:
            rechaza_ko |= OPERADORES[op](df_e[var].to_numpy(dtype=float), val)
        aprueba_ko = ~rechaza_ko
        k = int(aprueba_ko.sum())
        orden = np.argsort(-SC[m_e], kind="stable")
        aprueba_sc = np.zeros(len(df_e), dtype=bool)
        aprueba_sc[orden[:k]] = True
        y_e = df_e["malo"].to_numpy(dtype=float)

        def celda(mask):
            return {"n": int(mask.sum()), "tasa_malos": float(y_e[mask].mean()) if mask.any() else np.nan}

        swap = pd.DataFrame({
            "scorecard_aprueba": [celda(aprueba_ko & aprueba_sc), celda(rechaza_ko & aprueba_sc)],
            "scorecard_rechaza": [celda(aprueba_ko & ~aprueba_sc), celda(rechaza_ko & ~aprueba_sc)],
        }, index=["knockouts_aprueban", "knockouts_rechazan"])
        swap_in = rechaza_ko & aprueba_sc
        swap_out = aprueba_ko & ~aprueba_sc
        n_si, malos_si = int(swap_in.sum()), int(y_e[swap_in].sum())
        pd_si = float(PDC[m_e][swap_in].mean()) if n_si else np.nan
        p_si = motor["binom_p"](malos_si, n_si, pd_si) if n_si else np.nan
        res_swap = {"aprobacion": float(aprueba_ko.mean()), "cutoff_equivalente": float(SC[m_e][orden[k - 1]]) if k else np.nan,
                    "mora_ko": float(y_e[aprueba_ko].mean()), "mora_sc": float(y_e[aprueba_sc].mean()),
                    "n_swap_in": n_si, "tasa_swap_in": float(malos_si / n_si) if n_si else np.nan,
                    "pd_swap_in": pd_si, "p_swap_in": p_si, "n_swap_out": int(swap_out.sum()),
                    "tasa_swap_out": float(y_e[swap_out].mean()) if swap_out.any() else np.nan}
        # decisión con el ancla alternativa: el δ solo traslada el score
        otra = "pit" if cal["ancla"] == "ttc" else "ttc"
        despl = -sc["factor"] * (cal["anclas"][otra]["delta"] - cal["delta"])
        aprob_otra = float((SC[m_e] + despl >= cutoff).mean())
        eventos = [ev("estrategia", "master_scale", monotona=monotona,
                      pct_ttd_DE=float(ms.loc[etiq[:2], "pct_ttd"].sum())),
                   ev("estrategia", "cutoff_recomendado", cutoff=cutoff, estado=estado,
                      mora_max=T["mora_max"], aprobacion_min=T["aprobacion_min"], muestra=m_e,
                      aprobacion=float(imp.loc[cutoff, "aprobacion"]),
                      mora=float(imp.loc[cutoff, "mora_observada"])),
                   ev("estrategia", "swap_set", **{k_: v for k_, v in res_swap.items()})]
        return {"master_scale": ms, "monotona": monotona, "impacto": imp, "cutoff": cutoff,
                "estado_cutoff": estado, "swap": swap, "res_swap": res_swap,
                "decision_otra_ancla": {"ancla": otra, "desplazamiento_score": despl, "aprobacion": aprob_otra},
                "eventos": eventos}

    return banda_de, etapa_estrategia


@app.cell
def _(
    banda_de,
    c2,
    csi_curso,
    etapa_calibracion,
    etapa_embudo,
    etapa_estrategia,
    etapa_scorecard,
    ev,
    fmt_p,
    np,
    pct,
    pd,
    semaforo_mayor,
    semaforo_menor,
    time,
):
    def bootstrap_gini(pd_pred, malo, B, semilla, auc):
        """Mecánica del Lab 3: réplicas con reemplazo; si falta una clase, se repite."""
        y = np.asarray(malo, dtype=float)
        p = np.asarray(pd_pred, dtype=float)
        rng = np.random.default_rng(semilla)
        g = np.empty(B)
        for b in range(B):
            while True:
                idx = rng.integers(0, len(y), len(y))
                yb = y[idx]
                if 0 < yb.sum() < len(yb):
                    break
            g[b] = 2 * auc(yb, p[idx]) - 1
        return g


    def hosmer_lemeshow(pd_pred, malo, n_grupos, S, semilla, chi2_sf):
        """HL por deciles de PD: estadístico, p de la χ² y p SIMULADO bajo H0 (clase 5)."""
        d = pd.DataFrame({"pd": np.asarray(pd_pred, dtype=float), "malo": np.asarray(malo, dtype=float)})
        d["g"] = pd.qcut(d["pd"], n_grupos, labels=False, duplicates="drop")
        t = d.groupby("g").agg(n=("malo", "size"), obs=("malo", "sum"), pd_media=("pd", "mean"))
        t["esp"] = t["n"] * t["pd_media"]
        var_g = (t["esp"] * (1 - t["pd_media"])).to_numpy()
        t["aporte"] = (t["obs"] - t["esp"]) ** 2 / var_g
        hl = float(t["aporte"].sum())
        p_chi2 = chi2_sf(hl, len(t) - 2)
        rng = np.random.default_rng(semilla)
        hl_sim = np.zeros(S)
        for j, (g_, fila) in enumerate(t.iterrows()):
            pds = d.loc[d["g"] == g_, "pd"].to_numpy()
            obs_sim = (rng.random((S, len(pds))) < pds).sum(axis=1)
            hl_sim += (obs_sim - t["esp"].iloc[j]) ** 2 / var_g[j]
        return {"hl": hl, "p_chi2": p_chi2, "p_sim": float((hl_sim >= hl).mean()), "tabla": t}


    def diagnosticar(tablero, cal, val):
        """Lectura por PATRÓN (clase 5): familias ranking / población / calibración."""
        peso = {"🟢": 0, "⏳": 0, "⚪": 0, "🟡": 1, "🔴": 2}
        fam = {f: max([peso[e] for e in tablero.loc[tablero["familia"] == f, "estado"]], default=0)
               for f in ("ranking", "poblacion", "calibracion")}
        glob = tablero.loc["binomial", "estado"]
        subestima = val["residuo_global"] > 0
        lectura, accion, no_hacer, firma = [], [], [], []
        if fam["ranking"] == 2:
            lectura.append("pérdida de discriminación: el ranking se deterioró más allá del umbral rojo")
            accion.append("abrir re-desarrollo (gatillo: caída relativa de Gini > 30% o KS < 0,20)")
            no_hacer.append("recalibrar solo el δ: mueve el nivel, no repara el orden")
            firma.append("Comité de Riesgo")
        else:
            lectura.append("el ranking se sostiene" + (" con una advertencia (🟡)" if fam["ranking"] == 1 else ""))
        if fam["calibracion"] == 2 and glob == "🔴":
            lectura.append(("descalibración de NIVEL: el modelo " + ("subestima" if subestima else "sobreestima")
                            + f" la mora en {val['muestra_validacion']} (residuo {c2(100 * val['residuo_global'], 1)} pts)"))
            accion.append("recalibrar el δ con acta, re-anclando a la tendencia central actualizada, "
                          "y revisar el cutoff con la PD nueva")
            if fam["ranking"] < 2:
                no_hacer.append("re-desarrollar: el problema es de nivel y el ranking no lo justifica")
            firma.append("Jefe de Modelos (δ) + Comité de Riesgo (política)")
        elif fam["calibracion"] == 2:
            lectura.append("descalibración de FORMA (bandas o HL en rojo con nivel global aceptable)")
            accion.append("revisar la master scale por tramo; el δ solo no corrige la forma")
            firma.append("Jefe de Modelos")
        elif fam["calibracion"] == 1:
            lectura.append("señales amarillas de calibración")
            accion.append("vigilancia reforzada 2 meses y recalibración programada si persiste")
            firma.append("Modelador")
        if fam["poblacion"] >= 1:
            lectura.append("la población de la bandeja se movió (" +
                           ", ".join(tablero.index[(tablero["familia"] == "poblacion") & tablero["estado"].isin(["🟡", "🔴"])]) + ")")
            accion.append("rastrear el origen con el CSI por variable antes de decidir; documentar el mix")
        if cal["circular"]:
            lectura.append("OJO: la muestra de validación entra en la ventana de calibración (backtesting circular: "
                           "pierde poder; si la ventana ES la muestra, el binomial global da p≈1 por construcción)")
        if not accion:
            accion.append("operación normal: monitoreo con la frecuencia declarada")
        n_tests = int(tablero["estado"].isin(["🟢", "🟡", "🔴"]).sum())
        return {"familias": fam, "patron": "; ".join(lectura), "accion": "; ".join(accion),
                "no_hacer": "; ".join(no_hacer) or "escalar por un color suelto",
                "firma": "; ".join(dict.fromkeys(firma)) or "Modelador",
                "prob_amarillo_azar": 1 - 0.95 ** n_tests, "n_indicadores": n_tests}


    def etapa_validacion(cfg, datos, emb, sc, cal, est, motor):
        V, T = cfg["validacion"], cfg["estrategia"]
        U = V["umbrales"]
        mu = datos["muestras"]
        PDC, SC = cal["PD_CAL"], cal["SCORE_CAL"]
        mv = cfg["calibracion"]["muestra_validacion"]
        met = {}
        for m in ("DEV", "HO", "OOT"):
            y = mu[m]["malo"].to_numpy()
            auc = motor["auc"](y, PDC[m])
            met[m] = {"auc": auc, "gini": 2 * auc - 1, "ks": motor["ks"](y, PDC[m]),
                      "gini_cruda": 2 * motor["auc"](y, sc["PD_RAW"][m]) - 1}
        met = pd.DataFrame(met).T
        met["caida_gini_rel"] = 1 - met["gini"] / met.loc["DEV", "gini"]
        boot = {m: bootstrap_gini(PDC[m], mu[m]["malo"].to_numpy(), V["B_bootstrap"],
                                  V["semillas_bootstrap"][m], motor["auc"]) for m in ("HO", "OOT")}
        ic = {m: tuple(np.percentile(boot[m], [2.5, 97.5])) for m in boot}
        ic["caida_HO_OOT"] = tuple(np.percentile(boot["HO"] - boot["OOT"], [2.5, 97.5]))
        # deciles en la muestra de validación
        dd = pd.DataFrame({"malo": mu[mv]["malo"].to_numpy(), "score": SC[mv]})
        dd["decil"] = pd.qcut(dd["score"], 10, labels=False, duplicates="drop") + 1
        dec = dd.groupby("decil").agg(n=("malo", "size"), malos=("malo", "sum"), tasa_malos=("malo", "mean"))
        dec["captura_malos"] = dec["malos"].cumsum() / dd["malo"].sum()
        dec["pob_acum"] = dec["n"].cumsum() / len(dd)
        dec["lift_acum"] = dec["captura_malos"] / dec["pob_acum"]
        # PSI del score por bandas y CSI
        cortes, etiq = T["cortes_banda"], T["etiquetas_banda"]

        def psi_bandas(se, sa, eps=1e-4):
            e = pd.Series(banda_de(se, cortes, etiq)).value_counts(normalize=True).reindex(etiq).fillna(0)
            a = pd.Series(banda_de(sa, cortes, etiq)).value_counts(normalize=True).reindex(etiq).fillna(0)
            t = pd.DataFrame({"pct_dev": e, "pct_actual": a,
                              "aporte": (a + eps - e - eps) * np.log((a + eps) / (e + eps))})
            return motor["psi"](e.to_numpy() + eps, a.to_numpy() + eps), t

        psi_oot, t_psi_oot = psi_bandas(SC["DEV"], SC["OOT"])
        psi_ttd, t_psi_ttd = psi_bandas(SC["DEV"], SC["TTD"])
        csi = pd.Series({v: csi_curso(mu["DEV"][v], mu["TTD"][v], cfg["embudo"]["bins_woe"], 1e-4, motor["psi"])
                         for v in emb["elegidas"]}).sort_values(ascending=False)
        # backtesting binomial global y por banda
        y_v = mu[mv]["malo"].to_numpy()
        n_v, malos_v = len(y_v), int(y_v.sum())
        pd_media = float(PDC[mv].mean())
        p_global = motor["binom_p"](malos_v, n_v, pd_media)
        bt = (pd.DataFrame({"banda": banda_de(SC[mv], cortes, etiq), "pd": PDC[mv], "malo": y_v})
              .groupby("banda", observed=False)
              .agg(n=("malo", "size"), malos=("malo", "sum"), pd_calibrada=("pd", "mean"), tasa_observada=("malo", "mean")))
        bt["malos"] = bt["malos"].astype(int)
        bt["esperados"] = bt["n"] * bt["pd_calibrada"]
        bt["p_valor"] = [motor["binom_p"](f["malos"], f["n"], f["pd_calibrada"]) if f["n"] > 0 else np.nan
                         for _, f in bt.iterrows()]
        a_, r_ = U["p_valor"]
        bt["semaforo"] = [("—" if not np.isfinite(p) else ("🔴" if p < r_ else "🟡" if p < a_ else "🟢")) for p in bt["p_valor"]]
        hl = hosmer_lemeshow(PDC[mv], y_v, V["hl_grupos"], V["hl_simulaciones"], V["hl_semilla"], motor["chi2_sf"])
        mix_mod = float(pd.Series(banda_de(np.concatenate([SC[m] for m in ("DEV", "HO", "OOT")]), cortes, etiq))
                        .value_counts(normalize=True)[etiq[:2]].sum())
        mix_ttd = float(pd.Series(banda_de(SC["TTD"], cortes, etiq)).value_counts(normalize=True)[etiq[:2]].sum())
        n_am = int(((bt["p_valor"] < a_) & (bt["p_valor"] >= r_)).sum())
        n_ro = int((bt["p_valor"] < r_).sum())
        rs = est["res_swap"]
        caida = float(met.loc[mv, "caida_gini_rel"])
        filas = [
            ("gini", "ranking", f"Gini {mv} (caída vs DEV)", f"{c2(met.loc[mv, 'gini'], 3)} (−{pct(caida)})",
             "caída > 20% / > 30%", semaforo_mayor(caida, *U["caida_gini_rel"]), "trimestral"),
            ("ks", "ranking", f"KS {mv}", c2(met.loc[mv, "ks"], 3), "< 0,30 / < 0,20",
             semaforo_menor(met.loc[mv, "ks"], *U["ks"]), "trimestral"),
            ("psi", "poblacion", "PSI score DEV→TTD (8 bandas)", c2(psi_ttd, 3), "> 0,10 / > 0,25",
             semaforo_mayor(psi_ttd, *U["psi"]), "mensual"),
            ("csi", "poblacion", f"CSI máximo ({csi.idxmax()})", c2(csi.max(), 3), "> 0,10 / > 0,25",
             semaforo_mayor(float(csi.max()), *U["psi"]), "mensual"),
            ("mix", "poblacion", f"Mix {'+'.join(etiq[:2])} en bandeja TTD",
             f"{pct(mix_ttd)} ({'+' if mix_ttd >= mix_mod else '−'}{c2(abs(mix_ttd - mix_mod) * 100, 1)} pts)", "> +3 / > +6 pts",
             semaforo_mayor((mix_ttd - mix_mod) * 100, *U["mix_de_pts"]), "mensual"),
            ("binomial", "calibracion", f"Binomial global {mv}", f"p = {fmt_p(p_global)}", "p < 0,05 / < 0,01",
             semaforo_menor(p_global, *U["p_valor"]), "trimestral"),
            ("bandas", "calibracion", f"Bandas binomial fuera ({mv})", f"{n_am} 🟡 · {n_ro} 🔴", "≥ 1 🟡 / ≥ 1 🔴",
             "🔴" if n_ro else ("🟡" if n_am else "🟢"), "trimestral"),
            ("hl", "calibracion", f"Hosmer-Lemeshow {mv} (p simulado)", f"χ² {c2(hl['hl'], 1)} (p sim. {fmt_p(hl['p_sim'], V['hl_simulaciones'])})",
             "p < 0,05 / < 0,01", semaforo_menor(hl["p_sim"], *U["p_valor"]), "trimestral"),
            ("swap_in", "calibracion", "Cohorte swap-in vs su PD",
             (f"{pct(rs['tasa_swap_in'])} vs {pct(rs['pd_swap_in'])} (p {fmt_p(rs['p_swap_in'])})" if rs["n_swap_in"] else "sin cohorte"),
             "p < 0,05 / < 0,01", semaforo_menor(rs["p_swap_in"], *U["p_valor"]) if rs["n_swap_in"] else "⚪", "trimestral"),
            ("tc_futura", "calibracion", "TC realizada vs ancla (post-recalibración)",
             "por medir: cosechas TTD maduran 12 meses después", "desvío > 1 / > 2 pts", "⏳", "trimestral"),
        ]
        tablero = pd.DataFrame(filas, columns=["id", "familia", "indicador", "valor_hoy", "umbral", "estado",
                                               "frecuencia"]).set_index("id")
        val = {"metricas": met, "boot": boot, "ic": ic, "deciles": dec, "psi_oot": psi_oot, "psi_ttd": psi_ttd,
               "t_psi_ttd": t_psi_ttd, "t_psi_oot": t_psi_oot, "csi": csi, "p_global": p_global,
               "malos_v": malos_v, "esperados_v": n_v * pd_media, "residuo_global": float(y_v.mean() - pd_media),
               "backtest": bt, "hl": hl, "mix": (mix_mod, mix_ttd), "tablero": tablero, "muestra_validacion": mv}
        val["diagnostico"] = diagnosticar(tablero, cal, val)
        eventos = [ev("validacion", "discriminacion", **{m: {"gini": met.loc[m, "gini"], "ks": met.loc[m, "ks"]} for m in met.index},
                      ic95={k: list(v) for k, v in ic.items()}, B=V["B_bootstrap"]),
                   ev("validacion", "estabilidad", psi_oot=psi_oot, psi_ttd=psi_ttd, csi_max=float(csi.max())),
                   ev("validacion", "backtesting", muestra=mv, p_global=p_global, bandas_amarillas=n_am,
                      bandas_rojas=n_ro, hl=hl["hl"], hl_p_sim=hl["p_sim"], hl_p_chi2=hl["p_chi2"]),
                   ev("validacion", "tablero", estados=tablero["estado"].to_dict(),
                      patron=val["diagnostico"]["patron"], accion=val["diagnostico"]["accion"])]
        val["eventos"] = eventos
        return val


    def correr_pipeline(cfg, datos, motor, ins, psi_tab):
        """Etapas 2–6 en orden. Devuelve artefactos por etapa, eventos y tiempos."""
        tiempos = {}
        t = time.perf_counter()
        emb = etapa_embudo(cfg, datos, motor, ins, psi_tab)
        tiempos["embudo"] = time.perf_counter() - t
        t = time.perf_counter()
        sc = etapa_scorecard(cfg, datos, emb, ins, motor)
        cal = etapa_calibracion(cfg, datos, sc, motor)
        est = etapa_estrategia(cfg, datos, sc, cal, motor)
        tiempos["scorecard_calibracion_estrategia"] = time.perf_counter() - t
        t = time.perf_counter()
        val = etapa_validacion(cfg, datos, emb, sc, cal, est, motor)
        tiempos["validacion"] = time.perf_counter() - t
        return {"motor": motor["nombre"], "embudo": emb, "scorecard": sc, "calibracion": cal,
                "estrategia": est, "validacion": val, "tiempos": tiempos,
                "eventos": emb["eventos"] + sc["eventos"] + cal["eventos"] + est["eventos"] + val["eventos"]}

    return (correr_pipeline,)


@app.cell
def _(
    CONFIG,
    MOTORES,
    c2,
    correr_pipeline,
    datos,
    insumos_woe,
    mo,
    tabla_psi,
    time,
):
    _t = time.perf_counter()
    insumos_actual = insumos_woe(datos, CONFIG["embudo"]["bins_woe"])
    t_insumos = time.perf_counter() - _t
    psi_por_motor = {m: tabla_psi(CONFIG, datos, MOTORES[m]) for m in MOTORES}
    corridas = {m: correr_pipeline(CONFIG, datos, MOTORES[m], insumos_actual, psi_por_motor[m]) for m in MOTORES}
    corrida = corridas[CONFIG["motor"]]
    emb = corrida["embudo"]
    sc = corrida["scorecard"]
    cal = corrida["calibracion"]
    est = corrida["estrategia"]
    val = corrida["validacion"]
    t_pipeline = time.perf_counter() - _t
    mo.md(f"""
    Pipeline ejecutado con **ambos motores** en {c2(t_pipeline, 1)} s (insumos WoE {c2(t_insumos, 1)} s).
    Se muestra el motor `{CONFIG['motor']}`; el arnés de paridad (sección 9) compara los dos.
    """)
    return (
        cal,
        corrida,
        corridas,
        emb,
        est,
        insumos_actual,
        psi_por_motor,
        sc,
        val,
    )


@app.cell
def _(CONFIG, c2, datos, emb, mo, pd):
    _conteo = pd.DataFrame(emb["conteo"], columns=["paso", "variables"]).set_index("paso")
    _top_psi = (emb["tabla"].sort_values("psi_ttd", ascending=False, na_position="last")
                .head(12)[["psi_oot", "psi_ttd", "iv", "estado"]])
    _cat_fuera = [v for v in emb["fuera_psi"] if not pd.api.types.is_numeric_dtype(datos["X"][v])]
    mo.vstack([
        mo.md(f"""
    ### 2. Embudo · {' → '.join(str(n) for _, n in emb['conteo'])}

    - **PSI vs TTD**: {len(emb['fuera_psi'])} fuera por PSI > {c2(CONFIG['embudo']['psi_max_ttd'])} y
      {len(emb['no_medibles'])} **no medibles** (deciles de DEV colapsados por masa de ceros; regla del curso:
      sin certificado de estabilidad no siguen). Categóricas inestables: **{', '.join(_cat_fuera) or 'ninguna'}**
      — la variable plantada del Lab 2.
    - **IV ≥ {c2(CONFIG['embudo']['iv_min'])}**: {len(emb['candidatas_iv'])} candidatas; pares con |ρ WoE| > 0,90: {emb['pares_090']}.
    - **Correlación greedy ≤ {c2(CONFIG['embudo']['corr_max'])}**: quedan {len(emb['seleccion'])}.
      **VIF**: máximo {c2(emb['vif'].max())} ({'nada sale' if not emb['vif_fuera'] else 'salen ' + ', '.join(emb['vif_fuera'])}).
    - **Stepwise** (α = {c2(CONFIG['embudo']['alpha_entrada'])}, tope {CONFIG['embudo']['max_variables']}) en
      {emb['corridas_stepwise']} corrida(s); excluidas por signo: {', '.join(emb['excluidas_signo']) or 'ninguna'}.
      **{len(emb['elegidas'])} variables finales.**
    """),
        mo.hstack([_conteo, _top_psi], justify="start"),
        mo.md("**No medibles (reportadas con su IV: un descarte silencioso es indefendible):**"),
        emb["tabla"].loc[emb["no_medibles"], ["iv", "psi_oot"]].sort_values("iv", ascending=False).round(4),
    ])
    return


@app.cell
def _(emb, mo):
    mo.vstack([
        mo.md("**Modelo final** (β en DEV; re-estimación en HO solo para *contrastar* estabilidad de signos):"),
        emb["coef"].round(4),
        mo.md(f"""
    Todos los β de las variables son {'**negativos** ✓' if (emb['coef'].drop('const')['beta'] < 0).all() else '**NO todos negativos** ✗'}
    (WoE alto = bin bueno). Signos invertidos al re-estimar en HO:
    {', '.join(emb['coef'].index[emb['coef']['signo_invertido_HO']]) or 'ninguno'}.
    Bitácora del stepwise: {' · '.join(f"{b['accion']} {b.get('variable', b.get('motivo', ''))}" for b in emb['bitacora'])}.
    """),
    ])
    return


@app.cell
def _(CONFIG, c2, mo, sc):
    mo.vstack([
        mo.md(f"""
    ### 3. Scorecard · PDO {CONFIG['scorecard']['pdo']} · {CONFIG['scorecard']['score_base']} a odds {CONFIG['scorecard']['odds_base']}:1

    factor = 20/ln 2 = **{c2(sc['factor'], 4)}** · offset = 600 − factor·ln 50 = **{c2(sc['offset'], 4)}** ·
    {len(sc['puntos'])} filas (variable × bin). Invariante: la suma de puntos reproduce el score crudo con
    error máximo {sc['dif_suma_puntos']:.1e}. Rango de puntos por variable (no ordena igual que |β|: depende
    también de la dispersión del WoE):
    """),
        sc["rango"].round(1),
        mo.md("**Reason codes** (brecha = máximo de la variable − puntos obtenidos; las 3 mayores) para tres "
              "solicitudes TTD en los percentiles 1, 50 y 99 del score:"),
        sc["reason_codes"],
    ])
    return


@app.cell
def _(CONFIG, c2, cal, est, matplotlib, mo, pct, plt):
    _A = cal["anclas"]
    _fig, _ax = plt.subplots(figsize=(8.5, 3.6))
    _cos = cal["cosechas"]
    _ax.plot(_cos.index, _cos["tasa"], "o-", color="#C0392B", label="tasa observada (malos 90+)")
    _ax.plot(_cos.index, _cos["pd_cruda"], "s--", color="#95A5A6", label="PD cruda media")
    _ax.plot(_cos.index, _cos["pd_calibrada"], "^-", color="#2E6FF2", label=f"PD calibrada media ({cal['ancla'].upper()})")
    _ax.axhline(_A["ttc"]["tc"], color="#27AE60", lw=1, ls=":", label=f"TC TTC {_A['ttc']['tc']:.2%}")
    _ax.axhline(_A["pit"]["tc"], color="#8E44AD", lw=1, ls=":", label=f"tasa PIT {_A['pit']['tc']:.2%}")
    _ax.set_title("Cosechas DEV+HO+OOT: tasa real vs PD (Financiera Andes)")
    _ax.set_xlabel("cosecha (mes de solicitud)")
    _ax.set_ylabel("tasa / PD")
    _ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _ax.tick_params(axis="x", rotation=45)
    _ax.legend(fontsize=7, ncol=2)
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
    ### 4. Calibración: TTC y PIT (ambas calculadas; ancla elegida: **{cal['ancla'].upper()}**)

    | Ancla | Ventana (cosechas) | Muestras que calibran | Objetivo | PD cruda media | δ exacto | δ aprox. logit |
    |---|---|---|---|---|---|---|
    | TTC | {' … '.join(_A['ttc']['ventana'])} ({_A['ttc']['n_cosechas']}) | {'+'.join(_A['ttc']['muestras'])} | {pct(_A['ttc']['tc'], 2)} (promedio simple de cosechas) | {pct(_A['ttc']['pd_cruda_media'], 2)} | {c2(_A['ttc']['delta'], 4)} | {c2(_A['ttc']['delta_aprox'], 4)} |
    | PIT | {' … '.join(_A['pit']['ventana'])} ({_A['pit']['n_cosechas']}) | {'+'.join(_A['pit']['muestras'])} | {pct(_A['pit']['tc'], 2)} (tasa agregada) | {pct(_A['pit']['pd_cruda_media'], 2)} | {c2(_A['pit']['delta'], 4)} | {c2(_A['pit']['delta_aprox'], 4)} |

    **Calibra** {'+'.join(_A[cal['ancla']]['muestras'])}; **valida** {CONFIG['calibracion']['muestra_validacion']}
    → {'⚠️ CIRCULAR: la muestra que valida entra en la ventana de calibración; el backtesting pierde poder (si la ventana ES la muestra, el binomial global da p≈1 por construcción).' if cal['circular'] else 'independientes ✓ (la regla del validador: calibrar y validar con muestras distintas, ambas declaradas).'}
    Cambiar de ancla traslada **todos** los scores en {c2(est['decision_otra_ancla']['desplazamiento_score'], 1)} puntos
    (= −factor·Δδ) y no toca el ranking: con el cutoff recomendado, la aprobación en {CONFIG['estrategia']['muestra']}
    pasaría de {pct(est['impacto'].loc[est['cutoff'], 'aprobacion'])} a {pct(est['decision_otra_ancla']['aprobacion'])}.
    """),
        mo.hstack([cal["resumen"].round(4), _fig], justify="start"),
    ])
    return


@app.cell
def _(CONFIG, c2, est, fmt_p, matplotlib, mo, pct, plt):
    _ms = est["master_scale"].iloc[::-1]
    _imp = est["impacto"]
    _fig2, _ax2 = plt.subplots(figsize=(7.5, 3.4))
    _ax2.plot(_imp["aprobacion"], _imp["mora_observada"], "o-", color="#2E6FF2")
    for _c in _imp.index[::2]:
        _ax2.annotate(str(_c), (_imp.loc[_c, "aprobacion"], _imp.loc[_c, "mora_observada"]), fontsize=7,
                      xytext=(3, 3), textcoords="offset points")
    _ax2.axhline(CONFIG["estrategia"]["mora_max"], color="#C0392B", ls="--", lw=1, label="mora máxima (apetito)")
    _ax2.axvline(CONFIG["estrategia"]["aprobacion_min"], color="#8E44AD", ls="--", lw=1, label="aprobación mínima")
    _ax2.scatter([_imp.loc[est["cutoff"], "aprobacion"]], [_imp.loc[est["cutoff"], "mora_observada"]], s=120,
                 facecolors="none", edgecolors="k", label=f"cutoff recomendado {est['cutoff']}")
    _ax2.set_title(f"Frontera aprobación–mora ({CONFIG['estrategia']['muestra']})")
    _ax2.set_xlabel("tasa de aprobación")
    _ax2.set_ylabel("mora observada de aprobados")
    _ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _ax2.legend(fontsize=7)
    _fig2.tight_layout()
    _rs = est["res_swap"]
    mo.vstack([
        mo.md(f"""
    ### 5. Master scale, estrategia y swap-set

    Master scale de 8 bandas (cortes {CONFIG['estrategia']['cortes_banda']}; con PDO 20 cada banda duplica las odds).
    Tasa observada monótona de E a A1: {'sí ✓' if est['monotona'] else '**NO** ✗ (revisar tramos)'}.
    """),
        _ms.round(4),
        mo.md(f"""
    **Tabla de estrategia** ({CONFIG['estrategia']['muestra']}, LGD {pct(CONFIG['estrategia']['lgd'], 0)},
    EAD = `{CONFIG['estrategia']['ead']}`). Apetito de la config: mora ≤ {pct(CONFIG['estrategia']['mora_max'])} y
    aprobación ≥ {pct(CONFIG['estrategia']['aprobacion_min'], 0)} → **cutoff {est['cutoff']}** ({est['estado_cutoff']}):
    aprobación {pct(_imp.loc[est['cutoff'], 'aprobacion'])}, mora {pct(_imp.loc[est['cutoff'], 'mora_observada'], 2)},
    EL/monto {pct(_imp.loc[est['cutoff'], 'el_sobre_monto'], 2)}.
    """),
        mo.hstack([_imp.round(4), _fig2], justify="start"),
        mo.md(f"""
    **Swap-set a igual aprobación ({pct(_rs['aprobacion'])}) contra los knock-outs** {CONFIG['estrategia']['knockouts']}
    (score equivalente {c2(_rs['cutoff_equivalente'], 1)}): la cartera aprobada pasa de {pct(_rs['mora_ko'], 2)} de mora
    (knock-outs) a {pct(_rs['mora_sc'], 2)} (scorecard). Entran {_rs['n_swap_in']} con {pct(_rs['tasa_swap_in'])} de malos
    (PD prometida {pct(_rs['pd_swap_in'])}, binomial p = {fmt_p(_rs['p_swap_in'])}) y salen {_rs['n_swap_out']} con
    {pct(_rs['tasa_swap_out'])}. Profundiza: M18.
    """),
        est["swap"].map(lambda d: f"{d['n']:,} · {d['tasa_malos']:.1%}".replace(",", ".")),
    ])
    return


@app.cell
def _(CONFIG, c2, fmt_p, mo, val):
    _v = val
    _met = _v["metricas"]
    _dg = _v["diagnostico"]
    mo.vstack([
        mo.md(f"""
    ### 6. Validación

    **Discriminación** (sobre PD calibrada; Gini crudo idéntico: el δ no cambia el orden). Bootstrap B = {CONFIG['validacion']['B_bootstrap']}
    (semillas del Lab 3): Gini HO {c2(_met.loc['HO', 'gini'], 3)} [{c2(_v['ic']['HO'][0], 3)}; {c2(_v['ic']['HO'][1], 3)}] ·
    OOT {c2(_met.loc['OOT', 'gini'], 3)} [{c2(_v['ic']['OOT'][0], 3)}; {c2(_v['ic']['OOT'][1], 3)}] · caída HO−OOT IC95
    [{c2(_v['ic']['caida_HO_OOT'][0], 3)}; {c2(_v['ic']['caida_HO_OOT'][1], 3)}] →
    {'contiene el 0: no concluyente' if _v['ic']['caida_HO_OOT'][0] <= 0 <= _v['ic']['caida_HO_OOT'][1] else 'no contiene el 0: caída demostrable'}.
    """),
        _met.round(4),
        mo.md(f"**Deciles en {_v['muestra_validacion']}** (decil 1 = peor score):"),
        _v["deciles"].round(4),
        mo.md(f"""
    **Estabilidad**: PSI del score (8 bandas) DEV→OOT {c2(_v['psi_oot'], 4)} · DEV→TTD {c2(_v['psi_ttd'], 4)}.
    CSI DEV→TTD de las variables del modelo:
    """),
        _v["csi"].round(4).to_frame("CSI_TTD"),
        mo.md(f"""
    **Backtesting en {_v['muestra_validacion']}**: {_v['malos_v']} malos observados vs {c2(_v['esperados_v'], 1)} esperados
    (residuo {c2(100 * _v['residuo_global'], 2)} pts) → binomial global p = {fmt_p(_v['p_global'])}.
    Hosmer-Lemeshow χ² {c2(_v['hl']['hl'], 1)}: p de la tabla χ² {fmt_p(_v['hl']['p_chi2'])} vs **p simulado {fmt_p(_v['hl']['p_sim'], CONFIG['validacion']['hl_simulaciones'])}**
    (el simulado manda cuando hay celdas con esperados < 5).
    """),
        _v["backtest"].iloc[::-1].round(4),
    ])
    return


@app.cell
def _(mo, pct, val):
    _dg = val["diagnostico"]
    mo.vstack([
        mo.md("### Tablero de monitoreo (umbrales de la config, fijados antes de mirar el dato)"),
        val["tablero"],
        mo.md(f"""
    **Diagnóstico por patrón** (familias → ranking {['🟢', '🟡', '🔴'][_dg['familias']['ranking']]} ·
    población {['🟢', '🟡', '🔴'][_dg['familias']['poblacion']]} · calibración {['🟢', '🟡', '🔴'][_dg['familias']['calibracion']]}):

    - **Patrón**: {_dg['patron']}.
    - **Acción proporcionada**: {_dg['accion']}.
    - **Qué NO hacer**: {_dg['no_hacer']}.
    - **Quién firma**: {_dg['firma']}.
    - Aritmética del amarillo: con {_dg['n_indicadores']} indicadores al 5%, P(≥ 1 amarillo por azar) ≈
      {pct(_dg['prob_amarillo_azar'], 0)} si fueran independientes — se escala por conjunto coherente y persistencia.
    """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Implementación y gobierno

    «En producción NO existe DEV» (clase 6): todo lo que dependía de DEV se congela como **datos** en un
    artefacto JSON (`allow_nan=False`), y el motor `puntuar(df, artefacto)` lo usa sin mirar DEV. Se
    verifica **paridad** con el pipeline, independencia **lote-vs-fila** (anti bug A: re-ajustar el
    binner con el lote del día), el **contrato de datos** (bloquea / avisa), el **audit trail** JSONL con
    cadena de hashes y **sello** externo, el **lineage**, y se generan la **model card** y el
    **resumen ejecutivo** desde la corrida. Profundizan: M21 (artefacto, contrato, paridad) y M22
    (expediente, trail, model card).
    """)
    return


@app.cell
def _(
    banda_de,
    hash_json,
    json,
    limpiar,
    logit_np_,
    np,
    pd,
    score_desde_pd,
    sig,
):
    def es_numerica(s):
        """Numérica de verdad: el StringDtype de pandas NO lo es."""
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


    def clave_discreta(v):
        """0 y 0.0 son EL MISMO bin (Lab 3)."""
        if pd.isna(v):
            return "MISSING"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return str(int(v)) if float(v).is_integer() else repr(float(v))
        return str(v)


    def woe_normalizado(mapa, spec):
        if spec["tipo"] != "discreto":
            return {str(b): float(w) for b, w in mapa.items()}
        salida = {}
        for b, w in mapa.items():
            if str(b) == "MISSING":
                salida["MISSING"] = float(w)
                continue
            try:
                salida[clave_discreta(float(b))] = float(w)
            except (TypeError, ValueError):
                salida[str(b)] = float(w)
        return salida


    def ajustar_bins(base, bins=5, umbral_moda=0.35):
        """La MISMA partición que binear(x, ref=base), como datos (±inf → None)."""
        base = pd.Series(base).reset_index(drop=True)
        if not es_numerica(base):
            return {"tipo": "categorico"}
        if base.nunique(dropna=True) <= bins:
            return {"tipo": "discreto", "valores": sorted({clave_discreta(v) for v in base.dropna()})}
        moda = base.mode().iloc[0]
        frac = float((base == moda).mean())
        resto = base[base != moda] if frac > umbral_moda else base
        cortes = np.unique(np.nanquantile(resto.dropna(), np.linspace(0, 1, bins + 1)))
        if len(cortes) < 3:
            return {"tipo": "discreto", "valores": sorted({clave_discreta(v) for v in base.dropna()})}
        spec = {"tipo": "numerico", "cortes": [None] + [float(c) for c in cortes[1:-1]] + [None],
                "moda_aparte": frac > umbral_moda}
        if spec["moda_aparte"]:
            spec["moda"] = int(moda) if float(moda).is_integer() else float(moda)
            spec["etiqueta_moda"] = f"= {moda:.4g}"
        return spec


    def aplicar_bins(x, spec):
        """Asigna el bin con la spec congelada. PRESERVA el índice."""
        x = pd.Series(x)
        if spec["tipo"] == "categorico":
            return x.astype(str).where(x.notna(), "MISSING")
        if spec["tipo"] == "discreto":
            return pd.Series([clave_discreta(v) for v in x], index=x.index)
        bordes = [-np.inf] + [float(c) for c in spec["cortes"][1:-1]] + [np.inf]
        et = pd.cut(x, bordes).astype(str)
        if spec["moda_aparte"]:
            et = et.where(x != spec["moda"], spec["etiqueta_moda"])
        return et.where(x.notna(), "MISSING")


    def contrato_datos(dev, variables, tolerancia_missing):
        cl = {}
        for v in variables:
            s = dev[v]
            tope = min(1.0, float(s.isna().mean()) * tolerancia_missing + 0.01)
            if not es_numerica(s):
                cl[v] = {"tipo": "categorico", "categorias": sorted(s.dropna().astype(str).unique().tolist()),
                         "pct_missing_dev": float(s.isna().mean()), "pct_missing_max": tope}
            else:
                cl[v] = {"tipo": "numerico", "minimo": float(np.nanmin(s.values)), "maximo": float(np.nanmax(s.values)),
                         "pct_missing_dev": float(s.isna().mean()), "pct_missing_max": tope}
        return cl


    def construir_artefacto(cfg, datos, corrida, lineage_min):
        I = cfg["implementacion"]
        emb, sc, cal, est = corrida["embudo"], corrida["scorecard"], corrida["calibracion"], corrida["estrategia"]
        dev = datos["muestras"]["DEV"]
        el = emb["elegidas"]
        binning = {v: ajustar_bins(dev[v], cfg["embudo"]["bins_woe"], cfg["embudo"]["umbral_moda"]) for v in el}
        return {
            "formato": {"version_esquema": "1.0"},
            "modelo_id": I["modelo_id"], "version": I["version"],
            "cartera": "Crédito de consumo — Financiera Andes",
            "fecha_construccion": I["fecha_construccion"],
            "variables": list(el),
            "binning": binning,
            "woe": {v: woe_normalizado(sc["mapas"][v], binning[v]) for v in el},
            "coeficientes": {k: float(v) for k, v in emb["modelo"]["params"].items()},
            "escalado": {"pdo": cfg["scorecard"]["pdo"], "score_base": cfg["scorecard"]["score_base"],
                         "odds_base": cfg["scorecard"]["odds_base"], "factor": float(sc["factor"]),
                         "offset": float(sc["offset"])},
            "calibracion": {"metodo": "desplazamiento de intercepto", "ancla": cal["ancla"],
                            "tendencia_central": cal["tc"], "delta": float(cal["delta"]),
                            "ventana": cal["anclas"][cal["ancla"]]["ventana"]},
            "master_scale": {"cortes": cfg["estrategia"]["cortes_banda"], "etiquetas": cfg["estrategia"]["etiquetas_banda"]},
            "politica": {"cutoff": int(est["cutoff"]), "regla": "aprobar si score >= cutoff; revisar si hay bin sin mapa"},
            "contrato_datos": contrato_datos(dev, el, I["tolerancia_missing"]),
            "woe_faltante": 0.0,
            "lineage": lineage_min,
        }


    def puntuar(df, art):
        """PD cruda → PD calibrada → score → banda → decisión, SOLO con el artefacto."""
        lp = np.full(len(df), art["coeficientes"]["const"], dtype=float)
        sin_mapa = np.zeros(len(df), dtype=int)
        for v in art["variables"]:
            w = aplicar_bins(df[v], art["binning"][v]).map(art["woe"][v])
            sin_mapa += w.isna().to_numpy().astype(int)
            lp = lp + art["coeficientes"][v] * w.fillna(art["woe_faltante"]).to_numpy(dtype=float)
        pd_c = sig(logit_np_(sig(lp)) + art["calibracion"]["delta"])
        sc_ = score_desde_pd(pd_c, art["escalado"]["factor"], art["escalado"]["offset"])
        salida = pd.DataFrame({"pd_calibrada": pd_c, "score": sc_,
                               "banda": banda_de(sc_, art["master_scale"]["cortes"], art["master_scale"]["etiquetas"]),
                               "bins_sin_mapa": sin_mapa, "apto_automatico": sin_mapa == 0}, index=df.index)
        salida["decision"] = np.where(~salida["apto_automatico"], "revisar",
                                      np.where(salida["score"] >= art["politica"]["cutoff"], "aprobar", "rechazar"))
        return salida


    def puntuar_con_bug_a(df, art, bins, umbral_moda):
        """EMULACIÓN del bug A (clase 6): el binner se RE-AJUSTA con el lote del día y el WoE de DEV
        se asigna por posición del bin. No falla ninguna línea; cambian decisiones."""
        lp = np.full(len(df), art["coeficientes"]["const"], dtype=float)
        for v in art["variables"]:
            spec_dev = art["binning"][v]
            spec_lote = ajustar_bins(df[v], bins, umbral_moda)
            if spec_dev["tipo"] != "numerico" or spec_lote["tipo"] != "numerico":
                w = aplicar_bins(df[v], spec_dev).map(art["woe"][v])
            else:
                def orden(spec):
                    bordes = [-np.inf] + [float(c) for c in spec["cortes"][1:-1]] + [np.inf]
                    cats = [str(c) for c in pd.cut(pd.Series([np.nan]), bordes).cat.categories]
                    return ([spec["etiqueta_moda"]] if spec["moda_aparte"] else []) + cats
                o_dev, o_lote = orden(spec_dev), orden(spec_lote)
                traduce = {b: o_dev[min(i, len(o_dev) - 1)] for i, b in enumerate(o_lote)}
                traduce["MISSING"] = "MISSING"
                w = aplicar_bins(df[v], spec_lote).map(traduce).map(art["woe"][v])
            lp = lp + art["coeficientes"][v] * w.fillna(art["woe_faltante"]).to_numpy(dtype=float)
        pd_c = sig(logit_np_(sig(lp)) + art["calibracion"]["delta"])
        return score_desde_pd(pd_c, art["escalado"]["factor"], art["escalado"]["offset"])


    def validar_contrato(df, contrato, I):
        """Tabla de hallazgos del lote: 🔴 bloquea la corrida, 🟡 puntúa y deja constancia."""
        h = []

        def anotar(v, sev, regla, detalle, valor):
            h.append({"variable": v, "severidad": sev, "regla": regla, "detalle": detalle, "valor": valor})

        for v, c in contrato.items():
            if v not in df.columns:
                anotar(v, "🔴", "columna_faltante", "la columna no viene en el lote", None)
                continue
            s = df[v]
            num = es_numerica(s)
            if num != (c["tipo"] == "numerico"):
                anotar(v, "🔴", "tipo_incompatible", f"esperado {c['tipo']}", str(s.dtype))
                continue
            pm = float(s.isna().mean())
            if pm > c["pct_missing_max"]:
                bloquea = pm > I["factor_bloqueo"] * c["pct_missing_max"] or pm > I["missing_bloqueo"]
                anotar(v, "🔴" if bloquea else "🟡", "missing_excesivo",
                       f"{pm:.1%} vs tope {c['pct_missing_max']:.1%}", pm)
            obs = s.dropna()
            if c["tipo"] == "numerico" and len(obs):
                fuera = float(((obs < c["minimo"]) | (obs > c["maximo"])).mean())
                if fuera > I["fuera_rango_aviso"]:
                    anotar(v, "🔴" if fuera > I["fuera_rango_bloqueo"] else "🟡", "fuera_de_rango",
                           f"{fuera:.1%} de observados fuera de [{c['minimo']:.4g}, {c['maximo']:.4g}]", fuera)
            elif c["tipo"] == "categorico" and len(obs):
                nuevas = sorted(set(obs.astype(str)) - set(c["categorias"]))
                if nuevas:
                    frac = float(obs.astype(str).isin(nuevas).mean())
                    anotar(v, "🔴" if frac > I["fuera_rango_bloqueo"] else "🟡", "categoria_nueva",
                           f"{nuevas[:5]} ({frac:.1%})", frac)
        return pd.DataFrame(h, columns=["variable", "severidad", "regla", "detalle", "valor"])


    class RegistroAuditoria:
        """Trail con hashes encadenados, reloj lógico determinista y verificación estricta."""

        def __init__(self, corrida_id):
            self.corrida_id = corrida_id
            self.eventos = []
            self._ts = 0

        def registrar(self, paso, evento, **payload):
            cuerpo = {"n": len(self.eventos) + 1, "corrida_id": self.corrida_id, "t": self._ts,
                      "paso": paso, "evento": evento, "payload": limpiar(payload),
                      "hash_previo": self.eventos[-1]["hash"] if self.eventos else "0" * 64}
            self._ts += 1
            cuerpo["hash"] = hash_json({k: v for k, v in cuerpo.items()})
            self.eventos.append(cuerpo)
            return cuerpo

        def sello(self):
            return (self.eventos[-1]["hash"], len(self.eventos)) if self.eventos else (None, 0)

        def verificar_cadena(self, sello=None, cierres=("corrida_terminada", "corrida_abortada")):
            if not self.eventos:
                return False, "trail vacío"
            previo = "0" * 64
            for k, e in enumerate(self.eventos, start=1):
                if e["n"] != k:
                    return False, f"numeración rota en la posición {k}"
                if e["corrida_id"] != self.corrida_id:
                    return False, f"el evento {k} pertenece a otra corrida"
                cuerpo = {c: v for c, v in e.items() if c != "hash"}
                if cuerpo["hash_previo"] != previo:
                    return False, f"el evento {k} no encadena con el anterior"
                if hash_json(cuerpo) != e["hash"]:
                    return False, f"el contenido del evento {k} fue alterado"
                previo = e["hash"]
            if self.eventos[-1]["evento"] not in cierres:
                return False, "el trail no termina en un evento de cierre (truncado)"
            if sello is not None and (len(self.eventos) != sello[1] or self.eventos[-1]["hash"] != sello[0]):
                return False, "no coincide con el sello externo"
            return True, None

        def jsonl(self):
            return "\n".join(json.dumps(e, sort_keys=True, ensure_ascii=False, allow_nan=False) for e in self.eventos) + "\n"

        def tabla(self):
            return pd.DataFrame([{"n": e["n"], "paso": e["paso"], "evento": e["evento"],
                                  "hash": e["hash"][:12]} for e in self.eventos])

    return (
        RegistroAuditoria,
        construir_artefacto,
        puntuar,
        puntuar_con_bug_a,
        validar_contrato,
    )


@app.cell
def _(
    RegistroAuditoria,
    c2,
    construir_artefacto,
    copy,
    fmt_p,
    hash_df,
    hash_json,
    hash_texto,
    json,
    np,
    pct,
    pd,
    puntuar,
    puntuar_con_bug_a,
    snapshot_entorno,
    validar_contrato,
):
    def corrida_produccion(art, lote, I, registro, muestra):
        """Contrato → artefacto → scoring, cada eslabón deja su hash. Aborta si hay 🔴."""
        presentes = [c for c in art["variables"] if c in lote.columns]
        registro.registrar("produccion", "lote_recibido", muestra=muestra, n=int(len(lote)),
                           hash_lote=hash_df(lote[presentes]))
        h = validar_contrato(lote, art["contrato_datos"], I)
        bloq = int((h["severidad"] == "🔴").sum())
        registro.registrar("produccion", "contrato_validado", n_hallazgos=int(len(h)), bloqueantes=bloq,
                           reglas=sorted(h["regla"].unique().tolist()))
        if bloq:
            registro.registrar("produccion", "corrida_abortada", motivo="hallazgos bloqueantes",
                               detalle=h[h["severidad"] == "🔴"][["variable", "regla"]].to_dict("records"))
            return None, h
        registro.registrar("produccion", "artefacto_cargado", hash_artefacto=hash_json(art))
        s = puntuar(lote, art)
        registro.registrar("produccion", "lote_puntuado", n=int(len(s)), score_medio=float(s["score"].mean()),
                           tasa_aprobacion=float((s["decision"] == "aprobar").mean()),
                           n_a_revision=int((s["decision"] == "revisar").sum()),
                           hash_salida=hash_df(s.astype({"banda": "string"})))
        return s, h


    def model_card(cfg, datos, corrida, art, lineage, sello, contrato_ttd, bug_a):
        emb, cal, est, val = corrida["embudo"], corrida["calibracion"], corrida["estrategia"], corrida["validacion"]
        met = val["metricas"]
        mu = datos["muestras"]
        mv = val["muestra_validacion"]
        n_malos_dev = int(mu["DEV"]["malo"].sum())
        imp = est["impacto"].loc[est["cutoff"]]
        return {
            "identidad": {"modelo": art["modelo_id"], "version": art["version"],
                          "tipo": "Scorecard logístico sobre WoE de admisión (PD 90+ a 12 meses)",
                          "cartera": art["cartera"], "fecha_construccion": art["fecha_construccion"],
                          "hash_artefacto": lineage["hash_artefacto"], "config_hash": lineage["config_hash"],
                          "data_hash": lineage["data_hash"], "sello_trail": f"{sello[0]} ({sello[1]} eventos)",
                          "motor": corrida["motor"]},
            "proposito": {
                "uso_previsto": "ordenar solicitudes de crédito de consumo de clientes con ≥ 6 meses de antigüedad y "
                                "apoyar la decisión aprobar/rechazar con un cutoff de política",
                "usuarios": "riesgo de admisión, comité de crédito, validación independiente",
                "usos_no_previstos": [
                    "clientes nuevos sin historia interna (antigüedad < 6 meses): fuera de la población de desarrollo",
                    "fijar precio o cupo: el modelo ordena riesgo de default, no estima pérdida ni elasticidad",
                    "provisiones IFRS 9 / normativa CMF: la PD es TTC/PIT de admisión a 12 meses, sin escenarios ni lifetime",
                    "cobranza o comportamiento de clientes ya cursados: la población y el t₀ son de originación",
                    "otros productos (hipotecario, pymes) o financiamiento de motos sin re-desarrollo y validación"]},
            "datos": {"fuente": f"nexolabs-gh/datos-riesgo-credito@{cfg['datos']['commit'][:10]} (4 parquet, SHA-256 verificados)",
                      "poblacion": datos["modo"],
                      "definicion_target": "malo = peor DPD ≥ 90 en los 12 meses posteriores a t₀; 30–89 indeterminado (fuera)",
                      "muestras": {m: int(len(d)) for m, d in mu.items()},
                      "tendencia_central": f"{cal['ancla'].upper()} {pct(cal['tc'], 2)} (δ = {'+' if cal['delta'] >= 0 else '−'}{c2(abs(cal['delta']), 4)})",
                      "variables_finales": art["variables"]},
            "desempeno": {
                "gini": {m: c2(met.loc[m, "gini"], 3) for m in met.index},
                "ks": {m: c2(met.loc[m, "ks"], 3) for m in met.index},
                "ic95_gini_bootstrap": {k: f"[{c2(v[0], 3)}; {c2(v[1], 3)}]" for k, v in val["ic"].items()},
                "psi_score_ttd": c2(val["psi_ttd"], 3),
                f"binomial_global_{mv}_p": fmt_p(val["p_global"]),
                "hosmer_lemeshow_p_simulado": fmt_p(val["hl"]["p_sim"], cfg["validacion"]["hl_simulaciones"]),
                "tablero": " · ".join(f"{k} {v}" for k, v in val["tablero"]["estado"].value_counts().items()),
                "cutoff": {"valor": int(est["cutoff"]), "aprobacion": pct(imp["aprobacion"]),
                           "mora_observada": pct(imp["mora_observada"], 2)}},
            "limitaciones": [
                f"DEV tiene {c2(len(mu['DEV']), 0)} créditos y {n_malos_dev} malos: el IC95 del Gini OOT mide "
                f"{c2(val['ic']['OOT'][1] - val['ic']['OOT'][0], 3)} de ancho; diferencias menores no son concluyentes.",
                f"Andes se deterioró: tasa DEV {pct(mu['DEV']['malo'].mean())} vs OOT {pct(mu['OOT']['malo'].mean())}. "
                f"Con ancla {cal['ancla'].upper()} el backtesting en {mv} da p = {fmt_p(val['p_global'])}: la PD "
                f"{'subestima' if val['residuo_global'] > 0 else 'sobreestima'} el nivel actual en {c2(abs(val['residuo_global']) * 100, 1)} pts.",
                "El modelo se entrenó solo con aprobados (sin reject inference): el riesgo de la población "
                f"rechazada es extrapolación; la cohorte swap-in ({est['res_swap']['n_swap_in']} casos) debe monitorearse.",
                f"El cutoff {est['cutoff']} sale de un apetito SUPUESTO (mora ≤ {pct(cfg['estrategia']['mora_max'])}, "
                f"aprobación ≥ {pct(cfg['estrategia']['aprobacion_min'], 0)}) y de LGD {pct(cfg['estrategia']['lgd'], 0)} fija con EAD = monto; "
                "no son parámetros estimados.",
                f"La selección depende de convenciones (umbral PSI, IV, correlación, bins, α): la grilla de "
                "sensibilidad de este notebook muestra cuánto cambian variables y cutoff.",
                (f"El contrato de datos sobre la bandeja TTD real deja {len(contrato_ttd)} hallazgo(s) 🟡: "
                 "población fuera del rango observado en DEV." if len(contrato_ttd) else
                 "El contrato de datos no encuentra hallazgos en la bandeja TTD real: sus topes (rango de DEV, 3× el "
                 "missing de DEV) no detectan corrimientos DENTRO del rango, que es lo que mide el PSI/CSI y el mix."),
                f"Re-ajustar el binner con el lote (bug A) cambiaría {c2(bug_a['n_cambian'], 0)} de {c2(bug_a['n'], 0)} decisiones TTD "
                f"({pct(bug_a['frac'])}): el artefacto congelado es obligatorio."],
            "gobierno": {"estado": "desarrollo terminado; pendiente validación independiente",
                         "proxima_revalidacion": "12 meses o al disparar un gatillo",
                         "frecuencia_monitoreo": "mensual (estabilidad) · trimestral (discriminación y calibración)",
                         "diagnostico_hoy": val["diagnostico"]["patron"],
                         "accion_recomendada": val["diagnostico"]["accion"], "firma": val["diagnostico"]["firma"]},
            "entorno": lineage["versiones"],
        }


    def card_markdown(card):
        L = [f"# Model Card — {card['identidad']['modelo']} v{card['identidad']['version']}", ""]
        titulos = {"identidad": "1. Identidad", "proposito": "2. Propósito y uso previsto", "datos": "3. Datos",
                   "desempeno": "4. Desempeño", "limitaciones": "5. Limitaciones", "gobierno": "6. Gobierno",
                   "entorno": "7. Entorno de construcción"}
        for k, t in titulos.items():
            L += [f"## {t}", ""]
            v = card[k]
            if isinstance(v, list):
                L += [f"- {x}" for x in v]
            else:
                for k2, v2 in v.items():
                    et = k2.replace("_", " ").capitalize()
                    if isinstance(v2, dict):
                        L.append(f"- **{et}**: " + " · ".join(f"{a}: `{b}`" for a, b in v2.items()))
                    elif isinstance(v2, list):
                        L.append(f"- **{et}**:")
                        L += [f"  - {x}" for x in v2]
                    else:
                        L.append(f"- **{et}**: {v2}")
            L.append("")
        return "\n".join(L)


    def resumen_ejecutivo(cfg, datos, corrida, lineage, sello):
        emb, cal, est, val = corrida["embudo"], corrida["calibracion"], corrida["estrategia"], corrida["validacion"]
        imp = est["impacto"].loc[est["cutoff"]]
        rs = est["res_swap"]
        met = val["metricas"]
        dg = val["diagnostico"]
        rojos = val["tablero"].index[val["tablero"]["estado"] == "🔴"].tolist()
        return f"""
    ### Resumen ejecutivo · {cfg['implementacion']['modelo_id']} v{cfg['implementacion']['version']}

    **1. Qué se pide aprobar.** El scorecard de admisión de consumo v{cfg['implementacion']['version']}
    ({len(emb['elegidas'])} variables, artefacto `{lineage['hash_artefacto'][:12]}`) con cutoff **{est['cutoff']}**,
    vigente 12 meses o hasta que se dispare un gatillo.

    **2. Qué gana Financiera Andes.** A la misma tasa de aprobación que hoy dejan los knock-outs
    ({pct(rs['aprobacion'])}), la mora de la cartera aprobada baja de {pct(rs['mora_ko'], 2)} a {pct(rs['mora_sc'], 2)}.
    Con el cutoff {est['cutoff']} se aprueba {pct(imp['aprobacion'])} de las solicitudes con {pct(imp['mora_observada'], 2)}
    de mora observada y una pérdida esperada de {pct(imp['el_sobre_monto'], 2)} del monto (LGD {pct(cfg['estrategia']['lgd'], 0)} supuesta).

    **3. Qué se sabe que no funciona.** (i) El nivel de riesgo subió: el modelo, anclado al ciclo, promete
    {pct(val['esperados_v'] / len(datos['muestras'][val['muestra_validacion']]), 1)} de mora y en el periodo más reciente se observó
    {pct(datos['muestras'][val['muestra_validacion']]['malo'].mean(), 1)}. (ii) La capacidad de ordenar se mantiene
    (Gini {c2(met.loc['OOT', 'gini'], 2)} en el periodo reciente) pero se midió con una muestra chica: el margen de error es de
    ±{c2((val['ic']['OOT'][1] - val['ic']['OOT'][0]) / 2, 2)}. (iii) El modelo nunca vio a los clientes rechazados.

    **4. Qué se va a vigilar y con qué gatillo.** Mensual: estabilidad de la bandeja (PSI {c2(val['psi_ttd'], 3)} hoy).
    Trimestral: ranking y nivel. Gatillos: recalibración del nivel si el test global sigue en rojo un trimestre más
    (Jefe de Modelos); re-desarrollo si el Gini cae > 30% o el KS < 0,20 (Comité de Riesgo); contingencia con los
    knock-outs si el contrato de datos bloquea (TI / Producción). Hoy en rojo: {', '.join(rojos) or 'ninguno'}.

    **5. Qué se pide de vuelta.** {dg['accion'][:1].upper() + dg['accion'][1:]}. Firma: {dg['firma']}. Designar al validador independiente
    y confirmar el apetito de mora ({pct(cfg['estrategia']['mora_max'])}) que fija el cutoff.

    <small>config `{lineage['config_hash'][:12]}` · datos `{lineage['data_hash'][:12]}` · sello `{sello[0][:12]}` ({sello[1]} eventos)</small>
    """


    def etapa_implementacion(cfg, config_hash, data_hash, info_fuentes, datos, corrida):
        I = cfg["implementacion"]
        mu = datos["muestras"]
        cal = corrida["calibracion"]
        versiones = snapshot_entorno()
        lineage_min = {"config_hash": config_hash, "data_hash": data_hash, "commit_datos": cfg["datos"]["commit"],
                       "motor": corrida["motor"]}
        art = construir_artefacto(cfg, datos, corrida, lineage_min)
        texto = json.dumps(art, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=1)
        releido = json.loads(texto)
        h_art = hash_json(art)
        # paridad pipeline vs artefacto (y artefacto releído)
        paridad = pd.DataFrame([{"muestra": m, "n": len(mu[m]),
                                 "max_dif_score": float(np.max(np.abs(puntuar(mu[m], art)["score"].to_numpy() - cal["SCORE_CAL"][m]))),
                                 "max_dif_releido": float(np.max(np.abs(puntuar(mu[m], releido)["score"].to_numpy()
                                                                        - puntuar(mu[m], art)["score"].to_numpy())))}
                                for m in ("DEV", "HO", "OOT", "TTD")]).set_index("muestra")
        ttd = mu["TTD"]
        s_ttd = puntuar(ttd, releido)
        # lote vs fila (anti bug A) + identidad del caso
        idx = [3, 40, 200, len(ttd) - 1]
        fila_sola = np.array([puntuar(ttd.iloc[[i]], releido)["score"].iloc[0] for i in idx])
        en_lote = s_ttd["score"].to_numpy()[idx]
        barajado = puntuar(ttd.sample(frac=1.0, random_state=7), releido)["score"].reindex(ttd.index).to_numpy()
        lote_fila = {"fila_vs_lote_max_dif": float(np.max(np.abs(fila_sola - en_lote))),
                     "barajado_max_dif": float(np.max(np.abs(barajado - s_ttd["score"].to_numpy()))),
                     "indice_preservado": bool(list(puntuar(ttd.iloc[idx], releido).index) == list(ttd.index[idx]))}
        s_bug = puntuar_con_bug_a(ttd, releido, cfg["embudo"]["bins_woe"], cfg["embudo"]["umbral_moda"])
        dec_ok = s_ttd["score"].to_numpy() >= releido["politica"]["cutoff"]
        dec_bug = s_bug >= releido["politica"]["cutoff"]
        bug_a = {"n": int(len(ttd)), "n_cambian": int((dec_ok != dec_bug).sum()), "frac": float((dec_ok != dec_bug).mean()),
                 "max_dif_score": float(np.max(np.abs(s_bug - s_ttd["score"].to_numpy())))}
        # contrato: bandeja real y lote roto de laboratorio (receta del Lab 3)
        con = art["contrato_datos"]
        h_ttd = validar_contrato(ttd, con, I)
        numericas = [v for v in art["variables"] if con[v]["tipo"] == "numerico"]
        v0 = numericas[0]
        v1 = next((v for v in art["variables"] if v != v0 and con[v]["pct_missing_max"] < 0.25), None) \
            or next(v for v in art["variables"] if v != v0)
        v2 = next(v for v in art["variables"] if v not in (v0, v1))
        lote_roto = ttd.copy()
        lote_roto[v0] = lote_roto[v0] * 1000
        lote_roto.loc[lote_roto.index[: len(lote_roto) // 2], v1] = np.nan
        lote_roto = lote_roto.drop(columns=[v2])
        h_roto = validar_contrato(lote_roto, con, I)
        # audit trail de la corrida completa
        corrida_id = f"{I['modelo_id']}-{I['version']}-{config_hash[:8]}"
        reg = RegistroAuditoria(corrida_id)
        reg.registrar("inicio", "corrida_iniciada", config_hash=config_hash, data_hash=data_hash,
                      version_config=cfg["version_config"], motor=corrida["motor"], entorno=versiones)
        reg.registrar("datos", "fuentes_verificadas", tablas={t: {"sha256": i["sha256"], "filas": i["filas"]}
                                                              for t, i in info_fuentes.items()})
        for e in datos["eventos"] + corrida["eventos"]:
            reg.registrar(e["paso"], e["evento"], **e["payload"])
        reg.registrar("implementacion", "artefacto_congelado", hash_artefacto=h_art, bytes=len(texto),
                      n_bins=sum(len(m) for m in art["woe"].values()))
        reg.registrar("implementacion", "paridad_verificada", max_dif=float(paridad["max_dif_score"].max()),
                      tolerancia=I["tol_paridad"], lote_vs_fila=lote_fila, bug_a=bug_a)
        salida_ttd, _ = corrida_produccion(releido, ttd, I, reg, "TTD")
        reg.registrar("fin", "corrida_terminada", estado="ok")
        sello = reg.sello()
        ok_cadena, motivo = reg.verificar_cadena(sello=sello)
        # ataques: edición torpe (la cadena la caza) y reescritura prolija (solo el sello la caza)
        torpe = copy.deepcopy(reg)
        n_obj = next(e["n"] for e in torpe.eventos if e["evento"] == "lote_puntuado")
        torpe.eventos[n_obj - 1]["payload"]["tasa_aprobacion"] = 0.99
        ok_torpe, motivo_torpe = torpe.verificar_cadena()
        prolijo = copy.deepcopy(torpe)
        previo = "0" * 64
        for e in prolijo.eventos:
            e["hash_previo"] = previo
            e["hash"] = hash_json({k: v for k, v in e.items() if k != "hash"})
            previo = e["hash"]
        ok_prolijo_cadena = prolijo.verificar_cadena()[0]
        ok_prolijo_sello = prolijo.verificar_cadena(sello=sello)[0]
        reg_roto = RegistroAuditoria(corrida_id + "-lote-roto")
        reg_roto.registrar("inicio", "corrida_iniciada", config_hash=config_hash)
        s_roto, _ = corrida_produccion(releido, lote_roto, I, reg_roto, "TTD-roto")
        jsonl = reg.jsonl()
        lineage = {"config_hash": config_hash, "data_hash": data_hash,
                   "fuentes": {t: i["sha256"] for t, i in info_fuentes.items()},
                   "commit_datos": cfg["datos"]["commit"], "semilla": datos["semilla"], "modo": datos["modo"],
                   "hash_matriz": next(e["payload"]["hash_matriz"] for e in datos["eventos"] if e["evento"] == "matriz_construida"),
                   "hash_artefacto": h_art, "corrida_id": corrida_id, "sello": {"hash_terminal": sello[0], "n_eventos": sello[1]},
                   "hash_jsonl": hash_texto(jsonl), "motor": corrida["motor"], "versiones": versiones}
        card = model_card(cfg, datos, corrida, art, lineage, sello, h_ttd, bug_a)
        return {"artefacto": art, "texto_artefacto": texto, "releido": releido, "hash_artefacto": h_art,
                "paridad": paridad, "lote_fila": lote_fila, "bug_a": bug_a, "salida_ttd": salida_ttd,
                "contrato_ttd": h_ttd, "lote_roto": {"v0": v0, "v1": v1, "v2": v2}, "contrato_roto": h_roto,
                "roto_abortado": s_roto is None and reg_roto.eventos[-1]["evento"] == "corrida_abortada",
                "registro": reg, "sello": sello, "cadena_ok": (ok_cadena, motivo),
                "ataques": {"torpe_detectado": not ok_torpe, "motivo_torpe": motivo_torpe,
                            "prolijo_pasa_cadena": ok_prolijo_cadena, "prolijo_cazado_por_sello": not ok_prolijo_sello},
                "jsonl": jsonl, "lineage": lineage, "card": card, "card_md": card_markdown(card),
                "resumen_md": resumen_ejecutivo(cfg, datos, corrida, lineage, sello)}

    return (etapa_implementacion,)


@app.cell
def _(
    CONFIG,
    c2,
    config_hash,
    corrida,
    data_hash,
    datos,
    etapa_implementacion,
    info_fuentes,
    mo,
    pct,
    time,
):
    _t = time.perf_counter()
    impl = etapa_implementacion(CONFIG, config_hash, data_hash, info_fuentes, datos, corrida)
    t_impl = time.perf_counter() - _t
    _bug = impl["bug_a"]
    _at = impl["ataques"]
    mo.vstack([
        mo.md(f"""
    ### 7a. Artefacto, paridad y lote-vs-fila ({c2(t_impl, 1)} s)

    Artefacto JSON: {c2(len(impl['texto_artefacto']), 0)} caracteres, `allow_nan=False` ✓, hash `{impl['hash_artefacto'][:16]}…`.
    Paridad pipeline ↔ `puntuar(artefacto)` y artefacto ↔ artefacto **releído** desde el texto:
    """),
        impl["paridad"],
        mo.md(f"""
    Lote-vs-fila: puntuar un caso solo o dentro del lote difiere en {impl['lote_fila']['fila_vs_lote_max_dif']:.1e};
    barajar el lote, {impl['lote_fila']['barajado_max_dif']:.1e}; índice preservado: {impl['lote_fila']['indice_preservado']}.
    **Bug A emulado** (re-ajustar el binner con la bandeja TTD y asignar el WoE de DEV por posición):
    **{c2(_bug['n_cambian'], 0)} de {c2(_bug['n'], 0)} decisiones cambian ({pct(_bug['frac'])})** sin ningún error de Python
    (Banco Austral: 587 de 8.585 = 6,8%). El score se mueve hasta {c2(_bug['max_dif_score'], 1)} puntos.
    """),
        mo.md(f"""
    ### 7b. Contrato de datos

    Bandeja TTD real ({len(impl['contrato_ttd'])} hallazgo(s), {int((impl['contrato_ttd']['severidad'] == '🔴').sum())} bloqueante(s)):
    """),
        impl["contrato_ttd"] if len(impl["contrato_ttd"]) else mo.md("_(sin hallazgos)_"),
        mo.md(f"""
    Lote roto de laboratorio (`{impl['lote_roto']['v0']}` ×1000 · `{impl['lote_roto']['v1']}` con 50% NaN ·
    `{impl['lote_roto']['v2']}` ausente) → corrida abortada y registrada: **{impl['roto_abortado']}**.
    """),
        impl["contrato_roto"],
    ])
    return (impl,)


@app.cell
def _(impl, json, limpiar, mo):
    _reg = impl["registro"]
    _at = impl["ataques"]
    mo.vstack([
        mo.md(f"""
    ### 7c. Audit trail, sello y lineage

    {len(_reg.eventos)} eventos encadenados (reloj lógico; cada evento sella al anterior). Cadena + sello:
    **{'verifica ✓' if impl['cadena_ok'][0] else 'NO verifica ✗ ' + str(impl['cadena_ok'][1])}**.
    Edición torpe (cambiar la tasa de aprobación sin re-sellar) → detectada: {_at['torpe_detectado']} («{_at['motivo_torpe']}»).
    Reescritura prolija (recalcular toda la cola) → pasa la cadena sola: {_at['prolijo_pasa_cadena']}; la caza el sello
    externo: {_at['prolijo_cazado_por_sello']}. Por eso el sello `{impl['sello'][0][:16]}…` ({impl['sello'][1]} eventos)
    se archiva **fuera** del log (repositorio WORM o sello de tiempo de un tercero).
    """),
        _reg.tabla(),
        mo.md("**Lineage**"),
        mo.md("```json\n" + json.dumps(limpiar({k: v for k, v in impl["lineage"].items()}), indent=1, ensure_ascii=False) + "\n```"),
        mo.hstack([
            mo.download(data=impl["texto_artefacto"].encode("utf-8"), filename="modelo_andes.json",
                        mimetype="application/json", label="Artefacto JSON"),
            mo.download(data=impl["jsonl"].encode("utf-8"), filename="audit_trail_andes.jsonl",
                        mimetype="application/x-ndjson", label="Audit trail JSONL"),
            mo.download(data=impl["card_md"].encode("utf-8"), filename="model_card_andes.md",
                        mimetype="text/markdown", label="Model card"),
            mo.download(data=impl["resumen_md"].encode("utf-8"), filename="resumen_ejecutivo_andes.md",
                        mimetype="text/markdown", label="Resumen ejecutivo"),
        ], justify="start"),
    ])
    return


@app.cell
def _(impl, mo):
    mo.vstack([mo.md("### 7d. Model card (generada desde la corrida)"), mo.md(impl["card_md"])])
    return


@app.cell
def _(impl, mo):
    mo.vstack([mo.md("### 7e. Resumen ejecutivo de una página (generado desde la corrida)"), mo.md(impl["resumen_md"])])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Sensibilidad de decisiones: cuánto del modelo es convención

    Se re-ejecuta el embudo → scorecard → calibración → estrategia variando **una convención a la vez**
    alrededor de la config vigente (grilla reducida, por defecto) o **todas las combinaciones** (48,
    bajo demanda). Para cada variante: nº de variables, cuáles, Gini por muestra (sobre PD cruda: el
    δ no lo altera), δ TTC/PIT del ancla vigente, cutoff recomendado con el mismo apetito y la similitud
    de Jaccard del conjunto de variables contra la corrida base. La grilla usa el motor `numpy`
    (paridad demostrada en la sección 9).
    """)
    return


@app.cell
def _(
    MOTORES,
    copy,
    etapa_calibracion,
    etapa_embudo,
    etapa_estrategia,
    etapa_scorecard,
    np,
    pd,
    time,
):
    def aplicar_variante(cfg, cambios):
        c = copy.deepcopy(cfg)
        for k, v in cambios.items():
            if k == "alpha":
                c["embudo"]["alpha_entrada"] = c["embudo"]["alpha_salida"] = v
            else:
                c["embudo"][k] = v
        return c


    def evaluar_variante(cfg, datos, motor, ins, psi_tab, base_vars, etiqueta):
        t = time.perf_counter()
        emb_v = etapa_embudo(cfg, datos, motor, ins, psi_tab)
        sc_v = etapa_scorecard(cfg, datos, emb_v, ins, motor)
        cal_v = etapa_calibracion(cfg, datos, sc_v, motor)
        est_v = etapa_estrategia(cfg, datos, sc_v, cal_v, motor)
        g = {m: 2 * motor["auc"](datos["muestras"][m]["malo"].to_numpy(), sc_v["PD_RAW"][m]) - 1 for m in ("DEV", "HO", "OOT")}
        el = emb_v["elegidas"]
        E = cfg["embudo"]
        return {"variante": etiqueta, "psi_max": E["psi_max_ttd"], "iv_min": E["iv_min"], "corr_max": E["corr_max"],
                "bins": E["bins_woe"], "alpha": E["alpha_entrada"],
                "n_embudo": " → ".join(str(n) for _, n in emb_v["conteo"][1:]), "n_vars": len(el),
                "gini_DEV": g["DEV"], "gini_HO": g["HO"], "gini_OOT": g["OOT"],
                "delta": cal_v["delta"], "cutoff": est_v["cutoff"],
                "aprob_cutoff": float(est_v["impacto"].loc[est_v["cutoff"], "aprobacion"]),
                "mora_cutoff": float(est_v["impacto"].loc[est_v["cutoff"], "mora_observada"]),
                "jaccard_base": len(set(el) & set(base_vars)) / len(set(el) | set(base_vars)) if el else 0.0,
                "excl_signo": len(emb_v["excluidas_signo"]),
                "variables": ", ".join(el), "segundos": time.perf_counter() - t}


    def correr_grilla(cfg, datos, variantes, ins_por_bins, psi_tab, base_vars):
        motor = MOTORES["numpy"]
        filas = []
        for etiqueta, cambios in variantes:
            c = aplicar_variante(cfg, cambios)
            filas.append(evaluar_variante(c, datos, motor, ins_por_bins[c["embudo"]["bins_woe"]], psi_tab,
                                          base_vars, etiqueta))
        return pd.DataFrame(filas).set_index("variante")


    def variantes_oat(cfg):
        S, E = cfg["sensibilidad"], cfg["embudo"]
        actual = {"psi_max_ttd": E["psi_max_ttd"], "iv_min": E["iv_min"], "corr_max": E["corr_max"],
                  "bins_woe": E["bins_woe"], "alpha": E["alpha_entrada"]}
        vs = [("base (config vigente)", {})]
        for k, valores in S.items():
            for v in valores:
                if not np.isclose(v, actual[k]):
                    vs.append((f"{k} = {v}", {k: v}))
        return vs


    def variantes_completas(cfg):
        S = cfg["sensibilidad"]
        claves = list(S)
        combos = [dict(zip(claves, vals)) for vals in __import__("itertools").product(*[S[k] for k in claves])]
        return [(" · ".join(f"{k}={v}" for k, v in c.items()), c) for c in combos]

    return correr_grilla, variantes_completas, variantes_oat


@app.cell
def _(
    CONFIG,
    c2,
    correr_grilla,
    corridas,
    datos,
    insumos_actual,
    insumos_woe,
    mo,
    psi_por_motor,
    time,
    variantes_oat,
):
    _t = time.perf_counter()
    _bins_grilla = sorted(set(CONFIG["sensibilidad"]["bins_woe"]) | {CONFIG["embudo"]["bins_woe"]})
    insumos_por_bins = {b: (insumos_actual if b == CONFIG["embudo"]["bins_woe"] else insumos_woe(datos, b))
                        for b in _bins_grilla}
    grilla_reducida = correr_grilla(CONFIG, datos, variantes_oat(CONFIG), insumos_por_bins,
                                    psi_por_motor["numpy"], corridas["numpy"]["embudo"]["elegidas"])
    t_grilla = time.perf_counter() - _t
    mo.vstack([
        mo.md(f"**Grilla reducida (una convención a la vez)**: {len(grilla_reducida)} corridas en {c2(t_grilla, 1)} s."),
        grilla_reducida.drop(columns=["variables", "segundos"]).round(4),
        mo.md("Variables elegidas por variante:"),
        grilla_reducida[["n_vars", "variables"]],
    ])
    return grilla_reducida, insumos_por_bins


@app.cell
def _(c2, grilla_reducida, mo, np, plt, val):
    _g = grilla_reducida
    _fig3, _ax3 = plt.subplots(figsize=(8, 3.6))
    _x = np.arange(len(_g))
    _ax3.bar(_x - 0.2, _g["gini_HO"], 0.4, label="Gini HO", color="#95A5A6")
    _ax3.bar(_x + 0.2, _g["gini_OOT"], 0.4, label="Gini OOT", color="#2E6FF2")
    for _i, (_n, _c) in enumerate(zip(_g["n_vars"], _g["cutoff"])):
        _ax3.text(_i, max(_g["gini_HO"].iloc[_i], _g["gini_OOT"].iloc[_i]) + 0.01, f"{_n} v\n{_c}",
                  ha="center", fontsize=7)
    _ax3.set_xticks(_x)
    _ax3.set_xticklabels(_g.index, rotation=30, ha="right", fontsize=7)
    _ax3.set_ylim(0, max(_g[["gini_HO", "gini_OOT"]].max()) + 0.12)
    _ax3.set_ylabel("Gini")
    _ax3.set_title("Sensibilidad: Gini, nº de variables y cutoff por variante")
    _ax3.legend(fontsize=7)
    _fig3.tight_layout()
    _base = _g.iloc[0]
    _rango_oot = (_g["gini_OOT"].min(), _g["gini_OOT"].max())
    mo.vstack([_fig3, mo.md(f"""
    **Lectura.** Entre las variantes, el nº de variables va de {_g['n_vars'].min()} a {_g['n_vars'].max()}, la
    similitud de Jaccard con la base baja hasta {c2(_g['jaccard_base'].min(), 2)}, el Gini OOT se mueve entre
    {c2(_rango_oot[0], 3)} y {c2(_rango_oot[1], 3)} (rango {c2(_rango_oot[1] - _rango_oot[0], 3)}, contra un IC95
    bootstrap de ancho {c2(val['ic']['OOT'][1] - val['ic']['OOT'][0], 3)}) y el cutoff recomendado toma
    {len(set(_g['cutoff']))} valor(es): {sorted(set(int(c) for c in _g['cutoff']))}. Las convenciones cambian
    **qué variables** cuentan la historia mucho más que **cuánto ordena** el modelo.
    """)])
    return


@app.cell
def _(CONFIG, mo, variantes_completas):
    boton_grilla = mo.ui.run_button(label=f"Correr grilla completa ({len(variantes_completas(CONFIG))} combinaciones, ~1–2 min)")
    boton_grilla
    return (boton_grilla,)


@app.cell
def _(
    CONFIG,
    boton_grilla,
    c2,
    correr_grilla,
    corridas,
    datos,
    insumos_por_bins,
    mo,
    pd,
    psi_por_motor,
    time,
    variantes_completas,
):
    if boton_grilla.value:
        _t = time.perf_counter()
        grilla_completa = correr_grilla(CONFIG, datos, variantes_completas(CONFIG), insumos_por_bins,
                                        psi_por_motor["numpy"], corridas["numpy"]["embudo"]["elegidas"])
        _res = (grilla_completa.groupby(["n_vars"]).size().rename("variantes").to_frame())
        _salida_grilla = mo.vstack([
            mo.md(f"Grilla completa: {len(grilla_completa)} corridas en {c2(time.perf_counter() - _t, 1)} s. "
                  f"Gini OOT: min {c2(grilla_completa['gini_OOT'].min(), 3)} · máx {c2(grilla_completa['gini_OOT'].max(), 3)} · "
                  f"cutoffs distintos: {sorted(set(int(c) for c in grilla_completa['cutoff']))}."),
            grilla_completa.drop(columns=["segundos"]).round(4),
            mo.md(f"Sobreajuste por parsimonia: corr(nº variables, Gini HO) = "
                  f"{c2(grilla_completa['n_vars'].corr(grilla_completa['gini_HO']), 2)} · "
                  f"corr(nº variables, Gini OOT) = {c2(grilla_completa['n_vars'].corr(grilla_completa['gini_OOT']), 2)}. "
                  "Gini medio por nº de variables:"),
            grilla_completa.groupby("n_vars")[["gini_DEV", "gini_HO", "gini_OOT"]].mean().round(3),
            mo.md("Frecuencia de cada variable en las corridas de la grilla:"),
            pd.Series([v for s in grilla_completa["variables"] for v in s.split(", ") if v]).value_counts().to_frame("corridas"),
        ])
    else:
        grilla_completa = None
        _salida_grilla = mo.md("_La grilla completa corre bajo demanda (botón de arriba)._")
    _salida_grilla
    return


@app.cell
def _(
    LogisticRegression,
    c2,
    corridas,
    datos,
    insumos_actual,
    mo,
    np,
    pd,
    warnings,
):
    def comparar_corridas(a, b, tol):
        """Arnés de paridad: los números que importan, motor a motor."""
        ea, eb = a["embudo"], b["embudo"]
        filas = []

        def fila(item, va, vb, t=tol, exacto=False):
            if exacto:
                ok = va == vb
                dif = 0.0 if ok else np.nan
            else:
                dif = float(np.max(np.abs(np.asarray(va, dtype=float) - np.asarray(vb, dtype=float))))
                ok = dif <= t
            filas.append({"ítem": item, "numpy": str(va)[:60], "librerias": str(vb)[:60],
                          "max_dif": dif, "tolerancia": ("igualdad" if exacto else t), "ok": bool(ok)})

        fila("variables finales (orden)", ea["elegidas"], eb["elegidas"], exacto=True)
        fila("fuera por PSI", sorted(ea["fuera_psi"]), sorted(eb["fuera_psi"]), exacto=True)
        fila("PSI TTD (todas las candidatas)", ea["tabla"]["psi_ttd"].fillna(-1).to_numpy(),
             eb["tabla"]["psi_ttd"].fillna(-1).to_numpy(), 1e-12)
        fila("VIF (selección)", ea["vif"].sort_index().to_numpy(), eb["vif"].sort_index().to_numpy(), 1e-8)
        fila("β del modelo final", ea["modelo"]["params"].to_numpy(), eb["modelo"]["params"][ea["modelo"]["params"].index].to_numpy(), 1e-6)
        fila("p-valores del modelo", ea["modelo"]["pvalues"].to_numpy(), eb["modelo"]["pvalues"][ea["modelo"]["pvalues"].index].to_numpy(), 1e-8)
        fila("log-verosimilitud", ea["modelo"]["llf"], eb["modelo"]["llf"], 1e-7)
        fila("δ TTC", a["calibracion"]["anclas"]["ttc"]["delta"], b["calibracion"]["anclas"]["ttc"]["delta"], 1e-9)
        fila("δ PIT", a["calibracion"]["anclas"]["pit"]["delta"], b["calibracion"]["anclas"]["pit"]["delta"], 1e-9)
        fila("score calibrado OOT", a["calibracion"]["SCORE_CAL"]["OOT"], b["calibracion"]["SCORE_CAL"]["OOT"], 1e-5)
        fila("cutoff recomendado", a["estrategia"]["cutoff"], b["estrategia"]["cutoff"], exacto=True)
        fila("Gini DEV/HO/OOT", a["validacion"]["metricas"]["gini"].to_numpy(), b["validacion"]["metricas"]["gini"].to_numpy(), 1e-9)
        fila("KS DEV/HO/OOT", a["validacion"]["metricas"]["ks"].to_numpy(), b["validacion"]["metricas"]["ks"].to_numpy(), 1e-9)
        fila("IC95 bootstrap (HO, OOT, caída)", np.ravel(list(a["validacion"]["ic"].values())),
             np.ravel(list(b["validacion"]["ic"].values())), 1e-9)
        fila("binomial global p", a["validacion"]["p_global"], b["validacion"]["p_global"], 1e-9)
        fila("binomial por banda p", a["validacion"]["backtest"]["p_valor"].fillna(-1).to_numpy(),
             b["validacion"]["backtest"]["p_valor"].fillna(-1).to_numpy(), 1e-6)
        fila("HL p χ²", a["validacion"]["hl"]["p_chi2"], b["validacion"]["hl"]["p_chi2"], 1e-10)
        fila("HL p simulado", a["validacion"]["hl"]["p_sim"], b["validacion"]["hl"]["p_sim"], 1e-12)
        fila("PSI score DEV→TTD", a["validacion"]["psi_ttd"], b["validacion"]["psi_ttd"], 1e-12)
        fila("tablero (estados)", a["validacion"]["tablero"]["estado"].tolist(), b["validacion"]["tablero"]["estado"].tolist(), exacto=True)
        return pd.DataFrame(filas).set_index("ítem")


    paridad_motores = comparar_corridas(corridas["numpy"], corridas["librerias"], 1e-9)
    _el = corridas["numpy"]["embudo"]["elegidas"]
    _Wd = insumos_actual["W"]["DEV"][_el].to_numpy()
    _yd = datos["muestras"]["DEV"]["malo"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # sklearn 1.8 avisa que C=inf equivale a penalty=None
        _sk_mle = LogisticRegression(C=np.inf, max_iter=10_000, tol=1e-12).fit(_Wd, _yd)
    _sk_c1 = LogisticRegression(max_iter=10_000).fit(_Wd, _yd)
    _b_np = corridas["numpy"]["embudo"]["modelo"]["params"][_el].to_numpy()
    dif_sklearn = {"C=inf (MLE)": float(np.max(np.abs(_sk_mle.coef_.ravel() - _b_np))),
                   "C=1 (default, L2)": float(np.max(np.abs(_sk_c1.coef_.ravel() - _b_np)))}
    mo.vstack([
        mo.md(f"""
    ## 9. Arnés de paridad: motor `numpy` vs motor `librerias`

    Ambos motores corren el pipeline completo con la misma config. La paridad exige **mismas decisiones**
    (variables, cutoff, semáforos) y números dentro de tolerancia. {int(paridad_motores['ok'].sum())} de
    {len(paridad_motores)} ítems OK. Tiempos: numpy {c2(sum(corridas['numpy']['tiempos'].values()), 1)} s ·
    librerías {c2(sum(corridas['librerias']['tiempos'].values()), 1)} s.

    Tercer oráculo, scikit-learn sobre la misma matriz WoE: `LogisticRegression(C=np.inf)` (MLE; en
    sklearn 1.8 `penalty` está deprecado) difiere de nuestros β en {dif_sklearn['C=inf (MLE)']:.1e};
    el **default `C=1` regulariza** y difiere en {c2(dif_sklearn['C=1 (default, L2)'], 3)} — la trampa clásica (M23).
    """),
        paridad_motores,
    ])
    return dif_sklearn, paridad_motores


@app.cell
def _(
    CONFIG,
    MOTORES,
    T0_NOTEBOOK,
    c2,
    config_hash,
    corrida,
    corridas,
    datos,
    dif_sklearn,
    grilla_reducida,
    hash_json,
    hash_texto,
    impl,
    logit_np_,
    mo,
    np,
    paridad_motores,
    sig,
    time,
):
    # ============================ CHECKS DEL INTEGRADOR ============================
    _c = corrida
    _emb, _sc, _cal, _val = _c["embudo"], _c["scorecard"], _c["calibracion"], _c["validacion"]
    _mu = datos["muestras"]
    checks = {}
    # 1. paridad numpy vs librerías
    checks["paridad numpy vs librerías (todos los ítems)"] = bool(paridad_motores["ok"].all())
    checks["sklearn C=inf ≈ MLE propio (1e-4)"] = dif_sklearn["C=inf (MLE)"] < 1e-4
    # 2. invariantes del modelo
    checks["β < 0 en todas las variables finales"] = bool((_emb["modelo"]["params"].drop("const") < 0).all())
    checks["p < α en el modelo final"] = bool((_emb["modelo"]["pvalues"].drop("const") < CONFIG["embudo"]["alpha_salida"]).all())
    checks["factor y offset del curso"] = abs(_sc["factor"] - 28.8539) < 1e-4 and abs(_sc["offset"] - 487.1229) < 1e-4
    checks["score crudo = suma de puntos"] = _sc["dif_suma_puntos"] < 1e-9
    checks["score calibrado = suma de puntos − factor·δ"] = max(
        float(np.max(np.abs(_cal["SCORE_CAL"][m] - (_sc["SCORE_RAW"][m] - _sc["factor"] * _cal["delta"])))) for m in _mu) < 1e-6
    _auc = MOTORES["numpy"]["auc"]
    _gini_inv = []
    for _m in ("DEV", "HO", "OOT"):
        _y = _mu[_m]["malo"].to_numpy()
        _g0 = 2 * _auc(_y, _sc["PD_RAW"][_m]) - 1
        _g1 = 2 * _auc(_y, sig(logit_np_(_sc["PD_RAW"][_m]) + 0.7)) - 1
        _gini_inv.append(abs(_g0 - _val["metricas"].loc[_m, "gini"]) + abs(_g0 - _g1))
    checks["Gini invariante a δ (δ elegido y δ = 0,7)"] = max(_gini_inv) < 1e-12
    _a = _cal["anclas"][_cal["ancla"]]
    _marco_lp = np.concatenate([_sc["LP"][m][_mu[m]["fecha_solicitud"].between(*_a["ventana"]).to_numpy()]
                                for m in ("DEV", "HO", "OOT")])
    checks["calibración: PD cal media = objetivo en la ventana"] = abs(sig(_marco_lp + _cal["delta"]).mean() - _a["tc"]) < 1e-10
    checks["TTC y PIT: calibrar no valida con la misma muestra (ancla TTC)"] = (_cal["ancla"] != "ttc") or (not _cal["circular"])
    # 3. implementación
    checks["artefacto JSON sin NaN/Infinity"] = "NaN" not in impl["texto_artefacto"] and "Infinity" not in impl["texto_artefacto"]
    checks["paridad artefacto ↔ pipeline (tol 1e-9)"] = float(impl["paridad"]["max_dif_score"].max()) < CONFIG["implementacion"]["tol_paridad"]
    checks["artefacto releído = mismo score (exacto)"] = float(impl["paridad"]["max_dif_releido"].max()) == 0.0
    checks["lote vs fila y barajado (exacto)"] = impl["lote_fila"]["fila_vs_lote_max_dif"] == 0.0 and impl["lote_fila"]["barajado_max_dif"] == 0.0 and impl["lote_fila"]["indice_preservado"]
    checks["contrato: lote roto bloquea (3 reglas, 🔴 rango y missing)"] = (
        {"columna_faltante", "fuera_de_rango", "missing_excesivo"} <= set(impl["contrato_roto"]["regla"])
        and {"fuera_de_rango", "missing_excesivo"} <= set(impl["contrato_roto"].loc[impl["contrato_roto"]["severidad"] == "🔴", "regla"])
        and impl["roto_abortado"])
    checks["cadena de hashes + sello verifican"] = impl["cadena_ok"][0]
    checks["edición torpe detectada; prolija solo por el sello"] = (impl["ataques"]["torpe_detectado"]
                                                                  and impl["ataques"]["prolijo_pasa_cadena"]
                                                                  and impl["ataques"]["prolijo_cazado_por_sello"])
    checks["JSONL: una línea por evento y hash reproducible"] = (len(impl["jsonl"].strip().split("\n")) == impl["sello"][1]
                                                                and hash_texto(impl["jsonl"]) == impl["lineage"]["hash_jsonl"])
    checks["config_hash reproducible"] = hash_json(CONFIG) == config_hash
    checks["grilla: la variante base reproduce la corrida numpy"] = (
        grilla_reducida.iloc[0]["variables"] == ", ".join(corridas["numpy"]["embudo"]["elegidas"])
        and grilla_reducida.iloc[0]["cutoff"] == corridas["numpy"]["estrategia"]["cutoff"])
    # hallazgos que dependen de los DATOS (no de la construcción): se reportan, no se afirman
    avisos = {
        "master scale monótona (tasa observada de E a A1)": bool(_c["estrategia"]["monotona"]),
        "bandeja TTD real sin hallazgos bloqueantes del contrato": bool((impl["contrato_ttd"]["severidad"] != "🔴").all()),
        "el bug A emulado cambia decisiones en TTD": impl["bug_a"]["n_cambian"] > 0,
        "cutoff factible con el apetito declarado": _c["estrategia"]["estado_cutoff"].startswith("factible"),
        "calibración y validación con muestras distintas": not _c["calibracion"]["circular"],
    }
    t_total = time.perf_counter() - T0_NOTEBOOK
    _fallan = [k for k, v in checks.items() if not v]
    assert not _fallan, f"Checks que fallan: {_fallan}"
    mo.md(f"""
    ## Checks del integrador

    {len(checks)} de {len(checks)} OK ✓ · tiempo total del notebook {c2(t_total, 1)} s.

    """ + "\n".join(f"- ✓ {k}" for k in checks)
          + "\n\n**Hallazgos dependientes de los datos** (se reportan; no detienen el notebook):\n\n"
          + "\n".join(f"- {'✓' if v else '⚠️'} {k}" for k, v in avisos.items()))
    return


if __name__ == "__main__":
    app.run()

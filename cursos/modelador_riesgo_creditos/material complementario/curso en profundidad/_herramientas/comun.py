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

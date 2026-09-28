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
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # M00 · Plantilla
    Texto introductorio. Fórmula: $\text{PSI}=\sum_b (a_b-e_b)\ln(a_b/e_b)$.
    """)
    return


@app.cell
def _(np, pd):
    # === COMÚN (pegado verbatim desde _spec/comun.py) ===
    def _sig(z):
        return 1 / (1 + np.exp(-z))
    demo = pd.DataFrame({"x": np.arange(5)})
    return (demo,)


@app.cell
def _(mo):
    semilla = mo.ui.slider(1, 100, value=7, label="Semilla")
    semilla
    return (semilla,)


@app.cell
def _(demo, mo, np, plt, semilla):
    _rng = np.random.default_rng(semilla.value)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(demo["x"], _rng.normal(size=5))
    mo.vstack([mo.md(f"Semilla = {semilla.value}"), _fig])
    return


@app.cell
def _(np):
    # CHECK automático: debe pasar siempre
    assert np.isclose(1 + 1, 2)
    return


if __name__ == "__main__":
    app.run()

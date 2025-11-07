## Acerca de

Este repositorio reúne notas de trabajo, cuadernos y recursos de apoyo de mi recorrido como Decision Scientist y Analytic Engineer. Funciona como un archivo vivo de técnicas, implementaciones de referencia y experimentos que revisito y actualizo con el tiempo.

## Estructura del repositorio

- `credit_intelligence_&_modelling/` – estudios específicos del dominio. El módulo `cap12_borrowed_measures` actualmente incluye notebooks, presentaciones y datos sobre modelado de riesgo, teoría de probabilidades y analítica relacionada.
- `requirements.txt` – especificación del entorno Python para reproducir los notebooks exploratorios.
- `LICENSE` – términos de la licencia (GPLv3).

## Primeros pasos

1. Asegúrate de tener Python 3.10 o superior instalado.
2. (Recomendado) Crea un entorno virtual dedicado:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Inicia Jupyter para explorar los notebooks:
   ```bash
   jupyter lab
   ```

## Notas sobre los datos

Los conjuntos de datos de ejemplo se encuentran en `credit_intelligence_&_modelling/cap12_borrowed_measures/data/`. Las carpetas siguen la convención de [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) (`raw/`, `interim/`, `processed/`, `visualizations/`). Mantén los datos crudos inmutables y guarda los artefactos derivados en la capa correspondiente.

## Licencia

Este proyecto está licenciado bajo la GNU General Public License v3.0 (GPLv3).  
Eres libre de usarlo para cualquier propósito, pero cualquier modificación u obra derivada también debe compartirse bajo la misma licencia e incluir la atribución correspondiente.  
Consulta el archivo [LICENSE](./LICENSE) para más detalles.

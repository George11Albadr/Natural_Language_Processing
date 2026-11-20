# Foodie Advisor — Procesamiento de Lenguaje Natural (UFM)

**Autor:** George Albadr

## Cómo Ejecutar el Proyecto

### 1. Entorno virtual e instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
```

### 2. Generar los datos y los modelos

La app lee los datos procesados (`data/procesado/`) y los modelos entrenados (`modelos/`), que no se versionan por su tamaño. Se generan corriendo los notebooks en orden:

```bash
cd notebooks
for nb in 01_datos_y_exploracion 02_detector 03_costos_y_producto; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
cd ..
```

- El `01` descarga el dataset de TripAdvisor desde Kaggle con `kagglehub` (≈2.3 GB, queda en caché) y lee SemEval-2016 desde `data/`.
- El `02` necesita los CSV de anotación en `data/anotacion/` (no se versionan porque contienen el texto de las reseñas).

### 3. Abrir la aplicación

```bash
streamlit run producto/app.py
```

Se abre en el navegador (por defecto en `http://localhost:8501`). Es un recorrido de seis vistas:

1. **El problema:** el ciclo de servicio, sus cuatro etapas y los datos.
2. **Cómo funciona:** los retos, el conjunto de prueba, la comparación de modelos y una caja para probar oraciones en vivo con el léxico, la regresión logística y el LSTM.
3. **Mi restaurante:** el local frente a sus pares de la ciudad, con su etapa prioritaria, la evolución por año y citas textuales de clientes.
4. **Reseñas nuevas:** pegar reseñas o subir un CSV de un local que no está en los datos y obtener la misma recomendación.
5. **Mi ciudad:** cuánto cuesta cada queja y qué restaurantes tienen cada punto débil.
6. **Resultados:** costo de las fallas, estabilidad de la recomendación en el tiempo y límites.

# Instrucciones de Ejecución del Prototipo Funcional (03_App)

## 1. Requisitos del Sistema y Dependencias
Asegúrese de contar con Python 3.9+ instalado. Instale las dependencias ejecutando en su terminal:

```bash
pip install -r requirements.txt
```

## 2. Estructura de Artefactos
Copie la carpeta `artefactos/` (generada al final del notebook `02_modelos_evaluacion.ipynb`) dentro de `03_App/artefactos/` o indique su ruta directamente desde la barra lateral de la aplicación.
*(Nota: Si ejecuta `app.py` sin la carpeta `artefactos/`, la aplicación iniciará automáticamente en **Modo Demostración Verificado** mostrando los casos exactos del Notebook 02).*

## 3. Comando de Ejecución en Vivo
Dentro de la carpeta `03_App`, ejecute:

```bash
streamlit run app.py
```

El navegador se abrirá automáticamente en `http://localhost:8501` con los tres módulos interactivos:
1. **Usuario Existente (Warm Start):** Recomendación personalizada Top-K con el modelo ganador `EASE (λ = 70)` y visualización del historial consumido.
2. **Usuario Nuevo (Cold Start):** Filtrado interactivo por sistema operativo (`Windows`, `macOS`, `Linux`, `Steam Deck`), presupuesto máximo en USD y porcentaje mínimo de reseñas positivas.
3. **Evaluación Experimental:** Tablero comparativo con las métricas oficiales en Test (`Precision@K`, `Recall@K`, `HitRate@K`, `NDCG@K`).

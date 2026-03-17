"""
predictor.py — Motor de predicción del riesgo de defecto en gel coat
Replica exactamente el pipeline del notebook:
  feature engineering → OHE → StandardScaler → modelo sklearn
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    recall_score, precision_score, f1_score, roc_auc_score
)

# ── Rutas — busca los .pkl en la raíz del repo O en modelo_produccion/ ────────
_BASE = Path(__file__).parent

def _buscar_pkl(nombres):
    """Busca el archivo probando varios nombres y ubicaciones."""
    for carpeta in [_BASE, _BASE / "modelo_produccion"]:
        for nombre in nombres:
            ruta = carpeta / nombre
            if ruta.exists():
                return ruta
    return _BASE / nombres[0]  # ruta preferida (para el mensaje de error)

PKL_PATH  = _buscar_pkl(["pipeline_gelcoat.pkl", "Pipeline_gelcoat.pkl"])
META_PATH = _buscar_pkl(["metadatos_modelo.pkl",  "Metadatos_modelo.pkl"])

# ── Opciones válidas para variables categóricas ───────────────────────────────
OPCIONES = {
    "shift":                  ["mañana", "tarde", "noche"],
    "boat_model":             ["R-14","W-16","R-18","J-18","W-22","W-22 BA","R-22","W-23 II","W-23 M","W-25","W-25 BA","W-26","W-26 BA","W-267","W-29","W-33","IM-22","IM-28-A"],
    "compressor_oil_level":   ["alto", "medio", "bajo"],
    "air_hose_status":        ["ok", "desgaste", "fuga"],
    "catalyst_filter_status": ["ok", "requiere_cambio"],
    "mold_wash_method":       ["solvente", "detergente", "otro"],
}

# ── Umbrales de riesgo ────────────────────────────────────────────────────────
NIVEL_BAJO     = 0.30
# NIVEL_ALTO  = umbral calibrado (viene del pkl)


# =============================================================================
# Carga del modelo
# =============================================================================

def cargar_modelo():
    """Carga el pipeline y los metadatos serializados."""
    if not PKL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {PKL_PATH}.\n"
            "Ejecuta primero la celda 'exportar_modelo' del notebook "
            "para generar modelo_produccion/pipeline_gelcoat.pkl"
        )
    pipeline  = joblib.load(PKL_PATH)
    metadatos = joblib.load(META_PATH) if META_PATH.exists() else {}
    return pipeline, metadatos


# =============================================================================
# Feature Engineering  — replica exacta del notebook
# =============================================================================

def _feature_engineering(datos: dict) -> dict:
    """
    Recibe los valores crudos del formulario y genera todas las
    características derivadas tal como las hace el notebook.
    """
    d = dict(datos)

    # ── Variables temporales ────────────────────────────────────────────────
    ahora = datetime.now()
    d["hora"]          = ahora.hour
    d["dia_semana"]    = ahora.weekday()          # 0=lunes … 6=domingo
    d["mes"]           = ahora.month
    d["es_fin_semana"] = 1 if ahora.weekday() >= 5 else 0
    d["catalyst_batch_year"] = d.get("catalyst_batch_year", ahora.year)

    # ── Features de interacción (Cell 45 del notebook) ──────────────────────
    fan  = float(d.get("spray_fan_pressure_psi",  0) or 0)
    main = float(d.get("spray_main_pressure_psi", 0) or 0)
    d["ratio_presion_spray"]     = main / (fan + 1)
    d["delta_temp"]              = (float(d.get("mold_temp_c",      0) or 0)
                                  - float(d.get("ambient_temp_c",   0) or 0))
    d["viscosidad_x_catalisis"]  = (float(d.get("gelcoat_viscosity_cps", 0) or 0)
                                  * float(d.get("catalyst_percent",      0) or 0))
    d["presion_x_horas_filtro"]  = (float(d.get("compressor_main_pressure_psi", 0) or 0)
                                  * float(d.get("catalyst_filter_hours",        0) or 0))

    # ── Booleanos → entero ───────────────────────────────────────────────────
    for col in ["mold_washed", "wax_applied", "release_agent_applied"]:
        d[col] = 1 if d.get(col) else 0

    return d


def construir_dataframe(datos: dict, feature_names: list) -> pd.DataFrame:
    """
    Aplica feature engineering + OHE y construye un DataFrame
    con exactamente las columnas que espera el pipeline.
    """
    d = _feature_engineering(datos)

    # OHE manual para las variables categóricas
    # (el scaler dentro del pipeline ya espera las columnas numéricas)
    fila = {}
    for col in feature_names:
        # Detectar columnas OHE: "variable_categoria"
        ohe_match = _detectar_ohe(col)
        if ohe_match:
            var_origen, categoria = ohe_match
            valor_actual = str(d.get(var_origen, ""))
            fila[col] = 1 if valor_actual == categoria else 0
        else:
            val = d.get(col, 0)
            try:
                fila[col] = float(val) if val not in (None, "") else 0.0
            except (ValueError, TypeError):
                fila[col] = 0.0

    return pd.DataFrame([fila], columns=feature_names)


# Categorías conocidas de OHE (se infieren del nombre de columna)
_OHE_VARS = [
    "shift", "boat_model", "compressor_oil_level",
    "air_hose_status", "catalyst_filter_status", "mold_wash_method",
]

def _detectar_ohe(nombre_col: str):
    """
    Detecta si una columna es resultado de OHE.
    pandas genera: variable_categoria (con drop_first=True).
    Retorna (var_origen, categoria) o None.
    """
    for var in _OHE_VARS:
        prefijo = var + "_"
        if nombre_col.startswith(prefijo):
            categoria = nombre_col[len(prefijo):]
            return var, categoria
    return None


# =============================================================================
# Predicción
# =============================================================================

def predecir(datos: dict, pipeline, metadatos: dict) -> dict:
    """
    Predicción completa con semáforo, factores y metadatos.

    Parámetros
    ----------
    datos     : dict con los valores del formulario
    pipeline  : Pipeline sklearn (scaler + modelo)
    metadatos : dict con umbral, feature_names, etc.

    Retorna
    -------
    dict con probabilidad, nivel, factores y más.
    """
    feature_names     = metadatos.get("features", [])
    umbral_produccion = metadatos.get("umbral_produccion", 0.35)

    # Construir vector de entrada
    X_input = construir_dataframe(datos, feature_names)

    # Probabilidad de defecto
    prob = float(pipeline.predict_proba(X_input)[0, 1])

    # Clasificación
    clasificacion = int(prob >= umbral_produccion)

    # Semáforo
    if prob < NIVEL_BAJO:
        nivel  = "BAJO"
        color  = "green"
        emoji  = "🟢"
        accion = "Proceso dentro de parámetros. Proceder con la aplicación."
    elif prob < umbral_produccion:
        nivel  = "MODERADO"
        color  = "orange"
        emoji  = "🟡"
        accion = "Revisar variables críticas con el supervisor antes de iniciar."
    else:
        nivel  = "ALTO"
        color  = "red"
        emoji  = "🔴"
        accion = "¡DETENER! Corregir parámetros antes de aplicar el gel coat."

    # Top factores de riesgo
    modelo = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]
    X_sc   = scaler.transform(X_input)

    if hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_[0])
    elif hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
    else:
        importancias = np.zeros(len(feature_names))

    top_idx = np.argsort(importancias)[::-1][:10]
    factores = [
        {
            "variable":    feature_names[i].replace("_", " "),
            "importancia": float(importancias[i]),
            "direccion":   "↑ riesgo" if (
                hasattr(modelo, "coef_") and modelo.coef_[0][i] > 0
            ) else "↓ riesgo",
        }
        for i in top_idx
    ]

    return {
        "probabilidad":   round(prob, 4),
        "porcentaje":     f"{prob*100:.1f}%",
        "clasificacion":  clasificacion,
        "nivel":          nivel,
        "color":          color,
        "emoji":          emoji,
        "accion":         accion,
        "umbral":         umbral_produccion,
        "factores":       factores,
        "modelo_nombre":  metadatos.get("modelo", "—"),
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================================================================
# Análisis de umbral (para la página de calibración)
# =============================================================================

def analizar_umbral(pipeline, metadatos: dict, X_test: pd.DataFrame,
                    y_test: pd.Series) -> dict:
    """
    Calcula curvas PR, ROC y métricas por umbral para la página de calibración.
    """
    feature_names = metadatos.get("features", [])
    X_sc = pipeline.named_steps["scaler"].transform(
        X_test.reindex(columns=feature_names, fill_value=0)
    )
    probs = pipeline.named_steps["model"].predict_proba(X_sc)[:, 1]

    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, probs)
    fpr, tpr, thresholds_roc           = roc_curve(y_test, probs)
    roc_auc                            = roc_auc_score(y_test, probs)

    # Métricas por umbral (0.05 a 0.95)
    umbrales = np.arange(0.05, 0.96, 0.05)
    filas = []
    for u in umbrales:
        yp = (probs >= u).astype(int)
        cm = confusion_matrix(y_test, yp)
        fn = cm[1, 0] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        filas.append({
            "umbral":    round(float(u), 2),
            "recall":    round(recall_score(y_test, yp, zero_division=0), 3),
            "precision": round(precision_score(y_test, yp, zero_division=0), 3),
            "f1":        round(f1_score(y_test, yp, zero_division=0), 3),
            "fn":        int(fn),
            "fp":        int(fp),
        })

    return {
        "precisions":      precisions.tolist(),
        "recalls":         recalls.tolist(),
        "thresholds_pr":   thresholds_pr.tolist(),
        "fpr":             fpr.tolist(),
        "tpr":             tpr.tolist(),
        "roc_auc":         round(roc_auc, 4),
        "metricas_umbral": filas,
    }

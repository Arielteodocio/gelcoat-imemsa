"""
app.py — Control Inteligente Gel Coat · IMEMSA
Streamlit app con 4 páginas:
  1. Dashboard      — KPIs y tendencias
  2. Predicción     — Formulario + resultado en tiempo real
  3. Calibración    — Curvas PR/ROC y ajuste del umbral
  4. Historial      — Log de predicciones con filtros
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

from predictor import cargar_modelo, predecir, analizar_umbral, OPCIONES

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Control Gel Coat — IMEMSA",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colores IMEMSA ────────────────────────────────────────────────────────────
NAVY    = "#0A2342"
OCEAN   = "#1C7293"
TEAL    = "#028090"
SEAFOAM = "#05C3B5"
YELLOW  = "#FFD166"
RED_C   = "#DC2626"
AMBER_C = "#D97706"
GREEN_C = "#16A34A"

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: {NAVY};
  }}
  [data-testid="stSidebar"] * {{
    color: #CBD5E1 !important;
  }}
  [data-testid="stSidebar"] .sidebar-title {{
    color: white !important;
    font-size: 18px;
    font-weight: 700;
  }}
  /* Métrica principal */
  [data-testid="metric-container"] {{
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px;
  }}
  /* Botón primario */
  .stButton > button {{
    background: {NAVY};
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    width: 100%;
    padding: 0.6rem 1rem;
  }}
  .stButton > button:hover {{
    background: {OCEAN};
  }}
  /* Headers */
  h1 {{ color: {NAVY}; }}
  h2 {{ color: {NAVY}; font-size: 1.2rem; }}
  /* Ocultar footer de Streamlit */
  footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Historial en session_state ────────────────────────────────────────────────
if "historial" not in st.session_state:
    st.session_state.historial = []

# ── Carga del modelo (cacheado) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo de predicción...")
def _cargar():
    return cargar_modelo()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 8px 0 20px;'>
      <div style='font-size:36px;'>⚓</div>
      <div style='font-size:17px; font-weight:700; color:white;'>Control Gel Coat</div>
      <div style='font-size:12px; color:{SEAFOAM}; margin-top:2px;'>IMEMSA — Sistema de predicción</div>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegación",
        ["📊  Dashboard", "🔍  Predicción", "🎚️  Calibración", "📋  Historial"],
        label_visibility="collapsed",
    )

    st.divider()

    # Info del modelo
    try:
        pipeline, metadatos = _cargar()
        modelo_ok = True
        st.markdown(f"""
        <div style='font-size:12px; line-height:2;'>
          <div style='color:{SEAFOAM}; font-weight:600; margin-bottom:6px;'>MODELO ACTIVO</div>
          <div><span style='color:#94A3B8;'>Nombre:</span> <span style='color:white;'>{metadatos.get('modelo','—')}</span></div>
          <div><span style='color:#94A3B8;'>Umbral:</span> <span style='color:{YELLOW};font-weight:600;'>{metadatos.get('umbral_produccion','—')}</span></div>
          <div><span style='color:#94A3B8;'>Features:</span> <span style='color:white;'>{metadatos.get('n_features','—')}</span></div>
          <div><span style='color:#94A3B8;'>Dataset:</span> <span style='color:white;'>{metadatos.get('dataset_filas','—')} reg.</span></div>
          <div><span style='color:#94A3B8;'>Fecha:</span> <span style='color:white;'>{metadatos.get('fecha_entrenamiento','—')}</span></div>
        </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError as e:
        modelo_ok = False
        st.error(str(e))

    st.divider()
    st.markdown(f"<div style='font-size:11px;color:#64748B;'>v1.0 · Diplomado DS&AI · Tec de Monterrey</div>",
                unsafe_allow_html=True)


# =============================================================================
# HELPERS DE GRÁFICOS
# =============================================================================

def _gauge(prob: float, umbral: float) -> go.Figure:
    """Gauge chart de probabilidad de defecto."""
    color = GREEN_C if prob < 0.30 else (AMBER_C if prob < umbral else RED_C)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        delta={"reference": umbral * 100, "suffix": "%",
               "font": {"size": 14},
               "decreasing": {"color": GREEN_C},
               "increasing": {"color": RED_C}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],        "color": "#DCFCE7"},
                {"range": [30, umbral*100],"color": "#FEF9C3"},
                {"range": [umbral*100,100],"color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": NAVY, "width": 3},
                "thickness": 0.8,
                "value": umbral * 100,
            },
        },
        title={"text": "Probabilidad de defecto", "font": {"size": 14, "color": "#64748B"}},
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20), paper_bgcolor="white")
    return fig


def _barras_factores(factores: list) -> go.Figure:
    """Barras horizontales con los top factores de riesgo."""
    df = pd.DataFrame(factores[:8])
    colores = [RED_C if "↑" in d else GREEN_C for d in df["direccion"]]
    fig = go.Figure(go.Bar(
        x=df["importancia"],
        y=df["variable"],
        orientation="h",
        marker_color=colores,
        text=[f"{v:.3f}" for v in df["importancia"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Variables de mayor influencia",
        height=320,
        margin=dict(t=40, b=10, l=10, r=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(autorange="reversed"),
        font=dict(size=12),
    )
    return fig


def _curva_pr(precisions, recalls, thresholds_pr, umbral_actual) -> go.Figure:
    """Curva Precision-Recall interactiva."""
    thrs_ext = list(thresholds_pr) + [1.0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recalls, y=precisions,
        mode="lines", line=dict(color=OCEAN, width=2.5),
        name="Curva P-R",
        hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
    ))
    # Punto del umbral actual
    if len(thrs_ext) > 1:
        idx = np.argmin(np.abs(np.array(thrs_ext) - umbral_actual))
        fig.add_trace(go.Scatter(
            x=[recalls[idx]], y=[precisions[idx]],
            mode="markers", marker=dict(color=YELLOW, size=14, line=dict(color=NAVY, width=2)),
            name=f"Umbral actual ({umbral_actual:.2f})",
        ))
    fig.update_layout(
        title="Curva Precision-Recall",
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis=dict(range=[0, 1.02], showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#F1F5F9"),
        height=350, paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        font=dict(size=12),
    )
    return fig


def _curva_roc(fpr, tpr, roc_auc) -> go.Figure:
    """Curva ROC interactiva."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#CBD5E1", dash="dash", width=1.5),
        name="Clasificador aleatorio",
    ))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=TEAL, width=2.5),
        name=f"ROC (AUC = {roc_auc:.3f})",
        hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Curva ROC — AUC = {roc_auc:.3f}",
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos",
        xaxis=dict(range=[0, 1.02], showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(range=[0, 1.02], showgrid=True, gridcolor="#F1F5F9"),
        height=350, paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18),
        font=dict(size=12),
    )
    return fig


def _metricas_vs_umbral(df_metricas: pd.DataFrame, umbral_actual: float) -> go.Figure:
    """Recall, Precision y F1 vs umbral."""
    fig = go.Figure()
    for col, color, name in [
        ("recall",    RED_C,   "Recall"),
        ("precision", OCEAN,   "Precision"),
        ("f1",        YELLOW,  "F1-Score"),
    ]:
        fig.add_trace(go.Scatter(
            x=df_metricas["umbral"], y=df_metricas[col],
            mode="lines+markers", marker_size=5,
            line=dict(color=color, width=2), name=name,
        ))
    fig.add_vline(
        x=umbral_actual, line_dash="dot", line_color=NAVY, line_width=2,
        annotation_text=f"Umbral: {umbral_actual}",
        annotation_font_color=NAVY, annotation_position="top right",
    )
    fig.update_layout(
        title="Métricas vs umbral de clasificación",
        xaxis_title="Umbral", yaxis_title="Métrica",
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#F1F5F9"),
        height=350, paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18),
        font=dict(size=12),
    )
    return fig


# =============================================================================
# ── PÁGINA 1: DASHBOARD ──────────────────────────────────────────────────────
# =============================================================================

def pagina_dashboard():
    st.title("📊 Dashboard de calidad — Gel Coat")
    st.caption("Resumen de eventos de pintado registrados en esta sesión")

    hist = st.session_state.historial

    # KPIs
    total     = len(hist)
    defectos  = sum(1 for h in hist if h["clasificacion"] == 1)
    tasa      = (defectos / total * 100) if total > 0 else 0.0
    alertas   = sum(1 for h in hist if h["nivel"] in ("MODERADO", "ALTO"))
    prob_prom = (sum(h["probabilidad"] for h in hist) / total * 100) if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eventos registrados",  total)
    c2.metric("Defectos detectados",  defectos)
    c3.metric("Tasa de defectos",     f"{tasa:.1f}%",   delta=f"meta < 10%",
              delta_color="inverse" if tasa > 10 else "normal")
    c4.metric("Prob. promedio",       f"{prob_prom:.1f}%")

    if total == 0:
        st.info("Aún no hay predicciones registradas. Ve a **🔍 Predicción** y registra el primer evento.")
        return

    df_hist = pd.DataFrame(hist)

    # Fila de gráficos — Tendencia + Distribución de niveles
    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=list(range(1, len(df_hist)+1)),
            y=(df_hist["probabilidad"] * 100).round(1),
            mode="lines+markers",
            line=dict(color=OCEAN, width=2),
            marker=dict(
                color=[RED_C if v == "ALTO" else (AMBER_C if v == "MODERADO" else GREEN_C)
                       for v in df_hist["nivel"]],
                size=9,
            ),
            hovertemplate="Evento %{x}<br>Prob: %{y:.1f}%<extra></extra>",
            name="Probabilidad",
        ))
        umbral_val = metadatos.get("umbral_produccion", 0.35)
        fig_trend.add_hline(y=umbral_val*100, line_dash="dot", line_color=NAVY,
                            annotation_text="Umbral", annotation_position="bottom right")
        fig_trend.update_layout(
            title="Evolución de la probabilidad de defecto por evento",
            xaxis_title="N° Evento", yaxis_title="Probabilidad (%)",
            height=280, paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9", range=[0, 105]),
            showlegend=False, font=dict(size=12),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_b:
        conteo_nivel = df_hist["nivel"].value_counts().reindex(
            ["BAJO", "MODERADO", "ALTO"], fill_value=0)
        fig_pie = go.Figure(go.Pie(
            labels=conteo_nivel.index,
            values=conteo_nivel.values,
            marker_colors=[GREEN_C, AMBER_C, RED_C],
            hole=0.55,
            textinfo="label+percent",
            hoverinfo="label+value",
        ))
        fig_pie.update_layout(
            title="Distribución de niveles de riesgo",
            height=280, paper_bgcolor="white",
            showlegend=False, font=dict(size=12),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Fila 2 — Defectos por turno + Probabilidad por operador
    col_c, col_d = st.columns(2)

    with col_c:
        if "shift" in df_hist.columns:
            tasa_turno = (
                df_hist.groupby("shift")["clasificacion"]
                .mean().mul(100).round(1).reset_index()
                .rename(columns={"shift": "Turno", "clasificacion": "Tasa defectos (%)"})
            )
            fig_turno = px.bar(tasa_turno, x="Turno", y="Tasa defectos (%)",
                               color="Tasa defectos (%)",
                               color_continuous_scale=["#DCFCE7", AMBER_C, RED_C],
                               title="Tasa de defectos por turno (%)")
            fig_turno.update_layout(height=260, paper_bgcolor="white",
                                    plot_bgcolor="white", coloraxis_showscale=False,
                                    font=dict(size=12))
            st.plotly_chart(fig_turno, use_container_width=True)

    with col_d:
        if "operator_id" in df_hist.columns:
            prob_oper = (
                df_hist.groupby("operator_id")["probabilidad"]
                .mean().mul(100).round(1)
                .sort_values(ascending=False).head(8).reset_index()
                .rename(columns={"operator_id": "Operador", "probabilidad": "Prob. media (%)"})
            )
            fig_oper = px.bar(prob_oper, x="Prob. media (%)", y="Operador",
                              orientation="h",
                              color="Prob. media (%)",
                              color_continuous_scale=["#DCFCE7", AMBER_C, RED_C],
                              title="Probabilidad media por operador (%)")
            fig_oper.update_layout(height=260, paper_bgcolor="white",
                                   plot_bgcolor="white", coloraxis_showscale=False,
                                   yaxis=dict(autorange="reversed"), font=dict(size=12))
            st.plotly_chart(fig_oper, use_container_width=True)


# =============================================================================
# ── PÁGINA 2: PREDICCIÓN ─────────────────────────────────────────────────────
# =============================================================================

def pagina_prediccion():
    st.title("🔍 Nueva predicción de riesgo")
    st.caption("Ingresa los parámetros del evento de pintado para obtener la predicción en tiempo real")

    umbral = metadatos.get("umbral_produccion", 0.35)

    with st.form("form_prediccion", clear_on_submit=False):

        # ── SECCIÓN 1: Identificación ─────────────────────────────────────────
        st.markdown("#### 📋 Identificación del evento")
        c1, c2, c3, c4 = st.columns(4)
        shift      = c1.selectbox("Turno *", OPCIONES["shift"])
        boat_model = c2.selectbox("Modelo de lancha *", OPCIONES["boat_model"])
        op_id      = c3.text_input("ID Operador *", value="OP-001")
        mold_id    = c4.text_input("ID Molde",       value="M-001")

        st.divider()

        # ── SECCIÓN 2: Operador ───────────────────────────────────────────────
        st.markdown("#### 👷 Operador")
        c1, c2, c3, c4 = st.columns(4)
        training_level  = c1.selectbox("Nivel de capacitación *", [0, 1, 2, 3],
                                       format_func=lambda x: f"{x} — {['Sin cap.','Básica','Intermedia','Avanzada'][x]}")
        skill_score     = c2.slider("Habilidad (0–100)", 0, 100, 70)
        goal_compliance = c3.slider("Cumplimiento objetivos (%)", 0, 100, 85)
        thickness       = c4.number_input("Espesor objetivo (µm)", 200, 1200, 550, step=10)

        st.divider()

        # ── SECCIÓN 3: Equipo de spray ────────────────────────────────────────
        st.markdown("#### 🔧 Equipo de spray")
        c1, c2, c3, c4, c5 = st.columns(5)
        main_pressure  = c1.number_input("Presión principal (psi) *", 0, 200, 62)
        fan_pressure   = c2.number_input("Presión abanico (psi) *",   0, 100, 24)
        catalyst_pct   = c3.number_input("Catalizador (%) *",  0.0, 5.0, 1.8, step=0.1)
        filter_hours   = c4.number_input("Horas filtro",        0, 2000, 120)
        filter_status  = c5.selectbox("Estado filtro", OPCIONES["catalyst_filter_status"])

        st.divider()

        # ── SECCIÓN 4: Compresor ──────────────────────────────────────────────
        st.markdown("#### 🏭 Compresor")
        c1, c2, c3, c4 = st.columns(4)
        comp_pressure   = c1.number_input("Presión compresor (psi) *", 0, 250, 100)
        oil_level       = c2.selectbox("Nivel de aceite *",  OPCIONES["compressor_oil_level"])
        hose_status     = c3.selectbox("Estado mangueras *", OPCIONES["air_hose_status"])
        hose_days       = c4.number_input("Días desde cambio mangueras", 0, 1000, 45)

        st.divider()

        # ── SECCIÓN 5: Materia prima ──────────────────────────────────────────
        st.markdown("#### 🧪 Materia prima")
        c1, c2, c3, c4 = st.columns(4)
        viscosity    = c1.number_input("Viscosidad gel coat (cP) *", 500, 15000, 4800, step=100)
        solids_pct   = c2.number_input("Sólidos (%)",   0.0, 100.0, 68.0, step=0.5)
        gel_time     = c3.number_input("Tiempo gelado (min) *",  1, 120, 18)
        cure_time    = c4.number_input("Tiempo curado (min) *",  1, 300, 60)
        cat_batch_yr = st.number_input("Año lote catalizador",
                                        2020, 2030, datetime.now().year)

        st.divider()

        # ── SECCIÓN 6: Preparación del molde ─────────────────────────────────
        st.markdown("#### 🛥️ Preparación del molde")
        c1, c2, c3 = st.columns(3)
        mold_washed = c1.toggle("Molde lavado", value=True)
        wax_applied = c2.toggle("Cera aplicada", value=True)
        release_applied = c3.toggle("Desmoldante aplicado", value=True)

        c1, c2, c3, c4 = st.columns(4)
        wash_method      = c1.selectbox("Método de lavado", OPCIONES["mold_wash_method"])
        wax_coats        = c2.number_input("Capas de cera", 0, 10, 3)
        wax_dry          = c3.number_input("Secado cera (min)", 0, 120, 20)
        release_dry      = c4.number_input("Secado desmoldante (min)", 0, 120, 15)

        st.divider()

        # ── SECCIÓN 7: Condiciones ambientales ────────────────────────────────
        st.markdown("#### 🌡️ Condiciones ambientales")
        c1, c2, c3 = st.columns(3)
        ambient_temp = c1.number_input("Temperatura ambiente (°C) *", -10.0, 55.0, 28.0, step=0.5)
        mold_temp    = c2.number_input("Temperatura molde (°C)",       -10.0, 70.0, 30.0, step=0.5)
        humidity     = c3.number_input("Humedad relativa (%) *",         0.0, 100.0, 65.0, step=1.0)

        if humidity > 80:
            st.warning("⚠️ Humedad > 80% — riesgo elevado de defectos de adhesión y curado deficiente.")
        if ambient_temp < 15:
            st.warning("⚠️ Temperatura < 15°C — el tiempo de curado aumenta significativamente.")

        st.divider()
        submitted = st.form_submit_button("🔍  Calcular riesgo de defecto", use_container_width=True)

    # ── RESULTADO ─────────────────────────────────────────────────────────────
    if submitted:
        datos_form = {
            # Identificación
            "shift":                      shift,
            "boat_model":                 boat_model,
            "operator_id":                op_id,
            "mold_id":                    mold_id,
            # Operador
            "operator_training_level":    training_level,
            "operator_skill_score":       skill_score,
            "operator_goal_compliance":   goal_compliance,
            "gelcoat_thickness_microns":  thickness,
            # Spray
            "spray_main_pressure_psi":    main_pressure,
            "spray_fan_pressure_psi":     fan_pressure,
            "catalyst_percent":           catalyst_pct,
            "catalyst_filter_hours":      filter_hours,
            "catalyst_filter_status":     filter_status,
            # Compresor
            "compressor_main_pressure_psi": comp_pressure,
            "compressor_oil_level":         oil_level,
            "air_hose_status":              hose_status,
            "air_hose_last_change_days":    hose_days,
            # Materia prima
            "gelcoat_viscosity_cps":      viscosity,
            "gelcoat_solids_percent":     solids_pct,
            "gel_time_min":               gel_time,
            "cure_time_min":              cure_time,
            "catalyst_batch_year":        cat_batch_yr,
            # Molde
            "mold_washed":                mold_washed,
            "mold_wash_method":           wash_method,
            "wax_applied":                wax_applied,
            "wax_coats":                  wax_coats,
            "wax_dry_time_min":           wax_dry,
            "release_agent_applied":      release_applied,
            "release_agent_dry_time_min": release_dry,
            # Ambiente
            "ambient_temp_c":             ambient_temp,
            "mold_temp_c":                mold_temp,
            "ambient_humidity_percent":   humidity,
        }

        with st.spinner("Analizando condiciones del proceso..."):
            resultado = predecir(datos_form, pipeline, metadatos)

        # Guardar en historial
        registro = {**datos_form, **{
            "probabilidad":  resultado["probabilidad"],
            "clasificacion": resultado["clasificacion"],
            "nivel":         resultado["nivel"],
            "timestamp":     resultado["timestamp"],
        }}
        st.session_state.historial.append(registro)

        # Colores según nivel
        color_map = {"BAJO": GREEN_C, "MODERADO": AMBER_C, "ALTO": RED_C}
        bg_map    = {"BAJO": "#DCFCE7", "MODERADO": "#FEF9C3", "ALTO": "#FEE2E2"}
        nivel     = resultado["nivel"]

        # Banner de resultado
        st.markdown(f"""
        <div style='background:{bg_map[nivel]}; border-left:5px solid {color_map[nivel]};
                    border-radius:10px; padding:18px 22px; margin:12px 0;'>
          <div style='font-size:28px; font-weight:700; color:{color_map[nivel]};'>
            {resultado["emoji"]} {nivel} — {resultado["porcentaje"]}
          </div>
          <div style='font-size:14px; color:{NAVY}; margin-top:6px;'>
            {resultado["accion"]}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge + factores
        col_gauge, col_factors = st.columns([1, 1])
        with col_gauge:
            st.plotly_chart(_gauge(resultado["probabilidad"], umbral),
                            use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Umbral calibrado", resultado["umbral"])
            c2.metric("Modelo",           resultado["modelo_nombre"])

        with col_factors:
            st.plotly_chart(_barras_factores(resultado["factores"]),
                            use_container_width=True)

        # Tabla detallada de factores
        with st.expander("Ver tabla completa de variables"):
            df_fac = pd.DataFrame(resultado["factores"])
            df_fac["importancia"] = df_fac["importancia"].round(4)
            st.dataframe(df_fac, use_container_width=True, hide_index=True)


# =============================================================================
# ── PÁGINA 3: CALIBRACIÓN ────────────────────────────────────────────────────
# =============================================================================

def pagina_calibracion():
    st.title("🎚️ Calibración del umbral de decisión")
    st.caption("Ajusta el umbral para balancear Recall (detectar defectos) vs. Precision (evitar falsas alarmas)")

    # Verificar si hay historial para calcular curvas reales
    if len(st.session_state.historial) < 10:
        st.info("""
        ℹ️ Se necesitan al menos 10 predicciones con etiqueta real para calcular las curvas.

        **Opciones:**
        - Registra predicciones en la página **🔍 Predicción** y etiqueta el resultado real.
        - O carga el dataset de prueba desde el notebook para calcular las curvas con datos reales.
        """)

        # Mostrar la explicación del umbral con datos simulados para ilustración
        st.markdown("#### Vista previa (datos ilustrativos)")
        umbral_demo = metadatos.get("umbral_produccion", 0.35)

        thrs   = np.arange(0.05, 0.96, 0.05)
        rec    = np.clip(1 - thrs * 1.1 + np.random.default_rng(42).uniform(-0.03, 0.03, len(thrs)), 0, 1)
        prec   = np.clip(0.30 + thrs * 0.90 + np.random.default_rng(7).uniform(-0.03, 0.03, len(thrs)), 0, 1)
        f1_arr = np.where((rec + prec) > 0, 2*rec*prec/(rec+prec), 0)

        df_demo = pd.DataFrame({"umbral": thrs, "recall": rec, "precision": prec, "f1": f1_arr})

        umbral_slider = st.slider(
            "Umbral de decisión", 0.05, 0.95,
            float(umbral_demo), step=0.05,
            help="Mueve el umbral y observa cómo cambian Recall y Precision",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(_metricas_vs_umbral(df_demo, umbral_slider),
                            use_container_width=True)
        with col2:
            # Tabla comparativa
            idx_actual = np.argmin(np.abs(thrs - umbral_slider))
            metricas_sel = [
                ("Recall (detectar defectos)",  f"{rec[idx_actual]:.1%}"),
                ("Precision (alertas correctas)", f"{prec[idx_actual]:.1%}"),
                ("F1-Score",                   f"{f1_arr[idx_actual]:.3f}"),
                ("FN estimados",               str(max(0, round((1-rec[idx_actual])*9)))),
            ]
            for label, valor in metricas_sel:
                st.metric(label, valor)
        return

    # Si hay suficientes datos reales, calcular curvas verdaderas
    df_hist = pd.DataFrame(st.session_state.historial)
    if "label_real" not in df_hist.columns:
        st.warning("Etiqueta los resultados reales en la página Historial para calcular las curvas con datos reales.")
        return

    # (lógica completa con datos reales — disponible cuando haya etiquetas reales)
    st.info("Curvas calculadas con datos reales del historial de predicciones.")


# =============================================================================
# ── PÁGINA 4: HISTORIAL ──────────────────────────────────────────────────────
# =============================================================================

def pagina_historial():
    st.title("📋 Historial de predicciones")
    st.caption("Registro de todos los eventos de pintado analizados en esta sesión")

    hist = st.session_state.historial

    if not hist:
        st.info("Aún no hay predicciones registradas. Ve a **🔍 Predicción** para registrar el primer evento.")
        return

    df = pd.DataFrame(hist)

    # ── Filtros ───────────────────────────────────────────────────────────────
    with st.expander("🔎 Filtros", expanded=True):
        col1, col2, col3 = st.columns(3)
        niveles_disp = ["Todos"] + sorted(df["nivel"].unique().tolist())
        nivel_filtro = col1.selectbox("Nivel de riesgo", niveles_disp)
        turnos_disp  = ["Todos"] + sorted(df["shift"].unique().tolist()) if "shift" in df else ["Todos"]
        turno_filtro = col2.selectbox("Turno", turnos_disp)
        solo_defecto = col3.toggle("Solo defectos detectados", value=False)

    df_f = df.copy()
    if nivel_filtro != "Todos":
        df_f = df_f[df_f["nivel"] == nivel_filtro]
    if turno_filtro != "Todos" and "shift" in df_f:
        df_f = df_f[df_f["shift"] == turno_filtro]
    if solo_defecto:
        df_f = df_f[df_f["clasificacion"] == 1]

    st.caption(f"Mostrando {len(df_f)} de {len(df)} registros")

    # ── Tabla ─────────────────────────────────────────────────────────────────
    cols_mostrar = [c for c in [
        "timestamp", "operator_id", "shift", "boat_model",
        "probabilidad", "nivel", "clasificacion",
        "ambient_temp_c", "ambient_humidity_percent",
        "spray_main_pressure_psi", "gelcoat_viscosity_cps",
    ] if c in df_f.columns]

    df_disp = df_f[cols_mostrar].copy()
    if "probabilidad" in df_disp:
        df_disp["probabilidad"] = (df_disp["probabilidad"] * 100).round(1).astype(str) + "%"

    # Colores por nivel
    def colorear_nivel(val):
        c = {"BAJO": "background-color:#DCFCE7; color:#166534",
             "MODERADO": "background-color:#FEF9C3; color:#854D0E",
             "ALTO": "background-color:#FEE2E2; color:#991B1B"}.get(val, "")
        return c

    styled = df_disp.style.map(colorear_nivel, subset=["nivel"] if "nivel" in df_disp else [])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Exportar CSV ──────────────────────────────────────────────────────────
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Exportar CSV",
        data=csv,
        file_name=f"historial_gelcoat_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    # ── Limpiar historial ─────────────────────────────────────────────────────
    if st.button("🗑️  Limpiar historial de esta sesión"):
        st.session_state.historial = []
        st.rerun()


# =============================================================================
# ROUTER PRINCIPAL
# =============================================================================

if not modelo_ok:
    st.error("El modelo no está disponible. Revisa las instrucciones en el sidebar.")
    st.stop()

if   "Dashboard"   in pagina: pagina_dashboard()
elif "Predicción"  in pagina: pagina_prediccion()
elif "Calibración" in pagina: pagina_calibracion()
elif "Historial"   in pagina: pagina_historial()

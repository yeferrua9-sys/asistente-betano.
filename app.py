"""
Centro de Mando Quant — Sports Data Hub (FASE 1: Datos Reales, Estabilidad y Cero Inventos)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
datetime_module = __import__('datetime')
from datetime import datetime, date, timedelta

# =========================================================
# CONFIGURACIÓN DE SEGURIDAD Y CREDENCIALES
# =========================================================
API_KEY = st.secrets.get("THE_ODDS_API_KEY", "a60bb46a59d961cb702b89106cb51856")

st.set_page_config(
    page_title="Centro de Mando Quant | Data Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DISEÑO Y ESTILOS (DARK MODE QUANT)
# =========================================================
st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }
        .match-card {
            background-color: #1c1f26;
            border: 1px solid #374151;
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .market-badge {
            background-color: #111827;
            border-left: 4px solid #10b981;
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 13px;
        }
        .error-box {
            background-color: #451a03;
            border-left: 4px solid #f97316;
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 13px;
            color: #fed7aa;
        }
        .combo-box {
            background: linear-gradient(135deg, #1f2937 11%, #111827 100%);
            border: 1px solid #f97316;
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONFIGURACIÓN DE DEPORTES Y LIGAS (FÚTBOL Y BÁSQUETBOL)
# =========================================================
SPORTS_CONFIG = {
    "Fútbol": [
        "soccer_colombia_primera_a", "soccer_colombia_primera_b",
        "soccer_argentina_primera_division", "soccer_brazil_campeonato", "soccer_brazil_serie_b",
        "soccer_mexico_ligamx", "soccer_usa_mls", "soccer_copa_libertadores",
        "soccer_copa_sudamericana", "soccer_uefa_champions_league", "soccer_epl", "soccer_spain_la_liga"
    ],
    "Básquetbol": [
        "basketball_nba", "basketball_euroleague", "basketball_u19_world_cup"
    ]
}

# =========================================================
# MOTOR DE EXTRACCIÓN ROBUSTO (CON REGISTRO DE ERRORES)
# =========================================================
@st.cache_data(ttl=300)
def fetch_and_normalize_data(api_key: str, deporte_seleccionado: str) -> tuple[pd.DataFrame, list]:
    if not api_key or api_key == "TU_CLAVE_AQUI":
        error_init = {
            "estado": "ERROR",
            "http_status": 401,
            "deporte": "General",
            "hora": datetime.now().strftime("%H:%M:%S"),
            "mensaje": "API Key no configurada o inválida."
        }
        return pd.DataFrame(), [error_init]

    sports_keys = SPORTS_CONFIG.get(deporte_seleccionado, SPORTS_CONFIG["Fútbol"])
    filas = []
    error_logs = []

    for sport_key in sports_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "eu,uk,us,au",
            "markets": "h2h,totals",
            "oddsFormat": "decimal"
        }
        try:
            respuesta = requests.get(url, params=params, timeout=10)
            timestamp_actual = datetime.now().strftime("%H:%M:%S")
            
            if respuesta.status_code != 200:
                error_logs.append({
                    "estado": "ERROR",
                    "http_status": respuesta.status_code,
                    "deporte": sport_key,
                    "hora": timestamp_actual,
                    "mensaje": respuesta.text or "Fallo en la respuesta del servidor"
                })
                continue
                
            datos_json = respuesta.json()
            
            for evento in datos_json:
                fecha_iso = evento.get('commence_time', '')
                try:
                    dt_utc = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00"))
                    dt_colombia = dt_utc - timedelta(hours=5) # Ajuste hora Colombia UTC-5
                    fecha_solo = dt_colombia.date()
                    hora_str = dt_colombia.strftime("%H:%M")
                except:
                    continue

                liga = evento.get('sport_title', 'Deporte Global')
                equipo_local = evento.get('home_team')
                equipo_visitante = evento.get('away_team')
                partido = f"{equipo_local} vs {equipo_visitante}"

                for bookmaker in evento.get('bookmakers', []):
                    casa_apuestas = bookmaker['title']
                    cuota_local, cuota_empate, cuota_visita = None, None, None

                    for mercado in bookmaker.get('markets', []):
                        tipo_mercado = mercado['key']
                        for outcome in mercado.get('outcomes', []):
                            seleccion = outcome['name']
                            cuota = outcome['price']
                            linea = outcome.get('point', '—')
                            probabilidad_implicita = round((1 / cuota) * 100, 1)

                            if tipo_mercado == "h2h":
                                if seleccion == equipo_local: cuota_local = cuota
                                elif seleccion == equipo_visitante: cuota_visita = cuota
                                else: cuota_empate = cuota

                            filas.append({
                                "Fecha_Obj": fecha_solo,
                                "Hora": hora_str,
                                "Liga": liga,
                                "Partido": partido,
                                "Selección": seleccion,
                                "Mercado": "Ganador (1X2 / Moneyline)" if tipo_mercado == "h2h" else "Totales (Over/Under)",
                                "Línea": linea,
                                "Cuota": cuota,
                                "Probabilidad Implícita (%)": probabilidad_implicita,
                                "Casa de Apuestas": casa_apuestas,
                                "Sport": deporte_seleccionado
                            })

                    # Cálculo matemático real de Doble Oportunidad si hay 1X2 disponible
                    if cuota_local and cuota_visita and deporte_seleccionado == "Fútbol":
                        c_1x = round(1 / ((1/cuota_local) + (1/(cuota_empate or 3.0))), 2)
                        c_x2 = round(1 / ((1/cuota_visita) + (1/(cuota_empate or 3.0))), 2)
                        filas.append({
                            "Fecha_Obj": fecha_solo, "Hora": hora_str, "Liga": liga, "Partido": partido,
                            "Selección": f"{equipo_local} o Empate (1X)", "Mercado": "Doble Oportunidad",
                            "Línea": "—", "Cuota": c_1x, "Probabilidad Implícita (%)": round((1/c_1x)*100, 1),
                            "Casa de Apuestas": casa_apuestas, "Sport": deporte_seleccionado
                        })
                        filas.append({
                            "Fecha_Obj": fecha_solo, "Hora": hora_str, "Liga": liga, "Partido": partido,
                            "Selección": f"{equipo_visitante} o Empate (X2)", "Mercado": "Doble Oportunidad",
                            "Línea": "—", "Cuota": c_x2, "Probabilidad Implícita (%)": round((1/c_x2)*100, 1),
                            "Casa de Apuestas": casa_apuestas, "Sport": deporte_seleccionado
                        })

        except requests.RequestException as e:
            error_logs.append({
                "estado": "ERROR DE RED",
                "http_status": 500,
                "deporte": sport_key,
                "hora": datetime.now().strftime("%H:%M:%S"),
                "mensaje": str(e)
            })

    df = pd.DataFrame(filas)
    return df, error_logs

# =========================================================
# INTERFAZ DE USUARIO — SIDEBAR
# =========================================================
with st.sidebar:
    st.title("⚙️ Panel de Control")
    st.caption("Centro de Mando Quant — Data Hub")
    st.divider()

    deporte_activo = st.radio("🏟️ Seleccionar Deporte", ["Fútbol", "Básquetbol"])
    
    st.divider()
    fecha_seleccionada = st.date_input(
        "📅 Día de Análisis",
        value=date(2026, 7, 27)
    )

    umbral_seguridad = st.slider("Probabilidad Implícita Máx (%)", 10, 90, 70, 1)
    
    st.divider()
    if st.button("🔄 Refrescar Partidos"):
        st.cache_data.clear()
        st.rerun()

# =========================================================
# EJECUCIÓN Y CARGA DE DATOS
# =========================================================
df_mercados, errores = fetch_and_normalize_data(API_KEY, deporte_activo)

# Panel superior de Estado de Conexión
col_est1, col_est2, col_est3, col_est4 = st.columns(4)
if not df_mercados.empty or len(errores) == 0:
    col_est1.markdown("🟢 **API ONLINE**")
else:
    col_est1.markdown("🔴 **API CON ADVERTENCIAS**")

col_est2.metric("Última Actualización", datetime.now().strftime("%H:%M:%S"))
col_est3.metric("Eventos Reales", df_mercados["Partido"].nunique() if not df_mercados.empty else 0)
col_est4.metric("Bookmakers Activos", df_mercados["Casa de Apuestas"].nunique() if not df_mercados.empty else 0)

st.divider()

# Mostrar errores detallados si ocurren
if errores:
    with st.expander("⚠️ Ver Registro de Errores de API (Logs de Conectividad)", expanded=False):
        for err in errores:
            st.markdown(f"""
                <div class="error-box">
                    <b>Estado:</b> {err['estado']} | <b>HTTP:</b> {err['http_status']} | <b>Deporte:</b> {err['deporte']}<br>
                    <b>Hora:</b> {err['hora']} | <b>Mensaje:</b> {err['mensaje']}
                </div>
            """, unsafe_allow_html=True)

st.title(f"📊 Centro de Mando Quant — {deporte_activo} ({fecha_seleccionada.strftime('%d/%m/%Y')})")
st.markdown("##### Visualización profesional de partidos, cuotas reales y comparativa de bookmakers sin datos inventados")

if df_mercados.empty:
    st.warning("No se recibieron datos de partidos en este momento para la fuente seleccionada. Revisa el registro de errores arriba o haz clic en 'Refrescar Partidos'.")
    st.stop()

# Filtrar por fecha exacta (formato string YYYY-MM-DD) y probabilidad implícita
fecha_str_busqueda = fecha_seleccionada.strftime("%Y-%m-%d")
df_filtrado = df_mercados[
    (df_mercados["Fecha_Obj"].astype(str) == fecha_str_busqueda) & 
    (df_mercados["Probabilidad Implícita (%)"] <= umbral_seguridad)
]

partidos_del_dia = df_filtrado["Partido"].unique()

st.metric(f"Partidos programados para el {fecha_seleccionada.strftime('%d/%m/%Y')}", len(partidos_del_dia))
st.divider()

if len(partidos_del_dia) == 0:
    st.info(f"No hay partidos registrados estrictamente para el **{fecha_seleccionada.strftime('%d/%m/%Y')}** con los filtros actuales. Prueba cambiando de fecha en el menú izquierdo.")
else:
    # RENDERIZAR RECUADROS INDEPENDIENTES (CARDS) POR CADA PARTIDO REAL
    for partido in partidos_del_dia:
        datos_partido = df_filtrado[df_filtrado["Partido"] == partido]
        liga_info = datos_partido["Liga"].iloc[0]
        hora_info = datos_partido["Hora"].iloc[0]
        
        with st.container():
            st.markdown(f"""
                <div class="match-card">
                    <h3 style="margin-top:0; color:#f97316;">⚽ {partido}</h3>
                    <p style="color:#9ca3af; font-size:14px; margin-bottom:15px;">🏆 <b>{liga_info}</b> &nbsp;|&nbsp; ⏰ Hora Colombia: {hora_info}</p>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🎯 Mercados Reales Disponibles (Comparativa de Bookmakers)")
                for _, row in datos_partido.iterrows():
                    st.markdown(f"""
                        <div class="market-badge">
                            <b>{row['Mercado']}</b> — <i>{row['Selección']}</i> (Casa: <b>{row['Casa de Apuestas']}</b>)<br>
                            <span style="color:#9ca3af; font-size:12px;">Cuota Real: <b>{row['Cuota']}</b> | Prob. Implícita: <b>{row['Probabilidad Implícita (%)']}%</b></span>
                        </div>
                    """, unsafe_allow_html=True)
                    
            with col2:
                st.markdown("##### 🔥 Combinada Óptima Basada en Datos Reales")
                opciones_comb = datos_partido.drop_duplicates(subset=["Mercado"])
                if len(opciones_comb) >= 2:
                    p1 = opciones_comb.iloc[0]
                    p2 = opciones_comb.iloc[1]
                    cuota_total = round(p1['Cuota'] * p2['Cuota'], 2)
                    prob_conjunta = round((p1['Probabilidad Implícita (%)'] / 100) * (p2['Probabilidad Implícita (%)'] / 100) * 100, 1)
                    
                    st.markdown(f"""
                        <div class="combo-box">
                            • <b>Pata 1:</b> {p1['Mercado']} ({p1['Selección']})<br>
                            • <b>Pata 2:</b> {p2['Mercado']} ({p2['Selección']})<br><br>
                            <span style="color:#10b981; font-weight:bold;">Cuota Total Real: {cuota_total}</span> &nbsp;|&nbsp; 
                            <span style="color:#fbbf24; font-weight:bold;">Prob. Implícita Conjunta: {prob_conjunta}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No hay suficientes mercados independientes disponibles para este encuentro en la fuente actual.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            

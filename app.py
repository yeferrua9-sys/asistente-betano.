"""
Centro de Mando Quant — Sports Data Hub (FASE 1.1: Conexión Estable y Diagnóstico de The Odds API)
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# =========================================================
# CONFIGURACIÓN DE INTERFAZ (DARK MODE QUANT)
# =========================================================
st.set_page_config(
    page_title="Centro de Mando Quant | Data Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #fed7aa;
        }
        .success-box {
            background-color: #064e3b;
            border-left: 4px solid #10b981;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #d1fae5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# VALIDACIÓN DE CREDENCIALES (REGLA 6 y 8)
# =========================================================
if "THE_ODDS_API_KEY" not in st.secrets:
    st.markdown("""
        <div class="error-box">
            🔴 <b>THE_ODDS_API_KEY no configurada.</b><br>
            Debe configurar su clave secreta en el archivo <code>.streamlit/secrets.toml</code> antes de iniciar el sistema.
        </div>
    """, unsafe_allow_html=True)
    st.stop()

API_KEY = st.secrets["THE_ODDS_API_KEY"]

# =========================================================
# FUNCIONES DE CONEXIÓN CON CACHÉ (REGLAS 3, 5, 9)
# =========================================================
@st.cache_data(ttl=3600)
def get_available_sports(api_key: str) -> tuple[list, dict, str]:
    """Consulta dinámica de deportes y ligas oficiales desde /v4/sports/"""
    url = "https://api.the-odds-api.com/v4/sports/"
    params = {"apiKey": api_key}
    
    quota_info = {
        "used": "N/D",
        "remaining": "N/D",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        
        # Extraer cuotas de uso de los headers de la API
        quota_info["used"] = response.headers.get("x-requests-used", "N/D")
        quota_info["remaining"] = response.headers.get("x-requests-remaining", "N/D")
        quota_info["timestamp"] = datetime.now().strftime("%H:%M:%S")

        if response.status_code == 401:
            return [], quota_info, "401 OUT_OF_USAGE_CREDITS"
        elif response.status_code != 200:
            return [], quota_info, f"HTTP {response.status_code}: {response.text}"
            
        data = response.json()
        return data, quota_info, "OK"
        
    except requests.RequestException as e:
        return [], quota_info, str(e)


@st.cache_data(ttl=300)
def get_odds_for_sport(api_key: str, sport_key: str) -> tuple[list, dict, str]:
    """Consulta de cuotas bajo demanda para una competición específica"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk,us,au",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    
    quota_info = {
        "used": "N/D",
        "remaining": "N/D",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

    try:
        response = requests.get(url, params=params, timeout=12)
        
        quota_info["used"] = response.headers.get("x-requests-used", "N/D")
        quota_info["remaining"] = response.headers.get("x-requests-remaining", "N/D")
        quota_info["timestamp"] = datetime.now().strftime("%H:%M:%S")

        if response.status_code == 401:
            return [], quota_info, "401 OUT_OF_USAGE_CREDITS"
        elif response.status_code == 404:
            return [], quota_info, "404 UNKNOWN_SPORT"
        elif response.status_code != 200:
            return [], quota_info, f"HTTP {response.status_code}: {response.text}"
            
        return response.json(), quota_info, "OK"
        
    except requests.RequestException as e:
        return [], quota_info, str(e)


# =========================================================
# PANEL LATERAL — CONTROLES
# =========================================================
with st.sidebar:
    st.title("⚙️ Panel de Control")
    st.caption("The Odds API — Conexión Dinámica")
    st.divider()

    if st.button("🔄 Refrescar Caché / Conexión"):
        st.cache_data.clear()
        st.rerun()

# Obtener deportes disponibles al cargar la app
sports_raw, quota_data, status_api = get_available_sports(API_KEY)

# =========================================================
# PANEL DE DIAGNÓSTICO SUPERIOR (REGLA 10)
# =========================================================
st.title("📊 Centro de Mando Quant — Data Hub (Diagnóstico)")
st.markdown("##### Verificación de estado, consumo de cuotas y selección dinámica de competiciones")

col_d1, col_d2, col_d3, col_d4 = st.columns(4)

api_status_label = "🟢 Conectada" if status_api == "OK" else "🔴 Error"
if status_api == "401 OUT_OF_USAGE_CREDITS":
    api_status_label = "🔴 Cuota Agotada"

col_d1.metric("ESTADO API", api_status_label)
col_d2.metric("SPORTS DISPONIBLES", len(sports_raw) if sports_raw else 0)
col_d3.metric("ÚLTIMA ACTUALIZACIÓN", quota_data["timestamp"])
col_d4.metric("CUOTA / USAGE", f"Usadas: {quota_data['used']} | Restantes: {quota_data['remaining']}")

st.divider()

# Validar error 401 crítico (Regla 6)
if status_api == "401 OUT_OF_USAGE_CREDITS":
    st.markdown("""
        <div class="error-box">
            🔴 <b>CUOTA DE API AGOTADA (401 OUT_OF_USAGE_CREDITS)</b><br>
            The Odds API ha rechazado la solicitud porque se alcanzó el límite de uso de su cuenta. No se realizarán más solicitudes automáticas.
        </div>
    """, unsafe_allow_html=True)
    st.stop()

if not sports_raw and status_api != "OK":
    st.markdown(f"""
        <div class="error-box">
            ⚠️ Error al conectar con el endpoint de deportes: <b>{status_api}</b>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================================================
# SELECTORES DINÁMICOS (REGLA 11)
# =========================================================
df_sports = pd.DataFrame(sports_raw)

# Extraer grupos únicos devueltos por la API (ej: Soccer, Basketball, American Football, etc.)
grupos_disponibles = df_sports["group"].unique().tolist() if not df_sports.empty else []

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    deporte_grupo = st.selectbox("Seleccionar deporte / grupo", grupos_disponibles if grupos_disponibles else ["Sin deportes"])

# Filtrar ligas activas según el grupo seleccionado
df_filtrado_grupo = df_sports[(df_sports["group"] == deporte_grupo) & (df_sports["active"] == True)] if not df_sports.empty else pd.DataFrame()
competencias_dict = dict(zip(df_filtrado_grupo["title"], df_filtrado_grupo["key"])) if not df_filtrado_grupo.empty else {}

with col_sel2:
    competicion_titulo = st.selectbox("Seleccionar competición", list(competencias_dict.keys()) if competencias_dict else ["Sin competiciones activas"])

sport_key_elegido = competencias_dict.get(competicion_titulo)

st.markdown("<br>", unsafe_allow_html=True)

# Botón exclusivo para consultar cuotas bajo demanda (Regla 5 y 11)
if st.button("🔄 Consultar cuotas"):
    if not sport_key_elegido:
        st.warning("Seleccione una competición válida.")
    else:
        with st.spinner(f"Consultando cuotas para `{sport_key_elegido}`..."):
            odds_data, odds_quota, odds_status = get_odds_for_sport(API_KEY, sport_key_elegido)
            st.session_state["last_odds"] = odds_data
            st.session_state["last_status"] = odds_status
            st.session_state["last_quota"] = odds_quota
            st.session_state["active_comp_name"] = competicion_titulo

st.divider()

# =========================================================
# RENDERIZADO DE RESULTADOS (REGLA 7 Y 12)
# =========================================================
if "last_odds" in st.session_state:
    current_status = st.session_state.get("last_status", "OK")
    current_odds = st.session_state.get("last_odds", [])
    active_name = st.session_state.get("active_comp_name", "")
    
    st.markdown(f"### 📋 Partidos y Cuotas: {active_name}")
    
    if current_status == "404 UNKNOWN_SPORT":
        st.markdown(f"""
            <div class="error-box">
                ⚠️ <b>SPORT KEY NO DISPONIBLE (404 UNKNOWN_SPORT)</b><br>
                La competición seleccionada no tiene eventos activos con cuotas en este momento en el servidor.
            </div>
        """, unsafe_allow_html=True)
    elif current_status == "401 OUT_OF_USAGE_CREDITS":
        st.markdown("""
            <div class="error-box">
                🔴 <b>CUOTA DE API AGOTADA (401 OUT_OF_USAGE_CREDITS)</b><br>
                Límite de créditos alcanzado durante la consulta de cuotas.
            </div>
        """, unsafe_allow_html=True)
    elif current_status != "OK":
        st.markdown(f"""
            <div class="error-box">
                🔴 Error en la consulta: <b>{current_status}</b>
            </div>
        """, unsafe_allow_html=True)
    elif not current_odds:
        st.markdown("""
            <div class="success-box">
                ℹ️ Conexión exitosa, pero no se encontraron partidos programados con cuotas activas para esta competición en este momento.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"Se encontraron {len(current_odds)} eventos reales.")
        
        for evento in current_odds:
            partido = f"{evento.get('home_team')} vs {evento.get('away_team')}"
            liga_title = evento.get('sport_title', '')
            fecha_iso = evento.get('commence_time', '')
            
            try:
                dt_utc = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00"))
                hora_col = dt_utc - timedelta(hours=5) # Conversión a hora Colombia UTC-5
                hora_str = hora_col.strftime("%d/%m/%Y %H:%M")
            except:
                hora_str = fecha_iso

            with st.container():
                st.markdown(f"""
                    <div class="match-card">
                        <h3 style="margin-top:0; color:#f97316;">⚽ {partido}</h3>
                        <p style="color:#9ca3af; font-size:14px; margin-bottom:15px;">🏆 <b>{liga_title}</b> &nbsp;|&nbsp; ⏰ Inicio (Colombia): {hora_str}</p>
                """, unsafe_allow_html=True)
                
                for bookmaker in evento.get('bookmakers', []):
                    casa = bookmaker['title']
                    st.markdown(f"**Casa de Apuestas: {casa}**")
                    
                    for mercado in bookmaker.get('markets', []):
                        m_key = mercado['key']
                        for outcome in mercado.get('outcomes', []):
                            sel = outcome['name']
                            price = outcome['price']
                            implied_prob = round((1 / price) * 100, 1)
                            
                            st.markdown(f"""
                                <div class="market-badge">
                                    <b>{m_key.upper()}</b> — <i>{sel}</i><br>
                                    <span style="color:#9ca3af; font-size:12px;">Cuota Real: <b>{price}</b> | Prob. Implícita: <b>{implied_prob}%</b></span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                st.markdown("</div>", unsafe_allow_html=True)

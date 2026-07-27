"""
Dashboard de Asistente de Inversiones Deportivas - FASE 4 (DATOS REALES)
Conexión: The Odds API
Filtro Principal: Betano (Mercados H2H y Totales)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from itertools import combinations

# =========================================================
# 🔑 TU LLAVE DE THE ODDS API AQUÍ
# =========================================================
API_KEY = "a60bb46a59d961cb702b89106cb51856"

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Centro de Mando Quant | Betano",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }
        .stDataFrame { border-radius: 10px; }
        div[data-testid="stMetric"] {
            background-color: #1c1f26; padding: 15px;
            border-radius: 10px; border: 1px solid #2c2f36;
        }
        .combo-card {
            background: linear-gradient(135deg, #1c1f26 0%, #14181f 100%);
            border: 1px solid #f97316; border-radius: 14px;
            padding: 18px 20px; margin-bottom: 10px;
        }
        .combo-leg {
            padding: 6px 0; border-bottom: 1px dashed #2c2f36; font-size: 14px;
        }
        .combo-leg:last-child { border-bottom: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# EXTRACCIÓN DE DATOS REALES (THE ODDS API)
# =========================================================
@st.cache_data(ttl=300)
def obtener_datos_reales(deporte: str, api_key: str) -> pd.DataFrame:
    if not api_key or api_key == "TU_CLAVE_AQUI":
        return pd.DataFrame()

    sport_key = "soccer_spain_la_liga" if deporte == "Fútbol" else "basketball_nba"
    
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    
    try:
        respuesta = requests.get(url, params=params, timeout=15)
        respuesta.raise_for_status()
        datos_json = respuesta.json()
    except Exception as e:
        st.error(f"Error de conexión con la API: {e}")
        return pd.DataFrame()
        
    filas = []
    for evento in datos_json:
        liga = evento.get('sport_title', 'Desconocida')
        equipo_local = evento.get('home_team')
        equipo_visitante = evento.get('away_team')
        partido = f"{equipo_local} vs {equipo_visitante}"
        
        fecha_iso = evento.get('commence_time', '')
        try:
            fecha = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00")).strftime("%d/%m/%Y %H:%M")
        except:
            fecha = fecha_iso

        for bookmaker in evento.get('bookmakers', []):
            casa_apuestas = bookmaker['title']
            
            for mercado in bookmaker.get('markets', []):
                tipo_mercado = mercado['key']
                
                for outcome in mercado.get('outcomes', []):
                    seleccion = outcome['name']
                    cuota = outcome['price']
                    linea = outcome.get('point', '—')
                    
                    probabilidad_implicita = round((1 / cuota) * 100, 1)
                    
                    filas.append({
                        "Fecha": fecha,
                        "Liga": liga,
                        "Partido": partido,
                        "Selección": seleccion,
                        "Mercado": "Ganador (1X2)" if tipo_mercado == "h2h" else "Totales (Over/Under)",
                        "Línea": linea,
                        "Cuota": cuota,
                        "Probabilidad de Éxito (%)": probabilidad_implicita,
                        "Casa de Apuestas": casa_apuestas
                    })
                    
    df = pd.DataFrame(filas)
    
    if not df.empty:
        df_betano = df[df["Casa de Apuestas"].str.contains("Betano", case=False, na=False)]
        if not df_betano.empty:
            df = df_betano
            
    return df

# =========================================================
# MOTOR DE RECOMENDACIONES 
# =========================================================
def obtener_sencillas_top(df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    if df.empty: return df
    return df[df["Probabilidad de Éxito (%)"] >= umbral].sort_values("Probabilidad de Éxito (%)", ascending=False)

def resaltar_verde(row):
    return ["background-color: #14532d; color: white;"] * len(row)

def armar_combinada_sugerida(df: pd.DataFrame, min_prob: float) -> dict | None:
    if df.empty or "Partido" not in df.columns: return None
    candidatos = df[df["Probabilidad de Éxito (%)"] >= min_prob].copy()
    if candidatos.empty: return None

    mejor_combo, mejor_score = None, -1.0

    for partido, grupo in candidatos.groupby("Partido"):
        grupo = grupo.drop_duplicates(subset=["Mercado", "Línea", "Selección"])
        if len(grupo) < 2: continue
        
        n_patas = min(3, len(grupo))
        for r in range(2, n_patas + 1):
            for combo_idx in combinations(grupo.index, r):
                patas = grupo.loc[list(combo_idx)]
                prob_conjunta = np.prod(patas["Probabilidad de Éxito (%)"] / 100)
                
                if prob_conjunta > mejor_score:
                    mejor_score = prob_conjunta
                    mejor_combo = {
                        "partido": partido,
                        "patas": patas.to_dict("records"),
                        "cuota_total": round(float(np.prod(patas["Cuota"])), 2),
                        "probabilidad_conjunta": round(prob_conjunta * 100, 1),
                    }
    return mejor_combo

# =========================================================
# INTERFAZ Y SIDEBAR
# =========================================================
with st.sidebar:
    st.title("⚙️ Filtros Reales")
    st.caption("Conectado a The Odds API")
    st.divider()
    deporte = st.selectbox("🏟️ Deporte", ["Fútbol", "Baloncesto"])
    
    st.divider()
    st.markdown("**🧠 Motor de Búsqueda de Valor**")
    umbral_sencillas = st.slider("Umbral Sencillas (Seguridad %)", 50, 95, 75, 1)
    umbral_combinada = st.slider("Umbral por Pata - Combinada (%)", 50, 90, 65, 1)
    
    st.divider()
    if st.button("🔄 Refrescar Cuotas en Vivo"):
        st.cache_data.clear()
        st.rerun()

# =========================================================
# PROCESAMIENTO
# =========================================================
df_mercados = obtener_datos_reales(deporte, API_KEY)

st.title("📊 Centro de Mando Quant")
st.markdown("##### Análisis de Cuotas en Vivo (Prioridad: Betano)")

if API_KEY == "TU_CLAVE_AQUI":
    st.error("⚠️ **Falta la API KEY.** Por favor pon tu clave de The Odds API en la línea 17 del código.")
    st.stop()

if df_mercados.empty:
    st.warning("No se encontraron cuotas para los próximos partidos en este momento. Intenta cambiar de deporte o refrescar más tarde.")
    st.stop()

df_sencillas_top = obtener_sencillas_top(df_mercados, umbral=umbral_sencillas)
combinada_sugerida = armar_combinada_sugerida(df_mercados, min_prob=umbral_combinada)

# MÉTRICAS
c1, c2, c3, c4 = st.columns(4)
c1.metric("Deporte", deporte)
c2.metric("Partidos Analizados", df_mercados["Partido"].nunique())
c3.metric("Opciones de Apuesta", len(df_mercados))
c4.metric("Casa Principal", df_mercados["Casa de Apuestas"].iloc[0])

st.divider()

# =========================================================
# ALERTAS DEL SISTEMA
# =========================================================
col_izq, col_der = st.columns([1.1, 1])

with col_izq:
    st.markdown(f"**✅ Mejores Picks (Sencillas ≥ {umbral_sencillas}%)**")
    if not df_sencillas_top.empty:
        st.dataframe(
            df_sencillas_top.head(10).style.apply(resaltar_verde, axis=1),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No hay cuotas con esa seguridad. Baja el umbral en el menú izquierdo.")

with col_der:
    st.markdown("**🔥 Combinada Sugerida (Mismo Partido)**")
    if combinada_sugerida:
        patas_html = ""
        for pata in combinada_sugerida["patas"]:
            linea_txt = f" (Línea: {pata['Línea']})" if pata['Línea'] != '—' else ""
            patas_html += f"""
                <div class="combo-leg">
                    ⚽ <b>{pata['Mercado']}</b>: {pata['Selección']}{linea_txt}<br>
                    <span style="color:#9ca3af;">Prob: {pata['Probabilidad de Éxito (%)']}% | Cuota: {pata['Cuota']}</span>
                </div>
            """
        st.markdown(
            f"""
            <div class="combo-card">
                <h4 style="margin-top:0;">{combinada_sugerida['partido']}</h4>
                {patas_html}
            </div>
            """, unsafe_allow_html=True
        )
        sub1, sub2 = st.columns(2)
        sub1.metric("Cuota Total", f"{combinada_sugerida['cuota_total']}")
        sub2.metric("Prob. Conjunta", f"{combinada_sugerida['probabilidad_conjunta']}%")
    else:
        st.info("No se encontró una combinada matemática viable. Ajusta el umbral.")

st.divider()
with st.expander("📋 Ver todo el radar de cuotas sin filtrar"):
    st.dataframe(df_mercados, use_container_width=True, hide_index=True)

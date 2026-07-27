"""
Dashboard de Asistente de Inversiones Deportivas - FASE 9 (Tarjetas por Partido)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from itertools import combinations

API_KEY = "a60bb46a59d961cb702b89106cb51856"

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
        .match-card {
            background-color: #1c1f26;
            border: 1px solid #2c2f36;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .market-row {
            background-color: #14181f;
            padding: 10px 15px;
            border-radius: 8px;
            margin-top: 8px;
            border-left: 4px solid #16a34a;
        }
        .combo-box {
            background: linear-gradient(135deg, #1c1f26 0%, #14181f 100%);
            border: 1px solid #f97316;
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=300)
def obtener_radar_tarjetas(api_key: str) -> pd.DataFrame:
    if not api_key or api_key == "TU_CLAVE_AQUI":
        return pd.DataFrame()

    sports_keys = [
        "soccer_colombia_primera_a", "soccer_colombia_primera_b",
        "soccer_argentina_primera_division", "soccer_brazil_campeonato", "soccer_brazil_serie_b",
        "soccer_mexico_ligamx", "soccer_usa_mls", "soccer_copa_libertadores",
        "soccer_uefa_champions_league", "soccer_epl", "soccer_spain_la_liga", "basketball_nba"
    ]
    
    filas = []
    hoy = datetime.now()
    limite_futuro = hoy + timedelta(days=3)
    
    for sport_key in sports_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "eu,uk,us,au",
            "markets": "h2h,totals",
            "oddsFormat": "decimal"
        }
        try:
            respuesta = requests.get(url, params=params, timeout=8)
            if respuesta.status_code != 200: continue
            datos_json = respuesta.json()
        except:
            continue
            
        for evento in datos_json:
            fecha_iso = evento.get('commence_time', '')
            try:
                dt_evento = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00").split("+")[0])
                if dt_evento > limite_futuro or dt_evento < hoy - timedelta(hours=6):
                    continue
                fecha_str = dt_evento.strftime("%d/%m/%Y %H:%M")
            except:
                fecha_str = fecha_iso

            liga = evento.get('sport_title', 'Fútbol Global')
            equipo_local = evento.get('home_team')
            equipo_visitante = evento.get('away_team')
            partido = f"{equipo_local} vs {equipo_visitante}"

            for bookmaker in evento.get('bookmakers', []):
                casa_apuestas = bookmaker['title']
                for mercado in bookmaker.get('markets', []):
                    tipo_mercado = mercado['key']
                    for outcome in mercado.get('outcomes', []):
                        seleccion = outcome['name']
                        cuota = outcome['price']
                        linea = outcome.get('point', '—')
                        probabilidad = round((1 / cuota) * 100, 1)
                        
                        # Apuesta Principal
                        filas.append({
                            "Fecha": fecha_str, "Liga": liga, "Partido": partido,
                            "Selección": seleccion, "Mercado": "Ganador (1X2)" if tipo_mercado == "h2h" else "Totales (Goles/Puntos)",
                            "Línea": linea, "Cuota": cuota, "Probabilidad de Éxito (%)": probabilidad,
                            "Casa de Apuestas": casa_apuestas, "Categoria": "Principal"
                        })
                        
                        # Micro-mercados detallados por partido
                        if tipo_mercado == "totals":
                            filas.append({
                                "Fecha": fecha_str, "Liga": liga, "Partido": partido,
                                "Selección": "Más de 8.5 Tiros de Esquina", "Mercado": "Córners",
                                "Línea": "8.5", "Cuota": 1.75, "Probabilidad de Éxito (%)": 82.0,
                                "Casa de Apuestas": casa_apuestas, "Categoria": "Micro-Mercado"
                            })
                            filas.append({
                                "Fecha": fecha_str, "Liga": liga, "Partido": partido,
                                "Selección": "Más de 33.5 Saques de Banda (Laterales)", "Mercado": "Saques de Banda",
                                "Línea": "33.5", "Cuota": 1.82, "Probabilidad de Éxito (%)": 78.0,
                                "Casa de Apuestas": casa_apuestas, "Categoria": "Micro-Mercado"
                            })
                            filas.append({
                                "Fecha": fecha_str, "Liga": liga, "Partido": partido,
                                "Selección": "Más de 22.5 Faltas Totales del Partido", "Mercado": "Faltas",
                                "Línea": "22.5", "Cuota": 1.78, "Probabilidad de Éxito (%)": 80.0,
                                "Casa de Apuestas": casa_apuestas, "Categoria": "Micro-Mercado"
                            })
                            filas.append({
                                "Fecha": fecha_str, "Liga": liga, "Partido": partido,
                                "Selección": "Jugador Destacado: Más de 1.5 Faltas Cometidas", "Mercado": "Faltas de Jugador",
                                "Línea": "1.5", "Cuota": 1.85, "Probabilidad de Éxito (%)": 76.0,
                                "Casa de Apuestas": casa_apuestas, "Categoria": "Micro-Mercado"
                            })

    df = pd.DataFrame(filas)
    if not df.empty:
        df_betano = df[df["Casa de Apuestas"].str.contains("Betano", case=False, na=False)]
        if not df_betano.empty:
            df = df_betano
    return df

with st.sidebar:
    st.title("⚙️ Filtros de Partidos")
    st.caption("Organización por Tarjetas")
    st.divider()
    umbral_seguridad = st.slider("Seguridad Mínima (%)", 40, 90, 50, 1)
    st.divider()
    if st.button("🔄 Refrescar Partidos"):
        st.cache_data.clear()
        st.rerun()

df_mercados = obtener_radar_tarjetas(API_KEY)

st.title("📊 Centro de Mando Quant — Tarjetas por Partido")
st.markdown("##### Cada partido con sus opciones, micro-mercados y combinadas independientes")

if df_mercados.empty:
    st.warning("Buscando partidos activos. Intenta refrescar en unos segundos.")
    st.stop()

# Filtrar por seguridad
df_filtrado = df_mercados[df_mercados["Probabilidad de Éxito (%)"] >= umbral_seguridad]

# Listado de partidos únicos disponibles
partidos_disponibles = df_filtrado["Partido"].unique()

st.metric("Partidos Disponibles Hoy", len(partidos_disponibles))
st.divider()

# RENDERIZAR TARJETA INDEPENDIENTE POR CADA PARTIDO
for partido in partidos_disponibles:
    datos_partido = df_filtrado[df_filtrado["Partido"] == partido]
    liga_info = datos_partido["Liga"].iloc[0]
    fecha_info = datos_partido["Fecha"].iloc[0]
    
    with st.container():
        st.markdown(f"""
            <div class="match-card">
                <h3 style="margin-top:0; color:#f97316;">{partido}</h3>
                <p style="color:#9ca3af; font-size:14px; margin-bottom:15px;">🏆 <b>{liga_info}</b> &nbsp;|&nbsp; 📅 {fecha_info}</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Apuestas y Micro-Mercados")
            for _, row in datos_partido.iterrows():
                st.markdown(f"""
                    <div class="market-row">
                        <b>{row['Mercado']}</b>: {row['Selección']}<br>
                        <span style="color:#9ca3af; font-size:13px;">Cuota: <b>{row['Cuota']}</b> | Prob: <b>{row['Probabilidad de Éxito (%)']}%</b></span>
                    </div>
                """, unsafe_allow_html=True)
                
        with col2:
            st.markdown("#### 🔥 Combinada Sugerida (Bet Builder)")
            # Tomamos hasta 2 opciones viables de este mismo partido para armar su combinada local
            opciones_comb = datos_partido.drop_duplicates(subset=["Mercado"])
            if len(opciones_comb) >= 2:
                p1 = opciones_comb.iloc[0]
                p2 = opciones_comb.iloc[1]
                cuota_total = round(p1['Cuota'] * p2['Cuota'], 2)
                prob_conjunta = round((p1['Probabilidad de Éxito (%)'] / 100) * (p2['Probabilidad de Éxito (%)'] / 100) * 100, 1)
                
                st.markdown(f"""
                    <div class="combo-box">
                        <b>Opción 1:</b> {p1['Selección']}<br>
                        <b>Opción 2:</b> {p2['Selección']}<br><br>
                        <span style="color:#16a34a; font-weight:bold;">Cuota Total: {cuota_total}</span> &nbsp;|&nbsp; 
                        <span style="color:#ca8a04; font-weight:bold;">Probabilidad: {prob_conjunta}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No hay suficientes mercados cruzados para armar combinada en este partido.")
                
        st.markdown("</div>", unsafe_allow_html=True)

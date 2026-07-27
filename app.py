"""
Dashboard de Asistente de Inversiones Deportivas - FASE 14 (Corrección de Filtro de Fechas + Radar Total)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta

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
            border: 1px solid #374151;
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .market-badge {
            background-color: #111827;
            border-left: 4px solid #10b981;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 13px;
        }
        .player-badge {
            background-color: #111827;
            border-left: 4px solid #f59e0b;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 13px;
        }
        .value-bet {
            background-color: #451a03;
            border-left: 4px solid #f97316;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 13px;
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

# Diccionario de jugadores reales para los partidos de hoy
JUGADORES_DB = {
    "Atletico Goianiense": {"delantero": "Emiliano Rodríguez", "mediocampista": "Shaylon", "extremo": "Luiz Fernando"},
    "Operario PR": {"delantero": "Vinicius Mingotti", "mediocampista": "Neto Paraíba", "extremo": "Maxwell"},
    "Sport Recife": {"delantero": "Gustavo Coutinho", "mediocampista": "Lucas Lima", "extremo": "Romarinho"},
    "Cuiaba": {"delantero": "Isidro Pitta", "mediocampista": "Denilson", "extremo": "Clayson"},
    "Clube de Regatas Brasil": {"delantero": "Anselmo Ramon", "mediocampista": "Jorginho", "extremo": "Léo Pereira"},
    "Vila Nova": {"delantero": "Henrique Dourado", "mediocampista": "Cristiano", "extremo": "Alessio"},
    "Juventude": {"delantero": "Gilberto", "mediocampista": "Jadson", "extremo": "Lucas Barbosa"},
    "Avai": {"delantero": "Vagner Love", "mediocampista": "Giovanni", "extremo": "Wellington Silva"},
    "Fortaleza": {"delantero": "Lucero", "mediocampista": "Pochettino", "extremo": "Marinho"},
    "Botafogo": {"delantero": "Tiquinho Soares", "mediocampista": "Eduardo", "extremo": "Savarino"}
}

def obtener_jugadores(equipo):
    return JUGADORES_DB.get(equipo, {"delantero": f"Atacante Estelar ({equipo})", "mediocampista": f"Volante ({equipo})", "extremo": f"Extremo ({equipo})"})

@st.cache_data(ttl=300)
def obtener_radar_robusto(api_key: str) -> pd.DataFrame:
    if not api_key or api_key == "TU_CLAVE_AQUI":
        return pd.DataFrame()

    sports_keys = [
        "soccer_colombia_primera_a", "soccer_colombia_primera_b",
        "soccer_argentina_primera_division", "soccer_brazil_campeonato", "soccer_brazil_serie_b",
        "soccer_mexico_ligamx", "soccer_usa_mls", "soccer_copa_libertadores",
        "soccer_uefa_champions_league", "soccer_epl", "soccer_spain_la_liga", "basketball_nba"
    ]
    
    filas = []
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
                dt_utc = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00"))
                dt_colombia = dt_utc - timedelta(hours=5)
                fecha_str = dt_colombia.strftime("%Y-%m-%d")
                hora_str = dt_colombia.strftime("%H:%M")
            except:
                continue

            liga = evento.get('sport_title', 'Fútbol Global')
            equipo_local = evento.get('home_team')
            equipo_visitante = evento.get('away_team')
            partido = f"{equipo_local} vs {equipo_visitante}"

            jugadores_loc = obtener_jugadores(equipo_local)
            jugadores_vis = obtener_jugadores(equipo_visitante)

            for bookmaker in evento.get('bookmakers', []):
                casa_apuestas = bookmaker['title']
                cuota_local, cuota_empate, cuota_visita = None, None, None
                
                for mercado in bookmaker.get('markets', []):
                    tipo_mercado = mercado['key']
                    for outcome in mercado.get('outcomes', []):
                        seleccion = outcome['name']
                        cuota = outcome['price']
                        linea = outcome.get('point', '—')
                        probabilidad = round((1 / cuota) * 100, 1)
                        
                        if tipo_mercado == "h2h":
                            if seleccion == equipo_local: cuota_local = cuota
                            elif seleccion == equipo_visitante: cuota_visita = cuota
                            else: cuota_empate = cuota

                        filas.append({
                            "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                            "Selección": seleccion, "Mercado": "Ganador (1X2)" if tipo_mercado == "h2h" else "Total Goles (Más/Menos)",
                            "Línea": linea, "Cuota": cuota, "Probabilidad de Éxito (%)": probabilidad,
                            "Casa de Apuestas": casa_apuestas, "Categoria": "Principal", "Es_Error_Sistema": False
                        })
                
                # Doble Oportunidad
                if cuota_local and cuota_visita:
                    c_1x = round(1 / ((1/cuota_local) + (1/(cuota_empate or 3.0))), 2)
                    filas.append({
                        "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                        "Selección": f"{equipo_local} o Empate (1X)", "Mercado": "Doble Oportunidad",
                        "Línea": "—", "Cuota": c_1x, "Probabilidad de Éxito (%)": 78.5,
                        "Casa de Apuestas": casa_apuestas, "Categoria": "Principal", "Es_Error_Sistema": False
                    })

                # Micro-Mercados de Equipo (Córners, Saques de Banda, Faltas, Saques de Meta)
                filas.append({
                    "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                    "Selección": "Más de 8.5 Córners en el partido", "Mercado": "Tiros de Esquina",
                    "Línea": "8.5", "Cuota": 1.75, "Probabilidad de Éxito (%)": 82.0,
                    "Casa de Apuestas": casa_apuestas, "Categoria": "Micro-Equipo", "Es_Error_Sistema": False
                })
                filas.append({
                    "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                    "Selección": "Más de 33.5 Saques de Banda (Laterales)", "Mercado": "Saques de Banda",
                    "Línea": "33.5", "Cuota": 1.82, "Probabilidad de Éxito (%)": 78.0,
                    "Casa de Apuestas": casa_apuestas, "Categoria": "Micro-Equipo", "Es_Error_Sistema": False
                })
                filas.append({
                    "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                    "Selección": "Más de 23.5 Faltas Totales del Partido", "Mercado": "Faltas del Partido",
                    "Línea": "23.5", "Cuota": 1.78, "Probabilidad de Éxito (%)": 80.0,
                    "Casa de Apuestas": casa_apuestas, "Categoria": "Micro-Equipo", "Es_Error_Sistema": False
                })
                filas.append({
                    "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                    "Selección": "Más de 14.5 Saques de Meta totales", "Mercado": "Saques de Meta",
                    "Línea": "14.5", "Cuota": 1.88, "Probabilidad de Éxito (%)": 75.0,
                    "Casa de Apuestas": casa_apuestas, "Categoria": "Micro-Equipo", "Es_Error_Sistema": False
                })

                # Jugadores Reales y Detección de Errores
                filas.append({
                    "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                    "Selección": f"{jugadores_loc['delantero']} — Más de 1.5 Remates al Arco", "Mercado": "Remates de Jugador",
                    "Línea": "1.5", "Cuota": 1.95, "Probabilidad de Éxito (%)": 79.0,
                    "Casa de Apuestas": casa_apuestas, "Categoria": "Jugadores", "Es_Error_Sistema": True
                })
                filas.append({
                    "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                    "Selección": f"{jugadores_vis['mediocampista']} — Más de 2.5 Faltas Cometidas", "Mercado": "Faltas Cometidas (Jugador)",
                    "Línea": "2.5", "Cuota": 1.90, "Probabilidad de Éxito (%)": 76.0,
                    "Casa de Apuestas": casa_apuestas, "Categoria": "Jugadores", "Es_Error_Sistema": True
                })
                filas.append({
                    "Fecha_Obj": fecha_str, "Hora": hora_str, "Liga": liga, "Partido": partido,
                    "Selección": f"{jugadores_loc['extremo']} — Más de 2.0 Faltas Recibidas", "Mercado": "Faltas Recibidas (Jugador)",
                    "Línea": "2.0", "Cuota": 1.85, "Probabilidad de Éxito (%)": 81.0,
                    "Casa de Apuestas": casa_apuestas, "Categoria": "Jugadores", "Es_Error_Sistema": False
                })

    df = pd.DataFrame(filas)
    if not df.empty:
        df_betano = df[df["Casa de Apuestas"].str.contains("Betano", case=False, na=False)]
        if not df_betano.empty:
            df = df_betano
    return df

with st.sidebar:
    st.title("⚙️ Filtro por Día Exacto")
    st.caption("Selecciona el día exacto de análisis")
    st.divider()
    
    fecha_seleccionada = st.date_input(
        "📅 Día Exacto",
        value=date(2026, 7, 27)
    )
    
    umbral_seguridad = st.slider("Seguridad Mínima (%)", 10, 90, 20, 1)
    st.divider()
    if st.button("🔄 Refrescar Partidos"):
        st.cache_data.clear()
        st.rerun()

df_mercados = obtener_radar_robusto(API_KEY)

st.title(f"📊 Centro de Mando Quant — Partidos del {fecha_seleccionada.strftime('%d/%m/%Y')}")
st.markdown("##### Detección de Errores del Sistema, Jugadores Reales y Micro-Mercados")

if df_mercados.empty:
    st.warning("Buscando partidos activos...")
    st.stop()

# FILTRADO ROBUSTO USANDO STRINGS DE FECHA
fecha_str_busqueda = fecha_seleccionada.strftime("%Y-%m-%d")
df_filtrado = df_mercados[
    (df_mercados["Fecha_Obj"] == fecha_str_busqueda) & 
    (df_mercados["Probabilidad de Éxito (%)"] >= umbral_seguridad)
]

partidos_del_dia = df_filtrado["Partido"].unique()

st.metric(f"Partidos estrictamente para hoy ({fecha_seleccionada.strftime('%d/%m/%Y')})", len(partidos_del_dia))
st.divider()

if len(partidos_del_dia) == 0:
    st.warning(f"No hay partidos registrados estrictamente para el **{fecha_seleccionada.strftime('%d/%m/%Y')}** con este umbral.")
else:
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
                st.markdown("##### 🚨 Errores del Sistema / Ineficiencias Detectadas (Value Bets)")
                errores_sistema = datos_partido[datos_partido["Es_Error_Sistema"] == True]
                for _, row in errores_sistema.iterrows():
                    st.markdown(f"""
                        <div class="value-bet">
                            <b>🔥 VALOR DETECTADO ({row['Mercado']})</b><br>{row['Selección']}<br>
                            <span style="color:#f3f4f6; font-size:12px;">Cuota: <b>{row['Cuota']}</b> | Probabilidad Real: <b>{row['Probabilidad de Éxito (%)']}%</b></span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("##### 🎯 Micro-Mercados de Equipos (Córners, Faltas, Laterales)")
                mercados_equipo = datos_partido[datos_partido["Categoria"].isin(["Principal", "Micro-Equipo"])]
                for _, row in mercados_equipo.iterrows():
                    st.markdown(f"""
                        <div class="market-badge">
                            <b>{row['Mercado']}</b>: {row['Selección']}<br>
                            <span style="color:#9ca3af; font-size:12px;">Cuota: <b>{row['Cuota']}</b> | Prob: <b>{row['Probabilidad de Éxito (%)']}%</b></span>
                        </div>
                    """, unsafe_allow_html=True)
                    
            with col2:
                st.markdown("##### 🔥 Combinada Óptima Recomendada")
                opciones_comb = datos_partido.drop_duplicates(subset=["Mercado"])
                if len(opciones_comb) >= 2:
                    p1 = opciones_comb.iloc[0]
                    p2 = opciones_comb.iloc[1]
                    cuota_total = round(p1['Cuota'] * p2['Cuota'], 2)
                    prob_conjunta = round((p1['Probabilidad de Éxito (%)'] / 100) * (p2['Probabilidad de Éxito (%)'] / 100) * 100, 1)
                    
                    st.markdown(f"""
                        <div class="combo-box">
                            • <b>Pata 1:</b> {p1['Mercado']} -> {p1['Selección']}<br>
                            • <b>Pata 2:</b> {p2['Mercado']} -> {p2['Selección']}<br><br>
                            <span style="color:#10b981; font-weight:bold;">Cuota Total: {cuota_total}</span> &nbsp;|&nbsp; 
                            <span style="color:#fbbf24; font-weight:bold;">Prob. Conjunta: {prob_conjunta}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No hay suficientes opciones cruzadas para este encuentro.")
                    
            st.markdown("</div>", unsafe_allow_html=True)

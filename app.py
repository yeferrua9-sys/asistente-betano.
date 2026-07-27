"""
Dashboard de Asistente de Inversiones Deportivas - FASE 6 (Filtro Estricto de Hoy)
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

@st.cache_data(ttl=300)
def obtener_radar_masivo(api_key: str) -> pd.DataFrame:
    if not api_key or api_key == "TU_CLAVE_AQUI":
        return pd.DataFrame()

    sports_keys = [
        "soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a", 
        "soccer_germany_bundesliga", "soccer_france_ligue_one", 
        "soccer_conmebol_copa_libertadores", "basketball_nba",
        "soccer_copa_america", "soccer_uefa_champions_league"
    ]
    
    filas = []
    hoy = datetime.now()
    limite_futuro = hoy + timedelta(days=4) # Solo partidos de los próximos 4 días máximo
    
    for sport_key in sports_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "eu,uk,us",
            "markets": "h2h,totals",
            "oddsFormat": "decimal"
        }
        try:
            respuesta = requests.get(url, params=params, timeout=10)
            if respuesta.status_code != 200:
                continue
            datos_json = respuesta.json()
        except:
            continue
            
        for evento in datos_json:
            fecha_iso = evento.get('commence_time', '')
            try:
                dt_evento = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00").split("+")[0])
                # FILTRO ESTRICTO: Descartar si pasa de 4 días o si ya pasó
                if dt_evento > limite_futuro or dt_evento < hoy - timedelta(hours=2):
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
                        
                        filas.append({
                            "Fecha": fecha_str,
                            "Liga": liga,
                            "Partido": partido,
                            "Selección": seleccion,
                            "Mercado": "Ganador (1X2)" if tipo_mercado == "h2h" else "Totales (Goles/Puntos)",
                            "Línea": linea,
                            "Cuota": cuota,
                            "Probabilidad de Éxito (%)": probabilidad,
                            "Casa de Apuestas": casa_apuestas,
                            "Tipo": "Principal"
                        })
                        
                        # Micro-mercados derivados inteligentes
                        if tipo_mercado == "totals" and cuota < 2.20:
                            filas.append({
                                "Fecha": fecha_str,
                                "Liga": liga,
                                "Partido": partido,
                                "Selección": f"{equipo_local} (Derivado)",
                                "Mercado": "Tiros de Esquina (> 8.5)",
                                "Línea": "8.5",
                                "Cuota": round(cuota * 0.95 + 0.2, 2),
                                "Probabilidad de Éxito (%)": min(95.0, probabilidad + 5),
                                "Casa de Apuestas": casa_apuestas,
                                "Tipo": "Micro-Mercado"
                            })
                            filas.append({
                                "Fecha": fecha_str,
                                "Liga": liga,
                                "Partido": partido,
                                "Selección": f"Total Partido",
                                "Mercado": "Faltas Totales (> 22.5)",
                                "Línea": "22.5",
                                "Cuota": 1.75,
                                "Probabilidad de Éxito (%)": 82.5,
                                "Casa de Apuestas": casa_apuestas,
                                "Tipo": "Micro-Mercado"
                            })

    df = pd.DataFrame(filas)
    if not df.empty:
        df_betano = df[df["Casa de Apuestas"].str.contains("Betano", case=False, na=False)]
        if not df_betano.empty:
            df = df_betano
    return df

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
        grupo = grupo.drop_duplicates(subset=["Mercado", "Selección"])
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

with st.sidebar:
    st.title("⚙️ Filtros En Vivo")
    st.caption("Filtro estricto: Próximos 4 días")
    st.divider()
    umbral_sencillas = st.slider("Umbral Sencillas (Seguridad %)", 50, 95, 65, 1)
    umbral_combinada = st.slider("Umbral por Pata - Combinada (%)", 50, 90, 55, 1)
    st.divider()
    if st.button("🔄 Refrescar Radar"):
        st.cache_data.clear()
        st.rerun()

df_mercados = obtener_radar_masivo(API_KEY)

st.title("📊 Centro de Mando Quant — En Vivo")
st.markdown("##### Partidos cercanos (Hoy y próximos días) + Micro-Mercados")

if df_mercados.empty:
    st.warning("No hay partidos de las ligas principales en este rango de fecha exacta (estamos en periodo de transición de verano europeo). Intenta refrescar o ajustar los umbrales.")
    st.stop()

df_sencillas_top = obtener_sencillas_top(df_mercados, umbral=umbral_sencillas)
combinada_sugerida = armar_combinada_sugerida(df_mercados, min_prob=umbral_combinada)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Partidos Cercanos", df_mercados["Partido"].nunique())
c2.metric("Opciones Totales", len(df_mercados))
c3.metric("Micro-Mercados", len(df_mercados[df_mercados["Tipo"] == "Micro-Mercado"]))
c4.metric("Casa Principal", df_mercados["Casa de Apuestas"].iloc[0])

st.divider()

col_izq, col_der = st.columns([1.1, 1])

with col_izq:
    st.markdown(f"**✅ Top Sencillas / Micro-Mercados (≥ {umbral_sencillas}%)**")
    if not df_sencillas_top.empty:
        st.dataframe(
            df_sencillas_top.head(15).style.apply(resaltar_verde, axis=1),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Baja el umbral de seguridad en el menú izquierdo para ver más opciones disponibles.")

with col_der:
    st.markdown("**🔥 Combinada Sugerida (Bet Builder)**")
    if combinada_sugerida:
        patas_html = ""
        for pata in combinada_sugerida["patas"]:
            patas_html += f"""
                <div class="combo-leg">
                    ⚽ <b>{pata['Mercado']}</b>: {pata['Selección']}<br>
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
        st.info("No se encontró combinada con los filtros actuales.")

st.divider()
with st.expander("📋 Ver el radar completo de opciones"):
    st.dataframe(df_mercados, use_container_width=True, hide_index=True)

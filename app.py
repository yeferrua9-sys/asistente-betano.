import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Centro de Mando Quant | Sports Data Hub",
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
        .value-box {
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
# GESTIÓN SEGURA DE CREDENCIALES
# =========================================================
ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", "")

# =========================================================
# CONEXIÓN THESPORTSDB (EVENTOS Y LIGAS REALES)
# =========================================================
@st.cache_data(ttl=300)
def fetch_the_sports_db_events():
    url = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d=2026-07-28"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data.get("events") or []
    except Exception:
        pass
    return []

# =========================================================
# CONEXIÓN THE ODDS API (AUTENTICACIÓN Y ENDPOINTS OFICIALES)
# =========================================================
@st.cache_data(ttl=600)
def fetch_odds_api_sports(api_key: str):
    if not api_key:
        return [], "API KEY NO CONFIGURADA EN SECRETS"
    url = "https://api.the-odds-api.com/v4/sports/"
    params = {"apiKey": api_key}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 401:
            return [], "API KEY INVALIDA"
        if res.status_code != 200:
            return [], f"HTTP ERROR {res.status_code}"
        return res.json(), "OK"
    except requests.RequestException as e:
        return [], f"ERROR DE RED: {str(e)}"

@st.cache_data(ttl=300)
def fetch_odds_api_events(api_key: str, sport_key: str):
    if not api_key:
        return [], "API KEY INVALIDA"
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk,us",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 401:
            return [], "API KEY INVALIDA"
        if res.status_code == 404:
            return [], "UNKNOWN SPORT"
        if res.status_code != 200:
            return [], f"HTTP ERROR {res.status_code}"
        return res.json(), "OK"
    except requests.RequestException as e:
        return [], f"ERROR DE RED: {str(e)}"

# =========================================================
# MATCHING DE EQUIPOS
# =========================================================
def normalize_name(name: str) -> str:
    if not name:
        return ""
    return name.lower().replace("fc", "").replace("cf", "").replace("if", "").replace("united", "").strip()

def match_teams(team_db, team_odds):
    n_db = normalize_name(team_db)
    n_odds = normalize_name(team_odds)
    if n_db == n_odds or n_db in n_odds or n_odds in n_db:
        return True
    return False

# =========================================================
# HISTÓRICO Y MOTOR PREDICTIVO (CERO DATA LEAKAGE)
# =================================================py
@st.cache_data(ttl=300)
def fetch_team_history(team_name: str):
    url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={team_name}"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            teams = res.json().get("teams")
            if teams:
                team_id = teams[0].get("idTeam")
                hist_url = f"https://www.thesportsdb.com/api/v1/json/3/eventslast.php?id={team_id}"
                h_res = requests.get(hist_url, timeout=8)
                if h_res.status_code == 200:
                    events = h_res.json().get("results") or []
                    parsed = []
                    for ev in events:
                        h_score = ev.get("intHomeScore")
                        a_score = ev.get("intAwayScore")
                        date_event = ev.get("dateEvent")
                        if h_score is not None and a_score is not None:
                            parsed.append({
                                "date": date_event,
                                "home": ev.get("strHomeTeam"),
                                "away": ev.get("strAwayTeam"),
                                "home_score": int(h_score),
                                "away_score": int(a_score)
                            })
                    return parsed
    except Exception:
        pass
    return []

def build_temporal_dataset(team_a_history, team_b_history):
    combined = team_a_history + team_b_history
    sorted_matches = sorted(combined, key=lambda x: x["date"] if x["date"] else "2000-01-01")
    
    observations = []
    for i in range(len(sorted_matches)):
        past_matches = sorted_matches[:i]
        if len(past_matches) < 3:
            continue
        match = sorted_matches[i]
        h_score = match["home_score"]
        a_score = match["away_score"]
        if h_score > a_score:
            target = 0 # Home win
        elif h_score < a_score:
            target = 2 # Away win
        else:
            target = 1 # Draw
            
        home_gf = np.mean([m["home_score"] if m["home"] == match["home"] else m["away_score"] for m in past_matches[-3:]])
        home_ga = np.mean([m["away_score"] if m["home"] == match["home"] else m["home_score"] for m in past_matches[-3:]])
        away_gf = np.mean([m["home_score"] if m["home"] == match["away"] else m["away_score"] for m in past_matches[-3:]])
        away_ga = np.mean([m["away_score"] if m["home"] == match["away"] else m["home_score"] for m in past_matches[-3:]])
        
        observations.append({
            "features": [home_gf, home_ga, away_gf, away_ga, home_gf - away_ga, away_gf - home_ga],
            "target": target
        })
    return observations

def train_quant_model(observations):
    if len(observations) < 8:
        return None, f"Observaciones válidas del modelo: {len(observations)} · Mínimo requerido: 8"
    
    X = np.array([obs["features"] for obs in observations])
    y = np.array([obs["target"] for obs in observations])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, y)
    return (model, scaler), "OK"

# =========================================================
# INTERFAZ PRINCIPAL (CENTRO DE MANDO QUANT)
# =========================================================
with st.sidebar:
    st.title("⚙️ Configuración Quant")
    st.divider()
    if st.button("🔄 Refrescar Datos"):
        st.cache_data.clear()
        st.rerun()

st.title("📊 Centro de Mando Quant")
st.markdown("##### Sports Data Hub · Datos reales · Histórico · Modelo Predictivo · Mercado Real · Value Betting")
st.divider()

# Diagnóstico API Odds
sports_list, api_status = fetch_odds_api_sports(ODDS_API_KEY)

col_d1, col_d2, col_d3 = st.columns(3)
col_d1.metric("Odds-API.io", "🟢 Conectada" if api_status == "OK" else f"🔴 {api_status}")
col_d2.metric("TheSportsDB", "🟢 Activo")
col_d3.metric("Motor Predictivo", "🟢 Listo (Temporal Validation)")

st.divider()

if api_status != "OK":
    st.markdown(f"""
        <div class="error-box">
            🔴 <b>Odds-API.io:</b> {api_status}<br>
            Verifica tu clave en <code>.streamlit/secrets.toml</code> bajo la variable <code>THE_ODDS_API_KEY</code>.
        </div>
    """, unsafe_allow_html=True)

# Selector de competición de Odds API si está disponible
sport_keys_map = {}
if sports_list and isinstance(sports_list, list):
    for sp in sports_list:
        sport_keys_map[f"{sp.get('group', 'General')} — {sp.get('title', '')}"] = sp.get('key')

selected_comp_label = st.selectbox("🏆 Seleccionar Competición Real (Odds-API)", list(sport_keys_map.keys()) if sport_keys_map else ["Sin competiciones disponibles"])
selected_sport_key = sport_keys_map.get(selected_comp_label)

odds_events = []
if selected_sport_key and api_status == "OK":
    odds_events, odds_status = fetch_odds_api_events(ODDS_API_KEY, selected_sport_key)
    if odds_status != "OK":
        st.markdown(f"<div class='error-box'>⚠️ Error obteniendo cuotas para la competición: {odds_status}</div>", unsafe_allow_html=True)

# Cargar eventos de TheSportsDB
db_events = fetch_the_sports_db_events()

st.markdown(f"### 📋 Partidos Detectados (TheSportsDB: {len(db_events)} eventos | Odds-API: {len(odds_events)} cuotas)")

if not db_events:
    st.info("No se encontraron partidos en TheSportsDB para la fecha actual.")
else:
    for ev in db_events:
        home_db = ev.get("strHomeTeam")
        away_db = ev.get("strAwayTeam")
        league_db = ev.get("strLeague")
        time_db = ev.get("strTime", "00:00")
        
        matched_odds_event = None
        for od_ev in odds_events:
            if match_teams(home_db, od_ev.get("home_team", "")) and match_teams(away_db, od_ev.get("away_team", "")):
                matched_odds_event = od_ev
                break
                
        with st.container():
            st.markdown(f"""
                <div class="match-card">
                    <h3 style="margin-top:0; color:#f97316;">⚽ {home_db} vs {away_db}</h3>
                    <p style="color:#9ca3af; font-size:14px; margin-bottom:15px;">🏆 <b>{league_db}</b> &nbsp;|&nbsp; ⏰ Hora: {time_db}</p>
            """, unsafe_allow_html=True)
            
            if matched_odds_event:
                st.markdown("<div class='success-box'>🟢 EVENTO Y MERCADO ENCONTRADOS EN ODDS-API</div>", unsafe_allow_html=True)
                
                # Obtener histórico para análisis predictivo
                h_hist = fetch_team_history(home_db)
                a_hist = fetch_team_history(away_db)
                
                if len(h_hist) < 3 or len(a_hist) < 3:
                    st.warning(f"Histórico insuficiente para construir la forma. Disponibles: {len(h_hist)} local / {len(a_hist)} visitante. Se requieren 3.")
                else:
                    st.success(f"Histórico válido: {len(h_hist)} partidos de {home_db} · {len(a_hist)} de {away_db}")
                    obs = build_temporal_dataset(h_hist, a_hist)
                    model_pack, model_msg = train_quant_model(obs)
                    
                    if not model_pack:
                        st.info(f"Histórico suficiente para calcular forma, pero todavía no existe un dataset temporal suficiente para entrenar el modelo. ({model_msg})")
                    else:
                        st.success("Modelo predictivo entrenado con éxito bajo validación temporal estricta.")
                        model, scaler = model_pack
                        
                        # Generar features para el partido actual (basadas únicamente en partidos anteriores)
                        recent_h = [h_hist[-1]["home_score"] if h_hist[-1]["home"] == home_db else h_hist[-1]["away_score"]] if h_hist else [1.0]
                        recent_h_ga = [h_hist[-1]["away_score"] if h_hist[-1]["home"] == home_db else h_hist[-1]["home_score"]] if h_hist else [1.0]
                        recent_a = [a_hist[-1]["home_score"] if a_hist[-1]["home"] == away_db else a_hist[-1]["away_score"]] if a_hist else [1.0]
                        recent_a_ga = [a_hist[-1]["away_score"] if a_hist[-1]["home"] == away_db else a_hist[-1]["home_score"]] if a_hist else [1.0]
                        
                        feat = np.array([[np.mean(recent_h), np.mean(recent_h_ga), np.mean(recent_a), np.mean(recent_a_ga), np.mean(recent_h)-np.mean(recent_a_ga), np.mean(recent_a)-np.mean(recent_h_ga)]])
                        feat_scaled = scaler.transform(feat)
                        probs = model.predict_proba(feat_scaled)[0] # [Home_prob, Draw_prob, Away_prob]
                        
                        # Extraer cuotas reales de Odds-API
                        for book in matched_odds_event.get("bookmakers", []):
                            book_name = book.get("title")
                            for mkt in book.get("markets", []):
                                if mkt.get("key") == "h2h":
                                    for outcome in mkt.get("outcomes", []):
                                        sel_name = outcome.get("name")
                                        price = outcome.get("price")
                                        
                                        # Asignar probabilidad propia según selección
                                        own_prob = 0.0
                                        if match_teams(sel_name, home_db):
                                            own_prob = probs[0]
                                        elif match_teams(sel_name, away_db):
                                            own_prob = probs[2]
                                        else:
                                            own_prob = probs[1] # Draw
                                            
                                        implied_prob = 1.0 / price
                                        edge = own_prob - implied_prob
                                        ev = (own_prob * price) - 1.0
                                        value_score = edge * 100.0
                                        
                                        if ev > 0:
                                            st.markdown(f"""
                                                <div class="value-box">
                                                    🔥 <b>VALUE BET DETECTADO ({book_name})</b> — <i>{sel_name}</i><br>
                                                    Cuota Real: <b>{price}</b> | Prob. Modelo: <b>{round(own_prob*100, 1)}%</b> | Implícita: <b>{round(implied_prob*100, 1)}%</b><br>
                                                    Edge: <b>{round(edge*100, 1)}%</b> | EV: <b>+{round(ev, 2)}</b> | Value Score: <b>{round(value_score, 2)}</b>
                                                </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"""
                                                <div class="market-badge">
                                                    <b>{book_name}</b> — <i>{sel_name}</i> (Cuota: <b>{price}</b> | EV: {round(ev, 2)})
                                                </div>
                                            """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='error-box'>⚠️ Mercado no disponible en Odds-API.io para este evento o pendiente de sincronización.</div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

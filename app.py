import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Centro de Mando Quant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TSDB_BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

# ============================================================
# ESTILO VISUAL
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #0b0f15;
    color: #f5f7fa;
}

section[data-testid="stSidebar"] {
    background: #151a23;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    color: #9ca3af;
    font-size: 17px;
    margin-bottom: 25px;
}

.card {
    background: #151a23;
    border: 1px solid #293241;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
}

.match-card {
    background: #111720;
    border: 1px solid #293241;
    border-radius: 18px;
    padding: 24px;
    margin: 12px 0;
}

.team {
    font-size: 20px;
    font-weight: 700;
}

.vs {
    color: #6b7280;
    font-size: 14px;
    text-align: center;
}

.odd {
    font-size: 22px;
    font-weight: 800;
}

.green {
    color: #10b981;
}

.yellow {
    color: #f59e0b;
}

.red {
    color: #ef4444;
}

.gray {
    color: #9ca3af;
}

.section-title {
    font-size: 27px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
}

.status {
    padding: 7px 12px;
    border-radius: 10px;
    display: inline-block;
    font-weight: 700;
    font-size: 13px;
}

.status-green {
    background: #063c2d;
    color: #34d399;
    border: 1px solid #10b981;
}

.status-yellow {
    background: #3b2b08;
    color: #fbbf24;
    border: 1px solid #eab308;
}

.status-gray {
    background: #202630;
    color: #9ca3af;
    border: 1px solid #374151;
}

div[data-testid="stMetric"] {
    background: #151a23;
    border: 1px solid #293241;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MAPEO THE ODDS API
# ============================================================

ODDS_SPORTS = {
    "Fútbol": {
        "soccer_epl": "Premier League",
        "soccer_uefa_champs_league": "Champions League",
        "soccer_uefa_europa_league": "Europa League",
        "soccer_spain_la_liga": "La Liga",
        "soccer_italy_serie_a": "Serie A",
        "soccer_germany_bundesliga": "Bundesliga",
    },

    "Baloncesto": {
        "basketball_nba": "NBA",
        "basketball_wnba": "WNBA",
        "basketball_ncaab": "NCAAB",
    },

    "Béisbol": {
        "baseball_mlb": "MLB",
    },

    "Hockey": {
        "icehockey_nhl": "NHL",
    },

    "Fútbol americano": {
        "americanfootball_nfl": "NFL",
        "americanfootball_ncaaf": "NCAAF",
    },
}


# ============================================================
# THE SPORTSB DB
# ============================================================

@st.cache_data(ttl=300)
def get_events_day(selected_date, sport_filter):

    url = f"{TSDB_BASE_URL}/eventsday.php"

    params = {"d": selected_date}

    if sport_filter != "Todos":
        params["s"] = sport_filter

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        if response.status_code != 200:
            return [], f"HTTP {response.status_code}"

        data = response.json()

        events = data.get("events") or []

        return events, "OK" if events else "NO_EVENTS"

    except requests.RequestException as error:

        return [], f"CONNECTION_ERROR: {error}"

    except ValueError:

        return [], "JSON_ERROR"


# ============================================================
# FORMATO DE HORA
# ============================================================

def format_time(event):

    timestamp = event.get("strTimestamp")

    if timestamp:

        try:

            dt = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )

            return dt.strftime("%H:%M")

        except Exception:
            pass

    raw = event.get("strTime")

    if raw:
        return raw[:5]

    return "--"


# ============================================================
# EVENTOS → DATAFRAME
# ============================================================

def events_to_dataframe(events):

    rows = []

    for event in events:

        home = event.get("strHomeTeam")
        away = event.get("strAwayTeam")

        if home and away:
            matchup = f"{home} vs {away}"

        else:
            matchup = event.get(
                "strEvent",
                "Evento deportivo"
            )

        rows.append({
            "ID": event.get("idEvent"),
            "Deporte": event.get("strSport"),
            "Liga": event.get("strLeague"),
            "Evento": matchup,
            "Fecha": event.get("dateEvent"),
            "Hora": format_time(event),
            "Local": home,
            "Visitante": away,
            "Estadio": event.get("strVenue"),
            "Ciudad": event.get("strCity"),
            "País": event.get("strCountry"),
        })

    return pd.DataFrame(rows)


# ============================================================
# THE ODDS API
# ============================================================

@st.cache_data(ttl=120)
def get_odds(
    api_key,
    sport_key,
    regions,
    market
):

    if not api_key:
        return [], "NO_API_KEY"

    url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds"

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": market,
        "oddsFormat": "decimal",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code == 401:
            return [], "INVALID_API_KEY"

        if response.status_code == 429:
            return [], "RATE_LIMIT"

        if response.status_code != 200:
            return [], f"HTTP_{response.status_code}"

        data = response.json()

        return data or [], "OK"

    except requests.RequestException as error:

        return [], f"CONNECTION_ERROR: {error}"

    except ValueError:

        return [], "JSON_ERROR"


# ============================================================
# CUOTAS → DATAFRAME
# ============================================================

def odds_to_dataframe(data):

    rows = []

    for event in data:

        home = event.get("home_team", "")
        away = event.get("away_team", "")

        for bookmaker in event.get("bookmakers", []):

            bookmaker_name = bookmaker.get(
                "title",
                "Casa"
            )

            for market in bookmaker.get("markets", []):

                market_key = market.get("key")

                for outcome in market.get(
                    "outcomes",
                    []
                ):

                    price = outcome.get("price")

                    if price is None:
                        continue

                    rows.append({
                        "Evento": f"{home} vs {away}",
                        "Selección": outcome.get("name"),
                        "Cuota": float(price),
                        "Casa": bookmaker_name,
                        "Mercado": market_key,
                        "Inicio": event.get(
                            "commence_time"
                        ),
                    })

    return pd.DataFrame(rows)


# ============================================================
# MOTOR QUANT
# ============================================================

def implied_probability(odds):

    if odds is None or odds <= 1:
        return None

    return 1 / odds


def calculate_ev(probability, odds):

    if probability is None:
        return None

    return (
        probability * odds
    ) - 1


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Centro de Mando")

    st.caption("Sports Data Hub — FASE 5")

    st.divider()

    st.markdown("### 📅 Fecha")

    selected_date = st.date_input(
        "Día de análisis",
        value=date.today()
    )

    st.divider()

    st.markdown("### 🏟️ Deporte")

    sport_filter = st.selectbox(
        "Selecciona deporte",
        [
            "Todos",
            "Soccer",
            "Basketball",
            "Tennis",
            "Baseball",
            "Ice Hockey",
            "American Football",
            "Athletics",
        ]
    )

    st.divider()

    st.markdown("### 💰 Mercado")

    odds_api_key = st.text_input(
        "API Key de The Odds API",
        type="password"
    )

    odds_region = st.selectbox(
        "Región",
        [
            "us,uk,eu",
            "us",
            "uk",
            "eu",
        ]
    )

    odds_market = st.selectbox(
        "Mercado",
        [
            "h2h",
            "spreads",
            "totals",
        ]
    )

    st.divider()

    if st.button(
        "🔄 Actualizar datos",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Centro de Mando Quant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sports Data Hub · Datos reales · Cuotas reales · Análisis cuantitativo'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ESTADO
# ============================================================

events, event_status = get_events_day(
    selected_date.strftime("%Y-%m-%d"),
    sport_filter
)

df_events = events_to_dataframe(events)


status1, status2, status3 = st.columns(3)

with status1:

    st.markdown(
        '<div class="status status-green">'
        '🟢 DATOS DEPORTIVOS'
        '</div>',
        unsafe_allow_html=True
    )

with status2:

    if odds_api_key:

        st.markdown(
            '<div class="status status-green">'
            '🟢 CUOTAS CONECTADAS'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="status status-yellow">'
            '🟡 CUOTAS PENDIENTES'
            '</div>',
            unsafe_allow_html=True
        )

with status3:

    st.markdown(
        '<div class="status status-yellow">'
        '🟡 MODELO PREDICTIVO PENDIENTE'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# KPIs
# ============================================================

st.markdown(
    '<div class="section-title">📈 Resumen</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Eventos",
    len(df_events)
)

c2.metric(
    "Deportes",
    df_events["Deporte"].nunique()
    if not df_events.empty else 0
)

c3.metric(
    "Ligas",
    df_events["Liga"].nunique()
    if not df_events.empty else 0
)

c4.metric(
    "Países",
    df_events["País"].nunique()
    if not df_events.empty else 0
)


# ============================================================
# FILTROS
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Filtrar partidos</div>',
    unsafe_allow_html=True
)

if not df_events.empty:

    f1, f2 = st.columns(2)

    with f1:

        league_options = sorted(
            df_events["Liga"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_league = st.selectbox(
            "🏆 Competición",
            ["Todas"] + league_options
        )

    with f2:

        country_options = sorted(
            df_events["País"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_country = st.selectbox(
            "🌎 País",
            ["Todos"] + country_options
        )

    filtered = df_events.copy()

    if selected_league != "Todas":

        filtered = filtered[
            filtered["Liga"] == selected_league
        ]

    if selected_country != "Todos":

        filtered = filtered[
            filtered["País"] == selected_country
        ]

else:

    filtered = df_events


# ============================================================
# PARTIDOS
# ============================================================

st.markdown(
    '<div class="section-title">🏟️ Partidos y eventos</div>',
    unsafe_allow_html=True
)

if filtered.empty:

    st.info(
        "No hay eventos deportivos para los filtros seleccionados."
    )

else:

    for index, row in filtered.iterrows():

        with st.container():

            st.markdown(
                f"""
                <div class="match-card">

                    <div class="gray">
                    {row["Liga"] or "Competición no disponible"}
                    </div>

                    <br>

                    <div class="team">
                    {row["Evento"]}
                    </div>

                    <br>

                    <div class="gray">
                    📅 {row["Fecha"]}
                    &nbsp;&nbsp; ⏰ {row["Hora"]}
                    </div>

                    <div class="gray">
                    📍 {row["Estadio"] or "Estadio no disponible"}
                    &nbsp; · &nbsp;
                    {row["Ciudad"] or ""}
                    &nbsp; · &nbsp;
                    {row["País"] or ""}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "🔍 Analizar evento",
                key=f"analyze_{row['ID']}"
            ):

                st.session_state[
                    "selected_event"
                ] = row["Evento"]


# ============================================================
# EVENTO SELECCIONADO
# ============================================================

selected_event = st.session_state.get(
    "selected_event"
)

if selected_event:

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🎯 Análisis del evento'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">

        <div class="team">
        {selected_event}
        </div>

        <br>

        <div class="gray">
        Este evento fue seleccionado para análisis.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CUOTAS
# ============================================================

st.markdown(
    '<div class="section-title">💰 Mercado de cuotas</div>',
    unsafe_allow_html=True
)

odds_df = pd.DataFrame()

if not odds_api_key:

    st.markdown(
        """
        <div class="card">

        <b>🟡 Cuotas todavía no conectadas</b>

        <br><br>

        Introduce tu API Key de The Odds API
        en el panel izquierdo.

        </div>
        """,
        unsafe_allow_html=True
    )

elif sport_filter == "Todos":

    st.info(
        "Selecciona un deporte compatible con The Odds API."
    )

elif sport_filter not in ODDS_SPORTS:

    st.info(
        "Este deporte todavía no está conectado "
        "al módulo de cuotas."
    )

else:

    sport_options = ODDS_SPORTS[sport_filter]

    selected_odds_sport = st.selectbox(
        "Competición de cuotas",
        list(sport_options.keys()),
        format_func=lambda x: sport_options[x]
    )

    odds_data, odds_status = get_odds(
        odds_api_key,
        selected_odds_sport,
        odds_region,
        odds_market
    )

    if odds_status == "OK":

        odds_df = odds_to_dataframe(
            odds_data
        )

        if odds_df.empty:

            st.warning(
                "La API respondió, pero no devolvió "
                "cuotas para este mercado."
            )

        else:

            st.success(
                f"🟢 {len(odds_df)} cuotas reales recibidas."
            )

            # =================================================
            # TARJETAS DE CUOTAS
            # =================================================

            unique_events = (
                odds_df["Evento"]
                .drop_duplicates()
                .tolist()
            )

            for event_name in unique_events[:20]:

                event_odds = odds_df[
                    odds_df["Evento"] == event_name
                ]

                st.markdown(
                    f"""
                    <div class="card">

                    <div class="team">
                    {event_name}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                for _, odd in event_odds.head(12).iterrows():

                    probability = implied_probability(
                        odd["Cuota"]
                    )

                    col1, col2, col3, col4 = st.columns(
                        [4, 1.5, 2, 2]
                    )

                    with col1:

                        st.write(
                            f"**{odd['Selección']}**"
                        )

                    with col2:

                        st.markdown(
                            f'<div class="odd">'
                            f'{odd["Cuota"]:.2f}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with col3:

                        st.write(
                            f"{probability * 100:.2f}%"
                        )

                    with col4:

                        st.caption(
                            odd["Casa"]
                        )

    elif odds_status == "INVALID_API_KEY":

        st.error(
            "🔴 La API Key no es válida."
        )

    elif odds_status == "RATE_LIMIT":

        st.warning(
            "🟡 Se alcanzó el límite de consultas."
        )

    else:

        st.error(
            f"No se pudieron obtener cuotas: {odds_status}"
        )


# ============================================================
# MOTOR QUANT
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Motor Quant</div>',
    unsafe_allow_html=True
)

q1, q2, q3 = st.columns(3)

with q1:

    st.markdown(
        """
        <div class="card">

        <b>Probabilidad implícita</b>

        <br><br>

        🟢 Disponible con cuotas reales.

        </div>
        """,
        unsafe_allow_html=True
    )

with q2:

    st.markdown(
        """
        <div class="card">

        <b>Probabilidad del modelo</b>

        <br><br>

        🟡 Pendiente de conectar el modelo predictivo.

        </div>
        """,
        unsafe_allow_html=True
    )

with q3:

    st.markdown(
        """
        <div class="card">

        <b>Edge / EV / Value</b>

        <br><br>

        🟡 Se activarán cuando exista
        una probabilidad propia.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INTEGRIDAD
# ============================================================

st.markdown(
    """
    <div class="card">

    <b>🔐 Integridad del sistema</b>

    <br><br>

    El sistema NO inventa cuotas.
    <br>
    El sistema NO inventa probabilidades.
    <br>
    El sistema NO genera Value Score artificial.
    <br><br>

    Las métricas predictivas aparecerán únicamente
    cuando conectemos el modelo estadístico real.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ARQUITECTURA
# ============================================================

with st.expander("🏗️ Ver arquitectura del sistema"):

    st.markdown("""
    ### CAPA 1 — Datos deportivos

    TheSportsDB

    Eventos → Equipos → Ligas → Fechas → Estadios

    ### CAPA 2 — Mercado

    The Odds API

    Casas → Mercados → Selecciones → Cuotas

    ### CAPA 3 — Modelo

    Datos históricos → Variables → Probabilidad

    ### CAPA 4 — Quant

    Probabilidad modelo  
    ↓  
    Probabilidad implícita  
    ↓  
    Edge  
    ↓  
    EV  
    ↓  
    Value Score  
    ↓  
    Ranking

    ### CAPA 5 — Dashboard

    El usuario ve únicamente la información
    relevante y accionable.
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE 5"
)

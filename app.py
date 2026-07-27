"""
CENTRO DE MANDO QUANT — SPORTS DATA HUB
FASE 4B — CUOTAS REALES + MOTOR QUANT

CAPAS:
1. TheSportsDB → eventos deportivos reales
2. The Odds API → cuotas reales
3. Motor Quant → probabilidad implícita, Edge, EV, Value Score
4. Centro de Mando → ranking de oportunidades

IMPORTANTE:
- NO se inventan cuotas.
- NO se inventan probabilidades.
- Las métricas Quant solo aparecen cuando existen cuotas reales.
- La API Key de The Odds API se introduce mediante Streamlit Secrets.
"""

import streamlit as st
import pandas as pd
import requests

from datetime import datetime, date


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Centro de Mando Quant | Sports Data Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# ESTILOS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0b0f15;
        color: #f5f7fa;
    }

    section[data-testid="stSidebar"] {
        background: #20232d;
    }

    .quant-header {
        background: linear-gradient(
            135deg,
            #151a24,
            #10141c
        );
        border: 1px solid #2c3442;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 25px;
    }

    .event-card {
        background: #181d25;
        border: 1px solid #303846;
        border-radius: 16px;
        padding: 20px;
        margin: 14px 0;
    }

    .real-badge {
        background: #063c2d;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 8px 12px;
        display: inline-block;
        margin-bottom: 12px;
    }

    .odds-badge {
        background: #30220a;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 8px 12px;
        display: inline-block;
        margin-bottom: 12px;
    }

    .info-box {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }

    .warning-box {
        background: #3b2b08;
        border: 1px solid #eab308;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }

    .quant-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# API — THESPORTSDB
# =========================================================

TSDB_BASE_URL = (
    "https://www.thesportsdb.com/api/v1/json/123"
)


# =========================================================
# API — THE ODDS API
# =========================================================

ODDS_BASE_URL = (
    "https://api.the-odds-api.com/v4"
)


# =========================================================
# MAPEO DE DEPORTES
# =========================================================

ODDS_SPORTS = {
    "Soccer": {
        "soccer_epl": "Premier League",
        "soccer_uefa_champs_league": "Champions League",
        "soccer_uefa_europa_league": "Europa League",
        "soccer_spain_la_liga": "La Liga",
        "soccer_italy_serie_a": "Serie A",
        "soccer_germany_bundesliga": "Bundesliga",
    },

    "Basketball": {
        "basketball_nba": "NBA",
        "basketball_wnba": "WNBA",
        "basketball_ncaab": "NCAAB",
    },

    "Baseball": {
        "baseball_mlb": "MLB",
    },

    "Ice Hockey": {
        "icehockey_nhl": "NHL",
    },

    "American Football": {
        "americanfootball_nfl": "NFL",
        "americanfootball_ncaaf": "NCAAF",
    },
}


# =========================================================
# THE SPORTSB DB — EVENTOS
# =========================================================

@st.cache_data(ttl=300)
def get_events_day(
    selected_date,
    sport_filter="Todos"
):

    url = f"{TSDB_BASE_URL}/eventsday.php"

    params = {
        "d": selected_date
    }

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

        events = data.get("events")

        if not events:
            return [], "NO_EVENTS"

        return events, "OK"

    except requests.RequestException as error:

        return [], f"ERROR_CONNECTION: {error}"

    except ValueError:

        return [], "ERROR_JSON"


# =========================================================
# FORMATEAR HORA
# =========================================================

def format_event_time(event):

    timestamp = event.get("strTimestamp")

    if timestamp:

        try:

            dt = datetime.fromisoformat(
                timestamp.replace("Z", "")
            )

            return dt.strftime("%H:%M")

        except Exception:
            pass

    raw_time = event.get("strTime")

    if raw_time:
        return raw_time[:5]

    return "Hora no disponible"


# =========================================================
# EVENTOS → DATAFRAME
# =========================================================

def events_to_dataframe(events):

    rows = []

    for event in events:

        home = event.get("strHomeTeam")
        away = event.get("strAwayTeam")

        if home and away:

            matchup = f"{home} vs {away}"

        elif event.get("strEvent"):

            matchup = event.get("strEvent")

        else:

            matchup = "Evento deportivo"

        rows.append(
            {
                "ID Evento": event.get("idEvent"),
                "Deporte": event.get("strSport"),
                "Liga": event.get("strLeague"),
                "Evento": matchup,
                "Fecha": event.get("dateEvent"),
                "Hora": format_event_time(event),
                "Local": home or "N/A",
                "Visitante": away or "N/A",
                "Temporada": event.get("strSeason"),
                "Estadio": event.get("strVenue"),
                "Ciudad": event.get("strCity"),
                "País": event.get("strCountry"),
                "Estado": event.get("strStatus") or "Programado",
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# THE ODDS API — OBTENER CUOTAS
# =========================================================

@st.cache_data(ttl=120)
def get_odds(
    api_key,
    sport_key,
    regions="us,uk,eu",
    markets="h2h"
):

    if not api_key:

        return [], "NO_API_KEY"

    url = (
        f"{ODDS_BASE_URL}/sports/"
        f"{sport_key}/odds"
    )

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
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
            return [], f"HTTP {response.status_code}"

        data = response.json()

        if not data:
            return [], "NO_ODDS"

        return data, "OK"

    except requests.RequestException as error:

        return [], f"ERROR_CONNECTION: {error}"

    except ValueError:

        return [], "ERROR_JSON"


# =========================================================
# CUOTAS → DATAFRAME
# =========================================================

def odds_to_dataframe(odds_data):

    rows = []

    for event in odds_data:

        home = event.get("home_team")
        away = event.get("away_team")

        bookmakers = event.get(
            "bookmakers",
            []
        )

        for bookmaker in bookmakers:

            bookmaker_name = bookmaker.get(
                "title",
                "Casa desconocida"
            )

            markets = bookmaker.get(
                "markets",
                []
            )

            for market in markets:

                market_key = market.get(
                    "key"
                )

                outcomes = market.get(
                    "outcomes",
                    []
                )

                for outcome in outcomes:

                    outcome_name = outcome.get(
                        "name"
                    )

                    price = outcome.get(
                        "price"
                    )

                    if price:

                        rows.append(
                            {
                                "Evento Odds":
                                    f"{home} vs {away}",

                                "Home":
                                    home,

                                "Away":
                                    away,

                                "Mercado":
                                    market_key,

                                "Selección":
                                    outcome_name,

                                "Cuota":
                                    float(price),

                                "Casa":
                                    bookmaker_name,

                                "Inicio":
                                    event.get(
                                        "commence_time"
                                    ),

                                "Sport Key":
                                    event.get(
                                        "sport_key"
                                    ),
                            }
                        )

    return pd.DataFrame(rows)


# =========================================================
# MOTOR QUANT
# =========================================================

def implied_probability(odds):

    if odds is None:
        return None

    if odds <= 1:
        return None

    return 1 / odds


def calculate_edge(
    model_probability,
    implied_prob
):

    if (
        model_probability is None
        or implied_prob is None
    ):
        return None

    return (
        model_probability
        - implied_prob
    )


def calculate_ev(
    model_probability,
    odds
):

    if (
        model_probability is None
        or odds is None
    ):
        return None

    return (
        model_probability * odds
    ) - 1


def calculate_value_score(
    edge,
    ev
):

    if edge is None or ev is None:
        return None

    return (
        (edge * 100) * 0.5
        +
        (ev * 100) * 0.5
    )


def classify_value(
    value_score
):

    if value_score is None:
        return "SIN MODELO"

    if value_score >= 10:
        return "MUY ALTO"

    if value_score >= 5:
        return "ALTO"

    if value_score >= 2:
        return "POSITIVO"

    if value_score >= 0:
        return "NEUTRO"

    return "NEGATIVO"


# =========================================================
# APLICAR MOTOR QUANT
# =========================================================

def apply_quant_engine(df):

    if df.empty:
        return df

    result = df.copy()

    result["Prob. Implícita"] = (
        result["Cuota"]
        .apply(implied_probability)
    )

    # =====================================================
    # IMPORTANTE
    # =====================================================
    # Todavía no tenemos un modelo predictivo propio.
    # Por eso NO inventamos una probabilidad.
    #
    # La columna queda vacía hasta que conectemos
    # el modelo estadístico.
    # =====================================================

    result["Prob. Modelo"] = pd.NA

    result["Edge"] = pd.NA

    result["EV"] = pd.NA

    result["Value Score"] = pd.NA

    result["Clasificación"] = "SIN MODELO"

    return result


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Centro de Mando")

    st.caption(
        "Sports Data Hub — FASE 4B"
    )

    st.divider()

    st.markdown("### 🗓️ Fecha")

    selected_date = st.date_input(
        "Día de análisis",
        value=date.today()
    )

    st.divider()

    st.markdown("### 🏟️ Deporte")

    sport_filter = st.selectbox(
        "Filtrar deporte",
        [
            "Todos",
            "Soccer",
            "Basketball",
            "Tennis",
            "Baseball",
            "Ice Hockey",
            "American Football",
            "Motorsport",
            "Athletics",
        ]
    )

    st.divider()

    st.markdown("### 💰 Cuotas")

    odds_api_key = st.text_input(
        "The Odds API Key",
        type="password",
        help=(
            "Introduce aquí tu API Key de "
            "The Odds API."
        )
    )

    odds_region = st.selectbox(
        "Región de casas",
        [
            "us",
            "uk",
            "eu",
            "us,uk,eu",
        ],
        index=3
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

    st.divider()

    st.info(
        "Datos deportivos: TheSportsDB\n\n"
        "Cuotas: The Odds API"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="quant-header">

        <h1>📊 Centro de Mando Quant</h1>

        <h3>Sports Data Hub</h3>

        <p>
        Plataforma de análisis deportivo con datos reales,
        cuotas reales y motor cuantitativo.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ESTADO GENERAL
# =========================================================

col_a, col_b, col_c = st.columns(3)

with col_a:

    st.markdown(
        """
        <div class="real-badge">
        🟢 DATOS DEPORTIVOS REALES
        </div>
        """,
        unsafe_allow_html=True
    )

with col_b:

    if odds_api_key:

        st.markdown(
            """
            <div class="odds-badge">
            🟢 CUOTAS CONECTADAS
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="odds-badge">
            🟡 CUOTAS PENDIENTES
            </div>
            """,
            unsafe_allow_html=True
        )

with col_c:

    st.markdown(
        """
        <div class="real-badge">
        🟢 MOTOR QUANT ACTIVO
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# EVENTOS THE SPORTSB DB
# =========================================================

events, api_status = get_events_day(
    selected_date.strftime("%Y-%m-%d"),
    sport_filter
)


if api_status == "OK":

    st.success(
        "🟢 Conexión correcta con TheSportsDB."
    )

elif api_status == "NO_EVENTS":

    st.warning(
        "TheSportsDB respondió correctamente, "
        "pero no existen eventos para los filtros."
    )

else:

    st.error(
        f"Error en TheSportsDB: {api_status}"
    )


df_events = events_to_dataframe(events)


# =========================================================
# KPIs
# =========================================================

if not df_events.empty:

    total_events = len(df_events)

    sports_count = (
        df_events["Deporte"].nunique()
    )

    leagues_count = (
        df_events["Liga"].nunique()
    )

    countries_count = (
        df_events["País"]
        .dropna()
        .nunique()
    )

else:

    total_events = 0
    sports_count = 0
    leagues_count = 0
    countries_count = 0


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "EVENTOS",
    total_events
)

col2.metric(
    "DEPORTES",
    sports_count
)

col3.metric(
    "LIGAS",
    leagues_count
)

col4.metric(
    "PAÍSES",
    countries_count
)


# =========================================================
# FILTROS LOCALES
# =========================================================

filtered_events = df_events.copy()

if not df_events.empty:

    st.divider()

    st.markdown(
        "## 🔎 Filtros del Centro de Mando"
    )

    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:

        leagues = sorted(
            df_events["Liga"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_league = st.selectbox(
            "🏆 Liga",
            ["Todas"] + leagues
        )

    with col_filter2:

        countries = sorted(
            df_events["País"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_country = st.selectbox(
            "🌎 País",
            ["Todos"] + countries
        )

    if selected_league != "Todas":

        filtered_events = filtered_events[
            filtered_events["Liga"]
            == selected_league
        ]

    if selected_country != "Todos":

        filtered_events = filtered_events[
            filtered_events["País"]
            == selected_country
        ]


# =========================================================
# EVENTOS
# =========================================================

st.divider()

st.markdown(
    "## 🏟️ Eventos deportivos reales"
)


if filtered_events.empty:

    st.info(
        "No hay eventos disponibles."
    )

else:

    st.caption(
        f"Mostrando {len(filtered_events)} eventos."
    )

    for _, row in filtered_events.iterrows():

        st.markdown(
            f"""
            <div class="event-card">

                <h3>
                    🏟️ {row["Evento"]}
                </h3>

                <p>
                    🏆 <b>{row["Liga"]}</b>
                    &nbsp; | &nbsp;
                    🏅 {row["Deporte"]}
                </p>

                <p>
                    📅 {row["Fecha"]}
                    &nbsp; | &nbsp;
                    ⏰ {row["Hora"]}
                </p>

                <p>
                    📍 {row["Estadio"] or "Estadio no disponible"}
                    &nbsp; | &nbsp;
                    {row["Ciudad"] or ""}
                    &nbsp; | &nbsp;
                    {row["País"] or ""}
                </p>

                <p>
                    🆔 ID evento:
                    <b>{row["ID Evento"]}</b>
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# CUOTAS REALES
# =========================================================

st.divider()

st.markdown(
    "## 💰 Cuotas reales de mercado"
)


odds_df = pd.DataFrame()


if not odds_api_key:

    st.warning(
        "🟡 Introduce tu API Key de The Odds API "
        "en la barra lateral para cargar cuotas reales."
    )

else:

    if sport_filter == "Todos":

        st.info(
            "Selecciona un deporte compatible con "
            "The Odds API para consultar sus cuotas."
        )

    elif sport_filter not in ODDS_SPORTS:

        st.info(
            f"The Odds API no está configurada todavía "
            f"para {sport_filter} en esta versión."
        )

    else:

        sport_options = ODDS_SPORTS[
            sport_filter
        ]

        selected_odds_sport = st.selectbox(
            "Competición de cuotas",
            list(sport_options.keys()),
            format_func=lambda x:
                sport_options[x]
        )

        odds_data, odds_status = get_odds(
            odds_api_key,
            selected_odds_sport,
            odds_region,
            odds_market
        )

        if odds_status == "OK":

            st.success(
                "🟢 Cuotas reales recibidas "
                "desde The Odds API."
            )

            odds_df = odds_to_dataframe(
                odds_data
            )

        elif odds_status == "INVALID_API_KEY":

            st.error(
                "🔴 La API Key de The Odds API "
                "no es válida."
            )

        elif odds_status == "RATE_LIMIT":

            st.warning(
                "🟡 Se alcanzó el límite de consultas "
                "de The Odds API."
            )

        elif odds_status == "NO_ODDS":

            st.info(
                "No hay cuotas disponibles "
                "para este deporte/mercado."
            )

        else:

            st.error(
                f"Error al obtener cuotas: {odds_status}"
            )


# =========================================================
# MOTOR QUANT
# =========================================================

st.divider()

st.markdown(
    "## 🧠 Motor Quant"
)

st.markdown(
    """
    <div class="info-box">

    <b>Estado: 🟢 OPERATIVO</b>

    <br><br>

    El motor calcula:

    <br>

    • Probabilidad implícita<br>
    • Probabilidad del modelo<br>
    • Edge<br>
    • EV<br>
    • Value Score<br>
    • Clasificación de valor

    <br><br>

    <b>Regla de integridad:</b>

    La probabilidad del modelo NO se inventa.
    Hasta conectar el modelo predictivo, las métricas
    que dependen de ella permanecen vacías.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PROCESAMIENTO CUANT
# =========================================================

if not odds_df.empty:

    quant_df = apply_quant_engine(
        odds_df
    )

    st.markdown(
        "### 📊 Mercado recibido"
    )

    display_quant = quant_df.copy()

    display_quant["Prob. Implícita"] = (
        display_quant["Prob. Implícita"]
        .apply(
            lambda x:
            f"{x * 100:.2f}%"
            if pd.notna(x)
            else "—"
        )
    )

    display_quant["Prob. Modelo"] = "—"
    display_quant["Edge"] = "—"
    display_quant["EV"] = "—"
    display_quant["Value Score"] = "—"

    columns = [
        "Evento Odds",
        "Mercado",
        "Selección",
        "Cuota",
        "Prob. Implícita",
        "Casa",
        "Prob. Modelo",
        "Edge",
        "EV",
        "Value Score",
        "Clasificación",
    ]

    st.dataframe(
        display_quant[columns],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Las métricas Quant aparecerán cuando "
        "existan cuotas reales y una probabilidad "
        "generada por el modelo."
    )


# =========================================================
# ARQUITECTURA
# =========================================================

st.divider()

st.markdown(
    "## 🏗️ Arquitectura del sistema"
)

st.markdown(
    """
    ### CAPA 1 — Datos deportivos 🟢

    **TheSportsDB**

    Eventos · Equipos · Ligas · Fechas · Horarios ·
    Estadios · Países · IDs

    ---

    ### CAPA 2 — Cuotas 🟢 / 🟡

    **The Odds API**

    Casas de apuestas · Mercados · Selecciones ·
    Cuotas · Horarios

    ---

    ### CAPA 3 — Motor Quant 🟢

    Probabilidad implícita

    ↓

    Probabilidad del modelo

    ↓

    Edge

    ↓

    EV

    ↓

    Value Score

    ↓

    Ranking

    ---

    ### CAPA 4 — Centro de Mando 🟢

    **Datos → Cuotas → Modelo → Comparación → Edge → EV → Value → Ranking**
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Centro de Mando Quant — Sports Data Hub | "
    "FASE 4B — Cuotas reales + Motor Quant"
)

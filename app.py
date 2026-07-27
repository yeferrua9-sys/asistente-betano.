"""
CENTRO DE MANDO QUANT — SPORTS DATA HUB
FASE 3 — DATOS DEPORTIVOS REALES

Fuente:
TheSportsDB V1 — API gratuita

IMPORTANTE:
- Los eventos deportivos proceden de TheSportsDB.
- NO se presentan cuotas inventadas.
- NO se presentan probabilidades inventadas.
- El motor Quant todavía NO genera recomendaciones de apuesta.
- Esta fase construye la capa de datos reales.
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

    .error-box {
        background: #3b1111;
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONFIGURACIÓN THE SPORTS DB
# =========================================================

TSDB_BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"


# =========================================================
# FUNCIÓN — OBTENER EVENTOS DEL DÍA
# =========================================================

@st.cache_data(ttl=300)
def get_events_day(selected_date, sport_filter="Todos"):

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
# FUNCIÓN — FORMATEAR HORA
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
# FUNCIÓN — CONSTRUIR DATAFRAME
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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Centro de Mando")

    st.caption(
        "Sports Data Hub — FASE 3"
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

    if st.button(
        "🔄 Actualizar datos",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.info(
        "Fuente deportiva activa: "
        "TheSportsDB API gratuita."
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
        Plataforma de análisis deportivo con datos reales.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ESTADO DE CONEXIÓN
# =========================================================

st.markdown(
    """
    <div class="real-badge">
        🟢 DATOS DEPORTIVOS REALES
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">

    <b>Fuente activa:</b> TheSportsDB V1 API

    <br><br>

    Esta pantalla obtiene directamente eventos deportivos
    desde la API y no utiliza partidos inventados.

    <br><br>

    <b>Cuotas de apuestas:</b> todavía NO conectadas.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CONSULTAR API
# =========================================================

events, api_status = get_events_day(
    selected_date.strftime("%Y-%m-%d"),
    sport_filter
)


# =========================================================
# ESTADO API
# =========================================================

if api_status == "OK":

    st.success(
        "🟢 Conexión correcta con TheSportsDB."
    )

elif api_status == "NO_EVENTS":

    st.warning(
        "La API respondió correctamente, "
        "pero no hay eventos disponibles para "
        "los filtros seleccionados."
    )

elif api_status.startswith("HTTP"):

    st.error(
        f"Error HTTP de TheSportsDB: {api_status}"
    )

else:

    st.error(
        f"No fue posible obtener los datos: {api_status}"
    )


# =========================================================
# DATAFRAME
# =========================================================

df = events_to_dataframe(events)


# =========================================================
# KPIs
# =========================================================

if not df.empty:

    total_events = len(df)

    sports_count = df["Deporte"].nunique()

    leagues_count = df["Liga"].nunique()

    countries_count = df["País"].replace(
        "None",
        pd.NA
    ).dropna().nunique()

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


st.divider()


# =========================================================
# FILTROS LOCALES
# =========================================================

if not df.empty:

    st.markdown(
        "## 🔎 Filtros del Centro de Mando"
    )

    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:

        leagues = sorted(
            [
                x
                for x in df["Liga"].dropna().unique()
            ]
        )

        selected_league = st.selectbox(
            "🏆 Liga",
            ["Todas"] + leagues
        )

    with col_filter2:

        countries = sorted(
            [
                x
                for x in df["País"].dropna().unique()
            ]
        )

        selected_country = st.selectbox(
            "🌎 País",
            ["Todos"] + countries
        )

    filtered_df = df.copy()

    if selected_league != "Todas":

        filtered_df = filtered_df[
            filtered_df["Liga"] == selected_league
        ]

    if selected_country != "Todos":

        filtered_df = filtered_df[
            filtered_df["País"] == selected_country
        ]

else:

    filtered_df = df


# =========================================================
# PARTIDOS / EVENTOS
# =========================================================

st.divider()

st.markdown(
    "## 🏟️ Eventos deportivos reales"
)


if filtered_df.empty:

    st.info(
        "No hay eventos que coincidan con los filtros."
    )

else:

    st.caption(
        f"Mostrando {len(filtered_df)} eventos."
    )

    for _, row in filtered_df.iterrows():

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
# TABLA DE DATOS
# =========================================================

st.divider()

st.markdown(
    "## 📋 Base de datos recibida"
)

if not filtered_df.empty:

    display_columns = [
        "ID Evento",
        "Deporte",
        "Liga",
        "Evento",
        "Fecha",
        "Hora",
        "Estadio",
        "Ciudad",
        "País",
        "Estado",
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# INFORMACIÓN TÉCNICA
# =========================================================

st.divider()

st.markdown(
    "## 🧠 Arquitectura actual"
)

st.markdown(
    """
    ### CAPA 1 — Datos deportivos

    TheSportsDB proporciona:

    - Eventos
    - Equipos
    - Ligas
    - Fechas
    - Horarios
    - Estadios
    - Países
    - IDs de eventos

    ### CAPA 2 — Motor Quant

    🚧 En construcción.

    Esta capa será responsable de calcular posteriormente:

    - Probabilidades
    - Edge
    - EV
    - Value Score
    - Ranking de oportunidades

    ### CAPA 3 — Cuotas

    🚧 Pendiente.

    Aquí incorporaremos una fuente de cuotas de apuestas
    para comparar mercados y casas de apuestas.

    ### CAPA 4 — Centro de Mando

    Finalmente tendremos:

    **Datos → Modelo → Cuotas → Comparación → Value → Ranking**
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Centro de Mando Quant — Sports Data Hub | "
    "FASE 3 — Datos deportivos reales"
)

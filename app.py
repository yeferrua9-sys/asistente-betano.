"""
CENTRO DE MANDO QUANT — SPORTS DATA HUB
FASE 4A — MOTOR QUANT

Fuente deportiva:
TheSportsDB V1 — API gratuita

IMPORTANTE:
- Los eventos proceden de datos reales de TheSportsDB.
- No se inventan cuotas.
- No se inventan probabilidades.
- El motor Quant queda preparado para recibir cuotas reales.
- Las métricas solo se calculan cuando existen datos suficientes.

Arquitectura:

CAPA 1 — Datos deportivos
CAPA 2 — Motor Quant
CAPA 3 — Cuotas
CAPA 4 — Centro de Mando
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date
import math


# =========================================================
# CONFIGURACIÓN GENERAL
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

    .quant-box {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 18px;
        margin: 12px 0;
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
# THE SPORTS DB
# =========================================================

TSDB_BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"


# =========================================================
# OBTENER EVENTOS DEL DÍA
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
# CONSTRUIR DATAFRAME
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
# MOTOR QUANT — FUNCIONES BASE
# =========================================================

def implied_probability(decimal_odds):

    """
    Convierte una cuota decimal en probabilidad implícita.

    Ejemplo:
    cuota 2.00 -> 50%
    cuota 1.50 -> 66.67%
    """

    if decimal_odds is None:
        return None

    try:

        odds = float(decimal_odds)

        if odds <= 1:
            return None

        return 1 / odds

    except (ValueError, TypeError):

        return None


def calculate_edge(model_probability, market_probability):

    """
    Edge = probabilidad del modelo
            - probabilidad implícita del mercado.
    """

    if model_probability is None:
        return None

    if market_probability is None:
        return None

    return model_probability - market_probability


def calculate_ev(model_probability, decimal_odds):

    """
    EV simplificado para una apuesta de cuota decimal.

    EV = (probabilidad × cuota) - 1
    """

    if model_probability is None:
        return None

    if decimal_odds is None:
        return None

    try:

        odds = float(decimal_odds)

        if odds <= 1:
            return None

        return (model_probability * odds) - 1

    except (ValueError, TypeError):

        return None


def calculate_value_score(edge, ev):

    """
    Value Score interno.

    Se mantiene vacío cuando todavía no existen
    datos suficientes para hacer una evaluación real.
    """

    if edge is None or ev is None:
        return None

    score = (
        (edge * 100) * 60
        +
        (ev * 100) * 40
    )

    return round(score, 2)


def classify_value(value_score):

    """
    Clasificación descriptiva del Value Score.

    No constituye una recomendación automática.
    """

    if value_score is None:
        return "SIN DATOS"

    if value_score >= 10:
        return "VALUE ALTO"

    if value_score >= 5:
        return "VALUE MEDIO"

    if value_score > 0:
        return "VALUE BAJO"

    return "SIN VALUE"


def quant_analysis(
    model_probability=None,
    decimal_odds=None
):

    market_probability = implied_probability(
        decimal_odds
    )

    edge = calculate_edge(
        model_probability,
        market_probability
    )

    ev = calculate_ev(
        model_probability,
        decimal_odds
    )

    value_score = calculate_value_score(
        edge,
        ev
    )

    classification = classify_value(
        value_score
    )

    return {
        "model_probability": model_probability,
        "market_probability": market_probability,
        "edge": edge,
        "ev": ev,
        "value_score": value_score,
        "classification": classification,
    }


# =========================================================
# FORMATEAR MÉTRICAS
# =========================================================

def percentage(value):

    if value is None:
        return "—"

    return f"{value * 100:.2f}%"


def signed_percentage(value):

    if value is None:
        return "—"

    return f"{value * 100:+.2f}%"


def number(value):

    if value is None:
        return "—"

    return f"{value:.2f}"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Centro de Mando")

    st.caption(
        "Sports Data Hub — FASE 4A"
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
        Plataforma de análisis deportivo con datos reales
        y motor cuantitativo en construcción.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ESTADO GENERAL
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

    Los eventos mostrados proceden directamente
    de la fuente deportiva conectada.

    <br><br>

    <b>Motor Quant:</b> 🟢 ACTIVO

    <br>

    <b>Cuotas:</b> 🟡 PENDIENTES DE CONEXIÓN

    <br>

    <b>Probabilidades del modelo:</b>
    🟡 PENDIENTES DE DATOS / MODELO

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

    countries_count = (
        df["País"]
        .replace("None", pd.NA)
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
# PANEL MOTOR QUANT
# =========================================================

st.divider()

st.markdown(
    "## 🧠 Motor Quant"
)

st.markdown(
    """
    <div class="quant-box">

    <b>Estado:</b> 🟢 Motor Quant operativo

    <br><br>

    El motor ya dispone de las funciones matemáticas
    necesarias para trabajar con:

    <br><br>

    • Probabilidad del modelo<br>
    • Probabilidad implícita de mercado<br>
    • Edge<br>
    • EV<br>
    • Value Score<br>
    • Clasificación de valor

    <br><br>

    <b>Importante:</b> todavía no se generan números
    artificiales. Las métricas permanecen vacías hasta
    disponer de probabilidades del modelo y cuotas reales.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DEMOSTRACIÓN MATEMÁTICA DEL MOTOR
# =========================================================

with st.expander(
    "🔬 Ver funcionamiento matemático del Motor Quant"
):

    st.write(
        "Esta sección permite comprobar el motor sin "
        "introducirlo todavía en los partidos reales."
    )

    demo_probability = st.number_input(
        "Probabilidad hipotética del modelo (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

    demo_odds = st.number_input(
        "Cuota decimal hipotética",
        min_value=1.01,
        max_value=100.0,
        value=2.00,
        step=0.01
    )

    demo_model_probability = (
        demo_probability / 100
    )

    demo = quant_analysis(
        model_probability=demo_model_probability,
        decimal_odds=demo_odds
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Prob. modelo",
        percentage(
            demo["model_probability"]
        )
    )

    d2.metric(
        "Prob. mercado",
        percentage(
            demo["market_probability"]
        )
    )

    d3.metric(
        "Edge",
        signed_percentage(
            demo["edge"]
        )
    )

    d4.metric(
        "EV",
        signed_percentage(
            demo["ev"]
        )
    )

    st.write(
        f"**Value Score:** "
        f"{number(demo['value_score'])}"
    )

    st.write(
        f"**Clasificación:** "
        f"{demo['classification']}"
    )

    st.caption(
        "Los valores de esta sección son únicamente "
        "una demostración matemática y no representan "
        "una recomendación de apuesta."
    )


# =========================================================
# EVENTOS DEPORTIVOS
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
# ARQUITECTURA
# =========================================================

st.divider()

st.markdown(
    "## 🧠 Arquitectura del sistema"
)

st.markdown(
    """
    ### CAPA 1 — Datos deportivos 🟢

    TheSportsDB proporciona:

    - Eventos
    - Equipos
    - Ligas
    - Fechas
    - Horarios
    - Estadios
    - Países
    - IDs de eventos

    ---

    ### CAPA 2 — Motor Quant 🟢

    El motor ya está preparado para calcular:

    - Probabilidad implícita
    - Edge
    - EV
    - Value Score
    - Clasificación de valor

    **Todavía no recibe probabilidades propias ni cuotas
    reales, por lo que no genera recomendaciones.**

    ---

    ### CAPA 3 — Cuotas 🟡

    Pendiente de conectar una fuente real de cuotas.

    Esta capa proporcionará los precios de mercado
    necesarios para comparar contra el modelo.

    ---

    ### CAPA 4 — Centro de Mando 🟢

    El objetivo final será:

    **Datos → Modelo → Cuotas → Comparación → Edge → EV → Value → Ranking**
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Centro de Mando Quant — Sports Data Hub | "
    "FASE 4A — Motor Quant"
)

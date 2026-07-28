import streamlit as st
import pandas as pd
import requests
import numpy as np

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


# ============================================================
# ESTILO
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
    font-size: 21px;
    font-weight: 700;
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
# UTILIDADES
# ============================================================

def safe_value(value, fallback="No disponible"):

    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass

    if str(value).strip() == "":
        return fallback

    return value


def format_date_display(value):

    if not value:
        return "No disponible"

    try:
        parsed = datetime.strptime(
            str(value),
            "%Y-%m-%d"
        )

        return parsed.strftime("%d/%m/%Y")

    except Exception:
        return str(value)


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
# THE SPORTS DB — EVENTOS DEL DÍA
# ============================================================

@st.cache_data(ttl=300)
def get_events_day(selected_date, sport_filter):

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
            return [], f"HTTP_{response.status_code}"

        data = response.json()

        events = data.get("events") or []

        if not events:
            return [], "NO_EVENTS"

        return events, "OK"

    except requests.RequestException as error:

        return [], f"CONNECTION_ERROR: {error}"

    except ValueError:

        return [], "JSON_ERROR"


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

            "Deporte": event.get(
                "strSport",
                "No disponible"
            ),

            "Liga": event.get(
                "strLeague",
                "No disponible"
            ),

            "IDLiga": event.get(
                "idLeague"
            ),

            "Evento": matchup,

            "Fecha": event.get(
                "dateEvent",
                ""
            ),

            "Hora": format_time(event),

            "Local": home or "",

            "Visitante": away or "",

            "IDLocal": event.get("idHomeTeam"),

            "IDVisitante": event.get("idAwayTeam"),

            "Estadio": event.get(
                "strVenue",
                ""
            ),

            "Ciudad": event.get(
                "strCity",
                ""
            ),

            "País": event.get(
                "strCountry",
                ""
            ),

            "Temporada": event.get(
                "strSeason",
                ""
            ),

            "Ronda": event.get(
                "intRound",
                ""
            ),

            "Estado": event.get(
                "strStatus",
                ""
            ),

            "ResultadoLocal": event.get(
                "intHomeScore"
            ),

            "ResultadoVisitante": event.get(
                "intAwayScore"
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# HISTÓRICO DE EQUIPO
# ============================================================

@st.cache_data(ttl=3600)
def get_team_history(team_id):

    if not team_id:
        return [], "NO_TEAM_ID"

    url = f"{TSDB_BASE_URL}/eventslast.php"

    params = {
        "id": team_id
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        if response.status_code != 200:
            return [], f"HTTP_{response.status_code}"

        data = response.json()

        events = data.get("results") or []

        if not events:
            return [], "NO_HISTORY"

        return events, "OK"

    except requests.RequestException as error:

        return [], f"CONNECTION_ERROR: {error}"

    except ValueError:

        return [], "JSON_ERROR"


# ============================================================
# HISTÓRICO → DATAFRAME
# ============================================================

def history_to_dataframe(events):

    rows = []

    for event in events:

        home = event.get("strHomeTeam")
        away = event.get("strAwayTeam")

        home_score = event.get("intHomeScore")
        away_score = event.get("intAwayScore")

        try:
            home_score = int(home_score)
            away_score = int(away_score)
        except (TypeError, ValueError):
            home_score = None
            away_score = None

        rows.append({

            "ID": event.get("idEvent"),

            "Fecha": event.get(
                "dateEvent",
                ""
            ),

            "Liga": event.get(
                "strLeague",
                ""
            ),

            "Temporada": event.get(
                "strSeason",
                ""
            ),

            "Local": home or "",

            "Visitante": away or "",

            "GolesLocal": home_score,

            "GolesVisitante": away_score,

        })

    df = pd.DataFrame(rows)

    if not df.empty:

        df["Fecha"] = pd.to_datetime(
            df["Fecha"],
            errors="coerce"
        )

        df = df.sort_values(
            "Fecha",
            ascending=False
        )

    return df


# ============================================================
# ESTADÍSTICAS DE FORMA
# ============================================================

def calculate_team_form(history_df, team_id, team_name, n=5):

    if history_df.empty:
        return None

    relevant = history_df[
        (
            history_df["Local"] == team_name
        )
        |
        (
            history_df["Visitante"] == team_name
        )
    ].copy()

    relevant = relevant.dropna(
        subset=[
            "GolesLocal",
            "GolesVisitante"
        ]
    )

    if relevant.empty:
        return None

    relevant = relevant.head(n)

    points = 0
    wins = 0
    draws = 0
    losses = 0

    goals_for = 0
    goals_against = 0

    home_matches = 0
    away_matches = 0

    for _, row in relevant.iterrows():

        is_home = row["Local"] == team_name

        if is_home:

            gf = row["GolesLocal"]
            ga = row["GolesVisitante"]

            home_matches += 1

        else:

            gf = row["GolesVisitante"]
            ga = row["GolesLocal"]

            away_matches += 1

        goals_for += gf
        goals_against += ga

        if gf > ga:

            wins += 1
            points += 3

        elif gf == ga:

            draws += 1
            points += 1

        else:

            losses += 1

    matches = len(relevant)

    if matches == 0:
        return None

    return {

        "Equipo": team_name,

        "Partidos": matches,

        "Victorias": wins,

        "Empates": draws,

        "Derrotas": losses,

        "Puntos": points,

        "Puntos_por_partido":
            points / matches,

        "Goles_favor":
            goals_for,

        "Goles_contra":
            goals_against,

        "GF_por_partido":
            goals_for / matches,

        "GC_por_partido":
            goals_against / matches,

        "Diferencia_goles":
            goals_for - goals_against,

        "Local_partidos":
            home_matches,

        "Visitante_partidos":
            away_matches,

    }


# ============================================================
# DATASET DEL EVENTO
# ============================================================

def build_match_features(
    selected,
    home_history,
    away_history
):

    home_name = selected["Local"]
    away_name = selected["Visitante"]

    home_form = calculate_team_form(
        home_history,
        selected["IDLocal"],
        home_name,
        n=5
    )

    away_form = calculate_team_form(
        away_history,
        selected["IDVisitante"],
        away_name,
        n=5
    )

    if home_form is None or away_form is None:
        return None

    features = {

        "Partido":
            f"{home_name} vs {away_name}",

        "Local":
            home_name,

        "Visitante":
            away_name,

        "Local_PPP":
            home_form["Puntos_por_partido"],

        "Visitante_PPP":
            away_form["Puntos_por_partido"],

        "Local_GF":
            home_form["GF_por_partido"],

        "Visitante_GF":
            away_form["GF_por_partido"],

        "Local_GC":
            home_form["GC_por_partido"],

        "Visitante_GC":
            away_form["GC_por_partido"],

        "Local_DG":
            home_form["Diferencia_goles"],

        "Visitante_DG":
            away_form["Diferencia_goles"],

        "Local_Victorias":
            home_form["Victorias"],

        "Visitante_Victorias":
            away_form["Victorias"],

        "Local_Derrotas":
            home_form["Derrotas"],

        "Visitante_Derrotas":
            away_form["Derrotas"],

        "Local_Muestra":
            home_form["Partidos"],

        "Visitante_Muestra":
            away_form["Partidos"],
    }

    return features


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Centro de Mando")

    st.caption(
        "Sports Data Hub — FASE 8"
    )

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

    st.markdown("### 🧠 Histórico")

    history_window = st.slider(
        "Partidos recientes",
        min_value=3,
        max_value=10,
        value=5
    )

    st.caption(
        "La fase actual utiliza los últimos "
        "partidos disponibles de cada equipo."
    )

    st.divider()

    if st.button(
        "🔄 Actualizar datos",
        use_container_width=True
    ):

        st.cache_data.clear()

        st.session_state.pop(
            "selected_event_id",
            None
        )

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📊 Centro de Mando Quant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sports Data Hub · Datos reales · '
    'Histórico deportivo · Motor Quant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EVENTOS
# ============================================================

events, event_status = get_events_day(
    selected_date.strftime("%Y-%m-%d"),
    sport_filter
)

df_events = events_to_dataframe(events)


# ============================================================
# ESTADOS
# ============================================================

status1, status2, status3 = st.columns(3)

with status1:

    st.markdown(
        '<div class="status status-green">'
        '🟢 DATOS DEPORTIVOS'
        '</div>',
        unsafe_allow_html=True
    )

with status2:

    st.markdown(
        '<div class="status status-green">'
        '🟢 HISTÓRICO DISPONIBLE'
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
# MENSAJES
# ============================================================

if event_status.startswith("CONNECTION_ERROR"):

    st.error(
        "No fue posible conectar con TheSportsDB."
    )

elif event_status == "NO_EVENTS":

    st.info(
        f"No se encontraron eventos para "
        f"{selected_date.strftime('%d/%m/%Y')}."
    )

elif event_status != "OK":

    st.warning(
        f"Estado de la fuente deportiva: "
        f"{event_status}"
    )


# ============================================================
# KPIs
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📈 Resumen'
    '</div>',
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
    '<div class="section-title">'
    '🔎 Filtrar partidos'
    '</div>',
    unsafe_allow_html=True
)

if not df_events.empty:

    f1, f2 = st.columns(2)

    with f1:

        league_options = sorted(
            [
                x for x in
                df_events["Liga"]
                .dropna()
                .unique()
                if str(x).strip()
            ]
        )

        selected_league = st.selectbox(
            "🏆 Competición",
            ["Todas"] + league_options
        )

    with f2:

        country_options = sorted(
            [
                x for x in
                df_events["País"]
                .dropna()
                .unique()
                if str(x).strip()
            ]
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
    '<div class="section-title">'
    '🏟️ Partidos y eventos'
    '</div>',
    unsafe_allow_html=True
)

if filtered.empty:

    st.info(
        "No hay eventos deportivos."
    )

else:

    for _, row in filtered.iterrows():

        event_id = row["ID"]

        st.markdown(
            '<div class="match-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="gray">'
            f'{safe_value(row["Liga"], "Competición")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="team">'
            f'{safe_value(row["Evento"], "Evento deportivo")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="gray">'
            f'📅 {format_date_display(row["Fecha"])}'
            f' &nbsp;&nbsp; '
            f'⏰ {safe_value(row["Hora"], "--")}'
            f'</div>',
            unsafe_allow_html=True
        )

        location_parts = []

        for field in [
            "Estadio",
            "Ciudad",
            "País"
        ]:

            value = safe_value(
                row[field],
                ""
            )

            if value:
                location_parts.append(
                    value
                )

        location = " · ".join(
            location_parts
        )

        st.markdown(
            f'<div class="gray">'
            f'📍 {location or "Ubicación no disponible"}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "🔍 Analizar histórico",
            key=f"analyze_{event_id}"
        ):

            st.session_state[
                "selected_event_id"
            ] = event_id

            st.rerun()


# ============================================================
# EVENTO SELECCIONADO
# ============================================================

selected_event_id = st.session_state.get(
    "selected_event_id"
)

if selected_event_id is not None:

    selected_rows = df_events[
        df_events["ID"] == selected_event_id
    ]

    if not selected_rows.empty:

        selected = selected_rows.iloc[0]

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🎯 Análisis del evento'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="card">'
            f'<div class="team">'
            f'{safe_value(selected["Evento"])}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # DATOS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Datos del evento'
            '</div>',
            unsafe_allow_html=True
        )

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Deporte",
            safe_value(selected["Deporte"])
        )

        a2.metric(
            "Competición",
            safe_value(selected["Liga"])
        )

        a3.metric(
            "Fecha",
            format_date_display(selected["Fecha"])
        )

        a4.metric(
            "Hora",
            safe_value(selected["Hora"], "--")
        )

        # ====================================================
        # HISTÓRICO LOCAL
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📚 Histórico del local'
            '</div>',
            unsafe_allow_html=True
        )

        home_history_raw, home_status = get_team_history(
            selected["IDLocal"]
        )

        home_history = history_to_dataframe(
            home_history_raw
        )

        # ====================================================
        # HISTÓRICO VISITANTE
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📚 Histórico del visitante'
            '</div>',
            unsafe_allow_html=True
        )

        away_history_raw, away_status = get_team_history(
            selected["IDVisitante"]
        )

        away_history = history_to_dataframe(
            away_history_raw
        )

        # ====================================================
        # ESTADO HISTÓRICO
        # ====================================================

        h1, h2 = st.columns(2)

        with h1:

            if home_status == "OK":

                st.success(
                    f"Histórico disponible: "
                    f"{len(home_history)} registros"
                )

            else:

                st.warning(
                    f"Histórico local: "
                    f"{home_status}"
                )

        with h2:

            if away_status == "OK":

                st.success(
                    f"Histórico disponible: "
                    f"{len(away_history)} registros"
                )

            else:

                st.warning(
                    f"Histórico visitante: "
                    f"{away_status}"
                )

        # ====================================================
        # FORMA
        # ====================================================

        home_form = calculate_team_form(
            home_history,
            selected["IDLocal"],
            selected["Local"],
            history_window
        )

        away_form = calculate_team_form(
            away_history,
            selected["IDVisitante"],
            selected["Visitante"],
            history_window
        )

        if home_form and away_form:

            st.markdown(
                '<div class="section-title">'
                '📊 Variables de forma'
                '</div>',
                unsafe_allow_html=True
            )

            form_df = pd.DataFrame([

                {
                    "Variable": "Partidos analizados",
                    selected["Local"]:
                        home_form["Partidos"],
                    selected["Visitante"]:
                        away_form["Partidos"],
                },

                {
                    "Variable": "Puntos por partido",
                    selected["Local"]:
                        round(
                            home_form["Puntos_por_partido"],
                            3
                        ),
                    selected["Visitante"]:
                        round(
                            away_form["Puntos_por_partido"],
                            3
                        ),
                },

                {
                    "Variable": "Goles por partido",
                    selected["Local"]:
                        round(
                            home_form["GF_por_partido"],
                            3
                        ),
                    selected["Visitante"]:
                        round(
                            away_form["GF_por_partido"],
                            3
                        ),
                },

                {
                    "Variable": "Goles recibidos/partido",
                    selected["Local"]:
                        round(
                            home_form["GC_por_partido"],
                            3
                        ),
                    selected["Visitante"]:
                        round(
                            away_form["GC_por_partido"],
                            3
                        ),
                },

                {
                    "Variable": "Victorias",
                    selected["Local"]:
                        home_form["Victorias"],
                    selected["Visitante"]:
                        away_form["Victorias"],
                },

                {
                    "Variable": "Empates",
                    selected["Local"]:
                        home_form["Empates"],
                    selected["Visitante"]:
                        away_form["Empates"],
                },

                {
                    "Variable": "Derrotas",
                    selected["Local"]:
                        home_form["Derrotas"],
                    selected["Visitante"]:
                        away_form["Derrotas"],
                },

                {
                    "Variable": "Diferencia de goles",
                    selected["Local"]:
                        home_form["Diferencia_goles"],
                    selected["Visitante"]:
                        away_form["Diferencia_goles"],
                },

            ])

            st.dataframe(
                form_df,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # DATASET DEL MODELO
            # =================================================

            features = build_match_features(
                selected,
                home_history,
                away_history
            )

            if features:

                st.markdown(
                    '<div class="section-title">'
                    '🧠 Dataset preparado para modelo'
                    '</div>',
                    unsafe_allow_html=True
                )

                feature_df = pd.DataFrame(
                    [features]
                )

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    "🟢 Las variables históricas "
                    "fueron calculadas desde datos "
                    "obtenidos de TheSportsDB."
                )

            else:

                st.warning(
                    "No hay suficiente histórico "
                    "válido para construir las variables."
                )

        else:

            st.warning(
                "No se pudo construir la forma "
                "reciente de ambos equipos."
            )

        # ====================================================
        # CUOTAS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '💰 Mercado'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Las cuotas continúan fuera de esta fase. "
            "No se mostrarán valores inventados."
        )

        # ====================================================
        # MODELO
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🧠 Modelo predictivo'
            '</div>',
            unsafe_allow_html=True
        )

        st.warning(
            "🟡 Modelo todavía no entrenado. "
            "La FASE 8 prepara las variables reales "
            "que alimentarán el entrenamiento."
        )

        # ====================================================
        # VALUE
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📈 Value Betting'
            '</div>',
            unsafe_allow_html=True
        )

        v1, v2, v3 = st.columns(3)

        v1.metric(
            "Edge",
            "Pendiente"
        )

        v2.metric(
            "EV",
            "Pendiente"
        )

        v3.metric(
            "Value Score",
            "Pendiente"
        )


# ============================================================
# INTEGRIDAD
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🔐 Integridad del sistema'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <b>Reglas del Centro de Mando</b>

    <br><br>

    🟢 Datos deportivos obtenidos de fuente real.

    <br><br>

    🟢 Histórico obtenido de TheSportsDB.

    <br><br>

    🟢 Variables calculadas a partir de datos históricos.

    <br><br>

    🔒 No se inventan cuotas.

    <br>

    🔒 No se inventan probabilidades.

    <br>

    🔒 No se fabrica Value Score.

    <br>

    🔒 No se utiliza una estimación manual como
    probabilidad de modelo.

    <br><br>

    El modelo predictivo se activará únicamente
    cuando exista un dataset suficiente y
    pueda ser entrenado y validado correctamente.
    """,
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ARQUITECTURA
# ============================================================

with st.expander(
    "🏗️ Ver arquitectura del sistema"
):

    st.markdown("""
    ### CAPA 1 — Datos deportivos

    TheSportsDB

    Eventos → Equipos → Ligas → Fechas

    🟢 Activa

    ---

    ### CAPA 2 — Histórico

    Partidos anteriores → resultados → forma

    🟢 Activa

    ---

    ### CAPA 3 — Feature Engineering

    Forma reciente
    → goles
    → puntos
    → victorias
    → derrotas
    → diferencia de goles

    🟢 Activa

    ---

    ### CAPA 4 — Dataset

    Partido
    ↓
    Variables históricas
    ↓
    Dataset de entrenamiento

    🟢 Preparado

    ---

    ### CAPA 5 — Modelo

    Variables
    ↓
    Modelo estadístico
    ↓
    Probabilidad propia

    🟡 Siguiente fase

    ---

    ### CAPA 6 — Mercado

    Cuotas reales
    ↓
    Probabilidad implícita
    ↓
    Comparación

    🟡 Pendiente

    ---

    ### CAPA 7 — Motor Quant

    Probabilidad propia
    ↓
    Edge
    ↓
    EV
    ↓
    Value Score
    ↓
    Ranking

    🟡 Pendiente

    ---

    ### CAPA 8 — Dashboard

    Información real
    + modelo
    + mercado
    + motor Quant

    🟡 En construcción
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE 8"
)

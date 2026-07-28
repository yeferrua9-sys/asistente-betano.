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

.green {
    color: #10b981;
}

.yellow {
    color: #f59e0b;
}

.red {
    color: #ef4444;
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
# PETICIÓN GENÉRICA
# ============================================================

def tsdb_request(endpoint, params=None):

    url = f"{TSDB_BASE_URL}/{endpoint}"

    try:

        response = requests.get(
            url,
            params=params or {},
            timeout=15
        )

        if response.status_code != 200:
            return None, f"HTTP_{response.status_code}"

        return response.json(), "OK"

    except requests.RequestException as error:

        return None, f"CONNECTION_ERROR: {error}"

    except ValueError:

        return None, "JSON_ERROR"


# ============================================================
# EVENTOS DEL DÍA
# ============================================================

@st.cache_data(ttl=300)
def get_events_day(selected_date, sport_filter):

    params = {
        "d": selected_date
    }

    if sport_filter != "Todos":
        params["s"] = sport_filter

    data, status = tsdb_request(
        "eventsday.php",
        params
    )

    if status != "OK":
        return [], status

    events = data.get("events") or []

    if not events:
        return [], "NO_EVENTS"

    return events, "OK"


# ============================================================
# EVENTOS DE EQUIPO — HISTORIAL
# ============================================================

@st.cache_data(ttl=600)
def get_team_history(team_id):

    if not team_id:
        return [], "NO_TEAM_ID"

    data, status = tsdb_request(
        "eventslast.php",
        {
            "id": team_id
        }
    )

    if status != "OK":
        return [], status

    events = data.get("results") or data.get("events") or []

    return events, "OK"


# ============================================================
# RESULTADOS DE EVENTO
# ============================================================

@st.cache_data(ttl=600)
def get_event_results(event_id):

    if not event_id:
        return [], "NO_EVENT_ID"

    data, status = tsdb_request(
        "eventresults.php",
        {
            "id": event_id
        }
    )

    if status != "OK":
        return [], status

    results = data.get("results") or []

    return results, "OK"


# ============================================================
# ESTADÍSTICAS DEL EVENTO
# ============================================================

@st.cache_data(ttl=600)
def get_event_stats(event_id):

    if not event_id:
        return [], "NO_EVENT_ID"

    data, status = tsdb_request(
        "lookupeventstats.php",
        {
            "id": event_id
        }
    )

    if status != "OK":
        return [], status

    stats = data.get("eventstats") or []

    return stats, "OK"


# ============================================================
# FORMATO HORA
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

            "Deporte": event.get(
                "strSport",
                "No disponible"
            ),

            "Liga": event.get(
                "strLeague",
                "No disponible"
            ),

            "Evento": matchup,

            "Fecha": event.get(
                "dateEvent",
                ""
            ),

            "Hora": format_time(event),

            "Local": home or "",

            "Visitante": away or "",

            "IDLocal": event.get(
                "idHomeTeam"
            ),

            "IDVisitante": event.get(
                "idAwayTeam"
            ),

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
# HISTORIAL → DATAFRAME
# ============================================================

def history_to_dataframe(events, team_id):

    rows = []

    for event in events:

        home_id = event.get("idHomeTeam")
        away_id = event.get("idAwayTeam")

        home = event.get("strHomeTeam", "")
        away = event.get("strAwayTeam", "")

        try:

            home_score = (
                int(event["intHomeScore"])
                if event.get("intHomeScore") not in [None, ""]
                else None
            )

        except (ValueError, TypeError):

            home_score = None

        try:

            away_score = (
                int(event["intAwayScore"])
                if event.get("intAwayScore") not in [None, ""]
                else None
            )

        except (ValueError, TypeError):

            away_score = None

        if home_score is None or away_score is None:
            continue

        if str(home_id) == str(team_id):

            team_side = "Home"

            team_goals = home_score
            opponent_goals = away_score

            opponent = away

            if home_score > away_score:
                result = "W"
            elif home_score == away_score:
                result = "D"
            else:
                result = "L"

        else:

            team_side = "Away"

            team_goals = away_score
            opponent_goals = home_score

            opponent = home

            if away_score > home_score:
                result = "W"
            elif away_score == home_score:
                result = "D"
            else:
                result = "L"

        rows.append({

            "Fecha": event.get(
                "dateEvent"
            ),

            "Equipo": event.get(
                "strHomeTeam"
            )
            if str(home_id) == str(team_id)
            else event.get("strAwayTeam"),

            "Rival": opponent,

            "LocalVisitante": team_side,

            "GF": team_goals,

            "GC": opponent_goals,

            "Resultado": result,

            "Liga": event.get(
                "strLeague",
                ""
            ),

            "IDEvento": event.get(
                "idEvent"
            ),
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
# MÉTRICAS HISTÓRICAS
# ============================================================

def calculate_team_metrics(df):

    if df.empty:

        return {

            "Partidos": 0,
            "Victorias": 0,
            "Empates": 0,
            "Derrotas": 0,
            "GF": 0,
            "GC": 0,
            "GF_Partido": 0,
            "GC_Partido": 0,
            "Puntos": 0,
            "PPG": 0,

        }

    matches = len(df)

    wins = int(
        (df["Resultado"] == "W").sum()
    )

    draws = int(
        (df["Resultado"] == "D").sum()
    )

    losses = int(
        (df["Resultado"] == "L").sum()
    )

    goals_for = float(
        df["GF"].sum()
    )

    goals_against = float(
        df["GC"].sum()
    )

    points = (
        wins * 3
        + draws
    )

    return {

        "Partidos": matches,

        "Victorias": wins,

        "Empates": draws,

        "Derrotas": losses,

        "GF": goals_for,

        "GC": goals_against,

        "GF_Partido":
            round(goals_for / matches, 2),

        "GC_Partido":
            round(goals_against / matches, 2),

        "Puntos": points,

        "PPG":
            round(points / matches, 2),

    }


# ============================================================
# MÉTRICAS RECIENTES
# ============================================================

def recent_form(df, n=5):

    if df.empty:
        return []

    return (
        df.head(n)["Resultado"]
        .tolist()
    )


# ============================================================
# SAFE VALUE
# ============================================================

def safe_value(
    value,
    fallback="No disponible"
):

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


# ============================================================
# FECHA
# ============================================================

def format_date_display(value):

    if not value:
        return "No disponible"

    try:

        parsed = datetime.strptime(
            str(value),
            "%Y-%m-%d"
        )

        return parsed.strftime(
            "%d/%m/%Y"
        )

    except Exception:

        return str(value)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Centro de Mando"
    )

    st.caption(
        "Sports Data Hub — FASE 7"
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

df_events = events_to_dataframe(
    events
)


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
# ERROR
# ============================================================

if event_status == "NO_EVENTS":

    st.info(
        "No hay eventos para la fecha seleccionada."
    )

elif event_status != "OK":

    st.error(
        f"Error obteniendo datos: {event_status}"
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

if not df_events.empty:

    st.markdown(
        '<div class="section-title">'
        '🔎 Filtrar partidos'
        '</div>',
        unsafe_allow_html=True
    )

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
            filtered["Liga"]
            == selected_league
        ]

    if selected_country != "Todos":

        filtered = filtered[
            filtered["País"]
            == selected_country
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
        "No hay partidos disponibles."
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
            f'{safe_value(row["Liga"])}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="team">'
            f'{safe_value(row["Evento"])}'
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

        st.markdown(
            f'<div class="gray">'
            f'📍 {safe_value(row["Estadio"])}'
            f' · '
            f'{safe_value(row["País"])}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "🔍 Analizar evento",
            key=f"analyze_{event_id}"
        ):

            st.session_state[
                "selected_event_id"
            ] = event_id

            st.rerun()


# ============================================================
# ANÁLISIS
# ============================================================

selected_event_id = st.session_state.get(
    "selected_event_id"
)

if selected_event_id is not None:

    selected_rows = df_events[
        df_events["ID"]
        == selected_event_id
    ]

    if not selected_rows.empty:

        selected = selected_rows.iloc[0]

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🎯 Análisis cuantitativo'
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
        # HISTORIAL LOCAL
        # ====================================================

        home_id = selected["IDLocal"]

        away_id = selected["IDVisitante"]

        home_history, home_status = (
            get_team_history(home_id)
        )

        away_history, away_status = (
            get_team_history(away_id)
        )

        home_df = history_to_dataframe(
            home_history,
            home_id
        )

        away_df = history_to_dataframe(
            away_history,
            away_id
        )

        # ====================================================
        # MÉTRICAS
        # ====================================================

        home_metrics = calculate_team_metrics(
            home_df
        )

        away_metrics = calculate_team_metrics(
            away_df
        )

        st.markdown(
            '<div class="section-title">'
            '📊 Rendimiento histórico'
            '</div>',
            unsafe_allow_html=True
        )

        metric_table = pd.DataFrame({

            "Métrica": [
                "Partidos",
                "Victorias",
                "Empates",
                "Derrotas",
                "Goles a favor",
                "Goles en contra",
                "GF / partido",
                "GC / partido",
                "Puntos",
                "Puntos / partido",
            ],

            safe_value(
                selected["Local"],
                "Local"
            ): [
                home_metrics["Partidos"],
                home_metrics["Victorias"],
                home_metrics["Empates"],
                home_metrics["Derrotas"],
                home_metrics["GF"],
                home_metrics["GC"],
                home_metrics["GF_Partido"],
                home_metrics["GC_Partido"],
                home_metrics["Puntos"],
                home_metrics["PPG"],
            ],

            safe_value(
                selected["Visitante"],
                "Visitante"
            ): [
                away_metrics["Partidos"],
                away_metrics["Victorias"],
                away_metrics["Empates"],
                away_metrics["Derrotas"],
                away_metrics["GF"],
                away_metrics["GC"],
                away_metrics["GF_Partido"],
                away_metrics["GC_Partido"],
                away_metrics["Puntos"],
                away_metrics["PPG"],
            ],
        })

        st.dataframe(
            metric_table,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # FORMA
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🔥 Forma reciente'
            '</div>',
            unsafe_allow_html=True
        )

        form1, form2 = st.columns(2)

        with form1:

            st.markdown(
                f"### 🏠 {safe_value(selected['Local'])}"
            )

            form = recent_form(
                home_df,
                5
            )

            if form:

                st.write(
                    " → ".join(form)
                )

            else:

                st.info(
                    "No hay suficiente historial."
                )

        with form2:

            st.markdown(
                f"### ✈️ {safe_value(selected['Visitante'])}"
            )

            form = recent_form(
                away_df,
                5
            )

            if form:

                st.write(
                    " → ".join(form)
                )

            else:

                st.info(
                    "No hay suficiente historial."
                )

        # ====================================================
        # DATASET
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🧮 Variables disponibles para el modelo'
            '</div>',
            unsafe_allow_html=True
        )

        feature_data = {

            "home_matches":
                home_metrics["Partidos"],

            "away_matches":
                away_metrics["Partidos"],

            "home_win_rate":
                round(
                    home_metrics["Victorias"]
                    / home_metrics["Partidos"],
                    4
                )
                if home_metrics["Partidos"]
                else None,

            "away_win_rate":
                round(
                    away_metrics["Victorias"]
                    / away_metrics["Partidos"],
                    4
                )
                if away_metrics["Partidos"]
                else None,

            "home_draw_rate":
                round(
                    home_metrics["Empates"]
                    / home_metrics["Partidos"],
                    4
                )
                if home_metrics["Partidos"]
                else None,

            "away_draw_rate":
                round(
                    away_metrics["Empates"]
                    / away_metrics["Partidos"],
                    4
                )
                if away_metrics["Partidos"]
                else None,

            "home_goals_for_avg":
                home_metrics["GF_Partido"],

            "away_goals_for_avg":
                away_metrics["GF_Partido"],

            "home_goals_against_avg":
                home_metrics["GC_Partido"],

            "away_goals_against_avg":
                away_metrics["GC_Partido"],

            "home_points_per_game":
                home_metrics["PPG"],

            "away_points_per_game":
                away_metrics["PPG"],
        }

        feature_df = pd.DataFrame(
            [feature_data]
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
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

        st.warning(
            "🟡 Cuotas pendientes de conectar. "
            "No se calculará EV ni Value sin una fuente "
            "real de mercado."
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

        st.info(
            "Las variables históricas ya están siendo "
            "preparadas. El siguiente paso será entrenar "
            "el modelo estadístico con un dataset histórico."
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
            "Probabilidad modelo",
            "—"
        )

        v2.metric(
            "Edge",
            "—"
        )

        v3.metric(
            "EV",
            "—"
        )

        st.caption(
            "Bloqueado hasta disponer de modelo "
            "estadístico y cuotas reales."
        )


# ============================================================
# INTEGRIDAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔐 Integridad del sistema'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">

    🟢 Datos deportivos: reales.

    <br><br>

    🟢 Datos históricos: obtenidos desde TheSportsDB.

    <br><br>

    🔒 No se inventan cuotas.

    <br>

    🔒 No se inventan probabilidades.

    <br>

    🔒 No se fabrica Value.

    <br><br>

    El modelo solo utilizará variables derivadas
    de datos disponibles y trazables.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ARQUITECTURA
# ============================================================

with st.expander(
    "🏗️ Ver arquitectura del sistema"
):

    st.markdown("""
    ## FASE 7

    ### CAPA 1 — Eventos

    TheSportsDB

    ↓

    Partidos reales

    ### CAPA 2 — Histórico

    Equipos

    ↓

    Resultados anteriores

    ↓

    GF / GC / W / D / L

    ↓

    Forma reciente

    ### CAPA 3 — Feature Engineering

    Variables del local

    +

    Variables del visitante

    ↓

    Dataset predictivo

    ### CAPA 4 — Modelo

    Dataset histórico

    ↓

    Entrenamiento

    ↓

    Probabilidad 1X2

    ### CAPA 5 — Mercado

    Cuotas reales

    ↓

    Probabilidad implícita

    ### CAPA 6 — Quant

    Probabilidad modelo

    ↓

    Edge

    ↓

    EV

    ↓

    Value Score

    ↓

    Ranking
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE 7"
)

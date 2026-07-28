import streamlit as st
import pandas as pd
import requests
import numpy as np

from datetime import datetime, date
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss


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

# Número mínimo de partidos históricos para intentar entrenar
MIN_TRAINING_ROWS = 15

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

.status-red {
    background: #3c1010;
    color: #f87171;
    border: 1px solid #ef4444;
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


def normalize_score(value):

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
                ""
            ),

            "Liga": event.get(
                "strLeague",
                ""
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

            "ResultadoLocal":
                normalize_score(
                    event.get("intHomeScore")
                ),

            "ResultadoVisitante":
                normalize_score(
                    event.get("intAwayScore")
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

        home_score = normalize_score(
            event.get("intHomeScore")
        )

        away_score = normalize_score(
            event.get("intAwayScore")
        )

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

        df = df.dropna(
            subset=["Fecha"]
        )

        df = df.sort_values(
            "Fecha",
            ascending=False
        )

    return df


# ============================================================
# FORMA DEL EQUIPO
# ============================================================

def calculate_team_form(
    history_df,
    team_name,
    n=5,
    before_date=None
):

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

    if before_date is not None:

        before_date = pd.to_datetime(
            before_date
        )

        relevant = relevant[
            relevant["Fecha"] < before_date
        ]

    relevant = relevant.dropna(
        subset=[
            "GolesLocal",
            "GolesVisitante"
        ]
    )

    relevant = relevant.sort_values(
        "Fecha",
        ascending=False
    )

    relevant = relevant.head(n)

    if relevant.empty:
        return None

    points = 0
    wins = 0
    draws = 0
    losses = 0

    goals_for = 0
    goals_against = 0

    home_matches = 0
    away_matches = 0

    for _, row in relevant.iterrows():

        is_home = (
            row["Local"] == team_name
        )

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

    return {

        "Partidos": matches,

        "Victorias": wins,

        "Empates": draws,

        "Derrotas": losses,

        "Puntos": points,

        "PPP": points / matches,

        "GF": goals_for / matches,

        "GC": goals_against / matches,

        "DG": (
            goals_for - goals_against
        ) / matches,

    }


# ============================================================
# RESULTADO DEL PARTIDO
# ============================================================

def get_match_target(
    home_score,
    away_score
):

    if (
        home_score is None
        or away_score is None
    ):
        return None

    if home_score > away_score:
        return "H"

    if home_score < away_score:
        return "A"

    return "D"


# ============================================================
# VARIABLES PRE-PARTIDO
# ============================================================

def build_pre_match_features(
    match,
    home_history,
    away_history,
    window=5
):

    home_name = match["Local"]
    away_name = match["Visitante"]

    match_date = match["Fecha"]

    home_form = calculate_team_form(
        home_history,
        home_name,
        n=window,
        before_date=match_date
    )

    away_form = calculate_team_form(
        away_history,
        away_name,
        n=window,
        before_date=match_date
    )

    if (
        home_form is None
        or away_form is None
    ):
        return None

    return {

        "home_ppp":
            home_form["PPP"],

        "away_ppp":
            away_form["PPP"],

        "home_gf":
            home_form["GF"],

        "away_gf":
            away_form["GF"],

        "home_gc":
            home_form["GC"],

        "away_gc":
            away_form["GC"],

        "home_dg":
            home_form["DG"],

        "away_dg":
            away_form["DG"],

        "home_wins":
            home_form["Victorias"],

        "away_wins":
            away_form["Victorias"],

        "home_losses":
            home_form["Derrotas"],

        "away_losses":
            away_form["Derrotas"],

        "home_matches":
            home_form["Partidos"],

        "away_matches":
            away_form["Partidos"],

        "ppp_diff":
            home_form["PPP"]
            - away_form["PPP"],

        "gf_diff":
            home_form["GF"]
            - away_form["GF"],

        "gc_diff":
            home_form["GC"]
            - away_form["GC"],

        "dg_diff":
            home_form["DG"]
            - away_form["DG"],

    }


# ============================================================
# DATASET DE ENTRENAMIENTO
# ============================================================

def build_training_dataset(
    home_history,
    away_history,
    window=5
):

    if (
        home_history.empty
        or away_history.empty
    ):
        return pd.DataFrame()

    # Unimos históricos de ambos equipos
    all_history = pd.concat(
        [
            home_history,
            away_history
        ],
        ignore_index=True
    )

    all_history = all_history.drop_duplicates(
        subset=["ID"]
    )

    all_history = all_history.dropna(
        subset=[
            "Fecha",
            "GolesLocal",
            "GolesVisitante"
        ]
    )

    all_history = all_history.sort_values(
        "Fecha"
    )

    rows = []

    # Construimos observaciones históricas
    # utilizando solamente información anterior
    # a cada partido.

    for _, match in all_history.iterrows():

        home_name = match["Local"]
        away_name = match["Visitante"]

        if not home_name or not away_name:
            continue

        # Para evitar leakage:
        # solo usamos partidos anteriores
        home_previous = all_history[
            all_history["Fecha"]
            < match["Fecha"]
        ].copy()

        away_previous = home_previous.copy()

        home_form = calculate_team_form(
            home_previous,
            home_name,
            n=window,
            before_date=match["Fecha"]
        )

        away_form = calculate_team_form(
            away_previous,
            away_name,
            n=window,
            before_date=match["Fecha"]
        )

        if (
            home_form is None
            or away_form is None
        ):
            continue

        # Necesitamos una muestra mínima
        if (
            home_form["Partidos"] < window
            or away_form["Partidos"] < window
        ):
            continue

        target = get_match_target(
            match["GolesLocal"],
            match["GolesVisitante"]
        )

        if target is None:
            continue

        rows.append({

            "home_ppp":
                home_form["PPP"],

            "away_ppp":
                away_form["PPP"],

            "home_gf":
                home_form["GF"],

            "away_gf":
                away_form["GF"],

            "home_gc":
                home_form["GC"],

            "away_gc":
                away_form["GC"],

            "home_dg":
                home_form["DG"],

            "away_dg":
                away_form["DG"],

            "home_wins":
                home_form["Victorias"],

            "away_wins":
                away_form["Victorias"],

            "home_losses":
                home_form["Derrotas"],

            "away_losses":
                away_form["Derrotas"],

            "ppp_diff":
                home_form["PPP"]
                - away_form["PPP"],

            "gf_diff":
                home_form["GF"]
                - away_form["GF"],

            "gc_diff":
                home_form["GC"]
                - away_form["GC"],

            "dg_diff":
                home_form["DG"]
                - away_form["DG"],

            "target":
                target,

            "fecha":
                match["Fecha"],

        })

    return pd.DataFrame(rows)


# ============================================================
# ENTRENAMIENTO DEL MODELO
# ============================================================

FEATURE_COLUMNS = [

    "home_ppp",
    "away_ppp",

    "home_gf",
    "away_gf",

    "home_gc",
    "away_gc",

    "home_dg",
    "away_dg",

    "home_wins",
    "away_wins",

    "home_losses",
    "away_losses",

    "ppp_diff",
    "gf_diff",
    "gc_diff",
    "dg_diff",

]


def train_model(training_df):

    if training_df.empty:
        return None, None

    if len(training_df) < MIN_TRAINING_ROWS:
        return None, "INSUFFICIENT_DATA"

    X = training_df[
        FEATURE_COLUMNS
    ]

    y = training_df["target"]

    # Debemos tener al menos dos resultados
    # diferentes para entrenar clasificación.
    if y.nunique() < 2:
        return None, "ONE_CLASS_ONLY"

    # Modelo reproducible y sencillo.
    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    # División temporal:
    # primeras observaciones -> entrenamiento
    # últimas observaciones -> validación

    training_df = training_df.sort_values(
        "fecha"
    )

    split_index = int(
        len(training_df) * 0.80
    )

    if split_index < 10:
        return None, "SMALL_TRAINING_SET"

    train = training_df.iloc[
        :split_index
    ]

    validation = training_df.iloc[
        split_index:
    ]

    if (
        train["target"].nunique() < 2
        or validation.empty
    ):
        return None, "INVALID_VALIDATION"

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train["target"]

    X_valid = validation[
        FEATURE_COLUMNS
    ]

    y_valid = validation["target"]

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_valid
    )

    accuracy = accuracy_score(
        y_valid,
        predictions
    )

    validation_info = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "accuracy": accuracy,
        "classes": list(
            model.named_steps[
                "classifier"
            ].classes_
        )
    }

    return model, validation_info


# ============================================================
# PROBABILIDAD DEL PARTIDO ACTUAL
# ============================================================

def predict_match(
    model,
    features
):

    if model is None:
        return None

    X = pd.DataFrame(
        [features]
    )

    probabilities = model.predict_proba(
        X[FEATURE_COLUMNS]
    )[0]

    classes = model.named_steps[
        "classifier"
    ].classes_

    result = {}

    for cls, prob in zip(
        classes,
        probabilities
    ):
        result[cls] = float(prob)

    # Aseguramos las tres categorías
    result.setdefault("H", 0.0)
    result.setdefault("D", 0.0)
    result.setdefault("A", 0.0)

    return result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Centro de Mando"
    )

    st.caption(
        "Sports Data Hub — FASE 9"
    )

    st.divider()

    st.markdown(
        "### 📅 Fecha"
    )

    selected_date = st.date_input(
        "Día de análisis",
        value=date.today()
    )

    st.divider()

    st.markdown(
        "### 🏟️ Deporte"
    )

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

    st.markdown(
        "### 🧠 Histórico"
    )

    history_window = st.slider(
        "Partidos recientes",
        min_value=3,
        max_value=10,
        value=5
    )

    st.caption(
        "Las variables pre-partido utilizan "
        "únicamente información anterior al evento."
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
    'Histórico · Modelo Predictivo · Motor Quant'
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
        '🟡 MODELO EN VALIDACIÓN'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# MENSAJES
# ============================================================

if event_status.startswith(
    "CONNECTION_ERROR"
):

    st.error(
        "No fue posible conectar "
        "con TheSportsDB."
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
            "🔍 Analizar evento",
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
        df_events["ID"]
        == selected_event_id
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
        # COMPATIBILIDAD CON MODELO 1X2
        # ====================================================

        model_compatible = (
            bool(selected["IDLocal"])
            and bool(selected["IDVisitante"])
            and bool(selected["Local"])
            and bool(selected["Visitante"])
        )

        if not model_compatible:

            st.warning(
                "⚠️ Este evento no tiene estructura "
                "Local vs Visitante compatible con "
                "el modelo 1X2. Se mostrará como "
                "evento deportivo, pero no se fabricará "
                "una probabilidad predictiva."
            )

            st.info(
                "Esto es esperado en eventos como "
                "atletismo individual, donde no existen "
                "dos equipos enfrentados."
            )

        else:

            # =================================================
            # DATOS DEL EVENTO
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📋 Datos del evento'
                '</div>',
                unsafe_allow_html=True
            )

            a1, a2, a3, a4 = st.columns(4)

            a1.metric(
                "Deporte",
                safe_value(
                    selected["Deporte"]
                )
            )

            a2.metric(
                "Competición",
                safe_value(
                    selected["Liga"]
                )
            )

            a3.metric(
                "Fecha",
                format_date_display(
                    selected["Fecha"]
                )
            )

            a4.metric(
                "Hora",
                safe_value(
                    selected["Hora"],
                    "--"
                )
            )

            # =================================================
            # HISTÓRICO
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📚 Histórico de los equipos'
                '</div>',
                unsafe_allow_html=True
            )

            home_history_raw, home_status = (
                get_team_history(
                    selected["IDLocal"]
                )
            )

            away_history_raw, away_status = (
                get_team_history(
                    selected["IDVisitante"]
                )
            )

            home_history = (
                history_to_dataframe(
                    home_history_raw
                )
            )

            away_history = (
                history_to_dataframe(
                    away_history_raw
                )
            )

            h1, h2 = st.columns(2)

            with h1:

                if home_status == "OK":

                    st.success(
                        f"Local: {len(home_history)} "
                        f"registros históricos"
                    )

                else:

                    st.warning(
                        f"Local: {home_status}"
                    )

            with h2:

                if away_status == "OK":

                    st.success(
                        f"Visitante: {len(away_history)} "
                        f"registros históricos"
                    )

                else:

                    st.warning(
                        f"Visitante: {away_status}"
                    )

            # =================================================
            # FORMA ACTUAL
            # =================================================

            home_form = calculate_team_form(
                home_history,
                selected["Local"],
                history_window,
                before_date=selected["Fecha"]
            )

            away_form = calculate_team_form(
                away_history,
                selected["Visitante"],
                history_window,
                before_date=selected["Fecha"]
            )

            if (
                home_form is not None
                and away_form is not None
            ):

                st.markdown(
                    '<div class="section-title">'
                    '📊 Forma reciente'
                    '</div>',
                    unsafe_allow_html=True
                )

                form_df = pd.DataFrame([

                    {
                        "Variable":
                            "Partidos analizados",

                        selected["Local"]:
                            home_form["Partidos"],

                        selected["Visitante"]:
                            away_form["Partidos"],
                    },

                    {
                        "Variable":
                            "Puntos por partido",

                        selected["Local"]:
                            round(
                                home_form["PPP"],
                                3
                            ),

                        selected["Visitante"]:
                            round(
                                away_form["PPP"],
                                3
                            ),
                    },

                    {
                        "Variable":
                            "Goles por partido",

                        selected["Local"]:
                            round(
                                home_form["GF"],
                                3
                            ),

                        selected["Visitante"]:
                            round(
                                away_form["GF"],
                                3
                            ),
                    },

                    {
                        "Variable":
                            "Goles recibidos/partido",

                        selected["Local"]:
                            round(
                                home_form["GC"],
                                3
                            ),

                        selected["Visitante"]:
                            round(
                                away_form["GC"],
                                3
                            ),
                    },

                    {
                        "Variable":
                            "Victorias",

                        selected["Local"]:
                            home_form["Victorias"],

                        selected["Visitante"]:
                            away_form["Victorias"],
                    },

                    {
                        "Variable":
                            "Empates",

                        selected["Local"]:
                            home_form["Empates"],

                        selected["Visitante"]:
                            away_form["Empates"],
                    },

                    {
                        "Variable":
                            "Derrotas",

                        selected["Local"]:
                            home_form["Derrotas"],

                        selected["Visitante"]:
                            away_form["Derrotas"],
                    },

                    {
                        "Variable":
                            "Diferencia de goles",

                        selected["Local"]:
                            round(
                                home_form["DG"],
                                3
                            ),

                        selected["Visitante"]:
                            round(
                                away_form["DG"],
                                3
                            ),
                    },

                ])

                st.dataframe(
                    form_df,
                    use_container_width=True,
                    hide_index=True
                )

            # =================================================
            # DATASET DE ENTRENAMIENTO
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '🧪 Dataset histórico de entrenamiento'
                '</div>',
                unsafe_allow_html=True
            )

            training_df = build_training_dataset(
                home_history,
                away_history,
                history_window
            )

            if training_df.empty:

                st.warning(
                    "No se pudo construir un dataset "
                    "histórico suficiente para entrenar."
                )

                model = None
                model_status = (
                    "INSUFFICIENT_DATA"
                )

            else:

                st.success(
                    f"Dataset construido: "
                    f"{len(training_df)} "
                    f"observaciones históricas."
                )

                target_counts = (
                    training_df["target"]
                    .value_counts()
                    .to_dict()
                )

                d1, d2, d3 = st.columns(3)

                d1.metric(
                    "Local",
                    target_counts.get(
                        "H",
                        0
                    )
                )

                d2.metric(
                    "Empate",
                    target_counts.get(
                        "D",
                        0
                    )
                )

                d3.metric(
                    "Visitante",
                    target_counts.get(
                        "A",
                        0
                    )
                )

                with st.expander(
                    "🔎 Ver dataset histórico"
                ):

                    st.dataframe(
                        training_df,
                        use_container_width=True,
                        hide_index=True
                    )

                # =============================================
                # ENTRENAR
                # =============================================

                model, model_status = (
                    train_model(
                        training_df
                    )
                )

            # =================================================
            # MODELO
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '🧠 Modelo predictivo'
                '</div>',
                unsafe_allow_html=True
            )

            if model is None:

                if model_status == (
                    "INSUFFICIENT_DATA"
                ):

                    st.warning(
                        "🟡 Modelo bloqueado: "
                        "no hay suficientes observaciones "
                        "históricas válidas."
                    )

                elif model_status == (
                    "ONE_CLASS_ONLY"
                ):

                    st.warning(
                        "🟡 Modelo bloqueado: "
                        "el histórico contiene un único "
                        "tipo de resultado."
                    )

                else:

                    st.warning(
                        f"🟡 Modelo no entrenado: "
                        f"{model_status}"
                    )

            else:

                st.success(
                    "🟢 Modelo entrenado y validado "
                    "con división temporal."
                )

                v1, v2, v3 = st.columns(3)

                v1.metric(
                    "Entrenamiento",
                    model_status[
                        "train_rows"
                    ]
                )

                v2.metric(
                    "Validación",
                    model_status[
                        "validation_rows"
                    ]
                )

                v3.metric(
                    "Accuracy",
                    f"{model_status['accuracy']:.1%}"
                )

                # =============================================
                # PREDICCIÓN ACTUAL
                # =============================================

                current_features = (
                    build_pre_match_features(
                        selected,
                        home_history,
                        away_history,
                        history_window
                    )
                )

                if current_features:

                    probabilities = predict_match(
                        model,
                        current_features
                    )

                    if probabilities:

                        st.markdown(
                            '<div class="section-title">'
                            '🎯 Probabilidad propia del modelo'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        p1, p2, p3 = st.columns(3)

                        p1.metric(
                            "🏠 Local",
                            f"{probabilities['H']:.1%}"
                        )

                        p2.metric(
                            "🤝 Empate",
                            f"{probabilities['D']:.1%}"
                        )

                        p3.metric(
                            "✈️ Visitante",
                            f"{probabilities['A']:.1%}"
                        )

                        best = max(
                            probabilities,
                            key=probabilities.get
                        )

                        labels = {
                            "H": "Local",
                            "D": "Empate",
                            "A": "Visitante"
                        }

                        st.info(
                            f"Resultado con mayor "
                            f"probabilidad estimada: "
                            f"**{labels[best]}** "
                            f"({probabilities[best]:.1%})"
                        )

                        st.caption(
                            "Esta es una probabilidad "
                            "producida por el modelo; "
                            "no es una cuota ni una "
                            "probabilidad implícita."
                        )

                    else:

                        st.warning(
                            "No fue posible generar "
                            "la predicción."
                        )

                else:

                    st.warning(
                        "No fue posible construir "
                        "las variables pre-partido."
                    )

            # =================================================
            # MERCADO
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '💰 Mercado'
                '</div>',
                unsafe_allow_html=True
            )

            st.info(
                "🟡 Cuotas no conectadas. "
                "El sistema no inventará cuotas "
                "ni probabilidades implícitas."
            )

            # =================================================
            # VALUE
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📈 Value Betting'
                '</div>',
                unsafe_allow_html=True
            )

            v1, v2, v3 = st.columns(3)

            v1.metric(
                "Edge",
                "BLOQUEADO"
            )

            v2.metric(
                "EV",
                "BLOQUEADO"
            )

            v3.metric(
                "Value Score",
                "BLOQUEADO"
            )

            st.caption(
                "Edge, EV y Value Score solo se "
                "activarán cuando exista una cuota "
                "real y trazable para comparar "
                "contra la probabilidad propia."
            )


# ============================================================
# RANKING
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🏆 Ranking de oportunidades'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.write(
    "🟡 Ranking pendiente de mercado."
)

st.write(
    "El ranking no asignará puntuaciones "
    "artificiales."
)

st.write(
    "Para activarlo se necesitará:"
)

st.write(
    "1. Probabilidad propia del modelo."
)

st.write(
    "2. Cuota real."
)

st.write(
    "3. Probabilidad implícita."
)

st.write(
    "4. Edge."
)

st.write(
    "5. EV."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
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

    🟢 Variables pre-partido calculadas únicamente
    con información anterior al evento.

    <br><br>

    🟢 Dataset histórico construido con resultados reales.

    <br><br>

    🟢 Modelo entrenado únicamente cuando existe
    suficiente información.

    <br><br>

    🔒 No se inventan cuotas.

    <br>

    🔒 No se inventan probabilidades.

    <br>

    🔒 No se fabrica Value Score.

    <br>

    🔒 No se utiliza una estimación manual como
    probabilidad de modelo.

    <br>

    🔒 No se utiliza información futura para
    construir variables históricas.

    <br><br>

    El sistema bloquea automáticamente cualquier
    cálculo predictivo cuando los datos no son
    suficientes o no son compatibles.
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

    Partidos anteriores → resultados

    🟢 Activa

    ---

    ### CAPA 3 — Feature Engineering

    Histórico anterior al partido
    ↓
    Forma reciente
    ↓
    Goles
    ↓
    Puntos
    ↓
    Victorias
    ↓
    Derrotas
    ↓
    Diferencias

    🟢 Activa

    ---

    ### CAPA 4 — Dataset de entrenamiento

    Variables pre-partido
    +
    Resultado real

    🟢 Activa

    ---

    ### CAPA 5 — Modelo predictivo

    Dataset histórico
    ↓
    Regresión logística
    ↓
    Validación temporal
    ↓
    Probabilidades

    🟢 Activa / Validada

    ---

    ### CAPA 6 — Mercado

    Cuotas reales
    ↓
    Probabilidad implícita

    🟡 Pendiente

    ---

    ### CAPA 7 — Motor Quant

    Probabilidad propia
    +
    Probabilidad implícita
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

    Datos
    +
    Histórico
    +
    Modelo
    +
    Mercado
    +
    Motor Quant

    🟡 En construcción
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE 9"
)

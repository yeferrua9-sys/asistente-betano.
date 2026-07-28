import streamlit as st
import pandas as pd
import requests
import re
import unicodedata
from datetime import datetime, date, timedelta

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


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
ODDS_API_BASE_URL = "https://api.odds-api.io/v3"

MIN_TRAINING_ROWS = 15


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILIDADES GENERALES
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


def normalize_text(value):

    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKD",
        value
    ).encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    value = value.lower()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def team_tokens(value):

    normalized = normalize_text(value)

    return set(
        token
        for token in normalized.split()
        if len(token) >= 3
    )


def team_similarity(a, b):

    na = normalize_text(a)
    nb = normalize_text(b)

    if not na or not nb:
        return 0.0

    if na == nb:
        return 1.0

    if na in nb or nb in na:
        return 0.90

    ta = team_tokens(a)
    tb = team_tokens(b)

    if not ta or not tb:
        return 0.0

    intersection = len(ta & tb)
    union = len(ta | tb)

    return intersection / union if union else 0.0


def pair_similarity(
    home_a,
    away_a,
    home_b,
    away_b
):

    direct = (
        team_similarity(home_a, home_b)
        + team_similarity(away_a, away_b)
    ) / 2

    reverse = (
        team_similarity(home_a, away_b)
        + team_similarity(away_a, home_b)
    ) / 2

    return max(direct, reverse), direct >= reverse


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


def format_time(event):

    timestamp = event.get(
        "strTimestamp"
    )

    if timestamp:

        try:

            dt = datetime.fromisoformat(
                timestamp.replace(
                    "Z",
                    "+00:00"
                )
            )

            return dt.strftime(
                "%H:%M"
            )

        except Exception:
            pass

    raw = event.get(
        "strTime"
    )

    if raw:
        return raw[:5]

    return "--"


def normalize_score(value):

    try:
        return int(value)

    except (
        TypeError,
        ValueError
    ):
        return None


def parse_odds_value(value):

    try:

        value = float(value)

        if value <= 1:
            return None

        return value

    except (
        TypeError,
        ValueError
    ):

        return None


# ============================================================
# THE SPORTSDb — EVENTOS DEL DÍA
# ============================================================

@st.cache_data(ttl=300)
def get_events_day(
    selected_date,
    sport_filter
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

        response.raise_for_status()

        data = response.json()

        events = data.get(
            "events"
        ) or []

        if not events:
            return [], "NO_EVENTS"

        return events, "OK"

    except requests.RequestException as error:

        return [], (
            f"CONNECTION_ERROR: {error}"
        )

    except ValueError:

        return [], "JSON_ERROR"


# ============================================================
# EVENTOS → DATAFRAME
# ============================================================

def events_to_dataframe(events):

    rows = []

    for event in events:

        home = event.get(
            "strHomeTeam"
        )

        away = event.get(
            "strAwayTeam"
        )

        if home and away:

            matchup = (
                f"{home} vs {away}"
            )

        else:

            matchup = event.get(
                "strEvent",
                "Evento deportivo"
            )

        rows.append({

            "ID":
                event.get(
                    "idEvent"
                ),

            "Deporte":
                event.get(
                    "strSport",
                    ""
                ),

            "Liga":
                event.get(
                    "strLeague",
                    ""
                ),

            "IDLiga":
                event.get(
                    "idLeague"
                ),

            "Evento":
                matchup,

            "Fecha":
                event.get(
                    "dateEvent",
                    ""
                ),

            "Hora":
                format_time(event),

            "Local":
                home or "",

            "Visitante":
                away or "",

            "IDLocal":
                event.get(
                    "idHomeTeam"
                ),

            "IDVisitante":
                event.get(
                    "idAwayTeam"
                ),

            "Estadio":
                event.get(
                    "strVenue",
                    ""
                ),

            "Ciudad":
                event.get(
                    "strCity",
                    ""
                ),

            "País":
                event.get(
                    "strCountry",
                    ""
                ),

            "Temporada":
                event.get(
                    "strSeason",
                    ""
                ),

            "Ronda":
                event.get(
                    "intRound",
                    ""
                ),

            "ResultadoLocal":
                normalize_score(
                    event.get(
                        "intHomeScore"
                    )
                ),

            "ResultadoVisitante":
                normalize_score(
                    event.get(
                        "intAwayScore"
                    )
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

    url = (
        f"{TSDB_BASE_URL}/eventslast.php"
    )

    params = {
        "id": team_id
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        events = data.get(
            "results"
        ) or []

        if not events:
            return [], "NO_HISTORY"

        return events, "OK"

    except requests.RequestException as error:

        return [], (
            f"CONNECTION_ERROR: {error}"
        )

    except ValueError:

        return [], "JSON_ERROR"


# ============================================================
# HISTÓRICO → DATAFRAME
# ============================================================

def history_to_dataframe(events):

    rows = []

    for event in events:

        home = event.get(
            "strHomeTeam"
        )

        away = event.get(
            "strAwayTeam"
        )

        home_score = normalize_score(
            event.get(
                "intHomeScore"
            )
        )

        away_score = normalize_score(
            event.get(
                "intAwayScore"
            )
        )

        rows.append({

            "ID":
                event.get(
                    "idEvent"
                ),

            "Fecha":
                event.get(
                    "dateEvent",
                    ""
                ),

            "Liga":
                event.get(
                    "strLeague",
                    ""
                ),

            "Temporada":
                event.get(
                    "strSeason",
                    ""
                ),

            "Local":
                home or "",

            "Visitante":
                away or "",

            "GolesLocal":
                home_score,

            "GolesVisitante":
                away_score,

        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["Fecha"] = pd.to_datetime(
        df["Fecha"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Fecha"
        ]
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
            history_df["Local"]
            == team_name
        )
        |
        (
            history_df["Visitante"]
            == team_name
        )
    ].copy()

    if before_date is not None:

        before_date = pd.to_datetime(
            before_date
        )

        relevant = relevant[
            relevant["Fecha"]
            < before_date
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

    for _, row in relevant.iterrows():

        is_home = (
            row["Local"]
            == team_name
        )

        if is_home:

            gf = row["GolesLocal"]
            ga = row["GolesVisitante"]

        else:

            gf = row["GolesVisitante"]
            ga = row["GolesLocal"]

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

        "Partidos":
            matches,

        "Victorias":
            wins,

        "Empates":
            draws,

        "Derrotas":
            losses,

        "Puntos":
            points,

        "PPP":
            points / matches,

        "GF":
            goals_for / matches,

        "GC":
            goals_against / matches,

        "DG":
            (
                goals_for
                - goals_against
            ) / matches,

    }


# ============================================================
# RESULTADO
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
# FEATURES
# ============================================================

def forms_to_features(
    home_form,
    away_form
):

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
            (
                home_form["PPP"]
                - away_form["PPP"]
            ),

        "gf_diff":
            (
                home_form["GF"]
                - away_form["GF"]
            ),

        "gc_diff":
            (
                home_form["GC"]
                - away_form["GC"]
            ),

        "dg_diff":
            (
                home_form["DG"]
                - away_form["DG"]
            ),

    }


def build_pre_match_features(
    match,
    home_history,
    away_history,
    window=5
):

    home_name = match["Local"]
    away_name = match["Visitante"]

    match_date = match["Fecha"]

    all_history = pd.concat(
        [
            home_history,
            away_history
        ],
        ignore_index=True
    ).drop_duplicates(
        subset=["ID"]
    )

    home_form = calculate_team_form(
        all_history,
        home_name,
        n=window,
        before_date=match_date
    )

    away_form = calculate_team_form(
        all_history,
        away_name,
        n=window,
        before_date=match_date
    )

    if (
        home_form is None
        or away_form is None
    ):
        return None

    if (
        home_form["Partidos"]
        < window
        or
        away_form["Partidos"]
        < window
    ):
        return None

    return forms_to_features(
        home_form,
        away_form
    )


# ============================================================
# DATASET
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
    ).reset_index(
        drop=True
    )

    rows = []

    for index, match in all_history.iterrows():

        home_name = match["Local"]
        away_name = match["Visitante"]

        if not home_name or not away_name:
            continue

        previous_matches = all_history.iloc[
            :index
        ].copy()

        if previous_matches.empty:
            continue

        home_form = calculate_team_form(
            previous_matches,
            home_name,
            n=window
        )

        away_form = calculate_team_form(
            previous_matches,
            away_name,
            n=window
        )

        if (
            home_form is None
            or away_form is None
        ):
            continue

        if (
            home_form["Partidos"]
            < window
            or
            away_form["Partidos"]
            < window
        ):
            continue

        target = get_match_target(
            match["GolesLocal"],
            match["GolesVisitante"]
        )

        if target is None:
            continue

        features = forms_to_features(
            home_form,
            away_form
        )

        features["target"] = target
        features["fecha"] = match["Fecha"]

        rows.append(features)

    return pd.DataFrame(rows)


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
    "home_matches",
    "away_matches",
    "ppp_diff",
    "gf_diff",
    "gc_diff",
    "dg_diff",

]


# ============================================================
# MODELO
# ============================================================

def train_model(training_df):

    if training_df.empty:
        return None, "NO_DATA"

    if len(training_df) < MIN_TRAINING_ROWS:
        return None, "INSUFFICIENT_DATA"

    training_df = training_df.sort_values(
        "fecha"
    ).reset_index(
        drop=True
    )

    if training_df["target"].nunique() < 2:
        return None, "ONE_CLASS_ONLY"

    split_index = int(
        len(training_df) * 0.80
    )

    if split_index < 10:
        return None, "SMALL_TRAINING_SET"

    train = training_df.iloc[
        :split_index
    ].copy()

    validation = training_df.iloc[
        split_index:
    ].copy()

    if validation.empty:
        return None, "INVALID_VALIDATION"

    if train["target"].nunique() < 2:
        return None, "INVALID_TRAIN_CLASSES"

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train["target"]

    X_valid = validation[
        FEATURE_COLUMNS
    ]

    y_valid = validation["target"]

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

        "train_rows":
            len(train),

        "validation_rows":
            len(validation),

        "accuracy":
            accuracy,

        "classes":
            list(
                model.named_steps[
                    "classifier"
                ].classes_
            ),

    }

    return model, validation_info


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

    result.setdefault(
        "H",
        0.0
    )

    result.setdefault(
        "D",
        0.0
    )

    result.setdefault(
        "A",
        0.0
    )

    return result


# ============================================================
# ODDS-API.IO
# ============================================================

def odds_api_request(
    endpoint,
    api_key,
    params=None
):

    if not api_key:
        return None, "NO_API_KEY"

    request_params = dict(
        params or {}
    )

    request_params["apiKey"] = api_key

    try:

        response = requests.get(
            f"{ODDS_API_BASE_URL}/{endpoint}",
            params=request_params,
            timeout=20
        )

        if response.status_code == 401:
            return None, "INVALID_API_KEY"

        if response.status_code == 403:
            return None, "API_FORBIDDEN"

        if response.status_code == 429:
            return None, "RATE_LIMIT"

        response.raise_for_status()

        return response.json(), "OK"

    except requests.RequestException as error:

        return None, (
            f"CONNECTION_ERROR: {error}"
        )

    except ValueError:

        return None, "INVALID_JSON"


@st.cache_data(ttl=180)
def get_odds_events(
    api_key,
    sport="football"
):

    events, status = odds_api_request(
        "events",
        api_key,
        {
            "sport": sport,
            "status": "pending",
            "limit": 100,
        }
    )

    if status != "OK":
        return [], status

    if isinstance(events, list):
        return events, "OK"

    return [], "INVALID_EVENTS_STRUCTURE"


def odds_event_date(
    event
):

    raw = event.get(
        "date"
    )

    if not raw:
        return None

    try:

        return pd.to_datetime(
            raw,
            utc=True,
            errors="coerce"
        )

    except Exception:

        return None


def find_odds_event_for_match(
    match,
    odds_events
):

    home = match["Local"]
    away = match["Visitante"]

    target_date = pd.to_datetime(
        match["Fecha"],
        errors="coerce"
    )

    if pd.isna(target_date):
        return None, 0.0

    best_event = None
    best_score = 0.0

    for event in odds_events:

        event_home = event.get(
            "home"
        )

        event_away = event.get(
            "away"
        )

        if not event_home or not event_away:
            continue

        score, direct = pair_similarity(
            home,
            away,
            event_home,
            event_away
        )

        if score < 0.55:
            continue

        event_date = odds_event_date(
            event
        )

        if event_date is not None:

            event_day = event_date.date()

            if event_day != target_date.date():

                date_distance = abs(
                    (
                        event_day
                        - target_date.date()
                    ).days
                )

                if date_distance > 1:
                    continue

                score *= 0.90

        if score > best_score:

            best_score = score
            best_event = event

    return best_event, best_score


def extract_ml_market(
    odds_data,
    bookmakers
):

    if not isinstance(
        odds_data,
        dict
    ):
        return []

    bookmaker_data = (
        odds_data.get(
            "bookmakers"
        )
        or {}
    )

    rows = []

    for bookmaker in bookmakers:

        markets = (
            bookmaker_data.get(
                bookmaker
            )
            or []
        )

        for market in markets:

            market_name = str(
                market.get(
                    "name",
                    ""
                )
            ).upper()

            if market_name not in [
                "ML",
                "MONEYLINE",
                "MATCH RESULT"
            ]:
                continue

            odds_list = (
                market.get(
                    "odds"
                )
                or []
            )

            for odd in odds_list:

                home_odd = parse_odds_value(
                    odd.get(
                        "home"
                    )
                )

                draw_odd = parse_odds_value(
                    odd.get(
                        "draw"
                    )
                )

                away_odd = parse_odds_value(
                    odd.get(
                        "away"
                    )
                )

                if (
                    home_odd is None
                    and
                    draw_odd is None
                    and
                    away_odd is None
                ):
                    continue

                rows.append({

                    "bookmaker":
                        bookmaker,

                    "home_odds":
                        home_odd,

                    "draw_odds":
                        draw_odd,

                    "away_odds":
                        away_odd,

                    "updatedAt":
                        market.get(
                            "updatedAt"
                        ),

                })

    return rows


def get_event_odds(
    event_id,
    api_key,
    bookmakers
):

    data, status = odds_api_request(
        "odds",
        api_key,
        {
            "eventId": event_id,
            "bookmakers":
                ",".join(bookmakers),
        }
    )

    if status != "OK":
        return [], status

    rows = extract_ml_market(
        data,
        bookmakers
    )

    return rows, (
        "OK"
        if rows
        else "NO_ML_MARKET"
    )


def calculate_market_metrics(
    probability,
    odds
):

    if (
        probability is None
        or odds is None
        or odds <= 1
    ):
        return None

    implied_probability = 1 / odds

    edge = (
        probability
        - implied_probability
    )

    ev = (
        probability * odds
    ) - 1

    value_score = (
        edge * 100
    )

    return {

        "implied_probability":
            implied_probability,

        "edge":
            edge,

        "ev":
            ev,

        "value_score":
            value_score,

    }


def build_market_for_match(
    match,
    api_key,
    bookmakers
):

    if not api_key:

        return {
            "status":
                "NO_API_KEY"
        }

    odds_events, status = (
        get_odds_events(
            api_key,
            sport="football"
        )
    )

    if status != "OK":

        return {
            "status":
                status
        }

    odds_event, match_score = (
        find_odds_event_for_match(
            match,
            odds_events
        )
    )

    if odds_event is None:

        return {
            "status":
                "NO_MATCH",
            "match_score":
                match_score
        }

    event_id = odds_event.get(
        "id"
    )

    if not event_id:

        return {
            "status":
                "INVALID_EVENT_ID"
        }

    odds_rows, odds_status = (
        get_event_odds(
            event_id,
            api_key,
            bookmakers
        )
    )

    if odds_status != "OK":

        return {
            "status":
                odds_status,
            "odds_event":
                odds_event,
            "match_score":
                match_score
        }

    return {

        "status":
            "OK",

        "odds_event":
            odds_event,

        "match_score":
            match_score,

        "odds":
            odds_rows,

    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Centro de Mando"
    )

    st.caption(
        "Sports Data Hub — FASE 16"
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
        ],
        index=1
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
        "El modelo utiliza únicamente "
        "información anterior al partido."
    )

    st.divider()

    st.markdown(
        "### 💰 Mercado real"
    )

    odds_api_key = st.text_input(
        "API Key de Odds-API.io",
        type="password",
        placeholder="Pega aquí tu API Key"
    )

    bookmaker_options = [
        "Bet365",
        "Unibet"
    ]

    selected_bookmakers = st.multiselect(
        "Casas a consultar",
        bookmaker_options,
        default=bookmaker_options
    )

    st.caption(
        "Odds-API.io · flujo oficial "
        "Events → Event ID → Odds"
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
    """
    <div class="main-title">
        📊 Centro de Mando Quant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Sports Data Hub · Datos reales · Histórico ·
        Modelo Predictivo · Mercado Real · Motor Quant
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EVENTOS THE SPORTSDb
# ============================================================

events, event_status = get_events_day(
    selected_date.strftime(
        "%Y-%m-%d"
    ),
    sport_filter
)

df_events = events_to_dataframe(
    events
)


# ============================================================
# MERCADO — PRECALCULAR COINCIDENCIAS
# ============================================================

market_results = {}

if (
    odds_api_key
    and sport_filter == "Soccer"
    and selected_bookmakers
    and not df_events.empty
):

    for _, row in df_events.iterrows():

        event_id = row["ID"]

        market_results[event_id] = (
            build_market_for_match(
                row,
                odds_api_key,
                selected_bookmakers
            )
        )


# ============================================================
# ESTADOS
# ============================================================

market_connected = bool(
    odds_api_key
    and selected_bookmakers
)

market_error = False

for result in market_results.values():

    if result.get("status") in [
        "INVALID_API_KEY",
        "API_FORBIDDEN",
        "RATE_LIMIT",
        "CONNECTION_ERROR",
        "INVALID_JSON",
        "INVALID_EVENTS_STRUCTURE",
    ]:

        market_error = True
        break


status1, status2, status3 = st.columns(3)

with status1:

    st.markdown(
        """
        <div class="status status-green">
            🟢 DATOS DEPORTIVOS
        </div>
        """,
        unsafe_allow_html=True
    )

with status2:

    st.markdown(
        """
        <div class="status status-green">
            🟢 HISTÓRICO
        </div>
        """,
        unsafe_allow_html=True
    )

with status3:

    if market_error:

        st.markdown(
            """
            <div class="status status-red">
                🔴 ERROR DE MERCADO
            </div>
            """,
            unsafe_allow_html=True
        )

    elif market_connected:

        st.markdown(
            """
            <div class="status status-green">
                🟢 MERCADO REAL CONECTADO
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status status-yellow">
                🟡 MERCADO SIN API KEY
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MENSAJES
# ============================================================

if event_status.startswith(
    "CONNECTION_ERROR"
):

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

if market_error:

    st.error(
        "Odds-API.io respondió con un error "
        "al consultar eventos o cuotas. "
        "Revisa la API Key y el estado de la API."
    )


# ============================================================
# KPIs
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📈 Resumen
    </div>
    """,
    unsafe_allow_html=True
)

events_with_market = 0

for result in market_results.values():

    if result.get(
        "status"
    ) == "OK":

        events_with_market += 1


c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Eventos",
    len(df_events)
)

c2.metric(
    "Deportes",
    df_events["Deporte"].nunique()
    if not df_events.empty
    else 0
)

c3.metric(
    "Ligas",
    df_events["Liga"].nunique()
    if not df_events.empty
    else 0
)

c4.metric(
    "Eventos con mercado",
    events_with_market
)


# ============================================================
# FILTROS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🔎 Filtrar partidos
    </div>
    """,
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
            ["Todas"]
            + league_options
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
            ["Todos"]
            + country_options
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
    """
    <div class="section-title">
        🏟️ Partidos y eventos
    </div>
    """,
    unsafe_allow_html=True
)

if filtered.empty:

    st.info(
        "No hay eventos deportivos para mostrar."
    )

else:

    for _, row in filtered.iterrows():

        event_id = row["ID"]

        st.markdown(
            '<div class="match-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="gray">
                {safe_value(
                    row["Liga"],
                    "Competición"
                )}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="team">
                {safe_value(
                    row["Evento"],
                    "Evento deportivo"
                )}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="gray">
                📅 {format_date_display(
                    row["Fecha"]
                )}
                &nbsp;&nbsp;
                ⏰ {safe_value(
                    row["Hora"],
                    "--"
                )}
            </div>
            """,
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
            f"""
            <div class="gray">
                📍 {location or
                "Ubicación no disponible"}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # ESTADO DE MERCADO
        # ----------------------------------------------------

        result = market_results.get(
            event_id
        )

        if result:

            if result.get(
                "status"
            ) == "OK":

                odds_event = result[
                    "odds_event"
                ]

                best_odds = result[
                    "odds"
                ]

                st.success(
                    "🟢 CUOTA COINCIDENTE"
                )

                st.caption(
                    f"Odds-API.io: "
                    f"{odds_event.get('home')} "
                    f"vs "
                    f"{odds_event.get('away')}"
                )

                if best_odds:

                    market_preview = []

                    for odd in best_odds:

                        market_preview.append({
                            "Casa":
                                odd["bookmaker"],

                            "Local":
                                odd["home_odds"],

                            "Empate":
                                odd["draw_odds"],

                            "Visitante":
                                odd["away_odds"],
                        })

                    st.dataframe(
                        pd.DataFrame(
                            market_preview
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

            elif result.get(
                "status"
            ) == "NO_API_KEY":

                st.warning(
                    "🟡 SIN API KEY"
                )

            elif result.get(
                "status"
            ) == "NO_MATCH":

                st.warning(
                    "🟡 SIN EVENTO COINCIDENTE"
                )

            elif result.get(
                "status"
            ) == "NO_ML_MARKET":

                st.warning(
                    "🟡 SIN MERCADO 1X2"
                )

            else:

                st.warning(
                    "🟡 MERCADO NO DISPONIBLE"
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

selected_event_id = (
    st.session_state.get(
        "selected_event_id"
    )
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
            """
            <div class="section-title">
                🎯 Análisis del evento
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="team">
                    {safe_value(
                        selected["Evento"]
                    )}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        model_compatible = (
            bool(selected["IDLocal"])
            and bool(selected["IDVisitante"])
            and bool(selected["Local"])
            and bool(selected["Visitante"])
        )

        if not model_compatible:

            st.warning(
                "Este evento no tiene estructura "
                "Local vs Visitante compatible "
                "con el modelo 1X2."
            )

        else:

            # =================================================
            # DATOS
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    📋 Datos del evento
                </div>
                """,
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
                """
                <div class="section-title">
                    📚 Histórico de los equipos
                </div>
                """,
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

            home_history = history_to_dataframe(
                home_history_raw
            )

            away_history = history_to_dataframe(
                away_history_raw
            )

            h1, h2 = st.columns(2)

            with h1:

                if home_status == "OK":

                    st.success(
                        f"Local: "
                        f"{len(home_history)} "
                        f"registros"
                    )

                else:

                    st.warning(
                        f"Local: {home_status}"
                    )

            with h2:

                if away_status == "OK":

                    st.success(
                        f"Visitante: "
                        f"{len(away_history)} "
                        f"registros"
                    )

                else:

                    st.warning(
                        f"Visitante: {away_status}"
                    )

            # =================================================
            # FORMA
            # =================================================

            combined_history = pd.concat(
                [
                    home_history,
                    away_history
                ],
                ignore_index=True
            ).drop_duplicates(
                subset=["ID"]
            )

            home_form = calculate_team_form(
                combined_history,
                selected["Local"],
                history_window,
                before_date=selected["Fecha"]
            )

            away_form = calculate_team_form(
                combined_history,
                selected["Visitante"],
                history_window,
                before_date=selected["Fecha"]
            )

            if (
                home_form is not None
                and away_form is not None
            ):

                st.markdown(
                    """
                    <div class="section-title">
                        📊 Forma reciente
                    </div>
                    """,
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
            # DATASET
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    🧪 Dataset histórico
                </div>
                """,
                unsafe_allow_html=True
            )

            training_df = (
                build_training_dataset(
                    home_history,
                    away_history,
                    history_window
                )
            )

            if training_df.empty:

                st.warning(
                    "No existe suficiente información "
                    "histórica para construir el dataset."
                )

                model = None
                model_status = (
                    "INSUFFICIENT_DATA"
                )

            else:

                st.success(
                    f"Dataset construido: "
                    f"{len(training_df)} "
                    f"observaciones."
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

                model, model_status = (
                    train_model(
                        training_df
                    )
                )

            # =================================================
            # MODELO
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    🧠 Modelo predictivo
                </div>
                """,
                unsafe_allow_html=True
            )

            probabilities = None

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

                elif model_status == (
                    "INVALID_TRAIN_CLASSES"
                ):

                    st.warning(
                        "🟡 Modelo bloqueado: "
                        "la muestra de entrenamiento "
                        "no contiene suficientes clases."
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

                current_features = (
                    build_pre_match_features(
                        selected,
                        home_history,
                        away_history,
                        history_window
                    )
                )

                if current_features:

                    probabilities = (
                        predict_match(
                            model,
                            current_features
                        )
                    )

                    if probabilities:

                        st.markdown(
                            """
                            <div class="section-title">
                                🎯 Probabilidad propia
                            </div>
                            """,
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
                            "H":
                                "Local",

                            "D":
                                "Empate",

                            "A":
                                "Visitante"
                        }

                        st.info(
                            f"Resultado con mayor "
                            f"probabilidad estimada: "
                            f"**{labels[best]}** "
                            f"({probabilities[best]:.1%})"
                        )

            # =================================================
            # MERCADO REAL
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    💰 Mercado real
                </div>
                """,
                unsafe_allow_html=True
            )

            market = market_results.get(
                selected_event_id
            )

            if not market:

                st.warning(
                    "🟡 Mercado no consultado."
                )

            elif market.get(
                "status"
            ) != "OK":

                status = market.get(
                    "status"
                )

                messages = {

                    "NO_API_KEY":
                        "Introduce la API Key "
                        "de Odds-API.io.",

                    "NO_MATCH":
                        "No se encontró un evento "
                        "coincidente en Odds-API.io.",

                    "NO_ML_MARKET":
                        "El evento existe en Odds-API.io "
                        "pero no tiene mercado ML/1X2.",

                    "INVALID_API_KEY":
                        "La API Key de Odds-API.io "
                        "no es válida.",

                    "RATE_LIMIT":
                        "Se alcanzó el límite de "
                        "solicitudes de Odds-API.io.",

                }

                st.warning(
                    f"🟡 {messages.get(
                        status,
                        'Mercado no disponible.'
                    )}"
                )

            else:

                odds_event = market[
                    "odds_event"
                ]

                st.success(
                    "🟢 Evento y mercado "
                    "coincidentes."
                )

                st.caption(
                    f"Evento Odds-API.io: "
                    f"{odds_event.get('home')} "
                    f"vs "
                    f"{odds_event.get('away')}"
                )

                odds_rows = market[
                    "odds"
                ]

                market_table = []

                for odd in odds_rows:

                    market_table.append({

                        "Casa":
                            odd["bookmaker"],

                        "Local":
                            odd["home_odds"],

                        "Empate":
                            odd["draw_odds"],

                        "Visitante":
                            odd["away_odds"],

                    })

                if market_table:

                    st.dataframe(
                        pd.DataFrame(
                            market_table
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

            # =================================================
            # VALUE BETTING
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    📈 Value Betting
                </div>
                """,
                unsafe_allow_html=True
            )

            opportunities = []

            if (
                probabilities
                and market
                and market.get(
                    "status"
                ) == "OK"
            ):

                for odd in market[
                    "odds"
                ]:

                    outcomes = [

                        (
                            "H",
                            "Local",
                            odd["home_odds"]
                        ),

                        (
                            "D",
                            "Empate",
                            odd["draw_odds"]
                        ),

                        (
                            "A",
                            "Visitante",
                            odd["away_odds"]
                        ),

                    ]

                    for code, label, decimal_odds in outcomes:

                        if decimal_odds is None:
                            continue

                        probability = (
                            probabilities.get(
                                code
                            )
                        )

                        metrics = (
                            calculate_market_metrics(
                                probability,
                                decimal_odds
                            )
                        )

                        if metrics:

                            opportunities.append({

                                "Resultado":
                                    label,

                                "Casa":
                                    odd["bookmaker"],

                                "Cuota":
                                    decimal_odds,

                                "Probabilidad propia":
                                    probability,

                                "Probabilidad implícita":
                                    metrics[
                                        "implied_probability"
                                    ],

                                "Edge":
                                    metrics[
                                        "edge"
                                    ],

                                "EV":
                                    metrics[
                                        "ev"
                                    ],

                                "Value Score":
                                    metrics[
                                        "value_score"
                                    ],

                            })

            if not opportunities:

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

            else:

                opportunities_df = pd.DataFrame(
                    opportunities
                )

                opportunities_df = (
                    opportunities_df
                    .sort_values(
                        "EV",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )

                st.dataframe(
                    opportunities_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={

                        "Probabilidad propia":
                            st.column_config.NumberColumn(
                                format="%.2f%%"
                            ),

                        "Probabilidad implícita":
                            st.column_config.NumberColumn(
                                format="%.2f%%"
                            ),

                        "Edge":
                            st.column_config.NumberColumn(
                                format="%.2f%%"
                            ),

                        "EV":
                            st.column_config.NumberColumn(
                                format="%.2f%%"
                            ),

                        "Value Score":
                            st.column_config.NumberColumn(
                                format="%.2f"
                            ),

                    }
                )

                best_opportunity = (
                    opportunities_df.iloc[0]
                )

                st.success(
                    f"🎯 Mejor oportunidad detectada: "
                    f"{best_opportunity['Resultado']} "
                    f"· {best_opportunity['Casa']} "
                    f"· cuota "
                    f"{best_opportunity['Cuota']:.2f} "
                    f"· EV "
                    f"{best_opportunity['EV']:.2%}"
                )


# ============================================================
# RANKING
# ============================================================

st.divider()

st.markdown(
    """
    <div class="section-title">
        🏆 Ranking de oportunidades
    </div>
    """,
    unsafe_allow_html=True
)

ranking_rows = []

if (
    not df_events.empty
    and odds_api_key
):

    for _, row in df_events.iterrows():

        event_id = row["ID"]

        market = market_results.get(
            event_id
        )

        if not market:
            continue

        if market.get(
            "status"
        ) != "OK":
            continue

        home_history_raw, hs = (
            get_team_history(
                row["IDLocal"]
            )
        )

        away_history_raw, aws = (
            get_team_history(
                row["IDVisitante"]
            )
        )

        if (
            hs != "OK"
            or aws != "OK"
        ):
            continue

        home_history = history_to_dataframe(
            home_history_raw
        )

        away_history = history_to_dataframe(
            away_history_raw
        )

        training_df = build_training_dataset(
            home_history,
            away_history,
            history_window
        )

        model, model_status = train_model(
            training_df
        )

        if model is None:
            continue

        features = build_pre_match_features(
            row,
            home_history,
            away_history,
            history_window
        )

        if not features:
            continue

        probabilities = predict_match(
            model,
            features
        )

        if not probabilities:
            continue

        for odd in market[
            "odds"
        ]:

            for code, label, decimal_odds in [

                (
                    "H",
                    "Local",
                    odd["home_odds"]
                ),

                (
                    "D",
                    "Empate",
                    odd["draw_odds"]
                ),

                (
                    "A",
                    "Visitante",
                    odd["away_odds"]
                ),

            ]:

                if decimal_odds is None:
                    continue

                metrics = calculate_market_metrics(
                    probabilities.get(
                        code
                    ),
                    decimal_odds
                )

                if metrics is None:
                    continue

                ranking_rows.append({

                    "Evento":
                        row["Evento"],

                    "Liga":
                        row["Liga"],

                    "Resultado":
                        label,

                    "Casa":
                        odd["bookmaker"],

                    "Cuota":
                        decimal_odds,

                    "Probabilidad":
                        probabilities.get(
                            code
                        ),

                    "Probabilidad implícita":
                        metrics[
                            "implied_probability"
                        ],

                    "Edge":
                        metrics[
                            "edge"
                        ],

                    "EV":
                        metrics[
                            "ev"
                        ],

                    "Value Score":
                        metrics[
                            "value_score"
                        ],

                })


if ranking_rows:

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    ranking_df = (
        ranking_df
        .sort_values(
            "EV",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Probabilidad":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),

            "Probabilidad implícita":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),

            "Edge":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),

            "EV":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),

            "Value Score":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),

        }
    )

else:

    st.info(
        "No existen oportunidades cuantitativas "
        "completas para construir el ranking."
    )

    st.caption(
        "El ranking requiere simultáneamente "
        "probabilidad propia + cuota real coincidente "
        "+ probabilidad implícita + Edge + EV."
    )


# ============================================================
# INTEGRIDAD
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🔐 Integridad del sistema
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">

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

    🟢 Mercado obtenido desde Odds-API.io.

    <br><br>

    🟢 Eventos identificados mediante el ID propio
    de Odds-API.io.

    <br><br>

    🟢 Matching protegido entre equipos y fecha.

    <br><br>

    🟢 Cuotas 1X2 obtenidas de mercados ML reales.

    <br><br>

    🟢 Mejor cuota seleccionada entre las casas conectadas.

    <br><br>

    🟢 Probabilidad implícita calculada directamente
    desde la cuota decimal.

    <br><br>

    🟢 Edge calculado contra la probabilidad propia.

    <br><br>

    🟢 EV calculado con cuota real.

    <br><br>

    🟢 Value Score calculado únicamente con
    datos reales.

    <br><br>

    🔒 No se inventan cuotas.

    <br>

    🔒 No se inventan probabilidades.

    <br>

    🔒 No se fabrica Value Score.

    <br>

    🔒 No se utiliza información futura.

    <br><br>

    El sistema bloquea automáticamente cualquier
    cálculo cuando los datos no son suficientes,
    el evento no coincide de forma segura o
    no existe mercado real.

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

    st.markdown(
        """
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

        🟢 Activa

        ---

        ### CAPA 6 — Mercado real

        Odds-API.io
        ↓
        Eventos propios
        ↓
        Event ID
        ↓
        Bet365 + Unibet
        ↓
        Mercado ML / 1X2

        🟢 Activa

        ---

        ### CAPA 7 — Motor Quant

        Probabilidad propia
        +
        Cuota real
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

        🟢 Activa

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

        🟢 Activa

        ---

        ### FASE 16

        Integración robusta
        Odds-API.io

        Events
        ↓
        Matching de evento
        ↓
        Event ID oficial
        ↓
        Odds
        ↓
        ML / 1X2
        ↓
        Value Betting
        ↓
        Ranking
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE 16"
)

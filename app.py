import streamlit as st
import pandas as pd
import requests
import re
import unicodedata
from datetime import datetime, date, timedelta, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURACIÓN — FASE 12
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
DEFAULT_HISTORY_WINDOW = 5
MATCH_DATE_TOLERANCE_HOURS = 18

# IMPORTANTE:
# The Odds API anterior NO se consulta en esta fase.
# Se utiliza Odds-API.io como proveedor alternativo.
# La API key se introduce en la interfaz.
ODDS_PROVIDER_NAME = "Odds-API.io"


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

    text = str(value).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    replacements = {
        "&": " and ",
        "@": " at ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"\b(fc|cf|sc|afc|club|deportivo|de|cd|ac)\b",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return " ".join(
        text.split()
    )


def team_tokens(value):

    normalized = normalize_text(value)

    if not normalized:
        return set()

    return set(
        normalized.split()
    )


def team_similarity(name_a, name_b):

    a = normalize_text(name_a)
    b = normalize_text(name_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    tokens_a = team_tokens(a)
    tokens_b = team_tokens(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(
        tokens_a.intersection(tokens_b)
    )

    union = len(
        tokens_a.union(tokens_b)
    )

    jaccard = (
        intersection / union
        if union
        else 0
    )

    containment = 0

    if (
        a in b
        or b in a
    ):
        containment = 0.90

    return max(
        jaccard,
        containment
    )


def parse_datetime_utc(value):

    if value is None:
        return None

    try:

        text = str(value).strip()

        if not text:
            return None

        dt = pd.to_datetime(
            text,
            utc=True,
            errors="coerce"
        )

        if pd.isna(dt):
            return None

        return dt.to_pydatetime()

    except Exception:

        return None


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


def decimal_odds(value):

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

    url = (
        f"{TSDB_BASE_URL}/eventsday.php"
    )

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

        events = (
            data.get("events")
            or []
        )

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

            "ID": event.get(
                "idEvent"
            ),

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

            "Hora": format_time(
                event
            ),

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

        events = (
            data.get("results")
            or []
        )

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

            "ID": event.get(
                "idEvent"
            ),

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

            "GolesVisitante":
                away_score,

        })

    df = pd.DataFrame(
        rows
    )

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

            gf = row[
                "GolesLocal"
            ]

            ga = row[
                "GolesVisitante"
            ]

        else:

            gf = row[
                "GolesVisitante"
            ]

            ga = row[
                "GolesLocal"
            ]

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

    matches = len(
        relevant
    )

    return {

        "Partidos": matches,

        "Victorias": wins,

        "Empates": draws,

        "Derrotas": losses,

        "Puntos": points,

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

    home_name = match[
        "Local"
    ]

    away_name = match[
        "Visitante"
    ]

    match_date = match[
        "Fecha"
    ]

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

    all_history = pd.concat(
        [
            home_history,
            away_history
        ],
        ignore_index=True
    )

    all_history = (
        all_history
        .drop_duplicates(
            subset=["ID"]
        )
    )

    all_history = (
        all_history
        .dropna(
            subset=[
                "Fecha",
                "GolesLocal",
                "GolesVisitante"
            ]
        )
    )

    all_history = (
        all_history
        .sort_values("Fecha")
        .reset_index(
            drop=True
        )
    )

    rows = []

    for index, match in (
        all_history.iterrows()
    ):

        home_name = match[
            "Local"
        ]

        away_name = match[
            "Visitante"
        ]

        if not home_name or not away_name:
            continue

        previous_matches = (
            all_history
            .iloc[:index]
            .copy()
        )

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

        features[
            "target"
        ] = target

        features[
            "fecha"
        ] = match["Fecha"]

        rows.append(
            features
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FEATURE COLUMNS
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

def train_model(
    training_df
):

    if training_df.empty:

        return None, "NO_DATA"

    if len(training_df) < MIN_TRAINING_ROWS:

        return None, (
            "INSUFFICIENT_DATA"
        )

    training_df = (
        training_df
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    if (
        training_df["target"]
        .nunique()
        < 2
    ):

        return None, (
            "ONE_CLASS_ONLY"
        )

    split_index = int(
        len(training_df)
        * 0.80
    )

    if split_index < 10:

        return None, (
            "SMALL_TRAINING_SET"
        )

    train = (
        training_df
        .iloc[:split_index]
        .copy()
    )

    validation = (
        training_df
        .iloc[split_index:]
        .copy()
    )

    if validation.empty:

        return None, (
            "INVALID_VALIDATION"
        )

    if (
        train["target"]
        .nunique()
        < 2
    ):

        return None, (
            "INVALID_TRAIN_CLASSES"
        )

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "target"
    ]

    X_valid = validation[
        FEATURE_COLUMNS
    ]

    y_valid = validation[
        "target"
    ]

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
                model
                .named_steps[
                    "classifier"
                ]
                .classes_
            ),

    }

    return model, validation_info


# ============================================================
# PREDICCIÓN
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

    probabilities = (
        model
        .predict_proba(
            X[FEATURE_COLUMNS]
        )[0]
    )

    classes = (
        model
        .named_steps[
            "classifier"
        ]
        .classes_
    )

    result = {}

    for cls, prob in zip(
        classes,
        probabilities
    ):

        result[
            cls
        ] = float(prob)

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

@st.cache_data(ttl=300)
def get_odds_events(
    api_key,
    selected_date
):

    if not api_key:

        return [], (
            "NO_API_KEY"
        )

    start_dt = datetime.combine(
        selected_date,
        datetime.min.time()
    ).replace(
        tzinfo=timezone.utc
    )

    end_dt = (
        start_dt
        + timedelta(days=1)
    )

    params = {

        "apiKey":
            api_key,

        "sport":
            "football",

        "status":
            "pending",

        "from":
            start_dt.isoformat(),

        "to":
            end_dt.isoformat(),

    }

    try:

        response = requests.get(
            f"{ODDS_API_BASE_URL}/events",
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(
            data,
            list
        ):

            return [], (
                "INVALID_RESPONSE"
            )

        return data, "OK"

    except requests.RequestException as error:

        return [], (
            f"CONNECTION_ERROR: {error}"
        )

    except ValueError:

        return [], "JSON_ERROR"


@st.cache_data(ttl=120)
def get_event_odds(
    api_key,
    event_id,
    bookmakers
):

    if not api_key:
        return None, "NO_API_KEY"

    if not event_id:
        return None, "NO_EVENT_ID"

    params = {

        "apiKey":
            api_key,

        "eventId":
            str(event_id),

        "bookmakers":
            bookmakers,

    }

    try:

        response = requests.get(
            f"{ODDS_API_BASE_URL}/odds",
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        return data, "OK"

    except requests.RequestException as error:

        return None, (
            f"CONNECTION_ERROR: {error}"
        )

    except ValueError:

        return None, "JSON_ERROR"


# ============================================================
# MATCHING SPORTSDB ↔ ODDS-API.IO
# ============================================================

def calculate_event_match_score(
    sports_event,
    odds_event
):

    home_score = team_similarity(
        sports_event["Local"],
        odds_event.get("home")
    )

    away_score = team_similarity(
        sports_event["Visitante"],
        odds_event.get("away")
    )

    date_a = parse_datetime_utc(
        sports_event["Fecha"]
        + "T"
        + str(
            sports_event["Hora"]
        )
        + ":00"
        if sports_event["Hora"] != "--"
        else sports_event["Fecha"]
    )

    date_b = parse_datetime_utc(
        odds_event.get("date")
    )

    date_score = 0.0

    if (
        date_a is not None
        and date_b is not None
    ):

        difference = abs(
            (
                date_a
                - date_b
            ).total_seconds()
        )

        hours = (
            difference / 3600
        )

        if hours <= 2:

            date_score = 1.0

        elif hours <= 6:

            date_score = 0.85

        elif hours <= 12:

            date_score = 0.70

        elif hours <= 18:

            date_score = 0.50

    team_score = (
        home_score
        + away_score
    ) / 2

    total_score = (
        team_score * 0.85
        + date_score * 0.15
    )

    return {

        "score":
            total_score,

        "home_score":
            home_score,

        "away_score":
            away_score,

        "date_score":
            date_score,

    }


def match_events(
    sports_events,
    odds_events
):

    results = []

    for sports_event in sports_events:

        best_match = None

        best_details = None

        for odds_event in odds_events:

            details = (
                calculate_event_match_score(
                    sports_event,
                    odds_event
                )
            )

            if (
                best_details is None
                or
                details["score"]
                > best_details["score"]
            ):

                best_match = odds_event
                best_details = details

        if (
            best_match is not None
            and best_details["score"]
            >= 0.72
            and best_details["home_score"]
            >= 0.65
            and best_details["away_score"]
            >= 0.65
        ):

            results.append({

                "sports_event_id":
                    sports_event["ID"],

                "odds_event_id":
                    best_match.get("id"),

                "match_score":
                    best_details["score"],

                "home_score":
                    best_details["home_score"],

                "away_score":
                    best_details["away_score"],

                "date_score":
                    best_details["date_score"],

                "odds_event":
                    best_match,

            })

    return results


def build_market_index(
    sports_events,
    odds_events
):

    matches = match_events(
        sports_events,
        odds_events
    )

    index = {}

    for match in matches:

        index[
            match["sports_event_id"]
        ] = match

    return index


# ============================================================
# EXTRAER H2H DE ODDS-API.IO
# ============================================================

def extract_h2h_odds(
    odds_data
):

    if not odds_data:
        return []

    bookmakers = (
        odds_data.get(
            "bookmakers",
            {}
        )
    )

    if not isinstance(
        bookmakers,
        dict
    ):
        return []

    extracted = []

    for bookmaker_name, markets in (
        bookmakers.items()
    ):

        if not isinstance(
            markets,
            list
        ):
            continue

        for market in markets:

            market_name = str(
                market.get(
                    "name",
                    market.get(
                        "market",
                        ""
                    )
                )
            ).lower()

            market_key = str(
                market.get(
                    "key",
                    ""
                )
            ).lower()

            if not (
                market_key == "h2h"
                or
                market_name
                in [
                    "h2h",
                    "moneyline",
                    "match winner",
                    "1x2",
                    "winner",
                ]
            ):
                continue

            outcomes = market.get(
                "outcomes",
                []
            )

            if isinstance(
                outcomes,
                dict
            ):

                outcomes = [
                    {
                        "name":
                            name,
                        "price":
                            price,
                    }
                    for name, price
                    in outcomes.items()
                ]

            if not isinstance(
                outcomes,
                list
            ):
                continue

            values = {

                "home": None,

                "draw": None,

                "away": None,

            }

            for outcome in outcomes:

                name = outcome.get(
                    "name",
                    ""
                )

                price = decimal_odds(
                    outcome.get(
                        "price"
                    )
                )

                if price is None:
                    continue

                normalized = (
                    normalize_text(
                        name
                    )
                )

                if normalized == "draw":

                    values[
                        "draw"
                    ] = price

                elif (
                    normalized
                    == normalize_text(
                        odds_data.get(
                            "home"
                        )
                    )
                ):

                    values[
                        "home"
                    ] = price

                elif (
                    normalized
                    == normalize_text(
                        odds_data.get(
                            "away"
                        )
                    )
                ):

                    values[
                        "away"
                    ] = price

                else:

                    home_similarity = (
                        team_similarity(
                            name,
                            odds_data.get(
                                "home"
                            )
                        )
                    )

                    away_similarity = (
                        team_similarity(
                            name,
                            odds_data.get(
                                "away"
                            )
                        )
                    )

                    if (
                        home_similarity
                        > away_similarity
                        and home_similarity
                        >= 0.65
                    ):

                        values[
                            "home"
                        ] = price

                    elif (
                        away_similarity
                        >= 0.65
                    ):

                        values[
                            "away"
                        ] = price

            if any(
                value is not None
                for value in values.values()
            ):

                extracted.append({

                    "bookmaker":
                        bookmaker_name,

                    "home":
                        values["home"],

                    "draw":
                        values["draw"],

                    "away":
                        values["away"],

                })

    return extracted


def select_best_market(
    markets
):

    if not markets:
        return None

    candidates = []

    for market in markets:

        available = sum(
            value is not None
            for value in [
                market["home"],
                market["draw"],
                market["away"],
            ]
        )

        if available >= 2:

            candidates.append(
                market
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: sum(
            value is not None
            for value in [
                x["home"],
                x["draw"],
                x["away"],
            ]
        ),
        reverse=True
    )

    return candidates[0]


# ============================================================
# MARKET → QUANT
# ============================================================

def implied_probability(
    decimal_odd
):

    odd = decimal_odds(
        decimal_odd
    )

    if odd is None:
        return None

    return 1 / odd


def calculate_market_metrics(
    model_probabilities,
    odds
):

    if not model_probabilities:
        return None

    result = {}

    mapping = {

        "H": "home",

        "D": "draw",

        "A": "away",

    }

    for outcome, field in mapping.items():

        model_probability = (
            model_probabilities
            .get(outcome)
        )

        odd = odds.get(
            field
        )

        if (
            model_probability is None
            or odd is None
        ):
            continue

        implied = (
            implied_probability(
                odd
            )
        )

        if implied is None:
            continue

        edge = (
            model_probability
            - implied
        )

        ev = (
            model_probability
            * odd
            - 1
        )

        result[outcome] = {

            "odd":
                odd,

            "model_probability":
                model_probability,

            "implied_probability":
                implied,

            "edge":
                edge,

            "ev":
                ev,

        }

    if not result:
        return None

    for outcome in result:

        result[outcome][
            "value_score"
        ] = (
            result[outcome]["edge"]
            * 100
            + result[outcome]["ev"]
            * 100
        )

    return result


def build_value_ranking(
    opportunities
):

    rows = []

    for opportunity in opportunities:

        metrics = (
            opportunity.get(
                "metrics"
            )
        )

        if not metrics:
            continue

        for outcome, values in (
            metrics.items()
        ):

            if (
                values.get(
                    "odd"
                ) is None
                or
                values.get(
                    "edge"
                ) is None
                or
                values.get(
                    "ev"
                ) is None
            ):
                continue

            rows.append({

                "Partido":
                    opportunity[
                        "match"
                    ],

                "Mercado":
                    outcome,

                "Casa":
                    opportunity[
                        "bookmaker"
                    ],

                "Cuota":
                    values[
                        "odd"
                    ],

                "Prob. modelo":
                    values[
                        "model_probability"
                    ],

                "Prob. implícita":
                    values[
                        "implied_probability"
                    ],

                "Edge":
                    values[
                        "edge"
                    ],

                "EV":
                    values[
                        "ev"
                    ],

                "Value Score":
                    values[
                        "value_score"
                    ],

            })

    if not rows:
        return pd.DataFrame()

    ranking = pd.DataFrame(
        rows
    )

    ranking = ranking.sort_values(
        "Value Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    return ranking


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Centro de Mando"
    )

    st.caption(
        "Sports Data Hub — FASE 12"
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
        value=DEFAULT_HISTORY_WINDOW
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
        placeholder="Pega aquí tu nueva API Key"
    )

    bookmaker_options = [
        "Bet365",
        "Unibet",
        "Betway",
        "William Hill",
        "Betfred",
        "Betfair",
        "Paddy Power",
        "BetMGM",
        "SingBet",
    ]

    selected_bookmakers = st.multiselect(
        "Casas a consultar",
        bookmaker_options,
        default=[
            "Bet365",
            "Unibet"
        ]
    )

    if not selected_bookmakers:

        selected_bookmakers = [
            "Bet365",
            "Unibet"
        ]

    st.caption(
        "Proveedor: Odds-API.io · "
        "The Odds API no se consulta en esta fase."
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
# EVENTOS SPORTSDb
# ============================================================

events, event_status = (
    get_events_day(
        selected_date.strftime(
            "%Y-%m-%d"
        ),
        sport_filter
    )
)

df_events = events_to_dataframe(
    events
)


# ============================================================
# MERCADO — UNA CONSULTA GLOBAL
# ============================================================

odds_events = []
odds_status = "NO_API_KEY"

market_index = {}

if odds_api_key:

    if sport_filter == "Soccer":

        odds_events, odds_status = (
            get_odds_events(
                odds_api_key,
                selected_date
            )
        )

        if odds_status == "OK":

            market_index = (
                build_market_index(
                    events,
                    odds_events
                )
            )

    else:

        odds_status = (
            "SOCCER_ONLY_PHASE_12"
        )


# ============================================================
# ESTADOS
# ============================================================

status1, status2, status3 = (
    st.columns(3)
)

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

    if odds_status == "OK":

        st.markdown(
            """
            <div class="status status-green">
                🟢 MERCADO REAL
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
        "No fue posible conectar "
        "con TheSportsDB."
    )

elif event_status == "NO_EVENTS":

    st.info(
        f"No se encontraron eventos "
        f"para {selected_date.strftime('%d/%m/%Y')}."
    )

elif event_status != "OK":

    st.warning(
        f"Estado de la fuente deportiva: "
        f"{event_status}"
    )


if odds_api_key:

    if odds_status.startswith(
        "CONNECTION_ERROR"
    ):

        st.error(
            "No fue posible conectar "
            "con Odds-API.io."
        )

    elif odds_status == "INVALID_RESPONSE":

        st.error(
            "Odds-API.io devolvió "
            "una respuesta no válida."
        )

    elif odds_status == "OK":

        st.success(
            f"Mercado cargado: "
            f"{len(odds_events)} eventos "
            f"de Odds-API.io."
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

c1, c2, c3, c4 = (
    st.columns(4)
)

c1.metric(
    "Eventos",
    len(df_events)
)

c2.metric(
    "Deportes",
    (
        df_events["Deporte"].nunique()
        if not df_events.empty
        else 0
    )
)

c3.metric(
    "Ligas",
    (
        df_events["Liga"].nunique()
        if not df_events.empty
        else 0
    )
)

c4.metric(
    "Eventos con mercado",
    len(market_index)
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
                x
                for x in
                df_events[
                    "Liga"
                ]
                .dropna()
                .unique()
                if str(x).strip()
            ]
        )

        selected_league = (
            st.selectbox(
                "🏆 Competición",
                ["Todas"]
                + league_options
            )
        )

    with f2:

        country_options = sorted(
            [
                x
                for x in
                df_events[
                    "País"
                ]
                .dropna()
                .unique()
                if str(x).strip()
            ]
        )

        selected_country = (
            st.selectbox(
                "🌎 País",
                ["Todos"]
                + country_options
            )
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
        "No hay eventos deportivos "
        "para mostrar."
    )

else:

    for _, row in (
        filtered.iterrows()
    ):

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
                📍 {
                    location
                    or
                    "Ubicación no disponible"
                }
            </div>
            """,
            unsafe_allow_html=True
        )

        if event_id in market_index:

            st.markdown(
                """
                <div class="status status-green">
                    🟢 CUOTA COINCIDENTE
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="status status-yellow">
                    🟡 SIN CUOTA COINCIDENTE
                </div>
                """,
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

        selected = (
            selected_rows.iloc[0]
        )

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

        # ====================================================
        # COMPATIBILIDAD 1X2
        # ====================================================

        model_compatible = (

            bool(
                selected["IDLocal"]
            )

            and

            bool(
                selected["IDVisitante"]
            )

            and

            bool(
                selected["Local"]
            )

            and

            bool(
                selected["Visitante"]
            )

        )

        if not model_compatible:

            st.warning(
                "Este evento no tiene "
                "estructura Local vs "
                "Visitante compatible "
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

            a1, a2, a3, a4 = (
                st.columns(4)
            )

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

            h1, h2 = (
                st.columns(2)
            )

            with h1:

                if home_status == "OK":

                    st.success(
                        f"Local: "
                        f"{len(home_history)} "
                        f"registros"
                    )

                else:

                    st.warning(
                        f"Local: "
                        f"{home_status}"
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
                        f"Visitante: "
                        f"{away_status}"
                    )

            # =================================================
            # FORMA
            # =================================================

            combined_history = (
                pd.concat(
                    [
                        home_history,
                        away_history
                    ],
                    ignore_index=True
                )
                .drop_duplicates(
                    subset=["ID"]
                )
            )

            home_form = (
                calculate_team_form(
                    combined_history,
                    selected["Local"],
                    history_window,
                    before_date=selected["Fecha"]
                )
            )

            away_form = (
                calculate_team_form(
                    combined_history,
                    selected["Visitante"],
                    history_window,
                    before_date=selected["Fecha"]
                )
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
                            home_form[
                                "Partidos"
                            ],

                        selected["Visitante"]:
                            away_form[
                                "Partidos"
                            ],
                    },

                    {
                        "Variable":
                            "Puntos por partido",

                        selected["Local"]:
                            round(
                                home_form[
                                    "PPP"
                                ],
                                3
                            ),

                        selected["Visitante"]:
                            round(
                                away_form[
                                    "PPP"
                                ],
                                3
                            ),
                    },

                    {
                        "Variable":
                            "Goles por partido",

                        selected["Local"]:
                            round(
                                home_form[
                                    "GF"
                                ],
                                3
                            ),

                        selected["Visitante"]:
                            round(
                                away_form[
                                    "GF"
                                ],
                                3
                            ),
                    },

                    {
                        "Variable":
                            "Goles recibidos/partido",

                        selected["Local"]:
                            round(
                                home_form[
                                    "GC"
                                ],
                                3
                            ),

                        selected["Visitante"]:
                            round(
                                away_form[
                                    "GC"
                                ],
                                3
                            ),
                    },

                    {
                        "Variable":
                            "Victorias",

                        selected["Local"]:
                            home_form[
                                "Victorias"
                            ],

                        selected["Visitante"]:
                            away_form[
                                "Victorias"
                            ],
                    },

                    {
                        "Variable":
                            "Empates",

                        selected["Local"]:
                            home_form[
                                "Empates"
                            ],

                        selected["Visitante"]:
                            away_form[
                                "Empates"
                            ],
                    },

                    {
                        "Variable":
                            "Derrotas",

                        selected["Local"]:
                            home_form[
                                "Derrotas"
                            ],

                        selected["Visitante"]:
                            away_form[
                                "Derrotas"
                            ],
                    },

                    {
                        "Variable":
                            "Diferencia de goles",

                        selected["Local"]:
                            round(
                                home_form[
                                    "DG"
                                ],
                                3
                            ),

                        selected["Visitante"]:
                            round(
                                away_form[
                                    "DG"
                                ],
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
                    "No existe suficiente "
                    "información histórica "
                    "para construir el dataset."
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
                    training_df[
                        "target"
                    ]
                    .value_counts()
                    .to_dict()
                )

                d1, d2, d3 = (
                    st.columns(3)
                )

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

                if (
                    model_status
                    == "INSUFFICIENT_DATA"
                ):

                    st.warning(
                        "🟡 Modelo bloqueado: "
                        "no hay suficientes "
                        "observaciones "
                        "históricas válidas."
                    )

                elif (
                    model_status
                    == "ONE_CLASS_ONLY"
                ):

                    st.warning(
                        "🟡 Modelo bloqueado: "
                        "el histórico contiene "
                        "un único tipo de "
                        "resultado."
                    )

                elif (
                    model_status
                    == "INVALID_TRAIN_CLASSES"
                ):

                    st.warning(
                        "🟡 Modelo bloqueado: "
                        "la muestra de "
                        "entrenamiento no "
                        "contiene suficientes "
                        "clases."
                    )

                else:

                    st.warning(
                        f"🟡 Modelo no entrenado: "
                        f"{model_status}"
                    )

            else:

                st.success(
                    "🟢 Modelo entrenado y "
                    "validado con división "
                    "temporal."
                )

                v1, v2, v3 = (
                    st.columns(3)
                )

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
                                del modelo
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        p1, p2, p3 = (
                            st.columns(3)
                        )

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
                                "Visitante",

                        }

                        st.info(
                            f"Resultado con mayor "
                            f"probabilidad estimada: "
                            f"**{labels[best]}** "
                            f"({probabilities[best]:.1%})"
                        )

                        st.caption(
                            "Probabilidad generada "
                            "por el modelo estadístico. "
                            "No representa una cuota "
                            "ni una probabilidad "
                            "implícita de mercado."
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

            selected_market_match = (
                market_index.get(
                    selected_event_id
                )
            )

            market_metrics = None
            selected_odds_data = None

            if not odds_api_key:

                st.warning(
                    "🟡 Introduce la API Key "
                    "de Odds-API.io en la "
                    "barra lateral."
                )

            elif odds_status != "OK":

                st.warning(
                    "🟡 Mercado no disponible: "
                    f"{odds_status}"
                )

            elif (
                selected_market_match
                is None
            ):

                st.warning(
                    "🟡 No existe coincidencia "
                    "real entre TheSportsDB y "
                    "Odds-API.io para este evento."
                )

            else:

                odds_event_id = (
                    selected_market_match[
                        "odds_event_id"
                    ]
                )

                odds_data, odds_fetch_status = (
                    get_event_odds(
                        odds_api_key,
                        odds_event_id,
                        ",".join(
                            selected_bookmakers
                        )
                    )
                )

                if (
                    odds_fetch_status
                    != "OK"
                ):

                    st.warning(
                        "No fue posible "
                        "obtener las cuotas "
                        "del evento."
                    )

                else:

                    selected_odds_data = (
                        odds_data
                    )

                    markets = (
                        extract_h2h_odds(
                            odds_data
                        )
                    )

                    best_market = (
                        select_best_market(
                            markets
                        )
                    )

                    if best_market is None:

                        st.warning(
                            "🟡 El evento coincide "
                            "pero no contiene "
                            "mercado 1X2/h2h válido."
                        )

                    else:

                        st.success(
                            "🟢 Cuota real "
                            "coincidente."
                        )

                        m1, m2, m3 = (
                            st.columns(3)
                        )

                        m1.metric(
                            "🏠 Local",
                            (
                                f"{best_market['home']:.2f}"
                                if best_market[
                                    "home"
                                ] is not None
                                else "--"
                            )
                        )

                        m2.metric(
                            "🤝 Empate",
                            (
                                f"{best_market['draw']:.2f}"
                                if best_market[
                                    "draw"
                                ] is not None
                                else "--"
                            )
                        )

                        m3.metric(
                            "✈️ Visitante",
                            (
                                f"{best_market['away']:.2f}"
                                if best_market[
                                    "away"
                                ] is not None
                                else "--"
                            )
                        )

                        st.caption(
                            "Casa utilizada: "
                            f"{best_market['bookmaker']}"
                        )

                        if probabilities:

                            market_metrics = (
                                calculate_market_metrics(
                                    probabilities,
                                    best_market
                                )
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

            if market_metrics:

                value_rows = []

                labels = {

                    "H":
                        "🏠 Local",

                    "D":
                        "🤝 Empate",

                    "A":
                        "✈️ Visitante",

                }

                for outcome in [
                    "H",
                    "D",
                    "A"
                ]:

                    if outcome not in (
                        market_metrics
                    ):
                        continue

                    values = (
                        market_metrics[
                            outcome
                        ]
                    )

                    value_rows.append({

                        "Resultado":
                            labels[
                                outcome
                            ],

                        "Cuota":
                            round(
                                values[
                                    "odd"
                                ],
                                2
                            ),

                        "Prob. modelo":
                            f"{values['model_probability']:.1%}",

                        "Prob. implícita":
                            f"{values['implied_probability']:.1%}",

                        "Edge":
                            f"{values['edge']:.1%}",

                        "EV":
                            f"{values['ev']:.1%}",

                        "Value Score":
                            round(
                                values[
                                    "value_score"
                                ],
                                2
                            ),

                    })

                st.dataframe(
                    pd.DataFrame(
                        value_rows
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                valid_values = {

                    k: v
                    for k, v in
                    market_metrics.items()
                    if (
                        v.get(
                            "edge"
                        ) is not None
                        and
                        v.get(
                            "ev"
                        ) is not None
                    )
                }

                if valid_values:

                    best_value = max(
                        valid_values,
                        key=lambda key:
                            valid_values[
                                key
                            ]["value_score"]
                    )

                    labels = {

                        "H":
                            "Local",

                        "D":
                            "Empate",

                        "A":
                            "Visitante",

                    }

                    best_values = (
                        valid_values[
                            best_value
                        ]
                    )

                    st.success(
                        f"Mayor Value Score: "
                        f"**{labels[best_value]}** · "
                        f"Cuota "
                        f"**{best_values['odd']:.2f}** · "
                        f"Edge "
                        f"**{best_values['edge']:.1%}** · "
                        f"EV "
                        f"**{best_values['ev']:.1%}**"
                    )

            else:

                v1, v2, v3 = (
                    st.columns(3)
                )

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
                    "Edge, EV y Value Score "
                    "requieren simultáneamente "
                    "probabilidad propia y "
                    "cuota real coincidente."
                )


# ============================================================
# RANKING GLOBAL
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

ranking_opportunities = []

if (
    odds_api_key
    and odds_status == "OK"
    and not df_events.empty
):

    progress = st.progress(
        0,
        text="Construyendo ranking..."
    )

    total_events = len(
        df_events
    )

    for position, (_, row) in enumerate(
        df_events.iterrows(),
        start=1
    ):

        try:

            event_id = row["ID"]

            market_match = (
                market_index.get(
                    event_id
                )
            )

            if market_match is None:
                continue

            home_history_raw, _ = (
                get_team_history(
                    row["IDLocal"]
                )
            )

            away_history_raw, _ = (
                get_team_history(
                    row["IDVisitante"]
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

            training_df = (
                build_training_dataset(
                    home_history,
                    away_history,
                    history_window
                )
            )

            model, model_status = (
                train_model(
                    training_df
                )
            )

            if model is None:
                continue

            features = (
                build_pre_match_features(
                    row,
                    home_history,
                    away_history,
                    history_window
                )
            )

            if not features:
                continue

            probabilities = (
                predict_match(
                    model,
                    features
                )
            )

            if not probabilities:
                continue

            odds_data, odds_fetch_status = (
                get_event_odds(
                    odds_api_key,
                    market_match[
                        "odds_event_id"
                    ],
                    ",".join(
                        selected_bookmakers
                    )
                )
            )

            if (
                odds_fetch_status
                != "OK"
            ):
                continue

            markets = (
                extract_h2h_odds(
                    odds_data
                )
            )

            best_market = (
                select_best_market(
                    markets
                )
            )

            if best_market is None:
                continue

            metrics = (
                calculate_market_metrics(
                    probabilities,
                    best_market
                )
            )

            if not metrics:
                continue

            ranking_opportunities.append({

                "match":
                    row["Evento"],

                "bookmaker":
                    best_market[
                        "bookmaker"
                    ],

                "metrics":
                    metrics,

            })

        except Exception:

            continue

        progress.progress(
            position / total_events,
            text=(
                f"Analizando "
                f"{position}/{total_events}"
            )
        )

    progress.empty()


ranking_df = build_value_ranking(
    ranking_opportunities
)


if ranking_df.empty:

    st.info(
        "No existen oportunidades "
        "cuantitativas completas "
        "para construir el ranking."
    )

    st.caption(
        "El ranking requiere "
        "simultáneamente probabilidad "
        "propia, cuota real coincidente, "
        "probabilidad implícita, Edge y EV."
    )

else:

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Ranking construido con "
        f"{len(ranking_df)} oportunidades."
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

    🟢 Datos deportivos obtenidos de TheSportsDB.

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

    🟢 Mercado obtenido de Odds-API.io.

    <br><br>

    🟢 Matching de equipos y fecha entre fuentes.

    <br><br>

    🟢 Probabilidad implícita calculada directamente
    desde la cuota decimal.

    <br><br>

    🟢 Edge calculado contra la probabilidad propia.

    <br><br>

    🟢 EV calculado con cuota real.

    <br><br>

    🟢 Value Score calculado únicamente cuando
    existen probabilidad propia y mercado coincidente.

    <br><br>

    🔒 No se inventan cuotas.

    <br>

    🔒 No se inventan probabilidades.

    <br>

    🔒 No se fabrica Value Score.

    <br>

    🔒 No se utiliza información futura para
    construir variables históricas.

    <br><br>

    El sistema bloquea automáticamente cualquier
    cálculo cuando los datos no son suficientes
    o no existe coincidencia real de mercado.

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
        Eventos reales
        ↓
        Matching por equipos + fecha
        ↓
        Cuotas h2h / 1X2

        🟢 ACTIVA

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

        🟢 ACTIVA

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

        🟢 ACTIVA
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · "
    "Sports Data Hub · FASE 12"
)

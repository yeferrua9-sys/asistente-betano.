import streamlit as st
import pandas as pd
import requests
import math
import re
import unicodedata
from datetime import datetime, date

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

MIN_TRAINING_ROWS = 15

ODDS_API_BASE_URL = "https://odds-api.io/api/v3"

DEFAULT_BOOKMAKERS = [
    "Bet365",
    "Unibet",
]

DEFAULT_REGIONS = [
    "us",
    "uk",
    "eu",
]

DEFAULT_MARKET = "h2h"


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


def normalize_text(value):

    if value is None:
        return ""

    value = str(value).lower().strip()

    value = unicodedata.normalize(
        "NFKD",
        value
    ).encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return " ".join(
        value.split()
    )


def team_names_match(name_a, name_b):

    a = normalize_text(name_a)
    b = normalize_text(name_b)

    if not a or not b:
        return False

    if a == b:
        return True

    if a in b or b in a:
        return True

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return False

    overlap = len(
        tokens_a.intersection(tokens_b)
    )

    minimum = min(
        len(tokens_a),
        len(tokens_b)
    )

    return (
        overlap >= 1
        and overlap / minimum >= 0.5
    )


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


# ============================================================
# THE SPORTSDb — EVENTOS
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

            "ID":
                event.get("idEvent"),

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
# HISTÓRICO
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
        subset=["Fecha"]
    )

    df = df.sort_values(
        "Fecha",
        ascending=False
    )

    return df


# ============================================================
# FORMA
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

    matches = len(
        relevant
    )

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
# TARGET
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

    all_history = (
        all_history
        .drop_duplicates(
            subset=["ID"]
        )
        .dropna(
            subset=[
                "Fecha",
                "GolesLocal",
                "GolesVisitante"
            ]
        )
        .sort_values(
            "Fecha"
        )
        .reset_index(
            drop=True
        )
    )

    rows = []

    for index, match in all_history.iterrows():

        home_name = match[
            "Local"
        ]

        away_name = match[
            "Visitante"
        ]

        if (
            not home_name
            or not away_name
        ):
            continue

        previous_matches = (
            all_history.iloc[:index]
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

        features["target"] = target

        features["fecha"] = (
            match["Fecha"]
        )

        rows.append(
            features
        )

    return pd.DataFrame(
        rows
    )


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

    if (
        len(training_df)
        < MIN_TRAINING_ROWS
    ):
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

    train = training_df.iloc[
        :split_index
    ].copy()

    validation = training_df.iloc[
        split_index:
    ].copy()

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
                model.named_steps[
                    "classifier"
                ].classes_
            ),

    }

    return (
        model,
        validation_info
    )


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
        model.predict_proba(
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

        result[cls] = float(
            prob
        )

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

def extract_decimal_odds(value):

    try:

        value = float(value)

        if (
            math.isfinite(value)
            and value > 1.0
        ):
            return value

    except (
        TypeError,
        ValueError
    ):
        pass

    return None


def extract_bookmaker_name(
    bookmaker
):

    if isinstance(
        bookmaker,
        str
    ):
        return bookmaker

    if not isinstance(
        bookmaker,
        dict
    ):
        return "Unknown"

    for key in [
        "name",
        "title",
        "bookmaker",
        "bookmaker_name",
    ]:

        if bookmaker.get(key):
            return str(
                bookmaker[key]
            )

    return "Unknown"


def extract_outcome_name(
    outcome
):

    if isinstance(
        outcome,
        str
    ):
        return outcome

    if not isinstance(
        outcome,
        dict
    ):
        return ""

    for key in [
        "name",
        "label",
        "outcome",
        "team",
        "participant",
    ]:

        if outcome.get(key):
            return str(
                outcome[key]
            )

    return ""


def extract_outcome_price(
    outcome
):

    if not isinstance(
        outcome,
        dict
    ):
        return None

    for key in [
        "price",
        "odds",
        "decimal",
        "value",
    ]:

        value = extract_decimal_odds(
            outcome.get(key)
        )

        if value is not None:
            return value

    return None


def flatten_odds_response(
    data
):

    if isinstance(
        data,
        list
    ):
        return data

    if not isinstance(
        data,
        dict
    ):
        return []

    for key in [
        "data",
        "events",
        "matches",
        "odds",
        "fixtures",
        "results",
    ]:

        value = data.get(key)

        if isinstance(
            value,
            list
        ):
            return value

    return []


def event_matches_market(
    event,
    market
):

    home = (
        event.get("home_team")
        or event.get("homeTeam")
        or event.get("home")
        or event.get("team1")
        or event.get("participant1")
    )

    away = (
        event.get("away_team")
        or event.get("awayTeam")
        or event.get("away")
        or event.get("team2")
        or event.get("participant2")
    )

    market_home = (
        market.get("home_team")
        or market.get("homeTeam")
        or market.get("home")
        or market.get("team1")
        or market.get("participant1")
    )

    market_away = (
        market.get("away_team")
        or market.get("awayTeam")
        or market.get("away")
        or market.get("team2")
        or market.get("participant2")
    )

    if (
        home
        and away
        and market_home
        and market_away
    ):

        return (
            team_names_match(
                home,
                market_home
            )
            and
            team_names_match(
                away,
                market_away
            )
        )

    event_name = (
        event.get("name")
        or event.get("event")
        or event.get("match")
        or ""
    )

    market_name = (
        market.get("name")
        or market.get("event")
        or market.get("match")
        or ""
    )

    if (
        event_name
        and market_name
    ):

        return (
            normalize_text(
                event_name
            )
            == normalize_text(
                market_name
            )
        )

    return False


def parse_market_event(
    item
):

    if not isinstance(
        item,
        dict
    ):
        return []

    event_home = (
        item.get("home_team")
        or item.get("homeTeam")
        or item.get("home")
        or item.get("team1")
        or item.get("participant1")
    )

    event_away = (
        item.get("away_team")
        or item.get("awayTeam")
        or item.get("away")
        or item.get("team2")
        or item.get("participant2")
    )

    bookmakers = (
        item.get("bookmakers")
        or item.get("sportsbooks")
        or item.get("sites")
        or item.get("providers")
        or []
    )

    if isinstance(
        bookmakers,
        dict
    ):

        bookmakers = [
            {
                "name": key,
                **(
                    value
                    if isinstance(
                        value,
                        dict
                    )
                    else {
                        "odds": value
                    }
                )
            }
            for key, value
            in bookmakers.items()
        ]

    rows = []

    for bookmaker in bookmakers:

        bookmaker_name = (
            extract_bookmaker_name(
                bookmaker
            )
        )

        outcomes = (
            bookmaker.get("outcomes")
            or bookmaker.get("markets")
            or bookmaker.get("odds")
            or []
        )

        if isinstance(
            outcomes,
            dict
        ):

            possible_h2h = (
                outcomes.get("h2h")
                or outcomes.get("1x2")
                or outcomes.get("match_winner")
                or outcomes
            )

            outcomes = possible_h2h

        if isinstance(
            outcomes,
            dict
        ):

            outcomes = [
                {
                    "name": key,
                    "price": value
                }
                for key, value
                in outcomes.items()
            ]

        if not isinstance(
            outcomes,
            list
        ):
            continue

        for outcome in outcomes:

            name = extract_outcome_name(
                outcome
            )

            price = extract_outcome_price(
                outcome
            )

            if (
                name
                and price is not None
            ):

                rows.append({

                    "Home":
                        event_home or "",

                    "Away":
                        event_away or "",

                    "Bookmaker":
                        bookmaker_name,

                    "Outcome":
                        name,

                    "Odds":
                        price,

                })

    return rows


@st.cache_data(ttl=60)
def get_market_data(
    api_key,
    selected_date,
    sport,
    bookmakers
):

    if not api_key:
        return [], "NO_API_KEY"

    headers = {
        "x-api-key": api_key,
        "Authorization":
            f"Bearer {api_key}",
    }

    params = {
        "sport":
            sport.lower(),

        "date":
            selected_date,

        "bookmakers":
            ",".join(bookmakers),

        "market":
            DEFAULT_MARKET,
    }

    candidate_urls = [

        f"{ODDS_API_BASE_URL}/odds",

        f"{ODDS_API_BASE_URL}/odds-by-sport",

        f"{ODDS_API_BASE_URL}/odds-by-date",

    ]

    last_error = ""

    for url in candidate_urls:

        try:

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            if response.status_code == 404:
                continue

            if response.status_code in [
                401,
                403
            ]:

                return [], (
                    "INVALID_API_KEY"
                )

            response.raise_for_status()

            data = response.json()

            events = flatten_odds_response(
                data
            )

            if events:

                return events, "OK"

            last_error = (
                "EMPTY_RESPONSE"
            )

        except requests.RequestException as error:

            last_error = str(
                error
            )

        except ValueError:

            last_error = (
                "INVALID_JSON"
            )

    return [], (
        last_error
        or "NO_MARKET_DATA"
    )


def get_matching_market(
    selected,
    market_events,
    bookmakers
):

    matches = []

    for market_event in market_events:

        market_home = (
            market_event.get(
                "home_team"
            )
            or market_event.get(
                "homeTeam"
            )
            or market_event.get(
                "home"
            )
            or market_event.get(
                "team1"
            )
            or market_event.get(
                "participant1"
            )
        )

        market_away = (
            market_event.get(
                "away_team"
            )
            or market_event.get(
                "awayTeam"
            )
            or market_event.get(
                "away"
            )
            or market_event.get(
                "team2"
            )
            or market_event.get(
                "participant2"
            )
        )

        if (
            market_home
            and market_away
            and team_names_match(
                selected["Local"],
                market_home
            )
            and team_names_match(
                selected["Visitante"],
                market_away
            )
        ):

            matches.append(
                market_event
            )

    rows = []

    for event in matches:

        rows.extend(
            parse_market_event(
                event
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    allowed = [
        normalize_text(x)
        for x in bookmakers
    ]

    df = df[
        df["Bookmaker"]
        .map(normalize_text)
        .isin(allowed)
    ].copy()

    if df.empty:
        return df

    return df


# ============================================================
# NORMALIZACIÓN DEL OUTCOME
# ============================================================

def classify_outcome(
    outcome,
    home_name,
    away_name
):

    normalized = normalize_text(
        outcome
    )

    if normalized in [
        "draw",
        "tie",
        "x",
        "empate",
    ]:

        return "D"

    if (
        team_names_match(
            outcome,
            home_name
        )
    ):

        return "H"

    if (
        team_names_match(
            outcome,
            away_name
        )
    ):

        return "A"

    if normalized in [
        "home",
        "1",
        "local",
        "home team",
    ]:

        return "H"

    if normalized in [
        "away",
        "2",
        "visitante",
        "away team",
    ]:

        return "A"

    return None


def normalize_market_rows(
    market_df,
    selected
):

    if market_df.empty:
        return pd.DataFrame()

    df = market_df.copy()

    df["Selection"] = df[
        "Outcome"
    ].apply(
        lambda x:
        classify_outcome(
            x,
            selected["Local"],
            selected["Visitante"]
        )
    )

    df = df.dropna(
        subset=["Selection"]
    )

    return df


# ============================================================
# MOTOR QUANT
# ============================================================

def calculate_quant_metrics(
    probabilities,
    odds,
    selection
):

    if (
        probabilities is None
        or selection not in probabilities
    ):
        return None

    probability = float(
        probabilities[
            selection
        ]
    )

    if (
        odds is None
        or odds <= 1
    ):
        return None

    implied_probability = (
        1.0 / odds
    )

    edge = (
        probability
        - implied_probability
    )

    ev = (
        probability
        * odds
        - 1.0
    )

    value_score = (
        edge * 100
        + ev * 100
    )

    if (
        edge >= 0.05
        and ev >= 0.05
    ):

        classification = "VALUE"

    elif (
        edge >= 0.02
        and ev >= 0.02
    ):

        classification = (
            "VALUE MODERADO"
        )

    else:

        classification = (
            "NO VALUE"
        )

    return {

        "probability":
            probability,

        "odds":
            odds,

        "implied_probability":
            implied_probability,

        "edge":
            edge,

        "ev":
            ev,

        "value_score":
            value_score,

        "classification":
            classification,

    }


def build_quant_table(
    probabilities,
    market_df,
    selected
):

    if (
        probabilities is None
        or market_df.empty
    ):
        return pd.DataFrame()

    rows = []

    for _, row in market_df.iterrows():

        selection = row[
            "Selection"
        ]

        metrics = (
            calculate_quant_metrics(
                probabilities,
                row["Odds"],
                selection
            )
        )

        if metrics is None:
            continue

        rows.append({

            "Casa":
                row["Bookmaker"],

            "Selección":
                {
                    "H": "Local",
                    "D": "Empate",
                    "A": "Visitante",
                }[
                    selection
                ],

            "Cuota":
                metrics["odds"],

            "Prob. modelo":
                metrics[
                    "probability"
                ],

            "Prob. implícita":
                metrics[
                    "implied_probability"
                ],

            "Edge":
                metrics["edge"],

            "EV":
                metrics["ev"],

            "Value Score":
                metrics[
                    "value_score"
                ],

            "Estado":
                metrics[
                    "classification"
                ],

            "_selection":
                selection,

        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Value Score",
            ascending=False
        )
        .reset_index(drop=True)
    )


def build_best_market(
    quant_df
):

    if quant_df.empty:
        return pd.DataFrame()

    best_rows = []

    for selection in [
        "H",
        "D",
        "A"
    ]:

        subset = quant_df[
            quant_df[
                "_selection"
            ]
            == selection
        ]

        if subset.empty:
            continue

        best_rows.append(
            subset.iloc[0]
        )

    if not best_rows:
        return pd.DataFrame()

    return pd.DataFrame(
        best_rows
    ).sort_values(
        "Value Score",
        ascending=False
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Centro de Mando"
    )

    st.caption(
        "Sports Data Hub — FASE 13"
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

    selected_bookmakers = st.multiselect(
        "Casas a consultar",
        DEFAULT_BOOKMAKERS,
        default=DEFAULT_BOOKMAKERS
    )

    st.caption(
        "Proveedor: Odds-API.io · "
        "Motor Quant FASE 13"
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
# DATOS DEPORTIVOS
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
# MERCADO
# ============================================================

market_events = []
market_status = "NO_API_KEY"

if odds_api_key:

    market_events, market_status = (
        get_market_data(
            odds_api_key,
            selected_date.strftime(
                "%Y-%m-%d"
            ),
            sport_filter,
            selected_bookmakers
        )
    )


# ============================================================
# MATCHING GLOBAL
# ============================================================

market_match_count = 0

if (
    not df_events.empty
    and market_events
):

    for _, selected_event in (
        df_events.iterrows()
    ):

        matched = False

        for market_event in market_events:

            market_home = (
                market_event.get(
                    "home_team"
                )
                or market_event.get(
                    "homeTeam"
                )
                or market_event.get(
                    "home"
                )
                or market_event.get(
                    "team1"
                )
                or market_event.get(
                    "participant1"
                )
            )

            market_away = (
                market_event.get(
                    "away_team"
                )
                or market_event.get(
                    "awayTeam"
                )
                or market_event.get(
                    "away"
                )
                or market_event.get(
                    "team2"
                )
                or market_event.get(
                    "participant2"
                )
            )

            if (
                market_home
                and market_away
                and team_names_match(
                    selected_event["Local"],
                    market_home
                )
                and team_names_match(
                    selected_event[
                        "Visitante"
                    ],
                    market_away
                )
            ):

                matched = True
                break

        if matched:
            market_match_count += 1


# ============================================================
# ESTADOS
# ============================================================

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

    if not odds_api_key:

        label = (
            "🟡 MERCADO SIN API KEY"
        )

        css = "status-yellow"

    elif market_status != "OK":

        label = (
            "🔴 MERCADO NO DISPONIBLE"
        )

        css = "status-red"

    elif market_match_count == 0:

        label = (
            "🟡 SIN MATCHING"
        )

        css = "status-yellow"

    else:

        label = (
            "🟢 MERCADO CONECTADO"
        )

        css = "status-green"

    st.markdown(
        f"""
        <div class="status {css}">
            {label}
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
        f"para "
        f"{selected_date.strftime('%d/%m/%Y')}."
    )

elif event_status != "OK":

    st.warning(
        f"Estado de la fuente deportiva: "
        f"{event_status}"
    )

if odds_api_key:

    if market_status == "INVALID_API_KEY":

        st.error(
            "La API Key de Odds-API.io "
            "no fue aceptada."
        )

    elif market_status != "OK":

        st.warning(
            "El proveedor de mercado no "
            f"devolvió cuotas utilizables: "
            f"{market_status}"
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

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Eventos",
    len(df_events)
)

c2.metric(
    "Deportes",
    (
        df_events["Deporte"]
        .nunique()
        if not df_events.empty
        else 0
    )
)

c3.metric(
    "Ligas",
    (
        df_events["Liga"]
        .nunique()
        if not df_events.empty
        else 0
    )
)

c4.metric(
    "Eventos con mercado",
    market_match_count
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
                x
                for x in
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

    filtered = (
        df_events.copy()
    )

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
                    or "Ubicación no disponible"
                }
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # ESTADO DE MERCADO
        # ----------------------------------------------------

        market_found = False

        if market_events:

            for market_event in (
                market_events
            ):

                market_home = (
                    market_event.get(
                        "home_team"
                    )
                    or market_event.get(
                        "homeTeam"
                    )
                    or market_event.get(
                        "home"
                    )
                    or market_event.get(
                        "team1"
                    )
                )

                market_away = (
                    market_event.get(
                        "away_team"
                    )
                    or market_event.get(
                        "awayTeam"
                    )
                    or market_event.get(
                        "away"
                    )
                    or market_event.get(
                        "team2"
                    )
                )

                if (
                    market_home
                    and market_away
                    and team_names_match(
                        row["Local"],
                        market_home
                    )
                    and team_names_match(
                        row["Visitante"],
                        market_away
                    )
                ):

                    market_found = True
                    break

        if not odds_api_key:

            st.warning(
                "🟡 SIN API KEY"
            )

        elif market_found:

            st.success(
                "🟢 MERCADO COINCIDENTE"
            )

        else:

            st.warning(
                "🟡 SIN CUOTA COINCIDENTE"
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
        # COMPATIBILIDAD
        # ====================================================

        model_compatible = (
            bool(selected["IDLocal"])
            and bool(
                selected["IDVisitante"]
            )
            and bool(
                selected["Local"]
            )
            and bool(
                selected["Visitante"]
            )
        )

        if not model_compatible:

            st.warning(
                "Este evento no tiene "
                "estructura Local vs "
                "Visitante compatible."
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

            (
                home_history_raw,
                home_status
            ) = get_team_history(
                selected["IDLocal"]
            )

            (
                away_history_raw,
                away_status
            ) = get_team_history(
                selected["IDVisitante"]
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
                    before_date=selected[
                        "Fecha"
                    ]
                )
            )

            away_form = (
                calculate_team_form(
                    combined_history,
                    selected["Visitante"],
                    history_window,
                    before_date=selected[
                        "Fecha"
                    ]
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

                st.warning(
                    "🟡 Modelo bloqueado: "
                    "no existe suficiente "
                    "información histórica válida."
                )

            else:

                st.success(
                    "🟢 Modelo entrenado y "
                    "validado temporalmente."
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

            selected_market_df = pd.DataFrame()

            if not odds_api_key:

                st.warning(
                    "Introduce la API Key "
                    "de Odds-API.io."
                )

            elif market_status != "OK":

                st.warning(
                    "No existe mercado "
                    "disponible para este análisis."
                )

            else:

                selected_market_df = (
                    get_matching_market(
                        selected,
                        market_events,
                        selected_bookmakers
                    )
                )

                selected_market_df = (
                    normalize_market_rows(
                        selected_market_df,
                        selected
                    )
                )

                if selected_market_df.empty:

                    st.warning(
                        "🟡 No existe cuota "
                        "coincidente para "
                        "este evento."
                    )

                else:

                    st.success(
                        "🟢 Mercado coincidente "
                        "encontrado."
                    )

                    display_market = (
                        selected_market_df[
                            [
                                "Bookmaker",
                                "Outcome",
                                "Odds"
                            ]
                        ]
                        .rename(
                            columns={
                                "Bookmaker":
                                    "Casa",

                                "Outcome":
                                    "Selección",

                                "Odds":
                                    "Cuota",
                            }
                        )
                    )

                    st.dataframe(
                        display_market,
                        use_container_width=True,
                        hide_index=True
                    )

            # =================================================
            # MOTOR QUANT
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    📈 Motor Quant
                </div>
                """,
                unsafe_allow_html=True
            )

            if (
                probabilities is None
                or selected_market_df.empty
            ):

                q1, q2, q3 = (
                    st.columns(3)
                )

                q1.metric(
                    "Edge",
                    "BLOQUEADO"
                )

                q2.metric(
                    "EV",
                    "BLOQUEADO"
                )

                q3.metric(
                    "Value Score",
                    "BLOQUEADO"
                )

                st.info(
                    "El Motor Quant requiere "
                    "simultáneamente probabilidad "
                    "propia y mercado real "
                    "coincidente."
                )

            else:

                quant_df = (
                    build_quant_table(
                        probabilities,
                        selected_market_df,
                        selected
                    )
                )

                if quant_df.empty:

                    st.warning(
                        "No existen resultados "
                        "cuantitativos válidos."
                    )

                else:

                    best_market = (
                        build_best_market(
                            quant_df
                        )
                    )

                    if best_market.empty:

                        st.warning(
                            "No existe mercado "
                            "cuantitativo válido."
                        )

                    else:

                        top = (
                            best_market.iloc[0]
                        )

                        q1, q2, q3, q4 = (
                            st.columns(4)
                        )

                        q1.metric(
                            "Mejor selección",
                            top["Selección"]
                        )

                        q2.metric(
                            "Mejor cuota",
                            f"{top['Cuota']:.2f}"
                        )

                        q3.metric(
                            "Edge",
                            f"{top['Edge']:.1%}"
                        )

                        q4.metric(
                            "EV",
                            f"{top['EV']:.1%}"
                        )

                        if (
                            top["Estado"]
                            == "VALUE"
                        ):

                            st.success(
                                f"🟢 VALUE DETECTADO · "
                                f"{top['Selección']} · "
                                f"{top['Cuota']:.2f}"
                            )

                        elif (
                            top["Estado"]
                            == "VALUE MODERADO"
                        ):

                            st.warning(
                                f"🟡 VALUE MODERADO · "
                                f"{top['Selección']} · "
                                f"{top['Cuota']:.2f}"
                            )

                        else:

                            st.info(
                                "🔵 NO VALUE"
                            )

                        st.markdown(
                            """
                            ### Comparación de mercado
                            """
                        )

                        table_to_show = (
                            quant_df[
                                [
                                    "Casa",
                                    "Selección",
                                    "Cuota",
                                    "Prob. modelo",
                                    "Prob. implícita",
                                    "Edge",
                                    "EV",
                                    "Value Score",
                                    "Estado",
                                ]
                            ].copy()
                        )

                        st.dataframe(
                            table_to_show,
                            use_container_width=True,
                            hide_index=True,
                            column_config={

                                "Cuota":
                                    st.column_config.NumberColumn(
                                        format="%.2f"
                                    ),

                                "Prob. modelo":
                                    st.column_config.NumberColumn(
                                        format="%.1%"
                                    ),

                                "Prob. implícita":
                                    st.column_config.NumberColumn(
                                        format="%.1%"
                                    ),

                                "Edge":
                                    st.column_config.NumberColumn(
                                        format="%.1%"
                                    ),

                                "EV":
                                    st.column_config.NumberColumn(
                                        format="%.1%"
                                    ),

                                "Value Score":
                                    st.column_config.NumberColumn(
                                        format="%.2f"
                                    ),

                            }
                        )

                        st.caption(
                            "Edge = probabilidad propia "
                            "menos probabilidad implícita. "
                            "EV = probabilidad propia × "
                            "cuota − 1."
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

ranking_rows = []

if (
    odds_api_key
    and market_status == "OK"
    and not df_events.empty
):

    for _, event in df_events.iterrows():

        if (
            not event["IDLocal"]
            or not event["IDVisitante"]
        ):
            continue

        home_raw, hs = (
            get_team_history(
                event["IDLocal"]
            )
        )

        away_raw, aws = (
            get_team_history(
                event["IDVisitante"]
            )
        )

        if (
            hs != "OK"
            or aws != "OK"
        ):
            continue

        home_hist = (
            history_to_dataframe(
                home_raw
            )
        )

        away_hist = (
            history_to_dataframe(
                away_raw
            )
        )

        training = (
            build_training_dataset(
                home_hist,
                away_hist,
                history_window
            )
        )

        model, model_status = (
            train_model(
                training
            )
        )

        if model is None:
            continue

        features = (
            build_pre_match_features(
                event,
                home_hist,
                away_hist,
                history_window
            )
        )

        if not features:
            continue

        probs = predict_match(
            model,
            features
        )

        market_df = (
            get_matching_market(
                event,
                market_events,
                selected_bookmakers
            )
        )

        market_df = (
            normalize_market_rows(
                market_df,
                event
            )
        )

        quant = (
            build_quant_table(
                probs,
                market_df,
                event
            )
        )

        if quant.empty:
            continue

        best = quant.iloc[0]

        ranking_rows.append({

            "Partido":
                event["Evento"],

            "Casa":
                best["Casa"],

            "Selección":
                best["Selección"],

            "Cuota":
                best["Cuota"],

            "Prob. modelo":
                best["Prob. modelo"],

            "Prob. implícita":
                best["Prob. implícita"],

            "Edge":
                best["Edge"],

            "EV":
                best["EV"],

            "Value Score":
                best["Value Score"],

            "Estado":
                best["Estado"],

        })


if ranking_rows:

    ranking_df = (
        pd.DataFrame(
            ranking_rows
        )
        .sort_values(
            "Value Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Cuota":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),

            "Prob. modelo":
                st.column_config.NumberColumn(
                    format="%.1%"
                ),

            "Prob. implícita":
                st.column_config.NumberColumn(
                    format="%.1%"
                ),

            "Edge":
                st.column_config.NumberColumn(
                    format="%.1%"
                ),

            "EV":
                st.column_config.NumberColumn(
                    format="%.1%"
                ),

            "Value Score":
                st.column_config.NumberColumn(
                    format="%.2f"
                ),

        }
    )

else:

    st.info(
        "No existen oportunidades "
        "cuantitativas completas para "
        "construir el ranking."
    )

    st.caption(
        "El ranking requiere "
        "probabilidad propia + cuota "
        "real coincidente + "
        "probabilidad implícita + "
        "Edge + EV."
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

    🟢 Matching de equipos entre fuentes.

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
    cálculo cuando los datos no son suficientes o
    no existe coincidencia real de mercado.

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

        🟢 Activa

        ---

        ### CAPA 2 — Histórico

        Partidos anteriores → resultados

        🟢 Activa

        ---

        ### CAPA 3 — Feature Engineering

        Histórico anterior
        ↓
        Forma reciente
        ↓
        Goles
        ↓
        Puntos
        ↓
        Victorias / derrotas
        ↓
        Diferencias

        🟢 Activa

        ---

        ### CAPA 4 — Dataset

        Variables pre-partido
        +
        Resultado real

        🟢 Activa

        ---

        ### CAPA 5 — Modelo

        Regresión logística
        ↓
        Validación temporal
        ↓
        Probabilidad propia

        🟢 Activa

        ---

        ### CAPA 6 — Mercado

        Odds-API.io
        ↓
        Bet365 / Unibet
        ↓
        Matching
        ↓
        Cuotas reales

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

        🟢 FASE 13 ACTIVA

        ---

        ### CAPA 8 — Ranking

        Todas las oportunidades
        ↓
        Mejor cuota
        ↓
        Mayor Value Score

        🟢 FASE 13 ACTIVA

        ---

        ### CAPA 9 — Backtesting

        Histórico
        ↓
        Predicción
        ↓
        Cuota histórica
        ↓
        Value
        ↓
        Resultado
        ↓
        ROI

        🟡 SIGUIENTE FASE
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · "
    "Sports Data Hub · FASE 13"
)

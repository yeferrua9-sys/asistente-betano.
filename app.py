import streamlit as st
import pandas as pd
import requests
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

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return " ".join(text.split())


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
        minimum >= 2
        and overlap >= minimum
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


def decimal_to_probability(odds):

    try:

        odds = float(odds)

        if odds <= 1:
            return None

        return 1 / odds

    except (
        TypeError,
        ValueError
    ):

        return None


def calculate_edge(
    model_probability,
    market_probability
):

    if (
        model_probability is None
        or market_probability is None
    ):
        return None

    return (
        model_probability
        - market_probability
    )


def calculate_ev(
    model_probability,
    decimal_odds
):

    if (
        model_probability is None
        or decimal_odds is None
    ):
        return None

    try:

        odds = float(decimal_odds)

        if odds <= 1:
            return None

        return (
            model_probability * odds
        ) - 1

    except (
        TypeError,
        ValueError
    ):

        return None


def calculate_value_score(
    edge,
    ev
):

    if edge is None or ev is None:
        return None

    return (
        max(edge, 0) * 100
        + max(ev, 0) * 100
    )


def outcome_label(outcome):

    labels = {

        "H": "Local",

        "D": "Empate",

        "A": "Visitante",

    }

    return labels.get(
        outcome,
        outcome
    )


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
# FEATURES PRE-PARTIDO
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
        or away_form["Partidos"]
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

        home_name = match["Local"]
        away_name = match["Visitante"]

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
            or away_form["Partidos"]
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


# ============================================================
# FEATURES
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

def train_model(training_df):

    if training_df.empty:
        return None, "NO_DATA"

    if len(training_df) < MIN_TRAINING_ROWS:
        return None, "INSUFFICIENT_DATA"

    training_df = (
        training_df
        .sort_values("fecha")
        .reset_index(drop=True)
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

@st.cache_data(ttl=60)
def get_odds_api_events(
    api_key,
    sport="football"
):

    if not api_key:

        return [], "NO_API_KEY"

    url = (
        f"{ODDS_API_BASE_URL}/events"
    )

    params = {

        "apiKey":
            api_key,

        "sport":
            sport,

        "status":
            "pending",

        "limit":
            100,

    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code == 401:

            return [], "INVALID_API_KEY"

        if response.status_code == 403:

            return [], "API_ACCESS_DENIED"

        if response.status_code == 429:

            return [], "RATE_LIMIT"

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):

            events = (
                data.get("events")
                or data.get("results")
                or []
            )

        elif isinstance(data, list):

            events = data

        else:

            events = []

        if not events:

            return [], "NO_ODDS_EVENTS"

        return events, "OK"

    except requests.RequestException as error:

        return [], (
            f"CONNECTION_ERROR: {error}"
        )

    except ValueError:

        return [], "JSON_ERROR"


@st.cache_data(ttl=30)
def get_odds_for_event(
    api_key,
    event_id,
    bookmakers
):

    if (
        not api_key
        or not event_id
    ):

        return None, "NO_API_KEY"

    url = (
        f"{ODDS_API_BASE_URL}/odds"
    )

    params = {

        "apiKey":
            api_key,

        "eventId":
            event_id,

    }

    if bookmakers:

        params["bookmakers"] = (
            ",".join(bookmakers)
        )

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code == 401:

            return None, "INVALID_API_KEY"

        if response.status_code == 403:

            return None, "API_ACCESS_DENIED"

        if response.status_code == 429:

            return None, "RATE_LIMIT"

        response.raise_for_status()

        data = response.json()

        return data, "OK"

    except requests.RequestException as error:

        return None, (
            f"CONNECTION_ERROR: {error}"
        )

    except ValueError:

        return None, "JSON_ERROR"


def extract_decimal_odds(
    odds_data,
    home_name,
    away_name
):

    if not odds_data:
        return []

    if isinstance(
        odds_data,
        list
    ):

        if len(odds_data) == 1:

            root = odds_data[0]

        else:

            root = {
                "bookmakers": {}
            }

    else:

        root = odds_data

    bookmakers_data = (
        root.get(
            "bookmakers",
            {}
        )
        if isinstance(root, dict)
        else {}
    )

    if not isinstance(
        bookmakers_data,
        dict
    ):
        return []

    results = []

    for bookmaker_name, markets in (
        bookmakers_data.items()
    ):

        if not isinstance(
            markets,
            list
        ):
            continue

        for market in markets:

            if not isinstance(
                market,
                dict
            ):
                continue

            market_name = str(
                market.get(
                    "name",
                    market.get(
                        "market",
                        ""
                    )
                )
            ).lower()

            if (
                market_name
                not in [
                    "h2h",
                    "moneyline",
                    "match winner",
                    "1x2",
                    "winner",
                ]
            ):
                continue

            outcomes = (
                market.get(
                    "outcomes"
                )
                or market.get(
                    "selections"
                )
                or market.get(
                    "values"
                )
                or []
            )

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

            for outcome in outcomes:

                if not isinstance(
                    outcome,
                    dict
                ):
                    continue

                name = (
                    outcome.get("name")
                    or outcome.get("label")
                    or outcome.get("outcome")
                    or outcome.get("team")
                )

                price = (
                    outcome.get("price")
                    or outcome.get("odds")
                    or outcome.get("decimal")
                    or outcome.get("value")
                )

                if not name:
                    continue

                try:
                    price = float(price)
                except (
                    TypeError,
                    ValueError
                ):
                    continue

                if price <= 1:
                    continue

                if team_names_match(
                    name,
                    home_name
                ):

                    outcome_code = "H"

                elif team_names_match(
                    name,
                    away_name
                ):

                    outcome_code = "A"

                elif normalize_text(
                    name
                ) in [
                    "draw",
                    "empate",
                    "tie",
                    "x",
                ]:

                    outcome_code = "D"

                else:

                    continue

                results.append({

                    "bookmaker":
                        bookmaker_name,

                    "outcome":
                        outcome_code,

                    "outcome_name":
                        name,

                    "odds":
                        price,

                })

    return results


def select_best_market_odds(
    odds_rows
):

    best = {}

    for row in odds_rows:

        outcome = row[
            "outcome"
        ]

        if (
            outcome not in best
            or row["odds"]
            > best[outcome]["odds"]
        ):

            best[outcome] = row

    return best


def find_matching_odds_event(
    ts_event,
    odds_events
):

    home_name = ts_event[
        "Local"
    ]

    away_name = ts_event[
        "Visitante"
    ]

    event_date = str(
        ts_event[
            "Fecha"
        ]
    )

    candidates = []

    for event in odds_events:

        if not isinstance(
            event,
            dict
        ):
            continue

        odds_home = (
            event.get("home")
            or event.get(
                "homeTeam"
            )
            or event.get(
                "home_team"
            )
        )

        odds_away = (
            event.get("away")
            or event.get(
                "awayTeam"
            )
            or event.get(
                "away_team"
            )
        )

        if not odds_home or not odds_away:
            continue

        if not team_names_match(
            home_name,
            odds_home
        ):
            continue

        if not team_names_match(
            away_name,
            odds_away
        ):
            continue

        event_id = (
            event.get("id")
            or event.get("eventId")
            or event.get(
                "event_id"
            )
        )

        if not event_id:
            continue

        event_date_raw = (
            event.get("date")
            or event.get(
                "commence_time"
            )
            or event.get(
                "startTime"
            )
        )

        date_score = 0

        if event_date_raw:

            try:

                odds_date = str(
                    event_date_raw
                )[:10]

                if odds_date == event_date:

                    date_score = 1

            except Exception:
                pass

        candidates.append(
            (
                date_score,
                event
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[0][1]


def build_market_snapshot(
    ts_event,
    odds_events,
    api_key,
    bookmakers
):

    matched_event = (
        find_matching_odds_event(
            ts_event,
            odds_events
        )
    )

    if matched_event is None:

        return {
            "status":
                "NO_MATCH",
            "rows":
                [],
            "best":
                {},
            "event_id":
                None,
        }

    event_id = (
        matched_event.get("id")
        or matched_event.get(
            "eventId"
        )
        or matched_event.get(
            "event_id"
        )
    )

    odds_data, status = (
        get_odds_for_event(
            api_key,
            event_id,
            bookmakers
        )
    )

    if status != "OK":

        return {
            "status":
                status,
            "rows":
                [],
            "best":
                {},
            "event_id":
                event_id,
        }

    rows = extract_decimal_odds(
        odds_data,
        ts_event["Local"],
        ts_event["Visitante"]
    )

    best = select_best_market_odds(
        rows
    )

    if not best:

        return {
            "status":
                "NO_1X2_MARKET",
            "rows":
                [],
            "best":
                {},
            "event_id":
                event_id,
        }

    return {
        "status":
            "OK",
        "rows":
            rows,
        "best":
            best,
        "event_id":
            event_id,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Centro de Mando"
    )

    st.caption(
        "Sports Data Hub — FASE 14"
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
        placeholder="Pega aquí tu API Key"
    )

    bookmaker_options = [
        "Bet365",
        "Unibet",
        "Betway",
        "DraftKings",
        "FanDuel",
        "BetMGM",
        "BetRivers",
    ]

    selected_bookmakers = st.multiselect(
        "Casas a consultar",
        bookmaker_options,
        default=[
            "Bet365",
            "Unibet"
        ]
    )

    st.caption(
        "Proveedor: Odds-API.io · "
        "FASE 14 · Comparador real de cuotas"
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
# EVENTOS
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
# ODDS API — EVENTOS
# ============================================================

if odds_api_key:

    odds_events, odds_status = (
        get_odds_api_events(
            odds_api_key,
            "football"
        )
    )

else:

    odds_events = []
    odds_status = "NO_API_KEY"


# ============================================================
# MERCADO POR EVENTO
# ============================================================

market_cache = {}

if (
    odds_api_key
    and not df_events.empty
    and odds_status == "OK"
):

    for _, event_row in (
        df_events.iterrows()
    ):

        event_id = event_row["ID"]

        market_cache[
            event_id
        ] = build_market_snapshot(
            event_row,
            odds_events,
            odds_api_key,
            selected_bookmakers
        )


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

    if odds_status == "OK":

        st.markdown(
            """
            <div class="status status-green">
                🟢 MERCADO REAL
            </div>
            """,
            unsafe_allow_html=True
        )

    elif odds_status == "NO_API_KEY":

        st.markdown(
            """
            <div class="status status-yellow">
                🟡 MERCADO SIN API KEY
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status status-red">
                🔴 ERROR MERCADO
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


if odds_status == "INVALID_API_KEY":

    st.error(
        "La API Key de Odds-API.io "
        "no es válida."
    )

elif odds_status == "RATE_LIMIT":

    st.warning(
        "Odds-API.io alcanzó el límite "
        "de solicitudes. El sistema "
        "no inventará cuotas."
    )

elif odds_status == "API_ACCESS_DENIED":

    st.error(
        "Odds-API.io rechazó el acceso "
        "a la API."
    )

elif odds_status == "CONNECTION_ERROR":

    st.error(
        "No fue posible conectar "
        "con Odds-API.io."
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

if market_cache:

    events_with_market = sum(
        1
        for snapshot
        in market_cache.values()
        if snapshot["status"] == "OK"
    )

c1, c2, c3, c4 = st.columns(4)

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
                📅 {
                    format_date_display(
                        row["Fecha"]
                    )
                }
                &nbsp;&nbsp;
                ⏰ {
                    safe_value(
                        row["Hora"],
                        "--"
                    )
                }
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

        snapshot = market_cache.get(
            event_id
        )

        if (
            odds_api_key
            and odds_status == "OK"
        ):

            if snapshot:

                if snapshot["status"] == "OK":

                    st.success(
                        "🟢 MERCADO COINCIDENTE"
                    )

                elif snapshot[
                    "status"
                ] == "NO_MATCH":

                    st.warning(
                        "🟡 SIN CUOTA COINCIDENTE"
                    )

                elif snapshot[
                    "status"
                ] == "NO_1X2_MARKET":

                    st.warning(
                        "🟡 SIN MERCADO 1X2"
                    )

                else:

                    st.warning(
                        "🟡 MERCADO NO DISPONIBLE"
                    )

        elif not odds_api_key:

            st.warning(
                "🟡 SIN API KEY"
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

            bool(
                selected["IDLocal"]
            )

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
                "estructura Local vs Visitante "
                "compatible con el modelo 1X2."
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
                    "No existe suficiente "
                    "información histórica para "
                    "construir el dataset."
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

                if (
                    model_status
                    == "INSUFFICIENT_DATA"
                ):

                    st.warning(
                        "🟡 Modelo bloqueado: "
                        "no hay suficientes "
                        "observaciones históricas "
                        "válidas."
                    )

                elif (
                    model_status
                    == "ONE_CLASS_ONLY"
                ):

                    st.warning(
                        "🟡 Modelo bloqueado: "
                        "el histórico contiene "
                        "un único tipo de resultado."
                    )

                elif (
                    model_status
                    == "INVALID_TRAIN_CLASSES"
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
                    "🟢 Modelo entrenado y "
                    "validado con división temporal."
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
                    f"""
                    {
                        model_status[
                            "accuracy"
                        ]:.1%
                    }
                    """
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
                            f"""
                            {
                                probabilities["H"]
                                :.1%
                            }
                            """
                        )

                        p2.metric(
                            "🤝 Empate",
                            f"""
                            {
                                probabilities["D"]
                                :.1%
                            }
                            """
                        )

                        p3.metric(
                            "✈️ Visitante",
                            f"""
                            {
                                probabilities["A"]
                                :.1%
                            }
                            """
                        )

                        best = max(
                            probabilities,
                            key=probabilities.get
                        )

                        st.info(
                            f"Resultado con mayor "
                            f"probabilidad estimada: "
                            f"**{outcome_label(best)}** "
                            f"("
                            f"{probabilities[best]:.1%}"
                            f")"
                        )

                    else:

                        st.warning(
                            "No fue posible generar "
                            "la predicción."
                        )

                else:

                    st.warning(
                        "No existe suficiente "
                        "información pre-partido "
                        "para generar una predicción."
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

            selected_market = (
                market_cache.get(
                    selected_event_id
                )
            )

            if not odds_api_key:

                st.warning(
                    "🟡 Introduce la API Key "
                    "de Odds-API.io en la barra lateral."
                )

            elif odds_status != "OK":

                st.warning(
                    f"🟡 Mercado no disponible: "
                    f"{odds_status}"
                )

            elif not selected_market:

                st.warning(
                    "🟡 No existe información "
                    "de mercado para este evento."
                )

            elif (
                selected_market[
                    "status"
                ] != "OK"
            ):

                status_text = {

                    "NO_MATCH":
                        "No se encontró coincidencia "
                        "real entre el partido y "
                        "Odds-API.io.",

                    "NO_1X2_MARKET":
                        "Se encontró el evento, "
                        "pero no existe mercado 1X2 "
                        "disponible.",

                }.get(
                    selected_market[
                        "status"
                    ],
                    selected_market[
                        "status"
                    ]
                )

                st.warning(
                    f"🟡 {status_text}"
                )

            else:

                best_market = (
                    selected_market[
                        "best"
                    ]
                )

                market_rows = []

                for code in [
                    "H",
                    "D",
                    "A"
                ]:

                    if code in best_market:

                        row = best_market[
                            code
                        ]

                        market_rows.append({

                            "Resultado":
                                outcome_label(
                                    code
                                ),

                            "Casa":
                                row[
                                    "bookmaker"
                                ],

                            "Cuota":
                                row[
                                    "odds"
                                ],

                            "Prob. implícita":
                                (
                                    decimal_to_probability(
                                        row[
                                            "odds"
                                        ]
                                    )
                                ),

                        })

                if market_rows:

                    market_df = pd.DataFrame(
                        market_rows
                    )

                    market_df[
                        "Cuota"
                    ] = market_df[
                        "Cuota"
                    ].round(3)

                    market_df[
                        "Prob. implícita"
                    ] = (
                        market_df[
                            "Prob. implícita"
                        ]
                        .map(
                            lambda x:
                            f"{x:.2%}"
                            if x is not None
                            else "--"
                        )
                    )

                    st.dataframe(
                        market_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Se muestra la mejor cuota "
                        "decimal disponible entre "
                        "las casas seleccionadas."
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

            quant_rows = []

            if (
                probabilities
                and selected_market
                and selected_market[
                    "status"
                ] == "OK"
            ):

                best_market = (
                    selected_market[
                        "best"
                    ]
                )

                for code in [
                    "H",
                    "D",
                    "A"
                ]:

                    if code not in best_market:
                        continue

                    odds_row = best_market[
                        code
                    ]

                    model_probability = (
                        probabilities.get(
                            code
                        )
                    )

                    market_probability = (
                        decimal_to_probability(
                            odds_row["odds"]
                        )
                    )

                    edge = calculate_edge(
                        model_probability,
                        market_probability
                    )

                    ev = calculate_ev(
                        model_probability,
                        odds_row["odds"]
                    )

                    value_score = (
                        calculate_value_score(
                            edge,
                            ev
                        )
                    )

                    quant_rows.append({

                        "Resultado":
                            outcome_label(code),

                        "Casa":
                            odds_row[
                                "bookmaker"
                            ],

                        "Cuota":
                            odds_row[
                                "odds"
                            ],

                        "Prob. modelo":
                            model_probability,

                        "Prob. implícita":
                            market_probability,

                        "Edge":
                            edge,

                        "EV":
                            ev,

                        "Value Score":
                            value_score,

                    })

            if not quant_rows:

                q1, q2, q3 = st.columns(3)

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

                st.caption(
                    "El Motor Quant requiere "
                    "simultáneamente probabilidad "
                    "propia y mercado 1X2 coincidente."
                )

            else:

                quant_df = pd.DataFrame(
                    quant_rows
                )

                display_df = quant_df.copy()

                display_df[
                    "Cuota"
                ] = display_df[
                    "Cuota"
                ].map(
                    lambda x:
                    f"{x:.2f}"
                )

                for column in [
                    "Prob. modelo",
                    "Prob. implícita",
                    "Edge",
                    "EV"
                ]:

                    display_df[
                        column
                    ] = display_df[
                        column
                    ].map(
                        lambda x:
                        f"{x:.2%}"
                    )

                display_df[
                    "Value Score"
                ] = display_df[
                    "Value Score"
                ].map(
                    lambda x:
                    f"{x:.2f}"
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                positive = quant_df[
                    (
                        quant_df["Edge"]
                        > 0
                    )
                    &
                    (
                        quant_df["EV"]
                        > 0
                    )
                ]

                if positive.empty:

                    st.warning(
                        "🟡 Mercado disponible, "
                        "pero no se detecta Value "
                        "positivo con los datos actuales."
                    )

                else:

                    best_value = (
                        positive.sort_values(
                            "Value Score",
                            ascending=False
                        ).iloc[0]
                    )

                    st.success(
                        f"🟢 Mejor oportunidad detectada: "
                        f"**{best_value['Resultado']}** · "
                        f""
                        f"Cuota {best_value['Cuota']:.2f} · "
                        f""
                        f"Edge {best_value['Edge']:.2%} · "
                        f""
                        f"EV {best_value['EV']:.2%} · "
                        f""
                        f"Value Score "
                        f"{best_value['Value Score']:.2f}"
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
    odds_api_key
    and odds_status == "OK"
):

    for _, event_row in (
        df_events.iterrows()
    ):

        event_id = event_row[
            "ID"
        ]

        snapshot = market_cache.get(
            event_id
        )

        if not snapshot:
            continue

        if snapshot[
            "status"
        ] != "OK":
            continue

        (
            home_raw,
            _
        ) = get_team_history(
            event_row["IDLocal"]
        )

        (
            away_raw,
            _
        ) = get_team_history(
            event_row["IDVisitante"]
        )

        home_history = (
            history_to_dataframe(
                home_raw
            )
        )

        away_history = (
            history_to_dataframe(
                away_raw
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
                event_row,
                home_history,
                away_history,
                history_window
            )
        )

        if not features:
            continue

        probabilities = predict_match(
            model,
            features
        )

        if not probabilities:
            continue

        for code in [
            "H",
            "D",
            "A"
        ]:

            if code not in snapshot[
                "best"
            ]:
                continue

            odds_row = snapshot[
                "best"
            ][code]

            model_probability = (
                probabilities.get(
                    code
                )
            )

            implied_probability = (
                decimal_to_probability(
                    odds_row["odds"]
                )
            )

            edge = calculate_edge(
                model_probability,
                implied_probability
            )

            ev = calculate_ev(
                model_probability,
                odds_row["odds"]
            )

            value_score = (
                calculate_value_score(
                    edge,
                    ev
                )
            )

            if (
                edge is None
                or ev is None
                or value_score is None
            ):
                continue

            ranking_rows.append({

                "Evento":
                    event_row[
                        "Evento"
                    ],

                "Liga":
                    event_row[
                        "Liga"
                    ],

                "Resultado":
                    outcome_label(
                        code
                    ),

                "Casa":
                    odds_row[
                        "bookmaker"
                    ],

                "Cuota":
                    odds_row[
                        "odds"
                    ],

                "Prob. modelo":
                    model_probability,

                "Prob. implícita":
                    implied_probability,

                "Edge":
                    edge,

                "EV":
                    ev,

                "Value Score":
                    value_score,

            })


if not ranking_rows:

    st.info(
        "No existen oportunidades "
        "cuantitativas completas para "
        "construir el ranking."
    )

    st.caption(
        "El ranking requiere simultáneamente "
        "probabilidad propia + cuota real "
        "coincidente + probabilidad implícita "
        "+ Edge + EV."
    )

else:

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    ranking_df = (
        ranking_df
        .sort_values(
            "Value Score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    ranking_df.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking_df) + 1
        )
    )

    display_ranking = (
        ranking_df.copy()
    )

    display_ranking[
        "Cuota"
    ] = display_ranking[
        "Cuota"
    ].map(
        lambda x:
        f"{x:.2f}"
    )

    for column in [
        "Prob. modelo",
        "Prob. implícita",
        "Edge",
        "EV"
    ]:

        display_ranking[
            column
        ] = display_ranking[
            column
        ].map(
            lambda x:
            f"{x:.2%}"
        )

    display_ranking[
        "Value Score"
    ] = display_ranking[
        "Value Score"
    ].map(
        lambda x:
        f"{x:.2f}"
    )

    st.dataframe(
        display_ranking,
        use_container_width=True,
        hide_index=True
    )

    top = ranking_df.iloc[0]

    st.success(
        f"🏆 TOP 1 · "
        f"{top['Evento']} · "
        f"{top['Resultado']} · "
        f"Cuota {top['Cuota']:.2f} · "
        f"Edge {top['Edge']:.2%} · "
        f"EV {top['EV']:.2%} · "
        f"Value Score "
        f"{top['Value Score']:.2f}"
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

    🟢 Cuotas 1X2 obtenidas de mercados reales.

    <br><br>

    🟢 Probabilidad implícita calculada directamente
    desde la cuota decimal.

    <br><br>

    🟢 Edge calculado contra la probabilidad propia.

    <br><br>

    🟢 EV calculado con cuota real.

    <br><br>

    🟢 Value Score calculado únicamente cuando existen
    probabilidad propia y mercado coincidente.

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
        Eventos
        ↓
        Matching de equipos
        ↓
        Mercado 1X2
        ↓
        Cuotas decimales

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

        🟢 Activa

        ---

        ### CAPA 8 — Ranking

        Oportunidades cuantitativas
        ↓
        Ordenamiento por Value Score
        ↓
        TOP 1

        🟢 Activa

        ---

        ### CAPA 9 — Dashboard

        Datos
        +
        Histórico
        +
        Modelo
        +
        Mercado
        +
        Motor Quant
        +
        Ranking

        🟢 Activa
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · "
    "Sports Data Hub · FASE 14"
)

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
# NORMALIZACIÓN DE NOMBRES
# ============================================================

def normalize_name(value):

    if value is None:
        return ""

    text = str(value).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text
    ).encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return " ".join(
        text.split()
    )


def names_match(name_a, name_b):

    a = normalize_name(name_a)
    b = normalize_name(name_b)

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

    intersection = tokens_a.intersection(tokens_b)

    ratio = len(intersection) / max(
        len(tokens_a),
        len(tokens_b)
    )

    return ratio >= 0.60


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

        return [], f"CONNECTION_ERROR: {error}"

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
            matchup = f"{home} vs {away}"

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

        response.raise_for_status()

        data = response.json()

        events = data.get(
            "results"
        ) or []

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

            "GolesVisitante": away_score,

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

    relevant = relevant.head(
        n
    )

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
            row["Local"] == team_name
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

        "Partidos": matches,

        "Victorias": wins,

        "Empates": draws,

        "Derrotas": losses,

        "Puntos": points,

        "PPP": points / matches,

        "GF": goals_for / matches,

        "GC": goals_against / matches,

        "DG": (
            goals_for -
            goals_against
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
            home_form["PPP"]
            -
            away_form["PPP"],

        "gf_diff":
            home_form["GF"]
            -
            away_form["GF"],

        "gc_diff":
            home_form["GC"]
            -
            away_form["GC"],

        "dg_diff":
            home_form["DG"]
            -
            away_form["DG"],
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
        home_form["Partidos"] < window
        or away_form["Partidos"] < window
    ):
        return None

    return forms_to_features(
        home_form,
        away_form
    )


# ============================================================
# DATASET ENTRENAMIENTO
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

        features = forms_to_features(
            home_form,
            away_form
        )

        features["target"] = target
        features["fecha"] = match["Fecha"]

        rows.append(
            features
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COLUMNAS
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

@st.cache_data(ttl=60)
def get_odds_events(
    api_key,
    sport_slug,
    bookmakers
):

    if not api_key:
        return [], "NO_API_KEY"

    url = (
        f"{ODDS_API_BASE_URL}/events"
    )

    params = {

        "apiKey": api_key,

        "sport": sport_slug,

        "bookmaker": ",".join(
            bookmakers
        )

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
            return [], "FORBIDDEN"

        if response.status_code == 429:
            return [], "RATE_LIMIT"

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            events = (
                data.get("data")
                or data.get("events")
                or []
            )
        else:
            events = data or []

        return events, "OK"

    except requests.RequestException as error:

        return [], (
            f"CONNECTION_ERROR: {error}"
        )

    except ValueError:

        return [], "JSON_ERROR"


@st.cache_data(ttl=60)
def get_event_odds(
    api_key,
    event_id,
    bookmakers
):

    if not api_key:
        return None, "NO_API_KEY"

    url = (
        f"{ODDS_API_BASE_URL}/odds"
    )

    params = {

        "apiKey": api_key,

        "eventId": event_id,

        "bookmakers": ",".join(
            bookmakers
        )

    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code == 401:
            return None, "INVALID_API_KEY"

        if response.status_code == 403:
            return None, "FORBIDDEN"

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


# ============================================================
# EXTRACCIÓN DE MONEYLINE 1X2
# ============================================================

def extract_1x2_odds(
    odds_response,
    home_name,
    away_name
):

    if not odds_response:
        return []

    if isinstance(
        odds_response,
        list
    ):

        if len(
            odds_response
        ) == 1:

            event_data = odds_response[0]

        else:

            event_data = (
                odds_response[0]
                if odds_response
                else {}
            )

    else:

        event_data = odds_response

    response_home = event_data.get(
        "home"
    )

    response_away = event_data.get(
        "away"
    )

    if response_home and not names_match(
        home_name,
        response_home
    ):
        return []

    if response_away and not names_match(
        away_name,
        response_away
    ):
        return []

    bookmakers_data = event_data.get(
        "bookmakers"
    ) or {}

    results = []

    for bookmaker, markets in bookmakers_data.items():

        if not isinstance(
            markets,
            list
        ):
            continue

        for market in markets:

            market_name = str(
                market.get(
                    "name",
                    ""
                )
            ).lower()

            if market_name not in [
                "ml",
                "moneyline",
                "h2h",
                "1x2"
            ]:
                continue

            odds_list = market.get(
                "odds"
            ) or []

            for odds in odds_list:

                try:

                    home_odd = float(
                        odds.get(
                            "home"
                        )
                    )

                    away_odd = float(
                        odds.get(
                            "away"
                        )
                    )

                    draw_value = odds.get(
                        "draw"
                    )

                    draw_odd = (
                        float(draw_value)
                        if draw_value is not None
                        else None
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    continue

                if home_odd <= 1:
                    continue

                if away_odd <= 1:
                    continue

                if (
                    draw_odd is not None
                    and draw_odd <= 1
                ):
                    draw_odd = None

                results.append({

                    "bookmaker":
                        bookmaker,

                    "home_odd":
                        home_odd,

                    "draw_odd":
                        draw_odd,

                    "away_odd":
                        away_odd,

                    "updatedAt":
                        market.get(
                            "updatedAt"
                        )

                })

    return results


# ============================================================
# MEJORES CUOTAS
# ============================================================

def best_market_odds(
    odds_rows
):

    if not odds_rows:
        return None

    best_home = max(
        odds_rows,
        key=lambda x:
            x["home_odd"]
    )

    best_away = max(
        odds_rows,
        key=lambda x:
            x["away_odd"]
    )

    draw_rows = [
        x for x in odds_rows
        if x["draw_odd"] is not None
    ]

    best_draw = (
        max(
            draw_rows,
            key=lambda x:
                x["draw_odd"]
        )
        if draw_rows
        else None
    )

    return {

        "H": {
            "odd":
                best_home["home_odd"],
            "bookmaker":
                best_home["bookmaker"],
            "updatedAt":
                best_home["updatedAt"],
        },

        "D": (
            {
                "odd":
                    best_draw["draw_odd"],
                "bookmaker":
                    best_draw["bookmaker"],
                "updatedAt":
                    best_draw["updatedAt"],
            }
            if best_draw
            else None
        ),

        "A": {
            "odd":
                best_away["away_odd"],
            "bookmaker":
                best_away["bookmaker"],
            "updatedAt":
                best_away["updatedAt"],
        },

    }


# ============================================================
# CUOTA → PROBABILIDAD IMPLÍCITA
# ============================================================

def implied_probability(
    odd
):

    if odd is None:
        return None

    try:

        odd = float(
            odd
        )

        if odd <= 1:
            return None

        return 1 / odd

    except (
        TypeError,
        ValueError
    ):

        return None


# ============================================================
# EDGE
# ============================================================

def calculate_edge(
    model_probability,
    odd
):

    implied = implied_probability(
        odd
    )

    if (
        model_probability is None
        or implied is None
    ):
        return None

    return (
        model_probability
        - implied
    )


# ============================================================
# EV
# ============================================================

def calculate_ev(
    model_probability,
    odd
):

    if (
        model_probability is None
        or odd is None
    ):
        return None

    try:

        return (
            model_probability
            * float(odd)
        ) - 1

    except (
        TypeError,
        ValueError
    ):

        return None


# ============================================================
# VALUE SCORE
# ============================================================

def calculate_value_score(
    edge,
    ev
):

    if edge is None or ev is None:
        return None

    return (
        0.60 * edge
        +
        0.40 * ev
    )


# ============================================================
# RANKING
# ============================================================

def build_quant_row(
    event_row,
    probabilities,
    market
):

    if not probabilities or not market:
        return []

    output = []

    labels = {

        "H":
            event_row["Local"],

        "D":
            "Empate",

        "A":
            event_row["Visitante"],

    }

    for outcome in [
        "H",
        "D",
        "A"
    ]:

        market_item = market.get(
            outcome
        )

        if not market_item:
            continue

        odd = market_item["odd"]

        probability = probabilities.get(
            outcome,
            0.0
        )

        implied = implied_probability(
            odd
        )

        edge = calculate_edge(
            probability,
            odd
        )

        ev = calculate_ev(
            probability,
            odd
        )

        value_score = calculate_value_score(
            edge,
            ev
        )

        if (
            implied is None
            or edge is None
            or ev is None
            or value_score is None
        ):
            continue

        output.append({

            "Evento":
                event_row["Evento"],

            "Liga":
                event_row["Liga"],

            "Resultado":
                labels[outcome],

            "Casa":
                market_item[
                    "bookmaker"
                ],

            "Cuota":
                odd,

            "Probabilidad modelo":
                probability,

            "Probabilidad implícita":
                implied,

            "Edge":
                edge,

            "EV":
                ev,

            "Value Score":
                value_score,

        })

    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Centro de Mando"
    )

    st.caption(
        "Sports Data Hub — FASE 15"
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
        placeholder="Pega aquí tu nueva API Key"
    )

    bookmakers = st.multiselect(
        "Casas a consultar",
        [
            "Bet365",
            "Unibet"
        ],
        default=[
            "Bet365",
            "Unibet"
        ]
    )

    st.caption(
        "Odds-API.io · FASE 15 · "
        "Comparador real de cuotas + Motor Quant"
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
# OBTENER EVENTOS DEPORTIVOS
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
# MERCADO GLOBAL
# ============================================================

odds_status = "NO_API_KEY"
odds_events = []

if odds_api_key:

    odds_sport = "football"

    if sport_filter == "Basketball":
        odds_sport = "basketball"

    elif sport_filter == "Tennis":
        odds_sport = "tennis"

    elif sport_filter == "Baseball":
        odds_sport = "baseball"

    elif sport_filter == "Ice Hockey":
        odds_sport = "ice-hockey"

    elif sport_filter == "American Football":
        odds_sport = "american-football"

    odds_events, odds_status = get_odds_events(
        odds_api_key,
        odds_sport,
        bookmakers
    )


# ============================================================
# MATCHING EVENTOS
# ============================================================

def find_odds_event(
    event_row,
    odds_events
):

    target_date = str(
        event_row["Fecha"]
    )

    home_name = event_row[
        "Local"
    ]

    away_name = event_row[
        "Visitante"
    ]

    candidates = []

    for odds_event in odds_events:

        odds_home = odds_event.get(
            "home"
        )

        odds_away = odds_event.get(
            "away"
        )

        if not odds_home or not odds_away:
            continue

        if not names_match(
            home_name,
            odds_home
        ):
            continue

        if not names_match(
            away_name,
            odds_away
        ):
            continue

        odds_date = str(
            odds_event.get(
                "date",
                ""
            )
        )

        if odds_date:
            odds_date_only = (
                odds_date[:10]
            )

            if (
                odds_date_only
                != target_date
            ):
                continue

        candidates.append(
            odds_event
        )

    if not candidates:
        return None

    return candidates[0]


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
                🟢 MERCADO REAL CONECTADO
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


if odds_status == "INVALID_API_KEY":

    st.error(
        "La API Key de Odds-API.io no es válida."
    )

elif odds_status == "RATE_LIMIT":

    st.error(
        "Odds-API.io alcanzó el límite de solicitudes."
    )

elif odds_status == "FORBIDDEN":

    st.error(
        "Odds-API.io rechazó la solicitud."
    )

elif odds_status.startswith(
    "CONNECTION_ERROR"
):

    st.error(
        "No fue posible conectar con Odds-API.io."
    )


# ============================================================
# PRECALCULAR MERCADOS
# ============================================================

market_cache = {}

if not df_events.empty:

    for _, event_row in df_events.iterrows():

        event_id = event_row[
            "ID"
        ]

        if (
            not odds_api_key
            or not bookmakers
            or not event_row["Local"]
            or not event_row["Visitante"]
        ):

            market_cache[
                event_id
            ] = None

            continue

        odds_event = find_odds_event(
            event_row,
            odds_events
        )

        if not odds_event:

            market_cache[
                event_id
            ] = None

            continue

        odds_event_id = odds_event.get(
            "id"
        )

        if not odds_event_id:

            market_cache[
                event_id
            ] = None

            continue

        odds_response, single_status = (
            get_event_odds(
                odds_api_key,
                odds_event_id,
                bookmakers
            )
        )

        if single_status != "OK":

            market_cache[
                event_id
            ] = None

            continue

        odds_rows = extract_1x2_odds(
            odds_response,
            event_row["Local"],
            event_row["Visitante"]
        )

        market = best_market_odds(
            odds_rows
        )

        market_cache[
            event_id
        ] = market


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

market_count = sum(
    1
    for value in market_cache.values()
    if value
)

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
    market_count
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
            ["Todas"] +
            league_options
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
            ["Todos"] +
            country_options
        )

    filtered = df_events.copy()

    if selected_league != "Todas":

        filtered = filtered[
            filtered["Liga"]
            ==
            selected_league
        ]

    if selected_country != "Todos":

        filtered = filtered[
            filtered["País"]
            ==
            selected_country
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

        event_id = row[
            "ID"
        ]

        market = market_cache.get(
            event_id
        )

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

        if market:

            st.markdown(
                """
                <div class="status status-green">
                    🟢 CUOTA 1X2 CONECTADA
                </div>
                """,
                unsafe_allow_html=True
            )

        elif not odds_api_key:

            st.markdown(
                """
                <div class="status status-yellow">
                    🟡 SIN API KEY
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

selected_event_id = st.session_state.get(
    "selected_event_id"
)

if selected_event_id is not None:

    selected_rows = df_events[
        df_events["ID"]
        ==
        selected_event_id
    ]

    if not selected_rows.empty:

        selected = selected_rows.iloc[0]

        market = market_cache.get(
            selected_event_id
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
            and
            bool(selected["IDVisitante"])
            and
            bool(selected["Local"])
            and
            bool(selected["Visitante"])
        )

        if not model_compatible:

            st.warning(
                "Este evento no es compatible "
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

            training_df = build_training_dataset(
                home_history,
                away_history,
                history_window
            )

            model = None
            model_status = "INSUFFICIENT_DATA"

            if training_df.empty:

                st.warning(
                    "No existe suficiente información "
                    "histórica para construir el dataset."
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

                model, model_status = train_model(
                    training_df
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
                    "no hay suficientes observaciones "
                    "históricas válidas."
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

                    probabilities = predict_match(
                        model,
                        current_features
                    )

                    if probabilities:

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

            if market:

                market_rows = []

                labels = {

                    "H":
                        selected["Local"],

                    "D":
                        "Empate",

                    "A":
                        selected["Visitante"],

                }

                for outcome in [
                    "H",
                    "D",
                    "A"
                ]:

                    item = market.get(
                        outcome
                    )

                    if not item:
                        continue

                    market_rows.append({

                        "Resultado":
                            labels[outcome],

                        "Casa":
                            item["bookmaker"],

                        "Cuota":
                            item["odd"],

                        "Actualizada":
                            item["updatedAt"]
                            or
                            "No disponible",

                    })

                st.dataframe(
                    pd.DataFrame(
                        market_rows
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            elif not odds_api_key:

                st.warning(
                    "🟡 Introduce la API Key "
                    "de Odds-API.io en la barra lateral."
                )

            else:

                st.warning(
                    "🟡 No existe cuota 1X2 "
                    "coincidente para este evento."
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
                and market
            ):

                quant_rows = build_quant_row(
                    selected,
                    probabilities,
                    market
                )

            if quant_rows:

                quant_df = pd.DataFrame(
                    quant_rows
                )

                st.dataframe(
                    quant_df.style.format({

                        "Cuota":
                            "{:.2f}",

                        "Probabilidad modelo":
                            "{:.1%}",

                        "Probabilidad implícita":
                            "{:.1%}",

                        "Edge":
                            "{:.1%}",

                        "EV":
                            "{:.1%}",

                        "Value Score":
                            "{:.4f}",

                    }),
                    use_container_width=True,
                    hide_index=True
                )

                best_quant = max(
                    quant_rows,
                    key=lambda x:
                        x["Value Score"]
                )

                q1, q2, q3 = st.columns(3)

                q1.metric(
                    "Mejor oportunidad",
                    best_quant["Resultado"]
                )

                q2.metric(
                    "Edge",
                    f"{best_quant['Edge']:.2%}"
                )

                q3.metric(
                    "EV",
                    f"{best_quant['EV']:.2%}"
                )

                if (
                    best_quant["EV"] > 0
                    and best_quant["Edge"] > 0
                ):

                    st.success(
                        "🟢 VALUE DETECTADO: "
                        "probabilidad propia superior "
                        "a la probabilidad implícita "
                        "y EV positivo."
                    )

                else:

                    st.info(
                        "🟡 No existe value positivo "
                        "en este evento."
                    )

            else:

                st.info(
                    "🟡 Motor Quant bloqueado: "
                    "requiere simultáneamente "
                    "probabilidad propia y mercado "
                    "1X2 coincidente."
                )


# ============================================================
# RANKING FINAL
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

if not df_events.empty:

    for _, event_row in df_events.iterrows():

        event_id = event_row[
            "ID"
        ]

        market = market_cache.get(
            event_id
        )

        if not market:
            continue

        home_id = event_row[
            "IDLocal"
        ]

        away_id = event_row[
            "IDVisitante"
        ]

        if not home_id or not away_id:
            continue

        home_raw, hs = get_team_history(
            home_id
        )

        away_raw, aws = get_team_history(
            away_id
        )

        if hs != "OK" or aws != "OK":
            continue

        home_history = history_to_dataframe(
            home_raw
        )

        away_history = history_to_dataframe(
            away_raw
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
            event_row,
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

        rows = build_quant_row(
            event_row,
            probabilities,
            market
        )

        ranking_rows.extend(
            rows
        )


if ranking_rows:

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    ranking_df = ranking_df.sort_values(
        "Value Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    ranking_df.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking_df) + 1
        )
    )

    st.dataframe(
        ranking_df.style.format({

            "Cuota":
                "{:.2f}",

            "Probabilidad modelo":
                "{:.1%}",

            "Probabilidad implícita":
                "{:.1%}",

            "Edge":
                "{:.1%}",

            "EV":
                "{:.1%}",

            "Value Score":
                "{:.4f}",

        }),
        use_container_width=True,
        hide_index=True
    )

    best = ranking_df.iloc[0]

    st.success(
        f"🏆 Mejor oportunidad actual: "
        f"**{best['Evento']} — "
        f"{best['Resultado']}** | "
        f"{best['Casa']} | "
        f"Cuota {best['Cuota']:.2f} | "
        f"Edge {best['Edge']:.1%} | "
        f"EV {best['EV']:.1%}"
    )

else:

    st.info(
        "No existen oportunidades cuantitativas "
        "completas para construir el ranking."
    )

    st.caption(
        "Se requiere probabilidad propia + "
        "cuota real coincidente + "
        "probabilidad implícita + Edge + EV."
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

    🟢 Mejor cuota seleccionada entre las casas conectadas.

    <br><br>

    🟢 Probabilidad implícita calculada directamente
    desde la cuota decimal.

    <br><br>

    🟢 Edge calculado contra la probabilidad propia.

    <br><br>

    🟢 EV calculado con cuota real.

    <br><br>

    🟢 Value Score calculado únicamente con datos reales.

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
        Victorias / Derrotas
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
        Dataset histórico
        ↓
        Regresión logística
        ↓
        Validación temporal
        ↓
        Probabilidades propias
        🟢 Activa

        ---

        ### CAPA 6 — Mercado real
        Odds-API.io
        ↓
        Bet365 / Unibet
        ↓
        Matching de eventos
        ↓
        Cuotas 1X2
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
        Todas las oportunidades
        ↓
        Ordenadas por Value Score
        ↓
        Mejor oportunidad
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

        🟢 SISTEMA INTEGRADO
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE 15 · "
    "Sistema Quant Integrado"
)

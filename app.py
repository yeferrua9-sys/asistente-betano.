import streamlit as st
import pandas as pd
import requests
import re
import unicodedata
from datetime import date
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/123"
ODDS_BASE = "https://api.odds-api.io/v3"

MIN_TRAINING_ROWS = 15
DEFAULT_WINDOW = 5


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
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .title {
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

    .match {
        background: #111720;
        border: 1px solid #293241;
        border-radius: 18px;
        padding: 22px;
        margin: 12px 0;
    }

    .team {
        font-size: 22px;
        font-weight: 700;
        margin: 6px 0;
    }

    .gray {
        color: #9ca3af;
    }

    .section {
        font-size: 27px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 15px;
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

def normalize_text(value):
    if value is None:
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKD",
        text
    ).encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    text = text.lower()
    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def token_set(value):
    return {
        token
        for token in normalize_text(value).split()
        if len(token) >= 3
    }


def team_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.90

    ta = token_set(a)
    tb = token_set(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def pair_similarity(home_a, away_a, home_b, away_b):
    direct = (
        team_similarity(home_a, home_b)
        + team_similarity(away_a, away_b)
    ) / 2

    reverse = (
        team_similarity(home_a, away_b)
        + team_similarity(away_a, home_b)
    ) / 2

    return max(direct, reverse)


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value):
    try:
        value = float(value)

        if value <= 1:
            return None

        return value

    except (TypeError, ValueError):
        return None


def format_date(value):
    if not value:
        return "--"

    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


# ============================================================
# THESPORTSDB
# ============================================================

@st.cache_data(ttl=300)
def get_events(selected_date):

    try:
        response = requests.get(
            f"{TSDB_BASE}/eventsday.php",
            params={
                "d": selected_date,
                "s": "Soccer",
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("events") or [], "OK"

    except requests.RequestException as error:
        return [], f"CONNECTION_ERROR: {error}"

    except ValueError:
        return [], "INVALID_JSON"


@st.cache_data(ttl=3600)
def get_team_history(team_id):

    if not team_id:
        return [], "NO_TEAM_ID"

    try:
        response = requests.get(
            f"{TSDB_BASE}/eventslast.php",
            params={
                "id": team_id
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("results") or [], "OK"

    except requests.RequestException as error:
        return [], f"CONNECTION_ERROR: {error}"

    except ValueError:
        return [], "INVALID_JSON"


def history_to_df(events):

    rows = []

    for event in events:

        rows.append(
            {
                "ID": event.get("idEvent"),
                "Fecha": event.get("dateEvent"),
                "Local": event.get("strHomeTeam"),
                "Visitante": event.get("strAwayTeam"),
                "GolesLocal": safe_int(
                    event.get("intHomeScore")
                ),
                "GolesVisitante": safe_int(
                    event.get("intAwayScore")
                ),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["Fecha"] = pd.to_datetime(
        df["Fecha"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Fecha",
            "GolesLocal",
            "GolesVisitante",
        ]
    )

    df = df.drop_duplicates(
        subset=["ID"]
    )

    return df.sort_values("Fecha")


# ============================================================
# FORMA
# ============================================================

def calculate_form(
    history,
    team,
    before_date,
    window
):

    if history.empty:
        return None

    before_date = pd.to_datetime(
        before_date
    )

    data = history[
        (
            history["Local"] == team
        )
        |
        (
            history["Visitante"] == team
        )
    ].copy()

    data = data[
        data["Fecha"] < before_date
    ]

    data = data.sort_values(
        "Fecha",
        ascending=False
    ).head(window)

    if len(data) < window:
        return None

    points = 0
    wins = 0
    draws = 0
    losses = 0
    goals_for = 0
    goals_against = 0

    for _, row in data.iterrows():

        if row["Local"] == team:

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

    matches = len(data)

    return {
        "matches": matches,
        "ppp": points / matches,
        "gf": goals_for / matches,
        "ga": goals_against / matches,
        "dg": (goals_for - goals_against) / matches,
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "home_ppp",
    "away_ppp",
    "home_gf",
    "away_gf",
    "home_ga",
    "away_ga",
    "home_dg",
    "away_dg",
    "home_wins",
    "away_wins",
    "home_losses",
    "away_losses",
    "ppp_diff",
    "gf_diff",
    "ga_diff",
    "dg_diff",
]


def make_features(home_form, away_form):

    return {
        "home_ppp": home_form["ppp"],
        "away_ppp": away_form["ppp"],
        "home_gf": home_form["gf"],
        "away_gf": away_form["gf"],
        "home_ga": home_form["ga"],
        "away_ga": away_form["ga"],
        "home_dg": home_form["dg"],
        "away_dg": away_form["dg"],
        "home_wins": home_form["wins"],
        "away_wins": away_form["wins"],
        "home_losses": home_form["losses"],
        "away_losses": away_form["losses"],
        "ppp_diff": (
            home_form["ppp"]
            - away_form["ppp"]
        ),
        "gf_diff": (
            home_form["gf"]
            - away_form["gf"]
        ),
        "ga_diff": (
            home_form["ga"]
            - away_form["ga"]
        ),
        "dg_diff": (
            home_form["dg"]
            - away_form["dg"]
        ),
    }


def result_target(home_score, away_score):

    if home_score > away_score:
        return "H"

    if home_score < away_score:
        return "A"

    return "D"


# ============================================================
# DATASET HISTÓRICO
# ============================================================

def build_dataset(history, window):

    if history.empty:
        return pd.DataFrame()

    history = history.sort_values(
        "Fecha"
    ).reset_index(
        drop=True
    )

    rows = []

    for index, match in history.iterrows():

        previous = history.iloc[:index]

        home = match["Local"]
        away = match["Visitante"]

        if not home or not away:
            continue

        home_form = calculate_form(
            previous,
            home,
            match["Fecha"],
            window
        )

        away_form = calculate_form(
            previous,
            away,
            match["Fecha"],
            window
        )

        if (
            home_form is None
            or away_form is None
        ):
            continue

        features = make_features(
            home_form,
            away_form
        )

        features["target"] = result_target(
            match["GolesLocal"],
            match["GolesVisitante"]
        )

        features["Fecha"] = match["Fecha"]

        rows.append(features)

    return pd.DataFrame(rows)


# ============================================================
# MODELO
# ============================================================

def train_model(dataset):

    if dataset.empty:
        return None, None

    if len(dataset) < MIN_TRAINING_ROWS:
        return None, None

    dataset = dataset.sort_values(
        "Fecha"
    ).reset_index(
        drop=True
    )

    if dataset["target"].nunique() < 2:
        return None, None

    split = int(
        len(dataset) * 0.80
    )

    train = dataset.iloc[:split]
    validation = dataset.iloc[split:]

    if len(train) < 10:
        return None, None

    if validation.empty:
        return None, None

    if train["target"].nunique() < 2:
        return None, None

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            ),
        ]
    )

    model.fit(
        train[FEATURES],
        train["target"]
    )

    predictions = model.predict(
        validation[FEATURES]
    )

    accuracy = accuracy_score(
        validation["target"],
        predictions
    )

    return model, {
        "train": len(train),
        "validation": len(validation),
        "accuracy": accuracy,
    }


def predict_probabilities(
    model,
    features
):

    if model is None:
        return None

    X = pd.DataFrame(
        [features]
    )[FEATURES]

    probabilities = model.predict_proba(X)[0]

    classes = model.named_steps[
        "classifier"
    ].classes_

    result = {
        "H": 0.0,
        "D": 0.0,
        "A": 0.0,
    }

    for cls, probability in zip(
        classes,
        probabilities
    ):
        result[cls] = float(probability)

    return result


# ============================================================
# ODDS API
# ============================================================

def odds_request(
    endpoint,
    api_key,
    params
):

    if not api_key:
        return None, "NO_API_KEY"

    request_params = dict(params)

    request_params["apiKey"] = api_key

    try:

        response = requests.get(
            f"{ODDS_BASE}/{endpoint}",
            params=request_params,
            timeout=20,
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
        return None, f"CONNECTION_ERROR: {error}"

    except ValueError:
        return None, "INVALID_JSON"


@st.cache_data(ttl=120)
def get_odds_events(api_key):

    data, status = odds_request(
        "events",
        api_key,
        {
            "sport": "football",
            "status": "pending",
            "limit": 100,
        },
    )

    if status != "OK":
        return [], status

    if not isinstance(data, list):
        return [], "INVALID_EVENTS_STRUCTURE"

    return data, "OK"


def find_odds_event(
    home,
    away,
    match_date,
    odds_events
):

    target_date = pd.to_datetime(
        match_date
    ).date()

    best_event = None
    best_score = 0.0

    for event in odds_events:

        event_home = event.get("home")
        event_away = event.get("away")

        if not event_home or not event_away:
            continue

        score = pair_similarity(
            home,
            away,
            event_home,
            event_away
        )

        if score < 0.65:
            continue

        event_date = pd.to_datetime(
            event.get("date"),
            errors="coerce",
            utc=True
        )

        if pd.isna(event_date):
            continue

        if event_date.date() != target_date:
            continue

        if score > best_score:
            best_event = event
            best_score = score

    return best_event, best_score


def get_ml_odds(
    event_id,
    api_key,
    bookmakers
):

    data, status = odds_request(
        "odds",
        api_key,
        {
            "eventId": event_id,
            "bookmakers": ",".join(bookmakers),
        },
    )

    if status != "OK":
        return [], status

    if not isinstance(data, dict):
        return [], "INVALID_ODDS_STRUCTURE"

    bookmaker_data = (
        data.get("bookmakers")
        or {}
    )

    rows = []

    for bookmaker, markets in bookmaker_data.items():

        if not isinstance(markets, list):
            continue

        for market in markets:

            market_name = str(
                market.get("name", "")
            ).upper()

            if market_name not in [
                "ML",
                "MONEYLINE",
                "MATCH RESULT",
            ]:
                continue

            odds_list = (
                market.get("odds")
                or []
            )

            for item in odds_list:

                home = safe_float(
                    item.get("home")
                )

                draw = safe_float(
                    item.get("draw")
                )

                away = safe_float(
                    item.get("away")
                )

                if (
                    home is None
                    and draw is None
                    and away is None
                ):
                    continue

                rows.append(
                    {
                        "bookmaker": bookmaker,
                        "home": home,
                        "draw": draw,
                        "away": away,
                    }
                )

    if not rows:
        return [], "NO_ML_MARKET"

    return rows, "OK"


# ============================================================
# MOTOR QUANT
# ============================================================

def calculate_metrics(
    probability,
    odds
):

    if (
        probability is None
        or odds is None
        or odds <= 1
    ):
        return None

    implied = 1 / odds

    edge = (
        probability
        - implied
    )

    ev = (
        probability
        * odds
    ) - 1

    value_score = edge * 100

    return {
        "implied": implied,
        "edge": edge,
        "ev": ev,
        "value_score": value_score,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Centro de Mando")

    st.caption(
        "Sports Data Hub · FASE FINAL"
    )

    st.divider()

    selected_date = st.date_input(
        "📅 Día de análisis",
        value=date.today()
    )

    st.divider()

    api_key = st.text_input(
        "🔑 API Key de Odds-API.io",
        type="password",
        placeholder="Pega tu API Key"
    )

    st.divider()

    bookmakers = st.multiselect(
        "💰 Casas de apuestas",
        [
            "Bet365",
            "Unibet",
            "Betano",
            "Betfair",
            "Pinnacle",
        ],
        default=[
            "Bet365",
            "Unibet",
        ]
    )

    st.divider()

    window = st.slider(
        "📚 Partidos recientes",
        min_value=3,
        max_value=10,
        value=DEFAULT_WINDOW
    )

    st.divider()

    if st.button(
        "🔄 Actualizar datos",
        use_container_width=True
    ):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">📊 Centro de Mando Quant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sports Data Hub · Datos reales · Histórico · '
    'Modelo Predictivo · Mercado Real · Value Betting'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EVENTOS
# ============================================================

events, event_status = get_events(
    selected_date.strftime("%Y-%m-%d")
)

if event_status != "OK":

    st.error(
        f"No fue posible obtener los eventos: "
        f"{event_status}"
    )

    st.stop()


event_rows = []

for event in events:

    home = event.get("strHomeTeam")
    away = event.get("strAwayTeam")

    if not home or not away:
        continue

    event_rows.append(
        {
            "id": event.get("idEvent"),
            "league": event.get("strLeague", ""),
            "home": home,
            "away": away,
            "date": event.get("dateEvent"),
            "time": (
                event.get("strTime")
                or "--"
            )[:5],
            "venue": (
                event.get("strVenue")
                or "Sin estadio"
            ),
            "home_id": event.get("idHomeTeam"),
            "away_id": event.get("idAwayTeam"),
        }
    )

df = pd.DataFrame(event_rows)


# ============================================================
# ESTADOS
# ============================================================

s1, s2, s3 = st.columns(3)

with s1:
    st.success("🟢 DATOS DEPORTIVOS")

with s2:
    st.success("🟢 HISTÓRICO")

with s3:
    if api_key:
        st.success("🟢 MERCADO REAL CONECTADO")
    else:
        st.warning("🟡 MERCADO SIN API KEY")


# ============================================================
# ODDS EVENTS
# ============================================================

odds_events = []
odds_status = "NO_API_KEY"

if api_key:

    odds_events, odds_status = get_odds_events(
        api_key
    )

    if odds_status != "OK":

        st.error(
            f"Odds-API.io: {odds_status}"
        )


# ============================================================
# MERCADOS
# ============================================================

market_cache = {}

if (
    api_key
    and odds_status == "OK"
    and not df.empty
):

    for _, row in df.iterrows():

        odds_event, match_score = find_odds_event(
            row["home"],
            row["away"],
            row["date"],
            odds_events
        )

        if odds_event is None:

            market_cache[row["id"]] = {
                "status": "NO_MATCH",
                "score": match_score,
            }

            continue

        event_id = odds_event.get("id")

        if not event_id:

            market_cache[row["id"]] = {
                "status": "INVALID_EVENT_ID"
            }

            continue

        odds_rows, status = get_ml_odds(
            event_id,
            api_key,
            bookmakers
        )

        market_cache[row["id"]] = {
            "status": status,
            "event": odds_event,
            "odds": odds_rows,
            "score": match_score,
        }


# ============================================================
# KPI
# ============================================================

st.markdown(
    '<div class="section">📈 Resumen</div>',
    unsafe_allow_html=True
)

events_with_odds = sum(
    1
    for result in market_cache.values()
    if result.get("status") == "OK"
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Eventos",
    len(df)
)

k2.metric(
    "Deporte",
    "Soccer"
)

k3.metric(
    "Mercado",
    "Real" if api_key else "No conectado"
)

k4.metric(
    "Eventos con cuota",
    events_with_odds
)


# ============================================================
# PARTIDOS
# ============================================================

st.markdown(
    '<div class="section">🏟️ Partidos y eventos</div>',
    unsafe_allow_html=True
)

ranking = []

if df.empty:

    st.info(
        "No hay partidos de fútbol para esta fecha."
    )

else:

    for _, row in df.iterrows():

        st.markdown(
            '<div class="match">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="gray">{row["league"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="team">'
            f'{row["home"]} vs {row["away"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="gray">'
            f'📅 {format_date(row["date"])} · '
            f'⏰ {row["time"]} · '
            f'🏟️ {row["venue"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        market = market_cache.get(
            row["id"]
        )

        # ----------------------------------------------------
        # ESTADO DEL MERCADO
        # ----------------------------------------------------

        if not api_key:

            st.warning(
                "🟡 SIN API KEY"
            )

        elif not market:

            st.warning(
                "🟡 MERCADO NO CONSULTADO"
            )

        elif market["status"] == "NO_MATCH":

            st.warning(
                "🟡 SIN EVENTO COINCIDENTE"
            )

        elif market["status"] == "NO_ML_MARKET":

            st.warning(
                "🟡 EVENTO ENCONTRADO, "
                "PERO SIN MERCADO 1X2"
            )

        elif market["status"] != "OK":

            st.warning(
                f'🟡 {market["status"]}'
            )

        else:

            odds_event = market["event"]

            st.success(
                "🟢 EVENTO Y MERCADO ENCONTRADOS"
            )

            st.caption(
                f'Odds-API.io: '
                f'{odds_event.get("home")} vs '
                f'{odds_event.get("away")}'
            )

            odds_rows = market["odds"]

            if odds_rows:

                table = []

                for odd in odds_rows:

                    table.append(
                        {
                            "Casa": odd["bookmaker"],
                            "Local": odd["home"],
                            "Empate": odd["draw"],
                            "Visitante": odd["away"],
                        }
                    )

                st.dataframe(
                    pd.DataFrame(table),
                    use_container_width=True,
                    hide_index=True
                )

        # ----------------------------------------------------
        # HISTÓRICO + MODELO
        # ----------------------------------------------------

        if not row["home_id"] or not row["away_id"]:

            st.warning(
                "🟡 No existen IDs de equipos compatibles "
                "con el modelo."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            continue

        home_raw, home_status = get_team_history(
            row["home_id"]
        )

        away_raw, away_status = get_team_history(
            row["away_id"]
        )

        if (
            home_status != "OK"
            or away_status != "OK"
        ):

            st.warning(
                "🟡 Histórico insuficiente "
                "para este evento."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            continue

        home_history = history_to_df(
            home_raw
        )

        away_history = history_to_df(
            away_raw
        )

        combined = pd.concat(
            [
                home_history,
                away_history,
            ],
            ignore_index=True
        )

        if combined.empty:

            st.warning(
                "🟡 No hay histórico válido."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            continue

        combined = combined.drop_duplicates(
            subset=["ID"]
        )

        home_form = calculate_form(
            combined,
            row["home"],
            row["date"],
            window
        )

        away_form = calculate_form(
            combined,
            row["away"],
            row["date"],
            window
        )

        if (
            home_form is None
            or away_form is None
        ):

            st.info(
                f"🟡 Histórico disponible, "
                f"pero se requieren {window} "
                f"partidos válidos anteriores "
                f"por equipo."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            continue

        features = make_features(
            home_form,
            away_form
        )

        dataset = build_dataset(
            combined,
            window
        )

        model, validation = train_model(
            dataset
        )

        if model is None:

            st.info(
                "🟡 Modelo bloqueado: "
                "dataset histórico insuficiente "
                "o sin suficientes clases."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            continue

        probabilities = predict_probabilities(
            model,
            features
        )

        if not probabilities:
            continue

        # ----------------------------------------------------
        # MOSTRAR MODELO
        # ----------------------------------------------------

        with st.expander(
            "🧠 Ver análisis cuantitativo"
        ):

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Local",
                f"{probabilities['H']:.1%}"
            )

            m2.metric(
                "Empate",
                f"{probabilities['D']:.1%}"
            )

            m3.metric(
                "Visitante",
                f"{probabilities['A']:.1%}"
            )

            st.caption(
                f"Entrenamiento: "
                f"{validation['train']} · "
                f"Validación: "
                f"{validation['validation']} · "
                f"Accuracy temporal: "
                f"{validation['accuracy']:.1%}"
            )

        # ----------------------------------------------------
        # MOTOR QUANT
        # ----------------------------------------------------

        if (
            market
            and market.get("status") == "OK"
        ):

            for odd in market["odds"]:

                outcomes = [
                    (
                        "H",
                        "Local",
                        odd["home"]
                    ),
                    (
                        "D",
                        "Empate",
                        odd["draw"]
                    ),
                    (
                        "A",
                        "Visitante",
                        odd["away"]
                    ),
                ]

                for code, label, decimal_odds in outcomes:

                    if decimal_odds is None:
                        continue

                    probability = probabilities.get(
                        code
                    )

                    metrics = calculate_metrics(
                        probability,
                        decimal_odds
                    )

                    if metrics is None:
                        continue

                    ranking.append(
                        {
                            "Evento":
                                f'{row["home"]} vs {row["away"]}',

                            "Liga":
                                row["league"],

                            "Resultado":
                                label,

                            "Casa":
                                odd["bookmaker"],

                            "Cuota":
                                decimal_odds,

                            "Probabilidad":
                                probability,

                            "Implícita":
                                metrics["implied"],

                            "Edge":
                                metrics["edge"],

                            "EV":
                                metrics["ev"],

                            "Value Score":
                                metrics["value_score"],

                            "Accuracy":
                                validation["accuracy"],

                            "Entrenamiento":
                                validation["train"],
                        }
                    )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# RANKING
# ============================================================

st.markdown(
    '<div class="section">🏆 Ranking de oportunidades</div>',
    unsafe_allow_html=True
)

if ranking:

    ranking_df = pd.DataFrame(
        ranking
    )

    ranking_df = ranking_df.sort_values(
        "EV",
        ascending=False
    ).reset_index(
        drop=True
    )

    display_df = ranking_df.copy()

    for column in [
        "Probabilidad",
        "Implícita",
        "Edge",
        "EV",
        "Accuracy",
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    display_df["Cuota"] = (
        display_df["Cuota"].round(2)
    )

    display_df["Value Score"] = (
        display_df["Value Score"].round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    best = ranking_df.iloc[0]

    st.success(
        f'🎯 MEJOR OPORTUNIDAD: '
        f'{best["Resultado"]} · '
        f'{best["Evento"]} · '
        f'{best["Casa"]} · '
        f'cuota {best["Cuota"]:.2f} · '
        f'Prob. {best["Probabilidad"]:.1%} · '
        f'Edge {best["Edge"]:.1%} · '
        f'EV {best["EV"]:.1%}'
    )

else:

    st.info(
        "No existen oportunidades cuantitativas "
        "completas para construir el ranking."
    )

    st.caption(
        "Se necesitan simultáneamente: "
        "histórico suficiente + modelo válido + "
        "evento coincidente + mercado ML/1X2 real."
    )


# ============================================================
# INTEGRIDAD
# ============================================================

st.markdown(
    '<div class="section">🔐 Integridad del sistema</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">

    🟢 Datos deportivos reales desde TheSportsDB.

    <br><br>

    🟢 Histórico real de partidos anteriores.

    <br><br>

    🟢 Variables calculadas únicamente con
    información anterior al evento.

    <br><br>

    🟢 Dataset histórico construido con
    resultados reales.

    <br><br>

    🟢 Modelo predictivo con validación temporal.

    <br><br>

    🟢 Mercado obtenido desde Odds-API.io.

    <br><br>

    🟢 Matching protegido por equipos y fecha.

    <br><br>

    🟢 Cuotas ML/1X2 obtenidas del mercado real.

    <br><br>

    🟢 Probabilidad implícita = 1 / cuota.

    <br><br>

    🟢 Edge = probabilidad propia −
    probabilidad implícita.

    <br><br>

    🟢 EV = probabilidad propia × cuota − 1.

    <br><br>

    🟢 Value Score calculado únicamente
    desde datos reales.

    <br><br>

    🔒 No se inventan cuotas.

    <br>

    🔒 No se inventan probabilidades.

    <br>

    🔒 No se fabrica Value Score.

    <br>

    🔒 No se utiliza información futura.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE FINAL"
)

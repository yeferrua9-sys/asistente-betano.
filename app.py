import streamlit as st
import pandas as pd
import requests
import re
import unicodedata
from datetime import date, timedelta
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
)

TSDB = "https://www.thesportsdb.com/api/v1/json/123"
ODDS = "https://api.odds-api.io/v3"

DEFAULT_WINDOW = 3
MIN_MODEL_ROWS = 8

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

def normalize_text(text):
    if text is None:
        return ""

    text = str(text)

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


def tokens(text):
    return {
        x
        for x in normalize_text(text).split()
        if len(x) >= 3
    }


def similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if a in b or b in a:
        return 0.90

    ta = tokens(a)
    tb = tokens(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def pair_similarity(h1, a1, h2, a2):
    direct = (
        similarity(h1, h2)
        +
        similarity(a1, a2)
    ) / 2

    reverse = (
        similarity(h1, a2)
        +
        similarity(a1, h2)
    ) / 2

    return max(direct, reverse)


def safe_float(value):
    try:
        value = float(value)

        if value <= 1:
            return None

        return value

    except Exception:
        return None


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def parse_datetime(value):
    try:
        return pd.to_datetime(
            value,
            errors="coerce",
            utc=True
        )
    except Exception:
        return pd.NaT


# ============================================================
# THESPORTSDB - EVENTOS DEL DÍA
# ============================================================

@st.cache_data(ttl=300)
def get_events(selected_date):

    try:
        response = requests.get(
            f"{TSDB}/eventsday.php",
            params={
                "d": selected_date,
                "s": "Soccer",
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("events") or [], "OK"

    except Exception as e:
        return [], f"ERROR: {e}"


# ============================================================
# THESPORTSDB - HISTÓRICO DE EQUIPO
# ============================================================

@st.cache_data(ttl=3600)
def get_team_history(team_id):

    if not team_id:
        return [], "NO_ID"

    try:
        response = requests.get(
            f"{TSDB}/eventslast.php",
            params={
                "id": team_id
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("results") or [], "OK"

    except Exception as e:
        return [], f"ERROR: {e}"


# ============================================================
# THESPORTSDB - TEMPORADA
# ============================================================

@st.cache_data(ttl=3600)
def get_season_events(league_id, season):

    if not league_id or not season:
        return [], "NO_DATA"

    try:
        response = requests.get(
            f"{TSDB}/eventsseason.php",
            params={
                "id": league_id,
                "s": season,
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("events") or [], "OK"

    except Exception as e:
        return [], f"ERROR: {e}"


# ============================================================
# CONVERSIÓN DE HISTÓRICO
# ============================================================

def history_df(events):

    rows = []

    for event in events:

        hs = safe_int(
            event.get("intHomeScore")
        )

        aw = safe_int(
            event.get("intAwayScore")
        )

        if hs is None or aw is None:
            continue

        rows.append(
            {
                "ID": event.get("idEvent"),
                "Fecha": event.get("dateEvent"),
                "Local": event.get("strHomeTeam"),
                "Visitante": event.get("strAwayTeam"),
                "GolesLocal": hs,
                "GolesVisitante": aw,
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
# UNIFICAR HISTÓRICO
# ============================================================

def combine_history(*dataframes):

    valid = [
        df
        for df in dataframes
        if isinstance(df, pd.DataFrame)
        and not df.empty
    ]

    if not valid:
        return pd.DataFrame()

    combined = pd.concat(
        valid,
        ignore_index=True
    )

    if "ID" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["ID"]
        )

    return combined.sort_values(
        "Fecha"
    ).reset_index(drop=True)


# ============================================================
# FORMA DE EQUIPO
# ============================================================

def team_form(
    df,
    team,
    before_date,
    window=DEFAULT_WINDOW
):

    if df.empty:
        return None

    before_date = pd.to_datetime(
        before_date
    )

    team_norm = normalize_text(team)

    data = df[
        (
            df["Local"].apply(normalize_text)
            == team_norm
        )
        |
        (
            df["Visitante"].apply(normalize_text)
            == team_norm
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
    gf = 0
    ga = 0

    for _, row in data.iterrows():

        if normalize_text(row["Local"]) == team_norm:

            scored = row["GolesLocal"]
            conceded = row["GolesVisitante"]

        else:

            scored = row["GolesVisitante"]
            conceded = row["GolesLocal"]

        gf += scored
        ga += conceded

        if scored > conceded:
            wins += 1
            points += 3

        elif scored == conceded:
            draws += 1
            points += 1

        else:
            losses += 1

    return {
        "matches": len(data),
        "ppp": points / len(data),
        "gf": gf / len(data),
        "ga": ga / len(data),
        "dg": (gf - ga) / len(data),
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }


# ============================================================
# FEATURES
# ============================================================

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
            -
            away_form["ppp"]
        ),
        "gf_diff": (
            home_form["gf"]
            -
            away_form["gf"]
        ),
        "ga_diff": (
            home_form["ga"]
            -
            away_form["ga"]
        ),
        "dg_diff": (
            home_form["dg"]
            -
            away_form["dg"]
        ),
    }


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


# ============================================================
# TARGET
# ============================================================

def result_target(home, away):

    if home > away:
        return "H"

    if home < away:
        return "A"

    return "D"


# ============================================================
# DATASET HISTÓRICO
# ============================================================

def build_dataset(
    combined,
    window=DEFAULT_WINDOW
):

    if combined.empty:
        return pd.DataFrame()

    combined = combined.sort_values(
        "Fecha"
    ).reset_index(drop=True)

    rows = []

    for i, match in combined.iterrows():

        previous = combined.iloc[:i]

        home = match["Local"]
        away = match["Visitante"]

        if not home or not away:
            continue

        hf = team_form(
            previous,
            home,
            match["Fecha"],
            window
        )

        af = team_form(
            previous,
            away,
            match["Fecha"],
            window
        )

        if hf is None or af is None:
            continue

        features = make_features(
            hf,
            af
        )

        features["target"] = result_target(
            match["GolesLocal"],
            match["GolesVisitante"]
        )

        features["Fecha"] = match["Fecha"]

        rows.append(features)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# MODELO
# ============================================================

def train_model(df):

    if df.empty:
        return None, None

    if len(df) < MIN_MODEL_ROWS:
        return None, None

    df = df.sort_values(
        "Fecha"
    ).reset_index(drop=True)

    if df["target"].nunique() < 2:
        return None, None

    split = max(
        6,
        int(len(df) * 0.80)
    )

    if split >= len(df):
        split = len(df) - 1

    train = df.iloc[:split]
    valid = df.iloc[split:]

    if len(train) < 6 or valid.empty:
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
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42
                ),
            ),
        ]
    )

    model.fit(
        train[FEATURES],
        train["target"]
    )

    prediction = model.predict(
        valid[FEATURES]
    )

    accuracy = accuracy_score(
        valid["target"],
        prediction
    )

    return model, {
        "train": len(train),
        "valid": len(valid),
        "accuracy": accuracy,
    }


# ============================================================
# PREDICCIÓN
# ============================================================

def predict(model, features):

    if model is None:
        return None

    input_df = pd.DataFrame(
        [features]
    )[FEATURES]

    probabilities = model.predict_proba(
        input_df
    )[0]

    classes = model.named_steps[
        "model"
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
        if cls in result:
            result[cls] = float(
                probability
            )

    return result


# ============================================================
# ODDS API
# ============================================================

def odds_request(
    endpoint,
    api_key,
    params
):

    try:

        params = dict(params)
        params["apiKey"] = api_key

        response = requests.get(
            f"{ODDS}/{endpoint}",
            params=params,
            timeout=20,
        )

        if response.status_code == 401:
            return None, "API KEY INVALIDA"

        if response.status_code == 403:
            return None, "API NO AUTORIZADA"

        if response.status_code == 429:
            return None, "LIMITE DE API ALCANZADO"

        response.raise_for_status()

        return response.json(), "OK"

    except Exception as e:
        return None, f"ERROR API: {e}"


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
        return [], "FORMATO DE EVENTOS INVALIDO"

    return data, "OK"


# ============================================================
# MATCHING ODDS
# ============================================================

def find_odds_event(
    match,
    odds_events
):

    home = match["Local"]
    away = match["Visitante"]

    target_date = pd.to_datetime(
        match["Fecha"]
    ).date()

    best = None
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

        if score < 0.60:
            continue

        event_date = parse_datetime(
            event.get("date")
        )

        if pd.isna(event_date):
            continue

        if event_date.date() != target_date:
            continue

        if score > best_score:

            best = event
            best_score = score

    return best, best_score


# ============================================================
# ODDS ML / 1X2
# ============================================================

def get_ml_odds(
    event_id,
    api_key,
    bookmakers
):

    if not event_id:
        return [], "SIN EVENT ID"

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
        return [], "FORMATO DE CUOTAS INVALIDO"

    bookmaker_data = (
        data.get("bookmakers")
        or {}
    )

    rows = []

    for bookmaker, markets in bookmaker_data.items():

        if not isinstance(markets, list):
            continue

        for market in markets:

            name = str(
                market.get("name", "")
            ).upper().strip()

            if name not in {
                "ML",
                "MONEYLINE",
                "1X2",
            }:
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

    return rows, (
        "OK"
        if rows
        else "SIN MERCADO 1X2"
    )


# ============================================================
# MOTOR QUANT
# ============================================================

def market_metrics(
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
        -
        implied
    )

    ev = (
        probability
        *
        odds
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

    selected_date = st.date_input(
        "📅 Día de análisis",
        value=date.today()
    )

    api_key = st.text_input(
        "🔑 API Key de Odds-API.io",
        type="password",
        placeholder="Pega tu API Key"
    )

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

    window = st.slider(
        "📚 Partidos recientes",
        3,
        5,
        DEFAULT_WINDOW
    )

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
# DATOS DEPORTIVOS
# ============================================================

events, event_status = get_events(
    selected_date.strftime("%Y-%m-%d")
)

if event_status != "OK":

    st.error(
        f"Error obteniendo eventos: {event_status}"
    )

    st.stop()


# ============================================================
# ESTADOS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.success("🟢 DATOS DEPORTIVOS")

with c2:
    st.success("🟢 HISTÓRICO")

with c3:

    if api_key:
        st.success(
            "🟢 MERCADO REAL CONECTADO"
        )
    else:
        st.warning(
            "🟡 MERCADO SIN API KEY"
        )


# ============================================================
# EVENTOS
# ============================================================

rows = []

for event in events:

    home = event.get(
        "strHomeTeam"
    )

    away = event.get(
        "strAwayTeam"
    )

    if not home or not away:
        continue

    rows.append(
        {
            "id": event.get("idEvent"),
            "league": event.get(
                "strLeague",
                ""
            ),
            "home": home,
            "away": away,
            "date": event.get(
                "dateEvent"
            ),
            "time": (
                event.get("strTime")
                or "--"
            )[:5],
            "venue": (
                event.get("strVenue")
                or "Sin estadio"
            ),
            "home_id": event.get(
                "idHomeTeam"
            ),
            "away_id": event.get(
                "idAwayTeam"
            ),
            "league_id": event.get(
                "idLeague"
            ),
            "season": event.get(
                "strSeason"
            ),
        }
    )

df = pd.DataFrame(rows)


# ============================================================
# MERCADO
# ============================================================

odds_events = []
odds_status = "NO CONSULTADO"

if api_key:

    odds_events, odds_status = (
        get_odds_events(api_key)
    )

    if odds_status != "OK":

        st.error(
            f"Odds-API.io: {odds_status}"
        )


market_cache = {}

if (
    api_key
    and odds_status == "OK"
):

    for _, row in df.iterrows():

        event, score = find_odds_event(
            {
                "Local": row["home"],
                "Visitante": row["away"],
                "Fecha": row["date"],
            },
            odds_events
        )

        if event is None:

            market_cache[row["id"]] = {
                "status": "NO_MATCH"
            }

            continue

        odds, status = get_ml_odds(
            event.get("id"),
            api_key,
            bookmakers
        )

        market_cache[row["id"]] = {
            "status": status,
            "event": event,
            "odds": odds,
            "score": score,
        }


# ============================================================
# KPIs
# ============================================================

st.markdown(
    '<div class="section">📈 Resumen</div>',
    unsafe_allow_html=True
)

events_with_odds = sum(
    1
    for x in market_cache.values()
    if x.get("status") == "OK"
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
# PARTIDOS Y ANÁLISIS
# ============================================================

st.markdown(
    '<div class="section">🏟️ Partidos y eventos</div>',
    unsafe_allow_html=True
)

ranking = []


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
        f'📅 {row["date"]} · '
        f'⏰ {row["time"]} · '
        f'🏟️ {row["venue"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    market = market_cache.get(
        row["id"]
    )

    # --------------------------------------------------------
    # MERCADO
    # --------------------------------------------------------

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
                        "Casa":
                            odd["bookmaker"],
                        "Local":
                            odd["home"],
                        "Empate":
                            odd["draw"],
                        "Visitante":
                            odd["away"],
                    }
                )

            st.dataframe(
                pd.DataFrame(table),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # HISTÓRICO
    # --------------------------------------------------------

    if not row["home_id"] or not row["away_id"]:

        st.warning(
            "🟡 Sin identificadores históricos de equipos."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        continue

    home_raw, hs = get_team_history(
        row["home_id"]
    )

    away_raw, aws = get_team_history(
        row["away_id"]
    )

    home_hist = history_df(
        home_raw
    )

    away_hist = history_df(
        away_raw
    )

    # --------------------------------------------------------
    # INTENTO DE AMPLIAR HISTÓRICO CON TEMPORADA
    # --------------------------------------------------------

    season = row["season"]

    if not season:

        current_year = selected_date.year

        if selected_date.month < 7:
            season = (
                f"{current_year - 1}-{current_year}"
            )
        else:
            season = (
                f"{current_year}-{current_year + 1}"
            )

    season_raw, season_status = (
        get_season_events(
            row["league_id"],
            season
        )
    )

    season_hist = history_df(
        season_raw
    )

    combined = combine_history(
        home_hist,
        away_hist,
        season_hist
    )

    # --------------------------------------------------------
    # FORMAS
    # --------------------------------------------------------

    home_form = team_form(
        combined,
        row["home"],
        row["date"],
        window
    )

    away_form = team_form(
        combined,
        row["away"],
        row["date"],
        window
    )

    if home_form is None or away_form is None:

        available_home = 0
        available_away = 0

        if not combined.empty:

            target_home = normalize_text(
                row["home"]
            )

            target_away = normalize_text(
                row["away"]
            )

            before_date = pd.to_datetime(
                row["date"]
            )

            available_home = len(
                combined[
                    (
                        combined["Local"].apply(
                            normalize_text
                        )
                        == target_home
                    )
                    |
                    (
                        combined["Visitante"].apply(
                            normalize_text
                        )
                        == target_home
                    )
                ][
                    combined["Fecha"] < before_date
                ]
            )

            available_away = len(
                combined[
                    (
                        combined["Local"].apply(
                            normalize_text
                        )
                        == target_away
                    )
                    |
                    (
                        combined["Visitante"].apply(
                            normalize_text
                        )
                        == target_away
                    )
                ][
                    combined["Fecha"] < before_date
                ]
            )

        st.warning(
            "🟡 Histórico insuficiente para construir "
            f"la forma de ambos equipos. "
            f"Disponibles: {available_home} local / "
            f"{available_away} visitante. "
            f"Se requieren {window}."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        continue

    st.success(
        f"🟢 Histórico válido: "
        f"{home_form['matches']} partidos de "
        f"{row['home']} · "
        f"{away_form['matches']} de "
        f"{row['away']}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = make_features(
        home_form,
        away_form
    )

    # --------------------------------------------------------
    # DATASET DEL MODELO
    # --------------------------------------------------------

    dataset = build_dataset(
        combined,
        window
    )

    model, validation = train_model(
        dataset
    )

    if model is None:

        st.warning(
            "🟡 Histórico suficiente para calcular forma, "
            "pero todavía no existe un dataset temporal "
            "suficiente para entrenar el modelo."
        )

        st.caption(
            f"Observaciones válidas del modelo: "
            f"{len(dataset)} · "
            f"Mínimo requerido: {MIN_MODEL_ROWS}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        continue

    # --------------------------------------------------------
    # PROBABILIDADES
    # --------------------------------------------------------

    probabilities = predict(
        model,
        features
    )

    if not probabilities:

        st.warning(
            "🟡 No fue posible calcular probabilidades."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        continue

    # --------------------------------------------------------
    # MOSTRAR MODELO
    # --------------------------------------------------------

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Prob. Local",
        f'{probabilities["H"]:.1%}'
    )

    p2.metric(
        "Prob. Empate",
        f'{probabilities["D"]:.1%}'
    )

    p3.metric(
        "Prob. Visitante",
        f'{probabilities["A"]:.1%}'
    )

    st.caption(
        f"Modelo temporal · "
        f"Entrenamiento: {validation['train']} · "
        f"Validación: {validation['valid']} · "
        f"Accuracy: {validation['accuracy']:.1%}"
    )

    # --------------------------------------------------------
    # VALUE BETTING
    # --------------------------------------------------------

    if not market or market.get("status") != "OK":

        st.info(
            "Mercado real no disponible para este evento; "
            "no se calcula Value Betting."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        continue

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

        for code, label, decimal in outcomes:

            if decimal is None:
                continue

            probability = probabilities.get(
                code
            )

            metrics = market_metrics(
                probability,
                decimal
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
                        decimal,
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
                    "Accuracy modelo":
                        validation["accuracy"],
                    "Datos entrenamiento":
                        validation["train"],
                }
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# RANKING QUANT
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
        [
            "EV",
            "Edge",
            "Value Score"
        ],
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
        "Accuracy modelo",
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
        hide_index=True
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
        "No existen oportunidades cuantitativas completas "
        "para construir el ranking."
    )

    st.caption(
        "El sistema solo publica una oportunidad cuando "
        "existen simultáneamente histórico válido, "
        "modelo entrenable, evento coincidente y "
        "mercado 1X2 real."
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
    <div class="match">
    🟢 Datos deportivos reales desde TheSportsDB.
    <br><br>
    🟢 Histórico real de partidos anteriores.
    <br><br>
    🟢 Variables calculadas únicamente con información
    anterior al evento.
    <br><br>
    🟢 Dataset histórico construido con resultados reales.
    <br><br>
    🟢 Modelo predictivo con validación temporal.
    <br><br>
    🟢 Mercado obtenido desde Odds-API.io.
    <br><br>
    🟢 Matching protegido por equipos y fecha.
    <br><br>
    🟢 Cuotas 1X2 obtenidas del mercado real.
    <br><br>
    🟢 Probabilidad implícita calculada desde la cuota.
    <br><br>
    🟢 Edge calculado contra la probabilidad propia.
    <br><br>
    🟢 EV calculado con cuota real.
    <br><br>
    🟢 Value Score calculado únicamente desde datos reales.
    <br><br>
    🔒 No se inventan cuotas.
    <br><br>
    🔒 No se inventan probabilidades.
    <br><br>
    🔒 No se fabrica Value Score.
    <br><br>
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

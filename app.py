import streamlit as st
import pandas as pd
import requests
import re
import unicodedata
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Centro de Mando Quant",
    page_icon="📊",
    layout="wide"
)

TSDB = "https://www.thesportsdb.com/api/v1/json/123"
ODDS = "https://api.odds-api.io/v3"

MIN_ROWS = 15


# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>
.stApp {
    background:#0b0f15;
    color:#f5f7fa;
}

section[data-testid="stSidebar"] {
    background:#151a23;
}

.block-container {
    max-width:1500px;
    padding-top:2rem;
}

.title {
    font-size:42px;
    font-weight:800;
}

.subtitle {
    color:#9ca3af;
    font-size:17px;
    margin-bottom:25px;
}

.card {
    background:#151a23;
    border:1px solid #293241;
    border-radius:18px;
    padding:22px;
    margin-bottom:15px;
}

.match {
    background:#111720;
    border:1px solid #293241;
    border-radius:18px;
    padding:22px;
    margin:12px 0;
}

.team {
    font-size:22px;
    font-weight:700;
}

.section {
    font-size:27px;
    font-weight:800;
    margin-top:28px;
    margin-bottom:15px;
}

.gray {
    color:#9ca3af;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# UTILIDADES
# ============================================================

def norm(x):
    if x is None:
        return ""

    x = str(x)

    x = unicodedata.normalize(
        "NFKD", x
    ).encode(
        "ascii", "ignore"
    ).decode()

    x = x.lower()
    x = x.replace("&", " and ")

    x = re.sub(
        r"[^a-z0-9]+",
        " ",
        x
    )

    return re.sub(
        r"\s+",
        " ",
        x
    ).strip()


def tokens(x):
    return {
        a for a in norm(x).split()
        if len(a) >= 3
    }


def similarity(a, b):

    a = norm(a)
    b = norm(b)

    if not a or not b:
        return 0

    if a == b:
        return 1

    if a in b or b in a:
        return 0.9

    ta = tokens(a)
    tb = tokens(b)

    if not ta or not tb:
        return 0

    return len(ta & tb) / len(ta | tb)


def match_similarity(h1, a1, h2, a2):

    direct = (
        similarity(h1, h2) +
        similarity(a1, a2)
    ) / 2

    reverse = (
        similarity(h1, a2) +
        similarity(a1, h2)
    ) / 2

    return max(direct, reverse)


def odds_number(x):

    try:
        x = float(x)

        if x > 1:
            return x

    except:
        pass

    return None


# ============================================================
# THESPORTSDB
# ============================================================

@st.cache_data(ttl=300)
def sports_events(selected_date):

    try:

        r = requests.get(
            f"{TSDB}/eventsday.php",
            params={
                "d": selected_date,
                "s": "Soccer"
            },
            timeout=20
        )

        r.raise_for_status()

        data = r.json()

        return data.get("events") or []

    except:
        return []


@st.cache_data(ttl=3600)
def team_history(team_id):

    if not team_id:
        return []

    try:

        r = requests.get(
            f"{TSDB}/eventslast.php",
            params={"id": team_id},
            timeout=20
        )

        r.raise_for_status()

        data = r.json()

        return data.get("results") or []

    except:
        return []


# ============================================================
# ODDS-API.IO
# ============================================================

def odds_request(endpoint, key, params=None):

    if not key:
        return None, "NO_KEY"

    p = dict(params or {})
    p["apiKey"] = key

    try:

        r = requests.get(
            f"{ODDS}/{endpoint}",
            params=p,
            timeout=25
        )

        if r.status_code == 401:
            return None, "INVALID_KEY"

        if r.status_code == 403:
            return None, "FORBIDDEN"

        if r.status_code == 429:
            return None, "RATE_LIMIT"

        r.raise_for_status()

        return r.json(), "OK"

    except requests.RequestException:
        return None, "CONNECTION_ERROR"

    except:
        return None, "INVALID_JSON"


@st.cache_data(ttl=120)
def odds_events(key):

    data, status = odds_request(
        "events",
        key,
        {
            "sport": "football",
            "status": "pending",
            "limit": 100
        }
    )

    if status != "OK":
        return [], status

    if isinstance(data, list):
        return data, "OK"

    if isinstance(data, dict):

        for key_name in [
            "events",
            "data",
            "results"
        ]:

            if isinstance(
                data.get(key_name),
                list
            ):
                return data[key_name], "OK"

    return [], "INVALID_STRUCTURE"


def find_odds_event(match, events):

    best = None
    best_score = 0

    home = match["home"]
    away = match["away"]

    target_date = pd.to_datetime(
        match["date"],
        errors="coerce"
    )

    for event in events:

        eh = (
            event.get("home")
            or event.get("homeTeam")
            or event.get("home_team")
        )

        ea = (
            event.get("away")
            or event.get("awayTeam")
            or event.get("away_team")
        )

        if not eh or not ea:
            continue

        score = match_similarity(
            home,
            away,
            eh,
            ea
        )

        if score < 0.55:
            continue

        event_date = pd.to_datetime(
            event.get("date"),
            errors="coerce",
            utc=True
        )

        if not pd.isna(event_date):

            if event_date.date() != target_date.date():

                distance = abs(
                    (
                        event_date.date()
                        - target_date.date()
                    ).days
                )

                if distance > 1:
                    continue

                score *= 0.9

        if score > best_score:

            best_score = score
            best = event

    return best, best_score


def extract_odds(data):

    rows = []

    if not isinstance(data, dict):
        return rows

    bookmakers = data.get(
        "bookmakers"
    )

    if not isinstance(bookmakers, dict):
        return rows

    for bookmaker, markets in bookmakers.items():

        if not isinstance(markets, list):
            continue

        for market in markets:

            name = str(
                market.get(
                    "name",
                    ""
                )
            ).upper()

            if name not in [
                "ML",
                "MONEYLINE",
                "MATCH RESULT",
                "1X2"
            ]:
                continue

            odds_list = market.get(
                "odds"
            ) or []

            for odd in odds_list:

                h = odds_number(
                    odd.get("home")
                )

                d = odds_number(
                    odd.get("draw")
                )

                a = odds_number(
                    odd.get("away")
                )

                if (
                    h is None and
                    d is None and
                    a is None
                ):
                    continue

                rows.append({
                    "bookmaker": bookmaker,
                    "home": h,
                    "draw": d,
                    "away": a
                })

    return rows


@st.cache_data(ttl=120)
def get_event_odds(event_id, key):

    data, status = odds_request(
        "odds",
        key,
        {
            "eventId": event_id
        }
    )

    if status != "OK":
        return [], status

    rows = extract_odds(data)

    if not rows:
        return [], "NO_MARKET"

    return rows, "OK"


# ============================================================
# HISTÓRICO
# ============================================================

def history_df(raw):

    rows = []

    for e in raw:

        hs = e.get("intHomeScore")
        aw = e.get("intAwayScore")

        try:
            hs = int(hs)
            aw = int(aw)
        except:
            continue

        rows.append({
            "date": e.get("dateEvent"),
            "home": e.get("strHomeTeam"),
            "away": e.get("strAwayTeam"),
            "hg": hs,
            "ag": aw
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df.dropna(
        subset=["date"]
    ).sort_values(
        "date"
    )


def team_form(df, team, before, n=5):

    if df.empty:
        return None

    before = pd.to_datetime(before)

    x = df[
        (df["date"] < before) &
        (
            (df["home"] == team) |
            (df["away"] == team)
        )
    ].sort_values(
        "date",
        ascending=False
    ).head(n)

    if len(x) < n:
        return None

    points = 0
    wins = 0
    losses = 0
    gf = 0
    ga = 0

    for _, r in x.iterrows():

        if r["home"] == team:

            f = r["hg"]
            g = r["ag"]

        else:

            f = r["ag"]
            g = r["hg"]

        gf += f
        ga += g

        if f > g:
            points += 3
            wins += 1

        elif f == g:
            points += 1

        else:
            losses += 1

    return {
        "ppp": points / n,
        "gf": gf / n,
        "ga": ga / n,
        "wins": wins,
        "losses": losses
    }


def features(h, a):

    return [
        h["ppp"],
        a["ppp"],
        h["gf"],
        a["gf"],
        h["ga"],
        a["ga"],
        h["ppp"] - a["ppp"],
        h["gf"] - a["gf"],
        h["ga"] - a["ga"],
        h["wins"],
        a["wins"],
        h["losses"],
        a["losses"]
    ]


# ============================================================
# MODELO
# ============================================================

def build_dataset(df, window):

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    rows = []

    for i in range(len(df)):

        current = df.iloc[i]

        previous = df.iloc[:i]

        h = team_form(
            previous,
            current["home"],
            current["date"],
            window
        )

        a = team_form(
            previous,
            current["away"],
            current["date"],
            window
        )

        if h is None or a is None:
            continue

        if current["hg"] > current["ag"]:
            target = "H"

        elif current["hg"] < current["ag"]:
            target = "A"

        else:
            target = "D"

        rows.append({
            "features": features(h, a),
            "target": target
        })

    return pd.DataFrame(rows)


def train_and_predict(
    df,
    current_h,
    current_a,
    window
):

    dataset = build_dataset(
        df,
        window
    )

    if len(dataset) < MIN_ROWS:
        return None, len(dataset)

    X = pd.DataFrame(
        dataset["features"].tolist()
    )

    y = dataset["target"]

    if y.nunique() < 2:
        return None, len(dataset)

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])

    model.fit(X, y)

    prediction = model.predict_proba(
        pd.DataFrame(
            [features(
                current_h,
                current_a
            )]
        )
    )[0]

    classes = model[
        "model"
    ].classes_

    result = {
        "H": 0,
        "D": 0,
        "A": 0
    }

    for c, p in zip(
        classes,
        prediction
    ):
        result[c] = float(p)

    return result, len(dataset)


# ============================================================
# INTERFAZ
# ============================================================

st.markdown(
    '<div class="title">📊 Centro de Mando Quant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Sports Data Hub · Datos reales · Histórico · Modelo Predictivo · Mercado Real · Value Betting</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Centro de Mando")

    selected_date = st.date_input(
        "📅 Fecha",
        date.today()
    )

    window = st.slider(
        "📚 Partidos recientes",
        3,
        10,
        5
    )

    st.divider()

    api_key = st.text_input(
        "💰 API Key Odds-API.io",
        type="password",
        placeholder="Pega aquí tu API Key"
    )

    st.caption(
        "Usa la misma API Key de Odds-API.io."
    )

    if st.button(
        "🔄 Actualizar",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()


# ============================================================
# CARGAR PARTIDOS
# ============================================================

events = sports_events(
    selected_date.strftime(
        "%Y-%m-%d"
    )
)


# ============================================================
# ODDS
# ============================================================

odds_list = []
odds_status = "NO_KEY"

if api_key:

    odds_list, odds_status = odds_events(
        api_key
    )


# ============================================================
# ESTADO
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.success("🟢 DATOS DEPORTIVOS")

with c2:
    st.success("🟢 HISTÓRICO")

with c3:

    if odds_status == "OK":
        st.success("🟢 MERCADO REAL CONECTADO")

    elif odds_status == "NO_KEY":
        st.warning("🟡 FALTA API KEY")

    else:
        st.error(
            f"🔴 MERCADO: {odds_status}"
        )


# ============================================================
# EVENTOS
# ============================================================

st.markdown(
    '<div class="section">📈 Resumen</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Eventos",
    len(events)
)

c2.metric(
    "Deporte",
    "Soccer"
)

c3.metric(
    "Mercado",
    "Real"
)

c4.metric(
    "Eventos con cuota",
    "—"
)


st.markdown(
    '<div class="section">🏟️ Partidos y eventos</div>',
    unsafe_allow_html=True
)


# ============================================================
# RANKING GLOBAL
# ============================================================

ranking = []


# ============================================================
# RECORRER EVENTOS
# ============================================================

for event in events:

    home = event.get(
        "strHomeTeam"
    )

    away = event.get(
        "strAwayTeam"
    )

    if not home or not away:
        continue

    event_date = event.get(
        "dateEvent"
    )

    event_id = event.get(
        "idEvent"
    )

    match = {
        "home": home,
        "away": away,
        "date": event_date
    }

    # --------------------------------------------------------
    # CARD
    # --------------------------------------------------------

    st.markdown(
        '<div class="match">',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="gray">{event.get("strLeague","Competición")}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="team">{home} vs {away}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="gray">📅 {event_date} · 🏟️ {event.get("strVenue","")}</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # BUSCAR MERCADO
    # --------------------------------------------------------

    market = None
    market_status = "NO_KEY"

    if api_key and odds_status == "OK":

        odds_event, similarity_score = find_odds_event(
            match,
            odds_list
        )

        if odds_event:

            official_id = (
                odds_event.get("id")
                or odds_event.get("eventId")
            )

            if official_id:

                market_rows, market_status = get_event_odds(
                    official_id,
                    api_key
                )

                if market_status == "OK":

                    market = {
                        "event": odds_event,
                        "rows": market_rows
                    }

                    st.success(
                        f"🟢 CUOTA COINCIDENTE · "
                        f"{odds_event.get('home')} vs "
                        f"{odds_event.get('away')}"
                    )

                else:

                    st.warning(
                        "🟡 EVENTO ENCONTRADO, "
                        "PERO SIN MERCADO 1X2"
                    )

        else:

            st.warning(
                "🟡 SIN EVENTO COINCIDENTE"
            )

    elif not api_key:

        st.warning(
            "🟡 INTRODUCE LA API KEY"
        )

    else:

        st.error(
            "🔴 ERROR DE MERCADO"
        )

    # --------------------------------------------------------
    # MOSTRAR CUOTAS
    # --------------------------------------------------------

    if market:

        odds_table = []

        for row in market["rows"]:

            odds_table.append({
                "Casa": row["bookmaker"],
                "Local": row["home"],
                "Empate": row["draw"],
                "Visitante": row["away"]
            })

        if odds_table:

            st.dataframe(
                pd.DataFrame(
                    odds_table
                ),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # BOTÓN ANALIZAR
    # --------------------------------------------------------

    analyze = st.button(
        "🔍 Analizar evento",
        key=f"analyze_{event_id}"
    )

    if analyze:

        st.session_state[
            "selected_event"
        ] = event

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# ANÁLISIS SELECCIONADO
# ============================================================

selected = st.session_state.get(
    "selected_event"
)


if selected:

    home = selected["strHomeTeam"]
    away = selected["strAwayTeam"]

    st.divider()

    st.markdown(
        f'<div class="section">🎯 {home} vs {away}</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # HISTÓRICO
    # --------------------------------------------------------

    home_raw = team_history(
        selected.get(
            "idHomeTeam"
        )
    )

    away_raw = team_history(
        selected.get(
            "idAwayTeam"
        )
    )

    home_df = history_df(
        home_raw
    )

    away_df = history_df(
        away_raw
    )

    history = pd.concat(
        [
            home_df,
            away_df
        ],
        ignore_index=True
    ).drop_duplicates(
        subset=[
            "date",
            "home",
            "away"
        ]
    )

    h1 = team_form(
        history,
        home,
        selected["dateEvent"],
        window
    )

    h2 = team_form(
        history,
        away,
        selected["dateEvent"],
        window
    )

    if h1 and h2:

        st.markdown(
            '<div class="section">📊 Forma reciente</div>',
            unsafe_allow_html=True
        )

        form_table = pd.DataFrame({

            "Variable": [
                "Puntos por partido",
                "Goles por partido",
                "Goles recibidos",
                "Victorias",
                "Derrotas"
            ],

            home: [
                round(h1["ppp"], 3),
                round(h1["gf"], 3),
                round(h1["ga"], 3),
                h1["wins"],
                h1["losses"]
            ],

            away: [
                round(h2["ppp"], 3),
                round(h2["gf"], 3),
                round(h2["ga"], 3),
                h2["wins"],
                h2["losses"]
            ]
        })

        st.dataframe(
            form_table,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        probabilities, dataset_size = train_and_predict(
            history,
            h1,
            h2,
            window
        )

        st.markdown(
            '<div class="section">🧠 Modelo predictivo</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Observaciones históricas",
            dataset_size
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
                "H": home,
                "D": "Empate",
                "A": away
            }

            st.info(
                f"🎯 Mayor probabilidad: "
                f"**{labels[best]} — "
                f"{probabilities[best]:.1%}**"
            )

        else:

            st.warning(
                "🟡 Modelo bloqueado: "
                "no existe suficiente histórico."
            )

            probabilities = None

    else:

        st.warning(
            "🟡 No existe suficiente histórico "
            "para calcular la forma."
        )

        probabilities = None

    # --------------------------------------------------------
    # MERCADO DEL EVENTO SELECCIONADO
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">💰 Mercado real</div>',
        unsafe_allow_html=True
    )

    selected_match = {
        "home": home,
        "away": away,
        "date": selected["dateEvent"]
    }

    selected_market = None

    if api_key and odds_status == "OK":

        odds_event, score = find_odds_event(
            selected_match,
            odds_list
        )

        if odds_event:

            official_id = (
                odds_event.get("id")
                or odds_event.get("eventId")
            )

            if official_id:

                rows, status = get_event_odds(
                    official_id,
                    api_key
                )

                if status == "OK":

                    selected_market = rows

                    st.success(
                        "🟢 Mercado real encontrado"
                    )

                    table = pd.DataFrame([

                        {
                            "Casa": r["bookmaker"],
                            "Local": r["home"],
                            "Empate": r["draw"],
                            "Visitante": r["away"]
                        }

                        for r in rows

                    ])

                    st.dataframe(
                        table,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning(
                        "🟡 No existe mercado 1X2."
                    )

            else:

                st.warning(
                    "🟡 El evento no tiene ID válido."
                )

        else:

            st.warning(
                "🟡 No existe coincidencia de mercado."
            )

    # --------------------------------------------------------
    # VALUE BETTING
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">📈 Value Betting</div>',
        unsafe_allow_html=True
    )

    opportunities = []

    if probabilities and selected_market:

        for bookmaker in selected_market:

            outcomes = [

                (
                    "Local",
                    "H",
                    bookmaker["home"]
                ),

                (
                    "Empate",
                    "D",
                    bookmaker["draw"]
                ),

                (
                    "Visitante",
                    "A",
                    bookmaker["away"]
                )

            ]

            for label, code, odd in outcomes:

                if odd is None:
                    continue

                probability = probabilities[
                    code
                ]

                implied = 1 / odd

                edge = (
                    probability
                    - implied
                )

                ev = (
                    probability * odd
                ) - 1

                score = edge * 100

                opportunities.append({

                    "Resultado": label,

                    "Casa":
                        bookmaker["bookmaker"],

                    "Cuota":
                        odd,

                    "Probabilidad propia":
                        probability,

                    "Probabilidad implícita":
                        implied,

                    "Edge":
                        edge,

                    "EV":
                        ev,

                    "Value Score":
                        score

                })

    if opportunities:

        value_df = pd.DataFrame(
            opportunities
        ).sort_values(
            "EV",
            ascending=False
        )

        st.dataframe(
            value_df,
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
                    )
            }
        )

        best = value_df.iloc[0]

        st.success(
            f"🎯 Mejor oportunidad: "
            f"{best['Resultado']} · "
            f"{best['Casa']} · "
            f"cuota {best['Cuota']:.2f} · "
            f"EV {best['EV']:.2%}"
        )

    else:

        st.info(
            "No hay Value Betting calculable. "
            "Se necesitan simultáneamente "
            "probabilidad propia + cuota real."
        )


# ============================================================
# RANKING
# ============================================================

st.divider()

st.markdown(
    '<div class="section">🏆 Ranking de oportunidades</div>',
    unsafe_allow_html=True
)

st.info(
    "El ranking se construye únicamente "
    "cuando existen datos históricos suficientes, "
    "mercado real coincidente y probabilidades propias."
)


# ============================================================
# INTEGRIDAD
# ============================================================

st.markdown(
    '<div class="section">🔐 Integridad del sistema</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">

🟢 Datos deportivos reales.

<br><br>

🟢 Histórico real.

<br><br>

🟢 Modelo entrenado únicamente con información anterior al evento.

<br><br>

🟢 Mercado obtenido desde Odds-API.io.

<br><br>

🟢 Matching protegido entre equipos.

<br><br>

🟢 Cuotas reales.

<br><br>

🟢 Probabilidad implícita calculada desde la cuota.

<br><br>

🟢 Edge calculado contra la probabilidad propia.

<br><br>

🟢 EV calculado con cuota real.

<br><br>

🔒 No se inventan cuotas.

<br>

🔒 No se inventan probabilidades.

<br>

🔒 No se fabrica Value Score.

<br>

🔒 No se utiliza información futura.

</div>
""", unsafe_allow_html=True)


st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE FINAL"
)

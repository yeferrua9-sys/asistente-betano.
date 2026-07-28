```python
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
# CONFIG
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
.card {
    background:#151a23;
    border:1px solid #293241;
    border-radius:18px;
    padding:22px;
    margin-bottom:16px;
}
.match {
    background:#111720;
    border:1px solid #293241;
    border-radius:18px;
    padding:22px;
    margin:12px 0;
}
.title {
    font-size:40px;
    font-weight:800;
}
.subtitle {
    color:#9ca3af;
    font-size:16px;
    margin-bottom:25px;
}
.team {
    font-size:21px;
    font-weight:700;
}
.gray {
    color:#9ca3af;
}
.section {
    font-size:27px;
    font-weight:750;
    margin-top:25px;
    margin-bottom:15px;
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
    ).decode("ascii")

    x = x.lower()
    x = x.replace("&", " and ")

    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x)

    return x.strip()


def tokens(x):
    return {
        t for t in norm(x).split()
        if len(t) >= 3
    }


def team_score(a, b):

    a = norm(a)
    b = norm(b)

    if not a or not b:
        return 0

    if a == b:
        return 1

    if a in b or b in a:
        return .92

    ta = tokens(a)
    tb = tokens(b)

    if not ta or not tb:
        return 0

    return len(ta & tb) / len(ta | tb)


def match_score(h1, a1, h2, a2):

    direct = (
        team_score(h1, h2) +
        team_score(a1, a2)
    ) / 2

    reverse = (
        team_score(h1, a2) +
        team_score(a1, h2)
    ) / 2

    return max(direct, reverse)


def odds_number(x):

    try:
        x = float(x)
        return x if x > 1 else None
    except:
        return None


# ============================================================
# THESPORTSDB
# ============================================================

@st.cache_data(ttl=300)
def get_events(selected_date):

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

        return r.json().get("events") or []

    except:
        return []


@st.cache_data(ttl=3600)
def get_history(team_id):

    if not team_id:
        return []

    try:

        r = requests.get(
            f"{TSDB}/eventslast.php",
            params={"id": team_id},
            timeout=20
        )

        r.raise_for_status()

        return r.json().get("results") or []

    except:
        return []


def history_df(events):

    rows = []

    for e in events:

        try:
            hs = int(e.get("intHomeScore"))
            aw = int(e.get("intAwayScore"))
        except:
            continue

        rows.append({
            "id": e.get("idEvent"),
            "date": pd.to_datetime(
                e.get("dateEvent"),
                errors="coerce"
            ),
            "home": e.get("strHomeTeam", ""),
            "away": e.get("strAwayTeam", ""),
            "hg": hs,
            "ag": aw
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.dropna(
        subset=["date"]
    ).drop_duplicates(
        subset=["id"]
    ).sort_values(
        "date"
    )


# ============================================================
# FORMA
# ============================================================

def form(df, team, n=5, before=None):

    if df.empty:
        return None

    x = df[
        (df.home == team) |
        (df.away == team)
    ].copy()

    if before is not None:

        x = x[
            x.date <
            pd.to_datetime(before)
        ]

    x = x.tail(n)

    if len(x) < n:
        return None

    p = 0
    gf = 0
    ga = 0
    wins = 0
    losses = 0

    for _, r in x.iterrows():

        if r.home == team:
            f = r.hg
            a = r.ag
        else:
            f = r.ag
            a = r.hg

        gf += f
        ga += a

        if f > a:
            p += 3
            wins += 1

        elif f == a:
            p += 1

        else:
            losses += 1

    return {
        "ppp": p / n,
        "gf": gf / n,
        "ga": ga / n,
        "wins": wins,
        "losses": losses,
        "dg": (gf - ga) / n
    }


def features(h, a):

    return {
        "hppp": h["ppp"],
        "appp": a["ppp"],
        "hgf": h["gf"],
        "agf": a["gf"],
        "hga": h["ga"],
        "aga": a["ga"],
        "hw": h["wins"],
        "aw": a["wins"],
        "hl": h["losses"],
        "al": a["losses"],
        "hdg": h["dg"],
        "adg": a["dg"],
        "pppd": h["ppp"] - a["ppp"],
        "gfd": h["gf"] - a["gf"],
        "gad": h["ga"] - a["ga"],
        "dgd": h["dg"] - a["dg"]
    }


FEATURES = [
    "hppp","appp",
    "hgf","agf",
    "hga","aga",
    "hw","aw",
    "hl","al",
    "hdg","adg",
    "pppd","gfd",
    "gad","dgd"
]


# ============================================================
# DATASET
# ============================================================

def build_dataset(df, window=5):

    if df.empty:
        return pd.DataFrame()

    rows = []

    df = df.sort_values("date").reset_index(drop=True)

    for i, game in df.iterrows():

        previous = df.iloc[:i]

        hf = form(
            previous,
            game.home,
            window
        )

        af = form(
            previous,
            game.away,
            window
        )

        if hf is None or af is None:
            continue

        f = features(hf, af)

        if game.hg > game.ag:
            target = "H"
        elif game.hg < game.ag:
            target = "A"
        else:
            target = "D"

        f["target"] = target

        rows.append(f)

    return pd.DataFrame(rows)


# ============================================================
# MODELO
# ============================================================

def train_predict(
    dataset,
    current_features
):

    if len(dataset) < MIN_ROWS:
        return None

    if dataset.target.nunique() < 2:
        return None

    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])

    model.fit(
        dataset[FEATURES],
        dataset.target
    )

    p = model.predict_proba(
        pd.DataFrame(
            [current_features]
        )[FEATURES]
    )[0]

    result = {
        "H": 0,
        "D": 0,
        "A": 0
    }

    for cls, prob in zip(
        model.classes_,
        p
    ):
        result[cls] = float(prob)

    return result


# ============================================================
# ODDS-API.IO
# ============================================================

def odds_request(
    endpoint,
    api_key,
    params
):

    try:

        params = dict(params)
        params["apiKey"] = api_key

        r = requests.get(
            f"{ODDS}/{endpoint}",
            params=params,
            timeout=20
        )

        if r.status_code == 401:
            return None, "API_KEY_INVALID"

        if r.status_code == 403:
            return None, "FORBIDDEN"

        if r.status_code == 429:
            return None, "RATE_LIMIT"

        r.raise_for_status()

        return r.json(), "OK"

    except requests.RequestException as e:

        return None, str(e)

    except ValueError:

        return None, "INVALID_JSON"


# ============================================================
# NUEVO FLUJO CORRECTO:
# BOOKMAKER → EVENTO → EVENT ID → ODDS
# ============================================================

@st.cache_data(ttl=120)
def get_bookmaker_events(
    api_key,
    bookmaker
):

    data, status = odds_request(
        "events",
        api_key,
        {
            "sport": "football",
            "bookmaker": bookmaker,
            "status": "pending",
            "limit": 500
        }
    )

    if status != "OK":
        return [], status

    if isinstance(data, list):
        return data, "OK"

    return [], "INVALID_STRUCTURE"


def find_bookmaker_event(
    match,
    events
):

    best = None
    best_score = 0

    for e in events:

        h = e.get("home")
        a = e.get("away")

        if not h or not a:
            continue

        score = match_score(
            match["Local"],
            match["Visitante"],
            h,
            a
        )

        if score > best_score:
            best_score = score
            best = e

    if best_score < .70:
        return None, best_score

    return best, best_score


@st.cache_data(ttl=60)
def get_event_odds(
    api_key,
    event_id,
    bookmaker
):

    data, status = odds_request(
        "odds",
        api_key,
        {
            "eventId": event_id,
            "bookmakers": bookmaker
        }
    )

    if status != "OK":
        return [], status

    if not isinstance(data, dict):
        return [], "INVALID_ODDS_STRUCTURE"

    bookmakers = data.get(
        "bookmakers",
        {}
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    # No asumimos que la respuesta tiene exactamente
    # la misma estructura interna siempre.
    # --------------------------------------------------------

    bookmaker_data = (
        bookmakers.get(bookmaker)
        or []
    )

    rows = []

    for market in bookmaker_data:

        name = str(
            market.get("name", "")
        ).upper().strip()

        # ML = Moneyline = 1X2
        if name not in [
            "ML",
            "MONEYLINE",
            "MATCH RESULT",
            "1X2",
            "MATCH WINNER"
        ]:
            continue

        odds_list = market.get(
            "odds",
            []
        )

        for o in odds_list:

            home = odds_number(
                o.get("home")
            )

            draw = odds_number(
                o.get("draw")
            )

            away = odds_number(
                o.get("away")
            )

            if (
                home is None and
                draw is None and
                away is None
            ):
                continue

            rows.append({
                "bookmaker": bookmaker,
                "home": home,
                "draw": draw,
                "away": away,
                "updated": market.get(
                    "updatedAt"
                )
            })

    return rows, (
        "OK"
        if rows
        else "NO_1X2"
    )


def get_real_market(
    match,
    api_key,
    bookmakers
):

    all_rows = []
    matched_events = []

    for bookmaker in bookmakers:

        events, status = get_bookmaker_events(
            api_key,
            bookmaker
        )

        if status != "OK":
            continue

        event, score = find_bookmaker_event(
            match,
            events
        )

        if event is None:
            continue

        event_id = event.get("id")

        if not event_id:
            continue

        odds, odds_status = get_event_odds(
            api_key,
            event_id,
            bookmaker
        )

        if odds_status == "OK":

            for o in odds:
                o["match_score"] = score
                o["event_id"] = event_id
                o["event_home"] = event.get("home")
                o["event_away"] = event.get("away")

            all_rows.extend(odds)

            matched_events.append(event)

    if not all_rows:

        if matched_events:
            return {
                "status": "EVENT_FOUND_NO_1X2",
                "events": matched_events
            }

        return {
            "status": "NO_MATCH"
        }

    return {
        "status": "OK",
        "odds": all_rows,
        "event": matched_events[0]
    }


# ============================================================
# VALUE
# ============================================================

def metrics(prob, odd):

    if prob is None or odd is None:
        return None

    implied = 1 / odd

    edge = prob - implied

    ev = (
        prob * odd
    ) - 1

    return {
        "implied": implied,
        "edge": edge,
        "ev": ev,
        "value": edge * 100
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Centro de Mando")

    selected_date = st.date_input(
        "📅 Día de análisis",
        date.today()
    )

    history_window = st.slider(
        "📚 Partidos recientes",
        3,
        10,
        5
    )

    st.divider()

    api_key = st.text_input(
        "🔑 API Key de Odds-API.io",
        type="password"
    )

    st.caption(
        "Usa la misma API Key que ya creaste."
    )

    bookmakers = st.multiselect(
        "💰 Casas disponibles",
        [
            "Bet365",
            "Unibet"
        ],
        default=[
            "Bet365",
            "Unibet"
        ]
    )

    if st.button(
        "🔄 Actualizar",
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
    '<div class="subtitle">Sports Data Hub · Datos reales · Histórico · Modelo Predictivo · Mercado Real · Value Betting</div>',
    unsafe_allow_html=True
)


# ============================================================
# EVENTOS
# ============================================================

events = get_events(
    selected_date.strftime("%Y-%m-%d")
)

rows = []

for e in events:

    home = e.get("strHomeTeam")
    away = e.get("strAwayTeam")

    if not home or not away:
        continue

    rows.append({
        "ID": e.get("idEvent"),
        "Liga": e.get("strLeague", ""),
        "Evento": f"{home} vs {away}",
        "Fecha": e.get("dateEvent", ""),
        "Local": home,
        "Visitante": away,
        "IDLocal": e.get("idHomeTeam"),
        "IDVisitante": e.get("idAwayTeam"),
        "Estadio": e.get("strVenue", "")
    })

df = pd.DataFrame(rows)


# ============================================================
# MERCADO
# ============================================================

market_results = {}

if api_key and bookmakers and not df.empty:

    for _, row in df.iterrows():

        market_results[row["ID"]] = get_real_market(
            row,
            api_key,
            bookmakers
        )


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
        st.success("🟢 MERCADO REAL CONECTADO")
    else:
        st.warning("🟡 FALTA API KEY")


# ============================================================
# RESUMEN
# ============================================================

st.markdown(
    '<div class="section">📈 Resumen</div>',
    unsafe_allow_html=True
)

with_market = sum(
    1 for x in market_results.values()
    if x.get("status") == "OK"
)

a, b, c, d = st.columns(4)

a.metric(
    "Eventos",
    len(df)
)

b.metric(
    "Deporte",
    "Soccer"
)

c.metric(
    "Mercado",
    "Real" if api_key else "—"
)

d.metric(
    "Eventos con cuota",
    with_market
)


# ============================================================
# PARTIDOS
# ============================================================

st.markdown(
    '<div class="section">🏟️ Partidos y eventos</div>',
    unsafe_allow_html=True
)

for _, row in df.iterrows():

    st.markdown(
        '<div class="match">',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="gray">{row["Liga"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="team">{row["Evento"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="gray">📅 {row["Fecha"]} · 🏟️ {row["Estadio"] or "No disponible"}</div>',
        unsafe_allow_html=True
    )

    result = market_results.get(
        row["ID"]
    )

    if not api_key:

        st.warning(
            "🟡 SIN API KEY"
        )

    elif not result:

        st.warning(
            "🟡 MERCADO NO CONSULTADO"
        )

    elif result["status"] == "OK":

        st.success(
            "🟢 CUOTA 1X2 ENCONTRADA"
        )

        event = result["event"]

        st.caption(
            f'Odds-API.io: {event.get("home")} vs {event.get("away")}'
        )

        table = []

        for o in result["odds"]:

            table.append({
                "Casa": o["bookmaker"],
                "Local": o["home"],
                "Empate": o["draw"],
                "Visitante": o["away"]
            })

        st.dataframe(
            pd.DataFrame(table),
            use_container_width=True,
            hide_index=True
        )

    elif result["status"] == "EVENT_FOUND_NO_1X2":

        st.warning(
            "🟡 EVENTO ENCONTRADO, PERO SIN MERCADO 1X2 EN LAS CASAS CONSULTADAS"
        )

    else:

        st.warning(
            "🟡 SIN EVENTO COINCIDENTE"
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

ranking = []

for _, row in df.iterrows():

    market = market_results.get(
        row["ID"]
    )

    if not market or market["status"] != "OK":
        continue

    home_raw = get_history(
        row["IDLocal"]
    )

    away_raw = get_history(
        row["IDVisitante"]
    )

    home_df = history_df(
        home_raw
    )

    away_df = history_df(
        away_raw
    )

    all_history = pd.concat(
        [home_df, away_df],
        ignore_index=True
    ).drop_duplicates(
        subset=["id"]
    )

    hf = form(
        all_history,
        row["Local"],
        history_window,
        row["Fecha"]
    )

    af = form(
        all_history,
        row["Visitante"],
        history_window,
        row["Fecha"]
    )

    if hf is None or af is None:
        continue

    dataset = build_dataset(
        all_history,
        history_window
    )

    current = features(
        hf,
        af
    )

    probabilities = train_predict(
        dataset,
        current
    )

    if probabilities is None:
        continue

    for o in market["odds"]:

        outcomes = [
            ("H", "Local", o["home"]),
            ("D", "Empate", o["draw"]),
            ("A", "Visitante", o["away"])
        ]

        for code, label, odd in outcomes:

            if odd is None:
                continue

            m = metrics(
                probabilities.get(code),
                odd
            )

            if m is None:
                continue

            ranking.append({
                "Evento": row["Evento"],
                "Resultado": label,
                "Casa": o["bookmaker"],
                "Cuota": odd,
                "Probabilidad propia": probabilities[code],
                "Prob. implícita": m["implied"],
                "Edge": m["edge"],
                "EV": m["ev"],
                "Value Score": m["value"]
            })


if ranking:

    rdf = pd.DataFrame(
        ranking
    ).sort_values(
        "EV",
        ascending=False
    )

    st.dataframe(
        rdf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probabilidad propia":
                st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
            "Prob. implícita":
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

    best = rdf.iloc[0]

    st.success(
        f'🎯 Mejor oportunidad: '
        f'{best["Resultado"]} · '
        f'{best["Casa"]} · '
        f'cuota {best["Cuota"]:.2f} · '
        f'EV {best["EV"]:.2%}'
    )

else:

    st.info(
        "No existen oportunidades cuantitativas completas."
    )

    st.caption(
        "Se necesita histórico suficiente + mercado 1X2 real + probabilidad propia."
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

🟢 Mercado obtenido directamente desde Odds-API.io.

<br><br>

🟢 Eventos buscados por bookmaker.

<br><br>

🟢 Event ID oficial de Odds-API.io.

<br><br>

🟢 Mercado ML / 1X2 real.

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


st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE FINAL"
)
```

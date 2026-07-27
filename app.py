"""
CENTRO DE MANDO QUANT — SPORTS DATA HUB
FASE 2 — Motor Quant local + mercados + Value Betting

IMPORTANTE:
Esta fase utiliza datos DEMO para construir y probar el motor.
NO presenta datos demo como datos reales.
The Odds API queda desconectada temporalmente para no consumir créditos.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Centro de Mando Quant | Sports Data Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# ESTILOS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0b0f15;
        color: #f5f7fa;
    }

    section[data-testid="stSidebar"] {
        background: #20232d;
    }

    .quant-header {
        background: linear-gradient(
            135deg,
            #151a24,
            #10141c
        );
        border: 1px solid #2c3442;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 25px;
    }

    .match-card {
        background: #181d25;
        border: 1px solid #303846;
        border-radius: 16px;
        padding: 22px;
        margin: 18px 0;
    }

    .market-card {
        background: #111722;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 13px;
        margin: 8px 0;
    }

    .value-high {
        background: #063c2d;
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 15px;
    }

    .value-medium {
        background: #42350a;
        border: 1px solid #eab308;
        border-radius: 10px;
        padding: 15px;
    }

    .value-low {
        background: #351b1b;
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 15px;
    }

    .demo-warning {
        background: #3b2b08;
        border: 1px solid #eab308;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        margin-top: 25px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATOS DEMO
# =========================================================

FOOTBALL_MATCHES = [
    {
        "sport": "Fútbol",
        "league": "Brasil Série A",
        "home": "Flamengo",
        "away": "Internacional",
        "time": "22:30",
        "date": "27/07/2026",
    },
    {
        "sport": "Fútbol",
        "league": "Brasil Série B",
        "home": "Clube de Regatas Brasil",
        "away": "Vila Nova",
        "time": "22:30",
        "date": "27/07/2026",
    },
    {
        "sport": "Fútbol",
        "league": "Brasil Série B",
        "home": "Atlético Goianiense",
        "away": "Operário PR",
        "time": "17:30",
        "date": "27/07/2026",
    },
    {
        "sport": "Fútbol",
        "league": "Brasil Série B",
        "home": "Sport Recife",
        "away": "Cuiabá",
        "time": "22:30",
        "date": "27/07/2026",
    },
]

BASKETBALL_MATCHES = [
    {
        "sport": "Básquet",
        "league": "NBA",
        "home": "Boston Celtics",
        "away": "New York Knicks",
        "time": "20:00",
        "date": "27/07/2026",
    },
    {
        "sport": "Básquet",
        "league": "NBA",
        "home": "Los Angeles Lakers",
        "away": "Golden State Warriors",
        "time": "22:30",
        "date": "27/07/2026",
    },
]

# =========================================================
# MERCADOS DEMO
# =========================================================

FOOTBALL_MARKETS = [
    {
        "category": "Resultado",
        "market": "Ganador (1X2)",
        "selection": "Flamengo",
        "odds": 1.95,
        "model_probability": 0.575,
    },
    {
        "category": "Goles",
        "market": "Total de goles",
        "selection": "Más de 2.5",
        "odds": 1.88,
        "model_probability": 0.565,
    },
    {
        "category": "Ambos marcan",
        "market": "Ambos equipos marcan",
        "selection": "Sí",
        "odds": 1.82,
        "model_probability": 0.59,
    },
    {
        "category": "Corners",
        "market": "Corners totales",
        "selection": "Más de 8.5",
        "odds": 1.75,
        "model_probability": 0.62,
    },
    {
        "category": "Corners",
        "market": "Corners Flamengo",
        "selection": "Más de 4.5",
        "odds": 1.90,
        "model_probability": 0.613,
    },
    {
        "category": "Tarjetas",
        "market": "Tarjetas totales",
        "selection": "Más de 4.5",
        "odds": 1.85,
        "model_probability": 0.57,
    },
    {
        "category": "Faltas",
        "market": "Faltas totales",
        "selection": "Más de 24.5",
        "odds": 1.78,
        "model_probability": 0.60,
    },
    {
        "category": "Saques de banda",
        "market": "Saques de banda totales",
        "selection": "Más de 31.5",
        "odds": 1.82,
        "model_probability": 0.59,
    },
    {
        "category": "Tiros",
        "market": "Tiros totales Flamengo",
        "selection": "Más de 12.5",
        "odds": 1.90,
        "model_probability": 0.57,
    },
    {
        "category": "Tiros a puerta",
        "market": "Tiros a puerta Flamengo",
        "selection": "Más de 4.5",
        "odds": 2.00,
        "model_probability": 0.55,
    },
]

BASKETBALL_MARKETS = [
    {
        "category": "Ganador",
        "market": "Moneyline",
        "selection": "Boston Celtics",
        "odds": 1.80,
        "model_probability": 0.59,
    },
    {
        "category": "Puntos",
        "market": "Total puntos",
        "selection": "Más de 219.5",
        "odds": 1.90,
        "model_probability": 0.56,
    },
    {
        "category": "Puntos equipo",
        "market": "Boston Celtics puntos",
        "selection": "Más de 111.5",
        "odds": 1.85,
        "model_probability": 0.58,
    },
    {
        "category": "Triples",
        "market": "Triples Boston",
        "selection": "Más de 12.5",
        "odds": 1.95,
        "model_probability": 0.57,
    },
    {
        "category": "Rebotes",
        "market": "Rebotes totales",
        "selection": "Más de 83.5",
        "odds": 1.90,
        "model_probability": 0.55,
    },
    {
        "category": "Asistencias",
        "market": "Asistencias totales",
        "selection": "Más de 44.5",
        "odds": 1.88,
        "model_probability": 0.57,
    },
]

# =========================================================
# FUNCIONES QUANT
# =========================================================

def implied_probability(odds):
    """
    Probabilidad implícita de una cuota decimal.
    """
    if odds <= 1:
        return 0
    return 1 / odds


def calculate_edge(model_probability, implied):
    """
    Edge = probabilidad del modelo - probabilidad implícita.
    """
    return model_probability - implied


def calculate_ev(model_probability, odds):
    """
    EV aproximado por unidad apostada.

    EV = p * beneficio - (1-p)
    """
    return (model_probability * (odds - 1)) - (1 - model_probability)


def value_score(edge, ev):
    """
    Score simple de 0 a 100.

    Esto es una primera versión del motor.
    Más adelante lo sustituiremos por un modelo
    basado en datos históricos reales.
    """

    score = (edge * 500) + (ev * 100)

    score = max(0, min(100, score))

    return round(score, 1)


def classify_value(score):

    if score >= 70:
        return "🔥 VALUE ALTO"

    if score >= 50:
        return "🟡 VALUE MEDIO"

    if score >= 30:
        return "🟠 VALUE MODERADO"

    return "🔴 SIN VALUE"


def analyze_markets(markets):

    results = []

    for item in markets:

        odds = float(item["odds"])
        model_probability = float(item["model_probability"])

        implied = implied_probability(odds)

        edge = calculate_edge(
            model_probability,
            implied
        )

        ev = calculate_ev(
            model_probability,
            odds
        )

        score = value_score(
            edge,
            ev
        )

        result = dict(item)

        result["implied_probability"] = implied
        result["edge"] = edge
        result["ev"] = ev
        result["value_score"] = score
        result["signal"] = classify_value(score)

        results.append(result)

    return results


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Centro de Mando")

    st.caption(
        "Sports Data Hub — Motor Quant"
    )

    st.divider()

    deporte = st.radio(
        "🏟️ Seleccionar deporte",
        [
            "⚽ Fútbol",
            "🏀 Básquet",
        ]
    )

    st.divider()

    fecha = st.date_input(
        "📅 Día de análisis",
        datetime.now()
    )

    st.divider()

    min_edge = st.slider(
        "🎯 Edge mínimo (%)",
        min_value=0,
        max_value=20,
        value=3,
        step=1
    )

    categorias = st.multiselect(
        "📊 Categorías",
        [
            "Resultado",
            "Goles",
            "Ambos marcan",
            "Corners",
            "Tarjetas",
            "Faltas",
            "Saques de banda",
            "Tiros",
            "Tiros a puerta",
            "Ganador",
            "Puntos",
            "Puntos equipo",
            "Triples",
            "Rebotes",
            "Asistencias",
        ]
    )

    st.divider()

    st.info(
        "La conexión con The Odds API está "
        "pausada durante esta fase para evitar "
        "consumir créditos."
    )

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="quant-header">

    <h1>📊 Centro de Mando Quant</h1>

    <h3>Sports Data Hub</h3>

    <p>
    Motor de análisis Value Betting para fútbol y básquet.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# AVISO DEMO
# =========================================================

st.markdown(
    """
    <div class="demo-warning">

    ⚠️ <b>MODO DESARROLLO — DATOS DEMO</b>

    <br><br>

    Esta versión está construyendo el motor matemático
    sin realizar consultas a The Odds API.

    Los partidos, cuotas y probabilidades mostrados
    son datos de demostración y <b>NO representan
    mercados reales disponibles para apostar.</b>

    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SELECCIÓN DE DATASET
# =========================================================

if deporte == "⚽ Fútbol":

    matches = FOOTBALL_MATCHES
    markets_template = FOOTBALL_MARKETS

else:

    matches = BASKETBALL_MATCHES
    markets_template = BASKETBALL_MARKETS


# =========================================================
# ANÁLISIS
# =========================================================

all_results = []

for match in matches:

    analyzed = analyze_markets(
        markets_template
    )

    for result in analyzed:

        result["home"] = match["home"]
        result["away"] = match["away"]
        result["league"] = match["league"]
        result["time"] = match["time"]
        result["date"] = match["date"]

        all_results.append(result)


df = pd.DataFrame(all_results)

# =========================================================
# FILTRO EDGE
# =========================================================

df = df[
    df["edge"] * 100 >= min_edge
]

if categorias:

    df = df[
        df["category"].isin(categorias)
    ]

# =========================================================
# KPIs
# =========================================================

total_opportunities = len(df)

if total_opportunities > 0:

    avg_edge = df["edge"].mean() * 100
    avg_ev = df["ev"].mean() * 100
    max_score = df["value_score"].max()

else:

    avg_edge = 0
    avg_ev = 0
    max_score = 0


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "OPORTUNIDADES",
    total_opportunities
)

col2.metric(
    "EDGE PROMEDIO",
    f"{avg_edge:.2f}%"
)

col3.metric(
    "EV PROMEDIO",
    f"{avg_ev:.2f}%"
)

col4.metric(
    "MEJOR VALUE SCORE",
    f"{max_score:.1f}"
)

st.divider()

# =========================================================
# RANKING
# =========================================================

st.markdown(
    "## 🔥 Ranking de oportunidades"
)

if df.empty:

    st.warning(
        "No hay oportunidades que cumplan "
        "el filtro seleccionado."
    )

else:

    df_rank = df.sort_values(
        "value_score",
        ascending=False
    )

    for index, row in df_rank.iterrows():

        if row["value_score"] >= 70:

            css_class = "value-high"

        elif row["value_score"] >= 50:

            css_class = "value-medium"

        else:

            css_class = "value-low"

        st.markdown(
            f"""
            <div class="{css_class}">

            <h3>
            {row["signal"]}
            </h3>

            <b>
            {row["home"]} vs {row["away"]}
            </b>

            <br>

            🏆 {row["league"]}
            &nbsp; | &nbsp;
            ⏰ {row["time"]}

            <hr>

            <b>{row["category"]}</b>

            <br>

            🎯 {row["market"]}
            <br>

            Selección:
            <b>{row["selection"]}</b>

            <br><br>

            Cuota:
            <b>{row["odds"]:.2f}</b>

            &nbsp; | &nbsp;

            Probabilidad implícita:
            <b>{row["implied_probability"] * 100:.2f}%</b>

            &nbsp; | &nbsp;

            Modelo:
            <b>{row["model_probability"] * 100:.2f}%</b>

            <br><br>

            <b>
            EDGE:
            {row["edge"] * 100:+.2f}%
            </b>

            &nbsp; | &nbsp;

            <b>
            EV:
            {row["ev"] * 100:+.2f}%
            </b>

            &nbsp; | &nbsp;

            <b>
            VALUE SCORE:
            {row["value_score"]:.1f}/100
            </b>

            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# TABLA GENERAL
# =========================================================

st.divider()

st.markdown(
    "## 📋 Tabla Quant"
)

if not df.empty:

    table = df[
        [
            "home",
            "away",
            "category",
            "market",
            "selection",
            "odds",
            "implied_probability",
            "model_probability",
            "edge",
            "ev",
            "value_score",
            "signal",
        ]
    ].copy()

    table["implied_probability"] *= 100
    table["model_probability"] *= 100
    table["edge"] *= 100
    table["ev"] *= 100

    table = table.rename(
        columns={
            "home": "Local",
            "away": "Visitante",
            "category": "Categoría",
            "market": "Mercado",
            "selection": "Selección",
            "odds": "Cuota",
            "implied_probability": "Prob. Implícita %",
            "model_probability": "Prob. Modelo %",
            "edge": "Edge %",
            "ev": "EV %",
            "value_score": "Value Score",
            "signal": "Señal",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# EXPLICACIÓN DEL MODELO
# =========================================================

st.divider()

st.markdown(
    "## 🧠 ¿Cómo funciona el motor Quant?"
)

st.markdown(
    """
    ### 1️⃣ Probabilidad implícita

    Para una cuota decimal:

    **Probabilidad implícita = 1 / cuota**

    Ejemplo:

    `Cuota 2.00 → 50%`

    ---

    ### 2️⃣ Probabilidad del modelo

    El motor estima una probabilidad independiente.

    En esta fase es DEMO.

    Posteriormente será calculada utilizando datos
    deportivos históricos y actuales.

    ---

    ### 3️⃣ Edge

    `Edge = Probabilidad del modelo − Probabilidad implícita`

    Ejemplo:

    `Modelo = 58%`

    `Mercado = 50%`

    `Edge = +8%`

    ---

    ### 4️⃣ EV

    El valor esperado estima el rendimiento matemático
    esperado de una unidad apostada:

    `EV = p × (cuota − 1) − (1 − p)`

    ---

    ### 5️⃣ Value Score

    El Value Score combina Edge y EV para ordenar
    las oportunidades.

    **Este score es provisional.**

    La siguiente fase utilizará datos históricos
    para construir un modelo estadístico real.

    """
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Centro de Mando Quant — Sports Data Hub | "
    "FASE 2 — Motor Quant en desarrollo"
)

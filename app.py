import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date

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

# ============================================================
# ESTILO VISUAL
# ============================================================

st.markdown("""
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

.green {
    color: #10b981;
}

.yellow {
    color: #f59e0b;
}

.red {
    color: #ef4444;
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

.status-gray {
    background: #202630;
    color: #9ca3af;
    border: 1px solid #374151;
}

div[data-testid="stMetric"] {
    background: #151a23;
    border: 1px solid #293241;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# THE SPORTS DB — EVENTOS
# ============================================================

@st.cache_data(ttl=300)
def get_events_day(selected_date, sport_filter):

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

        if response.status_code != 200:
            return [], f"HTTP_{response.status_code}"

        data = response.json()

        events = data.get("events") or []

        if not events:
            return [], "NO_EVENTS"

        return events, "OK"

    except requests.RequestException as error:

        return [], f"CONNECTION_ERROR: {error}"

    except ValueError:

        return [], "JSON_ERROR"


# ============================================================
# FORMATO DE HORA
# ============================================================

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


# ============================================================
# EVENTOS → DATAFRAME
# ============================================================

def events_to_dataframe(events):

    rows = []

    for event in events:

        home = event.get("strHomeTeam")
        away = event.get("strAwayTeam")

        if home and away:
            matchup = f"{home} vs {away}"
        else:
            matchup = event.get(
                "strEvent",
                "Evento deportivo"
            )

        rows.append({

            "ID": event.get("idEvent"),

            "Deporte": event.get(
                "strSport",
                "No disponible"
            ),

            "Liga": event.get(
                "strLeague",
                "No disponible"
            ),

            "Evento": matchup,

            "Fecha": event.get(
                "dateEvent",
                ""
            ),

            "Hora": format_time(event),

            "Local": home or "",

            "Visitante": away or "",

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

            "Estado": event.get(
                "strStatus",
                ""
            ),

            "ResultadoLocal": event.get(
                "intHomeScore"
            ),

            "ResultadoVisitante": event.get(
                "intAwayScore"
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# UTILIDADES
# ============================================================

def safe_value(value, fallback="No disponible"):

    if value is None:
        return fallback

    if pd.isna(value):
        return fallback

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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Centro de Mando")

    st.caption("Sports Data Hub — FASE 6")

    st.divider()

    st.markdown("### 📅 Fecha")

    selected_date = st.date_input(
        "Día de análisis",
        value=date.today()
    )

    st.divider()

    st.markdown("### 🏟️ Deporte")

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

    st.markdown("### 🔎 Filtros")

    st.caption(
        "Los filtros de competición y país "
        "se aplican después de cargar los eventos."
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
    '<div class="main-title">📊 Centro de Mando Quant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sports Data Hub · Datos deportivos reales · Motor Quant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# OBTENER EVENTOS
# ============================================================

events, event_status = get_events_day(
    selected_date.strftime("%Y-%m-%d"),
    sport_filter
)

df_events = events_to_dataframe(events)


# ============================================================
# ESTADOS DEL SISTEMA
# ============================================================

status1, status2, status3 = st.columns(3)

with status1:

    st.markdown(
        '<div class="status status-green">'
        '🟢 DATOS DEPORTIVOS'
        '</div>',
        unsafe_allow_html=True
    )

with status2:

    st.markdown(
        '<div class="status status-yellow">'
        '🟡 CUOTAS PENDIENTES'
        '</div>',
        unsafe_allow_html=True
    )

with status3:

    st.markdown(
        '<div class="status status-yellow">'
        '🟡 MODELO PREDICTIVO PENDIENTE'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# MENSAJES DE CONEXIÓN
# ============================================================

if event_status == "CONNECTION_ERROR":

    st.error(
        "No fue posible conectar con TheSportsDB."
    )

elif event_status == "NO_EVENTS":

    st.info(
        f"No se encontraron eventos para "
        f"{selected_date.strftime('%d/%m/%Y')} "
        f"con los filtros seleccionados."
    )

elif event_status != "OK":

    st.warning(
        f"Estado de la fuente deportiva: {event_status}"
    )


# ============================================================
# KPIs
# ============================================================

st.markdown(
    '<div class="section-title">📈 Resumen</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Eventos",
    len(df_events)
)

c2.metric(
    "Deportes",
    df_events["Deporte"].nunique()
    if not df_events.empty else 0
)

c3.metric(
    "Ligas",
    df_events["Liga"].nunique()
    if not df_events.empty else 0
)

c4.metric(
    "Países",
    df_events["País"].nunique()
    if not df_events.empty else 0
)


# ============================================================
# FILTROS
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Filtrar partidos</div>',
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
            ["Todas"] + league_options
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
            ["Todos"] + country_options
        )

    filtered = df_events.copy()

    if selected_league != "Todas":

        filtered = filtered[
            filtered["Liga"] == selected_league
        ]

    if selected_country != "Todos":

        filtered = filtered[
            filtered["País"] == selected_country
        ]

else:

    filtered = df_events


# ============================================================
# PARTIDOS
# ============================================================

st.markdown(
    '<div class="section-title">🏟️ Partidos y eventos</div>',
    unsafe_allow_html=True
)

if filtered.empty:

    st.info(
        "No hay eventos deportivos para los "
        "filtros seleccionados."
    )

else:

    for _, row in filtered.iterrows():

        event_id = row["ID"]

        st.markdown(
            '<div class="match-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="gray">'
            f'{safe_value(row["Liga"], "Competición no disponible")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="team">'
            f'{safe_value(row["Evento"], "Evento deportivo")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="gray">'
            f'📅 {format_date_display(row["Fecha"])}'
            f' &nbsp;&nbsp; '
            f'⏰ {safe_value(row["Hora"], "--")}'
            f'</div>',
            unsafe_allow_html=True
        )

        location_parts = []

        if safe_value(row["Estadio"], "") != "":
            location_parts.append(
                safe_value(row["Estadio"], "")
            )

        if safe_value(row["Ciudad"], "") != "":
            location_parts.append(
                safe_value(row["Ciudad"], "")
            )

        if safe_value(row["País"], "") != "":
            location_parts.append(
                safe_value(row["País"], "")
            )

        location = " · ".join(location_parts)

        st.markdown(
            f'<div class="gray">'
            f'📍 {location if location else "Ubicación no disponible"}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "🔍 Analizar evento",
            key=f"analyze_{event_id}",
            use_container_width=False
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
        df_events["ID"] == selected_event_id
    ]

    if not selected_rows.empty:

        selected = selected_rows.iloc[0]

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🎯 Análisis del evento'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="card">'
            f'<div class="team">'
            f'{safe_value(selected["Evento"])}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # DATOS DEL EVENTO
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📋 Datos del evento'
            '</div>',
            unsafe_allow_html=True
        )

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Deporte",
            safe_value(selected["Deporte"])
        )

        a2.metric(
            "Competición",
            safe_value(selected["Liga"])
        )

        a3.metric(
            "Fecha",
            format_date_display(selected["Fecha"])
        )

        a4.metric(
            "Hora",
            safe_value(selected["Hora"], "--")
        )

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.write(
            f"**🏠 Local:** "
            f"{safe_value(selected['Local'])}"
        )

        st.write(
            f"**✈️ Visitante:** "
            f"{safe_value(selected['Visitante'])}"
        )

        st.write(
            f"**🏟️ Estadio:** "
            f"{safe_value(selected['Estadio'])}"
        )

        st.write(
            f"**🌎 País:** "
            f"{safe_value(selected['País'])}"
        )

        st.write(
            f"**📍 Ciudad:** "
            f"{safe_value(selected['Ciudad'])}"
        )

        st.write(
            f"**🏆 Temporada:** "
            f"{safe_value(selected['Temporada'])}"
        )

        st.write(
            f"**🔢 Ronda:** "
            f"{safe_value(selected['Ronda'])}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # MERCADO
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '💰 Mercado de cuotas'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<b>🟡 CUOTAS NO CONECTADAS</b>',
            unsafe_allow_html=True
        )

        st.write(
            "TheSportsDB proporciona los datos deportivos "
            "del evento, pero no un feed documentado de "
            "bookmakers/cuotas de apuestas."
        )

        st.write(
            "Por integridad del sistema, no se mostrará "
            "ninguna cuota inventada."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # PROBABILIDAD IMPLÍCITA
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📐 Probabilidad implícita'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Pendiente de cuotas reales. "
            "La probabilidad implícita se calculará "
            "automáticamente cuando exista una cuota válida."
        )

        # ====================================================
        # MODELO
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🧠 Modelo predictivo'
            '</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)

        with m1:

            st.markdown(
                '<div class="card">'
                '<b>Datos históricos</b>'
                '<br><br>'
                '🟡 Pendientes de integrar'
                '</div>',
                unsafe_allow_html=True
            )

        with m2:

            st.markdown(
                '<div class="card">'
                '<b>Probabilidad propia</b>'
                '<br><br>'
                '🟡 Modelo pendiente'
                '</div>',
                unsafe_allow_html=True
            )

        with m3:

            st.markdown(
                '<div class="card">'
                '<b>Confianza del modelo</b>'
                '<br><br>'
                '🟡 Pendiente'
                '</div>',
                unsafe_allow_html=True
            )

        # ====================================================
        # VALUE
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📈 Value Betting'
            '</div>',
            unsafe_allow_html=True
        )

        v1, v2, v3 = st.columns(3)

        with v1:

            st.markdown(
                '<div class="card">'
                '<b>Edge</b>'
                '<br><br>'
                '🟡 Pendiente'
                '</div>',
                unsafe_allow_html=True
            )

        with v2:

            st.markdown(
                '<div class="card">'
                '<b>EV</b>'
                '<br><br>'
                '🟡 Pendiente'
                '</div>',
                unsafe_allow_html=True
            )

        with v3:

            st.markdown(
                '<div class="card">'
                '<b>Value Score</b>'
                '<br><br>'
                '🟡 Pendiente'
                '</div>',
                unsafe_allow_html=True
            )


# ============================================================
# RANKING
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🏆 Ranking de oportunidades'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.write(
    "El ranking se activará cuando el sistema tenga "
    "simultáneamente:"
)

st.write(
    "1. Cuotas reales."
)

st.write(
    "2. Probabilidad propia del modelo."
)

st.write(
    "3. Edge y EV calculados."
)

st.write(
    "No se asignarán puntuaciones artificiales."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INTEGRIDAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔐 Integridad del sistema'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <b>Reglas del Centro de Mando</b>

    <br><br>

    🟢 Solo se muestran datos obtenidos de fuentes reales.

    <br><br>

    🔒 El sistema NO inventa cuotas.

    <br>

    🔒 El sistema NO inventa probabilidades.

    <br>

    🔒 El sistema NO genera Value Score artificial.

    <br>

    🔒 El sistema NO convierte una estimación manual
    en una probabilidad de modelo.

    <br><br>

    Las métricas predictivas aparecerán únicamente
    cuando exista una fuente de datos suficiente
    y un modelo estadístico entrenado.
    """,
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ARQUITECTURA
# ============================================================

with st.expander("🏗️ Ver arquitectura del sistema"):

    st.markdown("""
    ### CAPA 1 — Datos deportivos

    **TheSportsDB**

    Eventos → Equipos → Ligas → Fechas → Estadios
    → resultados → estadísticas disponibles

    ---

    ### CAPA 2 — Mercado

    **Fuente de cuotas**

    Casas → Mercados → Selecciones → Cuotas

    Estado actual: 🟡 Pendiente

    ---

    ### CAPA 3 — Datos históricos

    Resultados históricos → forma → local/visitante
    → rendimiento → variables

    Estado actual: 🟡 Pendiente

    ---

    ### CAPA 4 — Modelo predictivo

    Variables
    ↓
    Modelo estadístico
    ↓
    Probabilidad propia

    Estado actual: 🟡 Pendiente

    ---

    ### CAPA 5 — Motor Quant

    Probabilidad propia
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

    Estado actual: 🟡 Pendiente

    ---

    ### CAPA 6 — Dashboard

    El usuario ve únicamente información
    real, calculada y trazable.
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Centro de Mando Quant · Sports Data Hub · FASE 6"
)

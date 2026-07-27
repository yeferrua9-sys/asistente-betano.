"""
Dashboard de Asistente de Inversiones Deportivas
Fase 2: El Cerebro del Asistente (Motor de Recomendaciones sobre Mock Data)
Enfoque: Micro-mercados (faltas, saques de banda, corners, tiros al arco)

Nota: Toda la probabilidad, cuota y lógica de selección usa datos SIMULADOS.
No hay conexión real a Betano ni a ninguna fuente de datos en vivo.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations

# =========================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Sports Investing Assistant | Micro-Mercados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# ESTILOS PERSONALIZADOS (CSS)
# =========================================================
st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
        }
        .stDataFrame {
            border-radius: 10px;
        }
        div[data-testid="stMetric"] {
            background-color: #1c1f26;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #2c2f36;
        }
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
        }
        .badge-alta {
            background-color: #16a34a;
            color: white;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 12px;
        }
        .badge-media {
            background-color: #ca8a04;
            color: white;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 12px;
        }
        .badge-baja {
            background-color: #dc2626;
            color: white;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 12px;
        }
        .combo-card {
            background: linear-gradient(135deg, #1c1f26 0%, #14181f 100%);
            border: 1px solid #f97316;
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 10px;
        }
        .combo-leg {
            padding: 6px 0;
            border-bottom: 1px dashed #2c2f36;
            font-size: 14px;
        }
        .combo-leg:last-child {
            border-bottom: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATOS BASE (catálogos para filtros)
# =========================================================
LIGAS_FUTBOL = [
    "LaLiga (España)",
    "Premier League (Inglaterra)",
    "Serie A (Italia)",
    "Bundesliga (Alemania)",
    "Ligue 1 (Francia)",
    "Liga BetPlay (Colombia)",
    "Brasileirão (Brasil)",
    "Copa Libertadores",
]

LIGAS_BALONCESTO = [
    "NBA (EE.UU.)",
    "EuroLiga",
    "ACB (España)",
    "Liga Nacional de Básquet (Colombia)",
    "BSN (Puerto Rico)",
]

MERCADOS_FUTBOL_EQUIPO = [
    "Ganador del Partido",
    "Total de Faltas del Equipo",
    "Saques de Banda a Favor (Equipo)",
    "Tiros de Esquina - Total Equipo",
    "Tiros al Arco (Equipo)",
]

MERCADOS_FUTBOL_JUGADOR = [
    "Total de Faltas Cometidas (Jugador)",
    "Tiros al Arco (Jugador)",
    "Tarjetas Amarillas (Jugador)",
]

MERCADOS_BALONCESTO_EQUIPO = [
    "Ganador del Partido",
    "Total de Robos (Equipo)",
]

MERCADOS_BALONCESTO_JUGADOR = [
    "Total de Rebotes (Jugador)",
    "Total de Asistencias (Jugador)",
    "Total de Robos (Jugador)",
    "Puntos + Rebotes + Asistencias",
    "Triples Anotados (Jugador)",
]

EQUIPOS_MOCK = [
    "Real Madrid", "FC Barcelona", "Manchester City", "Liverpool",
    "Inter de Milán", "Bayern Múnich", "PSG", "Atlético Nacional",
    "Millonarios FC", "Flamengo", "River Plate", "Boca Juniors",
]

JUGADORES_MOCK = [
    "J. Bellingham", "V. Junior", "E. Haaland", "M. Salah",
    "L. Martínez", "H. Kane", "K. Mbappé", "F. Muriel",
    "R. Falcao", "G. Barrios", "D. Martínez", "J. Álvarez",
]

# Umbral mínimo de probabilidad para considerar una "Sencilla" destacada
UMBRAL_SENCILLAS = 85.0
# Rango de probabilidad aceptado para las patas de una Combinada
UMBRAL_MIN_COMBINADA = 60.0


# =========================================================
# FUNCIÓN GENERADORA DE MOCK DATA (ahora agrupada por "Partido")
# =========================================================
def generar_datos_mock(deporte: str, liga: str, tipo_mercado: str, n_filas: int = 25) -> pd.DataFrame:
    """
    Genera un DataFrame simulado de micro-mercados deportivos, agrupado
    por partido (Equipo A vs Equipo B). Agrupar por partido es lo que
    permite luego construir "Combinadas" lógicas del mismo evento.

    Esta función NO contiene lógica predictiva real: solo crea
    datos de ejemplo con la estructura que usará el dashboard.
    """
    np.random.seed(hash((liga, tipo_mercado, n_filas)) % (2**32))

    if deporte == "Fútbol":
        mercados_equipo = MERCADOS_FUTBOL_EQUIPO
        mercados_jugador = MERCADOS_FUTBOL_JUGADOR
    else:
        mercados_equipo = MERCADOS_BALONCESTO_EQUIPO
        mercados_jugador = MERCADOS_BALONCESTO_JUGADOR

    # 1) Generar un pool de "partidos" (Equipo A vs Equipo B)
    n_partidos = max(3, n_filas // 6)
    partidos = []
    equipos_disponibles = EQUIPOS_MOCK.copy()
    np.random.shuffle(equipos_disponibles)
    for i in range(n_partidos):
        equipo_a = equipos_disponibles[(2 * i) % len(equipos_disponibles)]
        equipo_b = equipos_disponibles[(2 * i + 1) % len(equipos_disponibles)]
        if equipo_a == equipo_b:
            continue
        fecha = (datetime.now() + timedelta(days=int(np.random.randint(0, 7)))).strftime("%Y-%m-%d")
        partidos.append({
            "partido": f"{equipo_a} vs {equipo_b}",
            "equipo_a": equipo_a,
            "equipo_b": equipo_b,
            "fecha": fecha,
        })

    # 2) Generar filas de micro-mercados repartidas entre esos partidos
    filas = []
    for _ in range(n_filas):
        partido = partidos[np.random.randint(0, len(partidos))]
        equipo = np.random.choice([partido["equipo_a"], partido["equipo_b"]])
        usa_mercado_jugador = np.random.rand() < 0.6

        if usa_mercado_jugador:
            mercado = np.random.choice(mercados_jugador)
            jugador = np.random.choice(JUGADORES_MOCK)
        else:
            mercado = np.random.choice(mercados_equipo)
            jugador = "—"

        if mercado == "Ganador del Partido":
            linea_texto = f"Gana {equipo}"
        else:
            linea = round(np.random.uniform(1.5, 9.5), 1)
            linea_texto = f"Más de {linea}"

        # Cuota simulada (formato decimal, típico de Betano)
        cuota = round(np.random.uniform(1.35, 3.80), 2)

        # Probabilidad de éxito simulada (placeholder, sin modelo real de datos en vivo)
        probabilidad = round(np.random.uniform(45, 96), 1)

        if tipo_mercado == "Combinadas":
            n_selecciones = int(np.random.randint(2, 5))
            etiqueta_tipo = "Combinada"
        else:
            n_selecciones = 1
            etiqueta_tipo = "Sencilla"

        valor_referencial = round((probabilidad / 100) * cuota, 2)

        filas.append({
            "Fecha": partido["fecha"],
            "Liga": liga,
            "Partido": partido["partido"],
            "Equipo": equipo,
            "Jugador": jugador,
            "Mercado": mercado,
            "Línea": linea_texto,
            "Cuota": cuota,
            "Probabilidad de Éxito (%)": probabilidad,
            "Tipo de Apuesta": etiqueta_tipo,
            "N° Selecciones": n_selecciones,
            "Valor Referencial": valor_referencial,
        })

    df = pd.DataFrame(filas)
    return df


def badge_probabilidad(valor: float) -> str:
    """Devuelve un badge HTML de color según el rango de probabilidad."""
    if valor >= 75:
        return f'<span class="badge-alta">{valor}%</span>'
    elif valor >= 60:
        return f'<span class="badge-media">{valor}%</span>'
    else:
        return f'<span class="badge-baja">{valor}%</span>'


# =========================================================
# MOTOR DE RECOMENDACIONES (Fase 2)
# =========================================================
def obtener_sencillas_top(df: pd.DataFrame, umbral: float = UMBRAL_SENCILLAS) -> pd.DataFrame:
    """
    Filtra las apuestas individuales (micro-mercados) cuya probabilidad
    de éxito simulada supera el umbral definido.
    """
    if df.empty:
        return df
    top = df[df["Probabilidad de Éxito (%)"] >= umbral].copy()
    top = top.sort_values("Probabilidad de Éxito (%)", ascending=False)
    return top


def resaltar_fila_verde(row: pd.Series) -> list:
    """Aplica fondo verde a toda la fila (usado con df.style.apply)."""
    return ["background-color: #14532d; color: white;"] * len(row)


def armar_combinada_sugerida(df: pd.DataFrame, min_prob: float = UMBRAL_MIN_COMBINADA,
                              max_patas: int = 3) -> dict | None:
    """
    Bet Builder simplificado: busca, dentro de un mismo partido, entre 2 y
    `max_patas` micro-mercados cuya probabilidad individual supere `min_prob`,
    y arma una combinada multiplicando las cuotas.

    Prioriza el partido cuya combinación de patas tenga la mejor
    probabilidad conjunta simulada (producto de probabilidades).

    Retorna un diccionario con las patas elegidas y la cuota total,
    o None si no hay suficientes candidatos.
    """
    if df.empty or "Partido" not in df.columns:
        return None

    candidatos = df[df["Probabilidad de Éxito (%)"] >= min_prob].copy()
    if candidatos.empty:
        return None

    mejor_combo = None
    mejor_score = -1.0

    for partido, grupo in candidatos.groupby("Partido"):
        grupo = grupo.drop_duplicates(subset=["Mercado", "Línea"])
        if len(grupo) < 2:
            continue

        n_patas = min(max_patas, len(grupo))
        # Se evalúan combinaciones de 2 y hasta n_patas mercados del mismo partido
        for r in range(2, n_patas + 1):
            for combo_idx in combinations(grupo.index, r):
                patas = grupo.loc[list(combo_idx)]
                prob_conjunta = np.prod(patas["Probabilidad de Éxito (%)"] / 100)
                score = prob_conjunta  # métrica simple para elegir la mejor combinada

                if score > mejor_score:
                    mejor_score = score
                    cuota_total = float(np.prod(patas["Cuota"]))
                    mejor_combo = {
                        "partido": partido,
                        "patas": patas.to_dict("records"),
                        "cuota_total": round(cuota_total, 2),
                        "probabilidad_conjunta": round(prob_conjunta * 100, 1),
                    }

    return mejor_combo


# =========================================================
# SIDEBAR — FILTROS
# =========================================================
with st.sidebar:
    st.title("⚙️ Filtros del Dashboard")
    st.caption("Configura el mercado que deseas analizar")

    st.divider()

    deporte = st.selectbox(
        "🏟️ Deporte",
        options=["Fútbol", "Baloncesto"],
        index=0,
    )

    if deporte == "Fútbol":
        ligas_disponibles = LIGAS_FUTBOL
    else:
        ligas_disponibles = LIGAS_BALONCESTO

    liga = st.selectbox(
        "🏆 Liga",
        options=ligas_disponibles,
        index=0,
    )

    tipo_mercado = st.radio(
        "🎯 Tipo de Mercado",
        options=["Sencillas", "Combinadas"],
        index=0,
        horizontal=True,
    )

    st.divider()

    n_filas = st.slider(
        "Cantidad de eventos simulados",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
    )

    st.divider()

    st.markdown("**🧠 Parámetros del Motor de Recomendación**")
    umbral_sencillas_sel = st.slider(
        "Umbral mínimo — Sencillas (%)",
        min_value=70,
        max_value=95,
        value=int(UMBRAL_SENCILLAS),
        step=1,
    )
    umbral_combinada_sel = st.slider(
        "Umbral mínimo por pata — Combinada (%)",
        min_value=50,
        max_value=85,
        value=int(UMBRAL_MIN_COMBINADA),
        step=1,
    )

    st.divider()
    st.caption("📌 Fase 2: Motor de recomendación sobre datos simulados.")


# =========================================================
# GENERAR DATOS (una sola vez por corrida, usados en todo el dashboard)
# =========================================================
df_mercados = generar_datos_mock(deporte, liga, tipo_mercado, n_filas)
df_sencillas_top = obtener_sencillas_top(df_mercados, umbral=umbral_sencillas_sel)
combinada_sugerida = armar_combinada_sugerida(df_mercados, min_prob=umbral_combinada_sel)


# =========================================================
# ENCABEZADO PRINCIPAL
# =========================================================
st.title("📊 Sports Investing Assistant")
st.markdown(
    "##### Análisis de micro-mercados: faltas, saques de banda, corners y tiros al arco"
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Deporte", deporte)
col_b.metric("Liga Seleccionada", liga)
col_c.metric("Tipo de Mercado", tipo_mercado)
col_d.metric("Eventos Cargados", len(df_mercados))

st.divider()

# =========================================================
# 🚨 ALERTAS Y RECOMENDACIONES (parte superior del dashboard)
# =========================================================
st.subheader("🚨 Recomendaciones del Motor (Fase 2)")

col_izq, col_der = st.columns([1.1, 1])

# --- Columna izquierda: mejores Sencillas ---
with col_izq:
    st.markdown(f"**✅ Mejores Sencillas — Probabilidad ≥ {umbral_sencillas_sel}%**")

    if df_sencillas_top.empty:
        st.warning(
            "No se encontraron micro-mercados sencillos que superen el umbral "
            f"de {umbral_sencillas_sel}% con los datos simulados actuales. "
            "Prueba bajar el umbral o generar más eventos.",
            icon="⚠️",
        )
    else:
        mejor_fila = df_sencillas_top.iloc[0]
        st.success(
            f"Mejor pick simulado: **{mejor_fila['Mercado']}** "
            f"({mejor_fila['Equipo']}{'' if mejor_fila['Jugador'] == '—' else ' — ' + mejor_fila['Jugador']}) "
            f"en *{mejor_fila['Partido']}* → **{mejor_fila['Probabilidad de Éxito (%)']}%** "
            f"de probabilidad simulada, cuota **{mejor_fila['Cuota']}**.",
            icon="🎯",
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Sencillas Destacadas", len(df_sencillas_top))
        m2.metric("Probabilidad Promedio", f"{df_sencillas_top['Probabilidad de Éxito (%)'].mean():.1f}%")
        m3.metric("Cuota Promedio", f"{df_sencillas_top['Cuota'].mean():.2f}")

        tabla_verde = df_sencillas_top[[
            "Partido", "Equipo", "Jugador", "Mercado", "Línea",
            "Cuota", "Probabilidad de Éxito (%)",
        ]].reset_index(drop=True)

        st.dataframe(
            tabla_verde.style.apply(resaltar_fila_verde, axis=1),
            use_container_width=True,
            hide_index=True,
        )

# --- Columna derecha: Bet Builder / Combinada sugerida ---
with col_der:
    st.markdown("**🔥 Combinada Sugerida (Bet Builder)**")

    if combinada_sugerida is None:
        st.warning(
            "No se encontró una combinación lógica de 2 o 3 micro-mercados "
            f"del mismo partido con probabilidad individual ≥ {umbral_combinada_sel}%. "
            "Prueba bajar el umbral o generar más eventos.",
            icon="⚠️",
        )
    else:
        patas_html = ""
        for pata in combinada_sugerida["patas"]:
            jugador_txt = "" if pata["Jugador"] == "—" else f" ({pata['Jugador']})"
            patas_html += (
                f'<div class="combo-leg">'
                f'⚽ <b>{pata["Mercado"]}</b>{jugador_txt} — {pata["Línea"]}<br>'
                f'<span style="color:#9ca3af;">Prob. simulada: {pata["Probabilidad de Éxito (%)"]}% '
                f'| Cuota: {pata["Cuota"]}</span>'
                f'</div>'
            )

        st.markdown(
            f"""
            <div class="combo-card">
                <h4 style="margin-top:0;">🔥 Combinada Sugerida</h4>
                <p style="color:#9ca3af; margin-bottom:10px;">
                    Partido: <b>{combinada_sugerida['partido']}</b> ·
                    {len(combinada_sugerida['patas'])} selecciones
                </p>
                {patas_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("Cuota Total Combinada", f"{combinada_sugerida['cuota_total']}")
        c2.metric("Prob. Conjunta Simulada", f"{combinada_sugerida['probabilidad_conjunta']}%")

st.divider()

# =========================================================
# TABLA COMPLETA DE MICRO-MERCADOS
# =========================================================
st.subheader("📋 Tabla Completa de Micro-Mercados")
st.caption("Datos simulados — todos los eventos generados, sin filtrar por umbral")

st.dataframe(
    df_mercados,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Cuota": st.column_config.NumberColumn(format="%.2f"),
        "Probabilidad de Éxito (%)": st.column_config.ProgressColumn(
            "Probabilidad de Éxito (%)",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
        "Valor Referencial": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.divider()

# =========================================================
# RESUMEN RÁPIDO
# =========================================================
st.subheader("📈 Resumen Rápido")

col1, col2, col3 = st.columns(3)
col1.metric("Probabilidad Promedio (Todos)", f"{df_mercados['Probabilidad de Éxito (%)'].mean():.1f}%")
col2.metric("Cuota Promedio (Todos)", f"{df_mercados['Cuota'].mean():.2f}")
col3.metric(
    "Mercado Más Frecuente",
    df_mercados["Mercado"].mode()[0] if not df_mercados.empty else "—",
)

st.info(
    "Este dashboard se encuentra en **Fase 2**: motor de recomendación (Sencillas + Bet Builder) "
    "operando sobre datos **100% simulados**. Las siguientes fases podrán incorporar "
    "fuentes de datos reales y modelos estadísticos más robustos.",
    icon="🛠️",
)

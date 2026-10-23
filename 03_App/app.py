"""
Prototipo Funcional: Sistema de Recomendación de Videojuegos en Steam (2020–2022)
Conectado directamente a 01_Datos (games.csv, recommendations.csv, resultados_test.csv, tuning_cv_temporal.csv)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, save_npz, load_npz

st.set_page_config(
    page_title="Steam RecSys | Motor EASE Top-N",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(90deg, #1B2838 0%, #2B5B84 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        border-bottom: 4px solid #00B894;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-banner h1 { color: #FFFFFF; margin-bottom: 0.2rem; font-size: 2rem; }
    .main-banner p { color: #00B894; font-weight: 600; margin: 0; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-banner">
    <h1>🎮 Steam Game Recommender — Prototipo Funcional Top-N</h1>
    <p>Motor de Recomendación Personalizada basado en Autoencoders Lineales (EASE, λ = 70) + Estrategia Cold Start</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DETECCIÓN AUTOMÁTICA DE 01_Datos Y ARTEFACTOS
# ---------------------------------------------------------
RUTA_DATOS_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_Datos"))
if not os.path.exists(RUTA_DATOS_DEFAULT):
    RUTA_DATOS_DEFAULT = os.path.abspath("01_Datos")

st.sidebar.header("⚙️ Configuración de Datos")
ruta_datos = st.sidebar.text_input("Carpeta 01_Datos:", value=RUTA_DATOS_DEFAULT)
ruta_artefactos = os.path.join(ruta_datos, "artefactos")

def construir_artefactos_desde_csv(carpeta_datos, carpeta_salida):
    """Construye los artefactos EASE directamente desde games.csv y recommendations.csv en 01_Datos."""
    os.makedirs(carpeta_salida, exist_ok=True)
    N_ITEMS = 2000
    USER_MOD = 5
    MIN_USER_POS = 8
    MAX_USERS = 30000
    SEED = 42
    CHUNK = 2_000_000
    LAM_BEST = 70

    vj = pd.read_csv(os.path.join(carpeta_datos, "games.csv"))
    vj["date_release"] = pd.to_datetime(vj["date_release"], errors="coerce")
    vj = vj[vj["date_release"].dt.year.between(2020, 2022)].copy()
    top_items = set(vj.nlargest(N_ITEMS, "user_reviews")["app_id"])

    partes = []
    for ch in pd.read_csv(
        os.path.join(carpeta_datos, "recommendations.csv"),
        usecols=["user_id", "app_id", "date", "is_recommended"],
        chunksize=CHUNK
    ):
        ch = ch[ch["is_recommended"] & ch["app_id"].isin(top_items) & (ch["user_id"] % USER_MOD == 0)]
        partes.append(ch[["user_id", "app_id", "date"]])

    pos = pd.concat(partes, ignore_index=True)
    pos["date"] = pd.to_datetime(pos["date"])
    pos = pos.sort_values("date").drop_duplicates(["user_id", "app_id"], keep="first")

    cnt = pos["user_id"].value_counts()
    pos = pos[pos["user_id"].isin(cnt[cnt >= MIN_USER_POS].index)]

    us = pos["user_id"].unique()
    if len(us) > MAX_USERS:
        rng = np.random.default_rng(SEED)
        pos = pos[pos["user_id"].isin(rng.choice(us, MAX_USERS, replace=False))]

    users = np.sort(pos["user_id"].unique())
    items = np.sort(pos["app_id"].unique())
    uid = {u: i for i, u in enumerate(users)}
    iid = {a: i for i, a in enumerate(items)}
    nU, nI = len(users), len(items)

    c_test = pos["date"].quantile(0.80)
    tv_df = pos[pos["date"] <= c_test]
    r = tv_df["user_id"].map(uid).values
    c = tv_df["app_id"].map(iid).values
    X_trva = csr_matrix((np.ones(len(tv_df)), (r, c)), shape=(nU, nI))

    # Entrenamiento cerrado EASE (λ = 70)
    G = (X_trva.T @ X_trva).toarray() + LAM_BEST * np.eye(nI)
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0)

    pop_final = np.asarray(X_trva.sum(axis=0)).ravel()
    catalogo = vj.set_index("app_id").loc[items].reset_index()
    catalogo = catalogo[["app_id", "title", "rating", "positive_ratio", "price_final", "win", "mac", "linux", "steam_deck"]]
    catalogo["popularidad"] = pop_final

    np.save(os.path.join(carpeta_salida, "ease_B.npy"), B.astype(np.float32))
    save_npz(os.path.join(carpeta_salida, "X_hist.npz"), X_trva)
    np.save(os.path.join(carpeta_salida, "users.npy"), users)
    catalogo.to_csv(os.path.join(carpeta_salida, "catalogo.csv"), index=False)
    with open(os.path.join(carpeta_salida, "config.json"), "w") as f:
        json.dump({"lam": LAM_BEST, "k_default": 10}, f)

@st.cache_resource
def cargar_motor(carpeta_datos):
    posibles_rutas = [
        os.path.join(carpeta_datos, "artefactos"),
        os.path.join(os.path.dirname(__file__), "artefactos"),
        carpeta_datos
    ]
    archivos_req = ["ease_B.npy", "X_hist.npz", "users.npy", "catalogo.csv", "config.json"]

    for r_art in posibles_rutas:
        if all(os.path.exists(os.path.join(r_art, f)) for f in archivos_req):
            B = np.load(os.path.join(r_art, "ease_B.npy"))
            X_hist = load_npz(os.path.join(r_art, "X_hist.npz"))
            users = np.load(os.path.join(r_art, "users.npy"))
            catalogo = pd.read_csv(os.path.join(r_art, "catalogo.csv"))
            with open(os.path.join(r_art, "config.json"), "r") as f:
                config = json.load(f)
            uid_map = {int(u): i for i, u in enumerate(users)}
            return B, X_hist, users, catalogo, config, uid_map, True

    return None, None, np.array([10610195]), None, {"lam": 70, "k_default": 10}, {10610195: 0}, False

B, X_hist, users, catalogo, config, uid_map, modo_completo = cargar_motor(ruta_datos)

# Si existen games.csv y recommendations.csv en 01_Datos pero no se han generado los artefactos aún:
csv_crudos_existen = (
    os.path.exists(os.path.join(ruta_datos, "games.csv")) and
    os.path.exists(os.path.join(ruta_datos, "recommendations.csv"))
)

if modo_completo:
    st.sidebar.success(f"✅ Motor conectado ({len(users):,} usuarios | {len(catalogo):,} juegos)")
elif csv_crudos_existen:
    st.sidebar.warning("⚠️ Se detectaron `games.csv` y `recommendations.csv` en `01_Datos`, pero falta compilar `artefactos/`.")
    if st.sidebar.button("⚡ Compilar Artefactos desde 01_Datos", type="primary"):
        with st.spinner("Procesando CSVs de 01_Datos y entrenando matriz EASE (toma ~1 min)..."):
            construir_artefactos_desde_csv(ruta_datos, ruta_artefactos)
            st.cache_resource.clear()
            st.rerun()

k_top = st.sidebar.slider("Tamaño de lista Top-K (K):", min_value=5, max_value=20, value=int(config.get("k_default", 10)), step=1)

# ---------------------------------------------------------
# PESTAÑAS INTERACTIVAS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "👤 1. Usuario Existente (EASE)",
    "🆕 2. Usuario Nuevo (Cold Start)",
    "📊 3. Tablas de Evaluación (01_Datos)"
])

with tab1:
    st.subheader("Recomendación Colaborativa Personalizada (Warm Start)")
    if modo_completo:
        opciones_u = [10610195] + [int(u) for u in users[:100] if int(u) != 10610195]
        user_sel = st.selectbox("Seleccionar ID de Usuario (Steam):", options=opciones_u, index=0)
        i = uid_map[user_sel]
        vista_u = X_hist[i].toarray().ravel()
        scores = (vista_u @ B).ravel()
        scores[vista_u > 0] = -np.inf
        idx = np.argsort(-scores)[:k_top]
        rec_df = catalogo.iloc[idx].copy()
        rec_df["Score EASE"] = np.round(scores[idx], 4)

        with st.expander(f"📚 Ver Historial de Entrenamiento del Usuario {user_sel} ({int(vista_u.sum())} juegos)"):
            st.dataframe(catalogo.iloc[np.where(vista_u > 0)[0]][["app_id", "title", "rating", "price_final"]], use_container_width=True, hide_index=True)

        st.dataframe(rec_df[["app_id", "title", "rating", "positive_ratio", "price_final", "Score EASE"]], use_container_width=True, hide_index=True)
    else:
        st.info("Presione el botón **'⚡ Compilar Artefactos desde 01_Datos'** en la barra lateral izquierda (o copie su carpeta `artefactos/` dentro de `01_Datos/`) para consultar los 15,759 usuarios en vivo.")

with tab2:
    st.subheader("Atención a Usuarios Nuevos sin Historial (Cold Start)")
    if modo_completo:
        c1, c2, c3, c4 = st.columns(4)
        plat = c1.selectbox("Sistema Operativo:", ["mac", "win", "linux", "Todos"], index=0)
        p_max = c2.slider("Precio Máximo ($ USD):", 0.0, 60.0, 20.0, 1.0)
        r_min = c3.slider("Valoración Positiva Mínima (%):", 50, 98, 80, 2)
        s_deck = c4.checkbox("Compatible con Steam Deck", value=False)

        c_df = catalogo.copy()
        if plat in ("win", "mac", "linux"):
            c_df = c_df[c_df[plat].astype(bool)]
        if s_deck:
            c_df = c_df[c_df["steam_deck"].astype(bool)]
        c_df = c_df[(c_df["price_final"] <= p_max) & (c_df["positive_ratio"] >= r_min)]
        st.dataframe(c_df.sort_values("popularidad", ascending=False).head(k_top), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Resultados Leídos Directamente de `01_Datos`")
    ruta_test_csv = os.path.join(ruta_datos, "resultados_test.csv")
    ruta_cv_csv = os.path.join(ruta_datos, "tuning_cv_temporal.csv")

    if os.path.exists(ruta_test_csv):
        st.markdown("#### 🏆 Evaluación Final en Conjunto de Prueba (`resultados_test.csv`)")
        st.dataframe(pd.read_csv(ruta_test_csv), use_container_width=True, hide_index=True)
    if os.path.exists(ruta_cv_csv):
        st.markdown("#### 🔄 Validación Cruzada Temporal Rolling (`tuning_cv_temporal.csv`)")
        st.dataframe(pd.read_csv(ruta_cv_csv), use_container_width=True, hide_index=True)
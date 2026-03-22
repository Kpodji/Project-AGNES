import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Dashboard climat-agriculture Sénégal",
    page_icon="🌍",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def generate_demo_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rainfall = rng.normal(650, 180, n).clip(80, 1200)
    temp = rng.normal(31, 3.5, n).clip(20, 43)
    ndvi = rng.normal(0.48, 0.15, n).clip(0.05, 0.9)
    humidity = rng.normal(65, 12, n).clip(20, 95)
    soil_proxy = (0.45 * (rainfall / 1200) + 0.35 * ndvi + 0.20 * (humidity / 100)).clip(0, 1)

    # Risk generation rule with noise
    raw_score = (
        1.8 * (temp > 34).astype(float)
        + 1.4 * (rainfall < 500).astype(float)
        + 1.6 * (ndvi < 0.35).astype(float)
        + 1.2 * (humidity < 50).astype(float)
        + rng.normal(0, 0.55, n)
    )
    risk = (raw_score > 2.0).astype(int)

    region_names = [
        "Dakar", "Thiès", "Saint-Louis", "Louga", "Diourbel", "Fatick",
        "Kaolack", "Kaffrine", "Tambacounda", "Kolda", "Ziguinchor", "Sédhiou"
    ]
    region = rng.choice(region_names, n)

    lat_map = {
        "Dakar": 14.69, "Thiès": 14.79, "Saint-Louis": 16.03, "Louga": 15.61,
        "Diourbel": 14.66, "Fatick": 14.34, "Kaolack": 14.15, "Kaffrine": 14.11,
        "Tambacounda": 13.77, "Kolda": 12.89, "Ziguinchor": 12.57, "Sédhiou": 12.71,
    }
    lon_map = {
        "Dakar": -17.45, "Thiès": -16.93, "Saint-Louis": -16.49, "Louga": -16.22,
        "Diourbel": -16.23, "Fatick": -16.41, "Kaolack": -16.07, "Kaffrine": -15.55,
        "Tambacounda": -13.67, "Kolda": -14.95, "Ziguinchor": -16.27, "Sédhiou": -15.55,
    }

    df = pd.DataFrame({
        "region": region,
        "latitude": [lat_map[r] for r in region],
        "longitude": [lon_map[r] for r in region],
        "rainfall_mm": rainfall.round(1),
        "temperature_c": temp.round(1),
        "ndvi": ndvi.round(3),
        "humidity_pct": humidity.round(1),
        "soil_moisture_proxy": soil_proxy.round(3),
        "risk": risk,
    })
    return df


@st.cache_data(show_spinner=False)
def fetch_open_meteo(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    daily = data.get("daily", {})
    if not daily:
        return pd.DataFrame()
    return pd.DataFrame(daily)


def train_model(df: pd.DataFrame):
    feature_cols = ["rainfall_mm", "temperature_c", "ndvi", "humidity_pct", "soil_moisture_proxy"]
    X = df[feature_cols]
    y = df["risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cm": confusion_matrix(y_test, y_pred),
        "feature_importances": pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False),
        "fpr_tpr": roc_curve(y_test, y_proba),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return model, metrics


def make_confusion_matrix_plot(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm)
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité")
    ax.set_xticks([0, 1], ["Faible", "Élevé"])
    ax.set_yticks([0, 1], ["Faible", "Élevé"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def make_roc_plot(fpr, tpr, auc_score: float):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Courbe ROC")
    ax.legend()
    fig.tight_layout()
    return fig


def risk_label(prob: float) -> str:
    if prob < 0.33:
        return "Faible"
    if prob < 0.66:
        return "Moyen"
    return "Élevé"


# -----------------------------
# Data
# -----------------------------
st.title("🌍 Dashboard Streamlit — Climat & Agriculture au Sénégal")
st.caption("Prototype basé sur ton notebook : météo, modèle ML, visualisations et lecture décisionnelle.")

with st.sidebar:
    st.header("Paramètres")
    uploaded = st.file_uploader("Importer un CSV optionnel", type=["csv"])
    st.markdown("**Colonnes attendues** : rainfall_mm, temperature_c, ndvi, humidity_pct, soil_moisture_proxy, risk")
    demo_rows = st.slider("Taille du jeu de données démo", 200, 2000, 600, 100)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = generate_demo_data(demo_rows)

required_cols = {"rainfall_mm", "temperature_c", "ndvi", "humidity_pct", "soil_moisture_proxy", "risk"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Le fichier importé ne contient pas les colonnes requises : {sorted(missing)}")
    st.stop()

model, metrics = train_model(df)

# -----------------------------
# Layout tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vue d'ensemble", "Météo", "Modèle ML", "Carte & décision", "GitHub"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", len(df))
    c2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c3.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
    c4.metric("Part risque élevé", f"{df['risk'].mean():.1%}")

    st.subheader("Aperçu des données")
    st.dataframe(df.head(15), use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Distribution des variables")
        numeric_cols = ["rainfall_mm", "temperature_c", "ndvi", "humidity_pct", "soil_moisture_proxy"]
        selected_col = st.selectbox("Variable à visualiser", numeric_cols)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[selected_col], bins=25)
        ax.set_title(f"Distribution — {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Fréquence")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.subheader("Lecture rapide")
        st.markdown(
            """
            - **rainfall_mm** : pluie cumulée ou estimée
            - **temperature_c** : température moyenne
            - **ndvi** : vigueur de la végétation
            - **humidity_pct** : humidité relative
            - **soil_moisture_proxy** : proxy d’humidité du sol
            - **risk** : 0 = faible risque, 1 = risque élevé
            """
        )

with tab2:
    st.subheader("Historique météo Open-Meteo")
    col1, col2, col3, col4 = st.columns(4)
    latitude = col1.number_input("Latitude", value=14.6937, format="%.4f")
    longitude = col2.number_input("Longitude", value=-17.4441, format="%.4f")
    end = col3.date_input("Date de fin", value=date.today() - timedelta(days=5))
    start = col4.date_input("Date de début", value=end - timedelta(days=30))

    if start > end:
        st.warning("La date de début doit être antérieure à la date de fin.")
    else:
        try:
            weather = fetch_open_meteo(latitude, longitude, str(start), str(end))
            if weather.empty:
                st.info("Aucune donnée météo récupérée pour cette période.")
            else:
                st.dataframe(weather, use_container_width=True)
                a, b = st.columns(2)
                with a:
                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    ax.plot(pd.to_datetime(weather["time"]), weather["temperature_2m_mean"])
                    ax.set_title("Température moyenne")
                    ax.set_ylabel("°C")
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    st.pyplot(fig)
                with b:
                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    ax.bar(pd.to_datetime(weather["time"]), weather["precipitation_sum"])
                    ax.set_title("Précipitations")
                    ax.set_ylabel("mm")
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Impossible de récupérer les données météo : {e}")

with tab3:
    st.subheader("Performance du modèle")
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
    bnds = "±3 à ±6 %"
    c.metric("Incertitude estimée", bnds)

    left, right = st.columns(2)
    with left:
        st.pyplot(make_confusion_matrix_plot(metrics["cm"]))
    with right:
        fpr, tpr, _ = metrics["fpr_tpr"]
        st.pyplot(make_roc_plot(fpr, tpr, metrics["roc_auc"]))

    st.subheader("Importance des variables")
    fi = metrics["feature_importances"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(fi.index, fi.values)
    ax.set_title("Feature importance")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=20)
    fig.tight_layout()
    st.pyplot(fig)

    st.dataframe(
    fi.rename("importance")
    .reset_index()
    .rename(columns={"index":"variable"}),
    width="stretch"
)

with tab4:
    st.subheader("Carte des observations par région")
    map_df = df.groupby(["region", "latitude", "longitude"], as_index=False).agg(
        risk_rate=("risk", "mean"),
        rainfall_mm=("rainfall_mm", "mean"),
        temperature_c=("temperature_c", "mean"),
        ndvi=("ndvi", "mean"),
    )
    map_df["risk_label"] = map_df["risk_rate"].apply(risk_label)
    st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
    st.dataframe(map_df.sort_values("risk_rate", ascending=False), use_container_width=True)

    st.subheader("Simulateur de risque")
    c1, c2, c3, c4, c5 = st.columns(5)
    in_rain = c1.slider("Pluie (mm)", 50, 1200, 450)
    in_temp = c2.slider("Température (°C)", 18, 45, 35)
    in_ndvi = c3.slider("NDVI", 0.05, 0.90, 0.30)
    in_hum = c4.slider("Humidité (%)", 20, 95, 45)
    in_soil = c5.slider("Humidité sol (proxy)", 0.0, 1.0, 0.28)

    sample = pd.DataFrame([{
        "rainfall_mm": in_rain,
        "temperature_c": in_temp,
        "ndvi": in_ndvi,
        "humidity_pct": in_hum,
        "soil_moisture_proxy": in_soil,
    }])
    risk_prob = float(model.predict_proba(sample)[0, 1])
    st.metric("Probabilité de risque agricole élevé", f"{risk_prob:.1%}")
    st.info(f"Niveau interprété : **{risk_label(risk_prob)}**")

    st.markdown(
        """
        ### Brief décisionnel
        - **Risque faible** : surveillance standard.
        - **Risque moyen** : renforcer le suivi local et préparer des actions ciblées.
        - **Risque élevé** : prioriser les zones concernées pour alerte, irrigation, ou appui terrain.
        """
    )

with tab5:
    st.subheader("Envoyer le projet sur GitHub")
    st.markdown(
        """
        #### 1) Crée un dépôt GitHub
        Exemple : `dashboard-streamlit-senegal`

        #### 2) Mets ces fichiers dans ton dossier projet
        - `app.py`
        - `requirements.txt`
        - `README.md`
        - `.gitignore`

        #### 3) Commandes à lancer dans le terminal
        ```bash
        git init
        git add .
        git commit -m "Initial commit - dashboard Streamlit Senegal"
        git branch -M main
        git remote add origin TON_URL_GITHUB
        git push -u origin main
        ```

        #### 4) Pour lancer en local
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```

        #### 5) Pour déployer ensuite
        Tu peux utiliser **Streamlit Community Cloud** en connectant directement ton dépôt GitHub.
        """
    )

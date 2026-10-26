import os
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.inspection import permutation_importance

# ------------------------- Configuration -------------------------
st.set_page_config(
    page_title="HeartCare AI | ML Capstone",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.getenv("HEARTCARE_DB_PATH", "heartcare_history.sqlite3")
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "heart_disease_model.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "model_metrics.csv")
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ------------------------- Theme -------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #071426; color: #e6f1ff; }
[data-testid="stSidebar"] { background: #0b1d33; border-right: 1px solid #164e63; }
h1, h2, h3 { color: #dff7ff !important; }
p, label, .stMarkdown { color: #d7e5f4; }
div[data-testid="stMetric"] {
  background: linear-gradient(135deg, #10243d, #0d3040);
  border: 1px solid #155e75; padding: 14px; border-radius: 14px;
}
.stButton > button, .stDownloadButton > button, button[kind="primary"] {
  background: #0d9488; color: white; border: 0; border-radius: 9px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: #14b8a6; color: white; border: 1px solid #5eead4;
}
div[data-baseweb="input"] input, div[data-baseweb="select"] {
  background-color: #10243d;
}
.small-muted { color: #9fb3c8; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Database / authentication -------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            created_at TEXT NOT NULL,
            predicted_class INTEGER NOT NULL,
            class1_score REAL NOT NULL,
            model_name TEXT NOT NULL,
            input_json TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 250_000)
    return salt.hex(), digest.hex()

def register_user(username, password):
    username = username.strip()
    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if len(password) < 10 or not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        return False, "Use at least 10 characters, including a letter and a number."
    conn = get_conn()
    salt, digest = hash_password(password)
    try:
        conn.execute(
            "INSERT INTO users(username, salt, password_hash, created_at) VALUES(?,?,?,?)",
            (username, salt, digest, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username already exists."
    finally:
        conn.close()

def authenticate(username, password):
    conn = get_conn()
    row = conn.execute(
        "SELECT salt, password_hash FROM users WHERE username=?",
        (username.strip(),)
    ).fetchone()
    conn.close()
    if not row:
        return False
    _, digest = hash_password(password, row[0])
    return hmac.compare_digest(digest, row[1])

# ------------------------- Dataset and model pipeline -------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    response = requests.get(DATA_URL, timeout=25)
    response.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(response.text), header=None, names=COLUMNS, na_values="?")
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])

def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
    }

@st.cache_resource(show_spinner="Loading dataset and training models (first run may take a moment)...")
def prepare_experiment():
    df = load_dataset()
    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    models = {}
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, estimator in make_models().items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])
        # CV is computed only on training partition; held-out test is reserved for final evaluation.
        cv_result = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "precision": "precision",
                     "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=None, error_score="raise"
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        models[name] = pipe
        rows.append({
            "Model": name,
            "CV F1 mean": cv_result["test_f1"].mean(),
            "CV F1 std": cv_result["test_f1"].std(),
            "Test Accuracy": accuracy_score(y_test, pred),
            "Test Precision": precision_score(y_test, pred, zero_division=0),
            "Test Recall": recall_score(y_test, pred, zero_division=0),
            "Test F1": f1_score(y_test, pred, zero_division=0),
            "Test ROC-AUC": roc_auc_score(y_test, prob),
        })

    metrics = pd.DataFrame(rows).sort_values("CV F1 mean", ascending=False).reset_index(drop=True)
    selected_name = metrics.iloc[0]["Model"]  # selected by training-only CV, not test scores
    selected_model = models[selected_name]

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    metrics.to_csv(METRICS_PATH, index=False)

    return df, X_train, X_test, y_train, y_test, models, metrics, selected_name, selected_model

def save_history(username, model_name, inputs, predicted_class, score):
    import json
    conn = get_conn()
    conn.execute(
        """INSERT INTO prediction_history
        (username, created_at, predicted_class, class1_score, model_name, input_json)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (
            username, datetime.now(timezone.utc).isoformat(),
            int(predicted_class), float(score), model_name,
            json.dumps(inputs, sort_keys=True)
        )
    )
    conn.commit()
    conn.close()

def read_history(username):
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT created_at, predicted_class, class1_score, model_name, input_json
        FROM prediction_history WHERE username=? ORDER BY created_at DESC""",
        conn, params=(username,)
    )
    conn.close()
    return df

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ------------------------- Login / register -------------------------
get_conn().close()
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.username:
    st.markdown("<h1>💙 HeartCare AI</h1>", unsafe_allow_html=True)
    st.caption("Machine Learning Capstone • Dark navy & teal")
    st.warning(
        "Educational demonstration only. This application is not a medical device, "
        "does not diagnose disease, and must not guide treatment or urgent-care decisions."
    )
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Sign in", type="primary")
        if login_submit:
            if authenticate(login_username, login_password):
                st.session_state.username = login_username.strip()
                st.rerun()
            else:
                st.error("Sign-in failed. Check your username and password.")
    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Create password (10+ characters)", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            register_submit = st.form_submit_button("Create account", type="primary")
        if register_submit:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = register_user(new_username, new_password)
                (st.success if ok else st.error)(message)
    st.stop()

# ------------------------- Main app -------------------------
st.sidebar.markdown("# 💙 HeartCare AI")
st.sidebar.caption("Machine Learning Capstone")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction Demo", "Model Comparison", "Explainability", "History & Export", "Project Notes"]
)
st.sidebar.markdown("---")
st.sidebar.write(f"Signed in as **{st.session_state.username}**")
if st.sidebar.button("Sign out"):
    st.session_state.username = None
    st.rerun()

st.title("HeartCare AI")
st.caption("Public-dataset ML experiment • Not for clinical use")
st.warning(
    "MEDICAL DISCLAIMER: This is an academic demonstration trained on a small historical dataset. "
    "Its outputs are not diagnoses, are not clinically validated, and must not be used for medical "
    "decisions. Contact a qualified healthcare professional for medical concerns."
)

try:
    df, X_train, X_test, y_train, y_test, models, metrics, selected_name, selected_model = prepare_experiment()
except Exception as exc:
    st.error("The dataset or model pipeline could not be loaded.")
    st.info("Check your internet connection and try again. The app fetches the public UCI CSV on first run.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

if page == "Dashboard":
    st.subheader("Project overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{len(df):,}")
    c2.metric("Input features", len(FEATURES))
    c3.metric("Models compared", len(models))
    c4.metric("Selected model", selected_name)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Target class distribution")
        target_counts = df["target"].map({0: "Class 0", 1: "Class 1"}).value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        target_counts.plot(kind="bar", ax=ax, color=["#14b8a6", "#38bdf8"])
        ax.set_ylabel("Records")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Model metrics")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.download_button("Download dataset CSV", csv_bytes(df), "heart_dataset_processed.csv", "text/csv")

elif page == "Prediction Demo":
    st.subheader("Prediction demo")
    st.info(
        "Enter illustrative dataset values only. Category fields use the UCI Cleveland dataset's "
        "coded values; they are not a clinical intake form."
    )
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
            sex = st.selectbox("Sex code (dataset)", [0, 1], format_func=lambda x: f"{x}")
            cp = st.selectbox("Chest pain type code", [1, 2, 3, 4])
            trestbps = st.number_input("Resting blood pressure", 50, 250, 130)
            chol = st.number_input("Serum cholesterol", 50, 700, 240)
        with col2:
            fbs = st.selectbox("Fasting blood sugar code", [0, 1])
            restecg = st.selectbox("Resting ECG code", [0, 1, 2])
            thalach = st.number_input("Maximum heart rate", 50, 250, 150)
            exang = st.selectbox("Exercise-induced angina code", [0, 1])
        with col3:
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope code", [1, 2, 3])
            ca = st.selectbox("Major vessels code", [0, 1, 2, 3])
            thal = st.selectbox("Thal code", [3, 6, 7])
        submit = st.form_submit_button("Run educational demo", type="primary")

    if submit:
        row = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        input_df = pd.DataFrame([row], columns=FEATURES)
        prediction = int(selected_model.predict(input_df)[0])
        score = float(selected_model.predict_proba(input_df)[0, 1])
        st.metric("Predicted dataset class", str(prediction))
        st.metric("Uncalibrated model score for class 1", f"{score:.1%}")
        st.caption(
            "This score is not an individual's clinical probability of disease. "
            "It is not a diagnosis and should not be interpreted medically."
        )
        save_history(st.session_state.username, selected_name, row, prediction, score)
        st.success("Demo result saved to this app's local history database.")

elif page == "Model Comparison":
    st.subheader("Model comparison")
    st.caption("Model selection is based on mean 5-fold cross-validation F1 on training data. Test metrics are for final reporting only.")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    metric_choices = ["CV F1 mean", "Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC"]
    chosen_metric = st.selectbox("Chart metric", metric_choices)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=metrics, x="Model", y=chosen_metric, ax=ax, color="#14b8a6")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    ax.set_title(chosen_metric)
    st.pyplot(fig)
    plt.close(fig)
    st.download_button("Download model metrics CSV", csv_bytes(metrics), "model_comparison_metrics.csv", "text/csv")

    if st.checkbox("Show selected model's held-out test report"):
        test_pred = selected_model.predict(X_test)
        st.text(classification_report(y_test, test_pred, zero_division=0))
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, test_pred, display_labels=["Class 0", "Class 1"], ax=ax, cmap="Blues"
        )
        st.pyplot(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_estimator(selected_model, X_test, y_test, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

elif page == "Explainability":
    st.subheader("Feature importance & explainability")
    st.caption("Permutation importance measures how much shuffling a feature affects this test set's F1 score; it does not establish causation.")
    with st.spinner("Calculating permutation importance..."):
        pi = permutation_importance(
            selected_model, X_test, y_test, scoring="f1", n_repeats=10, random_state=42
        )
    imp_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean F1 decrease": pi.importances_mean,
        "Std": pi.importances_std
    }).sort_values("Mean F1 decrease", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=imp_df, x="Mean F1 decrease", y="Feature", ax=ax, color="#2dd4bf")
    ax.set_title("Permutation importance on held-out test set")
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(imp_df, use_container_width=True, hide_index=True)
    st.download_button("Download feature importance CSV", csv_bytes(imp_df), "feature_importance.csv", "text/csv")
    st.info("Importance can be unstable on small datasets and correlated features. It should not be treated as medical explanation.")

elif page == "History & Export":
    st.subheader("Your demo prediction history")
    history = read_history(st.session_state.username)
    if history.empty:
        st.info("No prediction demo records yet. Use the Prediction Demo page to add one.")
    else:
        st.dataframe(history.drop(columns=["input_json"]), use_container_width=True, hide_index=True)
        st.download_button("Download my history CSV", csv_bytes(history), "my_prediction_history.csv", "text/csv")
        report_text = (
            "HeartCare AI — Educational Project Report\n"
            f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n"
            f"Username: {st.session_state.username}\n"
            f"Demo records: {len(history)}\n"
            f"Selected model: {selected_name}\n\n"
            "IMPORTANT: Not a medical device; not clinically validated; not for diagnosis or treatment.\n\n"
            "Model comparison:\n" + metrics.to_string(index=False) + "\n\n"
            "History summary:\n" + history.drop(columns=["input_json"]).to_string(index=False)
        )
        st.download_button("Download text report", report_text, "heartcare_project_report.txt", "text/plain")
    st.caption("History is stored locally in SQLite by this app. It is not encrypted at rest; do not enter real patient data.")

elif page == "Project Notes":
    st.subheader("Capstone documentation")
    st.markdown("""
**Problem definition:** Explore whether a supervised classifier can distinguish the dataset's two target classes from selected input attributes.

**Dataset:** UCI Heart Disease, processed Cleveland file. This is a small, historical dataset; it is not a representative contemporary clinical population.

**Target:** `num > 0` is mapped to class 1; `num == 0` is mapped to class 0. The target is dataset-defined, not a diagnosis.

**Preprocessing:** Numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding inside an sklearn pipeline.

**Validation:** Stratified 5-fold cross-validation on the training partition selects the model by mean F1. A held-out test partition is used for final metrics.

**Metrics:** Accuracy, precision, recall, F1 and ROC-AUC are reported. Metrics depend on the split and dataset.

**Limitations:** Small sample, old data, potential dataset bias, no external validation, no clinical calibration, and no prospective evaluation. Feature importance is not causation.

**Responsible use:** Academic demonstration only. Never use outputs to diagnose, triage, or choose treatment. Seek qualified clinical care for health concerns.
""")
    st.download_button("Download current model metrics", csv_bytes(metrics), "model_metrics.csv", "text/csv")

st.markdown("---")
st.markdown('<p class="small-muted">HeartCare AI • Student capstone prototype • Educational use only</p>', unsafe_allow_html=True)

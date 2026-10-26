# HeartCare AI — Machine Learning Capstone

A Streamlit educational prototype with:
- Dark navy / teal theme
- Local account registration and sign-in (PBKDF2 password hashing)
- UCI Cleveland dataset loading
- Logistic Regression, Random Forest, Extra Trees comparison
- Training-only stratified 5-fold cross-validation for model selection
- Held-out test metrics, confusion matrix and ROC curve
- Permutation importance
- Per-user demo history and CSV/text export

## Run locally

1. Install Python 3.10+.
2. Open this folder in VS Code.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

The first launch needs internet access to fetch the public UCI dataset. It trains models and writes artifacts under `artifacts/`.

## Data and artifacts

- Dataset source: UCI Machine Learning Repository, Heart Disease (processed Cleveland data).
- The app creates `artifacts/heart_disease_model.joblib` and `artifacts/model_metrics.csv` at runtime.
- The SQLite database `heartcare_history.sqlite3` is created on first launch.

No pre-existing model file is required. The app trains models on first run and caches them for the session.

## Important limitations

This is a student project, not a medical device. Predictions are not diagnoses, not clinically validated, and must not be used for medical decisions. Do not enter real patient data.

Authentication and history are suitable only for a local demo. SQLite history is not encrypted at rest; use a proper identity provider and secured database before any public deployment. Do not deploy this as a real patient-facing clinical application.

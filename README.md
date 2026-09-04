# Soil Engineering Properties Prediction Using SVR

## Features
- Separate training Excel upload
- Flexible input-feature and target selection
- Automatic SVR configuration selection
- Median imputation
- Automatic StandardScaler normalization
- 5-fold cross-validation
- Independent unseen testing
- R², RMSE, MAE and MAPE
- Actual vs Predicted graph
- Manual prediction within the training-data range
- SHAP Explainable AI
- Gemini AI interpretation

## Workflow
Training Data
→ Preprocessing
→ StandardScaler
→ Automatic SVR Optimization
→ 5-Fold Cross-Validation
→ Final SVR Model
→ Independent Unseen Testing
→ Manual Prediction
→ SHAP Explainable AI
→ Gemini Interpretation

## Important
SVR performs all numerical predictions.

Gemini does NOT train the SVR model and does NOT calculate the numerical prediction. Gemini only generates written interpretations.

## SHAP
Because SVR is not a tree model, the app uses Kernel SHAP on a representative sample.

A larger Mean Absolute SHAP Value means greater overall influence on model predictions.

SHAP importance does not prove causality.

## Gemini API Key
Add your key to Streamlit Secrets:

```toml
GEMINI_API_KEY = "your_actual_api_key"
```

Do not place the actual API key inside `app.py`.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub Safety
Do not upload:
- `.streamlit/secrets.toml`
- `.env`
- API keys
- passwords or tokens

Use this `.gitignore`:

```text
.streamlit/secrets.toml
.env
__pycache__/
*.pyc
```

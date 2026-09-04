# Soil Engineering Properties Prediction Using SVR

## Features
- Automatic SVR optimization
- StandardScaler normalization
- 5-fold cross-validation
- Independent unseen testing
- R², RMSE, MAE and MAPE
- Actual vs Predicted graph
- Manual prediction
- Strict input limits based on the training-data range

## Workflow
Training Data → Preprocessing → StandardScaler → Automatic SVR Optimization → 5-Fold CV → Final SVR Model → Unseen Testing → Manual Prediction

## Important
SVR is sensitive to feature scale, therefore normalization is automatically included inside the machine-learning pipeline.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

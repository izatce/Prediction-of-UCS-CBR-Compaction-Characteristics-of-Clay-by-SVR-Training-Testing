import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Soil SVR Prediction", page_icon="🌍", layout="wide")

CONFIGS = [
    {"kernel":"rbf","C":1.0,"epsilon":0.05,"gamma":"scale"},
    {"kernel":"rbf","C":10.0,"epsilon":0.05,"gamma":"scale"},
    {"kernel":"rbf","C":50.0,"epsilon":0.05,"gamma":"scale"},
    {"kernel":"rbf","C":100.0,"epsilon":0.10,"gamma":"scale"},
    {"kernel":"rbf","C":100.0,"epsilon":0.20,"gamma":"scale"},
    {"kernel":"linear","C":1.0,"epsilon":0.05,"gamma":"scale"},
    {"kernel":"linear","C":10.0,"epsilon":0.10,"gamma":"scale"},
]

def load_excel(f):
    df = pd.read_excel(f)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def model_pipeline(cfg):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel=cfg["kernel"], C=float(cfg["C"]),
                    epsilon=float(cfg["epsilon"]), gamma=cfg["gamma"]))
    ])

def metrics(y, p):
    return {
        "R2": float(r2_score(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "MAPE": float(mean_absolute_percentage_error(y, p) * 100)
    }

def clear_results():
    for k in ["svr_model","svr_cv","svr_config","svr_X","svr_y",
              "svr_test_results","svr_manual_result","svr_shap_results",
              "svr_gemini_test","svr_gemini_manual"]:
        st.session_state.pop(k, None)

def gemini_generate(prompt):
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY was not found. Add it to Streamlit Secrets.")
    from google import genai
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text

st.title("🌍 Soil Engineering Properties Prediction Using SVR")
st.caption("Automatic Optimization • 5-Fold CV • Unseen Testing • Manual Prediction • SHAP • Gemini")

st.sidebar.header("⚙️ SVR Settings")
random_state = st.sidebar.number_input("Random State", 0, 9999, 42, 1)
st.sidebar.info("StandardScaler normalization is automatically included.")

tabs = st.tabs([
    "📘 Training Data",
    "⚙️ Train SVR",
    "📊 CV Results",
    "📕 Unseen Testing",
    "🎯 Manual Prediction",
    "🔍 SHAP Explainable AI",
    "🤖 Gemini Interpretation"
])

# =========================================================
# TRAINING DATA
# =========================================================
with tabs[0]:
    st.header("1. Upload Training Dataset")
    train_file = st.file_uploader(
        "Upload Training Excel File",
        type=["xlsx","xls"],
        key="svr_training_uploader"
    )

    if train_file:
        try:
            file_id = f"{train_file.name}_{train_file.size}"
            if st.session_state.get("svr_train_file_id") != file_id:
                st.session_state["svr_train_file_id"] = file_id
                st.session_state.pop("svr_features_widget", None)
                clear_results()

            df = load_excel(train_file)
            st.session_state["svr_training_df"] = df

            st.success(f"Training dataset loaded: {len(df)} samples")
            st.dataframe(df.head(10), use_container_width=True)

            cols = list(df.columns)
            target_index = cols.index("UCS (kPa)") if "UCS (kPa)" in cols else len(cols)-1

            target = st.selectbox(
                "Select Output / Target Property",
                cols,
                index=target_index,
                key="svr_target_widget"
            )

            available = [c for c in cols if c != target]
            id_names = {"id","sample id","sample_id","sample no","sample no.","s.no","s.no."}
            defaults = [c for c in available if str(c).strip().lower() not in id_names]

            if "svr_features_widget" not in st.session_state:
                st.session_state["svr_features_widget"] = defaults

            st.session_state["svr_features_widget"] = [
                c for c in st.session_state["svr_features_widget"] if c in available
            ]

            features = st.multiselect(
                "Select Input Features",
                available,
                key="svr_features_widget"
            )

            if len(features) >= 2:
                st.success(f"Target: {target} | Input Features: {len(features)}")
            else:
                st.warning("Select at least two input features.")

        except Exception as e:
            st.error(f"Could not read training data: {e}")

ready = (
    "svr_training_df" in st.session_state
    and "svr_target_widget" in st.session_state
    and len(st.session_state.get("svr_features_widget", [])) >= 2
)

if ready:
    target = st.session_state["svr_target_widget"]
    features = list(st.session_state["svr_features_widget"])
    work = st.session_state["svr_training_df"][features + [target]].copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[target]).reset_index(drop=True)
    X = work[features]
    y = work[target]

# =========================================================
# TRAIN SVR
# =========================================================
with tabs[1]:
    st.header("2. Automatic SVR Optimization and Training")

    if not ready:
        st.warning("Upload training data and select a target plus at least two input features.")
    else:
        st.write(f"**Target:** {target}")
        st.write(f"**Valid training samples:** {len(X)}")
        st.write("**Preprocessing:** Median imputation + StandardScaler")

        if len(X) < 10:
            st.error("At least 10 valid samples are recommended.")
        elif st.button("🚀 Automatically Optimize and Train SVR", type="primary", key="svr_train_button"):
            try:
                kf = KFold(n_splits=5, shuffle=True, random_state=int(random_state))
                rows = []
                progress = st.progress(0)

                for i, cfg in enumerate(CONFIGS):
                    fold_scores = []

                    for tr, va in kf.split(X):
                        m = model_pipeline(cfg)
                        m.fit(X.iloc[tr], y.iloc[tr])
                        p = m.predict(X.iloc[va])
                        fold_scores.append(metrics(y.iloc[va], p))

                    rows.append({
                        **cfg,
                        "Mean R2": np.mean([z["R2"] for z in fold_scores]),
                        "Mean RMSE": np.mean([z["RMSE"] for z in fold_scores]),
                        "Mean MAE": np.mean([z["MAE"] for z in fold_scores]),
                        "Mean MAPE (%)": np.mean([z["MAPE"] for z in fold_scores])
                    })
                    progress.progress(int((i+1)/len(CONFIGS)*100))

                cv = pd.DataFrame(rows).sort_values(
                    ["Mean RMSE","Mean R2"],
                    ascending=[True,False]
                ).reset_index(drop=True)

                cv.insert(0, "Rank", range(1, len(cv)+1))
                best = cv.iloc[0]

                cfg = {
                    "kernel": best["kernel"],
                    "C": float(best["C"]),
                    "epsilon": float(best["epsilon"]),
                    "gamma": best["gamma"]
                }

                final_model = model_pipeline(cfg)
                final_model.fit(X, y)

                st.session_state["svr_model"] = final_model
                st.session_state["svr_cv"] = cv
                st.session_state["svr_config"] = cfg
                st.session_state["svr_X"] = X.copy()
                st.session_state["svr_y"] = y.copy()

                for k in ["svr_test_results","svr_manual_result","svr_shap_results",
                          "svr_gemini_test","svr_gemini_manual"]:
                    st.session_state.pop(k, None)

                progress.empty()
                st.success("SVR model trained successfully.")
                st.dataframe(pd.DataFrame([cfg]), use_container_width=True)

            except Exception as e:
                st.error(f"Training error: {e}")

# =========================================================
# CV RESULTS
# =========================================================
with tabs[2]:
    st.header("3. Five-Fold Cross-Validation Results")

    if "svr_cv" not in st.session_state:
        st.info("Train the SVR model first.")
    else:
        cv = st.session_state["svr_cv"]
        best = cv.iloc[0]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mean R²", f"{best['Mean R2']:.4f}")
        c2.metric("Mean RMSE", f"{best['Mean RMSE']:.4f}")
        c3.metric("Mean MAE", f"{best['Mean MAE']:.4f}")
        c4.metric("Mean MAPE", f"{best['Mean MAPE (%)']:.2f}%")

        st.dataframe(
            cv.style.format({
                "Mean R2":"{:.4f}",
                "Mean RMSE":"{:.4f}",
                "Mean MAE":"{:.4f}",
                "Mean MAPE (%)":"{:.2f}"
            }),
            use_container_width=True
        )

# =========================================================
# UNSEEN TESTING
# =========================================================
with tabs[3]:
    st.header("4. Independent Unseen Data Testing")

    if "svr_model" not in st.session_state:
        st.warning("Train the SVR model first.")
    else:
        model = st.session_state["svr_model"]
        model_features = list(st.session_state["svr_X"].columns)
        model_target = st.session_state["svr_y"].name

        test_file = st.file_uploader(
            "Upload Independent Unseen Testing Excel File",
            type=["xlsx","xls"],
            key="svr_test_uploader"
        )

        if test_file:
            try:
                test_df = load_excel(test_file)
                st.dataframe(test_df.head(10), use_container_width=True)

                required = model_features + [model_target]
                missing = [c for c in required if c not in test_df.columns]

                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                elif st.button("🔬 Run Independent Unseen Testing", type="primary", key="svr_run_test_button"):
                    w = test_df[required].copy()
                    for c in required:
                        w[c] = pd.to_numeric(w[c], errors="coerce")

                    w = w.dropna(subset=[model_target])

                    if len(w) == 0:
                        st.error("No valid testing rows were found.")
                    else:
                        p = model.predict(w[model_features])
                        result_metrics = metrics(w[model_target], p)

                        result_table = w.copy()
                        result_table[f"Predicted {model_target}"] = p
                        result_table["Residual (Actual - Predicted)"] = w[model_target].values - p

                        st.session_state["svr_test_results"] = {
                            "actual": w[model_target].to_numpy(),
                            "predicted": p,
                            "metrics": result_metrics,
                            "table": result_table,
                            "target": model_target
                        }

                        st.session_state.pop("svr_gemini_test", None)
                        st.success("Independent unseen testing completed successfully.")

            except Exception as e:
                st.error(f"Testing error: {e}")

        if "svr_test_results" in st.session_state:
            r = st.session_state["svr_test_results"]
            m = r["metrics"]

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Testing R²", f"{m['R2']:.4f}")
            c2.metric("Testing RMSE", f"{m['RMSE']:.4f}")
            c3.metric("Testing MAE", f"{m['MAE']:.4f}")
            c4.metric("Testing MAPE", f"{m['MAPE']:.2f}%")

            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(r["actual"], r["predicted"], alpha=0.75)
            lo = min(np.min(r["actual"]), np.min(r["predicted"]))
            hi = max(np.max(r["actual"]), np.max(r["predicted"]))
            ax.plot([lo,hi], [lo,hi], linestyle="--")
            ax.set_xlabel(f"Actual {r['target']}")
            ax.set_ylabel(f"Predicted {r['target']}")
            ax.set_title("Actual vs Predicted")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            st.dataframe(r["table"], use_container_width=True)

            st.download_button(
                "⬇️ Download Testing Predictions",
                r["table"].to_csv(index=False).encode("utf-8"),
                "SVR_Unseen_Testing_Results.csv",
                "text/csv",
                key="svr_download_results"
            )

# =========================================================
# MANUAL PREDICTION
# =========================================================
with tabs[4]:
    st.header("5. Predict One New Soil Sample")

    if "svr_model" not in st.session_state:
        st.warning("Train the SVR model first.")
    else:
        st.info("All inputs are restricted to the minimum and maximum values observed in the training data.")

        model = st.session_state["svr_model"]
        train_X = st.session_state["svr_X"]
        target_name = st.session_state["svr_y"].name

        entered = {}

        for feat in train_X.columns:
            values = pd.to_numeric(train_X[feat], errors="coerce").dropna()
            mn, mx = float(values.min()), float(values.max())
            med = float(values.median())

            st.caption(f"Allowed range for {feat}: {mn:.4f} to {mx:.4f}")

            entered[feat] = st.number_input(
                f"Enter {feat}",
                min_value=mn,
                max_value=mx,
                value=med,
                format="%.6f",
                key=f"svr_manual_{feat}"
            )

        if st.button(f"🎯 Predict {target_name}", type="primary", key="svr_manual_predict_button"):
            try:
                new_data = pd.DataFrame([entered], columns=list(train_X.columns))
                prediction = float(model.predict(new_data)[0])

                st.session_state["svr_manual_result"] = {
                    "target": target_name,
                    "prediction": prediction,
                    "inputs": entered
                }

                st.session_state.pop("svr_gemini_manual", None)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        if "svr_manual_result" in st.session_state:
            r = st.session_state["svr_manual_result"]
            st.subheader("🎉 Final Predicted Value")
            st.metric(f"Predicted {r['target']}", f"{r['prediction']:.4f}")

            st.dataframe(pd.DataFrame({
                "Input Parameter": list(r["inputs"].keys()),
                "Entered Value": list(r["inputs"].values())
            }), use_container_width=True)

# =========================================================
# SHAP EXPLAINABLE AI
# =========================================================
with tabs[5]:
    st.header("6. SHAP Explainable AI")

    if "svr_model" not in st.session_state:
        st.warning("Train the SVR model first.")
    else:
        st.info(
            "SVR uses Kernel SHAP, which can be slower than Tree SHAP. "
            "The app uses a representative sample to keep computation practical."
        )

        if st.button("🔍 Generate SHAP Feature Importance", type="primary", key="svr_shap_button"):
            try:
                import shap

                with st.spinner("Running SHAP analysis..."):
                    pipeline = st.session_state["svr_model"]
                    X_train = st.session_state["svr_X"]
                    feature_names = list(X_train.columns)

                    sample_size = min(50, len(X_train))
                    X_sample = X_train.sample(
                        n=sample_size,
                        random_state=int(random_state)
                    )

                    imputer = pipeline.named_steps["imputer"]
                    scaler = pipeline.named_steps["scaler"]
                    svr = pipeline.named_steps["svr"]

                    X_imputed = imputer.transform(X_sample)
                    X_scaled = scaler.transform(X_imputed)

                    background_size = min(20, len(X_scaled))
                    background = shap.sample(
                        X_scaled,
                        background_size,
                        random_state=int(random_state)
                    )

                    explainer = shap.KernelExplainer(
                        svr.predict,
                        background
                    )

                    nsamples = min(
                        200,
                        max(50, 2 * len(feature_names) + 10)
                    )

                    shap_values = np.asarray(
                        explainer.shap_values(
                            X_scaled,
                            nsamples=nsamples
                        )
                    )

                    importance = pd.DataFrame({
                        "Feature": feature_names,
                        "Mean Absolute SHAP Value": np.mean(np.abs(shap_values), axis=0)
                    }).sort_values(
                        "Mean Absolute SHAP Value",
                        ascending=False
                    ).reset_index(drop=True)

                    st.session_state["svr_shap_results"] = {
                        "importance": importance
                    }

                st.success("SHAP analysis completed successfully.")

            except ImportError:
                st.error("SHAP is not installed. Check requirements.txt.")
            except Exception as e:
                st.error(f"SHAP error: {e}")

        if "svr_shap_results" in st.session_state:
            importance = st.session_state["svr_shap_results"]["importance"]

            st.subheader("Global Feature Importance")
            st.dataframe(importance, use_container_width=True)

            fig, ax = plt.subplots(figsize=(9,6))
            plot_data = importance.sort_values("Mean Absolute SHAP Value")
            ax.barh(
                plot_data["Feature"],
                plot_data["Mean Absolute SHAP Value"]
            )
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.set_title("Global Feature Importance Based on SHAP")
            ax.grid(axis="x", alpha=0.3)
            st.pyplot(fig)

            st.info(
                "A larger Mean Absolute SHAP Value means that the feature "
                "has greater overall influence on the SVR model predictions. "
                "SHAP importance does not prove causality."
            )

# =========================================================
# GEMINI INTERPRETATION
# =========================================================
with tabs[6]:
    st.header("7. Gemini AI Interpretation")

    st.info(
        "SVR performs the numerical prediction. Gemini only provides "
        "a written interpretation of model results and predictions."
    )

    # TESTING INTERPRETATION
    st.subheader("📊 Interpret Independent Testing Results")

    if "svr_test_results" not in st.session_state:
        st.info("Complete unseen testing first.")
    else:
        if st.button(
            "🤖 Generate Gemini Interpretation of Testing Results",
            type="primary",
            key="svr_gemini_test_button"
        ):
            try:
                r = st.session_state["svr_test_results"]
                m = r["metrics"]
                cfg = st.session_state["svr_config"]
                features = list(st.session_state["svr_X"].columns)

                shap_text = "SHAP analysis has not been performed."
                if "svr_shap_results" in st.session_state:
                    shap_text = (
                        st.session_state["svr_shap_results"]["importance"]
                        .head(5)
                        .to_string(index=False)
                    )

                prompt = f"""
You are assisting with a PhD-level geotechnical engineering study.

A Support Vector Regression model predicts: {r['target']}

Input features:
{", ".join(features)}

Best SVR configuration:
Kernel: {cfg['kernel']}
C: {cfg['C']}
Epsilon: {cfg['epsilon']}
Gamma: {cfg['gamma']}

Independent unseen testing metrics:
R² = {m['R2']:.4f}
RMSE = {m['RMSE']:.4f}
MAE = {m['MAE']:.4f}
MAPE = {m['MAPE']:.2f}%

Top SHAP information:
{shap_text}

Write a concise academic interpretation under these headings:
1. Overall Model Performance
2. Generalization on Unseen Data
3. Meaning of Error Metrics
4. Feature Importance
5. Research Caution

Do not invent results.
Do not claim causality from SHAP.
Clearly state that SVR produced numerical predictions and Gemini only generated the written interpretation.
"""
                with st.spinner("Gemini is preparing the interpretation..."):
                    st.session_state["svr_gemini_test"] = gemini_generate(prompt)

            except Exception as e:
                st.error(f"Gemini error: {e}")

        if "svr_gemini_test" in st.session_state:
            st.markdown(st.session_state["svr_gemini_test"])

    st.markdown("---")

    # MANUAL PREDICTION INTERPRETATION
    st.subheader("🎯 Interpret Manual Soil Prediction")

    if "svr_manual_result" not in st.session_state:
        st.info("Make a manual prediction first.")
    else:
        if st.button(
            "🤖 Generate Gemini Interpretation of This Prediction",
            type="primary",
            key="svr_gemini_manual_button"
        ):
            try:
                r = st.session_state["svr_manual_result"]
                inputs_text = "\n".join(
                    [f"{k}: {v}" for k,v in r["inputs"].items()]
                )

                prompt = f"""
You are assisting with geotechnical engineering research.

A trained Support Vector Regression model predicted:

Target property: {r['target']}
Predicted value: {r['prediction']:.4f}

Input values:
{inputs_text}

Write a concise academic interpretation under:
1. Predicted Result
2. Input Data Context
3. Engineering Interpretation
4. Prediction Limitation

Important:
- SVR produced the numerical prediction.
- Gemini only provides written interpretation.
- Do not make unsupported causal claims.
- The result should be interpreted within the training-data range and study conditions.
"""
                with st.spinner("Gemini is interpreting the prediction..."):
                    st.session_state["svr_gemini_manual"] = gemini_generate(prompt)

            except Exception as e:
                st.error(f"Gemini error: {e}")

        if "svr_gemini_manual" in st.session_state:
            st.markdown(st.session_state["svr_gemini_manual"])

st.markdown("---")
st.caption(
    "Soil Engineering Properties Prediction • SVR • 5-Fold CV • "
    "Independent Unseen Testing • Manual Prediction • SHAP • Gemini"
)

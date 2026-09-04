import os
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

st.set_page_config(page_title="Soil SVR Prediction", page_icon="🌍", layout="wide")
st.title("🌍 Soil Engineering Properties Prediction Using SVR")
st.caption("Automatic SVR Optimization • 5-Fold CV • Unseen Testing • Manual Prediction")

CONFIGS=[
 {"kernel":"rbf","C":1.0,"epsilon":0.05,"gamma":"scale"},
 {"kernel":"rbf","C":10.0,"epsilon":0.05,"gamma":"scale"},
 {"kernel":"rbf","C":50.0,"epsilon":0.10,"gamma":"scale"},
 {"kernel":"rbf","C":100.0,"epsilon":0.10,"gamma":"scale"},
 {"kernel":"linear","C":1.0,"epsilon":0.05,"gamma":"scale"},
 {"kernel":"linear","C":10.0,"epsilon":0.10,"gamma":"scale"},
]

def load(f):
    d=pd.read_excel(f); d.columns=[str(c).strip() for c in d.columns]; return d

def make_model(c):
    return Pipeline([("imputer",SimpleImputer(strategy="median")),
                     ("scaler",StandardScaler()),
                     ("svr",SVR(kernel=c["kernel"],C=c["C"],epsilon=c["epsilon"],gamma=c["gamma"]))])

def score(y,p):
    return {"R2":r2_score(y,p),"RMSE":np.sqrt(mean_squared_error(y,p)),
            "MAE":mean_absolute_error(y,p),"MAPE":mean_absolute_percentage_error(y,p)*100}

if "model" not in st.session_state: st.session_state.model=None

tabs=st.tabs(["📘 Training Data","⚙️ Train SVR","📊 CV Results","📕 Unseen Testing","🎯 Manual Prediction","ℹ️ Roadmap"])

with tabs[0]:
    f=st.file_uploader("Upload Training Dataset",type=["xlsx","xls"])
    if f:
        d=load(f); st.session_state.train=d
        st.success(f"{len(d)} training samples loaded")
        st.dataframe(d.head(),use_container_width=True)
        cols=list(d.columns)
        target=st.selectbox("Select Output / Target Property",cols,
                            index=cols.index("UCS (kPa)") if "UCS (kPa)" in cols else len(cols)-1)
        features=st.multiselect("Select Input Features",[c for c in cols if c!=target],
                                default=[c for c in cols if c!=target and c.lower() not in ["id","sample_id","sample id"]])
        st.session_state.target=target; st.session_state.features=features

ready=hasattr(st.session_state,"train") and len(st.session_state.get("features",[]))>=2
if ready:
    work=st.session_state.train[st.session_state.features+[st.session_state.target]].copy()
    for c in work.columns: work[c]=pd.to_numeric(work[c],errors="coerce")
    work=work.dropna(subset=[st.session_state.target])
    X=work[st.session_state.features]; y=work[st.session_state.target]

with tabs[1]:
    if not ready: st.warning("Upload data and select at least two inputs.")
    else:
        st.info("Normalization is automatically included using StandardScaler.")
        if st.button("🚀 Automatically Optimize and Train SVR",type="primary"):
            kf=KFold(n_splits=5,shuffle=True,random_state=42); rows=[]
            for c in CONFIGS:
                fold=[]
                for tr,va in kf.split(X):
                    m=make_model(c); m.fit(X.iloc[tr],y.iloc[tr])
                    fold.append(score(y.iloc[va],m.predict(X.iloc[va])))
                rows.append({**c,"Mean R²":np.mean([z["R2"] for z in fold]),
                             "Mean RMSE":np.mean([z["RMSE"] for z in fold]),
                             "Mean MAE":np.mean([z["MAE"] for z in fold]),
                             "Mean MAPE (%)":np.mean([z["MAPE"] for z in fold])})
            cv=pd.DataFrame(rows).sort_values(["Mean RMSE","Mean R²"],ascending=[True,False]).reset_index(drop=True)
            best=cv.iloc[0].to_dict()
            cfg={k:best[k] for k in ["kernel","C","epsilon","gamma"]}
            m=make_model(cfg); m.fit(X,y)
            st.session_state.model=m; st.session_state.cv=cv
            st.session_state.X=X; st.session_state.cfg=cfg
            st.success("SVR model trained successfully.")
            st.write("Best configuration:",cfg)

with tabs[2]:
    if "cv" not in st.session_state: st.info("Train the model first.")
    else:
        b=st.session_state.cv.iloc[0]
        a,b1,c,d=st.columns(4)
        a.metric("Mean R²",f"{b['Mean R²']:.4f}")
        b1.metric("Mean RMSE",f"{b['Mean RMSE']:.4f}")
        c.metric("Mean MAE",f"{b['Mean MAE']:.4f}")
        d.metric("Mean MAPE",f"{b['Mean MAPE (%)']:.2f}%")
        st.dataframe(st.session_state.cv,use_container_width=True)

with tabs[3]:
    if st.session_state.model is None: st.warning("Train the SVR model first.")
    else:
        f=st.file_uploader("Upload Independent Unseen Testing Dataset",type=["xlsx","xls"],key="test")
        if f and st.button("🔬 Run Unseen Testing",type="primary"):
            d=load(f); target=st.session_state.target; features=st.session_state.features
            missing=[c for c in features+[target] if c not in d.columns]
            if missing: st.error("Missing columns: "+", ".join(missing))
            else:
                w=d[features+[target]].copy()
                for c in w.columns: w[c]=pd.to_numeric(w[c],errors="coerce")
                w=w.dropna(subset=[target]); p=st.session_state.model.predict(w[features])
                st.session_state.test=(w[target].to_numpy(),p)
        if "test" in st.session_state and isinstance(st.session_state.test,tuple):
            actual,pred=st.session_state.test; s=score(actual,pred)
            a,b,c,d=st.columns(4)
            a.metric("Testing R²",f"{s['R2']:.4f}"); b.metric("Testing RMSE",f"{s['RMSE']:.4f}")
            c.metric("Testing MAE",f"{s['MAE']:.4f}"); d.metric("Testing MAPE",f"{s['MAPE']:.2f}%")
            fig,ax=plt.subplots(); ax.scatter(actual,pred)
            lo=min(actual.min(),pred.min()); hi=max(actual.max(),pred.max())
            ax.plot([lo,hi],[lo,hi],"--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.grid(alpha=.3)
            st.pyplot(fig)

with tabs[4]:
    if st.session_state.model is None: st.warning("Train the SVR model first.")
    else:
        vals={}
        for feat in st.session_state.features:
            z=pd.to_numeric(st.session_state.X[feat],errors="coerce").dropna()
            mn,mx=float(z.min()),float(z.max())
            st.caption(f"Allowed range: {mn:.4f} to {mx:.4f}")
            vals[feat]=st.number_input(feat,min_value=mn,max_value=mx,value=float(z.median()),format="%.6f")
        if st.button(f"🎯 Predict {st.session_state.target}"):
            p=st.session_state.model.predict(pd.DataFrame([vals]))[0]
            st.metric(f"Predicted {st.session_state.target}",f"{p:.4f}")

with tabs[5]:
    st.markdown("""### SVR Roadmap
1. Upload training dataset
2. Select inputs and target
3. Impute missing input values
4. Normalize inputs using StandardScaler
5. Compare SVR configurations using 5-fold CV
6. Select best configuration by CV performance
7. Train final SVR model
8. Upload independent unseen testing data
9. Evaluate R², RMSE, MAE and MAPE
10. Predict a new soil sample within the training-data range
""")

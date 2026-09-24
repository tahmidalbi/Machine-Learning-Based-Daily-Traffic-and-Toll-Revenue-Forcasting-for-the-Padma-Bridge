from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "padma_bridge_ml_complete" / "data" / "raw" / "padma_toll_report_with_holidays_weather.csv"
OUT, MODELS, FIGS = ROOT/"outputs", ROOT/"models", ROOT/"outputs"/"figures"
for p in (OUT, MODELS, FIGS): p.mkdir(parents=True, exist_ok=True)
SEED = 42

def number(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def metrics(y, p):
    ok=np.isfinite(y)&np.isfinite(p); y=np.asarray(y)[ok]; p=np.asarray(p)[ok]
    return {"MAE":mean_absolute_error(y,p), "RMSE":mean_squared_error(y,p)**.5,
            "MAPE":np.mean(np.abs((y-p)/np.maximum(np.abs(y),1)))*100, "R2":r2_score(y,p), "N":len(y)}

def load_features():
    d=pd.read_csv(SOURCE); d["date"]=pd.to_datetime(d.Date,dayfirst=True); d=d.sort_values("date").reset_index(drop=True)
    ren={"Traffic_Mawa":"mawa_traffic","Traffic_Jajira":"jajira_traffic","Cash_Mawa":"mawa_cash","Cash_Jajira":"jajira_cash","Total_Traffic":"total_traffic","Total_Cash":"total_cash"}
    for a,b in ren.items(): d[b]=number(d[a])
    d["dow"]=d.date.dt.dayofweek; d["month"]=d.date.dt.month; d["doy"]=d.date.dt.dayofyear
    for col,period in [("dow",7),("month",12),("doy",365.25)]:
        d[col+"_sin"]=np.sin(2*np.pi*d[col]/period); d[col+"_cos"]=np.cos(2*np.pi*d[col]/period)
    d["railway"]= (d.date>=pd.Timestamp("2023-11-01")).astype(int)
    d["eid_relative_day"]=np.nan
    eid_dates=pd.DatetimeIndex(d.loc[d.eid.eq(1),"date"].unique())
    if len(eid_dates):
        d["eid_relative_day"]=[int((x-eid_dates[np.argmin(np.abs((eid_dates-x).days))]).days) for x in d.date]
    hist=["mawa_traffic","jajira_traffic","mawa_cash","jajira_cash","total_traffic","total_cash"]
    d["mawa_traffic_share"]=d.mawa_traffic/(d.mawa_traffic+d.jajira_traffic)
    d["mawa_cash_share"]=d.mawa_cash/(d.mawa_cash+d.jajira_cash)
    d["direction_difference"]=d.mawa_traffic-d.jajira_traffic
    d["mawa_rpv"]=d.mawa_cash/d.mawa_traffic; d["jajira_rpv"]=d.jajira_cash/d.jajira_traffic
    hist += ["mawa_traffic_share","mawa_cash_share","direction_difference","mawa_rpv","jajira_rpv"]
    for c in hist:
        for lag in (1,2,3,7,14,28): d[f"{c}_lag{lag}"]=d[c].shift(lag)
        for w in (7,14,28):
            d[f"{c}_mean{w}"]=d[c].shift(1).rolling(w).mean(); d[f"{c}_std{w}"]=d[c].shift(1).rolling(w).std()
    base=["dow_sin","dow_cos","month_sin","month_cos","doy_sin","doy_cos","weekend","is_holiday","eid","days_to_nearest_eid","eid_relative_day","temp_mean_c","temp_max_c","temp_min_c","rainfall_mm","humidity_pct","wind_speed_kmh","railway"]
    features=base+[c for c in d if "_lag" in c or "_mean" in c or "_std" in c]
    return d.dropna(subset=features).reset_index(drop=True),features

def estimators():
    return {
      "Linear Regression":make_pipeline(SimpleImputer(),StandardScaler(),LinearRegression()),
      "Random Forest":make_pipeline(SimpleImputer(),RandomForestRegressor(n_estimators=350,max_features=.7,min_samples_leaf=2,n_jobs=-1,random_state=SEED)),
      "XGBoost":make_pipeline(SimpleImputer(),XGBRegressor(n_estimators=450,max_depth=5,learning_rate=.04,subsample=.85,colsample_bytree=.8,objective="reg:squarederror",n_jobs=-1,random_state=SEED)),
      "CatBoost":make_pipeline(SimpleImputer(),CatBoostRegressor(iterations=450,depth=7,learning_rate=.05,loss_function="MAE",verbose=False,random_seed=SEED,thread_count=-1)),
      "Residual MLP":make_pipeline(SimpleImputer(),StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,32,16),activation="relu",early_stopping=True,max_iter=450,random_state=SEED))}

def main():
    d,features=load_features(); n=len(d); a=int(.70*n); b=int(.85*n); train=d.iloc[:b]; test=d.iloc[b:].copy()
    targets=["mawa_traffic","jajira_traffic","mawa_cash","jajira_cash"]
    rows=[]; preds=pd.DataFrame({"date":test.date}); fitted={}
    for target in targets:
        ytr=train[target].values; yte=test[target].values
        naive1=test[f"{target}_lag1"].values; naive7=test[f"{target}_lag7"].values
        for name,p in [("Persistence-1",naive1),("Seasonal-7",naive7)]: rows.append({"strategy":"Direct","target":target,"model":name,**metrics(yte,p)})
        for name,model in estimators().items():
            if name=="Residual MLP":
                base=train[f"{target}_lag1"].values; model.fit(train[features],ytr-base); p=test[f"{target}_lag1"].values+model.predict(test[features])
            else: model.fit(train[features],ytr); p=model.predict(test[features])
            rows.append({"strategy":"Direct","target":target,"model":name,**metrics(yte,p)}); preds[f"{target}_{name}"]=p; fitted[(target,name)]=model
            joblib.dump(model,MODELS/f"direct_{target}_{name.lower().replace(' ','_')}.joblib")
    # Total-and-share: use actual published total as an oracle-like strategy comparison on the test set.
    for target,share,total,other in [("mawa_traffic_share","mawa_traffic_share","total_traffic","traffic"),("mawa_cash_share","mawa_cash_share","total_cash","cash")]:
        ytr=train[share].values; yte=test[share].values
        for name,model in estimators().items():
            if name=="Residual MLP":
                base=train[f"{share}_lag1"].values; model.fit(train[features],ytr-base); ps=test[f"{share}_lag1"].values+model.predict(test[features])
            else: model.fit(train[features],ytr); ps=model.predict(test[features])
            ps=np.clip(ps,0,1); totalv=test[total].values; mawa=totalv*ps; jajira=totalv*(1-ps)
            for side,y,p in [("mawa_"+other,test["mawa_"+other].values,mawa),("jajira_"+other,test["jajira_"+other].values,jajira)]: rows.append({"strategy":"Total-and-share","target":side,"model":name,**metrics(y,p),"share_MAE":mean_absolute_error(yte,ps),"reconciliation_MAE":0.0})
            joblib.dump(model,MODELS/f"share_{other}_{name.lower().replace(' ','_')}.joblib")
    res=pd.DataFrame(rows); res.to_csv(OUT/"directional_test_metrics.csv",index=False); preds.to_csv(OUT/"direct_test_predictions.csv",index=False)
    # event profile and plot
    ev=d[d.eid_relative_day.between(-14,14)].groupby("eid_relative_day").agg(mawa_traffic=("mawa_traffic","mean"),jajira_traffic=("jajira_traffic","mean"),mawa_share=("mawa_traffic_share","mean"),n=("date","size")).reset_index(); ev.to_csv(OUT/"eid_direction_profile.csv",index=False)
    plt.figure(figsize=(9,4.8)); plt.plot(ev.eid_relative_day,ev.mawa_traffic,label="Mawa"); plt.plot(ev.eid_relative_day,ev.jajira_traffic,label="Jajira"); plt.axvline(0,color="k",ls="--"); plt.xlabel("Days relative to Eid"); plt.ylabel("Mean vehicles"); plt.legend(); plt.tight_layout(); plt.savefig(FIGS/"eid_direction_profile.png",dpi=180); plt.close()
    # best-model plot
    best=res[(res.strategy=="Direct") & (~res.model.str.contains("Persistence|Seasonal"))].sort_values("MAE").groupby("target").first().reset_index()
    best.to_csv(OUT/"best_directional_models.csv",index=False)
    fig,axs=plt.subplots(2,2,figsize=(11,7));
    for ax,t in zip(axs.ravel(),targets):
        name=best.loc[best.target==t,"model"].iloc[0]; ax.plot(test.date,test[t],label="Actual",lw=1); ax.plot(test.date,preds[f"{t}_{name}"],label=name,lw=1); ax.set_title(t); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS/"best_actual_vs_predicted.png",dpi=180); plt.close(fig)
    summary={"source":str(SOURCE),"rows_modelled":n,"features":len(features),"split":{"train_validation_rows":b,"test_rows":n-b,"test_start":str(test.date.min().date()),"test_end":str(test.date.max().date())},"leakage_policy":"All observed traffic, toll, share and rolling features shifted >=1 day.","note":"Total-and-share test comparison uses published test-day totals to isolate share allocation accuracy; deployment requires a separately forecast total."}
    (OUT/"run_summary.json").write_text(json.dumps(summary,indent=2))
    make_pdf(res,best,summary)

def make_pdf(res,best,summary):
    styles=getSampleStyleSheet(); small=ParagraphStyle("small",parent=styles["BodyText"],fontSize=8,leading=10)
    doc=SimpleDocTemplate(str(ROOT/"directional_model_training_results.pdf"),pagesize=landscape(A4),rightMargin=28,leftMargin=28,topMargin=28,bottomMargin=28)
    story=[Paragraph("Padma Bridge Directional Forecasting — Trained Model Results",styles["Title"]),Spacer(1,10),Paragraph("Implementation of ashique portion work.pdf. Existing project files were not modified.",styles["BodyText"]),Spacer(1,10),Paragraph(f"Modelled {summary['rows_modelled']} complete daily rows with {summary['features']} leakage-safe features. Test period: {summary['split']['test_start']} to {summary['split']['test_end']} ({summary['split']['test_rows']} days).",styles["BodyText"]),Spacer(1,10),Paragraph("Best direct models by test MAE",styles["Heading2"])]
    tab=[["Target","Model","MAE","RMSE","MAPE %","R2"]]+[[r.target,r.model,f"{r.MAE:,.2f}",f"{r.RMSE:,.2f}",f"{r.MAPE:.2f}",f"{r.R2:.3f}"] for _,r in best.iterrows()]
    t=Table(tab,colWidths=[1.35*inch,1.25*inch,.95*inch,.95*inch,.8*inch,.65*inch]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244062")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [t,Spacer(1,10),Image(str(FIGS/"best_actual_vs_predicted.png"),width=9.2*inch,height=5.85*inch),PageBreak(),Paragraph("Complete model comparison",styles["Heading2"])]
    view=res.copy(); view=view.sort_values(["strategy","target","MAE"])
    tab=[["Strategy","Target","Model","MAE","RMSE","MAPE %","R2"]]+[[r.strategy,r.target,r.model,f"{r.MAE:,.1f}",f"{r.RMSE:,.1f}",f"{r.MAPE:.2f}",f"{r.R2:.3f}"] for _,r in view.iterrows()]
    t=Table(tab,repeatRows=1,colWidths=[1.05*inch,1.15*inch,1.1*inch,.85*inch,.85*inch,.7*inch,.55*inch]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244062")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.5)])); story += [t,PageBreak(),Paragraph("Eid directional pattern",styles["Heading2"]),Image(str(FIGS/"eid_direction_profile.png"),width=9*inch,height=4.8*inch),Spacer(1,8),Paragraph("Interpret Mawa/Jajira as toll-plaza sides, not northbound/southbound, until BBA definitions are independently confirmed. Published side and total fields sometimes disagree. The total-and-share comparison therefore measures share allocation using the published total and is not a stand-alone deployable total forecast.",small),Spacer(1,8),Paragraph("Files delivered: trained estimators in models/, full metrics and predictions in outputs/, and reproducible source in train_directional.py.",small)]
    doc.build(story)

if __name__=="__main__": main()


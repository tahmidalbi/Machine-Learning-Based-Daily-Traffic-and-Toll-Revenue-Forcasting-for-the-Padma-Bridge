from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from catboost import CatBoostRegressor
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parent; SOURCE=ROOT.parent/"padma_bridge_ml_complete"/"data"/"raw"/"padma_toll_report_with_holidays_weather.csv"
OUT,MODELS,FIGS=ROOT/"outputs",ROOT/"models",ROOT/"outputs"/"figures"
for p in (OUT,MODELS,FIGS): p.mkdir(parents=True,exist_ok=True)
SEED=42
def num(s): return pd.to_numeric(s.astype(str).str.replace(",","",regex=False).str.replace(r"[^0-9.\-]","",regex=True),errors="coerce")
def met(y,p):
    y=np.asarray(y);p=np.asarray(p);return {"MAE":mean_absolute_error(y,p),"RMSE":mean_squared_error(y,p)**.5,"MAPE":np.mean(np.abs((y-p)/np.maximum(np.abs(y),1)))*100,"R2":r2_score(y,p),"N":len(y)}
def build():
    d=pd.read_csv(SOURCE); d["date"]=pd.to_datetime(d.Date,dayfirst=True); d=d.sort_values("date").reset_index(drop=True)
    d["traffic"]=num(d.Total_Traffic); d["toll"]=num(d.Total_Cash); d["dow"]=d.date.dt.dayofweek; d["month"]=d.date.dt.month; d["doy"]=d.date.dt.dayofyear
    for c in ["traffic","toll"]:
        for lag in range(0,29): d[f"origin_{c}_lag{lag}"]=d[c].shift(lag)
        for w in (7,14,28): d[f"origin_{c}_mean{w}"]=d[c].shift(1).rolling(w).mean();d[f"origin_{c}_std{w}"]=d[c].shift(1).rolling(w).std()
    # Each row is an origin date; only origin/past observations are inputs. Target calendar/weather is joined by horizon.
    frames=[]
    for h in range(1,31):
        x=d.copy(); x["forecast_horizon"]=h; x["target_date"]=x.date+pd.to_timedelta(h,unit="D")
        x["target_traffic"]=d.traffic.shift(-h); x["target_toll"]=d.toll.shift(-h)
        x["target_dow"]=x.target_date.dt.dayofweek; x["target_month"]=x.target_date.dt.month; x["target_doy"]=x.target_date.dt.dayofyear
        for c,per in [("target_dow",7),("target_month",12),("target_doy",365.25)]: x[c+"_sin"]=np.sin(2*np.pi*x[c]/per);x[c+"_cos"]=np.cos(2*np.pi*x[c]/per)
        # Known-in-advance event values are obtained from target date in the historical evaluation.
        future=d[["date","weekend","is_holiday","eid","days_to_nearest_eid","temp_mean_c","temp_max_c","temp_min_c","rainfall_mm","humidity_pct","wind_speed_kmh"]].rename(columns={"date":"target_date",**{c:"target_"+c for c in ["weekend","is_holiday","eid","days_to_nearest_eid","temp_mean_c","temp_max_c","temp_min_c","rainfall_mm","humidity_pct","wind_speed_kmh"]}})
        x=x.merge(future,on="target_date",how="left"); x["target_railway"]=(x.target_date>=pd.Timestamp("2023-11-01")).astype(int); frames.append(x)
    z=pd.concat(frames,ignore_index=True)
    feats=[c for c in z if c.startswith("origin_") or c.startswith("target_") and c not in ["target_date","target_traffic","target_toll"]]+["forecast_horizon"]
    z=z.dropna(subset=feats+["target_traffic","target_toll"]).sort_values(["date","forecast_horizon"]).reset_index(drop=True)
    return z,feats
def models(): return {
 "Random Forest":make_pipeline(SimpleImputer(),RandomForestRegressor(n_estimators=280,max_features=.75,min_samples_leaf=2,n_jobs=-1,random_state=SEED)),
 "XGBoost":make_pipeline(SimpleImputer(),XGBRegressor(n_estimators=380,max_depth=6,learning_rate=.04,subsample=.85,colsample_bytree=.8,objective="reg:squarederror",n_jobs=-1,random_state=SEED)),
 "CatBoost":make_pipeline(SimpleImputer(),CatBoostRegressor(iterations=380,depth=7,learning_rate=.05,loss_function="MAE",verbose=False,random_seed=SEED,thread_count=-1)),
 "Residual MLP":make_pipeline(SimpleImputer(),StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,32),early_stopping=True,max_iter=300,random_state=SEED))}
def main():
    z,feats=build(); origins=np.array(sorted(z.date.unique())); c1=origins[int(.70*len(origins))]; c2=origins[int(.85*len(origins))]; train=z[z.date<c2]; test=z[z.date>=c2].copy()
    rows=[]; pred=pd.DataFrame({"origin_date":test.date,"target_date":test.target_date,"horizon":test.forecast_horizon})
    for target in ["traffic","toll"]:
        ytr=train["target_"+target].values;yte=test["target_"+target].values
        seasonal=test.apply(lambda r:r["origin_"+target+"_lag"+str(int((7-r.forecast_horizon%7)%7))],axis=1).values
        for name,p in [("Persistence",test["origin_"+target+"_lag0"].values),("Seasonal-7",seasonal)]:
            for band,mask in bands(test): rows.append({"target":target,"model":name,"horizon_band":band,**met(yte[mask],p[mask])})
        for name,m in models().items():
            if name=="Residual MLP":
                base=train["origin_"+target+"_lag0"].values;m.fit(train[feats],ytr-base);p=test["origin_"+target+"_lag0"].values+m.predict(test[feats])
            else:m.fit(train[feats],ytr);p=m.predict(test[feats])
            pred[target+"_"+name]=p; joblib.dump(m,MODELS/f"{target}_{name.lower().replace(' ','_')}.joblib")
            for band,mask in bands(test): rows.append({"target":target,"model":name,"horizon_band":band,**met(yte[mask],p[mask])})
    res=pd.DataFrame(rows);res.to_csv(OUT/"multihorizon_test_metrics.csv",index=False);pred.to_csv(OUT/"multihorizon_test_predictions.csv",index=False)
    best=res[~res.model.str.contains("Persistence|Seasonal")].query("horizon_band=='all'").sort_values("MAE").groupby("target").first().reset_index();best.to_csv(OUT/"best_models.csv",index=False)
    # aggregate weekly and 30-day origin-level errors
    ag=[]
    for target in ["traffic","toll"]:
      for model in ["Random Forest","XGBoost","CatBoost","Residual MLP"]:
       for H in [7,30]:
        q=test[test.forecast_horizon<=H][["date","target_"+target]].copy();q["p"]=pred.loc[test.forecast_horizon<=H,target+"_"+model].values;g=q.groupby("date").sum(numeric_only=True);ag.append({"target":target,"model":model,"period_days":H,**met(g["target_"+target],g.p)})
    pd.DataFrame(ag).to_csv(OUT/"aggregate_period_metrics.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,target in zip(axs,["traffic","toll"]):
      q=res[(res.target==target)&(~res.model.str.contains("Persistence|Seasonal"))&res.horizon_band.isin(["1 day","2-7 days","8-30 days"])]
      for name,g in q.groupby("model"):ax.plot(g.horizon_band,g.MAPE,marker="o",label=name)
      ax.set_title(target.title()+" error by horizon");ax.set_ylabel("MAPE (%)");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIGS/"error_by_horizon.png",dpi=180);plt.close(fig)
    summary={"expanded_samples":len(z),"features":len(feats),"origin_split":{"train_validation_end":str(pd.Timestamp(c2).date()),"test_origins":int(test.date.nunique()),"test_start":str(test.date.min().date()),"test_end":str(test.date.max().date())},"leakage_policy":"All samples from one origin stay in one split; observed bridge values stop at the origin date.","weather_note":"Historical target-date weather is used for backtest evaluation; operational 8-30 day forecasts require forecasts or seasonal climatology."};(OUT/"run_summary.json").write_text(json.dumps(summary,indent=2));make_pdf(res,best,summary)
def bands(d):
    h=d.forecast_horizon.values
    return [("all",np.ones(len(d),bool)),("1 day",h==1),("2-7 days",(h>=2)&(h<=7)),("8-30 days",h>=8),("holiday",d.target_is_holiday.values==1),("Eid +/-7d",d.target_days_to_nearest_eid.values<=7)]
def make_pdf(res,best,s):
    st=getSampleStyleSheet();small=ParagraphStyle("small",parent=st["BodyText"],fontSize=8,leading=10);doc=SimpleDocTemplate(str(ROOT/"multihorizon_model_training_results.pdf"),pagesize=landscape(A4),rightMargin=28,leftMargin=28,topMargin=28,bottomMargin=28)
    story=[Paragraph("Padma Bridge Multi-Horizon Forecasting — Trained Results",st["Title"]),Spacer(1,8),Paragraph("Implementation of ashique portion work 2.pdf. Existing project files were not modified.",st["BodyText"]),Spacer(1,8),Paragraph(f"Expanded training design: {s['expanded_samples']:,} origin-horizon samples, {s['features']} predictors, horizons 1–30. Untouched test origins: {s['origin_split']['test_start']} to {s['origin_split']['test_end']}.",st["BodyText"]),Spacer(1,8),Paragraph("Best global models by overall test MAE",st["Heading2"])]
    tab=[["Target","Model","MAE","RMSE","MAPE %","R2"]]+[[r.target,r.model,f"{r.MAE:,.2f}",f"{r.RMSE:,.2f}",f"{r.MAPE:.2f}",f"{r.R2:.3f}"] for _,r in best.iterrows()];t=Table(tab,colWidths=[1.1*inch,1.25*inch,1.1*inch,1.1*inch,.8*inch,.7*inch]);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244062")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey)]));story += [t,Spacer(1,10),Image(str(FIGS/"error_by_horizon.png"),width=9.4*inch,height=3.85*inch),PageBreak(),Paragraph("Complete horizon and event comparison",st["Heading2"])]
    q=res.sort_values(["target","horizon_band","MAE"]);tab=[["Target","Band","Model","N","MAE","RMSE","MAPE %","R2"]]+[[r.target,r.horizon_band,r.model,str(int(r.N)),f"{r.MAE:,.1f}",f"{r.RMSE:,.1f}",f"{r.MAPE:.2f}",f"{r.R2:.3f}"] for _,r in q.iterrows()];t=Table(tab,repeatRows=1,colWidths=[.7*inch,.85*inch,1.0*inch,.55*inch,.9*inch,.9*inch,.65*inch,.55*inch]);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244062")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.4)]));story += [t,PageBreak(),Paragraph("Interpretation and deployment notes",st["Heading2"]),Paragraph("The backtest preserves forecast-origin grouping and uses no bridge observation after the origin date. Historical target-date weather is used as if it were a supplied forecast; for real 8–30 day use, replace unavailable weather with an actual forecast or climatological average and report the extra uncertainty. Holiday/Eid samples are sparse, so their metrics are less stable. Models and all row-level predictions are supplied for reproducibility.",small),Spacer(1,8),Paragraph("Files delivered: models/, outputs/multihorizon_test_metrics.csv, outputs/aggregate_period_metrics.csv, outputs/multihorizon_test_predictions.csv, and train_multihorizon.py.",small)];doc.build(story)
if __name__=="__main__":main()

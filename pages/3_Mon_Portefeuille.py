# -*- coding: utf-8 -*-
"""
v7.9 — Mon Portefeuille IA stricte & Conseils de Vente + Priorité
Affichage optimisé + Benchmark Total / PEA / CTO
"""

import os, json, numpy as np, pandas as pd, altair as alt, streamlit as st
from lib import (
    fetch_prices, compute_metrics, price_levels_from_row, decision_label_from_row,
    company_name_from_ticker, get_profile_params, load_profile
)

# --- CONFIG
st.set_page_config(page_title="Mon Portefeuille", page_icon="💼", layout="wide")
st.title("💼 Mon Portefeuille — IA stricte & suivi performance")


# --- Chargement portefeuille
DATA_PATH = "data/portfolio.json"
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    pd.DataFrame(columns=["Ticker", "Type", "Qty", "PRU", "Name"]).to_json(
        DATA_PATH, orient="records", indent=2, force_ascii=False
    )

try:
    pf = pd.read_json(DATA_PATH)
except:
    pf = pd.DataFrame(columns=["Ticker", "Type", "Qty", "PRU", "Name"])

for c in ["Ticker","Type","Qty","PRU","Name"]:
    if c not in pf.columns:
        pf[c] = "" if c in ("Ticker","Type","Name") else 0.0


# --- Choix affichage & Benchmark
periode = st.sidebar.radio("Période graphique", ["1 jour", "7 jours", "30 jours"], index=0)
days = {"1 jour":2, "7 jours":10, "30 jours":35}[periode]

bench_name = st.sidebar.selectbox("Indice de comparaison", ["CAC 40","DAX","S&P 500","NASDAQ 100"], index=0)
bench_map = {"CAC 40":"^FCHI", "DAX":"^GDAXI", "S&P 500":"^GSPC", "NASDAQ 100":"^NDX"}
bench = bench_map[bench_name]


# --- Portefeuille éditable
st.subheader("📝 Portefeuille")
edited = st.data_editor(
    pf, use_container_width=True, hide_index=True, num_rows="dynamic"
)

if st.button("💾 Enregistrer Modifs"):
    edited["Ticker"] = edited["Ticker"].str.upper()
    edited.to_json(DATA_PATH, orient="records", indent=2, force_ascii=False)
    st.rerun()

if edited.empty:
    st.info("Ajoute des actions pour commencer."); st.stop()


# --- Analyse IA stricte
tickers = edited["Ticker"].dropna().unique().tolist()
h = fetch_prices(tickers, days=240)
m = compute_metrics(h)
merged = edited.merge(m, on="Ticker", how="left")

profil = load_profile()
volmax = get_profile_params(profil)["vol_max"]

rows = []
for _, r in merged.iterrows():
    px = float(r.get("Close", np.nan))
    qty, pru = r["Qty"], r["PRU"]
    name = r["Name"] or company_name_from_ticker(r["Ticker"])

    levels = price_levels_from_row(r, profil)
    dec = decision_label_from_row(r, held=True, vol_max=volmax)

    val = px * qty if np.isfinite(px) else np.nan
    gain = (px - pru) * qty if np.isfinite(px) and np.isfinite(pru) else np.nan
    perf = ((px/pru)-1)*100 if np.isfinite(px) and pru>0 else np.nan

    ma120 = float(r.get("MA120", np.nan))
    ma240 = float(r.get("MA240", np.nan))
    trend = "🌱" if ma120 > ma240 else ("🌧" if ma120 < ma240 else "⚖️")

    px_stop = levels["stop"]; px_target = levels["target"]
    if np.isfinite(px) and np.isfinite(px_target) and px >= px_target:
        priority = "🎯 Vendre"
    elif np.isfinite(perf) and perf > 12 and trend != "🌱":
        priority = "⚖️ Alléger"
    elif np.isfinite(px) and np.isfinite(px_stop) and px <= px_stop:
        priority = "🚨 Couper"
    else:
        priority = "✅ Conserver"

    prox = ((px/levels["entry"])-1)*100 if np.isfinite(px) else np.nan
    emoji = "🟢" if abs(prox)<=2 else ("⚠️" if abs(prox)<=5 else "🔴")

    rows.append({
        "Nom": name,
        "Ticker": r["Ticker"],
        "Type": r["Type"],
        "Décision IA": dec,
        "🎯 Priorité": priority,
        "Cours (€)": round(px,2) if np.isfinite(px) else None,
        "Qté": qty,
        "PRU (€)": round(pru,2) if np.isfinite(pru) else None,
        "Valeur (€)": round(val,2) if np.isfinite(val) else None,
        "Gain (€)": round(gain,2) if np.isfinite(gain) else None,
        "Perf%": round(perf,2) if np.isfinite(perf) else None,
        "Entrée (€)": levels["entry"],
        "Objectif (€)": levels["target"],
        "Stop (€)": levels["stop"],
        "Proximité (%)": round(prox,2) if np.isfinite(prox) else None,
        "Signal Entrée": emoji
    })

out = pd.DataFrame(rows)

# --- Style
st.dataframe(out, use_container_width=True, hide_index=True)


# --- Synthèse Performance
def synth(df,t):
    df=df[df["Type"]==t]
    if df.empty: return 0,0
    val=df["Valeur (€)"].sum(); gain=df["Gain (€)"].sum()
    pct=(gain/(val-gain)*100) if val-gain!=0 else 0
    return gain,pct

pea_gain,pea_pct=synth(out,"PEA")
cto_gain,cto_pct=synth(out,"CTO")
tot_gain=out["Gain (€)"].sum()
tot_val=out["Valeur (€)"].sum()
tot_pct=(tot_gain/(tot_val-tot_gain)*100) if tot_val!=0 else 0

st.markdown(f"""
### 📊 Synthèse de performance
**PEA** : {pea_gain:+.2f} € ({pea_pct:+.2f}%)  
**CTO** : {cto_gain:+.2f} € ({cto_pct:+.2f}%)  
**Total** : {tot_gain:+.2f} € ({tot_pct:+.2f}%)  
""")

st.divider()


# --- Benchmark Comparatif
st.subheader(f"📈 Portefeuille vs {bench_name} ({periode})")

hist_graph = fetch_prices(tickers + [bench], days=days)
if "Date" not in hist_graph.columns:
    st.caption("Pas assez d'historique.")
else:
    df_val=[]
    for _,r in edited.iterrows():
        d=hist_graph[hist_graph["Ticker"]==r["Ticker"]].copy()
        if d.empty: continue
        d["Valeur"]=d["Close"]*r["Qty"]; d["Type"]=r["Type"]
        df_val.append(d[["Date","Valeur","Type"]])

    D=pd.concat(df_val)
    agg=D.groupby(["Date","Type"]).sum().reset_index()
    tot=agg.groupby("Date")["Valeur"].sum().reset_index().assign(Type="Total")

    bmk = hist_graph[hist_graph["Ticker"]==bench].copy()
    base_val=float(tot["Valeur"].iloc[0])
    bmk=bmk.assign(Type=bench_name, Valeur=bmk["Close"]/bmk["Close"].iloc[0]*base_val)

    full=pd.concat([agg,tot,bmk])
    base=full.groupby("Type").apply(lambda g:g.assign(Pct=(g["Valeur"]/g["Valeur"].iloc[0]-1)*100)).reset_index(drop=True)

    def perf(t):
        try: return base[base["Type"]==t]["Pct"].iloc[-1]
        except: return np.nan

    perf_total, perf_pea, perf_cto, perf_bmk = perf("Total"), perf("PEA"), perf("CTO"), perf(bench_name)

    def msg(n,p):
        if np.isnan(p) or np.isnan(perf_bmk): return ""
        d=p-perf_bmk
        return f"✅ **{n} surperforme** {bench_name} de **{d:+.2f}%**." if d>0 \
               else f"⚠️ **{n} sous-performe** {bench_name} de **{abs(d):.2f}%**."

    st.markdown(msg("Portefeuille TOTAL", perf_total))
    st.markdown(msg("PEA", perf_pea))
    st.markdown(msg("CTO", perf_cto))

    chart=alt.Chart(base).mark_line().encode(
        x="Date:T", y="Pct:Q", color="Type:N", tooltip=["Date:T","Type:N","Pct:Q"]
    )
    st.altair_chart(chart, use_container_width=True)


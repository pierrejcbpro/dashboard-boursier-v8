# -*- coding: utf-8 -*-
"""
v7.8 — Mon Portefeuille IA stricte & Conseils de Vente
- Structure identique à la V6.9
- IA stricte maintenue (hold=True => décisions conservatrices)
- Ajout colonne "Signal Vente 💰" + Priorité d'action
- Styles sécurisés (zéro crash pandas/streamlit)
"""

import os, json, numpy as np, pandas as pd, altair as alt, streamlit as st
from lib import (
    fetch_prices, compute_metrics, price_levels_from_row, decision_label_from_row,
    style_variations, company_name_from_ticker, get_profile_params, load_profile,
    resolve_identifier, find_ticker_by_name, load_mapping, save_mapping, maybe_guess_yahoo
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


# --- Boutons de gestion
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("💾 Sauvegarder"):
        pf.to_json(DATA_PATH, orient="records", indent=2, force_ascii=False)
        st.success("✅ Sauvegardé.")

with c2:
    if st.button("♻️ Réinitialiser"):
        os.remove(DATA_PATH)
        pd.DataFrame(columns=["Ticker","Type","Qty","PRU","Name"]).to_json(DATA_PATH, orient="records", indent=2)
        st.rerun()

with c3:
    st.download_button("⬇️ Exporter JSON", json.dumps(pf.to_dict(orient="records"), ensure_ascii=False, indent=2),
                       file_name="portfolio.json")

with c4:
    up = st.file_uploader("📥 Importer JSON", type=["json"], label_visibility="collapsed")
    if up:
        try:
            imp = pd.DataFrame(json.load(up))
            for c in ["Ticker","Type","Qty","PRU","Name"]:
                if c not in imp.columns:
                    imp[c] = "" if c in ("Ticker","Type","Name") else 0.0
            imp.to_json(DATA_PATH, orient="records", indent=2, force_ascii=False)
            st.rerun()
        except Exception as e:
            st.error(e)

st.divider()


# --- Tableau principal éditable
st.subheader("📝 Portefeuille")
edited = st.data_editor(
    pf, use_container_width=True, hide_index=True, num_rows="dynamic",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Type": st.column_config.SelectboxColumn("Type", options=["PEA","CTO"]),
        "Qty": st.column_config.NumberColumn("Qté"),
        "PRU": st.column_config.NumberColumn("PRU (€)", format="%.2f"),
        "Name": st.column_config.TextColumn("Nom"),
    }
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
    qty = r["Qty"]; pru = r["PRU"]
    name = r["Name"] or company_name_from_ticker(r["Ticker"])
    
    levels = price_levels_from_row(r, profil)
    dec = decision_label_from_row(r, held=True, vol_max=volmax)

    val = px * qty if np.isfinite(px) else np.nan
    gain = (px - pru) * qty if np.isfinite(px) and np.isfinite(pru) else np.nan
    perf = ((px/pru)-1)*100 if np.isfinite(px) and pru>0 else np.nan

    prox = ((px/levels["entry"])-1)*100 if np.isfinite(px) and np.isfinite(levels["entry"]) else np.nan
    emoji = "🟢" if abs(prox)<=2 else ("⚠️" if abs(prox)<=5 else "🔴")

    # 🧠 Nouvel indicateur Vente
    if pd.isna(perf):
        conseil = "—"
    elif perf > 8:
        conseil = "💰 Vendre partiellement"
    elif perf < -8:
        conseil = "🛑 Réduire / Réévaluer"
    else:
        conseil = "⏳ Continuer"

    rows.append({
        "Nom": name,
        "Ticker": r["Ticker"],
        "Type": r["Type"],
        "Cours (€)": round(px,2) if np.isfinite(px) else None,
        "Qté": qty,
        "PRU (€)": round(pru,2) if np.isfinite(pru) else None,
        "Valeur (€)": round(val,2) if np.isfinite(val) else None,
        "Gain (€)": round(gain,2) if np.isfinite(gain) else None,
        "Perf%": round(perf,2) if np.isfinite(perf) else None,
        "Entrée (€)": levels["entry"],
        "Objectif (€)": levels["target"],
        "Stop (€)": levels["stop"],
        "Décision IA": dec,
        "Proximité (%)": round(prox,2) if np.isfinite(prox) else None,
        "Signal Entrée": emoji,
        "Signal Vente 💰": conseil
    })

out = pd.DataFrame(rows)

# --- Styles SÉCURISÉS (plus jamais d'erreur applymap)
def sty_dec(v):
    if "Acheter" in str(v): return "background-color:rgba(0,180,0,0.18);font-weight:600;"
    if "Vendre" in str(v): return "background-color:rgba(255,0,0,0.18);font-weight:600;"
    if "Surveiller" in str(v): return "background-color:rgba(0,90,255,0.18);font-weight:600;"
    return ""

def sty_prox(v):
    if pd.isna(v): return ""
    if abs(v)<=2: return "background-color:#e8f5e9;color:#0b8043;font-weight:600;"
    if abs(v)<=5: return "background-color:#fff8e1;color:#a67c00;"
    return "background-color:#ffebee;color:#b71c1c;"

styled = (
    out.style
    .applymap(sty_dec, subset=["Décision IA"])
    .applymap(sty_prox, subset=["Proximité (%)"])
)

st.dataframe(styled, use_container_width=True, hide_index=True)

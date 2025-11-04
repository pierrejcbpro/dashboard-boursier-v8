# -*- coding: utf-8 -*-
"""
v7.8 — Mon Portefeuille IA stricte avec signal de vente 💰
---------------------------------------------------------
- Reprend intégralement la structure de ta version 7.6
- Ajoute une colonne "Signal Vente 💰" (IA + Objectif + Stop)
- Affiche une alerte automatique en haut de page quand des ventes / stops sont détectés
"""

import os, json, numpy as np, pandas as pd, altair as alt, streamlit as st
from lib import (
    fetch_prices, compute_metrics, price_levels_from_row, decision_label_from_row,
    style_variations, company_name_from_ticker, get_profile_params, load_profile,
    resolve_identifier, find_ticker_by_name, load_mapping, save_mapping, maybe_guess_yahoo
)

# --- CONFIG -------------------------------------------------------------------
st.set_page_config(page_title="Mon Portefeuille", page_icon="💼", layout="wide")
st.title("💼 Mon Portefeuille — IA stricte avec signal de vente 💰")

# --- PARAMÈTRES UTILISATEUR ---------------------------------------------------
periode = st.sidebar.radio("Période (graphique)", ["1 jour", "7 jours", "30 jours"], index=0)
days_map = {"1 jour": 2, "7 jours": 10, "30 jours": 35}
days_hist = days_map[periode]

benchmark_label = st.sidebar.selectbox(
    "Indice de référence (benchmark)",
    ["CAC 40", "DAX", "S&P 500", "NASDAQ 100"],
    index=0
)
benchmark_tickers = {"CAC 40": "^FCHI", "DAX": "^GDAXI", "S&P 500": "^GSPC", "NASDAQ 100": "^NDX"}
benchmark_symbol = benchmark_tickers[benchmark_label]

# --- CHARGEMENT DU PORTEFEUILLE -----------------------------------------------
DATA_PATH = "data/portfolio.json"
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    pd.DataFrame(columns=["Ticker", "Type", "Qty", "PRU", "Name"]).to_json(
        DATA_PATH, orient="records", indent=2, force_ascii=False
    )

try:
    pf = pd.read_json(DATA_PATH)
except Exception:
    pf = pd.DataFrame(columns=["Ticker", "Type", "Qty", "PRU", "Name"])

# Normalisation
for c, default in [("Ticker", ""), ("Type", "PEA"), ("Qty", 0.0), ("PRU", 0.0), ("Name", "")]:
    if c not in pf.columns:
        pf[c] = default

# --- GESTION FICHIERS ---------------------------------------------------------
cols = st.columns(4)
with cols[0]:
    if st.button("💾 Sauvegarder"):
        pf.to_json(DATA_PATH, orient="records", indent=2, force_ascii=False)
        st.success("✅ Sauvegardé.")
with cols[1]:
    if st.button("🗑 Réinitialiser"):
        try: os.remove(DATA_PATH)
        except FileNotFoundError: pass
        pd.DataFrame(columns=["Ticker","Type","Qty","PRU","Name"]).to_json(DATA_PATH, orient="records", indent=2)
        st.success("♻️ Réinitialisé."); st.rerun()
with cols[2]:
    st.download_button(
        "⬇️ Exporter JSON",
        json.dumps(pf.to_dict(orient="records"), ensure_ascii=False, indent=2),
        file_name="portfolio.json", mime="application/json"
    )
with cols[3]:
    up = st.file_uploader("📥 Importer JSON", type=["json"], label_visibility="collapsed")
    if up:
        try:
            imp = pd.DataFrame(json.load(up))
            for c in ["Ticker","Type","Qty","PRU","Name"]:
                if c not in imp.columns:
                    imp[c] = "" if c in ("Ticker","Type","Name") else 0.0
            imp.to_json(DATA_PATH, orient="records", indent=2, force_ascii=False)
            st.success("✅ Importé."); st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")

st.divider()

# --- RECHERCHE ET AJOUT -------------------------------------------------------
with st.expander("🔎 Recherche par nom / ISIN / WKN / Ticker"):
    q = st.text_input("Nom ou identifiant", "")
    t = st.selectbox("Type", ["PEA", "CTO"])
    qty = st.number_input("Qté", min_value=0.0, step=1.0)
    if st.button("Rechercher"):
        if not q.strip():
            st.warning("Entre un terme.")
        else:
            sym, _ = resolve_identifier(q)
            if sym:
                st.session_state["search_res"] = [{"symbol": sym, "shortname": company_name_from_ticker(sym)}]
            else:
                st.session_state["search_res"] = find_ticker_by_name(q) or []
    res = st.session_state.get("search_res", [])
    if res:
        labels = [f"{r['symbol']} — {r.get('shortname','')}" for r in res]
        sel = st.selectbox("Résultats", labels)
        if st.button("➕ Ajouter"):
            i = labels.index(sel)
            sym = res[i]["symbol"]
            nm = res[i].get("shortname", sym)
            pf = pd.concat([pf, pd.DataFrame([{"Ticker": sym.upper(), "Type": t, "Qty": qty, "PRU": 0.0, "Name": nm}])], ignore_index=True)
            pf.to_json(DATA_PATH, orient="records", indent=2, force_ascii=False)
            st.success(f"Ajouté : {nm} ({sym})"); st.rerun()

st.divider()

# --- TABLEAU D'ÉDITION --------------------------------------------------------
st.subheader("📝 Mon Portefeuille IA")

edited = st.data_editor(
    pf, num_rows="dynamic", use_container_width=True, hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Type": st.column_config.SelectboxColumn("Type", options=["PEA","CTO"]),
        "Qty": st.column_config.NumberColumn("Qté", format="%.2f"),
        "PRU": st.column_config.NumberColumn("PRU (€)", format="%.2f"),
        "Name": st.column_config.TextColumn("Nom"),
    }
)

if edited.empty:
    st.info("Ajoute une action pour commencer."); st.stop()

# --- ANALYSE IA ---------------------------------------------------------------
tickers = edited["Ticker"].dropna().unique().tolist()
hist_full = fetch_prices(tickers, days=240)
met = compute_metrics(hist_full)
merged = edited.merge(met, on="Ticker", how="left")

profil = load_profile()
volmax = get_profile_params(profil)["vol_max"]

rows = []
for _, r in merged.iterrows():
    px = float(r.get("Close", np.nan))
    qty = float(r.get("Qty", 0))
    pru = float(r.get("PRU", np.nan))
    name = r.get("Name") or company_name_from_ticker(r.get("Ticker"))
    levels = price_levels_from_row(r, profil)
    val = px * qty if np.isfinite(px) else np.nan
    gain_eur = (px - pru) * qty if np.isfinite(px) and np.isfinite(pru) else np.nan
    perf = ((px / pru) - 1) * 100 if (np.isfinite(px) and np.isfinite(pru) and pru > 0) else np.nan
    dec = decision_label_from_row(r, held=True, vol_max=volmax)

    # Volatilité
    ma20, ma50 = float(r.get("MA20", np.nan)), float(r.get("MA50", np.nan))
    vola = abs(ma20 - ma50) / ma50 * 100 if np.isfinite(ma20) and np.isfinite(ma50) and ma50 != 0 else np.nan
    vol_ind = "🟢" if vola < 2 else ("🟡" if vola < 5 else "🔴")

    # Tendance LT
    ma120, ma240 = float(r.get("MA120", np.nan)), float(r.get("MA240", np.nan))
    trend_icon = "🌱" if ma120 > ma240 else ("🌧" if ma120 < ma240 else "⚖️")

    # Score IA combiné
    score_ia = 100 - min((abs(ma20 - ma50) + abs(ma120 - ma240)) * 10, 100)

    rows.append({
        "Type": r["Type"], "Nom": name, "Ticker": r["Ticker"],
        "Cours (€)": round(px, 2) if np.isfinite(px) else None,
        "Qté": qty, "PRU (€)": round(pru, 2) if np.isfinite(pru) else None,
        "Valeur (€)": round(val, 2) if np.isfinite(val) else None,
        "Gain/Perte (€)": round(gain_eur, 2) if np.isfinite(gain_eur) else None,
        "Perf%": round(perf, 2) if np.isfinite(perf) else None,
        "Volatilité": vol_ind, "Tendance LT": trend_icon,
        "Score IA": round(score_ia, 1),
        "Entrée (€)": levels["entry"], "Objectif (€)": levels["target"], "Stop (€)": levels["stop"],
        "Décision IA": dec
    })

out = pd.DataFrame(rows)

# --- SIGNAL DE VENTE 💰 -------------------------------------------------------
def signal_vente(row):
    dec = str(row.get("Décision IA", ""))
    perf = row.get("Perf%", 0)
    px = row.get("Cours (€)", np.nan)
    tgt = row.get("Objectif (€)", np.nan)
    stp = row.get("Stop (€)", np.nan)

    if "Vendre" in dec or (np.isfinite(tgt) and px >= tgt):
        return "💰 Prendre bénéfice"
    elif "Surveiller" in dec or (perf > 0 and perf < 5):
        return "👁️ Surveiller"
    elif "Acheter" in dec:
        return "🟢 Opportunité / Renforcer"
    elif np.isfinite(stp) and px <= stp:
        return "❌ Stop conseillé"
    return "—"

out["Signal Vente 💰"] = out.apply(signal_vente, axis=1)

# --- ALERTE AUTOMATIQUE -------------------------------------------------------
vente_rows = out[out["Signal Vente 💰"].isin(["💰 Prendre bénéfice", "❌ Stop conseillé"])]
if not vente_rows.empty:
    st.warning(f"⚠️ {len(vente_rows)} signal(s) détecté(s) : **{', '.join(vente_rows['Nom'].head(5))}**")

# --- STYLES -------------------------------------------------------------------
def style_signal(val):
    if "💰" in val: return "background-color:#e8f5e9; color:#0b8043; font-weight:600;"
    if "👁️" in val: return "background-color:#fff8e1; color:#a67c00;"
    if "❌" in val: return "background-color:#ffebee; color:#b71c1c;"
    if "🟢" in val: return "background-color:#e3f2fd; color:#01579b;"
    return ""

st.dataframe(
    out.style
        .applymap(style_signal, subset=["Signal Vente 💰"])
        .applymap(style_variations, subset=["Perf%"]),
    use_container_width=True, hide_index=True
)

# --- SYNTHÈSE ---------------------------------------------------------------
st.markdown(f"### 📊 Synthèse {periode}")
tot_gain = out["Gain/Perte (€)"].sum()
tot_val = out["Valeur (€)"].sum()
pct = (tot_gain / (tot_val - tot_gain) * 100) if tot_val > 0 else 0
st.markdown(f"**Gain total : {tot_gain:+.2f} € ({pct:+.2f}%)** — Score IA moyen : {out['Score IA'].mean():.1f}/100")

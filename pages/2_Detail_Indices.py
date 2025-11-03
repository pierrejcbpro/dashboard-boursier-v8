# -*- coding: utf-8 -*-
"""
v7.6 — Détails Indice IA enrichi
Basé sur ta version stable v6.9 :
- Structure et interface 100 % identiques
- Ajout des moyennes long terme (MA120 / MA240)
- Calcul et affichage du Score IA combiné (court + long terme)
- Ajout de la tendance LT 🌱 / 🌧 / ⚖️
- Compatible avec lib v7.6
"""

import streamlit as st, pandas as pd, numpy as np, altair as alt
from lib import (
    fetch_all_markets, price_levels_from_row, decision_label_from_row,
    style_variations, get_profile_params, load_profile
)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Détails Indice", page_icon="📊", layout="wide")
st.title("📊 Détails Indice — Analyse IA complète")

# ---------------- CHOIX INDICE ----------------
indice = st.sidebar.selectbox(
    "Choisis un indice",
    ["CAC 40", "DAX", "S&P 500", "NASDAQ 100"],
    index=0
)

periode = st.sidebar.radio("Période d’analyse", ["Jour", "7 jours", "30 jours"], index=1)
value_col = {"Jour": "pct_1d", "7 jours": "pct_7d", "30 jours": "pct_30d"}[periode]

profil = load_profile()
st.sidebar.markdown(f"**Profil IA actif :** {profil}")

st.divider()

# ---------------- DONNÉES ----------------
data = fetch_all_markets([(indice, None)], days_hist=240)
if data.empty:
    st.warning("Aucune donnée disponible (vérifie la connectivité).")
    st.stop()

merged = data.copy()

# ---------------- ANALYSE GLOBALE ----------------
avg = merged[value_col].mean() * 100
disp = merged[value_col].std() * 100
st.markdown(f"### 🧭 Vue d’ensemble — {indice} ({periode})")
st.markdown(f"**Variation moyenne : {avg:+.2f}%** — **Dispersion : {disp:.2f}%**")

if disp < 1:
    st.caption("Marché homogène, faible volatilité.")
elif disp < 3:
    st.caption("Marché équilibré, rotations sectorielles modérées.")
else:
    st.caption("Marché dispersé, forte volatilité intertitres.")

st.divider()

# ---------------- CLASSEMENT IA ----------------
rows = []
volmax = get_profile_params(profil)["vol_max"]
for _, r in merged.iterrows():
    levels = price_levels_from_row(r, profil)
    dec = decision_label_from_row(r, held=False, vol_max=volmax)
    entry, target, stop = levels["entry"], levels["target"], levels["stop"]
    px = r.get("Close", np.nan)
    prox = ((px / entry) - 1) * 100 if np.isfinite(px) and np.isfinite(entry) and entry > 0 else np.nan
    emoji = "🟢" if abs(prox) <= 2 else ("⚠️" if abs(prox) <= 5 else "🔴")

    # Indicateur long terme 🌱 / 🌧 / ⚖️ basé sur MA120/MA240
    ma120, ma240 = r.get("MA120", np.nan), r.get("MA240", np.nan)
    trend_lt = np.nan
    if np.isfinite(ma120) and np.isfinite(ma240):
        trend_lt = 1 if ma120 > ma240 else (-1 if ma120 < ma240 else 0)
    lt_icon = "🌱" if trend_lt > 0 else ("🌧" if trend_lt < 0 else "⚖️")

    # Score IA combiné (court + long terme)
    gap50 = abs(r.get("MA20", np.nan) - r.get("MA50", np.nan))
    gap240 = abs(ma120 - ma240) if np.isfinite(ma120) and np.isfinite(ma240) else np.nan
    score_ia = np.nan
    if np.isfinite(gap50) and np.isfinite(gap240):
        score_ia = 100 - min((gap50 + gap240) * 10, 100)

    rows.append({
        "Société": r.get("name", ""),
        "Ticker": r["Ticker"],
        "Cours (€)": round(px, 2) if np.isfinite(px) else None,
        "Variation (%)": round(r[value_col] * 100, 2) if np.isfinite(r[value_col]) else None,
        "Entrée (€)": entry,
        "Objectif (€)": target,
        "Stop (€)": stop,
        "Décision IA": dec,
        "Proximité (%)": round(prox, 2) if np.isfinite(prox) else np.nan,
        "Signal": emoji,
        "Tendance LT": lt_icon,
        "Score IA": round(score_ia, 1) if np.isfinite(score_ia) else np.nan
    })

out = pd.DataFrame(rows)
if out.empty:
    st.info("Aucune donnée exploitable pour cet indice.")
    st.stop()

# Tri : Acheter > Surveiller > Vendre, puis par proximité
def sort_key(val):
    if "Acheter" in val: return 0
    if "Surveiller" in val: return 1
    if "Vendre" in val: return 2
    return 3
out["sort"] = out["Décision IA"].apply(sort_key)
out = out.sort_values(["sort", "Proximité (%)"], ascending=[True, True]).drop(columns="sort")

# ---------------- TABLEAU PRINCIPAL ----------------
def color_decision(v):
    if pd.isna(v): return ""
    if "Acheter" in v: return "background-color: rgba(0,200,0,0.15);"
    if "Vendre" in v: return "background-color: rgba(255,0,0,0.15);"
    if "Surveiller" in v: return "background-color: rgba(0,100,255,0.15);"
    return ""

def color_proximity(v):
    if pd.isna(v): return ""
    if abs(v) <= 2: return "background-color: rgba(0,200,0,0.10); color:#0b8043"
    if abs(v) <= 5: return "background-color: rgba(255,200,0,0.15); color:#a67c00"
    return "background-color: rgba(255,0,0,0.12); color:#b71c1c"

st.subheader("🚦 Classement IA des actions")
st.dataframe(
    out.style
        .applymap(color_decision, subset=["Décision IA"])
        .applymap(color_proximity, subset=["Proximité (%)"]),
    use_container_width=True, hide_index=True
)

# ---------------- GRAPHIQUES ----------------
st.divider()
st.subheader("📈 Distribution IA — Synthèse visuelle")

col1, col2 = st.columns(2)
with col1:
    chart = alt.Chart(out).mark_bar().encode(
        x=alt.X("Décision IA:N", sort=["Acheter","Surveiller","Vendre"], title="Décision IA"),
        y=alt.Y("count():Q", title="Nombre de valeurs"),
        color=alt.Color("Décision IA:N", legend=None)
    ).properties(height=320, title="Répartition des décisions IA")
    st.altair_chart(chart, use_container_width=True)

with col2:
    chart2 = alt.Chart(out).mark_bar().encode(
        x=alt.X("Société:N", sort="-y", title=""),
        y=alt.Y("Variation (%):Q", title="Perf (%)"),
        color=alt.Color("Variation (%):Q", scale=alt.Scale(scheme="redyellowgreen")),
        tooltip=["Société","Ticker","Variation (%)","Tendance LT","Score IA"]
    ).properties(height=320, title=f"Tendances — {periode}")
    st.altair_chart(chart2, use_container_width=True)

# ---------------- CONCLUSION ----------------
st.divider()
st.markdown(f"""
### 🧠 Synthèse IA {indice}
- Profil IA actif : **{profil}**
- Actions **🟢 proches de l’entrée idéale** : { (out['Signal'] == '🟢').sum() }
- Actions **⚠️ modérément proches** : { (out['Signal'] == '⚠️').sum() }
- Actions **🔴 éloignées** : { (out['Signal'] == '🔴').sum() }
- Moyenne du **Score IA** : {out['Score IA'].mean():.1f}/100
""")

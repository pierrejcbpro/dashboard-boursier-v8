# -*- coding: utf-8 -*-
"""
v7.8 — Synthèse Flash IA (interactive)
Base: v7.7 qui te convenait, avec correctifs mineurs :
- ✅ Pas de KeyError si colonnes manquantes (tolérance)
- ✅ 'Ticker' toujours présent (fallback 'Symbole')
- ✅ Ajout au suivi virtuel SÉLECTIF via cases à cocher (pas de session_state piégeux)
- ✅ JSON propre (list[dict]) + dossier data auto
"""

import os, json
import streamlit as st, pandas as pd, numpy as np, altair as alt
from lib import (
    fetch_all_markets, style_variations, load_profile, save_profile,
    news_summary, select_top_actions
)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Synthèse Flash IA", page_icon="⚡", layout="wide")
st.title("⚡ Synthèse Flash — Marché Global (IA enrichie)")

# ---------------- Sidebar ----------------
periode = st.sidebar.radio("Période d’analyse", ["Jour","7 jours","30 jours"], index=0)
value_col = {"Jour": "pct_1d", "7 jours": "pct_7d", "30 jours": "pct_30d"}[periode]

profil = st.sidebar.radio(
    "Profil IA", ["Prudent", "Neutre", "Agressif"],
    index=["Prudent", "Neutre", "Agressif"].index(load_profile())
)
if st.sidebar.button("💾 Mémoriser le profil"):
    save_profile(profil)
    st.sidebar.success("Profil sauvegardé.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Marchés inclus")
include_eu = st.sidebar.checkbox("🇫🇷 CAC 40 + 🇩🇪 DAX", value=True)
include_us = st.sidebar.checkbox("🇺🇸 NASDAQ 100 + S&P 500", value=False)
include_ls = st.sidebar.checkbox("🧠 LS Exchange (perso)", value=False)

# ---------------- Données marchés ----------------
MARKETS = []
if include_eu: MARKETS += [("CAC 40", None), ("DAX", None)]
if include_us: MARKETS += [("NASDAQ 100", None), ("S&P 500", None)]
if include_ls: MARKETS += [("LS Exchange", None)]

if not MARKETS:
    st.warning("Aucun marché sélectionné. Active au moins un marché dans la barre latérale.")
    st.stop()

data = fetch_all_markets(MARKETS, days_hist=240)
if data.empty:
    st.warning("Aucune donnée disponible (vérifie la connectivité ou ta sélection de marchés).")
    st.stop()

# Tolérance colonnes
for c in ["pct_1d", "pct_7d", "pct_30d", "Close", "Ticker", "name", "Indice", "MA120", "MA240", "lt_trend_score"]:
    if c not in data.columns:
        data[c] = np.nan

# LT 🌱 / 🌧 / ⚖️
def lt_icon(row):
    ma120 = row.get("MA120", np.nan)
    ma240 = row.get("MA240", np.nan)
    if np.isfinite(ma120) and np.isfinite(ma240):
        if ma120 > ma240: return "🌱"
        if ma120 < ma240: return "🌧"
        return "⚖️"
    v = row.get("lt_trend_score", np.nan)
    if np.isfinite(v):
        return "🌱" if v > 0 else ("🌧" if v < 0 else "⚖️")
    return "⚪"

valid = data.dropna(subset=["Close"]).copy()
valid["LT"] = valid.apply(lt_icon, axis=1)

# IA Score local si manquant
if "IA_Score" not in valid.columns:
    for c in ["trend_score", "lt_trend_score", "pct_7d", "pct_30d", "ATR14"]:
        if c not in valid.columns: valid[c] = np.nan
    valid["Volatilité"] = valid["ATR14"] / valid["Close"]
    valid["IA_Score"] = (
        valid["lt_trend_score"].fillna(0)*60
        + valid["trend_score"].fillna(0)*40
        + valid["pct_30d"].fillna(0)*100
        + valid["pct_7d"].fillna(0)*50
        - valid["Volatilité"].fillna(0)*10
    )

# ---------------- Résumé global ----------------
avg = (valid[value_col].dropna().mean() * 100.0) if not valid.empty else np.nan
up = int((valid[value_col] > 0).sum())
down = int((valid[value_col] < 0).sum())

st.markdown(f"### 🧭 Résumé global ({periode})")
if np.isfinite(avg):
    st.markdown(f"**Variation moyenne : {avg:+.2f}%** — {up} hausses / {down} baisses")
else:
    st.markdown("Variation indisponible pour cette période.")

disp = (valid[value_col].std() * 100.0) if not valid.empty else np.nan
if np.isfinite(disp):
    if disp < 1.0:
        st.caption("Marché calme — consolidation technique.")
    elif disp < 2.5:
        st.caption("Volatilité modérée — quelques leaders sectoriels.")
    else:
        st.caption("Marché dispersé — forte rotation / flux macro.")

st.divider()

# ---------------- TOP / FLOP ----------------
st.subheader(f"🏆 Top 10 hausses & ⛔ Baisses — {periode}")

def prep_table(df, asc=False, n=10):
    if df.empty: return pd.DataFrame()
    cols = ["Ticker", "name", "Close", value_col, "Indice", "IA_Score", "LT"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    out = df.sort_values(value_col, ascending=asc).head(n).copy()
    out.rename(columns={"name": "Société", "Close": "Cours (€)"}, inplace=True)
    out["Variation %"] = (out[value_col] * 100).round(2)
    out["Cours (€)"] = out["Cours (€)"].round(2)
    return out[["Indice", "Société", "Ticker", "Cours (€)", "Variation %", "LT", "IA_Score"]]

col1, col2 = st.columns(2)
with col1:
    top = prep_table(valid, asc=False, n=10)
    st.dataframe(style_variations(top, ["Variation %"]), use_container_width=True, hide_index=True)
with col2:
    flop = prep_table(valid, asc=True, n=10)
    st.dataframe(style_variations(flop, ["Variation %"]), use_container_width=True, hide_index=True)

st.divider()

# ---------------- SÉLECTION IA (Top 10) ----------------
st.subheader("🚀 Sélection IA — Opportunités idéales (TOP 10)")
top_actions = select_top_actions(valid, profile=profil, n=10, include_proximity=True)

if top_actions.empty:
    st.info("Aucune opportunité IA détectée aujourd’hui selon ton profil.")
else:
    # Harmonise nom des colonnes (Ticker présent même si lib retourne 'Symbole')
    if "Ticker" not in top_actions.columns and "Symbole" in top_actions.columns:
        top_actions["Ticker"] = top_actions["Symbole"]

    def proximity_marker(v):
        if pd.isna(v): return "⚪"
        if abs(v) <= 2: return "🟢"
        elif abs(v) <= 5: return "⚠️"
        else: return "🔴"
    if "Proximité (%)" in top_actions.columns:
        top_actions["Signal Entrée"] = top_actions["Proximité (%)"].apply(proximity_marker)

    def style_prox(v):
        if pd.isna(v): return ""
        if abs(v) <= 2:  return "background-color:#e8f5e9; color:#0b8043; font-weight:600;"
        if abs(v) <= 5:  return "background-color:#fff8e1; color:#a67c00;"
        return "background-color:#ffebee; color:#b71c1c;"

    show_cols = []
    for c in ["Société","name","Ticker","Cours (€)","Entrée (€)","Objectif (€)","Stop (€)","Proximité (%)","Signal Entrée","IA_Score","Trend ST","Trend LT","MA20","MA50","MA120","MA240","Signal","Indice"]:
        if c in top_actions.columns:
            show_cols.append(c)
    # alias 'name' -> 'Société' si besoin, sans casser le style
    show = top_actions.copy()
    if "Société" not in show.columns and "name" in show.columns:
        show.rename(columns={"name":"Société"}, inplace=True)

    styled = show[show_cols].style
    if "Proximité (%)" in show.columns:
        styled = styled.applymap(style_prox, subset=["Proximité (%)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ---------------- Injection IA interactive ----------------
st.divider()
st.subheader("💸 Injection IA — Simulateur micro-investissement")

st.caption("Analyse IA pour des tickets 7–30 jours avec frais inclus (1€ entrée + 1€ sortie).")

invest_amount = st.number_input("💰 Montant d’investissement par action (€)", min_value=5.0, max_value=500.0, step=5.0, value=40.0)
fee_in = 1.0
fee_out = 1.0

# Base IA (pré-remplissage)
rows = []
if not top_actions.empty:
    for _, r in top_actions.head(15).iterrows():
        entry = float(r.get("Entrée (€)", np.nan))
        target = float(r.get("Objectif (€)", np.nan))
        stop = float(r.get("Stop (€)", np.nan))
        score = float(r.get("IA_Score", 50))
        name = r.get("Société") or r.get("name")
        tkr = r.get("Ticker") or r.get("Symbole")
        if not np.isfinite(entry) or not np.isfinite(target) or entry <= 0:
            continue
        # prix d’achat “effectif” avec frais d’entrée dilués
        shares = invest_amount / (entry + fee_in / max(shares:= (invest_amount/entry), 1e-8))  # robustesse
        buy_price = invest_amount / shares
        brut_gain = (target - buy_price) * shares
        net_gain = brut_gain - fee_out
        net_return_pct = (net_gain / invest_amount) * 100
        rows.append({
            "Ajouter": False,
            "Société": name,
            "Ticker": tkr,
            "Entrée (€)": round(entry, 2),
            "Objectif (€)": round(target, 2),
            "Stop (€)": round(stop, 2),
            "Score IA": round(score, 1),
            "Durée visée": "7–30 j",
            "Rendement net estimé (%)": round(net_return_pct, 2)
        })

df_inject = pd.DataFrame(rows)
if df_inject.empty:
    df_inject = pd.DataFrame(columns=["Ajouter","Société","Ticker","Entrée (€)","Objectif (€)","Stop (€)","Score IA","Durée visée","Rendement net estimé (%)"])

# Éditeur interactif (cases à cocher pour ajout sélectif)
edited = st.data_editor(
    df_inject,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    key="micro_invest_editor",
    column_config={
        "Ajouter": st.column_config.CheckboxColumn("Ajouter"),
        "Société": st.column_config.TextColumn("Société"),
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Entrée (€)": st.column_config.NumberColumn("Entrée (€)", format="%.2f"),
        "Objectif (€)": st.column_config.NumberColumn("Objectif (€)", format="%.2f"),
        "Stop (€)": st.column_config.NumberColumn("Stop (€)", format="%.2f"),
        "Score IA": st.column_config.NumberColumn("Score IA", format="%.1f"),
        "Durée visée": st.column_config.SelectboxColumn("Durée visée", options=["7–30 j", "<7 j", "1–3 mois"]),
        "Rendement net estimé (%)": st.column_config.NumberColumn("Rendement net estimé (%)", format="%.2f"),
    },
)

# Recalcule le rendement net estimé selon le montant saisi
def recompute_returns(df, invest_amount, fee_in, fee_out):
    out = df.copy()
    res = []
    for _, r in out.iterrows():
        entry = float(r.get("Entrée (€)", np.nan))
        target = float(r.get("Objectif (€)", np.nan))
        if not np.isfinite(entry) or not np.isfinite(target) or entry <= 0:
            res.append(np.nan); continue
        # dilution frais entrée + frais sortie
        shares_approx = invest_amount / max(entry, 1e-8)
        buy_price = entry + fee_in / max(shares_approx, 1e-8)
        shares = invest_amount / buy_price
        brut_gain = (target - buy_price) * shares
        net_gain = brut_gain - fee_out
        res.append(round((net_gain / invest_amount) * 100, 2))
    out["Rendement net estimé (%)"] = res
    return out

if not edited.empty:
    edited = recompute_returns(edited, invest_amount, fee_in, fee_out)

    def style_gain(v):
        if pd.isna(v): return ""
        if v > 5: return "background-color:#e8f5e9; color:#0b8043; font-weight:600;"
        if v > 0: return "background-color:#fff8e1; color:#a67c00;"
        return "background-color:#ffebee; color:#b71c1c;"

    styled = edited.style.applymap(style_gain, subset=["Rendement net estimé (%)"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if edited["Rendement net estimé (%)"].notna().any():
        best = edited.loc[edited["Rendement net estimé (%)"].idxmax()]
        st.success(
            f"💡 **Idée optimale : {best.get('Société','?')} ({best.get('Ticker','?')})** — "
            f"rendement net estimé **{best.get('Rendement net estimé (%)',0):+.2f}%** "
            f"pour un ticket de **{invest_amount:.0f} €** sur {best.get('Durée visée','7–30 j')}."
        )
else:
    st.caption("Ajoute une ou plusieurs lignes ci-dessus pour simuler ton investissement.")

# --- Ajout au suivi virtuel (sélectif via 'Ajouter' = True)
save_path = "data/suivi_virtuel.json"
os.makedirs("data", exist_ok=True)

if st.button("💹 ➕ Ajouter la sélection au suivi virtuel"):
    try:
        to_add = edited[edited.get("Ajouter", False) == True].copy() if not edited.empty else pd.DataFrame()
        if to_add.empty:
            st.warning("Aucune ligne cochée dans la colonne “Ajouter”.")
        else:
            # Nettoyage + sérialisation
            export_cols = ["Société","Ticker","Entrée (€)","Objectif (€)","Stop (€)","Score IA","Durée visée","Rendement net estimé (%)"]
            for c in export_cols:
                if c not in to_add.columns: to_add[c] = None
            new_items = to_add[export_cols].to_dict(orient="records")

            # Charge JSON existant (liste)
            try:
                if os.path.exists(save_path):
                    with open(save_path, "r", encoding="utf-8") as f:
                        cur = json.load(f)
                        if not isinstance(cur, list): cur = []
                else:
                    cur = []
            except Exception:
                cur = []

            # Ajoute & dédoublonne sur (Ticker, Entrée)
            cur.extend(new_items)
            seen = set()
            dedup = []
            for it in cur:
                key = (str(it.get("Ticker")), str(it.get("Entrée (€)")))
                if key in seen: continue
                seen.add(key); dedup.append(it)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(dedup, f, ensure_ascii=False, indent=2)
            st.success(f"💾 {len(new_items)} ligne(s) ajoutée(s) au suivi virtuel.")
    except Exception as e:
        st.error(f"Erreur lors de l’ajout : {e}")

# ---------------- Charts ----------------
st.divider()
st.markdown("### 📊 Visualisation rapide")
def bar_chart(df, title):
    if df.empty:
        st.caption("—"); return
    d = df.copy()
    # Assure les colonnes attendues
    for c in ["Société","Ticker","Variation %","Cours (€)","Indice","LT","IA_Score"]:
        if c not in d.columns: d[c] = np.nan
    d["Label"] = d["Société"].astype(str) + " (" + d["Ticker"].astype(str) + ")"
    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("Label:N", sort="-y", title=""),
            y=alt.Y("Variation %:Q", title="Variation (%)"),
            color=alt.Color("Variation %:Q", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["Société","Ticker","Variation %","Cours (€)","Indice","LT","IA_Score"]
        )
        .properties(height=320, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

col3, col4 = st.columns(2)
with col3: bar_chart(top, f"Top 10 hausses ({periode})")
with col4: bar_chart(flop, f"Top 10 baisses ({periode})")

# ---------------- Actualités ----------------
st.markdown("### 📰 Actualités principales")
def short_news(row):
    nm = str(row.get("Société") or row.get("name") or "")
    tk = str(row.get("Ticker") or "")
    txt, score, items = news_summary(nm, tk, lang="fr")
    return txt

if not top.empty:
    st.markdown("**Top hausses — explication probable :**")
    for _, r in top.iterrows():
        st.markdown(f"- **{r['Société']} ({r['Ticker']})** : {short_news(r)}")
if not flop.empty:
    st.markdown("**Baisses — explication probable :**")
    for _, r in flop.iterrows():
        st.markdown(f"- **{r['Société']} ({r['Ticker']})** : {short_news(r)}")

st.divider()
st.caption("💡 Utilise la section d’injection IA pour simuler tes investissements rapides entre 7 et 30 jours.")

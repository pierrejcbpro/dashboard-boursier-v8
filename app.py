# -*- coding: utf-8 -*-
"""
Dash Boursier — v7.6 (base V6 + IA long terme)
"""
import streamlit as st
from lib import get_profile_params, load_profile, save_profile

st.set_page_config(
    page_title="Dash Boursier v7.6",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💹 Dash Boursier — v7.6")

# Profil IA (mémoire)
if "profil" not in st.session_state:
    st.session_state["profil"] = load_profile()

st.sidebar.title("🧭 Paramètres IA")
profil = st.sidebar.radio(
    "Profil d'investisseur",
    ["Prudent", "Neutre", "Agressif"],
    index=["Prudent","Neutre","Agressif"].index(st.session_state["profil"])
)
if profil != st.session_state["profil"]:
    st.session_state["profil"] = profil
    save_profile(profil)
    st.toast(f"Profil IA mis à jour → {profil}", icon="🤖")

params = get_profile_params(profil)

st.markdown("""
### ✅ Nouveautés clés
- **IA Long Terme** : MA120 / MA240, **Tendance LT** (🌱 / ⚖️ / 🌧) et **Score IA global**.
- **Décision combinée** CT+LT sur toutes les pages (Portefeuille, Synthèse, Recherche).
- **Proximité d’entrée** 🟢⚠️🔴 et **Potentiel (€)** conservés.
- **Design V6** intact (tableaux, boutons, expandeurs, graphiques).
""")

st.divider()
st.subheader("⚙️ Paramètres IA actuels")
c1,c2,c3 = st.columns(3)
with c1: st.metric("Profil", profil)
with c2: st.metric("Volatilité max", f"{params['vol_max']*100:.1f}%")
with c3: st.metric("Cible LT (MAs)", "MA120 / MA240")

st.info(
    f"🧠 **Mode IA : {profil}** — Décisions pondérées par la tendance **court terme (MA20/50)** et **long terme (MA120/240)**."
)

st.divider()
st.markdown("""
### 🚀 Navigation
- ⚡ **Synthèse Flash IA** — marché multi-indices, Top/Flop 10, Sélection IA TOP 10
- 🧩 **Détail par indice** — membres + signaux CT/LT
- 💼 **Mon Portefeuille** — PEA/CTO, décisions IA combinées, benchmark & camembert
- 🔍 **Recherche universelle** — Analyse complète + actualités + ajout direct au portefeuille
""")
st.success("✅ Choisis une page dans le menu à gauche.")

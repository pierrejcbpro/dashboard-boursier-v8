# -*- coding: utf-8 -*-
import os, json, math, re, html, requests
import numpy as np
import pandas as pd
import yfinance as yf
from functools import lru_cache
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# =========================
# FICHIERS & PRESETS
# =========================
DATA_DIR = "data"
MAPPING_PATH = os.path.join(DATA_DIR, "id_mapping.json")
WL_PATH = os.path.join(DATA_DIR, "watchlist_ls.json")          # ← LS Exchange perso
PROFILE_PATH = os.path.join(DATA_DIR, "profile.json")
LAST_SEARCH_PATH = os.path.join(DATA_DIR, "last_search.json")

os.makedirs(DATA_DIR, exist_ok=True)
_defaults = [
    (MAPPING_PATH, {}),
    (WL_PATH, []),
    (PROFILE_PATH, {"profil": "Neutre"}),
    (LAST_SEARCH_PATH, {"last": "TTE.PA"}),
]
for p, default in _defaults:
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(default, f)

UA = {"User-Agent": "Mozilla/5.0"}

# =========================
# SENTIMENT (VADER)
# =========================
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon")
    except Exception:
        pass

try:
    SIA = SentimentIntensityAnalyzer()
except Exception:
    SIA = None

# =========================
# PROFILS IA
# =========================
PROFILE_PARAMS = {
    "Agressif": {"vol_max": 0.08, "target_mult": 1.10, "stop_mult": 0.92, "entry_mult": 0.990},
    "Neutre":   {"vol_max": 0.05, "target_mult": 1.07, "stop_mult": 0.95, "entry_mult": 0.990},
    "Prudent":  {"vol_max": 0.03, "target_mult": 1.05, "stop_mult": 0.97, "entry_mult": 0.995},
}

def get_profile_params(profile: str):
    return PROFILE_PARAMS.get(profile or "Neutre", PROFILE_PARAMS["Neutre"])

def load_profile():
    try:
        return json.load(open(PROFILE_PATH, "r", encoding="utf-8")).get("profil", "Neutre")
    except Exception:
        return "Neutre"

def save_profile(p):
    try:
        json.dump({"profil": p}, open(PROFILE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_last_search():
    try:
        return json.load(open(LAST_SEARCH_PATH, "r", encoding="utf-8")).get("last", "")
    except Exception:
        return ""

def save_last_search(t):
    try:
        json.dump({"last": t}, open(LAST_SEARCH_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass

# =========================
# MAPPING / LS→YAHOO (optionnel)
# =========================
def load_mapping():
    try:
        return json.load(open(MAPPING_PATH, "r", encoding="utf-8"))
    except Exception:
        return {}

def save_mapping(m):
    json.dump(m, open(MAPPING_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def load_watchlist_ls():
    try:
        return json.load(open(WL_PATH, "r", encoding="utf-8"))
    except Exception:
        return []

def save_watchlist_ls(lst):
    json.dump(lst, open(WL_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def _norm(s): return (s or "").strip().upper()
_PARIS = {"AIR","ORA","MC","TTE","BNP","SGO","ENGI","SU","DG","ACA","GLE","RI","KER","HO","EN","CAP","AI","PUB","VIE","VIV","STM"}

def guess_yahoo_from_ls(ticker: str):
    """Heuristique simple si on tombe sur des symboles courts 'LS-like'."""
    if not ticker: return None
    t = _norm(ticker)
    if "." in t and not t.endswith(".LS"):   # déjà suffixé Yahoo
        return t
    if t.endswith(".LS"):                    # Londres
        return f"{t[:-3]}.L"
    if t == "TOTB":                          # exemple spécifique
        return "TOTB.F"
    if t.endswith("B") and not t.endswith("AB"):  # Heuristique DE
        return f"{t}.F"
    if t in _PARIS:
        return f"{t}.PA"
    if len(t) <= 6 and t.isalpha():
        return f"{t}.PA"
    return t

def maybe_guess_yahoo(s):
    s = _norm(s)
    m = load_mapping().get(s)
    return m or guess_yahoo_from_ls(s)

def resolve_identifier(id_or_ticker):
    raw = _norm(id_or_ticker)
    if not raw: return None, {}
    mapping = load_mapping()
    if raw in mapping:
        return mapping[raw], {"source": "mapping"}
    guess = maybe_guess_yahoo(raw)
    if guess:
        try:
            hist = yf.download(guess, period="5d", interval="1d",
                               auto_adjust=True, progress=False, threads=False)
            if not hist.empty:
                mapping[raw] = guess
                save_mapping(mapping)
                return guess, {"source": "heuristic"}
        except Exception:
            pass
    return None, {}

# =========================
# RECHERCHE YAHOO
# =========================
@lru_cache(maxsize=256)
def yahoo_search(query: str, region="FR", lang="fr-FR", quotesCount=20):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": quotesCount, "newsCount": 0, "lang": lang, "region": region}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=12)
        r.raise_for_status()
        data = r.json()
        quotes = data.get("quotes", [])
        out = []
        for q in quotes:
            out.append({
                "symbol": q.get("symbol"),
                "shortname": q.get("shortname") or q.get("longname") or "",
                "longname": q.get("longname") or q.get("shortname") or "",
                "exchDisp": q.get("exchDisp") or "",
                "typeDisp": q.get("typeDisp") or "",
            })
        return out
    except Exception:
        return []

def find_ticker_by_name(company_name: str, prefer_markets=("Paris","XETRA","Frankfurt","NasdaqGS","NYSE")):
    if not company_name: return []
    q = company_name.strip()
    res = yahoo_search(q)
    if not res: return []
    eq = [r for r in res if (r.get("typeDisp","").lower() in ("equity","action","stock","actions") or r.get("symbol",""))]
    ranked=[]
    for r in eq:
        score = 0
        exch = (r.get("exchDisp") or "").lower()
        name = (r.get("shortname") or r.get("longname") or "").lower()
        sym  = (r.get("symbol") or "").upper()
        if any(pm.lower() in exch for pm in prefer_markets): score += 3
        if q.lower() in name: score += 2
        if q.lower() in sym.lower(): score += 1
        ranked.append((score, r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked]

# =========================
# MEMBRES D’INDICES (CAC40, DAX, NASDAQ100, S&P500)
# =========================
@lru_cache(maxsize=32)
def _read_tables(url: str):
    html = requests.get(url, headers=UA, timeout=20).text
    return pd.read_html(html)

def _extract_name_ticker(tables):
    table=None
    for df in tables:
        cols={str(c).lower() for c in df.columns}
        if (("company" in cols or "name" in cols) and ("ticker" in cols or "symbol" in cols)):
            table=df.copy(); break
    if table is None: table=tables[0].copy()
    table.rename(columns={c:str(c).lower() for c in table.columns}, inplace=True)
    tcol=next((c for c in table.columns if "ticker" in c or "symbol" in c), table.columns[0])
    ncol=next((c for c in table.columns if "company" in c or "name" in c), table.columns[1])
    out=table[[tcol,ncol]].copy(); out.columns=["ticker","name"]
    out["ticker"]=out["ticker"].astype(str).str.strip()
    return out.dropna().drop_duplicates(subset=["ticker"])

@lru_cache(maxsize=8)
def members_cac40():
    df=_extract_name_ticker(_read_tables("https://en.wikipedia.org/wiki/CAC_40"))
    df["ticker"]=df["ticker"].apply(lambda x: x if "." in x else f"{x}.PA")
    df["index"]="CAC 40"
    return df

@lru_cache(maxsize=8)
def members_dax():
    df=_extract_name_ticker(_read_tables("https://en.wikipedia.org/wiki/DAX"))
    df["ticker"]=df["ticker"].apply(lambda x: x if "." in x else f"{x}.DE")
    df["index"]="DAX"
    return df

@lru_cache(maxsize=8)
def members_nasdaq100():
    df=_extract_name_ticker(_read_tables("https://en.wikipedia.org/wiki/Nasdaq-100"))
    df["index"]="NASDAQ 100"     # US => pas de suffix Yahoo
    return df

@lru_cache(maxsize=8)
def members_sp500():
    # Constituants S&P 500 (table Wikipedia "List of S&P 500 companies")
    tables = _read_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    # Cherche colonnes Symbol / Security
    table=None
    for df in tables:
        cols={str(c).lower() for c in df.columns}
        if ("symbol" in cols) and ("security" in cols or "company" in cols or "name" in cols):
            table=df.copy(); break
    if table is None: table=tables[0].copy()
    table.rename(columns={c:str(c).lower() for c in table.columns}, inplace=True)
    tcol=next((c for c in table.columns if "symbol" in c or "ticker" in c), table.columns[0])
    ncol=next((c for c in table.columns if "security" in c or "company" in c or "name" in c), table.columns[1])
    out=table[[tcol,ncol]].copy(); out.columns=["ticker","name"]
    out["ticker"]=out["ticker"].astype(str).str.strip()
    out["index"]="S&P 500"
    return out.dropna().drop_duplicates(subset=["ticker"])

def members(index_name: str):
    if index_name=="CAC 40": return members_cac40()
    if index_name=="DAX": return members_dax()
    if index_name=="NASDAQ 100": return members_nasdaq100()
    if index_name=="S&P 500": return members_sp500()
    return pd.DataFrame(columns=["ticker","name","index"])

# =========================
# PRIX (AJUSTÉS) & HISTO
# =========================
@lru_cache(maxsize=64)
def fetch_prices_cached(tickers_tuple, period="120d"):
    tickers=list(tickers_tuple)
    if not tickers: return pd.DataFrame()
    try:
        data=yf.download(
            tickers, period=period, interval="1d",
            auto_adjust=True, group_by="ticker", threads=False, progress=False
        )
    except Exception:
        return pd.DataFrame()
    if data is None or len(data)==0: return pd.DataFrame()
    frames=[]
    if isinstance(data,pd.DataFrame) and {"Open","High","Low","Close"}.issubset(data.columns):
        df=data.copy(); df["Ticker"]=tickers[0]; frames.append(df)
    else:
        for t in tickers:
            try:
                if t in data and isinstance(data[t],pd.DataFrame):
                    df=data[t].copy(); df["Ticker"]=t; frames.append(df)
            except Exception:
                continue
    if not frames: return pd.DataFrame()
    out=pd.concat(frames); out.reset_index(inplace=True); return out

def fetch_prices(tickers, days=120):
    return fetch_prices_cached(tuple(tickers), period=f"{days}d")

# =========================
# VARIATIONS CALENDAIRES
# =========================
def _calendar_returns(last_rows: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Variations calendaire J/7j/30j (tolérant jours sans cotations)."""
    if full_df.empty or last_rows.empty:
        for k in ("pct_1d","pct_7d","pct_30d"): last_rows[k]=np.nan
        return last_rows
    full=full_df.copy()
    full["Ticker"]=full["Ticker"].astype(str).str.upper()
    full=full.sort_values(["Ticker","Date"])
    last=last_rows.copy()
    last["Ticker"]=last["Ticker"].astype(str).str.upper()

    def lookup_price(tkr, ref_date, days_back):
        target=pd.to_datetime(ref_date)-pd.Timedelta(days=days_back)
        hist=full[full["Ticker"]==tkr][["Date","Close"]].dropna().sort_values("Date")
        if hist.empty: return np.nan
        hist=hist[hist["Date"]<=target]
        if hist.empty:
            return float(full[full["Ticker"]==tkr]["Close"].iloc[0])
        return float(hist["Close"].iloc[-1])

    vals_1, vals_7, vals_30 = [], [], []
    for _, r in last.iterrows():
        tkr, dref, pref = r["Ticker"], r["Date"], float(r["Close"])
        p1, p7, p30 = lookup_price(tkr, dref, 1), lookup_price(tkr, dref, 7), lookup_price(tkr, dref, 30)
        v1  = (pref/p1-1)  if (np.isfinite(pref) and np.isfinite(p1)  and p1>0)  else np.nan
        v7  = (pref/p7-1)  if (np.isfinite(pref) and np.isfinite(p7)  and p7>0)  else np.nan
        v30 = (pref/p30-1) if (np.isfinite(pref) and np.isfinite(p30) and p30>0) else np.nan
        if np.isfinite(v1) and abs(v1)>0.4: v1=np.nan    # anti-split extrême
        vals_1.append(v1); vals_7.append(v7); vals_30.append(v30)
    last["pct_1d"], last["pct_7d"], last["pct_30d"] = vals_1, vals_7, vals_30
    return last

# =========================
# MÉTRIQUES (MA20/50/120/240 + trend)
# =========================
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne 1 ligne par ticker avec :
    - Close, ATR14
    - MA20/50 (ST), MA120/240 (LT)
    - Gaps MA, trend scores ST & LT
    - pct_1d / pct_7d / pct_30d
    """
    cols=["Ticker","Date","Close","ATR14",
          "MA20","MA50","MA120","MA240",
          "gap20","gap50","gap120","gap240",
          "trend_score","lt_trend_score",
          "pct_1d","pct_7d","pct_30d"]
    if df is None or df.empty: return pd.DataFrame(columns=cols)
    df=df.copy()
    if "Date" not in df.columns:
        df=df.reset_index().rename(columns={df.index.name or "index":"Date"})
    need={"Ticker","Date","High","Low","Close"}
    if need - set(df.columns): return pd.DataFrame(columns=cols)

    df["Ticker"]=df["Ticker"].astype(str).str.upper()
    df=df.sort_values(["Ticker","Date"])
    df["PrevClose"]=df.groupby("Ticker")["Close"].shift(1)

    # TR & ATR
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["PrevClose"]).abs()
    tr3 = (df["Low"]  - df["PrevClose"]).abs()
    df["TR"] = np.nanmax(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
    df["ATR14"]=df.groupby("Ticker")["TR"].transform(lambda s:s.rolling(14,min_periods=5).mean())

    # MAs
    df["MA20"]  = df.groupby("Ticker")["Close"].transform(lambda s:s.rolling(20,  min_periods=5).mean())
    df["MA50"]  = df.groupby("Ticker")["Close"].transform(lambda s:s.rolling(50,  min_periods=10).mean())
    df["MA120"] = df.groupby("Ticker")["Close"].transform(lambda s:s.rolling(120, min_periods=20).mean())
    df["MA240"] = df.groupby("Ticker")["Close"].transform(lambda s:s.rolling(240, min_periods=30).mean())

    last=df.groupby("Ticker").tail(1)[["Ticker","Date","Close","ATR14","MA20","MA50","MA120","MA240"]].copy()

    # Gaps vectorisés
    def _gap(a, b):
        a=np.asarray(a, float); b=np.asarray(b, float)
        mask = np.isfinite(a) & np.isfinite(b) & (b!=0)
        out = np.full_like(a, np.nan, dtype=float)
        out[mask] = a[mask]/b[mask] - 1.0
        return out

    last["gap20"]  = _gap(last["Close"], last["MA20"])
    last["gap50"]  = _gap(last["Close"], last["MA50"])
    last["gap120"] = _gap(last["Close"], last["MA120"])
    last["gap240"] = _gap(last["Close"], last["MA240"])

    # Trend scores
    last["trend_score"]    = 0.6*last["gap20"]  + 0.4*last["gap50"]
    last["lt_trend_score"] = 0.6*last["gap120"] + 0.4*last["gap240"]

    # Variations calendaire
    last = _calendar_returns(last, df)

    return last.reset_index(drop=True)

# =========================
# INFOS SOCIÉTÉ & DIVIDENDES
# =========================
@lru_cache(maxsize=1024)
def company_name_from_ticker(ticker: str) -> str:
    if not ticker: return ""
    try:
        t = yf.Ticker(ticker)
        name = None
        try:
            name = t.fast_info.get("shortName", None)
        except Exception:
            pass
        if not name:
            info = t.get_info()
            name = info.get("shortName") or info.get("longName")
        return name or ticker
    except Exception:
        return ticker

def dividends_summary(ticker: str):
    try:
        t = yf.Ticker(ticker)
        div = t.dividends
        if div is None or div.empty:
            return [], None
        div = div.sort_index(ascending=False)
        recent = [(str(idx.date()), float(val)) for idx, val in div.head(8).items()]
        px = t.history(period="5d")["Close"].iloc[-1]
        trailing = float(div.head(4).sum()/px) if px and px>0 else None
        return recent, trailing
    except Exception:
        return [], None

# =========================
# NEWS (avec dates) & RÉSUMÉ
# =========================
@lru_cache(maxsize=256)
def google_news_titles(query, lang="fr"):
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl={lang}-{lang.upper()}&gl={lang.upper()}&ceid={lang.upper()}:{lang.upper()}"
    try:
        xml = requests.get(url, headers=UA, timeout=12).text
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        items = []
        for it in root.iter("item"):
            title = it.findtext("title") or ""
            link  = it.findtext("link") or ""
            pub   = it.findtext("{http://www.w3.org/2005/Atom}updated") or it.findtext("pubDate") or ""
            items.append((title, link, pub))
        return items[:10]
    except Exception:
        return []

def filter_company_news(ticker, company_name, items):
    if not items: return []
    tkr=(ticker or "").lower()
    name=(company_name or "").lower()
    name_short = name.split(" ")[0] if name else ""
    keep=[]
    for title, link, pub in items:
        tl=title.lower()
        if (tkr and tkr in tl) or (name and name in tl) or (name_short and name_short in tl):
            keep.append((title, link, pub))
    return keep

def news_summary(name, ticker, lang="fr"):
    items = google_news_titles(f"{name} {ticker}", lang) or google_news_titles(name, lang)
    items = filter_company_news(ticker, name, items)
    titles = [t for t, _, _ in items]
    if not titles:
        return ("Pas d’actualité saillante — mouvement technique / macro.", 0.0, [])
    POS=["résultats","bénéfice","contrat","relève","guidance","record","upgrade","partenariat","dividende","approbation"]
    NEG=["profit warning","retard","procès","amende","downgrade","abaisse","enquête","rappel","départ","incident"]
    scores=[]
    for t in titles:
        s=0.0
        if SIA:
            try: s=SIA.polarity_scores(t.lower())["compound"]
            except Exception: s=0.0
        tl=t.lower()
        if any(k in tl for k in POS): s += 0.2
        if any(k in tl for k in NEG): s -= 0.2
        scores.append(s)
    m=float(np.mean(scores)) if scores else 0.0
    txt = ("Hausse soutenue par des nouvelles positives."
           if m>0.15 else
           "Baisse liée à des nouvelles défavorables."
           if m<-0.15 else
           "Actualité mitigée/neutre — mouvement surtout technique.")
    return (txt, m, items)

# =========================
# DÉCISIONS IA & NIVEAUX (STRICT)
# =========================
def price_levels_from_row(row, profile="Neutre"):
    p=get_profile_params(profile)
    px=float(row.get("Close", math.nan))
    ma20=float(row.get("MA20", math.nan)) if pd.notna(row.get("MA20", math.nan)) else math.nan
    base=ma20 if math.isfinite(ma20) else px
    if not math.isfinite(base): return {"entry":math.nan,"target":math.nan,"stop":math.nan}
    return {
        "entry": round(base*p["entry_mult"],2),
        "target": round(base*p["target_mult"],2),
        "stop":   round(base*p["stop_mult"],2),
    }

def decision_label_strict(row, profile="Neutre", held=False):
    """
    IA stricte :
    - ST OK : Close ≥ MA20 ET Close ≥ MA50
    - LT OK : MA120 ≥ MA240 ET Close ≥ MA120
    - Volatilité <= seuil du profil
    - Momentum 7/30j pas trop négatif (mélange pondéré >= -1%)
    """
    p=get_profile_params(profile)
    vol_max=p["vol_max"]

    px=row.get("Close",np.nan)
    ma20=row.get("MA20",np.nan); ma50=row.get("MA50",np.nan)
    ma120=row.get("MA120",np.nan); ma240=row.get("MA240",np.nan)
    atr=row.get("ATR14",np.nan)

    if not np.isfinite(px):
        return "👁️ Surveiller"

    # Conditions cœur
    ct_ok=(np.isfinite(ma20) and px>=ma20) and (np.isfinite(ma50) and px>=ma50)
    lt_ok=(np.isfinite(ma120) and np.isfinite(ma240) and ma120>=ma240 and px>=ma120)

    vol=(atr/px) if (np.isfinite(atr) and px>0) else 0.03
    if profile=="Prudent":
        vol_ok=vol<=vol_max*0.9
    elif profile=="Agressif":
        vol_ok=vol<=vol_max*1.2
    else:
        vol_ok=vol<=vol_max

    m7=row.get("pct_7d",np.nan); m30=row.get("pct_30d",np.nan)
    mom_ok=True
    if np.isfinite(m7) or np.isfinite(m30):
        mix=(0.6*(m7 if np.isfinite(m7) else 0)+0.4*(m30 if np.isfinite(m30) else 0))
        mom_ok=(mix>=-0.01)  # >= -1%

    # Décision
    if not ct_ok:
        if held and (not lt_ok) and (vol>vol_max*1.2):
            return "🔴 Vendre"
        return "👁️ Surveiller"

    if not lt_ok:
        if held and (vol>vol_max*1.2):
            return "🔴 Vendre"
        return "👁️ Surveiller"

    if not vol_ok or not mom_ok:
        return "👁️ Surveiller"

    return "🟢 Acheter"

# Compat : anciennes fonctions pages
def decision_label_from_row(row, held=False, vol_max=0.05):
    # On utilise le profil sauvegardé pour rester cohérent entre pages
    profile = load_profile()
    return decision_label_strict(row, profile=profile, held=held)

def decision_label_combined(row, held=False, vol_max=0.05):
    profile = load_profile()
    return decision_label_strict(row, profile=profile, held=held)

# =========================
# STYLE TABLEAUX (couleurs)
# =========================
def style_variations(df, cols):
    def color_var(v):
        if pd.isna(v): return ""
        if v>0: return "background-color:#e8f5e9; color:#0b8f3a"
        if v<0: return "background-color:#ffebee; color:#d5353a"
        return "background-color:#e8f0fe; color:#1e88e5"
    sty=df.style
    for c in cols:
        if c in df.columns: sty=sty.applymap(color_var, subset=[c])
    return sty

# Helpers de surbrillance proximité (si besoin dans pages)
def color_proximity_adaptive(v):
    if pd.isna(v): return ""
    if abs(v) <= 2: return "background-color:#e6f4ea; color:#0b8043"
    if abs(v) <= 5: return "background-color:#fff8e1; color:#a67c00"
    return "background-color:#ffebee; color:#b71c1c"

def highlight_near_entry_adaptive(row):
    p = row.get("Proximité (%)")
    if pd.notna(p) and abs(p) <= 2:
        return ["background-color: #2a2a2a20; font-weight:600"] * len(row)
    return [""] * len(row)

# =========================
# AGGRÉGATION MARCHÉS (CAC40 / DAX / NASDAQ100 / S&P500 / LS)
# =========================
def fetch_all_markets(markets, days_hist=240):
    """
    markets: liste de tuples (Indice, source) – source ignorée
    Supporte: "CAC 40", "DAX", "NASDAQ 100", "S&P 500", "LS Exchange"
    """
    frames=[]
    for idx, _ in markets:
        if idx=="CAC 40":
            mem=members("CAC 40")
        elif idx=="DAX":
            mem=members("DAX")
        elif idx=="NASDAQ 100":
            mem=members("NASDAQ 100")
        elif idx=="S&P 500":
            mem=members("S&P 500")
        elif idx=="LS Exchange":
            # Watchlist perso, convertie en Yahoo via mapping/heuristique
            raw = load_watchlist_ls()
            tickers=[maybe_guess_yahoo(x) or x for x in raw] if raw else []
            mem=pd.DataFrame({"ticker": tickers, "name": raw})
        else:
            continue

        if mem.empty: continue

        px=fetch_prices(mem["ticker"].tolist(), days=days_hist)
        if px.empty: continue

        met=compute_metrics(px).merge(mem, left_on="Ticker", right_on="ticker", how="left")
        met["Indice"]=idx
        frames.append(met)

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

# =========================
# SÉLECTION IA OPTIMALE (TOP N)
# =========================
def select_top_actions(df, profile="Neutre", n=10, include_proximity=True):
    """
    Retourne les meilleures actions (≤ n) selon IA stricte :
    - tendance ST (MA20/50) & LT (MA120/240)
    - momentum (7j/30j), volatilité (ATR/Close)
    - décision IA stricte
    - niveaux Entrée/Objectif/Stop
    - Potentiel (€) & Proximité (%)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    p = get_profile_params(profile)
    vol_max = p["vol_max"]

    data = df.copy()
    needed = ["trend_score","lt_trend_score","pct_7d","pct_30d","ATR14","Close"]
    for c in needed:
        if c not in data.columns: data[c] = np.nan
    data = data.dropna(subset=["Close"]).copy()
    data["Volatilité"] = data["ATR14"] / data["Close"]

    # Score IA global (LT un peu + fort)
    data["IA_Score"] = (
        (data["lt_trend_score"].fillna(0) * 60.0)
        + (data["trend_score"].fillna(0)     * 40.0)
        + (data["pct_30d"].fillna(0)         * 100.0)
        + (data["pct_7d"].fillna(0)          * 50.0)
        - (data["Volatilité"].fillna(0)      * 10.0)
    )

    data["Signal"] = data.apply(lambda r: decision_label_strict(r, profile=profile, held=False), axis=1)

    # Garde signaux "Acheter" et vol maîtrisée
    filt = (data["Signal"].str.contains("🟢", na=False)) & (data["Volatilité"] <= vol_max * 1.5)
    data = data[filt].sort_values("IA_Score", ascending=False)

    def _levels(r):
        lev = price_levels_from_row(r, profile)
        pot_eur = None
        prox_pct = None
        if lev["entry"] and lev["target"]:
            pot_eur = lev["target"] - lev["entry"]
        if lev["entry"] and r.get("Close"):
            entry = lev["entry"]
            if entry and entry > 0:
                prox_pct = ((r["Close"] / entry) - 1) * 100
        return pd.Series({
            "Entrée (€)": lev["entry"],
            "Objectif (€)": lev["target"],
            "Stop (€)": lev["stop"],
            "Potentiel (€)": pot_eur,
            "Proximité (%)": prox_pct
        })

    levs = data.apply(_levels, axis=1)
    top = pd.concat([data.reset_index(drop=True), levs.reset_index(drop=True)], axis=1).head(n)

    keep = ["Ticker","name","Close","MA20","MA50","MA120","MA240",
            "trend_score","lt_trend_score","pct_7d","pct_30d",
            "Volatilité","IA_Score","Signal",
            "Entrée (€)","Objectif (€)","Stop (€)","Potentiel (€)","Proximité (%)"]
    for k in keep:
        if k not in top.columns:
            top[k] = np.nan
    top = top[keep]

    top.rename(columns={
        "Ticker":"Symbole","name":"Société","Close":"Cours (€)",
        "trend_score":"Trend ST","lt_trend_score":"Trend LT",
        "pct_7d":"Perf 7j (%)","pct_30j":"Perf 30j (%)",
        "Volatilité":"Risque","IA_Score":"Score IA"
    }, inplace=True)

    # Formats
    if "Perf 7j (%)" in top.columns:   top["Perf 7j (%)"]   = (top["Perf 7j (%)"]*100).round(2)
    if "Perf 30j (%)" in top.columns and "Perf 30j (%)" not in top:  # garde compat si nom différent
        pass
    if "pct_30d" in df.columns and "Perf 30j (%)" in top.columns:
        top["Perf 30j (%)"]  = top["Perf 30j (%)"].round(2)
    if "Risque" in top.columns:        top["Risque"]        = (top["Risque"]*100).round(2)
    if "Score IA" in top.columns:      top["Score IA"]      = top["Score IA"].round(2)
    if "Cours (€)" in top.columns:     top["Cours (€)"]     = top["Cours (€)"].round(2)
    if "Potentiel (€)" in top.columns: top["Potentiel (€)"] = top["Potentiel (€)"].round(2)
    if include_proximity and "Proximité (%)" in top.columns:
        top["Proximité (%)"] = top["Proximité (%)"].round(2)

    return top.reset_index(drop=True)

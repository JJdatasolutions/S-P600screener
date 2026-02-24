import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import concurrent.futures
from datetime import datetime, timedelta
import requests
import io
import google.generativeai as genai

# --- 1. CONFIGURATIE & CACHING ---
st.set_page_config(page_title="S&P 600 Quality Screener", layout="wide", page_icon="📈")

@st.cache_data(ttl=86400)
def get_sp600_tickers():
    """Haalt de S&P 600 lijst op met een User-Agent bypass voor Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    tickers = tables[0]['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
    return tickers

@st.cache_data(ttl=3600)
def get_bulk_prices(tickers, benchmark="IWM", periods=200):
    """Downloadt prijzen gevectoriseerd."""
    all_tickers = tickers + [benchmark]
    end = datetime.today()
    start = end - timedelta(days=periods)
    df_prices = yf.download(all_tickers, start=start, end=end, progress=False)
    
    if isinstance(df_prices.columns, pd.MultiIndex):
        df_close = df_prices['Adj Close'] if 'Adj Close' in df_prices.columns.levels[0] else df_prices['Close']
    else:
        df_close = df_prices

    if benchmark in df_close.columns:
        return df_close.drop(columns=[benchmark]), df_close[benchmark]
    return None, None

@st.cache_data(ttl=86400)
def get_fundamentals_bulk(tickers):
    """Asynchroon ophalen van fundamenten inclusief P/E ratio."""
    def fetch_info(ticker):
        try:
            info = yf.Ticker(ticker).info
            return {
                'Ticker': ticker,
                'ROE': info.get('returnOnEquity', 0) or 0,
                'Beta': info.get('beta', 1.5) or 1.5,
                'Gross Margins': info.get('grossMargins', 0) or 0,
                'PE': info.get('trailingPE', np.nan)
            }
        except:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(fetch_info, tickers):
            if res is not None:
                results.append(res)
    return pd.DataFrame(results)

def calculate_rrg_vectorized(stock_prices, benchmark_prices, window=14):
    """Gevectoriseerde RRG (Relative Rotation Graph) berekening."""
    stock_prices, benchmark_prices = stock_prices.align(benchmark_prices, join='inner', axis=0)
    rel_prices = stock_prices.div(benchmark_prices, axis=0)
    rs_ratio = 100 + ((rel_prices / rel_prices.rolling(window).mean()) - 1) * 100
    rs_momentum = 100 + ((rs_ratio / rs_ratio.rolling(window).mean()) - 1) * 100
    
    df_rrg = pd.DataFrame({'RS-Ratio': rs_ratio.iloc[-1], 'RS-Momentum': rs_momentum.iloc[-1]}).dropna()
    df_rrg['Distance'] = np.sqrt((df_rrg['RS-Ratio'] - 100)**2 + (df_rrg['RS-Momentum'] - 100)**2)
    
    heading = np.degrees(np.arctan2(df_rrg['RS-Momentum'] - 100, df_rrg['RS-Ratio'] - 100))
    df_rrg['Heading'] = np.where(heading < 0, heading + 360, heading)
    df_rrg['Sweet_Spot_Multiplier'] = np.where((df_rrg['Heading'] >= 0) & (df_rrg['Heading'] <= 90),
                                               np.cos(np.radians(df_rrg['Heading'] - 45)), 0.1)
    
    conditions = [
        (df_rrg['RS-Ratio'] >= 100) & (df_rrg['RS-Momentum'] >= 100),
        (df_rrg['RS-Ratio'] < 100) & (df_rrg['RS-Momentum'] >= 100),
        (df_rrg['RS-Ratio'] < 100) & (df_rrg['RS-Momentum'] < 100),
        (df_rrg['RS-Ratio'] >= 100) & (df_rrg['RS-Momentum'] < 100)
    ]
    choices = ['Leading', 'Improving', 'Lagging', 'Weakening']
    df_rrg['Quadrant'] = np.select(conditions, choices, default='Unknown')
    
    return df_rrg.reset_index(names='Ticker')

# --- 3. STREAMLIT UI ---

st.title("🔬 S&P 600 QARP Screener (Quality At a Reasonable Price)")

if 'df_top' not in st.session_state:
    st.session_state.df_top = pd.DataFrame()

tab1, tab2, tab3 = st.tabs(["📚 Waarom deze Screener?", "⚙️ Screener & Resultaten", "🤖 AI Analyst Prompt"])

with tab1:
    st.header("Financiële Geletterdheid: Waarom deze criteria?")
    st.write("Het **'Small Cap effect'** wordt vaak tenietgedaan door 'junk' bedrijven. We filteren op:")
    colA, colB, colC = st.columns(3)
    with colA:
        st.info("**1. Quality (Kwaliteit - 40%)**\n\n**ROE:** Hoe efficiënt maakt het bedrijf winst?\n\n**Beta:** We zoeken naar rustige stijgers (<1).")
    with colB:
        st.warning("**2. Value (Waarde - 30%)**\n\n**P/E Ratio:** We willen kwaliteit tegen een eerlijke prijs.")
    with colC:
        st.success("**3. Momentum (Trend - 30%)**\n\n**RRG:** Aandelen die momentum opbouwen en de markt verslaan.")

with tab2:
    if st.button("Start Volledige S&P 600 Scan", type="primary"):
        with st.spinner('Bezig met rekenen... (S&P 600 Tickers ophalen)'):
            tickers = get_sp600_tickers()
        with st.spinner('Prijzen downloaden (Vectorized)...'):
            stock_prices, benchmark_prices = get_bulk_prices(tickers)
        if stock_prices is not None:
            with st.spinner('Fundamenten & P/E ratio\'s ophalen...'):
                df_fundamentals = get_fundamentals_bulk(tickers)
            with st.spinner('Scores normaliseren en Alpha berekenen...'):
                df_rrg = calculate_rrg_vectorized(stock_prices, benchmark_prices)
                
                df_fundamentals['Quality_Raw'] = (df_fundamentals['ROE'] * 100) + (df_fundamentals['Gross Margins'] * 100) + np.maximum(0, (2.0 - df_fundamentals['Beta']) * 50)
                df_fundamentals['PE_Clean'] = pd.to_numeric(df_fundamentals['PE'], errors='coerce').fillna(999)
                df_fundamentals['Value_Raw'] = np.where((df_fundamentals['PE_Clean'] > 0) & (df_fundamentals['PE_Clean'] < 100), 1 / df_fundamentals['PE_Clean'], 0)
                df_rrg['Momentum_Raw'] = df_rrg['Distance'] * df_rrg['Sweet_Spot_Multiplier']
                
                df_final = pd.merge(df_rrg, df_fundamentals, on='Ticker', how='inner')
                
                for col in ['Quality_Raw', 'Value_Raw', 'Momentum_Raw']:
                    min_val, max_val = df_final[col].min(), df_final[col].max()
                    df_final[col.replace('_Raw', '_Score')] = (df_final[col] - min_val) / (max_val - min_val + 1e-9) * 100
                
                df_final['Alpha Score'] = (0.4 * df_final['Quality_Score']) + (0.3 * df_final['Value_Score']) + (0.3 * df_final['Momentum_Score'])
                
                st.session_state.df_top = df_final.sort_values(by='Alpha Score', ascending=False).head(50)
            st.success("Screener voltooid!")

    if not st.session_state.df_top.empty:
        df_display = st.session_state.df_top.copy()
        df_display[['Alpha Score', 'Quality_Score', 'Value_Score', 'Momentum_Score', 'ROE', 'PE_Clean']] = df_display[['Alpha Score', 'Quality_Score', 'Value_Score', 'Momentum_Score', 'ROE', 'PE_Clean']].round(2)
        
        st.markdown("""
        ### 🎯 Koopgids: Hoe selecteer je de winnaars?
        Kijk naar de grafiek hieronder en zoek naar de **"Sweet Spot"**:
        * 📍 **Positie (Rechtsboven):** Dit is het *Leading* kwadrant. Deze aandelen verslaan de benchmark.
        * 🟢 **Kleur (Donkergroen):** De kleur toont de **Alpha Score**. Hoe groener, hoe beter de mix van Kwaliteit, Waarde en Momentum. Rood betekent vermijden.
        * 🔵 **Grootte (Grote bollen):** Hoe groter de bol, hoe sterker de fundamenten (**Quality Score**). 
        
        👉 **Jouw ideale aandeel is een GROTE, DONKERGROENE bol in de RECHTERBOVENHOEK.**
        ---
        """)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter(
                df_display, x="RS-Ratio", y="RS-Momentum", text="Ticker", 
                size="Quality_Score",          # Grootte = Kwaliteit/Fundamenten
                color="Alpha Score",           # Kleur = Totale Alpha Score
                color_continuous_scale="RdYlGn", # Rood (Slecht) naar Groen (Goed)
                title="Interactieve RRG: Vind de Donkergroene Bollen", 
                hover_data=['ROE', 'Beta', 'PE_Clean', 'Quadrant']
            )
            fig.add_hline(y=100, line_dash="dash", line_color="gray")
            fig.add_vline(x=100, line_dash="dash", line_color="gray")
            
            # Voeg labels toe voor de kwadranten om het nóg duidelijker te maken
            fig.add_annotation(x=102, y=102, text="LEADING (Kopen)", showarrow=False, font=dict(color="green", size=14))
            fig.add_annotation(x=98, y=102, text="IMPROVING (Watchlist)", showarrow=False, font=dict(color="blue", size=10))
            fig.add_annotation(x=98, y=98, text="LAGGING (Verkopen)", showarrow=False, font=dict(color="red", size=10))
            fig.add_annotation(x=102, y=98, text="WEAKENING (Let op)", showarrow=False, font=dict(color="orange", size=10))
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Top Kooplijst (Gesorteerd op Alpha)")
            st.dataframe(df_display[['Ticker', 'Alpha Score', 'Quadrant', 'PE_Clean']], hide_index=True, use_container_width=True)

with tab3:
    st.header("🤖 Live AI Agent Analyse")
    st.write("Laat de AI direct een diepgaande analyse schrijven op basis van de screener-data én het meest recente nieuws.")
    
    # 1. Check of de API key is ingesteld (bijv. in Streamlit Secrets)
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    
    if not api_key:
        st.warning("⚠️ Voeg je GEMINI_API_KEY toe aan je Streamlit Secrets om de Live Agent te gebruiken.")
    elif not st.session_state.df_top.empty:
        # Configureer de Gemini API
        genai.configure(api_key=api_key)
        
        selected_ticker = st.selectbox("Selecteer een aandeel voor de AI Agent:", st.session_state.df_top['Ticker'].tolist())
        
        if st.button(f"Genereer Rapport voor {selected_ticker}", type="primary"):
            with st.spinner(f"Agent is actueel nieuws aan het ophalen voor {selected_ticker}..."):
                
                # 2. Haal actueel nieuws op via yfinance
                ticker_obj = yf.Ticker(selected_ticker)
                news_items = ticker_obj.news
                
                news_context = ""
                if news_items:
                    news_context = "Recente Nieuwskoppen:\n"
                    for item in news_items[:5]: # Pak de 5 meest recente artikelen
                        title = item.get('title', 'Geen titel')
                        publisher = item.get('publisher', 'Onbekend')
                        news_context += f"- {title} (Bron: {publisher})\n"
                else:
                    news_context = "Geen recent nieuws gevonden via Yahoo Finance."

                # 3. Verzamel de kwantitatieve data
                stock_data = st.session_state.df_top[st.session_state.df_top['Ticker'] == selected_ticker].iloc[0]
                
                # 4. Bouw de krachtige, dynamische prompt voor de AI
                agent_prompt = f"""
                Je bent een expert kwantitatief analist en portfolio manager. Analyseer aandeel {selected_ticker}. 

                Hier zijn de fundamentele feiten uit onze QARP-screener:
                - ROE (Return on Equity): {stock_data['ROE']:.2f}
                - Beta (Volatiliteit/Risico): {stock_data['Beta']:.2f}
                - P/E ratio: {stock_data['PE_Clean']:.2f}
                - RRG Trend: Het aandeel bevindt zich in het '{stock_data['Quadrant']}' kwadrant.

                Hier is de actuele nieuwscontext (gebruik dit om de katalysatoren en risico's te duiden):
                {news_context}

                Schrijf een kort, krachtig en professioneel beleggingsrapport. 
                1. Beoordeel de fundamentele gezondheid via de "Quality minus Junk" theorie van Asness.
                2. Beoordeel de waardering volgens Benjamin Graham.
                3. Integreer de nieuwskoppen om uit te leggen *waarom* het aandeel momenteel in het {stock_data['Quadrant']} kwadrant staat.
                4. Geef een duidelijke conclusie: is dit een koop, houd, of verkoop kandidaat op basis van deze mix van data en nieuws?
                """
            
            with st.spinner("Agent schrijft het rapport..."):
                try:
                    # 5. Roep het Gemini model aan
                    model = genai.GenerativeModel('gemini-pro') # Snel en krachtig model
                    response = model.generate_content(agent_prompt)
                    
                    # 6. Toon het resultaat prachtig in Streamlit
                    st.success("Analyse voltooid!")
                    st.markdown(f"### 📊 AI Agent Rapport: {selected_ticker}")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Er ging iets mis met de AI: {e}")
    else:
        st.info("Draai eerst de screener in de 'Screener & Resultaten' tab om aandelen te kunnen selecteren.")

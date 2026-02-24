import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import concurrent.futures
from datetime import datetime, timedelta
import requests  # <-- NIEUW
import io        # <-- NIEUW

# --- 1. CONFIGURATIE & CACHING ---
st.set_page_config(page_title="Scientific Small-Cap Screener", layout="wide", page_icon="🔬")

@st.cache_data(ttl=86400) # Cache voor 24 uur
def get_sp600_tickers():
    """Haalt de actuele S&P 600 lijst van Wikipedia met een User-Agent bypass."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
    
    # Voeg een User-Agent header toe zodat Wikipedia ons niet blokkeert
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Haal de pagina op via requests in plaats van direct via pandas
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Check of het ophalen gelukt is
    
    # Lees de HTML via io.StringIO (voorkomt Pandas deprecation warnings)
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    
    # Vervang punten door streepjes voor Yahoo Finance (bijv. BRK.B -> BRK-B)
    tickers = df['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
    return tickers

@st.cache_data(ttl=3600) # Cache prijzen voor 1 uur
def get_bulk_prices(tickers, benchmark="IWM", periods=200):
    """Downloadt prijzen voor de hele index én benchmark in één keer gevectoriseerd."""
    all_tickers = tickers + [benchmark]
    end = datetime.today()
    start = end - timedelta(days=periods)
    
    # Bulk download is enorm veel sneller dan loopen
    df_prices = yf.download(all_tickers, start=start, end=end, progress=False)
    
    # Fix voor de KeyError: Omgaan met nieuwe yfinance MultiIndex structuur
    if isinstance(df_prices.columns, pd.MultiIndex):
        if 'Close' in df_prices.columns.levels[0]:
            df_close = df_prices['Close']
        elif 'Adj Close' in df_prices.columns.levels[0]:
            df_close = df_prices['Adj Close']
        else:
            st.error("Kon geen 'Close' of 'Adj Close' prijzen vinden in de yfinance data.")
            return None, None
    else:
        df_close = df_prices

    # Splits benchmark en aandelen
    if benchmark in df_close.columns:
        benchmark_prices = df_close[benchmark]
        stock_prices = df_close.drop(columns=[benchmark])
    else:
        return None, None
        
    return stock_prices, benchmark_prices

@st.cache_data(ttl=86400)
def get_fundamentals_bulk(tickers):
    """Haalt fundamentele data op via asynchrone multithreading."""
    def fetch_info(ticker):
        try:
            info = yf.Ticker(ticker).info
            # Filter 'Junk' via default waarden (slechte ROE, hoge Beta als data ontbreekt)
            return {
                'Ticker': ticker,
                'Market Cap': info.get('marketCap', np.nan),
                'ROE': info.get('returnOnEquity', 0) if info.get('returnOnEquity') is not None else 0,
                'Beta': info.get('beta', 1.5) if info.get('beta') is not None else 1.5,
                'Gross Margins': info.get('grossMargins', 0) if info.get('grossMargins') is not None else 0
            }
        except:
            return None

    results = []
    # Beperk workers tot 10 om Yahoo Finance rate-limits te voorkomen
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(fetch_info, tickers):
            if res is not None:
                results.append(res)
                
    return pd.DataFrame(results)

# --- 2. GEVECTORISEERDE WETENSCHAPPELIJKE BEREKENINGEN ---

def calculate_rrg_vectorized(stock_prices, benchmark_prices, window=14):
    """Berekent RRG metrics voor ALLE aandelen tegelijk m.b.v. Pandas en Numpy."""
    # Lijn de dataframes uit op index (datums)
    stock_prices, benchmark_prices = stock_prices.align(benchmark_prices, join='inner', axis=0)
    
    # Relatieve prijs t.o.v. benchmark (Matrix deling)
    rel_prices = stock_prices.div(benchmark_prices, axis=0)
    
    # RS-Ratio
    rs_ratio = 100 + ((rel_prices / rel_prices.rolling(window).mean()) - 1) * 100
    
    # RS-Momentum
    rs_momentum = 100 + ((rs_ratio / rs_ratio.rolling(window).mean()) - 1) * 100
    
    # We pakken alleen de laatste actuele dag
    current_ratio = rs_ratio.iloc[-1]
    current_mom = rs_momentum.iloc[-1]
    
    # Combineer in één dataframe
    df_rrg = pd.DataFrame({
        'RS-Ratio': current_ratio,
        'RS-Momentum': current_mom
    }).dropna()
    
    # Vectorized Distance & Heading berekening met Numpy
    df_rrg['Distance'] = np.sqrt((df_rrg['RS-Ratio'] - 100)**2 + (df_rrg['RS-Momentum'] - 100)**2)
    
    angle_rad = np.arctan2(df_rrg['RS-Momentum'] - 100, df_rrg['RS-Ratio'] - 100)
    heading = np.degrees(angle_rad)
    df_rrg['Heading'] = np.where(heading < 0, heading + 360, heading)
    
    # Sweet Spot multiplier (0-90 graden belonen met cosinus van hoek t.o.v. 45 graden)
    df_rrg['Sweet_Spot_Multiplier'] = np.where(
        (df_rrg['Heading'] >= 0) & (df_rrg['Heading'] <= 90),
        np.cos(np.radians(df_rrg['Heading'] - 45)),
        0.1
    )
    
    df_rrg.reset_index(inplace=True)
    df_rrg.rename(columns={'index': 'Ticker', 'Ticker': 'Ticker'}, inplace=True)
    return df_rrg

# --- 3. STREAMLIT UI ---

st.title("🔬 Scientific S&P 600 Quality Screener")
st.markdown("*Volledig gevectoriseerde engine gebaseerd op 'Size Matters, if You Control Your Junk' (Asness et al., 2018)*")

tab1, tab2 = st.tabs(["Dashboard & RRG", "Methodologie & Alpha Score"])

with tab2:
    st.header("Wetenschappelijke Context")
    st.write("""
    Het 'Size Premium' stelt dat kleine bedrijven grote bedrijven verslaan. Asness (2018) toonde aan dat dit effect sterker is als je 'Junk' filtert.
    In dit systeem wordt Quality berekend als: `(ROE + Marges) + (Lage Beta Bonus)`.
    Alpha Score combineert dit met de RRG positie (Distance x Heading).
    """)

with tab1:
    if st.button("Start Volledige S&P 600 Scan (Ca. 15-30 seconden)"):
        with st.spinner('S&P 600 Tickers ophalen...'):
            tickers = get_sp600_tickers()
            
        with st.spinner('Gevectoriseerde prijsdata downloaden (RRG)...'):
            stock_prices, benchmark_prices = get_bulk_prices(tickers)
            
        if stock_prices is not None:
            with st.spinner('Fundamenten ophalen via asynchrone threads (Quality Score)...'):
                df_fundamentals = get_fundamentals_bulk(tickers)
                
            with st.spinner('Alpha Scores genereren...'):
                # 1. RRG Berekenen
                df_rrg = calculate_rrg_vectorized(stock_prices, benchmark_prices)
                
                # 2. Quality Score Berekenen in Fundamentals
                df_fundamentals['Profitability'] = (df_fundamentals['ROE'] * 100) + (df_fundamentals['Gross Margins'] * 100)
                df_fundamentals['Safety'] = np.maximum(0, (2.0 - df_fundamentals['Beta']) * 50) # Betting Against Beta
                df_fundamentals['Quality Score'] = df_fundamentals['Profitability'] + df_fundamentals['Safety']
                
                # 3. Mergen en Alpha Score berekenen
                df_final = pd.merge(df_rrg, df_fundamentals, on='Ticker', how='inner')
                df_final['Alpha Score'] = df_final['Distance'] * df_final['Sweet_Spot_Multiplier'] * df_final['Quality Score']
                
                # Sorteer en filter op de beste aandelen
                df_top = df_final[df_final['Quality Score'] > 0].sort_values(by='Alpha Score', ascending=False).head(50)
            
            st.success("Screener succesvol voltooid op de gehele S&P 600!")
            
            # --- VISUALISATIES ---
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("RRG Scatter Plot (Top 50 Alpha's)")
                fig = px.scatter(
                    df_top, 
                    x="RS-Ratio", 
                    y="RS-Momentum", 
                    text="Ticker",
                    size="Quality Score",
                    color="Alpha Score",
                    color_continuous_scale="Viridis",
                    title="Relative Rotation Graph t.o.v. IWM",
                    hover_data=['ROE', 'Beta', 'Heading']
                )
                fig.add_hline(y=100, line_dash="dash", line_color="gray")
                fig.add_vline(x=100, line_dash="dash", line_color="gray")
                fig.add_annotation(x=105, y=105, text="Leading (Noordoost)", showarrow=False)
                fig.add_annotation(x=95, y=95, text="Lagging (Zuidwest)", showarrow=False)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("🏆 Top 15 Alpha Picks")
                df_display = df_top.head(15)[['Ticker', 'Alpha Score', 'Quality Score', 'Heading']]
                df_display['Alpha Score'] = df_display['Alpha Score'].round(0)
                df_display['Quality Score'] = df_display['Quality Score'].round(1)
                df_display['Heading'] = df_display['Heading'].round(1).astype(str) + "°"
                st.dataframe(df_display, hide_index=True)

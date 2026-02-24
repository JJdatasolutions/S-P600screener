import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import concurrent.futures
from datetime import datetime, timedelta
import requests
import io

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
                'PE': info.get('trailingPE', np.nan) # Nodig voor Value Score
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
    
    # Bepaal Kwadrant
    conditions = [
        (df_rrg['RS-Ratio'] > 100) & (df_rrg['RS-Momentum'] > 100),
        (df_rrg['RS-Ratio'] < 100) & (df_rrg['RS-Momentum'] > 100),
        (df_rrg['RS-Ratio'] < 100) & (df_rrg['RS-Momentum'] < 100),
        (df_rrg['RS-Ratio'] > 100) & (df_rrg['RS-Momentum'] < 100)
    ]
    choices = ['Leading', 'Improving', 'Lagging', 'Weakening']
    df_rrg['Quadrant'] = np.select(conditions, choices, default='Unknown')
    
    return df_rrg.reset_index(names='Ticker')

# --- 3. STREAMLIT UI ---

st.title("🔬 S&P 600 QARP Screener (Quality At a Reasonable Price)")

# Initialiseer session_state voor resultaten
if 'df_top' not in st.session_state:
    st.session_state.df_top = pd.DataFrame()

tab1, tab2, tab3 = st.tabs(["📚 Waarom deze Screener?", "⚙️ Screener & Resultaten", "🤖 AI Analyst Prompt"])

with tab1:
    st.header("Financiële Geletterdheid: Waarom deze criteria?")
    st.write("""
    Het **'Small Cap effect'** stelt dat kleine bedrijven historisch gezien sneller groeien dan grote reuzen. Echter, dit effect wordt vaak tenietgedaan door zogenaamde 'junk' bedrijven (bedrijven met hoge schulden en structurele verliezen). 
    
    Om het échte potentieel van kleine aandelen te vangen, filtert deze screener op drie pijlers:
    """)
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.info("**1. Quality (Kwaliteit - 40%)**\n\n**ROE (Rendement op Eigen Vermogen):** Hoe efficiënt maakt het bedrijf winst met jouw geld?\n\n**Beta (Stabiliteit):** Een score onder de 1 betekent dat het aandeel minder wild beweegt dan de brede markt. We zoeken naar rustige stijgers.")
    with colB:
        st.warning("**2. Value (Waarde - 30%)**\n\n**P/E Ratio (Koers-winstverhouding):** We willen kwaliteit, maar we willen er niet te veel voor betalen. Bedrijven met een gezonde, relatief lage P/E ratio scoren hier hoger.")
    with colC:
        st.success("**3. Momentum (Trend - 30%)**\n\n**RRG (Relative Rotation Graph):** Presteert het aandeel momenteel beter dan de markt? We zoeken aandelen die momentum opbouwen en in het 'Leading' of 'Improving' kwadrant zitten.")

    st.markdown("### De Formule")
    st.latex(r"Alpha \ Score = (0.4 \times Quality) + (0.3 \times Value) + (0.3 \times Momentum)")

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
                
                # Raw Scores
                df_fundamentals['Quality_Raw'] = (df_fundamentals['ROE'] * 100) + (df_fundamentals['Gross Margins'] * 100) + np.maximum(0, (2.0 - df_fundamentals['Beta']) * 50)
                # Value Raw: Hoge voorkeur voor P/E tussen 0 en 30. (Inversie)
                df_fundamentals['PE_Clean'] = pd.to_numeric(df_fundamentals['PE'], errors='coerce').fillna(999)
                df_fundamentals['Value_Raw'] = np.where((df_fundamentals['PE_Clean'] > 0) & (df_fundamentals['PE_Clean'] < 100), 1 / df_fundamentals['PE_Clean'], 0)
                df_rrg['Momentum_Raw'] = df_rrg['Distance'] * df_rrg['Sweet_Spot_Multiplier']
                
                df_final = pd.merge(df_rrg, df_fundamentals, on='Ticker', how='inner')
                
                # Cross-Sectional Min-Max Scaling (0-100)
                for col in ['Quality_Raw', 'Value_Raw', 'Momentum_Raw']:
                    min_val, max_val = df_final[col].min(), df_final[col].max()
                    df_final[col.replace('_Raw', '_Score')] = (df_final[col] - min_val) / (max_val - min_val + 1e-9) * 100
                
                df_final['Alpha Score'] = (0.4 * df_final['Quality_Score']) + (0.3 * df_final['Value_Score']) + (0.3 * df_final['Momentum_Score'])
                
                # Opslaan in session state voor de AI tab
                st.session_state.df_top = df_final.sort_values(by='Alpha Score', ascending=False).head(50)
            st.success("Screener voltooid!")

    if not st.session_state.df_top.empty:
        df_display = st.session_state.df_top.copy()
        df_display[['Alpha Score', 'Quality_Score', 'Value_Score', 'Momentum_Score', 'ROE', 'PE_Clean']] = df_display[['Alpha Score', 'Quality_Score', 'Value_Score', 'Momentum_Score', 'ROE', 'PE_Clean']].round(2)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Interactive RRG Scatter Plot")
            fig = px.scatter(
                df_display, x="RS-Ratio", y="RS-Momentum", text="Ticker", size="Alpha Score", color="Quadrant",
                color_discrete_map={'Leading':'green', 'Improving':'blue', 'Lagging':'red', 'Weakening':'orange'},
                title="RRG t.o.v. IWM Benchmark", hover_data=['ROE', 'Beta', 'PE_Clean']
            )
            fig.add_hline(y=100, line_dash="dash", line_color="gray")
            fig.add_vline(x=100, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Top 50 QARP Aandelen")
            st.dataframe(df_display[['Ticker', 'Alpha Score', 'Quadrant', 'ROE', 'Beta']], hide_index=True, use_container_width=True)

with tab3:
    st.header("🤖 Genereer AI Analyse Prompt")
    st.write("Gebruik deze tool om direct een op maat gemaakte prompt te genereren voor je favoriete AI (zoals Gemini of ChatGPT) om diepgaand fundamenteel onderzoek te doen.")
    
    if not st.session_state.df_top.empty:
        selected_ticker = st.selectbox("Selecteer een aandeel uit je Top 50:", st.session_state.df_top['Ticker'].tolist())
        
        # Haal de specifieke data op voor het gekozen aandeel
        stock_data = st.session_state.df_top[st.session_state.df_top['Ticker'] == selected_ticker].iloc[0]
        
        prompt_text = f"""Analyseer aandeel {selected_ticker}. 

Het heeft een ROE van {stock_data['ROE']:.2f}, een Beta van {stock_data['Beta']:.2f}, een P/E ratio van {stock_data['PE_Clean']:.2f} en staat momenteel in het '{stock_data['Quadrant']}' kwadrant van de Relative Rotation Graph. 

Gebruik de principes van Benjamin Graham (Value Investing) en de "Quality minus Junk" theorie van Asness (2018) om de fundamentele gezondheid en het opwaarts potentieel van dit aandeel in kaart te brengen voor een belegger met een lange horizon. Wat zijn de specifieke bedrijfsrisico's en katalysatoren?"""

        st.text_area("Kopieer deze tekst en plak hem in je AI assistent:", value=prompt_text, height=200)
    else:
        st.info("Draai eerst de screener in de 'Screener & Resultaten' tab om aandelen te kunnen selecteren.")

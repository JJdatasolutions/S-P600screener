import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import math
from datetime import datetime, timedelta

# --- 1. CONFIGURATIE & CACHING ---
st.set_page_config(page_title="Scientific Small-Cap Screener", layout="wide")

@st.cache_data(ttl=3600)
def get_benchmark_data(ticker="IWM", periods=200):
    """Haalt benchmark data op voor RRG berekening."""
    end = datetime.today()
    start = end - timedelta(days=periods)
    data = yf.download(ticker, start=start, end=end, progress=False)['Adj Close']
    return data

def get_test_tickers():
    """Geeft een testset van S&P 600 tickers om yfinance time-outs te voorkomen."""
    # In productie: vervang dit door pd.read_html() van een S&P 600 Wikipedia pagina
    return ['AAON', 'AAT', 'ABCB', 'ABM', 'ADTN', 'AEIS', 'AGYS', 'ALRM', 'AMWD', 'ATNI']

# --- 2. WETENSCHAPPELIJKE FUNCTIES ---

def calculate_quality_metrics(ticker_symbol):
    """Berekent de Quality Score (Profitability, Safety)."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Ophalen fundamenten (met default fallback naar NaN of penalty waarden)
        market_cap = info.get('marketCap', np.nan)
        roe = info.get('returnOnEquity', 0)
        beta = info.get('beta', 1.5) # Default hoge beta = safety penalty
        
        # Gross Profitability = Gross Profit / Total Assets (Vaak lastig via yf.info, we benaderen via ROE & Marges)
        gross_margins = info.get('grossMargins', 0)
        
        # Quality Score Logica (versimpeld voor yfinance data)
        # Hoge ROE is goed, Hoge Marges is goed, Lage Beta is goed (Safety)
        profitability_score = (roe * 100) + (gross_margins * 100)
        safety_score = max(0, (2.0 - beta) * 50) # Betting Against Beta: Beta < 1 krijgt hogere score
        
        quality_score = profitability_score + safety_score
        
        return {
            'Ticker': ticker_symbol,
            'Market Cap': market_cap,
            'ROE': roe,
            'Beta': beta,
            'Quality Score': quality_score
        }
    except Exception as e:
        return None

def calculate_rrg_scores(df_prices, benchmark_prices, window=14):
    """Berekent JdK RS-Ratio, RS-Momentum, Heading en Distance."""
    # Relatieve prijs t.o.v. benchmark
    rel_price = df_prices / benchmark_prices
    
    # RS-Ratio (Gesmoothe relatieve sterkte)
    rs_ratio = 100 + ((rel_price / rel_price.rolling(window).mean()) - 1) * 100
    
    # RS-Momentum (Rate of change van RS-Ratio)
    rs_momentum = 100 + ((rs_ratio / rs_ratio.rolling(window).mean()) - 1) * 100
    
    # We pakken de meest recente waarden
    current_ratio = rs_ratio.iloc[-1]
    current_mom = rs_momentum.iloc[-1]
    
    # Bepaal Distance & Heading
    # Distance = wortel van de kwadratensom (afstand tot middelpunt 100,100)
    distance = math.sqrt((current_ratio - 100)**2 + (current_mom - 100)**2)
    
    # Heading (Hoek in graden, 0 tot 360)
    angle_rad = math.atan2(current_mom - 100, current_ratio - 100)
    heading = math.degrees(angle_rad)
    if heading < 0:
        heading += 360
        
    # Sweet Spot multiplier (45 graden is perfect noordoost = maximale score)
    # We belonen headings in het Noordoostelijke kwadrant (0-90 graden)
    sweet_spot_multiplier = math.cos(math.radians(heading - 45)) if 0 <= heading <= 90 else 0.1
    
    return current_ratio, current_mom, distance, heading, sweet_spot_multiplier

def filter_small_cap_quality(df_results, max_market_cap=3e9):
    """Filtert op marktkapitalisatie (Size) en sorteert op Quality."""
    # Filter 'Junk' eruit en houd alleen 'Small' over
    filtered = df_results[(df_results['Market Cap'] <= max_market_cap) & 
                          (df_results['Quality Score'] > df_results['Quality Score'].median())]
    return filtered.sort_values(by='Alpha Score', ascending=False)


# --- 3. STREAMLIT UI ---

st.title("🔬 Scientific Small-Cap Quality Screener")
st.markdown("*Gebaseerd op 'Size Matters, if You Control Your Junk' (Asness et al., 2018)*")

tab1, tab2, tab3 = st.tabs(["Methodologie", "Screener", "AI Analyst Prompt"])

with tab1:
    st.header("Wetenschappelijke Context")
    st.write("""
    Het 'Size Premium' (Banz, 1981) stelt dat kleine bedrijven grote bedrijven verslaan. Echter, dit effect lijkt vaak te verdwijnen in moderne markten. 
    Asness et al. (2018) toonden aan dat dit komt door de invloed van 'Junk' aandelen (lage winstgevendheid, hoge schulden, hoge volatiliteit).
    
    **Dit systeem gebruikt drie pijlers:**
    1. **Size:** Filtert onder de grens van small-caps (bijv. < $3 Miljard).
    2. **Quality (QMJ):** Berekent een score op basis van Return on Equity en Betting Against Beta (lage volatiliteit).
    3. **Momentum (RRG):** Gebruikt Relative Rotation Graphs om de fase van de cyclus te bepalen.
    """)
    st.latex(r"Alpha \ Score = Distance \times \cos(Heading - 45^\circ) \times Quality \ Score")

with tab2:
    st.header("Live Screener")
    st.warning("Let op: Deze demo gebruikt een kleine subset van de S&P 600 om time-outs bij Yahoo Finance te voorkomen. Voor de volledige S&P 600 is een lokale database vereist.")
    
    if st.button("Start Screening"):
        tickers = get_test_tickers()
        benchmark_prices = get_benchmark_data("IWM") # Russell 2000 ETF als benchmark
        
        results = []
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            # 1. Quality Metrics
            q_metrics = calculate_quality_metrics(ticker)
            
            # 2. Prijs Data & RRG
            end = datetime.today()
            start = end - timedelta(days=200)
            prices = yf.download(ticker, start=start, end=end, progress=False)['Adj Close']
            
            if q_metrics and not prices.empty and len(prices) == len(benchmark_prices):
                ratio, mom, distance, heading, ss_mult = calculate_rrg_scores(prices, benchmark_prices)
                
                # Bereken Alpha Score
                alpha_score = distance * ss_mult * q_metrics['Quality Score']
                
                results.append({
                    'Ticker': ticker,
                    'Market Cap': q_metrics['Market Cap'],
                    'Quality Score': q_metrics['Quality Score'],
                    'RS-Ratio': ratio,
                    'RS-Momentum': mom,
                    'Distance': distance,
                    'Heading': heading,
                    'Alpha Score': alpha_score
                })
            
            progress_bar.progress((i + 1) / len(tickers))
            
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            st.subheader("RRG Scatter Plot (Relative Rotation Graph)")
            # RRG Plotly Visualisatie
            fig = px.scatter(
                df_results, 
                x="RS-Ratio", 
                y="RS-Momentum", 
                text="Ticker",
                size="Quality Score",
                color="Alpha Score",
                color_continuous_scale="Viridis",
                title="RRG t.o.v. IWM Benchmark",
                labels={"RS-Ratio": "Relatieve Sterkte (RS-Ratio)", "RS-Momentum": "Relatief Momentum (RS-Momentum)"}
            )
            # Voeg kwadrantlijnen toe
            fig.add_hline(y=100, line_dash="dash", line_color="gray")
            fig.add_vline(x=100, line_dash="dash", line_color="gray")
            # Annotaties voor kwadranten
            fig.add_annotation(x=105, y=105, text="Leading (Noordoost)", showarrow=False)
            fig.add_annotation(x=95, y=105, text="Improving (Noordwest)", showarrow=False)
            fig.add_annotation(x=95, y=95, text="Lagging (Zuidwest)", showarrow=False)
            fig.add_annotation(x=105, y=95, text="Weakening (Zuidoost)", showarrow=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🏆 Alpha Picks (High Quality Small-Caps)")
            df_alpha = filter_small_cap_quality(df_results)
            
            # Formatteer voor weergave
            df_display = df_alpha.copy()
            df_display['Market Cap'] = df_display['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B")
            df_display['Quality Score'] = df_display['Quality Score'].round(2)
            df_display['Alpha Score'] = df_display['Alpha Score'].round(2)
            df_display['Heading'] = df_display['Heading'].round(1)
            
            st.dataframe(df_display[['Ticker', 'Market Cap', 'Quality Score', 'Heading', 'Alpha Score']], hide_index=True)

with tab3:
    st.header("🤖 AI Analyst Prompt")
    st.write("Kopieer de resultaten uit de tabel hierboven en voeg ze samen met deze prompt om een AI de data te laten interpreteren.")
    
    prompt_text = """
    Je bent een Senior Quantitative Analyst. Analyseer de volgende data van mijn 'Scientific Small-Cap Quality Screener' (gebaseerd op Asness et al., 2018). 
    
    De 'Alpha Score' is berekend op basis van Quality (ROE, Marges, Beta) vermenigvuldigd met de Distance en de 'Sweet Spot' Heading (0-90 graden) in de RRG.
    
    Hier is de data van de top aandelen:
    [PLAK HIER JE DATAFRAME]
    
    Beantwoord de volgende vragen:
    1. Welke 2 aandelen vertonen de krachtigste combinatie van Quality en Momentum ('Leading' kwadrant)?
    2. Zijn er potentiële 'Value Traps' in de lijst (aandelen met een hoge Quality, maar een negatief RS-Momentum richting 'Lagging')?
    3. Geef een kort, actiegericht oordeel over de spreiding van sectoren binnen deze resultaten.
    """
    st.code(prompt_text, language="text")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta

# Load stock symbols
@st.cache_data
def load_stock_symbols():
    # In a real app, you would load from your Excel file
    # For this example, I'll use a sample list
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", 
        "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "LT.NS", "SBIN.NS", "BAJFINANCE.NS",
        "ASIANPAINT.NS", "HDFC.NS", "MARUTI.NS", "TITAN.NS",
        "NESTLEIND.NS", "ONGC.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS"
    ]

def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def calculate_technical_indicators(df):
    # Calculate MACD
    indicator_macd = MACD(df['Close'])
    df['MACD'] = indicator_macd.macd()
    df['MACD_signal'] = indicator_macd.macd_signal()
    df['MACD_hist'] = indicator_macd.macd_diff()
    
    # Calculate RSI
    indicator_rsi = RSIIndicator(df['Close'], window=14)
    df['RSI'] = indicator_rsi.rsi()
    
    # Calculate Bollinger Bands
    indicator_bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = indicator_bb.bollinger_hband()
    df['BB_middle'] = indicator_bb.bollinger_mavg()
    df['BB_lower'] = indicator_bb.bollinger_lband()
    
    # Calculate Simple Moving Averages
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    
    return df

def generate_trading_signals(df):
    signals = []
    
    # MACD Signal
    if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
        signals.append("MACD: Bullish (MACD above signal line)")
    else:
        signals.append("MACD: Bearish (MACD below signal line)")
    
    # RSI Signal
    if df['RSI'].iloc[-1] > 70:
        signals.append("RSI: Overbought (>70)")
    elif df['RSI'].iloc[-1] < 30:
        signals.append("RSI: Oversold (<30)")
    else:
        signals.append("RSI: Neutral (30-70)")
    
    # Bollinger Bands Signal
    if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1]:
        signals.append("Bollinger Bands: Potential Buy (Price below lower band)")
    elif df['Close'].iloc[-1] > df['BB_upper'].iloc[-1]:
        signals.append("Bollinger Bands: Potential Sell (Price above upper band)")
    else:
        signals.append("Bollinger Bands: Neutral (Price within bands)")
    
    # Moving Averages Signal
    if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
        signals.append("Moving Averages: Golden Cross (50-day above 200-day)")
    else:
        signals.append("Moving Averages: Death Cross (50-day below 200-day)")
    
    return signals

def plot_stock_data(df, ticker):
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_upper'],
        line=dict(color='rgba(255, 0, 0, 0.5)'),
        name='Upper Band'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_middle'],
        line=dict(color='rgba(0, 0, 255, 0.5)'),
        name='Middle Band'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_lower'],
        line=dict(color='rgba(0, 255, 0, 0.5)'),
        name='Lower Band'
    ))
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_50'],
        line=dict(color='orange', width=1.5),
        name='50-day SMA'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_200'],
        line=dict(color='purple', width=1.5),
        name='200-day SMA'
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_macd(df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        line=dict(color='blue', width=2),
        name='MACD'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_signal'],
        line=dict(color='red', width=2),
        name='Signal Line'
    ))
    
    # Histogram
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_hist'],
        marker_color=colors,
        name='Histogram'
    ))
    
    fig.update_layout(
        title='MACD Indicator',
        xaxis_title='Date',
        yaxis_title='Value',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        line=dict(color='purple', width=2),
        name='RSI'
    ))
    
    # Add overbought and oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Stock Technical Analysis", layout="wide")
    
    st.title("📈 Stock Technical Analysis Dashboard")
    st.write("Analyze stock movements using technical indicators")
    
    # Load stock symbols
    stock_symbols = load_stock_symbols()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks", 
        stock_symbols,
        default=["RELIANCE.NS", "TCS.NS"]
    )
    
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    selected_period = st.sidebar.selectbox(
        "Select Time Period",
        list(period_options.keys()),
        index=3
    )
    
    # Main content
    if not selected_stocks:
        st.warning("Please select at least one stock from the sidebar")
        return
    
    for ticker in selected_stocks:
        st.subheader(f"Analysis for {ticker}")
        
        # Get stock data
        try:
            df = get_stock_data(ticker, period_options[selected_period])
            if df.empty:
                st.error(f"No data available for {ticker}")
                continue
                
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Display current price
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_close
            percent_change = (price_change / prev_close) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{current_price:,.2f}")
            col2.metric("Daily Change", f"₹{price_change:,.2f}", f"{percent_change:.2f}%")
            
            # Generate trading signals
            signals = generate_trading_signals(df)
            
            # Display signals
            with st.expander("Trading Signals", expanded=True):
                for signal in signals:
                    if "Bullish" in signal or "Buy" in signal or "Oversold" in signal:
                        st.success(signal)
                    elif "Bearish" in signal or "Sell" in signal or "Overbought" in signal:
                        st.error(signal)
                    else:
                        st.info(signal)
            
            # Plot charts
            plot_stock_data(df, ticker)
            
            col1, col2 = st.columns(2)
            with col1:
                plot_macd(df)
            with col2:
                plot_rsi(df)
                
            st.divider()
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides technical analysis of stocks using indicators like "
        "MACD, RSI, Bollinger Bands, and Moving Averages. "
        "Use it to identify potential trading opportunities."
    )

if __name__ == "__main__":
    main()

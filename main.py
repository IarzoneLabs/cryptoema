import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nest_asyncio
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import streamlit as st
from binance.client import Client
from datetime import datetime, timedelta
from binance.client import Client
from sklearn.linear_model import LinearRegression
from PIL import Image

nest_asyncio.apply()

# Set your Binance API credentials
API_KEY = 'L20wkbLFpCUb073Idp77gPFHVIQCvZMwWqCgr2gUI5TmKMB4KJawLm6XrsI2doLG'
API_SECRET = 'RJxUalW7TEZRQvlfAyLAzGDg2iWs8r5ArJVEzyF7GM8IvLhLttFos8AkOGBhSDTp'

# Initialize Binance client
client = Client(API_KEY, API_SECRET)


def fetch_crypto_data(symbol, interval, days=2):
    """Fetch historical data for a given symbol and interval."""
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    candles = client.get_klines(symbol=symbol,
                                interval=interval,
                                startTime=since)
    df = pd.DataFrame(candles,
                      columns=[
                          'timestamp', 'open', 'high', 'low', 'close',
                          'volume', 'close_time', 'quote_asset_volume',
                          'number_of_trades', 'taker_buy_base_asset_volume',
                          'taker_buy_quote_asset_volume', 'ignore'
                      ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close',
        'volume']] = df[['open', 'high', 'low', 'close',
                         'volume']].astype(float)
    return df


def calculate_regression(df):
    """Calculate regression analysis to determine trend."""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['close'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    slope = model.coef_[0]
    return 'Positive' if slope > 0 else 'Negative' if slope < 0 else 'Flat', y_pred


def plot_regression(df, y_pred, height=1000):
    """Plot regression analysis."""
    plt.figure(figsize=(12, height / 100))
    plt.plot(df['timestamp'], df['close'], label='BTC Price', color='blue')
    plt.plot(df['timestamp'],
             y_pred,
             label='Regression Line',
             color='red',
             linestyle='--')
    plt.title('BTC Price with Regression Line')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    st.pyplot(plt)


def calculate_price_reaction_velocity(df, ema_col, k=5):
    """Calculate Price Reaction Velocity."""
    touches = df[(df['close'].shift(1) > df[ema_col]) &
                 (df['close'] <= df[ema_col]) |
                 (df['close'].shift(1) < df[ema_col]) &
                 (df['close'] >= df[ema_col])]
    velocities = []
    for idx in touches.index:
        if idx + k < len(df):
            reaction = abs(df.loc[idx + k, 'close'] - df.loc[idx, 'close'])
            velocities.append(reaction)
    return np.mean(velocities) if velocities else 0


def calculate_bounce_efficiency(df, ema_col, k=5, threshold=0.01):
    """Calculate Bounce Efficiency."""
    touches = df[(df['close'].shift(1) > df[ema_col]) &
                 (df['close'] <= df[ema_col]) |
                 (df['close'].shift(1) < df[ema_col]) &
                 (df['close'] >= df[ema_col])]
    significant_bounces = 0
    for idx in touches.index:
        if idx + k < len(df):
            reaction = abs(df.loc[idx + k, 'close'] - df.loc[idx, 'close'])
            if reaction > threshold * df.loc[idx, 'close']:
                significant_bounces += 1
    return significant_bounces / len(touches) if len(touches) > 0 else 0


def identify_best_ma_ema(df):
    """Identify the best performing MA or EMA using combined metrics."""
    results = []

    # Loop through EMA periods in the specified range
    for period in range(15, 91, 2):
        # Calculate EMA
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Calculate metrics
        velocity = calculate_price_reaction_velocity(df, f'EMA_{period}')
        efficiency = calculate_bounce_efficiency(df, f'EMA_{period}')

        # Handle missing or invalid values for velocity/efficiency
        if velocity is None or efficiency is None:
            velocity = 0
            efficiency = 0

        results.append({
            'Period': period,
            'Metric': f'EMA_{period}',
            'Velocity': velocity,
            'Efficiency': efficiency,
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Scale velocity and efficiency between 0 and 1
    results_df['Scaled Velocity'] = (results_df['Velocity'] - results_df['Velocity'].min()) / \
                                    (results_df['Velocity'].max() - results_df['Velocity'].min())
    results_df['Scaled Efficiency'] = (results_df['Efficiency'] - results_df['Efficiency'].min()) / \
                                       (results_df['Efficiency'].max() - results_df['Efficiency'].min())

    # Calculate the combined score with weights: 41% velocity, 59% efficiency
    results_df['Combined Score'] = 0.41 * results_df[
        'Scaled Velocity'] + 0.59 * results_df['Scaled Efficiency'].fillna(0)

    # Identify the best EMA based on the highest combined score
    best = results_df.sort_values('Combined Score', ascending=False).iloc[0]
    return best['Metric'], results_df


##############################################################################################
def calculate_correlation_and_sensitivity(base_df, target_df, decimals=4):
    """Calculate correlation and sensitivity between base and target dataframes."""
    # Calculate percentage changes
    base_pct_change = base_df['close'].pct_change().dropna()
    target_pct_change = target_df['close'].pct_change().dropna()

    # Align indices
    aligned_data = pd.concat([base_pct_change, target_pct_change],
                             axis=1).dropna()
    aligned_data.columns = ['Base_pct_change', 'Target_pct_change']

    # Check if we have any data to work with
    if len(aligned_data) == 0:
        return 0.0, 0.0, 0.0, 0.0  # Return default values if no data

    # Calculate correlation coefficient
    correlation = aligned_data.corr().iloc[0, 1]

    # Regression for sensitivity (beta coefficient)
    X = aligned_data['Base_pct_change'].values.reshape(-1, 1)
    y = aligned_data['Target_pct_change'].values
    reg = LinearRegression()
    try:
        reg.fit(X, y)
        sensitivity = reg.coef_[0]
    except ValueError:
        sensitivity = 0.0

    # Regression for trend direction score
    indices = np.arange(len(target_df['close'])).reshape(
        -1, 1)  # Use index as a proxy for time
    target_close = target_df['close'].values
    trend_reg = LinearRegression()
    try:
        trend_reg.fit(indices, target_close)
        trend_direction_score = trend_reg.coef_[0]  # Slope of the trend line
    except ValueError:
        trend_direction_score = 0.0

    # Scale correlation, sensitivity, and trend direction score to 0-1
    correlation_scaled = (correlation +
                          1) / 2  # Assuming correlation is in [-1, 1]
    sensitivity_scaled = sensitivity / (abs(sensitivity) +
                                        1) if sensitivity != 0 else 0
    trend_direction_scaled = (trend_direction_score - min(trend_direction_score, 0)) / \
                            (max(trend_direction_score, 0) + abs(min(trend_direction_score, 0))) \
                            if trend_direction_score != 0 else 0

    # Weighted combined score
    combined_score = (0.31 * correlation_scaled + 0.32 * sensitivity_scaled +
                      0.37 * trend_direction_scaled)

    # Format results to fixed decimals
    correlation = round(correlation, decimals)
    sensitivity = round(sensitivity, decimals)
    trend_direction_score = round(trend_direction_score, decimals)
    combined_score = round(combined_score, decimals)

    return correlation, sensitivity, trend_direction_score, combined_score


##############################################################################################


def suggest_trades(base_df, target_df, best_metric, trend):
    """Suggest long or short trades based on trend with detailed conditions."""
    high = target_df['high'].max()
    low = target_df['low'].min()
    variance = (high - low) / 5  # Adjusted variance division
    take_profit = variance
    stop_loss = take_profit / 4

    target_df['signal'] = None
    target_df['take_profit'] = None
    target_df['stop_loss'] = None

    latest_row = target_df.iloc[-1]

    if trend == 'Positive' and latest_row['close'] > latest_row[best_metric]:
        target_df.loc[target_df.index[-1], 'signal'] = 'Long'
        target_df.loc[target_df.index[-1],
                      'take_profit'] = latest_row[best_metric] + take_profit
        target_df.loc[target_df.index[-1],
                      'stop_loss'] = latest_row[best_metric] - stop_loss
    elif trend == 'Negative' and latest_row['close'] < latest_row[best_metric]:
        target_df.loc[target_df.index[-1], 'signal'] = 'Short'
        target_df.loc[target_df.index[-1],
                      'take_profit'] = latest_row[best_metric] - take_profit
        target_df.loc[target_df.index[-1],
                      'stop_loss'] = latest_row[best_metric] + stop_loss

    return target_df


def plot_candlestick_with_signals(df,
                                  metric_list,
                                  title,
                                  plot_positions=True,
                                  future_time_minutes=300,
                                  height=500):
    """Plot candlestick chart with optional trade signals and multiple EMAs/MAs."""
    future_time = df['timestamp'].iloc[-1] + timedelta(
        minutes=future_time_minutes)

    fig = go.Figure(data=[
        go.Candlestick(x=df['timestamp'],
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'])
    ])

    for metric in metric_list:
        fig.add_trace(
            go.Scatter(x=df['timestamp'],
                       y=df[metric],
                       mode='lines',
                       name=metric))

    if plot_positions and 'signal' in df.columns:
        last_signal_row = df.iloc[-1]
        if last_signal_row['signal'] == 'Long':
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'],
                          x1=future_time,
                          y0=last_signal_row[metric_list[0]],
                          y1=last_signal_row['take_profit'],
                          fillcolor="green",
                          opacity=0.2,
                          line_width=0)
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'],
                          x1=future_time,
                          y0=last_signal_row[metric_list[0]],
                          y1=last_signal_row['stop_loss'],
                          fillcolor="red",
                          opacity=0.2,
                          line_width=0)
        elif last_signal_row['signal'] == 'Short':
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'],
                          x1=future_time,
                          y0=last_signal_row[metric_list[0]],
                          y1=last_signal_row['stop_loss'],
                          fillcolor="red",
                          opacity=0.2,
                          line_width=0)
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'],
                          x1=future_time,
                          y0=last_signal_row[metric_list[0]],
                          y1=last_signal_row['take_profit'],
                          fillcolor="green",
                          opacity=0.2,
                          line_width=0)

    fig.update_layout(title=title,
                      xaxis_title='Time',
                      yaxis_title='Price',
                      height=height,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


# Streamlit App Configuration
st.set_page_config(layout="wide", page_title="Crypto EMA Analysis")
st.markdown("""
<style>
    .main {background-color: #f2f2f2;}
    .stCard {
        border: none;
        text-align: center;
        margin: 5px;
        padding: 15px;
        border-radius: 5px;
    }
    .yellowCard {background-color: #ffffcc;}
    .greenCard {background-color: #ccffcc;}
    .redCard {background-color: #ffcccc;}
    .whiteCard {background-color: #ffffff;}
    .subheader-centered {text-align: center;}
</style>
""",
            unsafe_allow_html=True)

# Sidebar

st.sidebar.header("Settings")

# Sidebar: Add input for days
days = st.sidebar.number_input("Number of Days to Fetch Data:",
                               min_value=1,
                               max_value=30,
                               value=2)
interval = st.sidebar.selectbox("Select Interval:",
                                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                                index=2)
calculate_button = st.sidebar.button("Calculate Now")

# Main Section
st.title("Best EMA for BTC and Crypto Positions Suggestions", anchor="center")

if calculate_button:
    # Fetch BTC data
    btc_df = fetch_crypto_data("BTCUSDT", interval, days=days)
    btc_trend, y_pred = calculate_regression(btc_df)

    # Find top EMAs
    best_metric, all_metrics_df = identify_best_ma_ema(btc_df)
    top_3_metrics = all_metrics_df.sort_values(
        "Combined Score", ascending=False).head(3)["Metric"].tolist()

    # Fetch and calculate correlation and sensitivity for other coins
    coins = [
        'USDTTRY', 'USDTARS', 'USDTCOP', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT',
        'USDCUSDT', 'SUIUSDT', 'SOLUSDT', 'XRPUSDT', 'FDUSDUSDT', 'BIOUSDT',
        'PEPEUSDT', 'ADAUSDT', 'ENAUSDT', 'TRXUSDT', 'XLMUSDT', 'FDUSDTRY',
        'BNBUSDT', 'PENGUUSDT', 'HBARUSDT', 'USDTBRL', 'USUALUSDT', 'PHAUSDT',
        'PNUTUSDT', 'LINKUSDT', 'NEIROUSDT', 'ARBUSDT', 'SHIBUSDT', 'AVAXUSDT',
        'WLDUSDT', 'BONKUSDT', 'MOVEUSDT', 'AAVEUSDT', 'RSRUSDT', 'WIFUSDT',
        'SUSHIUSDT', 'UNIUSDT', 'LTCUSDT', 'STGUSDT', 'TAOUSDT', 'FLOKIUSDT',
        'FTMUSDT', 'GALAUSDT', 'TROYUSDT', 'CRVUSDT', 'ACTUSDT', 'DOTUSDT',
        'SANDUSDT', 'ZENUSDT'
    ]

    results = []
    for coin in coins:
        target_df = fetch_crypto_data(coin, '15m', days=1)
        base_df = fetch_crypto_data('BTCUSDT', '15m', days=1)
        correlation, sensitivity, trend_direction_score, combined_score = calculate_correlation_and_sensitivity(
            base_df, target_df, 4)
        results.append({
            'Coin': coin,
            'Correlation': correlation,
            'Sensitivity': sensitivity,
            'Trend Direction Score': trend_direction_score,
            'Combined Score': combined_score
        })

    results_df = pd.DataFrame(results)

    # Getting the best coin:
    if btc_trend == 'Positive':
        # Sort by highest Combined Score
        best_coin = results_df.sort_values(['Combined Score'],
                                           ascending=False).iloc[0]['Coin']
    else:
        # Sort by lowest trend_direction_score, then highest correlation, then highest sensitivity
        best_coin = results_df.sort_values(
            ['Trend Direction Score', 'Correlation', 'Sensitivity'],
            ascending=[True, False, False]).iloc[0]['Coin']

    # Calculate the EMA for the best metric
    best_coin_df = fetch_crypto_data(best_coin, interval)
    best_coin_df[best_metric] = best_coin_df['close'].ewm(span=int(
        best_metric.split('_')[1]),
                                                          adjust=False).mean()

    # Suggest trades for the target coin
    best_coin_df = suggest_trades(btc_df, best_coin_df, best_metric, btc_trend)

    # Extract values for the cards
    target_price = best_coin_df["close"].iloc[-1]
    btc_price = btc_df["close"].iloc[-1]
    price_to_buy_or_sell = best_coin_df.iloc[-1][best_metric]

    # Ensure take_profit has a valid numeric value
    take_profit = best_coin_df.iloc[-1].get('take_profit', None)
    if take_profit is None:
        take_profit = 0.0  # Default fallback value for take_profit

    # Ensure stop_loss has a valid numeric value
    stop_loss = best_coin_df.iloc[-1].get('stop_loss', None)
    if stop_loss is None:
        stop_loss = 0.0  # Default fallback value for stop_loss

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f'<div class="stCard yellowCard">Latest BTC Price<br><b>${btc_price:,.0f}</b></div>',
            unsafe_allow_html=True)

    with col2:
        trend_color = "greenCard" if btc_trend == "Positive" else "redCard"
        st.markdown(
            f'<div class="stCard {trend_color}">BTC Trend<br><b>{btc_trend}</b></div>',
            unsafe_allow_html=True)

    with col3:
        st.markdown(
            f'<div class="stCard whiteCard">Best Performing Coin<br><b>{best_coin}</b></div>',
            unsafe_allow_html=True)

    with col4:
        st.markdown(
            f'<div class="stCard greenCard">Price to Buy or Sell<br><b>${price_to_buy_or_sell:,.6f}</b></div>',
            unsafe_allow_html=True)

    with col5:
        st.markdown(
            f'<div class="stCard greenCard">Take Profit Price<br><b>${take_profit:,.6f}</b></div>',
            unsafe_allow_html=True)

    with col6:
        st.markdown(
            f'<div class="stCard redCard">Stop Loss<br><b>${stop_loss:,.6f}</b></div>',
            unsafe_allow_html=True)

    # Section 1
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(
            '<div class="subheader-centered"><h3>Scatterplot: BTC Regression Line</h3></div>',
            unsafe_allow_html=True)
        plot_regression(btc_df, y_pred, height=1120)

    with col2:
        st.markdown(
            '<div class="subheader-centered"><h3>Top 10 EMAs by Combined Score</h3></div>',
            unsafe_allow_html=True)
        st.dataframe(all_metrics_df[[
            'Period', 'Metric', 'Velocity', 'Efficiency', 'Combined Score'
        ]].sort_values("Combined Score", ascending=False).head(10),
                     height=400)

    with col3:
        st.markdown(
            '<div class="subheader-centered"><h3>BTC Candlestick with EMAs</h3></div>',
            unsafe_allow_html=True)
        st.plotly_chart(plot_candlestick_with_signals(btc_df,
                                                      top_3_metrics,
                                                      "BTC with Top 3 EMAs",
                                                      height=400),
                        use_container_width=True)

    # Section 2
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            '<div class="subheader-centered"><h3>Top Coins by Combined Score</h3></div>',
            unsafe_allow_html=True)
        st.dataframe(results_df.sort_values("Combined Score", ascending=False),
                     height=600)

    with col2:
        st.markdown(
            f'<div class="subheader-centered"><h3>{best_coin} Candlestick with Suggested Position</h3></div>',
            unsafe_allow_html=True)
        st.plotly_chart(plot_candlestick_with_signals(
            best_coin_df, [best_metric],
            f"{best_coin} with Suggested Position",
            future_time_minutes=300,
            height=600),
                        use_container_width=True)

# Streamlit run web.py

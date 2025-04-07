# Full content for ui/streamlit_app.py

import streamlit as st
import sys
import os
import pandas as pd
import datetime
import plotly.graph_objects as go
import random
import time

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_service import ApiService
from src.models.data_models import PredictionRequest
from src.data_acquisition import StockDataFetcher # Ensure this import is correct

# --- Session State Initialization --- #
# Prediction related
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = {}
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None
if 'current_timeframe' not in st.session_state:
    st.session_state.current_timeframe = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'include_confidence' not in st.session_state:
    st.session_state.include_confidence = True
if 'symbols_cache' not in st.session_state:
    st.session_state.symbols_cache = None
    st.session_state.symbols_cache_time = 0

# Trading related
if 'trading_balance' not in st.session_state:
    st.session_state.trading_balance = 100000.0  # Initial balance ₹100,000
if 'portfolio' not in st.session_state:
    # Portfolio format: {symbol: {'quantity': qty, 'avg_price': avg_cost_per_share, 'cost_basis': total_cost}}
    st.session_state.portfolio = {}
if 'order_history' not in st.session_state:
    st.session_state.order_history = []

# --- Constants and Config --- #
CACHE_DURATION = 900 # Cache duration in seconds (15 minutes)
TRANSACTION_COST_PERCENTAGE = 0.001 # 0.1% for trading simulation

# --- Initialize Services --- #
try:
    api_service = ApiService()
    fetcher = StockDataFetcher() # Initialize data fetcher once
except NameError:
    st.error("Critical Error: Could not initialize ApiService or StockDataFetcher. Check imports and dependencies.")
    st.stop() # Stop execution if core services fail
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()

# --- Page Config and Styling --- #
st.set_page_config(
    page_title="Stock Price Prediction - Indian Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling (ensure CSS classes match usage below)
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #1E3A8A; margin-top: 2rem; }
    .disclaimer { font-size: 0.8rem; color: #6B7280; font-style: italic; }
    .prediction-card { background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1E3A8A; margin-bottom: 1rem; }
    .search-container { padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 10px; }
    .history-container { margin-top: 15px; padding: 10px; background-color: #f0f2f6; border-radius: 5px; }
    .indicator-positive { color: green; font-weight: 500; }
    .indicator-negative { color: red; font-weight: 500; }
    .indicator-neutral { color: gray; font-weight: 500; }
    .news-item { margin-bottom: 8px; padding: 5px; border-left: 3px solid #ddd; }
    .news-positive { border-left-color: green; }
    .news-negative { border-left-color: red; }
    .news-neutral { border-left-color: gray; }
    .trading-section { margin-top: 2rem; padding: 1rem; background-color: #f0f8ff; border-radius: 0.5rem; }
    /* Ensure DataFrames look good */
    .stDataFrame { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- Header --- #
st.markdown("<h1 class='main-header'>Stock Price Prediction Platform</h1>", unsafe_allow_html=True)
st.markdown("### Analysis, Forecasting & Trading Simulator for Indian Markets")
st.markdown("""
This platform uses machine learning for stock price predictions and includes a mock trading simulator
to practice investing with virtual currency.
""")

# --- Sidebar --- #
st.sidebar.title("Controls")

# Stock Search
st.sidebar.markdown("<div class='search-container'>", unsafe_allow_html=True)
search_query = st.sidebar.text_input("Search Stocks", value=st.session_state.search_query)
if search_query != st.session_state.search_query:
    st.session_state.search_query = search_query
    st.session_state.symbols_cache = None # Clear cache on new search
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Get Symbols (Cached)
@st.cache_data(ttl=CACHE_DURATION)
def get_filtered_symbols_cached(query=""):
    try:
        if query:
            return api_service.search_symbols(query)
        else:
            return api_service.get_available_symbols()
    except Exception as e:
        st.sidebar.error(f"Error fetching symbols: {e}")
        return None

with st.sidebar.container():
    symbol_list_data = get_filtered_symbols_cached(search_query)
    symbols = {}
    if symbol_list_data and hasattr(symbol_list_data, 'symbols'):
        symbols = {f"{s.symbol} - {s.name}": s.symbol for s in symbol_list_data.symbols}
    else:
        st.sidebar.warning("No stocks found or error fetching list.")
        symbols = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN"} # Fallback

# Stock Selection
selected_symbol_display = st.sidebar.selectbox(
    "Select Stock or Index",
    options=list(symbols.keys()),
    index=0 # Ensure options list is not empty before setting index
)
# Make default value retrieval safer
default_symbol = list(symbols.values())[0] if symbols else "^NSEI" # Use first symbol if available, else fallback
selected_symbol = symbols.get(selected_symbol_display, default_symbol)

# Prediction Timeframe
timeframe_options = {
    "Short Term (7 days)": "short_term",
    "Medium Term (30 days)": "medium_term",
    "Long Term (90 days)": "long_term"
}
selected_timeframe_display = st.sidebar.radio(
    "Prediction Timeframe",
    options=list(timeframe_options.keys())
)
selected_timeframe = timeframe_options[selected_timeframe_display]

include_confidence = st.sidebar.checkbox("Include Confidence Intervals", value=st.session_state.include_confidence)

# Chart Interval
interval_options = {
    "1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m", "30 Minutes": "30m",
    "1 Hour": "1h", "1 Day": "1d", "1 Week": "1wk", "1 Month": "1mo"
}
selected_interval_display = st.sidebar.selectbox(
    "Chart Interval",
    options=list(interval_options.keys()),
    index=5 # Default to 1 Day
)
interval_choice = interval_options[selected_interval_display]

# Update session state for current selections
if selected_symbol != st.session_state.current_symbol or selected_timeframe != st.session_state.current_timeframe:
    st.session_state.current_symbol = selected_symbol
    st.session_state.current_timeframe = selected_timeframe
st.session_state.include_confidence = include_confidence # Update confidence state

# Generate Analysis Button Callback
def generate_analysis_callback():
    symbol = st.session_state.current_symbol
    timeframe = st.session_state.current_timeframe
    confidence = st.session_state.include_confidence
    if not symbol or not timeframe:
        st.error("Please select a valid stock and timeframe.")
        return

    prediction_request = PredictionRequest(
        symbol=symbol,
        timeframe=timeframe,
        include_confidence=confidence
    )
    try:
        with st.spinner("Generating prediction..."):
            result = api_service.predict(prediction_request)
        prediction_key = f"{symbol}_{timeframe}"
        st.session_state.prediction_history[prediction_key] = {
            'name': f"{result.symbol} - {result.name}",
            'timestamp': datetime.datetime.now().isoformat(),
            'result': result
        }
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

st.sidebar.button("Generate Analysis", type="primary", on_click=generate_analysis_callback)

# API Status
try:
    api_status = api_service.get_api_status()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    st.sidebar.markdown(f"**Model:** {api_status.model_version}")
    st.sidebar.markdown(f"**API:** {api_status.version} ({api_status.status})")
except Exception as e:
     st.sidebar.warning(f"Could not get API status: {e}")

# Recent Analyses History
if st.session_state.prediction_history:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Recent Analyses")
    history_items = sorted(st.session_state.prediction_history.items(), key=lambda x: x[1]['timestamp'], reverse=True)
    for key, data in history_items[:5]:
        sym, tf = key.split('_', 1)
        if st.sidebar.button(f"{data['name']} ({tf.replace('_', ' ').title()})"):
            st.session_state.current_symbol = sym
            st.session_state.current_timeframe = tf
            st.rerun()

# --- Main Content Area --- #

# Display Prediction Results
current_prediction_key = f"{st.session_state.current_symbol}_{st.session_state.current_timeframe}"
if current_prediction_key in st.session_state.prediction_history:
    display_prediction = st.session_state.prediction_history[current_prediction_key]
    result = display_prediction['result']

    pred_col1, pred_col2 = st.columns([2, 1]) # Layout columns

    with pred_col1:
        st.markdown(f"<h2 class='sub-header'>{result.name} ({result.symbol})</h2>", unsafe_allow_html=True)

        # Price Metrics
        price_m_col1, price_m_col2 = st.columns(2)
        with price_m_col1:
            st.metric("Current Price", f"₹{result.current_price:.2f}")
        with price_m_col2:
            change = result.prediction.change_percentage
            st.metric("Predicted Price", f"₹{result.prediction.predicted_price:.2f}",
                      f"{change:+.2f}%", delta_color="normal" if change >= 0 else "inverse")

        # Prediction Details Card
        with st.container():
             st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
             st.markdown(f"**Target Date:** {result.prediction.target_date}")
             st.markdown(f"**Timeframe:** {result.prediction.timeframe.replace('_', ' ').title()}")
             if result.prediction.confidence_interval:
                 st.markdown("**95% Confidence Interval:** "
                             f"Lower: ₹{result.prediction.confidence_interval.lower:.2f} | "
                             f"Upper: ₹{result.prediction.confidence_interval.upper:.2f}")
             st.markdown("</div>", unsafe_allow_html=True)

        # Prediction Chart (Simplified placeholder - replace with actual logic if needed)
        try:
            # Placeholder chart logic (as before)
            end_date = datetime.datetime.now().date()
            start_date = end_date - datetime.timedelta(days=30)
            dates = pd.date_range(start=start_date, end=end_date)
            future_days = {'short_term': 7, 'medium_term': 30, 'long_term': 90}.get(selected_timeframe, 7)
            future_dates = pd.date_range(start=end_date + datetime.timedelta(days=1), periods=future_days)
            predicted_prices = [result.current_price + (i / future_days) * (result.prediction.predicted_price - result.current_price) for i in range(future_days + 1)]

            fig_pred = go.Figure()
            # Add historical trace (mock)
            fig_pred.add_trace(go.Scatter(x=dates, y=[result.current_price * (1 + random.uniform(-0.02, 0.02)) for _ in dates], mode='lines', name='Historical (Mock)', line=dict(color='#6B7280')))
            # Add prediction trace
            fig_pred.add_trace(go.Scatter(x=future_dates, y=predicted_prices[1:], mode='lines', name='Prediction', line=dict(color='#10B981', dash='dash')))
            # Add confidence interval fill
            if include_confidence and result.prediction.confidence_interval:
                 lower_prices = [result.prediction.confidence_interval.lower] * len(future_dates)
                 upper_prices = [result.prediction.confidence_interval.upper] * len(future_dates)
                 fig_pred.add_trace(go.Scatter(x=future_dates.tolist() + future_dates[::-1].tolist(), y=upper_prices + lower_prices[::-1],
                                            fill='toself', fillcolor='rgba(16, 185, 129, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                                            hoverinfo="skip", showlegend=False, name='Confidence Interval'))

            fig_pred.update_layout(title=f"Price Forecast for {result.symbol}", xaxis_title="Date", yaxis_title="Price (₹)",
                                   hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig_pred, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not generate prediction chart: {e}")

    with pred_col2:
        st.markdown("<h3 class='sub-header'>Market Insights</h3>", unsafe_allow_html=True)

        # Technical Indicators
        st.markdown("#### Technical Indicators")
        if 'technical_indicators' in result.metadata:
            indicators = result.metadata['technical_indicators']
            # Display indicators safely
            for key, value in indicators.items():
                 style_class = ""
                 display_value = f"{value:.2f}" if isinstance(value, (int, float)) else value
                 if key == "RSI (14)" and isinstance(value, (int, float)):
                     if value > 70: style_class = "indicator-negative"
                     elif value < 30: style_class = "indicator-positive"
                     else: style_class = "indicator-neutral"
                 elif key == "MACD" and isinstance(value, (int, float)):
                     style_class = "indicator-positive" if value > 0 else "indicator-negative"

                 st.markdown(f"**{key}:** <span class='{style_class}'>{display_value}</span>", unsafe_allow_html=True)
        else:
            st.markdown("_No technical indicators available._")

        # Market Sentiment
        st.markdown("#### Market Sentiment")
        if 'sentiment' in result.metadata:
            sentiment = result.metadata['sentiment']
            overall = sentiment.get('overall', 'Neutral')
            color = {"Bullish": "green", "Bearish": "red", "Neutral": "gray"}.get(overall, "gray")
            st.markdown(f"**Overall:** <span style='color:{color}'>{overall}</span>", unsafe_allow_html=True)

            st.markdown("##### Recent News")
            news_items = sentiment.get('news_items', [])
            if news_items:
                for news in news_items:
                    s_class = {"Positive": "news-positive", "Negative": "news-negative", "Neutral": "news-neutral"}.get(news['sentiment'], "news-neutral")
                    s_emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}.get(news['sentiment'], "⚪")
                    st.markdown(f"<div class='news-item {s_class}'>{s_emoji} {news['title']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("_No recent news found._")
        else:
            st.markdown("_Sentiment analysis not available._")

        st.markdown("---")
        st.markdown(f"*Analysis generated: {datetime.datetime.fromisoformat(display_prediction['timestamp']).strftime('%Y-%m-%d %H:%M')}*")

    # Disclaimer for prediction
    st.markdown("---")
    st.markdown("<p class='disclaimer'>Prediction Disclaimer: Forecasts are based on historical data and models, not financial advice. Invest responsibly.</p>", unsafe_allow_html=True)

else:
    # Show placeholder if no prediction generated yet
    st.info("👈 Select a stock and timeframe, then click 'Generate Analysis' to view forecast.")
    # Optionally show a default sample chart here

# --- Candlestick Chart --- #
st.markdown("<h2 class='sub-header'>Price Chart</h2>", unsafe_allow_html=True)
try:
    if st.session_state.current_symbol:
        st.markdown(f"#### {st.session_state.current_symbol} - {selected_interval_display} Interval")
        with st.spinner(f"Loading {selected_interval_display} chart data..."):
            # Determine appropriate period based on interval
            chart_period = "1y" # Default
            if interval_choice in ['1m', '5m', '15m', '30m', '1h']: chart_period = "1mo"
            if interval_choice == '1m': chart_period = "7d"

            # Fetch data using the correct method
            chart_data = fetcher.get_historical_data(
                st.session_state.current_symbol,
                period=chart_period,
                interval=interval_choice
            )

            if not chart_data.empty:
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
                    low=chart_data['Low'], close=chart_data['Close'], name='Candlesticks'
                )])
                fig_candle.update_layout(
                    # title=f"Candlestick Chart", # Title provided by markdown above
                    xaxis_title="Time", yaxis_title="Price (₹)",
                    xaxis_rangeslider_visible=False, hovermode="x unified",
                    margin=dict(l=0, r=0, t=20, b=0) # Reduced top margin
                )
                st.plotly_chart(fig_candle, use_container_width=True)
            else:
                st.warning(f"No chart data available for {st.session_state.current_symbol} ({selected_interval_display}, {chart_period}).")
    else:
        st.info("Select a stock to view its price chart.")
except Exception as e:
    st.error(f"Error creating candlestick chart: {str(e)}")

# --- Mock Trading Simulator --- #
st.markdown("<h2 class='sub-header trading-section'>Mock Trading Simulator</h2>", unsafe_allow_html=True)
st.markdown("""
Practice trading with virtual INR currency. Learn about different order types (Market, Limit, Stop-Loss)
and see how simulated transaction costs impact your returns.
""")

# Trading Account Display
trade_col1, trade_col2 = st.columns(2)
with trade_col1:
    st.subheader("Trading Account")
    st.metric("Available Balance", f"₹{st.session_state.trading_balance:,.2f}")

    # Calculate portfolio value and cost basis
    portfolio_market_value = 0.0
    portfolio_cost_basis = 0.0
    if fetcher: # Check if fetcher is valid
        for symbol, details in st.session_state.portfolio.items():
            try:
                current_price = fetcher.get_current_price(symbol)
                if current_price > 0:
                    portfolio_market_value += details['quantity'] * current_price
                    portfolio_cost_basis += details.get('cost_basis', 0)
                # else: # Don't warn repeatedly here, portfolio display handles it
            except Exception:
                pass # Ignore errors here, handled in portfolio display

    st.metric("Portfolio Market Value", f"₹{portfolio_market_value:,.2f}")
    unrealized_pnl = portfolio_market_value - portfolio_cost_basis
    st.metric("Unrealized P&L", f"₹{unrealized_pnl:,.2f}",
              delta=f"{unrealized_pnl:+.2f}", delta_color="normal" if unrealized_pnl >= 0 else "inverse")
    total_value = st.session_state.trading_balance + portfolio_market_value
    st.metric("Total Account Value", f"₹{total_value:,.2f}")

# Order Placement Form
with trade_col2:
    st.subheader("Place Order")
    if not fetcher:
        st.error("Trading disabled: Data fetcher not available.")
    else:
        with st.form("order_form"):
            order_symbol = st.session_state.current_symbol
            st.write(f"**Stock:** {order_symbol if order_symbol else 'None Selected'}")

            # Get current price
            current_price = 0.0
            price_fetch_error = None
            if order_symbol:
                try:
                    current_price = fetcher.get_current_price(order_symbol)
                    if current_price <= 0: price_fetch_error = "Invalid market price."
                except Exception as e: price_fetch_error = f"Price fetch error: {e}"
            else: price_fetch_error = "No stock selected."

            form_disabled = bool(price_fetch_error)
            if price_fetch_error: st.error(price_fetch_error + " Order placement disabled.")
            else: st.write(f"**Market Price:** ₹{current_price:.2f}")

            # Order Inputs
            order_action = st.radio("Action", ["Buy", "Sell"], horizontal=True, disabled=form_disabled)
            order_type_selection = st.selectbox("Order Type", ["Market", "Limit", "Stop-Loss (Sell Only)"], disabled=form_disabled,
                                                help="Market: Execute ASAP. Limit: Execute at specified price or better. Stop-Loss: Trigger market sell if price drops.")

            limit_price = None
            stop_price = None
            order_price_for_calc = current_price

            if order_type_selection == "Limit":
                limit_price = st.number_input("Limit Price (₹)", min_value=0.01, value=float(current_price), format="%.2f", disabled=form_disabled,
                                              help="Buy <= this price, Sell >= this price.") # Added closing quote here
                order_price_for_calc = limit_price
            elif order_type_selection == "Stop-Loss (Sell Only)":
                if order_action == "Buy":
                    st.warning("Stop-Loss is for Sell orders. Changed to Market.", icon="⚠️")
                    order_type_selection = "Market"
                else:
                    stop_price = st.number_input("Stop Price (₹)", min_value=0.01, value=float(current_price * 0.95), format="%.2f", disabled=form_disabled, help="Trigger market sell if price drops to this.")

            quantity = st.number_input("Quantity", min_value=1, value=1, step=1, disabled=form_disabled)

            # Cost Estimation
            order_value = quantity * order_price_for_calc
            transaction_cost = order_value * TRANSACTION_COST_PERCENTAGE
            if order_action == "Buy":
                estimated_total = order_value + transaction_cost
                st.write(f"Est. Cost: ₹{order_value:,.2f} + ₹{transaction_cost:,.2f} (fees) = **₹{estimated_total:,.2f}**")
            else: # Sell
                estimated_total = order_value - transaction_cost
                st.write(f"Est. Proceeds: ₹{order_value:,.2f} - ₹{transaction_cost:,.2f} (fees) = **₹{estimated_total:,.2f}**")

            # Submit Button
            submit_order = st.form_submit_button("Place Order", disabled=form_disabled)

            # --- Order Execution Logic (Inside Form Submission) --- #
            if submit_order and not form_disabled:
                execution_price = current_price # Default for Market/Stop
                order_status = "Filled"

                # Simulate Limit Order Fill/Warning
                if order_type_selection == "Limit":
                    execution_price = limit_price
                    if order_action == "Buy" and limit_price < current_price: st.warning(f"Limit Buy ₹{limit_price:.2f} < Market ₹{current_price:.2f}. Filled at limit (simulation).", icon="⚠️")
                    elif order_action == "Sell" and limit_price > current_price: st.warning(f"Limit Sell ₹{limit_price:.2f} > Market ₹{current_price:.2f}. Filled at limit (simulation).", icon="⚠️")

                # Simulate Stop-Loss Trigger/Warning
                if order_type_selection == "Stop-Loss (Sell Only)":
                    if stop_price >= current_price: st.warning(f"Stop ₹{stop_price:.2f} >= Market ₹{current_price:.2f}. Triggered immediately.", icon="⚠️")
                    execution_price = current_price # Stop triggers market sell
                    order_status = "Filled (Stop Triggered)"

                # Final Calculation
                final_order_value = quantity * execution_price
                final_transaction_cost = final_order_value * TRANSACTION_COST_PERCENTAGE

                # Buy Logic
                if order_action == "Buy":
                    final_total_cost = final_order_value + final_transaction_cost
                    if final_total_cost <= st.session_state.trading_balance:
                        st.session_state.trading_balance -= final_total_cost
                        # Update Portfolio
                        if order_symbol in st.session_state.portfolio:
                            p_entry = st.session_state.portfolio[order_symbol]
                            new_qty = p_entry['quantity'] + quantity
                            new_cost_basis = p_entry.get('cost_basis', 0) + final_total_cost
                            p_entry['quantity'] = new_qty
                            p_entry['cost_basis'] = new_cost_basis
                            p_entry['avg_price'] = new_cost_basis / new_qty # Update avg price based on cost
                        else:
                            st.session_state.portfolio[order_symbol] = {
                                'quantity': quantity,
                                'cost_basis': final_total_cost,
                                'avg_price': final_total_cost / quantity
                            }
                        # Record Order
                        order_record = {'timestamp': datetime.datetime.now().isoformat(), 'type': order_action, 'order_type': order_type_selection, 'symbol': order_symbol, 'quantity': quantity, 'price': execution_price, 'cost': final_transaction_cost, 'total': final_total_cost, 'status': order_status}
                        if limit_price: order_record['limit_price'] = limit_price
                        st.session_state.order_history.append(order_record)
                        st.success(f"Bought {quantity} {order_symbol} @ ~₹{execution_price:.2f}. Cost: ₹{final_total_cost:.2f}")
                        st.rerun()
                    else:
                        st.error(f"Insufficient balance! Need ₹{final_total_cost:.2f}, have ₹{st.session_state.trading_balance:.2f}")

                # Sell Logic
                elif order_action == "Sell":
                    if order_symbol in st.session_state.portfolio and st.session_state.portfolio[order_symbol]['quantity'] >= quantity:
                        final_proceeds = final_order_value - final_transaction_cost
                        st.session_state.trading_balance += final_proceeds
                        # Update Portfolio
                        p_entry = st.session_state.portfolio[order_symbol]
                        cost_basis_per_share = p_entry.get('cost_basis', 0) / p_entry['quantity'] if p_entry['quantity'] > 0 else 0
                        removed_cost_basis = quantity * cost_basis_per_share
                        new_qty = p_entry['quantity'] - quantity
                        if new_qty > 0:
                            p_entry['quantity'] = new_qty
                            p_entry['cost_basis'] = p_entry.get('cost_basis', 0) - removed_cost_basis
                            # Avg price (cost basis per share) remains the same
                        else:
                            del st.session_state.portfolio[order_symbol] # Remove if all sold
                        # Record Order
                        order_record = {'timestamp': datetime.datetime.now().isoformat(), 'type': order_action, 'order_type': order_type_selection, 'symbol': order_symbol, 'quantity': quantity, 'price': execution_price, 'cost': final_transaction_cost, 'total': final_proceeds, 'status': order_status}
                        if limit_price: order_record['limit_price'] = limit_price
                        if stop_price: order_record['stop_price'] = stop_price
                        st.session_state.order_history.append(order_record)
                        st.success(f"Sold {quantity} {order_symbol} @ ~₹{execution_price:.2f}. Proceeds: ₹{final_proceeds:.2f}")
                        st.rerun()
                    else:
                        owned_qty = st.session_state.portfolio.get(order_symbol, {}).get('quantity', 0)
                        st.error(f"Sell failed! You own {owned_qty}, tried to sell {quantity}")

# --- Portfolio Display --- #
st.subheader("Your Portfolio")
if st.session_state.portfolio:
    portfolio_data = []
    total_portfolio_cost_display = 0.0
    total_market_value_display = 0.0
    if fetcher: # Check fetcher again
        for symbol, details in st.session_state.portfolio.items():
            row = {"Symbol": symbol, "Quantity": details['quantity']}
            cost_basis = details.get('cost_basis', 0)
            row["Cost Basis"] = f"₹{cost_basis:,.2f}"
            row["Avg Price (Cost)"] = f"₹{cost_basis / details['quantity']:.2f}" if details['quantity'] > 0 else "₹0.00"
            total_portfolio_cost_display += cost_basis
            try:
                current_price = fetcher.get_current_price(symbol)
                if current_price > 0:
                    market_value = details['quantity'] * current_price
                    unrealized_pnl = market_value - cost_basis
                    unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis != 0 else 0
                    row["Current Price"] = f"₹{current_price:,.2f}"
                    row["Market Value"] = f"₹{market_value:,.2f}"
                    row["Unrealized P&L"] = f"₹{unrealized_pnl:+.2f} ({unrealized_pnl_pct:+.1f}%)"
                    total_market_value_display += market_value
                else:
                    row["Current Price"] = "N/A"
                    row["Market Value"] = "N/A"
                    row["Unrealized P&L"] = "N/A"
            except Exception:
                row["Current Price"] = "Error"
                row["Market Value"] = "Error"
                row["Unrealized P&L"] = "Error"
            portfolio_data.append(row)

    if portfolio_data:
        portfolio_df = pd.DataFrame(portfolio_data)
        column_order = ["Symbol", "Quantity", "Avg Price (Cost)", "Cost Basis", "Current Price", "Market Value", "Unrealized P&L"]
        display_columns = [col for col in column_order if col in portfolio_df.columns]
        st.dataframe(portfolio_df[display_columns], use_container_width=True, hide_index=True)
        # Display Totals
        st.markdown(f"**Total Cost Basis:** ₹{total_portfolio_cost_display:,.2f} | **Total Market Value:** ₹{total_market_value_display:,.2f}")
        total_unrealized_pnl_display = total_market_value_display - total_portfolio_cost_display
        st.markdown(f"**Total Unrealized P&L:** ₹{total_unrealized_pnl_display:+.2f}")
    else:
        st.info("Portfolio empty or price data unavailable.")
else:
    st.info("Your portfolio is empty. Place a 'Buy' order to start.")

# --- Order History Display --- #
st.subheader("Order History")
if st.session_state.order_history:
    history = sorted(st.session_state.order_history, key=lambda x: x['timestamp'], reverse=True)
    history_data = []
    for order in history[:20]: # Show more history
        record = {
            "Time": datetime.datetime.fromisoformat(order['timestamp']).strftime('%y-%m-%d %H:%M'),
            "Action": order['type'], "O-Type": order['order_type'], "Symbol": order['symbol'],
            "Qty": order['quantity'], "Exec Price": f"₹{order['price']:.2f}",
            "Fees": f"₹{order['cost']:.2f}", "Total": f"₹{order['total']:.2f}", "Status": order['status']
        }
        if 'limit_price' in order: record['Limit'] = f"₹{order['limit_price']:.2f}"
        if 'stop_price' in order: record['Stop'] = f"₹{order['stop_price']:.2f}"
        history_data.append(record)

    history_df = pd.DataFrame(history_data)
    history_cols = ["Time", "Action", "O-Type", "Symbol", "Qty", "Exec Price", "Limit", "Stop", "Fees", "Total", "Status"]
    display_cols = [col for col in history_cols if col in history_df.columns]
    st.dataframe(history_df[display_cols], use_container_width=True, hide_index=True)
else:
    st.info("No order history yet.")

# --- Footer --- #
st.markdown("---")
st.markdown("Built with Streamlit | Trading Simulator for Educational Purposes Only")

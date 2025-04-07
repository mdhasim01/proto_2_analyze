"""
Data acquisition module for stock price prediction platform.
This module handles fetching real-time and historical stock data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NSE_SUFFIX = ".NS"  # Suffix for NSE stocks in Yahoo Finance
BSE_SUFFIX = ".BO"  # Suffix for BSE stocks in Yahoo Finance
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
STOCKS_CACHE_FILE = os.path.join(CACHE_DIR, "nse_stocks.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache expiration in seconds (24 hours)
CACHE_EXPIRATION = 86400  


class StockDataFetcher:
    """Class to handle fetching stock data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the StockDataFetcher."""
        self.nse_stocks_cache = None
        self.nse_stocks_cache_timestamp = 0
    
    def get_nse_stocks(self, force_refresh=False) -> List[Dict[str, Any]]:
        """
        Get list of stocks listed on NSE.
        
        Args:
            force_refresh: If True, fetch fresh data even if cache exists
            
        Returns:
            List of dictionaries containing stock info
        """
        current_time = datetime.datetime.now().timestamp()
        
        # Check if we have valid cached data
        if not force_refresh and self._is_cache_valid(STOCKS_CACHE_FILE, CACHE_EXPIRATION):
            logger.info("Using cached NSE stocks data")
            with open(STOCKS_CACHE_FILE, 'r') as f:
                return json.load(f)
        
        logger.info("Fetching NSE stocks list")
        
        # This approach uses a combination of hard-coded major stocks and indices
        # In a production environment, you'd ideally fetch this data from an official source
        
        # List of major NSE stocks and indices
        major_symbols = [
            {"symbol": "RELIANCE", "name": "Reliance Industries Ltd.", "type": "stock", "sector": "Energy"},
            {"symbol": "TCS", "name": "Tata Consultancy Services Ltd.", "type": "stock", "sector": "Technology"},
            {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd.", "type": "stock", "sector": "Banking"},
            {"symbol": "INFY", "name": "Infosys Ltd.", "type": "stock", "sector": "Technology"},
            {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd.", "type": "stock", "sector": "FMCG"},
            {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd.", "type": "stock", "sector": "Banking"},
            {"symbol": "SBIN", "name": "State Bank of India", "type": "stock", "sector": "Banking"},
            {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd.", "type": "stock", "sector": "Telecom"},
            {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank Ltd.", "type": "stock", "sector": "Banking"},
            {"symbol": "HCLTECH", "name": "HCL Technologies Ltd.", "type": "stock", "sector": "Technology"},
            {"symbol": "WIPRO", "name": "Wipro Ltd.", "type": "stock", "sector": "Technology"},
            {"symbol": "TECHM", "name": "Tech Mahindra Ltd.", "type": "stock", "sector": "Technology"},
            {"symbol": "LTI", "name": "Larsen & Toubro Infotech Ltd.", "type": "stock", "sector": "Technology"},
            {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries Ltd.", "type": "stock", "sector": "Pharma"},
            {"symbol": "CIPLA", "name": "Cipla Ltd.", "type": "stock", "sector": "Pharma"},
            {"symbol": "DRREDDY", "name": "Dr. Reddy's Laboratories Ltd.", "type": "stock", "sector": "Pharma"},
            {"symbol": "DIVISLAB", "name": "Divi's Laboratories Ltd.", "type": "stock", "sector": "Pharma"},
            {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd.", "type": "stock", "sector": "Automotive"},
            {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd.", "type": "stock", "sector": "Automotive"},
            {"symbol": "M&M", "name": "Mahindra & Mahindra Ltd.", "type": "stock", "sector": "Automotive"},
            {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp Ltd.", "type": "stock", "sector": "Automotive"},
            {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto Ltd.", "type": "stock", "sector": "Automotive"},
            {"symbol": "ITC", "name": "ITC Ltd.", "type": "stock", "sector": "FMCG"},
            {"symbol": "NESTLEIND", "name": "Nestle India Ltd.", "type": "stock", "sector": "FMCG"},
            {"symbol": "BRITANNIA", "name": "Britannia Industries Ltd.", "type": "stock", "sector": "FMCG"},
            {"symbol": "TATASTEEL", "name": "Tata Steel Ltd.", "type": "stock", "sector": "Metals"},
            {"symbol": "HINDALCO", "name": "Hindalco Industries Ltd.", "type": "stock", "sector": "Metals"},
            {"symbol": "JSWSTEEL", "name": "JSW Steel Ltd.", "type": "stock", "sector": "Metals"},
            {"symbol": "ONGC", "name": "Oil and Natural Gas Corporation Ltd.", "type": "stock", "sector": "Energy"},
            {"symbol": "BPCL", "name": "Bharat Petroleum Corporation Ltd.", "type": "stock", "sector": "Energy"},
            {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd.", "type": "stock", "sector": "Financial Services"},
            {"symbol": "AXISBANK", "name": "Axis Bank Ltd.", "type": "stock", "sector": "Banking"},
            {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd.", "type": "stock", "sector": "Consumer Durables"},
            {"symbol": "TITAN", "name": "Titan Company Ltd.", "type": "stock", "sector": "Consumer Durables"},
            {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Ltd.", "type": "stock", "sector": "Cement"},
            {"symbol": "ADANIPORTS", "name": "Adani Ports and Special Economic Zone Ltd.", "type": "stock", "sector": "Infrastructure"},
            {"symbol": "ADANIENT", "name": "Adani Enterprises Ltd.", "type": "stock", "sector": "Conglomerate"},
            {"symbol": "EICHERMOT", "name": "Eicher Motors Ltd.", "type": "stock", "sector": "Automotive"},
            {"symbol": "ZOMATO", "name": "Zomato Ltd.", "type": "stock", "sector": "Technology"},
            {"symbol": "NYKAA", "name": "FSN E-Commerce Ventures Ltd. (Nykaa)", "type": "stock", "sector": "Retail"},
            {"symbol": "PAYTM", "name": "One 97 Communications Ltd. (Paytm)", "type": "stock", "sector": "Technology"},
            {"symbol": "POLICYBZR", "name": "PB Fintech Ltd. (PolicyBazaar)", "type": "stock", "sector": "Financial Services"},
            {"symbol": "JUBLFOOD", "name": "Jubilant FoodWorks Ltd.", "type": "stock", "sector": "Food & Beverages"},
            # Indices
            {"symbol": "^NSEI", "name": "NIFTY 50", "type": "index", "sector": None},
            {"symbol": "^BSESN", "name": "S&P BSE SENSEX", "type": "index", "sector": None},
            {"symbol": "NIFTYBANK.NS", "name": "Nifty Bank", "type": "index", "sector": "Banking"},
            {"symbol": "NIFTYIT.NS", "name": "Nifty IT", "type": "index", "sector": "Technology"},
            {"symbol": "NIFTYPHARM.NS", "name": "Nifty Pharma", "type": "index", "sector": "Pharma"},
            {"symbol": "NIFTYAUTO.NS", "name": "Nifty Auto", "type": "index", "sector": "Automotive"},
            {"symbol": "NIFTYFMCG.NS", "name": "Nifty FMCG", "type": "index", "sector": "FMCG"},
        ]
        
        # Try to get more symbols using yfinance tickers for the NIFTY 500 constituents
        try:
            nifty500 = yf.Ticker("^CRSLDX")  # NIFTY 500 index ticker
            # Extract info on all holdings (may not be complete)
            holdings = nifty500.get_holdings()
            if holdings is not None and not holdings.empty:
                # Add additional stocks from NIFTY 500 that aren't already in our list
                existing_symbols = {item["symbol"] for item in major_symbols}
                for _, row in holdings.iterrows():
                    symbol = row.get('ticker', "").replace(".NS", "")
                    if symbol and symbol not in existing_symbols:
                        major_symbols.append({
                            "symbol": symbol,
                            "name": row.get('name', symbol),
                            "type": "stock",
                            "sector": row.get('sector', "Unknown")
                        })
                        existing_symbols.add(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch NIFTY 500 constituents: {str(e)}")
        
        # Verify that the symbols are valid and add current price
        valid_symbols = []
        for item in major_symbols:
            symbol = item["symbol"]
            
            # Skip the processing of indices as they use different formats
            if item["type"] == "index":
                valid_symbols.append(item)
                continue
                
            # For stocks, append the NSE suffix and verify
            try:
                yahoo_symbol = f"{symbol}.NS"
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                
                if info and "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                    price = info["regularMarketPrice"]
                    item["current_price"] = price
                    valid_symbols.append(item)
                    logger.info(f"Added symbol {symbol} with price {price}")
                else:
                    logger.warning(f"Skipping {symbol}: No valid price data")
            except Exception as e:
                logger.warning(f"Failed to verify {symbol}: {str(e)}")
        
        # Cache the valid symbols
        with open(STOCKS_CACHE_FILE, 'w') as f:
            json.dump(valid_symbols, f)
        
        return valid_symbols
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a stock symbol.
        
        Args:
            symbol: Stock symbol without exchange suffix
            
        Returns:
            Current price as float
        """
        try:
            # For NSE stocks, append .NS suffix
            if symbol in ("^NSEI", "^BSESN") or symbol.endswith(".NS"):
                yahoo_symbol = symbol
            else:
                yahoo_symbol = f"{symbol}.NS"
                
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                return info["regularMarketPrice"]
            else:
                logger.warning(f"No price data for {symbol}, returning 0")
                return 0
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return 0
    
    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical data for a stock symbol.

        Args:
            symbol: Stock symbol without exchange suffix.
            period: Period to fetch (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max).
                    Adjust based on interval: Intraday data is limited to shorter periods.
            interval: Data interval (e.g., 1m, 5m, 15m, 30m, 1h, 1d, 1wk).

        Returns:
            DataFrame with historical data.
        """
        try:
            # Adjust period for intraday intervals if necessary (yfinance limitations)
            # For intervals < 1d, max period is often 60d, and for < 1h, often 7d.
            valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
            if interval not in valid_intervals:
                logger.warning(f"Interval '{interval}' not directly supported by yfinance, defaulting to '1d'.")
                interval = '1d'

            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
                 # Map common period names to days for comparison
                period_map_days = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'max': 10000}
                max_period_days = 60 # yfinance limit for intraday is often 60 days
                if interval == '1m':
                    max_period_days = 7 # 1m data often limited to 7 days
                
                requested_days = period_map_days.get(period, 30) # Default to 1mo if period string not recognized

                if requested_days > max_period_days:
                    logger.warning(f"Period '{period}' too long for interval '{interval}'. Limiting to {max_period_days} days.")
                    # Find the closest valid period string <= max_period_days
                    valid_periods = {days: p for p, days in period_map_days.items() if days <= max_period_days}
                    closest_days = max(valid_periods.keys())
                    period = valid_periods[closest_days]


            # For NSE stocks, append .NS suffix
            if symbol in ("^NSEI", "^BSESN") or symbol.endswith((".NS", ".BO")): # Added .BO check
                yahoo_symbol = symbol
            else:
                # Assume NSE if not index or already suffixed
                yahoo_symbol = f"{symbol}.NS"

            ticker = yf.Ticker(yahoo_symbol)
            logger.info(f"Fetching historical data for {yahoo_symbol} with period={period}, interval={interval}")
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No historical data returned for {yahoo_symbol} with period={period}, interval={interval}")
                return pd.DataFrame()

            # Ensure index is timezone-naive for consistency, or handle timezone properly if needed
            if data.index.tz is not None:
                 data.index = data.index.tz_localize(None)

            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol} (interval: {interval}): {str(e)}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for a stock symbol based on daily data.
        Note: Indicators are typically calculated on daily data unless specified otherwise.
        Fetching daily data for indicators regardless of chart interval.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary of technical indicators
        """
        try:
            # Fetch daily data specifically for standard indicator calculation
            data = self.get_historical_data(symbol, period="6mo", interval="1d") # Fetch daily data

            if data.empty:
                logger.warning(f"Cannot calculate technical indicators for {symbol}: No daily historical data")
                return self._generate_random_indicators() # Keep fallback

            # Calculate RSI (14)
            data['RSI'] = ta.rsi(data['Close'], length=14)

            # Calculate MACD
            macd = ta.macd(data['Close'])
            data = pd.concat([data, macd], axis=1)

            # Calculate Stochastic Oscillator
            stoch = ta.stoch(data['High'], data['Low'], data['Close'])
            data = pd.concat([data, stoch], axis=1)

            # Calculate Bollinger Bands
            bbands = ta.bbands(data['Close'])
            data = pd.concat([data, bbands], axis=1)

            # Calculate Moving Averages
            data['SMA_50'] = ta.sma(data['Close'], length=50)
            data['SMA_200'] = ta.sma(data['Close'], length=200)

            # Get latest values
            if data.empty: # Check again after calculations
                 return self._generate_random_indicators()
                 
            latest = data.iloc[-1]
            close = latest['Close']

            # Ensure indicators exist before accessing
            rsi_val = latest.get('RSI', 50)
            macd_val = latest.get('MACD_12_26_9', 0)
            stoch_val = latest.get('STOCHk_14_3_3', 50)
            bbu_val = latest.get('BBU_5_2.0')
            bbl_val = latest.get('BBL_5_2.0')
            sma50_val = latest.get('SMA_50')
            sma200_val = latest.get('SMA_200')

            indicators = {
                "RSI (14)": round(rsi_val, 2) if not pd.isna(rsi_val) else 50,
                "MACD": round(macd_val, 2) if not pd.isna(macd_val) else 0,
                "Stochastic": round(stoch_val, 2) if not pd.isna(stoch_val) else 50,
                "Bollinger Bands": "N/A", # Default
                "Moving Avg (50)": "N/A", # Default
                "Moving Avg (200)": "N/A" # Default
            }

            # Safely calculate Bollinger Bands status
            if bbu_val is not None and bbl_val is not None and not pd.isna(bbu_val) and not pd.isna(bbl_val):
                if close > bbu_val: indicators["Bollinger Bands"] = "Above"
                elif close < bbl_val: indicators["Bollinger Bands"] = "Below"
                else: indicators["Bollinger Bands"] = "Inside"

            # Safely calculate Moving Average status
            if sma50_val is not None and not pd.isna(sma50_val):
                indicators["Moving Avg (50)"] = f"{'Above' if close > sma50_val else 'Below'} price"
            if sma200_val is not None and not pd.isna(sma200_val):
                 indicators["Moving Avg (200)"] = f"{'Above' if close > sma200_val else 'Below'} price"


            return indicators

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
            return self._generate_random_indicators() # Keep fallback
    
    def _generate_random_indicators(self) -> Dict[str, float]:
        """Generate random indicators as fallback."""
        import random
        
        indicators = {
            "RSI (14)": round(random.uniform(30, 70), 2),
            "MACD": round(random.uniform(-2, 2), 2),
            "Stochastic": round(random.uniform(20, 80), 2),
            "Bollinger Bands": "Above" if random.random() > 0.5 else "Below",
            "Moving Avg (50)": f"{'Above' if random.random() > 0.5 else 'Below'} price",
            "Moving Avg (200)": f"{'Above' if random.random() > 0.5 else 'Below'} price"
        }
        
        return indicators
    
    def get_stock_sentiment(self, symbol: str, company_name: str) -> Dict[str, Any]:
        """
        Get news sentiment for a stock.
        
        Args:
            symbol: Stock symbol
            company_name: Company name for search
            
        Returns:
            Dictionary with sentiment analysis results and news items
        """
        try:
            from textblob import TextBlob
            from newsapi import NewsApiClient
            import os
            
            # Initialize NewsAPI
            # Note: In a real app, store API key in environment variable
            api_key = os.environ.get("NEWS_API_KEY", "dummy_key")
            
            # If no API key, generate mock data
            if api_key == "dummy_key":
                return self._generate_mock_news_sentiment(symbol, company_name)
                
            newsapi = NewsApiClient(api_key=api_key)
            
            # Search for news articles about the company
            query = f"{company_name} OR {symbol} stocks"
            news = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=10)
            
            if not news or 'articles' not in news or len(news['articles']) == 0:
                return self._generate_mock_news_sentiment(symbol, company_name)
            
            # Process news articles and calculate sentiment
            articles = news['articles']
            
            # List to store processed news with sentiment
            news_with_sentiment = []
            total_polarity = 0
            
            for article in articles:
                title = article['title']
                # Analyze sentiment using TextBlob
                blob = TextBlob(title)
                polarity = blob.sentiment.polarity
                
                # Determine sentiment category
                sentiment_category = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
                
                news_with_sentiment.append({
                    "title": title,
                    "sentiment": sentiment_category
                })
                total_polarity += polarity
            
            # Calculate overall sentiment
            avg_polarity = total_polarity / len(articles) if articles else 0
            overall_sentiment = "Bullish" if avg_polarity > 0.1 else "Bearish" if avg_polarity < -0.1 else "Neutral"
            
            return {
                "overall": overall_sentiment,
                "news_items": news_with_sentiment[:4]  # Return top 4 news items
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return self._generate_mock_news_sentiment(symbol, company_name)
    
    def _generate_mock_news_sentiment(self, symbol: str, company_name: str) -> Dict[str, Any]:
        """Generate mock news sentiment as fallback."""
        import random
        
        sentiments = ["Positive", "Neutral", "Positive", "Negative"]
        overall = random.choice(["Bullish", "Bearish", "Neutral"])
        
        news_items = [
            {"title": f"Q3 Results: {company_name} reports strong growth", "sentiment": "Positive"},
            {"title": "Markets react to RBI policy announcement", "sentiment": "Neutral"},
            {"title": f"Analyst upgrades {symbol} to 'Buy'", "sentiment": "Positive"},
            {"title": "Global economic concerns impact Indian markets", "sentiment": "Negative"}
        ]
        
        # Make the sentiment more deterministic based on symbol
        seed_value = sum(ord(c) for c in symbol)
        random.seed(seed_value)
        random.shuffle(news_items)
        
        return {
            "overall": overall,
            "news_items": news_items
        }
    
    def _is_cache_valid(self, cache_file: str, expiration_seconds: int) -> bool:
        """Check if cache file exists and is not expired."""
        if not os.path.exists(cache_file):
            return False
            
        file_time = os.path.getmtime(cache_file)
        current_time = datetime.datetime.now().timestamp()
        
        return (current_time - file_time) < expiration_seconds

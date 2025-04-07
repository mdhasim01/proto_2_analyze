import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List, Optional
import random  # Just for generating mock data
import os
import json
import logging
from .data_acquisition import StockDataFetcher
from .models.data_models import (
    PredictionRequest, 
    PredictionResponse, 
    PredictionDetail, 
    ConfidenceInterval,
    SymbolInfo,
    SymbolList,
    ApiStatus
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path for storing cached predictions
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class ApiService:
    def __init__(self):
        """Initialize the API service with a StockDataFetcher."""
        self.start_time = datetime.datetime.now()
        self.version = "1.0.0"
        self.model_version = "0.1.0"
        self.predictions_cache = {}  # In-memory cache for recent predictions
        self.data_fetcher = StockDataFetcher()  # Initialize stock data fetcher
        
    def get_api_status(self) -> ApiStatus:
        """Get the current API status"""
        uptime = datetime.datetime.now() - self.start_time
        return ApiStatus(
            status="healthy",
            version=self.version,
            uptime=str(uptime),
            model_version=self.model_version
        )
    
    def get_available_symbols(self) -> SymbolList:
        """Get list of available symbols for prediction"""
        try:
            # Fetch stock data from the data fetcher
            stocks_data = self.data_fetcher.get_nse_stocks()
            
            symbols = []
            for item in stocks_data:
                # Get current price for each stock using the data fetcher if not already in cached data
                if "current_price" not in item:
                    current_price = self.data_fetcher.get_current_price(item["symbol"])
                else:
                    current_price = item["current_price"]
                
                symbols.append(
                    SymbolInfo(
                        symbol=item["symbol"],
                        name=item["name"],
                        type=item["type"],
                        sector=item["sector"],
                        current_price=current_price,
                        last_updated=datetime.datetime.now()
                    )
                )
            
            return SymbolList(
                symbols=symbols,
                count=len(symbols)
            )
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            # Return empty list in case of error
            return SymbolList(symbols=[], count=0)
    
    def search_symbols(self, query: str) -> SymbolList:
        """Search for symbols matching the query (by name or symbol)"""
        if not query or len(query) < 2:
            return self.get_available_symbols()
            
        try:
            # Get all symbols
            all_symbols = self.get_available_symbols()
            
            # Filter by query
            filtered_symbols = []
            for symbol in all_symbols.symbols:
                if (query.upper() in symbol.symbol.upper() or 
                    (symbol.name and query.lower() in symbol.name.lower())):
                    filtered_symbols.append(symbol)
                    
            return SymbolList(
                symbols=filtered_symbols,
                count=len(filtered_symbols)
            )
        except Exception as e:
            logger.error(f"Error searching symbols: {str(e)}")
            return SymbolList(symbols=[], count=0)
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get stock data for a given symbol"""
        try:
            # Get current price from data fetcher
            current_price = self.data_fetcher.get_current_price(symbol)
            
            # Get stock name from available symbols
            all_symbols = self.data_fetcher.get_nse_stocks()
            stock_info = next((item for item in all_symbols if item["symbol"] == symbol), None)
            
            if not stock_info:
                raise ValueError(f"Symbol {symbol} not found")
                
            return {
                "symbol": symbol,
                "name": stock_info["name"],
                "current_price": current_price
            }
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            raise ValueError(f"Error getting stock data for {symbol}: {str(e)}")
    
    def get_cached_prediction(self, symbol: str, timeframe: str) -> Optional[PredictionResponse]:
        """Get cached prediction for a symbol and timeframe if available"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Check in-memory cache first
        if cache_key in self.predictions_cache:
            return self.predictions_cache[cache_key]
            
        # Check file cache
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Convert to PredictionResponse
                    prediction = PredictionResponse(**cached_data)
                    # Update in-memory cache
                    self.predictions_cache[cache_key] = prediction
                    return prediction
            except Exception as e:
                logger.warning(f"Error reading cache for {cache_key}: {str(e)}")
                # If there's any error reading the cache, just return None
                return None
                
        return None
    
    def cache_prediction(self, prediction: PredictionResponse, timeframe: str) -> None:
        """Cache a prediction for future use"""
        symbol = prediction.symbol
        cache_key = f"{symbol}_{timeframe}"
        
        # Update in-memory cache
        self.predictions_cache[cache_key] = prediction
        
        # Save to file cache
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                # Convert to dict and save as JSON
                json.dump(prediction.dict(), f)
        except Exception as e:
            logger.warning(f"Error caching prediction for {cache_key}: {str(e)}")
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate a prediction for the given request"""
        symbol = request.symbol
        timeframe = request.timeframe
        include_confidence = request.include_confidence
        
        # Check if we have a cached prediction that's less than 1 hour old
        cached_prediction = self.get_cached_prediction(symbol, timeframe)
        if cached_prediction:
            cache_time = datetime.datetime.fromisoformat(cached_prediction.metadata.get("prediction_created", ""))
            current_time = datetime.datetime.now()
            
            # If the cache is less than 1 hour old, use it
            if (current_time - cache_time).total_seconds() < 3600:
                logger.info(f"Using cached prediction for {symbol}_{timeframe}")
                # Update the confidence interval based on user preference
                if include_confidence and not cached_prediction.prediction.confidence_interval:
                    # Generate a new confidence interval
                    margin = cached_prediction.current_price * 0.05
                    cached_prediction.prediction.confidence_interval = ConfidenceInterval(
                        lower=round(cached_prediction.prediction.predicted_price - margin, 2),
                        upper=round(cached_prediction.prediction.predicted_price + margin, 2)
                    )
                elif not include_confidence:
                    cached_prediction.prediction.confidence_interval = None
                    
                return cached_prediction
        
        # Get stock data with real current price
        stock_data = self.get_stock_data(symbol)
        current_price = stock_data["current_price"]
        
        # Calculate target date based on timeframe
        today = datetime.date.today()
        target_date = None
        
        if timeframe == "short_term":
            # 7 days ahead
            target_date = today + datetime.timedelta(days=7)
        elif timeframe == "medium_term":
            # 30 days ahead
            target_date = today + datetime.timedelta(days=30)
        elif timeframe == "long_term":
            # 90 days ahead
            target_date = today + datetime.timedelta(days=90)
        else:
            # Default to 7 days
            target_date = today + datetime.timedelta(days=7)
            timeframe = "short_term"
        
        # In a real system, we would use ML models here
        # For the demo, we'll use a simple algorithm based on historical volatility
        try:
            # Get historical data to determine trend and volatility
            historical_data = self.data_fetcher.get_historical_data(symbol, period="1y")
            
            if historical_data.empty:
                # Fallback to mock prediction if no historical data
                change_factor = random.uniform(-0.15, 0.20)  # -15% to +20% change
            else:
                # Calculate recent trend (last 30 days vs previous 30 days)
                recent = historical_data.tail(30)['Close'].mean()
                previous = historical_data.iloc[-60:-30]['Close'].mean() if len(historical_data) >= 60 else historical_data.head(30)['Close'].mean()
                
                trend_factor = (recent / previous - 1) if previous > 0 else 0
                
                # Calculate volatility
                if len(historical_data) >= 30:
                    returns = historical_data['Close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                else:
                    volatility = 0.25  # Default volatility if not enough data
                
                # Deterministic prediction based on symbol, trend, and target date
                # This ensures same parameters always give same prediction
                seed_value = sum(ord(c) for c in symbol) + int(target_date.strftime("%Y%m%d"))
                random.seed(seed_value)
                
                # Generate prediction with a blend of trend and random factor
                if timeframe == "short_term":
                    days = 7
                    trend_weight = 0.7  # Higher weight on trend for short term
                elif timeframe == "medium_term":
                    days = 30
                    trend_weight = 0.5  # Balanced for medium term
                else:  # long_term
                    days = 90
                    trend_weight = 0.3  # Lower weight on trend for long term
                
                # Random component scaled by volatility
                random_component = random.uniform(-volatility, volatility) * (days/252)**0.5
                
                # Change factor combines trend and random component
                change_factor = trend_factor * trend_weight + random_component * (1 - trend_weight)
                
                # Limit extreme predictions
                change_factor = max(min(change_factor, 0.4), -0.3)
        
        except Exception as e:
            logger.error(f"Error calculating prediction: {str(e)}")
            # Fallback to random but deterministic prediction
            seed_value = sum(ord(c) for c in symbol)
            random.seed(seed_value)
            change_factor = random.uniform(-0.15, 0.20)  # -15% to +20% change
        
        predicted_price = current_price * (1 + change_factor)
        predicted_price = round(predicted_price, 2)
        change_percentage = round(change_factor * 100, 2)
        
        # Generate confidence interval if requested
        confidence_interval = None
        if include_confidence:
            # Base margin on historical volatility if available
            try:
                if historical_data is not None and not historical_data.empty:
                    returns = historical_data['Close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    
                    # Adjust for timeframe (square root of time rule)
                    if timeframe == "short_term":
                        days = 7
                    elif timeframe == "medium_term":
                        days = 30
                    else:  # long_term
                        days = 90
                    
                    time_factor = (days / 252) ** 0.5
                    confidence_margin = current_price * volatility * time_factor * 1.96  # 95% confidence interval
                else:
                    confidence_margin = current_price * 0.05  # Default 5% margin
            except:
                confidence_margin = current_price * 0.05  # Default 5% margin
                
            confidence_interval = ConfidenceInterval(
                lower=round(predicted_price - confidence_margin, 2),
                upper=round(predicted_price + confidence_margin, 2)
            )
        
        # Create prediction detail
        prediction_detail = PredictionDetail(
            timeframe=timeframe,
            target_date=target_date,
            predicted_price=predicted_price,
            change_percentage=change_percentage,
            confidence_interval=confidence_interval
        )
        
        # Get technical indicators and sentiment for this stock
        technical_indicators = self.data_fetcher.calculate_technical_indicators(symbol)
        sentiment = self.data_fetcher.get_stock_sentiment(symbol, stock_data["name"])
        
        # Create response
        prediction_response = PredictionResponse(
            symbol=symbol,
            name=stock_data["name"],
            current_price=current_price,
            prediction=prediction_detail,
            metadata={
                "prediction_created": datetime.datetime.now().isoformat(),
                "model": f"StockPredictor-{timeframe}-{self.model_version}",
                "technical_indicators": technical_indicators,
                "sentiment": sentiment
            }
        )
        
        # Cache the prediction for future use
        self.cache_prediction(prediction_response, timeframe)
        
        return prediction_response
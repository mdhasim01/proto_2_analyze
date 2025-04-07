from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import datetime

class ConfidenceInterval(BaseModel):
    lower: float
    upper: float

class PredictionDetail(BaseModel):
    timeframe: str
    target_date: datetime.date
    predicted_price: float
    change_percentage: float
    confidence_interval: Optional[ConfidenceInterval] = None

class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "short_term"
    include_confidence: bool = True

class NewsItem(BaseModel):
    title: str
    sentiment: str

class SentimentAnalysis(BaseModel):
    overall: str
    news_items: List[NewsItem]

class PredictionResponse(BaseModel):
    symbol: str
    name: str
    current_price: float
    prediction: PredictionDetail
    metadata: Dict[str, Any]

class SymbolInfo(BaseModel):
    symbol: str
    name: str
    type: str  # "stock" or "index"
    sector: Optional[str] = None
    current_price: Optional[float] = None
    last_updated: Optional[datetime.datetime] = None

class SymbolList(BaseModel):
    symbols: List[SymbolInfo]
    count: int

class ApiStatus(BaseModel):
    status: str
    version: str
    uptime: str
    model_version: str
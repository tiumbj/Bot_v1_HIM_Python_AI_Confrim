"""
Hybrid Intelligence Mentor (HIM) - Configuration Module
Centralized credentials, API keys, and strategy parameters
"""

import MetaTrader5 as mt5
from typing import Dict, Any
from dataclasses import dataclass

# ============================================================================
# MT5 CREDENTIALS
# ============================================================================
MT5_LOGIN = 168021026
MT5_PASS = "Tium@232520"
MT5_SERVER = "XMGlobal-MT5 2"

# ============================================================================
# API KEYS
# ============================================================================
TELEGRAM_TOKEN = "8280027714:AAHELs-PAS7u0JctXElxyxz_hnFwgXBm22nl"
CHAT_ID = "8385962634"
AI_API_KEY = "sk-462b368e928148f08193668f48efe7eb"

# Multiple News API Keys (สำหรับ Fallback)
NEWS_API_KEYS = [
    "ygBt0Cds219wKrjAG87OvnZNXpUw6mhYfxOrFg1w",  # MarketAux Key 1
    "XFTPOTVEFBIOX2IZ"                             # Alpha Vantage Key
]

# ============================================================================
# SYMBOL CONFIGURATION
# ============================================================================
SYMBOL = "GOLD"  # MT5 symbol for XAUUSD
SYMBOL_NEWS = "XAU"  # News API symbol

# ============================================================================
# TIMEFRAME MAPPING
# ============================================================================
TIMEFRAMES = {
    "HTF": mt5.TIMEFRAME_D1,   # High Timeframe - Daily
    "MTF": mt5.TIMEFRAME_H1,   # Medium Timeframe - Hourly
    "LTF": mt5.TIMEFRAME_M15   # Low Timeframe - 15min
}

TIMEFRAME_NAMES = {
    mt5.TIMEFRAME_D1: "D1",
    mt5.TIMEFRAME_H1: "H1",
    mt5.TIMEFRAME_M30: "M30",
    mt5.TIMEFRAME_M15: "M15",
    mt5.TIMEFRAME_M5: "M5"
}

# ============================================================================
# TECHNICAL INDICATOR PARAMETERS
# ============================================================================
@dataclass
class IndicatorConfig:
    EMA_FAST: int = 50
    EMA_SLOW: int = 200
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    RSI_PERIOD: int = 14
    ATR_PERIOD: int = 14
    SAR_STEP: float = 0.02
    SAR_MAX: float = 0.2
    SWING_LOOKBACK: int = 10
    FVG_MIN_GAP: float = 0.0001
    OB_LOOKBACK: int = 20
    PINBAR_WICK_RATIO: float = 0.70
    PINBAR_BODY_RATIO: float = 0.30
    LIQUIDITY_SWEEP_THRESHOLD: int = 3

INDICATORS = IndicatorConfig()

# ============================================================================
# RISK MANAGEMENT PARAMETERS
# ============================================================================
@dataclass
class RiskConfig:
    RISK_PERCENT: float = 1.0
    MIN_RR_RATIO: float = 2.0
    MAX_SPREAD: float = 30.0
    MAX_SLIPPAGE: int = 10
    CONFLUENCE_THRESHOLD: int = 3
    AI_CONFIDENCE_THRESHOLD: float = 80.0
    MAX_DAILY_TRADES: int = 3
    MAX_OPEN_POSITIONS: int = 1
    TRAILING_STOP_ACTIVATION: float = 1.5

RISK = RiskConfig()

# ============================================================================
# AI CONFIGURATION
# ============================================================================
AI_API_URL = "https://api.deepseek.com/v1/chat/completions"
AI_MODEL = "deepseek-chat"
AI_MAX_TOKENS = 1000
AI_TEMPERATURE = 0.3

# ============================================================================
# NEWS API CONFIGURATION
# ============================================================================
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
NEWS_LOOKBACK_HOURS = 24
NEWS_MIN_IMPACT = "medium"

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
@dataclass
class SystemConfig:
    HEARTBEAT_INTERVAL: int = 60
    RECONNECT_ATTEMPTS: int = 5
    RECONNECT_DELAY: int = 5
    RECONNECT_BACKOFF: float = 2.0
    DATA_LOOKBACK: int = 500
    TIMEZONE: str = "UTC"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "him_system.log"

SYSTEM = SystemConfig()

# ============================================================================
# CONFLUENCE SCORING WEIGHTS
# ============================================================================
CONFLUENCE_WEIGHTS = {
    "trend_alignment": 1.5,
    "ema_cross": 1.0,
    "macd_signal": 0.5,
    "rsi_zone": 0.5,
    "sar_flip": 0.5,
    "support_resistance": 1.0,
    "order_block": 1.5,
    "fvg": 1.0,
    "pinbar": 1.0,
    "liquidity_sweep": 1.5,
    "bos": 1.0,
    "choch": 1.5,
    "news_sentiment": 1.0
}

# ============================================================================
# TELEGRAM MESSAGE TEMPLATES
# ============================================================================
TELEGRAM_TEMPLATES = {
    "signal_header": "[SIGNAL] <b>HIM TRADING SIGNAL</b>\n{'='*40}\n",
    "mentor_intro": "📚 <b>MENTOR EXPLANATION:</b>\n",
    "ai_verdict": "🤖 <b>AI VERIFICATION:</b>\n",
    "execution": "[FAST] <b>EXECUTION DETAILS:</b>\n",
    "error": "[ERROR] <b>ERROR:</b>\n",
    "heartbeat": "[HEARTBEAT] System Heartbeat: All connections healthy\n"
}

# ============================================================================
# EXPORT ALL CONFIGURATIONS
# ============================================================================
__all__ = [
    'MT5_LOGIN', 'MT5_PASS', 'MT5_SERVER',
    'TELEGRAM_TOKEN', 'CHAT_ID', 'AI_API_KEY', 'NEWS_API_KEYS',
    'SYMBOL', 'SYMBOL_NEWS', 'TIMEFRAMES', 'TIMEFRAME_NAMES',
    'INDICATORS', 'RISK', 'SYSTEM',
    'AI_API_URL', 'AI_MODEL', 'AI_MAX_TOKENS', 'AI_TEMPERATURE',
    'NEWS_API_URL', 'ALPHA_VANTAGE_URL', 'NEWS_LOOKBACK_HOURS', 'NEWS_MIN_IMPACT',
    'CONFLUENCE_WEIGHTS', 'TELEGRAM_TEMPLATES'
]
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import anthropic
import os
from cachetools import TTLCache

app = FastAPI(title="OptiMax Stock Analysis API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for 30 minutes
cache = TTLCache(maxsize=100, ttl=1800)

# Shariah-compliant stocks (AAOIFI + MSCI Islamic Index standards)
SHARIAH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "TSLA", "META", "AVGO", "COST", "NFLX",
    "AMD", "ADBE", "CSCO", "INTC", "QCOM", "INTU", "AMAT", "ISRG", "BKNG", "VRTX",
    "PANW", "SBUX", "GILD", "ADI", "LRCX", "MDLZ", "REGN", "KLAC", "SNPS", "CDNS",
    "MRVL", "ORLY", "CTAS", "MAR", "ADSK", "FTNT", "ABNB", "MNST", "WDAY", "CHTR",
    "MELI", "AEP", "NXPI", "PAYX", "LULU", "ODFL", "ROST", "FAST", "CPRT", "KDP",
    "CTSH", "DXCM", "EA", "VRSK", "BKR", "PCAR", "KHC", "GEHC", "CCEP", "EXC",
    "XEL", "TEAM", "IDXX", "ANSS", "CSGP", "FANG", "ON", "ZS", "TTWO", "DDOG",
    "BIIB", "ILMN", "CDW", "GFS", "WBD", "MDB", "SMCI", "CRWD", "WBA", "ZM",
    "MRNA", "ALGN", "ENPH", "DLTR", "LCID", "RIVN", "BMRN", "NTES", "JD", "BIDU",
    "PDD", "BILI", "LI", "XPEV", "NIO", "BABA", "TME", "VIPS", "WB", "DOYU",
    "IQ", "HUYA", "MOMO", "YY", "ATHM", "BZUN", "TIGR", "FUTU", "DADA", "KC",
    "CBAT", "VNET", "SOHU", "SINA", "JOBS", "CAAS", "SOS", "SXTC", "LX", "TANH",
    "AIHS", "RENN", "CNIT", "BEST", "QTT", "CANG", "MOXC", "CJJD", "RERE", "ACST",
    "JAGX", "ATNM", "AKER", "PULM", "CLSD", "GNUS", "GBNH", "EDSA", "EAST", "LTRX",
    "ALPP", "SIOX", "MDMP", "VYNE", "TXMD", "OPGN", "ATOS", "OCGN", "CODX", "INO",
    "NVAX", "VXRT", "BCEL", "MBRX", "ADTX", "ADMP", "KTOV", "TBLT", "APDN", "IBIO",
    "GNPX", "APTO", "BNGO", "OBSV", "ONTX", "NVCN", "DTIL", "PRQR", "TRVN", "TCON",
    "EIGR", "LIFE", "DMTK", "CYCN", "CLBS", "PBLA", "CYTO", "QURE", "AFMD", "TNXP",
    "AGRX", "ARTL", "LJPC", "EYEN", "PETQ", "MDWD", "GLYC", "HGEN", "XERS", "KDMN",
    "RLMD", "SVRA", "CLOV", "BBIG", "MULN", "GREE", "ZKIN", "HUSA", "HMBL", "SNDL",
    "EXPR", "BBBY", "KOSS", "NAKD", "CTRM", "TOPS", "SHIP", "GLBS", "BIOC", "INND",
    "INTV", "RGBP", "TSNP", "OZSC", "HCMC", "SNGX", "RLFTF", "ALYI", "TLSS", "CLIS",
    "PASO", "PHIL", "SING", "HMNY", "BOTY", "FORZ", "VPER", "TCKR", "MJWL", "HEMP",
    "HIPH", "USMJ", "CBDS", "GRCU", "MDCN", "MYEC", "PAOG", "PRPM", "RXMD", "SANP",
    "SEEK", "SKYF", "SSOK", "SWRM", "TLSS", "BTCS", "MARA", "RIOT", "EBON", "SOS",
    "CAN", "EQOS", "ARBKF", "HUTMF", "HIVE", "DMGI", "BITF", "HUT", "CLSK", "APLD",
    "GREE", "WULF", "CIFR", "CORZ", "IREN", "BTDR", "SDIG", "SOLV", "BTCS", "MGI",
    "CNET", "FRMO", "GBBK", "BBBY", "OSTK", "NEGG", "GSBC", "ALOT", "NCTY", "XELA",
    "BBAI", "FFIE", "DTC", "MULN", "CEI", "NILE", "GFAI", "MMMB", "CARV", "GOEV",
    "WKHS", "RIDE", "FSR", "ENVX", "QS", "BLNK", "CHPT", "EVGO", "DRIV", "PSNY",
    "NKLA", "HYLN", "ARVL", "LEV", "LCID", "RIVN", "PTRA", "LAZR", "VLDR", "OUST",
    "LIDR", "INVZ", "TALK", "AEYE", "INDI", "RAAC", "AMGN"
]

def calculate_trading_days_ahead(start_date, num_days):
    """Calculate future date accounting for trading days only (Mon-Fri)"""
    current = start_date
    days_added = 0
    
    while days_added < num_days:
        current += timedelta(days=1)
        # Only count weekdays (0=Monday, 4=Friday)
        if current.weekday() < 5:
            days_added += 1
    
    return current.strftime('%Y-%m-%d')

def get_stock_data_yf(symbol, period="3mo"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
        
        info = stock.info
        
        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': round(hist['Close'].iloc[-1], 2),
            'history': hist,
            'info': info
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
        return None

def calculate_indicators(hist):
    """Calculate technical indicators"""
    df = hist.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean() if len(df) >= 100 else None
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # MFI (Money Flow Index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # ADX (Average Directional Index)
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(window=14).mean()
    
    # ATR (Average True Range)
    df['ATR'] = atr
    
    # ROC (Rate of Change)
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
    
    return df

def calculate_score(indicators, info):
    """
    Calculate comprehensive score from 0-13
    - 10 points from technical indicators
    - +1 for strong volume (> -20%)
    - +1 for positive momentum (MACD > 0)
    - +1 for good timing (RSI rising and > 30)
    """
    score = 0
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2] if len(indicators) > 1 else latest
    
    # RSI (0-2 points)
    rsi = latest['RSI']
    if 30 <= rsi <= 70:
        score += 2
    elif 25 <= rsi < 30 or 70 < rsi <= 75:
        score += 1
    
    # MACD (0-2 points)
    macd = latest['MACD']
    signal = latest['Signal']
    if macd > signal and macd > 0:
        score += 2
    elif macd > signal:
        score += 1
    
    # Price vs SMA (0-2 points)
    price = latest['Close']
    if price > latest['SMA_20'] and price > latest['SMA_50']:
        score += 2
    elif price > latest['SMA_20']:
        score += 1
    
    # Bollinger Bands (0-1 point)
    if price <= latest['BB_Lower']:
        score += 1
    
    # MFI (0-1 point)
    mfi = latest['MFI']
    if 20 <= mfi <= 80:
        score += 1
    
    # ADX (0-1 point)
    adx = latest['ADX']
    if adx > 25:
        score += 1
    
    # Volume trend (0-0.5 points)
    current_volume = latest['Volume']
    avg_volume = indicators['Volume'].tail(20).mean()
    if current_volume > avg_volume:
        score += 0.5
    
    # ATR for volatility (0-0.5 points)
    atr = latest['ATR']
    if atr < price * 0.05:  # Low volatility
        score += 0.5
    
    # ROC momentum (0-0.5 points)
    roc = latest['ROC']
    if roc > 0:
        score += 0.5
    
    # Price gaps (0-0.5 points)
    if len(indicators) > 1:
        gap = abs(latest['Open'] - prev['Close']) / prev['Close']
        if gap < 0.02:  # No significant gap
            score += 0.5
    
    # === NEW: +3 bonus points for entry signals ===
    
    # +1 for strong volume (not too weak)
    volume_diff_pct = ((current_volume - avg_volume) / avg_volume) * 100
    if volume_diff_pct > -20:  # Volume not critically low
        score += 1
    
    # +1 for positive momentum (MACD > 0)
    if macd > 0:
        score += 1
    
    # +1 for good timing (RSI rising and healthy)
    rsi_rising = rsi > prev['RSI'] if len(indicators) > 1 else False
    if rsi > 30 and rsi_rising:
        score += 1
    
    return round(score, 1)

def get_signal(score):
    """
    Convert score (0-13) to trading signal
    """
    if score >= 11:
        return "Super Strong Buy"
    elif score >= 9:
        return "Strong Buy"
    elif score >= 7:
        return "Buy"
    elif score >= 5:
        return "Hold"
    else:
        return "Sell"

def is_market_open():
    """Check if US market is currently open"""
    now = datetime.now()
    # US Eastern Time market hours: 9:30 AM - 4:00 PM
    # Simplified check - just see if it's a weekday
    return now.weekday() < 5

@app.get("/")
async def root():
    return {
        "name": "OptiMax Stock Analysis API",
        "version": "5.0.0",
        "description": "Shariah-compliant stock analysis with enhanced Claude AI insights",
        "endpoints": {
            "/top-opportunities": "Get top stock opportunities",
            "/analysis/{symbol}": "Get detailed analysis for a specific stock"
        }
    }

@app.get("/top-opportunities")
async def get_top_opportunities(limit: int = 10):
    """Get top stock opportunities based on technical analysis"""
    
    cache_key = f"top_opps_{limit}"
    if cache_key in cache:
        return cache[cache_key]
    
    opportunities = []
    
    for symbol in SHARIAH_STOCKS:
        try:
            data = get_stock_data_yf(symbol)
            if not data:
                continue
            
            indicators = calculate_indicators(data['history'])
            score = calculate_score(indicators, data['info'])
            
            latest = indicators.iloc[-1]
            prev_close = data['history']['Close'].iloc[-2] if len(data['history']) > 1 else latest['Close']
            change = latest['Close'] - prev_close
            change_pct = (change / prev_close) * 100
            
            opportunities.append({
                'symbol': symbol,
                'name': data['name'],
                'price': data['price'],
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'score': score,
                'signal': get_signal(score),
                'rsi': round(latest['RSI'], 2),
                'macd': round(latest['MACD'], 4)
            })
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            continue
    
    # Sort by score
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    
    result = {
        "total_analyzed": len(SHARIAH_STOCKS),
        "top_opportunities": opportunities[:limit],
        "market_open": is_market_open(),
        "updated_at": datetime.now().isoformat()
    }
    
    cache[cache_key] = result
    return result

@app.get("/analysis/{symbol}")
async def get_detailed_analysis(symbol: str):
    """Get comprehensive analysis for a specific stock including Claude AI insights"""
    
    symbol = symbol.upper()
    
    if symbol not in SHARIAH_STOCKS:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found in Shariah-compliant list")
    
    cache_key = f"analysis_{symbol}"
    if cache_key in cache:
        return cache[cache_key]
    
    data = get_stock_data_yf(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")
    
    indicators = calculate_indicators(data['history'])
    score = calculate_score(indicators, data['info'])
    
    latest = indicators.iloc[-1]
    prev_close = data['history']['Close'].iloc[-2] if len(data['history']) > 1 else latest['Close']
    change = latest['Close'] - prev_close
    change_pct = (change / prev_close) * 100
    
    # Volume analysis
    current_volume = int(latest['Volume'])
    avg_volume = int(indicators['Volume'].tail(20).mean())
    volume_diff = current_volume - avg_volume
    volume_diff_pct = ((current_volume - avg_volume) / avg_volume) * 100
    
    # Fundamental data
    info = data['info']
    pe_ratio = info.get('trailingPE', info.get('forwardPE'))
    eps = info.get('trailingEps', info.get('forwardEps'))
    market_cap = info.get('marketCap', 0)
    sector = info.get('sector', 'Unknown')
    
    # Advanced volume analysis
    volume_trend = "صاعد" if current_volume > avg_volume else "هابط"
    is_unusual_volume = bool(abs(volume_diff_pct) > 50)
    
    # Market context
    daily_trend = "صاعد" if change_pct > 0 else "هابط"
    
    analysis_data = {
        "symbol": symbol,
        "name": data['name'],
        "price": data['price'],
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "score": score,
        "signal": get_signal(score),
        "indicators": {
            "rsi": round(latest['RSI'], 2),
            "macd": round(latest['MACD'], 4),
            "macd_signal": round(latest['Signal'], 4),
            "sma_20": round(latest['SMA_20'], 2),
            "sma_50": round(latest['SMA_50'], 2),
            "sma_100": round(latest['SMA_100'], 2) if latest['SMA_100'] and not pd.isna(latest['SMA_100']) else None,
            "bb_upper": round(latest['BB_Upper'], 2),
            "bb_middle": round(latest['BB_Middle'], 2),
            "bb_lower": round(latest['BB_Lower'], 2),
            "mfi": round(latest['MFI'], 2),
            "adx": round(latest['ADX'], 2),
            "atr": round(latest['ATR'], 2),
            "roc": round(latest['ROC'], 2)
        },
        "fundamentals": {
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
            "eps": round(eps, 2) if eps else None,
            "market_cap": market_cap,
            "sector": sector
        },
        "volume_analysis": {
            "current": current_volume,
            "average": avg_volume,
            "difference": volume_diff,
            "difference_pct": round(volume_diff_pct, 2),
            "trend": volume_trend,
            "is_unusual": is_unusual_volume
        },
        "market_context": {
            "daily_trend": daily_trend,
            "sector_strength": "متوسط"
        }
    }
    
    # Get Claude AI analysis
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Determine trading days based on score
        if score >= 9:
            trading_days = 7
        elif score >= 7:
            trading_days = 5
        elif score >= 5:
            trading_days = 3
        else:
            trading_days = 1
        
        today = datetime.now()
        valid_until = calculate_trading_days_ahead(today, trading_days)
        
        prompt = f"""أنت محلل مالي خبير متخصص في الأسهم الشرعية. قم بتحليل السهم التالي وقدم رأيك بصيغة JSON فقط بدون أي نص إضافي:

السهم: {symbol} - {data['name']}
السعر الحالي: ${data['price']}
التغير اليومي: {change_pct:+.2f}%

المؤشرات الفنية:
- النقاط: {score}/13
- RSI: {latest['RSI']:.2f}
- MACD: {latest['MACD']:.4f}
- ADX: {latest['ADX']:.2f}
- MFI: {latest['MFI']:.2f}
- ROC: {latest['ROC']:.2f}%

البيانات الأساسية:
- P/E Ratio: {pe_ratio if pe_ratio else 'N/A'}
- EPS: ${eps if eps else 'N/A'}
- Market Cap: ${market_cap/1e9:.1f}B
- القطاع: {sector}

تحليل حجم التداول:
- الحجم الحالي: {current_volume:,}
- المتوسط: {avg_volume:,}
- الفرق: {volume_diff_pct:+.2f}%
- الاتجاه: {volume_trend}
- غير طبيعي: {"نعم" if is_unusual_volume else "لا"}

السياق:
- الاتجاه اليومي: {daily_trend}
- التاريخ: {today.strftime('%Y-%m-%d')}

أعطني تحليلاً شاملاً بصيغة JSON التالية فقط (بدون ```json):
{{
  "confidence": 85,
  "success_rating": 8.5,
  "profit_probability": 78,
  "entry_point": {data['price']},
  "target_short": {data['price'] * 1.08},
  "target_medium": {data['price'] * 1.15},
  "stop_loss": {data['price'] * 0.94},
  "valid_until": "{valid_until}",
  "trading_days": {trading_days},
  "summary": "تحليل شامل يشمل المؤشرات الفنية والأساسية وحجم التداول...",
  "risks": ["مخاطرة 1", "مخاطرة 2", "مخاطرة 3"],
  "opportunities": ["فرصة 1", "فرصة 2", "فرصة 3"],
  "alerts": ["تنبيه 1", "تنبيه 2", "تنبيه 3"],
  "sector_flow": "تحليل تدفق السيولة في القطاع...",
  "historical_success": "نسبة النجاح التاريخية للإشارات المماثلة...",
  "recommendation": "التوصية النهائية بناءً على جميع العوامل...",
  "glossary": {{
    "RSI": "مؤشر القوة النسبية - يقيس...",
    "MACD": "تقارب وتباعد المتوسطات - يكشف...",
    "Volume": "حجم التداول - كم سهم...",
    "P/E Ratio": "نسبة السعر للأرباح...",
    "Support": "الدعم - سعر يصعب...",
    "Resistance": "المقاومة - سعر يصعب...",
    "Money Flow": "تدفق الأموال - هل الأموال...",
    "Institutional Buying": "شراء المؤسسات - عندما البنوك..."
  }}
}}"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        # Clean response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        import json
        claude_analysis = json.loads(response_text)
        analysis_data["claude_analysis"] = claude_analysis
        
    except Exception as e:
        print(f"Claude API error: {str(e)}")
        analysis_data["claude_analysis"] = None
    
    cache[cache_key] = analysis_data
    return analysis_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

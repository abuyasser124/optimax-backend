from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import anthropic
import os
from cachetools import TTLCache

app = FastAPI(title="OptiMax Stock Analysis API", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = TTLCache(maxsize=100, ttl=1800)

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
    current = start_date
    days_added = 0
    while days_added < num_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days_added += 1
    return current.strftime('%Y-%m-%d')

def get_stock_data_yf(symbol, period="3mo"):
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

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return obv

def calculate_stochastic(df, period=14):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=3).mean()
    return k, d

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def detect_candlestick_pattern(df):
    if len(df) < 2:
        return "غير محدد"
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    open_p = latest['Open']
    close = latest['Close']
    high = latest['High']
    low = latest['Low']
    body = abs(close - open_p)
    range_size = high - low
    if range_size == 0:
        return "غير محدد"
    lower_shadow = min(open_p, close) - low
    upper_shadow = high - max(open_p, close)
    if lower_shadow > (2 * body) and upper_shadow < body and close > open_p:
        return "صاعد"
    if upper_shadow > (2 * body) and lower_shadow < body and close < open_p:
        return "هابط"
    if (close > open_p and prev['Close'] < prev['Open'] and 
        close > prev['Open'] and open_p < prev['Close']):
        return "صاعد"
    if (close < open_p and prev['Close'] > prev['Open'] and 
        close < prev['Open'] and open_p > prev['Close']):
        return "هابط"
    if body < (range_size * 0.1):
        return "محايد"
    return "محايد"

def calculate_volume_profile(df):
    if len(df) < 20:
        return "محايد"
    recent_vol = df['Volume'].tail(5).mean()
    avg_vol = df['Volume'].mean()
    if recent_vol > avg_vol * 1.5:
        return "قوي"
    elif recent_vol < avg_vol * 0.5:
        return "ضعيف"
    else:
        return "متوسط"

def calculate_indicators(hist):
    df = hist.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean() if len(df) >= 100 else None
    df['EMA_9'] = calculate_ema(df['Close'], 9)
    df['EMA_21'] = calculate_ema(df['Close'], 21)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
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
    df['ATR'] = atr
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
    df['OBV'] = calculate_obv(df)
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df)
    return df

def calculate_confirmation_signals(indicators):
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2] if len(indicators) > 1 else latest
    confirmations = {}
    positive_count = 0
    if len(indicators) >= 5:
        obv_trend = latest['OBV'] > indicators['OBV'].iloc[-5]
        confirmations['obv'] = "صاعد" if obv_trend else "هابط"
        if obv_trend:
            positive_count += 1
    else:
        confirmations['obv'] = "غير محدد"
    stoch_k = latest['Stoch_K']
    if pd.notna(stoch_k):
        if stoch_k < 20:
            confirmations['stochastic'] = "تشبع بيعي (فرصة)"
            positive_count += 1
        elif stoch_k > 80:
            confirmations['stochastic'] = "تشبع شرائي (حذر)"
        else:
            confirmations['stochastic'] = "محايد"
            positive_count += 0.5
    else:
        confirmations['stochastic'] = "غير محدد"
    pattern = detect_candlestick_pattern(indicators)
    confirmations['candlestick'] = pattern
    if pattern == "صاعد":
        positive_count += 1
    elif pattern == "محايد":
        positive_count += 0.5
    vol_profile = calculate_volume_profile(indicators)
    confirmations['volume_profile'] = vol_profile
    if vol_profile == "قوي":
        positive_count += 1
    elif vol_profile == "متوسط":
        positive_count += 0.5
    if positive_count >= 3:
        verdict = "ادخل الآن!"
    elif positive_count >= 2:
        verdict = "راقب - تحتاج تأكيد"
    else:
        verdict = "لا تدخل!"
    return {
        'signals': confirmations,
        'positive_count': round(positive_count, 1),
        'total': 4,
        'verdict': verdict
    }

def calculate_score(indicators, info):
    score = 0
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2] if len(indicators) > 1 else latest
    rsi = latest['RSI']
    if 30 <= rsi <= 70:
        score += 2
    elif 25 <= rsi < 30 or 70 < rsi <= 75:
        score += 1
    macd = latest['MACD']
    signal = latest['Signal']
    if macd > signal and macd > 0:
        score += 2
    elif macd > signal:
        score += 1
    price = latest['Close']
    if price > latest['SMA_20'] and price > latest['SMA_50']:
        score += 2
    elif price > latest['SMA_20']:
        score += 1
    if price <= latest['BB_Lower']:
        score += 1
    mfi = latest['MFI']
    if 20 <= mfi <= 80:
        score += 1
    adx = latest['ADX']
    if adx > 25:
        score += 1
    current_volume = latest['Volume']
    avg_volume = indicators['Volume'].tail(20).mean()
    if current_volume > avg_volume:
        score += 0.5
    atr = latest['ATR']
    if atr < price * 0.05:
        score += 0.5
    roc = latest['ROC']
    if roc > 0:
        score += 0.5
    if len(indicators) > 1:
        gap = abs(latest['Open'] - prev['Close']) / prev['Close']
        if gap < 0.02:
            score += 0.5
    volume_diff_pct = ((current_volume - avg_volume) / avg_volume) * 100
    if volume_diff_pct > -20:
        score += 1
    if macd > 0:
        score += 1
    rsi_rising = rsi > prev['RSI'] if len(indicators) > 1 else False
    if rsi > 30 and rsi_rising:
        score += 1
    return round(score, 1)

def get_signal(score):
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
    now = datetime.now()
    return now.weekday() < 5

@app.get("/")
async def root():
    return {
        "name": "OptiMax Stock Analysis API",
        "version": "6.0.0",
        "description": "Enhanced Shariah-compliant stock analysis with advanced confirmation signals",
        "endpoints": {
            "/top-opportunities": "Get top stock opportunities",
            "/analysis/{symbol}": "Get detailed analysis for a specific stock"
        }
    }

@app.get("/top-opportunities")
async def get_top_opportunities(limit: int = 10):
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
    confirmation = calculate_confirmation_signals(indicators)
    latest = indicators.iloc[-1]
    prev_close = data['history']['Close'].iloc[-2] if len(data['history']) > 1 else latest['Close']
    change = latest['Close'] - prev_close
    change_pct = (change / prev_close) * 100
    current_volume = int(latest['Volume'])
    avg_volume = int(indicators['Volume'].tail(20).mean())
    volume_diff = current_volume - avg_volume
    volume_diff_pct = ((current_volume - avg_volume) / avg_volume) * 100
    info = data['info']
    pe_ratio = info.get('trailingPE', info.get('forwardPE'))
    eps = info.get('trailingEps', info.get('forwardEps'))
    market_cap = info.get('marketCap', 0)
    sector = info.get('sector', 'Unknown')
    volume_trend = "صاعد" if current_volume > avg_volume else "هابط"
    is_unusual_volume = bool(abs(volume_diff_pct) > 50)
    daily_trend = "صاعد" if change_pct > 0 else "هابط"
    analysis_data = {
        "symbol": symbol,
        "name": data['name'],
        "price": data['price'],
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "score": score,
        "signal": get_signal(score),
        "confirmation": confirmation,
        "indicators": {
            "rsi": round(latest['RSI'], 2),
            "macd": round(latest['MACD'], 4),
            "macd_signal": round(latest['Signal'], 4),
            "sma_20": round(latest['SMA_20'], 2),
            "sma_50": round(latest['SMA_50'], 2),
            "sma_100": round(latest['SMA_100'], 2) if latest['SMA_100'] and not pd.isna(latest['SMA_100']) else None,
            "ema_9": round(latest['EMA_9'], 2),
            "ema_21": round(latest['EMA_21'], 2),
            "bb_upper": round(latest['BB_Upper'], 2),
            "bb_middle": round(latest['BB_Middle'], 2),
            "bb_lower": round(latest['BB_Lower'], 2),
            "mfi": round(latest['MFI'], 2),
            "adx": round(latest['ADX'], 2),
            "atr": round(latest['ATR'], 2),
            "roc": round(latest['ROC'], 2),
            "stoch_k": round(latest['Stoch_K'], 2) if pd.notna(latest['Stoch_K']) else None,
            "stoch_d": round(latest['Stoch_D'], 2) if pd.notna(latest['Stoch_D']) else None
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
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
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
        stoch_k_display = f"{latest['Stoch_K']:.2f}" if pd.notna(latest['Stoch_K']) else 'N/A'
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
- Stochastic K: {stoch_k_display}

إشارات التأكيد ({confirmation['positive_count']}/4):
- OBV: {confirmation['signals']['obv']}
- Stochastic: {confirmation['signals']['stochastic']}
- Candlestick: {confirmation['signals']['candlestick']}
- Volume Profile: {confirmation['signals']['volume_profile']}
- الحكم: {confirmation['verdict']}

البيانات الأساسية:
- P/E Ratio: {pe_ratio if pe_ratio else 'N/A'}
- EPS: ${eps if eps else 'N/A'}
- Market Cap: ${market_cap/1e9:.1f}B
- القطاع: {sector}

تحليل حجم التداول:
- الحجم الحالي: {current_volume:,}
- المتوسط: {avg_volume:,}
- الفرق: {volume_diff_pct:+.2f}%

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
  "summary": "تحليل شامل يشمل المؤشرات الفنية وإشارات التأكيد...",
  "risks": ["مخاطرة 1", "مخاطرة 2", "مخاطرة 3"],
  "opportunities": ["فرصة 1", "فرصة 2", "فرصة 3"],
  "alerts": ["تنبيه 1", "تنبيه 2", "تنبيه 3"],
  "sector_flow": "تحليل تدفق السيولة في القطاع...",
  "historical_success": "نسبة النجاح التاريخية للإشارات المماثلة...",
  "recommendation": "التوصية النهائية بناءً على جميع العوامل...",
  "glossary": {{
    "RSI": "مؤشر القوة النسبية - يقيس...",
    "MACD": "تقارب وتباعد المتوسطات - يكشف...",
    "OBV": "حجم التوازن - يتتبع تدفق الأموال...",
    "Stochastic": "مذبذب عشوائي - يكشف التشبع...",
    "Candlestick": "الشموع اليابانية - أنماط السعر...",
    "Volume Profile": "ملف الحجم - توزيع السيولة...",
    "EMA": "المتوسط المتحرك الأسي - أسرع من SMA...",
    "Support": "الدعم - سعر يصعب كسره للأسفل..."
  }}
}}"""
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text.strip()
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

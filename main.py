from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import anthropic
import os
from cachetools import TTLCache
import re
import json

app = FastAPI(title="OptiMax Stock Analysis API", version="8.0.0")

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
    "BIIB", "ILMN", "CDW", "GFS", "WBD", "MDB", "SMCI", "CRWD", "ZM",
    "MRNA", "ALGN", "ENPH", "DLTR", "LCID", "RIVN", "BMRN", "NTES", "JD", "BIDU",
    "PDD", "BILI", "LI", "XPEV", "NIO", "BABA", "TME", "VIPS", "AMGN"
]

def convert_numpy(obj):
    """تحويل numpy types إلى Python native types"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

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
        current_price = hist['Close'].iloc[-1]
        if current_price < 1:
            return None
        info = stock.info
        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': round(current_price, 2),
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

def calculate_support_resistance_advanced(hist, current_price):
    df = hist.tail(90)
    peaks = []
    for i in range(2, len(df) - 2):
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and 
            df['High'].iloc[i] > df['High'].iloc[i-2] and
            df['High'].iloc[i] > df['High'].iloc[i+1] and 
            df['High'].iloc[i] > df['High'].iloc[i+2]):
            level = df['High'].iloc[i]
            touches = 0
            for j in range(len(df)):
                if abs(df['High'].iloc[j] - level) / level < 0.02:
                    touches += 1
            peaks.append({'price': level, 'strength': touches, 'distance': abs(current_price - level)})
    troughs = []
    for i in range(2, len(df) - 2):
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i-2] and
            df['Low'].iloc[i] < df['Low'].iloc[i+1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i+2]):
            level = df['Low'].iloc[i]
            touches = 0
            for j in range(len(df)):
                if abs(df['Low'].iloc[j] - level) / level < 0.02:
                    touches += 1
            troughs.append({'price': level, 'strength': touches, 'distance': abs(current_price - level)})
    peaks.sort(key=lambda x: (x['strength'], -x['distance']), reverse=True)
    troughs.sort(key=lambda x: (x['strength'], -x['distance']), reverse=True)
    resistances = [p['price'] for p in peaks if p['price'] > current_price][:3]
    supports = [t['price'] for t in troughs if t['price'] < current_price][:3]
    return {
        'resistance_1': round(resistances[0], 2) if len(resistances) > 0 else None,
        'resistance_2': round(resistances[1], 2) if len(resistances) > 1 else None,
        'resistance_3': round(resistances[2], 2) if len(resistances) > 2 else None,
        'support_1': round(supports[0], 2) if len(supports) > 0 else None,
        'support_2': round(supports[1], 2) if len(supports) > 1 else None,
        'support_3': round(supports[2], 2) if len(supports) > 2 else None
    }

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

def analyze_technical_signals(indicators, volume_analysis):
    """
    المرحلة 1: تحليل الإشارات الفنية البسيط
    """
    latest = indicators.iloc[-1]
    signals_analysis = []
    
    # 1. RSI
    rsi = convert_numpy(latest['RSI'])
    if rsi and not pd.isna(rsi):
        rsi = float(rsi)
        if 40 <= rsi <= 60:
            signals_analysis.append({'name': 'RSI', 'value': round(rsi, 2), 'status': '⚪ محايد', 'score': 1.8, 'max': 2.0, 'interpretation': 'منطقة متوازنة'})
        elif 35 <= rsi < 40:
            signals_analysis.append({'name': 'RSI', 'value': round(rsi, 2), 'status': '🟢 إيجابي', 'score': 1.6, 'max': 2.0, 'interpretation': 'قرب منطقة شراء'})
        elif 60 < rsi <= 65:
            signals_analysis.append({'name': 'RSI', 'value': round(rsi, 2), 'status': '🟢 إيجابي', 'score': 1.6, 'max': 2.0, 'interpretation': 'قوة معتدلة'})
        elif rsi > 70:
            signals_analysis.append({'name': 'RSI', 'value': round(rsi, 2), 'status': '🔴 تشبع شرائي', 'score': 0.5, 'max': 2.0, 'interpretation': 'احتمال تصحيح'})
        elif rsi < 30:
            signals_analysis.append({'name': 'RSI', 'value': round(rsi, 2), 'status': '🟢 فرصة قوية', 'score': 2.0, 'max': 2.0, 'interpretation': 'تشبع بيعي حاد'})
        else:
            signals_analysis.append({'name': 'RSI', 'value': round(rsi, 2), 'status': '⚪ محايد', 'score': 1.0, 'max': 2.0, 'interpretation': 'حالة عادية'})
    
    # 2. MACD
    macd = convert_numpy(latest['MACD'])
    signal = convert_numpy(latest['Signal'])
    if macd and signal and not pd.isna(macd) and not pd.isna(signal):
        macd = float(macd)
        signal = float(signal)
        diff = macd - signal
        if macd > 0 and diff > 0:
            signals_analysis.append({'name': 'MACD', 'value': round(macd, 2), 'status': '🟢 إيجابي قوي', 'score': 2.0, 'max': 2.0, 'interpretation': 'اتجاه صاعد قوي'})
        elif macd > 0 and diff <= 0:
            signals_analysis.append({'name': 'MACD', 'value': round(macd, 2), 'status': '🟡 إيجابي ضعيف', 'score': 1.2, 'max': 2.0, 'interpretation': 'بداية ضعف'})
        elif macd <= 0 and diff > 0:
            signals_analysis.append({'name': 'MACD', 'value': round(macd, 2), 'status': '🟢 بداية انعكاس', 'score': 1.0, 'max': 2.0, 'interpretation': 'إشارة شراء محتملة'})
        else:
            signals_analysis.append({'name': 'MACD', 'value': round(macd, 2), 'status': '🔴 سلبي', 'score': 0.0, 'max': 2.0, 'interpretation': 'اتجاه هابط'})
    
    # 3. ADX
    adx = convert_numpy(latest['ADX'])
    if adx and not pd.isna(adx):
        adx = float(adx)
        if adx >= 40:
            signals_analysis.append({'name': 'ADX', 'value': round(adx, 2), 'status': '🟢 اتجاه قوي جداً', 'score': 2.0, 'max': 2.0, 'interpretation': 'ترند واضح'})
        elif adx >= 25:
            signals_analysis.append({'name': 'ADX', 'value': round(adx, 2), 'status': '🟢 اتجاه واضح', 'score': 1.4, 'max': 2.0, 'interpretation': 'قوة متوسطة'})
        elif adx >= 20:
            signals_analysis.append({'name': 'ADX', 'value': round(adx, 2), 'status': '🟡 اتجاه ضعيف', 'score': 0.7, 'max': 2.0, 'interpretation': 'بداية اتجاه'})
        else:
            signals_analysis.append({'name': 'ADX', 'value': round(adx, 2), 'status': '⚪ لا يوجد اتجاه', 'score': 0.2, 'max': 2.0, 'interpretation': 'سوق عرضي'})
    
    # 4. MFI
    mfi = convert_numpy(latest['MFI'])
    if mfi and not pd.isna(mfi):
        mfi = float(mfi)
        if 40 <= mfi <= 60:
            signals_analysis.append({'name': 'MFI', 'value': round(mfi, 2), 'status': '⚪ محايد', 'score': 1.0, 'max': 1.0, 'interpretation': 'تدفق متوازن'})
        elif mfi > 70:
            signals_analysis.append({'name': 'MFI', 'value': round(mfi, 2), 'status': '🔴 تشبع شرائي', 'score': 0.3, 'max': 1.0, 'interpretation': 'تدفق مبالغ'})
        elif mfi < 30:
            signals_analysis.append({'name': 'MFI', 'value': round(mfi, 2), 'status': '🟢 فرصة', 'score': 1.0, 'max': 1.0, 'interpretation': 'تشبع بيعي'})
        else:
            signals_analysis.append({'name': 'MFI', 'value': round(mfi, 2), 'status': '⚪ عادي', 'score': 0.5, 'max': 1.0, 'interpretation': 'حالة طبيعية'})
    
    # 5. ROC
    roc = convert_numpy(latest['ROC'])
    if roc and not pd.isna(roc):
        roc = float(roc)
        if roc >= 10:
            signals_analysis.append({'name': 'ROC', 'value': round(roc, 2), 'status': '🟢 قوي جداً', 'score': 1.5, 'max': 1.5, 'interpretation': 'زخم قوي'})
        elif roc >= 5:
            signals_analysis.append({'name': 'ROC', 'value': round(roc, 2), 'status': '🟢 قوي', 'score': 1.2, 'max': 1.5, 'interpretation': 'زخم جيد'})
        elif roc >= 0:
            signals_analysis.append({'name': 'ROC', 'value': round(roc, 2), 'status': '🟢 إيجابي', 'score': 0.6, 'max': 1.5, 'interpretation': 'نمو خفيف'})
        else:
            signals_analysis.append({'name': 'ROC', 'value': round(roc, 2), 'status': '🔴 سلبي', 'score': 0.0, 'max': 1.5, 'interpretation': 'هبوط'})
    
    # 6. Volume
    vol_diff = float(volume_analysis['difference_pct'])
    if vol_diff > 50:
        signals_analysis.append({'name': 'Volume', 'value': round(vol_diff, 2), 'status': '🟢 قوي جداً', 'score': 1.0, 'max': 1.0, 'interpretation': 'اهتمام كبير'})
    elif vol_diff > 20:
        signals_analysis.append({'name': 'Volume', 'value': round(vol_diff, 2), 'status': '🟢 قوي', 'score': 0.8, 'max': 1.0, 'interpretation': 'نشاط جيد'})
    elif vol_diff > -20:
        signals_analysis.append({'name': 'Volume', 'value': round(vol_diff, 2), 'status': '⚪ متوسط', 'score': 0.5, 'max': 1.0, 'interpretation': 'حجم عادي'})
    else:
        signals_analysis.append({'name': 'Volume', 'value': round(vol_diff, 2), 'status': '🔴 ضعيف جداً', 'score': 0.0, 'max': 1.0, 'interpretation': 'حجم منخفض جداً'})
    
    # 7. Trend Alignment
    price = convert_numpy(latest['Close'])
    sma_20 = convert_numpy(latest['SMA_20'])
    sma_50 = convert_numpy(latest['SMA_50'])
    if price and sma_20 and sma_50:
        price = float(price)
        sma_20 = float(sma_20)
        sma_50 = float(sma_50)
        if price > sma_20 > sma_50:
            signals_analysis.append({'name': 'Trend Alignment', 'value': round(price, 2), 'status': '🟢 ترتيب صاعد', 'score': 2.0, 'max': 2.0, 'interpretation': 'اتجاه قوي'})
        elif price > sma_20:
            signals_analysis.append({'name': 'Trend Alignment', 'value': round(price, 2), 'status': '🟢 جزئي صاعد', 'score': 1.2, 'max': 2.0, 'interpretation': 'اتجاه معتدل'})
        else:
            signals_analysis.append({'name': 'Trend Alignment', 'value': round(price, 2), 'status': '🔴 هابط', 'score': 0.0, 'max': 2.0, 'interpretation': 'اتجاه سلبي'})
    
    # 8. Stochastic
    stoch = convert_numpy(latest['Stoch_K'])
    if stoch and not pd.isna(stoch):
        stoch = float(stoch)
        if stoch < 20:
            signals_analysis.append({'name': 'Stochastic', 'value': round(stoch, 2), 'status': '🟢 فرصة شراء', 'score': 1.0, 'max': 1.0, 'interpretation': 'تشبع بيعي'})
        elif stoch > 80:
            signals_analysis.append({'name': 'Stochastic', 'value': round(stoch, 2), 'status': '🔴 تشبع شرائي', 'score': 0.0, 'max': 1.0, 'interpretation': 'حذر'})
        else:
            signals_analysis.append({'name': 'Stochastic', 'value': round(stoch, 2), 'status': '⚪ محايد', 'score': 0.5, 'max': 1.0, 'interpretation': 'منطقة وسطى'})
    
    # 9. OBV
    if len(indicators) >= 5:
        obv_current = convert_numpy(latest['OBV'])
        obv_prev = convert_numpy(indicators['OBV'].iloc[-5])
        if obv_current and obv_prev:
            if float(obv_current) > float(obv_prev):
                signals_analysis.append({'name': 'OBV', 'value': 0, 'status': '🟢 صاعد', 'score': 1.0, 'max': 1.0, 'interpretation': 'تدفق نقدي إيجابي'})
            else:
                signals_analysis.append({'name': 'OBV', 'value': 0, 'status': '🔴 هابط', 'score': 0.0, 'max': 1.0, 'interpretation': 'تدفق نقدي سلبي'})
    
    # 10. Candlestick
    pattern = detect_candlestick_pattern(indicators)
    if pattern == "صاعد":
        signals_analysis.append({'name': 'Candlestick', 'value': 0, 'status': '🟢 صاعد', 'score': 1.0, 'max': 1.0, 'interpretation': 'نموذج إيجابي'})
    elif pattern == "هابط":
        signals_analysis.append({'name': 'Candlestick', 'value': 0, 'status': '🔴 هابط', 'score': 0.0, 'max': 1.0, 'interpretation': 'نموذج سلبي'})
    else:
        signals_analysis.append({'name': 'Candlestick', 'value': 0, 'status': '⚪ محايد', 'score': 0.5, 'max': 1.0, 'interpretation': 'لا إشارة واضحة'})
    
    # الخلاصة
    total_score = sum(float(s['score']) for s in signals_analysis)
    total_max = sum(float(s['max']) for s in signals_analysis)
    
    positive_count = len([s for s in signals_analysis if '🟢' in s['status']])
    neutral_count = len([s for s in signals_analysis if '⚪' in s['status'] or '🟡' in s['status']])
    negative_count = len([s for s in signals_analysis if '🔴' in s['status']])
    
    percentage = (total_score / total_max) * 100 if total_max > 0 else 0
    
    if percentage >= 75:
        overall = 'قوية جداً'
    elif percentage >= 60:
        overall = 'قوية'
    elif percentage >= 45:
        overall = 'متوسطة'
    elif percentage >= 30:
        overall = 'ضعيفة'
    else:
        overall = 'ضعيفة جداً'
    
    return {
        'signals': signals_analysis,
        'total_score': round(total_score, 1),
        'total_max': round(total_max, 1),
        'percentage': round(percentage, 1),
        'positive_count': int(positive_count),
        'neutral_count': int(neutral_count),
        'negative_count': int(negative_count),
        'overall_assessment': overall
    }

def get_claude_deep_analysis(
    symbol,
    price,
    change_pct,
    technical_signals,
    support_resistance,
    momentum,
    risk_reward,
    opportunity_quality,
    position_risk,
    late_entry,
    volume_analysis,
    indicators,
    recent_performance
):
    """المرحلة 2: تحليل Claude AI"""
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        latest = indicators.iloc[-1]
        
        signals_detail = "\n".join([
            f"- {s['name']}: {s['value']} | {s['status']} | {s['score']}/{s['max']} | {s['interpretation']}"
            for s in technical_signals['signals']
        ])
        
        r1 = support_resistance.get('resistance_1')
        s1 = support_resistance.get('support_1')
        
        dist_r1 = ((r1 - price) / price * 100) if r1 else None
        dist_s1 = ((price - s1) / price * 100) if s1 else None
        
        prompt = f"""أنت محلل مالي. قدم تحليلاً بصيغة JSON:

السهم: {symbol} | السعر: ${price:.2f} | التغير: {change_pct:+.2f}%

المؤشرات:
{signals_detail}

النتيجة: {technical_signals['total_score']}/{technical_signals['total_max']} ({technical_signals['percentage']}%)
الدعم/المقاومة: R1=${r1 or 'N/A'} | S1=${s1 or 'N/A'}
الزخم: {momentum['score']}/10 | R/R: {risk_reward['ratio'] if risk_reward else 'N/A'} | Grade: {opportunity_quality['grade']}

JSON:
{{
  "detailed_indicators": [
    {{"name": "اسم", "status": "حالة", "interpretation": "تفسير", "weight": "وزن", "impact": "تأثير"}}
  ],
  "support_resistance_analysis": "تحليل نصي",
  "momentum_analysis": "تحليل نصي",
  "risk_reward_analysis": "تحليل نصي",
  "opportunity_quality_analysis": "تحليل نصي",
  "volume_and_liquidity": "تحليل نصي",
  "final_recommendation": {{
    "decision": "ادخل أو راقب أو لا تدخل",
    "comprehensive_analysis": "3-5 جمل",
    "reasons": ["سبب1", "سبب2"],
    "alternatives": "بدائل",
    "conditions": "شروط أو null",
    "confidence": 70,
    "success_probability": 60
  }}
}}"""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3500,
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
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Claude error: {str(e)}")
        return {
            "detailed_indicators": [{"name": "خطأ", "status": "N/A", "interpretation": str(e), "weight": "N/A", "impact": "N/A"}],
            "support_resistance_analysis": "غير متوفر",
            "momentum_analysis": "غير متوفر",
            "risk_reward_analysis": "غير متوفر",
            "opportunity_quality_analysis": "غير متوفر",
            "volume_and_liquidity": "غير متوفر",
            "final_recommendation": {
                "decision": "خطأ",
                "comprehensive_analysis": "حدث خطأ في التحليل",
                "reasons": ["خطأ فني"],
                "alternatives": "استخدم المرحلة 1",
                "conditions": None,
                "confidence": 0,
                "success_probability": 0
            }
        }

def analyze_recent_performance(indicators):
    recent_5d = indicators.tail(5)
    daily_changes = recent_5d['Close'].pct_change()
    avg_change = daily_changes.mean() * 100
    volatility = daily_changes.std() * 100
    positive_days = (daily_changes > 0).sum()
    return {
        'trend': 'صاعد قوي' if positive_days >= 4 else 'صاعد' if positive_days >= 3 else 'هابط',
        'avg_daily_change': round(avg_change, 2),
        'volatility': round(volatility, 2),
        'positive_days': int(positive_days),
        'consistency': 'عالية' if positive_days >= 4 else 'متوسطة' if positive_days >= 2 else 'منخفضة'
    }

def calculate_momentum_score(indicators):
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2] if len(indicators) > 1 else latest
    momentum = 0
    roc = convert_numpy(latest['ROC'])
    macd = convert_numpy(latest['MACD'])
    rsi = convert_numpy(latest['RSI'])
    rsi_prev = convert_numpy(prev['RSI'])
    
    if roc and roc > 10:
        momentum += 3
    elif roc and roc > 5:
        momentum += 2
    elif roc and roc > 0:
        momentum += 1
    
    if macd and abs(macd) > 3:
        momentum += 3
    elif macd and abs(macd) > 2:
        momentum += 2
    elif macd and abs(macd) > 1:
        momentum += 1
    
    if rsi and rsi_prev and rsi > rsi_prev and rsi > 50:
        momentum += 2
    elif rsi and rsi_prev and rsi > rsi_prev:
        momentum += 1
    
    avg_volume = indicators['Volume'].tail(20).mean()
    if latest['Volume'] > avg_volume * 2:
        momentum += 2
    elif latest['Volume'] > avg_volume * 1.5:
        momentum += 1
    
    return {
        'score': int(momentum),
        'strength': 'قوي جداً' if momentum >= 8 else 'قوي' if momentum >= 6 else 'متوسط' if momentum >= 4 else 'ضعيف'
    }

def detect_late_entry(indicators):
    latest = indicators.iloc[-1]
    warnings = []
    risk_level = 'low'
    rsi = convert_numpy(latest['RSI'])
    stoch_k = convert_numpy(latest['Stoch_K'])
    
    if rsi and stoch_k and rsi > 75 and stoch_k > 85:
        warnings.append('⚠️ تشبع شرائي حاد - احتمال نهاية الموجة!')
        risk_level = 'critical'
    elif rsi and stoch_k and rsi > 70 and stoch_k > 80:
        warnings.append('⚠️ تشبع شرائي - قد تكون متأخر!')
        risk_level = 'high'
    elif (rsi and rsi > 65) or (stoch_k and stoch_k > 75):
        warnings.append('⚠️ قرب التشبع - راقب عن كثب')
        risk_level = 'medium'
    
    return {
        'warnings': warnings,
        'risk_level': risk_level,
        'is_late': risk_level in ['high', 'critical']
    }

def check_position_risk(price, resistance, support):
    risk_analysis = {
        'distance_to_resistance': None,
        'distance_to_support': None,
        'position_quality': 'neutral',
        'warnings': [],
        'opportunities': []
    }
    if resistance:
        dist_res = (resistance - price) / price * 100
        risk_analysis['distance_to_resistance'] = round(dist_res, 2)
        if dist_res < 2:
            risk_analysis['warnings'].append('⚠️ قريب جداً من المقاومة!')
            risk_analysis['position_quality'] = 'poor'
        elif dist_res < 5:
            risk_analysis['warnings'].append('⚠️ المقاومة قريبة')
            risk_analysis['position_quality'] = 'fair'
    else:
        risk_analysis['opportunities'].append('✅ لا توجد مقاومة واضحة')
    
    if support:
        dist_sup = (price - support) / price * 100
        risk_analysis['distance_to_support'] = round(dist_sup, 2)
        if dist_sup < 3:
            risk_analysis['opportunities'].append('✅ قريب من الدعم')
            if risk_analysis['position_quality'] == 'neutral':
                risk_analysis['position_quality'] = 'good'
    
    return risk_analysis

def calculate_risk_reward(price, target, stop_loss):
    if not target or not stop_loss or stop_loss >= price:
        return None
    potential_gain = (target - price) / price * 100
    potential_loss = (price - stop_loss) / price * 100
    if potential_loss == 0:
        return None
    ratio = potential_gain / potential_loss
    return {
        'ratio': round(ratio, 2),
        'potential_gain': round(potential_gain, 1),
        'potential_loss': round(potential_loss, 1),
        'verdict': 'ممتاز' if ratio >= 2.5 else 'جيد' if ratio >= 1.5 else 'ضعيف'
    }

def calculate_opportunity_quality(score, indicators, support_resistance, position_risk, late_entry, momentum, risk_reward=None):
    quality_score = 0
    factors = []
    percentage = (score / 14.5) * 100
    
    if percentage >= 75:
        quality_score += 3
        factors.append('نقاط ممتازة')
    elif percentage >= 60:
        quality_score += 2
        factors.append('نقاط جيدة')
    elif percentage >= 45:
        quality_score += 1
    
    if position_risk['position_quality'] == 'good':
        quality_score += 2
    elif position_risk['position_quality'] == 'fair':
        quality_score += 1
    
    if not late_entry['is_late']:
        quality_score += 2
    
    if momentum['score'] >= 8:
        quality_score += 2
    elif momentum['score'] >= 6:
        quality_score += 1
    
    latest = indicators.iloc[-1]
    adx = convert_numpy(latest['ADX'])
    if adx and adx > 25:
        quality_score += 0.7
    
    if risk_reward and risk_reward['ratio'] >= 2.0:
        quality_score += 1.4
    elif risk_reward and risk_reward['ratio'] >= 1.5:
        quality_score += 1.0
    
    if quality_score >= 10:
        grade = 'A+'
    elif quality_score >= 9:
        grade = 'A'
    elif quality_score >= 7:
        grade = 'B'
    elif quality_score >= 5:
        grade = 'C'
    else:
        grade = 'D'
    
    return {
        'score': round(quality_score, 1),
        'grade': grade,
        'factors': factors,
        'recommendation': 'فرصة ذهبية!' if grade in ['A+', 'A'] else 'فرصة جيدة' if grade == 'B' else 'راقب' if grade == 'C' else 'تجنب'
    }

def analyze_daily_change_context(change_pct, indicators, support_resistance, price):
    context = {}
    if change_pct < -1:
        context['type'] = 'تصحيح صحي'
        context['message'] = '✅ النزول اليومي طبيعي'
    elif change_pct > 1:
        context['type'] = 'زخم قوي'
        context['message'] = '✅ الصعود مدعوم'
    else:
        context['type'] = 'طبيعي'
        context['message'] = 'حركة سعرية عادية'
    return context

def calculate_confirmation_signals(indicators):
    latest = indicators.iloc[-1]
    signals = {}
    score = 0
    
    if len(indicators) >= 5:
        obv_current = convert_numpy(latest['OBV'])
        obv_prev = convert_numpy(indicators['OBV'].iloc[-5])
        if obv_current and obv_prev and float(obv_current) > float(obv_prev):
            signals['obv'] = 'صاعد'
            score += 1
        else:
            signals['obv'] = 'هابط'
    
    stoch = convert_numpy(latest['Stoch_K'])
    if stoch:
        if stoch < 20:
            signals['stochastic'] = 'تشبع بيعي'
            score += 1
        elif stoch > 80:
            signals['stochastic'] = 'تشبع شرائي'
        else:
            signals['stochastic'] = 'محايد'
            score += 0.5
    
    pattern = detect_candlestick_pattern(indicators)
    signals['candlestick'] = pattern
    if pattern == "صاعد":
        score += 1
    elif pattern == "محايد":
        score += 0.5
    
    vol_profile = calculate_volume_profile(indicators)
    signals['volume_profile'] = vol_profile
    if vol_profile == "قوي":
        score += 1
    elif vol_profile == "متوسط":
        score += 0.5
    
    macd = convert_numpy(latest['MACD'])
    signal_line = convert_numpy(latest['Signal'])
    if macd and signal_line:
        if macd > signal_line:
            signals['macd_histogram'] = 'إيجابي'
            score += 1
        else:
            signals['macd_histogram'] = 'سلبي'
    
    if score >= 5:
        verdict = 'إشارة قوية'
    elif score >= 3.5:
        verdict = 'إشارة متوسطة'
    else:
        verdict = 'إشارة ضعيفة'
    
    return {
        'signals': signals,
        'positive_count': round(score, 1),
        'total': 6,
        'verdict': verdict
    }

def get_final_recommendation(score, confirmation, position_risk, late_entry, risk_reward):
    if risk_reward and risk_reward['ratio'] < 0.8:
        return {'action': 'لا تدخل', 'reason': 'R/R ضعيف', 'confidence': 'منخفضة'}
    if position_risk['position_quality'] == 'poor':
        return {'action': 'راقب', 'reason': 'قريب من المقاومة', 'confidence': 'متوسطة'}
    if late_entry['risk_level'] == 'critical':
        return {'action': 'لا تدخل', 'reason': 'تشبع حاد', 'confidence': 'منخفضة'}
    if score >= 10 and confirmation['positive_count'] >= 4.5:
        return {'action': 'ادخل الآن', 'reason': 'مؤشرات قوية', 'confidence': 'عالية'}
    if score >= 8:
        return {'action': 'راقب', 'reason': 'مؤشرات جيدة', 'confidence': 'متوسطة'}
    return {'action': 'لا تدخل', 'reason': 'مؤشرات ضعيفة', 'confidence': 'منخفضة'}

def calculate_score(indicators, info):
    latest = indicators.iloc[-1]
    score = 0
    
    rsi = convert_numpy(latest['RSI'])
    if rsi and 40 <= rsi <= 60:
        score += 2.0
    elif rsi and 35 <= rsi < 40:
        score += 1.5
    
    macd = convert_numpy(latest['MACD'])
    signal = convert_numpy(latest['Signal'])
    if macd and signal and macd > 0 and macd > signal:
        score += 2.0
    elif macd and macd > 0:
        score += 1.0
    
    adx = convert_numpy(latest['ADX'])
    if adx and adx >= 40:
        score += 2.0
    elif adx and adx >= 25:
        score += 1.0
    
    roc = convert_numpy(latest['ROC'])
    if roc and roc >= 10:
        score += 1.5
    elif roc and roc >= 5:
        score += 1.0
    
    price = convert_numpy(latest['Close'])
    sma_20 = convert_numpy(latest['SMA_20'])
    sma_50 = convert_numpy(latest['SMA_50'])
    if price and sma_20 and sma_50 and price > sma_20 > sma_50:
        score += 2.0
    
    return max(0, round(score, 1))

def calculate_targets_advanced(price, confirmation_score, atr, support_resistance):
    if confirmation_score >= 4:
        target_short = support_resistance.get('resistance_1') or round(price + (atr * 2), 2)
        stop_loss = support_resistance.get('support_1') or round(price - (atr * 1.5), 2)
    else:
        target_short = round(price + (atr * 1.5), 2)
        stop_loss = round(price - (atr * 1.2), 2)
    
    return {
        'entry': round(price, 2),
        'target_short': round(target_short, 2),
        'target_medium': round(target_short * 1.05, 2),
        'stop_loss': round(stop_loss, 2),
        'direction': 'صاعد' if confirmation_score >= 4 else 'محايد'
    }

def get_signal(score):
    percentage = (score / 14.5) * 100
    if percentage >= 75:
        return "إشارة قوية جداً"
    elif percentage >= 60:
        return "إشارة قوية"
    elif percentage >= 45:
        return "إشارة إيجابية"
    elif percentage >= 30:
        return "إشارة محايدة"
    else:
        return "إشارة سلبية"

def is_market_open():
    now = datetime.now()
    return now.weekday() < 5

@app.get("/")
async def root():
    return {
        "name": "OptiMax API",
        "version": "8.0.0",
        "status": "running"
    }

@app.get("/top-opportunities")
async def get_top_opportunities(limit: int = 10):
    opportunities = []
    for symbol in SHARIAH_STOCKS[:20]:
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
                'price': float(data['price']),
                'change': float(round(change, 2)),
                'change_pct': float(round(change_pct, 2)),
                'score': float(score),
                'signal': get_signal(score),
                'rsi': float(round(convert_numpy(latest['RSI']), 2)),
                'macd': float(round(convert_numpy(latest['MACD']), 4))
            })
        except:
            continue
    
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    return {
        "top_opportunities": opportunities[:limit],
        "updated_at": datetime.now().isoformat()
    }

@app.get("/analysis/{symbol}")
async def get_detailed_analysis(symbol: str):
    symbol = symbol.upper()
    
    data = get_stock_data_yf(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")
    
    indicators = calculate_indicators(data['history'])
    score = calculate_score(indicators, data['info'])
    confirmation = calculate_confirmation_signals(indicators)
    support_resistance = calculate_support_resistance_advanced(data['history'], data['price'])
    
    latest = indicators.iloc[-1]
    atr = convert_numpy(latest['ATR'])
    targets = calculate_targets_advanced(data['price'], confirmation['positive_count'], atr, support_resistance)
    
    position_risk = check_position_risk(data['price'], support_resistance.get('resistance_1'), support_resistance.get('support_1'))
    late_entry = detect_late_entry(indicators)
    risk_reward = calculate_risk_reward(data['price'], targets['target_short'], targets['stop_loss'])
    momentum = calculate_momentum_score(indicators)
    recent_performance = analyze_recent_performance(indicators)
    opportunity_quality = calculate_opportunity_quality(score, indicators, support_resistance, position_risk, late_entry, momentum, risk_reward)
    
    prev_close = data['history']['Close'].iloc[-2] if len(data['history']) > 1 else latest['Close']
    change = latest['Close'] - prev_close
    change_pct = (change / prev_close) * 100
    
    daily_context = analyze_daily_change_context(change_pct, indicators, support_resistance, data['price'])
    final_recommendation = get_final_recommendation(score, confirmation, position_risk, late_entry, risk_reward)
    
    current_volume = int(convert_numpy(latest['Volume']))
    avg_volume = int(indicators['Volume'].tail(20).mean())
    volume_diff_pct = ((current_volume - avg_volume) / avg_volume) * 100
    
    volume_analysis = {
        'current': current_volume,
        'average': avg_volume,
        'difference': current_volume - avg_volume,
        'difference_pct': float(round(volume_diff_pct, 2)),
        'trend': "صاعد" if current_volume > avg_volume else "هابط",
        'is_unusual': bool(abs(volume_diff_pct) > 50)
    }
    
    technical_signals = analyze_technical_signals(indicators, volume_analysis)
    
    claude_deep_analysis = get_claude_deep_analysis(
        symbol, data['price'], change_pct, technical_signals, support_resistance,
        momentum, risk_reward, opportunity_quality, position_risk, late_entry,
        volume_analysis, indicators, recent_performance
    )
    
    info = data['info']
    pe_ratio = info.get('trailingPE', info.get('forwardPE'))
    eps = info.get('trailingEps', info.get('forwardEps'))
    
    return {
        "symbol": symbol,
        "name": data['name'],
        "price": float(data['price']),
        "change": float(round(change, 2)),
        "change_pct": float(round(change_pct, 2)),
        "score": float(score),
        "signal": get_signal(score),
        "technical_signals": technical_signals,
        "claude_deep_analysis": claude_deep_analysis,
        "confirmation": confirmation,
        "targets": targets,
        "support_resistance": support_resistance,
        "position_risk": position_risk,
        "late_entry_warning": late_entry,
        "risk_reward": risk_reward,
        "momentum_analysis": momentum,
        "recent_performance": recent_performance,
        "opportunity_quality": opportunity_quality,
        "daily_change_context": daily_context,
        "final_recommendation": final_recommendation,
        "indicators": {
            "rsi": float(round(convert_numpy(latest['RSI']), 2)),
            "macd": float(round(convert_numpy(latest['MACD']), 4)),
            "adx": float(round(convert_numpy(latest['ADX']), 2)),
            "mfi": float(round(convert_numpy(latest['MFI']), 2)),
            "atr": float(round(convert_numpy(latest['ATR']), 2)),
            "roc": float(round(convert_numpy(latest['ROC']), 2))
        },
        "fundamentals": {
            "pe_ratio": float(round(pe_ratio, 2)) if pe_ratio else None,
            "eps": float(round(eps, 2)) if eps else None,
            "market_cap": int(info.get('marketCap', 0)),
            "sector": info.get('sector', 'Unknown')
        },
        "volume_analysis": volume_analysis
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

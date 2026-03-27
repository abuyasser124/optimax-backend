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
    "BIIB", "ILMN", "CDW", "GFS", "WBD", "MDB", "SMCI", "CRWD", "WBA", "ZM",
    "MRNA", "ALGN", "ENPH", "DLTR", "LCID", "RIVN", "BMRN", "NTES", "JD", "BIDU",
    "PDD", "BILI", "LI", "XPEV", "NIO", "BABA", "TME", "VIPS", "AMGN"
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
    يعطي كل مؤشر: حالة + نقاط + وزن
    """
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2] if len(indicators) > 1 else latest
    signals_analysis = []
    
    # 1. RSI
    rsi = latest['RSI']
    if pd.notna(rsi):
        if 40 <= rsi <= 60:
            rsi_data = {'name': 'RSI', 'value': rsi, 'status': '⚪ محايد', 'score': 1.8, 'max': 2.0, 'interpretation': 'منطقة متوازنة'}
        elif 35 <= rsi < 40:
            rsi_data = {'name': 'RSI', 'value': rsi, 'status': '🟢 إيجابي', 'score': 1.6, 'max': 2.0, 'interpretation': 'قرب منطقة شراء'}
        elif 60 < rsi <= 65:
            rsi_data = {'name': 'RSI', 'value': rsi, 'status': '🟢 إيجابي', 'score': 1.6, 'max': 2.0, 'interpretation': 'قوة معتدلة'}
        elif 30 <= rsi < 35:
            rsi_data = {'name': 'RSI', 'value': rsi, 'status': '🟢 فرصة شراء', 'score': 1.4, 'max': 2.0, 'interpretation': 'تشبع بيعي'}
        elif 65 < rsi <= 70:
            rsi_data = {'name': 'RSI', 'value': rsi, 'status': '🟡 حذر', 'score': 1.2, 'max': 2.0, 'interpretation': 'قرب التشبع'}
        elif rsi > 70:
            rsi_data = {'name': 'RSI', 'value': rsi, 'status': '🔴 تشبع شرائي', 'score': 0.5, 'max': 2.0, 'interpretation': 'احتمال تصحيح'}
        elif rsi < 30:
            rsi_data = {'name': 'RSI', 'value': rsi, 'status': '🟢 فرصة قوية', 'score': 2.0, 'max': 2.0, 'interpretation': 'تشبع بيعي حاد'}
        else:
            rsi_data = {'name': 'RSI', 'value': rsi, 'status': '⚪ محايد', 'score': 1.0, 'max': 2.0, 'interpretation': 'حالة عادية'}
        signals_analysis.append(rsi_data)
    
    # 2. MACD
    macd = latest['MACD']
    signal = latest['Signal']
    if pd.notna(macd) and pd.notna(signal):
        diff = macd - signal
        if macd > 0 and diff > 0:
            macd_data = {'name': 'MACD', 'value': macd, 'status': '🟢 إيجابي قوي', 'score': 2.0, 'max': 2.0, 'interpretation': 'اتجاه صاعد قوي'}
        elif macd > 0 and diff <= 0:
            macd_data = {'name': 'MACD', 'value': macd, 'status': '🟡 إيجابي ضعيف', 'score': 1.2, 'max': 2.0, 'interpretation': 'بداية ضعف'}
        elif macd <= 0 and diff > 0:
            macd_data = {'name': 'MACD', 'value': macd, 'status': '🟢 بداية انعكاس', 'score': 1.0, 'max': 2.0, 'interpretation': 'إشارة شراء محتملة'}
        else:
            macd_data = {'name': 'MACD', 'value': macd, 'status': '🔴 سلبي', 'score': 0.0, 'max': 2.0, 'interpretation': 'اتجاه هابط'}
        signals_analysis.append(macd_data)
    
    # 3. ADX
    adx = latest['ADX']
    if pd.notna(adx):
        if adx >= 40:
            adx_data = {'name': 'ADX', 'value': adx, 'status': '🟢 اتجاه قوي جداً', 'score': 2.0, 'max': 2.0, 'interpretation': 'ترند واضح'}
        elif adx >= 25:
            adx_data = {'name': 'ADX', 'value': adx, 'status': '🟢 اتجاه واضح', 'score': 1.4, 'max': 2.0, 'interpretation': 'قوة متوسطة'}
        elif adx >= 20:
            adx_data = {'name': 'ADX', 'value': adx, 'status': '🟡 اتجاه ضعيف', 'score': 0.7, 'max': 2.0, 'interpretation': 'بداية اتجاه'}
        else:
            adx_data = {'name': 'ADX', 'value': adx, 'status': '⚪ لا يوجد اتجاه', 'score': 0.2, 'max': 2.0, 'interpretation': 'سوق عرضي'}
        signals_analysis.append(adx_data)
    
    # 4. MFI
    mfi = latest['MFI']
    if pd.notna(mfi):
        if 40 <= mfi <= 60:
            mfi_data = {'name': 'MFI', 'value': mfi, 'status': '⚪ محايد', 'score': 1.0, 'max': 1.0, 'interpretation': 'تدفق متوازن'}
        elif 30 <= mfi < 40:
            mfi_data = {'name': 'MFI', 'value': mfi, 'status': '🟢 إيجابي', 'score': 0.8, 'max': 1.0, 'interpretation': 'تدفق إيجابي'}
        elif 60 < mfi <= 70:
            mfi_data = {'name': 'MFI', 'value': mfi, 'status': '🟡 حذر', 'score': 0.6, 'max': 1.0, 'interpretation': 'قرب التشبع'}
        elif mfi > 70:
            mfi_data = {'name': 'MFI', 'value': mfi, 'status': '🔴 تشبع شرائي', 'score': 0.3, 'max': 1.0, 'interpretation': 'تدفق مبالغ'}
        elif mfi < 30:
            mfi_data = {'name': 'MFI', 'value': mfi, 'status': '🟢 فرصة', 'score': 1.0, 'max': 1.0, 'interpretation': 'تشبع بيعي'}
        else:
            mfi_data = {'name': 'MFI', 'value': mfi, 'status': '⚪ عادي', 'score': 0.5, 'max': 1.0, 'interpretation': 'حالة طبيعية'}
        signals_analysis.append(mfi_data)
    
    # 5. ROC
    roc = latest['ROC']
    if pd.notna(roc):
        if roc >= 10:
            roc_data = {'name': 'ROC', 'value': roc, 'status': '🟢 قوي جداً', 'score': 1.5, 'max': 1.5, 'interpretation': 'زخم قوي'}
        elif roc >= 5:
            roc_data = {'name': 'ROC', 'value': roc, 'status': '🟢 قوي', 'score': 1.2, 'max': 1.5, 'interpretation': 'زخم جيد'}
        elif roc >= 0:
            roc_data = {'name': 'ROC', 'value': roc, 'status': '🟢 إيجابي', 'score': 0.6, 'max': 1.5, 'interpretation': 'نمو خفيف'}
        elif roc >= -5:
            roc_data = {'name': 'ROC', 'value': roc, 'status': '🔴 سلبي', 'score': 0.3, 'max': 1.5, 'interpretation': 'تراجع خفيف'}
        else:
            roc_data = {'name': 'ROC', 'value': roc, 'status': '🔴 سلبي قوي', 'score': 0.0, 'max': 1.5, 'interpretation': 'هبوط قوي'}
        signals_analysis.append(roc_data)
    
    # 6. Volume
    vol_diff = volume_analysis['difference_pct']
    if vol_diff > 50:
        vol_data = {'name': 'Volume', 'value': vol_diff, 'status': '🟢 قوي جداً', 'score': 1.0, 'max': 1.0, 'interpretation': 'اهتمام كبير'}
    elif vol_diff > 20:
        vol_data = {'name': 'Volume', 'value': vol_diff, 'status': '🟢 قوي', 'score': 0.8, 'max': 1.0, 'interpretation': 'نشاط جيد'}
    elif vol_diff > -20:
        vol_data = {'name': 'Volume', 'value': vol_diff, 'status': '⚪ متوسط', 'score': 0.5, 'max': 1.0, 'interpretation': 'حجم عادي'}
    elif vol_diff > -50:
        vol_data = {'name': 'Volume', 'value': vol_diff, 'status': '🟡 ضعيف', 'score': 0.2, 'max': 1.0, 'interpretation': 'قلة اهتمام'}
    else:
        vol_data = {'name': 'Volume', 'value': vol_diff, 'status': '🔴 ضعيف جداً', 'score': 0.0, 'max': 1.0, 'interpretation': 'حجم منخفض جداً'}
    signals_analysis.append(vol_data)
    
    # 7. Trend Alignment
    price = latest['Close']
    sma_20 = latest['SMA_20']
    sma_50 = latest['SMA_50']
    if pd.notna(sma_20) and pd.notna(sma_50):
        if price > sma_20 > sma_50:
            trend_data = {'name': 'Trend Alignment', 'value': price, 'status': '🟢 ترتيب صاعد', 'score': 2.0, 'max': 2.0, 'interpretation': 'اتجاه قوي'}
        elif price > sma_20:
            trend_data = {'name': 'Trend Alignment', 'value': price, 'status': '🟢 جزئي صاعد', 'score': 1.2, 'max': 2.0, 'interpretation': 'اتجاه معتدل'}
        elif price > sma_50:
            trend_data = {'name': 'Trend Alignment', 'value': price, 'status': '🟡 ضعيف', 'score': 0.6, 'max': 2.0, 'interpretation': 'اتجاه خفيف'}
        else:
            trend_data = {'name': 'Trend Alignment', 'value': price, 'status': '🔴 هابط', 'score': 0.0, 'max': 2.0, 'interpretation': 'اتجاه سلبي'}
        signals_analysis.append(trend_data)
    
    # 8. Stochastic
    stoch = latest['Stoch_K']
    if pd.notna(stoch):
        if stoch < 20:
            stoch_data = {'name': 'Stochastic', 'value': stoch, 'status': '🟢 فرصة شراء', 'score': 1.0, 'max': 1.0, 'interpretation': 'تشبع بيعي'}
        elif stoch < 50:
            stoch_data = {'name': 'Stochastic', 'value': stoch, 'status': '🟢 إيجابي', 'score': 0.7, 'max': 1.0, 'interpretation': 'منطقة سفلية'}
        elif stoch < 80:
            stoch_data = {'name': 'Stochastic', 'value': stoch, 'status': '⚪ محايد', 'score': 0.5, 'max': 1.0, 'interpretation': 'منطقة وسطى'}
        else:
            stoch_data = {'name': 'Stochastic', 'value': stoch, 'status': '🔴 تشبع شرائي', 'score': 0.0, 'max': 1.0, 'interpretation': 'حذر'}
        signals_analysis.append(stoch_data)
    
    # 9. OBV
    if len(indicators) >= 5:
        if latest['OBV'] > indicators['OBV'].iloc[-5]:
            obv_data = {'name': 'OBV', 'value': latest['OBV'], 'status': '🟢 صاعد', 'score': 1.0, 'max': 1.0, 'interpretation': 'تدفق نقدي إيجابي'}
        else:
            obv_data = {'name': 'OBV', 'value': latest['OBV'], 'status': '🔴 هابط', 'score': 0.0, 'max': 1.0, 'interpretation': 'تدفق نقدي سلبي'}
        signals_analysis.append(obv_data)
    
    # 10. Candlestick
    pattern = detect_candlestick_pattern(indicators)
    if pattern == "صاعد":
        candle_data = {'name': 'Candlestick', 'value': 0, 'status': '🟢 صاعد', 'score': 1.0, 'max': 1.0, 'interpretation': 'نموذج إيجابي'}
    elif pattern == "هابط":
        candle_data = {'name': 'Candlestick', 'value': 0, 'status': '🔴 هابط', 'score': 0.0, 'max': 1.0, 'interpretation': 'نموذج سلبي'}
    else:
        candle_data = {'name': 'Candlestick', 'value': 0, 'status': '⚪ محايد', 'score': 0.5, 'max': 1.0, 'interpretation': 'لا إشارة واضحة'}
    signals_analysis.append(candle_data)
    
    # الخلاصة
    total_score = sum(s['score'] for s in signals_analysis)
    total_max = sum(s['max'] for s in signals_analysis)
    
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
        'positive_count': positive_count,
        'neutral_count': neutral_count,
        'negative_count': negative_count,
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
    """
    المرحلة 2: تحليل Claude AI المتقدم والشامل
    """
    
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        latest = indicators.iloc[-1]
        
        # بناء تفاصيل المؤشرات
        signals_detail = "\n".join([
            f"- {s['name']}: {s['value']:.2f if isinstance(s['value'], (int, float)) and s['value'] != 0 else s['value']} | {s['status']} | الوزن: {s['score']:.1f}/{s['max']:.1f} | {s['interpretation']}"
            for s in technical_signals['signals']
        ])
        
        # معلومات الدعم والمقاومة
        r1 = support_resistance.get('resistance_1')
        r2 = support_resistance.get('resistance_2')
        s1 = support_resistance.get('support_1')
        s2 = support_resistance.get('support_2')
        
        dist_r1 = ((r1 - price) / price * 100) if r1 else None
        dist_s1 = ((price - s1) / price * 100) if s1 else None
        
        prompt = f"""أنت محلل مالي خبير. قدم تحليلاً شاملاً ومفصلاً.

السهم: {symbol}
السعر: ${price:.2f}
التغير: {change_pct:+.2f}%

المؤشرات الفنية:
{signals_detail}

النتيجة: {technical_signals['total_score']:.1f}/{technical_signals['total_max']:.1f} ({technical_signals['percentage']:.1f}%)

الدعم/المقاومة:
- المقاومة 1: ${r1:.2f if r1 else 'N/A'} ({dist_r1:+.1f}% أعلى)
- الدعم 1: ${s1:.2f if s1 else 'N/A'} ({dist_s1:-.1f}% أسفل)

الزخم: {momentum['score']}/10 ({momentum['strength']})
R/R: {risk_reward['ratio']:.2f if risk_reward else 'N/A'}
Grade: {opportunity_quality['grade']}
حجم التداول: {volume_analysis['difference_pct']:+.1f}%

قدم تحليلاً بصيغة JSON:

{{
  "detailed_indicators": [
    {{
      "name": "اسم المؤشر",
      "status": "الحالة",
      "interpretation": "التفسير",
      "weight": "الوزن",
      "impact": "التأثير"
    }}
  ],
  "support_resistance_analysis": "تحليل نصي للدعم والمقاومة",
  "momentum_analysis": "تحليل نصي للزخم",
  "risk_reward_analysis": "تحليل نصي للمخاطرة/العائد",
  "opportunity_quality_analysis": "تحليل نصي لجودة الفرصة",
  "volume_and_liquidity": "تحليل نصي للحجم والسيولة",
  "final_recommendation": {{
    "decision": "ادخل الآن أو راقب أو لا تدخل",
    "comprehensive_analysis": "تحليل شامل 3-5 جمل",
    "reasons": ["سبب 1", "سبب 2", "سبب 3"],
    "alternatives": "البدائل المقترحة",
    "conditions": "شروط الدخول إن وجدت",
    "confidence": 75,
    "success_probability": 65
  }}
}}

قدم JSON فقط بدون نص إضافي."""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        # تنظيف JSON
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        claude_analysis = json.loads(response_text)
        
        return claude_analysis
        
    except Exception as e:
        print(f"Claude API error: {str(e)}")
        # Return default structure
        return {
            "detailed_indicators": [
                {
                    "name": "خطأ في التحليل",
                    "status": "غير متوفر",
                    "interpretation": f"حدث خطأ: {str(e)}",
                    "weight": "N/A",
                    "impact": "N/A"
                }
            ],
            "support_resistance_analysis": "غير متوفر حالياً",
            "momentum_analysis": "غير متوفر حالياً",
            "risk_reward_analysis": "غير متوفر حالياً",
            "opportunity_quality_analysis": "غير متوفر حالياً",
            "volume_and_liquidity": "غير متوفر حالياً",
            "final_recommendation": {
                "decision": "خطأ في التحليل",
                "comprehensive_analysis": "حدث خطأ أثناء معالجة تحليل Claude AI. البيانات الفنية متوفرة في المرحلة 1.",
                "reasons": ["خطأ فني في الاتصال بـ Claude API"],
                "alternatives": "استخدم البيانات من المرحلة 1 للتقييم",
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
    if latest['ROC'] > 10:
        momentum += 3
    elif latest['ROC'] > 5:
        momentum += 2
    elif latest['ROC'] > 0:
        momentum += 1
    if abs(latest['MACD']) > 3:
        momentum += 3
    elif abs(latest['MACD']) > 2:
        momentum += 2
    elif abs(latest['MACD']) > 1:
        momentum += 1
    if latest['RSI'] > prev['RSI'] and latest['RSI'] > 50:
        momentum += 2
    elif latest['RSI'] > prev['RSI']:
        momentum += 1
    avg_volume = indicators['Volume'].tail(20).mean()
    if latest['Volume'] > avg_volume * 2:
        momentum += 2
    elif latest['Volume'] > avg_volume * 1.5:
        momentum += 1
    return {
        'score': momentum,
        'strength': 'قوي جداً' if momentum >= 8 else 'قوي' if momentum >= 6 else 'متوسط' if momentum >= 4 else 'ضعيف'
    }

def detect_late_entry(indicators):
    latest = indicators.iloc[-1]
    warnings = []
    risk_level = 'low'
    rsi = latest['RSI']
    stoch_k = latest['Stoch_K']
    if rsi > 75 and pd.notna(stoch_k) and stoch_k > 85:
        warnings.append('⚠️ تشبع شرائي حاد - احتمال نهاية الموجة!')
        risk_level = 'critical'
    elif rsi > 70 and pd.notna(stoch_k) and stoch_k > 80:
        warnings.append('⚠️ تشبع شرائي - قد تكون متأخر!')
        risk_level = 'high'
    elif rsi > 65 or (pd.notna(stoch_k) and stoch_k > 75):
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
            risk_analysis['warnings'].append('⚠️ قريب جداً من المقاومة - مساحة محدودة!')
            risk_analysis['position_quality'] = 'poor'
        elif dist_res < 5:
            risk_analysis['warnings'].append('⚠️ المقاومة قريبة - حذر!')
            risk_analysis['position_quality'] = 'fair'
    else:
        risk_analysis['opportunities'].append('✅ لا توجد مقاومة واضحة - مجال مفتوح')
    if support:
        dist_sup = (price - support) / price * 100
        risk_analysis['distance_to_support'] = round(dist_sup, 2)
        if dist_sup < 3:
            risk_analysis['opportunities'].append('✅ قريب من الدعم - منطقة شراء جيدة!')
            if risk_analysis['position_quality'] == 'neutral':
                risk_analysis['position_quality'] = 'good'
        elif dist_sup < 5:
            risk_analysis['opportunities'].append('✅ قرب الدعم - فرصة مناسبة')
            if risk_analysis['position_quality'] == 'neutral':
                risk_analysis['position_quality'] = 'fair'
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
        factors.append('مكان ممتاز')
    elif position_risk['position_quality'] == 'fair':
        quality_score += 1
    elif position_risk['position_quality'] == 'poor':
        quality_score += 0
        factors.append('مكان سيء')
    
    if not late_entry['is_late']:
        quality_score += 2
        factors.append('توقيت جيد')
    else:
        quality_score += 0
        factors.append('دخول متأخر')
    
    if momentum['score'] >= 8:
        quality_score += 2
        factors.append('زخم قوي')
    elif momentum['score'] >= 6:
        quality_score += 1
    elif momentum['score'] >= 4:
        quality_score += 0.5
    
    latest = indicators.iloc[-1]
    adx = latest['ADX']
    if adx > 40:
        quality_score += 1
        factors.append('اتجاه قوي جداً')
    elif adx > 25:
        quality_score += 0.7
        factors.append('اتجاه واضح')
    elif adx > 20:
        quality_score += 0.3
    
    if risk_reward:
        rr = risk_reward['ratio']
        if rr >= 3.0:
            quality_score += 2
            factors.append('R/R ممتاز')
        elif rr >= 2.5:
            quality_score += 1.7
            factors.append('R/R ممتاز')
        elif rr >= 2.0:
            quality_score += 1.4
            factors.append('R/R جيد جداً')
        elif rr >= 1.5:
            quality_score += 1.0
            factors.append('R/R جيد')
        elif rr >= 1.0:
            quality_score += 0.5
            factors.append('R/R مقبول')
        elif rr >= 0.7:
            quality_score += 0.2
    
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
    latest = indicators.iloc[-1]
    context = {}
    if change_pct < -1:
        healthy_correction = True
        if len(indicators) >= 5:
            if latest['OBV'] < indicators['OBV'].iloc[-5]:
                healthy_correction = False
        if support_resistance.get('support_1'):
            if price < support_resistance['support_1']:
                healthy_correction = False
        if latest['RSI'] > 70:
            healthy_correction = False
        if healthy_correction:
            context['type'] = 'تصحيح صحي'
            context['message'] = '✅ النزول اليومي طبيعي - الاتجاه العام لسه قوي'
        else:
            context['type'] = 'تحذير'
            context['message'] = '⚠️ النزول قد يكون بداية تغير في الاتجاه'
    elif change_pct > 1:
        sustainable = True
        if latest['RSI'] > 75:
            sustainable = False
        if pd.notna(latest['Stoch_K']) and latest['Stoch_K'] > 85:
            sustainable = False
        if sustainable:
            context['type'] = 'زخم قوي'
            context['message'] = '✅ الصعود مدعوم بمؤشرات قوية'
        else:
            context['type'] = 'حذر'
            context['message'] = '⚠️ الصعود قد يكون مبالغ فيه'
    else:
        context['type'] = 'طبيعي'
        context['message'] = 'حركة سعرية عادية'
    return context

def calculate_confirmation_signals(indicators):
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2] if len(indicators) > 1 else latest
    signals = {}
    score = 0
    if len(indicators) >= 5:
        if latest['OBV'] > indicators['OBV'].iloc[-5]:
            signals['obv'] = 'صاعد'
            score += 1
        else:
            signals['obv'] = 'هابط'
            score += 0
    else:
        signals['obv'] = 'غير محدد'
    stoch = latest['Stoch_K']
    if pd.notna(stoch):
        if stoch < 20:
            signals['stochastic'] = 'تشبع بيعي (فرصة)'
            score += 1
        elif stoch > 80:
            signals['stochastic'] = 'تشبع شرائي (حذر)'
            score += 0
        else:
            signals['stochastic'] = 'محايد'
            score += 0.5
    else:
        signals['stochastic'] = 'غير محدد'
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
    macd_hist = latest['MACD'] - latest['Signal']
    prev_hist = prev['MACD'] - prev['Signal'] if len(indicators) > 1 else 0
    if macd_hist > 0 and macd_hist > prev_hist:
        signals['macd_histogram'] = 'تزايد إيجابي'
        score += 1
    elif macd_hist > 0:
        signals['macd_histogram'] = 'إيجابي'
        score += 0.5
    else:
        signals['macd_histogram'] = 'سلبي'
    if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_50']):
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals['ma_alignment'] = 'ترتيب صاعد'
            score += 1
        elif latest['Close'] > latest['SMA_20']:
            signals['ma_alignment'] = 'جزئي صاعد'
            score += 0.5
        else:
            signals['ma_alignment'] = 'هابط'
    macd = latest['MACD']
    rsi = latest['RSI']
    roc = latest['ROC']
    is_bearish = macd < 0 and rsi < 40 and roc < 0
    if is_bearish:
        score = min(score, 1.5)
        verdict = 'لا تدخل - اتجاه هابط!'
    else:
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
        return {
            'action': 'لا تدخل',
            'reason': 'نسبة المخاطرة/العائد ضعيفة جداً',
            'confidence': 'منخفضة'
        }
    if position_risk['position_quality'] == 'poor':
        return {
            'action': 'راقب',
            'reason': 'قريب جداً من المقاومة - مساحة محدودة',
            'confidence': 'متوسطة'
        }
    if late_entry['risk_level'] == 'critical':
        return {
            'action': 'لا تدخل',
            'reason': 'تشبع شرائي حاد - احتمال نهاية الموجة',
            'confidence': 'منخفضة'
        }
    if score >= 10 and confirmation['positive_count'] >= 4.5:
        return {
            'action': 'ادخل الآن',
            'reason': 'مؤشرات قوية وتأكيد ممتاز',
            'confidence': 'عالية'
        }
    if score >= 8 and confirmation['positive_count'] >= 3.5:
        return {
            'action': 'راقب',
            'reason': 'مؤشرات جيدة لكن تحتاج تأكيد أقوى',
            'confidence': 'متوسطة'
        }
    if score >= 6:
        return {
            'action': 'راقب',
            'reason': 'مؤشرات متوسطة - انتظر تحسن',
            'confidence': 'متوسطة'
        }
    return {
        'action': 'لا تدخل',
        'reason': 'مؤشرات ضعيفة',
        'confidence': 'منخفضة'
    }

def score_rsi(rsi):
    if pd.isna(rsi):
        return 0
    if 40 <= rsi <= 60:
        return 2.0
    elif 35 <= rsi < 40:
        return 1.5 + (rsi - 35) * 0.1
    elif 60 < rsi <= 65:
        return 2.0 - (rsi - 60) * 0.1
    elif 30 <= rsi < 35:
        return 1.0 + (rsi - 30) * 0.1
    elif 65 < rsi <= 70:
        return 1.5 - (rsi - 65) * 0.1
    elif 25 <= rsi < 30:
        return 0.5 + (rsi - 25) * 0.1
    elif 70 < rsi <= 75:
        return 1.0 - (rsi - 70) * 0.1
    else:
        return 0

def score_macd(macd, signal):
    if pd.isna(macd) or pd.isna(signal):
        return 0
    diff = macd - signal
    if macd > 0 and diff > 0:
        strength = min(abs(macd), 5) / 5
        return 1.5 + (strength * 0.5)
    elif macd > 0 and diff <= 0:
        return 1.0
    elif macd <= 0 and diff > 0:
        strength = min(abs(diff), 2) / 2
        return 0.5 + (strength * 0.5)
    else:
        return 0

def score_adx(adx):
    if pd.isna(adx):
        return 0
    if adx >= 50:
        return 2.0
    elif adx >= 40:
        return 1.7 + (adx - 40) * 0.03
    elif adx >= 30:
        return 1.4 + (adx - 30) * 0.03
    elif adx >= 25:
        return 1.0 + (adx - 25) * 0.08
    elif adx >= 20:
        return 0.5 + (adx - 20) * 0.1
    elif adx >= 15:
        return 0.2 + (adx - 15) * 0.06
    else:
        return 0

def score_roc(roc):
    if pd.isna(roc):
        return 0
    if roc >= 15:
        return 1.5
    elif roc >= 10:
        return 1.2 + (roc - 10) * 0.06
    elif roc >= 5:
        return 0.8 + (roc - 5) * 0.08
    elif roc >= 0:
        return 0.3 + (roc) * 0.1
    elif roc >= -5:
        return 0.3 + (roc + 5) * 0.06
    else:
        return 0

def score_mfi(mfi):
    if pd.isna(mfi):
        return 0
    if 40 <= mfi <= 60:
        return 1.0
    elif 30 <= mfi < 40:
        return 0.5 + (mfi - 30) * 0.05
    elif 60 < mfi <= 70:
        return 1.0 - (mfi - 60) * 0.05
    elif 20 <= mfi < 30:
        return 0.2 + (mfi - 20) * 0.03
    elif 70 < mfi <= 80:
        return 0.5 - (mfi - 70) * 0.03
    else:
        return 0

def score_volume(current_volume, avg_volume):
    if avg_volume == 0:
        return 0
    ratio = current_volume / avg_volume
    if ratio >= 2.0:
        return 1.0
    elif ratio >= 1.5:
        return 0.8 + (ratio - 1.5) * 0.4
    elif ratio >= 1.2:
        return 0.6 + (ratio - 1.2) * 0.67
    elif ratio >= 1.0:
        return 0.4 + (ratio - 1.0) * 1.0
    elif ratio >= 0.8:
        return 0.2 + (ratio - 0.8) * 1.0
    elif ratio >= 0.5:
        return (ratio - 0.5) * 0.67
    else:
        return 0

def score_trend_alignment(price, sma_20, sma_50):
    if pd.isna(sma_20) or pd.isna(sma_50):
        return 0
    if price > sma_20 and price > sma_50 and sma_20 > sma_50:
        distance_20 = (price - sma_20) / sma_20
        if distance_20 > 0.1:
            return 1.5
        else:
            return 2.0
    elif price > sma_20 and sma_20 > sma_50:
        return 1.5
    elif price > sma_20:
        return 1.0
    elif price > sma_50:
        return 0.5
    else:
        return 0

def calculate_score(indicators, info):
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2] if len(indicators) > 1 else latest
    score = 0
    score += score_rsi(latest['RSI'])
    score += score_macd(latest['MACD'], latest['Signal'])
    score += score_trend_alignment(latest['Close'], latest['SMA_20'], latest['SMA_50'])
    score += score_adx(latest['ADX'])
    score += score_roc(latest['ROC'])
    score += score_mfi(latest['MFI'])
    current_volume = latest['Volume']
    avg_volume = indicators['Volume'].tail(20).mean()
    score += score_volume(current_volume, avg_volume)
    price = latest['Close']
    if pd.notna(latest['BB_Lower']) and price <= latest['BB_Lower']:
        score += 0.5
    atr = latest['ATR']
    if atr < price * 0.05:
        score += 0.5
    if len(indicators) > 1:
        gap = abs(latest['Open'] - prev['Close']) / prev['Close']
        if gap < 0.02:
            score += 0.5
    return max(0, round(score, 1))

def calculate_targets_advanced(price, confirmation_score, atr, support_resistance):
    if confirmation_score >= 4:
        target_short = support_resistance.get('resistance_1') or round(price + (atr * 2), 2)
        target_medium = support_resistance.get('resistance_2') or round(price + (atr * 3), 2)
        stop_loss = support_resistance.get('support_1') or round(price - (atr * 1.5), 2)
        return {
            'entry': round(price, 2),
            'target_short': round(target_short, 2),
            'target_medium': round(target_medium, 2),
            'stop_loss': round(stop_loss, 2),
            'direction': 'صاعد'
        }
    elif confirmation_score >= 2:
        target_short = round(price + (atr * 1.5), 2)
        target_medium = round(price + (atr * 2.5), 2)
        stop_loss = round(price - (atr * 1.2), 2)
        return {
            'entry': round(price, 2),
            'target_short': target_short,
            'target_medium': target_medium,
            'stop_loss': stop_loss,
            'direction': 'محايد'
        }
    else:
        target_short = support_resistance.get('resistance_1') or round(price + (atr * 1), 2)
        target_medium = support_resistance.get('resistance_2') or round(price + (atr * 2), 2)
        stop_loss = support_resistance.get('support_1') or round(price - (atr * 1.5), 2)
        return {
            'entry': round(price, 2),
            'target_short': target_short,
            'target_medium': target_medium,
            'stop_loss': stop_loss,
            'direction': 'هابط'
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
        "name": "OptiMax Stock Analysis API",
        "version": "8.0.0",
        "description": "Professional Edition - تحليل احترافي من مرحلتين",
        "features": [
            "المرحلة 1: تقييم الإشارات الفنية البسيط",
            "المرحلة 2: تحليل Claude AI المتقدم والشامل"
        ],
        "endpoints": {
            "/top-opportunities": "Get top stock opportunities",
            "/analysis/{symbol}": "Get detailed two-stage analysis"
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
    cache_key = f"analysis_{symbol}"
    if cache_key in cache:
        return cache[cache_key]
    data = get_stock_data_yf(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")
    indicators = calculate_indicators(data['history'])
    score = calculate_score(indicators, data['info'])
    confirmation = calculate_confirmation_signals(indicators)
    support_resistance = calculate_support_resistance_advanced(data['history'], data['price'])
    latest = indicators.iloc[-1]
    atr = latest['ATR']
    targets = calculate_targets_advanced(data['price'], confirmation['positive_count'], atr, support_resistance)
    position_risk = check_position_risk(
        data['price'], 
        support_resistance.get('resistance_1'), 
        support_resistance.get('support_1')
    )
    late_entry = detect_late_entry(indicators)
    risk_reward = calculate_risk_reward(
        data['price'], 
        targets['target_short'], 
        targets['stop_loss']
    )
    momentum = calculate_momentum_score(indicators)
    recent_performance = analyze_recent_performance(indicators)
    opportunity_quality = calculate_opportunity_quality(
        score, 
        indicators, 
        support_resistance, 
        position_risk, 
        late_entry, 
        momentum,
        risk_reward
    )
    prev_close = data['history']['Close'].iloc[-2] if len(data['history']) > 1 else latest['Close']
    change = latest['Close'] - prev_close
    change_pct = (change / prev_close) * 100
    daily_context = analyze_daily_change_context(
        change_pct,
        indicators,
        support_resistance,
        data['price']
    )
    final_recommendation = get_final_recommendation(
        score,
        confirmation,
        position_risk,
        late_entry,
        risk_reward
    )
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
    
    volume_analysis = {
        'current': current_volume,
        'average': avg_volume,
        'difference': volume_diff,
        'difference_pct': round(volume_diff_pct, 2),
        'trend': volume_trend,
        'is_unusual': is_unusual_volume
    }
    
    # المرحلة 1: تحليل الإشارات الفنية
    technical_signals = analyze_technical_signals(indicators, volume_analysis)
    
    # المرحلة 2: تحليل Claude المتقدم
    claude_deep_analysis = get_claude_deep_analysis(
        symbol=symbol,
        price=data['price'],
        change_pct=change_pct,
        technical_signals=technical_signals,
        support_resistance=support_resistance,
        momentum=momentum,
        risk_reward=risk_reward,
        opportunity_quality=opportunity_quality,
        position_risk=position_risk,
        late_entry=late_entry,
        volume_analysis=volume_analysis,
        indicators=indicators,
        recent_performance=recent_performance
    )
    
    analysis_data = {
        "symbol": symbol,
        "name": data['name'],
        "price": data['price'],
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "score": score,
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
        "volume_analysis": volume_analysis,
        "market_context": {
            "daily_trend": daily_trend,
            "sector_strength": "متوسط"
        }
    }
    
    cache[cache_key] = analysis_data
    return analysis_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

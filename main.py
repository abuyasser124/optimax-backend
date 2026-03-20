from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Optional
import anthropic
import os
from functools import lru_cache
import logging
import json

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_time_ago_arabic(timestamp):
    """حساب الوقت بالعربي"""
    try:
        if timestamp == 0:
            return "غير محدد"
        
        news_date = datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern'))
        now = datetime.now(pytz.timezone('US/Eastern'))
        diff = now - news_date
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "منذ لحظات"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"منذ {mins} دقيقة"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"منذ {hours} ساعة"
        else:
            days = int(seconds / 86400)
            return f"منذ {days} يوم"
    except:
        return "غير محدد"

app = FastAPI(title="OptiMax API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# قائمة الأسهم الشرعية (400 سهم)
SHARIAH_STOCKS = [
    ("AAPL", "Apple Inc"), ("MSFT", "Microsoft Corporation"), ("GOOGL", "Alphabet Inc Class A"),
    ("GOOG", "Alphabet Inc Class C"), ("NVDA", "NVIDIA Corporation"), ("META", "Meta Platforms Inc"),
    ("TSLA", "Tesla Inc"), ("AMZN", "Amazon.com Inc"), ("AMD", "Advanced Micro Devices Inc"),
    ("AVGO", "Broadcom Inc"), ("INTC", "Intel Corporation"), ("QCOM", "QUALCOMM Inc"),
    ("TXN", "Texas Instruments Inc"), ("AMAT", "Applied Materials Inc"), ("LRCX", "Lam Research Corporation"),
    ("KLAC", "KLA Corporation"), ("NXPI", "NXP Semiconductors NV"), ("MU", "Micron Technology Inc"),
    ("MRVL", "Marvell Technology Inc"), ("SNPS", "Synopsys Inc"), ("CDNS", "Cadence Design Systems Inc"),
    ("MCHP", "Microchip Technology Inc"), ("ON", "ON Semiconductor Corporation"), ("SWKS", "Skyworks Solutions Inc"),
    ("QRVO", "Qorvo Inc"), ("MPWR", "Monolithic Power Systems Inc"), ("ADI", "Analog Devices Inc"),
    ("ASML", "ASML Holding NV"), ("ADBE", "Adobe Inc"), ("CRM", "Salesforce Inc"),
    ("ORCL", "Oracle Corporation"), ("NOW", "ServiceNow Inc"), ("WDAY", "Workday Inc"),
    ("DDOG", "Datadog Inc"), ("SNOW", "Snowflake Inc"), ("MDB", "MongoDB Inc"),
    ("TEAM", "Atlassian Corporation"), ("ZM", "Zoom Video Communications"), ("DOCU", "DocuSign Inc"),
    ("PLTR", "Palantir Technologies Inc"), ("PANW", "Palo Alto Networks Inc"), ("CRWD", "CrowdStrike Holdings Inc"),
    ("FTNT", "Fortinet Inc"), ("ZS", "Zscaler Inc"), ("OKTA", "Okta Inc"),
    ("NET", "Cloudflare Inc"), ("CSCO", "Cisco Systems Inc"), ("SHOP", "Shopify Inc"),
    ("EBAY", "eBay Inc"), ("ETSY", "Etsy Inc"), ("DASH", "DoorDash Inc"),
    ("UBER", "Uber Technologies Inc"), ("LYFT", "Lyft Inc"), ("ABNB", "Airbnb Inc"),
    ("PYPL", "PayPal Holdings Inc"), ("SQ", "Block Inc"), ("COIN", "Coinbase Global Inc"),
    ("AFRM", "Affirm Holdings Inc"), ("AMGN", "Amgen Inc"), ("GILD", "Gilead Sciences Inc"),
    ("REGN", "Regeneron Pharmaceuticals"), ("VRTX", "Vertex Pharmaceuticals Inc"), ("BIIB", "Biogen Inc"),
    ("ILMN", "Illumina Inc"), ("ISRG", "Intuitive Surgical Inc"), ("DXCM", "DexCom Inc"),
    ("ALGN", "Align Technology Inc"), ("ABT", "Abbott Laboratories"), ("TMO", "Thermo Fisher Scientific Inc"),
    ("DHR", "Danaher Corporation"), ("SYK", "Stryker Corporation"), ("EW", "Edwards Lifesciences Corporation"),
    ("BSX", "Boston Scientific Corporation"), ("COST", "Costco Wholesale Corporation"), ("NKE", "NIKE Inc"),
    ("SBUX", "Starbucks Corporation"), ("MCD", "McDonald's Corporation"), ("DIS", "The Walt Disney Company"),
    ("LULU", "Lululemon Athletica Inc"), ("NFLX", "Netflix Inc"), ("FSLR", "First Solar Inc"),
    ("ENPH", "Enphase Energy Inc"), ("SEDG", "SolarEdge Technologies Inc"), ("RUN", "Sunrun Inc"),
    ("GNRC", "Generac Holdings Inc"), ("INCY", "Incyte Corporation"), ("ALNY", "Alnylam Pharmaceuticals Inc"),
    ("MRNA", "Moderna Inc"), ("BNTX", "BioNTech SE"), ("IDXX", "IDEXX Laboratories Inc"),
    ("ZBH", "Zimmer Biomet Holdings Inc"), ("BDX", "Becton Dickinson and Company"), ("BAX", "Baxter International Inc"),
    ("RMD", "ResMed Inc"), ("WMT", "Walmart Inc"), ("TGT", "Target Corporation"),
    ("HD", "The Home Depot Inc"), ("LOW", "Lowe's Companies Inc"), ("TJX", "The TJX Companies Inc"),
    ("ROST", "Ross Stores Inc"), ("DG", "Dollar General Corporation"), ("DLTR", "Dollar Tree Inc"),
    ("ULTA", "Ulta Beauty Inc"), ("YUM", "Yum! Brands Inc"), ("CMG", "Chipotle Mexican Grill Inc"),
    ("SPOT", "Spotify Technology SA"), ("ROKU", "Roku Inc"), ("PINS", "Pinterest Inc"),
    ("SNAP", "Snap Inc"), ("MTCH", "Match Group Inc"), ("RBLX", "Roblox Corporation"),
    ("U", "Unity Software Inc"), ("TTWO", "Take-Two Interactive Software"), ("EA", "Electronic Arts Inc"),
    ("RIVN", "Rivian Automotive Inc"), ("LCID", "Lucid Group Inc"), ("F", "Ford Motor Company"),
    ("GM", "General Motors Company"), ("PLUG", "Plug Power Inc"), ("CHPT", "ChargePoint Holdings Inc"),
    ("BLNK", "Blink Charging Co"), ("QS", "QuantumScape Corporation"), ("CAT", "Caterpillar Inc"),
    ("DE", "Deere & Company"), ("ETN", "Eaton Corporation"), ("EMR", "Emerson Electric Co"),
    ("LIN", "Linde plc"), ("APD", "Air Products and Chemicals Inc"), ("ECL", "Ecolab Inc"),
    ("DD", "DuPont de Nemours Inc"), ("BA", "Boeing Company"), ("GE", "General Electric Company"),
    ("HON", "Honeywell International Inc"), ("DELL", "Dell Technologies Inc"), ("HPQ", "HP Inc"),
    ("HPE", "Hewlett Packard Enterprise"), ("AI", "C3.ai Inc"), ("PATH", "UiPath Inc"),
    ("TTD", "The Trade Desk Inc"), ("Z", "Zillow Group Inc"), ("CMCSA", "Comcast Corporation"),
    ("T", "AT&T Inc"), ("VZ", "Verizon Communications Inc"), ("TMUS", "T-Mobile US Inc"),
    ("PEP", "PepsiCo Inc"), ("KO", "Coca-Cola Company"), ("MDLZ", "Mondelez International"),
    ("GIS", "General Mills Inc"), ("K", "Kellogg Company"), ("CPB", "Campbell Soup Company"),
]

# Cache
_cache = {}
_cache_time = {}
CACHE_DURATION = 600  # 10 minutes

def get_cache(key):
    if key in _cache:
        if datetime.now().timestamp() - _cache_time[key] < CACHE_DURATION:
            return _cache[key]
    return None

def set_cache(key, value):
    _cache[key] = value
    _cache_time[key] = datetime.now().timestamp()

def is_market_open():
    """فحص إذا السوق مفتوح"""
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    # تحقق من يوم الأسبوع (الاثنين=0، الأحد=6)
    if now_et.weekday() >= 5:  # السبت أو الأحد
        return False
    
    # تحقق من ساعات التداول
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_et <= market_close

@app.get("/")
def root():
    return {
        "app": "OptiMax API",
        "version": "2.0.0",
        "endpoints": {
            "top_opportunities": "/top-opportunities",
            "stock_analysis": "/analysis/{symbol}",
            "market_status": "/market-status"
        },
        "total_stocks": len(SHARIAH_STOCKS),
        "market_open": is_market_open(),
        "message": "Advanced Shariah-compliant stock analysis powered by Claude AI"
    }

def calculate_indicators(df):
    """حساب المؤشرات الفنية"""
    try:
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
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['SMA_20'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['SMA_20'] - (df['BB_std'] * 2)
        
        # SMA 50
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

def calculate_score(row, current_price):
    """حساب نقاط السهم من 0-10"""
    score = 0.0
    
    try:
        # RSI - Oversold (0-3 points)
        if row['RSI'] < 30:
            score += 3.0
        elif row['RSI'] < 40:
            score += 1.5
        elif row['RSI'] < 50:
            score += 0.5
            
        # MACD Bullish (0-3 points)
        if row['MACD'] > row['MACD_Signal']:
            if row['MACD_Hist'] > 0:
                score += 3.0
            else:
                score += 1.5
                
        # Near Bollinger Lower (0-2 points)
        if current_price < row['BB_lower'] * 1.02:
            score += 2.0
        elif current_price < row['BB_lower'] * 1.05:
            score += 1.0
            
        # Price above SMA-20 (0-1 point)
        if current_price > row['SMA_20']:
            score += 1.0
            
        # Volume surge (0-1 point)
        if row['Volume'] > row['Volume_SMA'] * 5:
            score += 1.0
        elif row['Volume'] > row['Volume_SMA'] * 3:
            score += 0.5
            
    except:
        pass
        
    return min(score, 10.0)  # Cap at 10

def get_signal(score):
    """تحديد الإشارة"""
    if score >= 8:
        return "Strong Buy"
    elif score >= 6:
        return "Buy"
    elif score >= 4:
        return "Hold"
    elif score >= 2:
        return "Sell"
    else:
        return "Strong Sell"

@app.get("/top-opportunities")
def get_top_opportunities():
    """الحصول على أفضل 20 فرصة بعد تحليل Claude AI"""
    
    cache_key = "top_opportunities"
    cached = get_cache(cache_key)
    if cached:
        return cached
    
    logger.info(f"Starting analysis of {len(SHARIAH_STOCKS)} stocks...")
    
    # المرحلة 1: التحليل الفني لكل الأسهم
    all_scores = []
    symbols = [s[0] for s in SHARIAH_STOCKS]
    
    try:
        data = yf.download(symbols, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        
        for symbol, name in SHARIAH_STOCKS:
            try:
                if len(symbols) > 1:
                    stock_data = data[symbol].copy()
                else:
                    stock_data = data.copy()
                
                if stock_data.empty or len(stock_data) < 50:
                    continue
                
                stock_data = calculate_indicators(stock_data)
                latest = stock_data.iloc[-1]
                current_price = latest['Close']
                
                score = calculate_score(latest, current_price)
                
                all_scores.append({
                    "symbol": symbol,
                    "name": name,
                    "price": round(float(current_price), 2),
                    "change_pct": round(((current_price - stock_data.iloc[-2]['Close']) / stock_data.iloc[-2]['Close']) * 100, 2),
                    "score": round(score, 1),
                    "signal": get_signal(score),
                    "rsi": round(float(latest['RSI']), 1) if not pd.isna(latest['RSI']) else None,
                    "macd": round(float(latest['MACD']), 2) if not pd.isna(latest['MACD']) else None,
                })
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # ترتيب وأخذ أفضل 50
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        top_50 = all_scores[:50]
        
        logger.info(f"Top 50 stocks selected. Sending to Claude AI for final analysis...")
        
        # المرحلة 2: تحليل Claude للـ 50 الأفضل
        final_top_20 = []
        
        if claude_client:
            try:
                # إعداد البيانات لـ Claude
                stocks_summary = "\n".join([
                    f"{i+1}. {s['symbol']} ({s['name']}) - Score: {s['score']}/10, Price: ${s['price']}, RSI: {s['rsi']}, Signal: {s['signal']}"
                    for i, s in enumerate(top_50)
                ])
                
                prompt = f"""أنت محلل أسهم خبير متخصص في الأسهم الشرعية الأمريكية.

لديك قائمة بأفضل 50 سهم شرعي بناءً على التحليل الفني:

{stocks_summary}

مهمتك:
1. اختر أفضل 20 سهم من القائمة للتداول القصير والمتوسط المدى
2. رتبهم حسب قوة الفرصة (الأقوى أولاً)
3. أرجع فقط رموز الأسهم (Symbols) مفصولة بفاصلة

مثال للرد: AAPL,NVDA,MSFT,TSLA,...

الرد (فقط رموز الأسهم):"""

                message = claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                selected_symbols = message.content[0].text.strip().split(',')
                selected_symbols = [s.strip() for s in selected_symbols][:20]
                
                # ترتيب حسب اختيار Claude
                for symbol in selected_symbols:
                    stock = next((s for s in top_50 if s['symbol'] == symbol), None)
                    if stock:
                        final_top_20.append(stock)
                
                logger.info(f"Claude AI selected {len(final_top_20)} stocks")
                
            except Exception as e:
                logger.error(f"Claude AI error: {e}")
                final_top_20 = top_50[:20]
        else:
            logger.warning("Claude API not configured, using top 20 from technical analysis")
            final_top_20 = top_50[:20]
        
        result = {
            "updated_at": datetime.now(pytz.timezone('US/Eastern')).isoformat(),
            "market_open": is_market_open(),
            "total_analyzed": len(all_scores),
            "top_opportunities": final_top_20,
            "analysis_method": "Claude AI" if claude_client and final_top_20 else "Technical Only"
        }
        
        set_cache(cache_key, result)
        return result
        
    except Exception as e:
        logger.error(f"Error in get_top_opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{symbol}")
def get_detailed_analysis(symbol: str):
    """تحليل مفصل لسهم معين باستخدام Claude AI"""
    
    symbol = symbol.upper()
    cache_key = f"analysis_{symbol}"
    cached = get_cache(cache_key)
    if cached:
        return cached
    
    stock_info = next((s for s in SHARIAH_STOCKS if s[0] == symbol), None)
    if not stock_info:
        raise HTTPException(status_code=404, detail="Stock not in Shariah-compliant list")
    
    try:
        # جلب البيانات
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        score = calculate_score(latest, current_price)
        signal = get_signal(score)
        
        # البيانات الأساسية
        basic_data = {
            "symbol": symbol,
            "name": stock_info[1],
            "price": round(float(current_price), 2),
            "change_pct": round(((current_price - df.iloc[-2]['Close']) / df.iloc[-2]['Close']) * 100, 2),
            "score": round(score, 1),
            "signal": signal,
            "indicators": {
                "rsi": round(float(latest['RSI']), 2),
                "macd": round(float(latest['MACD']), 4),
                "macd_signal": round(float(latest['MACD_Signal']), 4),
                "sma_20": round(float(latest['SMA_20']), 2),
                "sma_50": round(float(latest['SMA_50']), 2),
                "bb_upper": round(float(latest['BB_upper']), 2),
                "bb_lower": round(float(latest['BB_lower']), 2),
            }
        }
        
        # تحليل Claude AI
        if claude_client:
            try:
                prompt = f"""أنت محلل أسهم خبير. قدم تحليلاً مفصلاً للسهم التالي:

الشركة: {stock_info[1]} ({symbol})
السعر الحالي: ${current_price:.2f}
التغير: {basic_data['change_pct']}%

المؤشرات الفنية:
- RSI: {basic_data['indicators']['rsi']}
- MACD: {basic_data['indicators']['macd']}
- SMA 20: ${basic_data['indicators']['sma_20']}
- SMA 50: ${basic_data['indicators']['sma_50']}
- Bollinger Upper: ${basic_data['indicators']['bb_upper']}
- Bollinger Lower: ${basic_data['indicators']['bb_lower']}

قدم تحليلاً يتضمن:
1. نقطة الدخول المثالية (رقم محدد)
2. الهدف الأول (قصير المدى 1-3 أيام) - رقم محدد
3. الهدف الثاني (متوسط المدى أسبوع-أسبوعين) - رقم محدد
4. وقف الخسارة (رقم محدد)
5. نسبة النجاح المتوقعة (%)
6. تاريخ صلاحية التحليل - التاريخ الحالي هو {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')}، أعط تاريخ صلاحية بعد 1-3 أيام بصيغة YYYY-MM-DD
7. تحليل مختصر (3-4 أسطر)

أرجع الرد بصيغة JSON:
{{
  "entry_point": 000.00,
  "target_short": 000.00,
  "target_medium": 000.00,
  "stop_loss": 000.00,
  "success_rate": 00,
  "valid_until": "YYYY-MM-DD",
  "analysis": "النص"
}}"""

                message = claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                claude_response = message.content[0].text.strip()
                # استخراج JSON من الرد
                if "```json" in claude_response:
                    claude_response = claude_response.split("```json")[1].split("```")[0].strip()
                elif "```" in claude_response:
                    claude_response = claude_response.split("```")[1].split("```")[0].strip()
                
                claude_analysis = json.loads(claude_response)
                basic_data["claude_analysis"] = claude_analysis
                
            except Exception as e:
                logger.error(f"Claude analysis error: {e}")
                basic_data["claude_analysis"] = None
        else:
            basic_data["claude_analysis"] = None
        
        basic_data["updated_at"] = datetime.now(pytz.timezone('US/Eastern')).isoformat()
        
        # جلب الأخبار مع تحليل Claude
        try:
            news_list = []
            ticker_news = stock.news
            if ticker_news and len(ticker_news) > 0 and claude_client:
                # أخذ أول 3 أخبار
                news_titles = [item.get("title", "") for item in ticker_news[:3] if item.get("title")]
                
                if news_titles:
                    # تحليل الأخبار بـ Claude
                    news_prompt = f"""حلل هذه الأخبار عن شركة {stock_info[1]} وصنفها (إيجابي/سلبي/محايد):

{chr(10).join([f"{i+1}. {title}" for i, title in enumerate(news_titles)])}

أرجع JSON فقط:
[
  {{"sentiment": "positive/negative/neutral", "emoji": "🟢/🔴/🟡"}},
  ...
]"""
                    
                    try:
                        sentiment_response = claude_client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=300,
                            messages=[{"role": "user", "content": news_prompt}]
                        )
                        
                        sentiment_text = sentiment_response.content[0].text.strip()
                        if "```json" in sentiment_text:
                            sentiment_text = sentiment_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in sentiment_text:
                            sentiment_text = sentiment_text.split("```")[1].split("```")[0].strip()
                        
                        sentiments = json.loads(sentiment_text)
                        
                        for i, item in enumerate(ticker_news[:3]):
                            if i < len(sentiments):
                                news_list.append({
                                    "title": item.get("title", ""),
                                    "publisher": item.get("publisher", ""),
                                    "link": item.get("link", ""),
                                    "sentiment": sentiments[i].get("sentiment", "neutral"),
                                    "emoji": sentiments[i].get("emoji", "🟡"),
                                    "time_ago": get_time_ago_arabic(item.get("providerPublishTime", 0))
                                })
                    except:
                        # إذا فشل تحليل Claude، نضيف بدون تحليل
                        for item in ticker_news[:3]:
                            news_list.append({
                                "title": item.get("title", ""),
                                "publisher": item.get("publisher", ""),
                                "link": item.get("link", ""),
                                "sentiment": "neutral",
                                "emoji": "🟡",
                                "time_ago": get_time_ago_arabic(item.get("providerPublishTime", 0))
                            })
            
            basic_data["news"] = news_list
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            basic_data["news"] = []
        
        set_cache(cache_key, basic_data)
        return basic_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-status")
def market_status():
    """حالة السوق والنظام"""
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    return {
        "market_open": is_market_open(),
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET (%A)"),
        "market_hours": "9:30 AM - 4:00 PM ET (Mon-Fri)",
        "total_stocks": len(SHARIAH_STOCKS),
        "cache_duration": f"{CACHE_DURATION} seconds",
        "claude_enabled": claude_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OptiMax API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قائمة الأسهم الشرعية (396 سهم)
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
    ("ANET", "Arista Networks Inc"), ("VEEV", "Veeva Systems Inc"), ("TWLO", "Twilio Inc"),
    ("AKAM", "Akamai Technologies Inc"), ("BILL", "BILL Holdings Inc"), ("PAYC", "Paycom Software Inc"),
    ("TOST", "Toast Inc"), ("SOFI", "SoFi Technologies Inc"), ("NU", "Nu Holdings Ltd"),
    ("FOUR", "Shift4 Payments Inc"), ("W", "Wayfair Inc"), ("CHWY", "Chewy Inc"),
    ("BKNG", "Booking Holdings Inc"), ("EXPE", "Expedia Group Inc"), ("NFLX", "Netflix Inc"),
    ("PARA", "Paramount Global"), ("WBD", "Warner Bros Discovery Inc"), ("NWSA", "News Corporation"),
    ("NYT", "The New York Times Company"), ("BE", "Bloom Energy Corporation"), ("NOVA", "Sunnova Energy International"),
    ("SGEN", "Seagen Inc"), ("NBIX", "Neurocrine Biosciences Inc"), ("BGNE", "BeiGene Ltd"),
    ("SRPT", "Sarepta Therapeutics Inc"), ("EXAS", "Exact Sciences Corporation"), ("FATE", "Fate Therapeutics Inc"),
    ("CRSP", "CRISPR Therapeutics AG"), ("NTLA", "Intellia Therapeutics Inc"), ("BEAM", "Beam Therapeutics Inc"),
    ("TECH", "Bio-Techne Corporation"), ("HOLX", "Hologic Inc"), ("PODD", "Insulet Corporation"),
    ("TDOC", "Teladoc Health Inc"), ("FIVE", "Five Below Inc"), ("QSR", "Restaurant Brands International"),
    ("DPZ", "Domino's Pizza Inc"), ("WING", "Wingstop Inc"), ("SHAK", "Shake Shack Inc"),
    ("CAVA", "CAVA Group Inc"), ("ROK", "Rockwell Automation Inc"), ("PH", "Parker-Hannifin Corporation"),
    ("ITW", "Illinois Tool Works Inc"), ("CARR", "Carrier Global Corporation"), ("OTIS", "Otis Worldwide Corporation"),
    ("JCI", "Johnson Controls International"), ("DOW", "Dow Inc"), ("PPG", "PPG Industries Inc"),
    ("SHW", "The Sherwin-Williams Company"), ("NEM", "Newmont Corporation"), ("FCX", "Freeport-McMoRan Inc"),
    ("NTAP", "NetApp Inc"), ("PSTG", "Pure Storage Inc"), ("WDC", "Western Digital Corporation"),
    ("STX", "Seagate Technology Holdings"), ("SOUN", "SoundHound AI Inc"), ("MGNI", "Magnite Inc"),
    ("PUBM", "PubMatic Inc"), ("RDFN", "Redfin Corporation"), ("OPEN", "Opendoor Technologies Inc"),
    ("COMP", "Compass Inc"), ("CHTR", "Charter Communications Inc"), ("ACLS", "Axcelis Technologies Inc"),
    ("ALGM", "Allegro MicroSystems Inc"), ("AMKR", "Amkor Technology Inc"), ("CRUS", "Cirrus Logic Inc"),
    ("DIOD", "Diodes Incorporated"), ("ENTG", "Entegris Inc"), ("LSCC", "Lattice Semiconductor Corp"),
    ("MKSI", "MKS Instruments Inc"), ("MTSI", "MACOM Technology Solutions"), ("POWI", "Power Integrations Inc"),
    ("SLAB", "Silicon Laboratories Inc"), ("SMTC", "Semtech Corporation"), ("WOLF", "Wolfspeed Inc"),
    ("ACAD", "ACADIA Pharmaceuticals Inc"), ("AKRO", "Akero Therapeutics Inc"), ("ARWR", "Arrowhead Pharmaceuticals Inc"),
    ("AXSM", "Axsome Therapeutics Inc"), ("BHVN", "Biohaven Ltd"), ("BLUE", "bluebird bio Inc"),
    ("BMRN", "BioMarin Pharmaceutical Inc"), ("BPMC", "Blueprint Medicines Corporation"), ("CYTK", "Cytokinetics Inc"),
    ("DAWN", "Day One Biopharmaceuticals Inc"), ("DVAX", "Dynavax Technologies Corporation"), ("FOLD", "Amicus Therapeutics Inc"),
    ("HALO", "Halozyme Therapeutics Inc"), ("IMVT", "Immunovant Inc"), ("IONS", "Ionis Pharmaceuticals Inc"),
    ("ITCI", "Intra-Cellular Therapies Inc"), ("JAZZ", "Jazz Pharmaceuticals plc"), ("KYMR", "Kymera Therapeutics Inc"),
    ("LEGN", "Legend Biotech Corporation"), ("MDGL", "Madrigal Pharmaceuticals Inc"), ("MNKD", "MannKind Corporation"),
    ("NTRA", "Natera Inc"), ("NVCR", "NovoCure Limited"), ("PCVX", "Vaxcyte Inc"),
    ("RARE", "Ultragenyx Pharmaceutical Inc"), ("RCKT", "Rocket Pharmaceuticals Inc"), ("ROIV", "Roivant Sciences Ltd"),
    ("RXRX", "Recursion Pharmaceuticals Inc"), ("SAVA", "Cassava Sciences Inc"), ("SGMO", "Sangamo Therapeutics Inc"),
    ("TGTX", "TG Therapeutics Inc"), ("UTHR", "United Therapeutics Corporation"), ("VKTX", "Viking Therapeutics Inc"),
    ("VTRS", "Viatris Inc"), ("XENE", "Xenon Pharmaceuticals Inc"), ("ATRC", "AtriCure Inc"),
    ("AXNX", "Axonics Inc"), ("BRKR", "Bruker Corporation"), ("GMED", "Globus Medical Inc"),
    ("HSIC", "Henry Schein Inc"), ("INSP", "Inspire Medical Systems Inc"), ("LMAT", "LeMaitre Vascular Inc"),
    ("MMSI", "Merit Medical Systems Inc"), ("NEOG", "Neogen Corporation"), ("NVST", "Envista Holdings Corporation"),
    ("OMCL", "Omnicell Inc"), ("PDCO", "Patterson Companies Inc"), ("RGEN", "Repligen Corporation"),
    ("STAA", "STAAR Surgical Company"), ("TMDX", "TransMedics Group Inc"), ("TNDM", "Tandem Diabetes Care Inc"),
    ("VCYT", "Veracyte Inc"), ("XRAY", "DENTSPLY SIRONA Inc"), ("BBY", "Best Buy Co Inc"),
    ("BJ", "BJ's Wholesale Club Holdings"), ("BURL", "Burlington Stores Inc"), ("CASY", "Casey's General Stores Inc"),
    ("DECK", "Deckers Outdoor Corporation"), ("BROS", "Dutch Bros Inc"), ("FL", "Foot Locker Inc"),
    ("GPC", "Genuine Parts Company"), ("KDP", "Keurig Dr Pepper Inc"), ("KHC", "The Kraft Heinz Company"),
    ("KMB", "Kimberly-Clark Corporation"), ("MNST", "Monster Beverage Corporation"), ("ORLY", "O'Reilly Automotive Inc"),
    ("PEP", "PepsiCo Inc"), ("PG", "Procter & Gamble Co"), ("POOL", "Pool Corporation"),
    ("SKX", "Skechers U.S.A. Inc"), ("TXRH", "Texas Roadhouse Inc"), ("UAA", "Under Armour Inc"),
    ("WEN", "The Wendy's Company"), ("AOS", "A. O. Smith Corporation"), ("BWA", "BorgWarner Inc"),
    ("CTAS", "Cintas Corporation"), ("DOV", "Dover Corporation"), ("FAST", "Fastenal Company"),
    ("GWW", "W.W. Grainger Inc"), ("IR", "Ingersoll Rand Inc"), ("MIDD", "The Middleby Corporation"),
    ("MSA", "MSA Safety Incorporated"), ("NDSN", "Nordson Corporation"), ("PCAR", "PACCAR Inc"),
    ("PWR", "Quanta Services Inc"), ("ROP", "Roper Technologies Inc"), ("SNA", "Snap-on Incorporated"),
    ("SWK", "Stanley Black & Decker Inc"), ("TT", "Trane Technologies plc"), ("VMI", "Valmont Industries Inc"),
    ("WAB", "Westinghouse Air Brake Technologies"), ("WSO", "Watsco Inc"), ("XYL", "Xylem Inc"),
    ("ARRY", "Array Technologies Inc"), ("FLNC", "Fluence Energy Inc"), ("ORA", "Ormat Technologies Inc"),
    ("SHLS", "Shoals Technologies Group Inc"), ("SPWR", "SunPower Corporation"), ("STEM", "Stem Inc"),
    ("CHRW", "C.H. Robinson Worldwide Inc"), ("EXPD", "Expeditors International"), ("JBHT", "J.B. Hunt Transport Services"),
    ("KNX", "Knight-Swift Transportation"), ("ODFL", "Old Dominion Freight Line Inc"), ("SNDR", "Schneider National Inc"),
    ("UPS", "United Parcel Service Inc"), ("XPO", "XPO Inc"), ("AZPN", "Aspen Technology Inc"),
    ("BOX", "Box Inc"), ("COUP", "Coupa Software Inc"), ("DBX", "Dropbox Inc"),
    ("DOCN", "DigitalOcean Holdings Inc"), ("DT", "Dynatrace Inc"), ("EQIX", "Equinix Inc"),
    ("FSLY", "Fastly Inc"), ("GTLB", "GitLab Inc"), ("HUBS", "HubSpot Inc"),
    ("IONQ", "IonQ Inc"), ("JAMF", "Jamf Holding Corp"), ("MNDY", "Monday.com Ltd"),
    ("NICE", "NICE Ltd"), ("PCOR", "Procore Technologies Inc"), ("PD", "PagerDuty Inc"),
    ("QLYS", "Qualys Inc"), ("QTWO", "Q2 Holdings Inc"), ("RPD", "Rapid7 Inc"),
    ("TYL", "Tyler Technologies Inc"), ("VRNS", "Varonis Systems Inc"), ("WIX", "Wix.com Ltd"),
    ("ZI", "ZoomInfo Technologies Inc"), ("ESTC", "Elastic NV"), ("CFLT", "Confluent Inc"),
    ("SPLK", "Splunk Inc"), ("CYBR", "CyberArk Software Ltd"), ("TENB", "Tenable Holdings Inc"),
    ("JNPR", "Juniper Networks Inc"), ("FFIV", "F5 Inc"), ("CIEN", "Ciena Corporation"),
    ("S", "SentinelOne Inc"), ("SMAR", "Smartsheet Inc"), ("APP", "AppLovin Corporation"),
    ("AAOI", "Applied Optoelectronics Inc"), ("LITE", "Lumentum Holdings Inc"), ("VIAV", "Viavi Solutions Inc"),
    ("ALSN", "Allison Transmission Holdings"), ("OLLI", "Ollie's Bargain Outlet Holdings"), ("EAT", "Brinker International Inc"),
    ("WDFC", "WD-40 Company"), ("CWEN", "Clearway Energy Inc"), ("NEP", "NextEra Energy Partners LP"),
    ("DKNG", "DraftKings Inc"), ("CVX", "Chevron Corporation"), ("XOM", "Exxon Mobil Corporation"),
    ("SLB", "Schlumberger NV"), ("HAL", "Halliburton Company"), ("SHEL", "Shell plc"),
]

# Cache
_cache = {}
_cache_time = {}
CACHE_DURATION = 300  # 5 minutes

def get_cache(key):
    if key in _cache:
        if datetime.now().timestamp() - _cache_time[key] < CACHE_DURATION:
            return _cache[key]
    return None

def set_cache(key, value):
    _cache[key] = value
    _cache_time[key] = datetime.now().timestamp()

@app.get("/")
def root():
    return {
        "app": "OptiMax API",
        "version": "1.0.0",
        "endpoints": {
            "opportunities": "/opportunities",
            "stock_analysis": "/stock/{symbol}",
            "scheduler_status": "/scheduler/status",
            "documentation": "/docs"
        },
        "total_stocks": len(SHARIAH_STOCKS),
        "message": "Shariah-compliant stock analysis for halal investing"
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
    """حساب نقاط السهم"""
    score = 0
    
    try:
        # RSI - Oversold
        if row['RSI'] < 30:
            score += 3
        elif row['RSI'] < 40:
            score += 1.5
            
        # MACD Bullish
        if row['MACD'] > row['MACD_Signal']:
            if row['MACD_Hist'] > 0:
                score += 3  # Fresh bullish crossover
            else:
                score += 1.5  # Bullish continuation
                
        # Near Bollinger Lower
        if current_price < row['BB_lower'] * 1.02:
            score += 2
            
        # Price above SMA-20
        if current_price > row['SMA_20']:
            score += 1
            
        # Volume surge
        if row['Volume'] > row['Volume_SMA'] * 5:
            score += 1
            
    except:
        pass
        
    return score

def get_signal(score):
    """تحديد الإشارة"""
    if score >= 6:
        return "Strong Buy"
    elif score >= 4:
        return "Buy"
    elif score >= 2:
        return "Hold"
    elif score >= 0:
        return "Sell"
    else:
        return "Strong Sell"

@app.get("/opportunities")
def get_opportunities(limit: int = 20):
    """الحصول على أفضل الفرص"""
    
    # التحقق من الـ cache
    cache_key = f"opportunities_{limit}"
    cached = get_cache(cache_key)
    if cached:
        return cached
    
    logger.info(f"Analyzing {len(SHARIAH_STOCKS)} stocks...")
    
    opportunities = []
    symbols = [s[0] for s in SHARIAH_STOCKS[:100]]  # تحليل أول 100 للسرعة
    
    try:
        # تحميل جماعي
        data = yf.download(symbols, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
        
        for symbol, name in SHARIAH_STOCKS[:100]:
            try:
                if len(symbols) > 1:
                    stock_data = data[symbol].copy()
                else:
                    stock_data = data.copy()
                
                if stock_data.empty:
                    continue
                
                # حساب المؤشرات
                stock_data = calculate_indicators(stock_data)
                
                if stock_data.empty or len(stock_data) < 50:
                    continue
                
                latest = stock_data.iloc[-1]
                current_price = latest['Close']
                
                # حساب النقاط
                score = calculate_score(latest, current_price)
                signal = get_signal(score)
                
                opportunities.append({
                    "symbol": symbol,
                    "name": name,
                    "price": round(float(current_price), 2),
                    "change_pct": round(((current_price - stock_data.iloc[-2]['Close']) / stock_data.iloc[-2]['Close']) * 100, 2),
                    "score": round(score, 1),
                    "signal": signal,
                    "rsi": round(float(latest['RSI']), 1) if not pd.isna(latest['RSI']) else None,
                    "macd": round(float(latest['MACD']), 2) if not pd.isna(latest['MACD']) else None,
                })
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # ترتيب حسب النقاط
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        result = {
            "updated_at": datetime.now(pytz.timezone('US/Eastern')).isoformat(),
            "total_analyzed": len(opportunities),
            "opportunities": opportunities[:limit]
        }
        
        set_cache(cache_key, result)
        return result
        
    except Exception as e:
        logger.error(f"Error in get_opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock/{symbol}")
def get_stock_analysis(symbol: str):
    """تحليل كامل لسهم معين"""
    
    symbol = symbol.upper()
    
    # التحقق من الـ cache
    cache_key = f"stock_{symbol}"
    cached = get_cache(cache_key)
    if cached:
        return cached
    
    # التحقق من أن السهم في القائمة
    stock_info = next((s for s in SHARIAH_STOCKS if s[0] == symbol), None)
    if not stock_info:
        raise HTTPException(status_code=404, detail="Stock not in Shariah-compliant list")
    
    try:
        # جلب البيانات
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # حساب المؤشرات
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        # حساب النقاط
        score = calculate_score(latest, current_price)
        signal = get_signal(score)
        
        # تحليل Claude AI
        claude_analysis = None
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                
                prompt = f"""أنت محلل مالي محترف. حلل السهم التالي:

السهم: {symbol} - {stock_info[1]}
السعر الحالي: ${current_price:.2f}
RSI: {latest['RSI']:.1f}
MACD: {latest['MACD']:.2f}
Signal: {signal}

قدم تحليل شامل يتضمن:
1. نقطة الدخول المقترحة
2. هدف الخروج
3. وقف الخسارة
4. نسبة النجاح المتوقعة (%)
5. نسبة المخاطرة/المكافأة
6. توقعات يومية
7. توقعات أسبوعية
8. توقعات شهرية
9. توصية واضحة (Buy/Sell/Hold)
10. المخاطر الرئيسية

قدم الإجابة بصيغة JSON فقط بدون أي نص إضافي."""

                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                claude_text = message.content[0].text
                # محاولة استخراج JSON
                import json
                try:
                    claude_analysis = json.loads(claude_text)
                except:
                    claude_analysis = {"raw_analysis": claude_text}
                    
            except Exception as e:
                logger.error(f"Claude API error: {e}")
        
        result = {
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
                "macd_hist": round(float(latest['MACD_Hist']), 4),
                "sma_20": round(float(latest['SMA_20']), 2),
                "sma_50": round(float(latest['SMA_50']), 2),
                "bb_upper": round(float(latest['BB_upper']), 2),
                "bb_lower": round(float(latest['BB_lower']), 2),
            },
            "claude_analysis": claude_analysis,
            "updated_at": datetime.now(pytz.timezone('US/Eastern')).isoformat()
        }
        
        set_cache(cache_key, result)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scheduler/status")
def scheduler_status():
    """حالة النظام"""
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    # التحقق من ساعات السوق
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    is_weekday = now_et.weekday() < 5
    is_market_hours = market_open <= now_et <= market_close and is_weekday
    
    return {
        "scheduler_running": True,
        "market_open": is_market_hours,
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET (%A)"),
        "market_hours": "9:30 AM - 4:00 PM ET (Mon-Fri)",
        "total_stocks": len(SHARIAH_STOCKS),
        "cache_duration": f"{CACHE_DURATION} seconds"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

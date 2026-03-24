from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import anthropic
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OptiMax API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/app")
def serve_frontend():
    return FileResponse("index.html")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

SHARIAH_STOCKS = [("AAPL", "Apple Inc"), ("MSFT", "Microsoft Corporation"), ("GOOGL", "Alphabet Inc Class A"), ("GOOG", "Alphabet Inc Class C"), ("NVDA", "NVIDIA Corporation"), ("META", "Meta Platforms Inc"), ("TSLA", "Tesla Inc"), ("AMZN", "Amazon.com Inc"), ("AMD", "Advanced Micro Devices Inc"), ("AVGO", "Broadcom Inc"), ("INTC", "Intel Corporation"), ("QCOM", "QUALCOMM Inc"), ("TXN", "Texas Instruments Inc"), ("AMAT", "Applied Materials Inc"), ("LRCX", "Lam Research Corporation"), ("KLAC", "KLA Corporation"), ("NXPI", "NXP Semiconductors NV"), ("MU", "Micron Technology Inc"), ("MRVL", "Marvell Technology Inc"), ("SNPS", "Synopsys Inc"), ("CDNS", "Cadence Design Systems Inc"), ("MCHP", "Microchip Technology Inc"), ("ON", "ON Semiconductor Corporation"), ("SWKS", "Skyworks Solutions Inc"), ("QRVO", "Qorvo Inc"), ("MPWR", "Monolithic Power Systems Inc"), ("ADI", "Analog Devices Inc"), ("ASML", "ASML Holding NV"), ("ADBE", "Adobe Inc"), ("CRM", "Salesforce Inc"), ("ORCL", "Oracle Corporation"), ("NOW", "ServiceNow Inc"), ("WDAY", "Workday Inc"), ("DDOG", "Datadog Inc"), ("SNOW", "Snowflake Inc"), ("MDB", "MongoDB Inc"), ("TEAM", "Atlassian Corporation"), ("ZM", "Zoom Video Communications"), ("DOCU", "DocuSign Inc"), ("PLTR", "Palantir Technologies Inc"), ("PANW", "Palo Alto Networks Inc"), ("CRWD", "CrowdStrike Holdings Inc"), ("FTNT", "Fortinet Inc"), ("ZS", "Zscaler Inc"), ("OKTA", "Okta Inc"), ("NET", "Cloudflare Inc"), ("CSCO", "Cisco Systems Inc"), ("IBM", "IBM Corporation"), ("DELL", "Dell Technologies Inc"), ("HPQ", "HP Inc"), ("HPE", "Hewlett Packard Enterprise"), ("FICO", "Fair Isaac Corporation"), ("ANSS", "ANSYS Inc"), ("INTU", "Intuit Inc"), ("TYL", "Tyler Technologies Inc"), ("RNG", "RingCentral Inc"), ("TWLO", "Twilio Inc"), ("DBX", "Dropbox Inc"), ("BOX", "Box Inc"), ("ZI", "ZoomInfo Technologies Inc"), ("VEEV", "Veeva Systems Inc"), ("SHOP", "Shopify Inc"), ("EBAY", "eBay Inc"), ("ETSY", "Etsy Inc"), ("DASH", "DoorDash Inc"), ("UBER", "Uber Technologies Inc"), ("LYFT", "Lyft Inc"), ("ABNB", "Airbnb Inc"), ("PYPL", "PayPal Holdings Inc"), ("SQ", "Block Inc"), ("COIN", "Coinbase Global Inc"), ("AFRM", "Affirm Holdings Inc"), ("SPOT", "Spotify Technology SA"), ("ROKU", "Roku Inc"), ("PINS", "Pinterest Inc"), ("SNAP", "Snap Inc"), ("MTCH", "Match Group Inc"), ("RBLX", "Roblox Corporation"), ("U", "Unity Software Inc"), ("TTWO", "Take-Two Interactive Software"), ("EA", "Electronic Arts Inc"), ("ATVI", "Activision Blizzard"), ("NFLX", "Netflix Inc"), ("DIS", "The Walt Disney Company"), ("PARA", "Paramount Global"), ("WBD", "Warner Bros Discovery"), ("FWONK", "Liberty Media Formula One"), ("LSXMA", "Liberty Media SiriusXM"), ("CHTR", "Charter Communications"), ("CMCSA", "Comcast Corporation"), ("TMUS", "T-Mobile US Inc"), ("AMGN", "Amgen Inc"), ("GILD", "Gilead Sciences Inc"), ("REGN", "Regeneron Pharmaceuticals"), ("VRTX", "Vertex Pharmaceuticals Inc"), ("BIIB", "Biogen Inc"), ("ILMN", "Illumina Inc"), ("ISRG", "Intuitive Surgical Inc"), ("DXCM", "DexCom Inc"), ("ALGN", "Align Technology Inc"), ("ABT", "Abbott Laboratories"), ("TMO", "Thermo Fisher Scientific Inc"), ("DHR", "Danaher Corporation"), ("SYK", "Stryker Corporation"), ("EW", "Edwards Lifesciences Corporation"), ("BSX", "Boston Scientific Corporation"), ("INCY", "Incyte Corporation"), ("ALNY", "Alnylam Pharmaceuticals Inc"), ("MRNA", "Moderna Inc"), ("BNTX", "BioNTech SE"), ("IDXX", "IDEXX Laboratories Inc"), ("ZBH", "Zimmer Biomet Holdings Inc"), ("BDX", "Becton Dickinson and Company"), ("BAX", "Baxter International Inc"), ("RMD", "ResMed Inc"), ("HOLX", "Hologic Inc"), ("TECH", "Bio-Techne Corporation"), ("RVTY", "Revvity Inc"), ("IQV", "IQVIA Holdings Inc"), ("CRL", "Charles River Laboratories"), ("RGEN", "Repligen Corporation"), ("EXAS", "Exact Sciences Corporation"), ("ARWR", "Arrowhead Pharmaceuticals"), ("IONS", "Ionis Pharmaceuticals"), ("RARE", "Ultragenyx Pharmaceutical"), ("BMRN", "BioMarin Pharmaceutical"), ("SRPT", "Sarepta Therapeutics"), ("NBIX", "Neurocrine Biosciences"), ("UTHR", "United Therapeutics"), ("JAZZ", "Jazz Pharmaceuticals"), ("HALO", "Halozyme Therapeutics"), ("BLUE", "bluebird bio"), ("FOLD", "Amicus Therapeutics"), ("LEGN", "Legend Biotech"), ("KRYS", "Krystal Biotech"), ("RCKT", "Rocket Pharmaceuticals"), ("AGIO", "Agios Pharmaceuticals"), ("NTRA", "Natera Inc"), ("NVTA", "Invitae Corporation"), ("PACB", "Pacific Biosciences"), ("VCYT", "Veracyte Inc"), ("CDNA", "CareDx Inc"), ("IRTC", "iRhythm Technologies"), ("OFIX", "Orthofix Medical"), ("NVCR", "NovoCure Limited"), ("TMDX", "TransMedics Group"), ("PRVA", "Privia Health Group"), ("CRVL", "CorVel Corporation"), ("LFST", "LifeStance Health Group"), ("HIMS", "Hims & Hers Health"), ("DOCS", "Doximity Inc"), ("TDOC", "Teladoc Health Inc"), ("OSCR", "Oscar Health"), ("CLOV", "Clover Health"), ("SDGR", "Schrodinger Inc"), ("RXRX", "Recursion Pharmaceuticals"), ("ABCL", "AbCellera Biologics"), ("RLAY", "Relay Therapeutics"), ("VERV", "Verve Therapeutics"), ("BEAM", "Beam Therapeutics"), ("CRSP", "CRISPR Therapeutics"), ("EDIT", "Editas Medicine"), ("NTLA", "Intellia Therapeutics"), ("CRBU", "Caribou Biosciences"), ("PRME", "Prime Medicine"), ("ACLX", "Arcellx Inc"), ("SANA", "Sana Biotechnology"), ("DAWN", "Day One Biopharmaceuticals"), ("KROS", "Keros Therapeutics"), ("IMVT", "Immunovant Inc"), ("ARVN", "Arvinas Inc"), ("KYMR", "Kymera Therapeutics"), ("COST", "Costco Wholesale Corporation"), ("NKE", "NIKE Inc"), ("SBUX", "Starbucks Corporation"), ("MCD", "McDonald's Corporation"), ("LULU", "Lululemon Athletica Inc"), ("WMT", "Walmart Inc"), ("TGT", "Target Corporation"), ("HD", "The Home Depot Inc"), ("LOW", "Lowe's Companies Inc"), ("TJX", "The TJX Companies Inc"), ("ROST", "Ross Stores Inc"), ("DG", "Dollar General Corporation"), ("DLTR", "Dollar Tree Inc"), ("ULTA", "Ulta Beauty Inc"), ("YUM", "Yum! Brands Inc"), ("CMG", "Chipotle Mexican Grill Inc"), ("BKNG", "Booking Holdings Inc"), ("MAR", "Marriott International"), ("HLT", "Hilton Worldwide Holdings"), ("MGM", "MGM Resorts International"), ("WYNN", "Wynn Resorts Limited"), ("LVS", "Las Vegas Sands Corp"), ("CZR", "Caesars Entertainment"), ("PENN", "PENN Entertainment"), ("DKNG", "DraftKings Inc"), ("FLUT", "Flutter Entertainment"), ("CHWY", "Chewy Inc"), ("CVNA", "Carvana Co"), ("CPNG", "Coupang Inc"), ("BABA", "Alibaba Group Holding"), ("JD", "JD.com Inc"), ("PDD", "PDD Holdings Inc"), ("MELI", "MercadoLibre Inc"), ("SE", "Sea Limited"), ("GRAB", "Grab Holdings Limited"), ("BEKE", "KE Holdings Inc"), ("VIPS", "Vipshop Holdings Limited"), ("LI", "Li Auto Inc"), ("XPEV", "XPeng Inc"), ("NIO", "NIO Inc"), ("BILI", "Bilibili Inc"), ("TME", "Tencent Music Entertainment"), ("DIDI", "DiDi Global Inc"), ("TAL", "TAL Education Group"), ("EDU", "New Oriental Education"), ("GOTU", "Gaotu Techedu Inc"), ("IQ", "iQIYI Inc"), ("NTES", "NetEase Inc"), ("BIDU", "Baidu Inc"), ("FSLR", "First Solar Inc"), ("ENPH", "Enphase Energy Inc"), ("SEDG", "SolarEdge Technologies Inc"), ("RUN", "Sunrun Inc"), ("GNRC", "Generac Holdings Inc"), ("RIVN", "Rivian Automotive Inc"), ("LCID", "Lucid Group Inc"), ("PLUG", "Plug Power Inc"), ("CHPT", "ChargePoint Holdings Inc"), ("BLNK", "Blink Charging Co"), ("QS", "QuantumScape Corporation"), ("ALB", "Albemarle Corporation"), ("SQM", "Sociedad Quimica y Minera"), ("LAC", "Lithium Americas Corp"), ("LTHM", "Livent Corporation"), ("MP", "MP Materials Corp"), ("WOLF", "Wolfspeed Inc"), ("OLED", "Universal Display Corporation"), ("POWI", "Power Integrations"), ("VICR", "Vicor Corporation"), ("SLDP", "Solid Power Inc"), ("STEM", "Stem Inc"), ("FLNC", "Fluence Energy Inc"), ("BE", "Bloom Energy Corporation"), ("FCEL", "FuelCell Energy Inc"), ("BLDP", "Ballard Power Systems"), ("CAT", "Caterpillar Inc"), ("DE", "Deere & Company"), ("ETN", "Eaton Corporation"), ("EMR", "Emerson Electric Co"), ("LIN", "Linde plc"), ("APD", "Air Products and Chemicals Inc"), ("ECL", "Ecolab Inc"), ("DD", "DuPont de Nemours Inc"), ("BA", "Boeing Company"), ("GE", "General Electric Company"), ("HON", "Honeywell International Inc"), ("MMM", "3M Company"), ("ITW", "Illinois Tool Works"), ("ROK", "Rockwell Automation"), ("PH", "Parker-Hannifin Corporation"), ("IR", "Ingersoll Rand Inc"), ("CARR", "Carrier Global Corporation"), ("OTIS", "Otis Worldwide Corporation"), ("TT", "Trane Technologies plc"), ("JCI", "Johnson Controls International"), ("AME", "AMETEK Inc"), ("ROP", "Roper Technologies Inc"), ("FTV", "Fortive Corporation"), ("WSO", "Watsco Inc"), ("FAST", "Fastenal Company"), ("WM", "Waste Management Inc"), ("RSG", "Republic Services Inc"), ("WCN", "Waste Connections Inc"), ("SRCL", "Stericycle Inc"), ("GFL", "GFL Environmental Inc"), ("RYN", "Rayonier Inc"), ("WY", "Weyerhaeuser Company"), ("PCH", "PotlatchDeltic Corporation"), ("LPX", "Louisiana-Pacific Corporation"), ("BCC", "Boise Cascade Company"), ("UFPI", "UFP Industries Inc"), ("TREX", "Trex Company Inc"), ("AZEK", "AZEK Company Inc"), ("FND", "Floor & Decor Holdings"), ("BECN", "Beacon Roofing Supply"), ("OC", "Owens Corning"), ("VMC", "Vulcan Materials Company"), ("MLM", "Martin Marietta Materials"), ("SUM", "Summit Materials Inc"), ("USLM", "United States Lime & Minerals"), ("CRH", "CRH plc"), ("STRL", "Sterling Infrastructure Inc"), ("MTZ", "MasTec Inc"), ("PRIM", "Primoris Services Corporation"), ("F", "Ford Motor Company"), ("GM", "General Motors Company"), ("STLA", "Stellantis NV"), ("TM", "Toyota Motor Corporation"), ("HMC", "Honda Motor Co Ltd"), ("NSANY", "Nissan Motor Co Ltd"), ("HYMTF", "Hyundai Motor Company"), ("BMWYY", "BMW AG"), ("VWAGY", "Volkswagen AG"), ("RACE", "Ferrari NV"), ("POAHY", "Porsche Automobil Holding"), ("GELYF", "Geely Automobile Holdings"), ("FUJHY", "Subaru Corporation"), ("MZDAY", "Mazda Motor Corporation"), ("DDAIF", "Daimler Truck Holding AG"), ("VLKAF", "Volvo AB"), ("PCAR", "PACCAR Inc"), ("NAV", "Navistar International"), ("CMI", "Cummins Inc"), ("LEA", "Lear Corporation"), ("PEP", "PepsiCo Inc"), ("KO", "Coca-Cola Company"), ("MDLZ", "Mondelez International"), ("GIS", "General Mills Inc"), ("K", "Kellogg Company"), ("CPB", "Campbell Soup Company"), ("HSY", "Hershey Company"), ("SJM", "J.M. Smucker Company"), ("CAG", "Conagra Brands Inc"), ("HRL", "Hormel Foods Corporation"), ("TSN", "Tyson Foods Inc"), ("BG", "Bunge Limited"), ("ADM", "Archer-Daniels-Midland Company"), ("CALM", "Cal-Maine Foods Inc"), ("INGR", "Ingredion Incorporated"), ("MKC", "McCormick & Company"), ("LANC", "Lancaster Colony Corporation"), ("JJSF", "J & J Snack Foods Corp"), ("SENEA", "Seneca Foods Corporation"), ("FARM", "Farmer Bros Co"), ("SAM", "Boston Beer Company"), ("TAP", "Molson Coors Beverage Company"), ("BUD", "Anheuser-Busch InBev SA"), ("STZ", "Constellation Brands Inc"), ("CELH", "Celsius Holdings Inc"), ("MNST", "Monster Beverage Corporation"), ("KDP", "Keurig Dr Pepper Inc"), ("COKE", "Coca-Cola Consolidated Inc"), ("FIZZ", "National Beverage Corp"), ("PRMW", "Primo Water Corporation")]

_cache = {}
_cache_time = {}
CACHE_DURATION = 1800

def get_cache(key):
    if key in _cache:
        if datetime.now().timestamp() - _cache_time[key] < CACHE_DURATION:
            return _cache[key]
    return None

def set_cache(key, value):
    _cache[key] = value
    _cache_time[key] = datetime.now().timestamp()

def is_market_open():
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    if now_et.weekday() >= 5:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close

def get_stock_data_yf(symbol):
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.error(f"YFinance error {symbol}: {e}")
        return None

def calculate_trading_days_ahead(start_date, num_days):
    """حساب عدد أيام التداول (تخطي السبت والأحد)"""
    current = start_date
    days_added = 0
    while days_added < num_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # 0-4 = Monday-Friday
            days_added += 1
    return current.strftime('%Y-%m-%d')

@app.get("/")
def root():
    return {"app": "OptiMax API", "version": "5.0.0", "total_stocks": len(SHARIAH_STOCKS), "market_open": is_market_open(), "message": "Complete Enhanced Analysis - Sector Flow, Multi-Indicators, Term Explanations"}

def calculate_indicators(df):
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['SMA_20'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['SMA_20'] - (df['BB_std'] * 2)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_100'] = df['Close'].rolling(window=100).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean() if len(df) >= 200 else None
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        df['MFI_Change'] = df['MFI'].diff()
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        pos_di = 100 * (pos_dm.rolling(window=14).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=14).mean() / atr)
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        df['ADX'] = dx.rolling(window=14).mean()
        df['ATR'] = atr
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

def calculate_score(row, current_price, prev_close, df):
    score = 0.0
    signals = []
    sma_100_penalty = 0
    try:
        if not pd.isna(row['SMA_100']):
            if current_price > row['SMA_100']:
                score += 2.0
                signals.append("فوق SMA-100 (اتجاه صاعد)")
            else:
                sma_100_penalty = -0.5
                signals.append("تحت SMA-100 (تحذير)")
        volume_ratio = row['Volume'] / row['Volume_SMA'] if row['Volume_SMA'] > 0 else 1
        if row['RSI'] < 30:
            if volume_ratio < 2:
                score += 2.0
                signals.append("RSI تشبع بيعي + حجم معتدل (فرصة)")
            else:
                score += 0.5
                signals.append("RSI منخفض + حجم ضخم (هروب محتمل)")
        elif row['RSI'] < 40:
            score += 1.0
            signals.append("RSI تشبع بيعي")
        elif row['RSI'] > 70:
            score -= 0.5
            signals.append("RSI تشبع شرائي")
        if row['MACD'] > row['MACD_Signal']:
            if row['MACD_Hist'] > 0 and row['MACD_Hist'] > df['MACD_Hist'].iloc[-2]:
                score += 2.0
                signals.append("MACD تقاطع صاعد قوي")
            else:
                score += 1.0
                signals.append("MACD إيجابي")
        if current_price < row['BB_lower'] * 1.02:
            score += 1.0
            signals.append("قرب حد بولينجر السفلي")
        if not pd.isna(row['MFI']):
            if row['MFI'] < 30:
                score += 1.0
                signals.append("MFI تشبع بيعي (تدفق أموال سلبي)")
            elif row['MFI'] > 70:
                score -= 0.5
                signals.append("MFI تشبع شرائي")
        if not pd.isna(row['ADX']):
            if row['ADX'] > 25:
                score += 1.0
                signals.append(f"ADX قوي ({row['ADX']:.1f}) - اتجاه واضح")
            elif row['ADX'] < 20:
                signals.append("ADX ضعيف (سوق جانبي)")
        strong_momentum = False
        if not pd.isna(row['MFI']) and not pd.isna(row['ADX']) and not pd.isna(row['MFI_Change']):
            if row['MFI_Change'] > 0 and row['ADX'] > 25:
                score += 2.5
                strong_momentum = True
                signals.append("⚡ MFI صاعد + ADX قوي (زخم استثنائي)")
        if not pd.isna(row['ROC']):
            if row['ROC'] > 5:
                score += 0.5
                signals.append("زخم إيجابي قوي")
            elif row['ROC'] > 2:
                score += 0.3
                signals.append("زخم إيجابي")
            elif row['ROC'] < -5:
                score -= 0.3
                signals.append("زخم سلبي")
        gap_pct = ((current_price - prev_close) / prev_close) * 100
        if abs(gap_pct) > 3:
            if gap_pct < -3 and volume_ratio > 2:
                score -= 1.0
                signals.append(f"فجوة هابطة ({gap_pct:.1f}%) + حجم عالي (تحذير)")
            elif gap_pct > 3 and volume_ratio > 1.5:
                score += 0.5
                signals.append(f"فجوة صاعدة ({gap_pct:.1f}%) + حجم (دخول محتمل)")
        if strong_momentum and not pd.isna(row['ROC']):
            if row['ROC'] > 5:
                score -= sma_100_penalty
                if sma_100_penalty < 0:
                    signals.append("✓ زخم قوي يتجاوز SMA-100")
        else:
            score += sma_100_penalty
    except Exception as e:
        logger.error(f"Error in score: {e}")
    return min(max(score, 0), 10.0), signals

def get_signal(score):
    if score >= 9:
        return "Super Strong Buy"
    elif score >= 7:
        return "Strong Buy"
    elif score >= 5:
        return "Buy"
    elif score >= 3:
        return "Hold"
    elif score >= 1:
        return "Sell"
    else:
        return "Strong Sell"

@app.get("/top-opportunities")
def get_top_opportunities():
    cache_key = "top_opportunities"
    cached = get_cache(cache_key)
    if cached:
        return cached
    logger.info(f"Starting YFinance analysis of {len(SHARIAH_STOCKS)} stocks...")
    sp500_return = 0
    all_scores = []
    total = len(SHARIAH_STOCKS)
    for idx, (symbol, name) in enumerate(SHARIAH_STOCKS, 1):
        try:
            logger.info(f"Analyzing {idx}/{total}: {symbol}")
            stock_data = get_stock_data_yf(symbol)
            if stock_data is None or stock_data.empty or len(stock_data) < 100:
                continue
            stock_data = calculate_indicators(stock_data)
            latest = stock_data.iloc[-1]
            prev_close = stock_data.iloc[-2]['Close']
            current_price = latest['Close']
            score, signals = calculate_score(latest, current_price, prev_close, stock_data)
            stock_return = ((current_price - stock_data.iloc[-30]['Close']) / stock_data.iloc[-30]['Close']) * 100
            relative_strength = stock_return - sp500_return
            all_scores.append({"symbol": symbol, "name": name, "price": round(float(current_price), 2), "change_pct": round(((current_price - prev_close) / prev_close) * 100, 2), "score": round(score, 1), "signal": get_signal(score), "rsi": round(float(latest['RSI']), 1) if not pd.isna(latest['RSI']) else None, "macd": round(float(latest['MACD']), 2) if not pd.isna(latest['MACD']) else None, "relative_strength": round(relative_strength, 2)})
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            continue
    logger.info(f"Analysis complete! Analyzed {len(all_scores)} stocks successfully")
    all_scores.sort(key=lambda x: x['score'], reverse=True)
    top_50 = all_scores[:50]
    final_top_20 = []
    if claude_client and len(top_50) > 0:
        try:
            stocks_summary = "\n".join([f"{i+1}. {s['symbol']} - Score: {s['score']}/10" for i, s in enumerate(top_50)])
            prompt = f"""اختر أفضل 20 سهم:\n\n{stocks_summary}\n\nأرجع فقط رموز مفصولة بفاصلة:"""
            message = claude_client.messages.create(model="claude-sonnet-4-20250514", max_tokens=500, messages=[{"role": "user", "content": prompt}])
            selected_symbols = message.content[0].text.strip().split(',')
            selected_symbols = [s.strip() for s in selected_symbols][:20]
            for symbol in selected_symbols:
                stock = next((s for s in top_50 if s['symbol'] == symbol), None)
                if stock:
                    final_top_20.append(stock)
        except Exception as e:
            logger.error(f"Claude error: {e}")
            final_top_20 = top_50[:20]
    else:
        final_top_20 = top_50[:20]
    result = {"updated_at": datetime.now(pytz.timezone('US/Eastern')).isoformat(), "market_open": is_market_open(), "total_analyzed": len(all_scores), "sp500_performance": round(sp500_return, 2), "top_opportunities": final_top_20, "analysis_method": "YFinance API"}
    set_cache(cache_key, result)
    return result

@app.get("/analysis/{symbol}")
def get_detailed_analysis(symbol: str):
    symbol = symbol.upper()
    cache_key = f"analysis_{symbol}"
    cached = get_cache(cache_key)
    if cached:
        return cached
    stock_info = next((s for s in SHARIAH_STOCKS if s[0] == symbol), None)
    if not stock_info:
        raise HTTPException(status_code=404, detail="Stock not in Shariah-compliant list")
    try:
        import yfinance as yf
        df = get_stock_data_yf(symbol)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]['Close']
        current_price = latest['Close']
        score, signals = calculate_score(latest, current_price, prev_close, df)
        signal = get_signal(score)
        atr_value = latest['ATR']
        dynamic_stop = current_price - (1.5 * atr_value)
        
        # جلب البيانات الأساسية
        stock = yf.Ticker(symbol)
        info = stock.info
        pe_ratio = info.get('trailingPE', 0)
        eps = info.get('trailingEps', 0)
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Unknown')
        
        # حساب Volume Analysis المتقدم
        current_volume = latest['Volume']
        avg_volume = latest['Volume_SMA']
        volume_diff_pct = ((current_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
        
        # تحليل اتجاه Volume (آخر 5 أيام)
        recent_volumes = df['Volume'].tail(5).tolist()
        volume_trend = "صاعد" if recent_volumes[-1] > recent_volumes[0] else "هابط"
        
        # Multi-Timeframe Analysis (تقدير من البيانات المتاحة)
        daily_trend = "صاعد" if current_price > latest['SMA_50'] else "هابط"
        
        # تحليل القطاع (تقدير من Volume + Price)
        sector_strength = "قوي" if volume_diff_pct > 20 and score >= 7 else "متوسط" if score >= 5 else "ضعيف"
        
        basic_data = {
            "symbol": symbol,
            "name": stock_info[1],
            "price": round(float(current_price), 2),
            "change_pct": round(((current_price - prev_close) / prev_close) * 100, 2),
            "score": round(score, 1),
            "signal": signal,
            "signals_detail": signals,
            "indicators": {
                "rsi": round(float(latest['RSI']), 2),
                "macd": round(float(latest['MACD']), 4),
                "macd_signal": round(float(latest['MACD_Signal']), 4),
                "macd_histogram": round(float(latest['MACD_Hist']), 4),
                "sma_20": round(float(latest['SMA_20']), 2),
                "sma_50": round(float(latest['SMA_50']), 2),
                "sma_100": round(float(latest['SMA_100']), 2) if not pd.isna(latest['SMA_100']) else None,
                "bb_upper": round(float(latest['BB_upper']), 2),
                "bb_lower": round(float(latest['BB_lower']), 2),
                "mfi": round(float(latest['MFI']), 2) if not pd.isna(latest['MFI']) else None,
                "adx": round(float(latest['ADX']), 2) if not pd.isna(latest['ADX']) else None,
                "atr": round(float(latest['ATR']), 2),
                "roc": round(float(latest['ROC']), 2) if not pd.isna(latest['ROC']) else None
            },
            "fundamentals": {
                "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                "eps": round(eps, 2) if eps else None,
                "market_cap": market_cap,
                "sector": sector
            },
        "volume_analysis": {
            "current": int(current_volume),
            "average": int(avg_volume),
            "difference_pct": round(volume_diff_pct, 2),
            "trend": volume_trend,
            "is_unusual": bool(abs(volume_diff_pct) > 50)
        },
        "market_context": {
            "daily_trend": daily_trend,
            "sector_strength": sector_strength
        },
        "dynamic_stop_loss": round(float(dynamic_stop), 2)
    }

        if claude_client:
            try:
                today = datetime.now(pytz.timezone('US/Eastern'))
                
                # حساب الصلاحية الذكية
                if score >= 9:
                    trading_days = 7
                elif score >= 7:
                    trading_days = 5
                elif score >= 5:
                    trading_days = 3
                else:
                    trading_days = 1
                
                valid_until = calculate_trading_days_ahead(today, trading_days)
                
                prompt = f"""أنت محلل مالي خبير متخصص في الأسواق الأمريكية. حلل السهم التالي بشكل شامل ومفصل واحترافي:

━━━━━━━━━━━━━━━━━━━━━
📊 معلومات السهم
━━━━━━━━━━━━━━━━━━━━━
الشركة: {stock_info[1]} ({symbol})
القطاع: {sector}
السعر الحالي: ${current_price:.2f}
التغير اليومي: {basic_data['change_pct']}%
النقاط الفنية: {basic_data['score']}/10
التاريخ الحالي: {today.strftime('%Y-%m-%d')}

━━━━━━━━━━━━━━━━━━━━━
📈 التحليل الفني الكامل
━━━━━━━━━━━━━━━━━━━━━
RSI: {basic_data['indicators']['rsi']} (0-30=تشبع بيعي، 30-70=محايد، 70+=تشبع شرائي)
MACD: {basic_data['indicators']['macd']} (إيجابي=صاعد، سلبي=هابط)
ADX: {basic_data['indicators']['adx']} (25+=اتجاه قوي، <20=جانبي)
MFI: {basic_data['indicators']['mfi']} (تدفق الأموال: 30-=بيعي، 70+=شرائي)
ROC: {basic_data['indicators']['roc']}% (الزخم)
ATR: {basic_data['indicators']['atr']} (التذبذب)

SMA-20: ${basic_data['indicators']['sma_20']}
SMA-50: ${basic_data['indicators']['sma_50']}
SMA-100: ${basic_data['indicators']['sma_100']}
Bollinger Upper: ${basic_data['indicators']['bb_upper']}
Bollinger Lower: ${basic_data['indicators']['bb_lower']}

━━━━━━━━━━━━━━━━━━━━━
💼 التحليل الأساسي
━━━━━━━━━━━━━━━━━━━━━
P/E Ratio: {basic_data['fundamentals']['pe_ratio']} (كم تدفع مقابل $1 ربح)
EPS: ${basic_data['fundamentals']['eps']} (ربحية السهم)
Market Cap: ${basic_data['fundamentals']['market_cap']:,.0f}

━━━━━━━━━━━━━━━━━━━━━
📊 تحليل حجم التداول
━━━━━━━━━━━━━━━━━━━━━
الحجم الحالي: {basic_data['volume_analysis']['current']:,}
المتوسط: {basic_data['volume_analysis']['average']:,}
الفرق: {basic_data['volume_analysis']['difference_pct']}%
الاتجاه: {basic_data['volume_analysis']['trend']}
غير طبيعي: {'نعم ⚠️' if basic_data['volume_analysis']['is_unusual'] else 'لا'}

━━━━━━━━━━━━━━━━━━━━━
🌍 سياق السوق
━━━━━━━━━━━━━━━━━━━━━
الاتجاه اليومي: {basic_data['market_context']['daily_trend']}
قوة القطاع: {basic_data['market_context']['sector_strength']}

━━━━━━━━━━━━━━━━━━━━━

أرجع JSON فقط بهذا التنسيق الدقيق (بدون أي نص إضافي):
{{
  "confidence": 85,
  "success_rating": 8.5,
  "profit_probability": 78,
  "entry_point": {current_price:.2f},
  "target_short": 0.00,
  "target_medium": 0.00,
  "stop_loss": {basic_data['dynamic_stop_loss']},
  "valid_until": "{valid_until}",
  "trading_days": {trading_days},
  "summary": "تحليل شامل يغطي جميع المؤشرات الفنية والأساسية وحجم التداول (3-4 أسطر)",
  "risks": [
    "مخاطرة محددة 1",
    "مخاطرة محددة 2",
    "مخاطرة محددة 3"
  ],
  "opportunities": [
    "فرصة محددة 1",
    "فرصة محددة 2",
    "فرصة محددة 3"
  ],
  "alerts": [
    "تنبيه محدد 1 (مثل: كسر $X = خروج)",
    "تنبيه محدد 2 (مثل: وصول $Y = أرباح جزئية)",
    "تنبيه محدد 3 (مثل: حدث قادم في تاريخ معين)"
  ],
  "sector_flow": "تحليل تدفق السيولة في القطاع {sector} - هل المؤسسات والبنوك تشتري؟ ما معنى ذلك للسهم؟ (2-3 أسطر)",
  "historical_success": "نسبة نجاح الإشارات المماثلة تاريخياً (تقدير ذكي بناءً على قوة الإشارة)",
  "recommendation": "التوصية النهائية والنصيحة العملية للمتداول (سطرين)",
  "glossary": {{
    "RSI": "مؤشر القوة النسبية - يقيس سرعة تغير السعر. أقل من 30 = تشبع بيعي (فرصة شراء)، أكثر من 70 = تشبع شرائي (احذر الهبوط)",
    "MACD": "تقارب وتباعد المتوسطات - يكشف قوة واتجاه الترند. إيجابي = صاعد، سلبي = هابط، تقاطع صاعد = إشارة قوية",
    "Volume": "حجم التداول - كم سهم تم تداوله. حجم عالي = اهتمام كبير (موثوق)، حجم منخفض = اهتمام ضعيف",
    "P/E Ratio": "نسبة السعر للأرباح - كم تدفع مقابل كل $1 ربح. منخفض = رخيص، مرتفع = غالي",
    "Support": "الدعم - سعر يصعب كسره للأسفل. كسره = احتمال هبوط",
    "Resistance": "المقاومة - سعر يصعب تجاوزه للأعلى. كسره = احتمال صعود",
    "Money Flow": "تدفق الأموال - هل الأموال تدخل السهم أو تخرج؟ إيجابي = صعود، سلبي = هبوط",
    "Institutional Buying": "شراء المؤسسات - عندما البنوك والصناديق الكبرى تشتري = إشارة قوية للصعود"
  }}
}}

ملاحظات حاسمة:
1. confidence: درجة الثقة من 0-100 بناءً على قوة كل المؤشرات
2. success_rating: تقييم نجاح الصفقة من 0-10
3. profit_probability: احتمالية الربح بالنسبة المئوية
4. summary: يجب أن يشمل تحليل شامل لكل المؤشرات (الفنية + الأساسية + الحجم + القطاع)
5. risks/opportunities: 3 نقاط محددة وواضحة لكل منهما
6. alerts: 3 تنبيهات عملية مع أسعار محددة
7. sector_flow: تحليل مفصل لتدفق السيولة وماذا يعني
8. historical_success: نسبة تقديرية ذكية
9. glossary: شرح مبسط لـ 8 مصطلحات رئيسية"""

                message = claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=3000,
                    messages=[{"role": "user", "content": prompt}]
                )
                claude_response = message.content[0].text.strip()
                if "```json" in claude_response:
                    claude_response = claude_response.split("```json")[1].split("```")[0].strip()
                elif "```" in claude_response:
                    claude_response = claude_response.split("```")[1].split("```")[0].strip()
                claude_analysis = json.loads(claude_response)
                basic_data["claude_analysis"] = claude_analysis
            except Exception as e:
                logger.error(f"Claude error: {e}")
                basic_data["claude_analysis"] = None
        else:
            basic_data["claude_analysis"] = None

        basic_data["updated_at"] = datetime.now(pytz.timezone('US/Eastern')).isoformat()
        set_cache(cache_key, basic_data)
        return basic_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-status")
def market_status():
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    return {
        "market_open": is_market_open(),
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET (%A)"),
        "market_hours": "9:30 AM - 4:00 PM ET (Mon-Fri)",
        "total_stocks": len(SHARIAH_STOCKS),
        "cache_duration": f"{CACHE_DURATION} seconds (30 min)",
        "claude_enabled": claude_client is not None,
        "yfinance_enabled": True,
        "analysis_version": "5.0.0 - Complete Enhanced Analysis",
        "features": [
            "✅ Smart validity dates (trading days only)",
            "✅ Fundamental analysis (P/E, EPS, Market Cap, Sector)",
            "✅ Advanced volume analysis with trend detection",
            "✅ Sector flow & institutional buying analysis",
            "✅ Confidence & success ratings (0-100, 0-10)",
            "✅ Risk/Opportunity assessment (3 points each)",
            "✅ Smart alerts with specific prices",
            "✅ Historical success rate estimation",
            "✅ Multi-timeframe context",
            "✅ Term explanations (8 key terms)",
            "✅ Comprehensive analysis of all 10+ indicators"
        ]
    }

app = app

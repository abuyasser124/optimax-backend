 OptiMax - Shariah-Compliant Stock Analysis API

تطبيق تحليل الأسهم الشرعية بالذكاء الاصطناعي

## المميزات

- ✅ 396 سهم أمريكي شرعي (NASDAQ & NYSE)
- ✅ تحليل فني شامل (RSI, MACD, Bollinger Bands)
- ✅ تحليل ذكي بواسطة Claude AI
- ✅ نقاط دخول/خروج محددة
- ✅ نسبة نجاح متوقعة
- ✅ توقعات يومية/أسبوعية/شهرية
- ✅ متوافق مع معايير AAOIFI

## API Endpoints

### 1. الصفحة الرئيسية
```
GET /
```

### 2. أفضل الفرص
```
GET /opportunities?limit=20
```

### 3. تحليل سهم معين
```
GET /stock/AAPL
```

### 4. حالة النظام
```
GET /scheduler/status
```

## التشغيل المحلي

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
uvicorn main:app --reload
```

## Deploy على Render

1. Fork هذا المشروع
2. ربطه مع Render
3. إضافة ANTHROPIC_API_KEY في Environment Variables
4. Deploy!

## المعايير الشرعية

جميع الأسهم متوافقة مع:
- AAOIFI Standards
- MSCI Islamic Index
- Business Activity: Halal operations
- Financial Ratios: Debt < 33%, Interest income < 5%

---

Built with ❤️ for Halal Investing

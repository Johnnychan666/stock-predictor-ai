# AI 股票開盤漲跌預測系統

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Johnnychan666/stock-predictor-ai)

這是非 Streamlit 股票預測網站。後端使用 Python HTTP server 提供 API，前端使用原生 HTML/CSS/JavaScript，包含搜尋、K 線圖、新聞情緒、關聯美股/產業指標、TWSE 三大法人籌碼、十因子矩陣、模型回測，以及台股下次開盤攻擊候選推薦。

重要限制：模型輸出是機率估計，不是保證獲利或投資建議。請把回測勝率、是否打敗基準、新聞與籌碼訊號、自己的風險控管一起判斷。

## 本機啟動

```powershell
python app.py
```

打開：

```text
http://127.0.0.1:8000
```

## Render 部署

專案已包含 `render.yaml` 和 `runtime.txt`，可部署成 Render Web Service。

Render 會使用：

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Health check: `/api/health`

`app.py` 支援 Render 的 `PORT` 環境變數，部署時會自動綁定 `0.0.0.0:$PORT`。

## 使用方式

- 搜尋框可輸入股票代號或公司名，例如 `2408`、`南亞科`、`華邦電`、`2330.TW`、`AAPL`、`MU`
- 預設預測「下一次開盤」相對最新收盤是上漲或下跌
- 可切換成「下一日收盤」預測
- 頁面會顯示 K 線圖、上漲/下跌機率、技術模型機率、外部訊號修正、回測勝率、新聞、關聯市場、法人買賣超
- 十因子矩陣納入：大盤方向、產業同業、重大消息、財報預期差、估值、技術、量能流動性、籌碼、選擇權/衍生品、宏觀事件
- 「台股下次開盤攻擊候選」會掃描高流動性台股池，推薦目前綜合分數最高的一檔

## 主要檔案

- `app.py`：網站後端與 API
- `web/index.html`：前端頁面
- `web/assets/styles.css`：股票介面樣式
- `web/assets/app.js`：搜尋、API 呼叫、Canvas K 線圖與互動
- `stock_predictor/model.py`：量化預測與回測
- `stock_predictor/research.py`：新聞、法人、關聯市場資料
- `stock_predictor/recommender.py`：台股下次開盤候選推薦

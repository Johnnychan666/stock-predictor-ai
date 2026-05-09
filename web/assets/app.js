const $ = (id) => document.getElementById(id);

const state = {
  selected: { symbol: "2330.TW", name: "台積電" },
  prices: [],
  hover: null
};

const el = {
  form: $("searchForm"),
  search: $("symbolSearch"),
  suggestions: $("suggestions"),
  market: $("market"),
  targetMode: $("targetMode"),
  years: $("years"),
  chartDays: $("chartDays"),
  backtestDays: $("backtestDays"),
  retrainEvery: $("retrainEvery"),
  threshold: $("threshold"),
  external: $("external"),
  status: $("statusBar"),
  canvas: $("klineCanvas"),
  hoverInfo: $("hoverInfo"),
  chartTitle: $("chartTitle"),
  messageBox: $("messageBox"),
  direction: $("direction"),
  confidence: $("confidence"),
  probUp: $("probUp"),
  probDown: $("probDown"),
  technicalProb: $("technicalProb"),
  externalAdjustment: $("externalAdjustment"),
  latestClose: $("latestClose"),
  latestDate: $("latestDate"),
  backtestAccuracy: $("backtestAccuracy"),
  backtestEdge: $("backtestEdge"),
  newsLabel: $("newsLabel"),
  newsScore: $("newsScore"),
  relatedLabel: $("relatedLabel"),
  relatedScore: $("relatedScore"),
  institutionalLabel: $("institutionalLabel"),
  flowSummary: $("flowSummary"),
  factorAdjustment: $("factorAdjustment"),
  factorGrid: $("factorGrid"),
  newsRows: $("newsRows"),
  marketRows: $("marketRows"),
  macroSummary: $("macroSummary"),
  macroList: $("macroList"),
  relatedRows: $("relatedRows"),
  flowRows: $("flowRows"),
  batchSymbols: $("batchSymbols"),
  batchButton: $("batchButton"),
  batchRows: $("batchRows"),
  bestPickSymbol: $("bestPickSymbol"),
  bestPickMeta: $("bestPickMeta"),
  bestPickReasons: $("bestPickReasons"),
  recommendButton: $("recommendButton"),
  clockDate: $("clockDate"),
  clockTime: $("clockTime")
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m]));
}

function pct(value, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "--";
}

function signed(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(digits)}`;
}

function price(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(n >= 1000 ? 0 : 2) : "--";
}

function setStatus(text, tone = "") {
  if (!el.status) return;
  el.status.textContent = text;
  el.status.className = `status ${tone}`.trim();
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function readParams() {
  return new URLSearchParams({
    years: el.years?.value || "3",
    market: el.market?.value || "tw",
    target_mode: el.targetMode?.value || "next_open",
    threshold: el.threshold?.value || "0.005",
    chart_days: el.chartDays?.value || "160",
    backtest_days: el.backtestDays?.value || "120",
    retrain_every: el.retrainEvery?.value || "5",
    external: el.external?.checked ? "1" : "0"
  });
}

async function searchSymbols() {
  const query = (el.search?.value || "").trim();
  if (!query) {
    el.suggestions.innerHTML = "";
    return;
  }
  try {
    const params = new URLSearchParams({ q: query, market: el.market?.value || "tw" });
    const data = await getJson(`/api/search?${params}`);
    el.suggestions.innerHTML = (data.results || []).map((item) => `
      <button type="button" class="suggestion" data-symbol="${esc(item.symbol)}" data-name="${esc(item.name || item.symbol)}">
        <strong>${esc(item.symbol)}</strong><span>${esc(item.name || "")}</span>
      </button>
    `).join("");
  } catch (error) {
    el.suggestions.innerHTML = `<div class="empty">搜尋暫時失敗：${esc(error.message)}</div>`;
  }
}

function selectSymbol(symbol, name = "") {
  state.selected = { symbol, name };
  el.search.value = name ? `${symbol} ${name}` : symbol;
  el.suggestions.innerHTML = "";
}

async function analyzeSelected(event) {
  event?.preventDefault();
  const input = (el.search?.value || "").trim();
  if (!state.selected.symbol || !input.includes(state.selected.symbol)) {
    const token = input.split(/\s+/)[0].trim();
    if (token) selectSymbol(token, "");
  }
  if (!state.selected.symbol) return;

  setStatus(`分析 ${state.selected.symbol} 中，正在讀取價格、新聞、籌碼與關聯市場...`);
  el.messageBox.textContent = "模型正在整合技術面、消息面、大盤、同業、籌碼、估值、宏觀與衍生品訊號。";
  try {
    const params = readParams();
    params.set("symbol", state.selected.symbol);
    params.set("company_name", state.selected.name || "");
    const data = await getJson(`/api/analyze?${params}`);
    renderAnalysis(data);
    setStatus(`完成 ${data.symbol} 隔日預測`, "ok");
  } catch (error) {
    setStatus(`分析失敗：${error.message}`, "bad");
    el.messageBox.textContent = "請確認股票代號格式，例如台股 2330.TW、美股 MU。免費資料源偶爾會被限流，稍後再試通常可以恢復。";
  }
}

function renderAnalysis(data) {
  state.prices = data.price_history || [];
  const pred = data.prediction || {};
  const context = pred.context || {};
  const name = data.company_name || state.selected.name || data.symbol;
  state.selected = { symbol: data.symbol, name };

  el.direction.textContent = pred.direction === "up" ? "看漲" : "看跌";
  el.direction.className = `metric-value ${pred.direction === "up" ? "up" : "down"}`;
  el.confidence.textContent = pct(pred.confidence);
  el.probUp.textContent = pct(pred.probability_up);
  el.probDown.textContent = pct(pred.probability_down);
  el.technicalProb.textContent = pct(pred.raw_probability_up);
  el.externalAdjustment.textContent = signed((pred.external_adjustment || 0) * 100, 1) + " pts";
  el.latestClose.textContent = price(data.latest_close);
  el.latestDate.textContent = data.latest_date || "--";
  el.backtestAccuracy.textContent = pct(pred.backtest_accuracy);
  el.backtestEdge.textContent = signed(((pred.backtest_accuracy || 0) - 0.5) * 100, 1) + " pts";

  el.newsLabel.textContent = context.news?.label || "--";
  el.newsScore.textContent = signed(context.news?.score);
  el.relatedLabel.textContent = context.related_markets?.label || "--";
  el.relatedScore.textContent = signed(context.related_markets?.score);
  el.institutionalLabel.textContent = context.institutional?.label || "--";
  el.flowSummary.textContent = context.institutional?.summary || "暫無籌碼摘要";
  el.chartTitle.textContent = `${data.symbol} ${name}`;
  el.messageBox.textContent = pred.explanation || "完成分析。";

  renderFactors(context, pred.external_adjustment || 0);
  renderNews(context.news || {});
  renderMarket(context);
  renderRelated(context.related_markets || {});
  renderFlows(context);
  drawCandles();
}

function renderFactors(context, adjustment) {
  const factors = context.factors || [];
  el.factorAdjustment.textContent = `${signed(adjustment * 100, 1)} pts`;
  el.factorGrid.innerHTML = factors.map((factor) => {
    const tone = factor.score > 0.15 ? "up" : factor.score < -0.15 ? "down" : "flat";
    return `
      <div class="factor-card ${tone}">
        <div><strong>${esc(factor.name)}</strong><span>${esc(factor.label || "")}</span></div>
        <b>${signed(factor.score)}</b>
        <p>${esc(factor.summary || "")}</p>
      </div>
    `;
  }).join("") || `<div class="empty">尚無外部因子。</div>`;
}

function renderNews(news) {
  const rows = news.items || [];
  el.newsRows.innerHTML = rows.map((item) => `
    <tr>
      <td><a href="${esc(item.url || "#")}" target="_blank" rel="noreferrer">${esc(item.title || "無標題")}</a></td>
      <td>${esc(item.source || "--")}</td>
      <td>${esc(item.published || "--")}</td>
      <td class="num ${item.sentiment_score > 0 ? "up" : item.sentiment_score < 0 ? "down" : ""}">${signed(item.sentiment_score)}</td>
    </tr>
  `).join("") || `<tr><td colspan="4" class="empty">目前沒有抓到新聞。</td></tr>`;
}

function renderMarket(context) {
  const market = context.market_regime || {};
  const rows = market.items || [];
  el.marketRows.innerHTML = rows.map((item) => `
    <tr>
      <td>${esc(item.name || item.symbol)}</td>
      <td>${esc(item.symbol || "")}</td>
      <td class="num ${item.change_pct > 0 ? "up" : item.change_pct < 0 ? "down" : ""}">${pct((item.change_pct || 0) / 100)}</td>
      <td>${esc(item.last_date || "--")}</td>
    </tr>
  `).join("") || `<tr><td colspan="4" class="empty">市場資料不足。</td></tr>`;

  const macro = context.macro || {};
  el.macroSummary.textContent = macro.summary || "宏觀資料會依資料源可用性更新。";
  el.macroList.innerHTML = (macro.events || []).map((item) => `<li>${esc(item)}</li>`).join("") || `<li>尚無重大宏觀事件摘要。</li>`;
}

function renderRelated(related) {
  const rows = related.items || [];
  el.relatedRows.innerHTML = rows.map((item) => `
    <tr>
      <td>${esc(item.name || item.symbol)}</td>
      <td>${esc(item.symbol || "")}</td>
      <td class="num ${item.change_pct > 0 ? "up" : item.change_pct < 0 ? "down" : ""}">${pct((item.change_pct || 0) / 100)}</td>
      <td>${esc(item.reason || "同業/供應鏈")}</td>
    </tr>
  `).join("") || `<tr><td colspan="4" class="empty">未找到明確同業關聯。</td></tr>`;
}

function renderFlows(context) {
  const inst = context.institutional || {};
  const valuation = context.valuation || {};
  const derivatives = context.derivatives || {};
  const rows = [
    ["籌碼/法人", inst.label, inst.summary],
    ["估值", valuation.label, valuation.message],
    ["選擇權/衍生品", derivatives.label, derivatives.summary]
  ];
  el.flowRows.innerHTML = rows.map(([name, label, summary]) => `
    <tr><td>${esc(name)}</td><td>${esc(label || "--")}</td><td>${esc(summary || "資料不足")}</td></tr>
  `).join("");
}

function movingAverage(values, window) {
  return values.map((_, i) => {
    if (i + 1 < window) return null;
    const slice = values.slice(i + 1 - window, i + 1);
    return slice.reduce((a, b) => a + b, 0) / window;
  });
}

function drawCandles() {
  const canvas = el.canvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#07111f";
  ctx.fillRect(0, 0, w, h);

  const data = state.prices.slice(-Number(el.chartDays?.value || 160));
  if (!data.length) {
    ctx.fillStyle = "#8ea4bd";
    ctx.fillText("等待價格資料", 24, 36);
    return;
  }

  const pad = { left: 48, right: 18, top: 18, bottom: 30 };
  const innerW = w - pad.left - pad.right;
  const innerH = h - pad.top - pad.bottom;
  const highs = data.map((d) => Number(d.High));
  const lows = data.map((d) => Number(d.Low));
  const max = Math.max(...highs);
  const min = Math.min(...lows);
  const span = Math.max(max - min, 0.01);
  const y = (v) => pad.top + (max - v) / span * innerH;
  const x = (i) => pad.left + (i + 0.5) * innerW / data.length;

  ctx.strokeStyle = "rgba(142,164,189,.18)";
  ctx.lineWidth = 1;
  ctx.font = "11px Arial";
  ctx.fillStyle = "#8ea4bd";
  for (let i = 0; i <= 4; i++) {
    const yy = pad.top + innerH * i / 4;
    const val = max - span * i / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(w - pad.right, yy);
    ctx.stroke();
    ctx.fillText(price(val), 6, yy + 4);
  }

  const candleW = Math.max(3, Math.min(13, innerW / data.length * 0.62));
  data.forEach((d, i) => {
    const open = Number(d.Open), high = Number(d.High), low = Number(d.Low), close = Number(d.Close);
    const up = close >= open;
    const color = up ? "#1fd39a" : "#ff5c7a";
    const xx = x(i);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(xx, y(high));
    ctx.lineTo(xx, y(low));
    ctx.stroke();
    const top = Math.min(y(open), y(close));
    const bodyH = Math.max(2, Math.abs(y(open) - y(close)));
    ctx.fillRect(xx - candleW / 2, top, candleW, bodyH);
  });

  const closes = data.map((d) => Number(d.Close));
  [[5, "#f5c542"], [20, "#55a6ff"], [60, "#b079ff"]].forEach(([win, color]) => {
    const ma = movingAverage(closes, win);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let started = false;
    ma.forEach((v, i) => {
      if (v == null) return;
      if (!started) { ctx.moveTo(x(i), y(v)); started = true; }
      else ctx.lineTo(x(i), y(v));
    });
    ctx.stroke();
  });

  if (state.hover != null) {
    const i = Math.max(0, Math.min(data.length - 1, state.hover));
    const d = data[i];
    const xx = x(i);
    ctx.strokeStyle = "rgba(255,255,255,.38)";
    ctx.beginPath();
    ctx.moveTo(xx, pad.top);
    ctx.lineTo(xx, h - pad.bottom);
    ctx.stroke();
    el.hoverInfo.textContent = `${d.Date} O ${price(d.Open)} H ${price(d.High)} L ${price(d.Low)} C ${price(d.Close)} Vol ${Number(d.Volume || 0).toLocaleString()}`;
  } else {
    const last = data[data.length - 1];
    el.hoverInfo.textContent = `${last.Date} 收 ${price(last.Close)} 量 ${Number(last.Volume || 0).toLocaleString()}`;
  }
}

async function recommend() {
  setStatus("掃描台股候選名單，挑選隔日上漲機率最高標的...");
  try {
    const params = new URLSearchParams({
      limit: "12",
      years: el.years?.value || "3",
      threshold: el.threshold?.value || "0.005",
      target_mode: el.targetMode?.value || "next_open"
    });
    const data = await getJson(`/api/recommend?${params}`);
    const best = data.best_pick;
    if (!best) throw new Error("沒有可用推薦結果");
    el.bestPickSymbol.textContent = best.symbol;
    el.bestPickMeta.textContent = `${best.company_name || "台股候選"}｜上漲機率 ${pct(best.probability_up)}｜信心 ${pct(best.confidence)}｜${best.direction === "up" ? "看漲" : "看跌"}`;
    el.bestPickReasons.innerHTML = (best.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("") || `<li>${esc(best.explanation || "模型偏多。")}</li>`;
    setStatus(`推薦完成：${best.symbol}`, "ok");
  } catch (error) {
    setStatus(`推薦失敗：${error.message}`, "bad");
  }
}

async function runBatch() {
  const symbols = (el.batchSymbols?.value || "").split(/[\s,，]+/).map((s) => s.trim()).filter(Boolean);
  if (!symbols.length) return;
  el.batchRows.innerHTML = `<tr><td colspan="5" class="empty">批次分析中...</td></tr>`;
  try {
    const data = await getJson("/api/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbols,
        years: Number(el.years?.value || 3),
        market: el.market?.value || "tw",
        target_mode: el.targetMode?.value || "next_open",
        threshold: Number(el.threshold?.value || 0.005),
        external: Boolean(el.external?.checked)
      })
    });
    el.batchRows.innerHTML = (data.results || []).map((item) => `
      <tr>
        <td>${esc(item.symbol)}</td>
        <td>${esc(item.company_name || "")}</td>
        <td class="${item.direction === "up" ? "up" : "down"}">${item.direction === "up" ? "看漲" : "看跌"}</td>
        <td class="num">${pct(item.probability_up)}</td>
        <td class="num">${pct(item.confidence)}</td>
      </tr>
    `).join("") || `<tr><td colspan="5" class="empty">沒有結果。</td></tr>`;
  } catch (error) {
    el.batchRows.innerHTML = `<tr><td colspan="5" class="empty">批次失敗：${esc(error.message)}</td></tr>`;
  }
}

function updateClock() {
  const now = new Date();
  el.clockDate.textContent = now.toLocaleDateString("zh-TW", { year: "numeric", month: "2-digit", day: "2-digit", weekday: "short" });
  el.clockTime.textContent = now.toLocaleTimeString("zh-TW", { hour12: false });
}

el.form?.addEventListener("submit", analyzeSelected);
el.search?.addEventListener("input", () => {
  window.clearTimeout(state.searchTimer);
  state.searchTimer = window.setTimeout(searchSymbols, 180);
});
el.suggestions?.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-symbol]");
  if (!button) return;
  selectSymbol(button.dataset.symbol, button.dataset.name || "");
  analyzeSelected();
});
el.recommendButton?.addEventListener("click", recommend);
el.batchButton?.addEventListener("click", runBatch);
el.chartDays?.addEventListener("change", drawCandles);
window.addEventListener("resize", drawCandles);

el.canvas?.addEventListener("mousemove", (event) => {
  const rect = el.canvas.getBoundingClientRect();
  const data = state.prices.slice(-Number(el.chartDays?.value || 160));
  if (!data.length) return;
  const padLeft = 48, padRight = 18;
  const innerW = rect.width - padLeft - padRight;
  const x = Math.max(0, Math.min(innerW, event.clientX - rect.left - padLeft));
  state.hover = Math.round(x / innerW * (data.length - 1));
  drawCandles();
});
el.canvas?.addEventListener("mouseleave", () => { state.hover = null; drawCandles(); });

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach((body) => body.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`)?.classList.add("active");
  });
});

updateClock();
setInterval(updateClock, 1000);
selectSymbol("2330.TW", "台積電");
searchSymbols();
analyzeSelected();

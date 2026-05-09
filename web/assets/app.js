const $ = (id) => document.getElementById(id);

const state = {
  selected: { symbol: "2408.TW", name: "南亞科" },
  prices: [],
  hoverIndex: null,
  searchTimer: null
};

const el = {
  form: $("searchForm"), search: $("symbolSearch"), suggestions: $("suggestions"), market: $("market"),
  targetMode: $("targetMode"), years: $("years"), chartDays: $("chartDays"), backtestDays: $("backtestDays"),
  retrainEvery: $("retrainEvery"), threshold: $("threshold"), external: $("external"), status: $("statusBar"),
  canvas: $("klineCanvas"), hoverInfo: $("hoverInfo"), chartTitle: $("chartTitle"), messageBox: $("messageBox"),
  direction: $("direction"), confidence: $("confidence"), probUp: $("probUp"), probDown: $("probDown"),
  technicalProb: $("technicalProb"), externalAdjustment: $("externalAdjustment"), latestClose: $("latestClose"), latestDate: $("latestDate"),
  backtestAccuracy: $("backtestAccuracy"), backtestEdge: $("backtestEdge"), newsLabel: $("newsLabel"), newsScore: $("newsScore"),
  relatedLabel: $("relatedLabel"), relatedScore: $("relatedScore"), institutionalLabel: $("institutionalLabel"), flowSummary: $("flowSummary"),
  factorAdjustment: $("factorAdjustment"), factorGrid: $("factorGrid"), newsRows: $("newsRows"), marketRows: $("marketRows"),
  macroSummary: $("macroSummary"), macroList: $("macroList"), relatedRows: $("relatedRows"), flowRows: $("flowRows"),
  batchSymbols: $("batchSymbols"), batchButton: $("batchButton"), batchRows: $("batchRows"),
  bestPickSymbol: $("bestPickSymbol"), bestPickMeta: $("bestPickMeta"), bestPickReasons: $("bestPickReasons"), recommendButton: $("recommendButton"),
  clockDate: $("clockDate"), clockTime: $("clockTime")
};

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m]));
const num = (value) => Number.isFinite(Number(value)) ? Number(value) : null;
const pct = (value, digits = 1) => num(value) === null ? "--" : `${(Number(value) * 100).toFixed(digits)}%`;
const signed = (value, digits = 2) => num(value) === null ? "--" : `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(digits)}`;
const money = (value) => num(value) === null ? "--" : Number(value).toLocaleString("zh-TW", { maximumFractionDigits: Number(value) >= 1000 ? 0 : 2 });
const flow = (value) => num(value) === null ? "--" : Number(value).toLocaleString("zh-TW", { maximumFractionDigits: 0 });
const isUp = (direction) => String(direction || "").includes("上") || String(direction || "").toLowerCase() === "up";

function setStatus(text, type = "") {
  el.status.textContent = text;
  el.status.className = `status-bar ${type}`.trim();
}

function showMessage(text) {
  el.messageBox.hidden = !text;
  el.messageBox.textContent = text || "";
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function analysisParams() {
  return new URLSearchParams({
    symbol: state.selected.symbol,
    name: state.selected.name || "",
    market: el.market.value,
    target_mode: el.targetMode.value,
    years: el.years.value,
    chart_days: el.chartDays.value,
    backtest_days: el.backtestDays.value,
    retrain_every: el.retrainEvery.value,
    threshold: el.threshold.value,
    external: el.external.checked ? "true" : "false"
  });
}

function selectSymbol(symbol, name = "") {
  state.selected = { symbol: symbol.trim(), name: name.trim() };
  el.search.value = name ? `${symbol} ${name}` : symbol;
  el.suggestions.hidden = true;
  el.suggestions.innerHTML = "";
}

async function searchSymbols() {
  const query = el.search.value.trim();
  if (!query) {
    el.suggestions.hidden = true;
    return;
  }
  try {
    const data = await getJson(`/api/search?${new URLSearchParams({ q: query, market: el.market.value, limit: "12" })}`);
    const items = data.items || [];
    el.suggestions.innerHTML = items.map((item) => `
      <button type="button" class="suggestion" data-symbol="${esc(item.symbol)}" data-name="${esc(item.name || "")}">
        <strong>${esc(item.symbol)}</strong><span>${esc(item.name || item.exchange || "")}</span>
      </button>`).join("") || `<div class="suggestion"><span>找不到符合的股票，會直接使用你輸入的代號。</span></div>`;
    el.suggestions.hidden = false;
  } catch (error) {
    el.suggestions.innerHTML = `<div class="suggestion"><span>搜尋失敗：${esc(error.message)}</span></div>`;
    el.suggestions.hidden = false;
  }
}

async function analyze(event) {
  event?.preventDefault();
  const typed = el.search.value.trim();
  if (!typed) return;
  if (!typed.includes(state.selected.symbol)) selectSymbol(typed.split(/\s+/)[0], "");

  setStatus(`正在分析 ${state.selected.symbol}：價格、新聞、大盤、同業、籌碼與十因子...`, "loading");
  showMessage("模型分析需要抓取多個免費市場資料源，第一次載入可能會比較久。");
  try {
    const data = await getJson(`/api/analyze?${analysisParams()}`);
    renderAnalysis(data);
    setStatus(`完成 ${data.symbol} 隔日開盤預測`, "");
  } catch (error) {
    setStatus(`分析失敗：${error.message}`, "error");
    showMessage("請檢查股票代號格式，例如 2408.TW、2344.TW、MU。免費資料源偶爾會限流，稍後再試通常可恢復。");
  }
}

function renderAnalysis(data) {
  const prediction = data.prediction || {};
  const context = data.context || {};
  state.prices = data.prices || [];
  state.selected = { symbol: data.symbol || prediction.symbol || state.selected.symbol, name: data.name || state.selected.name };

  const up = isUp(prediction.direction);
  el.direction.textContent = up ? "看漲" : "看跌";
  el.direction.closest(".signal-card")?.classList.toggle("up", up);
  el.direction.closest(".signal-card")?.classList.toggle("down", !up);
  el.confidence.textContent = `信心 ${pct(prediction.confidence)} (${prediction.confidence_label || "--"})`;
  el.probUp.textContent = pct(prediction.probability_up);
  el.probDown.textContent = `下跌 ${pct(prediction.probability_down)}`;
  el.technicalProb.textContent = pct(prediction.technical_probability_up);
  el.externalAdjustment.textContent = `外部修正 ${signed((prediction.external_adjustment || 0) * 100, 1)} pts`;
  el.latestClose.textContent = money(prediction.latest_close);
  el.latestDate.textContent = `資料日 ${prediction.latest_date || "--"}`;
  el.backtestAccuracy.textContent = pct(prediction.backtest_accuracy);
  el.backtestEdge.textContent = `相對基準 ${signed((prediction.backtest_edge || 0) * 100, 1)} pts`;
  el.newsLabel.textContent = prediction.news_label || context.news?.label || "--";
  el.newsScore.textContent = `score ${signed(prediction.news_score ?? context.news?.score)}`;
  el.relatedLabel.textContent = prediction.related_label || context.related?.label || "--";
  el.relatedScore.textContent = `score ${signed(prediction.related_score ?? context.related?.score)}`;
  el.institutionalLabel.textContent = prediction.institutional_label || context.institutional?.label || "--";
  el.flowSummary.textContent = context.institutional?.message || `外資5日 ${flow(prediction.foreign_net_buy_5d)} / 法人5日 ${flow(prediction.institutional_net_buy_5d)}`;
  el.chartTitle.textContent = `${state.selected.symbol}${state.selected.name ? ` ${state.selected.name}` : ""}`;
  showMessage(prediction.message || (context.notes || []).join("；"));

  renderFactors(context.factors || [], context.adjustment ?? prediction.external_adjustment);
  renderNews(context.news || {});
  renderMarket(context.market || {}, context.macro || {});
  renderRelated(context.related || {});
  renderFlow(context.institutional || {});
  drawCandles();
}

function renderFactors(factors, adjustment) {
  el.factorAdjustment.textContent = `外部總修正 ${signed((adjustment || 0) * 100, 1)} pts`;
  el.factorGrid.innerHTML = factors.map((factor) => {
    const cls = factor.score > 0.12 ? "positive" : factor.score < -0.12 ? "negative" : "";
    return `<article class="factor-card ${cls}">
      <div><span>${esc(factor.key || "")}</span><strong>${esc(factor.name || "因子")}</strong></div>
      <b>${signed(factor.score)}</b>
      <small>${esc(factor.label || factor.detail || "中性")}</small>
    </article>`;
  }).join("") || `<article class="factor-card"><strong>等待資料</strong><small>尚無十因子資料。</small></article>`;
}

function renderNews(news) {
  const items = news.items || [];
  el.newsRows.innerHTML = items.map((item) => `
    <tr>
      <td>${esc((item.published_at || "").slice(0, 16) || "--")}</td>
      <td>${esc(item.publisher || "--")}</td>
      <td><a href="${esc(item.link || "#")}" target="_blank" rel="noreferrer">${esc(item.title || "無標題")}</a></td>
      <td class="${Number(item.score) >= 0 ? "up-text" : "down-text"}">${signed(item.score)}</td>
    </tr>`).join("") || `<tr><td colspan="4">目前沒有抓到新聞。</td></tr>`;
}

function renderMarket(market, macro) {
  const moves = market.moves || [];
  el.marketRows.innerHTML = moves.map((item) => `
    <tr>
      <td>${esc(item.name || item.symbol)}</td><td>${esc(item.category || "--")}</td><td>${esc(item.latest_date || "--")}</td>
      <td class="${Number(item.return_1d) >= 0 ? "up-text" : "down-text"}">${pct(item.return_1d)}</td>
      <td class="${Number(item.return_5d) >= 0 ? "up-text" : "down-text"}">${pct(item.return_5d)}</td><td>${signed(item.score)}</td>
    </tr>`).join("") || `<tr><td colspan="6">市場資料不足。</td></tr>`;
  el.macroSummary.textContent = macro.label || market.message || "宏觀資料會依資料源可用性更新。";
  el.macroList.innerHTML = (macro.highlights || []).map((item) => `<li>${esc(item)}</li>`).join("") || `<li>尚無重大宏觀事件摘要。</li>`;
}

function renderRelated(related) {
  const moves = related.moves || [];
  el.relatedRows.innerHTML = moves.map((item) => `
    <tr><td>${esc(item.symbol)}</td><td>${esc(item.latest_date || "--")}</td>
      <td class="${Number(item.return_1d) >= 0 ? "up-text" : "down-text"}">${pct(item.return_1d)}</td>
      <td class="${Number(item.return_5d) >= 0 ? "up-text" : "down-text"}">${pct(item.return_5d)}</td>
      <td class="${Number(item.return_20d) >= 0 ? "up-text" : "down-text"}">${pct(item.return_20d)}</td></tr>`).join("") || `<tr><td colspan="5">未找到明確同業關聯。</td></tr>`;
}

function renderFlow(institutional) {
  const rows = institutional.rows || [];
  el.flowRows.innerHTML = rows.map((item) => `
    <tr><td>${esc(item.date)}</td><td>${flow(item.foreign_net)}</td><td>${flow(item.investment_trust_net)}</td><td>${flow(item.dealer_net)}</td><td>${flow(item.total_net)}</td></tr>`).join("") || `<tr><td colspan="5">${esc(institutional.message || "目前沒有法人籌碼資料。")}</td></tr>`;
}

function candleValue(row, key) {
  return row[key] ?? row[key.toLowerCase()] ?? row[key.charAt(0).toUpperCase() + key.slice(1)];
}

function drawCandles() {
  const canvas = el.canvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, w, h);

  const data = state.prices.slice(-Number(el.chartDays.value || 180));
  if (!data.length) {
    ctx.fillStyle = "#8b98a8";
    ctx.fillText("等待價格資料", 24, 34);
    return;
  }
  const pad = { l: 52, r: 20, t: 18, b: 32 };
  const innerW = w - pad.l - pad.r, innerH = h - pad.t - pad.b;
  const highs = data.map((d) => Number(candleValue(d, "high"))).filter(Number.isFinite);
  const lows = data.map((d) => Number(candleValue(d, "low"))).filter(Number.isFinite);
  const max = Math.max(...highs), min = Math.min(...lows), span = Math.max(max - min, 0.01);
  const y = (v) => pad.t + (max - v) / span * innerH;
  const x = (i) => pad.l + (i + 0.5) * innerW / data.length;

  ctx.strokeStyle = "rgba(255,255,255,.11)";
  ctx.fillStyle = "#8b98a8";
  ctx.font = "11px Segoe UI, Arial";
  for (let i = 0; i <= 4; i++) {
    const yy = pad.t + innerH * i / 4;
    const value = max - span * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    ctx.fillText(money(value), 6, yy + 4);
  }

  const candleW = Math.max(3, Math.min(12, innerW / data.length * 0.62));
  data.forEach((d, i) => {
    const open = Number(candleValue(d, "open"));
    const high = Number(candleValue(d, "high"));
    const low = Number(candleValue(d, "low"));
    const close = Number(candleValue(d, "close"));
    if (![open, high, low, close].every(Number.isFinite)) return;
    const color = close >= open ? "#e5534b" : "#18a66a";
    const xx = x(i);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.moveTo(xx, y(high)); ctx.lineTo(xx, y(low)); ctx.stroke();
    const top = Math.min(y(open), y(close));
    const bodyH = Math.max(2, Math.abs(y(open) - y(close)));
    ctx.fillRect(xx - candleW / 2, top, candleW, bodyH);
  });

  drawAverage(ctx, data, 5, x, y, "#f5c542");
  drawAverage(ctx, data, 20, x, y, "#26a6b9");
  drawAverage(ctx, data, 60, x, y, "#b58cff");

  const hover = state.hoverIndex;
  const shown = hover == null ? data[data.length - 1] : data[Math.max(0, Math.min(data.length - 1, hover))];
  if (hover != null) {
    const xx = x(Math.max(0, Math.min(data.length - 1, hover)));
    ctx.strokeStyle = "rgba(255,255,255,.36)";
    ctx.beginPath(); ctx.moveTo(xx, pad.t); ctx.lineTo(xx, h - pad.b); ctx.stroke();
  }
  el.hoverInfo.textContent = `${candleValue(shown, "date")} O ${money(candleValue(shown, "open"))} H ${money(candleValue(shown, "high"))} L ${money(candleValue(shown, "low"))} C ${money(candleValue(shown, "close"))} Vol ${flow(candleValue(shown, "volume"))}`;
}

function drawAverage(ctx, data, windowSize, x, y, color) {
  const closes = data.map((d) => Number(candleValue(d, "close")));
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  closes.forEach((_, i) => {
    if (i + 1 < windowSize) return;
    const slice = closes.slice(i + 1 - windowSize, i + 1);
    const avg = slice.reduce((a, b) => a + b, 0) / windowSize;
    if (!started) { ctx.moveTo(x(i), y(avg)); started = true; }
    else ctx.lineTo(x(i), y(avg));
  });
  ctx.stroke();
}

async function recommend() {
  setStatus("正在掃描台股高流動性候選池...", "loading");
  try {
    const params = new URLSearchParams({ limit: "16", years: Math.min(Number(el.years.value || 2), 5).toString() });
    const data = await getJson(`/api/recommend?${params}`);
    const best = data.best;
    if (!best) throw new Error("沒有推薦結果");
    el.bestPickSymbol.textContent = best.symbol;
    el.bestPickMeta.textContent = `${best.label || "候選"}｜上漲機率 ${pct(best.estimated_up_probability)}｜分數 ${signed(best.score)}｜${best.latest_date || "--"}`;
    el.bestPickReasons.innerHTML = (best.reasons || []).map((item) => `<span>${esc(item)}</span>`).join("");
    setStatus(`推薦完成：${best.symbol}`, "");
  } catch (error) {
    setStatus(`推薦失敗：${error.message}`, "error");
  }
}

async function runBatch() {
  const symbols = el.batchSymbols.value.split(/[\s,，]+/).map((item) => item.trim()).filter(Boolean);
  if (!symbols.length) return;
  el.batchRows.innerHTML = `<tr><td colspan="8">批次分析中...</td></tr>`;
  try {
    const data = await getJson("/api/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbols,
        years: Number(el.years.value || 5),
        market: el.market.value,
        target_mode: el.targetMode.value,
        threshold: Number(el.threshold.value || 0),
        backtest_days: Number(el.backtestDays.value || 252),
        retrain_every: Number(el.retrainEvery.value || 20),
        external: el.external.checked
      })
    });
    el.batchRows.innerHTML = (data.rows || []).map((item) => {
      const up = isUp(item.direction);
      return `<tr><td>${esc(item.symbol)}</td><td class="${up ? "up-text" : "down-text"}">${up ? "看漲" : "看跌"}</td><td>${pct(item.probability_up)}</td><td>${pct(item.probability_down)}</td><td>${pct(item.confidence)}</td><td>${esc(item.news_label || "--")}</td><td>${esc(item.related_label || "--")}</td><td>${esc(item.institutional_label || "--")}</td></tr>`;
    }).join("") || `<tr><td colspan="8">沒有批次結果。</td></tr>`;
  } catch (error) {
    el.batchRows.innerHTML = `<tr><td colspan="8">批次失敗：${esc(error.message)}</td></tr>`;
  }
}

function updateClock() {
  const now = new Date();
  el.clockDate.textContent = now.toLocaleDateString("zh-TW", { year: "numeric", month: "2-digit", day: "2-digit", weekday: "short" });
  el.clockTime.textContent = now.toLocaleTimeString("zh-TW", { hour12: false });
}

el.form.addEventListener("submit", analyze);
el.search.addEventListener("input", () => {
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(searchSymbols, 220);
});
el.suggestions.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-symbol]");
  if (!button) return;
  selectSymbol(button.dataset.symbol, button.dataset.name || "");
  analyze();
});
el.recommendButton.addEventListener("click", recommend);
el.batchButton.addEventListener("click", runBatch);
el.chartDays.addEventListener("change", drawCandles);
window.addEventListener("resize", drawCandles);

el.canvas.addEventListener("mousemove", (event) => {
  const data = state.prices.slice(-Number(el.chartDays.value || 180));
  if (!data.length) return;
  const rect = el.canvas.getBoundingClientRect();
  const innerW = rect.width - 72;
  const x = Math.max(0, Math.min(innerW, event.clientX - rect.left - 52));
  state.hoverIndex = Math.round(x / innerW * (data.length - 1));
  drawCandles();
});
el.canvas.addEventListener("mouseleave", () => { state.hoverIndex = null; drawCandles(); });

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach((item) => item.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab)?.classList.add("active");
  });
});

updateClock();
setInterval(updateClock, 1000);
selectSymbol("2408.TW", "南亞科");
searchSymbols();
analyze();

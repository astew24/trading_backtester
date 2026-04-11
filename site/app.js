const metricDefinitions = [
  { key: "Total Return", label: "Total Return", format: "percent" },
  { key: "Benchmark Return", label: "Buy and Hold", format: "percent" },
  { key: "Excess Return", label: "Active Return", format: "percent" },
  { key: "Sharpe Ratio", label: "Sharpe", format: "number" },
  { key: "Max Drawdown", label: "Max Drawdown", format: "percent" },
  { key: "Win Rate", label: "Win Rate", format: "percent" },
  { key: "Profit Factor", label: "Profit Factor", format: "number" },
  { key: "Total Trades", label: "Trades", format: "integer" },
];

const state = {
  payload: null,
  activeScenario: 0,
};

function formatMetric(value, format) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }

  if (format === "percent") {
    return `${(value * 100).toFixed(1)}%`;
  }

  if (format === "integer") {
    return `${Math.round(value)}`;
  }

  return Number(value).toFixed(2);
}

function formatCurrency(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function metricClass(value, format) {
  if (format !== "percent" || value === null || value === undefined) {
    return "";
  }

  if (value > 0) {
    return "metric-positive";
  }

  if (value < 0) {
    return "metric-negative";
  }

  return "";
}

function buildScenarioButtons() {
  const container = document.getElementById("scenario-buttons");
  container.innerHTML = "";

  state.payload.scenarios.forEach((scenario, index) => {
    const button = document.createElement("button");
    button.className = `scenario-button${index === state.activeScenario ? " active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <span class="label">${scenario.label}</span>
      <span class="meta">${scenario.ticker} · ${scenario.strategy_family.replace("_", " ")}</span>
    `;
    button.addEventListener("click", () => {
      state.activeScenario = index;
      buildScenarioButtons();
      renderScenario();
    });
    container.appendChild(button);
  });
}

function buildChips(scenario) {
  const container = document.getElementById("scenario-chips");
  const chips = [
    scenario.ticker,
    `${scenario.period.bars} bars`,
    `${scenario.period.start} to ${scenario.period.end}`,
    scenario.parameters.allow_short ? "Shorts enabled" : "Long only",
  ];

  container.innerHTML = chips.map((chip) => `<span class="chip">${chip}</span>`).join("");
}

function buildMetrics(scenario) {
  const container = document.getElementById("metrics-grid");
  container.innerHTML = metricDefinitions
    .map((definition) => {
      const value = scenario.metrics[definition.key];
      return `
        <div class="metric-card">
          <div class="metric-label">${definition.label}</div>
          <div class="metric-value ${metricClass(value, definition.format)}">
            ${formatMetric(value, definition.format)}
          </div>
        </div>
      `;
    })
    .join("");
}

function renderPriceChart(scenario) {
  const traces = [
    {
      x: scenario.price.dates,
      y: scenario.price.close,
      type: "scatter",
      mode: "lines",
      name: "Close",
      line: { color: "#102033", width: 2.6 },
    },
  ];

  scenario.price.overlays.forEach((overlay) => {
    traces.push({
      x: scenario.price.dates,
      y: overlay.values,
      type: "scatter",
      mode: "lines",
      name: overlay.label,
      line: { color: overlay.color, width: 1.8, dash: "dot" },
    });
  });

  traces.push(
    {
      x: scenario.price.entries.map((item) => item.date),
      y: scenario.price.entries.map((item) => item.price),
      type: "scatter",
      mode: "markers",
      name: "Entry",
      marker: { color: "#1F7A5B", size: 8, symbol: "triangle-up" },
    },
    {
      x: scenario.price.exits.map((item) => item.date),
      y: scenario.price.exits.map((item) => item.price),
      type: "scatter",
      mode: "markers",
      name: "Exit",
      marker: { color: "#B3412D", size: 8, symbol: "triangle-down" },
    },
  );

  Plotly.react(
    "price-chart",
    traces,
    {
      margin: { l: 48, r: 20, t: 10, b: 42 },
      legend: { orientation: "h", y: -0.2 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: { gridcolor: "rgba(16,32,51,0.08)" },
      yaxis: { gridcolor: "rgba(16,32,51,0.08)" },
    },
    { displayModeBar: false, responsive: true },
  );
}

function renderEquityChart(scenario) {
  Plotly.react(
    "equity-chart",
    [
      {
        x: scenario.equity.dates,
        y: scenario.equity.strategy,
        type: "scatter",
        mode: "lines",
        name: "Strategy",
        line: { color: "#102033", width: 2.8 },
      },
      {
        x: scenario.equity.dates,
        y: scenario.equity.benchmark,
        type: "scatter",
        mode: "lines",
        name: "Buy and Hold",
        line: { color: "#5D6A78", width: 2, dash: "dash" },
      },
    ],
    {
      margin: { l: 56, r: 20, t: 10, b: 42 },
      legend: { orientation: "h", y: -0.2 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: { gridcolor: "rgba(16,32,51,0.08)" },
      yaxis: { gridcolor: "rgba(16,32,51,0.08)" },
    },
    { displayModeBar: false, responsive: true },
  );
}

function renderBullets(scenario) {
  const list = document.getElementById("scenario-bullets");
  list.innerHTML = scenario.bullets.map((bullet) => `<li>${bullet}</li>`).join("");
}

function renderTrades(scenario) {
  const container = document.getElementById("trade-table");

  if (!scenario.recent_trades.length) {
    container.innerHTML = "<p>No completed trades for this run.</p>";
    return;
  }

  const rows = scenario.recent_trades
    .map(
      (trade) => `
        <tr>
          <td>${trade.entry_date}</td>
          <td>${trade.exit_date}</td>
          <td>${trade.side}</td>
          <td>${formatCurrency(trade.net_pnl)}</td>
          <td>${formatMetric(trade.return_pct, "percent")}</td>
          <td>${trade.exit_reason}</td>
        </tr>
      `,
    )
    .join("");

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Entry</th>
          <th>Exit</th>
          <th>Side</th>
          <th>Net PnL</th>
          <th>Return</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderScenario() {
  const scenario = state.payload.scenarios[state.activeScenario];
  document.getElementById("scenario-strategy").textContent = scenario.strategy_family.replace(
    "_",
    " ",
  );
  document.getElementById("scenario-title").textContent = scenario.label;
  document.getElementById("scenario-description").textContent = scenario.description;

  buildChips(scenario);
  buildMetrics(scenario);
  renderBullets(scenario);
  renderTrades(scenario);
  renderPriceChart(scenario);
  renderEquityChart(scenario);
}

async function init() {
  const response = await fetch("./data/demo-data.json");
  const payload = await response.json();
  state.payload = payload;

  document.getElementById(
    "generated-at",
  ).textContent = `Illustrative runs generated on ${payload.generated_at} from live Yahoo Finance daily data using the repository's current engine.`;

  buildScenarioButtons();
  renderScenario();
}

init().catch((error) => {
  document.getElementById("generated-at").textContent =
    "Unable to load demo data. Open the repository for the local Streamlit app instead.";
  console.error(error);
});

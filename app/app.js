// Client-side inference that reproduces the trained scikit-learn Ridge pipeline
// (preprocess.py + train.py) exactly, using coefficients exported to
// model_coefficients.json. No server/API calls, no secrets - safe for static
// Netlify hosting.

let coeffs = null;

async function loadModel() {
  const res = await fetch("model_coefficients.json");
  coeffs = await res.json();

  const industrySel = document.getElementById("industry");
  coeffs.industry_categories.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c;
    industrySel.appendChild(opt);
  });
  industrySel.value = "Technology";

  const regionSel = document.getElementById("region");
  coeffs.region_categories.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c;
    regionSel.appendChild(opt);
  });
  regionSel.value = "North America";
}

function syncSliders() {
  const env = document.getElementById("env");
  const soc = document.getElementById("soc");
  const gov = document.getElementById("gov");
  const ovr = document.getElementById("ovr");

  function refresh() {
    document.getElementById("env-out").textContent = env.value;
    document.getElementById("soc-out").textContent = soc.value;
    document.getElementById("gov-out").textContent = gov.value;
    const overall = (Number(env.value) + Number(soc.value) + Number(gov.value)) / 3;
    ovr.value = overall.toFixed(0);
    document.getElementById("ovr-out").textContent = overall.toFixed(1);
  }
  [env, soc, gov].forEach((el) => el.addEventListener("input", refresh));
  refresh();
}

function log1p(x) { return Math.log(1 + x); }

function predict() {
  if (!coeffs) return null;

  const raw = {
    Revenue: Number(document.getElementById("revenue").value),
    MarketCap: Number(document.getElementById("marketcap").value),
    CarbonEmissions: Number(document.getElementById("carbon").value),
    WaterUsage: Number(document.getElementById("water").value),
    EnergyConsumption: Number(document.getElementById("energy").value),
    Year: Number(document.getElementById("year").value),
    GrowthRate: Number(document.getElementById("growth").value),
    ESG_Overall: Number(document.getElementById("ovr").value),
    ESG_Environmental: Number(document.getElementById("env").value),
    ESG_Social: Number(document.getElementById("soc").value),
    ESG_Governance: Number(document.getElementById("gov").value),
    Industry: document.getElementById("industry").value,
    Region: document.getElementById("region").value,
  };

  let z = coeffs.intercept;

  // log1p + standard-scaled skewed columns
  coeffs.log_skewed_cols.forEach((col, i) => {
    const logged = log1p(raw[col]);
    const scaled = (logged - coeffs.log_skewed_mean[i]) / coeffs.log_skewed_scale[i];
    const key = `log_skewed__${col}`;
    if (coeffs.coefficients[key] !== undefined) z += coeffs.coefficients[key] * scaled;
  });

  // standard-scaled remaining numeric columns
  coeffs.scale_rest_cols.forEach((col, i) => {
    const scaled = (raw[col] - coeffs.scale_rest_mean[i]) / coeffs.scale_rest_scale[i];
    const key = `scale_rest__${col}`;
    if (coeffs.coefficients[key] !== undefined) z += coeffs.coefficients[key] * scaled;
  });

  // one-hot Industry
  coeffs.industry_categories.forEach((cat) => {
    const key = `onehot__Industry_${cat}`;
    if (coeffs.coefficients[key] !== undefined) {
      z += coeffs.coefficients[key] * (raw.Industry === cat ? 1 : 0);
    }
  });

  // one-hot Region
  coeffs.region_categories.forEach((cat) => {
    const key = `onehot__Region_${cat}`;
    if (coeffs.coefficients[key] !== undefined) {
      z += coeffs.coefficients[key] * (raw.Region === cat ? 1 : 0);
    }
  });

  return z;
}

function runEstimate() {
  const val = predict();
  const out = document.getElementById("result-value");
  if (val === null) { out.textContent = "—"; return; }
  out.textContent = val.toFixed(1);
}

document.addEventListener("DOMContentLoaded", async () => {
  syncSliders();
  await loadModel();
  runEstimate();
  document.getElementById("estimate-btn").addEventListener("click", runEstimate);
});

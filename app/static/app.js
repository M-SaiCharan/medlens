const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const rupees = (n) => "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2 });

const store = {
  get() { try { return JSON.parse(localStorage.getItem("medlens.mine") || "[]"); } catch { return []; } },
  set(v) { try { localStorage.setItem("medlens.mine", JSON.stringify(v)); } catch {} },
};
let mine = store.get();

function status(msg) { $("#status").textContent = msg; }

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

function renderMatches(matches) {
  $("#matches").innerHTML = matches.length
    ? matches.map((m, i) => `
      <li>
        <div><h3>${esc(m.name)}</h3>
          <div class="meta">${esc(m.composition)} · ${esc(m.pack)} · ${rupees(m.price)}${i === 0 ? " · best match" : ""}</div></div>
        <button data-add="${m.id}" data-name="${esc(m.name)}">Add</button>
      </li>`).join("")
    : `<li class="muted">No match found. Try the brand name as printed on the strip.</li>`;
}

function renderMine() {
  $("#mine").innerHTML = mine.length
    ? mine.map((m) => `<li><span>${esc(m.name)}</span><button data-remove="${m.id}" aria-label="Remove ${esc(m.name)}">✕</button></li>`).join("")
    : `<li class="muted">Add the medicines you take to check them together.</li>`;
  $("#check").disabled = mine.length === 0;
  store.set(mine);
}

function renderReport(r) {
  const s = r.potential_savings;
  const ix = r.interactions;
  let html = "";
  if (s.pct > 0) {
    html += `<div class="savings">Switching to the cheapest equivalent brands could save
      <strong>${s.pct}% · ${rupees(s.current - s.cheapest)} per pack set</strong>
      ${rupees(s.current)} → ${rupees(s.cheapest)}</div>`;
  }
  html += `<div class="card"><h2>Interactions</h2>`;
  html += ix.length
    ? ix.map((i) => `<div class="ix ${esc(i.severity)}"><div class="sev">${esc(i.severity)}</div>
        <div><b>${esc(i.salt_a)}</b> (${esc(i.medicine_a)}) + <b>${esc(i.salt_b)}</b> (${esc(i.medicine_b)})</div>
        <div class="meta">${esc(i.advice)}</div></div>`).join("")
    : `<p class="muted">${mine.length < 2 ? "Add at least two medicines to check interactions." : "No known interactions between these medicines."}</p>`;
  if (r.unchecked_salts.length) {
    html += `<div class="warn">⚠️ No interaction data for: <b>${r.unchecked_salts.map(esc).join(", ")}</b>.
      This does <u>not</u> mean they are safe together. Ask your pharmacist.</div>`;
  }
  html += `</div>`;
  for (const m of r.medicines) {
    html += `<div class="card"><h3>${esc(m.name)}</h3>
      <div class="meta">${esc(m.composition)} · ${esc(m.manufacturer)} · ${esc(m.pack)} · ${rupees(m.price)}</div>`;
    if (m.market) {
      html += `<p class="meta">${m.market.brands} brands sell this exact composition as ${esc(m.form)}.
        Per unit: ${rupees(m.market.min_per_unit)} – ${rupees(m.market.max_per_unit)} (median ${rupees(m.market.median_per_unit)}); yours: ${rupees(m.price_per_unit)}.</p>`;
    }
    html += m.alternatives.length
      ? `<div class="table-scroll"><table><thead><tr><th>Same medicine, cheaper</th><th>Pack</th><th class="num">Price</th><th class="num">Saves</th></tr></thead><tbody>
          ${m.alternatives.map((a) => `<tr><td>${esc(a.name)}<div class="meta">${esc(a.manufacturer)}</div></td><td>${esc(a.pack)}</td>
            <td class="num">${rupees(a.price)}</td><td class="num pct">${a.savings_pct}%</td></tr>`).join("")}
        </tbody></table></div>`
      : `<p class="muted">This is already among the cheapest options for this composition.</p>`;
    html += `</div>`;
  }
  $("#report").innerHTML = html;
  $("#report").scrollIntoView({ behavior: "smooth" });
}

$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#ocr-box").hidden = true;
  status("Searching…");
  try {
    const r = await api(`/api/search?q=${encodeURIComponent($("#q").value)}`);
    renderMatches(r.matches); status("");
  } catch (err) { status(`Search failed: ${err.message}`); }
});

$("#photo").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  status("Reading the photo…");
  const body = new FormData();
  body.append("image", file);
  try {
    const r = await api("/api/scan", { method: "POST", body });
    $("#ocr-text").textContent = r.ocr_text || "(no text found)";
    $("#ocr-box").hidden = false;
    renderMatches(r.matches);
    status(r.matches.length ? "Pick the medicine that matches your strip." : "Couldn't read a medicine name. Try a sharper, well-lit photo.");
  } catch (err) { status(`Scan failed: ${err.message}`); }
  e.target.value = "";
});

document.addEventListener("click", (e) => {
  const add = e.target.closest("[data-add]");
  const rm = e.target.closest("[data-remove]");
  if (add) {
    const id = Number(add.dataset.add);
    if (!mine.some((m) => m.id === id)) mine.push({ id, name: add.dataset.name });
    renderMine(); status(`Added ${add.dataset.name}.`);
  } else if (rm) {
    mine = mine.filter((m) => m.id !== Number(rm.dataset.remove));
    renderMine();
  }
});

$("#check").addEventListener("click", async () => {
  status("Checking…");
  try {
    renderReport(await api("/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: mine.map((m) => m.id) }) }));
    status("");
  } catch (err) { status(`Check failed: ${err.message}`); }
});

renderMine();

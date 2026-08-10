/* Relevant Section Identification — UI */
pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

const $ = (s) => document.querySelector(s);
const state = { docId: null, pdf: null, page: 1, scale: 1.25, tab: "find",
                rendering: false, pending: null, poll: null };

/* ------------------------------------------------------------------ notices */
function notice(msg, bad) {
  const el = $("#notice");
  if (!msg) { el.hidden = true; return; }
  el.textContent = msg;
  el.className = "notice" + (bad ? " bad" : "");
  el.hidden = false;
}

/* ------------------------------------------------------------------- viewer */
async function loadPdf(docId) {
  $("#viewerEmpty").hidden = false;
  $("#viewerEmpty").textContent = "Loading document…";
  try {
    state.pdf = await pdfjsLib.getDocument(`/api/pdf/${docId}`).promise;
    state.rendering = false;
    state.pending = null;
    $("#pageCount").textContent = state.pdf.numPages;
    $("#pageInput").max = state.pdf.numPages;
    // The document is open; don't keep the overlay up waiting for the raster.
    $("#viewerEmpty").hidden = true;
    showPage(1);
  } catch (e) {
    $("#viewerEmpty").textContent = "Could not load this PDF.";
  }
}

async function showPage(n) {
  if (!state.pdf) return;
  n = Math.min(Math.max(1, n), state.pdf.numPages);
  state.page = n;
  $("#pageInput").value = n;
  if (state.rendering) { state.pending = n; return; }

  state.rendering = true;
  try {
    const page = await state.pdf.getPage(n);
    const viewport = page.getViewport({ scale: state.scale });
    const canvas = $("#pdf");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(viewport.width * dpr);
    canvas.height = Math.floor(viewport.height * dpr);
    canvas.style.width = viewport.width + "px";
    canvas.style.height = viewport.height + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    await page.render({ canvasContext: ctx, viewport }).promise;
  } catch (err) {
    // A failed or cancelled raster must not wedge the viewer: without this the
    // flag stays true and every later page change queues forever.
    console.warn("page render failed", err);
  } finally {
    state.rendering = false;
  }

  if (state.pending !== null && state.pending !== n) {
    const next = state.pending; state.pending = null; showPage(next);
  } else { state.pending = null; }
}

function jumpTo(page, cardEl) {
  showPage(page);
  document.querySelectorAll(".hit.active").forEach((e) => e.classList.remove("active"));
  if (cardEl) cardEl.classList.add("active");
  $("#canvasWrap").scrollTop = 0;
}

/* ---------------------------------------------------------------- documents */
async function loadDocs(selectId) {
  const docs = await (await fetch("/api/documents")).json();
  const sel = $("#doc");
  sel.innerHTML = "";
  if (!docs.length) {
    notice("No documents registered. Upload a PDF to begin.", true);
    return;
  }
  docs.forEach((d) => {
    const o = document.createElement("option");
    o.value = d.doc_id;
    o.textContent = `${d.name}${d.pages ? ` — ${d.pages}p` : ""}` +
                    (d.status !== "ready" ? `  [${d.status}]` : "");
    sel.appendChild(o);
  });
  const target = selectId && docs.some((d) => d.doc_id === selectId)
    ? selectId : (state.docId && docs.some((d) => d.doc_id === state.docId)
      ? state.docId : docs[0].doc_id);
  sel.value = target;
  await selectDoc(target);
}

async function selectDoc(docId) {
  state.docId = docId;
  $("#results").innerHTML = "";
  const d = await (await fetch(`/api/documents/${docId}`)).json();
  if (d.status === "indexing") {
    notice(`Indexing “${d.name}” — ${d.progress || "working"}. ` +
           `Parsing, captioning figures and embedding take a few minutes.`);
    startPolling(docId);
  } else if (d.status === "error") {
    notice(`Indexing failed: ${d.message}`, true);
  } else {
    notice(null);
    stopPolling();
  }
  await loadPdf(docId);
}

function startPolling(docId) {
  stopPolling();
  state.poll = setInterval(async () => {
    const d = await (await fetch(`/api/documents/${docId}`)).json();
    if (d.status === "indexing") {
      notice(`Indexing “${d.name}” — ${d.progress || "working"}…`);
    } else if (d.status === "error") {
      notice(`Indexing failed: ${d.message}`, true); stopPolling();
    } else {
      notice(`“${d.name}” is ready — ${d.chunks} chunks indexed.`);
      stopPolling(); loadDocs(docId);
      setTimeout(() => notice(null), 4000);
    }
  }, 3000);
}
function stopPolling() { if (state.poll) { clearInterval(state.poll); state.poll = null; } }

/* ------------------------------------------------------------------ queries */
function renderFind(data) {
  const box = $("#results");
  box.innerHTML = "";
  if (!data.results.length) {
    box.innerHTML = `<p class="muted">No page scored high enough to be called relevant.</p>`;
    return;
  }
  const head = document.createElement("p");
  head.className = "hitcount";
  head.textContent = `${data.results.length} relevant page${data.results.length > 1 ? "s" : ""}`;
  box.appendChild(head);

  data.results.forEach((r) => {
    const card = document.createElement("article");
    card.className = "hit";
    card.tabIndex = 0;
    const sections = r.sections.length ? r.sections.join(" · ") : "—";
    card.innerHTML = `
      <div class="hit-head">
        <span class="pagechip">p. ${r.page}</span>
        <span class="sections"></span>
        <span class="score">${r.score.toFixed(3)}</span>
      </div>
      <div class="hit-body"></div>`;
    card.querySelector(".sections").textContent = sections;
    const body = card.querySelector(".hit-body");
    r.evidence.forEach((e) => {
      const row = document.createElement("div");
      row.className = "ev";
      row.innerHTML = `<span class="t ${e.type}">${e.type.slice(0, 3).toUpperCase()}</span>
                       <span class="snip"></span>`;
      row.querySelector(".snip").textContent = e.snippet;
      body.appendChild(row);
    });
    const go = () => jumpTo(r.page, card);
    card.addEventListener("click", go);
    card.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go(); }
    });
    box.appendChild(card);
  });
  jumpTo(data.results[0].page, box.querySelector(".hit"));
}

function renderAnswer(data) {
  const box = $("#results");
  box.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "answer";
  const p = document.createElement("p");
  p.textContent = data.answer;
  wrap.appendChild(p);
  if (data.citations && data.citations.length) {
    const cites = document.createElement("div");
    cites.className = "cites";
    data.citations.forEach((c) => {
      const b = document.createElement("button");
      b.className = "cite";
      b.textContent = `p. ${c.page}${c.title ? " · " + c.title : ""}`;
      b.addEventListener("click", () => jumpTo(c.page, null));
      cites.appendChild(b);
    });
    wrap.appendChild(cites);
  }
  box.appendChild(wrap);
  const note = document.createElement("p");
  note.className = "muted";
  note.textContent = "Answered by a small local model from the cited pages — " +
                     "check them before relying on it.";
  box.appendChild(note);
}

async function run(ev) {
  ev.preventDefault();
  const q = $("#q").value.trim();
  if (!q || !state.docId) return;
  const btn = $("#go");
  btn.disabled = true;
  const isAsk = state.tab === "ask";
  $("#results").innerHTML =
    `<p class="muted spin">${isAsk ? "Reading the retrieved pages" : "Searching"}</p>`;
  try {
    const res = await fetch(isAsk ? "/api/chat" : "/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: state.docId, query: q }),
    });
    const data = await res.json();
    if (!res.ok) {
      $("#results").innerHTML = `<p class="muted">${data.detail || "Request failed."}</p>`;
    } else if (isAsk) { renderAnswer(data); } else { renderFind(data); }
  } catch (e) {
    $("#results").innerHTML = `<p class="muted">${e.message}</p>`;
  } finally {
    btn.disabled = false;
  }
}

/* -------------------------------------------------------------------- wiring */
$("#form").addEventListener("submit", run);
$("#q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(e); }
});
$("#doc").addEventListener("change", (e) => selectDoc(e.target.value));
$("#prev").addEventListener("click", () => showPage(state.page - 1));
$("#next").addEventListener("click", () => showPage(state.page + 1));
$("#pageInput").addEventListener("change", (e) => showPage(parseInt(e.target.value, 10) || 1));
$("#zoomIn").addEventListener("click", () => { state.scale = Math.min(3, state.scale + 0.25); showPage(state.page); });
$("#zoomOut").addEventListener("click", () => { state.scale = Math.max(0.5, state.scale - 0.25); showPage(state.page); });

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => {
      x.classList.remove("active"); x.setAttribute("aria-selected", "false");
    });
    t.classList.add("active"); t.setAttribute("aria-selected", "true");
    state.tab = t.dataset.tab;
    $("#go").textContent = state.tab === "ask" ? "Ask" : "Search";
    $("#results").innerHTML = "";
  });
});

$("#examples").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") { $("#q").value = e.target.textContent; run(new Event("submit")); }
});

$("#uploadBtn").addEventListener("click", () => $("#file").click());
$("#file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  notice(`Uploading “${f.name}”…`);
  const fd = new FormData();
  fd.append("file", f);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { notice(data.detail || "Upload failed.", true); return; }
    await loadDocs(data.doc_id);
  } catch (err) { notice(err.message, true); }
  e.target.value = "";
});

loadDocs();

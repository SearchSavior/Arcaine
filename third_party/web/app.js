// Arcaine SemIf server-backed decision lab.
(() => {
  const $ = (id) => document.getElementById(id);

  const MAX_OPTIONS = 255;
  const MIN_OPTIONS = 2;
  const optionList = $("option-list");

  let model = null;
  let qtype = "choice";
  let ready = false;

  function setSupport(text, kind = "") {
    const p = document.querySelector("#support .support");
    p.className = "support " + kind;
    $("support-text").textContent = text;
  }

  function seconds(ms) { return `${(ms / 1000).toFixed(3)} s`; }
  function dash(id) { $(id).textContent = "—"; }

  function headers(json) {
    const h = { "content-type": "application/json" };
    const key = $("key").value.trim();
    if (key) h["authorization"] = "Bearer " + key;
    return h;
  }

  async function api(path, body) {
    const res = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: headers(),
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let payload = null;
    try { payload = await res.json(); } catch (_) { /* no JSON body */ }
    if (!res.ok) {
      const msg = payload && payload.error && payload.error.message
        ? payload.error.message : `HTTP ${res.status}`;
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    return payload;
  }

  const rows = () => [...optionList.querySelectorAll(".option-row")];
  function syncControls() {
    rows().forEach((row, i) => { row.querySelector("b").textContent = optionLetter(i); });
    $("option-count").textContent = `${rows().length} / ${MAX_OPTIONS}`;
    $("remove-option").disabled = rows().length <= MIN_OPTIONS;
    $("add-option").disabled = rows().length >= MAX_OPTIONS;
  }
  function appendOption(value = "") {
    if (rows().length >= MAX_OPTIONS) return null;
    const row = document.createElement("label");
    row.className = "option-row";
    const b = document.createElement("b");
    const input = document.createElement("input");
    input.className = "option";
    input.value = value;
    input.placeholder = "Describe this option";
    row.append(b, input);
    optionList.append(row);
    syncControls();
    return input;
  }
  function setOptions(values) {
    while (rows().length > values.length) rows().at(-1).remove();
    while (rows().length < values.length) appendOption();
    rows().forEach((row, i) => { row.querySelector(".option").value = values[i]; });
    syncControls();
  }
  $("add-option").addEventListener("click", () => appendOption()?.focus());
  $("remove-option").addEventListener("click", () => {
    if (rows().length > MIN_OPTIONS) rows().at(-1).remove();
    syncControls();
  });

  const TYPE_NOTE = {
    choice: "Every option becomes one answer slot the model scores against the full vocabulary; probabilities are normalized only across the options you supplied.",
    noul: "A yes/no statement: the model returns the probability of yes. No confidence field is reported for noul.",
    score: "Ordered level descriptions (lowest first). The answer is the probability-weighted level index; probabilities and the legend are keyed by level.",
  };
  document.querySelectorAll("[data-qtype]").forEach((b) => {
    b.addEventListener("click", () => {
      qtype = b.dataset.qtype;
      document.querySelectorAll("[data-qtype]").forEach((x) => x.classList.toggle("selected", x === b));
      $("type-note").textContent = TYPE_NOTE[qtype];
      if (qtype === "choice" && rows().length < 2) setOptions(["Option A", "Option B"]);
    });
  });

  const presets = {
    account: {
      type: "choice",
      state: "A customer says a password reset succeeded, but every login attempt still returns ‘account locked’. Two unlock emails were requested and neither arrived.",
      question: "Which queue should handle this request?",
      options: ["Account access support", "Billing support", "Close as resolved"],
    },
    email: {
      type: "choice",
      state: "An email claims to be from the payroll team and says the recipient’s salary payment will be suspended today. It comes from payroll-review@outlook.com and links to a non-company sign-in page asking for a password and verification code.",
      question: "How should this email be classified?",
      options: ["Legitimate", "Spam", "Phishing"],
    },
    incident: {
      type: "noul",
      state: "Everything is down and we have a demo at noon.",
      question: "Does this require a response within an hour?",
      options: ["A service outage needs prompt action", "The request can wait"],
    },
  };
  document.querySelectorAll("[data-preset]").forEach((b) => {
    b.addEventListener("click", () => {
      const p = presets[b.dataset.preset];
      if (!p) return;
      document.querySelector(`[data-qtype="${p.type}"]`).click();
      $("state").value = p.state;
      $("question").value = p.question;
      setOptions(p.options);
      $("state").focus();
    });
  });

  async function connect() {
    const btn = $("load");
    btn.disabled = true;
    btn.innerHTML = '<span class="icon spin" aria-hidden="true">&#8635;</span> connecting…';
    setSupport("Checking the server…");
    try {
      const models = await api("/v1/models");
      model = models.data[0].id;
      $("model-value").textContent = model;
      $("selected-model").textContent = model;
      $("model-detail").textContent = "DiffusionGemma backend, weights on the server";
      $("model-size").textContent = `${model} on the server`;

      $("policy-value").textContent = "server-configured";
      ready = true;
      $("run").disabled = false;
      $("host-note").textContent = `served from ${location.host}`;
      setSupport(`Connected. ${model} answers on this server — no WebGPU needed.`, "ok");
      btn.innerHTML = '<span class="icon" aria-hidden="true">&#10003;</span> connected';
    } catch (e) {
      if (e.status === 401) {
        setSupport("This server requires a bearer token. Enter the API key and connect again.", "error");
      } else {
        setSupport(`Cannot reach this server's API: ${e.message}`, "error");
      }
      btn.innerHTML = '<span class="icon" aria-hidden="true">&#8635;</span> retry connect';
      btn.disabled = false;
    }
  }
  $("load").addEventListener("click", connect);
  $("key").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });

  function buildQuestion() {
    const options = rows().map((r) => r.querySelector(".option").value.trim());
    const question = $("question").value.trim();
    const state = $("state").value.trim();
    if (!state) return { error: "State must be nonempty." };
    if (qtype === "noul") {
      return { state, questions: { q: { type: "noul", instructions: question || undefined } } };
    }
    if (options.some((o) => !o)) return { error: "Every option must be nonempty." };
    if (options.length < MIN_OPTIONS) return { error: "At least two options are required." };
    if (qtype === "choice") {
      const criteria = {};
      options.forEach((o) => { criteria[o] = null; });
      return { state, questions: { q: { type: "choice", instructions: question || undefined, criteria } } };
    }
    // Score options keep their order.
    return { state, questions: { q: { type: "score", instructions: question || undefined, criteria: options } } };
  }

  function optionLetter(i) {
    // Use A..Z, then AA..ZZ.
    let s = "";
    let n = i;
    do { s = String.fromCharCode(65 + (n % 26)) + s; n = Math.floor(n / 26) - 1; } while (n >= 0);
    return s;
  }

  function renderChoice(probabilities, choice, confidence) {
    const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const max = Math.max(...entries.map(([, p]) => p), 1e-9);
    const out = $("direct-output");
    out.classList.remove("empty");
    out.replaceChildren(...entries.map(([name, p], i) => {
      const row = document.createElement("div");
      row.className = "choice";
      const label = document.createElement("span");
      label.className = "choice-label";
      const b = document.createElement("b");
      b.textContent = optionLetter(i);
      const small = document.createElement("small");
      small.textContent = name === choice ? name + " ←" : name;
      if (name === choice) small.style.fontWeight = "900";
      label.append(b, small);
      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("i");
      fill.style.width = `${Math.max(1, (p / max) * 100)}%`;
      bar.append(fill);
      const score = document.createElement("em");
      score.textContent = p.toFixed(3);
      row.append(label, bar, score);
      return row;
    }));
    $("t-conf").textContent = confidence == null ? "—" : confidence.toFixed(3);
  }

  function renderScore(legend, probabilities, score, confidence) {
    const out = $("direct-output");
    out.classList.remove("empty");
    const levels = Object.keys(probabilities).map(Number).sort((a, b) => a - b);
    const max = Math.max(...levels.map((k) => probabilities[k]), 1e-9);
    const describe = (v) => {
      if (v == null) return "";
      if (typeof v === "string") return v;
      return JSON.stringify(v);
    };
    out.replaceChildren(...levels.map((k) => {
      const row = document.createElement("div");
      row.className = "choice";
      const label = document.createElement("span");
      label.className = "choice-label";
      const b = document.createElement("b");
      b.textContent = String(k);
      const small = document.createElement("small");
      small.textContent = describe(legend[k]);
      label.append(b, small);
      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("i");
      fill.style.width = `${Math.max(1, (probabilities[k] / max) * 100)}%`;
      bar.append(fill);
      const em = document.createElement("em");
      em.textContent = probabilities[k].toFixed(3);
      row.append(label, bar, em);
      return row;
    }));
    const head = document.createElement("div");
    head.className = "scoregrid";
    head.innerHTML = `<small>expected score</small><strong>${score.toFixed(3)}</strong><span></span>`;
    out.prepend(head);
    $("t-conf").textContent = confidence == null ? "—" : confidence.toFixed(3);
  }

  function resetResults() {
    $("direct-output").textContent = "asking the server…";
    $("direct-output").className = "output empty";
    $("timing-output").textContent = "waiting for a run";
    $("timing-output").className = "output empty";
    ["direct-total", "direct-input", "direct-readouts", "t-prefill", "t-decode", "t-conf"].forEach(dash);
    $("ratio").textContent = "measuring…";
  }

  $("run").addEventListener("click", async () => {
    const req = buildQuestion();
    if (req.error) { setSupport(req.error, "error"); return; }
    req.model = model;
    req.arcaine = { diagnostics: true };

    resetResults();
    const btn = $("run");
    btn.disabled = true;
    btn.innerHTML = '<span class="icon spin" aria-hidden="true">&#9654;</span> running…';
    setSupport("Asking the server for one readout…");

    const t0 = performance.now();
    try {
      const res = await api("/v1/systemone", req);
      const wall = performance.now() - t0;
      const a = res.answers && res.answers.q;
      if (!a) throw new Error("response had no answer for the question");

      $("direct-total").textContent = seconds(wall);
      $("direct-input").textContent = `${res.usage.input_tokens} tok`;
      const reads = (res.arcaine && res.arcaine.diagnostics && res.arcaine.diagnostics.q)
        ? res.arcaine.diagnostics.q.reads : res.usage.output_tokens;
      $("direct-readouts").textContent = `${reads} read${reads === 1 ? "" : "s"}`;

      if (a.type === "choice") renderChoice(a.probabilities, a.choice, a.confidence);
      else if (a.type === "noul") {
        const out = $("direct-output");
        out.classList.remove("empty");
        out.replaceChildren();
        const p = a.noul;
        const row = document.createElement("div");
        row.className = "choice";
        row.innerHTML = `<span class="choice-label"><b>yes</b><small>probability of yes</small></span>` +
          `<span class="bar"><i style="width:${Math.max(1, p * 100)}%"></i></span><em>${p.toFixed(3)}</em>`;
        const row2 = document.createElement("div");
        row2.className = "choice";
        row2.innerHTML = `<span class="choice-label"><b>no</b><small>probability of no</small></span>` +
          `<span class="bar"><i style="width:${Math.max(1, (1 - p) * 100)}%"></i></span><em>${(1 - p).toFixed(3)}</em>`;
        out.append(row, row2);
        $("t-conf").textContent = "—";
      } else if (a.type === "score") renderScore(a.legend, a.probabilities, a.score, a.confidence);

      const d = res.arcaine && res.arcaine.diagnostics && res.arcaine.diagnostics.q;
      if (d) {
        $("t-prefill").textContent = `${d.prefill_ms.toFixed(1)} ms`;
        $("t-decode").textContent = `${d.decode_ms.toFixed(1)} ms`;
        $("timing-output").classList.remove("empty");
        $("timing-output").textContent =
          `reads=${d.reads} canvas=${d.canvas_width} tok prompt=${d.prompt_tokens} tok\n` +
          `prefill passes=${d.prefill_calls} decode passes=${d.decode_calls}\n` +
          `label mass=${Array.isArray(d.label_mass) ? d.label_mass.map((x) => x.toFixed(3)).join(", ") : "—"}\n` +
          `argmax is a label: ${Array.isArray(d.argmax_is_label) ? d.argmax_is_label.join(", ") : "—"}`;
      }
      $("last-value").textContent = seconds(wall);
      $("ratio").textContent = seconds(wall);
      $("run-note").textContent =
        `Wall time ${seconds(wall)} measured around the HTTP round-trip. ` +
        `Server-side prefill ${d ? d.prefill_ms.toFixed(1) : "—"} ms and decode ${d ? d.decode_ms.toFixed(1) : "—"} ms are synchronized server timers. ` +
        `Conditional probabilities; confidence is the server's normalized-entropy approximation.`;
      setSupport("Done. Edit the decision and run again whenever you like.", "ok");
    } catch (e) {
      setSupport(`Request failed: ${e.message}`, "error");
      $("direct-output").textContent = "the read did not complete";
      $("ratio").textContent = "—";
    } finally {
      btn.disabled = !ready;
      btn.innerHTML = '<span class="icon" aria-hidden="true">&#9654;</span> run decision';
    }
  });

  $("device-note").textContent =
    `Open this page from any device on the network at http://${location.host} — the browser only edits text and renders bars.`;
  connect();
})();

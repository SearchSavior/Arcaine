#include "viz.hpp"
#include "chat.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <stdexcept>

void DiffVizRecorder::record(const DiffStepEvent& ev) {
    Frame f;
    f.block = ev.block;
    f.cur_step = ev.cur_step;
    f.temperature = ev.temperature;
    f.mean_entropy = ev.mean_entropy;
    f.committed = ev.committed;
    f.ids = *ev.canvas;
    if (ev.entropy)  f.entropy  = *ev.entropy;
    if (ev.accepted) f.accepted = *ev.accepted;
    frames_.push_back(std::move(f));
}

void DiffVizRecorder::write_html(const std::string& path, const std::string& prompt,
                                 TokenizerBridge& tok, const DiffPerfStats& perf,
                                 const std::vector<DiffGpuMem>& gpu_mem) const {
    using nlohmann::json;

    // Per-token surface strings for every distinct id seen.
    std::set<int> id_set;
    for (auto& f : frames_) id_set.insert(f.ids.begin(), f.ids.end());
    std::vector<int> ids(id_set.begin(), id_set.end());
    std::vector<std::string> strs = tok.pieces(ids);

    json pieces = json::object();
    for (size_t i = 0; i < ids.size(); ++i)
        pieces[std::to_string(ids[i])] = strs[i];

    json jframes = json::array();
    for (auto& f : frames_) {
        json jf;
        jf["block"] = f.block;
        jf["step"] = f.cur_step;
        jf["temp"] = f.temperature;
        jf["H"] = f.mean_entropy;
        jf["committed"] = f.committed;
        jf["ids"] = f.ids;
        if (!f.entropy.empty())  jf["ent"] = f.entropy;
        if (!f.accepted.empty()) jf["acc"] = std::vector<int>(f.accepted.begin(), f.accepted.end());
        jframes.push_back(std::move(jf));
    }

    json data;
    data["prompt"] = prompt;
    data["frames"] = std::move(jframes);
    data["pieces"] = std::move(pieces);
    data["perf"] = {
        {"prefill_tokens",    perf.prefill_tokens},
        {"prefill_s",         perf.prefill_s},
        {"prefill_tps",       perf.prefill_tps()},
        {"decode_passes",     perf.decode_passes},
        {"decode_s",          perf.decode_s},
        {"decode_passes_ps",  perf.decode_passes_ps()},
        {"output_tokens",     perf.output_tokens},
        {"effective_tps",     perf.effective_tps()},
        {"tokens_per_forward", perf.tokens_per_forward()},
    };
    json jgpus = json::array();
    for (auto& g : gpu_mem)
        jgpus.push_back({{"used_gb", g.used_gb}, {"total_gb", g.total_gb}});
    data["gpus"] = std::move(jgpus);

    static const char* kHtml = R"HTML(<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DiffusionGemma denoising replay</title>
<style>
  body { background:#111; color:#ddd; font-family:system-ui,sans-serif; margin:0; padding:20px; }
  h2 { font-weight:500; margin:0 0 4px; }
  .meta { color:#888; font-size:13px; margin-bottom:14px; }
  #bar { display:flex; gap:12px; align-items:center; margin-bottom:14px; }
  #step { flex:1; }
  button { background:#2a2a2a; color:#ddd; border:1px solid #444; border-radius:6px;
           padding:6px 14px; cursor:pointer; font-size:14px; }
  button:hover { background:#3a3a3a; }
  #info { font-variant-numeric:tabular-nums; color:#aaa; font-size:13px; min-width:330px; }
  #canvas { font-family:ui-monospace,Menlo,monospace; font-size:14px; line-height:1.85;
            white-space:pre-wrap; word-break:break-word; background:#181818;
            border:1px solid #2c2c2c; border-radius:8px; padding:18px; max-width:980px; }
  .committed { color:#9ece6a; }
  .tok { border-radius:3px; }
  .acc { text-decoration:underline; text-decoration-color:#666; text-underline-offset:3px; }
  .legend { color:#777; font-size:12px; margin-top:10px; }
  .sw { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:-2px; }
  #perf { display:flex; gap:10px; flex-wrap:wrap; margin:0 0 14px; }
  .stat { background:#1c1c1c; border:1px solid #2c2c2c; border-radius:8px; padding:8px 14px; }
  .stat .v { font-size:18px; font-weight:600; color:#e0e0e0; font-variant-numeric:tabular-nums; }
  .stat .l { font-size:11px; color:#888; margin-top:2px; }
</style></head><body>
<h2>DiffusionGemma denoising replay</h2>
<div class="meta" id="prompt"></div>
<div id="perf"></div>
<div id="bar">
  <button id="play">&#9654; play</button>
  <input type="range" id="step" min="0" value="0">
  <span id="info"></span>
</div>
<div id="canvas"></div>
<div class="legend">
  <span class="sw" style="background:rgba(255,60,60,.85)"></span> high entropy (noise) &nbsp;
  <span class="sw" style="background:rgba(255,60,60,.25)"></span> uncertain &nbsp;
  <span class="sw" style="background:transparent;border:1px solid #444"></span> converged &nbsp;
  <u style="text-decoration-color:#666">underline</u> = accepted by sampler &nbsp;
  <span class="committed">green</span> = committed blocks
</div>
<script>
const DATA = __DATA__;
const frames = DATA.frames, pieces = DATA.pieces;
const slider = document.getElementById('step'), canvas = document.getElementById('canvas');
const info = document.getElementById('info'), play = document.getElementById('play');
document.getElementById('prompt').textContent = 'prompt: ' + DATA.prompt;
slider.max = frames.length - 1;

const P = DATA.perf;
const stats = [
  [P.prefill_tps.toFixed(1) + ' tok/s', `prefill (${P.prefill_tokens} tok / ${P.prefill_s.toFixed(2)} s)`],
  [P.effective_tps.toFixed(1) + ' tok/s', `decode, effective (${P.output_tokens} tok / ${P.decode_s.toFixed(2)} s)`],
  [P.decode_passes_ps.toFixed(2) + ' /s', `denoiser forward passes (${P.decode_passes} total)`],
  [P.tokens_per_forward.toFixed(1), 'tokens per forward'],
];
(DATA.gpus || []).forEach((g, i) => {
  const pct = g.total_gb > 0 ? (100 * g.used_gb / g.total_gb).toFixed(0) : '?';
  stats.push([`${g.used_gb.toFixed(1)} / ${g.total_gb.toFixed(1)} GB`, `GPU ${i} VRAM (${pct}%)`]);
});
document.getElementById('perf').innerHTML = stats.map(
  ([v, l]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');

function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;'); }

// Committed text living *before* frame i (blocks already finished).
function committedBefore(i){
  let out = '';
  for (let j = 0; j < i; j++)
    if (frames[j].committed)
      out += frames[j].ids.map(id => pieces[id] ?? '?').join('');
  return out;
}

function render(i){
  const f = frames[i];
  let html = '<span class="committed">' + esc(committedBefore(i)) + '</span>';
  if (f.committed) {
    html = '<span class="committed">' + esc(committedBefore(i + 1)) + '</span>';
    info.textContent = `frame ${i+1}/${frames.length} · block ${f.block} committed`;
  } else {
    for (let t = 0; t < f.ids.length; t++) {
      const piece = pieces[f.ids[t]] ?? '?';
      const H = f.ent ? f.ent[t] : 0;
      const a = Math.min(0.85, H / 4);           // entropy -> red alpha
      const cls = (f.acc && f.acc[t]) ? 'tok acc' : 'tok';
      html += `<span class="${cls}" style="background:rgba(255,60,60,${a.toFixed(3)})">` +
              esc(piece) + '</span>';
    }
    info.textContent = `frame ${i+1}/${frames.length} · block ${f.block} · step ${f.step}` +
                       ` · T=${f.temp.toFixed(2)} · H̄=${f.H.toFixed(4)}`;
  }
  canvas.innerHTML = html;
}

slider.oninput = () => render(+slider.value);

let timer = null;
play.onclick = () => {
  if (timer) { clearInterval(timer); timer = null; play.innerHTML = '&#9654; play'; return; }
  if (+slider.value >= frames.length - 1) slider.value = 0;
  play.innerHTML = '&#9646;&#9646; pause';
  timer = setInterval(() => {
    if (+slider.value >= frames.length - 1) { clearInterval(timer); timer = null; play.innerHTML = '&#9654; play'; return; }
    slider.value = +slider.value + 1; render(+slider.value);
  }, 350);
};

render(0);
</script></body></html>
)HTML";

    std::string html = kHtml;
    std::string marker = "__DATA__";
    size_t pos = html.find(marker);
    if (pos == std::string::npos) throw std::runtime_error("viz template marker missing");
    html.replace(pos, marker.size(), data.dump());

    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << html;
}

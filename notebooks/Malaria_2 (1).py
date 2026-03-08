import os
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("mobilenet_model1.h5")

IMG_SIZE = 128

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Malaria Detection AI</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Mono', monospace;
    background-color: #060d1a;
    color: var(--ink);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    position: relative;
    overflow-x: hidden;
  }

  /* Full-screen canvas background */
  #bg-canvas {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    pointer-events: none;
  }

  /* Mosquito canvas on top of bg */
  #mosquito-canvas {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
    pointer-events: none;
  }

  /* Subtle vignette overlay */
  body::after {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse at center, transparent 40%, rgba(4, 8, 18, 0.72) 100%);
    pointer-events: none;
    z-index: 0;
  }

  /* Override text colors for dark background */
  :root {
    --ink: #f0ece3;
    --paper: #f5f0e8;
    --accent: #ff6b6b;
    --accent-light: #ff8e8e;
    --safe: #4ecb82;
    --safe-light: #6ee8a0;
    --muted: #8a9ab5;
    --border: rgba(255,255,255,0.12);
    --card: rgba(10, 18, 35, 0.75);
  }

  .page-wrapper {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 620px;
    animation: fadeUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── Header ── */
  .header {
    margin-bottom: 8px;
    display: flex;
    align-items: baseline;
    gap: 12px;
  }

  .header-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 2px;
    background: rgba(255,255,255,0.04);
  }

  .header-line {
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 6vw, 3.2rem);
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin-bottom: 6px;
  }

  h1 em {
    font-style: italic;
    color: var(--accent);
  }

  .subtitle {
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 0.05em;
    margin-bottom: 36px;
  }

  /* ── Card ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 36px;
    box-shadow: 0 8px 48px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
  }

  .card:hover {
    box-shadow: 0 12px 64px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.1);
    transform: translateY(-2px);
  }

  /* ── Upload Zone ── */
  .upload-zone {
    border: 1px dashed rgba(255,255,255,0.2);
    border-radius: 8px;
    padding: 32px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    margin-bottom: 24px;
    background: rgba(255,255,255,0.03);
  }

  .upload-zone:hover, .upload-zone.dragover {
    border-color: rgba(100,180,255,0.5);
    background: rgba(100,180,255,0.05);
  }

  .upload-zone input[type="file"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    width: 100%;
    height: 100%;
  }

  .upload-icon {
    font-size: 2.4rem;
    margin-bottom: 12px;
    display: block;
    line-height: 1;
  }

  .upload-label {
    font-size: 13px;
    color: var(--ink);
    font-weight: 500;
    display: block;
    margin-bottom: 4px;
  }

  .upload-hint {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.04em;
  }

  .file-selected {
    display: none;
    font-size: 12px;
    color: var(--safe);
    margin-top: 10px;
    font-weight: 500;
  }

  /* ── Button ── */
  .btn {
    width: 100%;
    padding: 14px 24px;
    background: linear-gradient(135deg, #4a90e2, #7b5ea7);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
    box-shadow: 0 4px 20px rgba(74,144,226,0.35);
  }

  .btn span { position: relative; z-index: 1; }

  .btn:hover { opacity: 0.88; box-shadow: 0 6px 28px rgba(74,144,226,0.5); }
  .btn:active { transform: scale(0.985); }

  /* ── Divider ── */
  .divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0;
  }

  .divider::before, .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .divider span {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  /* ── Preview Image ── */
  .preview-wrap {
    text-align: center;
    animation: fadeUp 0.5s cubic-bezier(0.16,1,0.3,1) both;
  }

  .preview-frame {
    display: inline-block;
    border: 1.5px solid var(--border);
    padding: 8px;
    background: white;
    box-shadow: 4px 4px 0 var(--border);
    margin-bottom: 24px;
  }

  .preview-frame img {
    display: block;
    width: 240px;
    height: 240px;
    object-fit: cover;
  }

  /* ── Result ── */
  .result-box {
    border: 1px solid;
    border-radius: 8px;
    padding: 20px 24px;
    animation: fadeUp 0.5s 0.1s cubic-bezier(0.16,1,0.3,1) both;
    backdrop-filter: blur(8px);
  }

  .result-box.infected {
    border-color: rgba(255,107,107,0.4);
    background: rgba(255,60,60,0.08);
  }

  .result-box.healthy {
    border-color: rgba(78,203,130,0.4);
    background: rgba(78,203,130,0.08);
  }

  .result-tag {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: block;
  }

  .result-box.infected .result-tag { color: var(--accent); }
  .result-box.healthy .result-tag { color: var(--safe); }

  .result-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    line-height: 1.2;
    letter-spacing: -0.01em;
  }

  .result-box.infected .result-text { color: var(--accent); }
  .result-box.healthy .result-text { color: var(--safe); }

  .result-note {
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
    line-height: 1.5;
  }

  /* ── Footer ── */
  .footer {
    margin-top: 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .footer-note {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.06em;
  }

  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--safe-light);
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.75); }
  }

</style>
</head>
<body>

<canvas id="bg-canvas"></canvas>
<canvas id="mosquito-canvas"></canvas>

<div class="page-wrapper">

  <div class="header">
    <span class="header-eyebrow">AI Diagnostics v1.0</span>
    <div class="header-line"></div>
  </div>

  <h1>Malaria<br><em>Detection</em></h1>
  <p class="subtitle">Upload a blood smear cell image for instant analysis</p>

  <div class="card">

    <form method="POST" enctype="multipart/form-data" id="upload-form">
      <div class="upload-zone" id="drop-zone">
        <input type="file" name="image" accept="image/*" required id="file-input">
        <span class="upload-icon">🔬</span>
        <span class="upload-label">Drop cell image here</span>
        <span class="upload-hint">or click to browse — PNG, JPG, TIFF</span>
        <span class="file-selected" id="file-name">✓ File ready</span>
      </div>

      <button type="submit" class="btn"><span>Run Analysis →</span></button>
    </form>

    {% if prediction %}
    <div class="divider"><span>Result</span></div>

    {% if img_path %}
    <div class="preview-wrap">
      <div class="preview-frame">
        <img src="/{{ img_path }}" alt="Uploaded cell">
      </div>
    </div>
    {% endif %}

    <div class="result-box {% if 'Healthy' in prediction %}healthy{% else %}infected{% endif %}">
      <span class="result-tag">
        {% if 'Healthy' in prediction %}Status — Negative{% else %}Status — Positive{% endif %}
      </span>
      <div class="result-text">{{ prediction }}</div>
      <p class="result-note">
        This result is generated by a MobileNet deep learning model.
        Always confirm findings with a qualified medical professional.
      </p>
    </div>
    {% endif %}

  </div>

  <div class="footer">
    <span class="footer-note">Model: MobileNetV1 · IMG 128×128</span>
    <span><span class="dot"></span></span>
  </div>

</div>

<script>
// ════════════════════════════════════════════════════════
//  MULTI-LAYER ANIMATED BACKGROUND — Malaria Detection AI
// ════════════════════════════════════════════════════════
const canvas = document.getElementById('bg-canvas');
const ctx    = canvas.getContext('2d');
let W, H;

// ── Layer data ──
let stars = [], orbs = [], cells = [], trails = [], ripples = [], dnaStrands = [];
let mouseX = 0, mouseY = 0;

function resize() {
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;
}

// ── 1. STAR FIELD ──
function makeStar() {
  return {
    x: Math.random() * W, y: Math.random() * H,
    r: 0.4 + Math.random() * 1.6,
    alpha: 0.2 + Math.random() * 0.8,
    twinkle: Math.random() * Math.PI * 2,
    speed: 0.003 + Math.random() * 0.012,
    vx: (Math.random() - 0.5) * 0.05,
    vy: (Math.random() - 0.5) * 0.05,
  };
}

// ── 2. NEBULA ORBS ──
function makeOrb() {
  return {
    x: Math.random() * W, y: Math.random() * H,
    r: 80 + Math.random() * 220,
    vx: (Math.random() - 0.5) * 0.18,
    vy: (Math.random() - 0.5) * 0.18,
    hue: [210, 250, 280, 190, 320][Math.floor(Math.random()*5)] + Math.random()*25,
    alpha: 0.03 + Math.random() * 0.055,
    phase: Math.random() * Math.PI * 2,
  };
}

// ── 3. FLOATING BLOOD CELLS ──
function makeCell() {
  return {
    x: Math.random() * W, y: Math.random() * H,
    r: 7 + Math.random() * 22,
    vx: (Math.random() - 0.5) * 0.22,
    vy: (Math.random() - 0.5) * 0.22,
    rot: Math.random() * Math.PI * 2,
    vrot: (Math.random() - 0.5) * 0.008,
    infected: Math.random() < 0.28,
    alpha: 0.18 + Math.random() * 0.3,
    pulse: Math.random() * Math.PI * 2,
    wobble: Math.random() * Math.PI * 2,
    wSpeed: 0.01 + Math.random() * 0.02,
  };
}

// ── 4. PARTICLE TRAILS (flowing streams) ──
function makeTrail() {
  const side = Math.floor(Math.random() * 4);
  let x, y, angle;
  if (side === 0) { x = Math.random()*W; y = -10; angle = Math.PI/2 + (Math.random()-0.5)*0.8; }
  else if (side===1){ x=W+10; y=Math.random()*H; angle=Math.PI+(Math.random()-0.5)*0.8; }
  else if (side===2){ x=Math.random()*W; y=H+10; angle=-Math.PI/2+(Math.random()-0.5)*0.8; }
  else { x=-10; y=Math.random()*H; angle=(Math.random()-0.5)*0.8; }
  const spd = 0.6 + Math.random() * 1.2;
  return {
    x, y, angle, spd,
    vx: Math.cos(angle)*spd,
    vy: Math.sin(angle)*spd,
    hue: Math.random()<0.5 ? 200+Math.random()*40 : 280+Math.random()*40,
    alpha: 0.5 + Math.random()*0.5,
    r: 1.2 + Math.random()*2,
    history: [],
    maxLen: 18 + Math.floor(Math.random()*24),
    life: 1,
    dead: false,
  };
}

// ── 5. RIPPLE SHOCKWAVES ──
function makeRipple(x, y) {
  return { x, y, r: 0, maxR: 90+Math.random()*120, alpha: 0.35, speed: 1.4+Math.random()*1.2,
           hue: Math.random()<0.5?200:0 };
}

// ── 6. DNA DOUBLE-HELIX STRANDS ──
function makeDNA() {
  return {
    x: Math.random() * W,
    y: H + 60,
    vy: -(0.25 + Math.random()*0.35),
    phase: Math.random() * Math.PI * 2,
    len: 120 + Math.random()*160,
    alpha: 0.12 + Math.random()*0.18,
    hue1: 200 + Math.random()*40,
    hue2: 340 + Math.random()*30,
    twist: 0.06 + Math.random()*0.05,
  };
}

// ── Init ──
function init() {
  resize();
  stars      = Array.from({length: 160}, makeStar);
  orbs       = Array.from({length: 9},  makeOrb);
  cells      = Array.from({length: 55}, makeCell);
  trails     = Array.from({length: 22}, makeTrail);
  dnaStrands = Array.from({length: 5},  makeDNA);
  ripples    = [];

  // spawn ripples periodically
  setInterval(() => {
    if (ripples.length < 8) {
      ripples.push(makeRipple(Math.random()*W, Math.random()*H));
    }
  }, 1200);
}

function wrap(v, max, pad=200) {
  if (v < -pad) return max+pad;
  if (v > max+pad) return -pad;
  return v;
}

// ── Draw: stars ──
function drawStars(t) {
  stars.forEach(s => {
    s.x = wrap(s.x + s.vx, W, 10);
    s.y = wrap(s.y + s.vy, H, 10);
    const a = s.alpha * (0.5 + 0.5 * Math.sin(t * s.speed + s.twinkle));
    const gradient = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r*2);
    gradient.addColorStop(0, `rgba(200,220,255,${a})`);
    gradient.addColorStop(1, `rgba(200,220,255,0)`);
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.r*2, 0, Math.PI*2);
    ctx.fillStyle = gradient;
    ctx.fill();
  });
}

// ── Draw: nebula orbs ──
function drawOrbs(t) {
  orbs.forEach(o => {
    o.x = wrap(o.x + o.vx + Math.sin(t*0.0003+o.phase)*0.12, W);
    o.y = wrap(o.y + o.vy + Math.cos(t*0.0004+o.phase)*0.10, H);
    const pulse = 1 + 0.08*Math.sin(t*0.0006+o.phase);
    const r = o.r * pulse;
    const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, r);
    g.addColorStop(0,   `hsla(${o.hue},85%,65%,${o.alpha*1.4})`);
    g.addColorStop(0.4, `hsla(${o.hue},75%,50%,${o.alpha})`);
    g.addColorStop(1,   `hsla(${o.hue},60%,25%,0)`);
    ctx.beginPath();
    ctx.arc(o.x, o.y, r, 0, Math.PI*2);
    ctx.fillStyle = g;
    ctx.fill();
  });
}

// ── Draw: blood cells ──
function drawCells(t) {
  cells.forEach(c => {
    c.x   = wrap(c.x + c.vx, W);
    c.y   = wrap(c.y + c.vy, H);
    c.rot += c.vrot;
    c.wobble += c.wSpeed;
    const pulse = 1 + 0.07*Math.sin(t*0.0012+c.pulse);
    const rx = c.r * pulse;
    const ry = c.r * 0.78 * (1 + 0.04*Math.sin(c.wobble));
    const col = c.infected ? `hsla(0,90%,62%,${c.alpha})` : `hsla(195,85%,62%,${c.alpha})`;
    const fill= c.infected ? `hsla(0,70%,40%,${c.alpha*0.18})` : `hsla(195,70%,45%,${c.alpha*0.14})`;

    ctx.save();
    ctx.translate(c.x, c.y);
    ctx.rotate(c.rot);

    // outer glow
    const glow = ctx.createRadialGradient(0,0,rx*0.5, 0,0,rx*1.8);
    glow.addColorStop(0, c.infected?`rgba(255,80,80,0.06)`:`rgba(60,180,255,0.06)`);
    glow.addColorStop(1, `rgba(0,0,0,0)`);
    ctx.beginPath(); ctx.ellipse(0,0,rx*1.8,ry*1.8,0,0,Math.PI*2);
    ctx.fillStyle = glow; ctx.fill();

    // membrane
    ctx.beginPath(); ctx.ellipse(0,0,rx,ry,0,0,Math.PI*2);
    ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.fillStyle = fill; ctx.fill();

    // nucleus
    ctx.beginPath(); ctx.arc(rx*0.08, ry*0.05, rx*0.24, 0, Math.PI*2);
    ctx.fillStyle = c.infected?`hsla(0,85%,58%,${c.alpha*0.6})`:`hsla(210,80%,62%,${c.alpha*0.5})`;
    ctx.fill();

    // infected parasites (dots)
    if (c.infected) {
      for (let i=0; i<3; i++) {
        const px = Math.cos(i*2.1+c.rot*2)*rx*0.45;
        const py = Math.sin(i*2.1+c.rot*2)*ry*0.4;
        ctx.beginPath(); ctx.arc(px,py,rx*0.09,0,Math.PI*2);
        ctx.fillStyle = `hsla(40,100%,65%,${c.alpha*0.7})`; ctx.fill();
      }
    }
    ctx.restore();
  });
}

// ── Draw: flowing particle trails ──
function drawTrails() {
  trails.forEach((tr, i) => {
    tr.history.push({x: tr.x, y: tr.y});
    if (tr.history.length > tr.maxLen) tr.history.shift();
    tr.x += tr.vx; tr.y += tr.vy;

    // slight curve
    tr.angle += (Math.random()-0.5)*0.03;
    tr.vx = Math.cos(tr.angle)*tr.spd;
    tr.vy = Math.sin(tr.angle)*tr.spd;

    if (tr.x<-50||tr.x>W+50||tr.y<-50||tr.y>H+50) {
      trails[i] = makeTrail(); return;
    }

    if (tr.history.length < 2) return;
    for (let j=1; j<tr.history.length; j++) {
      const a = tr.alpha * (j/tr.history.length) * 0.55;
      ctx.beginPath();
      ctx.moveTo(tr.history[j-1].x, tr.history[j-1].y);
      ctx.lineTo(tr.history[j].x,   tr.history[j].y);
      ctx.strokeStyle = `hsla(${tr.hue},80%,65%,${a})`;
      ctx.lineWidth = tr.r * (j/tr.history.length);
      ctx.stroke();
    }
    // head glow
    const hg = ctx.createRadialGradient(tr.x,tr.y,0,tr.x,tr.y,tr.r*3);
    hg.addColorStop(0, `hsla(${tr.hue},90%,75%,${tr.alpha*0.8})`);
    hg.addColorStop(1, `hsla(${tr.hue},80%,55%,0)`);
    ctx.beginPath(); ctx.arc(tr.x,tr.y,tr.r*3,0,Math.PI*2);
    ctx.fillStyle = hg; ctx.fill();
  });
}

// ── Draw: ripples ──
function drawRipples() {
  ripples = ripples.filter(r => r.alpha > 0.01);
  ripples.forEach(r => {
    r.r   += r.speed;
    r.alpha *= 0.965;
    for (let ring=0; ring<3; ring++) {
      const rr = r.r - ring*14;
      if (rr < 0) continue;
      ctx.beginPath();
      ctx.arc(r.x, r.y, rr, 0, Math.PI*2);
      ctx.strokeStyle = `hsla(${r.hue},80%,65%,${r.alpha*(1-ring*0.3)})`;
      ctx.lineWidth = 1.2 - ring*0.3;
      ctx.stroke();
    }
  });
}

// ── Draw: DNA helices ──
function drawDNA(t) {
  dnaStrands.forEach((d, idx) => {
    d.y += d.vy;
    d.phase += 0.012;
    if (d.y < -d.len - 80) { dnaStrands[idx] = makeDNA(); return; }

    const steps = 32;
    const stepH = d.len / steps;
    const amp   = 18;

    ctx.save();
    ctx.translate(d.x, d.y + d.len);

    for (let i=0; i<steps; i++) {
      const y0 = -i * stepH;
      const y1 = -(i+1) * stepH;
      const a0 = d.phase + i * d.twist * 10;
      const a1 = d.phase + (i+1) * d.twist * 10;

      const x0a = Math.cos(a0)*amp, x0b = Math.cos(a0+Math.PI)*amp;
      const x1a = Math.cos(a1)*amp, x1b = Math.cos(a1+Math.PI)*amp;

      // strand A
      ctx.beginPath(); ctx.moveTo(x0a, y0); ctx.lineTo(x1a, y1);
      ctx.strokeStyle = `hsla(${d.hue1},80%,65%,${d.alpha})`; ctx.lineWidth=1.5; ctx.stroke();
      // strand B
      ctx.beginPath(); ctx.moveTo(x0b, y0); ctx.lineTo(x1b, y1);
      ctx.strokeStyle = `hsla(${d.hue2},80%,65%,${d.alpha})`; ctx.lineWidth=1.5; ctx.stroke();
      // rungs (every 2)
      if (i%2===0) {
        ctx.beginPath(); ctx.moveTo(x0a, y0); ctx.lineTo(x0b, y0);
        ctx.strokeStyle = `hsla(${(d.hue1+d.hue2)/2},60%,70%,${d.alpha*0.5})`; ctx.lineWidth=1; ctx.stroke();
        // node dots
        ctx.beginPath(); ctx.arc(x0a, y0, 2.2, 0, Math.PI*2);
        ctx.fillStyle = `hsla(${d.hue1},90%,72%,${d.alpha*0.9})`; ctx.fill();
        ctx.beginPath(); ctx.arc(x0b, y0, 2.2, 0, Math.PI*2);
        ctx.fillStyle = `hsla(${d.hue2},90%,72%,${d.alpha*0.9})`; ctx.fill();
      }
    }
    ctx.restore();
  });
}

// ── Mouse interaction: spawn ripple on click ──
canvas.style.pointerEvents = 'none'; // keep pass-through
document.addEventListener('click', e => {
  ripples.push(makeRipple(e.clientX, e.clientY));
});
document.addEventListener('mousemove', e => { mouseX=e.clientX; mouseY=e.clientY; });

// ── Main loop ──
function animate(t) {
  requestAnimationFrame(animate);

  // base bg
  ctx.fillStyle = '#060d1a';
  ctx.fillRect(0, 0, W, H);

  drawOrbs(t);
  drawStars(t);
  drawDNA(t);
  drawTrails();
  drawCells(t);
  drawRipples();
}

init();
animate(0);
window.addEventListener('resize', () => { resize(); });
</script>

<script>
// ════════════════════════════════════════════
//  ANIMATED MOSQUITO — Canvas-drawn, flying
// ════════════════════════════════════════════
(function() {
  const mc  = document.getElementById('mosquito-canvas');
  const mctx = mc.getContext('2d');
  let MW, MH;

  function mresize() {
    MW = mc.width  = window.innerWidth;
    MH = mc.height = window.innerHeight;
  }
  mresize();
  window.addEventListener('resize', mresize);

  // ── Mosquito object ──
  function makeMosquito() {
    // spawn from a random edge
    const edge = Math.floor(Math.random()*4);
    let x, y;
    if (edge===0){ x=Math.random()*MW; y=-60; }
    else if(edge===1){ x=MW+60; y=Math.random()*MH; }
    else if(edge===2){ x=Math.random()*MW; y=MH+60; }
    else { x=-60; y=Math.random()*MH; }

    // target somewhere on screen
    const tx = MW*0.2 + Math.random()*MW*0.6;
    const ty = MH*0.2 + Math.random()*MH*0.6;
    const ang = Math.atan2(ty-y, tx-x);
    const spd = 1.2 + Math.random()*1.4;

    return {
      x, y,
      vx: Math.cos(ang)*spd,
      vy: Math.sin(ang)*spd,
      angle: ang,
      targetX: tx, targetY: ty,
      scale: 0.55 + Math.random()*0.55,
      wingPhase: Math.random()*Math.PI*2,
      wingSpeed: 0.38 + Math.random()*0.18,
      wobblePhase: Math.random()*Math.PI*2,
      legPhase: Math.random()*Math.PI*2,
      alpha: 0.82 + Math.random()*0.18,
      dead: false,
      // path wander
      wanderAngle: ang,
      wanderTimer: 0,
    };
  }

  // draw one mosquito at origin, facing right, scale=1
  function drawMosquito(mctx, t, m) {
    const wf = m.wingPhase + t * m.wingSpeed;   // wing flap
    const wingBeat = Math.sin(wf);
    const wingY    = Math.abs(Math.cos(wf));     // 0..1 for shadow

    mctx.save();
    mctx.globalAlpha = m.alpha;
    const s = m.scale;

    // ── SHADOW under mosquito ──
    mctx.save();
    mctx.scale(s, s*0.35);
    mctx.translate(0, 38);
    mctx.beginPath();
    mctx.ellipse(0, 0, 10, 5, 0, 0, Math.PI*2);
    mctx.fillStyle = `rgba(0,0,0,${0.12*wingY})`;
    mctx.fill();
    mctx.restore();

    // ── ABDOMEN (elongated segmented body) ──
    // segments drawn as overlapping ellipses
    const segColors = ['#3a6b3a','#2e7d32','#388e3c','#43a047','#2e7d32','#1b5e20'];
    for (let i=0; i<6; i++) {
      const bx = i * 5.5 * s - 15*s;
      const bw = (7 - i*0.4)*s;
      const bh = (4 - i*0.3)*s;
      mctx.beginPath();
      mctx.ellipse(bx, 8*s, bw, bh, 0, 0, Math.PI*2);
      mctx.fillStyle = segColors[i];
      // stripe highlight
      mctx.fill();
      mctx.strokeStyle = 'rgba(0,0,0,0.25)';
      mctx.lineWidth = 0.5*s;
      mctx.stroke();
      // pale band
      mctx.beginPath();
      mctx.ellipse(bx, 8*s - bh*0.35, bw*0.8, bh*0.25, 0, 0, Math.PI*2);
      mctx.fillStyle = 'rgba(180,255,180,0.15)';
      mctx.fill();
    }

    // ── THORAX ──
    mctx.beginPath();
    mctx.ellipse(-20*s, 3*s, 9*s, 7*s, -0.2, 0, Math.PI*2);
    mctx.fillStyle = '#1a4a1a';
    mctx.fill();
    mctx.strokeStyle = 'rgba(0,0,0,0.3)';
    mctx.lineWidth = 0.8*s;
    mctx.stroke();

    // thorax shine
    mctx.beginPath();
    mctx.ellipse(-21*s, 0*s, 4*s, 3*s, -0.3, 0, Math.PI*2);
    mctx.fillStyle = 'rgba(120,200,120,0.25)';
    mctx.fill();

    // ── HEAD ──
    mctx.beginPath();
    mctx.ellipse(-30*s, 2*s, 6*s, 5.5*s, 0, 0, Math.PI*2);
    mctx.fillStyle = '#0d2e0d';
    mctx.fill();
    mctx.strokeStyle = 'rgba(0,0,0,0.4)';
    mctx.lineWidth = 0.6*s;
    mctx.stroke();

    // compound eye
    mctx.beginPath();
    mctx.ellipse(-28*s, 0*s, 3.5*s, 3*s, 0.2, 0, Math.PI*2);
    mctx.fillStyle = '#c62828';
    mctx.fill();
    mctx.beginPath();
    mctx.ellipse(-27*s, -0.5*s, 1.2*s, 1*s, 0.2, 0, Math.PI*2);
    mctx.fillStyle = 'rgba(255,200,200,0.5)';
    mctx.fill();

    // ── PROBOSCIS (needle) ──
    mctx.beginPath();
    mctx.moveTo(-34*s, 3*s);
    mctx.quadraticCurveTo(-44*s, 5*s, -52*s, 2*s);
    mctx.strokeStyle = '#1b1b1b';
    mctx.lineWidth = 1.2*s;
    mctx.stroke();
    // tip glow
    mctx.beginPath();
    mctx.arc(-52*s, 2*s, 1.5*s, 0, Math.PI*2);
    mctx.fillStyle = '#ff4444';
    mctx.fill();

    // ── ANTENNAE ──
    mctx.strokeStyle = '#1a3a1a';
    mctx.lineWidth = 0.8*s;
    // antenna 1
    mctx.beginPath();
    mctx.moveTo(-31*s, -2*s);
    mctx.quadraticCurveTo(-38*s, -14*s + Math.sin(wf*2)*2*s, -34*s, -20*s);
    mctx.stroke();
    // antenna 2
    mctx.beginPath();
    mctx.moveTo(-29*s, -2*s);
    mctx.quadraticCurveTo(-24*s, -16*s + Math.cos(wf*2)*2*s, -28*s, -21*s);
    mctx.stroke();
    // fluffy antenna tips (arista)
    for(let j=-2;j<=2;j++){
      mctx.beginPath();
      mctx.moveTo(-34*s+j*1.5*s, -20*s);
      mctx.lineTo(-34*s+j*1.5*s, -24*s);
      mctx.stroke();
      mctx.beginPath();
      mctx.moveTo(-28*s+j*1.5*s, -21*s);
      mctx.lineTo(-28*s+j*1.5*s, -25*s);
      mctx.stroke();
    }

    // ── WINGS ──
    // Left upper wing
    const wFlap = wingBeat * 18 * s;
    function drawWing(side) {
      const sy = side === 1 ? 1 : -1;
      const wAlpha = 0.28 + 0.12*Math.abs(wingBeat);
      // main wing membrane
      mctx.beginPath();
      mctx.moveTo(-18*s, 0);
      mctx.bezierCurveTo(
        -10*s, sy*(-26*s + wFlap*sy),
         10*s, sy*(-30*s + wFlap*sy),
         18*s, sy*(-8*s + wFlap*0.3*sy)
      );
      mctx.bezierCurveTo(12*s, sy*(-2*s), -6*s, sy*(2*s), -18*s, 0);
      mctx.fillStyle = `rgba(160,220,160,${wAlpha})`;
      mctx.fill();
      mctx.strokeStyle = `rgba(80,140,80,${wAlpha+0.15})`;
      mctx.lineWidth = 0.5*s;
      mctx.stroke();

      // wing veins
      mctx.strokeStyle = `rgba(60,120,60,${wAlpha*0.8})`;
      mctx.lineWidth = 0.4*s;
      // main vein
      mctx.beginPath();
      mctx.moveTo(-18*s, 0);
      mctx.quadraticCurveTo(0, sy*(-22*s + wFlap*sy*0.7), 16*s, sy*(-6*s));
      mctx.stroke();
      // secondary veins
      for(let v=0; v<3; v++){
        const vt = 0.3 + v*0.25;
        const vx = -18*s + vt*36*s;
        const vy_base = sy*((-22*s + wFlap*sy*0.7)*vt);
        mctx.beginPath();
        mctx.moveTo(vx, vy_base);
        mctx.lineTo(vx + (v-1)*3*s, sy*(-28*s + wFlap*sy*0.9));
        mctx.stroke();
      }
    }

    mctx.save(); mctx.translate(-18*s, 2*s);
    drawWing(1);   // top wing
    drawWing(-1);  // bottom wing (mirror)
    mctx.restore();

    // ── LEGS (6 legs) ──
    const legConfigs = [
      // [attach-x, attach-y, leg-length, base-angle, knee-angle-offset]
      [-22*s, 5*s,  22*s,  1.1,  0.35],
      [-18*s, 6*s,  20*s,  1.4,  0.30],
      [-12*s, 7*s,  22*s,  1.6,  0.25],
      [-22*s, 5*s,  22*s, -1.1, -0.35],
      [-18*s, 6*s,  20*s, -1.4, -0.30],
      [-12*s, 7*s,  22*s, -1.6, -0.25],
    ];
    legConfigs.forEach((lc, li) => {
      const legWobble = Math.sin(m.legPhase + t*0.008 + li*0.7) * 0.12;
      const baseA = lc[3] + legWobble;
      const knee  = lc[4];
      const len   = lc[2];
      const kx = lc[0] + Math.cos(baseA)*len*0.55;
      const ky = lc[1] + Math.sin(baseA)*len*0.55;
      const ex = kx + Math.cos(baseA + knee)*len*0.55;
      const ey = ky + Math.sin(baseA + knee)*len*0.55;

      mctx.beginPath();
      mctx.moveTo(lc[0], lc[1]);
      mctx.lineTo(kx, ky);
      mctx.lineTo(ex, ey);
      mctx.strokeStyle = '#1a3a1a';
      mctx.lineWidth   = 0.9*s;
      mctx.lineCap     = 'round';
      mctx.lineJoin    = 'round';
      mctx.stroke();
      // claw tip
      mctx.beginPath();
      mctx.arc(ex, ey, 1.2*s, 0, Math.PI*2);
      mctx.fillStyle = '#0d1a0d';
      mctx.fill();
    });

    mctx.restore();
  }

  // ── Mosquito swarm ──
  let mosquitoes = [];
  function spawnMosquito() {
    if (mosquitoes.length < 5) mosquitoes.push(makeMosquito());
  }
  // initial batch
  for(let i=0;i<3;i++) setTimeout(()=>mosquitoes.push(makeMosquito()), i*1200);
  setInterval(spawnMosquito, 3500);

  // ── Mosquito animation loop ──
  function animateMosquitoes(t) {
    requestAnimationFrame(animateMosquitoes);
    mctx.clearRect(0, 0, MW, MH);

    mosquitoes = mosquitoes.filter(m => !m.dead);

    mosquitoes.forEach(m => {
      // wander: periodically pick new target
      m.wanderTimer++;
      if (m.wanderTimer > 140 + Math.random()*80) {
        m.wanderTimer = 0;
        m.targetX = MW*0.15 + Math.random()*MW*0.7;
        m.targetY = MH*0.15 + Math.random()*MH*0.7;
      }

      // steer toward target smoothly
      const dx   = m.targetX - m.x;
      const dy   = m.targetY - m.y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      const desiredAngle = Math.atan2(dy, dx);
      let angleDiff = desiredAngle - m.wanderAngle;
      // normalise to -PI..PI
      while(angleDiff >  Math.PI) angleDiff -= 2*Math.PI;
      while(angleDiff < -Math.PI) angleDiff += 2*Math.PI;
      m.wanderAngle += angleDiff * 0.028;

      // add tiny random flutter
      m.wanderAngle += (Math.random()-0.5)*0.06;

      const spd = dist < 60 ? dist/60*1.5 + 0.3 : (1.2 + m.scale*0.6);
      m.vx = Math.cos(m.wanderAngle) * spd;
      m.vy = Math.sin(m.wanderAngle) * spd;
      m.x += m.vx;
      m.y += m.vy;

      // kill if way off screen
      if (m.x < -150 || m.x > MW+150 || m.y < -150 || m.y > MH+150) {
        m.dead = true; return;
      }

      // draw
      mctx.save();
      mctx.translate(m.x, m.y);
      // face direction of travel + slight bob
      const bob = Math.sin(t*0.004 + m.wobblePhase)*0.08;
      mctx.rotate(m.wanderAngle + bob);
      drawMosquito(mctx, t, m);
      mctx.restore();
    });
  }

  animateMosquitoes(0);
})();
</script>

<script>
  const input = document.getElementById('file-input');
  const label = document.getElementById('file-name');
  const zone  = document.getElementById('drop-zone');

  input.addEventListener('change', () => {
    if (input.files.length) {
      label.style.display = 'block';
      label.textContent = '✓ ' + input.files[0].name;
    }
  });

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const dt = e.dataTransfer;
    if (dt.files.length) {
      input.files = dt.files;
      label.style.display = 'block';
      label.textContent = '✓ ' + dt.files[0].name;
    }
  });
</script>

</body>
</html>
"""


def preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(img_path):
    processed = preprocess(img_path)
    prediction = model.predict(processed)
    if prediction[0][0] > 0.5:
        return "Malaria Detected (Parasitized)"
    else:
        return "Healthy Cell (Uninfected)"


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            prediction = predict(filepath)
            img_path = filepath

    return render_template_string(HTML_PAGE, prediction=prediction, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
/* ══════════════════════════════════════════════════════════════════════
   ImageCompose · Narrative Engine
   Native Scroll → Story Progress → Chapter → Stage Render + Agent
   —— 无滚轮劫持 / 无逐帧 setState / 全部渲染在 rAF 内完成
   ══════════════════════════════════════════════════════════════════════ */
(function () {
'use strict';

/* ═══════════════════ 0 · 章节注册表 ═══════════════════ */

const CHAPTERS = [
  { id:'hero',      s:0.000, e:0.065, num:'00', en:'HERO',      zh:'',            desc:'',
    caption:'我在这里。', meta:'FROM IMAGE TO REALITY', tool:'—', agent:[0.79,0.30] },
  { id:'plan',      s:0.065, e:0.130, num:'01', en:'PLAN',      zh:'理解意图',     desc:'先理解你的目标，再决定怎么做。',
    caption:'我会先分离主体，再建立场景，最后调整光影。', meta:'PLANNER · DAG', tool:'planner', agent:[0.70,0.62] },
  { id:'matting',   s:0.130, e:0.210, num:'02', en:'MATTING',   zh:'主体分离',     desc:'背景退场，边缘被精确找到。',
    caption:'主体已经锁定，我正在精修边缘。', meta:'BOUNDARY REFINEMENT', tool:'T01 matting', agent:[0.73,0.44] },
  { id:'scene',     s:0.210, e:0.300, num:'03', en:'SCENE',     zh:'空间生成',     desc:'不是换背景，是建立一个新的空间。',
    caption:'我在为它建立一个新的空间。', meta:'SPATIAL SYNTHESIS', tool:'T02 background', agent:[0.79,0.36] },
  { id:'light',     s:0.300, e:0.375, num:'04', en:'LIGHT',     zh:'读取环境光',   desc:'找到环境里的主光方向。',
    caption:'我已经找到环境中的主光方向。', meta:'LIGHT ESTIMATION', tool:'T03 lighting', agent:[0.72,0.24] },
  { id:'relight',   s:0.375, e:0.520, num:'05', en:'RELIGHT',   zh:'重新组织光线', desc:'让主体真正接受环境里的光。',
    caption:'现在，重新调整人物光线。', meta:'FOREGROUND RELIGHTING', tool:'T04 relight', agent:[0.20,0.30] },
  { id:'shadow',    s:0.520, e:0.595, num:'06', en:'SHADOW',    zh:'接触关系',     desc:'让它真正站在这里。',
    caption:'让它真正站在这里。', meta:'CONTACT SHADOW', tool:'T05 shadow', agent:[0.64,0.80] },
  { id:'harmonize', s:0.595, e:0.695, num:'07', en:'HARMONIZE', zh:'整体融合',     desc:'让画面看起来像原本就在那里。',
    caption:'统一整个画面的质感。', meta:'HARMONIZATION', tool:'T06 harmonize', agent:[0.52,0.79] },
  { id:'enhance',   s:0.695, e:0.770, num:'08', en:'ENHANCE',   zh:'景深与细节',   desc:'最后处理成成片该有的样子。',
    caption:'最后处理景深和细节。', meta:'DEPTH · GRAIN · DETAIL', tool:'T07 enhance', agent:[0.71,0.72] },
  { id:'critic',    s:0.770, e:0.865, num:'09', en:'CRITIC',    zh:'自检',         desc:'让我检查一下。',
    caption:'让我检查一下。', meta:'SELF EVALUATION', tool:'critic', agent:[0.83,0.50] },
  { id:'final',     s:0.865, e:0.920, num:'10', en:'FINAL',     zh:'完成',         desc:'',
    caption:'好了。', meta:'OUTPUT', tool:'T08 export', agent:[0.87,0.22] },
  { id:'create',    s:0.920, e:1.000, num:'11', en:'CREATE',    zh:'开始你的创作', desc:'现在，把你的图片交给我。',
    caption:'把你的图片交给我。', meta:'YOUR TURN', tool:'—', agent:[0.50,0.26] },
];

/* ═══════════════════ 1 · 工具函数 ═══════════════════ */

const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
const lerp = (a, b, t) => a + (b - a) * t;
const seg = (p, a, b) => clamp01((p - a) / (b - a));
const easeInOut = t => t < .5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2) / 2;
const easeOut = t => 1 - Math.pow(1 - t, 3);
const easeIn = t => t*t*t;

/* ── 颜色：HSL 降饱和 ── */
function hex2hsl(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = ((n >> 16) & 255) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
  const mx = Math.max(r,g,b), mn = Math.min(r,g,b);
  let h = 0, ss = 0; const l = (mx + mn) / 2;
  if (mx !== mn) {
    const d = mx - mn;
    ss = l > .5 ? d / (2 - mx - mn) : d / (mx + mn);
    h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
    h *= 60;
  }
  return [h, ss, l];
}
/** 按 desat 量（0=原色 1=全灰）输出 css 颜色，并可叠加明度偏移与透明度 */
function tone(hex, desat, alpha, lightShift) {
  const [h, s, l] = hex2hsl(hex);
  const L = clamp01(l + (lightShift || 0));
  const S = Math.round(s * (1 - desat) * 100);
  return alpha === undefined || alpha === 1
    ? `hsl(${h.toFixed(0)},${S}%,${(L*100).toFixed(1)}%)`
    : `hsla(${h.toFixed(0)},${S}%,${(L*100).toFixed(1)}%,${alpha})`;
}

/** 两个十六进制色按 t 插值，用于深浅章的连续过渡 */
function mixHex(a, b, t) {
  const A = parseInt(a.slice(1), 16), B = parseInt(b.slice(1), 16);
  const r = Math.round(lerp((A >> 16) & 255, (B >> 16) & 255, t));
  const g = Math.round(lerp((A >> 8) & 255, (B >> 8) & 255, t));
  const bl = Math.round(lerp(A & 255, B & 255, t));
  return `rgb(${r},${g},${bl})`;
}

/* ── 噪声贴图（预生成一次，逐帧只做 drawImage） ── */
function makeNoiseTile(size) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const x = c.getContext('2d');
  const img = x.createImageData(size, size);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = 128 + (Math.random() - .5) * 118;
    d[i] = d[i+1] = d[i+2] = v; d[i+3] = 255;
  }
  x.putImageData(img, 0, 0);
  return c;
}

/* ── Catmull-Rom → 三次贝塞尔，用于平滑剪影 ── */
function smoothPath(ctx, pts, tx, ty, sx, sy) {
  const n = pts.length;
  const P = i => pts[(i % n + n) % n];
  const X = p => tx + p[0] * sx, Y = p => ty + p[1] * sy;
  ctx.beginPath();
  ctx.moveTo(X(P(0)), Y(P(0)));
  for (let i = 0; i < n; i++) {
    const p0 = P(i-1), p1 = P(i), p2 = P(i+1), p3 = P(i+2);
    ctx.bezierCurveTo(
      X(p1) + (X(p2) - X(p0)) / 6, Y(p1) + (Y(p2) - Y(p0)) / 6,
      X(p2) - (X(p3) - X(p1)) / 6, Y(p2) - (Y(p3) - Y(p1)) / 6,
      X(p2), Y(p2)
    );
  }
  ctx.closePath();
}

/* ═══════════════════ 2 · 主体剪影点位 ═══════════════════ */
/* 面向左的女性侧影，站姿全身。拆成两个独立轮廓——躯干 + 垂落的长发。
   分开绘制可避免单一路径自交（发丝并入主轮廓会长出尖角），
   也让头发自然压在肩背后侧。
   ⚠ 归一化包围盒的宽高比 ≈ 0.43，横向缩放必须按真实人体比例压缩，
     否则人物会被撑成上宽下窄的色块。 */
const BODY_PTS = [
  // 前侧：颈 → 下颌 → 面部 → 头顶
  [0.462,0.224],[0.424,0.206],[0.402,0.184],[0.382,0.160],
  [0.358,0.132],[0.382,0.098],[0.398,0.074],[0.412,0.050],
  [0.452,0.022],[0.510,0.006],[0.568,0.024],[0.604,0.068],
  [0.614,0.120],[0.596,0.182],
  // 后侧：斜方肌 → 肩 → 臂 → 腿后 → 脚跟
  [0.650,0.220],[0.716,0.268],[0.762,0.372],[0.786,0.500],
  [0.790,0.640],[0.772,0.740],[0.740,0.822],[0.700,0.920],
  [0.676,0.982],[0.664,1.000],
  // 脚底
  [0.452,1.000],
  // 前侧：脚趾 → 踝 → 胫 → 膝 → 大腿 → 髋 → 腰 → 胸 → 肩前
  [0.462,0.978],[0.480,0.900],[0.502,0.800],[0.526,0.680],
  [0.548,0.540],[0.530,0.440],[0.470,0.360],[0.440,0.300],
  [0.416,0.252],
];
const HAIR_PTS = [
  [0.452,0.022],[0.510,0.002],[0.572,0.026],[0.618,0.074],
  [0.652,0.150],[0.678,0.250],[0.692,0.350],[0.680,0.412],
  [0.652,0.430],[0.634,0.372],[0.618,0.276],[0.598,0.180],
  [0.560,0.100],[0.508,0.046],
];

/* ═══════════════════ 3 · 关键帧轨道 ═══════════════════ */
/* 每一列是一帧画面的物理状态，按 storyProgress 插值 */

const TRACK = [
  // p      desat  bgA   bgBlur ridge ground subj  edge  lightI angle   warm  shadow grain warmG depth
  { p:0.000, d:1.00, bg:0.34, bl:22, rg:0, gr:0,  sb:0.46, eg:0,    li:0,   an:-2.42, wm:0.22, sh:0,    gn:0.62, wg:0,    dp:0 },
  { p:0.065, d:1.00, bg:0.34, bl:22, rg:0, gr:0,  sb:0.46, eg:0.05, li:0,   an:-2.42, wm:0.22, sh:0,    gn:0.62, wg:0,    dp:0 },
  { p:0.130, d:0.95, bg:0.36, bl:20, rg:0, gr:0,  sb:0.58, eg:0.16, li:0,   an:-2.42, wm:0.22, sh:0,    gn:0.60, wg:0,    dp:0 },
  { p:0.210, d:0.88, bg:0.05, bl:30, rg:0, gr:0,  sb:0.96, eg:1.00, li:0,   an:-2.42, wm:0.24, sh:0,    gn:0.56, wg:0,    dp:0 },
  { p:0.300, d:0.60, bg:1.00, bl:2,  rg:1, gr:1,  sb:0.96, eg:0.46, li:0.12,an:-2.42, wm:0.34, sh:0.05, gn:0.50, wg:0.10, dp:0 },
  { p:0.375, d:0.48, bg:1.00, bl:1,  rg:1, gr:1,  sb:0.96, eg:0.30, li:0.26,an:-2.10, wm:0.46, sh:0.16, gn:0.47, wg:0.16, dp:0 },
  { p:0.520, d:0.20, bg:1.00, bl:1,  rg:1, gr:1,  sb:0.99, eg:0.88, li:1.00,an:-2.02, wm:0.88, sh:0.36, gn:0.42, wg:0.46, dp:0 },
  { p:0.595, d:0.18, bg:1.00, bl:1,  rg:1, gr:1,  sb:0.99, eg:0.50, li:0.92,an:-2.02, wm:0.86, sh:1.00, gn:0.40, wg:0.46, dp:0 },
  { p:0.695, d:0.07, bg:1.00, bl:1,  rg:0.94,gr:1, sb:1.00,eg:0.28, li:0.86,an:-2.02, wm:0.80, sh:0.96, gn:0.34, wg:0.50, dp:0.35 },
  { p:0.770, d:0.02, bg:1.00, bl:1,  rg:0.86,gr:1, sb:1.00,eg:0.18, li:0.82,an:-2.02, wm:0.78, sh:0.93, gn:0.24, wg:0.50, dp:1.00 },
  { p:0.865, d:0.01, bg:1.00, bl:1,  rg:0.86,gr:1, sb:1.00,eg:0.16, li:0.80,an:-2.02, wm:0.78, sh:0.92, gn:0.22, wg:0.50, dp:1.00 },
  { p:0.920, d:0.00, bg:1.00, bl:1,  rg:0.86,gr:1, sb:1.00,eg:0.14, li:0.80,an:-2.02, wm:0.78, sh:0.92, gn:0.20, wg:0.50, dp:1.00 },
  { p:1.000, d:0.00, bg:1.00, bl:1,  rg:0.86,gr:1, sb:1.00,eg:0.14, li:0.80,an:-2.02, wm:0.78, sh:0.92, gn:0.20, wg:0.50, dp:1.00 },
];

function sampleTrack(p) {
  let i = 0;
  while (i < TRACK.length - 2 && p > TRACK[i+1].p) i++;
  const a = TRACK[i], b = TRACK[i+1];
  const t = easeInOut(clamp01((p - a.p) / (b.p - a.p)));
  return {
    desat:lerp(a.d,b.d,t), bgA:lerp(a.bg,b.bg,t), bgBlur:lerp(a.bl,b.bl,t),
    ridge:lerp(a.rg,b.rg,t), ground:lerp(a.gr,b.gr,t),
    subj:lerp(a.sb,b.sb,t), edge:lerp(a.eg,b.eg,t),
    lightI:lerp(a.li,b.li,t), angle:lerp(a.an,b.an,t), warm:lerp(a.wm,b.wm,t),
    shadow:lerp(a.sh,b.sh,t), grain:lerp(a.gn,b.gn,t), warmG:lerp(a.wg,b.wg,t),
    depth:lerp(a.dp,b.dp,t),
    raw:p
  };
}

/* ═══════════════════ 4 · 调色板 ═══════════════════ */
const PAL = {
  hazeHi:'#D9E1E2', hazeLo:'#C3D0D4',
  skyTop:'#9FB6C0', skyMid:'#C9D3D2', skyLow:'#E4DCCD',
  ridgeFar:'#93A8B0', ridgeNear:'#7C929B',
  groundHi:'#9AACAE', groundLo:'#7B8E92',
  subject:'#333C43', subjectLit:'#EBCFAA',
  shadowCol:'#2A3238',
  paper:'#F3F2EE', paperDark:'#1F2427',
};

/* ═══════════════════ 5 · 舞台渲染器 ═══════════════════ */

const Stage = {
  cv:null, ctx:null, W:0, H:0, dpr:1,
  noise:null, _pat:null, F:{x:0,y:0,w:0,h:0},

  init(cv) {
    this.cv = cv;
    this.ctx = cv.getContext('2d', { alpha:false });
    this.noise = makeNoiseTile(220);
    this.resize();
  },

  resize() {
    const W = window.innerWidth, H = window.innerHeight;
    // DPR 上限 1.6 —— 兼顾清晰度与逐帧填充成本
    this.dpr = Math.min(window.devicePixelRatio || 1, 1.6);
    this.W = W; this.H = H;
    this.cv.width = Math.round(W * this.dpr);
    this.cv.height = Math.round(H * this.dpr);
    this.cv.style.width = W + 'px';
    this.cv.style.height = H + 'px';
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    // 画面框：落在「左侧章节文字」与「右侧 Agent」之间
    const availW = W > 900 ? W * 0.56 : W * 0.82;
    const availH = H * 0.76;
    let fh = Math.min(availH, availW / 1.42);
    let fw = fh * 1.42;
    if (fw > availW) { fw = availW; fh = fw / 1.42; }
    const leftEdge = W > 900 ? W * 0.30 : (W - fw) / 2;
    this.F = {
      x: leftEdge + (availW - fw) / 2,
      y: (H - fh) / 2,
      w: fw, h: fh,
    };
  },

  /* ─── 主渲染 ─── */
  render(s, time) {
    const ctx = this.ctx, W = this.W, H = this.H, F = this.F;
    const breathe = Math.sin(time * 0.00042) * 0.5 + 0.5;

    // 底：纸张色 ↔ 深色章，随 darkMix 连续过渡
    ctx.fillStyle = mixHex(PAL.paper, PAL.paperDark, s.darkMix || 0);
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    ctx.beginPath();
    ctx.rect(F.x, F.y, F.w, F.h);
    ctx.clip();

    // 画框底
    ctx.fillStyle = tone(PAL.hazeLo, s.desat, 1);
    ctx.fillRect(F.x, F.y, F.w, F.h);

    this._haze(ctx, F, s);
    if (s.bgA > 0.02) this._sky(ctx, F, s.bgA, s.desat);
    if (s.ridge > 0.02) this._ridges(ctx, F, s);
    if (s.ground > 0.02) this._ground(ctx, F, s);
    if (s.shadow > 0.02) this._contact(ctx, F, s, breathe);
    if (s.subj > 0.02) this._subject(ctx, F, s);
    if (s.lightI > 0.02) this._light(ctx, F, s, breathe);
    this._grade(ctx, F, s);

    ctx.restore();
  },

  /* ── 雾：始终存在的最底层，负责“未完成感” ── */
  _haze(ctx, F, s) {
    const g = ctx.createLinearGradient(0, F.y, 0, F.y + F.h);
    g.addColorStop(0, tone(PAL.hazeHi, s.desat, 1));
    g.addColorStop(.55, tone(PAL.hazeLo, s.desat, 1));
    g.addColorStop(1, tone(PAL.hazeLo, s.desat, 1, .04));
    ctx.fillStyle = g;
    ctx.fillRect(F.x, F.y, F.w, F.h);

    // 环境光斑：给灰雾原图一点可读的明暗结构，避免画面塌成一块平渐层
    const lx = F.x + F.w * (s.lx === undefined ? .34 : s.lx);
    const ly = F.y + F.h * (s.ly === undefined ? .18 : s.ly);
    const rg = ctx.createRadialGradient(lx, ly, 0, lx, ly, Math.max(F.w, F.h) * .74);
    rg.addColorStop(0, `rgba(255,248,234,${.30 * (1 - s.warmG * .25)})`);
    rg.addColorStop(.48, 'rgba(255,246,228,.085)');
    rg.addColorStop(1, 'rgba(255,240,220,0)');
    ctx.fillStyle = rg;
    ctx.fillRect(F.x, F.y, F.w, F.h);
  },

  /* ── 天空：空间建立后淡入 ── */
  _sky(ctx, F, a, d) {
    const g = ctx.createLinearGradient(0, F.y, 0, F.y + F.h * .72);
    g.addColorStop(0, tone(PAL.skyTop, d, a));
    g.addColorStop(.52, tone(PAL.skyMid, d, a));
    g.addColorStop(1, tone(PAL.skyLow, d, a));
    ctx.fillStyle = g;
    ctx.fillRect(F.x, F.y, F.w, F.h * .74);
  },

  /* ── 远山：两层正弦叠加 ── */
  _ridges(ctx, F, s) {
    const layers = [
      { base:.628, amp:.050, f:1.5, ph:0.4, col:PAL.ridgeFar,  a:.85 },
      { base:.678, amp:.034, f:2.6, ph:2.1, col:PAL.ridgeNear, a:.95 },
    ];
    layers.forEach((L, k) => {
      const al = s.ridge * L.a * (k === 0 ? 1 : Math.min(1, s.ridge * 1.2));
      if (al < .02) return;
      ctx.beginPath();
      ctx.moveTo(F.x, F.y + F.h);
      const N = 42;
      for (let i = 0; i <= N; i++) {
        const t = i / N;
        const y = L.base
          + Math.sin(t * Math.PI * L.f + L.ph) * L.amp
          + Math.sin(t * Math.PI * L.f * 2.7 + L.ph * 1.7) * L.amp * .38;
        ctx.lineTo(F.x + t * F.w, F.y + y * F.h);
      }
      ctx.lineTo(F.x + F.w, F.y + F.h);
      ctx.closePath();
      ctx.fillStyle = tone(L.col, s.desat, al);
      ctx.fill();
    });
  },

  /* ── 地面 / 水面 ── */
  _ground(ctx, F, s) {
    const top = F.y + F.h * .695;
    const g = ctx.createLinearGradient(0, top, 0, F.y + F.h);
    g.addColorStop(0, tone(PAL.groundHi, s.desat, s.ground, .03));
    g.addColorStop(1, tone(PAL.groundLo, s.desat, s.ground, -.06));
    ctx.fillStyle = g;
    ctx.fillRect(F.x, top, F.w, F.y + F.h - top);

    // 水面横纹（极淡）
    ctx.save();
    ctx.globalAlpha = s.ground * .12 * (1 - s.desat * .6);
    ctx.strokeStyle = tone('#FFFFFF', 0, 1);
    ctx.lineWidth = 1;
    for (let i = 1; i <= 7; i++) {
      const y = top + (F.h - (top - F.y)) * (i / 8) * .92;
      ctx.beginPath();
      ctx.moveTo(F.x + F.w * (.10 + i * .028), y);
      ctx.lineTo(F.x + F.w * (.42 + i * .05), y);
      ctx.stroke();
    }
    ctx.restore();
  },

  /* ── 接触阴影：位置与硬度随光位变化 ── */
  _contact(ctx, F, s, breathe) {
    const feetY = F.y + F.h * .902;
    const off = Math.sin(s.angle) * F.w * .055;
    const cx = F.x + F.w * .50 + off;
    const rx = F.w * (.072 + .022 * s.shadow);
    const ry = F.h * .014;

    ctx.save();
    ctx.globalCompositeOperation = 'multiply';
    ctx.translate(cx, feetY + ry * .4);
    ctx.scale(1, ry / rx);
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
    g.addColorStop(0, tone(PAL.shadowCol, s.desat * .5, .58 * s.shadow));
    g.addColorStop(.45, tone(PAL.shadowCol, s.desat * .5, .26 * s.shadow));
    g.addColorStop(1, tone(PAL.shadowCol, s.desat * .5, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(0, 0, rx, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  },

  /* ── 主体：剪影 + 方向性受光 + 边缘光 ── */
  _subject(ctx, F, s) {
    const bh = F.h * .72;                      // 人物高度：站在中景
    const bw = bh * .52;                       // 横向压缩，还原真实人体宽高比
    const bx = F.x + F.w * .50 - bw * .574;    // 视觉中轴对齐画框中心
    const by = F.y + F.h * .90 - bh;           // 脚底落在地面线上

    ctx.save();
    ctx.globalAlpha = s.subj;

    // 落影：极小偏移，只给剪影一点厚度，不做重影
    ctx.save();
    ctx.translate(bw * .010, bh * .003);
    ctx.fillStyle = `rgba(26,32,38,${.22 * s.subj})`;
    smoothPath(ctx, HAIR_PTS, bx, by, bw, bh); ctx.fill();
    smoothPath(ctx, BODY_PTS, bx, by, bw, bh); ctx.fill();
    ctx.restore();

    // 头发：压在躯干之后（背光面，稍深一档）
    smoothPath(ctx, HAIR_PTS, bx, by, bw, bh);
    ctx.fillStyle = tone(PAL.subject, s.desat * .55, 1, -.045);
    ctx.fill();

    // 躯干
    smoothPath(ctx, BODY_PTS, bx, by, bw, bh);
    ctx.fillStyle = tone(PAL.subject, s.desat * .55, 1);
    ctx.fill();

    // 受光面：clip 在躯干内，沿光方向铺一层渐变
    const ax = Math.cos(s.angle), ay = Math.sin(s.angle);
    ctx.save();
    smoothPath(ctx, BODY_PTS, bx, by, bw, bh);
    ctx.clip();
    const lx = bx + bw * .50, ly = by + bh * .45;
    const R = bw * .95;
    const lg = ctx.createLinearGradient(lx + ax * R, ly + ay * R, lx - ax * R, ly - ay * R);
    const warm = s.warm, li = Math.min(1, s.lightI);
    // 多段渐变：受光面 → 过渡 → 暗面 → 逆光边缘，让人形有体积而不是一块平色
    lg.addColorStop(0,   `rgba(255,${Math.round(236 - warm * 8)},${Math.round(206 - warm * 22)},${1.00 * li})`);
    lg.addColorStop(.20, `rgba(255,${Math.round(228 - warm * 10)},${Math.round(196 - warm * 24)},${.64 * li})`);
    lg.addColorStop(.46, `rgba(212,188,162,${.26 * li})`);
    lg.addColorStop(.70, `rgba(44,56,66,${.34 * li})`);
    lg.addColorStop(1,   `rgba(18,26,34,${.62 * li})`);
    ctx.fillStyle = lg;
    ctx.fillRect(bx - bw * .2, by - bh * .1, bw * 1.5, bh * 1.2);

    // 明暗过渡柔化
    const sg = ctx.createLinearGradient(
      lx + ax * R * .3, ly + ay * R * .3,
      lx - ax * R * .75, ly - ay * R * .75
    );
    sg.addColorStop(0, `rgba(255,238,214,${.16 * li})`);
    sg.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = sg;
    ctx.fillRect(bx - bw * .2, by - bh * .1, bw * 1.5, bh * 1.2);
    ctx.restore();

    // 边缘光 / 轮廓识别
    if (s.edge > .015) {
      ctx.save();
      ctx.lineWidth = Math.max(.6, bw * .0022);
      const eg = ctx.createLinearGradient(
        bx + bw * .18, by + bh * .14, bx + bw * .84, by + bh * .74
      );
      eg.addColorStop(0, `rgba(228,242,248,${.74 * s.edge})`);
      eg.addColorStop(.5, `rgba(198,224,234,${.38 * s.edge})`);
      eg.addColorStop(1, `rgba(160,196,212,${.16 * s.edge})`);
      ctx.strokeStyle = eg;
      smoothPath(ctx, HAIR_PTS, bx, by, bw, bh); ctx.stroke();
      smoothPath(ctx, BODY_PTS, bx, by, bw, bh); ctx.stroke();
      ctx.restore();
    }

    ctx.restore();
  },

  /* ── 光：光晕 + 环境提亮 ── */
  _light(ctx, F, s, breathe) {
    const li = s.lightI;
    const lx = F.x + F.w * (s.lx === undefined ? .30 : s.lx);
    const ly = F.y + F.h * (s.ly === undefined ? .16 : s.ly);
    const R = Math.max(F.w, F.h) * (.62 + breathe * .04);

    ctx.save();
    ctx.globalCompositeOperation = 'screen';

    // 主光晕
    const g = ctx.createRadialGradient(lx, ly, 0, lx, ly, R);
    const w = s.warm;
    g.addColorStop(0,   `rgba(255,${Math.round(240 - w * 8)},${Math.round(212 - w * 30)},${.52 * li})`);
    g.addColorStop(.22, `rgba(255,${Math.round(226 - w * 10)},${Math.round(190 - w * 32)},${.22 * li})`);
    g.addColorStop(.55, `rgba(246,214,178,${.075 * li})`);
    g.addColorStop(1,   'rgba(240,200,160,0)');
    ctx.fillStyle = g;
    ctx.fillRect(F.x, F.y, F.w, F.h);

    // 光柱（极克制的方向性）
    const ax = Math.cos(s.angle), ay = Math.sin(s.angle);
    ctx.globalAlpha = .10 * li;
    ctx.translate(lx, ly);
    ctx.rotate(Math.atan2(ay, ax));
    const bg = ctx.createLinearGradient(0, 0, R * 1.15, 0);
    bg.addColorStop(0, 'rgba(255,236,206,.55)');
    bg.addColorStop(.45, 'rgba(255,226,190,.16)');
    bg.addColorStop(1, 'rgba(255,220,180,0)');
    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.moveTo(0, -F.h * .035);
    ctx.lineTo(R * 1.15, -F.h * .26);
    ctx.lineTo(R * 1.15,  F.h * .26);
    ctx.lineTo(0,  F.h * .035);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  },

  /* ── 调色 / 颗粒 / 暗角 ── */
  _grade(ctx, F, s) {
    // 暖调统一
    if (s.warmG > .01) {
      ctx.save();
      ctx.globalCompositeOperation = 'soft-light';
      ctx.fillStyle = `rgba(255,208,158,${.42 * s.warmG})`;
      ctx.fillRect(F.x, F.y, F.w, F.h);
      ctx.restore();
    }
    // 暗角
    const g = ctx.createRadialGradient(
      F.x + F.w * .5, F.y + F.h * .48, Math.min(F.w, F.h) * .26,
      F.x + F.w * .5, F.y + F.h * .5, Math.max(F.w, F.h) * .72
    );
    g.addColorStop(0, 'rgba(20,26,32,0)');
    g.addColorStop(1, `rgba(20,26,32,${.36 * (1 - s.warmG * .35)})`);
    ctx.fillStyle = g;
    ctx.fillRect(F.x, F.y, F.w, F.h);

    // 颗粒
    if (s.grain > .01) {
      ctx.save();
      ctx.globalAlpha = s.grain * .05;
      ctx.globalCompositeOperation = 'overlay';
      if (!this._pat) this._pat = ctx.createPattern(this.noise, 'repeat');
      ctx.fillStyle = this._pat;
      ctx.fillRect(F.x, F.y, F.w, F.h);
      ctx.restore();
    }

    // 景深：整体轻推（在前景层级已由主体锐度体现，这里做背景柔化替代）
    if (s.depth > .02) {
      ctx.save();
      ctx.globalAlpha = .16 * s.depth;
      const dg = ctx.createLinearGradient(0, F.y, 0, F.y + F.h * .42);
      dg.addColorStop(0, 'rgba(214,224,224,.85)');
      dg.addColorStop(1, 'rgba(214,224,224,0)');
      ctx.fillStyle = dg;
      ctx.fillRect(F.x, F.y, F.w, F.h * .42);
      ctx.restore();
    }
  },
};

/* ═══════════════════ 6 · Agent 存在 ═══════════════════ */

const Agent = {
  cv:null, ctx:null, state:'idle', t:0,
  SPEC:{
    idle:      { amp:.10, freq:1.1, speed:.0009, hollow:false, line:0 },
    listening: { amp:.30, freq:2.0, speed:.0034, hollow:false, line:0 },
    thinking:  { amp:.06, freq:0.6, speed:.0022, hollow:true,  line:1 },
    speaking:  { amp:.86, freq:3.4, speed:.0052, hollow:false, line:0 },
    working:   { amp:.40, freq:2.4, speed:.0041, hollow:false, line:0 },
    checking:  { amp:.03, freq:0.4, speed:.0011, hollow:true,  line:1 },
    done:      { amp:.05, freq:0.5, speed:.0006, hollow:false, line:1 },
  },
  init(cv) {
    this.cv = cv;
    this.ctx = cv.getContext('2d');
    const r = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = 240 * r; cv.height = 54 * r;
    cv.style.width = '120px'; cv.style.height = '27px';
    this.ctx.setTransform(r, 0, 0, r, 0, 0);
  },
  set(state) { if (this.state !== state) { this.state = state; } },
  draw(time) {
    const ctx = this.ctx, w = 240, h = 54, sp = this.SPEC[this.state] || this.SPEC.idle;
    ctx.clearRect(0, 0, w, h);
    const mid = h / 2;
    const phase = time * sp.speed;

    // 平线态（thinking / checking / done）
    if (sp.line) {
      ctx.strokeStyle = 'rgba(126,167,180,.62)';
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.moveTo(0, mid); ctx.lineTo(w, mid);
      ctx.stroke();
      if (this.state === 'thinking') {
        const px = ((time * .00042) % 1) * w;
        ctx.fillStyle = '#7EA7B4';
        ctx.beginPath(); ctx.arc(px, mid, 2.4, 0, Math.PI * 2); ctx.fill();
      }
      return;
    }

    // 波形：双正弦叠加 × 端部包络
    ctx.strokeStyle = this.state === 'speaking' ? 'rgba(126,167,180,.95)' : 'rgba(126,167,180,.62)';
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    const N = 96;
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      const env = Math.pow(Math.sin(t * Math.PI), .72);
      const y = mid
        + Math.sin(t * Math.PI * 2 * sp.freq + phase * 6.1) * sp.amp * h * .40 * env
        + Math.sin(t * Math.PI * 2 * sp.freq * 1.83 + phase * 9.3) * sp.amp * h * .17 * env
        + Math.sin(t * Math.PI * 2 * sp.freq * .47 + phase * 3.1) * sp.amp * h * .12 * env;
      i ? ctx.lineTo(t * w, y) : ctx.moveTo(t * w, y);
    }
    ctx.stroke();
  },
};

/* ═══════════════════ 7 · CRITIC 子脚本 ═══════════════════ */

const CRITIC = {
  // localProgress 分镜
  stage(lp) {
    if (lp < 0.06) return 'enter';
    if (lp < 0.30) return 'scan';
    if (lp < 0.42) return 'fail';
    if (lp < 0.60) return 'reroll';
    if (lp < 0.86) return 'repair';
    if (lp < 0.96) return 'recheck';
    return 'pass';
  },
  SCORES: { lighting:94, shadow:78, color:95, edge:97 },
  FIXED:  { shadow:93 },
  WEIGHT: { lighting:.30, shadow:.25, color:.20, edge:.25 },
  overall(sc) { return Object.keys(this.WEIGHT).reduce((a,k) => a + sc[k] * this.WEIGHT[k], 0); },
};

/* ═══════════════════ 8 · 上传 / 处理演示 ═══════════════════ */

const STEPS = ['图片已接收','主体已识别','场景分析','光照分析','准备创作'];

const Upload = {
  running:false, idx:-1, raf:0,
  start() {
    if (this.running) return;
    this.running = true; this.idx = -1;
    document.getElementById('createPanel').classList.remove('on');
    const pp = document.getElementById('processPanel');
    pp.classList.add('on');
    const ul = document.getElementById('prSteps');
    ul.innerHTML = STEPS.map(t => `<li><i></i>${t}</li>`).join('');
    const items = [...ul.children];
    const tick = () => {
      this.idx++;
      if (this.idx >= items.length) {
        this.running = false;
        setTimeout(() => {
          pp.classList.remove('on');
          document.getElementById('createPanel').classList.add('on');
        }, 1400);
        return;
      }
      items.forEach((el,i) => {
        el.className = i < this.idx ? 'done' : i === this.idx ? 'active' : '';
      });
      Agent.set(this.idx < 2 ? 'listening' : this.idx < 4 ? 'working' : 'speaking');
      setTimeout(tick, 780 + Math.random() * 420);
    };
    tick();
  },
};

/* ═══════════════════ 9 · 叙事控制器 ═══════════════════ */

const app = {
  progress:0, target:0, chapter:null, lastId:'', time:0,
  dark:false, darkMix:0, rollback:0,
  lightX:.455, lightY:.075, dragging:false,

  init() {
    Stage.init(document.getElementById('cv'));
    Agent.init(document.getElementById('wave'));
    this.buildIndex();
    this.bind();
    this.positionAgent(CHAPTERS[0]);
    this.jumpFromHash();
    this.loop(0);
  },

  /* 定位入口：
     ?p=0.45  → 直接渲染该进度（不改滚动位置，用于预览/校对）
     #relight → 滚动到某一章，便于分享 */
  jumpFromHash() {
    const q = new URLSearchParams(location.search);
    const qp = parseFloat(q.get('p'));
    if (!isNaN(qp)) {
      this.progress = this.target = clamp01(qp);
      this.lastId = '';
      return;
    }
    const id = (location.hash || '').replace('#', '');
    if (!id) return;
    const c = CHAPTERS.find(x => x.id === id);
    if (!c) return;
    const story = document.getElementById('story');
    const total = story.offsetHeight - window.innerHeight;
    if (total <= 0) return;
    const p = (c.s + c.e) / 2;
    this.target = this.progress = p;
    this.lastId = '';
    window.scrollTo(0, story.offsetTop + p * total);
  },

  positionAgent(ch) {
    const ag = document.getElementById('agentEl');
    ag.style.left = (ch.agent[0] * 100) + '%';
    ag.style.top  = (ch.agent[1] * 100) + '%';
  },

  /* 右侧章节索引 */
  buildIndex() {
    const el = document.getElementById('chapterIndex');
    el.innerHTML = CHAPTERS.map(c =>
      `<div class="ci-item" data-id="${c.id}"><span>${c.num} ${c.en}</span><i></i></div>`
    ).join('');
  },

  bind() {
    window.addEventListener('scroll', () => this.onScroll(), { passive:true });
    window.addEventListener('resize', () => Stage.resize());

    // 索引点击 → 跳转
    document.getElementById('chapterIndex').addEventListener('click', e => {
      const it = e.target.closest('.ci-item'); if (!it) return;
      const c = CHAPTERS.find(x => x.id === it.dataset.id); if (!c) return;
      const story = document.getElementById('story');
      const total = story.offsetHeight - window.innerHeight;
      window.scrollTo({ top: story.offsetTop + (c.s + c.e) / 2 * total, behavior:'smooth' });
    });

    // 光源拖拽
    const grab = document.getElementById('lightGrab');
    const move = e => {
      if (!this.dragging) return;
      const p = e.touches ? e.touches[0] : e;
      const F = Stage.F;
      this.lightX = clamp01((p.clientX - F.x) / F.w);
      this.lightY = clamp01((p.clientY - F.y) / F.h);
    };
    grab.addEventListener('mousedown', e => { this.dragging = true; grab.classList.add('dragging'); e.preventDefault(); });
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', () => { this.dragging = false; grab.classList.remove('dragging'); });
    grab.addEventListener('touchstart', e => { this.dragging = true; grab.classList.add('dragging'); }, { passive:true });
    window.addEventListener('touchmove', move, { passive:true });
    window.addEventListener('touchend', () => { this.dragging = false; grab.classList.remove('dragging'); });

    // 虚拟拖拽：直接点画框也能移动光
    document.getElementById('stage').addEventListener('pointerdown', e => {
      if (!this.chapter || this.chapter.id !== 'relight') return;
      if (e.target.closest('.agent,.chapter-panel,.light-grab,.critic-panel,.create-panel')) return;
      this.dragging = true; move(e);
    });

    // 上传交互
    document.getElementById('drop').addEventListener('click', () => Upload.start());
    document.getElementById('demoBtn').addEventListener('click', () => Upload.start());
  },

  onScroll() {
    const story = document.getElementById('story');
    const total = story.offsetHeight - window.innerHeight;
    if (total <= 0) { this.target = 1; return; }
    const y = window.scrollY - story.offsetTop;
    this.target = clamp01(y / total);
  },

  chapterAt(p) {
    for (const c of CHAPTERS) if (p >= c.s && p <= c.e) return c;
    return p < CHAPTERS[0].s ? CHAPTERS[0] : CHAPTERS[CHAPTERS.length - 1];
  },

  /* ── 主循环 ── */
  loop(t) {
    this.time = t;

    // 阻尼跟随：滚动有惯性，但不产生“延迟感”
    const d = this.target - this.progress;
    this.progress += Math.abs(d) < .0004 ? d : d * .19;
    const p = this.progress;

    const ch = this.chapterAt(p);
    const lp = clamp01((p - ch.s) / (ch.e - ch.s));

    /* 场景状态 */
    let s = sampleTrack(p);
    s.raw = p;

    /* CRITIC：视觉回放（不是真的跳回滚动位置） */
    const isCritic = ch.id === 'critic';
    const cStage = isCritic ? CRITIC.stage(lp) : null;
    if (isCritic) {
      if (cStage === 'reroll') {
        const m = Math.sin(clamp01((lp - .42) / .18) * Math.PI);
        this.rollback = m;
      } else if (cStage === 'repair') {
        this.rollback = Math.max(0, this.rollback - .06);
      } else {
        this.rollback = 0;
      }
      const r = this.rollback;
      // 回退到 SHADOW 阶段的视觉状态
      s.shadow = lerp(s.shadow, .34, r * .8);
      s.lightI = lerp(s.lightI, .95, r * .5);
      s.warmG  = lerp(s.warmG, .44, r * .6);
      s.desat  = lerp(s.desat, .17, r * .7);
    } else {
      this.rollback = 0;
    }

    /* 深色章 */
    const wantDark = isCritic || (ch.id === 'final' && lp < .35);
    if (wantDark !== this.dark) {
      this.dark = wantDark;
      document.body.classList.toggle('is-dark', wantDark);
    }
    s.dark = wantDark;

    /* CREATE：画面淡出 */
    if (ch.id === 'create') s.subj *= (1 - easeOut(clamp01(lp / .45))) * .92 + .08;
    if (ch.id === 'final')  s.subj *= lerp(1, .96, lp);

    /* 深浅章连续过渡 */
    this.darkMix += ((this.dark ? 1 : 0) - this.darkMix) * .10;
    s.darkMix = this.darkMix;

    /* 光位：拖拽优先，否则沿轨道自动推进（太阳自上而下移至左上方） */
    const autoLX = lerp(.455, .215, clamp01((p - .28) / .24));
    const autoLY = lerp(.075, .255, clamp01((p - .28) / .24));
    this.lightX += (autoLX - this.lightX) * (this.dragging ? 0 : .06);
    this.lightY += (autoLY - this.lightY) * (this.dragging ? 0 : .06);
    s.lx = this.lightX;
    s.ly = this.lightY;

    /* 受光方向由光源位置反推 —— 拖动光源时主体受光面实时跟随 */
    const autoAngle = Math.atan2(this.lightY - .48, this.lightX - .50);
    s.angle = lerp(s.angle, autoAngle, clamp01((p - .26) / .07));

    /* 渲染 */
    Stage.render(s, t);
    Agent.draw(t);

    /* DOM */
    this.syncDOM(ch, lp, p, s, cStage);

    requestAnimationFrame(nt => this.loop(nt));
  },

  /* 覆盖光晕中心（在 Stage 之后补一层，保证拖拽即时响应） */
  paintLightOverride(s) {
    if (s._lx === undefined) return;
    const F = Stage.F, ctx = Stage.ctx;
    const lx = F.x + F.w * s._lx, ly = F.y + F.h * s._ly;
    ctx.save();
    ctx.beginPath(); ctx.rect(F.x, F.y, F.w, F.h); ctx.clip();
    ctx.globalCompositeOperation = 'screen';
    const R = Math.max(F.w, F.h) * .58;
    const g = ctx.createRadialGradient(lx, ly, 0, lx, ly, R);
    g.addColorStop(0,  `rgba(255,242,216,${.40 * s.lightI})`);
    g.addColorStop(.3, `rgba(255,228,192,${.14 * s.lightI})`);
    g.addColorStop(1,  'rgba(255,220,180,0)');
    ctx.fillStyle = g;
    ctx.fillRect(F.x, F.y, F.w, F.h);
    ctx.restore();
  },

  /* ── DOM 同步 ── */
  syncDOM(ch, lp, p, s, cStage) {
    const $ = id => document.getElementById(id);

    /* 章节文字 */
    if (ch.id !== this.lastId) {
      this.lastId = ch.id;
      const panel = $('chapterPanel');
      panel.classList.add('out');
      setTimeout(() => {
        $('chNum').textContent = ch.num;
        $('chEn').textContent = ch.en;
        $('chZh').textContent = ch.zh;
        $('chDesc').textContent = ch.desc;
        $('chMeta').textContent = ch.meta;
        panel.classList.toggle('is-hero', ch.id === 'hero');
        panel.classList.remove('out');
      }, 200);

      /* Agent 位置 */
      this.positionAgent(ch);
      $('agentEl').classList.toggle('hidden',
        (ch.id === 'final' && lp > .5) || ch.id === 'create');

      /* Agent 状态 */
      const st = ch.id === 'critic' ? 'checking'
        : ch.id === 'create' ? 'speaking'
        : ch.id === 'final' ? 'done'
        : ch.id === 'hero' ? 'idle'
        : ch.id === 'plan' ? 'listening' : 'speaking';
      Agent.set(st);
      $('orb').classList.toggle('hollow', st === 'checking' || st === 'thinking');
      $('orb').classList.toggle('pulse', st === 'idle');

      /* 台词 */
      const cap = $('caption');
      cap.classList.remove('on');
      cap.textContent = ch.caption;
      setTimeout(() => cap.classList.add('on'), 320);

      /* 索引高亮 */
      [...$('chapterIndex').children].forEach(el =>
        el.classList.toggle('on', el.dataset.id === ch.id));
    }

    /* 完成度条 */
    const active = p > .02 && p < .995;
    $('compositionBar').classList.toggle('on', active);
    const pct = Math.round(clamp01((p - .065) / (.865 - .065)) * 100);
    $('cbFill').style.width = pct + '%';
    $('cbDot').style.left = pct + '%';
    $('cbPct').textContent = pct + '%';

    /* 顶栏在叙事中隐藏 */
    $('siteHead').classList.toggle('is-hidden', p > .07 && p < .93);

    /* 技术读数 */
    const ro = $('readout');
    const showRO = ['matting','light','relight','shadow','harmonize','enhance'].indexOf(ch.id) >= 0;
    ro.classList.toggle('on', showRO);
    if (showRO) {
      $('roModule').textContent = ch.tool;
      $('roDir').textContent = Math.round(((s.angle * 180 / Math.PI) + 360) % 360) + '°';
      $('roTemp').textContent = Math.round(lerp(5200, 3400, s.warm)) + 'K';
      $('roAlpha').textContent = (ch.id === 'matting' ? lerp(62, 98.2, easeOut(lp)) : 98.2).toFixed(1) + '%';
    }

    /* PLAN 路径 */
    $('planPath').classList.toggle('on', ch.id === 'plan');
    if (ch.id === 'plan') {
      const nodes = [...$('planPath').querySelectorAll('.pp-node')];
      const lit = Math.floor(clamp01((lp - .1) / .7) * nodes.length);
      nodes.forEach((n,i) => n.classList.toggle('on', i < lit));
    }

    /* 光源拖拽把手 */
    const showGrab = ch.id === 'relight';
    const grab = $('lightGrab');
    grab.classList.toggle('on', showGrab);
    if (showGrab) {
      grab.style.left = (this.lightX * 100) + '%';
      grab.style.top  = (this.lightY * 100) + '%';
    }

    /* CRITIC 面板 */
    const cp = $('criticPanel');
    const showCP = ch.id === 'critic' && lp > .04;
    cp.classList.toggle('on', showCP);
    if (showCP) this.syncCritic(lp, cStage);

    /* FINAL 徽标 */
    $('finalBadge').classList.toggle('on', ch.id === 'final' && lp > .25);

    /* CREATE 面板 */
    const showCreate = ch.id === 'create' && lp > .3 && !Upload.running;
    $('createPanel').classList.toggle('on', showCreate);
  },

  /* CRITIC 分镜同步 */
  syncCritic(lp, st) {
    const $ = id => document.getElementById(id);
    const dims = [...document.querySelectorAll('.cp-dim')];
    const keys = ['lighting','shadow','color','edge'];

    // 逐个亮起
    const revealed = lp < .30 ? Math.floor(clamp01((lp - .06) / .20) * 4) : 4;
    dims.forEach((el, i) => {
      const on = i < revealed;
      el.classList.toggle('on', on);
      if (!on) return;
      const k = keys[i];
      const finalScore = (st === 'enter' || st === 'scan' || st === 'fail') ? CRITIC.SCORES[k]
        : (st === 'reroll') ? CRITIC.SCORES[k]
        : CRITIC.FIXED[k] !== undefined ? CRITIC.FIXED[k] : CRITIC.SCORES[k];
      el.querySelector('.cp-score').textContent = finalScore;
      el.querySelector('.cp-bar i').style.right = (100 - finalScore) + '%';
      el.classList.toggle('fail', k === 'shadow' && (st === 'fail' || st === 'reroll'));
    });

    const stateEl = $('cpState');
    const verdict = $('cpVerdict');
    const reroll = $('cpReroll');

    if (st === 'enter' || st === 'scan') {
      stateEl.textContent = 'CHECKING'; stateEl.classList.remove('fail');
      $('cpOverall').textContent = '—';
      verdict.textContent = ''; verdict.className = 'cp-verdict';
      reroll.classList.remove('on'); reroll.textContent = '';
    } else if (st === 'fail' || st === 'reroll') {
      stateEl.textContent = 'BELOW THRESHOLD'; stateEl.classList.add('fail');
      $('cpOverall').textContent = CRITIC.overall(CRITIC.SCORES).toFixed(1);
      verdict.textContent = 'shadow · threshold 80';
      verdict.className = 'cp-verdict fail';
      if (st === 'reroll') {
        reroll.classList.add('on');
        reroll.textContent = 'reroll → T05 shadow_generate';
      }
    } else {
      stateEl.textContent = 'RECHECKING'; stateEl.classList.remove('fail');
      const sc = Object.assign({}, CRITIC.SCORES, CRITIC.FIXED);
      $('cpOverall').textContent = CRITIC.overall(sc).toFixed(1);
      if (st === 'pass') {
        stateEl.textContent = 'PASSED';
        verdict.textContent = 'ready';
        verdict.className = 'cp-verdict pass';
        reroll.classList.remove('on'); reroll.textContent = '';
      } else {
        verdict.textContent = 'shadow resolved';
        verdict.className = 'cp-verdict';
        reroll.classList.remove('on'); reroll.textContent = '';
      }
    }
  },
};

/* ═══════════════════ 10 · 启动 ═══════════════════ */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => app.init());
} else {
  app.init();
}
window.__app = app;
})();

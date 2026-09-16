/* ══════════════════════════════════════════════════════════════════════
   ImageCompose 首页 v2 · Editorial Studio
   单一 Sticky ImageStage：滚动只驱动图层参数与舞台几何 morph。
   原生滚动，不劫持滚轮；全部渲染收敛在 rAF。
   ══════════════════════════════════════════════════════════════════════ */
(function () {
'use strict';

var clamp = function (v, a, b) { return v < a ? a : (v > b ? b : v); };
var lerp = function (a, b, t) { return a + (b - a) * t; };
/* 线性段映射：p 落在 [a,b] → 0..1（平滑） */
var seg = function (p, a, b) {
  if (b <= a) return p >= b ? 1 : 0;
  var t = clamp((p - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
var easeOut = function (t) { return 1 - Math.pow(1 - t, 3); };

var $ = function (sel, root) { return (root || document).querySelector(sel); };
var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

/* ═══════════════════ 1 · 章节表（len 单位 = vh 的滚动行程） ═══════════════════ */

var CHS = [
  { ch: 1,  len: 110 },
  { ch: 2,  len: 150 },
  { ch: 3,  len: 175 },
  { ch: 4,  len: 100 },
  { ch: 5,  len: 240 },
  { ch: 6,  len: 120 },
  { ch: 7,  len: 200 },
  { ch: 8,  len: 150 },
  { ch: 9,  len: 330, dark: true },
  { ch: 10, len: 130 },
  { ch: 11, len: 100 },
  { ch: 12, len: 140 },
  { ch: 13, len: 115 }
];
var TOTAL = 0;
CHS.forEach(function (c) { c.start = TOTAL; TOTAL += c.len; });   // start 单位 vh
var byCh = {};
CHS.forEach(function (c) { byCh[c.ch] = c; });

/* ═══════════════════ 2 · 元素缓存 ═══════════════════ */

var showcase = $('#showcase');
var frame = $('#frame');
var stage = $('#stage');
var stageBg = $('#stageBg');
var rails = {};        $$('.rail', frame).forEach(function (r) { rails[r.dataset.ch] = r; });
var layers = {};       $$('.layer', frame).forEach(function (l) { layers[l.dataset.ch] = l; });
var finalHead = $('.final-head');
var finalSub = $('.final-sub');
var body = document.body;

var idxHost = $('.chapter-index');
var fileInput = $('#fileInput');

/* ═══════════════════ 3 · 布局测量 ═══════════════════ */

var M = {
  vh: 900, vw: 1440, fw: 1264,
  scTop: 0, scLenPx: 1,
  mobile: false
};

function measure() {
  M.vh = window.innerHeight;
  M.vw = window.innerWidth;
  M.fw = frame.clientWidth;
  M.mobile = M.vw <= 920;
  M.scTop = showcase.offsetTop;
  M.scLenPx = showcase.offsetHeight - M.vh;
}

function setSize() {
  showcase.style.height = 'calc(' + (TOTAL + 100) + 'vh)';
}

/* 舞台几何（frame 内坐标系，单位 px） */
function geo(ch) {
  var s = Math.min(1, M.vh / 860, M.fw / 1264);
  var railW = Math.min(289, M.fw * 0.229);
  var w, h, x, top;

  if (M.mobile) {
    x = 0;
    w = M.fw;
    var base = { 5: 0.75, 6: 0.482, 10: 0.64 }[ch] || 0.635;
    h = Math.round(Math.min(w * base, M.vh * 0.55));
    if (ch === 10) h = Math.round(Math.min(w * 0.64, M.vh * 0.6));
    return { x: x, top: 0, w: w, h: h, rad: 16 };
  }

  switch (ch) {
    case 5:   /* RELIGHT 视觉高潮：图占 ~70% */
      x = Math.round(182 * s); w = M.fw - x;
      h = Math.round(Math.min(656 * s, M.vh - 104));
      top = Math.round((M.vh - h) / 2 - M.vh * 0.012);
      break;
    case 6:   /* GROUND 976×470 特写横幅 */
      x = Math.round(railW); w = M.fw - x;
      h = Math.round(Math.min(470 * s, (M.vh - 160) * 0.8));
      top = Math.round((M.vh - h) / 2 - M.vh * 0.012);
      break;
    case 10:  /* FINAL 满宽静屏（顶部避开页眉与副标） */
      top = Math.max(120, Math.round(M.vh * 0.095) + 84);
      w = M.fw; x = 0;
      h = Math.round(Math.min(760 * s, M.vh - top - 22));
      break;
    default:
      x = Math.round(railW); w = M.fw - x;
      h = Math.round(Math.min(620 * s, M.vh - 136));
      top = Math.round((M.vh - h) / 2 - M.vh * 0.012);
  }
  return { x: x, top: Math.max(16, top), w: w, h: h, rad: 22 };
}

function applyGeometry(g) {
  if (M.mobile) {
    stage.style.setProperty('--mh', g.h + 'px');
    return;
  }
  stage.style.setProperty('--sx', g.x + 'px');
  stage.style.setProperty('--sw', g.w + 'px');
  stage.style.setProperty('--sh', g.h + 'px');
  stage.style.setProperty('--sy', g.top + 'px');
  stage.style.setProperty('--st', '0px');
  stage.style.setProperty('--srad', g.rad + 'px');
}

function mixGeo(a, b, t) {
  return {
    x: lerp(a.x, b.x, t), top: lerp(a.top, b.top, t),
    w: lerp(a.w, b.w, t), h: lerp(a.h, b.h, t),
    rad: lerp(a.rad, b.rad, t)
  };
}

/* ═══════════════════ 4 · 各章渲染器 ═══════════════════ */

var planLine = $('#planLine');
var planNodes = $$('#planNodes circle');
var planSteps = $$('.plan-steps span');
var ph02bg = $('.ph02-bg');
var ph01 = $('.ph01');
var ph02sub = $('.ph02-sub');
var bbox = $('.bbox');
var scanline = $('.scanline');
var alphaVal = $('#alphaVal');
var sceneReveal = $('#sceneReveal');
var ph03sub = $('.ph03-sub');
var lightCursor4 = $('[data-ch="4"] .light-cursor');
var lightDeg = $('#lightDeg');
var vidRelight = $('#vidRelight');
var cursor5 = $('[data-ch="5"] .light-cursor');
var feetShadow = $('#feetShadow');
var harmonTop = $('#harmonTop');
var harmWords = $$('.harm-words span');
var macros = $$('.macro');
var vidCritic = $('#vidCritic');
var cpRows = $$('.cp-row');
var cpShadow = $('#cpShadow');
var cpFlag = $('#cpFlag');
var cpPill = $('#cpPill');
var finalScore = $('#finalScore');
var bubblePlan = $('.bubble--plan');
var bubbleCompose = $('.bubble--compose');
var bubbleFinal = $('.bubble--final');
var bubbleAsk = $('.bubble--ask');
var bubbleGround = $('.bubble--ground');
var tags3 = $$('.pill--tag');
var turnBubbles = $$('.turn-chat .bubble');
var turnHint = $('.turn-hint');
var turnRows = $$('.plan-rows > div');
var btnReplay = $('#btnReplay');
var heroImg = $('#heroImg');
var ph02Edge = $('#ph02Edge');
var planLabel = $('.plan-label');
var tempPill = $('#tempPill');
var tempK = $('#tempK');
var lightPill = $('.pill--light');
var relightGlow = $('#relightGlow');
var lightDrag = $('#lightDrag');
var dragHint = $('#dragHint');
var criticSay = $('#criticSay');
var cpLight = $('#cpLight'), cpColor = $('#cpColor'), cpEdge = $('#cpEdge'), cpThr = $('#cpThr');
var composeBar = $('#composeBar');
var cbFill = $('#cbFill');
var upOk = $('#upOk');
var upBar = $('#upBar');
var upList = $$('#upList li');
var upLive = $('#upLive');

function setBubble(el, on) {
  if (el) el.classList.toggle('is-on', !!on);
}
/* 懒加载：进入章节才真正拉取视频（HTML 上 preload=none） */
function ensureVideo(v) {
  if (!v || v.dataset.armed) return;
  v.dataset.armed = '1';
  v.preload = 'auto';
  v.load();
}
function vidDur(video, fallback) {
  return (video && isFinite(video.duration) && video.duration > 0) ? video.duration : fallback;
}
function scrub(video, t) {
  if (!video || video.readyState < 1) return;
  var d = video.duration || 0;
  if (!isFinite(d) || d <= 0) return;
  var target = clamp(t, 0, d - 0.05);
  if (video.paused !== true) video.pause();
  if (Math.abs(video.currentTime - target) > 0.033) video.currentTime = target;
}

/* 主体在舞台里的两档摆位：A=照片内原位（02 开场），B=独占展示 */
function subjectPos(stW, stH, mode, arArg) {
  var ar = arArg || (1024 / 1536);            // 主体宽高比（job 时按真实图片）
  if (mode === 'A') {
    var ph = stH;                             // 照片以高度撑满（contain）
    var pw = ph * ar;
    var px0 = (stW - pw) / 2;
    return { l: px0, t: 0, h: ph, w: pw };
  }
  var h2 = stH * 0.985;
  var w2 = h2 * ar;
  return { l: stW * 0.585 - w2 / 2, t: (stH - h2) / 2, h: h2, w: w2 };
}

var RENDER = {
  /* 01 UNDERSTAND：先出现“你说了什么”，Agent 回应后计划路径才逐步形成。
     设计稿语义：计划成形时原片褪色（~0.42），路径与步骤标签压在照片上仍清晰可读 */
  1: function (p) {
    if (ph01) ph01.style.opacity = lerp(0.94, 0.42, seg(p, .3, .62)).toFixed(3);
    setOn(bubbleAsk, p > .06);
    setOn(planLabel, p > .4);
    if (planLine) planLine.style.strokeDashoffset = (744 * (1 - seg(p, .42, .8))).toFixed(1);
    planNodes.forEach(function (n, i) {
      var on = p > .48 + i * .075;
      if (n.classList.contains('on') !== on) n.classList.toggle('on', on);
    });
    planSteps.forEach(function (s, i) {
      var on = p > .5 + i * .075;
      if (s.classList.contains('on') !== on) s.classList.toggle('on', on);
    });
    setBubble(bubblePlan, p > .88);
  },

  /* 02 EXTRACT：背景退场 → 主体独立 → 极细描边沿扫描渐显 → 边框退场 */
  2: function (p, st) {
    var isJob = STORY && STORY.mode === 'job';
    var ar = (ph02sub.naturalWidth && ph02sub.naturalHeight)
      ? ph02sub.naturalWidth / ph02sub.naturalHeight : (1024 / 1536);
    var bgOut = 1 - seg(p, .05, .48);
    ph02bg.style.opacity = bgOut.toFixed(3);
    var a = subjectPos(st.w, st.h, 'A', ar);
    var b = subjectPos(st.w, st.h, 'B', ar);
    var t = seg(p, .48, .72);
    var l = lerp(a.l, b.l, t), tp = lerp(a.t, b.t, t);
    var hh = lerp(a.h, b.h, t), ww = lerp(a.w, b.w, t);
    ph02sub.style.left = l + 'px';
    ph02sub.style.top = tp + 'px';
    ph02sub.style.height = hh + 'px';
    ph02sub.style.width = ww + 'px';
    var sp = seg(p, .58, .9);
    if (!isJob) {
      /* 描边层与主体同摆位，随扫描线自上而下渐显（demo 专属素材） */
      ph02Edge.style.left = l + 'px';
      ph02Edge.style.top = tp + 'px';
      ph02Edge.style.height = hh + 'px';
      ph02Edge.style.width = ww + 'px';
      var edgeOn = seg(p, .56, .62) * (1 - seg(p, .88, .97));
      ph02Edge.style.opacity = edgeOn.toFixed(3);
      ph02Edge.style.clipPath = 'inset(0 0 ' + ((1 - sp) * 100).toFixed(2) + '% 0)';
    }
    var bo = seg(p, .55, .64) * (1 - seg(p, .84, .94));
    bbox.style.opacity = bo.toFixed(3);
    var pad = 8 + 6 * (1 - easeOut(seg(p, .55, .7)));
    var bl = clamp(l - pad, 8, st.w - 8);
    var bt = clamp(tp - pad, 8, st.h - 8);
    var br = clamp(l + ww + pad, 8, st.w - 8);
    var bb = clamp(tp + hh + pad, 8, st.h - 8);
    bbox.style.left = bl + 'px';
    bbox.style.top = bt + 'px';
    bbox.style.width = (br - bl) + 'px';
    bbox.style.height = (bb - bt) + 'px';
    scanline.style.opacity = sp > 0 && sp < 1 ? '1' : '0';
    scanline.style.top = (st.h * (0.06 + 0.88 * sp)) + 'px';
    alphaVal.textContent = (62 + 36.2 * seg(p, .5, .95)).toFixed(1) + '%';
    var pill2 = alphaVal.parentElement;
    pill2.style.opacity = seg(p, .48, .6).toFixed(3);
  },

  /* 03 COMPOSE：场景从地平线带状逐层长出（job 时主体按 cover 数学对位） */
  3: function (p, st) {
    var keys = [[0, 46, 46], [.3, 16, 32], [.62, 3, 9], [1, 0, 0]];
    var t = p, i = 0;
    while (i < keys.length - 2 && t > keys[i + 1][0]) i++;
    var k0 = keys[i], k1 = keys[i + 1];
    var tt = clamp((t - k0[0]) / (k1[0] - k0[0]), 0, 1);
    var top = lerp(k0[1], k1[1], tt), bot = lerp(k0[2], k1[2], tt);
    sceneReveal.style.clipPath = 'inset(' + top.toFixed(2) + '% 0% ' + bot.toFixed(2) + '% 0%)';
    if (STORY && STORY.mode === 'job' && ph03sub) {
      var arJ = (ph03sub.naturalWidth && ph03sub.naturalHeight)
        ? ph03sub.naturalWidth / ph03sub.naturalHeight : (1024 / 1536);
      var posJ = subjectPos(st.w, st.h, 'A', arJ);
      ph03sub.style.left = posJ.l + 'px';
      ph03sub.style.top = posJ.t + 'px';
      ph03sub.style.height = posJ.h + 'px';
      ph03sub.style.width = posJ.w + 'px';
      ph03sub.style.transform = 'none';
      ph03sub.style.opacity = seg(p, .5, .66).toFixed(3);
    }
    tags3.forEach(function (tag) {
      setOn(tag, p > parseFloat(tag.dataset.beat));
    });
    setBubble(bubbleCompose, p > .82);
  },

  /* 04 ILLUMINATE：参数渐进出现（光向 → 角度 → 色温；job 用真实数值） */
  4: function (p) {
    setOn(lightCursor4, p > .28);
    lightPill.style.opacity = seg(p, .32, .42).toFixed(3);
    lightDeg.textContent = Math.round(LIGHT_TARGETS.deg * seg(p, .45, .68));
    tempPill.style.opacity = seg(p, .72, .82).toFixed(3);
    tempK.textContent = Math.round(LIGHT_TARGETS.temp * seg(p, .74, .96));
  },

  /* 05 RELIGHT：demo 滚动读视频帧；job 为真实节点输出图（无 scrub） */
  5: function (p) {
    setOn(cursor5, p > .12);
    if (!(STORY && STORY.mode === 'job')) {
      ensureVideo(vidRelight);
      scrub(vidRelight, seg(p, .04, .96) * vidDur(vidRelight, 3.4));
    }
  },

  /* 06 GROUND：接触阴影出现 → 停（demo= feet 对；job= composited 溶解） */
  6: function (p) {
    var topEl = (STORY && STORY.mode === 'job') ? document.getElementById('groundTop') : feetShadow;
    if (topEl) topEl.style.opacity = seg(p, .06, .42).toFixed(3);
    setBubble(bubbleGround, p > .55);
  },

  /* 07 HARMONIZE：全图极慢统一 */
  7: function (p) {
    harmonTop.style.opacity = seg(p, .04, .8).toFixed(3);
    harmWords.forEach(function (w, i) {
      w.classList.toggle('lit', p > .3 + i * .18);
    });
  },

  /* 08 ENHANCE：demo 三联微距交错入场；job 全帧增强前→后 */
  8: function (p) {
    if (STORY && STORY.mode === 'job') {
      var eTop = document.getElementById('enhTop');
      if (eTop) eTop.style.opacity = seg(p, .1, .7).toFixed(3);
      return;
    }
    macros.forEach(function (m, i) {
      setOn(m, p > .1 + i * .12);
    });
  },

  /* 09 CRITIC：demo= 分数动画+视频；job= 真实 critic_history 轮次回放 */
  9: function (p) {
    setOn(criticSay, p > .03 && p < .9);
    var rounds = (STORY && STORY.mode === 'job' && STORY.critic && STORY.critic.length)
      ? STORY.critic : null;
    if (rounds) {
      var idx = Math.min(rounds.length - 1, Math.floor(p * rounds.length * 0.999));
      var r = rounds[idx];
      cpThr.textContent = r.threshold;
      cpLight.textContent = Math.round(r.scores.lighting);
      cpShadow.textContent = Math.round(r.scores.shadow);
      cpColor.textContent = Math.round(r.scores.color);
      cpEdge.textContent = Math.round(r.scores.edge);
      cpRows.forEach(function (row, i) { setOn(row, p > .06 + i * .075); });
      var rowFail = !r.passed && 'shadow' === r.lowest_dim;
      cpFlag.style.opacity = (rowFail && p > .18) ? '1' : '0';
      cpPill.classList.toggle('pass', r.passed);
      var key = idx + ':' + (r.passed ? 'p' : 'f');
      if (cpPill.dataset.mode !== key) {
        cpPill.dataset.mode = key;
        cpPill.textContent = r.passed
          ? 'OVERALL ' + Math.round(r.overall) + ' ≥ ' + r.threshold + ' · PASS'
          : (r.comment || (('SHADOW ' + Math.round(r.scores.shadow)) + ' < ' + r.threshold +
             '\u00a0\u00a0→\u00a0\u00a0rerun ' + (r.rerun_role || '')));
      }
      setOn(cpPill, p > .4);
      return;
    }
    cpRows.forEach(function (r, i) { setOn(r, p > .06 + i * .075); });
    var repair = seg(p, .87, .95);
    if (repair > 0) {
      cpShadow.textContent = Math.round(lerp(78, 93, repair));
    } else {
      cpShadow.textContent = '78';
    }
    var pass = p > .955;
    cpFlag.style.opacity = pass || p < .18 ? '0' : '1';
    cpPill.classList.toggle('pass', pass);
    if (pass && cpPill.dataset.mode !== 'pass') {
      cpPill.dataset.mode = 'pass';
      cpPill.textContent = 'SHADOW 93 ≥ 80 · PASS';
    } else if (!pass && cpPill.dataset.mode !== 'fail') {
      cpPill.dataset.mode = 'fail';
      cpPill.textContent = 'SHADOW 78 < 80\u00a0\u00a0→\u00a0\u00a0reroll T04 回到 GROUND 重跑';
    }
    setOn(cpPill, p > .4);
    ensureVideo(vidCritic);
    scrub(vidCritic, seg(p, .48, .86) * vidDur(vidCritic, 5.6));
  },

  /* 10 FINAL：静屏，评分计数（job= 真实 overall） */
  10: function (p) {
    finalScore.textContent = Math.round(FINAL_TARGET * seg(p, .2, .66));
    setBubble(bubbleFinal, p > .5);
  },

  /* 11 CREATE：交互在事件层（波形独立动画） */
  11: function () {},

  /* 12 UPLOAD：REPLAY(job)= 全部完成态；演示按滚动节拍推进 */
  12: function (p) {
    var jobStory = STORY && STORY.mode === 'job';
    if (upLive) upLive.hidden = true;        /* LIVE 徽标只在 RUN 面板出现 */
    setOn(upOk, p > .06 || jobStory);
    if (jobStory) {
      upBar.style.width = '100%';
      upList.forEach(function (li) {
        li.classList.add('is-done');
        li.classList.remove('is-doing');
        li.querySelector('.li-status').textContent = '完成';
      });
      return;
    }
    setOn(upOk, p > .06);
    upBar.style.width = (seg(p, .08, .36) * 100).toFixed(1) + '%';
    var done = [.3, .46, .62], doing = .78;
    upList.forEach(function (li, i) {
      var isDone = i < 3 && p > done[i];
      var isDoing = i === 3 ? p > doing : (!isDone && p > (i === 0 ? 0 : done[i - 1]));
      li.classList.toggle('is-done', isDone);
      li.classList.toggle('is-doing', isDoing && !isDone);
      var st = li.querySelector('.li-status');
      if (isDone) st.textContent = '完成';
      else if (isDoing) st.textContent = '进行中';
      else st.textContent = i === 3 ? '—' : '排队中';
    });
  },

  /* 13 YOUR TURN：对话 + 计划卡 */
  13: function (p) {
    setOn(turnBubbles[0], p > .14);
    setOn(turnBubbles[1], p > .34);
    setOn(turnHint, p > .48);
    turnRows.forEach(function (r, i) {
      var on = p > .5 + i * .06;
      r.style.opacity = on ? '1' : '0';
      r.style.transform = on ? 'none' : 'translateY(6px)';
    });
    setOn(btnReplay, p > .72);
  }
};

function setOn(el, on) {
  if (el && el.classList.contains('is-on') !== !!on) el.classList.toggle('is-on', !!on);
}

/* ═══════════════════ 5 · 主循环 ═══════════════════ */

var reduceMotion = false;
try { reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) {}
var DAMP = reduceMotion ? 1 : 0.19;

var curP = 0;            /* 阻尼后的 showcase 进度 0..1 */
var lastK = -1;

function activeIndex(uVh) {
  for (var i = CHS.length - 1; i >= 0; i--) {
    if (uVh >= CHS[i].start) return i;
  }
  return 0;
}

function tick() {
  var raw = clamp((window.scrollY - M.scTop) / Math.max(1, M.scLenPx), 0, 1);
  curP += (raw - curP) * DAMP;
  if (Math.abs(raw - curP) < 0.0004) curP = raw;

  var uVh = curP * TOTAL;                    /* 已行走的 vh */
  var k = activeIndex(uVh);
  var c = CHS[k];
  var lp = clamp((uVh - c.start) / c.len, 0, 1);

  /* —— HERO：照片从灰雾渐清（低饱和 → 正常 → 主体明确） —— */
  if (heroImg) {
    var hp = clamp(window.scrollY / (M.vh * 0.7), 0, 1);
    heroImg.style.filter = 'saturate(' + (0.5 + 0.5 * hp).toFixed(3) + ') brightness(' +
      (1.06 - 0.06 * hp).toFixed(3) + ') contrast(' + (0.96 + 0.09 * hp).toFixed(3) + ')';
    heroImg.style.transform = 'scale(' + (1.045 - 0.045 * hp).toFixed(4) + ')';
  }

  /* —— 全页创作进度条（仅 INTRO 演示滚动时显示；REPLAY 用回放轴） —— */
  if (composeBar) {
    var barEnd = CHS[9].start + CHS[9].len;               /* FINAL 章结束 */
    var showBar = AppState === 'intro' && window.scrollY > M.vh * 0.85 && uVh < barEnd + 4;
    setOn(composeBar, showBar);
    if (showBar || composeBar.classList.contains('is-on')) {
      cbFill.style.left = (clamp(uVh / barEnd, 0, 1) * 100).toFixed(2) + '%';
    }
  }

  /* —— RUN：真实状态低频更新，视觉状态 60fps 插值 —— */
  if (AppState === 'run' && Job) {
    Job.viz = Job.viz === undefined ? 0 : Job.viz;
    Job.viz += (Job.vizTarget - Job.viz) * 0.12;
    runFillEl.style.left = 'auto';
    runFillEl.style.width = '100%';
    runFillEl.style.transform = 'scaleX(' + clamp(Job.viz, 0, 1).toFixed(4) + ')';
    runFillEl.style.transformOrigin = 'left';
    runElapsed.textContent = ((performance.now() - Job.startedPerf) / 1000).toFixed(1) + 's';
  }

  /* —— REPLAY：回放轴填充 + 自动播放 —— */
  if (AppState === 'replay') {
    if (autoplay) {
      var step = M.vh * 0.09 * (1000 / 60);   /* ≈90vh/s */
      window.scrollTo(0, window.scrollY + step);
      if (window.scrollY >= M.scTop + M.scLenPx - 2) {
        autoplay = false;
        rpPlay.textContent = 'PLAY';
      }
    }
    var rpFrac = clamp(uVh / TOTAL, 0, 1);
    rpFillEl.style.width = (rpFrac * 100).toFixed(2) + '%';
    rpDot.style.left = (rpFrac * 100).toFixed(2) + '%';
    rpLabel.textContent = String(k < 9 ? '0' + (k + 1) : (k + 1)) + '\u00a0 ' + (CH_NAMES[c.ch] || '');
  }

  /* —— 溶解带宽（章节边界左右各一半） —— */
  var F = Math.min(7, c.len * 0.18);           /* 溶解带宽 vh */
  var half = F / 2;

  /* —— 舞台几何：与下一章图层的淡入严格同步，杜绝提前 morph —— */
  var g = geo(c.ch);
  if (k < CHS.length - 1) {
    var s1g = c.start + c.len;
    var nextOp = clamp((uVh - (s1g - half)) / F, 0, 1);
    if (nextOp > 0) g = mixGeo(g, geo(CHS[k + 1].ch), easeOut(nextOp));
  }
  applyGeometry(g);

  /* —— 图层可见性（边界交叉溶解） —— */
  var op8 = 0;                                  /* ENHANCE：舞台底色需退成纸色 */
  for (var name in layers) {
    var cc = byCh[name];
    var s0 = cc.start, s1 = cc.start + cc.len;
    var opIn = clamp((uVh - (s0 - half)) / F, 0, 1);
    var opOut = clamp(((s1 + half) - uVh) / F, 0, 1);
    var op = Math.min(opIn, opOut);
    var el = layers[name];
    if (op <= 0.002) {
      if (el.classList.contains('is-on')) el.classList.remove('is-on');
      continue;
    }
    if (!el.classList.contains('is-on')) el.classList.add('is-on');
    el.style.opacity = op >= 1 ? '' : op.toFixed(3);
    if (name === '8') op8 = op;
    if (RENDER[name] && op > 0.05) {
      var l2 = clamp((uVh - s0) / cc.len, 0, 1);
      RENDER[name](l2, { w: g.w, h: g.h });
    }
  }
  stageBg.style.opacity = op8 > 0 ? (1 - op8).toFixed(3) : '';

  /* —— 左栏文案（200 出 / 420 入交给 CSS transition） —— */
  for (var name2 in rails) {
    var cc2 = byCh[name2];
    var on = uVh > cc2.start + 1 && uVh < cc2.start + cc2.len - 1;
    setOn(rails[name2], on);
  }
  var fhOn = k === 9;
  setOn(finalHead, fhOn);
  setOn(finalSub, fhOn);

  /* —— 深色章 —— */
  var dark = uVh > CHS[8].start - F * 0.6 && uVh < CHS[8].start + CHS[8].len + F * 0.4;
  body.classList.toggle('dark', !!dark);

  /* —— 英雄索引激活态（英雄屏内 00 常亮，进入 showcase 后跟随章节） —— */
  var act = window.scrollY < M.scTop - M.vh * 0.35 ? 0 : k + 1;
  if (lastK !== act) {
    lastK = act;
    var items = idxHost ? idxHost.children : [];
    for (var i2 = 0; i2 < items.length; i2++) {
      items[i2].classList.toggle('is-active', i2 === act);
    }
  }

  drawWaves();
  requestAnimationFrame(tick);
}

/* ═══════════════════ 6 · 波形（Agent 声音存在） ═══════════════════ */

var waves = $$('.wave').map(function (cv) {
  return { cv: cv, ctx: cv.getContext('2d'), mode: cv.dataset.wave || 'idle' };
});

function drawWaves() {
  var t = performance.now() / 1000;
  waves.forEach(function (w) {
    var ctx = w.ctx, cv = w.cv;
    var W = cv.width, H = cv.height, n = Math.floor(W / 6);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#7EA7B4';
    var amp = w.mode === 'listen' ? 0.95 : 0.72;
    for (var i = 0; i < n; i++) {
      var ph = Math.sin(i * 1.7 + t * 2.1) * Math.cos(i * 0.53 - t * 1.3);
      var hgt = Math.max(3, Math.abs(ph) * H * amp * (0.55 + 0.45 * Math.sin(i * 0.9 + 1)));
      var y = (H - hgt) / 2;
      ctx.fillRect(i * 6 + 2, y, 3, hgt);
    }
  });
}

/* ═══════════════════ 7 · 交互：上传 / 示例 / 重放 / 索引 ═══════════════════ */

/* 固定时长的动画滚动（原生 smooth 距离越长越慢，不可控） */
var scrollAnim = null;
function smoothScrollTo(target) {
  if (reduceMotion) { window.scrollTo(0, target); return; }
  if (scrollAnim) cancelAnimationFrame(scrollAnim.raf);
  var from = window.scrollY, t0 = performance.now(), D = 850;
  var ease = function (t) { return t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; };
  var stepFn = function () {
    var t = clamp((performance.now() - t0) / D, 0, 1);
    window.scrollTo(0, from + (target - from) * ease(t));
    if (t < 1) scrollAnim.raf = requestAnimationFrame(stepFn);
  };
  scrollAnim = { raf: requestAnimationFrame(stepFn) };
}

function goToCh(ch, atStart) {
  var c = byCh[ch];
  if (!c) return;
  var y = M.scTop + (c.start + (atStart ? 0 : c.len * 0.32)) / TOTAL * M.scLenPx;
  smoothScrollTo(Math.max(0, y));
}

function buildIndex() {
  if (!idxHost) return;
  var names = ['HERO', 'UNDERSTAND', 'EXTRACT', 'COMPOSE', 'ILLUMINATE', 'RELIGHT',
    'GROUND', 'HARMONIZE', 'ENHANCE', 'CRITIC', 'FINAL', 'CREATE', 'UPLOAD', 'YOUR TURN'];
  names.forEach(function (lb, i) {
    var a = document.createElement('a');
    a.href = i === 0 ? '#top' : '#showcase';
    a.dataset.name = lb;
    var dot = document.createElement('i');
    var num = document.createElement('span');
    num.textContent = (i < 10 ? '0' + i : String(i));
    a.appendChild(dot);
    a.appendChild(num);
    if (i === 0) a.classList.add('is-active');
    a.addEventListener('click', function (ev) {
      ev.preventDefault();
      if (i === 0) smoothScrollTo(0);
      else goToCh(i, true);
    });
    idxHost.appendChild(a);
  });
}

/* ════════════════════════════════════════════════════════════════
   工作室引擎：INTRO / CREATE / RUN / SUMMARY / REPLAY 五态。
   Job Timeline = 动画控制器：真实状态低频轮询，视觉状态 60fps 插值。
   全部同源相对路径（dev_server.py 反代 /api、/healthz、/artifacts）。
   API 基址可覆盖：meta[imc-api-base] 或 ?api=，仅接受 http/https，默认同源。
   ════════════════════════════════════════════════════════════════ */
var API_ORIGIN = (function () {
  var meta = document.querySelector('meta[name="imc-api-base"]');
  var q = new URLSearchParams(location.search).get('api');
  var base = String(q || (meta && meta.content) || '').trim().replace(/\/+$/, '');
  if (!base || !/^https?:\/\//i.test(base)) return location.origin;
  try { return new URL(base, location.href).origin; } catch (e) { return location.origin; }
})();

var INSTRUCTION_TEXT = '把这个人物放到夕阳湖边，光线要暖。';

/* 最终 URL 构建：仅接受以 / 开头的路径，并在白名单 API 源上重建。
   不接受任何绝对地址（含 "//" 协议相对写法）。 */
function apiUrl(path) {
  var p = String(path || '');
  if (p.charAt(0) !== '/') throw new Error('api path must start with /: ' + p);
  var u = new URL(p, API_ORIGIN);
  if (u.origin !== API_ORIGIN) throw new Error('blocked fetch target: ' + p);
  return u.href;
}

/* 全站唯一 fetch 入口：
   1) 路径模板必须是 / 开头的字面量，动态段用 :name 占位符；
   2) 占位符只能由 opts.params 提供，且经 safePathSeg 白名单校验后替换；
   3) 最终 URL 在白名单 API 源上重建，不接受任何绝对地址。
   调用点因此不存在"变量拼 URL"，SSRF 类风险在此单点收口。 */
function callApi(pathTmpl, opts, ms) {
  opts = opts || {};
  var params = opts.params || {};
  var p = String(pathTmpl || '').replace(/:([A-Za-z_][A-Za-z0-9_]*)/g, function (_, k) {
    return safePathSeg(params[k]);
  });
  var target = apiUrl(p);
  return Promise.race([
    fetch(target, opts),
    new Promise(function (_, reject) {
      setTimeout(function () { reject(new Error('timeout')); }, ms);
    })
  ]);
}

/* 文本入 HTML 前统一转义（节点 label/role 等来自服务端响应） */
function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
  });
}

/* URL 路径段校验：服务端返回的 runId/sessionId 只允许白名单字符，禁止穿越与注入 */
function safePathSeg(s) {
  var v = String(s == null ? '' : s);
  if (!/^[A-Za-z0-9._:-]{1,80}$/.test(v) || v.indexOf('..') >= 0) {
    throw new Error('blocked path segment: ' + v);
  }
  return v;
}

/* 仅接受同源相对路径或 http/https 绝对地址作为图片源 */
function safeImageUrl(u) {
  if (typeof u !== 'string' || !u) return '';
  if (u.charAt(0) === '/') return u;
  try {
    var p = new URL(u, location.href);
    return (p.protocol === 'http:' || p.protocol === 'https:') ? p.href : '';
  } catch (e) { return ''; }
}

async function backendAlive() {
  try {
    var r = await callApi('/healthz', {}, 1500);
    return r.ok;
  } catch (e) { return false; }
}

var upThumb = $('#upThumb');
var upName = $('#upName');

/* —— 工作室元素 —— */
var btnStart = $('#btnStart');
var studioCreate = $('#studioCreate'), studioRun = $('#studioRun'), studioSummary = $('#studioSummary');
var studioDrop = $('#studioDrop'), studioThumb = $('#studioThumb'), studioName = $('#studioName');
var instructionInput = $('#instructionInput');
var intentScene = $('#intentScene'), intentLight = $('#intentLight');
var btnLaunch = $('#btnLaunch'), agentMode = $('#agentMode'), studioNote = $('#studioNote');
var btnCloseCreate = $('#btnCloseCreate'), btnCloseRun = $('#btnCloseRun');
var runModeEl = $('#runMode'), runInstruction = $('#runInstruction');
var rfA = $('#rfA'), rfB = $('#rfB'), rfSubjectWrap = $('#rfSubjectWrap'), rfSubject = $('#rfSubject');
var rfChips = $('#rfChips'), rfDeg = $('#rfDeg'), rfTemp = $('#rfTemp');
var agentFloat = $('#agentFloat'), agentCaption = $('#agentCaption');
var runFillEl = $('#runFill'), runListEl = $('#runList'), runElapsed = $('#runElapsed');
var criticLive = $('#criticLive'), clRows = $('#clRows'), clPill = $('#clPill'), clThreshold = $('#clThreshold');
var sumFinal = $('#sumFinal'), sumTime = $('#sumTime'), sumTools = $('#sumTools'), sumFix = $('#sumFix');
var btnReplayJob = $('#btnReplayJob'), btnAgain = $('#btnAgain');
var replayBar = $('#replayBar'), rpPlay = $('#rpPlay'), rpTrack = $('#rpTrack');
var rpFillEl = $('#rpFill'), rpDot = $('#rpDot'), rpTicks = $('#rpTicks'), rpLabel = $('#rpLabel'), rpExit = $('#rpExit');

/* —— 全局叙事状态 —— */
var AppState = 'intro';          /* intro | create | run | summary | replay */
var STORY = null;                /* null = 演示故事；job 模式见 buildJobStory */
var savedScroll = 0;
var Job = null;                  /* 当前 job（真实或模拟），结构见 startRealJob/startDemoJob */
var LIGHT_TARGETS = { deg: 32, temp: 4800 };
var FINAL_TARGET = 95;

function setState(s) {
  AppState = s;
  body.dataset.state = s;
  var locked = (s === 'create' || s === 'run' || s === 'summary');
  document.documentElement.classList.toggle('lock', locked);  /* 窗口滚动锁 */
  studioCreate.classList.toggle('is-open', s === 'create');
  studioRun.classList.toggle('is-open', s === 'run');
  studioSummary.classList.toggle('is-open', s === 'summary');
  replayBar.hidden = s !== 'replay';
  if (locked && !document.body.classList.contains('studio-saved')) {
    savedScroll = window.scrollY;
    document.body.classList.add('studio-saved');
  }
  if (s === 'replay') {
    document.body.classList.remove('studio-saved');
    buildReplayTicks();
    window.scrollTo(0, M.scTop);
    curP = 0;
  }
  if (s === 'intro') {
    document.body.classList.remove('studio-saved');
  }
}

function restoreScroll() {
  if (document.body.classList.contains('studio-saved')) {
    window.scrollTo(0, savedScroll);
    document.body.classList.remove('studio-saved');
  }
}

function openCreate() {
  if (AppState === 'intro' || AppState === 'replay') savedScroll = window.scrollY;
  setState('create');
  backendAlive().then(function (alive) {
    agentMode.textContent = alive ? 'AGENT · 在线 LIVE' : 'AGENT · 离线 · 演示模式';
    agentMode.classList.toggle('live', alive);
  });
}

function closeStudio() {
  setState('intro');
  restoreScroll();
}

/* —— 指令 → 意图卡（关键词启发，不做假装 AI 的对话） —— */
var SCENE_WORDS = ['夕阳', '日落', '湖', '海', '雪山', '森林', '城市', '街道', '山', '星空', '室内', '工作室', '草地', '沙漠'];
var LIGHT_WORDS = ['暖', '冷', '黄昏', '夕阳', '逆光', '柔光', '侧光', '明亮', '夜晚', '晨光'];
function updateIntent() {
  var t = instructionInput.value || '';
  var scene = SCENE_WORDS.filter(function (w) { return t.indexOf(w) >= 0; });
  var light = LIGHT_WORDS.filter(function (w) { return t.indexOf(w) >= 0; });
  intentScene.textContent = scene.length ? scene.slice(0, 3).join(' · ') : '按你的描述';
  intentLight.textContent = light.length ? light.slice(0, 3).join(' · ') : '按你的描述';
}

/* —— Job Timeline（真实 / 模拟共用协议） ——
   Job = { mode:'real'|'demo', instruction, filename, uploadUrl,
           truth:{ nodes:[{id,role,label,status,version,outputs,artifact_urls}],
                   critic_history:[], status, created_at, finished_at },
           runId, sessionId, timer, startedAt, frames:{...}, events:[] } */

function nodeMainArt(n, prefix) {
  var urls = n.artifact_urls || [];
  for (var i = 0; i < urls.length; i++) {
    var f = String(urls[i]).split('/').pop();
    if (f.indexOf(prefix) === 0) return safeImageUrl(urls[i]);
  }
  return (urls[0] && safeImageUrl(urls[0])) || '';
}

function jobProgress(truth) {
  var nodes = truth.nodes || [];
  if (!nodes.length) return 0.04;
  var done = nodes.filter(function (n) { return n.status === 'done' || n.status === 'skipped'; }).length;
  return 0.04 + 0.92 * (done / nodes.length);
}

/* RUN 舞台帧：由当前已完成节点推导（单一数据链 → 舞台视觉）。
   relightFull = bg+前景 的画布合成（保证像素对齐），见 composeImages。 */
function currentFrame(truth) {
  var byRole = {};
  (truth.nodes || []).forEach(function (n) { byRole[n.role] = n; });
  var F = Job.frames || {};
  var stage = { base: F.photo, subject: null, chips: false };
  var m = byRole.matting, bg = byRole.background_generate;
  var lit = byRole.lighting_estimate, rel = byRole.relight;
  var sh = byRole.shadow_generate, har = byRole.harmonize;
  var en = byRole.enhance, ex = byRole.export;
  if (m && (m.status === 'done')) stage = { base: null, subject: F.cutout, chips: false };
  if (bg && bg.status === 'done') stage = { base: F.env, subject: null, chips: false };
  if (lit && lit.status === 'done') stage = { base: F.env, subject: null, chips: true };
  if (rel && rel.status === 'done') stage = { base: F.relitFull, subject: null, chips: true };
  if (sh && sh.status === 'done') stage = { base: F.shadowed, subject: null, chips: true };
  if (har && har.status === 'done') stage = { base: F.harmon, subject: null, chips: true };
  if (en && en.status === 'done') stage = { base: F.enh, subject: null, chips: true };
  if (ex && ex.status === 'done') stage = { base: F.final, subject: null, chips: true };
  return stage;
}

/* 画布合成：bg 与 fg（RGBA）按同尺寸铺放，输出 dataURL 并缓存。
   跨域 artifact（?api= 指向独立后端时）需 anonymous + CORS，否则 toDataURL 污染抛错 */
var composeCache = {};
function loadImg(src) {
  return new Promise(function (res, rej) {
    var i = new Image();
    if (/^https?:\/\//i.test(src) && src.indexOf(location.origin + '/') !== 0) {
      i.crossOrigin = 'anonymous';
    }
    i.onload = function () { res(i); };
    i.onerror = function () { rej(new Error('img ' + src)); };
    i.src = src;
  });
}
function composeImages(bgUrl, fgUrl, key) {
  if (composeCache[key]) return Promise.resolve(composeCache[key]);
  if (!bgUrl || !fgUrl) return Promise.reject(new Error('compose need 2 urls'));
  return Promise.all([loadImg(bgUrl), loadImg(fgUrl)]).then(function (imgs) {
    var w = imgs[0].naturalWidth || 1024, h = imgs[0].naturalHeight || 1536;
    var cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    var ctx = cv.getContext('2d');
    ctx.drawImage(imgs[0], 0, 0, w, h);
    ctx.drawImage(imgs[1], 0, 0, w, h);
    var url = cv.toDataURL('image/jpeg', 0.92);
    composeCache[key] = url;
    return url;
  });
}

var rfFrontIsA = false;
function showFrame(f) {
  if (!f) return;
  var url = f.base || '';
  var front = rfFrontIsA ? rfA : rfB;
  var back = rfFrontIsA ? rfB : rfA;
  var curSrc = front.getAttribute('src') || '';
  if (url && url !== curSrc) {
    back.src = url;
    back.classList.add('is-front');
    front.classList.remove('is-front');
    rfFrontIsA = !rfFrontIsA;
  } else if (!url && front.classList.contains('is-front')) {
    front.classList.remove('is-front');
  }
  if (f.subject) {
    if (rfSubject.getAttribute('src') !== f.subject) rfSubject.src = f.subject;
    rfSubjectWrap.classList.add('is-front');
  } else {
    rfSubjectWrap.classList.remove('is-front');
  }
  rfChips.hidden = !f.chips;
}

/* Agent 浮动定位 + 台词 */
var AGENT_POS = {
  _default: { left: '34px', bottom: '30px', top: 'auto', transform: 'none' },
  matting: { left: '52%', bottom: '62%', top: 'auto', transform: 'none' },
  lighting_estimate: { left: '20%', bottom: '56%', top: 'auto', transform: 'none' },
  relight: { left: '20%', bottom: '40%', top: 'auto', transform: 'none' },
  shadow_generate: { left: '44%', bottom: '12%', top: 'auto', transform: 'none' },
  critic: { left: '50%', bottom: 'auto', top: '26px', transform: 'translateX(-50%)' }
};
var ROLE_MSG = {
  matting: { run: '先把人物从原始背景里分离出来。', done: '人物已经分离完成。' },
  background_generate: { run: '正在生成新的环境。', done: '环境已经准备好了。' },
  lighting_estimate: { run: '我在看清环境里的光。', done: '主光方向确定了。' },
  relight: { run: '正在调整人物与环境的光线关系。', done: '光已经落到人物身上。' },
  shadow_generate: { run: '让阴影落地。', done: '它站稳了。' },
  harmonize: { run: '统一整张图的质感。', done: '色温、边缘、噪点拉齐了。' },
  enhance: { run: '最后统一细节。', done: '细节完成。' },
  export: { run: '输出成片。', done: '完成了。' }
};
var agentPosKey = '_default';
function agentMove(role, caption) {
  var pos = AGENT_POS[role] || AGENT_POS._default;
  if (agentPosKey !== role) {
    agentPosKey = role;
    agentFloat.style.left = pos.left;
    agentFloat.style.bottom = pos.bottom;
    agentFloat.style.top = pos.top;
    agentFloat.style.transform = pos.transform;
    agentFloat.classList.toggle('pos-center', role === 'critic');
  }
  if (caption) agentCaption.textContent = caption;
}

/* RUN 清单 */
var runRowById = {};
function renderRunList(truth) {
  var nodes = truth.nodes || [];
  if (!nodes.length) return;
  if (!runRowById._built || runRowById._count !== nodes.length) {
    runListEl.innerHTML = '';
    runRowById = { _built: true, _count: nodes.length };
    nodes.forEach(function (n, i) {
      var li = document.createElement('li');
      var idx = document.createElement('span');
      idx.className = 'rl-idx';
      idx.textContent = 'T' + String(i + 1).padStart(2, '0');
      var nm = document.createElement('span');
      nm.className = 'rl-name';
      nm.textContent = n.label || n.role;
      var st = document.createElement('span');
      st.className = 'rl-status';
      st.textContent = '○';
      li.appendChild(idx); li.appendChild(nm); li.appendChild(st);
      runListEl.appendChild(li);
      runRowById[n.id] = li;
    });
  }
  nodes.forEach(function (n) {
    var li = runRowById[n.id];
    if (!li) return;
    var st = li.querySelector('.rl-status');
    var rerun = n.version > 1;
    li.classList.toggle('is-done', n.status === 'done' || n.status === 'skipped');
    li.classList.toggle('is-running', n.status === 'running');
    li.classList.toggle('is-rerun', rerun && n.status !== 'done');
    st.textContent = n.status === 'done' ? (rerun ? '✓ v' + n.version : '✓')
      : n.status === 'running' ? '●' : n.status === 'failed' ? '×' : '○';
  });
}

/* Critic 实时面板 */
function renderCriticLive(c) {
  if (!c) { criticLive.classList.remove('is-on'); return; }
  clThreshold.textContent = c.threshold;
  var rows = [['LIGHTING', c.scores.lighting], ['SHADOW', c.scores.shadow],
              ['COLOR', c.scores.color], ['EDGE', c.scores.edge]];
  clRows.textContent = '';
  rows.forEach(function (r) {
    var fail = !c.passed && r[0] === String(c.lowest_dim || '').toUpperCase();
    var row = document.createElement('div');
    row.className = 'cp-row' + (fail ? ' fail' : '');
    var k = document.createElement('span'); k.textContent = r[0];
    var v = document.createElement('b'); v.textContent = Math.round(r[1]);
    var e = document.createElement('em'); e.textContent = fail ? '← 未达标' : '';
    row.appendChild(k); row.appendChild(v); row.appendChild(e);
    clRows.appendChild(row);
  });
  if (c.action === 'rerun') {
    clPill.className = 'cp-pill mono';
    clPill.textContent = (c.lowest_dim || '').toUpperCase() + ' ' + Math.round(c.scores[c.lowest_dim]) +
      ' < ' + c.threshold + '\u00a0\u00a0→\u00a0\u00a0rerun ' + c.rerun_role;
  } else {
    clPill.className = 'cp-pill mono pass';
    clPill.textContent = 'OVERALL ' + Math.round(c.overall) + ' ≥ ' + c.threshold + ' · PASS';
  }
  criticLive.classList.add('is-on');
}

/* 每次 truth 更新后的 RUN 视图同步 */
var lastCriticCount = 0;
function renderRun() {
  if (!Job) return;
  var t = Job.truth;
  renderRunList(t);
  showFrame(currentFrame(t));
  var prog = jobProgress(t);
  Job.vizTarget = (t.status === 'done' || t.status === 'failed') ? 1 : prog;
  /* Agent：当前 running 节点 → 移动 + 台词 */
  var running = (t.nodes || []).filter(function (n) { return n.status === 'running'; })[0];
  if (t.status === 'failed') {
    agentMove('export', '执行中断了——这一步的工具出错了。');
  } else if (t.status === 'done') {
    agentMove('export', '完成了。要我回放整个过程吗？');
  } else if (t.status === 'critiquing' || (t.critic_history || []).length > lastCriticCount) {
    var c = (t.critic_history || [])[t.critic_history.length - 1];
    if (c) {
      lastCriticCount = t.critic_history.length;
      renderCriticLive(c);
      agentMove('critic', c.passed ? '现在可以了。' : (c.comment || '让我检查一下。'));
      if (c.action === 'rerun' && c.rerun_role) Job.rerunRole = c.rerun_role;
    }
  } else if (running) {
    var msg = ROLE_MSG[running.role] || {};
    agentMove(running.role, msg.run || ('正在' + (running.label || running.role) + '…'));
  } else {
    var lastDone = (t.nodes || []).filter(function (n) { return n.status === 'done'; }).pop();
    if (lastDone) {
      var dm = ROLE_MSG[lastDone.role] || {};
      agentMove(lastDone.role, dm.done || '这一步完成了。');
    }
  }
  /* 光照徽章真实数值（模型输出字段缺失时保持占位，不出现 NaN） */
  var lit = (t.nodes || []).filter(function (n) { return n.role === 'lighting_estimate'; })[0];
  if (lit && lit.outputs && lit.outputs.light_dir && typeof lit.outputs.light_dir.azimuth === 'number') {
    rfDeg.textContent = Math.round(lit.outputs.light_dir.azimuth);
    if (typeof lit.outputs.color_temp === 'number') {
      rfTemp.textContent = Math.round(lit.outputs.color_temp);
    }
  }
}

/* 轮询韧性：失败可见提示；持续 POLL_STALL_MS 无一次成功才终止（慢机上
   真实流水线执行期间事件循环可能让单次轮询超时，计数制会误杀健康 job） */
var POLL_STALL_MS = 60000;
var POLL_MAX_MS = 10 * 60 * 1000;

function pollTick() {
  if (!Job || Job.mode !== 'real' || !Job.runId) return;
  var now = Date.now();
  if (now - Job.startedAt > POLL_MAX_MS) {
    clearInterval(Job.timer); Job.timer = null;
    agentCaption.textContent = '执行超时了，先看看已完成的部分。';
    finishJob(null, new Error('timeout'));
    return;
  }
  callApi('/api/v1/runs/:runId', { params: { runId: Job.runId } }, 6000).then(function (r) { return r.json(); })
    .then(function (run) {
      if (!Job) return;
      Job.lastGoodPoll = Date.now();
      Job.truth = {
        nodes: ((run.dag && run.dag.nodes) || []).map(function (n) {
          return { id: n.id, role: n.role, label: n.label, status: n.status,
                   version: n.version, outputs: n.outputs, artifact_urls: n.artifact_urls || [],
                   started_at: n.started_at, finished_at: n.finished_at };
        }),
        critic_history: run.critic_history || [],
        status: run.status,
        created_at: run.created_at, finished_at: run.finished_at
      };
      refreshRealFrames();
      renderRun();
      if (run.status === 'done' || run.status === 'failed' || run.status === 'cancelled') {
        clearInterval(Job.timer); Job.timer = null;
        finishJob(run);
      }
    }).catch(function () {
      /* 网络抖动：下一轮继续；持续 POLL_STALL_MS 无一次成功轮询才如实终止 */
      if (!Job) return;
      var stalled = Date.now() - (Job.lastGoodPoll || Job.startedAt);
      if (stalled > 8000) {
        agentCaption.textContent = '连接不稳定，正在重试…';
      }
      if (stalled >= POLL_STALL_MS) {
        clearInterval(Job.timer); Job.timer = null;
        agentCaption.textContent = '连接中断了。';
        finishJob(null, new Error('poll stalled'));
      }
    });
}

/* 由节点 artifacts 填充 RUN 帧图（bg+relit 需画布合成保证对齐） */
function refreshRealFrames() {
  var byRole = {};
  (Job.truth.nodes || []).forEach(function (n) { byRole[n.role] = n; });
  var F = Job.frames;
  F.photo = F.photo || Job.uploadUrl || '';
  function done(role) { return byRole[role] && byRole[role].status === 'done'; }
  function art(role, prefix) { return done(role) ? nodeMainArt(byRole[role], prefix) : ''; }
  var cut = art('matting', 'rgba');
  if (cut) F.cutout = cut;
  var env = art('background_generate', 'bg');
  if (env) F.env = env;
  var relit = art('relight', 'relit');
  if (env && relit) {
    composeImages(env, relit, 'relit:' + relit).then(function (url) {
      if (Job && Job.frames) {
        Job.frames.relitFull = url;
        if (AppState === 'run') renderRun();
      }
    }).catch(function () {
      if (Job && Job.frames && !Job.frames.relitFull) Job.frames.relitFull = env;
    });
  }
  var sh = art('shadow_generate', 'composited');
  if (sh) F.shadowed = sh;
  var har = art('harmonize', 'harmonized');
  if (har) F.harmon = har;
  var en = art('enhance', 'enhanced');
  if (en) F.enh = en;
  var fin = art('export', 'final');
  if (fin) F.final = fin;
}

/* —— 真实 Job —— */
async function startRealJob(file, text) {
  var s = await (await callApi('/api/v1/sessions', { method: 'POST' }, 4000)).json();
  var sessionId = s.session_id;
  activeSessionId = sessionId;
  var fd = new FormData();
  fd.append('file', file, file.name || 'upload.png');
  var up = await (await callApi('/api/v1/sessions/:sessionId/uploads', {
    params: { sessionId: sessionId }, method: 'POST', body: fd
  }, 30000)).json();
  var r = await (await callApi('/api/v1/sessions/:sessionId/instructions', { params: { sessionId: sessionId },
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text, quality: 'draft' })
  }, 15000)).json();
  Job = {
    mode: 'real', instruction: text, filename: file.name || 'upload.png',
    uploadUrl: safeImageUrl(up.url) || '', sessionId: sessionId, runId: r.run_id,
    truth: { nodes: [], critic_history: [], status: 'pending' },
    frames: {}, timer: null, startedAt: Date.now(), startedPerf: performance.now(),
    viz: 0, vizTarget: 0.04, lastGoodPoll: Date.now()
  };
  rfA.src = Job.uploadUrl || 'media/source_original.jpg';
  rfA.classList.add('is-front');
  rfFrontIsA = true;
  if (Job.timer) clearInterval(Job.timer);
  Job.timer = setInterval(pollTick, 1200);
  lastCriticCount = 0;
  runModeEl.textContent = 'LIVE';
  agentCaption.textContent = '计划已生成，开始执行。';
  pollTick();
}

/* —— 模拟 Job（离线演示）：同一 Timeline 协议，demo 素材 —— */
var DEMO_ROLES = [
  { role: 'matting', label: '前景抠取', ms: 1600 },
  { role: 'background_generate', label: '背景生成', ms: 1900 },
  { role: 'lighting_estimate', label: '环境光分析', ms: 1200 },
  { role: 'relight', label: '前景重打光', ms: 2100 },
  { role: 'shadow_generate', label: '接触阴影', ms: 1500 },
  { role: 'harmonize', label: '和谐化', ms: 1700 },
  { role: 'enhance', label: '细节增强', ms: 1400 },
  { role: 'export', label: '最终输出', ms: 1100 }
];
function startDemoJob(text) {
  var t0 = Date.now();
  var nodes = DEMO_ROLES.map(function (r, i) {
    return { id: 'n' + (i + 1), role: r.role, label: r.label, status: 'pending',
             version: 1, outputs: null, artifact_urls: [], _ms: r.ms };
  });
  var photoUrl = StudioFile ? URL.createObjectURL(StudioFile) : 'media/source_original.jpg';
  Job = {
    mode: 'demo', instruction: text,
    filename: 'portrait_0427.jpg',
    uploadUrl: photoUrl,
    truth: { nodes: nodes, critic_history: [], status: 'running', created_at: t0 },
    frames: {
      photo: photoUrl, cutout: 'media/subject_clean.png',
      env: 'media/scene_lakeside.jpg', relitFull: 'media/stage_relight.jpg',
      shadowed: 'media/stage_ground.jpg', harmon: 'media/stage_final.jpg',
      enh: 'media/stage_final.jpg', final: 'media/stage_final.jpg',
      light: { deg: 32, temp: 4800 }
    },
    timer: null, startedAt: t0, startedPerf: performance.now(), viz: 0, vizTarget: 0.04, _step: 0
  };
  runModeEl.textContent = 'DEMO';
  rfA.src = Job.frames.photo;
  rfA.classList.add('is-front');
  rfFrontIsA = true;
  lastCriticCount = 0;
  demoStep();
}
function demoStep() {
  if (!Job || Job.mode !== 'demo') return;
  var t = Job.truth;
  var pending = t.nodes.filter(function (n) { return n.status === 'pending'; })[0];
  if (pending) {
    pending.status = 'running';
    renderRun();
    Job.timer = setTimeout(function () {
      pending.status = 'done';
      if (pending.role === 'lighting_estimate') {
        pending.outputs = { light_dir: { azimuth: 32, polar: 78 }, color_temp: 4800 };
      }
      renderRun();
      demoStep();
    }, pending._ms);
    return;
  }
  /* 第一轮 critic：失败 → 重跑 shadow/harmonize/export → PASS */
  var rounds = t.critic_history.length;
  if (rounds === 0) {
    t.status = 'critiquing';
    renderRun();
    Job.timer = setTimeout(function () {
      t.critic_history.push({
        scores: { lighting: 82, shadow: 78, color: 84, edge: 86 },
        overall: 82.2, threshold: 85, passed: false, lowest_dim: 'shadow',
        action: 'rerun', rerun_role: 'shadow_generate',
        comment: 'overall 82.2 < 85：shadow 最低，触发 shadow_generate 升档重跑'
      });
      t.nodes.forEach(function (n) {
        if (n.role === 'shadow_generate' || n.role === 'harmonize' || n.role === 'export') {
          n.status = 'pending'; n.version += 1;
        }
      });
      renderRun();
      demoStep();
    }, 1800);
    return;
  }
  if (rounds === 1) {
    t.status = 'critiquing';
    renderRun();
    Job.timer = setTimeout(function () {
      t.critic_history.push({
        scores: { lighting: 93, shadow: 91, color: 92, edge: 94 },
        overall: 92.4, threshold: 85, passed: true, lowest_dim: 'shadow',
        action: 'pass', rerun_role: null, comment: 'overall 92.4 ≥ 85：通过'
      });
      t.status = 'done';
      t.finished_at = Date.now();
      renderRun();
      finishJob(null);
    }, 1600);
    return;
  }
}

function resetRunUI(text, modeLabel) {
  runInstruction.textContent = text;
  runModeEl.textContent = modeLabel || '…';
  rfA.classList.remove('is-front'); rfB.classList.remove('is-front');
  rfFrontIsA = false;
  rfSubjectWrap.classList.remove('is-front');
  rfChips.hidden = true;
  criticLive.classList.remove('is-on');
  clRows.innerHTML = ''; clPill.textContent = '';
  agentPosKey = '_reset';
  agentMove('_default', '正在与 Agent 建立会话…');
  runFillEl.style.width = '0%';
  runListEl.innerHTML = ''; runRowById = { _built: false };
  setState('run');
}

/* —— Job 完成 → SUMMARY（+ 拉取事件流构建回放故事） —— */
function fmtDur(ms) { return (ms / 1000).toFixed(1) + 's'; }

function finishJob(runFromLastPoll, err) {
  if (!Job || Job.summaryDone) return;
  Job.summaryDone = true;
  loadVersions();  /* 收尾刷新版本树（C6） */
  if (Job.timer) { clearInterval(Job.timer); Job.timer = null; }
  var t = Job.truth;
  var failed = !!err || t.status === 'failed' ||
    (runFromLastPoll && (runFromLastPoll.status === 'failed' || runFromLastPoll.status === 'cancelled'));
  if (t.status !== 'done' && t.status !== 'failed') t.status = failed ? 'failed' : 'done';
  if (!t.finished_at) t.finished_at = Date.now();
  renderRun();
  var durMs = (t.finished_at || Date.now()) - (t.created_at || Job.startedAt);
  var fixes = (t.critic_history || []).filter(function (c) { return c.action === 'rerun'; }).length;
  var last = (t.critic_history || []).slice(-1)[0];
  var finSrc = (Job.frames && Job.frames.final) || currentFrame(t).base || '';
  if (finSrc) { sumFinal.src = finSrc; sumFinal.style.display = ''; }
  else { sumFinal.removeAttribute('src'); sumFinal.style.display = 'none'; }
  sumTime.textContent = fmtDur(durMs);
  sumTools.textContent = (t.nodes || []).length + ' 个节点';
  if (failed) {
    sumFix.textContent = '执行中断';
    sumQuote.textContent = err && err.message === 'timeout'
      ? '执行超时了——可以先回看已完成的部分。'
      : '这一轮执行中断了。回看里可以看到中断前已完成的过程。';
  } else {
    sumFix.textContent = fixes ? fixes + ' 次重跑' : '一次通过';
    sumQuote.textContent = fixes ? ('Agent 自己发现了 ' + fixes + ' 处不足并修正。') : '要不要看看我是怎么做的？';
  }
  FINAL_TARGET = (last && !failed) ? Math.round(last.overall) : Math.min(FINAL_TARGET, 95);
  lastSpokenText = '';  // 收尾播报不受字幕节流限制
  speak(failed ? '这一轮执行中断了，可以先回看已完成的部分。'
               : ('生成完成，整体评分 ' + FINAL_TARGET + ' 分。'));
  if (Job.mode === 'real') {
    callApi('/api/v1/runs/:runId/replay', { params: { runId: Job.runId } }, 8000)
      .then(function (r) { return r.json(); })
      .then(function (rp) { Job.events = (rp && rp.events) || []; })
      .catch(function () { Job.events = []; })
      .then(function () {
        STORY = buildJobStory();
        setState('summary');
      });
  } else {
    STORY = null;
    setState('summary');
  }
}

/* —— artifact:// URI → 同源 HTTP 路径 —— */
function artUrl(u) {
  if (typeof u !== 'string') return '';
  if (u.indexOf('artifact://') === 0) return '/artifacts/' + u.slice('artifact://'.length);
  return safeImageUrl(u);
}

/* —— 真实 run → 回放故事 —— */
var CH_ROLE = { 2: 'matting', 3: 'background_generate', 4: 'lighting_estimate', 5: 'relight',
                6: 'shadow_generate', 7: 'harmonize', 8: 'enhance', 9: 'critic', 10: 'export' };
var ROLE_SHORT = { matting: '分离主体', background_generate: '生成环境', lighting_estimate: '估计光向',
                   relight: '重打光', shadow_generate: '接触阴影', harmonize: '和谐化',
                   enhance: '细节增强', export: '输出成片', critic: 'Agent 自检' };

function buildJobStory() {
  var t = Job.truth;
  var byRole = {};
  t.nodes.forEach(function (n) { byRole[n.role] = n; });
  function art(role, prefix) {
    var n = byRole[role];
    return n ? nodeMainArt(n, prefix || '') : '';
  }
  var lit = byRole.lighting_estimate;
  var ld = (lit && lit.outputs && lit.outputs.light_dir) || null;
  var light = (ld && typeof ld.azimuth === 'number')
    ? { deg: Math.round(ld.azimuth),
        temp: Math.round((lit.outputs && typeof lit.outputs.color_temp === 'number') ? lit.outputs.color_temp : 4800) }
    : { deg: 32, temp: 4800 };
  var last = (t.critic_history || []).slice(-1)[0];
  /* 每节点耗时：优先 node started/finished，回退 events 成对 ts */
  var evDur = {};
  (Job.events || []).forEach(function (e) {
    if (e.event !== 'node_update') return;
    var k = e.node_id + ':' + e.version;
    if (e.status === 'running') evDur[k] = e.ts;
    else if (e.status === 'done' && evDur[k]) {
      var nid = e.node_id;
      evDur['sum:' + nid] = (evDur['sum:' + nid] || 0) + (e.ts - evDur[k]);
    }
  });
  var nodeMeta = {};
  t.nodes.forEach(function (n) {
    var dur = (n.started_at && n.finished_at) ? (n.finished_at - n.started_at)
      : (evDur['sum:' + n.id] || 0);
    nodeMeta[n.role] = { id: n.id, role: n.role, label: n.label || ROLE_SHORT[n.role] || n.role,
      tool: n.role, dur: dur, version: n.version || 1,
      out: nodeMainArt(n, ''), outs: n.artifact_urls || [] };
  });
  var upAsset = Job.uploadUrl || 'media/source_original.jpg';
  return {
    mode: 'job', instruction: Job.instruction, filename: Job.filename,
    original: upAsset,
    cutout: art('matting', 'rgba'),
    bg: art('background_generate', 'bg'),
    relit: art('relight', 'relit'),
    composited: art('shadow_generate', 'composited'),
    harmonized: art('harmonize', 'harmonized'),
    enhanced: art('enhance', 'enhanced'),
    final: art('export', 'final'),
    light: light,
    critic: t.critic_history || [],
    score: last ? Math.round(last.overall) : 95,
    threshold: last ? last.threshold : 85,
    nodeMeta: nodeMeta,
    dagNodes: t.nodes,
    duration: (t.finished_at || Date.now()) - (t.created_at || Job.startedAt)
  };
}

/* —— 回放故事应用到展示舞台 —— */
function applyStory(s) {
  var isJob = !!(s && s.mode === 'job');
  document.body.dataset.story = isJob ? 'job' : 'demo';
  if (!isJob) {
    LIGHT_TARGETS = { deg: 32, temp: 4800 };
    FINAL_TARGET = 95;
    return;
  }
  var ph02bg = $('.ph02-bg'), ph02sub = $('.ph02-sub'), ph03sub = $('.ph03-sub');
  if (ph02bg) ph02bg.src = s.original;
  if (ph02sub) ph02sub.src = s.cutout;
  if (ph03sub) ph03sub.src = s.cutout;
  if (ph02Edge) ph02Edge.style.display = 'none';   /* 描边素材是演示专属 */
  var reveal = $('#sceneReveal');
  if (reveal) reveal.style.backgroundImage = 'url("' + s.bg + '")';
  /* 画布合成：真实 bg + RGBA 前景保证像素对齐（ch4/ch5/ch6 底图） */
  composeImages(s.bg, s.cutout, 'cut:' + s.bg).then(function (url) {
    var illuFill = $('#illuFill');
    if (illuFill) illuFill.src = url;
  }).catch(function () {});
  var illuSub = $('#illuSub');
  if (illuSub) illuSub.style.display = 'none';
  composeImages(s.bg, s.relit, 'relit:' + s.relit).then(function (url) {
    var rBg = $('#relightBg'), rFg = $('#relightFg'), gBg = $('#groundBg'), gFg = $('#groundFg');
    if (rBg) rBg.src = url;
    if (rFg) rFg.style.display = 'none';
    if (gBg) gBg.src = url;
    if (gFg) gFg.style.display = 'none';
  }).catch(function () {});
  var gTop = $('#groundTop');
  if (gTop) gTop.src = s.composited;
  var hBase = $('#harmBase'), hTop = $('#harmonTop');
  if (hBase) hBase.src = s.composited;
  if (hTop) hTop.src = s.harmonized;
  var eBase = $('#enhBase'), eTop = $('#enhTop');
  if (eBase) eBase.src = s.harmonized;
  if (eTop) eTop.src = s.enhanced;
  var cBase = $('#criticBase');
  if (cBase) cBase.src = s.enhanced;
  var fFill = $('#finalFill');
  if (fFill) fFill.src = s.final;
  if (upThumb) upThumb.src = s.original;
  if (upName) upName.textContent = s.filename;
  LIGHT_TARGETS = { deg: s.light.deg, temp: s.light.temp };
  FINAL_TARGET = s.score;
  var meta5 = $('.onimg-meta');
  var rel = s.nodeMeta.relight;
  if (meta5 && rel) {
    meta5.textContent = '';
    meta5.appendChild(document.createTextNode('RELIGHT\u00a0 '));
    var mi1 = document.createElement('i'); mi1.textContent = '·';
    meta5.appendChild(mi1);
    meta5.appendChild(document.createTextNode('\u00a0 真实节点 ' + rel.tool + '\u00a0 '));
    var mi2 = document.createElement('i'); mi2.textContent = '·';
    meta5.appendChild(mi2);
    meta5.appendChild(document.createTextNode('\u00a0 耗时 ' + fmtDur(rel.dur)));
    if (rel.version > 1) {
      meta5.appendChild(document.createTextNode('\u00a0 '));
      var mi3 = document.createElement('i'); mi3.textContent = '·';
      meta5.appendChild(mi3);
      meta5.appendChild(document.createTextNode('\u00a0 v' + rel.version + ' 重跑'));
    }
  }
  rebuildPlan(s.dagNodes);
  fillRailNodes(s);
  if (upOk) setOn(upOk, true);
  var rp = $('#btnReplay');
  if (rp) rp.textContent = '开始新创作';
}

/* 动态 PLAN 路径（job 的真实节点序列） */
function rebuildPlan(dagNodes) {
  var svg = $('.plan-svg');
  var steps = $('.plan-steps');
  if (!svg || !steps || !dagNodes || !dagNodes.length) return;
  var N = dagNodes.length;
  var nodesG = $('#planNodes');
  var line = $('#planLine');
  var x0 = 128, x1 = 872, y = 516;
  var SVG_NS = 'http://www.w3.org/2000/svg';
  nodesG.textContent = '';
  steps.textContent = '';
  for (var i = 0; i < N; i++) {
    var x = x0 + (x1 - x0) * (N === 1 ? 0.5 : i / (N - 1));
    var r = i === N - 1 ? 7 : 6;
    var c = document.createElementNS(SVG_NS, 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', r);
    nodesG.appendChild(c);
    var n = dagNodes[i];
    var name = ROLE_SHORT[n.role] || n.role;
    var sp = document.createElement('span');
    sp.style.left = (x / 976 * 100).toFixed(1) + '%';
    sp.textContent = 'T' + String(i + 1).padStart(2, '0') + '\u00a0 ' + name;
    steps.appendChild(sp);
  }
  if (line) { line.setAttribute('x1', x0); line.setAttribute('x2', x1); }
  planLine = line;
  planNodes = $$('#planNodes circle');
  planSteps = $$('.plan-steps span');
}

/* 左栏 MOTION 盒 → NODE 盒（真实节点信息） */
function fillRailNodes(s) {
  var railCh = { 1: null, 2: 'matting', 3: 'background_generate', 4: 'lighting_estimate',
                 5: 'relight', 6: 'shadow_generate', 7: 'harmonize', 8: 'enhance',
                 9: 'critic', 10: 'export' };
  Object.keys(railCh).forEach(function (ch) {
    var rail = rails[ch];
    if (!rail) return;
    var box = rail.querySelector('.motion');
    if (!box) return;
    var role = railCh[ch];
    if (!role) {  /* 01 UNDERSTAND：计划级信息 */
      box.textContent = '';
      var p1 = document.createElement('p'); p1.className = 'mono';
      p1.appendChild(document.createTextNode('RUN\u00a0 '));
      var i1 = document.createElement('i'); i1.textContent = '·';
      p1.appendChild(i1); p1.appendChild(document.createTextNode('\u00a0 ' + s.instruction));
      var p2 = document.createElement('p');
      var s2 = document.createElement('span'); s2.textContent = '计划';
      p2.appendChild(s2); p2.appendChild(document.createTextNode(s.dagNodes.length + ' 个节点'));
      var p3 = document.createElement('p');
      var s3 = document.createElement('span'); s3.textContent = '总耗时';
      var c3 = document.createElement('code'); c3.className = 'mono'; c3.textContent = fmtDur(s.duration);
      p3.appendChild(s3); p3.appendChild(c3);
      var p4 = document.createElement('p');
      var s4 = document.createElement('span'); s4.textContent = '修正';
      p4.appendChild(s4); p4.appendChild(document.createTextNode(
        String((s.critic || []).filter(function (cc) { return cc.action === 'rerun'; }).length || '无')));
      box.appendChild(p1); box.appendChild(p2); box.appendChild(p3); box.appendChild(p4);
      return;
    }
    var m = s.nodeMeta[role];
    if (!m) return;
    var outName = m.outs.length ? String(m.outs[0]).split('/').pop() : '—';
    var comment = '';
    if (role === 'critic') {
      var lastC = (s.critic || []).slice(-1)[0];
      comment = lastC ? lastC.comment : '';
    } else {
      var rc = (s.critic || []).filter(function (c) { return c.rerun_role === role; })[0];
      comment = rc ? rc.comment : '';
    }
    box.textContent = '';
    var q1 = document.createElement('p'); q1.className = 'mono';
    q1.appendChild(document.createTextNode('NODE\u00a0 '));
    var qi = document.createElement('i'); qi.textContent = '·';
    q1.appendChild(qi); q1.appendChild(document.createTextNode('\u00a0 ' + m.tool));
    var q2 = document.createElement('p');
    var qs = document.createElement('span'); qs.textContent = '输出';
    q2.appendChild(qs); q2.appendChild(document.createTextNode(outName));
    var q3 = document.createElement('p');
    var qs3 = document.createElement('span'); qs3.textContent = '耗时';
    var qc = document.createElement('code'); qc.className = 'mono';
    qc.textContent = m.dur ? fmtDur(m.dur) : '—';
    q3.appendChild(qs3); q3.appendChild(qc);
    var q4 = document.createElement('p');
    var qs4 = document.createElement('span'); qs4.textContent = '版本';
    var qc4 = document.createElement('code'); qc4.className = 'mono'; qc4.textContent = 'v' + m.version;
    q4.appendChild(qs4); q4.appendChild(qc4);
    if (m.version > 1) {
      q4.appendChild(document.createTextNode('\u00a0 '));
      var qi4 = document.createElement('i'); qi4.textContent = '·';
      q4.appendChild(qi4); q4.appendChild(document.createTextNode('\u00a0 重跑'));
    }
    box.appendChild(q1); box.appendChild(q2); box.appendChild(q3); box.appendChild(q4);
    if (comment) {
      var q5 = document.createElement('p'); q5.className = 'node-comment'; q5.textContent = comment;
      box.appendChild(q5);
    }
  });
}

/* —— 回放轴 —— */
var CH_NAMES = { 1: 'UNDERSTAND', 2: 'EXTRACT', 3: 'COMPOSE', 4: 'ILLUMINATE', 5: 'RELIGHT',
                 6: 'GROUND', 7: 'HARMONIZE', 8: 'ENHANCE', 9: 'CRITIC', 10: 'FINAL',
                 11: 'CREATE', 12: 'UPLOAD', 13: 'YOUR TURN' };
var autoplay = false;
var rpBuilt = false;
function buildReplayTicks() {
  if (rpBuilt) return;
  rpBuilt = true;
  CHS.forEach(function (c) {
    var mid = (c.start + c.len / 2) / TOTAL * 100;
    var tick = document.createElement('i');
    tick.style.left = mid.toFixed(2) + '%';
    tick.dataset.ch = c.ch;
    rpTicks.appendChild(tick);
  });
  $$('#rpTicks i').forEach(function (tick) {
    tick.addEventListener('click', function (ev) {
      ev.stopPropagation();
      autoplay = false;
      rpPlay.textContent = 'PLAY';
      goToCh(parseInt(tick.dataset.ch, 10), true);
    });
  });
}
function replayScrub(ev) {
  var rect = rpTrack.getBoundingClientRect();
  var frac = clamp((ev.clientX - rect.left) / rect.width, 0, 1);
  autoplay = false;
  rpPlay.textContent = 'PLAY';
  var y = M.scTop + frac * M.scLenPx;
  window.scrollTo(0, y);
  curP = clamp((y - M.scTop) / Math.max(1, M.scLenPx), 0, 1);
}
function enterReplay() {
  if (STORY && STORY.mode === 'job') applyStory(STORY);
  setState('replay');
  /* §9 完成后自动播放：进入回放即开始制作过程放映（可随时暂停/拖动） */
  autoplay = true;
  rpPlay.textContent = 'PAUSE';
}

/* —— 创建流程 —— */
var StudioFile = null;
function setStudioFile(file) {
  if (!file) return;
  if (!/image\/(jpeg|png|webp)/.test(file.type)) return;
  StudioFile = file;
  studioThumb.src = URL.createObjectURL(file);
  studioThumb.hidden = false;
  studioName.textContent = file.name || 'image';
}

function launch() {
  var text = String(instructionInput.value || '').trim() || INSTRUCTION_TEXT;
  if (StudioFile) { launchWith(StudioFile, text); return; }
  /* 未上传时用示例图作为 ORIGINAL */
  fetch('media/source_original.jpg').then(function (r) { return r.blob(); }).then(function (blob) {
    launchWith(new File([blob], 'portrait_0427.jpg', { type: 'image/jpeg' }), text);
  });
}
function launchWith(file, text) {
  btnLaunch.disabled = true;
  resetRunUI(text, '…');
  var aliveP = backendAlive();
  aliveP.then(function (alive) {
    if (alive) {
      return startRealJob(file, text).catch(function (e) {
        studioNote.textContent = 'Agent 暂时不可用，切换到演示模式。';
        startDemoJob(text);
      });
    }
    startDemoJob(text);
  }).finally(function () {
    btnLaunch.disabled = false;
    studioNote.textContent = 'Agent 会先理解意图，生成执行计划，再逐个调用工具。';
  });
}

function bindCreate() {
  /* HERO CTA → CREATE */
  if (btnStart) btnStart.addEventListener('click', openCreate);

  /* 工作室 CREATE 面板 */
  if (studioDrop) {
    studioDrop.addEventListener('click', function () { fileInput.click(); });
    studioDrop.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); fileInput.click(); }
    });
    ['dragenter', 'dragover'].forEach(function (t) {
      studioDrop.addEventListener(t, function (ev) { ev.preventDefault(); studioDrop.classList.add('is-drag'); });
    });
    ['dragleave', 'drop'].forEach(function (t) {
      studioDrop.addEventListener(t, function (ev) { ev.preventDefault(); studioDrop.classList.remove('is-drag'); });
    });
    studioDrop.addEventListener('drop', function (ev) {
      var f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      setStudioFile(f);
    });
  }
  if (fileInput) fileInput.addEventListener('change', function () {
    setStudioFile(fileInput.files && fileInput.files[0]);
  });
  if (instructionInput) instructionInput.addEventListener('input', updateIntent);
  if (btnLaunch) btnLaunch.addEventListener('click', launch);
  if (btnCloseCreate) btnCloseCreate.addEventListener('click', closeStudio);
  if (btnCloseRun) btnCloseRun.addEventListener('click', function () {
    setState('intro'); restoreScroll();      /* 后台运行：job 继续轮询 */
  });
  if (btnReplayJob) btnReplayJob.addEventListener('click', enterReplay);
  if (btnAgain) btnAgain.addEventListener('click', function () {
    if (Job && Job.timer) { clearTimeout(Job.timer); clearInterval(Job.timer); }
    lastCriticCount = 0;
    openCreate();
  });

  /* 展示页尾章（11 CREATE / 13 YOUR TURN）作为工作室入口 */
  var dz = $('#dropZone');
  if (dz) {
    dz.addEventListener('click', function (ev) {
      if (ev.target.closest('#btnDemo')) return;
      openCreate();
    });
    dz.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openCreate(); }
    });
  }
  var demo = $('#btnDemo');
  if (demo) demo.addEventListener('click', function (ev) {
    ev.stopPropagation();
    if (instructionInput) instructionInput.value = INSTRUCTION_TEXT;
    updateIntent();
    openCreate();
  });
  var replay = $('#btnReplay');
  if (replay) replay.addEventListener('click', function () {
    if (STORY && STORY.mode === 'job') setState('summary');
    else goToCh(2, true);                    /* Demo：回到 02 → 10 循环 */
  });

  /* 回放轴 */
  if (rpPlay) rpPlay.addEventListener('click', function () {
    autoplay = !autoplay;
    rpPlay.textContent = autoplay ? 'PAUSE' : 'PLAY';
  });
  if (rpTrack) {
    rpTrack.addEventListener('pointerdown', function (ev) {
      replayScrub(ev);
      var move = function (e) { replayScrub(e); };
      var up = function () {
        document.removeEventListener('pointermove', move);
        document.removeEventListener('pointerup', up);
      };
      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', up);
    });
  }
  if (rpExit) rpExit.addEventListener('click', function () {
    var toJob = STORY && STORY.mode === 'job';
    setState(toJob ? 'summary' : 'intro');
    if (!toJob) restoreScroll();
  });

  /* ESC：面板关闭 / 退出回放 */
  document.addEventListener('keydown', function (ev) {
    if (ev.key !== 'Escape') return;
    if (AppState === 'create' || AppState === 'run' || AppState === 'summary') closeStudio();
    else if (AppState === 'replay') setState(STORY && STORY.mode === 'job' ? 'summary' : 'intro');
  });

  bindLightDrag();
}

/* —— RELIGHT：拖拽光源实时改变受光（screen 混合暖光层 + 轻微整体提亮） —— */
function bindLightDrag() {
  if (!lightDrag) return;
  var ring = lightDrag.querySelector('.lc-ring');
  var line = lightDrag.querySelector('.lc-line');
  var lineEl = line ? line.querySelector('line') : null;
  var dot = lightDrag.querySelector('.lc-dot');
  var ARM = { x: 45.5, y: 62.5 };                /* 受光参考点：人物小臂（% of stage） */
  var SUN = { x: 17.5, y: 24 };
  var dragging = false;

  function layout(sx, sy) {
    SUN.x = sx; SUN.y = sy;
    stage.style.setProperty('--lx', sx + '%');
    stage.style.setProperty('--ly', sy + '%');
    ring.style.left = sx + '%';
    ring.style.top = sy + '%';
    var lx = Math.min(sx, ARM.x), ty = Math.min(sy, ARM.y);
    line.style.left = lx + '%';
    line.style.top = ty + '%';
    line.style.width = Math.abs(ARM.x - sx) + '%';
    line.style.height = Math.abs(ARM.y - sy) + '%';
    if (lineEl) {
      var flipX = ARM.x < sx, flipY = ARM.y < sy;
      lineEl.setAttribute('x1', flipX ? '100' : '0');
      lineEl.setAttribute('y1', flipY ? '100' : '0');
      lineEl.setAttribute('x2', flipX ? '0' : '100');
      lineEl.setAttribute('y2', flipY ? '0' : '100');
    }
  }

  lightDrag.addEventListener('pointerdown', function (ev) {
    if (!layers[5] || !layers[5].classList.contains('is-on')) return;
    var r = stage.getBoundingClientRect();
    var dx = ev.clientX - (r.left + SUN.x / 100 * r.width);
    var dy = ev.clientY - (r.top + SUN.y / 100 * r.height);
    if (Math.hypot(dx, dy) > Math.max(52, r.width * 0.06)) return;  /* 只认光源附近 */
    dragging = true;
    lightDrag.classList.add('is-dragging');
    if (dragHint) dragHint.style.display = 'none';
    if (relightGlow) setOn(relightGlow, true);
    if (lightDrag.setPointerCapture && ev.pointerId !== undefined) {
      try { lightDrag.setPointerCapture(ev.pointerId); } catch (e) {}
    }
    ev.preventDefault();
  });
  lightDrag.addEventListener('pointermove', function (ev) {
    if (!dragging) return;
    var r = stage.getBoundingClientRect();
    layout(clamp((ev.clientX - r.left) / r.width * 100, 2, 98),
           clamp((ev.clientY - r.top) / r.height * 100, 2, 98));
    stage.style.filter = 'brightness(1.04) saturate(1.05)';
  });
  function endDrag() {
    if (!dragging) return;
    dragging = false;
    lightDrag.classList.remove('is-dragging');
    stage.style.filter = '';
  }
  lightDrag.addEventListener('pointerup', endDrag);
  lightDrag.addEventListener('pointercancel', endDrag);
}

/* ═══════════════════ 7.5 · 语音链路（C3）：麦克风 ASR + 播报 TTS ═══════════════════
   录音：getUserMedia + ScriptProcessor -> 16kHz 单声道 PCM WAV（浏览器原生，零依赖）
   转写：POST /speech/api/v1/speech/transcribe（dev_server 同源反代 -> speech 服务 :8200）
   识别文本回填指令框、可修改后再开始（规范 5.5"我理解的是XXX，确认吗"二次确认降级）；
   播报：浏览器 speechSynthesis（Windows 本地语音引擎），跟读 Agent 字幕 + 完成评分。
   语音任何一环不可用：按钮置灰 + 文案提示，文本输入始终兜底（不白屏）。
   ═══════════════════ */
var SPEECH_BASE = '/speech';
var voiceOk = false;
var ttsOn = true;
var recState = 'idle';
var mediaStream = null, audioCtx = null, processor = null, sourceNode = null;
var recChunks = [], recRate = 0, recTimer = null;
var lastSpokenAt = 0, lastSpokenText = '';

var btnMic = $('#btnMic');
var voiceStatus = $('#voiceStatus');
var btnTts = $('#btnTts');

function probeSpeech() {
  voiceOk = false;
  if (!btnMic) return;
  fetch(SPEECH_BASE + '/healthz')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (h) {
      if (h && h.status === 'ok' && h.asr_provider && h.asr_provider !== 'off') {
        voiceOk = true;
        voiceStatus.textContent = '语音输入 · 已就绪（' + (h.asr_provider === 'qwen' ? '千问' : '本地') + '）';
      } else {
        voiceStatus.textContent = '语音输入 · 未启动（python speech/run.py，文本输入不受影响）';
      }
    })
    .catch(function () {
      voiceStatus.textContent = '语音输入 · 未启动（python speech/run.py，文本输入不受影响）';
    });
}

function encodeWav(samples, rate) {
  var buf = new ArrayBuffer(44 + samples.length * 2);
  var v = new DataView(buf);
  function ws(o, s) { for (var i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); }
  ws(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); ws(8, 'WAVE');
  ws(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (var i = 0; i < samples.length; i++) {
    var s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function stopRecording() {
  recState = 'busy';
  voiceStatus.textContent = '识别中…';
  if (recTimer) { clearTimeout(recTimer); recTimer = null; }
  try { if (processor) processor.disconnect(); } catch (e) {}
  try { if (sourceNode) sourceNode.disconnect(); } catch (e) {}
  if (mediaStream) { mediaStream.getTracks().forEach(function (t) { t.stop(); }); mediaStream = null; }
  try { if (audioCtx) audioCtx.close(); } catch (e) {}
  audioCtx = null; processor = null; sourceNode = null;
  var flat = [];
  for (var i = 0; i < recChunks.length; i++) flat.push.apply(flat, recChunks[i]);
  recChunks = [];
  btnMic.classList.remove('is-rec');
  if (flat.length < 3200) {  // < 0.2s 视为误触
    recState = 'idle';
    voiceStatus.textContent = '没听到声音，再试一次';
    return;
  }
  var target = 16000, ratio = Math.max(1, Math.floor(recRate / target));
  var down = [];
  for (var j = 0; j < flat.length; j += ratio) {
    var acc = 0;
    for (var k = 0; k < ratio && j + k < flat.length; k++) acc += flat[j + k];
    down.push(acc / ratio);
  }
  var wav = encodeWav(down, Math.floor(recRate / ratio));
  fetch(SPEECH_BASE + '/api/v1/speech/transcribe', { method: 'POST', body: wav })
    .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
    .then(function (res) {
      recState = 'idle';
      if (res.ok && res.j && res.j.text) {
        instructionInput.value = String(res.j.text).slice(0, 200);
        voiceStatus.textContent = '已识别，可修改后开始生成';
        updateIntent();
      } else {
        voiceStatus.textContent = '识别失败：' + ((res.j && res.j.error && res.j.error.message) || '请直接输入文字');
      }
    })
    .catch(function () {
      recState = 'idle';
      voiceStatus.textContent = '语音服务不可达，请直接输入文字';
    });
}

function toggleRecording() {
  if (!voiceOk) { probeSpeech(); return; }
  if (recState === 'rec') { stopRecording(); return; }
  if (recState !== 'idle') return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    voiceStatus.textContent = '浏览器不支持录音，请直接输入文字';
    return;
  }
  navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
  }).then(function (stream) {
    mediaStream = stream;
    var Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
    sourceNode = audioCtx.createMediaStreamSource(stream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    recChunks = [];
    recRate = audioCtx.sampleRate;
    processor.onaudioprocess = function (e) {
      if (recState !== 'rec') return;
      recChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    sourceNode.connect(processor);
    processor.connect(audioCtx.destination);
    recState = 'rec';
    btnMic.classList.add('is-rec');
    voiceStatus.textContent = '录音中… 再点一次结束（最长 12 秒）';
    recTimer = setTimeout(stopRecording, 12000);
  }).catch(function () {
    voiceStatus.textContent = '麦克风不可用（检查权限），请直接输入文字';
  });
}

function speak(text) {
  if (!ttsOn || !('speechSynthesis' in window) || !text) return;
  try {
    speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.lang = 'zh-CN';
    u.rate = 1.05;
    var vs = speechSynthesis.getVoices();
    for (var i = 0; i < vs.length; i++) {
      if ((vs[i].lang || '').indexOf('zh') === 0) { u.voice = vs[i]; break; }
    }
    speechSynthesis.speak(u);
  } catch (e) { /* 播报失败静默：字幕条仍然可见 */ }
}

/* Agent 字幕变化 -> 跟读（节流 2.5s，同一句不重复） */
function watchCaptions() {
  if (!agentCaption || !('MutationObserver' in window)) return;
  new MutationObserver(function () {
    var t = (agentCaption.textContent || '').trim();
    if (!t || t === lastSpokenText) return;
    var now = Date.now();
    if (now - lastSpokenAt < 2500) return;
    lastSpokenAt = now;
    lastSpokenText = t;
    speak(t);
  }).observe(agentCaption, { childList: true, characterData: true, subtree: true });
}

function bindVoice() {
  if (!btnMic) return;
  btnMic.addEventListener('click', toggleRecording);
  if (btnTts) {
    btnTts.addEventListener('click', function () {
      ttsOn = !ttsOn;
      btnTts.textContent = ttsOn ? 'TTS  ON' : 'TTS  OFF';
      btnTts.classList.toggle('is-off', !ttsOn);
      if (!ttsOn && 'speechSynthesis' in window) speechSynthesis.cancel();
    });
  }
  if (!('speechSynthesis' in window) && btnTts) {
    btnTts.textContent = 'TTS  N/A';
    btnTts.disabled = true;
  }
  probeSpeech();
  watchCaptions();
}

/* ═══════════════════ 7.6 · 多轮编辑 + 版本树 / 条件回滚（C4 / C6） ═══════════════════
   SUMMARY 面板里对同一 session 连续下发指令（后端只重跑受影响的下游子链）；
   拉取 /versions 渲染节点版本树；"回滚到此版"走 POST /rollback（preserve 语义由
   自然语言或按钮 preserve 选择决定）。5 轮连续编辑剧本见 tools/multiround_test.js。
   ═══════════════════ */
var activeSessionId = '';   /* 与 startRealJob 的响应局部 sessionId 同源赋值 */
var btnRefine = $('#btnRefine');
var refineInput = $('#refineInput');
var refineStatus = $('#refineStatus');
var versionsList = $('#versionsList');
var versionsNote = $('#versionsNote');

function loadVersions() {
  if (!versionsList) return;
  var sid = activeSessionId;
  if (!sid) return;
  callApi('/api/v1/sessions/:sessionId/versions', { params: { sessionId: sid } }, 8000)
    .then(function (r) { return r.json(); })
    .then(function (v) { renderVersions(v.node_versions || {}); })
    .catch(function () { renderVersions({}); });
}

function renderVersions(nodeVersions) {
  if (!versionsList) return;
  versionsList.innerHTML = '';
  var roles = Object.keys(nodeVersions);
  var total = 0;
  roles.forEach(function (role) {
    var hist = (nodeVersions[role] || []).slice()
      .sort(function (a, b) { return (Number(b && b.version) || 0) - (Number(a && a.version) || 0); });
    total += hist.length;
    var li = document.createElement('li');
    var head = document.createElement('span');
    head.className = 'ver-role';
    head.textContent = role;
    li.appendChild(head);
    hist.forEach(function (h) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'ver-btn' + (h.version === hist[0].version ? ' is-current' : '');
      b.textContent = 'v' + h.version + (h.version === hist[0].version ? ' · 当前' : ' · 回滚到此版');
      if (h.version !== hist[0].version) {
        var sid = activeSessionId, ver = h.version;
        b.addEventListener('click', function () { startRollback(role, ver); });
      }
      li.appendChild(b);
    });
    versionsList.appendChild(li);
  });
  if (versionsNote) {
    versionsNote.textContent = total
      ? '点某个旧版本即回滚：恢复该节点并只重跑受影响的下游（如"换背景保留光"）。'
      : '本轮还没有版本记录。';
  }
}

function startRollback(role, version) {
  var sid = activeSessionId;
  if (!sid) return;
  if (refineStatus) refineStatus.textContent = '回滚 ' + role + ' -> v' + version + '，重跑下游…';
  callApi('/api/v1/sessions/:sessionId/rollback', { params: { sessionId: sid },
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role: role, version: version, preserve: [] })
  }, 15000).then(function (r) { return r.json(); }).then(function (res) {
    if (!res || !res.run_id) throw new Error('no run_id');
    refineStatus.textContent = '已按回滚语义重建计划。';
    attachRun(sid, res.run_id, '回滚 ' + role + ' 到 v' + version + ' 并重跑下游');
  }).catch(function () {
    if (refineStatus) refineStatus.textContent = '回滚失败，请重试。';
  });
}

function startRefine(text) {
  var sid = activeSessionId;
  if (!sid) return;
  if (refineStatus) refineStatus.textContent = '重新规划中…';
  callApi('/api/v1/sessions/:sessionId/instructions', { params: { sessionId: sid },
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text, quality: 'draft' })
  }, 15000).then(function (r) { return r.json(); }).then(function (res) {
    if (!res || !res.run_id) throw new Error('no run_id');
    refineStatus.textContent = '已生成新计划，只重跑受影响的下游。';
    attachRun(sid, res.run_id, text);
  }).catch(function () {
    if (refineStatus) refineStatus.textContent = '指令发送失败，请重试。';
  });
}

/* 已有 session 上重挂一个新 run：复用 RUN 面板与轮询循环 */
function attachRun(sessionId, runId, instruction) {
  var prev = Job;
  var sid = sessionId;
  Job = {
    mode: 'real', instruction: instruction, filename: prev ? prev.filename : 'image',
    uploadUrl: prev ? prev.uploadUrl : '', sessionId: sid, runId: runId,
    truth: { nodes: [], critic_history: [], status: 'pending' },
    frames: prev ? prev.frames : {},
    timer: null, startedAt: Date.now(), startedPerf: performance.now(),
    viz: 0, vizTarget: 0.04, lastGoodPoll: Date.now(),
    round: (prev && prev.round ? prev.round : 1) + 1
  };
  if (Job.uploadUrl) { rfA.src = Job.uploadUrl; rfA.classList.add('is-front'); rfFrontIsA = true; }
  resetRunUI('第 ' + Job.round + ' 轮 · ' + instruction, 'LIVE');
  lastCriticCount = 0;
  agentCaption.textContent = '收到新指令，Agent 只重跑受影响的下游节点。';
  if (Job.timer) clearInterval(Job.timer);
  Job.timer = setInterval(pollTick, 1200);
  pollTick();
}

function bindRefine() {
  if (btnRefine) {
    btnRefine.addEventListener('click', function () {
      var t = String(refineInput && refineInput.value || '').trim();
      if (!t) { if (refineStatus) refineStatus.textContent = '先写下这一轮想改什么。'; return; }
      startRefine(t);
    });
  }
}

/* ═══════════════════ 8 · 启动 ═══════════════════ */

function applyDeepLink() {
  var q = new URLSearchParams(location.search);
  if (q.has('p')) {
    var p = clamp(parseFloat(q.get('p')) || 0, 0, 1);
    curP = p;
    window.scrollTo(0, M.scTop + p * M.scLenPx);
    return true;
  }
  if (q.has('ch')) {
    var ch = parseInt(q.get('ch'), 10);
    var frac = parseFloat(q.get('cp') || '0.45');
    var c = byCh[ch];
    if (c) {
      var y = M.scTop + (c.start + c.len * clamp(frac, 0, 1)) / TOTAL * M.scLenPx;
      window.scrollTo(0, y);
      curP = clamp((y - M.scTop) / M.scLenPx, 0, 1);
      return true;
    }
  }
  if (location.hash === '#top') window.scrollTo(0, 0);
  return false;
}

function init() {
  setSize();
  measure();
  buildIndex();
  bindCreate();
  bindVoice();
  bindRefine();
  var handled = applyDeepLink();
  if (!handled && window.scrollY > 2) {
    /* 刷新落在页中：直接就位，不做全程滑行 */
    curP = clamp((window.scrollY - M.scTop) / Math.max(1, M.scLenPx), 0, 1);
  }
  window.addEventListener('resize', function () { measure(); }, { passive: true });
  requestAnimationFrame(tick);
  if (!handled) window.scrollTo(0, 0);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
})();

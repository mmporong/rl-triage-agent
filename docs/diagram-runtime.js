'use strict';

const frame = document.getElementById('diagram');

function installDiagramRuntime() {
  const win = frame.contentWindow;
  const doc = frame.contentDocument;
  if (!win.Archify || doc.getElementById('rl-flow-overlay')) return;
  const svg = doc.querySelector('.diagram-container svg');
  const paths = [...svg.querySelectorAll('path[data-edge-id]')];
  const skin = doc.createElement('style');
  skin.textContent = `
    html[data-theme="dark"] { --bg:#000; --panel:#0a0a0d; --mask:#101015; --grid:#24242b; --toolbar-bg:#111116; --toolbar-menu-bg:#111116; }
    #rl-flow-overlay { position:fixed; inset:0; pointer-events:none; z-index:4; }
    .rl-flow-token { position:absolute; left:0; top:0; width:7px; height:7px; border-radius:50%; box-shadow:0 0 12px currentColor; pointer-events:none; }
  `;
  doc.head.append(skin);
  const overlay = doc.createElement('div');
  overlay.id = 'rl-flow-overlay';
  overlay.setAttribute('aria-hidden', 'true');
  doc.body.append(overlay);
  const tokens = paths.map((path, index) => {
    const token = doc.createElement('span');
    token.className = 'rl-flow-token';
    token.style.color = win.getComputedStyle(path).stroke;
    token.style.background = 'currentColor';
    overlay.append(token);
    return { path, token, phase: index / paths.length, length: path.getTotalLength() };
  });
  let animationId = null;
  let startedAt = 0;
  let elapsed = 0;

  function draw(now) {
    const matrix = svg.getScreenCTM();
    const bounds = svg.getBoundingClientRect();
    if (matrix) {
      for (const { path, token, phase, length } of tokens) {
        const progress = ((now - startedAt + elapsed) / 3200 + phase) % 1;
        const local = path.getPointAtLength(progress * length);
        const point = new win.DOMPoint(local.x, local.y).matrixTransform(matrix);
        token.hidden = point.x < bounds.left || point.x > bounds.right || point.y < bounds.top || point.y > bounds.bottom;
        token.style.transform = `translate(${point.x - 3.5}px,${point.y - 3.5}px)`;
      }
    }
    animationId = win.requestAnimationFrame(draw);
  }

  function syncMotion() {
    const running = win.Archify.motionGovernor.mode() === 'live' && !doc.hidden && !win.Archify.motionGovernor.owner();
    overlay.hidden = !running;
    doc.documentElement.dataset.continuousFlow = running ? 'running' : 'paused';
    if (running && animationId === null) {
      startedAt = win.performance.now();
      animationId = win.requestAnimationFrame(draw);
    } else if (!running && animationId !== null) {
      elapsed += win.performance.now() - startedAt;
      win.cancelAnimationFrame(animationId);
      animationId = null;
    }
  }

  new win.MutationObserver(syncMotion).observe(doc.documentElement, {
    attributes: true,
    attributeFilter: ['data-motion', 'data-motion-owner'],
  });
  doc.addEventListener('visibilitychange', syncMotion);
  win.addEventListener('pagehide', () => {
    if (animationId !== null) win.cancelAnimationFrame(animationId);
  }, { once: true });
  win.Archify.motionGovernor.resume();
  syncMotion();
}

frame.addEventListener('load', installDiagramRuntime);
if (frame.contentDocument?.readyState === 'complete') installDiagramRuntime();

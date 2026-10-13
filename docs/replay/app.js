/* 발표 재생 화면. data.js(window.REPLAY_DATA)만 읽고 네트워크를 쓰지 않는다. */
(function (root) {
  "use strict";

  var LINE_COLORS = ["#b4451f", "#7c3aed", "#047857"];

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function statusLabel(data, status) {
    return (data.status_labels && data.status_labels[status]) || status;
  }

  function formatNumber(value, digits, format) {
    if (format === "sci") return value.toExponential(1);
    if (digits === null || digits === undefined) return String(value);
    return Number(value).toFixed(digits);
  }

  function collectSources(block) {
    var ids = [];
    function add(id) { if (id && ids.indexOf(id) < 0) ids.push(id); }
    (block.status_sources || []).forEach(add);
    (block.rows || []).forEach(function (row) {
      row.forEach(function (cell) { if (cell && typeof cell === "object") add(cell.source); });
    });
    (block.items || []).forEach(function (item) { add(item.source); });
    add(block.source);
    return ids;
  }

  function refLink(data, ids, sourceId) {
    var index = ids.indexOf(sourceId) + 1;
    var row = data.sources[sourceId] || {};
    var title = (row.path || "경로 미정") + (row.sha256 ? " · sha256 " + row.sha256 : "") + " · " + statusLabel(data, row.status);
    if (!row.path) return '<span class="ref" title="' + esc(title) + '">[' + index + "]</span>";
    return '<a class="ref" href="../../' + esc(row.path) + '" title="' + esc(title) + '">[' + index + "]</a>";
  }

  function renderCell(data, ids, cell) {
    if (cell === null || cell === undefined) return "<td></td>";
    if (typeof cell !== "object") return "<td>" + esc(cell) + "</td>";
    var row = data.sources[cell.source] || { status: "waiting" };
    var body;
    if (cell.value === null || cell.value === undefined) {
      var state = row.status === "ready" ? "waiting" : row.status;
      body = '<span class="pending ' + esc(state) + '">' + esc(statusLabel(data, state)) + "</span>";
    } else if (typeof cell.value === "number") {
      body = esc(formatNumber(cell.value, cell.digits, cell.format));
    } else {
      body = esc(cell.value);
    }
    var numeric = typeof cell.value === "number" ? ' class="num"' : "";
    return "<td" + numeric + ">" + body + refLink(data, ids, cell.source) + "</td>";
  }

  function sourceSummary(data, ids) {
    var counts = {};
    ids.forEach(function (id) { var status = (data.sources[id] || { status: "waiting" }).status; counts[status] = (counts[status] || 0) + 1; });
    return ["ready", "waiting", "mismatch"].filter(function (status) { return counts[status]; }).map(function (status) {
      return '<span class="status ' + status + '">' + esc(statusLabel(data, status)) + " " + counts[status] + "</span>";
    }).join(" ");
  }

  function renderSources(data, ids) {
    if (!ids.length) return "";
    return '<details class="evidence"><summary>근거 파일 ' + ids.length + "개 " + sourceSummary(data, ids) + "</summary>" +
      '<ol class="sources">' + ids.map(function (id, index) {
      var row = data.sources[id] || { status: "waiting" };
      var badge = '<span class="status ' + esc(row.status) + '">' + esc(statusLabel(data, row.status)) + "</span>";
      var path = row.path ? '<a href="../../' + esc(row.path) + '">' + esc(row.path) + "</a>" : "경로 미정";
      var sha = row.sha256 ? ' <span class="sha" title="' + esc(row.sha256) + '">sha256 ' + esc(row.sha256.slice(0, 12)) + "…</span>" : "";
      return "<li>[" + (index + 1) + "] " + badge + " " + path + sha + "</li>";
    }).join("") + "</ol></details>";
  }

  function renderTable(data, block, ids) {
    var head = "<tr>" + block.columns.map(function (column) { return "<th>" + esc(column) + "</th>"; }).join("") + "</tr>";
    var rows = block.rows.length ? block.rows.map(function (row) {
      return "<tr>" + row.map(function (cell) { return renderCell(data, ids, cell); }).join("") + "</tr>";
    }).join("") : '<tr><td colspan="' + block.columns.length + '">' + pendingFor(data, ids) + "</td></tr>";
    return "<table><thead>" + head + "</thead><tbody>" + rows + "</tbody></table>";
  }

  function pendingFor(data, ids) {
    var worst = "waiting";
    ids.forEach(function (id) { if ((data.sources[id] || {}).status === "mismatch") worst = "mismatch"; });
    return '<span class="pending ' + worst + '">' + esc(statusLabel(data, worst)) + "</span>";
  }

  function extent(panel) {
    var values = [];
    if (panel.envelope) values = values.concat(panel.envelope.lo, panel.envelope.hi);
    panel.lines.forEach(function (line) { values = values.concat(line.values); });
    values = values.filter(function (value) { return typeof value === "number"; });
    var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
    if (lo === hi) { lo -= 1; hi += 1; }
    var pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  }

  function renderPanel(panel) {
    if (!panel.envelope) return '<div class="chart"><h3>' + esc(panel.title) + '</h3><div class="placeholder">대기</div></div>';
    var width = 560, height = 250, left = 58, right = 10, top = 10, bottom = 28;
    var count = panel.envelope.lo.length, range = extent(panel);
    function x(i) { return left + (i + panel.x0) / 299 * (width - left - right); }
    function y(v) { return top + (range[1] - v) / (range[1] - range[0]) * (height - top - bottom); }
    var band = panel.envelope.lo.map(function (v, i) { return x(i).toFixed(1) + "," + y(v).toFixed(1); })
      .concat(panel.envelope.hi.slice().reverse().map(function (v, i) {
        return x(count - 1 - i).toFixed(1) + "," + y(v).toFixed(1);
      })).join(" ");
    var lines = panel.lines.map(function (line, index) {
      var points = line.values.map(function (v, i) { return x(i).toFixed(1) + "," + y(v).toFixed(1); }).join(" ");
      return '<polyline fill="none" stroke="' + LINE_COLORS[index % LINE_COLORS.length] + '" stroke-width="2" points="' + points + '"/>';
    }).join("");
    var ticks = [range[0], (range[0] + range[1]) / 2, range[1]].map(function (v) {
      var label = Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
      return '<line x1="' + left + '" x2="' + (width - right) + '" y1="' + y(v).toFixed(1) + '" y2="' + y(v).toFixed(1) +
        '" stroke="#e1e5eb"/><text x="' + (left - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end" font-size="12" fill="#5b6576">' + label + "</text>";
    }).join("");
    var xticks = [0, 100, 200, 299].map(function (t) {
      return '<text x="' + x(t - panel.x0).toFixed(1) + '" y="' + (height - 8) + '" text-anchor="middle" font-size="12" fill="#5b6576">' + t + "</text>";
    }).join("");
    return '<div class="chart"><h3>' + esc(panel.title) + '</h3><svg viewBox="0 0 ' + width + " " + height +
      '" role="img" aria-label="' + esc(panel.title) + '">' + ticks + '<polygon points="' + band + '" fill="#cbd5e1" opacity="0.75"/>' +
      lines + xticks + "</svg></div>";
  }

  function renderChart(data, block) {
    var panels = block.panels.map(renderPanel).join("");
    var first = block.panels.filter(function (panel) { return panel.envelope; })[0];
    var legend = first ? '<div class="legend"><span><span class="swatch" style="background:#cbd5e1"></span>' + esc(first.envelope.label) + "</span>" +
      first.lines.map(function (line, index) {
        return '<span><span class="swatch" style="background:' + LINE_COLORS[index % LINE_COLORS.length] + '"></span>' + esc(line.label) + "</span>";
      }).join("") + "<span>가로축: 학습 iteration</span></div>" : "";
    return '<div class="charts">' + panels + "</div>" + legend;
  }

  function renderVideos(data, block) {
    return '<div class="videos">' + block.items.map(function (item) {
      var row = data.sources[item.source] || { status: "waiting" };
      var media = row.status === "ready"
        ? '<video controls preload="metadata" src="../../' + esc(row.path) + '"></video>'
        : '<div class="placeholder"><span class="status ' + esc(row.status) + '">' + esc(statusLabel(data, row.status)) + "</span></div>";
      return '<div class="video-cell">' + media + "<div>" + esc(item.label) + "</div></div>";
    }).join("") + "</div>";
  }

  function renderSlot(data, block) {
    var row = data.sources[block.source] || { status: "waiting" };
    return '<div class="slot"><span class="status ' + esc(row.status) + '">' + esc(statusLabel(data, row.status)) + "</span><span>" +
      esc(block.label) + "</span></div>";
  }

  function renderBlock(data, block) {
    var ids = collectSources(block);
    var body;
    if (block.type === "table") body = renderTable(data, block, ids);
    else if (block.type === "chart") body = renderChart(data, block);
    else if (block.type === "video") body = renderVideos(data, block);
    else if (block.type === "slot") body = renderSlot(data, block);
    else if (block.type === "commands") body = "<pre>" + block.lines.map(esc).join("\n") + "</pre>";
    else throw new Error("unknown block type: " + block.type);
    var title = block.title ? "<h2>" + esc(block.title) + "</h2>" : "";
    return '<section class="block' + (block.wide ? " wide" : "") + '">' + title + body + renderSources(data, ids) + "</section>";
  }

  function renderScene(data, index) {
    var scene = data.scenes[index];
    var total = data.scenes.length;
    var dots = data.scenes.map(function (item, i) {
      return '<a href="#' + (i + 1) + '" class="' + (i === index ? "on" : "") + '" aria-label="장면 ' + (i + 1) + '"></a>';
    }).join("");
    return '<div class="topline"><span>RL Triage · 발표 재생</span><span class="counter">' + scene.number + " / " + total + "</span></div>" +
      "<header><h1>" + esc(scene.title) + '</h1><p class="lead">' + esc(scene.lead) + "</p></header>" +
      '<div class="blocks' + (scene.blocks.length === 1 ? " single" : "") + '">' + scene.blocks.map(function (block) {
        return renderBlock(data, block);
      }).join("") + "</div>" +
      '<div><ul class="notes">' + (scene.notes || []).map(function (note) { return "<li>" + esc(note) + "</li>"; }).join("") + "</ul>" +
      "<footer><span>" + esc(data.scope) + "</span>" +
      '<nav><button type="button" data-step="-1">이전</button><div class="dots">' + dots +
      '</div><button type="button" data-step="1">다음</button></nav></footer></div>';
  }

  function sceneIndex(data, hash) {
    var number = parseInt(String(hash || "").replace("#", ""), 10);
    if (!(number >= 1 && number <= data.scenes.length)) return 0;
    return number - 1;
  }

  function start(window) {
    var document = window.document, stage = document.getElementById("stage");
    var data = window.REPLAY_DATA;
    if (!data || !data.scenes) {
      stage.innerHTML = '<p class="boot">data.js가 없다. <code>evals/build_replay_page.py</code>를 실행한다.</p>';
      return;
    }
    function fit() {
      var scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
      stage.style.transform = "scale(" + scale + ")";
      stage.style.left = ((window.innerWidth - 1920 * scale) / 2) + "px";
      stage.style.top = ((window.innerHeight - 1080 * scale) / 2) + "px";
    }
    function show() {
      stage.innerHTML = renderScene(data, sceneIndex(data, window.location.hash));
    }
    function go(step) {
      var next = Math.max(0, Math.min(data.scenes.length - 1, sceneIndex(data, window.location.hash) + step));
      window.location.hash = "#" + (next + 1);
    }
    stage.addEventListener("click", function (event) {
      var step = event.target.getAttribute && event.target.getAttribute("data-step");
      if (step) go(parseInt(step, 10));
    });
    window.addEventListener("keydown", function (event) {
      if (["ArrowRight", "PageDown", " "].indexOf(event.key) >= 0) { event.preventDefault(); go(1); }
      if (["ArrowLeft", "PageUp"].indexOf(event.key) >= 0) { event.preventDefault(); go(-1); }
      if (event.key === "Home") window.location.hash = "#1";
      if (event.key === "End") window.location.hash = "#" + data.scenes.length;
    });
    window.addEventListener("hashchange", show);
    window.addEventListener("resize", fit);
    fit();
    show();
  }

  var api = { esc: esc, formatNumber: formatNumber, collectSources: collectSources, renderScene: renderScene, sceneIndex: sceneIndex };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root && root.document) start(root);
})(typeof window !== "undefined" ? window : null);
